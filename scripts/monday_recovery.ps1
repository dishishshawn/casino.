# Monday 2026-05-18 one-shot recovery wrapper for the 2026-05-15 17:30 ET
# auto-kill-switch incident.
#
# Sequence (every step halts on failure with a critical Discord alert):
#   1. Pre-flight: broker.positions == 0  (Mon-open SELL MARKETs filled)
#   2. Pre-flight: broker.open_orders == 0  (no stragglers, no new bracket children)
#   3. Pre-flight: book.is_trading_disabled() == True  (recovery still needed)
#   4. reconcile.sync_book_from_broker() to canonicalize book = 0 positions
#   5. reconcile.reconcile() to verify drift == 0
#   6. kill_switch --reenable
#   7. Verify book.is_trading_disabled() == False
#   8. Discord info "Monday recovery complete"
#
# On ANY failure the wrapper sends a critical Discord with the log tail
# and exits non-zero. The task self-deletes 1 day after firing regardless.
# If the wrapper fails, manual recovery per RUNBOOK section 5 is required.
#
# This script is ASCII-only by design (PS 5.1 parser breaks on em-dashes
# inside string literals; see feedback_powershell_unicode_strings memory).

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

. (Join-Path $PSScriptRoot "_common.ps1")

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("monday_recovery_" + (Get-Date -Format "yyyyMMdd") + ".log")
$envPath = Join-Path $projectRoot ".env"
$uv = "C:\Users\TNTMi\.local\bin\uv.exe"

Write-CasinoLog -LogFile $logFile -Message "=== monday_recovery START ==="

function Send-FailureAlert {
    param([string]$Title, [string]$Stage)
    $tail = ""
    try { $tail = (Get-Content $logFile -Tail 80 -ErrorAction SilentlyContinue) -join "`n" } catch {}
    $body = "Stage failed: $Stage`n`nLog tail:`n$tail"
    Send-CasinoDiscordAlert `
        -Title $Title `
        -Body $body `
        -EnvPath $envPath `
        -Color 15158332 `
        -LogFile $logFile | Out-Null
    Write-CasinoLog -LogFile $logFile -Message "=== monday_recovery END exit_code=1 stage=$Stage ==="
}

# ---- Step 1-3: pre-flight checks ----
# Single Python call so we get one consistent snapshot of broker + book.
# Exits 0 if all three preflight checks pass, 10 / 11 / 12 / 13 otherwise.
$preflightOutFile = Join-Path $logDir ("monday_recovery_preflight_" + (Get-Date -Format "yyyyMMddHHmmss") + ".out")
& $uv run python -c @"
import json, sys
from casino.execution.alpaca_broker import build_default_broker
from casino.execution import book

b = build_default_broker()
positions = list(b.get_positions())
open_orders = list(b.get_orders(status='open'))
disabled = book.is_trading_disabled()
snapshot = {
    'n_positions': len(positions),
    'n_open_orders': len(open_orders),
    'trading_disabled': bool(disabled),
    'positions_sym_qty': [(p.symbol, int(p.qty)) for p in positions],
    'open_orders_sym_side_qty': [(o.symbol, str(o.side), int(o.qty)) for o in open_orders],
}
with open(r'$preflightOutFile', 'w') as f:
    json.dump(snapshot, f)
if len(positions) != 0:
    sys.exit(10)
if len(open_orders) != 0:
    sys.exit(11)
if not disabled:
    sys.exit(12)
sys.exit(0)
"@ *>> $logFile
$preflightExit = $LASTEXITCODE
Write-CasinoLog -LogFile $logFile -Message "preflight exit=$preflightExit"

$preflightJson = if (Test-Path $preflightOutFile) { Get-Content $preflightOutFile -Raw } else { "(no snapshot file)" }
Write-CasinoLog -LogFile $logFile -Message "preflight snapshot: $preflightJson"

if ($preflightExit -ne 0) {
    $reasons = @{
        10 = "broker still holds positions (Mon-open flatten did not complete)";
        11 = "broker still has open orders (cancel did not complete)";
        12 = "trading_disabled is False (kill switch is not engaged; recovery unnecessary or state diverged)";
    }
    $reason = $reasons[$preflightExit]
    if (-not $reason) { $reason = "unknown preflight failure" }
    Send-FailureAlert -Title "casino: MONDAY RECOVERY ABORTED ($reason)" -Stage "preflight"
    exit 1
}
Write-CasinoLog -LogFile $logFile -Message "preflight OK: 0 positions, 0 open orders, trading_disabled=True"

# ---- Step 4-5: sync book + verify reconcile ----
& $uv run python -c @"
import sys
from casino.execution.alpaca_broker import build_default_broker
from casino.execution.reconcile import sync_book_from_broker, reconcile
from casino.execution import book

b = build_default_broker()
sync_book_from_broker(broker=b)
book_positions = book.fetch_positions()
if len(book_positions) != 0:
    print(f'POST-SYNC BOOK NOT EMPTY: {book_positions}')
    sys.exit(20)
rec = reconcile(broker=b)
drift_count = sum(1 for d in rec.drift if d.kind in ('broker_only','book_only','qty_mismatch','side_mismatch'))
if drift_count != 0:
    print(f'POST-SYNC RECONCILE STILL DRIFTING: {rec.drift}')
    sys.exit(21)
print('sync_and_reconcile OK: 0 book positions, 0 drift entries')
"@ *>> $logFile
$syncExit = $LASTEXITCODE
Write-CasinoLog -LogFile $logFile -Message "sync_and_reconcile exit=$syncExit"

if ($syncExit -ne 0) {
    Send-FailureAlert -Title "casino: MONDAY RECOVERY ABORTED (sync/reconcile failed exit=$syncExit)" -Stage "sync_and_reconcile"
    exit 1
}

# ---- Step 6-7: re-enable trading ----
& $uv run python -m casino.execution.kill_switch --reenable *>> $logFile
$reenableExit = $LASTEXITCODE
Write-CasinoLog -LogFile $logFile -Message "kill_switch_reenable exit=$reenableExit"

if ($reenableExit -ne 0) {
    Send-FailureAlert -Title "casino: MONDAY RECOVERY ABORTED (kill_switch --reenable failed exit=$reenableExit)" -Stage "kill_switch_reenable"
    exit 1
}

& $uv run python -c "from casino.execution import book; import sys; sys.exit(1 if book.is_trading_disabled() else 0)" *>> $logFile
$verifyExit = $LASTEXITCODE
Write-CasinoLog -LogFile $logFile -Message "verify_reenable exit=$verifyExit (0=trading_enabled)"

if ($verifyExit -ne 0) {
    Send-FailureAlert -Title "casino: MONDAY RECOVERY ABORTED (re-enable command returned 0 but trading_disabled still True)" -Stage "verify_reenable"
    exit 1
}

# ---- Step 8: success ----
$body = "Pre-flight clean (0 broker positions, 0 open orders, kill switch was engaged). "
$body += "sync_book_from_broker OK. Reconcile clean (0 drift). "
$body += "kill_switch --reenable OK. trading_disabled now False. "
$body += "System ready for the 2026-05-29 month-end rebal (or an operator --force entry). "
$body += "Task 50 (post-fill book-sync gap) remains open; do not run --force again until it is closed."
Send-CasinoDiscordAlert `
    -Title "casino: Monday recovery COMPLETE" `
    -Body $body `
    -EnvPath $envPath `
    -Color 3066993 `
    -LogFile $logFile | Out-Null

Write-CasinoLog -LogFile $logFile -Message "=== monday_recovery END exit_code=0 (RECOVERED) ==="
exit 0
