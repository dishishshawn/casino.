# One-shot installer for the Monday 2026-05-18 post-flatten recovery.
#
# Registers a Windows Scheduled Task that fires once on 2026-05-18 at
# 08:40 local (09:40 ET = 10 min after NYSE open + 10 min cushion for the
# 7 SELL MARKET orders queued from the 2026-05-15 auto-kill to fill).
# The task self-deletes 1 day after firing.
#
# Run from the project root in a regular (non-admin) PowerShell:
#   .\scripts\install_monday_recovery_task.ps1
#
# Uninstall (revert to manual recovery):
#   Unregister-ScheduledTask -TaskName "Casino_Monday_Recovery" -Confirm:$false
#
# What this task does on fire (see scripts\monday_recovery.ps1 for the
# full sequence): pre-flight checks (broker.positions==0, open_orders==0,
# trading_disabled==True), sync_book_from_broker, reconcile-clean check,
# kill_switch --reenable, verify trading_disabled==False. Halts with a
# critical Discord alert on ANY step failure.
#
# PC-OFF behavior: if your PC is off at 08:40 local on 2026-05-18,
# StartWhenAvailable will catch it the next time you boot + log in. If
# the PC is off ALL of Mon 5/18, the trigger lapses and you must run
# scripts\monday_recovery.ps1 by hand once the SELL MARKETs have filled.

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

# 2026-05-18 08:40 local. Matches 09:40 ET = 10 min after NYSE open.
$startTime = [datetime]::ParseExact("2026-05-18 08:40:00", "yyyy-MM-dd HH:mm:ss", $null)

if ((Get-Date) -gt $startTime) {
    Write-Host "WARNING: 2026-05-18 08:40 is already in the past on this machine." -ForegroundColor Yellow
    Write-Host "         The task will fire as soon as it's registered if StartWhenAvailable applies." -ForegroundColor Yellow
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$projectRoot\scripts\monday_recovery.ps1`"" `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -Once -At $startTime
$trigger.EndBoundary = $startTime.AddHours(4).ToString("s")

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew `
    -DeleteExpiredTaskAfter (New-TimeSpan -Days 1)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName "Casino_Monday_Recovery" `
    -Description "ONE-SHOT 2026-05-18 08:40 local (09:40 ET). Verifies the auto-kill SELL MARKETs flattened, syncs book to broker, re-enables trading. Critical Discord alert on any failure. Self-deletes 1 day after expiry." `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "`n--- Registered ---" -ForegroundColor Green
Get-ScheduledTask -TaskName "Casino_Monday_Recovery" |
    Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}} |
    Format-Table -AutoSize
