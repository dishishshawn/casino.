# One-shot installer for the three daily Branch C tasks.
#
# Run from the project root in a regular (non-admin) PowerShell:
#   .\scripts\install_scheduled_tasks.ps1
#
# Uninstall later with:
#   Unregister-ScheduledTask -TaskName "Casino_Daily_*" -Confirm:$false
#
# IMPORTANT — LogonType caveat:
#   These tasks register with -LogonType Interactive, meaning they only fire
#   when the user is signed in. If you sign out / lock-and-disconnect over a
#   weekend, the missed runs are NOT caught up. -StartWhenAvailable handles
#   sleep/hibernate but NOT logged-out state.
#
#   On an "always-on PC" with the user staying logged in, this is fine.
#   If that assumption changes, switch to -LogonType ServiceAccount or use
#   Task Scheduler GUI > "Run whether user is logged on or not".
#
# Timezone caveat:
#   Triggers fire at 17:00/17:15/17:30 LOCAL TIME. Currently CT (UTC-5 in DST)
#   = NYSE close + 2 hr. If the OS timezone changes (travel) the schedule
#   moves with local time and could fall before NYSE close. Re-evaluate if
#   travelling east of UTC or to UTC itself.

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$weekdays = @('Monday','Tuesday','Wednesday','Thursday','Friday')

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

function Register-CasinoTask {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Description,
        [Parameter(Mandatory=$true)][string]$Wrapper,   # filename in scripts\
        [Parameter(Mandatory=$true)][datetime]$At,
        [int]$TimeLimitMinutes = 15
    )
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$projectRoot\scripts\$Wrapper`"" `
        -WorkingDirectory $projectRoot
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $At
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $TimeLimitMinutes) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask `
        -TaskName $Name -Description $Description `
        -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
}

Register-CasinoTask `
    -Name "Casino_Daily_TSMOM_Ingest" `
    -Description "Branch C: refresh TSMOM 10-ETF OHLCV in DuckDB. Logs to logs\daily_ingest_*.log; Discord alert on failure." `
    -Wrapper "daily_tsmom_ingest.ps1" `
    -At 5:00PM `
    -TimeLimitMinutes 30

Register-CasinoTask `
    -Name "Casino_Daily_TSMOM_BookSync" `
    -Description "Branch C: post-fill broker->book position sync at 09:45 CT (15 min after NYSE open). Closes the 2026-05-15 incident gap. Logs to logs\daily_book_sync_*.log; Discord alert on failure." `
    -Wrapper "daily_tsmom_book_sync.ps1" `
    -At 9:45AM `
    -TimeLimitMinutes 10

Register-CasinoTask `
    -Name "Casino_Daily_TSMOM_Reconcile" `
    -Description "Branch C: reconcile Alpaca paper vs SQLite book, daily P&L, drawdown. Logs to logs\daily_reconcile_*.log; Discord alert on failure." `
    -Wrapper "daily_tsmom_reconcile.ps1" `
    -At 5:15PM `
    -TimeLimitMinutes 10

Register-CasinoTask `
    -Name "Casino_Daily_TSMOM_KillCheck" `
    -Description "Branch C 30-day cap kill-criteria daily monitor (no-op pre-clock; active May 30+). Logs to logs\daily_killcheck_*.log; Discord alert on failure." `
    -Wrapper "daily_tsmom_kill_check.ps1" `
    -At 5:30PM `
    -TimeLimitMinutes 10

Write-Host "`n--- Registered ---" -ForegroundColor Green
Get-ScheduledTask -TaskName "Casino_Daily_*" |
    Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}} |
    Format-Table -AutoSize
