# One-shot installer for the daily Falcon task (Casino_Daily_TSMOM_Falcon).
#
# Run from the project root in a regular (non-admin) PowerShell:
#   .\scripts\install_falcon_task.ps1
#
# Uninstall later with:
#   Unregister-ScheduledTask -TaskName "Casino_Daily_TSMOM_Falcon" -Confirm:$false
#
# Falcon is the aggressive sim sibling of the live DiCaprio bot (run_id=
# "Falcon", SimBroker). This registers ONE recurring task that runs the
# Falcon runner daily. Unlike DiCaprio, a sim run gets NO ingest task (it
# shares DiCaprio's DuckDB), NO reconcile/book-sync (live-broker-only), and
# NO kill-check (the daily kill path flattens the LIVE Alpaca account
# regardless of --run-id - see daily_tsmom_falcon.ps1).
#
# Same caveats as install_scheduled_tasks.ps1:
#   * -LogonType Interactive: only fires while the user is signed in.
#   * 17:45 LOCAL TIME trigger; moves with the OS timezone.
#
# Timing: 17:45 puts Falcon AFTER the live DiCaprio chain (ingest 17:00,
# reconcile 17:15, kill-check 17:30) so the ingest task has finished
# writing the shared DuckDB before this read-only sim step runs.

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$weekdays = @('Monday','Tuesday','Wednesday','Thursday','Friday')

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$projectRoot\scripts\daily_tsmom_falcon.ps1`"" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At 5:45PM
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "Casino_Daily_TSMOM_Falcon" `
    -Description "Falcon: aggressive vanilla-TSMOM sim sibling of DiCaprio (run_id=Falcon). Daily mark-to-market + monthly rebal in SimBroker. Logs to logs\daily_falcon_*.log; Discord alert on failure." `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "`n--- Registered ---" -ForegroundColor Green
Get-ScheduledTask -TaskName "Casino_Daily_TSMOM_Falcon" |
    Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}} |
    Format-Table -AutoSize
