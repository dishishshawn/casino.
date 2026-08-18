# One-shot installer for the Branch C 30-day clock starter.
#
# Registers a single Windows Scheduled Task that fires once on 2026-05-29
# at 17:10 local (10 min after the daily ingest at 17:00; well within
# the 30 min window before the kill-check at 17:30 picks up the new
# clock state). The task self-deletes after firing.
#
# Run from the project root in a regular (non-admin) PowerShell:
#   .\scripts\install_clock_start_task.ps1
#
# Uninstall later (if you want to revert to manual-only):
#   Unregister-ScheduledTask -TaskName "Casino_Branch_C_Clock_Start" -Confirm:$false
#
# IMPORTANT
# ---------
# This OVERRIDES the runner's "manual-only" design from PRD §6.3 amendment.
# That convention was about ensuring the operator is intentional when
# starting the clock — by scheduling this 22 days in advance, you ARE being
# intentional. The pre-flight is the daily heartbeat: if you see "Branch C
# clock not started" in your evening pings on 5/28, this will fire 5/29.
# If anything looks wrong before 5/29, run:
#     Unregister-ScheduledTask -TaskName "Casino_Branch_C_Clock_Start" -Confirm:$false
#
# PC-OFF behavior: if your PC is off at 17:10 on 2026-05-29, Windows will
# try to fire the task the next time you boot and log in (StartWhenAvailable
# is set). If your PC is off all of 2026-05-29, the trigger lapses and the
# clock will not start until you manually run tsmom_runner on a future
# month-end (~2026-06-30).

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

$startTime = [datetime]::ParseExact("2026-05-29 17:10:00", "yyyy-MM-dd HH:mm:ss", $null)

if ((Get-Date) -gt $startTime) {
    Write-Host "WARNING: 2026-05-29 17:10 is already in the past on this machine." -ForegroundColor Yellow
    Write-Host "         The task will fire as soon as it's registered if StartWhenAvailable applies." -ForegroundColor Yellow
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$projectRoot\scripts\start_branch_c_clock.ps1`"" `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -Once -At $startTime
# DeleteExpiredTaskAfter requires the trigger to have an EndBoundary;
# -Once -At doesn't set one, so we attach it manually. Schema expects
# ISO-8601 local time (no zone designator) and Task Scheduler interprets
# it in the host's timezone.
$trigger.EndBoundary = $startTime.AddHours(2).ToString("s")

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew `
    -DeleteExpiredTaskAfter (New-TimeSpan -Days 1)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName "Casino_Branch_C_Clock_Start" `
    -Description "ONE-SHOT 2026-05-29 17:10 local: invokes casino.execution.tsmom_runner to start the Branch C 30-day clock. Self-deletes 1 day after expiry. Fires Discord info alert on success, critical alert on no-op or failure." `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "`n--- Registered ---" -ForegroundColor Green
Get-ScheduledTask -TaskName "Casino_Branch_C_Clock_Start" |
    Select-Object TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}} |
    Format-Table -AutoSize
