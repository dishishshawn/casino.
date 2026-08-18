# Shared helpers for the daily Branch C wrapper scripts.
# Dot-source from each wrapper:  . (Join-Path $PSScriptRoot "_common.ps1")
#
# PS 5.1 note: closures here intentionally avoid referencing other functions
# defined in this file, because closure scope does not capture parent functions
# reliably. Each helper is independent.
#
# Provides:
#   Get-CasinoUtcStamp         — true-UTC timestamp string (yyyy-MM-dd HH:mm:ssZ)
#   Write-CasinoLog            — write a UTC-timestamped line to a log file + stdout
#   Get-DiscordWebhookUrl      — read DISCORD_WEBHOOK_URL from project .env
#   Send-CasinoDiscordAlert    — POST an embed to the webhook (best-effort)

function Get-CasinoUtcStamp {
    # PS 5.1 'u' format ("yyyy-MM-dd HH:mm:ssZ") is *literal* text — does NOT
    # convert to UTC. Calling ToUniversalTime() first makes the 'Z' honest.
    return ((Get-Date).ToUniversalTime().ToString('u'))
}

function Write-CasinoLog {
    # Match the encoding used by the *>> operator (UTF-16 LE in PS 5.1) so
    # the file stays consistent when native command output is appended later.
    # Add-Content's default ASCII vs *>>'s Unicode produced mojibake.
    param(
        [Parameter(Mandatory=$true)][string]$LogFile,
        [Parameter(Mandatory=$true)][string]$Message
    )
    $line = "$((Get-Date).ToUniversalTime().ToString('u')) $Message"
    $line | Out-File -FilePath $LogFile -Append
    Write-Host $line
}

function Get-DiscordWebhookUrl {
    param([Parameter(Mandatory=$true)][string]$EnvPath)
    if (-not (Test-Path $EnvPath)) { return $null }
    $line = Get-Content $EnvPath | Where-Object { $_ -match '^\s*DISCORD_WEBHOOK_URL\s*=' } | Select-Object -First 1
    if (-not $line) { return $null }
    $url = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
    if (-not $url -or $url -match '^your_' -or $url -match '_here$') { return $null }
    return $url
}

function Send-CasinoDiscordAlert {
    # Best-effort. Never throws. Returns $true on 2xx, $false otherwise.
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][string]$Body,
        [Parameter(Mandatory=$true)][string]$EnvPath,
        [int]$Color = 15158332,  # red
        [string]$LogFile = $null
    )
    $url = Get-DiscordWebhookUrl -EnvPath $EnvPath
    if (-not $url) { return $false }

    $titleTrim = if ($Title.Length -gt 256) { $Title.Substring(0,256) } else { $Title }
    $bodyTrim  = if ($Body.Length -gt 3500) { $Body.Substring(0,3500) } else { $Body }
    $payload = @{
        embeds = @(@{
            title       = $titleTrim
            description = "``````" + $bodyTrim + "``````"
            color       = $Color
            footer      = @{ text = "casino · scheduled-task wrapper" }
        })
    } | ConvertTo-Json -Depth 5 -Compress

    try {
        Invoke-RestMethod -Uri $url -Method Post -Body $payload -ContentType 'application/json' -TimeoutSec 10 | Out-Null
        return $true
    } catch {
        if ($LogFile) {
            $msg = "$((Get-Date).ToUniversalTime().ToString('u')) discord_alert_failed: " + $_.Exception.Message
            $msg | Out-File -FilePath $LogFile -Append
        }
        return $false
    }
}
