# TBOT outside watchdog -- B13 item 21 (Desire 5 Oct). Runs every 5 minutes from Task Scheduler (as SYSTEM).
# If the bot's log has been quiet for 10 minutes: Telegram alert + restart the bot's scheduled task.
# At most one restart per 30 minutes; if it is still quiet after a restart, it alerts only (no restart loop).
# When the log moves again after an alert, it says so once.
$ErrorActionPreference = "Continue"
$Root     = "C:\TradingBot\TBOT"
$TaskName = "PUT-THE-BOT-TASK-NAME-HERE"          # <- Stephen: the bot's Task Scheduler task name (see the plan)
$Log      = Join-Path $Root "logs\trading_bot.log"
$State    = Join-Path $Root "data\watchdog_state.json"
$QuietMin = 10
$GapMin   = 30

function Send-TG([string]$Text) {
    try {
        $envf = Join-Path $Root ".env"
        $tok = ((Select-String -Path $envf -Pattern '^\s*TELEGRAM_BOT_TOKEN\s*=\s*(.+)$').Matches[0].Groups[1].Value).Trim().Trim('"').Trim("'")
        $ids = ((Select-String -Path $envf -Pattern '^\s*TELEGRAM_ADMIN_IDS\s*=\s*(.+)$').Matches[0].Groups[1].Value).Trim().Trim('"').Trim("'")
        foreach ($id in $ids.Split(',')) {
            if ($id.Trim()) {
                Invoke-RestMethod -Uri "https://api.telegram.org/bot$tok/sendMessage" -Method Post -TimeoutSec 15 `
                    -Body @{ chat_id = $id.Trim(); text = $Text } | Out-Null
            }
        }
    } catch { }
}

$st = @{ alerted = $false; restart_at = "" }
if (Test-Path $State) { try { $j = Get-Content $State -Raw | ConvertFrom-Json; $st.alerted = [bool]$j.alerted; $st.restart_at = [string]$j.restart_at } catch { } }

if (-not (Test-Path $Log)) { Send-TG "TBOT WATCHDOG: log file not found ($Log)"; exit }
$ageMin = [math]::Round(((Get-Date) - (Get-Item $Log).LastWriteTime).TotalMinutes, 1)

if ($ageMin -gt $QuietMin) {
    $sinceRestart = 9999
    if ($st.restart_at) { $sinceRestart = ((Get-Date) - [datetime]$st.restart_at).TotalMinutes }
    if ($sinceRestart -lt $GapMin) {
        if (-not $st.alerted) { Send-TG "TBOT WATCHDOG: log still quiet ($ageMin min) after a restart -- NOT restarting again. Please check the box." }
        $st.alerted = $true
    } else {
        Send-TG "TBOT WATCHDOG: log quiet for $ageMin min -- restarting the bot."
        try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*" } |
            ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch { } }
        Start-Sleep -Seconds 5
        try { Start-ScheduledTask -TaskName $TaskName; $ok = $true } catch { $ok = $false }
        if (-not $ok) { Send-TG "TBOT WATCHDOG: could not start the task '$TaskName' -- please check the box." }
        $st.restart_at = (Get-Date).ToString("o")
        $st.alerted = $true
    }
} elseif ($st.alerted) {
    Send-TG "TBOT WATCHDOG: the bot is writing its log again (last write $ageMin min ago)."
    $st.alerted = $false
}
$st | ConvertTo-Json | Set-Content -Path $State -Encoding UTF8
