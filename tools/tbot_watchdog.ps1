# TBOT outside watchdog v2 -- B13 item 21 (Desire 5 Oct); v2 = B14 Part 3A (Desire 20 Oct: D2 A). Runs every 5 minutes
# from Task Scheduler (as SYSTEM).
# If the bot's log has been quiet for 10 minutes: a Telegram, then the bot's task is restarted and checked 30 seconds later
# (the task must be Running with the bot's python processes up), and that result is sent too.
# If the log is still quiet 20 minutes after a restart, it says so plainly -- "the restart did NOT bring the bot back" --
# and tries again, every 20 minutes, until the log moves; then it says the bot is back, after how many tries.
# A deliberate stop (tools\stop_bot.ps1) writes logs\watchdog_pause.txt with the time the pause ends (yyyy-MM-dd HH:mm,
# box time): until then nothing is restarted; when it ends, the watchdog says so and watches again.
# Every action goes into logs\watchdog.log, one line each (no secrets), Telegram failures included, plus an "ok" line
# once an hour so the file shows the watchdog itself is running.
$ErrorActionPreference = "Continue"
$Root     = "C:\TradingBot\TBOT"
$TaskName = "TBOT"
$Log      = Join-Path $Root "logs\trading_bot.log"
$State    = Join-Path $Root "data\watchdog_state.json"
$WdLog    = Join-Path $Root "logs\watchdog.log"
$Pause    = Join-Path $Root "logs\watchdog_pause.txt"
$PidFile  = Join-Path $Root "logs\bot.pid"
$QuietMin = 10
$GapMin   = 19.5          # a restart that didn't work is tried again every 20 minutes (19.5: room for the 5-minute timer's jitter)
$Inv      = [System.Globalization.CultureInfo]::InvariantCulture   # every time below is written as 12:34, whatever the language settings
$Epoch    = [datetime]::SpecifyKind([datetime]"2000-01-01", [System.DateTimeKind]::Utc)   # state-file times: minutes since 2000 UTC (a plain number:
                                     # read back the same in every PowerShell; UTC, so the clock changes in March and October can't fake a quiet log)

function Write-WdLog([string]$Text) {
    try {
        if ((Test-Path $WdLog) -and ((Get-Item $WdLog).Length -gt 2MB)) { Move-Item $WdLog ($WdLog + ".old") -Force }
        Add-Content -Path $WdLog -Value ((Get-Date).ToString("yyyy-MM-dd HH:mm:ss", $Inv) + "  " + $Text) -Encoding UTF8
    } catch { }
}

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
        Write-WdLog ("telegram sent: " + $Text)
    } catch {
        $m = ([string]$_.Exception.Message) -replace '\d{8,10}:[A-Za-z0-9_-]{35}', '<hidden>' -replace 'bot[^/\s]+/', 'bot<hidden>/'
        Write-WdLog ("telegram FAILED (" + $m + "): " + $Text)
    }
}

function Get-BotProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*" }) }
function Get-DashProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*src\dashboard\server.py*" }) }

function Restart-Bot {
    $notes = New-Object System.Collections.Generic.List[string]
    try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop; $notes.Add("task stopped") } catch { $notes.Add("task stop: " + $_.Exception.Message) }
    $ended = 0
    foreach ($p in @(Get-BotProcs) + @(Get-DashProcs)) { try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; $ended++ } catch { } }
    $notes.Add("python processes ended: $ended (the bot and its dashboard)")
    Start-Sleep -Seconds 5
    $left = @(Get-BotProcs).Count
    if ($left -gt 0) {
        $line = "FAILED -- $left bot process(es) could not be ended, so the bot was not started again (two bots must never run)"
        Write-WdLog ("restart: " + $line + " | " + ($notes -join "; "))
        return $line
    }
    if (Test-Path $PidFile) {
        try { Remove-Item $PidFile -Force -ErrorAction Stop; $notes.Add("old logs\bot.pid removed") } catch { $notes.Add("logs\bot.pid NOT removed: " + $_.Exception.Message) }
    }
    $startErr = $null
    try { Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop } catch { $startErr = $_.Exception.Message }
    Start-Sleep -Seconds 30
    $ts = "?"
    try { $ts = [string](Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop).State } catch { $ts = "unknown (" + $_.Exception.Message + ")" }
    $n = @(Get-BotProcs).Count
    if ($startErr) { $line = "FAILED -- the task did not start: " + $startErr }
    elseif ($ts -eq "Running" -and $n -ge 1) { $line = "started -- 30 s later the task is Running with $n bot process(es); I'll say when its log moves" }
    else { $line = "FAILED -- 30 s after the start the task is $ts with $n bot process(es)" }
    Write-WdLog ("restart: " + $line + " | " + ($notes -join "; "))
    return $line
}

function Get-Min($v) { try { return [double]$v } catch { return 0.0 } }       # an old (v1) or empty value reads as "never"

$st = @{ v = 2; alerted = $false; restart_at = 0.0; tries = 0; pause_noted = ""; missing_at = 0.0 }
$first = $true
if (Test-Path $State) {
    try {
        $j = Get-Content $State -Raw | ConvertFrom-Json
        $st.alerted = [bool]$j.alerted; $st.restart_at = Get-Min $j.restart_at; $st.tries = [int]$j.tries
        $st.pause_noted = [string]$j.pause_noted; $st.missing_at = Get-Min $j.missing_at
        $first = ([int]$j.v -ne 2)
    } catch { Write-WdLog "state file unreadable -- starting fresh" }
}
if ($first) { Write-WdLog "watchdog v2 is running (its first run) -- quiet over $QuietMin min = restart, then every 20 min until the log moves" }
$now = Get-Date
$nowUtc = $now.ToUniversalTime()
$nowMin = ($nowUtc - $Epoch).TotalMinutes

# ---- a deliberate stop pauses the watchdog ----
if (Test-Path $Pause) {
    $until = $null
    try { $until = [datetime]::ParseExact(((Get-Content $Pause -Raw).Trim()), "yyyy-MM-dd HH:mm", [System.Globalization.CultureInfo]::InvariantCulture) } catch { }
    if ($null -eq $until) {
        Remove-Item $Pause -Force -ErrorAction SilentlyContinue
        Write-WdLog "pause file unreadable -- removed, watching again"
        Send-TG "TBOT WATCHDOG: the pause file (logs\watchdog_pause.txt) could not be read -- removed, watching the bot again."
    } elseif ($now -lt $until) {
        $u = "until " + $until.ToString("yyyy-MM-dd HH:mm", $Inv)
        if ($st.pause_noted -ne $u) {
            $st.pause_noted = $u
            Write-WdLog ("paused " + $u + " (a deliberate stop)")
            Send-TG ("TBOT WATCHDOG: paused until " + $until.ToString("HH:mm", $Inv) + " box time (a deliberate stop) -- no restarts until then.")
        } elseif ($now.Minute -lt 5) {
            Write-WdLog ("ok -- paused " + $u + " (a deliberate stop)")       # the hourly line, during a pause too
        }
        $st | ConvertTo-Json | Set-Content -Path $State -Encoding UTF8
        exit
    } else {
        Remove-Item $Pause -Force -ErrorAction SilentlyContinue
        Write-WdLog ("pause ended (" + $until.ToString("yyyy-MM-dd HH:mm", $Inv) + ") -- watching again")
        Send-TG ("TBOT WATCHDOG: the pause ended at " + $until.ToString("HH:mm", $Inv) + " box time -- watching the bot again.")
    }
}
$st.pause_noted = ""

# ---- the bot's log ----
if (-not (Test-Path $Log)) {
    if ($nowMin - $st.missing_at -ge 59.5) {
        Send-TG "TBOT WATCHDOG: the bot's log file is missing ($Log) -- please check the box."
        $st.missing_at = $nowMin
    }
    Write-WdLog "the log file is missing"
    $st | ConvertTo-Json | Set-Content -Path $State -Encoding UTF8
    exit
}
$st.missing_at = 0.0
$ageMin = [math]::Round(($nowUtc - (Get-Item $Log).LastWriteTimeUtc).TotalMinutes, 1)    # in UTC: box time jumps an hour twice a year

if ($ageMin -gt $QuietMin) {
    $sinceRestart = $nowMin - $st.restart_at
    if ($sinceRestart -lt $GapMin) {
        Write-WdLog ("log quiet " + $ageMin + " min -- waiting: restart try " + $st.tries + " was " + [math]::Round($sinceRestart, 1) + " min ago")
    } else {
        $st.tries = [int]$st.tries + 1
        if ($st.tries -le 1) {
            Send-TG "TBOT WATCHDOG: log quiet for $ageMin min -- restarting the bot (try 1)."
        } else {
            Send-TG ("TBOT WATCHDOG: the restart at " + $Epoch.AddMinutes($st.restart_at).ToLocalTime().ToString("HH:mm", $Inv) + " did NOT bring the bot back (log quiet " + $ageMin + " min) -- trying again (try " + $st.tries + "; every 20 minutes until it works).")
        }
        $st.restart_at = $nowMin          # when this try BEGAN, so the next one comes 20 minutes later, not 25
        $res = Restart-Bot
        Send-TG ("TBOT WATCHDOG: restart try " + $st.tries + ": " + $res)
        $st.alerted = $true
    }
} elseif ($st.alerted) {
    $tw = "tries"; if ([int]$st.tries -eq 1) { $tw = "try" }
    Send-TG ("TBOT WATCHDOG: the bot is writing its log again (last write " + $ageMin + " min ago) -- back after " + $st.tries + " restart " + $tw + ".")
    Write-WdLog ("the log moves again after " + $st.tries + " restart " + $tw)
    $st.alerted = $false; $st.tries = 0; $st.restart_at = 0.0
} elseif ($now.Minute -lt 5) {
    Write-WdLog ("ok -- the bot's log was written " + $ageMin + " min ago")       # once an hour: the watchdog itself is alive
}
$st | ConvertTo-Json | Set-Content -Path $State -Encoding UTF8
