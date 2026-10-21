# TBOT -- start the bot through its task, then switch the outside watchdog back on (B14 Part 3A, Desire 20 Oct 2026).
# It never starts a second bot: if the bot is already running it only ends the watchdog's pause.
# Run (PowerShell as Administrator):  powershell -ExecutionPolicy Bypass -File C:\TradingBot\TBOT\tools\start_bot.ps1
#Requires -RunAsAdministrator
$ErrorActionPreference = "Continue"
Set-Location C:\TradingBot\TBOT
function BotProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*" }) }
function DashProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*src\dashboard\server.py*" }) }
$Inv = [System.Globalization.CultureInfo]::InvariantCulture   # times always written as 12:34, whatever the language settings
$t0 = Get-Date
if ((BotProcs).Count -gt 0) {
    "1. the bot is already running (" + (BotProcs).Count + " processes) -- not starting a second one"
} else {
    if (Test-Path logs\stop.flag) { Remove-Item logs\stop.flag -Force; "   removed an old logs\stop.flag (a start overrides a stop)" }
    if (Test-Path logs\bot.pid) { Remove-Item logs\bot.pid -Force; "   removed an old logs\bot.pid (no bot is running)" }
    DashProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "   ended a dashboard left from an earlier start" }
    if ((Get-ScheduledTask -TaskName "TBOT").State -eq "Disabled") { Enable-ScheduledTask -TaskName "TBOT" | Out-Null; "   the TBOT task was disabled -- enabled it" }
    if ((Get-ScheduledTask -TaskName "TBOT").State -eq "Running") { Stop-ScheduledTask -TaskName "TBOT"; Start-Sleep -Seconds 3; "   the task said Running with no bot -- ended it first" }
    Start-ScheduledTask -TaskName "TBOT"
    "1. the TBOT task started -- waiting for the bot's start-up line (usually 1-3 minutes; up to 8) ..."
    $cut = $t0.AddSeconds(-5).ToString("yyyy-MM-dd HH:mm:ss", $Inv); $seen = $null
    do {
        Start-Sleep -Seconds 10
        $seen = Get-Content logs\trading_bot.log -Tail 3000 -Encoding UTF8 -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -ge 19 -and $_.Substring(0, 19) -ge $cut -and $_.Contains('[B1A] Ctrl+C guard:') } | Select-Object -Last 1
    } while (-not $seen -and ((Get-Date) - $t0).TotalSeconds -lt 480)
    $lw = $null; if (Test-Path logs\trading_bot.log) { $lw = (Get-Item logs\trading_bot.log).LastWriteTime }
    if ($seen) { "2. the bot is up: " + $seen.Substring([Math]::Min(26, $seen.Length)) }
    elseif ($lw -and $lw -gt $t0) { "2. the bot is still starting (its log is being written) but its start-up line is not there after 8 minutes -- run C1 in 5 minutes and send both to Desire" }
    else { "2. NO sign of the bot after 8 minutes (its log was not written since the start) -- send this output to Desire (the watchdog is switched back on below and will act)" }
}
if (Test-Path logs\watchdog_pause.txt) { Remove-Item logs\watchdog_pause.txt -Force; "3. the watchdog's pause ended -- it watches the bot again" } else { "3. the watchdog was not paused" }
$wd = Get-ScheduledTask -TaskName "TBOT watchdog" -ErrorAction SilentlyContinue
if (-not $wd) { "   WARNING: there is no task called 'TBOT watchdog'" } elseif ($wd.State -eq "Disabled") { Enable-ScheduledTask -TaskName "TBOT watchdog" | Out-Null; "   the TBOT watchdog task was disabled -- enabled it" }
"4. now: task " + (Get-ScheduledTask -TaskName "TBOT").State + " | bot processes " + (BotProcs).Count + " | dashboard processes " + (DashProcs).Count + " | watchdog task " + (Get-ScheduledTask -TaskName "TBOT watchdog" -ErrorAction SilentlyContinue).State + "   (expect: Running | 2 | 0 to 2 -- the dashboard comes up a minute or two after the bot -- | Ready)"
$lf = Get-ChildItem logs\bot_2*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($lf) { "5. the launcher's protection line: " + ((Get-Content $lf.FullName -Tail 400 | Where-Object { $_ -match 'Ctrl\+C protection' } | Select-Object -Last 1)) }
"===== END ====="
