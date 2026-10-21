# TBOT -- stop the bot ON PURPOSE, gently (B14 Part 3A, Desire 20 Oct 2026).
# 1. It pauses the outside watchdog (logs\watchdog_pause.txt), so the watchdog doesn't restart the bot.
# 2. It writes logs\stop.flag. The bot sees it within a second or two (or at the end of a look, up to ~2 minutes), runs its
#    whole stop -- state saved, "TBOT STOPPING" on Telegram -- and exits.
# 3. If the bot is still running after 4 minutes (12 if it started in the last 15 minutes), it is stopped the hard way.
# Start it again with tools\start_bot.ps1 (that also ends the watchdog's pause). If nobody does, the watchdog's pause ends
# by itself after -PauseMinutes (120 by default) and the watchdog then restarts the bot -- it is never left off silently.
# Run (PowerShell as Administrator):  powershell -ExecutionPolicy Bypass -File C:\TradingBot\TBOT\tools\stop_bot.ps1
#Requires -RunAsAdministrator
param([int]$PauseMinutes = 120)
$ErrorActionPreference = "Continue"
Set-Location C:\TradingBot\TBOT
function BotProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*" }) }
function DashProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*src\dashboard\server.py*" }) }
$Inv = [System.Globalization.CultureInfo]::InvariantCulture   # times always written as 12:34, whatever the language settings
$t0 = Get-Date
$until = $t0.AddMinutes($PauseMinutes).ToString("yyyy-MM-dd HH:mm", $Inv)
Set-Content -Path logs\watchdog_pause.txt -Value $until -NoNewline -Encoding ascii
"1. the outside watchdog is paused until $until box time (tools\start_bot.ps1 ends the pause at once)"
if ((BotProcs).Count -eq 0) {
    "2. the bot is not running -- nothing to stop"
} else {
    # a bot that started in the last 15 minutes may still be in its first look (the stop file is read after it): wait longer
    $young = @(BotProcs | Where-Object { $_.CreationDate -and $_.CreationDate -gt (Get-Date).AddMinutes(-15) }).Count
    $limit = 240; $why = ""; if ($young) { $limit = 720; $why = "; it started less than 15 minutes ago, so its first look may still be running" }
    Set-Content -Path logs\stop.flag -Value ("stop_bot.ps1, " + $t0.ToString("yyyy-MM-dd HH:mm", $Inv) + " box time") -Encoding ascii
    "2. logs\stop.flag written -- waiting for the bot to stop by itself (up to " + ($limit / 60) + " minutes" + $why + ") ..."
    do { Start-Sleep -Seconds 5; $n = (BotProcs).Count } while ($n -gt 0 -and ((Get-Date) - $t0).TotalSeconds -lt $limit)
    if ($n -eq 0) {
        "3. the bot stopped by itself after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
    } else {
        "3. still running after " + ($limit / 60) + " minutes -- stopping it the hard way (no TBOT STOPPING message this time)"
        Stop-ScheduledTask -TaskName "TBOT" -ErrorAction SilentlyContinue
        BotProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 3
        Remove-Item logs\stop.flag -Force -ErrorAction SilentlyContinue
    }
}
DashProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }    # a dashboard left behind
Start-Sleep -Seconds 2
"4. now: task " + (Get-ScheduledTask -TaskName "TBOT").State + " | bot processes " + (BotProcs).Count + " | dashboard processes " + (DashProcs).Count + "   (expect: Ready | 0 | 0)"
"5. the bot's last lines (since this stop began):"
$cut = $t0.AddSeconds(-5).ToString("yyyy-MM-dd HH:mm:ss", $Inv)
Get-Content logs\trading_bot.log -Tail 3000 -Encoding UTF8 | Where-Object { $_.Length -ge 19 -and $_.Substring(0, 19) -ge $cut -and
    $_ -match '\[CONTROL\]|SHUTTING DOWN|\[TELEGRAM\] direct send|\[OK\] Trading bot stopped|bot\.pid' } |
    Select-Object -Last 8 | ForEach-Object { "   " + $_.Substring(0, [Math]::Min(170, $_.Length)) }
$lf = Get-ChildItem logs\bot_2*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($lf) { "6. the launcher's last line ($($lf.Name)): " + ((Get-Content $lf.FullName -Tail 4 | Where-Object { $_ -match 'exited with code' }) -join " / ") }
"===== END -- the bot is stopped; start it again with tools\start_bot.ps1 ====="
