# TBOT -- the Ctrl+C test (B14 Part 3A, Desire 20 Oct 2026). Sends one Ctrl+C (or Ctrl+Break) to the bot's own console,
# the way Windows does when a Ctrl+C reaches it, then reports what the bot, its launcher (run_bot.ps1) and its dashboard
# did. On the old code it copies the 19 Oct stops (the bot starts its shutdown and is killed half-way); on the new code
# the bot logs "[B1A] a Ctrl+C reached the bot ... IGNORED" and keeps running.
# A small python helper (in %TEMP%) leaves its own console, joins the bot's, sends the signal, and leaves again. This
# PowerShell window is not touched.
# Run (PowerShell as Administrator):
#   powershell -ExecutionPolicy Bypass -File C:\TradingBot\TBOT\tools\b14_ctrl_test.ps1 -Event CtrlC      (or -Event CtrlBreak)
#Requires -RunAsAdministrator
param([ValidateSet("CtrlC", "CtrlBreak")][string]$Event = "CtrlC")
$ErrorActionPreference = "Continue"
Set-Location C:\TradingBot\TBOT
function BotProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*" }) }
function DashProcs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*src\dashboard\server.py*" }) }
$ev = 0; if ($Event -eq "CtrlBreak") { $ev = 1 }
$bp = 0; if (Test-Path logs\bot.pid) { [int]::TryParse((Get-Content logs\bot.pid -Raw).Trim(), [ref]$bp) | Out-Null }
$b0 = BotProcs; $d0 = DashProcs
"1. before: bot processes " + $b0.Count + " | dashboard processes " + $d0.Count + " | task " + (Get-ScheduledTask -TaskName "TBOT").State + " | logs\bot.pid " + $bp
if (-not ($b0 | Where-Object { $_.ProcessId -eq $bp })) { "STOP: logs\bot.pid does not name a running bot -- nothing was sent. Send this output to Desire."; exit }
$py = Join-Path $env:TEMP "tbot_ctrl_event.py"; $out = Join-Path $env:TEMP "tbot_ctrl_event.json"
Remove-Item $out -Force -ErrorAction SilentlyContinue
@'
import ctypes, json, signal, sys, time
from ctypes import wintypes
ev, pid, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
res = {"event": ev, "pid": pid}
try:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.FreeConsole.restype = wintypes.BOOL
    k.AttachConsole.argtypes = [wintypes.DWORD]
    k.AttachConsole.restype = wintypes.BOOL
    k.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    k.SetConsoleCtrlHandler.restype = wintypes.BOOL
    k.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
    k.SetConsoleCtrlHandler(None, True)
    res["left_own_console"] = bool(k.FreeConsole())
    res["joined_bot_console"] = bool(k.AttachConsole(pid))
    res["join_error"] = ctypes.get_last_error()
    if res["joined_bot_console"]:
        res["sent"] = bool(k.GenerateConsoleCtrlEvent(ev, 0))
        res["send_error"] = ctypes.get_last_error()
        time.sleep(1.5)
        k.FreeConsole()
except Exception as e:
    res["error"] = repr(e)
with open(out, "w") as f:
    f.write(json.dumps(res))
'@ | Set-Content -Path $py -Encoding ascii
$Inv = [System.Globalization.CultureInfo]::InvariantCulture   # times always written as 12:34, whatever the language settings
$t0 = Get-Date
& .\venv\Scripts\python.exe $py $ev $bp $out
"2. sent at " + $t0.ToString("HH:mm:ss", $Inv) + ": " + $(if (Test-Path $out) { Get-Content $out -Raw } else { "the helper wrote no result" })
Start-Sleep -Seconds 10
$b1 = BotProcs; $d1 = DashProcs; $ts = (Get-ScheduledTask -TaskName "TBOT").State
"3. 10 s later: bot processes " + $b1.Count + " | dashboard processes " + $d1.Count + " | task " + $ts
$cut = $t0.AddSeconds(-2).ToString("yyyy-MM-dd HH:mm:ss", $Inv)
$L = @(Get-Content logs\trading_bot.log -Tail 3000 -Encoding UTF8 | Where-Object { $_.Length -ge 19 -and $_.Substring(0, 19) -ge $cut })
"4. the bot's log since then (" + $L.Count + " lines):"
$L | Where-Object { $_ -match '\[B1A\]|SHUTTING DOWN|KeyboardInterrupt|\[SHUTDOWN\]|\[OK\] Trading bot stopped|\[TELEGRAM\] direct send' } | Select-Object -First 8 |
    ForEach-Object { "   " + $_.Substring(0, [Math]::Min(400, $_.Length)) }
$lf = Get-ChildItem logs\bot_2*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
$ex = @(); if ($lf) { $ex = @(Get-Content $lf.FullName -Tail 50 | Where-Object { $_.Length -ge 19 -and $_.Substring(0, 19) -ge $cut -and $_ -match 'exited with code|execution completed' }) }
"5. the launcher ($($lf.Name)) since then: " + $(if ($ex.Count) { $ex -join " / " } else { "no 'exited' line" })
$ign = @($L | Where-Object { $_ -match '\[B1A\] a Ctrl' }).Count
$shut = @($L | Where-Object { $_ -match 'SHUTTING DOWN TRADING BOT' }).Count
$okst = @($L | Where-Object { $_ -match '\[OK\] Trading bot stopped' }).Count
if ($ign -gt 0 -and $b1.Count -ge 1 -and $ts -eq "Running" -and $shut -eq 0) { "RESULT: PASS -- the bot logged the $Event, ignored it and keeps running; its launcher keeps running" }
elseif ($shut -gt 0 -and $okst -eq 0 -and $b1.Count -eq 0 -and -not $ex.Count) { "RESULT: REPRODUCED the 19 Oct stop -- the bot began its shutdown and was killed half-way, and its launcher was stopped too (no 'exited' line)" }
elseif ($shut -gt 0 -and $okst -gt 0) { "RESULT: the bot stopped COMPLETELY (its whole shutdown ran) -- not what 19 Oct showed. Send this output to Desire." }
else { "RESULT: something else -- send this output to Desire." }
"===== END -- send everything above to Desire ====="
