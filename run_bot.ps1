# Trading Bot Wrapper Script
# This script runs your bot and handles logging
# B14 Part 3A (Desire 20 Oct 2026): this script now ignores Ctrl+C and Ctrl+Break, so it can no longer stop the bot.
# Before, a Ctrl+C on the bot's console made PowerShell stop this script and kill python half-way through the bot's
# shutdown (19 Oct: "TBOT STOPPING" never went out). The bot now handles a Ctrl+C itself: it logs it and keeps running.
# To stop the bot on purpose, run tools\stop_bot.ps1 (it writes logs\stop.flag); tools\start_bot.ps1 starts it again.
# The log's times are written in one fixed format (yyyy-MM-dd HH:mm:ss), whatever the Windows language settings.

$LogFile = "C:\TradingBot\TBOT\logs\bot_$(Get-Date -Format 'yyyyMMdd').log"
$ErrorLog = "C:\TradingBot\TBOT\logs\bot_error_$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log {
    param([string]$Message)
    $Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss", [System.Globalization.CultureInfo]::InvariantCulture)
    "$Timestamp - $Message" | Tee-Object -FilePath $LogFile -Append
}

Write-Log "=========================================="
Write-Log "Starting Trading Bot"
Write-Log "=========================================="

# B14 Part 3A: the Ctrl+C / Ctrl+Break protection, before the bot is started.
# 1. Ctrl+C: this PowerShell ignores it, and Windows passes "ignore Ctrl+C" on to the programs it starts (the bot then
#    switches it back on for itself only, to log each one and carry on).
# 2. Ctrl+Break: a small handler, added after PowerShell's own, answers "handled" first, so PowerShell's handler (which
#    would stop this script or pause it in the debugger) never runs. Closing the window or a shutdown still work as before.
try {
    Add-Type -ErrorAction Stop -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TbotCtrlGuard {
    public delegate bool Handler(uint ctrlType);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
    [DllImport("kernel32.dll", SetLastError = true, EntryPoint = "SetConsoleCtrlHandler")]
    private static extern bool SetConsoleCtrlHandlerFn(Handler handler, bool add);
    private static Handler keep;
    private static bool OnSignal(uint ctrlType) { return ctrlType == 0 || ctrlType == 1; }
    public static bool IgnoreCtrlC() { return SetConsoleCtrlHandler(IntPtr.Zero, true); }
    public static bool CatchCtrlBreak() { keep = new Handler(OnSignal); return SetConsoleCtrlHandlerFn(keep, true); }
}
'@
    $g1 = [TbotCtrlGuard]::IgnoreCtrlC()
    $g2 = [TbotCtrlGuard]::CatchCtrlBreak()
    $env:TBOT_CTRL_GUARD = "Ctrl+C ignored = $g1 | Ctrl+Break caught = $g2"     # the bot reads it: anything not True alarms
    Write-Log ("Ctrl+C protection: " + $env:TBOT_CTRL_GUARD)
} catch {
    $env:TBOT_CTRL_GUARD = "NOT set -- " + $_.Exception.Message
    Write-Log ("Ctrl+C protection: " + $env:TBOT_CTRL_GUARD)
}

try {
    # Set working directory
    Set-Location "C:\TradingBot\TBOT"

    # Run the bot
    Write-Log "Executing: C:\TradingBot\TBOT\venv\Scripts\python.exe C:\TradingBot\TBOT\main.py"
    & "C:\TradingBot\TBOT\venv\Scripts\python.exe" "C:\TradingBot\TBOT\main.py" 2>&1 | Tee-Object -FilePath $LogFile -Append

    $ExitCode = $LASTEXITCODE
    Write-Log "Bot exited with code: $ExitCode"

    if ($ExitCode -ne 0) {
        Write-Log "ERROR: Bot exited with non-zero code" | Tee-Object -FilePath $ErrorLog -Append
    }

} catch {
    Write-Log "EXCEPTION: $($_.Exception.Message)" | Tee-Object -FilePath $ErrorLog -Append
    Write-Log "Stack Trace: $($_.ScriptStackTrace)" | Tee-Object -FilePath $ErrorLog -Append
}

Write-Log "=========================================="
Write-Log "Bot execution completed"
Write-Log "=========================================="
