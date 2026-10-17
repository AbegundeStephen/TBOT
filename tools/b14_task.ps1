# THE NIGHTLY ALARM SUMMARY (B13 item 14; Desire 16 Oct: every night). Task Scheduler "TBOT alarm summary", every day
# at 00:10 box time (after the log rotates at midnight -- the summary reads every logs\trading_bot.log* file, so the
# rotation can't empty it). Read-only: it reads yesterday's log lines and sends ONE Telegram summary.
# Keeps its whole output in logs\b13_alarm_summary_<date>.txt and sends a Telegram if it fails -- never silent.
#   Task Scheduler runs: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\TradingBot\TBOT\tools\b14_task.ps1
# B14 (16 Oct): the old "-Job forward" (the B15 weekly replay) is gone with the B15 rules; the new weekly report runs in
# the Saturday job (tools\rl1_weekly.ps1). The box never had a task for either job (checked 16 Oct) -- the install sets
# up this one.
param([ValidateSet("summary")][string]$Job = "summary")
Set-Location C:\TradingBot\TBOT
$py = "C:\TradingBot\TBOT\venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
$stamp = Get-Date -Format "yyyy-MM-dd"
$out = "logs\b13_alarm_summary_$stamp.txt"
"=== the nightly alarm summary, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $out -Encoding utf8
& $py tools\b13_alarm_summary.py 2>&1 | ForEach-Object { "$_" } | Out-File $out -Append -Encoding utf8
$rc = $LASTEXITCODE
"=== finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), exit code $rc ===" | Out-File $out -Append -Encoding utf8
if ($rc -ne 0) {
    & $py tools\b14_tg_send.py "TBOT NIGHTLY ALARM SUMMARY FAILED ($stamp, exit code $rc) -- see $out on the box" 2>&1 | ForEach-Object { "$_" } | Out-File $out -Append -Encoding utf8
}
exit $rc
