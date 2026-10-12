# B14 items 7.3 / 7.4 (B13 steps 18 and 19, moved into B14 on 8 Oct): runs ONE scheduled read-only job, keeps its
# whole output in logs\, and sends a Telegram if the job fails -- so a scheduled run can never fail silently.
#   -Job forward : the weekly replay         (tools\b15_weekly.py: B15 Part 5 -- the box test's own replay, B13 and B15
#                  side by side, next to MT5; it replaced tools\b13_forward_test.py on 12 Oct) -> logs\b15_weekly_<date>.txt
#   -Job summary : the midnight alarm summary (tools\b13_alarm_summary.py) -> logs\b13_alarm_summary_<date>.txt
# Task Scheduler runs: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\TradingBot\TBOT\tools\b14_task.ps1 -Job forward
param([Parameter(Mandatory = $true)][ValidateSet("forward", "summary")][string]$Job)
Set-Location C:\TradingBot\TBOT
$py = "C:\TradingBot\TBOT\venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Get-Location).Path
$stamp = Get-Date -Format "yyyy-MM-dd"
if ($Job -eq "forward") { $tool = "tools\b15_weekly.py"; $out = "logs\b15_weekly_$stamp.txt" }
else { $tool = "tools\b13_alarm_summary.py"; $out = "logs\b13_alarm_summary_$stamp.txt" }
"=== B14 scheduled job '$Job' started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $out -Encoding utf8
& $py $tool 2>&1 | ForEach-Object { "$_" } | Out-File $out -Append -Encoding utf8
$rc = $LASTEXITCODE
"=== finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), exit code $rc ===" | Out-File $out -Append -Encoding utf8
if ($rc -ne 0) {
    & $py tools\b14_tg_send.py "TBOT SCHEDULED JOB FAILED ($Job, $stamp, exit code $rc) -- see $out on the box" 2>&1 | ForEach-Object { "$_" } | Out-File $out -Append -Encoding utf8
}
exit $rc
