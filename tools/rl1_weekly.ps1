# THE SATURDAY JOB (Task Scheduler "TBOT RL-1 weekly", Saturday 10:00 box time). Read-only towards the bot: it reads
# the logs and the price files, repairs the saved price files from MT5 (a backup is kept), writes reports, and sends
# Telegram messages. It never changes the bot's settings.
# Writes: logs\rl1_weekly_<date>.txt (this report), logs\b14_box_test_<stamp>.txt, logs\b14_forward_<stamp>.txt,
# logs\rl\ (the learner's files).
Set-Location C:\TradingBot\TBOT
$py = "C:\TradingBot\TBOT\venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyy-MM-dd"
$out = "logs\rl1_weekly_$stamp.txt"
if (-not (Test-Path $py)) {
    "=== Saturday job $stamp FAILED: bot python not found at $py ===" | Out-File $out -Encoding utf8
    exit 1
}
$env:PYTHONPATH = (Get-Location).Path
$fail = 0
$failed = @()       # B14 item 7.7 (Desire 8 Oct): the names of the steps that failed, for the Telegram at the end
"=== Saturday job $stamp ===" | Out-File $out -Encoding utf8
"`n--- refresh 15-minute price files ---" | Out-File $out -Append -Encoding utf8
& $py tools\refresh_15m.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "refresh_15m" }
# B12 (decisions 36 A and 37 A): first repair the saved price files from MT5 (a backup is kept), then check them.
"`n--- B12: repair the saved price files (decision 36) ---" | Out-File $out -Append -Encoding utf8
& $py tools\data_repair.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "data_repair" }
"`n--- B12: saved price files against MT5 (decision 37) ---" | Out-File $out -Append -Encoding utf8
& $py tools\data_check.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "data_check" }
"`n--- B12: forward test, paper ideas, proof supply, council vote, labels, exploration ---" | Out-File $out -Append -Encoding utf8
& $py tools\all_tests.py --weekly 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "all_tests" }
"`n--- B12: every label, yesterday's high/low, the paper market, the new markets ---" | Out-File $out -Append -Encoding utf8
& $py tools\weekly_b12.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "weekly_b12" }
"`n--- B10 D4: proofs versus random ---" | Out-File $out -Append -Encoding utf8
& $py tools\weekly_report.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "weekly_report" }
# (B13 item 14, the alarm summary, runs every night instead -- tools\b14_task.ps1, Desire 16 Oct)
"`n--- B14 (Desire 15-16 Oct): the week's B14 trades (live and practice), then the B14 box test ---" | Out-File $out -Append -Encoding utf8
& $py tools\b14_weekly.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "b14_weekly" }
& $py tools\b14_box_test.py --months 16 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "b14_box_test" }
# B14 (Desire 16 Oct, "keep all 3"): the week's trades of both rule sets next to what MT5 traded, what is waiting,
# and the 30-trade rule (D3) on the new B14 rules' live trades -- also kept in logs\b14_forward_<stamp>.txt
"`n--- B14: the weekly forward report (the week's replay next to MT5, waiting setups, the 30-trade rule) ---" | Out-File $out -Append -Encoding utf8
& $py tools\b14_forward.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "b14_forward" }
# B14 (Desire 16 Oct): the learner -- suggestions only, it changes nothing; then its one-line Telegram summary
"`n--- B14: the learner (RL-1 for the new B14 rules, suggestions only) ---" | Out-File $out -Append -Encoding utf8
Remove-Item "logs\rl\b14_rl_telegram.txt" -ErrorAction SilentlyContinue     # never last week's summary again
& $py tools\b14_rl.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1; $failed += "b14_rl" }
if (Test-Path "logs\rl\b14_rl_telegram.txt") {
    $rlmsg = (Get-Content "logs\rl\b14_rl_telegram.txt" -Raw).Trim()
    & $py tools\b14_tg_send.py $rlmsg 2>&1 | Out-File $out -Append -Encoding utf8
}
# B14 item 7.7 (Desire 8 Oct): a failed Saturday run is no longer silent -- one Telegram naming the failed steps
if ($fail -ne 0) {
    "`n--- B14: Telegram about the failed steps ---" | Out-File $out -Append -Encoding utf8
    & $py tools\b14_tg_send.py "TBOT SATURDAY JOB FAILED ($stamp): $($failed -join ', ') -- see logs\rl1_weekly_$stamp.txt on the box" 2>&1 | Out-File $out -Append -Encoding utf8
}
exit $fail
