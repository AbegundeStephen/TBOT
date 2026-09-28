# RL-1 weekly run (suggestions only). Reads logs only; never touches the bot or its settings.
# B8-9: first refreshes the 15-minute price files, so the replayer has a path to walk.
# Writes: data\raw\<symbol>_15m.csv (appends), logs\replayer_arms_<knob>_<date>.json,
#         one run line in logs\rl1_proposals.jsonl, and a readable report logs\rl1_weekly_<date>.txt
Set-Location C:\TradingBot\TBOT
$py = "C:\TradingBot\TBOT\venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyy-MM-dd"
$out = "logs\rl1_weekly_$stamp.txt"
if (-not (Test-Path $py)) {
    "=== RL-1 weekly run $stamp FAILED: bot python not found at $py ===" | Out-File $out -Encoding utf8
    exit 1
}
$env:PYTHONPATH = (Get-Location).Path
$fail = 0
"=== RL-1 weekly run $stamp ===" | Out-File $out -Encoding utf8
"`n--- refresh 15-minute price files ---" | Out-File $out -Append -Encoding utf8
& $py tools\refresh_15m.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
# B12 (Desire 28 Sep, RL ruling C): RL-1's three knobs are off for new-engine trades -- its arms are no longer run.
# B12 (decisions 36 A and 37 A): first repair the saved price files from MT5 (a backup is kept), then check them.
"`n--- B12: repair the saved price files (decision 36) ---" | Out-File $out -Append -Encoding utf8
& $py tools\data_repair.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B12: saved price files against MT5 (decision 37) ---" | Out-File $out -Append -Encoding utf8
& $py tools\data_check.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B12: forward test, paper ideas, proof supply, council vote, labels, exploration ---" | Out-File $out -Append -Encoding utf8
& $py tools\all_tests.py --weekly 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B12: every label, yesterday's high/low, the paper market, the new markets ---" | Out-File $out -Append -Encoding utf8
& $py tools\weekly_b12.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B10 D4: proofs versus random ---" | Out-File $out -Append -Encoding utf8
& $py tools\weekly_report.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
exit $fail
