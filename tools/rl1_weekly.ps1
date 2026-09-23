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
foreach ($k in "trail_mult", "be_r", "grade_table") {
    "`n--- replayer arms: $k ---" | Out-File $out -Append -Encoding utf8
    & $py tools\replayer.py --arms $k 2>&1 | Out-File $out -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { $fail = 1 }
}
"`n--- bandit (suggestions only) ---" | Out-File $out -Append -Encoding utf8
& $py tools\bandit.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- replayer by asset ---" | Out-File $out -Append -Encoding utf8
& $py tools\replayer.py --by asset 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B10 D3: trail x breakeven grid ---" | Out-File $out -Append -Encoding utf8
& $py tools\replayer.py --grid 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
"`n--- B10 D4: proofs versus random ---" | Out-File $out -Append -Encoding utf8
& $py tools\weekly_report.py 2>&1 | Out-File $out -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { $fail = 1 }
exit $fail
