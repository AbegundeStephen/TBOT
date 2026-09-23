"""B10 D5: fill the gaps in data/btc_flow_1h.csv -- the file BTC's volume
overlay actually reads (data_manager.py, the "[VOL-SOURCE] BTC ..." lines).

The overlay swaps MT5 tick volume for real Binance volume only when EVERY bar
in the window is covered. Gaps in this file are why every fetch says "mixed
coverage 10/12 -- using tick volume", and why BTC's volume judge sat dead for
340 sittings. B9's backfill wrote into the price files instead -- the wrong
file (that tool is deleted in this batch).

APPEND-ONLY. Existing rows -- which also carry open interest, funding and
basis -- are never touched. Only missing hours are added, stamped exactly as
the harvester stamps them (open time of the closed kline, UTC hour, seconds),
with volume and taker-buy ratio from Binance and the other columns blank
(nothing live reads them). The overlay de-duplicates on bar_ts.

Run:  python tools/btc_flow_backfill.py [days]      (default 120)
"""
import csv
import os
import sys
import time

import requests

CSV = "data/btc_flow_1h.csv"
COLS = ["bar_ts", "volume", "taker_buy_ratio", "oi", "oi_delta_pct", "funding_rate", "basis_pct"]


def existing_hours():
    have = set()
    if os.path.exists(CSV):
        with open(CSV, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    have.add(int(float(r["bar_ts"])))
                except Exception:
                    pass
    return have


def klines(start_ms, end_ms):
    out = []
    while start_ms < end_ms:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1h",
                                 "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                         timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        start_ms = rows[-1][0] + 3_600_000
        time.sleep(0.2)
    return out


def coverage(have, hours=500):
    last_closed = int(time.time() // 3600 * 3600) - 3600
    return sum(1 for i in range(hours) if last_closed - 3600 * i in have), hours


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    have = existing_hours()
    before, n = coverage(have)
    now_ms = int(time.time() * 1000)
    add = []
    for k in klines(now_ms - days * 86_400_000, now_ms):
        ts = int(k[0] // 1000 // 3600 * 3600)          # same stamp as btc_flow_harvester
        if int(k[6]) >= now_ms or ts in have:            # skip the forming hour and hours already there
            continue
        vol = float(k[5])
        add.append([ts, vol, (float(k[9]) / vol) if vol > 0 else "", "", "", "", ""])
        have.add(ts)
    new_file = not os.path.exists(CSV)
    os.makedirs("data", exist_ok=True)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        w.writerows(sorted(add))
    after, _ = coverage(have)
    print(f"[FLOW-BACKFILL] added {len(add)} missing hours; last {n} hours covered: {before} -> {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
