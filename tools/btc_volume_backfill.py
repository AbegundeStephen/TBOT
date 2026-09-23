"""B9 S14: fill BTC's real-volume history so the volume judge has something to read.

The bot falls back to tick volume whenever coverage is mixed ("mixed coverage
10/12"), which makes every reading look below average -- 340 dead sittings.
This pulls real volume from Binance for the same window and writes it into the
BTC price files, leaving every other column untouched.
"""

import sys
import pandas as pd
import requests

PAIRS = {"1h": "1h", "4h": "4h", "1d": "1d"}
LIMIT = 1000


def fetch(interval, start_ms):
    out = []
    while True:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval,
                    "startTime": start_ms, "limit": LIMIT},
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        if len(rows) < LIMIT:
            break
        start_ms = rows[-1][0] + 1
    return out


def main():
    for tf, interval in PAIRS.items():
        path = f"data/raw/BTCUSDm_{tf}.csv"
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.index = idx
        start_ms = int(idx.min().timestamp() * 1000)
        rows = fetch(interval, start_ms)
        if not rows:
            print(f"[BACKFILL] {tf}: nothing returned")
            continue
        vol = pd.Series(
            {pd.to_datetime(r[0], unit="ms", utc=True): float(r[5]) for r in rows},
            name="volume",
        )
        before = int((df["volume"] > 0).sum())
        df["volume"] = vol.reindex(df.index).fillna(df["volume"])
        after = int((df["volume"] > 0).sum())
        df.to_csv(path)
        print(f"[BACKFILL] {tf}: real-volume bars {before} -> {after} of {len(df)}")


if __name__ == "__main__":
    sys.exit(main())
