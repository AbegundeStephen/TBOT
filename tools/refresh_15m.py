"""B8-9: refresh the 15-minute price files the replayer walks
(data/raw/<SYMBOL>_15m.csv). Appends every closed 15-minute bar after the last
one already in each file. Read-only on the broker side; its own MT5 connection
is closed at the end. tools/rl1_weekly.ps1 runs it before the replayer.

Why: on 22 Sep all six files ended on 28 Aug, so no trade since the proof
engine went live (14 Sep) had a price path and every arm table was empty.

Run:  python tools/refresh_15m.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

SYMBOLS = ["BTCUSDm", "XAUUSDm", "USTECm", "EURUSDm", "USOILm", "GBPAUDm"]
FIRST_START = datetime(2025, 6, 1, tzinfo=timezone.utc)   # only used if a file is missing


def refresh(mt5, sym, now_utc):
    path = os.path.join("data", "raw", f"{sym}_15m.csv")
    old = pd.read_csv(path, parse_dates=[0], index_col=0) if os.path.exists(path) else None
    # HOTFIX (22 Sep, post-B8): a file written with an explicit UTC offset
    # (tz-aware index) reads back tz-aware here, while freshly fetched bars
    # below are built tz-naive. Concatenating the two silently degrades the
    # index to dtype=object, and the next .sort_index() crashes with
    # "Cannot compare tz-naive and tz-aware timestamps" -- reproduced locally
    # against a file the box had already written with a "+00:00" suffix.
    if old is not None and getattr(old.index, "tz", None) is not None:
        old.index = old.index.tz_localize(None)
    if old is not None and len(old):
        start = old.index.max().to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=15)
    else:
        start = FIRST_START
    # timezone-aware UTC on purpose: the MT5 package treats naive times as
    # local time (the FRAME-1 two-hour lag, 3 Sep).
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M15, start, now_utc)
    if rates is None or len(rates) == 0:
        print(f"[REFRESH-15M] {sym}: no new bars ({mt5.last_error()})")
        return 0
    new = pd.DataFrame(rates)
    new["time"] = pd.to_datetime(new["time"], unit="s")
    new = new.set_index("time")
    # the newest bar is still forming -- keep only closed bars
    new = new[new.index <= (now_utc.replace(tzinfo=None) - timedelta(minutes=15))]
    if old is not None:
        if "volume" in old.columns and "volume" not in new.columns and "tick_volume" in new.columns:
            new = new.rename(columns={"tick_volume": "volume"})
        missing = [c for c in old.columns if c not in new.columns]
        if missing:
            raise RuntimeError(f"new bars lack columns {missing}; file left unchanged")
        new = new[list(old.columns)]
        new.index.name = old.index.name
        merged = pd.concat([old, new])
    else:
        merged = new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = path + ".tmp"
    merged.to_csv(tmp)
    os.replace(tmp, path)
    print(f"[REFRESH-15M] {sym}: +{len(new)} bars, file now {merged.index.min()} .. {merged.index.max()}")
    return len(new)


def main():
    try:
        import MetaTrader5 as mt5
    except Exception as e:
        print(f"[REFRESH-15M] MetaTrader5 package not available: {e}")
        return 1
    if not mt5.initialize():
        print(f"[REFRESH-15M] could not connect to MT5: {mt5.last_error()}")
        return 1
    fail = 0
    try:
        now_utc = datetime.now(timezone.utc)
        for sym in SYMBOLS:
            try:
                refresh(mt5, sym, now_utc)
            except Exception as e:
                fail = 1
                print(f"[REFRESH-15M] {sym}: FAILED -- {e}")
    finally:
        mt5.shutdown()
    return fail


if __name__ == "__main__":
    sys.exit(main())
