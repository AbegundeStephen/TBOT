"""B10 D6: put MT5 tick volume back into BTC's price files.

B9's volume backfill overwrote the volume column of data/raw/BTCUSDm_1h,
_4h and _1d.csv with Binance volume -- different units -- and the historical
updater has kept appending MT5 tick volume since, so the column now mixes two
kinds of volume. Nothing live reads it (the regime detector, dashboard and
replayer use price only), but any study that touched it would be wrong.
This restores MT5 tick volume wherever MT5 still has the bar, and says how many
bars it could not reach (those keep the Binance figure).

Run:  python tools/restore_btc_raw_volume.py
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd

SYM = "BTCUSDm"
FILES = {"1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4", "1d": "TIMEFRAME_D1"}


def main():
    try:
        import MetaTrader5 as mt5
    except Exception as e:
        print(f"[RESTORE-VOL] MetaTrader5 package not available: {e}")
        return 1
    if not mt5.initialize():
        print(f"[RESTORE-VOL] could not connect to MT5: {mt5.last_error()}")
        return 1
    fail = 0
    try:
        now = datetime.now(timezone.utc)
        for tf, tf_name in FILES.items():
            path = os.path.join("data", "raw", f"{SYM}_{tf}.csv")
            try:
                if not os.path.exists(path):
                    print(f"[RESTORE-VOL] {tf}: {path} missing -- skipped")
                    continue
                df = pd.read_csv(path, parse_dates=[0], index_col=0)
                idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                start = idx.min().to_pydatetime().replace(tzinfo=timezone.utc)
                # timezone-aware on purpose: MT5 reads naive times as local
                rates = mt5.copy_rates_range(SYM, getattr(mt5, tf_name), start, now)
                if rates is None or len(rates) == 0:
                    print(f"[RESTORE-VOL] {tf}: MT5 returned nothing ({mt5.last_error()}) -- file left as is")
                    continue
                m = pd.DataFrame(rates)
                m["time"] = pd.to_datetime(m["time"], unit="s")
                tick = m.set_index("time")["tick_volume"]
                hit = pd.Series(idx, index=df.index).map(tick)
                mask = hit.notna().values
                df["volume"] = df["volume"].astype("float64")
                df.loc[mask, "volume"] = hit[mask].astype("float64").values
                tmp = path + ".tmp"
                df.to_csv(tmp)
                os.replace(tmp, path)
                print(f"[RESTORE-VOL] {tf}: {int(mask.sum())} of {len(df)} bars back on MT5 tick volume; "
                      f"{len(df) - int(mask.sum())} older than MT5's history keep the Binance figure")
            except Exception as e:
                fail = 1
                print(f"[RESTORE-VOL] {tf}: FAILED -- {e} (file left as is)")
    finally:
        mt5.shutdown()
    return fail


if __name__ == "__main__":
    sys.exit(main())
