import os, shutil, pandas as pd
from datetime import datetime, timezone
mt5 = __import__("MetaTrader5")
# B14 item 7.5 (Desire 5 Oct ruling): without MT5 this tool used to print "MT5 connected: False" and carry on,
# so nothing was repaired and the Saturday job still looked fine. It now stops with an error code, so the job
# reports the failure (and sends its Telegram).
import sys as _sys
_ok = mt5.initialize()
print("MT5 connected:", _ok)
if not _ok:
    print("FAILED: MT5 is not connected (" + str(mt5.last_error()) + ") -- nothing was repaired.")
    _sys.exit(2)
SYMS = ["BTCUSDm", "XAUUSDm", "USTECm", "EURUSDm", "USOILm", "GBPAUDm", "JP225m", "EURJPYm", "XAGUSDm", "AUDJPYm"]
TFS = [("15m", mt5.TIMEFRAME_M15, 0.25), ("1h", mt5.TIMEFRAME_H1, 1.0), ("4h", mt5.TIMEFRAME_H4, 4.0)]
NOW = datetime.now(timezone.utc)
SETTLED = pd.Timestamp(NOW.replace(tzinfo=None)) - pd.Timedelta(hours=6)
BK = "data/raw_backup_" + NOW.strftime("%Y%m%d_%H%M")
FILES = [(s, tf, k, h, "data/raw/%s_%s.csv" % (s, tf)) for s in SYMS for tf, k, h in TFS if os.path.exists("data/raw/%s_%s.csv" % (s, tf))]
os.makedirs(BK, exist_ok=True)
[shutil.copy2(x[4], BK) for x in FILES]
print("step 1: backed up %d files to %s" % (len(FILES), BK))
NAME = {x[4]: pd.read_csv(x[4], nrows=0).columns[0] for x in FILES}
OLD = {x[4]: pd.read_csv(x[4], parse_dates=[0], index_col=0) for x in FILES}
TZ = {p: getattr(df.index, "tz", None) is not None for p, df in OLD.items()}
OLD = {p: df.set_axis((df.index.tz_localize(None) if TZ[p] else df.index).astype("datetime64[ns]")) for p, df in OLD.items()}
OLD = {p: df[~df.index.duplicated(keep="last")].sort_index() for p, df in OLD.items()}
RAW = {x[4]: pd.DataFrame(mt5.copy_rates_range(x[0], x[2], OLD[x[4]].index.min().to_pydatetime().replace(tzinfo=timezone.utc), NOW)) for x in FILES}
HRS = {x[4]: x[3] for x in FILES}
MT = {p: r.set_axis(pd.to_datetime(r["time"], unit="s").astype("datetime64[ns]")) for p, r in RAW.items() if len(r)}
MT = {p: m[m.index + pd.Timedelta(hours=HRS[p]) <= SETTLED] for p, m in MT.items()}
PX = ["open", "high", "low", "close"]
COMMON = {p: OLD[p].index.intersection(MT[p].index) for p in MT}
BAD = {p: int(((OLD[p].loc[COMMON[p], PX].astype(float) - MT[p].loc[COMMON[p], PX].astype(float)).abs() > 1e-4 * MT[p].loc[COMMON[p], ["close"]].astype(float).abs().values).any(axis=1).sum()) for p in MT}
FIX = {p: OLD[p].copy() for p in MT}
[FIX[p].update(MT[p][PX]) for p in MT]
ADD = {p: MT[p].loc[MT[p].index.difference(OLD[p].index)] for p in MT}
ADD = {p: pd.DataFrame({c: (a[c] if c in a.columns else (a["tick_volume"] if c == "volume" and "tick_volume" in a.columns else float("nan"))) for c in OLD[p].columns}, index=a.index) for p, a in ADD.items()}
OUT = {p: pd.concat([FIX[p], ADD[p]]).sort_index() if len(ADD[p]) else FIX[p] for p in MT}
OUT = {p: df.set_axis(df.index.tz_localize("UTC") if TZ[p] else df.index) for p, df in OUT.items()}
[(OUT[p].to_csv(p + ".tmp", index=True, index_label=NAME[p]), os.replace(p + ".tmp", p)) for p in OUT]
print("step 2: repaired from MT5 (candles settled for 6+ hours only; older rows MT5 no longer has are kept as they were)")
[print("   %-24s rows %6d -> %6d   stale candles fixed %4d   missing candles added %4d" % (os.path.basename(p), len(OLD[p]), len(OUT[p]), BAD[p], len(ADD[p]))) for p in sorted(OUT)]
print("   files MT5 returned nothing for (left untouched):", [os.path.basename(x[4]) for x in FILES if x[4] not in MT] or "none")
print("TOTAL stale candles fixed: %d   |   to undo: copy the files in %s back into data\\raw" % (sum(BAD.values()), BK))
OLDBK = sorted(d for d in os.listdir("data") if d.startswith("raw_backup_"))[:-4]
[shutil.rmtree(os.path.join("data", d), ignore_errors=True) for d in OLDBK]
print("old backups removed (the newest 4 are kept):", OLDBK or "none")
