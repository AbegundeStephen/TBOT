import pandas as pd
from datetime import datetime
mt5 = __import__("MetaTrader5")
# B14 item 7.5 (Desire 5 Oct ruling): without MT5 this tool used to print "MT5 connected: False" and carry on,
# so nothing was checked and the Saturday job still looked fine. It now stops with an error code, so the job
# reports the failure (and sends its Telegram).
import sys as _sys
_ok = mt5.initialize()
print("MT5 connected:", _ok)
if not _ok:
    print("FAILED: MT5 is not connected (" + str(mt5.last_error()) + ") -- nothing was checked.")
    _sys.exit(2)
S = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "EURUSD": "EURUSDm", "USOIL": "USOILm", "GBPAUD": "GBPAUDm"}
T = [("1h", mt5.TIMEFRAME_H1, 1), ("4h", mt5.TIMEFRAME_H4, 4)]
F = {(a, tf): pd.read_csv("data/raw/%s_%s.csv" % (s, tf), parse_dates=[0], index_col=0) for a, s in S.items() for tf, k, h in T}
F = {key: f.set_axis((f.index.tz_localize(None) if getattr(f.index, "tz", None) is not None else f.index).astype("datetime64[ns]")) for key, f in F.items()}
F = {key: f[~f.index.duplicated(keep="last")] for key, f in F.items()}
R = {(a, tf): pd.DataFrame(mt5.copy_rates_range(s, k, datetime(2025, 6, 1), datetime.now())) for a, s in S.items() for tf, k, h in T}
M = {key: r.set_axis(pd.to_datetime(r["time"], unit="s").astype("datetime64[ns]")) for key, r in R.items() if len(r)}
C = {key: F[key].index.intersection(M[key].index) for key in M}
D = {key: [ts for ts in C[key] if abs(float(F[key]["close"][ts]) - float(M[key]["close"][ts])) > 1e-4 * abs(float(M[key]["close"][ts]))] for key in M}
H = {tf: h for tf, k, h in T}
print("--- saved price files (data/raw) against MT5's own candles, 1 Jun 2025 to now ---")
[print("%-7s %-3s  compared %6d  differ %4d  (%.2f%%)  since 24 Sep: %d   last 3: %s" % (key[0], key[1], len(C[key]), len(D[key]), 100.0 * len(D[key]) / max(1, len(C[key])), sum(1 for ts in D[key] if ts >= pd.Timestamp("2026-09-24")), ", ".join("%s (file %.6g / MT5 %.6g)" % ((ts + pd.Timedelta(hours=H[key[1]])).strftime("%d %b %H:%M"), float(F[key]["close"][ts]), float(M[key]["close"][ts])) for ts in D[key][-3:]) or "-")) for key in sorted(M)]
print("--- hour of day (UTC, candle close) of every differing candle ---")
print(pd.Series([(ts + pd.Timedelta(hours=H[key[1]])).hour for key in M for ts in D[key]]).value_counts().sort_index().to_dict())
