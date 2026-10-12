"""B15 (rule set 4.4): the price export for running the box test HERE (with Claude) instead of on the box. Run on the
box, from C:\\TradingBot\\TBOT, with MT5 open. Read-only for the bot: it only READS MT5 and the saved files, and writes
  data\\b15_export\\<symbol>_30m.csv, _1d.csv, _1w.csv  (MT5's own 30-minute / daily / weekly candles, as the box test
                                                   reads them -- its OWN folder: the bot's updater owns data\\raw\\<symbol>_1d.csv)
  data\\b15_export\\lot_values.json                  (dollars per 1.0 at the smallest lot, and the spreads, from MT5)
  b15_export_<date>.zip  (in C:\\TradingBot\\TBOT)  = those files + data\\raw\\*_1h.csv + the newest log files (for the spreads paid)
Send the zip to Claude. usage: python tools\\b15_export.py"""
import glob, importlib.util, json, os, sys, time, zipfile
import pandas as pd
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
_sp = importlib.util.spec_from_file_location("b14_box_test", os.path.join(ROOT, "tools", "b14_box_test.py"))
B14 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(B14)
try:
    import MetaTrader5 as mt5
except ImportError:
    print("FAILED: this python has no MetaTrader5 module -- run with the bot's python (venv\\Scripts\\python.exe)")
    sys.exit(2)
if not mt5.initialize():
    print("FAILED: MT5 is not connected (%s) -- open the terminal and log in, then run again" % str(mt5.last_error()))
    sys.exit(2)
stamp = time.strftime("%Y-%m-%d_%H%M")
OUT = os.path.join("data", "b15_export")
os.makedirs(OUT, exist_ok=True)
written = []
for a, sym in B14.MARKETS.items():
    mt5.symbol_select(sym, True)
    for name, tf, n, suf in (("M30", mt5.TIMEFRAME_M30, 30000, "30m"), ("D1", mt5.TIMEFRAME_D1, 2500, "1d"),
                             ("W1", mt5.TIMEFRAME_W1, 700, "1w")):
        r = mt5.copy_rates_from_pos(sym, tf, 0, n)
        if r is None or len(r) == 0:
            print("   %-7s %s: MT5 returned nothing (%s)" % (a, name, str(mt5.last_error())))
            continue
        f = pd.DataFrame(r)
        f.index = pd.to_datetime(f["time"], unit="s")
        f.index.name = "timestamp"
        p = os.path.join(OUT, "%s_%s.csv" % (sym, suf))
        f[["open", "high", "low", "close"]].to_csv(p)
        written.append(p)
        print("   %-7s %s: %d candles -> %s" % (a, name, len(f), p))
usd, spreads = B14.lot_values()
pj = os.path.join(OUT, "lot_values.json")
json.dump({"exported": stamp, "usd": usd, "spreads": spreads}, open(pj, "w", encoding="utf-8"), indent=1)
written.append(pj)
print("   lot values: %s" % json.dumps({a: round(v, 4) for a, v in usd.items()}))
mt5.shutdown()
z = "b15_export_%s.zip" % stamp
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in written + sorted(glob.glob(os.path.join("data", "raw", "*_1h.csv"))) + \
            sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime)[-3:]:
        zf.write(p)
print("written: %s (%.1f MB) -- send this file to Claude" % (z, os.path.getsize(z) / 1e6))
