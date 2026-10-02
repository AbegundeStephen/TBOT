# B13 pre-check (READ-ONLY): the council switches and trial ledger, today's entry settings, and MT5 30m/daily/weekly
# access. Changes nothing. Run from C:\TradingBot\TBOT with the bot stopped or running.
import json, os, sys
pc = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig")).get("phase_config", {})
print("council_advisory:", pc.get("council_advisory"), "| council_suspended:", pc.get("council_suspended"),
      "| paper-only markets:", pc.get("ns_paper_markets"))
for a, m in sorted((pc.get("ns_markets") or {}).items()):
    print("   %-7s entry=%s package=%s exit=%s" % (a, m.get("entry"), m.get("package"), m.get("exit")))
p = os.path.join("data", "council_trial.json")
print("council trial ledger:", json.dumps(json.load(open(p, encoding="utf-8")))[:600] if os.path.exists(p) else "none")
try:
    import MetaTrader5 as mt5
    import pandas as pd
    if not mt5.initialize():
        sys.exit("MT5 not reachable: %s" % (mt5.last_error(),))
    for tf, name in ((mt5.TIMEFRAME_M30, "M30"), (mt5.TIMEFRAME_D1, "D1"), (mt5.TIMEFRAME_W1, "W1")):
        r = mt5.copy_rates_from_pos("XAUUSDm", tf, 0, 5)
        print("MT5 %s:" % name, ("OK, last candle opened %s" % pd.to_datetime(r[-1]["time"], unit="s")) if r is not None and len(r) else "NO DATA")
    mt5.shutdown()
except Exception as e:
    print("MT5 check failed:", e)
