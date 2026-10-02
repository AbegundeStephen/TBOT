# B13 config change: entry E on all 10 markets, the package switched on, the council suspended.
# Makes a dated backup first, edits only these keys, then reads the file back to prove it. RUN WITH THE BOT STOPPED.
import datetime, json, os, shutil, sys
p = os.path.join("config", "config.json")
bak = p + ".bak_b13_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
shutil.copy2(p, bak)
cfg = json.load(open(p, encoding="utf-8-sig"))
pc = cfg.get("phase_config")
if not isinstance(pc, dict) or "ns_markets" not in pc:
    sys.exit("STOP: phase_config / ns_markets is not where it should be -- nothing changed. Tell Claude.")
MARKETS = ["BTC", "GOLD", "USTEC", "USOIL", "EURUSD", "GBPAUD", "JP225", "EURJPY", "SILVER", "AUDJPY"]
for a in MARKETS:
    m = pc["ns_markets"].setdefault(a, {})
    m["entry"] = "E"
    m["package"] = True
pc["council_suspended"] = True
with open(p, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2, ensure_ascii=False)
chk = json.load(open(p, encoding="utf-8")).get("phase_config", {})
ok = all(chk["ns_markets"][a]["entry"] == "E" and chk["ns_markets"][a]["package"] is True for a in MARKETS) \
    and chk.get("council_suspended") is True
print("backup:", bak)
print("RESULT:", "OK -- all 10 markets entry=E, package on; council suspended" if ok else "NOT OK -- restore the backup and tell Claude")
