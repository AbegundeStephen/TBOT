"""B12.1 config -- run ONCE from C:\\TradingBot\\TBOT with the bot stopped. Sets your break rules (decisions 48/49/50/54)
for all 10 markets in every config file that exists (a .pre_B121 copy of each is kept first), then reads every value
back through the engine's own settings reader (rule 12)."""
import json
import os
import shutil
import sys
sys.path.insert(0, os.getcwd())
FILES = [f for f in ("config/config.json", "config/config.template.json", "config/config.prod.json") if os.path.exists(f)]
MARKETS = ["GOLD", "USOIL", "USTEC", "BTC", "EURUSD", "GBPAUD", "JP225", "EURJPY", "SILVER", "AUDJPY"]
def rules(a):
    return {"quality_lines": True,                          # 48: only lines that held before or sit on a 4H brain level
            "break_rule": "big_or_1h_hold",                 # 48 + 1b: big breaks count at once; small ones need the next 1H candle
            "big_break_atr": 0.25,                          # 48: "big" = a 4H close at least 0.25 of a 4H move past the line
            "reversal_rule": "vote2" if a == "BTC" else "brain4",   # 50 A / 54: brain turning point; BTC 2 of 3
            "room_required": False,                         # 48: room is a tag, not a requirement
            "room_atr": 1.0}                                # 48: room = one 4H move ahead (for the tag)
for f in FILES:
    c = json.load(open(f, encoding="utf-8-sig"))
    shutil.copy2(f, f + ".pre_B121")
    nm = c.setdefault("phase_config", {}).setdefault("ns_markets", {})
    for a in MARKETS:
        m = nm.setdefault(a, {})
        m.update(rules(a))
        m.pop("break_margin_atr", None)                     # the 29 Sep margin rule is replaced by the rules above
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=2, ensure_ascii=False)
    print("%s | rules set for: %s" % (f, ", ".join(MARKETS)))
from src.execution.ns_engine import market_settings     # rule 12: read back the way the engine reads it
pc = json.load(open("config/config.json", encoding="utf-8-sig")).get("phase_config", {})
bad = []
for a in MARKETS:
    s = market_settings(a, pc) or {}
    if any(s.get(k) != v for k, v in rules(a).items()):
        bad.append(a)
print("engine reads your rules for all %d markets (BTC: 2 of 3): %s" % (len(MARKETS), "YES" if not bad else "NO -- " + ", ".join(bad)))
