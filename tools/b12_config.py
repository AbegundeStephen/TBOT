"""B12 config -- run once from C:\\TradingBot\\TBOT. Edits the live config and the template the same way, plus the
council presets file. A .pre_B12 copy of every file is kept first."""
import copy, json, os, shutil
FILES = [f for f in ("config/config.json", "config/config.template.json", "config/config.prod.json") if os.path.exists(f)]
LIVE6 = ["GOLD", "USOIL", "USTEC", "BTC", "EURUSD", "GBPAUD"]
B4 = dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False)
NEW = {  # market: (broker symbol, settings copied from, tested profile (Test 6), stop floor)
    "JP225": ("JP225m", "USTEC", B4, 0.0025),
    "EURJPY": ("EURJPYm", "EURJPY", B4, 0.0012),
    "SILVER": ("XAGUSDm", "GOLD", B4, 0.0035),
    "AUDJPY": ("AUDJPYm", "EURJPY", dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True), 0.0012),
}
for f in FILES:
    c = json.load(open(f, encoding="utf-8-sig"))
    shutil.copy2(f, f + ".pre_B12")
    pc = c.setdefault("phase_config", {})
    pc["council_trial_addons_off"] = False          # the loosening trial ends...
    pc["council_advisory"] = True                   # ...advisory mode replaces it (ruling 1A)
    nm = pc.setdefault("ns_markets", {})
    for a in ("EURUSD", "GBPAUD", "EURJPY"):        # no $5 minimum for these (28 Sep; EURJPY decision 3)
        nm.setdefault(a, {})["min_reward_ccy"] = 0.0
    nm.setdefault("BTC", {})["entry"] = "E2"          # decision 9: BTC's pause-then-turn entry
    for a, (sym, src, prof, floor) in NEW.items():
        e = copy.deepcopy(c["assets"].get(a) or c["assets"][src])
        e.update(symbol=sym, exchange="mt5", enabled=True)
        e.setdefault("risk", {})["min_sl_pct"] = floor
        e["_comment_b12"] = "B12: lot size and step come from MT5 itself (volume_min / volume_step)"
        c["assets"][a] = e
        m = nm.setdefault(a, {})
        m.update(prof)
        m.update(min_rr=0.5, min_sl_pct=floor)
        for sc in (c.get("strategy_configs") or {}).values():          # the old strategies read these (defaults otherwise)
            if isinstance(sc, dict) and a not in sc and src in sc:
                sc[a] = copy.deepcopy(sc[src])
    pc["ns_paper_markets"] = ["AUDJPY"]              # decision 7: practice lane only
    pc["council_ladder"] = "new"                     # decision 13: the council reads the new ladder
    pc["ns_explore"] = {"enabled": True, "1h_break_markets": LIVE6, "all_proofs_markets": ["EURUSD", "GBPAUD"],
                        "bounce_pockets": {"USTEC": ["open FVG (4H)", "4H brain level"],
                                           "GOLD": ["1H brain level", "EMA 20 (1H)"]}}
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=2, ensure_ascii=False)
    en = [a for a, x in c["assets"].items() if x.get("enabled")]
    print("%s | markets on: %s | BTC entry %s | paper %s | council ladder %s | advisory %s" % (
        f, ", ".join(en), nm["BTC"]["entry"], pc["ns_paper_markets"], pc["council_ladder"], pc["council_advisory"]))
    print("   check: session_filter_enabled=%s (must be false)  autotrainer_enabled=%s  small-account threshold=%s" % (
        (c.get("trading") or {}).get("session_filter_enabled"), pc.get("autotrainer_enabled"),
        (c.get("trading") or {}).get("mt5_small_account_threshold_usd", "200 (default)")))
p = "config/aggregator_presets.json"
P = json.load(open(p, encoding="utf-8-sig"))
shutil.copy2(p, p + ".pre_B12")
for a in ("JP225", "SILVER", "AUDJPY"):   # brain settings = the defaults these markets were tested with (else: BTC's)
    P.setdefault("LIVERMORE_PIVOTS", {}).setdefault(a, {"major_mult": 3.5, "minor_mult": 1.0, "dual_confirm": 2, "atr_period": 14,
                                                        "_calibration": "B12: defaults, as tested in Test 6 (not calibrated)"})
with open(p, "w", encoding="utf-8") as fh:
    json.dump(P, fh, indent=2, ensure_ascii=False)
print(p, "| brain settings added:", [a for a in ("JP225", "SILVER", "AUDJPY") if a in P["LIVERMORE_PIVOTS"]])
