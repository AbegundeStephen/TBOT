"""B14 (Desire 8-9 Oct) -- the B14 settings in config\\config.json. Run with the bot STOPPED (the bot reads
config.json only at start-up). A backup is made first; only the named keys change; running it twice changes nothing
more; it prints every change.

  python tools\\b14_config.py phase1 [files...]  -- at the B14 install (live at once):
        risk_management.daily_loss_brakes_enabled = false   (2.1/2.2: the 4% and the hidden 10% day brakes off)
        portfolio.profit_lock_enabled             = false   (2.3: the profit lock off; the 15% top brake and the
                                                              5-loss brake stay)
        phase_config.b14_rules_enabled            = false   (the new entry/exit rules: present but OFF)
        phase_config.ns_engine_days               = 75      (1.2: the engine's own history, in days)
        trading.look_on_candle_close              = true    (3.1: one look at every 30-minute close)
        trading.look_delay_seconds                = 30
        trading.practice_tick_seconds             = 5       (3.3: practice exits in the trade loop)
  python tools\\b14_config.py phase2 [files...]  -- ONLY after Desire has reviewed the box test:
        phase_config.b14_rules_enabled            = true    (sections 5 and 6 switch on)
        phase_config.ns_markets.BTC.target_atr    = 4.0     (6.1: BTC aims for 4 hourly moves ...)
        phase_config.ns_markets.BTC.exit          = "FIXED" (... with a fixed exit; its runner goes)
        The engine rebuilds every market's memory by itself on the next start (the rules changed).
  python tools\\b14_config.py undo2 [files...]   -- back to phase 1: the B14 rules off, BTC back on its runner
        (target_atr null, exit "RUNNER" -- the B13 values). The memory rebuilds by itself again.
  python tools\\b14_config.py set KEY=VALUE [files...]  -- one quick switch, only these keys:
        trading.look_on_candle_close=false     (the old 5-minute timer, no code change)
        risk_management.daily_loss_brakes_enabled=true   portfolio.profit_lock_enabled=true   (brakes back on)
        ...or any other key listed under phase1 / phase2 above. VALUE is JSON: true, false, 75, 4.0, "FIXED", null
  python tools\\b14_config.py show [files...]    -- print the current values, change nothing
Default file: config\\config.json. Give the repo copies too (config\\config.prod.json config\\config.template.json)
so all three stay the same."""
import json
import os
import shutil
import sys
import time

PHASE1 = [("risk_management", "daily_loss_brakes_enabled", False), ("portfolio", "profit_lock_enabled", False),
          ("phase_config", "b14_rules_enabled", False), ("phase_config", "ns_engine_days", 75),
          ("trading", "look_on_candle_close", True), ("trading", "look_delay_seconds", 30),
          ("trading", "practice_tick_seconds", 5)]
PHASE2 = [("phase_config", "b14_rules_enabled", True)]
PHASE2_BTC = [("target_atr", 4.0), ("exit", "FIXED")]
UNDO2 = [("phase_config", "b14_rules_enabled", False)]
UNDO2_BTC = [("target_atr", None), ("exit", "RUNNER")]
SET_OK = {"%s.%s" % (s, k) for s, k, _v in PHASE1 + PHASE2}
SET_OK_BTC = {"phase_config.ns_markets.BTC.target_atr": "target_atr", "phase_config.ns_markets.BTC.exit": "exit"}


def show(d):
    for sec, key, _v in PHASE1:
        print("   %s.%s = %s" % (sec, key, json.dumps((d.get(sec) or {}).get(key, "(not set)"))))
    btc = ((d.get("phase_config") or {}).get("ns_markets") or {}).get("BTC")
    print("   phase_config.ns_markets.BTC: %s" % (json.dumps({k: btc.get(k) for k in ("entry", "target_atr", "exit")})
                                                if btc else "(NO BTC BLOCK -- BTC would run on the built-in settings)"))


def parse_set(arg):
    """'section.key=value' -> (top-level changes, BTC changes); refuses any key B14 does not own."""
    if "=" not in arg:
        sys.exit("set needs KEY=VALUE, e.g. trading.look_on_candle_close=false")
    key, raw = arg.split("=", 1)
    key = key.strip()
    try:
        val = json.loads(raw.strip())
    except ValueError:
        sys.exit("the value must be JSON (true, false, a number, \"text\" or null): %s" % raw)
    if key in SET_OK:
        sec, k = key.split(".", 1)
        return [(sec, k, val)], []
    if key in SET_OK_BTC:
        return [], [(SET_OK_BTC[key], val)]
    sys.exit("not a B14 key: %s (allowed: %s)" % (key, ", ".join(sorted(SET_OK | set(SET_OK_BTC)))))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("phase1", "phase2", "undo2", "set", "show"):
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    rest = sys.argv[2:]
    top, btc_changes = [], []
    if mode == "phase1":
        top = PHASE1
    elif mode == "phase2":
        top, btc_changes = PHASE2, PHASE2_BTC
    elif mode == "undo2":
        top, btc_changes = UNDO2, UNDO2_BTC
    elif mode == "set":
        if not rest:
            sys.exit("set needs KEY=VALUE")
        top, btc_changes = parse_set(rest[0])
        rest = rest[1:]
    files = rest or [os.path.join("config", "config.json")]
    bad = 0
    for path in files:
        print("=== %s ===" % path)
        if not os.path.exists(path):
            print("   NOT FOUND -- skipped")
            bad += 1
            continue
        raw = open(path, encoding="utf-8-sig").read()
        d = json.loads(raw)
        if mode == "show":
            show(d)
            continue
        changed = []
        for sec, key, val in top:
            blk = d.setdefault(sec, {})
            if blk.get(key, "(not set)") != val:
                changed.append("%s.%s: %s -> %s" % (sec, key, json.dumps(blk.get(key, "(not set)")), json.dumps(val)))
                blk[key] = val
        if btc_changes:
            btc = ((d.get("phase_config") or {}).get("ns_markets") or {}).get("BTC")
            if btc is None:
                print("   NO phase_config.ns_markets.BTC BLOCK -- BTC left alone; tell Claude")
                bad += 1
            else:
                for key, val in btc_changes:
                    if btc.get(key, "(not set)") != val:
                        changed.append("ns_markets.BTC.%s: %s -> %s" % (key, json.dumps(btc.get(key, "(not set)")),
                                                                        json.dumps(val)))
                        btc[key] = val
        if not changed:
            print("   nothing to change (already applied)")
            show(d)
            continue
        bk = path + ".bak_b14_%s_%s" % (mode, time.strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(path, bk)
        tmp = path + ".tmp"
        _ind = 2                                                      # keep the file's own indentation
        for _ln in raw.splitlines()[1:]:
            if _ln.strip():
                _ind = max(1, len(_ln) - len(_ln.lstrip(" ")))
                break
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=_ind, ensure_ascii=False)
            f.write("\n")
        json.load(open(tmp, encoding="utf-8"))                     # reads back cleanly before it replaces the file
        os.replace(tmp, path)
        print("   backup: %s" % bk)
        for c in changed:
            print("   changed: " + c)
        show(d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
