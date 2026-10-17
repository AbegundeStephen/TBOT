"""B14 (Desire 15-16 Oct 2026) -- the new B14 settings in config\\config.json. Run with the bot STOPPED (the bot reads
config.json only at start-up). A backup is made first; only the named keys change; running it twice changes nothing
more; every change is printed.

  python tools\\b14_config.py install [files...]  -- at the install, before the bot is started:
        phase_config.b14_new_rules_enabled         = false   (the new B14 rules: present but OFF -- the B13 rules trade)
        the old switches removed (their code is gone): b14_rules_enabled, b14_knobs, b15_small_4h_only, lane_b_enabled,
                                                        b18_rules_enabled
        AUDJPY live (Desire 15 Oct): off phase_config.ns_paper_markets; ns_markets.AUDJPY.min_reward_ccy = 0.0 (no $5
        minimum, like EURUSD / GBPAUD / EURJPY); ns_markets.AUDJPY.cont_only = false (both ways, like the others)
  python tools\\b14_config.py on [files...]       -- ONLY when Desire says so: phase_config.b14_new_rules_enabled = true.
        Refused until 'install' has run (an old switch still in the file). The engine rebuilds every market's memory
        by itself on the next start (the rules changed).
  python tools\\b14_config.py off [files...]      -- phase_config.b14_new_rules_enabled = false (the B13 rules trade
        again; AUDJPY stays live -- 'audjpy_paper' puts it back on the paper list)
  python tools\\b14_config.py audjpy_paper [files...] -- AUDJPY back to the 15 Oct values: paper list ["AUDJPY"],
        min_reward_ccy removed (the $5 default), cont_only true
  python tools\\b14_config.py show [files...]     -- print the current values, change nothing
The learner's settings (phase_config.b14_tuning) are written only by tools\\b14_rl_apply.py.
Default file: config\\config.json. Give the repo copies too (config\\config.prod.json config\\config.template.json) so
all three stay the same."""
import json
import os
import shutil
import sys
import time

MODES = ("install", "on", "off", "audjpy_paper", "show")
OLD_KEYS = ("b14_rules_enabled", "b14_knobs", "b15_small_4h_only", "lane_b_enabled", "b18_rules_enabled")


def show(d):
    pc = d.get("phase_config") or {}
    print("   phase_config.b14_new_rules_enabled = %s" % json.dumps(pc.get("b14_new_rules_enabled", "(not set)")))
    print("   phase_config.b14_tuning            = %s" % json.dumps(pc.get("b14_tuning", "(not set: the rules file's own)")))
    print("   phase_config.ns_paper_markets      = %s" % json.dumps(pc.get("ns_paper_markets", "(not set)")))
    old = [k for k in OLD_KEYS if k in pc]
    print("   old switches still in the file      = %s" % (", ".join(old) if old else "none"))
    aj = (pc.get("ns_markets") or {}).get("AUDJPY")
    print("   phase_config.ns_markets.AUDJPY: %s" % (json.dumps({k: aj.get(k, "(not set)") for k in
                                                                 ("min_reward_ccy", "cont_only", "target_atr", "exit")})
                                                    if aj else "(NO AUDJPY BLOCK -- tell Claude)"))


def apply(d, mode):
    """-> (list of change texts, error text or None). Changes d in place."""
    pc = d.setdefault("phase_config", {})
    changed = []

    def setk(blk, key, val, name):
        if blk.get(key, "(not set)") != val:
            changed.append("%s: %s -> %s" % (name, json.dumps(blk.get(key, "(not set)")), json.dumps(val)))
            blk[key] = val

    if mode == "on":
        old = [k for k in OLD_KEYS if k in pc]
        if old:
            return changed, "the old switches are still in this file (%s) -- run 'install' first" % ", ".join(old)
        setk(pc, "b14_new_rules_enabled", True, "phase_config.b14_new_rules_enabled")
        return changed, None
    if mode == "off":
        setk(pc, "b14_new_rules_enabled", False, "phase_config.b14_new_rules_enabled")
        return changed, None
    aj = (pc.get("ns_markets") or {}).get("AUDJPY")
    if aj is None:
        return changed, "no phase_config.ns_markets.AUDJPY block -- nothing changed; tell Claude"
    if mode == "install":
        setk(pc, "b14_new_rules_enabled", False, "phase_config.b14_new_rules_enabled")
        for k in OLD_KEYS:
            if k in pc:
                changed.append("phase_config.%s: %s -> (removed: its code is gone)" % (k, json.dumps(pc[k])))
                del pc[k]
        paper = [m for m in (pc.get("ns_paper_markets") or []) if str(m).upper() != "AUDJPY"]
        setk(pc, "ns_paper_markets", paper, "phase_config.ns_paper_markets")
        setk(aj, "min_reward_ccy", 0.0, "ns_markets.AUDJPY.min_reward_ccy")
        setk(aj, "cont_only", False, "ns_markets.AUDJPY.cont_only")
        return changed, None
    if mode == "audjpy_paper":
        paper = [m for m in (pc.get("ns_paper_markets") or []) if str(m).upper() != "AUDJPY"] + ["AUDJPY"]
        setk(pc, "ns_paper_markets", paper, "phase_config.ns_paper_markets")
        if "min_reward_ccy" in aj:
            changed.append("ns_markets.AUDJPY.min_reward_ccy: %s -> (removed: the $%s default)" % (
                json.dumps(aj["min_reward_ccy"]), json.dumps(pc.get("min_reward_ccy", 5.0))))
            del aj["min_reward_ccy"]
        setk(aj, "cont_only", True, "ns_markets.AUDJPY.cont_only")
        return changed, None
    return changed, "unknown mode"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    files = sys.argv[2:] or [os.path.join("config", "config.json")]
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
        changed, err = apply(d, mode)
        if err:
            print("   REFUSED: %s" % err)
            bad += 1
            show(json.loads(raw))
            continue
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
        print("   backup: %s  (a full copy of the config -- never commit it, never send it)" % bk)
        for c in changed:
            print("   changed: " + c)
        show(d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
