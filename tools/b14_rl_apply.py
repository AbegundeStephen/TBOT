"""THE B14 LEARNER'S APPLY TOOL (Desire 16 Oct 2026) -- writes ONE approved suggestion of the learner (tools/b14_rl.py)
into the config. Only when Desire says yes. Run with the bot STOPPED: the bot reads config.json only at start-up.
  - a backup of every file first (a full copy of the config -- never commit it, never send it);
  - only phase_config.b14_tuning changes; the new value is checked with the rules' own checks before anything is
    written (a value the rules would refuse is refused here too, and nothing is written);
  - every change is recorded in logs/rl/b14_rl_applied.jsonl, so the learner follows it up on the new trades and
    suggests undoing it if it fails there.

  python tools\\b14_rl_apply.py list                     the suggestions so far, newest first, and what was applied
  python tools\\b14_rl_apply.py show <id>                one suggestion: its evidence and the exact change
  python tools\\b14_rl_apply.py apply <id> [files...]    ONLY when Desire says yes: write it (backup first)
  python tools\\b14_rl_apply.py undo <id> [files...]     put back what was there before <id> was applied (backup first)
Default file: config\\config.json. Give the repo copies as well (config\\config.prod.json config\\config.template.json)
so all three stay the same. Exit code 0 when everything asked was done."""
import copy
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
from src.execution import b14_rules as R14      # noqa: E402

SUGG = os.path.join("logs", "rl", "b14_rl_suggestions.jsonl")
APPLIED = os.path.join("logs", "rl", "b14_rl_applied.jsonl")


def _jsonl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def _write_config(path, d, raw, tag):
    bk = path + ".bak_b14_rl_%s_%s" % (tag, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(path, bk)
    ind = 2
    for ln in raw.splitlines()[1:]:
        if ln.strip():
            ind = max(1, len(ln) - len(ln.lstrip(" ")))
            break
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=ind, ensure_ascii=False)
        f.write("\n")
    json.load(open(tmp, encoding="utf-8"))                   # reads back cleanly before it replaces the file
    os.replace(tmp, path)
    return bk


def _check(tuning):
    """The rules' own checks on the whole block -> list of refusals (empty = fine). The rules are put back after."""
    try:
        _v, _c, bad = R14.apply_settings(tuning)
        return bad
    finally:
        R14.apply_settings(None)


def cmd_list():
    S, A = _jsonl(SUGG), _jsonl(APPLIED)
    if not S:
        print("no suggestions yet (%s) -- the learner writes one only when a version passes every check" % SUGG)
    for s in reversed(S):
        done = [a for a in A if a.get("id") == s["id"]]
        print("%s  %s  %s %s -> %s  %s" % (s["id"], s["date"], s["lever"], s["from"], s["to"],
                                          ("APPLIED %s" % done[-1]["date"]) if done and not done[-1].get("undo") else
                                          ("undone %s" % done[-1]["date"]) if done else "not applied"))
    return 0


def cmd_show(sid):
    s = next((x for x in _jsonl(SUGG) if x["id"] == sid), None)
    if s is None:
        print("no suggestion %s" % sid)
        return 1
    print(json.dumps(s, indent=1, default=str))
    return 0


def cmd_apply(sid, files, undo=False):
    S = {x["id"]: x for x in _jsonl(SUGG)}
    if sid not in S:
        print("no suggestion %s -- see: python tools\\b14_rl_apply.py list" % sid)
        return 1
    s, bad_n, wrote = S[sid], 0, 0
    for path in files:
        print("=== %s ===" % path)
        if not os.path.exists(path):
            print("   NOT FOUND -- skipped")
            bad_n += 1
            continue
        raw = open(path, encoding="utf-8-sig").read()
        d = json.loads(raw)
        pc = d.setdefault("phase_config", {})
        before = copy.deepcopy(pc.get("b14_tuning"))
        if undo:
            recs = [a for a in _jsonl(APPLIED) if a.get("id") == sid and a.get("file") == path and not a.get("undo")]
            if not recs:
                print("   %s was never applied to this file -- nothing to undo" % sid)
                bad_n += 1
                continue
            after = recs[-1].get("before")
        else:
            after = dict(before or {})
            after.update(s["change"])
        refused = _check(after)
        if refused:
            print("   REFUSED by the rules' own checks -- nothing written: %s" % "; ".join("%s (%s)" % x for x in refused))
            bad_n += 1
            continue
        if after == before:
            print("   nothing to change (already as asked)")
            continue
        if after is None or after == {}:
            pc.pop("b14_tuning", None)
        else:
            pc["b14_tuning"] = after
        bk = _write_config(path, d, raw, ("undo_" if undo else "") + sid)
        wrote += 1
        os.makedirs(os.path.dirname(APPLIED), exist_ok=True)
        with open(APPLIED, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": sid, "date": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "file": path, "undo": bool(undo),
                                "lever": s["lever"], "from": s["from"], "to": s["to"], "before": before,
                                "after": pc.get("b14_tuning")}) + "\n")
        print("   backup: %s  (a full copy of the config -- never commit it, never send it)" % bk)
        print("   phase_config.b14_tuning: %s -> %s" % (json.dumps(before), json.dumps(pc.get("b14_tuning"))))
    if wrote:
        print("start the bot again: its start-up line '[B14] settings in force' must show the new value")
    return 1 if bad_n else 0


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("list", "show", "apply", "undo"):
        print(__doc__)
        return 1
    if a[0] == "list":
        return cmd_list()
    if len(a) < 2:
        print(__doc__)
        return 1
    if a[0] == "show":
        return cmd_show(a[1])
    return cmd_apply(a[1], a[2:] or [os.path.join("config", "config.json")], undo=(a[0] == "undo"))


if __name__ == "__main__":
    sys.exit(main())
