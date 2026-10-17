"""B14 WEEKLY (Desire 15 Oct) -- read-only. The week's trades under the B14 rules, live and practice, from the diary
(logs\\episodes\\episodes_<day>.jsonl: one row per closed trade, live or practice, with the proof's fields), plus the
B14 lines in the bot log. Run by the Saturday job (tools\\rl1_weekly.ps1); by hand: python tools\\b14_weekly.py [--days 7]

It answers, for the last N days (default 7) and since the B14 switch went on (if the log shows when):
  - how many closed trades carried the B14 rules (live / practice lane A / practice lane X / lane C), won, R
  - the witness split: trades WITH a witness in the 24 h before the entry vs WITHOUT (recorded only -- not a filter)
  - where the targets came from (a line 2-3R, a line out to 3.5R, today's target / BTC's 4 moves) and how trades ended
  - how far up the ladder trades got (rungs 1-3) -- live from the row, practice from the row
  - the log: [B14] start-up lines, [NS-LADDER] moves, signals retired for the clearance, [DIAG-CONFIRM] takes
Nothing here changes a file. Exit code 0 unless the diary or the log could not be read at all."""
import argparse
import collections
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.getcwd())


def _t(v):
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def rows_since(t0):
    out = []
    for p in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        try:
            for ln in open(p, encoding="utf-8"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                ct = _t(r.get("close_time") or r.get("exit_time"))
                if ct is None or ct < t0:
                    continue
                out.append(r)
        except Exception as e:
            print("   could not read %s: %s" % (p, e))
    return out


SNAP = {}          # composite_state_ref -> the proof details (see load_snapshots)


def load_snapshots(refs):
    """The diary writer (src/utils/episode_ledger.py, B9 S9) takes every row's composite_state OUT of the row and keeps
    it once in logs/episodes/snapshots_<day>.jsonl under a 16-character hash; the row keeps only composite_state_ref.
    Every proof detail B14 records (ns_b14n, the witness, the target kind, ns_entry) is in there. Only the refs needed
    are loaded. -> (found, missing)"""
    need = {x for x in refs if x}
    rx = re.compile(r'^\{"ref": "([0-9a-f]{16})"')
    for p in sorted(glob.glob(os.path.join("logs", "episodes", "snapshots_*.jsonl"))):
        try:
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    m = rx.match(ln)
                    if not m or m.group(1) not in need or m.group(1) in SNAP:
                        continue
                    try:
                        SNAP[m.group(1)] = json.loads(ln).get("composite_state") or {}
                    except Exception:
                        pass
        except Exception as e:
            print("   could not read %s: %s" % (p, e))
    return len(need & set(SNAP)), len(need - set(SNAP))


def cs(r):
    """A row's proof details: inline if the row still has them, else from the snapshot files by its ref."""
    return r.get("composite_state") or SNAP.get(r.get("composite_state_ref")) or {}


def r_of(r):
    v = r.get("net_pnl_r")
    if v is None:
        v = r.get("gross_r")
    try:
        return float(v)
    except Exception:
        return None


def lane_of(r):
    if r.get("source") == "live":
        return "live"
    ln = str(r.get("lane") or "?")
    return "practice lane %s" % ln[:1] if ln else "practice"


def block(title, rs):
    print("   %s: %d trades" % (title, len(rs)))
    if not rs:
        return
    R = [x for x in (r_of(r) for r in rs) if x is not None]
    won = sum(1 for x in R if x > 0)
    print("      result: %d with an R, %d won (%.0f%%), %+.1fR in all, %+.3fR per trade" % (
        len(R), won, 100.0 * won / len(R) if R else 0, sum(R), (sum(R) / len(R)) if R else 0))
    w1 = [r for r in rs if cs(r).get("ns_b14n_witness")]
    w0 = [r for r in rs if not cs(r).get("ns_b14n_witness")]
    for nm, grp in (("with a witness", w1), ("without", w0)):
        Rg = [x for x in (r_of(r) for r in grp) if x is not None]
        print("      %-16s %3d trades  %3d won  %+7.1fR  (%+.3f per trade)" % (
            nm, len(grp), sum(1 for x in Rg if x > 0), sum(Rg), (sum(Rg) / len(Rg)) if Rg else 0))
    kinds = collections.Counter(str(cs(r).get("ns_b14n_target_kind")) for r in rs)
    print("      targets: " + ", ".join("%s %d" % kv for kv in sorted(kinds.items())))
    ends = collections.Counter(str(r.get("close_reason") or r.get("exit_reason")) for r in rs)
    print("      ended by: " + ", ".join("%s %d" % kv for kv in sorted(ends.items())))
    rungs = collections.Counter(int(r.get("ns_b14n_step") or 0) for r in rs)
    print("      ladder rungs reached: " + ", ".join("%d: %d" % kv for kv in sorted(rungs.items())))
    wits = collections.Counter(str(cs(r).get("ns_b14n_witness_name") or "-") for r in w1)
    if wits:
        print("      the witnesses: " + ", ".join("%s %d" % kv for kv in wits.most_common(6)))


def log_lines(t0):
    files = sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime)
    pats = collections.OrderedDict([
        ("[B14] start-up lines", re.compile(r"\[B14\] rules")),
        ("[NS-LADDER] stop moves (live)", re.compile(r"\[NS-LADDER\]")),
        ("[NS-LADDER-FAIL] (must be 0)", re.compile(r"\[NS-LADDER-FAIL\]")),
        ("signals retired for the clearance (hourly entry)", re.compile(r"\[NS-SKIP\].*B14: the close beat the peak")),
        ("package records cancelled for the clearance", re.compile(r"\[PKG-CANCEL\].*B14: the push closed only")),
        ("[DIAG-CONFIRM] taken at once on a witness", re.compile(r"\[DIAG-CONFIRM\]")),
        ("[B14] warnings (no map / list not built)", re.compile(r"\[B14\] .*(no line map|not built)")),
    ])
    n = collections.Counter()
    first_on = None
    day0 = t0.strftime("%Y-%m-%d")
    for p in files:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if len(ln) < 10 or ln[:10] < day0:
                        if "[B14] rules B14: B14 rules ON" in ln and first_on is None:
                            first_on = ln[:16]
                        continue
                    if "[B14] rules B14: B14 rules ON" in ln and first_on is None:
                        first_on = ln[:16]
                    for k, rx in pats.items():
                        if rx.search(ln):
                            n[k] += 1
        except Exception as e:
            print("   could not read %s: %s" % (p, e))
    return n, first_on, list(pats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(days=args.days)
    print("=" * 100)
    print("B14 WEEKLY -- %s UTC -- the last %d days (since %s)" % (now.strftime("%Y-%m-%d %H:%M"), args.days, t0.strftime("%Y-%m-%d %H:%M")))
    print("=" * 100)
    try:
        cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
        pc = cfg.get("phase_config") or {}
        print("switch now: b14_new_rules_enabled = %s | ns_paper_markets = %s" % (pc.get("b14_new_rules_enabled", "(absent)"), pc.get("ns_paper_markets")))
    except Exception as e:
        print("config not read: %s" % e)
    rows = rows_since(t0)
    found, missing = load_snapshots([r.get("composite_state_ref") for r in rows])
    print("proof details joined from logs/episodes/snapshots_*.jsonl: %d found, %d missing%s" % (
        found, missing, "  <-- rows without their details count as 'other' -- tell Claude" if missing else ""))
    b14n = [r for r in rows if cs(r).get("ns_b14n")]
    b13 = [r for r in rows if not cs(r).get("ns_b14n")]
    print("closed trades in the diary: %d (B14 rules %d, other %d)" % (len(rows), len(b14n), len(b13)))
    for lane in ("live", "practice lane A", "practice lane X", "practice lane C"):        # lane B removed 16 Oct
        grp = [r for r in b14n if lane_of(r) == lane]
        if grp or lane in ("live", "practice lane A"):
            block("B14 -- %s" % lane, grp)
    other = [r for r in b14n if lane_of(r) not in ("live", "practice lane A", "practice lane X", "practice lane C")]
    if other:
        block("B14 -- other", other)
    live13 = [r for r in b13 if r.get("source") == "live"]
    if live13:
        R = [x for x in (r_of(r) for r in live13) if x is not None]
        print("   live trades NOT under B14 in the window (B13 rules): %d, %+.1fR" % (len(live13), sum(R)))
    n, first_on, keys = log_lines(t0)
    print("the log (last %d days):" % args.days)
    for k in keys:
        print("   %-52s %d" % (k, n[k]))
    print("   first '[B14] rules B14: B14 rules ON' start-up line seen: %s" % (first_on or "none -- the switch has not been on"))
    print("done")


if __name__ == "__main__":
    main()
