"""B12 weekly additions (Saturday's run, after all_tests.py --weekly). Read-only: the trade diary and the log.
Every label (decisions 27, 28, 30, 32 + the originals), yesterday's high/low (34), the paper market (7), the new markets
(2, 3, 5), and how often a candle was corrected after use (35, 40 -- rule 13)."""
import glob, json, os
from datetime import datetime, timedelta, timezone
import pandas as pd

def diary(days):
    since, out = datetime.now(timezone.utc) - timedelta(days=days), []
    for f in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
                t = pd.Timestamp(r.get("close_time") or r.get("exit_time") or r.get("closed_at"))
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            except Exception:
                continue
            if t >= since:
                rv = next((r.get(k) for k in ("net_pnl_r", "pnl_r", "r_multiple", "gross_r") if isinstance(r.get(k), (int, float))), None)
                out.append((r, None if rv is None else float(rv)))
    return out

def line(lab, v):
    v = [x for x in v if x is not None]
    if not v:
        return "   %-46s%6d" % (lab, 0)
    return "   %-46s%6d%+10.2f%8.0f%%" % (lab, len(v), sum(v) / len(v), 100.0 * sum(1 for x in v if x > 0) / len(v))

def main():
    print("=" * 110)
    print("B12 WEEKLY ADDITIONS -- %s -- read-only" % datetime.now().strftime("%d %b %Y %H:%M"))
    print("=" * 110)
    rows = diary(120)
    live = [(r, rv) for r, rv in rows if str(r.get("source", "live")) == "live" and isinstance(r.get("ns_labels"), dict)]
    print("\nEVERY LABEL -- live trades, last 120 days (trades, R a trade, win rate)")
    labs = {}
    for r, rv in live:
        for k in ("aplus", "watch"):
            for lab in (r["ns_labels"].get(k) or []):
                labs.setdefault(("A+" if k == "aplus" else "watch") + ": " + lab, []).append(rv)
    for lab in sorted(labs):
        print(line(lab, labs[lab]))
    rev = [(r, rv) for r, rv in live if (r["ns_labels"].get("yday_hl") is not None)]
    print(line("reversals turning at yesterday's high/low (34)", [rv for r, rv in rev if r["ns_labels"].get("yday_hl")]))
    print(line("reversals turning elsewhere", [rv for r, rv in rev if not r["ns_labels"].get("yday_hl")]))
    print("   (tracking only -- nothing trades on these; a rule is only ever proposed with 20-30+ trades, for your ruling)")
    print("\nNEW MARKETS -- live trades, last 120 days")
    for a in ("JP225", "EURJPY", "SILVER"):
        print(line(a, [rv for r, rv in rows if str(r.get("asset", "")).upper() == a and str(r.get("source", "live")) == "live"]))
    pap = [rv for r, rv in rows if str(r.get("gate_id") or r.get("gate_blocked_by") or "").startswith("explore_paper")]
    print(line("AUDJPY on paper (practice lane)", pap))
    print("\nCANDLES CORRECTED AFTER USE -- log lines, last 7 days (rule 13: these used to be silent)")
    since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
    n = {"[NS-DATA]": 0, "[UPDATE-REVISED]": 0, "[COUNCIL-LADDER]": 0, "[PAPER-MARKET]": 0}
    for f in glob.glob(os.path.join("logs", "trading_bot.log*")):
        for ln in open(f, encoding="utf-8", errors="replace"):
            if ln[:16] >= since:
                for k in n:
                    if k in ln:
                        n[k] += 1
    for k, v in n.items():
        print("   %-24s %d" % (k, v))
    print("   [NS-DATA] = the engine saw a used candle change and re-checked; [UPDATE-REVISED] = a saved candle was")
    print("   corrected; [PAPER-MARKET] should always be 0 (a live signal on a paper-only market).")

if __name__ == "__main__":
    main()
