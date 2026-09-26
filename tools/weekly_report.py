"""B11 weekly report (replaces B10 D4).

Rows: the diary, logs/episodes/episodes_*.jsonl.
  LIVE        real trades
  A           practice trades the council sat on (proof-based)
  B-TF, B-MR  practice trades a strategy held back because the proof belonged to the other lane
  C           the random control lane
Plus BTC's own line: its running live total against the -6R pause (data/ns_market_pause.json).
Rows missing a result are COUNTED and listed, never silently dropped.

Run:  python tools/weekly_report.py [days]      (default 7)
"""
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
PCT_KEYS = ("net_pnl_pct", "pnl_pct", "realized_pnl_pct", "gross_pnl_pct")
R_KEYS = ("net_pnl_r", "pnl_r", "r_multiple", "gross_r")
TIME_KEYS = ("close_time", "exit_time", "closed_at")
LANES = {"LIVE": "real trades", "A": "practice: the council sat on a proof",
         "B-TF": "practice: trend strategy held back (proof in the other lane)",
         "B-MR": "practice: reversal strategy held back (proof in the other lane)", "C": "practice: random control"}


def _first(r, keys):
    for k in keys:
        if r.get(k) is not None:
            return r.get(k)
    return None


def _utc(value, source):
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.astimezone() if source != "shadow" else d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _lane(r):
    lane = str(r.get("lane") or "").upper()
    if r.get("source") == "live":
        return "LIVE"
    if lane.startswith("C"):
        return "C"
    return lane or "?"


def main():
    since = datetime.now(timezone.utc) - timedelta(days=DAYS)
    groups, skipped = defaultdict(list), defaultdict(int)
    for f in sorted(glob.glob("logs/episodes/episodes_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = _utc(_first(r, TIME_KEYS), r.get("source"))
            if t is None or t < since:
                continue
            pct = _first(r, PCT_KEYS)
            if pct is None:
                skipped[_lane(r)] += 1
                continue
            r["_pct"] = float(pct)
            rv = _first(r, R_KEYS)
            r["_r"] = float(rv) if rv is not None else None
            groups[(str(r.get("asset")), _lane(r))].append(r)

    out = [f"WEEKLY REPORT -- last {DAYS} days, to {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
           f"{'market':<8}{'lane':<6}{'n':>5}{'win%':>7}{'mean %':>10}{'mean R':>9}{'total R':>9}"]
    for key in sorted(groups):
        rows = groups[key]
        pct = [x["_pct"] for x in rows]
        rr = [x["_r"] for x in rows if x["_r"] is not None]
        mean_r = (sum(rr) / len(rr)) if rr else float("nan")
        out.append(f"{key[0]:<8}{key[1]:<6}{len(rows):>5}{100 * sum(p > 0 for p in pct) / len(pct):>6.0f}%"
                   f"{sum(pct) / len(pct):>+10.3f}{mean_r:>+9.3f}{sum(rr):>+9.2f}")
    out += ["", "proofs minus random -- mean % per trade, lane A minus lane C:"]
    for asset in sorted({a for a, _ in groups}):
        a = [x["_pct"] for x in groups.get((asset, "A"), [])]
        c = [x["_pct"] for x in groups.get((asset, "C"), [])]
        out.append(f"  {asset:<8}" + (f"{sum(a) / len(a) - sum(c) / len(c):+.3f}%   (A n={len(a)}, C n={len(c)})"
                                      if a and c else f"not enough rows yet (A n={len(a)}, C n={len(c)})"))
    try:
        with open("data/ns_market_pause.json", encoding="utf-8") as fh:
            p = json.load(fh).get("BTC") or {}
        out += ["", "BTC (-6R pause): running live total %+.2fR over %d trade(s) -- %s" % (
            float(p.get("total_r", 0.0)), int(p.get("trades", 0)), "PAUSED (paper only)" if p.get("paused") else "live")]
    except FileNotFoundError:
        out += ["", "BTC (-6R pause): no live BTC trades closed yet"]
    if skipped:
        out += ["", "rows with no result field (NOT counted above -- tell Claude): " +
                ", ".join(f"{k} {v}" for k, v in sorted(skipped.items()))]
    out += ["", "lanes: " + "; ".join(f"{k} = {v}" for k, v in LANES.items())]
    text = "\n".join(out)
    print(text)
    os.makedirs("logs/reports", exist_ok=True)
    path = f"logs/reports/weekly_{datetime.now():%Y-%m-%d}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
