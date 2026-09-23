"""B10 D4: the weekly proofs-versus-random report (Desire, 22 Sep).

Lane A = practice trades the council actually sat on -- proof-based.
Lane C = the random control lane -- both sides, both variants.
If proofs carry information, lane A beats lane C market by market.
Read-only: diary rows in, a table out (screen and logs/reports/).

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


def _utc(value, source):
    """Diary times on UTC: unmarked live times are box-local, practice times UTC."""
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
    groups = defaultdict(list)
    for f in sorted(glob.glob("logs/episodes/episodes_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = _utc(r.get("close_time"), r.get("source"))
            if t is None or t < since or r.get("net_pnl_pct") is None:
                continue
            groups[(str(r.get("asset")), _lane(r))].append(r)

    out = [f"WEEKLY PROOFS vs RANDOM -- last {DAYS} days, to {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
           "",
           f"{'market':<8}{'lane':<6}{'n':>5}{'win%':>7}{'mean %':>10}{'mean R':>9}"]
    for key in sorted(groups):
        rows = groups[key]
        pct = [float(x["net_pnl_pct"]) for x in rows]
        rr = [float(x["net_pnl_r"]) for x in rows if x.get("net_pnl_r") is not None]
        mean_r = (sum(rr) / len(rr)) if rr else float("nan")
        out.append(f"{key[0]:<8}{key[1]:<6}{len(rows):>5}{100 * sum(p > 0 for p in pct) / len(pct):>6.0f}%"
                   f"{sum(pct) / len(pct):>+10.3f}{mean_r:>+9.3f}")
    out += ["", "proofs minus random -- mean % per trade, lane A minus lane C:"]
    for asset in sorted({a for a, _ in groups}):
        a = [float(x["net_pnl_pct"]) for x in groups.get((asset, "A"), [])]
        c = [float(x["net_pnl_pct"]) for x in groups.get((asset, "C"), [])]
        if a and c:
            out.append(f"  {asset:<8}{sum(a) / len(a) - sum(c) / len(c):+.3f}%   (A n={len(a)}, C n={len(c)})")
        else:
            out.append(f"  {asset:<8}not enough rows yet (A n={len(a)}, C n={len(c)})")
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
