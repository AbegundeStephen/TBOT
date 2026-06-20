#!/usr/bin/env python3
"""
Observability Analyzer  (Remediation Phase 2 → 3 bridge)
========================================================
Read-only. Turns the funnel + shadow-engine data collected during the soak
into a single Phase-3 decision report. Touches nothing the bot uses — safe to
run on the live box at any time.

Inputs (created by the running bot):
  logs/funnel/funnel_<date>.jsonl     - every aggregator evaluation, by stage
  logs/funnel/ai_rejects_<date>.jsonl - signals the AI validator rejected
  logs/shadow/closed_<date>.jsonl     - closed shadow trades (durable archive)
  logs/shadow_state.json              - live snapshot (open + recent scorecard)

What it answers:
  1. FUNNEL  - of all evaluations, how many produced a raw signal, how many
               survived to execution, and which veto family killed the rest.
  2. SHADOW  - for signals blocked by downstream EXECUTION gates, what was the
               forward P&L per gate? Gates that block PROFITABLE signals are
               costing money and are the first Phase-3 candidates to relax.
  3. AI A/B  - how often the AI filter rejected, and (where shadow data exists)
               whether those would have paid.

Usage:
  python scripts/analyze_observability.py                 # last 30 days
  python scripts/analyze_observability.py --days 7        # last 7 days
  python scripts/analyze_observability.py --out report.md # also write markdown
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# repo root = parent of this script's dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNNEL_DIR = os.path.join(ROOT, "logs", "funnel")
SHADOW_DIR = os.path.join(ROOT, "logs", "shadow")
SHADOW_SNAPSHOT = os.path.join(ROOT, "logs", "shadow_state.json")


def _read_jsonl(paths):
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return rows


def _files_within(directory, prefix, days):
    """Return <prefix>_<date>.jsonl files within the last `days` days."""
    found = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.jsonl")))
    if not days:
        return found
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    keep = []
    for fp in found:
        base = os.path.basename(fp)
        date_part = base.replace(f"{prefix}_", "").replace(".jsonl", "")
        if date_part >= cutoff:
            keep.append(fp)
    return keep


def _pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def analyze_funnel(days):
    rows = _read_jsonl(_files_within(FUNNEL_DIR, "funnel", days))
    out = []
    if not rows:
        return ["FUNNEL: no data yet (logs/funnel/funnel_*.jsonl empty or absent)."]

    per_asset = defaultdict(lambda: defaultdict(int))
    stage_totals = defaultdict(int)
    for r in rows:
        a = r.get("asset", "?")
        stage = r.get("stage", "unknown")
        per_asset[a]["evaluations"] += 1
        if stage == "executed":
            # mark_executed records are separate stage rows
            per_asset[a]["executed"] += 1
            continue
        if any(r.get(k) for k in ("mr", "tf", "ema")):
            per_asset[a]["raw_signal"] += 1
        per_asset[a][stage] += 1
        stage_totals[stage] += 1

    out.append("=" * 78)
    out.append("1. SIGNAL FUNNEL  (all aggregator evaluations)")
    out.append("=" * 78)
    for a, c in sorted(per_asset.items()):
        ev = c.get("evaluations", 0)
        out.append(
            f"\n[{a}]  evaluations={ev}  raw_signal={c.get('raw_signal', 0)} "
            f"({_pct(c.get('raw_signal',0), ev)})  "
            f"passed={c.get('passed_to_execution', 0)}  executed={c.get('executed', 0)}"
        )
        blocks = {k: v for k, v in c.items()
                  if k.startswith("blocked") or k == "no_raw_signal"}
        for k, v in sorted(blocks.items(), key=lambda x: -x[1]):
            out.append(f"      {k:<26} {v:>6}  ({_pct(v, ev)} of evals)")

    out.append("\n--- Veto families ranked across all assets (most blocking first) ---")
    for k, v in sorted(stage_totals.items(), key=lambda x: -x[1]):
        if k.startswith("blocked") or k == "no_raw_signal":
            out.append(f"  {k:<26} {v:>7}")
    return out


def _scorecard_from_rows(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        g = r.get(key)
        pnl = r.get("net_pnl_pct")
        if g is None or pnl is None:
            continue
        buckets[g].append(pnl)
    card = {}
    for g, pnls in buckets.items():
        wins = sum(1 for p in pnls if p > 0)
        card[g] = {
            "count": len(pnls),
            "win_rate": round(wins / len(pnls) * 100, 1) if pnls else 0.0,
            "avg_net_pnl": round(sum(pnls) / len(pnls), 3) if pnls else 0.0,
            "total_pnl": round(sum(pnls), 3),
        }
    return card


def analyze_shadow(days):
    rows = _read_jsonl(_files_within(SHADOW_DIR, "closed", days))
    out = ["", "=" * 78,
           "2. SHADOW GATE SCORECARD  (forward P&L of signals blocked by execution gates)",
           "=" * 78]

    if not rows:
        # Fall back to the live snapshot if the durable archive isn't populated yet.
        snap = None
        try:
            with open(SHADOW_SNAPSHOT, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            pass
        if snap and snap.get("closed_results"):
            rows = snap["closed_results"]
            out.append("(using live snapshot logs/shadow_state.json; durable archive empty)")
        else:
            out.append("SHADOW: no closed shadow trades yet — needs soak time to populate.")
            if snap:
                out.append(f"  (snapshot: {snap.get('summary', {})})")
            return out

    out.append(f"closed shadow trades analysed: {len(rows)}\n")
    card = _scorecard_from_rows(rows, "gate_blocked_by")
    # Rank by total_pnl DESC: gates with the most positive blocked P&L are the
    # ones costing you money — they vetoed trades that would have won.
    ranked = sorted(card.items(), key=lambda x: -x[1]["total_pnl"])
    out.append(f"{'GATE':<34}{'n':>5}{'win%':>7}{'avgP&L%':>9}{'totP&L%':>9}   verdict")
    out.append("-" * 78)
    for g, s in ranked:
        verdict = ""
        if s["count"] >= 10:
            if s["total_pnl"] > 0 and s["win_rate"] >= 50:
                verdict = ">> RELAX? blocking profitable signals"
            elif s["total_pnl"] < 0:
                verdict = "keep (blocked losers)"
        else:
            verdict = "(low sample)"
        out.append(
            f"{str(g)[:33]:<34}{s['count']:>5}{s['win_rate']:>7}"
            f"{s['avg_net_pnl']:>9}{s['total_pnl']:>9}   {verdict}"
        )

    out.append("\n--- By strategy source ---")
    scard = _scorecard_from_rows(rows, "strategy_source")
    for g, s in sorted(scard.items(), key=lambda x: -x[1]["total_pnl"]):
        out.append(f"  {str(g):<8} n={s['count']:>4} win={s['win_rate']}% "
                   f"avg={s['avg_net_pnl']}% tot={s['total_pnl']}%")
    return out


def analyze_ai(days):
    rows = _read_jsonl(_files_within(FUNNEL_DIR, "ai_rejects", days))
    out = ["", "=" * 78, "3. AI FILTER REJECTIONS", "=" * 78]
    if not rows:
        out.append("AI: no rejections logged yet.")
        return out
    per_asset = defaultdict(int)
    per_pattern = defaultdict(int)
    for r in rows:
        per_asset[r.get("asset", "?")] += 1
        per_pattern[r.get("pattern") or "n/a"] += 1
    out.append(f"total AI rejections: {len(rows)}")
    out.append("by asset: " + ", ".join(f"{a}={n}" for a, n in sorted(per_asset.items(), key=lambda x: -x[1])))
    out.append("by pattern: " + ", ".join(f"{p}={n}" for p, n in sorted(per_pattern.items(), key=lambda x: -x[1])[:10]))
    out.append("\nNote: AI-rejected signals now ALSO flow to the shadow engine under")
    out.append("gate 'ai_validation' — see that row in the SHADOW scorecard above for")
    out.append("their forward P&L (the true AI A/B). If it shows '>> RELAX?', the AI")
    out.append("filter is vetoing net-winning trades and should be retuned/demoted.")
    return out


def main():
    ap = argparse.ArgumentParser(description="TBOT observability analyzer (read-only).")
    ap.add_argument("--days", type=int, default=30, help="lookback window in days (0 = all).")
    ap.add_argument("--out", type=str, default=None, help="optional path to write the report as markdown.")
    args = ap.parse_args()

    lines = []
    lines.append(f"TBOT OBSERVABILITY REPORT  (lookback={args.days or 'all'} days, "
                 f"generated {datetime.now(timezone.utc).isoformat()})")
    lines += analyze_funnel(args.days)
    lines += analyze_shadow(args.days)
    lines += analyze_ai(args.days)
    lines.append("")
    lines.append("=" * 78)
    lines.append("PHASE-3 READING GUIDE")
    lines.append("=" * 78)
    lines.append("- FUNNEL: the biggest 'blocked_*' family + 'no_raw_signal' tell you whether")
    lines.append("  the bottleneck is the strategies being silent (Audit: MRS) or the vetoes.")
    lines.append("- SHADOW: any gate marked '>> RELAX?' vetoed trades that would have WON —")
    lines.append("  those are the first candidates to loosen in Phase 3 (with a backtest).")
    lines.append("- Treat anything with n<10 as not yet conclusive; let the soak run longer.")

    report = "\n".join(str(x) for x in lines)
    print(report)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("```\n" + report + "\n```\n")
            print(f"\n[written] {args.out}")
        except Exception as e:
            print(f"[warn] could not write {args.out}: {e}")


if __name__ == "__main__":
    main()
