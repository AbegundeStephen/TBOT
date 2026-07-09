#!/usr/bin/env python3
"""
promote_pullback.py — S7.3 auto-promoter

Reads shadow [7.3] log lines + MFE/MAE log, evaluates promotion gates,
and flips phase_config.pullback_completion_enabled = true in config.json
when all gates clear.

Gates (all must pass):
  1. min_shadow_signals  : >= 30 [7.3] scored signals found in log
  2. min_expectancy_ratio: mean(C_pb-discounted quality) / mean(raw quality) >= 0.95
                           (discount must not cost more than 5% quality on average)
  3. min_positive_cpb   : >= 60% of signals have C_pb >= 0.50
                           (score is actually discriminating, not returning noise)

Usage:
  python scripts/promote_pullback.py              # dry-run (print report, no write)
  python scripts/promote_pullback.py --promote    # write to config.json if gates pass
  python scripts/promote_pullback.py --force      # force flip regardless of gates (manual override)
  python scripts/promote_pullback.py --reset      # flip back to false

Run this manually after a week of shadow data, or wire it into a cron / scheduled task.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
CONFIG     = ROOT / "config" / "config.json"
LOG_DIR    = ROOT / "logs"
BOT_LOG    = LOG_DIR / "trading_bot.log"
MFE_LOG    = LOG_DIR / "mfe_mae_log.csv"

# Rotated logs (trading_bot.log.1 … .5)
ROTATED    = [LOG_DIR / f"trading_bot.log.{i}" for i in range(1, 6)]

# ── Gate thresholds (can be overridden via config trade_management block) ─────
DEFAULT_MIN_SIGNALS       = 30
DEFAULT_MIN_EXP_RATIO     = 0.95
DEFAULT_MIN_POSITIVE_RATE = 0.60

# ── Regex for [7.3] shadow log lines ─────────────────────────────────────────
# Format: [7.3] C_pb=0.72 {'fib':0.8,...} -> quality 0.65->0.69
RE_73 = re.compile(
    r"\[7\.3\]\s+C_pb=(?P<cpb>[0-9.]+)"
    r".*?->\s*quality\s+(?P<raw>[0-9.]+)->(?P<adj>[0-9.]+)"
)


def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[PROMOTE] config.json written.")


def read_log_lines():
    """Read all available log lines (current + rotated), oldest first."""
    paths = list(reversed(ROTATED)) + [BOT_LOG]
    lines = []
    for p in paths:
        if p.exists():
            try:
                lines.extend(p.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
    return lines


def parse_shadow_signals(lines):
    """Extract (cpb, raw_quality, adj_quality) tuples from [7.3] log lines."""
    records = []
    for line in lines:
        m = RE_73.search(line)
        if m:
            try:
                records.append((
                    float(m.group("cpb")),
                    float(m.group("raw")),
                    float(m.group("adj")),
                ))
            except ValueError:
                pass
    return records


def evaluate_gates(records, cfg):
    """
    Returns (passed: bool, report: list[str], gate_results: dict).
    """
    tm       = cfg.get("trade_management", {})
    min_sig  = tm.get("pullback_min_shadow_signals",  DEFAULT_MIN_SIGNALS)
    min_er   = tm.get("pullback_min_expectancy_ratio", DEFAULT_MIN_EXP_RATIO)
    min_pos  = tm.get("pullback_min_positive_rate",    DEFAULT_MIN_POSITIVE_RATE)

    n = len(records)
    report = []
    gates  = {}

    # Gate 1 — signal count
    g1 = n >= min_sig
    gates["signal_count"] = g1
    report.append(f"  Gate 1 — shadow signals:  {n} / {min_sig} required  {'✓' if g1 else '✗'}")

    if n == 0:
        report.append("  (No [7.3] lines found — is pullback_completion_enabled still false?)")
        return False, report, gates

    cpbs       = [r[0] for r in records]
    raw_quals  = [r[1] for r in records]
    adj_quals  = [r[2] for r in records]

    mean_raw = sum(raw_quals) / n
    mean_adj = sum(adj_quals) / n
    exp_ratio = mean_adj / mean_raw if mean_raw > 0 else 0.0
    g2 = exp_ratio >= min_er
    gates["expectancy_ratio"] = g2
    report.append(
        f"  Gate 2 — expectancy ratio: {exp_ratio:.3f} / {min_er} required  "
        f"(mean raw={mean_raw:.3f}, adj={mean_adj:.3f})  {'✓' if g2 else '✗'}"
    )

    positive_n = sum(1 for c in cpbs if c >= 0.50)
    pos_rate   = positive_n / n
    g3 = pos_rate >= min_pos
    gates["positive_rate"] = g3
    report.append(
        f"  Gate 3 — C_pb>=0.50 rate:  {pos_rate:.1%} ({positive_n}/{n}) / "
        f"{min_pos:.0%} required  {'✓' if g3 else '✗'}"
    )

    mean_cpb = sum(cpbs) / n
    report.append(f"\n  Stats:  mean C_pb={mean_cpb:.3f}  n={n}")

    passed = all(gates.values())
    return passed, report, gates


def main():
    parser = argparse.ArgumentParser(description="S7.3 pullback completion auto-promoter")
    parser.add_argument("--promote", action="store_true", help="Write to config.json if gates pass")
    parser.add_argument("--force",   action="store_true", help="Force-flip to true regardless of gates")
    parser.add_argument("--reset",   action="store_true", help="Flip back to false")
    args = parser.parse_args()

    cfg     = load_config()
    current = cfg.get("phase_config", {}).get("pullback_completion_enabled", False)

    print(f"[PROMOTE] pullback_completion_enabled currently: {current}")
    print(f"[PROMOTE] Timestamp: {datetime.now(timezone.utc).isoformat()}\n")

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        cfg.setdefault("phase_config", {})["pullback_completion_enabled"] = False
        save_config(cfg)
        print("[PROMOTE] Reset to false.")
        return

    # ── Force ─────────────────────────────────────────────────────────────────
    if args.force:
        cfg.setdefault("phase_config", {})["pullback_completion_enabled"] = True
        save_config(cfg)
        print("[PROMOTE] Force-promoted to true (gates bypassed).")
        return

    # ── Shadow evaluation ─────────────────────────────────────────────────────
    lines   = read_log_lines()
    records = parse_shadow_signals(lines)
    passed, report, gates = evaluate_gates(records, cfg)

    print("Gate evaluation:")
    for line in report:
        print(line)

    print(f"\n  Overall: {'ALL GATES PASS ✓' if passed else 'GATES NOT MET ✗'}")

    if not passed:
        print("\n[PROMOTE] Not promoting. Re-run with --promote when gates clear.")
        sys.exit(0)

    # Gates passed
    if args.promote:
        if current:
            print("\n[PROMOTE] Already enabled — nothing to do.")
        else:
            cfg.setdefault("phase_config", {})["pullback_completion_enabled"] = True
            save_config(cfg)
            print("\n[PROMOTE] ✓ Promoted! pullback_completion_enabled = true")
    else:
        print("\n[PROMOTE] Dry run — pass --promote to write config.json.")


if __name__ == "__main__":
    main()
