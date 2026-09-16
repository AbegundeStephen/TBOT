"""
RL-1: the first propose-only bandit (BATCH 3, rung 1 of the ladder).

Reads REPLAYER-2's arm re-simulation directly (imports replayer.py's own
machinery -- same process, same data, no subprocess/JSON round-trip).
Hard threshold gate; proposes only past it, and only when the best arm's
90% CI lower bound clears the current setting's mean. Never writes config,
never imports main.py or anything live -- this script only reads
logs/episodes/*.jsonl and appends to logs/rl1_proposals.jsonl.

Desire rules on any proposal; Stephen edits config.json by hand; the edit
commit message carries the proposal_id so the diary can attribute later rows.

Run:  python tools/bandit.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replayer import (  # noqa: E402
    load_closed_trades, _ARM_VALUES, _replay_row_arm, _bootstrap_ci,
    _RUNNER_TRAIL_DEFAULT, BE_TRIGGER_R,
)

HEADER = ("Replay path = 1H bars; results are directional, not precise "
          "(1H sims inflate). 15m path pending.")

# The setting each knob's arm list is measured against -- current live
# config values, not arbitrary arm-list entries. trail_mult/be_r match the
# floor of their own arm lists exactly (0.8 = runner_atr_mult_flat,
# 0.75 = phase_config.r_breakeven_trigger); grade_table's baseline is its
# own "current" arm by construction.
_CURRENT_SETTING = {
    "trail_mult": _RUNNER_TRAIL_DEFAULT,
    "be_r": BE_TRIGGER_R,
    "grade_table": "current",
}

_MIN_CLOSED, _MIN_TRAILING, _MIN_ASSETS = 30, 10, 3
_PROPOSALS_FILE = "logs/rl1_proposals.jsonl"


def _ensure_header_file():
    """Doc's own check: the file exists with a header only, even before the
    threshold is ever met -- so today's run leaves real, inspectable state."""
    p = Path(_PROPOSALS_FILE)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "header": HEADER,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")


def _arm_tables_by_asset(knob, rows):
    by_asset_arm = defaultdict(lambda: defaultdict(list))
    for e in rows:
        for value in _ARM_VALUES[knob]:
            r = _replay_row_arm(e, knob, value)
            if r is not None:
                by_asset_arm[e.get("asset")][value].append(r)
    tables = {}
    for asset, arms in by_asset_arm.items():
        tables[asset] = {}
        for value, vals in arms.items():
            mean, lo, hi = _bootstrap_ci(vals)
            tables[asset][value] = {"n": len(vals), "mean_r": mean,
                                     "ci90_lo": lo, "ci90_hi": hi}
    return tables


def propose(knob, rows):
    """One proposal per asset, only when the best OTHER arm's own sample
    clears the floor and its CI lower bound beats the current setting's
    mean -- both conditions hard, per the doc."""
    tables = _arm_tables_by_asset(knob, rows)
    current_value = _CURRENT_SETTING[knob]
    proposals = []
    for asset, arms in tables.items():
        current = arms.get(current_value)
        if not current or current["n"] < _MIN_CLOSED or current["mean_r"] is None:
            continue
        candidates = [
            (v, s) for v, s in arms.items()
            if v != current_value and s["n"] >= _MIN_CLOSED and s["mean_r"] is not None
        ]
        if not candidates:
            continue
        best_value, best = max(candidates, key=lambda vs: vs[1]["mean_r"])
        if best["ci90_lo"] is None or best["ci90_lo"] <= current["mean_r"]:
            continue
        proposal_id = f"{knob}_{asset}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        delta = best["mean_r"] - current["mean_r"]
        print(f"[RL-1] {asset} {knob}: {current_value} -> {best_value} "
              f"({delta:+.2f}R/trade, n={best['n']}, "
              f"CI {best['ci90_lo']:.2f}..{best['ci90_hi']:.2f}, proposal_id={proposal_id})")
        proposals.append({
            "proposal_id": proposal_id,
            "knob": knob,
            "asset": asset,
            "current_value": current_value,
            "proposed_value": best_value,
            "delta_r_per_trade": delta,
            "n": best["n"],
            "ci90_lo": best["ci90_lo"],
            "ci90_hi": best["ci90_hi"],
            "arm_table": arms,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    return proposals


def main():
    print(HEADER)
    _ensure_header_file()

    rows = load_closed_trades(include_all=False)
    n_total = len(rows)
    n_trailing = sum(1 for r in rows if r.get("exit_via") == "trail")
    n_assets = len(set(r.get("asset") for r in rows))

    if n_total < _MIN_CLOSED or n_trailing < _MIN_TRAILING or n_assets < _MIN_ASSETS:
        print(f"below threshold (n={n_total}) -- need >= {_MIN_CLOSED} closed, "
              f">= {_MIN_TRAILING} trailing exits, >= {_MIN_ASSETS} assets")
        return

    all_proposals = []
    for knob in _ARM_VALUES:
        all_proposals.extend(propose(knob, rows))

    if not all_proposals:
        print("above threshold, but no arm's CI clears its current setting this run.")
        return

    with open(_PROPOSALS_FILE, "a", encoding="utf-8") as f:
        for p in all_proposals:
            f.write(json.dumps(p, default=str) + "\n")


if __name__ == "__main__":
    main()
