#!/usr/bin/env python3
"""
calibrate_livermore_atr.py — Livermore State Machine ATR Calibration

Grid-searches major_mult / minor_mult / dual_confirm for each asset using
4H historical data. Scores each combination on:
  1. Transition frequency  (target: 1-2 state changes per week)
  2. State duration balance (MAIN states should be longer than NATURAL states)
  3. Directional alignment  (MAIN_UP during up-moves, MAIN_DOWN during down-moves)

Usage:
  python scripts/calibrate_livermore_atr.py

Output:
  Prints a JSON block you can paste directly into
  config/aggregator_presets.json > LIVERMORE_PIVOTS
  (BTC entry is kept as-is — already calibrated.)
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from itertools import product
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.execution.livermore_state_machine import LivermoreStateMachine

logging.basicConfig(level=logging.WARNING)  # suppress LSM debug spam

# ── Asset → 4H data file mapping ────────────────────────────────────────────
ASSET_FILES: Dict[str, str] = {
    "GOLD":   "XAUUSDm_4h.csv",
    "EURUSD": "EURUSDm_4h.csv",
    "EURJPY": "EURJPYm_4h.csv",
    "USTEC":  "USTECm_4h.csv",
    "GBPAUD": "GBPAUDm_4h.csv",
    "GBPUSD": "GBPUSDm_4h.csv",
    "USOIL":  "USOILm_4h.csv",
}

DATA_DIR = ROOT / "data" / "raw"

# ── Grid search space ────────────────────────────────────────────────────────
MAJOR_MULTS  = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MINOR_MULTS  = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DUAL_CONFIRMS = [1, 2, 3]

# ── Scoring targets ──────────────────────────────────────────────────────────
# FX / commodity 4H: 1 week ≈ 30 bars (5 days × 6 bars/day)
# Crypto 4H:         1 week ≈ 42 bars (7 days × 6 bars/day)
BARS_PER_WEEK_FX     = 30
BARS_PER_WEEK_CRYPTO = 42

TARGET_TRANSITIONS_PER_WEEK_MIN = 0.8   # below this = too slow / missing turns
TARGET_TRANSITIONS_PER_WEEK_MAX = 2.5   # above this = too noisy
TARGET_MAIN_FRAC_MIN = 0.40             # MAIN states should cover ≥40% of bars
TARGET_MAIN_FRAC_MAX = 0.80             # but not ≥80% (never gives pullbacks)


def load_4h(asset: str) -> Optional[pd.DataFrame]:
    fname = ASSET_FILES.get(asset)
    if not fname:
        return None
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  [WARN] {asset}: file not found at {path}")
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        # ensure close column
        if "close" not in df.columns:
            close_candidates = [c for c in df.columns if "close" in c]
            if close_candidates:
                df = df.rename(columns={close_candidates[0]: "close"})
            else:
                print(f"  [WARN] {asset}: no close column found")
                return None
        df = df.dropna(subset=["close"])
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  [WARN] {asset}: load error — {e}")
        return None


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR14 from OHLCV. Falls back to high-low if H/L present, else uses close pct."""
    if "high" in df.columns and "low" in df.columns:
        high = df["high"]
        low  = df["low"]
        close_prev = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low  - close_prev).abs(),
        ], axis=1).max(axis=1)
    else:
        # fallback: approximate ATR from close range
        tr = df["close"].pct_change().abs() * df["close"]

    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def run_lsm(df: pd.DataFrame, major_mult: float, minor_mult: float,
            dual_confirm: int, atr_period: int = 14) -> pd.DataFrame:
    """Run LSM over the full dataframe, return a series of states."""
    atr = compute_atr(df, period=atr_period)
    lsm = LivermoreStateMachine(
        asset="CAL", timeframe="4H",
        major_mult=major_mult, minor_mult=minor_mult,
        dual_confirm=dual_confirm, atr_period=atr_period,
    )

    states = []
    for i in range(len(df)):
        c   = float(df["close"].iloc[i])
        a   = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
        snap = lsm.update(close=c, atr=a)
        states.append(snap.state)

    df = df.copy()
    df["lsm_state"] = states
    return df


def score_combo(df: pd.DataFrame, bars_per_week: int) -> Tuple[float, dict]:
    """
    Score a calibration run. Returns (score, metrics_dict).
    Higher score = better parameters.
    """
    states = df["lsm_state"]
    n      = len(states)

    if n < 50:
        return -999.0, {}

    # 1. Transition count
    transitions = (states != states.shift(1)).sum() - 1  # -1 for first bar
    trans_per_week = transitions / (n / bars_per_week)

    # 2. Fraction of time in MAIN states
    main_states = {"MAIN_UP", "MAIN_DOWN"}
    main_frac = states.isin(main_states).mean()

    # 3. Directional alignment: when in MAIN_UP, is price making progress upward?
    #    Compute 10-bar forward return for each MAIN_UP bar
    close = df["close"]
    fwd10 = close.shift(-10) / close - 1  # 10-bar forward return

    main_up_mask   = states == "MAIN_UP"
    main_down_mask = states == "MAIN_DOWN"

    align_up   = fwd10[main_up_mask].mean()   if main_up_mask.any()   else 0.0
    align_down = -fwd10[main_down_mask].mean() if main_down_mask.any() else 0.0
    # positive align_up = MAIN_UP bars are followed by upward price action
    # positive align_down = MAIN_DOWN bars are followed by downward price action
    directional_score = (align_up + align_down) / 2.0

    # 4. Average MAIN state duration (longer = more stable)
    main_dur = []
    curr_dur = 0
    in_main  = False
    for s in states:
        if s in main_states:
            curr_dur += 1
            in_main   = True
        else:
            if in_main and curr_dur > 0:
                main_dur.append(curr_dur)
            curr_dur = 0
            in_main  = False
    avg_main_dur = np.mean(main_dur) if main_dur else 0.0

    # ── Penalty / bonus functions ─────────────────────────────────────────
    # Transition frequency: penalise outside [0.8, 2.5] per week
    if trans_per_week < TARGET_TRANSITIONS_PER_WEEK_MIN:
        freq_score = -2.0 * (TARGET_TRANSITIONS_PER_WEEK_MIN - trans_per_week)
    elif trans_per_week > TARGET_TRANSITIONS_PER_WEEK_MAX:
        freq_score = -1.5 * (trans_per_week - TARGET_TRANSITIONS_PER_WEEK_MAX)
    else:
        # Gaussian peak at 1.5 transitions/week
        freq_score = 1.0 - 0.5 * ((trans_per_week - 1.5) / 0.5) ** 2
        freq_score = max(0.0, freq_score)

    # MAIN fraction: penalise outside [40%, 80%]
    if main_frac < TARGET_MAIN_FRAC_MIN:
        frac_score = -2.0 * (TARGET_MAIN_FRAC_MIN - main_frac)
    elif main_frac > TARGET_MAIN_FRAC_MAX:
        frac_score = -1.5 * (main_frac - TARGET_MAIN_FRAC_MAX)
    else:
        frac_score = 0.5  # flat bonus for being in range

    # Directional alignment: 0.5 weight
    dir_score = min(directional_score * 100, 2.0)  # cap at 2.0

    # Avg main duration bonus (longer is better, up to 40 bars)
    dur_score = min(avg_main_dur / 40.0, 1.0)

    total = freq_score + frac_score + dir_score + dur_score

    metrics = {
        "transitions_per_week": round(trans_per_week, 2),
        "main_fraction":        round(float(main_frac), 3),
        "align_up":             round(float(align_up) * 100, 3),
        "align_down":           round(float(align_down) * 100, 3),
        "avg_main_dur_bars":    round(float(avg_main_dur), 1),
        "score":                round(total, 4),
    }
    return total, metrics


def calibrate_asset(asset: str) -> Optional[dict]:
    print(f"\n{'='*60}")
    print(f"  Calibrating {asset}")
    print(f"{'='*60}")

    df = load_4h(asset)
    if df is None or len(df) < 100:
        print(f"  SKIP: insufficient data")
        return None

    bars_per_week = BARS_PER_WEEK_FX  # FX / commodity default (override BTC below)
    print(f"  Bars: {len(df)} | {df.index[0].date()} → {df.index[-1].date()}")

    best_score  = -999.0
    best_params = {}
    best_metrics= {}

    combos = list(product(MAJOR_MULTS, MINOR_MULTS, DUAL_CONFIRMS))
    print(f"  Grid: {len(combos)} combinations")

    for major_mult, minor_mult, dual_confirm in combos:
        # minor must be < major (otherwise silent zone never fires)
        if minor_mult >= major_mult:
            continue
        try:
            result_df = run_lsm(df, major_mult, minor_mult, dual_confirm)
            score, metrics = score_combo(result_df, bars_per_week)
        except Exception as e:
            continue

        if score > best_score:
            best_score  = score
            best_params = {
                "major_mult":  major_mult,
                "minor_mult":  minor_mult,
                "dual_confirm": dual_confirm,
                "atr_period":  14,
            }
            best_metrics = metrics

    if not best_params:
        print(f"  WARN: no valid combination found, using defaults")
        return {"major_mult": 3.5, "minor_mult": 1.0, "dual_confirm": 2, "atr_period": 14}

    print(f"  Best: major={best_params['major_mult']} minor={best_params['minor_mult']} "
          f"dual_confirm={best_params['dual_confirm']}")
    print(f"  Metrics:")
    for k, v in best_metrics.items():
        print(f"    {k}: {v}")

    return best_params


def main():
    print("Livermore ATR Calibration")
    print("=" * 60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Grid: major={MAJOR_MULTS} minor={MINOR_MULTS} dual={DUAL_CONFIRMS}")

    results: Dict[str, dict] = {}

    # BTC — already calibrated, skip but keep in output
    results["BTC"] = {
        "major_mult":   3.5,
        "minor_mult":   1.0,
        "dual_confirm": 2,
        "atr_period":   14,
        "_calibration": "BTC 4H Jan2024–Feb2026: 1.48 transitions/week, 5/6 turning-point score (pre-existing)",
    }

    for asset in ASSET_FILES:
        params = calibrate_asset(asset)
        if params:
            n_bars = 0
            df = load_4h(asset)
            if df is not None:
                n_bars = len(df)
                start  = str(df.index[0].date())
                end    = str(df.index[-1].date())
            else:
                start = end = "unknown"
            params["_calibration"] = (
                f"{asset} 4H {start}–{end}: "
                f"calibrated from {n_bars} bars | "
                f"major={params['major_mult']} minor={params['minor_mult']} "
                f"dual_confirm={params['dual_confirm']}"
            )
            results[asset] = params

    # ── Print final JSON block ────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("RESULT — paste into config/aggregator_presets.json > LIVERMORE_PIVOTS")
    print("=" * 60)
    print(json.dumps({"LIVERMORE_PIVOTS": results}, indent=2))


if __name__ == "__main__":
    main()
