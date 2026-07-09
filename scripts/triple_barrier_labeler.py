"""
Triple Barrier Labeler — Tier 6.1 (Brain Rebuild Part 1.11)
=============================================================
Standalone, offline research script. Scans historical 4H OHLCV for BOS/CHoCH
events (using the same bounded-lookback, close-based, symmetric detection
fix applied to the live pipeline in signal_aggregator.py — Brain Rebuild
Part 0.5) and labels each event win/loss/timeout via a triple-barrier test
(ATR-scaled target and stop, N-bar timeout). Prints per-symbol, per-event
win rates.

Not wired into the live trading pipeline or any training job — run manually
to sanity-check whether BOS/CHoCH actually precedes a profitable move before
trusting judges that key off it. Sequencing note: once shadow_data carries
real judge-level context (Brain Rebuild Part 1.6), that becomes the primary
source for grading real blocked signals — this script is a historical
cross-check, not a replacement for it.

Usage: python scripts/triple_barrier_labeler.py
Expects: data/raw/*_4h.csv  (columns: open, high, low, close, ...)
"""
import glob
import os

import numpy as np
import pandas as pd

TARGET_ATR, STOP_ATR, TIMEOUT_BARS = 2.0, 1.0, 20
MAX_LOOKBACK_BARS = 50


def detect_bos_choch_at_bar(highs, lows, closes, atrs, i, min_depth_mult=0.3):
    if i < 7 or i >= len(highs) - 2:
        return False, False
    min_depth = min_depth_mult * atrs[i] if atrs[i] > 0 else 0.0
    floor = max(6, i - MAX_LOOKBACK_BARS)
    swing_highs, swing_lows = [], []
    for j in range(i - 2, floor, -1):
        if (
            highs[j] > highs[j - 1] and highs[j] > highs[j - 2] and highs[j] > highs[j - 3]
            and highs[j] > highs[j - 4] and highs[j] > highs[j + 1] and highs[j] > highs[j + 2]
            and (highs[j] - max(closes[j - 4:j]) >= min_depth)
        ):
            swing_highs.append(closes[j])
            if len(swing_highs) >= 2:
                break
    for j in range(i - 2, floor, -1):
        if (
            lows[j] < lows[j - 1] and lows[j] < lows[j - 2] and lows[j] < lows[j - 3]
            and lows[j] < lows[j - 4] and lows[j] < lows[j + 1] and lows[j] < lows[j + 2]
            and (min(closes[j - 4:j]) - lows[j] >= min_depth)
        ):
            swing_lows.append(closes[j])
            if len(swing_lows) >= 2:
                break
    bos = choch = False
    if len(swing_highs) >= 2:
        bos = swing_highs[0] > swing_highs[1] or bos
        choch = swing_highs[0] < swing_highs[1] or choch
    if len(swing_lows) >= 2:
        bos = swing_lows[0] < swing_lows[1] or bos
        choch = swing_lows[0] > swing_lows[1] or choch
    return bos, choch


def label_events(df):
    df = df.reset_index(drop=True)
    atr = df["high"].sub(df["low"]).rolling(14).mean().values
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    results = []
    for i in range(60, len(df) - TIMEOUT_BARS):
        bos, choch = detect_bos_choch_at_bar(highs, lows, closes, atr, i)
        if not (bos or choch):
            continue
        entry = closes[i]
        target, stop = entry + TARGET_ATR * atr[i], entry - STOP_ATR * atr[i]
        window_hi = df["high"].iloc[i + 1:i + 1 + TIMEOUT_BARS]
        window_lo = df["low"].iloc[i + 1:i + 1 + TIMEOUT_BARS]
        hit_t = window_hi.index[window_hi >= target]
        hit_s = window_lo.index[window_lo <= stop]
        if len(hit_t) and (not len(hit_s) or hit_t[0] < hit_s[0]):
            outcome = "win"
        elif len(hit_s):
            outcome = "loss"
        else:
            outcome = "timeout"
        results.append({"bar": i, "event": "BOS" if bos else "CHoCH", "outcome": outcome})
    return pd.DataFrame(results)


if __name__ == "__main__":
    for path in glob.glob("data/raw/*_4h.csv"):
        symbol = os.path.basename(path).split("_")[0]
        out = label_events(pd.read_csv(path))
        for ev in ("BOS", "CHoCH"):
            sub = out[out["event"] == ev]
            if len(sub):
                wr = (sub["outcome"] == "win").mean()
                print(f"{symbol} {ev}: n={len(sub)} win_rate={wr:.1%}")
