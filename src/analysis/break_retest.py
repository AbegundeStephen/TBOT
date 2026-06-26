"""
Break and Retest Validation Engine
Validates if a structural Break and Retest sequence has occurred.

Primary mode: Uses Livermore anchors from composite_state as structural reference.
    - NATURAL_RETRACEMENT long:  anchor = livermore_anchor_natural_low
    - NATURAL_REBOUND short:     anchor = livermore_anchor_natural_high
    Level is set by the Livermore state machine using ATR multiples — not
    rolling pivot detection. Independent of _update_structure pivot quality.

Fallback mode: When composite_state not provided, uses 50-bar high/low
    as structural reference (legacy behaviour, kept for backward compatibility).
"""

import pandas as pd
import numpy as np
import talib as ta
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreakRetestResult:
    is_valid: bool
    score: float        # 0.0 to 1.0
    type: str           # "BULLISH_RETEST", "BEARISH_RETEST", "NONE"
    level: float
    explanation: str


class BreakRetestValidator:
    """
    Validates Break and Retest sequences against structural levels.
    When composite_state is provided, uses Livermore anchors (preferred).
    Falls back to 50-bar high/low when composite_state is absent.
    """

    def __init__(self, lookback: int = 50, retest_threshold_atr: float = 0.5):
        self.lookback = lookback
        self.retest_threshold_atr = retest_threshold_atr

    def validate(
        self,
        df: pd.DataFrame,
        composite_state_or_asset=None,   # accepts composite_state OR legacy asset string
        composite_state=None,            # explicit kwarg for new callers
    ) -> BreakRetestResult:
        """
        Validate a Break and Retest sequence.

        New callers:  validator.validate(df, composite_state=state)
        Legacy callers: validator.validate(df, asset_string)  — unchanged behaviour
        """
        # Resolve composite_state from either positional or keyword argument
        _cs = composite_state
        if _cs is None and not isinstance(composite_state_or_asset, str):
            _cs = composite_state_or_asset

        # Route to appropriate validation method
        if _cs is not None:
            return self._validate_with_anchors(df, _cs)
        else:
            return self._validate_legacy(df)

    # ──────────────────────────────────────────────────────────────────────
    # PRIMARY: Livermore anchor-based validation
    # ──────────────────────────────────────────────────────────────────────

    def _validate_with_anchors(self, df: pd.DataFrame, state) -> BreakRetestResult:
        """
        Validate using Livermore anchors as structural reference.

        The anchor is a price level the Livermore state machine has confirmed
        using ATR multiples — not rolling swing detection. This makes the
        check independent of _update_structure pivot calibration.

        Sequence for BULLISH_RETEST (NATURAL_RETRACEMENT → long):
          1. Price wicks below livermore_anchor_natural_low in last 8 bars
             (the spring — market tested the proven structural low)
          2. Current bar closes back above the anchor
             (confirmation — sellers rejected, buyers stepped in)

        Sequence for BEARISH_RETEST (NATURAL_REBOUND → short):
          1. Price wicks above livermore_anchor_natural_high in last 8 bars
             (the upthrust — market tested the proven structural high)
          2. Current bar closes back below the anchor
             (confirmation — buyers rejected, sellers stepped in)
        """
        if len(df) < 5:
            return BreakRetestResult(False, 0.0, "NONE", 0.0, "Insufficient data")

        lsm_state  = getattr(state, "livermore_state_1h", None)   # correct field name
        close_now  = float(df["close"].iloc[-1])
        atr        = self._get_atr(df)

        # ── BULLISH RETEST ─────────────────────────────────────────────────
        # Used by Mode 1 (NATURAL_RETRACEMENT long spring)
        if lsm_state in ("NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT"):
            anchor = getattr(state, "livermore_anchor_natural_low", None)

            if anchor is None or anchor <= 0:
                # Anchor not confirmed yet — state machine hasn't locked the low
                # Fall back to legacy check rather than block the trade
                logger.debug(
                    "[BRV] %s: nl_anchor not yet confirmed — using legacy check",
                    lsm_state,
                )
                return self._validate_legacy(df)

            # Step 1: Did price wick below the anchor in the last 8 bars?
            recent_lows = df["low"].values[-8:]
            swept_below = any(l < anchor for l in recent_lows)

            # Step 2: Did price close back above it?
            closed_above = close_now > anchor

            if swept_below and closed_above:
                dist = abs(close_now - anchor)
                score = max(0.7, 1.0 - (dist / max(atr, 1e-10)) * 0.3)
                logger.info(
                    "[BRV] BULLISH_RETEST confirmed: swept below anchor %.5g, "
                    "close=%.5g back above — spring at Livermore natural low",
                    anchor, close_now,
                )
                return BreakRetestResult(
                    True, round(score, 2), "BULLISH_RETEST", anchor,
                    f"Spring at Livermore anchor {anchor:.5g}: "
                    f"wick swept below, close={close_now:.5g} confirmed above",
                )

        # ── BEARISH RETEST ─────────────────────────────────────────────────
        # Used for NATURAL_REBOUND short setups
        elif lsm_state in ("NATURAL_REBOUND", "SECONDARY_REBOUND"):
            anchor = getattr(state, "livermore_anchor_natural_high", None)

            if anchor is None or anchor <= 0:
                logger.debug(
                    "[BRV] %s: nh_anchor not yet confirmed — using legacy check",
                    lsm_state,
                )
                return self._validate_legacy(df)

            # Step 1: Did price wick above the anchor in the last 8 bars?
            recent_highs = df["high"].values[-8:]
            swept_above = any(h > anchor for h in recent_highs)

            # Step 2: Did price close back below it?
            closed_below = close_now < anchor

            if swept_above and closed_below:
                dist = abs(close_now - anchor)
                score = max(0.7, 1.0 - (dist / max(atr, 1e-10)) * 0.3)
                logger.info(
                    "[BRV] BEARISH_RETEST confirmed: swept above anchor %.5g, "
                    "close=%.5g back below — upthrust at Livermore natural high",
                    anchor, close_now,
                )
                return BreakRetestResult(
                    True, round(score, 2), "BEARISH_RETEST", anchor,
                    f"Upthrust at Livermore anchor {anchor:.5g}: "
                    f"wick swept above, close={close_now:.5g} confirmed below",
                )

        # ── MAIN state exhaustion (Mode 3 — not yet active, prepared) ──────
        # Kept here for when Mode 3 is enabled. Not reached in Phase 1.
        elif lsm_state == "MAIN_UP":
            anchor = getattr(state, "livermore_anchor_main_up_max", None)
            if anchor and close_now < anchor:
                recent_highs = df["high"].values[-8:]
                if any(h > anchor for h in recent_highs):
                    return BreakRetestResult(
                        True, 0.8, "BEARISH_RETEST", anchor,
                        f"Exhaustion upthrust above MAIN_UP max {anchor:.5g}",
                    )

        elif lsm_state == "MAIN_DOWN":
            anchor = getattr(state, "livermore_anchor_main_down_min", None)
            if anchor and close_now > anchor:
                recent_lows = df["low"].values[-8:]
                if any(l < anchor for l in recent_lows):
                    return BreakRetestResult(
                        True, 0.8, "BULLISH_RETEST", anchor,
                        f"Exhaustion spring below MAIN_DOWN min {anchor:.5g}",
                    )

        return BreakRetestResult(False, 0.0, "NONE", 0.0, "No anchor retest sequence detected")

    # ──────────────────────────────────────────────────────────────────────
    # FALLBACK: Legacy 50-bar high/low validation
    # ──────────────────────────────────────────────────────────────────────

    def _validate_legacy(self, df: pd.DataFrame) -> BreakRetestResult:
        """Original 50-bar max/min logic. Runs when composite_state not available."""
        if len(df) < self.lookback:
            return BreakRetestResult(False, 0.0, "NONE", 0.0, "Insufficient data")

        try:
            struct_df = df.iloc[-self.lookback:-5]
            high_ref  = struct_df["high"].max()
            low_ref   = struct_df["low"].min()
            latest    = df.iloc[-1]
            prev_bars = df.iloc[-5:-1]

            highs = df["high"].values
            lows  = df["low"].values
            closes= df["close"].values
            atr   = ta.ATR(highs, lows, closes, timeperiod=14)[-1]
            zone  = self.retest_threshold_atr * atr

            if any(prev_bars["high"] > high_ref):
                lowest_since = df["low"].iloc[-3:].min()
                if abs(lowest_since - high_ref) <= zone and latest["close"] > high_ref:
                    return BreakRetestResult(
                        True, 0.9, "BULLISH_RETEST", high_ref,
                        f"Bullish B&R: retested {high_ref:.5g} and held",
                    )

            if any(prev_bars["low"] < low_ref):
                highest_since = df["high"].iloc[-3:].max()
                if abs(highest_since - low_ref) <= zone and latest["close"] < low_ref:
                    return BreakRetestResult(
                        True, 0.9, "BEARISH_RETEST", low_ref,
                        f"Bearish B&R: retested {low_ref:.5g} and held",
                    )

            return BreakRetestResult(False, 0.0, "NONE", 0.0, "No B&R setup detected")

        except Exception as e:
            logger.error(f"[BRV] Legacy validation error: {e}")
            return BreakRetestResult(False, 0.0, "NONE", 0.0, f"Error: {e}")

    def _get_atr(self, df: pd.DataFrame) -> float:
        """ATR from precomputed column or simple TR fallback."""
        if "atr" in df.columns:
            v = float(df["atr"].iloc[-1])
            if not np.isnan(v) and v > 0:
                return v
        n = min(14, len(df) - 1)
        if n >= 1:
            h = df["high"].iloc[-(n+1):-1].values
            l = df["low"].iloc[-(n+1):-1].values
            c = df["close"].iloc[-(n+2):-2].values
            tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
            return float(np.mean(tr)) if len(tr) > 0 else 1e-10
        return max(float(df["high"].iloc[-1] - df["low"].iloc[-1]), 1e-10)