"""
5-Tier Retest Engine — Phase 3B
Classifies the entry context relative to nearby structural levels.
Returns a RetestResult with a score-modifier delta and entry_type for VTM.

Priority cascade (evaluated top-to-bottom; first match wins):
  1. CLEAN        (−0.20)  — price at a defended level; textbook pullback entry
  1b. CHOCH_HOLD  (−0.30)  — CLEAN, plus a CHoCH just fired at that level
  2. BREAKOUT     (+0.10 / +0.20 / +0.40) — fresh Livermore state (age ≤ 5 bars)
  2B. CONTINUATION (+0.10 / +0.20) — Phase 4 ext, gated by phase_config.continuation_targets_enabled
                  (default OFF). Tight 1H consolidation breaking out inside an
                  already-established 4H MAIN leg (age_1h > breakout_age_max).
  3. WICK         (  0.00) — sweep + close recovery through level (spring entry)
  4. CHASE_HARD   (+1.50)  — price too extended; entry_type = REJECT
  5. CHASE_SOFT   (+0.75)  — moderately extended; elevated threshold
  6. NO_LEVEL_NEARBY (+0.35 / +0.40) — fallback when no structural reference

All numeric thresholds are loaded from config/aggregator_presets.json
RETEST_ENGINE section — zero magic numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── entry_type string constants ───────────────────────────────────────────
ET_MR_PULLBACK     = "MR_PULLBACK"
ET_TREND_FOLLOWING = "TREND_FOLLOWING"
ET_SPRING_ENTRY    = "SPRING_ENTRY"
ET_RANGE_BOUNDARY  = "RANGE_BOUNDARY"
ET_CONTINUATION    = "CONTINUATION"
ET_REJECT          = "REJECT"

# ── retest_type string constants ─────────────────────────────────────────
RT_CLEAN           = "CLEAN"
RT_CHOCH_HOLD      = "CHOCH_HOLD"
RT_BREAKOUT        = "BREAKOUT"
RT_CONTINUATION    = "CONTINUATION"
RT_WICK            = "WICK"
RT_CHASE_SOFT      = "CHASE_SOFT"
RT_CHASE_HARD      = "CHASE_HARD"
RT_NO_LEVEL_NEARBY = "NO_LEVEL_NEARBY"

# Livermore states that confirm a LONG direction on 4H
_LONG_CONFIRMING_4H_STATES  = frozenset({
    "MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT"
})
# Livermore states that confirm a SHORT direction on 4H
_SHORT_CONFIRMING_4H_STATES = frozenset({
    "MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND"
})

# Symbols treated as BTC-class (crypto volatility profile)
_BTC_SYMBOLS = frozenset({"BTC", "BTCUSDT", "BTC/USDT", "BTCUSD"})

# Symbols with FX volatility profile
_FX_PREFIXES = ("EUR", "GBP", "AUD", "JPY", "USD", "CAD", "CHF", "NZD")


@dataclass
class RetestResult:
    """Output of RetestEngine.classify()."""
    retest_type: str            # one of the RT_* constants above
    modifier: float             # score threshold delta (negative = easier, positive = harder)
    entry_type: Optional[str]   # ET_* constant; None only on error paths
    direction: int              # +1 LONG / -1 SHORT (pass-through from caller)
    level: Optional[float]      # reference level used; None when NO_LEVEL_NEARBY
    # Phase 4 ext — gated by phase_config.continuation_targets_enabled (default OFF).
    # Dual semantics depending on entry_type:
    #   CLEAN (RANGE_BOUNDARY) / WICK (SPRING_ENTRY): the structural range
    #     top/bottom — the target-ladder "box" consumed by VTM to build the
    #     midpoint/top/measured-move ladder.
    #   CONTINUATION: the 1H consolidation box that was just broken — used
    #     only for stop placement (_compute_structural_stop), NOT the target
    #     ladder (that ladder is built in VTM directly from the Livermore
    #     NATURAL anchor + entry price).
    # None when the feature is off or no box was computed.
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    # Second and third nearest 4H structural levels. A4: also checked for
    # their own defended-wick status (level_2_defended/level_3_defended) as
    # a CLEAN fallback when the primary level doesn't qualify — otherwise
    # visible context only, does not alter BREAKOUT/CHASE classification.
    level_2: Optional[float] = None
    level_3: Optional[float] = None


class RetestEngine:
    """
    Classifies each candidate trade entry into one of the tiers based on
    proximity to structural levels, Livermore state age, and sweep detection.

    Instantiate once and call classify() per candle.
    Thread-safe (no mutable state after __init__).
    """

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg : dict
            The RETEST_ENGINE sub-dict from aggregator_presets.json.
        """
        self._cfg = cfg

        # ── top-level scalar thresholds ────────────────────────
        self._clean_atr_mult        = float(cfg.get("clean_proximity_atr_mult", 0.5))
        self._breakout_age_max      = int(cfg.get("breakout_age_max_bars", 5))

        # ── fixed modifiers ────────────────────────────────
        self._mod_clean             = float(cfg.get("modifier_clean",          -0.20))
        self._mod_wick              = float(cfg.get("modifier_wick",            0.00))
        self._mod_chase_soft        = float(cfg.get("modifier_chase_soft",      0.75))
        self._mod_chase_hard        = float(cfg.get("modifier_chase_hard",      1.50))
        self._mod_no_level_default  = float(cfg.get("modifier_no_level_default", 0.35))

        # ── breakout alignment modifiers ────────────────────────
        self._mod_breakout_aligned_btc  = float(cfg.get("modifier_breakout_aligned_btc",  0.10))
        self._mod_breakout_aligned_fx   = float(cfg.get("modifier_breakout_aligned_fx",   0.20))
        self._mod_breakout_misaligned   = float(cfg.get("modifier_breakout_misaligned",   0.40))

        # ── per-asset override sub-dicts ────────────────────────
        self._assets = cfg.get("assets", {})

        # ── Phase 4 ext: CONTINUATION tier (gated, default OFF) ──────────
        _cont_cfg = cfg.get("continuation", {})
        self._cont_min_bars                 = int(_cont_cfg.get("min_bars", 4))
        self._cont_max_bars                 = int(_cont_cfg.get("max_bars", 15))
        self._cont_max_range_atr_mult       = float(_cont_cfg.get("max_range_atr_mult", 2.0))
        self._cont_breakout_buffer_atr_mult = float(_cont_cfg.get("breakout_buffer_atr_mult", 0.1))
        self._cont_min_flagpole_atr_mult    = float(_cont_cfg.get("min_flagpole_atr_mult", 1.0))
        self._cont_mod_aligned_btc          = float(_cont_cfg.get("modifier_aligned_btc", 0.10))
        self._cont_mod_aligned_fx           = float(_cont_cfg.get("modifier_aligned_fx",  0.20))

        # ── Phase 4 ext: range-ladder lookback (gated, default OFF) ─────────
        _range_cfg = cfg.get("range_ladder", {})
        self._range_lookback_bars = int(_range_cfg.get("lookback_bars", 30))

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def classify(
        self,
        df: pd.DataFrame,
        state,          # CompositeState — typed loosely to avoid circular import
        symbol: str,
        direction: int, # +1 LONG / -1 SHORT
    ) -> RetestResult:
        """
        Run the priority cascade and return a RetestResult.

        Minimum requirements on `df`:
          - At least 2 rows of 1H OHLCV with an 'atr' column populated.
          - df.iloc[-1] is the last *closed* candle.

        Minimum requirements on `state`:
          - nearby_4h_level, level_defended
          - livermore_state_age_1h, livermore_state_1h, livermore_state_4h
          - livermore_anchor_* (used for BREAKOUT level lookup)
          - sweep_detected, sweep_level
        """
        if len(df) < 2:
            logger.debug("retest_engine: df too short (%d rows) — NO_LEVEL_NEARBY", len(df))
            return self._no_level(symbol, direction)

        atr = self._get_atr(df)
        if atr <= 0:
            logger.debug("retest_engine: ATR=0 — NO_LEVEL_NEARBY")
            return self._no_level(symbol, direction)

        close     = float(df["close"].iloc[-1])
        # Item 3.6: direction-correct level — the nearest support for a LONG
        # setup, nearest resistance for a SHORT setup, instead of whichever of
        # either type happened to be globally nearest (which could be the
        # wrong side of price for this direction). None (not a fallback to
        # the old direction-agnostic level) when no level exists on the
        # correct side — falling back would reintroduce the wrong-direction
        # level this item exists to fix.
        if direction == 1:
            level = getattr(state, "nearby_support_level", None)
            _level_tests = getattr(state, "nearby_support_level_tests", 0)
        else:
            level = getattr(state, "nearby_resistance_level", None)
            _level_tests = getattr(state, "nearby_resistance_level_tests", 0)
        level_2   = getattr(state, "nearby_4h_level_2", None)
        level_3   = getattr(state, "nearby_4h_level_3", None)
        asset_cfg = self._assets.get(symbol, self._assets.get("DEFAULT", {}))

        # ── 1. CLEAN / 1b. CHOCH_HOLD ────────────────────────────────
        # Price within clean_proximity_atr_mult × ATR of the nearby 4H level,
        # AND the level has been actively defended (level_defended = True).
        if level is not None:
            dist_atr = abs(close - level) / atr
            level_defended = getattr(state, "level_defended", False)
            if dist_atr <= self._clean_atr_mult and level_defended:
                # Part 1.10 (Brain Rebuild) — Tier 2.1: a defended clean level
                # where a CHoCH just fired is a stronger tell than a plain
                # clean retest (structure just flipped, not merely holding),
                # so it earns a slightly better modifier. Checked first since
                # its condition is a strict superset of plain CLEAN's — placed
                # after would make it unreachable (CLEAN always matches first).
                choch_now = getattr(state, "choch_detected", False)
                if choch_now:
                    logger.debug("retest_engine: CHOCH_HOLD @ %.5f (dist=%.2f ATR)", level, dist_atr)
                    _rh, _rl = self._maybe_swing_range(state, df)
                    return RetestResult(
                        retest_type=RT_CHOCH_HOLD,
                        modifier=self._mod_clean - 0.1,
                        entry_type=ET_RANGE_BOUNDARY,
                        direction=direction,
                        level=level,
                        range_high=_rh,
                        range_low=_rl,
                        level_2=level_2,
                        level_3=level_3,
                    )
                logger.debug("retest_engine: CLEAN @ %.5f (dist=%.2f ATR)", level, dist_atr)
                _rh, _rl = self._maybe_swing_range(state, df)
                # A level tested multiple times is more proven than a fresh one —
                # deepen the CLEAN discount (more negative = easier) rather than
                # scoring a 4x-tested level identically to a first touch.
                # Item 3.6: use the direction-specific test count (_level_tests)
                # instead of the old shared level_test_count.
                _proven_bonus = min(_level_tests * 0.05, 0.20)
                return RetestResult(
                    retest_type=RT_CLEAN,
                    modifier=self._mod_clean - _proven_bonus,
                    entry_type=ET_RANGE_BOUNDARY,
                    direction=direction,
                    level=level,
                    range_high=_rh,
                    range_low=_rl,
                    level_2=level_2,
                    level_3=level_3,
                )

        # ── 1c. CLEAN off a secondary/tertiary level (A4) ─────────────
        # level_defended was only ever computed for the single primary
        # nearby_4h_level. A strong rejection wick can defend the 2nd/3rd
        # nearest level instead (e.g. price sits between two levels and the
        # "nearest" one isn't where the actual defense happened) — check
        # those independently before falling through to BREAKOUT/WICK/CHASE.
        for _lvl, _defended in (
            (level_2, bool(getattr(state, "level_2_defended", False))),
            (level_3, bool(getattr(state, "level_3_defended", False))),
        ):
            if _lvl is None or not _defended:
                continue
            _dist_atr_alt = abs(close - _lvl) / atr
            if _dist_atr_alt <= self._clean_atr_mult:
                logger.debug(
                    "retest_engine: CLEAN (secondary level) @ %.5f (dist=%.2f ATR)",
                    _lvl, _dist_atr_alt,
                )
                _rh, _rl = self._maybe_swing_range(state, df)
                return RetestResult(
                    retest_type=RT_CLEAN,
                    modifier=self._mod_clean,
                    entry_type=ET_RANGE_BOUNDARY,
                    direction=direction,
                    level=_lvl,
                    range_high=_rh,
                    range_low=_rl,
                    level_2=level_2,
                    level_3=level_3,
                )

        # ── 2. BREAKOUT ──────────────────────────────────────────────
        # Livermore 1H state flipped; price still within
        # breakout_proximity_atr_mult × ATR of the anchor that was broken.
        #
        # Brain Rebuild Part 3.1 (applied verbatim per explicit instruction,
        # 2026-07-09): age_1h no longer gates whether BREAKOUT fires at all —
        # it only adjusts the modifier (+0.15 when stale, age_1h >
        # breakout_age_max). Doc's own open flag ("for Stephen"): this makes
        # 2B. CONTINUATION's stale-case largely unreachable, since BREAKOUT
        # (checked first) now also handles stale ages, just at a higher bar.
        # Left unresolved per the doc — low risk today since CONTINUATION is
        # gated behind phase_config.continuation_targets_enabled, off in the
        # live config.
        age_1h = int(getattr(state, "livermore_state_age_1h", 999))
        bo_level = self._get_breakout_level(state, direction)
        if bo_level is not None:
            dist_atr = abs(close - bo_level) / atr
            prox_mult = float(asset_cfg.get(
                "breakout_proximity_atr_mult",
                2.0 if self._is_btc(symbol) else 1.25,
            ))
            if dist_atr <= prox_mult:
                mod = self._breakout_modifier(symbol, state, direction)
                _fresh = age_1h <= self._breakout_age_max
                if not _fresh:
                    mod += 0.15
                logger.debug(
                    "retest_engine: BREAKOUT @ %.5f (dist=%.2f ATR, mod=%.2f, fresh=%s)",
                    bo_level, dist_atr, mod, _fresh,
                )
                return RetestResult(
                    retest_type=RT_BREAKOUT,
                    modifier=mod,
                    entry_type=ET_TREND_FOLLOWING,
                    direction=direction,
                    level=bo_level,
                    level_2=level_2,
                    level_3=level_3,
                )

        # ── 2B. CONTINUATION (Scenario B) ────────────────────────
        # Tight 1H consolidation breaking out INSIDE an already-established 4H
        # MAIN leg (age_1h > breakout_age_max — distinguishes it from the fresh
        # state-flip BREAKOUT tier above). Gated by
        # state.phase_config.continuation_targets_enabled (default OFF).
        if getattr(state, "phase_config", {}).get("continuation_targets_enabled", False):
            _cont = self._classify_continuation(df, state, symbol, direction, atr, age_1h)
            if _cont is not None:
                return _cont

        # ── 3. WICK ───────────────────────────────────────────────
        # A liquidity sweep (sweep_detected) where price has already closed back
        # through the swept level in the trade direction.  Classic spring/upthrust.
        sweep_detected = getattr(state, "sweep_detected", False)
        if sweep_detected and self._wick_recovered(df, state, direction):
            sweep_level = getattr(state, "sweep_level", None)
            # If the swept level coincides with a tracked 4H structure level,
            # validate its CURRENT role — support/resistance can flip (see the
            # role-reversal logic in signal_aggregator.py's
            # _update_structure_memory) so this reads the live label fresh
            # each cycle rather than trusting the type from whenever the level
            # was first recorded. A spring only makes sense off a level
            # presently acting as support (LONG) or resistance (SHORT); if the
            # live label disagrees, this isn't really a spring off structure —
            # fall through to the next tier instead of returning WICK here.
            _role_conflict = False
            if (
                level is not None and sweep_level is not None
                and abs(sweep_level - level) / atr < self._clean_atr_mult
            ):
                _level_type = getattr(state, "nearby_4h_level_type", None)
                _expected_type = "swing_low" if direction == 1 else "swing_high"
                _role_conflict = _level_type is not None and _level_type != _expected_type

            if not _role_conflict:
                logger.debug("retest_engine: WICK @ sweep_level=%s", sweep_level)
                _rh, _rl = self._maybe_swing_range(state, df)
                return RetestResult(
                    retest_type=RT_WICK,
                    modifier=self._mod_wick,
                    entry_type=ET_SPRING_ENTRY,
                    direction=direction,
                    level=sweep_level,
                    range_high=_rh,
                    range_low=_rl,
                )
            logger.debug(
                "retest_engine: WICK candidate rejected — level %.5f now labeled %s, "
                "expected %s for direction=%d",
                level, _level_type, _expected_type, direction,
            )

        # ── 4 & 5. CHASE (requires a nearby level to measure distance from) ──
        if level is not None:
            dist_atr = abs(close - level) / atr
            chase_hard = float(asset_cfg.get(
                "chase_hard_atr_mult",
                2.0 if self._is_btc(symbol) else 2.5,
            ))
            chase_soft = float(asset_cfg.get(
                "chase_soft_atr_mult",
                1.2 if self._is_btc(symbol) else 1.5,
            ))
            # CHASE_HARD takes precedence over CHASE_SOFT
            if dist_atr >= chase_hard:
                logger.debug(
                    "retest_engine: CHASE_HARD @ %.5f (dist=%.2f ATR)", level, dist_atr
                )
                return RetestResult(
                    retest_type=RT_CHASE_HARD,
                    modifier=self._mod_chase_hard,
                    entry_type=ET_REJECT,
                    direction=direction,
                    level=level,
                    level_2=level_2,
                    level_3=level_3,
                )
            if dist_atr >= chase_soft:
                logger.debug(
                    "retest_engine: CHASE_SOFT @ %.5f (dist=%.2f ATR)", level, dist_atr
                )
                return RetestResult(
                    retest_type=RT_CHASE_SOFT,
                    modifier=self._mod_chase_soft,
                    entry_type=ET_MR_PULLBACK,
                    direction=direction,
                    level=level,
                    level_2=level_2,
                    level_3=level_3,
                )

        # ── 6. NO_LEVEL_NEARBY (fallback) ────────────────────────
        return self._no_level(symbol, direction)

    # ────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────

    def _no_level(self, symbol: str, direction: int) -> RetestResult:
        asset_cfg = self._assets.get(symbol, self._assets.get("DEFAULT", {}))
        mod = float(asset_cfg.get("modifier_no_level", self._mod_no_level_default))
        return RetestResult(
            retest_type=RT_NO_LEVEL_NEARBY,
            modifier=mod,
            entry_type=ET_TREND_FOLLOWING,
            direction=direction,
            level=None,
        )

    def _get_atr(self, df: pd.DataFrame) -> float:
        """
        Return ATR of the last candle.
        Prefers the 'atr' column populated by DataManager; falls back to a
        14-bar simplified TR mean if that column is absent or NaN.
        """
        if "atr" in df.columns:
            v = float(df["atr"].iloc[-1])
            if not np.isnan(v) and v > 0:
                return v
        # Simplified ATR fallback
        n = min(14, len(df) - 1)
        if n >= 1:
            highs  = df["high"].iloc[-(n + 1):-1].reset_index(drop=True)
            lows   = df["low"].iloc[-(n + 1):-1].reset_index(drop=True)
            closes = df["close"].iloc[-(n + 2):-2].reset_index(drop=True)
            tr = pd.concat([
                highs - lows,
                (highs - closes).abs(),
                (lows  - closes).abs(),
            ], axis=1).max(axis=1)
            v = float(tr.mean())
            if v > 0:
                return v
        # Last-resort: single-bar range
        return max(float(df["high"].iloc[-1] - df["low"].iloc[-1]), 1e-10)

    @staticmethod
    def _is_btc(symbol: str) -> bool:
        return symbol.upper() in _BTC_SYMBOLS

    @staticmethod
    def _is_fx(symbol: str) -> bool:
        s = symbol.upper()
        return any(s.startswith(p) for p in _FX_PREFIXES)

    def _get_breakout_level(self, state, direction: int) -> Optional[float]:
        """
        Return the most relevant Livermore anchor level for a fresh-breakout entry.
        For a LONG breakout: the level that was just cleared to the upside.
        For a SHORT breakout: the level that was just broken to the downside.
        Falls back to nearby_4h_level if no anchor is available.
        """
        ls = getattr(state, "livermore_state_1h", None)
        if direction == 1:
            if ls in ("MAIN_UP", "NATURAL_RETRACEMENT"):
                anchor = getattr(state, "livermore_anchor_main_up_max", None)
                if anchor is not None:
                    return anchor
            if ls == "SECONDARY_RETRACEMENT":
                anchor = getattr(state, "livermore_anchor_natural_low", None)
                if anchor is not None:
                    return anchor
        elif direction == -1:
            if ls in ("MAIN_DOWN", "NATURAL_REBOUND"):
                anchor = getattr(state, "livermore_anchor_main_down_min", None)
                if anchor is not None:
                    return anchor
            if ls == "SECONDARY_REBOUND":
                anchor = getattr(state, "livermore_anchor_natural_high", None)
                if anchor is not None:
                    return anchor
        # Final fallback
        return getattr(state, "nearby_4h_level", None)

    def _breakout_modifier(self, symbol: str, state, direction: int) -> float:
        """
        Determine the breakout threshold modifier based on 4H alignment.
          +0.10 — BTC, 4H-aligned (smallest raise)
          +0.20 — FX / GOLD / USTEC, 4H-aligned
          +0.40 — any symbol, 4H-misaligned (largest raise: counter-trend breakout)
        """
        ls4h = getattr(state, "livermore_state_4h", None)
        aligned = (
            (direction == 1  and ls4h in _LONG_CONFIRMING_4H_STATES) or
            (direction == -1 and ls4h in _SHORT_CONFIRMING_4H_STATES)
        )
        if not aligned:
            return self._mod_breakout_misaligned
        return (
            self._mod_breakout_aligned_btc
            if self._is_btc(symbol)
            else self._mod_breakout_aligned_fx
        )

    def _classify_continuation(
        self, df: pd.DataFrame, state, symbol: str, direction: int,
        atr: float, age_1h: int,
    ) -> Optional["RetestResult"]:
        """
        Tier 2B — CONTINUATION (Scenario B from MRS §11).

        Detects a tight 1H consolidation (continuation.min_bars..max_bars) that
        price has just broken out of, *inside* an already-established 4H MAIN
        leg (age_1h > breakout_age_max — this is what distinguishes it from the
        fresh state-flip BREAKOUT tier). Used to build the CONTINUATION target
        ladder in VTM (T1 = half-flagpole, T2 = full flagpole, measured from
        the leg-origin NATURAL anchor to the entry price).

        Returns None (falls through to WICK/CHASE/NO_LEVEL) if any
        disqualifying condition is met: wrong tier age, Livermore states not
        in a MAIN-aligned configuration, no tight-enough consolidation box
        found, price hasn't broken out of it yet, or the resulting flagpole
        is too small to be worth a dedicated ladder.
        """
        if age_1h <= self._breakout_age_max:
            return None  # fresh-breakout tier's job, not ours

        ls1h = getattr(state, "livermore_state_1h", None)
        ls4h = getattr(state, "livermore_state_4h", None)

        if direction == 1:
            if ls1h != "MAIN_UP" or ls4h not in _LONG_CONFIRMING_4H_STATES:
                return None
            leg_origin = getattr(state, "livermore_anchor_natural_low", None)
        elif direction == -1:
            if ls1h != "MAIN_DOWN" or ls4h not in _SHORT_CONFIRMING_4H_STATES:
                return None
            leg_origin = getattr(state, "livermore_anchor_natural_high", None)
        else:
            return None

        if leg_origin is None:
            return None

        close = float(df["close"].iloc[-1])
        buf = self._cont_breakout_buffer_atr_mult * atr

        cons_high = cons_low = None
        for n in range(self._cont_min_bars, self._cont_max_bars + 1):
            if len(df) < n + 1:
                break
            window = df.iloc[-(n + 1):-1]
            wh = float(window["high"].max())
            wl = float(window["low"].min())
            if (wh - wl) > self._cont_max_range_atr_mult * atr:
                continue
            if direction == 1 and close > wh + buf:
                cons_high, cons_low = wh, wl
                break
            if direction == -1 and close < wl - buf:
                cons_high, cons_low = wh, wl
                break

        if cons_high is None:
            return None

        flagpole = (close - leg_origin) if direction == 1 else (leg_origin - close)
        if flagpole < self._cont_min_flagpole_atr_mult * atr:
            return None

        mod = (
            self._cont_mod_aligned_btc
            if self._is_btc(symbol)
            else self._cont_mod_aligned_fx
        )
        level = cons_low if direction == 1 else cons_high
        logger.debug(
            "retest_engine: CONTINUATION @ %.5f (flagpole=%.5f, box=[%.5f, %.5f])",
            level, flagpole, cons_low, cons_high,
        )
        return RetestResult(
            retest_type=RT_CONTINUATION,
            modifier=mod,
            entry_type=ET_CONTINUATION,
            direction=direction,
            level=level,
            range_high=cons_high,
            range_low=cons_low,
        )

    def _maybe_swing_range(self, state, df: pd.DataFrame):
        """
        Compute the swing high/low "box" used for the RANGE_BOUNDARY /
        SPRING_ENTRY target ladder, gated by
        phase_config.continuation_targets_enabled (default OFF).
        Returns (None, None) when the flag is off or no box can be computed.
        """
        if not getattr(state, "phase_config", {}).get("continuation_targets_enabled", False):
            return None, None
        return self._swing_range(df)

    def _swing_range(self, df: pd.DataFrame):
        """
        Lookback-window swing high/low over range_ladder.lookback_bars,
        excluding the current (still-forming reference) candle.
        """
        n = self._range_lookback_bars
        if len(df) > n:
            window = df.iloc[-(n + 1):-1]
        else:
            window = df.iloc[:-1]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())

    def _wick_recovered(self, df: pd.DataFrame, state, direction: int) -> bool:
        """
        Return True if the close has recovered back through the swept level.
          LONG setup: sweep went below level, close is now *above* level.
          SHORT setup: sweep went above level, close is now *below* level.
        """
        sweep_level = getattr(state, "sweep_level", None)
        if sweep_level is None:
            return False
        close = float(df["close"].iloc[-1])
        if direction == 1:
            return close > sweep_level
        if direction == -1:
            return close < sweep_level
        return False
