"""
Mean Reversion Strategy — Phase 3A Three-Mode Rebuild
======================================================
Livermore State Machine routing replaces the legacy scorecard that produced
$0 P&L. Requires Phase 1 (Livermore) and Phase 2 (Hard Veto / RSM gates) to
be running. Falls back to legacy scorecard during Livermore warmup period.

Mode 1 — Pullback Completion  (1H NATURAL_RETRACEMENT, LONG only)
  - Spring MANDATORY: wick sweeps prior swing low, closes back above it.
      Penetration 0.5–5% of swing level. Recovery within 1–3 bars.
      Current bar volume must be lower than spring bar volume.
  - 2 of 4 optional: vol_contraction, hidden_divergence, bb_contraction, ma_proximity
  - vol_down_ratio > 1.2  →  full block (distribution, not re-accumulation)
  - BTC near 4H EMA200     →  −0.10 confidence modifier

Mode 2 — Counter-Trend  (1H SECONDARY_RETRACEMENT / SECONDARY_REBOUND)
  SECONDARY_RETRACEMENT → LONG  (counter to the deep pullback)
  SECONDARY_REBOUND     → SHORT (counter to the deep bounce)
  - ADX < 25  (mandatory — market must not be trending)
  - 4 of 4 optional conditions  (all required)
  - BB must have closed back inside bands before entry
  - BTC only: LONG z-score < −2.0; SHORT z-score > +3.5

Mode 3 — Climax Fade  (1H MAIN_UP / MAIN_DOWN)
  MAIN_UP   → SHORT (fade the overextended leg)
  MAIN_DOWN → LONG  (fade the selling climax)
  - leg_stretch_ratio  > 1.5× typical leg  (distance from Livermore anchor)
  - price              > 2.5×ATR from anchor
  - High-rank reversal candle (bearish/bullish engulfing or evening/morning star)
  - Above-average volume on signal bar
  - Target is EMA20 within MAIN state — NOT a full reversal call

All numeric thresholds in config/aggregator_presets.json → MR_THREE_MODE.
"""

import json
import os
from typing import Optional
import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

# Resolved relative to this module's directory (src/strategies/ → ../../config/)
_PRESETS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "aggregator_presets.json")
)


class MeanReversionStrategy(BaseStrategy):
    """
    Phase 3A: Livermore-state-conditional three-mode mean reversion.
    Wired into PerformanceWeightedAggregator via generate_signal().
    """

    def __init__(self, config: dict):
        super().__init__(config, "MeanReversion")

        # Core indicator parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std    = config.get("bb_std", 2.0)
        self.rsi_period  = config.get("rsi_period", 14)
        self.stoch_k     = config.get("stoch_k", 14)
        self.stoch_d     = config.get("stoch_d", 3)
        self.reversion_window = config.get("reversion_window", 3)
        self.asset = config.get("asset", "BTC")

        # Legacy scorecard thresholds (fallback during warmup)
        self.rsi_overbought      = config.get("rsi_overbought", 64)
        self.rsi_oversold        = config.get("rsi_oversold", 35)
        self.bb_lower_threshold  = config.get("bb_lower_threshold", 0.25)
        self.bb_upper_threshold  = config.get("bb_upper_threshold", 0.75)
        self.min_return_threshold = config.get("min_return_threshold", 0.0025)
        self.min_score_threshold = config.get("min_conditions", 3.0)
        self.use_4h_context = config.get("use_4h_context", True)

        # Phase 3A thresholds — loaded once from presets JSON
        self._mr3_cfg = self._load_mr3_config()

        logger.info(
            f"[{self.name}] Phase 3A initialized: asset={self.asset} | "
            f"mode1 spring={self._mr3_cfg['mode1']['spring_min_penetration']:.1%}–"
            f"{self._mr3_cfg['mode1']['spring_max_penetration']:.1%} "
            f"opt≥{self._mr3_cfg['mode1']['optional_min_count']} | "
            f"mode2 adx<{self._mr3_cfg['mode2']['adx_max']} opt={self._mr3_cfg['mode2']['optional_min_count']}/4 | "
            f"mode3 stretch≥{self._mr3_cfg['mode3']['leg_stretch_ratio_min']}×"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIG LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def _load_mr3_config(self) -> dict:
        """
        Load MR_THREE_MODE section from aggregator_presets.json.
        Deep-merges config values over safe built-in defaults so any missing
        key falls back gracefully (never raises at runtime).
        """
        defaults = {
            "mode1": {
                "spring_min_penetration":  0.005,
                "spring_max_penetration":  0.05,
                "spring_recovery_max_bars": 3,
                "spring_swing_lookback":   20,
                "vol_contraction_avg_pct": 0.80,
                "vol_spike_max_pct":       1.50,
                "bb_lower_zone_threshold": 0.20,
                "ma_proximity_atr_mult":   0.5,
                "optional_min_count":      2,
                "vol_down_ratio_veto":     1.2,
                "btc_ema200_4h_modifier": -0.10,
            },
            "mode2": {
                "adx_max":                     25,
                "optional_min_count":          4,
                "bb_inside_required":          True,
                "btc_long_zscore_threshold":  -2.0,
                "btc_short_zscore_threshold":  3.5,
            },
            "mode3": {
                "leg_stretch_ratio_min":     1.5,
                "leg_stretch_window":        10,
                "leg_stretch_lookback":      50,
                "price_anchor_atr_mult":     2.5,
                "require_above_avg_volume":  True,
                "volume_avg_lookback":       20,
            },
            "bb_kc_squeeze": {
                "min_squeeze_bars":        5,
                "bbw_percentile_threshold": 30.0,
                "kc_ema_period":           20,
                "kc_atr_mult":             1.5,
            },
            "nr7_lookback": 7,
        }
        try:
            if os.path.exists(_PRESETS_PATH):
                with open(_PRESETS_PATH, "r") as _f:
                    _data = json.load(_f)
                _section = _data.get("MR_THREE_MODE", {})
                for key, val in _section.items():
                    if isinstance(val, dict) and key in defaults and isinstance(defaults[key], dict):
                        defaults[key].update(val)
                    else:
                        defaults[key] = val
        except Exception as _e:
            logger.warning(f"[{self.name}] Could not load MR_THREE_MODE config: {_e}. Using defaults.")
        return defaults

    # ─────────────────────────────────────────────────────────────────────────
    # WARMUP + FEATURE GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    def get_warmup_period(self) -> int:
        return max(
            self.bb_period,
            self.rsi_period,
            50,   # EMA50
            200,  # EMA200 (Mode 1 MA proximity check)
        )

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators used across modes + legacy scorecard."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        df["bb_upper"]    = bb_upper
        df["bb_middle"]   = bb_middle
        df["bb_lower"]    = bb_lower
        df["bb_position"] = (close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-10)
        df["bb_width_norm"] = (bb_upper - bb_lower) / np.maximum(np.abs(bb_middle), 1e-10)

        # RSI + ADX + ATR
        df["rsi"] = ta.RSI(close, timeperiod=self.rsi_period)
        df["adx"] = ta.ADX(high, low, close, timeperiod=14)
        df["atr"] = ta.ATR(high, low, close, timeperiod=14)

        # EMAs  (200 needed for Mode 1 MA proximity)
        df["ema_20"]  = ta.EMA(close, timeperiod=20)
        df["ema_50"]  = ta.EMA(close, timeperiod=50)
        df["ema_200"] = ta.EMA(close, timeperiod=200)

        # L6: Livermore state one-hot ML features (flag-gated, see base_strategy.py)
        df = self._add_livermore_features(df, timeframe="1H")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # SPRING DETECTION  (Mode 1 mandatory)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_spring(self, df: pd.DataFrame, anchor: Optional[float] = None, direction: int = 1) -> tuple:
        """
        Wyckoff spring (direction=+1): a bar's wick sweeps below a prior swing
        low then closes back above it within spring_recovery_max_bars of the
        current bar.

        Unit 2 mirror — Wyckoff upthrust (direction=-1): a bar's wick sweeps
        ABOVE a prior swing high then closes back BELOW it within the same
        window. Same penetration/volume conditions, mirrored.

        Penetration must be 0.5–5% of the swing low/high level.
        Current-bar volume must be lower than the spring/upthrust bar's volume
        (that bar = climax; entry bar = quiet absorption).

        `anchor`: when provided (the same livermore_anchor_natural_low/_high
        the structural gate — BreakRetestValidator — already confirmed a
        sweep-and-recover against), this is used as the reference level
        instead of an independently-computed rolling swing low/high.

        Investigation finding (2026-07-14): BRV's "structural proof" for
        NATURAL_RETRACEMENT is itself a spring check ("the spring — market
        tested the proven structural low") against livermore_anchor_natural_low
        — a state-machine-confirmed, cycle-locked level. This function was
        independently recomputing its own rolling 20-bar swing low as the
        reference instead, a different level that drifts every bar. The two
        checks were effectively re-verifying the same event against two
        unrelated reference points — across a full backtest, 26 bars passed
        BRV's spring check and only 1 of those 26 also passed this one.
        Reusing the same anchor BRV already validated removes that mismatch
        (Unit 2: the same reasoning applies to the short-side anchor, hence
        the upthrust mirror also takes its anchor from BRV, never recomputing
        an independent zone-ladder or rolling level).
        Falls back to the old rolling-swing-low/high calculation when no
        anchor is available (state machine hasn't locked one yet).

        Returns: (found: bool, strength: float 0–1, spring_bar_idx: Optional[int])
        spring_bar_idx is the negative array index of the identified spring/
        upthrust bar, so callers (the optional-confirmation checks) can
        exclude it from their own scans instead of penalising the exact
        high-volume bar the check mechanically requires.
        """
        cfg = self._mr3_cfg["mode1"]
        min_pen  = cfg["spring_min_penetration"]    # 0.005
        max_pen  = cfg["spring_max_penetration"]    # 0.050
        max_bars = cfg["spring_recovery_max_bars"]  # 3
        swing_lb = cfg["spring_swing_lookback"]     # 20

        close  = df["close"].values
        low    = df["low"].values
        high   = df["high"].values
        volume = df["volume"].values if "volume" in df.columns else None
        _ref_arr = low if direction == 1 else high

        if anchor is not None and float(anchor) > 0:
            if len(df) < max_bars + 2:
                return False, 0.0, None
            prior_swing = float(anchor)
        else:
            # Fallback: independently-computed rolling swing low/high (legacy
            # path, used only when the Livermore anchor isn't confirmed yet).
            min_total = swing_lb + max_bars + 2
            if len(df) < min_total:
                return False, 0.0, None
            # Prior swing is established from the window BEFORE the search
            # bars so we don't confuse the spring/upthrust bar itself as the swing.
            _win_end   = -(max_bars + 1)           # index of last bar before search window
            _win_start = _win_end - swing_lb
            _prior_arr = _ref_arr[_win_start:_win_end]
            if len(_prior_arr) < 5:
                return False, 0.0, None
            prior_swing = float(np.min(_prior_arr)) if direction == 1 else float(np.max(_prior_arr))
            if prior_swing <= 0:
                return False, 0.0, None

        # Scan the last max_bars bars (not including current bar) for the spring/upthrust
        for k in range(1, max_bars + 1):
            bar_idx   = -(k + 1)   # k positions before current bar
            if abs(bar_idx) > len(df):
                continue

            bar_close = float(close[bar_idx])

            if direction == 1:
                bar_low = float(low[bar_idx])
                # Condition 1: wick swept below swing low
                if bar_low >= prior_swing:
                    continue
                # Condition 2: bar closed BACK ABOVE the swept level
                if bar_close <= prior_swing:
                    continue
                # Condition 3: penetration in [0.5%, 5%]
                penetration = (prior_swing - bar_low) / prior_swing
            else:
                bar_high = float(high[bar_idx])
                # Condition 1: wick swept above swing high
                if bar_high <= prior_swing:
                    continue
                # Condition 2: bar closed BACK BELOW the swept level
                if bar_close >= prior_swing:
                    continue
                # Condition 3: penetration in [0.5%, 5%]
                penetration = (bar_high - prior_swing) / prior_swing

            if not (min_pen <= penetration <= max_pen):
                continue

            # Condition 4: current-bar volume < spring/upthrust-bar volume
            if volume is not None:
                event_vol   = float(volume[bar_idx])
                current_vol = float(volume[-1])
                if event_vol > 0 and current_vol >= event_vol:
                    continue

            # Spring/upthrust confirmed. Strength peaks near 2.5% penetration.
            _optimal  = 0.025
            strength  = max(0.30, 1.0 - abs(penetration - _optimal) / _optimal)
            return True, float(min(strength, 1.0)), bar_idx

        return False, 0.0, None

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIONAL CONDITIONS  (shared by Mode 1 and Mode 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_vol_contraction(self, df: pd.DataFrame, direction: int, exclude_bar_idx: Optional[int] = None) -> bool:
        """
        Volume Contraction:
          - direction=+1 (long):  down-close bar volumes < 80% of 20-bar avg
          - direction=-1 (short): up-close bar volumes < 80% of 20-bar avg
          - No single bar in the last 5 exceeds 150% of average
        Signals quiet re-accumulation (long) / re-distribution (short), not
        continued adverse pressure.

        `exclude_bar_idx`: Mode 1's mandatory spring/upthrust bar is, by its
        own definition, a high-volume adverse-close bar (_detect_spring
        requires that bar's volume > current-bar volume — "climax bar"). That
        bar normally falls inside this function's own last-5-bars scan window
        (spring_recovery_max_bars=3 vs this scan's -6:-1), so a genuine spring
        was mechanically vetoing this check on the exact bar that makes it a
        spring. Excluding the identified spring/upthrust bar from the scan
        resolves that self-contradiction without loosening the volume
        thresholds.
        """
        if "volume" not in df.columns or len(df) < 22:
            return False
        cfg    = self._mr3_cfg["mode1"]
        avg_pct   = cfg["vol_contraction_avg_pct"]   # 0.80
        spike_pct = cfg["vol_spike_max_pct"]          # 1.50
        try:
            vol    = df["volume"].values
            close  = df["close"].values
            open_  = df["open"].values
            n      = min(22, len(vol))
            vol_n  = vol[-n:]
            avg_vol = float(np.mean(vol_n[:-2]))    # 20-bar avg excluding last 2 bars
            if avg_vol <= 0:
                return False
            # Scan last 5 bars (excluding current)
            for j in range(-6, -1):
                if exclude_bar_idx is not None and j == exclude_bar_idx:
                    continue
                bv = float(vol[j])
                _is_adverse = (
                    float(close[j]) < float(open_[j]) if direction == 1
                    else float(close[j]) > float(open_[j])
                )
                if _is_adverse and bv > avg_vol * avg_pct:
                    return False    # Adverse-close bar with above-threshold volume
                if bv > avg_vol * spike_pct:
                    return False    # Spike bar breaks contraction
            return True
        except Exception:
            return False

    def _check_hidden_divergence(self, df: pd.DataFrame, direction: int) -> bool:
        """
        Hidden Bullish Divergence (direction=+1): price makes lower lows but RSI
        makes higher lows — momentum holding while price dips. Confirms pullback.

        Hidden Bearish Divergence (direction=−1): price higher highs, RSI lower
        highs — hidden weakness at top.
        """
        try:
            if len(df) < 40 or "rsi" not in df.columns:
                return False
            low_arr  = df["low"].values
            high_arr = df["high"].values
            rsi_arr  = df["rsi"].values
            if pd.isna(rsi_arr[-1]):
                return False

            if direction == 1:
                recent_low  = float(np.min(low_arr[-6:-1]))
                prior_low   = float(np.min(low_arr[-26:-6]))
                recent_rsi  = float(np.nanmin(rsi_arr[-6:-1]))
                prior_rsi   = float(np.nanmin(rsi_arr[-26:-6]))
                # Price lower low AND RSI higher low = hidden bullish divergence
                return recent_low < prior_low and recent_rsi > prior_rsi

            elif direction == -1:
                recent_high = float(np.max(high_arr[-6:-1]))
                prior_high  = float(np.max(high_arr[-26:-6]))
                recent_rsi  = float(np.nanmax(rsi_arr[-6:-1]))
                prior_rsi   = float(np.nanmax(rsi_arr[-26:-6]))
                # Price higher high AND RSI lower high = hidden bearish divergence
                return recent_high > prior_high and recent_rsi < prior_rsi

        except Exception:
            return False
        return False

    def _check_bb_contraction(self, df: pd.DataFrame, direction: int, eval_bar_idx: Optional[int] = None) -> bool:
        """
        BB Contraction: price is in the lower 20% (long) or upper 80%+ (short)
        of the Bollinger Band while bandwidth is stable or declining.
        Confirms the price is stretched AND momentum is bleeding off.

        `eval_bar_idx`: for Mode 1, defaults to the current bar (-1), but a
        real spring's own recovery rally routinely carries price and
        bandwidth well past this zone within 1-3 bars — investigation
        (2026-07-14) found bb_position >0.88 (i.e. near/above the *upper*
        band) and expanding bandwidth at all 4 sampled post-spring bars.
        This check needs to confirm the coil existed AT the spring, not
        that it's still there now — pass spring_bar_idx to evaluate there.
        """
        try:
            if len(df) < self.bb_period + 5:
                return False
            cfg      = self._mr3_cfg["mode1"]
            zone_pct = cfg["bb_lower_zone_threshold"]   # 0.20
            bb_pos   = df["bb_position"].values
            bb_wid   = df["bb_width_norm"].values
            _idx = eval_bar_idx if eval_bar_idx is not None else -1
            if pd.isna(bb_pos[_idx]) or pd.isna(bb_wid[_idx]):
                return False
            pos        = float(bb_pos[_idx])
            _win_end   = _idx
            _win_start = _win_end - 5
            bw_prev5   = bb_wid[_win_start:_win_end]
            bw_prev5   = bw_prev5[~np.isnan(bw_prev5)]
            if len(bw_prev5) == 0:
                return False
            bw_mean      = float(np.mean(bw_prev5))
            bw_declining = float(bb_wid[_idx]) <= bw_mean

            if direction == 1:
                return pos < zone_pct and bw_declining
            elif direction == -1:
                return pos > (1.0 - zone_pct) and bw_declining
        except Exception:
            return False
        return False

    def _check_ma_proximity(self, df: pd.DataFrame, direction: int, eval_bar_idx: Optional[int] = None) -> bool:
        """
        MA Proximity: price is within 0.5×ATR of EMA50 or EMA200.
        Classic Wyckoff Last Point of Support — price testing a major MA.

        `eval_bar_idx`: same rationale as _check_bb_contraction — a spring's
        recovery rally moves price away from the MA it bounced off within a
        few bars (investigation found 1.86-5.19 ATR away at the sampled
        events). Pass spring_bar_idx to check proximity at the spring itself.
        """
        try:
            if len(df) < 5:
                return False
            cfg    = self._mr3_cfg["mode1"]
            mult   = cfg["ma_proximity_atr_mult"]    # 0.5
            _idx   = eval_bar_idx if eval_bar_idx is not None else -1
            close  = float(df["close"].values[_idx])
            atr_v  = float(df["atr"].values[_idx]) if not pd.isna(df["atr"].values[_idx]) else 0.0
            if atr_v <= 0:
                return False
            threshold = mult * atr_v
            ema50  = df["ema_50"].values[_idx]  if "ema_50"  in df.columns else None
            ema200 = df["ema_200"].values[_idx] if "ema_200" in df.columns else None
            if ema50  is not None and not pd.isna(ema50)  and abs(close - float(ema50))  <= threshold:
                return True
            if ema200 is not None and not pd.isna(ema200) and abs(close - float(ema200)) <= threshold:
                return True
        except Exception:
            return False
        return False

    def _count_optional(
        self, df: pd.DataFrame, direction: int,
        exclude_bar_idx: Optional[int] = None, eval_bar_idx: Optional[int] = None,
    ) -> int:
        """Count how many of the 4 optional conditions are met."""
        return sum([
            self._check_vol_contraction(df, direction, exclude_bar_idx=exclude_bar_idx),
            self._check_hidden_divergence(df, direction),
            self._check_bb_contraction(df, direction, eval_bar_idx=eval_bar_idx),
            self._check_ma_proximity(df, direction, eval_bar_idx=eval_bar_idx),
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # BB Z-SCORE  (Mode 2 BTC-specific gate)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_bb_zscore(self, df: pd.DataFrame) -> float:
        """
        Bollinger Band z-score: (close − middle) / σ
        For 2σ bands: σ = (upper − lower) / 4.
        z < 0 → below mean, z < −2 → below lower band.
        """
        try:
            close    = float(df["close"].values[-1])
            bb_mid   = float(df["bb_middle"].values[-1])
            bb_upper = float(df["bb_upper"].values[-1])
            bb_lower = float(df["bb_lower"].values[-1])
            band_width = bb_upper - bb_lower
            if band_width <= 0:
                return 0.0
            sigma = band_width / 4.0    # 2σ bands → σ = (upper − lower) / 4
            return (close - bb_mid) / sigma
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1: Pullback Completion
    # ─────────────────────────────────────────────────────────────────────────

    def _mode1_pullback_completion(
        self, df: pd.DataFrame, composite_state, side: str = "long"
    ) -> tuple:
        """
        Mode 1 — Pullback Completion (1H NATURAL_RETRACEMENT, LONG by default).

        Unit 2: side="short" mirrors the whole method for NATURAL_REBOUND —
        upthrust instead of spring, BRV's BEARISH_RETEST anchor
        (livermore_anchor_natural_high) instead of BULLISH_RETEST's, and
        every directional optional-confirmation check flipped to direction=-1.

        [MANDATORY] BB/KC squeeze OR NR7/NR7-ID active (compression required)
        [MANDATORY] Spring/upthrust detected in last 1–3 bars
        [2 of 4]    Optional: vol_contraction, hidden_div, bb_contraction, ma_proximity
        [VETO]      vol_down_ratio > 1.2 (Unit 2: dampener when flagged — see caller)
        [MODIFIER]  BTC near 4H EMA200 → −0.10 confidence
        """
        _direction = 1 if side == "long" else -1
        # Unit 2: reset defensively at entry, not just after use — several
        # gates below this point can return 0, 0.0 before ever reaching the
        # apply-and-reset line near the confidence calc. Resetting here
        # guarantees no stale dampener from an earlier early-exit cycle ever
        # reaches a later, successful cycle's confidence.
        self._vdr_damp = 1.0
        cfg      = self._mr3_cfg["mode1"]
        features = self.generate_features(df.tail(260))
        if len(features) < 50:
            return 0, 0.0

        # ── vol_down_ratio veto ────────────────────────────────────────────
        if composite_state is not None:
            _vdr       = composite_state.vol_down_ratio
            _vdr_valid = composite_state.vol_down_ratio_valid
            _veto_thr  = cfg["vol_down_ratio_veto"]
            if _vdr_valid and _vdr is not None and float(_vdr) > _veto_thr:
                _phase_cfg_vdr = getattr(composite_state, "phase_config", {}) or {}
                if _phase_cfg_vdr.get("mr_vetoes_as_dampeners_enabled", False):
                    # Unit 2: dampen instead of kill. Scale confidence down by
                    # how far over the threshold we are (min 0.4x floor) so a
                    # strong spring can still fire against mild distribution.
                    _over = float(_vdr) - _veto_thr
                    _damp = max(0.4, 1.0 - min(0.6, _over))
                    self._vdr_damp = _damp   # applied to final conf below
                    logger.info(
                        f"[MR Mode1] {self.asset}: vol_down_ratio={_vdr:.2f} > "
                        f"{_veto_thr} → DAMPEN x{_damp:.2f} (flag ON)"
                    )
                else:
                    logger.info(
                        f"[MR Mode1] {self.asset}: vol_down_ratio={_vdr:.2f} > {_veto_thr} → VETO"
                    )
                    return 0, 0.0

        # ── Compression gate: BB/KC squeeze OR NR7/NR7-ID (mandatory) ─────
        # Spring setups require prior volatility compression; without it the
        # spring detection fires on ordinary pullbacks with no coiled energy.
        # Togglable via phase_config.bb_kc_squeeze_gate_enabled (default: True).
        if composite_state is not None:
            _phase_cfg = getattr(composite_state, "phase_config", {}) or {}
            _squeeze_gate_on = _phase_cfg.get("bb_kc_squeeze_gate_enabled", True)
            _nr7_gate_on     = _phase_cfg.get("nr7_gate_enabled", True)
            _squeeze = getattr(composite_state, "bb_kc_squeeze_active", False)
            _nr7     = getattr(composite_state, "nr7_active", False)
            _nr7_id  = getattr(composite_state, "nr7_id_active", False)
            _compression_ok = (
                (_squeeze_gate_on and _squeeze)
                or (_nr7_gate_on and (_nr7 or _nr7_id))
            )
            if not _compression_ok:
                logger.info(
                    f"[MR Mode1] {self.asset}: no compression "
                    f"(squeeze={_squeeze} nr7={_nr7} nr7_id={_nr7_id}) → VETO"
                )
                return 0, 0.0

         # ── STRUCTURAL GATE: Break and Retest required before spring ────────
        # The BRV checks whether price has swept the Livermore anchor_natural_low
        # and closed back above it. This is the exact spring-at-structure
        # sequence that Livermore waited for before committing.
        #
        # Uses Livermore anchors when available (independent of pivot calibration).
        # Falls back to legacy 50-bar logic when anchor not yet confirmed.
        # Neither path blocks Mode 1 permanently — both require structural proof.
        # ───────────────────────────────────────────────────────────────────
        if not hasattr(self, "_brv_validator"):
            from src.analysis.break_retest import BreakRetestValidator
            self._brv_validator = BreakRetestValidator()

        _brv_result = self._brv_validator.validate(df, composite_state=composite_state)

        if not _brv_result.is_valid:
            logger.info(
                "[MR Mode1] %s: no structural proof — BRV returned %s "
                "(anchor retest or legacy B&R required before spring entry)",
                self.asset, _brv_result.type,
            )
            return 0, 0.0

        logger.info(
            "[MR Mode1] %s: structural proof confirmed via %s @ %.5g — "
            "proceeding to spring check",
            self.asset, _brv_result.type, _brv_result.level,
        )

        # ── Spring/upthrust (mandatory) ─────────────────────────────────────
        # Reuse the exact level BRV just validated the sweep-and-recover
        # against, instead of independently recomputing a rolling swing
        # low/high (see _detect_spring's docstring for why these must not
        # diverge). BRV already resolves the correct anchor for direction
        # via lsm_state — natural_low for NATURAL_RETRACEMENT (long),
        # natural_high for NATURAL_REBOUND (short).
        spring_ok, spring_strength, spring_bar_idx = self._detect_spring(
            features, anchor=_brv_result.level, direction=_direction
        )
        if not spring_ok:
            logger.info(f"[MR Mode1] {self.asset}: no spring/upthrust detected → 0")
            return 0, 0.0

        # ── O3a: Orphan Confluence signals for spring confirmation ─────────
        # These four signals are computed every candle, directly relevant to
        # Mode 1 spring setup quality, and were previously never read here.
        _coiled_spring   = bool(getattr(composite_state, "coiled_spring", False)) if composite_state else False
        _inside_bar      = bool(getattr(composite_state, "inside_bar", False)) if composite_state else False
        _outside_bar     = bool(getattr(composite_state, "outside_bar", False)) if composite_state else False
        _failed_breakout = bool(getattr(composite_state, "failed_breakout", False)) if composite_state else False

        # coiled_spring during NATURAL_RETRACEMENT = Confluence confirms the
        # structural precondition Mode 1 already checks via BB/KC squeeze.
        # Log it for observability without adding a new hard gate.
        if _coiled_spring:
            logger.info(f"[MR Mode1] {self.asset}: coiled_spring=True — strong spring precondition")

        # ── Optional conditions (2 of 4 required) ─────────────────────────
        opt_count = self._count_optional(
            features, direction=_direction, exclude_bar_idx=spring_bar_idx, eval_bar_idx=spring_bar_idx,
        )
        min_opt   = cfg["optional_min_count"]

        # failed_breakout = price wicked above recent high, closed below it.
        # This is always a bearish candle. During a genuine NATURAL_RETRACEMENT
        # spring it's a valid precursor — but we must also confirm RSI is NOT
        # in bearish territory (< 40), which would indicate a downtrend rally
        # being rejected rather than a true spring forming.
        # Unit 2: no symmetric "failed_breakdown" field exists on the board to
        # build an equally-reasoned mirror for the short/upthrust path, and a
        # bearish failed_breakout is directionally ALIGNED (not a caution
        # flag) for a short setup rather than ambiguous the way it is for a
        # long spring — so this gate-reduction heuristic only applies long;
        # short never gets this shortcut (stays at least as strict, never less).
        if _failed_breakout and side == "long":
            _rsi_val = float(features.get("rsi", 50)) if isinstance(features, dict) else 50.0
            _directionally_valid = _rsi_val >= 40  # above 40 = not in bearish flush
            if _directionally_valid:
                min_opt = max(1, cfg["optional_min_count"] - 1)
                logger.info(
                    f"[MR Mode1] {self.asset}: failed_breakout confirmed (RSI={_rsi_val:.1f}) — "
                    f"optional requirement reduced to {min_opt}"
                )
            else:
                logger.info(
                    f"[MR Mode1] {self.asset}: failed_breakout present but RSI={_rsi_val:.1f} "
                    f"< 40 — directionally invalid for long spring, gate not reduced"
                )
        if opt_count < min_opt:
            _vc = self._check_vol_contraction(features, _direction, exclude_bar_idx=spring_bar_idx)
            _hd = self._check_hidden_divergence(features, _direction)
            _bc = self._check_bb_contraction(features, _direction, eval_bar_idx=spring_bar_idx)
            _mp = self._check_ma_proximity(features, _direction, eval_bar_idx=spring_bar_idx)
            logger.info(
                f"[MR Mode1] {self.asset}: spring OK but opt={opt_count}/{min_opt} → 0 "
                f"[vol={'✓' if _vc else '✗'} "
                f"hdiv={'✓' if _hd else '✗'} "
                f"bbc={'✓' if _bc else '✗'} "
                f"map={'✓' if _mp else '✗'}]"
            )
            # TEMP-DIAGNOSTIC: raw near-miss margins for the optional-gate
            # investigation (2026-07-14) — remove once the bottleneck is found.
            try:
                _vol_arr = features["volume"].values
                _n = min(22, len(_vol_arr))
                _avg_vol = float(np.mean(_vol_arr[-_n:][:-2]))
                _bars_dump = []
                for _j in range(-6, -1):
                    if spring_bar_idx is not None and _j == spring_bar_idx:
                        _bars_dump.append(f"{_j}=EXCLUDED(spring)")
                        continue
                    _bv = float(_vol_arr[_j])
                    _is_dn = float(features["close"].values[_j]) < float(features["open"].values[_j])
                    _bars_dump.append(f"{_j}={'DN' if _is_dn else 'UP'}:{_bv/_avg_vol:.2f}x")
                logger.info(f"[MR DIAG vol] {self.asset}: avg_vol={_avg_vol:.0f} bars={_bars_dump}")

                _low_arr, _high_arr, _rsi_arr = features["low"].values, features["high"].values, features["rsi"].values
                _recent_low, _prior_low = float(np.min(_low_arr[-6:-1])), float(np.min(_low_arr[-26:-6]))
                _recent_rsi, _prior_rsi = float(np.nanmin(_rsi_arr[-6:-1])), float(np.nanmin(_rsi_arr[-26:-6]))
                logger.info(
                    f"[MR DIAG hdiv] {self.asset}: recent_low={_recent_low:.4g}(vs prior={_prior_low:.4g}, "
                    f"lower={_recent_low < _prior_low}) recent_rsi={_recent_rsi:.1f}(vs prior={_prior_rsi:.1f}, "
                    f"higher={_recent_rsi > _prior_rsi})"
                )

                _eval_idx = spring_bar_idx if spring_bar_idx is not None else -1
                _bb_pos = float(features["bb_position"].values[_eval_idx])
                _bb_wid = features["bb_width_norm"].values
                _bw_prev5 = _bb_wid[_eval_idx - 5:_eval_idx]
                _bw_prev5 = _bw_prev5[~np.isnan(_bw_prev5)]
                _bw_mean = float(np.mean(_bw_prev5)) if len(_bw_prev5) else float("nan")
                logger.info(
                    f"[MR DIAG bbc] {self.asset}: @idx={_eval_idx} bb_pos={_bb_pos:.3f}(need<0.20) "
                    f"bw_now={float(_bb_wid[_eval_idx]):.4f} bw_prev5_mean={_bw_mean:.4f} "
                    f"declining={float(_bb_wid[_eval_idx]) <= _bw_mean if _bw_prev5.size else 'N/A'}"
                )

                _close_v = float(features["close"].values[_eval_idx])
                _atr_v = float(features["atr"].values[_eval_idx])
                _ema50 = float(features["ema_50"].values[_eval_idx]) if "ema_50" in features.columns else float("nan")
                _ema200 = float(features["ema_200"].values[_eval_idx]) if "ema_200" in features.columns else float("nan")
                _dist50 = abs(_close_v - _ema50) / _atr_v if _atr_v > 0 else float("nan")
                _dist200 = abs(_close_v - _ema200) / _atr_v if _atr_v > 0 else float("nan")
                logger.info(
                    f"[MR DIAG map] {self.asset}: dist_to_ema50={_dist50:.2f}ATR "
                    f"dist_to_ema200={_dist200:.2f}ATR (need <=0.5ATR either)"
                )
            except Exception as _diag_err:
                logger.info(f"[MR DIAG] {self.asset}: diagnostic dump failed: {_diag_err}")
            return 0, 0.0

        # ── BTC near 4H EMA200: small confidence penalty ──────────────────
        conf_mod = 0.0
        if self.asset in ("BTC", "BTCUSDT") and composite_state is not None:
            try:
                _ema_dist = getattr(composite_state, "ema200_1d_dist_atr", None)
                if _ema_dist is not None and float(_ema_dist) < 1.0:
                    conf_mod = float(cfg["btc_ema200_4h_modifier"])   # −0.10
            except Exception:
                pass

        # ── Confidence: spring strength + optional bonus + modifier ───────
        # spring_strength 0.30–1.0 → base 0.55–0.75
        base_conf = 0.55 + float(spring_strength) * 0.20
        extra_opt = max(0, opt_count - min_opt)
        # ── O3a: Orphan confidence modifiers ─────────────────────────────
        # outside_bar after a spring = candle breaking out of the coil's
        # range. Positive confirmation of spring breakout. Boost confidence.
        if _outside_bar:
            conf_mod += 0.05
            logger.debug(f"[MR Mode1] {self.asset}: outside_bar +0.05 confidence")
        # inside_bar during compression = price tightening within range.
        # Neutral to mildly positive for spring setup quality.
        if _inside_bar and not _outside_bar:
            conf_mod += 0.02
        confidence = float(min(1.0, base_conf + extra_opt * 0.05 + conf_mod))
        # Unit 2: apply the vol_down_ratio dampener if it was set above, then
        # reset — without this reset a dampener from one cycle leaks into
        # the next call's confidence unrelated to this one's own veto check.
        confidence = confidence * getattr(self, "_vdr_damp", 1.0)
        self._vdr_damp = 1.0

        logger.info(
            f"[MR Mode1] {self.asset}: {side.upper()} "
            f"{'spring' if side == 'long' else 'upthrust'}_str={spring_strength:.2f} "
            f"opt={opt_count}/4 conf={confidence:.2f}"
        )
        return _direction, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2: Counter-Trend
    # ─────────────────────────────────────────────────────────────────────────

    def _mode2_counter_trend(
        self, df: pd.DataFrame, composite_state
    ) -> tuple:
        """
        Mode 2 — Counter-Trend (SECONDARY states only).
        ...
        """
        # Unit 2: reset defensively at entry, not just after use — several
        # gates below this point can return 0, 0.0 before ever reaching the
        # apply-and-reset line near the confidence calc. Resetting here
        # guarantees no stale dampener from an earlier early-exit cycle ever
        # reaches a later, successful cycle's confidence.
        self._rc_damp = 1.0
        cfg      = self._mr3_cfg["mode2"]
        features = self.generate_features(df.tail(260))
        if len(features) < 50:
            return 0, 0.0

        # ── STRUCTURAL GATE: CHoCH required in SECONDARY states ─────────────
        # SECONDARY_RETRACEMENT and SECONDARY_REBOUND are the deepest, most
        # dangerous correction environments. Catching a knife here without
        # structural proof of exhaustion is the fastest way to take a maximum
        # loss. CHoCH — a higher low or lower high forming in the secondary
        # move — is the minimum evidence that pressure is beginning to reverse.
        # ───────────────────────────────────────────────────────────────────
        if composite_state is not None:
            _choch = getattr(composite_state, "choch_detected", False)
            if not _choch:
                logger.info(
                    "[MR Mode2] %s: no CHoCH detected in SECONDARY state — "
                    "counter-trend entry blocked (market has not shown "
                    "exhaustion signal yet)",
                    self.asset,
                )
                return 0, 0.0
            logger.info(
                "[MR Mode2] %s: CHoCH confirmed in SECONDARY state — "
                "exhaustion signal present, proceeding",
                self.asset,
            )

        # ── TRENDING veto: counter-trend in a strong trend is fatal ──────────
        if composite_state is not None:
            _rc = getattr(composite_state, "range_classification", None)
            if _rc == "TRENDING":
                _phase_cfg_rc = getattr(composite_state, "phase_config", {}) or {}
                if _phase_cfg_rc.get("mr_vetoes_as_dampeners_enabled", False):
                    # Unit 2: dampen instead of kill. Categorical (not a
                    # continuous over-threshold reading like vol_down_ratio),
                    # so a fixed 0.5x rather than a scaled multiplier.
                    self._rc_damp = 0.5
                    logger.info(
                        f"[MR Mode2] {self.asset}: range_classification=TRENDING → "
                        f"DAMPEN x0.50 (flag ON)"
                    )
                else:
                    logger.info(
                        f"[MR Mode2] {self.asset}: range_classification=TRENDING → VETO"
                    )
                    return 0, 0.0

        # Direction from Livermore state
        lsm_state = getattr(composite_state, "livermore_state_1h", None) if composite_state else None
        if lsm_state == "SECONDARY_RETRACEMENT":
            direction = 1
        elif lsm_state == "SECONDARY_REBOUND":
            direction = -1
        else:
            return 0, 0.0

        # ── ADX gate ──────────────────────────────────────────────────────
        adx_max = int(cfg["adx_max"])
        adx_val = float(features["adx"].values[-1]) if not pd.isna(features["adx"].values[-1]) else 99.0
        if adx_val >= adx_max:
            logger.info(f"[MR Mode2] {self.asset}: ADX={adx_val:.1f} ≥ {adx_max} (need <{adx_max}) → 0")
            return 0, 0.0

        # ── 4 of 4 optional conditions ────────────────────────────────────
        opt_count = self._count_optional(features, direction)
        min_opt   = int(cfg["optional_min_count"])   # 4
        if opt_count < min_opt:
            _vc = self._check_vol_contraction(features, direction)
            _hd = self._check_hidden_divergence(features, direction)
            _bc = self._check_bb_contraction(features, direction)
            _mp = self._check_ma_proximity(features, direction)
            logger.info(
                f"[MR Mode2] {self.asset}: opt={opt_count}/{min_opt} (need all {min_opt}) → 0 "
                f"[vol={'✓' if _vc else '✗'} "
                f"hdiv={'✓' if _hd else '✗'} "
                f"bbc={'✓' if _bc else '✗'} "
                f"map={'✓' if _mp else '✗'}]"
            )
            return 0, 0.0

        # ── BB closed back inside ─────────────────────────────────────────
        if cfg.get("bb_inside_required", True):
            bb_pos_arr = features["bb_position"].values
            if len(bb_pos_arr) < 5:
                return 0, 0.0
            if direction == 1:
                # Was outside lower band (bb_pos < 0) and has now closed back inside
                was_outside = any(float(v) < 0.0 for v in bb_pos_arr[-5:-1])
                if not was_outside:
                    _bb_recent = [f"{float(v):.2f}" for v in bb_pos_arr[-5:]]
                    logger.info(
                        f"[MR Mode2] {self.asset}: BB not recently outside lower band "
                        f"(long) — bb_pos last 5: {_bb_recent} → 0"
                    )
                    return 0, 0.0
            elif direction == -1:
                was_outside = any(float(v) > 1.0 for v in bb_pos_arr[-5:-1])
                if not was_outside:
                    _bb_recent = [f"{float(v):.2f}" for v in bb_pos_arr[-5:]]
                    logger.info(
                        f"[MR Mode2] {self.asset}: BB not recently outside upper band "
                        f"(short) — bb_pos last 5: {_bb_recent} → 0"
                    )
                    return 0, 0.0

        # ── BTC-specific z-score gates ────────────────────────────────────
        if self.asset in ("BTC", "BTCUSDT"):
            z_score = self._compute_bb_zscore(features)
            if direction == 1:
                z_thr = float(cfg["btc_long_zscore_threshold"])   # -2.0
                if z_score >= z_thr:
                    logger.info(
                        f"[MR Mode2 BTC] LONG z={z_score:.2f} ≥ {z_thr} (need < {z_thr}) → 0"
                    )
                    return 0, 0.0
            elif direction == -1:
                z_thr = float(cfg["btc_short_zscore_threshold"])  # 3.5
                if z_score <= z_thr:
                    logger.info(
                        f"[MR Mode2 BTC] SHORT z={z_score:.2f} ≤ {z_thr} (need > {z_thr}) → 0"
                    )
                    return 0, 0.0

        # ── Confidence: lower ADX = more range-bound = stronger MR edge ──
        adx_factor = max(0.0, (adx_max - adx_val) / adx_max)
        confidence = float(min(1.0, 0.60 + adx_factor * 0.15))
        # Unit 2: apply the range_classification=TRENDING dampener if set
        # above, then reset — same leak-prevention rule as Mode 1's _vdr_damp.
        confidence = confidence * getattr(self, "_rc_damp", 1.0)
        self._rc_damp = 1.0

        _dir_str = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"[MR Mode2] {self.asset}: {_dir_str} "
            f"adx={adx_val:.1f} opt=4/4 conf={confidence:.2f}"
        )
        return direction, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3: Climax Fade
    # ─────────────────────────────────────────────────────────────────────────

    def _mode3_climax_fade(
        self, df: pd.DataFrame, composite_state
    ) -> tuple:
        """
        Mode 3 — Climax Fade (MAIN_UP / MAIN_DOWN).

        MAIN_UP   → SHORT  (fade the overextended upleg)
        MAIN_DOWN → LONG   (fade the selling climax downleg)

        [MANDATORY] leg_stretch_ratio > 1.5× (current move from Livermore anchor
                    vs median window-range over last 50 bars)
        [MANDATORY] price > 2.5×ATR from Livermore anchor
        [MANDATORY] High-rank reversal candle:
                      SHORT: bearish engulfing or evening star
                      LONG:  bullish engulfing or morning star
        [MANDATORY] Above-average volume on signal bar
        NOTE: Target is EMA20 within MAIN state — this is NOT a reversal call.
        """
        cfg      = self._mr3_cfg["mode3"]
        features = self.generate_features(df.tail(260))
        if len(features) < 60:
            return 0, 0.0

        # Direction from Livermore state
        lsm_state = getattr(composite_state, "livermore_state_1h", None) if composite_state else None
        if lsm_state == "MAIN_UP":
            direction = -1  # SHORT: fade the climax
        elif lsm_state == "MAIN_DOWN":
            direction = 1   # LONG: fade the selling climax
        else:
            return 0, 0.0

        close    = features["close"].values
        high_arr = features["high"].values
        low_arr  = features["low"].values
        atr_arr  = features["atr"].values
        atr_now  = float(atr_arr[-1]) if not pd.isna(atr_arr[-1]) else 0.0
        if atr_now <= 0:
            return 0, 0.0

        current_close = float(close[-1])

        # ── Pull Livermore anchor from composite state ─────────────────────
        anchor = None
        if composite_state is not None:
            if lsm_state == "MAIN_UP":
                anchor = getattr(composite_state, "livermore_anchor_main_up_max", None)
            elif lsm_state == "MAIN_DOWN":
                anchor = getattr(composite_state, "livermore_anchor_main_down_min", None)

        # ── leg_stretch_ratio: current leg vs typical leg ─────────────────
        stretch_min = float(cfg["leg_stretch_ratio_min"])
        leg_stretch_ok = False

        if anchor is not None and float(anchor) > 0:
            current_leg = abs(current_close - float(anchor))
            # Typical leg: median of range across N windows of W bars each
            lb  = min(int(cfg["leg_stretch_lookback"]), len(close) - 1)
            win = int(cfg["leg_stretch_window"])
            _h  = high_arr[-lb:]
            _l  = low_arr[-lb:]
            _window_ranges = []
            for _w in range(0, lb - win, win):
                _window_ranges.append(float(np.max(_h[_w:_w+win]) - np.min(_l[_w:_w+win])))
            if len(_window_ranges) >= 3:
                typical_leg = float(np.median(_window_ranges))
            else:
                typical_leg = atr_now * 8.0   # Fallback: 8×ATR
            if typical_leg <= 0:
                typical_leg = atr_now * 8.0

            leg_stretch_ratio = current_leg / typical_leg
            if leg_stretch_ratio >= stretch_min:
                leg_stretch_ok = True
            else:
                logger.info(
                    f"[MR Mode3] {self.asset}: leg_stretch={leg_stretch_ratio:.2f} < {stretch_min} "
                    f"(leg={current_leg:.4f}, typical={typical_leg:.4f}) → 0"
                )
                return 0, 0.0
        # If anchor unavailable, leg check bypassed (dist_ok becomes sole gate)

        # ── Price distance from anchor: > 2.5×ATR ─────────────────────────
        _atr_mult = float(cfg["price_anchor_atr_mult"])
        if anchor is not None and float(anchor) > 0:
            _anchor_dist_atr = abs(current_close - float(anchor)) / atr_now
        else:
            # Fallback: distance from EMA50
            _ema50 = float(features["ema_50"].values[-1]) if not pd.isna(features["ema_50"].values[-1]) else current_close
            _anchor_dist_atr = abs(current_close - _ema50) / atr_now

        if _anchor_dist_atr <= _atr_mult:
            logger.info(
                f"[MR Mode3] {self.asset}: anchor_dist={_anchor_dist_atr:.2f}×ATR ≤ {_atr_mult} "
                f"(close={current_close:.4f}, anchor={anchor}) → 0"
            )
            return 0, 0.0

        # ── High-rank reversal candle ──────────────────────────────────────
        # Checked over the last candle_lookback_bars bars (default 3) so the
        # gate doesn't require the pattern to land on the exact current bar.
        # Only engulfing and morning/evening star — shooting star excluded
        # (lower historical reliability ~59%).
        _n  = min(len(features), 30)
        _o  = features["open"].values[-_n:]
        _h  = features["high"].values[-_n:]
        _l  = features["low"].values[-_n:]
        _c  = features["close"].values[-_n:]

        _candle_lb = max(1, int(cfg.get("candle_lookback_bars", 1)))
        reversal_ok  = False
        reversal_bar = None   # track which bar triggered (for logging)

        if direction == -1:   # MAIN_UP → SHORT
            _eng = ta.CDLENGULFING(_o, _h, _l, _c)
            _eve = ta.CDLEVENINGSTAR(_o, _h, _l, _c, penetration=0.3)
            for _k in range(_candle_lb):
                if int(_eng[-(_k + 1)]) < 0 or int(_eve[-(_k + 1)]) < 0:
                    reversal_ok  = True
                    reversal_bar = _k   # 0 = current bar, 1 = one bar ago, etc.
                    break
        elif direction == 1:  # MAIN_DOWN → LONG
            _eng  = ta.CDLENGULFING(_o, _h, _l, _c)
            _morn = ta.CDLMORNINGSTAR(_o, _h, _l, _c, penetration=0.3)
            for _k in range(_candle_lb):
                if int(_eng[-(_k + 1)]) > 0 or int(_morn[-(_k + 1)]) > 0:
                    reversal_ok  = True
                    reversal_bar = _k
                    break

        if not reversal_ok:
            logger.info(
                f"[MR Mode3] {self.asset}: no high-rank reversal candle "
                f"(engulfing/star) in last {_candle_lb} bars → 0"
            )
            return 0, 0.0

        # ── Above-average volume ───────────────────────────────────────────
        if cfg.get("require_above_avg_volume", True) and "volume" in features.columns:
            try:
                _vol_lb   = int(cfg["volume_avg_lookback"])
                _vol_avg  = float(np.mean(features["volume"].values[-_vol_lb-1:-1]))
                _vol_now  = float(features["volume"].values[-1])
                if _vol_avg > 0 and _vol_now <= _vol_avg:
                    logger.info(
                        f"[MR Mode3] {self.asset}: volume {_vol_now:.0f} ≤ avg {_vol_avg:.0f} "
                        f"(need above-avg) → 0"
                    )
                    return 0, 0.0
            except Exception:
                pass   # Missing volume data: do not block

        # ── Confidence ─────────────────────────────────────────────────────
        # Scales with how overextended the anchor distance is
        dist_factor = min(1.5, _anchor_dist_atr / _atr_mult)
        confidence  = float(min(1.0, 0.60 + (dist_factor - 1.0) * 0.15))

        _dir_str  = "SHORT" if direction == -1 else "LONG"
        _bar_str  = "current bar" if reversal_bar == 0 else f"{reversal_bar} bar(s) ago"
        logger.info(
            f"[MR Mode3] {self.asset}: {_dir_str} climax fade "
            f"anchor_dist={_anchor_dist_atr:.1f}×ATR "
            f"reversal_candle={_bar_str} conf={confidence:.2f}"
        )
        return direction, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY SCORECARD  (fallback during Livermore warmup)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_swing_pivot(
        self, values: np.ndarray, direction: str, lookback: int = 60, n_bars: int = 5
    ) -> int:
        """Find a significant swing high/low via N-bar comparison window."""
        if len(values) < lookback:
            lookback = len(values)
        for j in range(n_bars, lookback - n_bars):
            idx = len(values) - j - 1
            if idx < n_bars or idx + n_bars >= len(values):
                continue
            window = values[idx - n_bars: idx + n_bars + 1]
            if direction == "low"  and values[idx] == np.min(window): return idx
            if direction == "high" and values[idx] == np.max(window): return idx
        return -1

    def _check_divergence_legacy(
        self, df: pd.DataFrame, signal: int, period: int = 60
    ) -> bool:
        """Classic price/RSI divergence check (used by legacy scorecard only)."""
        try:
            if len(df) < period:
                return False
            close    = df["close"].values
            rsi_vals = df["rsi"].values
            if signal == 1:
                ci = self._find_swing_pivot(close, "low", lookback=20, n_bars=5)
                if ci == -1: return False
                pi = self._find_swing_pivot(close, "low", lookback=period, n_bars=5)
                if pi == -1 or pi >= ci - 5: return False
                return close[ci] < close[pi] and rsi_vals[ci] > rsi_vals[pi]
            elif signal == -1:
                ci = self._find_swing_pivot(close, "high", lookback=20, n_bars=5)
                if ci == -1: return False
                pi = self._find_swing_pivot(close, "high", lookback=period, n_bars=5)
                if pi == -1 or pi >= ci - 5: return False
                return close[ci] > close[pi] and rsi_vals[ci] < rsi_vals[pi]
        except Exception:
            return False
        return False

    def _legacy_scorecard(
        self, df: pd.DataFrame, df_4h: pd.DataFrame = None
    ) -> tuple:
        """
        Macro Reversion Scorecard — active during Livermore warmup period only.
        Mirrors the pre-Phase-3A logic exactly to avoid any warmup-period gap.
        """
        try:
            features = self.generate_features(df.tail(150))
            if len(features) < 100:
                return 0, 0.0

            close    = features["close"].values
            high_arr = features["high"].values
            low_arr  = features["low"].values
            bb_pos   = features["bb_position"].values
            bb_upper = features["bb_upper"].values
            bb_lower = features["bb_lower"].values
            rsi_arr  = features["rsi"].values
            atr_arr  = features["atr"].values
            ema50    = features["ema_50"].values

            cur = float(close[-1])
            vel_drop = float(close[-4]) - cur
            vel_rise = cur - float(close[-4])
            atr_thr  = 4.0 * float(atr_arr[-1])

            sl = 0; ss = 0  # score_long, score_short

            # PILLAR 1: Stretch (2 pts)
            sl_full  = (float(ema50[-1]) - cur > 1.5 * float(atr_arr[-1])) or (cur < float(bb_lower[-1]))
            ss_full  = (cur - float(ema50[-1]) > 1.5 * float(atr_arr[-1])) or (cur > float(bb_upper[-1]))
            if sl_full: sl += 2
            if ss_full: ss += 2
            if not sl_full and float(ema50[-1]) - cur > 0.8 * float(atr_arr[-1]): sl += 1
            if not ss_full and cur - float(ema50[-1]) > 0.8 * float(atr_arr[-1]): ss += 1

            # PILLAR 1.5: RSI extreme (1 pt)
            if float(bb_pos[-1]) < 0.15 and float(rsi_arr[-1]) < 35: sl += 1
            if float(bb_pos[-1]) > 0.85 and float(rsi_arr[-1]) > 65: ss += 1

            # PILLAR 2: Divergence (1 pt)
            if self._check_divergence_legacy(features, signal=1):  sl += 1
            if self._check_divergence_legacy(features, signal=-1): ss += 1

            # PILLAR 3: Liquidity sweep (1 pt)
            _lb_low  = low_arr[-31:-1]
            _lb_high = high_arr[-31:-1]
            if len(_lb_low) > 0:
                if cur < float(np.min(_lb_low)):  sl += 1
                if cur > float(np.max(_lb_high)): ss += 1

            # PILLAR 4: Exhaustion candle at extremes (1 pt)
            if float(bb_pos[-1]) < 0.15 or float(bb_pos[-1]) > 0.85:
                _o = features["open"].values
                _h = features["high"].values
                _l = features["low"].values
                _c = features["close"].values
                for _fn in (ta.CDLHAMMER, ta.CDLDOJI, ta.CDLENGULFING):
                    _res = _fn(_o, _h, _l, _c)
                    if int(_res[-1]) > 0: sl += 1
                    if int(_res[-1]) < 0: ss += 1

            logger.info(
                f"[MR LEGACY] {self.asset}: long={sl}/6 short={ss}/6 (need ≥3, Livermore warmup)"
            )

            sig, conf = 0, 0.0
            if sl >= 3.0 and vel_drop <= atr_thr:
                sig  =  1
                conf = min(1.0, 0.6 + (sl - 3.0) * 0.2)
            elif ss >= 3.0 and vel_rise <= atr_thr:
                sig  = -1
                conf = min(1.0, 0.6 + (ss - 3.0) * 0.2)
            return sig, conf

        except Exception as e:
            logger.error(f"[{self.name}] Legacy scorecard error: {e}")
            return 0, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN SIGNAL ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(
        self,
        df: pd.DataFrame,
        df_4h: pd.DataFrame = None,
        composite_state=None,
    ) -> tuple:
        """
        Phase 3A: Route to the appropriate MR mode based on Livermore 1H state.

        State routing:
          NATURAL_RETRACEMENT   → Mode 1 (Pullback Completion, LONG only)
          NATURAL_REBOUND       → zero  (MR silent zone; LONGs blocked by Phase 2 HVL)
          SECONDARY_RETRACEMENT → Mode 2 (Counter-Trend, LONG)
          SECONDARY_REBOUND     → Mode 2 (Counter-Trend, SHORT)
          MAIN_UP               → Mode 3 (Climax Fade, SHORT)
          MAIN_DOWN             → Mode 3 (Climax Fade, LONG)
          None (warmup)         → Legacy scorecard fallback

        Signature is backward-compatible: composite_state is optional.
        """
        if len(df) < self.get_warmup_period():
            return 0, 0.0

        try:
            lsm_state = None
            if composite_state is not None:
                lsm_state = getattr(composite_state, "livermore_state_1h", None)

            # ── MR routing diagnostic (Issue 2 Step 1 — always visible at INFO) ──
            _mode_label = {
                "NATURAL_RETRACEMENT":   "Mode1(Pullback/LONG)",
                "NATURAL_REBOUND":       "SILENT_ZONE",
                "SECONDARY_RETRACEMENT": "Mode2(Counter/LONG)",
                "SECONDARY_REBOUND":     "Mode2(Counter/SHORT)",
                "MAIN_UP":               "Mode3(Fade/SHORT)",
                "MAIN_DOWN":             "Mode3(Fade/LONG)",
            }
            # Plain-English what's-actually-happening line, added 2026-06-23 so the
            # GATE line is self-explanatory without needing to know the codebase.
            # Cosmetic only — does not affect routing/decision logic below.
            _mode_desc = {
                "NATURAL_RETRACEMENT":   "buying the pullback in an uptrend (needs Wyckoff spring)",
                "NATURAL_REBOUND":       "silent zone — no MR setup, longs blocked",
                "SECONDARY_RETRACEMENT": "counter-trend LONG — fading a deep pullback, expecting a bounce",
                "SECONDARY_REBOUND":     "counter-trend SHORT — fading a deep bounce, expecting reversion down",
                "MAIN_UP":               "climax fade SHORT — fading an overextended up-leg",
                "MAIN_DOWN":             "climax fade LONG — fading an overextended down-leg",
            }
            _label = _mode_label.get(lsm_state, "LEGACY(warmup)") if lsm_state else "LEGACY(warmup)"
            _desc  = _mode_desc.get(lsm_state, "Livermore state unavailable — using legacy 6-pillar scorecard")
            logger.info(
                f"[MR GATE] {self.asset}: LSM 1H={lsm_state} → {_label} — {_desc}"
            )

            if lsm_state == "NATURAL_RETRACEMENT":
                return self._mode1_pullback_completion(df, composite_state)

            elif lsm_state == "NATURAL_REBOUND":
                # Unit 2: rebound-short. When flagged ON, MR may short a
                # rebound that is behaving like a reversal (the mirror of the
                # NATURAL_RETRACEMENT long-spring). Longs remain blocked here.
                # When OFF, behaviour is unchanged (silent zone, return 0).
                _phase_cfg = getattr(composite_state, "phase_config", {}) or {}
                if _phase_cfg.get("mr_rebound_short_enabled", False):
                    return self._mode1_pullback_completion(df, composite_state, side="short")
                # Flag OFF → original silent-zone behaviour.
                return 0, 0.0

            elif lsm_state in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
                _phase_cfg = getattr(composite_state, "phase_config", {}) or {}
                if not _phase_cfg.get("mr_mode2_counter_trend_enabled", False):
                    logger.debug(f"[MR GATE] {self.asset}: Mode 2 disabled by phase_config — holding.")
                    return 0, 0.0
                return self._mode2_counter_trend(df, composite_state)

            elif lsm_state in ("MAIN_UP", "MAIN_DOWN"):
                _phase_cfg = getattr(composite_state, "phase_config", {}) or {}
                if not _phase_cfg.get("mr_mode3_climax_fade_enabled", False):
                    logger.debug(f"[MR GATE] {self.asset}: Mode 3 disabled by phase_config — holding.")
                    return 0, 0.0
                return self._mode3_climax_fade(df, composite_state)

            else:
                # Livermore state unavailable (warmup or None) → legacy fallback
                return self._legacy_scorecard(df, df_4h)

        except Exception as e:
            logger.error(f"[{self.name}] Signal error: {e}", exc_info=True)
            return 0, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE LABELS  (training — legacy scorecard, no composite state)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_labels(
        self,
        df: pd.DataFrame,
        df_4h: pd.DataFrame = None,
        pattern_miner=None,
    ) -> pd.Series:
        """
        Training labels via legacy macro-reversion scorecard.
        Phase 3A mode-aware labeling requires per-bar Livermore state
        (available in Phase 3B backtest pass or standalone calibration run).
        """
        df     = df.copy()
        labels = pd.Series(0, index=df.index)
        df     = self.generate_features(df)

        close    = df["close"].values
        high_arr = df["high"].values
        low_arr  = df["low"].values
        bb_upper = df["bb_upper"].values
        bb_lower = df["bb_lower"].values
        bb_pos   = df["bb_position"].values
        rsi_arr  = df["rsi"].values
        atr_arr  = df["atr"].values
        ema50    = df["ema_50"].values

        signal_count = {"buy": 0, "sell": 0, "hold": 0}

        for i in range(100, len(df) - self.reversion_window - 1):
            if pd.isna(ema50[i]) or pd.isna(rsi_arr[i]):
                continue

            cur      = float(close[i])
            vel_drop = float(close[i-3]) - cur
            vel_rise = cur - float(close[i-3])
            atr_thr  = 4.0 * float(atr_arr[i])

            sl = ss = 0

            # PILLAR 1: Stretch
            sl_full  = (float(ema50[i]) - cur > 1.5 * float(atr_arr[i])) or (cur < float(bb_lower[i]))
            ss_full  = (cur - float(ema50[i]) > 1.5 * float(atr_arr[i])) or (cur > float(bb_upper[i]))
            if sl_full: sl += 2
            if ss_full: ss += 2
            if not sl_full and float(ema50[i]) - cur > 0.8 * float(atr_arr[i]): sl += 1
            if not ss_full and cur - float(ema50[i]) > 0.8 * float(atr_arr[i]): ss += 1

            # PILLAR 1.5: RSI extreme
            if float(bb_pos[i]) < 0.15 and float(rsi_arr[i]) < 35: sl += 1
            if float(bb_pos[i]) > 0.85 and float(rsi_arr[i]) > 65: ss += 1

            # PILLAR 2: Divergence
            _slc = df.iloc[:i+1]
            if self._check_divergence_legacy(_slc, signal=1):  sl += 1
            if self._check_divergence_legacy(_slc, signal=-1): ss += 1

            # PILLAR 3: Sweep
            _lb_l = low_arr[max(0, i-30):i]
            _lb_h = high_arr[max(0, i-30):i]
            if len(_lb_l) > 0:
                if cur < float(np.min(_lb_l)):  sl += 1
                if cur > float(np.max(_lb_h)): ss += 1

            # PILLAR 4: Exhaustion at extremes
            if float(bb_pos[i]) < 0.15 or float(bb_pos[i]) > 0.85:
                _o = df["open"].values[:i+1]
                _h = df["high"].values[:i+1]
                _l = df["low"].values[:i+1]
                _c = df["close"].values[:i+1]
                for _fn in (ta.CDLHAMMER, ta.CDLDOJI, ta.CDLENGULFING):
                    _res = _fn(_o, _h, _l, _c)
                    if int(_res[-1]) > 0: sl += 1
                    if int(_res[-1]) < 0: ss += 1

            # Future return validation
            _atr_pct   = float(atr_arr[i]) / max(cur, 1e-10)
            min_return = max(self.min_return_threshold, 0.30 * _atr_pct)
            _fut       = close[i+1: i+1+5]
            _fut_ret   = (float(np.mean(_fut)) - cur) / cur if len(_fut) > 0 else 0.0

            if sl >= 3.0 and vel_drop <= atr_thr and _fut_ret > min_return:
                labels.iloc[i] = 1;  signal_count["buy"] += 1
            elif ss >= 3.0 and vel_rise <= atr_thr and _fut_ret < -min_return:
                labels.iloc[i] = -1; signal_count["sell"] += 1

        labels.iloc[-self.reversion_window:] = 0
        signal_count["hold"] = len(labels) - signal_count["buy"] - signal_count["sell"]
        total = len(labels)
        logger.info(f"[{self.name}] Label distribution (legacy scorecard):")
        logger.info(f"  SELL: {signal_count['sell']:>5} ({signal_count['sell']/total*100:>5.2f}%)")
        logger.info(f"  HOLD: {signal_count['hold']:>5} ({signal_count['hold']/total*100:>5.2f}%)")
        logger.info(f"  BUY:  {signal_count['buy']:>5}  ({signal_count['buy']/total*100:>5.2f}%)")
        return labels
