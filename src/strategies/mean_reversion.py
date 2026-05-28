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

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # SPRING DETECTION  (Mode 1 mandatory)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_spring(self, df: pd.DataFrame) -> tuple:
        """
        Wyckoff spring: a bar's wick sweeps below a prior swing low then closes
        back above it within spring_recovery_max_bars of the current bar.

        Penetration must be 0.5–5% of the swing low level.
        Current-bar volume must be lower than the spring bar's volume
        (spring bar = selling climax; entry bar = quiet absorption).

        Returns: (found: bool, strength: float 0–1)
        """
        cfg = self._mr3_cfg["mode1"]
        min_pen  = cfg["spring_min_penetration"]    # 0.005
        max_pen  = cfg["spring_max_penetration"]    # 0.050
        max_bars = cfg["spring_recovery_max_bars"]  # 3
        swing_lb = cfg["spring_swing_lookback"]     # 20

        min_total = swing_lb + max_bars + 2
        if len(df) < min_total:
            return False, 0.0

        close  = df["close"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # Prior swing low is established from the window BEFORE the search bars
        # so we don't confuse the spring bar itself as the swing low.
        _win_end   = -(max_bars + 1)           # index of last bar before search window
        _win_start = _win_end - swing_lb
        _prior_arr = low[_win_start:_win_end]
        if len(_prior_arr) < 5:
            return False, 0.0
        prior_swing_low = float(np.min(_prior_arr))
        if prior_swing_low <= 0:
            return False, 0.0

        # Scan the last max_bars bars (not including current bar) for the spring
        for k in range(1, max_bars + 1):
            bar_idx   = -(k + 1)   # k positions before current bar
            if abs(bar_idx) > len(df):
                continue

            bar_low   = float(low[bar_idx])
            bar_close = float(close[bar_idx])

            # Condition 1: wick swept below swing low
            if bar_low >= prior_swing_low:
                continue

            # Condition 2: bar closed BACK ABOVE the swept level
            if bar_close <= prior_swing_low:
                continue

            # Condition 3: penetration in [0.5%, 5%]
            penetration = (prior_swing_low - bar_low) / prior_swing_low
            if not (min_pen <= penetration <= max_pen):
                continue

            # Condition 4: current-bar volume < spring-bar volume
            if volume is not None:
                spring_vol  = float(volume[bar_idx])
                current_vol = float(volume[-1])
                if spring_vol > 0 and current_vol >= spring_vol:
                    continue

            # Spring confirmed.  Strength peaks near 2.5% penetration.
            _optimal  = 0.025
            strength  = max(0.30, 1.0 - abs(penetration - _optimal) / _optimal)
            return True, float(min(strength, 1.0))

        return False, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIONAL CONDITIONS  (shared by Mode 1 and Mode 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_vol_contraction(self, df: pd.DataFrame, direction: int) -> bool:
        """
        Volume Contraction:
          - Down-close bar volumes < 80% of 20-bar average (for longs)
          - No single bar in the last 5 exceeds 150% of average
        Signals quiet re-accumulation, not continued distribution.
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
                bv    = float(vol[j])
                is_dn = float(close[j]) < float(open_[j])
                if is_dn and bv > avg_vol * avg_pct:
                    return False    # Down bar with above-threshold volume
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

    def _check_bb_contraction(self, df: pd.DataFrame, direction: int) -> bool:
        """
        BB Contraction: price is in the lower 20% (long) or upper 80%+ (short)
        of the Bollinger Band while bandwidth is stable or declining.
        Confirms the price is stretched AND momentum is bleeding off.
        """
        try:
            if len(df) < self.bb_period + 5:
                return False
            cfg      = self._mr3_cfg["mode1"]
            zone_pct = cfg["bb_lower_zone_threshold"]   # 0.20
            bb_pos   = df["bb_position"].values
            bb_wid   = df["bb_width_norm"].values
            if pd.isna(bb_pos[-1]) or pd.isna(bb_wid[-1]):
                return False
            pos       = float(bb_pos[-1])
            bw_prev5  = bb_wid[-6:-1]
            bw_prev5  = bw_prev5[~np.isnan(bw_prev5)]
            if len(bw_prev5) == 0:
                return False
            bw_mean     = float(np.mean(bw_prev5))
            bw_declining = float(bb_wid[-1]) <= bw_mean

            if direction == 1:
                return pos < zone_pct and bw_declining
            elif direction == -1:
                return pos > (1.0 - zone_pct) and bw_declining
        except Exception:
            return False
        return False

    def _check_ma_proximity(self, df: pd.DataFrame, direction: int) -> bool:
        """
        MA Proximity: current price is within 0.5×ATR of EMA50 or EMA200.
        Classic Wyckoff Last Point of Support — price testing a major MA.
        """
        try:
            if len(df) < 5:
                return False
            cfg    = self._mr3_cfg["mode1"]
            mult   = cfg["ma_proximity_atr_mult"]    # 0.5
            close  = float(df["close"].values[-1])
            atr_v  = float(df["atr"].values[-1]) if not pd.isna(df["atr"].values[-1]) else 0.0
            if atr_v <= 0:
                return False
            threshold = mult * atr_v
            ema50  = df["ema_50"].values[-1]  if "ema_50"  in df.columns else None
            ema200 = df["ema_200"].values[-1] if "ema_200" in df.columns else None
            if ema50  is not None and not pd.isna(ema50)  and abs(close - float(ema50))  <= threshold:
                return True
            if ema200 is not None and not pd.isna(ema200) and abs(close - float(ema200)) <= threshold:
                return True
        except Exception:
            return False
        return False

    def _count_optional(self, df: pd.DataFrame, direction: int) -> int:
        """Count how many of the 4 optional conditions are met."""
        return sum([
            self._check_vol_contraction(df, direction),
            self._check_hidden_divergence(df, direction),
            self._check_bb_contraction(df, direction),
            self._check_ma_proximity(df, direction),
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
        self, df: pd.DataFrame, composite_state
    ) -> tuple:
        """
        Mode 1 — Pullback Completion (1H NATURAL_RETRACEMENT, LONG only).

        [MANDATORY] Spring detected in last 1–3 bars
        [2 of 4]    Optional: vol_contraction, hidden_div, bb_contraction, ma_proximity
        [VETO]      vol_down_ratio > 1.2
        [MODIFIER]  BTC near 4H EMA200 → −0.10 confidence
        """
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
                logger.debug(
                    f"[MR Mode1] {self.asset}: vol_down_ratio={_vdr:.2f} > {_veto_thr} → VETO"
                )
                return 0, 0.0

        # ── Spring (mandatory) ─────────────────────────────────────────────
        spring_ok, spring_strength = self._detect_spring(features)
        if not spring_ok:
            logger.debug(f"[MR Mode1] {self.asset}: no spring → 0")
            return 0, 0.0

        # ── Optional conditions (2 of 4 required) ─────────────────────────
        opt_count = self._count_optional(features, direction=1)
        min_opt   = cfg["optional_min_count"]
        if opt_count < min_opt:
            logger.debug(
                f"[MR Mode1] {self.asset}: spring OK but opt={opt_count}<{min_opt} → 0"
            )
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
        confidence = float(min(1.0, base_conf + extra_opt * 0.05 + conf_mod))

        logger.info(
            f"[MR Mode1] {self.asset}: LONG "
            f"spring_str={spring_strength:.2f} opt={opt_count}/4 conf={confidence:.2f}"
        )
        return 1, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2: Counter-Trend
    # ─────────────────────────────────────────────────────────────────────────

    def _mode2_counter_trend(
        self, df: pd.DataFrame, composite_state
    ) -> tuple:
        """
        Mode 2 — Counter-Trend (SECONDARY states only).

        SECONDARY_RETRACEMENT → LONG  (counter the deep pullback)
        SECONDARY_REBOUND     → SHORT (counter the deep bounce)

        [MANDATORY] ADX < 25
        [MANDATORY] 4 of 4 optional conditions
        [MANDATORY] BB has closed back inside bands (was recently outside)
        [BTC only]  LONG: BB z-score < −2.0; SHORT: z-score > +3.5
        """
        cfg      = self._mr3_cfg["mode2"]
        features = self.generate_features(df.tail(260))
        if len(features) < 50:
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
            logger.debug(f"[MR Mode2] {self.asset}: ADX={adx_val:.1f} ≥ {adx_max} → 0")
            return 0, 0.0

        # ── 4 of 4 optional conditions ────────────────────────────────────
        opt_count = self._count_optional(features, direction)
        min_opt   = int(cfg["optional_min_count"])   # 4
        if opt_count < min_opt:
            logger.debug(
                f"[MR Mode2] {self.asset}: opt={opt_count} < {min_opt} → 0"
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
                    logger.debug(f"[MR Mode2] {self.asset}: BB not recently outside (long) → 0")
                    return 0, 0.0
            elif direction == -1:
                was_outside = any(float(v) > 1.0 for v in bb_pos_arr[-5:-1])
                if not was_outside:
                    logger.debug(f"[MR Mode2] {self.asset}: BB not recently outside (short) → 0")
                    return 0, 0.0

        # ── BTC-specific z-score gates ────────────────────────────────────
        if self.asset in ("BTC", "BTCUSDT"):
            z_score = self._compute_bb_zscore(features)
            if direction == 1:
                z_thr = float(cfg["btc_long_zscore_threshold"])   # -2.0
                if z_score >= z_thr:
                    logger.debug(
                        f"[MR Mode2 BTC] LONG z={z_score:.2f} not < {z_thr} → 0"
                    )
                    return 0, 0.0
            elif direction == -1:
                z_thr = float(cfg["btc_short_zscore_threshold"])  # 3.5
                if z_score <= z_thr:
                    logger.debug(
                        f"[MR Mode2 BTC] SHORT z={z_score:.2f} not > {z_thr} → 0"
                    )
                    return 0, 0.0

        # ── Confidence: lower ADX = more range-bound = stronger MR edge ──
        adx_factor = max(0.0, (adx_max - adx_val) / adx_max)
        confidence = float(min(1.0, 0.60 + adx_factor * 0.15))

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
