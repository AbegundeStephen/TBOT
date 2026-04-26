"""
Enhanced PerformanceWeightedAggregator with AI Safety Features
===============================================================
IMPROVEMENTS:
- AI circuit breaker to prevent over-filtering
- Regime context passed to AI validator
- Better cold-start handling for regime detection
- AI performance tracking
- Graceful degradation if AI fails
"""

import pandas as pd
from src.utils.trap_filter import validate_candle_structure
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
Enhanced Signal Aggregator with World-Class Filters
====================================================
Adds Governor + Volatility + Sniper checks to existing aggregator
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
        config: Dict = None,
        ai_validator=None,
        mtf_integration=None,  # For Governor access
        enable_world_class_filters: bool = True,
        enable_ai_circuit_breaker: bool = False,
        enable_detailed_logging: bool = False,
        strong_signal_bypass_threshold: float = 0.70,
        use_macro_governor: bool = True,
        use_gatekeeper: bool = True
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()
        self.use_macro_governor = use_macro_governor
        self.use_gatekeeper = use_gatekeeper

        # Initialize regime tracking
        self.previous_regime = None
        self.regime_initialized = False

        # Logging and Thresholds
        self.detailed_logging = enable_detailed_logging
        self.strong_signal_bypass = strong_signal_bypass_threshold

        # ================================================================
        # AI VALIDATOR SETUP
        # ================================================================
        self.ai_validator = None
        self.ai_enabled = True
        
        # ✨ NEW: Store MTF integration for Governor
        self.mtf_integration = mtf_integration
        self.enable_filters = enable_world_class_filters

        # ✨ NEW: Initialize filter thresholds
        # Volatility gate: asset-class-specific defaults so FX pairs (EURJPY ATR ~0.07%)
        # aren't permanently blocked by a crypto-calibrated 0.35% threshold.
        #   FX  (EUR*, GBP*, USD*, JPY*, CHF*, AUD*, NZD*, CAD*) → 0.03% (0.0003)
        #   Metals / Indices (XAU, GOLD, USTEC, NAS*, SP5*, GER*) → 0.10% (0.0010)
        #   Crypto (BTC*, ETH*, BNB*)                             → 0.20% (0.0020)
        # Config override still wins if explicitly set.
        _asset_upper = asset_type.upper()
        _is_fx = any(
            fx in _asset_upper
            for fx in ("EUR", "GBP", "USD", "JPY", "CHF", "AUD", "NZD", "CAD")
        ) and "BTC" not in _asset_upper and "ETH" not in _asset_upper
        _is_crypto = any(c in _asset_upper for c in ("BTC", "ETH", "BNB", "SOL", "XRP"))
        _is_metals_indices = any(
            m in _asset_upper
            for m in ("XAU", "GOLD", "USTEC", "NAS", "SP5", "GER", "UK1", "NDX")
        )
        if _is_fx:
            _default_vol_threshold = 0.0003   # 0.03% — FX pairs
        elif _is_metals_indices:
            _default_vol_threshold = 0.0010   # 0.10% — metals / indices
        elif _is_crypto:
            _default_vol_threshold = 0.0020   # 0.20% — crypto (relaxed from original 0.35%)
        else:
            _default_vol_threshold = 0.0010   # 0.10% — safe generic fallback

        self.filter_thresholds = {
            'volatility_gate': config.get('world_class_filters', {}).get(
                'volatility_gate_threshold', _default_vol_threshold
            ),
            'sniper_confidence': config.get('world_class_filters', {}).get(
                'sniper_pattern_confidence', 0.60
            ),
            'min_profit': config.get('world_class_filters', {}).get(
                'min_profit_potential', 0.005
            ),
        }
        
        if self.enable_filters:
            logger.info(f"[FILTERS] World-Class Filters ENABLED for {asset_type}")
            logger.info(f"  Volatility Gate: {self.filter_thresholds['volatility_gate']:.3%}")
            logger.info(f"  Sniper Min:      {self.filter_thresholds['sniper_confidence']:.0%}")
            logger.info(f"  Min Profit:      {self.filter_thresholds['min_profit']:.2%}")
            
            
        if ai_validator is not None:
            try:
                # Validate AI is properly initialized
                assert hasattr(ai_validator, "sniper"), "Sniper not initialized"
                assert hasattr(ai_validator.sniper, "model"), "Model not loaded"
                assert hasattr(
                    ai_validator, "pattern_id_map"
                ), "Pattern mapping missing"
                assert len(ai_validator.pattern_id_map) > 0, "Pattern mapping empty"

                self.ai_validator = ai_validator
                self.ai_enabled = True

                logger.info(f"[AGGREGATOR] AI validation: ✓ ENABLED")
                logger.info(
                    f"[AGGREGATOR] Patterns loaded: {len(ai_validator.pattern_id_map)}"
                )

            except (AssertionError, AttributeError) as e:
                logger.error(f"[AGGREGATOR] AI validation setup failed: {e}")
                logger.warning("[AGGREGATOR] Continuing without AI validation")
                self.ai_validator = None
                self.ai_enabled = False

        # AI statistics tracking
        if self.ai_enabled:
            self.ai_stats = {
                "mr_signals_checked": 0,
                "mr_approved": 0,
                "mr_rejected": 0,
                "tf_signals_checked": 0,
                "tf_approved": 0,
                "tf_rejected": 0,
                "bypassed_strong_signal": 0,
            }

            # Circuit breaker configuration
            self.enable_circuit_breaker = enable_ai_circuit_breaker
            self.ai_rejection_window = deque(maxlen=50)
            self.ai_bypass_active = False
            self.ai_bypass_threshold = 0.85
            self.ai_bypass_cooldown = 0

            logger.info(
                f"[AGGREGATOR] AI circuit breaker: {'ENABLED' if enable_ai_circuit_breaker else 'DISABLED'}"
            )
            logger.info(
                f"[AGGREGATOR] Strong signal bypass: {self.strong_signal_bypass:.2%}"
            )
            logger.info(
                f"[AGGREGATOR] Detailed logging: {'ENABLED' if self.detailed_logging else 'DISABLED'}"
            )

        # Strategy weights — read from config, used for priority when multiple strategies fire
        # NOT for consensus voting. Hardcoded 0.50/0.50 was ignoring mean_reversion_weight: 0.0
        # in all presets, causing MR opposition penalty to bleed into every BTC TF score.
        # Will be updated after config merge below so we use a temporary default here.
        self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}

        # ================================================================
        # CONFIGURATION MERGE (Safety Fix)
        # ================================================================
        # 1. Define Defaults first (guarantees all keys exist)
        if self.asset_type == "BTC":
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.26,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "bull_buy_boost": 0.25,
                "bull_sell_penalty": 0.20,
                "bear_sell_boost": 0.25,
                "bear_buy_penalty": 0.30,
                "min_confidence_to_use": 0.08,
                "min_signal_quality": 0.28,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }
        else:  # GOLD (Default)
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.24,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "bull_buy_boost": 0.22,
                "bull_sell_penalty": 0.15,
                "bear_sell_boost": 0.22,
                "bear_buy_penalty": 0.28,
                "min_confidence_to_use": 0.06,
                "min_signal_quality": 0.25,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }
        
        # 2. Update with passed config (Merge instead of Overwrite)
        if config is not None:
            # This ensures keys missing from 'config' are filled by defaults above
            self.config.update(config)

        # 3. Wire strategy weights from merged config (T1.2 fix)
        # Previously hardcoded to 0.50/0.50, ignoring mean_reversion_weight: 0.0 in presets.
        # EMA weight now included so all three strategies contribute to consensus scoring.
        self.weights = {
            "mean_reversion": self.config.get("mean_reversion_weight", 0.50),
            "trend_following": self.config.get("trend_following_weight", 0.50),
            "ema": self.config.get("ema_weight", 0.40),
        }
        logger.info(
            f"[AGGREGATOR] Strategy weights: MR={self.weights['mean_reversion']:.2f}, "
            f"TF={self.weights['trend_following']:.2f}, "
            f"EMA={self.weights['ema']:.2f}"
        )

        # 4. Independent strategy thresholds (T1.1 fix)
        # allow_single_override and single_override_threshold exist in presets but were
        # never read by this class — orphaned config keys. Now wired.
        self.independent_thresholds = {
            "trend_following": self.config.get("single_override_threshold", 0.72),
            "mean_reversion": self.config.get("single_override_threshold", 0.75),
            "ema": self.config.get("single_override_threshold", 0.72),
        }
        self.allow_independent = self.config.get("allow_single_override", True)
        logger.info(
            f"[AGGREGATOR] Independent firing: {'ENABLED' if self.allow_independent else 'DISABLED'} "
            f"(TF≥{self.independent_thresholds['trend_following']:.2f}, "
            f"MR≥{self.independent_thresholds['mean_reversion']:.2f})"
        )

        self.stats = {
            "total_evaluations": 0,
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "bull_regime_count": 0,
            "bear_regime_count": 0,
            "regime_changes": 0,
            "consensus_signals": 0,
            "single_strategy_signals": 0,
            "regime_detection_failures": 0,
        }

        # T1.5: Stale price detection state
        # Tracks (last_price, last_change_time) per asset to catch frozen data feeds.
        self._last_prices = {}
        self._stale_threshold_minutes = 30

        # T3.4: Economic calendar — loaded at startup, hot-reloaded by CalendarUpdater
        self._econ_cal_path = "config/economic_calendar.json"
        self._econ_events = []
        self._load_calendar_file()

        self._log_initialization()

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯   PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("   ✓ STRICT MODE: Counter-trend trades blocked > 50% conf")
        logger.info("   ✓ RANGING SAFEGUARD: Max 1 trade when trend is weak")
        logger.info("   ✓ DYNAMIC threshold adjustment")
        if self.ai_enabled:
            logger.info("   ✓ AI VALIDATION: Active with circuit breaker")
        else:
            logger.info("   ⚠ AI VALIDATION: Disabled")
        logger.info("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # CALENDAR HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _load_calendar_file(self):
        """Load economic events from the JSON file on disk."""
        try:
            import json as _json
            with open(self._econ_cal_path, encoding="utf-8") as _f:
                self._econ_events = _json.load(_f).get("events", [])
            logger.info(
                f"[CALENDAR] Loaded {len(self._econ_events)} events "
                f"from {self._econ_cal_path}"
            )
        except Exception as _e:
            logger.warning(f"[CALENDAR] Could not load {self._econ_cal_path}: {_e}")
            self._econ_events = []

    def reload_calendar(self):
        """
        Hot-reload the economic calendar from disk.
        Called by CalendarUpdater after each successful write so the
        aggregator picks up fresh data without a bot restart.
        """
        self._load_calendar_file()
        logger.info(
            f"[CALENDAR] 🔄 Hot-reloaded — "
            f"{len(self._econ_events)} active events in memory"
        )

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics"""
        total = max(self.stats["total_evaluations"], 1)
        base_stats = {
            **self.stats,
            "signal_rate": (self.stats["signals_generated"] / total) * 100,
            "buy_rate": (self.stats["buy_signals"] / total) * 100,
            "sell_rate": (self.stats["sell_signals"] / total) * 100,
            "bull_regime_pct": (self.stats["bull_regime_count"] / total) * 100,
            "bear_regime_pct": (self.stats["bear_regime_count"] / total) * 100,
        }

        # Add AI statistics
        if self.ai_enabled and hasattr(self, "ai_stats"):
            mr_total = self.ai_stats["mr_signals_checked"]
            tf_total = self.ai_stats["tf_signals_checked"]

            base_stats["ai_validation"] = {
                "enabled": True,
                "circuit_breaker_active": self.ai_bypass_active,
                "mr_checked": mr_total,
                "mr_approved": self.ai_stats["mr_approved"],
                "mr_rejected": self.ai_stats["mr_rejected"],
                "mr_rejection_rate": (
                    (self.ai_stats["mr_rejected"] / mr_total * 100)
                    if mr_total > 0
                    else 0
                ),
                "tf_checked": tf_total,
                "tf_approved": self.ai_stats["tf_approved"],
                "tf_rejected": self.ai_stats["tf_rejected"],
                "tf_rejection_rate": (
                    (self.ai_stats["tf_rejected"] / tf_total * 100)
                    if tf_total > 0
                    else 0
                ),
            }
            

        return base_stats

    def _check_ai_circuit_breaker(self) -> bool:
        """
        Check if AI is rejecting too many signals
        Returns True if AI should be bypassed
        """
        if not self.enable_circuit_breaker or len(self.ai_rejection_window) < 20:
            return False

        # Calculate rejection rate (True = rejected, False = approved)
        rejection_rate = sum(self.ai_rejection_window) / len(self.ai_rejection_window)

        if rejection_rate > self.ai_bypass_threshold:
            if not self.ai_bypass_active:
                logger.warning("")
                logger.warning("=" * 70)
                logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
                logger.warning(
                    f"   Rejection rate: {rejection_rate:.0%} (threshold: {self.ai_bypass_threshold:.0%})"
                )
                logger.warning(
                    f"   AI validation temporarily DISABLED for next 10 signals"
                )
                logger.warning("=" * 70)
                logger.warning("")
                self.ai_bypass_active = True
                self.ai_bypass_cooldown = 10  # Bypass next 10 signals

            return True

        # Check if cooldown expired
        if self.ai_bypass_active and self.ai_bypass_cooldown <= 0:
            logger.info("🔄 AI circuit breaker reset - validation RE-ENABLED")
            self.ai_bypass_active = False
            self.ai_rejection_window.clear()  # Reset tracking

        return self.ai_bypass_active

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Multi-factor regime detection with cold-start handling
        Returns: (is_bull, confidence)
        """
        try:
            MIN_DATA_POINTS = 50

            # ===============================
            # 1️⃣ Cold-start & data sufficiency
            # ===============================
            if len(df) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for regime detection: {len(df)} rows"
                )
                self.stats["regime_detection_failures"] += 1

                if self.previous_regime is not None:
                    return self.previous_regime, 0.3

                if len(df) >= 20:
                    recent_momentum = (
                        df["close"].iloc[-1] - df["close"].iloc[-20]
                    ) / df["close"].iloc[-20]
                    emergency_regime = recent_momentum > 0
                    logger.info(
                        f"[REGIME] Emergency mode: {'BULL' if emergency_regime else 'BEAR'} "
                        f"(20-day momentum: {recent_momentum:.2%})"
                    )
                    return emergency_regime, 0.3

                logger.warning(
                    "[REGIME] Insufficient data - defaulting to BEAR (conservative)"
                )
                return False, 0.3

            # ===============================
            # 2️⃣ Feature generation
            # ===============================
            features_df = self.s_ema.generate_features(df.tail(250))
            if features_df.empty or len(features_df) < MIN_DATA_POINTS:
                logger.warning(f"EMA features insufficient: {len(features_df)} rows")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            latest = features_df.iloc[-1]

            ema_fast = latest.get("ema_fast", np.nan)
            ema_slow = latest.get("ema_slow", np.nan)
            ema_diff_pct = latest.get("ema_diff_pct", 0.0)

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                logger.warning("Invalid EMA values")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # ====================================================================
            # 3️⃣ Thresholds (Rolling Quantile) - ✅ TASK 21 (Phase 3)
            # ====================================================================
            # Reason: Fixed thresholds fail in different volatility regimes.
            # We use the last 100 bars to find the 65th/35th percentiles.
            ema_diff_series = features_df["ema_diff_pct"].tail(100).dropna()
            
            if len(ema_diff_series) >= 50: # Minimum bars for meaningful quantile
                # Calculate percentiles
                BULLISH_THRESHOLD = ema_diff_series.quantile(0.65)
                BEARISH_THRESHOLD = ema_diff_series.quantile(0.35)
                
                # Clamp to institutional bounds [0.05, 0.40]
                BULLISH_THRESHOLD = max(0.05, min(0.40, BULLISH_THRESHOLD))
                BEARISH_THRESHOLD = min(-0.05, max(-0.40, BEARISH_THRESHOLD))
            else:
                # Fallback to defaults
                BULLISH_THRESHOLD = 0.15 if self.asset_type == "BTC" else 0.10
                BEARISH_THRESHOLD = -0.15 if self.asset_type == "BTC" else -0.10

            close_prices = features_df["close"].values

            ret_20 = (
                (close_prices[-1] - close_prices[-20]) / close_prices[-20]
                if len(close_prices) >= 20
                else 0.0
            )
            ret_50 = (
                (close_prices[-1] - close_prices[-50]) / close_prices[-50]
                if len(close_prices) >= 50
                else 0.0
            )

            if len(close_prices) >= 21:
                returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                vol_20 = np.std(returns) * np.sqrt(252)
            else:
                vol_20 = 0.2

            adx = latest.get("adx", 20)
            macd_hist = latest.get("macd_hist", 0)
            rsi = latest.get("rsi", 50)
            
            # Asset-specific ADX threshold
            adx_threshold = getattr(self.s_trend_following, 'adx_threshold', 25)

            # ===============================
            # 4️⃣ Multi-factor scoring
            # ===============================
            bullish_score = 0
            bearish_score = 0

            # EMA positioning (dominant factor)
            if ema_diff_pct > BULLISH_THRESHOLD:
                bullish_score += 3
            elif ema_diff_pct < BEARISH_THRESHOLD:
                bearish_score += 3

            # Short-term momentum
            if ret_20 > 0.02:
                bullish_score += 2
            elif ret_20 < -0.02:
                bearish_score += 2

            # Medium-term momentum
            if ret_50 > 0.05:
                bullish_score += 2
            elif ret_50 < -0.05:
                bearish_score += 2

            # MACD
            if macd_hist > 0:
                bullish_score += 1
            elif macd_hist < 0:
                bearish_score += 1

            # ADX trend strength
            if adx > adx_threshold:
                if ema_diff_pct > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            # RSI
            if rsi > 60:
                bullish_score += 1
            elif rsi < 40:
                bearish_score += 1

            # ===============================
            # 5️⃣ Hysteresis-based decision
            # ===============================
            if self.previous_regime is None:
                is_bull = bullish_score > bearish_score
            else:
                if self.previous_regime:
                    is_bull = not (bearish_score > bullish_score + 2)
                else:
                    is_bull = bullish_score > bearish_score + 2

            # ===============================
            # 6️⃣ Confidence scoring
            # ===============================
            confidence = 0.5

            if abs(ema_diff_pct) > 0.5:
                confidence += 0.15

            if (is_bull and ret_20 > 0.03) or (not is_bull and ret_20 < -0.03):
                confidence += 0.15

            if adx > 25:
                confidence += 0.1

            if abs(bullish_score - bearish_score) >= 4:
                confidence += 0.1

            confidence = min(1.0, max(0.3, confidence))

            # ===============================
            # 7️⃣ Logging & stats
            # ===============================
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
                logger.info(
                    f"⚡ REGIME FLIP → {'BULL' if is_bull else 'BEAR'} | "
                    f"Scores B:{bullish_score} / R:{bearish_score} | "
                    f"Confidence: {confidence:.2f}"
                )

            elif not self.regime_initialized:
                logger.info(
                    f"🎬 INITIAL REGIME → {'BULL' if is_bull else 'BEAR'} | "
                    f"Confidence: {confidence:.2f}"
                )
                self.regime_initialized = True

            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            return is_bull, confidence

        # ======================================================
        # 8️⃣ HARD FALLBACK: EMA-only regime detection
        # ======================================================
        except Exception as e:
            logger.error(f"Primary regime detection failed: {e}", exc_info=True)
            self.stats["regime_detection_failures"] += 1

            try:
                ema_signal, ema_conf = self.s_ema.generate_signal(df)
                is_bull = ema_signal >= 0

                self.previous_regime = is_bull
                if is_bull:
                    self.stats["bull_regime_count"] += 1
                else:
                    self.stats["bear_regime_count"] += 1

                return is_bull, ema_conf

            except Exception as e:
                logger.error(f"EMA fallback failed: {e}", exc_info=True)
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3


    def calculate_regime_adjusted_thresholds(
        self, is_bull: bool, regime_confidence: float
    ) -> Tuple[float, float]:
        """
        Dynamically adjust thresholds based on regime strength
        """
        base_buy = self.config["buy_threshold"]
        base_sell = self.config["sell_threshold"]

        # Fix E: proportional adjustments (percentage of base) instead of fixed offsets.
        # Fixed offsets were regime-blind: a 0.10 offset on a 0.23 scalper threshold
        # is a 43% swing, while the same offset on a 0.33 conservative is only 30%.
        strength = (regime_confidence - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
        strength = max(0.0, min(1.0, strength))

        if is_bull:
            # Bull: ease buy gate by up to 18%, tighten sell gate by up to 15%
            adjusted_buy = base_buy * (1.0 - 0.18 * strength)
            adjusted_sell = base_sell * (1.0 + 0.15 * strength)
        else:
            # Bear: tighten buy gate by up to 20%, ease sell gate by up to 18%
            adjusted_buy = base_buy * (1.0 + 0.20 * strength)
            adjusted_sell = base_sell * (1.0 - 0.18 * strength)

        # Safety bounds
        adjusted_buy = max(0.15, min(0.60, adjusted_buy))
        adjusted_sell = max(0.15, min(0.60, adjusted_sell))

        # Log significant changes
        if abs(adjusted_buy - base_buy) > 0.05:
            logger.debug(
                f"[THRESHOLD] Buy: {base_buy:.2f}→{adjusted_buy:.2f} ({'BULL' if is_bull else 'BEAR'}, conf:{regime_confidence:.2f})"
            )

        return adjusted_buy, adjusted_sell

    def _format_ai_validation_for_viz(
        self, final_signal: int, details: dict, df: pd.DataFrame
    ) -> dict:
        """
        CRITICAL FIX: Format AI validation results for visualization
        ✅ FIXED: Proper type conversions for pattern_detected and near_sr_level
        """
        try:
            # Initialize with safe defaults
            viz_data = {
                "pattern_detected": False,  # ← Must be bool
                "validation_passed": False,
                "pattern_name": "None",
                "pattern_id": None,
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,  # ← Must be bool
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "action": "none",
                "rejection_reasons": [],
                "error": None,
            }

            # Check if AI validator exists
            if not self.ai_validator or not self.ai_enabled:
                viz_data["action"] = "ai_disabled"
                return viz_data

            current_price = float(df["close"].iloc[-1])

            # ================================================================
            # STEP 1: Get S/R Analysis
            # ================================================================
            try:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type,
                    df=df,
                    current_price=current_price,
                    signal=final_signal,
                    threshold=self.ai_validator.current_sr_threshold,
                )

                # ✅ FIX: Convert numpy.bool to Python bool
                near_level = sr_result.get("near_level", False)
                if isinstance(near_level, np.bool_):
                    near_level = bool(near_level)

                viz_data["sr_analysis"] = {
                    "near_sr_level": near_level,  # ← Now guaranteed Python bool
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }

            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")
                viz_data["error"] = f"S/R error: {str(e)}"

            # ================================================================
            # STEP 2: Get Pattern Detection
            # ================================================================
            try:
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=final_signal,
                    min_confidence=self.ai_validator.current_pattern_threshold,
                )

                # ✅ FIX: pattern_detected should be BOOL, not string
                pattern_confirmed = pattern_result.get("pattern_confirmed", False)
                pattern_name = pattern_result.get("pattern_name", "None")
                
                # Convert to proper bool
                if isinstance(pattern_confirmed, str):
                    pattern_confirmed = pattern_confirmed not in ["None", "Noise", ""]
                
                viz_data["pattern_detected"] = bool(pattern_confirmed)  # ← Force bool
                viz_data["pattern_name"] = pattern_name  # ← Separate field for name
                viz_data["pattern_id"] = pattern_result.get("pattern_id")
                viz_data["pattern_confidence"] = pattern_result.get("confidence", 0.0)

                # Get top 3 patterns
                if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                    try:
                        snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                        first_open = snippet[0, 0]

                        if first_open > 0:
                            snippet_norm = snippet / first_open - 1
                            snippet_input = snippet_norm.reshape(1, 15, 4)

                            predictions = self.ai_validator.sniper.model.predict(
                                snippet_input, verbose=0
                            )[0]

                            top3_indices = predictions.argsort()[-3:][::-1]
                            top3_confidences = predictions[top3_indices]

                            top3_patterns = []
                            for idx in top3_indices:
                                pattern_name = self.ai_validator.reverse_pattern_map.get(
                                    idx, f"Pattern_{idx}"
                                )
                                top3_patterns.append(pattern_name)

                            viz_data["top3_patterns"] = top3_patterns
                            viz_data["top3_confidences"] = top3_confidences.tolist()

                    except Exception as e:
                        logger.debug(f"[VIZ] Top3 patterns failed: {e}")

            except Exception as e:
                logger.error(f"[VIZ] Pattern detection failed: {e}")
                viz_data["error"] = f"Pattern error: {str(e)}"

            # ================================================================
            # STEP 3: Determine Validation Status
            # ================================================================
            original_signal = details.get("original_signal", final_signal)

            if final_signal == 0 and original_signal != 0:
                viz_data["validation_passed"] = False
                viz_data["action"] = "rejected"

                reasons = []
                if not viz_data["sr_analysis"]["near_sr_level"]:
                    reasons.append("No nearby S/R level")
                if not viz_data["pattern_detected"]:
                    reasons.append("No pattern detected")
                if viz_data["pattern_confidence"] < self.ai_validator.current_pattern_threshold:
                    reasons.append(f"Low confidence ({viz_data['pattern_confidence']:.1%})")

                viz_data["rejection_reasons"] = reasons

            elif final_signal != 0:
                viz_data["validation_passed"] = True

                if details.get("ai_bypassed", False):
                    viz_data["action"] = "bypassed"
                elif details.get("signal_quality", 0) >= self.strong_signal_bypass:
                    viz_data["action"] = "bypassed_strong_signal"
                else:
                    viz_data["action"] = "approved"
            else:
                viz_data["action"] = "hold"

            # ================================================================
            # ✅ FINAL TYPE VALIDATION
            # ================================================================
            # Ensure all bools are Python bool, not numpy.bool
            viz_data["pattern_detected"] = bool(viz_data["pattern_detected"])
            viz_data["validation_passed"] = bool(viz_data["validation_passed"])
            viz_data["sr_analysis"]["near_sr_level"] = bool(viz_data["sr_analysis"]["near_sr_level"])

            return viz_data

        except Exception as e:
            logger.error(f"[VIZ] AI formatting failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "ERROR",
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "action": "error",
                "error": str(e),
            }

    def _calculate_score(
        self,
        target_signal: int,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        ema_signal: int,
        ema_conf: float,
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """Calculate aggregated score for all three strategies (MR + TF + EMA)."""
        components = []
        total_score = 0.0
        agreement_count = 0
        min_conf = self.config["min_confidence_to_use"]
        hold_contrib = self.config["hold_contribution_pct"]
        opposition_penalty = self.config["opposition_penalty"]

        # Mean Reversion contribution
        if mr_signal == target_signal:
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_agree:{contribution:.3f}")
            agreement_count += 1
        elif mr_signal == 0:
            effective_conf = max(mr_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "mean_reversion"
            ]
            total_score += contribution
            components.append(f"MR_hold:{contribution:.3f}")
        else:
            effective_conf = max(mr_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "mean_reversion"
            ]
            total_score -= penalty
            components.append(f"MR_oppose:-{penalty:.3f}")

        # Trend Following contribution
        if tf_signal == target_signal:
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_agree:{contribution:.3f}")
            agreement_count += 1
        elif tf_signal == 0:
            effective_conf = max(tf_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "trend_following"
            ]
            total_score += contribution
            components.append(f"TF_hold:{contribution:.3f}")
        else:
            effective_conf = max(tf_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "trend_following"
            ]
            total_score -= penalty
            components.append(f"TF_oppose:-{penalty:.3f}")

        # EMA contribution (previously excluded — now a full voting member)
        if ema_signal == target_signal:
            effective_conf = max(ema_conf, min_conf)
            contribution = effective_conf * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_agree:{contribution:.3f}")
            agreement_count += 1
        elif ema_signal == 0:
            effective_conf = max(ema_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_hold:{contribution:.3f}")
        else:
            effective_conf = max(ema_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights["ema"]
            total_score -= penalty
            components.append(f"EMA_oppose:-{penalty:.3f}")

        explanation = " + ".join(components) if components else "no_agreement"

        # Agreement bonus — tiered (two_strategy_bonus and three_strategy_bonus now both active)
        if agreement_count == 3:
            bonus = self.config.get("three_strategy_bonus", self.config["two_strategy_bonus"])
            total_score += bonus
            explanation += f" + bonus3({bonus:.2f})"
        elif agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus2({bonus:.2f})"

        # Regime context
        if target_signal == 1:  # BUY
            if is_bull:
                regime_adj = self.config["bull_buy_boost"]
                total_score += regime_adj
                explanation += f" + bull({regime_adj:.2f})"
            else:
                regime_adj = -self.config["bear_buy_penalty"]
                total_score = max(0.0, total_score + regime_adj)
                explanation += f" - bear({abs(regime_adj):.2f})"
        else:  # SELL
            if is_bull:
                regime_adj = -self.config["bull_sell_penalty"]
                total_score = max(0.0, total_score + regime_adj)
                explanation += f" - bull({abs(regime_adj):.2f})"
            else:
                regime_adj = self.config["bear_sell_boost"]
                total_score += regime_adj
                explanation += f" + bear({regime_adj:.2f})"

        total_score = max(0.0, total_score)
        return total_score, explanation, agreement_count
    
    def _check_governor_filter(self, signal: int) -> Tuple[bool, Optional[str]]:
        """
        Filter 1: Governor (Daily 200 EMA) Check
        
        Returns:
            (passed, trade_type)
        """
        if not self.use_macro_governor:
            return True, "TREND"

        if not self.enable_filters or not self.mtf_integration:
            return True, "TREND"  # Skip if disabled
        
        try:
            # Get Governor analysis from MTF
            regime_data = self.mtf_integration._current_regime_data.get(self.asset_type)
            
            if not regime_data:
                logger.debug(f"[GOV] No data for {self.asset_type}, allowing trade")
                return True, "TREND"
            
            # ✨ IMPROVED: Robust key check
            governor = regime_data.get('governor') or regime_data.get('full_regime_status')
            
            if not governor:
                logger.debug(f"[GOV] No governor object for {self.asset_type}, allowing trade")
                return True, "TREND"
            
            # ✨ IMPROVED: Handle Enum vs String vs Attribute
            raw_trade_type = getattr(governor, 'trade_type', None)
            if raw_trade_type is None:
                # Fallback to consensus_regime if trade_type is missing
                regime_name = getattr(governor, 'consensus_regime', "NEUTRAL")
                trade_type = "NEUTRAL" if regime_name == "NEUTRAL" else "TREND"
            else:
                trade_type = getattr(raw_trade_type, 'value', str(raw_trade_type))

            # T2.1 fix: NEUTRAL used to block all trading.
            # Simulation: 129 blocked signals at 70.5% WR, +70.2% P&L.
            # NEUTRAL is MR's best environment (+159% P&L, 71% WR).
            # Now returns TRANSITION so trades fire at 50% position size
            # (sizing reduction applied in get_aggregated_signal below).
            if trade_type == "NEUTRAL":
                logger.info("[GOV] ⚠️ TRANSITION — market neutral, allowing at 50% size")
                return True, "TRANSITION"

            return True, trade_type
        
        except Exception as e:
            logger.error(f"[GOV] Error: {e}")
            return True, "TREND"  # Fail-open
    
    def _check_volatility_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 2: Volatility Gate
        
        Returns:
            (passed, atr_pct)
        """
        if not self.enable_filters:
            return True, 0.005
        
        try:
            if len(df) < 20:
                return True, 0.005
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            atr_pct = atr / current_price
            
            threshold = self.filter_thresholds['volatility_gate']
            passed = atr_pct >= threshold
            
            if not passed:
                logger.info(f"[VOL] ❌ BLOCKED - ATR {atr_pct:.3%} < {threshold:.3%}")
            
            return passed, atr_pct
        
        except Exception as e:
            logger.error(f"[VOL] Error: {e}")
            return True, 0.005
    
    def _check_sniper_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, Dict]:
        """
        Filter 3: Sniper Lock - Institutional Edge Confirmation
        =======================================================
        A trade is confirmed if ANY of the following institutional edge conditions are met.
        This prevents rejecting high-quality trades due to cosmetic candle issues.
        
        Confirmation Logic (OR-based):
        1. AI Pattern: A high-confidence AI pattern is detected.
        2. Momentum Candle: The candle body is at least 60% of the total range.
        3. Turtle Breakout: Price closes above the 20-period Donchian High or below the Low.
        4. Volume Surge: Volume is >= 150% of its 20-period rolling average.
        5. Volatility Breach: Price closes outside the 2.0 standard deviation Bollinger Bands.
        
        Returns:
            (passed, details)
        """
        if not self.enable_filters:
            return True, {'trigger_type': 'DISABLED'}

        try:
            # Trap filter moved to pre-consensus veto phase

            latest = df.iloc[-1]
            reasons = []

            # ================================================================
            # 1. AI Pattern Confidence
            # ================================================================
            # Reason: The AI model has already encoded a multi-factor edge.
            if self.ai_validator and hasattr(self.ai_validator, 'sniper'):
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=signal,
                    min_confidence=self.filter_thresholds['sniper_confidence']
                )
                if pattern_result.get('pattern_confirmed'):
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'AI_PATTERN',
                        'pattern_name': pattern_result.get('pattern_name'),
                        'confidence': pattern_result.get('confidence'),
                    })

            # ================================================================
            # 2. Momentum Candle
            # ================================================================
            # Reason: Confirms strong conviction from buyers or sellers in the current period.
            body = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']
            if total_range > 0:
                body_ratio = body / total_range
                if body_ratio >= 0.60:
                    is_bullish_candle = latest['close'] > latest['open']
                    if (signal == 1 and is_bullish_candle) or (signal == -1 and not is_bullish_candle):
                        reasons.append({
                            'passed': True,
                            'trigger_type': 'MOMENTUM_CANDLE',
                            'body_ratio': body_ratio,
                        })

            # Check if we have enough data for rolling indicators
            if len(df) < 21: # Need 20 periods + current
                if reasons:
                    logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {[r['trigger_type'] for r in reasons]}")
                    return True, reasons[0]
                else:
                    logger.warning(f"[SNIPER] ❌ BLOCKED - Insufficient data for full institutional checks (need 21 bars, have {len(df)}).")
                    return False, {'trigger_type': None, 'reason': f'Insufficient data for breakouts (have {len(df)})'}

            # ================================================================
            # 3. Turtle Breakout (20-period Donchian Channel)
            # ================================================================
            # Reason: Captures classic institutional breakout entries.
            # We look at the previous 20 candles to define the channel *before* the current candle.
            high_20 = df['high'].iloc[-21:-1].max()
            low_20 = df['low'].iloc[-21:-1].min()

            if signal == 1 and latest['close'] > high_20:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'TURTLE_BREAKOUT',
                    'breakout_level': high_20,
                    'price': latest['close'],
                })
            elif signal == -1 and latest['close'] < low_20:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'TURTLE_BREAKOUT',
                    'breakout_level': low_20,
                    'price': latest['close'],
                })

            # ================================================================
            # 4. Volume Surge
            # ================================================================
            # Reason: Confirms institutional participation and conviction behind a move.
            volume_rolling_avg = df['volume'].iloc[-21:-1].mean()
            if volume_rolling_avg > 0 and latest['volume'] >= (volume_rolling_avg * 1.5):
                reasons.append({
                    'passed': True,
                    'trigger_type': 'VOLUME_SURGE',
                    'volume': latest['volume'],
                    'avg_volume': volume_rolling_avg,
                    'surge_factor': latest['volume'] / volume_rolling_avg if volume_rolling_avg > 0 else 0,
                })

            # ================================================================
            # 5. Volatility Breach (Bollinger Bands)
            # ================================================================
            # Reason: Detects that price has moved into a new volatility regime.
            close_rolling_mean = df['close'].iloc[-21:-1].mean()
            close_rolling_std = df['close'].iloc[-21:-1].std()
            
            if close_rolling_std > 0:
                upper_band = close_rolling_mean + (2.0 * close_rolling_std)
                lower_band = close_rolling_mean - (2.0 * close_rolling_std)

                if signal == 1 and latest['close'] > upper_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'upper',
                        'price': latest['close'],
                    })
                elif signal == -1 and latest['close'] < lower_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'lower',
                        'price': latest['close'],
                    })
            
            # ================================================================
            # Final Decision
            # ================================================================
            if reasons:
                # Log all triggers that passed
                trigger_types = [r['trigger_type'] for r in reasons]
                logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {trigger_types}")
                # Return the details of the first trigger found
                return True, reasons[0]

            logger.info(f"[SNIPER] ❌ BLOCKED - No institutional edge confirmed.")
            return False, {'trigger_type': None, 'reason': 'No confirmation criteria met'}

        except Exception as e:
            logger.error(f"[SNIPER] Error in institutional edge check: {e}", exc_info=True)
            # Fail-open: If the filter fails, we allow the trade to avoid blocking valid signals due to code errors.
            return True, {'trigger_type': 'ERROR_FALLBACK'}
    
    def _check_profit_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 4: Minimum Profit Potential
        
        Returns:
            (passed, potential_pct)
        """
        if not self.enable_filters:
            return True, 0.01
        
        try:
            if len(df) < 20:
                return True, 0.01
            
            # Use ATR as proxy for potential move
            high_low = df['high'] - df['low']
            atr = high_low.rolling(14).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            potential_pct = atr / current_price
            
            threshold = self.filter_thresholds['min_profit']
            passed = potential_pct >= threshold
            
            if not passed:
                logger.info(f"[PROFIT] ❌ BLOCKED - Potential {potential_pct:.2%} < {threshold:.2%}")
            
            return passed, potential_pct
        
        except Exception as e:
            logger.error(f"[PROFIT] Error: {e}")
            return True, 0.01
    


    def _check_atr_expansion_filter(self, df: pd.DataFrame, trade_type: str) -> bool:
        """
        Fix C: Replaced ATR Expansion (candle_range >= 1.5*ATR) with ADX Trend Confirmation.

        Old logic required the latest candle's range to exceed 1.5× ATR. This blocked
        valid signals in slow-grinding trends (GOLD, EURUSD) where candles are small but
        direction is clear. The 1.5× bar was consistently failing even when ADX showed a
        strong trend (ADX > 25).

        New logic: confirm a trend is in force (ADX > 18). This threshold is intentionally
        low — 18 separates genuine trend from pure noise without demanding strong momentum.
        Counter-trend and REVERSION trades bypass the check (trade_type != "TREND").
        """
        if trade_type != "TREND":
            return True

        try:
            if len(df) < 20:
                return True

            # Calculate ADX (14)
            try:
                import talib
                adx_series = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                adx = adx_series[-1]
            except Exception:
                # Manual ADX fallback: use DM-based approximation via TR rolling
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean()
                dm_plus = (df['high'].diff()).clip(lower=0)
                dm_minus = (-df['low'].diff()).clip(lower=0)
                # Use only the dominant direction
                dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
                dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
                di_plus = 100 * dm_plus.rolling(14).mean() / atr14
                di_minus = 100 * dm_minus.rolling(14).mean() / atr14
                dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
                adx = dx.rolling(14).mean().iloc[-1]

            if pd.isna(adx):
                return True

            ADX_MIN = 18
            passed = adx >= ADX_MIN

            if not passed:
                logger.info(f"[ADX_TREND] ❌ BLOCKED - ADX {adx:.1f} < {ADX_MIN} (insufficient trend strength)")
            else:
                logger.debug(f"[ADX_TREND] ✅ PASSED - ADX {adx:.1f}")

            return passed

        except Exception as e:
            logger.error(f"[ADX_TREND] Error: {e}")
            return True  # Fail-open

    def get_aggregated_signal(
        self,
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",
        is_bull_market: bool = True,
        governor_data: Dict = None,
    ) -> Tuple[int, Dict]:
        """
        Main aggregation logic with AI validation and external regime context.
        """
        self.stats["total_evaluations"] += 1
        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

            # AI-5: Clear per-cycle pattern cache so sniper filter and format_viz share results.
            if self.ai_validator and hasattr(self.ai_validator, 'clear_pattern_cache'):
                self.ai_validator.clear_pattern_cache()

            # ═══════════════════════════════════════════════════════════════
            # T1.5: STALE PRICE DETECTION
            # Gold was frozen at 5021.08 for 47.6 hours (March 13–15), firing
            # 12 SELL signals on dead data. Block evaluation if price has not
            # moved by even 1 pip in over 30 minutes.
            # ═══════════════════════════════════════════════════════════════
            from datetime import datetime as _dt
            _current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
            _now = _dt.now()
            _last = self._last_prices.get(self.asset_type)
            if _last:
                _last_price, _last_time = _last
                _minutes_since_move = (_now - _last_time).total_seconds() / 60
                _price_moved = abs(_current_price - _last_price) / max(_last_price, 1) > 0.00001
                if not _price_moved and _minutes_since_move > self._stale_threshold_minutes:
                    logger.warning(
                        f"[STALE] ❌ {self.asset_type} price frozen at {_current_price} "
                        f"for {_minutes_since_move:.0f}min — blocking signal evaluation"
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "regime": "UNKNOWN",
                        "reasoning": f"stale_price_{_minutes_since_move:.0f}min",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0, "mr_confidence": 0.0,
                        "tf_signal": 0, "tf_confidence": 0.0,
                        "ema_signal": 0, "ema_confidence": 0.0,
                    }
            # Update last-seen price only when it actually moves
            if not _last or abs(_current_price - _last[0]) / max(_last[0], 1) > 0.00001:
                self._last_prices[self.asset_type] = (_current_price, _now)

            # ═══════════════════════════════════════════════════════════════
            # T3.3: NY OPEN HOUR BLOCK (13:00–13:59 UTC)
            # TF signals at NY open: 53% WR, -21.2% P&L (stop-hunting territory).
            # Trades 1–2 hours later: 60% WR, +101.5% P&L.
            # BTC trades 24/7 — only block market-hours assets.
            # ═══════════════════════════════════════════════════════════════
            _hour_utc = _dt.utcnow().hour
            if _hour_utc == 13 and self.asset_type in ("USTEC", "GOLD", "EURJPY", "EURUSD"):
                logger.info(
                    f"[SESSION] ⏸️ NY open hour block — no new entries for {self.asset_type}"
                )
                return 0, {
                    "timestamp": timestamp,
                    "regime": "UNKNOWN",
                    "reasoning": "ny_open_block",
                    "final_signal": 0, "signal_quality": 0.0,
                    "mr_signal": 0, "mr_confidence": 0.0,
                    "tf_signal": 0, "tf_confidence": 0.0,
                    "ema_signal": 0, "ema_confidence": 0.0,
                }

            # ═══════════════════════════════════════════════════════════════
            # T3.4: ECONOMIC CALENDAR BLOCK
            # Trading through NFP/FOMC/CPI on a 1H timeframe is gambling.
            # Block N hours before each high-impact event.
            # ═══════════════════════════════════════════════════════════════
            if self._econ_events:
                from datetime import timezone as _tz, timedelta as _td
                _utc_now = _dt.now(_tz.utc)
                for _evt in self._econ_events:
                    try:
                        _evt_time = _dt.fromisoformat(_evt["datetime"].replace("Z", "+00:00"))
                        _hours_before = _evt.get("block_hours_before", 2)
                        _block_start = _evt_time - _td(hours=_hours_before)
                        if _block_start <= _utc_now < _evt_time:
                            _mins_to_evt = (_evt_time - _utc_now).total_seconds() / 60
                            logger.warning(
                                f"[CALENDAR] ⏸️ Blocking — {_evt['event']} in "
                                f"{_mins_to_evt:.0f}min"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "regime": "UNKNOWN",
                                "reasoning": f"econ_calendar_{_evt['event'].replace(' ', '_')}",
                                "final_signal": 0, "signal_quality": 0.0,
                                "mr_signal": 0, "mr_confidence": 0.0,
                                "tf_signal": 0, "tf_confidence": 0.0,
                                "ema_signal": 0, "ema_confidence": 0.0,
                            }
                    except Exception:
                        continue

            # STEP 1: Use EXTERNAL regime context, not internal detection
            is_bull = is_bull_market
            regime_conf = governor_data.get('confidence', 0.5) if governor_data else 0.5
            regime_name = governor_data.get('regime', 'NEUTRAL') if governor_data else "NEUTRAL"
            
            # Update stats based on provided regime
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1


            # STEP 2: Get strategy signals
            # Pass 4H context to strategies if available
            df_4h = governor_data.get('df_4h') if governor_data else None
            
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df, df_4h=df_4h)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df, df_4h=df_4h)
            ema_signal, ema_conf = self.s_ema.generate_signal(df, df_4h=df_4h)

            # Store originals for logging
            mr_original = mr_signal
            tf_original = tf_signal

            # ═══════════════════════════════════════════════════════════════
            # T3.5: BTC FUNDING RATE Z-SCORE CONFIDENCE MULTIPLIER
            # Extreme funding rates (Z ≥ 2.0) indicate crowded positioning.
            # Over-leveraged longs → MR short setups become highest probability.
            # Z-score adapts to sustained bull runs; static threshold doesn't.
            # ═══════════════════════════════════════════════════════════════
            _funding_z = governor_data.get("funding_rate_zscore", 0.0) if governor_data else 0.0
            if self.asset_type in ("BTC", "BTCUSDT") and abs(_funding_z) >= 2.0:
                if mr_signal != 0:
                    mr_conf = min(1.0, mr_conf * 1.15)
                    logger.info(
                        f"[FUNDING] Extreme positioning (Z={_funding_z:+.1f}): "
                        f"MR conf boosted to {mr_conf:.2f}"
                    )

            # ═══════════════════════════════════════════════════════════════
            # T3.6: DXY PROXY CONFIDENCE MULTIPLIER
            # Rising EUR/USD = falling dollar = bullish for GOLD/USTEC/EURJPY.
            # Computed from already-traded EUR/USD data — zero API cost.
            # ═══════════════════════════════════════════════════════════════
            _dxy_falling = governor_data.get("dxy_falling") if governor_data else None
            if _dxy_falling is not None and self.asset_type in ("GOLD", "USTEC", "EURJPY"):
                if self.asset_type == "GOLD":
                    # Dollar weakness → gold strength
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Weak dollar: GOLD TF BUY conf boosted to {tf_conf:.2f}")
                    elif not _dxy_falling and tf_signal == -1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Strong dollar: GOLD TF SELL conf boosted to {tf_conf:.2f}")
                elif self.asset_type == "USTEC":
                    # Dollar weakness generally supportive of risk assets
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.05)
                        logger.debug(f"[DXY] Weak dollar: USTEC TF BUY conf boosted to {tf_conf:.2f}")

            # ═══════════════════════════════════════════════════════════════
            # T2.6: CONSECUTIVE CANDLE CONFIDENCE MULTIPLIER
            # BTC after 3 consecutive same-direction bars + low ADX: 66% MR WR
            # GOLD after 5 consecutive bars: 85% TF continue rate
            # ADX guard prevents counter-trend fading during strong momentum
            # (MR fading streaks in high ADX: 33% WR on GOLD, 56% on BTC).
            # This is a confidence bonus, not a new gate — fails silently.
            # ═══════════════════════════════════════════════════════════════
            try:
                _closes = df['close'].values
                _consec = 0
                for _i in range(len(_closes) - 1, max(len(_closes) - 10, 0), -1):
                    if _i == 0:
                        break
                    if _closes[_i] > _closes[_i - 1]:
                        if _consec >= 0:
                            _consec += 1
                        else:
                            break
                    else:
                        if _consec <= 0:
                            _consec -= 1
                        else:
                            break

                # Compute ADX for the guard
                _adx_guard = 25.0  # default if calculation fails
                try:
                    import talib as _talib_c
                    _adx_raw = _talib_c.ADX(
                        df['high'].values, df['low'].values, _closes, timeperiod=14
                    )[-1]
                    if not np.isnan(_adx_raw):
                        _adx_guard = _adx_raw
                except Exception:
                    pass

                # BTC: boost MR when price has made 3+ consecutive candles in one
                # direction AND momentum is low — classic mean reversion setup
                if self.asset_type == "BTC" and abs(_consec) >= 3 and _adx_guard < 25:
                    if mr_signal != 0:
                        mr_conf = min(1.0, mr_conf * 1.20)
                        logger.debug(
                            f"[CANDLE] BTC {_consec}-bar streak + low ADX ({_adx_guard:.0f}): "
                            f"MR conf boosted to {mr_conf:.2f}"
                        )

                # GOLD: boost TF when riding a 5+ bar streak — trend continuation
                if self.asset_type == "GOLD" and abs(_consec) >= 5:
                    if tf_signal != 0:
                        tf_conf = min(1.0, tf_conf * 1.15)
                        logger.debug(
                            f"[CANDLE] GOLD {_consec}-bar streak: "
                            f"TF conf boosted to {tf_conf:.2f}"
                        )
            except Exception:
                pass  # Bonus only — never block execution on failure

            # Extract regime score for Gatekeeper (Phase 3)
            regime_score = governor_data.get("regime_score", 0.0)
            regime_is_bullish = governor_data.get("is_bullish", False)
            regime_is_bearish = governor_data.get("is_bearish", False)

            # ═══════════════════════════════════════════════════════════════
            # SMART GATEKEEPER — Strategy-Aware Routing (T1.3 fix)
            # ═══════════════════════════════════════════════════════════════
            # OLD BEHAVIOUR (bug): regime_score == 0.0 (NEUTRAL) killed ALL signals.
            # MR was treated identically to TF despite being an opposite strategy type.
            #
            # NEW RULES (from 30-day simulation data):
            #   TF/EMA: Block counter-trend ALWAYS. Allow trend-aligned ALWAYS.
            #   MR: Block counter-trend in BULLISH/BEARISH/SLIGHTLY regimes.
            #       Allow counter-trend ONLY in NEUTRAL (+159% P&L, 71% WR).
            #       Allow trend-aligned ALWAYS (+110% in SLIGHTLY_BEAR, 87% WR).
            #   NEUTRAL (regime_score == 0): All strategies fire freely.
            #       50% sizing is applied via T2.1 TRANSITION state in governor.
            # ═══════════════════════════════════════════════════════════════
            if self.use_gatekeeper:
                is_neutral = (regime_score == 0.0) or (not regime_is_bullish and not regime_is_bearish)

                if is_neutral:
                    # NEUTRAL: all strategies allowed in any direction
                    logger.debug(f"[GATEKEEPER] NEUTRAL — all strategies allowed ({self.asset_type})")

                elif regime_is_bullish:
                    # TF/EMA: block shorts (counter-trend in bull)
                    if tf_signal < 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (TF): Bullish regime for {self.asset_type}")
                        tf_signal = 0; tf_conf = 0.0
                    if ema_signal < 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (EMA): Bullish regime for {self.asset_type}")
                        ema_signal = 0; ema_conf = 0.0
                    # MR: block counter-trend SELL, allow trend-aligned BUY (dip buying)
                    if mr_signal < 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (MR): Counter-trend in Bullish for {self.asset_type}")
                        mr_signal = 0; mr_conf = 0.0
                    elif mr_signal > 0:
                        logger.info(f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in Bullish for {self.asset_type}")

                elif regime_is_bearish:
                    # TF/EMA: block longs (counter-trend in bear)
                    if tf_signal > 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (TF): Bearish regime for {self.asset_type}")
                        tf_signal = 0; tf_conf = 0.0
                    if ema_signal > 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (EMA): Bearish regime for {self.asset_type}")
                        ema_signal = 0; ema_conf = 0.0
                    # MR: block counter-trend BUY (falling knives), allow trend-aligned SELL
                    if mr_signal > 0:
                        logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (MR): Counter-trend in Bearish for {self.asset_type}")
                        mr_signal = 0; mr_conf = 0.0
                    elif mr_signal < 0:
                        logger.info(f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in Bearish for {self.asset_type}")
            # --- End Smart Gatekeeper ---
            
            # Initialize core variables for details building (prevents UnboundLocalError if we skip)
            buy_score = 0.0
            sell_score = 0.0
            signal_quality = 0.0
            ai_validation_details = {}
            original_signal = 0
            final_signal = 0
            reasoning = "hold (no strategy agreement)"
            trade_type = "TREND"

            # COMPUTATIONAL OPTIMIZATION: If all signals are zero, skip heavy validation
            if mr_signal == 0 and tf_signal == 0 and ema_signal == 0:
                logger.debug(f"[AGGREGATOR] {self.asset_type}: No signals to validate, skipping to end.")
                # We can skip to building the details dictionary
            else:
                # Ranging Detection — keeps position limits, counter-trend blocking
                # now handled exclusively by the Smart Gatekeeper above (T1.3 fix).
                is_ranging = regime_conf <= 0.50
                max_trades_override = None
                filter_reason = ""
                if is_ranging:
                    max_trades_override = 1
                    filter_reason = "Ranging Mode (Max 1 Trade)"

                signal_quality = max(mr_conf, tf_conf)

                # --- Directional Trap Filter Veto (T2.3: regime-aware) ---
                if mr_signal != 0 or tf_signal != 0 or ema_signal != 0:
                    test_direction = "long" if (mr_signal > 0 or tf_signal > 0 or ema_signal > 0) else "short"
                    # regime_aligned: signal direction matches macro regime
                    _trap_aligned = (
                        (test_direction == "long" and is_bull) or
                        (test_direction == "short" and not is_bull)
                    )
                    if not validate_candle_structure(
                        df, self.asset_type,
                        direction=test_direction,
                        regime_confidence=regime_conf,
                        regime_aligned=_trap_aligned,
                    ):
                        logger.info(f"[TRAP] VETO - Candidate rejected by structure check.")
                        return 0, {
                            "timestamp": timestamp,
                            "regime": regime_name,
                            "reasoning": "blocked_by_trap_filter",
                            "final_signal": 0,
                            "signal_quality": 0.0,
                            "mr_signal": 0,
                            "mr_confidence": 0.0,
                            "tf_signal": 0,
                            "tf_confidence": 0.0,
                            "ema_signal": 0,
                            "ema_confidence": 0.0
                        }

                # STEP 3: PRE-SCORE AI VALIDATION — DISABLED (T2.2)
                # Previously killed individual MR/TF votes before scoring, destroying
                # the consensus and independent evaluation pipeline.
                # Blocked 13 signals with 92.3% WR and +17.2% P&L.
                # AI validation now runs post-score via hybrid_validator.py.
                # The circuit breaker and stats objects are preserved for post-score use.
                ai_bypass = False
                ai_validation_details = {}

                # STEP 4: Calculate scores (MR + TF + EMA all contribute)
                buy_score, buy_explanation, buy_agreement = self._calculate_score(1, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull)
                sell_score, sell_explanation, sell_agreement = self._calculate_score(-1, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull)

                # STEP 5: Dynamic thresholds
                adj_buy_thresh, adj_sell_thresh = self.calculate_regime_adjusted_thresholds(is_bull, regime_conf)

                # STEP 6: Make decision
                if buy_score >= adj_buy_thresh and buy_score > sell_score:
                    final_signal = 1
                elif sell_score >= adj_sell_thresh and sell_score > buy_score:
                    final_signal = -1

                reasoning = f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})" if final_signal == 1 else f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})" if final_signal == -1 else f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                original_signal = final_signal

                # Fix F: removed hard cap at 0.7 — score can now reflect true 3-strategy consensus
                raw_quality = max(buy_score, sell_score)
                if buy_agreement < 2 and sell_agreement < 2: raw_quality *= 0.7
                if (final_signal == 1 and is_bull) or (final_signal == -1 and not is_bull): raw_quality *= 1.15
                signal_quality = min(raw_quality, 1.0)

                if final_signal != 0 and signal_quality < self.config["min_signal_quality"]:
                    final_signal = 0
                    reasoning = f"hold_lowquality (original:{reasoning}, quality:{signal_quality:.2f})"

                # ═══════════════════════════════════════════════════════════
                # INDEPENDENT STRATEGY EVALUATION (T1.1 fix)
                # Consensus failed (final_signal still 0). Check if any single
                # strategy has enough individual confidence to fire alone.
                # Priority: TF > EMA > MR (based on solo P&L simulation data).
                # allow_single_override and single_override_threshold are config
                # keys that existed in presets but were never read — now wired.
                # ═══════════════════════════════════════════════════════════
                if final_signal == 0 and self.allow_independent:
                    candidates = []

                    # TF: use post-gatekeeper signal (consistent with MR/EMA treatment).
                    # tf_original pre-bypass was causing asymmetric gatekeeper application.
                    if tf_signal != 0 and tf_conf >= self.independent_thresholds["trend_following"]:
                        candidates.append(("TF", tf_signal, tf_conf))

                    # EMA: evaluated post-gatekeeper (gatekeeper treats EMA same as TF)
                    if ema_signal != 0 and ema_conf >= self.independent_thresholds["ema"]:
                        candidates.append(("EMA", ema_signal, ema_conf))

                    # MR: use post-gatekeeper signal (Smart Gatekeeper already filtered it)
                    if mr_signal != 0 and mr_conf >= self.independent_thresholds["mean_reversion"]:
                        candidates.append(("MR", mr_signal, mr_conf))

                    if candidates:
                        # Sort by confidence descending; TF wins ties (listed first)
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        best_name, best_signal, best_conf = candidates[0]
                        final_signal = best_signal
                        signal_quality = best_conf * 0.85  # Solo signals get a small quality discount

                        # Multi-strategy confirmation bonus: any agreeing strategy lifts quality
                        agreeing = [c for c in candidates if c[1] == best_signal]
                        if len(agreeing) >= 2:
                            signal_quality = min(1.0, best_conf * 1.1)

                        reasoning = (
                            f"{'BUY' if final_signal == 1 else 'SELL'} "
                            f"(independent:{best_name}, conf:{best_conf:.2f}, "
                            f"confirmations:{len(agreeing)})"
                        )
                        logger.info(
                            f"[INDEPENDENT] {self.asset_type}: {best_name} fires alone "
                            f"(conf={best_conf:.2f}, aligned={len(agreeing)} strategies)"
                        )

                # World-Class Filters
                # Fix D: profit filter removed — it duplicated the volatility filter (both
                # measured ATR/price%) while adding an independent failure point that blocked
                # valid signals in low-ATR trending regimes (e.g. GOLD steady grind moves).
                # Fix C: ATR expansion filter replaced with ADX trend confirmation (see method).
                if final_signal != 0 and self.enable_filters:
                    gov_passed, trade_type = self._check_governor_filter(final_signal)
                    if not gov_passed: final_signal = 0; reasoning = "blocked_by_governor"
                    else:
                        vol_passed, _ = self._check_volatility_filter(df)
                        if not vol_passed: final_signal = 0; reasoning = "low_volatility"
                        else:
                            sniper_passed, _ = self._check_sniper_filter(df, final_signal)
                            if not sniper_passed: final_signal = 0; reasoning = "no_sniper_confirmation"
                            else:
                                atr_exp_passed = self._check_atr_expansion_filter(df, trade_type)
                                if not atr_exp_passed: final_signal = 0; reasoning = "insufficient_trend_strength"
            
            # STEP 7: Build base response
            details = {
                "timestamp": timestamp,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "original_signal": original_signal,
                "final_signal": final_signal,
                "reasoning": reasoning,
                "signal_quality": signal_quality,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
                "ema_signal": ema_signal,
                "ema_confidence": ema_conf,
                "governor_data": governor_data, # Pass governor data through
                "ai_validation": ai_validation_details,
                "trade_type": trade_type,
            }

            # STEP 8: Format AI validation for visualization
            if self.ai_validator:
                try:
                    # Pass copies to avoid accidental modification
                    ai_validation_details = self._format_ai_validation_for_viz(
                        final_signal=final_signal,
                        details={**details},
                        df=df
                    )
                except Exception as e:
                    logger.error(f"[AGGREGATOR] AI formatting failed: {e}")

            # STEP 9: Final Response update
            details.update({
                "ai_validation": ai_validation_details,
                "mr_signal_raw": mr_original,  # Ensure originals are present
                "tf_signal_raw": tf_original,
            })

            # T2.1: TRANSITION sizing — governor approved but market is neutral.
            # Apply 50% risk multiplier so these trades fire at half normal size.
            # T1.7 already wires mtf_risk_multiplier into both execution handlers.
            if final_signal != 0 and trade_type == "TRANSITION":
                current_multiplier = details.get("mtf_risk_multiplier", 1.0)
                details["mtf_risk_multiplier"] = current_multiplier * 0.5
                logger.info(
                    f"[TRANSITION] {self.asset_type}: signal approved at 50% size "
                    f"(mtf_risk_multiplier={details['mtf_risk_multiplier']:.2f})"
                )

            return final_signal, details
        

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {
                "error": str(e),
                "timestamp": timestamp,
                "reasoning": f"error: {str(e)[:50]}",
                "signal_quality": 0.0,
                "final_signal": 0,
                "mr_signal": 0,
                "mr_confidence": 0.0,
                "tf_signal": 0,
                "tf_confidence": 0.0,
                "ema_signal": 0,
                "ema_confidence": 0.0,
            }

    def _calculate_ai_impact(self, ai_stats: dict) -> dict:
        """Calculate AI validation impact on trading"""
        total_checks = ai_stats.get("total_checks", 0)
        if total_checks == 0:
            return {"message": "No AI checks performed"}

        approved = ai_stats.get("approved", 0)
        rejected = ai_stats.get("rejected", 0)
        bypassed_strong = ai_stats.get("bypassed_strong_signal", 0)
        bypassed_breaker = ai_stats.get("bypassed_circuit_breaker", 0)

        effective_signals = approved + bypassed_strong + bypassed_breaker
        filter_rate = (rejected / total_checks) * 100 if total_checks > 0 else 0

        return {
            "total_signals_checked": total_checks,
            "effective_signals": effective_signals,
            "filtered_signals": rejected,
            "filter_rate": f"{filter_rate:.1f}%",
            "strong_signal_bypasses": bypassed_strong,
            "circuit_breaker_bypasses": bypassed_breaker,
            "net_approval_rate": f"{(effective_signals/total_checks)*100:.1f}%",
            "assessment": self._assess_ai_performance(filter_rate),
        }

    def _assess_ai_performance(self, filter_rate: float) -> str:
        """Assess if AI filtering is appropriate"""
        if filter_rate > 75:
            return "⚠️ OVER-FILTERING: AI rejecting too many signals"
        elif filter_rate > 50:
            return "⚠️ HIGH FILTERING: AI may be too strict"
        elif filter_rate > 25:
            return "✓ BALANCED: AI filtering is reasonable"
        elif filter_rate > 10:
            return "✓ LIGHT FILTERING: AI approving most signals"
        else:
            return "ℹ️ MINIMAL FILTERING: AI rarely rejecting signals"
