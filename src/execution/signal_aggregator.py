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
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()

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
        self.filter_thresholds = {
            'volatility_gate': config.get('world_class_filters', {}).get(
                'volatility_gate_threshold', 0.0035
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

        # Strategy weights
        self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}

        # ================================================================
        # CONFIGURATION MERGE (Safety Fix)
        # ================================================================
        # 1. Define Defaults first (guarantees all keys exist)
        if self.asset_type == "BTC":
            self.config = {
                "buy_threshold": 0.32,
                "sell_threshold": 0.26,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "bull_buy_boost": 0.25,
                "bull_sell_penalty": 0.20,
                "bear_sell_boost": 0.25,
                "bear_buy_penalty": 0.30,
                "min_confidence_to_use": 0.08,
                "min_signal_quality": 0.28,
                "hold_contribution_pct": 0.20,
                "opposition_penalty": 0.40,
            }
        else:  # GOLD (Default)
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.24,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.35,
                "bull_buy_boost": 0.22,
                "bull_sell_penalty": 0.15,
                "bear_sell_boost": 0.22,
                "bear_buy_penalty": 0.28,
                "min_confidence_to_use": 0.06,
                "min_signal_quality": 0.25,
                "hold_contribution_pct": 0.18,
                "opposition_penalty": 0.40,
            }
        
        # 2. Update with passed config (Merge instead of Overwrite)
        if config is not None:
            # This ensures keys missing from 'config' are filled by defaults above
            self.config.update(config)

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

            # ===============================
            # 3️⃣ Thresholds (asset-specific)
            # ===============================
            if self.asset_type == "BTC":
                BULLISH_THRESHOLD = 0.15
                BEARISH_THRESHOLD = -0.15
            else:  # GOLD
                BULLISH_THRESHOLD = 0.10
                BEARISH_THRESHOLD = -0.10

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
            if adx > 25:
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

        # Scale adjustment by regime confidence
        strength = (regime_confidence - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
        strength = max(0.0, min(1.0, strength))

        if is_bull:
            # Bull market: encourage buying
            adjusted_buy = base_buy - (0.10 * strength)
            adjusted_sell = base_sell + (0.10 * strength)
        else:
            # Bear market: discourage buying
            adjusted_buy = base_buy + (0.15 * strength)
            adjusted_sell = base_sell - (0.12 * strength)

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
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """Calculate aggregated score"""
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

        explanation = " + ".join(components) if components else "no_agreement"

        # Agreement bonus
        if agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus({bonus:.2f})"

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
        if not self.enable_filters or not self.mtf_integration:
            return True, "TREND"  # Skip if disabled
        
        try:
            # Get Governor analysis from MTF
            regime_data = self.mtf_integration._current_regime_data.get(self.asset_type)
            
            if not regime_data or 'governor' not in regime_data:
                logger.debug(f"[GOV] No data for {self.asset_type}, allowing trade")
                return True, "TREND"
            
            governor = regime_data['governor']
            trade_type = governor.trade_type.value  # "TREND", "SCALP", or "V_SHAPE"

            # IF the Governor says NEUTRAL, BLOCK THE TRADE.
            if trade_type == "NEUTRAL":
                logger.info("[GOV] ❌ BLOCKED - Market is Neutral/Cash")
                return False, trade_type

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
            if not validate_candle_structure(df, self.asset_type):
                logger.info(f"[SNIPER] ❌ BLOCKED - Trap candle detected for {self.asset_type}")
                return False, {'trigger_type': 'TRAP_CANDLE', 'reason': 'Candle structure indicates a trap'}

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
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            ema_signal, ema_conf = self.s_ema.generate_signal(df)

            # Store originals for logging
            mr_original = mr_signal
            tf_original = tf_signal

            # Extract regime score for Gatekeeper (Phase 3)
            regime_score = governor_data.get("regime_score", 0.0)
            regime_is_bullish = governor_data.get("is_bullish", False)
            regime_is_bearish = governor_data.get("is_bearish", False)

            # --- Gatekeeper Filtering (Phase 3) ---
            # If regime_score is 0, block all signals
            if regime_score == 0.0:
                if mr_signal != 0 or tf_signal != 0 or ema_signal != 0:
                    logger.warning(f"[GATEKEEPER] ❌ BLOCKED ALL: Regime score is 0.0 ({self.asset_type}).")
                mr_signal = 0
                tf_signal = 0
                ema_signal = 0
            # If regime is bullish (score > 0) and a signal is bearish (short), block short signals
            elif regime_is_bullish and (mr_signal < 0 or tf_signal < 0 or ema_signal < 0):
                if mr_signal < 0:
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (MR): Bullish regime ({regime_score:.2f}) for {self.asset_type}.")
                    mr_signal = 0
                if tf_signal < 0:
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (TF): Bullish regime ({regime_score:.2f}) for {self.asset_type}.")
                    tf_signal = 0
                if ema_signal < 0: # Assuming EMA signal can also be filtered
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (EMA): Bullish regime ({regime_score:.2f}) for {self.asset_type}.")
                    ema_signal = 0
            # If regime is bearish (score < 0) and a signal is bullish (long), block long signals
            elif regime_is_bearish and (mr_signal > 0 or tf_signal > 0 or ema_signal > 0):
                if mr_signal > 0:
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (MR): Bearish regime ({regime_score:.2f}) for {self.asset_type}.")
                    mr_signal = 0
                if tf_signal > 0:
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (TF): Bearish regime ({regime_score:.2f}) for {self.asset_type}.")
                    tf_signal = 0
                if ema_signal > 0: # Assuming EMA signal can also be filtered
                    logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (EMA): Bearish regime ({regime_score:.2f}) for {self.asset_type}.")
                    ema_signal = 0
            # --- End Gatekeeper Filtering ---
            
            # Strict Trend Enforcement & Ranging Logic
            is_ranging = regime_conf <= 0.50
            max_trades_override = None
            filter_reason = ""
            if is_ranging:
                max_trades_override = 1
                filter_reason = "Ranging Mode (Max 1 Trade)"
            else:
                if is_bull:
                    if tf_signal == -1: tf_signal = 0; filter_reason += "TF Short Blocked; "
                    if mr_signal == -1: mr_signal = 0; filter_reason += "MR Short Blocked; "
                else:
                    if tf_signal == 1: tf_signal = 0; filter_reason += "TF Long Blocked; "
                    if mr_signal == 1: mr_signal = 0; filter_reason += "MR Long Blocked; "

            signal_quality = max(mr_conf, tf_conf)

            # STEP 3: AI VALIDATION (with circuit breaker)
            ai_bypass = False
            ai_validation_details = {}
            if self.ai_enabled and self.ai_validator is not None:
                ai_bypass = self._check_ai_circuit_breaker()
                if self.ai_bypass_active and self.ai_bypass_cooldown > 0:
                    self.ai_bypass_cooldown -= 1

                if not ai_bypass:
                    if mr_signal != 0:
                        self.ai_stats["mr_signals_checked"] += 1
                        validated_mr, mr_details = self.ai_validator.validate_signal(mr_signal, {"strategy": "mean_reversion", "confidence": mr_conf, "regime": "bull" if is_bull else "bear", "regime_confidence": regime_conf, "asset": self.asset_type, "signal_quality": signal_quality}, df)
                        if validated_mr == 0:
                            self.ai_stats["mr_rejected"] += 1; self.ai_rejection_window.append(True); mr_signal = 0; mr_conf = 0.0
                        else:
                            self.ai_stats["mr_approved"] += 1; self.ai_rejection_window.append(False)
                        ai_validation_details = mr_details.get('ai_validation', {})

                    if tf_signal != 0:
                        self.ai_stats["tf_signals_checked"] += 1
                        validated_tf, tf_details = self.ai_validator.validate_signal(tf_signal, {"strategy": "trend_following", "confidence": tf_conf, "regime": "bull" if is_bull else "bear", "regime_confidence": regime_conf, "asset": self.asset_type, "signal_quality": signal_quality}, df)
                        if validated_tf == 0:
                            self.ai_stats["tf_rejected"] += 1; self.ai_rejection_window.append(True); tf_signal = 0; tf_conf = 0.0
                        else:
                            self.ai_stats["tf_approved"] += 1; self.ai_rejection_window.append(False)
                        ai_validation_details = tf_details.get('ai_validation', {})

            # STEP 4: Calculate scores
            buy_score, buy_explanation, buy_agreement = self._calculate_score(1, mr_signal, mr_conf, tf_signal, tf_conf, is_bull)
            sell_score, sell_explanation, sell_agreement = self._calculate_score(-1, mr_signal, mr_conf, tf_signal, tf_conf, is_bull)

            # STEP 5: Dynamic thresholds
            adj_buy_thresh, adj_sell_thresh = self.calculate_regime_adjusted_thresholds(is_bull, regime_conf)

            # STEP 6: Make decision
            final_signal = 0
            if buy_score >= adj_buy_thresh and buy_score > sell_score:
                final_signal = 1
            elif sell_score >= adj_sell_thresh and sell_score > buy_score:
                final_signal = -1

            reasoning = f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})" if final_signal == 1 else f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})" if final_signal == -1 else f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
            original_signal = final_signal

            raw_quality = max(min(buy_score, 0.7), min(sell_score, 0.7))
            if buy_agreement < 2 and sell_agreement < 2: raw_quality *= 0.7
            if (final_signal == 1 and is_bull) or (final_signal == -1 and not is_bull): raw_quality *= 1.15
            signal_quality = min(raw_quality, 1.0)

            if final_signal != 0 and signal_quality < self.config["min_signal_quality"]:
                final_signal = 0
                reasoning = f"hold_lowquality (original:{reasoning}, quality:{signal_quality:.2f})"

            # World-Class Filters
            trade_type = "TREND"
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
                            profit_passed, _ = self._check_profit_filter(df)
                            if not profit_passed: final_signal = 0; reasoning = "insufficient_profit_potential"
            
            # STEP 7: Build response
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
                "tf_signal": tf_signal,
                "ema_signal": ema_signal,
                "governor_data": governor_data, # Pass governor data through
                "ai_validation": ai_validation_details,
                "trade_type": trade_type,
            }

            return final_signal, details
        

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {
                "error": str(e),
                "timestamp": timestamp,
                "reasoning": f"error: {str(e)[:50]}",
                "signal_quality": 0.0,
                "final_signal": 0,
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
