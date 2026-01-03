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
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
    Multi-Strategy Aggregator with AI validation and safety features
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
        config: Dict = None,
        ai_validator=None,
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
                "opposition_penalty": 0.5,
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
                "opposition_penalty": 0.5,
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
        logger.info("")
        logger.info("   ✓ TIGHTER HYSTERESIS: ±0.15% (BTC) / ±0.10% (GOLD)")
        logger.info("   ✓ Multi-factor regime: EMA + momentum + volatility")
        logger.info("   ✓ DYNAMIC threshold adjustment")
        if self.ai_enabled:
            logger.info("   ✓ AI VALIDATION: Active with circuit breaker")
        else:
            logger.info("   ⚠ AI VALIDATION: Disabled")
        logger.info("")
        logger.info("📊 STRATEGY ROLES:")
        logger.info("   Mean Reversion:   DECISION MAKER (weight: 0.50)")
        logger.info("   Trend Following:  DECISION MAKER (weight: 0.50)")
        logger.info(
            f"   EMA 50/{'200' if self.asset_type == 'BTC' else '100'}:       REGIME DETECTOR"
        )
        logger.info("=" * 80)
        logger.info("")

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
        Multi-factor regime detection with  cold-start handling
        Returns: (is_bull, confidence)
        """
        try:
            MIN_DATA_POINTS = 50
            if len(df) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for regime detection: {len(df)} rows"
                )
                self.stats["regime_detection_failures"] += 1

                #  Better fallback logic
                if self.previous_regime is not None:
                    # Use previous regime
                    return self.previous_regime, 0.3

                # Emergency fallback: simple momentum check
                if len(df) >= 20:
                    recent_momentum = (
                        df["close"].iloc[-1] - df["close"].iloc[-20]
                    ) / df["close"].iloc[-20]
                    emergency_regime = recent_momentum > 0
                    logger.info(
                        f"[REGIME] Emergency mode: {'BULL' if emergency_regime else 'BEAR'} (20-day momentum: {recent_momentum:.2%})"
                    )
                    return emergency_regime, 0.3

                # Last resort: default to BEAR (conservative)
                logger.warning(
                    "[REGIME] Insufficient data - defaulting to BEAR (conservative)"
                )
                return False, 0.3

            # Generate features
            features_df = self.s_ema.generate_features(df.tail(250))
            if features_df.empty or len(features_df) < MIN_DATA_POINTS:
                logger.warning(f"EMA features insufficient: {len(features_df)} rows")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            latest = features_df.iloc[-1]

            # Core indicators
            ema_fast = latest.get("ema_fast", np.nan)
            ema_slow = latest.get("ema_slow", np.nan)
            ema_diff_pct = latest.get("ema_diff_pct", 0.0)

            # Asset-specific hysteresis thresholds
            if self.asset_type == "BTC":
                BULLISH_THRESHOLD = 0.15
                BEARISH_THRESHOLD = -0.15
            else:  # GOLD
                BULLISH_THRESHOLD = 0.10
                BEARISH_THRESHOLD = -0.10

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                logger.warning("Invalid EMA values")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # Calculate supporting indicators
            close_prices = features_df["close"].values

            # Multi-timeframe momentum
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

            # Volatility
            if len(close_prices) >= 21:
                returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                vol_20 = np.std(returns) * np.sqrt(252)
            else:
                vol_20 = 0.2

            # Technical indicators
            adx = latest.get("adx", 20)
            macd_hist = latest.get("macd_hist", 0)
            rsi = latest.get("rsi", 50)

            # Multi-factor regime scoring
            bullish_score = 0
            bearish_score = 0

            # Factor 1: EMA positioning (most important)
            if ema_diff_pct > BULLISH_THRESHOLD:
                bullish_score += 3
            elif ema_diff_pct < BEARISH_THRESHOLD:
                bearish_score += 3

            # Factor 2: Short-term momentum
            if ret_20 > 0.02:
                bullish_score += 2
            elif ret_20 < -0.02:
                bearish_score += 2

            # Factor 3: Medium-term momentum
            if ret_50 > 0.05:
                bullish_score += 2
            elif ret_50 < -0.05:
                bearish_score += 2

            # Factor 4: MACD histogram
            if macd_hist > 0:
                bullish_score += 1
            elif macd_hist < 0:
                bearish_score += 1

            # Factor 5: ADX (strong trend)
            if adx > 25:
                if ema_diff_pct > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            # Factor 6: RSI extremes
            if rsi > 60:
                bullish_score += 1
            elif rsi < 40:
                bearish_score += 1

            # Regime decision with hysteresis
            if self.previous_regime is None:
                is_bull = bullish_score > bearish_score
            else:
                if self.previous_regime:
                    # Currently BULLISH - need clear bearish signals to flip
                    if bearish_score > bullish_score + 2:
                        is_bull = False
                        logger.info(
                            f"   🔄 Regime flip: BULL→BEAR (scores: bull={bullish_score}, bear={bearish_score})"
                        )
                    else:
                        is_bull = True
                else:
                    # Currently BEARISH - need clear bullish signals to flip
                    if bullish_score > bearish_score + 2:
                        is_bull = True
                        logger.info(
                            f"   🔄 Regime flip: BEAR→BULL (scores: bull={bullish_score}, bear={bearish_score})"
                        )
                    else:
                        is_bull = False

            # Confidence calculation
            confidence = 0.5
            if abs(ema_diff_pct) > 0.5:
                confidence += 0.15
            if (is_bull and ret_20 > 0.03) or (not is_bull and ret_20 < -0.03):
                confidence += 0.15
            if adx > 25:
                confidence += 0.1
            score_diff = abs(bullish_score - bearish_score)
            if score_diff >= 4:
                confidence += 0.1
            confidence = min(1.0, max(0.3, confidence))

            # Logging
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
                regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
                logger.info("")
                logger.info("=" * 70)
                logger.info(
                    f"⚡ REGIME CHANGE #{self.stats['regime_changes']} → {regime_name}"
                )
                logger.info(
                    f"   EMA Fast: ${ema_fast:.2f} | Slow: ${ema_slow:.2f} | Diff: {ema_diff_pct:.2f}%"
                )
                logger.info(f"   Momentum: 20d={ret_20:.2%} | 50d={ret_50:.2%}")
                logger.info(f"   Scores: Bull={bullish_score} | Bear={bearish_score}")
                logger.info(
                    f"   ADX: {adx:.1f} | MACD: {macd_hist:.3f} | RSI: {rsi:.1f}"
                )
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info("=" * 70)
                logger.info("")

            elif not self.regime_initialized:
                regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
                logger.info("")
                logger.info("=" * 70)
                logger.info(f"🎬 INITIAL REGIME → {regime_name}")
                logger.info(f"   Scores: Bull={bullish_score} | Bear={bearish_score}")
                logger.info(
                    f"   EMA Diff: {ema_diff_pct:.2f}% | Momentum: {ret_20:.2%}"
                )
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info("=" * 70)
                logger.info("")
                self.regime_initialized = True

            # Update tracking
            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            return is_bull, confidence

        except Exception as e:
            logger.error(f"Error detecting regime: {e}", exc_info=True)
            self.stats["regime_detection_failures"] += 1
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
        Call this AFTER AI validation completes

        Args:
            final_signal: Final signal after AI validation
            details: Signal details dict
            df: Market data

        Returns:
            Formatted AI validation data ready for visualization
        """
        try:
            # Initialize with safe defaults
            viz_data = {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "None",
                "pattern_id": None,
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
                "action": "none",
                "rejection_reasons": [],
                "error": None,
            }

            # Check if AI validator exists and was used
            if not self.ai_validator or not self.ai_enabled:
                viz_data["action"] = "ai_disabled"
                return viz_data

            # Get current price for S/R analysis
            current_price = float(df["close"].iloc[-1])

            # ================================================================
            # STEP 1: Get S/R Analysis from AI Validator
            # ================================================================
            try:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    df=df,
                    current_price=current_price,
                    signal=final_signal,
                    threshold=self.ai_validator.current_sr_threshold,
                )

                viz_data["sr_analysis"] = {
                    "near_sr_level": sr_result.get("near_level", False),
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],  # Top 5 levels
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }

            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")
                viz_data["error"] = f"S/R error: {str(e)}"

            # ================================================================
            # STEP 2: Get Pattern Detection from AI Validator
            # ================================================================
            try:
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=final_signal,
                    min_confidence=self.ai_validator.current_pattern_threshold,
                )

                viz_data["pattern_detected"] = pattern_result.get("pattern_name", True)
                viz_data["pattern_name"] = pattern_result.get("pattern_name", "None")
                viz_data["pattern_id"] = pattern_result.get("pattern_id")
                viz_data["pattern_confidence"] = pattern_result.get("confidence", 0.0)

                # Get top 3 patterns if available
                if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                    try:
                        # Get last 15 candles for pattern detection
                        snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                        first_open = snippet[0, 0]

                        if first_open > 0:
                            snippet_norm = snippet / first_open - 1
                            snippet_input = snippet_norm.reshape(1, 15, 4)

                            # Get predictions
                            predictions = self.ai_validator.sniper.model.predict(
                                snippet_input, verbose=0
                            )[0]

                            # Get top 3
                            top3_indices = predictions.argsort()[-3:][::-1]
                            top3_confidences = predictions[top3_indices]

                            # Map to pattern names
                            top3_patterns = []
                            for idx in top3_indices:
                                pattern_name = (
                                    self.ai_validator.reverse_pattern_map.get(
                                        idx, f"Pattern_{idx}"
                                    )
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

            # Check if signal was modified by AI
            original_signal = details.get("original_signal", final_signal)

            if final_signal == 0 and original_signal != 0:
                # AI rejected the signal
                viz_data["validation_passed"] = False
                viz_data["action"] = "rejected"

                # Collect rejection reasons
                reasons = []
                if not viz_data["sr_analysis"]["near_sr_level"]:
                    reasons.append("No nearby S/R level")
                if not viz_data["pattern_detected"]:
                    reasons.append("No pattern detected")
                if (
                    viz_data["pattern_confidence"]
                    < self.ai_validator.current_pattern_threshold
                ):
                    reasons.append(
                        f"Low confidence ({viz_data['pattern_confidence']:.1%})"
                    )

                viz_data["rejection_reasons"] = reasons

            elif final_signal != 0:
                # Signal was approved (or bypassed)
                viz_data["validation_passed"] = True

                # Check if it was a bypass
                if details.get("ai_bypassed", False):
                    viz_data["action"] = "bypassed"
                elif details.get("signal_quality", 0) >= self.strong_signal_bypass:
                    viz_data["action"] = "bypassed_strong_signal"
                else:
                    viz_data["action"] = "approved"
            else:
                # HOLD signal
                viz_data["action"] = "hold"

            return viz_data

        except Exception as e:
            logger.error(f"[VIZ] AI formatting failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "error": str(e),
                "action": "error",
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

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main aggregation logic with AI validation
        """
        self.stats["total_evaluations"] += 1
        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

            # STEP 1: Detect regime
            is_bull, regime_conf = self._detect_regime(df)
            regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"

            # STEP 2: Get strategy signals
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            ema_signal, ema_conf = self.s_ema.generate_signal(df)

            # Store originals for logging
            mr_original = mr_signal
            tf_original = tf_signal

            # ================================================================
            # FIX: Initialize signal_quality early for AI validation
            # ================================================================
            signal_quality = max(mr_conf, tf_conf)  # Preliminary quality estimate

            # ================================================================
            # STEP 3: AI VALIDATION (with circuit breaker)
            # ================================================================
            ai_bypass = False
            if self.ai_enabled and self.ai_validator is not None:
                # Check circuit breaker
                ai_bypass = self._check_ai_circuit_breaker()

                if self.ai_bypass_active and self.ai_bypass_cooldown > 0:
                    self.ai_bypass_cooldown -= 1

                # Validate signals if AI not bypassed
                if not ai_bypass:
                    # Validate Mean Reversion
                    if mr_signal != 0:
                        self.ai_stats["mr_signals_checked"] += 1

                        validated_mr, mr_details = self.ai_validator.validate_signal(
                            signal=mr_signal,
                            signal_details={
                                "strategy": "mean_reversion",
                                "confidence": mr_conf,
                                "regime": "bull" if is_bull else "bear",
                                "regime_confidence": regime_conf,
                                "asset": self.asset_type,
                                "signal_quality": signal_quality,  # ✓ Now defined
                            },
                            df=df,
                        )

                        if validated_mr == 0 and mr_signal != 0:
                            # AI rejected
                            self.ai_stats["mr_rejected"] += 1
                            self.ai_rejection_window.append(True)
                            logger.debug(
                                f"[AI] MR {mr_signal} rejected: {mr_details.get('ai_validation', 'unknown')}"
                            )
                            mr_signal = 0
                            mr_conf = 0.0
                        else:
                            self.ai_stats["mr_approved"] += 1
                            self.ai_rejection_window.append(False)

                    # Validate Trend Following
                    if tf_signal != 0:
                        self.ai_stats["tf_signals_checked"] += 1

                        validated_tf, tf_details = self.ai_validator.validate_signal(
                            signal=tf_signal,
                            signal_details={
                                "strategy": "trend_following",
                                "confidence": tf_conf,
                                "regime": "bull" if is_bull else "bear",
                                "regime_confidence": regime_conf,
                                "asset": self.asset_type,
                                "signal_quality": signal_quality,  # ✓ Now defined
                            },
                            df=df,
                        )

                        if validated_tf == 0 and tf_signal != 0:
                            # AI rejected
                            self.ai_stats["tf_rejected"] += 1
                            self.ai_rejection_window.append(True)
                            logger.debug(
                                f"[AI] TF {tf_signal} rejected: {tf_details.get('ai_validation', 'unknown')}"
                            )
                            tf_signal = 0
                            tf_conf = 0.0
                        else:
                            self.ai_stats["tf_approved"] += 1
                            self.ai_rejection_window.append(False)
            else:
                logger.debug(
                    "[AI] Validator not initialized or disabled, skipping AI validation."
                )

            # STEP 4: Calculate scores
            buy_score, buy_explanation, buy_agreement = self._calculate_score(
                target_signal=1,
                mr_signal=mr_signal,
                mr_conf=mr_conf,
                tf_signal=tf_signal,
                tf_conf=tf_conf,
                is_bull=is_bull,
            )

            sell_score, sell_explanation, sell_agreement = self._calculate_score(
                target_signal=-1,
                mr_signal=mr_signal,
                mr_conf=mr_conf,
                tf_signal=tf_signal,
                tf_conf=tf_conf,
                is_bull=is_bull,
            )

            # STEP 5: Dynamic thresholds
            adj_buy_thresh, adj_sell_thresh = self.calculate_regime_adjusted_thresholds(
                is_bull, regime_conf
            )

            base_buy_thresh = self.config["buy_threshold"]
            base_sell_thresh = self.config["sell_threshold"]

            # STEP 6: Make decision FIRST (before calculating signal_quality)
            final_signal = 0  # Initialize with default
            reasoning = ""

            if buy_score >= adj_buy_thresh and buy_score > sell_score:
                final_signal = 1
                reasoning = f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})"
            elif sell_score >= adj_sell_thresh and sell_score > buy_score:
                final_signal = -1
                reasoning = (
                    f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})"
                )
            else:
                final_signal = 0
                reasoning = f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"

            # Capture original signal before AI validation
            original_signal = final_signal

            # NOW calculate signal_quality (after final_signal is defined)
            base_buy = min(buy_score, 0.7)  # Cap contribution
            base_sell = min(sell_score, 0.7)
            raw_quality = max(base_buy, base_sell)

            # Apply agreement penalty
            if buy_agreement < 2 and sell_agreement < 2:
                raw_quality *= 0.7  # Penalize if no consensus

            # Apply regime alignment bonus
            if (final_signal == 1 and is_bull) or (final_signal == -1 and not is_bull):
                raw_quality *= 1.15  # Small boost for regime alignment

            # Cap at 1.0
            signal_quality = min(raw_quality, 1.0)
            min_quality = self.config["min_signal_quality"]

            # Apply quality filter to the decision
            if final_signal != 0 and signal_quality < min_quality:
                final_signal = 0
                reasoning = f"hold_lowquality (original:{reasoning}, quality:{signal_quality:.2f})"
                self.stats["hold_signals"] += 1
            elif final_signal == 1:
                self.stats["buy_signals"] += 1
                self.stats["signals_generated"] += 1
            elif final_signal == -1:
                self.stats["sell_signals"] += 1
                self.stats["signals_generated"] += 1
            else:
                self.stats["hold_signals"] += 1

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
                "buy_threshold": adj_buy_thresh,
                "sell_threshold": adj_sell_thresh,
                "buy_threshold_base": base_buy_thresh,
                "sell_threshold_base": base_sell_thresh,
                "mr_signal": mr_signal,
                "mr_signal_original": mr_original,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_signal_original": tf_original,
                "tf_confidence": tf_conf,
                "ema_regime_signal": ema_signal,
                "ema_regime_confidence": ema_conf,
                
                # ✅ NEW: Pass the full preset config (contains risk_overrides)
                "preset_config": self.config
            }

            if self.ai_enabled:
                details["ai_enabled"] = True
                details["ai_bypassed"] = ai_bypass
                if mr_original != mr_signal or tf_original != tf_signal:
                    details["ai_modified"] = True
                    details["ai_changes"] = {
                        "mr": (
                            f"{mr_original}→{mr_signal}"
                            if mr_original != mr_signal
                            else "unchanged"
                        ),
                        "tf": (
                            f"{tf_original}→{tf_signal}"
                            if tf_original != tf_signal
                            else "unchanged"
                        ),
                    }

                # ✅ CRITICAL FIX: Format AI validation data for visualization
                try:
                    ai_viz_data = self._format_ai_validation_for_viz(
                        final_signal=final_signal, details=details.copy(), df=df
                    )
                    details["ai_validation"] = ai_viz_data

                    if self.detailed_logging:
                        logger.info(
                            f"[AI VIZ] Pattern: {ai_viz_data.get('pattern_name', 'N/A')}"
                        )
                        logger.info(
                            f"[AI VIZ] Confidence: {ai_viz_data.get('pattern_confidence', 0):.2%}"
                        )
                        logger.info(
                            f"[AI VIZ] Action: {ai_viz_data.get('action', 'N/A')}"
                        )

                except Exception as e:
                    logger.error(f"[AI VIZ] Formatting failed: {e}")
                    details["ai_validation"] = {
                        "pattern_detected": False,
                        "validation_passed": False,
                        "error": str(e),
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
