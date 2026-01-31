"""
 AI Signal Validator with Realistic S/R Thresholds
========================================================
Key fixes:
1. Base S/R threshold: 0.5% → 2.5% (5x more realistic)
2. Directional S/R logic (BUY needs support, SELL needs resistance)
3. Strategy-aware adjustments (TF gets wider thresholds)
4. Better adaptive scaling based on volatility and regime
5. Comprehensive logging preserved
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HybridSignalValidator:
    """
    AI-powered signal validation with  realistic thresholds
    """

    # Pattern classifications
    BULLISH_PATTERNS = {
        "Engulfing",
        "Morning Star",
        "Hammer",
        "Inverted Hammer",
        "Three White Soldiers",
        "Piercing",
        "Harami",
        "Three Inside",
        "Dragonfly Doji",
        "Bullish Engulfing",
        "Bullish Harami",
        "Marubozu",  # ← ADDED (can be bullish if long white body)
    }

    BEARISH_PATTERNS = {
        "Evening Star",
        "Shooting Star",
        "Hanging Man",
        "Three Black Crows",
        "Dark Cloud",
        "Gravestone Doji",
        "Bearish Engulfing",
        "Three Outside",
        "Dark Cloud Cover",
        "Bearish Harami",
    }
    NEUTRAL_PATTERNS = {
        "Doji",  # Context-dependent
        "Spinning Top",
    }

    def __init__(
        self,
        analyst,
        sniper,
        pattern_id_map,
        sr_threshold_pct=0.0035,  # 0.35% instead of 2.5%
        pattern_confidence_min=0.65,
        use_ai_validation=True,
        enable_adaptive_thresholds=True,
        strong_signal_bypass_threshold=0.85,
        circuit_breaker_threshold=0.70,
        enable_detailed_logging=False,
    ):
        """
        Initialize validator with REALISTIC thresholds

        Args:
            sr_threshold_pct: Base S/R distance (2.5% = realistic for volatile assets)
            pattern_confidence_min: Minimum pattern confidence
            use_ai_validation: Toggle AI validation
            enable_adaptive_thresholds: Adjust based on market conditions
            strong_signal_bypass_threshold: Skip AI for very strong signals
            circuit_breaker_threshold: Bypass if rejection rate exceeds this
            enable_detailed_logging: Verbose logging
        """
        self.analyst = analyst
        self.sniper = sniper
        self.pattern_id_map = pattern_id_map
        self.reverse_pattern_map = {v: k for k, v in pattern_id_map.items()}
        #  Update pattern mapping to include Noise
        # self._update_pattern_mapping()

        # Configuration
        self.base_sr_threshold = sr_threshold_pct
        self.base_pattern_confidence = pattern_confidence_min
        self.use_ai_validation = use_ai_validation
        self.enable_adaptive = enable_adaptive_thresholds
        logger.info(f"[AI VALIDATOR] Initialized")
        logger.info(
            f"  Strong Signal Bypass: {strong_signal_bypass_threshold:.0%} (was 70%)"
        )
        logger.info(f"  Expected validation rate: ~40-60% of signals")
        self.strong_signal_bypass = strong_signal_bypass_threshold
        self.bypass_threshold = circuit_breaker_threshold
        self.detailed_logging = enable_detailed_logging

        # Current adaptive thresholds
        self.current_sr_threshold = sr_threshold_pct
        self.current_pattern_threshold = pattern_confidence_min

        # S/R cache
        self.sr_cache = {}
        self.last_sr_update = None
        self.sr_update_interval = 3600  # 1 hour

        # Circuit breaker
        self.rejection_window = deque(maxlen=50)
        self.bypass_mode = False
        self.bypass_cooldown = 0

        # Statistics tracking
        self.stats = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "rejected_no_sr": 0,
            "rejected_no_pattern": 0,
            "rejected_low_confidence": 0,
            "rejected_direction_mismatch": 0,
            "bypassed_strong_signal": 0,
            "bypassed_circuit_breaker": 0,
            "adaptive_adjustments": 0,
        }

        # Rejection reason tracking
        self.rejection_reasons = defaultdict(int)

        # Performance metrics per strategy
        self.strategy_stats = defaultdict(
            lambda: {
                "checks": 0,
                "approved": 0,
                "rejected": 0,
            }
        )

        # Historical validation data
        self.validation_history = deque(maxlen=1000)

        # Threshold adjustment history
        self.threshold_history = deque(maxlen=100)

        self._log_initialization()

    def _log_initialization(self):
        """Log initialization details"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🤖  AI SIGNAL VALIDATOR (Realistic Thresholds)")
        logger.info("=" * 70)
        logger.info(
            f"  Status:           {'ENABLED' if self.use_ai_validation else 'DISABLED'}"
        )
        logger.info(f"  Base S/R:         {self.base_sr_threshold:.2%} ( was 0.5%)")
        logger.info(f"  Base Pattern:     {self.base_pattern_confidence:.0%}")
        logger.info(f"  Adaptive:         {'ON' if self.enable_adaptive else 'OFF'}")
        logger.info(f"  Strong Bypass:    {self.strong_signal_bypass:.0%}")
        logger.info(f"  Circuit Breaker:  {self.bypass_threshold:.0%}")
        logger.info(f"  Detailed Logging: {'ON' if self.detailed_logging else 'OFF'}")
        logger.info(f"  Patterns Loaded:  {len(self.pattern_id_map)}")
        logger.info("=" * 70)
        logger.info("")

    def validate_signal(
        self, signal: int, signal_details: dict, df: pd.DataFrame
    ) -> Tuple[int, dict]:
        """
        Main validation with  realistic thresholds
        """
        validation_start = datetime.now()
        self.stats["total_checks"] += 1

        # Extract metadata
        asset = signal_details.get("asset", "UNKNOWN")
        strategy = signal_details.get("strategy", "UNKNOWN")
        original_signal = signal

        # Update strategy stats
        self.strategy_stats[strategy]["checks"] += 1

        # Skip validation if disabled or HOLD signal
        if not self.use_ai_validation:
            return self._skip_validation(signal, signal_details, "ai_disabled")

        if signal == 0:
            return self._skip_validation(signal, signal_details, "hold_signal")

        if self.detailed_logging:
            logger.info("")
            logger.info("─" * 70)
            logger.info(f"[AI VALIDATOR] Starting validation for {asset} {strategy}")
            logger.info(f"  Signal:     {self._signal_str(signal)}")
            logger.info(f"  Quality:    {signal_details.get('signal_quality', 0):.3f}")
            logger.info(f"  Regime:     {signal_details.get('regime', 'N/A')}")
            logger.info("─" * 70)



        # ============================================================
        # LAYER 1: Circuit Breaker Check
        # ============================================================
        if self.bypass_mode:
            self.bypass_cooldown -= 1
            if self.bypass_cooldown <= 0:
                self._reset_circuit_breaker()
            else:
                result = self._bypass_validation(
                    signal,
                    signal_details,
                    reason="circuit_breaker",
                    cooldown=self.bypass_cooldown,
                )
                self.stats["bypassed_circuit_breaker"] += 1
                self.strategy_stats[strategy]["approved"] += 1

                if self.detailed_logging:
                    logger.info(
                        f"  ⚡ BYPASS: Circuit breaker (cooldown: {self.bypass_cooldown})"
                    )

                return result

        # ============================================================
        # LAYER 2: Adaptive Threshold Adjustment ()
        # ============================================================
        if self.enable_adaptive:
            self._update_adaptive_thresholds_fixed(df, signal_details, strategy)

        if self.detailed_logging:
            logger.info(f"  Thresholds:")
            logger.info(
                f"    S/R:      {self.current_sr_threshold:.2%} (base: {self.base_sr_threshold:.2%})"
            )
            logger.info(
                f"    Pattern:  {self.current_pattern_threshold:.0%} (base: {self.base_pattern_confidence:.0%})"
            )

        # ============================================================
        # LAYER 3: Support/Resistance Check ( DIRECTIONAL LOGIC)
        # ============================================================
        current_price = float(df["close"].iloc[-1])
        sr_result = self._check_support_resistance_fixed(
            df, current_price, signal, threshold=self.current_sr_threshold
        )

        if self.detailed_logging:
            self._log_sr_check(sr_result)

        if not sr_result["near_level"]:
            result = self._reject_signal(
                signal_details, sr_result, None, reason="no_sr_level", strategy=strategy
            )
            self.stats["rejected_no_sr"] += 1
            self.rejection_reasons["no_sr_level"] += 1
            self.strategy_stats[strategy]["rejected"] += 1

            return result

        # ============================================================
        # LAYER 4: Pattern Confirmation Check
        # ============================================================
        pattern_result = self._check_pattern(
            df, signal, min_confidence=self.current_pattern_threshold
        )

        if self.detailed_logging:
            self._log_pattern_check(pattern_result)

        if not pattern_result["pattern_confirmed"]:
            result = self._reject_signal(
                signal_details,
                sr_result,
                pattern_result,
                reason=pattern_result["reason"],
                strategy=strategy,
            )
            self.stats["rejected_no_pattern"] += 1
            self.rejection_reasons[pattern_result["reason"]] += 1
            self.strategy_stats[strategy]["rejected"] += 1

            return result

        # ============================================================
        # VALIDATION PASSED!
        # ============================================================
        result = self._approve_signal(
            signal,
            signal_details,
            sr_result,
            pattern_result,
            strategy=strategy,
            validation_time=(datetime.now() - validation_start).total_seconds(),
        )

        self.stats["approved"] += 1
        self.strategy_stats[strategy]["approved"] += 1
        self.rejection_window.append(False)

        if self.detailed_logging:
            logger.info(f"  ✅ APPROVED: All validation layers passed")
            logger.info(
                f"  Confidence Boost: +{result[1].get('confidence_boost', 0):.2%}"
            )

        return result

    def _update_adaptive_thresholds_fixed(
        self, df: pd.DataFrame, signal_details: dict, strategy: str
    ):
        """
        More realistic adaptive threshold adjustments
        """
        regime = signal_details.get("regime", "BEAR")
        regime_confidence = signal_details.get("regime_confidence", 0.5)
        signal_quality = signal_details.get("signal_quality", 0.0)

        # Calculate volatility
        if len(df) >= 20:
            returns = df["close"].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.20

        # Store old thresholds
        old_sr = self.current_sr_threshold
        old_pattern = self.current_pattern_threshold

        # ============================================================
        # S/R THRESHOLD ADJUSTMENT ()
        # ============================================================

        # Base: 2.5%
        sr_threshold = self.base_sr_threshold

        # 1. Strategy-specific multipliers
        if strategy == "mean_reversion":
            # MR relies heavily on S/R - keep strict
            sr_threshold *= 1.0
        elif strategy == "trend_following":
            # TF can work further from S/R - more lenient
            sr_threshold *= 1.5  # 2.5% → 3.75%
        else:
            # Default strategies
            sr_threshold *= 1.2

        # 2. Volatility adjustment
        # High volatility = wider threshold needed
        if volatility > 0.40:
            sr_threshold *= 1.3
        elif volatility > 0.30:
            sr_threshold *= 1.15
        elif volatility < 0.15:
            sr_threshold *= 0.9  # Low vol = tighter is OK

        # 3. Regime adjustment
        if "BULL" in regime.upper():
            # Bull markets - be more lenient on BUY signals
            sr_threshold *= 1.2
        elif "BEAR" in regime.upper():
            # Bear markets - be strict
            if signal_quality < 0.6:
                sr_threshold *= 0.85

        # 4. Signal quality scaling
        if signal_quality > 0.7:
            sr_threshold *= 1.2  # High quality = more leeway

        # 5. Rejection rate adjustment
        if len(self.rejection_window) >= 20:
            rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
            if rejection_rate > 0.60:
                sr_threshold *= 1.4  # Too many rejections = relax
                self.stats["adaptive_adjustments"] += 1

        # Safety bounds (2x wider than before)
        self.current_sr_threshold = np.clip(sr_threshold, 0.015, 0.060)  # 1.5% - 6%

        # ============================================================
        # PATTERN THRESHOLD ADJUSTMENT
        # ============================================================

        pattern_threshold = self.base_pattern_confidence

        # Regime-based adjustment
        regime_strength = (regime_confidence - 0.5) * 2  # 0.5-1.0 → 0-1
        regime_strength = max(0.0, min(1.0, regime_strength))

        if "BULL" in regime.upper():
            pattern_threshold *= 0.90 + regime_strength * 0.05
        else:
            pattern_threshold *= 0.95 + regime_strength * 0.10

        # Rejection rate adjustment
        if len(self.rejection_window) >= 20:
            rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
            if rejection_rate > 0.60:
                pattern_threshold *= 0.85

        # Safety bounds
        self.current_pattern_threshold = np.clip(pattern_threshold, 0.40, 0.75)

        # ============================================================
        # LOGGING
        # ============================================================

        sr_change = abs(self.current_sr_threshold - old_sr) / old_sr
        pattern_change = abs(self.current_pattern_threshold - old_pattern) / old_pattern

        if sr_change > 0.15 or pattern_change > 0.15:
            self.threshold_history.append(
                {
                    "timestamp": datetime.now(),
                    "sr_threshold": self.current_sr_threshold,
                    "pattern_threshold": self.current_pattern_threshold,
                    "volatility": volatility,
                    "regime": regime,
                    "strategy": strategy,
                }
            )

            if self.detailed_logging:
                logger.debug(f"  [ADAPTIVE] Threshold adjustment:")
                logger.debug(
                    f"    S/R: {old_sr:.2%} → {self.current_sr_threshold:.2%} ({sr_change:+.1%})"
                )
                logger.debug(
                    f"    Pattern: {old_pattern:.0%} → {self.current_pattern_threshold:.0%} ({pattern_change:+.1%})"
                )
                logger.debug(
                    f"    Factors: vol={volatility:.2f}, regime={regime}, quality={signal_quality:.2f}"
                )

    def _check_support_resistance_fixed(
        self, df: pd.DataFrame, current_price: float, signal: int, threshold: float
    ) -> dict:
        """
         Directional S/R logic
        BUY needs support BELOW, SELL needs resistance ABOVE
        """
        # Update S/R levels if cache stale
        now = pd.Timestamp.now()
        if (
            self.last_sr_update is None
            or (now - self.last_sr_update).total_seconds() > self.sr_update_interval
        ):
            self._update_sr_levels(df)
            self.last_sr_update = now

        all_levels = self.sr_cache.get("levels", [])

        if not all_levels:
            return {
                "near_level": False,
                "level_type": "none",
                "nearest_level": None,
                "distance_pct": None,
                "threshold_used": threshold,
                "all_levels": [],
                "total_levels_found": 0,
                "reason": "no_sr_levels_found",
            }

        # ============================================================
        # DIRECTIONAL LOGIC ()
        # ============================================================

        if signal == 1:  # BUY signal
            # Look for SUPPORT levels BELOW current price
            relevant_levels = [l for l in all_levels if l < current_price]
            level_type = "support"

            if not relevant_levels:
                # No support below - check if we're AT a level
                any_level_distances = [
                    abs(current_price - l) / current_price for l in all_levels
                ]
                min_any_dist = (
                    min(any_level_distances) if any_level_distances else float("inf")
                )

                if min_any_dist < threshold:
                    closest_idx = np.argmin(any_level_distances)
                    closest = all_levels[closest_idx]
                    return {
                        "near_level": True,
                        "level_type": "boundary",
                        "nearest_level": closest,
                        "distance_pct": min_any_dist * 100,
                        "threshold_used": threshold,
                        "all_levels": all_levels[:3],
                        "total_levels_found": len(all_levels),
                        "reason": f"at_level_${closest:.2f}",
                    }

                return {
                    "near_level": False,
                    "level_type": level_type,
                    "nearest_level": None,
                    "distance_pct": None,
                    "threshold_used": threshold,
                    "all_levels": all_levels[:3],
                    "total_levels_found": len(all_levels),
                    "reason": "no_support_below",
                }

        else:  # SELL signal
            # Look for RESISTANCE levels ABOVE current price
            relevant_levels = [l for l in all_levels if l > current_price]
            level_type = "resistance"

            if not relevant_levels:
                # No resistance above - check if we're AT a level
                any_level_distances = [
                    abs(current_price - l) / current_price for l in all_levels
                ]
                min_any_dist = (
                    min(any_level_distances) if any_level_distances else float("inf")
                )

                if min_any_dist < threshold:
                    closest_idx = np.argmin(any_level_distances)
                    closest = all_levels[closest_idx]
                    return {
                        "near_level": True,
                        "level_type": "boundary",
                        "nearest_level": closest,
                        "distance_pct": min_any_dist * 100,
                        "threshold_used": threshold,
                        "all_levels": all_levels[:3],
                        "total_levels_found": len(all_levels),
                        "reason": f"at_level_${closest:.2f}",
                    }

                return {
                    "near_level": False,
                    "level_type": level_type,
                    "nearest_level": None,
                    "distance_pct": None,
                    "threshold_used": threshold,
                    "all_levels": all_levels[:3],
                    "total_levels_found": len(all_levels),
                    "reason": "no_resistance_above",
                }

        # Find closest relevant level
        distances = [
            (abs(current_price - level) / current_price, level)
            for level in relevant_levels
        ]
        min_distance_pct, nearest_level = min(distances)

        near_level = min_distance_pct < threshold

        return {
            "near_level": near_level,
            "level_type": level_type,
            "nearest_level": nearest_level,
            "distance_pct": min_distance_pct * 100,
            "threshold_used": threshold,
            "all_levels": relevant_levels[:3],
            "total_levels_found": len(relevant_levels),
            "reason": (
                f"near_{level_type}_${nearest_level:.2f}"
                if near_level
                else f"{level_type}_too_far_${nearest_level:.2f}"
            ),
        }

    # ============================================================
    # HELPER METHODS (unchanged but included for completeness)
    # ============================================================

    def _signal_str(self, signal: int) -> str:
        """Convert signal to readable string"""
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")

    def _skip_validation(
        self, signal: int, details: dict, reason: str
    ) -> Tuple[int, dict]:
        """Skip validation and return original signal"""
        return signal, {
            **details,
            "ai_validation": f"skipped_{reason}",
            "final_signal": signal,
        }

    def _bypass_validation(
        self, signal: int, details: dict, reason: str, **kwargs
    ) -> Tuple[int, dict]:
        """Bypass validation and return approved signal"""
        return signal, {
            **details,
            "ai_validation": f"bypassed_{reason}",
            "ai_bypass_reason": ", ".join(f"{k}={v}" for k, v in kwargs.items()),
            "final_signal": signal,
        }

    def _reject_signal(
        self,
        signal_details: dict,
        sr_result: dict,
        pattern_result: Optional[dict],
        reason: str,
        strategy: str,
    ) -> Tuple[int, dict]:
        """Reject signal and return HOLD"""
        self.rejection_window.append(True)
        self._check_circuit_breaker()

        self.validation_history.append(
            {
                "timestamp": datetime.now(),
                "strategy": strategy,
                "original_signal": signal_details.get("signal", 0),
                "result": "rejected",
                "reason": reason,
                "sr_distance": sr_result.get("distance_pct"),
                "pattern": (
                    pattern_result.get("pattern_name") if pattern_result else None
                ),
                "confidence": (
                    pattern_result.get("confidence") if pattern_result else None
                ),
            }
        )

        result_details = {
            **signal_details,
            "ai_validation": "rejected",
            "ai_rejection_reason": reason,
            "ai_sr_check": sr_result,
            "ai_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.current_pattern_threshold:.0%}",
            },
            "final_signal": 0,
        }

        if pattern_result:
            result_details["ai_pattern_check"] = pattern_result

        if self.detailed_logging:
            logger.warning(f"  ❌ REJECTED: {reason}")
            if sr_result.get("distance_pct"):
                logger.warning(f"     S/R distance: {sr_result['distance_pct']:.2f}%")
            if pattern_result:
                logger.warning(
                    f"     Pattern: {pattern_result.get('pattern_name', 'N/A')}"
                )
                logger.warning(
                    f"     Confidence: {pattern_result.get('confidence', 0):.0%}"
                )

        return 0, result_details

    def _approve_signal(
        self,
        signal: int,
        signal_details: dict,
        sr_result: dict,
        pattern_result: dict,
        strategy: str,
        validation_time: float,
    ) -> Tuple[int, dict]:
        """Approve signal and add confidence boost"""
        self.rejection_window.append(False)

        pattern_conf = pattern_result.get("confidence", 0)
        base_boost = 0.10

        if pattern_conf > 0.80:
            boost = base_boost + 0.05
        elif pattern_conf > 0.65:
            boost = base_boost
        else:
            boost = base_boost - 0.02

        self.validation_history.append(
            {
                "timestamp": datetime.now(),
                "strategy": strategy,
                "original_signal": signal,
                "result": "approved",
                "reason": "all_layers_passed",
                "sr_distance": sr_result.get("distance_pct"),
                "pattern": pattern_result.get("pattern_name"),
                "confidence": pattern_result.get("confidence"),
                "validation_time_ms": validation_time * 1000,
            }
        )

        return signal, {
            **signal_details,
            "ai_validation": "approved_all_layers",
            "ai_sr_check": sr_result,
            "ai_pattern_check": pattern_result,
            "ai_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.current_pattern_threshold:.0%}",
            },
            "ai_validation_time_ms": validation_time * 1000,
            "final_signal": signal,
            "confidence_boost": boost,
        }

    def _log_sr_check(self, result: dict):
        """Log S/R check results"""
        near = result.get("near_level", False)
        level_type = result.get("level_type", "N/A")
        nearest = result.get("nearest_level")
        distance = result.get("distance_pct")
        reason = result.get("reason", "N/A")

        if near and nearest is not None:
            logger.info(f"  S/R Check: ✅ {reason} (dist: {distance:.2f}%)")
        else:
            if distance is not None:
                logger.info(f"  S/R Check: ❌ {reason} (dist: {distance:.2f}%)")
            else:
                logger.info(f"  S/R Check: ❌ {reason}")

    def _check_circuit_breaker(self):
        """Activate bypass if rejection rate too high"""
        if len(self.rejection_window) < 30:
            return

        rejection_rate = sum(self.rejection_window) / len(self.rejection_window)

        if rejection_rate > self.bypass_threshold and not self.bypass_mode:
            self.bypass_mode = True
            self.bypass_cooldown = 15

            logger.warning("")
            logger.warning("=" * 70)
            logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
            logger.warning(
                f"   Rejection rate: {rejection_rate:.0%} > {self.bypass_threshold:.0%}"
            )
            logger.warning(
                f"   AI validation DISABLED for next {self.bypass_cooldown} signals"
            )

            top_reasons = sorted(
                self.rejection_reasons.items(), key=lambda x: x[1], reverse=True
            )[:3]

            if top_reasons:
                logger.warning("   Top rejection reasons:")
                for reason, count in top_reasons:
                    logger.warning(f"     - {reason}: {count} times")

            logger.warning("=" * 70)
            logger.warning("")

    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.bypass_mode = False
        self.rejection_window.clear()

        logger.info("🔄 AI circuit breaker reset - validation RE-ENABLED")

    def _update_sr_levels(self, df: pd.DataFrame):
        """More robust S/R level extraction"""
        pivots = self._extract_pivots(df, window=7)

        #  Lower threshold from 5 to 3 pivots
        if len(pivots) < 3:
            logger.warning(
                f"[SR UPDATE] Only {len(pivots)} pivots - using price quantiles as fallback"
            )

            # Fallback: Use price distribution quantiles as S/R levels
            closes = df["close"].values
            levels = np.percentile(closes, [10, 25, 50, 75, 90]).tolist()

            self.sr_cache = {
                "levels": sorted(levels),
                "updated_at": datetime.now(),
                "pivot_count": 0,
                "fallback_mode": True,
            }

            logger.info(f"[SR UPDATE] Fallback: {len(levels)} quantile-based levels")
            return

        try:
            levels = self.analyst.get_support_resistance_levels(
                pivot_points=pivots,
                highs=df["high"].values,
                lows=df["low"].values,
                closes=df["close"].values,
                n_levels=7,
            )

            #  If clustering fails, use pivots directly
            if not levels:
                logger.warning(
                    "[SR UPDATE] Clustering returned no levels - using raw pivots"
                )
                levels = sorted(np.unique(pivots).tolist())

            self.sr_cache = {
                "levels": levels,
                "updated_at": datetime.now(),
                "pivot_count": len(pivots),
                "fallback_mode": False,
            }

            logger.debug(f"[SR UPDATE] {len(levels)} levels from {len(pivots)} pivots")

        except Exception as e:
            logger.error(f"[SR UPDATE] Failed: {e}")

            # Emergency fallback: use price range
            closes = df["close"].values
            levels = np.percentile(closes, [10, 30, 50, 70, 90]).tolist()

            self.sr_cache = {
                "levels": sorted(levels),
                "updated_at": datetime.now(),
                "pivot_count": 0,
                "fallback_mode": True,
            }

            logger.warning(
                f"[SR UPDATE] Emergency fallback: {len(levels)} quantile levels"
            )

    def _extract_pivots(self, df: pd.DataFrame, window=7) -> np.ndarray:
        """Extract more pivots with lower window"""
        highs = df["high"].values
        lows = df["low"].values
        pivots = []

        #  Try smaller window if too few pivots
        for window_size in [window, 5, 3]:
            pivots = []

            for i in range(window_size, len(df) - window_size):
                # High pivot
                if highs[i] == max(highs[i - window_size : i + window_size + 1]):
                    pivots.append(highs[i])

                # Low pivot
                if lows[i] == min(lows[i - window_size : i + window_size + 1]):
                    pivots.append(lows[i])

            if len(pivots) >= 3:
                logger.debug(
                    f"[PIVOT] Found {len(pivots)} pivots with window={window_size}"
                )
                break

        return np.array(pivots) if pivots else np.array([])

    def _resolve_pattern_name(self, pattern_id: int) -> str:
        """
         Multi-stage pattern name resolution

        Tries multiple lookup strategies to find the pattern name
        """
        # Strategy 1: Check reverse_pattern_map (most reliable)
        if pattern_id in self.reverse_pattern_map:
            return self.reverse_pattern_map[pattern_id]

        # Strategy 2: Check pattern_id_map values (in case reverse map is incomplete)
        for name, pid in self.pattern_id_map.items():
            if pid == pattern_id:
                logger.debug(
                    f"[PATTERN] Found {name} for ID {pattern_id} via forward map"
                )
                return name

        # Strategy 3: Check if it's a known pattern by common IDs
        # (Add your known mappings here if pattern_id_map is incomplete)
        KNOWN_PATTERNS = {
            1: "Hammer",
            2: "Inverted Hammer",
            3: "Engulfing",
            4: "Harami",
            5: "Piercing",
            6: "Morning Star",
            7: "Evening Star",
            8: "Shooting Star",
            9: "Hanging Man",
            10: "Dark Cloud",  # ← YOUR PATTERN ID 10!
            # Add more as needed
        }

        if pattern_id in KNOWN_PATTERNS:
            name = KNOWN_PATTERNS[pattern_id]
            logger.warning(
                f"[PATTERN] Using hardcoded mapping: ID {pattern_id} → {name}"
            )
            return name

        # Strategy 4: Give up and return generic name
        logger.warning(f"[PATTERN] Unknown pattern ID: {pattern_id}")
        logger.warning(
            f"[PATTERN] Available IDs: {sorted(self.reverse_pattern_map.keys())}"
        )
        return f"Unknown_Pattern_{pattern_id}"

    def _check_pattern(
        self, df: pd.DataFrame, signal: int, min_confidence: float = 0.60
    ) -> dict:
        """
        Pattern confirmation with proper noise handling
        """
        try:
            # Get last 15 candles for pattern detection
            if len(df) < 15:
                return {
                    "pattern_confirmed": False,
                    "reason": "insufficient_data",
                    "pattern_name": None,
                    "pattern_id": None,
                    "confidence": 0.0,
                }

            # Extract OHLC snippet
            snippet = df[["open", "high", "low", "close"]].iloc[-15:].values

            # Normalize (percentage change from first candle)
            first_open = snippet[0, 0]
            if first_open <= 0:
                return {
                    "pattern_confirmed": False,
                    "reason": "invalid_data",
                    "pattern_name": None,
                    "pattern_id": None,
                    "confidence": 0.0,
                }

            snippet_norm = snippet / first_open - 1
            snippet_input = snippet_norm.reshape(1, 15, 4)

            # Get AI prediction
            predicted_id, confidence = self.sniper.predict_single(snippet_input)
            pattern_name = self.reverse_pattern_map.get(predicted_id, "Unknown")

            if self.detailed_logging:
                logger.debug(
                    f"[PATTERN] Detected: {pattern_name} (ID={predicted_id}, "
                    f"confidence={confidence:.2%})"
                )

            # ============================================================
            # CRITICAL FIX 1: Handle Noise Class (ID 0)
            # ============================================================
            if predicted_id == 0:
                if self.detailed_logging:
                    logger.info(f"[PATTERN] No clear pattern (Noise class)")

                # Don't reject immediately - check if confidence is borderline
                if confidence < 0.70:
                    # Low confidence noise = truly no pattern, might still be valid
                    return {
                        "pattern_confirmed": True,  # Allow signal through
                        "reason": "no_strong_pattern_but_acceptable",
                        "pattern_name": "Noise",
                        "pattern_id": 0,
                        "confidence": confidence,
                        "warning": "No pattern detected - relying on indicators only",
                    }
                else:
                    # High confidence noise = definitely no pattern
                    return {
                        "pattern_confirmed": False,
                        "reason": "no_pattern_detected",
                        "pattern_name": "Noise",
                        "pattern_id": 0,
                        "confidence": confidence,
                    }

            # ============================================================
            # CRITICAL FIX 2: Check Pattern-Signal Alignment
            # ============================================================
            alignment_result = self._check_pattern_signal_alignment(
                predicted_id, pattern_name, signal, confidence
            )

            if not alignment_result["aligned"]:
                return {
                    "pattern_confirmed": False,
                    "reason": alignment_result["reason"],
                    "pattern_name": pattern_name,
                    "pattern_id": predicted_id,
                    "confidence": confidence,
                    "details": alignment_result["details"],
                }

            # ============================================================
            # Check Confidence Threshold
            # ============================================================
            if confidence < min_confidence:
                return {
                    "pattern_confirmed": False,
                    "reason": "low_confidence",
                    "pattern_name": pattern_name,
                    "pattern_id": predicted_id,
                    "confidence": confidence,
                    "required_confidence": min_confidence,
                }

            # ============================================================
            # Pattern Confirmed!
            # ============================================================
            return {
                "pattern_confirmed": True,
                "reason": "pattern_matches_signal",
                "pattern_name": pattern_name,
                "pattern_id": predicted_id,
                "confidence": confidence,
                "alignment": "perfect",
            }

        except Exception as e:
            logger.error(f"[PATTERN] Error checking pattern: {e}", exc_info=True)
            return {
                "pattern_confirmed": False,
                "reason": "error",
                "pattern_name": None,
                "pattern_id": None,
                "confidence": 0.0,
                "error": str(e),
            }

    def _check_pattern_signal_alignment(
        self, pattern_id: int, pattern_name: str, signal: int, confidence: float
    ) -> dict:
        """
        CRITICAL FIX: Check if detected pattern aligns with signal direction
        """
        # Define pattern groups
        BULLISH_PATTERNS = {
            # Pattern names and IDs that indicate bullish reversal/continuation
            "Engulfing",
            "Morning Star",
            "Hammer",
            "Inverted Hammer",
            "Three White Soldiers",
            "Dragonfly Doji",
            "Piercing",
            "Harami",
        }

        BEARISH_PATTERNS = {
            # Pattern names and IDs that indicate bearish reversal/continuation
            "Evening Star",
            "Shooting Star",
            "Hanging Man",
            "Three Black Crows",
            "Gravestone Doji",
            "Dark Cloud",
        }

        NEUTRAL_PATTERNS = {
            # Patterns that can go either way
            "Doji",
            "Marubozu",
        }

        # Check if pattern matches signal
        is_bullish_pattern = pattern_name in BULLISH_PATTERNS
        is_bearish_pattern = pattern_name in BEARISH_PATTERNS
        is_neutral_pattern = pattern_name in NEUTRAL_PATTERNS

        # BUY signal (1) should have bullish or neutral patterns
        if signal == 1:
            if is_bearish_pattern:
                return {
                    "aligned": False,
                    "reason": "bearish_pattern_for_buy",
                    "details": f"Detected {pattern_name} (bearish) but signal is BUY",
                }

            # Neutral patterns need higher confidence
            if is_neutral_pattern and confidence < 0.75:
                return {
                    "aligned": False,
                    "reason": "neutral_pattern_low_confidence",
                    "details": f"{pattern_name} is neutral, needs 75%+ confidence for BUY",
                }

            return {
                "aligned": True,
                "reason": "bullish_pattern_matches_buy",
                "details": f"{pattern_name} supports BUY signal",
            }

        # SELL signal (-1) should have bearish or neutral patterns
        elif signal == -1:
            if is_bullish_pattern:
                return {
                    "aligned": False,
                    "reason": "bullish_pattern_for_sell",
                    "details": f"Detected {pattern_name} (bullish) but signal is SELL",
                }

            # Neutral patterns need higher confidence
            if is_neutral_pattern and confidence < 0.75:
                return {
                    "aligned": False,
                    "reason": "neutral_pattern_low_confidence",
                    "details": f"{pattern_name} is neutral, needs 75%+ confidence for SELL",
                }

            return {
                "aligned": True,
                "reason": "bearish_pattern_matches_sell",
                "details": f"{pattern_name} supports SELL signal",
            }

        # HOLD signal (0) - shouldn't reach here but handle gracefully
        else:
            return {
                "aligned": True,
                "reason": "hold_signal",
                "details": "No pattern validation needed for HOLD",
            }

    def _update_pattern_mapping(self):
        """
         Ensure pattern mapping includes Noise class
        Call this in __init__ after loading the model
        """
        try:
            # Load pattern mapping
            mapping_path = self.model_path.replace(".weights.h5", "_mapping.pkl")

            with open(mapping_path, "rb") as f:
                self.pattern_map = pickle.load(f)

            # CRITICAL FIX: Add Noise class if missing
            if "Noise" not in self.pattern_map:
                logger.warning(
                    "[AI INIT] 'Noise' class missing from mapping - adding it"
                )
                self.pattern_map["Noise"] = 0

                # Save corrected mapping
                with open(mapping_path, "wb") as f:
                    pickle.dump(self.pattern_map, f)

                logger.info("[AI INIT] ✓ Updated mapping file with Noise class")

            # Create reverse mapping
            self.reverse_pattern_map = {v: k for k, v in self.pattern_map.items()}

            logger.info(f"[AI INIT] Loaded {len(self.pattern_map)} patterns:")
            for name, pid in sorted(self.pattern_map.items(), key=lambda x: x[1]):
                logger.info(f"  {pid}: {name}")

        except Exception as e:
            logger.error(f"[AI INIT] Failed to load pattern mapping: {e}")
            raise

    def _log_pattern_check(self, pattern_result: dict):
        """Enhanced logging for pattern checks"""
        pattern_name = pattern_result.get("pattern_name", "Unknown")
        pattern_id = pattern_result.get("pattern_id", "N/A")
        confidence = pattern_result.get("confidence", 0)
        confirmed = pattern_result.get("pattern_confirmed", False)
        reason = pattern_result.get("reason", "N/A")

        status = "✅ CONFIRMED" if confirmed else "❌ REJECTED"

        logger.info(f"  Pattern Check:")
        logger.info(f"    Pattern:    {pattern_name} (ID={pattern_id})")
        logger.info(f"    Confidence: {confidence:.2%}")
        logger.info(f"    Status:     {status}")
        logger.info(f"    Reason:     {reason}")

        if "warning" in pattern_result:
            logger.warning(f"    ⚠️  {pattern_result['warning']}")

        if "details" in pattern_result:
            logger.info(f"    Details:    {pattern_result['details']}")

    def adjust_validation_strictness(self, recent_performance: dict):
        """
        ADAPTIVE: Adjust thresholds based on recent performance
        Call this periodically (e.g., every 50 signals)
        """
        win_rate = recent_performance.get("win_rate", 0.5)
        rejection_rate = len([r for r in self.rejection_window if r]) / max(
            len(self.rejection_window), 1
        )

        # If rejection rate is very high but win rate is good, loosen thresholds
        if rejection_rate > 0.70 and win_rate > 0.55:
            self.current_pattern_threshold = max(
                0.50, self.current_pattern_threshold - 0.05
            )
            self.current_sr_threshold = max(0.01, self.current_sr_threshold - 0.005)

            logger.warning(
                f"[ADAPTIVE] High rejection ({rejection_rate:.0%}) with good win rate "
                f"({win_rate:.0%}) - loosening thresholds"
            )
            logger.info(f"  Pattern threshold: {self.current_pattern_threshold:.0%}")
            logger.info(f"  S/R threshold: {self.current_sr_threshold:.2%}")

        # If rejection rate is low but win rate is poor, tighten thresholds
        elif rejection_rate < 0.30 and win_rate < 0.45:
            self.current_pattern_threshold = min(
                0.80, self.current_pattern_threshold + 0.05
            )
            self.current_sr_threshold = min(0.05, self.current_sr_threshold + 0.005)

            logger.warning(
                f"[ADAPTIVE] Low rejection ({rejection_rate:.0%}) with poor win rate "
                f"({win_rate:.0%}) - tightening thresholds"
            )
            logger.info(f"  Pattern threshold: {self.current_pattern_threshold:.0%}")
            logger.info(f"  S/R threshold: {self.current_sr_threshold:.2%}")

    def get_statistics(self) -> dict:
        """Return comprehensive validation statistics"""
        total = max(self.stats["total_checks"], 1)

        base_stats = {
            "ai_enabled": self.use_ai_validation,
            "total_checks": self.stats["total_checks"],
            "approved": self.stats["approved"],
            "rejected": self.stats["rejected"],
            "approval_rate": f"{(self.stats['approved']/total)*100:.1f}%",
            "rejection_rate": f"{(self.stats['rejected']/total)*100:.1f}%",
            "rejection_breakdown": {
                "no_sr_level": self.stats["rejected_no_sr"],
                "no_pattern": self.stats["rejected_no_pattern"],
                "low_confidence": self.stats.get("rejected_low_confidence", 0),
                "direction_mismatch": self.stats.get("rejected_direction_mismatch", 0),
            },
            "bypasses": {
                "strong_signal": self.stats["bypassed_strong_signal"],
                "circuit_breaker": self.stats["bypassed_circuit_breaker"],
            },
            "adaptive_adjustments": self.stats["adaptive_adjustments"],
            "current_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_confidence": f"{self.current_pattern_threshold:.0%}",
            },
            "circuit_breaker": {
                "active": self.bypass_mode,
                "cooldown": self.bypass_cooldown if self.bypass_mode else 0,
            },
        }

        # Add top rejection reasons
        if self.rejection_reasons:
            top_reasons = sorted(
                self.rejection_reasons.items(), key=lambda x: x[1], reverse=True
            )[:5]
            base_stats["top_rejection_reasons"] = {
                reason: count for reason, count in top_reasons
            }

        # Add per-strategy stats
        if self.strategy_stats:
            base_stats["per_strategy"] = {}
            for strategy, strat_stats in self.strategy_stats.items():
                total_strat = max(strat_stats["checks"], 1)
                base_stats["per_strategy"][strategy] = {
                    "checks": strat_stats["checks"],
                    "approved": strat_stats["approved"],
                    "rejected": strat_stats["rejected"],
                    "approval_rate": f"{(strat_stats['approved']/total_strat)*100:.1f}%",
                }

        return base_stats

    def diagnose_pattern_mapping(self):
        """
        Run this once at startup to verify pattern mapping is correct

        Usage:
            validator = HybridSignalValidator(...)
            validator.diagnose_pattern_mapping()
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔍 PATTERN MAPPING DIAGNOSTIC")
        logger.info("=" * 70)

        # Check map sizes
        logger.info(f"pattern_id_map size: {len(self.pattern_id_map)}")
        logger.info(f"reverse_pattern_map size: {len(self.reverse_pattern_map)}")
        logger.info(f"BULLISH_PATTERNS size: {len(self.BULLISH_PATTERNS)}")
        logger.info(f"BEARISH_PATTERNS size: {len(self.BEARISH_PATTERNS)}")
        logger.info("")

        # Show sample mappings
        logger.info("Sample pattern_id_map (name → ID):")
        for i, (name, pid) in enumerate(sorted(self.pattern_id_map.items())[:10]):
            classification = (
                "BULLISH"
                if name in self.BULLISH_PATTERNS
                else "BEARISH" if name in self.BEARISH_PATTERNS else "UNCLASSIFIED"
            )
            logger.info(f"  {name:25s} → ID {pid:3d} [{classification}]")

        if len(self.pattern_id_map) > 10:
            logger.info(f"  ... and {len(self.pattern_id_map) - 10} more")
        logger.info("")

        # Show reverse map
        logger.info("Sample reverse_pattern_map (ID → name):")
        for i, (pid, name) in enumerate(sorted(self.reverse_pattern_map.items())[:10]):
            logger.info(f"  ID {pid:3d} → {name}")

        if len(self.reverse_pattern_map) > 10:
            logger.info(f"  ... and {len(self.reverse_pattern_map) - 10} more")
        logger.info("")

        # Check for missing classifications
        all_pattern_names = set(self.pattern_id_map.keys())
        classified = self.BULLISH_PATTERNS | self.BEARISH_PATTERNS
        unclassified = all_pattern_names - classified

        if unclassified:
            logger.warning(f"⚠️  {len(unclassified)} patterns UNCLASSIFIED:")
            for name in sorted(unclassified):
                pid = self.pattern_id_map.get(name, "?")
                logger.warning(f"  - {name} (ID: {pid})")
            logger.warning("")
            logger.warning("Add these to BULLISH_PATTERNS or BEARISH_PATTERNS!")
        else:
            logger.info("✅ All patterns are classified")

        logger.info("")
        logger.info("=" * 70)
        logger.info("")

    def diagnose_pattern_mapping(self):
        """Diagnostic method to check pattern mapping"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔍 PATTERN MAPPING DIAGNOSTIC")
        logger.info("=" * 70)
        logger.info(f"pattern_id_map entries: {len(self.pattern_id_map)}")
        logger.info(f"reverse_pattern_map entries: {len(self.reverse_pattern_map)}")
        logger.info("")
        logger.info("Bullish patterns defined: {len(self.BULLISH_PATTERNS)}")
        logger.info("Bearish patterns defined: {len(self.BEARISH_PATTERNS)}")
        logger.info("")
        logger.info("Sample pattern_id_map entries:")
        for i, (name, pid) in enumerate(list(self.pattern_id_map.items())[:5]):
            logger.info(f"  {name} → ID {pid}")
        logger.info("")
        logger.info("Sample reverse_pattern_map entries:")
        for i, (pid, name) in enumerate(list(self.reverse_pattern_map.items())[:5]):
            logger.info(f"  ID {pid} → {name}")
        logger.info("=" * 70)
        logger.info("")
