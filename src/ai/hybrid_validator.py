"""
ENHANCED AI Signal Validator with Comprehensive Logging
=====================================================
Key Improvements:
1. Detailed validation flow logging
2. Performance metrics tracking
3. Better error handling and fallbacks
4. Signal quality scoring system
5. Adaptive threshold monitoring
6. Rejection reason analytics
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
    AI-powered signal validation with detailed logging and monitoring
    """
    
    # Pattern classifications (unchanged)
    BULLISH_PATTERNS = {
        'Engulfing', 'Morning Star', 'Hammer', 'Inverted Hammer',
        'Three White Soldiers', 'Piercing', 'Harami', 'Three Inside',
        'Dragonfly Doji', 'Bullish Engulfing'
    }
    
    BEARISH_PATTERNS = {
        'Evening Star', 'Shooting Star', 'Hanging Man', 
        'Three Black Crows', 'Dark Cloud', 'Gravestone Doji',
        'Bearish Engulfing', 'Three Outside'
    }
    
    def __init__(
        self, 
        analyst, 
        sniper,
        pattern_id_map,
        sr_threshold_pct=0.015,
        pattern_confidence_min=0.50,
        use_ai_validation=True,
        enable_adaptive_thresholds=True,
        strong_signal_bypass_threshold=0.75,
        circuit_breaker_threshold=0.70,
        enable_detailed_logging=True,
    ):
        """
        Initialize validator with enhanced monitoring
        
        Args:
            analyst: DynamicAnalyst instance for S/R detection
            sniper: OHLCSniper instance for pattern recognition
            pattern_id_map: Dict mapping pattern IDs to names
            sr_threshold_pct: Base distance to S/R level (adapts with volatility)
            pattern_confidence_min: Base pattern confidence (adapts with regime)
            use_ai_validation: Toggle AI validation
            enable_adaptive_thresholds: Adjust thresholds based on market conditions
            strong_signal_bypass_threshold: Skip AI for very strong signals
            circuit_breaker_threshold: Bypass if rejection rate exceeds this
            enable_detailed_logging: Enable verbose logging
        """
        self.analyst = analyst
        self.sniper = sniper
        self.pattern_id_map = pattern_id_map
        self.reverse_pattern_map = {v: k for k, v in pattern_id_map.items()}
        
        # Configuration
        self.base_sr_threshold = sr_threshold_pct
        self.base_pattern_confidence = pattern_confidence_min
        self.use_ai_validation = use_ai_validation
        self.enable_adaptive = enable_adaptive_thresholds
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
        
        # Enhanced statistics tracking
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
        
        # NEW: Rejection reason tracking
        self.rejection_reasons = defaultdict(int)
        
        # NEW: Performance metrics per strategy
        self.strategy_stats = defaultdict(lambda: {
            "checks": 0,
            "approved": 0,
            "rejected": 0,
            "avg_signal_quality": 0.0,
        })
        
        # NEW: Historical validation data for analysis
        self.validation_history = deque(maxlen=1000)
        
        # NEW: Threshold adjustment history
        self.threshold_history = deque(maxlen=100)
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log initialization details"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🤖 ENHANCED AI SIGNAL VALIDATOR")
        logger.info("=" * 70)
        logger.info(f"  Status:           {'ENABLED' if self.use_ai_validation else 'DISABLED'}")
        logger.info(f"  Base S/R:         {self.base_sr_threshold:.2%}")
        logger.info(f"  Base Pattern:     {self.base_pattern_confidence:.0%}")
        logger.info(f"  Adaptive:         {'ON' if self.enable_adaptive else 'OFF'}")
        logger.info(f"  Strong Bypass:    {self.strong_signal_bypass:.0%}")
        logger.info(f"  Circuit Breaker:  {self.bypass_threshold:.0%}")
        logger.info(f"  Detailed Logging: {'ON' if self.detailed_logging else 'OFF'}")
        logger.info(f"  Patterns Loaded:  {len(self.pattern_id_map)}")
        logger.info("=" * 70)
        logger.info("")

    def validate_signal(
        self, 
        signal: int,
        signal_details: dict,
        df: pd.DataFrame
    ) -> Tuple[int, dict]:
        """
        Main validation function with comprehensive logging
        
        Returns:
            Tuple[int, dict]: (validated_signal, enriched_details)
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
        # LAYER 0: Strong Signal Bypass
        # ============================================================
        signal_quality = signal_details.get("signal_quality", 0)
        if signal_quality >= self.strong_signal_bypass:
            result = self._bypass_validation(
                signal, signal_details, 
                reason="strong_signal",
                quality=signal_quality
            )
            self.stats["bypassed_strong_signal"] += 1
            self.strategy_stats[strategy]["approved"] += 1
            
            if self.detailed_logging:
                logger.info(f"  ✅ BYPASS: Strong signal quality ({signal_quality:.2%})")
            
            return result
        
        # ============================================================
        # LAYER 1: Circuit Breaker Check
        # ============================================================
        if self.bypass_mode:
            self.bypass_cooldown -= 1
            if self.bypass_cooldown <= 0:
                self._reset_circuit_breaker()
            else:
                result = self._bypass_validation(
                    signal, signal_details,
                    reason="circuit_breaker",
                    cooldown=self.bypass_cooldown
                )
                self.stats["bypassed_circuit_breaker"] += 1
                self.strategy_stats[strategy]["approved"] += 1
                
                if self.detailed_logging:
                    logger.info(f"  ⚡ BYPASS: Circuit breaker active (cooldown: {self.bypass_cooldown})")
                
                return result
        
        # ============================================================
        # LAYER 2: Adaptive Threshold Adjustment
        # ============================================================
        if self.enable_adaptive:
            self._update_adaptive_thresholds(df, signal_details)
        
        if self.detailed_logging:
            logger.info(f"  Thresholds:")
            logger.info(f"    S/R:      {self.current_sr_threshold:.2%} (base: {self.base_sr_threshold:.2%})")
            logger.info(f"    Pattern:  {self.current_pattern_threshold:.0%} (base: {self.base_pattern_confidence:.0%})")
        
        # ============================================================
        # LAYER 3: Support/Resistance Check
        # ============================================================
        current_price = float(df['close'].iloc[-1])
        sr_result = self._check_support_resistance(
            df, current_price, signal, 
            threshold=self.current_sr_threshold
        )
        
        if self.detailed_logging:
            self._log_sr_check(sr_result)
        
        if not sr_result["near_level"]:
            result = self._reject_signal(
                signal_details, sr_result, None,
                reason="no_sr_level",
                strategy=strategy
            )
            self.stats["rejected_no_sr"] += 1
            self.rejection_reasons["no_sr_level"] += 1
            self.strategy_stats[strategy]["rejected"] += 1
            
            return result
        
        # ============================================================
        # LAYER 4: Pattern Confirmation Check
        # ============================================================
        pattern_result = self._check_pattern(
            df, signal, 
            min_confidence=self.current_pattern_threshold
        )
        
        if self.detailed_logging:
            self._log_pattern_check(pattern_result)
        
        if not pattern_result["pattern_confirmed"]:
            result = self._reject_signal(
                signal_details, sr_result, pattern_result,
                reason=pattern_result["reason"],
                strategy=strategy
            )
            self.stats["rejected_no_pattern"] += 1
            self.rejection_reasons[pattern_result["reason"]] += 1
            self.strategy_stats[strategy]["rejected"] += 1
            
            return result
        
        # ============================================================
        # VALIDATION PASSED!
        # ============================================================
        result = self._approve_signal(
            signal, signal_details, sr_result, pattern_result,
            strategy=strategy,
            validation_time=(datetime.now() - validation_start).total_seconds()
        )
        
        self.stats["approved"] += 1
        self.strategy_stats[strategy]["approved"] += 1
        self.rejection_window.append(False)
        
        if self.detailed_logging:
            logger.info(f"  ✅ APPROVED: All validation layers passed")
            logger.info(f"  Confidence Boost: +{result[1].get('confidence_boost', 0):.2%}")
        
        return result
    
    def _signal_str(self, signal: int) -> str:
        """Convert signal to readable string"""
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")
    
    def _skip_validation(self, signal: int, details: dict, reason: str) -> Tuple[int, dict]:
        """Skip validation and return original signal"""
        return signal, {
            **details,
            "ai_validation": f"skipped_{reason}",
            "final_signal": signal
        }
    
    def _bypass_validation(
        self, signal: int, details: dict, 
        reason: str, **kwargs
    ) -> Tuple[int, dict]:
        """Bypass validation and return approved signal"""
        return signal, {
            **details,
            "ai_validation": f"bypassed_{reason}",
            "ai_bypass_reason": ", ".join(f"{k}={v}" for k, v in kwargs.items()),
            "final_signal": signal
        }
    
    def _reject_signal(
        self, signal_details: dict, sr_result: dict, 
        pattern_result: Optional[dict], reason: str, strategy: str
    ) -> Tuple[int, dict]:
        """Reject signal and return HOLD"""
        self.rejection_window.append(True)
        self._check_circuit_breaker()
        
        # Record validation history
        self.validation_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy,
            "original_signal": signal_details.get("signal", 0),
            "result": "rejected",
            "reason": reason,
            "sr_distance": sr_result.get("distance_pct"),
            "pattern": pattern_result.get("pattern_name") if pattern_result else None,
            "confidence": pattern_result.get("confidence") if pattern_result else None,
        })
        
        result_details = {
            **signal_details,
            "ai_validation": "rejected",
            "ai_rejection_reason": reason,
            "ai_sr_check": sr_result,
            "ai_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.current_pattern_threshold:.0%}"
            },
            "final_signal": 0
        }
        
        if pattern_result:
            result_details["ai_pattern_check"] = pattern_result
        
        if self.detailed_logging:
            logger.warning(f"  ❌ REJECTED: {reason}")
            if sr_result.get("distance_pct"):
                logger.warning(f"     S/R distance: {sr_result['distance_pct']:.2f}%")
            if pattern_result:
                logger.warning(f"     Pattern: {pattern_result.get('pattern_name', 'N/A')}")
                logger.warning(f"     Confidence: {pattern_result.get('confidence', 0):.0%}")
        
        return 0, result_details
    
    def _approve_signal(
        self, signal: int, signal_details: dict,
        sr_result: dict, pattern_result: dict,
        strategy: str, validation_time: float
    ) -> Tuple[int, dict]:
        """Approve signal and add confidence boost"""
        self.rejection_window.append(False)
        
        # Calculate confidence boost based on pattern strength
        pattern_conf = pattern_result.get("confidence", 0)
        base_boost = 0.10
        
        if pattern_conf > 0.80:
            boost = base_boost + 0.05  # Strong pattern
        elif pattern_conf > 0.65:
            boost = base_boost
        else:
            boost = base_boost - 0.02  # Weaker pattern
        
        # Record validation history
        self.validation_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy,
            "original_signal": signal,
            "result": "approved",
            "reason": "all_layers_passed",
            "sr_distance": sr_result.get("distance_pct"),
            "pattern": pattern_result.get("pattern_name"),
            "confidence": pattern_result.get("confidence"),
            "validation_time_ms": validation_time * 1000,
        })
        
        return signal, {
            **signal_details,
            "ai_validation": "approved_all_layers",
            "ai_sr_check": sr_result,
            "ai_pattern_check": pattern_result,
            "ai_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.current_pattern_threshold:.0%}"
            },
            "ai_validation_time_ms": validation_time * 1000,
            "final_signal": signal,
            "confidence_boost": boost
        }
    
    def _log_sr_check(self, result: dict):
        """Log S/R check results"""
        near = result.get("near_level", False)
        level_type = result.get("level_type", "N/A")
        nearest = result.get("nearest_level")
        distance = result.get("distance_pct")
        
        if near and nearest is not None:
            logger.info(f"  S/R Check: ✅ Near {level_type} at ${nearest:.2f} (dist: {distance:.2f}%)")
        else:
            if distance is not None:
                logger.info(f"  S/R Check: ❌ Not near {level_type} (dist: {distance:.2f}%)")
            else:
                logger.info(f"  S/R Check: ❌ No {level_type} levels found")
    
    def _log_pattern_check(self, result: dict):
        """Log pattern check results"""
        confirmed = result.get("pattern_confirmed", False)
        pattern = result.get("pattern_name", "N/A")
        confidence = result.get("confidence", 0)
        reason = result.get("reason", "N/A")
        
        if confirmed:
            direction = result.get("direction", "N/A")
            logger.info(f"  Pattern: ✅ {pattern} ({direction}) - {confidence:.0%}")
        else:
            logger.info(f"  Pattern: ❌ {pattern} - {reason}")
    
    def _update_adaptive_thresholds(self, df: pd.DataFrame, signal_details: dict):
        """
        Dynamically adjust validation thresholds with detailed logging
        """
        regime = signal_details.get("regime", "BEAR")
        regime_confidence = signal_details.get("regime_confidence", 0.5)
        
        # Calculate volatility
        if len(df) >= 20:
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.20
        
        # Store old thresholds for comparison
        old_sr = self.current_sr_threshold
        old_pattern = self.current_pattern_threshold
        
        # Adjust S/R threshold based on volatility
        volatility_factor = np.clip(volatility / 0.30, 0.5, 2.0)
        self.current_sr_threshold = self.base_sr_threshold * volatility_factor
        
        # Adjust pattern confidence based on regime strength
        regime_factor = 0.7 + (regime_confidence * 0.3)
        self.current_pattern_threshold = self.base_pattern_confidence * regime_factor
        
        # Relax thresholds if recent rejection rate is high
        if len(self.rejection_window) >= 20:
            rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
            if rejection_rate > 0.60:
                self.current_sr_threshold *= 1.25
                self.current_pattern_threshold *= 0.85
                self.stats["adaptive_adjustments"] += 1
                
                if self.detailed_logging:
                    logger.debug(f"  [ADAPTIVE] Relaxed thresholds (rejection={rejection_rate:.0%})")
        
        # Safety bounds
        self.current_sr_threshold = np.clip(self.current_sr_threshold, 0.005, 0.030)
        self.current_pattern_threshold = np.clip(self.current_pattern_threshold, 0.40, 0.80)
        
        # Log significant changes
        sr_change = abs(self.current_sr_threshold - old_sr) / old_sr
        pattern_change = abs(self.current_pattern_threshold - old_pattern) / old_pattern
        
        if sr_change > 0.10 or pattern_change > 0.10:
            self.threshold_history.append({
                "timestamp": datetime.now(),
                "sr_threshold": self.current_sr_threshold,
                "pattern_threshold": self.current_pattern_threshold,
                "volatility": volatility,
                "regime_confidence": regime_confidence,
            })
            
            if self.detailed_logging:
                logger.debug(f"  [ADAPTIVE] Threshold adjustment:")
                logger.debug(f"    S/R: {old_sr:.3%} → {self.current_sr_threshold:.3%} ({sr_change:+.1%})")
                logger.debug(f"    Pattern: {old_pattern:.1%} → {self.current_pattern_threshold:.1%} ({pattern_change:+.1%})")
    
    def _check_circuit_breaker(self):
        """Activate bypass mode if rejection rate too high"""
        if len(self.rejection_window) < 30:
            return
        
        rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
        
        if rejection_rate > self.bypass_threshold and not self.bypass_mode:
            self.bypass_mode = True
            self.bypass_cooldown = 15
            
            logger.warning("")
            logger.warning("=" * 70)
            logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
            logger.warning(f"   Rejection rate: {rejection_rate:.0%} (threshold: {self.bypass_threshold:.0%})")
            logger.warning(f"   AI validation DISABLED for next {self.bypass_cooldown} signals")
            logger.warning("   Top rejection reasons:")
            
            # Show top 3 rejection reasons
            top_reasons = sorted(
                self.rejection_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for reason, count in top_reasons:
                logger.warning(f"     - {reason}: {count} times")
            
            logger.warning("=" * 70)
            logger.warning("")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker and clear tracking"""
        self.bypass_mode = False
        self.rejection_window.clear()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔄 AI CIRCUIT BREAKER RESET")
        logger.info("   Validation RE-ENABLED")
        logger.info("=" * 70)
        logger.info("")
    
    def _check_support_resistance(
        self, 
        df: pd.DataFrame,
        current_price: float,
        signal: int,
        threshold: float
    ) -> dict:
        """Check if price near S/R level with detailed tracking"""
        # Update S/R levels if cache stale
        now = pd.Timestamp.now()
        if (self.last_sr_update is None or 
            (now - self.last_sr_update).total_seconds() > self.sr_update_interval):
            self._update_sr_levels(df)
            self.last_sr_update = now
        
        # Determine relevant levels
        if signal == 1:  # BUY - check support
            relevant_levels = [l for l in self.sr_cache.get('levels', []) if l < current_price]
            level_type = "support"
        else:  # SELL - check resistance
            relevant_levels = [l for l in self.sr_cache.get('levels', []) if l > current_price]
            level_type = "resistance"
        
        # Find nearest level
        nearest_level = None
        min_distance = float('inf')
        
        for level in relevant_levels:
            distance = abs(current_price - level) / current_price
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        
        near_level = (nearest_level is not None and min_distance < threshold)
        
        return {
            "near_level": near_level,
            "level_type": level_type,
            "nearest_level": nearest_level,
            "distance_pct": min_distance * 100 if nearest_level is not None else None,
            "threshold_used": threshold,
            "all_levels": relevant_levels[:3],
            "total_levels_found": len(relevant_levels)
        }
    
    def _update_sr_levels(self, df: pd.DataFrame):
        """Recalculate S/R levels"""
        pivots = self._extract_pivots(df, window=7)
        
        if len(pivots) < 5:
            logger.debug(f"[SR UPDATE] Only {len(pivots)} pivots found")
            self.sr_cache = {'levels': []}
            return
        
        try:
            levels = self.analyst.get_support_resistance_levels(
                pivot_points=pivots,
                highs=df['high'].values,
                lows=df['low'].values,
                closes=df['close'].values,
                n_levels=7
            )
            
            self.sr_cache = {
                'levels': levels,
                'updated_at': datetime.now(),
                'pivot_count': len(pivots)
            }
            
            logger.debug(f"[SR UPDATE] {len(levels)} levels from {len(pivots)} pivots")
            
        except Exception as e:
            logger.error(f"[SR UPDATE] Failed: {e}")
            self.sr_cache = {'levels': []}
    
    def _extract_pivots(self, df: pd.DataFrame, window=7) -> np.ndarray:
        """Extract pivot points"""
        highs = df['high'].values
        lows = df['low'].values
        pivots = []
        
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                pivots.append(highs[i])
            if lows[i] == min(lows[i-window:i+window+1]):
                pivots.append(lows[i])
        
        return np.array(pivots) if pivots else np.array([])
    
    def _check_pattern(
        self, 
        df: pd.DataFrame, 
        signal: int,
        min_confidence: float
    ) -> dict:
        """Check for confirming pattern"""
        if len(df) < 15:
            return {
                "pattern_confirmed": False,
                "reason": "insufficient_data",
                "pattern_name": None,
                "confidence": 0.0
            }
        
        try:
            last_15 = df.tail(15)[['open', 'high', 'low', 'close']].values
            pattern_id, confidence = self.sniper.predict(last_15)
            
            if pattern_id == 0:
                return {
                    "pattern_confirmed": False,
                    "reason": "no_pattern_detected",
                    "pattern_name": "Noise",
                    "confidence": confidence
                }
            
            pattern_name = self.reverse_pattern_map.get(pattern_id, 'Unknown')
            is_bullish = pattern_name in self.BULLISH_PATTERNS
            is_bearish = pattern_name in self.BEARISH_PATTERNS
            
            # Direction mismatch
            if signal == 1 and not is_bullish:
                self.stats["rejected_direction_mismatch"] += 1
                return {
                    "pattern_confirmed": False,
                    "reason": f"bearish_pattern_for_buy_signal",
                    "pattern_name": pattern_name,
                    "confidence": confidence
                }
            
            if signal == -1 and not is_bearish:
                self.stats["rejected_direction_mismatch"] += 1
                return {
                    "pattern_confirmed": False,
                    "reason": f"bullish_pattern_for_sell_signal",
                    "pattern_name": pattern_name,
                    "confidence": confidence
                }
            
            # Confidence check
            if confidence < min_confidence:
                self.stats["rejected_low_confidence"] += 1
                return {
                    "pattern_confirmed": False,
                    "reason": f"low_confidence_{confidence:.0%}_threshold_{min_confidence:.0%}",
                    "pattern_name": pattern_name,
                    "confidence": confidence
                }
            
            # APPROVED
            return {
                "pattern_confirmed": True,
                "pattern_name": pattern_name,
                "confidence": confidence,
                "direction": "bullish" if is_bullish else "bearish",
                "threshold_used": min_confidence
            }
            
        except Exception as e:
            logger.error(f"[PATTERN CHECK] Error: {e}")
            return {
                "pattern_confirmed": False,
                "reason": f"error_{str(e)[:30]}",
                "pattern_name": None,
                "confidence": 0.0
            }
    
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
                "pattern_confidence": f"{self.current_pattern_threshold:.0%}"
            },
            "circuit_breaker": {
                "active": self.bypass_mode,
                "cooldown": self.bypass_cooldown if self.bypass_mode else 0
            }
        }
        
        # Add top rejection reasons if available
        if hasattr(self, 'rejection_reasons') and self.rejection_reasons:
            top_reasons = sorted(
                self.rejection_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            base_stats["top_rejection_reasons"] = {
                reason: count for reason, count in top_reasons
            }
        
        # Add per-strategy stats if available
        if hasattr(self, 'strategy_stats') and self.strategy_stats:
            base_stats["per_strategy"] = {}
            for strategy, strat_stats in self.strategy_stats.items():
                total_strat = max(strat_stats["checks"], 1)
                base_stats["per_strategy"][strategy] = {
                    "checks": strat_stats["checks"],
                    "approved": strat_stats["approved"],
                    "rejected": strat_stats["rejected"],
                    "approval_rate": f"{(strat_stats['approved']/total_strat)*100:.1f}%"
                }
        
        return base_stats