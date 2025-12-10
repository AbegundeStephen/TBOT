"""
Improved Hybrid Signal Validator with Adaptive Thresholds
- Dynamic S/R tolerance based on volatility
- Pattern confidence scaling by regime
- Better circuit breaker logic
- Validation bypass for strong signals
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger(__name__)


class HybridSignalValidator:
    """
    Validates trading signals using AI with adaptive thresholds
    """
    
    # Pattern classifications
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
        sr_threshold_pct=0.015,  # 1.5% default (was 0.5%)
        pattern_confidence_min=0.50,  # 50% default (was 65-70%)
        use_ai_validation=True,
        enable_adaptive_thresholds=True,  # NEW
        strong_signal_bypass_threshold=0.75,  # NEW: Bypass AI if signal very strong
    ):
        """
        Args:
            analyst: DynamicAnalyst instance
            sniper: OHLCSniper instance
            pattern_id_map: Dict mapping pattern IDs to names
            sr_threshold_pct: Base distance to S/R level (adapts with volatility)
            pattern_confidence_min: Base pattern confidence (adapts with regime)
            use_ai_validation: Toggle AI validation
            enable_adaptive_thresholds: Adjust thresholds based on market conditions
            strong_signal_bypass_threshold: Skip AI for very strong signals
        """
        self.analyst = analyst
        self.sniper = sniper
        self.pattern_id_map = pattern_id_map
        self.reverse_pattern_map = {v: k for k, v in pattern_id_map.items()}
        
        # Base thresholds (will adapt)
        self.base_sr_threshold = sr_threshold_pct
        self.base_pattern_confidence = pattern_confidence_min
        self.use_ai_validation = use_ai_validation
        self.enable_adaptive = enable_adaptive_thresholds
        self.strong_signal_bypass = strong_signal_bypass_threshold
        
        # Adaptive thresholds (updated dynamically)
        self.current_sr_threshold = sr_threshold_pct
        self.current_pattern_threshold = pattern_confidence_min
        
        # S/R cache with shorter update interval
        self.sr_cache = {}
        self.last_sr_update = None
        self.sr_update_interval = 3600  # 1 hour (was 4 hours)
        
        # Circuit breaker with better tracking
        self.rejection_window = deque(maxlen=50)  # Track last 50 decisions
        self.bypass_mode = False
        self.bypass_cooldown = 0
        self.bypass_threshold = 0.70  # Bypass if >70% rejection (was 85%)
        
        # Performance tracking
        self.stats = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "rejected_no_sr": 0,
            "rejected_no_pattern": 0,
            "bypassed_strong_signal": 0,
            "bypassed_circuit_breaker": 0,
            "adaptive_adjustments": 0,
        }
        
        logger.info(f"[VALIDATOR] Enhanced AI validation initialized")
        logger.info(f"  Base S/R threshold: {sr_threshold_pct:.2%}")
        logger.info(f"  Base pattern confidence: {pattern_confidence_min:.0%}")
        logger.info(f"  Adaptive thresholds: {'ENABLED' if enable_adaptive_thresholds else 'DISABLED'}")
        logger.info(f"  Strong signal bypass: {strong_signal_bypass_threshold:.0%}")
        logger.info(f"  Circuit breaker threshold: {self.bypass_threshold:.0%}")

    def validate_signal(
        self, 
        signal: int,
        signal_details: dict,
        df: pd.DataFrame
    ) -> Tuple[int, dict]:
        """
        Main validation function with adaptive logic
        """
        self.stats["total_checks"] += 1
        
        # Skip validation if disabled or signal is HOLD
        if not self.use_ai_validation or signal == 0:
            return signal, {
                **signal_details,
                "ai_validation": "skipped" if not self.use_ai_validation else "hold_signal"
            }
        
        # ============================================================
        # BYPASS 1: Strong Signal Override
        # ============================================================
        signal_quality = signal_details.get("signal_quality", 0)
        if signal_quality >= self.strong_signal_bypass:
            self.stats["bypassed_strong_signal"] += 1
            logger.debug(f"[VALIDATOR] BYPASS: Strong signal quality {signal_quality:.0%}")
            return signal, {
                **signal_details,
                "ai_validation": "bypassed_strong_signal",
                "ai_bypass_reason": f"quality={signal_quality:.2f}",
                "final_signal": signal
            }
        
        # ============================================================
        # BYPASS 2: Circuit Breaker Check
        # ============================================================
        if self.bypass_mode:
            self.bypass_cooldown -= 1
            if self.bypass_cooldown <= 0:
                self.bypass_mode = False
                self.rejection_window.clear()
                logger.info("[VALIDATOR] Circuit breaker RESET - validation re-enabled")
            else:
                self.stats["bypassed_circuit_breaker"] += 1
                return signal, {
                    **signal_details,
                    "ai_validation": "bypassed_circuit_breaker",
                    "ai_bypass_reason": f"cooldown={self.bypass_cooldown}",
                    "final_signal": signal
                }
        
        # ============================================================
        # Adaptive Threshold Adjustment
        # ============================================================
        if self.enable_adaptive:
            self._update_adaptive_thresholds(df, signal_details)
        
        # ============================================================
        # Layer 1: Support/Resistance Check
        # ============================================================
        current_price = float(df['close'].iloc[-1])
        sr_result = self._check_support_resistance(
            df, current_price, signal, 
            threshold=self.current_sr_threshold
        )
        
        if not sr_result["near_level"]:
            self.rejection_window.append(True)  # Track rejection
            self._check_circuit_breaker()
            
            self.stats["rejected"] += 1
            self.stats["rejected_no_sr"] += 1
            
            distance = sr_result.get('distance_pct')
            if distance is not None:
                logger.debug(
                    f"[VALIDATOR] REJECTED: Not near S/R "
                    f"(dist: {distance:.2f}%, threshold: {self.current_sr_threshold:.2f}%)"
                )
            
            return 0, {
                **signal_details,
                "ai_validation": "rejected_no_sr_level",
                "ai_sr_check": sr_result,
                "ai_thresholds": {
                    "sr_threshold": f"{self.current_sr_threshold:.2%}",
                    "pattern_threshold": f"{self.current_pattern_threshold:.0%}"
                },
                "final_signal": 0
            }
        
        # ============================================================
        # Layer 2: Pattern Confirmation Check
        # ============================================================
        pattern_result = self._check_pattern(
            df, signal, 
            min_confidence=self.current_pattern_threshold
        )
        
        if not pattern_result["pattern_confirmed"]:
            self.rejection_window.append(True)  # Track rejection
            self._check_circuit_breaker()
            
            self.stats["rejected"] += 1
            self.stats["rejected_no_pattern"] += 1
            
            logger.debug(f"[VALIDATOR] REJECTED: {pattern_result['reason']}")
            
            return 0, {
                **signal_details,
                "ai_validation": "rejected_no_pattern",
                "ai_sr_check": sr_result,
                "ai_pattern_check": pattern_result,
                "ai_thresholds": {
                    "sr_threshold": f"{self.current_sr_threshold:.2%}",
                    "pattern_threshold": f"{self.current_pattern_threshold:.0%}"
                },
                "final_signal": 0
            }
        
        # ============================================================
        # ALL VALIDATIONS PASSED!
        # ============================================================
        self.rejection_window.append(False)  # Track approval
        self.stats["approved"] += 1
        
        logger.info(
            f"[VALIDATOR] ✓ APPROVED: {signal} at {sr_result['level_type']} "
            f"with {pattern_result['pattern_name']} ({pattern_result['confidence']:.0%})"
        )
        
        return signal, {
            **signal_details,
            "ai_validation": "approved_all_layers",
            "ai_sr_check": sr_result,
            "ai_pattern_check": pattern_result,
            "ai_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_threshold": f"{self.current_pattern_threshold:.0%}"
            },
            "final_signal": signal,
            "confidence_boost": 0.10  # Moderate boost (was 0.15)
        }

    def _update_adaptive_thresholds(self, df: pd.DataFrame, signal_details: dict):
        """
        Dynamically adjust validation thresholds based on:
        1. Market volatility (ATR)
        2. Regime strength
        3. Recent rejection rate
        """
        regime = signal_details.get("regime", "BEAR")
        regime_confidence = signal_details.get("regime_confidence", 0.5)
        
        # Calculate volatility
        if len(df) >= 20:
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0.20  # Default 20%
        
        # Adjust S/R threshold based on volatility
        # Higher volatility = wider tolerance
        volatility_factor = np.clip(volatility / 0.30, 0.5, 2.0)  # 0.5x to 2.0x
        self.current_sr_threshold = self.base_sr_threshold * volatility_factor
        
        # Adjust pattern confidence based on regime strength
        # Weak regime = lower confidence requirement
        regime_factor = 0.7 + (regime_confidence * 0.3)  # 0.7x to 1.0x
        self.current_pattern_threshold = self.base_pattern_confidence * regime_factor
        
        # Relax thresholds if recent rejection rate is high
        if len(self.rejection_window) >= 20:
            rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
            if rejection_rate > 0.60:  # >60% rejection
                self.current_sr_threshold *= 1.25
                self.current_pattern_threshold *= 0.85
                self.stats["adaptive_adjustments"] += 1
                logger.debug(
                    f"[VALIDATOR] RELAXED thresholds: "
                    f"rejection_rate={rejection_rate:.0%}"
                )
        
        # Safety bounds
        self.current_sr_threshold = np.clip(
            self.current_sr_threshold, 
            0.005,  # Min 0.5%
            0.030   # Max 3.0%
        )
        self.current_pattern_threshold = np.clip(
            self.current_pattern_threshold,
            0.40,   # Min 40%
            0.80    # Max 80%
        )

    def _check_circuit_breaker(self):
        """
        Activate bypass mode if AI is rejecting too many signals
        """
        if len(self.rejection_window) < 30:  # Need 30 samples
            return
        
        rejection_rate = sum(self.rejection_window) / len(self.rejection_window)
        
        if rejection_rate > self.bypass_threshold and not self.bypass_mode:
            self.bypass_mode = True
            self.bypass_cooldown = 15  # Bypass next 15 signals (was 10)
            
            logger.warning("")
            logger.warning("=" * 70)
            logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
            logger.warning(f"   Rejection rate: {rejection_rate:.0%} (threshold: {self.bypass_threshold:.0%})")
            logger.warning(f"   AI validation DISABLED for next {self.bypass_cooldown} signals")
            logger.warning("=" * 70)
            logger.warning("")

    def _check_support_resistance(
        self, 
        df: pd.DataFrame,
        current_price: float,
        signal: int,
        threshold: float
    ) -> dict:
        """Check if price is near S/R level with dynamic threshold"""
        # Update S/R levels if cache is stale
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
            "all_levels": relevant_levels[:3]
        }

    def _update_sr_levels(self, df: pd.DataFrame):
        """Recalculate S/R levels with better pivot detection"""
        pivots = self._extract_pivots(df, window=7)  # Wider window (was 5)
        
        if len(pivots) < 5:
            logger.warning(f"[VALIDATOR] Only {len(pivots)} pivots found")
            self.sr_cache = {'levels': []}
            return
        
        levels = self.analyst.get_support_resistance_levels(
            pivot_points=pivots,
            highs=df['high'].values,
            lows=df['low'].values,
            closes=df['close'].values,
            n_levels=7  # More levels (was 5)
        )
        
        self.sr_cache = {'levels': levels}
        logger.debug(f"[VALIDATOR] S/R levels updated: {len(levels)} levels")

    def _extract_pivots(self, df: pd.DataFrame, window=7) -> np.ndarray:
        """Extract pivot points with configurable window"""
        highs = df['high'].values
        lows = df['low'].values
        pivots = []
        
        for i in range(window, len(df) - window):
            # Local high
            if highs[i] == max(highs[i-window:i+window+1]):
                pivots.append(highs[i])
            # Local low
            if lows[i] == min(lows[i-window:i+window+1]):
                pivots.append(lows[i])
        
        return np.array(pivots) if pivots else np.array([])

    def _check_pattern(
        self, 
        df: pd.DataFrame, 
        signal: int,
        min_confidence: float
    ) -> dict:
        """Check for confirming pattern with dynamic confidence"""
        if len(df) < 15:
            return {
                "pattern_confirmed": False,
                "reason": "insufficient_data",
                "pattern_name": None,
                "confidence": 0.0
            }
        
        last_15 = df.tail(15)[['open', 'high', 'low', 'close']].values
        pattern_id, confidence = self.sniper.predict(last_15)
        
        if pattern_id == 0:  # Noise
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
            return {
                "pattern_confirmed": False,
                "reason": f"bearish_pattern_for_buy_signal",
                "pattern_name": pattern_name,
                "confidence": confidence
            }
        
        if signal == -1 and not is_bearish:
            return {
                "pattern_confirmed": False,
                "reason": f"bullish_pattern_for_sell_signal",
                "pattern_name": pattern_name,
                "confidence": confidence
            }
        
        # Confidence check (now using dynamic threshold)
        if confidence < min_confidence:
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

    def get_statistics(self) -> dict:
        """Return detailed validation statistics"""
        total = max(self.stats["total_checks"], 1)
        
        return {
            "ai_enabled": self.use_ai_validation,
            "total_checks": self.stats["total_checks"],
            "approved": self.stats["approved"],
            "rejected": self.stats["rejected"],
            "approval_rate": f"{(self.stats['approved']/total)*100:.1f}%",
            "rejection_rate": f"{(self.stats['rejected']/total)*100:.1f}%",
            "rejected_no_sr": self.stats["rejected_no_sr"],
            "rejected_no_pattern": self.stats["rejected_no_pattern"],
            "bypassed_strong_signal": self.stats["bypassed_strong_signal"],
            "bypassed_circuit_breaker": self.stats["bypassed_circuit_breaker"],
            "adaptive_adjustments": self.stats["adaptive_adjustments"],
            "current_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_confidence": f"{self.current_pattern_threshold:.0%}"
            },
            "circuit_breaker_active": self.bypass_mode,
            "bypass_cooldown": self.bypass_cooldown if self.bypass_mode else 0
        }