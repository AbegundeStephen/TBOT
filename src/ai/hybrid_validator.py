"""
 AI Signal Validator with Realistic S/R Thresholds
========================================================
Key fixes:
1. Base S/R threshold: 0.5% → 2.5% (5x more realistic)
2. Directional S/R logic (BUY needs support, SELL needs resistance)
3. Strategy-aware adjustments (TF gets wider thresholds)
4. Better adaptive scaling based on volatility and regime
5. Comprehensive logging preserved
6. Weekly/Monthly AVWAP (Phase 3)
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
        "Marubozu",
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
        sr_threshold_pct=0.0035,
        pattern_confidence_min=0.65,
        use_ai_validation=True,
        enable_adaptive_thresholds=True,
        strong_signal_bypass_threshold=0.85,
        circuit_breaker_threshold=0.70,
        enable_detailed_logging=False,
    ):
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
        self.sr_cache: Dict[str, Dict] = {}
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
        self, signal: int, signal_details: dict, df: pd.DataFrame
    ) -> Tuple[int, dict]:
        validation_start = datetime.now()
        self.stats["total_checks"] += 1

        asset = signal_details.get("asset", "UNKNOWN")
        strategy = signal_details.get("strategy", "UNKNOWN")

        self.strategy_stats[strategy]["checks"] += 1

        if not self.use_ai_validation:
            return self._skip_validation(signal, signal_details, "ai_disabled")

        if signal == 0:
            return self._skip_validation(signal, signal_details, "hold_signal")

        # Layer 1: Circuit Breaker
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
                return result

        # Layer 2: Adaptive Thresholds
        if self.enable_adaptive:
            self._update_adaptive_thresholds_fixed(df, signal_details, strategy)

        # Layer 3: Support/Resistance Check
        current_price = float(df["close"].iloc[-1])
        sr_result = self._check_support_resistance_fixed(
            asset, df, current_price, signal, threshold=self.current_sr_threshold
        )

        if not sr_result["near_level"]:
            result = self._reject_signal(
                signal_details, sr_result, None, reason="no_sr_level", strategy=strategy
            )
            self.stats["rejected_no_sr"] += 1
            self.rejection_reasons["no_sr_level"] += 1
            self.strategy_stats[strategy]["rejected"] += 1
            return result

        # Layer 4: Pattern Confirmation
        pattern_result = self._check_pattern(
            df, signal, min_confidence=self.current_pattern_threshold, strategy=strategy
        )

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

        # Approval
        result = self._approve_signal(
            signal,
            signal_details,
            sr_result,
            pattern_result,
            strategy=strategy,
            validation_time=(datetime.now() - validation_start).total_seconds(),
            df=df,
        )

        self.stats["approved"] += 1
        self.strategy_stats[strategy]["approved"] += 1
        self.rejection_window.append(False)

        return result

    def _update_adaptive_thresholds_fixed(
        self, df: pd.DataFrame, signal_details: dict, strategy: str
    ):
        regime = signal_details.get("regime", "BEAR")
        signal_quality = signal_details.get("signal_quality", 0.0)

        if len(df) >= 20:
            returns = df["close"].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.20

        sr_threshold = self.base_sr_threshold
        if strategy == "mean_reversion": sr_threshold *= 1.0
        elif strategy == "trend_following": sr_threshold *= 1.5
        else: sr_threshold *= 1.2

        if volatility > 0.40: sr_threshold *= 1.3
        elif volatility > 0.30: sr_threshold *= 1.15
        elif volatility < 0.15: sr_threshold *= 0.9

        self.current_sr_threshold = np.clip(sr_threshold, 0.015, 0.060)

        pattern_threshold = self.base_pattern_confidence
        if "BULL" in regime.upper(): pattern_threshold *= 0.90
        else: pattern_threshold *= 0.95

        self.current_pattern_threshold = np.clip(pattern_threshold, 0.40, 0.75)

    def _check_support_resistance_fixed(
        self, asset: str, df: pd.DataFrame, current_price: float, signal: int, threshold: float
    ) -> dict:
        now = datetime.now()
        asset_cache = self.sr_cache.get(asset, {})
        last_update = asset_cache.get('updated_at')

        if not last_update or (now - last_update).total_seconds() > self.sr_update_interval:
            self._update_sr_levels(asset, df)
            asset_cache = self.sr_cache.get(asset, {})

        all_levels = asset_cache.get("levels", [])

        if not all_levels:
            return {"near_level": False, "reason": "no_sr_levels_found", "all_levels": []}

        if signal == 1:
            relevant_levels = [l for l in all_levels if l < current_price]
            level_type = "support"
        else:
            relevant_levels = [l for l in all_levels if l > current_price]
            level_type = "resistance"

        if not relevant_levels:
            ema_20_series = df["close"].ewm(span=20, adjust=False).mean()
            current_ema = ema_20_series.iloc[-1]
            prev_ema = ema_20_series.iloc[-2]

            if signal == 1 and current_price > current_ema and current_ema > prev_ema:
                return {
                    "near_level": True,
                    "level_type": "dynamic_ema_support",
                    "nearest_level": current_ema,
                    "distance_pct": ((current_price - current_ema) / current_price) * 100,
                    "reason": "riding_dynamic_20_ema"
                }

            elif signal == -1 and current_price < current_ema and current_ema < prev_ema:
                return {
                    "near_level": True,
                    "level_type": "dynamic_ema_resistance",
                    "nearest_level": current_ema,
                    "distance_pct": ((current_ema - current_price) / current_price) * 100,
                    "reason": "riding_dynamic_20_ema"
                }

            # FALLBACK: Check if at level (boundary detection)
            any_level_distances = [abs(current_price - l) / current_price for l in all_levels]
            min_any_dist = min(any_level_distances) if any_level_distances else float("inf")
            if min_any_dist < threshold:
                closest = all_levels[np.argmin(any_level_distances)]
                return {"near_level": True, "level_type": "boundary", "nearest_level": closest, "distance_pct": min_any_dist * 100, "reason": f"at_level_${closest:.2f}"}
            return {"near_level": False, "reason": f"no_{level_type}_below/above"}

        distances = [(abs(current_price - level) / current_price, level) for level in relevant_levels]
        min_distance_pct, nearest_level = min(distances)
        near_level = min_distance_pct < threshold

        return {
            "near_level": near_level,
            "level_type": level_type,
            "nearest_level": nearest_level,
            "distance_pct": min_distance_pct * 100,
            "all_levels": all_levels,
            "reason": f"near_{level_type}_${nearest_level:.2f}" if near_level else f"{level_type}_too_far"
        }

    def _update_sr_levels(self, asset: str, df: pd.DataFrame):
        """More robust S/R level extraction with Anchored VWAP (Phase 3)."""
        pivots = self._extract_pivots(df, window=7)
        avwap_levels = self._calculate_anchored_vwaps(df)

        if len(pivots) < 3:
            closes = df["close"].values
            levels = np.percentile(closes, [10, 25, 50, 75, 90]).tolist()
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": True}
            return

        try:
            levels = self.analyst.get_support_resistance_levels(
                pivot_points=pivots,
                highs=df["high"].values,
                lows=df["low"].values,
                closes=df["close"].values,
                n_levels=7,
            )
            if not levels: levels = sorted(np.unique(pivots).tolist())
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": False}
        except Exception as e:
            logger.error(f"[SR UPDATE] Failed for {asset}: {e}")
            closes = df["close"].values
            levels = np.percentile(closes, [10, 30, 50, 70, 90]).tolist()
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": True}

    def _calculate_anchored_vwaps(self, df: pd.DataFrame) -> Dict[str, float]:
        try:
            temp_df = df.copy()
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                if 'timestamp' in temp_df.columns:
                    temp_df.index = pd.to_datetime(temp_df['timestamp'])
                else: return {}
            
            last_date = temp_df.index[-1]
            week_start = last_date - timedelta(days=last_date.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            weekly_data = temp_df[temp_df.index >= week_start]
            w_vwap = (weekly_data['close'] * weekly_data['volume']).sum() / weekly_data['volume'].sum() if not weekly_data.empty and weekly_data['volume'].sum() > 0 else temp_df['close'].iloc[-1]

            month_start = last_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly_data = temp_df[temp_df.index >= month_start]
            m_vwap = (monthly_data['close'] * monthly_data['volume']).sum() / monthly_data['volume'].sum() if not monthly_data.empty and monthly_data['volume'].sum() > 0 else temp_df['close'].iloc[-1]

            return {"weekly_avwap": w_vwap, "monthly_avwap": m_vwap}
        except Exception as e:
            logger.error(f"[AVWAP] Error: {e}")
            return {}

    def _check_pattern(self, df: pd.DataFrame, signal: int, min_confidence: float = 0.60, strategy: str = "UNKNOWN") -> dict:
        try:
            if len(df) < 15: return {"pattern_confirmed": False, "reason": "insufficient_data"}
            
            # ================================================================
            # FLASH CRASH BREAKER: Extreme Volatility Circuit Breaker
            # ================================================================
            current_price = df["close"].iloc[-1]
            ema_20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            price_deviation = abs(current_price - ema_20)
            
            # Calculate ATR for scaling
            import talib as ta
            atr_fast = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            
            if price_deviation > (4.0 * atr_fast):
                if strategy == "mean_reversion":
                    logger.info(f"[AI] Flash Crash Breaker triggered (Deviation: {price_deviation:.2f} > 4xATR: {4.0*atr_fast:.2f}). Blocking MR trade.")
                    return {
                        "pattern_confirmed": False,
                        "confidence": 0.0,
                        "reason": "flash_crash_breaker"
                    }

            snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
            if snippet[0, 0] <= 0: return {"pattern_confirmed": False, "reason": "invalid_data"}
            snippet_input = (snippet / snippet[0, 0] - 1).reshape(1, 15, 4)
            predicted_id, confidence = self.sniper.predict_single(snippet_input)
            pattern_name = self.reverse_pattern_map.get(predicted_id, "Unknown")
            
            # --- Noise Filter ---
            if "noise" in pattern_name.lower():
                return {
                    "pattern_confirmed": False,
                    "confidence": 0,
                    "reason": "noise_detected"
                }

            if predicted_id == 0:
                return {"pattern_confirmed": confidence < 0.70, "reason": "no_pattern_detected", "pattern_name": "Noise", "confidence": confidence}

            # Alignment check
            is_bullish = pattern_name in self.BULLISH_PATTERNS
            is_bearish = pattern_name in self.BEARISH_PATTERNS
            
            if signal == 1 and not is_bullish: return {"pattern_confirmed": False, "reason": "direction_mismatch"}
            if signal == -1 and not is_bearish: return {"pattern_confirmed": False, "reason": "direction_mismatch"}

            # ================================================================
            # MR PATTERN CONFIRMATION: Strict Institutional Entry Rules
            # ================================================================
            if strategy == "mean_reversion":
                allowed_long = ["hammer", "morning star", "bullish engulfing"]
                allowed_short = ["shooting star", "evening star", "bearish engulfing"]
                
                pattern_lower = pattern_name.lower()
                
                if signal == 1 and pattern_lower not in allowed_long:
                    logger.info(f"[AI] MR Blocked: Pattern '{pattern_name}' is not in allowed institutional list for LONG.")
                    return {
                        "pattern_confirmed": False, 
                        "confidence": 0.0,
                        "reason": f"unsupported_mr_pattern_{pattern_lower}"
                    }
                
                if signal == -1 and pattern_lower not in allowed_short:
                    logger.info(f"[AI] MR Blocked: Pattern '{pattern_name}' is not in allowed institutional list for SHORT.")
                    return {
                        "pattern_confirmed": False, 
                        "confidence": 0.0,
                        "reason": f"unsupported_mr_pattern_{pattern_lower}"
                    }
            
            # --- Volume Weighting ---
            if 'volume' in df.columns and len(df) > 20:
                avg_vol = df['volume'].iloc[-21:-1].mean()
                volume = df['volume'].iloc[-1]
                if volume > (2.0 * avg_vol):
                    min_confidence = max(0.45, min_confidence - 0.20)

            if confidence < min_confidence:
                return {
                    "pattern_confirmed": False,
                    "reason": "low_confidence",
                    "confidence": confidence
                }
            
            return {"pattern_confirmed": True, "pattern_name": pattern_name, "confidence": confidence}
        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return {"pattern_confirmed": False, "reason": "error"}

    def _approve_signal(self, signal: int, signal_details: dict, sr_result: dict, pattern_result: dict, strategy: str, validation_time: float, df: Optional[pd.DataFrame] = None) -> Tuple[int, dict]:
        self.rejection_window.append(False)
        pattern_conf = pattern_result.get("confidence", 0)
        boost = 0.10
        if pattern_conf > 0.80: boost += 0.05
        
        regime = signal_details.get("regime", "NEUTRAL")
        current_price = float(df['close'].iloc[-1]) if df is not None else 0

        # AI Confluence Bonus
        if df is not None and len(df) >= 14:
            import talib as ta
            atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            distance = sr_result.get("distance_pct", 999) / 100 * current_price
            if distance < (0.25 * atr): boost += 0.15

        # AVWAP Floor Bonus
        asset = signal_details.get("asset", "UNKNOWN")
        asset_cache = self.sr_cache.get(asset, {})
        weekly_avwap = asset_cache.get("avwaps", {}).get("weekly_avwap")
        
        if signal == 1 and regime == "SLIGHTLY_BULLISH" and weekly_avwap:
            if abs(current_price - weekly_avwap) / weekly_avwap < 0.003:
                boost += 0.25
                logger.info(f"  🏛️ INSTITUTIONAL FLOOR: +25% boost (Price at Weekly AVWAP)")

        return signal, {**signal_details, "ai_validation": "approved", "ai_sr_check": sr_result, "ai_pattern_check": pattern_result, "confidence_boost": boost}

    def _extract_pivots(self, df: pd.DataFrame, window=7) -> np.ndarray:
        highs, lows = df["high"].values, df["low"].values
        pivots = []
        for w in [window, 5, 3]:
            pivots = []
            for i in range(w, len(df) - w):
                if highs[i] == max(highs[i - w : i + w + 1]): pivots.append(highs[i])
                if lows[i] == min(lows[i - w : i + w + 1]): pivots.append(lows[i])
            if len(pivots) >= 3: break
        return np.array(pivots)

    def _reject_signal(self, details: dict, sr: dict, pattern: Optional[dict], reason: str, strategy: str) -> Tuple[int, dict]:
        self.rejection_window.append(True)
        self._check_circuit_breaker()
        return 0, {
            **details, 
            "ai_validation": "rejected", 
            "ai_rejection_reason": reason, 
            "ai_sr_check": sr,
            "ai_pattern_check": pattern,
            "final_signal": 0
        }

    def _skip_validation(self, signal: int, details: dict, reason: str) -> Tuple[int, dict]:
        return signal, {**details, "ai_validation": f"skipped_{reason}"}

    def _bypass_validation(self, signal: int, details: dict, reason: str, **kwargs) -> Tuple[int, dict]:
        return signal, {**details, "ai_validation": f"bypassed_{reason}"}

    def _check_circuit_breaker(self):
        if len(self.rejection_window) < 30: return
        if sum(self.rejection_window) / len(self.rejection_window) > self.bypass_threshold:
            self.bypass_mode = True
            self.bypass_cooldown = 15

    def _reset_circuit_breaker(self):
        self.bypass_mode = False
        self.rejection_window.clear()

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics for the monitor.
        """
        total = max(self.stats["total_checks"], 1)
        
        # Calculate rates
        approval_rate = (self.stats["approved"] / total) * 100
        rejection_rate = (self.stats["rejected"] / total) * 100
        
        # Sort and get top rejection reasons
        top_reasons = dict(sorted(self.rejection_reasons.items(), key=lambda item: item[1], reverse=True)[:5])
        
        return {
            "total_checks": self.stats["total_checks"],
            "approved": self.stats["approved"],
            "rejected": self.stats["rejected"],
            "approval_rate": f"{approval_rate:.1f}%",
            "rejection_rate": f"{rejection_rate:.1f}%",
            "rejection_breakdown": {
                "no_sr_level": self.stats["rejected_no_sr"],
                "no_pattern": self.stats["rejected_no_pattern"],
                "low_confidence": self.stats["rejected_low_confidence"],
                "direction_mismatch": self.stats["rejected_direction_mismatch"],
            },
            "bypasses": {
                "strong_signal": self.stats["bypassed_strong_signal"],
                "circuit_breaker": self.stats["bypassed_circuit_breaker"],
            },
            "current_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_confidence": f"{self.current_pattern_threshold:.0%}",
            },
            "adaptive_adjustments": self.stats["adaptive_adjustments"],
            "circuit_breaker": {
                "active": self.bypass_mode,
                "cooldown": self.bypass_cooldown,
            },
            "top_rejection_reasons": top_reasons,
            "per_strategy": dict(self.strategy_stats)
        }
