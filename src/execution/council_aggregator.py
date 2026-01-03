"""
Institutional Council Aggregator - Hybrid Approach
Combines your existing multi-strategy system with Gemini's weighted council logic
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class InstitutionalCouncilAggregator:
    """
    "BlackRock-Style" Weighted Council with AI Validation
    
    Council Members (Judges):
    1. TREND (1.5 pts)     - The Boss: EMA alignment
    2. STRUCTURE (1.5 pts) - The Location: S/R + AI pivots
    3. MOMENTUM (1.0 pt)   - The Fuel: RSI + MACD
    4. PATTERN (0.5 pt)    - The Trigger: AI candlestick patterns
    5. VOLUME (0.5 pt)     - The Validator: Volume confirmation
    
    Total: 5.0 points
    Trade Threshold: 3.5 / 5.0 (70%)
    
    Regime Rules:
    - Trend-aligned: Need 3.5+ (simple majority)
    - Counter-trend: Need 4.0+ (unanimous overrule)
    """
    
    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
        ai_validator=None,
        enable_detailed_logging: bool = False,
        
        # Council thresholds
        trend_aligned_threshold: float = 3.5,
        counter_trend_threshold: float = 4.0,
        
        # Judge weights (must sum to 5.0)
        weight_trend: float = 1.5,
        weight_structure: float = 1.5,
        weight_momentum: float = 1.0,
        weight_pattern: float = 0.5,
        weight_volume: float = 0.5,
        
        # Asset-specific tuning
        config: Optional[Dict] = None,
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()
        self.ai_validator = ai_validator
        self.detailed_logging = enable_detailed_logging
        
        # ================================================================
        # CONFIGURATION MERGE FIX
        # ================================================================
        # Start with hardcoded defaults (contains technical keys like 'rsi_bullish_zone')
        self.config = self._get_default_config()
        
        # If a preset config is provided, UPDATE defaults instead of replacing them
        if config:
            self.config.update(config)

        # ================================================================
        # DYNAMIC THRESHOLD LOADING
        # ================================================================
        # Prioritize values from the merged config (presets).
        # If not found, fall back to the arguments passed to __init__.
        self.trend_aligned_threshold = self.config.get(
            'council_trend_aligned', trend_aligned_threshold
        )
        self.counter_trend_threshold = self.config.get(
            'council_counter_trend', counter_trend_threshold
        )
        
        # Weights
        self.w_trend = weight_trend
        self.w_structure = weight_structure
        self.w_momentum = weight_momentum
        self.w_pattern = weight_pattern
        self.w_volume = weight_volume
        
        # Validate weights sum to 5.0
        total_weight = sum([
            self.w_trend, self.w_structure, self.w_momentum,
            self.w_pattern, self.w_volume
        ])
        if abs(total_weight - 5.0) > 0.01:
            logger.warning(f"[COUNCIL] Weights sum to {total_weight:.2f}, not 5.0")
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'trend_aligned_trades': 0,
            'counter_trend_trades': 0,
            'avg_score_on_trade': [],
            'avg_score_on_hold': [],
        }
        
        # Decision history
        self.decision_history = deque(maxlen=100)
        
        # Regime tracking
        self.previous_regime = None
        self.regime_initialized = False
        
        self._log_initialization()
    
    def _get_default_config(self) -> Dict:
        """Asset-specific configurations"""
        if self.asset_type == "BTC":
            return {
                'rsi_bullish_zone': (40, 65),
                'rsi_oversold_bonus': 30,
                'sr_proximity_pct': 0.015,  # 1.5%
                'volume_ma_period': 20,
                'pattern_confidence_min': 0.60,
                'macd_confirmation': True,
            }
        else:  # GOLD
            return {
                'rsi_bullish_zone': (35, 60),
                'rsi_oversold_bonus': 25,
                'sr_proximity_pct': 0.010,  # 1.0%
                'volume_ma_period': 20,
                'pattern_confidence_min': 0.65,
                'macd_confirmation': True,
            }
    
    def _log_initialization(self):
        """Log startup configuration"""
        logger.info("=" * 80)
        logger.info(f"🏛️  INSTITUTIONAL COUNCIL AGGREGATOR - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("   COUNCIL MEMBERS (Judges):")
        logger.info(f"   1. TREND      ({self.w_trend:.1f} pts) - EMA alignment")
        logger.info(f"   2. STRUCTURE  ({self.w_structure:.1f} pts) - S/R + AI pivots")
        logger.info(f"   3. MOMENTUM   ({self.w_momentum:.1f} pt)  - RSI + MACD")
        logger.info(f"   4. PATTERN    ({self.w_pattern:.1f} pt)  - AI candlesticks")
        logger.info(f"   5. VOLUME     ({self.w_volume:.1f} pt)  - Volume confirmation")
        logger.info("")
        logger.info("   DECISION RULES:")
        logger.info(f"   • Trend-aligned:  ≥ {self.trend_aligned_threshold:.1f} / 5.0")
        logger.info(f"   • Counter-trend:  ≥ {self.counter_trend_threshold:.1f} / 5.0")
        logger.info("")
        logger.info(f"   AI Validation: {'ENABLED' if self.ai_validator else 'DISABLED'}")
        logger.info("=" * 80)
        logger.info("")
        
    def _format_ai_validation_for_viz(
        self, final_signal: int, details: dict, df: pd.DataFrame
    ) -> dict:
        """
        Format AI validation results for visualization
        (Same as PerformanceWeightedAggregator implementation)
        
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
            if not self.ai_validator:
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

                viz_data["pattern_detected"] = pattern_result.get("pattern_confirmed", False)
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
                # Signal was approved
                viz_data["validation_passed"] = True
                
                # Check if it was a bypass (council doesn't have bypass logic, so always approved)
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
    
    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main council decision logic with FIXED confidence extraction
        
        Returns:
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            details: Comprehensive breakdown with ALL confidence fields
        """
        self.stats['total_evaluations'] += 1
        timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"
        
        try:
            # Get regime context
            is_bull, regime_conf = self._detect_regime(df)
            regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"
            
            # ================================================================
            # FIX 1: Extract strategy signals AND confidences
            # ================================================================
            mr_signal, mr_conf = 0, 0.0
            tf_signal, tf_conf = 0, 0.0
            ema_signal, ema_conf = 0, 0.0
            
            # Mean Reversion
            if self.s_mean_reversion:
                try:
                    mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] MR signal error: {e}")
            
            # Trend Following
            if self.s_trend_following:
                try:
                    tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] TF signal error: {e}")
            
            # EMA Strategy (for regime)
            if self.s_ema:
                try:
                    ema_signal, ema_conf = self.s_ema.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] EMA signal error: {e}")
            
            # Initialize council scorecard
            scores = {
                'trend': 0.0,
                'structure': 0.0,
                'momentum': 0.0,
                'pattern': 0.0,
                'volume': 0.0,
            }
            explanations = []
            
            # Run all judges (unchanged)
            trend_score, trend_exp = self._judge_trend(df, is_bull)
            scores['trend'] = trend_score
            explanations.append(trend_exp)
            
            structure_score, structure_exp = self._judge_structure(df)
            scores['structure'] = structure_score
            explanations.append(structure_exp)
            
            momentum_score, momentum_exp = self._judge_momentum(df)
            scores['momentum'] = momentum_score
            explanations.append(momentum_exp)
            
            pattern_score, pattern_exp = self._judge_pattern(df)
            scores['pattern'] = pattern_score
            explanations.append(pattern_exp)
            
            volume_score, volume_exp = self._judge_volume(df)
            scores['volume'] = volume_score
            explanations.append(volume_exp)
            
            # Calculate total score
            total_score = sum(scores.values())
            
            # Decision logic
            signal = 0
            decision_type = "HOLD"
            
            if trend_score > 0 and total_score >= self.trend_aligned_threshold:
                signal = 1
                decision_type = "BUY (Trend-Aligned)"
                self.stats['trend_aligned_trades'] += 1
            elif trend_score == 0 and total_score >= self.counter_trend_threshold:
                signal = 1
                decision_type = "BUY (Counter-Trend Reversal)"
                self.stats['counter_trend_trades'] += 1
            
            # Update statistics
            if signal == 1:
                self.stats['buy_signals'] += 1
                self.stats['avg_score_on_trade'].append(total_score)
            elif signal == -1:
                self.stats['sell_signals'] += 1
                self.stats['avg_score_on_trade'].append(total_score)
            else:
                self.stats['hold_signals'] += 1
                self.stats['avg_score_on_hold'].append(total_score)
            
            required_score = self.trend_aligned_threshold if trend_score > 0 else self.counter_trend_threshold
            
            # ================================================================
            # FIX 2: Calculate signal_quality (normalize council score to 0-1)
            # ================================================================
            # Council score is out of 5.0, normalize to 0-1
            base_quality = min(total_score / 5.0, 1.0)
            
            # Boost quality if multiple judges agree
            judge_agreement = sum(1 for s in scores.values() if s > 0) / len(scores)
            signal_quality = base_quality * (0.8 + 0.2 * judge_agreement)
            signal_quality = min(signal_quality, 1.0)
            
            # ================================================================
            # FIX 3: Build comprehensive details dict with ALL fields
            # ================================================================
            details = {
                'timestamp': timestamp,
                'signal': signal,
                'decision_type': decision_type,
                'total_score': total_score,
                'required_score': required_score,
                'scores': scores,
                'regime': regime_name,
                'regime_confidence': regime_conf,
                'explanations': explanations,
                'signal_quality': signal_quality,
                
                # ✅ FIX: Add reasoning string for compatibility
                'reasoning': f"{decision_type} (Score: {total_score:.2f}/{required_score:.1f})",
                
                # ✅ FIX: Add individual strategy signals and confidences
                'mr_signal': mr_signal,
                'mr_confidence': mr_conf,
                'tf_signal': tf_signal,
                'tf_confidence': tf_conf,
                'ema_signal': ema_signal,
                'ema_confidence': ema_conf,
                
                # ✅ FIX: Add buy/sell scores for compatibility
                'buy_score': total_score if signal == 1 else 0.0,
                'sell_score': total_score if signal == -1 else 0.0,
                
                # Council-specific metadata
                'aggregator_type': 'council',
                'judge_agreement': judge_agreement,
            }
            
            # Log decision
            if self.detailed_logging or signal != 0:
                self._log_decision(details)
            
            # Store history
            self.decision_history.append({
                'timestamp': timestamp,
                'signal': signal,
                'score': total_score,
                'regime': regime_name,
            })
            
            # AI validation (if enabled and signal is not HOLD)
            if self.ai_validator:
                validated_signal, ai_details = self.ai_validator.validate_signal(
                    signal=signal,
                    signal_details=details,
                    df=df,
                )
                
                if validated_signal != signal:
                    logger.warning(f"[AI] Overruled: {signal} → {validated_signal}")
                    signal = validated_signal
                    details['ai_validation'] = ai_details
                    details['ai_modified'] = True
                    # Update signal in details
                    details['signal'] = signal
            
            return signal, details
            
        except Exception as e:
            logger.error(f"[COUNCIL] Error: {e}", exc_info=True)
            return 0, {
                'error': str(e),
                'timestamp': timestamp,
                'signal': 0,
                'total_score': 0.0,
                'signal_quality': 0.0,
                # Add default confidence fields to prevent crashes
                'mr_signal': 0,
                'mr_confidence': 0.0,
                'tf_signal': 0,
                'tf_confidence': 0.0,
                'ema_signal': 0,
                'ema_confidence': 0.0,
                'reasoning': f"error: {str(e)[:50]}",
            }
        
    def _judge_trend(self, df: pd.DataFrame, is_bull: bool) -> Tuple[float, str]:
        """
        JUDGE 1: TREND (1.5 pts)
        
        Rules:
        - Price above EMA50 AND EMA20 > EMA50 = 1.5 pts
        - Price above EMA50 BUT EMA20 < EMA50 = 0.75 pts (partial credit)
        - Price below EMA50 = 0.0 pts
        """
        try:
            features = self.s_ema.generate_features(df.tail(250))
            if features.empty:
                return 0.0, "TREND: No data"
            
            latest = features.iloc[-1]
            price = latest['close']
            ema_20 = latest.get('ema_fast', 0)
            ema_50 = latest.get('ema_slow', 0)
            
            if price > ema_50:
                if ema_20 > ema_50:
                    return self.w_trend, f"TREND: ✅ Full credit ({self.w_trend:.1f}) - Price > EMA50, EMA20 > EMA50"
                else:
                    partial = self.w_trend * 0.5
                    return partial, f"TREND: ⚠️  Partial ({partial:.1f}) - Price > EMA50 but EMA20 < EMA50"
            else:
                return 0.0, "TREND: ❌ No credit (0.0) - Price below EMA50"
                
        except Exception as e:
            logger.error(f"[TREND] Error: {e}")
            return 0.0, f"TREND: Error - {str(e)}"
    
    def _judge_structure(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        JUDGE 2: STRUCTURE (1.5 pts)
        
        Rules:
        - At key S/R level (AI + indicators) = 1.5 pts
        - Near S/R (within threshold) = 0.75 pts
        - No S/R nearby = 0.0 pts
        """
        try:
            current_price = float(df['close'].iloc[-1])
            threshold_pct = self.config['sr_proximity_pct']
            
            # Check AI validator for S/R (if available)
            if self.ai_validator:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    df=df,
                    current_price=current_price,
                    signal=1,  # Checking for support
                    threshold=threshold_pct,
                )
                
                if sr_result.get('near_level'):
                    level = sr_result.get('nearest_level')
                    dist_pct = sr_result.get('distance_pct', 0)
                    
                    if dist_pct < (threshold_pct * 50):  # Within 50% of threshold
                        return self.w_structure, f"STRUCTURE: ✅ Full credit ({self.w_structure:.1f}) - At S/R ${level:.2f} ({dist_pct:.2f}%)"
                    else:
                        partial = self.w_structure * 0.5
                        return partial, f"STRUCTURE: ⚠️  Partial ({partial:.1f}) - Near S/R ${level:.2f} ({dist_pct:.2f}%)"
            
            # Fallback: Use mean reversion's pivot detection
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            if mr_signal == 1 and mr_conf > 0.6:
                return self.w_structure * 0.75, f"STRUCTURE: ⚠️  Partial ({self.w_structure * 0.75:.1f}) - MR bounce signal"
            
            return 0.0, "STRUCTURE: ❌ No S/R nearby"
            
        except Exception as e:
            logger.error(f"[STRUCTURE] Error: {e}")
            return 0.0, f"STRUCTURE: Error - {str(e)}"
    
    def _judge_momentum(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        JUDGE 3: MOMENTUM (1.0 pt)
        
        Rules:
        - RSI in bullish zone (40-65) = 1.0 pt
        - RSI oversold (<30) = 1.0 pt (dip-buying bonus)
        - MACD confirmation adds +0.2 bonus (if enabled)
        """
        try:
            # Get RSI from mean reversion strategy
            features_mr = self.s_mean_reversion.generate_features(df.tail(100))
            if features_mr.empty:
                return 0.0, "MOMENTUM: No data"
            
            rsi = features_mr.iloc[-1].get('rsi', 50)
            bullish_min, bullish_max = self.config['rsi_bullish_zone']
            oversold = self.config['rsi_oversold_bonus']
            
            score = 0.0
            explanation = ""
            
            # RSI scoring
            if bullish_min <= rsi <= bullish_max:
                score = self.w_momentum
                explanation = f"MOMENTUM: ✅ Full credit ({self.w_momentum:.1f}) - RSI {rsi:.1f} in bullish zone"
            elif rsi < oversold:
                score = self.w_momentum
                explanation = f"MOMENTUM: ✅ Oversold bonus ({self.w_momentum:.1f}) - RSI {rsi:.1f} < {oversold}"
            else:
                explanation = f"MOMENTUM: ❌ No credit - RSI {rsi:.1f} outside zones"
            
            # MACD confirmation bonus (optional)
            if self.config['macd_confirmation'] and score > 0:
                macd = features_mr.iloc[-1].get('macd', 0)
                macd_signal = features_mr.iloc[-1].get('macd_signal', 0)
                
                if macd > macd_signal:
                    bonus = 0.2
                    score = min(score + bonus, self.w_momentum)  # Cap at max weight
                    explanation += f" + MACD bonus (+{bonus:.1f})"
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"[MOMENTUM] Error: {e}")
            return 0.0, f"MOMENTUM: Error - {str(e)}"
    
    def _judge_pattern(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        JUDGE 4: PATTERN (0.5 pt)
        
        Rules:
        - AI pattern confidence > 60% = 0.5 pt
        - AI pattern confidence > 75% = 0.5 pt + small bonus
        """
        try:
            if not self.ai_validator:
                return 0.0, "PATTERN: AI disabled"
            
            # Check AI pattern
            pattern_result = self.ai_validator._check_pattern(
                df=df,
                signal=1,
                min_confidence=self.config['pattern_confidence_min'],
            )
            
            if pattern_result.get('pattern_confirmed'):
                confidence = pattern_result.get('confidence', 0)
                pattern_name = pattern_result.get('pattern_name', 'Unknown')
                
                if confidence > 0.75:
                    return self.w_pattern, f"PATTERN: ✅ Full credit ({self.w_pattern:.1f}) - {pattern_name} ({confidence:.0%})"
                else:
                    return self.w_pattern * 0.8, f"PATTERN: ⚠️  Partial ({self.w_pattern * 0.8:.1f}) - {pattern_name} ({confidence:.0%})"
            
            return 0.0, f"PATTERN: ❌ No pattern (conf < {self.config['pattern_confidence_min']:.0%})"
            
        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return 0.0, f"PATTERN: Error - {str(e)}"
    
    def _judge_volume(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        JUDGE 5: VOLUME (0.5 pt)
        
        Rules:
        - Volume > MA(20) = 0.5 pt
        - Volume > 1.5x MA(20) = 0.5 pt + emphasis
        """
        try:
            if 'volume' not in df.columns:
                return 0.0, "VOLUME: No volume data"
            
            volume_ma_period = self.config['volume_ma_period']
            current_volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(volume_ma_period).mean().iloc[-1]
            
            if current_volume > volume_ma * 1.5:
                return self.w_volume, f"VOLUME: ✅ Strong ({self.w_volume:.1f}) - {current_volume/volume_ma:.1f}x avg"
            elif current_volume > volume_ma:
                return self.w_volume * 0.7, f"VOLUME: ⚠️  Partial ({self.w_volume * 0.7:.1f}) - {current_volume/volume_ma:.1f}x avg"
            else:
                return 0.0, f"VOLUME: ❌ Below average ({current_volume/volume_ma:.1f}x)"
            
        except Exception as e:
            logger.error(f"[VOLUME] Error: {e}")
            return 0.0, f"VOLUME: Error - {str(e)}"
    
    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Leverage existing EMA strategy for regime detection"""
        try:
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            
            # EMA signal: 1 = bullish, -1 = bearish, 0 = neutral
            is_bull = ema_signal >= 0
            
            return is_bull, ema_conf
            
        except Exception as e:
            logger.error(f"[REGIME] Error: {e}")
            return False, 0.5  # Default to bearish with low confidence
    
    def _log_decision(self, details: Dict):
        """Log council decision breakdown"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🏛️  COUNCIL DECISION - {details['regime']}")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {details['timestamp']}")
        logger.info(f"")
        logger.info(f"SCORECARD:")
        
        for judge, score in details['scores'].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")
        
        logger.info(f"")
        logger.info(f"TOTAL SCORE: {details['total_score']:.2f} / 5.0")
        logger.info(f"REQUIRED:    {details['required_score']:.2f}")
        logger.info(f"")
        logger.info(f"DECISION: {details['decision_type']}")
        logger.info(f"SIGNAL:   {details['signal']:+2d}")
        logger.info(f"")
        logger.info(f"REASONING:")
        for exp in details['explanations']:
            logger.info(f"  • {exp}")
        logger.info("=" * 80)
        logger.info("")
    
    def get_statistics(self) -> Dict:
        """Return aggregator statistics"""
        total = max(self.stats['total_evaluations'], 1)
        
        return {
            **self.stats,
            'buy_rate': (self.stats['buy_signals'] / total) * 100,
            'sell_rate': (self.stats['sell_signals'] / total) * 100,
            'hold_rate': (self.stats['hold_signals'] / total) * 100,
            'avg_score_on_trade': (
                np.mean(self.stats['avg_score_on_trade']) 
                if self.stats['avg_score_on_trade'] else 0.0
            ),
            'avg_score_on_hold': (
                np.mean(self.stats['avg_score_on_hold']) 
                if self.stats['avg_score_on_hold'] else 0.0
            ),
        }