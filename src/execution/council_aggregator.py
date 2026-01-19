"""
Institutional Council Aggregator - Bidirectional Version
Supports both BUY and SELL signals with symmetric logic
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
    "BlackRock-Style" Weighted Council with Bidirectional Signals
    
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
    
    NEW: Symmetric scoring for both BUY and SELL signals
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
        
        # Configuration merge
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Dynamic threshold loading
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
            'trend_aligned_buys': 0,
            'trend_aligned_sells': 0,
            'counter_trend_buys': 0,
            'counter_trend_sells': 0,
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
                'rsi_bearish_zone': (35, 60),
                'rsi_oversold_bonus': 30,
                'rsi_overbought_bonus': 70,
                'sr_proximity_pct': 0.015,  # 1.5%
                'volume_ma_period': 20,
                'pattern_confidence_min': 0.60,
                'macd_confirmation': True,
            }
        else:  # GOLD
            return {
                'rsi_bullish_zone': (35, 60),
                'rsi_bearish_zone': (40, 65),
                'rsi_oversold_bonus': 25,
                'rsi_overbought_bonus': 75,
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
        logger.info("   DECISION RULES (Bidirectional):")
        logger.info(f"   • Trend-aligned:  ≥ {self.trend_aligned_threshold:.1f} / 5.0")
        logger.info(f"   • Counter-trend:  ≥ {self.counter_trend_threshold:.1f} / 5.0")
        logger.info("")
        logger.info(f"   AI Validation: {'ENABLED' if self.ai_validator else 'DISABLED'}")
        logger.info("=" * 80)
        logger.info("")
    
    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main council decision logic with bidirectional support
        
        Returns:
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            details: Comprehensive breakdown
        """
        self.stats['total_evaluations'] += 1
        timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"
        
        try:
            # Get regime context
            is_bull, regime_conf = self._detect_regime(df)
            regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"
            
            # Extract strategy signals
            mr_signal, mr_conf = 0, 0.0
            tf_signal, tf_conf = 0, 0.0
            ema_signal, ema_conf = 0, 0.0
            
            if self.s_mean_reversion:
                try:
                    mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] MR signal error: {e}")
            
            if self.s_trend_following:
                try:
                    tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] TF signal error: {e}")
            
            if self.s_ema:
                try:
                    ema_signal, ema_conf = self.s_ema.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] EMA signal error: {e}")
            
            # ================================================================
            # BIDIRECTIONAL SCORING: Evaluate both BUY and SELL
            # ================================================================
            
            # BUY scorecard
            buy_scores = {
                'trend': 0.0,
                'structure': 0.0,
                'momentum': 0.0,
                'pattern': 0.0,
                'volume': 0.0,
            }
            buy_explanations = []
            
            # SELL scorecard
            sell_scores = {
                'trend': 0.0,
                'structure': 0.0,
                'momentum': 0.0,
                'pattern': 0.0,
                'volume': 0.0,
            }
            sell_explanations = []
            
            # Run all judges for both directions
            buy_scores['trend'], sell_scores['trend'], trend_exp = self._judge_trend_bidirectional(df, is_bull)
            buy_explanations.append(trend_exp['buy'])
            sell_explanations.append(trend_exp['sell'])
            
            buy_scores['structure'], sell_scores['structure'], structure_exp = self._judge_structure_bidirectional(df)
            buy_explanations.append(structure_exp['buy'])
            sell_explanations.append(structure_exp['sell'])
            
            buy_scores['momentum'], sell_scores['momentum'], momentum_exp = self._judge_momentum_bidirectional(df)
            buy_explanations.append(momentum_exp['buy'])
            sell_explanations.append(momentum_exp['sell'])
            
            buy_scores['pattern'], sell_scores['pattern'], pattern_exp = self._judge_pattern_bidirectional(df)
            buy_explanations.append(pattern_exp['buy'])
            sell_explanations.append(pattern_exp['sell'])
            
            buy_scores['volume'], sell_scores['volume'], volume_exp = self._judge_volume_bidirectional(df)
            buy_explanations.append(volume_exp['buy'])
            sell_explanations.append(volume_exp['sell'])
            
            # Calculate total scores
            buy_total = sum(buy_scores.values())
            sell_total = sum(sell_scores.values())
            
            # ================================================================
            # DECISION LOGIC: Choose strongest direction
            # ================================================================
            signal = 0
            decision_type = "HOLD"
            chosen_scores = {}
            chosen_explanations = []
            total_score = 0.0
            required_score = 0.0
            
            # BUY decision
            if is_bull and buy_total >= self.trend_aligned_threshold:
                signal = 1
                decision_type = "BUY (Trend-Aligned)"
                self.stats['trend_aligned_buys'] += 1
                chosen_scores = buy_scores
                chosen_explanations = buy_explanations
                total_score = buy_total
                required_score = self.trend_aligned_threshold
                
            elif not is_bull and buy_total >= self.counter_trend_threshold:
                signal = 1
                decision_type = "BUY (Counter-Trend Reversal)"
                self.stats['counter_trend_buys'] += 1
                chosen_scores = buy_scores
                chosen_explanations = buy_explanations
                total_score = buy_total
                required_score = self.counter_trend_threshold
            
            # SELL decision (only if no BUY signal)
            elif not is_bull and sell_total >= self.trend_aligned_threshold:
                signal = -1
                decision_type = "SELL (Trend-Aligned)"
                self.stats['trend_aligned_sells'] += 1
                chosen_scores = sell_scores
                chosen_explanations = sell_explanations
                total_score = sell_total
                required_score = self.trend_aligned_threshold
                
            elif is_bull and sell_total >= self.counter_trend_threshold:
                signal = -1
                decision_type = "SELL (Counter-Trend Reversal)"
                self.stats['counter_trend_sells'] += 1
                chosen_scores = sell_scores
                chosen_explanations = sell_explanations
                total_score = sell_total
                required_score = self.counter_trend_threshold
            
            else:
                # HOLD - show both scores
                decision_type = f"HOLD (BUY: {buy_total:.1f}, SELL: {sell_total:.1f})"
                chosen_scores = {'buy': buy_scores, 'sell': sell_scores}
                chosen_explanations = buy_explanations + sell_explanations
                total_score = max(buy_total, sell_total)
                required_score = self.trend_aligned_threshold
            
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
            
            # Calculate signal quality
            base_quality = min(total_score / 5.0, 1.0)
            if signal != 0:
                judge_agreement = sum(1 for s in chosen_scores.values() if s > 0) / len(chosen_scores)
            else:
                judge_agreement = 0.5
            signal_quality = base_quality * (0.8 + 0.2 * judge_agreement)
            signal_quality = min(signal_quality, 1.0)
            
            # Build details dict
            details = {
                'timestamp': timestamp,
                'signal': signal,
                'decision_type': decision_type,
                'total_score': total_score,
                'required_score': required_score,
                'scores': chosen_scores,
                'buy_scores': buy_scores,
                'sell_scores': sell_scores,
                'buy_total': buy_total,
                'sell_total': sell_total,
                'regime': regime_name,
                'regime_confidence': regime_conf,
                'explanations': chosen_explanations,
                'signal_quality': signal_quality,
                'reasoning': f"{decision_type} (Score: {total_score:.2f}/{required_score:.1f})",
                'mr_signal': mr_signal,
                'mr_confidence': mr_conf,
                'tf_signal': tf_signal,
                'tf_confidence': tf_conf,
                'ema_signal': ema_signal,
                'ema_confidence': ema_conf,
                'buy_score': buy_total,
                'sell_score': sell_total,
                'aggregator_type': 'council',
                'judge_agreement': judge_agreement,
            }
            
            # Log decision
            if self.detailed_logging or signal != 0:
                self._log_decision_bidirectional(details)
            
            # Store history
            self.decision_history.append({
                'timestamp': timestamp,
                'signal': signal,
                'score': total_score,
                'regime': regime_name,
            })
            
            # AI validation
            if self.ai_validator and signal != 0:
                original_signal = signal
                
                validated_signal, ai_details = self.ai_validator.validate_signal(
                    signal=signal,
                    signal_details=details,
                    df=df,
                )
                
                if validated_signal != signal:
                    logger.warning(f"[AI] Overruled: {signal} → {validated_signal}")
                    signal = validated_signal
                    details['ai_modified'] = True
                    details['signal'] = signal
                
                try:
                    formatted_ai = self._format_ai_validation_for_viz(
                        final_signal=signal,
                        details=details.copy(),
                        df=df
                    )
                    details['ai_validation'] = formatted_ai
                except Exception as e:
                    logger.error(f"[COUNCIL] AI formatting failed: {e}")
                    details['ai_validation'] = {
                        "pattern_detected": False,
                        "pattern_name": "Error",
                        "pattern_confidence": 0.0,
                        "validation_passed": signal != 0,
                        "action": "error_formatting",
                        "error": str(e),
                    }
            
            elif self.ai_validator and signal == 0:
                details['ai_validation'] = {
                    "pattern_detected": False,
                    "pattern_name": "None",
                    "pattern_confidence": 0.0,
                    "validation_passed": True,
                    "action": "hold",
                }
            
            return signal, details
            
        except Exception as e:
            logger.error(f"[COUNCIL] Error: {e}", exc_info=True)
            return 0, {
                'error': str(e),
                'timestamp': timestamp,
                'signal': 0,
                'total_score': 0.0,
                'signal_quality': 0.0,
                'mr_signal': 0,
                'mr_confidence': 0.0,
                'tf_signal': 0,
                'tf_confidence': 0.0,
                'ema_signal': 0,
                'ema_confidence': 0.0,
                'reasoning': f"error: {str(e)[:50]}",
            }
    
    # ========================================================================
    # BIDIRECTIONAL JUDGES
    # ========================================================================
    
    def _judge_trend_bidirectional(self, df: pd.DataFrame, is_bull: bool) -> Tuple[float, float, Dict]:
        """
        JUDGE 1: TREND (Bidirectional)
        
        BUY Rules:
        - Price > EMA50 AND EMA20 > EMA50 = 1.5 pts
        - Price > EMA50 BUT EMA20 < EMA50 = 0.75 pts
        
        SELL Rules:
        - Price < EMA50 AND EMA20 < EMA50 = 1.5 pts
        - Price < EMA50 BUT EMA20 > EMA50 = 0.75 pts
        """
        try:
            features = self.s_ema.generate_features(df.tail(250))
            if features.empty:
                return 0.0, 0.0, {'buy': "TREND: No data", 'sell': "TREND: No data"}
            
            latest = features.iloc[-1]
            price = latest['close']
            ema_20 = latest.get('ema_fast', 0)
            ema_50 = latest.get('ema_slow', 0)
            
            buy_score = 0.0
            sell_score = 0.0
            
            # BUY scoring
            if price > ema_50:
                if ema_20 > ema_50:
                    buy_score = self.w_trend
                    buy_exp = f"TREND BUY: ✅ Full ({self.w_trend:.1f}) - Price > EMA50, EMA20 > EMA50"
                else:
                    buy_score = self.w_trend * 0.5
                    buy_exp = f"TREND BUY: ⚠️ Partial ({buy_score:.1f}) - Price > EMA50 but EMA20 < EMA50"
            else:
                buy_exp = "TREND BUY: ❌ No credit - Price < EMA50"
            
            # SELL scoring
            if price < ema_50:
                if ema_20 < ema_50:
                    sell_score = self.w_trend
                    sell_exp = f"TREND SELL: ✅ Full ({self.w_trend:.1f}) - Price < EMA50, EMA20 < EMA50"
                else:
                    sell_score = self.w_trend * 0.5
                    sell_exp = f"TREND SELL: ⚠️ Partial ({sell_score:.1f}) - Price < EMA50 but EMA20 > EMA50"
            else:
                sell_exp = "TREND SELL: ❌ No credit - Price > EMA50"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[TREND] Error: {e}")
            return 0.0, 0.0, {'buy': f"TREND: Error", 'sell': f"TREND: Error"}
    
    def _judge_structure_bidirectional(self, df: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        JUDGE 2: STRUCTURE (Bidirectional)
        
        BUY: At support level
        SELL: At resistance level
        """
        try:
            current_price = float(df['close'].iloc[-1])
            threshold_pct = self.config['sr_proximity_pct']
            
            buy_score = 0.0
            sell_score = 0.0
            
            if self.ai_validator:
                # Check support (BUY)
                sr_buy = self.ai_validator._check_support_resistance_fixed(
                    df=df,
                    current_price=current_price,
                    signal=1,
                    threshold=threshold_pct,
                )
                
                if sr_buy.get('near_level'):
                    level = sr_buy.get('nearest_level')
                    dist_pct = sr_buy.get('distance_pct', 0)
                    
                    if dist_pct < (threshold_pct * 50):
                        buy_score = self.w_structure
                        buy_exp = f"STRUCT BUY: ✅ Full ({self.w_structure:.1f}) - At Support ${level:.2f}"
                    else:
                        buy_score = self.w_structure * 0.5
                        buy_exp = f"STRUCT BUY: ⚠️ Partial ({buy_score:.1f}) - Near Support ${level:.2f}"
                else:
                    buy_exp = "STRUCT BUY: ❌ No support nearby"
                
                # Check resistance (SELL)
                sr_sell = self.ai_validator._check_support_resistance_fixed(
                    df=df,
                    current_price=current_price,
                    signal=-1,
                    threshold=threshold_pct,
                )
                
                if sr_sell.get('near_level'):
                    level = sr_sell.get('nearest_level')
                    dist_pct = sr_sell.get('distance_pct', 0)
                    
                    if dist_pct < (threshold_pct * 50):
                        sell_score = self.w_structure
                        sell_exp = f"STRUCT SELL: ✅ Full ({self.w_structure:.1f}) - At Resistance ${level:.2f}"
                    else:
                        sell_score = self.w_structure * 0.5
                        sell_exp = f"STRUCT SELL: ⚠️ Partial ({sell_score:.1f}) - Near Resistance ${level:.2f}"
                else:
                    sell_exp = "STRUCT SELL: ❌ No resistance nearby"
            else:
                buy_exp = "STRUCT BUY: AI disabled"
                sell_exp = "STRUCT SELL: AI disabled"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[STRUCTURE] Error: {e}")
            return 0.0, 0.0, {'buy': "STRUCT: Error", 'sell': "STRUCT: Error"}
    
    def _judge_momentum_bidirectional(self, df: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        JUDGE 3: MOMENTUM (Bidirectional)
        
        BUY: RSI oversold or in bullish zone
        SELL: RSI overbought or in bearish zone
        """
        try:
            features_mr = self.s_mean_reversion.generate_features(df.tail(100))
            if features_mr.empty:
                return 0.0, 0.0, {'buy': "MOM: No data", 'sell': "MOM: No data"}
            
            rsi = features_mr.iloc[-1].get('rsi', 50)
            
            # Config values
            bullish_min, bullish_max = self.config['rsi_bullish_zone']
            bearish_min, bearish_max = self.config['rsi_bearish_zone']
            oversold = self.config['rsi_oversold_bonus']
            overbought = self.config['rsi_overbought_bonus']
            
            buy_score = 0.0
            sell_score = 0.0
            
            # BUY scoring
            if bullish_min <= rsi <= bullish_max:
                buy_score = self.w_momentum
                buy_exp = f"MOM BUY: ✅ Full ({self.w_momentum:.1f}) - RSI {rsi:.1f} bullish"
            elif rsi < oversold:
                buy_score = self.w_momentum
                buy_exp = f"MOM BUY: ✅ Oversold ({self.w_momentum:.1f}) - RSI {rsi:.1f}"
            else:
                buy_exp = f"MOM BUY: ❌ No credit - RSI {rsi:.1f}"
            
            # SELL scoring
            if bearish_min <= rsi <= bearish_max:
                sell_score = self.w_momentum
                sell_exp = f"MOM SELL: ✅ Full ({self.w_momentum:.1f}) - RSI {rsi:.1f} bearish"
            elif rsi > overbought:
                sell_score = self.w_momentum
                sell_exp = f"MOM SELL: ✅ Overbought ({self.w_momentum:.1f}) - RSI {rsi:.1f}"
            else:
                sell_exp = f"MOM SELL: ❌ No credit - RSI {rsi:.1f}"
            
            # MACD confirmation
            if self.config['macd_confirmation']:
                macd = features_mr.iloc[-1].get('macd', 0)
                macd_signal = features_mr.iloc[-1].get('macd_signal', 0)
                
                if buy_score > 0 and macd > macd_signal:
                    bonus = 0.2
                    buy_score = min(buy_score + bonus, self.w_momentum)
                    buy_exp += f" + MACD"
                
                if sell_score > 0 and macd < macd_signal:
                    bonus = 0.2
                    sell_score = min(sell_score + bonus, self.w_momentum)
                    sell_exp += f" + MACD"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[MOMENTUM] Error: {e}")
            return 0.0, 0.0, {'buy': "MOM: Error", 'sell': "MOM: Error"}
    
    def _judge_pattern_bidirectional(self, df: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        JUDGE 4: PATTERN (Bidirectional)
        
        BUY: Bullish AI pattern
        SELL: Bearish AI pattern
        """
        try:
            if not self.ai_validator:
                return 0.0, 0.0, {'buy': "PATTERN: AI disabled", 'sell': "PATTERN: AI disabled"}
            
            buy_score = 0.0
            sell_score = 0.0
            
            # Check bullish pattern
            pattern_buy = self.ai_validator._check_pattern(
                df=df,
                signal=1,
                min_confidence=self.config['pattern_confidence_min'],
            )
            
            if pattern_buy.get('pattern_confirmed'):
                conf = pattern_buy.get('confidence', 0)
                name = pattern_buy.get('pattern_name', 'Unknown')
                
                if conf > 0.75:
                    buy_score = self.w_pattern
                    buy_exp = f"PATTERN BUY: ✅ Full ({self.w_pattern:.1f}) - {name} ({conf:.0%})"
                else:
                    buy_score = self.w_pattern * 0.8
                    buy_exp = f"PATTERN BUY: ⚠️ Partial ({buy_score:.1f}) - {name} ({conf:.0%})"
            else:
                buy_exp = "PATTERN BUY: ❌ No pattern"
            
            # Check bearish pattern
            pattern_sell = self.ai_validator._check_pattern(
                df=df,
                signal=-1,
                min_confidence=self.config['pattern_confidence_min'],
            )
            
            if pattern_sell.get('pattern_confirmed'):
                conf = pattern_sell.get('confidence', 0)
                name = pattern_sell.get('pattern_name', 'Unknown')
                
                if conf > 0.75:
                    sell_score = self.w_pattern
                    sell_exp = f"PATTERN SELL: ✅ Full ({self.w_pattern:.1f}) - {name} ({conf:.0%})"
                else:
                    sell_score = self.w_pattern * 0.8
                    sell_exp = f"PATTERN SELL: ⚠️ Partial ({sell_score:.1f}) - {name} ({conf:.0%})"
            else:
                sell_exp = "PATTERN SELL: ❌ No pattern"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return 0.0, 0.0, {'buy': "PATTERN: Error", 'sell': "PATTERN: Error"}
    
    def _judge_volume_bidirectional(self, df: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        JUDGE 5: VOLUME (Same for both directions)
        
        Both BUY and SELL benefit from high volume
        """
        try:
            if 'volume' not in df.columns:
                return 0.0, 0.0, {'buy': "VOL: No data", 'sell': "VOL: No data"}
            
            volume_ma_period = self.config['volume_ma_period']
            current_volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(volume_ma_period).mean().iloc[-1]
            
            vol_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # Same scoring for both directions
            if vol_ratio > 1.5:
                score = self.w_volume
                exp = f"VOLUME: ✅ Strong ({self.w_volume:.1f}) - {vol_ratio:.1f}x avg"
            elif vol_ratio > 1.0:
                score = self.w_volume * 0.7
                exp = f"VOLUME: ⚠️ Partial ({score:.1f}) - {vol_ratio:.1f}x avg"
            else:
                score = 0.0
                exp = f"VOLUME: ❌ Below avg ({vol_ratio:.1f}x)"
            
            return score, score, {'buy': exp, 'sell': exp}
            
        except Exception as e:
            logger.error(f"[VOLUME] Error: {e}")
            return 0.0, 0.0, {'buy': "VOL: Error", 'sell': "VOL: Error"}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Leverage existing EMA strategy for regime detection"""
        try:
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            is_bull = ema_signal >= 0
            return is_bull, ema_conf
        except Exception as e:
            logger.error(f"[REGIME] Error: {e}")
            return False, 0.5
    
    def _log_decision_bidirectional(self, details: Dict):
        """Log council decision with bidirectional breakdown"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🏛️  COUNCIL DECISION - {details['regime']}")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {details['timestamp']}")
        logger.info(f"")
        
        # Show both BUY and SELL scores
        logger.info(f"BUY SCORECARD (Total: {details['buy_total']:.2f}/5.0):")
        for judge, score in details['buy_scores'].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")
        
        logger.info(f"")
        logger.info(f"SELL SCORECARD (Total: {details['sell_total']:.2f}/5.0):")
        for judge, score in details['sell_scores'].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")
        
        logger.info(f"")
        logger.info(f"DECISION: {details['decision_type']}")
        logger.info(f"SIGNAL:   {details['signal']:+2d}")
        logger.info(f"SCORE:    {details['total_score']:.2f} / {details['required_score']:.2f}")
        logger.info("=" * 80)
        logger.info("")
    
    def _format_ai_validation_for_viz(self, final_signal: int, details: dict, df: pd.DataFrame) -> dict:
        """Format AI validation results for visualization"""
        try:
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

            if not self.ai_validator:
                viz_data["action"] = "ai_disabled"
                return viz_data

            current_price = float(df["close"].iloc[-1])

            # S/R Analysis
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
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }
            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")

            # Pattern Detection
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

                if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                    try:
                        snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                        first_open = snippet[0, 0]

                        if first_open > 0:
                            snippet_norm = snippet / first_open - 1
                            snippet_input = snippet_norm.reshape(1, 15, 4)
                            predictions = self.ai_validator.sniper.model.predict(snippet_input, verbose=0)[0]

                            top3_indices = predictions.argsort()[-3:][::-1]
                            top3_confidences = predictions[top3_indices]

                            top3_patterns = []
                            for idx in top3_indices:
                                pattern_name = self.ai_validator.reverse_pattern_map.get(idx, f"Pattern_{idx}")
                                top3_patterns.append(pattern_name)

                            viz_data["top3_patterns"] = top3_patterns
                            viz_data["top3_confidences"] = top3_confidences.tolist()
                    except Exception as e:
                        logger.debug(f"[VIZ] Top3 patterns failed: {e}")
            except Exception as e:
                logger.error(f"[VIZ] Pattern detection failed: {e}")

            # Validation Status
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
                viz_data["action"] = "approved"
            else:
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