"""
Rewritten Signal Aggregator - Clean, Sensible Logic
Key principle: Confidence modulates strength, never gates participation
EMA is regime detector ONLY, not a decision-maker
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
    Clean Multi-Strategy Signal Aggregator
    
    Architecture:
    1. EMA 50/200: Market regime detector (context only, no decisions)
    2. Mean Reversion & Trend Following: Decision makers (weighted by accuracy)
    3. Confidence: Modulates signal strength (0.0-1.0 multiplier)
    4. Agreement: Bonus for strategy consensus
    5. Regime: Mild context adjustment (not a filter)
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()
        
        # ============================================================
        # STRATEGY WEIGHTS (based on CV accuracy)
        # These weight the importance of each strategy's vote
        # ============================================================
        if self.asset_type == "BTC":
            self.weights = {
                'trend_following': 0.50,   # 69.15% - Most accurate
                'mean_reversion': 0.50,    # 64.74% - Co-equal weight
            }
        else:  # GOLD
            self.weights = {
                'ema_strategy': 0.00,      # NOT A DECISION MAKER
                'mean_reversion': 0.50,    # 75.13% - Equally weighted
                'trend_following': 0.50,   # 65.45% - Equally weighted
            }
        
        # ============================================================
        # CONFIGURATION (thresholds and bonuses)
        # ============================================================
        self.config = {
            # Thresholds for final signal generation
            # These are AGGREGATED scores (0-1 range)
            "buy_threshold": 0.35,     # If buy_score >= 0.35 → BUY
            "sell_threshold": 0.35,    # If sell_score >= 0.35 → SELL
            
            # Bonuses for strategy agreement
            "two_strategy_bonus": 0.15,    # Both strategies agree
            "three_strategy_bonus": 0.25,  # All three agree (unlikely)
            
            # Regime modifiers (MILD adjustments, not filters)
            "bull_buy_boost": 0.05,        # Slight boost in bull markets
            "bull_sell_penalty": 0.05,     # Slight penalty in bull markets
            "bear_sell_boost": 0.05,       # Slight boost in bear markets
            "bear_buy_penalty": 0.05,      # Slight penalty in bear markets
            
            # Minimum confidence to use a signal at all
            # Below this, signal is treated as noise
            "min_confidence_to_use": 0.15,
            
            # Quality gate for final decision
            "min_signal_quality": 0.25,
        }
        
        # Statistics tracking
        self.stats = {
            'total_evaluations': 0,
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'bull_regime_count': 0,
            'bear_regime_count': 0,
            'regime_changes': 0,
            'consensus_signals': 0,
            'single_strategy_signals': 0,
        }
        
        self.previous_regime = None
        self._log_initialization()

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯 REWRITTEN PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📊 STRATEGY ROLES:")
        logger.info("   Mean Reversion:   DECISION MAKER (weight: 0.50)")
        logger.info("   Trend Following:  DECISION MAKER (weight: 0.50)")
        logger.info("   EMA 50/200:       REGIME DETECTOR ONLY")
        logger.info("")
        logger.info("🎚️ THRESHOLDS:")
        logger.info(f"   Buy Signal:       score >= {self.config['buy_threshold']:.2f}")
        logger.info(f"   Sell Signal:      score >= {self.config['sell_threshold']:.2f}")
        logger.info(f"   Min Confidence:   {self.config['min_confidence_to_use']:.2f} (to use signal at all)")
        logger.info(f"   Min Quality:      {self.config['min_signal_quality']:.2f} (for final decision)")
        logger.info("")
        logger.info("🤝 AGREEMENT BONUSES:")
        logger.info(f"   2 strategies agree: +{self.config['two_strategy_bonus']:.2f}")
        logger.info(f"   3 strategies agree: +{self.config['three_strategy_bonus']:.2f}")
        logger.info("")
        logger.info("📈 REGIME CONTEXT (mild adjustments):")
        logger.info(f"   Bull: BUY +{self.config['bull_buy_boost']:.2f} | SELL -{self.config['bull_sell_penalty']:.2f}")
        logger.info(f"   Bear: SELL +{self.config['bear_sell_boost']:.2f} | BUY -{self.config['bear_buy_penalty']:.2f}")
        logger.info("=" * 80)
        logger.info("")

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect market regime using EMA crossover
        CONTEXT ONLY - does not generate signals
        Returns: (is_bull_market, regime_confidence)
        """
        ema_signal, ema_conf = self.s_ema.generate_signal(df)
        is_bull = (ema_signal == 1)  # 1 = Golden Cross (Bull)
        
        # Track regime changes
        if self.previous_regime is not None and self.previous_regime != is_bull:
            self.stats['regime_changes'] += 1
            regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
            logger.info("")
            logger.info("⚡ REGIME CHANGE")
            logger.info(f"   New: {regime_name}")
            logger.info(f"   Confidence: {ema_conf:.3f}")
            logger.info("")
        
        self.previous_regime = is_bull
        
        if is_bull:
            self.stats['bull_regime_count'] += 1
        else:
            self.stats['bear_regime_count'] += 1
        
        return is_bull, ema_conf

    def _calculate_score(
        self,
        target_signal: int,  # 1 for BUY, -1 for SELL
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """
        Calculate weighted score for a given direction (BUY or SELL)
        
        Logic:
        1. Only count strategies that agree with target_signal
        2. Confidence modulates the contribution (0.15-1.0 range)
        3. Agreement adds bonus
        4. Regime provides mild context adjustment
        
        Returns: (score, explanation, agreement_count)
        """
        components = []
        total_score = 0.0
        agreement_count = 0
        
        min_conf = self.config['min_confidence_to_use']
        
        # ============================================================
        # MEAN REVERSION CONTRIBUTION
        # ============================================================
        if mr_signal == target_signal:  # Only if it agrees
            # Confidence modulates strength (floor at min_conf to avoid noise)
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights['mean_reversion']
            total_score += contribution
            components.append(f"MR:{effective_conf:.2f}×{self.weights['mean_reversion']:.2f}={contribution:.3f}")
            agreement_count += 1
        
        # ============================================================
        # TREND FOLLOWING CONTRIBUTION
        # ============================================================
        if tf_signal == target_signal:  # Only if it agrees
            # Confidence modulates strength (floor at min_conf to avoid noise)
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights['trend_following']
            total_score += contribution
            components.append(f"TF:{effective_conf:.2f}×{self.weights['trend_following']:.2f}={contribution:.3f}")
            agreement_count += 1
        
        # ============================================================
        # BUILD EXPLANATION
        # ============================================================
        if components:
            explanation = " + ".join(components)
        else:
            explanation = "no_agreement"
            return 0.0, explanation, 0
        
        # ============================================================
        # AGREEMENT BONUS (small, only for consensus)
        # ============================================================
        if agreement_count == 2:
            bonus = self.config['two_strategy_bonus']
            total_score += bonus
            explanation += f" + bonus({bonus:.2f})"
        
        # ============================================================
        # REGIME CONTEXT (very mild, never decides)
        # ============================================================
        if target_signal == 1:  # BUY
            if is_bull:
                regime_adj = self.config['bull_buy_boost']
                total_score += regime_adj
                explanation += f" + bull_ctx({regime_adj:.2f})"
            else:
                regime_adj = -self.config['bear_buy_penalty']
                total_score = max(0.0, total_score + regime_adj)
                explanation += f" - bear_ctx({abs(regime_adj):.2f})"
        
        else:  # SELL (target_signal == -1)
            if is_bull:
                regime_adj = -self.config['bull_sell_penalty']
                total_score = max(0.0, total_score + regime_adj)
                explanation += f" - bull_ctx({abs(regime_adj):.2f})"
            else:
                regime_adj = self.config['bear_sell_boost']
                total_score += regime_adj
                explanation += f" + bear_ctx({regime_adj:.2f})"
        
        return total_score, explanation, agreement_count

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main aggregation logic - clean and simple
        
        Returns: (signal, details_dict)
          signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        self.stats['total_evaluations'] += 1
        
        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"
            
            # ============================================================
            # STEP 1: Detect regime (context only)
            # ============================================================
            is_bull, regime_conf = self._detect_regime(df)
            regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"
            
            # ============================================================
            # STEP 2: Get decision-maker signals
            # ============================================================
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            
            # (We don't use EMA signal anymore, only its regime detection)
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            
            # ============================================================
            # STEP 3: Calculate BUY and SELL scores
            # ============================================================
            buy_score, buy_explanation, buy_agreement = self._calculate_score(
                target_signal=1,  # BUY
                mr_signal=mr_signal,
                mr_conf=mr_conf,
                tf_signal=tf_signal,
                tf_conf=tf_conf,
                is_bull=is_bull,
            )
            
            sell_score, sell_explanation, sell_agreement = self._calculate_score(
                target_signal=-1,  # SELL
                mr_signal=mr_signal,
                mr_conf=mr_conf,
                tf_signal=tf_signal,
                tf_conf=tf_conf,
                is_bull=is_bull,
            )
            
            # ============================================================
            # STEP 4: Make final decision
            # ============================================================
            buy_thresh = self.config['buy_threshold']
            sell_thresh = self.config['sell_threshold']
            min_quality = self.config['min_signal_quality']
            
            signal_quality = max(buy_score, sell_score)
            
            # Decision logic: which threshold is crossed?
            if buy_score >= buy_thresh and buy_score > sell_score:
                if signal_quality >= min_quality:
                    final_signal = 1
                    reasoning = f"buy_consensus_{buy_agreement}_strategies"
                    self.stats['buy_signals'] += 1
                    self.stats['signals_generated'] += 1
                    if buy_agreement == 1:
                        self.stats['single_strategy_signals'] += 1
                    else:
                        self.stats['consensus_signals'] += 1
                else:
                    final_signal = 0
                    reasoning = f"buy_lowquality (score:{signal_quality:.2f})"
                    self.stats['hold_signals'] += 1
            
            elif sell_score >= sell_thresh and sell_score > buy_score:
                if signal_quality >= min_quality:
                    final_signal = -1
                    reasoning = f"sell_consensus_{sell_agreement}_strategies"
                    self.stats['sell_signals'] += 1
                    self.stats['signals_generated'] += 1
                    if sell_agreement == 1:
                        self.stats['single_strategy_signals'] += 1
                    else:
                        self.stats['consensus_signals'] += 1
                else:
                    final_signal = 0
                    reasoning = f"sell_lowquality (score:{signal_quality:.2f})"
                    self.stats['hold_signals'] += 1
            
            else:
                final_signal = 0
                reasoning = f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                self.stats['hold_signals'] += 1
            
            # ============================================================
            # STEP 5: Build detailed response
            # ============================================================
            details = {
                'timestamp': timestamp,
                'regime': regime_name,
                'regime_confidence': regime_conf,
                'final_signal': final_signal,
                'reasoning': reasoning,
                'signal_quality': signal_quality,
                'buy_score': buy_score,
                'buy_explanation': buy_explanation,
                'sell_score': sell_score,
                'sell_explanation': sell_explanation,
                'buy_agreement_count': buy_agreement,
                'sell_agreement_count': sell_agreement,
                'mr_signal': mr_signal,
                'mr_confidence': mr_conf,
                'tf_signal': tf_signal,
                'tf_confidence': tf_conf,
                'ema_regime_signal': ema_signal,
                'ema_regime_confidence': ema_conf,
            }
            
            # Log if decision made
            if final_signal != 0:
                signal_emoji = "🟢 BUY" if final_signal == 1 else "🔴 SELL"
                logger.info(f"{signal_emoji} | Quality: {signal_quality:.3f} | {reasoning}")
                logger.info(f"  Regime: {regime_name} | Buy: {buy_score:.3f} | Sell: {sell_score:.3f}")
            
            return final_signal, details
        
        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {'error': str(e), 'timestamp': timestamp}

    def get_statistics(self) -> Dict:
        """Return aggregator statistics"""
        total = max(self.stats['total_evaluations'], 1)
        
        return {
            **self.stats,
            'signal_rate': (self.stats['signals_generated'] / total) * 100,
            'buy_rate': (self.stats['buy_signals'] / total) * 100,
            'sell_rate': (self.stats['sell_signals'] / total) * 100,
            'hold_rate': (self.stats['hold_signals'] / total) * 100,
            'bull_regime_pct': (self.stats['bull_regime_count'] / total) * 100,
            'bear_regime_pct': (self.stats['bear_regime_count'] / total) * 100,
            'consensus_rate': (self.stats['consensus_signals'] / max(self.stats['signals_generated'], 1)) * 100,
        }