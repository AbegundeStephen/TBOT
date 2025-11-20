"""
FIXED BullMarketFilteredAggregator - Resolves Signal Starvation
Key changes:
1. Lower thresholds (0.55 -> 0.40)
2. Single strong strategy can trigger signals
3. Better handling of low-confidence Trend Following
4. Diagnostic logging
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from src.strategies.base_strategy import BaseStrategy
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BullMarketFilteredAggregator:
    """
    FIXED WEIGHTED VOTING VERSION
    - Lower thresholds for more signals
    - Single strong strategy can trigger action
    - Better TF handling (since it has poor accuracy)
    """
    
    def __init__(
        self,
        mean_reversion_strategy: BaseStrategy,
        trend_following_strategy: BaseStrategy,
        ema_strategy: BaseStrategy,
        confidence_config: Optional[Dict] = None,
        asset_name: str = "UNKNOWN"
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_name = asset_name
        
        default_config = {
            # === WEIGHTED VOTING THRESHOLDS (LOWERED) ===
            "buy_score_threshold": 0.40,      # Was 0.55 - NOW MORE LENIENT
            "sell_score_threshold": 0.40,     # Was 0.55 - NOW MORE LENIENT
            "perfect_agreement_bonus": 0.15,  # Increased from 0.10
            
            # === STRATEGY WEIGHTS (ADJUSTED) ===
            "mean_reversion_weight": 1.2,     # Increased (83% accuracy!)
            "trend_following_weight": 0.5,    # Reduced (48% accuracy - basically random)
            
            # === SINGLE STRATEGY MODE (NEW) ===
            "allow_single_mr_signal": True,   # MR alone can trigger if strong
            "single_mr_threshold": 0.60,      # MR confidence needed
            "allow_single_tf_signal": False,  # TF alone cannot trigger (low accuracy)
            
            # === CONFIDENCE FILTERING (RELAXED) ===
            "min_confidence": 0.35,            # Was 0.40
            "signal_quality_threshold": 0.40,  # Was 0.45
            "high_confidence_threshold": 0.65,
            "confidence_normalization": True,
            
            # === BULL MARKET FILTER ===
            "enable_bull_filter": True,
            "block_sells_in_bull": True,
            "boost_buys_in_bull": 0.10,        # Increased from 0.08
            
            # === REGIME TRACKING ===
            "regime_confirmation_bars": 2,
            
            # === DIAGNOSTICS ===
            "verbose_logging": True,           # NEW: Detailed logs
        }
        
        self.config = confidence_config or {}
        for key, value in default_config.items():
            self.config.setdefault(key, value)
        
        # State tracking
        self.previous_regime = None
        self.regime_change_count = 0
        self.signal_history = []
        self.max_history = 10
        
        # Diagnostic counters
        self.total_evaluations = 0
        self.signals_generated = 0
        self.holds_generated = 0
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log aggregator initialization details"""
        logger.info("=" * 80)
        logger.info(f"BullMarketFilteredAggregator (FIXED) for {self.asset_name}")
        logger.info("=" * 80)
        logger.info(f"  Regime Filter: EMA Strategy (50/200 SMA Golden Cross)")
        logger.info(f"  Primary (High Accuracy): Mean Reversion (83% accuracy)")
        logger.info(f"  Secondary (Low Weight): Trend Following (49% accuracy)")
        logger.info("")
        logger.info("FIXED CONFIGURATION:")
        logger.info(f"  === Scoring Thresholds (LOWERED) ===")
        logger.info(f"    Buy Score Threshold: {self.config['buy_score_threshold']:.2f} ⬇")
        logger.info(f"    Sell Score Threshold: {self.config['sell_score_threshold']:.2f} ⬇")
        logger.info(f"    Perfect Agreement Bonus: +{self.config['perfect_agreement_bonus']:.2f} ⬆")
        logger.info(f"  === Strategy Weights (ADJUSTED) ===")
        logger.info(f"    Mean Reversion Weight: {self.config['mean_reversion_weight']:.2f} ⬆")
        logger.info(f"    Trend Following Weight: {self.config['trend_following_weight']:.2f} ⬇")
        logger.info(f"  === Single Strategy Mode (NEW) ===")
        logger.info(f"    MR Can Trigger Alone: {self.config['allow_single_mr_signal']}")
        logger.info(f"    MR Solo Threshold: {self.config['single_mr_threshold']:.2f}")
        logger.info(f"    TF Can Trigger Alone: {self.config['allow_single_tf_signal']}")
        logger.info(f"  === Bull Market Behavior ===")
        logger.info(f"    Block Sells in Bull: {self.config['block_sells_in_bull']}")
        logger.info(f"    Buy Boost in Bull: +{self.config['boost_buys_in_bull']:.2f}")
        logger.info("=" * 80)
    
    def _normalize_confidence(self, confidence: float) -> float:
        """Normalize confidence scores to 0-1 range"""
        if not self.config['confidence_normalization']:
            return confidence
        
        if 0 <= confidence <= 1:
            return confidence
        
        logger.warning(
            f"[{self.asset_name}] Confidence out of bounds: {confidence:.3f}, clipping"
        )
        return np.clip(confidence, 0, 1)
    
    def _calculate_weighted_buy_score(
        self,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        is_bull: bool
    ) -> Tuple[float, str]:
        """
        Calculate weighted score for BUY signals with diagnostics
        Returns: (score, explanation)
        """
        
        # Only consider positive signals
        mr_buy_score = mr_conf if mr_signal == 1 else 0.0
        tf_buy_score = tf_conf if tf_signal == 1 else 0.0
        
        explanation = []
        
        # Weighted average
        total_weight = self.config['mean_reversion_weight'] + self.config['trend_following_weight']
        weighted_score = (
            (mr_buy_score * self.config['mean_reversion_weight']) +
            (tf_buy_score * self.config['trend_following_weight'])
        ) / total_weight
        
        explanation.append(f"Base: {weighted_score:.3f}")
        explanation.append(f"(MR:{mr_buy_score:.2f}*{self.config['mean_reversion_weight']:.1f}")
        explanation.append(f"TF:{tf_buy_score:.2f}*{self.config['trend_following_weight']:.1f})")
        
        # Bonus if both soldiers agree on BUY
        if mr_signal == 1 and tf_signal == 1:
            old_score = weighted_score
            weighted_score = min(1.0, weighted_score + self.config['perfect_agreement_bonus'])
            explanation.append(f"→ +{self.config['perfect_agreement_bonus']:.2f} (both agree)")
            explanation.append(f"{old_score:.3f}→{weighted_score:.3f}")
        
        # Slight boost in bull market
        if is_bull and self.config['enable_bull_filter']:
            old_score = weighted_score
            weighted_score = min(1.0, weighted_score + self.config['boost_buys_in_bull'])
            explanation.append(f"→ +{self.config['boost_buys_in_bull']:.2f} (bull)")
            explanation.append(f"{old_score:.3f}→{weighted_score:.3f}")
        
        return weighted_score, " ".join(explanation)
    
    def _calculate_weighted_sell_score(
        self,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        is_bull: bool
    ) -> Tuple[float, str]:
        """
        Calculate weighted score for SELL signals with diagnostics
        Returns: (score, explanation)
        """
        
        # Only consider negative signals
        mr_sell_score = mr_conf if mr_signal == -1 else 0.0
        tf_sell_score = tf_conf if tf_signal == -1 else 0.0
        
        explanation = []
        
        # Weighted average
        total_weight = self.config['mean_reversion_weight'] + self.config['trend_following_weight']
        weighted_score = (
            (mr_sell_score * self.config['mean_reversion_weight']) +
            (tf_sell_score * self.config['trend_following_weight'])
        ) / total_weight
        
        explanation.append(f"Base: {weighted_score:.3f}")
        explanation.append(f"(MR:{mr_sell_score:.2f}*{self.config['mean_reversion_weight']:.1f}")
        explanation.append(f"TF:{tf_sell_score:.2f}*{self.config['trend_following_weight']:.1f})")
        
        # Bonus if both soldiers agree on SELL
        if mr_signal == -1 and tf_signal == -1:
            old_score = weighted_score
            weighted_score = min(1.0, weighted_score + self.config['perfect_agreement_bonus'])
            explanation.append(f"→ +{self.config['perfect_agreement_bonus']:.2f} (both agree)")
            explanation.append(f"{old_score:.3f}→{weighted_score:.3f}")
        
        # Penalty in bull market
        if is_bull and self.config['enable_bull_filter'] and self.config['block_sells_in_bull']:
            old_score = weighted_score
            weighted_score = max(0.0, weighted_score - 0.20)
            explanation.append(f"→ -0.20 (bull penalty)")
            explanation.append(f"{old_score:.3f}→{weighted_score:.3f}")
        
        return weighted_score, " ".join(explanation)
    
    def _check_single_strategy_signal(
        self,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        is_bull: bool
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        NEW: Check if a single strong strategy can trigger a signal
        Returns: (signal, reasoning) or (None, None)
        """
        
        # Mean Reversion solo signal (it has 83% accuracy!)
        if self.config['allow_single_mr_signal']:
            mr_threshold = self.config['single_mr_threshold']
            
            # Strong MR BUY
            if mr_signal == 1 and mr_conf >= mr_threshold:
                # In bull market, allow it
                if is_bull or not self.config['enable_bull_filter']:
                    return 1, f"single_mr_buy_strong (conf={mr_conf:.2f} >= {mr_threshold:.2f})"
                
            # Strong MR SELL
            if mr_signal == -1 and mr_conf >= mr_threshold:
                # In bear/neutral, allow it
                # In bull, only if VERY high confidence
                if not is_bull:
                    return -1, f"single_mr_sell_strong (conf={mr_conf:.2f} >= {mr_threshold:.2f})"
                elif mr_conf >= 0.75:  # Very high bar in bull market
                    return -1, f"single_mr_sell_very_strong_bull (conf={mr_conf:.2f})"
        
        # Trend Following solo signal (disabled by default due to low accuracy)
        if self.config['allow_single_tf_signal']:
            tf_threshold = self.config.get('single_tf_threshold', 0.70)  # Higher bar
            
            if tf_signal == 1 and tf_conf >= tf_threshold:
                return 1, f"single_tf_buy_strong (conf={tf_conf:.2f})"
            if tf_signal == -1 and tf_conf >= tf_threshold:
                if not is_bull:
                    return -1, f"single_tf_sell_strong (conf={tf_conf:.2f})"
        
        return None, None
    
    def _detect_regime_change(self, current_regime: bool) -> Tuple[bool, str]:
        """Detect if market regime has changed (bull <-> bear)"""
        if self.previous_regime is None:
            self.previous_regime = current_regime
            return False, "regime_initialization"
        
        if current_regime != self.previous_regime:
            self.regime_change_count += 1
            
            if self.regime_change_count >= self.config['regime_confirmation_bars']:
                self.previous_regime = current_regime
                direction = "BULL" if current_regime else "BEAR"
                logger.warning(
                    f"[{self.asset_name}] ⚠️  REGIME CHANGE CONFIRMED → {direction}"
                )
                return True, f"regime_changed_to_{direction}"
            else:
                return False, f"regime_change_pending_{self.regime_change_count}"
        else:
            self.regime_change_count = 0
            return True, "regime_stable"
    
    def _track_signal_history(self, signal: int, confidence: float, reasoning: str):
        """Track signal history for analysis"""
        entry = {
            "timestamp": datetime.now(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning
        }
        self.signal_history.append(entry)
        
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
    
    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main signal aggregation with FIXED WEIGHTED VOTING
        """
        self.total_evaluations += 1
        
        try:
            # Step 1: Get individual signals with normalization
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            ema_conf = self._normalize_confidence(ema_conf)
            is_bull_market = ema_signal == 1
            
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            mr_conf = self._normalize_confidence(mr_conf)
            
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            tf_conf = self._normalize_confidence(tf_conf)
            
            # Step 2: Detect regime changes
            regime_changed, regime_status = self._detect_regime_change(is_bull_market)
            
            # Step 3: Check single strategy signals first (NEW)
            single_signal, single_reasoning = self._check_single_strategy_signal(
                mr_signal, mr_conf, tf_signal, tf_conf, is_bull_market
            )
            
            if single_signal is not None:
                # Single strategy strong enough!
                if single_signal != 0:
                    self.signals_generated += 1
                else:
                    self.holds_generated += 1
                
                signal_quality = mr_conf if 'mr' in single_reasoning else tf_conf
                self._track_signal_history(single_signal, signal_quality, single_reasoning)
                
                signal_details = self._build_signal_details(
                    mr_signal, mr_conf, tf_signal, tf_conf,
                    ema_signal, ema_conf, is_bull_market,
                    regime_changed, regime_status,
                    single_signal, single_reasoning, signal_quality,
                    0.0, 0.0, f"Single strategy: {single_reasoning}",
                    df
                )
                
                self._log_decision_verbose(
                    mr_signal, mr_conf, tf_signal, tf_conf,
                    is_bull_market, single_signal, single_reasoning,
                    signal_quality, 0.0, 0.0, regime_changed
                )
                
                return single_signal, signal_details
            
            # Step 4: Apply Weighted Voting Logic
            final_signal, reasoning, buy_score, sell_score, decision_path = self._apply_weighted_voting_logic(
                mr_signal, mr_conf,
                tf_signal, tf_conf,
                is_bull_market
            )
            
            if final_signal != 0:
                self.signals_generated += 1
            else:
                self.holds_generated += 1
            
            # Step 5: Signal quality is the highest score
            signal_quality = max(buy_score, sell_score)
            
            # Step 6: Track history
            self._track_signal_history(final_signal, signal_quality, reasoning)
            
            # Step 7: Package Details
            signal_details = self._build_signal_details(
                mr_signal, mr_conf, tf_signal, tf_conf,
                ema_signal, ema_conf, is_bull_market,
                regime_changed, regime_status,
                final_signal, reasoning, signal_quality,
                buy_score, sell_score, decision_path,
                df
            )
            
            self._log_decision_verbose(
                mr_signal, mr_conf, tf_signal, tf_conf,
                is_bull_market, final_signal, reasoning,
                signal_quality, buy_score, sell_score, regime_changed
            )
            
            return final_signal, signal_details
        
        except Exception as e:
            logger.error(f"[{self.asset_name}] Error in get_aggregated_signal: {e}", exc_info=True)
            return 0, {
                "final_signal": 0,
                "reasoning": "error_in_aggregation",
                "error": str(e)
            }
    
    def _build_signal_details(
        self, mr_signal, mr_conf, tf_signal, tf_conf,
        ema_signal, ema_conf, is_bull_market,
        regime_changed, regime_status,
        final_signal, reasoning, signal_quality,
        buy_score, sell_score, decision_path, df
    ) -> Dict:
        """Build signal details dictionary"""
        return {
            # Market Regime
            "regime": "🚀 BULL" if is_bull_market else "⚖️  NEUTRAL/BEAR",
            "is_bull_market": is_bull_market,
            "regime_changed": regime_changed,
            "regime_status": regime_status,
            
            # Individual Strategy Signals
            "mean_reversion_signal": mr_signal,
            "mean_reversion_confidence": mr_conf,
            "trend_following_signal": tf_signal,
            "trend_following_confidence": tf_conf,
            "ema_signal": ema_signal,
            "ema_confidence": ema_conf,
            
            # Scoring Details
            "buy_score": buy_score,
            "sell_score": sell_score,
            "buy_threshold": self.config['buy_score_threshold'],
            "sell_threshold": self.config['sell_score_threshold'],
            
            # Final Decision
            "final_signal": final_signal,
            "reasoning": reasoning,
            "decision_path": decision_path,
            "signal_quality": signal_quality,
            
            # Quality Metrics
            "high_confidence": signal_quality >= self.config['high_confidence_threshold'],
            
            # Diagnostics
            "total_evaluations": self.total_evaluations,
            "signals_generated": self.signals_generated,
            "holds_generated": self.holds_generated,
            "signal_rate": f"{(self.signals_generated/self.total_evaluations*100):.1f}%" if self.total_evaluations > 0 else "0%",
            
            # Metadata
            "aggregator_mode": "sniper_weighted_voting_fixed",
            "asset": self.asset_name,
            "timestamp": (
                df.index[-1].isoformat()
                if hasattr(df.index[-1], "isoformat")
                else str(df.index[-1])
            ),
        }
    
    def _apply_weighted_voting_logic(
        self,
        mr_sig: int, mr_conf: float,
        tf_sig: int, tf_conf: float,
        is_bull: bool
    ) -> Tuple[int, str, float, float, str]:
        """
        FIXED VOTING LOGIC with weighted scores and diagnostics
        
        Returns: (signal, reasoning, buy_score, sell_score, decision_path)
        """
        
        decision_path = []
        
        # Calculate buy and sell scores with explanations
        buy_score, buy_explanation = self._calculate_weighted_buy_score(
            mr_sig, mr_conf, tf_sig, tf_conf, is_bull
        )
        sell_score, sell_explanation = self._calculate_weighted_sell_score(
            mr_sig, mr_conf, tf_sig, tf_conf, is_bull
        )
        
        buy_threshold = self.config['buy_score_threshold']
        sell_threshold = self.config['sell_score_threshold']
        
        decision_path.append(f"BUY: {buy_explanation}")
        decision_path.append(f"SELL: {sell_explanation}")
        decision_path.append(f"MR:{mr_sig}({mr_conf:.2f}) TF:{tf_sig}({tf_conf:.2f})")
        decision_path.append(f"Regime: {'BULL' if is_bull else 'BEAR'}")
        
        # === BUY DECISION ===
        if buy_score >= buy_threshold:
            decision_path.append(f"✓ BUY ({buy_score:.3f} >= {buy_threshold:.3f})")
            
            if mr_sig == 1 and tf_sig == 1:
                reasoning = "buy_both_soldiers_agree_weighted"
            elif mr_sig == 1:
                reasoning = "buy_mr_primary_weighted"
            elif tf_sig == 1:
                reasoning = "buy_tf_primary_weighted"
            else:
                reasoning = "buy_weighted_score_passed"
            
            return 1, reasoning, buy_score, sell_score, " | ".join(decision_path)
        
        # === SELL DECISION ===
        elif sell_score >= sell_threshold:
            decision_path.append(f"✓ SELL ({sell_score:.3f} >= {sell_threshold:.3f})")
            
            if is_bull and self.config['enable_bull_filter'] and self.config['block_sells_in_bull']:
                decision_path.append("→ IN_BULL (sell allowed via high score)")
                reasoning = "sell_high_confidence_bull_market"
            else:
                if mr_sig == -1 and tf_sig == -1:
                    reasoning = "sell_both_soldiers_agree_weighted"
                elif mr_sig == -1:
                    reasoning = "sell_mr_primary_weighted"
                elif tf_sig == -1:
                    reasoning = "sell_tf_primary_weighted"
                else:
                    reasoning = "sell_weighted_score_passed"
            
            return -1, reasoning, buy_score, sell_score, " | ".join(decision_path)
        
        # === NO SIGNAL ===
        else:
            decision_path.append(f"✗ HOLD (BUY:{buy_score:.3f}<{buy_threshold:.3f}, SELL:{sell_score:.3f}<{sell_threshold:.3f})")
            reasoning = "no_action_scores_below_threshold"
            return 0, reasoning, buy_score, sell_score, " | ".join(decision_path)
    
    def _log_decision_verbose(
        self,
        mr_sig: int, mr_conf: float,
        tf_sig: int, tf_conf: float,
        is_bull: bool,
        final_signal: int,
        reasoning: str,
        signal_quality: float,
        buy_score: float,
        sell_score: float,
        regime_changed: bool
    ):
        """Enhanced verbose logging"""
        if not self.config.get('verbose_logging', True):
            return
        
        mr_name = "BUY" if mr_sig == 1 else "SELL" if mr_sig == -1 else "HOLD"
        tf_name = "BUY" if tf_sig == 1 else "SELL" if tf_sig == -1 else "HOLD"
        regime = "🚀 BULL" if is_bull else "⚖️  BEAR"
        final_name = "→ BUY" if final_signal == 1 else "→ SELL" if final_signal == -1 else "→ HOLD"
        regime_mark = "⚡" if regime_changed else " "
        quality_mark = "★" if signal_quality >= 0.65 else "●" if signal_quality >= 0.50 else "○"
        
        signal_rate = (self.signals_generated / self.total_evaluations * 100) if self.total_evaluations > 0 else 0
        
        logger.info(
            f"[{self.asset_name}] {regime_mark} {regime} | "
            f"MR: {mr_name}({mr_conf:.2f}) | TF: {tf_name}({tf_conf:.2f}) | "
            f"Score B/S: {buy_score:.2f}/{sell_score:.2f} | "
            f"{final_name} | {quality_mark}{signal_quality:.2f} | "
            f"Rate: {signal_rate:.1f}% | {reasoning}"
        )
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information"""
        signal_rate = (self.signals_generated / self.total_evaluations * 100) if self.total_evaluations > 0 else 0
        
        return {
            "total_evaluations": self.total_evaluations,
            "signals_generated": self.signals_generated,
            "holds_generated": self.holds_generated,
            "signal_rate_percent": f"{signal_rate:.1f}%",
            "config": {
                "buy_threshold": self.config['buy_score_threshold'],
                "sell_threshold": self.config['sell_score_threshold'],
                "mr_weight": self.config['mean_reversion_weight'],
                "tf_weight": self.config['trend_following_weight'],
                "single_mr_allowed": self.config['allow_single_mr_signal'],
                "single_mr_threshold": self.config.get('single_mr_threshold', 'N/A')
            }
        }
    
    # ... (rest of the methods remain the same: get_regime_status, get_signal_statistics, 
    #      load_models, get_signal_breakdown)
    
    def get_regime_status(self, df: pd.DataFrame) -> Dict:
        """Get current market regime status"""
        try:
            ema_signal, ema_confidence = self.s_ema.generate_signal(df)
            ema_confidence = self._normalize_confidence(ema_confidence)
            is_bull = ema_signal == 1
            
            return {
                "regime": "BULL_MARKET" if is_bull else "NEUTRAL_OR_BEAR",
                "golden_cross_active": is_bull,
                "ema_confidence": ema_confidence,
                "override_active": is_bull and self.config['block_sells_in_bull'],
                "sells_penalized": is_bull,
                "buys_boosted": is_bull,
                "previous_regime": "BULL" if self.previous_regime else "BEAR" if self.previous_regime is False else "UNKNOWN",
                "description": (
                    f"Golden Cross ACTIVE - Sells penalized, buys boosted (EMA: {ema_confidence:.2f})"
                    if is_bull else
                    f"No Golden Cross - Both longs and shorts allowed (EMA: {ema_confidence:.2f})"
                )
            }
        except Exception as e:
            logger.error(f"[{self.asset_name}] Error getting regime status: {e}")
            return {"regime": "UNKNOWN", "error": str(e)}
    
    def load_models(
        self,
        mean_reversion_path: str,
        trend_following_path: str,
        ema_path: str
    ) -> bool:
        """Load all three strategy models"""
        logger.info(f"[{self.asset_name}] Loading strategy models...")
        
        mr_loaded = self.s_mean_reversion.load_model(mean_reversion_path)
        tf_loaded = self.s_trend_following.load_model(trend_following_path)
        ema_loaded = self.s_ema.load_model(ema_path)
        
        if mr_loaded and tf_loaded and ema_loaded:
            logger.info(f"[{self.asset_name}] ✅ All three models loaded")
            return True
        else:
            logger.error(f"[{self.asset_name}] ❌ Failed to load one or more models")
            return False