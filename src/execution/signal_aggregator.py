"""
Fixed Signal Aggregator - Strategies Contribute Even on HOLD
Key fix: HOLD signals with high confidence still provide directional context
"""

import pandas as pd
import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
    Fixed Multi-Strategy Aggregator
    
    Key Changes:
    1. HOLD signals contribute based on regime/trend context
    2. Lower thresholds for decision-making
    3. Confidence modulates strength, not participation
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

        # Strategy weights (based on CV accuracy)
        if self.asset_type == "BTC":
            self.weights = {
                "mean_reversion": 0.50,
                "trend_following": 0.50,
            }
        else:  # GOLD
            self.weights = {
                "mean_reversion": 0.50,
                "trend_following": 0.50,
            }

        # FIXED CONFIGURATION
        self.config = {
            # LOWER thresholds - easier to trigger signals
            "buy_threshold": 0.25,   # Was 0.35
            "sell_threshold": 0.25,  # Was 0.35
            
            # Agreement bonuses (unchanged)
            "two_strategy_bonus": 0.15,
            "three_strategy_bonus": 0.25,
            
            # Regime modifiers (reduced impact)
            "bull_buy_boost": 0.03,
            "bull_sell_penalty": 0.03,
            "bear_sell_boost": 0.03,
            "bear_buy_penalty": 0.03,
            
            # LOWER confidence floor - use more signals
            "min_confidence_to_use": 0.10,  # Was 0.15
            
            # LOWER quality gate
            "min_signal_quality": 0.20,  # Was 0.25
        }

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
        }

        self.previous_regime = None
        self._log_initialization()

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯  Fixed PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("🔧 KEY FIXES:")
        logger.info("   ✓ HOLD signals now contribute directional context")
        logger.info("   ✓ Lower thresholds (0.25 instead of 0.35)")
        logger.info("   ✓ Strategies always participate in scoring")
        logger.info("")
        logger.info("📊 STRATEGY ROLES:")
        logger.info("   Mean Reversion:   DECISION MAKER (weight: 0.50)")
        logger.info("   Trend Following:  DECISION MAKER (weight: 0.50)")
        logger.info("   EMA 50/200:       REGIME DETECTOR ONLY")
        logger.info("")
        logger.info("🎚️ THRESHOLDS:")
        logger.info(f"   Buy Signal:       score >= {self.config['buy_threshold']:.2f}")
        logger.info(f"   Sell Signal:      score >= {self.config['sell_threshold']:.2f}")
        logger.info(f"   Min Confidence:   {self.config['min_confidence_to_use']:.2f}")
        logger.info(f"   Min Quality:      {self.config['min_signal_quality']:.2f}")
        logger.info("=" * 80)
        logger.info("")

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect market regime using EMA crossover"""
        ema_signal, ema_conf = self.s_ema.generate_signal(df)
        is_bull = ema_signal == 1

        if self.previous_regime is not None and self.previous_regime != is_bull:
            self.stats["regime_changes"] += 1
            regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
            logger.info(f"⚡ REGIME CHANGE → {regime_name} (conf: {ema_conf:.3f})")

        self.previous_regime = is_bull

        if is_bull:
            self.stats["bull_regime_count"] += 1
        else:
            self.stats["bear_regime_count"] += 1

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
        FIXED: Calculate score with HOLD signals contributing context
        
        Logic:
        1. Direct agreement: Full contribution
        2. HOLD signal: Partial contribution based on confidence
        3. Opposition: Negative contribution (penalty)
        """
        components = []
        total_score = 0.0
        agreement_count = 0
        min_conf = self.config["min_confidence_to_use"]

        # ============================================================
        # MEAN REVERSION CONTRIBUTION
        # ============================================================
        if mr_signal == target_signal:
            # DIRECT AGREEMENT - Full contribution
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_agree:{contribution:.3f}")
            agreement_count += 1
            
        elif mr_signal == 0:
            # HOLD - Partial contribution based on confidence
            # High confidence HOLD = some directional lean
            effective_conf = max(mr_conf, min_conf)
            contribution = (effective_conf * 0.3) * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_hold:{contribution:.3f}")
            
        else:
            # OPPOSITION - Penalty
            effective_conf = max(mr_conf, min_conf)
            penalty = (effective_conf * 0.5) * self.weights["mean_reversion"]
            total_score -= penalty
            components.append(f"MR_oppose:-{penalty:.3f}")

        # ============================================================
        # TREND FOLLOWING CONTRIBUTION
        # ============================================================
        if tf_signal == target_signal:
            # DIRECT AGREEMENT - Full contribution
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_agree:{contribution:.3f}")
            agreement_count += 1
            
        elif tf_signal == 0:
            # HOLD - Partial contribution
            effective_conf = max(tf_conf, min_conf)
            contribution = (effective_conf * 0.3) * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_hold:{contribution:.3f}")
            
        else:
            # OPPOSITION - Penalty
            effective_conf = max(tf_conf, min_conf)
            penalty = (effective_conf * 0.5) * self.weights["trend_following"]
            total_score -= penalty
            components.append(f"TF_oppose:-{penalty:.3f}")

        # ============================================================
        # BUILD EXPLANATION
        # ============================================================
        explanation = " + ".join(components) if components else "no_agreement"

        # ============================================================
        # AGREEMENT BONUS
        # ============================================================
        if agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus({bonus:.2f})"

        # ============================================================
        # REGIME CONTEXT (mild adjustment)
        # ============================================================
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

        # Ensure non-negative
        total_score = max(0.0, total_score)

        return total_score, explanation, agreement_count

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """Main aggregation logic with FIXED scoring"""
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
            
            
            # STEP 3: Calculate BUY and SELL scores
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

            logger.info(f"[SCORES] BUY: {buy_score:.3f} | SELL: {sell_score:.3f}")
            logger.info(f"[BUY CALC] {buy_explanation}")
            logger.info(f"[SELL CALC] {sell_explanation}")

            # STEP 4: Make decision
            buy_thresh = self.config["buy_threshold"]
            sell_thresh = self.config["sell_threshold"]
            min_quality = self.config["min_signal_quality"]

            signal_quality = max(buy_score, sell_score)

            if buy_score >= buy_thresh and buy_score > sell_score:
                if signal_quality >= min_quality:
                    final_signal = 1
                    reasoning = f"BUY (score:{buy_score:.2f}, quality:{signal_quality:.2f})"
                    self.stats["buy_signals"] += 1
                    self.stats["signals_generated"] += 1
                    logger.info(f"🟢 {reasoning}")
                else:
                    final_signal = 0
                    reasoning = f"hold_lowquality ({signal_quality:.2f})"
                    self.stats["hold_signals"] += 1

            elif sell_score >= sell_thresh and sell_score > buy_score:
                if signal_quality >= min_quality:
                    final_signal = -1
                    reasoning = f"SELL (score:{sell_score:.2f}, quality:{signal_quality:.2f})"
                    self.stats["sell_signals"] += 1
                    self.stats["signals_generated"] += 1
                    logger.info(f"🔴 {reasoning}")
                else:
                    final_signal = 0
                    reasoning = f"hold_lowquality ({signal_quality:.2f})"
                    self.stats["hold_signals"] += 1

            else:
                final_signal = 0
                reasoning = f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                self.stats["hold_signals"] += 1

            # STEP 5: Build response
            details = {
                "timestamp": timestamp,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "final_signal": final_signal,
                "reasoning": reasoning,
                "signal_quality": signal_quality,
                "buy_score": buy_score,
                "buy_explanation": buy_explanation,
                "sell_score": sell_score,
                "sell_explanation": sell_explanation,
                "buy_agreement_count": buy_agreement,
                "sell_agreement_count": sell_agreement,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
                "ema_regime_signal": ema_signal,
                "ema_regime_confidence": ema_conf,
            }

            return final_signal, details

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {"error": str(e), "timestamp": timestamp}

    def get_statistics(self) -> Dict:
        """Return aggregator statistics"""
        total = max(self.stats["total_evaluations"], 1)
        return {
            **self.stats,
            "signal_rate": (self.stats["signals_generated"] / total) * 100,
            "buy_rate": (self.stats["buy_signals"] / total) * 100,
            "sell_rate": (self.stats["sell_signals"] / total) * 100,
            "hold_rate": (self.stats["hold_signals"] / total) * 100,
        }