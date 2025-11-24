"""
Signal Aggregator with Bull/Bear Regime Detection and Logging
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BullMarketFilteredAggregator:
    """
    Multi-Layer Signal Processing with Regime Detection:
    
    Architecture:
    - EMA 50/200: Regime Filter (Golden Cross = Bull, Death Cross = Bear)
    - Mean Reversion: Tactical reversals (RSI + Bollinger Bands)
    - Trend Following: Momentum entries (MA + ADX)
    - Aggregator: Weighted voting with regime-aware filtering
    
    VERSION - Weights match actual accuracy:
    - MR: 72-75% accuracy → Weight 1.1
    - TF: 72% accuracy → Weight 1.1
    - EMA: 72% accuracy (regime detection) → Weight 1.1
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        confidence_config: Optional[Dict] = None,
        asset_name: str = "UNKNOWN",
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_name = asset_name

        default_config = {
            # WEIGHTS: Match actual performance
            "mean_reversion_weight": 1.1,
            "trend_following_weight": 1.1,
            "ema_weight": 1.1,
            # Thresholds
            "buy_score_threshold": 0.35,
            "sell_score_threshold": 0.40,
            "perfect_agreement_bonus": 0.15,
            # Single strategy logic
            "allow_single_mr_signal": False,
            "allow_single_tf_signal": True,
            "allow_single_ema_signal": True,
            "single_mr_threshold": 0.65,
            "single_tf_threshold": 0.60,
            "single_ema_threshold": 0.60,
            # Confidence filtering
            "min_confidence": 0.35,
            "signal_quality_threshold": 0.40,
            "high_confidence_threshold": 0.65,
            "confidence_normalization": True,
            # Bull market filtering to allow exits
            "enable_bull_filter": True,
            "block_sells_in_bull": False,
            "boost_buys_in_bull": 0.15,
            "sell_penalty_in_bull": 0.20,
            "regime_confirmation_bars": 2,
            "regime_cooldown_hours": 6,
            "verbose_logging": True,
        }

        self.config = confidence_config or {}
        for key, value in default_config.items():
            self.config.setdefault(key, value)

        # State tracking for regime
        self.previous_regime = None
        self.regime_change_count = 0
        self.last_regime_change_time = None
        self.signal_history = []
        self.max_history = 10

        # Regime statistics
        self.bull_count = 0
        self.bear_count = 0
        self.total_evaluations = 0
        self.signals_generated = 0
        self.holds_generated = 0

        self._log_initialization()

    def _log_initialization(self):
        """Log corrected configuration"""
        logger.info("=" * 80)
        logger.info(f"🔧 BullMarketFilteredAggregator for {self.asset_name}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("ARCHITECTURE:")
        logger.info("  Layer 1: EMA 50/200 → Regime Filter (Bull/Bear Detection)")
        logger.info("  Layer 2: Mean Reversion → Tactical Reversals (RSI + BB)")
        logger.info("  Layer 3: Trend Following → Momentum Entries (MA + ADX)")
        logger.info("  Layer 4: Aggregator → Weighted Voting with Regime Context")
        logger.info("")
        logger.info("STRATEGY WEIGHTS (aligned with actual accuracy):")
        logger.info(
            f"  📊 Mean Reversion:   {self.config['mean_reversion_weight']:.1f} (72-75% acc)"
        )
        logger.info(
            f"  📊 Trend Following:  {self.config['trend_following_weight']:.1f} (72% acc)"
        )
        logger.info(
            f"  📊 EMA Strategy:     {self.config['ema_weight']:.1f} (72% acc - Regime Filter)"
        )
        logger.info("")
        logger.info("SINGLE STRATEGY PERMISSIONS (prefer high-confidence signals):")
        logger.info(
            f"  MR alone:  {'✅ YES' if self.config['allow_single_mr_signal'] else '❌ NO'} (threshold: {self.config.get('single_mr_threshold', 0):.2f})"
        )
        logger.info(
            f"  TF alone:  {'✅ YES' if self.config['allow_single_tf_signal'] else '❌ NO'} (threshold: {self.config['single_tf_threshold']:.2f})"
        )
        logger.info(
            f"  EMA alone: {'✅ YES' if self.config['allow_single_ema_signal'] else '❌ NO'} (threshold: {self.config['single_ema_threshold']:.2f})"
        )
        logger.info("")
        logger.info("BULL MARKET BEHAVIOR (regime-aware filtering):")
        logger.info(
            f"  Block sells: {'❌ NO' if not self.config['block_sells_in_bull'] else '✅ YES'} (allow exits for risk mgmt)"
        )
        logger.info(f"  Sell penalty: -{self.config['sell_penalty_in_bull']:.2f}")
        logger.info(f"  Buy boost: +{self.config['boost_buys_in_bull']:.2f}")
        logger.info("=" * 80)
        logger.info("")

    def _normalize_confidence(self, confidence: float) -> float:
        """Normalize confidence to 0-1"""
        if not self.config["confidence_normalization"]:
            return confidence
        return np.clip(confidence, 0, 1)

    def _log_regime_change(self, is_bull: bool, ema_signal: int, ema_conf: float, timestamp: str):
        """Log when market regime changes"""
        regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
        
        if self.previous_regime != is_bull:
            self.regime_change_count += 1
            self.last_regime_change_time = datetime.now()
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"⚡ REGIME CHANGE #{self.regime_change_count}")
            logger.info("=" * 80)
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"New Regime: {regime_name}")
            
            if is_bull:
                logger.info("📈 Golden Cross Detected (EMA50 > EMA200)")
                logger.info("   → Favoring BUY signals")
                logger.info("   → Penalizing SELL signals")
                logger.info("   → Expect trend-following opportunities")
            else:
                logger.info("📉 Death Cross Detected (EMA50 < EMA200)")
                logger.info("   → Favoring SELL signals")
                logger.info("   → Penalizing BUY signals")
                logger.info("   → Expect mean-reversion opportunities")
            
            logger.info(f"EMA Confidence: {ema_conf:.2f}")
            logger.info("=" * 80)
            logger.info("")
            
            self.previous_regime = is_bull

    def _log_regime_context(self, is_bull: bool, timestamp: str):
        """Log current regime on each signal (non-change bars)"""
        regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"
        regime_emoji = "📈" if is_bull else "📉"
        
        if is_bull:
            self.bull_count += 1
        else:
            self.bear_count += 1
        
        logger.debug(f"[{timestamp}] {regime_emoji} Regime: {regime_name}")

    def _calculate_weighted_buy_score(
        self, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull
    ) -> Tuple[float, str]:
        """Calculate BUY score with CORRECTED weights"""
        if mr_signal == 0 and tf_signal == 0 and ema_signal == 0:
            return 0.0, "All strategies HOLD"

        # Only positive signals contribute
        mr_buy = mr_conf if mr_signal == 1 else 0.0
        tf_buy = tf_conf if tf_signal == 1 else 0.0
        ema_buy = ema_conf if ema_signal == 1 else 0.0

        explanation = []

        # Weighted average with correct weights
        total_weight = (
            self.config["mean_reversion_weight"]
            + self.config["trend_following_weight"]
            + self.config["ema_weight"]
        )

        weighted_score = (
            (mr_buy * self.config["mean_reversion_weight"])
            + (tf_buy * self.config["trend_following_weight"])
            + (ema_buy * self.config["ema_weight"])
        ) / total_weight

        explanation.append(
            f"Base: {weighted_score:.3f} "
            f"(MR:{mr_buy:.2f}×{self.config['mean_reversion_weight']:.1f} "
            f"TF:{tf_buy:.2f}×{self.config['trend_following_weight']:.1f} "
            f"EMA:{ema_buy:.2f}×{self.config['ema_weight']:.1f})"
        )

        # Agreement bonus
        buy_count = sum([mr_signal == 1, tf_signal == 1, ema_signal == 1])
        if buy_count >= 2:
            old_score = weighted_score
            weighted_score = min(
                1.0, weighted_score + self.config["perfect_agreement_bonus"]
            )
            explanation.append(
                f"→ +{self.config['perfect_agreement_bonus']:.2f} ({buy_count}/3 agree) "
                f"{old_score:.3f}→{weighted_score:.3f}"
            )

        # Bull boost
        if is_bull and self.config["enable_bull_filter"]:
            old_score = weighted_score
            weighted_score = min(
                1.0, weighted_score + self.config["boost_buys_in_bull"]
            )
            explanation.append(
                f"→ +{self.config['boost_buys_in_bull']:.2f} (bull boost) "
                f"{old_score:.3f}→{weighted_score:.3f}"
            )

        return weighted_score, " | ".join(explanation)

    def _calculate_weighted_sell_score(
        self, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull
    ) -> Tuple[float, str]:
        """Calculate SELL score with CORRECTED weights"""
        if mr_signal == 0 and tf_signal == 0 and ema_signal == 0:
            return 0.0, "All strategies HOLD"

        # Only negative signals contribute
        mr_sell = mr_conf if mr_signal == -1 else 0.0
        tf_sell = tf_conf if tf_signal == -1 else 0.0
        ema_sell = ema_conf if ema_signal == -1 else 0.0

        explanation = []

        # Weighted average with correct weights
        total_weight = (
            self.config["mean_reversion_weight"]
            + self.config["trend_following_weight"]
            + self.config["ema_weight"]
        )

        weighted_score = (
            (mr_sell * self.config["mean_reversion_weight"])
            + (tf_sell * self.config["trend_following_weight"])
            + (ema_sell * self.config["ema_weight"])
        ) / total_weight

        explanation.append(
            f"Base: {weighted_score:.3f} "
            f"(MR:{mr_sell:.2f}×{self.config['mean_reversion_weight']:.1f} "
            f"TF:{tf_sell:.2f}×{self.config['trend_following_weight']:.1f} "
            f"EMA:{ema_sell:.2f}×{self.config['ema_weight']:.1f})"
        )

        # Agreement bonus
        sell_count = sum([mr_signal == -1, tf_signal == -1, ema_signal == -1])
        if sell_count >= 2:
            old_score = weighted_score
            weighted_score = min(
                1.0, weighted_score + self.config["perfect_agreement_bonus"]
            )
            explanation.append(
                f"→ +{self.config['perfect_agreement_bonus']:.2f} ({sell_count}/3 agree) "
                f"{old_score:.3f}→{weighted_score:.3f}"
            )

        # Bull penalty (not block)
        if is_bull and self.config["enable_bull_filter"]:
            old_score = weighted_score
            weighted_score = max(
                0.0, weighted_score - self.config["sell_penalty_in_bull"]
            )
            explanation.append(
                f"→ -{self.config['sell_penalty_in_bull']:.2f} (bull penalty) "
                f"{old_score:.3f}→{weighted_score:.3f}"
            )

        return weighted_score, " | ".join(explanation)

    def _check_single_strategy_signal(
        self, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull
    ) -> Tuple[Optional[int], Optional[str]]:
        """Check single strategy signals - prefer TF/EMA"""

        # EMA alone (72% accuracy - regime filter)
        if self.config.get("allow_single_ema_signal", False):
            ema_threshold = self.config["single_ema_threshold"]
            if ema_signal == 1 and ema_conf >= ema_threshold:
                return 1, f"single_ema_buy (conf={ema_conf:.2f})"
            if ema_signal == -1 and ema_conf >= ema_threshold:
                if not is_bull or ema_conf >= 0.70:
                    return -1, f"single_ema_sell (conf={ema_conf:.2f})"

        # TF alone (72% accuracy - momentum)
        if self.config.get("allow_single_tf_signal", False):
            tf_threshold = self.config["single_tf_threshold"]
            if tf_signal == 1 and tf_conf >= tf_threshold:
                return 1, f"single_tf_buy (conf={tf_conf:.2f})"
            if tf_signal == -1 and tf_conf >= tf_threshold:
                if not is_bull or tf_conf >= 0.70:
                    return -1, f"single_tf_sell (conf={tf_conf:.2f})"

        # MR alone (72-75% accuracy - reversals)
        if self.config.get("allow_single_mr_signal", False):
            mr_threshold = self.config.get("single_mr_threshold", 0.75)
            if mr_signal == 1 and mr_conf >= mr_threshold:
                return 1, f"single_mr_buy (conf={mr_conf:.2f})"
            if mr_signal == -1 and mr_conf >= mr_threshold:
                if not is_bull or mr_conf >= 0.80:
                    return -1, f"single_mr_sell (conf={mr_conf:.2f})"

        return None, None

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """Main aggregation with regime-aware filtering"""
        self.total_evaluations += 1

        try:
            # Get timestamp
            timestamp = (
                df.index[-1].isoformat()
                if hasattr(df.index[-1], "isoformat")
                else str(df.index[-1])
            )

            # Get all signals
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            ema_conf = self._normalize_confidence(ema_conf)
            is_bull_market = ema_signal == 1

            # Log regime context
            self._log_regime_change(is_bull_market, ema_signal, ema_conf, timestamp)
            self._log_regime_context(is_bull_market, timestamp)

            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            mr_conf = self._normalize_confidence(mr_conf)

            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            tf_conf = self._normalize_confidence(tf_conf)

            # Check single strategy first
            single_signal, single_reasoning = self._check_single_strategy_signal(
                mr_signal,
                mr_conf,
                tf_signal,
                tf_conf,
                ema_signal,
                ema_conf,
                is_bull_market,
            )

            if single_signal is not None:
                if single_signal != 0:
                    self.signals_generated += 1
                else:
                    self.holds_generated += 1

                quality = max(mr_conf, tf_conf, ema_conf)
                details = self._build_details(
                    mr_signal,
                    mr_conf,
                    tf_signal,
                    tf_conf,
                    ema_signal,
                    ema_conf,
                    is_bull_market,
                    single_signal,
                    single_reasoning,
                    quality,
                    0.0,
                    0.0,
                    f"Single: {single_reasoning}",
                    df,
                    timestamp,
                )
                return single_signal, details

            # Weighted voting
            buy_score, buy_exp = self._calculate_weighted_buy_score(
                mr_signal,
                mr_conf,
                tf_signal,
                tf_conf,
                ema_signal,
                ema_conf,
                is_bull_market,
            )
            sell_score, sell_exp = self._calculate_weighted_sell_score(
                mr_signal,
                mr_conf,
                tf_signal,
                tf_conf,
                ema_signal,
                ema_conf,
                is_bull_market,
            )

            buy_threshold = self.config["buy_score_threshold"]
            sell_threshold = self.config["sell_score_threshold"]

            # Decision
            if buy_score >= buy_threshold:
                final_signal = 1
                reasoning = "weighted_buy"
                self.signals_generated += 1
            elif sell_score >= sell_threshold:
                final_signal = -1
                reasoning = "weighted_sell"
                self.signals_generated += 1
            else:
                final_signal = 0
                reasoning = "hold_below_threshold"
                self.holds_generated += 1

            signal_quality = max(buy_score, sell_score)
            decision_path = f"Buy: {buy_exp} | Sell: {sell_exp}"

            details = self._build_details(
                mr_signal,
                mr_conf,
                tf_signal,
                tf_conf,
                ema_signal,
                ema_conf,
                is_bull_market,
                final_signal,
                reasoning,
                signal_quality,
                buy_score,
                sell_score,
                decision_path,
                df,
                timestamp,
            )

            return final_signal, details

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {"error": str(e)}

    def _build_details(
        self,
        mr_sig,
        mr_conf,
        tf_sig,
        tf_conf,
        ema_sig,
        ema_conf,
        is_bull,
        final_sig,
        reasoning,
        quality,
        buy_score,
        sell_score,
        decision_path,
        df,
        timestamp,
    ) -> Dict:
        """Build detail dictionary with regime information"""
        regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"
        
        return {
            "timestamp": timestamp,
            "regime": regime_name,
            "is_bull_market": is_bull,
            "regime_count_bull": self.bull_count,
            "regime_count_bear": self.bear_count,
            "regime_changes": self.regime_change_count,
            "mean_reversion_signal": mr_sig,
            "mean_reversion_confidence": mr_conf,
            "trend_following_signal": tf_sig,
            "trend_following_confidence": tf_conf,
            "ema_signal": ema_sig,
            "ema_confidence": ema_conf,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "final_signal": final_sig,
            "reasoning": reasoning,
            "decision_path": decision_path,
            "signal_quality": quality,
        }

    def get_regime_stats(self) -> Dict:
        """Return regime statistics"""
        return {
            "total_evaluations": self.total_evaluations,
            "bull_bars": self.bull_count,
            "bear_bars": self.bear_count,
            "bull_percentage": (self.bull_count / max(self.total_evaluations, 1)) * 100,
            "bear_percentage": (self.bear_count / max(self.total_evaluations, 1)) * 100,
            "regime_changes": self.regime_change_count,
            "signals_generated": self.signals_generated,
            "holds_generated": self.holds_generated,
        }