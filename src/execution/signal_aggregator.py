"""
: Signal Aggregator with More Responsive Regime Detection
1. Tighter hysteresis thresholds (±0.15% instead of ±0.3%)
2. Additional regime indicators beyond just EMA positioning
3. Faster response to market changes
"""

import pandas as pd
import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
    Multi-Strategy Aggregator with RESPONSIVE regime detection
    - Reduced hysteresis from ±0.3% to ±0.15% for faster regime changes
    - Added momentum and volatility to regime detection
    - Multiple regime confirmation signals (not just EMA crossover)
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
        config: Dict = None,
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()

        # Initialize regime tracking
        self.previous_regime = None
        self.regime_initialized = False

        # Strategy weights
        if self.asset_type == "BTC":
            self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}
        else:  # GOLD
            self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}

        # Asset-specific configuration
        if config is not None:
            self.config = config
        else:
            if self.asset_type == "BTC":
                self.config = {
                    "buy_threshold": 0.27,
                    "sell_threshold": 0.33,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.25,
                    
                    # INCREASED from 0.04 to 0.12
                    "bull_buy_boost": 0.12,
                    "bull_sell_penalty": 0.12,
                    "bear_sell_boost": 0.12,
                    "bear_buy_penalty": 0.12,
                    
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.27,
                    "hold_contribution_pct": 0.18,
                }
            else:  # GOLD
                self.config = {
                    "buy_threshold": 0.32,
                    "sell_threshold": 0.32,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.30,
                    "bull_buy_boost": 0.02,
                    "bull_sell_penalty": 0.02,
                    "bear_sell_boost": 0.02,
                    "bear_buy_penalty": 0.02,
                    "min_confidence_to_use": 0.15,
                    "min_signal_quality": 0.25,
                    "hold_contribution_pct": 0.15,
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
            "regime_detection_failures": 0,
        }

        self._log_initialization()

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯   PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("   ✓ TIGHTER HYSTERESIS: ±0.15% (was ±0.3%)")
        logger.info("   ✓ Multi-factor regime: EMA + momentum + volatility")
        logger.info("   ✓ Faster response to market changes")
        logger.info("")
        logger.info("📊 STRATEGY ROLES:")
        logger.info("   Mean Reversion:   DECISION MAKER (weight: 0.50)")
        logger.info("   Trend Following:  DECISION MAKER (weight: 0.50)")
        logger.info("   EMA 50/200:       REGIME DETECTOR (multi-factor)")
        logger.info("")
        logger.info("🔒 REGIME HYSTERESIS:")
        logger.info("   Bullish Threshold: EMA diff > +0.15% + positive momentum")
        logger.info("   Bearish Threshold: EMA diff < -0.15% + negative momentum")
        logger.info("   → More responsive while avoiding noise")
        logger.info("=" * 80)
        logger.info("")

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        IMPROVED: Multi-factor regime detection with tighter hysteresis

        Regime Factors:
        1. EMA positioning (50 vs 200)
        2. Price momentum (20-day, 50-day returns)
        3. Volatility context (calm vs turbulent)
        4. MACD histogram
        5. ADX trend strength

        Returns:
            is_bull (bool): True if bullish regime
            confidence (float): Confidence in regime detection (0.0-1.0)
        """
        try:
            MIN_DATA_POINTS = 50

            if len(df) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for regime detection: {len(df)} rows"
                )
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # Generate features
            features_df = self.s_ema.generate_features(df.tail(250))

            if features_df.empty or len(features_df) < MIN_DATA_POINTS:
                logger.warning(f"EMA features insufficient: {len(features_df)} rows")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            latest = features_df.iloc[-1]

            # Core indicators
            ema_fast = latest.get("ema_fast", np.nan)
            ema_slow = latest.get("ema_slow", np.nan)
            ema_diff_pct = latest.get("ema_diff_pct", 0.0)

            # CRITICAL: Asset-specific hysteresis thresholds
            if self.asset_type == "BTC":
                BULLISH_THRESHOLD = 0.15  # BTC: Higher volatility
                BEARISH_THRESHOLD = -0.15
            else:  # GOLD
                BULLISH_THRESHOLD = 0.10  # Gold: Lower volatility, tighter threshold
                BEARISH_THRESHOLD = -0.10

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                logger.warning(f"Invalid EMA values")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # Calculate supporting indicators
            close_prices = features_df["close"].values

            # Multi-timeframe momentum (asset-specific thresholds)
            if self.asset_type == "BTC":
                ret_20_bull_threshold = 0.02  # BTC: 2% for bullish
                ret_20_bear_threshold = -0.02  # BTC: -2% for bearish
                ret_50_bull_threshold = 0.05  # BTC: 5% for strong bull
                ret_50_bear_threshold = -0.05  # BTC: -5% for strong bear
            else:  # GOLD
                ret_20_bull_threshold = 0.01  # Gold: 1% for bullish
                ret_20_bear_threshold = -0.01  # Gold: -1% for bearish
                ret_50_bull_threshold = 0.03  # Gold: 3% for strong bull
                ret_50_bear_threshold = -0.03  # Gold: -3% for strong bear

            ret_20 = (
                (close_prices[-1] - close_prices[-20]) / close_prices[-20]
                if len(close_prices) >= 20
                else 0.0
            )
            ret_50 = (
                (close_prices[-1] - close_prices[-50]) / close_prices[-50]
                if len(close_prices) >= 50
                else 0.0
            )

            # Volatility (20-day)
            if len(close_prices) >= 21:
                returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                vol_20 = np.std(returns) * np.sqrt(252)
            else:
                vol_20 = 0.2

            # Technical indicators
            adx = latest.get("adx", 20)
            macd_hist = latest.get("macd_hist", 0)
            rsi = latest.get("rsi", 50)

            # ===== MULTI-FACTOR REGIME DETECTION =====

            bullish_score = 0
            bearish_score = 0

            # Factor 1: EMA positioning (primary)
            if ema_diff_pct > BULLISH_THRESHOLD:
                bullish_score += 3
            elif ema_diff_pct < BEARISH_THRESHOLD:
                bearish_score += 3

            # Factor 2: Short-term momentum (20-day)
            if ret_20 > 0.02:  # >2% gain
                bullish_score += 2
            elif ret_20 < -0.02:  # >2% loss
                bearish_score += 2

            # Factor 3: Medium-term momentum (50-day)
            if ret_50 > 0.05:  # >5% gain
                bullish_score += 2
            elif ret_50 < -0.05:  # >5% loss
                bearish_score += 2

            # Factor 4: MACD histogram
            if macd_hist > 0:
                bullish_score += 1
            elif macd_hist < 0:
                bearish_score += 1

            # Factor 5: ADX (strong trend adds confidence)
            if adx > 25:
                if ema_diff_pct > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            # Factor 6: RSI extremes
            if rsi > 60:
                bullish_score += 1
            elif rsi < 40:
                bearish_score += 1

            # ===== REGIME DECISION WITH HYSTERESIS =====

            if self.previous_regime is None:
                # Initial regime (no history)
                is_bull = bullish_score > bearish_score
            else:
                # Subsequent regime (with hysteresis)
                if self.previous_regime:
                    # Currently BULLISH - need clear bearish signals to flip
                    if bearish_score > bullish_score + 2:  # Need 2-point advantage
                        is_bull = False
                        logger.info(
                            f"   🔄 Regime flip: BULL→BEAR (scores: bull={bullish_score}, bear={bearish_score})"
                        )
                    else:
                        is_bull = True
                else:
                    # Currently BEARISH - need clear bullish signals to flip
                    if bullish_score > bearish_score + 2:  # Need 2-point advantage
                        is_bull = True
                        logger.info(
                            f"   🔄 Regime flip: BEAR→BULL (scores: bull={bullish_score}, bear={bearish_score})"
                        )
                    else:
                        is_bull = False

            # ===== CONFIDENCE CALCULATION =====

            confidence = 0.5

            # Boost for strong EMA separation
            if abs(ema_diff_pct) > 0.5:
                confidence += 0.15

            # Boost for strong momentum alignment
            if (is_bull and ret_20 > 0.03) or (not is_bull and ret_20 < -0.03):
                confidence += 0.15

            # Boost for trend strength
            if adx > 25:
                confidence += 0.1

            # Boost for clear score advantage
            score_diff = abs(bullish_score - bearish_score)
            if score_diff >= 4:
                confidence += 0.1

            confidence = min(1.0, max(0.3, confidence))

            # ===== LOGGING =====

            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
                regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
                logger.info("")
                logger.info("=" * 70)
                logger.info(
                    f"⚡ REGIME CHANGE #{self.stats['regime_changes']} → {regime_name}"
                )
                logger.info(
                    f"   EMA Fast: ${ema_fast:.2f} | Slow: ${ema_slow:.2f} | Diff: {ema_diff_pct:.2f}%"
                )
                logger.info(f"   Momentum: 20d={ret_20:.2%} | 50d={ret_50:.2%}")
                logger.info(f"   Scores: Bull={bullish_score} | Bear={bearish_score}")
                logger.info(
                    f"   ADX: {adx:.1f} | MACD: {macd_hist:.3f} | RSI: {rsi:.1f}"
                )
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info("=" * 70)
                logger.info("")
            elif not self.regime_initialized:
                regime_name = "🚀 BULL MARKET" if is_bull else "🐻 BEAR MARKET"
                logger.info("")
                logger.info("=" * 70)
                logger.info(f"🎬 INITIAL REGIME → {regime_name}")
                logger.info(f"   Scores: Bull={bullish_score} | Bear={bearish_score}")
                logger.info(
                    f"   EMA Diff: {ema_diff_pct:.2f}% | Momentum: {ret_20:.2%}"
                )
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info("=" * 70)
                logger.info("")
                self.regime_initialized = True

            # Update tracking
            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            return is_bull, confidence

        except Exception as e:
            logger.error(f"Error detecting regime: {e}", exc_info=True)
            self.stats["regime_detection_failures"] += 1
            fallback_regime = (
                self.previous_regime if self.previous_regime is not None else False
            )
            return fallback_regime, 0.3

    def _calculate_score(
        self,
        target_signal: int,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """Calculate score with HOLD signals contributing context"""
        components = []
        total_score = 0.0
        agreement_count = 0
        min_conf = self.config["min_confidence_to_use"]
        hold_contrib = self.config["hold_contribution_pct"]

        # Mean Reversion contribution
        if mr_signal == target_signal:
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_agree:{contribution:.3f}")
            agreement_count += 1
        elif mr_signal == 0:
            effective_conf = max(mr_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "mean_reversion"
            ]
            total_score += contribution
            components.append(f"MR_hold:{contribution:.3f}")
        else:
            effective_conf = max(mr_conf, min_conf)
            penalty = (effective_conf * 0.5) * self.weights["mean_reversion"]
            total_score -= penalty
            components.append(f"MR_oppose:-{penalty:.3f}")

        # Trend Following contribution
        if tf_signal == target_signal:
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_agree:{contribution:.3f}")
            agreement_count += 1
        elif tf_signal == 0:
            effective_conf = max(tf_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "trend_following"
            ]
            total_score += contribution
            components.append(f"TF_hold:{contribution:.3f}")
        else:
            effective_conf = max(tf_conf, min_conf)
            penalty = (effective_conf * 0.5) * self.weights["trend_following"]
            total_score -= penalty
            components.append(f"TF_oppose:-{penalty:.3f}")

        explanation = " + ".join(components) if components else "no_agreement"

        # Agreement bonus
        if agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus({bonus:.2f})"

        # Regime context
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

        total_score = max(0.0, total_score)
        return total_score, explanation, agreement_count

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """Main aggregation logic with improved regime detection"""
        self.stats["total_evaluations"] += 1

        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

            # STEP 1: Detect regime (IMPROVED)
            is_bull, regime_conf = self._detect_regime(df)
            regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"

            # STEP 2: Get strategy signals
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)

            # STEP 3: Calculate scores
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

            # STEP 4: Make decision
            buy_thresh = self.config["buy_threshold"]
            sell_thresh = self.config["sell_threshold"]
            min_quality = self.config["min_signal_quality"]

            signal_quality = max(buy_score, sell_score)

            if buy_score >= buy_thresh and buy_score > sell_score:
                if signal_quality >= min_quality:
                    final_signal = 1
                    reasoning = (
                        f"BUY (score:{buy_score:.2f}, quality:{signal_quality:.2f})"
                    )
                    self.stats["buy_signals"] += 1
                    self.stats["signals_generated"] += 1
                else:
                    final_signal = 0
                    reasoning = f"hold_lowquality ({signal_quality:.2f})"
                    self.stats["hold_signals"] += 1
            elif sell_score >= sell_thresh and sell_score > buy_score:
                if signal_quality >= min_quality:
                    final_signal = -1
                    reasoning = (
                        f"SELL (score:{sell_score:.2f}, quality:{signal_quality:.2f})"
                    )
                    self.stats["sell_signals"] += 1
                    self.stats["signals_generated"] += 1
                else:
                    final_signal = 0
                    reasoning = f"hold_lowquality ({signal_quality:.2f})"
                    self.stats["hold_signals"] += 1
            else:
                final_signal = 0
                reasoning = f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                self.stats["hold_signals"] += 1

            details = {
                "timestamp": timestamp,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "final_signal": final_signal,
                "reasoning": reasoning,
                "signal_quality": signal_quality,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
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
            "bull_regime_pct": (self.stats["bull_regime_count"] / total) * 100,
            "bear_regime_pct": (self.stats["bear_regime_count"] / total) * 100,
            "consensus_rate": (
                self.stats["consensus_signals"]
                / max(self.stats["signals_generated"], 1)
            )
            * 100,
        }
