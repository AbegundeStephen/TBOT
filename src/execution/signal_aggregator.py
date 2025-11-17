"""
Enhanced Signal Aggregator - Tiered Confidence System
Adapts to actual model performance instead of unrealistic thresholds
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Tiered confidence system that works with real-world model accuracies
    """

    def __init__(
        self,
        mean_reversion_strategy: BaseStrategy,
        trend_following_strategy: BaseStrategy,
        mode: str = "adaptive_tiered",
        confidence_config: Optional[Dict] = None
    ):
        self.s_range = mean_reversion_strategy
        self.s_trend = trend_following_strategy
        self.mode = mode
        
        # REALISTIC confidence tiers based on 50-60% accuracy models
        default_config = {
            "tier_1_threshold": 0.45,   # Minimum to consider (relaxed)
            "tier_2_threshold": 0.55,   # Good confidence
            "tier_3_threshold": 0.65,   # High confidence
            "agreement_bonus": 0.10,    # Boost when both agree
            "require_agreement": False,  # Allow single-strategy signals
        }
        
        self.conf_config = confidence_config or default_config
        
        logger.info(f"EnhancedSignalAggregator initialized:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Tier 1 (Min): {self.conf_config['tier_1_threshold']:.2f}")
        logger.info(f"  Tier 2 (Good): {self.conf_config['tier_2_threshold']:.2f}")
        logger.info(f"  Tier 3 (High): {self.conf_config['tier_3_threshold']:.2f}")

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Apply tiered confidence aggregation
        """
        # Get raw signals WITH confidence scores (no filtering yet)
        signal_range, conf_range = self.s_range.generate_signal(df)
        signal_trend, conf_trend = self.s_trend.generate_signal(df)

        # Apply aggregation logic based on mode
        if self.mode == "adaptive_tiered":
            final_signal, combined_conf, reasoning = self._adaptive_tiered_mode(
                signal_range, conf_range, signal_trend, conf_trend
            )
        elif self.mode == "score_based":
            final_signal, combined_conf, reasoning = self._score_based_mode(
                signal_range, conf_range, signal_trend, conf_trend
            )
        elif self.mode == "dynamic_threshold":
            final_signal, combined_conf, reasoning = self._dynamic_threshold_mode(
                signal_range, conf_range, signal_trend, conf_trend
            )
        else:
            logger.error(f"Unknown mode: {self.mode}")
            final_signal, combined_conf, reasoning = 0, 0.0, "error"

        signal_details = {
            "mean_reversion_signal": signal_range,
            "mean_reversion_confidence": conf_range,
            "trend_following_signal": signal_trend,
            "trend_following_confidence": conf_trend,
            "final_signal": final_signal,
            "combined_confidence": combined_conf,
            "reasoning": reasoning,
            "mode": self.mode,
            "timestamp": (
                df.index[-1].isoformat()
                if hasattr(df.index[-1], "isoformat")
                else str(df.index[-1])
            ),
        }

        logger.info(
            f"Signal [{self.mode}] - "
            f"MR: {signal_range:>2} ({conf_range:.3f}), "
            f"TF: {signal_trend:>2} ({conf_trend:.3f}) → "
            f"Final: {final_signal:>2} ({combined_conf:.3f}) [{reasoning}]"
        )

        return final_signal, signal_details

    def _adaptive_tiered_mode(
        self, 
        sig_mr: int, conf_mr: float,
        sig_tf: int, conf_tf: float
    ) -> Tuple[int, float, str]:
        """
        ADAPTIVE TIERED (RECOMMENDED)
        
        Uses realistic confidence tiers:
        - Tier 3 (≥0.65): Can trade alone
        - Tier 2 (≥0.55): Needs agreement or confirmation
        - Tier 1 (≥0.45): Only if both strategies align
        """
        t1 = self.conf_config['tier_1_threshold']
        t2 = self.conf_config['tier_2_threshold']
        t3 = self.conf_config['tier_3_threshold']
        
        # === CASE 1: Both strategies agree ===
        if sig_mr == sig_tf and sig_mr != 0:
            # Both pointing same direction
            avg_conf = (conf_mr + conf_tf) / 2
            
            # Apply agreement bonus
            boosted_conf = min(avg_conf + self.conf_config['agreement_bonus'], 1.0)
            
            # Accept if either confidence is above Tier 1
            if conf_mr >= t1 or conf_tf >= t1:
                return sig_mr, boosted_conf, f"agreement_tier1"
            else:
                return 0, boosted_conf, f"agreement_too_weak"
        
        # === CASE 2: One strategy very confident (Tier 3) ===
        # Mean Reversion is highly confident
        if sig_mr != 0 and conf_mr >= t3:
            # Allow if other is neutral OR weakly opposing
            if sig_tf == 0 or conf_tf < t2:
                return sig_mr, conf_mr, "mr_tier3_solo"
        
        # Trend Following is highly confident
        if sig_tf != 0 and conf_tf >= t3:
            if sig_mr == 0 or conf_mr < t2:
                return sig_tf, conf_tf, "tf_tier3_solo"
        
        # === CASE 3: One strategy confident (Tier 2), other neutral ===
        # Mean Reversion confident, Trend neutral
        if sig_mr != 0 and conf_mr >= t2 and sig_tf == 0:
            return sig_mr, conf_mr, "mr_tier2_neutral"
        
        # Trend confident, Mean Reversion neutral
        if sig_tf != 0 and conf_tf >= t2 and sig_mr == 0:
            return sig_tf, conf_tf, "tf_tier2_neutral"
        
        # === CASE 4: Strong disagreement - NO TRADE ===
        if sig_mr != 0 and sig_tf != 0 and sig_mr != sig_tf:
            # Both have signals but opposing
            if conf_mr >= t2 or conf_tf >= t2:
                return 0, max(conf_mr, conf_tf), "strong_disagreement"
        
        # === CASE 5: All other cases - HOLD ===
        return 0, max(conf_mr, conf_tf), "no_clear_signal"

    def _score_based_mode(
        self, 
        sig_mr: int, conf_mr: float,
        sig_tf: int, conf_tf: float
    ) -> Tuple[int, float, str]:
        """
        SCORE-BASED MODE
        
        Calculates weighted score for BUY/SELL/HOLD
        Trades if score exceeds threshold
        """
        # Initialize scores
        buy_score = 0.0
        sell_score = 0.0
        
        # Add weighted contributions
        if sig_mr == 1:
            buy_score += conf_mr
        elif sig_mr == -1:
            sell_score += conf_mr
        
        if sig_tf == 1:
            buy_score += conf_tf
        elif sig_tf == -1:
            sell_score += conf_tf
        
        # Decision logic
        threshold = self.conf_config['tier_2_threshold']  # 0.55
        
        if buy_score > sell_score and buy_score >= threshold:
            return 1, buy_score, f"buy_score_{buy_score:.2f}"
        elif sell_score > buy_score and sell_score >= threshold:
            return -1, sell_score, f"sell_score_{sell_score:.2f}"
        else:
            return 0, max(buy_score, sell_score), "scores_too_low"

    def _dynamic_threshold_mode(
        self, 
        sig_mr: int, conf_mr: float,
        sig_tf: int, conf_tf: float
    ) -> Tuple[int, float, str]:
        """
        DYNAMIC THRESHOLD MODE
        
        Adjusts required confidence based on market conditions
        Uses volatility proxy from confidence spread
        """
        t1 = self.conf_config['tier_1_threshold']
        
        # Calculate confidence spread (volatility proxy)
        conf_spread = abs(conf_mr - conf_tf)
        
        # Dynamic threshold: lower when strategies agree on confidence level
        if conf_spread < 0.15:  # Strategies have similar confidence
            effective_threshold = t1  # Lower bar (0.45)
        else:  # Strategies differ in confidence
            effective_threshold = t1 + 0.10  # Higher bar (0.55)
        
        # Both agree
        if sig_mr == sig_tf and sig_mr != 0:
            avg_conf = (conf_mr + conf_tf) / 2
            if avg_conf >= effective_threshold:
                return sig_mr, avg_conf, f"dynamic_agree_{effective_threshold:.2f}"
        
        # One very confident
        if sig_mr != 0 and conf_mr >= effective_threshold + 0.15:
            return sig_mr, conf_mr, "dynamic_mr_strong"
        
        if sig_tf != 0 and conf_tf >= effective_threshold + 0.15:
            return sig_tf, conf_tf, "dynamic_tf_strong"
        
        return 0, max(conf_mr, conf_tf), "dynamic_threshold_not_met"

    def load_models(self, mean_rev_path: str, trend_path: str) -> bool:
        """Load both strategy models"""
        mr_loaded = self.s_range.load_model(mean_rev_path)
        tf_loaded = self.s_trend.load_model(trend_path)

        if mr_loaded and tf_loaded:
            logger.info("✅ All strategy models loaded successfully")
            return True
        else:
            logger.error("❌ Failed to load one or more strategy models")
            return False


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

CONFIDENCE_PRESETS = {
    "conservative": {
        "tier_1_threshold": 0.50,
        "tier_2_threshold": 0.60,
        "tier_3_threshold": 0.70,
        "agreement_bonus": 0.05,
        "require_agreement": True,
    },
    "balanced": {
        "tier_1_threshold": 0.45,
        "tier_2_threshold": 0.55,
        "tier_3_threshold": 0.65,
        "agreement_bonus": 0.10,
        "require_agreement": False,
    },
    "aggressive": {
        "tier_1_threshold": 0.40,
        "tier_2_threshold": 0.50,
        "tier_3_threshold": 0.60,
        "agreement_bonus": 0.15,
        "require_agreement": False,
    }
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
EXAMPLE 1: Adaptive Tiered (Start Here)
----------------------------------------
aggregator = EnhancedSignalAggregator(
    mean_reversion_strategy=mr_strategy,
    trend_following_strategy=tf_strategy,
    mode="adaptive_tiered",
    confidence_config=CONFIDENCE_PRESETS["balanced"]
)

EXAMPLE 2: Score-Based (More Trades)
-------------------------------------
aggregator = EnhancedSignalAggregator(
    mean_reversion_strategy=mr_strategy,
    trend_following_strategy=tf_strategy,
    mode="score_based",
    confidence_config=CONFIDENCE_PRESETS["aggressive"]
)

EXAMPLE 3: Custom Configuration
--------------------------------
custom_config = {
    "tier_1_threshold": 0.42,
    "tier_2_threshold": 0.52,
    "tier_3_threshold": 0.62,
    "agreement_bonus": 0.12,
    "require_agreement": False,
}

aggregator = SignalAggregator(
    mean_reversion_strategy=mr_strategy,
    trend_following_strategy=tf_strategy,
    mode="adaptive_tiered",
    confidence_config=custom_config
)
"""