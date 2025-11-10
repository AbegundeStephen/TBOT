# **src/execution/signal_aggregator.py**
"""
Signal Aggregator - Central orchestrator for trade decisions
Implements the Confirmation Filter logic
"""

import pandas as pd
import logging
from typing import Dict, Tuple
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Aggregates signals from multiple strategies
    Only executes when both strategies agree
    """

    def __init__(
        self,
        mean_reversion_strategy: BaseStrategy,
        trend_following_strategy: BaseStrategy,
    ):
        self.s_range = mean_reversion_strategy
        self.s_trend = trend_following_strategy

    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Apply Confirmation Filter
        Returns: (final_signal, signal_details)

        Signal values:
        - 1: BUY (both strategies agree)
        - -1: SELL (both strategies agree)
        - 0: HOLD (strategies disagree or insufficient confidence)
        """
        # Get individual strategy signals
        signal_range = self.s_range.predict_signal(df)
        signal_trend = self.s_trend.predict_signal(df)

        # Apply Confirmation Filter
        if signal_range == 1 and signal_trend == 1:
            final_signal = 1  # Strong BUY
            confidence = "HIGH"
        elif signal_range == -1 and signal_trend == -1:
            final_signal = -1  # Strong SELL
            confidence = "HIGH"
        else:
            final_signal = 0  # HOLD (conflicting or neutral signals)
            confidence = "LOW"

        signal_details = {
            "mean_reversion_signal": signal_range,
            "trend_following_signal": signal_trend,
            "final_signal": final_signal,
            "confidence": confidence,
            "timestamp": (
                df.index[-1]
                if hasattr(df.index[-1], "isoformat")
                else str(df.index[-1])
            ),
        }

        logger.info(
            f"Signal Aggregation - Range: {signal_range}, Trend: {signal_trend}, "
            f"Final: {final_signal} ({confidence} confidence)"
        )

        return final_signal, signal_details

    def load_models(self, mean_rev_path: str, trend_path: str) -> bool:
        """Load both strategy models"""
        mr_loaded = self.s_range.load_model(mean_rev_path)
        tf_loaded = self.s_trend.load_model(trend_path)

        if mr_loaded and tf_loaded:
            logger.info("All strategy models loaded successfully")
            return True
        else:
            logger.error("Failed to load one or more strategy models")
            return False
