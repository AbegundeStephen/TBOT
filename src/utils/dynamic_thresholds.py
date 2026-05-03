"""
Dynamic Thresholds Engine
Replaces static magic numbers with market-derived baselines.
Every threshold adapts to the asset's own recent behaviour.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DynamicThresholds:
    """
    Converts any metric into a Z-score against its rolling distribution.
    Usage:
        is_extreme, z_score, threshold = self.thresholds.check(
            asset="BTC", metric="ema50_distance",
            value=current_distance, z_threshold=2.5,
            fallback=3.5  # Static fallback if insufficient data
        )
    """

    def __init__(self, lookback: int = 100, min_samples: int = 20):
        self._cache = {}  # {(asset, metric): [values]}
        self.lookback = lookback
        self.min_samples = min_samples

    def check(self, asset: str, metric: str, value: float,
              z_threshold: float = 2.0, fallback: float = None) -> tuple:
        """
        Returns (is_extreme: bool, z_score: float, dynamic_threshold: float)
        Falls back to static threshold if insufficient data.
        """
        key = (asset, metric)

        # Update rolling window
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(value)
        if len(self._cache[key]) > self.lookback:
            self._cache[key] = self._cache[key][-self.lookback:]

        values = self._cache[key]

        # Not enough data — use static fallback
        if len(values) < self.min_samples:
            if fallback is not None:
                return value > fallback, 0.0, fallback
            return False, 0.0, value

        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return False, 0.0, mean

        z_score = (value - mean) / std
        dynamic_threshold = mean + (z_threshold * std)

        return abs(z_score) > z_threshold, z_score, dynamic_threshold

    def get_percentile(self, asset: str, metric: str, value: float,
                       percentile: float = 0.90) -> tuple:
        """
        Returns (exceeds_percentile: bool, percentile_value: float)
        """
        key = (asset, metric)
        values = self._cache.get(key, [])
        if len(values) < self.min_samples:
            return False, value
        pct_val = np.percentile(values, percentile * 100)
        return value > pct_val, pct_val
