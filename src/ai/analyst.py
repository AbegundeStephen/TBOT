"""
COMPLETE DUAL TIMEFRAME TRADING SYSTEM
- Analyst: Works on 4H candles for S/R detection
- Sniper: Trains and predicts on 15min candles
- Training Script: Handles both timeframes correctly
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter
import pickle

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. ANALYST - Adapted for 4H Candles
# ==============================================================================


class DynamicAnalyst:
    """
    Strategic Support/Resistance detection using 4H candles
    Client Requirement: Must use 4H timeframe for structural analysis
    """

    def __init__(self, atr_multiplier=2.0, min_samples=3, min_level_separation=0.5):
        self.atr_multiplier = atr_multiplier
        self.min_samples = min_samples
        self.min_level_separation = min_level_separation
        self.timeframe = "4H"  # Explicit timeframe tracking

        logger.info(
            f"[ANALYST] Initialized for {self.timeframe} candles: "
            f"ATR×{atr_multiplier}, min_samples={min_samples}"
        )

    def calculate_atr(self, highs, lows, closes, period=14):
        """Calculate ATR from 4H candles"""
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)

        if len(closes) < period + 1:
            return np.mean(highs - lows) if len(highs) > 0 else 0.0

        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]

        tr1 = highs - lows
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)

        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(true_range[-period:])

    def _merge_close_levels(self, levels, min_distance):
        """Merge levels that are too close together"""
        if not levels or len(levels) == 1:
            return levels

        merged = [levels[0]]
        for level in levels[1:]:
            if abs(level - merged[-1]) < min_distance:
                merged[-1] = (merged[-1] + level) / 2
            else:
                merged.append(level)

        return merged

    def get_support_resistance_levels(
        self, pivot_points, highs, lows, closes, n_levels=5
    ):
        """
        Identify S/R levels from 4H pivots

        Args:
            pivot_points: Array of 4H pivot highs/lows
            highs, lows, closes: Full 4H price history
            n_levels: Target number of levels

        Returns:
            List of S/R price levels (sorted)
        """
        from sklearn.cluster import DBSCAN, KMeans

        if len(pivot_points) < self.min_samples:
            logger.warning(
                f"[ANALYST] Insufficient 4H pivots: {len(pivot_points)} "
                f"(need {self.min_samples}+)"
            )
            return []

        # Calculate volatility from 4H candles
        current_atr = self.calculate_atr(highs, lows, closes)
        dynamic_eps = current_atr * self.atr_multiplier

        if dynamic_eps <= 0:
            pivot_std = np.std(pivot_points)
            dynamic_eps = pivot_std * 0.15 if pivot_std > 0 else 0.001

        logger.debug(f"[ANALYST] 4H ATR={current_atr:.2f}, epsilon={dynamic_eps:.2f}")

        # DBSCAN noise filtering
        X = pivot_points.reshape(-1, 1)
        db = DBSCAN(eps=dynamic_eps, min_samples=self.min_samples).fit(X)

        clean_mask = db.labels_ != -1
        clean_X = X[clean_mask]

        # Fallback if clustering too aggressive
        min_required = max(3, n_levels // 2)
        if len(clean_X) < min_required:
            logger.warning(
                f"[ANALYST] Using quantile fallback (only {len(clean_X)} clean points)"
            )
            percentiles = np.linspace(10, 90, n_levels)
            levels = list(np.percentile(pivot_points, percentiles))
            return sorted(set(np.round(levels, 2)))

        # K-Means clustering
        n_clusters = min(n_levels, len(clean_X))

        try:
            kmeans = KMeans(
                n_clusters=n_clusters, init="k-means++", n_init=10, random_state=None
            ).fit(clean_X)

            levels = sorted(kmeans.cluster_centers_.flatten())
        except Exception as e:
            logger.error(f"[ANALYST] K-Means failed: {e}")
            return sorted(set(np.round(pivot_points, 2)))[:n_levels]

        # Merge close levels
        min_separation = current_atr * self.min_level_separation
        levels = self._merge_close_levels(levels, min_separation)

        logger.info(
            f"[ANALYST] ✓ Identified {len(levels)} S/R levels from 4H data "
            f"(min separation: {min_separation:.2f})"
        )

        return levels

    def is_near_level(self, current_price, level, threshold_percent=0.005):
        """Check if price is near a 4H S/R level"""
        distance = abs(current_price - level) / current_price
        return distance < threshold_percent

    def classify_levels(self, levels, current_price):
        """Separate into support (below) and resistance (above)"""
        support = [l for l in levels if l < current_price]
        resistance = [l for l in levels if l > current_price]

        return {
            "support": sorted(support, reverse=True),
            "resistance": sorted(resistance),
        }
