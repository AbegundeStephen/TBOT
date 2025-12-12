"""
The Analyst: Dynamic Support/Resistance Detection
Uses DBSCAN + K-Means with ATR-based filtering
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import logging

logger = logging.getLogger(__name__)


class DynamicAnalyst:
    """
    Automatically identifies Support & Resistance levels
    using volatility-adaptive clustering
    """
    
    def __init__(self, atr_multiplier=2.0, min_samples=3):  # Changed: 1.5→2.0, 5→3
        """
        Args:
            atr_multiplier: Wider noise filter (2.0x ATR instead of 1.5x)
            min_samples: Lower threshold (3 instead of 5)
        """
        self.atr_multiplier = atr_multiplier
        self.min_samples = min_samples
        logger.info(f"[ANALYST] Initialized with ATR multiplier: {atr_multiplier}, min_samples: {min_samples}")

    def calculate_atr(self, highs, lows, closes, period=14):
        """
        Calculate Average True Range (volatility measure)
        
        Args:
            highs, lows, closes: Price arrays
            period: Lookback period for ATR
            
        Returns:
            float: Current ATR value
        """
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)

        if len(closes) < period + 1:
            return np.mean(highs - lows)  # Fallback for short data

        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]

        tr1 = highs - lows
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(true_range[-period:])
        
        return atr

    def get_support_resistance_levels(
        self, 
        pivot_points, 
        highs, 
        lows, 
        closes, 
        n_levels=5
    ):
        """IMPROVED: More robust level detection"""
        
        # FIXED: Lower threshold from 5 to 3
        if len(pivot_points) < self.min_samples:
            logger.warning(f"[ANALYST] Insufficient pivots: {len(pivot_points)}")
            return []

        # Calculate volatility-based clustering epsilon
        current_atr = self.calculate_atr(highs, lows, closes)
        dynamic_eps = current_atr * self.atr_multiplier  # Now 2.0x instead of 1.5x
        
        if dynamic_eps <= 0:
            dynamic_eps = np.std(pivot_points) * 0.15  # More lenient fallback
        
        logger.debug(f"[ANALYST] ATR={current_atr:.2f}, epsilon={dynamic_eps:.2f}")

        # DBSCAN filtering
        X = pivot_points.reshape(-1, 1)
        db = DBSCAN(eps=dynamic_eps, min_samples=self.min_samples).fit(X)
        
        clean_mask = db.labels_ != -1
        clean_X = X[clean_mask]

        # FIXED: Return raw pivots if clustering removes too much
        if len(clean_X) < max(3, n_levels // 2):
            logger.warning(f"[ANALYST] Clustering too aggressive ({len(clean_X)} points), using raw pivots")
            return sorted(pivot_points.tolist())

        # K-means clustering
        n_clusters = min(n_levels, len(clean_X))
        kmeans = KMeans(
            n_clusters=n_clusters, 
            init='k-means++', 
            n_init=10,
            random_state=42
        ).fit(clean_X)
        
        levels = sorted(kmeans.cluster_centers_.flatten())
        
        logger.info(f"[ANALYST] Identified {len(levels)} S/R levels")
        return levels