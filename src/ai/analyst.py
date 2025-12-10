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
    
    def __init__(self, atr_multiplier=1.5, min_samples=5):
        """
        Args:
            atr_multiplier: Width of noise filter (1.5x ATR is standard)
            min_samples: Minimum points to form a valid cluster
        """
        self.atr_multiplier = atr_multiplier
        self.min_samples = min_samples
        logger.info(f"[ANALYST] Initialized with ATR multiplier: {atr_multiplier}")

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
        """
        Identify key Support/Resistance price levels
        
        Args:
            pivot_points: Array of local highs/lows
            highs, lows, closes: Full price history
            n_levels: Number of S/R levels to identify
            
        Returns:
            List[float]: Sorted S/R price levels
        """
        if len(pivot_points) < self.min_samples:
            logger.warning(f"[ANALYST] Insufficient pivots: {len(pivot_points)}")
            return []

        # Step 1: Measure current volatility
        current_atr = self.calculate_atr(highs, lows, closes)
        dynamic_eps = current_atr * self.atr_multiplier
        
        if dynamic_eps <= 0:
            dynamic_eps = np.std(pivot_points) * 0.1  # Fallback
        
        logger.debug(f"[ANALYST] ATR={current_atr:.2f}, epsilon={dynamic_eps:.2f}")

        # Step 2: Filter noise using DBSCAN
        X = pivot_points.reshape(-1, 1)
        db = DBSCAN(eps=dynamic_eps, min_samples=self.min_samples).fit(X)
        
        # Keep only non-noise points (label != -1)
        clean_mask = db.labels_ != -1
        clean_X = X[clean_mask]

        if len(clean_X) < n_levels:
            logger.warning(f"[ANALYST] Only {len(clean_X)} clean pivots found")
            return sorted(clean_X.flatten().tolist()) if len(clean_X) > 0 else []

        # Step 3: Cluster clean data to find levels
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