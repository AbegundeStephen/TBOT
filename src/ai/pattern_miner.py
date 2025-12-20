"""
COMPLETE  TRAINING SYSTEM
All components enhanced with production-grade features
"""

# ==============================================================================
# 1.  PATTERN_MINER.PY
# ==============================================================================

import numpy as np
import pandas as pd
import talib
import random
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import shift

logger = logging.getLogger(__name__)


# ==============================================================================
# 2. PATTERN MINER - Adapted for 15min Candles
# ==============================================================================


class PatternMiner:
    """
    Mines candlestick patterns from 15min historical data
    Client Requirement: Must use 15min timeframe for pattern detection
    """

    def __init__(self, sequence_length=15):
        self.seq_len = sequence_length
        self.timeframe = "15min"  # Explicit timeframe tracking

        # 16 most reliable patterns
        import talib

        self.target_patterns = {
            "Engulfing": talib.CDLENGULFING,
            "Morning Star": talib.CDLMORNINGSTAR,
            "Evening Star": talib.CDLEVENINGSTAR,
            "Hammer": talib.CDLHAMMER,
            "Shooting Star": talib.CDLSHOOTINGSTAR,
            "Hanging Man": talib.CDLHANGINGMAN,
            "Inverted Hammer": talib.CDLINVERTEDHAMMER,
            "Three White Soldiers": talib.CDL3WHITESOLDIERS,
            "Three Black Crows": talib.CDL3BLACKCROWS,
            "Doji": talib.CDLDOJI,
            "Dragonfly Doji": talib.CDLDRAGONFLYDOJI,
            "Gravestone Doji": talib.CDLGRAVESTONEDOJI,
            "Harami": talib.CDLHARAMI,
            "Piercing": talib.CDLPIERCING,
            "Dark Cloud": talib.CDLDARKCLOUDCOVER,
            "Marubozu": talib.CDLMARUBOZU,
        }

        logger.info(
            f"[MINER] Initialized for {self.timeframe} candles: "
            f"{len(self.target_patterns)} patterns"
        )

    def get_num_classes(self):
        """Total classes including noise"""
        return len(self.target_patterns) + 1

    def get_pattern_map(self):
        """Pattern ID mapping"""
        return {name: i + 1 for i, name in enumerate(self.target_patterns.keys())}

    def load_csv_data(
        self, filepath: str, expected_timeframe: str = "15min"
    ) -> pd.DataFrame:
        """
        Load 15min OHLC data with validation

        Args:
            filepath: Path to CSV file
            expected_timeframe: Expected timeframe ('15min' or '4h')
        """
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower()

            # Handle timestamps - FIXED to handle both string and numeric formats
            if "timestamp" in df.columns:
                # Try parsing as datetime string first
                df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")

                # If that didn't work, try as Unix timestamp
                if df["date"].isna().all():
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                    df["date"] = pd.to_datetime(
                        df["timestamp"], unit="s", errors="coerce"
                    )

            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            required = ["open", "high", "low", "close"]
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV must contain: {required}")

            # Sort chronologically
            if "date" in df.columns:
                df = df.sort_values("date").reset_index(drop=True)

            # Validate OHLC relationships
            valid_mask = (
                (df["high"] >= df["low"])
                & (df["high"] >= df["open"])
                & (df["high"] >= df["close"])
                & (df["low"] <= df["open"])
                & (df["low"] <= df["close"])
                & (df["open"] > 0)
                & (df["close"] > 0)
            )

            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(
                    f"[MINER] Removed {invalid_count} invalid candles from {filepath}"
                )
                df = df[valid_mask].reset_index(drop=True)

            # Detect timeframe - only if we have valid dates
            if "date" in df.columns and len(df) > 1 and df["date"].notna().sum() > 1:
                # Find first two non-NaN dates
                valid_dates = df[df["date"].notna()]["date"]
                if len(valid_dates) >= 2:
                    time_diff = (
                        valid_dates.iloc[1] - valid_dates.iloc[0]
                    ).total_seconds() / 60
                    detected_tf = (
                        f"{int(time_diff)}min"
                        if time_diff < 60
                        else f"{int(time_diff/60)}H"
                    )

                    if expected_timeframe == "15min" and abs(time_diff - 15) > 1:
                        logger.warning(
                            f"[MINER] Expected 15min data but detected {detected_tf} in {filepath}"
                        )
                    elif expected_timeframe == "4h" and abs(time_diff - 240) > 1:
                        logger.warning(
                            f"[MINER] Expected 4H data but detected {detected_tf} in {filepath}"
                        )

            logger.info(
                f"[MINER] Loaded {len(df)} {expected_timeframe} candles from {filepath}"
            )
            return df

        except Exception as e:
            logger.error(f"[MINER] Error loading {filepath}: {e}")
            raise

    def load_multiple_sources(
        self, filepaths: List[str], expected_timeframe: str = "15min"
    ) -> pd.DataFrame:
        """Load and combine multiple 15min CSV files"""
        all_data = []

        for filepath in filepaths:
            try:
                df = self.load_csv_data(filepath, expected_timeframe)
                all_data.append(df)
                logger.info(f"[MINER] ✓ {Path(filepath).name}")
            except Exception as e:
                logger.warning(f"[MINER] ✗ {filepath}: {e}")

        if not all_data:
            raise ValueError("No data files loaded")

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(
            f"[MINER] Combined: {len(combined)} {expected_timeframe} candles "
            f"from {len(all_data)} files"
        )
        return combined

    def augment_pattern(self, snippet, augmentation_strength="medium"):
        """Multi-strategy augmentation for 15min patterns"""
        from scipy.ndimage import shift
        import random

        augmented = [snippet.copy()]

        if augmentation_strength == "light":
            prob, max_augs, noise_std = 0.5, 2, 0.002
        elif augmentation_strength == "medium":
            prob, max_augs, noise_std = 0.7, 3, 0.003
        else:  # aggressive
            prob, max_augs, noise_std = 0.9, 5, 0.005

        aug_count = 0

        # Gaussian noise
        if random.random() < prob and aug_count < max_augs:
            noise = np.random.normal(0, noise_std, snippet.shape)
            augmented.append(snippet + noise)
            aug_count += 1

        # Temporal shift
        if random.random() < prob * 0.7 and aug_count < max_augs:
            shift_amount = random.choice([-1, 1])
            shifted = shift(snippet, (shift_amount, 0), mode="nearest")
            augmented.append(shifted)
            aug_count += 1

        # Scaling
        if random.random() < prob * 0.6 and aug_count < max_augs:
            scale = np.random.uniform(0.995, 1.005)
            augmented.append(snippet * scale)
            aug_count += 1

        return augmented

    def mine_from_dataframe(
        self,
        df: pd.DataFrame,
        samples_per_pattern: int = 2000,
        use_augmentation: bool = True,
        augmentation_strength: str = "medium",
        min_pattern_quality: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Mine patterns from 15min candles

        Returns:
            X: Pattern samples (N, 15, 4)
            y: Pattern labels (N,)
            pattern_map: Pattern name to ID mapping
        """
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        if len(df) < self.seq_len * 2:
            raise ValueError(f"Need at least {self.seq_len * 2} candles")

        X, y = [], []
        pattern_id_map = self.get_pattern_map()
        pattern_counts = Counter()

        logger.info(
            f"[MINER] Mining from {len(df)} {self.timeframe} candles "
            f"(quality ≥ {min_pattern_quality})..."
        )

        for name, func in self.target_patterns.items():
            pattern_id = pattern_id_map[name]
            found_count = 0

            try:
                result = func(o, h, l, c)
                strong_indices = np.where(np.abs(result) >= min_pattern_quality)[0]

                for idx in strong_indices:
                    if idx < self.seq_len or found_count >= samples_per_pattern:
                        continue

                    # Extract 15-candle snippet from 15min data
                    snippet = np.stack(
                        [
                            o[idx - self.seq_len + 1 : idx + 1],
                            h[idx - self.seq_len + 1 : idx + 1],
                            l[idx - self.seq_len + 1 : idx + 1],
                            c[idx - self.seq_len + 1 : idx + 1],
                        ],
                        axis=1,
                    )

                    if snippet[0, 0] <= 0:
                        continue

                    snippet_norm = snippet / snippet[0, 0] - 1

                    if use_augmentation:
                        augmented = self.augment_pattern(
                            snippet_norm, augmentation_strength
                        )
                        for aug in augmented:
                            if found_count < samples_per_pattern:
                                X.append(aug)
                                y.append(pattern_id)
                                found_count += 1
                    else:
                        X.append(snippet_norm)
                        y.append(pattern_id)
                        found_count += 1

                pattern_counts[name] = found_count
                logger.info(f"[MINER] {name}: {found_count} samples")

            except Exception as e:
                logger.warning(f"[MINER] Error mining {name}: {e}")

        logger.info(
            f"[MINER] ✓ Total: {len(X)} pattern samples from {self.timeframe} data"
        )
        return np.array(X), np.array(y), pattern_id_map

    def generate_noise_samples(
        self, df: pd.DataFrame, num_samples: int = 1000
    ) -> np.ndarray:
        """Generate 'no pattern' samples from 15min data"""
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # Find all pattern occurrences
        all_pattern_indices = set()
        for func in self.target_patterns.values():
            try:
                result = func(o, h, l, c)
                indices = np.where(result != 0)[0]
                all_pattern_indices.update(indices)
            except:
                pass

        # Create buffer zones (±3 candles around patterns)
        buffer_zone = set()
        for idx in all_pattern_indices:
            buffer_zone.update(range(max(0, idx - 3), min(len(df), idx + 4)))

        # Clean periods
        no_pattern_indices = [
            i for i in range(self.seq_len, len(df)) if i not in buffer_zone
        ]

        if len(no_pattern_indices) < num_samples:
            num_samples = len(no_pattern_indices)

        # Sample with diversity
        step = max(1, len(no_pattern_indices) // num_samples)
        selected_indices = no_pattern_indices[::step][:num_samples]

        noise_X = []
        for idx in selected_indices:
            snippet = np.stack(
                [
                    o[idx - self.seq_len + 1 : idx + 1],
                    h[idx - self.seq_len + 1 : idx + 1],
                    l[idx - self.seq_len + 1 : idx + 1],
                    c[idx - self.seq_len + 1 : idx + 1],
                ],
                axis=1,
            )

            if snippet[0, 0] > 0:
                snippet_norm = snippet / snippet[0, 0] - 1
                noise_X.append(snippet_norm)

        logger.info(
            f"[MINER] Generated {len(noise_X)} noise samples from {self.timeframe} data"
        )
        return np.array(noise_X)
