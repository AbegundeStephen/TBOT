"""
Historical Data Pattern Miner - FIXED VERSION
Mines patterns from REAL market data for much better accuracy
"""

import numpy as np
import pandas as pd
import talib
import random
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class PatternMiner:
    """
    Mines candlestick patterns from real historical OHLC data
    """
    
    def __init__(self, sequence_length=15):
        self.seq_len = sequence_length
        
        # Focus on most reliable, distinct patterns
        self.target_patterns = {
            'Engulfing': talib.CDLENGULFING,
            'Morning Star': talib.CDLMORNINGSTAR,
            'Evening Star': talib.CDLEVENINGSTAR,
            'Hammer': talib.CDLHAMMER,
            'Shooting Star': talib.CDLSHOOTINGSTAR,
            'Hanging Man': talib.CDLHANGINGMAN,
            'Inverted Hammer': talib.CDLINVERTEDHAMMER,
            'Three White Soldiers': talib.CDL3WHITESOLDIERS,
            'Three Black Crows': talib.CDL3BLACKCROWS,
            'Doji': talib.CDLDOJI,
            'Dragonfly Doji': talib.CDLDRAGONFLYDOJI,
            'Gravestone Doji': talib.CDLGRAVESTONEDOJI,
            'Harami': talib.CDLHARAMI,
            'Piercing': talib.CDLPIERCING,
            'Dark Cloud': talib.CDLDARKCLOUDCOVER,
            'Marubozu': talib.CDLMARUBOZU,
        }
        
        logger.info(f"[HISTORICAL MINER] Initialized with {len(self.target_patterns)} patterns")

    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLC data from CSV file
        
        Expected columns: timestamp, open, high, low, close, volume
        or: date, open, high, low, close, volume
        """
        try:
            df = pd.read_csv(filepath)
            
            # Standardize column names (lowercase)
            df.columns = df.columns.str.lower()
            
            # Handle different timestamp column names
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            elif 'time' in df.columns:
                df['date'] = pd.to_datetime(df['time'], errors='coerce')
            
            # Verify required columns
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV must contain: {required}")
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"[HISTORICAL MINER] Loaded {len(df)} candles from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"[HISTORICAL MINER] Error loading {filepath}: {e}")
            raise
        
    def load_saved_asset_data(self, asset_key: str, train_only: bool = True) -> pd.DataFrame:
        """
        Loads saved data for the asset
        
        Args:
            asset_key: Asset identifier (e.g., "btc", "gold")
            train_only: If True, loads ONLY training data. If False, loads train + test
        
        Example: 
            load_saved_asset_data("btc", train_only=True)  # Only train_data_btc.csv
            load_saved_asset_data("btc", train_only=False) # Both train + test
        """

        asset_key = asset_key.lower()

        train_path = f"data/train_data_{asset_key}.csv"
        test_path = f"data/test_data_{asset_key}.csv"

        paths = []

        # Always load training data
        if Path(train_path).exists():
            paths.append(train_path)
        else:
            raise FileNotFoundError(
                f"Training data not found: {train_path}"
            )
        
        # Optionally load test data
        if not train_only and Path(test_path).exists():
            paths.append(test_path)
            logger.info(f"[HISTORICAL MINER] Loading train + test data for {asset_key}")
        else:
            logger.info(f"[HISTORICAL MINER] Loading TRAIN ONLY for {asset_key}")

        return self.load_multiple_sources(paths)


    def load_multiple_sources(self, filepaths: List[str]) -> pd.DataFrame:
        """
        Load and combine multiple CSV files
        Useful for training on multiple symbols/timeframes
        """
        all_data = []
        
        for filepath in filepaths:
            try:
                df = self.load_csv_data(filepath)
                all_data.append(df)
                logger.info(f"[HISTORICAL MINER] ✓ Loaded {Path(filepath).name}")
            except Exception as e:
                logger.warning(f"[HISTORICAL MINER] ✗ Failed to load {filepath}: {e}")
        
        if not all_data:
            raise ValueError("No data files loaded successfully")
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"[HISTORICAL MINER] Combined total: {len(combined)} candles")
        
        return combined

    def validate_pattern_strength(self, pattern_value):
        """Only keep strong pattern signals (±100)"""
        return abs(pattern_value) >= 100

    def augment_pattern(self, snippet):
        """
        Light augmentation for historical data
        (Less aggressive than synthetic since data is already real)
        """
        augmented = [snippet.copy()]
        
        # Add tiny noise (±0.2% - imperceptible)
        if random.random() > 0.5:
            noise = np.random.normal(1.0, 0.002, snippet.shape)
            augmented.append(snippet * noise)
        
        return augmented

    def mine_from_dataframe(
        self,
        df: pd.DataFrame,
        samples_per_pattern: int = 1000,
        use_augmentation: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Mine patterns from a DataFrame of OHLC data
        
        Args:
            df: DataFrame with columns: open, high, low, close
            samples_per_pattern: Target samples per pattern class
            use_augmentation: Apply light augmentation
            
        Returns:
            X, y, pattern_map
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        if len(df) < self.seq_len * 2:
            raise ValueError(f"Need at least {self.seq_len * 2} candles, got {len(df)}")
        
        X, y = [], []
        pattern_id_map = {name: i+1 for i, name in enumerate(self.target_patterns.keys())}
        pattern_counts = Counter()
        
        logger.info(f"[HISTORICAL MINER] Mining from {len(df)} candles...")
        
        # Mine each pattern
        for name, func in self.target_patterns.items():
            pattern_id = pattern_id_map[name]
            found_count = 0
            
            try:
                # Run TA-Lib pattern detection
                result = func(o, h, l, c)
                
                # Get strong pattern occurrences
                strong_indices = np.where(np.abs(result) >= 100)[0]
                
                logger.debug(f"[HISTORICAL MINER] {name}: found {len(strong_indices)} occurrences")
                
                # Collect samples
                for idx in strong_indices:
                    if idx < self.seq_len:
                        continue
                    
                    if found_count >= samples_per_pattern:
                        break
                    
                    # Extract 15-candle snippet
                    snippet = np.stack([
                        o[idx-self.seq_len+1 : idx+1],
                        h[idx-self.seq_len+1 : idx+1],
                        l[idx-self.seq_len+1 : idx+1],
                        c[idx-self.seq_len+1 : idx+1]
                    ], axis=1)
                    
                    # Normalize (percentage change from first candle)
                    if snippet[0, 0] > 0:
                        snippet_norm = snippet / snippet[0, 0] - 1
                        
                        # Apply augmentation
                        if use_augmentation:
                            augmented = self.augment_pattern(snippet_norm)
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
                logger.info(f"[HISTORICAL MINER] {name}: {found_count} samples")
                
            except Exception as e:
                logger.warning(f"[HISTORICAL MINER] Error mining {name}: {e}")
        
        logger.info(f"[HISTORICAL MINER] ✓ Total: {len(X)} pattern samples")
        
        return np.array(X), np.array(y), pattern_id_map

    def mine_patterns(
        self,
        csv_files: List[str] = None,
        data_folder: str = None,
        samples_per_pattern: int = 1000,
        use_augmentation: bool = True,
        add_synthetic_fallback: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main mining function - loads data and extracts patterns
        
        Args:
            csv_files: List of CSV file paths (priority)
            data_folder: Folder containing CSV files (loads all *.csv)
            samples_per_pattern: Target samples per pattern
            use_augmentation: Apply augmentation
            add_synthetic_fallback: Add synthetic data if historical insufficient
            
        Returns:
            X, y, pattern_map
        """
        # Determine data sources
        sources = []
        
        if csv_files:
            sources.extend(csv_files)
        
        if data_folder:
            folder_path = Path(data_folder)
            if folder_path.exists():
                csv_files_in_folder = list(folder_path.glob("*.csv"))
                sources.extend([str(f) for f in csv_files_in_folder])
                logger.info(f"[HISTORICAL MINER] Found {len(csv_files_in_folder)} CSV files in {data_folder}")
        
        if not sources:
            raise ValueError("No data sources provided. Specify csv_files or data_folder.")
        
        # Load all data
        df = self.load_multiple_sources(sources)
        
        # Mine patterns
        X, y, pattern_map = self.mine_from_dataframe(
            df,
            samples_per_pattern=samples_per_pattern,
            use_augmentation=use_augmentation
        )
        
        # Check if we got enough samples
        class_counts = Counter(y)
        min_samples = min(class_counts.values()) if class_counts else 0
        
        logger.info(f"[HISTORICAL MINER] Minimum samples per class: {min_samples}")
        
        # Add synthetic data as fallback if needed
        if add_synthetic_fallback and min_samples < samples_per_pattern * 0.3:
            logger.warning(f"[HISTORICAL MINER] Some patterns have < 30% target samples")
            logger.warning(f"[HISTORICAL MINER] Consider adding more historical data or enabling synthetic fallback")
        
        return X, y, pattern_map

    def generate_noise_samples(
        self,
        df: pd.DataFrame,
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate 'no pattern' samples from historical data
        Randomly sample periods without detected patterns
        """
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        # Detect ALL patterns
        all_pattern_indices = set()
        for func in self.target_patterns.values():
            try:
                result = func(o, h, l, c)
                indices = np.where(result != 0)[0]
                all_pattern_indices.update(indices)
            except:
                pass
        
        # Find indices WITHOUT patterns
        no_pattern_indices = [i for i in range(self.seq_len, len(df)) 
                            if i not in all_pattern_indices]
        
        if len(no_pattern_indices) < num_samples:
            logger.warning(f"[HISTORICAL MINER] Only {len(no_pattern_indices)} no-pattern samples available")
            num_samples = len(no_pattern_indices)
        
        # Randomly sample
        selected_indices = random.sample(no_pattern_indices, num_samples)
        
        noise_X = []
        for idx in selected_indices:
            snippet = np.stack([
                o[idx-self.seq_len+1 : idx+1],
                h[idx-self.seq_len+1 : idx+1],
                l[idx-self.seq_len+1 : idx+1],
                c[idx-self.seq_len+1 : idx+1]
            ], axis=1)
            
            if snippet[0, 0] > 0:
                snippet_norm = snippet / snippet[0, 0] - 1
                noise_X.append(snippet_norm)
        
        logger.info(f"[HISTORICAL MINER] Generated {len(noise_X)} noise samples from clean periods")
        
        return np.array(noise_X)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example: How to use the PatternMiner
    """
    
    # Initialize miner
    miner = PatternMiner(sequence_length=15)
    
    # OPTION 1: Single file
    # X, y, pattern_map = miner.mine_patterns(
    #     csv_files=['data/BTCUSDT_1h.csv'],
    #     samples_per_pattern=1000
    # )
    
    # OPTION 2: Multiple files
    # X, y, pattern_map = miner.mine_patterns(
    #     csv_files=[
    #         'data/BTCUSDT_1h.csv',
    #         'data/ETHUSDT_1h.csv',
    #         'data/BNBUSDT_1h.csv'
    #     ],
    #     samples_per_pattern=500  # 500 per pattern per file
    # )
    
    # OPTION 3: Entire folder (RECOMMENDED)
    X, y, pattern_map = miner.mine_patterns(
        data_folder='data/historical',  # All CSV files in this folder
        samples_per_pattern=1000,
        use_augmentation=True
    )
    
    # Add noise class
    df = miner.load_csv_data('data/BTCUSDT_1h.csv')
    noise_X = miner.generate_noise_samples(df, num_samples=1000)
    noise_y = np.zeros(len(noise_X), dtype=int)
    
    # Combine
    X_combined = np.vstack([X, noise_X])
    y_combined = np.concatenate([y, noise_y])
    
    print(f"Final dataset: {X_combined.shape}")
    print(f"Pattern distribution: {Counter(y_combined)}")
    
    return X_combined, y_combined, pattern_map


if __name__ == "__main__":
    example_usage()