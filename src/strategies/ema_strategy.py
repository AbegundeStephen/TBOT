# src/strategies/ema_strategy.py
"""
EMA Crossover Strategy - IMPROVED Signal Generation
Now generates realistic buy/sell signals based on multiple conditions
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class EMAStrategy(BaseStrategy):
    """
    EMA Crossover Strategy with ML-based signal generation
    FIXED: Now generates sufficient trading signals
    """

    def __init__(self, config: dict):
        super().__init__(config, "EMA")

        # EMA periods
        self.fast_period = config.get("ema_fast", 50)
        self.slow_period = config.get("ema_slow", 200)

        # Signal thresholds from config
        self.min_distance_pct = config.get("min_distance_pct", 0.05)  # 0.05% from config
        self.min_return_threshold = config.get("min_return_threshold", 0.0001)  # 0.01% from config
        self.min_score_threshold = config.get("min_conditions", 1)  # 1 from config

        # Filters
        self.use_price_confirmation = config.get("use_price_confirmation", False)
        self.use_volume_filter = config.get("use_volume_filter", False)
        self.volume_multiplier = config.get("volume_multiplier", 1.2)

        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  EMA Fast: {self.fast_period}, Slow: {self.slow_period}")
        logger.info(f"  Min Distance: {self.min_distance_pct}%")
        logger.info(f"  Price Confirmation: {self.use_price_confirmation}")
        logger.info(f"  Min Return: {self.min_return_threshold:.4%}")

    def get_warmup_period(self) -> int:
        """Need enough data for slow EMA"""
        return self.slow_period + 20

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA-based features, ensuring all data is numeric."""
        df = df.copy()

        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype not in ["float64", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN values after conversion
        df = df.dropna()

        # Ensure close, high, low are numeric arrays
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")

        # === CORE EMA INDICATORS ===
        df["ema_fast"] = ta.EMA(close, timeperiod=self.fast_period)
        df["ema_slow"] = ta.EMA(close, timeperiod=self.slow_period)

        # EMA relationships
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        df["ema_diff_pct"] = (df["ema_diff"] / df["ema_slow"]) * 100

        # Price vs EMAs
        df["price_vs_fast"] = (close - df["ema_fast"]) / df["ema_fast"]
        df["price_vs_slow"] = (close - df["ema_slow"]) / df["ema_slow"]
        df["price_above_fast"] = (close > df["ema_fast"]).astype(int)
        df["price_above_slow"] = (close > df["ema_slow"]).astype(int)

        # === CROSSOVER DETECTION ===
        df["ema_cross"] = 0
        ema_diff_shift = df["ema_diff"].shift(1)

        # Golden Cross (Fast crosses above Slow)
        df.loc[(df["ema_diff"] > 0) & (ema_diff_shift <= 0), "ema_cross"] = 1

        # Death Cross (Fast crosses below Slow)
        df.loc[(df["ema_diff"] < 0) & (ema_diff_shift >= 0), "ema_cross"] = -1

        # Bars since last crossover (fixed dtype issue)
        df["bars_since_cross"] = 0
        cross_indices = df[df["ema_cross"] != 0].index
        for i in df.index:
            recent_crosses = cross_indices[cross_indices < i]
            if len(recent_crosses) > 0:
                # Convert timedelta to integer (number of periods)
                bars_back = len(df.loc[recent_crosses[-1]:i]) - 1
                df.at[i, "bars_since_cross"] = bars_back

        # === TREND STRENGTH ===
        df["ema_trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
        df["trend_strength"] = np.abs(df["ema_diff_pct"])

        # EMA slopes (rate of change)
        df["ema_fast_slope"] = df["ema_fast"].diff(5) / df["ema_fast"]
        df["ema_slow_slope"] = df["ema_slow"].diff(10) / df["ema_slow"]

        # === ADDITIONAL CONFIRMATION INDICATORS ===
        # Volume
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df["volume_ma"] = ta.SMA(
                df["volume"].values.astype("float64"), timeperiod=20
            )
            df["volume_ratio"] = df["volume"] / df["volume_ma"]
            df["high_volume"] = (df["volume_ratio"] > self.volume_multiplier).astype(
                int
            )
        else:
            df["volume_ratio"] = 1.0
            df["high_volume"] = 1

        # MACD (complementary momentum)
        macd, macd_signal, macd_hist = ta.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        df["macd_aligned"] = np.where(
            ((df["ema_trend"] == 1) & (macd_hist > 0)) |
            ((df["ema_trend"] == -1) & (macd_hist < 0)),
            1,
            0,
        )

        # RSI (for overbought/oversold context)
        df["rsi"] = ta.RSI(close, timeperiod=14)

        # ADX (trend strength)
        df["adx"] = ta.ADX(high, low, close, timeperiod=14)
        df["strong_trend"] = (df["adx"] > 25).astype(int)

        return df

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        IMPROVED: Generate more realistic trading signals
        Uses multiple conditions instead of just crossovers
        """
        df = df.copy()

        # Check initial data
        logger.info(f"[{self.name}] Starting label generation with {len(df)} rows")

        # Don't drop all NaN rows - only drop if specific required columns are NaN
        required_cols = ['close', 'ema_fast', 'ema_slow', 'ema_diff_pct', 'ema_cross', 'rsi', 'macd_hist']
        df = df.dropna(subset=required_cols)
        
        logger.info(f"[{self.name}] After filtering NaN in required columns: {len(df)} rows")

        if len(df) == 0:
            logger.error(f"[{self.name}] No valid rows after filtering NaN!")
            return pd.Series(0, index=pd.Index([]))

        # Extract values
        close = df['close'].values
        ema_fast = df['ema_fast'].values
        ema_slow = df['ema_slow'].values
        ema_diff_pct = df['ema_diff_pct'].values
        ema_cross = df['ema_cross'].values.astype(int)
        rsi = df['rsi'].values
        macd_hist = df['macd_hist'].values

        labels = pd.Series(0, index=df.index)
        lookforward = 3  # Shorter lookforward for more signals
        
        logger.info(f"[{self.name}] Analyzing {len(df) - lookforward} bars for signals...")

        for i in range(len(df) - lookforward):
            # Skip if essential values are NaN
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(close[i]):
                continue

            # Calculate future return
            future_closes = close[i+1:i+1+lookforward]
            if len(future_closes) == 0:
                continue

            future_return = (np.mean(future_closes) - close[i]) / close[i]

            # === BULLISH CONDITIONS (Score-Based) ===
            bullish_score = 0
            
            # 1. Strong Golden Cross
            if ema_cross[i] == 1:
                bullish_score += 3
            
            # 2. Established Uptrend (Fast > Slow with meaningful distance)
            if ema_fast[i] > ema_slow[i] and ema_diff_pct[i] > self.min_distance_pct:
                bullish_score += 2
            
            # 3. Price above both EMAs (bullish positioning)
            if close[i] > ema_fast[i] and close[i] > ema_slow[i]:
                bullish_score += 1
            
            # 4. MACD confirmation (bullish momentum)
            if not np.isnan(macd_hist[i]) and macd_hist[i] > 0:
                bullish_score += 1
            
            # 5. RSI not overbought (room to grow)
            if not np.isnan(rsi[i]) and 40 < rsi[i] < 70:
                bullish_score += 1
            
            # 6. Future return positive (backtest validation)
            if future_return > self.min_return_threshold:
                bullish_score += 2

            # === BEARISH CONDITIONS (Score-Based) ===
            bearish_score = 0
            
            # 1. Strong Death Cross
            if ema_cross[i] == -1:
                bearish_score += 3
            
            # 2. Established Downtrend (Fast < Slow with meaningful distance)
            if ema_fast[i] < ema_slow[i] and ema_diff_pct[i] < -self.min_distance_pct:
                bearish_score += 2
            
            # 3. Price below both EMAs (bearish positioning)
            if close[i] < ema_fast[i] and close[i] < ema_slow[i]:
                bearish_score += 1
            
            # 4. MACD confirmation (bearish momentum)
            if not np.isnan(macd_hist[i]) and macd_hist[i] < 0:
                bearish_score += 1
            
            # 5. RSI not oversold (room to fall)
            if not np.isnan(rsi[i]) and 30 < rsi[i] < 60:
                bearish_score += 1
            
            # 6. Future return negative (backtest validation)
            if future_return < -self.min_return_threshold:
                bearish_score += 2

            # === ASSIGN LABELS BASED ON SCORES ===
            # Require at least min_score_threshold points (default 1)
            if bullish_score >= self.min_score_threshold and bullish_score > bearish_score:
                labels.iloc[i] = 1
            elif bearish_score >= self.min_score_threshold and bearish_score > bullish_score:
                labels.iloc[i] = -1
            # else: HOLD (0)

        # Remove labels from last N bars (no future data)
        labels.iloc[-lookforward:] = 0

        # Count actual signals generated before distribution
        signals_generated = (labels != 0).sum()
        logger.info(f"[{self.name}] Generated {signals_generated} total signals from {len(df)} bars")

        # Log distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(labels)

        logger.info(f"[{self.name}] Label distribution:")
        logger.info(f"  SELL: {dist.get(-1, 0):>5} ({dist.get(-1, 0)/total*100 if total > 0 else 0:>5.2f}%)")
        logger.info(f"  HOLD: {dist.get(0, 0):>5} ({dist.get(0, 0)/total*100 if total > 0 else 0:>5.2f}%)")
        logger.info(f"  BUY:  {dist.get(1, 0):>5} ({dist.get(1, 0)/total*100 if total > 0 else 0:>5.2f}%)")

        # Warning if still too imbalanced
        buy_pct = dist.get(1, 0) / total * 100 if total > 0 else 0
        sell_pct = dist.get(-1, 0) / total * 100 if total > 0 else 0

        if total > 0 and (buy_pct < 2 or sell_pct < 2):
            logger.warning(f"  ⚠ Low signal rate detected (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)")
            logger.warning(f"  This is still trainable, but consider lowering min_return_threshold")
        elif total > 0:
            logger.info(f"  ✓ Healthy signal distribution (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)")

        return labels