"""
EMA Crossover Strategy - FIXED Signal Generation
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
    OPTIMIZED: Now generates higher-quality signals with regime detection
    """

    def __init__(self, config: dict):
        super().__init__(config, "EMA")
        # EMA periods
        self.fast_period = config.get("ema_fast", 50)
        self.slow_period = config.get("ema_slow", 200)
        # Signal thresholds from config
        self.min_distance_pct = config.get("min_distance_pct", 0.05)
        self.min_return_threshold = config.get("min_return_threshold", 0.0001)
        self.min_score_threshold = config.get("min_conditions", 1)
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
        """Need enough data for slow EMA + other indicators"""
        return max(self.slow_period, 26 + 9) + 50

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA-based features, ensuring all data is numeric."""
        if len(df) < self.get_warmup_period():
            logger.debug(f"[{self.name}] Insufficient data: {len(df)} < {self.get_warmup_period()}")
            empty_df = df.copy()
            for col in ['ema_fast', 'ema_slow', 'ema_diff', 'ema_diff_pct', 'ema_cross']:
                empty_df[col] = 0
            return empty_df

        df = df.copy()
        # Ensure all columns are numeric first
        for col in df.columns:
            if df[col].dtype not in ["float64", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle missing volume
        if "volume" not in df.columns:
            df["volume"] = 1.0

        # FIXED: Replace deprecated fillna(method='ffill') with ffill()
        df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0)
        df["high"] = pd.to_numeric(df["high"], errors="coerce").ffill().fillna(df["close"])
        df["low"] = pd.to_numeric(df["low"], errors="coerce").ffill().fillna(df["close"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").ffill().fillna(1.0)

        # Now extract the processed values
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")
        volume = df["volume"].values.astype("float64")

        # Check for all NaN
        if np.all(np.isnan(close)) or len(close) == 0:
            logger.warning(f"[{self.name}] All close prices are NaN, returning zero features")
            df['ema_fast'] = 0
            df['ema_slow'] = 0
            df['ema_diff'] = 0
            df['ema_diff_pct'] = 0
            df['ema_cross'] = 0
            return df

        # Fill NaN values with forward fill then backward fill
        close = np.nan_to_num(close, nan=np.nanmean(close))
        high = np.nan_to_num(high, nan=np.nanmean(high))
        low = np.nan_to_num(low, nan=np.nanmean(low))
        volume = np.nan_to_num(volume, nan=np.nanmean(volume))

        # === CORE EMA INDICATORS ===
        try:
            ema_fast_values = ta.EMA(close, timeperiod=self.fast_period)
            ema_slow_values = ta.EMA(close, timeperiod=self.slow_period)

            df["ema_fast"] = ema_fast_values
            df["ema_slow"] = ema_slow_values

            # FIXED: Use proper forward fill and backfill for NaN values
            df["ema_fast"] = df["ema_fast"].ffill().bfill()
            df["ema_slow"] = df["ema_slow"].ffill().bfill()

            # If still NaN after ffill/bfill, use close prices
            if df["ema_fast"].isna().any():
                df["ema_fast"] = df["ema_fast"].fillna(df["close"])
            if df["ema_slow"].isna().any():
                df["ema_slow"] = df["ema_slow"].fillna(df["close"])
        except Exception as e:
            logger.error(f"[{self.name}] Error calculating EMAs: {e}")
            df["ema_fast"] = df["close"]
            df["ema_slow"] = df["close"]

        # EMA relationships
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        df["ema_diff_pct"] = np.where(df["ema_slow"] != 0, (df["ema_diff"] / df["ema_slow"]) * 100, 0)

        # Price vs EMAs
        df["price_vs_fast"] = np.where(df["ema_fast"] != 0, (close - df["ema_fast"]) / df["ema_fast"], 0)
        df["price_vs_slow"] = np.where(df["ema_slow"] != 0, (close - df["ema_slow"]) / df["ema_slow"], 0)
        df["price_above_fast"] = (close > df["ema_fast"]).astype(int)
        df["price_above_slow"] = (close > df["ema_slow"]).astype(int)

        # === CROSSOVER DETECTION ===
        df["ema_cross"] = 0
        ema_diff_shift = df["ema_diff"].shift(1).fillna(0)

        # Golden Cross (Fast crosses above Slow)
        df.loc[(df["ema_diff"] > 0) & (ema_diff_shift <= 0), "ema_cross"] = 1
        # Death Cross (Fast crosses below Slow)
        df.loc[(df["ema_diff"] < 0) & (ema_diff_shift >= 0), "ema_cross"] = -1

        # Bars since last crossover
        df["bars_since_cross"] = 0
        cross_mask = df["ema_cross"] != 0
        if cross_mask.any():
            cross_indices = np.where(cross_mask)[0]
            for i in range(len(df)):
                recent_crosses = cross_indices[cross_indices <= i]
                if len(recent_crosses) > 0:
                    df.iloc[i, df.columns.get_loc("bars_since_cross")] = i - recent_crosses[-1]

        # === TREND STRENGTH ===
        df["ema_trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
        df["trend_strength"] = np.abs(df["ema_diff_pct"])

        # EMA slopes (rate of change)
        df["ema_fast_slope"] = df["ema_fast"].diff(5)
        df["ema_fast_slope"] = np.where(df["ema_fast"] != 0, df["ema_fast_slope"] / df["ema_fast"], 0)

        df["ema_slow_slope"] = df["ema_slow"].diff(10)
        df["ema_slow_slope"] = np.where(df["ema_slow"] != 0, df["ema_slow_slope"] / df["ema_slow"], 0)

        # === ADDITIONAL CONFIRMATION INDICATORS ===
        try:
            # FIXED: Calculate volume MA directly on numpy array, then properly fill NaN
            volume_ma_values = ta.SMA(volume, timeperiod=20)
            df["volume_ma"] = volume_ma_values
            df["volume_ma"] = df["volume_ma"].ffill().bfill()

            # If still NaN, use current volume
            if df["volume_ma"].isna().any():
                df["volume_ma"] = df["volume_ma"].fillna(df["volume"])

            df["volume_ratio"] = np.where(df["volume_ma"] != 0, df["volume"] / df["volume_ma"], 1.0)
            df["high_volume"] = (df["volume_ratio"] > self.volume_multiplier).astype(int)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating volume indicators: {e}")
            df["volume_ma"] = df["volume"]
            df["volume_ratio"] = 1.0
            df["high_volume"] = 1

        # MACD (complementary momentum)
        try:
            macd, macd_signal, macd_hist = ta.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist

            # Fill NaN values
            df["macd"] = df["macd"].fillna(0)
            df["macd_signal"] = df["macd_signal"].fillna(0)
            df["macd_hist"] = df["macd_hist"].fillna(0)

            df["macd_aligned"] = np.where(
                ((df["ema_trend"] == 1) & (macd_hist > 0)) |
                ((df["ema_trend"] == -1) & (macd_hist < 0)),
                1,
                0,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating MACD: {e}")
            df["macd"] = 0
            df["macd_signal"] = 0
            df["macd_hist"] = 0
            df["macd_aligned"] = 0

        # RSI (for overbought/oversold context)
        try:
            df["rsi"] = ta.RSI(close, timeperiod=14)
            df["rsi"] = df["rsi"].fillna(50)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating RSI: {e}")
            df["rsi"] = 50

        # ADX (trend strength)
        try:
            df["adx"] = ta.ADX(high, low, close, timeperiod=14)
            df["adx"] = df["adx"].fillna(20)
            df["strong_trend"] = (df["adx"] > 25).astype(int)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating ADX: {e}")
            df["adx"] = 20
            df["strong_trend"] = 0

        # Final NaN cleanup
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Log if any NaN still exists
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"[{self.name}] Still has NaN in columns: {nan_columns}")
            df[nan_columns] = df[nan_columns].fillna(0)

        return df

    def generate_signal(self, df: pd.DataFrame) -> tuple:
        """
        Generate real-time signal based on current EMA conditions
        OPTIMIZED: Now includes MACD, RSI, and volume confirmation
        Returns: (signal, confidence)
        """
        if len(df) < self.get_warmup_period():
            return 0, 0.0

        try:
            features_df = self.generate_features(df.tail(max(self.fast_period, self.slow_period) + 50))

            if features_df.empty or len(features_df) == 0:
                return 0, 0.0

            latest = features_df.iloc[-1]

            # Check for NaN in critical features
            critical_features = ['ema_fast', 'ema_slow', 'ema_diff', 'ema_cross', 'macd_hist', 'rsi']
            for feat in critical_features:
                if pd.isna(latest[feat]) or np.isnan(latest[feat]):
                    logger.debug(f"[{self.name}] NaN detected in {feat}, returning HOLD")
                    return 0, 0.0

            ema_cross = latest['ema_cross']
            trend_strength = latest['trend_strength']
            macd_aligned = latest['macd_aligned']
            rsi = latest['rsi']
            high_volume = latest['high_volume']

            # Calculate confidence score
            confidence = min(1.0, trend_strength / 5.0)

            # Add MACD alignment to confidence
            if macd_aligned == 1:
                confidence += 0.15

            # Add RSI confirmation to confidence
            if (ema_cross == 1 and 40 < rsi < 70) or (ema_cross == -1 and 30 < rsi < 60):
                confidence += 0.10

            # Add volume confirmation to confidence
            if high_volume == 1:
                confidence += 0.10

            # Cap confidence at 1.0
            confidence = min(1.0, confidence)

            if ema_cross == 1:  # Golden Cross
                return 1, confidence
            elif ema_cross == -1:  # Death Cross
                return -1, confidence
            else:
                if trend_strength > 1.0:
                    if latest['ema_fast'] > latest['ema_slow']:
                        return 1, confidence * 0.7
                    else:
                        return -1, confidence * 0.7

            return 0, 0.0

        except Exception as e:
            logger.error(f"[{self.name}] Error in generate_signal: {e}")
            return 0, 0.0

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        IMPROVED: Generate more realistic trading signals
        Uses multiple conditions instead of just crossovers
        """
        df = df.copy()
        logger.info(f"[{self.name}] Starting label generation with {len(df)} rows")

        required_cols = ['close', 'ema_fast', 'ema_slow', 'ema_diff_pct', 'ema_cross', 'rsi', 'macd_hist', 'high_volume']
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
        high_volume = df['high_volume'].values

        labels = pd.Series(0, index=df.index)
        lookforward = 3

        logger.info(f"[{self.name}] Analyzing {len(df) - lookforward} bars for signals...")

        for i in range(len(df) - lookforward):
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(close[i]):
                continue

            future_closes = close[i+1:i+1+lookforward]
            if len(future_closes) == 0:
                continue

            future_return = (np.mean(future_closes) - close[i]) / close[i]

            # === BULLISH CONDITIONS ===
            bullish_score = 0

            if ema_cross[i] == 1:
                bullish_score += 3

            if ema_fast[i] > ema_slow[i] and ema_diff_pct[i] > self.min_distance_pct:
                bullish_score += 2

            if close[i] > ema_fast[i] and close[i] > ema_slow[i]:
                bullish_score += 1

            if not np.isnan(macd_hist[i]) and macd_hist[i] > 0:
                bullish_score += 1

            if not np.isnan(rsi[i]) and 40 < rsi[i] < 70:
                bullish_score += 1

            if high_volume[i] == 1:
                bullish_score += 1

            if future_return > self.min_return_threshold:
                bullish_score += 2

            # === BEARISH CONDITIONS ===
            bearish_score = 0

            if ema_cross[i] == -1:
                bearish_score += 3

            if ema_fast[i] < ema_slow[i] and ema_diff_pct[i] < -self.min_distance_pct:
                bearish_score += 2

            if close[i] < ema_fast[i] and close[i] < ema_slow[i]:
                bearish_score += 1

            if not np.isnan(macd_hist[i]) and macd_hist[i] < 0:
                bearish_score += 1

            if not np.isnan(rsi[i]) and 30 < rsi[i] < 60:
                bearish_score += 1

            if high_volume[i] == 1:
                bearish_score += 1

            if future_return < -self.min_return_threshold:
                bearish_score += 2

            # === ASSIGN LABELS ===
            if bullish_score >= self.min_score_threshold and bullish_score > bearish_score:
                labels.iloc[i] = 1
            elif bearish_score >= self.min_score_threshold and bearish_score > bullish_score:
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        # Remove labels from last N bars
        labels.iloc[-lookforward:] = 0

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

        buy_pct = dist.get(1, 0) / total * 100 if total > 0 else 0
        sell_pct = dist.get(-1, 0) / total * 100 if total > 0 else 0

        if total > 0 and (buy_pct < 2 or sell_pct < 2):
            logger.warning(f"  ⚠ Low signal rate detected (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)")
            logger.warning(f"  This is still trainable, but consider lowering min_return_threshold")
        elif total > 0:
            logger.info(f"  ✓ Healthy signal distribution (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)")

        return labels
