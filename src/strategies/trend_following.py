# src/strategies/trend_following.py
"""
MULTI-TIMEFRAME Trend Following Strategy with 4H Context
Key improvements:
- 4H timeframe context for trend filtering
- Multi-tier signal strength (weak/medium/strong trends)
- Adaptive thresholds based on volatility
- Better balance between precision and signal generation
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
     trend following with multi-timeframe analysis
    Uses 4H context to filter and validate 1H signals
    """

    def __init__(self, config: dict):
        super().__init__(config, "TrendFollowing")

        # Moving average periods
        self.fast_ma = config.get("fast_ma", 20)
        self.slow_ma = config.get("slow_ma", 50)

        # MACD parameters
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)

        # ADX parameters
        self.adx_period = config.get("adx_period", 14)
        self.adx_threshold = config.get("adx_threshold", 15)
        self.require_adx = config.get("require_adx", False)

        # 4H context parameters
        self.use_4h_context = config.get("use_4h_context", True)
        self.require_4h_alignment = config.get("require_4h_alignment", True)
        self.h4_trend_weight = config.get(
            "h4_trend_weight", 1.5
        )  # Bonus points for 4H alignment
        self.h4_counter_penalty = config.get(
            "h4_counter_penalty", 2.0
        )  # Penalty for counter-trend

        # Return thresholds
        self.min_return_threshold = config.get("min_return_threshold", 0.001)

        # Score threshold
        self.min_score_threshold = config.get("min_conditions", 1.5)

        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  Fast MA: {self.fast_ma}, Slow MA: {self.slow_ma}")
        logger.info(
            f"  ADX Threshold: {self.adx_threshold} (Required: {self.require_adx})"
        )
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")
        logger.info(
            f"  4H Context: {self.use_4h_context} (Required: {self.require_4h_alignment})"
        )
        if self.use_4h_context:
            logger.info(
                f"  4H Trend Weight: {self.h4_trend_weight}, Counter Penalty: {self.h4_counter_penalty}"
            )

    def get_warmup_period(self) -> int:
        periods = [
            self.slow_ma,
            self.macd_slow + self.macd_signal,
            self.adx_period,
        ]
        return max(periods)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following features"""
        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Moving Averages
        df["sma_fast"] = ta.SMA(close, timeperiod=self.fast_ma)
        df["sma_slow"] = ta.SMA(close, timeperiod=self.slow_ma)
        df["ema_fast"] = ta.EMA(close, timeperiod=self.fast_ma)
        df["ema_slow"] = ta.EMA(close, timeperiod=self.slow_ma)

        # MA relationships
        df["ma_diff"] = df["sma_fast"] - df["sma_slow"]
        df["ma_diff_pct"] = (df["sma_fast"] - df["sma_slow"]) / df["sma_slow"]

        # MA crossovers
        df["ma_cross"] = 0
        ma_diff_shift = df["ma_diff"].shift(1)
        df.loc[(df["ma_diff"] > 0) & (ma_diff_shift <= 0), "ma_cross"] = 1
        df.loc[(df["ma_diff"] < 0) & (ma_diff_shift >= 0), "ma_cross"] = -1

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        # MACD crossovers
        df["macd_cross"] = 0
        macd_hist_shift = df["macd_hist"].shift(1)
        df.loc[(df["macd_hist"] > 0) & (macd_hist_shift <= 0), "macd_cross"] = 1
        df.loc[(df["macd_hist"] < 0) & (macd_hist_shift >= 0), "macd_cross"] = -1

        # ADX (trend strength)
        df["adx"] = ta.ADX(high, low, close, timeperiod=self.adx_period)
        df["strong_trend"] = (df["adx"] > self.adx_threshold).astype(int)

        # Directional indicators
        df["plus_di"] = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        df["minus_di"] = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        df["di_diff"] = df["plus_di"] - df["minus_di"]

        # Price position
        df["close_above_fast"] = (close > df["sma_fast"]).astype(int)
        df["close_above_slow"] = (close > df["sma_slow"]).astype(int)

        # Momentum
        df["momentum"] = ta.MOM(close, timeperiod=10)
        df["roc"] = ta.ROC(close, timeperiod=10)

        # Trend features
        df["trend_strength"] = np.abs(df["ma_diff_pct"])
        df["price_vs_ma_fast"] = (close - df["sma_fast"]) / df["sma_fast"]
        df["price_vs_ma_slow"] = (close - df["sma_slow"]) / df["sma_slow"]

        # MA slopes
        df["ma_fast_slope"] = df["sma_fast"].diff(5) / df["sma_fast"]
        df["ma_slow_slope"] = df["sma_slow"].diff(10) / df["sma_slow"]

        return df

    def _align_4h_to_1h(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Align 4H data to 1H timeframe using forward-fill
        Returns a DataFrame with same index as df_1h containing 4H context
        """
        if df_4h is None or df_4h.empty:
            return None

        # Ensure both have datetime index
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h = df_1h.set_index("timestamp")
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            df_4h = df_4h.set_index("timestamp")

        # Select 4H features to align
        h4_features = [
            "sma_fast",
            "sma_slow",
            "adx",
            "macd_hist",
            "plus_di",
            "minus_di",
            "close",
        ]

        # Create aligned dataframe
        df_4h_aligned = pd.DataFrame(index=df_1h.index)

        for feature in h4_features:
            if feature in df_4h.columns:
                # Reindex with forward fill (each 4H value applies to next 4 hours)
                df_4h_aligned[f"h4_{feature}"] = df_4h[feature].reindex(
                    df_1h.index, method="ffill"
                )

        return df_4h_aligned

    def _calculate_4h_trend_score(self, df_4h_aligned: pd.DataFrame, idx: int) -> tuple:
        """
        Calculate 4H trend direction and strength
        Returns: (bullish_score, bearish_score)
        """
        if df_4h_aligned is None or idx >= len(df_4h_aligned):
            return 0.0, 0.0

        bullish_4h = 0.0
        bearish_4h = 0.0

        row = df_4h_aligned.iloc[idx]

        # Check if we have valid data
        if pd.isna(row.get("h4_sma_fast")) or pd.isna(row.get("h4_sma_slow")):
            return 0.0, 0.0

        # 1. MA Alignment (0-2 points)
        if row["h4_sma_fast"] > row["h4_sma_slow"]:
            ma_sep = (row["h4_sma_fast"] - row["h4_sma_slow"]) / row["h4_sma_slow"]
            bullish_4h += 2.0 if ma_sep > 0.005 else 1.0
        elif row["h4_sma_fast"] < row["h4_sma_slow"]:
            ma_sep = (row["h4_sma_slow"] - row["h4_sma_fast"]) / row["h4_sma_slow"]
            bearish_4h += 2.0 if ma_sep > 0.005 else 1.0

        # 2. MACD (0-1 point)
        if not pd.isna(row.get("h4_macd_hist")):
            if row["h4_macd_hist"] > 0:
                bullish_4h += 1.0
            elif row["h4_macd_hist"] < 0:
                bearish_4h += 1.0

        # 3. Directional Indicators (0-1 point)
        if not pd.isna(row.get("h4_plus_di")) and not pd.isna(row.get("h4_minus_di")):
            if row["h4_plus_di"] > row["h4_minus_di"]:
                bullish_4h += 1.0
            elif row["h4_minus_di"] > row["h4_plus_di"]:
                bearish_4h += 1.0

        # 4. ADX Bonus (0-0.5 point)
        if not pd.isna(row.get("h4_adx")):
            if row["h4_adx"] > 25:
                # Strong trend - boost the dominant direction
                if bullish_4h > bearish_4h:
                    bullish_4h += 0.5
                elif bearish_4h > bullish_4h:
                    bearish_4h += 0.5

        return bullish_4h, bearish_4h

    def generate_labels(
        self, df: pd.DataFrame, df_4h: pd.DataFrame = None
    ) -> pd.Series:
        """
        MULTI-TIMEFRAME label generation with 4H context

        Parameters:
        -----------
        df : pd.DataFrame
            Primary timeframe data (1H) with features
        df_4h : pd.DataFrame, optional
            Higher timeframe data (4H) with features
            If provided, will be used to filter/weight signals

        Returns:
        --------
        pd.Series
            Labels: -1 (SELL), 0 (HOLD), 1 (BUY)
        """
        labels = pd.Series(0, index=df.index)
        close = df["close"].values
        sma_fast = df["sma_fast"].values
        sma_slow = df["sma_slow"].values
        adx = df["adx"].values
        macd_hist = df["macd_hist"].values
        plus_di = df["plus_di"].values
        minus_di = df["minus_di"].values

        # Align 4H data if provided
        df_4h_aligned = None
        if self.use_4h_context and df_4h is not None:
            # Generate features for 4H if not already done
            if "sma_fast" not in df_4h.columns:
                df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df, df_4h)
            logger.info(f"[{self.name}] 4H context aligned successfully")

        # Calculate rolling volatility for adaptive thresholds
        returns = pd.Series(close).pct_change()
        rolling_vol = returns.rolling(20).std()

        # Multiple lookforward periods
        short_term = 3
        medium_term = 7
        long_term = 12

        # Track filtering statistics
        filtered_by_4h = 0
        boosted_by_4h = 0

        for i in range(len(df) - long_term - 1):
            # Skip if not enough data
            if pd.isna(adx[i]) or pd.isna(rolling_vol.iloc[i]):
                continue

            # Adaptive return threshold
            vol = rolling_vol.iloc[i]
            min_return = max(self.min_return_threshold, 0.5 * vol)

            # === CALCULATE 1H TIMEFRAME SCORES ===
            bullish_score = 0.0
            bearish_score = 0.0

            # 1. MA Alignment (0-2 points)
            if sma_fast[i] > sma_slow[i]:
                ma_separation = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
                bullish_score += 2.0 if ma_separation > 0.002 else 1.0
            elif sma_fast[i] < sma_slow[i]:
                ma_separation = (sma_slow[i] - sma_fast[i]) / sma_slow[i]
                bearish_score += 2.0 if ma_separation > 0.002 else 1.0

            # 2. MACD (0-1.5 points)
            if macd_hist[i] > 0:
                macd_strength = abs(macd_hist[i])
                if macd_strength > macd_hist[max(0, i - 5) : i].std():
                    bullish_score += 1.5
                else:
                    bullish_score += 1.0
            elif macd_hist[i] < 0:
                macd_strength = abs(macd_hist[i])
                if macd_strength > macd_hist[max(0, i - 5) : i].std():
                    bearish_score += 1.5
                else:
                    bearish_score += 1.0

            # 3. Directional Indicators (0-1.5 points)
            if plus_di[i] > minus_di[i]:
                di_diff = plus_di[i] - minus_di[i]
                bullish_score += 1.5 if di_diff > 10 else 1.0
            elif minus_di[i] > plus_di[i]:
                di_diff = minus_di[i] - plus_di[i]
                bearish_score += 1.5 if di_diff > 10 else 1.0

            # 4. Price Position (0-1 point)
            if close[i] > sma_fast[i]:
                bullish_score += 1.0
            elif close[i] < sma_fast[i]:
                bearish_score += 1.0

            # 5. ADX Bonus (0-1 point)
            if adx[i] > 25:
                if bullish_score > bearish_score:
                    bullish_score += 1.0
                elif bearish_score > bullish_score:
                    bearish_score += 1.0
            elif adx[i] > self.adx_threshold:
                if bullish_score > bearish_score:
                    bullish_score += 0.5
                elif bearish_score > bullish_score:
                    bearish_score += 0.5

            # === APPLY 4H CONTEXT ===
            if df_4h_aligned is not None:
                h4_bullish, h4_bearish = self._calculate_4h_trend_score(
                    df_4h_aligned, i
                )

                # Determine 4H trend direction
                h4_trend = 0
                if h4_bullish > h4_bearish + 0.5:
                    h4_trend = 1  # Bullish
                elif h4_bearish > h4_bullish + 0.5:
                    h4_trend = -1  # Bearish

                # Apply 4H context effects
                if h4_trend == 1:  # 4H Bullish
                    bullish_score += self.h4_trend_weight  # Boost aligned signals
                    bearish_score -= self.h4_counter_penalty  # Penalize counter-trend
                    if bullish_score > bearish_score:
                        boosted_by_4h += 1
                elif h4_trend == -1:  # 4H Bearish
                    bearish_score += self.h4_trend_weight
                    bullish_score -= self.h4_counter_penalty
                    if bearish_score > bullish_score:
                        boosted_by_4h += 1

                # Hard filter if required
                if self.require_4h_alignment:
                    if h4_trend == 1 and bearish_score > bullish_score:
                        filtered_by_4h += 1
                        continue  # Skip bearish signals in 4H uptrend
                    elif h4_trend == -1 and bullish_score > bearish_score:
                        filtered_by_4h += 1
                        continue  # Skip bullish signals in 4H downtrend

            # === DETERMINE LABEL BASED ON SCORE + FORWARD RETURNS ===

            # Choose lookforward period based on trend strength
            if adx[i] > 25:
                lookforward = long_term
            elif adx[i] > self.adx_threshold:
                lookforward = medium_term
            else:
                lookforward = short_term

            # Calculate future return
            end_idx = min(i + lookforward + 1, len(close))
            future_closes = close[i + 1 : end_idx]
            if len(future_closes) == 0:
                continue

            future_return = (np.mean(future_closes) - close[i]) / close[i]

            # BUY Signal
            if bullish_score >= self.min_score_threshold and future_return > min_return:
                labels.iloc[i] = 1

            # SELL Signal
            elif (
                bearish_score >= self.min_score_threshold
                and future_return < -min_return
            ):
                labels.iloc[i] = -1

        # === LOG STATISTICS ===
        unique, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        total_labels = len(labels)

        sell_count = label_distribution.get(-1, 0)
        hold_count = label_distribution.get(0, 0)
        buy_count = label_distribution.get(1, 0)

        sell_pct = (sell_count / total_labels) * 100
        hold_pct = (hold_count / total_labels) * 100
        buy_pct = (buy_count / total_labels) * 100
        total_signals = buy_pct + sell_pct

        logger.info(f"[{self.name}] Label Distribution:")
        logger.info(f"  SELL: {sell_count:>5} ({sell_pct:>5.2f}%)")
        logger.info(f"  HOLD: {hold_count:>5} ({hold_pct:>5.2f}%)")
        logger.info(f"  BUY:  {buy_count:>5} ({buy_pct:>5.2f}%)")

        if df_4h_aligned is not None:
            logger.info(f"[{self.name}] 4H Context Impact:")
            logger.info(f"  Filtered: {filtered_by_4h} signals")
            logger.info(f"  Boosted: {boosted_by_4h} signals")

        if total_signals > 60:
            logger.warning(
                f"  ⚠ Too many signals ({total_signals:.1f}%) - tighten thresholds"
            )
        else:
            logger.info(f"  ✓ Good signal rate: {total_signals:.1f}%")

        if sell_pct < 5 or buy_pct < 5:
            logger.warning(
                f"  ⚠ Severe class imbalance! Consider lowering min_score_threshold"
            )
        elif sell_pct < 10 or buy_pct < 10:
            logger.warning(f"  ⚠ Class imbalance detected!")
        else:
            logger.info(f"  ✓ Reasonable signal distribution")

        return labels
