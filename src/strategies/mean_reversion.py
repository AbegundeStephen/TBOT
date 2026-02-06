"""
MULTI-TIMEFRAME Mean Reversion Strategy with 4H Context
Key improvements:
- 4H timeframe context for trend filtering
- Only takes mean reversion trades aligned with or counter to 4H extremes
- Maximum robustness with adaptive thresholds
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Production-ready mean reversion with multi-timeframe analysis
    Uses 4H context to identify favorable mean reversion setups
    """

    def __init__(self, config: dict):
        super().__init__(config, "MeanReversion")

        # Standard parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.rsi_period = config.get("rsi_period", 14)
        self.stoch_k = config.get("stoch_k", 14)
        self.stoch_d = config.get("stoch_d", 3)
        self.reversion_window = config.get("reversion_window", 3)
        self.asset = config.get("asset", "BTC")

        # Thresholds - VERY conservative
        self.rsi_overbought = config.get("rsi_overbought", 64)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.bb_lower_threshold = config.get("bb_lower_threshold", 0.35)
        self.bb_upper_threshold = config.get("bb_upper_threshold", 0.70)

        # Strict thresholds
        self.min_return_threshold = config.get("min_return_threshold", 0.003)
        self.min_score_threshold = config.get("min_conditions", 3.0)

        # 4H context parameters
        self.use_4h_context = config.get("use_4h_context", True)
        self.h4_reversion_mode = config.get(
            "h4_reversion_mode", "smart"
        )  # 'smart', 'counter', 'aligned'
        self.h4_extreme_weight = config.get(
            "h4_extreme_weight", 1.2
        )  # Boost for 4H extremes
        self.h4_trend_penalty = config.get(
            "h4_trend_penalty", 0.5
        )  # Penalty for trending 4H

        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  RSI: {self.rsi_oversold}/{self.rsi_overbought}")
        logger.info(
            f"  BB Position: {self.bb_lower_threshold}/{self.bb_upper_threshold}"
        )
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")
        logger.info(
            f"  4H Context: {self.use_4h_context} (Mode: {self.h4_reversion_mode})"
        )
        if self.use_4h_context:
            logger.info(
                f"  4H Extreme Weight: {self.h4_extreme_weight}, Trend Penalty: {self.h4_trend_penalty}"
            )

    def get_warmup_period(self) -> int:
        periods = [
            self.bb_period,
            self.rsi_period,
            self.stoch_k + self.stoch_d,
            14,  # ATR
            26 + 9 # MACD lookback
        ]
        return max(periods)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate minimal, non-redundant features"""
        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # === Core Indicators Only ===
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )

        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        df["bb_width_norm"] = (bb_upper - bb_lower) / bb_middle

        # RSI
        df["rsi"] = ta.RSI(close, timeperiod=self.rsi_period)
        df["rsi_normalized"] = (df["rsi"] - 50) / 50

        # Stochastic
        slowk, slowd = ta.STOCH(
            high,
            low,
            close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_d,
            slowd_period=self.stoch_d,
        )
        df["stoch_k"] = slowk
        df["stoch_d"] = slowd

        # MACD (Standard parameters 12, 26, 9)
        macd, macdsignal, macdhist = ta.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df["macd"] = macd
        df["macd_signal"] = macdsignal
        df["macd_hist"] = macdhist

        # Volatility (normalized)
        df["atr"] = ta.ATR(high, low, close, timeperiod=14)
        df["atr_pct"] = df["atr"] / close

        # Simple momentum
        df["roc_5"] = ta.ROC(close, timeperiod=5)

        # ADX for trend strength
        df["adx"] = ta.ADX(high, low, close, timeperiod=14)

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
            "bb_position",
            "rsi",
            "stoch_k",
            "adx",
            "bb_width_norm",
            "atr_pct",
            "close",
        ]

        # Create aligned dataframe
        df_4h_aligned = pd.DataFrame(index=df_1h.index)

        for feature in h4_features:
            if feature in df_4h.columns:
                df_4h_aligned[f"h4_{feature}"] = df_4h[feature].reindex(
                    df_1h.index, method="ffill"
                )

        return df_4h_aligned

    def _calculate_4h_reversion_context(
        self, df_4h_aligned: pd.DataFrame, idx: int
    ) -> dict:
        """
        Calculate 4H mean reversion context
        Returns: {
            'is_extreme_oversold': bool,
            'is_extreme_overbought': bool,
            'is_trending': bool,
            'trend_direction': int (-1, 0, 1),
            'reversion_score': float
        }
        """
        if df_4h_aligned is None or idx >= len(df_4h_aligned):
            return {
                "is_extreme_oversold": False,
                "is_extreme_overbought": False,
                "is_trending": False,
                "trend_direction": 0,
                "reversion_score": 0.0,
            }

        row = df_4h_aligned.iloc[idx]

        # Check if we have valid data
        if pd.isna(row.get("h4_bb_position")) or pd.isna(row.get("h4_rsi")):
            return {
                "is_extreme_oversold": False,
                "is_extreme_overbought": False,
                "is_trending": False,
                "trend_direction": 0,
                "reversion_score": 0.0,
            }

        context = {
            "is_extreme_oversold": False,
            "is_extreme_overbought": False,
            "is_trending": False,
            "trend_direction": 0,
            "reversion_score": 0.0,
        }

        # 1. Check for extreme conditions on 4H
        bb_pos = row["h4_bb_position"]
        rsi = row["h4_rsi"]
        stoch = row.get("h4_stoch_k", 50)
        adx = row.get("h4_adx", 0)

        # Extreme oversold on 4H
        oversold_count = 0
        if bb_pos < 0.15:
            oversold_count += 2
        elif bb_pos < 0.25:
            oversold_count += 1

        if rsi < 30:
            oversold_count += 2
        elif rsi < 35:
            oversold_count += 1

        if stoch < 20:
            oversold_count += 1

        if oversold_count >= 2:
            context["is_extreme_oversold"] = True
            context["reversion_score"] = oversold_count

        # Extreme overbought on 4H
        overbought_count = 0
        if bb_pos > 0.85:
            overbought_count += 2
        elif bb_pos > 0.75:
            overbought_count += 1

        if rsi > 70:
            overbought_count += 2
        elif rsi > 65:
            overbought_count += 1

        if stoch > 80:
            overbought_count += 1

        if overbought_count >= 2:
            context["is_extreme_overbought"] = True
            context["reversion_score"] = overbought_count

        # 2. Check if 4H is trending (bad for mean reversion)
        if adx > 25:
            context["is_trending"] = True

            # Determine trend direction
            if bb_pos > 0.6 and rsi > 55:
                context["trend_direction"] = 1  # Uptrend
            elif bb_pos < 0.4 and rsi < 45:
                context["trend_direction"] = -1  # Downtrend

        return context

    def generate_labels(
        self, df: pd.DataFrame, df_4h: pd.DataFrame = None
    ) -> pd.Series:
        """
        MULTI-TIMEFRAME label generation with 4H context

        Mean Reversion Logic with 4H:
        - BEST: 4H extreme + 1H counter-extreme (e.g., 4H oversold + 1H overbought)
        - GOOD: 4H ranging + 1H extreme
        - AVOID: 4H trending strongly

        Parameters:
        -----------
        df : pd.DataFrame
            Primary timeframe data (1H) with features
        df_4h : pd.DataFrame, optional
            Higher timeframe data (4H) with features

        Returns:
        --------
        pd.Series
            Labels: -1 (SELL), 0 (HOLD), 1 (BUY)
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)

        close = df["close"].values
        bb_position = df["bb_position"].values
        rsi = df["rsi"].values
        stoch_k = df["stoch_k"].values
        bb_width = df["bb_width_norm"].values
        atr_pct = df["atr_pct"].values

        # Align 4H data if provided
        df_4h_aligned = None
        if self.use_4h_context and df_4h is not None:
            # Generate features for 4H if not already done
            if "bb_position" not in df_4h.columns:
                df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df, df_4h)
            logger.info(f"[{self.name}] 4H context aligned successfully")

        # Calculate adaptive thresholds based on market regime
        median_vol = np.nanmedian(atr_pct)
        high_vol_threshold = np.nanpercentile(atr_pct, 75)

        signal_count = {"buy": 0, "sell": 0, "hold": 0}
        filtered_by_4h = 0
        boosted_by_4h = 0

        for i in range(len(df) - self.reversion_window - 1):
            # Skip if ANY missing data
            if (
                pd.isna(bb_position[i])
                or pd.isna(rsi[i])
                or pd.isna(stoch_k[i])
                or pd.isna(bb_width[i])
                or pd.isna(atr_pct[i])
            ):
                continue

            # Skip if extreme volatility
            if atr_pct[i] > high_vol_threshold:
                continue

            current_close = close[i]
            future_closes = close[i + 1 : i + 1 + self.reversion_window]

            if len(future_closes) < self.reversion_window:
                continue

            # Calculate future returns
            max_future = np.max(future_closes)
            min_future = np.min(future_closes)
            avg_future = np.mean(future_closes)

            future_return_up = (max_future - current_close) / current_close
            future_return_down = (current_close - min_future) / current_close
            avg_return = (avg_future - current_close) / current_close

            # Adaptive threshold based on volatility
            vol_mult = max(0.7, min(1.5, atr_pct[i] / median_vol))
            adj_min_return = self.min_return_threshold * vol_mult

            if self.asset == "GOLD":
                adj_min_return *= 0.7

            # === GET 4H CONTEXT ===
            h4_context = {
                "is_extreme_oversold": False,
                "is_extreme_overbought": False,
                "is_trending": False,
                "trend_direction": 0,
                "reversion_score": 0.0,
            }

            if df_4h_aligned is not None:
                h4_context = self._calculate_4h_reversion_context(df_4h_aligned, i)

            # === CALCULATE 1H BUY SIGNAL ===
            buy_score = 0.0

            # 1. Extreme oversold on 1H
            if bb_position[i] < 0.10:
                buy_score += 2.5
            elif bb_position[i] < 0.15:
                buy_score += 1.5
            elif bb_position[i] < self.bb_lower_threshold:
                buy_score += 0.5

            # 2. RSI extreme
            if rsi[i] < 25:
                buy_score += 2.0
            elif rsi[i] < 30:
                buy_score += 1.0
            elif rsi[i] < self.rsi_oversold:
                buy_score += 0.3

            # 3. Stochastic confirmation
            if stoch_k[i] < 15:
                buy_score += 1.5
            elif stoch_k[i] < 20:
                buy_score += 0.5

            # 4. NOT in squeeze
            if bb_width[i] > 0.015:
                buy_score += 1.0

            # 5. Require consistent upward movement
            direction_score = 0
            for j in range(1, min(4, len(future_closes))):
                if future_closes[j] > future_closes[j - 1]:
                    direction_score += 1

            if direction_score >= 2:
                buy_score += 1.0

            # === APPLY 4H CONTEXT TO BUY SIGNAL ===
            if df_4h_aligned is not None:
                # BEST CASE: 4H is also oversold (double bottom opportunity)
                if h4_context["is_extreme_oversold"]:
                    buy_score += self.h4_extreme_weight
                    boosted_by_4h += 1

                # GOOD CASE: 4H is overbought (expecting reversion from high to low)
                elif h4_context["is_extreme_overbought"]:
                    buy_score += self.h4_extreme_weight * 0.5  # Smaller boost

                # BAD CASE: 4H is in strong downtrend (avoid catching falling knife)
                elif h4_context["is_trending"] and h4_context["trend_direction"] == -1:
                    buy_score -= self.h4_trend_penalty
                    if self.h4_reversion_mode == "smart":
                        filtered_by_4h += 1
                        continue  # Skip this signal

                # NEUTRAL: 4H is ranging or trending up (allow signal)

            # Generate BUY label
            if (
                buy_score >= self.min_score_threshold
                and future_return_up > adj_min_return
                and avg_return > adj_min_return * 0.5
            ):
                labels.iloc[i] = 1
                signal_count["buy"] += 1

            # === CALCULATE 1H SELL SIGNAL ===
            sell_score = 0.0

            # 1. Extreme overbought on 1H
            if bb_position[i] > 0.90:
                sell_score += 2.5
            elif bb_position[i] > 0.85:
                sell_score += 1.5
            elif bb_position[i] > self.bb_upper_threshold:
                sell_score += 0.5

            # 2. RSI extreme
            if rsi[i] > 75:
                sell_score += 2.0
            elif rsi[i] > 70:
                sell_score += 1.0
            elif rsi[i] > self.rsi_overbought:
                sell_score += 0.3

            # 3. Stochastic confirmation
            if stoch_k[i] > 85:
                sell_score += 1.5
            elif stoch_k[i] > 80:
                sell_score += 0.5

            # 4. NOT in squeeze
            if bb_width[i] > 0.015:
                sell_score += 1.0

            # 5. Require consistent downward movement
            direction_score = 0
            for j in range(1, min(4, len(future_closes))):
                if future_closes[j] < future_closes[j - 1]:
                    direction_score += 1

            if direction_score >= 2:
                sell_score += 1.0

            # === APPLY 4H CONTEXT TO SELL SIGNAL ===
            if df_4h_aligned is not None:
                # BEST CASE: 4H is also overbought (double top opportunity)
                if h4_context["is_extreme_overbought"]:
                    sell_score += self.h4_extreme_weight
                    boosted_by_4h += 1

                # GOOD CASE: 4H is oversold (expecting reversion from low to high)
                elif h4_context["is_extreme_oversold"]:
                    sell_score += self.h4_extreme_weight * 0.5

                # BAD CASE: 4H is in strong uptrend (avoid selling into strength)
                elif h4_context["is_trending"] and h4_context["trend_direction"] == 1:
                    sell_score -= self.h4_trend_penalty
                    if self.h4_reversion_mode == "smart":
                        filtered_by_4h += 1
                        continue  # Skip this signal

            # Generate SELL label
            if (
                sell_score >= self.min_score_threshold
                and future_return_down > adj_min_return
                and avg_return < -adj_min_return * 0.5
            ):
                labels.iloc[i] = -1
                signal_count["sell"] += 1

        # Remove labels from last N bars
        labels.iloc[-self.reversion_window :] = 0

        # === Detailed logging ===
        signal_count["hold"] = len(labels) - signal_count["buy"] - signal_count["sell"]
        total = len(labels)

        logger.info(f"[{self.name}] Label distribution:")
        logger.info(
            f"  SELL: {signal_count['sell']:>5} ({signal_count['sell']/total*100:>5.2f}%)"
        )
        logger.info(
            f"  HOLD: {signal_count['hold']:>5} ({signal_count['hold']/total*100:>5.2f}%)"
        )
        logger.info(
            f"  BUY:  {signal_count['buy']:>5} ({signal_count['buy']/total*100:>5.2f}%)"
        )

        if df_4h_aligned is not None:
            logger.info(f"[{self.name}] 4H Context Impact:")
            logger.info(f"  Filtered: {filtered_by_4h} signals")
            logger.info(f"  Boosted: {boosted_by_4h} signals")

        buy_pct = signal_count["buy"] / total * 100
        sell_pct = signal_count["sell"] / total * 100
        total_signals = buy_pct + sell_pct

        if total_signals < 5:
            logger.warning(
                f"  ⚠ Very few signals ({total_signals:.1f}%) - consider loosening"
            )
        elif total_signals > 69:
            logger.warning(
                f"  ⚠ Too many signals ({total_signals:.1f}%) - tighten thresholds"
            )
        else:
            logger.info(f"  ✓ Good signal rate: {total_signals:.1f}%")

        return labels
