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

    def _find_swing_pivot(self, values: np.ndarray, direction: str, lookback: int = 60, n_bars: int = 5) -> int:
        """
        ✅ TASK 12: Find a significant swing high/low using an N-bar window.
        """
        if len(values) < lookback:
            lookback = len(values)
            
        for j in range(n_bars, lookback - n_bars):
            idx = len(values) - j - 1
            if idx < n_bars or idx + n_bars >= len(values):
                continue
            
            window = values[idx - n_bars : idx + n_bars + 1]
            
            if direction == 'low' and values[idx] == np.min(window):
                return idx
            if direction == 'high' and values[idx] == np.max(window):
                return idx
                
        return -1

    def _check_divergence(self, df: pd.DataFrame, signal: int, period: int = 60) -> bool:
        """
        Dynamic Divergence Engine (Institutional Grade)
        - 60-bar lookback window
        - n_bars=5 swing pivot detection
        """
        try:
            if len(df) < period:
                return False
                
            close = df['close'].values
            rsi = df['rsi'].values
            
            # Bullish Divergence (for Long)
            if signal == 1:
                # Current pivot low (recent)
                curr_idx = self._find_swing_pivot(close, 'low', lookback=15, n_bars=5)
                if curr_idx == -1: return False
                
                # Previous significant swing low
                prev_idx = self._find_swing_pivot(close, 'low', lookback=period, n_bars=5)
                if prev_idx == -1 or prev_idx >= curr_idx - 5: return False
                
                # Bullish: Price Lower Low, RSI Higher Low
                if close[curr_idx] < close[prev_idx] and rsi[curr_idx] > rsi[prev_idx]:
                    return True
                        
            # Bearish Divergence (for Short)
            elif signal == -1:
                # Current pivot high (recent)
                curr_idx = self._find_swing_pivot(close, 'high', lookback=15, n_bars=5)
                if curr_idx == -1: return False
                
                # Previous significant swing high
                prev_idx = self._find_swing_pivot(close, 'high', lookback=period, n_bars=5)
                if prev_idx == -1 or prev_idx >= curr_idx - 5: return False
                
                # Bearish: Price Higher High, RSI Lower High
                if close[curr_idx] > close[prev_idx] and rsi[curr_idx] < rsi[prev_idx]:
                    return True
                        
            return False
        except Exception as e:
            logger.debug(f"Divergence error: {e}")
            return False

    def generate_labels(
        self, df: pd.DataFrame, df_4h: pd.DataFrame = None, pattern_miner = None
    ) -> pd.Series:
        """
        INSTITUTIONAL Mean Reversion with 3 pillars (Phase 4)
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)

        # Generate core features
        df = self.generate_features(df)
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        bb_upper = df["bb_upper"].values
        bb_lower = df["bb_lower"].values
        rsi = df["rsi"].values
        atr = df["atr"].values
        
        # Calculate EMA 50 for "The Stretch"
        df["ema_50"] = ta.EMA(close, timeperiod=50)
        ema_50 = df["ema_50"].values

        # Align 4H data if provided
        df_4h_aligned = None
        if self.use_4h_context and df_4h is not None:
            if "bb_position" not in df_4h.columns:
                df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df, df_4h)

        signal_count = {"buy": 0, "sell": 0, "hold": 0}

        for i in range(100, len(df) - self.reversion_window - 1):
            if pd.isna(ema_50[i]) or pd.isna(rsi[i]):
                continue

            current_close = close[i]
            
            # ================================================================
            # ATR Velocity Veto (Step 3)
            # ================================================================
            velocity_drop = close[i-3] - current_close
            velocity_rise = current_close - close[i-3]
            atr_threshold = 4.0 * atr[i]
            
            # ================================================================
            # PILLAR 1: THE STRETCH (Volatility Exhaustion)
            # ================================================================
            stretch_long = (ema_50[i] - current_close > 2.0 * atr[i]) or (current_close < bb_lower[i])
            stretch_short = (current_close - ema_50[i] > 2.0 * atr[i]) or (current_close > bb_upper[i])

            # ================================================================
            # PILLAR 2: THE DIVERGENCE (1H)
            # ================================================================
            div_long = self._check_divergence(df.iloc[:i+1], signal=1, period=60)
            div_short = self._check_divergence(df.iloc[:i+1], signal=-1, period=60)
            
            # ================================================================
            # Liquidity Sweep OR-Gate (Step 4)
            # ================================================================
            lookback_100 = low[max(0, i-100):i]
            highback_100 = high[max(0, i-100):i]
            sweep_long = current_close < np.min(lookback_100) if len(lookback_100) > 0 else False
            sweep_short = current_close > np.max(highback_100) if len(highback_100) > 0 else False

            # ================================================================
            # PILLAR 3: THE EXHAUSTION (Pattern Recognition)
            # ================================================================
            exhaustion_long = False
            exhaustion_short = False
            
            # Pattern Detection (Must be Hammer, Doji, or Engulfing)
            patterns_to_check = {
                'Hammer': ta.CDLHAMMER,
                'Doji': ta.CDLDOJI,
                'Engulfing': ta.CDLENGULFING
            }
            o, h, l, c = df['open'].iloc[:i+1].values, df['high'].iloc[:i+1].values, df['low'].iloc[:i+1].values, df['close'].iloc[:i+1].values
            for name, func in patterns_to_check.items():
                res = func(o, h, l, c)
                if res[-1] != 0:
                    if stretch_long and res[-1] > 0: exhaustion_long = True
                    if stretch_short and res[-1] < 0: exhaustion_short = True

            # ================================================================
            # FINAL TRADE TRIGGER (Step 5)
            # ================================================================
            
            # BUY setup
            if stretch_long and (div_long or sweep_long) and exhaustion_long:
                if velocity_drop > atr_threshold:
                    logger.debug(f"[{self.name}] VETO BUY: Velocity drop {velocity_drop:.2f} > {atr_threshold:.2f}")
                else:
                    labels.iloc[i] = 1
                    signal_count["buy"] += 1
            
            # SELL setup
            elif stretch_short and (div_short or sweep_short) and exhaustion_short:
                if velocity_rise > atr_threshold:
                    logger.debug(f"[{self.name}] VETO SELL: Velocity rise {velocity_rise:.2f} > {atr_threshold:.2f}")
                else:
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

        buy_pct = signal_count["buy"] / total * 100
        sell_pct = signal_count["sell"] / total * 100
        total_signals = buy_pct + sell_pct

        if total_signals < 2:
            logger.warning(
                f"  ⚠ Very few signals ({total_signals:.1f}%) - Phase 3 is highly selective"
            )
        else:
            logger.info(f"  ✓ Phase 3 signal rate: {total_signals:.1f}%")

        return labels

