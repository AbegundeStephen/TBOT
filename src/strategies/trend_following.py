# src/strategies/trend_following.py
"""
Trend Following Strategy - Trending Market Specialist
Uses MA Crossover, MACD, and ADX
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Identifies trend continuation after MA crossovers
    """
    
    def __init__(self, config: dict):
        super().__init__(config, "TrendFollowing")
        self.fast_ma = config.get('fast_ma', 50)
        self.slow_ma = config.get('slow_ma', 200)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend following features
        All features use ONLY historical data
        """
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Moving Averages
        df['sma_fast'] = ta.SMA(close, timeperiod=self.fast_ma)
        df['sma_slow'] = ta.SMA(close, timeperiod=self.slow_ma)
        df['ema_fast'] = ta.EMA(close, timeperiod=self.fast_ma)
        df['ema_slow'] = ta.EMA(close, timeperiod=self.slow_ma)
        
        # MA Crossover signals
        df['ma_diff'] = df['sma_fast'] - df['sma_slow']
        df['ma_diff_pct'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
        
        # Detect crossovers (1 = golden cross, -1 = death cross, 0 = no cross)
        df['ma_cross'] = 0
        ma_diff_shift = df['ma_diff'].shift(1)
        df.loc[(df['ma_diff'] > 0) & (ma_diff_shift <= 0), 'ma_cross'] = 1  # Golden cross
        df.loc[(df['ma_diff'] < 0) & (ma_diff_shift >= 0), 'ma_cross'] = -1  # Death cross
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_cross'] = 0
        macd_hist_shift = df['macd_hist'].shift(1)
        df.loc[(df['macd_hist'] > 0) & (macd_hist_shift <= 0), 'macd_cross'] = 1
        df.loc[(df['macd_hist'] < 0) & (macd_hist_shift >= 0), 'macd_cross'] = -1
        
        # ADX (trend strength)
        df['adx'] = ta.ADX(high, low, close, timeperiod=self.adx_period)
        df['strong_trend'] = (df['adx'] > self.adx_threshold).astype(int)
        
        # Directional Movement
        df['plus_di'] = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        df['minus_di'] = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        df['di_diff'] = df['plus_di'] - df['minus_di']
        
        # Price position relative to MAs
        df['close_above_fast'] = (close > df['sma_fast']).astype(int)
        df['close_above_slow'] = (close > df['sma_slow']).astype(int)
        
        # Trend momentum
        df['momentum'] = ta.MOM(close, timeperiod=10)
        df['roc'] = ta.ROC(close, timeperiod=10)
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels based on trend continuation
        Label = 1 (BUY) if golden cross occurs and trend continues upward
        Label = -1 (SELL) if death cross occurs and trend continues downward
        Label = 0 (HOLD) otherwise
        
        CRITICAL: Uses only PAST data to label
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)
        
        close = df['close'].values
        sma_fast = df['sma_fast'].values
        sma_slow = df['sma_slow'].values
        adx = df['adx'].values
        
        continuation_bars = 10  # Look ahead 10 bars for trend confirmation
        
        for i in range(len(df) - continuation_bars):
            # Check if strong trend exists
            if adx[i] < self.adx_threshold:
                continue
            
            # Golden cross detection
            if sma_fast[i-1] <= sma_slow[i-1] and sma_fast[i] > sma_slow[i]:
                # Verify upward continuation
                future_closes = close[i+1:i+1+continuation_bars]
                if np.mean(future_closes) > close[i] * 1.01:  # 1% upward movement
                    labels.iloc[i] = 1
            
            # Death cross detection
            elif sma_fast[i-1] >= sma_slow[i-1] and sma_fast[i] < sma_slow[i]:
                # Verify downward continuation
                future_closes = close[i+1:i+1+continuation_bars]
                if np.mean(future_closes) < close[i] * 0.99:  # 1% downward movement
                    labels.iloc[i] = -1
        
        # Remove labels from last N bars
        labels.iloc[-continuation_bars:] = 0
        
        return labels
