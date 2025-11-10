# src/strategies/mean_reversion.py
"""
Mean Reversion Strategy - Ranging Market Specialist
Uses Bollinger Bands, RSI, and Stochastic Oscillator
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Identifies overbought/oversold conditions and predicts reversion to mean
    """
    
    def __init__(self, config: dict):
        super().__init__(config, "MeanReversion")
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.stoch_k = config.get('stoch_k', 14)
        self.stoch_d = config.get('stoch_d', 3)
        self.reversion_window = config.get('reversion_window', 5)
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion features
        All features use ONLY historical data (no lookahead)
        """
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # 0-1 normalized
        
        # Distance from bands (key mean reversion signal)
        df['dist_from_upper'] = (bb_upper - close) / close
        df['dist_from_lower'] = (close - bb_lower) / close
        
        # RSI
        df['rsi'] = ta.RSI(close, timeperiod=self.rsi_period)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # Stochastic Oscillator
        slowk, slowd = ta.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_d,
            slowd_period=self.stoch_d
        )
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_overbought'] = (slowk > 80).astype(int)
        df['stoch_oversold'] = (slowk < 20).astype(int)
        
        # Volatility (for filtering low-volatility ranges)
        df['volatility'] = ta.ATR(high, low, close, timeperiod=14) / close
        
        # Price momentum (for detecting potential reversions)
        df['momentum'] = ta.MOM(close, timeperiod=5)
        df['roc'] = ta.ROC(close, timeperiod=5)
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels based on mean reversion logic
        Label = 1 (BUY) if price touches lower band and reverts up
        Label = -1 (SELL) if price touches upper band and reverts down
        Label = 0 (HOLD) otherwise
        
        CRITICAL: Uses only PAST data to label (forward-looking with offset)
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)
        
        close = df['close'].values
        bb_upper = df['bb_upper'].values
        bb_lower = df['bb_lower'].values
        bb_middle = df['bb_middle'].values
        
        # Look forward N bars to see if reversion occurred
        for i in range(len(df) - self.reversion_window):
            current_close = close[i]
            future_closes = close[i+1:i+1+self.reversion_window]
            
            # BUY signal: touched lower band, then reverted toward middle
            if current_close <= bb_lower[i]:
                if np.any(future_closes > bb_middle[i]):
                    labels.iloc[i] = 1
            
            # SELL signal: touched upper band, then reverted toward middle
            elif current_close >= bb_upper[i]:
                if np.any(future_closes < bb_middle[i]):
                    labels.iloc[i] = -1
        
        # Remove labels from last N bars (we don't have future data)
        labels.iloc[-self.reversion_window:] = 0
        
        return labels