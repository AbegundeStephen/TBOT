# src/strategies/mean_reversion.py
"""
IMPROVED Mean Reversion Strategy with Better Label Generation
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Improved mean reversion with balanced label generation
    """
    
    def __init__(self, config: dict):
        super().__init__(config, "MeanReversion")
        
        # Standard parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.stoch_k = config.get('stoch_k', 14)
        self.stoch_d = config.get('stoch_d', 3)
        self.reversion_window = config.get('reversion_window', 5)
        
        # Thresholds
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.bb_lower_threshold = config.get('bb_lower_threshold', 0.30)
        self.bb_upper_threshold = config.get('bb_upper_threshold', 0.70)
        
        #  More realistic thresholds
        self.min_return_threshold = config.get('min_return_threshold', 0.0015)  # 0.15%
        
        #  Use raw value, not multiplier
        self.min_score_threshold = config.get('min_conditions', 2)  # Direct score threshold
        
        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  RSI: {self.rsi_oversold}/{self.rsi_overbought}")
        logger.info(f"  BB Position: {self.bb_lower_threshold}/{self.bb_upper_threshold}")
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion features"""
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
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Distance from bands
        df['dist_from_upper'] = (bb_upper - close) / close
        df['dist_from_lower'] = (close - bb_lower) / close
        
        # RSI
        df['rsi'] = ta.RSI(close, timeperiod=self.rsi_period)
        df['rsi_overbought'] = (df['rsi'] > self.rsi_overbought).astype(int)
        df['rsi_oversold'] = (df['rsi'] < self.rsi_oversold).astype(int)
        
        # Stochastic
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
        
        # Volatility
        df['volatility'] = ta.ATR(high, low, close, timeperiod=14) / close
        
        # Momentum
        df['momentum'] = ta.MOM(close, timeperiod=5)
        df['roc'] = ta.ROC(close, timeperiod=5)
        
        # Additional signals
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        df['price_from_middle'] = (close - bb_middle) / bb_middle
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        IMPROVED: More balanced label generation
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)
        
        close = df['close'].values
        bb_position = df['bb_position'].values
        rsi = df['rsi'].values
        stoch_k = df['stoch_k'].values
        
        for i in range(len(df) - self.reversion_window):
            if pd.isna(bb_position[i]) or pd.isna(rsi[i]):
                continue
                
            current_close = close[i]
            future_closes = close[i+1:i+1+self.reversion_window]
            
            if len(future_closes) == 0:
                continue
            
            # Calculate future return
            avg_future_close = np.mean(future_closes)
            future_return = (avg_future_close - current_close) / current_close
            
            # === BUY SIGNAL (Oversold -> Expect bounce) ===
            buy_score = 0
            
            # 1. BB position (lower = stronger signal)
            if bb_position[i] < 0.1:
                buy_score += 2  # Very oversold
            elif bb_position[i] < 0.2:
                buy_score += 1.5
            elif bb_position[i] < self.bb_lower_threshold:
                buy_score += 1
            
            # 2. RSI oversold
            if rsi[i] < 25:
                buy_score += 2  # Very oversold
            elif rsi[i] < 30:
                buy_score += 1.5
            elif rsi[i] < self.rsi_oversold:
                buy_score += 1
            
            # 3. Stochastic oversold
            if not pd.isna(stoch_k[i]):
                if stoch_k[i] < 15:
                    buy_score += 1.5
                elif stoch_k[i] < 20:
                    buy_score += 1
                elif stoch_k[i] < 30:
                    buy_score += 0.5
            
            # Label as BUY if score meets threshold AND positive return
            if buy_score >= self.min_score_threshold and future_return > self.min_return_threshold:
                labels.iloc[i] = 1
            
            # === SELL SIGNAL (Overbought -> Expect pullback) ===
            sell_score = 0
            
            # 1. BB position (higher = stronger signal)
            if bb_position[i] > 0.9:
                sell_score += 2  # Very overbought
            elif bb_position[i] > 0.8:
                sell_score += 1.5
            elif bb_position[i] > self.bb_upper_threshold:
                sell_score += 1
            
            # 2. RSI overbought
            if rsi[i] > 75:
                sell_score += 2
            elif rsi[i] > 70:
                sell_score += 1.5
            elif rsi[i] > self.rsi_overbought:
                sell_score += 1
            
            # 3. Stochastic overbought
            if not pd.isna(stoch_k[i]):
                if stoch_k[i] > 85:
                    sell_score += 1.5
                elif stoch_k[i] > 80:
                    sell_score += 1
                elif stoch_k[i] > 70:
                    sell_score += 0.5
            
            # Label as SELL if score meets threshold AND negative return
            if sell_score >= self.min_score_threshold and future_return < -self.min_return_threshold:
                labels.iloc[i] = -1
        
        # Remove labels from last N bars
        labels.iloc[-self.reversion_window:] = 0
        
        # Log distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(labels)
        
        logger.info(f"[{self.name}] Label distribution:")
        logger.info(f"  SELL: {dist.get(-1, 0):>5} ({dist.get(-1, 0)/total*100:>5.2f}%)")
        logger.info(f"  HOLD: {dist.get(0, 0):>5} ({dist.get(0, 0)/total*100:>5.2f}%)")
        logger.info(f"  BUY:  {dist.get(1, 0):>5} ({dist.get(1, 0)/total*100:>5.2f}%)")
        
        # Warning if too imbalanced
        buy_pct = dist.get(1, 0) / total * 100
        sell_pct = dist.get(-1, 0) / total * 100
        
        if buy_pct < 5 or sell_pct < 5:
            logger.warning(f"  ⚠ Class imbalance detected! Consider adjusting thresholds")
            logger.warning(f"  Current: min_return={self.min_return_threshold:.3%}, min_score={self.min_score_threshold}")
        
        return labels