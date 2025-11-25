"""
FINAL FIX: Mean Reversion Strategy - Maximum Robustness
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Production-ready mean reversion with anti-overfitting measures
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
        self.asset = config.get('asset', 'BTC')  # default BTC
        
        # Thresholds - VERY conservative
        self.rsi_overbought = config.get('rsi_overbought', 64)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.bb_lower_threshold = config.get('bb_lower_threshold', 0.25)
        self.bb_upper_threshold = config.get('bb_upper_threshold', 0.70)
        
        # Strict thresholds
        self.min_return_threshold = config.get('min_return_threshold', 0.003)  # 0.3%
        self.min_score_threshold = config.get('min_conditions', 3.0)  # Higher bar
        
        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  RSI: {self.rsi_oversold}/{self.rsi_overbought}")
        logger.info(f"  BB Position: {self.bb_lower_threshold}/{self.bb_upper_threshold}")
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate minimal, non-redundant features"""
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # === Core Indicators Only ===
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        df['bb_width_norm'] = (bb_upper - bb_lower) / bb_middle
        
        # RSI
        df['rsi'] = ta.RSI(close, timeperiod=self.rsi_period)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # -1 to 1
        
        # Stochastic
        slowk, slowd = ta.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_d,
            slowd_period=self.stoch_d
        )
        df['stoch_k'] = slowk
        
        # Volatility (normalized)
        df['atr'] = ta.ATR(high, low, close, timeperiod=14)
        df['atr_pct'] = df['atr'] / close
        
        # Simple momentum
        df['roc_5'] = ta.ROC(close, timeperiod=5)
        # Simple momentum
        df['adx'] = ta.ADX(high, low, close, timeperiod=14)
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        STRICT label generation - only highest quality signals
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)
        
        close = df['close'].values
        bb_position = df['bb_position'].values
        rsi = df['rsi'].values
        stoch_k = df['stoch_k'].values
        bb_width = df['bb_width_norm'].values
        atr_pct = df['atr_pct'].values
        
        # Calculate adaptive thresholds based on market regime
        median_vol = np.nanmedian(atr_pct)
        high_vol_threshold = np.nanpercentile(atr_pct, 75)
        
        signal_count = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for i in range(len(df) - self.reversion_window - 1):  # Extra safety margin
            # Skip if ANY missing data
            if (pd.isna(bb_position[i]) or pd.isna(rsi[i]) or 
                pd.isna(stoch_k[i]) or pd.isna(bb_width[i]) or
                pd.isna(atr_pct[i])):
                continue
            
            # Skip if extreme volatility (unreliable signals)
            if atr_pct[i] > high_vol_threshold:
                continue
                
            current_close = close[i]
            future_closes = close[i+1:i+1+self.reversion_window]
            
            if len(future_closes) < self.reversion_window:
                continue
            
            # Calculate BOTH max gain AND consistent direction
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
            
            # === STRICT BUY SIGNAL ===
            buy_score = 0.0
            
            # 1. Extreme oversold (not just oversold)
            if bb_position[i] < 0.10:
                buy_score += 2.5
            elif bb_position[i] < 0.15:
                buy_score += 1.5
            elif bb_position[i] < self.bb_lower_threshold:
                buy_score += 0.5  # Reduced
            
            # 2. RSI extreme
            if rsi[i] < 25:
                buy_score += 2.0
            elif rsi[i] < 30:
                buy_score += 1.0
            elif rsi[i] < self.rsi_oversold:
                buy_score += 0.3  # Reduced
            
            # 3. Stochastic confirmation
            if stoch_k[i] < 15:
                buy_score += 1.5
            elif stoch_k[i] < 20:
                buy_score += 0.5
            
            # 4. NOT in squeeze (minimum volatility required)
            if bb_width[i] > 0.015:  # Absolute minimum width
                buy_score += 1.0
            
            # 5. NEW: Require consistent upward movement
            direction_score = 0
            for j in range(1, min(4, len(future_closes))):
                if future_closes[j] > future_closes[j-1]:
                    direction_score += 1
            
            if direction_score >= 2:  # At least 2 consecutive up bars
                buy_score += 1.0
            
            # Label only if VERY strong signal + BOTH peak AND average returns good
            if (buy_score >= self.min_score_threshold and 
                future_return_up > adj_min_return and
                avg_return > adj_min_return * 0.5):  # Average must also be positive
                labels.iloc[i] = 1
                signal_count['buy'] += 1
            
            # === STRICT SELL SIGNAL ===
            sell_score = 0.0
            
            # 1. Extreme overbought
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
            
            # 5. NEW: Require consistent downward movement
            direction_score = 0
            for j in range(1, min(4, len(future_closes))):
                if future_closes[j] < future_closes[j-1]:
                    direction_score += 1
            
            if direction_score >= 2:
                sell_score += 1.0
            
            # Label only if VERY strong signal + BOTH trough AND average returns good
            if (sell_score >= self.min_score_threshold and 
                future_return_down > adj_min_return and
                avg_return < -adj_min_return * 0.5):  # Average must also be negative
                labels.iloc[i] = -1
                signal_count['sell'] += 1
        
        # Remove labels from last N bars
        labels.iloc[-self.reversion_window:] = 0
        
        # === Detailed logging ===
        signal_count['hold'] = len(labels) - signal_count['buy'] - signal_count['sell']
        total = len(labels)
        
        logger.info(f"[{self.name}] Label distribution:")
        logger.info(f"  SELL: {signal_count['sell']:>5} ({signal_count['sell']/total*100:>5.2f}%)")
        logger.info(f"  HOLD: {signal_count['hold']:>5} ({signal_count['hold']/total*100:>5.2f}%)")
        logger.info(f"  BUY:  {signal_count['buy']:>5} ({signal_count['buy']/total*100:>5.2f}%)")
        
        buy_pct = signal_count['buy'] / total * 100
        sell_pct = signal_count['sell'] / total * 100
        total_signals = buy_pct + sell_pct
        
        # Ideal range: 8-15% total signals
        if total_signals < 5:
            logger.warning(f"  ⚠ Very few signals ({total_signals:.1f}%) - consider loosening")
        elif total_signals > 30:
            logger.warning(f"  ⚠ Too many signals ({total_signals:.1f}%) - tighten thresholds")
        else:
            logger.info(f"  ✓ Good signal rate: {total_signals:.1f}%")
        
        return labels