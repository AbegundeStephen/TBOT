# src/strategies/trend_following.py
"""
IMPROVED Trend Following Strategy with Better Label Generation
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Improved trend following with balanced label generation
    """
    
    def __init__(self, config: dict):
        super().__init__(config, "TrendFollowing")
        
        # Moving average periods
        self.fast_ma = config.get('fast_ma', 20)
        self.slow_ma = config.get('slow_ma', 50)
        
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # ADX parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 15)
        self.require_adx = config.get('require_adx', False)
        
        # FIXED: More realistic thresholds
        self.min_return_threshold = config.get('min_return_threshold', 0.002)  # 0.2%
        
        # FIXED: Direct score threshold (not multiplier)
        self.min_score_threshold = config.get('min_conditions', 2)
        
        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  Fast MA: {self.fast_ma}, Slow MA: {self.slow_ma}")
        logger.info(f"  ADX Threshold: {self.adx_threshold} (Required: {self.require_adx})")
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")
    
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
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Moving Averages
        df['sma_fast'] = ta.SMA(close, timeperiod=self.fast_ma)
        df['sma_slow'] = ta.SMA(close, timeperiod=self.slow_ma)
        df['ema_fast'] = ta.EMA(close, timeperiod=self.fast_ma)
        df['ema_slow'] = ta.EMA(close, timeperiod=self.slow_ma)
        
        # MA relationships
        df['ma_diff'] = df['sma_fast'] - df['sma_slow']
        df['ma_diff_pct'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
        
        # MA crossovers
        df['ma_cross'] = 0
        ma_diff_shift = df['ma_diff'].shift(1)
        df.loc[(df['ma_diff'] > 0) & (ma_diff_shift <= 0), 'ma_cross'] = 1
        df.loc[(df['ma_diff'] < 0) & (ma_diff_shift >= 0), 'ma_cross'] = -1
        
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
        
        # MACD crossovers
        df['macd_cross'] = 0
        macd_hist_shift = df['macd_hist'].shift(1)
        df.loc[(df['macd_hist'] > 0) & (macd_hist_shift <= 0), 'macd_cross'] = 1
        df.loc[(df['macd_hist'] < 0) & (macd_hist_shift >= 0), 'macd_cross'] = -1
        
        # ADX (trend strength)
        df['adx'] = ta.ADX(high, low, close, timeperiod=self.adx_period)
        df['strong_trend'] = (df['adx'] > self.adx_threshold).astype(int)
        
        # Directional indicators
        df['plus_di'] = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        df['minus_di'] = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        df['di_diff'] = df['plus_di'] - df['minus_di']
        
        # Price position
        df['close_above_fast'] = (close > df['sma_fast']).astype(int)
        df['close_above_slow'] = (close > df['sma_slow']).astype(int)
        
        # Momentum
        df['momentum'] = ta.MOM(close, timeperiod=10)
        df['roc'] = ta.ROC(close, timeperiod=10)
        
        # Trend features
        df['trend_strength'] = np.abs(df['ma_diff_pct'])
        df['price_vs_ma_fast'] = (close - df['sma_fast']) / df['sma_fast']
        df['price_vs_ma_slow'] = (close - df['sma_slow']) / df['sma_slow']
        
        # MA slopes
        df['ma_fast_slope'] = df['sma_fast'].diff(5) / df['sma_fast']
        df['ma_slow_slope'] = df['sma_slow'].diff(10) / df['sma_slow']
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        IMPROVED: More balanced label generation with gradual scoring
        """
        df = df.copy()
        labels = pd.Series(0, index=df.index)
        
        close = df['close'].values
        sma_fast = df['sma_fast'].values
        sma_slow = df['sma_slow'].values
        adx = df['adx'].values
        macd_hist = df['macd_hist'].values
        plus_di = df['plus_di'].values
        minus_di = df['minus_di'].values
        
        continuation_bars = 5
        
        for i in range(len(df) - continuation_bars - 1):
            # Skip if required ADX not met
            if self.require_adx and (pd.isna(adx[i]) or adx[i] < self.adx_threshold):
                continue
            
            # Calculate future return
            future_closes = close[i+1:i+1+continuation_bars]
            if len(future_closes) == 0:
                continue
                
            future_return = (np.mean(future_closes) - close[i]) / close[i]
            
            # === BULLISH SIGNALS ===
            bullish_score = 0
            
            # 1. Fast MA vs Slow MA (stronger = more points)
            if not pd.isna(sma_fast[i]) and not pd.isna(sma_slow[i]):
                ma_gap_pct = (sma_fast[i] - sma_slow[i]) / sma_slow[i] * 100
                
                if sma_fast[i] > sma_slow[i]:
                    if ma_gap_pct > 3:  # Strong uptrend
                        bullish_score += 2
                    elif ma_gap_pct > 1.5:
                        bullish_score += 1.5
                    elif ma_gap_pct > 0.5:
                        bullish_score += 1
                    else:
                        bullish_score += 0.5  # Weak uptrend
            
            # 2. MACD histogram
            if not pd.isna(macd_hist[i]):
                if macd_hist[i] > 0:
                    # Check if increasing
                    if i > 0 and not pd.isna(macd_hist[i-1]):
                        if macd_hist[i] > macd_hist[i-1] * 1.1:  # 10% increase
                            bullish_score += 1.5
                        elif macd_hist[i] > macd_hist[i-1]:
                            bullish_score += 1
                        else:
                            bullish_score += 0.5
                    else:
                        bullish_score += 0.5
            
            # 3. Directional Movement
            if not pd.isna(plus_di[i]) and not pd.isna(minus_di[i]):
                di_gap = plus_di[i] - minus_di[i]
                
                if di_gap > 20:  # Very strong
                    bullish_score += 2
                elif di_gap > 10:
                    bullish_score += 1.5
                elif di_gap > 5:
                    bullish_score += 1
                elif di_gap > 0:
                    bullish_score += 0.5
            
            # 4. Price vs MAs
            if not pd.isna(sma_fast[i]) and close[i] > sma_fast[i]:
                bullish_score += 0.5
            if not pd.isna(sma_slow[i]) and close[i] > sma_slow[i]:
                bullish_score += 0.5
            
            # 5. ADX strength (bonus)
            if not pd.isna(adx[i]):
                if adx[i] > 30:  # Very strong trend
                    bullish_score += 1.5
                elif adx[i] > 25:
                    bullish_score += 1
                elif adx[i] > self.adx_threshold:
                    bullish_score += 0.5
            
            # Label as BUY if score meets threshold
            if bullish_score >= self.min_score_threshold and future_return > self.min_return_threshold:
                labels.iloc[i] = 1
            
            # === BEARISH SIGNALS ===
            bearish_score = 0
            
            # 1. Fast MA vs Slow MA
            if not pd.isna(sma_fast[i]) and not pd.isna(sma_slow[i]):
                ma_gap_pct = (sma_slow[i] - sma_fast[i]) / sma_slow[i] * 100
                
                if sma_fast[i] < sma_slow[i]:
                    if ma_gap_pct > 3:
                        bearish_score += 2
                    elif ma_gap_pct > 1.5:
                        bearish_score += 1.5
                    elif ma_gap_pct > 0.5:
                        bearish_score += 1
                    else:
                        bearish_score += 0.5
            
            # 2. MACD histogram
            if not pd.isna(macd_hist[i]):
                if macd_hist[i] < 0:
                    if i > 0 and not pd.isna(macd_hist[i-1]):
                        if macd_hist[i] < macd_hist[i-1] * 0.9:  # Declining 10%
                            bearish_score += 1.5
                        elif macd_hist[i] < macd_hist[i-1]:
                            bearish_score += 1
                        else:
                            bearish_score += 0.5
                    else:
                        bearish_score += 0.5
            
            # 3. Directional Movement
            if not pd.isna(plus_di[i]) and not pd.isna(minus_di[i]):
                di_gap = minus_di[i] - plus_di[i]
                
                if di_gap > 20:
                    bearish_score += 2
                elif di_gap > 10:
                    bearish_score += 1.5
                elif di_gap > 5:
                    bearish_score += 1
                elif di_gap > 0:
                    bearish_score += 0.5
            
            # 4. Price vs MAs
            if not pd.isna(sma_fast[i]) and close[i] < sma_fast[i]:
                bearish_score += 0.5
            if not pd.isna(sma_slow[i]) and close[i] < sma_slow[i]:
                bearish_score += 0.5
            
            # 5. ADX strength
            if not pd.isna(adx[i]):
                if adx[i] > 30:
                    bearish_score += 1.5
                elif adx[i] > 25:
                    bearish_score += 1
                elif adx[i] > self.adx_threshold:
                    bearish_score += 0.5
            
            # Label as SELL if score meets threshold
            if bearish_score >= self.min_score_threshold and future_return < -self.min_return_threshold:
                labels.iloc[i] = -1
        
        # Remove labels from last N bars
        labels.iloc[-continuation_bars:] = 0
        
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
        
        if buy_pct < 8 or sell_pct < 8:
            logger.warning(f"  ⚠ Class imbalance detected!")
            logger.warning(f"  Current: min_return={self.min_return_threshold:.3%}, min_score={self.min_score_threshold}")
            logger.warning(f"  Try lowering min_return_threshold or min_conditions in config")
        
        return labels