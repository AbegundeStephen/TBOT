# src/strategies/trend_following.py
"""
IMPROVED Trend Following Strategy with Balanced Label Generation
Key improvements:
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
    Improved trend following with balanced label generation
    Uses a scoring system with adaptive thresholds
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
        
        # ADX parameters - RELAXED
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 15)  # Lower threshold
        self.require_adx = config.get('require_adx', False)
        
        # Return thresholds - MORE LENIENT
        self.min_return_threshold = config.get('min_return_threshold', 0.001)  # 0.1% instead of 0.2%
        
        # Score threshold - REDUCED
        self.min_score_threshold = config.get('min_conditions', 1.5)  # Lower from 2.0
        
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
        IMPROVED label generation with adaptive scoring system
        - Uses weighted scoring for different signal strengths
        - Adaptive return thresholds based on recent volatility
        - Multiple lookforward periods for different trend types
        """
        labels = pd.Series(0, index=df.index)
        close = df['close'].values
        sma_fast = df['sma_fast'].values
        sma_slow = df['sma_slow'].values
        adx = df['adx'].values
        macd_hist = df['macd_hist'].values
        plus_di = df['plus_di'].values
        minus_di = df['minus_di'].values
        
        # Calculate rolling volatility for adaptive thresholds
        returns = pd.Series(close).pct_change()
        rolling_vol = returns.rolling(20).std()
        
        # Multiple lookforward periods for different trend strengths
        short_term = 3   # Quick reversals
        medium_term = 7  # Standard trends
        long_term = 12   # Strong trends
        
        for i in range(len(df) - long_term - 1):
            # Skip if not enough data
            if pd.isna(adx[i]) or pd.isna(rolling_vol.iloc[i]):
                continue
            
            # Adaptive return threshold (0.3x to 1.5x of recent volatility)
            vol = rolling_vol.iloc[i]
            min_return = max(self.min_return_threshold, 0.5 * vol)
            
            # === CALCULATE BULLISH SCORE ===
            bullish_score = 0.0
            
            # 1. MA Alignment (0-2 points)
            if sma_fast[i] > sma_slow[i]:
                ma_separation = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
                if ma_separation > 0.002:  # >0.2% separation
                    bullish_score += 2.0
                else:
                    bullish_score += 1.0
            
            # 2. MACD (0-1.5 points)
            if macd_hist[i] > 0:
                macd_strength = abs(macd_hist[i])
                if macd_strength > macd_hist[max(0, i-5):i].std():  # Strong signal
                    bullish_score += 1.5
                else:
                    bullish_score += 1.0
            
            # 3. Directional Indicators (0-1.5 points)
            if plus_di[i] > minus_di[i]:
                di_diff = plus_di[i] - minus_di[i]
                if di_diff > 10:  # Strong directional move
                    bullish_score += 1.5
                else:
                    bullish_score += 1.0
            
            # 4. Price Position (0-1 point)
            if close[i] > sma_fast[i]:
                bullish_score += 1.0
            
            # 5. ADX Bonus (0-1 point) - Only if strong trend
            if adx[i] > 25:
                bullish_score += 1.0
            elif adx[i] > self.adx_threshold:
                bullish_score += 0.5
            
            # === CALCULATE BEARISH SCORE ===
            bearish_score = 0.0
            
            # 1. MA Alignment (0-2 points)
            if sma_fast[i] < sma_slow[i]:
                ma_separation = (sma_slow[i] - sma_fast[i]) / sma_slow[i]
                if ma_separation > 0.002:
                    bearish_score += 2.0
                else:
                    bearish_score += 1.0
            
            # 2. MACD (0-1.5 points)
            if macd_hist[i] < 0:
                macd_strength = abs(macd_hist[i])
                if macd_strength > macd_hist[max(0, i-5):i].std():
                    bearish_score += 1.5
                else:
                    bearish_score += 1.0
            
            # 3. Directional Indicators (0-1.5 points)
            if minus_di[i] > plus_di[i]:
                di_diff = minus_di[i] - plus_di[i]
                if di_diff > 10:
                    bearish_score += 1.5
                else:
                    bearish_score += 1.0
            
            # 4. Price Position (0-1 point)
            if close[i] < sma_fast[i]:
                bearish_score += 1.0
            
            # 5. ADX Bonus (0-1 point)
            if adx[i] > 25:
                bearish_score += 1.0
            elif adx[i] > self.adx_threshold:
                bearish_score += 0.5
            
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
            future_closes = close[i+1:end_idx]
            if len(future_closes) == 0:
                continue
                
            future_return = (np.mean(future_closes) - close[i]) / close[i]
            
            # BUY Signal: Strong bullish score + positive return
            if bullish_score >= self.min_score_threshold and future_return > min_return:
                labels.iloc[i] = 1
            
            # SELL Signal: Strong bearish score + negative return
            elif bearish_score >= self.min_score_threshold and future_return < -min_return:
                labels.iloc[i] = -1
            
            # HOLD: Everything else stays 0
        
        # === LOG DETAILED STATISTICS ===
        unique, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        total_labels = len(labels)
        
        sell_count = label_distribution.get(-1, 0)
        hold_count = label_distribution.get(0, 0)
        buy_count = label_distribution.get(1, 0)
        
        sell_pct = (sell_count / total_labels) * 100
        hold_pct = (hold_count / total_labels) * 100
        buy_pct = (buy_count / total_labels) * 100
        
        logger.info(f"[{self.name}] Label Distribution:")
        logger.info(f"  SELL: {sell_count:>5} ({sell_pct:>5.2f}%)")
        logger.info(f"  HOLD: {hold_count:>5} ({hold_pct:>5.2f}%)")
        logger.info(f"  BUY:  {buy_count:>5} ({buy_pct:>5.2f}%)")
        
        # Quality checks
        if sell_pct < 5 or buy_pct < 5:
            logger.warning(f"  ⚠ Severe class imbalance! Consider lowering min_score_threshold")
        elif sell_pct < 10 or buy_pct < 10:
            logger.warning(f"  ⚠ Class imbalance detected! Consider adjusting thresholds.")
        else:
            logger.info(f"  ✓ Reasonable signal distribution")
        
        return labels