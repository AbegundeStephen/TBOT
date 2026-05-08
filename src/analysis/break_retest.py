"""
Break and Retest Validation Engine
Validates if a structural breakout has been successfully retested.
"""

import pandas as pd
import numpy as np
import talib as ta
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class BreakRetestResult:
    is_valid: bool
    score: float  # 0.0 to 1.0
    type: str  # "BULLISH_RETEST", "BEARISH_RETEST", "NONE"
    level: float
    explanation: str

class BreakRetestValidator:
    """
    Tracks price action around key levels to confirm 'Break and Retest' setups.
    Higher confidence than raw breakouts.
    """
    
    def __init__(self, lookback: int = 50, retest_threshold_atr: float = 0.5):
        self.lookback = lookback
        self.retest_threshold_atr = retest_threshold_atr

    def validate(self, df: pd.DataFrame, asset: str) -> BreakRetestResult:
        """Analyze the last few candles for a retest pattern."""
        if len(df) < self.lookback:
            return BreakRetestResult(False, 0.0, "NONE", 0.0, "Insufficient data")

        try:
            # 1. Identify recent structural high/low (Last 20-50 bars excluding last 5)
            # We exclude last 5 to see the breakout and retest clearly
            struct_df = df.iloc[-self.lookback:-5]
            high_20 = struct_df['high'].max()
            low_20 = struct_df['low'].min()
            
            latest = df.iloc[-1]
            prev_bars = df.iloc[-5:-1]
            
            high, low, close = df['high'].values, df['low'].values, df['close'].values
            atr = ta.ATR(high, low, close, timeperiod=14)[-1]
            
            retest_zone = self.retest_threshold_atr * atr
            
            # --- BULLISH RETEST CHECK ---
            # 1. Price broke above High_20 in the last 5 bars
            # 2. Price returned to High_20 (within zone)
            # 3. Current bar or prev bar is rejecting the level (low near high_20, close above)
            
            if any(prev_bars['high'] > high_20):
                # We have a breakout!
                # Now check for retest
                # Find the lowest point since the breakout
                lowest_since_break = df['low'].iloc[-3:].min()
                
                if abs(lowest_since_break - high_20) <= retest_zone:
                    # Retest happened! Now check for rejection
                    if latest['close'] > high_20:
                        # Success!
                        return BreakRetestResult(
                            True, 0.9, "BULLISH_RETEST", high_20,
                            f"Bullish B&R: Price retested broken high ${high_20:.2f} and held."
                        )

            # --- BEARISH RETEST CHECK ---
            if any(prev_bars['low'] < low_20):
                # We have a breakdown!
                # Check for retest (highest point since breakdown)
                highest_since_break = df['high'].iloc[-3:].max()
                
                if abs(highest_since_break - low_20) <= retest_zone:
                    # Retest happened!
                    if latest['close'] < low_20:
                        return BreakRetestResult(
                            True, 0.9, "BEARISH_RETEST", low_20,
                            f"Bearish B&R: Price retested broken low ${low_20:.2f} and held."
                        )

            return BreakRetestResult(False, 0.0, "NONE", 0.0, "No B&R setup detected")

        except Exception as e:
            logger.error(f"B&R validation error: {e}")
            return BreakRetestResult(False, 0.0, "NONE", 0.0, f"Error: {e}")
