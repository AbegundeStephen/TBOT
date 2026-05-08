"""
RSI Divergence Engine
Detects Regular and Hidden Divergences between Price and RSI using Swing Points.
"""

import pandas as pd
import numpy as np
import talib as ta
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DivergenceResult:
    type: str  # "BULLISH", "BEARISH", "HIDDEN_BULLISH", "HIDDEN_BEARISH", "NONE"
    score: float  # -1.0 to +1.0
    confidence: float # 0 to 100
    price_pivots: List[float]
    rsi_pivots: List[float]
    explanation: str

class RSIDivergenceDetector:
    """
    Professional-grade Divergence Detector using swing point analysis.
    Avoids naive candle comparisons to prevent noise triggers.
    """
    
    def __init__(self, pivot_window: int = 5, rsi_period: int = 14):
        self.pivot_window = pivot_window
        self.rsi_period = rsi_period

    def analyze(self, df: pd.DataFrame) -> DivergenceResult:
        """Analyze the last few pivots for divergence."""
        if len(df) < 50:
            return DivergenceResult("NONE", 0.0, 0.0, [], [], "Insufficient data")

        try:
            # 1. Calculate RSI
            rsi = ta.RSI(df['close'].values, timeperiod=self.rsi_period)
            df = df.copy()
            df['rsi'] = rsi
            
            # 2. Find Pivots (Swing Highs/Lows)
            highs = self._find_pivots(df['high'].values, is_high=True)
            lows = self._find_pivots(df['low'].values, is_high=False)
            
            rsi_highs = self._find_pivots(df['rsi'].values, is_high=True)
            rsi_lows = self._find_pivots(df['rsi'].values, is_high=False)

            # 3. Check for Divergences
            # We look at the last 2 pivots
            if len(highs) >= 2 and len(rsi_highs) >= 2:
                # Regular Bearish Divergence: Price Higher High, RSI Lower High
                if highs[-1][1] > highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                    if df['rsi'].iloc[highs[-1][0]] > 60: # Overbought context
                        return DivergenceResult(
                            "BEARISH", -1.0, 85.0, 
                            [highs[-2][1], highs[-1][1]], 
                            [rsi_highs[-2][1], rsi_highs[-1][1]],
                            f"Regular Bearish Divergence: Price HH (${highs[-1][1]:.2f}) vs RSI LH ({rsi_highs[-1][1]:.1f})"
                        )
                
                # Hidden Bearish Divergence: Price Lower High, RSI Higher High
                if highs[-1][1] < highs[-2][1] and rsi_highs[-1][1] > rsi_highs[-2][1]:
                    return DivergenceResult(
                        "HIDDEN_BEARISH", -0.7, 70.0,
                        [highs[-2][1], highs[-1][1]],
                        [rsi_highs[-2][1], rsi_highs[-1][1]],
                        f"Hidden Bearish Divergence (Trend Continuation): Price LH vs RSI HH"
                    )

            if len(lows) >= 2 and len(rsi_lows) >= 2:
                # Regular Bullish Divergence: Price Lower Low, RSI Higher Low
                if lows[-1][1] < lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                    if df['rsi'].iloc[lows[-1][0]] < 40: # Oversold context
                        return DivergenceResult(
                            "BULLISH", 1.0, 85.0,
                            [lows[-2][1], lows[-1][1]],
                            [rsi_lows[-2][1], rsi_lows[-1][1]],
                            f"Regular Bullish Divergence: Price LL (${lows[-1][1]:.2f}) vs RSI HL ({rsi_lows[-1][1]:.1f})"
                        )
                
                # Hidden Bullish Divergence: Price Higher Low, RSI Lower Low
                if lows[-1][1] > lows[-2][1] and rsi_lows[-1][1] < rsi_lows[-2][1]:
                    return DivergenceResult(
                        "HIDDEN_BULLISH", 0.7, 70.0,
                        [lows[-2][1], lows[-1][1]],
                        [rsi_lows[-2][1], rsi_lows[-1][1]],
                        f"Hidden Bullish Divergence (Trend Continuation): Price HL vs RSI LL"
                    )

            return DivergenceResult("NONE", 0.0, 0.0, [], [], "No divergence detected")

        except Exception as e:
            logger.error(f"Divergence detection error: {e}")
            return DivergenceResult("NONE", 0.0, 0.0, [], [], f"Error: {e}")

    def _find_pivots(self, data: np.ndarray, is_high: bool = True) -> List[Tuple[int, float]]:
        """Find local peaks or troughs within the window."""
        pivots = []
        for i in range(self.pivot_window, len(data) - self.pivot_window):
            window = data[i - self.pivot_window : i + self.pivot_window + 1]
            if is_high:
                if data[i] == np.max(window):
                    pivots.append((i, data[i]))
            else:
                if data[i] == np.min(window):
                    pivots.append((i, data[i]))
        return pivots
