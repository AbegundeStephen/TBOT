import pandas as pd
import logging
import talib as ta
import numpy as np

logger = logging.getLogger(__name__)

def validate_candle_structure(df: pd.DataFrame, asset_type: str, direction: str = "long") -> bool:
    """
    Validates the structure of the latest closed candle using Absolute ATR thresholds.
    Identifies 'trap' candles such as institutional stop-hunts (excessive wicks).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing candle data, including 'open', 'high', 'low', 'close', 'volume'.
    asset_type : str
        The type of asset, e.g., 'BTC', 'Gold'.
    direction : str
        The direction of the trade ('long' or 'short'). Defaults to 'long'.

    Returns:
    --------
    bool
        True if the candle structure is valid (not a trap), False otherwise.
    """
    if df.empty or len(df) < 15:
        # Need at least 15 bars for ATR(14)
        return True

    latest_candle = df.iloc[-1]

    open_price = latest_candle['open']
    high_price = latest_candle['high']
    low_price = latest_candle['low']
    close_price = latest_candle['close']
    volume = latest_candle.get('volume')

    # Calculate ATR(14)
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
    atr = ta.ATR(highs, lows, closes, timeperiod=14)[-1]
    
    if np.isnan(atr) or atr <= 0:
        return True

    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price

    # ================================================================
    # 1. INSTITUTIONAL VOLUME TRAP (BTC Only)
    # ================================================================
    if 'BTC' in asset_type.upper():
        # Institutional requirement: 1.5x Volume Surge
        # Using a 20-period rolling average for consistency with council
        volume_rolling_avg = df['volume'].iloc[-21:-1].mean()
        if volume_rolling_avg > 0:
            if volume < 1.5 * volume_rolling_avg:
                logger.debug(f"BTC Trap (Insufficient Volume): {volume} < 1.5x avg ({volume_rolling_avg:.2f})")
                return False

    # ================================================================
    # 2. ABSOLUTE ATR WICK TRAP (Universal)
    # ================================================================
    # Rule A: Block excessive wicks (Institutional stop-hunts)
    if upper_wick > (1.0 * atr) or lower_wick > (1.0 * atr):
        logger.info(
            f"[TRAP] ❌ BLOCKED - Institutional Wick Trap detected.\n"
            f"  Upper Wick: {upper_wick:.4f}\n"
            f"  Lower Wick: {lower_wick:.4f}\n"
            f"  ATR(14):    {atr:.4f} (Threshold: 1.0x)"
        )
        return False

    # Rule B: Pass spread noise
    # If the wicks are less than 0.5 * ATR, they are considered negligible noise.
    # Since we removed all % body logic, small wicks simply pass by default.
    if upper_wick < (0.5 * atr) and lower_wick < (0.5 * atr):
        return True

    return True
