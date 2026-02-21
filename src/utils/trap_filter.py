import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_candle_structure(df: pd.DataFrame, asset_type: str, direction: str = "long") -> bool:
    """
    Validates the structure of the latest closed candle based on asset-specific rules
    to identify potential 'trap' candles.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing candle data, including 'open', 'high', 'low', 'close', 'volume'.
        For BTC, 'average_volume' is also expected.
    asset_type : str
        The type of asset, e.g., 'BTC', 'Gold'.
    direction : str
        The direction of the trade ('long' or 'short'). Defaults to 'long'.

    Returns:
    --------
    bool
        True if the candle structure is valid (not a trap), False otherwise.
    """
    if df.empty or len(df) < 1:
        logger.warning("Empty or insufficient DataFrame provided to validate_candle_structure. Returning True.")
        return True # No candle to validate, so no trap identified.

    latest_candle = df.iloc[-1]

    open_price = latest_candle['open']
    high_price = latest_candle['high']
    low_price = latest_candle['low']
    close_price = latest_candle['close']
    volume = latest_candle.get('volume') # Will be used for BTC

    body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price

    if asset_type.upper() == 'BTC':
        avg_volume = latest_candle.get('average_volume')

        # Rule 1: Volume trap
        if volume is None or avg_volume is None:
            logger.warning("Missing 'volume' or 'average_volume' for BTC validation. Skipping volume check.")
        elif volume < 1.2 * avg_volume:
            logger.debug(f"BTC Trap (Volume): {volume=} < {1.2 * avg_volume=}")
            return False

        # Rule 2: Directional Wick trap
        if direction == "long":
            if upper_wick > 0.6 * body:
                logger.debug(f"BTC Long Trap (UpperWick): {upper_wick=} > {0.6 * body=}")
                return False
        else: # short
            if lower_wick > 0.6 * body:
                logger.debug(f"BTC Short Trap (LowerWick): {lower_wick=} > {0.6 * body=}")
                return False

    elif asset_type.upper() == 'GOLD':
        # Rule 1: Directional Wick trap
        if direction == "long":
            if upper_wick > 0.4 * body:
                logger.debug(f"Gold Long Trap (UpperWick): {upper_wick=} > {0.4 * body=}")
                return False
        else: # short
            if lower_wick > 0.4 * body:
                logger.debug(f"Gold Short Trap (LowerWick): {lower_wick=} > {0.4 * body=}")
                return False
    else:
        logger.warning(f"Unsupported asset_type '{asset_type}'. No trap filter applied. Returning True.")
        return True # If asset type is not recognized, assume valid by default.

    return True
