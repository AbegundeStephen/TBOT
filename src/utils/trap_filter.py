import pandas as pd
import logging
import talib as ta
import numpy as np

logger = logging.getLogger(__name__)


def validate_candle_structure(
    df: pd.DataFrame,
    asset_type: str,
    direction: str = "long",
    regime_confidence: float = 0.0,
    regime_aligned: bool = False,
) -> bool:
    """
    Regime-aware candle structure validation. (T2.3 — replaces fixed 1.0x ATR version)

    Changes from previous version:
    - Wick threshold is 1.0x ATR by default, raised to 1.5x when the signal is
      regime-aligned AND regime_confidence >= 0.6. In confirmed trends, normal
      pullback wicks regularly exceed 1x ATR and should not block entries.
    - BTC volume surge is only required when regime_aligned=False. In a confirmed
      trend with institutional backing, quiet-period breakouts are valid.

    Simulation data: previous fixed 1.0x threshold blocked 47 signals with
    76.6% WR and +13.3% P&L.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data.
    asset_type : str
        Asset identifier, e.g. 'BTC', 'GOLD'.
    direction : str
        Trade direction: 'long' or 'short'. Defaults to 'long'.
    regime_confidence : float
        Confidence of the current regime (0.0–1.0). Passed from signal_aggregator.
    regime_aligned : bool
        True when the signal direction matches the current macro regime.
        Raised wick threshold only activates when this is True AND regime_confidence >= 0.6.

    Returns
    -------
    bool
        True if candle structure is valid (not a trap), False if blocked.
    """
    if df.empty or len(df) < 15:
        # Need at least 15 bars for ATR(14)
        return True

    latest = df.iloc[-1]
    o = latest['open']
    h = latest['high']
    l = latest['low']
    c = latest['close']

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    atr = ta.ATR(highs, lows, closes, timeperiod=14)[-1]

    if np.isnan(atr) or atr <= 0:
        return True

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # ================================================================
    # 1. REGIME-AWARE WICK THRESHOLD
    # ================================================================
    # In confirmed trends (regime_aligned + high confidence), normal
    # retracement wicks exceed 1x ATR and should not block valid entries.
    # Raise the threshold to 1.5x only when the regime backing is strong.
    wick_multiplier = 1.5 if (regime_aligned and regime_confidence >= 0.6) else 1.0

    if upper_wick > (wick_multiplier * atr) or lower_wick > (wick_multiplier * atr):
        logger.info(
            f"[TRAP] ❌ BLOCKED — Wick {max(upper_wick, lower_wick):.4f} > "
            f"{wick_multiplier:.1f}x ATR {atr:.4f} "
            f"(regime_aligned={regime_aligned}, conf={regime_confidence:.2f})"
        )
        return False

    # ================================================================
    # 2. BTC INSTITUTIONAL VOLUME CHECK
    # ================================================================
    # Only require volume surge when NOT regime-aligned. In a confirmed
    # trend the institutional participation is already implied by the regime.
    # Previously required unconditionally — blocked quiet-period breakouts.
    if 'BTC' in asset_type.upper() and not regime_aligned:
        volume = latest.get('volume', 0)
        volume_rolling_avg = df['volume'].iloc[-21:-1].mean()
        if volume_rolling_avg > 0 and volume < 1.5 * volume_rolling_avg:
            logger.debug(
                f"[TRAP] BTC volume insufficient (not regime-aligned): "
                f"{volume:.0f} < 1.5x avg ({volume_rolling_avg:.0f})"
            )
            return False

    return True
