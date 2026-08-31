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
    brc_tier: str = None,
    tier_keyed: bool = False,
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
    brc_tier : str, optional
        RetestEngine setup tier for this signal (e.g. "RUNNER", "ZONE_LADDER").
        Only consulted when tier_keyed=True.
    tier_keyed : bool
        Gate ② re-key (17-Aug study, ships OFF). False reproduces the
        regime-keyed behavior above exactly. True keys the wick threshold on
        SETUP TYPE instead of regime direction: a 3,559-event panel showed
        wick size predicts outcome for retest-class entries (flagged retests
        -0.156R vs +0.018R clean, n=84) but not for RUNNER entries (+0.027 vs
        +0.021, n=63, indistinguishable) — a wick on a retest means the level
        didn't hold; a wick on a runner is momentum noise.

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
    #
    # ── GATE ② RE-KEY (17-Aug study) ─────────────────────────────────────
    # OLD: strict 1.0x unless regime-aligned + confident -> 1.5x. That keyed
    # the threshold on a directional claim the governor study disproved, and
    # aimed the strictest test at the better-performing (counter-trend) bucket.
    # NEW (tier_keyed=True): key on SETUP TYPE, which is what the panel shows
    # actually discriminates — strict on retest-class entries (wick = the level
    # failing to hold), lenient on RUNNER entries (wick = momentum noise, no
    # measurable relationship to outcome). 2.5x is an effective off-switch for
    # runners (only 0.7% of runner entries exceed 1.5x ATR in the panel) while
    # keeping this code path and its telemetry intact rather than branching
    # around it.
    if tier_keyed:
        _is_runner = (brc_tier or "").upper() == "RUNNER"
        wick_multiplier = 2.5 if _is_runner else 1.0
    else:
        wick_multiplier = 1.5 if (regime_aligned and regime_confidence >= 0.6) else 1.0

    logger.debug(
        f"[TRAP-TELEMETRY] mode={'tier_keyed' if tier_keyed else 'regime_keyed'} "
        f"brc_tier={brc_tier} wick_multiplier={wick_multiplier:.1f}"
    )

    _wick_blocked = upper_wick > (wick_multiplier * atr) or lower_wick > (wick_multiplier * atr)
    # FRAME-1 SEG 9: this telemetry has been at debug since it shipped -- zero
    # visible lines, ever, in any log.
    logger.info(
        f"[TRAP-TELEMETRY] wick_max={max(upper_wick, lower_wick):.4f} "
        f"threshold={wick_multiplier * atr:.4f} atr={atr:.4f} result={'BLOCK' if _wick_blocked else 'PASS'}"
    )

    if _wick_blocked:
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
        _vol_blocked = volume_rolling_avg > 0 and volume < 1.5 * volume_rolling_avg
        logger.debug(
            f"[TRAP-TELEMETRY] btc_volume={volume:.0f} avg20={volume_rolling_avg:.0f} "
            f"threshold={1.5 * volume_rolling_avg:.0f} result={'BLOCK' if _vol_blocked else 'PASS'}"
        )
        if _vol_blocked:
            logger.debug(
                f"[TRAP] BTC volume insufficient (not regime-aligned): "
                f"{volume:.0f} < 1.5x avg ({volume_rolling_avg:.0f})"
            )
            return False

    return True
