"""
pullback_completion.py — S7.3 (SHADOW→LIVE)

Scores how "complete" a pullback is before committing to a continuation entry.
Returns (c_pb, detail_dict) where c_pb is 0..1 confidence.

Deployment class: SHADOW→LIVE
- Starts inert: phase_config.pullback_completion_enabled = false
- Enable live when:
    (a) S5 shadow-dedup deployed
    (b) >= 30 real shadow signals scored
    (c) C_pb-discounted set shows equal-or-better realized expectancy than raw
- Fails OPEN (returns 1.0) on any error so it can never silently kill a signal.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def pullback_completion_score(df, side, lsm_state_4h, lsm_age_4h, anchor, weights):
    """
    Score pullback completion from 5 OHLCV factors (closed 1H bars only).
    BOS amplifies conviction — never vetoes.

    Parameters
    ----------
    df            : pd.DataFrame  1H OHLCV (at least 30 bars, latest bar = current)
    side          : str           "long" or "short"
    lsm_state_4h  : str | None    4H Livermore state (ages momentum/volume scores)
    lsm_age_4h    : int | None    bars since 4H state transition
    anchor        : float | None  Livermore main-up / main-down anchor price
    weights       : dict          factor weights from config trade_management block

    Returns
    -------
    (c_pb, detail) : (float, dict)
        c_pb   — 0..1 completion confidence (1.0 = fail-open)
        detail — per-factor scores for logging
    """
    try:
        import talib

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        vol_col = df.get("tick_volume", df.get("volume")) if hasattr(df, "get") else None
        if vol_col is None:
            try:
                vol_col = df["tick_volume"]
            except Exception:
                try:
                    vol_col = df["volume"]
                except Exception:
                    vol_col = None
        vol = vol_col.values if vol_col is not None else np.ones_like(close)

        w = weights or {}

        # ── Factor 1: Fibonacci retracement depth ────────────────────────────
        s_fib = 0.0
        if anchor:
            imp_hi = np.max(high[-30:])
            imp_lo = np.min(low[-30:])
            rng    = (imp_hi - imp_lo) or 1e-9
            depth  = (imp_hi - close[-1]) / rng if side == "long" else (close[-1] - imp_lo) / rng
            if 0.236 <= depth <= 0.786:
                s_fib = max(0.0, 1.0 - abs(depth - 0.5) / 0.286)

        # ── Factor 2: EMA confluence ──────────────────────────────────────────
        ema20 = talib.EMA(close, 20)[-1]
        ema50 = talib.EMA(close, 50)[-1]
        s_ema = 0.0
        for ema in (ema20, ema50):
            if not np.isnan(ema):
                if side == "long" and close[-1] >= ema >= close[-2]:
                    s_ema = max(s_ema, 1.0)
                elif side == "short" and close[-1] <= ema <= close[-2]:
                    s_ema = max(s_ema, 1.0)
                else:
                    gap_pct = abs(close[-1] - ema) / (abs(ema) * 0.003 + 1e-9)
                    s_ema = max(s_ema, 0.5 * max(0.0, 1.0 - min(1.0, gap_pct)))

        # ── Factor 3: Momentum turning (RSI) — age-weighted ──────────────────
        rsi   = talib.RSI(close, 14)
        s_mom = 0.0
        if len(rsi) >= 3 and not np.isnan(rsi[-1]):
            if side == "long" and rsi[-1] > rsi[-2]:
                s_mom = min(1.0, (rsi[-1] - rsi[-2]) / 5.0 + 0.5)
            elif side == "short" and rsi[-1] < rsi[-2]:
                s_mom = min(1.0, (rsi[-2] - rsi[-1]) / 5.0 + 0.5)
        age    = max(0.5, min(1.0, (lsm_age_4h or 1) / 5.0))
        s_mom *= age

        # ── Factor 4: Range contraction → expansion (volume confirmed) ───────
        s_vol = 0.0
        if len(close) >= 7:
            rng_now  = high[-1] - low[-1]
            rng_prev = np.mean(high[-4:-1] - low[-4:-1])
            contracted = np.mean(high[-4:-1] - low[-4:-1]) < np.mean(high[-7:-4] - low[-7:-4])
            expanding  = rng_now > rng_prev and vol[-1] > (np.mean(vol[-4:-1]) or 1e-9)
            s_vol = 1.0 if (contracted and expanding) else (0.5 if expanding else 0.0)
        s_vol *= age

        # ── Factor 5: Break of Structure (BOS) — amplifies, never vetoes ─────
        s_bos = 0.0
        lb    = min(10, len(close) - 2)
        if lb > 2:
            if side == "long":
                swing_hi = np.max(high[-lb - 1:-1])
                s_bos = 1.0 if close[-1] > swing_hi else (0.5 if high[-1] > swing_hi else 0.0)
            else:
                swing_lo = np.min(low[-lb - 1:-1])
                s_bos = 1.0 if close[-1] < swing_lo else (0.5 if low[-1] < swing_lo else 0.0)

        # ── Composite ─────────────────────────────────────────────────────────
        evidence = (
            w.get("fib", 0.20) * s_fib
            + w.get("ema", 0.20) * s_ema
            + w.get("mom", 0.20) * s_mom
            + w.get("vol", 0.15) * s_vol
        )
        c_pb = max(0.0, min(1.0, evidence * (0.6 + 0.4 * s_bos)))

        detail = {
            "fib":  round(s_fib,  2),
            "ema":  round(s_ema,  2),
            "mom":  round(s_mom,  2),
            "vol":  round(s_vol,  2),
            "bos":  round(s_bos,  2),
            "c_pb": round(c_pb,   2),
        }
        return c_pb, detail

    except Exception as e:
        logger.debug(f"[PULLBACK] score error (fail-open, non-blocking): {e}")
        return 1.0, {"error": str(e), "c_pb": 1.0}
