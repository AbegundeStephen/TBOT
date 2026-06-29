"""
Transition Evidence Collector
Gathers multi-source evidence for regime transitions in SLIGHTLY zones.
Feeds into the Gatekeeper's confidence scaling.

Architecture:
  - Called during CompositeState build for SLIGHTLY_BEARISH/SLIGHTLY_BULLISH regimes
  - Collects evidence from 4 independent sources (momentum, S/R, order flow, candle)
  - Outputs a TransitionEvidence score that the Gatekeeper uses to scale penalties
  - NEVER overrides the Gatekeeper — only modifies penalty severity

Evidence Sources & Weights:
  Momentum (4H EMA20 direction):  0.30
  Structure (S/R level bounces):  0.30
  Order Flow (CVD + L2 depth):    0.25
  Candle Structure (HL/LH):       0.15

Minimum 2 of 4 sources must agree for evidence to be actionable.
"""
import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransitionEvidence:
    """Evidence package for regime transition assessment."""
    momentum_score: float = 0.0
    structure_score: float = 0.0
    order_flow_score: float = 0.0
    candle_score: float = 0.0

    total_score: float = 0.0
    conditions_met: int = 0
    direction: str = "NEUTRAL"  # BULLISH_REVERSAL / BEARISH_REVERSAL / NEUTRAL

    details: str = ""


class TransitionDetector:
    """
    Collects evidence from multiple sources to detect regime transitions
    in SLIGHTLY_BEARISH / SLIGHTLY_BULLISH zones.

    This is NOT an override — it produces a score that the Gatekeeper
    uses to scale its penalty on counter-trend signals.

    Weights:
      Momentum (4H):     0.30 — fastest to respond
      Structure (S/R):   0.30 — most reliable (institutional levels)
      Order Flow (L2):   0.25 — real-time confirmation
      Candle Structure:  0.15 — lagging but confirming
    """

    WEIGHTS = {
        "momentum": 0.30,
        "structure": 0.30,
        "order_flow": 0.25,
        "candle": 0.15,
    }

    # FLOW REWEIGHT FIX: order_flow is only ever populated for BTC, via the
    # Binance CVD/L2 consumer in main.py (asset_name in ("BTC","BTCUSDT")).
    # Confirmed empirically via scripts/diagnostics/check_mt5_dom.py that this
    # broker (Exness) doesn't expose market_book_get() DOM for ANY symbol,
    # including BTCUSDm — so order_flow_score is structurally guaranteed to be
    # exactly 0.0 for every asset that isn't routed through that Binance feed.
    # Leaving its 0.25 weight in WEIGHTS doesn't bias direction (0 * 0.25 = 0),
    # but it does cap total_score's usable range at 0.75 instead of 1.0 for
    # those assets, understating real evidence from the 3 sources that DO work.
    # Redistribute proportionally (0.30/0.30/0.15 -> 0.40/0.40/0.20) whenever
    # order flow has no real source for the asset being evaluated.
    WEIGHTS_NO_FLOW = {
        "momentum": 0.40,
        "structure": 0.40,
        "order_flow": 0.0,
        "candle": 0.20,
    }

    # Assets with a real, wired order-flow source (Binance CVD/L2 via
    # main.py's cvd_consumer). Keep in sync with the gating condition there.
    #
    # "BTC" is intentionally excluded even though the CVD consumer always
    # subscribes to Binance btcusdt: every BTC asset config in this deployment
    # has exchange="mt5" (BTCUSDm) — a different instrument with a different
    # tape. Feeding MT5-priced BTC with Binance order-flow evidence blends two
    # unrelated markets. Only "BTCUSDT" (literal Binance-routed asset name)
    # gets FLOW credit. If BTC is ever moved back to Binance execution, the
    # asset_name used throughout the pipeline must become "BTCUSDT" to match.
    FLOW_CAPABLE_ASSETS = ("BTCUSDT",)

    MIN_CONDITIONS = 2

    def __init__(self):
        self._sr_bounce_history = {}

    def collect_evidence(
        self,
        asset: str,
        regime: str,
        df_4h,
        df_1h,
        composite_state,
        cvd_trend: int = 0,
        order_book_imbalance: float = 0.0,
        depth_data: Optional[Dict] = None,
    ) -> TransitionEvidence:
        """
        Collect evidence for a potential regime transition.
        Called in SLIGHTLY regimes and NEUTRAL+TRANSITION trades.

        For SLIGHTLY_BEARISH  → looks for bullish reversal evidence.
        For SLIGHTLY_BULLISH  → looks for bearish reversal evidence.
        For NEUTRAL           → direction-agnostic scan; positive total_score means
                                bullish conditions dominate, negative means bearish.
                                Used for TRANSITION trades (FX pairs in consolidation).
        """
        evidence = TransitionEvidence()
        details = []

        has_flow = asset in self.FLOW_CAPABLE_ASSETS
        weights = self.WEIGHTS if has_flow else self.WEIGHTS_NO_FLOW
        max_conditions = 4 if has_flow else 3

        if regime in ("BEARISH", "SLIGHTLY_BEARISH"):
            # In full BEARISH we look for bullish reversal evidence exactly as
            # we do in SLIGHTLY_BEARISH. The difference is that in full BEARISH
            # the gatekeeper used to never call us — now it does when the
            # post-score softener has already flagged momentum divergence.
            # Strong evidence (conditions_met >= 3) is used by the regime
            # detector's softener as a second confirmation layer.
            looking_for = "bullish"
        elif regime in ("BULLISH", "SLIGHTLY_BULLISH"):
            looking_for = "bearish"
        elif regime == "NEUTRAL":
            # TRANSITION trade: run analysis from bullish perspective.
            # A positive total_score = bullish breakout conditions dominate.
            # A negative total_score = bearish breakout conditions dominate.
            # The Gatekeeper reads the sign to determine the likely breakout direction.
            looking_for = "bullish"
        else:
            return evidence

        # ── SIGNAL 1: 4H Momentum ────────────────────────────────────
        mom_score = self._check_momentum(df_4h, looking_for)
        evidence.momentum_score = mom_score
        if abs(mom_score) > 0.3:
            evidence.conditions_met += 1
            details.append(f"MOM={'✅' if mom_score > 0 else '❌'} {mom_score:+.2f}")
        else:
            details.append(f"MOM=⚪ {mom_score:+.2f}")

        # ── SIGNAL 2: S/R Level Bounces ──────────────────────────────
        struct_score = self._check_structure(
            asset, df_4h, df_1h, composite_state, looking_for
        )
        evidence.structure_score = struct_score
        if abs(struct_score) > 0.3:
            evidence.conditions_met += 1
            details.append(f"S/R={'✅' if struct_score > 0 else '❌'} {struct_score:+.2f}")
        else:
            details.append(f"S/R=⚪ {struct_score:+.2f}")

        # ── SIGNAL 3: Order Flow (BTC only — no broker DOM for any MT5 asset) ──
        if has_flow:
            flow_score = self._check_order_flow(
                cvd_trend, order_book_imbalance, depth_data, looking_for
            )
            evidence.order_flow_score = flow_score
            if abs(flow_score) > 0.3:
                evidence.conditions_met += 1
                details.append(f"FLOW={'✅' if flow_score > 0 else '❌'} {flow_score:+.2f}")
            else:
                details.append(f"FLOW=⚪ {flow_score:+.2f}")
        else:
            flow_score = 0.0
            evidence.order_flow_score = 0.0
            details.append("FLOW=N/A (no broker DOM)")

        # ── SIGNAL 4: Candle Structure ───────────────────────────────
        candle_score = self._check_candles(df_1h, looking_for)
        evidence.candle_score = candle_score
        if abs(candle_score) > 0.3:
            evidence.conditions_met += 1
            details.append(f"CNDL={'✅' if candle_score > 0 else '❌'} {candle_score:+.2f}")
        else:
            details.append(f"CNDL=⚪ {candle_score:+.2f}")

        # ── COMPOSITE ────────────────────────────────────────────────
        evidence.total_score = (
            mom_score * weights["momentum"]
            + struct_score * weights["structure"]
            + flow_score * weights["order_flow"]
            + candle_score * weights["candle"]
        )

        if evidence.conditions_met >= self.MIN_CONDITIONS:
            if regime == "NEUTRAL":
                # TRANSITION trade: score sign determines breakout direction
                if evidence.total_score > 0.15:
                    evidence.direction = "BULLISH_BREAKOUT"
                elif evidence.total_score < -0.15:
                    evidence.direction = "BEARISH_BREAKOUT"
            else:
                # SLIGHTLY regimes: evidence of counter-trend reversal
                if looking_for == "bullish" and evidence.total_score > 0.15:
                    evidence.direction = "BULLISH_REVERSAL"
                elif looking_for == "bearish" and evidence.total_score < -0.15:
                    evidence.direction = "BEARISH_REVERSAL"

        evidence.details = " | ".join(details)
        logger.info(
            f"[TRANSITION] {asset} ({regime}): "
            f"score={evidence.total_score:+.3f} "
            f"conditions={evidence.conditions_met}/{max_conditions} "
            f"dir={evidence.direction} | {evidence.details}"
        )

        return evidence

    def _check_momentum(self, df_4h, looking_for: str) -> float:
        """4H EMA20 momentum direction and strength. Returns -1.0 to +1.0.

        ROC THRESHOLD FIX: the raw roc thresholds (0.002/0.001) were calibrated
        against BTC's volatility. A smoothed 20-period MA on Gold/EURUSD/EURJPY/
        USTEC routinely won't move 0.1-0.2% in a 5-bar (20H) window even during a
        real drift, which silently starved MOM evidence for every non-crypto
        asset. Normalizing roc by 4H ATR% makes the same thresholds meaningful
        across assets of different volatility instead of one flat magic number.
        """
        try:
            if df_4h is None or len(df_4h) < 25:
                return 0.0

            closes = df_4h['close'].values.astype(float)
            highs = df_4h['high'].values.astype(float)
            lows = df_4h['low'].values.astype(float)

            # Simple moving average as EMA proxy (avoids talib dependency)
            ema20 = np.convolve(closes, np.ones(20) / 20, mode='valid')
            if len(ema20) < 6:
                return 0.0

            roc = (ema20[-1] - ema20[-6]) / max(abs(ema20[-6]), 1e-9)

            atr_pct = self._atr_pct(highs, lows, closes, period=14)
            roc_norm = roc / atr_pct if atr_pct > 1e-9 else 0.0

            # Price vs 50-period average
            price_above_50 = None
            if len(closes) >= 50:
                ema50 = np.convolve(closes, np.ones(50) / 50, mode='valid')
                if len(ema50) > 0:
                    price_above_50 = closes[-1] > ema50[-1]

            score = 0.0
            if looking_for == "bullish":
                if roc_norm > 0.50:
                    score += 0.6
                elif roc_norm > 0.25:
                    score += 0.3
                if price_above_50 is True:
                    score += 0.4
            else:
                if roc_norm < -0.50:
                    score -= 0.6
                elif roc_norm < -0.25:
                    score -= 0.3
                if price_above_50 is False:
                    score -= 0.4

            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[TRANSITION] Momentum check error: {e}")
            return 0.0

    @staticmethod
    def _atr_pct(highs, lows, closes, period: int = 14) -> float:
        """Average True Range as a fraction of the latest close (e.g. 0.015 = 1.5%).
        Used to normalize momentum thresholds across assets of different volatility."""
        if len(closes) < period + 1:
            return 0.0
        prev_close = closes[:-1]
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - prev_close),
                np.abs(lows[1:] - prev_close),
            ),
        )
        atr = np.mean(tr[-period:])
        last_close = closes[-1]
        return atr / last_close if last_close > 1e-9 else 0.0

    def _check_structure(
        self, asset, df_4h, df_1h, state, looking_for: str
    ) -> float:
        """S/R level bounces. Multiple bounces = stronger evidence."""
        try:
            score = 0.0

            if state.nearby_4h_level is not None and state.level_test_count >= 2:
                if state.level_defended:
                    bounce_score = min(1.0, state.level_test_count * 0.25)

                    if df_1h is not None and len(df_1h) > 0:
                        current_price = float(df_1h['close'].iloc[-1])
                        if looking_for == "bullish" and current_price > state.nearby_4h_level:
                            score += bounce_score
                        elif looking_for == "bearish" and current_price < state.nearby_4h_level:
                            score -= bounce_score

            # Higher lows / lower highs from 1H data
            if df_1h is not None and len(df_1h) >= 4:
                lows = df_1h['low'].iloc[-4:]
                highs = df_1h['high'].iloc[-4:]
                higher_lows = (lows.iloc[-1] > lows.iloc[-2] > lows.iloc[-3])
                lower_highs = (highs.iloc[-1] < highs.iloc[-2] < highs.iloc[-3])

                if looking_for == "bullish" and higher_lows:
                    score += 0.3
                elif looking_for == "bearish" and lower_highs:
                    score -= 0.3

            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[TRANSITION] Structure check error: {e}")
            return 0.0

    def _check_order_flow(
        self, cvd_trend, imbalance, depth_data, looking_for: str
    ) -> float:
        """CVD + L2 imbalance + depth walls. Returns -1.0 to +1.0."""
        try:
            score = 0.0

            # CVD trend
            if looking_for == "bullish" and cvd_trend > 0:
                score += 0.4
            elif looking_for == "bearish" and cvd_trend < 0:
                score -= 0.4

            # Best bid/ask imbalance
            if looking_for == "bullish" and imbalance > 0.3:
                score += 0.3
            elif looking_for == "bearish" and imbalance < -0.3:
                score -= 0.3

            # L2 depth wall analysis
            if depth_data and depth_data.get("age", 999) < 60:
                bids = depth_data.get("bids", [])
                asks = depth_data.get("asks", [])
                if bids and asks:
                    try:
                        total_bid = sum(float(b[1]) for b in bids[:10])
                        total_ask = sum(float(a[1]) for a in asks[:10])
                        total = total_bid + total_ask
                        if total > 0:
                            depth_imb = (total_bid - total_ask) / total
                            if looking_for == "bullish" and depth_imb > 0.25:
                                score += 0.3
                            elif looking_for == "bearish" and depth_imb < -0.25:
                                score -= 0.3
                    except (ValueError, TypeError):
                        pass

            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[TRANSITION] Order flow check error: {e}")
            return 0.0

    def _check_candles(self, df_1h, looking_for: str) -> float:
        """Higher lows / lower highs pattern from 1H. Returns -1.0 to +1.0."""
        try:
            if df_1h is None or len(df_1h) < 6:
                return 0.0

            score = 0.0
            lows = df_1h['low'].iloc[-6:].values
            highs = df_1h['high'].iloc[-6:].values

            higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
            lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])

            if looking_for == "bullish":
                if higher_lows >= 4:
                    score += 0.8
                elif higher_lows >= 3:
                    score += 0.4
            else:
                if lower_highs >= 4:
                    score -= 0.8
                elif lower_highs >= 3:
                    score -= 0.4

            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[TRANSITION] Candle check error: {e}")
            return 0.0
