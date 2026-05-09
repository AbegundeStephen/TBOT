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
        Only called in SLIGHTLY regimes.
        """
        evidence = TransitionEvidence()
        details = []

        if regime == "SLIGHTLY_BEARISH":
            looking_for = "bullish"
        elif regime == "SLIGHTLY_BULLISH":
            looking_for = "bearish"
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

        # ── SIGNAL 3: Order Flow ─────────────────────────────────────
        flow_score = self._check_order_flow(
            cvd_trend, order_book_imbalance, depth_data, looking_for
        )
        evidence.order_flow_score = flow_score
        if abs(flow_score) > 0.3:
            evidence.conditions_met += 1
            details.append(f"FLOW={'✅' if flow_score > 0 else '❌'} {flow_score:+.2f}")
        else:
            details.append(f"FLOW=⚪ {flow_score:+.2f}")

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
            mom_score * self.WEIGHTS["momentum"]
            + struct_score * self.WEIGHTS["structure"]
            + flow_score * self.WEIGHTS["order_flow"]
            + candle_score * self.WEIGHTS["candle"]
        )

        if evidence.conditions_met >= self.MIN_CONDITIONS:
            if looking_for == "bullish" and evidence.total_score > 0.15:
                evidence.direction = "BULLISH_REVERSAL"
            elif looking_for == "bearish" and evidence.total_score < -0.15:
                evidence.direction = "BEARISH_REVERSAL"

        evidence.details = " | ".join(details)
        logger.info(
            f"[TRANSITION] {asset} ({regime}): "
            f"score={evidence.total_score:+.3f} "
            f"conditions={evidence.conditions_met}/4 "
            f"dir={evidence.direction} | {evidence.details}"
        )

        return evidence

    def _check_momentum(self, df_4h, looking_for: str) -> float:
        """4H EMA20 momentum direction and strength. Returns -1.0 to +1.0."""
        try:
            if df_4h is None or len(df_4h) < 25:
                return 0.0

            closes = df_4h['close'].values.astype(float)
            # Simple moving average as EMA proxy (avoids talib dependency)
            ema20 = np.convolve(closes, np.ones(20) / 20, mode='valid')
            if len(ema20) < 6:
                return 0.0

            roc = (ema20[-1] - ema20[-6]) / max(abs(ema20[-6]), 1e-9)

            # Price vs 50-period average
            price_above_50 = None
            if len(closes) >= 50:
                ema50 = np.convolve(closes, np.ones(50) / 50, mode='valid')
                if len(ema50) > 0:
                    price_above_50 = closes[-1] > ema50[-1]

            score = 0.0
            if looking_for == "bullish":
                if roc > 0.002:
                    score += 0.6
                elif roc > 0.001:
                    score += 0.3
                if price_above_50 is True:
                    score += 0.4
            else:
                if roc < -0.002:
                    score -= 0.6
                elif roc < -0.001:
                    score -= 0.3
                if price_above_50 is False:
                    score -= 0.4

            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"[TRANSITION] Momentum check error: {e}")
            return 0.0

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
