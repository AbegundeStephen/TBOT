"""
Scalp Alert Engine
====================
Personal-use notification feature — OFF by default (see config["scalp_alerts"]).
Not part of the trading/execution path; purely a read-only observer that
piggybacks on the MTF regime data the bot already computes every cycle.

Fires a single Telegram alert per asset whenever the 1H and 4H timeframes
structurally align on the same direction, with ADX/RSI levels for both
timeframes, the rising/falling direction of those indicators themselves,
and a conservative "GO IN" / "WAIT" advisory.

Alignment definition
---------------------
Deliberately does NOT use `timeframe_data["1h"]["regime"]` / `["4h"]["regime"]`.
Those fields are forced to "NEUTRAL" whenever the macro 1D Constitution Gate
(governor_status) is neutral — which is exactly the condition the main bot
stays silent in, and exactly the situation this alert exists to cover. Gating
on `regime` would mean this alert could structurally never fire on a quiet
bot day, defeating its purpose.

Instead, alignment is judged on the raw `timeframe_data["1h"]["trend_direction"]`
/ `["4h"]["trend_direction"]` fields — derived straight from ADX +DI/-DI
(`np.where(plus_di > minus_di, "UP", "DOWN")` in mtf_regime_detector.py), which
is always UP or DOWN and never collapses to NEUTRAL. "1H and 4H align" here
means both timeframes' local directional indicator point the same way,
independent of whatever the macro governor/consensus regime says. The
"regime" value is still surfaced in the alert message as context (so you can
see whether this is a manual-only setup or one the bot is also acting on).

ADX/RSI direction (rising/falling)
------------------------------------
`timeframe_data` only ever exposes the latest single-bar ADX/RSI scalar, not
a series. This engine keeps its own small rolling history per asset/timeframe
across cycles (default: last 6 cycles, ~30 min at the bot's 5-min cadence)
and classifies the slope the same way the existing ADX-slope helper in
strategies/trend_following.py does: RISING_FAST / RISING / FLAT / FALLING /
FALLING_FAST, generalized here to run on either ADX or RSI.
"""

import logging
from collections import deque
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BULLISH_TAG = "BULLISH"
_BEARISH_TAG = "BEARISH"


def _classify_slope(history: deque, thresh: float = 1.0, thresh_fast: float = 3.0) -> str:
    """
    Generic rising/falling classifier for a rolling ADX or RSI history.
    Mirrors the short/medium dual-window style of compute_adx_slope() in
    trend_following.py, but works on whatever window length is available
    (no hard 6-sample requirement) so it can speak as soon as it has 2+
    cycles of history instead of staying silent for 30 minutes after boot.
    """
    n = len(history)
    if n < 2:
        return "INSUFFICIENT_DATA"

    last = history[-1]
    short_ref = history[-3] if n >= 3 else history[0]
    med_ref = history[0]

    short_slope = last - short_ref
    med_slope = last - med_ref

    if short_slope > thresh_fast and med_slope > thresh_fast * 1.5:
        return "RISING_FAST"
    if short_slope < -thresh_fast and med_slope < -thresh_fast * 1.5:
        return "FALLING_FAST"
    if short_slope > thresh or med_slope > thresh * 1.5:
        return "RISING"
    if short_slope < -thresh or med_slope < -thresh * 1.5:
        return "FALLING"
    return "FLAT"


class ScalpAlertEngine:
    """
    One instance lives on the bot for its whole runtime. `process()` is
    called once per asset per trading cycle, right after the MTF regime
    dict is refreshed. Returns a ready-to-send Telegram message string,
    or None if nothing new should be announced this cycle.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.enabled: bool = bool(config.get("enabled", False))
        self.asset_whitelist = set(config.get("assets") or [])  # empty = all assets
        self.min_adx: float = float(config.get("min_adx", 20))
        self.rsi_overbought: float = float(config.get("rsi_overbought", 70))
        self.rsi_oversold: float = float(config.get("rsi_oversold", 30))
        self.window: int = max(2, int(config.get("history_window_cycles", 6)))
        self.fire_once_per_event: bool = bool(
            config.get("fire_once_per_alignment_event", True)
        )

        # Rolling per-asset, per-timeframe ADX/RSI history across cycles.
        self._adx_history: Dict[str, Dict[str, deque]] = {}
        self._rsi_history: Dict[str, Dict[str, deque]] = {}

        # Dedup: last-fired direction per asset ("BULLISH" / "BEARISH" / None).
        self._last_alert_direction: Dict[str, Optional[str]] = {}

        if self.enabled:
            scope = "ALL" if not self.asset_whitelist else sorted(self.asset_whitelist)
            logger.info(
                f"[SCALP ALERT] Enabled — min_adx={self.min_adx}, "
                f"rsi_zone=({self.rsi_oversold},{self.rsi_overbought}), "
                f"window={self.window} cycles, assets={scope}"
            )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _direction_from_trend(trend_direction: Optional[str]) -> Optional[str]:
        # trend_direction comes from the ADX +DI/-DI comparison — always
        # "UP" or "DOWN" (or "N/A" on a missing column), never "NEUTRAL".
        # This is what makes alignment detection possible on quiet bot days.
        if trend_direction == "UP":
            return _BULLISH_TAG
        if trend_direction == "DOWN":
            return _BEARISH_TAG
        return None

    def _push(
        self, asset: str, tf: str, adx: Optional[float], rsi: Optional[float]
    ) -> Tuple[deque, deque]:
        adx_dq = self._adx_history.setdefault(asset, {}).setdefault(
            tf, deque(maxlen=self.window)
        )
        rsi_dq = self._rsi_history.setdefault(asset, {}).setdefault(
            tf, deque(maxlen=self.window)
        )
        if adx is not None:
            adx_dq.append(float(adx))
        if rsi is not None:
            rsi_dq.append(float(rsi))
        return adx_dq, rsi_dq

    def _advisory(
        self,
        direction: str,
        adx_1h: Optional[float],
        adx_4h: Optional[float],
        adx_slope_1h: str,
        adx_slope_4h: str,
        rsi_1h: Optional[float],
        rsi_4h: Optional[float],
    ) -> Tuple[str, str]:
        """
        Conservative by design: defaults to WAIT unless trend strength,
        momentum direction, and exhaustion all line up. Returns
        (verdict, reason) where verdict is "GO IN" or "WAIT".
        """
        # 1. Trend strength gate — both timeframes need real conviction,
        #    not just a coin-flip regime that happened to dodge NEUTRAL.
        if adx_1h is None or adx_4h is None or adx_1h < self.min_adx or adx_4h < self.min_adx:
            return (
                "WAIT",
                f"ADX below {self.min_adx:.0f} on at least one timeframe — weak/choppy trend",
            )

        # 2. Exhaustion gate — don't chase a move that's already stretched.
        if direction == _BULLISH_TAG and (
            (rsi_1h is not None and rsi_1h >= self.rsi_overbought)
            or (rsi_4h is not None and rsi_4h >= self.rsi_overbought)
        ):
            return "WAIT", "RSI overbought on at least one timeframe — risk of pullback"
        if direction == _BEARISH_TAG and (
            (rsi_1h is not None and rsi_1h <= self.rsi_oversold)
            or (rsi_4h is not None and rsi_4h <= self.rsi_oversold)
        ):
            return "WAIT", "RSI oversold on at least one timeframe — risk of bounce"

        # 3. Not enough history yet to trust a slope reading.
        if "INSUFFICIENT_DATA" in (adx_slope_1h, adx_slope_4h):
            return "WAIT", "Still building ADX history for this asset — re-check next cycle"

        # 4. Momentum-fading gate — a rolling-over ADX kills the setup even
        #    if the current level still clears the min_adx bar.
        fading = {"FALLING", "FALLING_FAST"}
        if adx_slope_1h in fading or adx_slope_4h in fading:
            return "WAIT", "ADX is falling on at least one timeframe — momentum fading"

        # 5. Momentum-building — require at least one timeframe actively rising.
        rising = {"RISING", "RISING_FAST"}
        if adx_slope_1h in rising or adx_slope_4h in rising:
            return "GO IN", "ADX rising with 1H/4H aligned and RSI not extreme"

        # ADX flat on both, but everything else passes — stay cautious.
        return "WAIT", "Aligned and not stretched, but ADX is flat — no fresh momentum yet"

    @staticmethod
    def _fmt(value) -> str:
        return f"{value:.1f}" if isinstance(value, (int, float)) else "—"

    def _render_status(
        self,
        asset_name: str,
        tf_1h: Dict,
        tf_4h: Dict,
        regime_1h: Optional[str],
        dir_1h: Optional[str],
        dir_4h: Optional[str],
    ) -> str:
        """
        Shared message body for `process()` and `peek()`. Always reads
        history (never mutates it) so calling this from a manual /scalp
        check can't disturb the rolling slope windows the automatic
        alert relies on.
        """
        adx_1h, adx_4h = tf_1h.get("adx"), tf_4h.get("adx")
        rsi_1h, rsi_4h = tf_1h.get("rsi"), tf_4h.get("rsi")

        adx_dq_1h = self._adx_history.get(asset_name, {}).get("1h", deque())
        adx_dq_4h = self._adx_history.get(asset_name, {}).get("4h", deque())
        rsi_dq_1h = self._rsi_history.get(asset_name, {}).get("1h", deque())
        rsi_dq_4h = self._rsi_history.get(asset_name, {}).get("4h", deque())

        adx_slope_1h = _classify_slope(adx_dq_1h)
        adx_slope_4h = _classify_slope(adx_dq_4h)
        rsi_slope_1h = _classify_slope(rsi_dq_1h)
        rsi_slope_4h = _classify_slope(rsi_dq_4h)

        bot_note = (
            "bot regime NEUTRAL — sitting this out, manual play"
            if regime_1h == "NEUTRAL"
            else f"bot regime: {regime_1h}"
        )

        aligned = dir_1h is not None and dir_1h == dir_4h

        if not aligned:
            label_1h = dir_1h or "FLAT/UNKNOWN"
            label_4h = dir_4h or "FLAT/UNKNOWN"
            return (
                f"📊 <b>SCALP CHECK — {asset_name}</b>\n"
                f"Not aligned right now — 1H trending {label_1h}, 4H trending {label_4h} ({bot_note})\n\n"
                f"1H  ADX {self._fmt(adx_1h)} ({adx_slope_1h})  |  RSI {self._fmt(rsi_1h)} ({rsi_slope_1h})\n"
                f"4H  ADX {self._fmt(adx_4h)} ({adx_slope_4h})  |  RSI {self._fmt(rsi_4h)} ({rsi_slope_4h})\n\n"
                f"<i>No setup — waiting for 1H and 4H trend direction to match.</i>"
            )

        direction = dir_1h
        verdict, reason = self._advisory(
            direction, adx_1h, adx_4h, adx_slope_1h, adx_slope_4h, rsi_1h, rsi_4h
        )
        emoji = "📈" if direction == _BULLISH_TAG else "📉"
        verdict_emoji = "✅" if verdict == "GO IN" else "⏳"

        already_fired_note = ""
        if self.fire_once_per_event and self._last_alert_direction.get(asset_name) == direction:
            already_fired_note = "\n<i>(Already pushed as an alert — this is just a re-check.)</i>"

        return (
            f"📊 <b>SCALP CHECK — {asset_name}</b>\n"
            f"{emoji} <b>{direction}</b> — 1H and 4H trend aligned ({bot_note})\n\n"
            f"1H  ADX {self._fmt(adx_1h)} ({adx_slope_1h})  |  RSI {self._fmt(rsi_1h)} ({rsi_slope_1h})\n"
            f"4H  ADX {self._fmt(adx_4h)} ({adx_slope_4h})  |  RSI {self._fmt(rsi_4h)} ({rsi_slope_4h})\n\n"
            f"{verdict_emoji} <b>Advisory: {verdict}</b> — {reason}"
            f"{already_fired_note}"
        )

    def peek(self, asset_name: str, mtf_regime: Optional[Dict]) -> str:
        """
        On-demand, read-only status check for a single asset — meant for a
        manual `/scalp <ASSET>` Telegram command. Unlike `process()`, this:
          - ignores `self.enabled` and the asset whitelist (those only gate
            the automatic push alert, not a check you explicitly asked for)
          - never pushes to ADX/RSI history (read existing history only)
          - never touches the fire-once dedup state
        So calling this can never suppress, double-fire, or otherwise
        disturb the real automatic alert's behavior.
        """
        if not mtf_regime:
            return (
                f"📊 <b>SCALP CHECK — {asset_name}</b>\n\n"
                f"No MTF regime data available yet — try again next cycle."
            )

        full_status = mtf_regime.get("full_regime_status")
        tf_data = getattr(full_status, "timeframe_data", None) if full_status else None
        if not tf_data:
            return (
                f"📊 <b>SCALP CHECK — {asset_name}</b>\n\n"
                f"No timeframe data available yet — try again next cycle."
            )

        tf_1h = tf_data.get("1h", {}) or {}
        tf_4h = tf_data.get("4h", {}) or {}
        regime_1h = tf_1h.get("regime")
        dir_1h = self._direction_from_trend(tf_1h.get("trend_direction"))
        dir_4h = self._direction_from_trend(tf_4h.get("trend_direction"))

        return self._render_status(asset_name, tf_1h, tf_4h, regime_1h, dir_1h, dir_4h)

    # -- public API ----------------------------------------------------------

    def process(self, asset_name: str, mtf_regime: Dict) -> Optional[str]:
        """
        Call once per asset per cycle with the dict returned by
        MTFRegimeIntegration.get_regime_for_trading(). Returns a formatted
        Telegram message if a *new* alignment event should be announced,
        else None. Safe to call even when disabled (returns None instantly).
        """
        if not self.enabled:
            return None
        if self.asset_whitelist and asset_name not in self.asset_whitelist:
            return None
        if not mtf_regime:
            return None

        full_status = mtf_regime.get("full_regime_status")
        tf_data = getattr(full_status, "timeframe_data", None) if full_status else None
        if not tf_data:
            return None

        tf_1h = tf_data.get("1h", {}) or {}
        tf_4h = tf_data.get("4h", {}) or {}

        # "regime" is kept only for display context below — it's gated by the
        # macro governor and goes NEUTRAL on quiet bot days. Alignment itself
        # is judged on trend_direction (see module docstring).
        regime_1h = tf_1h.get("regime")
        regime_4h = tf_4h.get("regime")

        dir_1h = self._direction_from_trend(tf_1h.get("trend_direction"))
        dir_4h = self._direction_from_trend(tf_4h.get("trend_direction"))

        # Keep history warm regardless of alignment, so slopes are ready
        # the moment alignment forms instead of starting from zero.
        adx_dq_1h, rsi_dq_1h = self._push(asset_name, "1h", tf_1h.get("adx"), tf_1h.get("rsi"))
        adx_dq_4h, rsi_dq_4h = self._push(asset_name, "4h", tf_4h.get("adx"), tf_4h.get("rsi"))

        aligned = dir_1h is not None and dir_1h == dir_4h

        if not aligned:
            if self.fire_once_per_event:
                self._last_alert_direction[asset_name] = None
            return None

        direction = dir_1h

        if self.fire_once_per_event and self._last_alert_direction.get(asset_name) == direction:
            return None  # already announced this alignment event — stay quiet until it breaks

        adx_1h, adx_4h = tf_1h.get("adx"), tf_4h.get("adx")
        rsi_1h, rsi_4h = tf_1h.get("rsi"), tf_4h.get("rsi")

        adx_slope_1h = _classify_slope(adx_dq_1h)
        adx_slope_4h = _classify_slope(adx_dq_4h)
        rsi_slope_1h = _classify_slope(rsi_dq_1h)
        rsi_slope_4h = _classify_slope(rsi_dq_4h)

        verdict, reason = self._advisory(
            direction, adx_1h, adx_4h, adx_slope_1h, adx_slope_4h, rsi_1h, rsi_4h
        )

        if self.fire_once_per_event:
            self._last_alert_direction[asset_name] = direction

        emoji = "📈" if direction == _BULLISH_TAG else "📉"
        verdict_emoji = "✅" if verdict == "GO IN" else "⏳"

        # Surface whether the bot's own macro regime agrees with this
        # trend-aligned setup, or whether it's sitting it out (the case this
        # alert exists for in the first place).
        bot_note = (
            "bot regime NEUTRAL — sitting this out, manual play"
            if regime_1h == "NEUTRAL"
            else f"bot regime: {regime_1h}"
        )

        # HTML parse mode, not Markdown: slope labels contain underscores
        # (RISING_FAST, INSUFFICIENT_DATA, ...) which legacy Telegram
        # Markdown interprets as italic delimiters and mangles. HTML tags
        # don't have that collision. Caller must pass parse_mode="HTML".
        message = (
            f"🎯 <b>SCALP ALERT — {asset_name}</b>\n"
            f"{emoji} <b>{direction}</b> — 1H and 4H trend aligned ({bot_note})\n\n"
            f"1H  ADX {self._fmt(adx_1h)} ({adx_slope_1h})  |  RSI {self._fmt(rsi_1h)} ({rsi_slope_1h})\n"
            f"4H  ADX {self._fmt(adx_4h)} ({adx_slope_4h})  |  RSI {self._fmt(rsi_4h)} ({rsi_slope_4h})\n\n"
            f"{verdict_emoji} <b>Advisory: {verdict}</b> — {reason}\n\n"
            f"<i>Personal scalp signal only — not an order, not auto-executed. "
            f"Confirm price action before entering.</i>"
        )
        logger.info(
            f"[SCALP ALERT] {asset_name} {direction} aligned → {verdict} ({reason})"
        )
        return message
