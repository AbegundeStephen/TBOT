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

Direction confirmation (hysteresis)
-------------------------------------
The raw UP/DOWN direction above is a hard boolean (plus_di > minus_di) with
no dead-zone — when +DI/-DI sit close together it can flip every cycle. This
engine requires `confirm_cycles` (default 2) consecutive cycles of the same
raw direction per timeframe before treating it as "confirmed" for alignment
purposes, so a single noisy tick right at a DI crossover can't cause the
alert to flap.

ADX/RSI direction (rising/falling)
------------------------------------
`timeframe_data` only ever exposes the latest single-bar ADX/RSI scalar, not
a series. This engine keeps its own small rolling history per asset/timeframe
across cycles (default: last 6 cycles, ~30 min at the bot's 5-min cadence)
and classifies the slope the same way the existing ADX-slope helper in
strategies/trend_following.py does: RISING_FAST / RISING / FLAT / FALLING /
FALLING_FAST, generalized here to run on either ADX or RSI. The same rolling
window also keeps a plain close-price history, used for a simple RSI
divergence check (item 6).

Enhancements over the original alignment-only version
---------------------------------------------------------
1. Direction hysteresis (above) — reduces alert flapping right at a DI
   crossover.
2. Reads `composite_state` (support/resistance, spread_ratio) once the
   caller has it available — see `process()`'s new optional parameters.
   Callers should invoke this AFTER the aggregator call, not before, so
   composite_state actually exists (main.py wires this).
3. Uses the already-computed `momentum_dir` / `lower_highs` / `higher_lows`
   1H fields (mtf_regime_detector.py) as an extra structural-quality gate.
4. Optional 1D triple-alignment mode (`require_1d_alignment`) — stricter,
   fewer but higher-conviction alerts.
5. Spread-ratio awareness from composite_state — flags or blocks when
   spread is currently elevated relative to average.
6. Simple RSI divergence check over the rolling window (price vs. RSI).
7. Self-contained economic-calendar news-blackout check — duplicated (not
   imported) from the live gate in signal_aggregator.py so this module has
   zero dependency on the execution path; reads config/economic_calendar.json
   directly.
8. Self-scoring track record: "GO IN" alerts open a position on a small,
   dedicated ShadowTradingEngine instance (separate from the bot's own
   blocked-signal shadow tracker, so the two don't collide on the shared
   per-asset dedup/cooldown) and the alert message reports the running
   win-rate/avg-P&L once enough history exists.
9. Per-asset threshold overrides (`config["scalp_alerts"]["per_asset"]`).
"""

import json
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BULLISH_TAG = "BULLISH"
_BEARISH_TAG = "BEARISH"

# Item 7: asset -> currencies whose news events affect it. Duplicated from
# (not imported from) signal_aggregator.py's live T3.4 gate so this module
# stays a fully independent, read-only observer. Also fills a small gap in
# the live gate's own matching (GBPUSD/USDJPY have no explicit branch there
# today, so real news events never block them) — no reason to propagate
# that gap into a fresh copy.
_ASSET_CURRENCIES = {
    "BTC": {"USD"}, "BTCUSDT": {"USD"},
    "GOLD": {"USD"}, "XAUUSD": {"USD"},
    "EURUSD": {"EUR", "USD"},
    "EURJPY": {"EUR", "JPY"},
    "USTEC": {"USD"}, "US100": {"USD"}, "NAS100": {"USD"},
    "GBPAUD": {"GBP", "AUD"},
    "GBPUSD": {"GBP", "USD"},
    "USDJPY": {"USD", "JPY"},
    "USOIL": {"USD", "OIL"}, "USOILM": {"USD", "OIL"},
}


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


def _detect_price_rsi_divergence(price_history: deque, rsi_history: deque) -> Optional[str]:
    """
    Item 6: simple divergence check over whatever window is available (needs
    at least 3 samples). Compares the endpoints of the window rather than
    hunting for local swing highs/lows — a coarse but honest approximation
    given this engine only samples once per bot cycle (5 min), not per bar.

    Returns "BEARISH_DIVERGENCE" (price up, RSI down — warns against chasing
    a BULLISH alert), "BULLISH_DIVERGENCE" (price down, RSI up — warns
    against a BEARISH alert), or None.
    """
    if len(price_history) < 3 or len(rsi_history) < 3:
        return None
    price_chg = price_history[-1] - price_history[0]
    rsi_chg = rsi_history[-1] - rsi_history[0]
    # Require both moves to be non-trivial so flat/noisy windows don't
    # register as a "divergence" by accident.
    if price_chg > 0 and rsi_chg < -1.0:
        return "BEARISH_DIVERGENCE"
    if price_chg < 0 and rsi_chg > 1.0:
        return "BULLISH_DIVERGENCE"
    return None


def _check_news_blackout(asset_type: str, econ_events: List[dict]) -> Tuple[bool, str, float]:
    """
    Item 7: self-contained news-blackout check. Returns
    (blocked, event_name, minutes_to_event). Mirrors the matching semantics
    of the live T3.4 gate in signal_aggregator.py's get_aggregated_signal()
    (same block_hours_before/after window, same "no currencies listed ->
    block everything" fallback) but is a standalone copy, not a shared call,
    so this module has no import-time dependency on the execution path.
    """
    if not econ_events:
        return False, "", 0.0
    _utc_now = datetime.now(timezone.utc)
    _affected = _ASSET_CURRENCIES.get(asset_type.upper())
    for _evt in econ_events:
        try:
            _evt_time = datetime.fromisoformat(str(_evt["datetime"]).replace("Z", "+00:00"))
            _hours_before = _evt.get("block_hours_before", 2)
            _hours_after = _evt.get("block_hours_after", 0)
            _block_start = _evt_time - timedelta(hours=_hours_before)
            _block_end = _evt_time + timedelta(hours=_hours_after)
            if not (_block_start <= _utc_now < _block_end):
                continue
            _evt_currencies = _evt.get("currencies", _evt.get("currency"))
            if isinstance(_evt_currencies, str):
                _evt_currencies = [_evt_currencies]
            _evt_currencies = set(_evt_currencies or [])
            _blocked = (not _evt_currencies) or bool(
                _affected and (_affected & _evt_currencies)
            )
            if _blocked:
                _mins_to_evt = (_evt_time - _utc_now).total_seconds() / 60
                return True, str(_evt.get("event", "Unknown Event")), _mins_to_evt
        except Exception:
            continue
    return False, "", 0.0


class ScalpAlertEngine:
    """
    One instance lives on the bot for its whole runtime. `process()` is
    called once per asset per trading cycle, right after the MTF regime
    dict is refreshed AND after the aggregator call (so composite_state is
    available). Returns a ready-to-send Telegram message string, or None if
    nothing new should be announced this cycle.
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

        # Item 1: direction-flip hysteresis.
        self.confirm_cycles: int = max(1, int(config.get("confirm_cycles", 2)))

        # Item 4: optional stricter 1D-confirming mode.
        self.require_1d_alignment: bool = bool(config.get("require_1d_alignment", False))

        # Item 5: spread-ratio awareness thresholds (mirrors the Volume
        # judge's continuous tiers in council_aggregator.py for consistency).
        self.spread_ratio_caution: float = float(config.get("spread_ratio_caution", 1.5))
        self.spread_ratio_block: float = float(config.get("spread_ratio_block", 2.5))

        # Item 6: RSI divergence gate, with an escape hatch.
        self.rsi_divergence_gate_enabled: bool = bool(
            config.get("rsi_divergence_gate_enabled", True)
        )

        # Item 3: momentum/structure quality gate, with an escape hatch.
        self.momentum_structure_gate_enabled: bool = bool(
            config.get("momentum_structure_gate_enabled", True)
        )

        # Item 7: news-blackout gate, with an escape hatch and its own
        # independently-loaded, periodically-refreshed calendar file.
        self.econ_calendar_gate_enabled: bool = bool(
            config.get("econ_calendar_gate_enabled", True)
        )
        self._econ_calendar_path = Path(
            config.get("econ_calendar_path", "config/economic_calendar.json")
        )
        self._econ_events: List[dict] = []
        self._econ_events_loaded_at: float = 0.0
        self._econ_reload_seconds: float = float(config.get("econ_reload_seconds", 1800))

        # Item 8: self-scoring track record via a dedicated ShadowTradingEngine
        # instance — NOT the bot's main self.shadow_trader. Sharing that one
        # would collide: its dedup (same asset+side) and 60-min cooldown are
        # keyed by asset only, ignoring strategy_source, so frequent scalp
        # opens would silently vanish whenever the bot's own blocked-signal
        # tracking is active on the same asset. A separate instance avoids
        # that entirely. cooldown_minutes=0 because this engine's own
        # fire-once-per-alignment-event dedup already prevents re-tracking
        # the same undissolved setup — once it breaks and re-forms, a fresh
        # shadow trade is exactly what we want, not an artificial 60min gap.
        self.track_record_enabled: bool = bool(config.get("track_record_enabled", True))
        self._shadow = None
        if self.track_record_enabled:
            try:
                from src.execution.shadow_trader import ShadowTradingEngine
                self._shadow = ShadowTradingEngine(
                    max_positions=100,
                    max_closed=2000,
                    cooldown_minutes=0,
                    archive_dir=config.get("shadow_archive_dir", "logs/shadow_scalp"),
                )
                try:
                    self._shadow.load_state(lookback_days=30)
                except Exception as _le:
                    logger.debug(f"[SCALP ALERT] Shadow state restore skipped: {_le}")
            except Exception as e:
                logger.warning(f"[SCALP ALERT] Track record disabled — shadow engine init failed: {e}")
                self._shadow = None

        # Item 9: per-asset overrides for min_adx / rsi_overbought / rsi_oversold.
        self.per_asset_overrides: Dict[str, dict] = dict(config.get("per_asset") or {})

        # Rolling per-asset, per-timeframe ADX/RSI/price history across cycles.
        self._adx_history: Dict[str, Dict[str, deque]] = {}
        self._rsi_history: Dict[str, Dict[str, deque]] = {}
        self._price_history: Dict[str, Dict[str, deque]] = {}

        # Item 1: raw-direction streak tracker per asset/timeframe, feeding
        # the confirmed-direction hysteresis.
        self._direction_streak: Dict[str, Dict[str, Tuple[Optional[str], int]]] = {}

        # Dedup: last-fired direction per asset ("BULLISH" / "BEARISH" / None).
        self._last_alert_direction: Dict[str, Optional[str]] = {}

        if self.enabled:
            scope = "ALL" if not self.asset_whitelist else sorted(self.asset_whitelist)
            logger.info(
                f"[SCALP ALERT] Enabled — min_adx={self.min_adx}, "
                f"rsi_zone=({self.rsi_oversold},{self.rsi_overbought}), "
                f"window={self.window} cycles, confirm_cycles={self.confirm_cycles}, "
                f"require_1d={self.require_1d_alignment}, "
                f"track_record={'on' if self._shadow is not None else 'off'}, "
                f"assets={scope}"
            )

    # -- internals ---------------------------------------------------------

    def _threshold(self, asset_name: str, key: str, default: float) -> float:
        """Item 9: per-asset override lookup, falling back to the global default."""
        override = self.per_asset_overrides.get(asset_name.upper(), {})
        return float(override.get(key, default))

    @staticmethod
    def _direction_from_trend(trend_direction: Optional[str]) -> Optional[str]:
        # trend_direction comes from the ADX +DI/-DI comparison — always
        # "UP" or "DOWN" (or "N/A"/"SIDEWAYS" on a missing/flat column),
        # never "NEUTRAL". This is what makes alignment detection possible
        # on quiet bot days.
        if trend_direction == "UP":
            return _BULLISH_TAG
        if trend_direction == "DOWN":
            return _BEARISH_TAG
        return None

    def _confirm_direction(
        self, asset: str, tf: str, raw_direction: Optional[str]
    ) -> Optional[str]:
        """
        Item 1: hysteresis. Only returns a direction once it's been the raw
        reading for `confirm_cycles` consecutive calls; otherwise returns
        None (not-yet-confirmed), which fails alignment for this cycle
        rather than flapping on a single noisy DI crossover.
        """
        per_tf = self._direction_streak.setdefault(asset, {})
        last_raw, streak = per_tf.get(tf, (None, 0))
        if raw_direction is not None and raw_direction == last_raw:
            streak += 1
        else:
            streak = 1 if raw_direction is not None else 0
        per_tf[tf] = (raw_direction, streak)
        if raw_direction is not None and streak >= self.confirm_cycles:
            return raw_direction
        return None

    def _push(
        self, asset: str, tf: str, adx: Optional[float], rsi: Optional[float],
        price: Optional[float] = None,
    ) -> Tuple[deque, deque, deque]:
        adx_dq = self._adx_history.setdefault(asset, {}).setdefault(
            tf, deque(maxlen=self.window)
        )
        rsi_dq = self._rsi_history.setdefault(asset, {}).setdefault(
            tf, deque(maxlen=self.window)
        )
        price_dq = self._price_history.setdefault(asset, {}).setdefault(
            tf, deque(maxlen=self.window)
        )
        if adx is not None:
            adx_dq.append(float(adx))
        if rsi is not None:
            rsi_dq.append(float(rsi))
        if price is not None:
            price_dq.append(float(price))
        return adx_dq, rsi_dq, price_dq

    def _load_econ_events(self) -> List[dict]:
        """Item 7: periodic reload of the calendar file, independent of the
        live gate's own CalendarUpdater hot-reload mechanism."""
        import time as _time
        now = _time.time()
        if self._econ_events and (now - self._econ_events_loaded_at) < self._econ_reload_seconds:
            return self._econ_events
        try:
            with open(self._econ_calendar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._econ_events = data.get("events", []) or []
            self._econ_events_loaded_at = now
        except Exception as e:
            logger.debug(f"[SCALP ALERT] Could not load {self._econ_calendar_path}: {e}")
            self._econ_events = self._econ_events or []
        return self._econ_events

    def _advisory(
        self,
        asset_name: str,
        direction: str,
        adx_1h: Optional[float],
        adx_4h: Optional[float],
        adx_slope_1h: str,
        adx_slope_4h: str,
        rsi_1h: Optional[float],
        rsi_4h: Optional[float],
        momentum_1h: Optional[Dict] = None,
        spread_ratio: Optional[float] = None,
        divergence: Optional[str] = None,
        dir_1d: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Conservative by design: defaults to WAIT unless trend strength,
        momentum direction, and exhaustion all line up. Returns
        (verdict, reason) where verdict is "GO IN" or "WAIT".
        """
        min_adx = self._threshold(asset_name, "min_adx", self.min_adx)
        rsi_overbought = self._threshold(asset_name, "rsi_overbought", self.rsi_overbought)
        rsi_oversold = self._threshold(asset_name, "rsi_oversold", self.rsi_oversold)

        # 0. News blackout — overrides everything else; no technical read
        #    matters a few minutes before high-impact news.
        if self.econ_calendar_gate_enabled:
            blocked, event_name, mins = _check_news_blackout(
                asset_name, self._load_econ_events()
            )
            if blocked:
                return "WAIT", f"News blackout — {event_name} in {mins:.0f}min"

        # 0b. Spread blown out — a scalp's small target can't absorb it.
        if spread_ratio is not None and spread_ratio >= self.spread_ratio_block:
            return "WAIT", f"Spread blown out ({spread_ratio:.1f}x avg) — cost too high for a scalp"

        # 1. Trend strength gate — both timeframes need real conviction,
        #    not just a coin-flip regime that happened to dodge NEUTRAL.
        if adx_1h is None or adx_4h is None or adx_1h < min_adx or adx_4h < min_adx:
            return (
                "WAIT",
                f"ADX below {min_adx:.0f} on at least one timeframe — weak/choppy trend",
            )

        # 1b. Optional stricter 1D confirmation.
        if self.require_1d_alignment and dir_1d != direction:
            return "WAIT", f"1D not confirming (1D={dir_1d or 'flat'}) — require_1d_alignment is on"

        # 2. Exhaustion gate — don't chase a move that's already stretched.
        if direction == _BULLISH_TAG and (
            (rsi_1h is not None and rsi_1h >= rsi_overbought)
            or (rsi_4h is not None and rsi_4h >= rsi_overbought)
        ):
            return "WAIT", "RSI overbought on at least one timeframe — risk of pullback"
        if direction == _BEARISH_TAG and (
            (rsi_1h is not None and rsi_1h <= rsi_oversold)
            or (rsi_4h is not None and rsi_4h <= rsi_oversold)
        ):
            return "WAIT", "RSI oversold on at least one timeframe — risk of bounce"

        # 2b. RSI divergence — a contrary divergence is a stronger, earlier
        #     warning than a plain overbought/oversold level.
        if self.rsi_divergence_gate_enabled:
            if direction == _BULLISH_TAG and divergence == "BEARISH_DIVERGENCE":
                return "WAIT", "Bearish RSI divergence forming — momentum not confirming price"
            if direction == _BEARISH_TAG and divergence == "BULLISH_DIVERGENCE":
                return "WAIT", "Bullish RSI divergence forming — momentum not confirming price"

        # 2c. Momentum/structure contradiction — 1H's own already-computed
        #     momentum_dir / lower_highs / higher_lows can disagree with the
        #     DI-based direction even while DI itself is aligned.
        if self.momentum_structure_gate_enabled and momentum_1h:
            mom_dir = momentum_1h.get("momentum_dir")
            if direction == _BULLISH_TAG and mom_dir == "DOWN":
                return "WAIT", "1H session momentum is DOWN despite DI alignment — contradicting signal"
            if direction == _BEARISH_TAG and mom_dir == "UP":
                return "WAIT", "1H session momentum is UP despite DI alignment — contradicting signal"
            if direction == _BULLISH_TAG and momentum_1h.get("lower_highs"):
                return "WAIT", "1H printing lower highs despite alignment — structure quality questionable"
            if direction == _BEARISH_TAG and momentum_1h.get("higher_lows"):
                return "WAIT", "1H printing higher lows despite alignment — structure quality questionable"

        # 3. Not enough history yet to trust a slope reading.
        if "INSUFFICIENT_DATA" in (adx_slope_1h, adx_slope_4h):
            return "WAIT", "Still building ADX history for this asset — re-check next cycle"

        # 4. Momentum-fading gate — a rolling-over ADX kills the setup even
        #    if the current level still clears the min_adx bar.
        fading = {"FALLING", "FALLING_FAST"}
        if adx_slope_1h in fading or adx_slope_4h in fading:
            return "WAIT", "ADX is falling on at least one timeframe — momentum fading"

        # 4b. Spread elevated but not blocked — allow GO IN but the message
        #     layer will still attach a caution note (see process()/peek()).

        # 5. Momentum-building — require at least one timeframe actively rising.
        rising = {"RISING", "RISING_FAST"}
        if adx_slope_1h in rising or adx_slope_4h in rising:
            return "GO IN", "ADX rising with 1H/4H aligned and RSI not extreme"

        # ADX flat on both, but everything else passes — stay cautious.
        return "WAIT", "Aligned and not stretched, but ADX is flat — no fresh momentum yet"

    @staticmethod
    def _fmt(value) -> str:
        return f"{value:.1f}" if isinstance(value, (int, float)) else "—"

    def _track_record_line(self) -> str:
        """Item 8: running win-rate/avg-P&L for GO IN alerts, once there's
        enough closed history to be meaningful (avoids a misleading "0%"
        on the very first alert)."""
        if self._shadow is None:
            return ""
        try:
            card = self._shadow.get_strategy_scorecard().get("SCALP_ALERT")
        except Exception:
            card = None
        if not card or card.get("count", 0) < 3:
            return ""
        return (
            f"\n\n📊 <i>Track record: {card['count']} alerts, "
            f"{card['win_rate']:.0f}% profitable, avg net {card['avg_net_pnl']:+.2f}%</i>"
        )

    def _maybe_open_shadow(
        self, asset_name: str, direction: str, current_price: Optional[float],
        atr: Optional[float], composite_state: Optional[dict],
    ) -> None:
        """Item 8: track a GO IN alert's hypothetical outcome. Best-effort —
        never raises, never affects the alert message itself."""
        if self._shadow is None or current_price is None or current_price <= 0:
            return
        try:
            side = "long" if direction == _BULLISH_TAG else "short"
            self._shadow.open_position(
                asset=asset_name,
                side=side,
                entry_price=current_price,
                strategy_source="SCALP_ALERT",
                gate_blocked_by="scalp_alert",
                signal_details={"source": "scalp_alert_engine"},
                atr=atr,
                composite_state=composite_state or {},
            )
        except Exception as e:
            logger.debug(f"[SCALP ALERT] Shadow open skipped for {asset_name}: {e}")

    def update_shadow_prices(self, price_map: Dict[str, float]) -> None:
        """Call once per cycle (main.py) alongside the bot's own
        self.shadow_trader price updates, so this engine's dedicated shadow
        positions actually close on SL/TP/time-stop. No-op if track_record
        is disabled."""
        if self._shadow is None:
            return
        try:
            self._shadow.tick_update_all(price_map)
            self._shadow.candle_update_all(price_map)
        except Exception as e:
            logger.debug(f"[SCALP ALERT] Shadow price update failed: {e}")

    def _render_status(
        self,
        asset_name: str,
        tf_1h: Dict,
        tf_4h: Dict,
        tf_1d: Dict,
        regime_1h: Optional[str],
        dir_1h: Optional[str],
        dir_4h: Optional[str],
        dir_1d: Optional[str],
        spread_ratio: Optional[float],
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
        price_dq_1h = self._price_history.get(asset_name, {}).get("1h", deque())

        adx_slope_1h = _classify_slope(adx_dq_1h)
        adx_slope_4h = _classify_slope(adx_dq_4h)
        rsi_slope_1h = _classify_slope(rsi_dq_1h)
        rsi_slope_4h = _classify_slope(rsi_dq_4h)
        divergence = _detect_price_rsi_divergence(price_dq_1h, rsi_dq_1h)

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
            asset_name, direction, adx_1h, adx_4h, adx_slope_1h, adx_slope_4h,
            rsi_1h, rsi_4h, momentum_1h=tf_1h, spread_ratio=spread_ratio,
            divergence=divergence, dir_1d=dir_1d,
        )
        emoji = "📈" if direction == _BULLISH_TAG else "📉"
        verdict_emoji = "✅" if verdict == "GO IN" else "⏳"

        already_fired_note = ""
        if self.fire_once_per_event and self._last_alert_direction.get(asset_name) == direction:
            already_fired_note = "\n<i>(Already pushed as an alert — this is just a re-check.)</i>"

        spread_note = ""
        if (
            spread_ratio is not None
            and self.spread_ratio_caution <= spread_ratio < self.spread_ratio_block
        ):
            spread_note = f"\n⚠️ Spread elevated ({spread_ratio:.1f}x avg) — may eat into a tight scalp"

        d1d_note = f"  |  1D {dir_1d}" if dir_1d else ""

        return (
            f"📊 <b>SCALP CHECK — {asset_name}</b>\n"
            f"{emoji} <b>{direction}</b> — 1H and 4H trend aligned ({bot_note}){d1d_note}\n\n"
            f"1H  ADX {self._fmt(adx_1h)} ({adx_slope_1h})  |  RSI {self._fmt(rsi_1h)} ({rsi_slope_1h})\n"
            f"4H  ADX {self._fmt(adx_4h)} ({adx_slope_4h})  |  RSI {self._fmt(rsi_4h)} ({rsi_slope_4h})\n\n"
            f"{verdict_emoji} <b>Advisory: {verdict}</b> — {reason}"
            f"{spread_note}"
            f"{already_fired_note}"
            f"{self._track_record_line()}"
        )

    def peek(
        self, asset_name: str, mtf_regime: Optional[Dict],
        composite_state: Optional[dict] = None,
    ) -> str:
        """
        On-demand, read-only status check for a single asset — meant for a
        manual `/scalp <ASSET>` Telegram command. Unlike `process()`, this:
          - ignores `self.enabled` and the asset whitelist (those only gate
            the automatic push alert, not a check you explicitly asked for)
          - never pushes to ADX/RSI/price history (read existing history only)
          - never touches the fire-once dedup state
          - never opens a shadow-tracking position
        So calling this can never suppress, double-fire, or otherwise
        disturb the real automatic alert's behavior.

        `composite_state` is optional (best-effort spread_ratio display) —
        the caller may not have a fresh one cached for a manual check.
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
        tf_1d = tf_data.get("1d", {}) or {}
        regime_1h = tf_1h.get("regime")
        dir_1h = self._direction_from_trend(tf_1h.get("trend_direction"))
        dir_4h = self._direction_from_trend(tf_4h.get("trend_direction"))
        dir_1d = self._direction_from_trend(tf_1d.get("trend_direction"))

        spread_ratio = None
        if composite_state:
            spread_ratio = composite_state.get("spread_ratio")

        return self._render_status(
            asset_name, tf_1h, tf_4h, tf_1d, regime_1h, dir_1h, dir_4h, dir_1d, spread_ratio
        )

    # -- public API ----------------------------------------------------------

    def process(
        self,
        asset_name: str,
        mtf_regime: Dict,
        current_price: Optional[float] = None,
        composite_state: Optional[dict] = None,
        atr: Optional[float] = None,
    ) -> Optional[str]:
        """
        Call once per asset per cycle with the dict returned by
        MTFRegimeIntegration.get_regime_for_trading(), AFTER the aggregator
        call has run (so `composite_state` is available — pass
        `details.get("composite_state")`). Returns a formatted Telegram
        message if a *new* alignment event should be announced, else None.
        Safe to call even when disabled (returns None instantly).

        `current_price` and `atr` are used only for the self-scoring shadow
        position (item 8) opened on a "GO IN" verdict — omit them and that
        feature simply skips tracking this alert, everything else works the
        same.
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
        tf_1d = tf_data.get("1d", {}) or {}

        # "regime" is kept only for display context below — it's gated by the
        # macro governor and goes NEUTRAL on quiet bot days. Alignment itself
        # is judged on trend_direction (see module docstring).
        regime_1h = tf_1h.get("regime")

        raw_dir_1h = self._direction_from_trend(tf_1h.get("trend_direction"))
        raw_dir_4h = self._direction_from_trend(tf_4h.get("trend_direction"))
        dir_1d = self._direction_from_trend(tf_1d.get("trend_direction"))

        # Item 1: hysteresis — only alignment on CONFIRMED directions counts.
        dir_1h = self._confirm_direction(asset_name, "1h", raw_dir_1h)
        dir_4h = self._confirm_direction(asset_name, "4h", raw_dir_4h)

        # Keep history warm regardless of alignment, so slopes are ready
        # the moment alignment forms instead of starting from zero. Uses the
        # RAW direction-independent indicator values, not the confirmed
        # direction — history should reflect reality every cycle.
        adx_dq_1h, rsi_dq_1h, price_dq_1h = self._push(
            asset_name, "1h", tf_1h.get("adx"), tf_1h.get("rsi"), current_price
        )
        adx_dq_4h, rsi_dq_4h, _ = self._push(
            asset_name, "4h", tf_4h.get("adx"), tf_4h.get("rsi")
        )

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
        divergence = _detect_price_rsi_divergence(price_dq_1h, rsi_dq_1h)

        spread_ratio = composite_state.get("spread_ratio") if composite_state else None

        verdict, reason = self._advisory(
            asset_name, direction, adx_1h, adx_4h, adx_slope_1h, adx_slope_4h,
            rsi_1h, rsi_4h, momentum_1h=tf_1h, spread_ratio=spread_ratio,
            divergence=divergence, dir_1d=dir_1d,
        )

        if self.fire_once_per_event:
            self._last_alert_direction[asset_name] = direction

        if verdict == "GO IN":
            self._maybe_open_shadow(asset_name, direction, current_price, atr, composite_state)

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

        spread_note = ""
        if (
            spread_ratio is not None
            and self.spread_ratio_caution <= spread_ratio < self.spread_ratio_block
        ):
            spread_note = f"\n⚠️ Spread elevated ({spread_ratio:.1f}x avg) — may eat into a tight scalp"

        d1d_note = f"  |  1D {dir_1d}" if dir_1d else ""

        # HTML parse mode, not Markdown: slope labels contain underscores
        # (RISING_FAST, INSUFFICIENT_DATA, ...) which legacy Telegram
        # Markdown interprets as italic delimiters and mangles. HTML tags
        # don't have that collision. Caller must pass parse_mode="HTML".
        message = (
            f"🎯 <b>SCALP ALERT — {asset_name}</b>\n"
            f"{emoji} <b>{direction}</b> — 1H and 4H trend aligned ({bot_note}){d1d_note}\n\n"
            f"1H  ADX {self._fmt(adx_1h)} ({adx_slope_1h})  |  RSI {self._fmt(rsi_1h)} ({rsi_slope_1h})\n"
            f"4H  ADX {self._fmt(adx_4h)} ({adx_slope_4h})  |  RSI {self._fmt(rsi_4h)} ({rsi_slope_4h})\n\n"
            f"{verdict_emoji} <b>Advisory: {verdict}</b> — {reason}"
            f"{spread_note}\n\n"
            f"<i>Personal scalp signal only — not an order, not auto-executed. "
            f"Confirm price action before entering.</i>"
            f"{self._track_record_line()}"
        )
        logger.info(
            f"[SCALP ALERT] {asset_name} {direction} aligned → {verdict} ({reason})"
        )
        return message
