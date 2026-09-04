"""
Livermore State Machine — Phase 1 Brain.

Classifies the current market chapter into one of 6 Jesse Livermore states:

  MAIN_UP          — price making higher highs; trend entries allowed
  NATURAL_RETRACEMENT     — normal pullback after MAIN_UP; is_silent_zone=True
  SECONDARY_RETRACEMENT   — deeper probe below natural low; structural warning
  MAIN_DOWN        — price making lower lows; trend entries (short) allowed
  NATURAL_REBOUND  — normal bounce after MAIN_DOWN; is_silent_zone=True
  SECONDARY_REBOUND— deeper push above natural high; structural warning

Key design decisions (all debugged via ATR calibration on BTC 4H 2024-2026):
  1. nl_confirmed uses LOCK-ONCE bounce confirmation — not a running watermark.
     This prevents the "treadmill" where natural_low chases price down forever.
  2. SECONDARY_RETRACEMENT uses nl_entry (FIXED pivot at secondary entry) for all
     threshold comparisons — not sec_low. Eliminates second treadmill.
  3. MAIN state transitions require dual_confirm consecutive closes.
  4. ATR pivots are recalculated every bar (adaptive to changing volatility).

Calibrated multipliers (BTC 4H):  major=3.5, minor=1.0
  → 1.48 transitions/week, avg state duration 28 bars (~4.7 days)
  → 5/6 known turning points correctly classified

Integration:
  - Two instances per asset: one for 4H (trend), one for 1H (entry timing)
  - Called from signal_aggregator.py before any strategy scoring
  - Writes to CompositeState fields: livermore_state_4h/1h, anchors,
    state ages, livermore_dual_confirmation, is_silent_zone

Phase 1 — MRS §7.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Deque
from collections import deque

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
STATES = frozenset([
    "MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT",
    "MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND",
])
SILENT_STATES = frozenset(["NATURAL_RETRACEMENT", "NATURAL_REBOUND"])
UP_STATES     = frozenset(["MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT"])
DOWN_STATES   = frozenset(["MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND"])


@dataclass
class LivermoreSnapshot:
    """
    Immutable output of one bar's state machine update.
    Read by signal_aggregator.py → written to CompositeState.
    """
    state: str                                  # one of the 6 Livermore states
    state_age: int                              # bars since last state change
    is_silent_zone: bool                        # True = NATURAL_RETRACEMENT or NATURAL_REBOUND
    dual_confirmation: bool                     # True for dual-confirm MAIN state
    # Anchor prices (None until first relevant transition)
    anchor_main_up_max: Optional[float]         # highest close in current MAIN_UP
    anchor_main_down_min: Optional[float]       # lowest close in current MAIN_DOWN
    anchor_natural_high: Optional[float]        # nh_confirmed pivot
    anchor_natural_low: Optional[float]         # nl_confirmed pivot


class LivermoreStateMachine:
    """
    Single-timeframe Livermore state machine.

    Usage:
        lsm = LivermoreStateMachine(asset="BTC", timeframe="4H",
                                    major_mult=3.5, minor_mult=1.0,
                                    dual_confirm=2, atr_period=14)
        for _, row in df.iterrows():
            snap = lsm.update(close=row['close'], atr=row['atr'])
        current = lsm.snapshot()
    """

    def __init__(
        self,
        asset: str,
        timeframe: str,
        major_mult: float = 3.5,
        minor_mult: float = 1.0,
        dual_confirm: int = 2,
        atr_period: int = 14,
    ):
        self.asset     = asset
        self.timeframe = timeframe
        self.major_mult  = major_mult
        self.minor_mult  = minor_mult
        self.dual_confirm = dual_confirm
        self.atr_period  = atr_period

        # ── Internal state ─────────────────────────────────────────────────
        self._state: str = "MAIN_UP"
        self._state_age: int = 0

        # Upside anchors
        self._main_up_max: Optional[float]    = None
        self._nl_watermark: Optional[float]   = None   # running min in NATURAL_RETRACEMENT
        self._nl_bounce_high: Optional[float] = None   # running max after last low
        self._nl_confirmed: Optional[float]   = None   # LOCKED natural_low pivot
        self._nl_entry: Optional[float]       = None   # FIXED at secondary entry

        # Downside anchors
        self._main_down_min: Optional[float]  = None
        self._nh_watermark: Optional[float]   = None   # running max in NATURAL_REBOUND
        self._nh_dip_low: Optional[float]     = None   # running min after last high
        self._nh_confirmed: Optional[float]   = None   # LOCKED natural_high pivot
        self._nh_entry: Optional[float]       = None   # FIXED at secondary entry

        # Dual-confirm counters
        self._pending_down: int = 0
        self._pending_up: int   = 0

        # Last transition info (for logging)
        self._last_transition: Optional[str] = None

        logger.debug(
            "[Livermore] %s %s initialised | major=%.1f minor=%.1f dual=%d",
            asset, timeframe, major_mult, minor_mult, dual_confirm,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def _retire_stale_pivots(self, close: float, mn: float) -> None:
        """
        STOP-1 SEG I: retire a confirmed natural pivot the market has left.

        The existing demotion at _natural_retr / _natural_reb only fires while
        the machine is IN the state that created the pivot. Once price breaks
        through and the machine moves to MAIN_UP or MAIN_DOWN, that branch
        stops running and the pivot is never cleaned up -- confirmed live on
        GOLD, whose natural_low sat at 4604.957 for 19 hours with price at
        4462.8, 142 points below it.

        This runs every bar regardless of state. A confirmed natural LOW that
        price has closed decisively below is not a low any more; likewise a
        natural HIGH price has closed decisively above. Decisively = beyond one
        minor pivot unit (mn = self.minor_mult * atr), the same unit the machine
        already uses to confirm a pivot -- so this reuses the existing per-asset
        calibration rather than introducing a new number to tune.

        Only _nl_confirmed / _nh_confirmed are cleared. The watermarks
        (_nl_watermark, _nl_bounce_high, _nh_watermark, _nh_dip_low) are left
        alone, so the machine can confirm a fresh pivot immediately afterwards.
        The old level is released as a LIVE anchor, not erased from history --
        the same treatment the zone ladder already gives a broken level via its
        role_flipped_at / flip_count fields.
        """
        try:
            if self._nl_confirmed is not None and close < (self._nl_confirmed - mn):
                logger.info(
                    "[LSM-RETIRE] %s %s natural_low %.5g released "
                    "(close %.5g is more than one minor unit below it)",
                    self.asset, self.timeframe, self._nl_confirmed, close,
                )
                self._nl_confirmed = None
            if self._nh_confirmed is not None and close > (self._nh_confirmed + mn):
                logger.info(
                    "[LSM-RETIRE] %s %s natural_high %.5g released "
                    "(close %.5g is more than one minor unit above it)",
                    self.asset, self.timeframe, self._nh_confirmed, close,
                )
                self._nh_confirmed = None
        except Exception as e:
            logger.debug("[LSM-RETIRE] %s check error: %s", self.asset, e)

    def update(self, close: float, atr: float) -> LivermoreSnapshot:
        """
        Process one closed bar.  Returns current snapshot after update.
        Safe to call with NaN atr — will hold state without transitioning.
        """
        if math.isnan(close) or math.isnan(atr) or atr <= 0:
            return self.snapshot()

        # Bootstrap anchors on very first bar
        if self._main_up_max is None:
            self._main_up_max   = close
            self._main_down_min = close
            self._nl_watermark  = close
            self._nl_bounce_high = close
            self._nh_watermark  = close
            self._nh_dip_low    = close
            return self.snapshot()

        maj = self.major_mult * atr
        mn  = self.minor_mult * atr

        # STOP-1 SEG I: release pivots the market has left, in every state.
        # Must run BEFORE _dispatch so this bar's handler sees the cleared
        # anchor rather than acting on one price has already invalidated.
        self._retire_stale_pivots(close, mn)

        prev_state = self._state
        self._dispatch(close, atr, maj, mn)

        if self._state != prev_state:
            self._state_age = 0
            self._last_transition = f"{prev_state}→{self._state}"
            # Demoted INFO->DEBUG (2026-06-23): this fires once per bar transition
            # during every full-history replay (generate_features() replays the
            # whole df via update_from_series() each time it's called -- multiple
            # times per asset per cycle, see mean_reversion.py Mode2/3's own
            # `df.tail(260)` re-replay), producing dozens of lines per asset per
            # cycle. Purely internal state bookkeeping -- no decision reads this
            # log line. Set this logger to DEBUG if you need the play-by-play.
            logger.debug(
                "[Livermore] %s %s  %s→%s  close=%.5g  atr=%.5g",
                self.asset, self.timeframe, prev_state, self._state, close, atr,
            )
        else:
            self._state_age += 1

        logger.debug(
            "[LSM] %s %s → %s  close=%.5g  atr=%.5g",
            self.asset, self.timeframe, self._state, close, atr,
        )
        return self.snapshot()

    def update_from_series(self, df: pd.DataFrame) -> LivermoreSnapshot:
        """
        Replay an entire OHLCV dataframe (must contain 'close' and 'atr' columns).
        Useful for warm-start on bot restart with historical bars.
        Returns snapshot after last bar.
        """
        for _, row in df.iterrows():
            self.update(float(row["close"]), float(row["atr"]))
        return self.snapshot()

    # States where each natural anchor is a LIVE level rather than a leftover.
    #
    # N2 FOLLOWUP (2026-08-11): SECONDARY_RETRACEMENT/SECONDARY_REBOUND used to
    # be included here. A logging-only anchor-coherence probe added across the
    # system (composite_state_builder.py, [N2-ANCHOR]) caught this publishing a
    # broken level as if it were live support/resistance on every asset tested
    # (GOLD, BTC, USOIL, EURUSD -- not the 2-of-6 sample an earlier audit had
    # suggested). Root cause, confirmed against real BTC log lines:
    #
    #   lsm1h=SECONDARY_RETRACEMENT price=117320 natural_low=117380 (+0.1%)
    #   lsm1h=SECONDARY_RETRACEMENT price=116340 natural_low=117380 (+0.9%, same value, price fell further)
    #   ...later, after recovering back to MAIN_UP...
    #   lsm1h=MAIN_UP               price=111450 natural_low=117380 (+5.3%, SAME frozen value)
    #
    # By design (module docstring, decision #2), once price probes past the
    # natural low into SECONDARY_RETRACEMENT, _nl_confirmed is deliberately
    # FROZEN at the entry pivot (_nl_entry) so the dual-confirm-to-MAIN_DOWN
    # threshold math doesn't treadmill chasing price down. That freeze is
    # correct for the INTERNAL threshold check. The bug was publishing that
    # same frozen value externally via snapshot() as "the current natural
    # low" -- but SECONDARY means that low already broke; there is no valid
    # level left to report until a fresh one is confirmed. 44% of all N2
    # firings (BTC: 1447/3302) were this direct freeze.
    #
    # SECONDARY_RETRACEMENT/SECONDARY_REBOUND removed from the live-state
    # sets below: snapshot() now returns None for the natural anchor while in
    # a SECONDARY state, matching the "None = no live level, try the next
    # tier" contract every downstream reader (BRC, N1's
    # _resolve_setup_reference in composite_state_builder.py) already
    # implements. See _natural_retr/_secondary_retr (and the _rebound
    # mirrors) below for the paired fix to the recovery transition, which
    # stops the same stale value leaking forward into MAIN_UP/MAIN_DOWN
    # after a SECONDARY episode ends (a further 20% of N2 firings).
    _LOW_LIVE_STATES  = ("MAIN_UP", "NATURAL_RETRACEMENT")
    _HIGH_LIVE_STATES = ("MAIN_DOWN", "NATURAL_REBOUND")

    def snapshot(self) -> LivermoreSnapshot:
        """Return current state as an immutable snapshot."""
        return LivermoreSnapshot(
            state            = self._state,
            state_age        = self._state_age,
            is_silent_zone   = self._state in SILENT_STATES,
            dual_confirmation= self._state in ("MAIN_UP", "MAIN_DOWN"),
            anchor_main_up_max   = self._main_up_max,
            anchor_main_down_min = self._main_down_min,
            anchor_natural_high  = (
                self._nh_confirmed
                if self._state in self._HIGH_LIVE_STATES else None
            ),
            anchor_natural_low   = (
                self._nl_confirmed
                if self._state in self._LOW_LIVE_STATES else None
            ),
        )

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_silent_zone(self) -> bool:
        return self._state in SILENT_STATES

    # ── State dispatch ────────────────────────────────────────────────────────

    def _dispatch(self, close: float, atr: float, maj: float, mn: float) -> None:
        s = self._state
        if   s == "MAIN_UP":           self._main_up(close, maj)
        elif s == "NATURAL_RETRACEMENT":      self._natural_retr(close, maj, mn)
        elif s == "SECONDARY_RETRACEMENT":    self._secondary_retr(close, maj, mn)
        elif s == "MAIN_DOWN":         self._main_down(close, maj)
        elif s == "NATURAL_REBOUND":   self._natural_rebound(close, maj, mn)
        elif s == "SECONDARY_REBOUND": self._secondary_rebound(close, maj, mn)

    # ── MAIN_UP ───────────────────────────────────────────────────────────────

    def _main_up(self, close: float, maj: float) -> None:
        if close > self._main_up_max:
            self._main_up_max = close

        if close < self._main_up_max - maj:
            self._state = "NATURAL_RETRACEMENT"
            self._nl_watermark   = close
            self._nl_confirmed   = None
            self._nl_bounce_high = close
            self._pending_down   = 0

    # ── NATURAL_RETRACEMENT ──────────────────────────────────────────────────────────

    def _natural_retr(self, close: float, maj: float, mn: float) -> None:
        # Update running watermark
        if close < self._nl_watermark:
            self._nl_watermark   = close
            self._nl_bounce_high = close   # reset bounce tracker at each new low

        # Update bounce high
        if close > self._nl_bounce_high:
            self._nl_bounce_high = close

        # Lock natural_low once first minor-bounce is confirmed
        if self._nl_confirmed is None and self._nl_bounce_high > self._nl_watermark + mn:
            self._nl_confirmed = self._nl_watermark
            logger.debug(
                "[Livermore] %s %s  nl_confirmed locked @ %.5g",
                self.asset, self.timeframe, self._nl_confirmed,
            )

        # Transition: new ATH → resume MAIN_UP
        if close > self._main_up_max:
            self._state      = "MAIN_UP"
            self._main_up_max = close
            return

        # Transition: confirmed nl breached → SECONDARY_RETRACEMENT
        if self._nl_confirmed is not None and close < self._nl_confirmed:
            self._state    = "SECONDARY_RETRACEMENT"
            self._nl_entry = self._nl_confirmed   # FIXED pivot for secondary
            self._pending_down = 0

    # ── SECONDARY_RETRACEMENT ────────────────────────────────────────────────────────

    def _secondary_retr(self, close: float, maj: float, mn: float) -> None:
        # nl_entry is FIXED — never updated inside this state.
        nl = self._nl_entry

        # Recovery: back above natural_low + minor → return to NATURAL_RETRACEMENT
        if close > nl + mn:
            self._state          = "NATURAL_RETRACEMENT"
            self._nl_watermark   = nl
            # N2 FOLLOWUP: previously re-armed as `nl` (the just-broken pivot),
            # which left it non-None so the bounce-lock guard in _natural_retr
            # ("if self._nl_confirmed is None and ...") could never re-fire --
            # the same already-invalidated level then kept publishing as live
            # through this NATURAL_RETRACEMENT and into whatever MAIN_UP
            # followed it. Reset to None instead, mirroring the MAIN_UP ->
            # NATURAL_RETRACEMENT reset above (_main_up), so a genuinely fresh
            # low gets locked from here on.
            self._nl_confirmed   = None
            self._nl_bounce_high = close
            self._pending_down   = 0
            return

        # Dual-confirm MAIN_DOWN: close must be major*ATR below nl_entry twice
        if close < nl - maj:
            self._pending_down += 1
            if self._pending_down >= self.dual_confirm:
                self._state         = "MAIN_DOWN"
                self._main_down_min = close
                self._pending_down  = 0
        else:
            # Partial decay: lose one pending count if not in breakout zone
            self._pending_down = max(0, self._pending_down - 1)

    # ── MAIN_DOWN ─────────────────────────────────────────────────────────────

    def _main_down(self, close: float, maj: float) -> None:
        if close < self._main_down_min:
            self._main_down_min = close

        if close > self._main_down_min + maj:
            self._state          = "NATURAL_REBOUND"
            self._nh_watermark   = close
            self._nh_confirmed   = None
            self._nh_dip_low     = close
            self._pending_up     = 0

    # ── NATURAL_REBOUND ───────────────────────────────────────────────────────

    def _natural_rebound(self, close: float, maj: float, mn: float) -> None:
        if close > self._nh_watermark:
            self._nh_watermark = close
            self._nh_dip_low   = close   # reset dip tracker at each new high

        if close < self._nh_dip_low:
            self._nh_dip_low = close

        # Lock natural_high once first minor-dip is confirmed
        if self._nh_confirmed is None and self._nh_dip_low < self._nh_watermark - mn:
            self._nh_confirmed = self._nh_watermark
            logger.debug(
                "[Livermore] %s %s  nh_confirmed locked @ %.5g",
                self.asset, self.timeframe, self._nh_confirmed,
            )

        # Transition: new all-time low → resume MAIN_DOWN
        if close < self._main_down_min:
            self._state         = "MAIN_DOWN"
            self._main_down_min = close
            return

        # Transition: confirmed nh breached → SECONDARY_REBOUND
        if self._nh_confirmed is not None and close > self._nh_confirmed:
            self._state    = "SECONDARY_REBOUND"
            self._nh_entry = self._nh_confirmed   # FIXED pivot for secondary
            self._pending_up = 0

    # ── SECONDARY_REBOUND ─────────────────────────────────────────────────────

    def _secondary_rebound(self, close: float, maj: float, mn: float) -> None:
        nh = self._nh_entry   # FIXED — never updated inside this state

        # Failure: falls back below natural_high - minor → return to NATURAL_REBOUND
        if close < nh - mn:
            self._state          = "NATURAL_REBOUND"
            self._nh_watermark   = nh
            # N2 FOLLOWUP: mirror of the _secondary_retr fix above. Was
            # re-armed as `nh` (the just-broken pivot), which blocked the
            # bounce-lock guard in _natural_rebound from ever re-firing and
            # let the same broken level keep publishing as live through this
            # NATURAL_REBOUND and into whatever MAIN_DOWN followed. Reset to
            # None so a genuinely fresh high locks from here on.
            self._nh_confirmed   = None
            self._nh_dip_low     = close
            self._pending_up     = 0
            return

        # Dual-confirm MAIN_UP: close must be major*ATR above nh_entry twice
        if close > nh + maj:
            self._pending_up += 1
            if self._pending_up >= self.dual_confirm:
                self._state      = "MAIN_UP"
                self._main_up_max = close
                self._pending_up  = 0
        else:
            self._pending_up = max(0, self._pending_up - 1)


# ── Factory helper ────────────────────────────────────────────────────────────

def make_livermore_pair(
    asset: str,
    pivots_config: dict,
) -> tuple["LivermoreStateMachine", "LivermoreStateMachine"]:
    """
    Create a (4H, 1H) pair of state machines for a given asset.

    pivots_config should be the LIVERMORE_PIVOTS[asset] dict from
    aggregator_presets.json.

    Returns (lsm_4h, lsm_1h).
    """
    kw = dict(
        asset       = asset,
        major_mult  = pivots_config.get("major_mult", 3.5),
        minor_mult  = pivots_config.get("minor_mult", 1.0),
        dual_confirm= pivots_config.get("dual_confirm", 2),
        atr_period  = pivots_config.get("atr_period", 14),
    )
    lsm_4h = LivermoreStateMachine(timeframe="4H", **kw)
    lsm_1h = LivermoreStateMachine(timeframe="1H", **kw)
    logger.info("[Livermore] Created 4H+1H pair for %s", asset)
    return lsm_4h, lsm_1h


def atr14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR(period) on a DataFrame with columns: high, low, close.
    Uses EWM (exponential) smoothing — matches most charting platforms.
    """
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
