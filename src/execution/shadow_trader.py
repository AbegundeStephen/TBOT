"""
Shadow Trading Engine (T3.1)

Tracks every signal that was BLOCKED by any gate (gatekeeper, governor, sniper,
trap filter, AI validation, etc.) as a virtual position and records its outcome.

Purpose
-------
Every gate evaluation and ML retraining decision depends on labelled outcomes.
Currently blocked signals vanish — we have no idea if a blocked signal would
have been profitable. This module captures that missing ground truth.

Architecture
------------
Two-tier design for performance:
  Tick tier  (every ~5s): Pure price-vs-stop/target arithmetic. No TA-Lib.
                           ~0.05ms for 20 open positions.
  Candle tier (every 5min): Bar counter increment, MFE/MAE tracking.
                             Full ATR/ADX recalc only if needed.

Key fields per position
-----------------------
  strategy_source   : Which strategy (TF / MR / EMA / consensus) sourced the signal
  peak_profit_bar   : Bar number when Maximum Favourable Excursion occurred
  friction_penalty  : Asset-specific round-trip slippage
  net_pnl_pct       : Gross P&L minus friction (used for ML labels)
  regime_score      : Regime at entry (for regime-intensity analysis)
  gate_blocked_by   : Which gate killed the real signal (for gate scorecard)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from src.utils.episode_ledger import write_episode  # DATA-1 ITEM 6

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Asset-specific round-trip friction penalties (slippage + commission).
# Applied before ML labelling so models learn from net, not gross, P&L.
# ─────────────────────────────────────────────────────────────────────────────
FRICTION_PENALTIES: Dict[str, float] = {
    "BTC":    0.0003,   # 0.03% round-trip
    "BTCUSDT": 0.0003,
    "GOLD":   0.0008,   # 0.08%
    "XAUUSD": 0.0008,
    "XAUUSDm": 0.0008,
    "USTEC":  0.0005,   # 0.05%
    "USTECm": 0.0005,
    "EURJPY": 0.0004,   # 0.04%
    "EURJPYm": 0.0004,
    "EURUSD": 0.0003,
    "EURUSDm": 0.0003,
    "GBPUSD": 0.0003,   # 0.03% — tight spread major pair
    "GBPUSDm": 0.0003,
    "USDJPY": 0.0002,   # 0.02% — tightest spread major
    "USDJPYm": 0.0002,
    "USOIL":  0.0006,   # 0.06% round-trip (oil has wider spreads)
    "USOILm": 0.0006,
    "GBPAUD": 0.0005,   # 0.05% round-trip
    "GBPAUDm": 0.0005,
}
# S7a: lookups are .upper() — normalize keys so mixed-case MT5 variants
# ("XAUUSDm") can actually match instead of silently falling to default.
FRICTION_PENALTIES = {k.upper(): v for k, v in FRICTION_PENALTIES.items()}
_DEFAULT_FRICTION = 0.0005


@dataclass
class ShadowPosition:
    """A single virtual (shadow) trade tracking a blocked signal's outcome."""

    # Identity
    asset: str
    side: str               # "long" | "short"
    strategy_source: str    # "TF" | "MR" | "EMA" | "consensus" | "COUNCIL"
    gate_blocked_by: str    # e.g. "blocked_by_governor", "no_sniper_confirmation"

    # Entry
    entry_price: float
    entry_time: datetime

    # Item 2.17: categories matching the council judge system (previously only
    # the old single-strategy system's labels existed here).
    judge_driver: str = "unknown"    # which judge contributed most to the score
    score_pct_of_max: float = 0.0    # total_score / achievable_max (Item 2.5)
    qualify_tag: str = ""            # plain-English score-margin label (Item 2.6)
    livermore_state_1h: str = ""
    # K1: 4H is the context timeframe under the locked hierarchy. Recording
    # only 1H captured the trigger and discarded the permission, so any model
    # trained on this data cannot distinguish a pullback inside a 4H uptrend
    # from one inside a 4H downtrend.
    livermore_state_4h: str = ""

    # T4: the proof that generated this setup. Without these the shadow record
    # cannot answer the question it exists for — "was the council's refusal
    # right?" — because it does not know WHICH proof was refused. Tier and
    # level quality are the two things most likely to separate a proof that
    # pays from one that does not, and neither was captured.
    brc_confirmed: bool = False
    brc_kind: str = ""              # TF_CONT | MR_REV
    setup_ref: float = 0.0          # the frozen reference level
    setup_ref_tier: str = ""        # ANCHOR_1H | SWING_4H | ZONE_LADDER | RUNNER
    setup_ref_tests: int = 0        # how many times that level was defended
    setup_age_at_entry: int = 0     # bars from birth to entry
    retest_type: str = ""           # CLEAN | WICK | BREAKOUT | CHASE_* | NO_LEVEL
    stop_source: str = "atr"        # "structural" | "atr"
    # BATCH-610 ITEM 4: ATR at entry, retained so entry_distance_atr can be
    # computed at to_dict() time -- mirrors live's [ENTRY-MEASURE] dist field.
    entry_atr: float = 0.0
    # DATA-1 ITEM 1: single id that joins this record to the funnel, trade
    # events, move ledger and episode ledger for the same signal evaluation.
    episode_id: str = ""
    initial_stop_loss: float = 0.0   # S7d: entry-time SL, never mutated — R-units anchor
    friction_source: str = "map"     # S7d: "map" | "default" | "learned"
    gate_code: str = ""              # S7e: machine-stable gate identity
    outcome_class: str = ""          # S7b: "win" | "loss" | "scratch"

    regime_score: float = 0.0
    regime_name: str = "UNKNOWN"

    # Stop & target (rough estimates — we use ATR-based defaults)
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Live tracking
    current_price: float = 0.0
    bars_open: int = 0
    peak_profit_bar: int = 0

    # Extremes
    mfe_pct: float = 0.0    # Maximum Favourable Excursion
    mae_pct: float = 0.0    # Maximum Adverse Excursion

    # Outcome
    closed: bool = False
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    close_reason: str = ""

    # P&L
    gross_pnl_pct: float = 0.0
    friction_pct: float = 0.0
    net_pnl_pct: float = 0.0

    # Strategy vote snapshot at entry (for ML feature construction)
    strategy_votes: Dict = field(default_factory=dict)

    # J2.1: CompositeState snapshot at entry
    composite_state: Dict = field(default_factory=dict)

    # J2.2: VTM-lite trailing stop (standardized — same for every shadow trade)
    trailing_active: bool = False
    trailing_distance: float = 0.0        # Set at open from ATR × 1.5
    trailing_activation_pct: float = 0.0  # Set at open from ATR × 1.0 / entry
    highest_price: float = 0.0            # For longs
    lowest_price: float = 0.0             # For shorts

    # J2.3: Breakeven after TP1
    tp1_reached: bool = False
    tp1_price: float = 0.0               # First partial target: entry ± 1.5 × ATR

    # S7c: R-based breakeven, replaces the TP1-touch trigger above (verified
    # optimum from the 1,341-trade study). be_r set at open from config.
    be_r: float = 0.75
    breakeven_applied: bool = False

    lane: str = "A"                 # LANES L1: which shadow lane produced this
    resumed: bool = False               # MEASURE-2 S1: survived at least one restart
    resume_count: int = 0               # MEASURE-2 S1: how many
    restart_gap_minutes: float = 0.0    # MEASURE-2 S1: blind window -- MFE/MAE unknown across it
    rolls_taken: int = 0            # TARGET-1 T9c: how many times the target rolled

    def _profit_pct(self, price: float) -> float:
        """Current unrealised P&L as a fraction of entry price."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    def tick_update(self, current_price: float) -> bool:
        """
        Tick-tier update. Pure arithmetic — no TA-Lib calls.
        Returns True if the position closed on this tick.
        """
        if self.closed:
            return False

        self.current_price = current_price
        pnl = self._profit_pct(current_price)

        # Track MFE
        if pnl > self.mfe_pct:
            self.mfe_pct = pnl

        # Track MAE
        if pnl < self.mae_pct:
            self.mae_pct = pnl

        # J2.3: TP1 touch tracking retained for MFE/peak bookkeeping — no
        # longer moves the stop (S7c below replaces the BE trigger).
        if not self.tp1_reached and self.tp1_price > 0:
            if (self.side == "long" and current_price >= self.tp1_price) or \
               (self.side == "short" and current_price <= self.tp1_price):
                self.tp1_reached = True

        # S7c: BE at be_r × ENTRY-TIME risk (verified optimum), was TP1-touch.
        if not self.breakeven_applied and self.initial_stop_loss:
            _risk = abs(self.entry_price - self.initial_stop_loss)
            if _risk > 0:
                _prof = (current_price - self.entry_price) if self.side == "long" \
                        else (self.entry_price - current_price)
                if _prof >= self.be_r * _risk:
                    self.stop_loss = self.entry_price      # side-aware by construction
                    self.breakeven_applied = True

        # J2.2: VTM-lite trailing stop
        # Activate trailing after 1.0× ATR favorable move
        if not self.trailing_active and self.trailing_activation_pct > 0:
            if pnl > self.trailing_activation_pct:
                self.trailing_active = True

        if self.trailing_active and self.trailing_distance > 0:
            if self.side == "long":
                self.highest_price = max(self.highest_price, current_price)
                _trail_sl = self.highest_price - self.trailing_distance
                if _trail_sl > self.stop_loss:
                    self.stop_loss = _trail_sl
            else:
                self.lowest_price = min(self.lowest_price, current_price)
                _trail_sl = self.lowest_price + self.trailing_distance
                if _trail_sl < self.stop_loss:
                    self.stop_loss = _trail_sl

        # Check stop loss hit
        if self.stop_loss > 0:
            if self.side == "long" and current_price <= self.stop_loss:
                return self._close(current_price, "stop_loss")
            elif self.side == "short" and current_price >= self.stop_loss:
                return self._close(current_price, "stop_loss")

        # TARGET-1 T9c: roll the target instead of closing at it, up to 2R.
        # The trail is already armed by this point (R-breakeven at 0.75R above,
        # trailing after 1.0x ATR), so a roll gives back open profit, never
        # capital. Stops at 2R: only 38% of 1R trades reach 3R.
        if self.take_profit > 0 and self.initial_stop_loss:
            _risk_t9c = abs(self.entry_price - self.initial_stop_loss)
            if _risk_t9c > 0:
                _reached = (
                    (self.side == "long" and current_price >= self.take_profit)
                    or (self.side == "short" and current_price <= self.take_profit)
                )
                _cur_r_t9c = abs(self.take_profit - self.entry_price) / _risk_t9c
                if _reached and _cur_r_t9c < 2.0:
                    _step = 0.5 * _risk_t9c
                    _new_tp = (self.take_profit + _step) if self.side == "long" \
                              else (self.take_profit - _step)
                    _new_r = abs(_new_tp - self.entry_price) / _risk_t9c
                    if _new_r <= 2.05:
                        self.rolls_taken = getattr(self, "rolls_taken", 0) + 1
                        self.take_profit = _new_tp
                        logger.info(
                            "[SHADOW-ROLL] %s %s: reached %.2fR — target rolled to "
                            "%.5f (%.2fR), roll #%d",
                            self.asset, self.side.upper(), _cur_r_t9c,
                            _new_tp, _new_r, self.rolls_taken,
                        )

        # Check take profit hit
        if self.take_profit > 0:
            if self.side == "long" and current_price >= self.take_profit:
                return self._close(current_price, "take_profit")
            elif self.side == "short" and current_price <= self.take_profit:
                return self._close(current_price, "take_profit")

        return False

    def candle_update(self) -> None:
        """
        Candle-tier update — called every 5 minutes.
        Increments bar counter and records peak_profit_bar.
        """
        if self.closed:
            return
        self.bars_open += 1
        if self._profit_pct(self.current_price) >= self.mfe_pct:
            self.peak_profit_bar = self.bars_open

        # Time-based exit: close after 72 wall-clock hours (3 days) if still open.
        # Wall-clock comparison is used instead of bar count because candle_update()
        # is called every 5-min bot loop — 72 bars would only be 6 hours, not 3 days.
        elapsed_hours = (datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600.0
        if elapsed_hours >= 72.0:
            self._close(self.current_price, "time_stop_72h")

    def _close(self, price: float, reason: str) -> bool:
        """Record the final outcome including friction-adjusted net P&L."""
        self.closed = True
        self.close_price = price
        self.close_time = datetime.now(timezone.utc)
        self.close_reason = reason
        self.gross_pnl_pct = self._profit_pct(price) * 100  # in percent
        self.friction_pct = FRICTION_PENALTIES.get(
            self.asset.upper(), _DEFAULT_FRICTION
        ) * 100
        self.friction_source = "map" if self.asset.upper() in FRICTION_PENALTIES else "default"   # S7d
        self.net_pnl_pct = self.gross_pnl_pct - self.friction_pct
        # S7b: three-way outcome. A trade that protected capital is not a loss.
        # 0.05% gross band = Desire-ratified scratch threshold.
        self.outcome_class = ("scratch" if abs(self.gross_pnl_pct) < 0.05
                              else ("win" if self.net_pnl_pct > 0 else "loss"))
        logger.debug(
            f"[SHADOW] {self.asset} {self.side} closed: "
            f"reason={reason}, gross={self.gross_pnl_pct:.3f}%, "
            f"net={self.net_pnl_pct:.3f}%, bars={self.bars_open}"
        )
        return True

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for DataFrame construction."""
        return {
            "asset":            self.asset,
            "side":             self.side,
            "strategy_source":  self.strategy_source,
            "gate_blocked_by":  self.gate_blocked_by,
            "entry_price":      self.entry_price,
            "stop_loss":        self.stop_loss,
            "take_profit":      self.take_profit,
            "entry_time":       self.entry_time.isoformat() if self.entry_time else None,
            "close_price":      self.close_price,
            "close_time":       self.close_time.isoformat() if self.close_time else None,
            "close_reason":     self.close_reason,
            "regime_score":     self.regime_score,
            "regime_name":      self.regime_name,
            "bars_open":        self.bars_open,
            "peak_profit_bar":  self.peak_profit_bar,
            "mfe_pct":          round(self.mfe_pct * 100, 4),
            "mae_pct":          round(self.mae_pct * 100, 4),
            "gross_pnl_pct":    round(self.gross_pnl_pct, 4),
            "friction_pct":     round(self.friction_pct, 4),
            "net_pnl_pct":      round(self.net_pnl_pct, 4),
            "strategy_votes":   self.strategy_votes,
            "composite_state":  self.composite_state,
            # ── S7d: fields that existed on the position but never reached
            # the archive — proof identity, stop provenance, R-units. ──
            "brc_confirmed":    self.brc_confirmed,
            "brc_kind":         self.brc_kind,
            "stop_source":      self.stop_source,
            "initial_stop_loss": self.initial_stop_loss,
            "friction_source":  self.friction_source,
            "trailing_activated": self.trailing_active,
            "net_pnl_r":        self._net_r(),
            "gate_code":        self.gate_code,          # S7e
            "outcome_class":    self.outcome_class,       # S7b
            # BATCH-610 ITEM 4: mirror the live [ENTRY-MEASURE] fields so
            # blocked trades and taken trades can be compared directly. These
            # were already captured as attributes at open time (setup_ref
            # etc., S7d/T4) but never reached the persisted record.
            "proof_ref":        self.setup_ref,
            "proof_ref_tier":   self.setup_ref_tier,
            "proof_ref_tests":  self.setup_ref_tests,
            "proof_age_bars":   self.setup_age_at_entry,
            "retest_type":      self.retest_type,
            "entry_distance_atr": (
                round(abs(self.entry_price - self.setup_ref) / self.entry_atr, 4)
                if self.setup_ref and self.entry_atr else -1.0
            ),
            "episode_id": self.episode_id,   # DATA-1 ITEM 1
            "entry_atr":  self.entry_atr,    # FRAME-1 SEG 6: captured at open since 610 ITEM 4, never persisted
            "lane":       self.lane,         # LANES L1: A | B-TF | B-MR | C-RANDOM | C-BIASED
            "resumed":             self.resumed,               # MEASURE-2 S1
            "resume_count":        self.resume_count,          # MEASURE-2 S1
            "restart_gap_minutes": self.restart_gap_minutes,   # MEASURE-2 S1
            "rolls_taken": self.rolls_taken,     # TARGET-1 T9c
        }

    def _net_r(self):
        """S7d: net result in R using the FROZEN entry-time stop."""
        try:
            risk = abs(self.entry_price - self.initial_stop_loss)
            if risk <= 0 or not self.close_price:
                return None
            move = (self.close_price - self.entry_price) if self.side == "long" \
                   else (self.entry_price - self.close_price)
            fr = (self.friction_pct / 100.0) * self.entry_price
            return round((move - fr) / risk, 3)
        except Exception:
            return None


class ShadowTradingEngine:
    """
    Manages all open shadow positions and exposes the closed-trade results
    for ML labelling and gate scorecard analysis.

    Usage in main.py
    ----------------
    Initialise once after exchange handlers are ready:
        self.shadow_trader = ShadowTradingEngine()

    Open a shadow position when a signal is blocked:
        self.shadow_trader.open_position(asset, side, entry_price,
            strategy_source, gate_blocked_by, details)

    Call every 5 seconds (tick tier):
        self.shadow_trader.tick_update_all(price_map)

    Call every 5 minutes (candle tier):
        self.shadow_trader.candle_update_all(price_map)
    """

    def __init__(
        self,
        max_positions: int = 500,
        max_closed: int = 10000,
        cooldown_minutes: int = 60,
        archive_dir: str = "logs/shadow",
    ):
        self.open_positions: List[ShadowPosition] = []
        self.closed_results: List[dict] = []
        self._max_positions = max_positions
        self._max_closed    = max_closed
        self._cooldown_minutes = cooldown_minutes
        # last close time per asset (for cooldown gate)
        self._last_close_time: Dict[str, datetime] = {}

        # Phase 2.4: durable archive. Closed shadow trades were previously held
        # only in memory and wiped on every restart, so the gate scorecard never
        # accumulated the 200+ closed trades the calibration logic needs. We now
        # append every closed trade to an append-only daily JSONL and reload the
        # recent history on startup so the scorecard survives restarts.
        self._archive_dir = archive_dir
        try:
            import os as _os
            _os.makedirs(self._archive_dir, exist_ok=True)
        except Exception as _e:
            logger.warning(f"[SHADOW] Could not create archive dir {self._archive_dir}: {_e}")

        logger.info(
            f"[SHADOW] ShadowTradingEngine initialised "
            f"(max_open={max_positions}, max_closed={max_closed}, "
            f"cooldown={cooldown_minutes}min, archive={self._archive_dir})"
        )

    def update_friction_penalty(self, asset: str, observed_slippage_pct: float) -> None:
        """Item 4: let real fill slippage correct the static FRICTION_PENALTIES
        estimate, instead of it being a fixed number nobody ever revisits.

        Call once per real trade close, passing the slippage_pct already
        computed at the fill site (mt5_handler.py / binance_handler.py) —
        that value is in PERCENT units (e.g. 0.03 meaning 0.03%), matching
        the "(slippage_pct:.4f}%)" log line it comes from. FRICTION_PENALTIES
        stores FRACTIONS (e.g. 0.0003 = 0.03%, per its own "# 0.03%
        round-trip" comments), so this converts before writing — feeding the
        percent value in directly would overstate every asset's friction
        penalty by 100x.
        """
        try:
            _observed_fraction = observed_slippage_pct / 100.0
            _history = getattr(self, "_slippage_history", {})
            _history.setdefault(asset.upper(), []).append(_observed_fraction)
            _history[asset.upper()] = _history[asset.upper()][-50:]  # rolling window, last 50 trades
            self._slippage_history = _history
            FRICTION_PENALTIES[asset.upper()] = sum(_history[asset.upper()]) / len(_history[asset.upper()])
            logger.debug(
                f"[SHADOW] Friction penalty for {asset.upper()} updated to "
                f"{FRICTION_PENALTIES[asset.upper()]:.5f} from {len(_history[asset.upper()])} observed fills"
            )
        except Exception as e:
            logger.warning(f"[SHADOW] update_friction_penalty failed for {asset}: {e}")

    def open_position(
        self,
        asset: str,
        side: str,
        entry_price: float,
        strategy_source: str,
        gate_blocked_by: str,
        signal_details: dict,
        atr: Optional[float] = None,
        atr_multiplier: float = 1.8,
        tp_multiples: list = None,
        composite_state: dict = None,   # J2.1 — from CompositeState.to_dict()
        trail_mult: float = 0.8,        # S7c: was hardcoded 1.5
        be_r: float = 0.75,             # S7c: was TP1-touch trigger
        episode_id: str = "",           # DATA-1 ITEM 1B
        lane: str = "A",                # LANES L1: A | B-TF | B-MR | C-RANDOM | C-BIASED
        bypass_guards: bool = False,    # LANES L1: Lane C only -- see L1b
    ) -> Optional[ShadowPosition]:
        """
        Open a new shadow position for a blocked signal.

        Parameters
        ----------
        asset            : Asset name, e.g. "BTC"
        side             : "long" or "short"
        entry_price      : Price at signal time
        strategy_source  : "TF", "MR", "EMA", or "consensus"
        gate_blocked_by  : The reasoning string from signal_details
        signal_details   : Full details dict from get_aggregated_signal
        atr              : Regime-adaptive ATR value (VTM-style ATR7/14/28)
        atr_multiplier   : SL distance = atr × multiplier (from asset risk_config)
        tp_multiples     : TP ATR multiples [tp1, tp2, tp3] — first entry used for TP1
        """
        if len(self.open_positions) >= self._max_positions:
            logger.debug("[SHADOW] Max positions reached, skipping")
            return None

        if entry_price <= 0:
            return None

        asset_key = asset.upper()
        now = datetime.now(timezone.utc)

        # S5.1 — Dedup: skip if a shadow position already open for same asset+side
        # LANES L1: Lane C bypasses this. A random control group must hold several
        # independent samples on one asset at once; deduping it would silently
        # collapse a 24/day cap into ~6/day and bias the sample toward quiet
        # periods -- the exact bias the control exists to remove.
        for _existing in (self.open_positions if not bypass_guards else []):
            if _existing.asset.upper() == asset_key and _existing.side == side:
                logger.debug(
                    f"[SHADOW] Dedup: {asset_key} {side.upper()} already open, skipping"
                )
                return None

        # S5.2 — Cooldown: skip if a shadow closed for this asset within cooldown window
        # LANES L1: bypassed for Lane C, same reasoning as the dedup above.
        _last_close = None if bypass_guards else self._last_close_time.get(asset_key)
        if _last_close is not None:
            from datetime import timedelta as _td
            _elapsed = (now - _last_close).total_seconds() / 60
            if _elapsed < self._cooldown_minutes:
                logger.debug(
                    f"[SHADOW] Cooldown: {asset_key} last closed {_elapsed:.0f}min ago "
                    f"(need {self._cooldown_minutes}min), skipping"
                )
                return None

        # Compute SL/TP using VTM's formula:
        #   SL distance = atr × atr_multiplier  (clamped: min 0.5×atr, max 5.0×atr)
        #   TP1          = entry ± atr × first partial_target multiple
        _stop_loss = 0.0
        _take_profit = 0.0
        # T4: pre-initialised so the ShadowPosition(...) call below (which
        # reads this to set stop_source) never hits a NameError on the path
        # where atr is falsy and the whole block below is skipped — the same
        # shape as N8's _atr1 scope bug earlier this session.
        _t4_use_struct = False
        _t4_stop_source = "atr"  # Segment 4: same pre-init discipline as _t4_use_struct above
        if atr and atr > 0:
            _tp_mults = tp_multiples if tp_multiples else [2.5]
            _first_tp = float(_tp_mults[0]) if _tp_mults else 2.5

            # T4: live runs with structural_stops_enabled=true, so the real
            # trade exits at the level that invalidates the thesis — NOT at a
            # volatility multiple. A shadow trade stopped at 1.8xATR is a
            # different trade from the one the council declined, so its outcome
            # cannot be used to judge that decision.
            #
            # Prefer the setup's own frozen reference when available; fall back
            # to the ATR stop only when there is no structural level, and mark
            # which was used so the archive is honest about it.
            # BATCH-610 ITEM 2: setup_ref lives on composite_state, not on
            # signal_details. This read has silently returned 0.0 for the
            # entire life of the archive, so the structural branch below has
            # never once executed and every shadow stop has been an ATR
            # fallback. Confirmed via a live shadow record carrying
            # "setup_ref": 80.679 nested under its composite_state key.
            _t4_cs_ref = signal_details.get("composite_state") or {}
            _t4_struct_ref = float(
                signal_details.get("setup_ref")
                or (_t4_cs_ref.get("setup_ref") if isinstance(_t4_cs_ref, dict)
                    else getattr(_t4_cs_ref, "setup_ref", 0.0))
                or 0.0
            )
            _t4_use_struct = False

            # ── Segment 4 (17-Aug): mirror VTM's tier-conditional gate, same
            # config keys. ShadowTradingEngine carries no config reference of
            # its own (unlike VTM's self.risk_config), so this reads
            # phase_config off the composite_state dict passed in here — the
            # same field composite_state.py declares and VTM's council/LSM
            # companion already populates. RUNNER tier forces ATR regardless
            # of whether a structural reference exists, matching Segment 2's
            # VTM-side behavior; off (default), _attempt_struct_shadow is
            # always True and every path below is unchanged from today.
            _cs_dict_t4 = composite_state or {}
            _phase_cfg_shadow = _cs_dict_t4.get("phase_config", {}) or {}
            _tier_gate_on = bool(_phase_cfg_shadow.get("tier_conditional_stops_enabled", False))
            _shadow_tier = (_cs_dict_t4.get("brc_tier") or "").upper()
            _attempt_struct_shadow = (not _tier_gate_on) or (bool(_shadow_tier) and _shadow_tier != "RUNNER")

            if _attempt_struct_shadow and _t4_struct_ref > 0 and atr > 0:
                _t4_buf = 0.15 * atr        # same tolerance BRC uses
                if side == "long" and _t4_struct_ref < entry_price:
                    sl_dist = (entry_price - _t4_struct_ref) + _t4_buf
                    _t4_use_struct = True
                elif side == "short" and _t4_struct_ref > entry_price:
                    sl_dist = (_t4_struct_ref - entry_price) + _t4_buf
                    _t4_use_struct = True

            # BATCH-610 ITEM 3: give the structural path the same protection the
            # ATR path already has. It previously bottomed out at 0.15*atr, three
            # times tighter than the ATR path's floor. Live also honours a
            # per-asset min_stop_atr_mult (GBPAUD sets 1.0) which the shadow
            # never read.
            if _t4_use_struct:
                _min_mult = float(
                    (signal_details.get("min_stop_atr_mult") or 0.5)
                )
                _floor = max(0.5, _min_mult) * atr
                if sl_dist < _floor:
                    logger.info(
                        f"[T4-SHADOW-SL] {asset}: structural stop {sl_dist:.5g} "
                        f"below floor {_floor:.5g} -- raised"
                    )
                    sl_dist = _floor
                sl_dist = min(sl_dist, 5.0 * atr)

            if not _t4_use_struct:
                # Match VTM clamp: min 0.5×atr, max 5.0×atr
                sl_dist = max(0.5 * atr, min(5.0 * atr, atr_multiplier * atr))

            # stop_source: "structural"/"atr" preserved byte-identical when the
            # tier gate is off; new "_tier" suffix only appears once it's on,
            # so the archive can be split by anchor once tier-conditional
            # stops are actually enabled without relabeling historical rows.
            if _tier_gate_on:
                _t4_stop_source = "structural_tier" if _t4_use_struct else (
                    "atr_tier" if _shadow_tier == "RUNNER" else "atr_fallback"
                )
            else:
                _t4_stop_source = "structural" if _t4_use_struct else "atr"

            logger.info(
                "[T4-SHADOW-SL] %s: %s stop — dist=%.5g (%s)%s",
                asset, "STRUCTURAL" if _t4_use_struct else "ATR-fallback",
                sl_dist, f"ref={_t4_struct_ref:.5g}" if _t4_use_struct else f"{atr_multiplier}xATR",
                f" [tier={_shadow_tier or 'UNKNOWN'}]" if _tier_gate_on else "",
            )
            tp_dist = _first_tp * atr

            if side == "long":
                _stop_loss   = entry_price - sl_dist
                _take_profit = entry_price + tp_dist
            else:
                _stop_loss   = entry_price + sl_dist
                _take_profit = entry_price - tp_dist

            # TARGET-1 T9c: middle-rung target, mirroring live T8b.
            _tp_mult_t9c = float(_tp_mults[1]) if len(_tp_mults) >= 2 else _first_tp
            _tp_dist_t9c = _tp_mult_t9c * atr
            _take_profit_t9c = (entry_price + _tp_dist_t9c) if side == "long" \
                               else (entry_price - _tp_dist_t9c)
        else:
            # S7h: refuse an unmeasurable shadow rather than open a degenerate one.
            logger.warning("[SHADOW] open refused for %s: atr unavailable — no risk anchor", asset)
            return None

        # J2.2 + J2.3: Compute standardized trailing and TP1 params at entry time
        _trailing_distance = 0.0
        _trailing_activation_pct = 0.0
        _tp1_price = 0.0
        if atr and atr > 0 and entry_price > 0:
            _trailing_distance = atr * trail_mult   # S7c: was 1.5 — now config-driven
            _trailing_activation_pct = atr / entry_price * 1.0  # 1.0× ATR
            _tp1_dist = 1.5 * atr
            if side == "long":
                _tp1_price = entry_price + _tp1_dist
            else:
                _tp1_price = entry_price - _tp1_dist

        # Item 2.17: derive the new judge-system fields from signal_details —
        # judge_scores (Item 1.8), judge_weights, total_score/required_score
        # are all already present on council-sourced signals; default safely
        # for non-council (single-strategy) signals where they're absent.
        _judge_scores = signal_details.get("judge_scores") or {}
        _judge_driver = max(_judge_scores, key=_judge_scores.get) if _judge_scores else "unknown"

        _judge_weights = signal_details.get("judge_weights") or {}
        _achievable_max = sum(_judge_weights.values()) if _judge_weights else 0.0
        _total_score = signal_details.get("total_score", 0.0) or 0.0
        _score_pct_of_max = (_total_score / _achievable_max) if _achievable_max > 0 else 0.0

        _required_score = signal_details.get("required_score", 0.0) or 0.0
        if _required_score > 0:
            _margin = _total_score - _required_score
            _qualify_tag = "CLEARED" if _margin >= 0.5 else "MARGINAL" if _margin >= 0 else "BLOCKED"
        else:
            _qualify_tag = ""

        _lsm_1h = signal_details.get("livermore_state_1h")
        if hasattr(_lsm_1h, "value"):
            _lsm_1h = _lsm_1h.value
        _lsm_1h = _lsm_1h or ""

        _lsm_4h = signal_details.get("livermore_state_4h")
        if hasattr(_lsm_4h, "value"):
            _lsm_4h = _lsm_4h.value
        _lsm_4h = _lsm_4h or ""

        _t4_brc = bool(signal_details.get("brc_confirmed", False))
        _t4_kind = signal_details.get("brc_kind") or ""
        # BATCH-610 ITEM 2: same composite_state-nesting issue as the SL-calc
        # read above -- setup_ref/setup_ref_tier/setup_ref_tests/setup_age are
        # all genuine CompositeState fields (composite_state.py:330-336), so
        # they can be missing flat on signal_details the same way. retest_type
        # is NOT a CompositeState field (set elsewhere, by retest_engine) --
        # left as a plain flat read, unaffected by this bug.
        _t4_cs_sib = signal_details.get("composite_state") or {}
        def _t4_sib(key, cast, default):
            _flat = signal_details.get(key)
            if _flat is not None:
                return cast(_flat)
            _nested = (_t4_cs_sib.get(key) if isinstance(_t4_cs_sib, dict)
                       else getattr(_t4_cs_sib, key, None))
            return cast(_nested) if _nested is not None else default
        _t4_ref = _t4_sib("setup_ref", float, 0.0)
        _t4_tier = _t4_sib("setup_ref_tier", str, "")
        _t4_tests = _t4_sib("setup_ref_tests", int, 0)
        _t4_age = _t4_sib("setup_age", int, 0)
        _t4_retest = signal_details.get("retest_type") or ""

        # S7e: machine-stable gate identity — text before the first "(",
        # e.g. "HOLD (Score: 2.71/4.1)" -> "HOLD"; "NY_OPEN (session)" -> "NY_OPEN"
        _gate_code = (gate_blocked_by or "UNKNOWN").split("(")[0].strip() or "UNKNOWN"

        pos = ShadowPosition(
            asset=asset,
            side=side,
            strategy_source=strategy_source,
            gate_blocked_by=gate_blocked_by,
            gate_code=_gate_code,
            judge_driver=_judge_driver,
            score_pct_of_max=_score_pct_of_max,
            qualify_tag=_qualify_tag,
            livermore_state_1h=_lsm_1h,
            livermore_state_4h=_lsm_4h,
            brc_confirmed=_t4_brc,
            brc_kind=_t4_kind,
            setup_ref=_t4_ref,
            setup_ref_tier=_t4_tier,
            setup_ref_tests=_t4_tests,
            setup_age_at_entry=_t4_age,
            retest_type=_t4_retest,
            stop_source=_t4_stop_source,
            entry_atr=float(atr) if atr else 0.0,   # BATCH-610 ITEM 4
            episode_id=episode_id or "",            # DATA-1 ITEM 1B
            lane=lane,                              # LANES L1
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            regime_score=signal_details.get("regime_score",
                signal_details.get("governor_data", {}).get("regime_score", 0.0)
                if isinstance(signal_details.get("governor_data"), dict) else 0.0
            ),
            regime_name=signal_details.get("regime", "UNKNOWN"),
            stop_loss=_stop_loss,
            initial_stop_loss=_stop_loss,   # S7d: freeze entry-time risk
            # TARGET-1 T9c: hard TP restored, at the MIDDLE rung.
            # SEG 12 retired it because firing at tp_multiples[0] (1.5x ATR)
            # cut two winners at 0.64R and 1.04R. That was right about rung
            # one. 388 records: of 197 trades reaching 1R, 81% reached 2R but
            # only 38% reached 3R -- so the answer is rung two, and it rolls
            # (T9c-2) rather than sitting fixed.
            take_profit=_take_profit_t9c,
            strategy_votes={
                "mr_signal":    signal_details.get("mr_signal", 0),
                "mr_conf":      signal_details.get("mr_confidence", 0.0),
                "tf_signal":    signal_details.get("tf_signal", 0),
                "tf_conf":      signal_details.get("tf_confidence", 0.0),
                "ema_signal":   signal_details.get("ema_signal", 0),
                "ema_conf":     signal_details.get("ema_confidence", 0.0),
                "signal_quality": signal_details.get("signal_quality", 0.0),
                # Part 1.6 (Brain Rebuild): six council judge scorecards —
                # empty dicts (falsy, safely) for non-council signal_details.
                "judge_buy_scores":  signal_details.get("buy_scores", {}),
                "judge_sell_scores": signal_details.get("sell_scores", {}),
            },
            # J2.1: CompositeState snapshot
            composite_state=composite_state or {},
            # J2.2: Standardized trailing stop (same for every shadow trade)
            trailing_active=False,
            trailing_distance=_trailing_distance,
            trailing_activation_pct=_trailing_activation_pct,
            highest_price=entry_price,
            lowest_price=entry_price,
            # J2.3: Breakeven after TP1
            tp1_price=_tp1_price,
            be_r=be_r,   # S7c
        )

        self.open_positions.append(pos)
        logger.info(
            f"[SHADOW] Opened {side.upper()} {asset} @ {entry_price:.5f} "
            f"(src={strategy_source}, gate={gate_blocked_by})"
        )
        return pos

    def tick_update_all(self, price_map: Dict[str, float]) -> int:
        """
        Tick-tier update — call every ~5 seconds.
        price_map: {"BTC": 94250.0, "GOLD": 2850.0, ...}
        Returns number of positions closed this tick.
        """
        closed_count = 0
        still_open = []
        for pos in self.open_positions:
            price = price_map.get(pos.asset)
            if price is None or price <= 0:
                still_open.append(pos)
                continue
            if pos.tick_update(price):
                self._archive(pos)
                closed_count += 1
            else:
                still_open.append(pos)
        self.open_positions = still_open
        return closed_count

    def candle_update_all(self, price_map: Dict[str, float]) -> None:
        """
        Candle-tier update — call every ~5 minutes.
        Increments bar counters and applies time stops.
        """
        still_open = []
        for pos in self.open_positions:
            price = price_map.get(pos.asset)
            if price and price > 0:
                pos.current_price = price
            pos.candle_update()
            if pos.closed:
                self._archive(pos)
            else:
                still_open.append(pos)
        self.open_positions = still_open

    def save_open_positions(self) -> int:
        """MEASURE-2 S1: snapshot every OPEN position so a restart resumes
        instead of discarding.

        Written every cycle, NOT on exit: the bot is killed with taskkill /F
        (TerminateProcess -- no signal, no cleanup), so no shutdown hook can
        ever run. A crash or VPS reboot behaves the same way.

        On 1-Sept six shadows opened and only two closed records exist -- the
        rest were destroyed by four restarts, leaving no trace they existed.

        asdict() is used deliberately: it captures every dataclass field
        including ones added later, so this cannot silently drift as the
        dataclass grows.
        """
        try:
            import os as _os
            from dataclasses import asdict as _asdict
            from src.utils.run_status import write_json_atomic
            _rows = []
            for _p in self.open_positions:
                try:
                    _d = _asdict(_p)
                    for _k, _v in list(_d.items()):
                        if hasattr(_v, "isoformat"):
                            _d[_k] = _v.isoformat()
                    _rows.append(_d)
                except Exception:
                    continue
            write_json_atomic(
                _os.path.join(self._archive_dir, "open_positions.json"),
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(_rows),
                    "positions": _rows,
                },
            )
            return len(_rows)
        except Exception as _e:
            logger.warning(f"[SHADOW] save_open_positions failed: {_e}")
            return 0

    def restore_open_positions(self, price_map: dict = None,
                               max_gap_min: int = 720) -> dict:
        """MEASURE-2 S1: rehydrate open positions after a restart.

        Three outcomes per position:
          resumed   -- back in open_positions, gap recorded
          gapped    -- price moved beyond the stop while down; closed at the
                       stop as stop_loss_gap_inferred, NOT silently continued
          abandoned -- snapshot older than max_gap_min; resuming is meaningless

        MFE/MAE are restored untouched. They are running maxima and must never
        be recomputed -- only advanced by the normal tick loop.
        """
        _out = {"resumed": 0, "gapped": 0, "abandoned": 0, "gap_min": 0.0}
        try:
            import os as _os, json as _json
            _path = _os.path.join(self._archive_dir, "open_positions.json")
            if not _os.path.exists(_path):
                return _out
            _snap = _json.load(open(_path, encoding="utf-8"))
            _saved = datetime.fromisoformat(_snap.get("saved_at"))
            if _saved.tzinfo is None:
                _saved = _saved.replace(tzinfo=timezone.utc)
            _gap = (datetime.now(timezone.utc) - _saved).total_seconds() / 60.0
            _out["gap_min"] = round(_gap, 1)

            for _row in _snap.get("positions", []):
                try:
                    for _k in ("entry_time", "close_time"):
                        if _row.get(_k):
                            _row[_k] = datetime.fromisoformat(_row[_k])
                    _row["resumed"] = True
                    _row["resume_count"] = int(_row.get("resume_count", 0)) + 1
                    _row["restart_gap_minutes"] = round(_gap, 1)
                    _pos = ShadowPosition(**_row)

                    if _gap > max_gap_min:
                        _pos.closed = True
                        _pos.close_reason = "abandoned_restart_gap"
                        _pos.close_price = _pos.current_price
                        _pos.close_time = datetime.now(timezone.utc)
                        self._archive(_pos)
                        _out["abandoned"] += 1
                        continue

                    _now_px = (price_map or {}).get(_pos.asset)
                    _through = False
                    if _now_px and _pos.stop_loss:
                        _through = (
                            (_pos.side == "long" and _now_px <= _pos.stop_loss)
                            or (_pos.side == "short" and _now_px >= _pos.stop_loss)
                        )
                    if _through:
                        # The stop WAS hit. We cannot know when, so record the
                        # stop price and flag the blind window rather than
                        # inventing a clean exit.
                        _pos.closed = True
                        _pos.close_reason = "stop_loss_gap_inferred"
                        _pos.close_price = _pos.stop_loss
                        _pos.close_time = datetime.now(timezone.utc)
                        self._archive(_pos)
                        _out["gapped"] += 1
                        continue

                    if _now_px:
                        _pos.current_price = _now_px
                    self.open_positions.append(_pos)
                    self._last_close_time.pop(_pos.asset.upper(), None)
                    _out["resumed"] += 1
                except Exception as _re:
                    logger.warning(f"[SHADOW] resume failed for one position: {_re}")

            logger.warning(
                f"[SHADOW-RESUME] gap={_out['gap_min']:.0f}min | "
                f"resumed={_out['resumed']} gapped={_out['gapped']} "
                f"abandoned={_out['abandoned']}"
            )
        except Exception as _e:
            logger.warning(f"[SHADOW] restore_open_positions failed: {_e}")
        return _out

    def _archive(self, pos: ShadowPosition) -> None:
        """Move a closed position to results store (in-memory + durable JSONL)."""
        # TARGET-1 T2: idempotency guard. _archive is reachable from four
        # sites -- the two normal close paths and MEASURE-2 S1's
        # gapped/abandoned branches. A position that closed on the same cycle
        # it was restored was written twice (BTC 2-Sept 13:45:18.907440,
        # identical to the microsecond).
        if getattr(pos, "_archived", False):
            logger.debug("[SHADOW] %s already archived — skipping duplicate",
                         getattr(pos, "episode_id", "?"))
            return
        pos._archived = True
        _rec = pos.to_dict()
        self.closed_results.append(_rec)
        # DATA-1 ITEM 6: close the episode into the shared daily ledger,
        # tagged "shadow" so it's distinguishable from live closes when both
        # feed the trail-multiplier learner (Desire's ruling, 27 Aug -- both
        # sources feed it, tagged separately, live weighted). _rec already
        # carries episode_id via to_dict() (Item 1B).
        write_episode({**_rec, "source": "shadow"})
        # Keep results bounded
        if len(self.closed_results) > self._max_closed:
            self.closed_results = self.closed_results[-self._max_closed:]
        # S5.2 — record close time for per-asset cooldown
        self._last_close_time[pos.asset.upper()] = datetime.now(timezone.utc)

        # Phase 2.4: append-only durable record so the gate scorecard survives
        # restarts. One file per UTC day; failures never block trading.
        try:
            import os as _os
            import json as _json
            _day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            _path = _os.path.join(self._archive_dir, f"closed_{_day}.jsonl")
            with open(_path, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(_rec, default=str) + "\n")
        except Exception as _e:
            logger.debug(f"[SHADOW] archive append failed: {_e}")

        # LANES L5c: one flat row per closed trade alongside the full record.
        # ~200 bytes vs ~10KB -- a year of every lane fits in ~2MB and opens
        # instantly in a spreadsheet. The JSONL stays the source of truth;
        # this is the working file.
        # LANES: _rec is already pos.to_dict() (assigned above at line 808) --
        # reused directly rather than recomputing it a second time.
        try:
            import csv as _csv, os as _os2
            _sum_path = _os2.path.join(self._archive_dir, "summary.csv")
            _new = not _os2.path.exists(_sum_path)
            _d = _rec
            _row = {
                "close_time":   _d.get("close_time"),
                "lane":         _d.get("lane"),
                "asset":        _d.get("asset"),
                "side":         _d.get("side"),
                "entry_price":  _d.get("entry_price"),
                "close_price":  _d.get("close_price"),
                "close_reason": _d.get("close_reason"),
                "net_pnl_r":    _d.get("net_pnl_r"),
                "net_pnl_pct":  _d.get("net_pnl_pct"),
                "mfe_pct":      _d.get("mfe_pct"),
                "mae_pct":      _d.get("mae_pct"),
                "bars_open":    _d.get("bars_open"),
                "brc_kind":     _d.get("brc_kind"),
                "stop_source":  _d.get("stop_source"),
                "retest_type":  _d.get("retest_type"),
                "entry_distance_atr": _d.get("entry_distance_atr"),
                "regime_name":  _d.get("regime_name"),
                "trend_angle_deg": (_d.get("composite_state") or {}).get("trend_angle_deg"),
                "gate_blocked_by": str(_d.get("gate_blocked_by"))[:60],
                "episode_id":   _d.get("episode_id"),
            }
            with open(_sum_path, "a", newline="", encoding="utf-8") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(_row.keys()))
                if _new:
                    _w.writeheader()
                _w.writerow(_row)
        except Exception as _csv_err:
            logger.debug(f"[SHADOW] summary.csv append failed: {_csv_err}")

    def load_state(self, lookback_days: int = 30) -> int:
        """
        Phase 2.4: reload recently-closed shadow trades from the durable JSONL
        archive so the gate scorecard persists across restarts. Loads at most
        the last `lookback_days` files, bounded to `_max_closed`. Returns the
        number of records restored. Never raises.
        """
        try:
            import os as _os
            import json as _json
            import glob as _glob

            files = sorted(_glob.glob(_os.path.join(self._archive_dir, "closed_*.jsonl")))
            if lookback_days and len(files) > lookback_days:
                files = files[-lookback_days:]

            restored: List[dict] = []
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                restored.append(_json.loads(line))
                            except Exception:
                                continue
                except Exception:
                    continue

            if not restored:
                logger.info("[SHADOW] No prior closed-trade archive to restore.")
                return 0

            # Bound to capacity (keep most recent)
            if len(restored) > self._max_closed:
                restored = restored[-self._max_closed:]
            self.closed_results = restored + self.closed_results

            # Rebuild per-asset cooldown timestamps from the restored records.
            for r in restored:
                try:
                    _a = str(r.get("asset", "")).upper()
                    _ct = r.get("close_time")
                    if _a and _ct:
                        _dt = datetime.fromisoformat(str(_ct).replace("Z", "+00:00"))
                        prev = self._last_close_time.get(_a)
                        if prev is None or _dt > prev:
                            self._last_close_time[_a] = _dt
                except Exception:
                    continue

            logger.info(
                f"[SHADOW] Restored {len(restored)} closed shadow trades from "
                f"{len(files)} archive file(s) - gate scorecard now persists across restarts."
            )
            return len(restored)
        except Exception as _e:
            logger.warning(f"[SHADOW] load_state failed: {_e}")
            return 0

    def get_gate_scorecard(self) -> Dict[str, dict]:
        """
        Summarise performance by blocking gate — uses net_pnl_pct (after friction).
        Useful for identifying gates that are blocking profitable signals.

        Returns dict keyed by gate code:
            {"count": int, "win_rate": float, "avg_net_pnl": float,
             "scratch_count": int, "win_rate_ex_scratch": float}
        """
        from collections import defaultdict
        buckets: Dict[str, list] = defaultdict(list)
        for r in self.closed_results:
            # S7e: prefer the machine-stable gate_code; legacy records that
            # predate it fall back to parsing gate_blocked_by the same way.
            _code = r.get("gate_code") or (
                r.get("gate_blocked_by", "UNKNOWN").split("(")[0].strip()
            )
            buckets[_code or "UNKNOWN"].append(r)

        scorecard = {}
        for gate, recs in buckets.items():
            pnls = [r["net_pnl_pct"] for r in recs]
            wins = sum(1 for p in pnls if p > 0)   # legacy definition, unchanged
            # S7b: scratch-aware win rate — excludes protected-capital trades
            # from both sides of the ratio instead of letting them dilute it.
            _oc = [r.get("outcome_class") for r in recs]
            scratches = sum(1 for o in _oc if o == "scratch")
            _ex_wins = sum(1 for i, o in enumerate(_oc)
                           if o == "win" or (not o and pnls[i] > 0))
            _ex_losses = sum(1 for i, o in enumerate(_oc)
                             if o == "loss" or (not o and pnls[i] <= 0))
            _ex_denom = _ex_wins + _ex_losses
            scorecard[gate] = {
                "count":       len(pnls),
                "win_rate":    round(wins / len(pnls) * 100, 1) if pnls else 0.0,
                "avg_net_pnl": round(sum(pnls) / len(pnls), 3) if pnls else 0.0,
                "total_pnl":   round(sum(pnls), 3),
                "scratch_count": scratches,
                "win_rate_ex_scratch": round(_ex_wins / _ex_denom * 100, 1) if _ex_denom else 0.0,
            }
        return dict(sorted(scorecard.items(), key=lambda x: x[1]["total_pnl"]))

    def get_strategy_scorecard(self) -> Dict[str, dict]:
        """Summarise performance by strategy source."""
        from collections import defaultdict
        buckets: Dict[str, list] = defaultdict(list)
        for r in self.closed_results:
            buckets[r["strategy_source"]].append(r["net_pnl_pct"])

        scorecard = {}
        for src, pnls in buckets.items():
            wins = sum(1 for p in pnls if p > 0)
            scorecard[src] = {
                "count":       len(pnls),
                "win_rate":    round(wins / len(pnls) * 100, 1) if pnls else 0.0,
                "avg_net_pnl": round(sum(pnls) / len(pnls), 3) if pnls else 0.0,
            }
        return scorecard

    def dump_state(self, path: str) -> None:
        """
        Write a JSON snapshot of the shadow engine's current state to *path*.
        Called periodically by the bot (e.g. every candle) so the dashboard
        process can read it without needing direct in-process access.
        """
        import json
        import os

        open_list = []
        for pos in self.open_positions:
            d = pos.to_dict()
            d["current_price"] = pos.current_price
            d["bars_open"] = pos.bars_open
            d["mfe_pct"] = round(pos.mfe_pct * 100, 4)
            d["mae_pct"] = round(pos.mae_pct * 100, 4)
            # live unrealised P&L
            if pos.entry_price > 0:
                raw = pos._profit_pct(pos.current_price)
                d["live_pnl_pct"] = round(raw * 100, 4)
            else:
                d["live_pnl_pct"] = 0.0
            open_list.append(d)

        state = {
            "open_positions": open_list,
            "closed_results": self.closed_results[-200:],   # last 200 for dashboard
            "gate_scorecard": self.get_gate_scorecard(),
            "strategy_scorecard": self.get_strategy_scorecard(),
            "summary": {
                "open_count": len(self.open_positions),
                "closed_count": len(self.closed_results),
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str)
            try:
                os.replace(tmp, path)
            except OSError:
                # Windows: os.replace raises WinError 5 (Access Denied) when the
                # destination is briefly locked by a reader (dashboard, antivirus).
                # Fall back to remove-then-rename — not atomic but good enough for
                # a read-only dashboard snapshot.
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                os.rename(tmp, path)
        except Exception as exc:
            logger.warning(f"[SHADOW] dump_state failed: {exc}")

    @property
    def summary(self) -> str:
        return (
            f"ShadowTrader: {len(self.open_positions)} open, "
            f"{len(self.closed_results)} closed"
        )
