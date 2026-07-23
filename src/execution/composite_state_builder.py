"""
Composite State Builder
=======================
Builds the CompositeState board that judges, VTM, RetestEngine and the
strategies read from. Owns the Livermore state machines.

Relocated out of PerformanceWeightedAggregator (signal_aggregator.py).
Pure relocation — logic inside every method is identical to what it
replaced. No behaviour change intended.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

from src.indicators.divergence import RSIDivergenceDetector
from src.execution.transition_detector import TransitionDetector

logger = logging.getLogger(__name__)


class CompositeStateBuilder:
    """Owns the Livermore state machines and builds CompositeState."""

    def __init__(self, asset_type: str, mean_reversion_strategy=None, config: Dict = None):
        self.asset_type = asset_type.upper()
        self.s_mean_reversion = mean_reversion_strategy
        self._config = config or {}

        # ── Regime tracking ──
        self._previous_regime = {}
        self._regime_start_time = {}
        self._regime_durations = {}
        self._transition_counts = {}

        # ── Structure memory ──
        self._structure_levels = {}

        # ── Zone ladder store ──
        # Deliberately SEPARATE from _structure_levels. That store is built for
        # near-price stop levels and has three limits that make it unusable here:
        #   1. It hard-deletes any level >3 ATR from price — which is exactly
        #      what the zone ladder's outer levels are.
        #   2. Its "age_hours" increments once per CYCLE, not per hour, so its
        #      "90 day" window is really ~2160 cycles (a couple of days).
        #   3. It `break`s after the first swing per side — it never builds
        #      a history, only tracks the two most recent swings.
        # This store fixes all three. _structure_levels stays untouched.
        self._zone_levels = {}

        # ── Session levels ──
        self._pdh = {}
        self._pdl = {}
        self._pdh_date = None
        self._asian_high = {}
        self._asian_low = {}

        # ── Misc trackers ──
        self._last_loss_time = {}
        self._squeeze_was_active = {}
        # TRAJECTORY (Plan 1B): per-asset live-setup memory. None = no setup.
        # Shape when active: {"kind","dir","age","born_compression","last_compression"}
        self._active_setup = {}
        # Last cycle's compression dial per asset — used to classify the
        # setup's energy trend (building / holding / fading).
        self._prev_compression = {}
        self._spread_history = {}
        # NOTE: _lifecycle_age_cfg is deliberately NOT pre-set here. The
        # original class never initializes it either — _update_trend_lifecycle
        # relies on hasattr(self, "_lifecycle_age_cfg") being False on the
        # true first call to trigger its real load from
        # aggregator_presets.json. Pre-setting it to None here would make
        # hasattr return True immediately, permanently skipping that load and
        # silently falling back to hardcoded (4, 12) thresholds instead.

        # phase_config: also deliberately NOT set here. Set externally by
        # main.py/backtest.py after construction (e.g. `perf_agg.phase_config
        # = ...`) via the property forwarder on PerformanceWeightedAggregator
        # — see signal_aggregator.py's phase_config property (Step 4).

        # ── Engines ──
        self._transition_detector = TransitionDetector()
        self.divergence_detector = RSIDivergenceDetector(pivot_window=5)
        self.dynamic_thresholds = None  # injected — see Step 3

        # ── PHASE 1: Livermore State Machine ────────────────────────────────
        # Two instances per asset: 4H (trend structure) + 1H (entry timing).
        # Both are cold-started here; warm_start_livermore() must be called
        # once historical bars are available (see main.py startup sequence).
        self._livermore_4h = None
        self._livermore_1h = None
        self._livermore_warmed = False
        self._livermore_last_4h_ts = None  # deduplicate 4H bar updates
        try:
            import json as _json

            _presets_path = "config/aggregator_presets.json"
            with open(_presets_path) as _f:
                _presets = _json.load(_f)
            _lp = _presets.get("LIVERMORE_PIVOTS", {})
            # Map common asset aliases to preset keys
            # Include MT5 broker-suffix variants (e.g. XAUUSDm, EURUSDm)
            _alias_map = {
                # BTC
                "BTCUSDT": "BTC",
                "BTCUSDM": "BTC",
                # Gold
                "XAUUSD": "GOLD",
                "XAUUSDM": "GOLD",
                # FX
                "EURUSD": "EURUSD",
                "EURUSDM": "EURUSD",
                "EURJPY": "EURJPY",
                "EURJPYM": "EURJPY",
                "GBPAUD": "GBPAUD",
                "GBPAUDM": "GBPAUD",
                "GBPUSD": "GBPUSD",
                "GBPUSDM": "GBPUSD",
                "USDJPY": "USDJPY",
                "USDJPYM": "USDJPY",
                # Indices / Commodities
                "USTEC": "USTEC",
                "USTECM": "USTEC",
                "USOIL": "USOIL",
                "USOILM": "USOIL",
            }
            _lp_key = _alias_map.get(self.asset_type.upper(), self.asset_type)
            if _lp_key not in _lp:
                logger.warning(
                    "[LSM CALIBRATION] %s has no entry in LIVERMORE_PIVOTS — "
                    "falling back to BTC's calibration, which is tuned much "
                    "tighter than most other assets need.",
                    _lp_key,
                )
            _lp_cfg = _lp.get(_lp_key, _lp.get("BTC", {}))
            from src.execution.livermore_state_machine import make_livermore_pair

            self._livermore_4h, self._livermore_1h = make_livermore_pair(
                asset=self.asset_type, pivots_config=_lp_cfg
            )
            logger.info(
                "[Livermore] %s  major=%.1f  minor=%.1f  dual=%d",
                self.asset_type,
                _lp_cfg.get("major_mult", 3.5),
                _lp_cfg.get("minor_mult", 1.0),
                _lp_cfg.get("dual_confirm", 2),
            )
        except Exception as _lsm_err:
            logger.error(
                "[Livermore] Init failed — state machine disabled: %s", _lsm_err
            )

    # ── PHASE 1: Livermore warm-start ────────────────────────────────────────

    def warm_start_livermore(
        self, df_4h: "pd.DataFrame", df_1h: "pd.DataFrame"
    ) -> None:
        """
        Replay historical bars through the Livermore state machines so they
        arrive at the correct state before the live loop begins.

        Call once from main.py after the DataManager has fetched historical data,
        before the first get_aggregated_signal() call.

        df_4h / df_1h must have columns: open, high, low, close, volume.
        ATR is computed internally.
        """
        if self._livermore_4h is None:
            return
        if self._livermore_warmed:
            return

        from src.execution.livermore_state_machine import atr14 as _atr14

        try:
            if df_4h is not None and len(df_4h) >= 20:
                _df4 = df_4h.copy()
                _df4["atr"] = _atr14(_df4)
                self._livermore_4h.update_from_series(_df4)
                self._livermore_last_4h_ts = (
                    df_4h.index[-1] if not df_4h.empty else None
                )
                snap4 = self._livermore_4h.snapshot()
                logger.info(
                    "[Livermore] %s 4H warm-start complete | state=%s age=%d bars",
                    self.asset_type,
                    snap4.state,
                    snap4.state_age,
                )

            if df_1h is not None and len(df_1h) >= 20:
                _df1 = df_1h.copy()
                _df1["atr"] = _atr14(_df1)
                self._livermore_1h.update_from_series(_df1)
                snap1 = self._livermore_1h.snapshot()
                logger.info(
                    "[Livermore] %s 1H warm-start complete | state=%s age=%d bars",
                    self.asset_type,
                    snap1.state,
                    snap1.state_age,
                )

            self._livermore_warmed = True

        except Exception as _ws_err:
            logger.error("[Livermore] warm_start failed: %s", _ws_err)

    # ══════════════════════════════════════════════════════════════════════
    # CONTEXT ENGINE — Build composite state (called once per candle close)
    # ══════════════════════════════════════════════════════════════════════

    def _build_composite_state(self, df, df_4h, governor_data: dict):
        """Build a fresh CompositeState from closed-candle data."""
        from src.execution.composite_state import CompositeState
        import talib as ta
        from datetime import datetime

        state = CompositeState()
        # Fix 1: propagate phase_config gate flags into CompositeState
        state.phase_config = getattr(self, "phase_config", {})

        if df is None or len(df) < 20:
            return state

        # ── D.3: Session Context ──────────────────────────────────────────
        # Use bar timestamp when available (critical in backtest where wall-clock
        # is always "now" regardless of the bar's actual datetime).
        try:
            _bar_ts = df.index[-1] if not df.empty else None
            if _bar_ts is not None and hasattr(_bar_ts, "hour"):
                _hour = _bar_ts.hour
                _dow = _bar_ts.weekday()
            else:
                _hour = datetime.utcnow().hour
                _dow = datetime.utcnow().weekday()
            if 0 <= _hour < 8:
                state.session_name = "ASIAN"
            elif 8 <= _hour < 12:
                state.session_name = "LONDON"
            elif 12 <= _hour < 17:
                state.session_name = "OVERLAP"
            elif 17 <= _hour < 21:
                state.session_name = "NY_CLOSE"
            else:
                state.session_name = "OFF_HOURS"
        except Exception:
            pass

        # ── D.2: MTF Slope Agreement ──────────────────────────────────────
        try:
            _ema50_1h = df["close"].ewm(span=50, adjust=False).mean()
            _slope_1h = (_ema50_1h.iloc[-1] - _ema50_1h.iloc[-6]) / max(
                _ema50_1h.iloc[-6], 1
            )

            if df_4h is not None and len(df_4h) >= 10:
                _ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
                _slope_4h = (_ema50_4h.iloc[-1] - _ema50_4h.iloc[-6]) / max(
                    _ema50_4h.iloc[-6], 1
                )

                state.slopes_aligned = (_slope_1h > 0 and _slope_4h > 0) or (
                    _slope_1h < 0 and _slope_4h < 0
                )
                state.slope_diverging = not state.slopes_aligned

            # Structural decay = old regime + slopes fighting
            if state.regime_age_ratio > 1.5 and state.slope_diverging:
                state.structural_decay = True
        except Exception:
            pass

        # ── Shared ATR (used by multiple sub-modules) ─────────────────────
        _atr = 0.0
        try:
            _atr_arr = ta.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            _atr = float(_atr_arr[-1]) if not np.isnan(_atr_arr[-1]) else 0.0
        except Exception:
            pass

        # ── E.1: ChoCh / BOS Detection ────────────────────────────────────
        self._update_structure(state, df)

        # ── E.2: MTF Structure Memory ─────────────────────────────────────
        self._update_structure_memory(state, df, df_4h)

        # ── E.3: MA Defense Validator ─────────────────────────────────────
        self._update_ma_defense(state, df, span=50, suffix="")
        self._update_ma_defense(state, df, span=200, suffix="")

        _df_1d = governor_data.get("df_1d") if governor_data else None
        if _df_1d is not None and len(_df_1d) >= 90:
            self._update_ma_defense(state, _df_1d, span=50, suffix="_1d")
            self._update_ma_defense(state, _df_1d, span=200, suffix="_1d")

        # ── Zone ladder ──
        # _atr_now reuses the _atr local variable already computed above
        # ("Shared ATR — used by multiple sub-modules") instead of reading
        # state.atr, which doesn't exist on CompositeState — a getattr with
        # a 0 default there would silently zero out the tolerance/hysteresis
        # math below and risk a ZeroDivisionError in the test-hysteresis
        # distance check.
        _atr_now = _atr
        _price_now = float(df["close"].iloc[-1])

        if df_4h is not None:
            self._update_zone_levels(self.asset_type, df_4h, "4H", _atr_now, _price_now)
            _v4 = self._build_zone_view(df_4h, self.asset_type, "4H", _price_now)
            state.zone_4h_current_upper = _v4["current_upper"]
            state.zone_4h_current_lower = _v4["current_lower"]
            state.zone_4h_current_upper_tests = _v4["current_upper_tests"]
            state.zone_4h_current_lower_tests = _v4["current_lower_tests"]
            state.zone_4h_current_upper_type = _v4["current_upper_type"]
            state.zone_4h_current_lower_type = _v4["current_lower_type"]
            state.zone_4h_outer_high = _v4["outer_high"]
            state.zone_4h_outer_low = _v4["outer_low"]

        if _df_1d is not None:
            self._update_zone_levels(self.asset_type, _df_1d, "1D", _atr_now, _price_now)
            _v1 = self._build_zone_view(_df_1d, self.asset_type, "1D", _price_now)
            state.zone_1d_current_upper = _v1["current_upper"]
            state.zone_1d_current_lower = _v1["current_lower"]
            state.zone_1d_current_upper_tests = _v1["current_upper_tests"]
            state.zone_1d_current_lower_tests = _v1["current_lower_tests"]
            state.zone_1d_current_upper_type = _v1["current_upper_type"]
            state.zone_1d_current_lower_type = _v1["current_lower_type"]
            state.zone_1d_outer_high = _v1["outer_high"]
            state.zone_1d_outer_low = _v1["outer_low"]

        # ── E.4: Parabolic Space (Dynamic Z-Score) ────────────────────────
        try:
            _ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
            _price = df["close"].iloc[-1]
            _distance = abs(_price - _ema50) / max(_atr, 0.0001)

            _extreme, _z, _thresh = self.dynamic_thresholds.check(
                self.asset_type,
                "ema50_distance",
                _distance,
                z_threshold=2.5,
                fallback=3.5,
            )
            state.is_parabolic = _extreme
            state.distance_zscore = _z
        except Exception:
            pass

        # ── E.5: EMA Squeeze (ATR-Normalized) ────────────────────────────
        try:
            if _atr > 0:
                _ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
                _ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                _ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
                _spread = (
                    max(_ema20, _ema50, _ema200) - min(_ema20, _ema50, _ema200)
                ) / max(_atr, 0.0001)

                state.squeeze_active = _spread < 0.5
                state.squeeze_strength = max(0.0, 1.0 - _spread)

                _prev_squeeze = self._squeeze_was_active.get(self.asset_type, False)
                if _prev_squeeze and not state.squeeze_active and _spread > 1.0:
                    state.coiled_spring = True
                self._squeeze_was_active[self.asset_type] = state.squeeze_active
        except Exception:
            pass

        # ── E.6: Inside/Outside Bar + Failed Breakout ─────────────────────
        try:
            if len(df) >= 4:
                _prev_h = df["high"].iloc[-2]
                _prev_l = df["low"].iloc[-2]
                _curr_h = df["high"].iloc[-1]
                _curr_l = df["low"].iloc[-1]
                _curr_c = df["close"].iloc[-1]

                state.inside_bar = _curr_h <= _prev_h and _curr_l >= _prev_l
                state.outside_bar = _curr_h > _prev_h and _curr_l < _prev_l

                if state.squeeze_active and state.inside_bar:
                    state.coiled_spring = True

                _recent_high = df["high"].iloc[-4:-1].max()
                if _curr_h > _recent_high and _curr_c < _recent_high:
                    state.failed_breakout = True
                    logger.info(f"[SIGNAL] {self.asset_type}: failed_breakout=True")
        except Exception:
            pass

        # ── F.1: Effort vs Result (All Assets) ────────────────────────────
        try:
            _tick_vol = df["volume"].iloc[-1]
            _body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
            _er = _tick_vol / max(_body, 0.0001)

            _extreme, _z, _ = self.dynamic_thresholds.check(
                self.asset_type, "effort_result", _er, z_threshold=2.0, fallback=None
            )
            state.effort_result_zscore = _z
            if (
                _extreme
                and _z > 2.0
                and abs(governor_data.get("regime_score", 0)) >= 0.5
            ):
                state.absorption_detected = True
        except Exception:
            pass

        # ── F.2: Candle Body Ratio Trend ─────────────────────────────────
        try:
            _bodies = abs(df["close"] - df["open"]).tail(10).values
            if len(_bodies) >= 8:
                _recent = _bodies[-3:].mean()
                _older = _bodies[:5].mean()
                state.body_trend_ratio = _recent / max(_older, 0.0001)
                # Only trust the ratio when older-window bodies aren't near-zero —
                # dead chop makes the ratio pure noise.
                state.body_trend_ratio_valid = _older > 0.0001
                state.conviction_dying = (
                    state.body_trend_ratio_valid and state.body_trend_ratio < 0.5
                )
                if state.conviction_dying:
                    logger.info(
                        f"[SIGNAL] {self.asset_type}: conviction_dying=True "
                        f"(body_trend_ratio={state.body_trend_ratio:.2f})"
                    )
        except Exception:
            pass

        # ── F.5: BTC VPD (Volume-Price Divergence) ────────────────────────
        if self.asset_type == "BTC" and "volume" in df.columns:
            try:
                _vol = df["volume"].iloc[-1]
                _vol_sma = df["volume"].tail(20).mean()
                _regime_score = governor_data.get("regime_score", 0)

                if abs(_regime_score) >= 1.0 and _vol < _vol_sma * 0.80:
                    state.vpd_diverging = True
            except Exception:
                pass

        # ── G.1: Unified Liquidity Sweeps ─────────────────────────────────
        self._update_sweeps(state, df)

        # ── G.2: Rejection Profiling ──────────────────────────────────────
        try:
            if _atr > 0:
                _o = df["open"].iloc[-1]
                _h = df["high"].iloc[-1]
                _l = df["low"].iloc[-1]
                _c = df["close"].iloc[-1]
                _total = _h - _l
                if _total > 0:
                    _upper_wick = _h - max(_o, _c)
                    _lower_wick = min(_o, _c) - _l
                    _wick_ratio = max(_upper_wick, _lower_wick) / _total

                    if _wick_ratio > 0.75 and state.nearby_4h_level is not None:
                        _dist_to_level = abs(_c - state.nearby_4h_level) / max(
                            _atr, 0.001
                        )
                        if _dist_to_level < 0.5:
                            state.rejection_at_level = True
                            state.rejection_strength = _wick_ratio
                            state.level_defended = True

                    # A4: same defended-wick check extended to the 2nd/3rd
                    # nearest levels — independent of the primary level above,
                    # since a strong reversal wick can be rejecting off a
                    # different level than whichever is currently "nearest".
                    if _wick_ratio > 0.75:
                        if state.nearby_4h_level_2 is not None:
                            _dist_2 = abs(_c - state.nearby_4h_level_2) / max(_atr, 0.001)
                            if _dist_2 < 0.5:
                                state.level_2_defended = True
                        if state.nearby_4h_level_3 is not None:
                            _dist_3 = abs(_c - state.nearby_4h_level_3) / max(_atr, 0.001)
                            if _dist_3 < 0.5:
                                state.level_3_defended = True
        except Exception:
            pass

        # ── G.3: Session VWAP ─────────────────────────────────────────────
        try:
            if "volume" in df.columns and _atr > 0:
                _midnight_mask = df.index.hour == 0
                if _midnight_mask.any():
                    _session_start = df[_midnight_mask].index[-1]
                else:
                    _session_start = df.index[0]
                _session = df[df.index >= _session_start]
                if len(_session) > 1:
                    _vwap = (
                        _session["close"] * _session["volume"]
                    ).cumsum() / _session["volume"].cumsum()
                    state.vwap_price = float(_vwap.iloc[-1])
                    state.distance_to_vwap_atr = abs(
                        df["close"].iloc[-1] - state.vwap_price
                    ) / max(_atr, 0.001)
        except Exception:
            pass

        # ── G.5: Time since last loss ─────────────────────────────────────
        _last_loss = self._last_loss_time.get(self.asset_type)
        if _last_loss:
            from datetime import datetime as _dt2

            state.time_since_last_loss_hours = (
                _dt2.now() - _last_loss
            ).total_seconds() / 3600

        # ── F.4: BTC CVD from WebSocket (injected via governor_data) ─────
        if self.asset_type in ("BTC", "BTCUSDT") and governor_data:
            _raw_cvd = governor_data.get("cvd_trend", 0)
            # Robust int conversion — backtest supplies "FLAT" string, live supplies int
            if isinstance(_raw_cvd, str):
                _cvd_map = {
                    "UP": 1,
                    "BULL": 1,
                    "BULLISH": 1,
                    "DOWN": -1,
                    "BEAR": -1,
                    "BEARISH": -1,
                    "FLAT": 0,
                    "NEUTRAL": 0,
                    "": 0,
                }
                state.cvd_trend = _cvd_map.get(_raw_cvd.upper(), 0)
            else:
                state.cvd_trend = int(_raw_cvd)
            state.cvd_stale = bool(governor_data.get("cvd_stale", True))
            # ── F.6: L2 Order Book Imbalance ─────────────────────────────
            state.order_book_imbalance = float(
                governor_data.get("order_book_imbalance", 0.0)
            )
            state.order_book_wall_detected = bool(
                governor_data.get("order_book_wall_detected", False)
            )

        # ── TRANSITION EVIDENCE (SLIGHTLY + full trend regimes) ──────────────
        # Must run AFTER CVD/order-book fields are populated above so the
        # order_flow sub-score has live data. df_4h is already available as
        # the second parameter of _build_composite_state.
        state._transition_evidence = None
        _regime_name = (
            governor_data.get("consensus_regime", "UNKNOWN")
            if governor_data
            else "UNKNOWN"
        )
        # MRS §6 Phase 0: TRANSITION path removed. Transition evidence is now
        # only collected for directional (BEARISH/BULLISH) regimes where the
        # gatekeeper softener uses it to modulate counter-trend penalty scaling.
        # NEUTRAL regime no longer needs evidence — the Livermore Hard Veto
        # (is_silent_zone) handles structural gating for those assets.
        if _regime_name in (
            "BEARISH",
            "SLIGHTLY_BEARISH",
            "BULLISH",
            "SLIGHTLY_BULLISH",
        ):
            try:
                _depth = governor_data.get("depth_data") if governor_data else None
                state._transition_evidence = self._transition_detector.collect_evidence(
                    asset=self.asset_type,
                    regime=_regime_name,
                    df_4h=df_4h if df_4h is not None else pd.DataFrame(),
                    df_1h=df,
                    composite_state=state,
                    cvd_trend=state.cvd_trend,
                    order_book_imbalance=state.order_book_imbalance,
                    depth_data=_depth,
                )
            except Exception as _te_err:
                logger.debug(f"[TRANSITION] Evidence collection error: {_te_err}")
        # ─────────────────────────────────────────────────────────────────

        # ── F.7: MT5 Spread Velocity (synthetic L2 proxy for non-BTC) ────
        if self.asset_type not in ("BTC", "BTCUSDT") and governor_data:
            try:
                _current_spread = governor_data.get("current_spread", 0)
                if _current_spread and _current_spread > 0:
                    if self.asset_type not in self._spread_history:
                        self._spread_history[self.asset_type] = []
                    self._spread_history[self.asset_type].append(_current_spread)
                    if len(self._spread_history[self.asset_type]) > 20:
                        self._spread_history[self.asset_type] = self._spread_history[
                            self.asset_type
                        ][-20:]
                    _spreads = self._spread_history[self.asset_type]
                    if len(_spreads) >= 10:
                        import numpy as _np

                        _avg = _np.mean(_spreads)
                        state.spread_ratio = float(_current_spread) / max(_avg, 0.0001)
                        state.spread_velocity_spike = state.spread_ratio > 2.5
            except Exception:
                pass

        # ── PHASE 1: Livermore State Machine update ──────────────────────────
        # Update both timeframe instances with the latest closed bar.
        # 4H: only update when df_4h has a bar newer than the last processed one.
        # 1H: update every call (df is the 1H feed).
        # Writes to CompositeState livermore_* fields defined in Phase 1 reserved block.
        if self._livermore_4h is not None:
            try:
                from src.execution.livermore_state_machine import atr14 as _atr14_lsm

                # ── 4H update ────────────────────────────────────────────────
                if df_4h is not None and len(df_4h) >= 15:
                    _4h_ts = df_4h.index[-1]
                    if _4h_ts != self._livermore_last_4h_ts:
                        # New 4H candle — compute ATR and update
                        _atr4_series = _atr14_lsm(df_4h)
                        _atr4 = (
                            float(_atr4_series.iloc[-1])
                            if not np.isnan(_atr4_series.iloc[-1])
                            else 0.0
                        )
                        _close4 = float(df_4h["close"].iloc[-1])
                        snap4 = self._livermore_4h.update(_close4, _atr4)
                        self._livermore_last_4h_ts = _4h_ts
                    else:
                        snap4 = self._livermore_4h.snapshot()

                    state.livermore_state_4h = snap4.state
                    state.livermore_state_age_4h = snap4.state_age
                    state.livermore_anchor_main_up_max = snap4.anchor_main_up_max
                    state.livermore_anchor_main_down_min = snap4.anchor_main_down_min
                    state.livermore_anchor_natural_high = snap4.anchor_natural_high
                    state.livermore_anchor_natural_low = snap4.anchor_natural_low
                    state.livermore_dual_confirmation = snap4.dual_confirmation
                    # is_silent_zone = True when in NATURAL_RETR or NATURAL_REBOUND
                    # (counter-trend signals suppressed by Hard Veto Layer — Phase 2)
                    state.is_silent_zone = snap4.is_silent_zone

                # ── 1H update ────────────────────────────────────────────────
                if df is not None and len(df) >= 15:
                    _atr1_series = _atr14_lsm(df)
                    _atr1 = (
                        float(_atr1_series.iloc[-1])
                        if not np.isnan(_atr1_series.iloc[-1])
                        else 0.0
                    )
                    _close1 = float(df["close"].iloc[-1])
                    snap1 = self._livermore_1h.update(_close1, _atr1)
                    state.livermore_state_1h = snap1.state
                    state.livermore_state_age_1h = snap1.state_age

            except Exception as _lsm_err:
                logger.debug(
                    "[Livermore] _build_composite_state update error: %s", _lsm_err
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── Item 2.7: institutional pattern from a live data source ──────────
        # Runs every cycle so the Pattern judge has a real classification to
        # read most of the time, instead of relying solely on _score_confluence's
        # much stricter 5-way classifier (which only runs when a candidate
        # signal already exists, and otherwise leaves institutional_pattern
        # unset/None — silently falling back to the old candlestick-pattern
        # AI validator path in the judge).
        self._compute_institutional_pattern(df, state)

        # Item 6.1: price-OBV divergence, computed once here so the Volume
        # judge (council_aggregator.py, Item 6.2) can read it directly instead
        # of each consumer recomputing OBV independently.
        self._compute_volume_divergence(df, state)

        # ── PHASE 2: vol_down_ratio ──────────────────────────────────────────
        # Volume on down-close bars vs up-close bars over the last 20 1H bars.
        # Used as MR Mode 1 veto (Phase 3A) and Scenario B continuation veto.
        # > 1.2 means down-volume dominates (distribution, not re-accumulation).
        # Wire now — actual veto applied in Phase 3A.
        # NOT a directional predictor. Only valid as a blocking filter.
        try:
            if df is not None and len(df) >= 10:
                _vdr_n = min(20, len(df))
                _vdr_close = df["close"].values[-_vdr_n:]
                _vdr_open = df["open"].values[-_vdr_n:]
                _vdr_vol = df["volume"].values[-_vdr_n:].astype(float)
                _down_vol = float(np.sum(_vdr_vol[_vdr_close < _vdr_open]))
                _up_vol = float(np.sum(_vdr_vol[_vdr_close > _vdr_open]))
                if _up_vol > 0:
                    state.vol_down_ratio = _down_vol / _up_vol
                    state.vol_down_ratio_valid = True
                else:
                    state.vol_down_ratio = None
                    state.vol_down_ratio_valid = False
        except Exception as _vdr_err:
            logger.debug("[vol_down_ratio] compute error: %s", _vdr_err)
        # ─────────────────────────────────────────────────────────────────────

        # ── PHASE 3A: BB/KC Squeeze detection ───────────────────────────────────
        # BB(20,2sigma) inside Keltner(20 EMA, 1.5xATR14) for 5+ bars = volatility
        # compression. bbw_percentile: how compressed vs 6-month rolling history.
        # Fields consumed by MR Mode 1 bb_contraction optional condition and
        # Range Classification Scenario A/B.
        try:
            if df is not None and len(df) >= 25:
                _bkc_bb_period = 20
                _bkc_kc_period = 20
                _bkc_kc_mult = 1.5
                _bkc_min_bars = 5

                _bkc_close = df["close"].values
                _bkc_high = df["high"].values
                _bkc_low = df["low"].values

                # Bollinger Bands (2-sigma)
                _bb_u, _bb_m, _bb_l = ta.BBANDS(
                    _bkc_close, timeperiod=_bkc_bb_period, nbdevup=2.0, nbdevdn=2.0
                )
                # Keltner Channel: EMA(20) +/- 1.5xATR14
                _kc_mid = (
                    df["close"].ewm(span=_bkc_kc_period, adjust=False).mean().values
                )
                _kc_atr14 = ta.ATR(_bkc_high, _bkc_low, _bkc_close, timeperiod=14)
                _kc_upper = _kc_mid + _bkc_kc_mult * _kc_atr14
                _kc_lower = _kc_mid - _bkc_kc_mult * _kc_atr14

                # Squeeze when BB is entirely inside KC
                _valid_mask = (
                    ~np.isnan(_bb_u)
                    & ~np.isnan(_bb_l)
                    & ~np.isnan(_kc_upper)
                    & ~np.isnan(_kc_lower)
                )
                _in_squeeze = np.where(
                    _valid_mask, (_bb_u <= _kc_upper) & (_bb_l >= _kc_lower), False
                )

                # Count consecutive squeeze bars (most-recent bar backwards)
                _sq_dur = 0
                for _sj in range(
                    len(_in_squeeze) - 1, max(len(_in_squeeze) - 25, -1), -1
                ):
                    if _in_squeeze[_sj]:
                        _sq_dur += 1
                    else:
                        break
                state.bb_kc_squeeze_active = _sq_dur >= _bkc_min_bars
                state.bb_kc_squeeze_duration = int(_sq_dur)

                # BBW percentile: rank current bandwidth vs 6-month rolling window
                # (~1095 1H bars; capped to available history)
                _bbw = (_bb_u - _bb_l) / np.maximum(np.abs(_bb_m), 1e-10)
                _bbw_curr = _bbw[-1] if not np.isnan(_bbw[-1]) else None
                if _bbw_curr is not None:
                    _hist_lb = min(1095, len(_bbw))
                    _bbw_hist = _bbw[-_hist_lb:]
                    _bbw_valid = _bbw_hist[~np.isnan(_bbw_hist)]
                    if len(_bbw_valid) >= 10:
                        state.bbw_percentile = float(
                            np.sum(_bbw_valid < _bbw_curr) / len(_bbw_valid) * 100.0
                        )
        except Exception as _bkc_err:
            logger.debug("[BB/KC squeeze] compute error: %s", _bkc_err)
        # ─────────────────────────────────────────────────────────────────────

        # ── PHASE 3A: NR7-ID detection ───────────────────────────────────────
        # NR7: current bar's range is the narrowest of the last 7 bars.
        # NR7-ID: NR7 AND current bar is inside the previous bar.
        # Both flags indicate terminal tightness before breakout (Scenario B).
        try:
            if df is not None and len(df) >= 8:
                _nr7_lb = 7
                _bar_rng = df["high"].values - df["low"].values
                _curr_rng = float(_bar_rng[-1])
                _prior_rng = _bar_rng[-(_nr7_lb + 1) : -1]
                if len(_prior_rng) == _nr7_lb:
                    state.nr7_active = bool(_curr_rng <= float(np.min(_prior_rng)))
                if state.nr7_active:
                    _ph = float(df["high"].iloc[-2])
                    _pl = float(df["low"].iloc[-2])
                    _ch = float(df["high"].iloc[-1])
                    _cl_low = float(df["low"].iloc[-1])
                    state.nr7_id_active = bool(_ch <= _ph and _cl_low >= _pl)
        except Exception as _nr7_err:
            logger.debug("[NR7-ID] compute error: %s", _nr7_err)
        # ─────────────────────────────────────────────────────────────────────

        # ── PHASE 3A: RANGE_CLASSIFICATION ───────────────────────────────────
        # Classifies market context for MR routing and Council filtering.
        # Priority order: SQUEEZE > PULLBACK > TRENDING > RANGING
        #   SQUEEZE  — BB/KC squeeze active; volatility compressed, pre-breakout
        #   PULLBACK — Livermore in NATURAL_RETRACEMENT or NATURAL_REBOUND
        #   TRENDING — Livermore in MAIN_UP or MAIN_DOWN AND ADX > 25
        #   RANGING  — Livermore in SECONDARY states or ADX < 20; no clear trend
        try:
            _rc_lsm = getattr(state, "livermore_state_1h", None)
            _rc_adx = None
            if df is not None and len(df) >= 15:
                _rc_adx_arr = ta.ADX(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
                _v = _rc_adx_arr[-1]
                _rc_adx = float(_v) if not np.isnan(_v) else None

            if state.bb_kc_squeeze_active or state.nr7_active:
                state.range_classification = "SQUEEZE"
            elif _rc_lsm in ("NATURAL_RETRACEMENT", "NATURAL_REBOUND"):
                state.range_classification = "PULLBACK"
            elif _rc_lsm in ("MAIN_UP", "MAIN_DOWN") and (
                _rc_adx is not None and _rc_adx > 25
            ):
                state.range_classification = "TRENDING"
            else:
                state.range_classification = "RANGING"

            logger.debug(
                "[RANGE_CLASS] %s: %s (lsm=%s adx=%s squeeze=%s nr7=%s)",
                self.asset_type,
                state.range_classification,
                _rc_lsm,
                f"{_rc_adx:.1f}" if _rc_adx is not None else "n/a",
                state.bb_kc_squeeze_active,
                state.nr7_active,
            )
        except Exception as _rc_err:
            logger.debug("[RANGE_CLASS] compute error: %s", _rc_err)
        # ─────────────────────────────────────────────────────────────────────

        # ── PHASE 4: ema200_1d_dist_atr ──────────────────────────────────────
        # Distance from current close to the 200-period EMA on the daily
        # timeframe, normalised by the 1H ATR.
        # Item 19a: was resampling the 1H dataframe to daily bars itself (only
        # ~50 synthetic daily candles from a typical 1H feed — a much noisier
        # EMA200 than the real daily series the Constitution Gate already
        # computes). Now reads the real 1D-feed EMA200 the governor produced
        # (governor_data["ema_1d_200"], wired from GovernorStatus.ema_200 in
        # mtf_regime_detector.py), falling back to skip (leaving
        # ema200_1d_dist_atr at its default) if governor_data has no 1D read
        # yet — e.g. very first cycle before MTF regime has run once.
        # Used by MR Mode 1 to apply a −0.10 confidence penalty when BTC is
        # within 1 ATR of the daily EMA200 (historically a high-failure zone).
        try:
            _ema200_1d = governor_data.get("ema_1d_200") if governor_data else None
            if _ema200_1d and df is not None and len(df) > 0 and _atr > 0:
                _dist_1d = abs(float(df["close"].iloc[-1]) - float(_ema200_1d))
                state.ema200_1d_dist_atr = _dist_1d / _atr
        except Exception as _ema200_err:
            logger.debug("[ema200_1d_dist_atr] compute error: %s", _ema200_err)
        # ─────────────────────────────────────────────────────────────────────

        # A12: compute net_conviction unconditionally so main.py's pattern-
        # confluence veto (which reads it off composite_state) is reachable
        # in every mode, not just when _score_confluence separately runs it.
        # Neutral tf_conf/signal defaults here — _score_confluence overwrites
        # this with the real values when it runs afterward in performance mode.
        try:
            self._compute_net_conviction(state)
        except Exception as _nc_err:
            logger.debug("[net_conviction] compute error: %s", _nc_err)

        # ═══════════════════════════════════════════════════════════════
        # ACTIVITY LAYER (Plan 1A) — compute observation-only readouts.
        # Reads ONLY already-populated fields above. Writes nothing that
        # any decision path currently consumes. Fail-safe: on any error,
        # fields keep their neutral defaults.
        # ═══════════════════════════════════════════════════════════════
        try:
            # ---- 1. Compression dial (0..1) ----------------------------
            # Blend three lenses already on the board into one knob:
            #   squeeze_strength  (EMA convergence, 0..1)
            #   bb_kc squeeze     (volatility-band, scaled by duration)
            #   nr7_active        (single-bar terminal tightness)
            _sq = float(getattr(state, "squeeze_strength", 0.0) or 0.0)
            _bk = 1.0 if getattr(state, "bb_kc_squeeze_active", False) else 0.0
            _bk_dur = float(getattr(state, "bb_kc_squeeze_duration", 0) or 0)
            _bk_scaled = _bk * min(1.0, 0.4 + _bk_dur * 0.06)   # longer squeeze = tighter
            _nr = 0.5 if getattr(state, "nr7_active", False) else 0.0
            # Weighted max-style blend, capped at 1.0. Max() not sum() so
            # three overlapping lenses don't triple-count into >1.
            state.activity_compression = float(min(1.0, max(_sq, _bk_scaled, _nr)))

            # ---- 2. Coiled-release (transition event) ------------------
            # The board already computes this exact transition as
            # coiled_spring (squeeze was active, just released). Surface it
            # under the activity namespace so generators read one API.
            state.activity_coiled_release = bool(getattr(state, "coiled_spring", False))

            # ---- 3. Post-sweep (directional reversal event) ------------
            _sweep = bool(getattr(state, "sweep_detected", False))
            _sweep_dir = int(getattr(state, "sweep_direction", 0) or 0)
            if _sweep and _sweep_dir != 0:
                state.activity_post_sweep = True
                state.activity_post_sweep_dir = _sweep_dir

            # ---- 4. Ladder proximity (dist to NEAREST individual line) -
            # Q2b already put the nearest upper/lower ladder lines on the
            # board. Compute distance to whichever is closer, in ATR.
            _price = None
            try:
                _price = float(df["close"].iloc[-1])
            except Exception:
                _price = None
            _atr_prox = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
            _up = getattr(state, "zone_4h_current_upper", None)
            _lo = getattr(state, "zone_4h_current_lower", None)
            if _price is not None and _atr_prox > 0:
                _cands = []
                if _up is not None:
                    _cands.append((abs(_up - _price) / _atr_prox, "ABOVE"))
                if _lo is not None:
                    _cands.append((abs(_price - _lo) / _atr_prox, "BELOW"))
                if _cands:
                    _best = min(_cands, key=lambda c: c[0])
                    state.activity_ladder_dist_atr = float(_best[0])
                    state.activity_ladder_side = _best[1]

            # ---- 5. Breakout-imminence WITH texture --------------------
            # Fires only when compression is meaningful AND price is near a
            # ladder line AND volatility hasn't already expanded. Texture
            # names WHICH flavour is driving it, so MR/TF can later react
            # to the flavour relevant to their role.
            _near_line = (
                state.activity_ladder_dist_atr is not None
                and state.activity_ladder_dist_atr <= 0.75
            )
            if state.activity_compression >= 0.5 and _near_line:
                # Texture selection (priority: wick > sudden > crawl):
                #   WICK   — a sweep/rejection is the driver (rejection wick)
                #   SUDDEN — an outside/expansion bar is present
                #   CRAWL  — grinding compression with neither of the above
                if getattr(state, "sweep_detected", False) or getattr(state, "rejection_at_level", False):
                    _texture = "WICK"
                elif getattr(state, "outside_bar", False):
                    _texture = "SUDDEN"
                else:
                    _texture = "CRAWL"
                state.activity_breakout_imminent = True
                state.activity_breakout_texture = _texture
        except Exception as _act_err:
            logger.debug("[activity_layer] compute error: %s", _act_err)

        # Observation log — lets us watch the activity layer during soak.
        if state.activity_breakout_imminent or state.activity_coiled_release or state.activity_post_sweep:
            logger.info(
                "[ACTIVITY] %s: compression=%.2f%s%s%s ladder=%s@%.2fATR",
                self.asset_type,
                state.activity_compression,
                " COILED_RELEASE" if state.activity_coiled_release else "",
                (" POST_SWEEP:%+d" % state.activity_post_sweep_dir) if state.activity_post_sweep else "",
                (" BREAKOUT_IMMINENT:%s" % state.activity_breakout_texture) if state.activity_breakout_imminent else "",
                state.activity_ladder_side or "none",
                state.activity_ladder_dist_atr if state.activity_ladder_dist_atr is not None else -1.0,
            )

        # ═══════════════════════════════════════════════════════════════
        # TRAJECTORY LAYER (Plan 1B) — track a setup across cycles.
        # Observation-only: writes setup_* readouts, consumed by nobody yet.
        # Reads activity_compression + bos/choch fields set earlier this
        # method. Fail-safe: on error, no setup is tracked this cycle.
        # ═══════════════════════════════════════════════════════════════
        try:
            _asset = self.asset_type
            _cur = self._active_setup.get(_asset)   # None or dict
            _lsm_now = getattr(state, "livermore_state_1h", None)
            _comp = float(getattr(state, "activity_compression", 0.0) or 0.0)

            # Livermore up/down context (module-level frozensets in livermore_*).
            _UP = {"MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT"}
            _DOWN = {"MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND"}

            # ---- STEP 1: age + energy-trend an existing setup ----------
            if _cur is not None:
                _cur["age"] = int(_cur.get("age", 0)) + 1
                _prev_comp = float(_cur.get("last_compression", _comp))
                if _comp > _prev_comp + 0.05:
                    _cur["energy"] = "BUILDING"
                elif _comp < _prev_comp - 0.05:
                    _cur["energy"] = "FADING"
                else:
                    _cur["energy"] = "HOLDING"
                _cur["last_compression"] = _comp

            # ---- STEP 2: evidence-based death check --------------------
            # A setup dies when the tape invalidates it. Order: master
            # backstop first (state flip), then setup-specific evidence.
            _death_reason = None
            if _cur is not None:
                _born_state = _cur.get("born_state")
                _dir = int(_cur.get("dir", 0))
                # (a) MASTER BACKSTOP — Livermore 1H state transitioned away
                #     from the context the setup was born into.
                if _born_state is not None and _lsm_now != _born_state:
                    # Only kill if the new state flips the directional context
                    # (a benign retracement within the same trend shouldn't
                    # necessarily kill it, but a cross to the opposite camp does).
                    _born_up = _born_state in _UP
                    _now_up = _lsm_now in _UP
                    _now_down = _lsm_now in _DOWN
                    if (_born_up and _now_down) or ((not _born_up) and _now_up):
                        _death_reason = "LSM_STATE_FLIP"
                # (b) SETUP-SPECIFIC — a failed breakout against the setup.
                if _death_reason is None and getattr(state, "failed_breakout", False):
                    _death_reason = "FAILED_BREAKOUT"
                # (c) STRUCTURE-AGAINST — a directional structure break the
                #     opposite way (long setup sees bearish BOS, or vice versa).
                if _death_reason is None:
                    if _dir == 1 and getattr(state, "bos_bearish", False):
                        _death_reason = "OPPOSING_BOS"
                    elif _dir == -1 and getattr(state, "bos_bullish", False):
                        _death_reason = "OPPOSING_BOS"

            if _cur is not None and _death_reason is not None:
                state.setup_died = True
                state.setup_death_reason = _death_reason
                self._active_setup[_asset] = None
                _cur = None

            # ---- STEP 3: birth check (only if nothing is alive) --------
            if _cur is None:
                _born = None
                # TF setup is born at the BREAK (BOS), either direction.
                if getattr(state, "bos_bullish", False):
                    _born = {"kind": "TF_CONT", "dir": 1}
                elif getattr(state, "bos_bearish", False):
                    _born = {"kind": "TF_CONT", "dir": -1}
                # MR setup is born at the CHANGE OF CHARACTER (CHoCH) — the
                # earliest anomaly. CHoCH takes precedence when both appear
                # in a way that implies a reversal is starting.
                if getattr(state, "choch_bullish", False):
                    _born = {"kind": "MR_REV", "dir": 1}
                elif getattr(state, "choch_bearish", False):
                    _born = {"kind": "MR_REV", "dir": -1}

                if _born is not None:
                    _born.update({
                        "age": 0,
                        "born_state": _lsm_now,
                        "born_compression": _comp,
                        "last_compression": _comp,
                        "energy": "HOLDING",
                    })
                    self._active_setup[_asset] = _born
                    _cur = _born

            # ---- STEP 4: publish readouts ------------------------------
            if _cur is not None:
                state.setup_active = True
                state.setup_kind = _cur.get("kind")
                state.setup_dir = int(_cur.get("dir", 0))
                state.setup_age = int(_cur.get("age", 0))
                state.setup_energy_trend = _cur.get("energy")

            self._prev_compression[_asset] = _comp
        except Exception as _traj_err:
            logger.debug("[trajectory] compute error: %s", _traj_err)

        # Observation log — watch setups born / maturing / dying during soak.
        if state.setup_died:
            logger.info(
                "[TRAJECTORY] %s: setup DIED (%s)",
                self.asset_type, state.setup_death_reason,
            )
        elif state.setup_active:
            logger.info(
                "[TRAJECTORY] %s: %s dir=%+d age=%d energy=%s",
                self.asset_type, state.setup_kind, state.setup_dir,
                state.setup_age, state.setup_energy_trend,
            )

        # ══════════════════════════════════════════════════════════════════
        # BRC: Break-Retest-Close completed-proof state (OBSERVATION ONLY)
        #   Beat 1 (break, remembered): setup_active + setup_dir + setup_kind
        #   Beat 2 (retest): price wicked to the reference within 8 bars
        #   Beat 3 (strict close): THIS bar closes strictly through it
        #   Reference by origin:
        #     MR_REV  (CHoCH): Livermore anchor_natural_low/high
        #     TF_CONT (BOS):   last_swing_high_4h / last_swing_low_4h
        # ══════════════════════════════════════════════════════════════════
        try:
            _brc_active = bool(getattr(state, "setup_active", False))
            _brc_dir    = int(getattr(state, "setup_dir", 0) or 0)
            _brc_kind   = getattr(state, "setup_kind", None)

            if _brc_active and _brc_dir != 0 and df is not None and len(df) >= 9:
                _brc_ref = None
                if _brc_kind == "MR_REV":
                    _brc_ref = (
                        getattr(state, "livermore_anchor_natural_low", None) if _brc_dir == 1
                        else getattr(state, "livermore_anchor_natural_high", None)
                    )
                elif _brc_kind == "TF_CONT":
                    _brc_ref = (
                        getattr(state, "last_swing_high_4h", None) if _brc_dir == 1
                        else getattr(state, "last_swing_low_4h", None)
                    )

                if _brc_ref is not None and float(_brc_ref) > 0:
                    _brc_ref = float(_brc_ref)
                    _brc_close = float(df["close"].iloc[-1])
                    _brc_win_high = df["high"].iloc[-9:-1].values
                    _brc_win_low  = df["low"].iloc[-9:-1].values

                    if _brc_dir == 1:
                        _retested = any(l <= _brc_ref for l in _brc_win_low)
                        _closed_through = _brc_close > _brc_ref
                    else:
                        _retested = any(h >= _brc_ref for h in _brc_win_high)
                        _closed_through = _brc_close < _brc_ref

                    if _retested and _closed_through:
                        state.brc_confirmed = True
                        state.brc_direction = _brc_dir
                        state.brc_kind = _brc_kind
                        state.brc_tier = None
                        logger.info(
                            "[BRC] %s: CONFIRMED %s dir=%+d ref=%.5g close=%.5g "
                            "(strict close-through, 8-bar retest)",
                            self.asset_type, _brc_kind, _brc_dir, _brc_ref, _brc_close,
                        )
        except Exception as _brc_err:
            logger.debug("[BRC] compute error (non-blocking): %s", _brc_err)

        state.sanitise()
        return state

    # ── D.1: Trend Lifecycle Modifier ────────────────────────────────────

    def _update_trend_lifecycle(self, state, regime_name: str, current_dt=None):
        """
        Classify where in the trend lifecycle this asset sits.

        Phase 2: When Livermore state machine is warmed, derive lifecycle_phase
        directly from the Livermore state and its age (bars in state).
        This unblocks institutional patterns which were stuck on ESTABLISHED
        because the old regime-event-based derivation never transitioned.

        MRS lifecycle mapping (Livermore):
          MAIN_UP / MAIN_DOWN  age  1–5  bars → PICKUP
          MAIN_UP / MAIN_DOWN  age  5–15 bars → CONFIRMATION
          MAIN_UP / MAIN_DOWN  age  15+  bars → ESTABLISHED
          NATURAL_RETRACEMENT / NATURAL_REBOUND → FADING
          SECONDARY_RETRACEMENT / SECONDARY_REBOUND → EXHAUSTION

        Fallback: original regime-event logic if Livermore is not warmed.

        Args:
            state:       CompositeState to update in place.
            regime_name: Current regime string (e.g. "BULLISH").
            current_dt:  The bar's actual timestamp.  Supply this in backtesting
                         so that regime_age_hours reflects elapsed *bar* time, not
                         wall-clock time.  Defaults to datetime.now() for live use.
        """
        from datetime import datetime, timezone

        asset = self.asset_type

        # ── PHASE 2: Livermore-based lifecycle derivation ────────────────────
        # When the state machine is warmed (livermore_state_4h is populated),
        # skip the regime-event logic entirely and derive from state age.
        # regime_age_hours / regime_age_ratio still need updating so the
        # confluence engine has valid context — handled below.
        _lsm_state = getattr(state, "livermore_state_4h", None)
        _lsm_age = getattr(state, "livermore_state_age_4h", 0)
        if _lsm_state is not None:
            if _lsm_state in ("MAIN_UP", "MAIN_DOWN"):
                # Load per-asset age thresholds from config — avoids hardcoded 5/15.
                try:
                    if not hasattr(self, "_lifecycle_age_cfg"):
                        import json as _json_lac

                        with open("config/aggregator_presets.json") as _lac_f:
                            _lac_data = _json_lac.load(_lac_f)
                        self._lifecycle_age_cfg = _lac_data.get(
                            "REQUIRED_SCORE_MODIFIER", {}
                        ).get("lifecycle_age_thresholds", {})
                    _asset_lac = self._lifecycle_age_cfg.get(
                        self.asset_type,
                        self._lifecycle_age_cfg.get(
                            "default", {"pickup_max": 4, "confirmation_max": 12}
                        ),
                    )
                    _pickup_max = _asset_lac.get("pickup_max", 4)
                    _confirm_max = _asset_lac.get("confirmation_max", 12)
                except Exception:
                    _pickup_max, _confirm_max = 4, 12  # safe fallback
                if _lsm_age <= _pickup_max:
                    state.lifecycle_phase = "PICKUP"
                elif _lsm_age <= _confirm_max:
                    state.lifecycle_phase = "CONFIRMATION"
                else:
                    state.lifecycle_phase = "ESTABLISHED"
            elif _lsm_state in ("NATURAL_RETRACEMENT", "NATURAL_REBOUND"):
                state.lifecycle_phase = "FADING"
            elif _lsm_state in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
                state.lifecycle_phase = "EXHAUSTION"
            else:
                state.lifecycle_phase = "ESTABLISHED"
            logger.debug(
                "[LIFECYCLE] %s Livermore=%s age=%d → %s",
                asset,
                _lsm_state,
                _lsm_age,
                state.lifecycle_phase,
            )
            # Still update regime-age fields for confluence engine downstream.
            # Fall through to the regime-age calculation block below.
        # ─────────────────────────────────────────────────────────────────────
        # Use provided timestamp if given, otherwise real wall-clock time.
        # In backtesting all bars run in seconds of wall time, so datetime.now()
        # would keep regime_age_hours ≈ 0 for the entire run.
        now = current_dt if current_dt is not None else datetime.now()
        # Ensure tz-naive for consistent arithmetic
        if hasattr(now, "tzinfo") and now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        prev = self._previous_regime.get(asset)

        # Detect transition
        if prev and prev != regime_name:
            _prev_start = self._regime_start_time.get(asset, now)
            if hasattr(_prev_start, "tzinfo") and _prev_start.tzinfo is not None:
                _prev_start = _prev_start.replace(tzinfo=None)
            duration = (now - _prev_start).total_seconds() / 3600
            if asset not in self._regime_durations:
                self._regime_durations[asset] = []
            self._regime_durations[asset].append(duration)
            if len(self._regime_durations[asset]) > 50:
                self._regime_durations[asset] = self._regime_durations[asset][-50:]

            trans_key = (prev, regime_name)
            if asset not in self._transition_counts:
                self._transition_counts[asset] = {}
            self._transition_counts[asset][trans_key] = (
                self._transition_counts[asset].get(trans_key, 0) + 1
            )

            self._regime_start_time[asset] = now

            # Only overwrite lifecycle_phase from regime events when Livermore
            # is not yet warmed — once Livermore is running it has higher authority.
            if _lsm_state is None:
                if "NEUTRAL" in prev and "SLIGHTLY" in regime_name:
                    state.lifecycle_phase = "PICKUP"
                elif "SLIGHTLY" in prev and regime_name in ("BULLISH", "BEARISH"):
                    state.lifecycle_phase = "CONFIRMATION"
                elif regime_name in ("BULLISH", "BEARISH") and prev in (
                    "BULLISH",
                    "BEARISH",
                ):
                    state.lifecycle_phase = "ESTABLISHED"
                elif prev in ("BULLISH", "BEARISH") and "SLIGHTLY" in regime_name:
                    state.lifecycle_phase = "FADING"
                elif (
                    prev
                    in ("BULLISH", "BEARISH", "SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH")
                    and regime_name == "NEUTRAL"
                ):
                    state.lifecycle_phase = "EXHAUSTION"
                else:
                    state.lifecycle_phase = "ESTABLISHED"
        elif prev == regime_name:
            pass  # Same regime — keep current phase, just update age
        else:
            # First observation (after restart or new asset)
            self._regime_start_time[asset] = now
            # Default to ESTABLISHED so pattern layer is active immediately.
            # Overridden by Livermore derivation above if machine is warmed.
            if _lsm_state is None:
                state.lifecycle_phase = "ESTABLISHED"
                logger.info(f"[LIFECYCLE] {asset} initialized to ESTABLISHED (Startup)")

        self._previous_regime[asset] = regime_name

        # Regime age — uses bar timestamp in backtest, wall-clock in live
        _start = self._regime_start_time.get(asset, now)
        if hasattr(_start, "tzinfo") and _start.tzinfo is not None:
            _start = _start.replace(tzinfo=None)
        state.regime_age_hours = (now - _start).total_seconds() / 3600

        # Median regime duration (dynamic per asset)
        durations = self._regime_durations.get(asset, [])
        state.median_regime_duration = (
            float(np.median(durations)) if len(durations) >= 5 else 12.0
        )
        state.regime_age_ratio = state.regime_age_hours / max(
            state.median_regime_duration, 1.0
        )

    def _is_genuine_rejection(self, df, level, atr, direction, n_candles=2):
        """Gate Tier 4.3 — borrowed principle from momentum-alignment (Gate
        Tier 1.5): a rejection only counts if price actually traveled away
        from the level, not just closed on the technically-correct side of
        it. Standalone utility, not yet wired into a call site: the repo has
        no mechanism literally named "zone ladder" as the redesign plan
        assumed, and the closest real analog — _update_structure's
        nearby_4h_level/support/resistance "tests" counters just below —
        increments a test count exactly WHEN price is close to the level
        (best_dist < 0.3 ATR), which is structurally the opposite moment
        from "has price displaced 0.3+ ATR away from the level." Wiring this
        into that block using the same current price would make the "tests"
        counter never increment. Left available for whichever call site
        actually needs a post-retest displacement confirmation.
        """
        try:
            if atr <= 0 or len(df) < n_candles + 1:
                return True  # insufficient data, don't block on this alone
            _recent_closes = df["close"].iloc[-(n_candles):].values
            _displacement = abs(_recent_closes[-1] - level)
            return _displacement >= (0.3 * atr)
        except Exception:
            return True

    # ── E.1: ChoCh / BOS Detection ───────────────────────────────────────

    def _update_structure(self, state, df):
        """
        Detect Break of Structure and Change of Character.

        Pivot requirements:
          - 4 bars committed on the LEFT (proves the level was real, not a flicker)
          - 2 bars rejection on the RIGHT (catches the turn early enough to be useful)
          - Minimum 0.3 x ATR depth measured against prior CLOSES (not wicks, which
            include sweep noise that inflates the apparent size of micro-moves)

        Using numpy arrays throughout: faster and avoids pandas index alignment
        surprises when slicing with integers inside a loop.
        """
        try:
            if len(df) < 20:
                return

            highs_arr  = df["high"].values
            lows_arr   = df["low"].values
            closes_arr = df["close"].values

            _atr_raw   = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
            _min_depth = 0.3 * _atr_raw if _atr_raw > 0 else 0.0

            # Brain rebuild Part 0.5 (Tier X.1) — four real problems fixed:
            #  1. No bound on how far back "recent" could search — now capped
            #     at MAX_LOOKBACK_BARS.
            #  2. The swing-low classification used to be gated on
            #     `not state.bos_detected` / `not state.choch_detected`, so
            #     whichever direction's swing-high check fired first silently
            #     blocked the other direction's own, independent signal.
            #     Removed — both directions are now evaluated unconditionally.
            #  3. Appended the wick (highs_arr[i]/lows_arr[i]) despite the
            #     depth check already measuring against prior CLOSES (see
            #     this method's own docstring) — now appends closes_arr[i]
            #     for real close-based comparison, not wick noise.
            #  4. Failed completely silently (`except Exception: pass`) — now
            #     logs loudly and leaves state at its last good value instead
            #     of pretending nothing happened.
            MAX_LOOKBACK_BARS = 50  # bounds how far back a "recent" pivot can be
            swing_highs = []
            swing_lows  = []
            _search_floor = max(6, len(highs_arr) - 3 - MAX_LOOKBACK_BARS)

            # ── Swing HIGH: 4 left, 2 right, depth vs prior closes ──────────
            for i in range(len(highs_arr) - 3, _search_floor, -1):
                if (
                    highs_arr[i] > highs_arr[i - 1]
                    and highs_arr[i] > highs_arr[i - 2]
                    and highs_arr[i] > highs_arr[i - 3]
                    and highs_arr[i] > highs_arr[i - 4]
                    and highs_arr[i] > highs_arr[i + 1]
                    and highs_arr[i] > highs_arr[i + 2]
                    and (highs_arr[i] - max(closes_arr[i - 4:i]) >= _min_depth)
                ):
                    swing_highs.append(closes_arr[i])
                    if len(swing_highs) >= 2:
                        break

            # ── Swing LOW: symmetric, depth vs prior closes ─────────────────
            for i in range(len(lows_arr) - 3, _search_floor, -1):
                if (
                    lows_arr[i] < lows_arr[i - 1]
                    and lows_arr[i] < lows_arr[i - 2]
                    and lows_arr[i] < lows_arr[i - 3]
                    and lows_arr[i] < lows_arr[i - 4]
                    and lows_arr[i] < lows_arr[i + 1]
                    and lows_arr[i] < lows_arr[i + 2]
                    and (min(closes_arr[i - 4:i]) - lows_arr[i] >= _min_depth)
                ):
                    swing_lows.append(closes_arr[i])
                    if len(swing_lows) >= 2:
                        break

            # ── CHoCH / BOS classification — both directions independent ────
            if len(swing_highs) >= 2:
                if swing_highs[0] > swing_highs[1]:
                    state.bos_detected = True    # Higher high — trend continuing
                    state.bos_bullish = True
                elif swing_highs[0] < swing_highs[1]:
                    state.choch_detected = True  # Lower high — reversal warning
                    state.choch_bearish = True

            if len(swing_lows) >= 2:
                if swing_lows[0] < swing_lows[1]:
                    state.bos_detected = True    # Lower low — downtrend continuing
                    state.bos_bearish = True
                elif swing_lows[0] > swing_lows[1]:
                    state.choch_detected = True  # Higher low — reversal warning
                    state.choch_bullish = True

        except Exception as e:
            logger.error(
                f"[BOS/CHOCH] Swing detection failed for {self.asset_type}, state left stale: {e}"
            )

    # ── E.2: MTF Structure Memory ─────────────────────────────────────────

    def _update_structure_memory(self, state, df, df_4h):
        """Track 4H swing levels. Delete broken ones. Link to state."""
        import talib as ta

        asset = self.asset_type
        if asset not in self._structure_levels:
            self._structure_levels[asset] = []

        try:
            current_price = df["close"].iloc[-1]
            _atr_arr = ta.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
            _atr = float(_atr_arr[-1])
            if np.isnan(_atr) or _atr <= 0:
                return

            # 3.0 ATR threshold (was 0.5) — prevents premature deletion of the
            # entry level as a trade develops and price moves away from it.
            # 0.5 ATR caused structural stops to fall back to ATR within hours
            # of a trade opening as price moved in favour.
            self._structure_levels[asset] = [
                lvl
                for lvl in self._structure_levels[asset]
                if abs(current_price - lvl["price"]) / _atr <= 3.0
                and lvl.get("age_hours", 0) < 2160
            ]

            # Age all levels
            for lvl in self._structure_levels[asset]:
                lvl["age_hours"] = lvl.get("age_hours", 0) + 1

            # Role reversal: a broken ceiling that price holds above becomes a
            # new floor, and vice versa. 0.3% buffer avoids flip-flopping on
            # noise right at the level.
            for lvl in self._structure_levels[asset]:
                if lvl["type"] == "swing_high" and current_price > lvl["price"] * 1.003:
                    lvl["type"] = "swing_low"
                    lvl["role_flipped_at"] = datetime.now(timezone.utc).isoformat()
                    lvl["tests"] = 0
                elif lvl["type"] == "swing_low" and current_price < lvl["price"] * 0.997:
                    lvl["type"] = "swing_high"
                    lvl["role_flipped_at"] = datetime.now(timezone.utc).isoformat()
                    lvl["tests"] = 0

            # Add new 4H swing points if available
            if df_4h is not None and len(df_4h) >= 10:
                _4h_highs = df_4h["high"].values
                _4h_lows = df_4h["low"].values
                for i in range(len(_4h_highs) - 3, 4, -1):
                    if (
                        _4h_highs[i] > _4h_highs[i - 1]
                        and _4h_highs[i] > _4h_highs[i + 1]
                    ):
                        # ROUTE B: first backward hit = most recent 4H swing high.
                        state.last_swing_high_4h = float(_4h_highs[i])
                        _exists = any(
                            abs(lvl["price"] - _4h_highs[i]) / _atr < 0.3
                            for lvl in self._structure_levels[asset]
                        )
                        if not _exists:
                            self._structure_levels[asset].append(
                                {
                                    "price": _4h_highs[i],
                                    "tf": "4H",
                                    "type": "swing_high",
                                    "tests": 0,
                                    "age_hours": 0,
                                }
                            )
                        break

                for i in range(len(_4h_lows) - 3, 4, -1):
                    if _4h_lows[i] < _4h_lows[i - 1] and _4h_lows[i] < _4h_lows[i + 1]:
                        # ROUTE B: first backward hit = most recent 4H swing low.
                        state.last_swing_low_4h = float(_4h_lows[i])
                        _exists = any(
                            abs(lvl["price"] - _4h_lows[i]) / _atr < 0.3
                            for lvl in self._structure_levels[asset]
                        )
                        if not _exists:
                            self._structure_levels[asset].append({
                                "price": _4h_lows[i],
                                "tf": "4H",
                                "type": "swing_low",
                                "tests": 0,
                                "age_hours": 0,
                            })
                        break

            # Collect all candidate levels within 3.0 ATR.
            # Sort by quality: most-tested first, then nearest.
            # A level tested 3 times at a price is more significant
            # than a fresh level 0.1 ATR closer.
            candidates = [
                lvl for lvl in self._structure_levels[asset]
                if abs(current_price - lvl["price"]) / _atr <= 3.0
            ]
            candidates.sort(
                key=lambda l: (
                    -l.get("tests", 0),
                    abs(current_price - l["price"]) / _atr,
                )
            )
            top3 = candidates[:3]

            if top3:
                best = top3[0]
                best_dist = abs(current_price - best["price"]) / _atr
                state.nearby_4h_level      = best["price"]
                state.nearby_4h_level_type = best.get("type")
                state.level_test_count  = best.get("tests", 0)
                _was_away = best.get("_last_dist_atr", 99) >= 0.5
                if best_dist < 0.3 and _was_away:
                    best["tests"] = best.get("tests", 0) + 1
                best["_last_dist_atr"] = best_dist

                # Second and third levels — fallback references for structural
                # stops and RetestEngine when the primary level is broken.
                state.nearby_4h_level_2 = top3[1]["price"] if len(top3) > 1 else None
                state.nearby_4h_level_3 = top3[2]["price"] if len(top3) > 2 else None

            # Item 3.1/3.5: direction-split support/resistance, each with its
            # own distinct-visit test count. Kept alongside nearby_4h_level
            # above rather than replacing it — VTM structural stops, the
            # BREAKOUT/CHASE tiers, and transition_detector.py still read the
            # single-level fields and are out of this item's scope.
            # Uses its own _last_dist_atr_side distance-memory key (separate
            # from the block above's _last_dist_atr) so the two systems'
            # debounce bookkeeping can't interfere with each other on a level
            # that happens to be picked by both.
            supports = [
                l for l in self._structure_levels[asset]
                if l["type"] == "swing_low" and l["price"] < current_price
            ]
            resistances = [
                l for l in self._structure_levels[asset]
                if l["type"] == "swing_high" and l["price"] > current_price
            ]
            _best_support = min(supports, key=lambda l: current_price - l["price"], default=None)
            _best_resistance = min(resistances, key=lambda l: l["price"] - current_price, default=None)

            if _best_support is not None:
                _support_dist = (current_price - _best_support["price"]) / _atr
                _support_was_away = _best_support.get("_last_dist_atr_side", 99) >= 0.5
                if _support_dist < 0.3 and _support_was_away:
                    _best_support["tests"] = _best_support.get("tests", 0) + 1
                _best_support["_last_dist_atr_side"] = _support_dist
            state.nearby_support_level = _best_support["price"] if _best_support else None
            state.nearby_support_level_tests = (
                _best_support.get("tests", 0) if _best_support else 0
            )

            if _best_resistance is not None:
                _resistance_dist = (_best_resistance["price"] - current_price) / _atr
                _resistance_was_away = _best_resistance.get("_last_dist_atr_side", 99) >= 0.5
                if _resistance_dist < 0.3 and _resistance_was_away:
                    _best_resistance["tests"] = _best_resistance.get("tests", 0) + 1
                _best_resistance["_last_dist_atr_side"] = _resistance_dist
            state.nearby_resistance_level = _best_resistance["price"] if _best_resistance else None
            state.nearby_resistance_level_tests = (
                _best_resistance.get("tests", 0) if _best_resistance else 0
            )
        except Exception:
            pass

    # ── Zone Ladder ──────────────────────────────────────────────────────

    def _update_zone_levels(self, asset, df_tf, tf: str, atr: float, current_price: float):
        """Maintain the zone-ladder level store for one timeframe.

        Mirrors _update_structure_memory's proven mechanics (3-bar fractal,
        tolerance grouping, role reversal, test hysteresis) but with real
        timestamps, no distance prune, and a full-window scan.
        """
        import time as _time
        from datetime import datetime as _dt, timezone as _tz

        if df_tf is None or len(df_tf) < 10:
            return

        _now = _time.time()
        _window_secs = 180 * 24 * 3600  # 180 days — the widest view we support

        if asset not in self._zone_levels:
            self._zone_levels[asset] = []

        # ── Prune by TIME only. Never by distance. ──
        self._zone_levels[asset] = [
            lvl for lvl in self._zone_levels[asset]
            if (_now - lvl.get("first_seen", _now)) < _window_secs
        ]

        # ── Role reversal — copied from _update_structure_memory:1648.
        # 0.3% buffer stops flip-flopping on noise. Test count resets on flip:
        # a flipped level is a new level and must earn its history again.
        for lvl in self._zone_levels[asset]:
            if lvl["tf"] != tf:
                continue
            if lvl["type"] == "swing_high" and current_price > lvl["price"] * 1.003:
                lvl["type"] = "swing_low"
                lvl["role_flipped_at"] = _dt.now(_tz.utc).isoformat()
                lvl["tests"] = 0
            elif lvl["type"] == "swing_low" and current_price < lvl["price"] * 0.997:
                lvl["type"] = "swing_high"
                lvl["role_flipped_at"] = _dt.now(_tz.utc).isoformat()
                lvl["tests"] = 0

        # ── Tolerance: 0.3 ATR, capped at 0.4% of price.
        # The cap stops two genuinely distinct levels' bands growing into each
        # other in high volatility. 0.4% ≈ 75 pips on GBPAUD.
        _tol = min(0.3 * atr, 0.004 * current_price)

        _highs = df_tf["high"].values
        _lows = df_tf["low"].values

        # ── FULL-WINDOW scan. No `break` — that's the point.
        # 3-bar fractal: a candle is a peak if its high beats both neighbours.
        for i in range(2, len(_highs) - 2):
            if _highs[i] > _highs[i - 1] and _highs[i] > _highs[i + 1]:
                if not any(
                    abs(lvl["price"] - _highs[i]) < _tol and lvl["tf"] == tf
                    for lvl in self._zone_levels[asset]
                ):
                    self._zone_levels[asset].append({
                        "price": float(_highs[i]),
                        "tf": tf,
                        "type": "swing_high",
                        "tests": 0,
                        "first_seen": _now,
                        "last_test": _now,
                        "role_flipped_at": None,
                        "_last_dist_atr": 99,
                    })
            if _lows[i] < _lows[i - 1] and _lows[i] < _lows[i + 1]:
                if not any(
                    abs(lvl["price"] - _lows[i]) < _tol and lvl["tf"] == tf
                    for lvl in self._zone_levels[asset]
                ):
                    self._zone_levels[asset].append({
                        "price": float(_lows[i]),
                        "tf": tf,
                        "type": "swing_low",
                        "tests": 0,
                        "first_seen": _now,
                        "last_test": _now,
                        "role_flipped_at": None,
                        "_last_dist_atr": 99,
                    })

        # ── Test counting with hysteresis — copied from :1725.
        # Without _was_away, price loitering near a level racks up dozens of
        # fake "tests" per day. It must leave (>=0.5 ATR) before returning
        # (<0.3 ATR) for a touch to count.
        if atr > 0:
            for lvl in self._zone_levels[asset]:
                if lvl["tf"] != tf:
                    continue
                _dist = abs(current_price - lvl["price"]) / atr
                if _dist < 0.3 and lvl.get("_last_dist_atr", 99) >= 0.5:
                    lvl["tests"] = lvl.get("tests", 0) + 1
                    lvl["last_test"] = _now
                lvl["_last_dist_atr"] = _dist

    def _build_zone_view(self, df_tf, asset, tf: str, current_price: float,
                         extended: bool = False) -> dict:
        """Return the zone picture for one timeframe.

        tf="4H" reads only 4H levels. tf="1D" reads BOTH 4H and 1D levels —
        that's the "1D sees 4H's lines" rule, one-directional by design.

        extended=False → 90-day window (3 zones). True → 180-day (5 zones).
        """
        import time as _time

        _out = {"current_upper": None, "current_lower": None,
                "current_upper_tests": 0, "current_lower_tests": 0,
                "current_upper_type": None, "current_lower_type": None,
                "outer_high": None, "outer_low": None}
        if df_tf is None or len(df_tf) < 10:
            return _out

        _days = 180 if extended else 90
        _bars = _days * (6 if tf == "4H" else 1)
        _slice = df_tf.iloc[-_bars:] if len(df_tf) > _bars else df_tf

        # ── Outer edges: BODY-ONLY. Wicks never count.
        # Computed fresh, never stored — the dataframe fully answers this and
        # it can't go stale.
        _bodies_hi = _slice[["open", "close"]].max(axis=1)
        _bodies_lo = _slice[["open", "close"]].min(axis=1)
        _out["outer_high"] = float(_bodies_hi.max())
        _out["outer_low"] = float(_bodies_lo.min())

        # ── Inner lines: from the store. These MUST persist — their value is
        # the accumulated test count and role-flip history, which cannot be
        # recomputed from a snapshot of candles.
        _now = _time.time()
        _window = _days * 24 * 3600
        _visible = "4H" if tf == "4H" else None  # None = accept 4H and 1D

        _cands = [
            lvl for lvl in self._zone_levels.get(asset, [])
            if (_now - lvl.get("first_seen", _now)) < _window
            and (_visible is None or lvl["tf"] == _visible)
            and lvl.get("tests", 0) >= 1   # must have real history
        ]

        _above = [l for l in _cands if l["price"] > current_price]
        _below = [l for l in _cands if l["price"] < current_price]

        if _above:
            # Keep the whole level dict — we need its test-count and type,
            # not just the price. min() already selected the nearest line above.
            _u = min(_above, key=lambda l: l["price"] - current_price)
            _out["current_upper"] = _u["price"]
            _out["current_upper_tests"] = _u.get("tests", 0)
            _out["current_upper_type"] = _u.get("type")
        if _below:
            _d = max(_below, key=lambda l: l["price"])
            _out["current_lower"] = _d["price"]
            _out["current_lower_tests"] = _d.get("tests", 0)
            _out["current_lower_type"] = _d.get("type")

        return _out

    # ── E.3: MA Defense Validator ─────────────────────────────────────────

    def _update_ma_defense(self, state, df, span: int = 50, suffix: str = ""):
        """Check if key EMAs were tested and defended on this closed candle.

        span/suffix: parameterized (Stage 2 zone ladder) so the same
        wick/body defense math can run against EMA50/EMA200 on both 1H and
        1D data. Field names embed span (ema_{span}_status{suffix}) so the
        original span=50/suffix="" call keeps writing the exact original
        ema_50_status/ema_50_reclassified fields untouched. defense_strength
        and absorption_detected are NOT parameterized — every call (1H-50,
        1H-200, 1D-50, 1D-200) writes the same shared fields, so whichever
        call runs last in _build_composite_state's sequence wins.
        """
        try:
            candle = df.iloc[-1]
            _o = candle["open"]
            _h = candle["high"]
            _l = candle["low"]
            _c = candle["close"]
            _ema50 = df["close"].ewm(span=span, adjust=False).mean().iloc[-1]

            _pierced_from_above = _l < _ema50 < _c  # Wick below, closed above
            _broke_down = _c < _ema50 and _o > _ema50

            # ATR needed to judge "distance" — recalculate a quick estimate here
            try:
                import talib as _ta_ma

                _atr14 = _ta_ma.ATR(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
                _atr_val = float(_atr14[-1]) if not np.isnan(_atr14[-1]) else 0.0
            except Exception:
                _atr_val = abs(_c - _ema50) * 0.5  # rough fallback

            if _pierced_from_above:
                setattr(state, f"ema_{span}_status{suffix}", "DEFENDED")
                _wick = _ema50 - _l
                _body = abs(_c - _o)
                # F3: parameterised so the four calls (1H-50, 1H-200, 1D-50,
                # 1D-200) stop clobbering one another. The unsuffixed 1H-50
                # write is preserved as the canonical one STRUCTURE reads.
                _f3_strength = min(1.0, _wick / max(_body, 0.0001) / 3.0)
                setattr(state, f"defense_strength_{span}{suffix}", _f3_strength)
                if span == 50 and suffix == "":
                    state.defense_strength = _f3_strength

                # F3: judge THIS call's own strength (_f3_strength), not the
                # shared state.defense_strength field — that field only holds
                # the canonical 1H-50 value now, which would misjudge the
                # other three calls' own absorption/reclassification decision.
                if _f3_strength > 0.5 and state.effort_result_zscore > 1.5:
                    setattr(state, f"ema_{span}_reclassified{suffix}", "SUPPORT")
                    setattr(state, f"absorption_detected_{span}{suffix}", True)
                    if span == 50 and suffix == "":
                        state.absorption_detected = True
                else:
                    setattr(state, f"ema_{span}_reclassified{suffix}", "LINE")
            elif _broke_down:
                setattr(state, f"ema_{span}_status{suffix}", "BROKEN")
                setattr(state, f"ema_{span}_reclassified{suffix}", "RESISTANCE")
            elif _c > _ema50 and _atr_val > 0:
                # Price is above EMA50 — label "EMA_ABOVE" with a distance tier.
                # This unlocks the EMA_ABOVE branch in _score_confluence for
                # trend-continuation scoring without requiring a pierce/defend event.
                _dist_atr = (_c - _ema50) / _atr_val
                if _dist_atr < 3.0:
                    setattr(state, f"ema_{span}_status{suffix}", "EMA_ABOVE")  # close proximity — continuation
                    setattr(state, f"ema_{span}_reclassified{suffix}", "SUPPORT")  # treat as dynamic support
                else:
                    setattr(state, f"ema_{span}_status{suffix}", "EMA_ABOVE_FAR")  # extended from EMA50
                    setattr(state, f"ema_{span}_reclassified{suffix}", "LINE")
            elif _c < _ema50 and _atr_val > 0:
                _dist_atr = (_ema50 - _c) / _atr_val
                if _dist_atr < 3.0:
                    setattr(state, f"ema_{span}_status{suffix}", "EMA_BELOW")
                    setattr(state, f"ema_{span}_reclassified{suffix}", "RESISTANCE")
                else:
                    setattr(state, f"ema_{span}_status{suffix}", "EMA_BELOW_FAR")
                    setattr(state, f"ema_{span}_reclassified{suffix}", "LINE")
            else:
                setattr(state, f"ema_{span}_status{suffix}", "UNTESTED")

            # Fix #15: MA Defense diagnostics
            logger.debug(
                f"[MA_DEFENSE] {self.asset_type}: span={span}{suffix} "
                f"status={getattr(state, f'ema_{span}_status{suffix}')} "
                f"reclassified={getattr(state, f'ema_{span}_reclassified{suffix}')} "
                f"dist_to_50={abs(_c - _ema50):.4f} "
                f"defense_strength={state.defense_strength:.2f}"
            )
        except Exception:
            pass

    # ── G.1: Unified Liquidity Sweeps ────────────────────────────────────

    def _update_sweeps(self, state, df):
        """Check for PDH/PDL or Asian range sweeps (wicked through, closed back)."""
        asset = self.asset_type
        try:
            from datetime import datetime

            _h = df["high"].iloc[-1]
            _l = df["low"].iloc[-1]
            _c = df["close"].iloc[-1]

            # Use bar timestamp when available so backtest reflects actual bar time.
            # df.index[-1] is a pd.Timestamp for both live and backtest DataFrames.
            _bar_ts = df.index[-1] if not df.empty else None
            if _bar_ts is not None and hasattr(_bar_ts, "hour"):
                _hour = _bar_ts.hour
                _today = _bar_ts.date()
            else:
                _hour = datetime.utcnow().hour
                _today = datetime.utcnow().date()

            # Update Asian range (00:00-08:00 UTC)
            if 0 <= _hour < 8:
                self._asian_high[asset] = max(self._asian_high.get(asset, 0), _h)
                self._asian_low[asset] = min(
                    self._asian_low.get(asset, float("inf")), _l
                )

            # Update PDH/PDL daily — keyed on bar date not wall-clock date
            if self._pdh_date != _today:
                if len(df) > 24:
                    _yesterday = df.iloc[-25:-1]
                    self._pdh[asset] = _yesterday["high"].max()
                    self._pdl[asset] = _yesterday["low"].min()
                self._pdh_date = _today

            _asian_h = self._asian_high.get(asset)
            _asian_l = self._asian_low.get(asset)
            _pdh_val = self._pdh.get(asset)
            _pdl_val = self._pdl.get(asset)

            # Swept high = wicked above, closed below
            if _pdh_val and _h > _pdh_val and _c < _pdh_val:
                state.sweep_detected = True
                state.sweep_direction = 1
                state.sweep_level = _pdh_val
            elif _pdl_val and _l < _pdl_val and _c > _pdl_val:
                state.sweep_detected = True
                state.sweep_direction = -1
                state.sweep_level = _pdl_val
            elif _asian_h and 8 <= _hour <= 10 and _h > _asian_h and _c < _asian_h:
                state.sweep_detected = True
                state.sweep_direction = 1
                state.sweep_level = _asian_h
            elif _asian_l and 8 <= _hour <= 10 and _l < _asian_l and _c > _asian_l:
                state.sweep_detected = True
                state.sweep_direction = -1
                state.sweep_level = _asian_l
        except Exception:
            pass

    def _detect_upthrust(self, df) -> tuple:
        """
        Item 5.1: Wyckoff upthrust — the bearish mirror of
        s_mean_reversion._detect_spring(). A bar's wick sweeps above a prior
        swing high then closes back below it within spring_recovery_max_bars,
        with current-bar volume lower than the upthrust bar's (upthrust bar =
        buying climax; entry bar = quiet absorption). Reuses Mode 1's spring
        config thresholds since an upthrust is the mirror-image pattern of a
        spring, not a distinct phenomenon needing its own tuning.

        Returns: (found: bool, strength: float 0-1)
        """
        cfg = self.s_mean_reversion._mr3_cfg["mode1"]
        min_pen, max_pen = cfg["spring_min_penetration"], cfg["spring_max_penetration"]
        max_bars, swing_lb = cfg["spring_recovery_max_bars"], cfg["spring_swing_lookback"]
        if len(df) < swing_lb + max_bars + 2:
            return False, 0.0
        high, close, volume = df["high"].values, df["close"].values, df["volume"].values
        _win_end, _win_start = -(max_bars + 1), -(max_bars + 1) - swing_lb
        prior_swing_high = float(np.max(high[_win_start:_win_end]))
        for k in range(1, max_bars + 1):
            bar_idx = -(k + 1)
            bar_high, bar_close = float(high[bar_idx]), float(close[bar_idx])
            if bar_high <= prior_swing_high or bar_close >= prior_swing_high:
                continue
            penetration = (bar_high - prior_swing_high) / prior_swing_high
            if not (min_pen <= penetration <= max_pen):
                continue
            if volume[bar_idx] > 0 and volume[-1] >= volume[bar_idx]:
                continue
            strength = max(0.30, 1.0 - abs(penetration - 0.025) / 0.025)
            return True, float(min(strength, 1.0))
        return False, 0.0

    def _compute_institutional_pattern(self, df, state) -> None:
        """Item 5.2: rebuilt, tiered institutional-pattern classification.

        Real Wyckoff spring/upthrust detection takes priority (genuine price
        action evidence, high confidence) over the Livermore-state + volume/
        range heuristic (Item 2.7's version, now the fallback tier, lower
        confidence). Populates institutional_pattern_confidence alongside
        institutional_pattern so downstream consumers (AI validator Item 5.4,
        Pattern judge) can tell a confirmed spring from a soft heuristic read.
        """
        try:
            _spring_found, _spring_strength, _ = self.s_mean_reversion._detect_spring(df)
            if _spring_found and _spring_strength >= 0.5:
                state.institutional_pattern = "ACCUMULATION"
                state.institutional_pattern_confidence = _spring_strength
                return

            _upthrust_found, _upthrust_strength = self._detect_upthrust(df)
            if _upthrust_found and _upthrust_strength >= 0.5:
                state.institutional_pattern = "DISTRIBUTION"
                state.institutional_pattern_confidence = _upthrust_strength
                return

            # Brain Rebuild Part 2.3 (applied verbatim per explicit instruction,
            # 2026-07-09): fallback-tier "basing" definition changed from
            # Livermore-state category to nearest-level distance. Replaces
            # Item 2.7's lsm-based version (this was its only consumer of
            # `lsm`, so that lookup is dropped too).
            vol_ratio = df["volume"].iloc[-10:].mean() / max(df["volume"].iloc[-30:-10].mean(), 1e-9)
            range_pct = (df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / df["close"].iloc[-1]
            atr = getattr(state, "atr_fast", None) or df["close"].diff().abs().rolling(14).mean().iloc[-1]
            is_tight_range = range_pct < (2.5 * atr / df["close"].iloc[-1])

            _close = df["close"].iloc[-1]
            _supp = getattr(state, "nearby_support_level", None)
            _res = getattr(state, "nearby_resistance_level", None)
            _dist_supp = abs(_close - _supp) if _supp is not None else float("inf")
            _dist_res = abs(_close - _res) if _res is not None else float("inf")
            _basing_bullish = _dist_supp < _dist_res
            _basing_bearish = _dist_res < _dist_supp

            # P1: prefer the Livermore anchor — the same level BRV validates
            # and MR's spring check uses — over an independently-derived basing
            # level. Two references for one event is how they drift apart.
            _p1_anchor_lo = getattr(state, "livermore_anchor_natural_low", None)
            _p1_anchor_hi = getattr(state, "livermore_anchor_natural_high", None)
            if _p1_anchor_lo is not None and _basing_bullish:
                _dist_supp = abs(_close - float(_p1_anchor_lo))
            if _p1_anchor_hi is not None and _basing_bearish:
                _dist_res = abs(_close - float(_p1_anchor_hi))

            if _basing_bullish and (is_tight_range or vol_ratio > 1.3):
                _conf = 0.3 + 0.2 * min(1.0, atr / max(_dist_supp, 1e-9))
                state.institutional_pattern, state.institutional_pattern_confidence = "ACCUMULATION", _conf
            elif _basing_bearish and (is_tight_range or vol_ratio > 1.3):
                _conf = 0.3 + 0.2 * min(1.0, atr / max(_dist_res, 1e-9))
                state.institutional_pattern, state.institutional_pattern_confidence = "DISTRIBUTION", _conf
            elif is_tight_range:
                state.institutional_pattern, state.institutional_pattern_confidence = "COMPRESSION", 0.3
            else:
                state.institutional_pattern, state.institutional_pattern_confidence = None, 0.0
        except Exception as e:
            logger.debug(f"[PATTERN] compute error: {e}")
            state.institutional_pattern = None
            state.institutional_pattern_confidence = 0.0

    def _compute_volume_divergence(self, df, state) -> None:
        """Item 6.1: price-OBV divergence, computed once and shared via
        CompositeState instead of each consumer recomputing OBV separately."""
        try:
            price_chg = df["close"].diff()
            obv = (np.sign(price_chg) * df["volume"]).fillna(0).cumsum()
            obv_chg = obv.diff()
            state.bullish_divergence = bool((price_chg.iloc[-1] < 0) and (obv_chg.iloc[-1] > 0))
            state.bearish_divergence = bool((price_chg.iloc[-1] > 0) and (obv_chg.iloc[-1] < 0))
        except Exception as e:
            logger.debug(f"[VOLUME DIVERGENCE] compute error: {e}")
            state.bullish_divergence = False
            state.bearish_divergence = False

    # ── Section I: Confluence Engine ─────────────────────────────────────

    def _compute_net_conviction(self, state, tf_conf: float = 0.0, signal: int = 0) -> float:
        """
        A12: net_conviction extracted out of _score_confluence's STEP-2
        else-branch into its own method so it can be computed unconditionally
        from _build_composite_state — not just when _score_confluence itself
        runs (which only happens inside get_aggregated_signal's STEP 6B,
        gated on the Performance aggregator's own final_signal != 0). Council/
        hybrid mode only ever calls _build_composite_state directly (main.py's
        composite_state injection, to borrow Livermore/structure fields for
        the council judges) and never reaches STEP 6B, which previously left
        state.net_conviction permanently at its 0.0 dataclass default in
        those modes — making main.py's pattern-confluence veto's bypass
        check (`net_conviction <= threshold`, threshold defaults to -1.0)
        structurally unreachable there, since 0.0 is never <= -1.0.

        tf_conf/signal default to neutral (0.0/"unknown direction") when
        called from _build_composite_state before a signal direction is
        known — this only affects two minor sub-terms (CHoCH's direction-
        aware severity, and the order-book-wall direction approximation);
        every other term here is state-only. _score_confluence still passes
        the real tf_conf/signal it already has when it calls this itself.
        """
        _exhaust = 0.0
        if state.choch_detected:
            if signal == 1:
                _exhaust += 0.3 if state.regime_age_ratio <= 1.0 else 1.0
            elif signal == -1:
                _exhaust += 2.0
            else:
                _exhaust += 1.0  # unknown direction — moderate penalty
        if state.is_parabolic:
            _exhaust += 1.5
        if state.divergence_detected:
            _exhaust += state.divergence_strength * 2
        if state.regime_age_ratio > 1.5:
            _exhaust += min(2.0, state.regime_age_ratio - 1.5)
        if state.conviction_dying:
            _exhaust += 1.0
        if state.structural_decay:
            _exhaust += 1.5
        if state.absorption_detected:
            _exhaust += 1.0
        if state.vpd_diverging:
            _exhaust += 1.5
        if state.order_book_wall_detected:
            _tf_signal = 1 if tf_conf > 0 else -1  # approximate direction
            if _tf_signal == 1 and state.order_book_imbalance < -0.5:
                _exhaust += 1.5  # Sell wall blocking longs
            elif _tf_signal == -1 and state.order_book_imbalance > 0.5:
                _exhaust += 1.5  # Buy wall blocking shorts
        if state.spread_velocity_spike:
            _exhaust += 1.0
        if state.outside_bar:
            _exhaust += 0.5

        _confirm = 0.0
        if state.bos_detected:
            _confirm += 2.0
        if state.slopes_aligned:
            _confirm += 1.0
        if state.lifecycle_phase == "PICKUP":
            _confirm += 1.5
        if state.lifecycle_phase == "CONFIRMATION":
            _confirm += 1.0
        if state.squeeze_active:
            _confirm += 0.5
        if state.ema_50_status == "DEFENDED":
            _confirm += 1.0
        elif state.ema_50_status == "EMA_ABOVE":
            _confirm += 0.5
        if state.cvd_trend != 0 and not state.cvd_stale:
            _confirm += 1.0
        if state.level_defended:
            _confirm += 1.5

        _net = _confirm - _exhaust
        state.net_conviction = _net
        return _net
