"""
Composite State — The bot's shared nervous system.
Every analysis module reads from and writes to this object.
The Confluence Engine reads the complete state to score trades.

Phase 0B: sanitise() method added — NaN guard runs after all fields populated.
          order_book_imbalance and body_trend_ratio converted to Optional + _valid flag.
Phase 1:  Livermore fields populated once livermore_state_machine.py is built.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class CompositeState:
    """Updated on every closed candle. Passed between all analysis layers."""

    # ══════════════════════════════════════
    # LAYER 1: MACRO STATE
    # ══════════════════════════════════════
    lifecycle_phase: str = "ESTABLISHED"
    # Intentionally unconsumed — formatted debug string ("prev→new"), not judge-ready
    # data. RSM state_modifiers keys on current state alone, not a transition pair.
    # Build a fresh consumer if transition-specific logic is ever needed.
    regime_age_hours: float = 0.0
    regime_age_ratio: float = 0.0
    median_regime_duration: float = 12.0
    slopes_aligned: bool = False
    slope_diverging: bool = False
    structural_decay: bool = False

    # ══════════════════════════════════════
    # LAYER 2: STRUCTURAL REALITY
    # ══════════════════════════════════════
    choch_detected: bool = False
    bos_detected: bool = False
    # Directional siblings (added for VTM's human-alert layer, Part 3.5 follow-up):
    # choch_detected/bos_detected are each set from two independent, OPPOSITE-
    # meaning branches (lower-high vs higher-low for CHoCH; higher-high vs
    # lower-low for BOS) with no way to tell which one fired. These record
    # which direction actually triggered, additive — nothing that reads the
    # original two booleans needs to change.
    choch_bearish: bool = False  # lower high — bearish reversal warning
    choch_bullish: bool = False  # higher low — bullish reversal warning
    bos_bullish: bool = False    # higher high — uptrend continuing
    bos_bearish: bool = False    # lower low — downtrend continuing
    nearby_4h_level:   Optional[float] = None
    nearby_4h_level_2: Optional[float] = None   # Second nearest 4H structural level
    # ── Route B: confirmation anchors ──
    # Most recent 4H swing high / low before the current bar. The strategy's
    # confirmation rule ("close beyond the swing high before the retest began")
    # is a 4H measurement; RetestEngine only sees 1H, so the builder computes
    # this and RetestEngine reads it — same pattern as nearby_4h_level.
    last_swing_high_4h: Optional[float] = None
    last_swing_low_4h:  Optional[float] = None
    nearby_4h_level_3: Optional[float] = None   # Third nearest 4H structural level
    nearby_4h_level_type: Optional[str] = None  # "swing_high" / "swing_low" — current role, re-evaluated every cycle by role reversal
    # A4: level_defended was only ever computed against the single primary
    # nearby_4h_level. These extend the same rejection-wick defense check to
    # the 2nd/3rd nearest levels so RetestEngine can classify CLEAN off a
    # defended secondary/tertiary level when the primary isn't nearby.
    level_2_defended: bool = False
    level_3_defended: bool = False
    level_test_count: int = 0
    # Item 3.1/3.4: direction-split levels — nearest support (below price) and
    # nearest resistance (above price), each with its own distinct-visit test
    # count. Alongside nearby_4h_level (direction-agnostic nearest of either
    # type), not a replacement for it.
    nearby_support_level: Optional[float] = None
    nearby_support_level_tests: int = 0
    nearby_resistance_level: Optional[float] = None
    nearby_resistance_level_tests: int = 0
    # Item 3.4: volume/OBV divergence (Section 6) and Wyckoff pattern
    # confidence (Section 5) — computed once in _build_composite_state,
    # consumed by the Volume and Pattern judges respectively.
    bullish_divergence: bool = False
    bearish_divergence: bool = False
    institutional_pattern_confidence: float = 0.0
    level_defended: bool = False
    defense_strength: float = 0.0
    is_parabolic: bool = False
    distance_zscore: float = 0.0
    squeeze_active: bool = False
    squeeze_strength: float = 0.0
    coiled_spring: bool = False
    reversal_imminent: bool = False
    inside_bar: bool = False
    outside_bar: bool = False
    failed_breakout: bool = False
    ema_50_status: str = "UNTESTED"
    ema_50_reclassified: Optional[str] = None
    absorption_detected: bool = False
    effort_result_zscore: float = 0.0

    # ══════════════════════════════════════
    # LAYER 3: ORDER FLOW
    # ══════════════════════════════════════
    cvd_trend: int = 0
    cvd_stale: bool = True
    # Ratio fields — Optional + _valid flag. NaN means "no data", not zero.
    order_book_imbalance: Optional[float] = None
    order_book_imbalance_valid: bool = False
    order_book_wall_detected: bool = False
    spread_velocity_spike: bool = False
    spread_ratio: float = 1.0
    vpd_diverging: bool = False
    body_trend_ratio: Optional[float] = None
    body_trend_ratio_valid: bool = False
    conviction_dying: bool = False
    divergence_detected: bool = False
    divergence_strength: float = 0.0

    # ══════════════════════════════════════
    # LAYER 4: MICRO EXECUTION
    # ══════════════════════════════════════
    sweep_detected: bool = False
    sweep_direction: int = 0
    sweep_level: Optional[float] = None
    rejection_at_level: bool = False
    rejection_strength: float = 0.0
    session_name: str = "UNKNOWN"
    vwap_price: Optional[float] = None
    distance_to_vwap_atr: float = 0.0
    time_since_last_loss_hours: float = 999.0

    # ══════════════════════════════════════
    # AI INTEGRATION (reserved — sniper disconnected Phase 0B)
    # ══════════════════════════════════════

    # ══════════════════════════════════════
    # OUTPUT (set by Confluence Engine)
    # ══════════════════════════════════════
    institutional_pattern: Optional[str] = None
    net_conviction: float = 0.0

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 RESERVED — Livermore State Machine
    # All None/False/0 until livermore_state_machine.py populates them.
    # ══════════════════════════════════════════════════════════════
    livermore_state_4h: Optional[str] = None
    livermore_state_1h: Optional[str] = None
    livermore_anchor_main_up_max: Optional[float] = None
    livermore_anchor_main_down_min: Optional[float] = None
    livermore_anchor_natural_high: Optional[float] = None
    livermore_anchor_natural_low: Optional[float] = None
    livermore_state_age_4h: int = 0
    livermore_state_age_1h: int = 0
    livermore_dual_confirmation: bool = False
    is_silent_zone: bool = False
    # vol_down_ratio: volume on down-close / up-close bars. MR Mode 1 veto only.
    vol_down_ratio: Optional[float] = None
    vol_down_ratio_valid: bool = False
    # BB/KC squeeze fields
    bb_kc_squeeze_active: bool = False
    bb_kc_squeeze_duration: int = 0
    bbw_percentile: Optional[float] = None
    # NR7-ID fields
    nr7_active: bool = False
    nr7_id_active: bool = False
    # EMA 200 daily distance (Phase 4)
    ema200_1d_dist_atr: Optional[float] = None
    # Range classification (Phase 3A)
    # Values: TRENDING | PULLBACK | RANGING | SQUEEZE
    # TRENDING  — Livermore MAIN state, ADX > 25
    # PULLBACK  — Livermore NATURAL retracement/rebound within a MAIN leg
    # RANGING   — Livermore SECONDARY states or ADX < 20
    # SQUEEZE   — BB/KC squeeze active (pre-breakout, takes priority over RANGING)
    range_classification: Optional[str] = None
    # ══════════════════════════════════════════════════════════════
    # PHASE 3B RESERVED — Retest Engine
    # entry_type: populated by retest_engine; consumed by VTM.
    # Values: MR_PULLBACK / TREND_FOLLOWING / SPRING_ENTRY / RANGE_BOUNDARY /
    #         CONTINUATION / REJECT / None (no classification yet)
    # ══════════════════════════════════════════════════════════════
    entry_type: Optional[str] = None
    # Target-ladder box (Phase 4 ext, gated by phase_config.continuation_targets_enabled,
    # default OFF). Populated by retest_engine for RANGE_BOUNDARY/SPRING_ENTRY —
    # the structural range top/bottom used to build the midpoint/top/measured-move
    # ladder in VTM. None when the feature is off or no box was computed.
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    # ── Fix 1: phase_config gate flags (populated by main.py → consumed by aggregator + VTM)
    phase_config: dict = field(default_factory=dict)

    # ── Zone ladder: 4H tier ──
    zone_4h_current_upper: Optional[float] = None
    zone_4h_current_lower: Optional[float] = None
    zone_4h_outer_high: Optional[float] = None    # body-only max over window
    zone_4h_outer_low: Optional[float] = None     # body-only min over window
    zone_4h_extended: bool = False                # True = 180d/5-zone view

    # ── Zone ladder: 1D tier ──
    zone_1d_current_upper: Optional[float] = None
    zone_1d_current_lower: Optional[float] = None
    zone_1d_outer_high: Optional[float] = None
    zone_1d_outer_low: Optional[float] = None
    zone_1d_extended: bool = False

    # ── MA extension (mirrors ema_50_status/ema_50_reclassified above) ──
    ema_200_status: str = "UNTESTED"
    ema_200_reclassified: Optional[str] = None
    ema_50_status_1d: str = "UNTESTED"
    ema_50_reclassified_1d: Optional[str] = None
    ema_200_status_1d: str = "UNTESTED"
    ema_200_reclassified_1d: Optional[str] = None

    def sanitise(self) -> None:
        """
        NaN guard. Call after all fields populated, before any downstream read.
        Idempotent — safe to call multiple times.

        Two-path rule (MRS Phase 0B):
          - Count/additive fields: NaN -> 0.0
          - Ratio/proportion fields: NaN -> None + _valid = False
            (0.0 would mean the opposite of "no data" for ratio fields)
        """

        def _nan(v) -> bool:
            return isinstance(v, float) and math.isnan(v)

        # Count / additive — zero is a valid neutral value
        _count_fields = [
            "regime_age_hours",
            "regime_age_ratio",
            "defense_strength",
            "distance_zscore",
            "squeeze_strength",
            "effort_result_zscore",
            "divergence_strength",
            "rejection_strength",
            "distance_to_vwap_atr",
            "net_conviction",
            "spread_ratio",
            "time_since_last_loss_hours",
        ]
        for fname in _count_fields:
            v = getattr(self, fname, None)
            if _nan(v):
                setattr(self, fname, 0.0)

        # Ratio / proportion — NaN means no data, not zero
        if _nan(self.order_book_imbalance):
            self.order_book_imbalance = None
            self.order_book_imbalance_valid = False

        if _nan(self.body_trend_ratio):
            self.body_trend_ratio = None
            self.body_trend_ratio_valid = False

        if _nan(self.vol_down_ratio):
            self.vol_down_ratio = None
            self.vol_down_ratio_valid = False

        if _nan(self.bbw_percentile):
            self.bbw_percentile = None

        if _nan(self.range_high):
            self.range_high = None

        if _nan(self.range_low):
            self.range_low = None

    def to_dict(self) -> Dict:
        """For shadow trader logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
