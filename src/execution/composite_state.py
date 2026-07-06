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
    transition_type: Optional[str] = None
    regime_age_hours: float = 0.0
    regime_age_ratio: float = 0.0
    median_regime_duration: float = 12.0
    slopes_aligned: bool = False
    slope_diverging: bool = False
    structural_decay: bool = False
    is_friday_pm: bool = False

    # ══════════════════════════════════════
    # LAYER 2: STRUCTURAL REALITY
    # ══════════════════════════════════════
    choch_detected: bool = False
    bos_detected: bool = False
    nearby_4h_level:   Optional[float] = None
    nearby_4h_level_2: Optional[float] = None   # Second nearest 4H structural level
    nearby_4h_level_3: Optional[float] = None   # Third nearest 4H structural level
    nearby_4h_level_type: Optional[str] = None  # "swing_high" / "swing_low" — current role, re-evaluated every cycle by role reversal
    level_test_count: int = 0
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
    ema_20_status: str = "UNTESTED"
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
    friday_tighten: bool = False

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
