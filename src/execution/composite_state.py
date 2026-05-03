"""
Composite State — The bot's shared nervous system.
Every analysis module reads from and writes to this object.
The Confluence Engine reads the complete state to score trades.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class CompositeState:
    """Updated on every closed candle. Passed between all analysis layers."""

    # ══════════════════════════════════════
    # LAYER 1: MACRO STATE
    # ══════════════════════════════════════
    lifecycle_phase: str = "UNKNOWN"          # PICKUP / CONFIRMATION / ESTABLISHED / FADING / EXHAUSTION
    transition_type: Optional[str] = None     # e.g. "NEUTRAL→SLIGHTLY_BULL"
    regime_age_hours: float = 0.0
    regime_age_ratio: float = 0.0             # 1.0 = median duration for this asset
    median_regime_duration: float = 12.0      # Dynamic per asset
    slopes_aligned: bool = False              # 1H and 4H EMA slopes same direction
    slope_diverging: bool = False             # 1H fighting 4H
    structural_decay: bool = False            # Old regime + slopes fighting
    transition_probability: float = 0.5       # Markov: P(current regime continues)
    is_friday_pm: bool = False

    # ══════════════════════════════════════
    # LAYER 2: STRUCTURAL REALITY
    # ══════════════════════════════════════
    choch_detected: bool = False              # Change of Character (trend ending)
    bos_detected: bool = False                # Break of Structure (trend continuing)
    nearby_4h_level: Optional[float] = None   # From MTF Structure Memory
    level_test_count: int = 0                 # How many times this level tested
    level_defended: bool = False              # Rejection at structure level
    defense_strength: float = 0.0             # 0-1 scale
    is_parabolic: bool = False                # Price extended beyond dynamic threshold
    distance_zscore: float = 0.0              # Z-score of distance from 50 EMA
    squeeze_active: bool = False              # EMAs compressed
    squeeze_strength: float = 0.0             # How tight the squeeze is
    coiled_spring: bool = False               # Squeeze + Inside Bar
    reversal_imminent: bool = False           # Parabolic + divergence
    inside_bar: bool = False
    outside_bar: bool = False
    failed_breakout: bool = False

    # Layer 2b: MA Intelligence
    ema_20_status: str = "UNTESTED"           # DEFENDED / BROKEN / UNTESTED
    ema_50_status: str = "UNTESTED"
    ema_50_reclassified: Optional[str] = None # SUPPORT / RESISTANCE / LINE
    absorption_detected: bool = False         # High effort, low result at MA
    effort_result_zscore: float = 0.0

    # ══════════════════════════════════════
    # LAYER 3: ORDER FLOW
    # ══════════════════════════════════════
    cvd_trend: int = 0                        # +1 buyers, -1 sellers, 0 neutral (BTC only)
    cvd_stale: bool = True                    # True if WebSocket disconnected
    vpd_diverging: bool = False               # Volume dying on new extreme (BTC only)
    body_trend_ratio: float = 1.0             # <0.5 = bodies shrinking = conviction dying
    conviction_dying: bool = False
    divergence_detected: bool = False         # RSI divergence from MR strategy
    divergence_strength: float = 0.0          # 0-1 scale

    # ══════════════════════════════════════
    # LAYER 4: MICRO EXECUTION
    # ══════════════════════════════════════
    sweep_detected: bool = False              # PDH/PDL or Asian range swept
    sweep_direction: int = 0                  # +1 swept high, -1 swept low
    sweep_level: Optional[float] = None
    rejection_at_level: bool = False          # Strong wick at key level
    rejection_strength: float = 0.0
    session_name: str = "UNKNOWN"
    vwap_price: Optional[float] = None
    distance_to_vwap_atr: float = 0.0        # In ATR units
    time_since_last_loss_hours: float = 999.0

    # ══════════════════════════════════════
    # AI INTEGRATION
    # ══════════════════════════════════════
    ai_pattern_name: Optional[str] = None     # From Sniper CNN-LSTM
    ai_pattern_confidence: float = 0.0
    ai_reversal_probability: float = 0.0      # AI thinks reversal is likely

    # ══════════════════════════════════════
    # OUTPUT (set by Confluence Engine)
    # ══════════════════════════════════════
    institutional_pattern: Optional[str] = None  # ACCUMULATION / DISTRIBUTION / LIQUIDITY_HUNT / etc
    exhaustion_score: float = 0.0
    confirmation_score: float = 0.0
    net_conviction: float = 0.0
    friday_tighten: bool = False

    def to_dict(self) -> Dict:
        """For shadow trader logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
