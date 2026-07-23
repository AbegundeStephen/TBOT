"""
Veteran Trade Manager - Strategic/Tactical Risk Architecture
✨ REFACTORED: Centralized risk configuration from config.json.
📊 ROLE: Tactical execution engine (HOW to manage trades, not HOW MUCH to risk)
"""

import logging
import threading
import numpy as np
import talib
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons for tracking"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    MANUAL = "manual"
    TIME_STOP = "time_stop"
    EARLY_SCALE = "early_scale"
    # Smart market-condition exits
    VOLATILITY_SPIKE     = "volatility_spike"      # ATR explodes 2× → risk model invalid
    REVERSAL_CANDLE      = "reversal_candle"        # Strong engulfing bar against trade
    TREND_INVALIDATION   = "trend_invalidation"     # 3 bars against + ADX < 20
    MOMENTUM_EXHAUSTION  = "momentum_exhaustion"    # RSI extreme + MACD dying + ADX falling


def find_resistance_levels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    current_price: float,
    side: str,
    lookback: int = 50,
    min_touches: int = 2,
    tolerance: Optional[float] = None, # Use absolute price tolerance instead of %
) -> List[float]:
    """Find significant resistance/support levels using adaptive tolerance."""
    lookback = min(lookback, len(close))
    levels = []
    
    # Default tolerance if none provided (Fallback to 0.5% of price if ATR not provided)
    if tolerance is None:
        tolerance = current_price * 0.005

    if side == "long":
        highs = high[-lookback:]
        for i in range(2, len(highs) - 2):
            if highs[i] > current_price:
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    levels.append(highs[i])

        clustered = []
        for level in sorted(levels):
            if not clustered or (level - clustered[-1]) > tolerance:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for h in highs if abs(h - level) <= tolerance)
            if touches >= min_touches:
                verified.append(level)

        return sorted(verified)[:5]
    else:
        lows = low[-lookback:]
        for i in range(2, len(lows) - 2):
            if lows[i] < current_price:
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    levels.append(lows[i])

        clustered = []
        for level in sorted(levels, reverse=True):
            if not clustered or (clustered[-1] - level) > tolerance:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for l in lows if abs(l - level) <= tolerance)
            if touches >= min_touches:
                verified.append(level)

        return sorted(verified, reverse=True)[:5]


def calculate_hybrid_targets(
    entry_price: float,
    stop_loss: float,
    side: str,
    structure_levels: List[float],
    risk_multiples: List[float],
    partial_sizes: List[float],
    min_rr: float = 1.2,
) -> Tuple[List[float], List[float]]:
    """Calculate targets with structure awareness"""
    risk = abs(entry_price - stop_loss)
    targets = []
    adjusted_sizes = list(partial_sizes)

    logger.info(f"\n[VTM] Target Calculation:")
    logger.info(f"  Entry: ${entry_price:,.2f}")
    logger.info(f"  Stop:  ${stop_loss:,.2f}")
    logger.info(f"  Risk:  ${risk:,.2f} ({risk/entry_price:.2%})")

    if side == "long":
        for i, r_multiple in enumerate(risk_multiples):
            rr_target = entry_price + (risk * r_multiple)

            if r_multiple > 10:
                logger.warning(f"  ⚠️  TP{i+1}: {r_multiple}R exceeds 10R cap")
                continue

            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (closest - entry_price) / risk
                distance_pct = abs(closest - rr_target) / rr_target

                if structure_rr >= min_rr and distance_pct < 0.25:
                    targets.append(closest)
                    logger.info(f"  ✓ TP{i+1}: ${closest:,.2f} ({structure_rr:.1f}R) [Structure]")
                elif structure_rr < min_rr:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too close]")
                else:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too far]")
            else:
                targets.append(rr_target)
                logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R)")
    else:
        for i, r_multiple in enumerate(risk_multiples):
            rr_target = entry_price - (risk * r_multiple)

            if r_multiple > 10:
                continue

            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (entry_price - closest) / risk
                distance_pct = abs(closest - rr_target) / abs(rr_target)

                if structure_rr >= min_rr and distance_pct < 0.25:
                    targets.append(closest)
                    logger.info(f"  ✓ TP{i+1}: ${closest:,.2f} ({structure_rr:.1f}R) [Structure]")
                elif structure_rr < min_rr:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too close]")
                else:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too far]")
            else:
                targets.append(rr_target)
                logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R)")

    # ✅ M-3 FIX: Deduplicate targets that snapped to the same structure level.
    # When TP2 and TP3 both land on the nearest swing, 55% of the position
    # exits at the same price and the runner is never activated.
    if len(targets) > 1:
        seen_keys: set = set()
        unique_targets = []
        for _t in targets:
            _key = round(_t, 2)
            if _key not in seen_keys:
                seen_keys.add(_key)
                unique_targets.append(_t)
        if len(unique_targets) < len(targets):
            logger.info(
                f"  ℹ️  Deduplicated {len(targets) - len(unique_targets)} duplicate TP(s)"
            )
        targets = unique_targets

    if len(targets) < len(partial_sizes):
        logger.warning(f"  ⚠️  Only {len(targets)} targets (expected {len(partial_sizes)})")
        remaining = sum(partial_sizes[len(targets):])
        for i in range(len(targets)):
            adjusted_sizes[i] = partial_sizes[i] + (remaining / len(targets))
        adjusted_sizes = adjusted_sizes[:len(targets)]
        logger.info(f"  → Adjusted sizes: {[f'{s:.0%}' for s in adjusted_sizes]}")

    return targets, adjusted_sizes


class VeteranTradeManager:
    """
    ✨ REFACTORED: Strategic/Tactical Risk Architecture
    
    TACTICAL ROLE: Manages HOW to execute trades (stops, targets, trailing)
    STRATEGIC ROLE: Portfolio Manager decides HOW MUCH to risk
    
    KEY CHANGES:
    - Accepts risk configuration dictionary directly from config.json.
    - Validates trade economics before execution (pre-flight check).
    - Asymmetric constraints for TREND vs SCALP trades.
    """

    @classmethod
    def validate_trade_setup(
        cls,
        entry_price: float,
        stop_loss: float,
        risk_config: dict,
        trade_type: str = "TREND",
        atr_fast: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        ✨ INSTITUTIONAL: Strict TREND validation with ATR-based economics.
        """
        try:
            # Default atr_fast if not provided (Fallback to 1% of price if ATR not provided)
            if atr_fast is None:
                atr_fast = entry_price * 0.01

            # ATR-based adaptive cap (Replacing static max_stop_pct)
            max_stop_atr_mult = risk_config.get("max_stop_atr_mult", 5.0) if risk_config else 5.0
            max_stop_dist = atr_fast * max_stop_atr_mult
            min_rr = risk_config.get("min_rr", 1.5) if risk_config else 1.5

            stop_distance = abs(entry_price - stop_loss)

            if stop_distance > max_stop_dist:
                return False, f"Stop too wide: ${stop_distance:,.2f} > {max_stop_atr_mult}x ATR (${max_stop_dist:,.2f})"

            risk_multiples = risk_config.get("partial_targets", [1.0, 1.8, 3.0])
            partial_sizes  = risk_config.get("partial_sizes",   [0.45, 0.30, 0.25])
            if not risk_multiples:
                risk_multiples = [1.0, 1.8, 3.0]
            if not partial_sizes:
                partial_sizes = [1.0 / len(risk_multiples)] * len(risk_multiples)

            # ── Closest-target check (TP1 must clear half an ATR minimum) ──
            first_tp_dist = stop_distance * risk_multiples[0]
            if first_tp_dist < (0.5 * atr_fast):
                return False, (
                    f"TP1 too close to entry: ${first_tp_dist:,.2f} < 0.5×ATR (${0.5*atr_fast:,.2f})"
                )

            # ── Weighted R:R across ALL partial exits ──────────────────────
            # Using only TP1 as the R:R measure is wrong for partial-exit systems:
            # with TP1 at 1.0R (deliberate close first exit), the check always
            # fires even though weighted R:R across [1.0, 1.8, 3.0] is ~1.74.
            n = min(len(risk_multiples), len(partial_sizes))
            total_weight = sum(partial_sizes[:n])
            if total_weight > 0:
                weighted_rr = sum(
                    risk_multiples[i] * partial_sizes[i]
                    for i in range(n)
                ) / total_weight
            else:
                weighted_rr = risk_multiples[0]

            if weighted_rr < min_rr - 1e-9:
                return False, (
                    f"Weighted R:R too low: {weighted_rr:.2f}:1 < {min_rr:.2f}:1 "
                    f"(targets={risk_multiples}, sizes={partial_sizes})"
                )

            logger.info(
                f"[VTM PRE-FLIGHT] ✅ Trade Valid\n"
                f"  Type:        TREND\n"
                f"  Stop:        ${stop_distance:,.2f} ({(stop_distance/entry_price):.2%})\n"
                f"  TP1 dist:    ${first_tp_dist:,.2f} ({(first_tp_dist/entry_price):.2%})\n"
                f"  Weighted R:R:{weighted_rr:.2f}:1  (targets={risk_multiples})"
            )

            return True, "OK"

        except Exception as e:
            logger.error(f"[VTM PRE-FLIGHT] Error: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def compute_effective_atr_multiplier(
        trade_type: str,
        config_base: float,
        regime: str = "NEUTRAL",
        volatility_regime: str = "normal",
    ) -> float:
        """
        Phase 1.1: the regime-adaptive ATR multiplier the VTM uses for its
        initial stop. Extracted so the order handlers can size against the
        EXACT stop distance the VTM will place (single source of truth).

        REVERSION              → max(config_base, 2.0)
        TREND (bear/high-vol)  → max(config_base + 0.5, 2.5)
        TREND (normal/bull)    → max(config_base + 0.3, 2.0)
        other                  → config_base
        """
        tt = (trade_type or "TREND").upper()
        if tt == "REVERSION":
            return max(config_base, 2.0)
        if tt == "TREND":
            if "BEAR" in (regime or "").upper() or volatility_regime == "high":
                return max(config_base + 0.5, 2.5)
            return max(config_base + 0.3, 2.0)
        return config_base

    def __init__(
        self,
        entry_price: float,
        side: str,
        asset: str, # Asset key still needed for logging
        risk_config: dict,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        quantity: float,
        volume: Optional[np.ndarray] = None,
        signal_details: Dict = None,
        account_risk: float = 0.015,
        atr_period: int = 14,
        trade_type: str = "TREND",
        local_free_margin: float = 0.0, # ✨ NEW: For leverage ceiling
        current_ask: float = 0.0,       # ✨ NEW: For spread floor
        current_bid: float = 0.0,       # ✨ NEW: For spread floor
        min_lot_override: Optional[float] = None,      # ✨ NEW: Exness compatibility
        lot_precision_override: Optional[int] = None,  # ✨ NEW: Exness compatibility
        structure_levels_ref: Optional[list] = None,   # Item 5: level traded against
        entry_retest_type: Optional[str] = None,       # Item 5: RetestEngine tier at entry
        telegram=None,  # Brain rebuild Part 0.3
        council_ref=None,  # Gate Tier 4.1 — reuses _check_lifecycle_phase in the alert layer
    ):
        self.entry_price = entry_price
        self.side = side.lower()
        self.asset = asset.upper()
        self.risk_config = risk_config
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.atr_period = atr_period
        self.signal_details = signal_details or {}
        self.trade_type = trade_type
        
        self.position_size = quantity
        self.local_free_margin = local_free_margin
        self.current_ask = current_ask
        self.current_bid = current_bid
        self.min_lot_override = min_lot_override
        self.lot_precision_override = lot_precision_override

        # Item 5: structure-aware plumbing — shared prerequisite for this
        # batch and the brain rebuild's later tiers. Nothing reads these yet;
        # this just makes sure they exist so later work isn't blocked on it.
        self.structure_levels_ref = structure_levels_ref
        self.entry_retest_type = entry_retest_type
        self.telegram = telegram  # Brain rebuild Part 0.3 — used by 3.5's alert layer
        self.council_ref = council_ref  # Gate Tier 4.1 — used by the alert layer's lifecycle-phase reuse
        self.atr_at_entry = self._calculate_atr()

        # Determine asset type for leverage ceiling
        self.asset_category = "FOREX"
        crypto_keywords = ["BTC", "ETH", "SOL", "BNB", "XRP", "USDT"]
        if any(k in self.asset for k in crypto_keywords):
            self.asset_category = "CRYPTO"

        # Macro MAs from signal_details (for MA Shield/Front-run)
        gov_data = self.signal_details.get("governor_data", {})
        self.ema_1d_200 = gov_data.get("ema_1d_200")
        self.ema_4h_200 = gov_data.get("ema_4h_200")
        self.ema_4h_50 = gov_data.get("ema_4h_50")

        # ✅ TASK 18: Regime-Adaptive ATR Multipliers
        # config_base is the per-asset value from config.json (e.g. 1.5 for GBPUSD).
        # Regime logic adds on top rather than fully overriding — so raising atr_multiplier
        # in config.json actually widens the SL rather than being silently ignored.
        config_base = self.risk_config.get("atr_multiplier", 1.8)

        # Phase 1.1: single source of truth for the effective ATR multiplier so
        # position sizing (in the handlers) uses the SAME stop distance the VTM
        # actually applies here. Previously sizing used config_base directly
        # while the VTM widened it via the regime floors below, so realized risk
        # ran 30–60% over the intended risk_pct.
        self.atr_multiplier = self.compute_effective_atr_multiplier(
            trade_type=self.trade_type,
            config_base=config_base,
            regime=self.signal_details.get("regime", "NEUTRAL"),
            volatility_regime=self.signal_details.get("volatility_regime", "normal"),
        )

        logger.info(
            f"[VTM] ATR Multiplier: config={config_base}× → effective={self.atr_multiplier}× "
            f"({self.trade_type}, regime={self.signal_details.get('regime','NEUTRAL')})"
        )

        # T2.5: ADX-conditioned take profit targets.
        # Static targets fail in chop (ADX<20) where price never reaches them,
        # causing the 43% profit capture rate. Scale down in chop, up in momentum.
        base_targets = self.risk_config.get("partial_targets", [1.0, 1.8, 3.0])
        try:
            if len(close) >= 14:
                import talib as _talib
                adx_val = _talib.ADX(high, low, close, timeperiod=14)[-1]
                if not np.isnan(adx_val):
                    if adx_val < 20:
                        self.partial_targets = [max(1.0, t * 0.7) for t in base_targets]
                        logger.info(
                            f"[VTM] 📉 Chop mode (ADX={adx_val:.0f}): "
                            f"Targets scaled to {self.partial_targets}"
                        )
                    elif adx_val > 40:
                        self.partial_targets = [t * 1.3 for t in base_targets]
                        logger.info(
                            f"[VTM] 📈 Momentum mode (ADX={adx_val:.0f}): "
                            f"Targets scaled to {self.partial_targets}"
                        )
                    else:
                        self.partial_targets = list(base_targets)
                else:
                    self.partial_targets = list(base_targets)
            else:
                self.partial_targets = list(base_targets)
        except Exception as _e:
            logger.debug(f"[VTM] ADX target scaling failed ({_e}), using base targets")
            self.partial_targets = list(base_targets)

        self.partial_sizes = self.risk_config.get("partial_sizes", [0.45, 0.30, 0.25])

        # ── J: Pattern-Aware Exit Management ──────────────────────────────
        _pattern = self.signal_details.get("institutional_pattern")
        if _pattern == "LIQUIDITY_HUNT":
            # Snap-back reversals die fast — tighten everything
            self.partial_targets = [max(0.8, t * 0.6) for t in self.partial_targets]
            self.breakeven_profit_threshold = self.risk_config.get(
                "breakeven_profit_threshold", 0.01) * 0.5
            logger.info(f"[VTM] LIQUIDITY_HUNT pattern: Tight TPs {self.partial_targets}")

        elif _pattern == "ACCUMULATION":
            # Generational trend — give it room to breathe
            self.partial_targets = [t * 1.3 for t in self.partial_targets]
            self.atr_multiplier *= 1.2
            logger.info(f"[VTM] ACCUMULATION pattern: Wide targets {self.partial_targets}")

        elif _pattern == "SPRING_BREAKOUT":
            # Explosive but uncertain — take first TP early, let runner go
            self.partial_sizes = [0.60, 0.25, 0.15]  # Take 60% at TP1
            logger.info(f"[VTM] SPRING_BREAKOUT: Heavy first partial")
        # ──────────────────────────────────────────────────────────────────

        self.pivot_lookback = self.risk_config.get("pivot_lookback", 30)
        self.time_stop_bars = self.risk_config.get(
            f'time_stop_{self.trade_type.lower()}',
            self.risk_config.get(
                'time_stop_bars',
                self.risk_config.get('max_hold_bars', 72)  # max_hold_bars is legacy alias
            )
        )
        self.use_ema_structure = self.risk_config.get("use_ema_structure", False)
        self.use_structure_targets = self.risk_config.get("use_structure_targets", True)
        # Runner trailing stop ATR multiplier (replaces hardcoded 2.0)
        self.runner_trail_atr_multiplier = self.risk_config.get("runner_trail_atr_multiplier", 2.0)
        # Time-based break-even: move SL to entry after N bars if pnl >= threshold
        self.breakeven_after_bars = self.risk_config.get("breakeven_after_bars", None)
        self.breakeven_profit_threshold = self.risk_config.get("breakeven_profit_threshold", 0.01)

        # ── Phase 4: Livermore Awareness + Structural Stop Routing ────────────
        # All fields sourced from composite_state written by signal_aggregator.
        # entry_type was set by the Phase 3B retest_engine.
        _cs = (signal_details or {}).get("composite_state", {})
        self.livermore_state_4h         = _cs.get("livermore_state_4h")
        self.livermore_state_age_4h     = int(_cs.get("livermore_state_age_4h",  0))
        # Brain Rebuild Part 3.3 (D4): 1H companion to the 4H state above —
        # used by _apply_livermore_transition's thesis-confirmation check.
        self.livermore_state_1h         = _cs.get("livermore_state_1h")
        self.livermore_state_age_1h     = int(_cs.get("livermore_state_age_1h", 0))
        self.livermore_anchor_main_up   = _cs.get("livermore_anchor_main_up_max")
        self.livermore_anchor_main_down = _cs.get("livermore_anchor_main_down_min")
        self.livermore_anchor_nat_high  = _cs.get("livermore_anchor_natural_high")
        self.livermore_anchor_nat_low   = _cs.get("livermore_anchor_natural_low")
        self.nearby_4h_level            = _cs.get("nearby_4h_level")
        self.level_defended             = bool(_cs.get("level_defended",  False))
        self.sweep_level                = _cs.get("sweep_level")
        # entry_type: MR_PULLBACK / TREND_FOLLOWING / SPRING_ENTRY /
        #             RANGE_BOUNDARY / CONTINUATION / REJECT / None (no classification)
        self.vtm_entry_type             = _cs.get("entry_type")

        # 7.4: Populate _mfe_target_r from real trade history once enough
        # exists. Stays None (safely skipped by _blend_single_tp, see the
        # getattr fallback at its call site) until then. NOTE: this is inert
        # until something populates signal_details["db_manager_ref"] — no
        # call site does that yet (Position has no db_manager reference of
        # its own; only PortfolioManager does). Wiring that through is the
        # deferred "Fix L1" work, intentionally not done here.
        self._mfe_target_r = None
        _db = self.signal_details.get("db_manager_ref")
        if _db is not None:
            try:
                self._mfe_target_r = _db.get_historical_mfe_r(self.asset, min_samples=30)
            except Exception:
                pass
        # Target-ladder box (Phase 4 ext, gated by phase_config.continuation_targets_enabled).
        # Populated only for RANGE_BOUNDARY/SPRING_ENTRY — None otherwise.
        self.vtm_range_high             = _cs.get("range_high")
        self.vtm_range_low              = _cs.get("range_low")
        # Q3: zone ladder's nearest tested 4H lines (Q2a/Q2b). Read the same
        # way as everything else off _cs — VTM's own structural SL/TP source.
        self.zone_current_upper         = _cs.get("zone_4h_current_upper")
        self.zone_current_lower         = _cs.get("zone_4h_current_lower")
        # Squeeze flag — drives wider ATR selection in _calculate_atr()
        self.bb_kc_squeeze_active       = bool(_cs.get("bb_kc_squeeze_active", False))
        # Fix 1: merge runtime phase_config from CompositeState (overrides static config block)
        _cs_phase_cfg = _cs.get("phase_config", {})
        if _cs_phase_cfg:
            self.risk_config = {**self.risk_config, "phase_config": {
                **self.risk_config.get("phase_config", {}), **_cs_phase_cfg
            }}

        # Livermore-aware ATR multiplier overlay — all values from config
        _lv4h   = self.livermore_state_4h
        _lv_adj = self.risk_config.get("livermore_atr_adjustments", {})
        _main_stop_add    = _lv_adj.get("main_stop_add",         0.3)
        _sec_stop_sub     = _lv_adj.get("secondary_stop_sub",    0.2)
        _main_trail_mult  = _lv_adj.get("main_trail_mult",       1.3)
        _sec_trail_mult   = _lv_adj.get("secondary_trail_mult",  0.7)
        _be_young         = _lv_adj.get("breakeven_atr_young_state", 1.8)
        _be_mid           = _lv_adj.get("breakeven_atr_mid_state",   1.5)
        _be_old           = _lv_adj.get("breakeven_atr_old_state",   1.2)
        _young_max_age    = _lv_adj.get("young_state_max_age",   5)
        _old_min_age      = _lv_adj.get("old_state_min_age",    20)

        if _lv4h in ("MAIN_UP", "MAIN_DOWN"):
            self.atr_multiplier += _main_stop_add
            logger.info(
                f"[VTM] Livermore={_lv4h}: ATR mult +{_main_stop_add} → {self.atr_multiplier:.2f}×"
            )
        elif _lv4h in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
            self.atr_multiplier = max(1.5, self.atr_multiplier - _sec_stop_sub)
            logger.info(
                f"[VTM] Livermore={_lv4h}: ATR mult −{_sec_stop_sub} → {self.atr_multiplier:.2f}×"
            )

        if _lv4h in ("MAIN_UP", "MAIN_DOWN"):
            self.runner_trail_atr_multiplier *= _main_trail_mult
            logger.info(
                f"[VTM] Livermore={_lv4h}: trail mult ×{_main_trail_mult} → "
                f"{self.runner_trail_atr_multiplier:.2f}×"
            )
        elif _lv4h in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
            self.runner_trail_atr_multiplier *= _sec_trail_mult
            logger.info(
                f"[VTM] Livermore={_lv4h}: trail mult ×{_sec_trail_mult} → "
                f"{self.runner_trail_atr_multiplier:.2f}×"
            )

        # Livermore-aware break-even ATR trigger — values from config
        _age4h = self.livermore_state_age_4h
        if _age4h < _young_max_age:
            self.breakeven_atr_trigger = _be_young
        elif _age4h > _old_min_age:
            self.breakeven_atr_trigger = _be_old
        else:
            self.breakeven_atr_trigger = _be_mid
        # ─────────────────────────────────────────────────────────────────────

        # Early Scale: lock in a small partial exit within the first N bars
        # S6.3: phase-level locks (triple-lock with _can_partial / _can_add_position)
        self.partials_enabled   = self.risk_config.get("phase_config", {}).get("partials_enabled",   False)
        self.pyramiding_enabled = self.risk_config.get("phase_config", {}).get("pyramiding_enabled", False)
        self.early_scale_enabled = self.risk_config.get("early_scale_enabled", False)
        self.early_scale_threshold = self.risk_config.get("early_scale_threshold", 0.02)
        self.early_scale_bars = self.risk_config.get("early_scale_bars", 4)
        self.early_lock_atr_multiplier = self.risk_config.get("early_lock_atr_multiplier", 0.5)
        self._early_scaled = False          # fires at most once per trade
        self._greed_mode_activated = False  # fires at most once per trade
        self._time_stop_extended = False    # fires at most once per trade

        # A13: market_watcher.py runs on its own 15s-poll daemon thread and
        # writes current_stop_loss directly (via _push_sl) outside this
        # object's own tick loop (check_exit, called from the single-
        # threaded main loop). Shared lock so the two can't interleave a
        # read-modify-write on the same SL — market_watcher acquires this
        # same lock before its write (see market_watcher.py's _push_sl).
        self._sl_lock = threading.Lock()

        # State
        self.initial_stop_loss = None
        self.current_stop_loss = None
        self.take_profit_levels = []
        self.remaining_position = 1.0
        self.partials_hit = []
        self.bars_in_trade = 0
        self.highest_price_reached = entry_price
        self.lowest_price_reached = entry_price
        self.runner_activated = False
        self.stop_type = "atr"
        self.has_pyramided = False # ✨ NEW: Trend Pyramiding Flag
        self.entry_time = datetime.now()
        # Item 2.14: set by _compute_structural_stop when the trade has no
        # structural stop reference AND wasn't strong enough to justify a
        # plain distance-based stop. The caller (PortfolioManager, right
        # after registering this position) checks this flag and immediately
        # closes the position — VTM manages already-open positions, it can't
        # prevent the entry itself at this point.
        self.emergency_close_requested = False

        # Calculate initial SL/TP levels (must be last step of __init__)
        try:
            self._calculate_initial_levels()
        except Exception as e:
            logger.error(f"[VTM] Initialization error: {e}")
            raise

        # Item 5: captured here rather than earlier alongside
        # structure_levels_ref/entry_retest_type above — self.current_stop_loss
        # is only a None placeholder until _calculate_initial_levels() runs;
        # capturing it before that would always store None instead of the
        # real level this trade was actually stopped against.
        self.structural_stop_level = self.current_stop_loss

        # Log initialization summary
        logger.info("=" * 80)
        logger.info(f"🎯 VTM - {self.asset} {side.upper()} [{self.trade_type}]")
        logger.info("=" * 80)
        logger.info(f"Entry:    ${entry_price:,.2f}")
        logger.info(f"Stop:     ${self.initial_stop_loss:,.2f} (-{self._calc_pct_distance(entry_price, self.initial_stop_loss):.2f}%)")
        logger.info(f"Quantity: {self.position_size:.6f} units")
        logger.info(f"\n📊 TARGETS:")
        if not self.take_profit_levels or not self.partial_sizes:
            logger.info("  No take profit targets calculated or partial sizes defined.")
        for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes), 1):
            target_str = f"${target:,.2f}" if target is not None else "N/A"
            size_str = f"{size:.0%}" if size is not None else "N/A"
            if target is not None:
                pct = self._calc_pct_distance(entry_price, target)
                pct_str = f"(+{pct:.2f}%)"
            else:
                pct_str = ""
            logger.info(f"  {i}. {target_str} {pct_str} → Exit {size_str}")
        logger.info("=" * 80)

    def __getstate__(self):
        """
        Custom method for pickling. Mirrors Position.__getstate__: self.telegram
        (added Brain Rebuild Part 0.3) is a live reference to the running
        TradingTelegramBot, which back-references TradingBot itself — pickling
        it walks the entire live object graph (threads, sockets, Supabase
        client internals, ...) on every state save, none of which survives a
        restart anyway. Every VTM construction path (fresh open, state
        reload) already passes telegram=self.telegram_bot explicitly, so
        this is safe to drop and re-attach rather than pickle.
        """
        state = self.__dict__.copy()
        if "telegram" in state:
            del state["telegram"]
        # Gate Tier 4.1: council_ref is a live aggregator reference — same
        # pickling hazard as telegram (back-references TradingBot). Every VTM
        # construction path passes council_ref explicitly, so drop and
        # re-attach rather than pickle.
        if "council_ref" in state:
            del state["council_ref"]
        # _sl_lock is a raw threading.Lock() (A13) — not picklable at all.
        # Left in here it either crashes the save outright or, once already
        # None from a prior bad round-trip, silently pickles as None and
        # every with self._sl_lock: call downstream raises TypeError forever.
        if "_sl_lock" in state:
            del state["_sl_lock"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling. Re-initializes telegram as None —
        re-attached by whichever VTM-construction path picks this position
        back up (state reload passes a live telegram_bot explicitly).
        _sl_lock is rebuilt fresh — a restored lock can't preserve any
        cross-thread waiters anyway, and every construction path needs a
        real Lock, not None, before market_watcher.py or the VTM tick loop
        can safely enter their `with self._sl_lock:` sections."""
        self.__dict__.update(state)
        self.telegram = None
        self.council_ref = None
        self._sl_lock = threading.Lock()

    # ── S6.3: Lot geometry helpers (mirror the Lot Sanitizer at _calculate_initial_levels) ──
    def _lot_geom(self):
        """Return (precision, min_lot) for this asset — single source of truth."""
        precision = self.lot_precision_override if self.lot_precision_override is not None \
            else {'BTC': 4, 'GOLD': 2, 'USTEC': 2, 'EURJPY': 2, 'EURUSD': 2,
                  'GBPUSD': 2, 'USDJPY': 2, 'USOIL': 2, 'GBPAUD': 2}.get(self.asset.upper(), 2)
        raw_min = self.min_lot_override if (self.min_lot_override is not None and self.min_lot_override > 0) \
            else {'BTC': 0.003, 'GOLD': 0.01, 'USTEC': 0.02, 'EURJPY': 0.02, 'EURUSD': 0.01,
                  'GBPUSD': 0.01, 'USDJPY': 0.01, 'USOIL': 0.01, 'GBPAUD': 0.01}.get(self.asset.upper(), 0.01)
        return precision, raw_min

    def _can_partial(self, fraction: float) -> bool:
        """True only if peeling `fraction` of position rounds to >= min_lot AND leaves
        a legal remainder. On min-lot accounts no fraction < 1.0 passes → partials
        no-op until the account can subdivide. Full closes never call this.
        Fails closed on any doubt."""
        try:
            precision, min_lot = self._lot_geom()
            total = round(self.position_size, precision)
            if total <= 0:
                return False
            peel = round(total * fraction, precision)
            rem  = round(total - peel, precision)
            return peel >= min_lot - 1e-9 and rem >= min_lot - 1e-9
        except Exception:
            return False

    def _can_add_position(self) -> bool:
        """True only when pyramiding is enabled AND half the current position is still
        a legal lot AND (if the Livermore maturity gate is enabled) the live 4H
        Livermore state isn't in a late/exhaustion-prone phase. All conditions must
        hold — the flag alone is not enough.

        L11: phase_config.pyramid_livermore_maturity_gate_enabled defaults to True
        (unlike most new phase_config flags in this project, which default False).
        Reason: pyramiding itself is already live (phase_config.pyramiding_enabled
        =true in the running config) — this gate only *tightens* an already-active
        behavior by blocking the worst-timed adds; it never loosens anything, so
        there is no new risk to gate OFF by default. It can still be set false
        independently if it misbehaves.
        """
        if not getattr(self, "pyramiding_enabled", False):
            return False
        try:
            precision, min_lot = self._lot_geom()
            if round(self.position_size * 0.5, precision) < min_lot - 1e-9:
                return False
        except Exception:
            return False

        try:
            _phase_cfg = self.risk_config.get("phase_config", {}) or {}
            if not _phase_cfg.get("pyramid_livermore_maturity_gate_enabled", True):
                return True

            lv_state = getattr(self, "livermore_state_4h", None)
            if not lv_state:
                # No live Livermore read available yet (per_tick_livermore_enabled
                # off, or composite_state never supplied at entry) — fail open so
                # this new gate can't silently disable pyramiding everywhere.
                return True

            # Block adds during the late/violent corrective phases — this is the
            # single riskiest moment to be scaling exposure into a trend.
            if lv_state in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
                logger.info(
                    f"[VTM] 🚫 Pyramid maturity gate: blocked — Livermore state "
                    f"{lv_state} is a late-cycle corrective phase."
                )
                return False

            # Block adds into a trend that's already very old — most of the
            # statistically expected move is likely behind it by this point.
            max_age = _phase_cfg.get("pyramid_max_state_age_4h", 20)
            age = int(getattr(self, "livermore_state_age_4h", 0) or 0)
            if age > max_age:
                logger.info(
                    f"[VTM] 🚫 Pyramid maturity gate: blocked — Livermore state "
                    f"{lv_state} age={age} bars exceeds max {max_age}."
                )
                return False

            return True
        except Exception as e:
            logger.debug(f"[VTM] L11 pyramid maturity gate error, failing open: {e}")
            return True

    @property
    def profit_locked(self) -> bool:
        """Checks if stop loss is at break-even or better"""
        if self.side == "long":
            return self.current_stop_loss >= self.entry_price
        else:
            return self.current_stop_loss <= self.entry_price

    @property
    def current_take_profit(self) -> Optional[float]:
        """Returns the next active take profit target"""
        idx = len(self.partials_hit)
        if idx < len(self.take_profit_levels):
            return self.take_profit_levels[idx]
        return None

    def _calc_pct_distance(self, price1: float, price2: float) -> float:
        return abs(price1 - price2) / price1 * 100

    def _calculate_atr(self) -> float:
        """
        Regime-adaptive ATR: fast in expanding vol, slow in compressed vol.
        Squeeze-aware: when BB/KC squeeze is active, use ATR(50) so the stop
        reflects pre-squeeze volatility rather than the compressed current range.
        This prevents stops being placed inside the squeeze range and getting
        hit on the first bar of the expected expansion.
        """
        try:
            atr_fast  = talib.ATR(self.high, self.low, self.close, timeperiod=7)[-1]
            atr_mid   = talib.ATR(self.high, self.low, self.close, timeperiod=14)[-1]
            atr_slow  = talib.ATR(self.high, self.low, self.close, timeperiod=28)[-1]

            if np.isnan(atr_mid) or atr_slow == 0:
                return self.entry_price * 0.015

            # Squeeze-aware path: ATR(50) captures pre-squeeze range
            if getattr(self, "bb_kc_squeeze_active", False):
                if len(self.close) >= 55:
                    atr_squeeze = talib.ATR(
                        self.high, self.low, self.close, timeperiod=50
                    )[-1]
                    if not np.isnan(atr_squeeze) and atr_squeeze > atr_mid:
                        logger.info(
                            f"[VTM] Squeeze-aware ATR: ATR(50)={atr_squeeze:.5f} "
                            f"replaces ATR(14)={atr_mid:.5f} "
                            f"(ratio {atr_squeeze/atr_mid:.2f}×)"
                        )
                        return atr_squeeze

            ratio = atr_fast / atr_slow

            if ratio > 1.30:
                selected_atr = atr_fast
                reason = "Expanding Vol (Tighten)"
            elif ratio < 0.70:
                selected_atr = atr_slow
                reason = "Compressed Vol (Wide)"
            else:
                selected_atr = atr_mid
                reason = "Normal Vol"

            logger.debug(f"[VTM] Dynamic ATR Selection: {selected_atr:.5f} ({reason}, Ratio: {ratio:.2f})")
            return selected_atr

        except Exception as e:
            logger.error(f"[VTM] ATR error: {e}")
            return self.entry_price * 0.02
        
    def check_promotion_to_runner(
        self, 
        current_price: float
    ) -> bool:
        if len(self.partials_hit) != 1 or self.runner_activated:
            return False
        
        try:
            volume_ratio = 1.0
            if self.volume is not None and len(self.volume) > 20:
                avg_vol = np.mean(self.volume[-21:-1]) 
                current_vol = self.volume[-1]
                if avg_vol > 0:
                    volume_ratio = current_vol / avg_vol
            
            volume_strong = volume_ratio > 1.5
            
            candle_conviction = False
            if len(self.high) > 0 and len(self.low) > 0:
                latest_high, latest_low = self.high[-1], self.low[-1]
                candle_range = latest_high - latest_low
                if candle_range > 0:
                    if self.side == "long":
                        distance_from_high = (latest_high - current_price) / candle_range
                        candle_conviction = distance_from_high < 0.20
                    else:
                        distance_from_low = (current_price - latest_low) / candle_range
                        candle_conviction = distance_from_low < 0.20
            
            if volume_strong or candle_conviction:
                logger.info("\n" + "=" * 70 + "\n🚀 TRADE PROMOTION TRIGGERED\n" + "=" * 70)
                self.runner_activated = True
                self.take_profit_levels, self.partial_sizes = [], []
                # Keep current structural stop loss, do not force break-even
                logger.info(f"[VTM] Runner activated. SL remains at structural level: ${self.current_stop_loss:,.2f}")
                return True
            else:
                # Do not modify SL if promotion fails
                return False
        
        except Exception as e:
            logger.error(f"[VTM] Promotion check error: {e}")
            return False

    def _calculate_initial_levels(self):
        try:
            atr = self._calculate_atr()

            # STEP 1 — Venue Adaptive Leverage Ceiling
            # Reason: Prevents over-exposure based on venue-specific risk rules.
            if self.local_free_margin > 0:
                notional_value = self.position_size * self.entry_price
                max_notional = 0.0
                
                if self.asset_category == "CRYPTO":
                    max_notional = self.local_free_margin * 3.0
                elif self.asset_category == "FOREX":
                    max_notional = self.local_free_margin * 20.0
                
                if notional_value > max_notional and max_notional > 0:
                    logger.info(f"[VTM] ⚠️ Leverage Ceiling: Notional ${notional_value:,.2f} > Max ${max_notional:,.2f}. Scaling down.")
                    self.position_size = max_notional / self.entry_price

            if self.trade_type == "REVERSION":
                wick_buffer = 0.5 * atr

                if self.side == "long":
                    # Q3: SL = tested zone line BELOW price, wick-buffered.
                    # Falls back to the bar low if no zone line (check 1),
                    # then the existing ATR clamp handles bad geometry (check 2).
                    _sl_line = self.zone_current_lower if self.zone_current_lower else self.low[-1]
                    self.initial_stop_loss = _sl_line - wick_buffer

                    # Q3: TP = tested zone line ABOVE price (the next zone).
                    # This is where "TP at the next zone" reaches the live order.
                    _tp_line = self.zone_current_upper
                    tp_target = _tp_line if _tp_line else (
                        self.ema_4h_50 - (0.2 * atr) if self.ema_4h_50 else self.entry_price + (2.0 * atr)
                    )

                    # Guards unchanged — they ARE the fallback's second check.
                    if self.initial_stop_loss >= self.entry_price:
                        self.initial_stop_loss = self.entry_price - (1.5 * atr)
                        logger.warning(
                            f"[VTM] REVERSION LONG: SL ({self.initial_stop_loss:.5f}) "
                            f">= entry — clamped to entry - 1.5×ATR"
                        )
                    if tp_target <= self.entry_price:
                        tp_target = self.entry_price + (2.0 * atr)
                        logger.warning("[VTM] REVERSION LONG: TP was <= entry — clamped to entry + 2×ATR")

                else:
                    # Q3: SL = tested zone line ABOVE price, wick-buffered.
                    # Falls back to the bar high if no zone line (check 1),
                    # then the existing ATR clamp handles bad geometry (check 2).
                    _sl_line = self.zone_current_upper if self.zone_current_upper else self.high[-1]
                    self.initial_stop_loss = _sl_line + wick_buffer

                    # Q3: TP = tested zone line BELOW price (the next zone).
                    _tp_line = self.zone_current_lower
                    tp_target = _tp_line if _tp_line else (
                        self.ema_4h_50 - (0.2 * atr) if self.ema_4h_50 else self.entry_price - (2.0 * atr)
                    )

                    # Guards unchanged — they ARE the fallback's second check.
                    if self.initial_stop_loss <= self.entry_price:
                        self.initial_stop_loss = self.entry_price + (1.5 * atr)
                        logger.warning(
                            f"[VTM] REVERSION SHORT: SL ({self.initial_stop_loss:.5f}) "
                            f"<= entry — clamped to entry + 1.5×ATR"
                        )
                    if tp_target >= self.entry_price:
                        tp_target = self.entry_price - (2.0 * atr)
                        logger.warning("[VTM] REVERSION SHORT: TP was >= entry — clamped to entry - 2×ATR")

                self.current_stop_loss = self.initial_stop_loss
                self.take_profit_levels = [tp_target]
                self.partial_sizes = [1.0]

                # Q3 check 2: if the zone-derived geometry doesn't clear the
                # per-asset R:R floor, the zones are too tight for this trade —
                # fall back to ATR geometry rather than take a sub-threshold R:R.
                _min_rr = float(self.risk_config.get("min_rr", 1.5))
                _risk = abs(self.entry_price - self.initial_stop_loss)
                _reward = abs(tp_target - self.entry_price)
                if _risk > 0 and (_reward / _risk) < _min_rr:
                    _atr_tp = self.partial_targets[0] if self.partial_targets else 2.0
                    if self.side == "long":
                        tp_target = self.entry_price + (_atr_tp * atr)
                    else:
                        tp_target = self.entry_price - (_atr_tp * atr)
                    self.take_profit_levels = [tp_target]
                    logger.info(
                        f"[VTM] REVERSION: zone R:R {_reward/_risk:.2f} < {_min_rr} — "
                        f"TP fell back to ATR ({_atr_tp}×)"
                    )

                logger.info(f"[VTM] REVERSION MODE: SL={self.initial_stop_loss}, TP={tp_target}")

            else:
                # ATR-based adaptive floors and caps.
                # min_stop_atr_mult is configurable per asset (default 0.8 for
                # FX/commodities, 0.5 for crypto). Prevents structural stops that
                # land within noise distance in quiet-hour low-ATR conditions.
                _min_mult = self.risk_config.get("min_stop_atr_mult", 0.8)
                min_stop_dist = atr * _min_mult
                max_stop_dist = atr * 5.0

                if self.side == "long":
                    # 1. Standard ATR Baseline
                    target_stop_dist = atr * self.atr_multiplier
                    standard_sl = self.entry_price - target_stop_dist
                    final_sl = standard_sl

                    # 2. Joint Synergy: MA Shield (only active when use_ema_structure=true)
                    # Q3: zone_current_lower joins the MA candidates — it's a tested
                    # structural line same as the MAs, so it competes on equal terms
                    # rather than overriding or short-circuiting the existing shield.
                    if self.use_ema_structure:
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50, self.zone_current_lower]:
                            if ma and standard_sl < ma < self.entry_price:
                                buffered_ma_sl = ma - (0.5 * atr)
                                if buffered_ma_sl > final_sl:
                                    logger.info(f"[VTM] 🛡️ MA Shield Jointly Applied: SL tucked behind MA ${ma:,.2f}")
                                    final_sl = buffered_ma_sl

                    # 3. Apply global clamps
                    final_sl = max(
                        self.entry_price - max_stop_dist,
                        min(self.entry_price - min_stop_dist, final_sl)
                    )
                else: # short
                    # 1. Standard ATR Baseline
                    target_stop_dist = atr * self.atr_multiplier
                    standard_sl = self.entry_price + target_stop_dist
                    final_sl = standard_sl

                    # 2. Joint Synergy: MA Shield (only active when use_ema_structure=true)
                    # Q3: zone_current_upper joins the MA candidates — it's a tested
                    # structural line same as the MAs, so it competes on equal terms
                    # rather than overriding or short-circuiting the existing shield.
                    if self.use_ema_structure:
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50, self.zone_current_upper]:
                            if ma and standard_sl > ma > self.entry_price:
                                buffered_ma_sl = ma + (0.5 * atr)
                                if buffered_ma_sl < final_sl:
                                    logger.info(f"[VTM] 🛡️ MA Shield Jointly Applied: SL tucked behind MA ${ma:,.2f}")
                                    final_sl = buffered_ma_sl

                    # 3. Apply global clamps
                    final_sl = min(
                        self.entry_price + max_stop_dist,
                        max(self.entry_price + min_stop_dist, final_sl)
                    )

                # STEP 2.5 — Structural Stop Override (Phase 4)
                # Routes stop placement by entry_type from the retest engine.
                # Overrides the ATR baseline with a level-anchored stop when
                # the entry context provides a structural invalidation reference.
                # Global clamps (min/max ATR) applied again after to ensure safety.
                try:
                    _struct_sl = self._compute_structural_stop(atr)
                    if _struct_sl is not None:
                        # Validate the structural stop is on the correct side of entry
                        _valid = (
                            (self.side == "long"  and _struct_sl < self.entry_price) or
                            (self.side == "short" and _struct_sl > self.entry_price)
                        )
                        # MR_PULLBACK special rule: only override ATR if the structural
                        # stop is MORE conservative (wider) than the ATR baseline.
                        # The intent is to ensure the stop clears the nearby level —
                        # if ATR already clears it, no override is needed and tightening
                        # the stop would put it inside the active zone.
                        if _valid and self.vtm_entry_type == "MR_PULLBACK":
                            _more_conservative = (
                                (self.side == "long"  and _struct_sl < final_sl) or
                                (self.side == "short" and _struct_sl > final_sl)
                            )
                            _valid = _more_conservative
                        if _valid:
                            final_sl = _struct_sl
                            # Re-apply clamps to the structural stop
                            if self.side == "long":
                                final_sl = max(
                                    self.entry_price - max_stop_dist,
                                    min(self.entry_price - min_stop_dist, final_sl),
                                )
                            else:
                                final_sl = min(
                                    self.entry_price + max_stop_dist,
                                    max(self.entry_price + min_stop_dist, final_sl),
                                )
                            # Detect whether the clamp discarded the structural
                            # reference. If it did, the stop is the ATR cap in
                            # disguise — log it honestly so it can't be mistaken
                            # for a level-anchored stop in the trade record.
                            _anchor_discarded = abs(final_sl - _struct_sl) > (atr * 0.05)
                            if _anchor_discarded:
                                self.stop_type = "atr_capped"
                                logger.info(
                                    f"[VTM] ⚠️ Structural anchor too far "
                                    f"({self.vtm_entry_type} raw={_struct_sl:.5f}): "
                                    f"ATR cap applied → {final_sl:.5f}"
                                )
                            else:
                                self.stop_type = "structural"
                                logger.info(
                                    f"[VTM] 🎯 Structural stop "
                                    f"({self.vtm_entry_type}): {final_sl:.5f} "
                                    f"(raw anchor={_struct_sl:.5f})"
                                )
                        else:
                            logger.debug(
                                f"[VTM] Structural stop {_struct_sl:.5f} invalid "
                                f"side — keeping ATR stop {final_sl:.5f}"
                            )
                except Exception as _ss_err:
                    logger.debug(f"[VTM] Structural stop error (non-blocking): {_ss_err}")

                # STEP 2 — Spread-Aware SL Floor
                # Reason: Prevents stops from being too tight relative to broker spread.
                if self.current_ask > 0 and self.current_bid > 0:
                    spread = abs(self.current_ask - self.current_bid)
                    calculated_sl_dist = abs(self.entry_price - final_sl)
                    final_sl_distance = max(calculated_sl_dist, 3.0 * spread)
                    
                    if final_sl_distance > calculated_sl_dist:
                        logger.info(f"[VTM] ↔️ Spread Floor: SL distance expanded to {final_sl_distance:.4f} (3x spread)")
                        final_sl = self.entry_price - final_sl_distance if self.side == "long" else self.entry_price + final_sl_distance

                # Global clamped final_sl assignment
                self.initial_stop_loss = final_sl

                # ── min_sl_pct floor ─────────────────────────────────────────────
                # Hard minimum SL distance as a percentage of entry price.
                # Prevents ATR-derived stops from being dangerously tight during
                # volatility squeezes or low-spread exotic sessions.
                # Sourced from risk_config["min_sl_pct"] per asset in config.json.
                _min_sl_pct = self.risk_config.get("min_sl_pct", 0.0)
                if _min_sl_pct > 0 and self.entry_price > 0:
                    _min_sl_dist = self.entry_price * _min_sl_pct
                    if self.side == "long":
                        _floor_sl = self.entry_price - _min_sl_dist
                        if self.initial_stop_loss > _floor_sl:
                            logger.info(
                                f"[VTM] min_sl_pct floor: SL {self.initial_stop_loss:.5f} → {_floor_sl:.5f} "
                                f"({_min_sl_pct * 100:.2f}% of entry {self.entry_price:.5f})"
                            )
                            self.initial_stop_loss = _floor_sl
                    else:
                        _floor_sl = self.entry_price + _min_sl_dist
                        if self.initial_stop_loss < _floor_sl:
                            logger.info(
                                f"[VTM] min_sl_pct floor: SL {self.initial_stop_loss:.5f} → {_floor_sl:.5f} "
                                f"({_min_sl_pct * 100:.2f}% of entry {self.entry_price:.5f})"
                            )
                            self.initial_stop_loss = _floor_sl

                # ── PHASE 4: Flagpole TP ladder (NATURAL_RETRACEMENT / REBOUND) ──
                # When a trade is entered after a confirmed pullback within a MAIN
                # Livermore leg, static ATR multiples under-estimate the available
                # move.  The flagpole (distance from entry to the pre-retracement
                # MAIN pivot) gives a market-derived target ladder:
                #   TP1 = entry ± 0.5 × flagpole  (half-flag: midway to prev pivot)
                #   TP2 = prev MAIN pivot          (full-flag: re-test prior extreme)
                #   TP3 = prev pivot ± 0.5 × flagpole  (extension beyond pivot)
                # Only fires when the anchor is available and flagpole > 1.0 × ATR
                # (prevents firing on noise / anchor data lag).
                _flagpole_used = False
                try:
                    _lsm4 = self.livermore_state_4h
                    _is_nat_long  = (_lsm4 == "NATURAL_RETRACEMENT" and self.side == "long")
                    _is_nat_short = (_lsm4 == "NATURAL_REBOUND"     and self.side == "short")

                    if _is_nat_long and self.livermore_anchor_main_up is not None:
                        _anchor   = float(self.livermore_anchor_main_up)
                        _flagpole = _anchor - self.entry_price
                        if _flagpole > atr:  # sanity: anchor must be meaningfully above entry
                            _tp1 = self.entry_price + 0.5 * _flagpole
                            _tp2 = _anchor
                            _tp3 = _anchor + 0.5 * _flagpole
                            self.take_profit_levels = [_tp1, _tp2, _tp3]
                            self.partial_sizes      = [0.45, 0.30, 0.25]
                            _flagpole_used = True
                            logger.info(
                                f"[VTM] 🎯 Flagpole ladder (LONG): anchor={_anchor:.5f} "
                                f"flagpole={_flagpole:.5f} "
                                f"TPs=[{_tp1:.5f}, {_tp2:.5f}, {_tp3:.5f}]"
                            )

                    elif _is_nat_short and self.livermore_anchor_main_down is not None:
                        _anchor   = float(self.livermore_anchor_main_down)
                        _flagpole = self.entry_price - _anchor
                        if _flagpole > atr:  # sanity: anchor must be meaningfully below entry
                            _tp1 = self.entry_price - 0.5 * _flagpole
                            _tp2 = _anchor
                            _tp3 = _anchor - 0.5 * _flagpole
                            self.take_profit_levels = [_tp1, _tp2, _tp3]
                            self.partial_sizes      = [0.45, 0.30, 0.25]
                            _flagpole_used = True
                            logger.info(
                                f"[VTM] 🎯 Flagpole ladder (SHORT): anchor={_anchor:.5f} "
                                f"flagpole={_flagpole:.5f} "
                                f"TPs=[{_tp1:.5f}, {_tp2:.5f}, {_tp3:.5f}]"
                            )
                except Exception as _fp_err:
                    logger.debug(f"[VTM] Flagpole ladder error (non-blocking): {_fp_err}")

                # ── PHASE 4 ext: CONTINUATION + RANGE target ladders ──────────────
                # Gated by phase_config.continuation_targets_enabled (default OFF).
                # Does nothing while the flag is off — vtm_entry_type can only be
                # CONTINUATION when retest_engine's matching tier was itself gated
                # by the same flag, and vtm_range_high/low are only ever populated
                # under the same gate, so this block is a no-op until both the
                # flag is on AND the corresponding tier actually fired.
                if not _flagpole_used and self.risk_config.get("phase_config", {}).get(
                    "continuation_targets_enabled", False
                ):
                    try:
                        _cl_cfg = self.risk_config.get("continuation_ladder", {})

                        if self.vtm_entry_type == "CONTINUATION":
                            # Flagpole = distance from the leg-origin pivot (the
                            # NATURAL anchor that started the current MAIN leg) to
                            # the entry price (near the top of the leg, at the
                            # consolidation breakout). T1 = half flagpole (50%),
                            # T2 = full flagpole (remaining 50%).
                            _base_anchor = (
                                self.livermore_anchor_nat_low if self.side == "long"
                                else self.livermore_anchor_nat_high
                            )
                            if _base_anchor is not None:
                                _base_anchor = float(_base_anchor)
                                _flagpole = (
                                    self.entry_price - _base_anchor if self.side == "long"
                                    else _base_anchor - self.entry_price
                                )
                                _min_fp_mult = float(_cl_cfg.get("continuation_min_flagpole_atr_mult", 1.0))
                                if _flagpole > _min_fp_mult * atr:
                                    _sign = 1 if self.side == "long" else -1
                                    _tp1 = self.entry_price + _sign * 0.5 * _flagpole
                                    _tp2 = self.entry_price + _sign * 1.0 * _flagpole
                                    self.take_profit_levels = [_tp1, _tp2]
                                    self.partial_sizes = list(
                                        _cl_cfg.get("continuation_partial_sizes", [0.50, 0.50])
                                    )
                                    _flagpole_used = True
                                    logger.info(
                                        f"[VTM] 🎯 CONTINUATION ladder ({self.side.upper()}): "
                                        f"flagpole={_flagpole:.5f} TPs=[{_tp1:.5f}, {_tp2:.5f}]"
                                    )

                        elif (
                            self.vtm_entry_type in ("RANGE_BOUNDARY", "SPRING_ENTRY")
                            and self.vtm_range_high is not None
                            and self.vtm_range_low is not None
                        ):
                            # Range midpoint (60%), range top/bottom, and the
                            # height-measured-move extension beyond it.
                            _rh, _rl = float(self.vtm_range_high), float(self.vtm_range_low)
                            _range_height = _rh - _rl
                            if _range_height > 0.25 * atr:  # sanity floor — avoid degenerate boxes
                                _mid = (_rh + _rl) / 2.0
                                if self.side == "long":
                                    _tp1, _tp2, _tp3 = _mid, _rh, _rh + _range_height
                                else:
                                    _tp1, _tp2, _tp3 = _mid, _rl, _rl - _range_height
                                self.take_profit_levels = [_tp1, _tp2, _tp3]
                                self.partial_sizes = list(
                                    _cl_cfg.get("range_partial_sizes", [0.60, 0.25, 0.15])
                                )
                                _flagpole_used = True
                                logger.info(
                                    f"[VTM] 🎯 RANGE ladder ({self.side.upper()}): "
                                    f"box=[{_rl:.5f}, {_rh:.5f}] "
                                    f"TPs=[{_tp1:.5f}, {_tp2:.5f}, {_tp3:.5f}]"
                                )
                    except Exception as _cl_err:
                        logger.debug(f"[VTM] Continuation/range ladder error (non-blocking): {_cl_err}")

                # Structure-based targets (only when use_structure_targets=true
                # and flagpole ladder was not used)
                if not _flagpole_used and self.use_structure_targets:
                    tolerance = 0.5 * atr
                    structure_levels = find_resistance_levels(self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback, tolerance=tolerance)
                    raw_targets, self.partial_sizes = calculate_hybrid_targets(
                        self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                        self.partial_targets, self.partial_sizes,
                        min_rr=2.0  # Standard TREND requirement
                    )
                    logger.debug(f"[VTM] Structure targets active: {len(structure_levels)} levels found")
                else:
                    # Pure ATR-multiple targets, no pivot hunting
                    raw_targets = [
                        self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m)
                        for m in self.partial_targets
                    ]
                    logger.debug("[VTM] Structure targets disabled — using ATR multiples only")

                # ✅ PHASE 5: MA FRONT-RUN (Take Profit — only when use_ema_structure=true)
                # Skip when flagpole ladder was used — those TPs are already optimised.
                if not _flagpole_used:
                    self.take_profit_levels = []
                if not _flagpole_used and self.use_ema_structure:
                    for tp in raw_targets:
                        adjusted_tp = tp
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                            if ma:
                                if self.side == "long":
                                    if abs(tp - ma) < (0.5 * atr) or (tp > ma > self.entry_price):
                                        candidate_tp = ma - (0.25 * atr)
                                        if candidate_tp > self.entry_price + (0.5 * atr):
                                            adjusted_tp = max(adjusted_tp, candidate_tp)
                                else:  # short
                                    if abs(tp - ma) < (0.5 * atr) or (tp < ma < self.entry_price):
                                        candidate_tp = ma + (0.25 * atr)
                                        if candidate_tp < self.entry_price - (0.5 * atr):
                                            adjusted_tp = min(adjusted_tp, candidate_tp)
                        self.take_profit_levels.append(adjusted_tp)
                elif not _flagpole_used:
                    # No EMA adjustment — use raw targets directly
                    self.take_profit_levels = list(raw_targets)

                # Fallback targets (only when no flagpole and no other TPs set)
                if not self.take_profit_levels:
                    self.take_profit_levels = [self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m) for m in self.partial_targets]
                    self.partial_sizes = [0.45, 0.30, 0.25]  # fallback only

                # ── 7.1 LIVE: expectancy-weighted single-TP blend ─────────────
                # Runs before min_rr so the floor can raise the blend if needed.
                if self.risk_config.get("phase_config", {}).get("single_tp_blend_enabled", False):
                    _b = self._blend_single_tp()
                    if _b is not None and self.take_profit_levels:
                        self.take_profit_levels[0] = _b

                # ── min_rr enforcement on TP1 ─────────────────────────────────
                # When min_sl_pct widens the SL, the ATR-derived TP1 may produce
                # a sub-1R trade. Enforce minimum R:R on TP1 only; TP2/TP3 are
                # structure/flagpole targets and are left untouched.
                # Sourced from risk_config["min_rr"] per asset in config.json.
                _min_rr = self.risk_config.get("min_rr", 1.5)
                if _min_rr > 0 and self.take_profit_levels and self.entry_price > 0:
                    _sl_dist = abs(self.entry_price - self.initial_stop_loss)
                    if _sl_dist > 0:
                        _tp1_current      = self.take_profit_levels[0]
                        _tp1_current_dist = abs(_tp1_current - self.entry_price)
                        _tp1_floor_dist   = _sl_dist * _min_rr
                        if _tp1_current_dist < _tp1_floor_dist:
                            _tp1_floor = (
                                self.entry_price + _tp1_floor_dist if self.side == "long"
                                else self.entry_price - _tp1_floor_dist
                            )
                            logger.info(
                                f"[VTM] min_rr floor: TP1 {_tp1_current:.5f} → {_tp1_floor:.5f} "
                                f"(R:R {_tp1_current_dist / _sl_dist:.2f} → {_min_rr:.1f}, "
                                f"SL dist={_sl_dist:.5f})"
                            )
                            self.take_profit_levels[0] = _tp1_floor

            # STEP 3 — Lot Sanitizer
            # Reason: Ensures position size is valid for broker submission.
            LOT_PRECISION = {
                'BTC': 4,
                'GOLD': 2,
                'USTEC': 2,
                'EURJPY': 2,
                'EURUSD': 2,
                'GBPUSD': 2,
                'USDJPY': 2,
                'USOIL': 2,
                'GBPAUD': 2,
            }

            precision = self.lot_precision_override if self.lot_precision_override is not None else LOT_PRECISION.get(self.asset.upper(), 2)
            final_size = round(self.position_size, precision)

            MIN_LOT = {
                'BTC': 0.003,
                'GOLD': 0.01,
                'USTEC': 0.02,
                'EURJPY': 0.02,
                'EURUSD': 0.01,
                'GBPUSD': 0.01,
                'USDJPY': 0.01,
                'USOIL': 0.01,
                'GBPAUD': 0.01,
            }

            min_lot = self.min_lot_override if self.min_lot_override is not None else MIN_LOT.get(self.asset.upper(), 0.01)

            # Compare the RAW (unrounded) size against min_lot, not the display-
            # rounded final_size. When re-initializing VTM for an existing
            # position, min_lot_override is set to that position's own exact
            # quantity (e.g. 999.9627491152916) — rounding self.position_size
            # to the asset's lot precision first (e.g. 999.96) made a position
            # fail this check against its own unrounded quantity every single
            # retry, permanently blocking VTM re-init. Small epsilon absorbs
            # float-representation noise, not a real shortfall.
            if self.position_size < min_lot - 1e-9:
                logger.warning(f"[VTM] Trade aborted: Final size {final_size} below minimum lot {min_lot} for {self.asset}.")
                # We raise an exception here to signal the manager to abort trade creation
                raise ValueError(f"Size {final_size} below min {min_lot} for {self.asset}")
            
            self.position_size = final_size
            self.current_stop_loss = self.initial_stop_loss

        except ValueError as ve:
            raise # Re-raise lot size error to abort
        except Exception as e:
            logger.error(f"[VTM] Level calculation error: {e}", exc_info=True)
            raise

    def on_new_bar(self, new_high: float, new_low: float, new_close: float) -> Optional[Dict]:
        try:
            self.high, self.low, self.close = np.append(self.high, new_high), np.append(self.low, new_low), np.append(self.close, new_close)
            # ✨ MEMORY MANAGEMENT: Limit to 500 candles (Safe for 200 EMA + buffer)
            if len(self.close) > 500: 
                self.high, self.low, self.close = self.high[-500:], self.low[-500:], self.close[-500:]
            
            self.bars_in_trade += 1
            if self.side == "long": self.highest_price_reached = max(self.highest_price_reached, new_high)
            else: self.lowest_price_reached = min(self.lowest_price_reached, new_low)
            
            atr = self._calculate_atr() # Calculate ATR here
            return self.check_exit(new_close, atr) # Pass ATR to check_exit
        except Exception as e:
            logger.error(f"[VTM] Update error: {e}")
            return None

    def update_with_current_price(
        self,
        current_price: float,
        df_4h: Optional[pd.DataFrame] = None,
        composite_state=None,
        judge_scores: Optional[Dict] = None,
    ) -> Optional[Dict]:
        # A13: this method writes current_stop_loss directly (trailing/
        # breakeven updates below) before calling check_exit — hold the same
        # lock for the whole tick so market_watcher.py's cross-thread SL push
        # can't land in the middle of it.
        with self._sl_lock:
            return self._update_with_current_price_locked(
                current_price, df_4h=df_4h, composite_state=composite_state,
                judge_scores=judge_scores,
            )

    def _update_with_current_price_locked(
        self,
        current_price: float,
        df_4h: Optional[pd.DataFrame] = None,
        composite_state=None,
        judge_scores: Optional[Dict] = None,
    ) -> Optional[Dict]:
        try:
            atr = self._calculate_atr()

            # ── Structural prerequisite: store live composite_state on VTM ──────
            # Enables check_exit to read live structural signals (BOS, CHoCH,
            # Livermore state) without requiring a signature change to check_exit.
            # If composite_state is None, all structural checks in check_exit
            # gracefully skip and ATR behaviour is preserved.
            self._live_cs = composite_state
            # Brain Rebuild Part 3.4: same idea for judge scores — feeds the
            # human-alert layer's (Part 3.5) structure-break-against-position
            # check. {"buy": {...}, "sell": {...}} or None.
            self._live_judge_scores = judge_scores

            # Brain Rebuild Part 3.5: human-alert layer. Fires once per
            # position (not every tick) the moment 3+ independent signals
            # turn against it — same one-shot pattern as
            # _structural_warning_fired below, so this doesn't spam Telegram
            # once the condition is met.
            if not getattr(self, "_human_alert_fired", False):
                _alert_msg = self._check_alert_conditions()
                if _alert_msg:
                    self._human_alert_fired = True
                    self._safe_send_alert(_alert_msg)


            # ── PHASE 4: Live Livermore State Refresh ──────────────────────────
            # Refresh 4H Livermore state every cycle so VTM responds to structural
            # transitions that occur AFTER trade entry, not just at entry time.
            # Gated by phase_config.per_tick_livermore_enabled.
            # Only activate after Gate 3B criteria confirmed (21-day live observation
            # with structural stops enabled, no increase in premature stop-outs).
            _phase_cfg = self.risk_config.get("phase_config", {})
            _per_tick_enabled = _phase_cfg.get("per_tick_livermore_enabled", False)
            if composite_state is not None and _per_tick_enabled:
                # Brain Rebuild Part 3.3 (D4): keep the 1H companion current
                # every cycle — it changes far more often than 4H, so it isn't
                # gated behind the 4H change-detection below.
                _new_lv1h = getattr(composite_state, "livermore_state_1h", None)
                if _new_lv1h:
                    self.livermore_state_1h = _new_lv1h
                    self.livermore_state_age_1h = int(
                        getattr(composite_state, "livermore_state_age_1h", 0) or 0
                    )

                _new_lv4h = getattr(composite_state, "livermore_state_4h", None)
                _new_age   = int(getattr(composite_state, "livermore_state_age_4h", 0) or 0)
                if _new_lv4h and _new_lv4h != self.livermore_state_4h:
                    _prev = self.livermore_state_4h
                    self.livermore_state_4h     = _new_lv4h
                    self.livermore_state_age_4h = _new_age
                    logger.info(
                        "[VTM LIVE LSM] %s: 4H state %s → %s (age=%d bars)",
                        self.asset, _prev, _new_lv4h, _new_age,
                    )
                    self._apply_livermore_transition(
                        prev_state=_prev,
                        new_state=_new_lv4h,
                        current_price=current_price,
                        atr=atr,
                    )

                # Gate Tier 3.2: does the retracement this trade was opened
                # against still hold, or is it already failing? Independent
                # of Gate Tier 3.1's entry-side flag — emergency brake here is
                # its own switch (gate3_d4_extension_enabled, default True) so
                # a shadow-mode divergence can be attributed to one piece or
                # the other, not bundled.
                if _phase_cfg.get("gate3_d4_extension_enabled", True):
                    _retr_signal = self._check_retracement_holding(current_price, atr)
                    if _retr_signal == "retracement_failing":
                        _config_trail = self.risk_config.get("runner_trail_atr_multiplier", 1.0)
                        _sec_trail = self.risk_config.get("livermore_atr_adjustments", {}).get(
                            "secondary_trail_mult", 0.7
                        )
                        # Tighten, don't lock breakeven outright — genuinely
                        # different confidence than a confirmed 4H MAIN flip.
                        self.runner_trail_atr_multiplier = _config_trail * _sec_trail * 0.5
                        logger.warning(
                            "[VTM D4] %s: retracement failing (depth recovered >50%% "
                            "since entry) — trail tightened to %.2f× ATR",
                            self.asset, self.runner_trail_atr_multiplier,
                        )

            # ── L4: VTM structural exit awareness ──────────────────────────────
            # CHoCH/BOS detection is finer-grained and faster than a discrete
            # livermore_state_4h transition (which only updates on a 4H close).
            # When a structural break shows up against the position's direction,
            # lock in a one-time protective tighten — same mechanism as the
            # MAIN→SECONDARY case above, but triggered by the structural-break
            # flags directly instead of waiting for the state machine to catch up.
            # _structural_warning_fired ensures this fires once per trade, not
            # every tick the flags happen to still read true.
            if (
                composite_state is not None
                and atr > 0
                and self.risk_config.get("phase_config", {}).get(
                    "vtm_structural_exit_awareness_enabled", False
                )
                and not getattr(self, "_structural_warning_fired", False)
            ):
                _choch = bool(getattr(composite_state, "choch_detected", False))
                _bos = bool(getattr(composite_state, "bos_detected", False))
                _sweep_dir = getattr(composite_state, "sweep_direction", 0) or 0
                if _choch or _bos:
                    is_long = self.side == "long"
                    # against_position: sweep_direction < 0 means a bearish
                    # break/sweep — adverse for a long, favorable for a short.
                    _against_position = (
                        (is_long and _sweep_dir < 0) or
                        (not is_long and _sweep_dir > 0)
                    )
                    if _against_position:
                        self._structural_warning_fired = True
                        _in_profit = (
                            (is_long and current_price > self.entry_price) or
                            (not is_long and current_price < self.entry_price)
                        )
                        if _in_profit and not getattr(self, "_lv_breakeven_locked", False):
                            self.current_stop_loss = self.entry_price
                            self._lv_breakeven_locked = True
                        self.runner_trail_atr_multiplier *= 0.5
                        logger.warning(
                            "[VTM STRUCTURAL] %s: %s detected against %s position — "
                            "trail tightened to %.2f× ATR%s",
                            self.asset,
                            "CHoCH" if _choch else "BOS",
                            self.side,
                            self.runner_trail_atr_multiplier,
                            " + SL locked to breakeven" if getattr(self, "_lv_breakeven_locked", False) else "",
                        )

                # ── O1b: Orphan signal awareness ──────────────────────────
                # Three additional signals that wave L4's structural-flag
                # check doesn't capture. Each fires once per position via
                # a dedicated flag — no per-tick re-triggering.

                # 1. Rejection strength — price hard-rejected at a key level.
                # Scale trail tighten proportionally to the rejection magnitude.
                _rejection_str = float(getattr(composite_state, "rejection_strength", 0.0))
                _rejection_at  = bool(getattr(composite_state, "rejection_at_level", False))
                if _rejection_at and _rejection_str >= 0.60 and not getattr(self, "_rejection_trail_fired", False):
                    _sweep_dir = getattr(composite_state, "sweep_direction", 0)
                    # Use strict comparison: sweep_direction=0 means no sweep
                    # occurred. <= 0 / >= 0 would fire on BOTH sides when
                    # sweep_direction is neutral, tightening every open
                    # position on ordinary rejections. Only tighten when a
                    # directional sweep actually occurred against the position.
                    _rejection_against = (
                        (self.side == "long" and _sweep_dir < 0) or
                        (self.side == "short" and _sweep_dir > 0)
                    )
                    if _rejection_against:
                        self._rejection_trail_fired = True
                        # 0.60 strength → 0.76× multiplier, 0.90 strength → 0.64×.
                        # Calibrate via O1 telemetry after 30+ samples.
                        _rejection_mult = max(0.64, 1.0 - (_rejection_str * 0.40))
                        _new_mult = self.runner_trail_atr_multiplier * _rejection_mult
                        logger.info(
                            f"[VTM] {self.asset}: level rejection (strength={_rejection_str:.2f}) "
                            f"against {self.side} — trail mult "
                            f"{self.runner_trail_atr_multiplier:.2f}× → {_new_mult:.2f}×"
                        )
                        self.runner_trail_atr_multiplier = _new_mult

                # 2. is_parabolic in position's favor — lock in gains.
                # Parabolic extension is a take-profit-opportunity signal
                # on the winning side. Tighten trail so gains don't evaporate
                # when the extension inevitably reverses.
                _is_parabolic = bool(getattr(composite_state, "is_parabolic", False))
                if _is_parabolic and not getattr(self, "_parabolic_trail_locked", False):
                    _entry = self.entry_price
                    _min_profit_pct = 0.002  # 0.2% minimum profit to trigger
                    _in_profit = (
                        (self.side == "long" and
                         self.highest_price_reached is not None and
                         self.highest_price_reached > _entry * (1 + _min_profit_pct)) or
                        (self.side == "short" and
                         self.lowest_price_reached is not None and
                         self.lowest_price_reached < _entry * (1 - _min_profit_pct))
                    )
                    if _in_profit:
                        self._parabolic_trail_locked = True
                        _locked_mult = min(self.runner_trail_atr_multiplier, 0.80)
                        logger.info(
                            f"[VTM] {self.asset}: parabolic extension in {self.side}'s "
                            f"favor — locking trail "
                            f"({self.runner_trail_atr_multiplier:.2f}× → {_locked_mult:.2f}×)"
                        )
                        self.runner_trail_atr_multiplier = _locked_mult

                # 3. absorption_detected — institutional orders absorbing at price.
                # Only act outside silent zones (normal retracements expect this).
                _absorption = bool(getattr(composite_state, "absorption_detected", False))
                _silent_now = bool(getattr(composite_state, "is_silent_zone", False))
                if _absorption and not _silent_now and not getattr(self, "_absorption_warning_fired", False):
                    self._absorption_warning_fired = True
                    _abs_mult = min(self.runner_trail_atr_multiplier, 1.10)
                    logger.info(
                        f"[VTM] {self.asset}: absorption_detected against "
                        f"{self.side} (outside silent zone) — mild trail tighten "
                        f"({self.runner_trail_atr_multiplier:.2f}× → {_abs_mult:.2f}×)"
                    )
                    self.runner_trail_atr_multiplier = _abs_mult

            if self.side == "long":
                # Guard: highest_price_reached may be None if entry_price was None at construction
                if self.highest_price_reached is None:
                    self.highest_price_reached = current_price
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                if self.runner_activated and self.highest_price_reached > old_high and self.trade_type == "TREND":
                    new_trail = self.highest_price_reached - (self.runner_trail_atr_multiplier * atr)
                    if new_trail > self.current_stop_loss:
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail

                # ── STRUCTURAL SWING LOW TRAIL (Option 1 + Option B) ──────────
                # Fires for BOTH REVERSION and TREND types when in profit.
                # ATR runner trail (above) remains the floor for TREND type.
                # Structural trail takes over when it produces a higher stop.
                # Only active when structural_trailing_enabled = true in config.
                _current_profit_long = self.close[-1] - self.entry_price
                if _current_profit_long > 1.0 * atr:
                    _struct_sl = self._compute_swing_low_trail(atr)
                    if _struct_sl is not None and _struct_sl > self.current_stop_loss:
                        logger.info(
                            "[VTM] 🏗️ Structural trail: %s SL → %.5g "
                            "(swing low − 0.3×ATR, was %.5g)",
                            self.asset, _struct_sl, self.current_stop_loss,
                        )
                        self.current_stop_loss = _struct_sl
                # ───────────────────────────────────────────────
            else:
                # Guard: lowest_price_reached may be None if entry_price was None at construction
                if self.lowest_price_reached is None:
                    self.lowest_price_reached = current_price
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                if self.runner_activated and self.lowest_price_reached < old_low and self.trade_type == "TREND":
                    new_trail = self.lowest_price_reached + (self.runner_trail_atr_multiplier * atr)
                    if new_trail < self.current_stop_loss:
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail

                # ── STRUCTURAL SWING HIGH TRAIL (Option 1 + Option B, shorts) ─
                _current_profit_short = self.entry_price - self.close[-1]
                if _current_profit_short > 1.0 * atr:
                    _struct_sl = self._compute_swing_low_trail(atr)
                    if _struct_sl is not None and _struct_sl < self.current_stop_loss:
                        logger.info(
                            "[VTM] 🏗️ Structural trail: %s SL → %.5g "
                            "(swing high + 0.3×ATR, was %.5g)",
                            self.asset, _struct_sl, self.current_stop_loss,
                        )
                        self.current_stop_loss = _struct_sl
                # ──────────────────────────────────────────────────────────────

            # Q3: dynamic zone-aware TP. If price has cleared the zone that was
            # our target, the next zone up/down becomes the new target — the
            # position is managed against live structure, not a fixed entry-time TP.
            if composite_state is not None:
                _zu = (composite_state.get("zone_4h_current_upper")
                       if isinstance(composite_state, dict)
                       else getattr(composite_state, "zone_4h_current_upper", None))
                _zl = (composite_state.get("zone_4h_current_lower")
                       if isinstance(composite_state, dict)
                       else getattr(composite_state, "zone_4h_current_lower", None))
                _next = _zu if self.side == "long" else _zl
                if _next is not None and self.take_profit_levels:
                    _cur_tp = self.take_profit_levels[-1]
                    # Only ADVANCE the target in the trade's favour, never pull it in.
                    if self.side == "long" and _next > _cur_tp:
                        self.take_profit_levels[-1] = _next
                        logger.info(f"[VTM] {self.asset}: price cleared zone — TP advanced to {_next:.5f}")
                    elif self.side == "short" and _next < _cur_tp:
                        self.take_profit_levels[-1] = _next
                        logger.info(f"[VTM] {self.asset}: price cleared zone — TP advanced to {_next:.5f}")

            # A13: already holding _sl_lock (acquired by the update_with_
            # current_price wrapper) — call the unlocked impl directly to
            # avoid re-acquiring the same non-reentrant lock (deadlock).
            return self._check_exit_locked(current_price, atr, df_4h=df_4h)
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

    def _apply_livermore_transition(
        self,
        prev_state: str,
        new_state: str,
        current_price: float,
        atr: float,
    ) -> None:
        """
        Respond to a live 4H Livermore state transition during an open trade.

        Rules (MRS Phase 4):
        - MAIN → NATURAL: price starting a counter-move. Tighten trail to protect
          profits. Do NOT exit — natural moves are expected within trends.
        - MAIN → SECONDARY: deeper counter-move, structural warning. Move SL to
          breakeven if profitable; tighten trail aggressively.
        - NATURAL/SECONDARY → MAIN (same direction): trend resumed. Restore trail
          multiplier so runner can breathe again.
        - Any → MAIN (opposite direction): trend has reversed. Tighten to near
          breakeven — the original thesis is now invalidated.
        """
        if atr <= 0:
            return

        _lv_adj        = self.risk_config.get("livermore_atr_adjustments", {})
        _main_trail    = _lv_adj.get("main_trail_mult",      1.3)
        _sec_trail     = _lv_adj.get("secondary_trail_mult", 0.7)
        _config_trail  = self.risk_config.get("runner_trail_atr_multiplier", 1.0)

        _MAIN    = {"MAIN_UP", "MAIN_DOWN"}
        _NATURAL = {"NATURAL_RETRACEMENT", "NATURAL_REBOUND"}
        _SEC     = {"SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"}

        is_long  = self.side == "long"
        in_profit = (
            (is_long  and current_price > self.entry_price) or
            (not is_long and current_price < self.entry_price)
        )

        # ── MAIN → NATURAL: counter-move starting ────────────────────────
        if prev_state in _MAIN and new_state in _NATURAL:
            _same_dir = (
                (is_long  and new_state == "NATURAL_RETRACEMENT") or
                (not is_long and new_state == "NATURAL_REBOUND")
            )
            # Item 2.16: a counter-trend/REVERSION position (possible now that
            # trade_type actually reflects live regime — Item 2.11) held AGAINST
            # the prevailing MAIN leg has its own thesis confirmed by the
            # opposite transition: a LONG bet against MAIN_DOWN is confirmed by
            # NATURAL_REBOUND; a SHORT bet against MAIN_UP is confirmed by
            # NATURAL_RETRACEMENT. VTM previously had no branch for this at
            # all — it silently did nothing while a working reversion thesis
            # played out.
            _thesis_confirming = (
                (is_long and new_state == "NATURAL_REBOUND") or
                (not is_long and new_state == "NATURAL_RETRACEMENT")
            )
            if _same_dir:
                # Normal pullback against position — tighten trail
                self.runner_trail_atr_multiplier = _config_trail * _sec_trail
                logger.info(
                    "[VTM LIVE LSM] %s: MAIN→NATURAL — tightening trail to %.2f× ATR",
                    self.asset, self.runner_trail_atr_multiplier,
                )
            elif _thesis_confirming:
                self.runner_trail_atr_multiplier = _config_trail * 1.1
                logger.info(
                    "[VTM LIVE LSM] %s: counter-trend thesis confirming — trail eased",
                    self.asset,
                )

        # ── MAIN → SECONDARY: deep counter-move, structural warning ──────
        elif prev_state in _MAIN and new_state in _SEC:
            self.runner_trail_atr_multiplier = _config_trail * _sec_trail * 0.5
            if in_profit and not getattr(self, "_lv_breakeven_locked", False):
                self.current_stop_loss = self.entry_price
                self._lv_breakeven_locked = True
                logger.warning(
                    "[VTM LIVE LSM] %s: MAIN→SECONDARY — SL moved to breakeven $%.2f",
                    self.asset, self.entry_price,
                )
            logger.info(
                "[VTM LIVE LSM] %s: MAIN→SECONDARY — trail tightened to %.2f× ATR",
                self.asset, self.runner_trail_atr_multiplier,
            )

        # ── NATURAL → SECONDARY: counter-move escalating ─────────────────
        elif prev_state in _NATURAL and new_state in _SEC:
            self.runner_trail_atr_multiplier = _config_trail * _sec_trail * 0.5
            if in_profit and not getattr(self, "_lv_breakeven_locked", False):
                self.current_stop_loss = self.entry_price
                self._lv_breakeven_locked = True
                logger.warning(
                    "[VTM LIVE LSM] %s: NATURAL→SECONDARY — SL moved to breakeven $%.2f",
                    self.asset, self.entry_price,
                )
            logger.info(
                "[VTM LIVE LSM] %s: NATURAL→SECONDARY — trail tightened to %.2f× ATR",
                self.asset, self.runner_trail_atr_multiplier,
            )

        # ── NATURAL/SECONDARY → MAIN: trend resuming or reversing ────────
        elif prev_state in (_NATURAL | _SEC) and new_state in _MAIN:
            _same_dir = (
                (is_long  and new_state == "MAIN_UP") or
                (not is_long and new_state == "MAIN_DOWN")
            )
            if _same_dir:
                self.runner_trail_atr_multiplier = _config_trail * _main_trail
                self._lv_breakeven_locked = False
                logger.info(
                    "[VTM LIVE LSM] %s: →MAIN_%s (trend resumed) — trail restored to %.2f× ATR",
                    self.asset, "UP" if is_long else "DOWN",
                    self.runner_trail_atr_multiplier,
                )
            else:
                # Opposite MAIN — full reversal
                self.runner_trail_atr_multiplier = _config_trail * _sec_trail * 0.3
                if not getattr(self, "_lv_breakeven_locked", False):
                    self.current_stop_loss = self.entry_price
                    self._lv_breakeven_locked = True
                logger.warning(
                    "[VTM LIVE LSM] %s: →MAIN_%s (REVERSAL) — SL at breakeven, trail %.2f× ATR",
                    self.asset, "UP" if not is_long else "DOWN",
                    self.runner_trail_atr_multiplier,
                )

        # ── MAIN → opposite MAIN: direct structural reversal ─────────────
        elif prev_state in _MAIN and new_state in _MAIN and prev_state != new_state:
            # Brain Rebuild Part 3.3 (D4): the unconditional "thesis
            # invalidated" treatment below assumed the position was aligned
            # with prev_state, so any flip to the opposite MAIN was adverse.
            # That's not true for a counter-trend position (e.g. held LONG
            # while prev_state was MAIN_DOWN, betting on a reversal) — for
            # that position, flipping to MAIN_UP CONFIRMS the thesis rather
            # than invalidating it. thesis_confirmed checks the new 4H state
            # against position direction; lv_1h_confirmed requires the 1H
            # read to agree too before easing off, matching this codebase's
            # established "don't trust a single timeframe alone" pattern
            # (e.g. the Trend judge's 1H/4H agreement design).
            thesis_confirmed = (
                (is_long and new_state == "MAIN_UP")
                or (not is_long and new_state == "MAIN_DOWN")
            )
            lv_1h_confirmed = (
                (is_long and self.livermore_state_1h == "MAIN_UP")
                or (not is_long and self.livermore_state_1h == "MAIN_DOWN")
            )
            if thesis_confirmed and lv_1h_confirmed:
                self.runner_trail_atr_multiplier = _config_trail * _main_trail
                self._lv_breakeven_locked = False
                logger.info(
                    "[VTM LIVE LSM] %s: %s→%s (thesis confirmed, 1H agrees) — "
                    "trail restored to %.2f× ATR",
                    self.asset, prev_state, new_state,
                    self.runner_trail_atr_multiplier,
                )
            else:
                # e.g. MAIN_DOWN → MAIN_UP while holding SHORT — thesis invalidated
                self.runner_trail_atr_multiplier = _config_trail * _sec_trail * 0.3
                if not getattr(self, "_lv_breakeven_locked", False):
                    self.current_stop_loss = self.entry_price
                    self._lv_breakeven_locked = True
                logger.warning(
                    "[VTM LIVE LSM] %s: %s→%s (DIRECT REVERSAL) — SL at breakeven $%.2f, trail %.2f× ATR",
                    self.asset, prev_state, new_state,
                    self.entry_price, self.runner_trail_atr_multiplier,
                )

    def _estimate_retracement_depth(
        self, current_price: float, atr: float, lsm_state: str
    ) -> float:
        """Gate Tier 3.2 — how far price has already pulled back from its own
        recent extreme, in ATRs. Post-entry companion to main.py's entry-time
        helper of the same name (Gate Tier 3.1): that one reads a 1H candle
        df directly since it runs pre-entry; the tick loop here never
        receives a 1H df (update_with_current_price only gets current_price
        and df_4h), so this reads the extremes VTM already tracks every tick
        (highest/lowest_price_reached since THIS position's entry) instead.
        Approximation, not exact parity with the entry-time read — good
        enough to detect "is the retracement still deepening or has it
        already turned," which is what _check_retracement_holding needs.
        """
        try:
            if atr <= 0:
                return 1.0  # neutral default, doesn't over- or under-penalize
            if lsm_state == "NATURAL_RETRACEMENT":
                _extreme = self.highest_price_reached if self.highest_price_reached is not None else current_price
                return max(0.0, (_extreme - current_price) / atr)
            elif lsm_state == "NATURAL_REBOUND":
                _extreme = self.lowest_price_reached if self.lowest_price_reached is not None else current_price
                return max(0.0, (current_price - _extreme) / atr)
            return 1.0
        except Exception:
            return 1.0

    def _check_retracement_holding(self, current_price: float, atr: float) -> Optional[str]:
        """Gate Tier 3.2 — a genuinely new transition type: does the
        retracement a Gate-Tier-3.1 counter-trend-adjacent trade was opened
        against continue to hold, or has it already started failing (i.e.
        the original trend resuming)? D4's existing _apply_livermore_transition
        watch only covers discrete MAIN-to-opposite-MAIN state flips — a
        different and structurally larger risk than a retracement quietly
        failing while livermore_state_1h hasn't changed state label at all.
        """
        if self.livermore_state_1h not in ("NATURAL_RETRACEMENT", "NATURAL_REBOUND"):
            return None
        _current_depth = self._estimate_retracement_depth(current_price, atr, self.livermore_state_1h)
        _entry_depth = getattr(self, "_retracement_depth_at_entry", None)
        if _entry_depth is None:
            self._retracement_depth_at_entry = _current_depth
            return None
        if _current_depth < _entry_depth * 0.5:
            return "retracement_failing"  # already recovered more than half its own depth — original trend likely resuming
        return None

    def _check_alert_conditions(self) -> Optional[str]:
        """
        Brain Rebuild Part 3.5 (direction-aware follow-up): human-alert
        layer. Independent of any auto-management action VTM takes — this
        only decides whether a human should be told that multiple signals
        have turned against an open position. Returns the alert message,
        or None if fewer than 3 of the tracked signals have fired.

        CHoCH/BOS are each set from two independent, opposite-meaning swing
        patterns at the source (signal_aggregator.py's _update_structure)
        with no way to tell which one fired from the plain boolean alone —
        choch_bearish/choch_bullish and bos_bearish/bos_bullish disambiguate
        that. conviction_dying is direction-agnostic too (just "candle
        bodies shrinking", regardless of which way price is moving) — it
        only means something for THIS position if the trend losing
        conviction is the one that's been carrying it favorably (the
        tailwind stalling), not the market actively turning, so it's
        cross-checked against the Livermore 1H read.
        """
        if self._live_cs is None:
            return None
        _cs = self._live_cs
        is_long = self.side == "long"
        signals_fired = []

        if getattr(_cs, "choch_bearish" if is_long else "choch_bullish", False):
            signals_fired.append("CHoCH against position")

        if getattr(_cs, "bos_bearish" if is_long else "bos_bullish", False):
            signals_fired.append("structure break against position")

        _lsm_1h = getattr(_cs, "livermore_state_1h", None)
        _bull_states = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
        _bear_states = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
        _favorable_trend_dying = getattr(_cs, "conviction_dying", False) and (
            (is_long and _lsm_1h in _bull_states)
            or (not is_long and _lsm_1h in _bear_states)
        )
        if _favorable_trend_dying:
            signals_fired.append("conviction fading")

        if getattr(_cs, "bearish_divergence" if is_long else "bullish_divergence", False):
            signals_fired.append("momentum divergence against position")

        # X1: the trajectory tracker declares a setup dead on real evidence
        # (Livermore state flip, failed breakout, opposing BOS). Nothing read
        # it. If the setup that justified this position has died, that is a
        # genuine signal against the position.
        if getattr(_cs, "setup_died", False):
            _death = getattr(_cs, "setup_death_reason", "unknown")
            signals_fired.append(f"setup invalidated ({_death})")

        # A3: sweep_detected consumer — a liquidity sweep (stop-hunt wick)
        # against the position's direction. Distinct from the per-tick trail
        # tighten in check_exit (which reacts to sweep_direction alone); this
        # is the human-alert count, so it only counts a sweep that's both
        # detected AND directionally against the position.
        _sweep_dir = getattr(_cs, "sweep_direction", 0) or 0
        _sweep_against = (
            getattr(_cs, "sweep_detected", False)
            and ((is_long and _sweep_dir < 0) or (not is_long and _sweep_dir > 0))
        )
        if _sweep_against:
            signals_fired.append("liquidity sweep against position")

        # A9: governor/1D-macro-regime relay. Reuses the same live council_ref
        # connection Gate Tier 4.1 built for _check_lifecycle_phase reuse —
        # _check_macro_regime is a clean, read-only regime read (needs only
        # the asset name), so it costs nothing extra to call continuously
        # here. Flags when the 1D macro regime now opposes the position,
        # independent of the 1H Livermore/structure signals above.
        if self.council_ref is not None and hasattr(self.council_ref, "_check_macro_regime"):
            try:
                _macro_regime = self.council_ref._check_macro_regime(self.asset)
                _macro_against = (
                    (is_long and _macro_regime == "BEARISH")
                    or (not is_long and _macro_regime == "BULLISH")
                )
                if _macro_against:
                    signals_fired.append(f"1D macro regime against position ({_macro_regime})")
            except Exception as _macro_err:
                logger.debug(f"[VTM ALERT] macro regime reuse failed: {_macro_err}")

        # Gate Tier 4.1: reuse the council's entry-time lifecycle classifier
        # continuously, not just at entry. _check_lifecycle_phase already does
        # a richer exhaustion read (ADX decline + overextension + RSI
        # divergence) than this alert layer builds from raw flags alone.
        if self.council_ref is not None and hasattr(self.council_ref, "_check_lifecycle_phase"):
            try:
                _lc_df = pd.DataFrame({"high": self.high, "low": self.low, "close": self.close})
                _, _phase_label = self.council_ref._check_lifecycle_phase(
                    _lc_df,
                    1 if is_long else -1,
                    self._calculate_adx(),
                    0,  # required_score unused for this read-only call
                    governor_data={"composite_state": _cs},
                )
                if _phase_label == "EXHAUSTED":
                    signals_fired.append("lifecycle_phase: EXHAUSTED (reused entry-time classifier)")
            except Exception as _lc_err:
                logger.debug(f"[VTM ALERT] lifecycle_phase reuse failed: {_lc_err}")

        if len(signals_fired) >= 3:
            return (
                f"{self.asset} {self.side}: {len(signals_fired)} signals against — "
                + ", ".join(signals_fired)
            )
        return None

    def _safe_send_alert(self, message: str) -> None:
        """Brain Rebuild Part 3.5. Best-effort — never raises into the tick loop."""
        if not self.telegram or not getattr(self.telegram, "is_running", False):
            return
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.telegram.send_notification(message))
        except Exception as e:
            logger.error(f"[VTM ALERT] Failed to send: {e}")

    def _calculate_adx(self) -> float:
        try:
            adx = talib.ADX(self.high, self.low, self.close, timeperiod=self.atr_period)
            return adx[-1]
        except Exception as e:
            logger.error(f"[VTM] ADX error: {e}")
            return 0.0

    def _calculate_atr_slow(self) -> float:
        try:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=100)
            return atr[-1]
        except Exception as e:
            logger.error(f"[VTM] ATR Slow error: {e}")
            return self.entry_price * 0.02


    def _blend_single_tp(self):
        """7.1 LIVE. Collapse the target candidates already computed into ONE
        expectancy-weighted TP, conservative-biased for fill probability.
        Forward-compatible: this candidate dict seeds a real [TP1,TP2,TP3] ladder
        when partials turn on. Returns None if candidates cannot be built."""
        try:
            _tm   = self.risk_config.get("trade_management", {})
            w     = _tm.get("single_tp_weights",
                            {"struct": 0.30, "mm": 0.25, "fib": 0.15, "rmult": 0.15, "mfe": 0.15})
            risk  = abs(self.entry_price - self.initial_stop_loss) if self.initial_stop_loss else 0.0
            if risk <= 0:
                return None
            sign  = 1.0 if self.side == "long" else -1.0
            cands = {}
            if self.take_profit_levels:
                cands["struct"] = self.take_profit_levels[0]
            _anchor = (getattr(self, "livermore_anchor_main_up",   None) if self.side == "long"
                       else getattr(self, "livermore_anchor_main_down", None))
            if _anchor:
                cands["mm"]  = float(_anchor)
                cands["fib"] = self.entry_price + sign * 1.272 * abs(self.entry_price - float(_anchor))
            _rm = (self.partial_targets or [1.5])[0]
            cands["rmult"] = self.entry_price + sign * _rm * risk
            _mfe_r = getattr(self, "_mfe_target_r", None)   # absent until 7.4 has >=30 rows; skipped safely
            if _mfe_r:
                cands["mfe"] = self.entry_price + sign * _mfe_r * risk
            if not cands:
                return None
            tot     = sum(w.get(k, 0.0) for k in cands) or 1.0
            blended = sum(w.get(k, 0.0) * v for k, v in cands.items()) / tot
            lo, hi  = min(cands.values()), max(cands.values())
            blended = max(lo, min(hi, blended))
            if _tm.get("single_tp_conservative_clamp", True):
                mid     = (lo + hi) / 2.0
                blended = min(blended, mid) if self.side == "long" else max(blended, mid)
            logger.info(
                f"[7.1] single-TP blend "
                f"{{ {', '.join(f'{k}:{round(v, 5)}' for k, v in cands.items())} }} "
                f"-> {blended:.5f}"
            )
            return blended
        except Exception as _e71:
            logger.debug(f"[7.1] _blend_single_tp failed (non-blocking): {_e71}")
            return None

    def cancel_take_profit(self):
        """Cancel all remaining take profit targets."""
        self.take_profit_levels = []
        self.partial_sizes = []
        logger.debug(f"[VTM] All take profit orders cancelled for {self.asset}.")

    def enable_trailing_stop(self):
        """Activate the trailing stop mechanism (Runner Mode)."""
        self.runner_activated = True
        logger.debug(f"[VTM] Trailing stop (Greed Mode) enabled for {self.asset}.")

    def _log_mfe_mae(self, exit_reason: str) -> None:
        """
        Phase 7.4 ext: records the Maximum Favorable/Adverse Excursion (peak
        unrealized profit/loss in R-multiples) reached during a trade's
        lifetime, alongside the reason it actually closed. Records only —
        never changes trading behaviour. Call sites only invoke this on a
        full close (remaining_position == 0 after the exit), so partial
        scale-outs do not produce a premature/incomplete MFE record.
        Gated by phase_config.mfe_logging_enabled.
        """
        if not self.risk_config.get("phase_config", {}).get("mfe_logging_enabled", False):
            return
        try:
            _risk = abs(self.entry_price - self.initial_stop_loss) or 1e-9
            if self.side == "long":
                _mfe_r = (self.highest_price_reached - self.entry_price) / _risk
                _mae_r = (self.entry_price - self.lowest_price_reached) / _risk
            else:
                _mfe_r = (self.entry_price - self.lowest_price_reached) / _risk
                _mae_r = (self.highest_price_reached - self.entry_price) / _risk
            import csv as _csv, os as _os, datetime as _dt
            _path = "logs/mfe_mae_log.csv"
            _os.makedirs("logs", exist_ok=True)
            _new_file = not _os.path.exists(_path)
            with open(_path, "a", newline="") as _f:
                _w = _csv.writer(_f)
                if _new_file:
                    _w.writerow(["ts", "asset", "setup", "side", "exit_reason",
                                 "mfe_r", "mae_r", "result_r"])
                _w.writerow([
                    _dt.datetime.now().isoformat(),
                    getattr(self, "asset", ""),
                    getattr(self, "trade_type", ""),
                    self.side,
                    exit_reason,
                    round(_mfe_r, 3),
                    round(_mae_r, 3),
                    getattr(self, "_realized_pnl_r", ""),
                ])
            logger.info(f"[7.4] MFE={_mfe_r:.2f}R MAE={_mae_r:.2f}R exit={exit_reason} -> {_path}")
        except Exception as _e74:
            logger.debug(f"[7.4] MFE log failed (non-blocking): {_e74}")

    def check_exit(self, current_price: float, atr_value: Optional[float] = None, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        # A13: hold _sl_lock for the whole tick so market_watcher.py's
        # cross-thread SL push (which acquires the same lock) can't
        # interleave with this method's own current_stop_loss writes.
        with self._sl_lock:
            return self._check_exit_locked(current_price, atr_value=atr_value, df_4h=df_4h)

    def _check_exit_locked(self, current_price: float, atr_value: Optional[float] = None, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        if atr_value is None:
            atr_value = self._calculate_atr() # Fallback if ATR not passed
        if self.remaining_position <= 0: return None

        # ── J: Friday PM trailing tightener ───────────────────────────────
        try:
            from datetime import datetime as _dtf
            # friday_tighten's only writer lived in dead code (get_aggregated_signal),
            # so this always fell through to the local check anyway. Own it outright.
            _is_friday_pm = (_dtf.utcnow().weekday() == 4 and _dtf.utcnow().hour >= 15)
            if _is_friday_pm:
                if hasattr(self, 'runner_trail_distance') and self.runner_trail_distance:
                    self.runner_trail_distance *= 0.6  # 40% tighter on Friday PM
                if hasattr(self, 'runner_trail_atr_multiplier'):
                    self.runner_trail_atr_multiplier = min(self.runner_trail_atr_multiplier, 1.2)
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────

        # --- STEP 1: Volatility Break-Even Lock ---
        # Reason: Locks risk to zero once trade proves itself by moving 1.0 * ATR in profit.
        if self.side == "long":
            current_profit = current_price - self.entry_price
        else:
            current_profit = self.entry_price - current_price

        # --- STEP 0.25: Soft Risk Reduction (first profit confirmation) ---
        # Fires at 0.75×ATR profit — trade has moved meaningfully in our favour but
        # hasn't proven full momentum yet. SL shifts to entry − 0.75×initial_risk,
        # cutting risk by 25% while leaving ample room to breathe.
        # Fills the gap between entry and the 1.0×ATR intermediate trail so a
        # reversal after an early push exits at a reduced loss, not full risk.
        _initial_risk = abs(self.entry_price - self.initial_stop_loss) if self.initial_stop_loss is not None else atr_value
        if current_profit > 0.75 * atr_value:
            if self.side == "long":
                _soft_sl = self.entry_price - 0.75 * _initial_risk
                if _soft_sl > self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔰 Soft risk-cut: {self.asset} SL → {_soft_sl:,.5f} "
                        f"(profit ${current_profit:.4g} > 0.75×ATR ${0.75*atr_value:.4g}, risk −25%)"
                    )
                    self.current_stop_loss = _soft_sl
            else:
                _soft_sl = self.entry_price + 0.75 * _initial_risk
                if _soft_sl < self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔰 Soft risk-cut: {self.asset} SL → {_soft_sl:,.5f} "
                        f"(profit ${current_profit:.4g} > 0.75×ATR ${0.75*atr_value:.4g}, risk −25%)"
                    )
                    self.current_stop_loss = _soft_sl

        # --- STEP 0.5: Intermediate Trail (Early Protection) ---
        # Fires when profit exceeds 1.0×ATR — trade has proven itself with a full
        # ATR of room. SL moves to entry − (initial_risk − 0.5×ATR), reducing risk
        # by ~half while leaving 1.5×ATR of breathing room from current price.
        if current_profit > 1.0 * atr_value:
            if self.side == "long":
                _intermediate_sl = self.entry_price - max(0.0, _initial_risk - 0.5 * atr_value)
                if _intermediate_sl > self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔒 Intermediate trail: {self.asset} SL → {_intermediate_sl:,.5f} "
                        f"(Profit: ${current_profit:.2f} > 1.0×ATR: ${1.0*atr_value:.2f})"
                    )
                    self.current_stop_loss = _intermediate_sl
            else:
                _intermediate_sl = self.entry_price + max(0.0, _initial_risk - 0.5 * atr_value)
                if _intermediate_sl < self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔒 Intermediate trail: {self.asset} SL → {_intermediate_sl:,.5f} "
                        f"(Profit: ${current_profit:.2f} > 1.0×ATR: ${1.0*atr_value:.2f})"
                    )
                    self.current_stop_loss = _intermediate_sl

        # Phase 4: break-even trigger is Livermore-aware (set in __init__).
        # Young state (age < 5) = 1.8×ATR; aged state (>20) = 1.2×ATR; default 1.5×.
        _be_trigger = getattr(self, "breakeven_atr_trigger", 1.5)
        if current_profit > _be_trigger * atr_value:
            # T1.4 fix: only move SL TO entry if it hasn't already passed entry.
            # Original code fired every tick with no side check, pulling a trailing
            # stop BACKWARDS to entry even after it had advanced beyond it.
            if self.side == "long" and self.current_stop_loss < self.entry_price:
                logger.info(
                    f"[VTM] 🛡️ Break-even lock: {self.asset} "
                    f"(Profit: ${current_profit:.2f} > {_be_trigger:.1f}×ATR: "
                    f"${_be_trigger*atr_value:.2f})"
                )
                self.current_stop_loss = self.entry_price
            elif self.side == "short" and self.current_stop_loss > self.entry_price:
                logger.info(
                    f"[VTM] 🛡️ Break-even lock: {self.asset} "
                    f"(Profit: ${current_profit:.2f} > {_be_trigger:.1f}×ATR: "
                    f"${_be_trigger*atr_value:.2f})"
                )
                self.current_stop_loss = self.entry_price

        # --- STEP 1.5: Time-Based Break-Even Lock ---
        # Fires independently of ATR: if the trade has been open for N bars and
        # pnl >= threshold, lock SL to break-even. Gated by breakeven_after_bars config.
        if self.breakeven_after_bars is not None and self.bars_in_trade >= self.breakeven_after_bars:
            _tbe_pnl = (
                (current_price - self.entry_price) / self.entry_price
                if self.side == "long"
                else (self.entry_price - current_price) / self.entry_price
            )
            if _tbe_pnl >= self.breakeven_profit_threshold:
                if self.side == "long" and self.current_stop_loss < self.entry_price:
                    logger.info(
                        f"[VTM] ⏱ Time-based BE lock: {self.asset} bar={self.bars_in_trade} "
                        f"pnl={_tbe_pnl:.2%} >= {self.breakeven_profit_threshold:.2%}"
                    )
                    self.current_stop_loss = self.entry_price
                elif self.side == "short" and self.current_stop_loss > self.entry_price:
                    logger.info(
                        f"[VTM] ⏱ Time-based BE lock: {self.asset} bar={self.bars_in_trade} "
                        f"pnl={_tbe_pnl:.2%} >= {self.breakeven_profit_threshold:.2%}"
                    )
                    self.current_stop_loss = self.entry_price

        # Calculate ADX and atr_slow once — used by multiple steps below.
        adx_value = self._calculate_adx()
        atr_slow = self._calculate_atr_slow()
        # Guard: atr_slow may be NaN when trade has < 100 bars of history.
        # Fall back to atr_value (fast ATR) so greed-mode comparisons don't silently fail.
        if np.isnan(atr_slow) or atr_slow == 0:
            atr_slow = atr_value

        # --- STEP 2: Stop-Loss Check (HIGHEST PRIORITY after BE locks) ---
        # Must be evaluated before pyramiding / early-scale returns so that a bar
        # which closes below the SL and also happens to meet pyramid conditions is
        # always treated as a stop-loss, never as a scale-in signal.
        try:
            # current_stop_loss is None on freshly-imported positions (set to initial on first trail tick).
            # Fall back to initial_stop_loss so the SL check is never skipped.
            if self.current_stop_loss is None:
                self.current_stop_loss = self.initial_stop_loss
            if self.current_stop_loss is None:
                logger.warning("[VTM] SL check skipped — both current and initial stop_loss are None")
            elif (self.side == "long" and current_price <= self.current_stop_loss) or \
               (self.side == "short" and current_price >= self.current_stop_loss):
                reason = ExitReason.STOP_LOSS
                offset = 0.125 * atr_value
                if self.side == "long":
                    if self.current_stop_loss > self.entry_price + offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                else:
                    if self.current_stop_loss < self.entry_price - offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                self._log_mfe_mae(reason.value)
                return {"reason": reason, "price": current_price, "size": self.remaining_position}
        except Exception as e:
            logger.error(f"[VTM] SL check error: {e}")

        # ══════════════════════════════════════════════════════════════════
        # SMART MARKET-CONDITION EXITS  (Steps 2.5 – 2.7)
        # Fire AFTER the hard SL (highest priority) but BEFORE mechanical TPs,
        # so deteriorating market conditions are caught before price grinds to
        # the original stop.  Each check is one-shot (gate flag prevents repeat).
        # ══════════════════════════════════════════════════════════════════

        # --- STEP 2.5: Volatility Spike Exit ---
        # If ATR has suddenly doubled vs its 100-bar baseline the entire risk model
        # used at entry is now wrong.  The SL is too tight for the new noise level,
        # and a continued adverse move could be far larger than anticipated.
        # Action: take 75 % of the remaining position off immediately; keep a 25 %
        # runner so we don't fully exit a trade that might still be going our way.
        # Guard: skip if trade is already up > 2× ATR (it's proving itself).
        if not getattr(self, "_vol_spike_exited", False):
            try:
                is_winning = (
                    (self.side == "long"  and current_price > self.entry_price + 2 * atr_value) or
                    (self.side == "short" and current_price < self.entry_price - 2 * atr_value)
                )
                if atr_value > 2.0 * atr_slow and not is_winning:
                    if not (self.partials_enabled and self._can_partial(0.75)):
                        logger.debug(f"[VTM] Vol Spike partial suppressed — letting trade run.")
                    else:
                        self._vol_spike_exited = True
                        vol_exit_size = min(0.75, self.remaining_position)
                        if vol_exit_size > 0:
                            self.remaining_position = max(0.0, self.remaining_position - vol_exit_size)
                        logger.warning(
                            f"[VTM] ⚡ VOLATILITY SPIKE: {self.asset} — "
                            f"ATR {atr_value:.5f} > 2× slow-ATR {atr_slow:.5f}. "
                            f"Reducing {vol_exit_size:.0%}, keeping runner."
                        )
                        if self.remaining_position <= 1e-9:
                            self._log_mfe_mae("VOLATILITY_SPIKE")
                        return {"reason": ExitReason.VOLATILITY_SPIKE, "price": current_price, "size": vol_exit_size}
            except Exception as _e:
                logger.debug(f"[VTM] Vol-spike check skipped: {_e}")

        # --- STEP 2.6: Reversal Candle Exit ---
        # A bar whose range > 1.5× ATR that closes in the lower 40 % of its own
        # range on a long (upper 40 % on a short) is the price-action equivalent
        # of a counter-signal: conviction reversed and fast.
        # Action: close 50 %, tighten remaining SL to entry ± 0.3×ATR.
        # Only fires after bar 2 (needs at least one prior close to compare).
        if not getattr(self, "_reversal_candle_exited", False) and len(self.close) >= 3 and self.bars_in_trade >= 2:
            try:
                bar_range  = self.high[-1] - self.low[-1]
                bar_mid    = (self.high[-1] + self.low[-1]) / 2
                prev_close = self.close[-2]

                bearish_reversal = (
                    self.side == "long"
                    and bar_range > 1.5 * atr_value          # wide, high-conviction bar
                    and current_price < bar_mid               # closes in lower half
                    and current_price < prev_close            # closes below prior close
                )
                bullish_reversal = (
                    self.side == "short"
                    and bar_range > 1.5 * atr_value
                    and current_price > bar_mid               # closes in upper half
                    and current_price > prev_close
                )

                if bearish_reversal or bullish_reversal:
                    if not (self.partials_enabled and self._can_partial(0.50)):
                        logger.debug(f"[VTM] Reversal Candle partial suppressed — letting trade run.")
                    else:
                        self._reversal_candle_exited = True
                        rev_size = min(0.50, self.remaining_position)
                    if rev_size > 0:
                        self.remaining_position = max(0.0, self.remaining_position - rev_size)
                        # Tighten SL on the runner to 0.8×ATR inside break-even
                        # (was 0.3×ATR — too close to entry, often hit by normal noise)
                        if self.side == "long":
                            tight_sl = self.entry_price - 0.8 * atr_value
                            if tight_sl > self.current_stop_loss:
                                self.current_stop_loss = tight_sl
                        else:
                            tight_sl = self.entry_price + 0.8 * atr_value
                            if tight_sl < self.current_stop_loss:
                                self.current_stop_loss = tight_sl
                        logger.warning(
                            f"[VTM] 🕯️ REVERSAL CANDLE: {self.asset} {self.side.upper()} — "
                            f"Range={bar_range:.5f} (1.5×ATR={1.5*atr_value:.5f}), "
                            f"close={current_price:.5f} vs mid={bar_mid:.5f}. "
                            f"Closing {rev_size:.0%}, SL tightened to ${self.current_stop_loss:,.5f}."
                        )
                        if self.remaining_position <= 1e-9:
                            self._log_mfe_mae("REVERSAL_CANDLE")
                        return {"reason": ExitReason.REVERSAL_CANDLE, "price": current_price, "size": rev_size}
            except Exception as _e:
                logger.debug(f"[VTM] Reversal-candle check skipped: {_e}")

        # --- STEP 2.7: Trend Invalidation Exit ---
        # 3 consecutive bars closing against the trade direction AND ADX < 20
        # means the market has lost its trend entirely — the edge that opened this
        # trade no longer exists.  Close the full remaining position rather than
        # waiting for the original SL to be hit bar by bar.
        # Guard: only fires when we have ≥ 4 bars (need 3 prior closes to compare)
        # and the position has been open at least 3 bars to avoid day-1 noise.
        if not getattr(self, "_trend_invalidated", False) and len(self.close) >= 5 and self.bars_in_trade >= 3:
            try:
                bars_against = sum(
                    1 for k in range(-3, 0)
                    if (self.side == "long"  and self.close[k] < self.close[k - 1]) or
                    (self.side == "short" and self.close[k] > self.close[k - 1])
                )
                if bars_against >= 3 and adx_value < 20:
                    # STRUCTURAL GATE: require the Livermore state to also confirm
                    # structural weakness, not just statistical weakness.
                    # For longs: SECONDARY_RETRACEMENT or MAIN_DOWN = structural
                    #            warning against the position (more than normal pullback).
                    # For shorts: SECONDARY_REBOUND or MAIN_UP = structural warning.
                    # If Livermore state is NATURAL_RETRACEMENT / MAIN_UP (for longs),
                    # the three lower closes are normal breathing and should not close.
                    _lv4h = getattr(self, "livermore_state_4h", None)
                    _struct_against = (
                        (self.side == "long"  and _lv4h in ("SECONDARY_RETRACEMENT", "MAIN_DOWN")) or
                        (self.side == "short" and _lv4h in ("SECONDARY_REBOUND",      "MAIN_UP"))
                    )
                    if not _struct_against:
                        logger.debug(
                            "[VTM] Trend invalidation: 3 bars against + ADX<20 "
                            "but Livermore state %s does not confirm — holding. "
                            "Structural stop is the protection.",
                            _lv4h,
                        )
                    else:
                        self._trend_invalidated = True
                        ti_size = self.remaining_position
                        if ti_size > 0:
                            self.remaining_position = 0.0
                            logger.warning(
                                "[VTM] ❌ TREND INVALIDATION: %s %s — "
                                "3 bars against + ADX=%.1f < 20 + Livermore=%s. "
                                "Full close (%.0f%%).",
                                self.asset, self.side.upper(),
                                adx_value, _lv4h, ti_size * 100,
                            )
                            self._log_mfe_mae("TREND_INVALIDATION")
                            return {
                                "reason": ExitReason.TREND_INVALIDATION,
                                "price":  current_price,
                                "size":   ti_size,
                            }
            except Exception as _e:
                logger.debug(f"[VTM] Trend-invalidation check skipped: {_e}")

        # --- STEP 2.8: Counter-Momentum Early Cut ---
        # Detects strong opposing momentum building AGAINST the trade while it is
        # still a loser (i.e. before the trailing / break-even machinery can help).
        # This is the only VTM mechanism that fires on a losing trade.
        #
        # Conditions (all must hold):
        #   1. Trade is in a loss AND loss > 0.4×ATR  — meaningful move against, not noise
        #   2. Loss has worsened vs last check (price still moving away)
        #   3. RSI shows counter-direction momentum: for shorts RSI > 55 (bulls in control)
        #                                            for longs  RSI < 45 (bears in control)
        #   4. MACD histogram is pointing counter-direction (bullish for shorts, bearish for longs)
        #   5. bars_in_trade >= 2 — at least one full bar has closed (avoids entry-bar noise)
        #
        # Action: close 60 % immediately and tighten remaining SL to just beyond
        # entry (0.5×ATR) so the runner can only lose a small additional amount.
        # One-shot guard (_counter_momentum_cut) prevents repeat fires per position.
        if (not getattr(self, "_counter_momentum_cut", False)
                and self.bars_in_trade >= 2
                and len(self.close) >= 26):
            try:
                _loss = (
                    (self.entry_price - current_price)   # short: profit is negative when price rises
                    if self.side == "short"
                    else (current_price - self.entry_price)  # long: profit is negative when price falls
                )
                _loss = -_loss  # positive = trade is losing

                if _loss > 0.4 * atr_value:
                    # RSI — use historical closes (updated each 1H bar by main loop)
                    _rsi_series = talib.RSI(self.close, timeperiod=14)
                    _rsi = _rsi_series[-1] if not np.isnan(_rsi_series[-1]) else 50.0

                    # MACD histogram direction
                    _, _, _macd_hist = talib.MACD(
                        self.close, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    _h1 = _macd_hist[-1] if not np.isnan(_macd_hist[-1]) else 0.0
                    _h2 = _macd_hist[-2] if not np.isnan(_macd_hist[-2]) else 0.0

                    # Counter-momentum signals (from the perspective of "bears taking over" for shorts
                    # or "bulls taking over" for longs)
                    if self.side == "short":
                        _rsi_counter   = _rsi > 55          # bulls in control
                        _macd_counter  = _h1 > _h2 > 0      # histogram rising into positive = bullish
                    else:
                        _rsi_counter   = _rsi < 45          # bears in control
                        _macd_counter  = _h1 < _h2 < 0      # histogram falling into negative = bearish

                    if _rsi_counter and _macd_counter:
                        # STRUCTURAL GATE: same Livermore state check as Trend Invalidation.
                        # Statistical momentum signals alone should not override the
                        # structural stop on a position entered at a proven structural
                        # level. Both statistical AND structural must confirm before cutting.
                        _lv4h_cm = getattr(self, "livermore_state_4h", None)
                        _struct_against_cm = (
                            (self.side == "long"  and _lv4h_cm in ("SECONDARY_RETRACEMENT", "MAIN_DOWN")) or
                            (self.side == "short" and _lv4h_cm in ("SECONDARY_REBOUND",      "MAIN_UP"))
                        )
                        if not _struct_against_cm:
                            logger.debug(
                                "[VTM] Counter-momentum signals present but Livermore=%s "
                                "does not structurally confirm — letting structural stop "
                                "manage this trade.",
                                _lv4h_cm,
                            )
                        else:
                            if not (self.partials_enabled and self._can_partial(0.60)):
                                logger.debug(f"[VTM] Counter-momentum partial suppressed — letting trade run.")
                            else:
                                self._counter_momentum_cut = True
                                cut_size = min(0.60, self.remaining_position)
                                if cut_size > 0:
                                    self.remaining_position = max(0.0, self.remaining_position - cut_size)
                            # Tighten remaining SL to entry ± 0.5×ATR so runner risk is minimal
                            if self.side == "short":
                                _tight_sl = self.entry_price + 0.5 * atr_value
                                if _tight_sl < self.current_stop_loss:
                                    self.current_stop_loss = _tight_sl
                            else:
                                _tight_sl = self.entry_price - 0.5 * atr_value
                                if _tight_sl > self.current_stop_loss:
                                    self.current_stop_loss = _tight_sl
                            logger.warning(
                                f"[VTM] ⚔️ COUNTER-MOMENTUM CUT: {self.asset} {self.side.upper()} — "
                                f"Loss=${_loss:.2f} ({_loss/atr_value:.2f}×ATR), "
                                f"RSI={_rsi:.1f}, MACD hist {_h1:+.4f}. "
                                f"Closing {cut_size:.0%} early. SL tightened to ${self.current_stop_loss:,.5f}."
                            )
                            if self.remaining_position <= 1e-9:
                                self._log_mfe_mae("COUNTER_MOMENTUM_CUT")
                            return {"reason": ExitReason.TREND_INVALIDATION,
                                    "price": current_price, "size": cut_size}
            except Exception as _e:
                logger.debug(f"[VTM] Counter-momentum cut check skipped: {_e}")

        # --- STEP 2.9: Profit Guard Exit ---
        # A human trader watching a profitable trade recognises the moment momentum
        # has turned and exits rather than waiting for the SL to be hit bar by bar.
        # This step does exactly that: once the trade has built meaningful profit
        # (> 0.8×ATR), and the MACD histogram crosses zero AGAINST the trade
        # direction while RSI confirms that the opposing side has taken control,
        # close the full remaining position immediately.
        #
        # Why this threshold?
        #   • MACD zero-cross is a definitive, non-lagging momentum flip signal.
        #   • RSI < 50 for longs / > 50 for shorts confirms bears/bulls are now
        #     in session control — not just a single noisy bar.
        #   • 0.8×ATR profit floor means the bot does NOT exit on the first sign
        #     of a wiggle; it waits until there is real profit worth protecting.
        #
        # Distinct from Step 5.5 (Momentum Exhaustion) which requires RSI > 75
        # and 5+ bars — too late for most 1H setups on small accounts.
        if (not getattr(self, "_profit_reversal_exited", False)
                and len(self.close) >= 26
                and self.bars_in_trade >= 2):
            try:
                _profit_guard = (
                    (current_price - self.entry_price) if self.side == "long"
                    else (self.entry_price - current_price)
                )
                if _profit_guard > 0.8 * atr_value:
                    _rsi_pg   = talib.RSI(self.close, timeperiod=14)
                    _, _, _hist_pg = talib.MACD(
                        self.close, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    _rsi_now = _rsi_pg[-1] if not np.isnan(_rsi_pg[-1]) else 50.0
                    _h1_pg   = _hist_pg[-1] if not np.isnan(_hist_pg[-1]) else 0.0
                    _h2_pg   = _hist_pg[-2] if not np.isnan(_hist_pg[-2]) else 0.0

                    # MACD histogram zero-cross against the trade
                    _macd_flipped = (
                        (self.side == "long"  and _h2_pg > 0 and _h1_pg < 0) or
                        (self.side == "short" and _h2_pg < 0 and _h1_pg > 0)
                    )
                    # RSI confirms opposing side in control
                    _rsi_confirms = (
                        (self.side == "long"  and _rsi_now < 50) or
                        (self.side == "short" and _rsi_now > 50)
                    )

                    if _macd_flipped and _rsi_confirms:
                        # STRUCTURAL ACTION: tighten the structural trail aggressively
                        # rather than closing the full position on a statistical signal.
                        # MACD zero-cross + RSI < 50 is a warning, not a structural
                        # confirmation that the thesis has broken. The trade continues
                        # with a tighter stop. If structure subsequently breaks (swing
                        # low is hit), the structural trail will exit it correctly.
                        self._profit_reversal_exited = True
                        _pg_struct = self._compute_swing_low_trail(atr_value)
                        if _pg_struct is not None:
                            if self.side == "long" and _pg_struct > self.current_stop_loss:
                                self.current_stop_loss = _pg_struct
                                logger.warning(
                                    "[VTM] 💰 PROFIT GUARD: %s %s — MACD flipped, RSI=%.1f. "
                                    "Statistical warning: tightening SL to structural "
                                    "swing low %.5g (not closing — structure not yet broken).",
                                    self.asset, self.side.upper(), _rsi_now, _pg_struct,
                                )
                            elif self.side == "short" and _pg_struct < self.current_stop_loss:
                                self.current_stop_loss = _pg_struct
                                logger.warning(
                                    "[VTM] 💰 PROFIT GUARD: %s %s — MACD flipped, RSI=%.1f. "
                                    "Statistical warning: tightening SL to structural "
                                    "swing high %.5g (not closing — structure not yet broken).",
                                    self.asset, self.side.upper(), _rsi_now, _pg_struct,
                                )
                        else:
                            # No structural swing to trail behind — keep current stop,
                            # log the warning, let structural stop or trail handle exit.
                            logger.warning(
                                "[VTM] 💰 PROFIT GUARD: %s %s — MACD flipped, RSI=%.1f. "
                                "No structural swing trail available — holding current SL %.5g. "
                                "Structural stop is the protection.",
                                self.asset, self.side.upper(), _rsi_now,
                                self.current_stop_loss,
                            )
            except Exception as _e:
                logger.debug(f"[VTM] Profit-guard check skipped: {_e}")

        # --- STEP 3: Greed Mode Accelerator ---
        # During extreme trends/volatility, collapse early targets so the runner
        # trail captures the full move. One-shot: _greed_mode_activated prevents
        # re-executing every bar, avoiding log spam and repeated partial_sizes mutation.
        if not getattr(self, "_greed_mode_activated", False):
            if adx_value > 40 and atr_value > (1.5 * atr_slow):
                if len(self.take_profit_levels) > 1:
                    # Keep TP1 as a 30% partial lock — it gives certainty while the
                    # runner chases the full move.  Only TP2+ collapse into the runner.
                    logger.info(
                        f"[VTM] 🔥 GREED MODE: Strong trend (ADX:{adx_value:.1f}) & "
                        f"Volatility Expansion detected. Keeping TP1 (30%), collapsing rest to runner."
                    )
                    self.take_profit_levels = [self.take_profit_levels[0], self.take_profit_levels[-1]]
                    self.partial_sizes = [0.30, 0.70]
                    self._greed_mode_activated = True

        # --- STEP 3.5: Early Scale Exit ---
        # Objective: Lock in a small partial profit quickly in the first few bars before
        # momentum fades. Enabled via early_scale_enabled in per-asset risk config.
        if self.early_scale_enabled and not self._early_scaled:
            if self.bars_in_trade <= self.early_scale_bars:
                early_pnl_pct = (
                    (current_price - self.entry_price) / self.entry_price
                    if self.side == "long"
                    else (self.entry_price - current_price) / self.entry_price
                )
                if early_pnl_pct >= self.early_scale_threshold:
                    if not (self.partials_enabled and self._can_partial(0.20)):
                        # Suppressed — cannot subdivide this lot.
                        # Do NOT set _early_scaled, do NOT touch remaining_position,
                        # do NOT tighten SL here. Trade continues to real full-close exits.
                        logger.debug(
                            f"[VTM] Early Scale suppressed — "
                            f"partials_enabled={self.partials_enabled}, "
                            f"_can_partial(0.20)={self._can_partial(0.20)} — letting trade run."
                        )
                    else:
                        # Can subdivide — execute the partial and lock the stop.
                        self._early_scaled = True
                        early_size = 0.20

                        self.remaining_position = max(0.0, self.remaining_position - early_size)

                        # Tighten SL to lock in partial profit
                        lock_offset = self.early_lock_atr_multiplier * atr_value
                        if self.side == "long":
                            lock_sl = self.entry_price + lock_offset
                            if lock_sl > self.current_stop_loss:
                                logger.info(
                                    f"[VTM] ⚡ Early Scale SL lock: ${self.current_stop_loss:,.2f} → ${lock_sl:,.2f} "
                                    f"(entry + {self.early_lock_atr_multiplier}x ATR)"
                                )
                                self.current_stop_loss = lock_sl
                        else:
                            lock_sl = self.entry_price - lock_offset
                            if lock_sl < self.current_stop_loss:
                                logger.info(
                                    f"[VTM] ⚡ Early Scale SL lock: ${self.current_stop_loss:,.2f} → ${lock_sl:,.2f} "
                                    f"(entry - {self.early_lock_atr_multiplier}x ATR)"
                                )
                                self.current_stop_loss = lock_sl

                        logger.info(
                            f"[VTM] ⚡ EARLY SCALE: {self.asset} {self.side.upper()} — "
                            f"exiting {early_size:.0%} at ${current_price:,.2f} "
                            f"(bar {self.bars_in_trade}, pnl={early_pnl_pct:.2%})"
                        )
                        if self.remaining_position <= 1e-9:
                            self._log_mfe_mae("EARLY_SCALE")
                        return {"reason": ExitReason.EARLY_SCALE, "price": current_price, "size": early_size}

        # --- STEP 4: Trend Pyramiding ---
        # Objective: Scale into strong breakout trends. Fires only after SL is confirmed
        # safe (Step 2 above) so a fast reversal bar cannot be misclassified as a pyramid.
        if self.trade_type == "TREND" and not self.has_pyramided:
            if current_profit >= (1.0 * atr_value) and adx_value > 25:
                if not self._can_add_position():
                    # L11 fix: this branch previously only controlled which log
                    # line printed — the pyramid block below ran unconditionally
                    # regardless of _can_add_position()'s result. Now it actually
                    # gates the action, so the lot-size check and the new
                    # Livermore maturity gate inside _can_add_position() are
                    # finally binding instead of cosmetic.
                    logger.debug(f"[VTM] Pyramiding suppressed — pyramiding_enabled={self.pyramiding_enabled} "
                                 f"or position too small to add legal lot, or Livermore maturity gate blocked it.")
                else:
                    logger.info(f"[VTM] 🗼 TREND PYRAMIDING: Strong trend confirmed. Scaling in.")
                    # Move SL of position 1 to entry before adding exposure
                    self.current_stop_loss = self.entry_price
                    self.has_pyramided = True
                    return {
                        "action": "pyramid",
                        "asset": self.asset,
                        "side": self.side,
                        "new_size": self.position_size * 0.5,
                        "reason": "Trend Pyramiding Triggered"
                    }

        # --- STEP 5: Trade State Mutation ---
        # Objective: Allow profitable Mean Reversion trades to convert into trend trades.
        # Original trigger: ADX > 30 (statistical momentum confirmation).
        # Additional trigger: BOS detected on live composite_state (structural proof).
        # In NATURAL_RETRACEMENT — where Mode 1 spring entries happen — ADX is
        # typically below 30 because the pullback itself is not strong momentum.
        # Mutation was silently failing for most Mode 1 trades.
        # BOS = the market just made a higher high, breaking the local retracement
        # structure. That is Livermore's proof the pullback is over and the trend
        # is resuming. Either ADX OR BOS can now trigger mutation.
        if self.trade_type == "REVERSION":
            _bos_live = getattr(getattr(self, "_live_cs", None), "bos_detected", False)
            if (adx_value > 30 or _bos_live) and current_profit > 0:
                is_actually_profitable = (
                    (self.side == "long"  and current_price > self.entry_price) or
                    (self.side == "short" and current_price < self.entry_price)
                )
                if is_actually_profitable:
                    _trigger = "BOS structural break" if _bos_live else f"ADX={adx_value:.1f}"
                    logger.info(
                        "[VTM] 🧬 %s: Trade mutated REVERSION → TREND (%s). "
                        "Runner + structural trail now both active.",
                        self.asset, _trigger,
                    )
                    self.cancel_take_profit()
                    self.trade_type = "TREND"
                    self.enable_trailing_stop()

        # --- STEP 5.5: Momentum Exhaustion Exit ---
        # Three simultaneous conditions must hold:
        #   1. RSI is in the exhaustion zone (> 75 long / < 25 short) — price stretched
        #   2. MACD histogram declining 3 consecutive bars — momentum dying
        #   3. ADX falling over last 2 bars — trend weakening / losing steam
        # When all three align, the move is likely spent.  Close 50 % and lock the
        # runner to break-even so any remaining profit is protected, not gambled.
        # Guards: trade must be in profit and open ≥ 5 bars; needs 26 bars for MACD.
        if not getattr(self, "_momentum_exhausted", False) and len(self.close) >= 26 and self.bars_in_trade >= 5:
            try:
                _in_profit = (
                    (self.side == "long"  and current_price > self.entry_price) or
                    (self.side == "short" and current_price < self.entry_price)
                )
                if _in_profit:
                    rsi_arr  = talib.RSI(self.close, timeperiod=14)
                    _, _, macd_hist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
                    adx_arr  = talib.ADX(self.high, self.low, self.close, timeperiod=14)

                    rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else 50.0
                    rsi_exhausted = (self.side == "long" and rsi_val > 75) or \
                                    (self.side == "short" and rsi_val < 25)

                    h1, h2, h3 = macd_hist[-1], macd_hist[-2], macd_hist[-3]
                    macd_dying = (
                        not any(np.isnan(v) for v in [h1, h2, h3]) and (
                            (self.side == "long"  and h1 < h2 < h3) or
                            (self.side == "short" and h1 > h2 > h3)
                        )
                    )

                    adx_weakening = (
                        not np.isnan(adx_arr[-1]) and not np.isnan(adx_arr[-2])
                        and adx_arr[-1] < adx_arr[-2]
                    )

                    if rsi_exhausted and macd_dying and adx_weakening:
                        if not (self.partials_enabled and self._can_partial(0.50)):
                            logger.debug(f"[VTM] Momentum Exhaustion partial suppressed — letting trade run.")
                        else:
                            self._momentum_exhausted = True
                            exhaust_size = min(0.50, self.remaining_position)
                            if exhaust_size > 0:
                                self.remaining_position = max(0.0, self.remaining_position - exhaust_size)
                            # Structural trail on remainder instead of ATR breakeven lock.
                            # If a swing low/high has formed above/below the current stop,
                            # place the stop there — it is structurally more meaningful
                            # than locking to entry price regardless of market context.
                            # Fall back to entry price lock only if no structural level found.
                            _exhaust_struct = self._compute_swing_low_trail(atr)
                            if _exhaust_struct is not None:
                                if self.side == "long" and _exhaust_struct > self.current_stop_loss:
                                    self.current_stop_loss = _exhaust_struct
                                    logger.info(
                                        "[VTM] 📉 Momentum exhaustion remainder: "
                                        "SL → structural swing low %.5g",
                                        _exhaust_struct,
                                    )
                                elif self.side == "short" and _exhaust_struct < self.current_stop_loss:
                                    self.current_stop_loss = _exhaust_struct
                                    logger.info(
                                        "[VTM] 📉 Momentum exhaustion remainder: "
                                        "SL → structural swing high %.5g",
                                        _exhaust_struct,
                                    )
                            else:
                                # No structural swing formed yet — safety net
                                if self.side == "long" and self.current_stop_loss < self.entry_price:
                                    self.current_stop_loss = self.entry_price
                                elif self.side == "short" and self.current_stop_loss > self.entry_price:
                                    self.current_stop_loss = self.entry_price
                            logger.warning(
                                f"[VTM] 📉 MOMENTUM EXHAUSTION: {self.asset} {self.side.upper()} — "
                                f"RSI={rsi_val:.1f}, MACD hist declining, ADX={adx_arr[-1]:.1f}↓. "
                                f"Closing {exhaust_size:.0%}, SL locked to break-even ${self.entry_price:,.5f}."
                            )
                            if self.remaining_position <= 1e-9:
                                self._log_mfe_mae("MOMENTUM_EXHAUSTION")
                            return {"reason": ExitReason.MOMENTUM_EXHAUSTION, "price": current_price, "size": exhaust_size}
            except Exception as _e:
                logger.debug(f"[VTM] Momentum-exhaustion check skipped: {_e}")

        # --- STEP 6: Time Decay Protection (T4.2 — dynamic extension when in profit) ---
        # Objective: Prevent stale trades turning into long-term losses.
        # Extension rule: if the trade is in profit at the time-stop bar, grant ONE
        # +24-bar extension so a live winner is not forcefully closed.  A second
        # time-stop at bars_in_trade >= time_stop_bars + 24 closes unconditionally.
        if self.bars_in_trade >= self.time_stop_bars:
            _pnl_now = (
                (current_price - self.entry_price) / self.entry_price
                if self.side == "long"
                else (self.entry_price - current_price) / self.entry_price
            )
            _extended = getattr(self, "_time_stop_extended", False)

            if _pnl_now > 0 and not _extended:
                self._time_stop_extended = True
                logger.info(
                    f"[VTM] ⏳ Time stop reached for {self.asset} but trade is in profit "
                    f"({_pnl_now * 100:+.2f}%) — granting +24 bar extension "
                    f"(bars={self.bars_in_trade}, new_limit={self.time_stop_bars + 24})"
                )
            elif self.bars_in_trade < self.time_stop_bars + (24 if _extended else 0):
                pass  # still within extended window, no action
            else:
                logger.info(
                    f"[VTM] ⏳ Stale {self.trade_type} trade closed for {self.asset} "
                    f"(Bars: {self.bars_in_trade} >= "
                    f"{self.time_stop_bars + (24 if _extended else 0)}, "
                    f"pnl={_pnl_now * 100:+.2f}%)"
                )
                self._log_mfe_mae("TIME_STOP")
                return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}

        # --- STEP 7: TP Partial Exits ---
        try:
            _is_last_rung = lambda idx: idx == len(self.take_profit_levels) - 1
            for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes)):
                if i in self.partials_hit: continue
                if (self.side == "long" and current_price >= target) or (self.side == "short" and current_price <= target):
                    self.partials_hit.append(i)

                    # S10.2: Last rung — signal size=1.0 so handlers treat it as a
                    # full close regardless of the fractional partial_sizes value.
                    # This matters for Binance which has no partial-close path and
                    # gates on size < 1.0 to skip partials; the last rung must
                    # always pass that gate to exit cleanly.
                    # Also prevents floating-point dust on MT5 partial ladders.
                    if _is_last_rung(i):
                        size = 1.0   # full close — consume entire remaining position
                        self.remaining_position = 0.0
                    else:
                        self.remaining_position = max(0.0, self.remaining_position - size)

                    # After TP1 (first partial), attempt early runner promotion based on
                    # volume strength and candle conviction. Falls back to mechanical
                    # activation after TP2 if promotion conditions are not met.
                    if len(self.partials_hit) == 1 and not self.runner_activated and self.trade_type == "TREND":
                        promoted = self.check_promotion_to_runner(current_price)
                        if not promoted:
                            logger.info("[VTM] 🏃 TP1 hit — runner promotion skipped (conditions not met, waiting for TP2)")

                    # Mechanical fallback: activate runner after TP2
                    if len(self.partials_hit) >= 2 and not self.runner_activated and self.trade_type == "TREND":
                        self.runner_activated = True
                        logger.info(f"[VTM TACTICAL] 🏃 Runner Activated (mechanical — TP2 hit): trailing stop now follows price.")

                    tp_reasons = [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3]
                    reason = tp_reasons[i] if i < len(tp_reasons) else ExitReason.TAKE_PROFIT_3
                    if _is_last_rung(i):
                        self._log_mfe_mae(reason.value)
                    return {"reason": reason, "price": current_price, "size": size}

            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self, live_price: Optional[float] = None) -> Dict:
        current_price = live_price if live_price is not None else self.close[-1]
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100 if self.side == "long" else (self.entry_price - current_price) / self.entry_price * 100
        next_target_idx = len(self.partials_hit)
        next_target = self.take_profit_levels[next_target_idx] if next_target_idx < len(self.take_profit_levels) else None
        
        # Directional distance — always negative = risk / downside remaining to SL
        # LONG: SL is below current → negative value (price must fall to hit SL)
        # SHORT: SL is above current → negative value (price must rise to hit SL)
        if self.current_stop_loss > 0:
            if self.side == "long":
                distance_to_sl_pct = (self.current_stop_loss - current_price) / current_price * 100
            else:
                distance_to_sl_pct = (current_price - self.current_stop_loss) / current_price * 100
        else:
            distance_to_sl_pct = 0

        if next_target and current_price > 0:
            if self.side == "long":
                distance_to_tp_pct = (next_target - current_price) / current_price * 100
            else:
                distance_to_tp_pct = (current_price - next_target) / current_price * 100
        else:
            distance_to_tp_pct = 0
        
        return {
            "entry_price": self.entry_price,
            "current_price": current_price,
            "stop_loss": self.current_stop_loss,
            "initial_stop": self.initial_stop_loss,
            "take_profit": next_target,
            "all_targets": self.take_profit_levels,
            "profit_locked": self.profit_locked,
            "remaining_position_pct": self.remaining_position,
            "pnl_pct": pnl_pct,
            "update_count": self.bars_in_trade,
            "partials_hit": len(self.partials_hit),
            "runner_active": self.runner_activated,
            "highest_reached": self.highest_price_reached,
            "lowest_reached": self.lowest_price_reached,
            "side": self.side,
            "distance_to_sl_pct": distance_to_sl_pct,
            "distance_to_tp_pct": distance_to_tp_pct
        }

    def to_dict(self) -> Dict:
        return {
            "entry_price": self.entry_price,
            "side": self.side,
            "asset": self.asset,
            "position_size": self.position_size,
            "initial_stop_loss": self.initial_stop_loss,
            "current_stop_loss": self.current_stop_loss,
            "take_profit_levels": self.take_profit_levels,
            "partial_sizes": self.partial_sizes,
            "remaining_position": self.remaining_position,
            "partials_hit": self.partials_hit,
            "bars_in_trade": self.bars_in_trade,
            "highest_price_reached": self.highest_price_reached,
            "lowest_price_reached": self.lowest_price_reached,
            "runner_activated": self.runner_activated,
            "has_pyramided": self.has_pyramided,
            "trade_type": self.trade_type,
            "entry_time": self.entry_time.isoformat(),
            "local_free_margin": self.local_free_margin,
            "current_ask": self.current_ask,
            "current_bid": self.current_bid,
            # One-shot state flags — persisted so from_dict() restores them correctly
            "_greed_mode_activated": getattr(self, "_greed_mode_activated", False),
            "_early_scaled": getattr(self, "_early_scaled", False),
            "_time_stop_extended": getattr(self, "_time_stop_extended", False),
            "_counter_momentum_cut": getattr(self, "_counter_momentum_cut", False),
            # Snapshot of the ADX-adjusted partial targets used at open — needed so
            # from_dict() can pass them to risk_config and avoid recalculation drift
            "partial_targets_snapshot": list(self.partial_targets),
        }

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        # Pass a minimal risk_config that satisfies _calculate_initial_levels() without
        # triggering a lot-size ValueError. The position_size from state was already
        # validated when the trade was originally opened, so we set a min_lot of 0 and
        # bypass the leverage ceiling by leaving local_free_margin at 0.
        # All VTM state is fully overwritten from the stored dict immediately after.
        _restore_risk_config = {
            "partial_targets": state.get("partial_targets_snapshot", [1.0, 1.8, 3.0]),
            "partial_sizes": state.get("partial_sizes", [0.45, 0.30, 0.25]),
        }
        vtm = cls(
            entry_price=state["entry_price"],
            side=state["side"],
            asset=state["asset"],
            high=high,
            low=low,
            close=close,
            quantity=state["position_size"],
            trade_type=state.get("trade_type", "TREND"),
            risk_config=_restore_risk_config,
            local_free_margin=0.0,   # Suppress leverage ceiling during restore
            current_ask=state.get("current_ask", 0.0),
            current_bid=state.get("current_bid", 0.0),
            min_lot_override=0.0,    # Suppress lot-size ValueError during restore
        )
        # Overwrite all state from the persisted snapshot
        vtm.initial_stop_loss = state["initial_stop_loss"]
        vtm.current_stop_loss = state["current_stop_loss"]
        vtm.take_profit_levels = state["take_profit_levels"]
        vtm.partial_sizes = state["partial_sizes"]
        vtm.remaining_position = state["remaining_position"]
        vtm.partials_hit = state["partials_hit"]
        vtm.bars_in_trade = state["bars_in_trade"]
        vtm.highest_price_reached = state["highest_price_reached"]
        vtm.lowest_price_reached = state["lowest_price_reached"]
        vtm.runner_activated = state["runner_activated"]
        vtm.has_pyramided = state.get("has_pyramided", False)
        vtm._greed_mode_activated = state.get("_greed_mode_activated", False)
        vtm._early_scaled = state.get("_early_scaled", False)
        vtm._time_stop_extended = state.get("_time_stop_extended", False)
        vtm._counter_momentum_cut = state.get("_counter_momentum_cut", False)
        return vtm

    # ─────────────────────────────────────────────────────────────────────────
    # T3.2 — Manual Override Methods (Telegram /set_sl /set_tp /vtm_status)
    # ─────────────────────────────────────────────────────────────────────────

    def override_stop_loss(self, new_sl: float) -> str:
        """
        Manually override the current stop loss level via Telegram command.

        Validates the new SL is on the correct side of the CURRENT PRICE (not
        entry price) to allow profit-locking moves (e.g. setting a long SL
        above entry after a rally). Only rejects if the SL would trigger
        immediately against the current price.

        Returns a human-readable status string for the Telegram reply.
        """
        if new_sl <= 0:
            return f"❌ Invalid SL: {new_sl} — must be > 0"

        # Use current mark price for the sanity check; fall back to entry if
        # live price isn't available yet (position still being opened).
        current_price = getattr(self, "current_price", None) or self.entry_price

        # Reject only when the SL would fire against the current price
        # (i.e. it's already beyond where the market is right now).
        if self.side == "long" and new_sl >= current_price:
            return (
                f"❌ Rejected: SL {new_sl:.5f} is at or above current price "
                f"{current_price:.5f} — it would trigger immediately. "
                f"Use /set_tp to move take profit instead."
            )
        if self.side == "short" and new_sl <= current_price:
            return (
                f"❌ Rejected: SL {new_sl:.5f} is at or below current price "
                f"{current_price:.5f} — it would trigger immediately. "
                f"Use /set_tp to move take profit instead."
            )

        # A13: third writer of current_stop_loss (Telegram-triggered manual
        # override, its own async context) — same shared lock as the
        # market_watcher/check_exit pair.
        with self._sl_lock:
            old_sl = self.current_stop_loss
            self.current_stop_loss = new_sl
        logger.info(
            f"[VTM] 🖊️ Manual SL override: {self.asset} {self.side.upper()} "
            f"SL {old_sl:.5f} → {new_sl:.5f}"
        )
        direction = "tighter 🛡️" if (
            (self.side == "long" and new_sl > old_sl) or
            (self.side == "short" and new_sl < old_sl)
        ) else "looser ↔️"
        return (
            f"✅ SL updated ({direction})\n"
            f"  Asset : {self.asset} {self.side.upper()}\n"
            f"  Old SL: {old_sl:.5f}\n"
            f"  New SL: {new_sl:.5f}\n"
            f"  Entry : {self.entry_price:.5f}"
        )

    def override_take_profit(self, new_tp: float, target_index: int = 0) -> str:
        """
        Manually override a specific take profit level via Telegram command.

        target_index selects which TP tier to update (0 = first remaining TP,
        1 = second, etc.).  Defaults to the nearest unfilled TP.

        Returns a human-readable status string for the Telegram reply.
        """
        if new_tp <= 0:
            return f"❌ Invalid TP: {new_tp} — must be > 0"

        if self.side == "long" and new_tp <= self.entry_price:
            return (
                f"❌ Rejected: TP {new_tp:.5f} is below entry {self.entry_price:.5f} "
                f"for a LONG position."
            )
        if self.side == "short" and new_tp >= self.entry_price:
            return (
                f"❌ Rejected: TP {new_tp:.5f} is above entry {self.entry_price:.5f} "
                f"for a SHORT position."
            )

        # Find remaining (unhit) TP levels.
        # self.partials_hit is a list of hit *index integers* (e.g. [0, 1] means
        # TP0 and TP1 were hit). The old filter used self.partials_hit[i] as a bool
        # which is wrong — it returned another index integer, not a hit/unhit flag.
        remaining_indices = [
            i for i in range(len(self.take_profit_levels))
            if i not in self.partials_hit
        ]

        # Empty list: position has no TP at all (e.g. min-lot with partials cleared).
        # Instead of refusing, append the new price as the single exit target.
        if not remaining_indices:
            self.take_profit_levels = [new_tp]
            logger.info(
                f"[VTM] 🖊️ Manual TP added (was empty): {self.asset} {self.side.upper()} "
                f"→ {new_tp:.5f}"
            )
            return (
                f"✅ TP set to {new_tp:.5f} for {self.asset} {self.side.upper()}\n"
                f"(Position had no TP — added as single full-exit target)"
            )

        if target_index >= len(remaining_indices):
            target_index = 0  # fall back to nearest

        actual_idx = remaining_indices[target_index]
        old_tp = self.take_profit_levels[actual_idx]
        self.take_profit_levels[actual_idx] = new_tp

        logger.info(
            f"[VTM] 🖊️ Manual TP override: {self.asset} {self.side.upper()} "
            f"TP[{actual_idx}] {old_tp:.5f} → {new_tp:.5f}"
        )
        return (
            f"✅ TP[{actual_idx + 1}] updated\n"
            f"  Asset : {self.asset} {self.side.upper()}\n"
            f"  Old TP: {old_tp:.5f}\n"
            f"  New TP: {new_tp:.5f}\n"
            f"  Entry : {self.entry_price:.5f}"
        )

    def get_override_status(self) -> dict:
        """
        Return current trade levels for Telegram /vtm_status display.

        Returns a flat dict so the Telegram handler can format it freely.
        """
        levels = self.get_current_levels()
        remaining_tps = [
            round(tp, 5)
            for i, tp in enumerate(self.take_profit_levels)
            if i >= len(self.partials_hit) or not self.partials_hit[i]
        ]
        hit_tps = len(self.partials_hit) if hasattr(self, "partials_hit") else 0

        return {
            "asset":          self.asset,
            "side":           self.side.upper(),
            "entry_price":    round(self.entry_price, 5),
            "current_price":  round(levels.get("current_price", 0.0), 5),
            "stop_loss":      round(self.current_stop_loss, 5),
            "initial_sl":     round(self.initial_stop_loss, 5),
            "remaining_tps":  remaining_tps,
            "tps_hit":        hit_tps,
            "bars_in_trade":  self.bars_in_trade,
            "pnl_pct":        round(levels.get("pnl_pct", 0.0), 3),
            "remaining_pct":  round(
                self.remaining_position / self.position_size * 100
                if self.position_size > 0 else 0.0, 1
            ),
            "trade_type":     getattr(self, "trade_type", "UNKNOWN"),
            "runner_active":  getattr(self, "runner_activated", False),
        }

    def _compute_structural_stop(self, atr: float) -> "Optional[float]":
        """
        Phase 4 — Structural Stop Router.
        Returns a level-anchored stop loss price, or None if entry_type does
        not provide a structural reference (falls back to ATR baseline).

        Stop placement by entry_type:
          SPRING_ENTRY     — Below/above the sweep_level (liquidity grab reversal).
                             Invalidation = price returning through the swept level.
          RANGE_BOUNDARY   — Behind the defended nearby_4h_level.
                             Invalidation = level giving way.
          TREND_FOLLOWING  — Behind the Livermore anchor just broken.
                             Invalidation = price returning back below/above the breakout.
          MR_PULLBACK      — Ensure stop clears the nearby_4h_level (level must sit
                             between stop and entry; don't place stop inside the level).
          CONTINUATION     — Behind the 1H consolidation box that was broken
                             (vtm_range_low/high — Phase 4 ext, gated by
                             phase_config.continuation_targets_enabled).
          REJECT / None    — Return None → caller keeps ATR baseline.
        """
        _phase_cfg = self.risk_config.get("phase_config", {})
        if not _phase_cfg.get("structural_stops_enabled", False):
            return None

        entry_type = getattr(self, "vtm_entry_type", None)
        side       = self.side          # "long" or "short"
        entry      = self.entry_price

        if entry_type is None or entry_type == "REJECT":
            return None

        if entry_type == "SPRING_ENTRY":
            # Stop just beyond the sweep level (the wick tip defines invalidation).
            sweep = getattr(self, "sweep_level", None)
            if sweep is not None and atr > 0:
                buf = 0.3 * atr
                return (sweep - buf) if side == "long" else (sweep + buf)
            return None

        if entry_type == "RANGE_BOUNDARY":
            # Stop behind the defended 4H level.
            level = getattr(self, "nearby_4h_level", None)
            if level is not None and atr > 0:
                buf = 0.5 * atr
                return (level - buf) if side == "long" else (level + buf)
            return None

        if entry_type == "TREND_FOLLOWING":
            # Stop behind the Livermore anchor that price just broke through.
            # Reject the anchor if it is more than max_stop_atr_mult ATRs from
            # entry — at that distance the clamp would discard it anyway and the
            # stop would be the ATR cap wearing a structural label.
            _max_atr = float(self.risk_config.get("max_stop_atr_mult", 5.0))
            if side == "long":
                anchor = getattr(self, "livermore_anchor_main_up", None)
                if anchor is None:
                    anchor = getattr(self, "livermore_anchor_nat_low", None)
            else:
                anchor = getattr(self, "livermore_anchor_main_down", None)
                if anchor is None:
                    anchor = getattr(self, "livermore_anchor_nat_high", None)
            if anchor is not None and atr > 0:
                dist_atr = abs(entry - anchor) / atr
                if dist_atr > _max_atr:
                    logger.debug(
                        f"[VTM] TREND_FOLLOWING anchor {anchor:.5f} is "
                        f"{dist_atr:.1f}× ATR from entry — too far, "
                        f"falling back to ATR baseline"
                    )
                    return None
                buf = 0.5 * atr
                return (anchor - buf) if side == "long" else (anchor + buf)
            return None

        if entry_type == "MR_PULLBACK":
            # Ensure the stop is BEYOND the nearby_4h_level (level stays between stop
            # and entry — otherwise the stop is sitting inside the "active zone").
            level = getattr(self, "nearby_4h_level", None)
            if level is not None and atr > 0:
                buf = 0.5 * atr
                return (level - buf) if side == "long" else (level + buf)
            return None

        if entry_type == "CONTINUATION":
            # Stop behind the 1H consolidation box that was broken (the box
            # is populated on RetestResult only for the CONTINUATION tier;
            # NOT the same box used by the RANGE target ladder).
            level = self.vtm_range_low if side == "long" else self.vtm_range_high
            if level is not None and atr > 0:
                buf = 0.3 * atr
                return (level - buf) if side == "long" else (level + buf)
            return None

        # Item 2.14: no entry_type matched any structural reference above —
        # this trade has no real level to place a stop behind. A plain
        # distance-based (ATR) stop only protects a trade that was strong
        # enough to justify it; a weak, unanchored trade gets flagged for
        # immediate close instead of quietly falling back to a weak stop.
        # A stop must still be returned (None → caller keeps the ATR
        # baseline) since the position needs SOME protection while the
        # emergency close executes.
        # Independently flag-gated (default False) rather than piggybacking on
        # structural_stops_enabled (already True live) — auto-closing a
        # just-filled position is new, unvalidated, real-money behavior and
        # deserves its own soak period like every other Phase 4 feature here.
        if _phase_cfg.get("emergency_close_unanchored_weak_signal_enabled", False):
            _total_score = self.signal_details.get("total_score")
            _required_score = self.signal_details.get("required_score")
            if _total_score is not None and _required_score is not None:
                if _total_score < (_required_score + 1.0):
                    self.emergency_close_requested = True
                    logger.warning(
                        f"[VTM] {self.asset}: no structural stop reference and score "
                        f"{_total_score:.2f} < {_required_score + 1.0:.2f} — flagging "
                        f"for emergency close instead of a weak distance-based stop."
                    )
        return None

    def _compute_swing_low_trail(self, atr: float) -> Optional[float]:
        """
        Structural swing low trail — Option 1 (Pure Livermore).

        Scans accumulated OHLC bars for the most recent confirmed structural
        swing low (long) or swing high (short) that sits above the current
        stop loss. Returns a candidate stop just below that level.

        Pivot quality: 4 bars committed on the left (proves the level was
        real), 2 bars rejection on the right (catches the turn early).
        ATR depth filter: the swing must travel at least 0.3×ATR from prior
        closes — same standard as the entry pivot quality fix.

        Fires for both REVERSION and TREND trade types when profit > 1.0×ATR
        (Option B decision). The ATR runner trail stays as the floor —
        this trail takes over when it produces a higher stop level.

        Gates:
        - structural_trailing_enabled must be true in phase_config
        - Trade must be in profit (no trailing on losers)
        - Minimum 10 bars in trade (enough data for meaningful pivots)
        - Candidate must be ABOVE current_stop_loss (one-way ratchet)
        - Candidate must be BELOW current price (stop behind price)
        """
        _phase_cfg = self.risk_config.get("phase_config", {})
        if not _phase_cfg.get("structural_trailing_enabled", False):
            return None

        if atr <= 0 or self.bars_in_trade < 10:
            return None

        _in_profit = (
            (self.side == "long"  and self.close[-1] > self.entry_price) or
            (self.side == "short" and self.close[-1] < self.entry_price)
        )
        if not _in_profit:
            return None

        try:
            highs_arr  = self.high
            lows_arr   = self.low
            closes_arr = self.close
            _min_depth = 0.3 * atr

            if self.side == "long":
                # Find the highest confirmed swing LOW above current stop
                best_candidate = None
                for i in range(len(lows_arr) - 3, 6, -1):
                    if (
                        lows_arr[i] < lows_arr[i - 1]
                        and lows_arr[i] < lows_arr[i - 2]
                        and lows_arr[i] < lows_arr[i - 3]
                        and lows_arr[i] < lows_arr[i - 4]
                        and lows_arr[i] < lows_arr[i + 1]
                        and lows_arr[i] < lows_arr[i + 2]
                        and (min(closes_arr[i - 4:i]) - lows_arr[i] >= _min_depth)
                    ):
                        # Stop goes just below this swing low
                        _candidate = lows_arr[i] - (0.3 * atr)
                        # Valid only if it advances the stop (never pulls back)
                        # and stays behind current price
                        if (
                            _candidate > self.current_stop_loss
                            and _candidate < self.close[-1]
                        ):
                            if best_candidate is None or _candidate > best_candidate:
                                best_candidate = _candidate
                return best_candidate

            else:  # short
                # Find the lowest confirmed swing HIGH below current stop
                best_candidate = None
                for i in range(len(highs_arr) - 3, 6, -1):
                    if (
                        highs_arr[i] > highs_arr[i - 1]
                        and highs_arr[i] > highs_arr[i - 2]
                        and highs_arr[i] > highs_arr[i - 3]
                        and highs_arr[i] > highs_arr[i - 4]
                        and highs_arr[i] > highs_arr[i + 1]
                        and highs_arr[i] > highs_arr[i + 2]
                        and (highs_arr[i] - max(closes_arr[i - 4:i]) >= _min_depth)
                    ):
                        _candidate = highs_arr[i] + (0.3 * atr)
                        if (
                            _candidate < self.current_stop_loss
                            and _candidate > self.close[-1]
                        ):
                            if best_candidate is None or _candidate < best_candidate:
                                best_candidate = _candidate
                return best_candidate

        except Exception as _e:
            logger.debug(
                "[VTM STRUCT TRAIL] Swing pivot scan failed (non-blocking): %s", _e
            )
            return None


    def __repr__(self):
        levels = self.get_current_levels()
        return f"VTM({self.asset} {self.side.upper()}: Entry=${levels['entry_price']:.2f}, Current=${levels['current_price']:.2f}, SL=${levels['stop_loss']:.2f}, P&L={levels['pnl_pct']:+.2f}%)"
