"""
Veteran Trade Manager - Strategic/Tactical Risk Architecture
✨ REFACTORED: Centralized risk configuration from config.json.
📊 ROLE: Tactical execution engine (HOW to manage trades, not HOW MUCH to risk)
"""

import logging
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
            max_stop_dist = atr_fast * 5.0
            min_rr = 1.5
            
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance > max_stop_dist:
                return False, f"Stop too wide: ${stop_distance:,.2f} > 5.0 * ATR (${max_stop_dist:,.2f})"
            
            risk_multiples = risk_config.get("partial_targets", [1.5])
            first_target_multiple = risk_multiples[0] if risk_multiples else 1.5
            
            # expected_tp_distance = abs(tp - entry)
            expected_tp_distance = stop_distance * first_target_multiple
            
            # BLOCK trade IF: expected_tp_distance < (0.5 * atr_fast)
            if expected_tp_distance < (0.5 * atr_fast):
                return False, f"Profit target too close: ${expected_tp_distance:,.2f} < 0.5 * ATR (${0.5 * atr_fast:,.2f})"
            
            actual_rr = expected_tp_distance / stop_distance if stop_distance > 0 else 0
            if actual_rr < min_rr - 1e-9:
                return False, f"R:R too low: {actual_rr:.2f}:1 < {min_rr:.2f}:1 (min for TREND)"
            
            logger.info(
                f"[VTM PRE-FLIGHT] ✅ Trade Valid\n"
                f"  Type:   TREND\n"
                f"  Stop:   ${stop_distance:,.2f} ({(stop_distance/entry_price):.2%})\n"
                f"  Profit: ${expected_tp_distance:,.2f} ({(expected_tp_distance/entry_price):.2%})\n"
                f"  R:R:    {actual_rr:.2f}:1"
            )
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"[VTM PRE-FLIGHT] Error: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

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
        current_bid: float = 0.0        # ✨ NEW: For spread floor
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
        self.atr_multiplier = self.risk_config.get("atr_multiplier", 1.8)
        
        # Override with dynamic multipliers based on regime/type
        if self.trade_type == "REVERSION":
            self.atr_multiplier = 1.5 # Tighter for mean reversion
        elif self.trade_type == "TREND":
            # Check for high volatility or bearish regimes via signal_details
            regime = self.signal_details.get("regime", "NEUTRAL")
            volatility = self.signal_details.get("volatility_regime", "normal")
            
            if "BEAR" in regime or volatility == "high":
                self.atr_multiplier = 2.5 # Wide for safety in bears/volatility
            else:
                self.atr_multiplier = 2.0 # Standard trend breathing room

        logger.info(f"[VTM] Dynamic ATR Multiplier set to {self.atr_multiplier}x ({self.trade_type})")

        self.partial_targets = self.risk_config.get("partial_targets", [1.5, 3.0, 5.0])
        self.partial_sizes = self.risk_config.get("partial_sizes", [0.45, 0.30, 0.25])

        self.pivot_lookback = self.risk_config.get("pivot_lookback", 30)
        self.time_stop_bars = self.risk_config.get("time_stop_bars", 72)
        self.use_ema_structure = self.risk_config.get("use_ema_structure", False)
        
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
        self.has_pyramided = False # ✨ NEW: Trend Pyramiding Flag
        self.entry_time = datetime.now()
        
        # Calculate levels
        try:
            self._calculate_initial_levels()
        except Exception as e:
            logger.error(f"[VTM] Initialization error: {e}")
            raise

        # Log initialization
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

    @property
    def profit_locked(self) -> bool:
        """Checks if stop loss is at break-even or better"""
        if self.side == "long":
            return self.current_stop_loss >= self.entry_price
        else:
            return self.current_stop_loss <= self.entry_price

    def _calc_pct_distance(self, price1: float, price2: float) -> float:
        return abs(price1 - price2) / price1 * 100

    def _calculate_atr(self) -> float:
        """
        ✅ TASK 18: Regime-adaptive ATR: fast in expanding vol, slow in compressed vol.
        """
        try:
            atr_fast = talib.ATR(self.high, self.low, self.close, timeperiod=7)[-1]
            atr_mid  = talib.ATR(self.high, self.low, self.close, timeperiod=14)[-1]
            atr_slow = talib.ATR(self.high, self.low, self.close, timeperiod=28)[-1]
            
            if np.isnan(atr_mid) or atr_slow == 0:
                return self.entry_price * 0.015
                
            ratio = atr_fast / atr_slow
            
            if ratio > 1.35:
                # Expanding vol — tighten fast
                selected_atr = atr_fast
                reason = "Expanding Vol (Tighten)"
            elif ratio < 0.75:
                # Compressed vol — breathe wide
                selected_atr = atr_slow
                reason = "Compressed Vol (Wide)"
            else:
                selected_atr = atr_mid
                reason = "Normal Vol"
                
            logger.info(f"[VTM] Dynamic ATR Selection: {selected_atr:.4f} ({reason}, Ratio: {ratio:.2f})")
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
            
            volume_strong = volume_ratio > 1.2
            
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
                    self.initial_stop_loss = self.low[-1] - wick_buffer
                    tp_target = self.ema_4h_50 - (0.2 * atr) if self.ema_4h_50 else self.entry_price + (2.0 * atr)

                else:
                    self.initial_stop_loss = self.high[-1] + wick_buffer
                    tp_target = self.ema_4h_50 + (0.2 * atr) if self.ema_4h_50 else self.entry_price - (2.0 * atr)

                self.current_stop_loss = self.initial_stop_loss
                self.take_profit_levels = [tp_target]
                self.partial_sizes = [1.0]

                logger.info(f"[VTM] REVERSION MODE: SL={self.initial_stop_loss}, TP={tp_target}")

            else:
                # ATR-based adaptive floors and caps
                min_stop_dist = atr * 0.5
                max_stop_dist = atr * 5.0

                if self.side == "long":
                    # 1. Standard ATR Baseline
                    target_stop_dist = atr * self.atr_multiplier
                    standard_sl = self.entry_price - target_stop_dist
                    final_sl = standard_sl

                    # 2. Joint Synergy: MA Shield
                    for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
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

                    # 2. Joint Synergy: MA Shield
                    for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
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

                # ATR-based adaptive tolerance for structure identification
                tolerance = 0.5 * atr
                structure_levels = find_resistance_levels(self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback, tolerance=tolerance)
                
                raw_targets, self.partial_sizes = calculate_hybrid_targets(
                    self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                    self.partial_targets, self.partial_sizes,
                    min_rr=2.0 # Standard TREND requirement
                )
                
                # ✅ PHASE 5: MA FRONT-RUN (Take Profit)
                self.take_profit_levels = []
                for tp in raw_targets:
                    adjusted_tp = tp
                    for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                        if ma:
                            if self.side == "long":
                                if abs(tp - ma) < (0.5 * atr) or (tp > ma > self.entry_price):
                                    candidate_tp = ma - (0.25 * atr)
                                    if candidate_tp > self.entry_price + (0.5 * atr):
                                        adjusted_tp = min(adjusted_tp, candidate_tp)
                            else: # short
                                if abs(tp - ma) < (0.5 * atr) or (tp < ma < self.entry_price):
                                    candidate_tp = ma + (0.25 * atr)
                                    if candidate_tp < self.entry_price - (0.5 * atr):
                                        adjusted_tp = max(adjusted_tp, candidate_tp)
                    self.take_profit_levels.append(adjusted_tp)

                # Fallback targets
                if not self.take_profit_levels:
                    self.take_profit_levels = [self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m) for m in self.partial_targets]
                    self.partial_sizes = [0.45, 0.30, 0.25]

            # STEP 3 — Lot Sanitizer
            # Reason: Ensures position size is valid for broker submission.
            LOT_PRECISION = {
                'BTC': 4,
                'GOLD': 2,
                'USTEC': 1,
                'EURJPY': 2,
                'EURUSD': 2,
            }

            precision = LOT_PRECISION.get(self.asset.upper(), 2)
            final_size = round(self.position_size, precision)

            MIN_LOT = {
                'BTC': 0.0001,
                'GOLD': 0.01,
                'USTEC': 0.1,
                'EURJPY': 0.01,
                'EURUSD': 0.01
            }

            min_lot = MIN_LOT.get(self.asset.upper(), 0.01)

            if final_size < min_lot:
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

    def update_with_current_price(self, current_price: float, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        try:
            atr = self._calculate_atr() # Calculate ATR here
            
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                if self.runner_activated and self.highest_price_reached > old_high and self.trade_type == "TREND":
                    # ✅ PHASE 5: ATR-BASED RUNNER TRAIL
                    # SL breathes with 2.0 * ATR
                    new_trail = self.highest_price_reached - (2.0 * atr)
                    
                    if new_trail > self.current_stop_loss: 
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                if self.runner_activated and self.lowest_price_reached < old_low and self.trade_type == "TREND":
                    # ✅ PHASE 5: ATR-BASED RUNNER TRAIL - SHORT
                    new_trail = self.lowest_price_reached + (2.0 * atr)
                    
                    if new_trail < self.current_stop_loss: 
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail
            
            return self.check_exit(current_price, atr, df_4h=df_4h) # Pass ATR and df_4h to check_exit
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

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

    def cancel_take_profit(self):
        """Cancel all remaining take profit targets."""
        self.take_profit_levels = []
        self.partial_sizes = []
        logger.debug(f"[VTM] All take profit orders cancelled for {self.asset}.")

    def enable_trailing_stop(self):
        """Activate the trailing stop mechanism (Runner Mode)."""
        self.runner_activated = True
        logger.debug(f"[VTM] Trailing stop (Greed Mode) enabled for {self.asset}.")

    def check_exit(self, current_price: float, atr_value: Optional[float] = None, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        if atr_value is None:
            atr_value = self._calculate_atr() # Fallback if ATR not passed
        if self.remaining_position <= 0: return None

        # --- STEP 1: Volatility Break-Even Lock ---
        # Reason: Locks risk to zero once trade proves itself by moving 1.0 * ATR in profit.
        if self.side == "long":
            current_profit = current_price - self.entry_price
        else:
            current_profit = self.entry_price - current_price

        if current_profit > atr_value:
            if self.current_stop_loss != self.entry_price:
                logger.info(f"[VTM] 🛡️ Break-even lock activated for {self.asset} (Profit: ${current_profit:.2f} > ATR: ${atr_value:.2f})")
                self.current_stop_loss = self.entry_price

        # --- STEP 2: Greed Mode Accelerator ---
        # Reason: During extreme trends/volatility, remove early targets to ride the move with trailing stops.
        adx_value = self._calculate_adx()
        atr_slow = self._calculate_atr_slow()
        
        if adx_value > 40 and atr_value > (1.5 * atr_slow):
            if len(self.take_profit_levels) > 1:
                logger.info(f"[VTM] 🔥 GREED MODE: Strong trend (ADX:{adx_value:.1f}) & Volatility Expansion detected. Removing early targets.")
                # Keep only final target
                self.take_profit_levels = [self.take_profit_levels[-1]]
                # Exit full size at final target (managed by runner trail)
                self.partial_sizes = [1.0]

        # --- STEP 3: Trend Pyramiding ---
        # Objective: Scale into strong breakout trends.
        if self.trade_type == "TREND" and not self.has_pyramided:
            if current_profit >= (1.0 * atr_value) and adx_value > 25:
                logger.info(f"[VTM] 🗼 TREND PYRAMIDING: Strong trend confirmed. Scaling in.")
                
                # Action 1: Move SL of position 1 to entry
                self.current_stop_loss = self.entry_price
                
                # Action 2: Signal for second trade (new_size = original_size * 0.5)
                # Note: We set the flag here; the return dict signals the caller (PortfolioManager) to execute.
                self.has_pyramided = True
                
                return {
                    "action": "pyramid",
                    "asset": self.asset,
                    "side": self.side,
                    "new_size": self.position_size * 0.5,
                    "reason": "Trend Pyramiding Triggered"
                }

        # --- STEP 4: Trade State Mutation ---
        # Objective: Allow profitable Mean Reversion trades to convert into trend trades.
        if self.trade_type == "REVERSION":
            if adx_value > 30 and current_profit > 0:
                # Check actual direction of profit
                is_actually_profitable = (self.side == "long" and current_price > self.entry_price) or \
                                         (self.side == "short" and current_price < self.entry_price)
                
                if is_actually_profitable:
                    logger.info("[VTM] 🧬 Trade mutated from REVERSION to TREND. Ride the move.")
                    
                    # 1. Cancel existing take profit orders
                    self.cancel_take_profit()
                    
                    # 2. Switch trade tag
                    self.trade_type = "TREND"
                    
                    # 3. Activate greed mode trailing stop
                    self.enable_trailing_stop()

        # --- STEP 5: Time Decay Protection ---
        # Objective: Prevent Mean Reversion trades from turning into long-term losses.
        if self.trade_type == "REVERSION":
            if self.bars_in_trade >= 8:
                logger.info(f"[VTM] ⏳ Stale MR trade closed for {self.asset} (Bars: {self.bars_in_trade} >= 8)")
                return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}

        try:
            pnl_pct = (current_price - self.entry_price) / self.entry_price if self.side == "long" else (self.entry_price - current_price) / self.entry_price

            if (self.side == "long" and current_price <= self.current_stop_loss) or (self.side == "short" and current_price >= self.current_stop_loss):
                # Determine specific reason based on stop loss level relative to entry
                reason = ExitReason.STOP_LOSS
                
                # ATR-based adaptive offset check for trailing reason
                offset = 0.125 * atr_value
                if self.side == "long":
                    if self.current_stop_loss > self.entry_price + offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                else: # short
                    if self.current_stop_loss < self.entry_price - offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                
                return {"reason": reason, "price": current_price, "size": self.remaining_position}

            for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes)):
                if i in self.partials_hit: continue
                if (self.side == "long" and current_price >= target) or (self.side == "short" and current_price <= target):
                    self.partials_hit.append(i)
                    self.remaining_position -= size
                    
                    # ✅ PHASE 5: TP1 TRAILING (Simplified to ATR baseline)
                    if len(self.partials_hit) >= 2 and not self.runner_activated and self.trade_type == "TREND": 
                        self.runner_activated = True
                        logger.info(f"[VTM TACTICAL] 🏃 Runner Activated: Trailing stop will now follow price.")
                    
                    return {"reason": [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3][i], "price": current_price, "size": size}

            # if self.bars_in_trade >= self.time_stop_bars:
            #     return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}
            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self, live_price: Optional[float] = None) -> Dict:
        current_price = live_price if live_price is not None else self.close[-1]
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100 if self.side == "long" else (self.entry_price - current_price) / self.entry_price * 100
        next_target_idx = len(self.partials_hit)
        next_target = self.take_profit_levels[next_target_idx] if next_target_idx < len(self.take_profit_levels) else None
        
        distance_to_sl_pct = (current_price - self.current_stop_loss) / self.current_stop_loss * 100 if self.current_stop_loss > 0 else 0
        distance_to_tp_pct = (next_target - current_price) / current_price * 100 if next_target and current_price > 0 else 0
        
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
            "has_pyramided": self.has_pyramided, # ✨ NEW
            "trade_type": self.trade_type, 
            "entry_time": self.entry_time.isoformat(),
            "local_free_margin": self.local_free_margin,
            "current_ask": self.current_ask,
            "current_bid": self.current_bid
        }

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        vtm = cls(
            entry_price=state["entry_price"], 
            side=state["side"], 
            asset=state["asset"], 
            high=high, 
            low=low, 
            close=close, 
            quantity=state["position_size"], 
            trade_type=state.get("trade_type", "TREND"), 
            risk_config={},
            local_free_margin=state.get("local_free_margin", 0.0),
            current_ask=state.get("current_ask", 0.0),
            current_bid=state.get("current_bid", 0.0)
        )
        vtm.initial_stop_loss, vtm.current_stop_loss, vtm.take_profit_levels, vtm.partial_sizes, vtm.remaining_position, vtm.partials_hit, vtm.bars_in_trade, vtm.highest_price_reached, vtm.lowest_price_reached, vtm.runner_activated, vtm.has_pyramided = state["initial_stop_loss"], state["current_stop_loss"], state["take_profit_levels"], state["partial_sizes"], state["remaining_position"], state["partials_hit"], state["bars_in_trade"], state["highest_price_reached"], state["lowest_price_reached"], state["runner_activated"], state.get("has_pyramided", False)
        return vtm

    def __repr__(self):
        levels = self.get_current_levels()
        return f"VTM({self.asset} {self.side.upper()}: Entry=${levels['entry_price']:.2f}, Current=${levels['current_price']:.2f}, SL=${levels['stop_loss']:.2f}, P&L={levels['pnl_pct']:+.2f}%)"
