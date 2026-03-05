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
        self.trade_type = "TREND" # Strict TREND enforcement
        
        self.position_size = quantity
        
        # Macro MAs from signal_details (for MA Shield/Front-run)
        gov_data = self.signal_details.get("governor_data", {})
        self.ema_1d_200 = gov_data.get("ema_1d_200")
        self.ema_4h_200 = gov_data.get("ema_4h_200")
        self.ema_4h_50 = gov_data.get("ema_4h_50")

        # TREND ENFORCEMENT
        self.atr_multiplier = self.risk_config.get("atr_multiplier", 1.8)
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
        self.profit_locked = False
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
    
    # ... Rest of the file is the same ...
    # ... I will omit it for brevity but the logic remains ...

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
        try:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=self.atr_period)
            current_atr = atr[-1]
            if np.isnan(current_atr) or current_atr <= 0:
                return self.entry_price * 0.02
            return current_atr
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
                self.initial_stop_loss = max(
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
                self.initial_stop_loss = min(
                    self.entry_price + max_stop_dist,
                    max(self.entry_price + min_stop_dist, final_sl)
                )

            # ATR-based adaptive tolerance for structure identification
            tolerance = 0.5 * atr
            structure_levels = find_resistance_levels(self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback, tolerance=tolerance)
            
            raw_targets, self.partial_sizes = calculate_hybrid_targets(
                self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                self.partial_targets, self.partial_sizes,
                min_rr=2.0 # Standard TREND requirement
            )
            
            # ✅ PHASE 5: MA FRONT-RUN (Take Profit)
            # If macro MA ceiling near TP, adjust TP to front-run it
            self.take_profit_levels = []
            for tp in raw_targets:
                adjusted_tp = tp
                for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                    if ma:
                        # "Near" defined as within 0.5 * ATR
                        if self.side == "long":
                            if abs(tp - ma) < (0.5 * atr) or (tp > ma > self.entry_price):
                                adjusted_tp = min(adjusted_tp, ma - (0.25 * atr))
                                logger.info(f"[VTM] 🏃 MA Front-run: TP ${tp:,.2f} → ${adjusted_tp:,.2f} (MA: ${ma:,.2f})")
                        else: # short
                            if abs(tp - ma) < (0.5 * atr) or (tp < ma < self.entry_price):
                                adjusted_tp = max(adjusted_tp, ma + (0.25 * atr))
                                logger.info(f"[VTM] 🏃 MA Front-run: TP ${tp:,.2f} → ${adjusted_tp:,.2f} (MA: ${ma:,.2f})")
                self.take_profit_levels.append(adjusted_tp)

            # Fallback targets if structure calculation failed
            if not self.take_profit_levels:
                logger.warning("[VTM] No take profit levels calculated, using ATR-based fallback.")
                self.take_profit_levels = [self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m) for m in self.partial_targets]
                self.partial_sizes = [0.45, 0.30, 0.25] # Default partial sizes

            self.current_stop_loss = self.initial_stop_loss
        except Exception as e:
            logger.error(f"[VTM] Level calculation error: {e}", exc_info=True)
            raise

    def on_new_bar(self, new_high: float, new_low: float, new_close: float) -> Optional[Dict]:
        try:
            self.high, self.low, self.close = np.append(self.high, new_high), np.append(self.low, new_low), np.append(self.close, new_close)
            if len(self.close) > 200: self.high, self.low, self.close = self.high[-200:], self.low[-200:], self.close[-200:]
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

    def check_exit(self, current_price: float, atr_value: Optional[float] = None, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        if atr_value is None:
            atr_value = self._calculate_atr() # Fallback if ATR not passed
        if self.remaining_position <= 0: return None
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

            if self.bars_in_trade >= self.time_stop_bars:
                return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}
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
        return {"entry_price": self.entry_price, "side": self.side, "asset": self.asset, "position_size": self.position_size, "initial_stop_loss": self.initial_stop_loss, "current_stop_loss": self.current_stop_loss, "take_profit_levels": self.take_profit_levels, "partial_sizes": self.partial_sizes, "remaining_position": self.remaining_position, "partials_hit": self.partials_hit, "bars_in_trade": self.bars_in_trade, "highest_price_reached": self.highest_price_reached, "lowest_price_reached": self.lowest_price_reached, "runner_activated": self.runner_activated, "trade_type": self.trade_type, "entry_time": self.entry_time.isoformat()}

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        vtm = cls(entry_price=state["entry_price"], side=state["side"], asset=state["asset"], high=high, low=low, close=close, quantity=state["position_size"], trade_type=state.get("trade_type", "TREND"), risk_config={})
        vtm.initial_stop_loss, vtm.current_stop_loss, vtm.take_profit_levels, vtm.partial_sizes, vtm.remaining_position, vtm.partials_hit, vtm.bars_in_trade, vtm.highest_price_reached, vtm.lowest_price_reached, vtm.runner_activated = state["initial_stop_loss"], state["current_stop_loss"], state["take_profit_levels"], state["partial_sizes"], state["remaining_position"], state["partials_hit"], state["bars_in_trade"], state["highest_price_reached"], state["lowest_price_reached"], state["runner_activated"]
        return vtm

    def __repr__(self):
        levels = self.get_current_levels()
        return f"VTM({self.asset} {self.side.upper()}: Entry=${levels['entry_price']:.2f}, Current=${levels['current_price']:.2f}, SL=${levels['stop_loss']:.2f}, P&L={levels['pnl_pct']:+.2f}%)"

    