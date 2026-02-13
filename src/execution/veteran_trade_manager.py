"""
Veteran Trade Manager - Strategic/Tactical Risk Architecture
✨ REFACTORED: Centralized risk configuration from config.json.
📊 ROLE: Tactical execution engine (HOW to manage trades, not HOW MUCH to risk)
"""

import logging
import numpy as np
import talib
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
    tolerance_pct: float = 0.01,
) -> List[float]:
    """Find significant resistance/support levels"""
    lookback = min(lookback, len(close))
    levels = []

    if side == "long":
        highs = high[-lookback:]
        for i in range(2, len(highs) - 2):
            if highs[i] > current_price:
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    levels.append(highs[i])

        clustered = []
        for level in sorted(levels):
            if not clustered or (level - clustered[-1]) / level > tolerance_pct:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for h in highs if abs(h - level) / level <= tolerance_pct)
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
            if not clustered or (clustered[-1] - level) / level > tolerance_pct:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for l in lows if abs(l - level) / level <= tolerance_pct)
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
        min_profit_viability: float = 0.0025,
    ) -> Tuple[bool, str]:
        """
        ✨ REFACTORED: Pre-flight validation using risk_config dictionary.
        """
        try:
            # Asymmetric constraints
            if trade_type == "SCALP":
                base_max = risk_config.get("max_stop_pct", 0.03)
                max_stop = base_max * 0.7
                min_rr = 1.2
            else:  # TREND
                max_stop = risk_config.get("max_stop_pct", 0.06)
                min_rr = 1.5
            
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_pct = stop_distance / entry_price
            
            if stop_distance_pct > max_stop:
                return False, f"Stop too wide: {stop_distance_pct:.2%} > {max_stop:.2%} (max for {trade_type})"
            
            risk_multiples = risk_config.get("partial_targets", [1.5])
            first_target_multiple = risk_multiples[0] if risk_multiples else 1.5
            
            # Calculate first target based on side
            if stop_loss < entry_price: # LONG
                first_target = entry_price + (stop_distance * first_target_multiple)
                profit_distance = abs(first_target - entry_price)
            else: # SHORT
                first_target = entry_price - (stop_distance * first_target_multiple)
                profit_distance = abs(entry_price - first_target)
            
            profit_pct = profit_distance / entry_price
            
            if profit_pct < min_profit_viability:
                return False, f"Profit target too close: {profit_pct:.2%} < {min_profit_viability:.2%} (fees will eat it)"
            
            actual_rr = profit_distance / stop_distance if stop_distance > 0 else 0
            if actual_rr < min_rr - 1e-9:
                return False, f"R:R too low: {actual_rr:.2f}:1 < {min_rr:.2f}:1 (min for {trade_type})"
            
            logger.info(
                f"[VTM PRE-FLIGHT] ✅ Trade Valid\n"
                f"  Type:   {trade_type}\n"
                f"  Stop:   {stop_distance_pct:.2%}\n"
                f"  Profit: {profit_pct:.2%}\n"
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
        enable_early_profit_lock: bool = True,
        early_lock_threshold_pct: float = 0.01, # Keep for backward compatibility/default
        early_lock_atr_multiplier: Optional[float] = None, # NEW
        runner_trail_atr_multiplier: Optional[float] = None, # NEW
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
        self.trade_type = trade_type
        
        self.position_size = quantity
        
        # Original fixed parameters from __init__ arguments
        self.enable_early_profit_lock = enable_early_profit_lock
        self.early_lock_threshold_pct = early_lock_threshold_pct 
        self.runner_trail_pct = risk_config.get("runner_trail_pct", 0.025) # Default from risk_config
        
        # New dynamic multipliers (if provided in risk_config)
        self.early_lock_atr_multiplier = risk_config.get("early_lock_atr_multiplier") 
        self.runner_trail_atr_multiplier = risk_config.get("runner_trail_atr_multiplier")
        
        # Initialize current calculated dynamic percentages (to be updated during runtime)
        # These will reflect the actual threshold/trail used based on ATR or fixed default
        self.current_early_lock_threshold_pct = self.early_lock_threshold_pct # Initialize with fixed default
        self.current_runner_trail_pct = self.runner_trail_pct # Initialize with fixed default


        # ✨ Asymmetric constraints for SCALP vs TREND using risk_config
        if self.trade_type == "SCALP":
            self.min_stop_pct = self.risk_config.get("min_stop_pct", 0.01) * 0.5
            self.max_stop_pct = self.risk_config.get("max_stop_pct", 0.03) * 0.7
            self.atr_multiplier = self.risk_config.get("atr_multiplier", 1.5) * 0.75

            # FORCE tighter SCALP targets (do not read config)
            self.partial_targets = [1.2, 2.0, 3.0]
            self.partial_sizes = [0.5, 0.3, 0.2]
            
            # SCALP-specific fixed early lock and runner trail
            self.early_lock_threshold_pct = 0.005
            self.runner_trail_pct = 0.005 # Tight trail for scalping

        else:  # TREND
            self.min_stop_pct = self.risk_config.get("min_stop_pct", 0.008)
            self.max_stop_pct = self.risk_config.get("max_stop_pct", 0.06)
            self.atr_multiplier = self.risk_config.get("atr_multiplier", 1.8)
            self.partial_targets = self.risk_config.get("partial_targets", [1.5, 3.0, 5.0])
            self.partial_sizes = self.risk_config.get("partial_sizes", [0.45, 0.30, 0.25])
            # TREND-specific default runner trail (early_lock_threshold_pct already handled above)
            self.runner_trail_pct = risk_config.get("runner_trail_pct", 0.025) # Default from risk_config

        self.pivot_lookback = self.risk_config.get("pivot_lookback", 30)
        self.time_stop_bars = self.risk_config.get("time_stop_bars", 72)
        self.use_ema_structure = self.risk_config.get("use_ema_structure", False)
        
        # State
        self.early_profit_locked = False
        self.initial_stop_loss = None
        self.current_stop_loss = None
        self.take_profit_levels = []
        self.remaining_position = 1.0
        self.partials_hit = []
        self.bars_in_trade = 0
        self.highest_price_reached = entry_price
        self.lowest_price_reached = entry_price
        self.runner_activated = False
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
        
        # Log dynamic VTM parameters
        if self.early_lock_atr_multiplier is not None:
            logger.info(f"Early Lock: Dynamic ({self.early_lock_atr_multiplier}x ATR)")
            logger.info(f"  Initial Threshold: {self.early_lock_threshold_pct:.2%}")
        else:
            logger.info(f"Early Lock: Fixed ({self.early_lock_threshold_pct:.2%})")

        if self.runner_trail_atr_multiplier is not None:
            logger.info(f"Runner Trail: Dynamic ({self.runner_trail_atr_multiplier}x ATR)")
            logger.info(f"  Initial Trail: {self.runner_trail_pct:.2%}")
        else:
            logger.info(f"Runner Trail: Fixed ({self.runner_trail_pct:.2%})")

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

    def _get_adaptive_percentage(self, atr_value: float, multiplier: float) -> float:
        """
        Calculates an adaptive percentage based on ATR and a given multiplier.
        This adapts the percentage to current market volatility.
        """
        if not atr_value or self.entry_price == 0:
            return 0.0 # Avoid division by zero or invalid ATR

        # Calculate percentage as a fraction of entry price relative to ATR
        adaptive_pct = (atr_value * multiplier) / self.entry_price
        return adaptive_pct

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
                self.current_stop_loss = self.entry_price * (1.001 if self.side == "long" else 0.999)
                logger.info(f"[VTM] Stop moved to break-even: ${self.current_stop_loss:,.2f}")
                return True
            else:
                self.current_stop_loss = self.entry_price * (1.001 if self.side == "long" else 0.999)
                return False
        
        except Exception as e:
            logger.error(f"[VTM] Promotion check error: {e}")
            return False

    def _find_pivot_structure(self) -> Optional[float]:
        try:
            lookback = min(self.pivot_lookback, len(self.close) - 3)

            if self.side == "long":
                for i in range(lookback - 2, 1, -1):
                    current, left, right = self.low[-(i + 1)], self.low[-(i + 2)], self.low[-i]
                    if current < left and current < right: return current
                return np.min(self.low[-lookback:])
            else:
                for i in range(lookback - 2, 1, -1):
                    current, left, right = self.high[-(i + 1)], self.high[-(i + 2)], self.high[-i]
                    if current > left and current > right: return current
                return np.max(self.high[-lookback:])
        except Exception as e:
            logger.error(f"[VTM] Pivot error: {e}")
            return None

    def _calculate_initial_levels(self):
        try:
            atr = self._calculate_atr()
            pivot_level = self._find_pivot_structure()

            if self.side == "long":
                atr_stop = self.entry_price * (1 - max(self.min_stop_pct, min(self.max_stop_pct, (atr * self.atr_multiplier) / self.entry_price)))
                structure_stop = pivot_level - (atr * 1.0) if pivot_level else self.entry_price * (1 - self.min_stop_pct)
                preliminary_stop = min(atr_stop, structure_stop)
                self.initial_stop_loss = max(self.entry_price * (1 - self.max_stop_pct), min(self.entry_price * (1 - self.min_stop_pct), preliminary_stop))
            else: # short
                atr_stop = self.entry_price * (1 + max(self.min_stop_pct, min(self.max_stop_pct, (atr * self.atr_multiplier) / self.entry_price)))
                structure_stop = pivot_level + (atr * 1.0) if pivot_level else self.entry_price * (1 + self.min_stop_pct)
                preliminary_stop = max(atr_stop, structure_stop)
                self.initial_stop_loss = min(self.entry_price * (1 + self.max_stop_pct), max(self.entry_price * (1 + self.min_stop_pct), preliminary_stop))

            structure_levels = find_resistance_levels(self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback)
            self.take_profit_levels, self.partial_sizes = calculate_hybrid_targets(
                self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                self.partial_targets, self.partial_sizes,
                min_rr=1.5 if self.trade_type == "SCALP" else 2.0
            )

            # Temporary hack for paper mode if targets are not calculated
            if not self.take_profit_levels:
                logger.warning("[VTM] No take profit levels calculated, using dummy values for paper mode.")
                self.take_profit_levels = [self.entry_price * (1.01), self.entry_price * (1.02), self.entry_price * (1.03)] if self.side == "long" else [self.entry_price * (0.99), self.entry_price * (0.98), self.entry_price * (0.97)]
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

    def update_with_current_price(self, current_price: float) -> Optional[Dict]:
        try:
            atr = self._calculate_atr() # Calculate ATR here
            
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                if self.runner_activated and self.highest_price_reached > old_high and self.trade_type == "TREND":
                    # Use dynamic runner_trail_pct
                    if self.runner_trail_atr_multiplier is not None:
                        dynamic_runner_trail_pct = self._get_adaptive_percentage(atr, self.runner_trail_atr_multiplier)
                        new_trail = self.highest_price_reached * (1 - dynamic_runner_trail_pct)
                    else: # Fallback to fixed if multiplier not provided
                        new_trail = self.highest_price_reached * (1 - self.runner_trail_pct)
                    
                    if new_trail > self.current_stop_loss: 
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}). Dynamic Trail: {self.current_runner_trail_pct:.2%}")
                        self.current_stop_loss = new_trail
                        if self.runner_trail_atr_multiplier is not None:
                            self.current_runner_trail_pct = dynamic_runner_trail_pct # Store the dynamic trail
                        else:
                            self.current_runner_trail_pct = self.runner_trail_pct # Store the fixed trail
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                if self.runner_activated and self.lowest_price_reached < old_low and self.trade_type == "TREND":
                    # Use dynamic runner_trail_pct
                    if self.runner_trail_atr_multiplier is not None:
                        dynamic_runner_trail_pct = self._get_adaptive_percentage(atr, self.runner_trail_atr_multiplier)
                        new_trail = self.lowest_price_reached * (1 + dynamic_runner_trail_pct)
                    else: # Fallback to fixed if multiplier not provided
                        new_trail = self.lowest_price_reached * (1 + self.runner_trail_pct)
                    
                    if new_trail < self.current_stop_loss: 
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}). Dynamic Trail: {self.current_runner_trail_pct:.2%}")
                        self.current_stop_loss = new_trail
                        if self.runner_trail_atr_multiplier is not None:
                            self.current_runner_trail_pct = dynamic_runner_trail_pct # Store the dynamic trail
                        else:
                            self.current_runner_trail_pct = self.runner_trail_pct # Store the fixed trail
            
            return self.check_exit(current_price, atr) # Pass ATR to check_exit
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

    def check_exit(self, current_price: float, atr_value: Optional[float] = None) -> Optional[Dict]:
        if atr_value is None:
            atr_value = self._calculate_atr() # Fallback if ATR not passed
        if self.remaining_position <= 0: return None
        try:
            pnl_pct = (current_price - self.entry_price) / self.entry_price if self.side == "long" else (self.entry_price - current_price) / self.entry_price
            # Calculate dynamic early lock threshold
            actual_early_lock_threshold = self.early_lock_threshold_pct # Default to fixed
            if self.early_lock_atr_multiplier is not None:
                actual_early_lock_threshold = self._get_adaptive_percentage(atr_value, self.early_lock_atr_multiplier)

            if self.enable_early_profit_lock and not self.early_profit_locked and pnl_pct >= actual_early_lock_threshold:
                self.early_profit_locked = True
                self.current_early_lock_threshold_pct = actual_early_lock_threshold # Store the actual threshold that triggered the lock
                if self.side == "long":
                    if (new_stop := self.entry_price * 1.001) > self.current_stop_loss: self.current_stop_loss = new_stop
                else:
                    if (new_stop := self.entry_price * 0.999) < self.current_stop_loss: self.current_stop_loss = new_stop
                logger.info(f"[VTM] 🛡️ Break-even @ +{pnl_pct:.2%} (Dynamic Lock Threshold: {actual_early_lock_threshold:.2%})")

            if (self.side == "long" and current_price <= self.current_stop_loss) or (self.side == "short" and current_price >= self.current_stop_loss):
                return {"reason": ExitReason.STOP_LOSS, "price": current_price, "size": self.remaining_position}

            for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes)):
                if i in self.partials_hit: continue
                if (self.side == "long" and current_price >= target) or (self.side == "short" and current_price <= target):
                    self.partials_hit.append(i)
                    self.remaining_position -= size
                    if len(self.partials_hit) == 1 and self.trade_type == "TREND" and not self.check_promotion_to_runner(current_price) and not self.early_profit_locked:
                        self.current_stop_loss = max(self.current_stop_loss, self.entry_price * 1.001) if self.side == "long" else min(self.current_stop_loss, self.entry_price * 0.999)
                    elif len(self.partials_hit) >= 2 and not self.runner_activated and self.trade_type == "TREND": self.runner_activated = True
                    elif len(self.partials_hit) == 1 and not self.early_profit_locked and self.trade_type != "TREND":
                        self.current_stop_loss = max(self.current_stop_loss, self.entry_price * 1.001) if self.side == "long" else min(self.current_stop_loss, self.entry_price * 0.999)
                    return {"reason": [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3][i], "price": current_price, "size": size}

            if self.bars_in_trade >= self.time_stop_bars:
                return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}
            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self) -> Dict:
        current_price = self.close[-1]
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100 if self.side == "long" else (self.entry_price - current_price) / self.entry_price * 100
        next_target_idx = len(self.partials_hit)
        next_target = self.take_profit_levels[next_target_idx] if next_target_idx < len(self.take_profit_levels) else None
        return {
            "entry_price": self.entry_price,
            "current_price": current_price,
            "stop_loss": self.current_stop_loss,
            "initial_stop": self.initial_stop_loss,
            "next_target": next_target,
            "all_targets": self.take_profit_levels,
            "remaining_position_pct": self.remaining_position,
            "pnl_pct": pnl_pct,
            "bars_in_trade": self.bars_in_trade,
            "partials_hit": len(self.partials_hit),
            "runner_active": self.runner_activated,
            "highest_reached": self.highest_price_reached,
            "lowest_reached": self.lowest_price_reached,
            "early_lock_atr_multiplier": self.early_lock_atr_multiplier, # New
            "runner_trail_atr_multiplier": self.runner_trail_atr_multiplier, # New
            "current_early_lock_threshold_pct": self.current_early_lock_threshold_pct, # New
            "current_runner_trail_pct": self.current_runner_trail_pct # New
        }

    def to_dict(self) -> Dict:
        return {"entry_price": self.entry_price, "side": self.side, "asset": self.asset, "position_size": self.position_size, "initial_stop_loss": self.initial_stop_loss, "current_stop_loss": self.current_stop_loss, "take_profit_levels": self.take_profit_levels, "partial_sizes": self.partial_sizes, "remaining_position": self.remaining_position, "partials_hit": self.partials_hit, "bars_in_trade": self.bars_in_trade, "highest_price_reached": self.highest_price_reached, "lowest_price_reached": self.lowest_price_reached, "runner_activated": self.runner_activated, "early_profit_locked": self.early_profit_locked, "trade_type": self.trade_type, "entry_time": self.entry_time.isoformat()}

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        vtm = cls(entry_price=state["entry_price"], side=state["side"], asset=state["asset"], high=high, low=low, close=close, quantity=state["position_size"], trade_type=state.get("trade_type", "TREND"))
        vtm.initial_stop_loss, vtm.current_stop_loss, vtm.take_profit_levels, vtm.partial_sizes, vtm.remaining_position, vtm.partials_hit, vtm.bars_in_trade, vtm.highest_price_reached, vtm.lowest_price_reached, vtm.runner_activated, vtm.early_profit_locked = state["initial_stop_loss"], state["current_stop_loss"], state["take_profit_levels"], state["partial_sizes"], state["remaining_position"], state["partials_hit"], state["bars_in_trade"], state["highest_price_reached"], state["lowest_price_reached"], state["runner_activated"], state["early_profit_locked"]
        return vtm

    def __repr__(self):
        levels = self.get_current_levels()
        return f"VTM({self.asset} {self.side.upper()}: Entry=${levels['entry_price']:.2f}, Current=${levels['current_price']:.2f}, SL=${levels['stop_loss']:.2f}, P&L={levels['pnl_pct']:+.2f}%)"

    