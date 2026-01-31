"""
Veteran Trade Manager - Strategic/Tactical Risk Architecture
✨ REFACTORED: Institutional-grade risk management with pre-flight validation
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


class AssetProfile:
    """Asset-specific trading profiles"""
    
    BTC = {
        "name": "Bitcoin",
        "volatility": "high",
        "min_stop_pct": 0.008,      # 0.8% minimum (TIGHTENED FOR DYNAMIC ENTRY)
        "max_stop_pct": 0.06,       # 6% maximum
        "atr_multiplier": 1.8,
        "pivot_lookback": 30,
        "partial_targets": [1.5, 3.0, 5.0],  # BANK EARLY STRATEGY
        "partial_sizes": [0.45, 0.30, 0.25],
        "runner_trail_pct": 0.025,
        "time_stop_bars": 72,
        "use_ema_structure": False,
    }

    GOLD = {
        "name": "Gold",
        "volatility": "medium",
        "min_stop_pct": 0.0025,     # 0.25% minimum (ULTRA-TIGHT FOR PRECISION)
        "max_stop_pct": 0.025,      # 2.5% maximum
        "atr_multiplier": 1.5,
        "pivot_lookback": 25,
        "partial_targets": [1.5, 2.5, 4.0],  # FASTER BANKING
        "partial_sizes": [0.45, 0.30, 0.25],
        "runner_trail_pct": 0.015,
        "time_stop_bars": 60,
        "use_ema_structure": True,
    }

    @classmethod
    def get_profile(cls, asset: str) -> Dict:
        """Get trading profile for asset"""
        asset = asset.upper()
        if asset in ["BTC", "BITCOIN", "BTCUSD", "BTCUSDT"]:
            return cls.BTC
        elif asset in ["GOLD", "XAU", "XAUUSD"]:
            return cls.GOLD
        else:
            logger.warning(f"Unknown asset {asset}, using GOLD profile")
            return cls.GOLD


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
    - Accepts externally calculated position size (no internal risk calculation)
    - Validates trade economics before execution (pre-flight check)
    - Asymmetric constraints for TREND vs SCALP trades
    - Persistence support for state recovery
    """

    @classmethod
    def validate_trade_setup(
        cls,
        entry_price: float,
        stop_loss: float,
        trade_type: str = "TREND",
        asset: str = "BTC",
        min_profit_viability: float = 0.0025,  # 0.25% minimum (STRICT FEE GATE)
    ) -> Tuple[bool, str]:
        """
        ✨ NEW: Pre-flight validation (BEFORE paying exchange fees)
        
        Checks if the trade has viable economics:
        - Distance to first target > 0.5% (covers fees + spread)
        - Stop loss is valid for the trade type
        
        Returns:
            (is_valid, rejection_reason)
        """
        try:
            # Get profile
            profile = AssetProfile.get_profile(asset)
            
            # Asymmetric constraints
            if trade_type == "SCALP":
                max_stop = profile["min_stop_pct"] * 0.5  # Tighter for scalps
                min_rr = 1.5  # 1.5:1 minimum for scalps
            else:  # TREND
                max_stop = profile["max_stop_pct"]
                min_rr = 1.49  # 1.49:1 minimum for trends
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_pct = stop_distance / entry_price
            
            # Check 1: Stop too wide?
            if stop_distance_pct > max_stop:
                return False, f"Stop too wide: {stop_distance_pct:.2%} > {max_stop:.2%} (max for {trade_type})"
            
            # Check 2: Calculate first target
            risk_multiples = profile["partial_targets"]
            first_target = entry_price + (stop_distance * risk_multiples[0])
            
            # Check 3: Is profit viable?
            if stop_loss < entry_price:  # LONG
                profit_distance = abs(first_target - entry_price)
            else:  # SHORT
                profit_distance = abs(entry_price - first_target)
            
            profit_pct = profit_distance / entry_price
            
            if profit_pct < min_profit_viability:
                return False, f"Profit target too close: {profit_pct:.2%} < {min_profit_viability:.2%} (fees will eat it)"
            
            # Check 4: Risk/Reward ratio
            actual_rr = profit_distance / stop_distance
            if actual_rr < min_rr:
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
            logger.error(f"[VTM PRE-FLIGHT] Error: {e}")
            return False, f"Validation error: {str(e)}"

    def __init__(
        self,
        entry_price: float,
        side: str,
        asset: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        quantity: float,
        volume: Optional[np.ndarray] = None,
        # ✨ NEW: External position size (from Portfolio Manager)
        # Legacy compatibility arguments
        signal_details: Dict = None,
        account_balance: float = None,
        quantity_override: Optional[float] = None,
        account_risk: float = 0.015,
        atr_period: int = 14,
        custom_profile: Optional[Dict] = None,
        enable_early_profit_lock: bool = True,
        early_lock_threshold_pct: float = 0.01,
        trade_type: str = "TREND",
        # Legacy args (mapped but not used)
        min_stop_distance_pct: float = None,
        partial_targets: List[float] = None,
        use_ema_exit: bool = False,
        ema_period: int = 200,
        time_stop_bars: int = None,
    ):
        self.entry_price = entry_price
        self.side = side.lower()
        self.asset = asset.upper()
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.atr_period = atr_period
        self.signal_details = signal_details or {}
        self.trade_type = trade_type
        
        # ✨ CRITICAL: Use externally provided quantity
        self.position_size = quantity or quantity_override
        
        # Get profile (with legacy overrides)
        self.profile = custom_profile or AssetProfile.get_profile(asset)
        
        # ✨ Asymmetric constraints for SCALP vs TREND
        if self.trade_type == "SCALP":
            self.min_stop_pct = self.profile["min_stop_pct"] * 0.5
            self.max_stop_pct = self.profile["max_stop_pct"] * 0.7
            self.atr_multiplier = 1.0
            self.partial_targets = partial_targets or [1.5, 2.5]
            self.partial_sizes = [0.60, 0.40]
            self.enable_early_profit_lock = True
            self.early_lock_threshold_pct = 0.005  # 0.5% for scalps
        else:  # TREND
            self.min_stop_pct = min_stop_distance_pct or self.profile["min_stop_pct"]
            self.max_stop_pct = self.profile["max_stop_pct"]
            self.atr_multiplier = self.profile["atr_multiplier"]
            self.partial_targets = partial_targets or self.profile["partial_targets"]
            self.partial_sizes = self.profile["partial_sizes"]
            self.enable_early_profit_lock = enable_early_profit_lock
            self.early_lock_threshold_pct = early_lock_threshold_pct

        self.pivot_lookback = self.profile["pivot_lookback"]
        self.runner_trail_pct = self.profile["runner_trail_pct"]
        self.time_stop_bars = time_stop_bars or self.profile["time_stop_bars"]
        self.use_ema_structure = self.profile.get("use_ema_structure", False)
        
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
        logger.info(f"\n📊 TARGETS:")
        for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes), 1):
            pct = self._calc_pct_distance(entry_price, target)
            logger.info(f"  {i}. ${target:,.2f} (+{pct:.2f}%) → Exit {size:.0%}")
        logger.info("=" * 80)

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
        """
        ✨ ITERATION 3: Dynamic Promotion Logic
        
        After TP1 (1.5R) hits, check if trade should be promoted to Runner Mode.
        
        Criteria:
        - Volume > 1.2x Average, OR
        - Strong directional candle (close near high/low)
        
        Returns:
            True if promotion triggered (delete TP2/TP3, activate runner)
        """
        # Only check if TP1 was just hit
        if len(self.partials_hit) != 1:
            return False
        
        # Already promoted
        if self.runner_activated:
            return False
        
        try:
            # CRITERION 1: Volume Confirmation
            volume_ratio = 1.0
            if self.volume is not None and len(self.volume) > 20:
                # Exclude current (incomplete) bar's volume
                avg_vol = np.mean(self.volume[-21:-1]) 
                current_vol = self.volume[-1]
                if avg_vol > 0:
                    volume_ratio = current_vol / avg_vol
            
            volume_strong = volume_ratio > 1.2
            
            # CRITERION 2: Strong Directional Candle
            # Check if current price is in the "conviction zone" (top/bottom 20% of candle)
            if len(self.high) > 0 and len(self.low) > 0:
                latest_high = self.high[-1]
                latest_low = self.low[-1]
                candle_range = latest_high - latest_low
                
                if candle_range > 0:
                    if self.side == "long":
                        # For longs, check if closing near high
                        distance_from_high = (latest_high - current_price) / candle_range
                        candle_conviction = distance_from_high < 0.20
                    else:
                        # For shorts, check if closing near low
                        distance_from_low = (current_price - latest_low) / candle_range
                        candle_conviction = distance_from_low < 0.20
                else:
                    candle_conviction = False
            else:
                candle_conviction = False
            
            # PROMOTION DECISION
            if volume_strong or candle_conviction:
                logger.info("")
                logger.info("=" * 70)
                logger.info("🚀 TRADE PROMOTION TRIGGERED")
                logger.info("=" * 70)
                logger.info(f"  Volume Strong:  {'✓' if volume_strong else '✗'}")
                logger.info(f"  Candle Conviction: {'✓' if candle_conviction else '✗'}")
                logger.info(f"  Action: Deleting TP2/TP3, Activating Runner Mode")
                logger.info("=" * 70)
                logger.info("")
                
                # PROMOTE TO RUNNER
                self.runner_activated = True
                
                # Delete TP2 and TP3 (keep only remaining position for runner)
                if len(self.take_profit_levels) > 1:
                    self.take_profit_levels = []  # Clear all targets
                    self.partial_sizes = []  # Clear all partial sizes
                
                # Move stop to break-even immediately
                if self.side == "long":
                    self.current_stop_loss = max(
                        self.current_stop_loss, 
                        self.entry_price * 1.001
                    )
                else:
                    self.current_stop_loss = min(
                        self.current_stop_loss, 
                        self.entry_price * 0.999
                    )
                
                logger.info(f"[VTM] Stop moved to break-even: ${self.current_stop_loss:,.2f}")
                
                return True
            
            else:
                logger.info("[VTM] TP1 hit, but promotion criteria not met")
                logger.info(f"  Volume: {volume_ratio:.2f}x (need 1.2x)")
                logger.info(f"  Candle Conviction: {'✓' if candle_conviction else '✗'}")
                logger.info(f"  Keeping TP2/TP3, moving stop to break-even")
                
                # Keep TP2/TP3 but secure profit
                if self.side == "long":
                    self.current_stop_loss = max(
                        self.current_stop_loss, 
                        self.entry_price * 1.001
                    )
                else:
                    self.current_stop_loss = min(
                        self.current_stop_loss, 
                        self.entry_price * 0.999
                    )
                
                return False
        
        except Exception as e:
            logger.error(f"[VTM] Promotion check error: {e}")
            return False

    def _find_pivot_structure(self) -> Optional[float]:
        """Find market structure pivot for stop placement"""
        try:
            lookback = min(self.pivot_lookback, len(self.close) - 3)

            if self.side == "long":
                for i in range(lookback - 2, 1, -1):
                    current = self.low[-(i + 1)]
                    left = self.low[-(i + 2)]
                    right = self.low[-i]
                    if current < left and current < right:
                        logger.info(f"[VTM] ✓ Pivot LOW @ ${current:,.2f}")
                        return current
                fallback = np.min(self.low[-lookback:])
                logger.warning(f"[VTM] No pivot, using lowest: ${fallback:,.2f}")
                return fallback
            else:
                for i in range(lookback - 2, 1, -1):
                    current = self.high[-(i + 1)]
                    left = self.high[-(i + 2)]
                    right = self.high[-i]
                    if current > left and current > right:
                        logger.info(f"[VTM] ✓ Pivot HIGH @ ${current:,.2f}")
                        return current
                fallback = np.max(self.high[-lookback:])
                logger.warning(f"[VTM] No pivot, using highest: ${fallback:,.2f}")
                return fallback
        except Exception as e:
            logger.error(f"[VTM] Pivot error: {e}")
            return None

    def _calculate_initial_levels(self):
        """
        ✨ REFACTORED: Use SAFER (wider) stop between Structure and ATR
        """
        try:
            atr = self._calculate_atr()
            pivot_level = self._find_pivot_structure()

            if self.side == "long":
                # ATR-based stop
                atr_distance_pct = min(
                    (atr * self.atr_multiplier) / self.entry_price,
                    self.max_stop_pct
                )
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 - atr_distance_pct)

                # Structure-based stop
                if pivot_level:
                    structure_stop = pivot_level - (atr * 1.0)
                else:
                    structure_stop = self.entry_price * (1 - self.min_stop_pct)

                # ✨ CRITICAL: Use SAFER (wider/lower) stop
                preliminary_stop = min(atr_stop, structure_stop)
                
                # Apply bounds
                min_stop = self.entry_price * (1 - self.max_stop_pct)
                max_stop = self.entry_price * (1 - self.min_stop_pct)
                self.initial_stop_loss = max(min_stop, min(max_stop, preliminary_stop))

                # Calculate targets
                structure_levels = find_resistance_levels(
                    self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback
                )
                self.take_profit_levels, self.partial_sizes = calculate_hybrid_targets(
                    self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                    self.partial_targets, self.partial_sizes,
                    min_rr=1.5 if self.trade_type == "SCALP" else 2.0
                )

            else:  # short
                # ATR-based stop
                atr_distance_pct = min(
                    (atr * self.atr_multiplier) / self.entry_price,
                    self.max_stop_pct
                )
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 + atr_distance_pct)

                # Structure-based stop
                if pivot_level:
                    structure_stop = pivot_level + (atr * 1.0)
                else:
                    structure_stop = self.entry_price * (1 + self.min_stop_pct)

                # ✨ CRITICAL: Use SAFER (wider/higher) stop
                preliminary_stop = max(atr_stop, structure_stop)
                
                # Apply bounds
                min_stop = self.entry_price * (1 + self.max_stop_pct)
                max_stop = self.entry_price * (1 + self.min_stop_pct)
                self.initial_stop_loss = min(min_stop, max(max_stop, preliminary_stop))

                # Calculate targets
                structure_levels = find_resistance_levels(
                    self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback
                )
                self.take_profit_levels, self.partial_sizes = calculate_hybrid_targets(
                    self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                    self.partial_targets, self.partial_sizes,
                    min_rr=1.5 if self.trade_type == "SCALP" else 2.0
                )

            self.current_stop_loss = self.initial_stop_loss

        except Exception as e:
            logger.error(f"[VTM] Level calculation error: {e}", exc_info=True)
            raise

    def on_new_bar(self, new_high: float, new_low: float, new_close: float) -> Optional[Dict]:
        """
        ✨ RENAMED: Main update method (was update_with_new_bar)
        Process new bar and check for exits
        """
        try:
            self.high = np.append(self.high, new_high)
            self.low = np.append(self.low, new_low)
            self.close = np.append(self.close, new_close)

            if len(self.close) > 200:
                self.high = self.high[-200:]
                self.low = self.low[-200:]
                self.close = self.close[-200:]

            self.bars_in_trade += 1

            if self.side == "long":
                self.highest_price_reached = max(self.highest_price_reached, new_high)
            else:
                self.lowest_price_reached = min(self.lowest_price_reached, new_low)

            return self.check_exit(new_close)
        except Exception as e:
            logger.error(f"[VTM] Update error: {e}")
            return None

    def update_with_current_price(self, current_price: float) -> Optional[Dict]:
        """Real-time intra-bar update"""
        try:
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                
                if self.runner_activated and self.highest_price_reached > old_high and self.trade_type == "TREND":
                    new_trail = self.highest_price_reached * (1 - self.runner_trail_pct)
                    if new_trail > self.current_stop_loss:
                        self.current_stop_loss = new_trail
                        logger.info(f"[VTM] 🏃 Runner trail: ${self.current_stop_loss:,.2f}")
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                
                if self.runner_activated and self.lowest_price_reached < old_low and self.trade_type == "TREND":
                    new_trail = self.lowest_price_reached * (1 + self.runner_trail_pct)
                    if new_trail < self.current_stop_loss:
                        self.current_stop_loss = new_trail
                        logger.info(f"[VTM] 🏃 Runner trail: ${self.current_stop_loss:,.2f}")

            return self.check_exit(current_price)
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

    def check_exit(self, current_price: float) -> Optional[Dict]:
        """Check for exit conditions"""
        if self.remaining_position <= 0:
            return None

        try:
            if self.side == "long":
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            # Early profit lock (break-even)
            if (self.enable_early_profit_lock and not self.early_profit_locked and 
                pnl_pct >= self.early_lock_threshold_pct):
                self.early_profit_locked = True
                if self.side == "long":
                    new_stop = self.entry_price * 1.001
                    if new_stop > self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        logger.info(f"[VTM] 🛡️ Break-even @ +{pnl_pct:.2%}")
                else:
                    new_stop = self.entry_price * 0.999
                    if new_stop < self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        logger.info(f"[VTM] 🛡️ Break-even @ +{pnl_pct:.2%}")

            # Check stop loss
            if self.side == "long":
                if current_price <= self.current_stop_loss:
                    actual_pnl = ((current_price - self.entry_price) / self.entry_price) * 100
                    logger.info(f"🛑 STOP LOSS: {actual_pnl:+.2f}%")
                    return {
                        "reason": ExitReason.STOP_LOSS,
                        "price": current_price,
                        "size": self.remaining_position,
                    }
            else:
                if current_price >= self.current_stop_loss:
                    actual_pnl = ((self.entry_price - current_price) / self.entry_price) * 100
                    logger.info(f"🛑 STOP LOSS: {actual_pnl:+.2f}%")
                    return {
                        "reason": ExitReason.STOP_LOSS,
                        "price": current_price,
                        "size": self.remaining_position,
                    }

            # Check partials
            for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes)):
                if i in self.partials_hit:
                    continue

                hit = False
                if self.side == "long" and current_price >= target:
                    hit = True
                elif self.side == "short" and current_price <= target:
                    hit = True

                if hit:
                    self.partials_hit.append(i)
                    self.remaining_position -= size
                    pnl = abs(current_price - self.entry_price) / self.entry_price * 100
                    logger.info(f"💰 PARTIAL #{i+1}: +{pnl:.2f}%")

                    # ✨ ITERATION 3: Check for Promotion after TP1
                    if len(self.partials_hit) == 1 and self.trade_type == "TREND":
                        # NOTE: Volume ratio should be passed from the calling context
                        # For now, we'll trigger promotion check without volume
                        # This will be enhanced when integrated with execution handlers
                        promotion_triggered = self.check_promotion_to_runner(
                            current_price=current_price
                        )
                        
                        if not promotion_triggered:
                            # Promotion failed - secure profit by moving to break-even
                            if not self.early_profit_locked:
                                if self.side == "long":
                                    self.current_stop_loss = max(self.current_stop_loss, self.entry_price * 1.001)
                                else:
                                    self.current_stop_loss = min(self.current_stop_loss, self.entry_price * 0.999)
                                logger.info(f"🔒 Stop → break-even (TP2/TP3 remain)")

                    # Legacy runner activation (for multi-partial trades)
                    elif len(self.partials_hit) >= 2 and not self.runner_activated and self.trade_type == "TREND":
                        self.runner_activated = True
                        logger.info(f"🏃 RUNNER ACTIVATED (Legacy 2+ Partials)")

                    # Standard break-even for non-TREND or after first partial
                    elif len(self.partials_hit) == 1 and not self.early_profit_locked and self.trade_type != "TREND":
                        if self.side == "long":
                            self.current_stop_loss = max(self.current_stop_loss, self.entry_price * 1.001)
                        else:
                            self.current_stop_loss = min(self.current_stop_loss, self.entry_price * 0.999)
                        logger.info(f"🔒 Stop → break-even")

                    return {
                        "reason": [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3][i],
                        "price": current_price,
                        "size": size,
                    }

            # Time stop
            if self.bars_in_trade >= self.time_stop_bars:
                logger.info(f"⏰ TIME STOP: {self.bars_in_trade} bars")
                return {
                    "reason": ExitReason.TIME_STOP,
                    "price": current_price,
                    "size": self.remaining_position,
                }

            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self) -> Dict:
        """Get current trade status"""
        current_price = self.close[-1]
        if self.side == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

        next_target_idx = len(self.partials_hit)
        next_target = (
            self.take_profit_levels[next_target_idx]
            if next_target_idx < len(self.take_profit_levels)
            else None
        )

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
        }

    def to_dict(self) -> Dict:
        """
        ✨ NEW: Serialize state for persistence
        """
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
            "early_profit_locked": self.early_profit_locked,
            "trade_type": self.trade_type,
            "entry_time": self.entry_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        """
        ✨ NEW: Restore from saved state
        """
        vtm = cls(
            entry_price=state["entry_price"],
            side=state["side"],
            asset=state["asset"],
            high=high,
            low=low,
            close=close,
            quantity=state["position_size"],
            trade_type=state.get("trade_type", "TREND"),
        )
        
        # Restore state
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
        vtm.early_profit_locked = state["early_profit_locked"]
        
        return vtm

    def __repr__(self):
        levels = self.get_current_levels()
        return (
            f"VTM({self.asset} {self.side.upper()}: "
            f"Entry=${levels['entry_price']:.2f}, "
            f"Current=${levels['current_price']:.2f}, "
            f"SL=${levels['stop_loss']:.2f}, "
            f"P&L={levels['pnl_pct']:+.2f}%)"
        )

    