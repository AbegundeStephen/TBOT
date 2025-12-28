"""
Production-Ready Veteran Trade Manager
Optimized for live trading with BTC and GOLD
Features Gemini's structure-based stop logic + asset-specific tuning
ENHANCED: Structure-based take profits + critical bug fixes
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


class AssetProfile:
    """Asset-specific trading profiles"""

    BTC = {
        "name": "Bitcoin",
        "volatility": "high",
        "min_stop_pct": 0.08,
        "max_stop_pct": 0.15,
        "atr_multiplier": 2.0,
        "pivot_lookback": 30,
        "partial_targets": [1.5, 2.5, 3.5],  # FIXED: More realistic (was 4.0)
        "partial_sizes": [0.33, 0.33, 0.34],
        "runner_trail_pct": 0.20,
        "time_stop_bars": 30,
        "use_ema_structure": False,
        "use_structure_targets": True,  # NEW: Enable structure-based TPs
    }

    GOLD = {
        "name": "Gold",
        "volatility": "medium",
        "min_stop_pct": 0.02,
        "max_stop_pct": 0.05,
        "atr_multiplier": 1.5,
        "pivot_lookback": 20,
        "partial_targets": [1.5, 2.5, 3.5],
        "partial_sizes": [0.40, 0.30, 0.30],
        "runner_trail_pct": 0.12,
        "time_stop_bars": 20,
        "use_ema_structure": True,
        "use_structure_targets": True,  # NEW: Enable structure-based TPs
    }

    @classmethod
    def get_profile(cls, asset: str) -> Dict:
        """Get trading profile for asset"""
        asset = asset.upper()
        if asset in ["BTC", "BITCOIN", "BTCUSD"]:
            return cls.BTC
        elif asset in ["GOLD", "XAU", "XAUUSD"]:
            return cls.GOLD
        else:
            logger.warning(f"Unknown asset {asset}, using GOLD profile")
            return cls.GOLD


# ============================================================================
# HELPER FUNCTIONS FOR STRUCTURE-BASED TARGETS
# ============================================================================

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
    """
    Find significant resistance/support levels based on market structure
    """
    lookback = min(lookback, len(close))
    levels = []

    if side == "long":
        highs = high[-lookback:]
        
        for i in range(2, len(highs) - 2):
            if highs[i] > current_price:
                if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and 
                    highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
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
                if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and 
                    lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
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
    """
    Hybrid TP calculation: Uses structure when available, R:R as fallback
    """
    risk = abs(entry_price - stop_loss)
    targets = []
    adjusted_sizes = list(partial_sizes)

    if side == "long":
        rr_targets = [entry_price + (risk * r) for r in risk_multiples]
        
        for i, rr_target in enumerate(rr_targets):
            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (closest - entry_price) / risk
                
                if abs(closest - rr_target) / rr_target < 0.20 and structure_rr >= min_rr:
                    targets.append(closest)
                    logger.info(f"[VTM] ✓ TP{i+1}: Structure @ ${closest:,.2f} ({structure_rr:.1f}R)")
                else:
                    if structure_rr < risk_multiples[i]:
                        targets.append(closest)
                        logger.info(f"[VTM] ⚠ TP{i+1}: Capped by structure @ ${closest:,.2f} ({structure_rr:.1f}R)")
                    else:
                        targets.append(rr_target)
                        logger.info(f"[VTM] → TP{i+1}: Risk-based @ ${rr_target:,.2f} ({risk_multiples[i]:.1f}R)")
            else:
                targets.append(rr_target)
                logger.info(f"[VTM] → TP{i+1}: Risk-based @ ${rr_target:,.2f} ({risk_multiples[i]:.1f}R)")
    
    else:
        rr_targets = [entry_price - (risk * r) for r in risk_multiples]
        
        for i, rr_target in enumerate(rr_targets):
            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (entry_price - closest) / risk
                
                if abs(closest - rr_target) / abs(rr_target) < 0.20 and structure_rr >= min_rr:
                    targets.append(closest)
                    logger.info(f"[VTM] ✓ TP{i+1}: Structure @ ${closest:,.2f} ({structure_rr:.1f}R)")
                else:
                    if structure_rr < risk_multiples[i]:
                        targets.append(closest)
                        logger.info(f"[VTM] ⚠ TP{i+1}: Capped by structure @ ${closest:,.2f} ({structure_rr:.1f}R)")
                    else:
                        targets.append(rr_target)
                        logger.info(f"[VTM] → TP{i+1}: Risk-based @ ${rr_target:,.2f} ({risk_multiples[i]:.1f}R)")
            else:
                targets.append(rr_target)
                logger.info(f"[VTM] → TP{i+1}: Risk-based @ ${rr_target:,.2f} ({risk_multiples[i]:.1f}R)")
    
    if len(targets) < len(partial_sizes):
        remaining = sum(partial_sizes[len(targets):])
        for i in range(len(targets)):
            adjusted_sizes[i] = partial_sizes[i] + (remaining / len(targets))
        adjusted_sizes = adjusted_sizes[:len(targets)]
    
    return targets, adjusted_sizes


class VeteranTradeManager:
    """
    Production-ready trade manager with structure-based logic
    """

    def __init__(
        self,
        entry_price: float,
        side: str,
        asset: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        account_balance: float,
        signal_details: Dict,
        account_risk: float = 0.015,
        atr_period: int = 14,
        custom_profile: Optional[Dict] = None,
        enable_early_profit_lock: bool = False,
        early_lock_threshold_pct: float = 0.01,
    ):
        self.entry_price = entry_price
        self.side = side.lower()
        self.asset = asset.upper()
        self.high = high
        self.low = low
        self.close = close
        self.account_balance = account_balance
        self.account_risk = account_risk
        self.atr_period = atr_period
        self.signal_details = signal_details

        self.profile = custom_profile or AssetProfile.get_profile(asset)
        self.min_stop_pct = self.profile["min_stop_pct"]
        self.max_stop_pct = self.profile["max_stop_pct"]
        self.atr_multiplier = self.profile["atr_multiplier"]
        self.pivot_lookback = self.profile["pivot_lookback"]
        self.partial_targets = self.profile["partial_targets"]
        self.partial_sizes = self.profile["partial_sizes"]
        self.runner_trail_pct = self.profile["runner_trail_pct"]
        self.time_stop_bars = self.profile["time_stop_bars"]
        self.use_ema_structure = self.profile["use_ema_structure"]
        self.use_structure_targets = self.profile.get("use_structure_targets", True)

        self.enable_early_profit_lock = enable_early_profit_lock
        self.early_lock_threshold_pct = early_lock_threshold_pct
        self.early_profit_locked = False

        self.initial_stop_loss = None
        self.current_stop_loss = None
        self.take_profit_levels = []
        self.position_size = None
        self.remaining_position = 1.0
        self.partials_hit = []
        self.bars_in_trade = 0
        self.highest_price_reached = entry_price
        self.lowest_price_reached = entry_price
        self.runner_activated = False
        self.entry_time = datetime.now()

        self._calculate_initial_levels()

        logger.info("=" * 80)
        logger.info(f"🎯 VETERAN TRADE MANAGER - {self.asset} {side.upper()}")
        logger.info("=" * 80)
        logger.info(f"Asset Profile: {self.profile['name']} ({self.profile['volatility']} volatility)")
        logger.info(f"Entry Price:   ${entry_price:,.2f}")
        logger.info(f"Stop Loss:     ${self.initial_stop_loss:,.2f} (-{self._calc_pct_distance(entry_price, self.initial_stop_loss):.2f}%)")
        logger.info(f"Position Size: {self.position_size:.6f} units")
        logger.info(f"Risk Amount:   ${account_balance * account_risk:,.2f} ({account_risk:.1%})")
        logger.info(f"\n📊 PROFIT TARGETS:")
        for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes), 1):
            pct = self._calc_pct_distance(entry_price, target)
            logger.info(f"  {i}. ${target:,.2f} (+{pct:.2f}%) → Exit {size:.0%}")
        logger.info(f"\n⚙️  SETTINGS:")
        logger.info(f"  Stop Range: {self.min_stop_pct:.1%} - {self.max_stop_pct:.1%}")
        logger.info(f"  Runner Trail: {self.runner_trail_pct:.0%}")
        logger.info(f"  Structure TPs: {'Enabled' if self.use_structure_targets else 'Disabled'}")
        if self.enable_early_profit_lock:
            logger.info(f"  Early Profit Lock: {self.early_lock_threshold_pct:.1%} gain")
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

    def _find_pivot_structure(self) -> Optional[float]:
        try:
            lookback = min(self.pivot_lookback, len(self.close) - 3)

            if self.side == "long":
                for i in range(lookback - 2, 1, -1):
                    current = self.low[-(i + 1)]
                    left = self.low[-(i + 2)]
                    right = self.low[-i]
                    if current < left and current < right:
                        logger.info(f"[VTM] ✓ Pivot LOW @ ${current:,.2f} (bar -{i})")
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
                        logger.info(f"[VTM] ✓ Pivot HIGH @ ${current:,.2f} (bar -{i})")
                        return current
                fallback = np.max(self.high[-lookback:])
                logger.warning(f"[VTM] No pivot, using highest: ${fallback:,.2f}")
                return fallback
        except Exception as e:
            logger.error(f"[VTM] Pivot error: {e}")
            return None

    def _calculate_initial_levels(self):
        try:
            atr = self._calculate_atr()
            pivot_level = self._find_pivot_structure()

            if self.side == "long":
                atr_distance = atr * self.atr_multiplier
                atr_distance_pct = min(atr_distance / self.entry_price, self.max_stop_pct)
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 - atr_distance_pct)

                if pivot_level:
                    structure_stop = pivot_level - (atr * 1.0)  # FIXED: Full ATR buffer
                else:
                    structure_stop = self.entry_price * (1 - self.min_stop_pct)

                min_stop = self.entry_price * (1 - self.max_stop_pct)
                max_stop = self.entry_price * (1 - self.min_stop_pct)

                # FIXED: Take LOWER stop (wider)
                self.initial_stop_loss = max(min_stop, min(max_stop, min(atr_stop, structure_stop)))

                # FIXED: Safety check
                if abs(self.entry_price - self.initial_stop_loss) < atr:
                    logger.warning(f"[VTM] Stop too tight, widening to 1 ATR")
                    self.initial_stop_loss = self.entry_price - atr

                logger.info(f"[VTM] Stop Calc (LONG):")
                logger.info(f"  Pivot: ${pivot_level:,.2f}" if pivot_level else "  Pivot: None")
                logger.info(f"  ATR: ${atr:,.2f} (x{self.atr_multiplier})")
                logger.info(f"  ATR Stop: ${atr_stop:,.2f}")
                logger.info(f"  Structure Stop: ${structure_stop:,.2f}")
                logger.info(f"  → FINAL: ${self.initial_stop_loss:,.2f}")

                # Calculate TPs with structure
                if self.use_structure_targets:
                    logger.info(f"\n[VTM] Detecting resistance levels...")
                    structure_levels = find_resistance_levels(
                        self.high, self.low, self.close,
                        self.entry_price, self.side, self.pivot_lookback
                    )
                    if structure_levels:
                        logger.info(f"[VTM] Found {len(structure_levels)} resistance levels")
                    self.take_profit_levels, self.partial_sizes = calculate_hybrid_targets(
                        self.entry_price, self.initial_stop_loss, self.side,
                        structure_levels, self.partial_targets, self.partial_sizes, min_rr=1.2
                    )
                else:
                    risk = self.entry_price - self.initial_stop_loss
                    self.take_profit_levels = [self.entry_price + (risk * r) for r in self.partial_targets]

            else:  # short
                atr_distance = atr * self.atr_multiplier
                atr_distance_pct = min(atr_distance / self.entry_price, self.max_stop_pct)
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 + atr_distance_pct)

                if pivot_level:
                    structure_stop = pivot_level + (atr * 1.0)  # FIXED
                else:
                    structure_stop = self.entry_price * (1 + self.min_stop_pct)

                min_stop = self.entry_price * (1 + self.max_stop_pct)
                max_stop = self.entry_price * (1 + self.min_stop_pct)

                # FIXED: Take HIGHER stop (wider for shorts)
                self.initial_stop_loss = min(min_stop, max(max_stop, max(atr_stop, structure_stop)))

                if abs(self.initial_stop_loss - self.entry_price) < atr:
                    logger.warning(f"[VTM] Stop too tight, widening to 1 ATR")
                    self.initial_stop_loss = self.entry_price + atr

                logger.info(f"[VTM] Stop Calc (SHORT):")
                logger.info(f"  Pivot: ${pivot_level:,.2f}" if pivot_level else "  Pivot: None")
                logger.info(f"  ATR: ${atr:,.2f}")
                logger.info(f"  → FINAL: ${self.initial_stop_loss:,.2f}")

                if self.use_structure_targets:
                    logger.info(f"\n[VTM] Detecting support levels...")
                    structure_levels = find_resistance_levels(
                        self.high, self.low, self.close,
                        self.entry_price, self.side, self.pivot_lookback
                    )
                    if structure_levels:
                        logger.info(f"[VTM] Found {len(structure_levels)} support levels")
                    self.take_profit_levels, self.partial_sizes = calculate_hybrid_targets(
                        self.entry_price, self.initial_stop_loss, self.side,
                        structure_levels, self.partial_targets, self.partial_sizes, min_rr=1.2
                    )
                else:
                    risk = self.initial_stop_loss - self.entry_price
                    self.take_profit_levels = [self.entry_price - (risk * r) for r in self.partial_targets]

            self.current_stop_loss = self.initial_stop_loss

            risk_amount = self.account_balance * self.account_risk
            trade_risk = abs(self.entry_price - self.initial_stop_loss)
            self.position_size = risk_amount / trade_risk

            # FIXED: Position size validation
            if self.position_size * self.entry_price > self.account_balance * 2:
                logger.error(f"[VTM] Position size too large!")
                raise ValueError("Position exceeds 2x account balance")

            dollar_risk = abs(self.entry_price - self.initial_stop_loss) * self.position_size
            logger.info(f"[VTM] $ Risk: ${dollar_risk:,.2f} ({dollar_risk/self.account_balance:.2%})")

        except Exception as e:
            logger.error(f"[VTM] Init error: {e}", exc_info=True)
            raise

    def update_with_new_bar(self, new_high: float, new_low: float, new_close: float) -> Optional[Dict]:
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
        try:
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                if self.runner_activated and self.highest_price_reached > old_high:
                    new_trail = self.highest_price_reached * (1 - self.runner_trail_pct)
                    if new_trail > self.current_stop_loss:
                        self.current_stop_loss = new_trail
                        logger.info(f"[VTM] 🏃 Runner trail: ${self.current_stop_loss:,.2f}")
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                if self.runner_activated and self.lowest_price_reached < old_low:
                    new_trail = self.lowest_price_reached * (1 + self.runner_trail_pct)
                    if new_trail < self.current_stop_loss:
                        self.current_stop_loss = new_trail
                        logger.info(f"[VTM] 🏃 Runner trail: ${self.current_stop_loss:,.2f}")

            return self.check_exit(current_price)
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

    def check_exit(self, current_price: float) -> Optional[Dict]:
        if self.remaining_position <= 0:
            return None

        try:
            if self.side == "long":
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            # Early profit lock
            if (self.enable_early_profit_lock and not self.early_profit_locked and
                pnl_pct >= self.early_lock_threshold_pct):
                self.early_profit_locked = True
                if self.side == "long":
                    new_stop = self.entry_price * 1.002
                    if new_stop > self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        logger.info(f"🔒 Profit locked @ break-even")
                else:
                    new_stop = self.entry_price * 0.998
                    if new_stop < self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        logger.info(f"🔒 Profit locked @ break-even")

            # Check stop loss
            if self.side == "long":
                if current_price <= self.current_stop_loss:
                    actual_pnl = (current_price - self.entry_price) / self.entry_price * 100
                    if actual_pnl >= -0.5:
                        logger.info(f"🔒 STOPPED AT BREAK-EVEN")
                    else:
                        logger.info(f"🛑 STOP LOSS HIT: {actual_pnl:+.2f}%")
                    return {"reason": ExitReason.STOP_LOSS, "price": current_price, "size": self.remaining_position}
            else:
                if current_price >= self.current_stop_loss:
                    actual_pnl = (self.entry_price - current_price) / self.entry_price * 100
                    if actual_pnl >= -0.5:
                        logger.info(f"🔒 STOPPED AT BREAK-EVEN")
                    else:
                        logger.info(f"🛑 STOP LOSS HIT: {actual_pnl:+.2f}%")
                    return {"reason": ExitReason.STOP_LOSS, "price": current_price, "size": self.remaining_position}

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
                    logger.info(f"💰 PARTIAL #{i+1} HIT! Exit {size:.0%}, P&L: +{pnl:.2f}%")

                    if len(self.partials_hit) >= 2 and not self.runner_activated:
                        self.runner_activated = True
                        logger.info(f"🏃 RUNNER ACTIVATED")

                    if len(self.partials_hit) == 1 and not self.early_profit_locked:
                        if self.side == "long":
                            self.current_stop_loss = max(self.current_stop_loss, self.entry_price * 1.001)
                        else:
                            self.current_stop_loss = min(self.current_stop_loss, self.entry_price * 0.999)
                        logger.info(f"🔒 Stop → break-even")

                    return {
                        "reason": [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3][i],
                        "price": current_price,
                        "size": size
                    }

            # Check runner trailing
            if self.runner_activated:
                if self.side == "long" and current_price <= self.current_stop_loss:
                    logger.info(f"🏃 RUNNER TRAILING STOP HIT")
                    return {"reason": ExitReason.TRAILING_STOP, "price": current_price, "size": self.remaining_position}
                elif self.side == "short" and current_price >= self.current_stop_loss:
                    logger.info(f"🏃 RUNNER TRAILING STOP HIT")
                    return {"reason": ExitReason.TRAILING_STOP, "price": current_price, "size": self.remaining_position}

            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self) -> Dict:
        current_price = self.close[-1]
        if self.side == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

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
        }

    def __repr__(self):
        levels = self.get_current_levels()
        return (
            f"VeteranTradeManager({self.asset} {self.side.upper()}: "
            f"Entry=${levels['entry_price']:.2f}, "
            f"Current=${levels['current_price']:.2f}, "
            f"SL=${levels['stop_loss']:.2f}, "
            f"P&L={levels['pnl_pct']:+.2f}%)"
        )