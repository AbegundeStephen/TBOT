"""
Production-Ready Veteran Trade Manager
Optimized for live trading with BTC and GOLD
Features Gemini's structure-based stop logic + asset-specific tuning
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
        "min_stop_pct": 0.08,  # 8% minimum stop
        "max_stop_pct": 0.15,  # 15% maximum stop
        "atr_multiplier": 2.0,  # Wider ATR buffer
        "pivot_lookback": 30,  # Look back 30 bars for pivots
        "partial_targets": [1.5, 2.5, 4.0],  # R:R ratios
        "partial_sizes": [0.33, 0.33, 0.34],
        "runner_trail_pct": 0.20,  # 20% trail for runner
        "time_stop_bars": 30,  # Give more time
        "use_ema_structure": False,  # Don't use EMA for BTC (too whippy)
    }

    GOLD = {
        "name": "Gold",
        "volatility": "medium",
        "min_stop_pct": 0.02,  # 2% minimum stop
        "max_stop_pct": 0.05,  # 5% maximum stop
        "atr_multiplier": 1.5,  # Tighter ATR buffer
        "pivot_lookback": 20,  # Look back 20 bars
        "partial_targets": [1.5, 2.5, 3.5],  # More conservative
        "partial_sizes": [0.40, 0.30, 0.30],  # Take more profit early
        "runner_trail_pct": 0.12,  # 12% trail (tighter)
        "time_stop_bars": 20,  # Exit dead trades faster
        "use_ema_structure": True,  # Gold respects EMA better
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


class VeteranTradeManager:
    """
    Production-ready trade manager with:
    - Gemini's structure-based stop loss (pivot detection)
    - Asset-specific profiles for BTC and GOLD
    - Partial profit taking
    - Real-time updates for live trading
    - Comprehensive logging
    """

    def __init__(
        self,
        entry_price: float,
        side: str,  # 'long' or 'short'
        asset: str,  # 'BTC' or 'GOLD'
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        account_balance: float,
        account_risk: float = 0.015,  # 1.5% risk per trade
        atr_period: int = 14,
        custom_profile: Optional[Dict] = None,  # Override defaults
        enable_early_profit_lock: bool = False,  # NEW: Lock profit early
        early_lock_threshold_pct: float = 0.01,  # NEW: Lock at 1% gain
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

        # Load asset profile
        self.profile = custom_profile or AssetProfile.get_profile(asset)

        # Apply profile settings
        self.min_stop_pct = self.profile["min_stop_pct"]
        self.max_stop_pct = self.profile["max_stop_pct"]
        self.atr_multiplier = self.profile["atr_multiplier"]
        self.pivot_lookback = self.profile["pivot_lookback"]
        self.partial_targets = self.profile["partial_targets"]
        self.partial_sizes = self.profile["partial_sizes"]
        self.runner_trail_pct = self.profile["runner_trail_pct"]
        self.time_stop_bars = self.profile["time_stop_bars"]
        self.use_ema_structure = self.profile["use_ema_structure"]

        # Profit lock settings
        self.enable_early_profit_lock = enable_early_profit_lock
        self.early_lock_threshold_pct = early_lock_threshold_pct
        self.early_profit_locked = False

        # Trade state
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

        # Initialize levels
        self._calculate_initial_levels()

        logger.info("=" * 80)
        logger.info(f"🎯 VETERAN TRADE MANAGER - {self.asset} {side.upper()}")
        logger.info("=" * 80)
        logger.info(
            f"Asset Profile: {self.profile['name']} ({self.profile['volatility']} volatility)"
        )
        logger.info(f"Entry Price:   ${entry_price:,.2f}")
        logger.info(
            f"Stop Loss:     ${self.initial_stop_loss:,.2f} "
            f"(-{self._calc_pct_distance(entry_price, self.initial_stop_loss):.2f}%)"
        )
        logger.info(f"Position Size: {self.position_size:.6f} units")
        logger.info(
            f"Risk Amount:   ${account_balance * account_risk:,.2f} ({account_risk:.1%})"
        )
        logger.info(f"\n📊 PROFIT TARGETS:")
        for i, (target, size) in enumerate(
            zip(self.take_profit_levels, self.partial_sizes), 1
        ):
            pct = self._calc_pct_distance(entry_price, target)
            logger.info(f"  {i}. ${target:,.2f} (+{pct:.2f}%) → Exit {size:.0%}")
        logger.info(f"\n⚙️  SETTINGS:")
        logger.info(f"  Stop Range: {self.min_stop_pct:.1%} - {self.max_stop_pct:.1%}")
        logger.info(f"  Runner Trail: {self.runner_trail_pct:.0%}")
        logger.info(f"  Time Stop: {self.time_stop_bars} bars")
        logger.info(
            f"  EMA Structure: {'Enabled' if self.use_ema_structure else 'Disabled'}"
        )
        if self.enable_early_profit_lock:
            logger.info(
                f"  Early Profit Lock: {self.early_lock_threshold_pct:.1%} gain"
            )
        logger.info("=" * 80)

    def _calc_pct_distance(self, price1: float, price2: float) -> float:
        """Calculate percentage distance"""
        return abs(price1 - price2) / price1 * 100

    def _calculate_atr(self) -> float:
        """Calculate ATR with fallback"""
        try:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=self.atr_period)
            current_atr = atr[-1]
            if np.isnan(current_atr) or current_atr <= 0:
                # Fallback: 2% of price
                return self.entry_price * 0.02
            return current_atr
        except Exception as e:
            logger.error(f"[VTM] ATR error: {e}")
            return self.entry_price * 0.02

    def _find_pivot_structure(self) -> Optional[float]:
        """
        🔍 GEMINI'S STRUCTURE-BASED STOP LOGIC
        Find the most recent pivot (V-shape for long, Λ-shape for short)
        """
        try:
            lookback = min(self.pivot_lookback, len(self.close) - 3)

            if self.side == "long":
                # Find pivot LOW (V-shape: left > center < right)
                for i in range(lookback - 2, 1, -1):
                    current = self.low[-(i + 1)]
                    left = self.low[-(i + 2)]
                    right = self.low[-i]

                    # V-shape check
                    if current < left and current < right:
                        logger.info(
                            f"[VTM] ✓ Found pivot LOW @ ${current:,.2f} "
                            f"(V-shape confirmed at bar -{i})"
                        )
                        return current

                # Fallback: Use absolute lowest
                fallback = np.min(self.low[-lookback:])
                logger.warning(f"[VTM] No pivot found, using lowest: ${fallback:,.2f}")
                return fallback

            else:  # short
                # Find pivot HIGH (Λ-shape: left < current > right)
                for i in range(lookback - 2, 1, -1):
                    current = self.high[-(i + 1)]
                    left = self.high[-(i + 2)]
                    right = self.high[-i]

                    # Λ-shape check
                    if current > left and current > right:
                        logger.info(
                            f"[VTM] ✓ Found pivot HIGH @ ${current:,.2f} "
                            f"(Λ-shape confirmed at bar -{i})"
                        )
                        return current

                fallback = np.max(self.high[-lookback:])
                logger.warning(f"[VTM] No pivot found, using highest: ${fallback:,.2f}")
                return fallback

        except Exception as e:
            logger.error(f"[VTM] Pivot detection error: {e}")
            return None

    def _calculate_initial_levels(self):
        """Calculate initial stop and targets using Gemini's logic + ATR buffer"""
        try:
            atr = self._calculate_atr()
            pivot_level = self._find_pivot_structure()

            if self.side == "long":
                # Calculate stop options
                # 1. ATR-based stop
                atr_distance = atr * self.atr_multiplier
                atr_distance_pct = min(
                    atr_distance / self.entry_price, self.max_stop_pct
                )
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 - atr_distance_pct)

                # 2. Structure-based stop (Gemini's logic)
                if pivot_level:
                    # Apply ATR buffer BELOW pivot
                    structure_stop = pivot_level - (atr * 0.5)  # Half ATR below pivot
                else:
                    structure_stop = self.entry_price * (1 - self.min_stop_pct)

                # 3. Enforce min/max constraints
                min_stop = self.entry_price * (1 - self.max_stop_pct)
                max_stop = self.entry_price * (1 - self.min_stop_pct)

                # Choose the wider of ATR or structure (more breathing room)
                self.initial_stop_loss = max(
                    min_stop,  # Don't go too wide
                    min(max_stop, max(atr_stop, structure_stop)),  # Don't go too tight
                )

                # Log breakdown
                logger.info(f"[VTM] Stop Loss Calculation (LONG):")
                logger.info(
                    f"  Pivot Level:     ${pivot_level:,.2f}"
                    if pivot_level
                    else "  Pivot: Not found"
                )
                logger.info(f"  ATR:             ${atr:,.2f} (x{self.atr_multiplier})")
                logger.info(
                    f"  ATR Stop:        ${atr_stop:,.2f} ({atr_distance_pct:.2%})"
                )
                logger.info(f"  Structure Stop:  ${structure_stop:,.2f}")
                logger.info(
                    f"  → FINAL STOP:    ${self.initial_stop_loss:,.2f} "
                    f"({self._calc_pct_distance(self.entry_price, self.initial_stop_loss):.2%} below)"
                )

                # Calculate targets based on risk
                risk = self.entry_price - self.initial_stop_loss
                self.take_profit_levels = [
                    self.entry_price + (risk * r) for r in self.partial_targets
                ]

            else:  # short
                atr_distance = atr * self.atr_multiplier
                atr_distance_pct = min(
                    atr_distance / self.entry_price, self.max_stop_pct
                )
                atr_distance_pct = max(atr_distance_pct, self.min_stop_pct)
                atr_stop = self.entry_price * (1 + atr_distance_pct)

                if pivot_level:
                    structure_stop = pivot_level + (atr * 0.5)
                else:
                    structure_stop = self.entry_price * (1 + self.min_stop_pct)

                min_stop = self.entry_price * (1 + self.max_stop_pct)
                max_stop = self.entry_price * (1 + self.min_stop_pct)

                self.initial_stop_loss = min(
                    min_stop, max(max_stop, min(atr_stop, structure_stop))
                )

                logger.info(f"[VTM] Stop Loss Calculation (SHORT):")
                logger.info(
                    f"  Pivot Level:     ${pivot_level:,.2f}"
                    if pivot_level
                    else "  Pivot: Not found"
                )
                logger.info(f"  ATR:             ${atr:,.2f} (x{self.atr_multiplier})")
                logger.info(
                    f"  ATR Stop:        ${atr_stop:,.2f} ({atr_distance_pct:.2%})"
                )
                logger.info(f"  Structure Stop:  ${structure_stop:,.2f}")
                logger.info(
                    f"  → FINAL STOP:    ${self.initial_stop_loss:,.2f} "
                    f"({self._calc_pct_distance(self.entry_price, self.initial_stop_loss):.2%} above)"
                )

                risk = self.initial_stop_loss - self.entry_price
                self.take_profit_levels = [
                    self.entry_price - (risk * r) for r in self.partial_targets
                ]

            self.current_stop_loss = self.initial_stop_loss

            # Calculate position size
            risk_amount = self.account_balance * self.account_risk
            trade_risk = abs(self.entry_price - self.initial_stop_loss)
            self.position_size = risk_amount / trade_risk

        except Exception as e:
            logger.error(f"[VTM] Initialization error: {e}", exc_info=True)
            raise

    def update_with_new_bar(
        self, new_high: float, new_low: float, new_close: float
    ) -> Optional[Dict]:
        """
        Update manager with new bar data
        Returns exit info if exit triggered
        """
        try:
            # Update price arrays
            self.high = np.append(self.high, new_high)
            self.low = np.append(self.low, new_low)
            self.close = np.append(self.close, new_close)

            # Trim history
            max_history = 200
            if len(self.close) > max_history:
                self.high = self.high[-max_history:]
                self.low = self.low[-max_history:]
                self.close = self.close[-max_history:]

            self.bars_in_trade += 1

            # Update highest/lowest
            if self.side == "long":
                self.highest_price_reached = max(self.highest_price_reached, new_high)
            else:
                self.lowest_price_reached = min(self.lowest_price_reached, new_low)

            # Check all exit conditions
            return self.check_exit(new_close)

        except Exception as e:
            logger.error(f"[VTM] Update error: {e}", exc_info=True)
            return None

    def update_with_current_price(self, current_price: float) -> Optional[Dict]:
        """
        Update with current price (for intrabar updates in live trading)
        Use this for real-time monitoring between bars
        """
        try:
            # Update extremes
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(
                    self.highest_price_reached, current_price
                )

                # Trail stop if runner active and new high
                if self.runner_activated and self.highest_price_reached > old_high:
                    new_trail = self.highest_price_reached * (1 - self.runner_trail_pct)
                    if new_trail > self.current_stop_loss:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_trail
                        logger.info(
                            f"[VTM] 🏃 Runner trail: ${old_stop:,.2f} → "
                            f"${self.current_stop_loss:,.2f} (High: ${self.highest_price_reached:,.2f})"
                        )
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(
                    self.lowest_price_reached, current_price
                )

                if self.runner_activated and self.lowest_price_reached < old_low:
                    new_trail = self.lowest_price_reached * (1 + self.runner_trail_pct)
                    if new_trail < self.current_stop_loss:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_trail
                        logger.info(
                            f"[VTM] 🏃 Runner trail: ${old_stop:,.2f} → "
                            f"${self.current_stop_loss:,.2f} (Low: ${self.lowest_price_reached:,.2f})"
                        )

            # Check for exits
            return self.check_exit(current_price)

        except Exception as e:
            logger.error(f"[VTM] Real-time update error: {e}")
            return None

    def check_exit(self, current_price: float) -> Optional[Dict]:
        """
        Check all exit conditions
        Returns: {'reason': ExitReason, 'price': float, 'size': float} or None
        """
        if self.remaining_position <= 0:
            return None

        try:
            # Calculate current P&L
            if self.side == "long":
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            # 🔒 EARLY PROFIT LOCK (if enabled)
            if (
                self.enable_early_profit_lock
                and not self.early_profit_locked
                and pnl_pct >= self.early_lock_threshold_pct
            ):

                self.early_profit_locked = True

                # Move stop to break-even + small buffer
                if self.side == "long":
                    new_stop = self.entry_price * 1.002  # 0.2% above entry
                    if new_stop > self.current_stop_loss:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_stop
                        logger.info(
                            f"🔒 EARLY PROFIT LOCKED @ {pnl_pct:.2%} gain\n"
                            f"   Stop: ${old_stop:,.2f} → ${self.current_stop_loss:,.2f} (break-even)"
                        )
                else:
                    new_stop = self.entry_price * 0.998
                    if new_stop < self.current_stop_loss:
                        old_stop = self.current_stop_loss
                        self.current_stop_loss = new_stop
                        logger.info(
                            f"🔒 EARLY PROFIT LOCKED @ {pnl_pct:.2%} gain\n"
                            f"   Stop: ${old_stop:,.2f} → ${self.current_stop_loss:,.2f} (break-even)"
                        )

            # 1. Check stop loss (current_stop_loss, not initial - allows for profit lock)
            if self.side == "long":
                if current_price <= self.current_stop_loss:
                    # Determine if this is break-even or loss
                    actual_pnl = (
                        (current_price - self.entry_price) / self.entry_price * 100
                    )
                    if actual_pnl >= -0.5:  # Within 0.5% of break-even
                        logger.info(
                            f"\n{'='*80}\n"
                            f"🔒 STOPPED AT BREAK-EVEN (Profit was locked)\n"
                            f"{'='*80}\n"
                            f"Price: ${current_price:,.2f} | Stop: ${self.current_stop_loss:,.2f}\n"
                            f"P&L: {actual_pnl:+.2f}% | Bars: {self.bars_in_trade}\n"
                            f"Partials Taken: {len(self.partials_hit)}\n"
                            f"{'='*80}"
                        )
                    else:
                        logger.info(
                            f"\n{'='*80}\n"
                            f"🛑 STOP LOSS HIT\n"
                            f"{'='*80}\n"
                            f"Price: ${current_price:,.2f} | Stop: ${self.current_stop_loss:,.2f}\n"
                            f"P&L: {actual_pnl:+.2f}% | Bars: {self.bars_in_trade}\n"
                            f"{'='*80}"
                        )
                    return {
                        "reason": ExitReason.STOP_LOSS,
                        "price": current_price,
                        "size": self.remaining_position,
                    }
            else:
                if current_price >= self.current_stop_loss:
                    actual_pnl = (
                        (self.entry_price - current_price) / self.entry_price * 100
                    )
                    if actual_pnl >= -0.5:
                        logger.info(
                            f"\n{'='*80}\n"
                            f"🔒 STOPPED AT BREAK-EVEN (Profit was locked)\n"
                            f"{'='*80}\n"
                            f"Price: ${current_price:,.2f} | Stop: ${self.current_stop_loss:,.2f}\n"
                            f"P&L: {actual_pnl:+.2f}% | Bars: {self.bars_in_trade}\n"
                            f"Partials Taken: {len(self.partials_hit)}\n"
                            f"{'='*80}"
                        )
                    else:
                        logger.info(
                            f"\n{'='*80}\n"
                            f"🛑 STOP LOSS HIT\n"
                            f"{'='*80}\n"
                            f"Price: ${current_price:,.2f} | Stop: ${self.current_stop_loss:,.2f}\n"
                            f"P&L: {actual_pnl:+.2f}% | Bars: {self.bars_in_trade}\n"
                            f"{'='*80}"
                        )
                    return {
                        "reason": ExitReason.STOP_LOSS,
                        "price": current_price,
                        "size": self.remaining_position,
                    }

            # 2. Check partial profit targets
            for i, (target, size) in enumerate(
                zip(self.take_profit_levels, self.partial_sizes)
            ):
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

                    pnl_pct = (
                        abs(current_price - self.entry_price) / self.entry_price * 100
                    )

                    logger.info(
                        f"\n{'='*80}\n"
                        f"💰 PARTIAL PROFIT #{i+1} HIT!\n"
                        f"{'='*80}\n"
                        f"Target: ${target:,.2f} | Price: ${current_price:,.2f}\n"
                        f"Exit Size: {size:.0%} | Remaining: {self.remaining_position:.0%}\n"
                        f"P&L: +{pnl_pct:.2f}% | Bars: {self.bars_in_trade}\n"
                        f"{'='*80}"
                    )

                    # Activate runner after 2nd partial
                    if len(self.partials_hit) >= 2 and not self.runner_activated:
                        self.runner_activated = True
                        logger.info(
                            f"🏃 RUNNER ACTIVATED: {self.remaining_position:.0%} position "
                            f"now trails with {self.runner_trail_pct:.0%} stop"
                        )

                    # Move stop to break-even after first partial (if not already locked)
                    if len(self.partials_hit) == 1 and not self.early_profit_locked:
                        if self.side == "long":
                            self.current_stop_loss = max(
                                self.current_stop_loss,
                                self.entry_price * 1.001,  # 0.1% above entry
                            )
                        else:
                            self.current_stop_loss = min(
                                self.current_stop_loss, self.entry_price * 0.999
                            )
                        logger.info(
                            f"🔒 Stop moved to break-even: ${self.current_stop_loss:,.2f}"
                        )

                    return {
                        "reason": [
                            ExitReason.TAKE_PROFIT_1,
                            ExitReason.TAKE_PROFIT_2,
                            ExitReason.TAKE_PROFIT_3,
                        ][i],
                        "price": current_price,
                        "size": size,
                    }

            # 3. Check runner trailing stop
            if self.runner_activated:
                if self.side == "long" and current_price <= self.current_stop_loss:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"🏃 RUNNER TRAILING STOP HIT\n"
                        f"{'='*80}\n"
                        f"Price: ${current_price:,.2f} | Trail: ${self.current_stop_loss:,.2f}\n"
                        f"High: ${self.highest_price_reached:,.2f}\n"
                        f"{'='*80}"
                    )
                    return {
                        "reason": ExitReason.TRAILING_STOP,
                        "price": current_price,
                        "size": self.remaining_position,
                    }
                elif self.side == "short" and current_price >= self.current_stop_loss:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"🏃 RUNNER TRAILING STOP HIT\n"
                        f"{'='*80}\n"
                        f"Price: ${current_price:,.2f} | Trail: ${self.current_stop_loss:,.2f}\n"
                        f"Low: ${self.lowest_price_reached:,.2f}\n"
                        f"{'='*80}"
                    )
                    return {
                        "reason": ExitReason.TRAILING_STOP,
                        "price": current_price,
                        "size": self.remaining_position,
                    }

            return None

        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}", exc_info=True)
            return None

    def get_current_levels(self) -> Dict:
        """Get current stop/target levels for display"""
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

    def __repr__(self):
        levels = self.get_current_levels()
        return (
            f"VeteranTradeManager({self.asset} {self.side.upper()}: "
            f"Entry=${levels['entry_price']:.2f}, "
            f"Current=${levels['current_price']:.2f}, "
            f"SL=${levels['stop_loss']:.2f}, "
            f"P&L={levels['pnl_pct']:+.2f}%)"
        )


# ============================================================================
# USAGE EXAMPLES FOR LIVE TRADING
# ============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(
        """
╔════════════════════════════════════════════════════════════════════════╗
║        PRODUCTION VTM - LIVE TRADING USAGE EXAMPLES                    ║
╚════════════════════════════════════════════════════════════════════════╝

EXAMPLE 1: BTC Long Position
─────────────────────────────────────────────────────────────────────────
"""
    )

    # Simulate BTC price data
    np.random.seed(42)
    btc_close = 50000 + np.cumsum(np.random.randn(100) * 500)
    btc_high = btc_close + np.abs(np.random.randn(100) * 300)
    btc_low = btc_close - np.abs(np.random.randn(100) * 300)

    # Initialize for BTC long
    btc_manager = VeteranTradeManager(
        entry_price=50000,
        side="long",
        asset="BTC",
        high=btc_high[:50],
        low=btc_low[:50],
        close=btc_close[:50],
        account_balance=10000,
        account_risk=0.015,
    )

    print(f"\n✓ Manager initialized")
    print(f"\nCurrent Levels:")
    levels = btc_manager.get_current_levels()
    for key, value in levels.items():
        if isinstance(value, float):
            print(
                f"  {key}: ${value:,.2f}"
                if "price" in key or "stop" in key or "target" in key
                else f"  {key}: {value:.2%}" if "pct" in key else f"  {key}: {value}"
            )
        else:
            print(f"  {key}: {value}")

    print(
        """

EXAMPLE 2: GOLD Short Position
─────────────────────────────────────────────────────────────────────────
"""
    )

    # Simulate GOLD price data
    gold_close = 2000 + np.cumsum(np.random.randn(100) * 10)
    gold_high = gold_close + np.abs(np.random.randn(100) * 5)
    gold_low = gold_close - np.abs(np.random.randn(100) * 5)

    # Initialize for GOLD short
    gold_manager = VeteranTradeManager(
        entry_price=2000,
        side="short",
        asset="GOLD",
        high=gold_high[:50],
        low=gold_low[:50],
        close=gold_close[:50],
        account_risk=0.015,
    )
