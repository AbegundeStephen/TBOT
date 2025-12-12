"""
Dynamic Trade Manager with ATR-based trailing stops and S/R levels
Supports both LONG and SHORT positions with intelligent exit management
FIXED: Realistic stop loss calculations with ATR caps
"""

import logging
import numpy as np
import talib
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DynamicTradeManager:
    """
    Manages trades with dynamic Stop Loss and Take Profit.
    Adjusts SL and TP based on ATR and updated support/resistance levels.
    Supports both LONG and SHORT positions.
    """

    def __init__(
        self,
        entry_price: float,
        side: str,  # 'long' or 'short'
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        account_balance: float,
        account_risk: float = 0.01,
        atr_period: int = 14,
        reward_risk_ratio: float = 2.0,
        sr_window: int = 20,
        atr_multiplier: float = 1.5,
        min_profit_lock: float = 0.005,  # Lock profit after 0.5% gain
        aggressive_trail: bool = False,
        max_stop_loss_pct: float = 0.08,  # ✅ NEW: Maximum SL distance (8%)
        min_stop_loss_pct: float = 0.015,  # ✅ NEW: Minimum SL distance (1.5%)
    ):
        self.entry_price = entry_price
        self.side = side.lower()
        self.high = high
        self.low = low
        self.close = close
        self.account_balance = account_balance
        self.account_risk = account_risk
        self.atr_period = atr_period
        self.reward_risk_ratio = reward_risk_ratio
        self.sr_window = sr_window
        self.atr_multiplier = atr_multiplier
        self.min_profit_lock = min_profit_lock
        self.aggressive_trail = aggressive_trail
        self.max_stop_loss_pct = max_stop_loss_pct  # ✅ NEW
        self.min_stop_loss_pct = min_stop_loss_pct  # ✅ NEW

        # State tracking
        self.stop_loss = None
        self.take_profit = None
        self.position_size = None
        self.highest_price = entry_price if side == 'long' else entry_price
        self.lowest_price = entry_price if side == 'short' else entry_price
        self.profit_locked = False
        self.update_count = 0
        self.last_update = datetime.now()

        # Initialize levels
        self.update_levels(initial=True)

        logger.info(
            f"[DTM] Trade Manager initialized for {side.upper()} @ ${entry_price:,.2f}\n"
            f"      Initial SL: ${self.stop_loss:,.2f} | TP: ${self.take_profit:,.2f}\n"
            f"      Position Size: {self.position_size:.6f} | Risk: {account_risk:.1%}\n"
            f"      SL Range: {self.min_stop_loss_pct:.1%} - {self.max_stop_loss_pct:.1%}"
        )

    def calculate_atr(self) -> float:
        """Calculate current ATR"""
        try:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=self.atr_period)
            return atr[-1] if not np.isnan(atr[-1]) else 0.0
        except Exception as e:
            logger.error(f"[DTM] ATR calculation error: {e}")
            return 0.0

    def detect_support_resistance(self) -> Tuple[float, float]:
        """Detect dynamic support and resistance levels"""
        try:
            support = np.min(self.low[-self.sr_window:])
            resistance = np.max(self.high[-self.sr_window:])
            return support, resistance
        except Exception as e:
            logger.error(f"[DTM] S/R detection error: {e}")
            return self.entry_price * 0.95, self.entry_price * 1.05

    def _calculate_atr_based_stop(
        self, 
        reference_price: float, 
        atr: float, 
        is_long: bool
    ) -> float:
        """
        ✅ NEW: Calculate ATR-based stop with realistic caps
        
        Args:
            reference_price: Price to calculate from (entry or current)
            atr: Current ATR value
            is_long: True for long, False for short
            
        Returns:
            Stop loss price with ATR capped to reasonable percentage
        """
        # Calculate raw ATR-based distance
        raw_atr_distance = atr * self.atr_multiplier
        
        # Convert to percentage
        raw_atr_pct = raw_atr_distance / reference_price
        
        # ✅ Cap ATR distance to max_stop_loss_pct
        capped_atr_pct = min(raw_atr_pct, self.max_stop_loss_pct)
        
        # Ensure minimum distance for breathing room
        final_atr_pct = max(capped_atr_pct, self.min_stop_loss_pct)
        
        # Calculate stop loss price
        if is_long:
            stop_loss = reference_price * (1 - final_atr_pct)
        else:
            stop_loss = reference_price * (1 + final_atr_pct)
        
        # Log if ATR was capped
        if raw_atr_pct > self.max_stop_loss_pct:
            logger.info(
                f"[DTM] ATR capped: Raw={raw_atr_pct:.2%} → Capped={final_atr_pct:.2%} "
                f"(ATR=${raw_atr_distance:,.2f})"
            )
        
        return stop_loss

    def update_levels(self, initial: bool = False, current_price: Optional[float] = None):
        """Update SL and TP dynamically based on market conditions"""
        try:
            last_atr = self.calculate_atr()
            if last_atr == 0.0:
                logger.warning("[DTM] ATR is zero, using fallback values")
                last_atr = self.entry_price * 0.02  # 2% fallback

            support, resistance = self.detect_support_resistance()
            
            if current_price is None:
                current_price = self.close[-1]

            # Track highest/lowest prices
            if self.side == 'long':
                self.highest_price = max(self.highest_price, current_price)
            else:
                self.lowest_price = min(self.lowest_price, current_price)

            # Calculate profit percentage
            if self.side == 'long':
                profit_pct = (current_price - self.entry_price) / self.entry_price
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price

            # Check if we should lock in profit
            if profit_pct >= self.min_profit_lock and not self.profit_locked:
                self.profit_locked = True
                
                # ✅ ACTUALLY LOCK PROFIT: Move SL to break-even or better
                if self.side == 'long':
                    breakeven_sl = self.entry_price * 1.002  # 0.2% above entry
                    if breakeven_sl > self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = breakeven_sl
                        logger.info(
                            f"[DTM] 🔒 PROFIT LOCKED: SL → break-even "
                            f"${old_sl:,.2f} → ${self.stop_loss:,.2f}"
                        )
                else:  # short
                    breakeven_sl = self.entry_price * 0.998
                    if breakeven_sl < self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = breakeven_sl
                        logger.info(
                            f"[DTM] 🔒 PROFIT LOCKED: SL → break-even "
                            f"${old_sl:,.2f} → ${self.stop_loss:,.2f}"
                        )
                
                logger.info(f"[DTM] Profit locked at {profit_pct:.2%} gain")

            # === LONG POSITION ===
            if self.side == 'long':
                if initial:
                    # ✅ FIXED: Use capped ATR-based stop
                    sl_atr = self._calculate_atr_based_stop(
                        reference_price=self.entry_price,
                        atr=last_atr,
                        is_long=True
                    )
                    
                    # Consider support, but don't let it create excessive risk
                    sl_support = support * 0.995
                    
                    # Choose the tighter of the two (closer to entry = less risk)
                    self.stop_loss = max(sl_atr, sl_support)
                    
                    # ✅ Final safety check: ensure within max distance
                    max_sl_price = self.entry_price * (1 - self.max_stop_loss_pct)
                    if self.stop_loss < max_sl_price:
                        logger.warning(
                            f"[DTM] SL too wide, clamping: "
                            f"${self.stop_loss:,.2f} → ${max_sl_price:,.2f}"
                        )
                        self.stop_loss = max_sl_price
                    
                    # Log breakdown
                    actual_sl_pct = (self.entry_price - self.stop_loss) / self.entry_price
                    logger.info(
                        f"[DTM] Initial SL breakdown:\n"
                        f"      ATR-based: ${sl_atr:,.2f} (ATR=${last_atr:,.2f})\n"
                        f"      Support: ${sl_support:,.2f}\n"
                        f"      → Selected: ${self.stop_loss:,.2f} ({actual_sl_pct:.2%} below entry)"
                    )
                else:
                    # Trailing SL logic
                    if self.profit_locked or self.aggressive_trail:
                        # Aggressive trailing: follow price closely, IGNORE support
                        aggressive_atr_distance = min(
                            last_atr * 0.8,
                            self.highest_price * self.max_stop_loss_pct
                        )
                        new_sl = self.highest_price - aggressive_atr_distance
                    else:
                        # Conservative trailing: use capped ATR + support
                        new_sl = self._calculate_atr_based_stop(
                            reference_price=current_price,
                            atr=last_atr,
                            is_long=True
                        )
                        # Only respect support when NOT in profit
                        new_sl = max(new_sl, support * 0.995)
                    
                    # Only move SL up, never down
                    if new_sl > self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = new_sl
                        pct_move = ((self.stop_loss - old_sl) / old_sl * 100)
                        logger.info(
                            f"[DTM] SL trailed up: ${old_sl:,.2f} → ${self.stop_loss:,.2f} "
                            f"(+{pct_move:.2f}%)"
                        )

                # Take Profit calculation
                risk = self.entry_price - self.stop_loss
                tp_atr = self.entry_price + (risk * self.reward_risk_ratio)
                
                # Respect resistance but allow extension if momentum is strong
                if profit_pct > 0.01:  # If already 1% in profit
                    self.take_profit = tp_atr
                else:
                    self.take_profit = min(tp_atr, resistance * 1.002)

            # === SHORT POSITION ===
            else:
                if initial:
                    # ✅ FIXED: Use capped ATR-based stop
                    sl_atr = self._calculate_atr_based_stop(
                        reference_price=self.entry_price,
                        atr=last_atr,
                        is_long=False
                    )
                    
                    sl_resistance = resistance * 1.005
                    
                    # Choose the tighter of the two
                    self.stop_loss = min(sl_atr, sl_resistance)
                    
                    # ✅ Final safety check
                    max_sl_price = self.entry_price * (1 + self.max_stop_loss_pct)
                    if self.stop_loss > max_sl_price:
                        logger.warning(
                            f"[DTM] SL too wide, clamping: "
                            f"${self.stop_loss:,.2f} → ${max_sl_price:,.2f}"
                        )
                        self.stop_loss = max_sl_price
                    
                    actual_sl_pct = (self.stop_loss - self.entry_price) / self.entry_price
                    logger.info(
                        f"[DTM] Initial SL breakdown (SHORT):\n"
                        f"      ATR-based: ${sl_atr:,.2f} (ATR=${last_atr:,.2f})\n"
                        f"      Resistance: ${sl_resistance:,.2f}\n"
                        f"      → Selected: ${self.stop_loss:,.2f} ({actual_sl_pct:.2%} above entry)"
                    )
                else:
                    if self.profit_locked or self.aggressive_trail:
                        # Aggressive trailing: IGNORE resistance when in profit
                        aggressive_atr_distance = min(
                            last_atr * 0.8,
                            self.lowest_price * self.max_stop_loss_pct
                        )
                        new_sl = self.lowest_price + aggressive_atr_distance
                    else:
                        # Conservative trailing: respect resistance
                        new_sl = self._calculate_atr_based_stop(
                            reference_price=current_price,
                            atr=last_atr,
                            is_long=False
                        )
                        new_sl = min(new_sl, resistance * 1.005)
                    
                    # Only move SL down (closer to entry), never up
                    if new_sl < self.stop_loss:
                        old_sl = self.stop_loss
                        self.stop_loss = new_sl
                        pct_move = ((old_sl - self.stop_loss) / old_sl * 100)
                        logger.info(
                            f"[DTM] SL trailed down: ${old_sl:,.2f} → ${self.stop_loss:,.2f} "
                            f"(-{pct_move:.2f}%)"
                        )

                # Take Profit calculation
                risk = self.stop_loss - self.entry_price
                tp_atr = self.entry_price - (risk * self.reward_risk_ratio)
                
                if profit_pct > 0.01:
                    self.take_profit = tp_atr
                else:
                    self.take_profit = max(tp_atr, support * 0.998)

            # Position size (calculated once on initialization)
            if initial:
                trade_risk = abs(self.entry_price - self.stop_loss)
                if trade_risk <= 0:
                    raise ValueError(f"Invalid stop loss for {self.side.upper()} trade")
                self.position_size = (self.account_balance * self.account_risk) / trade_risk

            self.update_count += 1
            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"[DTM] Error updating levels: {e}", exc_info=True)

    def check_exit(self, current_price: float) -> Optional[str]:
        """
        Check if the trade should exit based on dynamic SL or TP.
        Returns: 'stop_loss', 'take_profit', or None
        """
        try:
            if self.side == 'long':
                if current_price <= self.stop_loss:
                    pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                    logger.info(
                        f"[DTM] STOP LOSS HIT: ${current_price:,.2f} <= ${self.stop_loss:,.2f} "
                        f"(P&L: {pnl_pct:+.2f}%)"
                    )
                    return 'stop_loss'
                elif current_price >= self.take_profit:
                    pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                    logger.info(
                        f"[DTM] TAKE PROFIT HIT: ${current_price:,.2f} >= ${self.take_profit:,.2f} "
                        f"(P&L: {pnl_pct:+.2f}%)"
                    )
                    return 'take_profit'
            else:  # short
                if current_price >= self.stop_loss:
                    pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                    logger.info(
                        f"[DTM] STOP LOSS HIT: ${current_price:,.2f} >= ${self.stop_loss:,.2f} "
                        f"(P&L: {pnl_pct:+.2f}%)"
                    )
                    return 'stop_loss'
                elif current_price <= self.take_profit:
                    pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                    logger.info(
                        f"[DTM] TAKE PROFIT HIT: ${current_price:,.2f} <= ${self.take_profit:,.2f} "
                        f"(P&L: {pnl_pct:+.2f}%)"
                    )
                    return 'take_profit'

            return None

        except Exception as e:
            logger.error(f"[DTM] Error checking exit: {e}")
            return None

    def update_with_current_price(self, current_price: float) -> Optional[str]:
        """
        ✅ NEW: Update trailing stop based on current price (not just bar close)
        Call this every 1-5 minutes for active management
        
        This allows the DTM to trail stops in real-time rather than waiting
        for the next bar close, which is critical for volatile assets like crypto.
        
        Args:
            current_price: Current market price
            
        Returns:
            'stop_loss', 'take_profit', or None
        """
        try:
            # Update highest/lowest
            if self.side == 'long':
                old_high = self.highest_price
                self.highest_price = max(self.highest_price, current_price)
                
                # If new high and we're in profit-lock mode, trail immediately
                if self.highest_price > old_high and (self.profit_locked or self.aggressive_trail):
                    last_atr = self.calculate_atr()
                    if last_atr > 0:
                        aggressive_atr_distance = min(
                            last_atr * 0.8,
                            self.highest_price * self.max_stop_loss_pct
                        )
                        new_sl = self.highest_price - aggressive_atr_distance
                        
                        if new_sl > self.stop_loss:
                            old_sl = self.stop_loss
                            self.stop_loss = new_sl
                            logger.info(
                                f"[DTM] 📈 New high ${self.highest_price:,.2f} - "
                                f"SL trailed: ${old_sl:,.2f} → ${self.stop_loss:,.2f}"
                            )
            else:  # short
                old_low = self.lowest_price
                self.lowest_price = min(self.lowest_price, current_price)
                
                if self.lowest_price < old_low and (self.profit_locked or self.aggressive_trail):
                    last_atr = self.calculate_atr()
                    if last_atr > 0:
                        aggressive_atr_distance = min(
                            last_atr * 0.8,
                            self.lowest_price * self.max_stop_loss_pct
                        )
                        new_sl = self.lowest_price + aggressive_atr_distance
                        
                        if new_sl < self.stop_loss:
                            old_sl = self.stop_loss
                            self.stop_loss = new_sl
                            logger.info(
                                f"[DTM] 📉 New low ${self.lowest_price:,.2f} - "
                                f"SL trailed: ${old_sl:,.2f} → ${self.stop_loss:,.2f}"
                            )
            
            # Check exit on current price
            return self.check_exit(current_price)
        
        except Exception as e:
            logger.error(f"[DTM] Error updating with current price: {e}")
            return None

    def on_new_bar(
        self, 
        new_high: float, 
        new_low: float, 
        new_close: float,
        force_update: bool = False
    ):
        """
        Call this method whenever a new bar/candle is formed.
        Updates the internal OHLC arrays and recalculates SL/TP dynamically.
        """
        try:
            # Append new data
            self.high = np.append(self.high, new_high)
            self.low = np.append(self.low, new_low)
            self.close = np.append(self.close, new_close)

            # Keep only recent history (performance optimization)
            max_history = max(100, self.sr_window * 2)
            if len(self.close) > max_history:
                self.high = self.high[-max_history:]
                self.low = self.low[-max_history:]
                self.close = self.close[-max_history:]

            # Update levels
            self.update_levels(initial=False, current_price=new_close)

            # Check for exit
            exit_signal = self.check_exit(new_close)
            if exit_signal:
                return exit_signal

            return None

        except Exception as e:
            logger.error(f"[DTM] Error on new bar: {e}", exc_info=True)
            return None

    def get_status(self) -> dict:
        """Get current status of the trade manager"""
        current_price = self.close[-1]
        
        if self.side == 'long':
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            distance_to_sl = ((current_price - self.stop_loss) / current_price) * 100
            distance_to_tp = ((self.take_profit - current_price) / current_price) * 100
        else:
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
            distance_to_sl = ((self.stop_loss - current_price) / current_price) * 100
            distance_to_tp = ((current_price - self.take_profit) / current_price) * 100

        return {
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'pnl_pct': pnl_pct,
            'distance_to_sl_pct': distance_to_sl,
            'distance_to_tp_pct': distance_to_tp,
            'profit_locked': self.profit_locked,
            'update_count': self.update_count,
            'last_update': self.last_update.isoformat(),
        }

    def __repr__(self):
        status = self.get_status()
        return (
            f"DynamicTradeManager({self.side.upper()}: "
            f"Entry=${status['entry_price']:.2f}, "
            f"Current=${status['current_price']:.2f}, "
            f"SL=${status['stop_loss']:.2f}, "
            f"TP=${status['take_profit']:.2f}, "
            f"P&L={status['pnl_pct']:+.2f}%)"
        )