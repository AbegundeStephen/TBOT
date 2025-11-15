# src/execution/binance_handler.py
"""
Binance Execution Handler with proper parameter passing
"""

import logging
from binance.client import Client
from binance.enums import *
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceExecutionHandler:
    """
     Handles trade execution with proper position management
    """

    def __init__(self, config: Dict, client: Client, portfolio_manager):
        self.config = config
        self.client = client
        self.portfolio_manager = portfolio_manager
        
        self.asset_config = config["assets"]["BTC"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]
        
        self.symbol = self.asset_config["symbol"]
        self.mode = self.trading_config.get("mode", "paper")
        
        logger.info(f"BinanceExecutionHandler initialized - Mode: {self.mode.upper()}")

    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """Get current market price"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def execute_signal(self, signal: int, current_price: float, asset_name: str = "BTC") -> bool:
        try:
            existing_position = self.portfolio_manager.get_position(asset_name)

            if existing_position:
                position_side = existing_position.side

                # Close long position on SELL signal
                if signal == -1 and position_side == 'long':
                    logger.info(f"[CLOSE] {asset_name}: Closing long position on SELL signal")
                    success = self._close_position(existing_position, current_price, asset_name, "sell_signal")
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset_name} position")
                        return False

                # Close short position on BUY signal
                elif signal == 1 and position_side == 'short':
                    logger.info(f"[CLOSE] {asset_name}: Closing short position on BUY signal")
                    success = self._close_position(existing_position, current_price, asset_name, "buy_signal")
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset_name} position")
                        return False

                # Check SL/TP on HOLD signal
                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(existing_position, current_price)
                    if should_close:
                        logger.info(f"[CLOSE] {asset_name}: Closing position - {close_reason}")
                        success = self._close_position(existing_position, current_price, asset_name, close_reason)
                        if not success:
                            logger.error(f"[FAIL] Failed to close {asset_name} position")
                            return False

            # Open new long position on BUY signal if no position exists
            if signal == 1 and not existing_position:
                return self._open_position(signal, current_price, asset_name)

            # Do nothing on SELL signal if no position exists
            return True

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False

    def _check_stop_loss_take_profit(self, position, current_price: float) -> tuple:
        """
        Check if stop-loss or take-profit is hit
        Returns: (should_close: bool, reason: str)
        """
        try:
            # Get position details - handle both dict and object
            if hasattr(position, 'entry_price'):
                entry_price = position.entry_price
                stop_loss = position.stop_loss
                take_profit = position.take_profit
                side = position.side
            else:
                entry_price = position.get('entry_price')
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                side = position.get('side')
            
            if side == 'long':
                # Long position: close if price drops below SL or rises above TP
                if stop_loss and current_price <= stop_loss:
                    return True, f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f})"
                if take_profit and current_price >= take_profit:
                    return True, f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f})"
            else:  # short
                # Short position: close if price rises above SL or drops below TP
                if stop_loss and current_price >= stop_loss:
                    return True, f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f})"
                if take_profit and current_price <= take_profit:
                    return True, f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f})"
            
            return False, None
        
        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}")
            return False, None

    def _close_position(self, position, current_price: float, asset_name: str, reason: str) -> bool:
        """Close existing position"""
        try:
            # Get position details - FIXED attribute access
            if hasattr(position, 'entry_price'):
                entry_price = position.entry_price
                #  Use correct attribute name
                position_size_usd = position.quantity * entry_price
                side = position.side
            else:
                entry_price = position['entry_price']
                quantity = position.get('quantity', 0)
                position_size_usd = quantity * entry_price
                side = position['side']
            
            # Calculate P&L
            if side == 'long':
                pnl = (current_price - entry_price) / entry_price * position_size_usd
            else:  # short
                pnl = (entry_price - current_price) / entry_price * position_size_usd
            
            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0
            
            logger.info(
                f"[CLOSE] {asset_name} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )
            
            # Close position in portfolio manager
            self.portfolio_manager.close_position(
                asset=asset_name,
                exit_price=current_price,
                reason=reason
            )
            
            logger.info(f"[OK] {asset_name} position closed")
            return True
        
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def _open_position(self, signal: int, current_price: float, asset_name: str) -> bool:
        """Open new position"""
        try:
            # Calculate position size
            position_size_usd = self.portfolio_manager.calculate_position_size(
                asset_name,
                current_price
            )
            
            if position_size_usd == 0:
                logger.warning(f"{asset_name}: Position size is 0, skipping trade")
                return False
            
            quantity = position_size_usd / current_price
            side = "long" if signal == 1 else "short"
            order_side = "BUY" if signal == 1 else "SELL"
            
            # Get risk parameters
            risk = self.asset_config.get("risk", {})
            stop_loss_pct = risk.get("stop_loss_pct", 0.05)
            take_profit_pct = risk.get("take_profit_pct", 0.10)
            trailing_stop_pct = risk.get("trailing_stop_pct", 0.03)
            
            # Calculate SL/TP prices
            if signal == 1:  # Long
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:  # Short
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            
            logger.info(
                f"[OPEN] {order_side} {quantity:.5f} {self.symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${position_size_usd:,.2f}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")
            
            #  Record position with correct parameters (removed stop_loss_pct and take_profit_pct)
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=stop_loss,  # Pass actual price
                take_profit=take_profit,  # Pass actual price
                trailing_stop_pct=trailing_stop_pct  # Only percentage parameter
            )
            
            if success:
                logger.info(f"[OK] {order_side} {asset_name} - Position opened")
                return True
            else:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset_name} position")
                return False
        
        except Exception as e:
            logger.error(f"Error opening position: {e}", exc_info=True)
            return False

    def check_and_update_positions(self, asset_name: str = "BTC"):
        """
        NEW: Actively check and update all positions (for HOLD signals)
        Should be called on every cycle
        """
        try:
            position = self.portfolio_manager.get_position(asset_name)
            
            if not position:
                return
            
            current_price = self.get_current_price()
            
            if current_price is None:
                logger.warning(f"Could not get price for {asset_name}")
                return
            
            # Check if SL/TP hit
            should_close, reason = self._check_stop_loss_take_profit(position, current_price)
            
            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset_name}: {reason}")
                self._close_position(position, current_price, asset_name, reason)
        
        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)