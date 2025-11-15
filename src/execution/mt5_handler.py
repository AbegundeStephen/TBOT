# src/execution/mt5_handler.py
"""
MT5 Execution Handler with proper SL/TP and position closing
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MT5ExecutionHandler:
    """
     Handles MT5 execution with proper position management
    """

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        
        self.asset_config = config["assets"]["GOLD"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]
        
        self.symbol = self.asset_config["symbol"]
        
        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            raise ValueError(f"Symbol {self.symbol} not found")
        
        logger.info("MT5ExecutionHandler initialized")

    def get_current_price(self, symbol: str = None) -> float:
        """Get current market price"""
        if symbol is None:
            symbol = self.symbol
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0
        
        return tick.ask

    def execute_signal(self, signal: int, symbol: str = None, asset: str = "GOLD") -> bool:
        if symbol is None:
            symbol = self.symbol
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                logger.error(f"{asset}: Failed to get current price")
                return False

            # STEP 1: Check and manage existing position
            existing_position = self.portfolio_manager.get_position(asset)

            if existing_position:
                position_side = existing_position.side if hasattr(existing_position, 'side') else existing_position.get('side')

                # Close long position on SELL signal
                if signal == -1 and position_side == 'long':
                    logger.info(f"[CLOSE] {asset}: Closing long position on SELL signal")
                    success = self._close_mt5_position(existing_position, current_price, asset, "sell_signal")
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} position")
                        return False

                # Close short position on BUY signal
                elif signal == 1 and position_side == 'short':
                    logger.info(f"[CLOSE] {asset}: Closing short position on BUY signal")
                    success = self._close_mt5_position(existing_position, current_price, asset, "buy_signal")
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} position")
                        return False

                # Check SL/TP on HOLD signal
                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(existing_position, current_price)
                    if should_close:
                        logger.info(f"[CLOSE] {asset}: Closing position - {close_reason}")
                        success = self._close_mt5_position(existing_position, current_price, asset, close_reason)
                        if not success:
                            logger.error(f"[FAIL] Failed to close {asset} position")
                            return False

            # Open new long position on BUY signal if no position exists
            if signal == 1 and not existing_position:
                return self._open_mt5_position(signal, current_price, symbol, asset)

            # Do nothing on SELL signal if no position exists
            return True

        except Exception as e:
            logger.error(f"Error executing {asset} signal: {e}", exc_info=True)
            return False

        except Exception as e:
            logger.error(f"Error executing {asset} signal: {e}", exc_info=True)
            return False

    def _check_stop_loss_take_profit(self, position, current_price: float) -> tuple:
        """
        Check if stop-loss or take-profit is hit
        Returns: (should_close: bool, reason: str)
        """
        try:
            # Get position details
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
                if stop_loss and current_price <= stop_loss:
                    return True, f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f})"
                if take_profit and current_price >= take_profit:
                    return True, f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f})"
            else:  # short
                if stop_loss and current_price >= stop_loss:
                    return True, f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f})"
                if take_profit and current_price <= take_profit:
                    return True, f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f})"
            
            return False, None
        
        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}")
            return False, None

    def _close_mt5_position(self, position, current_price: float, asset: str, reason: str) -> bool:
        """Close existing MT5 position"""
        try:
            # Get position details
            if hasattr(position, 'entry_price'):
                entry_price = position.entry_price
                quantity = position.quantity
                position_size_usd = quantity * entry_price
                side = position.side
                symbol = position.symbol if hasattr(position, 'symbol') else self.symbol
            else:
                entry_price = position['entry_price']
                quantity = position.get('quantity', 0)
                position_size_usd = quantity * entry_price
                side = position['side']
                symbol = position.get('symbol', self.symbol)
            
            # Calculate P&L
            if side == 'long':
                pnl = (current_price - entry_price) / entry_price * position_size_usd
            else:
                pnl = (entry_price - current_price) / entry_price * position_size_usd
            
            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0
            
            logger.info(
                f"[CLOSE] {asset} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )
            
            # Get MT5 positions for this symbol
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(f"{asset}: No MT5 position found, closing in portfolio only")
                self.portfolio_manager.close_position(
                    asset=asset,
                    exit_price=current_price,
                    reason=reason
                )
                return True
            
            # Close the MT5 position
            mt5_position = mt5_positions[0]
            order_type = mt5.ORDER_TYPE_SELL if mt5_position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Place closing order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": mt5_position.volume,
                "type": order_type,
                "price": current_price,
                "deviation": 10,
                "magic": 234000,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Close in portfolio manager
                self.portfolio_manager.close_position(
                    asset=asset,
                    exit_price=current_price,
                    reason=reason
                )
                
                logger.info(f"[OK] {asset} position closed on MT5 and portfolio")
                return True
            else:
                logger.error(f"[FAIL] Failed to close MT5 position: {result.comment if result else 'No result'}")
                return False
        
        except Exception as e:
            logger.error(f"Error closing MT5 position: {e}", exc_info=True)
            return False

    def _open_mt5_position(self, signal: int, current_price: float, symbol: str, asset: str) -> bool:
        """Open new MT5 position"""
        try:
            # Calculate position size
            position_size_usd, volume_lots = self.calculate_position_size(current_price, asset)
            
            if volume_lots <= 0:
                logger.error(f"{asset}: Invalid volume calculated")
                return False

            # Determine order type and side
            order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            position_side = 'long' if signal == 1 else 'short'

            # Calculate SL/TP
            risk = self.asset_config.get("risk", {})
            stop_loss_pct = risk.get("stop_loss_pct", 0.03)
            take_profit_pct = risk.get("take_profit_pct", 0.08)
            trailing_stop_pct = risk.get("trailing_stop_pct", 0.02)
            
            if signal == 1:  # BUY/Long
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:  # SELL/Short
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)

            logger.info(
                f"[OPEN] {'BUY' if signal == 1 else 'SELL'} {volume_lots} lots {symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${position_size_usd:,.2f}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")

            # Place MT5 order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_lots,
                "type": order_type,
                "price": current_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 234000,
                "comment": f"Signal_{signal}_{asset}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment if result else 'No result'}")
                return False

            #  Add position to portfolio manager with correct parameters
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=stop_loss,  # Pass actual price
                take_profit=take_profit,  # Pass actual price
                trailing_stop_pct=trailing_stop_pct  # Only percentage parameter
            )

            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset} position")
                # TODO: Close the MT5 position we just opened
                return False

            logger.info(f"[OK] {'BUY' if signal == 1 else 'SELL'} {asset} - Position opened")
            return True
        
        except Exception as e:
            logger.error(f"Error opening MT5 position: {e}", exc_info=True)
            return False

    def calculate_position_size(self, current_price: float, asset: str = "GOLD") -> tuple:
        """Calculate position size in USD and lots"""
        position_size_usd = self.portfolio_manager.calculate_position_size(
            asset=asset,
            current_price=current_price
        )
        
        contract_size = self.symbol_info.trade_contract_size
        volume_lots = position_size_usd / (current_price * contract_size)
        
        # Round to lot step
        volume_step = self.symbol_info.volume_step
        volume_lots = round(volume_lots / volume_step) * volume_step
        
        # Enforce min/max
        volume_lots = max(self.symbol_info.volume_min, volume_lots)
        volume_lots = min(self.symbol_info.volume_max, volume_lots)
        
        actual_position_size = volume_lots * current_price * contract_size
        
        return actual_position_size, volume_lots

    def check_and_update_positions(self, asset: str = "GOLD"):
        """
        NEW: Actively check and update all positions (for HOLD signals)
        Should be called on every cycle
        """
        try:
            position = self.portfolio_manager.get_position(asset)
            
            if not position:
                return
            
            current_price = self.get_current_price()
            
            if current_price == 0:
                logger.warning(f"Could not get price for {asset}")
                return
            
            # Check if SL/TP hit
            should_close, reason = self._check_stop_loss_take_profit(position, current_price)
            
            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset}: {reason}")
                self._close_mt5_position(position, current_price, asset, reason)
        
        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)