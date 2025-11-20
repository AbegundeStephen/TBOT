# src/execution/mt5_handler.py
"""
MT5 Execution Handler with proper SL/TP and position closing
FIXED: Duplicate position prevention and logic consistency
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional, Tuple
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
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading", {})

        self.symbol = self.asset_config["symbol"]

        # Try to fetch symbol_info but don't raise to avoid hard crash on missing symbol
        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            logger.warning(
                f"Symbol {self.symbol} not found in MT5. MT5 operations may fail until symbol is available."
            )
        else:
            logger.debug(f"Symbol {self.symbol} info loaded.")

        logger.info("MT5ExecutionHandler initialized")

        # Respect configuration flags: only auto-sync if both flags are explicitly True
        auto_sync_enabled = self.auto_sync_enabled = bool(
            self.trading_config.get("auto_sync_on_startup", False)
        )
        import_enabled = self.import_enabled = bool(
            self.config.get("portfolio", {}).get("import_existing_positions", False)
        )

        if auto_sync_enabled and import_enabled:
            logger.info("[INIT] Auto-syncing positions with MT5 (per config)...")
            self.sync_positions_with_mt5("GOLD")
        else:
            logger.info(
                "[INIT] MT5 auto-sync disabled (startup). "
                f"auto_sync_on_startup={auto_sync_enabled}, import_existing_positions={import_enabled}"
            )

    def get_current_price(self, symbol: str = None) -> float:
        """Get current market price"""
        if symbol is None:
            symbol = self.symbol

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0

        # FIXED: Return appropriate price based on operation
        # For buying: use ask, for selling: use bid
        return (tick.ask + tick.bid) / 2  # Use mid price for consistency

    def execute_signal(
        self, signal: int, symbol: str = None, asset: str = "GOLD"
    ) -> bool:
        """
        FIXED: Execute trading signal with proper duplicate position handling

        Args:
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            symbol: Trading symbol
            asset: Asset name (for portfolio tracking)

        Returns:
            True if execution successful, False otherwise
        """
        if symbol is None:
            symbol = self.symbol

        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                logger.error(f"{asset}: Failed to get current price")
                return False

            # STEP 1: Get existing position (if any)
            existing_position = self.portfolio_manager.get_position(asset)

            # STEP 2: Handle existing positions
            if existing_position:
                position_side = (
                    existing_position.side
                    if hasattr(existing_position, "side")
                    else existing_position.get("side")
                )

                logger.debug(
                    f"{asset}: Existing {position_side.upper()} position found, signal={signal}"
                )

                # === CLOSE POSITION SCENARIOS ===

                # Close long position on SELL signal
                if signal == -1 and position_side == "long":
                    logger.info(
                        f"[SIGNAL] {asset}: SELL signal received - Closing LONG position"
                    )
                    success = self._close_mt5_position(
                        existing_position, current_price, asset, "sell_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} LONG position")
                        return False
                    # Position closed, don't open new short position (only trade longs for GOLD typically)
                    return True

                # Close short position on BUY signal (if you trade shorts)
                elif signal == 1 and position_side == "short":
                    logger.info(
                        f"[SIGNAL] {asset}: BUY signal received - Closing SHORT position"
                    )
                    success = self._close_mt5_position(
                        existing_position, current_price, asset, "buy_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} SHORT position")
                        return False
                    # After closing short, continue to check if we should open long below

                # HOLD signal - check SL/TP
                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        existing_position, current_price
                    )
                    if should_close:
                        logger.info(f"[AUTO-CLOSE] {asset}: {close_reason}")
                        success = self._close_mt5_position(
                            existing_position, current_price, asset, close_reason
                        )
                        if not success:
                            logger.error(
                                f"[FAIL] Failed to close {asset} position on {close_reason}"
                            )
                            return False
                    else:
                        # Position still valid, just holding
                        logger.debug(f"{asset}: Holding position, no action needed")
                    return True

                # CRITICAL FIX: Prevent duplicate positions
                elif signal == 1 and position_side == "long":
                    logger.info(
                        f"[SKIP] {asset}: BUY signal ignored - Already have LONG position"
                    )
                    return True  # Not an error, just skip

                elif signal == -1 and position_side == "short":
                    logger.info(
                        f"[SKIP] {asset}: SELL signal ignored - Already have SHORT position"
                    )
                    return True  # Not an error, just skip

            # STEP 3: Open new position (only if no position exists)
            if signal == 1:  # BUY signal
                # Double-check no position exists (belt and suspenders)
                if self.portfolio_manager.has_position(asset, side="long"):
                    logger.warning(
                        f"[SKIP] {asset}: BUY signal - Position already exists (double-check)"
                    )
                    return True

                logger.info(f"[SIGNAL] {asset}: BUY signal - Opening LONG position")
                return self._open_mt5_position(signal, current_price, symbol, asset)

            elif signal == -1:  # SELL signal
                # For GOLD, typically we don't open short positions, only close longs
                # If you want to trade shorts, uncomment below:
                # if self.portfolio_manager.has_position(asset, side='short'):
                #     logger.warning(f"[SKIP] {asset}: SELL signal - Position already exists")
                #     return True
                # return self._open_mt5_position(signal, current_price, symbol, asset)

                logger.debug(f"{asset}: SELL signal - No position to close, no action")
                return True

            # HOLD signal with no position - do nothing
            return True

        except Exception as e:
            logger.error(f"Error executing {asset} signal: {e}", exc_info=True)
            return False

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if stop-loss or take-profit is hit

        Returns:
            Tuple of (should_close: bool, reason: str)
        """
        try:
            # Get position details
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                stop_loss = position.stop_loss
                take_profit = position.take_profit
                side = position.side
            else:
                entry_price = position.get("entry_price")
                stop_loss = position.get("stop_loss")
                take_profit = position.get("take_profit")
                side = position.get("side")

            # FIXED: More precise price comparisons with tolerance
            price_tolerance = 0.01  # $0.01 tolerance for floating point comparisons

            if side == "long":
                if stop_loss and current_price <= (stop_loss + price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            else:  # short
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}", exc_info=True)
            return False, ""

    def _close_mt5_position(
        self, position, current_price: float, asset: str, reason: str
    ) -> bool:
        """Close existing MT5 position"""
        try:
            # Get position details
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                quantity = position.quantity
                side = position.side
                symbol = position.symbol if hasattr(position, "symbol") else self.symbol
            else:
                entry_price = position["entry_price"]
                quantity = position.get("quantity", 0)
                side = position["side"]
                symbol = position.get("symbol", self.symbol)

            # Calculate actual position size
            contract_size = self.symbol_info.trade_contract_size
            position_size_usd = quantity * entry_price

            # Calculate P&L
            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset} {side.upper()} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )

            # Get MT5 positions for this symbol
            mt5_positions = mt5.positions_get(symbol=symbol)

            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(
                    f"{asset}: No MT5 position found, closing in portfolio only"
                )
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=reason
                )
                return True

            # FIXED: Match the correct position by side
            mt5_position = None
            for pos in mt5_positions:
                pos_type = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                if pos_type == side:
                    mt5_position = pos
                    break

            if mt5_position is None:
                logger.warning(
                    f"{asset}: MT5 {side} position not found, closing in portfolio only"
                )
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=reason
                )
                return True

            # Close the MT5 position
            order_type = (
                mt5.ORDER_TYPE_SELL
                if mt5_position.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            # FIXED: Use appropriate price (bid for sell, ask for buy)
            tick = mt5.symbol_info_tick(symbol)
            close_price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            # Place closing order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": mt5_position.volume,
                "type": order_type,
                "position": mt5_position.ticket,  # FIXED: Specify position ticket
                "price": close_price,
                "deviation": 20,  # Increased slippage tolerance
                "magic": 234000,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Close in portfolio manager
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=close_price, reason=reason
                )

                logger.info(
                    f"[OK] {asset} {side.upper()} position closed on MT5 and portfolio"
                )
                return True
            else:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(
                    f"[FAIL] Failed to close MT5 position: {error_msg} (code: {error_code})"
                )

                # CRITICAL: Still close in portfolio to avoid desync
                logger.warning(f"Closing {asset} in portfolio despite MT5 failure")
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=f"{reason}_mt5_failed"
                )
                return False

        except Exception as e:
            logger.error(f"Error closing MT5 position: {e}", exc_info=True)
            return False

    def _open_mt5_position(
        self, signal: int, current_price: float, symbol: str, asset: str
    ) -> bool:
        """Open new MT5 position"""
        try:
            # CRITICAL CHECK: Verify no position exists before opening
            can_open, reason = self.portfolio_manager.can_open_position(
                asset, "long" if signal == 1 else "short"
            )
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset} position: {reason}")
                return True  # Not an error, just skip

            # Calculate position size
            position_size_usd, volume_lots = self.calculate_position_size(
                current_price, asset
            )

            if volume_lots <= 0:
                logger.error(f"{asset}: Invalid volume calculated: {volume_lots}")
                return False

            # Determine order type and side
            order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            position_side = "long" if signal == 1 else "short"

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
                f"[OPEN] {'BUY' if signal == 1 else 'SELL'} {volume_lots:.2f} lots {symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${position_size_usd:,.2f}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")

            # FIXED: Use appropriate execution price
            tick = mt5.symbol_info_tick(symbol)
            execution_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

            # Place MT5 order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_lots,
                "type": order_type,
                "price": execution_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Signal_{signal}_{asset}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(f"[FAIL] Order failed: {error_msg} (code: {error_code})")
                return False

            # Add position to portfolio manager
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,
                entry_price=execution_price,  # Use actual execution price
                position_size_usd=position_size_usd,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
            )

            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset} position")
                # CRITICAL: Close the MT5 position we just opened
                logger.warning("Attempting to close unwanted MT5 position...")
                self._emergency_close_mt5_position(symbol, volume_lots, order_type)
                return False

            logger.info(
                f"[OK] {'BUY' if signal == 1 else 'SELL'} {asset} - Position opened successfully"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening MT5 position: {e}", exc_info=True)
            return False

    def _emergency_close_mt5_position(
        self, symbol: str, volume: float, original_order_type: int
    ):
        """Emergency close of an unwanted MT5 position"""
        try:
            close_order_type = (
                mt5.ORDER_TYPE_SELL
                if original_order_type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )
            tick = mt5.symbol_info_tick(symbol)
            close_price = (
                tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_order_type,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Emergency_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("[OK] Emergency close successful")
            else:
                logger.error(
                    f"[FAIL] Emergency close failed: {result.comment if result else 'No result'}"
                )
        except Exception as e:
            logger.error(f"Emergency close error: {e}", exc_info=True)

    def calculate_position_size(
        self, current_price: float, asset: str = "GOLD"
    ) -> Tuple[float, float]:
        """
        Calculate position size in USD and lots

        Returns:
            Tuple of (position_size_usd, volume_lots)
        """
        position_size_usd = self.portfolio_manager.calculate_position_size(
            asset=asset, current_price=current_price
        )

        contract_size = self.symbol_info.trade_contract_size
        volume_lots = position_size_usd / (current_price * contract_size)

        # Round to lot step
        volume_step = self.symbol_info.volume_step
        volume_lots = round(volume_lots / volume_step) * volume_step

        # Enforce min/max
        volume_lots = max(self.symbol_info.volume_min, volume_lots)
        volume_lots = min(self.symbol_info.volume_max, volume_lots)

        # Recalculate actual position size after rounding
        actual_position_size = volume_lots * current_price * contract_size

        logger.debug(
            f"{asset}: Calculated {volume_lots:.2f} lots = ${actual_position_size:,.2f}"
        )

        return actual_position_size, volume_lots

    def check_and_update_positions(self, asset: str = "GOLD"):
        """
        Actively check and update all positions (for HOLD signals)
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
            should_close, reason = self._check_stop_loss_take_profit(
                position, current_price
            )

            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset}: {reason}")
                self._close_mt5_position(position, current_price, asset, reason)

        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)

    def sync_positions_with_mt5(self, asset: str = "GOLD", symbol: str = None) -> bool:
        """
        Sync portfolio manager with actual MT5 positions.
        Auto-import is DISABLED by default—this function will detect MT5 positions
        but will NOT import them as bot-managed trades unless you explicitly enable
        import_existing_positions in config.
        """

        if symbol is None:
            symbol = self.symbol

        try:
            logger.info(f"[SYNC] Starting position sync for {asset} (MT5)...")

            mt5_positions = mt5.positions_get(symbol=symbol)
            portfolio_position = self.portfolio_manager.get_position(asset)

            # If no MT5 symbol info (symbol not available), bail out safely
            if self.symbol_info is None:
                logger.warning(
                    f"[SYNC] Symbol info for {symbol} unavailable; skipping MT5 sync."
                )
                return True

            # CASE A: MT5 has position(s) and portfolio has none -> DETECT but DO NOT IMPORT
            # Inside sync_positions_with_mt5, replace CASE A with:
            if mt5_positions and len(mt5_positions) > 0 and not portfolio_position:
                if self.import_enabled:  # Only import if enabled in config
                    logger.info(
                        f"[SYNC] Importing {len(mt5_positions)} MT5 position(s) for {asset}..."
                    )
                    for pos in mt5_positions:
                        pos_type = (
                            "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        )
                        logger.info(
                            f"  → Importing MT5 {pos_type}: entry={pos.price_open:.5f}, current={pos.price_current:.5f}, "
                            f"volume={pos.volume:.2f}, profit={pos.profit:.2f}"
                        )
                        # Add position to portfolio manager
                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=pos.volume
                            * pos.price_open
                            * self.symbol_info.trade_contract_size,
                            stop_loss=None,  # Set as needed
                            take_profit=None,  # Set as needed
                            trailing_stop_pct=None,  # Set as needed
                        )
                        if success:
                            logger.info(
                                f"[SYNC] ✓ Imported {asset} {pos_type} position"
                            )
                        else:
                            logger.error(
                                f"[SYNC] ✗ Failed to import {asset} {pos_type} position"
                            )
                else:
                    logger.info(
                        f"[SYNC] Detected MT5 positions for {asset} but auto-import is disabled.\n"
                        f"  MT5 Positions Count: {len(mt5_positions)}\n"
                        f"  To enable import on startup, set trading.auto_sync_on_startup = true AND "
                        f"portfolio.import_existing_positions = true in config."
                    )
                    for pos in mt5_positions:
                        pos_type = (
                            "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        )
                        logger.info(
                            f"  → MT5 {pos_type}: entry={pos.price_open:.5f}, current={pos.price_current:.5f}, "
                            f"volume={pos.volume:.2f}, profit={pos.profit:.2f}"
                        )
                return True

            # CASE B: Portfolio has a position, MT5 has none -> remove from portfolio
            if portfolio_position and (not mt5_positions or len(mt5_positions) == 0):
                logger.warning(
                    f"[SYNC] Portfolio shows {asset} position but MT5 has no matching positions.\n"
                    f"  → Removing from portfolio (likely closed manually in MT5)."
                )
                current_price = self.get_current_price(self.symbol)
                trade_result = self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason="sync_missing_mt5"
                )
                if trade_result:
                    logger.info(f"[SYNC] ✓ Removed {asset} position from portfolio")
                    return True
                else:
                    logger.error(
                        f"[SYNC] ✗ Failed to remove {asset} position from portfolio"
                    )
                    return False

            # CASE C: Both MT5 and Portfolio have positions -> verify match
            if mt5_positions and len(mt5_positions) > 0 and portfolio_position:
                # Use the first MT5 position as canonical for symbol
                mt5_pos = mt5_positions[0]
                mt5_side = "long" if mt5_pos.type == mt5.POSITION_TYPE_BUY else "short"
                portfolio_side = getattr(
                    portfolio_position,
                    "side",
                    (
                        portfolio_position.get("side")
                        if isinstance(portfolio_position, dict)
                        else None
                    ),
                )

                if mt5_side != portfolio_side:
                    logger.error(
                        f"[SYNC] MISMATCH: MT5 has {mt5_side.upper()} but portfolio has {portfolio_side}.\n"
                        f"  → Closing portfolio position to avoid mismatch (do not auto-import)."
                    )
                    current_price = self.get_current_price(self.symbol)
                    self.portfolio_manager.close_position(
                        asset, current_price, "sync_mismatch_mt5"
                    )
                    return True
                else:
                    logger.info(
                        f"[SYNC] ✓ {asset} positions already in sync ({mt5_side.upper()})"
                    )
                    return True

            # CASE D: Neither has positions -> OK
            logger.info(f"[SYNC] ✓ No positions for {asset} in MT5 or portfolio")
            return True

        except Exception as e:
            logger.error(f"[SYNC] Error syncing MT5 positions: {e}", exc_info=True)
            return False
