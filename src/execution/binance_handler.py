# src/execution/binance_handler.py
"""
Binance Execution Handler with proper parameter passing
FIXED: Duplicate position prevention and consistent logic
"""

import logging
from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
    TIME_IN_FORCE_GTC,
)
from typing import Dict, Optional, Tuple
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

        # Auto-sync on initialization if enabled and in live mode
        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")

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

    def execute_signal(
        self, signal: int, current_price: float, asset_name: str = "BTC"
    ) -> bool:
        """
        FIXED: Execute trading signal with proper duplicate position handling

        Args:
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            current_price: Current market price
            asset_name: Asset name (for portfolio tracking)

        Returns:
            True if execution successful, False otherwise
        """
        try:
            # Validate price
            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid current price: {current_price}")
                return False

            # STEP 1: Get existing position (if any)
            existing_position = self.portfolio_manager.get_position(asset_name)

            # STEP 2: Handle existing positions
            if existing_position:
                position_side = (
                    existing_position.side
                    if hasattr(existing_position, "side")
                    else existing_position.get("side")
                )

                logger.debug(
                    f"{asset_name}: Existing {position_side.upper()} position found, signal={signal}"
                )

                # === CLOSE POSITION SCENARIOS ===

                # Close long position on SELL signal
                if signal == -1 and position_side == "long":
                    logger.info(
                        f"[SIGNAL] {asset_name}: SELL signal received - Closing LONG position"
                    )
                    success = self._close_position(
                        existing_position, current_price, asset_name, "sell_signal"
                    )
                    if not success:
                        logger.error(
                            f"[FAIL] Failed to close {asset_name} LONG position"
                        )
                        return False
                    # Position closed, don't open new short position
                    return True

                # Close short position on BUY signal
                elif signal == 1 and position_side == "short":
                    logger.info(
                        f"[SIGNAL] {asset_name}: BUY signal received - Closing SHORT position"
                    )
                    success = self._close_position(
                        existing_position, current_price, asset_name, "buy_signal"
                    )
                    if not success:
                        logger.error(
                            f"[FAIL] Failed to close {asset_name} SHORT position"
                        )
                        return False
                    # After closing short, continue to check if we should open long below

                # HOLD signal - check SL/TP
                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        existing_position, current_price
                    )
                    if should_close:
                        logger.info(f"[AUTO-CLOSE] {asset_name}: {close_reason}")
                        success = self._close_position(
                            existing_position, current_price, asset_name, close_reason
                        )
                        if not success:
                            logger.error(
                                f"[FAIL] Failed to close {asset_name} position on {close_reason}"
                            )
                            return False
                    else:
                        # Position still valid, just holding
                        logger.debug(
                            f"{asset_name}: Holding position, no action needed"
                        )
                    return True

                # CRITICAL FIX: Prevent duplicate positions
                elif signal == 1 and position_side == "long":
                    logger.info(
                        f"[SKIP] {asset_name}: BUY signal ignored - Already have LONG position"
                    )
                    return True  # Not an error, just skip

                elif signal == -1 and position_side == "short":
                    logger.info(
                        f"[SKIP] {asset_name}: SELL signal ignored - Already have SHORT position"
                    )
                    return True  # Not an error, just skip

            # STEP 3: Open new position (only if no position exists)
            if signal == 1:  # BUY signal
                # Double-check no position exists (belt and suspenders)
                if self.portfolio_manager.has_position(asset_name, side="long"):
                    logger.warning(
                        f"[SKIP] {asset_name}: BUY signal - Position already exists (double-check)"
                    )
                    return True

                logger.info(
                    f"[SIGNAL] {asset_name}: BUY signal - Opening LONG position"
                )
                return self._open_position(signal, current_price, asset_name)

            elif signal == -1:  # SELL signal
                # For BTC, typically we don't open short positions, only close longs
                # If you want to trade shorts, uncomment below:
                # if self.portfolio_manager.has_position(asset_name, side='short'):
                #     logger.warning(f"[SKIP] {asset_name}: SELL signal - Position already exists")
                #     return True
                # return self._open_position(signal, current_price, asset_name)

                logger.debug(
                    f"{asset_name}: SELL signal - No position to close, no action"
                )
                return True

            # HOLD signal with no position - do nothing
            return True

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
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
            # Get position details - handle both dict and object
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
            price_tolerance = 0.50  # $0.50 tolerance for BTC (adjust based on asset)

            if side == "long":
                # Long position: close if price drops below SL or rises above TP
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
                # Short position: close if price rises above SL or drops below TP
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

    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """Close existing position"""
        try:
            # Get position details - FIXED attribute access
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                quantity = position.quantity
                side = position.side
            else:
                entry_price = position["entry_price"]
                quantity = position.get("quantity", 0)
                side = position["side"]

            # Calculate position size and P&L
            position_size_usd = quantity * entry_price

            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:  # short
                pnl = (entry_price - current_price) * quantity

            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset_name} {side.upper()} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )

            # FIXED: In paper mode, we don't need to place actual orders
            # In live mode, you would place a closing order here
            if self.mode.lower() != "paper":
                # TODO: Implement actual Binance order placement for live mode
                logger.warning(
                    f"[LIVE MODE] Actual Binance order placement not implemented yet"
                )
                # Example:
                # order = self.client.create_order(
                #     symbol=self.symbol,
                #     side=SIDE_SELL if side == 'long' else SIDE_BUY,
                #     type=ORDER_TYPE_MARKET,
                #     quantity=quantity
                # )

            # Close position in portfolio manager
            trade_result = self.portfolio_manager.close_position(
                asset=asset_name, exit_price=current_price, reason=reason
            )

            if trade_result:
                logger.info(f"[OK] {asset_name} {side.upper()} position closed")
                return True
            else:
                logger.error(
                    f"[FAIL] Portfolio manager failed to close {asset_name} position"
                )
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def _open_position(
        self, signal: int, current_price: float, asset_name: str
    ) -> bool:
        """Open new position"""
        try:
            # CRITICAL CHECK: Verify no position exists before opening
            side = "long" if signal == 1 else "short"
            can_open, reason = self.portfolio_manager.can_open_position(
                asset_name, side
            )
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset_name} position: {reason}")
                return True  # Not an error, just skip

            # Calculate position size
            position_size_usd = self.portfolio_manager.calculate_position_size(
                asset_name, current_price
            )

            if position_size_usd <= 0:
                logger.warning(
                    f"{asset_name}: Invalid position size: ${position_size_usd:.2f}, skipping trade"
                )
                return False

            quantity = position_size_usd / current_price

            # FIXED: Ensure minimum quantity for Binance
            # BTC typically has minimum 0.00001 BTC on Binance
            min_quantity = 0.00001
            if quantity < min_quantity:
                logger.warning(
                    f"{asset_name}: Quantity {quantity:.8f} below minimum {min_quantity}, skipping"
                )
                return False

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
                f"[OPEN] {order_side} {quantity:.8f} {self.symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${position_size_usd:,.2f}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")

            # FIXED: In paper mode, we don't need to place actual orders
            # In live mode, you would place an opening order here
            if self.mode.lower() != "paper":
                # TODO: Implement actual Binance order placement for live mode
                logger.warning(
                    f"[LIVE MODE] Actual Binance order placement not implemented yet"
                )
                # Example:
                # try:
                #     order = self.client.create_order(
                #         symbol=self.symbol,
                #         side=SIDE_BUY if signal == 1 else SIDE_SELL,
                #         type=ORDER_TYPE_MARKET,
                #         quantity=quantity
                #     )
                #     actual_price = float(order['fills'][0]['price'])
                #     actual_qty = float(order['executedQty'])
                # except Exception as e:
                #     logger.error(f"Binance order failed: {e}")
                #     return False

            # Record position with correct parameters
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=stop_loss,  # Pass actual price
                take_profit=take_profit,  # Pass actual price
                trailing_stop_pct=trailing_stop_pct,  # Only percentage parameter
            )

            if success:
                logger.info(
                    f"[OK] {order_side} {asset_name} - Position opened successfully"
                )
                return True
            else:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset_name} position")

                # CRITICAL: If we placed a live order but portfolio rejected it,
                # we need to close the order immediately
                if self.mode.lower() != "paper":
                    logger.warning(f"Attempting to close unwanted Binance position...")
                    # TODO: Close the position on Binance
                    # self._emergency_close_binance_position(quantity, order_side)

                return False

        except Exception as e:
            logger.error(f"Error opening position: {e}", exc_info=True)
            return False

    def _emergency_close_binance_position(self, quantity: float, original_side: str):
        """
        BONUS: Emergency close of an unwanted Binance position
        """
        try:
            close_side = SIDE_SELL if original_side == "BUY" else SIDE_BUY

            logger.warning(
                f"[EMERGENCY] Closing {quantity:.8f} {self.symbol} with {close_side}"
            )

            order = self.client.create_order(
                symbol=self.symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )

            if order:
                logger.info("[OK] Emergency close successful")
            else:
                logger.error("[FAIL] Emergency close failed")

        except Exception as e:
            logger.error(f"Emergency close error: {e}", exc_info=True)

    def check_and_update_positions(self, asset_name: str = "BTC"):
        """
        Actively check and update all positions (for HOLD signals)
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
            should_close, reason = self._check_stop_loss_take_profit(
                position, current_price
            )

            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset_name}: {reason}")
                self._close_position(position, current_price, asset_name, reason)

        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)

    def get_account_balance(self) -> Dict[str, float]:
        """
        BONUS: Get account balance from Binance
        Useful for reconciliation and monitoring
        """
        try:
            account = self.client.get_account()
            balances = {}

            for balance in account["balances"]:
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked

                if total > 0:
                    balances[asset] = {"free": free, "locked": locked, "total": total}

            return balances

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}

    def sync_positions_with_binance(self, asset_name: str = "BTC", symbol: str = None) -> bool:
        """
        Sync portfolio manager with actual Binance Spot holdings.
        This function only detects long positions (BTC balance) but does NOT import them.
        The bot will only open a new position on a BUY signal.
        """
        if symbol is None:
            symbol = self.symbol
        try:
            logger.info(f"[SYNC] Starting position sync for {asset_name}...")
            # Fetch balances
            account = self.client.get_account()
            portfolio_position = self.portfolio_manager.get_position(asset_name)
            # Extract BTC + USDT
            btc_balance = 0.0
            usdt_balance = 0.0
            for balance in account['balances']:
                if balance['asset'] == 'BTC':
                    btc_balance = float(balance['free']) + float(balance['locked'])
                elif balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free']) + float(balance['locked'])
            # Current price
            current_price = self.get_current_price(symbol)
            if current_price is None:
                logger.error(f"[SYNC] Failed to get current price for {symbol}")
                return False
            MIN_BTC_BALANCE = 0.0001  # ignore dust
            # --------------------------------------------------------
            # CASE 1: Binance has BTC, portfolio has none → DETECT but DO NOT IMPORT
            # --------------------------------------------------------
            if btc_balance > MIN_BTC_BALANCE and not portfolio_position:
                logger.info(
                    f"[SYNC] BTC balance {btc_balance:.8f} BTC detected on Binance.\n"
                    f"  → This is NOT a bot-managed position. The bot will only open a new position on a BUY signal."
                )
                return True
            # --------------------------------------------------------
            # CASE 2: Portfolio has a position, Binance has no BTC → REMOVE position
            # --------------------------------------------------------
            if portfolio_position and btc_balance <= MIN_BTC_BALANCE:
                logger.warning(
                    f"[SYNC] Portfolio shows {asset_name} position but Binance balance is {btc_balance:.8f} BTC\n"
                    f" → Removing position (likely sold manually)."
                )
                trade_result = self.portfolio_manager.close_position(
                    asset=asset_name,
                    exit_price=current_price,
                    reason="sync_missing_binance"
                )
                if trade_result:
                    logger.info(f"[SYNC] ✓ Removed {asset_name} position from portfolio")
                    return True
                else:
                    logger.error(f"[SYNC] ✗ Failed to remove {asset_name} position")
                    return False
            # --------------------------------------------------------
            # CASE 3: Both have positions → verify quantity matches
            # --------------------------------------------------------
            if btc_balance > MIN_BTC_BALANCE and portfolio_position:
                portfolio_qty = getattr(portfolio_position, "quantity", 0)
                # Tolerance check
                qty_diff = abs(btc_balance - portfolio_qty)
                qty_diff_pct = (qty_diff / btc_balance * 100) if btc_balance > 0 else 0
                if qty_diff_pct > 0.1:
                    logger.warning(
                        f"[SYNC] QUANTITY MISMATCH:\n"
                        f"  Binance: {btc_balance:.8f} BTC\n"
                        f"  Portfolio: {portfolio_qty:.8f} BTC\n"
                        f"  Difference: {qty_diff:.8f} BTC ({qty_diff_pct:.2f}%)\n"
                        f"  → Closing position and clearing mismatch"
                    )
                    # Close incorrect portfolio position
                    self.portfolio_manager.close_position(
                        asset_name,
                        current_price,
                        reason="sync_quantity_mismatch"
                    )
                    return True
                else:
                    position_value = btc_balance * current_price
                    logger.info(
                        f"[SYNC] ✓ {asset_name} position already in sync\n"
                        f"  Balance: {btc_balance:.8f} BTC\n"
                        f"  Value: ${position_value:,.2f}"
                    )
                    return True
            # --------------------------------------------------------
            # CASE 4: Neither Binance nor Portfolio has positions → OK
            # --------------------------------------------------------
            logger.info(f"[SYNC] ✓ No {asset_name} positions detected on Binance or portfolio")
            return True
        except Exception as e:
            logger.error(f"[SYNC] Error syncing positions: {e}", exc_info=True)
            return False
