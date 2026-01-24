"""
Binance Futures API Integration - COMPLETE VERSION
✅ FIXED: Take Profit Precision + All Integration Functions
✨ ENHANCED: Hedge Mode Integration for Asymmetric System (Simultaneous Long/Short)
"""

import logging
import time
from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_LIMIT,
    FUTURE_ORDER_TYPE_STOP_MARKET,
)
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BinanceFuturesHandler:
    """
    Unified Binance Futures API Handler for BOTH Long & Short positions
    ✅ FIXED: Take Profit precision handling
    ✨ ENHANCED: Forced Hedge Mode for Asymmetric Trading
    """

    def __init__(self, client: Client, symbol: str = "BTCUSDT", config: dict = None):
        self.client = client
        self.symbol = symbol
        self.filters = {}
        self.config = config or {}

        self.quantity_precision = 3  # Default for BTCUSDT
        self.price_precision = 2  # Default for BTCUSDT
        self.tick_size = 0.01  # Default tick size
        self.step_size = 0.001  # Default step size
        self.min_qty = 0.001  # Default min quantity
        self.min_notional = 5.0  # Default min notional

        # ✨ NEW: Hedging Configuration
        self.allow_hedging = self.config.get("trading", {}).get("allow_simultaneous_long_short", True)

        # Verify Futures API access and load filters
        try:
            self.client.futures_account()
            self._load_symbol_filters()
            
            # ====================================================================
            # ✨ NEW: FORCE BINANCE INTO HEDGE MODE
            # Required for the bot to hold a Trend Long and a Scalp Short simultaneously.
            # ====================================================================
            try:
                self.client.futures_change_position_mode(dualSidePosition=self.allow_hedging)
                mode_str = "HEDGE" if self.allow_hedging else "ONE-WAY"
                logger.info(f"[FUTURES] ✓ Account Position Mode set to: {mode_str}")
            except Exception as e:
                if "-4059" in str(e) or "No need to change" in str(e):
                    logger.debug("[FUTURES] Hedge Mode already correctly set.")
                else:
                    logger.error(f"[FUTURES] Failed to set Hedge Mode: {e}")

            logger.info(f"[FUTURES] ✓ Binance Futures API connected for {symbol}")
        except Exception as e:
            logger.error(f"[FUTURES] ✗ Futures API unavailable: {e}")
            raise

    def _load_symbol_filters(self):
        """Load LOT_SIZE, PRICE_FILTER, and MIN_NOTIONAL filters for the symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == self.symbol:
                    # ✅ Get base precision from symbol info
                    self.quantity_precision = int(s.get("quantityPrecision", 3))
                    self.price_precision = int(s.get("pricePrecision", 2))

                    # ✅ Parse filters
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            self.step_size = float(f["stepSize"])
                            self.min_qty = float(f["minQty"])
                            self.filters["step_size"] = self.step_size
                            self.filters["min_qty"] = self.min_qty

                        elif f["filterType"] == "PRICE_FILTER":
                            self.tick_size = float(f["tickSize"])
                            self.filters["tick_size"] = self.tick_size

                        elif f["filterType"] == "MIN_NOTIONAL":
                            self.min_notional = float(f.get("notional", 5.0))
                            self.filters["min_notional"] = self.min_notional

                    # ✅ Store precision in filters for backward compatibility
                    self.filters["precision"] = self.quantity_precision

                    logger.info(
                        f"[FUTURES] Filters loaded for {self.symbol}:\n"
                        f"  Quantity: precision={self.quantity_precision}, step={self.step_size}, min={self.min_qty}\n"
                        f"  Price:    precision={self.price_precision}, tick={self.tick_size}\n"
                        f"  Notional: min=${self.min_notional}"
                    )
                    return

            logger.warning(f"[FUTURES] Symbol {self.symbol} not found in exchange info")

        except Exception as e:
            logger.error(f"[FUTURES] Failed to load filters: {e}")
            # Fallback defaults already set in __init__
            logger.info(f"[FUTURES] Using default precision values for {self.symbol}")

    def _adjust_quantity(self, quantity: float) -> float:
        """Round quantity to valid step size"""
        import math

        step = self.filters.get("step_size", 0.001)
        precision = self.filters.get("precision", 3)

        # Round down to nearest step to avoid "LOT_SIZE" error
        quantity = math.floor(quantity / step) * step
        return round(quantity, precision)

    def set_leverage(self, leverage: int = 10) -> bool:
        """Set leverage for the trading pair"""
        try:
            response = self.client.futures_change_leverage(
                symbol=self.symbol, leverage=leverage
            )
            logger.info(f"[FUTURES] Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"[FUTURES] Failed to set leverage: {e}")
            return False

    def set_margin_type(self, margin_type: str = "CROSSED") -> bool:
        """Set margin type (ISOLATED or CROSSED)"""
        try:
            response = self.client.futures_change_margin_type(
                symbol=self.symbol, marginType=margin_type
            )
            logger.info(f"[FUTURES] Margin type set to {margin_type} for {self.symbol}")
            return True
        except Exception as e:
            # Error code -4046 means margin type is already set
            if "-4046" in str(e):
                logger.debug(f"[FUTURES] Margin type already {margin_type}")
                return True
            logger.error(f"[FUTURES] Failed to set margin type: {e}")
            return False

    def _round_price(self, price: float) -> float:
        """
        ✅ FIXED: Round price to exchange tick size
        """
        if not hasattr(self, "tick_size"):
            return round(price, self.price_precision)

        # Round to nearest tick
        rounded = round(price / self.tick_size) * self.tick_size
        # Then round to correct decimal places
        rounded = round(rounded, self.price_precision)

        logger.debug(f"[PRICE] {price:.8f} → {rounded:.{self.price_precision}f}")
        return rounded

    def _round_quantity(self, quantity: float) -> float:
        """
        ✅ FIXED: Round quantity to exchange step size
        """
        if not hasattr(self, "step_size"):
            return round(quantity, self.quantity_precision)

        # Round to nearest step
        rounded = round(quantity / self.step_size) * self.step_size
        # Then round to correct decimal places
        rounded = round(rounded, self.quantity_precision)

        logger.debug(f"[QTY] {quantity:.8f} → {rounded:.{self.quantity_precision}f}")
        return rounded

    def _validate_order(self, price: float, quantity: float) -> Tuple[bool, str]:
        """
        ✅ NEW: Validate order against Binance filters
        """
        # Check minimum quantity
        if quantity < self.min_qty:
            return False, f"Quantity {quantity} < minimum {self.min_qty}"

        # Check minimum notional (price * quantity)
        notional = price * quantity
        if notional < self.min_notional:
            return False, f"Notional ${notional:.2f} < minimum ${self.min_notional}"

        # Check price precision
        price_str = f"{price:.{self.price_precision}f}"
        if float(price_str) != price:
            return False, f"Price precision error: {price} vs {price_str}"

        return True, "OK"

    def open_short_position(
        self, quantity: float, stop_loss: float = None, take_profit: float = None
    ) -> Optional[Dict]:
        """
        Open a SHORT position on Binance Futures
        """
        return self._open_position(
            side="short",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def open_long_position(
        self, quantity: float, stop_loss: float = None, take_profit: float = None
    ) -> Optional[Dict]:
        """
        Open a LONG position on Binance Futures
        """
        return self._open_position(
            side="long", quantity=quantity, stop_loss=stop_loss, take_profit=take_profit
        )

    def _open_position(
        self,
        side: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Cancel existing stops before placing new ones
        ✨ ENHANCED: Injects `positionSide` for Hedge Mode compatibility
        """
        try:
            # Determine Binance side
            binance_side = SIDE_BUY if side == "long" else SIDE_SELL
            side_label = side.upper()
            
            # ✨ NEW: Define strictly required Position Side for Hedge Mode
            position_side = "LONG" if side == "long" else "SHORT"

            # ✅ Round values BEFORE validation
            quantity = self._round_quantity(quantity)
            if stop_loss:
                stop_loss = self._round_price(stop_loss)
            if take_profit:
                take_profit = self._round_price(take_profit)

            logger.info(
                f"[FUTURES] Opening {side_label} (Hedge Mode): {quantity:.{self.quantity_precision}f} {self.symbol}"
            )

            # Get current price for validation
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker["price"])

            # Validate entry order
            is_valid, error_msg = self._validate_order(current_price, quantity)
            if not is_valid:
                logger.error(f"[FUTURES] Entry validation failed: {error_msg}")
                return None

            # 1. Place MARKET ENTRY Order (✨ ENHANCED WITH positionSide)
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=binance_side,
                    positionSide=position_side, # ✨ REQUIRED FOR HEDGE MODE
                    type=FUTURE_ORDER_TYPE_MARKET,
                    quantity=quantity,
                )
            except Exception as e:
                logger.error(f"[FUTURES] Entry failed: {e}")
                return None

            order_id = order.get("orderId")
            avg_price = float(order.get("avgPrice", 0))

            logger.info(
                f"[FUTURES] ✓ {side_label} opened\n"
                f"  Order ID: {order_id}\n"
                f"  Quantity: {quantity:.{self.quantity_precision}f} BTC\n"
                f"  Entry:    ${avg_price:,.2f}"
            )

            # ============================================================
            # 2. ✅ FIX: Cancel existing stop orders BEFORE placing new one
            # ============================================================
            if stop_loss:
                logger.info(f"[SL] Preparing to place stop loss for {side_label}...")

                # Step 1: Cancel ALL existing stop orders (not just same side)
                try:
                    logger.info(f"[SL] Cancelling ALL existing stop orders...")
                    cancelled_orders = self.client.futures_cancel_all_open_orders(
                        symbol=self.symbol
                    )

                    if cancelled_orders:
                        logger.info(
                            f"[SL] ✓ Cancelled {len(cancelled_orders)} existing order(s)"
                        )
                    else:
                        logger.debug(f"[SL] No existing orders to cancel")

                except Exception as e:
                    # Error -2011 means "Unknown order sent" - no orders to cancel, which is fine
                    if "-2011" in str(e) or "Unknown order" in str(e):
                        logger.debug(f"[SL] No existing orders (expected)")
                    else:
                        logger.warning(f"[SL] Cancel error (continuing anyway): {e}")

                # Step 2: Wait for cancellation to process on exchange
                time.sleep(0.5)  # Increased from 0.2 to 0.5 seconds

                # Step 3: Verify no stop orders exist before placing new one
                max_verification_attempts = 3
                for verify_attempt in range(1, max_verification_attempts + 1):
                    try:
                        open_orders = self.client.futures_get_open_orders(
                            symbol=self.symbol
                        )
                        stop_orders = [
                            o for o in open_orders if o.get("type") == "STOP_MARKET"
                        ]

                        if not stop_orders:
                            logger.info(
                                f"[SL] ✓ Verified: No stop orders exist (attempt {verify_attempt})"
                            )
                            break
                        else:
                            logger.warning(
                                f"[SL] ⚠️ Found {len(stop_orders)} existing stop order(s) "
                                f"(attempt {verify_attempt}/{max_verification_attempts})"
                            )

                            # Try cancelling again
                            for order_to_cancel in stop_orders:
                                try:
                                    self.client.futures_cancel_order(
                                        symbol=self.symbol, orderId=order_to_cancel["orderId"]
                                    )
                                    logger.info(
                                        f"[SL] Cancelled lingering order {order_to_cancel['orderId']}"
                                    )
                                except Exception as cancel_err:
                                    logger.error(
                                        f"[SL] Failed to cancel {order_to_cancel['orderId']}: {cancel_err}"
                                    )

                            time.sleep(0.5)

                    except Exception as verify_err:
                        logger.warning(f"[SL] Verification error: {verify_err}")
                        break

                # Step 4: Validate and place stop loss
                sl_side = SIDE_SELL if side == "long" else SIDE_BUY

                # Validate SL price
                if side == "long" and stop_loss >= current_price:
                    logger.error(
                        f"[FUTURES] Invalid SL: ${stop_loss} >= entry ${current_price}"
                    )
                    stop_loss = current_price * 0.97
                    stop_loss = self._round_price(stop_loss)
                elif side == "short" and stop_loss <= current_price:
                    logger.error(
                        f"[FUTURES] Invalid SL: ${stop_loss} <= entry ${current_price}"
                    )
                    stop_loss = current_price * 1.03
                    stop_loss = self._round_price(stop_loss)

                # Step 5: Place stop loss with enhanced retry logic (✨ ENHANCED WITH positionSide)
                sl_success = False
                sl_last_error = None

                for attempt in range(1, 4):
                    try:
                        logger.info(
                            f"[SL] Attempt {attempt}/3: Placing stop @ ${stop_loss:,.{self.price_precision}f}"
                        )

                        self.client.futures_create_order(
                            symbol=self.symbol,
                            side=sl_side,
                            positionSide=position_side, # ✨ REQUIRED FOR HEDGE MODE
                            type=FUTURE_ORDER_TYPE_STOP_MARKET,
                            stopPrice=stop_loss,
                            closePosition=True,  # This closes the entire position when hit
                        )

                        logger.info(
                            f"  ✓ SL Placed: ${stop_loss:,.{self.price_precision}f}"
                        )
                        sl_success = True
                        break

                    except Exception as e:
                        sl_last_error = str(e)

                        # Check if it's the -4130 error (stop already exists)
                        if "-4130" in sl_last_error:
                            logger.error(
                                f"  ❌ SL Attempt {attempt}/3 FAILED with -4130\n"
                                f"  This means a stop order STILL exists on the exchange!\n"
                                f"  Trying more aggressive cancellation..."
                            )

                            # Emergency: try to cancel ALL orders again
                            try:
                                self.client.futures_cancel_all_open_orders(
                                    symbol=self.symbol
                                )
                                time.sleep(1.0)  # Longer wait
                            except:
                                pass

                        else:
                            logger.warning(f"  ⚠️ SL Attempt {attempt}/3 failed: {e}")

                        if attempt < 3:
                            time.sleep(0.5 * attempt)  # Exponential backoff

                # Step 6: Emergency failsafe (✨ ENHANCED WITH positionSide)
                if not sl_success:
                    logger.critical(
                        f"\n{'='*80}\n"
                        f"🛑 CRITICAL: STOP LOSS PLACEMENT FAILED!\n"
                        f"{'='*80}\n"
                        f"Last Error: {sl_last_error}\n"
                        f"Side: {side_label}\n"
                        f"Quantity: {quantity:.{self.quantity_precision}f}\n"
                        f"Entry: ${avg_price:,.2f}\n"
                        f"\n"
                        f"⚠️  POSITION IS NOW UNPROTECTED!\n"
                        f"Executing EMERGENCY CLOSE to prevent unlimited loss...\n"
                        f"{'='*80}\n"
                    )

                    close_side = SIDE_SELL if side == "long" else SIDE_BUY
                    try:
                        emergency_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            positionSide=position_side, # ✨ REQUIRED FOR HEDGE MODE
                            type=FUTURE_ORDER_TYPE_MARKET,
                            quantity=quantity,
                            reduceOnly=True,
                        )
                        logger.critical(
                            f"[FUTURES] ✓ EMERGENCY CLOSE SUCCESSFUL\n"
                            f"  Order ID: {emergency_order.get('orderId')}\n"
                            f"  Reason: Stop loss placement failed after 3 attempts"
                        )
                    except Exception as close_error:
                        logger.critical(
                            f"[FUTURES] ☠️ EMERGENCY CLOSE ALSO FAILED!\n"
                            f"  Error: {close_error}\n"
                            f"  ⚠️  MANUAL INTERVENTION REQUIRED!\n"
                            f"  Check Binance Futures UI immediately!"
                        )
                    return None

            # 4. Place Take Profit (✨ ENHANCED WITH positionSide)
            if take_profit:
                tp_side = SIDE_SELL if side == "long" else SIDE_BUY

                if side == "long" and take_profit <= current_price:
                    logger.warning(
                        f"[FUTURES] Invalid TP: ${take_profit} <= entry ${current_price}, skipping"
                    )
                    take_profit = None
                elif side == "short" and take_profit >= current_price:
                    logger.warning(
                        f"[FUTURES] Invalid TP: ${take_profit} >= entry ${current_price}, skipping"
                    )
                    take_profit = None

                if take_profit:
                    tp_notional = take_profit * quantity
                    if tp_notional < self.min_notional:
                        logger.warning(
                            f"[FUTURES] TP notional ${tp_notional:.2f} < ${self.min_notional}, "
                            f"adjusting quantity"
                        )
                        adjusted_qty = (self.min_notional / take_profit) * 1.01
                        adjusted_qty = self._round_quantity(adjusted_qty)

                        if adjusted_qty <= quantity:
                            quantity = adjusted_qty
                            logger.info(
                                f"[FUTURES] Adjusted TP quantity to {quantity:.{self.quantity_precision}f}"
                            )
                        else:
                            logger.warning(
                                f"[FUTURES] Cannot adjust TP quantity, skipping TP"
                            )
                            take_profit = None

                    if take_profit:
                        try:
                            tp_order = self.client.futures_create_order(
                                symbol=self.symbol,
                                side=tp_side,
                                positionSide=position_side, # ✨ REQUIRED FOR HEDGE MODE
                                type=FUTURE_ORDER_TYPE_LIMIT,
                                price=take_profit,
                                quantity=quantity,
                                timeInForce="GTC",
                                reduceOnly=True,
                            )
                            logger.info(
                                f"  ✓ TP Placed: ${take_profit:,.{self.price_precision}f}"
                            )
                        except Exception as e:
                            logger.warning(f"  ⚠️ TP Failed: {e}")

            return order

        except Exception as e:
            logger.error(f"[FUTURES] Failed to execute position: {e}", exc_info=True)
            return None

    def close_short_position(
        self, quantity: float = None, order_id: int = None
    ) -> bool:
        """Close a SHORT position (buy back)"""
        return self._close_position(side="short", quantity=quantity, order_id=order_id)

    def close_long_position(self, quantity: float = None, order_id: int = None) -> bool:
        """Close a LONG position (sell back)"""
        return self._close_position(side="long", quantity=quantity, order_id=order_id)

    def _close_position(
        self, side: str, quantity: float = None, order_id: int = None
    ) -> bool:
        """
        ✅ FIXED: Close LONG or SHORT position with detailed error reporting
        ✨ ENHANCED: Injects `positionSide` for Hedge Mode compatibility
        """
        try:
            # Step 1: Get position info if quantity not provided
            if quantity is None:
                position = self.get_position_info(side=side)
                if position:
                    pos_amt = float(position.get("positionAmt", 0))

                    if side == "short" and pos_amt >= 0:
                        logger.warning(
                            f"[FUTURES] No SHORT position to close (posAmt: {pos_amt})"
                        )
                        return False
                    elif side == "long" and pos_amt <= 0:
                        logger.warning(
                            f"[FUTURES] No LONG position to close (posAmt: {pos_amt})"
                        )
                        return False

                    quantity = abs(pos_amt)
                else:
                    logger.warning("[FUTURES] No open position found")
                    return False

            # Step 2: Round quantity to valid precision
            quantity = self._round_quantity(quantity)

            close_side = SIDE_SELL if side == "long" else SIDE_BUY
            side_label = side.upper()
            
            # ✨ NEW: Define strictly required Position Side for Hedge Mode
            position_side = "LONG" if side == "long" else "SHORT"

            logger.info(
                f"[FUTURES] Closing {side_label} (Hedge Mode): {quantity:.{self.quantity_precision}f} {self.symbol}"
            )

            # Step 3: Execute market close order with reduceOnly=True (✨ ENHANCED WITH positionSide)
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=close_side,
                    positionSide=position_side, # ✨ REQUIRED FOR HEDGE MODE
                    type=FUTURE_ORDER_TYPE_MARKET,
                    quantity=quantity,
                    reduceOnly=True,  # ← Critical: This ensures we're closing, not opening
                )

                logger.info(
                    f"[FUTURES] ✓ {side_label} closed\n"
                    f"  Order ID:  {order.get('orderId')}\n"
                    f"  Quantity:  {quantity:.{self.quantity_precision}f} BTC\n"
                    f"  Exit:      ${float(order.get('avgPrice', 0)):,.2f}\n"
                    f"  Status:    {order.get('status')}"
                )

                # Step 4: Cancel any remaining SL/TP orders
                try:
                    cancelled = self.client.futures_cancel_all_open_orders(
                        symbol=self.symbol
                    )
                    if cancelled:
                        logger.debug(
                            f"[FUTURES] Cancelled {len(cancelled)} remaining order(s)"
                        )
                except Exception as e:
                    # -2011 means no orders to cancel, which is fine
                    if "-2011" not in str(e):
                        logger.debug(f"[FUTURES] Cancel orders: {e}")

                return True

            except Exception as order_error:
                logger.error(
                    f"[FUTURES] ❌ Order execution failed\n"
                    f"  Side:     {side_label}\n"
                    f"  Quantity: {quantity:.{self.quantity_precision}f}\n"
                    f"  Error:    {str(order_error)}"
                )

                # Check if it's a known error
                error_str = str(order_error)
                if "ReduceOnly Order is rejected" in error_str:
                    logger.error(
                        f"[FUTURES] No position exists to close!\n"
                        f"  This usually means position was already closed manually"
                    )
                elif "Insufficient balance" in error_str:
                    logger.error(f"[FUTURES] Insufficient balance to close position")
                elif "Invalid quantity" in error_str:
                    logger.error(
                        f"[FUTURES] Invalid quantity: {quantity:.{self.quantity_precision}f}\n"
                        f"  Min: {self.min_qty}, Step: {self.step_size}"
                    )

                return False

        except Exception as e:
            logger.error(
                f"[FUTURES] ❌ Failed to close {side.upper()} position\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            return False

    def _cancel_existing_stop_orders(self, side: str) -> bool:
        """
        Cancel all existing stop-loss orders for the given side
        This is CRITICAL before placing new stop orders to avoid -4130 error

        Args:
            side: "long" or "short"

        Returns:
            True if successful or no orders to cancel
        """
        try:
            # Get all open orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)

            if not open_orders:
                logger.debug(f"[SL] No open orders for {self.symbol}")
                return True

            # Determine which stop side to look for
            # LONG positions have SELL stops, SHORT positions have BUY stops
            target_stop_side = "SELL" if side == "long" else "BUY"

            cancelled_count = 0
            for order in open_orders:
                order_type = order.get("type", "")
                order_side = order.get("side", "")

                # Look for STOP_MARKET orders on the opposite side
                if order_type == "STOP_MARKET" and order_side == target_stop_side:
                    order_id = order.get("orderId")

                    try:
                        self.client.futures_cancel_order(
                            symbol=self.symbol, orderId=order_id
                        )
                        cancelled_count += 1
                        logger.info(f"[SL] Cancelled existing stop order {order_id}")
                    except Exception as e:
                        logger.warning(f"[SL] Failed to cancel order {order_id}: {e}")

            if cancelled_count > 0:
                logger.info(f"[SL] Cancelled {cancelled_count} existing stop order(s)")

            return True

        except Exception as e:
            logger.error(f"[SL] Error cancelling stop orders: {e}")
            return False

    def get_position_info(self, side: str = None) -> Optional[Dict]:
        """
        Get current position information
        """
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)

            for pos in positions:
                if pos["symbol"] == self.symbol:
                    pos_amt = float(pos.get("positionAmt", 0))

                    if pos_amt == 0:
                        continue

                    if side == "short" and pos_amt >= 0:
                        continue
                    elif side == "long" and pos_amt <= 0:
                        continue

                    pos["side"] = "long" if pos_amt > 0 else "short"
                    return pos

            return None

        except Exception as e:
            logger.error(f"[FUTURES] Error getting position: {e}")
            return None

    def get_unrealized_pnl(self) -> float:
        """
        Get unrealized P&L for current position
        """
        try:
            position = self.get_position_info()
            if position:
                return float(position.get("unRealizedProfit", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[FUTURES] Error getting P&L: {e}")
            return 0.0

    def get_account_balance(self) -> float:
        """
        Get Futures account balance
        """
        try:
            account = self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] == "USDT":
                    return float(asset.get("availableBalance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[FUTURES] Error getting balance: {e}")
            return 0.0


# ============================================================================
# INTEGRATION WITH EXISTING BinanceExecutionHandler
# ============================================================================


def integrate_futures_into_handler(handler):
    """
    Integrate Futures API into existing BinanceExecutionHandler
    ✨ ENHANCED: Passes config to the FuturesHandler to load hedging rules.
    """
    # Check if Futures is enabled
    futures_enabled = (
        handler.config.get("assets", {}).get("BTC", {}).get("enable_futures", False)
    )

    if not futures_enabled:
        logger.warning("[FUTURES] Futures trading disabled in config")
        return False

    try:
        # Initialize Futures handler with Config
        handler.futures_handler = BinanceFuturesHandler(
            client=handler.client, 
            symbol=handler.symbol,
            config=handler.config # ✨ NEW: Pass config for hedging awareness
        )

        # Set leverage and margin type
        leverage = handler.config.get("assets", {}).get("BTC", {}).get("leverage", 10)
        margin_type = (
            handler.config.get("assets", {})
            .get("BTC", {})
            .get("margin_type", "CROSSED")
        )

        handler.futures_handler.set_leverage(leverage)
        handler.futures_handler.set_margin_type(margin_type)

        logger.info("[FUTURES] ✓ Futures handler integrated")
        return True

    except Exception as e:
        logger.error(f"[FUTURES] Integration failed: {e}")
        return False


def patch_open_position_method(handler):
    """
    Patch the _open_position method to use Futures for BOTH longs and shorts
    """

    original_open_position = handler._open_position

    def _open_position_with_futures(
        signal: int, current_price: float, asset_name: str, **kwargs
    ):
        """
        Enhanced _open_position that uses Futures API for both LONG and SHORT
        """

        side = "long" if signal == 1 else "short"

        # If Futures is enabled, use it for BOTH directions
        if hasattr(handler, "futures_handler"):
            try:
                logger.info(f"[FUTURES] Using Futures API for {side.upper()} position")

                # Calculate SL/TP prices
                risk = handler.asset_config.get("risk", {})

                if side == "long":
                    stop_loss_pct = risk.get("stop_loss_pct", 0.05)
                    take_profit_pct = risk.get("take_profit_pct", 0.10)
                    trailing_stop_pct = risk.get("trailing_stop_pct", 0.03)

                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    take_profit_price = current_price * (1 + take_profit_pct)
                else:  # short
                    stop_loss_pct = risk.get(
                        "stop_loss_pct_short", risk.get("stop_loss_pct", 0.04)
                    )
                    take_profit_pct = risk.get(
                        "take_profit_pct_short", risk.get("take_profit_pct", 0.08)
                    )
                    trailing_stop_pct = risk.get(
                        "trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.025)
                    )

                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    take_profit_price = current_price * (1 - take_profit_pct)

                # ✅ FIX: Round stop loss to correct precision BEFORE sizing
                stop_loss_price = round(stop_loss_price, 2)

                # Calculate position size using risk-based method
                position_size_usd, sizing_metadata = (
                    handler.sizer.calculate_size_risk_based(
                        asset=asset_name,
                        entry_price=current_price,
                        stop_loss_price=stop_loss_price,
                        signal=signal,
                        confidence_score=kwargs.get("confidence_score"),
                        market_condition=kwargs.get("market_condition", "neutral"),
                        sizing_mode=kwargs.get("sizing_mode", "automated"),
                        manual_size_usd=kwargs.get("manual_size_usd"),
                        override_reason=kwargs.get("override_reason"),
                    )
                )

                if position_size_usd <= 0:
                    logger.error(
                        f"[FUTURES] Invalid position size: ${position_size_usd:.2f}"
                    )
                    return False

                # Calculate quantity and round it
                quantity = position_size_usd / current_price
                quantity = handler.futures_handler._round_quantity(quantity)

                logger.info(
                    f"[FUTURES] Opening {side.upper()} position:\n"
                    f"  Size: ${position_size_usd:,.2f}\n"
                    f"  Quantity: {quantity:.6f} BTC\n"
                    f"  Entry: ${current_price:,.2f}\n"
                    f"  Stop Loss: ${stop_loss_price:,.2f}\n"
                    f"  Take Profit: ${take_profit_price:,.2f}"
                )

                # Open position on Futures
                if side == "long":
                    order = handler.futures_handler.open_long_position(
                        quantity=quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )
                else:
                    order = handler.futures_handler.open_short_position(
                        quantity=quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )

                if not order:
                    logger.error(f"[FUTURES] Failed to open {side.upper()} position")
                    return False

                order_id = order.get("orderId")

                # Fetch OHLC for VTM
                ohlc_data = None
                if handler.data_manager:
                    try:
                        from datetime import datetime, timedelta, timezone

                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = handler.data_manager.fetch_binance_data(
                            symbol=handler.symbol,
                            interval=handler.asset_config.get("interval", "1h"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        if len(df) > 0:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                    except Exception as e:
                        logger.warning(f"[VTM] OHLC fetch failed: {e}")

                # Add to portfolio
                success = handler.portfolio_manager.add_position(
                    asset=asset_name,
                    symbol=handler.symbol,
                    side=side,
                    entry_price=current_price,
                    position_size_usd=position_size_usd,
                    stop_loss=None,  # VTM will manage
                    take_profit=None,
                    trailing_stop_pct=trailing_stop_pct,
                    binance_order_id=order_id,
                    ohlc_data=ohlc_data,
                    use_dynamic_management=True,
                )

                if success:
                    logger.info(
                        f"[OK] {asset_name} {side.upper()} position opened via Futures"
                    )
                    logger.info(f"  └─ Order ID: {order_id}")
                    if ohlc_data:
                        logger.info(f"  └─ VTM: ACTIVE")
                    return True
                else:
                    logger.error(f"[FAIL] Portfolio rejected {side.upper()} position")
                    # Rollback - close the Futures position
                    if side == "long":
                        handler.futures_handler.close_long_position(quantity=quantity)
                    else:
                        handler.futures_handler.close_short_position(quantity=quantity)
                    return False

            except Exception as e:
                logger.error(
                    f"[FUTURES] Error opening {side.upper()}: {e}", exc_info=True
                )
                return False

        # Fallback to original method (Spot) if Futures disabled
        else:
            return original_open_position(
                signal=signal,
                current_price=current_price,
                asset_name=asset_name,
                **kwargs,
            )

    # Replace method
    handler._open_position = _open_position_with_futures
    logger.info("[FUTURES] _open_position method patched for LONG+SHORT")


def patch_close_position_method(handler):
    """
    Patch the _close_position method to use Futures for BOTH longs and shorts
    """

    original_close_position = handler._close_position

    def _close_position_with_futures(
        position, current_price: float, asset_name: str, reason: str
    ):
        """
        Enhanced _close_position that uses Futures API for both LONG and SHORT
        """

        # If Futures enabled, use it for both directions
        if hasattr(handler, "futures_handler") and handler.futures_handler:
            try:
                side = position.side
                quantity = position.quantity
                order_id = position.binance_order_id

                logger.info(
                    f"[FUTURES] Closing {side.upper()} position via Futures API\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Order ID:    {order_id}\n"
                    f"  Quantity:    {quantity:.6f}\n"
                    f"  Reason:      {reason}"
                )

                # Get current P&L from Futures before closing
                futures_position = handler.futures_handler.get_position_info(side=side)
                if futures_position:
                    futures_pnl = float(futures_position.get("unRealizedProfit", 0))
                    logger.info(f"  Unrealized P&L: ${futures_pnl:,.2f}")
                else:
                    logger.warning(f"  Could not fetch Futures position info")
                    futures_pnl = 0

                # ✅ CRITICAL FIX: Close on Futures with proper error handling
                success = False

                if side == "long":
                    success = handler.futures_handler.close_long_position(
                        quantity=quantity, order_id=order_id
                    )
                elif side == "short":
                    success = handler.futures_handler.close_short_position(
                        quantity=quantity, order_id=order_id
                    )
                else:
                    logger.error(f"[FUTURES] Invalid side: {side}")
                    return False

                # ✅ Check if close was successful
                if not success:
                    logger.error(
                        f"[FUTURES] ❌ Failed to close {side.upper()} position\n"
                        f"  Position ID: {position.position_id}\n"
                        f"  Order ID:    {order_id}\n"
                        f"  Quantity:    {quantity:.6f}\n"
                        f"  Reason:      Futures API returned False\n"
                        f"  Action:      Position remains open on exchange"
                    )
                    return False

                # ✅ Success - log details
                logger.info(
                    f"[FUTURES] ✅ {side.upper()} position closed successfully\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Final P&L:   ${futures_pnl:,.2f}"
                )

                # Close in portfolio
                trade_result = handler.portfolio_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=reason,
                )

                if trade_result:
                    logger.info(f"[OK] Portfolio updated after {side.upper()} close")
                    return True
                else:
                    logger.error(
                        f"[FAIL] Portfolio close failed for {side.upper()}\n"
                        f"  Warning: Position closed on exchange but not in portfolio!"
                    )
                    return False

            except Exception as e:
                logger.error(
                    f"[FUTURES] ❌ Exception closing {position.side.upper()} position\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Error:       {str(e)}\n"
                    f"  Traceback:",
                    exc_info=True,
                )
                return False

        # Fallback to original method (Spot) if Futures disabled
        else:
            logger.warning(f"[FUTURES] Handler not available, using fallback method")
            return original_close_position(
                position=position,
                current_price=current_price,
                asset_name=asset_name,
                reason=reason,
            )

    # Replace method
    handler._close_position = _close_position_with_futures
    logger.info("[FUTURES] _close_position method patched for LONG+SHORT")


# Open: src/execution/binance_futures.py
# Scroll to the bottom and replace this function:

def enable_futures_for_binance_handler(handler):
    """
    MAIN FUNCTION: Enable Futures trading for BOTH LONG and SHORT positions
    ✅ FIXED: Removed legacy "monkey patches". The handler now natively supports futures.
    """

    try:
        logger.info("\n" + "=" * 70)
        logger.info("ENABLING BINANCE FUTURES FOR LONG + SHORT TRADING")
        logger.info("=" * 70)

        # Step 1: Integrate Futures handler
        if not integrate_futures_into_handler(handler):
            return False

        # NOTE: We NO LONGER patch open/close methods here because 
        # binance_handler.py now natively supports Futures margin isolation.

        # Step 2: Verify connection
        balance = handler.futures_handler.get_account_balance()
        logger.info(f"[FUTURES] Account Balance: ${balance:,.2f} USDT")

        # Step 3: Check for existing positions
        position = handler.futures_handler.get_position_info()
        if position:
            side = position.get("side", "unknown")
            pos_amt = abs(float(position.get("positionAmt", 0)))
            unrealized = float(position.get("unRealizedProfit", 0))

            logger.info(f"[FUTURES] Existing {side.upper()} position detected:")
            logger.info(f"  Quantity: {pos_amt:.8f} BTC")
            logger.info(f"  P&L:      ${unrealized:,.2f}")

        logger.info("=" * 70)
        logger.info("✅ FUTURES TRADING ENABLED")
        logger.info("  - Native dynamic margin isolation active")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"[FUTURES] Enablement failed: {e}", exc_info=True)
        return False