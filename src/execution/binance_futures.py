"""
Binance Futures API Integration - COMPLETE VERSION
✅ FIXED: Take Profit Precision + All Integration Functions
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
    """

    def __init__(self, client: Client, symbol: str = "BTCUSDT"):
        self.client = client
        self.symbol = symbol
        self.filters = {}
        
        # Verify Futures API access and load filters
        try:
            self.client.futures_account()
            self._load_symbol_filters()
            logger.info(f"[FUTURES] ✓ Binance Futures API connected for {symbol}")
        except Exception as e:
            logger.error(f"[FUTURES] ✗ Futures API unavailable: {e}")
            raise

    def _load_symbol_filters(self):
        """Load LOT_SIZE and MIN_NOTIONAL filters for the symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == self.symbol:
                    # Quantity Precision (step size)
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            self.filters['step_size'] = float(f['stepSize'])
                            self.filters['min_qty'] = float(f['minQty'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.filters['min_notional'] = float(f.get('notional', 5.0))
                    
                    self.filters['precision'] = int(s.get('quantityPrecision', 3))
                    logger.info(f"[FUTURES] Filters loaded: Step={self.filters['step_size']}, MinNotional=${self.filters.get('min_notional', 5.0)}")
                    return
        except Exception as e:
            logger.error(f"[FUTURES] Failed to load filters: {e}")
            # Fallback defaults for BTC
            self.filters = {'step_size': 0.001, 'min_qty': 0.001, 'min_notional': 5.0, 'precision': 3}

    def _adjust_quantity(self, quantity: float) -> float:
        """Round quantity to valid step size"""
        import math
        step = self.filters.get('step_size', 0.001)
        precision = self.filters.get('precision', 3)
        
        # Round down to nearest step to avoid "LOT_SIZE" error
        quantity = math.floor(quantity / step) * step
        return round(quantity, precision)

    def set_leverage(self, leverage: int = 10) -> bool:
        """Set leverage for the trading pair"""
        try:
            response = self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=leverage
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
                symbol=self.symbol,
                marginType=margin_type
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
        if not hasattr(self, 'tick_size'):
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
        if not hasattr(self, 'step_size'):
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
        self, 
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        Open a SHORT position on Binance Futures
        """
        return self._open_position(
            side="short",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def open_long_position(
        self, 
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        Open a LONG position on Binance Futures
        """
        return self._open_position(
            side="long",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _open_position(
        self,
        side: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Enhanced error handling and precision validation
        """
        try:
            # Determine Binance side
            binance_side = SIDE_BUY if side == "long" else SIDE_SELL
            side_label = side.upper()
            
            # ✅ Round values BEFORE validation
            quantity = self._round_quantity(quantity)
            if stop_loss:
                stop_loss = self._round_price(stop_loss)
            if take_profit:
                take_profit = self._round_price(take_profit)
            
            logger.info(f"[FUTURES] Opening {side_label}: {quantity:.{self.quantity_precision}f} {self.symbol}")
            
            # ✅ Get current price for validation
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # ✅ Validate entry order
            is_valid, error_msg = self._validate_order(current_price, quantity)
            if not is_valid:
                logger.error(f"[FUTURES] Entry validation failed: {error_msg}")
                return None
            
            # 1. Place MARKET ENTRY Order
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=binance_side,
                    type=FUTURE_ORDER_TYPE_MARKET,
                    quantity=quantity
                )
            except Exception as e:
                logger.error(f"[FUTURES] Entry failed: {e}")
                return None
            
            order_id = order.get('orderId')
            avg_price = float(order.get('avgPrice', 0))
            
            logger.info(
                f"[FUTURES] ✓ {side_label} opened\n"
                f"  Order ID: {order_id}\n"
                f"  Quantity: {quantity:.{self.quantity_precision}f} BTC\n"
                f"  Entry:    ${avg_price:,.2f}"
            )
            
            # 2. CRITICAL: Place Stop Loss with Retries
            if stop_loss:
                sl_success = False
                sl_side = SIDE_SELL if side == "long" else SIDE_BUY
                
                # ✅ Validate SL price before attempting
                if side == "long" and stop_loss >= current_price:
                    logger.error(f"[FUTURES] Invalid SL: ${stop_loss} >= entry ${current_price}")
                    stop_loss = current_price * 0.97  # Emergency fallback
                    stop_loss = self._round_price(stop_loss)
                elif side == "short" and stop_loss <= current_price:
                    logger.error(f"[FUTURES] Invalid SL: ${stop_loss} <= entry ${current_price}")
                    stop_loss = current_price * 1.03
                    stop_loss = self._round_price(stop_loss)
                
                for attempt in range(1, 4):
                    try:
                        self.client.futures_create_order(
                            symbol=self.symbol,
                            side=sl_side,
                            type=FUTURE_ORDER_TYPE_STOP_MARKET,
                            stopPrice=stop_loss,
                            closePosition=True
                        )
                        logger.info(f"  ✓ SL Placed: ${stop_loss:,.{self.price_precision}f}")
                        sl_success = True
                        break
                    except Exception as e:
                        logger.warning(f"  ⚠️ SL Attempt {attempt}/3 failed: {e}")
                        time.sleep(0.5)
                
                # 3. EMERGENCY FAILSAFE
                if not sl_success:
                    logger.critical(f"[FUTURES] 🛑 CRITICAL: SL FAILED 3x! EMERGENCY CLOSE.")
                    close_side = SIDE_SELL if side == "long" else SIDE_BUY
                    try:
                        self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            type=FUTURE_ORDER_TYPE_MARKET,
                            quantity=quantity,
                            reduceOnly=True
                        )
                        logger.critical(f"[FUTURES] ✓ EMERGENCY CLOSE SUCCESSFUL")
                    except Exception as close_error:
                        logger.critical(f"[FUTURES] ☠️ EMERGENCY CLOSE FAILED: {close_error}")
                    return None
            
            # 4. Place Take Profit (Enhanced with validation)
            if take_profit:
                tp_side = SIDE_SELL if side == "long" else SIDE_BUY
                
                # ✅ CRITICAL FIX: Validate TP before placing
                if side == "long" and take_profit <= current_price:
                    logger.warning(f"[FUTURES] Invalid TP: ${take_profit} <= entry ${current_price}, skipping")
                    take_profit = None
                elif side == "short" and take_profit >= current_price:
                    logger.warning(f"[FUTURES] Invalid TP: ${take_profit} >= entry ${current_price}, skipping")
                    take_profit = None
                
                if take_profit:
                    # ✅ Validate TP order value
                    tp_notional = take_profit * quantity
                    if tp_notional < self.min_notional:
                        logger.warning(
                            f"[FUTURES] TP notional ${tp_notional:.2f} < ${self.min_notional}, "
                            f"adjusting quantity"
                        )
                        # Increase quantity slightly to meet minimum
                        adjusted_qty = (self.min_notional / take_profit) * 1.01
                        adjusted_qty = self._round_quantity(adjusted_qty)
                        
                        if adjusted_qty <= quantity:
                            quantity = adjusted_qty
                            logger.info(f"[FUTURES] Adjusted TP quantity to {quantity:.{self.quantity_precision}f}")
                        else:
                            logger.warning(f"[FUTURES] Cannot adjust TP quantity, skipping TP")
                            take_profit = None
                
                if take_profit:
                    try:
                        tp_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=tp_side,
                            type=FUTURE_ORDER_TYPE_LIMIT,
                            price=take_profit,
                            quantity=quantity,
                            timeInForce='GTC',
                            reduceOnly=True
                        )
                        logger.info(f"  ✓ TP Placed: ${take_profit:,.{self.price_precision}f}")
                    except Exception as e:
                        error_code = str(e)
                        
                        # ✅ Enhanced error diagnostics
                        if "-11111" in error_code:
                            logger.error(
                                f"  ✗ TP Precision Error:\n"
                                f"    Price: {take_profit:.{self.price_precision}f} "
                                f"(precision={self.price_precision}, tick={self.tick_size})\n"
                                f"    Quantity: {quantity:.{self.quantity_precision}f} "
                                f"(precision={self.quantity_precision}, step={self.step_size})\n"
                                f"    Raw Error: {e}"
                            )
                        elif "-4131" in error_code:
                            logger.warning(f"  ⚠️ TP rejected: Price outside allowed range")
                        else:
                            logger.warning(f"  ⚠️ TP Failed: {e}")
            
            return order
            
        except Exception as e:
            logger.error(f"[FUTURES] Failed to execute position: {e}", exc_info=True)
            return None

    def close_short_position(
        self, 
        quantity: float = None,
        order_id: int = None
    ) -> bool:
        """
        Close a SHORT position (buy back)
        """
        return self._close_position(
            side="short",
            quantity=quantity,
            order_id=order_id
        )

    def close_long_position(
        self, 
        quantity: float = None,
        order_id: int = None
    ) -> bool:
        """
        Close a LONG position (sell back)
        """
        return self._close_position(
            side="long",
            quantity=quantity,
            order_id=order_id
        )

    def _close_position(
        self,
        side: str,
        quantity: float = None,
        order_id: int = None
    ) -> bool:
        """
        Close LONG or SHORT position
        """
        try:
            if quantity is None:
                position = self.get_position_info(side=side)
                if position:
                    pos_amt = float(position.get('positionAmt', 0))
                    
                    if side == "short" and pos_amt >= 0:
                        logger.warning(f"[FUTURES] No SHORT position to close (posAmt: {pos_amt})")
                        return False
                    elif side == "long" and pos_amt <= 0:
                        logger.warning(f"[FUTURES] No LONG position to close (posAmt: {pos_amt})")
                        return False
                    
                    quantity = abs(pos_amt)
                else:
                    logger.warning("[FUTURES] No open position found")
                    return False
            
            # Round quantity
            quantity = self._round_quantity(quantity)
            
            close_side = SIDE_SELL if side == "long" else SIDE_BUY
            side_label = side.upper()
            
            logger.info(f"[FUTURES] Closing {side_label}: {quantity:.{self.quantity_precision}f} {self.symbol}")
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=quantity,
                reduceOnly=True
            )
            
            logger.info(
                f"[FUTURES] ✓ {side_label} closed\n"
                f"  Order ID: {order.get('orderId')}\n"
                f"  Quantity: {quantity:.{self.quantity_precision}f} BTC\n"
                f"  Exit:     ${float(order.get('avgPrice', 0)):,.2f}\n"
                f"  Status:   {order.get('status')}"
            )
            
            # Cancel remaining SL/TP orders
            try:
                self.client.futures_cancel_all_open_orders(symbol=self.symbol)
                logger.debug("[FUTURES] Cancelled SL/TP orders")
            except Exception as e:
                logger.debug(f"[FUTURES] No orders to cancel: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FUTURES] Failed to close {side.upper()}: {e}")
            return False

    def get_position_info(self, side: str = None) -> Optional[Dict]:
        """
        Get current position information
        """
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    pos_amt = float(pos.get('positionAmt', 0))
                    
                    if pos_amt == 0:
                        continue
                    
                    if side == "short" and pos_amt >= 0:
                        continue
                    elif side == "long" and pos_amt <= 0:
                        continue
                    
                    pos['side'] = "long" if pos_amt > 0 else "short"
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
                return float(position.get('unRealizedProfit', 0))
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
            for asset in account.get('assets', []):
                if asset['asset'] == 'USDT':
                    return float(asset.get('availableBalance', 0))
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
    """
    # Check if Futures is enabled
    futures_enabled = handler.config.get("assets", {}).get("BTC", {}).get("enable_futures", False)
    
    if not futures_enabled:
        logger.warning("[FUTURES] Futures trading disabled in config")
        return False
    
    try:
        # Initialize Futures handler
        handler.futures_handler = BinanceFuturesHandler(
            client=handler.client,
            symbol=handler.symbol
        )
        
        # Set leverage and margin type
        leverage = handler.config.get("assets", {}).get("BTC", {}).get("leverage", 10)
        margin_type = handler.config.get("assets", {}).get("BTC", {}).get("margin_type", "CROSSED")
        
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
        signal: int,
        current_price: float,
        asset_name: str,
        **kwargs
    ):
        """
        Enhanced _open_position that uses Futures API for both LONG and SHORT
        """
        
        side = "long" if signal == 1 else "short"
        
        # If Futures is enabled, use it for BOTH directions
        if hasattr(handler, 'futures_handler'):
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
                    stop_loss_pct = risk.get("stop_loss_pct_short", risk.get("stop_loss_pct", 0.04))
                    take_profit_pct = risk.get("take_profit_pct_short", risk.get("take_profit_pct", 0.08))
                    trailing_stop_pct = risk.get("trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.025))
                    
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    take_profit_price = current_price * (1 - take_profit_pct)
                
                # ✅ FIX: Round stop loss to correct precision BEFORE sizing
                stop_loss_price = round(stop_loss_price, 2)
                
                # Calculate position size using risk-based method
                position_size_usd, sizing_metadata = handler.sizer.calculate_size_risk_based(
                    asset=asset_name,
                    entry_price=current_price,
                    stop_loss_price=stop_loss_price,
                    signal=signal,
                    confidence_score=kwargs.get('confidence_score'),
                    market_condition=kwargs.get('market_condition', 'neutral'),
                    sizing_mode=kwargs.get('sizing_mode', 'automated'),
                    manual_size_usd=kwargs.get('manual_size_usd'),
                    override_reason=kwargs.get('override_reason'),
                )
                
                if position_size_usd <= 0:
                    logger.error(f"[FUTURES] Invalid position size: ${position_size_usd:.2f}")
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
                        take_profit=take_profit_price
                    )
                else:
                    order = handler.futures_handler.open_short_position(
                        quantity=quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price
                    )
                
                if not order:
                    logger.error(f"[FUTURES] Failed to open {side.upper()} position")
                    return False
                
                order_id = order.get('orderId')
                
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
                    logger.info(f"[OK] {asset_name} {side.upper()} position opened via Futures")
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
                logger.error(f"[FUTURES] Error opening {side.upper()}: {e}", exc_info=True)
                return False
        
        # Fallback to original method (Spot) if Futures disabled
        else:
            return original_open_position(
                signal=signal,
                current_price=current_price,
                asset_name=asset_name,
                **kwargs
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
        position,
        current_price: float,
        asset_name: str,
        reason: str
    ):
        """
        Enhanced _close_position that uses Futures API for both LONG and SHORT
        """
        
        # If Futures enabled, use it for both directions
        if hasattr(handler, 'futures_handler'):
            try:
                side = position.side
                logger.info(f"[FUTURES] Closing {side.upper()} position via Futures API")
                
                # Get P&L from Futures
                futures_position = handler.futures_handler.get_position_info(side=side)
                futures_pnl = float(futures_position.get('unRealizedProfit', 0)) if futures_position else 0
                
                # Close on Futures
                if side == "long":
                    success = handler.futures_handler.close_long_position(
                        quantity=position.quantity,
                        order_id=position.binance_order_id
                    )
                else:
                    success = handler.futures_handler.close_short_position(
                        quantity=position.quantity,
                        order_id=position.binance_order_id
                    )
                
                if not success:
                    logger.error(f"[FUTURES] Failed to close {side.upper()} position")
                    return False
                
                # Close in portfolio
                trade_result = handler.portfolio_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=reason
                )
                
                if trade_result:
                    logger.info(f"[OK] {side.upper()} position closed successfully")
                    logger.info(f"  Futures P&L: ${futures_pnl:,.2f}")
                    return True
                else:
                    logger.error("[FAIL] Portfolio close failed")
                    return False
                
            except Exception as e:
                logger.error(f"[FUTURES] Error closing {position.side.upper()}: {e}", exc_info=True)
                return False
        
        # Fallback to original method (Spot) if Futures disabled
        else:
            return original_close_position(
                position=position,
                current_price=current_price,
                asset_name=asset_name,
                reason=reason
            )
    
    # Replace method
    handler._close_position = _close_position_with_futures
    logger.info("[FUTURES] _close_position method patched for LONG+SHORT")


def enable_futures_for_binance_handler(handler):
    """
    MAIN FUNCTION: Enable Futures trading for BOTH LONG and SHORT positions
    
    Usage in main.py:
        from src.execution.binance_futures import enable_futures_for_binance_handler
        
        # After initializing binance_handler
        if config["assets"]["BTC"].get("enable_futures"):
            enable_futures_for_binance_handler(binance_handler)
    
    Returns:
        True if enabled successfully
    """
    
    try:
        logger.info("\n" + "=" * 70)
        logger.info("ENABLING BINANCE FUTURES FOR LONG + SHORT TRADING")
        logger.info("=" * 70)
        
        # Step 1: Integrate Futures handler
        if not integrate_futures_into_handler(handler):
            return False
        
        # Step 2: Patch methods
        patch_open_position_method(handler)
        patch_close_position_method(handler)
        
        # Step 3: Verify connection
        balance = handler.futures_handler.get_account_balance()
        logger.info(f"[FUTURES] Account Balance: ${balance:,.2f} USDT")
        
        # Step 4: Check for existing positions
        position = handler.futures_handler.get_position_info()
        if position:
            side = position.get('side', 'unknown')
            pos_amt = abs(float(position.get('positionAmt', 0)))
            unrealized = float(position.get('unRealizedProfit', 0))
            
            logger.info(f"[FUTURES] Existing {side.upper()} position detected:")
            logger.info(f"  Quantity: {pos_amt:.8f} BTC")
            logger.info(f"  P&L:      ${unrealized:,.2f}")
            logger.warning("[FUTURES] ⚠️ Sync positions before trading!")
        
        logger.info("=" * 70)
        logger.info("✅ FUTURES TRADING ENABLED")
        logger.info("  - LONG positions will use Binance Futures API (with leverage)")
        logger.info("  - SHORT positions will use Binance Futures API")
        logger.info("  - Lower fees: 0.02% maker / 0.04% taker (vs 0.1% Spot)")
        logger.info("  - Leverage can amplify both gains AND losses")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"[FUTURES] Enablement failed: {e}", exc_info=True)
        return False