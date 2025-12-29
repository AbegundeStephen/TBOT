"""
Binance Futures API Integration for SHORT Trading
Replaces simulated shorts with actual Binance Futures orders
"""

import logging
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
    
    Benefits over Spot:
    - Leverage trading (1-125x)
    - True shorting (not just selling holdings)
    - Lower fees (0.02% maker / 0.04% taker)
    - Better liquidity
    - Advanced order types (TP/SL combined)
    
    Integrates with existing BinanceExecutionHandler
    """

    def __init__(self, client: Client, symbol: str = "BTCUSDT"):
        self.client = client
        self.symbol = symbol
        
        # Verify Futures API access
        try:
            self.client.futures_account()
            logger.info("[FUTURES] ✓ Binance Futures API connected")
        except Exception as e:
            logger.error(f"[FUTURES] ✗ Futures API unavailabl6e: {e}")
            raise

    def set_leverage(self, leverage: int = 10) -> bool:
        """
        Set leverage for the trading pair
        
        Args:
            leverage: Leverage multiplier (1-125 depending on symbol)
        
        Returns:
            True if successful
        """
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
        """
        Set margin type (ISOLATED or CROSSED)
        
        Args:
            margin_type: "ISOLATED" or "CROSSED"
        
        Returns:
            True if successful
        """
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

    def open_short_position(
        self, 
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Dict]:
        """
        Open a SHORT position on Binance Futures
        
        Args:
            quantity: Amount of BTC to short
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        
        Returns:
            Order response dict or None if failed
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
        
        Args:
            quantity: Amount of BTC to long
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        
        Returns:
            Order response dict or None if failed
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
        Internal method to open LONG or SHORT position
        
        Args:
            side: "long" or "short"
            quantity: Amount of BTC
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order response dict or None if failed
        """
        try:
            # Determine Binance side
            binance_side = SIDE_BUY if side == "long" else SIDE_SELL
            side_label = side.upper()
            
            logger.info(f"[FUTURES] Opening {side_label}: {quantity:.8f} {self.symbol}")
            
            # Place MARKET order
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=binance_side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            order_id = order.get('orderId')
            avg_price = float(order.get('avgPrice', 0))
            
            logger.info(
                f"[FUTURES] ✓ {side_label} opened\n"
                f"  Order ID: {order_id}\n"
                f"  Quantity: {quantity:.8f} BTC\n"
                f"  Entry:    ${avg_price:,.2f}\n"
                f"  Status:   {order.get('status')}"
            )
            
            # Place Stop Loss (if provided)
            if stop_loss:
                try:
                    # For LONG: SL below entry (SELL to close)
                    # For SHORT: SL above entry (BUY to close)
                    sl_side = SIDE_SELL if side == "long" else SIDE_BUY
                    
                    sl_order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=sl_side,
                        type=FUTURE_ORDER_TYPE_STOP_MARKET,
                        stopPrice=stop_loss,
                        closePosition=True
                    )
                    logger.info(f"  SL Order: {sl_order.get('orderId')} @ ${stop_loss:,.2f}")
                except Exception as e:
                    logger.warning(f"[FUTURES] Failed to set SL: {e}")
            
            # Place Take Profit (if provided)
            if take_profit:
                try:
                    # For LONG: TP above entry (SELL to close)
                    # For SHORT: TP below entry (BUY to close)
                    tp_side = SIDE_SELL if side == "long" else SIDE_BUY
                    
                    tp_order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=tp_side,
                        type=FUTURE_ORDER_TYPE_LIMIT,
                        price=take_profit,
                        quantity=quantity,
                        timeInForce='GTC',
                        reduceOnly=True
                    )
                    logger.info(f"  TP Order: {tp_order.get('orderId')} @ ${take_profit:,.2f}")
                except Exception as e:
                    logger.warning(f"[FUTURES] Failed to set TP: {e}")
            
            return order
            
        except Exception as e:
            logger.error(f"[FUTURES] Failed to open {side.upper()}: {e}")
            return None

    def close_short_position(
        self, 
        quantity: float = None,
        order_id: int = None
    ) -> bool:
        """
        Close a SHORT position (buy back)
        
        Args:
            quantity: Amount to close (None = close all)
            order_id: Original order ID (for tracking)
        
        Returns:
            True if successful
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
        
        Args:
            quantity: Amount to close (None = close all)
            order_id: Original order ID (for tracking)
        
        Returns:
            True if successful
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
        Internal method to close LONG or SHORT position
        
        Args:
            side: "long" or "short"
            quantity: Amount to close (None = close all)
            order_id: Original order ID (for tracking)
        
        Returns:
            True if successful
        """
        try:
            # If no quantity specified, close entire position
            if quantity is None:
                position = self.get_position_info()
                if position:
                    pos_amt = float(position.get('positionAmt', 0))
                    
                    # Verify we have a position of the expected side
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
            
            # Determine close side (opposite of open)
            # LONG: opened with BUY, close with SELL
            # SHORT: opened with SELL, close with BUY
            close_side = SIDE_SELL if side == "long" else SIDE_BUY
            side_label = side.upper()
            
            logger.info(f"[FUTURES] Closing {side_label}: {quantity:.8f} {self.symbol}")
            
            # Place MARKET order to close
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=close_side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=quantity,
                reduceOnly=True  # Ensures this only closes position
            )
            
            logger.info(
                f"[FUTURES] ✓ {side_label} closed\n"
                f"  Order ID: {order.get('orderId')}\n"
                f"  Quantity: {quantity:.8f} BTC\n"
                f"  Exit:     ${float(order.get('avgPrice', 0)):,.2f}\n"
                f"  Status:   {order.get('status')}"
            )
            
            # Cancel any open SL/TP orders for this position
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
        
        Args:
            side: Optional filter ("long" or "short")
        
        Returns:
            Position dict or None
        """
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    pos_amt = float(pos.get('positionAmt', 0))
                    
                    # Skip if no position
                    if pos_amt == 0:
                        continue
                    
                    # Filter by side if requested
                    if side == "short" and pos_amt >= 0:
                        continue
                    elif side == "long" and pos_amt <= 0:
                        continue
                    
                    # Add side label for clarity
                    pos['side'] = "long" if pos_amt > 0 else "short"
                    return pos
            
            return None
            
        except Exception as e:
            logger.error(f"[FUTURES] Error getting position: {e}")
            return None

    def get_unrealized_pnl(self) -> float:
        """
        Get unrealized P&L for current position
        
        Returns:
            P&L in USDT
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
        
        Returns:
            Available balance in USDT
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
    
    Usage:
        from binance_futures_short import integrate_futures_into_handler
        integrate_futures_into_handler(binance_handler)
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
                
                # Calculate position size (reuse existing logic)
                from src.execution.binance_handler import PositionSizingRequest, SizingMode
                
                sizing_request = PositionSizingRequest(
                    asset=asset_name,
                    current_price=current_price,
                    signal=signal,
                    mode=kwargs.get('sizing_mode', SizingMode.AUTOMATED),
                    manual_size_usd=kwargs.get('manual_size_usd'),
                    confidence_score=kwargs.get('confidence_score'),
                    market_condition=kwargs.get('market_condition', 'neutral'),
                    override_reason=kwargs.get('override_reason'),
                )
                
                position_size_usd, _ = handler.sizer.calculate_size(sizing_request)
                quantity = position_size_usd / current_price
                quantity = round(quantity, 3)  # Futures precision
                
                # Calculate SL/TP based on side
                risk = handler.asset_config.get("risk", {})
                
                if side == "long":
                    stop_loss_pct = risk.get("stop_loss_pct", 0.05)
                    take_profit_pct = risk.get("take_profit_pct", 0.10)
                    trailing_stop_pct = risk.get("trailing_stop_pct", 0.03)
                    
                    stop_loss = current_price * (1 - stop_loss_pct)
                    take_profit = current_price * (1 + take_profit_pct)
                else:  # short
                    stop_loss_pct = risk.get("stop_loss_pct_short", risk.get("stop_loss_pct", 0.04))
                    take_profit_pct = risk.get("take_profit_pct_short", risk.get("take_profit_pct", 0.08))
                    trailing_stop_pct = risk.get("trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.025))
                    
                    stop_loss = current_price * (1 + stop_loss_pct)
                    take_profit = current_price * (1 - take_profit_pct)
                
                # Open position on Futures
                if side == "long":
                    order = handler.futures_handler.open_long_position(
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                else:
                    order = handler.futures_handler.open_short_position(
                        quantity=quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit
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
    
    Benefits of using Futures for LONG positions:
    - Leverage (multiply buying power)
    - Lower fees (0.02% vs 0.1% on Spot)
    - Better liquidity
    - Advanced order types (OCO, trailing stop)
    - Same capital can trade larger positions
    
    Usage in main.py:
        
        from src.execution.binance_futures_short import enable_futures_for_binance_handler
        
        # After initializing binance_handler
        if config["assets"]["BTC"].get("enable_futures"):
            enable_futures_for_binance_handler(binance_handler)
    
    Args:
        handler: BinanceExecutionHandler instance
    
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