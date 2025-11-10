# **src/execution/binance_handler.py**


"""
Binance Execution Handler
Manages order placement and position tracking
"""

import logging
from binance.client import Client
from binance.enums import *
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class BinanceExecutionHandler:
    """
    Handles trade execution on Binance
    """

    def __init__(self, config: Dict, client: Client):
        self.config = config
        self.client = client
        self.asset_config = config["trading"]["assets"]["btc"]
        self.risk_config = config["trading"]["risk_management"]
        self.symbol = self.asset_config["symbol"]
        self.position_size_pct = self.asset_config["position_size_pct"]
        self.open_positions = {}

    def get_account_balance(self) -> float:
        """Get current USDT balance"""
        try:
            account = self.client.get_account()
            for balance in account["balances"]:
                if balance["asset"] == "USDT":
                    return float(balance["free"])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on account equity"""
        balance = self.get_account_balance()
        position_value = balance * self.position_size_pct
        quantity = position_value / current_price

        # Round to appropriate precision (Binance requires specific lot sizes)
        quantity = round(quantity, 5)

        logger.info(f"Position size calculated: {quantity} BTC (${position_value:.2f})")
        return quantity

    def place_market_order(
        self, side: str, quantity: float, stop_loss_pct: float, take_profit_pct: float
    ) -> Optional[Dict]:
        """
        Place market order with stop-loss and take-profit
        """
        try:
            # Place main order
            order = self.client.create_order(
                symbol=self.symbol, side=side, type=ORDER_TYPE_MARKET, quantity=quantity
            )

            logger.info(f"Market order placed: {order}")

            # Get fill price
            fill_price = float(order["fills"][0]["price"])

            # Calculate stop-loss and take-profit prices
            if side == SIDE_BUY:
                stop_price = fill_price * (1 - stop_loss_pct)
                take_profit_price = fill_price * (1 + take_profit_pct)
                sl_side = SIDE_SELL
                tp_side = SIDE_SELL
            else:  # SELL
                stop_price = fill_price * (1 + stop_loss_pct)
                take_profit_price = fill_price * (1 - take_profit_pct)
                sl_side = SIDE_BUY
                tp_side = SIDE_BUY

            # Place OCO order (One-Cancels-Other) for SL and TP
            try:
                oco_order = self.client.create_oco_order(
                    symbol=self.symbol,
                    side=sl_side,
                    quantity=quantity,
                    price=round(take_profit_price, 2),
                    stopPrice=round(stop_price, 2),
                    stopLimitPrice=round(stop_price * 0.99, 2),
                    stopLimitTimeInForce=TIME_IN_FORCE_GTC,
                )
                logger.info(
                    f"OCO order placed: SL={stop_price}, TP={take_profit_price}"
                )
            except Exception as e:
                logger.warning(f"Failed to place OCO order: {e}")

            return order

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def execute_signal(self, signal: int, current_price: float) -> bool:
        """
        Execute trade based on signal
        Returns True if order was placed
        """
        if signal == 0:
            return False

        # Check if max positions reached
        if len(self.open_positions) >= self.risk_config["max_open_positions"]:
            logger.warning("Max open positions reached, skipping trade")
            return False

        quantity = self.calculate_position_size(current_price)

        if quantity <= 0:
            logger.error("Invalid position size calculated")
            return False

        side = SIDE_BUY if signal == 1 else SIDE_SELL

        order = self.place_market_order(
            side=side,
            quantity=quantity,
            stop_loss_pct=self.risk_config["stop_loss_pct"],
            take_profit_pct=self.risk_config["take_profit_pct"],
        )

        if order:
            self.open_positions[order["orderId"]] = {
                "side": side,
                "quantity": quantity,
                "price": current_price,
                "timestamp": order["transactTime"],
            }
            return True

        return False

    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0.0
