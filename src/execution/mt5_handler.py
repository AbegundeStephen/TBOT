# src/execution/mt5_handler.py**

"""
MT5 Execution Handler for Gold Trading
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MT5ExecutionHandler:
    """
    Handles trade execution on MT5 (Exness)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.asset_config = config["trading"]["assets"]["gold"]
        self.risk_config = config["trading"]["risk_management"]
        self.symbol = self.asset_config["symbol"]
        self.lot_size = self.asset_config["lot_size"]
        self.magic_number = 234000  # Unique identifier for this bot

    def get_account_balance(self) -> float:
        """Get current account balance"""
        account_info = mt5.account_info()
        if account_info:
            return account_info.balance
        return 0.0

    def calculate_position_size(self) -> float:
        """Calculate position size (in lots)"""
        balance = self.get_account_balance()
        # Simple fixed lot size for now (can be enhanced)
        return self.lot_size

    def place_order(
        self,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "AI_Bot",
    ) -> Optional[mt5.OrderSendResult]:
        """
        Place order on MT5
        """
        symbol_info = mt5.symbol_info(self.symbol)

        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                return None

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return None

        logger.info(f"Order placed successfully: {result}")
        return result

    def execute_signal(self, signal: int) -> bool:
        """
        Execute trade based on signal
        """
        if signal == 0:
            return False

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Failed to get current tick")
            return False

        volume = self.calculate_position_size()

        if signal == 1:  # BUY
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price * (1 - self.risk_config["stop_loss_pct"])
            tp = price * (1 + self.risk_config["take_profit_pct"])
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price * (1 + self.risk_config["stop_loss_pct"])
            tp = price * (1 - self.risk_config["take_profit_pct"])

        result = self.place_order(
            order_type=order_type, volume=volume, price=price, sl=sl, tp=tp
        )

        return result is not None

    def get_current_price(self) -> float:
        """Get current market price"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return (tick.ask + tick.bid) / 2
        return 0.0
