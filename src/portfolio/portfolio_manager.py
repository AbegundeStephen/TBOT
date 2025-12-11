"""
Portfolio Manager - Enhanced with MT5 real-time profit tracking
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.execution.dynamic_trade_manager import DynamicTradeManager


logger = logging.getLogger(__name__)


class Position:
    """Represents a single trading position"""

    def __init__(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,  # Track MT5 ticket number
        binance_order_id: int = None,  # Track Binance order ID
        ohlc_data: dict = None,  # {'high': np.ndarray, 'low': np.ndarray, 'close': np.ndarray}
        account_balance: float = None,
        use_dynamic_management: bool = True,
    ):
        self.asset = asset
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop_pct = trailing_stop_pct
        self.highest_price = entry_price if side == "long" else entry_price
        self.lowest_price = entry_price if side == "short" else entry_price
        
        # Exchange-specific tracking
        self.mt5_ticket = mt5_ticket
        self.mt5_profit = 0.0  # Real-time profit from MT5
        self.mt5_last_update = None
        
        self.binance_order_id = binance_order_id  # Track Binance order ID
        self.binance_profit = 0.0  # Real-time profit from Binance
        self.binance_last_update = None

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None
        
        # ✨ NEW: Dynamic Trade Manager
        self.trade_manager = None
        if use_dynamic_management and ohlc_data is not None and account_balance is not None:
            try:
                self.trade_manager = DynamicTradeManager(
                    entry_price=entry_price,
                    side=side,
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close'],
                    account_balance=account_balance,
                    account_risk=0.01,  # 1% risk per trade
                    atr_period=14,
                    reward_risk_ratio=2.0,
                    sr_window=20,
                    atr_multiplier=1.5,
                    min_profit_lock=0.005,  # Lock profit after 0.5% gain
                    aggressive_trail=False,
                )
                
                # Use DTM's calculated SL/TP
                self.stop_loss = self.trade_manager.stop_loss
                self.take_profit = self.trade_manager.take_profit
                
                logger.info(
                    f"[DTM] Initialized for {asset} {side.upper()}\n"
                    f"      SL: ${self.stop_loss:,.2f} | TP: ${self.take_profit:,.2f}"
                )
            except Exception as e:
                logger.error(f"[DTM] Failed to initialize: {e}")
                self.trade_manager = None
                
    def update_with_new_bar(self, high: float, low: float, close: float):
        """
        Update position with new OHLC data
        This should be called on every new bar/candle
        """
        if self.trade_manager:
            try:
                exit_signal = self.trade_manager.on_new_bar(high, low, close)
                
                # Update position's SL/TP with DTM's updated values
                self.stop_loss = self.trade_manager.stop_loss
                self.take_profit = self.trade_manager.take_profit
                
                return exit_signal  # Returns 'stop_loss', 'take_profit', or None
            except Exception as e:
                logger.error(f"[DTM] Error updating bar: {e}")
                return None
        return None

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed (with DTM integration)"""
        # If using DTM, check its exit signal
        if self.trade_manager:
            exit_signal = self.trade_manager.check_exit(current_price)
            if exit_signal:
                return True, exit_signal
        
        # Fallback to original logic
        if self.stop_loss:
            if self.side == "long" and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == "short" and current_price >= self.stop_loss:
                return True, "stop_loss"

        if self.take_profit:
            if self.side == "long" and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == "short" and current_price <= self.take_profit:
                return True, "take_profit"

        # Trailing stop (if no DTM)
        if not self.trade_manager:
            trail_stop = self.update_trailing_stop(current_price)
            if trail_stop:
                if self.side == "long" and current_price <= trail_stop:
                    return True, "trailing_stop"
                elif self.side == "short" and current_price >= trail_stop:
                    return True, "trailing_stop"

        return False, ""

    def get_dtm_status(self) -> dict:
        """Get Dynamic Trade Manager status"""
        if self.trade_manager:
            return self.trade_manager.get_status()
        return None

    def get_position_value(self, current_price: float) -> float:
        """Get current position value in USD"""
        return self.quantity * current_price

    def get_pnl(self, current_price: float) -> float:
        """Get current profit/loss"""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def get_mt5_pnl(self) -> float:
        """Get real-time P&L from MT5 position"""
        return self.mt5_profit
    
    def get_binance_pnl(self) -> float:
        """Get real-time P&L from Binance position"""
        return self.binance_profit
    
    def get_exchange_pnl(self) -> float:
        """Get real-time P&L from either exchange"""
        if self.mt5_ticket and self.mt5_profit != 0.0:
            return self.mt5_profit
        elif self.binance_order_id and self.binance_profit != 0.0:
            return self.binance_profit
        return 0.0

    def get_pnl_pct(self, current_price: float) -> float:
        """Get current profit/loss percentage"""
        position_value = self.entry_price * self.quantity
        return self.get_pnl(current_price) / position_value if position_value > 0 else 0

    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop based on current price"""
        if self.trailing_stop_pct is None:
            return None

        if self.side == "long":
            if current_price > self.highest_price:
                self.highest_price = current_price
            trail_stop = self.highest_price * (1 - self.trailing_stop_pct)
            return trail_stop
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            trail_stop = self.lowest_price * (1 + self.trailing_stop_pct)
            return trail_stop

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed based on stop loss/take profit"""
        if self.stop_loss:
            if self.side == "long" and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == "short" and current_price >= self.stop_loss:
                return True, "stop_loss"

        if self.take_profit:
            if self.side == "long" and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == "short" and current_price <= self.take_profit:
                return True, "take_profit"

        trail_stop = self.update_trailing_stop(current_price)
        if trail_stop:
            if self.side == "long" and current_price <= trail_stop:
                return True, "trailing_stop"
            elif self.side == "short" and current_price >= trail_stop:
                return True, "trailing_stop"

        return False, ""


class PortfolioManager:
    """
    Manages portfolio-level risk and position sizing
    Fetches actual capital from exchanges (MT5 and Binance)
    Tracks real-time MT5 profit for accurate P&L
    """

    def __init__(self, config: Dict, mt5_handler=None, binance_client=None):
        self.config = config
        self.portfolio_config = config["portfolio"]

        # Store exchange handlers for dynamic capital fetching
        self.mt5_handler = mt5_handler
        self.binance_client = binance_client

        # Trading mode
        self.mode = config["trading"].get("mode", "paper")
        self.is_paper_mode = self.mode.lower() == "paper"

        # Paper trading capital (fallback)
        self.paper_capital = self.portfolio_config["initial_capital"]

        # Fetch actual capital from exchanges
        self.initial_capital = self._fetch_total_capital()
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []

        # Historical data for correlation calculation
        self.price_history: Dict[str, List[float]] = {"BTC": [], "GOLD": []}

        # Track realized P&L for accurate daily calculations
        self.realized_pnl_today = 0.0  # Track closed position P&L

        logger.info(f"Portfolio Manager initialized in {self.mode.upper()} mode")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        if not self.is_paper_mode:
            logger.info("✓ Using LIVE account balances from exchanges")

    def _fetch_total_capital(self) -> float:
        """
        Fetch total available capital from all enabled exchanges

        Returns:
            Total capital in USD
        """
        if self.is_paper_mode:
            logger.info("[PAPER MODE] Using simulated capital from config")
            return self.paper_capital

        total_capital = 0.0

        # Fetch MT5 balance (for GOLD trading)
        if self.config["assets"]["GOLD"].get("enabled", False):
            mt5_balance = self._fetch_mt5_balance()
            if mt5_balance is not None:
                total_capital += mt5_balance
                logger.info(f"[MT5] Balance: ${mt5_balance:,.2f}")
            else:
                logger.warning("[MT5] Failed to fetch balance")

        # Fetch Binance balance (for BTC trading)
        if self.config["assets"]["BTC"].get("enabled", False):
            binance_balance = self._fetch_binance_balance()
            if binance_balance is not None:
                total_capital += binance_balance
                logger.info(f"[BINANCE] Balance: ${binance_balance:,.2f}")
            else:
                logger.warning("[BINANCE] Failed to fetch balance")

        # Fallback to paper capital if fetching failed
        if total_capital == 0.0:
            logger.error("[LIVE MODE] Failed to fetch any exchange balance!")
            logger.warning("[FALLBACK] Using paper capital from config")
            return self.paper_capital

        logger.info(f"[LIVE MODE] Total Capital: ${total_capital:,.2f}")
        return total_capital

    def _fetch_mt5_balance(self) -> Optional[float]:
        """
        Fetch account balance from MetaTrader 5

        Returns:
            Account balance in USD or None if failed
        """
        try:
            import MetaTrader5 as mt5

            # Get account info
            account_info = mt5.account_info()

            if account_info is None:
                logger.error("[MT5] Failed to get account info")
                return None

            # Get balance (in account currency, usually USD)
            balance = account_info.balance
            equity = account_info.equity
            margin = account_info.margin
            free_margin = account_info.margin_free

            logger.debug(f"[MT5] Balance: ${balance:,.2f}, Equity: ${equity:,.2f}")
            logger.debug(
                f"[MT5] Margin: ${margin:,.2f}, Free Margin: ${free_margin:,.2f}"
            )

            # Use equity (balance + floating P&L) for accurate capital
            return equity

        except Exception as e:
            logger.error(f"[MT5] Error fetching balance: {e}", exc_info=True)
            return None

    def _fetch_binance_balance(self) -> Optional[float]:
        """
        Fetch account balance from Binance with improved error handling and retries.
        Returns:
            Total balance in USDT or None if failed
        """
        try:
            if self.binance_client is None:
                logger.warning("[BINANCE] Client not initialized")
                return None

            # Test connectivity first
            try:
                self.binance_client.ping()
                logger.debug("[BINANCE] Ping successful")
            except Exception as e:
                logger.error(f"[BINANCE] Ping failed: {e}")
                return None

            # Get account information
            account = self.binance_client.get_account()
            if not account:
                logger.error("[BINANCE] Failed to get account info: Empty response")
                return None

            # Calculate total balance in USDT
            total_balance = 0.0
            for balance in account["balances"]:
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked

                if total > 0:
                    if asset == "USDT":
                        total_balance += total
                    elif asset == "BTC":
                        try:
                            ticker = self.binance_client.get_symbol_ticker(
                                symbol="BTCUSDT"
                            )
                            btc_price = float(ticker["price"])
                            total_balance += total * btc_price
                            logger.debug(
                                f"[BINANCE] BTC balance: {total} BTC = ${total * btc_price:,.2f}"
                            )
                        except Exception as e:
                            logger.error(f"[BINANCE] Failed to fetch BTC price: {e}")

            logger.info(f"[BINANCE] Total Balance: ${total_balance:,.2f} USDT")
            return total_balance

        except BinanceAPIException as e:
            logger.error(
                f"[BINANCE] API Error (code={e.status_code}): {e.message}\n"
                f"Check your API key, permissions, and IP restrictions."
            )
            return None

        except Exception as e:
            logger.error(
                f"[BINANCE] Unexpected error fetching balance: {e}", exc_info=True
            )
            return None

    def refresh_capital(self) -> bool:
        """
        Refresh current capital from exchanges
        Should be called periodically to sync with actual balances

        Returns:
            True if refresh successful, False otherwise
        """
        if self.is_paper_mode:
            logger.debug(
                "[PAPER MODE] Capital refresh skipped (using simulated capital)"
            )
            return True

        logger.info("[REFRESH] Fetching updated capital from exchanges...")

        new_capital = self._fetch_total_capital()

        if new_capital > 0:
            self.current_capital = new_capital
            self.equity = new_capital

            # Update peak equity if needed
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            logger.info(f"[REFRESH] Capital updated: ${new_capital:,.2f}")
            return True
        else:
            logger.error("[REFRESH] Failed to fetch capital")
            return False
    
    def update_mt5_positions_profit(self):
        """
        Update all MT5 positions with real-time profit from MT5
        Call this periodically to sync P&L
        """
        try:
            import MetaTrader5 as mt5
            
            # Get all open MT5 positions
            mt5_positions = mt5.positions_get()
            
            if mt5_positions is None or len(mt5_positions) == 0:
                return
            
            # Update profit for each tracked position
            for asset, position in self.positions.items():
                if position.mt5_ticket is None:
                    continue  # Skip non-MT5 positions (e.g., Binance)
                
                # Find matching MT5 position
                for mt5_pos in mt5_positions:
                    if mt5_pos.ticket == position.mt5_ticket:
                        position.mt5_profit = mt5_pos.profit
                        position.mt5_last_update = datetime.now()
                        
                        logger.debug(
                            f"[MT5] Updated {asset} profit: ${mt5_pos.profit:,.2f}"
                        )
                        break
        
        except Exception as e:
            logger.error(f"Error updating MT5 positions profit: {e}", exc_info=True)
    
    def update_binance_positions_profit(self):
        """
        Update all Binance positions with real-time profit
        Call this periodically to sync P&L
        """
        try:
            if self.binance_client is None:
                return
            
            # Get current prices for all Binance positions
            for asset, position in self.positions.items():
                if position.binance_order_id is None:
                    continue  # Skip non-Binance positions (e.g., MT5)
                
                # Get current price
                try:
                    ticker = self.binance_client.get_symbol_ticker(symbol=position.symbol)
                    current_price = float(ticker["price"])
                    
                    # Calculate real-time P&L
                    if position.side == "long":
                        position.binance_profit = (current_price - position.entry_price) * position.quantity
                    else:
                        position.binance_profit = (position.entry_price - current_price) * position.quantity
                    
                    position.binance_last_update = datetime.now()
                    
                    logger.debug(
                        f"[BINANCE] Updated {asset} profit: ${position.binance_profit:,.2f}"
                    )
                
                except Exception as e:
                    logger.debug(f"Error fetching Binance price for {asset}: {e}")
        
        except Exception as e:
            logger.error(f"Error updating Binance positions profit: {e}", exc_info=True)

    def calculate_position_size(
        self, asset: str, current_price: float, volatility: float = None
    ) -> float:
        """
        Calculate position size in USD based on portfolio rules
        Uses actual available capital from exchanges
        """
        asset_config = self.config["assets"][asset]

        # Base position size as percentage of capital
        base_size_pct = self.portfolio_config["base_position_size"]
        base_size_usd = self.current_capital * base_size_pct

        # Apply volatility scaling if enabled
        if self.portfolio_config["volatility_scaling"] and volatility:
            baseline_volatility = 0.02
            vol_scalar = min(baseline_volatility / volatility, 2.0)
            base_size_usd *= vol_scalar
            logger.debug(f"{asset} volatility scaling: {vol_scalar:.2f}x")

        # Apply asset weight
        asset_weight = asset_config.get("weight", 1.0)
        position_size = base_size_usd * asset_weight

        # Enforce minimum and maximum position sizes
        min_position = asset_config.get("min_position_usd", 100)
        max_position = asset_config.get("max_position_usd", 6000)
        position_size = max(min_position, min(position_size, max_position))

        # Check against max single asset exposure
        max_asset_exposure_pct = self.portfolio_config["max_single_asset_exposure"]
        max_asset_usd = self.current_capital * max_asset_exposure_pct
        position_size = min(position_size, max_asset_usd)

        # Check against max position size
        max_position_pct = self.portfolio_config["max_position_size"]
        max_position_usd = self.current_capital * max_position_pct
        position_size = min(position_size, max_position_usd)

        logger.info(f"{asset} calculated position size: ${position_size:,.2f}")
        return position_size

    def check_portfolio_limits(self, new_position_usd: float) -> bool:
        """Check if adding a new position would violate portfolio limits"""
        current_exposure = sum(
            pos.quantity * pos.entry_price for pos in self.positions.values()
        )

        max_exposure_pct = self.portfolio_config["max_portfolio_exposure"]
        max_exposure_usd = self.current_capital * max_exposure_pct

        if current_exposure + new_position_usd > max_exposure_usd:
            logger.warning(
                f"Portfolio exposure limit reached: "
                f"${current_exposure + new_position_usd:,.2f} > ${max_exposure_usd:,.2f}"
            )
            return False

        drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        max_drawdown = self.portfolio_config["max_drawdown"]

        if drawdown >= max_drawdown:
            logger.warning(
                f"Max drawdown reached: {drawdown:.2%} >= {max_drawdown:.2%}"
            )
            return False

        return True

    def check_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return 0.0

        min_points = 30
        if (
            len(self.price_history.get(asset1, [])) < min_points
            or len(self.price_history.get(asset2, [])) < min_points
        ):
            return 0.0

        returns1 = np.diff(np.log(self.price_history[asset1][-min_points:]))
        returns2 = np.diff(np.log(self.price_history[asset2][-min_points:]))

        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def should_reduce_position(self, new_asset: str) -> bool:
        """Check if position should be reduced due to correlation"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return False

        threshold = self.portfolio_config["correlation_threshold"]

        for existing_asset in self.positions.keys():
            if existing_asset != new_asset:
                corr = self.check_correlation(new_asset, existing_asset)
                if abs(corr) > threshold:
                    logger.warning(
                        f"High correlation detected between {new_asset} and {existing_asset}: "
                        f"{corr:.2f}"
                    )
                    return True

        return False

    def can_open_position(self, asset: str, side: str) -> Tuple[bool, str]:
        """
        Check if we can open a position for the given asset and side

        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Check if position already exists
        if asset in self.positions:
            existing_side = self.positions[asset].side
            if existing_side == side:
                return False, f"Position already open for {asset} {side.upper()}"
            else:
                return (
                    False,
                    f"Opposite position exists for {asset} ({existing_side.upper()})",
                )

        return True, "OK"

    def add_position(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,  # ✨ NEW: Pass OHLC data for DTM
        use_dynamic_management: bool = True,  # ✨ NEW: Enable/disable DTM
    ) -> bool:
        """Add a new position to the portfolio with optional dynamic management"""
        
        can_open, reason = self.can_open_position(asset, side)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return False

        if not self.check_portfolio_limits(position_size_usd):
            logger.warning(f"Portfolio limits exceeded for {asset}")
            return False

        if self.should_reduce_position(asset):
            position_size_usd *= 0.5
            logger.info(f"Position size reduced to ${position_size_usd:,.2f} due to correlation")

        quantity = position_size_usd / entry_price

        position = Position(
            asset=asset,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            mt5_ticket=mt5_ticket,
            binance_order_id=binance_order_id,
            # ✨ NEW: Pass OHLC and balance for DTM
            ohlc_data=ohlc_data,
            account_balance=self.current_capital,
            use_dynamic_management=use_dynamic_management,
        )

        self.positions[asset] = position

        logger.info(
            f"✓ Position opened: {asset} {side.upper()} "
            f"@ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f} "
            f"| Qty: {quantity:.6f}"
        )
        
        if mt5_ticket:
            logger.info(f"  └─ MT5 Ticket: {mt5_ticket}")
        if binance_order_id:
            logger.info(f"  └─ Binance Order ID: {binance_order_id}")
        if position.trade_manager:
            logger.info(f"  └─ Dynamic Trade Manager: ACTIVE")
        
        if stop_loss:
            logger.info(f"  └─ Stop Loss: ${stop_loss:,.2f}")
        if take_profit:
            logger.info(f"  └─ Take Profit: ${take_profit:,.2f}")

        return True
    
    def update_positions_with_ohlc(self, ohlc_data_dict: dict):
        """
        Update all positions with new OHLC data for dynamic management
        
        Args:
            ohlc_data_dict: Dict with asset keys and {'high': float, 'low': float, 'close': float} values
        
        Example:
            portfolio_manager.update_positions_with_ohlc({
                'BTC': {'high': 45000, 'low': 44500, 'close': 44800},
                'GOLD': {'high': 2050, 'low': 2045, 'close': 2048}
            })
        """
        positions_to_close = []
        
        for asset, position in self.positions.items():
            if asset not in ohlc_data_dict:
                continue
            
            ohlc = ohlc_data_dict[asset]
            
            try:
                # Update position with new bar
                exit_signal = position.update_with_new_bar(
                    high=ohlc['high'],
                    low=ohlc['low'],
                    close=ohlc['close']
                )
                
                # If DTM signals exit, mark for closure
                if exit_signal:
                    positions_to_close.append((asset, ohlc['close'], exit_signal))
                    logger.info(
                        f"[DTM] {asset} triggered {exit_signal.upper()} @ ${ohlc['close']:,.2f}"
                    )
            
            except Exception as e:
                logger.error(f"[DTM] Error updating {asset}: {e}")
        
        # Close positions that received exit signals
        for asset, exit_price, reason in positions_to_close:
            self.close_position(asset=asset, exit_price=exit_price, reason=f"dtm_{reason}")
        
        return len(positions_to_close)

    def close_position(
        self, asset: str, exit_price: float, reason: str = "manual"
    ) -> Optional[Dict]:
        """Close an existing position"""
        if asset not in self.positions:
            logger.warning(f"No position to close for {asset}")
            return None

        position = self.positions[asset]

        # Use exchange profit if available, otherwise calculate
        if position.mt5_ticket and position.mt5_profit != 0.0:
            pnl = position.mt5_profit
            logger.info(f"Using MT5 profit: ${pnl:,.2f}")
        elif position.binance_order_id and position.binance_profit != 0.0:
            pnl = position.binance_profit
            logger.info(f"Using Binance profit: ${pnl:,.2f}")
        else:
            pnl = position.get_pnl(exit_price)
        
        pnl_pct = position.get_pnl_pct(exit_price)

        # Track realized P&L from this closed position
        self.realized_pnl_today += pnl

        # Update capital consistently for both modes
        if self.is_paper_mode:
            self.current_capital += pnl
            self.equity = self.current_capital
        else:
            # In live mode, refresh actual capital from exchanges
            self.refresh_capital()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        trade_result = {
            "asset": asset,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "holding_time": (datetime.now() - position.entry_time).total_seconds()
            / 3600,
            "reason": reason,
            "mt5_ticket": position.mt5_ticket,
            "binance_order_id": position.binance_order_id,
        }

        self.closed_positions.append(trade_result)
        del self.positions[asset]

        logger.info(
            f"✓ Position closed: {asset} {position.side.upper()} "
            f"@ ${exit_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:.2%}) "
            f"| Reason: {reason}"
        )

        return trade_result

    def update_positions(self, prices: Dict[str, float] = None):
        """Update all positions with current prices and exchange profit"""
        # Update exchange positions with real-time profit
        if not self.is_paper_mode:
            self.update_mt5_positions_profit()
            self.update_binance_positions_profit()
        
        if prices:
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    if len(self.price_history[asset]) > 100:
                        self.price_history[asset].pop(0)

        # Calculate unrealized P&L
        total_unrealized_pnl = 0.0
        for pos in self.positions.values():
            # Prioritize exchange-reported profit
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                # Use MT5 profit for MT5 positions
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                # Use Binance profit for Binance positions
                total_unrealized_pnl += pos.binance_profit
            elif prices and pos.asset in prices:
                # Calculate for positions without exchange tracking
                total_unrealized_pnl += pos.get_pnl(prices[pos.asset])

        if self.is_paper_mode:
            # In paper mode: equity = cash + unrealized P&L
            self.equity = self.current_capital + total_unrealized_pnl
        else:
            # In live mode: periodically refresh from exchanges
            pass

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

    def get_position(self, asset: str) -> Optional[Position]:
        """Get position for a specific asset"""
        return self.positions.get(asset)

    def has_position(self, asset: str, side: str = None) -> bool:
        """
        Check if we have an open position for an asset

        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        if asset not in self.positions:
            return False

        if side is None:
            return True

        return self.positions[asset].side == side

    def reset_daily_pnl(self):
        """Reset realized P&L tracker (call this at start of each trading day)"""
        self.realized_pnl_today = 0.0
        logger.info("Daily P&L tracker reset")

    def start_trading_session(self):
        """
        Call this at the START of each trading day
        Captures the baseline equity/capital for daily P&L calculation
        """
        self.session_start_time = datetime.now()
        self.session_start_equity = self.equity
        self.session_start_capital = self.current_capital
        self.realized_pnl_today = 0.0

        logger.info(f"Trading session started at {self.session_start_time}")
        logger.info(f"Session starting equity: ${self.session_start_equity:,.2f}")

    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get current portfolio status with real-time data"""
        # Update exchange positions first
        if not self.is_paper_mode:
            self.update_mt5_positions_profit()
            self.update_binance_positions_profit()
        
        # Use current prices if provided, otherwise use entry prices
        if current_prices is None:
            current_prices = {
                asset: pos.entry_price for asset, pos in self.positions.items()
            }

        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        for pos in self.positions.values():
            current_price = current_prices.get(pos.asset, pos.entry_price)
            total_exposure += pos.get_position_value(current_price)
            
            # Use exchange profit if available
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                total_unrealized_pnl += pos.binance_profit
            else:
                total_unrealized_pnl += pos.get_pnl(current_price)

        # Calculate total portfolio value
        total_value = self.current_capital + total_unrealized_pnl

        # Calculate daily_pnl relative to session start
        if self.session_start_equity is not None:
            current_equity = self.current_capital + total_unrealized_pnl
            daily_pnl = current_equity - self.session_start_equity
        else:
            daily_pnl = self.realized_pnl_today + total_unrealized_pnl

        exposure_pct = (
            total_exposure / self.current_capital if self.current_capital > 0 else 0
        )
        drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )

        return {
            "mode": self.mode,
            "is_paper": self.is_paper_mode,
            "total_value": total_value,
            "cash": self.current_capital,
            "capital": self.current_capital,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "drawdown": drawdown,
            "open_positions": self.get_open_positions_count(),
            "total_trades": len(self.closed_positions),
            "daily_pnl": daily_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "total_unrealized_pnl": total_unrealized_pnl,
            "positions": {
                asset: {
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "current_price": current_prices.get(asset, pos.entry_price),
                    "current_value": pos.get_position_value(
                        current_prices.get(asset, pos.entry_price)
                    ),
                    "pnl": (
                        pos.mt5_profit if (pos.mt5_ticket and pos.mt5_profit != 0.0)
                        else pos.binance_profit if (pos.binance_order_id and pos.binance_profit != 0.0)
                        else pos.get_pnl(current_prices.get(asset, pos.entry_price))
                    ),
                    "pnl_pct": pos.get_pnl_pct(
                        current_prices.get(asset, pos.entry_price)
                    ),
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "mt5_ticket": pos.mt5_ticket,
                    "mt5_profit": pos.mt5_profit if pos.mt5_ticket else None,
                    "binance_order_id": pos.binance_order_id,
                    "binance_profit": pos.binance_profit if pos.binance_order_id else None,
                }
                for asset, pos in self.positions.items()
            },
        }

    def close_all_positions(self, prices: Dict[str, float] = None):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for asset in list(self.positions.keys()):
            position = self.positions[asset]
            exit_price = (
                prices.get(asset, position.entry_price)
                if prices
                else position.entry_price
            )
            self.close_position(asset, exit_price, reason="shutdown")

        logger.info("All positions closed")


# ====================================================================================
# MT5 EXECUTION HANDLER UPDATE - Pass MT5 ticket to Portfolio Manager
# ====================================================================================
"""
In your MT5ExecutionHandler._open_mt5_position() method, update the 
portfolio_manager.add_position() call to include the MT5 ticket:

OLD CODE:
    success = self.portfolio_manager.add_position(
        asset=asset,
        symbol=symbol,
        side=position_side,
        entry_price=execution_price,
        position_size_usd=actual_position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop_pct=trailing_stop_pct,
    )

NEW CODE:
    success = self.portfolio_manager.add_position(
        asset=asset,
        symbol=symbol,
        side=position_side,
        entry_price=execution_price,
        position_size_usd=actual_position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop_pct=trailing_stop_pct,
        mt5_ticket=result.order,  # ← ADD THIS LINE to track MT5 ticket
    )

Then in your main trading loop, call update_mt5_positions_profit() regularly:
    
    # Every iteration of your main loop
    portfolio_manager.update_positions(current_prices)
    
This will automatically fetch real-time profit from MT5 and include it in P&L calculations.
"""