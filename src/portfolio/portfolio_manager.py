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
from src.execution.veteran_trade_manager import VeteranTradeManager


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
        position_id: str = None,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,
        account_balance: float = None,
        use_dynamic_management: bool = True,
    ):
        self.asset = asset
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.position_id = position_id or f"{asset}_{side}_{int(entry_time.timestamp())}"
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop_pct = trailing_stop_pct
        self.highest_price = entry_price if side == "long" else entry_price
        self.lowest_price = entry_price if side == "short" else entry_price

        # Exchange-specific tracking
        self.mt5_ticket = mt5_ticket
        self.mt5_profit = 0.0
        self.mt5_last_update = None
        self.binance_order_id = binance_order_id
        self.binance_profit = 0.0
        self.binance_last_update = None

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None
        
        self.db_trade_id = None
        self.db_manager = None

        # ✅ CRITICAL FIX: Initialize VTM
        self.trade_manager = None
        if use_dynamic_management and ohlc_data is not None and account_balance is not None:
            try:
                self.trade_manager = VeteranTradeManager(
                    entry_price=entry_price,
                    side=side,
                    asset=asset,
                    high=ohlc_data["high"],
                    low=ohlc_data["low"],
                    close=ohlc_data["close"],
                    account_balance=account_balance,
                    account_risk=0.015,
                    atr_period=14,
                    enable_early_profit_lock=True,
                    early_lock_threshold_pct=0.01,
                )

                # Use VTM-calculated stops if available
                self.stop_loss = self.trade_manager.initial_stop_loss
                self.take_profit = self.trade_manager.take_profit_levels[0] if self.trade_manager.take_profit_levels else take_profit

                logger.info(
                    f"[VTM] ✓ Initialized for {asset} {side.upper()} (Position {self.position_id})\n"
                    f"      Profile: {self.trade_manager.profile['name']}\n"
                    f"      SL: ${self.stop_loss:,.2f} | TP: ${self.take_profit:,.2f}"
                )
            except Exception as e:
                logger.error(f"[VTM] Failed to initialize: {e}", exc_info=True)
                
                self.trade_manager = None
                self.trade_manager = None
                
    def update_with_new_bar(self, high: float, low: float, close: float):
            """
            ✅ CORRECTED: Update position with new OHLC bar
            Calls VTM's update method and handles exit signals properly
            """
            if self.trade_manager:
                try:
                    old_stop = self.current_stop_loss
                    # ✅ Call VTM's update method (returns Dict or None)
                    exit_info = self.trade_manager.update_with_new_bar(
                        new_high=high,
                        new_low=low,
                        new_close=close
                    )
                    
                    # ✨ NEW: Log VTM events
                    if self.db_manager and self.db_trade_id:
                        # Log stop updates
                        if self.current_stop_loss != old_stop:
                            self.db_manager.update_trade_vtm_event(
                                trade_id=self.db_trade_id,
                                event_type='stop_updated',
                                old_value=old_stop,
                                new_value=self.current_stop_loss,
                                current_price=close
                            )
                        

                    if exit_info:
                        
                        # ✅ Extract exit reason from ExitReason enum
                        reason = exit_info['reason']
                        self.db_manager.update_trade_vtm_event(
                        trade_id=self.db_trade_id,
                        event_type=exit_info['reason'].value,
                        current_price=exit_info['price'],
                        metadata={'size': exit_info['size']}
                    )
                        
                        # Convert enum to string for compatibility
                        if isinstance(reason, reason):
                            exit_signal = reason.value
                        else:
                            exit_signal = str(reason)
                        
                        logger.info(
                            f"[VTM] {self.asset} exit triggered: {exit_signal} "
                            f"@ ${exit_info['price']:,.2f}"
                        )
                        return exit_signal
                    
                    # ✅ Update position's SL/TP with VTM's current levels
                    # (VTM may trail stops or move to break-even)
                    self.stop_loss = self.trade_manager.current_stop_loss
                    
                    return None

                except Exception as e:
                    logger.error(f"[VTM] Error updating {self.asset}: {e}", exc_info=True)
                    return None
            
            return None
        
    def update_with_current_price(self, current_price: float):
        """
        ✅ NEW: Real-time intra-bar update (for trailing stops)
        Call this more frequently than bar updates
        """
        if self.trade_manager:
            try:
                exit_info = self.trade_manager.update_with_current_price(current_price)
                
                if exit_info:
                    reason = exit_info['reason']
                    exit_signal = reason.value if isinstance(reason, reason) else str(reason)
                    
                    logger.info(
                        f"[VTM] {self.asset} real-time exit: {exit_signal} "
                        f"@ ${current_price:,.2f}"
                    )
                    return exit_signal
                
                # Update position's stop loss (may have trailed)
                self.stop_loss = self.trade_manager.current_stop_loss
                
                return None
                
            except Exception as e:
                logger.error(f"[VTM] Real-time update error: {e}")
                return None
        
        return None


    def should_close(self, current_price: float) -> Tuple[bool, str]:
            """
            ✅ CORRECTED: Check if position should close
            Prioritizes VTM exit signals over traditional SL/TP
            """
            # 1. Check VTM first (if active)
            if self.trade_manager:
                exit_info = self.trade_manager.check_exit(current_price)
                if exit_info:
                    reason = exit_info['reason']
                    exit_signal = reason.value if isinstance(reason, reason) else str(reason)
                    return True, f"vtm_{exit_signal}"

            # 2. Fallback to traditional SL/TP (if no VTM)
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

            # 3. Traditional trailing stop (only if no VTM)
            if not self.trade_manager:
                trail_stop = self.update_trailing_stop(current_price)
                if trail_stop:
                    if self.side == "long" and current_price <= trail_stop:
                        return True, "trailing_stop"
                    elif self.side == "short" and current_price >= trail_stop:
                        return True, "trailing_stop"

            return False, ""


    def get_vtm_status(self) -> Optional[Dict]:
        """Get current VTM status for monitoring"""
        if not self.trade_manager:
            return None
            
        try:
            levels = self.trade_manager.get_current_levels()
            
            current_price = levels['current_price']
            stop_loss = levels['stop_loss']
            next_target = levels['next_target']
            
            # Calculate distances
            if self.side == "long":
                distance_to_sl_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
                distance_to_tp_pct = ((next_target - current_price) / current_price * 100) if next_target else 0
            else:
                distance_to_sl_pct = ((stop_loss - current_price) / current_price) * 100 if current_price > 0 else 0
                distance_to_tp_pct = ((current_price - next_target) / current_price * 100) if next_target else 0
            
            return {
                'side': self.side,
                'entry_price': levels['entry_price'],
                'current_price': current_price,
                'pnl_pct': levels['pnl_pct'],
                'stop_loss': stop_loss,
                'take_profit': next_target if next_target else levels['all_targets'][-1] if levels['all_targets'] else None,
                'distance_to_sl_pct': distance_to_sl_pct,
                'distance_to_tp_pct': distance_to_tp_pct,
                'profit_locked': getattr(self.trade_manager, 'early_profit_locked', False),
                'bars_in_trade': levels['bars_in_trade'],
                'partials_hit': levels['partials_hit'],
                'runner_active': levels['runner_active'],
                'update_count': levels['bars_in_trade'],
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            logger.error(f"Error getting VTM status: {e}")
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
        """Update trailing stop (only used if VTM is disabled)"""
        if self.trailing_stop_pct is None:
            return None

        if self.side == "long":
            if current_price > self.highest_price:
                self.highest_price = current_price
            return self.highest_price * (1 - self.trailing_stop_pct)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            return self.lowest_price * (1 + self.trailing_stop_pct)

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

    def __init__(self, config: Dict, mt5_handler=None, binance_client=None, db_manager=None):
        self.config = config
        self.portfolio_config = config["portfolio"]
        self.max_positions_per_asset = config.get("trading", {}).get("max_positions_per_asset", 3)

        self.mt5_handler = mt5_handler
        self.binance_client = binance_client
        self.db_manager = db_manager

        self.mode = config["trading"].get("mode", "paper")
        self.is_paper_mode = self.mode.lower() == "paper"

        self.paper_capital = self.portfolio_config["initial_capital"]
        self.initial_capital = self._fetch_total_capital()
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.price_history: Dict[str, List[float]] = {"BTC": [], "GOLD": []}
        self.realized_pnl_today = 0.0

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None

        logger.info(f"Portfolio Manager initialized in {self.mode.upper()} mode")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Max Positions Per Asset: {self.max_positions_per_asset}")

        if not self.is_paper_mode:
            logger.info("✓ Using LIVE account balances from exchanges")

    def _fetch_total_capital(self) -> float:
        """Fetch total available capital from exchanges"""
        if self.is_paper_mode:
            return self.paper_capital

        total_capital = 0.0

        if self.config["assets"]["GOLD"].get("enabled", False):
            mt5_balance = self._fetch_mt5_balance()
            if mt5_balance:
                total_capital += mt5_balance

        if self.config["assets"]["BTC"].get("enabled", False):
            binance_balance = self._fetch_binance_balance()
            if binance_balance:
                total_capital += binance_balance

        return total_capital if total_capital > 0 else self.paper_capital
    
    def _fetch_mt5_balance(self) -> Optional[float]:
        """Fetch MT5 account balance"""
        try:
            import MetaTrader5 as mt5
            account_info = mt5.account_info()
            return account_info.equity if account_info else None
        except Exception as e:
            logger.error(f"[MT5] Error fetching balance: {e}")
            return None
        
    def _fetch_binance_balance(self) -> Optional[float]:
            """Fetch Binance account balance"""
            try:
                if not self.binance_client:
                    return None

                account = self.binance_client.get_account()
                total_balance = 0.0

                for balance in account["balances"]:
                    asset = balance["asset"]
                    total = float(balance["free"]) + float(balance["locked"])

                    if total > 0:
                        if asset == "USDT":
                            total_balance += total
                        elif asset == "BTC":
                            ticker = self.binance_client.get_symbol_ticker(symbol="BTCUSDT")
                            btc_price = float(ticker["price"])
                            total_balance += total * btc_price

                return total_balance
            except Exception as e:
                logger.error(f"[BINANCE] Error fetching balance: {e}")
                return None

    def refresh_capital(self) -> bool:
        """Refresh capital from exchanges"""
        if self.is_paper_mode:
            return True

        new_capital = self._fetch_total_capital()
        if new_capital > 0:
            self.current_capital = new_capital
            self.equity = new_capital
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            return True
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
                    ticker = self.binance_client.get_symbol_ticker(
                        symbol=position.symbol
                    )
                    current_price = float(ticker["price"])

                    # Calculate real-time P&L
                    if position.side == "long":
                        position.binance_profit = (
                            current_price - position.entry_price
                        ) * position.quantity
                    else:
                        position.binance_profit = (
                            position.entry_price - current_price
                        ) * position.quantity

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
                f"Portfolio exposure limit: "
                f"${current_exposure + new_position_usd:,.2f} > ${max_exposure_usd:,.2f}"
            )
            return False

        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        max_drawdown = self.portfolio_config["max_drawdown"]

        if drawdown >= max_drawdown:
            logger.warning(f"Max drawdown: {drawdown:.2%} >= {max_drawdown:.2%}")
            return False

        return True
    
    def get_asset_positions(self, asset: str, side: str = None) -> List[Position]:
        """
        Get all positions for a specific asset
        
        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            side: Optional side filter ("long" or "short")
        
        Returns:
            List of Position objects
        """
        positions = [pos for pos in self.positions.values() if pos.asset == asset]
        
        if side:
            positions = [pos for pos in positions if pos.side == side]
        
        return positions

    def get_asset_position_count(self, asset: str, side: str = None) -> int:
        """
        Count open positions for an asset
        
        Args:
            asset: Asset name
            side: Optional side filter
        
        Returns:
            Number of open positions
        """
        return len(self.get_asset_positions(asset, side))


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
        Check if we can open a new position for the asset
        
        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        current_count = self.get_asset_position_count(asset, side)
        
        if current_count >= self.max_positions_per_asset:
            return False, f"Maximum {self.max_positions_per_asset} {side.upper()} positions reached for {asset} ({current_count} open)"
        
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
        ohlc_data: dict = None,
        use_dynamic_management: bool = True,
        entry_time: datetime = None,  # ✅ NEW: Allow passing entry_time for imports
    ) -> bool:
        """Add a new position to the portfolio"""

        can_open, reason = self.can_open_position(asset, side)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return False

        if not self.check_portfolio_limits(position_size_usd):
            logger.warning(f"Portfolio limits exceeded for {asset}")
            return False

        quantity = position_size_usd / entry_price

        position = Position(
            asset=asset,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time or datetime.now(),  # ✅ Use provided or current time
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            mt5_ticket=mt5_ticket,
            binance_order_id=binance_order_id,
            ohlc_data=ohlc_data,
            account_balance=self.current_capital,
            use_dynamic_management=use_dynamic_management,
        )

        self.positions[position.position_id] = position
        
        # Database logging
        if self.db_manager:
            try:
                trade_id, is_new = self.db_manager.insert_trade_entry(
                    asset=asset,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=position.quantity,
                    position_size_usd=position_size_usd,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    position_id=position.position_id,
                    exchange=self.config["assets"][asset].get("exchange", "binance"),
                    mt5_ticket=mt5_ticket,
                    binance_order_id=binance_order_id,
                    vtm_enabled=bool(position.trade_manager),
                    metadata={
                        'trailing_stop_pct': trailing_stop_pct,
                        'entry_time': position.entry_time.isoformat()
                    }
                )
                
                position.db_trade_id = trade_id
                position.db_manager = self.db_manager
                
                if is_new:
                    print(f"New trade created: {trade_id}")
                else:
                    print(f"Existing trade updated/found: {trade_id}")
                
            except Exception as e:
                logger.error(f"[DB] Error logging trade entry: {e}")

        current_count = self.get_asset_position_count(asset, side)
        logger.info(
            f"✓ Position #{current_count} opened: {asset} {side.upper()} "
            f"@ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f} "
            f"| ID: {position.position_id}"
        )

        if mt5_ticket:
            logger.info(f"  └─ MT5 Ticket: {mt5_ticket}")
        if binance_order_id:
            logger.info(f"  └─ Binance Order ID: {binance_order_id}")
        if position.trade_manager:
            logger.info(f"  └─ VTM: ACTIVE")

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
                    high=ohlc["high"], low=ohlc["low"], close=ohlc["close"]
                )

                # If VTM signals exit, mark for closure
                if exit_signal:
                    positions_to_close.append((asset, ohlc["close"], exit_signal))
                    logger.info(
                        f"[VTM] {asset} triggered {exit_signal.upper()} @ ${ohlc['close']:,.2f}"
                    )

            except Exception as e:
                logger.error(f"[VTM] Error updating {asset}: {e}")

        # Close positions that received exit signals
        for asset, exit_price, reason in positions_to_close:
            self.close_position(
                asset=asset, exit_price=exit_price, reason=f"VTM_{reason}"
            )

        return len(positions_to_close)

    def close_position(
        self, asset: str = None, position_id: str = None, exit_price: float = None, reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Close a specific position
        
        Args:
            asset: Asset name (for backward compatibility, closes first position)
            position_id: Specific position ID to close
            exit_price: Exit price
            reason: Close reason
        """
        # Find position to close
        if position_id:
            position = self.positions.get(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found")
                return None
        elif asset:
            positions = self.get_asset_positions(asset)
            if not positions:
                logger.warning(f"No positions to close for {asset}")
                return None
            position = positions[0]  # Close first position
            position_id = position.position_id
        else:
            logger.error("Must provide either asset or position_id")
            return None

        # Calculate P&L
        if position.mt5_ticket and position.mt5_profit != 0.0:
            pnl = position.mt5_profit
        elif position.binance_order_id and position.binance_profit != 0.0:
            pnl = position.binance_profit
        else:
            pnl = position.get_pnl(exit_price)

        pnl_pct = position.get_pnl_pct(exit_price)

        self.realized_pnl_today += pnl

        if self.is_paper_mode:
            self.current_capital += pnl
            self.equity = self.current_capital
        else:
            self.refresh_capital()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        trade_result = {
            "asset": asset or position.asset,
            "position_id": position_id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "holding_time": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "reason": reason,
            "mt5_ticket": position.mt5_ticket,
            "binance_order_id": position.binance_order_id,
        }

        # ✨ NEW: Log trade exit to database
        if self.db_manager and hasattr(position, 'db_trade_id') and position.db_trade_id:
            try:
                holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                
                self.db_manager.update_trade_exit(
                    trade_id=position.db_trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_time_hours=holding_time,
                    final_quantity=position.quantity,
                    metadata={
                        'exit_time': datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"[DB] Error logging trade exit: {e}")

        self.closed_positions.append(trade_result)
        del self.positions[position_id]

        remaining_count = self.get_asset_position_count(position.asset, position.side)
        logger.info(
            f"✓ Position closed: {position.asset} {position.side.upper()} "
            f"@ ${exit_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:.2%}) "
            f"| Remaining: {remaining_count}/{self.max_positions_per_asset}"
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

    def get_position(self, asset: str, position_id: str = None) -> Optional[Position]:
            """
            Get position(s) for an asset
            
            Args:
                asset: Asset name
                position_id: Optional specific position ID
            
            Returns:
                Position object if position_id provided, otherwise first position for asset
            """
            if position_id:
                return self.positions.get(position_id)
            
            # Return first position for asset (for backward compatibility)
            positions = self.get_asset_positions(asset)
            return positions[0] if positions else None

    def has_position(self, asset: str, side: str = None) -> bool:
        """
        Check if we have any open positions for an asset
        
        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        return self.get_asset_position_count(asset, side) > 0

    def reset_daily_pnl(self):
        """Reset realized P&L tracker (call this at start of each trading day)"""
        self.realized_pnl_today = 0.0
        logger.info("Daily P&L tracker reset")

    def start_trading_session(self):
        """Start trading session"""
        self.session_start_time = datetime.now()
        self.session_start_equity = self.equity
        self.session_start_capital = self.current_capital
        self.realized_pnl_today = 0.0
        logger.info(f"Trading session started at {self.session_start_time}")
        
    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get portfolio status with accurate position counts per asset"""
        if current_prices is None:
            current_prices = {pos.asset: pos.entry_price for pos in self.positions.values()}

        total_exposure = 0.0
        total_unrealized_pnl = 0.0

        # ✅ FIXED: Count positions per asset correctly
        asset_position_counts = {}
        asset_positions_detail = {}
        
        for asset in ["BTC", "GOLD"]:
            # Get all positions for this asset
            long_positions = [p for p in self.positions.values() if p.asset == asset and p.side == "long"]
            short_positions = [p for p in self.positions.values() if p.asset == asset and p.side == "short"]
            
            asset_position_counts[asset] = {
                "long": len(long_positions),
                "short": len(short_positions),
                "total": len(long_positions) + len(short_positions)
            }
            
            # Detailed info for debugging
            asset_positions_detail[asset] = {
                "long_ids": [p.position_id for p in long_positions],
                "short_ids": [p.position_id for p in short_positions],
                "long_tickets": [p.mt5_ticket for p in long_positions if p.mt5_ticket],
                "short_tickets": [p.mt5_ticket for p in short_positions if p.mt5_ticket]
            }

        # Calculate exposures and P&L
        for pos in self.positions.values():
            current_price = current_prices.get(pos.asset, pos.entry_price)
            total_exposure += pos.quantity * current_price

            # Use exchange-reported profit if available
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                total_unrealized_pnl += pos.binance_profit
            else:
                total_unrealized_pnl += pos.get_pnl(current_price)

        total_value = self.current_capital + total_unrealized_pnl
        
        # Calculate daily P&L
        if self.session_start_equity is not None:
            current_equity = self.current_capital + total_unrealized_pnl
            daily_pnl = current_equity - self.session_start_equity
        else:
            daily_pnl = self.realized_pnl_today + total_unrealized_pnl

        return {
            "mode": self.mode,
            "total_value": total_value,
            "capital": self.current_capital,
            "equity": self.equity,
            "cash": self.current_capital,  
            "total_exposure": total_exposure,
            "open_positions": len(self.positions),
            "daily_pnl": daily_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "total_unrealized_pnl": total_unrealized_pnl,
            "asset_position_counts": asset_position_counts,
            "asset_positions_detail": asset_positions_detail,
            "max_positions_per_asset": self.max_positions_per_asset,
            "positions": {
                pos.position_id: {
                    "asset": pos.asset,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "current_price": current_prices.get(pos.asset, pos.entry_price),
                    "current_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price),
                    "pnl": pos.get_pnl(current_prices.get(pos.asset, pos.entry_price)),
                    "pnl_pct": pos.get_pnl_pct(current_prices.get(pos.asset, pos.entry_price)),
                    "stop_loss": pos.stop_loss,  
                    "take_profit": pos.take_profit,  
                    "mt5_ticket": pos.mt5_ticket,
                    "mt5_profit": pos.mt5_profit if pos.mt5_ticket else None,  
                    "binance_order_id": pos.binance_order_id,
                    "binance_profit": pos.binance_profit if pos.binance_order_id else None,  
                }
                for pos in self.positions.values()
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