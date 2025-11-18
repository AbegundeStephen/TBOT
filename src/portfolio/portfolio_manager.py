"""
Portfolio Manager - Fixed version with proper position management
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException


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

    def get_position_value(self, current_price: float) -> float:
        """Get current position value in USD"""
        return self.quantity * current_price

    def get_pnl(self, current_price: float) -> float:
        """Get current profit/loss"""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

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
                    # Add other assets (e.g., ETH, BNB) if needed

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
        # FIXED: Use current prices for accurate exposure calculation
        current_exposure = sum(
            pos.quantity * pos.entry_price  # Will be updated with current price in handler
            for pos in self.positions.values()
        )

        max_exposure_pct = self.portfolio_config["max_portfolio_exposure"]
        max_exposure_usd = self.current_capital * max_exposure_pct

        if current_exposure + new_position_usd > max_exposure_usd:
            logger.warning(
                f"Portfolio exposure limit reached: "
                f"${current_exposure + new_position_usd:,.2f} > ${max_exposure_usd:,.2f}"
            )
            return False

        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
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
        FIXED: Check if we can open a position for the given asset and side
        
        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Check if position already exists
        if asset in self.positions:
            existing_side = self.positions[asset].side
            if existing_side == side:
                return False, f"Position already open for {asset} {side.upper()}"
            else:
                return False, f"Opposite position exists for {asset} ({existing_side.upper()})"
        
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
    ) -> bool:
        """Add a new position to the portfolio"""
        # FIXED: Properly check and prevent duplicate positions
        can_open, reason = self.can_open_position(asset, side)
        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return False

        if not self.check_portfolio_limits(position_size_usd):
            logger.warning(f"Portfolio limits exceeded for {asset}")
            return False

        if self.should_reduce_position(asset):
            position_size_usd *= 0.5
            logger.info(
                f"Position size reduced to ${position_size_usd:,.2f} due to correlation"
            )

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
        )

        self.positions[asset] = position

        logger.info(
            f"✓ Position opened: {asset} {side.upper()} "
            f"@ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f} "
            f"| Qty: {quantity:.6f}"
        )

        if stop_loss:
            logger.info(f"  └─ Stop Loss: ${stop_loss:,.2f}")
        if take_profit:
            logger.info(f"  └─ Take Profit: ${take_profit:,.2f}")
        if trailing_stop_pct:
            logger.info(f"  └─ Trailing Stop: {trailing_stop_pct:.1%}")

        return True

    def close_position(
        self, asset: str, exit_price: float, reason: str = "manual"
    ) -> Optional[Dict]:
        """Close an existing position"""
        if asset not in self.positions:
            logger.warning(f"No position to close for {asset}")
            return None

        position = self.positions[asset]

        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)

        # FIXED: Update capital consistently for both modes
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
        """FIXED: Update all positions with current prices"""
        if prices:
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    if len(self.price_history[asset]) > 100:
                        self.price_history[asset].pop(0)

        # FIXED: Calculate unrealized P&L with current prices
        if prices:
            total_unrealized_pnl = sum(
                pos.get_pnl(prices.get(pos.asset, pos.entry_price))
                for pos in self.positions.values()
            )
        else:
            total_unrealized_pnl = 0.0

        if self.is_paper_mode:
            # In paper mode: equity = cash + unrealized P&L
            self.equity = self.current_capital + total_unrealized_pnl
        else:
            # In live mode: periodically refresh from exchanges
            # Don't refresh on every update to avoid rate limits
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
        FIXED: Check if we have an open position for an asset
        
        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        if asset not in self.positions:
            return False
        
        if side is None:
            return True
        
        return self.positions[asset].side == side

    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """FIXED: Get current portfolio status with real-time data"""
        # Use current prices if provided, otherwise use entry prices
        if current_prices is None:
            current_prices = {asset: pos.entry_price for asset, pos in self.positions.items()}
        
        total_exposure = sum(
            pos.get_position_value(current_prices.get(pos.asset, pos.entry_price))
            for pos in self.positions.values()
        )

        total_unrealized_pnl = sum(
            pos.get_pnl(current_prices.get(pos.asset, pos.entry_price))
            for pos in self.positions.values()
        )

        # FIXED: Calculate total portfolio value correctly
        total_value = self.current_capital + total_unrealized_pnl

        exposure_pct = (
            total_exposure / self.current_capital if self.current_capital > 0 else 0
        )
        drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )

        daily_pnl = self.equity - self.initial_capital

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
                    "pnl": pos.get_pnl(current_prices.get(asset, pos.entry_price)),
                    "pnl_pct": pos.get_pnl_pct(current_prices.get(asset, pos.entry_price)),
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
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