"""
Portfolio Manager - FIXED with proper parameter handling
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Position:
    """Represents a single trading position"""
    
    def __init__(
        self,
        asset: str,
        symbol: str,
        side: str,  # 'long' or 'short'
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None  # FIXED: Changed from trailing_stop
    ):
        self.asset = asset
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop_pct = trailing_stop_pct  # FIXED: Now stores percentage
        self.highest_price = entry_price if side == 'long' else entry_price
        self.lowest_price = entry_price if side == 'short' else entry_price
        
    def get_position_value(self, current_price: float) -> float:
        """Get current position value in USD"""
        return self.quantity * current_price
    
    def get_pnl(self, current_price: float) -> float:
        """Get current profit/loss"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - current_price) * self.quantity
    
    def get_pnl_pct(self, current_price: float) -> float:
        """Get current profit/loss percentage"""
        position_value = self.entry_price * self.quantity
        return self.get_pnl(current_price) / position_value if position_value > 0 else 0
    
    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop based on current price"""
        if self.trailing_stop_pct is None:
            return None
        
        if self.side == 'long':
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
            # Calculate trailing stop
            trail_stop = self.highest_price * (1 - self.trailing_stop_pct)
            return trail_stop
        else:  # short
            # Update lowest price
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            # Calculate trailing stop
            trail_stop = self.lowest_price * (1 + self.trailing_stop_pct)
            return trail_stop
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed based on stop loss/take profit"""
        # Check stop loss
        if self.stop_loss:
            if self.side == 'long' and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == 'short' and current_price >= self.stop_loss:
                return True, "stop_loss"
        
        # Check take profit
        if self.take_profit:
            if self.side == 'long' and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == 'short' and current_price <= self.take_profit:
                return True, "take_profit"
        
        # Check trailing stop
        trail_stop = self.update_trailing_stop(current_price)
        if trail_stop:
            if self.side == 'long' and current_price <= trail_stop:
                return True, "trailing_stop"
            elif self.side == 'short' and current_price >= trail_stop:
                return True, "trailing_stop"
        
        return False, ""


class PortfolioManager:
    """
    Manages portfolio-level risk and position sizing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_config = config["portfolio"]
        
        # Portfolio state
        self.initial_capital = self.portfolio_config["initial_capital"]
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # asset_name -> Position
        self.closed_positions: List[Dict] = []
        
        # Historical data for correlation calculation
        self.price_history: Dict[str, List[float]] = {"BTC": [], "GOLD": []}
        
        logger.info(f"Portfolio Manager initialized with ${self.initial_capital:,.2f}")
    
    def calculate_position_size(
        self, 
        asset: str, 
        current_price: float,
        volatility: float = None
    ) -> float:
        """
        Calculate position size in USD based on portfolio rules
        
        Args:
            asset: Asset name (BTC, GOLD)
            current_price: Current asset price
            volatility: Asset volatility (for volatility scaling)
        
        Returns:
            Position size in USD
        """
        asset_config = self.config["assets"][asset]
        
        # Base position size as percentage of capital
        base_size_pct = self.portfolio_config["base_position_size"]
        base_size_usd = self.current_capital * base_size_pct
        
        # Apply volatility scaling if enabled
        if self.portfolio_config["volatility_scaling"] and volatility:
            # Scale down position size for high volatility
            # Assume baseline volatility of 0.02 (2%)
            baseline_volatility = 0.02
            vol_scalar = min(baseline_volatility / volatility, 2.0)  # Cap at 2x
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
        """
        Check if adding a new position would violate portfolio limits
        
        Args:
            new_position_usd: Size of new position in USD
        
        Returns:
            True if position is allowed, False otherwise
        """
        # Calculate current total exposure
        current_exposure = sum(
            pos.get_position_value(pos.entry_price) 
            for pos in self.positions.values()
        )
        
        # Check max portfolio exposure
        max_exposure_pct = self.portfolio_config["max_portfolio_exposure"]
        max_exposure_usd = self.current_capital * max_exposure_pct
        
        if current_exposure + new_position_usd > max_exposure_usd:
            logger.warning(
                f"Portfolio exposure limit reached: "
                f"${current_exposure + new_position_usd:,.2f} > ${max_exposure_usd:,.2f}"
            )
            return False
        
        # Check max drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        max_drawdown = self.portfolio_config["max_drawdown"]
        
        if drawdown >= max_drawdown:
            logger.warning(f"Max drawdown reached: {drawdown:.2%} >= {max_drawdown:.2%}")
            return False
        
        return True
    
    def check_correlation(self, asset1: str, asset2: str) -> float:
        """
        Calculate correlation between two assets
        
        Args:
            asset1: First asset name
            asset2: Second asset name
        
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if not self.portfolio_config["reduce_correlated_positions"]:
            return 0.0
        
        # Need at least 30 data points for meaningful correlation
        min_points = 30
        if (len(self.price_history.get(asset1, [])) < min_points or 
            len(self.price_history.get(asset2, [])) < min_points):
            return 0.0
        
        # Get last N returns
        returns1 = np.diff(np.log(self.price_history[asset1][-min_points:]))
        returns2 = np.diff(np.log(self.price_history[asset2][-min_points:]))
        
        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def should_reduce_position(self, new_asset: str) -> bool:
        """
        Check if position should be reduced due to correlation
        
        Args:
            new_asset: Asset we want to trade
        
        Returns:
            True if position should be reduced
        """
        if not self.portfolio_config["reduce_correlated_positions"]:
            return False
        
        threshold = self.portfolio_config["correlation_threshold"]
        
        # Check correlation with existing positions
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
    
    def add_position(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        position_size_usd: float,  # FIXED: Added this parameter
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None  # FIXED: Changed name
    ) -> bool:
        """
        Add a new position to the portfolio
        
        Returns:
            True if position was added successfully
        """
        # Check if we already have a position in this asset
        if asset in self.positions:
            logger.warning(f"Position already exists for {asset}")
            return False
        
        # Check portfolio limits
        if not self.check_portfolio_limits(position_size_usd):
            return False
        
        # Check correlation
        if self.should_reduce_position(asset):
            # Reduce position size by 50% due to correlation
            position_size_usd *= 0.5
            logger.info(f"Position size reduced to ${position_size_usd:,.2f} due to correlation")
        
        # Calculate quantity
        quantity = position_size_usd / entry_price
        
        # Create position
        position = Position(
            asset=asset,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct  # FIXED: Changed parameter name
        )
        
        self.positions[asset] = position
        
        logger.info(
            f"Position opened: {asset} {side.upper()} "
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
        self, 
        asset: str, 
        exit_price: float,
        reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Close an existing position
        
        Returns:
            Trade result dictionary or None
        """
        if asset not in self.positions:
            logger.warning(f"No position to close for {asset}")
            return None
        
        position = self.positions[asset]
        
        # Calculate P&L
        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)
        
        # Update capital
        self.current_capital += pnl
        self.equity = self.current_capital
        
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Record closed position
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
            "holding_time": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "reason": reason
        }
        
        self.closed_positions.append(trade_result)
        
        # Remove from active positions
        del self.positions[asset]
        
        logger.info(
            f"Position closed: {asset} {position.side.upper()} "
            f"@ ${exit_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:.2%}) "
            f"| Reason: {reason}"
        )
        
        return trade_result
    
    def update_positions(self, prices: Dict[str, float] = None):
        """
        Update all positions with current prices
        
        Args:
            prices: Dictionary of asset -> current_price
        """
        if prices:
            # Update price history for correlation calculation
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    # Keep last 100 prices
                    if len(self.price_history[asset]) > 100:
                        self.price_history[asset].pop(0)
        
        # Calculate current equity
        total_pnl = sum(
            pos.get_pnl(prices.get(pos.asset, pos.entry_price))
            for pos in self.positions.values()
            if prices
        )
        
        self.equity = self.current_capital + total_pnl
        
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)
    
    def get_position(self, asset: str) -> Optional[Position]:
        """Get position for a specific asset"""
        return self.positions.get(asset)
    
    def has_position(self, asset: str) -> bool:
        """Check if we have an open position for an asset"""
        return asset in self.positions
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        total_exposure = sum(
            pos.get_position_value(pos.entry_price)
            for pos in self.positions.values()
        )
        
        # FIXED: Calculate total value properly
        total_value = self.current_capital + sum(
            pos.get_pnl(pos.entry_price) for pos in self.positions.values()
        )
        
        exposure_pct = total_exposure / self.current_capital if self.current_capital > 0 else 0
        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Calculate daily P&L (simplified - you may want to track this more accurately)
        daily_pnl = self.equity - self.initial_capital
        
        return {
            "total_value": total_value,  # FIXED: Added this
            "cash": self.current_capital,  # FIXED: Added this
            "capital": self.current_capital,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "drawdown": drawdown,
            "open_positions": self.get_open_positions_count(),
            "total_trades": len(self.closed_positions),
            "daily_pnl": daily_pnl,  # FIXED: Added this
            "positions": {
                asset: {
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "current_value": pos.get_position_value(pos.entry_price),
                    "pnl": pos.get_pnl(pos.entry_price),
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit
                }
                for asset, pos in self.positions.items()
            }
        }
    
    def close_all_positions(self, prices: Dict[str, float] = None):
        """Close all open positions"""
        logger.info("Closing all positions...")
        
        for asset in list(self.positions.keys()):
            position = self.positions[asset]
            exit_price = prices.get(asset, position.entry_price) if prices else position.entry_price
            self.close_position(asset, exit_price, reason="shutdown")
        
        logger.info("All positions closed")