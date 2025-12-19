"""
Binance Execution Handler with Hybrid Position Sizing + Order Tracking
ENHANCED: Track Binance order IDs for accurate P&L tracking
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
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class SizingMode:
    """Position sizing modes"""

    AUTOMATED = "automated"
    MANUAL_OVERRIDE = "override"
    REDUCED_RISK = "reduced_risk"
    ELEVATED_RISK = "elevated"
    
    def count_binance_positions(client: Client, asset: str = "BTC") -> Tuple[int, float]:
        """
        Count positions by checking actual Binance balance
        
        Args:
            client: Binance client
            asset: Asset symbol (e.g., "BTC")
        
        Returns:
            Tuple of (position_count_estimate, total_quantity)
            Note: Binance spot doesn't track "positions" - we estimate from balance
        """
        try:
            account = client.get_account()
            
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    total_qty = float(balance["free"]) + float(balance["locked"])
                    
                    # Estimate position count (assume we use similar position sizes)
                    # This is approximate since Binance spot doesn't have discrete positions
                    MIN_BTC_PER_POSITION = 0.0001  # Adjust based on your typical position size
                    
                    if total_qty < MIN_BTC_PER_POSITION:
                        return 0, 0.0
                    
                    # Rough estimate of positions (not perfect, but helps)
                    estimated_positions = int(total_qty / MIN_BTC_PER_POSITION)
                    
                    return estimated_positions, total_qty
            
            return 0, 0.0
            
        except Exception as e:
            logger.error(f"Error counting Binance positions: {e}")
            return 0, 0.0



class PositionSizingRequest:
    """Request object for position sizing with manual override support"""

    def __init__(
        self,
        asset: str,
        current_price: float,
        signal: int,
        mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        confidence_score: float = None,
        market_condition: str = None,
        override_reason: str = None,
        max_override_pct: float = 2.0,
    ):
        self.asset = asset
        self.current_price = current_price
        self.signal = signal
        self.mode = mode
        self.manual_size_usd = manual_size_usd
        self.confidence_score = confidence_score or 0.5
        self.market_condition = market_condition or "neutral"
        self.override_reason = override_reason
        self.max_override_pct = max_override_pct


class HybridPositionSizer:
    """Hybrid position sizing with automated rules and manual overrides"""

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []
        logger.info("HybridPositionSizer initialized")

    def calculate_size(self, request: PositionSizingRequest) -> Tuple[float, Dict]:
        """Calculate position size with hybrid logic"""
        try:
            base_size = self._calculate_automated_size(
                request.asset, request.current_price, request.signal
            )

            confidence_adjusted = self._apply_confidence_adjustment(
                base_size, request.confidence_score, request.market_condition
            )

            if request.mode == SizingMode.MANUAL_OVERRIDE and request.manual_size_usd:
                final_size, override_result = self._apply_manual_override(
                    base_size,
                    confidence_adjusted,
                    request.manual_size_usd,
                    request.override_reason,
                    request.max_override_pct,
                )
            elif request.mode == SizingMode.REDUCED_RISK:
                final_size = confidence_adjusted * 0.5
                override_result = {
                    "mode": "REDUCED_RISK",
                    "reason": "Lower exposure due to uncertain market conditions",
                    "reduction_pct": 50,
                }
            elif request.mode == SizingMode.ELEVATED_RISK:
                final_size = min(
                    confidence_adjusted * 1.5,
                    self._get_max_position_size(request.asset),
                )
                override_result = {
                    "mode": "ELEVATED_RISK",
                    "reason": "Higher exposure for high-conviction trade",
                    "elevation_pct": 50,
                }
            else:
                final_size = confidence_adjusted
                override_result = {"mode": "AUTOMATED"}

            final_size = self._apply_hard_limits(
                request.asset, final_size, request.signal
            )

            metadata = {
                "asset": request.asset,
                "mode": request.mode,
                "signal": request.signal,
                "confidence_score": request.confidence_score,
                "market_condition": request.market_condition,
                "base_size_usd": base_size,
                "confidence_adjusted_usd": confidence_adjusted,
                "final_size_usd": final_size,
                "override_details": override_result,
            }

            self._log_sizing_decision(metadata)

            if request.mode != SizingMode.AUTOMATED:
                self.override_history.append(metadata)

            return base_size, {"mode": "AUTOMATED"}

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0, {"error": str(e)}

    def _calculate_automated_size(self, asset: str, price: float, signal: int) -> float:
        """Calculate base size (existing implementation)"""
        asset_cfg = self.config["assets"][asset]
        base_pct = self.portfolio_cfg.get("base_position_size", 0.10)
        base_size = self.portfolio_manager.current_capital * base_pct
        return base_size

    def _apply_confidence_adjustment(
        self, base_size: float, confidence_score: float, market_condition: str
    ) -> float:
        """Adjust size based on signal confidence and market conditions"""

        confidence_scalar = 0.5 + (confidence_score * 1.0)
        confidence_scalar = max(0.3, min(1.5, confidence_scalar))
        adjusted_size = base_size * confidence_scalar

        condition_scalars = {
            "bullish": 1.1,
            "neutral": 1.0,
            "bearish": 0.8,
            "uncertain": 0.6,
            "extreme_volatility": 0.5,
        }
        condition_scalar = condition_scalars.get(market_condition, 1.0)
        adjusted_size *= condition_scalar

        logger.debug(
            f"Confidence adjustment: ${base_size:.2f} → ${adjusted_size:.2f} "
            f"(confidence={confidence_score:.2f}, condition={market_condition})"
        )

        return adjusted_size

    def _apply_manual_override(
        self,
        base_size: float,
        confidence_adjusted: float,
        manual_size_usd: float,
        override_reason: str,
        max_override_pct: float,
    ) -> Tuple[float, Dict]:
        """Apply manual override with safety guards"""

        min_allowed = confidence_adjusted * (1 - max_override_pct / 100)
        max_allowed = confidence_adjusted * (1 + max_override_pct / 100)

        if manual_size_usd < min_allowed or manual_size_usd > max_allowed:
            logger.warning(
                f"Manual override ${manual_size_usd:,.2f} exceeds bounds "
                f"[${min_allowed:,.2f}, ${max_allowed:,.2f}]. Clamping."
            )
            manual_size_usd = max(min_allowed, min(manual_size_usd, max_allowed))

        deviation_pct = (
            ((manual_size_usd - confidence_adjusted) / confidence_adjusted * 100)
            if confidence_adjusted > 0
            else 0
        )

        result = {
            "mode": "MANUAL_OVERRIDE",
            "reason": override_reason or "User override",
            "manual_size": manual_size_usd,
            "deviation_pct": deviation_pct,
        }

        logger.info(
            f"Manual override applied: ${confidence_adjusted:,.2f} → ${manual_size_usd:,.2f} "
            f"({deviation_pct:+.1f}%) - Reason: {override_reason}"
        )

        return manual_size_usd, result

    def _apply_hard_limits(
        self, asset: str, position_size: float, signal: int
    ) -> float:
        """Apply absolute limits to prevent excessive exposure"""

        asset_cfg = self.config["assets"][asset]

        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
        max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)

        if position_size < min_size:
            logger.debug(
                f"Position size ${position_size:,.2f} below minimum ${min_size}"
            )
            return 0.0

        position_size = min(position_size, max_size)

        current_exposure = self._calculate_current_exposure()
        max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
        if current_exposure + position_size > max_portfolio_usd:
            position_size = max(0, max_portfolio_usd - current_exposure)
            logger.warning(
                f"Position clamped to portfolio limit: ${position_size:,.2f}"
            )

        max_asset_usd = self.portfolio_manager.current_capital * max_single_asset
        position_size = min(position_size, max_asset_usd)

        return position_size

    def _calculate_current_exposure(self) -> float:
        """Calculate total current portfolio exposure"""
        return sum(
            pos.quantity * pos.entry_price
            for pos in self.portfolio_manager.positions.values()
        )

    def _get_max_position_size(self, asset: str) -> float:
        """Get maximum allowed position size for an asset"""
        asset_cfg = self.config["assets"][asset]
        max_usd = asset_cfg.get("max_position_usd", 6000)
        max_asset_pct = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
        max_asset_usd = self.portfolio_manager.current_capital * max_asset_pct
        return min(max_usd, max_asset_usd)

    def _log_sizing_decision(self, metadata: Dict):
        """Log sizing decision for audit"""
        logger.info(
            f"[SIZING] {metadata['asset']} | Mode: {metadata['mode']} | "
            f"Confidence: {metadata['confidence_score']:.2f} | "
            f"Size: ${metadata['final_size_usd']:,.2f}"
        )


class BinanceExecutionHandler:
    """
    Binance Execution Handler with Hybrid Position Sizing + Order Tracking
    """

    def __init__(
        self, config: Dict, client: Client, portfolio_manager, data_manager=None
    ):
        self.config = config
        self.client = client
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)

        self.asset_config = config["assets"]["BTC"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]

        self.symbol = self.asset_config["symbol"]
        self.mode = self.trading_config.get("mode", "paper")
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )

        logger.info(
            f"BinanceExecutionHandler with HybridPositionSizer initialized - Mode: {self.mode.upper()}"
        )

        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")
            
    
    def can_open_binance_position(self, asset: str = "BTC", side: str = "long") -> Tuple[bool, str]:
        """
        Check if we can open a new position
        Checks BOTH portfolio manager AND actual Binance balance
        
        Returns:
            (can_open: bool, reason: str)
        """
        # Check 1: Portfolio manager limits
        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(asset, side)
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"
        
        # Check 2: Actual Binance balance (CRITICAL CHECK)
        # Note: For Binance spot, we estimate positions from balance
        estimated_positions, total_qty = count_binance_positions(self.client, "BTC")
        
        # Get portfolio count for comparison
        portfolio_count = self.portfolio_manager.get_asset_position_count(asset, side)
        
        # If Binance shows significantly more BTC than our portfolio tracks
        if estimated_positions > portfolio_count:
            logger.warning(
                f"[BINANCE] Balance shows ~{estimated_positions} positions "
                f"but portfolio has {portfolio_count}. Possible external holdings."
            )
        
        # Check if opening another position would exceed limits
        # We use portfolio count as source of truth (Binance balance is continuous)
        if portfolio_count >= self.max_positions_per_asset:
            logger.warning(
                f"[BINANCE] Cannot open {side.upper()} position: "
                f"{portfolio_count}/{self.max_positions_per_asset} positions already tracked"
            )
            return False, f"Already have {portfolio_count}/{self.max_positions_per_asset} {side.upper()} positions"
        
        return True, f"OK - {portfolio_count}/{self.max_positions_per_asset} positions open"
    
    
    
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
        self,
        signal: int,
        current_price: float = None,
        asset_name: str = "BTC",
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> bool:
        """
        Execute trading signal with FIXED logic:
        1. Checks actual Binance balance before opening
        2. Closes ALL positions on opposite signals
        """
        try:
            # Get current price if not provided
            if current_price is None:
                current_price = self.get_current_price()
            
            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid price: {current_price}")
                return False
            
            # Get existing positions from portfolio
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)
            
            # === SCENARIO 1: SELL SIGNAL - Close ALL long positions ===
            if signal == -1:
                long_positions = [p for p in existing_positions if p.side == "long"]
                
                if long_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📉 SELL SIGNAL - Closing {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )
                    
                    closed_count = 0
                    failed_count = 0
                    
                    for i, position in enumerate(long_positions, 1):
                        logger.info(f"\n[{i}/{len(long_positions)}] Closing position {position.position_id}...")
                        
                        success = self._close_position(
                            position=position,
                            current_price=current_price,
                            asset_name=asset_name,
                            reason="sell_signal"
                        )
                        
                        if success:
                            closed_count += 1
                            logger.info(f"  ✓ Position {position.position_id} closed successfully")
                        else:
                            failed_count += 1
                            logger.error(f"  ✗ Failed to close position {position.position_id}")
                    
                    logger.info(
                        f"\n{'='*80}\n"
                        f"CLOSE SUMMARY: {closed_count} closed, {failed_count} failed\n"
                        f"{'='*80}\n"
                    )
                    
                    return closed_count > 0
                
                else:
                    logger.debug(f"{asset_name}: SELL signal but no LONG positions to close")
                    return False
            
            # === SCENARIO 2: BUY SIGNAL - Close shorts (if any) then open long ===
            elif signal == 1:
                short_positions = [p for p in existing_positions if p.side == "short"]
                
                if short_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Closing {len(short_positions)} SHORT position(s) first\n"
                        f"{'='*80}"
                    )
                    
                    closed_count = 0
                    for i, position in enumerate(short_positions, 1):
                        logger.info(f"\n[{i}/{len(short_positions)}] Closing SHORT position {position.position_id}...")
                        
                        success = self._close_position(
                            position=position,
                            current_price=current_price,
                            asset_name=asset_name,
                            reason="buy_signal"
                        )
                        
                        if success:
                            closed_count += 1
                    
                    logger.info(f"  ✓ Closed {closed_count}/{len(short_positions)} SHORT positions")
                
                # Now check if we can open a new LONG position
                # 🔥 CRITICAL FIX: Check actual Binance balance
                can_open, reason = self.can_open_binance_position(asset_name, "long")
                
                if not can_open:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"⚠️  CANNOT OPEN NEW LONG POSITION\n"
                        f"Reason: {reason}\n"
                        f"{'='*80}\n"
                    )
                    return False
                
                logger.info(
                    f"\n{'='*80}\n"
                    f"📈 BUY SIGNAL - Opening new LONG position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )
                
                return self._open_position(
                    signal=signal,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                )
            
            # === SCENARIO 3: HOLD SIGNAL - Check SL/TP for all positions ===
            elif signal == 0:
                if not existing_positions:
                    logger.debug(f"{asset_name}: HOLD signal, no positions")
                    return False
                
                positions_closed = False
                
                for position in existing_positions:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        position, current_price
                    )
                    
                    if should_close:
                        logger.info(
                            f"[AUTO-CLOSE] {asset_name} {position.position_id}: {close_reason}"
                        )
                        success = self._close_position(
                            position, current_price, asset_name, close_reason
                        )
                        if success:
                            positions_closed = True
                
                if not positions_closed:
                    logger.debug(f"{asset_name}: All positions holding")
                
                return positions_closed
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False

    def _check_stop_loss_take_profit(
        self, 
        position, 
        current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit"""
        try:
            entry_price = position.entry_price
            stop_loss = position.stop_loss
            take_profit = position.take_profit
            side = position.side
            
            price_tolerance = 0.50
            
            if side == "long":
                if stop_loss and current_price <= (stop_loss + price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f}, {pnl_pct:+.2f}%)"
                
                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f}, {pnl_pct:+.2f}%)"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}")
            return False, ""


    def _close_position(
        self, 
        position, 
        current_price: float, 
        asset_name: str, 
        reason: str
    ) -> bool:
        """
        Close a single position
        Returns True if successful, False otherwise
        """
        try:
            entry_price = position.entry_price
            quantity = position.quantity
            side = position.side
            position_id = position.position_id
            binance_order_id = position.binance_order_id
            
            position_size_usd = quantity * entry_price
            
            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0
            
            logger.info(
                f"[CLOSE] {asset_name} {side.upper()} ({position_id})\n"
                f"  Entry: ${entry_price:,.2f} → Exit: ${current_price:,.2f}\n"
                f"  Quantity: {quantity:.8f} BTC\n"
                f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
                f"  Reason: {reason}"
            )
            
            # Close on Binance first (if live mode)
            binance_closed = False
            if self.mode.lower() != "paper":
                binance_closed = self._close_binance_order(
                    quantity=quantity,
                    asset_name=asset_name,
                    order_id=binance_order_id
                )
            else:
                binance_closed = True  # Paper mode
            
            # Close in portfolio manager
            trade_result = self.portfolio_manager.close_position(
                position_id=position_id,
                exit_price=current_price,
                reason=reason
            )
            
            if trade_result and binance_closed:
                logger.info(f"  ✓ Position {position_id} closed successfully")
                return True
            else:
                logger.error(f"  ✗ Failed to close position {position_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False
    
    def _close_binance_order(
        self, 
        quantity: float, 
        asset_name: str, 
        order_id: int = None
    ) -> bool:
        """
        Close position by selling on Binance
        
        Args:
            quantity: Amount of BTC to sell
            asset_name: Asset name for logging
            order_id: Optional order ID for tracking
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[BINANCE] Selling {quantity:.8f} BTC...")
            
            # Round quantity to 8 decimals (BTC precision)
            quantity = round(quantity, 8)
            
            # Place market sell order
            order = self.client.order_market_sell(
                symbol=self.symbol,
                quantity=quantity
            )
            
            if order and order.get('status') in ['FILLED', 'PARTIALLY_FILLED']:
                executed_qty = float(order.get('executedQty', 0))
                fills = order.get('fills', [])
                
                avg_price = 0.0
                if fills:
                    total_value = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    avg_price = total_value / total_qty if total_qty > 0 else 0
                
                logger.info(
                    f"[BINANCE] ✓ Sold {executed_qty:.8f} BTC @ ${avg_price:,.2f}\n"
                    f"  Order ID: {order.get('orderId')}\n"
                    f"  Status: {order.get('status')}"
                )
                return True
            else:
                logger.error(
                    f"[BINANCE] ✗ Sell order failed or pending\n"
                    f"  Status: {order.get('status') if order else 'No response'}"
                )
                return False
                
        except Exception as e:
            logger.error(f"[BINANCE] Error selling {asset_name}: {e}")
            return False
    

    def _open_position(
        self,
        signal: int,
        current_price: float,
        asset_name: str,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> bool:
        """Open new position with hybrid sizing"""
        try:
            side = "long" if signal == 1 else "short"
            
            # Position sizing
            from binance_handler import PositionSizingRequest  # Adjust import
            sizing_request = PositionSizingRequest(
                asset=asset_name,
                current_price=current_price,
                signal=signal,
                mode=sizing_mode,
                manual_size_usd=manual_size_usd,
                confidence_score=confidence_score,
                market_condition=market_condition or "neutral",
                override_reason=override_reason,
                max_override_pct=2.0,
            )
            
            position_size_usd, sizing_metadata = self.sizer.calculate_size(sizing_request)
            
            if position_size_usd <= 0:
                logger.warning(f"{asset_name}: Invalid size: ${position_size_usd:.2f}")
                return False
            
            quantity = position_size_usd / current_price
            quantity = round(quantity, 8)  # BTC precision
            
            MIN_BTC = 0.00001
            if quantity < MIN_BTC:
                logger.warning(f"{asset_name}: Quantity {quantity:.8f} below minimum")
                return False
            
            # Calculate SL/TP
            risk = self.asset_config.get("risk", {})
            stop_loss_pct = risk.get("stop_loss_pct", 0.05)
            take_profit_pct = risk.get("take_profit_pct", 0.10)
            trailing_stop_pct = risk.get("trailing_stop_pct", 0.03)
            
            if signal == 1:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            
            logger.info(
                f"[OPEN] BUY {quantity:.8f} {self.symbol} @ ${current_price:,.2f}\n"
                f"  Size: ${position_size_usd:,.2f}\n"
                f"  Mode: {sizing_mode} | Confidence: {confidence_score}\n"
                f"  SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})\n"
                f"  TP: ${take_profit:,.2f} ({take_profit_pct:.1%})"
            )
            
            # Execute on Binance (if live mode)
            order_id = None
            if self.mode.lower() != "paper":
                try:
                    order = self.client.order_market_buy(
                        symbol=self.symbol,
                        quantity=quantity
                    )
                    order_id = order.get('orderId')
                    logger.info(f"[BINANCE] ✓ Order placed: ID={order_id}")
                except Exception as e:
                    logger.error(f"[BINANCE] Order failed: {e}")
                    return False
            
            # Get OHLC for VTM
            ohlc_data = None
            if self.data_manager:
                try:
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)
                    
                    df = self.data_manager.fetch_binance_data(
                        symbol=self.symbol,
                        interval=self.asset_config.get("interval", "1h"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    
                    if len(df) > 0:
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                        }
                        logger.info(f"[VTM] Prepared {len(df)} bars")
                except Exception as e:
                    logger.warning(f"[VTM] OHLC fetch failed: {e}")
            
            # Add to portfolio
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=None,
                take_profit=None,
                trailing_stop_pct=trailing_stop_pct,
                binance_order_id=order_id,
                ohlc_data=ohlc_data,
                account_balance=self.portfolio_manager.current_capital,
                use_dynamic_management=True,
            )
            
            if success:
                logger.info(f"[OK] {asset_name} position opened successfully")
                if order_id:
                    logger.info(f"  └─ Binance Order ID: {order_id}")
                if ohlc_data:
                    logger.info(f"  └─ VTM: ACTIVE")
                return True
            else:
                logger.error(f"[FAIL] Portfolio rejected position")
                # If live mode, we should reverse the Binance order here
                return False
                
        except Exception as e:
            logger.error(f"Error opening position: {e}", exc_info=True)
            return False

    def check_and_update_positions_VTM(self, asset_name: str = "BTC"):
        """Check and update ALL positions for an asset with VTM"""
        try:
            # Get ALL positions for this asset
            positions = self.portfolio_manager.get_asset_positions(asset_name)
            
            if not positions:
                return False

            current_price = self.get_current_price()
            if current_price is None:
                logger.warning(f"Could not get price for {asset_name}")
                return False

            positions_closed = False

            # Check each position individually
            for position in positions:
                # Update VTM with current price (intra-bar trailing)
                if position.trade_manager:
                    exit_signal = position.trade_manager.update_with_current_price(current_price)

                    if exit_signal:
                        logger.info(
                            f"[VTM] {asset_name} {position.position_id} triggered "
                            f"{exit_signal.upper()} @ ${current_price:,.2f}"
                        )
                        self._close_position(
                            position, current_price, asset_name, f"VTM_{exit_signal}"
                        )
                        positions_closed = True
                        continue

                # Fallback: check traditional SL/TP
                should_close, reason = self._check_stop_loss_take_profit(position, current_price)
                if should_close:
                    logger.info(f"[AUTO-CLOSE] {asset_name} {position.position_id}: {reason}")
                    self._close_position(position, current_price, asset_name, reason)
                    positions_closed = True

            return positions_closed

        except Exception as e:
            logger.error(f"Error checking VTM positions: {e}", exc_info=True)
            return False


    def check_and_update_positions(self, asset_name: str = "BTC"):
        """
        Actively check and update all positions
        NOW WITH VTM SUPPORT
        """
        try:
            # Use VTM version if available
            if hasattr(self, "check_and_update_positions_VTM"):
                return self.check_and_update_positions_VTM(asset_name)

            # Fallback to original implementation
            position = self.portfolio_manager.get_position(asset_name)
            if not position:
                return

            current_price = self.get_current_price()

            if current_price is None:
                logger.warning(f"Could not get price for {asset_name}")
                return

            should_close, reason = self._check_stop_loss_take_profit(
                position, current_price
            )
            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset_name}: {reason}")
                self._close_position(position, current_price, asset_name, reason)

        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)

    def sync_positions_with_binance(self, asset_name: str = "BTC", symbol: str = None) -> bool:
        """Sync portfolio with actual Binance holdings (multi-position aware)"""
        if symbol is None:
            symbol = self.symbol
        try:
            logger.info(f"[SYNC] Starting position sync for {asset_name}...")
            account = self.client.get_account()
            
            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)

            # Get Binance balance
            btc_balance = 0.0
            for balance in account["balances"]:
                if balance["asset"] == "BTC":
                    btc_balance = float(balance["free"]) + float(balance["locked"])

            current_price = self.get_current_price(symbol)
            MIN_BTC_BALANCE = 0.0001

            # Calculate total portfolio quantity for this asset
            portfolio_total_qty = sum(pos.quantity for pos in portfolio_positions)
            # ================================================================
        # SCENARIO 1: Binance has BTC, portfolio is empty → IMPORT WITH VTM
        # ================================================================
        
            if btc_balance > MIN_BTC_BALANCE and not portfolio_positions:
                import_enabled = bool(
                self.config.get("portfolio", {}).get("import_existing_positions", False)
            )
                if import_enabled:
                    logger.info(
                        f"[SYNC] BTC balance {btc_balance:.8f} detected.\n"
                        f"  → Importing as LONG position WITH VTM support..."
                    )
                    # ✅ CRITICAL: Fetch OHLC data for VTM
                    ohlc_data = None
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = self.data_manager.fetch_binance_data(
                            symbol=symbol,
                            interval=self.config["assets"][asset_name].get("interval", "1h"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                            logger.info(f"[VTM] Fetched {len(df)} bars for dynamic management")
                        else:
                            logger.warning(f"[VTM] Insufficient data ({len(df)} bars), VTM disabled")

                    except Exception as e:
                        logger.error(f"[VTM] Failed to fetch OHLC: {e}")
                        ohlc_data = None

                    position_size_usd = btc_balance * current_price

                    # ✅ Import with VTM support
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side="long",
                        entry_price=current_price,
                        position_size_usd=position_size_usd,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=self.config["assets"][asset_name].get("risk", {}).get("trailing_stop_pct"),
                        binance_order_id=None,
                        ohlc_data=ohlc_data,  # ✅ Pass OHLC for VTM
                        use_dynamic_management=True,  # ✅ Enable VTM
                        entry_time=datetime.now(),  # Use current time for external holdings
                    )

                    if success:
                        logger.info(f"[SYNC] ✓ Imported {btc_balance:.8f} BTC as LONG position")
                        
                        # Verify VTM status
                        imported_pos = self.portfolio_manager.get_position(asset_name)
                        if imported_pos and imported_pos.trade_manager:
                            logger.info(
                                f"[VTM] ✓ VTM ACTIVE for imported position\n"
                                f"      Entry: ${imported_pos.entry_price:,.2f}\n"
                                f"      SL: ${imported_pos.stop_loss:,.2f}\n"
                                f"      TP: ${imported_pos.take_profit:,.2f}"
                            )
                        else:
                            logger.warning(f"[VTM] ⚠️ VTM not initialized for imported position")
                        
                        return True
                    else:
                        logger.error(f"[SYNC] Failed to import BTC position")
                        return False
                else:
                    logger.info(
                        f"[SYNC] BTC balance {btc_balance:.8f} detected but auto-import disabled.\n"
                        f"  → Bot will open new positions on BUY signals."
                    )
                    return True

            # ================================================================
            # SCENARIO 2: Portfolio has positions, Binance is empty → CLOSE ALL
            # ================================================================
            if portfolio_positions and btc_balance <= MIN_BTC_BALANCE:
                logger.warning(
                    f"[SYNC] Portfolio shows {len(portfolio_positions)} position(s) "
                    f"but Binance balance is {btc_balance:.8f} BTC\n"
                    f" → Removing all positions (likely sold manually)."
                )
                for position in portfolio_positions:
                    self.portfolio_manager.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        reason="sync_missing_binance"
                    )
                return True
            # ================================================================
            # SCENARIO 3: Both have positions → VALIDATE
            # ================================================================
            if btc_balance > MIN_BTC_BALANCE and portfolio_positions:
                qty_diff = abs(btc_balance - portfolio_total_qty)
                qty_diff_pct = (qty_diff / btc_balance * 100) if btc_balance > 0 else 0
                
                if qty_diff_pct > 0.1:
                    logger.warning(
                        f"[SYNC] QUANTITY MISMATCH:\n"
                        f"  Binance: {btc_balance:.8f} BTC\n"
                        f"  Portfolio: {portfolio_total_qty:.8f} BTC ({len(portfolio_positions)} positions)\n"
                        f"  Difference: {qty_diff:.8f} BTC ({qty_diff_pct:.2f}%)\n"
                        f"  → Closing all positions to clear mismatch"
                    )
                    for position in portfolio_positions:
                        self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason="sync_quantity_mismatch"
                        )
                    return True
                else:
                    logger.info(
                        f"[SYNC] ✓ {asset_name} positions in sync\n"
                        f"  Balance: {btc_balance:.8f} BTC\n"
                        f"  Positions: {len(portfolio_positions)}/{self.portfolio_manager.max_positions_per_asset}"
                    )
                    self._verify_vtm_status_after_sync(asset_name)
                    return True

            logger.info(f"[SYNC] ✓ No {asset_name} positions detected")
            return True

        except Exception as e:
            logger.error(f"[SYNC] Error: {e}", exc_info=True)
            return False
        
    def _verify_vtm_status_after_sync(self, asset: str):
        """
        ✅ NEW: Verify VTM is working after position sync
        """
        try:
            positions = self.portfolio_manager.get_asset_positions(asset)
            
            if not positions:
                return
            
            logger.info(f"\n[VTM VERIFICATION] Checking {len(positions)} position(s)...")
            
            vtm_active_count = 0
            vtm_missing_count = 0
            
            for pos in positions:
                if pos.trade_manager:
                    vtm_active_count += 1
                    status = pos.get_vtm_status()
                    logger.info(
                        f"  ✓ {pos.position_id}: VTM ACTIVE\n"
                        f"    Entry: ${status['entry_price']:,.2f} | Current: ${status['current_price']:,.2f}\n"
                        f"    SL: ${status['stop_loss']:,.2f} | TP: ${status['take_profit']:,.2f}\n"
                        f"    Profit Locked: {status['profit_locked']}"
                    )
                else:
                    vtm_missing_count += 1
                    logger.warning(
                        f"  ⚠️ {pos.position_id}: VTM NOT ACTIVE\n"
                        f"    Entry: ${pos.entry_price:,.2f}\n"
                        f"    Using static SL/TP instead"
                    )
            
            logger.info(
                f"\n[VTM VERIFICATION] Summary:\n"
                f"  Active:  {vtm_active_count}/{len(positions)}\n"
                f"  Missing: {vtm_missing_count}/{len(positions)}"
            )
            
        except Exception as e:
            logger.error(f"[VTM VERIFICATION] Error: {e}")
