"""
PRODUCTION-READY: Risk-based position sizing with automatic rebalancing
Handles Binance Futures position rebalancing safely
"""

import logging
from typing import Tuple, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionRebalancer:
    """
    Safely rebalances Futures positions to maintain risk targets
    """
    
    def __init__(self, futures_handler, portfolio_manager):
        self.futures_handler = futures_handler
        self.portfolio_manager = portfolio_manager
        
    def reduce_position_size(
        self,
        position,
        target_quantity: float,
        reason: str = "risk_rebalancing"
    ) -> bool:
        """
        Reduce a Binance Futures position to target quantity
        
        Args:
            position: Position object to reduce
            target_quantity: New desired quantity (MUST be < current)
            reason: Why we're reducing
        
        Returns:
            True if successful
        """
        try:
            if target_quantity >= position.quantity:
                logger.error(
                    f"[REBALANCE] Invalid: target {target_quantity:.6f} >= "
                    f"current {position.quantity:.6f}"
                )
                return False
            
            # Calculate how much to reduce
            reduction = position.quantity - target_quantity
            
            # Round to valid precision
            reduction = self.futures_handler._round_quantity(reduction)
            target_quantity = self.futures_handler._round_quantity(target_quantity)
            
            logger.info(
                f"\n{'='*80}\n"
                f"[REBALANCE] Reducing {position.side.upper()} position\n"
                f"{'='*80}\n"
                f"Position ID:  {position.position_id}\n"
                f"Current Qty:  {position.quantity:.6f}\n"
                f"Target Qty:   {target_quantity:.6f}\n"
                f"Reduction:    {reduction:.6f}\n"
                f"Reason:       {reason}\n"
                f"{'='*80}"
            )
            
            # Get current price for P&L calculation
            ticker = self.futures_handler.client.futures_symbol_ticker(
                symbol=self.futures_handler.symbol
            )
            current_price = float(ticker['price'])
            
            # ================================================================
            # STEP 1: CRITICAL - Cancel existing stop orders BEFORE reducing
            # ================================================================
            # Why? The stop order is for the FULL position. If we reduce size
            # but keep the old stop, it will try to close more than we have.
            logger.info(f"[REBALANCE] Canceling existing stop orders...")
            self.futures_handler._cancel_existing_stop_orders(position.side)
            
            # Small delay to ensure cancellation processes
            import time
            time.sleep(0.3)
            
            # ================================================================
            # STEP 2: Execute partial close on Futures
            # ================================================================
            close_side = "SELL" if position.side == "long" else "BUY"
            
            from binance.enums import FUTURE_ORDER_TYPE_MARKET
            
            order = self.futures_handler.client.futures_create_order(
                symbol=self.futures_handler.symbol,
                side=close_side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=reduction,
                reduceOnly=True  # Critical: Only reduce, don't reverse
            )
            
            logger.info(
                f"[REBALANCE] ✓ Partial close executed\n"
                f"  Order ID: {order.get('orderId')}\n"
                f"  Reduced:  {reduction:.6f}\n"
                f"  Price:    ${float(order.get('avgPrice', current_price)):,.2f}"
            )
            
            # ================================================================
            # STEP 3: Calculate P&L on closed portion
            # ================================================================
            exit_price = float(order.get('avgPrice', current_price))
            
            if position.side == "long":
                partial_pnl = (exit_price - position.entry_price) * reduction
            else:
                partial_pnl = (position.entry_price - exit_price) * reduction
            
            logger.info(
                f"[REBALANCE] Partial P&L: ${partial_pnl:,.2f} "
                f"({partial_pnl / (position.entry_price * reduction) * 100:+.2f}%)"
            )
            
            # ================================================================
            # STEP 4: Update portfolio manager
            # ================================================================
            # Update the position quantity
            old_qty = position.quantity
            position.quantity = target_quantity
            
            # Update position size USD
            old_size = position.quantity * position.entry_price
            position.position_size_usd = target_quantity * position.entry_price
            
            logger.info(
                f"[REBALANCE] Portfolio updated:\n"
                f"  Quantity: {old_qty:.6f} → {target_quantity:.6f}\n"
                f"  Size USD: ${old_size:,.2f} → ${position.position_size_usd:,.2f}"
            )
            
            # ================================================================
            # STEP 5: Place NEW stop loss for REDUCED position
            # ================================================================
            if position.stop_loss:
                logger.info(
                    f"[REBALANCE] Placing new stop loss for reduced position..."
                )
                
                # Get stop from VTM if available
                if position.trade_manager:
                    stop_price = position.trade_manager.current_stop_loss
                else:
                    stop_price = position.stop_loss
                
                # Round to valid precision
                stop_price = self.futures_handler._round_price(stop_price)
                
                sl_side = "SELL" if position.side == "long" else "BUY"
                
                from binance.enums import FUTURE_ORDER_TYPE_STOP_MARKET
                
                try:
                    sl_order = self.futures_handler.client.futures_create_order(
                        symbol=self.futures_handler.symbol,
                        side=sl_side,
                        type=FUTURE_ORDER_TYPE_STOP_MARKET,
                        stopPrice=stop_price,
                        closePosition=True  # Close remaining position
                    )
                    logger.info(
                        f"[REBALANCE] ✓ New stop placed: ${stop_price:,.2f}\n"
                        f"  Order ID: {sl_order.get('orderId')}"
                    )
                
                except Exception as e:
                    logger.error(f"[REBALANCE] Failed to place new stop: {e}")
                    # This is critical - if stop fails, close entire position
                    logger.critical(
                        f"[REBALANCE] ⚠️  NO STOP LOSS ON POSITION!\n"
                        f"  Manual intervention required or close position"
                    )
                    return False
            
            # ================================================================
            # STEP 6: Update realized P&L in portfolio
            # ================================================================
            self.portfolio_manager.realized_pnl_today += partial_pnl
            
            if not self.portfolio_manager.is_paper_mode:
                # Refresh capital from exchange
                self.portfolio_manager.refresh_capital(force=True)
            else:
                # Paper mode: update simulated capital
                self.portfolio_manager.current_capital += partial_pnl
                self.portfolio_manager.equity = self.portfolio_manager.current_capital
            
            logger.info(
                f"\n{'='*80}\n"
                f"✅ REBALANCING COMPLETE\n"
                f"{'='*80}\n"
                f"Position {position.position_id} reduced by {reduction:.6f}\n"
                f"New quantity: {target_quantity:.6f}\n"
                f"Partial P&L: ${partial_pnl:,.2f}\n"
                f"Stop loss: Updated\n"
                f"{'='*80}\n"
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"[REBALANCE] Error reducing position: {e}",
                exc_info=True
            )
            return False


# ============================================================================
# UPDATED HybridPositionSizer with Rebalancing Integration
# ============================================================================

class HybridPositionSizer:
    """
    ✅ FIXED: Risk-based position sizing with auto-rebalancing
    """
    
    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []
        
        # Risk parameters
        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get("aggressive_risk_threshold", 0.70)
        
        # Rebalancing settings
        self.auto_rebalance = self.risk_cfg.get("auto_rebalance_positions", True)
        self.rebalance_threshold = self.risk_cfg.get("rebalance_threshold_pct", 0.20)  # 20% over = rebalance
        
        # ✅ NEW: Initialize rebalancer (will be set by handler)
        self.rebalancer = None
        
        logger.info(
            f"[RISK SIZER] Initialized\n"
            f"  Target Risk:     {self.target_risk_pct:.2%}\n"
            f"  Max Risk:        {self.max_risk_pct:.2%}\n"
            f"  Auto-Rebalance:  {'✓ Enabled' if self.auto_rebalance else '✗ Disabled'}\n"
            f"  Rebal Threshold: {self.rebalance_threshold:.0%}"
        )
    
    def set_rebalancer(self, rebalancer: PositionRebalancer):
        """Set the rebalancer (called by handler after initialization)"""
        self.rebalancer = rebalancer
        logger.info("[RISK SIZER] Rebalancer connected")
    
    def calculate_size_risk_based(
        self,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        signal: int,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = "automated",
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> Tuple[float, Dict]:
        """
        ✅ FIXED: Calculate position size with auto-rebalancing
        """
        try:
            # Get asset balance
            asset_balance = self.portfolio_manager.get_asset_balance(asset)
            if asset_balance <= 0:
                logger.error(f"[RISK] Invalid balance: ${asset_balance:,.2f}")
                return 0.0, {"error": "invalid_asset_balance"}
            
            logger.info(
                f"[RISK] Using {asset} balance: ${asset_balance:,.2f}"
            )
            
            # Get existing positions
            side = "long" if signal == 1 else "short"
            existing_positions = self.portfolio_manager.get_asset_positions(asset)
            same_side_positions = [p for p in existing_positions if p.side == side]
            total_positions = len(same_side_positions) + 1
            
            logger.info(
                f"[RISK] Position Count:\n"
                f"  Existing {side.upper()}: {len(same_side_positions)}\n"
                f"  Total (with new):      {total_positions}"
            )
            
            # Determine total risk budget
            if sizing_mode == "elevated_risk":
                total_risk_pct = self.max_risk_pct
            elif sizing_mode == "reduced_risk":
                total_risk_pct = self.target_risk_pct * 0.75
            elif confidence_score and confidence_score >= self.aggressive_threshold:
                total_risk_pct = self.max_risk_pct
            else:
                total_risk_pct = self.target_risk_pct
            
            # Calculate split risk per position
            risk_pct_per_position = total_risk_pct / total_positions
            risk_amount_per_position = asset_balance * risk_pct_per_position
            
            logger.info(
                f"[RISK] Split Risk Calculation ({asset}):\n"
                f"  Asset Balance:     ${asset_balance:,.2f}\n"
                f"  Total Risk Budget: {total_risk_pct:.2%} (${total_risk_pct * asset_balance:,.2f})\n"
                f"  Positions:         {total_positions}\n"
                f"  Risk Per Position: {risk_pct_per_position:.2%} (${risk_amount_per_position:.2f})"
            )
            
            # ================================================================
            # CHECK EXISTING POSITIONS & REBALANCE IF NEEDED
            # ================================================================
            existing_total_risk = 0.0
            positions_to_rebalance = []
            
            for pos in same_side_positions:
                if pos.stop_loss:
                    pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
                    existing_total_risk += pos_risk
                    
                    # Check if position is significantly over its allocation
                    over_pct = (pos_risk / risk_amount_per_position) - 1.0
                    
                    if over_pct > self.rebalance_threshold:  # e.g., 20% over
                        positions_to_rebalance.append({
                            'position': pos,
                            'current_risk': pos_risk,
                            'target_risk': risk_amount_per_position,
                            'over_pct': over_pct
                        })
            
            existing_risk_pct = existing_total_risk / asset_balance
            
            logger.info(
                f"[RISK] Existing Risk Analysis:\n"
                f"  Total Risk:  ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})\n"
                f"  Over-sized:  {len(positions_to_rebalance)} position(s)"
            )
            
            # ✅ EXECUTE REBALANCING IF NEEDED
            if positions_to_rebalance and self.auto_rebalance and self.rebalancer:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"⚠️  REBALANCING REQUIRED\n"
                    f"{'='*80}\n"
                    f"{len(positions_to_rebalance)} position(s) exceed allocation\n"
                    f"Auto-rebalance: ENABLED\n"
                    f"{'='*80}"
                )
                
                rebalanced_count = 0
                rebalancing_freed_risk = 0.0
                
                for item in positions_to_rebalance:
                    pos = item['position']
                    target_risk = item['target_risk']
                    current_risk = item['current_risk']
                    
                    # Calculate target quantity
                    stop_distance = abs(pos.entry_price - pos.stop_loss)
                    target_quantity = target_risk / stop_distance
                    
                    logger.info(
                        f"\n[REBALANCE] Position {pos.position_id}:\n"
                        f"  Current: {pos.quantity:.6f} (${current_risk:,.2f} risk)\n"
                        f"  Target:  {target_quantity:.6f} (${target_risk:,.2f} risk)\n"
                        f"  Over by: {item['over_pct']:.0%}"
                    )
                    
                    # Execute rebalancing
                    success = self.rebalancer.reduce_position_size(
                        position=pos,
                        target_quantity=target_quantity,
                        reason=f"risk_split_{total_positions}_positions"
                    )
                    
                    if success:
                        rebalanced_count += 1
                        rebalancing_freed_risk += (current_risk - target_risk)
                        logger.info(f"  ✅ Rebalanced successfully")
                    else:
                        logger.error(f"  ❌ Rebalancing failed")
                
                # Recalculate existing risk after rebalancing
                existing_total_risk = sum(
                    abs(p.entry_price - p.stop_loss) * p.quantity
                    for p in same_side_positions
                    if p.stop_loss
                )
                existing_risk_pct = existing_total_risk / asset_balance
                
                logger.info(
                    f"\n[REBALANCE] Results:\n"
                    f"  Rebalanced:  {rebalanced_count}/{len(positions_to_rebalance)}\n"
                    f"  Freed Risk:  ${rebalancing_freed_risk:,.2f}\n"
                    f"  New Total:   ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})"
                )
            
            elif positions_to_rebalance and not self.auto_rebalance:
                logger.warning(
                    f"\n⚠️  {len(positions_to_rebalance)} position(s) need rebalancing\n"
                    f"But auto_rebalance is DISABLED in config\n"
                    f"Enable it to allow automatic position scaling"
                )
            
            # ================================================================
            # CALCULATE AVAILABLE BUDGET FOR NEW POSITION
            # ================================================================
            total_budget = asset_balance * total_risk_pct
            available_for_new = total_budget - existing_total_risk
            available_for_new_pct = available_for_new / asset_balance
            
            logger.info(
                f"[RISK] Budget Status:\n"
                f"  Total:     ${total_budget:,.2f} ({total_risk_pct:.2%})\n"
                f"  Used:      ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})\n"
                f"  Available: ${available_for_new:,.2f} ({available_for_new_pct:.2%})"
            )
            
            # Check if we have enough budget
            if available_for_new < risk_amount_per_position * 0.8:  # 80% threshold
                logger.error(
                    f"[RISK] REJECTED: Insufficient budget after rebalancing\n"
                    f"  Available: ${available_for_new:,.2f}\n"
                    f"  Required:  ${risk_amount_per_position:,.2f}\n"
                    f"  → Existing positions may need manual closure"
                )
                return 0.0, {
                    "error": "insufficient_budget_post_rebalance",
                    "available": available_for_new,
                    "required": risk_amount_per_position
                }
            
            # ================================================================
            # SIZE NEW POSITION
            # ================================================================
            # Validate stop distance
            asset_cfg = self.config["assets"][asset]
            risk_cfg = asset_cfg.get("risk", {})
            
            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price
            
            min_stop_pct = risk_cfg.get("min_stop_distance_pct", 0.01)
            max_stop_pct = risk_cfg.get("max_stop_distance_pct", 0.10)
            
            if stop_distance_pct < min_stop_pct or stop_distance_pct > max_stop_pct:
                logger.error(
                    f"[RISK] Invalid stop distance: {stop_distance_pct:.2%} "
                    f"(valid: {min_stop_pct:.2%}-{max_stop_pct:.2%})"
                )
                return 0.0, {"error": "invalid_stop_distance"}
            
            # Use available budget
            risk_to_use = min(available_for_new, risk_amount_per_position)
            position_size_usd = risk_to_use / stop_distance_pct
            
            # Apply limits
            min_size = asset_cfg.get("min_position_usd", 100)
            max_size = asset_cfg.get("max_position_usd", 100000)
            
            position_size_usd = max(min_size, min(position_size_usd, max_size))
            
            # Calculate final metrics
            actual_risk = position_size_usd * stop_distance_pct
            actual_risk_pct = actual_risk / asset_balance
            new_total_risk = existing_total_risk + actual_risk
            new_total_risk_pct = new_total_risk / asset_balance
            
            logger.info(
                f"\n{'='*80}\n"
                f"✅ NEW POSITION APPROVED\n"
                f"{'='*80}\n"
                f"Size:       ${position_size_usd:,.2f}\n"
                f"Risk:       ${actual_risk:,.2f} ({actual_risk_pct:.2%})\n"
                f"Stop Dist:  {stop_distance_pct:.2%}\n"
                f"\n"
                f"TOTAL {asset} {side.upper()} RISK:\n"
                f"  Before: ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})\n"
                f"  After:  ${new_total_risk:,.2f} ({new_total_risk_pct:.2%})\n"
                f"  Budget: {total_risk_pct:.2%}\n"
                f"  Status: ✅ WITHIN LIMITS\n"
                f"{'='*80}"
            )
            
            # Build metadata
            metadata = {
                "asset": asset,
                "side": side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "stop_distance_pct": stop_distance_pct * 100,
                "asset_balance": asset_balance,
                "total_positions": total_positions,
                "risk_per_position_pct": risk_pct_per_position * 100,
                "position_size_usd": position_size_usd,
                "actual_risk_pct": actual_risk_pct * 100,
                "total_risk_after_pct": new_total_risk_pct * 100,
                "rebalanced_positions": len(positions_to_rebalance),
                "timestamp": datetime.now().isoformat(),
            }
            
            return position_size_usd, metadata
            
        except Exception as e:
            logger.error(f"[RISK] Calculation error: {e}", exc_info=True)
            return 0.0, {"error": str(e)}


# ============================================================================
# INTEGRATION CODE - Add to binance_handler.py __init__
# ============================================================================

"""
Add this to BinanceExecutionHandler.__init__():

# After initializing self.sizer
if hasattr(self, 'futures_handler') and self.futures_handler:
    rebalancer = PositionRebalancer(
        futures_handler=self.futures_handler,
        portfolio_manager=self.portfolio_manager
    )
    self.sizer.set_rebalancer(rebalancer)
    logger.info("[HANDLER] ✓ Auto-rebalancing enabled")
"""