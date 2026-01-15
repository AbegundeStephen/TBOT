"""
Binance Execution Handler with Hybrid Position Sizing + Order Tracking
ENHANCED: Track Binance order IDs for accurate P&L tracking
"""

import logging
import time
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
from src.execution.binance_futures import BinanceFuturesHandler
from src.global_error_handler import handle_errors, ErrorSeverity
from src.execution.position_rebalancer import PositionRebalancer
logger = logging.getLogger(__name__)


class SizingMode:
    """Position sizing modes"""

    AUTOMATED = "automated"
    MANUAL_OVERRIDE = "override"
    REDUCED_RISK = "reduced_risk"
    ELEVATED_RISK = "elevated"

    def count_binance_positions(
        client: Client, asset: str = "BTC"
    ) -> Tuple[int, float]:
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
                    MIN_BTC_PER_POSITION = (
                        0.0001  # Adjust based on your typical position size
                    )

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
    """
    ✅ FIXED: Risk-based position sizing with VTM integration
    
    Key Changes:
    1. Sizes positions based on actual stop loss distance (not % of capital)
    2. Calculates risk FIRST, then position size
    3. Validates actual risk matches target (1.5-2%)
    """

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []
        
        # Get risk parameters from config
        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get("aggressive_risk_threshold", 0.70)
        
        logger.info(
            f"[RISK SIZER] Initialized\n"
            f"  Target Risk: {self.target_risk_pct:.2%}\n"
            f"  Max Risk:    {self.max_risk_pct:.2%}\n"
            f"  Aggressive:  >{self.aggressive_threshold:.0%} confidence"
        )
        
    def set_rebalancer(self, rebalancer: PositionRebalancer):
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
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> Tuple[float, Dict]:
        """
        ✅ FIXED: Calculate position size with automatic rebalancing of existing positions
        """
        try:
            # Get asset balance
            asset_balance = self.portfolio_manager.get_asset_balance(asset)
            if asset_balance <= 0:
                logger.error(f"[RISK] Invalid balance for {asset}: ${asset_balance:,.2f}")
                return 0.0, {"error": "invalid_asset_balance"}
            
            # Get side and existing positions
            side = "long" if signal == 1 else "short"
            existing_positions = self.portfolio_manager.get_asset_positions(asset)
            same_side_positions = [p for p in existing_positions if p.side == side]
            
            # Total positions INCLUDING the new one
            total_positions = len(same_side_positions) + 1
            
            logger.info(
                f"[RISK] Position Count:\n"
                f"  Existing {side.upper()}: {len(same_side_positions)}\n"
                f"  Total (with new):      {total_positions}"
            )
            
            # ================================================================
            # STEP 1: Calculate target risk per position (SPLIT)
            # ================================================================
            if sizing_mode == SizingMode.ELEVATED_RISK:
                total_risk_pct = self.max_risk_pct
            elif sizing_mode == SizingMode.REDUCED_RISK:
                total_risk_pct = self.target_risk_pct * 0.75
            elif confidence_score and confidence_score >= self.aggressive_threshold:
                total_risk_pct = self.max_risk_pct
            else:
                total_risk_pct = self.target_risk_pct
            
            # Split risk across ALL positions (existing + new)
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
            # STEP 2: Check if existing positions exceed their split allocation
            # ================================================================
            existing_total_risk = 0.0
            positions_over_budget = []
            
            for pos in same_side_positions:
                if pos.stop_loss:
                    pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
                    existing_total_risk += pos_risk
                    
                    # Check if this position uses more than its split allocation
                    if pos_risk > risk_amount_per_position * 1.1:  # 10% tolerance
                        positions_over_budget.append({
                            'position': pos,
                            'current_risk': pos_risk,
                            'target_risk': risk_amount_per_position,
                            'excess': pos_risk - risk_amount_per_position
                        })
            
            existing_risk_pct = existing_total_risk / asset_balance
            
            logger.info(
                f"[RISK] Existing Positions Analysis:\n"
                f"  Total Existing Risk: ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})\n"
                f"  Over Budget:         {len(positions_over_budget)} position(s)"
            )
            
            # ================================================================
            # STEP 3: Calculate available budget for new position
            # ================================================================
            total_budget = asset_balance * total_risk_pct
            available_for_new = total_budget - existing_total_risk
            available_for_new_pct = available_for_new / asset_balance
            
            logger.info(
                f"[RISK] Budget Allocation:\n"
                f"  Total Budget:  ${total_budget:,.2f} ({total_risk_pct:.2%})\n"
                f"  Used:          ${existing_total_risk:,.2f} ({existing_risk_pct:.2%})\n"
                f"  Available:     ${available_for_new:,.2f} ({available_for_new_pct:.2%})"
            )
            
            # ================================================================
            # STEP 4: Handle insufficient budget
            # ================================================================
            if available_for_new < risk_amount_per_position * 0.5:  # Less than 50% of target
                logger.warning(
                    f"\n{'='*80}\n"
                    f"⚠️  INSUFFICIENT RISK BUDGET\n"
                    f"{'='*80}\n"
                    f"Available: ${available_for_new:,.2f} ({available_for_new_pct:.2%})\n"
                    f"Required:  ${risk_amount_per_position:,.2f} ({risk_pct_per_position:.2%})\n"
                    f"\n"
                    f"🔧 REBALANCING REQUIRED\n"
                    f"{len(positions_over_budget)} position(s) need size reduction\n"
                    f"{'='*80}\n"
                )
                
                # ✅ OPTION A: Automatically rebalance (if enabled in config)
                auto_rebalance = self.config.get("risk_management", {}).get(
                    "auto_rebalance_positions", False
                )
                
                if auto_rebalance and positions_over_budget and self.rebalancer:
                    for item in positions_over_budget:
                        pos = item['position']
                        target_risk = item['target_risk']
                        stop_distance = abs(pos.entry_price - pos.stop_loss)
                        new_quantity = target_risk / stop_distance
                        reduction = pos.quantity - new_quantity

                        if reduction > 0.00001:
                            logger.info(f"  Reducing position {pos.position_id} by {reduction:.6f}...")
                            success = self.rebalancer.reduce_position_size(
                                position=pos,
                                target_quantity=new_quantity,
                                reason="risk_rebalancing"
                            )
                            if not success:
                                logger.error(f"  Failed to rebalance position {pos.position_id}")
                                return 0.0, {"error": "rebalancing_failed"}

                    # Recalculate available budget after rebalancing
                    new_existing_total_risk = 0.0
                    for pos in same_side_positions:
                        if pos.stop_loss:
                            pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
                            new_existing_total_risk += pos_risk

                    available_for_new = total_budget - new_existing_total_risk
                    available_for_new_pct = available_for_new / asset_balance

                    logger.info(
                        f"[REBALANCE] Budget recalculated:\n"
                        f"  Available: ${available_for_new:,.2f} ({available_for_new_pct:.2%})"
                    )

                    # Proceed to open new position if budget is sufficient
                    if available_for_new >= risk_amount_per_position * 0.8:  # 80% threshold
                        pass  # Continue to STEP 5
                    else:
                        logger.error(f"[REBALANCE] Still insufficient budget after rebalancing")
                        return 0.0, {"error": "insufficient_budget_post_rebalance"}

                
                # ✅ OPTION B: Reject if auto-rebalance disabled
                else:
                    logger.error(
                        f"[RISK] REJECTED: Insufficient budget\n"
                        f"  Enable 'auto_rebalance_positions' in config to fix automatically\n"
                        f"  Or manually close/reduce existing positions"
                    )
                    return 0.0, {
                        "error": "insufficient_risk_budget",
                        "available_risk": available_for_new_pct,
                        "required_risk": risk_pct_per_position,
                        "suggestion": "Enable auto_rebalance or reduce existing positions"
                    }
            
            # ================================================================
            # STEP 5: Size new position (normal flow)
            # ================================================================
            # Validate stop loss distance
            asset_cfg = self.config["assets"][asset]
            risk_cfg = asset_cfg.get("risk", {})
            
            min_stop_pct = risk_cfg.get("min_stop_distance_pct", 0.01)
            max_stop_pct = risk_cfg.get("max_stop_distance_pct", 0.10)
            
            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price
            
            if stop_distance_pct < min_stop_pct:
                logger.error(
                    f"[RISK] Stop too tight: {stop_distance_pct:.2%} < {min_stop_pct:.2%}"
                )
                return 0.0, {
                    "error": "stop_too_tight",
                    "stop_distance_pct": stop_distance_pct,
                    "min_required": min_stop_pct
                }
            
            if stop_distance_pct > max_stop_pct:
                logger.warning(
                    f"[RISK] Stop very wide: {stop_distance_pct:.2%} > {max_stop_pct:.2%}"
                )
                stop_distance_pct = max_stop_pct
                if signal == 1:
                    stop_loss_price = entry_price * (1 - max_stop_pct)
                else:
                    stop_loss_price = entry_price * (1 + max_stop_pct)
                stop_distance = abs(entry_price - stop_loss_price)
            
            # Use available budget (capped at per-position target)
            risk_to_use = min(available_for_new, risk_amount_per_position)
            position_size_usd = risk_to_use / stop_distance_pct
            
            logger.info(
                f"[RISK] Position Calculation:\n"
                f"  Risk Amount:     ${risk_to_use:.2f}\n"
                f"  Stop Distance:   ${stop_distance:.2f} ({stop_distance_pct:.2%})\n"
                f"  Units:           {risk_to_use / stop_distance:.6f}\n"
                f"  Position Size:   ${position_size_usd:,.2f}"
            )
            
            # ================================================================
            # STEP 6: Apply limits and calculate final metrics
            # ================================================================
            min_size = asset_cfg.get("min_position_usd", 100)
            max_size = asset_cfg.get("max_position_usd", 100000)
            
            if position_size_usd < min_size:
                logger.warning(f"[RISK] Position ${position_size_usd:.2f} below minimum ${min_size}")
                return 0.0, {
                    "error": "below_minimum",
                    "calculated_size": position_size_usd,
                    "minimum": min_size
                }
            
            position_size_usd = min(position_size_usd, max_size)
            
            # Calculate actual risk
            actual_risk = position_size_usd * stop_distance_pct
            actual_risk_pct = actual_risk / asset_balance
            
            # Calculate new total
            new_total_risk = existing_total_risk + actual_risk
            new_total_risk_pct = new_total_risk / asset_balance
            
            logger.info(
                f"[RISK] Total {asset} Risk ({side.upper()}) AFTER NEW POSITION:\n"
                f"  Asset Balance:    ${asset_balance:,.2f}\n"
                f"  Existing risk:    ${existing_total_risk:.2f} ({existing_risk_pct:.2%})\n"
                f"  New position:     ${actual_risk:.2f} ({actual_risk_pct:.2%})\n"
                f"  Total:            ${new_total_risk:.2f} ({new_total_risk_pct:.2%})\n"
                f"  Budget:           {total_risk_pct:.2%}\n"
                f"  Status:           {'✅ WITHIN BUDGET' if new_total_risk_pct <= total_risk_pct * 1.1 else '⚠️  OVER BUDGET'}"
            )
            
            # Final safety check
            if new_total_risk_pct > total_risk_pct * 1.1:  # 10% tolerance
                logger.error(
                    f"[RISK] REJECTED: Total {asset} {side.upper()} risk {new_total_risk_pct:.2%} "
                    f"would exceed budget {total_risk_pct:.2%}"
                )
                return 0.0, {
                    "error": "total_risk_budget_exceeded",
                    "total_risk": new_total_risk_pct,
                    "budget": total_risk_pct,
                    "asset": asset
                }
            
            # Build metadata
            metadata = {
                "asset": asset,
                "mode": sizing_mode,
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "stop_distance_pct": stop_distance_pct * 100,
                "asset_balance": asset_balance,
                "total_risk_budget_pct": total_risk_pct * 100,
                "total_positions": total_positions,
                "risk_per_position_pct": risk_pct_per_position * 100,
                "actual_risk_pct": actual_risk_pct * 100,
                "total_asset_risk_pct": new_total_risk_pct * 100,
                "position_size_usd": position_size_usd,
                "positions_over_budget": len(positions_over_budget),
                "rebalancing_required": len(positions_over_budget) > 0,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(
                f"\n{'='*80}\n"
                f"✅ POSITION APPROVED\n"
                f"{'='*80}\n"
                f"Size:  ${position_size_usd:,.2f}\n"
                f"Risk:  ${actual_risk:.2f} ({actual_risk_pct:.2%})\n"
                f"Total: ${new_total_risk:.2f} ({new_total_risk_pct:.2%}) of {total_risk_pct:.2%} budget\n"
                f"{'='*80}"
            )
            
            return position_size_usd, metadata
            
        except Exception as e:
            logger.error(f"[RISK] Error calculating position size: {e}", exc_info=True)
            return 0.0, {"error": str(e)}


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
        
        self.futures_handler = BinanceFuturesHandler(client=client, symbol=config["assets"]["BTC"]["symbol"])
        self.sizer.set_rebalancer(PositionRebalancer(self.futures_handler, self.portfolio_manager))
        logger.info("[HANDLER] ✓ Auto-rebalancing enabled")

        """ if hasattr(self, 'futures_handler') and self.futures_handler:
            rebalancer = PositionRebalancer(
                futures_handler=self.futures_handler,
                portfolio_manager=self.portfolio_manager
            )
            self.sizer.set_rebalancer(rebalancer)
            logger.info("[HANDLER] ✓ Auto-rebalancing enabled") """
        
        self.asset_config = config["assets"]["BTC"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]
        self.error_handler = None
        self.trading_bot = None 

        self.symbol = self.asset_config["symbol"]
        self.mode = self.trading_config.get("mode", "paper")
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )
        self.is_paper_mode = self.mode.lower() == "paper"

        logger.info(
            f"BinanceExecutionHandler with HybridPositionSizer initialized - Mode: {self.mode.upper()}"
        )

        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")

    def can_open_binance_position(
        self, asset: str = "BTC", side: str = "long"
    ) -> Tuple[bool, str]:
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
        estimated_positions, total_qty = SizingMode.count_binance_positions(
            self.client, "BTC"
        )

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
            return (
                False,
                f"Already have {portfolio_count}/{self.max_positions_per_asset} {side.upper()} positions",
            )

        return (
            True,
            f"OK - {portfolio_count}/{self.max_positions_per_asset} positions open",
        )
        
        
    def can_open_position_side(self, asset_name: str, side: str) -> Tuple[bool, str]:
        """
        Check if we can open a position on a specific SIDE
        
        Args:
            asset_name: Asset name (e.g., "BTC")
            side: "long" or "short"
        
        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Check if shorts are allowed in config
        if side == "short":
            allow_shorts = self.config["assets"][asset_name].get("allow_shorts", False)
            if not allow_shorts:
                return False, f"Short trading disabled for {asset_name} in config"

        # Check portfolio manager limits
        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(asset_name, side)
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        # Check max positions per side
        current_count = self.portfolio_manager.get_asset_position_count(asset_name, side)
        max_per_asset = self.max_positions_per_asset
        
        if current_count >= max_per_asset:
            return False, f"Already have {current_count}/{max_per_asset} {side.upper()} positions"

        return True, f"OK - {current_count}/{max_per_asset} {side.upper()} positions open"

    @handle_errors(
    component="binance_handler",
    severity=ErrorSeverity.ERROR,
    notify=True,
    reraise=False,
    default_return=None
)
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

    @handle_errors(
    component="binance_handler",
    severity=ErrorSeverity.CRITICAL,
    notify=True,
    reraise=False,
    default_return=False
)
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
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ BINANCE TWO-WAY TRADING: Execute trading signal
        
        Signal Logic:
        - BUY (+1):  Close ALL shorts → Open long
        - SELL (-1): Close ALL longs  → Open short
        - HOLD (0):  Check SL/TP only
        
        Args:
            signal: +1 (BUY), -1 (SELL), 0 (HOLD)
            current_price: Current market price
            asset_name: Asset name (e.g., "BTC")
            confidence_score: Signal confidence (0-1)
            market_condition: Market regime
            sizing_mode: Position sizing mode
            manual_size_usd: Manual position size override
            override_reason: Reason for manual override
            
        signal_details: Dict containing:
            - aggregator_mode: 'council' or 'performance'
            - mode_confidence: 0-1
            - regime_analysis: Market conditions
            - All other signal metadata
        
        Returns:
            True if action taken, False otherwise
        """
        
        if asset_name != "BTC":
            logger.error(
                f"[BINANCE HANDLER] ❌ WRONG ASSET!\n"
                f"  This handler is for BTC ONLY\n"
                f"  Received request for: {asset_name}\n"
                f"  REJECTING EXECUTION"
            )
            return False
        
        try:
            # ============================================================
            # STEP 1: Get current price
            # ============================================================
            if current_price is None:
                current_price = self.get_current_price()

            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid price: {current_price}")
                return False

            # ============================================================
            # STEP 2: Get existing positions
            # ============================================================
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)      
            long_positions = [p for p in existing_positions if p.side == "long"]
            short_positions = [p for p in existing_positions if p.side == "short"]

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} Signal: {signal:+2d}\n"
                f"[STATE] Current Positions: {len(long_positions)} LONG, {len(short_positions)} SHORT\n"
                f"{'='*80}"
            )
            
            # ✅ Log hybrid mode if present
            if signal_details and signal_details.get('aggregator_mode'):
                logger.info(
                    f"\n[HYBRID] Mode: {signal_details['aggregator_mode'].upper()} "
                    f"({signal_details.get('mode_confidence', 0):.0%} confidence)"
                )

            # ============================================================
            # SCENARIO 1: SELL SIGNAL (-1) → Close longs, Open short
            # ============================================================
            if signal == -1:
                # Step 1: Close ALL long positions
                if long_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📉 SELL SIGNAL - Closing {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )
                    
                    closed_count = 0
                    failed_count = 0

                    for i, position in enumerate(long_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(long_positions)}] Closing LONG position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  Order ID:    {position.binance_order_id}\n"
                            f"  Entry:       ${position.entry_price:,.2f}\n"
                            f"  Current:     ${current_price:,.2f}"
                        )

                        success = self._close_position(
                            position=position,
                            current_price=current_price,
                            asset_name=asset_name,
                            reason="sell_signal",
                        )

                        if success:
                            closed_count += 1
                            logger.info(f"  ✓ Position {position.position_id} closed")
                        else:
                            failed_count += 1
                            logger.error(f"  ✗ Failed to close {position.position_id}")

                    logger.info(
                        f"\n{'='*80}\n"
                        f"CLOSE SUMMARY: {closed_count} closed, {failed_count} failed\n"
                        f"{'='*80}\n"
                    )

                else:
                    logger.debug(f"{asset_name}: SELL signal but no LONG positions to close")

                # Step 2: Check if we can open SHORT
                can_open, reason = self.can_open_position_side(asset_name, "short")
                
                if not can_open:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"⚠️  CANNOT OPEN SHORT POSITION\n"
                        f"Reason: {reason}\n"
                        f"{'='*80}\n"
                    )
                    # Return True if we closed positions, False if nothing happened
                    return len(long_positions) > 0

                # Step 3: Open SHORT position
                logger.info(
                    f"\n{'='*80}\n"
                    f"📉 SELL SIGNAL - Opening new SHORT position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )

                return self._open_position(
                    signal=-1,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details
                )

            # ============================================================
            # SCENARIO 2: BUY SIGNAL (+1) → Close shorts, Open long
            # ============================================================
            elif signal == 1:
                # Step 1: Close ALL short positions
                if short_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Closing {len(short_positions)} SHORT position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    failed_count = 0

                    for i, position in enumerate(short_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(short_positions)}] Closing SHORT position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  Order ID:    {position.binance_order_id}\n"
                            f"  Entry:       ${position.entry_price:,.2f}\n"
                            f"  Current:     ${current_price:,.2f}"
                        )

                        success = self._close_position(
                            position=position,
                            current_price=current_price,
                            asset_name=asset_name,
                            reason="buy_signal",
                        )

                        if success:
                            closed_count += 1
                            logger.info(f"  ✓ Position {position.position_id} closed")
                        else:
                            failed_count += 1
                            logger.error(f"  ✗ Failed to close {position.position_id}")

                    logger.info(
                        f"\n{'='*80}\n"
                        f"CLOSE SUMMARY: {closed_count} closed, {failed_count} failed\n"
                        f"{'='*80}\n"
                    )

                else:
                    logger.debug(f"{asset_name}: BUY signal but no SHORT positions to close")

                # Step 2: Check if we can open LONG
                can_open, reason = self.can_open_position_side(asset_name, "long")

                if not can_open:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"⚠️  CANNOT OPEN LONG POSITION\n"
                        f"Reason: {reason}\n"
                        f"{'='*80}\n"
                    )
                    return len(short_positions) > 0

                # Step 3: Open LONG position
                logger.info(
                    f"\n{'='*80}\n"
                    f"📈 BUY SIGNAL - Opening new LONG position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )

                return self._open_position(
                    signal=1,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details
                )

            # ============================================================
            # SCENARIO 3: HOLD SIGNAL (0) → Check SL/TP for all positions
            # ============================================================
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
        self, position, current_price: float
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
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}")
            return False, ""
        
        
    def _close_binance_order(
        self, quantity: float, asset_name: str, order_id: int = None
    ) -> bool:
        """
        Close position on Binance (SPOT or simulated SHORT)
        
        Args:
            quantity: Amount to sell/buy back
            asset_name: Asset name
            order_id: Original order ID (optional, for tracking)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # ✅ PAPER MODE: Simulate close without actual API call
            if self.is_paper_mode:
                logger.info(
                    f"[PAPER] Simulated close of {quantity:.8f} {self.symbol} "
                    f"(Order ID: {order_id})"
                )
                return True
            
            # LIVE MODE: Execute actual close
            # For SPOT LONG positions: Market SELL
            order = self.client.order_market_sell(
                symbol=self.symbol, 
                quantity=quantity
            )
            
            logger.info(
                f"[BINANCE] ✓ Position closed: {quantity:.8f} {self.symbol}\n"
                f"  Order ID: {order.get('orderId')}\n"
                f"  Status: {order.get('status')}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[BINANCE] Error closing position: {e}")
            return False
        
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
            leverage = handler.config.get("assets", {}).get("BTC", {}).get("leverage", 20)
            margin_type = handler.config.get("assets", {}).get("BTC", {}).get("margin_type", "CROSSED")
            is_futures=True
            
            handler.futures_handler.set_leverage(leverage)
            handler.futures_handler.set_margin_type(margin_type)
            handler.futures_handler.is_futures = is_futures
            
            logger.info("[FUTURES] ✓ Futures handler integrated")
            return True
            
        except Exception as e:
            logger.error(f"[FUTURES] Integration failed: {e}")
            return False
        
    def _round_quantity_precision(self, quantity: float, symbol: str = "BTCUSDT", is_futures: bool = False) -> float:
        """
        Round quantity to the correct precision for Binance (spot or futures).
        """
        try:
            if is_futures:
                # Futures: 3 decimal places for BTCUSDT
                rounded_qty = round(quantity, 3)
            else:
                # Spot: 6 decimal places for BTCUSDT
                rounded_qty = round(quantity, 6)

            logger.info(f"[PRECISION] Rounded quantity: {quantity} → {rounded_qty}")
            return rounded_qty

        except Exception as e:
            logger.error(f"Error rounding quantity precision: {e}")
            return quantity
    
    
    def _round_quantity(self, quantity: float, symbol: str = "BTCUSDT", is_futures: bool = False) -> float:
        """
        Round quantity to the correct lot size step and precision for Binance.
        """
        try:
            # Get symbol info from Binance
            if is_futures:
                exchange_info = self.client.futures_exchange_info()
            else:
                exchange_info = self.client.get_exchange_info()

            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step_size = float(f["stepSize"])
                            min_qty = float(f["minQty"])
                            max_qty = float(f["maxQty"])

                            # Calculate precision from step_size
                            # E.g., step_size = 0.00001 → precision = 5
                            precision = len(str(step_size).rstrip('0').split('.')[-1])

                            logger.info(
                                f"[LOT_SIZE] {symbol} | Min: {min_qty} | Max: {max_qty} | "
                                f"Step: {step_size} | Precision: {precision}"
                            )

                            # Round to the nearest step size
                            rounded_qty = round(quantity / step_size) * step_size
                            
                            # Apply precision to remove floating point errors
                            rounded_qty = round(rounded_qty, precision)
                            
                            # Clamp to min/max bounds
                            rounded_qty = max(min_qty, min(rounded_qty, max_qty))
                            
                            logger.info(f"[QUANTITY] {quantity:.8f} → {rounded_qty:.{precision}f}")
                            
                            return rounded_qty

            logger.warning(f"Could not find LOT_SIZE filter for {symbol}, using default precision")
            # Fallback: use 5 decimals for BTC spot
            return round(quantity, 5)

        except Exception as e:
            logger.error(f"Error rounding quantity: {e}")
            # Fallback: use 5 decimals for BTC spot
            return round(quantity, 5)


    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False
    )
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
        signal_details: Dict = None,
        
    ) -> bool:
        """
        Open LONG or SHORT position with hybrid-aware VTM and Exchange Safety Stop
        """
        try:
            # ============================================================
            # STEP 1: Determine side from signal
            # ============================================================
            side = "long" if signal == 1 else "short"
            logger.info(f"[OPEN] Opening {side.upper()} position for {asset_name}")

            # ============================================================
            # STEP 2: Calculate position size
            # ============================================================
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

            position_size_usd, sizing_metadata = self.sizer.calculate_size_risk_based(sizing_request)

            # Apply short size reduction if configured
            if side == "short":
                short_config = self.config.get("portfolio", {}).get("short_position_sizing", {})
                use_reduced = short_config.get("use_reduced_size_for_shorts", False)
                multiplier = short_config.get("short_size_multiplier", 0.8)
                if use_reduced and multiplier < 1.0:
                    logger.info(f"[SHORT] Applying {multiplier}x size reduction")
                    position_size_usd *= multiplier

            if position_size_usd <= 0:
                logger.warning(f"{asset_name}: Invalid position size: ${position_size_usd:.2f}")
                return False
                
            # ✅ MERGE SIZING METADATA INTO SIGNAL DETAILS
            # This ensures sizing info is available in Step 7 and persisted
            if signal_details is None:
                signal_details = {}
            if sizing_metadata:
                signal_details['sizing'] = sizing_metadata

            # ============================================================
            # STEP 3: Calculate quantity
            # ============================================================
            quantity = position_size_usd / current_price
            
            # Round to correct precision and lot size
            is_futures = (
            hasattr(self, "futures_handler") and 
            self.futures_handler is not None and
            self.config.get("assets", {}).get(asset_name, {}).get("enable_futures", True)
        )
            quantity = self._round_quantity(quantity, self.symbol, is_futures)
            leverage = 20
            margin_type = "CROSSED"
            
            if is_futures:
            # Fetch from config (using safe gets)
                asset_conf = self.config.get("assets", {}).get(asset_name, {})
                leverage = asset_conf.get("leverage", 20)
                margin_type = asset_conf.get("margin_type", "CROSSED")

            MIN_BTC = 0.00001
            if quantity < MIN_BTC:
                logger.warning(f"{asset_name}: Quantity {quantity:.8f} below minimum {MIN_BTC}")
                return False

            # ============================================================
            # STEP 4: Calculate "Catastrophic" Safety Stop (Required for Futures)
            # ============================================================
            # Even though VTM manages the trade, we need a HARD STOP on the exchange
            # in case the bot crashes or loses internet.
            
            risk_config = self.asset_config.get("risk", {})
            safety_sl_price = None
            
            # Default to 5% if not specified, just to be safe
            sl_pct = risk_config.get("stop_loss_pct", 0.05) 
            
            if side == "long":
                safety_sl_price = current_price * (1 - sl_pct)
            else: # SHORT
                safety_sl_price = current_price * (1 + sl_pct)
                
            # Round safety stop to valid precision
            if hasattr(self, 'futures_handler'):
                # BTC usually allows 1 or 2 decimals for price, safe to use 2
                safety_sl_price = round(safety_sl_price, 2)

            logger.info(
                f"[OPEN] {side.upper()} {quantity:.8f} {self.symbol} @ ${current_price:,.2f}\n"
                f"  Size: ${position_size_usd:,.2f}\n"
                f"  Safety SL: ${safety_sl_price:,.2f} (Hard order on exchange)\n"
                f"  VTM: Active (Will manage dynamic exits)"
            )

            # ============================================================
            # STEP 5: Execute on Binance (Futures or Spot)
            # ============================================================
            order_id = None

            # Try Futures first if enabled
            if hasattr(self, "futures_handler"):
                try:
                    # ✅ PASS THE SAFETY STOP HERE
                    if side == "long":
                        order = self.futures_handler.open_long_position(
                            quantity=quantity,
                            stop_loss=safety_sl_price, # <--- Critical: Pass the value
                            take_profit=None 
                        )
                    else:
                        order = self.futures_handler.open_short_position(
                            quantity=quantity,
                            stop_loss=safety_sl_price, # <--- Critical: Pass the value
                            take_profit=None
                        )

                    if order:
                        order_id = order.get("orderId")
                        logger.info(f"[FUTURES] ✓ {side.upper()} position opened via Futures")
                    else:
                        logger.warning(f"[FUTURES] Failed to open {side.upper()} position, falling back to spot")

                except Exception as e:
                    logger.warning(f"[FUTURES] Error opening {side.upper()}: {e}, falling back to spot")

            # Fall back to spot if Futures fails or is not available
            if not order_id:
                if not self.is_paper_mode:
                    try:
                        if side == "long":
                            order = self.client.order_market_buy(
                                symbol=self.symbol,
                                quantity=quantity
                            )
                            order_id = order.get("orderId")
                            logger.info(f"[SPOT] ✓ LONG position opened via Spot")
                        else:  # SHORT
                            logger.error(
                                f"[SPOT] ❌ SHORT positions require Binance Futures API\n"
                                f"  Current mode: SPOT (only supports LONG)\n"
                                f"  Asset: {asset_name} will only trade LONG positions in live mode."
                            )
                            return False
                    except Exception as e:
                        logger.error(f"[SPOT] Order execution failed: {e}")
                        return False
                else:
                    # Paper mode: simulate order execution
                    order_id = f"PAPER_{side.upper()}_{int(time.time())}"
                    logger.info(
                        f"[PAPER] ✓ Simulated {side.upper()} order\n"
                        f"  Order ID: {order_id}\n"
                        f"  Quantity: {quantity:.8f} {self.symbol}\n"
                        f"  Entry: ${current_price:,.2f}"
                    )

            # ============================================================
            # STEP 6: Fetch OHLC data for VTM
            # ============================================================
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
                except Exception as e:
                    logger.warning(f"[VTM] OHLC fetch failed: {e}")

            # ============================================================
            # STEP 7: Add position to Portfolio Manager
            # ============================================================
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=None, # VTM will calculate its own soft stop
                take_profit=None, # VTM will calculate its own targets
                trailing_stop_pct=None, # VTM will manage
                binance_order_id=order_id,
                ohlc_data=ohlc_data,
                use_dynamic_management=True, # ✅ Enable VTM
                signal_details=signal_details, # ✅ Pass hybrid context (now includes sizing)
                leverage=leverage,
                margin_type=margin_type,
                is_futures=is_futures
            )

            if success:
                logger.info(f"[OK] {asset_name} {side.upper()} position opened successfully")
                if order_id:
                    logger.info(f"  └─ Order ID: {order_id}")
                if ohlc_data:
                    logger.info(f"  └─ VTM: ACTIVE")
                if self.is_paper_mode:
                    logger.info(f"  └─ Mode: PAPER (simulated)")
                return True
            else:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset_name} position")
                return False

        except Exception as e:
            logger.error(f"[OPEN] Error opening {asset_name} position: {e}", exc_info=True)
            return False

    @handle_errors(
    component="binance_handler",
    severity=ErrorSeverity.CRITICAL,
    notify=True,
    reraise=False,
    default_return=False
)   
    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """
        Close a LONG or SHORT position using Binance Futures API if available, otherwise fall back to spot.
        """
        try:
            side = position.side
            quantity = position.quantity
            binance_order_id = position.binance_order_id

            logger.info(f"[CLOSE] Closing {side.upper()} position for {asset_name}")

            # Try Futures first if enabled
            if hasattr(self, "futures_handler"):
                try:
                    if side == "long":
                        success = self.futures_handler.close_long_position(
                            quantity=quantity,
                            order_id=binance_order_id
                        )
                    else:
                        success = self.futures_handler.close_short_position(
                            quantity=quantity,
                            order_id=binance_order_id
                        )

                    if success:
                        logger.info(f"[FUTURES] ✓ {side.upper()} position closed via Futures")
                    else:
                        logger.warning(f"[FUTURES] Failed to close {side.upper()} position, falling back to spot")
                        success = False

                except Exception as e:
                    logger.warning(f"[FUTURES] Error closing {side.upper()}: {e}, falling back to spot")
                    success = False

            # Fall back to spot if Futures fails or is not available
            if not hasattr(self, "futures_handler") or not success:
                if not self.is_paper_mode:
                    try:
                        if side == "long":
                            order = self.client.order_market_sell(
                                symbol=self.symbol,
                                quantity=quantity
                            )
                            logger.info(f"[SPOT] ✓ LONG position closed via Spot")
                            success = True
                        else:  # SHORT
                            logger.error(
                                f"[SPOT] ❌ SHORT positions require Binance Futures API\n"
                                f"  Current mode: SPOT (only supports LONG)\n"
                                f"  Asset: {asset_name} will only trade LONG positions in live mode."
                            )
                            success = False
                    except Exception as e:
                        logger.error(f"[SPOT] Error closing position: {e}")
                        success = False
                else:
                    # Paper mode: simulate close
                    logger.info(
                        f"[PAPER] ✓ Simulated close of {side.upper()} position\n"
                        f"  Order ID: {binance_order_id}"
                    )
                    success = True

            # Close in portfolio manager
            if success:
                trade_result = self.portfolio_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=reason
                )

                if trade_result:
                    logger.info(f"[OK] {side.upper()} position closed successfully")
                    return True
                else:
                    logger.error(f"[FAIL] Portfolio close failed for {side.upper()}")
                    return False
            else:
                logger.error(f"[FAIL] Failed to close {side.upper()} position")
                return False

        except Exception as e:
            logger.error(f"[CLOSE] Error closing position: {e}", exc_info=True)
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
                    exit_signal = position.trade_manager.update_with_current_price(
                        current_price
                    )

                    # ✅ FIX: exit_signal is a dict, not a string
                    if exit_signal:
                        exit_reason = exit_signal.get("reason", "unknown")
                        exit_price = exit_signal.get("price", current_price)
                        exit_size = exit_signal.get("size", position.quantity)
                        
                        # Convert ExitReason enum to string
                        if hasattr(exit_reason, 'value'):
                            exit_reason_str = exit_reason.value
                        else:
                            exit_reason_str = str(exit_reason)
                        
                        logger.info(
                            f"[VTM] {asset_name} {position.position_id} triggered "
                            f"{exit_reason_str.upper()} @ ${exit_price:,.2f} "
                            f"(closing {exit_size:.0%} of position)"
                        )
                        
                        # Close the position
                        self._close_position(
                            position, current_price, asset_name, f"VTM_{exit_reason_str}"
                        )
                        positions_closed = True
                        continue

                # Fallback: check traditional SL/TP
                should_close, reason = self._check_stop_loss_take_profit(
                    position, current_price
                )
                if should_close:
                    logger.info(
                        f"[AUTO-CLOSE] {asset_name} {position.position_id}: {reason}"
                    )
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

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.WARNING,
        notify=True,
        reraise=False,
        default_return=False
    )
    def sync_positions_with_binance(self, asset_name: str = "BTC", symbol: str = None) -> bool:
        """
        ✅ FIXED: Sync portfolio with Binance (Futures-First + Short Support)
        
        Priority Logic:
        1. Check Futures API (supports LONG + SHORT)
        2. Fall back to Spot ONLY if Futures is unavailable
        3. Never mix Futures and Spot positions
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            logger.info(f"[SYNC] Starting position sync for {asset_name}...")
            
            # ============================================================
            # STEP 1: Get Portfolio Positions
            # ============================================================
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            portfolio_total_qty = sum(pos.quantity for pos in portfolio_positions)
            
            # Determine portfolio side(s)
            portfolio_side = None
            if portfolio_positions:
                longs = sum(1 for p in portfolio_positions if p.side == 'long')
                shorts = sum(1 for p in portfolio_positions if p.side == 'short')
                if longs > 0 and shorts == 0: 
                    portfolio_side = "long"
                elif shorts > 0 and longs == 0: 
                    portfolio_side = "short"
                elif longs > 0 and shorts > 0: 
                    portfolio_side = "mixed"

            # ============================================================
            # STEP 2: Get Binance Position (Futures-First)
            # ============================================================
            binance_qty = 0.0
            binance_side = None  # "long", "short", or None
            
            # ✅ PRIORITY 1: Check Futures API
            if hasattr(self, "futures_handler") and self.futures_handler:
                try:
                    pos_info = self.futures_handler.get_position_info()
                    
                    if pos_info:
                        pos_amt = float(pos_info.get('positionAmt', 0))
                        binance_qty = abs(pos_amt)
                        
                        if pos_amt > 0:
                            binance_side = "long"
                        elif pos_amt < 0:
                            binance_side = "short"
                        
                        if binance_qty > 0:
                            logger.info(
                                f"[SYNC] ✅ Futures Position Detected:\n"
                                f"  Amount: {pos_amt:+.8f} BTC\n"
                                f"  Side:   {binance_side.upper()}\n"
                                f"  Entry:  ${float(pos_info.get('entryPrice', 0)):,.2f}\n"
                                f"  P&L:    ${float(pos_info.get('unRealizedProfit', 0)):,.2f}"
                            )
                            
                            # ✅ CRITICAL: Mark that we're using Futures data
                            using_futures = True
                        else:
                            logger.info(f"[SYNC] No Futures position for {asset_name}")
                            using_futures = True  # Still checked Futures, just empty
                    else:
                        logger.info(f"[SYNC] No Futures position info available")
                        using_futures = True
                        
                except Exception as e:
                    logger.warning(f"[SYNC] Futures API error: {e}")
                    logger.info(f"[SYNC] Falling back to Spot balance check...")
                    using_futures = False
            else:
                logger.info(f"[SYNC] Futures handler not available, using Spot")
                using_futures = False
            
            # ✅ FALLBACK: Check Spot Balance (ONLY if Futures unavailable)
            if not using_futures:
                logger.warning(
                    f"[SYNC] ⚠️  USING SPOT FALLBACK\n"
                    f"  This mode does NOT support SHORT positions!\n"
                    f"  Enable Futures handler for full LONG+SHORT support."
                )
                
                account = self.client.get_account()
                for balance in account["balances"]:
                    if balance["asset"] == asset_name:  # e.g., "BTC"
                        qty = float(balance["free"]) + float(balance["locked"])
                        if qty > 0:
                            binance_qty = qty
                            binance_side = "long"  # Spot is always long
                            logger.info(f"[SYNC] Detected Spot Balance: {binance_qty:.8f} {asset_name}")
                        break

            # ============================================================
            # Get Current Price
            # ============================================================
            current_price = self.get_current_price(symbol)
            MIN_QTY_THRESHOLD = 0.0001

            # ================================================================
            # SCENARIO 1: Binance has Position, Portfolio Empty → IMPORT
            # ================================================================
            if binance_qty > MIN_QTY_THRESHOLD and not portfolio_positions:
                import_enabled = bool(
                    self.config.get("portfolio", {}).get("import_existing_positions", False)
                )
                
                if import_enabled and binance_side:
                    logger.info(
                        f"[SYNC] Found {binance_side.upper()} position of {binance_qty:.8f} {asset_name}\n"
                        f"  Source: {'Futures' if using_futures else 'Spot'}\n"
                        f"  → Importing into Portfolio with VTM support..."
                    )
                    
                    # Fetch OHLC for VTM
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
                    except Exception as e:
                        logger.error(f"[VTM] Failed to fetch OHLC: {e}")

                    # Generate signal details
                    signal_details = {
                        'imported': True,
                        'import_time': datetime.now().isoformat(),
                        'source': 'futures' if using_futures else 'spot',
                        'aggregator_mode': 'unknown',
                        'reasoning': f'{asset_name} {binance_side} position imported from Binance'
                    }

                    # Add to portfolio
                    position_size_usd = binance_qty * current_price
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side=binance_side,
                        entry_price=current_price,
                        position_size_usd=position_size_usd,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=self.config["assets"][asset_name].get("risk", {}).get("trailing_stop_pct"),
                        binance_order_id=None,
                        ohlc_data=ohlc_data,
                        use_dynamic_management=True,
                        signal_details=signal_details,
                        is_futures=using_futures,  # ✅ Track data source
                    )
                    
                    if success:
                        logger.info(
                            f"[SYNC] ✅ Imported {binance_qty:.8f} {asset_name} as {binance_side.upper()}\n"
                            f"  Source: {'Futures' if using_futures else 'Spot'}\n"
                            f"  Value:  ${position_size_usd:,.2f}"
                        )
                        return True
                    else:
                        logger.error(f"[SYNC] ❌ Failed to import position")
                        return False
                else:
                    logger.info(f"[SYNC] Import disabled or invalid side. Ignoring.")
                    return True

            # ================================================================
            # SCENARIO 2: Portfolio has Positions, Binance Empty → CLOSE ALL
            # ================================================================
            if portfolio_positions and binance_qty <= MIN_QTY_THRESHOLD:
                logger.warning(
                    f"[SYNC] ⚠️  DESYNC DETECTED!\n"
                    f"  Portfolio: {len(portfolio_positions)} position(s)\n"
                    f"  Binance:   Empty\n"
                    f"  → Closing portfolio positions (likely closed manually)"
                )
                
                for position in portfolio_positions:
                    self.portfolio_manager.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        reason="sync_missing_binance",
                    )
                return True

            # ================================================================
            # SCENARIO 3: Both Have Positions → VALIDATE
            # ================================================================
            if portfolio_positions and binance_qty > MIN_QTY_THRESHOLD:
                # Check 1: Side Mismatch (Critical for Futures)
                if binance_side != portfolio_side and portfolio_side != "mixed":
                    logger.error(
                        f"[SYNC] 🚨 CRITICAL SIDE MISMATCH!\n"
                        f"  Binance:   {binance_side.upper()}\n"
                        f"  Portfolio: {portfolio_side.upper()}\n"
                        f"  → This should NEVER happen with Futures!\n"
                        f"  → Closing portfolio to resync..."
                    )
                    
                    for position in portfolio_positions:
                        self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason="sync_critical_side_mismatch",
                        )
                    return True

                # Check 2: Quantity Mismatch
                qty_diff = abs(binance_qty - portfolio_total_qty)
                qty_diff_pct = (qty_diff / binance_qty * 100) if binance_qty > 0 else 0

                if qty_diff_pct > 10.0:  # Allow 10% variance
                    logger.warning(
                        f"[SYNC] ⚠️  QUANTITY MISMATCH (>10%):\n"
                        f"  Binance:   {binance_qty:.8f}\n"
                        f"  Portfolio: {portfolio_total_qty:.8f}\n"
                        f"  Diff:      {qty_diff_pct:.2f}%\n"
                        f"  → Closing portfolio to resync..."
                    )
                    
                    for position in portfolio_positions:
                        self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason="sync_quantity_mismatch",
                        )
                    return True
                
                else:
                    logger.info(
                        f"[SYNC] ✅ Positions in sync\n"
                        f"  Side: {binance_side.upper()}\n"
                        f"  Qty:  {binance_qty:.6f} BTC\n"
                        f"  Diff: {qty_diff_pct:.2f}%"
                    )
                    
                    # Verify VTM is active
                    self._verify_vtm_status_after_sync(asset_name)
                    return True

            # ================================================================
            # SCENARIO 4: Both Empty → OK
            # ================================================================
            logger.info(f"[SYNC] ✅ No {asset_name} positions detected (Clean state)")
            return True

        except Exception as e:
            logger.error(f"[SYNC] Error: {e}", exc_info=True)
            return False


    def _sync_futures_positions(self, asset_name: str, symbol: str) -> bool:
        """
        ✅ NEW: Sync Futures positions (supports LONG + SHORT)
        """
        try:
            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            
            # Get Futures positions from API
            futures_positions = self.futures_handler.client.futures_position_information(symbol=symbol)
            
            # Find active Futures positions
            active_futures = []
            for pos in futures_positions:
                pos_amt = float(pos.get('positionAmt', 0))
                if pos_amt != 0:
                    side = "long" if pos_amt > 0 else "short"
                    active_futures.append({
                        'side': side,
                        'quantity': abs(pos_amt),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                    })
            
            # Get current price
            current_price = self.get_current_price(symbol)
            
            logger.info(
                f"[SYNC] State:\n"
                f"  Portfolio: {len(portfolio_positions)} position(s)\n"
                f"  Futures:   {len(active_futures)} position(s)"
            )
            
            # ============================================================
            # SCENARIO 1: Futures has positions, portfolio is empty → IMPORT
            # ============================================================
            if active_futures and not portfolio_positions:
                import_enabled = self.config.get("portfolio", {}).get("import_existing_positions", False)
                
                if not import_enabled:
                    logger.warning(
                        f"[SYNC] Found {len(active_futures)} Futures position(s) but import disabled\n"
                        f"  → Enable 'import_existing_positions' in config to import them"
                    )
                    return True
                
                logger.info(f"[SYNC] Importing {len(active_futures)} Futures position(s)...")
                
                for fut_pos in active_futures:
                    # Fetch OHLC data for VTM
                    ohlc_data = None
                    try:
                        from datetime import datetime, timedelta, timezone
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)
                        
                        df = self.data_manager.fetch_binance_data(
                            symbol=symbol,
                            interval=self.asset_config.get("interval", "1h"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        
                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                    except Exception as e:
                        logger.error(f"[VTM] OHLC fetch failed: {e}")
                    
                    # Get signal details (basic fallback)
                    signal_details = {
                        'imported': True,
                        'import_time': datetime.now().isoformat(),
                        'aggregator_mode': 'unknown',
                        'mode_confidence': 0.5,
                        'regime_analysis': {
                            'regime_type': 'unknown',
                            'trend_strength': 'unknown',
                            'volatility_regime': 'normal',
                        },
                        'signal_quality': 0.5,
                        'reasoning': f'{fut_pos["side"].upper()} position imported from Binance Futures',
                    }
                    
                    # Calculate position size
                    position_size_usd = fut_pos['quantity'] * fut_pos['entry_price']
                    
                    # Import position
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side=fut_pos['side'],
                        entry_price=fut_pos['entry_price'],
                        position_size_usd=position_size_usd,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=self.asset_config.get("risk", {}).get("trailing_stop_pct"),
                        binance_order_id=None,
                        ohlc_data=ohlc_data,
                        use_dynamic_management=True,
                        entry_time=datetime.now(),
                        signal_details=signal_details,
                    )
                    
                    if success:
                        logger.info(
                            f"[SYNC] ✓ Imported {fut_pos['side'].upper()}: "
                            f"{fut_pos['quantity']:.6f} BTC @ ${fut_pos['entry_price']:,.2f}"
                        )
                    else:
                        logger.error(f"[SYNC] ✗ Failed to import {fut_pos['side'].upper()} position")
                
                return True
            
            # ============================================================
            # SCENARIO 2: Portfolio has positions, Futures is empty → CLOSE
            # ============================================================
            if portfolio_positions and not active_futures:
                logger.warning(
                    f"[SYNC] Portfolio has {len(portfolio_positions)} position(s) "
                    f"but Futures API shows none\n"
                    f"  → Removing portfolio positions (likely closed manually)"
                )
                
                for pos in portfolio_positions:
                    self.portfolio_manager.close_position(
                        position_id=pos.position_id,
                        exit_price=current_price,
                        reason="sync_missing_futures",
                    )
                
                return True
            
            # ============================================================
            # SCENARIO 3: Both have positions → VALIDATE
            # ============================================================
            if portfolio_positions and active_futures:
                # Group by side
                portfolio_by_side = {
                    'long': [p for p in portfolio_positions if p.side == 'long'],
                    'short': [p for p in portfolio_positions if p.side == 'short'],
                }
                
                futures_by_side = {
                    'long': [f for f in active_futures if f['side'] == 'long'],
                    'short': [f for f in active_futures if f['side'] == 'short'],
                }
                
                # Check each side
                for side in ['long', 'short']:
                    port_qty = sum(p.quantity for p in portfolio_by_side[side])
                    fut_qty = sum(f['quantity'] for f in futures_by_side[side])
                    
                    if abs(port_qty - fut_qty) > 0.0001:  # Allow for rounding
                        logger.warning(
                            f"[SYNC] {side.upper()} quantity mismatch:\n"
                            f"  Portfolio: {port_qty:.6f} BTC\n"
                            f"  Futures:   {fut_qty:.6f} BTC\n"
                            f"  → Closing all {side} positions to resync"
                        )
                        
                        for pos in portfolio_by_side[side]:
                            self.portfolio_manager.close_position(
                                position_id=pos.position_id,
                                exit_price=current_price,
                                reason=f"sync_{side}_mismatch",
                            )
                
                logger.info("[SYNC] ✓ Positions validated")
                return True
            
            # ============================================================
            # SCENARIO 4: Both empty → OK
            # ============================================================
            logger.info(f"[SYNC] ✓ No {asset_name} positions detected")
            return True
        
        except Exception as e:
            logger.error(f"[SYNC] Futures sync error: {e}", exc_info=True)
            return False


    def _sync_spot_positions(self, asset_name: str, symbol: str) -> bool:
        """
        ✅ Original Spot sync logic (for backward compatibility)
        """
        try:
            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            
            # Get Spot balance
            account = self.client.get_account()
            btc_balance = 0.0
            
            for balance in account["balances"]:
                if balance["asset"] == "BTC":
                    btc_balance = float(balance["free"]) + float(balance["locked"])
            
            current_price = self.get_current_price(symbol)
            MIN_BTC_BALANCE = 0.0001
            
            # Calculate total portfolio quantity
            portfolio_total_qty = sum(pos.quantity for pos in portfolio_positions)
            
            logger.info(
                f"[SYNC] State:\n"
                f"  Portfolio: {portfolio_total_qty:.6f} BTC\n"
                f"  Spot:      {btc_balance:.6f} BTC"
            )
            
            # SCENARIO 1: Spot has BTC, portfolio empty → IMPORT
            if btc_balance > MIN_BTC_BALANCE and not portfolio_positions:
                import_enabled = self.config.get("portfolio", {}).get("import_existing_positions", False)
                
                if not import_enabled:
                    logger.warning(
                        f"[SYNC] BTC balance {btc_balance:.6f} detected but import disabled\n"
                        f"  → Bot will open new positions on BUY signals"
                    )
                    return True
                
                logger.info(f"[SYNC] Importing {btc_balance:.6f} BTC as LONG position...")
                
                # Fetch OHLC data
                ohlc_data = None
                try:
                    from datetime import datetime, timedelta, timezone
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)
                    
                    df = self.data_manager.fetch_binance_data(
                        symbol=symbol,
                        interval=self.asset_config.get("interval", "1h"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    
                    if len(df) > 50:
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                        }
                except Exception as e:
                    logger.error(f"[VTM] OHLC fetch failed: {e}")
                
                position_size_usd = btc_balance * current_price
                
                success = self.portfolio_manager.add_position(
                    asset=asset_name,
                    symbol=symbol,
                    side="long",
                    entry_price=current_price,
                    position_size_usd=position_size_usd,
                    stop_loss=None,
                    take_profit=None,
                    trailing_stop_pct=self.asset_config.get("risk", {}).get("trailing_stop_pct"),
                    binance_order_id=None,
                    ohlc_data=ohlc_data,
                    use_dynamic_management=True,
                    entry_time=datetime.now(),
                    signal_details={
                        'imported': True,
                        'import_time': datetime.now().isoformat(),
                    },
                )
                
                if success:
                    logger.info(f"[SYNC] ✓ Imported {btc_balance:.6f} BTC")
                return success
            
            # SCENARIO 2: Portfolio has positions, Spot empty → CLOSE
            if portfolio_positions and btc_balance <= MIN_BTC_BALANCE:
                logger.warning(
                    f"[SYNC] Portfolio has {len(portfolio_positions)} position(s) "
                    f"but Spot balance is {btc_balance:.6f}\n"
                    f"  → Removing positions (likely sold manually)"
                )
                
                for pos in portfolio_positions:
                    self.portfolio_manager.close_position(
                        position_id=pos.position_id,
                        exit_price=current_price,
                        reason="sync_missing_spot",
                    )
                return True
            
            # SCENARIO 3: Both have positions → VALIDATE
            if btc_balance > MIN_BTC_BALANCE and portfolio_positions:
                qty_diff = abs(btc_balance - portfolio_total_qty)
                qty_diff_pct = (qty_diff / btc_balance * 100) if btc_balance > 0 else 0
                
                if qty_diff_pct > 0.1:
                    logger.warning(
                        f"[SYNC] Quantity mismatch:\n"
                        f"  Spot:      {btc_balance:.6f} BTC\n"
                        f"  Portfolio: {portfolio_total_qty:.6f} BTC ({len(portfolio_positions)} positions)\n"
                        f"  Difference: {qty_diff:.6f} BTC ({qty_diff_pct:.2f}%)\n"
                        f"  → Closing all positions to clear mismatch"
                    )
                    
                    for pos in portfolio_positions:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="sync_quantity_mismatch",
                        )
                    return True
                else:
                    logger.info("[SYNC] ✓ Positions in sync")
                    return True
            
            # SCENARIO 4: Both empty
            logger.info(f"[SYNC] ✓ No {asset_name} positions detected")
            return True
        
        except Exception as e:
            logger.error(f"[SYNC] Spot sync error: {e}", exc_info=True)
            return False


    def _verify_vtm_status_after_sync(self, asset: str):
        """
        ✅ Verify VTM is working after position sync
        (Same method for both MT5 and Binance - add to binance_handler.py if missing)
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

    def _verify_vtm_status_after_sync(self, asset: str):
        """
        ✅ NEW: Verify VTM is working after position sync
        """
        try:
            positions = self.portfolio_manager.get_asset_positions(asset)

            if not positions:
                return

            logger.info(
                f"\n[VTM VERIFICATION] Checking {len(positions)} position(s)..."
            )

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
