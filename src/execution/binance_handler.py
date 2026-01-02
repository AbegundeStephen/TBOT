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
        ✅ FIXED: Calculate position size based on ACTUAL RISK
        
        Correct Formula:
            Position Size = Risk Amount / Stop Loss Distance (in $)
            
        Example:
            - Account: $10,000
            - Risk: 1.5% = $150
            - Entry: $45,000
            - Stop: $43,650 (3% below entry)
            - Stop Distance: $1,350
            - Position Size: $150 / $1,350 = 0.111 BTC = $5,000
            - Actual Risk: 0.111 * $1,350 = $150 ✓
        """
        try:
            # ============================================================
            # STEP 1: Determine risk percentage to use
            # ============================================================
            if sizing_mode == SizingMode.ELEVATED_RISK:
                risk_pct = self.max_risk_pct
            elif sizing_mode == SizingMode.REDUCED_RISK:
                risk_pct = self.target_risk_pct * 0.75
            elif confidence_score and confidence_score >= self.aggressive_threshold:
                risk_pct = self.max_risk_pct
            else:
                risk_pct = self.target_risk_pct
            
            # Calculate risk amount in dollars
            risk_amount = self.portfolio_manager.current_capital * risk_pct
            
            # ============================================================
            # STEP 2: Validate stop loss distance
            # ============================================================
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
                if signal == 1:  # LONG
                    stop_loss_price = entry_price * (1 - max_stop_pct)
                else:  # SHORT
                    stop_loss_price = entry_price * (1 + max_stop_pct)
                stop_distance = abs(entry_price - stop_loss_price)
            
            # ============================================================
            # STEP 3: Calculate position size from risk (✅ FIXED)
            # ============================================================
            # ✅ CORRECT: Divide by DOLLAR distance, not percentage
            position_size_usd = risk_amount / stop_distance_pct
            
            # ✅ Alternative (more explicit) calculation:
            # units = risk_amount / stop_distance  # e.g., 0.111 BTC
            # position_size_usd = units * entry_price  # e.g., $5,000
            
            logger.info(
                f"[RISK] Position Calculation:\n"
                f"  Risk Amount:     ${risk_amount:.2f}\n"
                f"  Stop Distance:   ${stop_distance:.2f} ({stop_distance_pct:.2%})\n"
                f"  Units:           {risk_amount / stop_distance:.6f}\n"
                f"  Position Size:   ${position_size_usd:,.2f}"
            )
            
            # ============================================================
            # STEP 4: Apply manual override if requested
            # ============================================================
            if sizing_mode == SizingMode.MANUAL_OVERRIDE and manual_size_usd:
                min_allowed = position_size_usd * 0.5
                max_allowed = position_size_usd * 1.5
                
                if manual_size_usd < min_allowed or manual_size_usd > max_allowed:
                    logger.warning(
                        f"[OVERRIDE] Manual size ${manual_size_usd:,.2f} outside safe range "
                        f"[${min_allowed:,.2f}, ${max_allowed:,.2f}] - Clamping"
                    )
                    manual_size_usd = max(min_allowed, min(manual_size_usd, max_allowed))
                
                override_risk = manual_size_usd * stop_distance_pct
                override_risk_pct = override_risk / self.portfolio_manager.current_capital
                
                logger.info(
                    f"[OVERRIDE] Manual size applied\n"
                    f"  Calculated:  ${position_size_usd:,.2f}\n"
                    f"  Manual:      ${manual_size_usd:,.2f}\n"
                    f"  New Risk:    {override_risk_pct:.2%} (${override_risk:.2f})"
                )
                
                position_size_usd = manual_size_usd
            
            # ============================================================
            # STEP 5: Apply hard position limits
            # ============================================================
            min_size = asset_cfg.get("min_position_usd", 100)
            max_size = asset_cfg.get("max_position_usd", 6000)
            
            if position_size_usd < min_size:
                logger.warning(f"[RISK] Position ${position_size_usd:.2f} below minimum ${min_size}")
                return 0.0, {
                    "error": "below_minimum",
                    "calculated_size": position_size_usd,
                    "minimum": min_size
                }
            
            original_size = position_size_usd
            position_size_usd = min(position_size_usd, max_size)
            
            if original_size > max_size:
                logger.warning(
                    f"[RISK] Position clamped to max: ${original_size:,.2f} → ${max_size:,.2f}"
                )
            
            # ============================================================
            # STEP 6: Check portfolio-level limits (✅ CRITICAL)
            # ============================================================
            max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
            max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
            
            current_exposure = sum(
                pos.quantity * pos.entry_price
                for pos in self.portfolio_manager.positions.values()
            )
            
            # ✅ CRITICAL CHECK: Ensure new position doesn't exceed portfolio limits
            max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
            if current_exposure + position_size_usd > max_portfolio_usd:
                old_size = position_size_usd
                position_size_usd = max(0, max_portfolio_usd - current_exposure)
                logger.warning(
                    f"[RISK] Portfolio limit hit: ${old_size:,.2f} → ${position_size_usd:,.2f}\n"
                    f"  Current Exposure: ${current_exposure:,.2f}\n"
                    f"  Max Allowed:      ${max_portfolio_usd:,.2f}"
                )
            
            # ✅ CRITICAL CHECK: Single asset exposure
            max_asset_usd = self.portfolio_manager.current_capital * max_single_asset
            if position_size_usd > max_asset_usd:
                old_size = position_size_usd
                position_size_usd = max_asset_usd
                logger.warning(
                    f"[RISK] Single asset limit hit: ${old_size:,.2f} → ${position_size_usd:,.2f}"
                )
            
            # ✅ SAFETY CHECK: Never exceed 2% of capital per position
            max_position_safety = self.portfolio_manager.current_capital * 0.02
            if position_size_usd > max_position_safety:
                logger.error(
                    f"[RISK] SAFETY VIOLATION: Position ${position_size_usd:,.2f} exceeds 2% limit!"
                )
                position_size_usd = max_position_safety
            
            # ============================================================
            # STEP 7: Calculate ACTUAL risk with final position size
            # ============================================================
            actual_risk = position_size_usd * stop_distance_pct
            actual_risk_pct = actual_risk / self.portfolio_manager.current_capital
            
            # ✅ VERIFY: Actual risk should match target
            if actual_risk_pct > self.max_risk_pct:
                logger.error(
                    f"[RISK] CRITICAL: Actual risk {actual_risk_pct:.2%} exceeds max {self.max_risk_pct:.2%}!"
                )
                return 0.0, {"error": "risk_exceeds_maximum"}
            
            # ============================================================
            # STEP 8: Build metadata
            # ============================================================
            metadata = {
                "asset": asset,
                "mode": sizing_mode,
                "signal": signal,
                "confidence_score": confidence_score,
                "market_condition": market_condition,
                
                # Entry and stop
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "stop_distance_usd": stop_distance,
                "stop_distance_pct": stop_distance_pct * 100,
                
                # Risk calculation
                "target_risk_pct": risk_pct * 100,
                "target_risk_usd": risk_amount,
                "actual_risk_pct": actual_risk_pct * 100,
                "actual_risk_usd": actual_risk,
                
                # Position sizing
                "position_size_usd": position_size_usd,
                "position_size_pct": (position_size_usd / self.portfolio_manager.current_capital) * 100,
                "position_size_units": position_size_usd / entry_price,
                
                # Override info
                "override_details": {
                    "mode": sizing_mode,
                    "reason": override_reason
                } if sizing_mode != SizingMode.AUTOMATED else None,
                
                "timestamp": datetime.now().isoformat(),
            }
            
            # Log detailed sizing decision
            logger.info(
                f"\n{'='*80}\n"
                f"[RISK-BASED SIZING] {asset} {['SHORT', 'HOLD', 'LONG'][signal+1]}\n"
                f"{'='*80}\n"
                f"📊 MARKET INFO:\n"
                f"  Entry Price:    ${entry_price:,.2f}\n"
                f"  Stop Loss:      ${stop_loss_price:,.2f}\n"
                f"  Stop Distance:  ${stop_distance:,.2f} ({stop_distance_pct:.2%})\n"
                + (f"  Confidence:     {confidence_score:.0%}\n" if confidence_score else "")
                + f"  Condition:      {market_condition}\n"
                f"\n"
                f"💰 RISK CALCULATION:\n"
                f"  Account:        ${self.portfolio_manager.current_capital:,.2f}\n"
                f"  Target Risk:    {risk_pct:.2%} (${risk_amount:.2f})\n"
                f"  Position Size:  ${position_size_usd:,.2f} ({metadata['position_size_pct']:.2f}% of capital)\n"
                f"  Units:          {position_size_usd / entry_price:.6f}\n"
                f"\n"
                f"✅ ACTUAL RISK:\n"
                f"  Dollar Risk:    ${actual_risk:.2f}\n"
                f"  Risk %:         {actual_risk_pct:.2%}\n"
                f"  Status:         {'✅ WITHIN TARGET' if abs(actual_risk_pct - risk_pct) < 0.005 else '⚠️  ADJUSTED'}\n"
                f"{'='*80}"
            )
            
            # Track overrides
            if sizing_mode != SizingMode.AUTOMATED:
                self.override_history.append(metadata)
            
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
            leverage = handler.config.get("assets", {}).get("BTC", {}).get("leverage", 10)
            margin_type = handler.config.get("assets", {}).get("BTC", {}).get("margin_type", "CROSSED")
            
            handler.futures_handler.set_leverage(leverage)
            handler.futures_handler.set_margin_type(margin_type)
            
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
        ✅ FIXED: Open position with RISK-FIRST approach
        
        Flow:
        1. Calculate stop loss FIRST using VTM
        2. Size position based on actual stop loss distance
        3. Verify risk is within limits
        4. Execute trade
        """
        try:
            # ============================================================
            # STEP 1: Determine side
            # ============================================================
            side = "long" if signal == 1 else "short"
            logger.info(f"[OPEN] Opening {side.upper()} position for {asset_name}")

            # ============================================================
            # STEP 2: Fetch OHLC data for VTM
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
                    
                    if len(df) > 50:
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                        }
                        logger.info(f"[VTM] Fetched {len(df)} bars for stop calculation")
                    else:
                        logger.warning(f"[VTM] Insufficient data ({len(df)} bars)")
                
                except Exception as e:
                    logger.error(f"[VTM] OHLC fetch failed: {e}")
                    ohlc_data = None
            
            # ============================================================
            # STEP 3: Calculate stop loss FIRST using VTM
            # ============================================================
            stop_loss_price = None
            
            if ohlc_data and signal_details:
                try:
                    from src.trading.veteran_trade_manager import VeteranTradeManager
                    
                    logger.info(f"[VTM] Calculating structure-based stop loss...")
                    
                    # Initialize VTM temporarily just to get stop loss
                    temp_vtm = VeteranTradeManager(
                        entry_price=current_price,
                        side=side,
                        asset=asset_name,
                        high=ohlc_data["high"],
                        low=ohlc_data["low"],
                        close=ohlc_data["close"],
                        account_balance=self.portfolio_manager.current_capital,
                        signal_details=signal_details,
                        account_risk=0.015,  # Temporary, won't affect stop calculation
                    )
                    
                    stop_loss_price = temp_vtm.initial_stop_loss
                    
                    stop_distance_pct = abs(current_price - stop_loss_price) / current_price
                    
                    logger.info(
                        f"[VTM] ✓ Structure-based stop calculated:\n"
                        f"  Entry: ${current_price:,.2f}\n"
                        f"  Stop:  ${stop_loss_price:,.2f}\n"
                        f"  Distance: ${abs(current_price - stop_loss_price):,.2f} ({stop_distance_pct:.2%})"
                    )
                    
                except Exception as e:
                    logger.error(f"[VTM] Stop calculation failed: {e}")
                    stop_loss_price = None
            
            # ============================================================
            # STEP 4: Fallback to fixed % if VTM unavailable
            # ============================================================
            if stop_loss_price is None:
                risk = self.asset_config.get("risk", {})
                
                if side == "long":
                    stop_loss_pct = risk.get("stop_loss_pct", 0.02)
                    stop_loss_price = current_price * (1 - stop_loss_pct)
                else:
                    stop_loss_pct = risk.get("stop_loss_pct_short", risk.get("stop_loss_pct", 0.025))
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                
                logger.warning(
                    f"[VTM] Using fallback fixed stop: "
                    f"${stop_loss_price:,.2f} ({stop_loss_pct:.2%})"
                )
            
            # ============================================================
            # STEP 5: Calculate position size based on ACTUAL stop
            # ============================================================
            position_size_usd, sizing_metadata = self.sizer.calculate_size_risk_based(
                asset=asset_name,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,  # ✅ Use actual stop from VTM
                signal=signal,
                confidence_score=confidence_score,
                market_condition=market_condition,
                sizing_mode=sizing_mode,
                manual_size_usd=manual_size_usd,
                override_reason=override_reason,
            )
            
            # Check for errors
            if position_size_usd <= 0:
                error_msg = sizing_metadata.get("error", "Unknown error")
                logger.error(f"[FAIL] Position sizing failed: {error_msg}")
                return False
            
            # ============================================================
            # STEP 6: Apply short size reduction if configured
            # ============================================================
            if side == "short":
                short_config = self.config.get("portfolio", {}).get("short_position_sizing", {})
                use_reduced = short_config.get("use_reduced_size_for_shorts", False)
                multiplier = short_config.get("short_size_multiplier", 0.8)
                
                if use_reduced and multiplier < 1.0:
                    original_size = position_size_usd
                    position_size_usd *= multiplier
                    logger.info(
                        f"[SHORT] Applied {multiplier}x size reduction: "
                        f"${original_size:,.2f} → ${position_size_usd:,.2f}"
                    )
                    
                    # Recalculate actual risk after reduction
                    stop_distance_pct = abs(current_price - stop_loss_price) / current_price
                    actual_risk = position_size_usd * stop_distance_pct
                    actual_risk_pct = actual_risk / self.portfolio_manager.current_capital
                    
                    logger.info(
                        f"[SHORT] Adjusted risk: {actual_risk_pct:.2%} (${actual_risk:.2f})"
                    )
            
            # ============================================================
            # STEP 7: Final risk verification
            # ============================================================
            max_risk_pct = self.config.get("portfolio", {}).get("max_risk_per_trade", 0.025)
            
            if sizing_metadata.get("actual_risk_pct", 0) / 100 > max_risk_pct:
                logger.error(
                    f"[BLOCKED] Risk exceeds maximum!\n"
                    f"  Actual:  {sizing_metadata['actual_risk_pct']:.2%}\n"
                    f"  Maximum: {max_risk_pct:.2%}"
                )
                return False
            
            # ============================================================
            # STEP 8: Calculate quantity
            # ============================================================
            quantity = position_size_usd / current_price
            
            # Round to correct precision
            is_futures = hasattr(self, "futures_handler") and self.config.get("binance", {}).get("enable_futures", False)
            quantity = self._round_quantity(quantity, self.symbol, is_futures)
            
            MIN_BTC = 0.00001
            if quantity < MIN_BTC:
                logger.warning(f"{asset_name}: Quantity {quantity:.8f} below minimum {MIN_BTC}")
                return False
            
            # Recalculate actual position size after rounding
            actual_position_size = quantity * current_price
            
            logger.info(
                f"[SIZE] Final position details:\n"
                f"  Quantity: {quantity:.8f} {self.symbol}\n"
                f"  Value:    ${actual_position_size:,.2f}\n"
                f"  Entry:    ${current_price:,.2f}\n"
                f"  Stop:     ${stop_loss_price:,.2f}"
            )
            
            # ============================================================
            # STEP 9: Execute on Binance (Futures or Spot)
            # ============================================================
            order_id = None
            
            # Try Futures first if enabled
            if hasattr(self, "futures_handler"):
                try:
                    if side == "long":
                        order = self.futures_handler.open_long_position(
                            quantity=quantity,
                            stop_loss=None,  # VTM manages
                            take_profit=None  # VTM manages
                        )
                    else:
                        order = self.futures_handler.open_short_position(
                            quantity=quantity,
                            stop_loss=None,
                            take_profit=None
                        )
                    
                    if order:
                        order_id = order.get("orderId")
                        logger.info(f"[FUTURES] ✓ {side.upper()} position opened via Futures")
                    else:
                        logger.warning(f"[FUTURES] Failed, falling back to spot")
                
                except Exception as e:
                    logger.warning(f"[FUTURES] Error: {e}, falling back to spot")
            
            # Fall back to spot if Futures fails
            if not order_id:
                if not self.is_paper_mode:
                    try:
                        if side == "long":
                            order = self.client.order_market_buy(
                                symbol=self.symbol,
                                quantity=quantity
                            )
                            order_id = order.get("orderId")
                            logger.info(f"[SPOT] ✓ LONG position opened")
                        else:
                            logger.error(
                                f"[SPOT] ❌ SHORT requires Futures API\n"
                                f"  Enable futures or trade LONG only"
                            )
                            return False
                    
                    except Exception as e:
                        logger.error(f"[SPOT] Order failed: {e}")
                        return False
                else:
                    # Paper mode
                    order_id = f"PAPER_{side.upper()}_{int(time.time())}"
                    logger.info(f"[PAPER] ✓ Simulated {side.upper()} order: {order_id}")
            
            # ============================================================
            # STEP 10: Add position to Portfolio Manager with VTM
            # ============================================================
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=actual_position_size,
                stop_loss=None,  # ✅ VTM will manage this
                take_profit=None,  # ✅ VTM will manage this
                trailing_stop_pct=None,  # ✅ VTM will manage this
                binance_order_id=order_id,
                ohlc_data=ohlc_data,
                use_dynamic_management=True,  # ✅ Enable VTM
                signal_details=signal_details,
            )
            
            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected position")
                
                # Emergency: Close unwanted position
                if order_id and not self.is_paper_mode:
                    logger.warning(f"[EMERGENCY] Closing unwanted position...")
                    # Add emergency close logic here
                
                return False
            
            # ============================================================
            # STEP 11: Verify VTM initialized correctly
            # ============================================================
            positions = self.portfolio_manager.get_asset_positions(asset_name)
            if positions:
                new_position = positions[-1]
                
                if new_position.trade_manager:
                    vtm_status = new_position.get_vtm_status()
                    logger.info(
                        f"\n{'='*80}\n"
                        f"[VTM] ✅ ACTIVE for {side.upper()} position\n"
                        f"{'='*80}\n"
                        f"  Order ID: {order_id}\n"
                        f"  Entry:    ${vtm_status['entry_price']:,.2f}\n"
                        f"  SL:       ${vtm_status['stop_loss']:,.2f} (VTM calculated)\n"
                        f"  TP:       ${vtm_status.get('take_profit', 0):,.2f} (VTM calculated)\n"
                        f"  Size:     {quantity:.8f} {self.symbol}\n"
                        f"  Risk:     ${sizing_metadata['actual_risk_usd']:,.2f} ({sizing_metadata['actual_risk_pct']:.2%})\n"
                        f"{'='*80}"
                    )
                else:
                    logger.warning(f"[VTM] ⚠️ NOT INITIALIZED - Using static SL/TP")
            
            logger.info(f"[OK] {side.upper()} {asset_name} position opened successfully")
            return True
        
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
        ✅ COMPLETE FIX: Sync portfolio with Binance holdings (multi-position aware + VTM)
        Just copy-paste this entire method to replace your existing one in binance_handler.py
        """
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

            # Calculate total portfolio quantity
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
                    
                    # ============================================================
                    # STEP 1: Fetch OHLC data for VTM
                    # ============================================================
                    ohlc_data = None
                    df = None
                    
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
                        df = None

                    # ============================================================
                    # STEP 2: Get REAL market analysis from HybridAggregatorSelector
                    # ============================================================
                    signal_details_base = None

                    if df is not None and len(df) > 200:
                        try:
                            logger.info(f"[HYBRID] Analyzing market for imported BTC position...")

                            # Try to get hybrid selector from parent bot
                            hybrid_selector = None
                            if hasattr(self, 'trading_bot') and hasattr(self.trading_bot, 'hybrid_selector'):
                                hybrid_selector = self.trading_bot.hybrid_selector
                                logger.info(f"[HYBRID] Using existing hybrid_selector from bot")
                            else:
                                # Create temporary instance
                                from src.execution.hybrid_aggregator_selector import HybridAggregatorSelector
                                hybrid_selector = HybridAggregatorSelector(
                                    self.data_manager,
                                    self.config,
                                )
                                logger.info(f"[HYBRID] Created temporary hybrid_selector")

                            # Get current market analysis
                            mode_info = hybrid_selector.get_optimal_mode(asset_name, df)
                            analysis = mode_info['analysis']

                            # Build REAL signal_details from market analysis
                            signal_details_base = {
                                'imported': True,
                                'import_time': datetime.now().isoformat(),
                                
                                # Real aggregator mode
                                'aggregator_mode': mode_info['mode'],
                                'mode_confidence': mode_info['confidence'],
                                
                                # Real regime analysis
                                'regime_analysis': {
                                    'regime_type': analysis['regime_type'],
                                    'trend_strength': analysis['trend']['strength'],
                                    'trend_direction': analysis['trend']['direction'],
                                    'adx': analysis['trend']['adx'],
                                    'volatility_regime': analysis['volatility']['regime'],
                                    'volatility_ratio': analysis['volatility']['ratio'],
                                    'price_clarity': analysis['price_action']['clarity'],
                                    'indecision_pct': analysis['price_action']['indecision_pct'],
                                    'momentum_aligned': analysis['momentum_aligned'],
                                    'at_key_level': analysis['at_key_level'],
                                },
                                
                                'signal_quality': mode_info['confidence'],
                                'reasoning': f"BTC position imported from Binance - {analysis['reasoning']}",
                            }

                            logger.info(f"[HYBRID] ✅ Market analysis complete:")
                            logger.info(f"  Mode:       {mode_info['mode'].upper()}")
                            logger.info(f"  Confidence: {mode_info['confidence']:.0%}")
                            logger.info(f"  Regime:     {analysis['regime_type']}")
                            logger.info(f"  Trend:      {analysis['trend']['strength']} / {analysis['trend']['direction']}")
                            logger.info(f"  Volatility: {analysis['volatility']['regime']}")

                        except Exception as e:
                            logger.error(f"[HYBRID] ❌ Analysis failed: {e}")
                            signal_details_base = None

                    # Fallback if hybrid analysis fails
                    if signal_details_base is None:
                        logger.warning(f"[HYBRID] Using fallback signal_details (no market analysis)")
                        signal_details_base = {
                            'imported': True,
                            'import_time': datetime.now().isoformat(),
                            'aggregator_mode': 'unknown',
                            'mode_confidence': 0.5,
                            'regime_analysis': {
                                'regime_type': 'unknown',
                                'trend_strength': 'unknown',
                                'trend_direction': 'unknown',
                                'adx': 20.0,
                                'volatility_regime': 'normal',
                                'volatility_ratio': 1.0,
                                'price_clarity': 'unknown',
                                'indecision_pct': 0.0,
                                'momentum_aligned': False,
                                'at_key_level': False,
                            },
                            'signal_quality': 0.5,
                            'reasoning': 'BTC position imported from Binance - market analysis unavailable',
                        }

                    # ============================================================
                    # STEP 3: Get actual account balance
                    # ============================================================
                    account_balance = self.portfolio_manager.current_capital
                    logger.info(f"[BINANCE] Account capital: ${account_balance:,.2f}")

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
                        ohlc_data=ohlc_data,
                        use_dynamic_management=True,
                        entry_time=datetime.now(),
                        signal_details=signal_details_base,
                        #account_balance=account_balance,
                    )

                    if success:
                        logger.info(f"[SYNC] ✓ Imported {btc_balance:.8f} BTC as LONG position")

                        # Verify VTM status
                        imported_pos = self.portfolio_manager.get_position(asset_name)
                        if imported_pos and imported_pos.trade_manager:
                            vtm_status = imported_pos.get_vtm_status()
                            logger.info(
                                f"[VTM] ✓ VTM ACTIVE for imported BTC position\n"
                                f"      Mode:    {signal_details_base['aggregator_mode'].upper()}\n"
                                f"      Regime:  {signal_details_base['regime_analysis']['regime_type']}\n"
                                f"      Entry:   ${vtm_status['entry_price']:,.2f}\n"
                                f"      SL:      ${vtm_status['stop_loss']:,.2f} (VTM calculated)\n"
                                f"      TP:      ${vtm_status['take_profit']:,.2f} (VTM calculated)"
                            )
                        else:
                            logger.warning(f"[VTM] ⚠️ VTM not initialized for imported BTC position")

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
                        reason="sync_missing_binance",
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
                            reason="sync_quantity_mismatch",
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
