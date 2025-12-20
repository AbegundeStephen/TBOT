"""
MT5 Execution Handler with Hybrid Position Sizing
INTEGRATED: Automated risk management + manual override support
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def count_mt5_positions(symbol: str, side: str = None) -> int:
    """
    Count actual open positions on MT5

    Args:
        symbol: MT5 symbol (e.g., "XAUUSD")
        side: Optional filter - "long" or "short"

    Returns:
        Number of open positions
    """
    try:
        mt5_positions = mt5.positions_get(symbol=symbol)

        if mt5_positions is None:
            return 0

        if side is None:
            return len(mt5_positions)

        # Filter by side
        count = 0
        for pos in mt5_positions:
            pos_side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
            if pos_side == side:
                count += 1

        return count

    except Exception as e:
        logger.error(f"Error counting MT5 positions: {e}")
        return 0


class SizingMode:
    """Position sizing modes"""

    AUTOMATED = "automated"
    MANUAL_OVERRIDE = "override"
    REDUCED_RISK = "reduced_risk"
    ELEVATED_RISK = "elevated"


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
            # Step 1: Calculate base automated size
            base_size = self._calculate_automated_size(
                request.asset, request.current_price, request.signal
            )

            # Step 2: Apply confidence adjustments
            confidence_adjusted = self._apply_confidence_adjustment(
                base_size, request.confidence_score, request.market_condition
            )

            # Step 3: Apply manual override if requested
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

            # Step 4: Apply hard limits
            final_size = self._apply_hard_limits(
                request.asset, final_size, request.signal
            )

            # Build metadata
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
                "timestamp": datetime.now().isoformat(),
            }

            self._log_sizing_decision(metadata)

            if request.mode != SizingMode.AUTOMATED:
                self.override_history.append(metadata)

            return final_size, metadata

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.0, {"error": str(e)}

    def _calculate_automated_size(
        self, asset: str, current_price: float, signal: int
    ) -> float:
        """Calculate base position size using portfolio risk rules"""

        asset_cfg = self.config["assets"][asset]

        # Base: % of capital
        base_pct = self.portfolio_cfg.get("base_position_size", 0.10)
        base_size = self.portfolio_manager.current_capital * base_pct

        # Apply asset weight
        asset_weight = asset_cfg.get("weight", 1.0)
        base_size *= asset_weight

        # Apply signal adjustment
        if signal == -1:  # SELL signal
            base_size *= 0.8

        # Apply max risk per trade
        max_risk_pct = self.risk_cfg.get("max_risk_per_trade", 0.02)
        max_risk_usd = self.portfolio_manager.current_capital * max_risk_pct

        # Estimate SL distance
        risk_cfg = asset_cfg.get("risk", {})
        stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.02)

        # Max size based on risk
        max_size_by_risk = (
            max_risk_usd / stop_loss_pct if stop_loss_pct > 0 else base_size
        )
        base_size = min(base_size, max_size_by_risk)

        # Enforce min/max
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        base_size = max(min_size, min(base_size, max_size))

        logger.debug(
            f"{asset}: Automated base size = ${base_size:,.2f} "
            f"(weight={asset_weight}, signal={signal})"
        )

        return base_size

    def _apply_confidence_adjustment(
        self, base_size: float, confidence_score: float, market_condition: str
    ) -> float:
        """Adjust size based on signal confidence and market conditions"""

        # Confidence scaling: 0.3 to 1.5x
        confidence_scalar = 0.5 + (confidence_score * 1.0)
        confidence_scalar = max(0.3, min(1.5, confidence_scalar))
        adjusted_size = base_size * confidence_scalar

        # Market condition adjustments
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

        # Validate override is within reasonable bounds
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

        # Hard limits
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
        max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)

        # Check absolute limits
        if position_size < min_size:
            logger.debug(
                f"Position size ${position_size:,.2f} below minimum ${min_size}"
            )
            return 0.0

        position_size = min(position_size, max_size)

        # Check portfolio exposure
        current_exposure = self._calculate_current_exposure()
        max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
        if current_exposure + position_size > max_portfolio_usd:
            position_size = max(0, max_portfolio_usd - current_exposure)
            logger.warning(
                f"Position clamped to portfolio limit: ${position_size:,.2f}"
            )

        # Check single asset limit
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


class MT5ExecutionHandler:
    """
    MT5 Execution Handler with Hybrid Position Sizing
    """

    def __init__(self, config: Dict, portfolio_manager, data_manager=None):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)

        self.asset_config = config["assets"]["GOLD"]
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading", {})
        self.mode = self.trading_config.get("mode", "paper")

        self.symbol = config["assets"]["GOLD"]["symbol"]
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )

        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            logger.warning(
                f"Symbol {self.symbol} not found in MT5. MT5 operations may fail."
            )
        else:
            logger.debug(f"Symbol {self.symbol} info loaded.")

        logger.info("MT5ExecutionHandler with HybridPositionSizer initialized")

        auto_sync_enabled = bool(self.trading_config.get("auto_sync_on_startup", True))
        import_enabled = bool(
            self.config.get("portfolio", {}).get("import_existing_positions", True)
        )

        if auto_sync_enabled and import_enabled:
            logger.info("[INIT] Auto-syncing positions with MT5...")
            self.sync_positions_with_mt5("GOLD")

    def get_current_price(self, symbol: str = None) -> float:
        """Get current market price"""
        if symbol is None:
            symbol = self.symbol

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0

        return (tick.ask + tick.bid) / 2

    def can_open_mt5_position(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Check if we can open a new position
        Checks BOTH portfolio manager AND actual MT5 positions

        Returns:
            (can_open: bool, reason: str)
        """
        # Check 1: Portfolio manager limits
        can_open_pm, pm_reason = self.portfolio_manager.can_open_position("GOLD", side)
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        # Check 2: Actual MT5 positions (CRITICAL CHECK)
        mt5_count = count_mt5_positions(symbol, side)

        if mt5_count >= self.max_positions_per_asset:
            logger.warning(
                f"[MT5] Cannot open {side.upper()} position: "
                f"{mt5_count}/{self.max_positions_per_asset} positions already open on MT5"
            )
            return (
                False,
                f"MT5 has {mt5_count}/{self.max_positions_per_asset} {side.upper()} positions open",
            )

        # Check 3: Verify portfolio and MT5 are in sync
        portfolio_count = self.portfolio_manager.get_asset_position_count("GOLD", side)
        if portfolio_count != mt5_count:
            logger.warning(
                f"[SYNC] Position count mismatch: Portfolio={portfolio_count}, MT5={mt5_count}"
            )
            # Don't block, but log the discrepancy

        return True, f"OK - {mt5_count}/{self.max_positions_per_asset} positions open"

    def execute_signal(
        self,
        signal: int,
        symbol: str = None,
        asset_name: str = "GOLD",
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = "automated",
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> bool:
        """
        ✅  Execute signal with proper multi-position handling

        CRITICAL FIXES:
        1. SELL signal now closes ALL long positions (not just first)
        2. BUY signal now closes ALL short positions (not just first)
        3. After closing, verifies sync with MT5
        """
        try:
            if symbol is None:
                symbol = self.symbol

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                logger.error(f"{asset_name}: Failed to get current price")
                return False

            # Get ALL existing positions for this asset
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)

            # Log current state
            long_count = len([p for p in existing_positions if p.side == "long"])
            short_count = len([p for p in existing_positions if p.side == "short"])

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} Signal: {signal}\n"
                f"[STATE] Current Positions: {long_count} LONG, {short_count} SHORT\n"
                f"{'='*80}"
            )

            # === SCENARIO 1: SELL SIGNAL - Close ALL long positions ===
            if signal == -1:
                long_positions = [p for p in existing_positions if p.side == "long"]

                if long_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📉 SELL SIGNAL - Closing ALL {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    failed_count = 0

                    # ✅ FIX: Close ALL long positions, not just first
                    for i, position in enumerate(long_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(long_positions)}] Closing LONG position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  MT5 Ticket:  {position.mt5_ticket}\n"
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

                    # ✅ FIX: Verify sync with MT5 after closing
                    self._verify_position_sync(asset_name, symbol)

                    return closed_count > 0

                else:
                    logger.debug(
                        f"{asset_name}: SELL signal but no LONG positions to close"
                    )
                    return False

            # === SCENARIO 2: BUY SIGNAL - Close ALL short positions, then open long ===
            elif signal == 1:
                short_positions = [p for p in existing_positions if p.side == "short"]

                # Close ALL short positions first
                if short_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Closing ALL {len(short_positions)} SHORT position(s) first\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    # ✅ FIX: Close ALL short positions, not just first
                    for i, position in enumerate(short_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(short_positions)}] Closing SHORT position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  MT5 Ticket:  {position.mt5_ticket}"
                        )

                        success = self._close_position(
                            position=position,
                            current_price=current_price,
                            asset_name=asset_name,
                            reason="buy_signal",
                        )

                        if success:
                            closed_count += 1

                    logger.info(
                        f"  ✓ Closed {closed_count}/{len(short_positions)} SHORT positions"
                    )

                # Check if we can open new LONG position
                can_open, reason = self.can_open_mt5_position(symbol, "long")

                if not can_open:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"⚠️  CANNOT OPEN NEW LONG POSITION\n"
                        f"Reason: {reason}\n"
                        f"{'='*80}\n"
                    )

                    # ✅ FIX: Verify sync even if we can't open
                    self._verify_position_sync(asset_name, symbol)
                    return False

                logger.info(
                    f"\n{'='*80}\n"
                    f"📈 BUY SIGNAL - Opening new LONG position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )

                success = self._open_mt5_position(
                    signal=signal,
                    current_price=current_price,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                )

                # ✅ FIX: Verify sync after opening
                if success:
                    self._verify_position_sync(asset_name, symbol)

                return success

            # === SCENARIO 3: HOLD SIGNAL ===
            elif signal == 0:
                if not existing_positions:
                    logger.debug(f"{asset_name}: HOLD signal, no positions")
                    return False

                # Check SL/TP for all positions
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

                # ✅ FIX: Verify sync if positions were closed
                if positions_closed:
                    self._verify_position_sync(asset_name, symbol)

                return positions_closed

            return False

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False

    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
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
            mt5_ticket = position.mt5_ticket

            position_size_usd = quantity * entry_price

            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset_name} {side.upper()} ({position_id})\n"
                f"  Entry: ${entry_price:,.2f} → Exit: ${current_price:,.2f}\n"
                f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
                f"  Reason: {reason}"
            )

            # Close on MT5 first (if live mode)
            mt5_closed = False
            if self.mode.lower() != "paper" and mt5_ticket:
                mt5_closed = self._close_mt5_order(mt5_ticket, asset_name, side)
            else:
                mt5_closed = True  # Paper mode or no ticket

            # Close in portfolio manager
            trade_result = self.portfolio_manager.close_position(
                position_id=position_id, exit_price=current_price, reason=reason
            )

            if trade_result and mt5_closed:
                logger.info(f"  ✓ Position {position_id} closed successfully")
                return True
            else:
                logger.error(f"  ✗ Failed to close position {position_id}")
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def _close_mt5_order(self, ticket: int, asset: str, side: str) -> bool:
        """
        Close a specific MT5 order by ticket

        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the position by ticket
            mt5_positions = mt5.positions_get(ticket=ticket)

            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(f"[MT5] Position ticket {ticket} not found")
                return False

            mt5_position = mt5_positions[0]

            # Determine close order type
            order_type = (
                mt5.ORDER_TYPE_SELL
                if mt5_position.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            # Get current price
            tick = mt5.symbol_info_tick(mt5_position.symbol)
            close_price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            # Build close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_position.symbol,
                "volume": mt5_position.volume,
                "type": order_type,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Close_{asset}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MT5] ✓ Closed ticket {ticket} @ ${close_price:,.2f}")
                return True
            else:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(
                    f"[MT5] ✗ Failed to close ticket {ticket}: {error_msg} (code: {error_code})"
                )
                return False

        except Exception as e:
            logger.error(f"[MT5] Error closing ticket {ticket}: {e}")
            return False

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit"""
        try:
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                stop_loss = position.stop_loss
                take_profit = position.take_profit
                side = position.side
            else:
                entry_price = position.get("entry_price")
                stop_loss = position.get("stop_loss")
                take_profit = position.get("take_profit")
                side = position.get("side")

            price_tolerance = 0.01

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

            else:  # short
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}", exc_info=True)
            return False, ""

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit"""
        try:
            entry_price = position.entry_price
            stop_loss = position.stop_loss
            take_profit = position.take_profit
            side = position.side

            price_tolerance = 0.01

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

            else:  # short
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}", exc_info=True)
            return False, ""

    def _close_mt5_position(
        self, position, current_price: float, asset: str, reason: str
    ) -> bool:
        """Close existing MT5 position"""
        try:
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                quantity = position.quantity
                side = position.side
                symbol = position.symbol if hasattr(position, "symbol") else self.symbol
            else:
                entry_price = position["entry_price"]
                quantity = position.get("quantity", 0)
                side = position["side"]
                symbol = position.get("symbol", self.symbol)

            contract_size = self.symbol_info.trade_contract_size
            position_size_usd = quantity * entry_price

            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset} {side.upper()} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )

            mt5_positions = mt5.positions_get(symbol=symbol)

            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(
                    f"{asset}: No MT5 position found, closing in portfolio only"
                )
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=reason
                )
                return True

            mt5_position = None
            for pos in mt5_positions:
                pos_type = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                if pos_type == side:
                    mt5_position = pos
                    break

            if mt5_position is None:
                logger.warning(
                    f"{asset}: MT5 {side} position not found, closing in portfolio only"
                )
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=reason
                )
                return True

            order_type = (
                mt5.ORDER_TYPE_SELL
                if mt5_position.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            tick = mt5.symbol_info_tick(symbol)
            close_price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": mt5_position.volume,
                "type": order_type,
                "position": mt5_position.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=close_price, reason=reason
                )

                logger.info(
                    f"[OK] {asset} {side.upper()} position closed on MT5 and portfolio"
                )
                return True
            else:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(
                    f"[FAIL] Failed to close MT5 position: {error_msg} (code: {error_code})"
                )

                logger.warning(f"Closing {asset} in portfolio despite MT5 failure")
                self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason=f"{reason}_mt5_failed"
                )
                return False

        except Exception as e:
            logger.error(f"Error closing MT5 position: {e}", exc_info=True)
            return False

    def _open_mt5_position(
        self,
        signal: int,
        current_price: float,
        symbol: str,
        asset: str,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> bool:
        """Open new MT5 position with hybrid sizing"""
        try:
            can_open, reason = self.portfolio_manager.can_open_position(
                asset, "long" if signal == 1 else "short"
            )
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset} position: {reason}")
                return True

            # ========== HYBRID POSITION SIZING ==========
            sizing_request = PositionSizingRequest(
                asset=asset,
                current_price=current_price,
                signal=signal,
                mode=sizing_mode,
                manual_size_usd=manual_size_usd,
                confidence_score=confidence_score,
                market_condition=market_condition or "neutral",
                override_reason=override_reason,
                max_override_pct=2.0,
            )

            position_size_usd, sizing_metadata = self.sizer.calculate_size(
                sizing_request
            )
            # ==========================================

            if position_size_usd <= 0:
                logger.error(f"{asset}: Invalid position size calculated")
                return False

            # Calculate volume in lots
            contract_size = self.symbol_info.trade_contract_size
            volume_lots = position_size_usd / (current_price * contract_size)

            volume_step = self.symbol_info.volume_step
            volume_lots = round(volume_lots / volume_step) * volume_step

            volume_lots = max(self.symbol_info.volume_min, volume_lots)
            volume_lots = min(self.symbol_info.volume_max, volume_lots)

            actual_position_size = volume_lots * current_price * contract_size

            # Determine order type and side
            order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            position_side = "long" if signal == 1 else "short"

            # Calculate SL/TP
            risk = self.asset_config.get("risk", {})
            stop_loss_pct = risk.get("stop_loss_pct", 0.03)
            take_profit_pct = risk.get("take_profit_pct", 0.08)
            trailing_stop_pct = risk.get("trailing_stop_pct", 0.02)

            if signal == 1:  # BUY/Long
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:  # SELL/Short
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)

            logger.info(
                f"[OPEN] {'BUY' if signal == 1 else 'SELL'} {volume_lots:.2f} lots {symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${actual_position_size:,.2f}")
            logger.info(f"   Mode: {sizing_mode} | Confidence: {confidence_score}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")

            tick = mt5.symbol_info_tick(symbol)
            execution_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_lots,
                "type": order_type,
                "price": execution_price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Signal_{signal}_{asset}_{sizing_mode}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(f"[FAIL] Order failed: {error_msg} (code: {error_code})")
                return False

            try:
                if self.data_manager is None:
                    logger.warning("[VTM] data_manager not available, VTM disabled")
                    ohlc_data = None
                else:
                    # Fetch recent OHLC data for VTM
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)  # Get 10 days of data

                    df = self.data_manager.fetch_mt5_data(
                        symbol=symbol,
                        timeframe=self.config["assets"][asset].get("timeframe", "H1"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    # Prepare OHLC arrays for VTM
                    ohlc_data = {
                        "high": df["high"].values,
                        "low": df["low"].values,
                        "close": df["close"].values,
                    }

                    logger.debug(
                        f"[VTM] Prepared {len(df)} bars for dynamic management"
                    )

            except Exception as e:
                logger.warning(
                    f"[VTM] Failed to fetch OHLC data: {e}, VTM disabled for this trade"
                )
                ohlc_data = None

            # Add position to portfolio manager
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,
                entry_price=execution_price,
                position_size_usd=actual_position_size,
                stop_loss=None,
                take_profit=None,
                trailing_stop_pct=trailing_stop_pct,
                mt5_ticket=result.order,
                ohlc_data=ohlc_data,  # ✨ NEW: Pass OHLC for VTM
                use_dynamic_management=True,  # ✨ NEW: Enable VTM
            )

            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset} position")
                logger.warning("Attempting to close unwanted MT5 position...")
                self._emergency_close_mt5_position(symbol, volume_lots, order_type)
                return False

            logger.info(
                f"[OK] {'BUY' if signal == 1 else 'SELL'} {asset} - Position opened successfully"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening MT5 position: {e}", exc_info=True)
            return False

    def _verify_position_sync(self, asset_name: str, symbol: str):
        """
        ✅ NEW: Verify portfolio and MT5 are in sync after trade execution

        This is called after:
        - Opening new positions
        - Closing positions
        - Signal execution

        Logs detailed comparison and triggers re-sync if needed
        """
        try:
            import MetaTrader5 as mt5

            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            portfolio_long = len([p for p in portfolio_positions if p.side == "long"])
            portfolio_short = len([p for p in portfolio_positions if p.side == "short"])

            # Get MT5 positions
            mt5_positions = mt5.positions_get(symbol=symbol)
            mt5_long = 0
            mt5_short = 0

            if mt5_positions:
                for pos in mt5_positions:
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        mt5_long += 1
                    else:
                        mt5_short += 1

            # Compare
            sync_ok = (portfolio_long == mt5_long) and (portfolio_short == mt5_short)

            logger.info(
                f"\n{'='*80}\n"
                f"[SYNC CHECK] {asset_name}\n"
                f"{'='*80}\n"
                f"Portfolio:  {portfolio_long} LONG, {portfolio_short} SHORT (Total: {len(portfolio_positions)})\n"
                f"MT5:        {mt5_long} LONG, {mt5_short} SHORT (Total: {len(mt5_positions) if mt5_positions else 0})\n"
                f"Status:     {'✅ IN SYNC' if sync_ok else '⚠️  OUT OF SYNC'}\n"
                f"{'='*80}"
            )

            # If out of sync, trigger re-sync
            if not sync_ok:
                logger.warning(
                    f"[SYNC] Mismatch detected! Triggering automatic re-sync..."
                )
                self.sync_positions_with_mt5(asset_name, symbol)

            return sync_ok

        except Exception as e:
            logger.error(f"[SYNC CHECK] Error: {e}")
            return False

    def check_and_update_positions_VTM(self, asset_name: str = "GOLD"):
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

    def _emergency_close_mt5_position(
        self, symbol: str, volume: float, original_order_type: int
    ):
        """Emergency close of an unwanted MT5 position"""
        try:
            close_order_type = (
                mt5.ORDER_TYPE_SELL
                if original_order_type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )
            tick = mt5.symbol_info_tick(symbol)
            close_price = (
                tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_order_type,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Emergency_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("[OK] Emergency close successful")
            else:
                logger.error(
                    f"[FAIL] Emergency close failed: {result.comment if result else 'No result'}"
                )
        except Exception as e:
            logger.error(f"Emergency close error: {e}", exc_info=True)

    def check_and_update_positions(self, asset_name: str = "GOLD"):
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

    def sync_positions_with_mt5(self, asset: str = "GOLD", symbol: str = None) -> bool:
        """
        ✅  Check config settings correctly for position import
        """
        if symbol is None:
            symbol = self.symbol

        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"[SYNC] Starting position sync for {asset}")
            logger.info(f"{'='*80}")

            # Get MT5 positions
            mt5_positions = mt5.positions_get(symbol=symbol)
            mt5_count = len(mt5_positions) if mt5_positions else 0

            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset)
            portfolio_count = len(portfolio_positions)

            # Count by side
            mt5_long = sum(
                1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_BUY
            )
            mt5_short = sum(
                1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_SELL
            )
            portfolio_long = sum(1 for p in portfolio_positions if p.side == "long")
            portfolio_short = sum(1 for p in portfolio_positions if p.side == "short")

            logger.info(f"[SYNC] Position Count Comparison:")
            logger.info(
                f"  MT5:       {mt5_long} LONG, {mt5_short} SHORT (Total: {mt5_count})"
            )
            logger.info(
                f"  Portfolio: {portfolio_long} LONG, {portfolio_short} SHORT (Total: {portfolio_count})"
            )

            if mt5_positions:
                logger.info(f"\n[SYNC] MT5 Positions:")
                for i, pos in enumerate(mt5_positions, 1):
                    side = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                    logger.info(
                        f"  [{i}] {side} | Ticket: {pos.ticket} | "
                        f"Entry: ${pos.price_open:,.2f} | Volume: {pos.volume:.2f}"
                    )

            if portfolio_positions:
                logger.info(f"\n[SYNC] Portfolio Positions:")
                for i, pos in enumerate(portfolio_positions, 1):
                    logger.info(
                        f"  [{i}] {pos.side.upper()} | ID: {pos.position_id} | "
                        f"Ticket: {pos.mt5_ticket} | Entry: ${pos.entry_price:,.2f}"
                    )

            if self.symbol_info is None:
                logger.warning(
                    f"[SYNC] Symbol info for {symbol} unavailable; skipping MT5 sync."
                )
                return True

            logger.info(
                f"[SYNC] Found {mt5_count} MT5 position(s) and {portfolio_count} portfolio position(s)"
            )

            # ================================================================
            # SCENARIO 1: MT5 has positions, portfolio is empty → IMPORT
            # ================================================================
            if mt5_count > 0 and portfolio_count == 0:
                # ✅ FIX: Check config correctly (use self.config, not a different object)
                import_enabled = bool(
                    self.config.get("portfolio", {}).get(
                        "import_existing_positions", False
                    )
                )

                # ✅ DEBUG: Log what we found in config
                logger.info(f"[SYNC] Config check:")
                logger.info(f"  portfolio.import_existing_positions = {import_enabled}")
                logger.info(
                    f"  trading.auto_sync_on_startup = {self.config.get('trading', {}).get('auto_sync_on_startup', False)}"
                )

                if import_enabled:
                    logger.info(
                        f"[SYNC] ✅ Import ENABLED - Importing {mt5_count} MT5 position(s) WITH VTM..."
                    )

                    # ✅ Fetch OHLC data ONCE for all imports
                    ohlc_data = None
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = self.data_manager.fetch_mt5_data(
                            symbol=symbol,
                            timeframe=self.config["assets"][asset].get(
                                "timeframe", "H1"
                            ),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                            logger.info(
                                f"[VTM] ✅ Fetched {len(df)} bars for dynamic management"
                            )
                        else:
                            logger.warning(
                                f"[VTM] ⚠️ Insufficient data ({len(df)} bars), VTM disabled"
                            )

                    except Exception as e:
                        logger.error(f"[VTM] ❌ Failed to fetch OHLC: {e}")
                        ohlc_data = None

                    imported_count = 0
                    for pos in mt5_positions:
                        pos_type = (
                            "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        )

                        logger.info(
                            f"\n  → Importing MT5 {pos_type}: ticket={pos.ticket}, "
                            f"entry=${pos.price_open:.2f}, current=${pos.price_current:.2f}"
                        )

                        # Check if we can import
                        can_import, reason = self.portfolio_manager.can_open_position(
                            asset, pos_type
                        )
                        if not can_import:
                            logger.warning(f"[SYNC] ⚠️ Cannot import position: {reason}")
                            continue

                        # ✅ Import with VTM support
                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=(
                                pos.volume
                                * pos.price_open
                                * self.symbol_info.trade_contract_size
                            ),
                            stop_loss=pos.sl if pos.sl > 0 else None,
                            take_profit=pos.tp if pos.tp > 0 else None,
                            trailing_stop_pct=self.config["assets"][asset]
                            .get("risk", {})
                            .get("trailing_stop_pct"),
                            mt5_ticket=pos.ticket,
                            ohlc_data=ohlc_data,  # ✅ Pass OHLC for VTM
                            use_dynamic_management=True,  # ✅ Enable VTM
                            entry_time=datetime.fromtimestamp(
                                pos.time
                            ),  # ✅ Preserve entry time
                        )

                        if success:
                            imported_count += 1

                            # Get the imported position to check VTM status
                            imported_positions = (
                                self.portfolio_manager.get_asset_positions(asset)
                            )
                            if imported_positions:
                                imported_pos = imported_positions[-1]  # Get last added
                                if imported_pos.trade_manager:
                                    logger.info(
                                        f"[VTM] ✅ VTM ACTIVE for imported position\n"
                                        f"      Ticket: {pos.ticket}\n"
                                        f"      Entry: ${imported_pos.entry_price:,.2f}\n"
                                        f"      SL: ${imported_pos.stop_loss:,.2f}\n"
                                        f"      TP: ${imported_pos.take_profit:,.2f}"
                                    )
                                else:
                                    logger.warning(
                                        f"[VTM] ⚠️ VTM not initialized for ticket {pos.ticket}"
                                    )
                        else:
                            logger.error(
                                f"[SYNC] ❌ Failed to import {asset} {pos_type} position"
                            )

                    logger.info(
                        f"\n{'='*80}\n"
                        f"[SYNC] Import complete: {imported_count}/{mt5_count} positions imported\n"
                        f"{'='*80}"
                    )

                    # Verify VTM status after import
                    if imported_count > 0:
                        self._verify_vtm_status_after_sync(asset)

                    return True

                else:
                    # ✅ Clearer message when import is disabled
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"[SYNC] ⚠️ IMPORT DISABLED IN CONFIG\n"
                        f"{'='*80}\n"
                        f"Found {mt5_count} MT5 position(s) but import is disabled.\n"
                        f"\n"
                        f"Current settings:\n"
                        f"  portfolio.import_existing_positions = {import_enabled}\n"
                        f"  trading.auto_sync_on_startup = {self.config.get('trading', {}).get('auto_sync_on_startup', False)}\n"
                        f"\n"
                        f"These positions will NOT be managed by the bot.\n"
                        f"Bot will only manage NEW positions it opens.\n"
                        f"{'='*80}"
                    )

                    # Show what's on MT5
                    for pos in mt5_positions:
                        pos_type = (
                            "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        )
                        logger.info(
                            f"  → MT5 {pos_type}: ticket={pos.ticket}, entry=${pos.price_open:.2f}, "
                            f"current=${pos.price_current:.2f}, volume={pos.volume:.2f}, profit=${pos.profit:.2f}"
                        )

                    return True

            # ================================================================
            # SCENARIO 2: Portfolio has positions, MT5 is empty → CLOSE ALL
            # ================================================================
            if portfolio_count > 0 and mt5_count == 0:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"[SYNC] ⚠️ POSITION MISMATCH\n"
                    f"{'='*80}\n"
                    f"Portfolio shows {portfolio_count} position(s) but MT5 has 0.\n"
                    f"Positions were likely closed manually in MT5.\n"
                    f"Removing positions from portfolio...\n"
                    f"{'='*80}"
                )

                current_price = self.get_current_price(self.symbol)
                closed_count = 0

                for position in portfolio_positions:
                    trade_result = self.portfolio_manager.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        reason="sync_missing_mt5",
                    )
                    if trade_result:
                        closed_count += 1
                        logger.info(f"  ✅ Removed position {position.position_id}")

                logger.info(
                    f"\n[SYNC] Cleanup complete: {closed_count}/{portfolio_count} positions removed\n"
                )
                return closed_count == portfolio_count

            # ================================================================
            # SCENARIO 3: Both have positions → VALIDATE & RECONCILE
            # ================================================================
            if mt5_count > 0 and portfolio_count > 0:
                logger.info(
                    f"[SYNC] Validating {portfolio_count} portfolio vs {mt5_count} MT5 positions..."
                )

                # Build MT5 position map
                mt5_by_ticket = {pos.ticket: pos for pos in mt5_positions}

                # Check portfolio positions
                positions_to_remove = []
                for pos in portfolio_positions:
                    if pos.mt5_ticket and pos.mt5_ticket not in mt5_by_ticket:
                        logger.warning(
                            f"[SYNC] ⚠️ Portfolio position {pos.position_id} (ticket={pos.mt5_ticket}) "
                            f"not found in MT5 → Marking for removal"
                        )
                        positions_to_remove.append(pos)

                # Remove orphaned positions
                if positions_to_remove:
                    current_price = self.get_current_price(self.symbol)
                    for pos in positions_to_remove:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="sync_mt5_ticket_missing",
                        )
                        logger.info(f"  ✅ Removed orphaned position {pos.position_id}")

                # Final validation
                remaining_portfolio_count = (
                    self.portfolio_manager.get_asset_position_count(asset)
                )

                if remaining_portfolio_count == mt5_count:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"[SYNC] ✅ {asset} positions in sync ({remaining_portfolio_count} positions)\n"
                        f"{'='*80}"
                    )
                    self._verify_vtm_status_after_sync(asset)
                    return True
                else:
                    logger.warning(
                        f"[SYNC] ⚠️ Count mismatch after sync: Portfolio={remaining_portfolio_count}, MT5={mt5_count}"
                    )
                    return False

            # ================================================================
            # SCENARIO 4: Both empty → All good
            # ================================================================
            logger.info(f"[SYNC] ✅ No positions for {asset} in MT5 or portfolio")
            return True

        except Exception as e:
            logger.error(f"[SYNC] ❌ Error syncing MT5 positions: {e}", exc_info=True)
            return False

    def _verify_vtm_status_after_sync(self, asset: str):
        """Verify VTM is working after position sync"""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset)

            if not positions:
                return

            logger.info(f"\n{'='*80}")
            logger.info(f"[VTM VERIFICATION] Checking {len(positions)} position(s)...")
            logger.info(f"{'='*80}")

            vtm_active_count = 0
            vtm_missing_count = 0

            for pos in positions:
                if pos.trade_manager:
                    vtm_active_count += 1
                    try:
                        status = pos.get_vtm_status()
                        logger.info(
                            f"\n✅ {pos.position_id}: VTM ACTIVE\n"
                            f"   Ticket:  {pos.mt5_ticket}\n"
                            f"   Entry:   ${status['entry_price']:,.2f}\n"
                            f"   Current: ${status['current_price']:,.2f}\n"
                            f"   P&L:     {status['pnl_pct']:+.2f}%\n"
                            f"   SL:      ${status['stop_loss']:,.2f}\n"
                            f"   TP:      ${status.get('take_profit', 0):,.2f}\n"
                            f"   Locked:  {'Yes' if status['profit_locked'] else 'No'}"
                        )
                    except Exception as e:
                        logger.error(f"   Error getting VTM status: {e}")
                else:
                    vtm_missing_count += 1
                    logger.warning(
                        f"\n⚠️ {pos.position_id}: VTM NOT ACTIVE\n"
                        f"   Ticket:  {pos.mt5_ticket}\n"
                        f"   Entry:   ${pos.entry_price:,.2f}\n"
                        f"   → Using static SL/TP instead"
                    )

            logger.info(
                f"\n{'='*80}\n"
                f"[VTM VERIFICATION] Summary:\n"
                f"  Active:  {vtm_active_count}/{len(positions)}\n"
                f"  Missing: {vtm_missing_count}/{len(positions)}\n"
                f"{'='*80}\n"
            )

        except Exception as e:
            logger.error(f"[VTM VERIFICATION] ❌ Error: {e}")
