"""
MT5 Execution Handler with Hybrid Position Sizing
INTEGRATED: Automated risk management + manual override support
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


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
                request.asset,
                request.current_price,
                request.signal
            )
            
            # Step 2: Apply confidence adjustments
            confidence_adjusted = self._apply_confidence_adjustment(
                base_size,
                request.confidence_score,
                request.market_condition
            )
            
            # Step 3: Apply manual override if requested
            if request.mode == SizingMode.MANUAL_OVERRIDE and request.manual_size_usd:
                final_size, override_result = self._apply_manual_override(
                    base_size,
                    confidence_adjusted,
                    request.manual_size_usd,
                    request.override_reason,
                    request.max_override_pct
                )
            elif request.mode == SizingMode.REDUCED_RISK:
                final_size = confidence_adjusted * 0.5
                override_result = {
                    "mode": "REDUCED_RISK",
                    "reason": "Lower exposure due to uncertain market conditions",
                    "reduction_pct": 50
                }
            elif request.mode == SizingMode.ELEVATED_RISK:
                final_size = min(
                    confidence_adjusted * 1.5,
                    self._get_max_position_size(request.asset)
                )
                override_result = {
                    "mode": "ELEVATED_RISK",
                    "reason": "Higher exposure for high-conviction trade",
                    "elevation_pct": 50
                }
            else:
                final_size = confidence_adjusted
                override_result = {"mode": "AUTOMATED"}
            
            # Step 4: Apply hard limits
            final_size = self._apply_hard_limits(
                request.asset,
                final_size,
                request.signal
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
        self,
        asset: str,
        current_price: float,
        signal: int
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
        max_size_by_risk = max_risk_usd / stop_loss_pct if stop_loss_pct > 0 else base_size
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
        self,
        base_size: float,
        confidence_score: float,
        market_condition: str
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
            "extreme_volatility": 0.5
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
        max_override_pct: float
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
        
        deviation_pct = ((manual_size_usd - confidence_adjusted) / confidence_adjusted * 100) if confidence_adjusted > 0 else 0
        
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
    
    def _apply_hard_limits(self, asset: str, position_size: float, signal: int) -> float:
        """Apply absolute limits to prevent excessive exposure"""
        
        asset_cfg = self.config["assets"][asset]
        
        # Hard limits
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
        max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
        
        # Check absolute limits
        if position_size < min_size:
            logger.debug(f"Position size ${position_size:,.2f} below minimum ${min_size}")
            return 0.0
        
        position_size = min(position_size, max_size)
        
        # Check portfolio exposure
        current_exposure = self._calculate_current_exposure()
        max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
        if current_exposure + position_size > max_portfolio_usd:
            position_size = max(0, max_portfolio_usd - current_exposure)
            logger.warning(f"Position clamped to portfolio limit: ${position_size:,.2f}")
        
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

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)

        self.asset_config = config["assets"]["GOLD"]
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading", {})

        self.symbol = self.asset_config["symbol"]

        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            logger.warning(
                f"Symbol {self.symbol} not found in MT5. MT5 operations may fail."
            )
        else:
            logger.debug(f"Symbol {self.symbol} info loaded.")

        logger.info("MT5ExecutionHandler with HybridPositionSizer initialized")

        auto_sync_enabled = bool(self.trading_config.get("auto_sync_on_startup", False))
        import_enabled = bool(
            self.config.get("portfolio", {}).get("import_existing_positions", False)
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

    def execute_signal(
        self,
        signal: int,
        symbol: str = None,
        asset: str = "GOLD",
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
    ) -> bool:
        """
        Execute trading signal with hybrid position sizing
        Args:
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            symbol: Trading symbol
            asset: Asset name
            confidence_score: Signal confidence (0.0 to 1.0)
            market_condition: "bullish", "bearish", "neutral", "uncertain"
            sizing_mode: AUTOMATED, MANUAL_OVERRIDE, REDUCED_RISK, ELEVATED_RISK
            manual_size_usd: Manual position size (only if sizing_mode=MANUAL_OVERRIDE)
            override_reason: Reason for override
        Returns:
            True if execution successful, False otherwise
        """
        if symbol is None:
            symbol = self.symbol
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                logger.error(f"{asset}: Failed to get current price")
                return False

            # STEP 1: Get existing position (if any)
            existing_position = self.portfolio_manager.get_position(asset)

            # STEP 2: Handle existing positions
            if existing_position:
                position_side = (
                    existing_position.side
                    if hasattr(existing_position, "side")
                    else existing_position.get("side")
                )
                logger.debug(
                    f"{asset}: Existing {position_side.upper()} position found, signal={signal}"
                )

                # === CLOSE POSITION SCENARIOS ===
                # Close long position on SELL signal
                if signal == -1 and position_side == "long":
                    logger.info(
                        f"[SIGNAL] {asset}: SELL signal received - Closing LONG position"
                    )
                    success = self._close_mt5_position(
                        existing_position, current_price, asset, "sell_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} LONG position")
                        return False
                    return True

                # Close short position on BUY signal
                elif signal == 1 and position_side == "short":
                    logger.info(
                        f"[SIGNAL] {asset}: BUY signal received - Closing SHORT position"
                    )
                    success = self._close_mt5_position(
                        existing_position, current_price, asset, "buy_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset} SHORT position")
                        return False
                    return True

                # HOLD signal - check SL/TP
                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        existing_position, current_price
                    )
                    if should_close:
                        logger.info(f"[AUTO-CLOSE] {asset}: {close_reason}")
                        success = self._close_mt5_position(
                            existing_position, current_price, asset, close_reason
                        )
                        if not success:
                            logger.error(
                                f"[FAIL] Failed to close {asset} position on {close_reason}"
                            )
                            return False
                        return True
                    else:
                        logger.debug(f"{asset}: Holding position, no action needed")
                        return False  # Return False for HOLD signals with no action

                # Prevent duplicate positions
                elif signal == 1 and position_side == "long":
                    logger.info(
                        f"[SKIP] {asset}: BUY signal ignored - Already have LONG position"
                    )
                    return False  # No trade executed

                elif signal == -1 and position_side == "short":
                    logger.info(
                        f"[SKIP] {asset}: SELL signal ignored - Already have SHORT position"
                    )
                    return False  # No trade executed

            # STEP 3: Open new position with HYBRID SIZING
            if signal == 1:  # BUY signal
                if self.portfolio_manager.has_position(asset, side="long"):
                    logger.warning(
                        f"[SKIP] {asset}: BUY signal - Position already exists"
                    )
                    return False  # No trade executed

                logger.info(f"[SIGNAL] {asset}: BUY signal - Opening LONG position")
                return self._open_mt5_position(
                    signal,
                    current_price,
                    symbol,
                    asset,
                    confidence_score,
                    market_condition,
                    sizing_mode,
                    manual_size_usd,
                    override_reason,
                )

            elif signal == -1:  # SELL signal
                logger.debug(f"{asset}: SELL signal - No position to close, no action")
                return False  # No trade executed

            return False  # Default return for any other case

        except Exception as e:
            logger.error(f"Error executing {asset} signal: {e}", exc_info=True)
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

            position_size_usd, sizing_metadata = self.sizer.calculate_size(sizing_request)
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

            # Add position to portfolio manager
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,
                entry_price=execution_price,
                position_size_usd=actual_position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
                mt5_ticket=result.order,
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

    def check_and_update_positions(self, asset: str = "GOLD"):
        """Actively check and update all positions (for HOLD signals)"""
        try:
            position = self.portfolio_manager.get_position(asset)

            if not position:
                return

            current_price = self.get_current_price()

            if current_price == 0:
                logger.warning(f"Could not get price for {asset}")
                return

            should_close, reason = self._check_stop_loss_take_profit(
                position, current_price
            )

            if should_close:
                logger.info(f"[AUTO-CLOSE] {asset}: {reason}")
                self._close_mt5_position(position, current_price, asset, reason)

        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)

    def sync_positions_with_mt5(self, asset: str = "GOLD", symbol: str = None) -> bool:
        """Sync portfolio manager with actual MT5 positions"""

        if symbol is None:
            symbol = self.symbol

        try:
            logger.info(f"[SYNC] Starting position sync for {asset} (MT5)...")

            mt5_positions = mt5.positions_get(symbol=symbol)
            portfolio_position = self.portfolio_manager.get_position(asset)

            if self.symbol_info is None:
                logger.warning(
                    f"[SYNC] Symbol info for {symbol} unavailable; skipping MT5 sync."
                )
                return True

            if mt5_positions and len(mt5_positions) > 0 and not portfolio_position:
                import_enabled = bool(
                    self.config.get("portfolio", {}).get("import_existing_positions", False)
                )
                if import_enabled:
                    logger.info(
                        f"[SYNC] Importing {len(mt5_positions)} MT5 position(s) for {asset}..."
                    )
                    for pos in mt5_positions:
                        pos_type = (
                            "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        )
                        logger.info(
                            f"  → Importing MT5 {pos_type}: entry={pos.price_open:.5f}, "
                            f"current={pos.price_current:.5f}, volume={pos.volume:.2f}, "
                            f"profit={pos.profit:.2f}"
                        )
                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=pos.volume
                            * pos.price_open
                            * self.symbol_info.trade_contract_size,
                            stop_loss=None,
                            take_profit=None,
                            trailing_stop_pct=None,
                        )
                        if success:
                            logger.info(
                                f"[SYNC] ✓ Imported {asset} {pos_type} position"
                            )
                        else:
                            logger.error(
                                f"[SYNC] ✗ Failed to import {asset} {pos_type} position"
                            )
                else:
                    logger.info(
                        f"[SYNC] Detected MT5 positions for {asset} but auto-import is disabled.\n"
                        f"  MT5 Positions Count: {len(mt5_positions)}\n"
                        f"  To enable import on startup, set trading.auto_sync_on_startup = true "
                        f"AND portfolio.import_existing_positions = true in config."
                    )
                    for pos in mt5_positions:
                        pos_type = (
                            "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        )
                        logger.info(
                            f"  → MT5 {pos_type}: entry={pos.price_open:.5f}, "
                            f"current={pos.price_current:.5f}, volume={pos.volume:.2f}, "
                            f"profit={pos.profit:.2f}"
                        )
                return True

            if portfolio_position and (not mt5_positions or len(mt5_positions) == 0):
                logger.warning(
                    f"[SYNC] Portfolio shows {asset} position but MT5 has no matching positions.\n"
                    f"  → Removing from portfolio (likely closed manually in MT5)."
                )
                current_price = self.get_current_price(self.symbol)
                trade_result = self.portfolio_manager.close_position(
                    asset=asset, exit_price=current_price, reason="sync_missing_mt5"
                )
                if trade_result:
                    logger.info(f"[SYNC] ✓ Removed {asset} position from portfolio")
                    return True
                else:
                    logger.error(
                        f"[SYNC] ✗ Failed to remove {asset} position from portfolio"
                    )
                    return False

            if mt5_positions and len(mt5_positions) > 0 and portfolio_position:
                mt5_pos = mt5_positions[0]
                mt5_side = (
                    "long" if mt5_pos.type == mt5.POSITION_TYPE_BUY else "short"
                )
                portfolio_side = getattr(
                    portfolio_position,
                    "side",
                    (
                        portfolio_position.get("side")
                        if isinstance(portfolio_position, dict)
                        else None
                    ),
                )

                if mt5_side != portfolio_side:
                    logger.error(
                        f"[SYNC] MISMATCH: MT5 has {mt5_side.upper()} but portfolio has {portfolio_side}.\n"
                        f"  → Closing portfolio position to avoid mismatch."
                    )
                    current_price = self.get_current_price(self.symbol)
                    self.portfolio_manager.close_position(
                        asset, current_price, "sync_mismatch_mt5"
                    )
                    return True
                else:
                    logger.info(
                        f"[SYNC] ✓ {asset} positions already in sync ({mt5_side.upper()})"
                    )
                    return True

            logger.info(f"[SYNC] ✓ No positions for {asset} in MT5 or portfolio")
            return True

        except Exception as e:
            logger.error(f"[SYNC] Error syncing MT5 positions: {e}", exc_info=True)
            return False