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
from src.global_error_handler import handle_errors, ErrorSeverity
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
        self.error_handler = None
        self.trading_bot = None 


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


    @handle_errors(
    component="mt5_handler",
    severity=ErrorSeverity.ERROR,
    notify=True,
    reraise=False,
    default_return=0.0
)
    def get_current_price(self, symbol: str = None) -> float:
        """Get current market price"""
        if symbol is None:
            symbol = self.symbol

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0

        return (tick.ask + tick.bid) / 2
    
    def can_open_position_side(self, asset_name: str, side: str) -> Tuple[bool, str]:
        """
        Check if we can open a position on a specific SIDE
        
        Args:
            asset_name: Asset name (e.g., "GOLD")
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

        # Check MT5 margin requirements (optional)
        try:
            import MetaTrader5 as mt5
            
            account_info = mt5.account_info()
            if account_info:
                # Check if we have enough margin
                margin_free = account_info.margin_free
                margin_required = 1000  # Rough estimate, adjust based on leverage
                
                if margin_free < margin_required:
                    return False, f"Insufficient margin: ${margin_free:.2f} free"
        
        except Exception as e:
            logger.debug(f"[MT5] Margin check warning: {e}")

        return True, f"OK - {current_count}/{max_per_asset} {side.upper()} positions open"

    @handle_errors(
    component="mt5_handler",
    severity=ErrorSeverity.CRITICAL,
    notify=True,
    reraise=False,
    default_return=False
)
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
        """
        ✅ MT5 TWO-WAY TRADING: Open LONG or SHORT position
        
        MT5 supports both longs and shorts natively - no special setup needed!
        
        Args:
            signal: +1 for LONG, -1 for SHORT
            current_price: Entry price
            symbol: MT5 symbol (e.g., "XAUUSD")
            asset: Asset name (e.g., "GOLD")
            confidence_score: Signal confidence (0-1)
            market_condition: Market regime
            sizing_mode: Position sizing mode
            manual_size_usd: Manual position size override
            override_reason: Reason for manual override
        
        Returns:
            True if position opened successfully, False otherwise
        """
        try:
            import MetaTrader5 as mt5
            
            # ============================================================
            # STEP 1: Determine side from signal
            # ============================================================
            side = "long" if signal == 1 else "short"
            
            logger.info(f"[MT5] Opening {side.upper()} position for {asset}")

            # Check if we can open this position
            can_open, reason = self.portfolio_manager.can_open_position(asset, side)
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset} position: {reason}")
                return False

            # ============================================================
            # STEP 2: Calculate position size
            # ============================================================
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
            
            # Apply short size reduction if configured
            if side == "short":
                short_config = self.config.get("portfolio", {}).get("short_position_sizing", {})
                use_reduced = short_config.get("use_reduced_size_for_shorts", False)
                multiplier = short_config.get("short_size_multiplier", 0.8)
                
                if use_reduced and multiplier < 1.0:
                    logger.info(f"[SHORT] Applying {multiplier}x size reduction")
                    position_size_usd *= multiplier

            if position_size_usd <= 0:
                logger.error(f"{asset}: Invalid position size calculated")
                return False

            # ============================================================
            # STEP 3: Calculate volume in lots (MT5-specific)
            # ============================================================
            contract_size = self.symbol_info.trade_contract_size
            volume_lots = position_size_usd / (current_price * contract_size)

            volume_step = self.symbol_info.volume_step
            volume_lots = round(volume_lots / volume_step) * volume_step

            # Apply min/max limits
            volume_lots = max(self.symbol_info.volume_min, volume_lots)
            volume_lots = min(self.symbol_info.volume_max, volume_lots)

            actual_position_size = volume_lots * current_price * contract_size

            # ============================================================
            # STEP 4: Determine order type (BUY or SELL)
            # ============================================================
            # LONG  → ORDER_TYPE_BUY
            # SHORT → ORDER_TYPE_SELL
            order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
            position_side = side

            # ============================================================
            # STEP 5: Calculate Stop Loss & Take Profit
            # ============================================================
            risk = self.asset_config.get("risk", {})
            
            if side == "long":
                # LONG: SL below entry, TP above entry
                stop_loss_pct = risk.get("stop_loss_pct", 0.03)
                take_profit_pct = risk.get("take_profit_pct", 0.08)
                trailing_stop_pct = risk.get("trailing_stop_pct", 0.02)
                
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            
            else:  # SHORT
                # SHORT: SL above entry, TP below entry (inverted!)
                stop_loss_pct = risk.get("stop_loss_pct_short", risk.get("stop_loss_pct", 0.025))
                take_profit_pct = risk.get("take_profit_pct_short", risk.get("take_profit_pct", 0.07))
                trailing_stop_pct = risk.get("trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.018))
                
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)

            logger.info(
                f"[OPEN] {side.upper()} {volume_lots:.2f} lots {symbol} @ ${current_price:,.2f}\n"
                f"  Size: ${actual_position_size:,.2f}\n"
                f"  Mode: {sizing_mode} | Confidence: {confidence_score}\n"
                f"  SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})\n"
                f"  TP: ${take_profit:,.2f} ({take_profit_pct:.1%})"
            )

            # ============================================================
            # STEP 6: Execute on MT5
            # ============================================================
            mt5_ticket = None
            
            if self.mode.lower() != "paper":
                try:
                    if not self._is_trading_allowed(symbol):
                        logger.error(f"[MT5] ✗ Trading not allowed for {symbol} (market closed or trading disabled)")
                        return False
                    # Get current price for execution
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        logger.error(f"[MT5] ✗ Cannot get tick for {symbol} (market may be closed)")
                        return False
                    
                    execution_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

                    # Build MT5 order request
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": volume_lots,
                        "type": order_type,
                        "price": execution_price,
                        "sl": stop_loss,  # 0 =no SL (VTM manages)
                        "tp": take_profit,# 0 = no TP (VTM manages)
                        "deviation": 20,
                        "magic": 234000,
                        "comment": f"Signal_{signal}_{asset}_{sizing_mode}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    logger.info(f"[MT5] Sending order request:")
                    logger.info(f"  Symbol:  {request['symbol']}")
                    logger.info(f"  Type:    {order_type} ({'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'})")
                    logger.info(f"  Volume:  {request['volume']:.2f} lots")
                    logger.info(f"  Price:   ${request['price']:,.2f}")
                    logger.info(f"  SL:      ${request['sl']:,.2f} (0 = VTM managed)")
                    logger.info(f"  TP:      ${request['tp']:,.2f} (0 = VTM managed)")

                    # Send order to MT5
                    result = mt5.order_send(request)

                    if result is None:
                        last_error = mt5.last_error()
                        logger.error(
                            f"[MT5] ✗ order_send() returned None!\n"
                            f"  Last Error: {last_error}\n"
                            f"  This usually means:\n"
                            f"    - Market is closed\n"
                            f"    - Symbol not available\n"
                            f"    - Connection lost\n"
                            f"  Check MT5 terminal and market hours."
                        )
                        return False
                    
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        error_msg = result.comment if result else "Unknown"
                        error_code = result.retcode if result else "N/A"
                        logger.error(
                            f"[MT5] ✗ {side.upper()} order failed!\n"
                            f"  Error Code: {error_code}\n"
                            f"  Message:    {error_msg}\n"
                            f"  Retcode Meaning: {self._get_retcode_meaning(error_code)}"
                        )
                        return False


                    mt5_ticket = result.order
                    logger.info(f"[MT5] ✓ {side.upper()} order placed: Ticket #{mt5_ticket}")
                
                except Exception as e:
                    logger.error(f"[MT5] Order execution failed: {e}")
                    return False

            # ============================================================
            # STEP 7: Fetch OHLC data for VTM (Veteran Trade Manager)
            # ============================================================
            ohlc_data = None
            
            try:
                if self.data_manager is None:
                    logger.warning("[VTM] data_manager not available, VTM disabled")
                else:
                    from datetime import datetime, timedelta, timezone
                    
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)

                    df = self.data_manager.fetch_mt5_data(
                        symbol=symbol,
                        timeframe=self.config["assets"][asset].get("timeframe", "H1"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    if len(df) > 0:
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                        }
                        logger.debug(f"[VTM] Prepared {len(df)} bars for dynamic management")

            except Exception as e:
                logger.warning(f"[VTM] Failed to fetch OHLC data: {e}, VTM disabled for this trade")

            # ============================================================
            # STEP 8: Add position to Portfolio Manager
            # ============================================================
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,  # ✅ "long" or "short"
                entry_price=execution_price if mt5_ticket else current_price,
                position_size_usd=actual_position_size,
                stop_loss=None,  # VTM will manage this
                take_profit=None,  # VTM will manage this
                trailing_stop_pct=trailing_stop_pct,
                mt5_ticket=mt5_ticket,  # ✅ MT5-specific
                ohlc_data=ohlc_data,
                use_dynamic_management=True,
                signal_details=signal_details,
            )

            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset} position")
                
                # Emergency: Close unwanted MT5 position
                if mt5_ticket:
                    logger.warning(f"[MT5] Attempting to close unwanted position #{mt5_ticket}...")
                    self._emergency_close_mt5_position(symbol, volume_lots, order_type)
                
                return False

            logger.info(
                f"[OK] {side.upper()} {asset} position opened successfully"
            )
            if mt5_ticket:
                logger.info(f"  └─ MT5 Ticket: #{mt5_ticket}")
            if ohlc_data:
                logger.info(f"  └─ VTM: ACTIVE")
            
            return True

        except Exception as e:
            logger.error(f"[MT5] Error opening {asset} position: {e}", exc_info=True)
        return False

    @handle_errors(
    component="mt5_handler",
    severity=ErrorSeverity.CRITICAL,
    notify=True,
    reraise=False,
    default_return=False
)
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
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ MT5 TWO-WAY TRADING: Execute trading signal
        
        Signal Logic:
        - BUY (+1):  Close ALL shorts → Open long
        - SELL (-1): Close ALL longs  → Open short
        - HOLD (0):  Check SL/TP only
        
        Args:
            signal: +1 (BUY), -1 (SELL), 0 (HOLD)
            symbol: MT5 symbol (e.g., "XAUUSD")
            asset_name: Asset name (e.g., "GOLD")
            confidence_score: Signal confidence (0-1)
            market_condition: Market regime
            sizing_mode: Position sizing mode
            manual_size_usd: Manual position size override
            override_reason: Reason for manual override
        
        Returns:
            True if action taken, False otherwise
        """
        try:
            # ============================================================
            # STEP 1: Get current price
            # ============================================================
            if symbol is None:
                symbol = self.symbol

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                logger.error(f"{asset_name}: Failed to get current price")
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
                        f"📉 SELL SIGNAL - Closing ALL {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    failed_count = 0

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

                    # Verify sync with MT5 after closing
                    self._verify_position_sync(asset_name, symbol)

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
                    
                    # Verify sync even if we can't open
                    self._verify_position_sync(asset_name, symbol)
                    return len(long_positions) > 0

                # Step 3: Open SHORT position
                logger.info(
                    f"\n{'='*80}\n"
                    f"📉 SELL SIGNAL - Opening new SHORT position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )

                success = self._open_mt5_position(
                    signal=-1,
                    current_price=current_price,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details
                )

                # Verify sync after opening
                if success:
                    self._verify_position_sync(asset_name, symbol)

                return success

            # ============================================================
            # SCENARIO 2: BUY SIGNAL (+1) → Close shorts, Open long
            # ============================================================
            elif signal == 1:
                # Step 1: Close ALL short positions
                if short_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Closing ALL {len(short_positions)} SHORT position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    failed_count = 0

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
                            logger.info(f"  ✓ Position {position.position_id} closed")
                        else:
                            failed_count += 1
                            logger.error(f"  ✗ Failed to close {position.position_id}")

                    logger.info(
                        f"\n{'='*80}\n"
                        f"CLOSE SUMMARY: {closed_count} closed, {failed_count} failed\n"
                        f"{'='*80}\n"
                    )

                    # Verify sync with MT5 after closing
                    self._verify_position_sync(asset_name, symbol)

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

                    # Verify sync even if we can't open
                    self._verify_position_sync(asset_name, symbol)
                    return len(short_positions) > 0

                # Step 3: Open LONG position
                logger.info(
                    f"\n{'='*80}\n"
                    f"📈 BUY SIGNAL - Opening new LONG position\n"
                    f"Check: {reason}\n"
                    f"{'='*80}\n"
                )

                success = self._open_mt5_position(
                    signal=1,
                    current_price=current_price,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                )

                # Verify sync after opening
                if success:
                    self._verify_position_sync(asset_name, symbol)

                return success

            # ============================================================
            # SCENARIO 3: HOLD SIGNAL (0) → Check SL/TP
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

                # Verify sync if positions were closed
                if positions_closed:
                    self._verify_position_sync(asset_name, symbol)

                if not positions_closed:
                    logger.debug(f"{asset_name}: All positions holding")

                return positions_closed

            return False

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False


    # ============================================================================
    # HELPER METHOD - Position sync verification (already in your code)
    # ============================================================================

    def _verify_position_sync(self, asset_name: str, symbol: str):
        """
        ✅ Verify portfolio and MT5 are in sync after trade execution
        
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
                logger.warning(f"[SYNC] Mismatch detected! Triggering automatic re-sync...")
                self.sync_positions_with_mt5(asset_name, symbol)

            return sync_ok

        except Exception as e:
            logger.error(f"[SYNC CHECK] Error: {e}")
            return False
        
    @handle_errors(
    component="mt5_handler",
    severity=ErrorSeverity.CRITICAL,
    notify=True,
    reraise=False,
    default_return=False
)
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
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ MT5 TWO-WAY TRADING with VTM Dynamic Management
        
        CHANGES:
        - Stop Loss = None (VTM will calculate based on market structure)
        - Take Profit = None (VTM will calculate based on market structure)
        - Trailing Stop = Managed by VTM (not MT5 native)
        
        Args:
            signal: +1 for LONG, -1 for SHORT
            current_price: Entry price
            symbol: MT5 symbol (e.g., "XAUUSD")
            asset: Asset name (e.g., "GOLD")
            confidence_score: Signal confidence (0-1)
            market_condition: Market regime
            sizing_mode: Position sizing mode
            manual_size_usd: Manual position size override
            override_reason: Reason for manual override
            signal_details: Hybrid aggregator context
        
        Returns:
            True if position opened successfully, False otherwise
        """
        try:
            import MetaTrader5 as mt5
            
            # ============================================================
            # STEP 1: Determine side from signal
            # ============================================================
            side = "long" if signal == 1 else "short"
            
            logger.info(f"[MT5] Opening {side.upper()} position for {asset}")

            # Check if we can open this position
            can_open, reason = self.portfolio_manager.can_open_position(asset, side)
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset} position: {reason}")
                return False

            # ============================================================
            # STEP 2: Calculate position size
            # ============================================================
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
            
            # Apply short size reduction if configured
            if side == "short":
                short_config = self.config.get("portfolio", {}).get("short_position_sizing", {})
                use_reduced = short_config.get("use_reduced_size_for_shorts", False)
                multiplier = short_config.get("short_size_multiplier", 0.8)
                
                if use_reduced and multiplier < 1.0:
                    logger.info(f"[SHORT] Applying {multiplier}x size reduction")
                    position_size_usd *= multiplier

            if position_size_usd <= 0:
                logger.error(f"{asset}: Invalid position size calculated")
                return False

            # ============================================================
            # STEP 3: Calculate volume in lots (MT5-specific)
            # ============================================================
            contract_size = self.symbol_info.trade_contract_size
            volume_lots = position_size_usd / (current_price * contract_size)

            volume_step = self.symbol_info.volume_step
            volume_lots = round(volume_lots / volume_step) * volume_step

            # Apply min/max limits
            volume_lots = max(self.symbol_info.volume_min, volume_lots)
            volume_lots = min(self.symbol_info.volume_max, volume_lots)

            actual_position_size = volume_lots * current_price * contract_size

            # ============================================================
            # STEP 4: Determine order type (BUY or SELL)
            # ============================================================
            order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
            position_side = side

            # ============================================================
            # STEP 5: ❌ DO NOT CALCULATE SL/TP - Let VTM handle it!
            # ============================================================
            # OLD CODE (Removed):
            # risk = self.asset_config.get("risk", {})
            # stop_loss_pct = risk.get("stop_loss_pct", 0.03)
            # stop_loss = current_price * (1 - stop_loss_pct)
            # take_profit = current_price * (1 + take_profit_pct)
            
            # ✅ NEW: VTM will calculate SL/TP based on market structure
            stop_loss = 0  # MT5 requires a value, 0 = no SL set
            take_profit = 0  # MT5 requires a value, 0 = no TP set
            
            # Get trailing stop config (VTM will use this)
            risk = self.asset_config.get("risk", {})
            if side == "long":
                trailing_stop_pct = risk.get("trailing_stop_pct", 0.02)
            else:
                trailing_stop_pct = risk.get("trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.018))

            logger.info(
                f"[OPEN] {side.upper()} {volume_lots:.2f} lots {symbol} @ ${current_price:,.2f}\n"
                f"  Size: ${actual_position_size:,.2f}\n"
                f"  Mode: {sizing_mode} | Confidence: {confidence_score}\n"
                f"  ⚠️  VTM will calculate TP/SL based on market structure\n"
                f"  Trailing: {trailing_stop_pct:.2%} (VTM managed)"
            )

            # ============================================================
            # STEP 6: Execute on MT5
            # ============================================================
            mt5_ticket = None
            
            if self.mode.lower() != "paper":
                try:
                    # Get current price for execution
                    tick = mt5.symbol_info_tick(symbol)
                    execution_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

                    # Build MT5 order request WITHOUT SL/TP
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": volume_lots,
                        "type": order_type,
                        "price": execution_price,
                        "sl": stop_loss,  # 0 = no SL (VTM manages)
                        "tp": take_profit,  # 0 = no TP (VTM manages)
                        "deviation": 20,
                        "magic": 234000,
                        "comment": f"Signal_{signal}_{asset}_{sizing_mode}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    # Send order to MT5
                    result = mt5.order_send(request)

                    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                        error_msg = result.comment if result else "No result"
                        error_code = result.retcode if result else "N/A"
                        logger.error(
                            f"[MT5] ✗ {side.upper()} order failed: {error_msg} (code: {error_code})"
                        )
                        return False

                    mt5_ticket = result.order
                    logger.info(f"[MT5] ✓ {side.upper()} order placed: Ticket #{mt5_ticket}")
                
                except Exception as e:
                    logger.error(f"[MT5] Order execution failed: {e}")
                    return False

            # ============================================================
            # STEP 7: Fetch OHLC data for VTM (CRITICAL for dynamic SL/TP)
            # ============================================================
            ohlc_data = None
            
            try:
                if self.data_manager is None:
                    logger.warning("[VTM] data_manager not available, VTM disabled")
                else:
                    from datetime import datetime, timedelta, timezone
                    
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)

                    df = self.data_manager.fetch_mt5_data(
                        symbol=symbol,
                        timeframe=self.config["assets"][asset].get("timeframe", "H1"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    if len(df) > 50:  # Need sufficient data for VTM
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                        }
                        logger.debug(f"[VTM] Prepared {len(df)} bars for dynamic management")
                    else:
                        logger.warning(f"[VTM] Insufficient data ({len(df)} bars), using fallback SL/TP")

            except Exception as e:
                logger.warning(f"[VTM] Failed to fetch OHLC data: {e}, VTM disabled for this trade")

            # ============================================================
            # STEP 8: Add position to Portfolio Manager WITH VTM
            # ============================================================
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=position_side,
                entry_price=execution_price if mt5_ticket else current_price,
                position_size_usd=actual_position_size,
                stop_loss=None,  # ✅ VTM will calculate
                take_profit=None,  # ✅ VTM will calculate
                trailing_stop_pct=trailing_stop_pct,  # ✅ VTM will use this
                mt5_ticket=mt5_ticket,
                ohlc_data=ohlc_data,  # ✅ CRITICAL: Pass OHLC for VTM
                use_dynamic_management=True,  # ✅ Enable VTM
                signal_details=signal_details,  # ✅ Pass hybrid context
            )

            if not success:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset} position")
                
                # Emergency: Close unwanted MT5 position
                if mt5_ticket:
                    logger.warning(f"[MT5] Attempting to close unwanted position #{mt5_ticket}...")
                    self._emergency_close_mt5_position(symbol, volume_lots, order_type)
                
                return False

            # ============================================================
            # STEP 9: Verify VTM initialized correctly
            # ============================================================
            positions = self.portfolio_manager.get_asset_positions(asset)
            if positions:
                new_position = positions[-1]  # Get the position we just added
                
                if new_position.trade_manager:
                    vtm_status = new_position.get_vtm_status()
                    logger.info(
                        f"\n{'='*80}\n"
                        f"[VTM] ✅ ACTIVE for {side.upper()} position\n"
                        f"{'='*80}\n"
                        f"  Ticket:  #{mt5_ticket}\n"
                        f"  Entry:   ${vtm_status['entry_price']:,.2f}\n"
                        f"  SL:      ${vtm_status['stop_loss']:,.2f} (VTM calculated)\n"
                        f"  TP:      ${vtm_status.get('take_profit', 0):,.2f} (VTM calculated)\n"
                        f"  Trailing: {trailing_stop_pct:.2%} (VTM managed)\n"
                        f"  Mode:     Dynamic (market structure based)\n"
                        f"{'='*80}"
                    )
                else:
                    logger.warning(
                        f"[VTM] ⚠️ NOT INITIALIZED for {side.upper()} position\n"
                        f"  Ticket: #{mt5_ticket}\n"
                        f"  → Using fallback static SL/TP"
                    )

            logger.info(f"[OK] {side.upper()} {asset} position opened successfully")
            if mt5_ticket:
                logger.info(f"  └─ MT5 Ticket: #{mt5_ticket}")
            if ohlc_data:
                logger.info(f"  └─ VTM: ACTIVE (dynamic SL/TP)")
            else:
                logger.warning(f"  └─ VTM: DISABLED (static SL/TP fallback)")
            
            return True

        except Exception as e:
            logger.error(f"[MT5] Error opening {asset} position: {e}", exc_info=True)
            return False

    
        

    def check_and_update_positions_VTM(self, asset_name: str = "GOLD"):
        """Check and update ALL positions for an asset with VTM"""
        try:
            # Get ALL positions for this asset
            positions = self.portfolio_manager.get_asset_positions(asset_name)

            if not positions:
                return False

            current_price = self.get_current_price()
            if current_price is None or current_price == 0:
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

                    # ✅ FIX: exit_signal is a dict with 'reason', 'price', 'size'
                    if exit_signal:
                        # Extract data from dict
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

    def _emergency_close_mt5_position(
        self, symbol: str, volume: float, original_order_type: int
    ):
        """
        Emergency close of an unwanted MT5 position
        Works for BOTH long and short positions
        """
        try:
            import MetaTrader5 as mt5
            
            # Reverse the order type
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
                logger.info(f"[MT5] ✓ Emergency close successful")
            else:
                logger.error(
                    f"[MT5] ✗ Emergency close failed: {result.comment if result else 'No result'}"
                )
        
        except Exception as e:
            logger.error(f"[MT5] Emergency close error: {e}", exc_info=True)

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

    @handle_errors(
    component="mt5_handler",
    severity=ErrorSeverity.WARNING,
    notify=True,
    reraise=False,
    default_return=False
)
    def sync_positions_with_mt5(self, asset: str = "GOLD", symbol: str = None) -> bool:
        """
        ✅ COMPLETE FIX: Import positions WITH real market analysis for VTM
        Just copy-paste this entire method to replace your existing one
        """
        if symbol is None:
            symbol = self.symbol

        try:
            import MetaTrader5 as mt5
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
            mt5_long = sum(1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_BUY)
            mt5_short = sum(1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_SELL)
            portfolio_long = sum(1 for p in portfolio_positions if p.side == "long")
            portfolio_short = sum(1 for p in portfolio_positions if p.side == "short")

            logger.info(f"[SYNC] Position Count Comparison:")
            logger.info(f"  MT5:       {mt5_long} LONG, {mt5_short} SHORT (Total: {mt5_count})")
            logger.info(f"  Portfolio: {portfolio_long} LONG, {portfolio_short} SHORT (Total: {portfolio_count})")

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
                logger.warning(f"[SYNC] Symbol info for {symbol} unavailable; skipping MT5 sync.")
                return True

            logger.info(f"[SYNC] Found {mt5_count} MT5 position(s) and {portfolio_count} portfolio position(s)")

            # ================================================================
            # SCENARIO 1: MT5 has positions, portfolio is empty → IMPORT
            # ================================================================
            if mt5_count > 0 and portfolio_count == 0:
                import_enabled = bool(
                    self.config.get("portfolio", {}).get("import_existing_positions", False)
                )

                logger.info(f"[SYNC] Config check:")
                logger.info(f"  portfolio.import_existing_positions = {import_enabled}")

                if import_enabled:
                    logger.info(f"[SYNC] ✅ Import ENABLED - Importing {mt5_count} MT5 position(s) WITH VTM...")

                    # ============================================================
                    # STEP 1: Fetch OHLC data for VTM
                    # ============================================================
                    ohlc_data = None
                    df = None

                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = self.data_manager.fetch_mt5_data(
                            symbol=symbol,
                            timeframe=self.config["assets"][asset].get("timeframe", "H1"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                            logger.info(f"[VTM] ✅ Fetched {len(df)} bars for dynamic management")
                        else:
                            logger.warning(f"[VTM] ⚠️ Insufficient data ({len(df)} bars), VTM will be limited")

                    except Exception as e:
                        logger.error(f"[VTM] ❌ Failed to fetch OHLC: {e}")
                        ohlc_data = None
                        df = None

                    # ============================================================
                    # STEP 2: Get REAL market analysis from HybridAggregatorSelector
                    # ============================================================
                    signal_details_base = None

                    if df is not None and len(df) > 200:
                        try:
                            logger.info(f"[HYBRID] Analyzing market for imported positions...")

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
                            mode_info = hybrid_selector.get_optimal_mode(asset, df)
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
                                'reasoning': f"Position imported from MT5 - {analysis['reasoning']}",
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
                            'reasoning': 'Position imported from MT5 - market analysis unavailable',
                        }

                    # ============================================================
                    # STEP 3: Get actual account balance
                    # ============================================================
                    try:
                        import MetaTrader5 as mt5
                        account_info = mt5.account_info()
                        account_balance = account_info.equity if account_info else self.portfolio_manager.current_capital
                        logger.info(f"[MT5] Account balance: ${account_balance:,.2f}")
                    except:
                        account_balance = self.portfolio_manager.current_capital
                        logger.warning(f"[MT5] Using portfolio capital: ${account_balance:,.2f}")

                    # ============================================================
                    # STEP 4: Import each position
                    # ============================================================
                    imported_count = 0
                    for pos in mt5_positions:
                        pos_type = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"

                        logger.info(
                            f"\n  → Importing MT5 {pos_type.upper()}: ticket={pos.ticket}, "
                            f"entry=${pos.price_open:.2f}, current=${pos.price_current:.2f}"
                        )

                        # Check if we can import
                        can_import, reason = self.portfolio_manager.can_open_position(asset, pos_type)
                        if not can_import:
                            logger.warning(f"[SYNC] ⚠️ Cannot import position: {reason}")
                            continue

                        # Add position-specific details
                        signal_details = signal_details_base.copy()
                        signal_details['mt5_ticket'] = pos.ticket
                        signal_details['side'] = pos_type
                        signal_details['entry_price'] = pos.price_open

                        # Import position
                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=(
                                pos.volume * pos.price_open * self.symbol_info.trade_contract_size
                            ),
                            stop_loss=pos.sl if pos.sl > 0 else None,
                            take_profit=pos.tp if pos.tp > 0 else None,
                            trailing_stop_pct=self.config["assets"][asset].get("risk", {}).get("trailing_stop_pct"),
                            mt5_ticket=pos.ticket,
                            ohlc_data=ohlc_data,
                            use_dynamic_management=True,
                            entry_time=datetime.fromtimestamp(pos.time),
                            signal_details=signal_details,
                            #account_balance=account_balance,
                        )

                        if success:
                            imported_count += 1

                            # Verify VTM initialized
                            imported_positions = self.portfolio_manager.get_asset_positions(asset)
                            if imported_positions:
                                imported_pos = imported_positions[-1]
                                if imported_pos.trade_manager:
                                    vtm_status = imported_pos.get_vtm_status()
                                    logger.info(
                                        f"[VTM] ✅ ACTIVE with market analysis\n"
                                        f"      Ticket:  {pos.ticket}\n"
                                        f"      Mode:    {signal_details['aggregator_mode'].upper()}\n"
                                        f"      Regime:  {signal_details['regime_analysis']['regime_type']}\n"
                                        f"      Entry:   ${vtm_status['entry_price']:,.2f}\n"
                                        f"      SL:      ${vtm_status['stop_loss']:,.2f} (VTM calculated)\n"
                                        f"      TP:      ${vtm_status.get('take_profit', 0):,.2f} (VTM calculated)"
                                    )
                                else:
                                    logger.error(f"[VTM] ❌ NOT INITIALIZED for ticket {pos.ticket}")
                                    logger.error(f"      OHLC data: {ohlc_data is not None}")
                                    logger.error(f"      signal_details: {bool(signal_details)}")
                                    logger.error(f"      account_balance: {account_balance}")
                        else:
                            logger.error(f"[SYNC] ❌ Failed to import {asset} {pos_type} position")

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
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"[SYNC] ⚠️ IMPORT DISABLED IN CONFIG\n"
                        f"{'='*80}\n"
                        f"Found {mt5_count} MT5 position(s) but import is disabled.\n"
                        f"These positions will NOT be managed by the bot.\n"
                        f"{'='*80}"
                    )

                    for pos in mt5_positions:
                        pos_type = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        logger.info(
                            f"  → MT5 {pos_type}: ticket={pos.ticket}, entry=${pos.price_open:.2f}"
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

                logger.info(f"\n[SYNC] Cleanup complete: {closed_count}/{portfolio_count} positions removed\n")
                return closed_count == portfolio_count

            # ================================================================
            # SCENARIO 3: Both have positions → VALIDATE
            # ================================================================
            if mt5_count > 0 and portfolio_count > 0:
                logger.info(f"[SYNC] Validating {portfolio_count} portfolio vs {mt5_count} MT5 positions...")

                mt5_by_ticket = {pos.ticket: pos for pos in mt5_positions}
                positions_to_remove = []

                for pos in portfolio_positions:
                    if pos.mt5_ticket and pos.mt5_ticket not in mt5_by_ticket:
                        logger.warning(
                            f"[SYNC] ⚠️ Portfolio position {pos.position_id} (ticket={pos.mt5_ticket}) "
                            f"not found in MT5 → Marking for removal"
                        )
                        positions_to_remove.append(pos)

                if positions_to_remove:
                    current_price = self.get_current_price(self.symbol)
                    for pos in positions_to_remove:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="sync_mt5_ticket_missing",
                        )
                        logger.info(f"  ✅ Removed orphaned position {pos.position_id}")

                remaining_portfolio_count = self.portfolio_manager.get_asset_position_count(asset)

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
            # SCENARIO 4: Both empty
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
