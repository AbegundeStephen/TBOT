"""
MT5 Execution Handler with Hybrid Position Sizing
INTEGRATED: Automated risk management + manual override support
✨ ENHANCED: Hedging enabled - Allows simultaneous Long/Short positions (Asymmetric System)
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
    """
    ✅ FIXED: Position sizing that works for ALL account sizes

    Key Changes:
    1. Respects broker minimum lot sizes
    2. Scales appropriately for small accounts
    3. Prevents oversized positions on undercapitalized accounts
    """

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []

        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get(
            "aggressive_risk_threshold", 0.70
        )

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
        ✅ FIXED: Calculate position size with proper account size scaling
        """
        try:
            # ============================================================
            # STEP 1: Determine risk percentage
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

            min_stop_pct = risk_cfg.get("min_stop_distance_pct", 0.001)
            max_stop_pct = risk_cfg.get("max_stop_distance_pct", 0.10)

            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price

            if stop_distance_pct < 0.0001:
                stop_distance_pct = 0.0001

            if stop_distance_pct < min_stop_pct:
                logger.warning(
                    f"[RISK] Stop too tight: {stop_distance_pct:.2%} < {min_stop_pct:.2%}"
                )

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

            # ============================================================
            # STEP 3: Calculate position size from risk
            # ============================================================
            position_size_usd = risk_amount / stop_distance_pct

            logger.info(
                f"[RISK] Position Calculation:\n"
                f"  Account Capital: ${self.portfolio_manager.current_capital:,.2f}\n"
                f"  Risk Amount:     ${risk_amount:.2f} ({risk_pct:.2%})\n"
                f"  Stop Distance:   ${stop_distance:.2f} ({stop_distance_pct:.2%})\n"
                f"  Calc Position:   ${position_size_usd:,.2f}"
            )

            # ============================================================
            # STEP 4: Apply manual override if requested
            # ============================================================
            if sizing_mode == SizingMode.MANUAL_OVERRIDE and manual_size_usd:
                min_allowed = position_size_usd * 0.5
                max_allowed = position_size_usd * 2.0

                if manual_size_usd < min_allowed or manual_size_usd > max_allowed:
                    logger.warning(
                        f"[OVERRIDE] Manual size ${manual_size_usd:,.2f} outside range "
                        f"[${min_allowed:,.2f}, ${max_allowed:,.2f}] - Using calculated"
                    )
                else:
                    logger.info(
                        f"[OVERRIDE] Using manual size: ${manual_size_usd:,.2f}"
                    )
                    position_size_usd = manual_size_usd

            # ============================================================
            # STEP 5: Apply SMART position limits (account-aware)
            # ============================================================
            min_size = asset_cfg.get("min_position_usd", 100)

            # ✅ MARGIN TRADING FIX: Calculate max based on FREE MARGIN, not just account
            # This prevents "No money" errors by checking actual available margin
            config_max = asset_cfg.get("max_position_usd", 50000)

            # Get available margin from MT5
            try:
                import MetaTrader5 as mt5

                account_info = mt5.account_info()
                if account_info and account_info.margin_free > 0:
                    # Use free margin as the real constraint for margin trading
                    # Apply conservative 80% safety factor to avoid margin calls
                    available_margin = account_info.margin_free * 0.80
                    leverage = (
                        account_info.leverage if account_info.leverage > 0 else 100
                    )

                    # Max position = available margin × leverage × safety factor
                    margin_based_max = (
                        available_margin * leverage * 0.20
                    )  # 20% of leveraged margin

                    logger.info(
                        f"[MARGIN INFO]\n"
                        f"  Account Leverage: {leverage}:1\n"
                        f"  Free Margin:     ${account_info.margin_free:,.2f}\n"
                        f"  Margin Level:    {account_info.margin_level:.2f}%\n"
                        f"  Used Margin:     ${account_info.margin:,.2f}\n"
                        f"  Available (80%): ${available_margin:,.2f}"
                    )
                else:
                    # Fallback to capital-based if margin info unavailable
                    margin_based_max = self.portfolio_manager.current_capital * 0.50
                    logger.warning(
                        "[MARGIN] Could not get MT5 margin info, using capital-based limit"
                    )
            except Exception as e:
                logger.warning(
                    f"[MARGIN] Error getting margin info: {e}, using capital-based limit"
                )
                margin_based_max = self.portfolio_manager.current_capital * 0.50

            # Use LOWEST of: config max, margin-based max
            max_size = min(config_max, margin_based_max)

            logger.info(
                f"[LIMITS]\n"
                f"  Config Max:        ${config_max:,.2f}\n"
                f"  Margin-Based Max:  ${margin_based_max:,.2f}\n"
                f"  Applied Max:       ${max_size:,.2f}"
            )

            # Check minimum
            if position_size_usd < min_size:
                logger.warning(
                    f"[RISK] Position ${position_size_usd:.2f} below min ${min_size}"
                )
                return 0.0, {
                    "error": "below_minimum",
                    "calculated_size": position_size_usd,
                    "minimum": min_size,
                }

            # Apply maximum
            original_size = position_size_usd
            position_size_usd = min(position_size_usd, max_size)

            if original_size > max_size:
                logger.warning(
                    f"[RISK] Position clamped to account-based max: ${original_size:,.2f} → ${max_size:,.2f}"
                )

            # ============================================================
            # STEP 6: Check portfolio-level limits
            # ============================================================
            max_exposure_mult = self.portfolio_cfg.get("max_portfolio_exposure", 5.0)
            max_single_asset_mult = self.portfolio_cfg.get(
                "max_single_asset_exposure", 3.0
            )

            current_exposure = sum(
                pos.quantity * pos.entry_price
                for pos in self.portfolio_manager.positions.values()
            )

            max_portfolio_usd = (
                self.portfolio_manager.current_capital * max_exposure_mult
            )

            if current_exposure + position_size_usd > max_portfolio_usd:
                old_size = position_size_usd
                position_size_usd = max(0, max_portfolio_usd - current_exposure)
                logger.warning(
                    f"[RISK] Portfolio leverage limit: ${old_size:,.2f} → ${position_size_usd:,.2f}"
                )

            max_asset_usd = (
                self.portfolio_manager.current_capital * max_single_asset_mult
            )
            if position_size_usd > max_asset_usd:
                old_size = position_size_usd
                position_size_usd = max_asset_usd
                logger.warning(
                    f"[RISK] Single asset leverage limit: ${old_size:,.2f} → ${position_size_usd:,.2f}"
                )

            # ============================================================
            # STEP 6.5: Check Total Open Risk
            # ============================================================
            max_total_risk_pct = self.config.get("risk_management", {}).get(
                "max_total_open_risk", 0.10
            )

            current_total_risk = 0.0
            for pos in self.portfolio_manager.positions.values():
                dist = abs(pos.entry_price - (pos.stop_loss or pos.entry_price))
                current_total_risk += dist * pos.quantity

            new_trade_risk = position_size_usd * stop_distance_pct

            if current_total_risk + new_trade_risk > (
                self.portfolio_manager.current_capital * max_total_risk_pct
            ):
                logger.warning(
                    f"[RISK] Total Portfolio Risk Limit Hit!\n"
                    f"  Current Risk: ${current_total_risk:.2f}\n"
                    f"  New Trade:    ${new_trade_risk:.2f}\n"
                    f"  Limit:        ${self.portfolio_manager.current_capital * max_total_risk_pct:.2f}"
                )
                return 0.0, {"error": "total_risk_limit_exceeded"}

            # ============================================================
            # STEP 7: Calculate ACTUAL risk with final position size
            # ============================================================
            actual_risk = position_size_usd * stop_distance_pct
            actual_risk_pct = actual_risk / self.portfolio_manager.current_capital

            if actual_risk_pct > (self.max_risk_pct * 1.5):
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
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "stop_distance_usd": stop_distance,
                "stop_distance_pct": stop_distance_pct * 100,
                "target_risk_pct": risk_pct * 100,
                "target_risk_usd": risk_amount,
                "actual_risk_pct": actual_risk_pct * 100,
                "actual_risk_usd": actual_risk,
                "position_size_usd": position_size_usd,
                "position_size_pct": (
                    position_size_usd / self.portfolio_manager.current_capital
                )
                * 100,
                "position_size_units": position_size_usd / entry_price,
                "override_details": (
                    {"mode": sizing_mode, "reason": override_reason}
                    if sizing_mode != SizingMode.AUTOMATED
                    else None
                ),
                "timestamp": datetime.now().isoformat(),
            }

            if sizing_mode != SizingMode.AUTOMATED:
                self.override_history.append(metadata)

            return position_size_usd, metadata

        except Exception as e:
            logger.error(f"[RISK] Error calculating position size: {e}", exc_info=True)
            return 0.0, {"error": str(e)}


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
        default_return=0.0,
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
        """Check if we can open a position on a specific SIDE"""
        if side == "short":
            allow_shorts = self.config["assets"][asset_name].get("allow_shorts", False)
            if not allow_shorts:
                return False, f"Short trading disabled for {asset_name}"

        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(
            asset_name, side
        )
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        current_count = self.portfolio_manager.get_asset_position_count(
            asset_name, side
        )
        max_per_asset = self.max_positions_per_asset

        if current_count >= max_per_asset:
            return (
                False,
                f"Already have {current_count}/{max_per_asset} {side.upper()} positions",
            )

        try:
            import MetaTrader5 as mt5

            account_info = mt5.account_info()
            if account_info:
                margin_free = account_info.margin_free
                margin_required = 50.0

                if margin_free < margin_required:
                    return False, f"Insufficient margin: ${margin_free:.2f} free"

        except Exception as e:
            logger.debug(f"[MT5] Margin check warning: {e}")

        return (
            True,
            f"OK - {current_count}/{max_per_asset} {side.upper()} positions open",
        )

    def _is_trading_allowed(self, symbol: str) -> bool:
        """
        Check if trading is currently allowed for the symbol
        """
        try:
            import MetaTrader5 as mt5

            # 1. Get symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"[MT5] Symbol {symbol} not found")
                return False

            # 2. Check trade mode
            # SYMBOL_TRADE_MODE_DISABLED = 0
            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                logger.warning(
                    f"[MT5] Trading is DISABLED for {symbol} (Market closed or restricted)"
                )
                return False

            # 3. Check for tick data (proxy for market open)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(
                    f"[MT5] Market likely CLOSED (no tick data for {symbol})"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"[MT5] Error checking trading status: {e}")
            return False

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
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
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ FIXED: Position opening with proper account size handling
        """
        try:
            import MetaTrader5 as mt5
            from src.execution.veteran_trade_manager import VeteranTradeManager

            if not mt5.symbol_select(symbol, True):
                logger.error(f"[MT5] Failed to select symbol {symbol}")
                return False

            side = "long" if signal == 1 else "short"
            order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
            logger.info(f"[MT5] Opening {side.upper()} position for {asset}")
            asset_cfg = self.config["assets"].get(asset, {})
            leverage = asset_cfg.get("leverage", 2000)  # Default high leverage for MT5
            margin_type = "DYNAMIC"  # MT5 is always Derivatives/CFD
            is_futures = True

            # Get OHLC data for VTM
            ohlc_data = None
            if self.data_manager:
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
                except Exception as e:
                    logger.error(f"[VTM] OHLC fetch failed: {e}")

            # Calculate VTM Stop
            stop_loss_price = None
            if ohlc_data and signal_details:
                try:
                    temp_vtm = VeteranTradeManager(
                        entry_price=current_price,
                        side=side,
                        asset=asset,
                        high=ohlc_data["high"],
                        low=ohlc_data["low"],
                        close=ohlc_data["close"],
                        account_balance=self.portfolio_manager.current_capital,
                        signal_details=signal_details,
                        account_risk=0.015,
                    )
                    stop_loss_price = temp_vtm.initial_stop_loss
                    logger.info(f"[VTM] ✓ Stop: ${stop_loss_price:,.2f}")
                except Exception as e:
                    logger.error(f"[VTM] Stop calculation failed: {e}")

            # Fallback Stop
            if stop_loss_price is None:
                risk = self.asset_config.get("risk", {})
                if side == "long":
                    stop_loss_price = current_price * (
                        1 - risk.get("stop_loss_pct", 0.01)
                    )
                else:
                    stop_loss_price = current_price * (
                        1 + risk.get("stop_loss_pct", 0.01)
                    )

            # Calculate Sizing
            position_size_usd, sizing_metadata = self.sizer.calculate_size_risk_based(
                asset=asset,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                signal=signal,
                confidence_score=confidence_score,
                market_condition=market_condition,
                sizing_mode=sizing_mode,
                manual_size_usd=manual_size_usd,
                override_reason=override_reason,
            )

            if position_size_usd <= 0:
                logger.error(f"[FAIL] Sizing failed: {sizing_metadata.get('error')}")
                return False

            # ✅ FIXED: Calculate volume with ABORT on min lot violation
            contract_size = self.symbol_info.trade_contract_size
            volume_lots = position_size_usd / (current_price * contract_size)

            volume_step = self.symbol_info.volume_step
            volume_lots = round(volume_lots / volume_step) * volume_step

            # Abort if below minimum
            if volume_lots < self.symbol_info.volume_min:
                min_usd_value = (
                    self.symbol_info.volume_min * current_price * contract_size
                )
                logger.warning(
                    f"[RISK] Trade Aborted: Volume {volume_lots:.4f} < Min {self.symbol_info.volume_min}\n"
                    f"  Requested Size: ${position_size_usd:.2f}\n"
                    f"  Min Broker Size: ${min_usd_value:.2f}\n"
                    f"  → Account too small for this trade (need ~${min_usd_value:.0f} minimum)"
                )
                return False

            volume_lots = min(self.symbol_info.volume_max, volume_lots)

            actual_usd = volume_lots * current_price * contract_size
            logger.info(f"[SIZE] {volume_lots:.2f} lots = ${actual_usd:,.2f}")

            # ✅ MARGIN TRADING: Check actual margin requirements BEFORE placing order
            # This prevents "No money" rejections by validating margin availability
            if self.mode.lower() != "paper":
                # Pre-flight margin check
                try:
                    account_info = mt5.account_info()
                    if account_info:
                        # Calculate required margin for this trade
                        # Formula: (Volume × Contract Size × Price) / Leverage
                        leverage = (
                            account_info.leverage if account_info.leverage > 0 else 100
                        )
                        estimated_margin_required = (
                            volume_lots * contract_size * current_price
                        ) / leverage

                        # Add 10% buffer for price slippage and margin calculation differences
                        estimated_margin_required *= 1.10

                        logger.info(
                            f"[MARGIN CHECK]\n"
                            f"  Free Margin:      ${account_info.margin_free:,.2f}\n"
                            f"  Required Margin:  ${estimated_margin_required:,.2f}\n"
                            f"  Margin Level:     {account_info.margin_level:.2f}%"
                        )

                        if estimated_margin_required > account_info.margin_free:
                            logger.error(
                                f"[MARGIN] ❌ INSUFFICIENT MARGIN!\n"
                                f"  Need: ${estimated_margin_required:,.2f}\n"
                                f"  Have: ${account_info.margin_free:,.2f}\n"
                                f"  → Reducing position size to fit available margin"
                            )

                            # Recalculate volume to fit available margin
                            max_volume_by_margin = (
                                account_info.margin_free * 0.90 * leverage
                            ) / (contract_size * current_price)
                            max_volume_by_margin = (
                                round(max_volume_by_margin / volume_step) * volume_step
                            )

                            if max_volume_by_margin < self.symbol_info.volume_min:
                                logger.error(
                                    f"[MARGIN] ❌ Even minimum lot ({self.symbol_info.volume_min}) requires "
                                    f"${(self.symbol_info.volume_min * contract_size * current_price) / leverage:.2f} margin\n"
                                    f"  Available: ${account_info.margin_free:.2f}\n"
                                    f"  → CANNOT OPEN POSITION"
                                )
                                return False

                            # Use reduced volume
                            logger.warning(
                                f"[MARGIN] Adjusted volume: {volume_lots:.2f} → {max_volume_by_margin:.2f} lots"
                            )
                            volume_lots = max_volume_by_margin
                            actual_usd = volume_lots * current_price * contract_size

                except Exception as e:
                    logger.warning(f"[MARGIN] Pre-flight check warning: {e}")

                # Now place the order with validated margin
                if not self._is_trading_allowed(symbol):
                    logger.error(f"[MT5] Trading not allowed for {symbol}")
                    return False

                tick = mt5.symbol_info_tick(symbol)
                execution_price = (
                    tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
                )

                filling_mode = mt5.ORDER_FILLING_FOK
                symbol_filling = self.symbol_info.filling_mode

                if symbol_filling == 1:
                    filling_mode = mt5.ORDER_FILLING_FOK
                elif symbol_filling == 2:
                    filling_mode = mt5.ORDER_FILLING_IOC
                elif symbol_filling == 3:
                    filling_mode = mt5.ORDER_FILLING_FOK

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume_lots,
                    "type": order_type,
                    "price": execution_price,
                    "sl": 0.0,
                    "tp": 0.0,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"Sig_{signal}_{asset}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }

                result = mt5.order_send(request)

                if result is None:
                    last_error = mt5.last_error()
                    logger.error(f"[MT5] Order Failed: No result. Error: {last_error}")
                    return False

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(
                        f"[MT5] Rejected: {result.comment} (Code: {result.retcode})"
                    )
                    return False

                mt5_ticket = result.order
                logger.info(f"[MT5] ✓ {side.upper()} order placed: #{mt5_ticket}")

            else:
                mt5_ticket = f"PAPER_{int(datetime.now().timestamp())}"
                execution_price = current_price
                logger.info(f"[PAPER] Simulated: {mt5_ticket}")

            # Add to Portfolio
            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=side,
                entry_price=execution_price,
                position_size_usd=actual_usd,
                stop_loss=None,
                take_profit=None,
                trailing_stop_pct=None,
                mt5_ticket=mt5_ticket,
                ohlc_data=ohlc_data,
                use_dynamic_management=True,
                signal_details=signal_details,
                leverage=leverage,
                margin_type=margin_type,
                is_futures=is_futures,
            )

            if not success and self.mode.lower() != "paper" and mt5_ticket:
                logger.warning(f"[EMERGENCY] Closing orphaned MT5 #{mt5_ticket}")
                self._emergency_close_mt5_position(symbol, volume_lots, order_type)
                return False

            return True

        except Exception as e:
            logger.error(f"[MT5] Critical Error: {e}", exc_info=True)
            return False

        except Exception as e:
            logger.error(f"[MT5] Critical Error: {e}", exc_info=True)
            return False

    def _is_market_open_for_closing(self, symbol: str) -> Tuple[bool, str]:
        """
        ✅ NEW: Check if market is open for closing positions

        Returns:
            (is_open, message)
        """
        try:
            import MetaTrader5 as mt5

            # 1. Get symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                return False, f"Symbol {symbol} not found"

            # 2. Check if market session is active
            # trade_mode values:
            # 0 = SYMBOL_TRADE_MODE_DISABLED (trading disabled)
            # 1 = SYMBOL_TRADE_MODE_LONGONLY (only long positions)
            # 2 = SYMBOL_TRADE_MODE_SHORTONLY (only short positions)
            # 3 = SYMBOL_TRADE_MODE_CLOSEONLY (only closing allowed)
            # 4 = SYMBOL_TRADE_MODE_FULL (full trading)

            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                return False, "Market is CLOSED - Trading disabled"

            # Even in CLOSEONLY mode, we can close positions
            if info.trade_mode >= mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
                return True, "OK"

            # 3. Check for tick data (additional validation)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False, "Market is CLOSED - No price quotes available"

            # 4. Check if last tick is recent (within 5 minutes)
            from datetime import datetime, timezone

            current_time = datetime.now(timezone.utc)
            tick_time = datetime.fromtimestamp(tick.time, timezone.utc)
            time_diff = (current_time - tick_time).total_seconds()

            if time_diff > 300:  # 5 minutes
                return (
                    False,
                    f"Market is CLOSED - Last quote was {int(time_diff/60)} minutes ago",
                )

            return True, "OK"

        except Exception as e:
            logger.error(f"[MT5] Error checking market status: {e}")
            return False, f"Error checking market status: {str(e)}"

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
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
        ✅ MT5 TWO-WAY TRADING: Execute trading signal with Asymmetric Hedging Support

        Signal Logic:
        - BUY (+1):  Open long (Closes short ONLY IF hedging disabled)
        - SELL (-1): Open short (Closes long ONLY IF hedging disabled)
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

        if asset_name != "GOLD":
            logger.error(
                f"[MT5 HANDLER] ❌ WRONG ASSET!\n"
                f"  This handler is for GOLD ONLY\n"
                f"  Received request for: {asset_name}\n"
                f"  REJECTING EXECUTION"
            )
            return False

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
            # STEP 2: Get existing positions & HEDGING CONFIG
            # ============================================================
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)

            long_positions = [p for p in existing_positions if p.side == "long"]
            short_positions = [p for p in existing_positions if p.side == "short"]
            
            # ✨ NEW: Check if Asymmetric Hedging is enabled
            allow_hedging = self.config.get("trading", {}).get("allow_simultaneous_long_short", True)

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} Signal: {signal:+2d}\n"
                f"[STATE] Current Positions: {len(long_positions)} LONG, {len(short_positions)} SHORT\n"
                f"[CONFIG] Hedging Allowed: {allow_hedging}\n"
                f"{'='*80}"
            )

            if signal_details and signal_details.get("aggregator_mode"):
                logger.info(
                    f"\n[HYBRID] Mode: {signal_details['aggregator_mode'].upper()} "
                    f"({signal_details.get('mode_confidence', 0):.0%} confidence)"
                )

            # ============================================================
            # SCENARIO 1: SELL SIGNAL (-1) → Handle longs, Open short
            # ============================================================
            if signal == -1:
                # Step 1: Handle LONG positions (The "Kill Switch" Logic)
                if long_positions:
                    # ✨ NEW: Respect Hedging Configuration
                    if not allow_hedging:
                        logger.info(
                            f"\n{'='*80}\n"
                            f"📉 SELL SIGNAL - Hedging Disabled: Closing ALL {len(long_positions)} LONG position(s)\n"
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
                        logger.info(
                            f"\n{'='*80}\n"
                            f"📉 SELL SIGNAL - Hedging Enabled: Keeping {len(long_positions)} LONG position(s) OPEN.\n"
                            f"{'='*80}"
                        )

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
                    # If hedging was enabled, we still have longs open, so return True (bot maintains state)
                    return True if (long_positions and allow_hedging) else False

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
                    signal_details=signal_details,
                )

                # Verify sync after opening
                if success:
                    self._verify_position_sync(asset_name, symbol)

                return success

            # ============================================================
            # SCENARIO 2: BUY SIGNAL (+1) → Handle shorts, Open long
            # ============================================================
            elif signal == 1:
                # Step 1: Handle SHORT positions (The "Kill Switch" Logic)
                if short_positions:
                    # ✨ NEW: Respect Hedging Configuration
                    if not allow_hedging:
                        logger.info(
                            f"\n{'='*80}\n"
                            f"📈 BUY SIGNAL - Hedging Disabled: Closing ALL {len(short_positions)} SHORT position(s)\n"
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
                        logger.info(
                            f"\n{'='*80}\n"
                            f"📈 BUY SIGNAL - Hedging Enabled: Keeping {len(short_positions)} SHORT position(s) OPEN.\n"
                            f"{'='*80}"
                        )

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
                    # If hedging was enabled, we still have shorts open, so return True
                    return True if (short_positions and allow_hedging) else False

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
                logger.warning(
                    f"[SYNC] Mismatch detected! Triggering automatic re-sync..."
                )
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
        default_return=False,
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
        ✅ FIXED: Close MT5 order with market hours validation

        Returns:
            True if successfully closed, False otherwise
        """
        try:
            # ✅ FIX 1: Check if market is open for closing
            is_open, market_msg = self._is_market_open_for_closing(self.symbol)

            if not is_open:
                logger.error(
                    f"[MT5] ❌ CANNOT CLOSE POSITION\n"
                    f"  Ticket: {ticket}\n"
                    f"  Asset:  {asset}\n"
                    f"  Reason: {market_msg}\n"
                    f"  → Position will remain open until market reopens"
                )
                return False

            # Find the position by ticket
            mt5_positions = mt5.positions_get(ticket=ticket)

            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(f"[MT5] Position ticket {ticket} not found on exchange")
                # Return False - position doesn't exist on MT5
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
            if tick is None:
                logger.error(
                    f"[MT5] Cannot get current price for {mt5_position.symbol}"
                )
                return False

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

                # ✅ FIX 2: Specific error messages
                if result and result.retcode == 10018:  # Market closed
                    logger.error(
                        f"[MT5] ✗ MARKET CLOSED - Cannot close ticket {ticket}\n"
                        f"  Error: {error_msg}\n"
                        f"  → Position remains open, try again when market opens"
                    )
                else:
                    logger.error(
                        f"[MT5] ✗ Failed to close ticket {ticket}: {error_msg} (code: {error_code})"
                    )

                return False

        except Exception as e:
            logger.error(f"[MT5] Error closing ticket {ticket}: {e}", exc_info=True)
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
                        if hasattr(exit_reason, "value"):
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
                            position,
                            current_price,
                            asset_name,
                            f"VTM_{exit_reason_str}",
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
        default_return=False,
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
                import_enabled = bool(
                    self.config.get("portfolio", {}).get(
                        "import_existing_positions", False
                    )
                )

                logger.info(f"[SYNC] Config check:")
                logger.info(f"  portfolio.import_existing_positions = {import_enabled}")

                if import_enabled:
                    logger.info(
                        f"[SYNC] ✅ Import ENABLED - Importing {mt5_count} MT5 position(s) WITH VTM..."
                    )

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
                                f"[VTM] ⚠️ Insufficient data ({len(df)} bars), VTM will be limited"
                            )

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
                            logger.info(
                                f"[HYBRID] Analyzing market for imported positions..."
                            )

                            # Try to get hybrid selector from parent bot
                            hybrid_selector = None
                            if hasattr(self, "trading_bot") and hasattr(
                                self.trading_bot, "hybrid_selector"
                            ):
                                hybrid_selector = self.trading_bot.hybrid_selector
                                logger.info(
                                    f"[HYBRID] Using existing hybrid_selector from bot"
                                )
                            else:
                                # Create temporary instance
                                from src.execution.hybrid_aggregator_selector import (
                                    HybridAggregatorSelector,
                                )

                                hybrid_selector = HybridAggregatorSelector(
                                    self.data_manager,
                                    self.config,
                                )
                                logger.info(
                                    f"[HYBRID] Created temporary hybrid_selector"
                                )

                            # Get current market analysis
                            mode_info = hybrid_selector.get_optimal_mode(asset, df)
                            analysis = mode_info["analysis"]

                            # Build REAL signal_details from market analysis
                            signal_details_base = {
                                "imported": True,
                                "import_time": datetime.now().isoformat(),
                                # Real aggregator mode
                                "aggregator_mode": mode_info["mode"],
                                "mode_confidence": mode_info["confidence"],
                                # Real regime analysis
                                "regime_analysis": {
                                    "regime_type": analysis["regime_type"],
                                    "trend_strength": analysis["trend"]["strength"],
                                    "trend_direction": analysis["trend"]["direction"],
                                    "adx": analysis["trend"]["adx"],
                                    "volatility_regime": analysis["volatility"][
                                        "regime"
                                    ],
                                    "volatility_ratio": analysis["volatility"]["ratio"],
                                    "price_clarity": analysis["price_action"][
                                        "clarity"
                                    ],
                                    "indecision_pct": analysis["price_action"][
                                        "indecision_pct"
                                    ],
                                    "momentum_aligned": analysis["momentum_aligned"],
                                    "at_key_level": analysis["at_key_level"],
                                },
                                "signal_quality": mode_info["confidence"],
                                "reasoning": f"Position imported from MT5 - {analysis['reasoning']}",
                            }

                            logger.info(f"[HYBRID] ✅ Market analysis complete:")
                            logger.info(f"  Mode:       {mode_info['mode'].upper()}")
                            logger.info(f"  Confidence: {mode_info['confidence']:.0%}")
                            logger.info(f"  Regime:     {analysis['regime_type']}")
                            logger.info(
                                f"  Trend:      {analysis['trend']['strength']} / {analysis['trend']['direction']}"
                            )
                            logger.info(
                                f"  Volatility: {analysis['volatility']['regime']}"
                            )

                        except Exception as e:
                            logger.error(f"[HYBRID] ❌ Analysis failed: {e}")
                            signal_details_base = None

                    # Fallback if hybrid analysis fails
                    if signal_details_base is None:
                        logger.warning(
                            f"[HYBRID] Using fallback signal_details (no market analysis)"
                        )
                        signal_details_base = {
                            "imported": True,
                            "import_time": datetime.now().isoformat(),
                            "aggregator_mode": "unknown",
                            "mode_confidence": 0.5,
                            "regime_analysis": {
                                "regime_type": "unknown",
                                "trend_strength": "unknown",
                                "trend_direction": "unknown",
                                "adx": 20.0,
                                "volatility_regime": "normal",
                                "volatility_ratio": 1.0,
                                "price_clarity": "unknown",
                                "indecision_pct": 0.0,
                                "momentum_aligned": False,
                                "at_key_level": False,
                            },
                            "signal_quality": 0.5,
                            "reasoning": "Position imported from MT5 - market analysis unavailable",
                        }

                    # ============================================================
                    # STEP 3: Get actual account balance
                    # ============================================================
                    try:
                        import MetaTrader5 as mt5

                        account_info = mt5.account_info()
                        account_balance = (
                            account_info.equity
                            if account_info
                            else self.portfolio_manager.current_capital
                        )
                        logger.info(f"[MT5] Account balance: ${account_balance:,.2f}")
                    except:
                        account_balance = self.portfolio_manager.current_capital
                        logger.warning(
                            f"[MT5] Using portfolio capital: ${account_balance:,.2f}"
                        )

                    # ============================================================
                    # STEP 4: Import each position
                    # ============================================================
                    imported_count = 0
                    for pos in mt5_positions:
                        pos_type = (
                            "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        )

                        logger.info(
                            f"\n  → Importing MT5 {pos_type.upper()}: ticket={pos.ticket}, "
                            f"entry=${pos.price_open:.2f}, current=${pos.price_current:.2f}"
                        )

                        # Check if we can import
                        can_import, reason = self.portfolio_manager.can_open_position(
                            asset, pos_type
                        )
                        if not can_import:
                            logger.warning(f"[SYNC] ⚠️ Cannot import position: {reason}")
                            continue

                        # Add position-specific details
                        signal_details = signal_details_base.copy()
                        signal_details["mt5_ticket"] = pos.ticket
                        signal_details["side"] = pos_type
                        signal_details["entry_price"] = pos.price_open

                        # Import position
                        # The above code is attempting to call the `add_position` method of the
                        # `portfolio_manager` object within the `self` context.
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
                            ohlc_data=ohlc_data,
                            use_dynamic_management=True,
                            entry_time=datetime.fromtimestamp(pos.time),
                            signal_details=signal_details,
                            # account_balance=account_balance,
                        )

                        if success:
                            imported_count += 1

                            # Verify VTM initialized
                            imported_positions = (
                                self.portfolio_manager.get_asset_positions(asset)
                            )
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
                                    logger.error(
                                        f"[VTM] ❌ NOT INITIALIZED for ticket {pos.ticket}"
                                    )
                                    logger.error(
                                        f"      OHLC data: {ohlc_data is not None}"
                                    )
                                    logger.error(
                                        f"      signal_details: {bool(signal_details)}"
                                    )
                                    logger.error(
                                        f"      account_balance: {account_balance}"
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
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"[SYNC] ⚠️ IMPORT DISABLED IN CONFIG\n"
                        f"{'='*80}\n"
                        f"Found {mt5_count} MT5 position(s) but import is disabled.\n"
                        f"These positions will NOT be managed by the bot.\n"
                        f"{'='*80}"
                    )

                    for pos in mt5_positions:
                        pos_type = (
                            "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        )
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

                logger.info(
                    f"\n[SYNC] Cleanup complete: {closed_count}/{portfolio_count} positions removed\n"
                )
                return closed_count == portfolio_count

            # ================================================================
            # SCENARIO 3: Both have positions → VALIDATE
            # ================================================================
            if mt5_count > 0 and portfolio_count > 0:
                logger.info(
                    f"[SYNC] Validating {portfolio_count} portfolio vs {mt5_count} MT5 positions..."
                )

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