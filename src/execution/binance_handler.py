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


class DynamicMarginCalculator:
    """
    Calculates available margin and maximum position sizes for Binance Futures
    Ensures positions never exceed available margin
    """

    def __init__(self, futures_handler, config: Dict):
        self.futures_handler = futures_handler
        self.config = config

    def get_available_margin_info(self, asset: str) -> Dict:
        """
        Get comprehensive margin information from Binance Futures

        Returns:
            {
                'available_balance': float,  # USDT available for new positions
                'total_balance': float,      # Total wallet balance
                'used_margin': float,        # Margin locked in positions
                'unrealized_pnl': float,     # Unrealized profit/loss
                'leverage': int,             # Current leverage setting
                'max_position_notional': float,  # Max position size at current leverage
            }
        """
        try:
            # Get Futures account info
            account = self.futures_handler.client.futures_account()

            # Extract key metrics
            total_balance = float(account.get("totalWalletBalance", 0))
            available_balance = float(account.get("availableBalance", 0))
            total_unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
            total_margin_balance = float(account.get("totalMarginBalance", 0))

            # Get current positions to calculate used margin
            positions = self.futures_handler.client.futures_position_information()
            used_margin = 0.0

            for pos in positions:
                if pos["symbol"] == self.futures_handler.symbol:
                    pos_amt = abs(float(pos.get("positionAmt", 0)))
                    if pos_amt > 0:
                        entry_price = float(pos.get("entryPrice", 0))
                        leverage = float(pos.get("leverage", 1))
                        used_margin += (pos_amt * entry_price) / leverage

            # Get leverage setting
            leverage = self.config.get("assets", {}).get(asset, {}).get("leverage", 20)

            # Calculate max position based on available balance and leverage
            max_position_notional = available_balance * leverage

            logger.info(
                f"[MARGIN] Binance Futures Account Status:\n"
                f"  Total Balance:    ${total_balance:,.2f} USDT\n"
                f"  Available:        ${available_balance:,.2f} USDT\n"
                f"  Used Margin:      ${used_margin:,.2f} USDT\n"
                f"  Unrealized P&L:   ${total_unrealized_pnl:,.2f} USDT\n"
                f"  Leverage:         {leverage}x\n"
                f"  Max Position:     ${max_position_notional:,.2f} USDT"
            )

            return {
                "available_balance": available_balance,
                "total_balance": total_balance,
                "used_margin": used_margin,
                "unrealized_pnl": total_unrealized_pnl,
                "leverage": leverage,
                "max_position_notional": max_position_notional,
            }

        except Exception as e:
            logger.error(f"[MARGIN] Error getting margin info: {e}")
            return {
                "available_balance": 0.0,
                "total_balance": 0.0,
                "used_margin": 0.0,
                "unrealized_pnl": 0.0,
                "leverage": 1,
                "max_position_notional": 0.0,
            }

    def calculate_max_safe_position(
        self,
        available_margin: float,
        leverage: int,
        entry_price: float,
        stop_loss_price: float,
        buffer_pct: float = 0.10,
    ) -> Tuple[float, float]:
        """
        Calculate maximum safe position size that won't get liquidated

        Args:
            available_margin: Available USDT margin
            leverage: Leverage multiplier (e.g., 20)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            buffer_pct: Safety buffer (default 10%)

        Returns:
            (max_position_usd, max_quantity)
        """
        try:
            # Maximum position based on margin and leverage
            max_position_from_margin = available_margin * leverage

            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price

            # Calculate position size that fits margin requirements
            # Required Margin = Position Size / Leverage
            # Max Loss at SL = Position Size * Stop Distance %
            # We want: Required Margin + Max Loss <= Available Margin
            # Position * (1/Leverage + Stop %) <= Available
            # Position <= Available / (1/Leverage + Stop %)

            denominator = (1 / leverage) + stop_distance_pct
            max_safe_position = available_margin / denominator

            # Apply safety buffer (reduce by buffer% to account for fees, slippage)
            max_safe_position *= 1 - buffer_pct

            # Take minimum of margin limit and safe limit
            max_position_usd = min(max_position_from_margin, max_safe_position)

            # Calculate quantity
            max_quantity = max_position_usd / entry_price

            logger.debug(
                f"[MARGIN] Max Safe Position Calculation:\n"
                f"  Available Margin:   ${available_margin:,.2f}\n"
                f"  Leverage:           {leverage}x\n"
                f"  Stop Distance:      {stop_distance_pct:.2%}\n"
                f"  Max from Margin:    ${max_position_from_margin:,.2f}\n"
                f"  Max Safe (w/SL):    ${max_safe_position:,.2f}\n"
                f"  Final (w/buffer):   ${max_position_usd:,.2f}\n"
                f"  Quantity:           {max_quantity:.6f} BTC"
            )

            return max_position_usd, max_quantity

        except Exception as e:
            logger.error(f"[MARGIN] Error calculating max position: {e}")
            return 0.0, 0.0


class HybridPositionSizer:
    """
    Enhanced position sizer with dynamic margin awareness
    Automatically adjusts position sizes to fit available Binance Futures margin
    """

    def __init__(self, config: Dict, portfolio_manager, futures_handler=None):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.futures_handler = futures_handler
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []

        # Initialize margin calculator if futures available
        self.margin_calculator = None
        if futures_handler:
            self.margin_calculator = DynamicMarginCalculator(futures_handler, config)

        # Risk parameters
        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get(
            "aggressive_risk_threshold", 0.70
        )

        self.rebalancer = None

        logger.info(
            f"[RISK SIZER] Initialized\n"
            f"  Target Risk: {self.target_risk_pct:.2%}\n"
            f"  Max Risk:    {self.max_risk_pct:.2%}\n"
            f"  Futures:     {'✓ Dynamic Margin' if self.margin_calculator else '✗ Not available'}"
        )

    def set_rebalancer(self, rebalancer: PositionRebalancer):
        """Set the rebalancer (called by handler after initialization)"""
        self.rebalancer = rebalancer
        logger.info("[RISK SIZER] ✓ Rebalancer connected")

    def _get_available_balance(
        self,
        asset: str,
        is_futures: bool = False,
        entry_price: float = None,
        stop_loss_price: float = None,
    ) -> Tuple[float, Dict]:
        """
        Get available balance with margin info

        Returns:
            (balance, margin_info_dict)
        """
        try:
            if is_futures and self.margin_calculator:
                # Get real-time margin info from Binance
                margin_info = self.margin_calculator.get_available_margin_info(asset)

                # If we have price info, calculate max safe position
                if entry_price and stop_loss_price:
                    max_pos_usd, max_qty = (
                        self.margin_calculator.calculate_max_safe_position(
                            available_margin=margin_info["available_balance"],
                            leverage=margin_info["leverage"],
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                        )
                    )
                    margin_info["max_safe_position_usd"] = max_pos_usd
                    margin_info["max_safe_quantity"] = max_qty

                return margin_info["available_balance"], margin_info
            else:
                # Spot/Portfolio balance
                balance = self.portfolio_manager.get_asset_balance(asset)
                return balance, {"source": "portfolio", "balance": balance}

        except Exception as e:
            logger.error(f"[RISK] Error getting balance: {e}")
            return 0.0, {"error": str(e)}

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
        is_futures: bool = False,
    ) -> Tuple[float, Dict]:
        """
        ✅ ENHANCED: Calculate position size with dynamic margin awareness
        Automatically adjusts to fit available Binance Futures margin
        """
        try:
            # ================================================================
            # STEP 1: Get available balance and margin info
            # ================================================================
            asset_balance, margin_info = self._get_available_balance(
                asset=asset,
                is_futures=is_futures,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
            )

            if asset_balance <= 0:
                logger.error(
                    f"[RISK] No available balance for {asset}\n"
                    f"  Source: {'Futures wallet' if is_futures else 'Portfolio'}\n"
                    f"  Action: {'Fund Futures wallet with USDT' if is_futures else 'Add capital to portfolio'}"
                )
                return 0.0, {
                    "error": "insufficient_balance",
                    "margin_info": margin_info,
                }

            logger.info(
                f"[RISK] Using {'Futures Margin' if is_futures else 'Portfolio'} Balance: ${asset_balance:,.2f}"
            )

            # ================================================================
            # STEP 2: Get existing positions
            # ================================================================
            side = "long" if signal == 1 else "short"
            existing_positions = self.portfolio_manager.get_asset_positions(asset)
            same_side_positions = [p for p in existing_positions if p.side == side]
            total_positions = len(same_side_positions) + 1

            logger.info(
                f"[RISK] Position Count:\n"
                f"  Existing {side.upper()}: {len(same_side_positions)}\n"
                f"  Total (with new):      {total_positions}"
            )

            # ================================================================
            # STEP 3: Calculate target risk per position
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
            # STEP 4: Validate stop loss distance
            # ================================================================
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
                    "min_required": min_stop_pct,
                }

            if stop_distance_pct > max_stop_pct:
                logger.warning(
                    f"[RISK] Stop very wide: {stop_distance_pct:.2%} > {max_stop_pct:.2%}\n"
                    f"  Capping at {max_stop_pct:.2%}"
                )
                stop_distance_pct = max_stop_pct
                if signal == 1:
                    stop_loss_price = entry_price * (1 - max_stop_pct)
                else:
                    stop_loss_price = entry_price * (1 + max_stop_pct)
                stop_distance = abs(entry_price - stop_loss_price)

            # ================================================================
            # STEP 5: Calculate target position size from risk
            # ================================================================
            target_position_size = risk_amount_per_position / stop_distance_pct

            logger.info(
                f"[RISK] Target Position Calculation:\n"
                f"  Risk Amount:     ${risk_amount_per_position:.2f}\n"
                f"  Stop Distance:   ${stop_distance:.2f} ({stop_distance_pct:.2%})\n"
                f"  Target Size:     ${target_position_size:,.2f}"
            )

            # ================================================================
            # STEP 6: Apply Futures margin limits (CRITICAL FIX)
            # ================================================================
            final_position_size = target_position_size
            was_margin_limited = False

            if is_futures and "max_safe_position_usd" in margin_info:
                max_safe = margin_info["max_safe_position_usd"]

                if target_position_size > max_safe:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"⚠️  MARGIN LIMIT HIT - AUTO-ADJUSTING\n"
                        f"{'='*80}\n"
                        f"Target Size:     ${target_position_size:,.2f}\n"
                        f"Max Safe:        ${max_safe:,.2f}\n"
                        f"Available:       ${asset_balance:,.2f} USDT\n"
                        f"Leverage:        {margin_info['leverage']}x\n"
                        f"\n"
                        f"Position will be reduced to fit available margin\n"
                        f"{'='*80}"
                    )

                    final_position_size = max_safe
                    was_margin_limited = True

                    # Recalculate actual risk with reduced size
                    actual_risk = final_position_size * stop_distance_pct
                    actual_risk_pct = actual_risk / asset_balance

                    logger.info(
                        f"[RISK] Adjusted Position:\n"
                        f"  Final Size:    ${final_position_size:,.2f}\n"
                        f"  Actual Risk:   ${actual_risk:,.2f} ({actual_risk_pct:.2%})\n"
                        f"  Reduction:     {(1 - final_position_size/target_position_size):.1%}"
                    )

            # ================================================================
            # STEP 7: Apply config limits
            # ================================================================
            min_size = asset_cfg.get("min_position_usd", 100)
            max_size = asset_cfg.get("max_position_usd", 100000)

            if final_position_size < min_size:
                logger.error(
                    f"[RISK] Position ${final_position_size:.2f} below minimum ${min_size}\n"
                    f"  Possible solutions:\n"
                    f"  1. Add more USDT to Futures wallet\n"
                    f"  2. Reduce leverage (currently {margin_info.get('leverage', '?')}x)\n"
                    f"  3. Lower min_position_usd in config\n"
                    f"  4. Widen stop loss (currently {stop_distance_pct:.2%})"
                )
                return 0.0, {
                    "error": "below_minimum",
                    "calculated": final_position_size,
                    "minimum": min_size,
                    "margin_info": margin_info,
                }

            final_position_size = min(final_position_size, max_size)

            # ================================================================
            # STEP 8: Calculate final metrics
            # ================================================================
            actual_risk = final_position_size * stop_distance_pct
            actual_risk_pct = actual_risk / asset_balance

            # Calculate total risk after this position
            existing_total_risk = sum(
                abs(p.entry_price - p.stop_loss) * p.quantity
                for p in same_side_positions
                if p.stop_loss
            )
            new_total_risk = existing_total_risk + actual_risk
            new_total_risk_pct = new_total_risk / asset_balance

            metadata = {
                "asset": asset,
                "mode": sizing_mode,
                "is_futures": is_futures,
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "stop_distance_pct": stop_distance_pct * 100,
                "asset_balance": asset_balance,
                "target_position_size": target_position_size,
                "final_position_size": final_position_size,
                "actual_risk_usd": actual_risk,
                "actual_risk_pct": actual_risk_pct * 100,
                "total_risk_after": new_total_risk,
                "total_risk_pct_after": new_total_risk_pct * 100,
                "margin_info": margin_info,
                "was_margin_limited": was_margin_limited,
            }

            logger.info(
                f"\n{'='*80}\n"
                f"✅ POSITION APPROVED\n"
                f"{'='*80}\n"
                f"Size:      ${final_position_size:,.2f}\n"
                f"Risk:      ${actual_risk:.2f} ({actual_risk_pct:.2%})\n"
                f"Total:     ${new_total_risk:.2f} ({new_total_risk_pct:.2%})\n"
                f"Source:    {'Futures Margin' if is_futures else 'Portfolio'}\n"
                f"Adjusted:  {'YES - Margin Limited' if was_margin_limited else 'NO - Within Limits'}\n"
                f"{'='*80}"
            )

            return final_position_size, metadata

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

        # ✅ STEP 1: Initialize Futures handler FIRST
        self.futures_handler = None
        futures_enabled = self.asset_config.get("enable_futures", False)

        if futures_enabled:
            try:
                logger.info("[HANDLER] Initializing Binance Futures...")
                self.futures_handler = BinanceFuturesHandler(
                    client=client, symbol=self.symbol
                )

                # Set leverage and margin type
                leverage = self.asset_config.get("leverage", 20)
                margin_type = self.asset_config.get("margin_type", "CROSSED")

                self.futures_handler.set_leverage(leverage)
                self.futures_handler.set_margin_type(margin_type)

                logger.info(
                    f"[HANDLER] ✓ Futures initialized\n"
                    f"  Leverage: {leverage}x\n"
                    f"  Margin:   {margin_type}"
                )

            except Exception as e:
                logger.error(f"[HANDLER] Futures initialization failed: {e}")
                self.futures_handler = None
        else:
            logger.info("[HANDLER] Futures trading disabled in config")

        # ✅ STEP 2: Initialize sizer WITH futures_handler reference
        self.sizer = HybridPositionSizer(
            config, portfolio_manager, futures_handler=self.futures_handler
        )

        # ✅ STEP 3: Initialize rebalancer and connect to sizer
        if self.futures_handler:
            rebalancer = PositionRebalancer(
                futures_handler=self.futures_handler,
                portfolio_manager=self.portfolio_manager,
            )
            self.sizer.set_rebalancer(rebalancer)
            logger.info("[HANDLER] ✓ Auto-rebalancing enabled")

        logger.info(f"BinanceExecutionHandler initialized - Mode: {self.mode.upper()}")

        # ✅ STEP 4: Auto-sync on startup (if enabled)
        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")

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
        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(
            asset_name, side
        )
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        # Check max positions per side
        current_count = self.portfolio_manager.get_asset_position_count(
            asset_name, side
        )
        max_per_asset = self.max_positions_per_asset

        if current_count >= max_per_asset:
            return (
                False,
                f"Already have {current_count}/{max_per_asset} {side.upper()} positions",
            )

        return (
            True,
            f"OK - {current_count}/{max_per_asset} {side.upper()} positions open",
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
        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(
            asset_name, side
        )
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        # Check max positions per side
        current_count = self.portfolio_manager.get_asset_position_count(
            asset_name, side
        )
        max_per_asset = self.max_positions_per_asset

        if current_count >= max_per_asset:
            return (
                False,
                f"Already have {current_count}/{max_per_asset} {side.upper()} positions",
            )

        return (
            True,
            f"OK - {current_count}/{max_per_asset} {side.upper()} positions open",
        )

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
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
        default_return=False,
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
        ✅ ENHANCED: Execute trading signal with Asymmetric Risk Support
        
        Signal Logic:
        - BUY (+1):  Close ALL shorts → Open long
        - SELL (-1): Close ALL longs  → Open short
        - HOLD (0):  Check SL/TP only
        
        ✨ NEW: Trade Type Support
        - TREND trades: 2% risk, standard management
        - SCALP trades: 1% risk, aggressive break-even
        - V_SHAPE trades: 1.5% risk, moderate management
        
        ✨ NEW: Hedging Support
        - Can maintain long + short simultaneously if enabled
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
            # ================================================================
            # STEP 1: Get Current Price
            # ================================================================
            if current_price is None:
                current_price = self.get_current_price()

            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid price: {current_price}")
                return False

            # ================================================================
            # STEP 2: Extract Trade Type from Signal Details
            # ================================================================
            trade_type = "TREND"  # Default
            if signal_details:
                trade_type = signal_details.get('trade_type', 'TREND')
            
            # ================================================================
            # STEP 3: Get Existing Positions
            # ================================================================
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)
            long_positions = [p for p in existing_positions if p.side == "long"]
            short_positions = [p for p in existing_positions if p.side == "short"]

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} Signal: {signal:+2d} | Trade Type: {trade_type}\n"
                f"[STATE] Current Positions: {len(long_positions)} LONG, {len(short_positions)} SHORT\n"
                f"{'='*80}"
            )

            # Log hybrid mode if present
            if signal_details and signal_details.get("aggregator_mode"):
                logger.info(
                    f"\n[HYBRID] Mode: {signal_details['aggregator_mode'].upper()} "
                    f"({signal_details.get('mode_confidence', 0):.0%} confidence)"
                )
            
            # Log filter results if present
            if signal_details and signal_details.get("world_class_filters"):
                filters = signal_details["world_class_filters"]
                logger.info(f"\n[FILTERS] Status:")
                logger.info(f"  Governor:   {'✅' if filters.get('governor_passed') else '❌'}")
                logger.info(f"  Volatility: {'✅' if filters.get('volatility_passed') else '❌'}")
                logger.info(f"  Sniper:     {'✅' if filters.get('sniper_passed') else '❌'}")
                logger.info(f"  Profit:     {'✅' if filters.get('profit_passed') else '❌'}")

            # ================================================================
            # STEP 4: Check Hedging Configuration
            # ================================================================
            hedging_enabled = self.config.get("trading", {}).get("allow_hedging", False)
            
            if hedging_enabled:
                logger.info(f"[HEDGING] Enabled - Can maintain long + short simultaneously")

            # ================================================================
            # SCENARIO 1: SELL SIGNAL (-1)
            # ================================================================
            if signal == -1:
                # Close all longs (UNLESS hedging is enabled)
                if long_positions and not hedging_enabled:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📉 SELL SIGNAL - Closing {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    for i, position in enumerate(long_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(long_positions)}] Closing LONG position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  Entry:       ${position.entry_price:,.2f}\n"
                            f"  Current:     ${current_price:,.2f}"
                        )

                        if self._close_position(
                            position, current_price, asset_name, "sell_signal"
                        ):
                            closed_count += 1

                    logger.info(
                        f"\n[SUMMARY] Closed {closed_count}/{len(long_positions)} positions\n"
                    )
                
                elif long_positions and hedging_enabled:
                    logger.info(
                        f"\n[HEDGING] Keeping {len(long_positions)} LONG position(s) open "
                        f"(hedging enabled)"
                    )

                # Open SHORT
                can_open, reason = self.can_open_position_side(asset_name, "short")
                if not can_open:
                    logger.warning(f"[SKIP] Cannot open SHORT: {reason}")
                    return len(long_positions) > 0 and not hedging_enabled

                logger.info(
                    f"\n{'='*80}\n"
                    f"📉 SELL SIGNAL - Opening new SHORT position\n"
                    f"Trade Type: {trade_type}\n"
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
                    signal_details=signal_details,  # Contains trade_type
                )

            # ================================================================
            # SCENARIO 2: BUY SIGNAL (+1)
            # ================================================================
            elif signal == 1:
                # Close all shorts (UNLESS hedging is enabled)
                if short_positions and not hedging_enabled:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Closing {len(short_positions)} SHORT position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    for i, position in enumerate(short_positions, 1):
                        logger.info(
                            f"\n[{i}/{len(short_positions)}] Closing SHORT position:\n"
                            f"  Position ID: {position.position_id}\n"
                            f"  Entry:       ${position.entry_price:,.2f}\n"
                            f"  Current:     ${current_price:,.2f}"
                        )

                        if self._close_position(
                            position, current_price, asset_name, "buy_signal"
                        ):
                            closed_count += 1

                    logger.info(
                        f"\n[SUMMARY] Closed {closed_count}/{len(short_positions)} positions\n"
                    )
                
                elif short_positions and hedging_enabled:
                    logger.info(
                        f"\n[HEDGING] Keeping {len(short_positions)} SHORT position(s) open "
                        f"(hedging enabled)"
                    )

                # Open LONG
                can_open, reason = self.can_open_position_side(asset_name, "long")
                if not can_open:
                    logger.warning(f"[SKIP] Cannot open LONG: {reason}")
                    return len(short_positions) > 0 and not hedging_enabled

                logger.info(
                    f"\n{'='*80}\n"
                    f"📈 BUY SIGNAL - Opening new LONG position\n"
                    f"Trade Type: {trade_type}\n"
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
                    signal_details=signal_details,  # Contains trade_type
                )

            # ================================================================
            # SCENARIO 3: HOLD SIGNAL (0)
            # ================================================================
            elif signal == 0:
                if not existing_positions:
                    return False

                positions_closed = False
                for position in existing_positions:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        position, current_price
                    )
                    if should_close:
                        logger.info(
                            f"[AUTO-CLOSE] {position.position_id}: {close_reason}"
                        )
                        if self._close_position(
                            position, current_price, asset_name, close_reason
                        ):
                            positions_closed = True

                return positions_closed

            return False

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False
        
        
    def _calculate_asymmetric_risk(
        self, 
        trade_type: str, 
        base_risk: float = 0.015
    ) -> Tuple[float, Dict]:
        """
        ✨ NEW: Calculate risk based on trade type
        
        Args:
            trade_type: "TREND", "SCALP", or "V_SHAPE"
            base_risk: Base risk percentage (default 1.5%)
        
        Returns:
            (adjusted_risk, risk_profile)
        """
        risk_profiles = {
            "TREND": {
                "multiplier": 1.33,  # 2% risk (1.5% * 1.33)
                "description": "Full trend trade",
                "break_even_trigger": 0.015,  # Move to BE at +1.5%
                "trailing_stop": 0.025,  # 2.5% trailing
            },
            "SCALP": {
                "multiplier": 0.67,  # 1% risk (1.5% * 0.67)
                "description": "Conservative scalp",
                "break_even_trigger": 0.005,  # Move to BE at +0.5%
                "trailing_stop": 0.015,  # 1.5% trailing
            },
            "V_SHAPE": {
                "multiplier": 1.0,  # 1.5% risk (1.5% * 1.0)
                "description": "Recovery play",
                "break_even_trigger": 0.010,  # Move to BE at +1.0%
                "trailing_stop": 0.020,  # 2.0% trailing
            },
        }
        
        profile = risk_profiles.get(trade_type, risk_profiles["TREND"])
        adjusted_risk = base_risk * profile["multiplier"]
        
        logger.info(f"\n[RISK CALC] Trade Type: {trade_type}")
        logger.info(f"  Base Risk:     {base_risk:.2%}")
        logger.info(f"  Multiplier:    {profile['multiplier']:.2f}x")
        logger.info(f"  Adjusted Risk: {adjusted_risk:.2%}")
        logger.info(f"  Description:   {profile['description']}")
        
        return adjusted_risk, profile
        
    

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit (fallback for non-VTM)"""
        try:
            if not position.stop_loss and not position.take_profit:
                return False, ""

            side = position.side
            entry_price = position.entry_price
            stop_loss = position.stop_loss
            take_profit = position.take_profit

            price_tolerance = 0.50

            if side == "long":
                if stop_loss and current_price <= (stop_loss + price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"stop_loss_hit ({pnl_pct:+.2f}%)"

                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"take_profit_hit ({pnl_pct:+.2f}%)"

            elif side == "short":
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return True, f"stop_loss_hit ({pnl_pct:+.2f}%)"

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return True, f"take_profit_hit ({pnl_pct:+.2f}%)"

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
            order = self.client.order_market_sell(symbol=self.symbol, quantity=quantity)

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
        futures_enabled = (
            handler.config.get("assets", {}).get("BTC", {}).get("enable_futures", False)
        )

        if not futures_enabled:
            logger.warning("[FUTURES] Futures trading disabled in config")
            return False

        try:
            # Initialize Futures handler
            handler.futures_handler = BinanceFuturesHandler(
                client=handler.client, symbol=handler.symbol
            )

            # Set leverage and margin type
            leverage = (
                handler.config.get("assets", {}).get("BTC", {}).get("leverage", 20)
            )
            margin_type = (
                handler.config.get("assets", {})
                .get("BTC", {})
                .get("margin_type", "CROSSED")
            )
            is_futures = True

            handler.futures_handler.set_leverage(leverage)
            handler.futures_handler.set_margin_type(margin_type)
            handler.futures_handler.is_futures = is_futures

            logger.info("[FUTURES] ✓ Futures handler integrated")
            return True

        except Exception as e:
            logger.error(f"[FUTURES] Integration failed: {e}")
            return False

    def _round_quantity_precision(
        self, quantity: float, symbol: str = "BTCUSDT", is_futures: bool = False
    ) -> float:
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

    def _round_quantity(
        self, quantity: float, symbol: str = "BTCUSDT", is_futures: bool = False
    ) -> float:
        """Round quantity to correct lot size and precision for Binance"""
        try:
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
                            precision = len(str(step_size).rstrip("0").split(".")[-1])

                            rounded_qty = round(quantity / step_size) * step_size
                            rounded_qty = round(rounded_qty, precision)
                            rounded_qty = max(min_qty, rounded_qty)

                            return rounded_qty

            return round(quantity, 5)

        except Exception as e:
            logger.error(f"Error rounding quantity: {e}")
            return round(quantity, 5)

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
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
        ✅ COMPLETE: Open LONG or SHORT with dynamic margin + VTM
        """
        try:
            side = "long" if signal == 1 else "short"
            trade_type = signal_details.get('trade_type', 'TREND') if signal_details else 'TREND'
            base_risk = self.config["portfolio"]["target_risk_per_trade"]
            adjusted_risk, risk_profile = self._calculate_asymmetric_risk(trade_type, base_risk)
            # Determine if using Futures
            is_futures = (
                hasattr(self, "futures_handler")
                and self.futures_handler is not None
                and self.config.get("assets", {})
                .get(asset_name, {})
                .get("enable_futures", False)
            )

            logger.info(
                f"\n{'='*80}\n"
                f"[OPEN] Opening {side.upper()} position for {asset_name}\n"
                f"  Mode: {'Futures' if is_futures else 'Spot'}\n"
                f"{'='*80}"
            )

            # Calculate safety stop
            risk_config = self.asset_config.get("risk", {})
            sl_pct = risk_config.get("stop_loss_pct", 0.02)

            if side == "long":
                safety_sl_price = current_price * (1 - sl_pct)
            else:
                safety_sl_price = current_price * (1 + sl_pct)

            if is_futures:
                safety_sl_price = round(safety_sl_price, 2)

            # ✅ CRITICAL: Calculate position size WITH margin awareness
            position_size_usd, sizing_metadata = self.sizer.calculate_size_risk_based(
                asset=asset_name,
                entry_price=current_price,
                stop_loss_price=safety_sl_price,
                signal=signal,
                confidence_score=confidence_score,
                market_condition=market_condition or "neutral",
                sizing_mode=sizing_mode,
                manual_size_usd=manual_size_usd,
                override_reason=override_reason,
                is_futures=is_futures,  # ← Tells sizer to check Futures margin
            )

            # Check if margin limited
            if sizing_metadata.get("was_margin_limited"):
                logger.warning(
                    f"[MARGIN] Position auto-adjusted to fit available margin:\n"
                    f"  Target: ${sizing_metadata.get('target_position_size', 0):,.2f}\n"
                    f"  Final:  ${position_size_usd:,.2f}"
                )

            # Apply short reduction if configured
            if side == "short":
                short_config = self.config.get("portfolio", {}).get(
                    "short_position_sizing", {}
                )
                use_reduced = short_config.get("use_reduced_size_for_shorts", False)
                multiplier = short_config.get("short_size_multiplier", 0.8)
                if use_reduced and multiplier < 1.0:
                    position_size_usd *= multiplier
                    logger.info(f"[SHORT] Applied {multiplier}x size reduction")

            if position_size_usd <= 0:
                logger.error(f"[OPEN] Invalid size: ${position_size_usd:.2f}")
                return False

            # Merge sizing into signal details
            if signal_details is None:
                signal_details = {}
            signal_details["sizing"] = sizing_metadata

            # Calculate quantity
            quantity = position_size_usd / current_price

            leverage = 1
            margin_type = "SPOT"

            if is_futures:
                asset_conf = self.config.get("assets", {}).get(asset_name, {})
                leverage = asset_conf.get("leverage", 20)
                margin_type = asset_conf.get("margin_type", "CROSSED")
                quantity = self.futures_handler._round_quantity(quantity)
            else:
                quantity = self._round_quantity(quantity, self.symbol, False)

            MIN_BTC = 0.00001
            if quantity < MIN_BTC:
                logger.warning(f"[OPEN] Quantity {quantity:.8f} below minimum")
                return False

            logger.info(
                f"[OPEN] Position Details:\n"
                f"  Quantity:  {quantity:.8f} BTC\n"
                f"  Entry:     ${current_price:,.2f}\n"
                f"  Size:      ${position_size_usd:,.2f}\n"
                f"  Safety SL: ${safety_sl_price:,.2f}\n"
                f"  Leverage:  {leverage}x"
            )

            # Execute on exchange
            order_id = None

            if is_futures:
                try:
                    if side == "long":
                        order = self.futures_handler.open_long_position(
                            quantity=quantity,
                            stop_loss=safety_sl_price,
                            take_profit=None,
                        )
                    else:
                        order = self.futures_handler.open_short_position(
                            quantity=quantity,
                            stop_loss=safety_sl_price,
                            take_profit=None,
                        )

                    if order:
                        order_id = order.get("orderId")
                        logger.info(
                            f"[FUTURES] ✓ {side.upper()} opened (Order: {order_id})"
                        )
                    else:
                        logger.error(f"[FUTURES] Failed to open {side.upper()}")
                        return False

                except Exception as e:
                    logger.error(f"[FUTURES] Error: {e}")
                    return False
            else:
                if not self.is_paper_mode:
                    try:
                        if side == "long":
                            order = self.client.order_market_buy(
                                symbol=self.symbol, quantity=quantity
                            )
                            order_id = order.get("orderId")
                        else:
                            logger.error("[SPOT] SHORT requires Futures API")
                            return False
                    except Exception as e:
                        logger.error(f"[SPOT] Error: {e}")
                        return False
                else:
                    order_id = f"PAPER_{side.upper()}_{int(time.time())}"
                    logger.info(f"[PAPER] Simulated order: {order_id}")

            # Fetch OHLC for VTM
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

            # Add to Portfolio Manager
            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=None,  # VTM manages
                take_profit=None,
                trailing_stop_pct=None,
                binance_order_id=order_id,
                ohlc_data=ohlc_data,
                use_dynamic_management=True,
                signal_details=signal_details,
                leverage=leverage,
                margin_type=margin_type,
                is_futures=is_futures,
            )

            if success:
                logger.info(
                    f"\n{'='*80}\n"
                    f"✅ {asset_name} {side.upper()} POSITION OPENED\n"
                    f"{'='*80}\n"
                    f"Order ID:   {order_id}\n"
                    f"Size:       ${position_size_usd:,.2f}\n"
                    f"VTM:        {'ACTIVE' if ohlc_data else 'INACTIVE'}\n"
                    f"{'='*80}\n"
                )
                return True
            else:
                logger.error(f"[FAIL] Portfolio rejected {side.upper()} position")

                # Rollback
                if order_id and not self.is_paper_mode:
                    try:
                        if is_futures:
                            if side == "long":
                                self.futures_handler.close_long_position(
                                    quantity=quantity
                                )
                            else:
                                self.futures_handler.close_short_position(
                                    quantity=quantity
                                )
                        else:
                            self.client.order_market_sell(
                                symbol=self.symbol, quantity=quantity
                            )
                        logger.info("[ROLLBACK] ✓ Position closed")
                    except Exception as e:
                        logger.error(f"[ROLLBACK] Failed: {e}")

                return False

        except Exception as e:
            logger.error(f"[OPEN] Error: {e}", exc_info=True)
            return False

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """Close LONG or SHORT position"""
        try:
            side = position.side
            quantity = position.quantity
            order_id = position.binance_order_id

            logger.info(f"[CLOSE] Closing {side.upper()} ({reason})")

            # Try Futures first
            if hasattr(self, "futures_handler") and self.futures_handler:
                try:
                    if side == "long":
                        success = self.futures_handler.close_long_position(
                            quantity=quantity, order_id=order_id
                        )
                    else:
                        success = self.futures_handler.close_short_position(
                            quantity=quantity, order_id=order_id
                        )

                    if success:
                        logger.info(f"[FUTURES] ✓ {side.upper()} closed")
                    else:
                        success = False

                except Exception as e:
                    logger.warning(f"[FUTURES] Error: {e}")
                    success = False

            # Fallback to spot
            if not hasattr(self, "futures_handler") or not success:
                if not self.is_paper_mode:
                    try:
                        if side == "long":
                            self.client.order_market_sell(
                                symbol=self.symbol, quantity=quantity
                            )
                            success = True
                        else:
                            logger.error("[SPOT] SHORT requires Futures")
                            success = False
                    except Exception as e:
                        logger.error(f"[SPOT] Error: {e}")
                        success = False
                else:
                    logger.info(f"[PAPER] Simulated close: {order_id}")
                    success = True

            # Close in portfolio
            if success:
                trade_result = self.portfolio_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=reason,
                )
                return trade_result is not None

            return False

        except Exception as e:
            logger.error(f"[CLOSE] Error: {e}", exc_info=True)
            return False

    def check_and_update_positions_VTM(self, asset_name: str = "BTC"):
        """Check and update ALL positions with VTM"""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset_name)
            if not positions:
                return False

            current_price = self.get_current_price()
            if not current_price:
                return False

            positions_closed = False

            for position in positions:
                # VTM check
                if position.trade_manager:
                    exit_signal = position.trade_manager.update_with_current_price(
                        current_price
                    )

                    if exit_signal:
                        exit_reason = exit_signal.get("reason", "unknown")

                        if hasattr(exit_reason, "value"):
                            exit_reason_str = exit_reason.value
                        else:
                            exit_reason_str = str(exit_reason)

                        logger.info(
                            f"[VTM] {position.position_id} triggered {exit_reason_str.upper()}"
                        )

                        self._close_position(
                            position,
                            current_price,
                            asset_name,
                            f"VTM_{exit_reason_str}",
                        )
                        positions_closed = True
                        continue

                # Fallback SL/TP check
                should_close, reason = self._check_stop_loss_take_profit(
                    position, current_price
                )
                if should_close:
                    logger.info(f"[SL/TP] {position.position_id}: {reason}")
                    self._close_position(position, current_price, asset_name, reason)
                    positions_closed = True

            return positions_closed

        except Exception as e:
            logger.error(f"[VTM] Error: {e}", exc_info=True)
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
        default_return=False,
    )
    def sync_positions_with_binance(
        self, asset_name: str = "BTC", symbol: str = None
    ) -> bool:
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
                longs = sum(1 for p in portfolio_positions if p.side == "long")
                shorts = sum(1 for p in portfolio_positions if p.side == "short")
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
                        pos_amt = float(pos_info.get("positionAmt", 0))
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
                            logger.info(
                                f"[SYNC] Detected Spot Balance: {binance_qty:.8f} {asset_name}"
                            )
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
                    self.config.get("portfolio", {}).get(
                        "import_existing_positions", False
                    )
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
                            interval=self.config["assets"][asset_name].get(
                                "interval", "1h"
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
                    except Exception as e:
                        logger.error(f"[VTM] Failed to fetch OHLC: {e}")

                    # Generate signal details
                    signal_details = {
                        "imported": True,
                        "import_time": datetime.now().isoformat(),
                        "source": "futures" if using_futures else "spot",
                        "aggregator_mode": "unknown",
                        "reasoning": f"{asset_name} {binance_side} position imported from Binance",
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
                        trailing_stop_pct=self.config["assets"][asset_name]
                        .get("risk", {})
                        .get("trailing_stop_pct"),
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
            futures_positions = (
                self.futures_handler.client.futures_position_information(symbol=symbol)
            )

            # Find active Futures positions
            active_futures = []
            for pos in futures_positions:
                pos_amt = float(pos.get("positionAmt", 0))
                if pos_amt != 0:
                    side = "long" if pos_amt > 0 else "short"
                    active_futures.append(
                        {
                            "side": side,
                            "quantity": abs(pos_amt),
                            "entry_price": float(pos.get("entryPrice", 0)),
                            "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                        }
                    )

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
                import_enabled = self.config.get("portfolio", {}).get(
                    "import_existing_positions", False
                )

                if not import_enabled:
                    logger.warning(
                        f"[SYNC] Found {len(active_futures)} Futures position(s) but import disabled\n"
                        f"  → Enable 'import_existing_positions' in config to import them"
                    )
                    return True

                logger.info(
                    f"[SYNC] Importing {len(active_futures)} Futures position(s)..."
                )

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
                        "imported": True,
                        "import_time": datetime.now().isoformat(),
                        "aggregator_mode": "unknown",
                        "mode_confidence": 0.5,
                        "regime_analysis": {
                            "regime_type": "unknown",
                            "trend_strength": "unknown",
                            "volatility_regime": "normal",
                        },
                        "signal_quality": 0.5,
                        "reasoning": f'{fut_pos["side"].upper()} position imported from Binance Futures',
                    }

                    # Calculate position size
                    position_size_usd = fut_pos["quantity"] * fut_pos["entry_price"]

                    # Import position
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side=fut_pos["side"],
                        entry_price=fut_pos["entry_price"],
                        position_size_usd=position_size_usd,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=self.asset_config.get("risk", {}).get(
                            "trailing_stop_pct"
                        ),
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
                        logger.error(
                            f"[SYNC] ✗ Failed to import {fut_pos['side'].upper()} position"
                        )

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
                    "long": [p for p in portfolio_positions if p.side == "long"],
                    "short": [p for p in portfolio_positions if p.side == "short"],
                }

                futures_by_side = {
                    "long": [f for f in active_futures if f["side"] == "long"],
                    "short": [f for f in active_futures if f["side"] == "short"],
                }

                # Check each side
                for side in ["long", "short"]:
                    port_qty = sum(p.quantity for p in portfolio_by_side[side])
                    fut_qty = sum(f["quantity"] for f in futures_by_side[side])

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
                import_enabled = self.config.get("portfolio", {}).get(
                    "import_existing_positions", False
                )

                if not import_enabled:
                    logger.warning(
                        f"[SYNC] BTC balance {btc_balance:.6f} detected but import disabled\n"
                        f"  → Bot will open new positions on BUY signals"
                    )
                    return True

                logger.info(
                    f"[SYNC] Importing {btc_balance:.6f} BTC as LONG position..."
                )

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
                    trailing_stop_pct=self.asset_config.get("risk", {}).get(
                        "trailing_stop_pct"
                    ),
                    binance_order_id=None,
                    ohlc_data=ohlc_data,
                    use_dynamic_management=True,
                    entry_time=datetime.now(),
                    signal_details={
                        "imported": True,
                        "import_time": datetime.now().isoformat(),
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
