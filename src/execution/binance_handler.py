"""
Binance Execution Handler with Hybrid Position Sizing
INTEGRATED: Automated risk management + manual override support
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
            base_size = self._calculate_automated_size(
                request.asset,
                request.current_price,
                request.signal
            )
            
            confidence_adjusted = self._apply_confidence_adjustment(
                base_size,
                request.confidence_score,
                request.market_condition
            )
            
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
            
            final_size = self._apply_hard_limits(
                request.asset,
                final_size,
                request.signal
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
        base_pct = self.portfolio_cfg.get("base_position_size", 0.10)
        base_size = self.portfolio_manager.current_capital * base_pct
        
        asset_weight = asset_cfg.get("weight", 1.0)
        base_size *= asset_weight
        
        if signal == -1:
            base_size *= 0.8
        
        max_risk_pct = self.risk_cfg.get("max_risk_per_trade", 0.02)
        max_risk_usd = self.portfolio_manager.current_capital * max_risk_pct
        
        risk_cfg = asset_cfg.get("risk", {})
        stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.02)
        
        max_size_by_risk = max_risk_usd / stop_loss_pct if stop_loss_pct > 0 else base_size
        base_size = min(base_size, max_size_by_risk)
        
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
        
        confidence_scalar = 0.5 + (confidence_score * 1.0)
        confidence_scalar = max(0.3, min(1.5, confidence_scalar))
        adjusted_size = base_size * confidence_scalar
        
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
        
        min_size = asset_cfg.get("min_position_usd", 100)
        max_size = asset_cfg.get("max_position_usd", 6000)
        max_exposure = self.portfolio_cfg.get("max_portfolio_exposure", 0.95)
        max_single_asset = self.portfolio_cfg.get("max_single_asset_exposure", 0.60)
        
        if position_size < min_size:
            logger.debug(f"Position size ${position_size:,.2f} below minimum ${min_size}")
            return 0.0
        
        position_size = min(position_size, max_size)
        
        current_exposure = self._calculate_current_exposure()
        max_portfolio_usd = self.portfolio_manager.current_capital * max_exposure
        if current_exposure + position_size > max_portfolio_usd:
            position_size = max(0, max_portfolio_usd - current_exposure)
            logger.warning(f"Position clamped to portfolio limit: ${position_size:,.2f}")
        
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
    Binance Execution Handler with Hybrid Position Sizing
    """

    def __init__(self, config: Dict, client: Client, portfolio_manager):
        self.config = config
        self.client = client
        self.portfolio_manager = portfolio_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)

        self.asset_config = config["assets"]["BTC"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]

        self.symbol = self.asset_config["symbol"]
        self.mode = self.trading_config.get("mode", "paper")

        logger.info(f"BinanceExecutionHandler with HybridPositionSizer initialized - Mode: {self.mode.upper()}")

        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")

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
        current_price: float,
        asset_name: str = "BTC",
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
            current_price: Current market price
            asset_name: Asset name
            confidence_score: Signal confidence (0.0 to 1.0)
            market_condition: "bullish", "bearish", "neutral", "uncertain"
            sizing_mode: AUTOMATED, MANUAL_OVERRIDE, REDUCED_RISK, ELEVATED_RISK
            manual_size_usd: Manual position size (only if sizing_mode=MANUAL_OVERRIDE)
            override_reason: Reason for override

        Returns:
            True if execution successful, False otherwise
        """
        try:
            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid current price: {current_price}")
                return False

            existing_position = self.portfolio_manager.get_position(asset_name)

            if existing_position:
                position_side = (
                    existing_position.side
                    if hasattr(existing_position, "side")
                    else existing_position.get("side")
                )

                logger.debug(
                    f"{asset_name}: Existing {position_side.upper()} position found, signal={signal}"
                )

                # === CLOSE POSITION SCENARIOS ===

                if signal == -1 and position_side == "long":
                    logger.info(
                        f"[SIGNAL] {asset_name}: SELL signal received - Closing LONG position"
                    )
                    success = self._close_position(
                        existing_position, current_price, asset_name, "sell_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset_name} LONG position")
                        return False
                    return True

                elif signal == 1 and position_side == "short":
                    logger.info(
                        f"[SIGNAL] {asset_name}: BUY signal received - Closing SHORT position"
                    )
                    success = self._close_position(
                        existing_position, current_price, asset_name, "buy_signal"
                    )
                    if not success:
                        logger.error(f"[FAIL] Failed to close {asset_name} SHORT position")
                        return False

                elif signal == 0:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        existing_position, current_price
                    )
                    if should_close:
                        logger.info(f"[AUTO-CLOSE] {asset_name}: {close_reason}")
                        success = self._close_position(
                            existing_position, current_price, asset_name, close_reason
                        )
                        if not success:
                            logger.error(
                                f"[FAIL] Failed to close {asset_name} position on {close_reason}"
                            )
                            return False
                    else:
                        logger.debug(f"{asset_name}: Holding position, no action needed")
                    return True

                elif signal == 1 and position_side == "long":
                    logger.info(
                        f"[SKIP] {asset_name}: BUY signal ignored - Already have LONG position"
                    )
                    return True

                elif signal == -1 and position_side == "short":
                    logger.info(
                        f"[SKIP] {asset_name}: SELL signal ignored - Already have SHORT position"
                    )
                    return True

            # STEP 3: Open new position with HYBRID SIZING
            if signal == 1:
                if self.portfolio_manager.has_position(asset_name, side="long"):
                    logger.warning(
                        f"[SKIP] {asset_name}: BUY signal - Position already exists"
                    )
                    return True

                logger.info(f"[SIGNAL] {asset_name}: BUY signal - Opening LONG position")
                return self._open_position(
                    signal,
                    current_price,
                    asset_name,
                    confidence_score,
                    market_condition,
                    sizing_mode,
                    manual_size_usd,
                    override_reason,
                )

            elif signal == -1:
                logger.debug(f"{asset_name}: SELL signal - No position to close, no action")
                return True

            return True

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
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

            else:
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

    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """Close existing position"""
        try:
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                quantity = position.quantity
                side = position.side
            else:
                entry_price = position["entry_price"]
                quantity = position.get("quantity", 0)
                side = position["side"]

            position_size_usd = quantity * entry_price

            if side == "long":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            pnl_pct = (pnl / position_size_usd) * 100 if position_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset_name} {side.upper()} - Entry: ${entry_price:,.2f}, "
                f"Exit: ${current_price:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) - {reason}"
            )

            if self.mode.lower() != "paper":
                logger.warning(
                    f"[LIVE MODE] Actual Binance order placement not implemented yet"
                )

            trade_result = self.portfolio_manager.close_position(
                asset=asset_name, exit_price=current_price, reason=reason
            )

            if trade_result:
                logger.info(f"[OK] {asset_name} {side.upper()} position closed")
                return True
            else:
                logger.error(
                    f"[FAIL] Portfolio manager failed to close {asset_name} position"
                )
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
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
            can_open, reason = self.portfolio_manager.can_open_position(
                asset_name, side
            )
            if not can_open:
                logger.warning(f"[SKIP] Cannot open {asset_name} position: {reason}")
                return True

            # ========== HYBRID POSITION SIZING ==========
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
            # ==========================================

            if position_size_usd <= 0:
                logger.warning(
                    f"{asset_name}: Invalid position size: ${position_size_usd:.2f}, skipping trade"
                )
                return False

            quantity = position_size_usd / current_price

            min_quantity = 0.00001
            if quantity < min_quantity:
                logger.warning(
                    f"{asset_name}: Quantity {quantity:.8f} below minimum {min_quantity}, skipping"
                )
                return False

            order_side = "BUY" if signal == 1 else "SELL"

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
                f"[OPEN] {order_side} {quantity:.8f} {self.symbol} @ ${current_price:,.2f}"
            )
            logger.info(f"   Size: ${position_size_usd:,.2f}")
            logger.info(f"   Mode: {sizing_mode} | Confidence: {confidence_score}")
            logger.info(f"   SL: ${stop_loss:,.2f} ({stop_loss_pct:.1%})")
            logger.info(f"   TP: ${take_profit:,.2f} ({take_profit_pct:.1%})")

            if self.mode.lower() != "paper":
                logger.warning(
                    f"[LIVE MODE] Actual Binance order placement not implemented yet"
                )

            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
            )

            if success:
                logger.info(
                    f"[OK] {order_side} {asset_name} - Position opened successfully"
                )
                return True
            else:
                logger.error(f"[FAIL] Portfolio Manager rejected {asset_name} position")
                return False

        except Exception as e:
            logger.error(f"Error opening position: {e}", exc_info=True)
            return False

    def check_and_update_positions(self, asset_name: str = "BTC"):
        """Actively check and update all positions (for HOLD signals)"""
        try:
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
        """Sync portfolio manager with actual Binance Spot holdings"""
        if symbol is None:
            symbol = self.symbol
        try:
            logger.info(f"[SYNC] Starting position sync for {asset_name}...")
            account = self.client.get_account()
            portfolio_position = self.portfolio_manager.get_position(asset_name)
            
            btc_balance = 0.0
            usdt_balance = 0.0
            for balance in account['balances']:
                if balance['asset'] == 'BTC':
                    btc_balance = float(balance['free']) + float(balance['locked'])
                elif balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free']) + float(balance['locked'])
            
            current_price = self.get_current_price(symbol)
            if current_price is None:
                logger.error(f"[SYNC] Failed to get current price for {symbol}")
                return False
            
            MIN_BTC_BALANCE = 0.0001
            
            if btc_balance > MIN_BTC_BALANCE and not portfolio_position:
                logger.info(
                    f"[SYNC] BTC balance {btc_balance:.8f} BTC detected on Binance.\n"
                    f"  → This is NOT a bot-managed position. The bot will only open a new position on a BUY signal."
                )
                return True
            
            if portfolio_position and btc_balance <= MIN_BTC_BALANCE:
                logger.warning(
                    f"[SYNC] Portfolio shows {asset_name} position but Binance balance is {btc_balance:.8f} BTC\n"
                    f" → Removing position (likely sold manually)."
                )
                trade_result = self.portfolio_manager.close_position(
                    asset=asset_name,
                    exit_price=current_price,
                    reason="sync_missing_binance"
                )
                if trade_result:
                    logger.info(f"[SYNC] ✓ Removed {asset_name} position from portfolio")
                    return True
                else:
                    logger.error(f"[SYNC] ✗ Failed to remove {asset_name} position")
                    return False
            
            if btc_balance > MIN_BTC_BALANCE and portfolio_position:
                portfolio_qty = getattr(portfolio_position, "quantity", 0)
                qty_diff = abs(btc_balance - portfolio_qty)
                qty_diff_pct = (qty_diff / btc_balance * 100) if btc_balance > 0 else 0
                if qty_diff_pct > 0.1:
                    logger.warning(
                        f"[SYNC] QUANTITY MISMATCH:\n"
                        f"  Binance: {btc_balance:.8f} BTC\n"
                        f"  Portfolio: {portfolio_qty:.8f} BTC\n"
                        f"  Difference: {qty_diff:.8f} BTC ({qty_diff_pct:.2f}%)\n"
                        f"  → Closing position and clearing mismatch"
                    )
                    self.portfolio_manager.close_position(
                        asset_name,
                        current_price,
                        reason="sync_quantity_mismatch"
                    )
                    return True
                else:
                    position_value = btc_balance * current_price
                    logger.info(
                        f"[SYNC] ✓ {asset_name} position already in sync\n"
                        f"  Balance: {btc_balance:.8f} BTC\n"
                        f"  Value: ${position_value:,.2f}"
                    )
                    return True
            
            logger.info(f"[SYNC] ✓ No {asset_name} positions detected on Binance or portfolio")
            return True
        except Exception as e:
            logger.error(f"[SYNC] Error syncing positions: {e}", exc_info=True)
            return False