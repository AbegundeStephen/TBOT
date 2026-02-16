"""
Portfolio Manager - Enhanced with MT5 real-time profit tracking
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pickle
from pathlib import Path
from src.execution.veteran_trade_manager import VeteranTradeManager
from src.global_error_handler import handle_errors, ErrorSeverity
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)


class Position:
    """Represents a single trading position"""

    def __init__(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        risk_config: dict,
        signal_details: dict = None,
        position_id: str = None,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,
        account_balance: float = None,
        use_dynamic_management: bool = True,
        disable_partials: bool = False,
        vtm_overrides: Optional[Dict] = None,
        leverage: int = 1,
        margin_type: str = "SPOT",
        is_futures: bool = False,
    ):
        self.asset = asset
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.position_id = (
            position_id or f"{asset}_{side}_{int(entry_time.timestamp())}"
        )
        self.leverage = leverage
        self.margin_type = margin_type
        self.is_futures = is_futures
        self.closing = False

        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop_pct = None
        self.highest_price = entry_price if side == "long" else entry_price
        self.lowest_price = entry_price if side == "short" else entry_price

        # Exchange-specific tracking
        self.mt5_ticket = mt5_ticket
        self.mt5_profit = 0.0
        self.mt5_last_update = None
        self.binance_order_id = binance_order_id
        self.binance_profit = 0.0
        self.binance_last_update = None

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None

        self.db_trade_id = None
        self.db_manager = None

        # ✅ CRITICAL FIX: Initialize VTM with intelligent context
        self.trade_manager = None
        if use_dynamic_management and ohlc_data:
            try:
                # ✅ Extract hybrid context from signal_details (handle None case)
                if signal_details is None:
                    signal_details = {}

                hybrid_mode = signal_details.get("aggregator_mode")
                mode_confidence = signal_details.get("mode_confidence", 0.5)
                regime_analysis = signal_details.get("regime_analysis", {})

                # Get regime details
                trend_strength = regime_analysis.get("trend_strength", "weak")
                volatility_regime = regime_analysis.get("volatility_regime", "normal")
                price_clarity = regime_analysis.get("price_clarity", "mixed")
                momentum_aligned = regime_analysis.get("momentum_aligned", False)
                at_key_level = regime_analysis.get("at_key_level", False)

                logger.info(f"\n[VTM+HYBRID] Initializing with intelligent context:")
                logger.info(f"  Mode:        {hybrid_mode or 'N/A'}")
                logger.info(f"  Confidence:  {mode_confidence:.2%}")
                logger.info(f"  Trend:       {trend_strength}")
                logger.info(f"  Volatility:  {volatility_regime}")
                logger.info(f"  Clarity:     {price_clarity}")

                # --------------------------------------------------------
                # Adjust VTM parameters based on hybrid intelligence
                # --------------------------------------------------------
                account_risk = 0.015  # Base 1.5%
                early_lock_threshold_pct = 0.01  # Base 1%

                # Council + strong trend = More aggressive
                if hybrid_mode == "council" and trend_strength == "strong":
                    account_risk *= 1.2  # 1.5% → 1.8%
                    early_lock_threshold_pct = 0.008  # Lock @ 0.8%
                    logger.info("  → Council strong trend: Risk↑ Lock↓")

                # Performance + choppy = More conservative
                elif hybrid_mode == "performance" and volatility_regime == "high":
                    account_risk *= 0.8  # 1.5% → 1.2%
                    early_lock_threshold_pct = 0.007  # Lock @ 0.7%
                    logger.info("  → Performance choppy: Risk↓ Lock↓")

                # High confidence boost
                if mode_confidence > 0.75 and momentum_aligned:
                    account_risk *= 1.1
                    logger.info(f"  → High confidence ({mode_confidence:.0%}): Risk↑")

                # Noisy price = Reduce risk
                if price_clarity == "noisy":
                    account_risk *= 0.85
                    logger.info("  → Noisy price: Risk↓")

                # Enforce bounds
                account_risk = max(0.008, min(account_risk, 0.025))

                # ✅ Initialize VTM with optimized parameters
                # Use Keyword arguments to ensure correct mapping even if VTM signature changes

                # Apply VTM overrides if they exist
                if vtm_overrides:
                    risk_config = risk_config.copy()
                    risk_config.update(vtm_overrides)
                    logger.info(f"[VTM] Overrides applied: {vtm_overrides}")

                self.trade_manager = VeteranTradeManager(
                    entry_price=entry_price,
                    side=side,
                    asset=asset,
                    risk_config=risk_config,
                    high=ohlc_data["high"],
                    low=ohlc_data["low"],
                    close=ohlc_data["close"],
                    volume=ohlc_data.get("volume"),
                    quantity=quantity,
                    account_risk=account_risk,
                    signal_details=signal_details,
                    trade_type=signal_details.get("trade_type", "TREND"),
                )

                # ✅ Sync VTM's calculated levels back to the Position object
                if self.trade_manager:
                    self.stop_loss = self.trade_manager.initial_stop_loss
                    if self.trade_manager.take_profit_levels:
                        self.take_profit = self.trade_manager.take_profit_levels[0]

                    logger.info(f"\n[VTM] ✓ Initialized with hybrid-optimized parameters")
                    logger.info(f"  Account Risk: {account_risk:.3f}")
                    logger.info(f"  Early Lock:   {early_lock_threshold_pct:.2%}")
                    logger.info(f"  Stop Loss:    ${self.stop_loss:,.2f}")
                    logger.info(f"  Take Profit:  {f'${self.take_profit:,.2f}' if self.take_profit is not None else 'N/A'}")

            except Exception as e:
                # Catch failures (including "Position size too large") so the object still initializes
                logger.error(f"[PORTFOLIO] VTM initialization failed for {asset}: {e}")
                self.trade_manager = None

                # ✅ Fallback to provided basic levels if VTM fails to init
                self.stop_loss = stop_loss if stop_loss else None
                self.take_profit = take_profit if take_profit else None
                self.trailing_stop_pct = trailing_stop_pct if trailing_stop_pct else None

        else:
            # ✅ No VTM requested or missing data - use passed levels
            logger.debug(f"[PORTFOLIO] VTM not initialized (missing data or disabled)")
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trailing_stop_pct = trailing_stop_pct

    def update_with_new_bar(self, high: float, low: float, close: float):
        """
        ✅ CORRECTED: Update position with new OHLC bar
        Calls VTM's update method and handles exit signals properly
        """
        if self.trade_manager:
            try:
                old_stop = self.current_stop_loss
                # ✅ Call VTM's update method (returns Dict or None)
                exit_info = self.trade_manager.update_with_new_bar(
                    new_high=high, new_low=low, new_close=close
                )

                # ✨ NEW: Log VTM events
                if self.db_manager and self.db_trade_id:
                    # Log stop updates
                    if self.current_stop_loss != old_stop:
                        self.db_manager.update_trade_vtm_event(
                            trade_id=self.db_trade_id,
                            event_type="stop_updated",
                            old_value=old_stop,
                            new_value=self.current_stop_loss,
                            current_price=close,
                        )

                if exit_info:

                    # ✅ Extract exit reason from ExitReason enum
                    reason = exit_info["reason"]
                    self.db_manager.update_trade_vtm_event(
                        trade_id=self.db_trade_id,
                        event_type=exit_info["reason"].value,
                        current_price=exit_info["price"],
                        metadata={"size": exit_info["size"]},
                    )

                    # Convert enum to string for compatibility
                    if isinstance(reason, reason):
                        exit_signal = reason.value
                    else:
                        exit_signal = str(reason)

                    logger.info(
                        f"[VTM] {self.asset} exit triggered: {exit_signal} "
                        f"@ ${exit_info['price']:,.2f}"
                    )
                    return exit_signal

                # ✅ Update position's SL/TP with VTM's current levels
                # (VTM may trail stops or move to break-even)
                self.stop_loss = self.trade_manager.current_stop_loss

                return None

            except Exception as e:
                logger.error(f"[VTM] Error updating {self.asset}: {e}", exc_info=True)
                return None

        return None

    def update_with_current_price(self, current_price: float):
        """
        ✅ NEW: Real-time intra-bar update (for trailing stops)
        Call this more frequently than bar updates
        """
        if self.trade_manager:
            try:
                exit_info = self.trade_manager.update_with_current_price(current_price)

                if exit_info:
                    reason = exit_info["reason"]
                    exit_signal = (
                        reason.value if isinstance(reason, reason) else str(reason)
                    )

                    logger.info(
                        f"[VTM] {self.asset} real-time exit: {exit_signal} "
                        f"@ ${current_price:,.2f}"
                    )
                    return exit_signal

                # Update position's stop loss (may have trailed)
                self.stop_loss = self.trade_manager.current_stop_loss

                return None

            except Exception as e:
                logger.error(f"[VTM] Real-time update error: {e}")
                return None

        return None

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """
        ✅ CORRECTED: Check if position should close
        Prioritizes VTM exit signals over traditional SL/TP
        """
        # 1. Check VTM first (if active)
        if self.trade_manager:
            exit_info = self.trade_manager.check_exit(current_price)
            if exit_info:
                reason = exit_info["reason"]
                exit_signal = (
                    reason.value if isinstance(reason, reason) else str(reason)
                )
                return True, f"vtm_{exit_signal}"

        # 2. Fallback to traditional SL/TP (if no VTM)
        if self.stop_loss:
            if self.side == "long" and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == "short" and current_price >= self.stop_loss:
                return True, "stop_loss"

        if self.take_profit:
            if self.side == "long" and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == "short" and current_price <= self.take_profit:
                return True, "take_profit"

        # 3. Traditional trailing stop (only if no VTM)
        if not self.trade_manager:
            trail_stop = self.update_trailing_stop(current_price)
            if trail_stop:
                if self.side == "long" and current_price <= trail_stop:
                    return True, "trailing_stop"
                elif self.side == "short" and current_price >= trail_stop:
                    return True, "trailing_stop"

        return False, ""

    def get_vtm_status(self, live_price: Optional[float] = None) -> Optional[Dict]:
        """Get current VTM status for monitoring"""
        if not self.trade_manager:
            return None

        try:
            # VTM calculates the current price if not provided.
            levels = self.trade_manager.get_current_levels(live_price=live_price)
            if not levels:
                return None

            # ✅ FIX: Get the current_price that VTM calculated or used.
            current_price = levels["current_price"]
            
            next_target = levels.get("take_profit")

            # Calculate absolute P&L
            pnl_abs = (current_price - self.entry_price) * self.quantity if self.side == "long" else \
                      (self.entry_price - current_price) * self.quantity

            return {
                "side": self.side,
                "entry_price": levels["entry_price"],
                "current_price": levels["current_price"],
                "pnl_pct": levels["pnl_pct"],
                "pnl_abs": pnl_abs, # NEW: Absolute P&L
                "stop_loss": levels["stop_loss"],
                "take_profit": (
                    next_target
                    if next_target
                    else levels.get("all_targets", [])[-1] if levels.get("all_targets") else None
                ),
                "distance_to_sl_pct": levels["distance_to_sl_pct"],
                "distance_to_tp_pct": levels["distance_to_tp_pct"],
                "profit_locked": levels["profit_locked"],
                "bars_in_trade": levels["update_count"],
                "partials_hit": levels["partials_hit"],
                "runner_active": levels["runner_active"],
                "update_count": levels["update_count"],
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            logger.error(f"Error getting VTM status: {e}")
            return None

    def get_position_value(self, current_price: float) -> float:
        """Get current position value in USD"""
        return self.quantity * current_price

    def get_pnl(self, current_price: float) -> float:
        """Get current profit/loss"""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_mt5_pnl(self) -> float:
        """Get real-time P&L from MT5 position"""
        return self.mt5_profit

    def get_binance_pnl(self) -> float:
        """Get real-time P&L from Binance position"""
        return self.binance_profit

    def get_exchange_pnl(self) -> float:
        """Get real-time P&L from either exchange"""
        if self.mt5_ticket and self.mt5_profit != 0.0:
            return self.mt5_profit
        elif self.binance_order_id and self.binance_profit != 0.0:
            return self.binance_profit
        return 0.0

    def get_pnl_pct(self, current_price: float) -> float:
        """Get current profit/loss percentage"""
        position_value = self.entry_price * self.quantity
        return self.get_pnl(current_price) / position_value if position_value > 0 else 0

    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop (only used if VTM is disabled)"""
        if self.trailing_stop_pct is None:
            return None

        if self.side == "long":
            if current_price > self.highest_price:
                self.highest_price = current_price
            return self.highest_price * (1 - self.trailing_stop_pct)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            return self.lowest_price * (1 + self.trailing_stop_pct)

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed based on stop loss/take profit"""
        if self.stop_loss:
            if self.side == "long" and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == "short" and current_price >= self.stop_loss:
                return True, "stop_loss"

        if self.take_profit:
            if self.side == "long" and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == "short" and current_price <= self.take_profit:
                return True, "take_profit"

        trail_stop = self.update_trailing_stop(current_price)
        if trail_stop:
            if self.side == "long" and current_price <= trail_stop:
                return True, "trailing_stop"
            elif self.side == "short" and current_price >= trail_stop:
                return True, "trailing_stop"

        return False, ""

    def __getstate__(self):
        """
        Custom method for pickling. Excludes non-serializable attributes.
        """
        state = self.__dict__.copy()
        # Remove the unpickleable db_manager attribute
        if 'db_manager' in state:
            del state['db_manager']
        return state

    def __setstate__(self, state):
        """
        Custom method for unpickling. Restores state and re-initializes
        non-serializable attributes.
        """
        self.__dict__.update(state)
        # Re-initialize the db_manager attribute after unpickling.
        # It will need to be re-assigned by the PortfolioManager after loading.
        self.db_manager = None



class PortfolioManager:
    """
    Manages portfolio-level risk and position sizing
    Fetches actual capital from exchanges (MT5 and Binance)
    Tracks real-time MT5 profit for accurate P&L
    """

    def __init__(
        self,
        config: Dict,
        mt5_handler=None,
        binance_client=None,
        db_manager=None,
        execution_handlers: Dict = None,
        telegram_bot=None,
    ):
        self.config = config
        self.telegram_bot = telegram_bot
        self.portfolio_config = config["portfolio"]
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )

        self.mt5_handler = mt5_handler
        self.binance_client = binance_client
        self.db_manager = db_manager
        self.execution_handlers = execution_handlers or {}
        self.risk_cfg = config.get("risk_management", {})
        
        self.correlation_threshold = self.portfolio_config.get("correlation_threshold", 0.70)
        logger.info(f"[RISK] Correlation threshold: {self.correlation_threshold:.0%}")

        self.mode = config["trading"].get("mode", "paper")
        self.is_paper_mode = self.mode.lower() == "paper"

        # ✅ FIX 1: Remove paper_capital usage in live mode
        self.paper_capital = self.portfolio_config["initial_capital"]

        # ✅ FIX 2: Initialize with live balances (will raise error if unavailable in live mode)
        self.initial_capital = self._fetch_total_capital(strict=True)
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        # ✅ FIX 3: Track last balance refresh time
        self.last_balance_refresh = datetime.now()
        self.balance_refresh_interval = timedelta(minutes=5)  # Refresh every 5 min

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.price_history: Dict[str, List[float]] = {"BTC": [], "GOLD": []}
        self.realized_pnl_today = 0.0

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None
        self.state_file = Path("data/portfolio_state.pkl")

        logger.info(f"Portfolio Manager initialized in {self.mode.upper()} mode")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")

        if not self.is_paper_mode:
            logger.info("✓ Using LIVE account balances (will auto-refresh)")
            logger.info(
                f"  - MT5 Handler: {'Connected' if mt5_handler else 'Not Connected'}"
            )
            logger.info(
                f"  - Binance Client: {'Connected' if binance_client else 'Not Connected'}"
            )
        else:
            logger.info("✓ Using PAPER mode with simulated capital")

    def save_portfolio_state(self):
        """Saves the current open positions to a file atomically."""
        if self.is_paper_mode:
            logger.info("[STATE] Paper mode, skipping state save.")
            return

        # If there are no positions, ensure no state file is left
        if not self.positions:
            if self.state_file.exists():
                try:
                    self.state_file.unlink()
                    logger.info("[STATE] No open positions. Removed stale state file.")
                except Exception as e:
                    logger.error(f"[STATE] Error removing stale state file: {e}")
            return

        temp_file_path = self.state_file.with_suffix('.pkl.tmp')
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(temp_file_path, "wb") as f:
                pickle.dump(self.positions, f)
            
            # Atomically rename the temp file to the final file
            temp_file_path.rename(self.state_file)
            logger.info(f"[STATE] Successfully saved {len(self.positions)} open positions to {self.state_file}")

        except Exception as e:
            logger.error(f"[STATE] Failed to save portfolio state: {e}", exc_info=True)
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as e_del:
                    logger.error(f"[STATE] Failed to clean up temp file {temp_file_path}: {e_del}")

    def load_portfolio_state(self, data_manager):
        """Loads open positions from a file and re-initializes them."""
        if self.is_paper_mode:
            logger.info("[STATE] Paper mode, skipping state load.")
            return

        if not self.state_file.exists():
            logger.info("[STATE] No portfolio state file found. Starting fresh.")
            return
            
        # Check for empty file to prevent EOFError
        if self.state_file.stat().st_size == 0:
            logger.warning(f"[STATE] State file {self.state_file} is empty. Deleting and starting fresh.")
            self.state_file.unlink()
            return

        try:
            with open(self.state_file, "rb") as f:
                loaded_positions = pickle.load(f)

            if not loaded_positions:
                logger.info("[STATE] Portfolio state file is empty.")
                return

            logger.info(f"[STATE] Found {len(loaded_positions)} positions in state file. Reloading...")
            self.positions = {} # Clear current positions before loading

            for position_id, position in loaded_positions.items():
                logger.info(f"[STATE] Reloading position: {position_id} ({position.asset} {position.side})")
                
                # Critical step: Re-initialize the VTM with fresh OHLC data
                if position.trade_manager:
                    logger.info("[STATE] VTM found. Fetching fresh OHLC data to re-initialize...")
                    try:
                        asset_config = self.config['assets'][position.asset]
                        symbol = asset_config['symbol']
                        interval = asset_config.get('interval', '1h')
                        exchange = asset_config.get('exchange', 'binance')
                        
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10) # Fetch enough data

                        if exchange == 'binance':
                            df = data_manager.fetch_binance_data(
                                symbol=symbol, interval=interval,
                                start_date=start_time.strftime("%Y-%m-%d"),
                                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        else: # mt5
                            df = data_manager.fetch_mt5_data(
                                symbol=symbol, timeframe=interval,
                                start_date=start_time.strftime("%Y-%m-%d"),
                                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        
                        if len(df) > 50:
                            position.trade_manager.high = df['high'].values
                            position.trade_manager.low = df['low'].values
                            position.trade_manager.close = df['close'].values
                            position.trade_manager.volume = df['volume'].values if 'volume' in df else None
                            logger.info(f"[STATE] VTM for {position_id} successfully re-initialized with {len(df)} candles.")
                        else:
                            logger.warning(f"[STATE] Could not fetch enough OHLC data for {position_id}. VTM may be impaired.")
                            position.trade_manager = None # Disable VTM if data is bad

                    except Exception as e:
                        logger.error(f"[STATE] Failed to re-initialize VTM for {position_id}: {e}", exc_info=True)
                        position.trade_manager = None # Disable VTM on failure
                
                # Re-link the db_manager
                if self.db_manager:
                    position.db_manager = self.db_manager

                self.positions[position_id] = position

            logger.info(f"[STATE] Successfully loaded and re-initialized {len(self.positions)} positions.")
            
            # Clean up state file after successful load
            self.state_file.unlink()
            logger.info(f"[STATE] Removed state file {self.state_file} after successful load.")

        except Exception as e:
            logger.error(f"[STATE] Failed to load portfolio state: {e}", exc_info=True)
            # If loading fails, start with a clean slate to avoid corruption
            self.positions = {}
            
    def get_risk_budget(
        self, 
        asset: str, 
        strategy_type: str = "TREND"
    ) -> float:
        """
        ✨ STRATEGIC RISK GOVERNOR
        
        Calculates risk budget for a new trade based on:
        1. Base risk (from config)
        2. Strategy type (SCALP vs TREND asymmetric adjustment)
        3. Correlation malus (reduce risk if holding correlated assets)
        4. Drawdown shield (reduce risk in drawdown)
        5. Total risk limit (cap aggregate open risk)
        
        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            strategy_type: "TREND" or "SCALP"
            
        Returns:
            Risk percentage (e.g., 0.015 for 1.5%)
        """
        try:
            asset_cfg = self.config.get("assets", {}).get(asset, {})
            # ================================================================
            # STEP 1: Get base risk from config (Percentage or Fixed Dollar)
            # ================================================================
            fixed_risk_config = asset_cfg.get("fixed_risk_usd")
            base_risk = self.portfolio_config.get("target_risk_per_trade", 0.015)
            strategy_multiplier = 1.0
            
            logger.info(f"\n[RISK BUDGET] Calculating for {asset} {strategy_type}")

            if fixed_risk_config and isinstance(fixed_risk_config, dict):
                # FIXED DOLLAR RISK LOGIC
                risk_usd = fixed_risk_config.get(strategy_type)
                if risk_usd:
                    if self.current_capital > 0:
                        risk_pct = risk_usd / self.current_capital
                        logger.info(f"  Fixed Dollar Risk: ${risk_usd} ({strategy_type})")
                        logger.info(f"  Account Capital: ${self.current_capital:,.2f}")
                        logger.info(f"  Calculated Risk: {risk_pct:.3%}")
                    else:
                        logger.error("[RISK] Cannot calculate fixed risk, current capital is zero.")
                        return 0.0
                else:
                    # Fallback to percentage if strategy type not in fixed config
                    risk_pct = base_risk
                    logger.info(f"  Base Risk: {risk_pct:.3%}")
            else:
                # ORIGINAL PERCENTAGE-BASED LOGIC
                logger.info(f"  Base Risk: {base_risk:.3%}")
            
                # ================================================================
                # STEP 2: Strategy Type Adjustment (Asymmetric)
                # ================================================================
                if strategy_type == "SCALP":
                    # Scalps: Lower risk (quick in/out)
                    strategy_multiplier = 1.25  # 1.5% → 1.875%
                    logger.info(f"  SCALP Multiplier: {strategy_multiplier:.2f}x")
                elif strategy_type == "TREND":
                    # Trends: Higher risk (riding momentum)
                    strategy_multiplier = 1.33  # 1.5% → 2.0%
                    logger.info(f"  TREND Multiplier: {strategy_multiplier:.2f}x")
                else:
                    strategy_multiplier = 1.0
                    logger.info(f"  Default Multiplier: {strategy_multiplier:.2f}x")
                
                risk_pct = base_risk * strategy_multiplier
            
            # ================================================================
            # STEP 3: Correlation Malus
            # ================================================================
            correlation_threshold = self.portfolio_config.get(
                "correlation_threshold", 0.70
            )
            correlation_malus = 1.0
            
            # Check if we hold correlated positions
            if len(self.positions) > 0:
                # Simple correlation check (BTC-ETH, Gold-Silver, etc.)
                # For now, we implement a basic version
                # TODO: Enhance with actual correlation matrix
                
                asset_group = self._get_asset_group(asset)
                
                for pos_asset, position in self.positions.items():
                    other_group = self._get_asset_group(pos_asset)
                    
                    # If same group, apply malus
                    if asset_group == other_group and asset_group != "other":
                        correlation_malus = 0.5
                        logger.warning(
                            f"  ⚠️ Correlation Malus: Holding {pos_asset} "
                            f"(same group: {asset_group})"
                        )
                        logger.info(f"  Risk reduced by {1 - correlation_malus:.0%}")
                        break
            
            risk_pct *= correlation_malus
            
            # ================================================================
            # STEP 4: Drawdown Shield
            # ================================================================
            drawdown_threshold = self.portfolio_config.get("max_drawdown", 0.15)
            current_drawdown = 0.0
            
            if self.peak_equity > 0:
                current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
            
            drawdown_malus = 1.0
            
            if current_drawdown > 0.05:  # 5% drawdown trigger
                drawdown_malus = 0.5
                logger.warning(
                    f"  ⚠️ Drawdown Shield: {current_drawdown:.2%} drawdown detected"
                )
                logger.info(f"  Risk reduced by {1 - drawdown_malus:.0%}")
            
            risk_pct *= drawdown_malus
            
            # ================================================================
            # STEP 5: Total Risk Limit (Aggregate Cap)
            # ================================================================
            max_total_risk_pct = self.risk_cfg.get("max_total_open_risk", 0.10)
            
            # Calculate current total risk
            current_total_risk = 0.0
            for position in self.positions.values():
                if position.stop_loss:
                    pos_risk = abs(position.entry_price - position.stop_loss) * position.quantity
                    current_total_risk += pos_risk
            
            current_total_risk_pct = (
                current_total_risk / self.current_capital 
                if self.current_capital > 0 
                else 0
            )
            
            # Check if adding new trade would exceed limit
            remaining_risk_budget = max_total_risk_pct - current_total_risk_pct
            
            if risk_pct > remaining_risk_budget:
                logger.warning(
                    f"  ⚠️ Total Risk Limit: Current {current_total_risk_pct:.2%}, "
                    f"Max {max_total_risk_pct:.2%}"
                )
                logger.info(
                    f"  Risk capped from {risk_pct:.3%} to {remaining_risk_budget:.3%}"
                )
                risk_pct = max(0, remaining_risk_budget)
            
            # ================================================================
            # STEP 6: Final Validation
            # ================================================================
            # Ensure we don't go below minimum viable risk
            min_risk = 0.001  # 0.1% absolute minimum
            if risk_pct < min_risk:
                logger.error(
                    f"  ❌ Risk budget {risk_pct:.3%} below minimum {min_risk:.3%}"
                )
                logger.error(f"  → Trade should be rejected")
                return 0.0
            
            # Ensure we don't exceed maximum risk
            max_risk = self.portfolio_config.get("max_risk_per_trade", 0.025)
            if risk_pct > max_risk:
                logger.warning(f"  ⚠️ Risk capped at maximum {max_risk:.3%}")
                risk_pct = max_risk
            
            # ================================================================
            # STEP 7: Log Final Budget
            # ================================================================
            logger.info(f"\n[RISK BUDGET] FINAL: {risk_pct:.3%}")
            logger.info(f"  Breakdown:")
            logger.info(f"    Base:         {base_risk:.3%}")
            logger.info(f"    Strategy:     ×{strategy_multiplier:.2f}")
            logger.info(f"    Correlation:  ×{correlation_malus:.2f}")
            logger.info(f"    Drawdown:     ×{drawdown_malus:.2f}")
            logger.info(f"    Final:        {risk_pct:.3%}")
            logger.info(f"  → ${self.current_capital * risk_pct:,.2f} at risk\n")
            
            return risk_pct
            
        except Exception as e:
            logger.error(f"[RISK BUDGET] Error calculating risk: {e}", exc_info=True)
            # Return safe default
            return 0.01  # 1% fallback

    def _get_asset_group(self, asset: str) -> str:
        """
        Helper: Categorize assets into correlation groups
        
        Returns:
            Group name: "crypto", "precious_metals", "indices", "forex", "other"
        """
        asset = asset.upper()
        
        # Crypto group
        if asset in ["BTC", "BITCOIN", "BTCUSD", "BTCUSDT", "ETH", "ETHEREUM"]:
            return "crypto"
        
        # Precious metals group
        if asset in ["GOLD", "XAU", "XAUUSDm", "SILVER", "XAG", "XAGUSD"]:
            return "precious_metals"
        
        # Indices group
        if asset in ["SPX", "SPY", "QQQ", "NASDAQ", "DOW"]:
            return "indices"
        
        # Forex group
        if asset in ["EUR", "EURUSD", "GBP", "GBPUSD", "JPY", "USDJPY"]:
            return "forex"
        
        return "other"

    def _fetch_total_capital(self, strict: bool = False) -> float:
        """
        ✅ FIXED: Fetch total available capital from ALL exchanges

        Args:
            strict: If True, raise error when live balances unavailable in live mode

        Returns:
            Total capital in USD (MT5 + Binance combined)
        """
        if self.is_paper_mode:
            logger.info(f"[PAPER] Using simulated capital: ${self.paper_capital:,.2f}")
            return self.paper_capital

        total_capital = 0.0
        errors = []
        balances_found = []

        # ================================================================
        # ✅ FIX 1: Check MT5 balance (GOLD) - Always try if handler exists
        # ================================================================
        if self.mt5_handler is not None:
            try:
                mt5_balance = self._fetch_mt5_balance()
                if mt5_balance is not None and mt5_balance > 0:
                    total_capital += mt5_balance
                    balances_found.append(f"MT5: ${mt5_balance:,.2f}")
                    logger.info(f"[MT5] ✓ Balance fetched: ${mt5_balance:,.2f}")
                else:
                    logger.warning(f"[MT5] Balance is 0 or None")
                    errors.append("MT5 balance unavailable or 0")
            except Exception as e:
                logger.error(f"[MT5] Error fetching balance: {e}", exc_info=True)
                errors.append(f"MT5 error: {str(e)}")
        else:
            logger.debug("[MT5] Handler not available, skipping MT5 balance")

        # ================================================================
        # ✅ FIX 2: Check Binance balance (BTC) - Always try if client exists
        # ================================================================
        if self.binance_client is not None:
            try:
                binance_balance = self._fetch_binance_balance()
                if binance_balance is not None and binance_balance > 0:
                    total_capital += binance_balance
                    balances_found.append(f"Binance: ${binance_balance:,.2f}")
                    logger.info(f"[BINANCE] ✓ Balance fetched: ${binance_balance:,.2f}")
                else:
                    logger.warning(f"[BINANCE] Balance is 0 or None")
                    errors.append("Binance balance unavailable or 0")
            except Exception as e:
                logger.error(f"[BINANCE] Error fetching balance: {e}", exc_info=True)
                errors.append(f"Binance error: {str(e)}")
        else:
            logger.debug("[BINANCE] Client not available, skipping Binance balance")

        # ================================================================
        # ✅ FIX 3: Log combined results clearly
        # ================================================================
        logger.info(
            f"\n{'='*80}\n"
            f"[BALANCE SUMMARY]\n"
            f"{'='*80}\n"
            f"Balances Found: {len(balances_found)}\n"
            f"  {chr(10).join(balances_found) if balances_found else 'None'}\n"
            f"Total Capital:  ${total_capital:,.2f}\n"
            f"Errors: {len(errors)}\n"
            f"  {chr(10).join(errors) if errors else 'None'}\n"
            f"{'='*80}"
        )

        # ================================================================
        # ✅ FIX 4: Strict mode enforcement (for live trading)
        # ================================================================
        if strict and not self.is_paper_mode and total_capital == 0:
            if self.mt5_handler is None and self.binance_client is None:
                # This is okay, handlers aren't ready yet, will refresh later
                logger.warning("[BALANCE] No handlers initialized yet, initial capital set to 0. Will refresh later.")
                return 0.0
            else:
                error_msg = (
                    f"CRITICAL: Unable to fetch live account balances!\n"
                    f"Errors: {', '.join(errors)}\n"
                    f"Cannot proceed with live trading without valid balances."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # ================================================================
        # ✅ FIX 5: Fallback handling
        # ================================================================
        if total_capital == 0:
            if self.is_paper_mode:
                logger.warning("No balances fetched, using paper capital")
                return self.paper_capital
            else:
                logger.error(
                    "⚠️  NO BALANCES AVAILABLE FROM ANY EXCHANGE!\n"
                    "Check your MT5 connection and Binance API keys."
                )
                return 0.0

        return total_capital

    def _fetch_mt5_balance(self) -> Optional[float]:
        """
        ✅ FIXED: Fetch MT5 balance with better error handling
        """
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                logger.error("[MT5] Failed to initialize terminal")
                return None

            account_info = mt5.account_info()

            if account_info:
                balance = account_info.balance
                equity = account_info.equity
                margin = account_info.margin
                free_margin = account_info.margin_free

                logger.info(
                    f"[MT5] Account info:\n"
                    f"  Balance:      ${balance:,.2f}\n"
                    f"  Equity:       ${equity:,.2f}\n"
                    f"  Margin Used:  ${margin:,.2f}\n"
                    f"  Free Margin:  ${free_margin:,.2f}"
                )

                # Use equity (includes unrealized P&L)
                return equity
            else:
                logger.error("[MT5] No account info available")
                return None

        except Exception as e:
            logger.error(f"[MT5] Error fetching balance: {e}", exc_info=True)
            return None

    def _fetch_binance_balance(self) -> Optional[float]:
        """
        ✅ FIXED: Dynamically fetches Futures balance if enabled, otherwise Spot.
        """
        try:
            if not self.binance_client:
                logger.error("[BINANCE] Client not initialized")
                return None

            # Check if we are trading Futures or Spot
            is_futures = (
                self.config.get("assets", {})
                .get("BTC", {})
                .get("enable_futures", False)
            )

            if is_futures:
                logger.debug("[BINANCE] Fetching FUTURES account info...")
                account = self.binance_client.futures_account()

                total_balance = float(account.get("totalWalletBalance", 0))
                available = float(account.get("availableBalance", 0))
                unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))

                logger.info(
                    f"[BINANCE FUTURES] Balance breakdown:\n"
                    f"  USDT Total: ${total_balance:,.2f}\n"
                    f"  USDT Free:  ${available:,.2f}\n"
                    f"  Unrealized: ${unrealized_pnl:,.2f}\n"
                    f"  Total: ${total_balance:,.2f}"
                )
                return total_balance if total_balance > 0 else None

            else:
                # SPOT WALLET LOGIC (Original)
                logger.debug("[BINANCE] Fetching SPOT account info...")
                account = self.binance_client.get_account()

                total_balance = 0.0
                asset_details = []

                for balance in account["balances"]:
                    asset = balance["asset"]
                    free = float(balance["free"])
                    locked = float(balance["locked"])
                    total = free + locked

                    if total > 0.0001:
                        if asset == "USDT":
                            total_balance += total
                            asset_details.append(
                                f"  USDT: ${total:,.2f} (free: ${free:,.2f}, locked: ${locked:,.2f})"
                            )

                        elif asset == "BTC":
                            handler = self.execution_handlers.get("binance")
                            if not handler:
                                logger.error("[BINANCE] Cannot get BTC price, handler not available.")
                                continue
                            btc_price = handler.get_current_price("BTCUSDT")
                            if not btc_price:
                                logger.error("[BINANCE] Failed to get BTC price from handler.")
                                continue
                            
                            usd_value = total * btc_price
                            total_balance += usd_value
                            asset_details.append(
                                f"  BTC:  {total:.8f} @ ${btc_price:,.2f} = ${usd_value:,.2f}"
                            )

                if asset_details:
                    logger.info(
                        f"[BINANCE SPOT] Balance breakdown:\n"
                        + "\n".join(asset_details)
                        + f"\n  Total: ${total_balance:,.2f}"
                    )

                return total_balance if total_balance > 0 else None

        except BinanceAPIException as e:
            logger.error(f"[BINANCE] API error: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching balance: {e}", exc_info=True)
            return None

    def refresh_capital(self, force: bool = False) -> bool:
        """
        ✅ FIXED: Better logging for balance refresh
        """
        if self.is_paper_mode:
            return True

        # Check if refresh is needed
        now = datetime.now()
        time_since_refresh = now - self.last_balance_refresh

        if not force and time_since_refresh < self.balance_refresh_interval:
            logger.debug(
                f"[BALANCE] Skipping refresh (last: {time_since_refresh.seconds}s ago, "
                f"interval: {self.balance_refresh_interval.seconds}s)"
            )
            return True

        logger.info(
            f"\n{'='*80}\n"
            f"[BALANCE REFRESH]\n"
            f"{'='*80}\n"
            f"Last refresh: {time_since_refresh.seconds}s ago\n"
            f"Force:        {force}\n"
            f"{'='*80}"
        )

        # Fetch new balances
        new_capital = self._fetch_total_capital(strict=False)

        if new_capital > 0:
            old_capital = self.current_capital
            self.current_capital = new_capital
            self.equity = new_capital
            self.last_balance_refresh = now

            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            change = new_capital - old_capital
            change_pct = (change / old_capital * 100) if old_capital > 0 else 0

            logger.info(
                f"[BALANCE] ✓ Refreshed successfully\n"
                f"  Old: ${old_capital:,.2f}\n"
                f"  New: ${new_capital:,.2f}\n"
                f"  Δ:   ${change:+,.2f} ({change_pct:+.2f}%)"
            )
            return True
        else:
            logger.error(
                f"[BALANCE] ✗ Failed to refresh balances!\n"
                f"  Check MT5 connection and Binance API"
            )
            return False

    def update_mt5_positions_profit(self):
        """
        Update all MT5 positions with real-time profit from MT5
        Call this periodically to sync P&L
        """
        try:
            import MetaTrader5 as mt5

            # Get all open MT5 positions
            mt5_positions = mt5.positions_get()

            if mt5_positions is None or len(mt5_positions) == 0:
                return

            # Update profit for each tracked position
            for asset, position in self.positions.items():
                if position.mt5_ticket is None:
                    continue  # Skip non-MT5 positions (e.g., Binance)

                # Find matching MT5 position
                for mt5_pos in mt5_positions:
                    if mt5_pos.ticket == position.mt5_ticket:
                        position.mt5_profit = mt5_pos.profit
                        position.mt5_last_update = datetime.now()

                        logger.debug(
                            f"[MT5] Updated {asset} profit: ${mt5_pos.profit:,.2f}"
                        )
                        break

        except Exception as e:
            logger.error(f"Error updating MT5 positions profit: {e}", exc_info=True)

    def update_binance_positions_profit(self):
        """
        Update all Binance positions with real-time profit
        Call this periodically to sync P&L
        """
        try:
            if self.binance_client is None:
                return

            # Get current prices for all Binance positions
            for asset, position in self.positions.items():
                if position.binance_order_id is None:
                    continue  # Skip non-Binance positions (e.g., MT5)

                # Get current price
                try:
                    handler = self.execution_handlers.get("binance")
                    if not handler:
                        logger.debug(f"Cannot update Binance profit for {asset}, handler not available.")
                        continue
                    
                    current_price = handler.get_current_price(position.symbol)
                    if not current_price:
                        logger.debug(f"Could not fetch price for {position.symbol} via handler.")
                        continue

                    # Calculate real-time P&L
                    if position.side == "long":
                        position.binance_profit = (
                            current_price - position.entry_price
                        ) * position.quantity
                    else:
                        position.binance_profit = (
                            position.entry_price - current_price
                        ) * position.quantity

                    position.binance_last_update = datetime.now()

                    logger.debug(
                        f"[BINANCE] Updated {asset} profit: ${position.binance_profit:,.2f}"
                    )

                except Exception as e:
                    logger.debug(f"Error fetching Binance price for {asset}: {e}")

        except Exception as e:
            logger.error(f"Error updating Binance positions profit: {e}", exc_info=True)

    def calculate_position_size(
        self, asset: str, current_price: float, volatility: float = None
    ) -> float:
        """
        HARDENED: Calculate position size based STRICTLY on the single exchange's RAW equity.
        Ignores leverage buying power.
        """
        # Get RAW capital for THIS asset's specific exchange
        exchange = self.config["assets"][asset]["exchange"].lower()
        if exchange == "binance":
            # Force Binance to return actual cash, not leveraged buying power
            asset_capital = self.get_binance_balance()
        else:
            asset_capital = self._fetch_mt5_balance()

        if asset_capital <= 0:
            logger.error(
                f"Cannot calculate position size for {asset}: exchange capital is 0!"
            )
            return 0.0

        # Base position size as percentage of the RAW CASH
        base_size_pct = self.portfolio_config["base_position_size"]
        base_size_usd = asset_capital * base_size_pct

        # Apply asset weight
        asset_weight = self.config["assets"][asset].get("weight", 1.0)
        position_size = base_size_usd * asset_weight

        # Hard Cap Check: Never allow a single trade to exceed 50% of the account cash
        absolute_max = asset_capital * 0.50
        position_size = min(position_size, absolute_max)

        logger.info(
            f"{asset} isolated position size: ${position_size:,.2f} "
            f"(Based strictly on {exchange.upper()} balance: ${asset_capital:,.2f})"
        )
        return position_size

    def validate_balance_before_trade(self) -> Tuple[bool, str]:
        """
        ✅ NEW: Validate that we have valid balances before opening trades

        Returns:
            (is_valid, error_message)
        """
        if self.is_paper_mode:
            return True, "OK"

        # Force refresh to get latest balances
        if not self.refresh_capital(force=True):
            return False, "Failed to fetch account balances"

        if self.current_capital <= 0:
            return False, f"Invalid capital: ${self.current_capital:,.2f}"

        # Check minimum capital requirements
        min_capital = self.portfolio_config.get("min_capital_threshold", 1000)
        if self.current_capital < min_capital:
            return (
                False,
                f"Capital below minimum: ${self.current_capital:,.2f} < ${min_capital:,.2f}",
            )

        return True, "OK"

    def check_portfolio_limits(
        self, new_position_usd: float, new_side: str = None, asset: str = None
    ) -> bool:
        """
        ✅ FIXED: Check portfolio limits using margin exposure (not notional)
        """
        # Get limits
        max_exposure_pct = self.portfolio_config["max_portfolio_exposure"]
        max_exposure_usd = self.current_capital * max_exposure_pct
        
        # ✅ FIXED: Calculate current MARGIN exposure (not notional)
        long_margin = 0.0
        short_margin = 0.0
        
        for pos in self.positions.values():
            notional = pos.quantity * pos.entry_price
            leverage = getattr(pos, 'leverage', 1)
            margin = notional / leverage  # ← Use margin
            
            if pos.side == "long":
                long_margin += margin
            else:
                short_margin += margin
        
        current_gross_margin = long_margin + short_margin
        current_net_margin = abs(long_margin - short_margin)
        
        # ✅ Check hedging configuration
        allow_hedging = self.config.get("trading", {}).get(
            "allow_simultaneous_long_short", False
        )
        
        # Calculate new position margin
        # NOTE: new_position_usd should already be the NOTIONAL value
        # We need to get leverage for this asset to calculate margin
        
        # Get leverage from config (or default to 1)
        if asset:
            asset_cfg = self.config.get("assets", {}).get(asset, {})
            leverage = asset_cfg.get("leverage", 1)
        else:
            leverage = 1
        
        new_position_margin = new_position_usd / leverage
        
        # Check if it's a hedge
        is_hedge = False
        if asset and new_side:
            opposite_side = "short" if new_side == "long" else "long"
            opposite_positions = [
                p for p in self.positions.values()
                if p.asset == asset and p.side == opposite_side
            ]
            is_hedge = len(opposite_positions) > 0
        
        # Use NET margin for hedged strategies, GROSS for directional
        if allow_hedging and (is_hedge or new_side):
            if new_side == "long":
                new_net_margin = abs((long_margin + new_position_margin) - short_margin)
            elif new_side == "short":
                new_net_margin = abs(long_margin - (short_margin + new_position_margin))
            else:
                new_net_margin = current_net_margin + new_position_margin
            
            if new_net_margin > max_exposure_usd:
                logger.warning(
                    f"Portfolio NET margin limit exceeded:\n"
                    f"  Current Long Margin:  ${long_margin:,.2f}\n"
                    f"  Current Short Margin: ${short_margin:,.2f}\n"
                    f"  Current Net Margin:   ${current_net_margin:,.2f}\n"
                    f"  New {new_side or 'position'} (margin): ${new_position_margin:,.2f}\n"
                    f"  New Net Margin:       ${new_net_margin:,.2f}\n"
                    f"  Limit:                ${max_exposure_usd:,.2f}"
                )
                return False
            
            logger.info(
                f"[EXPOSURE] NET MARGIN: ${new_net_margin:,.2f} / ${max_exposure_usd:,.2f} "
                f"({new_net_margin/max_exposure_usd*100:.1f}%)"
                f"{' [HEDGE]' if is_hedge else ''}"
            )
        
        else:
            # Use GROSS margin for directional strategies
            new_gross_margin = current_gross_margin + new_position_margin
            
            if new_gross_margin > max_exposure_usd:
                logger.warning(
                    f"Portfolio GROSS margin limit: "
                    f"${new_gross_margin:,.2f} > ${max_exposure_usd:,.2f}"
                )
                return False
            
            logger.info(
                f"[EXPOSURE] GROSS MARGIN: ${new_gross_margin:,.2f} / ${max_exposure_usd:,.2f} "
                f"({new_gross_margin/max_exposure_usd*100:.1f}%)"
            )
        
        # Check drawdown limit (unchanged)
        drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        max_drawdown = self.portfolio_config["max_drawdown"]
        
        if drawdown >= max_drawdown:
            logger.warning(f"Max drawdown: {drawdown:.2%} >= {max_drawdown:.2%}")
            return False
        
        return True

    def get_asset_positions(self, asset: str, side: str = None) -> List[Position]:
        """
        Get all positions for a specific asset

        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            side: Optional side filter ("long" or "short")

        Returns:
            List of Position objects
        """
        positions = [pos for pos in self.positions.values() if pos.asset == asset]

        if side:
            positions = [pos for pos in positions if pos.side == side]

        return positions

    def get_asset_position_count(self, asset: str, side: str = None) -> int:
        """
        Count open positions for an asset

        Args:
            asset: Asset name
            side: Optional side filter

        Returns:
            Number of open positions
        """
        return len(self.get_asset_positions(asset, side))

    def check_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return 0.0

        min_points = 30
        if (
            len(self.price_history.get(asset1, [])) < min_points
            or len(self.price_history.get(asset2, [])) < min_points
        ):
            return 0.0

        returns1 = np.diff(np.log(self.price_history[asset1][-min_points:]))
        returns2 = np.diff(np.log(self.price_history[asset2][-min_points:]))

        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def should_reduce_position(self, new_asset: str) -> bool:
        """Check if position should be reduced due to correlation"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return False

        threshold = self.portfolio_config["correlation_threshold"]

        for existing_asset in self.positions.keys():
            if existing_asset != new_asset:
                corr = self.check_correlation(new_asset, existing_asset)
                if abs(corr) > threshold:
                    logger.warning(
                        f"High correlation detected between {new_asset} and {existing_asset}: "
                        f"{corr:.2f}"
                    )
                    return True

        return False

    def can_open_position(self, asset: str, side: str) -> Tuple[bool, str]:
        """Check both long and short separately"""
        current_count = self.get_asset_position_count(asset, side)

        if current_count >= self.max_positions_per_asset:
            return False, f"Max {side} positions reached"

        # Check if opposite side exists (if simultaneous trading disabled)
        if not self.config.get("trading", {}).get(
            "allow_simultaneous_long_short", False
        ):
            opposite_side = "short" if side == "long" else "long"
            opposite_count = self.get_asset_position_count(asset, opposite_side)
            if opposite_count > 0:
                return False, f"Have opposite {opposite_side} position"

        return True, "OK"

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def add_position(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,
        use_dynamic_management: bool = True,
        entry_time: datetime = None,
        signal_details: dict = None,
        vtm_overrides: Optional[Dict] = None,
        leverage: int = 1,
        margin_type: str = "CROSSED",
        is_futures: bool = True,
        disable_partials: bool = False,
    ) -> bool:
        """
        Add a new position with hybrid aware VTM support
        """

        # 1. Determine Max Positions allowed for this specific trade
        max_allowed = self.max_positions_per_asset

        # ✅ NEW: Check for Aggregator Override (Ranging Mode)
        if signal_details and signal_details.get("max_trades_override"):
            max_allowed = signal_details["max_trades_override"]

            # Check current open positions for this asset
            current_total = self.get_asset_position_count(asset)

            if current_total >= max_allowed:
                logger.warning(
                    f"[SAFEGUARD] 🛡️ Ranging Mode Active: Max positions capped at {max_allowed}. Cannot open new trade."
                )
                return False

        # ✅ NEW: Check if this is an import from sync
        is_sync_import = signal_details and signal_details.get("source") == "sync_import"
        # ✅ NEW: Check if this is a Small Account Protocol trade
        is_small_account_protocol_trade = signal_details and signal_details.get("small_account_protocol_active", False)

        # 2. Check portfolio exposure limits (SKIP IF IMPORTING OR SMALL ACCOUNT PROTOCOL TRADE)
        if not is_sync_import and not is_small_account_protocol_trade and not self.check_portfolio_limits(
            new_position_usd=position_size_usd, new_side=side, asset=asset
        ):
            logger.warning(f"Portfolio limits exceeded for {asset} {side.upper()}")
            return False

        # ✅ NEW: Log if the check was bypassed due to Small Account Protocol
        if is_small_account_protocol_trade:
            # Re-check the limits just to log the warning, but don't block the trade
            if not self.check_portfolio_limits(
                new_position_usd=position_size_usd, new_side=side, asset=asset
            ):
                logger.warning(
                    f"[SMALL ACCOUNT PROTOCOL] Bypassing portfolio exposure limits for {asset} (Sniper Mode active)."
                )

        # ✅ NEW: Log if the check was bypassed for sync import
        elif is_sync_import:
            # Re-check the limits just to log the warning, but don't block the trade
            if not self.check_portfolio_limits(
                new_position_usd=position_size_usd, new_side=side, asset=asset
            ):
                logger.warning(
                    f"[SYNC IMPORT] Bypassing portfolio exposure limits to import existing position for {asset}."
                )

        quantity = position_size_usd / entry_price
        logger.info(
            f"[PORTFOLIO] Adding {side.upper()} position:\n"
            f"  Size:     ${position_size_usd:,.2f}\n"
            f"  Quantity: {quantity:.8f}\n"
            f"  Entry:    ${entry_price:,.2f}"
        )

        # ============================================================================
        # 3. CREATE POSITION OBJECT
        # ✅ VTM is initialized INSIDE Position.__init__() - don't do it here!
        # ============================================================================
        risk_config = self.config.get("assets", {}).get(asset, {}).get("risk", {})
        
        position = Position(
            asset=asset,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time or datetime.now(),
            risk_config=risk_config,
            signal_details=signal_details,
            stop_loss=stop_loss,  # May be None if VTM will calculate
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            mt5_ticket=mt5_ticket,
            binance_order_id=binance_order_id,
            ohlc_data=ohlc_data,
            account_balance=self.current_capital,
            use_dynamic_management=use_dynamic_management,  # ← This triggers VTM init in Position.__init__()
            vtm_overrides=vtm_overrides,
            leverage=leverage,
                            margin_type=margin_type,
                            is_futures=is_futures,
                            disable_partials=disable_partials,
                        )
        if use_dynamic_management and ohlc_data:
            if position.trade_manager:
                logger.info(
                    f"[VTM] Initialized for {asset}. SL: ${position.stop_loss:,.2f}"
                )
            else:
                logger.warning(f"[VTM] Failed to initialize for {asset}")

        # ============================================================================
        # 5. STORE POSITION
        # ============================================================================
        self.positions[position.position_id] = position

        # 6. Database Logging
        if self.db_manager:
            try:
                trade_id, is_new = self.db_manager.insert_trade_entry(
                    asset=asset,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=position.quantity,
                    position_size_usd=position_size_usd,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    position_id=position.position_id,
                    exchange=self.config["assets"][asset].get("exchange", "binance"),
                    mt5_ticket=mt5_ticket,
                    binance_order_id=binance_order_id,
                    vtm_enabled=bool(position.trade_manager),
                    metadata={
                        "trailing_stop_pct": trailing_stop_pct,
                        "entry_time": position.entry_time.isoformat(),
                        "preset_used": (
                            signal_details.get("preset_config", {}).get(
                                "name", "default"
                            )
                            if signal_details
                            else "manual"
                        ),
                    },
                )

                position.db_trade_id = trade_id
                position.db_manager = self.db_manager

                if is_new:
                    logger.debug(f"[DB] New trade created: {trade_id}")

            except Exception as e:
                logger.error(f"[DB] Error logging trade entry: {e}")

        # 7. Final Logging
        current_count = self.get_asset_position_count(asset, side)
        logger.info(
            f"✓ Position #{current_count} opened: {asset} {side.upper()} "
            f"@ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f} "
            f"| ID: {position.position_id}"
        )

        return True

    def update_positions_with_ohlc(self, ohlc_data_dict: dict):
        """
        Update all positions with new OHLC data for dynamic management

        Args:
            ohlc_data_dict: Dict with asset keys and {'high': float, 'low': float, 'close': float} values

        Example:
            portfolio_manager.update_positions_with_ohlc({
                'BTC': {'high': 45000, 'low': 44500, 'close': 44800},
                'GOLD': {'high': 2050, 'low': 2045, 'close': 2048}
            })
        """
        positions_to_close = []

        for asset, position in self.positions.items():
            if asset not in ohlc_data_dict:
                continue

            ohlc = ohlc_data_dict[asset]

            try:
                # Update position with new bar
                exit_signal = position.update_with_new_bar(
                    high=ohlc["high"], low=ohlc["low"], close=ohlc["close"]
                )

                # If VTM signals exit, mark for closure
                if exit_signal:
                    positions_to_close.append((asset, ohlc["close"], exit_signal))
                    logger.info(
                        f"[VTM] {asset} triggered {exit_signal.upper()} @ ${ohlc['close']:,.2f}"
                    )

            except Exception as e:
                logger.error(f"[VTM] Error updating {asset}: {e}")

        # Close positions that received exit signals
        for asset, exit_price, reason in positions_to_close:
            self.close_position(
                asset=asset, exit_price=exit_price, reason=f"VTM_{reason}"
            )

        return len(positions_to_close)

    def get_asset_balance(self, asset: str) -> float:
        """
        Get balance for a specific asset's exchange

        Args:
            asset: "BTC" or "GOLD"

        Returns:
            Balance in USD for that asset's exchange
        """
        if self.is_paper_mode:
            # In paper mode, split capital proportionally
            if asset == "BTC":
                return self.paper_capital * 0.9  # 90% for BTC
            elif asset == "GOLD":
                return self.paper_capital * 0.1  # 10% for Gold
            return self.paper_capital

        # Live mode - fetch from specific exchange
        if asset == "GOLD":
            balance = self._fetch_mt5_balance()
            if balance:
                logger.debug(f"[ASSET BALANCE] {asset}: ${balance:,.2f} (MT5)")
                return balance

        elif asset == "BTC":
            balance = self._fetch_binance_balance()
            if balance:
                logger.debug(f"[ASSET BALANCE] {asset}: ${balance:,.2f} (Binance)")
                return balance

        # Fallback: use proportion of total capital
        logger.warning(
            f"[ASSET BALANCE] Could not fetch {asset} balance, using estimate"
        )

        # Estimate based on asset weight in config
        asset_weight = self.config["assets"].get(asset, {}).get("weight", 0.5)
        return self.current_capital * asset_weight

    def close_all_positions(self, prices: Dict[str, float] = None):
        """
        ✅ FIXED: Close all open positions
        Fixes bug where exit_price was being interpreted as position_id
        """
        logger.info("Closing all positions...")

        # Use list() to create a copy of keys since we modify dict during iteration
        position_ids = list(self.positions.keys())

        for pid in position_ids:
            if pid not in self.positions:
                continue

            position = self.positions[pid]
            asset_name = position.asset

            # Get correct exit price using ASSET name
            exit_price = (
                prices.get(asset_name, position.entry_price)
                if prices
                else position.entry_price
            )

            # ✅ FIX: Use keyword arguments to ensure data goes to correct parameters
            self.close_position(
                position_id=pid,  # ← String position ID
                exit_price=exit_price,  # ← Float price
                reason="manual_close_all",
            )

        logger.info("All positions closed")

    def close_all_positions_for_asset(
        self, asset: str, exit_price: float = None, reason: str = "manual_close_asset"
    ) -> List[Dict]:
        """
        ✅ NEW: Close ALL open positions for a specific asset (e.g., BTC)
        Used by Telegram 'Close BTC' buttons to ensure total exit.

        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            exit_price: Exit price (optional, will fetch if not provided)
            reason: Close reason

        Returns:
            List of trade results for closed positions
        """
        logger.info(f"Closing ALL positions for {asset}...")

        # Find all positions matching the asset
        positions_to_close = [p for p in self.positions.values() if p.asset == asset]

        if not positions_to_close:
            logger.warning(f"No positions found for {asset} to close.")
            return []

        results = []
        for pos in positions_to_close:
            # Determine exit price if not provided
            current_exit_price = exit_price

            if current_exit_price is None:
                # Try to get real-time price from handlers
                if hasattr(self, "mt5_handler") and pos.mt5_ticket:
                    try:
                        import MetaTrader5 as mt5

                        tick = mt5.symbol_info_tick(pos.symbol)
                        if tick:
                            current_exit_price = (tick.ask + tick.bid) / 2
                    except:
                        pass

                elif hasattr(self, "binance_handler") and pos.binance_order_id:
                    try:
                        # Get from binance handler if available
                        # (you'll need to pass handlers to portfolio manager for this)
                        pass
                    except:
                        pass

                # Fallback to entry price if live price unavailable
                if current_exit_price is None:
                    current_exit_price = pos.entry_price

            # Close the individual position
            result = self.close_position(
                position_id=pos.position_id,
                exit_price=current_exit_price,
                reason=reason,
            )

            if result:
                results.append(result)

        logger.info(
            f"Successfully closed {len(results)}/{len(positions_to_close)} positions for {asset}"
        )
        return results

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def close_position(
        self,
        asset: str = None,
        position_id: str = None,
        exit_price: float = None,
        reason: str = "manual",
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Close position - Validates exchange close before removing from portfolio
        """
        # Find position to close
        if position_id:
            position = self.positions.get(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found in portfolio")
                return None
            
            # Check if position is already being closed
            if position.closing:
                logger.info(f"Position {position_id} is already in the process of being closed. Skipping.")
                return None
            
            # Mark the position as closing to prevent race conditions
            position.closing = True
        elif asset:
            positions = self.get_asset_positions(asset)
            if not positions:
                logger.warning(f"No positions to close for {asset}")
                return None
            position = positions[0]
            position_id = position.position_id
        else:
            logger.error("Must provide either asset or position_id")
            return None

        # ================================================================
        # ✅ STEP 1: CLOSE ON EXCHANGE FIRST (MT5 or Binance)
        # ================================================================
        exchange_closed = False
        close_error_msg = "Unknown handler error" # Default error message

        if not self.is_paper_mode:
            asset_cfg = self.config["assets"].get(position.asset, {})
            exchange = asset_cfg.get("exchange", "binance")
            handler = self.execution_handlers.get(exchange)

            if not handler:
                close_error_msg = f"{exchange.upper()} handler not available"
            else:
                try:
                    logger.info(f"[{exchange.upper()}] Attempting to close position {position.position_id}...")
                    exchange_closed = handler._close_position(
                        position=position,
                        current_price=exit_price,
                        asset_name=position.asset,
                        reason=reason,
                    )
                    if not exchange_closed:
                        close_error_msg = f"{exchange.upper()} order was rejected or failed. Check handler logs."

                except Exception as e:
                    close_error_msg = f"Handler exception: {str(e)}"
                    logger.error(f"[{exchange.upper()}] Error closing position: {e}", exc_info=True)
        else:
            # Paper mode always succeeds
            exchange_closed = True

        # ================================================================
        # ✅ STEP 2: ABORT OR PROCEED
        # ================================================================
        if not exchange_closed:
            logger.error(
                f"[CRITICAL] Position close failed on exchange!\n"
                f"  Position ID: {position_id}\n"
                f"  Asset:       {position.asset}\n"
                f"  Error:       {close_error_msg}\n"
                f"  ⚠️  NOT removing from portfolio to prevent data mismatch."
            )
            return None

        # ================================================================
        # ✅ STEP 3: CALCULATE P&L (Only if exchange close succeeded)
        # ================================================================
        pnl = position.get_pnl(exit_price)
        pnl_pct = position.get_pnl_pct(exit_price)
        self.realized_pnl_today += pnl

        # Update capital
        if self.is_paper_mode:
            self.current_capital += pnl
            self.equity = self.current_capital
        else:
            self.refresh_capital()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # ================================================================
        # ✅ STEP 4: CREATE TRADE RESULT
        # ================================================================
        trade_result = {
            "asset": asset or position.asset,
            "position_id": position_id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "holding_time": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "reason": reason,
            "mt5_ticket": position.mt5_ticket,
            "binance_order_id": position.binance_order_id,
            "exchange_closed": exchange_closed,
        }

        # ================================================================
        # ✅ STEP 5: LOG TO DATABASE
        # ================================================================
        if self.db_manager and hasattr(position, "db_trade_id") and position.db_trade_id:
            try:
                holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                self.db_manager.update_trade_exit(
                    trade_id=position.db_trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_time_hours=holding_time,
                    final_quantity=position.quantity,
                    metadata={"exit_time": datetime.now().isoformat(), "exchange_closed": exchange_closed},
                )
                logger.debug(f"[DB] Trade exit logged: {position.db_trade_id}")
            except Exception as e:
                logger.error(f"[DB] Error logging trade exit: {e}")

        # ================================================================
        # ✅ STEP 6: REMOVE FROM PORTFOLIO
        # ================================================================
        self.closed_positions.append(trade_result)
        del self.positions[position_id]

        remaining_count = self.get_asset_position_count(position.asset, position.side)

        logger.info(
            f"✓ Position closed successfully:\n"
            f"  Asset:     {position.asset} {position.side.upper()}\n"
            f"  Exit:      ${exit_price:,.2f}\n"
            f"  P&L:       ${pnl:,.2f} ({pnl_pct:.2%})\n"
            f"  Remaining: {remaining_count}/{self.max_positions_per_asset}"
        )

        # Send notification
        if self.telegram_bot and self.telegram_bot._current_loop:
            try:
                # Use a thread-safe method to call the async notification
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.notify_trade_closed(
                        asset=trade_result["asset"],
                        side=trade_result["side"],
                        pnl=trade_result["pnl"],
                        pnl_pct=trade_result["pnl_pct"] * 100, # Convert to percentage points
                        reason=trade_result["reason"],
                    ),
                    self.telegram_bot._current_loop
                )
            except Exception as e:
                logger.error(f"[TELEGRAM] Failed to send close notification from PM: {e}")

        return trade_result

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.WARNING,
        notify=False,  # Don't notify for update errors
        reraise=False,
        default_return=None,
    )
    def update_positions(self, prices: Dict[str, float] = None):
        """Update all positions with current prices and exchange profit"""
        # Update exchange positions with real-time profit
        if not self.is_paper_mode:
            self.update_mt5_positions_profit()
            self.update_binance_positions_profit()

        if prices:
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    if len(self.price_history[asset]) > 100:
                        self.price_history[asset].pop(0)

        # Calculate unrealized P&L
        total_unrealized_pnl = 0.0
        for pos in self.positions.values():
            # Prioritize exchange-reported profit
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                # Use MT5 profit for MT5 positions
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                # Use Binance profit for Binance positions
                total_unrealized_pnl += pos.binance_profit
            elif prices and pos.asset in prices:
                # Calculate for positions without exchange tracking
                total_unrealized_pnl += pos.get_pnl(prices[pos.asset])

        if self.is_paper_mode:
            # In paper mode: equity = cash + unrealized P&L
            self.equity = self.current_capital + total_unrealized_pnl
        else:
            # In live mode: periodically refresh from exchanges
            pass

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

    def get_position(self, asset: str, position_id: str = None) -> Optional[Position]:
        """
        Get position(s) for an asset

        Args:
            asset: Asset name
            position_id: Optional specific position ID

        Returns:
            Position object if position_id provided, otherwise first position for asset
        """
        if position_id:
            return self.positions.get(position_id)

        # Return first position for asset (for backward compatibility)
        positions = self.get_asset_positions(asset)
        return positions[0] if positions else None

    def has_position(self, asset: str, side: str = None) -> bool:
        """
        Check if we have any open positions for an asset

        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        return self.get_asset_position_count(asset, side) > 0

    def reset_daily_pnl(self):
        """Reset realized P&L tracker (call this at start of each trading day)"""
        self.realized_pnl_today = 0.0
        logger.info("Daily P&L tracker reset")

    def start_trading_session(self):
        """Start trading session"""
        self.session_start_time = datetime.now()
        self.session_start_equity = self.equity
        self.session_start_capital = self.current_capital
        self.realized_pnl_today = 0.0
        logger.info(f"Trading session started at {self.session_start_time}")

    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        ✅ FIX: Auto-refresh balances when getting status
        """
        # ✅ Refresh if stale (respects time interval)
        self.refresh_capital(force=False)

        if current_prices is None:
            current_prices = {
                pos.asset: pos.entry_price for pos in self.positions.values()
            }

        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        total_notional_value = 0.0
        total_margin_used = 0.0

        # ✅  Count positions per asset correctly
        asset_position_counts = {}
        asset_positions_detail = {}

        for asset in ["BTC", "GOLD"]:
            # Get all positions for this asset
            long_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "long"
            ]
            short_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "short"
            ]

            asset_position_counts[asset] = {
                "long": len(long_positions),
                "short": len(short_positions),
                "total": len(long_positions) + len(short_positions),
            }

            # Detailed info for debugging
            asset_positions_detail[asset] = {
                "long_ids": [p.position_id for p in long_positions],
                "short_ids": [p.position_id for p in short_positions],
                "long_tickets": [p.mt5_ticket for p in long_positions if p.mt5_ticket],
                "short_tickets": [
                    p.mt5_ticket for p in short_positions if p.mt5_ticket
                ],
            }

        # Calculate exposures and P&L
        for pos in self.positions.values():
            current_price = current_prices.get(pos.asset, pos.entry_price)
            notional_value = pos.quantity * current_price
            
            # Get leverage (defaults to 1 for spot trading)
            leverage = getattr(pos, 'leverage', 1)
            
            # ✅ CORRECT: Actual margin used = notional / leverage
            # This represents your REAL capital at risk
            margin_used = notional_value / leverage
            
            # Accumulate
            total_notional_value += notional_value
            total_margin_used += margin_used
            total_exposure += margin_used  # ← Use margin, not notional
            
            # Calculate P&L (unchanged)
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                total_unrealized_pnl += pos.binance_profit
            else:
                total_unrealized_pnl += pos.get_pnl(current_price)
        
        total_value = self.current_capital + total_unrealized_pnl

        # Calculate daily P&L
        if self.session_start_equity is not None:
            current_equity = self.current_capital + total_unrealized_pnl
            daily_pnl = current_equity - self.session_start_equity
        else:
            daily_pnl = self.realized_pnl_today + total_unrealized_pnl

        return {
        "mode": self.mode,
        "total_value": total_value,
        "capital": self.current_capital,
        "equity": self.equity,
        "cash": self.current_capital,
        
        # ✅ NEW: Separate notional vs actual exposure
        "total_notional_value": total_notional_value,      # For information
        "total_margin_used": total_margin_used,            # For risk limits
        "total_exposure": total_exposure,                  # ← This is margin_used
        
        "open_positions": len(self.positions),
        "daily_pnl": daily_pnl,
        "realized_pnl_today": self.realized_pnl_today,
        "total_unrealized_pnl": total_unrealized_pnl,
        "asset_position_counts": asset_position_counts,
        "asset_positions_detail": asset_positions_detail,
        "max_positions_per_asset": self.max_positions_per_asset,
        
        # Individual positions...
        "positions": {
            pos.position_id: {
                "asset": pos.asset,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "current_price": current_prices.get(pos.asset, pos.entry_price),
                "current_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price),
                
                # ✅ NEW: Add leverage info to position details
                "leverage": getattr(pos, 'leverage', 1),
                "notional_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price),
                "margin_used": (pos.quantity * current_prices.get(pos.asset, pos.entry_price)) / getattr(pos, 'leverage', 1),
                
                "pnl": pos.get_pnl(current_prices.get(pos.asset, pos.entry_price)),
                "pnl_pct": pos.get_pnl_pct(current_prices.get(pos.asset, pos.entry_price)),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "mt5_ticket": pos.mt5_ticket,
                "mt5_profit": pos.mt5_profit if pos.mt5_ticket else None,
                "binance_order_id": pos.binance_order_id,
                "binance_profit": pos.binance_profit if pos.binance_order_id else None,
                "leverage": getattr(pos, "leverage", 1),
                "margin_type": getattr(pos, "margin_type", "SPOT"),
                "is_futures": getattr(pos, "is_futures", False),
            }
            for pos in self.positions.values()
        },
        }

    def close_all_positions(self, prices: Dict[str, float] = None):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for asset in list(self.positions.keys()):
            position = self.positions[asset]
            exit_price = (
                prices.get(asset, position.entry_price)
                if prices
                else position.entry_price
            )
            self.close_position(asset, exit_price, reason="shutdown")

        logger.info("All positions closed")
