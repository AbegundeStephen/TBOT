#!/usr/bin/env python3
"""
Main Trading Bot - WITH TELEGRAM INTEGRATION
FIXED: Proper EMA Strategy Integration
"""

import json
import logging
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
import schedule
import io
from threading import Thread

# FIX Windows encoding BEFORE any imports that use logging
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except:
        pass

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy
from src.execution.signal_aggregator import BullMarketFilteredAggregator
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.market_hours import MarketHours, should_trade_btc, should_trade_gold

# Import Telegram bot
from src.telegram import TradingTelegramBot
from telegram_config import TELEGRAM_CONFIG


# Setup logging with UTF-8 encoding
def setup_logging(config):
    """Setup logging with proper encoding"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trading_bot.log")

    Path(log_file).parent.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot - WITH TELEGRAM INTEGRATION
    """

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT WITH TELEGRAM")
        logger.info("=" * 70)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        setup_logging(self.config)
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None

        # FIXED: Use consistent strategy names
        self.strategies = {"BTC": {}, "GOLD": {}}

        self._initialize_strategies()
        self.aggregators = {}
        self._initialize_aggregators()

        self.binance_handler = None
        self.mt5_handler = None

        self.is_running = False
        self.trade_count_today = 0
        self.daily_loss = 0.0
        self.last_trade_date = None
        self.last_trade_times = {}
        self.last_market_status_log = None

        # TELEGRAM INTEGRATION
        self.telegram_bot = None
        self.telegram_loop = None
        self.telegram_thread = None
        self.telegram_ready = asyncio.Event()
        self._initialize_telegram()

    def initialize_exchanges(self):
        """
        Initialize exchange connections and portfolio manager
        Portfolio manager MUST be initialized AFTER exchanges to fetch actual capital
        """
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Exchange Connections")
        logger.info("-" * 70)

        # Track MT5 initialization status
        mt5_initialized = False

        # Initialize MT5 first (for GOLD)
        if self.config["assets"]["GOLD"].get("enabled", False):
            if self.data_manager.initialize_mt5():
                logger.info("[OK] MT5 connection established")
                mt5_initialized = True
            else:
                logger.error("[FAIL] Failed to initialize MT5 - GOLD trading disabled")

        # Initialize Binance (for BTC)
        if self.config["assets"]["BTC"].get("enabled", False):
            if self.data_manager.initialize_binance():
                logger.info("[OK] Binance connection established")
            else:
                logger.error(
                    "[FAIL] Failed to initialize Binance - BTC trading disabled"
                )

        # NOW initialize portfolio manager with exchange connections
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Portfolio Manager with Exchange Capital")
        logger.info("-" * 70)

        # Pass MT5 handler and Binance client to portfolio manager
        import MetaTrader5 as mt5

        mt5_handler = mt5 if mt5_initialized else None

        self.portfolio_manager = PortfolioManager(
            self.config,
            mt5_handler=mt5_handler,
            binance_client=self.data_manager.binance_client,
        )

        # Initialize execution handlers AFTER portfolio manager
        if (
            self.config["assets"]["BTC"].get("enabled", False)
            and self.data_manager.binance_client
        ):
            self.binance_handler = BinanceExecutionHandler(
                self.config,
                self.data_manager.binance_client,
                self.portfolio_manager,
            )
            logger.info("[OK] Binance handler initialized")

        if self.config["assets"]["GOLD"].get("enabled", False) and mt5_initialized:
            self.mt5_handler = MT5ExecutionHandler(self.config, self.portfolio_manager)
            logger.info("[OK] MT5 handler initialized")

    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            if not TELEGRAM_CONFIG.get("enabled", False):
                logger.info("[TELEGRAM] Disabled in config")
                return

            token = TELEGRAM_CONFIG.get("bot_token")
            admin_ids = TELEGRAM_CONFIG.get("admin_ids", [])

            if not token:
                logger.warning("[TELEGRAM] No bot token provided, skipping")
                return

            if not admin_ids:
                logger.warning("[TELEGRAM] No admin IDs provided, skipping")
                return

            self.telegram_bot = TradingTelegramBot(
                token=token, admin_ids=admin_ids, trading_bot=self
            )

            logger.info(f"[TELEGRAM] Bot initialized for {len(admin_ids)} admin(s)")

        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to initialize: {e}")
            self.telegram_bot = None

    def _initialize_strategies(self):
        """
        FIXED: Initialize all THREE strategies for each enabled asset
        Properly handles exponential_moving_averages config key
        """
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Strategies (Mean Reversion + Trend Following + EMA)")
        logger.info("-" * 70)

        for asset_name, asset_config in self.config["assets"].items():
            if not asset_config.get("enabled", False):
                logger.info(f"[!] {asset_name}: Disabled in config")
                continue

            strategies_config = asset_config.get("strategies", {})

            # 1. Mean Reversion
            if strategies_config.get("mean_reversion", {}).get("enabled", False):
                mr_config = self.config["strategy_configs"]["mean_reversion"][
                    asset_name
                ]
                self.strategies[asset_name]["mean_reversion"] = MeanReversionStrategy(
                    mr_config
                )
                logger.info(f"[OK] {asset_name}: Mean Reversion initialized")

            # 2. Trend Following
            if strategies_config.get("trend_following", {}).get("enabled", False):
                tf_config = self.config["strategy_configs"]["trend_following"][
                    asset_name
                ]
                self.strategies[asset_name]["trend_following"] = TrendFollowingStrategy(
                    tf_config
                )
                logger.info(f"[OK] {asset_name}: Trend Following initialized")

            # 3. EMA Strategy (FIXED: Check correct config key)
            # Config file uses "exponential_moving_averages" but we store as "ema_strategy"
            if strategies_config.get("exponential_moving_averages", {}).get(
                "enabled", False
            ):
                ema_config = self.config["strategy_configs"][
                    "exponential_moving_averages"
                ][asset_name]
                self.strategies[asset_name]["ema_strategy"] = EMAStrategy(ema_config)
                logger.info(f"[OK] {asset_name}: EMA Strategy initialized")

            # Log what we have
            enabled_count = len(self.strategies[asset_name])
            if enabled_count == 0:
                logger.warning(f"[!] {asset_name}: NO strategies enabled!")
            else:
                logger.info(
                    f"[OK] {asset_name}: {enabled_count}/3 strategies active: {list(self.strategies[asset_name].keys())}"
                )

    def _initialize_aggregators(self):
        """
        FIXED: Initialize signal aggregators with proper strategy checking
        """
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Enhanced Signal Aggregators")
        logger.info("-" * 70)

        AGGREGATOR_PRESETS = {
            "conservative": {
                "mean_reversion_weight": 0.6,
                "trend_following_weight": 1.2,
                "ema_weight": 1.2,
                "buy_score_threshold": 0.45,
                "sell_score_threshold": 0.50,
                "perfect_agreement_bonus": 0.15,
                "allow_single_mr_signal": False,
                "allow_single_tf_signal": True,
                "allow_single_ema_signal": True,
                "single_tf_threshold": 0.65,
                "single_ema_threshold": 0.65,
                "enable_bull_filter": True,
                "block_sells_in_bull": False,
                "boost_buys_in_bull": 0.10,
                "sell_penalty_in_bull": 0.15,
                "regime_confirmation_bars": 3,
                "regime_cooldown_hours": 6,
                "verbose_logging": True,
            },
            "balanced": {
                "mean_reversion_weight": 0.7,
                "trend_following_weight": 1.1,
                "ema_weight": 1.2,
                "buy_score_threshold": 0.35,
                "sell_score_threshold": 0.40,
                "perfect_agreement_bonus": 0.15,
                "allow_single_mr_signal": False,
                "allow_single_tf_signal": True,
                "allow_single_ema_signal": True,
                "single_tf_threshold": 0.60,
                "single_ema_threshold": 0.60,
                "enable_bull_filter": True,
                "block_sells_in_bull": False,
                "boost_buys_in_bull": 0.15,
                "sell_penalty_in_bull": 0.20,
                "regime_confirmation_bars": 2,
                "regime_cooldown_hours": 6,
                "verbose_logging": True,
            },
            "aggressive": {
                "mean_reversion_weight": 0.8,
                "trend_following_weight": 1.0,
                "ema_weight": 1.1,
                "buy_score_threshold": 0.30,
                "sell_score_threshold": 0.35,
                "perfect_agreement_bonus": 0.12,
                "allow_single_mr_signal": False,
                "allow_single_tf_signal": True,
                "allow_single_ema_signal": True,
                "single_tf_threshold": 0.55,
                "single_ema_threshold": 0.55,
                "enable_bull_filter": True,
                "block_sells_in_bull": False,
                "boost_buys_in_bull": 0.15,
                "sell_penalty_in_bull": 0.15,
                "regime_confirmation_bars": 1,
                "regime_cooldown_hours": 3,
                "verbose_logging": True,
            },
        }
        aggregator_mode = self.config.get("aggregator_settings", {}).get(
            "mode", "weighted_voting"
        )
        aggregator_preset = self.config.get("aggregator_settings", {}).get(
            "preset", "balanced"  # FIX: Changed from "weighted_voting" to "balanced"
        )
        if aggregator_preset not in AGGREGATOR_PRESETS:
            logger.warning(f"Unknown preset '{aggregator_preset}', using 'balanced'")
            aggregator_preset = "balanced"

        confidence_config = AGGREGATOR_PRESETS[aggregator_preset].copy()
        logger.info(f"Aggregator Mode: {aggregator_mode}")
        logger.info(f"Aggregator Preset: {aggregator_preset}")

        for asset_name, strategies in self.strategies.items():
            # FIXED: Get strategies with correct keys
            mr_strategy = strategies.get("mean_reversion")
            tf_strategy = strategies.get("trend_following")
            ema_strategy = strategies.get("ema_strategy")  # FIXED: Use consistent key

            # Count available strategies
            available = sum(
                [
                    mr_strategy is not None,
                    tf_strategy is not None,
                    ema_strategy is not None,
                ]
            )

            if available == 0:
                logger.warning(
                    f"[!] {asset_name}: No strategies available for aggregator"
                )
                continue

            # Use safe lookup for preset (fallback to balanced if missing)

            # Create aggregator (it will handle missing strategies gracefully)
            self.aggregators[asset_name] = BullMarketFilteredAggregator(
                mean_reversion_strategy=mr_strategy,
                trend_following_strategy=tf_strategy,
                ema_strategy=ema_strategy,
                confidence_config=confidence_config,  # Use the preset config we created above
                asset_name=asset_name,
            )
            logger.info(
                f"[OK] {asset_name}: Aggregator initialized with {available}/3 strategies "
                f"({aggregator_mode}/{aggregator_preset})"
            )

    def load_models(self):
        """
        FIXED: Load trained ML models for ALL strategies including EMA
        """
        logger.info("\n" + "-" * 70)
        logger.info("Loading Trained Models (Including EMA)")
        logger.info("-" * 70)

        models_loaded = 0
        models_expected = 0

        for asset_name, strategies in self.strategies.items():
            for strategy_name, strategy in strategies.items():
                models_expected += 1

                # FIXED: Map internal strategy name to model filename
                # Internal: "ema_strategy" -> File: "ema_strategy_btc.pkl"
                model_filename = f"{strategy_name}_{asset_name.lower()}.pkl"
                model_path = f"models/{model_filename}"

                if Path(model_path).exists():
                    if strategy.load_model(model_path):
                        logger.info(f"[OK] Loaded: {model_path}")
                        models_loaded += 1
                    else:
                        logger.error(f"[FAIL] Failed to load: {model_path}")
                else:
                    logger.error(f"[FAIL] Not found: {model_path}")

        if models_loaded == 0:
            logger.error("=" * 70)
            logger.error("NO MODELS LOADED!")
            logger.error("Please run: python train.py")
            logger.error("=" * 70)
            sys.exit(1)

        logger.info(
            f"\n[OK] Successfully loaded {models_loaded}/{models_expected} models"
        )

    def reset_daily_counters(self):
        """Reset daily trading counters"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trade_count_today = 0
            self.daily_loss = 0.0
            self.last_trade_date = current_date
            logger.info(f"[RESET] Daily counters reset for {current_date}")

            # Refresh capital from exchanges at start of new day
            if not self.portfolio_manager.is_paper_mode:
                logger.info("[REFRESH] Fetching updated capital for new trading day...")
                self.portfolio_manager.refresh_capital()

            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.send_daily_summary(), self.telegram_loop
                    )
                except Exception as e:
                    logger.error(f"Failed to send daily summary: {e}")

    def check_trading_limits(self) -> bool:
        """Check if trading limits are reached"""
        risk_config = self.config["risk_management"]

        max_daily_trades = risk_config.get("max_daily_trades", 10)
        if self.trade_count_today >= max_daily_trades:
            logger.warning(f"[LIMIT] Daily trade limit reached ({max_daily_trades})")
            return False

        max_daily_loss = risk_config.get("max_daily_loss_pct", 0.05)
        if self.daily_loss >= max_daily_loss:
            logger.warning(f"[LIMIT] Daily loss limit reached ({max_daily_loss:.2%})")
            return False

        circuit_breaker = risk_config.get("circuit_breaker_loss_pct", 0.10)
        total_loss_pct = self.daily_loss / self.portfolio_manager.initial_capital
        if total_loss_pct >= circuit_breaker:
            logger.error(f"[BREAKER] CIRCUIT BREAKER! Loss: {total_loss_pct:.2%}")

            # Notify via Telegram
            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.notify_error(
                            f"🚨 CIRCUIT BREAKER ACTIVATED!\n"
                            f"Loss: {total_loss_pct:.2%}\n"
                            f"Trading halted."
                        ),
                        self.telegram_loop,
                    )
                except:
                    pass

            return False

        trading_config = self.config["trading"]
        if trading_config.get("allow_simultaneous_positions", True):
            max_positions = trading_config.get("max_simultaneous_positions", 2)
            current_positions = self.portfolio_manager.get_open_positions_count()
            if current_positions >= max_positions:
                logger.warning(f"[LIMIT] Max positions reached ({max_positions})")
                return False

        return True

    def check_min_time_between_trades(self, asset_name: str) -> bool:
        """Check minimum time between trades for an asset"""
        min_time_minutes = self.config["trading"].get(
            "min_time_between_trades_minutes", 60
        )

        if asset_name in self.last_trade_times:
            time_since_last = datetime.now() - self.last_trade_times[asset_name]
            if time_since_last.total_seconds() < min_time_minutes * 60:
                remaining = min_time_minutes - (time_since_last.total_seconds() / 60)
                logger.info(f"[COOLDOWN] {asset_name}: {remaining:.0f} min remaining")
                return False

        return True

    def check_market_hours(self, asset_name: str) -> bool:
        """Check if market is open for the given asset"""
        if asset_name == "BTC":
            return True

        elif asset_name == "GOLD":
            is_open = should_trade_gold()

            if not is_open:
                status, message = MarketHours.get_market_status("gold")

                current_hour = datetime.now().hour
                if self.last_market_status_log != current_hour:
                    logger.info(f"[MARKET] {asset_name}: {message}")

                    seconds_until = MarketHours.time_until_market_open("gold")
                    if seconds_until > 0:
                        hours_until = seconds_until / 3600
                        logger.info(
                            f"[MARKET] {asset_name}: Market opens in {hours_until:.1f} hours"
                        )

                    self.last_market_status_log = current_hour

            return is_open

        return True

    def trade_asset(self, asset_name: str):
        """
        FIXED: Execute trading logic with proper signal logging for ALL 3 strategies
        """
        asset_config = self.config["assets"][asset_name]

        if not asset_config.get("enabled", False):
            return

        if not self.check_market_hours(asset_name):
            logger.debug(f"[SKIP] {asset_name}: Market is closed")
            return

        exchange = asset_config.get("exchange")
        symbol = asset_config.get("symbol")

        if exchange == "binance" and not self.binance_handler:
            logger.warning(f"[!] {asset_name}: Binance handler not available")
            return
        elif exchange == "mt5" and not self.mt5_handler:
            logger.warning(f"[!] {asset_name}: MT5 handler not available")
            return

        try:
            logger.info(f"\n{'-' * 70}")
            logger.info(f"Processing {asset_name}")
            logger.info(f"{'-' * 70}")

            # Show market status
            if asset_name == "GOLD":
                status, message = MarketHours.get_market_status("gold")
                logger.info(f"[MARKET] {message}")
            elif asset_name == "BTC":
                logger.info(f"[MARKET] BTC market is always open (24/7)")

            # Get data
            end_time = datetime.now(timezone.utc)

            if exchange == "binance":
                interval = asset_config.get("interval", "1h")
                lookback_days = (
                    15 if interval == "1h" else (60 if interval == "4h" else 365)
                )
            else:
                timeframe = asset_config.get("timeframe", "H1")
                lookback_days = 25 if timeframe in ["H1", "TIMEFRAME_H1"] else 75

            start_time = end_time - timedelta(days=lookback_days)

            logger.info(f"[DATA] Fetching {lookback_days} days up to NOW...")
            logger.info(
                f"[TIME] Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC"
            )

            # Fetch data
            if exchange == "binance":
                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = self.data_manager.clean_data(df)

            min_bars_for_prediction = 250  # Increased for EMA 200

            if len(df) < min_bars_for_prediction:
                logger.warning(
                    f"[!] {asset_name}: Insufficient data ({len(df)} bars, need {min_bars_for_prediction})"
                )
                return

            logger.info(
                f"[OK] Fetched {len(df)} bars "
                f"({df.index[0].strftime('%Y-%m-%d %H:%M')} to "
                f"{df.index[-1].strftime('%Y-%m-%d %H:%M')} UTC)"
            )

            # Get current price
            if exchange == "binance":
                current_price = self.binance_handler.get_current_price(symbol)
                if current_price is None:
                    logger.warning(
                        f"[!] Could not get live price, using last candle: ${df['close'].iloc[-1]:,.2f}"
                    )
                    current_price = df["close"].iloc[-1]
            else:
                current_price = df["close"].iloc[-1]

            logger.info(f"[PRICE] Current {asset_name} Price: ${current_price:,.2f}")

            # Get signal
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[!] {asset_name}: No aggregator configured")
                return

            signal, details = aggregator.get_aggregated_signal(df)
            self.telegram_bot.signal_monitor.record_signal(
                asset=asset_name,  # "BTC" or "GOLD"
                signal=signal,
                details=details,
                price=current_price,
                timestamp=datetime.now(),
            )

            # FIXED: Log ALL THREE strategy signals
            logger.info(f"\n[SIGNAL] Analysis (3 Strategies):")
            logger.info(
                f"  Mean Reversion: Signal={details.get('mean_reversion_signal', 0):>2}, "
                f"Confidence={details.get('mean_reversion_confidence', 0):.3f}"
            )
            logger.info(
                f"  Trend Following: Signal={details.get('trend_following_signal', 0):>2}, "
                f"Confidence={details.get('trend_following_confidence', 0):.3f}"
            )
            logger.info(
                f"  EMA Strategy:    Signal={details.get('ema_signal', 0):>2}, "
                f"Confidence={details.get('ema_confidence', 0):.3f}"
            )
            logger.info(f"  >> Final Signal: {signal:>2}")
            logger.info(
                f"  >> Signal Quality: {details.get('signal_quality', 0):.3f}"
            )
            logger.info(f"  >> Reasoning: {details.get('reasoning', 'N/A')}")

            # Check if we're opening a new position (for Telegram notification)
            was_new_position = False
            existing_position = self.portfolio_manager.get_position(asset_name)

            # Execute signal
            success = False

            if exchange == "binance":
                success = self.binance_handler.execute_signal(
                    signal, current_price, asset_name
                )
                self.binance_handler.check_and_update_positions(asset_name)
            else:
                success = self.mt5_handler.execute_signal(signal, symbol, asset_name)
                self.mt5_handler.check_and_update_positions(asset_name)

            # Check if a new position was opened
            new_position = self.portfolio_manager.get_position(asset_name)
            if new_position and not existing_position and signal != 0:
                was_new_position = True

            # Telegram notification for new position
            if was_new_position and success:
                try:
                    pos = new_position
                    self._send_telegram_notification(
                        self.telegram_bot.notify_trade_opened(
                            asset=asset_name,
                            side=pos.side if hasattr(pos, "side") else pos.get("side"),
                            price=current_price,
                            size=(
                                pos.quantity * current_price
                                if hasattr(pos, "quantity")
                                else pos.get("current_value", 0)
                            ),
                            sl=(
                                pos.stop_loss
                                if hasattr(pos, "stop_loss")
                                else pos.get("stop_loss", 0)
                            ),
                            tp=(
                                pos.take_profit
                                if hasattr(pos, "take_profit")
                                else pos.get("take_profit", 0)
                            ),
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to send Telegram notification: {e}")

            # Check if position was closed
            if existing_position and not new_position:
                try:
                    closed = self.portfolio_manager.closed_positions
                    if closed:
                        last_trade = closed[-1]
                        if last_trade["asset"] == asset_name:
                            self._send_telegram_notification(
                                self.telegram_bot.notify_trade_closed(
                                    asset=asset_name,
                                    side=last_trade["side"],
                                    pnl=last_trade["pnl"],
                                    pnl_pct=last_trade["pnl_pct"] * 100,
                                    reason=last_trade["reason"],
                                )
                            )
                except Exception as e:
                    logger.error(f"Failed to send close notification: {e}")

            # Count trade if opened
            if signal != 0 and success:
                if not self.check_trading_limits():
                    logger.info(
                        f"[LIMIT] {asset_name}: Trading limits prevent new position"
                    )
                    return

                if not self.check_min_time_between_trades(asset_name):
                    logger.info(f"[COOLDOWN] {asset_name}: Cooldown period active")
                    return

                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()

                signal_type = "BUY" if signal == 1 else "SELL"
                logger.info(
                    f"[SUCCESS] {asset_name} {signal_type} executed "
                    f"(Daily count: {self.trade_count_today})"
                )

                if self.config.get("logging", {}).get("save_trades", True):
                    self._log_trade(asset_name, signal, details, current_price)

        except Exception as e:
            logger.error(f"[ERROR] Error in {asset_name} trading: {e}", exc_info=True)

            # Telegram error notification
            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.notify_error(
                            f"Error in {asset_name} trading:\n{str(e)[:200]}"
                        ),
                        self.telegram_loop,
                    )
                except:
                    pass

    def _log_trade(self, asset_name: str, signal: int, details: dict, price: float):
        """Log trade details to separate file"""
        try:
            trade_log_file = Path("logs/trades.log")
            trade_log_file.parent.mkdir(exist_ok=True)

            with open(trade_log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()},{asset_name},{signal},{price:.2f},"
                    f"{details.get('combined_confidence', 0):.3f},{details.get('reasoning', 'N/A')}\n"
                )
        except Exception as e:
            logger.warning(f"Could not log trade: {e}")

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("\n" + "=" * 70)
        logger.info(
            f"[CYCLE] TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 70)

        # Refresh capital periodically (every cycle in live mode)
        if not self.portfolio_manager.is_paper_mode:
            self.portfolio_manager.refresh_capital()

        self.reset_daily_counters()
        self.portfolio_manager.update_positions()

        enabled_assets = [
            name
            for name, config in self.config["assets"].items()
            if config.get("enabled", False)
        ]

        logger.info(f"\n[ASSETS] Enabled: {', '.join(enabled_assets)}")

        for asset_name in enabled_assets:
            self.trade_asset(asset_name)
            time.sleep(2)

        portfolio_status = self.portfolio_manager.get_portfolio_status()

        logger.info(f"\n{'-' * 70}")
        logger.info("[PORTFOLIO] Status")
        logger.info(f"{'-' * 70}")
        logger.info(f"Mode: {portfolio_status['mode'].upper()}")
        logger.info(f"Total Value: ${portfolio_status.get('total_value', 0):,.2f}")
        logger.info(f"Cash: ${portfolio_status.get('cash', 0):,.2f}")
        logger.info(f"Open Positions: {portfolio_status.get('open_positions', 0)}")
        logger.info(f"Daily P&L: ${portfolio_status.get('daily_pnl', 0):,.2f}")

        logger.info("\n[OK] Trading cycle complete")
        logger.info("=" * 70)

    async def _start_telegram_bot(self):
        """Start Telegram bot with proper error handling and clean exit"""
        try:
            logger.info("[TELEGRAM] Initializing bot...")
            await self.telegram_bot.initialize()

            # Signal that bot is ready
            logger.info("[TELEGRAM] Bot ready, signaling main thread")

            # Keep the bot running until shutdown signal
            while self.is_running:
                await asyncio.sleep(1)

            # When is_running becomes False, exit gracefully
            logger.info("[TELEGRAM] Bot loop exiting due to shutdown signal...")

        except asyncio.CancelledError:
            logger.info("[TELEGRAM] Bot task cancelled")
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            logger.error(f"[TELEGRAM] Fatal error: {e}", exc_info=True)
        finally:
            # Final cleanup will happen in _run_telegram_loop's finally block
            logger.info("[TELEGRAM] _start_telegram_bot() finished")

    def _run_telegram_loop(self):
        """Run Telegram bot event loop with proper cleanup"""
        self.telegram_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.telegram_loop)
        try:
            self.telegram_loop.run_until_complete(self._start_telegram_bot())
        except Exception as e:
            logger.error(f"[TELEGRAM] Loop error: {e}", exc_info=True)
        finally:
            # Cancel all remaining tasks before closing loop
            try:
                # Get all tasks from THIS loop
                pending = asyncio.all_tasks(self.telegram_loop)

                if pending:
                    logger.info(
                        f"[TELEGRAM] Cancelling {len(pending)} pending tasks..."
                    )
                    for task in pending:
                        task.cancel()

                    # Wait for cancellations with timeout
                    try:
                        self.telegram_loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=3.0,
                            )
                        )
                        logger.info("[TELEGRAM] All tasks cancelled successfully")
                    except asyncio.TimeoutError:
                        logger.warning("[TELEGRAM] Task cancellation timed out")
                    except Exception as e:
                        logger.error(f"[TELEGRAM] Error during task cancellation: {e}")

            except Exception as e:
                logger.error(f"[TELEGRAM] Error in cleanup: {e}")
            finally:
                # Now safely close the loop
                try:
                    if not self.telegram_loop.is_closed():
                        self.telegram_loop.close()
                        logger.info("[TELEGRAM] Event loop closed")
                except Exception as e:
                    logger.error(f"[TELEGRAM] Error closing loop: {e}")

    def _send_telegram_notification(self, coro):
        """
        Helper to safely send Telegram notifications from main thread

        Args:
            coro: Coroutine to run (e.g., self.telegram_bot.notify_trade_opened(...))
        """
        if not self.telegram_bot or not self.telegram_loop:
            return

        if not self.telegram_bot._is_ready:
            logger.warning("[TELEGRAM] Bot not ready for notifications yet")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.telegram_loop)
            # Wait briefly for completion (non-blocking)
            future.result(timeout=5)
        except TimeoutError:
            logger.warning("[TELEGRAM] Notification timed out")
        except Exception as e:
            logger.error(f"[TELEGRAM] Notification error: {e}")

    def start(self):
        """Start the trading bot"""
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT STARTING")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.config['trading']['mode'].upper()}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Strategies: Mean Reversion + Trend Following + EMA Crossover")

        self.initialize_exchanges()
        self.load_models()

        logger.info("\n[MARKET HOURS]")
        logger.info("  BTC (Crypto): 24/7 - Always trading")
        logger.info("  GOLD (Forex):  Sun 22:00 GMT - Fri 22:00 GMT")
        logger.info("=" * 70)

        if not self.binance_handler and not self.mt5_handler:
            logger.error("=" * 70)
            logger.error("[FAIL] NO EXCHANGE HANDLERS AVAILABLE!")
            logger.error("Cannot proceed with trading.")
            logger.error("=" * 70)
            sys.exit(1)

        # Start Telegram bot in separate thread
        if self.telegram_bot:
            logger.info("\n[TELEGRAM] Starting Telegram bot...")
            self.telegram_thread = Thread(target=self._run_telegram_loop, daemon=True)
            self.telegram_thread.start()

            # CRITICAL: Wait for bot to be fully ready
            logger.info("[TELEGRAM] Waiting for bot to be ready...")
            timeout = 30
            start_wait = time.time()

            while (
                not self.telegram_bot._is_ready and (time.time() - start_wait) < timeout
            ):
                time.sleep(0.5)

            if self.telegram_bot._is_ready:
                logger.info("[TELEGRAM] ✅ Bot is ready for notifications")
            else:
                logger.error("[TELEGRAM] ❌ Bot failed to become ready within timeout")

        # Now start trading
        check_interval = self.config["trading"].get("check_interval_seconds", 300)
        schedule.every(check_interval).seconds.do(self.run_trading_cycle)

        self.is_running = True
        logger.info(f"\n[OK] Trading bot is now running")
        logger.info(
            f"[TIME] Checking market every {check_interval}s ({check_interval/60:.1f} minutes)"
        )
        logger.info(f"Press Ctrl+C to stop\n")

        try:
            self.run_trading_cycle()

            while self.is_running:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n[!] Shutdown signal received...")
            self.stop()

    def stop(self):
        """Stop the trading bot with improved Telegram shutdown"""
        logger.info("\n" + "=" * 70)
        logger.info("[STOP] STOPPING TRADING BOT")
        logger.info("=" * 70)

        # Signal bot to stop FIRST - this will exit the _start_telegram_bot loop
        self.is_running = False

        # Close positions if required
        if self.config["trading"].get("close_positions_on_shutdown", False):
            logger.info("Closing all open positions...")
            self.portfolio_manager.close_all_positions()

        # Shutdown Telegram bot gracefully
        if (
            self.telegram_bot
            and self.telegram_loop
            and not self.telegram_loop.is_closed()
        ):
            try:
                logger.info("[TELEGRAM] Initiating bot shutdown...")

                if self.telegram_bot.is_running:
                    # Schedule shutdown in the bot's event loop
                    future = asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.shutdown(), self.telegram_loop
                    )

                    # Wait for shutdown with timeout
                    try:
                        future.result(timeout=8)
                        logger.info("[TELEGRAM] Bot shutdown completed")
                    except TimeoutError:
                        logger.warning("[TELEGRAM] Shutdown timed out after 8s")
                        future.cancel()
                    except Exception as e:
                        logger.error(f"[TELEGRAM] Shutdown error: {e}")
                else:
                    logger.info("[TELEGRAM] Bot already stopped")

            except Exception as e:
                logger.error(f"[TELEGRAM] Error initiating shutdown: {e}")

        # Wait for event loop thread to finish its cleanup
        if self.telegram_thread and self.telegram_thread.is_alive():
            logger.info("[TELEGRAM] Waiting for event loop thread...")
            self.telegram_thread.join(timeout=7)

            if self.telegram_thread.is_alive():
                logger.warning("[TELEGRAM] Thread still alive after 7s timeout")
            else:
                logger.info("[TELEGRAM] Thread terminated successfully")

        # Shutdown data manager
        self.data_manager.shutdown()

        logger.info("=" * 70)
        logger.info("[OK] Trading bot stopped successfully")
        logger.info("=" * 70)


def main():
    """Main entry point"""
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        with open("config/config.json", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("[FAIL] config/config.json not found!")
        sys.exit(1)

    # Check required models (FIXED: Include EMA models)
    required_models = []
    for asset_name, asset_config in config["assets"].items():
        if asset_config.get("enabled", False):
            strategies = asset_config.get("strategies", {})

            # Mean Reversion
            if strategies.get("mean_reversion", {}).get("enabled", False):
                required_models.append(
                    f"models/mean_reversion_{asset_name.lower()}.pkl"
                )

            # Trend Following
            if strategies.get("trend_following", {}).get("enabled", False):
                required_models.append(
                    f"models/trend_following_{asset_name.lower()}.pkl"
                )

            # EMA Strategy (FIXED: Check correct config key)
            if strategies.get("exponential_moving_averages", {}).get("enabled", False):
                required_models.append(f"models/ema_strategy_{asset_name.lower()}.pkl")

    missing_models = [m for m in required_models if not Path(m).exists()]
    if missing_models:
        print("=" * 70)
        print("[FAIL] REQUIRED MODELS NOT FOUND")
        print("=" * 70)
        for model in missing_models:
            print(f"  [X] {model}")
        print("\nPlease run: python train.py")
        print("=" * 70)
        sys.exit(1)

    try:
        bot = TradingBot()
        bot.start()
    except Exception as e:
        logger.error(f"[FATAL] Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
