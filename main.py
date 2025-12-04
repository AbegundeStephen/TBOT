#!/usr/bin/env python3
"""
Main Trading Bot - CORRECTED HANDLER INTEGRATION
: Proper initialization order, configuration structure, and async handling
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
import signal
from threading import Thread

#  Windows encoding BEFORE any imports that use logging
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
from src.execution.signal_aggregator import PerformanceWeightedAggregator
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.market_hours import MarketHours, should_trade_btc, should_trade_gold
from src.execution.auto_preset_selector import AutoPresetSelector

# Import Telegram bot
from src.telegram import TradingTelegramBot
from telegram_config import TELEGRAM_CONFIG


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
    """Main trading bot with CORRECTED handler integration"""

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT WITH HANDLER INTEGRATION")
        logger.info("=" * 70)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        setup_logging(self.config)

        # Initialize core components
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None

        # Handler instances (initialized later)
        self.binance_handler = None
        self.mt5_handler = None

        # Strategy storage
        self.strategies = {"BTC": {}, "GOLD": {}}

        # Trading state
        self.is_running = False
        self.trade_count_today = 0
        self.daily_loss = 0.0
        self.last_trade_date = None
        self.last_trade_times = {}
        self.last_market_status_log = None

        # Signal aggregators (initialized AFTER strategies AND exchanges)
        self.aggregators = {}
        self.selected_presets = {}

        # Telegram integration
        self.telegram_bot = None
        self.telegram_loop = None
        self.telegram_thread = None
        self._shutdown_requested = False

        # Initialize in correct order (but NOT aggregators yet)
        self._initialize_telegram()
        self._initialize_strategies()  #

    def initialize_exchanges(self):
        """
        Initialize exchange connections and handlers

        CORRECT ORDER:
        1. Connect to data sources (MT5, Binance)
        2. Initialize portfolio manager with connections
        3. Initialize execution handlers with portfolio manager
        """
        logger.info("\n" + "-" * 70)
        logger.info("STEP 1: Initializing Exchange Connections")
        logger.info("-" * 70)

        mt5_initialized = False

        # Connect to MT5 (for GOLD)
        if self.config["assets"]["GOLD"].get("enabled", False):
            try:
                if self.data_manager.initialize_mt5():
                    logger.info("[OK] MT5 connection established")
                    mt5_initialized = True
                else:
                    logger.error(
                        "[FAIL] Failed to initialize MT5 - GOLD trading disabled"
                    )
                    self.config["assets"]["GOLD"]["enabled"] = False
            except Exception as e:
                logger.error(f"[FAIL] MT5 initialization error: {e}")
                self.config["assets"]["GOLD"]["enabled"] = False

        # Connect to Binance (for BTC)
        if self.config["assets"]["BTC"].get("enabled", False):
            try:
                if self.data_manager.initialize_binance():
                    logger.info("[OK] Binance connection established")
                else:
                    logger.error(
                        "[FAIL] Failed to initialize Binance - BTC trading disabled"
                    )
                    self.config["assets"]["BTC"]["enabled"] = False
            except Exception as e:
                logger.error(f"[FAIL] Binance initialization error: {e}")
                self.config["assets"]["BTC"]["enabled"] = False

        # Initialize portfolio manager with exchange clients
        logger.info("\n" + "-" * 70)
        logger.info("STEP 2: Initializing Portfolio Manager")
        logger.info("-" * 70)

        try:
            import MetaTrader5 as mt5

            mt5_handler = mt5 if mt5_initialized else None
        except ImportError:
            mt5_handler = None
            logger.warning("[WARN] MetaTrader5 not available")

        try:
            self.portfolio_manager = PortfolioManager(
                config=self.config,
                mt5_handler=mt5_handler,
                binance_client=self.data_manager.binance_client,
            )
            logger.info(
                f"[OK] Portfolio Manager initialized (Mode: {self.portfolio_manager.mode.upper()})"
            )
            logger.info(f"     Capital: ${self.portfolio_manager.current_capital:,.2f}")
        except Exception as e:
            logger.error(f"[FAIL] Portfolio Manager initialization failed: {e}")
            raise

        # Initialize execution handlers with portfolio manager
        logger.info("\n" + "-" * 70)
        logger.info("STEP 3: Initializing Execution Handlers")
        logger.info("-" * 70)

        # Initialize Binance handler if both BTC is enabled AND we have a Binance client
        if (
            self.config["assets"]["BTC"].get("enabled", False)
            and self.data_manager.binance_client is not None
        ):
            try:
                self.binance_handler = BinanceExecutionHandler(
                    config=self.config,
                    client=self.data_manager.binance_client,
                    portfolio_manager=self.portfolio_manager,
                )
                logger.info("[OK] Binance Execution Handler initialized")
            except Exception as e:
                logger.error(f"[FAIL] Binance handler initialization: {e}")
                self.binance_handler = None
                self.config["assets"]["BTC"]["enabled"] = False

        # Initialize MT5 handler if both GOLD is enabled AND MT5 is initialized
        if self.config["assets"]["GOLD"].get("enabled", False) and mt5_initialized:
            try:
                self.mt5_handler = MT5ExecutionHandler(
                    config=self.config,
                    portfolio_manager=self.portfolio_manager,
                )
                logger.info("[OK] MT5 Execution Handler initialized")
            except Exception as e:
                logger.error(f"[FAIL] MT5 handler initialization: {e}")
                self.mt5_handler = None
                self.config["assets"]["GOLD"]["enabled"] = False

        # Verify at least one handler is available
        if not self.binance_handler and not self.mt5_handler:
            raise RuntimeError(
                "No execution handlers available! Check configuration and API credentials."
            )

    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            if not TELEGRAM_CONFIG.get("enabled", False):
                logger.info("[TELEGRAM] Disabled in config")
                return

            token = TELEGRAM_CONFIG.get("bot_token")
            admin_ids = TELEGRAM_CONFIG.get("admin_ids", [])

            if not token or not admin_ids:
                logger.warning(
                    f"[TELEGRAM] Missing config: token={bool(token)}, admins={bool(admin_ids)}"
                )
                return

            self.telegram_bot = TradingTelegramBot(
                token=token, admin_ids=admin_ids, trading_bot=self
            )
            logger.info(f"[TELEGRAM] Initialized for {len(admin_ids)} admin(s)")

        except Exception as e:
            logger.warning(f"[TELEGRAM] Initialization failed: {e}")

    def _initialize_strategies(self):
        """Initialize all three strategies for enabled assets (WITHOUT aggregators)"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Strategies (MR + TF + EMA)")
        logger.info("-" * 70)

        for asset_name, asset_config in self.config["assets"].items():
            if not asset_config.get("enabled", False):
                logger.debug(f"[SKIP] {asset_name}: Disabled")
                continue

            strategies_cfg = asset_config.get("strategies", {})
            strategy_cfgs = self.config.get("strategy_configs", {})

            # Mean Reversion
            if strategies_cfg.get("mean_reversion", {}).get("enabled", False):
                try:
                    cfg = strategy_cfgs.get("mean_reversion", {}).get(asset_name, {})
                    self.strategies[asset_name]["mean_reversion"] = (
                        MeanReversionStrategy(cfg)
                    )
                    logger.info(f"[OK] {asset_name}: Mean Reversion")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} Mean Reversion: {e}")

            # Trend Following
            if strategies_cfg.get("trend_following", {}).get("enabled", False):
                try:
                    cfg = strategy_cfgs.get("trend_following", {}).get(asset_name, {})
                    self.strategies[asset_name]["trend_following"] = (
                        TrendFollowingStrategy(cfg)
                    )
                    logger.info(f"[OK] {asset_name}: Trend Following")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} Trend Following: {e}")

            # EMA Strategy
            if strategies_cfg.get("exponential_moving_averages", {}).get(
                "enabled", False
            ):
                try:
                    cfg = strategy_cfgs.get("exponential_moving_averages", {}).get(
                        asset_name, {}
                    )
                    self.strategies[asset_name]["ema_strategy"] = EMAStrategy(cfg)
                    logger.info(f"[OK] {asset_name}: EMA Strategy")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} EMA Strategy: {e}")

            enabled = len(self.strategies[asset_name])
            if enabled == 0:
                logger.warning(f"[!] {asset_name}: NO strategies enabled")
            else:
                strat_names = ", ".join(self.strategies[asset_name].keys())
                logger.info(
                    f"[OK] {asset_name}: {enabled}/3 strategies → {strat_names}"
                )

    def _initialize_aggregators(self):
        """Initialize signal aggregators with AUTO-PRESET support"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Signal Aggregators")
        logger.info("-" * 70)

        AGGREGATOR_PRESETS = {
            "BTC": {
                "conservative": {
                    "buy_threshold": 0.35,
                    "sell_threshold": 0.40,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.20,
                    # UPDATED: Stronger regime bias (was 0.03-0.05)
                    "bull_buy_boost": 0.10,  # Increased from 0.03
                    "bull_sell_penalty": 0.12,  # Increased from 0.05
                    "bear_sell_boost": 0.10,  # Increased from 0.03
                    "bear_buy_penalty": 0.12,  # Increased from 0.05
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.32,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.30,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.22,
                    # UPDATED: Stronger regime bias (was 0.035-0.04)
                    "bull_buy_boost": 0.11,  # Increased from 0.035
                    "bull_sell_penalty": 0.11,  # Increased from 0.04
                    "bear_sell_boost": 0.11,  # Increased from 0.035
                    "bear_buy_penalty": 0.11,  # Increased from 0.04
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.17,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.25,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.25,
                    # UPDATED: Stronger regime bias (was 0.03-0.04)
                    "bull_buy_boost": 0.12,  # Increased from 0.04
                    "bull_sell_penalty": 0.12,  # Increased from 0.03
                    "bear_sell_boost": 0.12,  # Increased from 0.04
                    "bear_buy_penalty": 0.12,  # Increased from 0.03
                    "min_confidence_to_use": 0.09,
                    "min_signal_quality": 0.25,
                    "hold_contribution_pct": 0.18,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "verbose": False,
                },
                "scalper": {
                    "buy_threshold": 0.23,
                    "sell_threshold": 0.20,
                    "two_strategy_bonus": 0.25,
                    "three_strategy_bonus": 0.30,
                    # UPDATED: Strongest regime bias for scalping (was 0.02-0.05)
                    "bull_buy_boost": 0.15,  # Increased from 0.05
                    "bull_sell_penalty": 0.10,  # Increased from 0.02
                    "bear_sell_boost": 0.15,  # Increased from 0.05
                    "bear_buy_penalty": 0.10,  # Increased from 0.02
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.20,
                    "hold_contribution_pct": 0.20,
                    "allow_single_override": True,
                    "single_override_threshold": 0.65,
                    "verbose": False,
                },
            },
            "GOLD": {
                "conservative": {
                    "buy_threshold": 0.38,
                    "sell_threshold": 0.42,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.25,
                    # UPDATED: Stronger regime bias (was 0.02-0.03)
                    "bull_buy_boost": 0.07,  # Increased from 0.02
                    "bull_sell_penalty": 0.09,  # Increased from 0.03
                    "bear_sell_boost": 0.07,  # Increased from 0.02
                    "bear_buy_penalty": 0.09,  # Increased from 0.03
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.30,
                    "hold_contribution_pct": 0.12,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.33,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.25,
                    # UPDATED: Stronger regime bias (was 0.02)
                    "bull_buy_boost": 0.08,  # Increased from 0.02
                    "bull_sell_penalty": 0.08,  # Increased from 0.02
                    "bear_sell_boost": 0.08,  # Increased from 0.02
                    "bear_buy_penalty": 0.08,  # Increased from 0.02
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.14,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.28,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.30,
                    # UPDATED: Stronger regime bias (was 0.01-0.03)
                    "bull_buy_boost": 0.09,  # Increased from 0.03
                    "bull_sell_penalty": 0.09,  # Increased from 0.01
                    "bear_sell_boost": 0.09,  # Increased from 0.03
                    "bear_buy_penalty": 0.09,  # Increased from 0.01
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.22,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "verbose": False,
                },
                "scalper": {
                "buy_threshold": 0.24,
                "sell_threshold": 0.30,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.35,
                "bull_buy_boost": 0.12,
                "bull_sell_penalty": 0.08,
                "bear_sell_boost": 0.12,
                "bear_buy_penalty": 0.08,
                "min_confidence_to_use": 0.06,
                "min_signal_quality": 0.18,
                "hold_contribution_pct": 0.18,
                "allow_single_override": True,
                "single_override_threshold": 0.65,
                "verbose": False,
                # ✨ NEW: Quality margin to filter weak signals
                "min_quality_margin": 0.06,  # ← ADD THIS LINE
            },
            },
        }

        # ============================================================
        # AUTO-PRESET SELECTION LOGIC
        # ============================================================
        aggregator_cfg = self.config.get("aggregator_settings", {})
        preset = aggregator_cfg.get("preset", "auto")

        if preset == "auto":
            logger.info("\n" + "=" * 70)
            logger.info("🤖 AUTO-PRESET MODE ENABLED")
            logger.info("=" * 70)
            logger.info("Analyzing market conditions to select optimal presets...")
            logger.info("")

            # Initialize auto-selector
            selector = AutoPresetSelector(self.data_manager, self.config)

            # Get optimal preset for each enabled asset
            asset_presets = selector.get_preset_for_all_assets()

            logger.info("\n" + "=" * 70)
            logger.info("📊 AUTO-PRESET RESULTS")
            logger.info("=" * 70)
            for asset, selected_preset in asset_presets.items():
                logger.info(f"  {asset:6} → {selected_preset.upper()}")
            logger.info("=" * 70 + "\n")

        elif preset not in ["conservative", "balanced", "aggressive", "scalper"]:
            logger.warning(f"Unknown preset '{preset}', using 'balanced'")
            asset_presets = {name: "balanced" for name in self.strategies.keys()}
        else:
            # Manual preset specified - use for all assets
            logger.info(f"Using manual preset: {preset.upper()}")
            asset_presets = {name: preset for name in self.strategies.keys()}

        self.selected_presets = asset_presets.copy()
        # ============================================================
        # CREATE AGGREGATORS WITH SELECTED PRESETS
        # ============================================================
        for asset_name, strategies in self.strategies.items():
            if not self.config["assets"][asset_name].get("enabled", False):
                continue

            strategy_count = len(strategies)
            if strategy_count == 0:
                logger.warning(f"[!] {asset_name}: No strategies, skipping aggregator")
                continue

            # Get the preset for this specific asset
            selected_preset = asset_presets.get(asset_name, "balanced")

            # Get configuration for the selected preset
            config_for_aggregator = AGGREGATOR_PRESETS[asset_name][selected_preset]

            try:
                self.aggregators[asset_name] = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_name,
                    config=config_for_aggregator,
                )
                logger.info(
                    f"[OK] {asset_name}: Aggregator ({strategy_count} strategies, {selected_preset})"
                )
            except Exception as e:
                logger.error(f"[FAIL] {asset_name} aggregator: {e}")

    def load_models(self):
        """Load trained ML models for all strategies"""
        logger.info("\n" + "-" * 70)
        logger.info("Loading Trained Models")
        logger.info("-" * 70)

        loaded = 0
        expected = 0

        for asset_name, strategies in self.strategies.items():
            for strategy_name, strategy in strategies.items():
                expected += 1

                model_filename = f"{strategy_name}_{asset_name.lower()}.pkl"
                model_path = f"models/{model_filename}"

                if Path(model_path).exists():
                    try:
                        if strategy.load_model(model_path):
                            logger.info(f"[OK] {model_path}")
                            loaded += 1
                        else:
                            logger.error(f"[FAIL] {model_path} (load returned False)")
                    except Exception as e:
                        logger.error(f"[FAIL] {model_path}: {e}")
                else:
                    logger.error(f"[FAIL] Not found: {model_path}")

        if loaded == 0:
            logger.error("=" * 70)
            logger.error("NO MODELS LOADED! Run: python train.py")
            logger.error("=" * 70)
            sys.exit(1)

        logger.info(f"\n[OK] Loaded {loaded}/{expected} models")

        # NOW initialize aggregators after models are loaded and exchanges connected
        self._initialize_aggregators()

    def reset_daily_counters(self):
        """Reset daily trading counters"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trade_count_today = 0
            self.daily_loss = 0.0
            self.last_trade_date = current_date
            logger.info(f"[RESET] Daily counters reset for {current_date}")
            self.portfolio_manager.start_trading_session()
            logger.info(f"[SESSION] Trading session started")
            if not self.portfolio_manager.is_paper_mode:
                logger.info("[REFRESH] Fetching updated capital...")
                self.portfolio_manager.refresh_capital()
                self.portfolio_manager.reset_daily_pnl()

            # Send daily summary via Telegram
            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.send_daily_summary(), self.telegram_loop
                    )
                except Exception as e:
                    logger.debug(f"[TELEGRAM] Daily summary error: {e}")

    def check_trading_limits(self) -> bool:
        """Check if trading limits are reached"""
        risk_cfg = self.config.get("risk_management", {})

        max_daily_trades = risk_cfg.get("max_daily_trades", 10)
        if self.trade_count_today >= max_daily_trades:
            logger.warning(
                f"[LIMIT] Daily trades ({self.trade_count_today}/{max_daily_trades})"
            )
            return False

        max_daily_loss = risk_cfg.get("max_daily_loss_pct", 0.05)
        if self.daily_loss >= max_daily_loss:
            logger.warning(f"[LIMIT] Daily loss ({self.daily_loss:.2%})")
            return False

        circuit_breaker = risk_cfg.get("circuit_breaker_loss_pct", 0.10)
        loss_pct = (
            self.daily_loss / self.portfolio_manager.initial_capital
            if self.portfolio_manager.initial_capital > 0
            else 0
        )
        if loss_pct >= circuit_breaker:
            logger.error(f"[BREAKER] CIRCUIT BREAKER! Loss: {loss_pct:.2%}")
            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.notify_error(
                            f"🚨 CIRCUIT BREAKER!\nLoss: {loss_pct:.2%}"
                        ),
                        self.telegram_loop,
                    )
                except:
                    pass
            return False

        trading_cfg = self.config.get("trading", {})
        if trading_cfg.get("allow_simultaneous_positions", True):
            max_positions = trading_cfg.get("max_simultaneous_positions", 2)
            current = self.portfolio_manager.get_open_positions_count()
            if current >= max_positions:
                logger.info(f"[LIMIT] Max positions ({current}/{max_positions})")
                return False

        return True

    def check_min_time_between_trades(self, asset_name: str) -> bool:
        """Check minimum time between trades"""
        min_minutes = self.config["trading"].get("min_time_between_trades_minutes", 60)

        if asset_name in self.last_trade_times:
            elapsed = datetime.now() - self.last_trade_times[asset_name]
            if elapsed.total_seconds() < min_minutes * 60:
                remaining = min_minutes - (elapsed.total_seconds() / 60)
                logger.debug(f"[COOLDOWN] {asset_name}: {remaining:.0f}min remaining")
                return False

        return True

    def check_market_hours(self, asset_name: str) -> bool:
        """Check if market is open for the asset"""
        if asset_name == "BTC":
            return True

        elif asset_name == "GOLD":
            is_open = should_trade_gold()

            if not is_open:
                status, message = MarketHours.get_market_status("gold")

                current_hour = datetime.now().hour
                if self.last_market_status_log != current_hour:
                    logger.info(f"[MARKET] {asset_name}: {message}")
                    self.last_market_status_log = current_hour

            return is_open

        return True

    def trade_asset(self, asset_name: str):
        """Execute trading logic for a single asset"""
        asset_cfg = self.config["assets"][asset_name]
        if not asset_cfg.get("enabled", False):
            return

        if not self.check_market_hours(asset_name):
            logger.debug(f"[SKIP] {asset_name}: Market closed")
            return

        exchange = asset_cfg.get("exchange", "binance")
        symbol = asset_cfg.get("symbol", "BTCUSDT")

        # Verify handler is available
        handler = self.binance_handler if exchange == "binance" else self.mt5_handler
        if not handler:
            logger.warning(f"[!] {asset_name}: {exchange.upper()} handler unavailable")
            return

        try:
            logger.info(f"\n{'-' * 70}")
            logger.info(f"Trading {asset_name}")
            logger.info(f"{'-' * 70}")

            # Show market info
            if asset_name == "GOLD":
                status, msg = MarketHours.get_market_status("gold")
                logger.info(f"[MARKET] {msg}")
            else:
                logger.info(f"[MARKET] {asset_name}: 24/7 (Cryptocurrency)")

            # Fetch data
            end_time = datetime.now(timezone.utc)
            if exchange == "binance":
                interval = asset_cfg.get("interval", "1h")
                lookback = 15 if interval == "1h" else 60
            else:
                timeframe = asset_cfg.get("timeframe", "H1")
                lookback = 25 if timeframe == "H1" else 75

            start_time = end_time - timedelta(days=lookback)
            logger.info(f"[DATA] Fetching {lookback} days...")

            if exchange == "binance":
                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=asset_cfg.get("interval", "1h"),
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=asset_cfg.get("timeframe", "H1"),
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = self.data_manager.clean_data(df)
            min_bars = 250
            if len(df) < min_bars:
                logger.warning(
                    f"[!] {asset_name}: Insufficient data ({len(df)}/{min_bars})"
                )
                return

            logger.info(f"[OK] {len(df)} bars fetched")

            # Get current price
            try:
                if exchange == "binance":
                    current_price = handler.get_current_price(symbol)
                else:  # MT5
                    current_price = handler.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"Failed to get current price: {e}, using last close")
                current_price = df["close"].iloc[-1]

            logger.info(f"[PRICE] {asset_name}: ${current_price:,.2f}")

            # Get aggregated signal
            aggregator = self.aggregators.get(asset_name)
            logger.info(f"[AGGREGATOR] {aggregator}")
            if not aggregator:
                logger.warning(f"[!] {asset_name}: No aggregator configured")
                return

            signal, details = aggregator.get_aggregated_signal(df)
            self.telegram_bot.signal_monitor.record_signal(
                asset=asset_name,
                signal=signal,
                details=details,
                price=current_price,
                timestamp=datetime.now(),
            )

            # Log all three strategy signals
            # Log all three strategy signals
            logger.info(f"\n[SIGNAL] Strategy Analysis:")
            logger.info(
                f"  Mean Reversion:   {details.get('mr_signal', 0):>2} "
                f"(confidence: {details.get('mr_confidence', 0):.3f})"
            )
            logger.info(
                f"  Trend Following:  {details.get('tf_signal', 0):>2} "
                f"(confidence: {details.get('tf_confidence', 0):.3f})"
            )
            logger.info(
                f"  EMA Regime:       {details.get('regime', 'N/A'):>2} "
                f"(confidence: {details.get('regime_confidence', 0):.3f})"
            )
            logger.info(f"\n[AGGREGATED] Signal: {signal:>2}")
            logger.info(f"[QUALITY] Score: {details.get('signal_quality', 0):.3f}")
            logger.info(f"[REASONING] {details.get('reasoning', 'N/A')}")

            # Skip if signal is HOLD (0)
            if signal == 0:
                logger.info(f"[HOLD] {asset_name}: No action (HOLD signal)")
                return

            # Check for open position before selling
            existing_pos = self.portfolio_manager.get_position(asset_name)
            if signal == -1 and not existing_pos:
                logger.warning(f"[SKIP] {asset_name}: No open position to sell")
                return

            # Check trading limits and cooldowns before proceeding
            if not self.check_trading_limits():
                logger.info(
                    f"[LIMIT] {asset_name}: Trading limits prevent new position"
                )
                return

            if not self.check_min_time_between_trades(asset_name):
                logger.info(f"[COOLDOWN] {asset_name}: Cooldown period active")
                return

            # Execute signal with appropriate handler
            success = False
            if exchange == "binance":
                success = self.binance_handler.execute_signal(
                    signal=signal,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=details.get("signal_quality", 0.5),
                    market_condition=(
                        "bull" if details.get("regime") == "🚀 BULL" else "bear"
                    ),
                )
                self.binance_handler.check_and_update_positions(asset_name)
            else:
                success = self.mt5_handler.execute_signal(
                    signal=signal,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=details.get("signal_quality", 0.5),
                    market_condition=(
                        "bull" if details.get("regime") == "🚀 BULL" else "bear"
                    ),
                )
                self.mt5_handler.check_and_update_positions(asset_name)

            # Only log success if the trade was actually executed
            if success:
                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()
                signal_type = "BUY" if signal == 1 else "SELL"
                logger.info(
                    f"[SUCCESS] {asset_name} {signal_type} executed (Daily count: {self.trade_count_today})"
                )

                # Check if new position was opened
                new_pos = self.portfolio_manager.get_position(asset_name)
                if new_pos and not existing_pos and signal != 0:
                    try:
                        pos = new_pos
                        self._send_telegram_notification(
                            self.telegram_bot.notify_trade_opened(
                                asset=asset_name,
                                side=(
                                    pos.side
                                    if hasattr(pos, "side")
                                    else pos.get("side")
                                ),
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
                if existing_pos and not new_pos:
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

                # Log trade
                if self.config.get("logging", {}).get("save_trades", True):
                    self._log_trade(asset_name, signal, details, current_price)

            else:
                logger.warning(
                    f"[SKIP] {asset_name}: Trade not executed (limits/cooldowns/handler failure)"
                )

        except Exception as e:
            logger.error(f"[ERROR] {asset_name} trading error: {e}", exc_info=True)
            if self.telegram_bot and self.telegram_loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.notify_error(
                            f"Error in {asset_name}:\n{str(e)[:200]}"
                        ),
                        self.telegram_loop,
                    )
                except:
                    pass

    def _log_trade(self, asset: str, signal: int, details: dict, price: float):
        """Log trade to file"""
        try:
            trade_log = Path("logs/trades.log")
            trade_log.parent.mkdir(exist_ok=True)

            with open(trade_log, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()},{asset},{signal},{price:.2f},"
                    f"{details.get('signal_quality', 0):.3f},"
                    f"{details.get('reasoning', 'N/A')}\n"
                )
        except Exception as e:
            logger.debug(f"Trade log error: {e}")

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("\n" + "=" * 70)
        logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

        # Refresh capital if live
        if not self.portfolio_manager.is_paper_mode:
            self.portfolio_manager.refresh_capital()

        self.reset_daily_counters()

        # ✅ FIX: Get current prices for all assets
        current_prices = {}
        enabled = [
            name
            for name, cfg in self.config["assets"].items()
            if cfg.get("enabled", False)
        ]

        for asset_name in enabled:
            try:
                asset_cfg = self.config["assets"][asset_name]
                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.binance_handler if exchange == "binance" else self.mt5_handler
                )

                if handler:
                    current_prices[asset_name] = handler.get_current_price()
            except Exception as e:
                logger.debug(f"Failed to get {asset_name} price: {e}")

        # ✅ FIX: Update positions with current prices (this will fetch MT5 profit)
        self.portfolio_manager.update_positions(current_prices)

        logger.info(f"[ASSETS] Enabled: {', '.join(enabled)}")

        for asset_name in enabled:
            self.trade_asset(asset_name)
            time.sleep(2)

        # ✅ FIX: Get portfolio status with current prices
        status = self.portfolio_manager.get_portfolio_status(current_prices)

        logger.info(f"\n{'-' * 70}")
        logger.info("[PORTFOLIO STATUS]")
        logger.info(f"{'-' * 70}")
        logger.info(f"Mode:           {status.get('mode', 'N/A').upper()}")
        logger.info(f"Total Value:    ${status.get('total_value', 0):,.2f}")
        logger.info(f"Cash:           ${status.get('cash', 0):,.2f}")
        logger.info(
            f"Exposure:       ${status.get('total_exposure', 0):,.2f} ({status.get('exposure_pct', 0):.1%})"
        )
        logger.info(f"Open Positions: {status.get('open_positions', 0)}")
        logger.info(f"Total Trades:   {status.get('total_trades', 0)}")

        # ✅ FIX: Display daily P&L correctly
        daily_pnl = status.get("daily_pnl", 0)
        daily_pnl_color = "+" if daily_pnl >= 0 else ""
        logger.info(f"Daily P&L:      {daily_pnl_color}${daily_pnl:,.2f}")

        # ✅ FIX: Display realized P&L today
        realized_pnl = status.get("realized_pnl_today", 0)
        realized_color = "+" if realized_pnl >= 0 else ""
        logger.info(f"Realized P&L:   {realized_color}${realized_pnl:,.2f}")

        # ✅ FIX: Show individual position P&L
        positions = status.get("positions", {})
        if positions:
            logger.info(f"\n{'-' * 70}")
            logger.info("[OPEN POSITIONS]")
            logger.info(f"{'-' * 70}")

            for asset, pos_data in positions.items():
                side = pos_data.get("side", "N/A").upper()
                entry = pos_data.get("entry_price", 0)
                current = pos_data.get("current_price", 0)
                pnl = pos_data.get("pnl", 0)
                pnl_pct = pos_data.get("pnl_pct", 0) * 100

                pnl_color = "+" if pnl >= 0 else ""

                logger.info(f"{asset} {side}:")
                logger.info(f"  Entry:   ${entry:,.2f}")
                logger.info(f"  Current: ${current:,.2f}")
                logger.info(
                    f"  P&L:     {pnl_color}${pnl:,.2f} ({pnl_color}{pnl_pct:.2f}%)"
                )

                # ✅ FIX: Show MT5 profit if available
                if pos_data.get("mt5_ticket"):
                    mt5_profit = pos_data.get("mt5_profit", 0)
                    logger.info(
                        f"  MT5 P&L: ${mt5_profit:,.2f} (Ticket: {pos_data['mt5_ticket']})"
                    )

                logger.info("")

        logger.info(f"{'-' * 70}")
        logger.info("[OK] Trading cycle complete")
        logger.info("=" * 70)

    def log_detailed_pnl_report(self):
        """Log detailed P&L report for monitoring"""
        try:
            # Get current prices
            current_prices = {}
            for asset_name, asset_cfg in self.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.binance_handler if exchange == "binance" else self.mt5_handler
                )

                if handler:
                    try:
                        current_prices[asset_name] = handler.get_current_price()
                    except:
                        pass

            # Update positions with MT5 profit
            self.portfolio_manager.update_positions(current_prices)

            # Get status
            status = self.portfolio_manager.get_portfolio_status(current_prices)

            logger.info("\n" + "=" * 70)
            logger.info("DETAILED P&L REPORT")
            logger.info("=" * 70)
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Mode: {status['mode'].upper()}")
            logger.info("")

            # Capital
            logger.info("[CAPITAL]")
            logger.info(
                f"  Initial:     ${self.portfolio_manager.initial_capital:,.2f}"
            )
            logger.info(f"  Current:     ${status['capital']:,.2f}")
            logger.info(f"  Total Value: ${status['total_value']:,.2f}")
            logger.info("")

            # P&L Breakdown
            logger.info("[P&L BREAKDOWN]")
            logger.info(f"  Daily P&L:      ${status['daily_pnl']:+,.2f}")
            logger.info(f"  Realized Today: ${status['realized_pnl_today']:+,.2f}")
            logger.info(f"  Unrealized:     ${status['total_unrealized_pnl']:+,.2f}")
            logger.info("")

            # Positions
            positions = status.get("positions", {})
            if positions:
                logger.info("[POSITION DETAILS]")
                for asset, pos in positions.items():
                    logger.info(f"  {asset} {pos['side'].upper()}:")
                    logger.info(f"    Entry:        ${pos['entry_price']:,.2f}")
                    logger.info(f"    Current:      ${pos['current_price']:,.2f}")
                    logger.info(f"    Quantity:     {pos['quantity']:.6f}")
                    logger.info(f"    Value:        ${pos['current_value']:,.2f}")
                    logger.info(
                        f"    P&L:          ${pos['pnl']:+,.2f} ({pos['pnl_pct']*100:+.2f}%)"
                    )

                    if pos.get("mt5_ticket"):
                        logger.info(f"    MT5 Ticket:   {pos['mt5_ticket']}")
                        logger.info(f"    MT5 Profit:   ${pos['mt5_profit']:+,.2f}")

                    logger.info("")
            else:
                logger.info("[NO OPEN POSITIONS]")
                logger.info("")

            # Risk Metrics
            logger.info("[RISK METRICS]")
            logger.info(
                f"  Exposure:     ${status['total_exposure']:,.2f} ({status['exposure_pct']:.1%})"
            )
            logger.info(f"  Drawdown:     {status['drawdown']:.2%}")
            logger.info(f"  Open Pos:     {status['open_positions']}")
            logger.info(f"  Total Trades: {status['total_trades']}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error generating P&L report: {e}", exc_info=True)

    async def _start_telegram_bot(self):
        """Start Telegram bot"""
        try:
            logger.info("[TELEGRAM] Initializing...")
            await self.telegram_bot.initialize()
            logger.info("[TELEGRAM] Ready")
            await self.telegram_bot.run_polling()
        except asyncio.CancelledError:
            logger.info("[TELEGRAM] Cancelled")
        except Exception as e:
            logger.error(f"[TELEGRAM] Error: {e}", exc_info=True)

    def _run_telegram_loop(self):
        """Run Telegram event loop"""
        self.telegram_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.telegram_loop)

        try:
            logger.info("[TELEGRAM] Event loop starting...")
            self.telegram_loop.run_until_complete(self._start_telegram_bot())
        except Exception as e:
            logger.error(f"[TELEGRAM] Loop error: {e}", exc_info=True)
        finally:
            logger.info("[TELEGRAM] Cleaning up event loop...")

            try:
                pending = asyncio.all_tasks(self.telegram_loop)
                if pending:
                    logger.info(f"[TELEGRAM] Cancelling {len(pending)} tasks...")
                    for task in pending:
                        task.cancel()

                    try:
                        self.telegram_loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=3.0,
                            )
                        )
                    except (asyncio.TimeoutError, Exception):
                        pass
            except Exception as e:
                logger.debug(f"[TELEGRAM] Cleanup error: {e}")
            finally:
                if not self.telegram_loop.is_closed():
                    self.telegram_loop.close()
                    logger.info("[TELEGRAM] Event loop closed")

    def _send_telegram_notification(self, coro):
        """Send Telegram notification from main thread"""
        if not self.telegram_bot or not self.telegram_loop:
            return

        if not self.telegram_bot._is_ready:
            logger.debug("[TELEGRAM] Bot not ready")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.telegram_loop)
            future.result(timeout=5)
        except TimeoutError:
            logger.debug("[TELEGRAM] Notification timeout")
        except Exception as e:
            logger.debug(f"[TELEGRAM] Notification error: {e}")

    def start(self):
        """Start the trading bot"""
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT INITIALIZING")
        logger.info("=" * 70)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {self.config['trading']['mode'].upper()}")
        logger.info("Strategies: Mean Reversion + Trend Following + EMA Crossover")

        try:
            # Initialize exchanges and handlers
            self.initialize_exchanges()

            # Load trained models
            self.load_models()

            logger.info("\n[MARKET HOURS]")
            logger.info("  BTC (Crypto):  24/7 - Always open")
            logger.info("  GOLD (Forex):  Sun 22:00 UTC - Fri 22:00 UTC")
            logger.info("=" * 70)

            # Verify handlers
            if not self.binance_handler and not self.mt5_handler:
                raise RuntimeError("No execution handlers available!")

            # Start Telegram bot if enabled
            if self.telegram_bot:
                logger.info("\n[TELEGRAM] Starting bot in background thread...")
                self.telegram_thread = Thread(
                    target=self._run_telegram_loop, daemon=True
                )
                self.telegram_thread.start()

                # Wait for bot ready
                logger.info("[TELEGRAM] Waiting for bot to be ready...")
                timeout = 30
                start = time.time()

                while (
                    not self.telegram_bot._is_ready and (time.time() - start) < timeout
                ):
                    time.sleep(0.5)

                if self.telegram_bot._is_ready:
                    logger.info("[TELEGRAM] ✅ Bot ready for notifications")
                else:
                    logger.warning("[TELEGRAM] ⚠️ Bot initialization timeout")

            # Setup signal handlers
            def signal_handler(signum, frame):
                logger.info(f"\n[!] Signal {signum} received, shutting down...")
                self.stop()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Schedule trading cycles
            check_interval = self.config["trading"].get("check_interval_seconds", 300)
            schedule.every(check_interval).seconds.do(self.run_trading_cycle)
            schedule.every(1).hours.do(self.log_detailed_pnl_report)

            self.is_running = True
            logger.info(f"\n[OK] Trading bot running")
            logger.info(
                f"[TIME] Cycle interval: {check_interval}s ({check_interval / 60:.1f}min)"
            )
            logger.info(f"Press Ctrl+C to stop\n")

            # Run initial cycle
            self.run_trading_cycle()

            # Main loop
            while self.is_running and not self._shutdown_requested:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n[!] KeyboardInterrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"[FATAL] Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)

    def stop(self):
        """Stop the trading bot gracefully"""
        if self._shutdown_requested:
            logger.info("[STOP] Shutdown already in progress")
            return

        logger.info("\n" + "=" * 70)
        logger.info("[STOP] SHUTTING DOWN TRADING BOT")
        logger.info("=" * 70)

        self._shutdown_requested = True
        self.is_running = False

        # Close positions if configured
        if self.config["trading"].get("close_positions_on_shutdown", False):
            logger.info("[STOP] Closing open positions...")
            try:
                self.portfolio_manager.close_all_positions()
                logger.info("[STOP] ✅ Positions closed")
            except Exception as e:
                logger.error(f"[STOP] Error closing positions: {e}")

        # Shutdown Telegram bot
        if self.telegram_bot and self.telegram_loop:
            try:
                logger.info("[TELEGRAM] Initiating bot shutdown...")

                if self.telegram_bot.is_running:
                    future = asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.shutdown(), self.telegram_loop
                    )

                    try:
                        future.result(timeout=10)
                        logger.info("[TELEGRAM] ✅ Bot shutdown complete")
                    except TimeoutError:
                        logger.warning("[TELEGRAM] ⚠️ Shutdown timeout")
                        future.cancel()
                    except Exception as e:
                        logger.error(f"[TELEGRAM] Shutdown error: {e}")

            except Exception as e:
                logger.error(f"[TELEGRAM] Error during shutdown: {e}")

        # Wait for event loop thread
        if self.telegram_thread and self.telegram_thread.is_alive():
            logger.info("[TELEGRAM] Waiting for event loop thread...")
            self.telegram_thread.join(timeout=5)

            if self.telegram_thread.is_alive():
                logger.warning("[TELEGRAM] ⚠️ Thread still alive")
            else:
                logger.info("[TELEGRAM] ✅ Thread terminated")

        # Shutdown data manager
        try:
            self.data_manager.shutdown()
            logger.info("[STOP] ✅ Data manager shutdown")
        except Exception as e:
            logger.error(f"[STOP] Data manager error: {e}")

        logger.info("=" * 70)
        logger.info("[OK] Trading bot stopped")
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

    # Check required models exist
    required_models = []
    for asset_name, asset_cfg in config["assets"].items():
        if asset_cfg.get("enabled", False):
            strategies = asset_cfg.get("strategies", {})

            if strategies.get("mean_reversion", {}).get("enabled", False):
                required_models.append(
                    f"models/mean_reversion_{asset_name.lower()}.pkl"
                )

            if strategies.get("trend_following", {}).get("enabled", False):
                required_models.append(
                    f"models/trend_following_{asset_name.lower()}.pkl"
                )

            if strategies.get("exponential_moving_averages", {}).get("enabled", False):
                required_models.append(f"models/ema_strategy_{asset_name.lower()}.pkl")

    missing = [m for m in required_models if not Path(m).exists()]
    if missing:
        print("=" * 70)
        print("[FAIL] REQUIRED MODELS NOT FOUND")
        print("=" * 70)
        for model in missing:
            print(f"  [X] {model}")
        print("\nRun: python train.py")
        print("=" * 70)
        sys.exit(1)

    try:
        bot = TradingBot()
        bot.start()
    except KeyboardInterrupt:
        logger.info("\n[!] KeyboardInterrupt")
    except Exception as e:
        logger.error(f"[FATAL] {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
