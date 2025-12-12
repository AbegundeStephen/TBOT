#!/usr/bin/env python3
"""
Main Trading Bot - IMPROVED STABILITY VERSION
Enhanced error handling, network resilience, and Telegram thread management
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
from threading import Thread, Event
from typing import Optional
from types import SimpleNamespace
from src.execution.dynamic_trade_manager import DynamicTradeManager

# Windows encoding fix
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
from src.ai import (
    DynamicAnalyst,
    OHLCSniper,
    HybridSignalValidator,
    AIValidatorMonitor,
    AIValidatorTuner,
)
import pickle

# Import Telegram bot
from src.telegram import TradingTelegramBot
from telegram_config import TELEGRAM_CONFIG


def setup_logging(config):
    """Setup logging with proper encoding and rotation"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trading_bot.log")

    Path(log_file).parent.mkdir(exist_ok=True)

    # ✨ IMPROVED: Add log rotation
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_file, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
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
    """Main trading bot with improved stability and error recovery"""

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT WITH ENHANCED STABILITY")
        logger.info("=" * 70)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        setup_logging(self.config)

        self.params = SimpleNamespace(
            use_ai_validation=True,
            ai_sr_threshold=0.020,
            ai_pattern_confidence=0.45,
            ai_enable_adaptive=True,
            ai_strong_signal_bypass=0.70,
        )

        # Initialize core components
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None

        # Handler instances
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

        # Signal aggregators
        self.aggregators = {}
        self.selected_presets = {}

        # ✨ NEW: Telegram thread management
        self.telegram_bot = None
        self.telegram_loop = None
        self.telegram_thread = None
        self._telegram_ready = Event()
        self._telegram_should_stop = Event()
        self._telegram_error_count = 0
        self._max_telegram_restarts = 3
        self._telegram_last_restart = None

        # ✨ NEW: Main bot state
        self._shutdown_requested = False
        self._main_loop_running = False
        self._last_successful_cycle = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Initialize components
        self._initialize_telegram()
        self._initialize_strategies()

    def initialize_exchanges(self):
        """Initialize exchange connections and handlers"""
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

        # Initialize portfolio manager
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

        # Initialize execution handlers
        logger.info("\n" + "-" * 70)
        logger.info("STEP 3: Initializing Execution Handlers")
        logger.info("-" * 70)

        if (
            self.config["assets"]["BTC"].get("enabled", False)
            and self.data_manager.binance_client is not None
        ):
            try:
                # ✨ FIXED: Pass data_manager to BinanceExecutionHandler
                self.binance_handler = BinanceExecutionHandler(
                    config=self.config,
                    client=self.data_manager.binance_client,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,  # ✨ NEW: Add this line
                )
                logger.info("[OK] Binance Execution Handler initialized")
            except Exception as e:
                logger.error(f"[FAIL] Binance handler initialization: {e}")
                self.binance_handler = None
                self.config["assets"]["BTC"]["enabled"] = False

        if self.config["assets"]["GOLD"].get("enabled", False) and mt5_initialized:
            try:
                # ✨ FIXED: Pass data_manager to MT5ExecutionHandler too
                self.mt5_handler = MT5ExecutionHandler(
                    config=self.config,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,  # ✨ NEW: Add this line
                )
                logger.info("[OK] MT5 Execution Handler initialized")
            except Exception as e:
                logger.error(f"[FAIL] MT5 handler initialization: {e}")
                self.mt5_handler = None
                self.config["assets"]["GOLD"]["enabled"] = False

        if not self.binance_handler and not self.mt5_handler:
            raise RuntimeError(
                "No execution handlers available! Check configuration and API credentials."
            )

    def _initialize_telegram(self):
        """Initialize Telegram bot with error handling"""
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
            self.telegram_bot = None

    def _initialize_strategies(self):
        """Initialize all strategies"""
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

    def initialize_ai_layer(self):
        """
        Initialize AI validation layer
        Add this method to your TradingBot class
        """
        try:
            logger.info("=" * 70)
            logger.info("Initializing AI Layer...")
            logger.info("=" * 70)

            models_dir = Path("models/ai")

            # Check if model files exist
            model_path = models_dir / "sniper_btc_gold_v2.weights.h5"
            mapping_path = models_dir / "sniper_btc_gold_v2_mapping.pkl"
            config_path = models_dir / "sniper_btc_gold_v2_config.pkl"

            if not model_path.exists():
                logger.error(f"[ERROR] Model not found: {model_path}")
                logger.error("Please run: python train_ai_layer.py")
                self.ai_validator = None
                return

            # Load pattern mapping
            with open(mapping_path, "rb") as f:
                pattern_map = pickle.load(f)

            # Load training config
            with open(config_path, "rb") as f:
                config = pickle.load(f)

            logger.info(f"Loaded {len(pattern_map)} patterns")
            logger.info(
                f"Model validation accuracy: {config['validation_accuracy']:.2%}"
            )

            # Initialize Analyst (Support/Resistance)
            self.analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
            logger.info("[OK] Analyst initialized (S/R detection)")

            # Initialize Sniper (Pattern recognition)
            self.sniper = OHLCSniper(
                input_shape=(15, 4), num_classes=config["num_classes"]
            )
            self.sniper.load_model(str(model_path))
            logger.info("[OK] Sniper initialized (Pattern recognition)")

            # Initialize Hybrid Validator (Integration layer)
            self.ai_validator = HybridSignalValidator(
                analyst=self.analyst,
                sniper=self.sniper,
                pattern_id_map=pattern_map,
                sr_threshold_pct=self.params.ai_sr_threshold,  # 0.5% distance to S/R level
                pattern_confidence_min=self.params.ai_pattern_confidence,
                enable_adaptive_thresholds=self.params.ai_enable_adaptive,  # NEW
                strong_signal_bypass_threshold=self.params.ai_strong_signal_bypass,  # NEW# 65% confidence (lowered from 70%)
                use_ai_validation=True,  # Toggle this to enable/disable
            )
            logger.info("[OK] AI Validator initialized")

            logger.info("=" * 70)
            logger.info("✓ AI Layer ready (Status: ENABLED)")
            logger.info("  - S/R Threshold: 0.5%")
            logger.info("  - Pattern Confidence: 65%")
            logger.info("  - Toggle: self.ai_validator.use_ai_validation")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize AI layer: {e}")
            logger.error("AI layer will be disabled")
            self.ai_validator = None
            import traceback

            traceback.print_exc()

    def _initialize_aggregators(self):
        """Initialize signal aggregators with AUTO-PRESET support"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Signal Aggregators")
        logger.info("-" * 70)

        # (Keep your existing AGGREGATOR_PRESETS dictionary here)
        AGGREGATOR_PRESETS = {
            "BTC": {
                "conservative": {
                    "buy_threshold": 0.35,
                    "sell_threshold": 0.40,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.20,
                    "bull_buy_boost": 0.10,
                    "bull_sell_penalty": 0.12,
                    "bear_sell_boost": 0.10,
                    "bear_buy_penalty": 0.12,
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.32,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.30,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.22,
                    "bull_buy_boost": 0.11,
                    "bull_sell_penalty": 0.11,
                    "bear_sell_boost": 0.11,
                    "bear_buy_penalty": 0.11,
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.17,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.25,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.12,
                    "bull_sell_penalty": 0.12,
                    "bear_sell_boost": 0.12,
                    "bear_buy_penalty": 0.12,
                    "min_confidence_to_use": 0.09,
                    "min_signal_quality": 0.25,
                    "hold_contribution_pct": 0.18,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "scalper": {
                    "buy_threshold": 0.24,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.25,
                    "three_strategy_bonus": 0.30,
                    "bull_buy_boost": 0.15,
                    "bull_sell_penalty": 0.10,
                    "bear_sell_boost": 0.15,
                    "bear_buy_penalty": 0.10,
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.20,
                    "hold_contribution_pct": 0.20,
                    "allow_single_override": True,
                    "single_override_threshold": 0.65,
                    "verbose": False,
                    "min_quality_margin": 0.05,
                    "opposition_penalty": 0.5,
                },
            },
            "GOLD": {
                "conservative": {
                    "buy_threshold": 0.38,
                    "sell_threshold": 0.42,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.07,
                    "bull_sell_penalty": 0.09,
                    "bear_sell_boost": 0.07,
                    "bear_buy_penalty": 0.09,
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.30,
                    "hold_contribution_pct": 0.12,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.33,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.08,
                    "bull_sell_penalty": 0.08,
                    "bear_sell_boost": 0.08,
                    "bear_buy_penalty": 0.08,
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.14,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.28,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.30,
                    "bull_buy_boost": 0.09,
                    "bull_sell_penalty": 0.09,
                    "bear_sell_boost": 0.09,
                    "bear_buy_penalty": 0.09,
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.22,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "opposition_penalty": 0.5,
                    "verbose": False,
                },
                "scalper": {
                    "buy_threshold": 0.23,
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
                    "min_quality_margin": 0.06,
                    "opposition_penalty": 0.5,
                },
            },
        }

        aggregator_cfg = self.config.get("aggregator_settings", {})
        preset = aggregator_cfg.get("preset", "auto")

        if preset == "auto":
            logger.info("\n" + "=" * 70)
            logger.info("🤖 AUTO-PRESET MODE ENABLED")
            logger.info("=" * 70)

            selector = AutoPresetSelector(self.data_manager, self.config)
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
            logger.info(f"Using manual preset: {preset.upper()}")
            asset_presets = {name: preset for name in self.strategies.keys()}

        self.selected_presets = asset_presets.copy()

        ai_validator_instance = None
        if hasattr(self, "ai_validator") and self.ai_validator is not None:
            ai_validator_instance = self.ai_validator
            logger.info("[AGGREGATOR] AI validator available: ✅")
        else:
            logger.warning("[AGGREGATOR] AI validator not yet initialized: ⚠️")

        # Ensure aggregators dict exists
        if not hasattr(self, "aggregators") or self.aggregators is None:
            self.aggregators = {}

        for asset_name, strategies in self.strategies.items():
            if not self.config["assets"][asset_name].get("enabled", False):
                continue

            strategy_count = len(strategies)
            if strategy_count == 0:
                logger.warning(f"[!] {asset_name}: No strategies, skipping aggregator")
                continue

            selected_preset = asset_presets.get(asset_name, "balanced")
            config_for_aggregator = AGGREGATOR_PRESETS.get(asset_name, {}).get(
                selected_preset
            )

            if config_for_aggregator is None:
                logger.error(
                    f"[!] No AGGREGATOR_PRESETS found for {asset_name} / {selected_preset}, skipping."
                )
                continue

            try:
                self.aggregators[asset_name] = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_name,
                    config=config_for_aggregator,
                    ai_validator=(
                        ai_validator_instance if self.params.use_ai_validation else None
                    ),
                    enable_ai_circuit_breaker=True,
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
        self.initialize_ai_layer()
        self._initialize_aggregators()
        # 1. Add monitoring
        self.ai_monitor = AIValidatorMonitor(self.ai_validator)

        # 2. Schedule periodic reports
        schedule.every(1).hours.do(self.ai_monitor.log_periodic_report)

        # 3. (Optional) Add tuner for analysis
        self.ai_tuner = AIValidatorTuner(self.ai_validator)

        # 4. Enable detailed logging during development
        self.ai_validator.detailed_logging = True

    # ✨ NEW: Improved Telegram thread management
    def _start_telegram_with_monitoring(self):
        """Start Telegram bot with health monitoring and auto-restart"""
        max_restart_interval = 300  # 5 minutes

        while not self._shutdown_requested:
            try:
                # Check if we should attempt restart
                if self._telegram_error_count >= self._max_telegram_restarts:
                    if self._telegram_last_restart:
                        time_since_restart = (
                            datetime.now() - self._telegram_last_restart
                        ).total_seconds()
                        if time_since_restart < max_restart_interval:
                            logger.warning(
                                f"[TELEGRAM] Max restarts reached, waiting {max_restart_interval - time_since_restart:.0f}s"
                            )
                            time.sleep(60)
                            continue

                    # Reset counter after cooldown
                    logger.info("[TELEGRAM] Resetting restart counter after cooldown")
                    self._telegram_error_count = 0

                logger.info(
                    f"[TELEGRAM] Starting bot (attempt {self._telegram_error_count + 1}/{self._max_telegram_restarts})..."
                )
                self._run_telegram_loop()

                # If we get here, the loop exited normally
                if self._shutdown_requested:
                    logger.info("[TELEGRAM] Normal shutdown")
                    break
                else:
                    logger.warning("[TELEGRAM] Loop exited unexpectedly")
                    self._telegram_error_count += 1
                    self._telegram_last_restart = datetime.now()

                    if not self._shutdown_requested:
                        logger.info("[TELEGRAM] Restarting in 10 seconds...")
                        time.sleep(10)

            except Exception as e:
                logger.error(f"[TELEGRAM] Critical error: {e}", exc_info=True)
                self._telegram_error_count += 1
                self._telegram_last_restart = datetime.now()

                if not self._shutdown_requested:
                    logger.info("[TELEGRAM] Restarting in 10 seconds...")
                    time.sleep(10)

        logger.info("[TELEGRAM] Monitoring thread stopped")

    async def _start_telegram_bot(self):
        """
        ✨ FIXED: Start bot with proper cancellation handling
        """
        try:
            logger.info("[TELEGRAM] Initializing bot...")
            await self.telegram_bot.initialize()

            self._telegram_ready.set()
            logger.info("[TELEGRAM] ✅ Bot ready")

            # Run polling until shutdown requested
            await self.telegram_bot.run_polling()

        except asyncio.CancelledError:
            # ✨ FIX: Expected during shutdown - don't log as error
            logger.info("[TELEGRAM] Bot cancelled (shutdown)")
            raise  # Re-raise to propagate cancellation

        except Exception as e:
            logger.error(f"[TELEGRAM] Bot error: {e}", exc_info=True)
            self._telegram_ready.clear()
            raise

    def _run_telegram_loop(self):
        """
        ✨ FIXED: Run Telegram with proper cleanup (NO RECURSION)
        """
        # Create fresh event loop for this thread
        self.telegram_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.telegram_loop)
        self._telegram_ready.clear()

        try:
            logger.info("[TELEGRAM] Event loop starting...")

            # Run the async bot
            self.telegram_loop.run_until_complete(self._start_telegram_bot())

        except KeyboardInterrupt:
            logger.info("[TELEGRAM] Keyboard interrupt received")

        except asyncio.CancelledError:
            # ✨ FIX: This is NORMAL during shutdown
            logger.info("[TELEGRAM] Tasks cancelled (shutdown)")

        except Exception as e:
            logger.error(f"[TELEGRAM] Loop error: {e}", exc_info=True)
            self._telegram_is_healthy = False

        finally:
            logger.info("[TELEGRAM] Cleaning up event loop...")
            self._telegram_ready.clear()

            try:
                # ✨ FIXED: Gentle task cleanup
                pending = [
                    task
                    for task in asyncio.all_tasks(self.telegram_loop)
                    if not task.done()
                ]

                if pending:
                    logger.info(f"[TELEGRAM] Cancelling {len(pending)} tasks...")

                    # Cancel each task
                    for task in pending:
                        if not task.cancelled():
                            task.cancel()

                    # Give them a moment to cancel
                    try:
                        self.telegram_loop.run_until_complete(
                            asyncio.wait(pending, timeout=2.0)
                        )
                    except Exception as e:
                        logger.debug(f"[TELEGRAM] Task wait: {e}")

                # Close the loop
                if not self.telegram_loop.is_closed():
                    self.telegram_loop.close()
                    logger.info("[TELEGRAM] Event loop closed")

            except Exception as e:
                logger.debug(f"[TELEGRAM] Cleanup error: {e}")

    def _send_telegram_notification(self, coro):
        """
        ✨ FIXED: Send notification with proper error handling
        """
        if not self.telegram_bot or not self.telegram_loop:
            logger.debug("[TELEGRAM] Bot/loop not available")
            return

        if not self._telegram_ready.is_set():
            logger.debug("[TELEGRAM] Bot not ready, skipping notification")
            return

        # ✨ FIX: Check if loop is still running
        if self.telegram_loop.is_closed():
            logger.debug("[TELEGRAM] Loop closed, skipping notification")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.telegram_loop)
            future.result(timeout=10)

        except TimeoutError:
            logger.debug("[TELEGRAM] Notification timeout")

        except RuntimeError as e:
            # ✨ FIX: Handle "Event loop is closed" gracefully
            if "closed" in str(e).lower():
                logger.debug("[TELEGRAM] Loop closed during notification")
            else:
                logger.debug(f"[TELEGRAM] Runtime error: {e}")

        except Exception as e:
            logger.debug(f"[TELEGRAM] Notification error: {e}")

    # ✨ IMPROVED: Trading cycle with better error handling
    def run_trading_cycle(self):
        """Execute one complete trading cycle with DTM support"""
        try:
            logger.info("\n" + "=" * 70)
            logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            # Refresh capital if live
            if not self.portfolio_manager.is_paper_mode:
                try:
                    self.portfolio_manager.refresh_capital()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh capital: {e}")

            self.reset_daily_counters()
            self._check_dtm_positions()

            # Get current prices for all assets
            
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
                        self.binance_handler
                        if exchange == "binance"
                        else self.mt5_handler
                    )

                    if handler:
                        current_prices[asset_name] = handler.get_current_price()
                except Exception as e:
                    logger.error(f"Failed to get {asset_name} price: {e}")

            # ✨ NEW: Update positions with OHLC data for DTM
            try:
                ohlc_data_dict = {}
                for asset_name in enabled:
                    # Only update if position exists
                    if self.portfolio_manager.has_position(asset_name):
                        handler = (
                            self.binance_handler
                            if self.config["assets"][asset_name].get("exchange")
                            == "binance"
                            else self.mt5_handler
                        )

                        if handler:
                            try:
                                # Fetch latest OHLC
                                end_time = datetime.now(timezone.utc)
                                start_time = end_time - timedelta(hours=24)

                                if handler == self.binance_handler:
                                    df = self.data_manager.fetch_binance_data(
                                        symbol=self.config["assets"][asset_name][
                                            "symbol"
                                        ],
                                        interval=self.config["assets"][asset_name].get(
                                            "interval", "1h"
                                        ),
                                        start_date=start_time.strftime("%Y-%m-%d"),
                                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    )
                                else:
                                    df = self.data_manager.fetch_mt5_data(
                                        symbol=self.config["assets"][asset_name][
                                            "symbol"
                                        ],
                                        timeframe=self.config["assets"][asset_name].get(
                                            "timeframe", "H1"
                                        ),
                                        start_date=start_time.strftime("%Y-%m-%d"),
                                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    )

                                if len(df) > 0:
                                    latest = df.iloc[-1]
                                    ohlc_data_dict[asset_name] = {
                                        "high": latest["high"],
                                        "low": latest["low"],
                                        "close": latest["close"],
                                    }
                            except Exception as e:
                                logger.debug(
                                    f"[DTM] Failed to fetch OHLC for {asset_name}: {e}"
                                )

                # Update all positions with DTM
                if ohlc_data_dict:
                    closed_count = self.portfolio_manager.update_positions_with_ohlc(
                        ohlc_data_dict
                    )
                    if closed_count > 0:
                        logger.info(
                            f"[DTM] Closed {closed_count} position(s) via dynamic management"
                        )

                        # Send Telegram notification
                        if self.telegram_bot and self._telegram_ready.is_set():
                            for asset in enabled:
                                if not self.portfolio_manager.has_position(asset):
                                    closed = self.portfolio_manager.closed_positions
                                    if closed:
                                        last_trade = closed[-1]
                                        if last_trade["asset"] == asset:
                                            self._send_telegram_notification(
                                                self.telegram_bot.notify_trade_closed(
                                                    asset=asset,
                                                    side=last_trade["side"],
                                                    pnl=last_trade["pnl"],
                                                    pnl_pct=last_trade["pnl_pct"] * 100,
                                                    reason=last_trade["reason"],
                                                )
                                            )

            except Exception as e:
                logger.error(f"[DTM] Error updating positions: {e}")

            # Update positions with current prices (traditional method)
            try:
                self.portfolio_manager.update_positions(current_prices)
            except Exception as e:
                logger.error(f"[ERROR] Failed to update positions: {e}")

            logger.info(f"[ASSETS] Enabled: {', '.join(enabled)}")

            # Trade each asset
            for asset_name in enabled:
                try:
                    self.trade_asset(asset_name)
                    time.sleep(2)
                except Exception as e:
                    logger.error(
                        f"[ERROR] Failed to trade {asset_name}: {e}", exc_info=True
                    )
                    self._consecutive_errors += 1

            # Get portfolio status
            try:
                status = self.portfolio_manager.get_portfolio_status(current_prices)
                self._log_portfolio_status(status)

                # ✨ NEW: Log DTM status for open positions
                self._log_dtm_status()

            except Exception as e:
                logger.error(f"[ERROR] Failed to get portfolio status: {e}")

            # Reset error counter on successful cycle
            self._consecutive_errors = 0
            self._last_successful_cycle = datetime.now()

            logger.info("[OK] Trading cycle complete")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[ERROR] Trading cycle failed: {e}", exc_info=True)
            self._consecutive_errors += 1

            # Check if we have too many consecutive errors
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.error(
                    f"[CRITICAL] Too many consecutive errors ({self._consecutive_errors}), "
                    "entering safe mode"
                )
                self._send_telegram_notification(
                    self.telegram_bot.notify_error(
                        f"🚨 CRITICAL: {self._consecutive_errors} consecutive errors!\n"
                        "Bot entering safe mode."
                    )
                )
                time.sleep(300)
                
    def _check_dtm_positions(self):
        """
        ✅ NEW: Check all DTM-managed positions for exits
        This runs EVERY cycle (every 5 min) for real-time trailing
        """
        try:
            for asset_name in ["BTC", "GOLD"]:
                if not self.config["assets"][asset_name].get("enabled", False):
                    continue
                
                position = self.portfolio_manager.get_position(asset_name)
                if not position or not position.trade_manager:
                    continue
                
                # Get handler
                exchange = self.config["assets"][asset_name].get("exchange", "binance")
                handler = self.binance_handler if exchange == "binance" else self.mt5_handler
                
                if not handler:
                    continue
                
                # Check DTM with real-time updates
                try:
                    if exchange == "binance":
                        handler.check_and_update_positions_dtm(asset_name)
                    else:
                        # MT5 equivalent
                        handler.check_and_update_positions(asset_name)
                
                except Exception as e:
                    logger.error(f"[DTM] Error checking {asset_name}: {e}")
        
        except Exception as e:
            logger.error(f"[DTM] Position check error: {e}")           

    def _log_dtm_status(self):
        """Log Dynamic Trade Manager status for all positions"""
        try:
            has_dtm = False

            for asset, position in self.portfolio_manager.positions.items():
                if position.trade_manager:
                    has_dtm = True
                    dtm_status = position.get_dtm_status()

                    logger.info(f"\n{'-' * 70}")
                    logger.info(f"[DTM STATUS] {asset} {dtm_status['side'].upper()}")
                    logger.info(f"{'-' * 70}")
                    logger.info(f"Entry Price:      ${dtm_status['entry_price']:,.2f}")
                    logger.info(
                        f"Current Price:    ${dtm_status['current_price']:,.2f}"
                    )
                    logger.info(f"P&L:              {dtm_status['pnl_pct']:+.2f}%")
                    logger.info(f"")
                    logger.info(
                        f"Stop Loss:        ${dtm_status['stop_loss']:,.2f} ({dtm_status['distance_to_sl_pct']:+.2f}% away)"
                    )
                    logger.info(
                        f"Take Profit:      ${dtm_status['take_profit']:,.2f} ({dtm_status['distance_to_tp_pct']:+.2f}% away)"
                    )
                    logger.info(f"")
                    logger.info(
                        f"Profit Locked:    {'✓ YES' if dtm_status['profit_locked'] else '✗ NO'}"
                    )
                    logger.info(f"Updates Count:    {dtm_status['update_count']}")
                    logger.info(f"Last Update:      {dtm_status['last_update']}")
                    logger.info(f"{'-' * 70}")

            if not has_dtm and len(self.portfolio_manager.positions) > 0:
                logger.debug("[DTM] No positions using dynamic management")

        except Exception as e:
            logger.error(
                f"Error logging DTM status: {e}"
            )  # Wait 5 minutes before next cycle # Wait 5 minutes before next cycle

    def _log_portfolio_status(self, status):
        """Log portfolio status"""
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

        daily_pnl = status.get("daily_pnl", 0)
        daily_pnl_color = "+" if daily_pnl >= 0 else ""
        logger.info(f"Daily P&L:      {daily_pnl_color}${daily_pnl:,.2f}")

        realized_pnl = status.get("realized_pnl_today", 0)
        realized_color = "+" if realized_pnl >= 0 else ""
        logger.info(f"Realized P&L:   {realized_color}${realized_pnl:,.2f}")

        # Show individual position P&L
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

                if pos_data.get("mt5_ticket"):
                    mt5_profit = pos_data.get("mt5_profit", 0)
                    logger.info(
                        f"  MT5 P&L: ${mt5_profit:,.2f} (Ticket: {pos_data['mt5_ticket']})"
                    )

                logger.info("")

        logger.info(f"{'-' * 70}")

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
                try:
                    self.portfolio_manager.refresh_capital()
                    self.portfolio_manager.reset_daily_pnl()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh capital: {e}")

            # Send daily summary via Telegram
            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.send_daily_summary()
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
            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.notify_error(
                            f"🚨 CIRCUIT BREAKER!\nLoss: {loss_pct:.2%}"
                        )
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
        """Execute trading logic for a single asset with error handling"""
        asset_cfg = self.config["assets"][asset_name]
        if not asset_cfg.get("enabled", False):
            return

        if not self.check_market_hours(asset_name):
            logger.debug(f"[SKIP] {asset_name}: Market closed")
            return

        exchange = asset_cfg.get("exchange", "binance")
        symbol = asset_cfg.get("symbol", "BTCUSDT")

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

            # Fetch data with error handling
            try:
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

            except Exception as e:
                logger.error(f"[ERROR] Failed to fetch data for {asset_name}: {e}")
                return

            # Get current price with fallback
            try:
                if exchange == "binance":
                    current_price = handler.get_current_price(symbol)
                else:
                    current_price = handler.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"Failed to get current price: {e}, using last close")
                current_price = df["close"].iloc[-1]

            logger.info(f"[PRICE] {asset_name}: ${current_price:,.2f}")

            # Get aggregated signal
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[!] {asset_name}: No aggregator configured")
                return

            signal, details = aggregator.get_aggregated_signal(df)

            # Log signals
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
            # Check if AI modified anything
            if details.get("ai_modified", False):
                logger.info(f"\n[AI] Signal modifications:")
                for strat, change in details.get("ai_changes", {}).items():
                    logger.info(f"  {strat.upper()}: {change}")

            # Record signal for monitoring
            if self.telegram_bot:
                self.telegram_bot.signal_monitor.record_signal(
                    asset=asset_name,
                    signal=signal,
                    details=details,
                    price=current_price,
                    timestamp=datetime.now(),
                )

            logger.info(f"\n[AGGREGATED] Signal: {signal:>2}")
            logger.info(f"[QUALITY] Score: {details.get('signal_quality', 0):.3f}")
            logger.info(f"[REASONING] {details.get('reasoning', 'N/A')}")

            # Skip if HOLD signal
            if signal == 0:
                logger.info(f"[HOLD] {asset_name}: No action (HOLD signal)")
                return

            # Check for open position before selling
            existing_pos = self.portfolio_manager.get_position(asset_name)
            if signal == -1 and not existing_pos:
                logger.warning(f"[SKIP] {asset_name}: No open position to sell")
                return

            # Check trading limits
            if not self.check_trading_limits():
                logger.info(
                    f"[LIMIT] {asset_name}: Trading limits prevent new position"
                )
                return

            if not self.check_min_time_between_trades(asset_name):
                logger.info(f"[COOLDOWN] {asset_name}: Cooldown period active")
                return

            # Execute signal
            success = False
            try:
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

            except Exception as e:
                logger.error(f"[ERROR] Failed to execute signal for {asset_name}: {e}")
                return

            # Handle success
            if success:
                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()
                signal_type = "BUY" if signal == 1 else "SELL"
                logger.info(
                    f"[SUCCESS] {asset_name} {signal_type} executed (Daily count: {self.trade_count_today})"
                )

                # Send Telegram notifications
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
            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.notify_error(
                            f"Error in {asset_name}:\n{str(e)[:200]}"
                        )
                    )
                except:
                    pass

    def toggle_ai_validation(self, enable: bool):
        """Toggle AI validation on/off"""
        if hasattr(self, "ai_validator") and self.ai_validator is not None:
            self.ai_validator.use_ai_validation = enable
            status = "ENABLED" if enable else "DISABLED"
            logger.info(f"[AI] Validation layer {status}")
            return f"✓ AI Validation {status}"
        else:
            return "✗ AI layer not initialized"

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

    def log_detailed_pnl_report(self):
        """Log detailed P&L report"""
        try:
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

            self.portfolio_manager.update_positions(current_prices)
            status = self.portfolio_manager.get_portfolio_status(current_prices)

            logger.info("\n" + "=" * 70)
            logger.info("DETAILED P&L REPORT")
            logger.info("=" * 70)
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Mode: {status['mode'].upper()}")
            logger.info("")

            logger.info("[CAPITAL]")
            logger.info(
                f"  Initial:     ${self.portfolio_manager.initial_capital:,.2f}"
            )
            logger.info(f"  Current:     ${status['capital']:,.2f}")
            logger.info(f"  Total Value: ${status['total_value']:,.2f}")
            logger.info("")

            logger.info("[P&L BREAKDOWN]")
            logger.info(f"  Daily P&L:      ${status['daily_pnl']:+,.2f}")
            logger.info(f"  Realized Today: ${status['realized_pnl_today']:+,.2f}")
            logger.info(f"  Unrealized:     ${status['total_unrealized_pnl']:+,.2f}")
            logger.info("")

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

    def _restart_telegram_thread(self):
        """
        ✨ NEW: Restart Telegram thread if it dies unexpectedly
        """
        try:
            logger.info("[TELEGRAM] Restarting thread...")

            self._telegram_ready.clear()
            self.telegram_thread = Thread(
                target=self._run_telegram_loop, daemon=True, name="TelegramBot"
            )
            self.telegram_thread.start()

            if self._telegram_ready.wait(timeout=30):
                logger.info("[TELEGRAM] ✅ Thread restarted successfully")
            else:
                logger.warning("[TELEGRAM] ⚠️ Restart timeout")

        except Exception as e:
            logger.error(f"[TELEGRAM] Restart failed: {e}")

    def start(self):
        """
        ✨ FIXED: Start bot with proper Telegram thread management
        """
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT INITIALIZING")
        logger.info("=" * 70)

        try:
            # Initialize exchanges and handlers
            self.initialize_exchanges()
            self.load_models()

            # ✨ FIXED: Start Telegram in dedicated thread
            if self.telegram_bot:
                logger.info("\n[TELEGRAM] Starting bot thread...")

                self._telegram_should_stop.clear()
                self.telegram_thread = Thread(
                    target=self._run_telegram_loop, daemon=True, name="TelegramBot"
                )
                self.telegram_thread.start()

                # Wait for bot to be ready (with timeout)
                logger.info("[TELEGRAM] Waiting for bot to be ready...")
                if self._telegram_ready.wait(timeout=30):
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
            self._main_loop_running = True

            logger.info(f"\n[OK] Trading bot running")
            logger.info(
                f"[TIME] Cycle interval: {check_interval}s ({check_interval / 60:.1f}min)"
            )
            logger.info(f"Press Ctrl+C to stop\n")

            # Run initial cycle
            self.run_trading_cycle()

            # ✨ FIXED: Main loop with health monitoring
            last_health_check = datetime.now()

            while self.is_running:
                try:
                    schedule.run_pending()

                    # Periodic Telegram health check
                    now = datetime.now()
                    if (now - last_health_check).total_seconds() >= 300:  # 5 min
                        if self.telegram_thread and not self.telegram_thread.is_alive():
                            logger.warning("[HEALTH] Telegram thread died!")
                            if (
                                self._telegram_is_healthy
                                and not self._telegram_should_stop.is_set()
                            ):
                                logger.info("[HEALTH] Attempting Telegram restart...")
                                self._restart_telegram_thread()

                        last_health_check = now

                    time.sleep(1)

                except KeyboardInterrupt:
                    raise

                except Exception as e:
                    logger.error(f"[ERROR] Main loop error: {e}", exc_info=True)
                    time.sleep(10)

            logger.info("[STOP] Main loop ended")

        except KeyboardInterrupt:
            logger.info("\n[!] KeyboardInterrupt received")
            self.stop()

        except Exception as e:
            logger.error(f"[FATAL] Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)

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
            self._main_loop_running = True
            logger.info(f"\n[OK] Trading bot running")
            logger.info(
                f"[TIME] Cycle interval: {check_interval}s ({check_interval / 60:.1f}min)"
            )
            logger.info(f"Press Ctrl+C to stop\n")

            # Run initial cycle
            self.run_trading_cycle()

            # Main loop with health monitoring
            last_health_check = datetime.now()
            health_check_interval = 300  # 5 minutes

            while self.is_running and not self._shutdown_requested:
                try:
                    schedule.run_pending()

                    # Periodic health check
                    now = datetime.now()
                    if (
                        now - last_health_check
                    ).total_seconds() >= health_check_interval:
                        self._perform_health_check()
                        last_health_check = now

                    time.sleep(1)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"[ERROR] Main loop error: {e}", exc_info=True)
                    time.sleep(10)  # Wait before continuing

            logger.info("[STOP] Main loop ended")

        except KeyboardInterrupt:
            logger.info("\n[!] KeyboardInterrupt received")
            self.stop()
        except Exception as e:
            logger.error(f"[FATAL] Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)

    def _perform_health_check(self):
        """Perform periodic health check"""
        try:
            logger.debug("[HEALTH] Performing health check...")

            # Check if we've had a successful cycle recently
            if self._last_successful_cycle:
                time_since_cycle = (
                    datetime.now() - self._last_successful_cycle
                ).total_seconds()
                if time_since_cycle > 1800:  # 30 minutes
                    logger.warning(
                        f"[HEALTH] No successful cycle in {time_since_cycle/60:.0f} minutes"
                    )

            # Check consecutive errors
            if self._consecutive_errors > 0:
                logger.warning(
                    f"[HEALTH] Consecutive errors: {self._consecutive_errors}"
                )

            # Check Telegram bot
            if self.telegram_bot and not self._telegram_ready.is_set():
                logger.warning("[HEALTH] Telegram bot not ready")
                if self._telegram_error_count < self._max_telegram_restarts:
                    logger.info("[HEALTH] Telegram bot will auto-restart")

            logger.debug("[HEALTH] Health check complete")

        except Exception as e:
            logger.error(f"[HEALTH] Health check error: {e}")

    def stop(self):
        """
        ✨ FIXED: Graceful shutdown with proper Telegram cleanup
        """
        if hasattr(self, "_shutdown_in_progress") and self._shutdown_in_progress:
            logger.info("[STOP] Shutdown already in progress")
            return

        self._shutdown_in_progress = True

        logger.info("\n" + "=" * 70)
        logger.info("[STOP] SHUTTING DOWN TRADING BOT")
        logger.info("=" * 70)

        self.is_running = False
        self._main_loop_running = False

        # Close positions if configured
        if self.config["trading"].get("close_positions_on_shutdown", False):
            logger.info("[STOP] Closing open positions...")
            try:
                self.portfolio_manager.close_all_positions()
                logger.info("[STOP] ✅ Positions closed")
            except Exception as e:
                logger.error(f"[STOP] Error closing positions: {e}")

        # ✨ FIXED: Shutdown Telegram properly
        if self.telegram_bot:
            logger.info("[TELEGRAM] Initiating shutdown...")

            # Signal shutdown
            self._telegram_should_stop.set()

            # Try to shutdown gracefully
            if self.telegram_loop and not self.telegram_loop.is_closed():
                try:
                    # Schedule shutdown in the Telegram loop
                    future = asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.shutdown(), self.telegram_loop
                    )

                    # Wait with timeout
                    try:
                        future.result(timeout=10)
                        logger.info("[TELEGRAM] ✅ Bot shutdown complete")
                    except TimeoutError:
                        logger.warning("[TELEGRAM] ⚠️ Shutdown timeout")
                        future.cancel()

                except RuntimeError as e:
                    if "closed" in str(e).lower():
                        logger.info("[TELEGRAM] ✅ Loop already closed")
                    else:
                        logger.debug(f"[TELEGRAM] Shutdown error: {e}")

                except Exception as e:
                    logger.debug(f"[TELEGRAM] Shutdown error: {e}")

            # Wait for thread to finish (with timeout)
            if self.telegram_thread and self.telegram_thread.is_alive():
                logger.info("[TELEGRAM] Waiting for thread (10s timeout)...")
                self.telegram_thread.join(timeout=10)

                if self.telegram_thread.is_alive():
                    logger.warning(
                        "[TELEGRAM] ⚠️ Thread still alive (will be abandoned)"
                    )
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
