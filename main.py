#!/usr/bin/env python3
"""
Main Trading Bot -  STABILITY VERSION
Enhanced error handling, network resilience, and Telegram thread management
"""


import subprocess
import json
import logging
import sys
import time
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import schedule
import io
import signal
from threading import Thread, Event
from typing import Optional, Tuple, Dict
from types import SimpleNamespace


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
from src.database.database_manager import (
    TradingDatabaseManager,
    calculate_daily_summary_from_trades,
)
from src.ai.visualization import (
    AIVisualizationGenerator,
    TelegramChartSender,
    create_visualization_system,
    should_send_chart,
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

    # ✨  Add log rotation
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
    """Main trading bot with  stability and error recovery"""

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT")
        logger.info("=" * 70)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        setup_logging(self.config)

        # ✨  Initialize AI components as None FIRST
        self.analyst = None
        self.sniper = None
        self.ai_validator = None
        self.ai_monitor = None
        self.ai_tuner = None
        # Visualization system (will be initialized after AI layer)
        self.chart_sender = None

        self.params = SimpleNamespace(
            use_ai_validation=True,
            ai_sr_threshold=0.020,
            ai_pattern_confidence=0.45,
            ai_enable_adaptive=True,
            ai_strong_signal_bypass=0.70,
        )

        # Core components
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None
        self.db_manager = None  # ✨ Initialize BEFORE portfolio

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

        # Telegram thread management
        self.telegram_bot = None
        self.telegram_loop = None
        self.telegram_thread = None
        self._telegram_ready = Event()
        self._telegram_should_stop = Event()
        self._telegram_error_count = 0
        self._max_telegram_restarts = 3
        self._telegram_last_restart = None

        # Main bot state
        self._shutdown_requested = False
        self._main_loop_running = False
        self._last_successful_cycle = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # ✨ NEW: Portfolio snapshot tracking
        self._last_snapshot_time = None
        self._snapshot_interval = self.config.get("database", {}).get(
            "snapshot_interval_seconds", 300
        )

        # Initialize components in CORRECT order
        self._initialize_telegram()
        self._initialize_strategies()

    def initialize_exchanges(self):
        """
        ✨  Initialize exchanges with proper database integration
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
                    logger.error("[FAIL] Failed to initialize MT5")
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
                    logger.error("[FAIL] Failed to initialize Binance")
                    self.config["assets"]["BTC"]["enabled"] = False
            except Exception as e:
                logger.error(f"[FAIL] Binance initialization error: {e}")
                self.config["assets"]["BTC"]["enabled"] = False

        # ✨ STEP 1.5: Initialize Database BEFORE Portfolio
        logger.info("\n" + "-" * 70)
        logger.info("STEP 1.5: Initializing Database Connection")
        logger.info("-" * 70)

        if self.config.get("database", {}).get("enabled", False):
            try:

                db_config = self.config["database"]
                self.db_manager = TradingDatabaseManager(
                    supabase_url=db_config["supabase_url"],
                    supabase_key=db_config["supabase_key"],
                )

                logger.info("[DB] ✓ Connected to Supabase")

                # Test connection
                self.db_manager.supabase.table("trades").select("id").limit(1).execute()
                logger.info("[DB] ✓ Connection test passed")

            except Exception as e:
                logger.error(f"[DB] Failed to initialize: {e}")
                logger.warning("[DB] Continuing without database logging")
                self.db_manager = None
        else:
            logger.info("[DB] Database disabled in config")
            self.db_manager = None

        # ✨ STEP 2: Initialize Portfolio Manager WITH Database
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

            # ✨ CRITICAL FIX: Link database to portfolio
            if self.db_manager:
                self.portfolio_manager.db_manager = self.db_manager
                logger.info("[DB] ✓ Database linked to portfolio manager")

            logger.info(
                f"[OK] Portfolio Manager initialized (Mode: {self.portfolio_manager.mode.upper()})"
            )
            logger.info(f"     Capital: ${self.portfolio_manager.current_capital:,.2f}")

            # ✨ Log system startup to database
            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="startup",
                    severity="info",
                    message="Trading bot started",
                    component="main",
                    metadata={
                        "mode": self.portfolio_manager.mode,
                        "initial_capital": self.portfolio_manager.initial_capital,
                        "assets_enabled": [
                            asset
                            for asset, cfg in self.config["assets"].items()
                            if cfg.get("enabled", False)
                        ],
                    },
                )

        except Exception as e:
            logger.error(f"[FAIL] Portfolio Manager initialization failed: {e}")
            raise

        # ✨ STEP 3: Initialize Execution Handlers WITH Database
        logger.info("\n" + "-" * 70)
        logger.info("STEP 3: Initializing Execution Handlers")
        logger.info("-" * 70)

        if (
            self.config["assets"]["BTC"].get("enabled", False)
            and self.data_manager.binance_client is not None
        ):
            try:

                self.binance_handler = BinanceExecutionHandler(
                    config=self.config,
                    client=self.data_manager.binance_client,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )
                
                # ✅ NEW: Enable Futures for SHORT trading
                if self.config["assets"]["BTC"].get("enable_futures", False):
                    from src.execution.binance_futures import enable_futures_for_binance_handler
                    
                    futures_enabled = enable_futures_for_binance_handler(self.binance_handler)
                    
                    if futures_enabled:
                        logger.info("[FUTURES] ✅ SHORT trading via Futures API enabled")
                    else:
                        logger.warning("[FUTURES] ⚠️ Futures unavailable, shorts will be simulated")

                # ✨ Link database to handler
                if self.db_manager:
                    self.binance_handler.db_manager = self.db_manager
                    logger.info("[DB] ✓ Database linked to Binance handler")

                logger.info("[OK] Binance Execution Handler initialized")

            except Exception as e:
                logger.error(f"[FAIL] Binance handler: {e}")
                self.binance_handler = None
                self.config["assets"]["BTC"]["enabled"] = False

        if self.config["assets"]["GOLD"].get("enabled", False) and mt5_initialized:
            try:

                self.mt5_handler = MT5ExecutionHandler(
                    config=self.config,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )

                # ✨ Link database to handler
                if self.db_manager:
                    self.mt5_handler.db_manager = self.db_manager
                    logger.info("[DB] ✓ Database linked to MT5 handler")

                logger.info("[OK] MT5 Execution Handler initialized")

            except Exception as e:
                logger.error(f"[FAIL] MT5 handler: {e}")
                self.mt5_handler = None
                self.config["assets"]["GOLD"]["enabled"] = False

        if not self.binance_handler and not self.mt5_handler:
            raise RuntimeError("No execution handlers available!")

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
        ✨  Safe AI initialization with proper error handling
        """
        try:
            logger.info("=" * 70)
            logger.info("Initializing AI Layer...")
            logger.info("=" * 70)

            models_dir = Path("models/ai")

            # Check if model files exist
            model_path = models_dir / "sniper_dual_timeframe_v1.weights.h5"
            mapping_path = models_dir / "sniper_dual_timeframe_v1_mapping.pkl"
            config_path = models_dir / "sniper_dual_timeframe_v1_config.pkl"

            if not model_path.exists():
                logger.error(f"[AI] Model not found: {model_path}")
                logger.error("[AI] Please run: python train_dual_timeframe.py")
                logger.warning("[AI] AI layer will be DISABLED")
                return False

            # Load pattern mapping
            try:
                with open(mapping_path, "rb") as f:
                    pattern_map = pickle.load(f)

                logger.info(f"[AI] Loaded {len(pattern_map)} patterns")

                # Ensure noise class exists
                if "Noise" not in pattern_map:
                    logger.warning("[AI] Adding missing 'Noise' class")
                    pattern_map["Noise"] = 0
                    with open(mapping_path, "wb") as f:
                        pickle.dump(pattern_map, f)

            except Exception as e:
                logger.error(f"[AI] Pattern mapping error: {e}")
                return False

            # Load config
            try:
                with open(config_path, "rb") as f:
                    config = pickle.load(f)

                logger.info(f"[AI] Model: {config.get('model_version', 'unknown')}")
                logger.info(f"[AI] Accuracy: {config.get('val_accuracy', 0):.2%}")

            except Exception as e:
                logger.warning(f"[AI] Config warning: {e}")
                config = {"num_classes": len(pattern_map)}

            # Initialize Analyst (4H)
            try:
                self.analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
                logger.info("[AI] ✓ Analyst (4H S/R)")

            except Exception as e:
                logger.error(f"[AI] Analyst failed: {e}")
                return False

            # Initialize Sniper (15min)
            try:
                num_classes = config.get("num_classes", len(pattern_map))

                self.sniper = OHLCSniper(
                    input_shape=(15, 4), num_classes=num_classes, dropout_rate=0.3
                )

                logger.info(f"[AI] Sniper created ({num_classes} classes)")

                # Load weights
                self.sniper.load_model(str(model_path))
                logger.info("[AI] ✓ Weights loaded")

            except ValueError as e:
                if "shape" in str(e).lower():
                    logger.error(f"[AI] ✗ ARCHITECTURE MISMATCH!")
                    logger.error(f"     {e}")
                    logger.error("[AI] Solution: Retrain model")
                    logger.error("     python train_dual_timeframe.py")
                    return False
                raise

            except Exception as e:
                logger.error(f"[AI] Sniper failed: {e}")
                return False

            # Initialize Validator
            try:
                self.ai_validator = HybridSignalValidator(
                    analyst=self.analyst,
                    sniper=self.sniper,
                    pattern_id_map=pattern_map,
                    sr_threshold_pct=self.params.ai_sr_threshold,
                    pattern_confidence_min=self.params.ai_pattern_confidence,
                    enable_adaptive_thresholds=self.params.ai_enable_adaptive,
                    strong_signal_bypass_threshold=self.params.ai_strong_signal_bypass,
                    use_ai_validation=self.params.use_ai_validation,
                )

                logger.info("[AI] ✓ Validator initialized")

            except Exception as e:
                logger.error(f"[AI] Validator failed: {e}")
                return False

            # Initialize monitoring (optional)
            try:
                self.ai_monitor = AIValidatorMonitor(self.ai_validator)
                self.ai_tuner = AIValidatorTuner(self.ai_validator)

                schedule.every(1).hours.do(self.ai_monitor.log_periodic_report)

                logger.info("[AI] ✓ Monitoring enabled")

            except Exception as e:
                logger.warning(f"[AI] Monitoring warning: {e}")

            logger.info("=" * 70)
            logger.info("✅ AI Layer READY")
            logger.info("  Analyst:   4H S/R detection")
            logger.info("  Sniper:    15min patterns")
            logger.info(
                f"  Status:    {'ENABLED' if self.params.use_ai_validation else 'DISABLED'}"
            )
            logger.info("=" * 70)

            return True

        except Exception as e:
            logger.error(f"[AI] Initialization failed: {e}", exc_info=True)
            logger.error("[AI] AI layer DISABLED")

            # Reset all AI components
            self.analyst = None
            self.sniper = None
            self.ai_validator = None
            self.ai_monitor = None
            self.ai_tuner = None

            return False

    def _initialize_aggregators(self):
        """
        Initialize signal aggregators with clear mode selection
        
        Modes:
        - 'performance' (default): Your existing advanced aggregator
        - 'council': New institutional-style weighted council
        - 'hybrid': Both aggregators for comparison
        """
        
        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING SIGNAL AGGREGATORS")
        logger.info("=" * 70)
        
        # ================================================================
        # STEP 1: Load Configuration
        # ================================================================
        aggregator_cfg = self.config.get("aggregator_settings", {})
        mode = aggregator_cfg.get("mode", "performance").lower()
        preset = aggregator_cfg.get("preset", "auto")
        
        logger.info(f"\nMode:   {mode.upper()}")
        logger.info(f"Preset: {preset.upper()}")
        
        # Validate mode
        valid_modes = ["performance", "council", "hybrid"]
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}', defaulting to 'performance'")
            mode = "performance"
        
        # ================================================================
        # STEP 2: Define Preset Configurations
        # ================================================================
        # These presets work for BOTH aggregator types
        AGGREGATOR_PRESETS = {
    "BTC": {
        "conservative": {
            "buy_threshold": 0.32,
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
            "buy_threshold": 0.33,
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
        
        # ================================================================
        # STEP 3: Auto-Preset Selection (if enabled)
        # ================================================================
        if preset == "auto":
            logger.info("\n🤖 AUTO-PRESET MODE ENABLED")
            logger.info("Analyzing market conditions...")
            
            try:
                from src.execution.auto_preset_selector import AutoPresetSelector
                selector = AutoPresetSelector(self.data_manager, self.config)
                asset_presets = selector.get_preset_for_all_assets()
                
                logger.info("\n📊 AUTO-PRESET RESULTS:")
                for asset, selected_preset in asset_presets.items():
                    logger.info(f"  {asset:6} → {selected_preset.upper()}")
                    
            except Exception as e:
                logger.error(f"Auto-preset failed: {e}, using 'balanced'")
                asset_presets = {name: "balanced" for name in self.strategies.keys()}
        
        elif preset in ["conservative", "balanced", "aggressive"]:
            logger.info(f"\nUsing manual preset: {preset.upper()}")
            asset_presets = {name: preset for name in self.strategies.keys()}
        
        else:
            logger.warning(f"Unknown preset '{preset}', using 'balanced'")
            asset_presets = {name: "balanced" for name in self.strategies.keys()}
        
        # Store selected presets
        self.selected_presets = asset_presets.copy()
        
        # ================================================================
        # STEP 4: Get AI Validator (if available)
        # ================================================================
        ai_validator = None
        if hasattr(self, 'ai_validator') and self.ai_validator is not None:
            ai_validator = self.ai_validator
            logger.info("\n✅ AI Validator available")
        else:
            logger.info("\n⚠️  AI Validator not available")
        
        # ================================================================
        # STEP 5: Initialize Aggregators for Each Asset
        # ================================================================
        logger.info("\n" + "-" * 70)
        logger.info("CREATING AGGREGATORS")
        logger.info("-" * 70)
        
        # Ensure aggregators dict exists
        if not hasattr(self, 'aggregators') or self.aggregators is None:
            self.aggregators = {}
        
        for asset_name, strategies in self.strategies.items():
            # Skip disabled assets
            if not self.config["assets"][asset_name].get("enabled", False):
                logger.info(f"\n[SKIP] {asset_name}: Asset disabled")
                continue
            
            # Check if we have strategies
            strategy_count = len(strategies)
            if strategy_count == 0:
                logger.warning(f"\n[SKIP] {asset_name}: No strategies available")
                continue
            
            # Get preset config for this asset
            selected_preset = asset_presets.get(asset_name, "balanced")
            preset_config = AGGREGATOR_PRESETS.get(asset_name, {}).get(selected_preset)
            
            if preset_config is None:
                logger.error(f"\n[ERROR] {asset_name}: No config for preset '{selected_preset}'")
                continue
            
            logger.info(f"\n{asset_name}:")
            logger.info(f"  Strategies: {strategy_count}")
            logger.info(f"  Preset:     {selected_preset}")
            
            # ============================================================
            # MODE SELECTION
            # ============================================================
            
            if mode == "performance":
                # --------------------------------------------------------
                # PERFORMANCE MODE (Your existing aggregator)
                # --------------------------------------------------------
                try:
                    from src.execution.signal_aggregator import PerformanceWeightedAggregator
                    
                    self.aggregators[asset_name] = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=ai_validator if self.params.use_ai_validation else None,
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=getattr(self, 'detailed_logging', False),
                        strong_signal_bypass_threshold=getattr(self.params, 'ai_strong_signal_bypass', 0.70),
                    )
                    
                    logger.info(f"  Type:       Performance Aggregator")
                    logger.info(f"  AI:         {'Enabled' if ai_validator else 'Disabled'}")
                    
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Performance aggregator: {e}")
                    continue
            
            elif mode == "council":
                # --------------------------------------------------------
                # COUNCIL MODE (New institutional aggregator)
                # --------------------------------------------------------
                try:
                    from src.execution.council_aggregator import InstitutionalCouncilAggregator
                    
                    self.aggregators[asset_name] = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=ai_validator if self.params.use_ai_validation else None,
                        enable_detailed_logging=getattr(self, 'detailed_logging', False),
                        
                        # Council-specific settings
                        trend_aligned_threshold=3.5,
                        counter_trend_threshold=4.0,
                    )
                    
                    logger.info(f"  Type:       Council Aggregator")
                    logger.info(f"  AI:         {'Enabled' if ai_validator else 'Disabled'}")
                    logger.info(f"  Thresholds: 3.5 (trend) / 4.0 (counter)")
                    
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Council aggregator: {e}")
                    continue
            
            elif mode == "hybrid":
                # --------------------------------------------------------
                # HYBRID MODE (Both aggregators for comparison)
                # --------------------------------------------------------
                try:
                    from src.execution.signal_aggregator import PerformanceWeightedAggregator
                    from src.execution.council_aggregator import InstitutionalCouncilAggregator
                    
                    # Create both
                    perf_agg = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=ai_validator if self.params.use_ai_validation else None,
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=False,  # Reduce noise in hybrid mode
                        strong_signal_bypass_threshold=getattr(self.params, 'ai_strong_signal_bypass', 0.70),
                    )
                    
                    council_agg = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=ai_validator if self.params.use_ai_validation else None,
                        enable_detailed_logging=False,
                        trend_aligned_threshold=3.5,
                        counter_trend_threshold=4.0,
                    )
                    
                    # Store both in a dict
                    self.aggregators[asset_name] = {
                        'performance': perf_agg,
                        'council': council_agg,
                        'mode': 'hybrid',
                    }
                    
                    logger.info(f"  Type:       Hybrid (Both aggregators)")
                    logger.info(f"  AI:         {'Enabled' if ai_validator else 'Disabled'}")
                    logger.info(f"  Note:       Signals require consensus")
                    
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Hybrid aggregators: {e}")
                    continue
        
        # ================================================================
        # STEP 6: Summary
        # ================================================================
        logger.info("\n" + "=" * 70)
        logger.info("AGGREGATOR INITIALIZATION COMPLETE")
        logger.info("=" * 70)
        
        successful = len([a for a in self.aggregators.values() if a is not None])
        total = len([a for a in self.config["assets"].values() if a.get("enabled")])
        
        logger.info(f"\nStatus: {successful}/{total} aggregators ready")
        logger.info(f"Mode:   {mode.upper()}")
        
        if mode == "hybrid":
            logger.info("\n⚠️  HYBRID MODE: Signals require consensus from both aggregators")
        
        logger.info("\n")
    
    
    def _get_aggregated_signal_hybrid(self, asset_name: str, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Helper method for hybrid mode with FIXED detail merging
        Gets signals from both aggregators and requires consensus
        """
        aggregator_dict = self.aggregators.get(asset_name)
        
        if not isinstance(aggregator_dict, dict) or aggregator_dict.get('mode') != 'hybrid':
            raise ValueError(f"Asset {asset_name} not in hybrid mode")
        
        perf_agg = aggregator_dict['performance']
        council_agg = aggregator_dict['council']
        
        # Get signals from both
        perf_signal, perf_details = perf_agg.get_aggregated_signal(df)
        council_signal, council_details = council_agg.get_aggregated_signal(df)
        
        # ================================================================
        # FIX: Merge confidence fields intelligently
        # ================================================================
        
        # Use AVERAGE confidences from both aggregators
        merged_details = {
            # Timestamp from either (should be same)
            'timestamp': perf_details.get('timestamp', council_details.get('timestamp')),
            
            # Regime from either (should be same since both use EMA)
            'regime': perf_details.get('regime', council_details.get('regime')),
            'regime_confidence': (
                perf_details.get('regime_confidence', 0) + 
                council_details.get('regime_confidence', 0)
            ) / 2,
            
            # ✅ FIX: Average the strategy confidences
            'mr_signal': perf_details.get('mr_signal', council_details.get('mr_signal', 0)),
            'mr_confidence': (
                perf_details.get('mr_confidence', 0) + 
                council_details.get('mr_confidence', 0)
            ) / 2,
            
            'tf_signal': perf_details.get('tf_signal', council_details.get('tf_signal', 0)),
            'tf_confidence': (
                perf_details.get('tf_confidence', 0) + 
                council_details.get('tf_confidence', 0)
            ) / 2,
            
            'ema_signal': perf_details.get('ema_signal', council_details.get('ema_signal', 0)),
            'ema_confidence': (
                perf_details.get('ema_confidence', 0) + 
                council_details.get('ema_confidence', 0)
            ) / 2,
            
            # Scores from both (for analysis)
            'buy_score': max(
                perf_details.get('buy_score', 0),
                council_details.get('buy_score', 0)
            ),
            'sell_score': max(
                perf_details.get('sell_score', 0),
                council_details.get('sell_score', 0)
            ),
            
            # Metadata
            'aggregator_type': 'hybrid',
            'performance_details': perf_details,
            'council_details': council_details,
        }
        
        # ================================================================
        # Consensus logic
        # ================================================================
        if perf_signal == council_signal and perf_signal != 0:
            # ✅ CONSENSUS: Both agree on BUY or SELL
            logger.info(f"\n✅ CONSENSUS: Both aggregators agree on {perf_signal:+d}")
            
            # Calculate consensus quality (average + bonus)
            perf_quality = perf_details.get('signal_quality', 0.5)
            council_quality = council_details.get('signal_quality', 0.5)
            
            # Average quality with +10% consensus bonus
            consensus_quality = min((perf_quality + council_quality) / 2 + 0.10, 1.0)
            
            return perf_signal, {
                **merged_details,
                'signal': perf_signal,
                'consensus': True,
                'signal_quality': consensus_quality,
                'reasoning': f"CONSENSUS {self._signal_str(perf_signal)}: Both aggregators agree (Quality: {consensus_quality:.2%})",
            }
        
        elif perf_signal != 0 and council_signal != 0:
            # ⚠️ CONFLICT: Both want to trade but disagree on direction
            logger.warning(f"\n⚠️  CONFLICT: Perf={perf_signal:+d}, Council={council_signal:+d} → HOLD")
            
            return 0, {
                **merged_details,
                'signal': 0,
                'consensus': False,
                'conflict': True,
                'signal_quality': 0.0,
                'reasoning': f"CONFLICT: Performance wants {self._signal_str(perf_signal)}, Council wants {self._signal_str(council_signal)}",
            }
        
        else:
            # ⏸️ NO CONSENSUS: At least one wants to HOLD
            logger.info(f"\n⏸️  NO CONSENSUS: Perf={perf_signal:+d}, Council={council_signal:+d} → HOLD")
            
            # Use quality from whichever aggregator wanted to trade
            active_quality = (
                perf_details.get('signal_quality', 0) if perf_signal != 0 
                else council_details.get('signal_quality', 0)
            )
            
            return 0, {
                **merged_details,
                'signal': 0,
                'consensus': False,
                'signal_quality': active_quality * 0.5,  # Penalize lack of consensus
                'reasoning': f"NO CONSENSUS: Performance={self._signal_str(perf_signal)}, Council={self._signal_str(council_signal)}",
            }

    def _signal_str(self, signal: int) -> str:
        """Convert signal to readable string"""
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")
        
    def _update_asset_signal(self, asset_name: str):
            """
            Update signal for an asset (handles all aggregator modes)
            """
            try:
                asset_cfg = self.config["assets"][asset_name]
                exchange = asset_cfg.get("exchange", "binance")
                symbol = asset_cfg.get("symbol")
                
                # Fetch latest data
                end_time = datetime.now(timezone.utc)
                
                if exchange == "binance":
                    interval = asset_cfg.get("interval", "1h")
                    lookback = 15 if interval == "1h" else 60
                    start_time = end_time - timedelta(days=lookback)
                    
                    df = self.data_manager.fetch_binance_data(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                else:
                    timeframe = asset_cfg.get("timeframe", "H1")
                    lookback = 25 if timeframe == "H1" else 75
                    start_time = end_time - timedelta(days=lookback)
                    
                    df = self.data_manager.fetch_mt5_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                
                df = self.data_manager.clean_data(df)
                
                if len(df) < 250:
                    logger.debug(f"[SIGNAL] {asset_name}: Insufficient data ({len(df)}/250)")
                    return
                
                # Get handler for current price
                handler = self.binance_handler if exchange == "binance" else self.mt5_handler
                if not handler:
                    logger.debug(f"[SIGNAL] {asset_name}: No handler available")
                    return
                
                try:
                    current_price = handler.get_current_price(symbol)
                except:
                    current_price = df["close"].iloc[-1]
                
                # Get aggregator
                aggregator = self.aggregators.get(asset_name)
                if not aggregator:
                    logger.debug(f"[SIGNAL] {asset_name}: No aggregator")
                    return
                
                # ============================================================
                # HANDLE DIFFERENT AGGREGATOR MODES
                # ============================================================
                if isinstance(aggregator, dict) and aggregator.get('mode') == 'hybrid':
                    # Hybrid mode: Get consensus signal
                    signal, details = self._get_aggregated_signal_hybrid(asset_name, df)
                else:
                    # Normal mode: Single aggregator
                    signal, details = aggregator.get_aggregated_signal(df)
                
                # Log signal to database (if enabled)
                if self.db_manager:
                    signal_id, is_new = self.db_manager.insert_signal_smart(
                        asset=asset_name,
                        signal=signal,
                        signal_quality=details.get("signal_quality", 0),
                        regime=details.get("regime", "UNKNOWN"),
                        regime_confidence=details.get("regime_confidence", 0),
                        mr_signal=details.get("mr_signal", 0),
                        mr_confidence=details.get("mr_confidence", 0),
                        tf_signal=details.get("tf_signal", 0),
                        tf_confidence=details.get("tf_confidence", 0),
                        ema_signal=details.get("ema_signal"),
                        ema_confidence=details.get("ema_confidence"),
                        buy_score=details.get("buy_score"),
                        sell_score=details.get("sell_score"),
                        reasoning=details.get("reasoning"),
                        price=current_price,
                        ai_validated=details.get("ai_validated", False),
                        ai_modified=details.get("ai_modified", False),
                        ai_details=details.get("ai_validation"),
                        executed=False,
                    )
                    
                    if is_new:
                        logger.info(
                            f"[SIGNAL] {asset_name}: {signal:+2d} "
                            f"(Q={details.get('signal_quality', 0):.2f})"
                        )
                
                # Update Telegram monitor (if available)
                if self.telegram_bot:
                    self.telegram_bot.signal_monitor.record_signal(
                        asset=asset_name,
                        signal=signal,
                        details=details,
                        price=current_price,
                        timestamp=datetime.now(),
                    )
            
            except Exception as e:
                logger.error(f"[SIGNAL] {asset_name} update error: {e}", exc_info=True)
        
        
    def load_models(self):
            """
            ✨  Load models with safe AI initialization
            """
            logger.info("\n" + "-" * 70)
            logger.info("Loading Trained Models")
            logger.info("-" * 70)

            loaded = 0
            expected = 0

            # Load strategy models
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
                                logger.error(f"[FAIL] {model_path}")
                        except Exception as e:
                            logger.error(f"[FAIL] {model_path}: {e}")
                    else:
                        logger.error(f"[FAIL] Not found: {model_path}")

            if loaded == 0:
                logger.error("=" * 70)
                logger.error("NO MODELS LOADED! Run: python train.py")
                logger.error("=" * 70)
                sys.exit(1)

            logger.info(f"\n[OK] Loaded {loaded}/{expected} strategy models")

            # ✨  Try to initialize AI (non-fatal)
            ai_success = False
            try:
                ai_success = self.initialize_ai_layer()
            except Exception as e:
                logger.error(f"[AI] Initialization error: {e}")
                ai_success = False

            if not ai_success:
                logger.warning("[AI] Continuing WITHOUT AI validation")
                # Ensure all AI components are None
                self.analyst = None
                self.sniper = None
                self.ai_validator = None

            # Initialize aggregators
            self._initialize_aggregators()

            # ✨  Only set logging if AI validator exists
            if self.ai_validator:
                try:
                    self.ai_validator.detailed_logging = True
                    logger.info("[AI] Detailed logging enabled")
                except Exception as e:
                    logger.warning(f"[AI] Logging config failed: {e}")

            if self.ai_validator and self.telegram_bot and self.analyst and self.sniper:
                try:
                    logger.info("[VIZ] Initializing AI visualization system...")
                    self.chart_sender = create_visualization_system(
                        telegram_bot=self.telegram_bot,
                        analyst=self.analyst,
                        sniper=self.sniper,
                        ai_validator=self.ai_validator,
                    )

                    if self.chart_sender:
                        logger.info("[VIZ] ✅ Visualization system ready")
                    else:
                        logger.warning("[VIZ] ⚠️ Visualization system failed")

                except Exception as e:
                    logger.error(f"[VIZ] Initialization error: {e}")
                    self.chart_sender = None
            else:
                logger.info("[VIZ] Skipping visualization (AI or Telegram not available)")
                self.chart_sender = None

    # ✨ NEW:  Telegram thread management
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
        ✨  Start bot with proper cancellation handling
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
        ✨  Run Telegram with proper cleanup (NO RECURSION)
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
                # ✨  Gentle task cleanup
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
        ✨  Send notification with proper error handling
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

    # ✨  Trading cycle with better error handling
    def run_trading_cycle(self):
        """Execute one complete trading cycle with VTM support"""
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
            self._check_VTM_positions()

            enabled = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            logger.info(f"[SIGNALS] Updating signals for: {', '.join(enabled)}")

            for asset_name in enabled:
                try:
                    # Update signal even if market is closed or can't trade
                    self._update_asset_signal(asset_name)
                except Exception as e:
                    logger.error(f"[SIGNAL] Error updating {asset_name}: {e}")

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

            # ✨ NEW: Update positions with OHLC data for VTM
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
                                    f"[VTM] Failed to fetch OHLC for {asset_name}: {e}"
                                )

                # Update all positions with VTM
                if ohlc_data_dict:
                    closed_count = self.portfolio_manager.update_positions_with_ohlc(
                        ohlc_data_dict
                    )
                    if closed_count > 0:
                        logger.info(
                            f"[VTM] Closed {closed_count} position(s) via dynamic management"
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
                logger.error(f"[VTM] Error updating positions: {e}")

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
                        f"[ERROR] {asset_name} trade failed: {e}", exc_info=True
                    )

                    # ✨ Log error to database
                    if self.db_manager:
                        self.db_manager.log_system_event(
                            event_type="error",
                            severity="error",
                            message=f"{asset_name} trading error: {str(e)}",
                            component="trade_execution",
                        )

            # Get portfolio status
            try:
                status = self.portfolio_manager.get_portfolio_status(current_prices)
                self._log_portfolio_status(status)
            except Exception as e:
                logger.error(f"[ERROR] Portfolio status: {e}")

            # Reset error counter
            self._consecutive_errors = 0
            self._last_successful_cycle = datetime.now()

            # ✨ Take periodic snapshot
            if self.db_manager:
                self._maybe_take_portfolio_snapshot()

            logger.info("[OK] Trading cycle complete")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[ERROR] Cycle failed: {e}", exc_info=True)
            self._consecutive_errors += 1

            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="error",
                    severity="critical",
                    message=f"Trading cycle error: {str(e)}",
                    component="main",
                )
                time.sleep(300)

    def _check_VTM_positions(self):
        """
        ✅ NEW: Check all VTM-managed positions for exits
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
                handler = (
                    self.binance_handler if exchange == "binance" else self.mt5_handler
                )

                if not handler:
                    continue

                # Check VTM with real-time updates
                try:
                    if exchange == "binance":
                        handler.check_and_update_positions_VTM(asset_name)
                    else:
                        # MT5 equivalent
                        handler.check_and_update_positions(asset_name)

                except Exception as e:
                    logger.error(f"[VTM] Error checking {asset_name}: {e}")

        except Exception as e:
            logger.error(f"[VTM] Position check error: {e}")

    def _log_VTM_status(self):
        """Log Dynamic Trade Manager status for all positions"""
        try:
            has_VTM = False

            for asset, position in self.portfolio_manager.positions.items():
                if position.trade_manager:
                    has_VTM = True
                    VTM_status = position.get_vtm_status()

                    logger.info(f"\n{'-' * 70}")
                    logger.info(f"[VTM STATUS] {asset} {VTM_status['side'].upper()}")
                    logger.info(f"{'-' * 70}")
                    logger.info(f"Entry Price:      ${VTM_status['entry_price']:,.2f}")
                    logger.info(
                        f"Current Price:    ${VTM_status['current_price']:,.2f}"
                    )
                    logger.info(f"P&L:              {VTM_status['pnl_pct']:+.2f}%")
                    logger.info(f"")
                    logger.info(
                        f"Stop Loss:        ${VTM_status['stop_loss']:,.2f} ({VTM_status['distance_to_sl_pct']:+.2f}% away)"
                    )
                    logger.info(
                        f"Take Profit:      ${VTM_status['take_profit']:,.2f} ({VTM_status['distance_to_tp_pct']:+.2f}% away)"
                    )
                    logger.info(f"")
                    logger.info(
                        f"Profit Locked:    {'✓ YES' if VTM_status['profit_locked'] else '✗ NO'}"
                    )
                    logger.info(f"Updates Count:    {VTM_status['update_count']}")
                    logger.info(f"Last Update:      {VTM_status['last_update']}")
                    logger.info(f"{'-' * 70}")

            if not has_VTM and len(self.portfolio_manager.positions) > 0:
                logger.debug("[VTM] No positions using dynamic management")

        except Exception as e:
            logger.error(
                f"Error logging VTM status: {e}"
            )  # Wait 5 minutes before next cycle # Wait 5 minutes before next cycle

    def _log_portfolio_status(self, status):
        """
        ✅  Enhanced logging with per-asset position breakdown
        """
        logger.info(f"\n{'-' * 70}")
        logger.info("[PORTFOLIO STATUS]")
        logger.info(f"{'-' * 70}")
        logger.info(f"Mode:           {status.get('mode', 'N/A').upper()}")
        logger.info(f"Total Value:    ${status.get('total_value', 0):,.2f}")
        logger.info(f"Cash:           ${status.get('capital', 0):,.2f}")
        logger.info(f"Exposure:       ${status.get('total_exposure', 0):,.2f}")

        # ✅ NEW: Per-asset position counts
        asset_counts = status.get("asset_position_counts", {})
        asset_details = status.get("asset_positions_detail", {})
        max_per_asset = status.get("max_positions_per_asset", 3)

        logger.info(f"\n[POSITION COUNTS] (Max: {max_per_asset} per asset per side)")
        for asset, counts in asset_counts.items():
            long_count = counts["long"]
            short_count = counts["short"]
            total_count = counts["total"]

            # Get position IDs for this asset
            details = asset_details.get(asset, {})
            long_ids = details.get("long_ids", [])
            short_ids = details.get("short_ids", [])
            long_tickets = details.get("long_tickets", [])
            short_tickets = details.get("short_tickets", [])

            logger.info(f"\n{asset}:")
            logger.info(f"  LONG:  {long_count}/{max_per_asset}")
            if long_ids:
                logger.info(f"    IDs:     {', '.join(long_ids)}")
            if long_tickets:
                logger.info(f"    Tickets: {', '.join(map(str, long_tickets))}")

            logger.info(f"  SHORT: {short_count}/{max_per_asset}")
            if short_ids:
                logger.info(f"    IDs:     {', '.join(short_ids)}")
            if short_tickets:
                logger.info(f"    Tickets: {', '.join(map(str, short_tickets))}")

            logger.info(f"  TOTAL: {total_count}/{max_per_asset * 2}")

        # Daily P&L
        daily_pnl = status.get("daily_pnl", 0)
        daily_pnl_color = "+" if daily_pnl >= 0 else ""
        logger.info(f"\n[P&L]")
        logger.info(f"Daily P&L:      {daily_pnl_color}${daily_pnl:,.2f}")

        realized_pnl = status.get("realized_pnl_today", 0)
        realized_color = "+" if realized_pnl >= 0 else ""
        logger.info(f"Realized P&L:   {realized_color}${realized_pnl:,.2f}")

        # Individual position P&L
        positions = status.get("positions", {})
        if positions:
            logger.info(f"\n{'-' * 70}")
            logger.info("[INDIVIDUAL POSITIONS]")
            logger.info(f"{'-' * 70}")

            for position_id, pos_data in positions.items():
                asset = pos_data.get("asset", "N/A")
                side = pos_data.get("side", "N/A").upper()
                entry = pos_data.get("entry_price", 0)
                current = pos_data.get("current_price", 0)
                pnl = pos_data.get("pnl", 0)
                pnl_pct = pos_data.get("pnl_pct", 0) * 100

                pnl_color = "+" if pnl >= 0 else ""

                logger.info(f"\n{position_id} ({asset} {side}):")
                logger.info(f"  Entry:   ${entry:,.2f}")
                logger.info(f"  Current: ${current:,.2f}")
                logger.info(
                    f"  P&L:     {pnl_color}${pnl:,.2f} ({pnl_color}{pnl_pct:.2f}%)"
                )

                if pos_data.get("mt5_ticket"):
                    logger.info(f"  MT5:     Ticket {pos_data['mt5_ticket']}")

        logger.info(f"\n{'-' * 70}")

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
        """
        Execute trading logic with FRESH signal details from aggregator
        """
        asset_cfg = self.config["assets"][asset_name]
        if not asset_cfg.get("enabled", False):
            return

        # Check market hours BEFORE trading
        if not self.check_market_hours(asset_name):
            logger.debug(f"[SKIP] {asset_name}: Market closed")
            return

        exchange = asset_cfg.get("exchange", "binance")
        symbol = asset_cfg.get("symbol", "BTCUSDT")
        handler = self.binance_handler if exchange == "binance" else self.mt5_handler

        if not handler:
            logger.warning(f"[!] {asset_name}: Handler unavailable")
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

            # ================================================================
            # FIX 1: Get FRESH signal from aggregator (not database)
            # ================================================================
            
            # Fetch latest data
            end_time = datetime.now(timezone.utc)
            
            if exchange == "binance":
                interval = asset_cfg.get("interval", "1h")
                lookback = 15 if interval == "1h" else 60
                start_time = end_time - timedelta(days=lookback)
                
                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "H1")
                lookback = 25 if timeframe == "H1" else 75
                start_time = end_time - timedelta(days=lookback)
                
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            
            df = self.data_manager.clean_data(df)
            
            if len(df) < 250:
                logger.warning(f"[SKIP] {asset_name}: Insufficient data ({len(df)}/250)")
                return
            
            # Get aggregator
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[SKIP] {asset_name}: No aggregator available")
                return
            
            # ✅ FIX: Get FRESH signal with full details
            if isinstance(aggregator, dict) and aggregator.get('mode') == 'hybrid':
                # Hybrid mode: Get consensus signal
                signal, details = self._get_aggregated_signal_hybrid(asset_name, df)
            else:
                # Normal mode: Single aggregator
                signal, details = aggregator.get_aggregated_signal(df)
                
            
            # Get current price
            try:
                current_price = handler.get_current_price(symbol)
            except:
                current_price = float(df['close'].iloc[-1])
            
            # Update details with current price
            details['price'] = current_price

            logger.info(f"[PRICE] {asset_name}: ${current_price:,.2f}")

            # ================================================================
            # FIX 2: Use FRESH details for logging (not database reconstruction)
            # ================================================================
            
            # Log AI validation if present
            if details.get("ai_validation"):
                ai_viz = details["ai_validation"]
                logger.info(f"\n[AI VALIDATION]")
                logger.info(f"  Pattern:  {ai_viz.get('pattern_name', 'N/A')}")
                logger.info(f"  Conf:     {ai_viz.get('pattern_confidence', 0):.2%}")
                logger.info(f"  S/R:      {'Yes' if ai_viz.get('sr_analysis', {}).get('near_sr_level') else 'No'}")
                logger.info(f"  Result:   {ai_viz.get('action', 'N/A').upper()}")

            # Log signals with FRESH confidences
            logger.info(f"\n[SIGNAL] Strategy Analysis:")
            logger.info(f"  Mean Reversion:   {details.get('mr_signal', 0):>2} "
                    f"(confidence: {details.get('mr_confidence', 0):.3f})")
            logger.info(f"  Trend Following:  {details.get('tf_signal', 0):>2} "
                    f"(confidence: {details.get('tf_confidence', 0):.3f})")
            logger.info(f"  EMA Regime:       {details.get('regime', 'N/A')} "
                    f"(confidence: {details.get('regime_confidence', 0):.3f})")

            if details.get("ai_modified", False):
                logger.info(f"\n[AI] Signal modifications detected")

            logger.info(f"\n[AGGREGATED] Signal: {signal:>2}")
            logger.info(f"[QUALITY] Score: {details.get('signal_quality', 0):.3f}")
            logger.info(f"[REASONING] {details.get('reasoning', 'N/A')}")
            
            # ✅ FIX: For council aggregator, show council scores
            if details.get('aggregator_type') == 'council':
                logger.info(f"\n[COUNCIL SCORES]")
                logger.info(f"  Total:     {details.get('total_score', 0):.2f} / 5.0")
                logger.info(f"  Required:  {details.get('required_score', 0):.2f}")
                
                scores = details.get('scores', {})
                for judge, score in scores.items():
                    logger.info(f"  {judge.capitalize():12s}: {score:.2f}")

            # Skip if HOLD signal
            if signal == 0:
                logger.info(f"[HOLD] {asset_name}: No action (HOLD signal)")
                
                # ✅ OPTIONAL: Still log to database for tracking
                if self.db_manager:
                    self.db_manager.insert_signal_smart(
                        asset=asset_name,
                        signal=signal,
                        signal_quality=details.get("signal_quality", 0),
                        regime=details.get("regime", "UNKNOWN"),
                        regime_confidence=details.get("regime_confidence", 0),
                        mr_signal=details.get("mr_signal", 0),
                        mr_confidence=details.get("mr_confidence", 0),
                        tf_signal=details.get("tf_signal", 0),
                        tf_confidence=details.get("tf_confidence", 0),
                        ema_signal=details.get("ema_signal"),
                        ema_confidence=details.get("ema_confidence"),
                        buy_score=details.get("buy_score"),
                        sell_score=details.get("sell_score"),
                        reasoning=details.get("reasoning"),
                        price=current_price,
                        ai_validated=details.get("ai_validated", False),
                        ai_modified=details.get("ai_modified", False),
                        ai_details=details.get("ai_validation"),
                        executed=False,
                    )
                
                return

            # Check trading limits
            if not self.check_trading_limits():
                logger.info(f"[LIMIT] {asset_name}: Trading limits prevent new position")
                return

            if not self.check_min_time_between_trades(asset_name):
                logger.info(f"[COOLDOWN] {asset_name}: Cooldown period active")
                return

            # Store BEFORE state
            positions_before = self.portfolio_manager.get_asset_positions(asset_name)
            position_ids_before = {p.position_id for p in positions_before}

            logger.info(f"[BEFORE] {len([p for p in positions_before if p.side == 'long'])} LONG, "
                    f"{len([p for p in positions_before if p.side == 'short'])} SHORT positions")

            # Execute signal (using FRESH details with correct confidences)
            success = False
            try:
                if exchange == "binance":
                    success = self.binance_handler.execute_signal(
                        signal=signal,
                        current_price=current_price,
                        asset_name=asset_name,
                        confidence_score=details.get("signal_quality", 0.5),
                        market_condition="bull" if details.get("regime") == "🚀 BULL" else "bear",
                    )
                    self.binance_handler.check_and_update_positions(asset_name)
                else:
                    success = self.mt5_handler.execute_signal(
                        signal=signal,
                        symbol=symbol,
                        asset_name=asset_name,
                        confidence_score=details.get("signal_quality", 0.5),
                        market_condition="bull" if details.get("regime") == "🚀 BULL" else "bear",
                    )
                    self.mt5_handler.check_and_update_positions(asset_name)

            except Exception as e:
                logger.error(f"[ERROR] Failed to execute signal for {asset_name}: {e}")
                return

            # Handle success (unchanged)
            if success:
                positions_after = self.portfolio_manager.get_asset_positions(asset_name)
                position_ids_after = {p.position_id for p in positions_after}
                new_position_ids = position_ids_after - position_ids_before
                closed_position_ids = position_ids_before - position_ids_after

                logger.info(f"[AFTER] {len([p for p in positions_after if p.side == 'long'])} LONG, "
                        f"{len([p for p in positions_after if p.side == 'short'])} SHORT positions")

                # Send notifications for NEW positions
                if new_position_ids:
                    for position_id in new_position_ids:
                        new_pos = next((p for p in positions_after if p.position_id == position_id), None)
                        
                        if new_pos and self.telegram_bot and self._telegram_ready.is_set():
                            try:
                                logger.info(f"[TELEGRAM] Notifying NEW {new_pos.side.upper()} position: {position_id}")
                                
                                self._send_telegram_notification(
                                    self.telegram_bot.notify_trade_opened(
                                        asset=asset_name,
                                        side=new_pos.side,
                                        price=new_pos.entry_price,
                                        size=new_pos.quantity * new_pos.entry_price,
                                        sl=new_pos.stop_loss,
                                        tp=new_pos.take_profit,
                                    )
                                )
                            except Exception as e:
                                logger.error(f"[TELEGRAM] Failed to send open notification: {e}")

                # Send notifications for CLOSED positions
                if closed_position_ids:
                    closed_trades = self.portfolio_manager.closed_positions
                    
                    for position_id in closed_position_ids:
                        matching_trade = next(
                            (t for t in reversed(closed_trades) if t.get("position_id") == position_id),
                            None
                        )
                        
                        if matching_trade and self.telegram_bot and self._telegram_ready.is_set():
                            try:
                                logger.info(f"[TELEGRAM] Notifying CLOSED position: {position_id}")
                                
                                self._send_telegram_notification(
                                    self.telegram_bot.notify_trade_closed(
                                        asset=asset_name,
                                        side=matching_trade["side"],
                                        pnl=matching_trade["pnl"],
                                        pnl_pct=matching_trade["pnl_pct"] * 100,
                                        reason=matching_trade["reason"],
                                    )
                                )
                            except Exception as e:
                                logger.error(f"[TELEGRAM] Failed to send close notification: {e}")

                # Send visualization chart (using FRESH details)
                if new_position_ids and self.chart_sender:
                    try:
                        logger.info(f"[VIZ] Trade executed, sending chart for {asset_name}...")
                        df_4h = self._fetch_4h_data(asset_name)

                        self._send_telegram_notification(
                            self.chart_sender.send_decision_chart(
                                asset_name=asset_name,
                                df_15min=df,  # Use the df we already fetched
                                df_4h=df_4h,
                                signal=signal,
                                details=details,  # ✅ FRESH details with correct confidences
                                current_price=current_price,
                            )
                        )
                        logger.info(f"[VIZ] ✓ Chart sent for executed {asset_name} trade")
                    except Exception as e:
                        logger.error(f"[VIZ] Chart error: {e}")

                # ✅ Log to database with FRESH details
                if self.db_manager:
                    try:
                        signal_id, is_new = self.db_manager.insert_signal_smart(
                            asset=asset_name,
                            signal=signal,
                            signal_quality=details.get("signal_quality", 0),
                            regime=details.get("regime", "UNKNOWN"),
                            regime_confidence=details.get("regime_confidence", 0),
                            mr_signal=details.get("mr_signal", 0),
                            mr_confidence=details.get("mr_confidence", 0),
                            tf_signal=details.get("tf_signal", 0),
                            tf_confidence=details.get("tf_confidence", 0),
                            ema_signal=details.get("ema_signal"),
                            ema_confidence=details.get("ema_confidence"),
                            buy_score=details.get("buy_score"),
                            sell_score=details.get("sell_score"),
                            reasoning=details.get("reasoning"),
                            price=current_price,
                            ai_validated=details.get("ai_validated", False),
                            ai_modified=details.get("ai_modified", False),
                            ai_details=details.get("ai_validation"),
                            executed=True,  # Mark as executed
                        )
                        
                        # Link to trade if new position created
                        if new_position_ids:
                            new_position_id = list(new_position_ids)[0]
                            new_pos = next(
                                (p for p in positions_after if p.position_id == new_position_id),
                                None
                            )
                            
                            if new_pos and hasattr(new_pos, "db_trade_id") and new_pos.db_trade_id:
                                self.db_manager.update_signal_execution(
                                    signal_id=signal_id,
                                    executed=True,
                                    trade_id=new_pos.db_trade_id,
                                )
                    
                    except Exception as e:
                        logger.error(f"[DB] Signal logging error: {e}")

                # Update counters
                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()
                signal_type = "BUY" if signal == 1 else "SELL"
                logger.info(f"[SUCCESS] {asset_name} {signal_type} executed "
                        f"(Daily count: {self.trade_count_today})")
                logger.info(f"  Opened: {len(new_position_ids)}, Closed: {len(closed_position_ids)}")

                # Log trade
                if self.config.get("logging", {}).get("save_trades", True):
                    self._log_trade(asset_name, signal, details, current_price)

            else:
                logger.warning(f"[SKIP] {asset_name}: Trade not executed "
                            f"(limits/cooldowns/handler failure)")

        except Exception as e:
            logger.error(f"[ERROR] {asset_name} trading error: {e}", exc_info=True)
            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="error",
                    severity="error",
                    message=f"{asset_name} trading error: {str(e)}",
                    component="trade_execution",
                )

            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.notify_error(f"Error in {asset_name}:\n{str(e)[:200]}")
                    )
                except:
                    pass

    def _fetch_4h_data(self, asset_name: str) -> pd.DataFrame:
        """
        Helper method to fetch 4H data for S/R analysis

        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            symbol = asset_cfg.get("symbol")
            exchange = asset_cfg.get("exchange", "binance")

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)

            if exchange == "binance":
                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval="4h",
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe="H4",
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            return self.data_manager.clean_data(df)

        except Exception as e:
            logger.error(f"[VIZ] Failed to fetch 4H data: {e}")
            # Return empty dataframe as fallback
            return pd.DataFrame()

    def _update_asset_signal(self, asset_name: str):
        """
        ✅ NEW: Update signal for an asset regardless of trading status
        This ensures continuous signal monitoring even when:
        - Market is closed
        - Position limits are reached
        - Cooldown period is active
        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            exchange = asset_cfg.get("exchange", "binance")
            symbol = asset_cfg.get("symbol")

            # Fetch latest data
            end_time = datetime.now(timezone.utc)

            if exchange == "binance":
                interval = asset_cfg.get("interval", "1h")
                lookback = 15 if interval == "1h" else 60
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "H1")
                lookback = 25 if timeframe == "H1" else 75
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = self.data_manager.clean_data(df)

            if len(df) < 250:
                logger.debug(
                    f"[SIGNAL] {asset_name}: Insufficient data ({len(df)}/250)"
                )
                return

            # Get handler for current price
            handler = (
                self.binance_handler if exchange == "binance" else self.mt5_handler
            )
            if not handler:
                logger.debug(f"[SIGNAL] {asset_name}: No handler available")
                return

            try:
                current_price = handler.get_current_price(symbol)
            except:
                current_price = df["close"].iloc[-1]

            # Get aggregated signal
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.debug(f"[SIGNAL] {asset_name}: No aggregator")
                return

            signal, details = aggregator.get_aggregated_signal(df)

            # ✅  Always log the signal (even HOLD=0)
            if self.db_manager:
                signal_id, is_new = self.db_manager.insert_signal_smart(
                    asset=asset_name,
                    signal=signal,
                    signal_quality=details.get("signal_quality", 0),
                    regime=details.get("regime", "UNKNOWN"),
                    regime_confidence=details.get("regime_confidence", 0),
                    mr_signal=details.get("mr_signal", 0),
                    mr_confidence=details.get("mr_confidence", 0),
                    tf_signal=details.get("tf_signal", 0),
                    tf_confidence=details.get("tf_confidence", 0),
                    ema_signal=details.get("ema_signal"),
                    ema_confidence=details.get("ema_confidence"),
                    buy_score=details.get("buy_score"),
                    sell_score=details.get("sell_score"),
                    reasoning=details.get("reasoning"),
                    price=current_price,
                    ai_validated=details.get("ai_validated", False),
                    ai_modified=details.get("ai_modified", False),
                    ai_details=details.get("ai_validation"),
                    executed=False,
                )

                if is_new:
                    logger.info(
                        f"[SIGNAL] {asset_name}: {signal:+2d} "
                        f"(Q={details.get('signal_quality', 0):.2f}, "
                        f"{details.get('regime', 'N/A')})"
                    )
                else:
                    logger.debug(f"[SIGNAL] {asset_name}: No change ({signal:+2d})")

            # Update Telegram monitor (if available)
            if self.telegram_bot:
                self.telegram_bot.signal_monitor.record_signal(
                    asset=asset_name,
                    signal=signal,
                    details=details,
                    price=current_price,
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"[SIGNAL] {asset_name} update error: {e}", exc_info=True)

    def _maybe_take_portfolio_snapshot(self):
        """Take periodic portfolio snapshots"""
        try:
            now = datetime.now()

            # Check if snapshot is due
            if (
                self._last_snapshot_time is None
                or (now - self._last_snapshot_time).total_seconds()
                >= self._snapshot_interval
            ):

                # Get current prices
                current_prices = {}
                for asset_name in ["BTC", "GOLD"]:
                    if not self.config["assets"][asset_name].get("enabled", False):
                        continue

                    handler = (
                        self.binance_handler
                        if asset_name == "BTC"
                        else self.mt5_handler
                    )
                    if handler:
                        try:
                            current_prices[asset_name] = handler.get_current_price()
                        except:
                            pass

                # Get portfolio status
                status = self.portfolio_manager.get_portfolio_status(current_prices)

                # Insert snapshot
                if self.db_manager:
                    self.db_manager.insert_portfolio_snapshot(
                        total_value=status["total_value"],
                        cash=status["capital"],
                        equity=status["equity"],
                        total_exposure=status["total_exposure"],
                        open_positions=status["open_positions"],
                        unrealized_pnl=status["total_unrealized_pnl"],
                        realized_pnl_today=status["realized_pnl_today"],
                        positions_detail=status.get("positions"),
                    )

                    self._last_snapshot_time = now
                    logger.debug(f"[DB] Portfolio snapshot taken")

        except Exception as e:
            logger.error(f"[DB] Snapshot error: {e}")

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

    def _fetch_current_data(self, asset_name: str) -> pd.DataFrame:
        """
        Helper to fetch current 15min data for chart generation

        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            symbol = asset_cfg.get("symbol")
            exchange = asset_cfg.get("exchange", "binance")

            end_time = datetime.now(timezone.utc)

            # Fetch appropriate data
            if exchange == "binance":
                interval = asset_cfg.get("interval", "15m")
                lookback = 15 if interval == "15m" else 60
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "M15")
                lookback = 25 if timeframe == "M15" else 75
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            return self.data_manager.clean_data(df)

        except Exception as e:
            logger.error(f"[VIZ] Failed to fetch current data for {asset_name}: {e}")
            return pd.DataFrame()

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
            exposure_pct = (
                status["total_exposure"] / status["equity"]
                if status["equity"] > 0
                else 0
            )

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
                f"  Exposure:     ${status['total_exposure']:,.2f} ({exposure_pct:.1%})"
            )
            # logger.info(f"  Drawdown:     {status['drawdown']:.2%}")
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
        ✨  Start bot with proper Telegram thread management
        """
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT INITIALIZING")
        logger.info("=" * 70)

        try:
            # Initialize exchanges and handlers
            self.initialize_exchanges()
            self.load_models()

            # ✨  Start Telegram in dedicated thread
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

            # ✨  Main loop with health monitoring
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
        ✨  Graceful shutdown with proper Telegram cleanup
        """
        if hasattr(self, "_shutdown_in_progress") and self._shutdown_in_progress:
            logger.info("[STOP] Shutdown already in progress")
            return

        self._shutdown_in_progress = True

        logger.info("\n" + "=" * 70)
        logger.info("[STOP] SHUTTING DOWN TRADING BOT")
        logger.info("=" * 70)

        # ✨ Finalize database
        if self.db_manager:
            try:
                calculate_daily_summary_from_trades(self.db_manager, datetime.now())

                self.db_manager.log_system_event(
                    event_type="shutdown",
                    severity="info",
                    message="Trading bot stopped",
                    component="main",
                    metadata={
                        "final_capital": self.portfolio_manager.current_capital,
                        "open_positions": self.portfolio_manager.get_open_positions_count(),
                    },
                )

                logger.info("[DB] ✓ Final updates complete")

            except Exception as e:
                logger.error(f"[DB] Shutdown error: {e}")

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

        # ✨  Shutdown Telegram properly
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


import requests


def start_dashboard_server():
    """
    ✅  Start dashboard server without blocking
    """
    try:
        logger.info("\n" + "=" * 70)
        logger.info("[DASHBOARD] Starting web dashboard...")
        logger.info("=" * 70)

        # Use the absolute path to server.py
        server_path = Path("src/dashboard/server.py").resolve()

        if not server_path.exists():
            logger.error(f"[DASHBOARD] Server file not found: {server_path}")
            return None

        # ✅ FIX 1: Don't capture stdout/stderr - let it write to console
        # This prevents the subprocess from hanging when buffers fill
        server_process = subprocess.Popen(
            [sys.executable, str(server_path)],
            # ✅  Use None instead of PIPE to prevent blocking
            stdout=None,  # Inherits parent's stdout
            stderr=None,  # Inherits parent's stderr
            cwd=Path("src/dashboard").resolve(),
            # ✅ FIX 2: Start new process group (optional, for better isolation)
            start_new_session=True if sys.platform != "win32" else False,
        )

        # Give server time to start
        logger.info("[DASHBOARD] Waiting for server to start...")
        time.sleep(3)

        # ✅ FIX 3: Verify server is actually running
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get("http://localhost:5000/api/health", timeout=2)
                if response.status_code == 200:
                    logger.info("[DASHBOARD] ✅ Server is responding")
                    logger.info("[DASHBOARD] 📊 Dashboard: http://localhost:5000")
                    logger.info(
                        "[DASHBOARD] 🔍 Health:    http://localhost:5000/api/health"
                    )
                    logger.info("=" * 70)
                    return server_process
            except requests.exceptions.RequestException:
                if attempt < max_retries:
                    logger.info(
                        f"[DASHBOARD] Waiting for server... ({attempt}/{max_retries})"
                    )
                    time.sleep(2)
                else:
                    logger.warning("[DASHBOARD] ⚠️ Server may not be fully ready")

        # Check if process is still running
        if server_process.poll() is None:
            logger.info(
                "[DASHBOARD] ✅ Web dashboard available at http://localhost:5000"
            )
            logger.info("=" * 70)
            return server_process
        else:
            logger.error("[DASHBOARD] ❌ Server process died immediately")
            return None

    except Exception as e:
        logger.error(f"[DASHBOARD] Error starting server: {e}")
        return None


def start_dashboard_server_threaded():
    """
    Alternative: Run Flask in thread instead of subprocess
    More reliable for development, no subprocess buffering issues
    """

    def run_flask():
        import sys

        sys.path.insert(0, str(Path("src/dashboard").resolve()))

        try:
            from server import app

            logger.info("=" * 70)
            logger.info("🚀 Dashboard Server (Thread Mode)")
            logger.info("📊 http://localhost:5000")
            logger.info("=" * 70)

            app.run(
                host="0.0.0.0",
                port=5000,
                debug=False,
                threaded=True,
                use_reloader=False,
            )
        except Exception as e:
            logger.error(f"[DASHBOARD] Error: {e}")

    try:
        dashboard_thread = threading.Thread(
            target=run_flask, daemon=True, name="DashboardServer"
        )
        dashboard_thread.start()

        time.sleep(3)

        try:
            response = requests.get("http://localhost:5000/api/health", timeout=2)
            if response.status_code == 200:
                logger.info("[DASHBOARD] ✅ Server ready at http://localhost:5000")
        except:
            logger.info("[DASHBOARD] Starting... check http://localhost:5000")

        return dashboard_thread

    except Exception as e:
        logger.error(f"[DASHBOARD] Error: {e}")
        return None


# Modify the main() function
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

    ai_model = Path("models/ai/sniper_dual_timeframe_v1.weights.h5")
    if not ai_model.exists():
        print("=" * 70)
        print("⚠️  AI MODEL NOT FOUND (Optional)")
        print("=" * 70)
        print(f"  Missing: {ai_model}")
        print("\nBot will run WITHOUT AI validation.")
        print("To enable AI:")
        print("  1. Run: python model_diagnostic.py")
        print("  2. Then: python train_dual_timeframe.py")
        print("=" * 70)
        time.sleep(3)

    # ✨ NEW: Start dashboard server
    server_process = None
    server_thread = None

    if config.get("dashboard", {}).get("enabled", True):
        # Try subprocess first
        server_process = start_dashboard_server()

        # If subprocess fails, try threaded approach
        if not server_process:
            logger.warning("[DASHBOARD] Subprocess failed, trying thread mode...")
            server_thread = start_dashboard_server_threaded()

    try:
        bot = TradingBot()

        if bot.db_manager:
            bot.portfolio_manager.db_manager = bot.db_manager

            # Also pass to all positions
            for position in bot.portfolio_manager.positions.values():
                position.db_manager = bot.db_manager

        bot.start()

    except KeyboardInterrupt:
        logger.info("\n[!] KeyboardInterrupt")
    except Exception as e:
        logger.error(f"[FATAL] {e}", exc_info=True)
        sys.exit(1)
    finally:
        # ✨ NEW: Stop dashboard server on exit
        if server_process:
            logger.info("[DASHBOARD] Stopping web server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                logger.info("[DASHBOARD] ✅ Server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("[DASHBOARD] Force killing server...")
                server_process.kill()

        if server_thread:
            logger.info("[DASHBOARD] Thread will terminate with main process")


if __name__ == "__main__":
    main()
