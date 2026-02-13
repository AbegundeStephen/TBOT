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
from src.execution.auto_preset_selector import DynamicPresetSelector
from src.execution.hybrid_aggregator_selector import HybridAggregatorSelector
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
from src.telegram.telegram_data_manager import ThreadSafeBotDataManager
from src.update.historical_updater import HistoricalDataUpdater
from src.portfolio.hedging_support import (
    enable_hedging_for_portfolio,
    log_hedging_status,
)


import pickle

# Import Telegram bot
from src.telegram import TradingTelegramBot, SignalMonitoringIntegration
from telegram_config import TELEGRAM_CONFIG
from src.global_error_handler import GlobalErrorHandler, ErrorSeverity, handle_errors
from src.execution.mtf_integration import MTFRegimeIntegration
from src.training.autotrainer import ContinuousLearningPipeline
from src.execution.council_aggregator import InstitutionalCouncilAggregator


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
        self.detailed_logging = True

        # Core components
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None
        self.data_manager_telegram = ThreadSafeBotDataManager(max_cache_age=10)
        self.db_manager = None  # ✨ Initialize BEFORE portfolio
        self.signal_monitor = SignalMonitoringIntegration(max_history=100) # MOVED HERE

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
        self._telegram_is_healthy = True

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
        self.mtf_integration = None
        self._current_regime_data = {}

        self.dynamic_selector = None
        self.hybrid_selector = None

        self.error_handler = GlobalErrorHandler(
            telegram_bot=self.telegram_bot,
            db_manager=self.db_manager,
            config={
                "error_window_seconds": 300,  # 5 minutes
                "max_duplicate_notifications": 3,
            },
        )

        logger.info("[ERROR HANDLER] Global error handler initialized")

        self.historical_updater = HistoricalDataUpdater(
            data_manager=self.data_manager, config=self.config
        )
        self._last_history_update = None
        self.autotrainer = None

    def initialize_exchanges(self):
        """
        ✅ FIXED: Initialize exchanges and link handlers to portfolio
        """
        logger.info("\n" + "-" * 70)
        logger.info("STEP 1: Initializing Exchange Connections")
        logger.info("-" * 70)

        mt5_initialized = False

        # ============================================================
        # Connect to MT5 (for GOLD)
        # ============================================================
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

        # ============================================================
        # Connect to Binance (for BTC)
        # ============================================================
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

        # ============================================================
        # STEP 1.5: Initialize Database BEFORE Portfolio
        # ============================================================
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
            except Exception as e:
                logger.error(f"[DB] Failed to initialize: {e}")
                logger.warning("[DB] Continuing without database logging")
                self.db_manager = None
        else:
            logger.info("[DB] Database disabled in config")
            self.db_manager = None

        # ============================================================
        # STEP 2: Initialize Portfolio Manager
        # ============================================================
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
                mt5_handler=mt5_initialized,
                binance_client=self.data_manager.futures_client,  # ✅ FIXED: Use Futures Client
                db_manager=self.db_manager,
            )

            # ✨ NEW: Enable hedging support
            hedging_enabled = self.config.get("trading", {}).get(
                "allow_simultaneous_long_short", True
            )
            if hedging_enabled:
                max_hedge_ratio = self.config.get("portfolio", {}).get(
                    "max_hedge_ratio", 1.0
                )
                enable_hedging_for_portfolio(self.portfolio_manager, max_hedge_ratio)
                logger.info(
                    f"[HEDGING] ✅ Enabled with max ratio {max_hedge_ratio:.0%}"
                )

            logger.info(
                f"[OK] Portfolio Manager initialized (Mode: {self.portfolio_manager.mode.upper()})"
            )
            logger.info(f"     Capital: ${self.portfolio_manager.current_capital:,.2f}")

            # Log system startup to database
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

        # ============================================================
        # STEP 3: Initialize Execution Handlers
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 3: Initializing Execution Handlers")
        logger.info("-" * 70)

        # ✅ BINANCE HANDLER
        if (
            self.config["assets"]["BTC"].get("enabled", False)
            and self.data_manager.get_futures_client() is not None
        ):
            try:
                # Temporarily disable auto-sync
                original_auto_sync = self.config["trading"].get(
                    "auto_sync_on_startup", True
                )
                self.config["trading"]["auto_sync_on_startup"] = False

                self.binance_handler = BinanceExecutionHandler(
                    config=self.config,
                    client=self.data_manager.get_futures_client(),
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )

                # Restore original setting
                self.config["trading"]["auto_sync_on_startup"] = original_auto_sync

                self.binance_handler.trading_bot = self
                if self.binance_handler:
                    self.binance_handler.error_handler = self.error_handler
                    self.binance_handler.trading_bot = self

                # Enable Futures BEFORE sync
                if self.config["assets"]["BTC"].get("enable_futures", False):
                    logger.info("\n[FUTURES] Enabling Futures handler...")
                    from src.execution.binance_futures import (
                        enable_futures_for_binance_handler,
                    )

                    futures_enabled = enable_futures_for_binance_handler(
                        self.binance_handler
                    )

                    if futures_enabled:
                        logger.info(
                            "[FUTURES] ✅ Futures API enabled for FUTURES trading"
                        )
                    else:
                        logger.warning(
                            "[FUTURES] ⚠️ Futures unavailable, shorts simulated"
                        )

                # NOW do the sync (after Futures is enabled)
                if original_auto_sync and not self.portfolio_manager.is_paper_mode:
                    logger.info(
                        "\n[SYNC] Running position sync (after Futures setup)..."
                    )
                    self.binance_handler.sync_positions_with_binance("BTC")

                # Link database to handler
                if self.db_manager:
                    self.binance_handler.db_manager = self.db_manager
                    logger.info("[DB] ✓ Database linked to Binance handler")

                logger.info("[OK] Binance Execution Handler initialized")

            except Exception as e:
                logger.error(f"[FAIL] Binance handler: {e}")
                self.binance_handler = None
                self.config["assets"]["BTC"]["enabled"] = False

        # ✅ MT5 HANDLER
        if self.config["assets"]["GOLD"].get("enabled", False) and mt5_initialized:
            try:
                self.mt5_handler = MT5ExecutionHandler(
                    config=self.config,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )

                self.mt5_handler.trading_bot = self

                if self.mt5_handler:
                    self.mt5_handler.error_handler = self.error_handler
                    self.mt5_handler.trading_bot = self

                # Link database to handler
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

        # ============================================================
        # ✅ STEP 4: LINK HANDLERS TO PORTFOLIO MANAGER
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 4: Linking Execution Handlers to Portfolio")
        logger.info("-" * 70)

        # Create execution_handlers dict
        execution_handlers = {}

        if self.binance_handler:
            execution_handlers["binance"] = self.binance_handler
            logger.info("[LINK] ✓ Binance handler linked")

        if self.mt5_handler:
            execution_handlers["mt5"] = self.mt5_handler
            logger.info("[LINK] ✓ MT5 handler linked")

        # ✅ NEW: Pass handlers to Portfolio Manager
        self.portfolio_manager.execution_handlers = execution_handlers

        logger.info("[OK] Portfolio can now close positions on exchanges")
        logger.info("-" * 70)

    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            if not TELEGRAM_CONFIG.get("enabled", False):
                logger.info("[TELEGRAM] Disabled in config")
                return

            token = TELEGRAM_CONFIG.get("bot_token")
            admin_ids = TELEGRAM_CONFIG.get("admin_ids", [])

            if not token or not admin_ids:
                logger.warning("[TELEGRAM] Missing config")
                return

            self.telegram_bot = TradingTelegramBot(
                token=token, 
                admin_ids=admin_ids, 
                trading_bot=self,
                signal_monitor=self.signal_monitor
            )
            
            # ✅ NEW: Start dedicated loop immediately
            self.telegram_bot._start_dedicated_loop()
            
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

    def run_mtf_regime_analysis(self):
        """
        Run multi-timeframe regime analysis for all enabled assets
        Scheduled to run every 4 hours
        """
        try:
            if not self.mtf_integration:
                logger.debug("[MTF] Not initialized, skipping analysis")
                return

            logger.info("\n" + "=" * 70)
            logger.info(f"[MTF] Running Multi-Timeframe Regime Analysis")
            logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            enabled_assets = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            for asset_name in enabled_assets:
                try:
                    asset_cfg = self.config["assets"][asset_name]
                    symbol = asset_cfg.get("symbol")
                    exchange = asset_cfg.get("exchange", "binance")

                    # Force refresh to get latest data
                    regime_data = self.mtf_integration.get_regime_for_trading(
                        asset_name=asset_name, symbol=symbol, exchange=exchange
                    )

                    # Store in cache for aggregators and trading logic
                    self._current_regime_data[asset_name] = regime_data

                    # Log summary
                    logger.info(f"\n[MTF] {asset_name} Analysis:")
                    logger.info(f"  Regime:           {regime_data['regime'].upper()}")
                    logger.info(
                        f"  Direction:        {'BULL' if regime_data['is_bull'] else 'BEAR'}"
                    )
                    logger.info(f"  Confidence:       {regime_data['confidence']:.2%}")
                    logger.info(
                        f"  TF Agreement:     {regime_data['timeframe_agreement']:.2%}"
                    )
                    logger.info(
                        f"  Recommended Mode: {regime_data['recommended_mode'].upper()}"
                    )
                    logger.info(
                        f"  Risk Level:       {regime_data['risk_level'].upper()}"
                    )
                    logger.info(
                        f"  Volatility:       {regime_data['volatility'].upper()}"
                    )
                    logger.info(
                        f"  Counter-Trend:    {'✓ Allowed' if regime_data['allow_counter_trend'] else '✗ Blocked'}"
                    )
                    logger.info(f"  Max Positions:    {regime_data['max_positions']}")

                except Exception as e:
                    logger.error(f"[MTF] Error analyzing {asset_name}: {e}")

            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"[MTF] Analysis error: {e}", exc_info=True)

    def initialize_mtf_regime_detection(self):
        """
        Initialize multi-timeframe regime detection
        Should be called AFTER AI and DB initialization
        """
        try:
            logger.info("\n" + "=" * 70)
            logger.info("[MTF] Initializing Multi-Timeframe Regime Detection")
            logger.info("=" * 70)

            self.mtf_integration = MTFRegimeIntegration(
                data_manager=self.data_manager,
                db_manager=self.db_manager,
                ai_validator=self.ai_validator,
                telegram_bot=self.telegram_bot,
            )

            logger.info("[MTF] ✅ Multi-Timeframe Regime Detection Ready")
            logger.info("=" * 70 + "\n")

            return True

        except Exception as e:
            logger.error(f"[MTF] Initialization failed: {e}", exc_info=True)
            self.mtf_integration = None
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
        AGGREGATOR_PRESETS = {}
        try:
            CONFIG_PATH = Path("config/aggregator_presets.json")
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                    AGGREGATOR_PRESETS = data.get("AGGREGATOR_PRESETS", {})
                logger.info("[INIT] Loaded aggregator presets from file")
            else:
                logger.warning("[INIT] aggregator_presets.json not found")
        except Exception as e:
            logger.error(f"[INIT] Error loading presets: {e}")

        # ================================================================
        # STEP 3: Auto-Preset Selection (if enabled)
        # ================================================================
        if preset == "auto":
            logger.info("\n🎯 AUTO-PRESET MODE ACTIVE")
            logger.info("Analyzing market conditions for each asset...")

            asset_presets = {}

            # Get preset for each enabled asset
            for asset_name in self.strategies.keys():
                if self.config["assets"][asset_name].get("enabled", False):
                    # Use the preset already calculated during init
                    selected_preset = self.dynamic_selector.current_presets.get(
                        asset_name, "balanced"
                    )
                    asset_presets[asset_name] = selected_preset

            logger.info("\n📊 CURRENT PRESETS:")
            for asset, selected_preset in asset_presets.items():
                logger.info(f"  {asset:6} → {selected_preset.upper()}")

        elif preset in ["conservative", "balanced", "aggressive", "scalper"]:
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
        if hasattr(self, "ai_validator") and self.ai_validator is not None:
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
        if not hasattr(self, "aggregators") or self.aggregators is None:
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

            # Handle asset key mapping (BTCUSDT -> BTC)
            config_key = "BTC" if "BTC" in asset_name.upper() else "GOLD"
            preset_config = AGGREGATOR_PRESETS.get(config_key, {}).get(selected_preset)

            if preset_config is None:
                logger.error(
                    f"\n[ERROR] {asset_name}: No config for preset '{selected_preset}'"
                )
                continue

            logger.info(f"\n{asset_name}:")
            logger.info(f"  Strategies: {strategy_count}")
            logger.info(f"  Preset:     {selected_preset}")

            # ============================================================
            # MODE SELECTION
            # ============================================================

            if mode == "performance":
                # --------------------------------------------------------
                # PERFORMANCE MODE
                # --------------------------------------------------------
                try:
                    self.aggregators[asset_name] = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                        enable_world_class_filters=True,  # Enable filters
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=getattr(
                            self, "detailed_logging", False
                        ),
                        strong_signal_bypass_threshold=getattr(
                            self.params, "ai_strong_signal_bypass", 0.70
                        ),
                    )

                    logger.info(f"  Type:       Performance Aggregator")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )

                except Exception as e:
                    logger.error(
                        f"  [ERROR] Failed to create Performance aggregator: {e}"
                    )
                    continue

            elif mode == "council":
                # --------------------------------------------------------
                # COUNCIL MODE (New institutional aggregator)
                # --------------------------------------------------------
                try:
                    self.aggregators[asset_name] = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        enable_detailed_logging=getattr(
                            self, "detailed_logging", False
                        ),
                        # Council-specific settings
                        config=preset_config,  # ✅ CORRECT: Pass config for dynamic thresholds
                        trend_aligned_threshold=3.5,  # Defaults (will be overridden by config)
                        counter_trend_threshold=4.0,
                    )

                    logger.info(f"  Type:       Council Aggregator")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )
                    # Log the actual active thresholds
                    thresh_trend = preset_config.get("council_trend_aligned", 3.5)
                    thresh_count = preset_config.get("council_counter_trend", 4.0)
                    logger.info(
                        f"  Thresholds: {thresh_trend} (trend) / {thresh_count} (counter)"
                    )

                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Council aggregator: {e}")
                    continue

            elif mode == "hybrid":
                # --------------------------------------------------------
                # HYBRID MODE (Both aggregators for comparison)
                # --------------------------------------------------------
                try:
                    # Create Performance Aggregator
                    perf_agg = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                        enable_world_class_filters=True,  # Enable filters
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=False,  # Reduce noise in hybrid mode
                        strong_signal_bypass_threshold=getattr(
                            self.params, "ai_strong_signal_bypass", 0.70
                        ),
                    )

                    # Create Council Aggregator
                    council_agg = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        enable_detailed_logging=False,
                        config=preset_config,  # ✅ CORRECT: Pass config for dynamic thresholds
                        trend_aligned_threshold=3.5,
                        counter_trend_threshold=4.0,
                    )

                    # Store both in a dict
                    self.aggregators[asset_name] = {
                        "performance": perf_agg,
                        "council": council_agg,
                        "mode": "hybrid",
                    }

                    logger.info(f"  Type:       Hybrid (Both aggregators)")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )
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
            logger.info(
                "\n⚠️  HYBRID MODE: Signals require consensus from both aggregators"
            )
            logger.info(f"  Type:       Hybrid (Dynamic Mode Selection)")
            logger.info(f"  AI:         {'Enabled' if ai_validator else 'Disabled'}")
            logger.info(f"  Note:       Intelligent mode switching enabled")

        if preset == "auto":
            logger.info(f"\n🎯 AUTO PRESET: Active (Dynamic adjustment enabled)")
            logger.info(
                f"  Cooldown:     {self.dynamic_selector.min_switch_interval} minutes"
            )
            logger.info(f"  Presets Used: {set(asset_presets.values())}")

    def _format_ai_validation_direct(self, asset_name: str, signal: int, df: pd.DataFrame) -> dict:
        """
        Direct AI validation formatting (fallback when aggregator doesn't have the method)
        MATCHES the full implementation from _format_ai_validation_for_viz

        Args:
            asset_name: The name of the asset being processed (e.g., 'BTC', 'GOLD').
            signal: Trading signal
            df: Market dataframe

        Returns:
            Formatted AI validation dict with top3 patterns
        """
        try:
            if not self.ai_validator:
                return {
                    "pattern_detected": False,
                    "validation_passed": False,
                    "pattern_name": "N/A",
                    "pattern_confidence": 0.0,
                    "action": "ai_disabled",
                    "error": "AI validator not initialized",
                    "top3_patterns": [],
                    "top3_confidences": [],
                    "sr_analysis": {
                        "near_sr_level": False,
                        "level_type": "none",
                        "nearest_level": None,
                        "distance_pct": None,
                        "levels": [],
                        "total_levels_found": 0,
                    },
                }

            current_price = float(df["close"].iloc[-1])

            # Get S/R analysis
            sr_result = self.ai_validator._check_support_resistance_fixed(
                asset=asset_name,
                df=df,
                current_price=current_price,
                signal=signal,
                threshold=self.ai_validator.current_sr_threshold,
            )

            # Get pattern detection
            pattern_result = self.ai_validator._check_pattern(
                df=df,
                signal=signal,
                min_confidence=self.ai_validator.current_pattern_threshold,
            )

            # ✅ FIX: Get top 3 patterns (was missing!)
            top3_patterns = []
            top3_confidences = []

            if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                try:
                    # Get last 15 candles for pattern detection
                    snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                    first_open = snippet[0, 0]

                    if first_open > 0:
                        snippet_norm = snippet / first_open - 1
                        snippet_input = snippet_norm.reshape(1, 15, 4)

                        # Get predictions
                        predictions = self.ai_validator.sniper.model.predict(
                            snippet_input, verbose=0
                        )[0]

                        # Get top 3
                        top3_indices = predictions.argsort()[-3:][::-1]
                        top3_confidences = predictions[top3_indices].tolist()

                        # Map to pattern names
                        for idx in top3_indices:
                            pattern_name = self.ai_validator.reverse_pattern_map.get(
                                idx, f"Pattern_{idx}"
                            )
                            top3_patterns.append(pattern_name)

                except Exception as e:
                    logger.debug(f"[AI DIRECT] Top3 patterns failed: {e}")

            # Build result
            return {
                "pattern_detected": pattern_result.get("pattern_confirmed", False),
                "validation_passed": signal != 0,  # If signal survived, it passed
                "pattern_name": pattern_result.get("pattern_name", "None"),
                "pattern_id": pattern_result.get("pattern_id"),
                "pattern_confidence": pattern_result.get("confidence", 0.0),
                "top3_patterns": top3_patterns,
                "top3_confidences": top3_confidences,
                "sr_analysis": {
                    "near_sr_level": sr_result.get("near_level", False),
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                },
                "action": "direct_validation",
                "rejection_reasons": [],
                "error": None,
            }

        except Exception as e:
            logger.error(f"[AI DIRECT] Validation failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "ERROR",
                "pattern_confidence": 0.0,
                "action": "error",
                "error": str(e),
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
            }

    def _validate_ai_details_structure(
        self, ai_validation: dict, context: str = ""
    ) -> bool:
        """
        ✅ ENHANCED: Validate AI validation dict with numpy type handling (NumPy 2.0 Safe)
        """
        import numpy as np  # Ensure numpy is imported

        if not ai_validation or not isinstance(ai_validation, dict):
            logger.error(
                f"[AI VIZ {context}] ❌ ai_validation is not a dict: {type(ai_validation)}"
            )
            return False

        required_fields = {
            "pattern_detected": bool,
            "pattern_name": str,
            "pattern_confidence": (int, float),
            "pattern_id": (int, type(None)),
            "top3_patterns": list,
            "top3_confidences": list,
            "sr_analysis": dict,
            "validation_passed": bool,
            "action": str,
        }

        sr_required_fields = {
            "near_sr_level": bool,
            "level_type": str,
            "nearest_level": (int, float, type(None)),
            "distance_pct": (int, float, type(None)),
            "levels": list,
            "total_levels_found": int,
        }

        all_valid = True

        # Check top-level fields
        for field, expected_type in required_fields.items():
            if field not in ai_validation:
                logger.error(f"[AI VIZ {context}] ❌ Missing field: {field}")
                all_valid = False
                continue

            value = ai_validation[field]

            # ✅ FIX: Robust NumPy conversion using duck typing
            # Standard Python types (bool, int, float) DO NOT have .item()
            # NumPy scalars DO have .item()
            if hasattr(value, "item"):
                try:
                    value = value.item()
                    ai_validation[field] = value  # Update in place
                except (ValueError, TypeError):
                    pass  # Ignore if conversion fails

            if not isinstance(value, expected_type):
                # Allow fallback for confidence if it's int/float compatible
                if field == "pattern_confidence" and isinstance(value, (int, float)):
                    pass
                else:
                    logger.error(
                        f"[AI VIZ {context}] ❌ {field} wrong type: "
                        f"expected {expected_type}, got {type(value)}"
                    )
                    all_valid = False

        # Check sr_analysis sub-fields
        if "sr_analysis" in ai_validation:
            sr_analysis = ai_validation["sr_analysis"]

            if not isinstance(sr_analysis, dict):
                logger.error(f"[AI VIZ {context}] ❌ sr_analysis is not a dict")
                all_valid = False
            else:
                for field, expected_type in sr_required_fields.items():
                    if field not in sr_analysis:
                        # Optional fields logic could go here, but for now log missing
                        # logger.error(f"[AI VIZ {context}] ❌ sr_analysis missing: {field}")
                        # all_valid = False
                        pass  # Be lenient on sub-fields to prevent crashes
                    else:
                        value = sr_analysis[field]

                        # ✅ FIX: Handle numpy types in sr_analysis using duck typing
                        if hasattr(value, "item"):
                            try:
                                value = value.item()
                                sr_analysis[field] = value
                            except (ValueError, TypeError):
                                pass

                        if not isinstance(value, expected_type):
                            logger.error(
                                f"[AI VIZ {context}] ❌ sr_analysis.{field} wrong type: "
                                f"expected {expected_type}, got {type(value)}"
                            )
                            all_valid = False

        if all_valid:
            logger.info(f"[AI VIZ {context}] ✅ All fields valid")
        else:
            logger.error(f"[AI VIZ {context}] ❌ Validation FAILED")

        return all_valid

    def _detect_regime(
        self, df: pd.DataFrame, asset_name: str = "BTC"
    ) -> Tuple[bool, float]:
        """
        ✅ ENHANCED: Use MTF regime detection if available, fallback to original logic
        """
        try:
            # PRIORITY 1: Use MTF Regime Detection if available
            if self.mtf_integration and hasattr(self, "_current_regime_data"):
                mtf_regime = self._current_regime_data.get(asset_name)

                if mtf_regime:
                    is_bull = mtf_regime["is_bull"]
                    confidence = mtf_regime["confidence"]

                    logger.debug(f"[REGIME] {asset_name}: Using MTF regime")
                    logger.debug(f"  Direction: {'BULL' if is_bull else 'BEAR'}")
                    logger.debug(f"  Confidence: {confidence:.2%}")

                    # Update previous regime for continuity
                    self.previous_regime = is_bull

                    return is_bull, confidence

            # FALLBACK: Your existing single-timeframe detection
            logger.debug(f"[REGIME] {asset_name}: Using fallback detection")

            # ... YOUR EXISTING _detect_regime CODE HERE ...
            # (Keep all your existing logic as fallback)

        except Exception as e:
            logger.error(f"[REGIME] Detection failed: {e}", exc_info=True)
            # Emergency fallback
            return False, 0.5

    def _log_ai_validation_summary(self, asset_name: str, details: dict):
        """
        ✅ NEW: Log comprehensive AI validation summary
        Call this before sending charts
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[AI VIZ SUMMARY] {asset_name}")
        logger.info(f"{'='*70}")

        ai_validation = details.get("ai_validation")

        if not ai_validation:
            logger.error(f"❌ ai_validation is missing from details")
            logger.error(f"Available keys: {list(details.keys())}")
            return

        if not isinstance(ai_validation, dict):
            logger.error(f"❌ ai_validation is not a dict: {type(ai_validation)}")
            return

        # Pattern info
        pattern_name = ai_validation.get("pattern_name", "N/A")
        pattern_conf = ai_validation.get("pattern_confidence", 0)
        pattern_detected = ai_validation.get("pattern_detected", False)

        logger.info(f"Pattern:")
        logger.info(f"  Name:       {pattern_name}")
        logger.info(f"  Confidence: {pattern_conf:.2%}")
        logger.info(f"  Detected:   {pattern_detected}")

        # Top 3 patterns
        top3 = ai_validation.get("top3_patterns", [])
        top3_conf = ai_validation.get("top3_confidences", [])

        logger.info(f"Top 3 Patterns:")
        if top3:
            for i, (name, conf) in enumerate(zip(top3, top3_conf), 1):
                logger.info(f"  {i}. {name}: {conf:.2%}")
        else:
            logger.warning(f"  ⚠️ No top3 patterns available")

        # S/R Analysis
        sr_analysis = ai_validation.get("sr_analysis", {})

        nearest = sr_analysis.get("nearest_level")
        distance = sr_analysis.get("distance_pct")

        logger.info("S/R Analysis:")
        logger.info(f"  Near Level: {sr_analysis.get('near_sr_level', False)}")
        logger.info(f"  Type:       {sr_analysis.get('level_type', 'N/A')}")
        logger.info(
            f"  Nearest:    ${nearest:,.2f}"
            if isinstance(nearest, (int, float))
            else "  Nearest:    N/A"
        )
        logger.info(
            f"  Distance:   {distance:.2f}%"
            if isinstance(distance, (int, float))
            else "  Distance:   N/A"
        )
        logger.info(f"  Total Levels: {sr_analysis.get('total_levels_found', 0)}")

        # Validation status
        validation_passed = ai_validation.get("validation_passed", False)
        action = ai_validation.get("action", "unknown")

        logger.info(f"Validation:")
        logger.info(f"  Passed: {validation_passed}")
        logger.info(f"  Action: {action}")

        # Rejection reasons (if any)
        rejection_reasons = ai_validation.get("rejection_reasons", [])
        if rejection_reasons:
            logger.info(f"Rejection Reasons:")
            for reason in rejection_reasons:
                logger.info(f"  - {reason}")

        # Error (if any)
        error = ai_validation.get("error")
        if error:
            logger.error(f"❌ Error: {error}")

        # Validate structure
        is_valid = self._validate_ai_details_structure(ai_validation, asset_name)

        logger.info(f"{'='*70}\n")

        return is_valid

    def get_aggregated_signal_hybrid_dynamic(
        self,
        asset_name: str,
        df: pd.DataFrame,
        aggregators: Dict,
        hybrid_selector,
    ) -> Tuple[int, Dict]:
        """
        ✅ FIXED: Ensures AI validation details are ALWAYS populated
        ✅ FIXED: Injects MTF Governor data into Council Aggregator
        """

        # ================================================================
        # STEP 1: Determine optimal aggregator mode
        # ================================================================
        mode_info = hybrid_selector.get_optimal_mode(asset_name, df)

        selected_mode = mode_info["mode"]
        confidence = mode_info["confidence"]
        switch_occurred = mode_info["switch_occurred"]
        analysis = mode_info["analysis"]

        # Log mode selection
        if switch_occurred:
            logger.info(f"\n{'='*70}")
            logger.info(f"[HYBRID] {asset_name}: MODE SWITCH → {selected_mode.upper()}")
            logger.info(f"{'='*70}")
        else:
            logger.debug(f"[HYBRID] {asset_name}: Using {selected_mode.upper()} mode")

        # ================================================================
        # STEP 2: Get signal from selected aggregator
        # ================================================================
        if selected_mode == "council":
            aggregator = aggregators["council"]

            # ✅ NEW: Fetch the latest MTF regime data for the Council
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

            # ✅ FIXED: Pass full market context to the Institutional Council
            signal, details = aggregator.get_aggregated_signal(
                df,
                current_regime=mtf_regime.get("regime", "NEUTRAL"),
                is_bull_market=mtf_regime.get("is_bull", False),
                governor_data=mtf_regime,  # This contains the trade_type needed for Asymmetry
            )

            logger.info(
                f"[COUNCIL] Total Score: {details.get('total_score', 0):.2f}/5.0"
            )
            logger.info(f"[COUNCIL] Decision: {details.get('decision_type', 'N/A')}")

        else:  # performance mode
            aggregator = aggregators["performance"]
            # Fetch the latest MTF regime data to pass to the aggregator
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

            signal, details = aggregator.get_aggregated_signal(
                df,
                current_regime=mtf_regime.get("regime", "NEUTRAL"),
                is_bull_market=mtf_regime.get("is_bull", False),
                governor_data=mtf_regime,
            )

            logger.info(
                f"[PERFORMANCE] Signal Quality: {details.get('signal_quality', 0):.2%}"
            )
            logger.info(f"[PERFORMANCE] Reasoning: {details.get('reasoning', 'N/A')}")

        # ================================================================
        # ✅ FIX: ALWAYS format AI validation (don't rely on aggregator)
        # ================================================================
        ai_validation = details.get("ai_validation")

        if ai_validation is None or not isinstance(ai_validation, dict):
            logger.warning(
                f"[HYBRID] ⚠️ No AI validation from {selected_mode} aggregator, "
                f"generating manually..."
            )

            # Get the actual aggregator instance
            actual_aggregator = aggregators.get(selected_mode)

            # Try aggregator's method first
            if actual_aggregator and hasattr(
                actual_aggregator, "_format_ai_validation_for_viz"
            ):
                try:
                    ai_validation = actual_aggregator._format_ai_validation_for_viz(
                        final_signal=signal, details=details.copy(), df=df
                    )
                    logger.info(
                        f"[HYBRID] ✅ AI validation from {selected_mode} aggregator"
                    )
                except Exception as e:
                    logger.error(
                        f"[HYBRID] Aggregator method failed: {e}, using fallback"
                    )
                    ai_validation = None

            # Fallback: Use direct AI validation
            if ai_validation is None:
                logger.info(f"[HYBRID] Using direct AI validation fallback")
                ai_validation = self._format_ai_validation_direct(signal, df)

            # Store in details
            details["ai_validation"] = ai_validation

        else:
            logger.info(
                f"[HYBRID] ✅ AI validation present from {selected_mode} aggregator"
            )
            logger.debug(
                f"[HYBRID] Pattern: {ai_validation.get('pattern_name', 'N/A')}, "
                f"Confidence: {ai_validation.get('pattern_confidence', 0):.2%}"
            )

        # ================================================================
        # STEP 3: Verify AI validation has all required fields
        # ================================================================
        required_fields = [
            "pattern_detected",
            "pattern_name",
            "pattern_confidence",
            "top3_patterns",
            "top3_confidences",
            "sr_analysis",
            "validation_passed",
            "action",
        ]

        missing_fields = [
            field for field in required_fields if field not in ai_validation
        ]

        if missing_fields:
            logger.warning(f"[HYBRID] ⚠️ AI validation missing fields: {missing_fields}")
            logger.warning(f"[HYBRID] Regenerating complete AI validation...")

            # Regenerate completely
            ai_validation = self._format_ai_validation_direct(asset_name, signal, df)
            details["ai_validation"] = ai_validation

        # ================================================================
        # STEP 4: Get current price
        # ================================================================
        try:
            current_price = float(df["close"].iloc[-1])
        except:
            current_price = 0.0

        # ================================================================
        # STEP 5: Calculate adaptive TP/SL if signal is not HOLD
        # ================================================================
        tp_sl_info = None

        if signal != 0:
            try:
                tp_sl_info = hybrid_selector.calculate_tp_sl(
                    asset_name=asset_name,
                    entry_price=current_price,
                    signal=signal,
                    df=df,
                    mode=selected_mode,
                    confidence=confidence,
                )

                logger.info(
                    f"\n[TP/SL] Adaptive Levels ({selected_mode.upper()} mode):"
                )
                logger.info(f"  Entry:         ${current_price:,.2f}")
                logger.info(f"  Stop Loss:     ${tp_sl_info['stop_loss']:,.2f}")
                logger.info(f"  Take Profit:   ${tp_sl_info['take_profit']:,.2f}")
                logger.info(f"  Risk/Reward:   {tp_sl_info['risk_reward_ratio']:.2f}:1")

            except Exception as e:
                logger.error(f"[TP/SL] Calculation failed: {e}")

        # ================================================================
        # STEP 6: Build merged details
        # ================================================================
        merged_details = details.copy()

        # Add hybrid-specific metadata
        hybrid_metadata = {
            "aggregator_mode": selected_mode,
            "mode_confidence": confidence,
            "mode_switched": switch_occurred,
            "regime_analysis": {
                "regime_type": analysis["regime_type"],
                "trend_strength": analysis["trend"]["strength"],
                "trend_direction": analysis["trend"]["direction"],
                "adx": analysis["trend"]["adx"],
                "volatility_regime": analysis["volatility"]["regime"],
                "volatility_ratio": analysis["volatility"]["ratio"],
                "price_clarity": analysis["price_action"]["clarity"],
                "momentum_aligned": analysis.get("momentum_aligned", False),
                "at_key_level": analysis.get("at_key_level", False),
            },
            "adaptive_tpsl": tp_sl_info,
            "signal_quality": max(details.get("signal_quality", 0), confidence * 0.8),
        }

        merged_details.update(hybrid_metadata)

        # ================================================================
        # ✅ CRITICAL: Verify ai_validation is in merged_details
        # ================================================================
        if "ai_validation" not in merged_details:
            logger.error(f"[HYBRID] ❌ CRITICAL: ai_validation lost during merge!")
            # Last resort: add placeholder
            merged_details["ai_validation"] = {
                "pattern_detected": False,
                "pattern_name": "ERROR",
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "validation_passed": False,
                "action": "error_lost_validation",
                "error": "AI validation lost during hybrid merge",
            }

        # ================================================================
        # STEP 7: Final validation check
        # ================================================================
        final_ai_validation = merged_details.get("ai_validation")

        if not final_ai_validation or not isinstance(final_ai_validation, dict):
            logger.error(f"[HYBRID] ❌ FINAL CHECK FAILED: ai_validation is invalid")
        else:
            logger.info(f"[HYBRID] ✅ Final AI validation verified:")
            logger.info(f"  Pattern: {final_ai_validation.get('pattern_name', 'N/A')}")
            logger.info(
                f"  Confidence: {final_ai_validation.get('pattern_confidence', 0):.2%}"
            )
            logger.info(f"  Action: {final_ai_validation.get('action', 'N/A')}")
            logger.info(
                f"  Top3: {len(final_ai_validation.get('top3_patterns', []))} patterns"
            )

        # ================================================================
        # STEP 8: Apply mode-specific quality filters
        # ================================================================
        if selected_mode == "council":
            min_score = merged_details.get("required_score", 3.5)
            actual_score = merged_details.get("total_score", 0)

            if signal != 0 and actual_score < min_score:
                logger.info(
                    f"[COUNCIL] Signal filtered: {actual_score:.2f} < {min_score:.2f}"
                )
                signal = 0
                merged_details["reasoning"] = (
                    f"Council score too low ({actual_score:.2f}/{min_score:.2f})"
                )

        elif selected_mode == "performance":
            min_quality = 0.28
            actual_quality = merged_details.get("signal_quality", 0)

            if signal != 0 and actual_quality < min_quality:
                logger.info(
                    f"[PERFORMANCE] Signal filtered: {actual_quality:.2%} < {min_quality:.2%}"
                )
                signal = 0
                merged_details["reasoning"] = (
                    f"Signal quality too low ({actual_quality:.2%})"
                )

        # Update signal in details
        merged_details["signal"] = signal

        return signal, merged_details

    def _signal_str(self, signal: int) -> str:
        """Convert signal to readable string"""
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")

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

        # ✅ NEW: Set initial presets using DynamicPresetSelector
        if self.config.get("aggregator_settings", {}).get("preset") == "auto":
            selector = DynamicPresetSelector(self.data_manager, self.config)

            # Get preset for EACH enabled asset
            self.selected_presets = {}
            for asset_name in self.strategies.keys():
                preset = selector.get_preset_for_asset(asset_name)
                self.selected_presets[asset_name] = preset
                self.dynamic_selector.current_presets[asset_name] = preset

        else:
            # Manual preset
            preset = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            self.selected_presets = {name: preset for name in self.strategies.keys()}

            for asset in self.selected_presets:
                self.dynamic_selector.current_presets[asset] = preset

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

    def initialize_autotrainer(self):
        """Initializes and starts the continuous learning pipeline."""
        if self.config.get("ml", {}).get("enable_autotrain", False):
            logger.info("\n" + "=" * 70)
            logger.info("INITIALIZING AUTO-TRAINER")
            logger.info("=" * 70)
            try:
                self.autotrainer = ContinuousLearningPipeline(
                    config=self.config,
                    trading_bot=self,
                    telegram_bot=self.telegram_bot
                )
                self.autotrainer.start()
            except Exception as e:
                logger.error(f"[AUTO-TRAIN] Failed to initialize: {e}", exc_info=True)
                self.autotrainer = None
        else:
            logger.info("[AUTO-TRAIN] Disabled in config.")

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
        """Run Telegram in its dedicated loop"""
        try:
            logger.info("[TELEGRAM] Starting bot...")
            
            # ✅ FIX: Start the dedicated loop FIRST
            if not self.telegram_bot._loop or self.telegram_bot._loop.is_closed():
                logger.info("[TELEGRAM] Creating new event loop...")
                self.telegram_bot.start_loop_thread()  # This creates _loop
            
            # Now the loop exists and is ready
            loop = self.telegram_bot._loop
            
            if not loop or loop.is_closed():
                logger.error("[TELEGRAM] Loop creation failed")
                return
            
            # Run initialization in the dedicated loop
            future = asyncio.run_coroutine_threadsafe(
                self.telegram_bot.initialize(), 
                loop
            )
            
            try:
                future.result(timeout=60)
                logger.info("[TELEGRAM] Bot initialized successfully")
                self._telegram_ready.set()
            except TimeoutError:
                logger.error("[TELEGRAM] Initialization timeout")
                self._telegram_ready.clear()
            except Exception as e:
                logger.error(f"[TELEGRAM] Initialization error: {e}")
                self._telegram_ready.clear()
            
        except Exception as e:
            logger.error(f"[TELEGRAM] Loop error: {e}", exc_info=True)
            self._telegram_ready.clear()

    def _send_telegram_notification(self, coro):
        """
        ✅ FIXED: Send notification safely from main thread to Telegram's event loop
        """
        if not self.telegram_bot or not self.telegram_bot._is_ready:
            logger.debug("[TELEGRAM] Bot not ready, queueing notification")
            if hasattr(self.telegram_bot, '_message_queue'):
                self.telegram_bot._message_queue.append(str(coro))
            return

        try:
            # ✅ FIX: Get the correct event loop from the Application
            if not self.telegram_bot.application:
                logger.warning("[TELEGRAM] Application not available")
                return

            # The Application runs in its own loop created by updater.start_polling()
            # We need to get that loop's reference
            
            # Option 1: Use the updater's loop if available
            loop = None
            
            if hasattr(self.telegram_bot.application, 'updater') and self.telegram_bot.application.updater:
                # Try to get the loop from the updater
                if hasattr(self.telegram_bot.application.updater, '_loop'):
                    loop = self.telegram_bot.application.updater._loop
            
            # Option 2: Try to detect the running loop
            if not loop:
                try:
                    # Get the loop that's currently running the bot
                    import asyncio
                    import threading
                    
                    # Store loop reference when bot starts
                    if hasattr(self.telegram_bot, '_application_loop'):
                        loop = self.telegram_bot._application_loop
                    else:
                        logger.warning("[TELEGRAM] Could not find application loop")
                        return
                except Exception as e:
                    logger.error(f"[TELEGRAM] Loop detection error: {e}")
                    return

            if not loop or loop.is_closed():
                logger.warning("[TELEGRAM] Event loop is closed or unavailable")
                return

            # ✅ Submit coroutine to the bot's event loop
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            
            # Wait for completion with timeout
            try:
                future.result(timeout=15.0)
                logger.debug("[TELEGRAM] Notification sent successfully")
            except TimeoutError:
                logger.warning("[TELEGRAM] Notification timeout (15s)")
                future.cancel()
            except Exception as e:
                logger.error(f"[TELEGRAM] Notification error: {e}")

        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to send notification: {e}", exc_info=True)

    def _reinitialize_aggregator(self, asset_name: str, preset: str):
        """
        Reinitialize aggregator with new preset
        ✅ ENHANCED: Better logging for auto preset changes
        """
        try:
            # Load Presets
            CONFIG_PATH = Path("config/aggregator_presets.json")
            with open(CONFIG_PATH, "r") as f:
                AGGREGATOR_PRESETS = json.load(f)["AGGREGATOR_PRESETS"]
            # Get strategies for this asset
            strategies = self.strategies.get(asset_name, {})
            if not strategies:
                logger.warning(f"[AUTO PRESET] No strategies for {asset_name}")
                return

            # Get preset config
            asset_type = "BTC" if "BTC" in asset_name.upper() else "GOLD"
            preset_config = AGGREGATOR_PRESETS.get(asset_type, {}).get(preset)

            if not preset_config:
                logger.error(f"[AUTO PRESET] No config for {asset_name} {preset}")
                return

            # Get AI validator if available
            ai_validator = None
            if hasattr(self, "ai_validator") and self.ai_validator:
                ai_validator = self.ai_validator

            # Determine aggregator mode from config
            global_mode = (
                self.config.get("aggregator_settings", {})
                .get("mode", "performance")
                .lower()
            )

            # Check current aggregator state
            current_aggregator = self.aggregators.get(asset_name)
            is_hybrid_state = (
                isinstance(current_aggregator, dict)
                and current_aggregator.get("mode") == "hybrid"
            )

            # Force hybrid if config says so OR state says so
            if global_mode == "hybrid" or is_hybrid_state:
                mode_to_init = "hybrid"
            elif global_mode == "council":
                mode_to_init = "council"
            else:
                mode_to_init = "performance"

            logger.info(
                f"[AUTO PRESET] Reinitializing {asset_name}\n"
                f"  Preset: {preset.upper()}\n"
                f"  Mode:   {mode_to_init.upper()}"
            )

            # ================================================================
            # RE-INITIALIZE BASED ON MODE
            # ================================================================

            if mode_to_init == "hybrid":
                # HYBRID MODE: Recreate both
                perf_agg = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    config=preset_config,
                    ai_validator=ai_validator,
                    mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                    enable_world_class_filters=True,  # Enable filters
                    enable_ai_circuit_breaker=True,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    strong_signal_bypass_threshold=getattr(
                        self.params, "ai_strong_signal_bypass", 0.70
                    ),
                )
                council_agg = InstitutionalCouncilAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    ai_validator=ai_validator,
                    enable_detailed_logging=False,
                    config=preset_config,
                    trend_aligned_threshold=3.5,
                    counter_trend_threshold=4.0,
                )

                self.aggregators[asset_name] = {
                    "performance": perf_agg,
                    "council": council_agg,
                    "mode": "hybrid",
                }
                logger.info(
                    f"[AUTO PRESET] ✓ Hybrid aggregators refreshed for {asset_name}"
                )

            elif mode_to_init == "council":
                # COUNCIL MODE
                new_aggregator = InstitutionalCouncilAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    ai_validator=ai_validator,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    config=preset_config,
                    trend_aligned_threshold=3.5,
                    counter_trend_threshold=4.0,
                )
                self.aggregators[asset_name] = new_aggregator
                logger.info(
                    f"[AUTO PRESET] ✓ Council aggregator refreshed for {asset_name}"
                )

            else:
                # PERFORMANCE MODE
                new_aggregator = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    config=preset_config,
                    ai_validator=ai_validator,
                    mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                    enable_world_class_filters=True,  # Enable filters
                    enable_ai_circuit_breaker=True,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    strong_signal_bypass_threshold=getattr(
                        self.params, "ai_strong_signal_bypass", 0.70
                    ),
                )
                self.aggregators[asset_name] = new_aggregator
                logger.info(
                    f"[AUTO PRESET] ✓ Performance aggregator refreshed for {asset_name}"
                )

        except Exception as e:
            logger.error(f"[AUTO PRESET] Aggregator reinit error: {e}", exc_info=True)

    def _update_dynamic_presets(self):
        """
        Check market conditions and update presets if regime changed
        """
        try:
            logger.info("\n[REGIME CHECK] Analyzing market conditions...")

            enabled_assets = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            preset_changed = False
            changes = []
            for asset_name in enabled_assets:
                # Get optimal preset for current market conditions
                new_preset = self.dynamic_selector.get_preset_for_asset(asset_name)

                if new_preset:
                    old_preset = self.selected_presets.get(asset_name)

                    # If preset changed, reinitialize aggregator
                    if old_preset != new_preset:
                        logger.info(
                            f"[AUTO PRESET] {asset_name}: {old_preset.upper()} → {new_preset.upper()}"
                        )

                        self.selected_presets[asset_name] = new_preset
                        self._reinitialize_aggregator(asset_name, new_preset)
                        preset_changed = True
                        changes.append(f"{asset_name}: {old_preset} → {new_preset}")

                    if preset_changed:
                        logger.info(f"[AUTO PRESET] ✓ Updated {len(changes)} preset(s)")
                        for change in changes:
                            logger.info(f"  • {change}")

                        # Log statistics
                        stats = self.dynamic_selector.get_statistics()
                        logger.info(
                            f"[AUTO PRESET] Total preset changes: {stats['total_changes']}"
                        )
                        logger.info(
                            f"[AUTO PRESET] Distribution: {stats['preset_distribution']}"
                        )
                    else:
                        logger.debug("[AUTO PRESET] No preset changes needed")

        except Exception as e:
            logger.error(f"[REGIME] Update error: {e}", exc_info=True)

    # ✨  Trading cycle with better error handling
    @handle_errors(
        component="main_trading_loop",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def run_trading_cycle(self):
        """Execute one complete trading cycle with VTM support"""
        try:
            logger.info("\n" + "=" * 70)
            logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            # ✨ NEW: Update historical data every hour (or every 12 cycles if running every 5 min)
            current_time = datetime.now()
            if (
                self._last_history_update is None
                or (current_time - self._last_history_update).total_seconds() > 3600
            ):  # 1 hour

                logger.info("[HISTORY] Updating historical CSV files...")
                try:
                    self.historical_updater.update_all_enabled_assets()
                    self._last_history_update = current_time
                except Exception as e:
                    logger.error(f"[HISTORY] Update failed: {e}")

            preset_mode = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            if preset_mode == "auto":
                self._update_dynamic_presets()
            # Refresh capital if live
            if not self.portfolio_manager.is_paper_mode:
                try:
                    self.portfolio_manager.refresh_capital()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh capital: {e}")

            self.reset_daily_counters()
            self._check_VTM_positions()
            self._consecutive_errors = 0
            self._last_successful_cycle = datetime.now()

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

            if hasattr(self, "data_manager_telegram"):
                self.data_manager_telegram.update_snapshot(self)
                self.data_manager_telegram.process_queued_commands(self)

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

            # ✅ NEW: Log hybrid statistics periodically
            # ✅ NEW: Log hybrid statistics periodically
            if hasattr(self, "hybrid_selector"):
                stats = self.hybrid_selector.get_statistics()

                # ✅ FIXED: Actively query the current modes for the dashboard/logs
                active_modes = {}
                for asset in self.config["assets"]:
                    if self.config["assets"][asset].get("enabled", False):
                        # Get mode without logging the switch to avoid log spam
                        mode_info = self.hybrid_selector.get_optimal_mode(
                            asset, pd.DataFrame()
                        )
                        active_modes[asset] = mode_info["mode"]

                logger.info(f"\n[HYBRID STATS]")
                logger.info(f"  Total Switches:      {stats['total_switches']}")
                logger.info(f"  Council Signals:     {stats['council_signals']}")
                logger.info(f"  Performance Signals: {stats['performance_signals']}")
                logger.info(
                    f"  Current Modes:       {active_modes}"
                )  # <-- Now uses active_modes

                logger.info("[OK] Trading cycle complete")
                logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[ERROR] Cycle failed: {e}", exc_info=True)
            self._consecutive_errors += 1

            # Send critical alert if too many failures
            if self._consecutive_errors >= self._max_consecutive_errors:
                if self.error_handler:
                    self.error_handler.handle_error(
                        exception=e,
                        component="main_trading_loop",
                        severity=ErrorSeverity.CRITICAL,
                        additional_info={
                            "consecutive_errors": self._consecutive_errors,
                            "last_successful": self._last_successful_cycle,
                        },
                        notify=True,
                    )

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
        """Log Veteran Trade Manager status for all positions"""
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

    @handle_errors(
        component="trade_asset",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def trade_asset(self, asset_name: str):
        """
        ✅ FIXED: Execute trading logic with proper MTF filtering for ALL aggregator types
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
            logger.info(f"[TRADE ASSET] Processing {asset_name}")
            logger.info(f"{'-' * 70}")

            # ============================================================
            # 1. Fetch FRESH Data & Signal
            # ============================================================
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
                logger.warning(
                    f"[SKIP] {asset_name}: Insufficient data ({len(df)}/250)"
                )
                return

            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[SKIP] {asset_name}: No aggregator available")
                return

            # ============================================================
            # 2. Generate Signal (Hybrid or Single)
            # ============================================================
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

            if isinstance(aggregator, dict) and aggregator.get("mode") == "hybrid":
                signal, details = self.get_aggregated_signal_hybrid_dynamic(
                    asset_name,
                    df,
                    aggregators=aggregator,
                    hybrid_selector=self.hybrid_selector,
                )
            else:
                if isinstance(aggregator, InstitutionalCouncilAggregator):
                    signal, details = aggregator.get_aggregated_signal(
                        df,
                        current_regime=mtf_regime.get("regime", "NEUTRAL"),
                        is_bull_market=mtf_regime.get("is_bull", False),
                        governor_data=mtf_regime,
                    )
                else:
                    # Performance Aggregator handles its own context internally
                    signal, details = aggregator.get_aggregated_signal(df)

            # Get current price
            try:
                # ✅ CRITICAL: Force a live price fetch ONLY at the moment of execution
                current_price = handler.get_current_price(symbol, force_live=True)
            except:
                current_price = float(df["close"].iloc[-1])

            details["price"] = current_price

            # Log Signal Quality
            logger.info(
                f"[SIGNAL] {asset_name} Signal: {signal} "
                f"(Quality: {details.get('signal_quality', 0):.2f})"
            )

            if details.get("aggregator_mode"):
                logger.info(
                    f"  Mode: {details['aggregator_mode'].upper()} | "
                    f"Conf: {details.get('mode_confidence', 0):.2%}"
                )

            # ============================================================
            # ✅ FIX: Apply MTF Filtering AFTER signal generation
            # This applies to ALL aggregator types (hybrid, council, performance)
            # ============================================================
            if (
                signal != 0
                and hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

                logger.info(f"\n[MTF FILTER] Checking regime filters:")
                logger.info(f"  Current Regime: {mtf_regime['regime'].upper()}")
                logger.info(
                    f"  Direction:      {'BULL' if mtf_regime['is_bull'] else 'BEAR'}"
                )
                logger.info(f"  Confidence:     {mtf_regime['confidence']:.2%}")
                logger.info(
                    f"  Recommended:    {mtf_regime['recommended_mode'].upper()}"
                )
                logger.info(f"  Risk Level:     {mtf_regime['risk_level'].upper()}")

                # --------------------------------------------------------
                # Filter 1: Counter-trend blocking
                # --------------------------------------------------------
                if not mtf_regime.get("allow_counter_trend", True):
                    is_counter_trend = (signal == 1 and not mtf_regime["is_bull"]) or (
                        signal == -1 and mtf_regime["is_bull"]
                    )

                    if is_counter_trend:
                        logger.warning(f"[MTF FILTER] ✗ BLOCKED: Counter-trend trade")
                        logger.info(
                            f"  Signal Direction: {'LONG' if signal == 1 else 'SHORT'}"
                        )
                        logger.info(
                            f"  MTF Regime:       {'BULL' if mtf_regime['is_bull'] else 'BEAR'}"
                        )
                        logger.info(
                            f"  Reason:           MTF confidence {mtf_regime['confidence']:.2%} "
                            f"blocks counter-trend"
                        )
                        return  # ← Block the trade

                # --------------------------------------------------------
                # Filter 2: Max positions limit
                # --------------------------------------------------------
                max_positions = mtf_regime.get("max_positions", 3)
                current_positions = len(
                    self.portfolio_manager.get_asset_positions(asset_name)
                )

                if current_positions >= max_positions:
                    logger.warning(f"[MTF FILTER] ✗ BLOCKED: Max positions reached")
                    logger.info(
                        f"  Current: {current_positions}, Max: {max_positions} "
                        f"(MTF Risk: {mtf_regime['risk_level'].upper()})"
                    )
                    return  # ← Block the trade

                # --------------------------------------------------------
                # Filter 3: High risk adjustment
                # --------------------------------------------------------
                if mtf_regime.get("risk_level") == "high":
                    logger.info(
                        f"[MTF FILTER] ⚠️  High risk regime - position size reduced to 70%"
                    )
                    details["mtf_risk_multiplier"] = 0.7
                elif mtf_regime.get("risk_level") == "low":
                    logger.info(
                        f"[MTF FILTER] ✅ Low risk regime - position size increased to 120%"
                    )
                    details["mtf_risk_multiplier"] = 1.2
                else:
                    details["mtf_risk_multiplier"] = 1.0

                # --------------------------------------------------------
                # Filter 4: Volatility adjustment
                # --------------------------------------------------------
                if mtf_regime.get("volatility") == "high":
                    logger.info(
                        f"[MTF FILTER] ⚠️  High volatility - wider stops recommended"
                    )
                    details["mtf_volatility_adjustment"] = 1.5
                else:
                    details["mtf_volatility_adjustment"] = 1.0

                # Add complete MTF data to signal details
                details["mtf_regime"] = mtf_regime

                logger.info(f"[MTF FILTER] ✓ All filters passed")

            elif signal != 0:
                logger.debug(
                    "[MTF FILTER] No MTF data available, skipping regime filters"
                )

            # ============================================================
            # 3. Check HOLD Signal
            # ============================================================
            if signal == 0:
                logger.info(f"[HOLD] {asset_name}: No action taken")
                return

            # ============================================================
            # 4. Check Trading Limits & Cooldowns
            # ============================================================
            if not self.check_trading_limits():
                logger.info(f"[LIMIT] Trading limits reached")
                return

            if not self.check_min_time_between_trades(asset_name):
                logger.info(f"[COOLDOWN] {asset_name} is in cooldown")
                return

            # Store BEFORE state
            positions_before = self.portfolio_manager.get_asset_positions(asset_name)
            position_ids_before = {p.position_id for p in positions_before}

            # ============================================================
            # 5. Execute Trade
            # ============================================================
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
                        signal_details=details,
                    )
                    self.binance_handler.check_and_update_positions(asset_name)
                else:
                    success = self.mt5_handler.execute_signal(
                        signal=signal,
                        symbol=symbol,
                        asset_name=asset_name,
                        confidence_score=details.get("signal_quality", 0.5),
                        market_condition=(
                            "bull" if details.get("regime") == "🚀 BULL" else "bear"
                        ),
                        signal_details=details,
                    )
                    self.mt5_handler.check_and_update_positions(asset_name)
            except Exception as e:
                logger.error(f"[ERROR] Failed to execute signal for {asset_name}: {e}")
                return

            # ============================================================
            # 6. Handle Success & DB Logging
            # ============================================================
            if success:
                positions_after = self.portfolio_manager.get_asset_positions(asset_name)
                position_ids_after = {p.position_id for p in positions_after}
                new_position_ids = position_ids_after - position_ids_before
                closed_position_ids = position_ids_before - position_ids_after

                # Update internal counters
                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()

                logger.info(
                    f"[SUCCESS] {asset_name} Trade Executed "
                    f"(Daily count: {self.trade_count_today})"
                )

                # Send Telegram Notifications (New Positions)
                if new_position_ids:
                    for position_id in new_position_ids:
                        new_pos = next(
                            (
                                p
                                for p in positions_after
                                if p.position_id == position_id
                            ),
                            None,
                        )
                        if (
                            new_pos
                            and self.telegram_bot
                            and self._telegram_ready.is_set()
                        ):
                            try:
                                leverage = getattr(new_pos, "leverage", 1)
                                margin_type = getattr(new_pos, "margin_type", "FUTURES")
                                is_futures = getattr(new_pos, "is_futures", False)

                                sl = new_pos.stop_loss if new_pos.stop_loss else 0.0
                                tp = new_pos.take_profit if new_pos.take_profit else 0.0

                                logger.info(
                                    f"[TELEGRAM] Sending notification for {asset_name}:\n"
                                    f"  Leverage:    {leverage}\n"
                                    f"  Margin Type: {margin_type}\n"
                                    f"  Is Futures:  {is_futures}\n"
                                    f"  Stop Loss:   ${sl:,.2f}\n"
                                    f"  Take Profit: ${tp:,.2f}"
                                )

                                vtm_is_active = new_pos.trade_manager is not None

                                self._send_telegram_notification(
                                    self.telegram_bot.notify_trade_opened(
                                        asset=asset_name,
                                        side=new_pos.side,
                                        price=new_pos.entry_price,
                                        size=new_pos.quantity * new_pos.entry_price,
                                        sl=sl,
                                        tp=tp,
                                        leverage=leverage,
                                        margin_type=margin_type,
                                        is_futures=is_futures,
                                        vtm_is_active=vtm_is_active,
                                    )
                                )

                                logger.info(
                                    f"[TELEGRAM] ✓ Trade opened notification sent"
                                )

                            except Exception as e:
                                logger.error(
                                    f"[TELEGRAM] Notification failed: {e}",
                                    exc_info=True,
                                )

                # Send Notifications (Closed Positions)
                if closed_position_ids:
                    closed_trades = self.portfolio_manager.closed_positions
                    for position_id in closed_position_ids:
                        matching_trade = next(
                            (
                                t
                                for t in reversed(closed_trades)
                                if t.get("position_id") == position_id
                            ),
                            None,
                        )
                        if (
                            matching_trade
                            and self.telegram_bot
                            and self._telegram_ready.is_set()
                        ):
                            try:
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
                                logger.error(
                                    f"[TELEGRAM] Close notification failed: {e}"
                                )

                # Send Visualization Chart
                if new_position_ids and self.chart_sender:
                    try:
                        logger.info(f"[VIZ] Preparing chart for {asset_name}...")

                        df_4h = self._fetch_4h_data(asset_name)

                        logger.info(f"[VIZ] Validating AI details structure...")
                        is_valid = self._log_ai_validation_summary(asset_name, details)

                        if not is_valid:
                            logger.error(
                                f"[VIZ] ⚠️ AI validation structure invalid, "
                                f"attempting repair..."
                            )

                            if not details.get("ai_validation") or not isinstance(
                                details["ai_validation"], dict
                            ):
                                logger.info(
                                    f"[VIZ] Regenerating AI validation from scratch..."
                                )
                                details["ai_validation"] = (
                                    self._format_ai_validation_direct(asset_name, signal, df)
                                )

                                is_valid = self._validate_ai_details_structure(
                                    details["ai_validation"], asset_name
                                )

                                if is_valid:
                                    logger.info(f"[VIZ] ✅ Repair successful")
                                else:
                                    logger.error(
                                        f"[VIZ] ❌ Repair failed, chart may be incomplete"
                                    )

                        logger.info(f"[VIZ] Sending chart to Telegram...")
                        self._send_telegram_notification(
                            self.chart_sender.send_decision_chart(
                                asset_name=asset_name,
                                df_15min=df,
                                df_4h=df_4h,
                                signal=signal,
                                details=details,
                                current_price=current_price,
                            )
                        )

                        logger.info(f"[VIZ] ✅ Chart sent successfully")

                    except Exception as e:
                        logger.error(
                            f"[VIZ] Chart generation error: {e}", exc_info=True
                        )

                # ✅ DATABASE LOGGING (Only on Success)
                if self.db_manager:
                    try:
                        logger.info(
                            f"[DB] Logging successful trade execution for {asset_name}"
                        )

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
                            executed=True,
                        )

                        # Link signal to trade ID if available
                        if new_position_ids and signal_id is not None:
                            new_pos_id = list(new_position_ids)[0]
                            new_pos = next(
                                (
                                    p
                                    for p in positions_after
                                    if p.position_id == new_pos_id
                                ),
                                None,
                            )
                            if (
                                new_pos
                                and hasattr(new_pos, "db_trade_id")
                                and new_pos.db_trade_id
                            ):
                                self.db_manager.update_signal_execution(
                                    signal_id=signal_id,
                                    executed=True,
                                    trade_id=new_pos.db_trade_id,
                                )

                    except Exception as e:
                        logger.error(f"[DB] Failed to log execution: {e}")

                # Log trade to local file
                if self.config.get("logging", {}).get("save_trades", True):
                    self._log_trade(asset_name, signal, details, current_price)

            else:
                logger.warning(
                    f"[SKIP] {asset_name}: Execution returned False "
                    f"(limits/cooldowns/handler failure)"
                )

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
                        self.telegram_bot.notify_error(
                            f"Error in {asset_name}:\n{str(e)[:200]}"
                        )
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
        Update signal for an asset (handles all aggregator modes)
        ✅ FIXED: Now handles hybrid mode correctly
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

            # Get aggregator
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.debug(f"[SIGNAL] {asset_name}: No aggregator")
                return

            # ================================================================
            # ✅ FIX: Handle hybrid vs single aggregator mode
            # ================================================================
            # ================================================================
            # ✅ FIX: Handle hybrid vs single aggregator mode with context
            # ================================================================
            # Get latest MTF data
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

            if isinstance(aggregator, dict) and aggregator.get("mode") == "hybrid":
                # HYBRID MODE: Use dynamic selector
                signal, details = self.get_aggregated_signal_hybrid_dynamic(
                    asset_name=asset_name,
                    df=df,
                    aggregators=aggregator,
                    hybrid_selector=self.hybrid_selector,
                )
            else:
                # SINGLE AGGREGATOR MODE:
                if isinstance(aggregator, InstitutionalCouncilAggregator):
                    # ✅ FIXED: Pass context to pure Council mode for monitoring
                    signal, details = aggregator.get_aggregated_signal(
                        df,
                        current_regime=mtf_regime.get("regime", "NEUTRAL"),
                        is_bull_market=mtf_regime.get("is_bull", False),
                        governor_data=mtf_regime,
                    )
                else:
                    # Performance mode
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
                # Add regime details to the 'details' dictionary for SignalMonitoringIntegration
                details["regime_score"] = mtf_regime.get("regime_score")
                details["regime_is_bullish"] = mtf_regime.get("is_bullish")
                details["regime_is_bearish"] = mtf_regime.get("is_bearish")
                details["regime_reasoning"] = mtf_regime.get("reasoning")
                
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
            # logger.info(f"  Total Trades: {status['total_trades']}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error generating P&L report: {e}", exc_info=True)

    def _restart_telegram_thread(self):
        """
        ✅ FIXED: Properly shutdown and restart with fresh loop
        """
        try:
            logger.info("[TELEGRAM] Attempting to restart bot...")

            # ================================================================
            # STEP 1: Stop old instance
            # ================================================================
            if self.telegram_bot:
                logger.info("[TELEGRAM] Stopping old instance...")
                
                # Signal shutdown
                if hasattr(self.telegram_bot, '_shutdown_event'):
                    self.telegram_bot._shutdown_event.set()
                
                # Stop the loop
                if hasattr(self.telegram_bot, '_loop') and self.telegram_bot._loop:
                    loop = self.telegram_bot._loop
                    
                    if not loop.is_closed():
                        try:
                            # Stop the loop
                            loop.call_soon_threadsafe(loop.stop)
                            
                            # Wait for it to stop
                            time.sleep(2)
                            
                            # Close it
                            if not loop.is_closed():
                                loop.close()
                                
                            logger.info("[TELEGRAM] Old loop closed")
                        except Exception as e:
                            logger.warning(f"[TELEGRAM] Loop cleanup: {e}")
                
                # Wait for thread to die
                if self.telegram_thread and self.telegram_thread.is_alive():
                    logger.info("[TELEGRAM] Waiting for old thread...")
                    self.telegram_thread.join(timeout=5)
                    
                    if self.telegram_thread.is_alive():
                        logger.warning("[TELEGRAM] Thread still alive, continuing anyway")

            # ================================================================
            # STEP 2: Wait for Telegram API cooldown
            # ================================================================
            logger.info("[TELEGRAM] Waiting 5s for API cooldown...")
            time.sleep(5)

            # ================================================================
            # STEP 3: Create fresh bot instance
            # ================================================================
            logger.info("[TELEGRAM] Creating new bot instance...")
            
            from src.telegram import TradingTelegramBot
            from telegram_config import TELEGRAM_CONFIG
            
            self.telegram_bot = TradingTelegramBot(
                            token=TELEGRAM_CONFIG["bot_token"],
                            admin_ids=TELEGRAM_CONFIG["admin_ids"],
                            trading_bot=self,
                            signal_monitor=self.signal_monitor
                        )            # ✅ CRITICAL: Don't call start_loop_thread here!
            # Let _run_telegram_loop do it
            
            # ================================================================
            # STEP 4: Start new thread
            # ================================================================
            logger.info("[TELEGRAM] Starting new thread...")
            
            self._telegram_ready.clear()
            self._telegram_should_stop.clear()
            
            self.telegram_thread = Thread(
                target=self._run_telegram_loop,
                daemon=True,
                name="TelegramBot-Restart"
            )
            self.telegram_thread.start()

            # Wait for ready signal
            if self._telegram_ready.wait(timeout=30):
                logger.info("[TELEGRAM] ✅ Restart successful")
                return True
            else:
                logger.error("[TELEGRAM] ❌ Restart timeout")
                return False

        except Exception as e:
            logger.error(f"[TELEGRAM] Restart error: {e}", exc_info=True)
            return False
    
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

            # ✅ FIXED: Initialize selectors AFTER exchanges are connected
            self.dynamic_selector = DynamicPresetSelector(
                self.data_manager, self.config, telegram_bot=self.telegram_bot
            )
            self.hybrid_selector = HybridAggregatorSelector(
                self.data_manager, self.config, telegram_bot=self.telegram_bot
            )

            preset_mode = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            if preset_mode == "auto":
                logger.info("\n" + "=" * 70)
                logger.info("🎯 AUTO PRESET MODE ENABLED")
                logger.info("=" * 70)
                logger.info("System will automatically select optimal preset per asset:")
                logger.info("  • CONSERVATIVE: High risk/volatility conditions")
                logger.info("  • BALANCED:     Normal market conditions")
                logger.info("  • AGGRESSIVE:   Strong trending markets")
                logger.info("  • SCALPER:      Ideal low-volatility conditions")
                logger.info("=" * 70 + "\n")

                # Set initial presets for all enabled assets
                logger.info(
                    "[AUTO PRESET] Analyzing market conditions for initial setup..."
                )
                for asset_name in self.strategies.keys():
                    if self.config["assets"][asset_name].get("enabled", False):
                        initial_preset = self.dynamic_selector.get_preset_for_asset(
                            asset_name
                        )
                        self.dynamic_selector.current_presets[asset_name] = initial_preset
                        logger.info(f"  {asset_name:6} → {initial_preset.upper()}")
            else:
                logger.info(f"[PRESET] Using manual preset: {preset_mode.upper()}")

            self.load_models()

            self.initialize_mtf_regime_detection()

            if self.mtf_integration:
                logger.info("[MTF] Running initial regime analysis...")
                self.run_mtf_regime_analysis()

            # Initialize and start the autotrainer
            self.initialize_autotrainer()

            # Start Telegram (existing code continues)
            if self.telegram_bot:
                logger.info("\n[TELEGRAM] Starting bot thread...")

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

            schedule.every(4).hours.do(self.run_mtf_regime_analysis)

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
            logger.info(f"[MTF] Regime updates: Every 4 hours")
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

                            if not self._telegram_should_stop.is_set():
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

        # Stop the autotrainer
        if self.autotrainer:
            self.autotrainer.stop()

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

    # Track auto_trainer so we can stop it later
    auto_trainer = None

    try:
        # 1. Initialize the MAIN BOT first
        bot = TradingBot()

        if bot.db_manager:
            bot.portfolio_manager.db_manager = bot.db_manager

            # Also pass to all positions
            for position in bot.portfolio_manager.positions.values():
                position.db_manager = bot.db_manager

        # 2. ✅ FIXED: Initialize Auto-Trainer AFTER the bot exists
        auto_trainer = ContinuousLearningPipeline(
            config=config,
            trading_bot=bot,  # Now 'bot' is defined!
            telegram_bot=bot.telegram_bot,  # Access telegram directly from the bot
        )
        auto_trainer.start()

        # 3. Start the main bot loop
        bot.start()

    except KeyboardInterrupt:
        logger.info("\n[!] KeyboardInterrupt")
    except Exception as e:
        logger.error(f"[FATAL] {e}", exc_info=True)
        sys.exit(1)
    finally:
        # ✨ NEW: Stop Auto-Trainer on exit
        if auto_trainer:
            auto_trainer.stop()

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
