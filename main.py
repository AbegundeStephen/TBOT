#!/usr/bin/env python3
"""
Main Trading Bot - WITH MARKET HOURS CHECKING
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import schedule
import io

# FIX Windows encoding BEFORE any imports that use logging
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.execution.signal_aggregator import SignalAggregator
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.market_hours import MarketHours, should_trade_btc, should_trade_gold

# Setup logging with UTF-8 encoding
def setup_logging(config):
    """Setup logging with proper encoding"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trading_bot.log")
    
    Path(log_file).parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    Main trading bot - WITH MARKET HOURS CHECKING
    """

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT")
        logger.info("=" * 70)

        with open(config_path, encoding='utf-8') as f:
            self.config = json.load(f)

        setup_logging(self.config)
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = PortfolioManager(self.config)

        self.strategies = {
            "BTC": {},
            "GOLD": {}
        }
        
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
        
        # Track last market status check
        self.last_market_status_log = None

    def _initialize_strategies(self):
        """Initialize strategies for each enabled asset"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Strategies")
        logger.info("-" * 70)
        
        for asset_name, asset_config in self.config["assets"].items():
            if not asset_config.get("enabled", False):
                logger.info(f"[!] {asset_name}: Disabled in config")
                continue
            
            strategies = asset_config.get("strategies", {})
            
            if strategies.get("mean_reversion", {}).get("enabled", False):
                mr_config = self.config["strategy_configs"]["mean_reversion"][asset_name]
                self.strategies[asset_name]["mean_reversion"] = MeanReversionStrategy(mr_config)
                logger.info(f"[OK] {asset_name}: Mean Reversion initialized")
            
            if strategies.get("trend_following", {}).get("enabled", False):
                tf_config = self.config["strategy_configs"]["trend_following"][asset_name]
                self.strategies[asset_name]["trend_following"] = TrendFollowingStrategy(tf_config)
                logger.info(f"[OK] {asset_name}: Trend Following initialized")

    def _initialize_aggregators(self):
        """Initialize ENHANCED signal aggregators for each asset"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Enhanced Signal Aggregators")
        logger.info("-" * 70)
        
        AGGREGATOR_PRESETS = {
            "conservative": {
                "tier_1_threshold": 0.48,
                "tier_2_threshold": 0.58,
                "tier_3_threshold": 0.68,
                "agreement_bonus": 0.08,
                "require_agreement": True,
            },
            "balanced": {
                "tier_1_threshold": 0.42,
                "tier_2_threshold": 0.52,
                "tier_3_threshold": 0.62,
                "agreement_bonus": 0.12,
                "require_agreement": False,
            },
            "aggressive": {
                "tier_1_threshold": 0.38,
                "tier_2_threshold": 0.48,
                "tier_3_threshold": 0.58,
                "agreement_bonus": 0.15,
                "require_agreement": False,
            }
        }
        
        aggregator_mode = self.config.get("aggregator_settings", {}).get("mode", "adaptive_tiered")
        aggregator_preset = self.config.get("aggregator_settings", {}).get("preset", "balanced")
        
        logger.info(f"Aggregator Mode: {aggregator_mode}")
        logger.info(f"Aggregator Preset: {aggregator_preset}")
        
        for asset_name, strategies in self.strategies.items():
            mr_strategy = strategies.get("mean_reversion")
            tf_strategy = strategies.get("trend_following")
            
            if mr_strategy and tf_strategy:
                self.aggregators[asset_name] = SignalAggregator(
                    mean_reversion_strategy=mr_strategy,
                    trend_following_strategy=tf_strategy,
                    mode=aggregator_mode,
                    confidence_config=AGGREGATOR_PRESETS[aggregator_preset]
                )
                
                logger.info(
                    f"[OK] {asset_name}: Enhanced Aggregator initialized "
                    f"({aggregator_mode}/{aggregator_preset})"
                )
            elif mr_strategy or tf_strategy:
                logger.warning(
                    f"[!] {asset_name}: Only one strategy available, "
                    f"will use single strategy mode"
                )
                self.aggregators[asset_name] = SignalAggregator(
                    mean_reversion_strategy=mr_strategy,
                    trend_following_strategy=tf_strategy,
                    mode=aggregator_mode,
                    confidence_config=AGGREGATOR_PRESETS[aggregator_preset]
                )

    def initialize_exchanges(self):
        """Initialize exchange connections based on enabled assets"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Exchange Connections")
        logger.info("-" * 70)

        if self.config["assets"]["BTC"].get("enabled", False):
            if self.data_manager.initialize_binance():
                self.binance_handler = BinanceExecutionHandler(
                    self.config, 
                    self.data_manager.binance_client,
                    self.portfolio_manager
                )
                logger.info("[OK] Binance handler initialized")
            else:
                logger.error("[FAIL] Failed to initialize Binance - BTC trading disabled")

        if self.config["assets"]["GOLD"].get("enabled", False):
            if self.data_manager.initialize_mt5():
                self.mt5_handler = MT5ExecutionHandler(
                    self.config,
                    self.portfolio_manager
                )
                logger.info("[OK] MT5 handler initialized")
            else:
                logger.error("[FAIL] Failed to initialize MT5 - GOLD trading disabled")

    def load_models(self):
        """Load trained ML models for each asset and strategy"""
        logger.info("\n" + "-" * 70)
        logger.info("Loading Trained Models")
        logger.info("-" * 70)
        
        models_loaded = 0
        models_expected = 0
        
        for asset_name, strategies in self.strategies.items():
            for strategy_name, strategy in strategies.items():
                models_expected += 1
                model_path = f"models/{strategy_name}_{asset_name.lower()}.pkl"
                
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
        
        logger.info(f"\n[OK] Successfully loaded {models_loaded}/{models_expected} models")

    def reset_daily_counters(self):
        """Reset daily trading counters"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trade_count_today = 0
            self.daily_loss = 0.0
            self.last_trade_date = current_date
            logger.info(f"[RESET] Daily counters reset for {current_date}")

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
        total_loss_pct = self.daily_loss / self.config["portfolio"]["initial_capital"]
        if total_loss_pct >= circuit_breaker:
            logger.error(f"[BREAKER] CIRCUIT BREAKER! Loss: {total_loss_pct:.2%}")
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
        min_time_minutes = self.config["trading"].get("min_time_between_trades_minutes", 60)
        
        if asset_name in self.last_trade_times:
            time_since_last = datetime.now() - self.last_trade_times[asset_name]
            if time_since_last.total_seconds() < min_time_minutes * 60:
                remaining = min_time_minutes - (time_since_last.total_seconds() / 60)
                logger.info(
                    f"[COOLDOWN] {asset_name}: {remaining:.0f} min remaining"
                )
                return False
        
        return True

    def check_market_hours(self, asset_name: str) -> bool:
        """
        NEW: Check if market is open for the given asset
        Returns True if trading is allowed, False otherwise
        """
        if asset_name == "BTC":
            # BTC is 24/7, always allow
            return True
        
        elif asset_name == "GOLD":
            # GOLD follows Forex hours (Sun 22:00 GMT - Fri 22:00 GMT)
            is_open = should_trade_gold()
            
            if not is_open:
                status, message = MarketHours.get_market_status("gold")
                
                # Only log market status once per hour to avoid spam
                current_hour = datetime.now().hour
                if self.last_market_status_log != current_hour:
                    logger.info(f"[MARKET] {asset_name}: {message}")
                    
                    # Show when market opens
                    seconds_until = MarketHours.time_until_market_open("gold")
                    if seconds_until > 0:
                        hours_until = seconds_until / 3600
                        logger.info(f"[MARKET] {asset_name}: Market opens in {hours_until:.1f} hours")
                    
                    self.last_market_status_log = current_hour
            
            return is_open
        
        # Default: allow trading (for any future assets)
        return True

    def trade_asset(self, asset_name: str):
        """
        Execute trading logic with market hours checking
        """
        asset_config = self.config["assets"][asset_name]
        
        if not asset_config.get("enabled", False):
            return
        
        # CHECK MARKET HOURS FIRST
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
                lookback_days = 15 if interval == "1h" else (60 if interval == "4h" else 365)
            else:
                timeframe = asset_config.get("timeframe", "H1")
                lookback_days = 25 if timeframe in ["H1", "TIMEFRAME_H1"] else 75
            
            start_time = end_time - timedelta(days=lookback_days)
            
            logger.info(f"[DATA] Fetching {lookback_days} days up to NOW...")
            logger.info(f"[TIME] Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
            
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
            
            min_bars_for_prediction = 100
            
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
                    logger.warning(f"[!] Could not get live price, using last candle: ${df['close'].iloc[-1]:,.2f}")
                    current_price = df['close'].iloc[-1]
            else:
                current_price = df['close'].iloc[-1]
            
            logger.info(f"[PRICE] Current {asset_name} Price: ${current_price:,.2f}")
            logger.info(f"[PRICE] Last Bar Close: ${df['close'].iloc[-1]:,.2f}")
            logger.info(f"[PRICE] Last Bar Time: {df.index[-1].strftime('%Y-%m-%d %H:%M')} UTC")
            
            # Get signal
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[!] {asset_name}: No aggregator configured")
                return
            
            signal, details = aggregator.get_aggregated_signal(df)
            
            # Log signal details
            logger.info(f"\n[SIGNAL] Analysis:")
            logger.info(f"  Mean Reversion: Signal={details['mean_reversion_signal']:>2}, Confidence={details['mean_reversion_confidence']:.3f}")
            logger.info(f"  Trend Following: Signal={details['trend_following_signal']:>2}, Confidence={details['trend_following_confidence']:.3f}")
            logger.info(f"  >> Final Signal: {signal:>2}")
            logger.info(f"  >> Combined Confidence: {details['combined_confidence']:.3f}")
            logger.info(f"  >> Reasoning: {details['reasoning']}")
            
            # Execute signal (even HOLD signals check SL/TP)
            success = False
            
            if exchange == "binance":
                success = self.binance_handler.execute_signal(
                    signal, 
                    current_price,
                    asset_name
                )
                self.binance_handler.check_and_update_positions(asset_name)
            else:
                success = self.mt5_handler.execute_signal(
                    signal,
                    symbol,
                    asset_name
                )
                self.mt5_handler.check_and_update_positions(asset_name)
            
            # Only count as trade if we opened a new position
            if signal != 0 and success:
                if not self.check_trading_limits():
                    logger.info(f"[LIMIT] {asset_name}: Trading limits prevent new position")
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
        
    def _log_trade(self, asset_name: str, signal: int, details: dict, price: float):
        """Log trade details to separate file"""
        try:
            trade_log_file = Path("logs/trades.log")
            trade_log_file.parent.mkdir(exist_ok=True)
            
            with open(trade_log_file, "a", encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()},{asset_name},{signal},{price:.2f},"
                       f"{details['combined_confidence']:.3f},{details['reasoning']}\n")
        except Exception as e:
            logger.warning(f"Could not log trade: {e}")

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("\n" + "=" * 70)
        logger.info(f"[CYCLE] TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        # Show current market status for all assets
        logger.info("\n[MARKET STATUS]")
        for asset_name in ["BTC", "GOLD"]:
            if self.config["assets"].get(asset_name, {}).get("enabled", False):
                if asset_name == "BTC":
                    logger.info(f"  {asset_name}: 24/7 - Always Open")
                elif asset_name == "GOLD":
                    status, message = MarketHours.get_market_status("gold")
                    logger.info(f"  {asset_name}: {status} - {message}")
        
        self.reset_daily_counters()
        self.portfolio_manager.update_positions()
        
        enabled_assets = [
            name for name, config in self.config["assets"].items() 
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
        logger.info(f"Total Value: ${portfolio_status.get('total_value', 0):,.2f}")
        logger.info(f"Cash: ${portfolio_status.get('cash', 0):,.2f}")
        logger.info(f"Open Positions: {portfolio_status.get('open_positions', 0)}")
        logger.info(f"Daily P&L: ${portfolio_status.get('daily_pnl', 0):,.2f}")
        
        logger.info("\n[OK] Trading cycle complete")
        logger.info("=" * 70)

    def start(self):
        """Start the trading bot"""
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT STARTING")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.config['trading']['mode'].upper()}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show market hours info
        logger.info("\n[MARKET HOURS]")
        logger.info("  BTC (Crypto): 24/7 - Always trading")
        logger.info("  GOLD (Forex):  Sun 22:00 GMT - Fri 22:00 GMT")
        logger.info("=" * 70)
        
        self.initialize_exchanges()
        self.load_models()
        
        if not self.binance_handler and not self.mt5_handler:
            logger.error("=" * 70)
            logger.error("[FAIL] NO EXCHANGE HANDLERS AVAILABLE!")
            logger.error("Cannot proceed with trading.")
            logger.error("=" * 70)
            sys.exit(1)
        
        check_interval = self.config["trading"].get("check_interval_seconds", 300)
        schedule.every(check_interval).seconds.do(self.run_trading_cycle)
        
        self.is_running = True
        logger.info(f"\n[OK] Trading bot is now running")
        logger.info(f"[TIME] Checking market every {check_interval}s ({check_interval/60:.1f} minutes)")
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
        """Stop the trading bot"""
        logger.info("\n" + "=" * 70)
        logger.info("[STOP] STOPPING TRADING BOT")
        logger.info("=" * 70)
        
        self.is_running = False
        
        if self.config["trading"].get("close_positions_on_shutdown", False):
            logger.info("Closing all open positions...")
            self.portfolio_manager.close_all_positions()
        
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
        with open("config/config.json", encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("[FAIL] config/config.json not found!")
        sys.exit(1)
    
    # Check required models
    required_models = []
    for asset_name, asset_config in config["assets"].items():
        if asset_config.get("enabled", False):
            strategies = asset_config.get("strategies", {})
            for strategy_name, strategy_config in strategies.items():
                if strategy_config.get("enabled", False):
                    model_file = f"models/{strategy_name}_{asset_name.lower()}.pkl"
                    required_models.append(model_file)
    
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