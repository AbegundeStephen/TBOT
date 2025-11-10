# **main.py**

#!/usr/bin/env python3
"""
Main Trading Bot - Live Paper Trading Deployment
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import schedule

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.execution.signal_aggregator import SignalAggregator
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator
    """

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("Initializing Trading Bot...")

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize components
        self.data_manager = DataManager(self.config)

        # Initialize strategies
        mr_config = {**self.config["strategies"]["mean_reversion"], **self.config["ml"]}
        tf_config = {
            **self.config["strategies"]["trend_following"],
            **self.config["ml"],
        }

        self.mean_reversion = MeanReversionStrategy(mr_config)
        self.trend_following = TrendFollowingStrategy(tf_config)

        # Initialize signal aggregator
        self.aggregator = SignalAggregator(self.mean_reversion, self.trend_following)

        # Initialize execution handlers
        self.binance_handler = None
        self.mt5_handler = None

        # Trading state
        self.is_running = False
        self.trade_count_today = 0
        self.last_trade_date = None

    def initialize_exchanges(self):
        """Initialize exchange connections"""
        logger.info("Initializing exchange connections...")

        # Initialize Binance
        if self.data_manager.initialize_binance():
            self.binance_handler = BinanceExecutionHandler(
                self.config, self.data_manager.binance_client
            )
            logger.info("Binance handler initialized")

        # Initialize MT5
        if self.data_manager.initialize_mt5():
            self.mt5_handler = MT5ExecutionHandler(self.config)
            logger.info("MT5 handler initialized")

    def load_models(self):
        """Load trained ML models"""
        logger.info("Loading trained models...")

        success = self.aggregator.load_models(
            "models/mean_reversion_btc.pkl", "models/trend_following_btc.pkl"
        )

        if not success:
            logger.error("Failed to load models. Please run train.py first.")
            sys.exit(1)

        logger.info("Models loaded successfully")

    def reset_daily_counters(self):
        """Reset daily trading counters"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trade_count_today = 0
            self.last_trade_date = current_date
            logger.info(f"Daily counters reset for {current_date}")

    def check_trading_limits(self) -> bool:
        """Check if trading limits are reached"""
        max_daily_trades = self.config["trading"]["risk_management"]["max_daily_trades"]

        if self.trade_count_today >= max_daily_trades:
            logger.warning(f"Daily trade limit reached ({max_daily_trades})")
            return False

        return True

    def trade_btc(self):
        """Execute BTC trading logic"""
        if not self.binance_handler:
            logger.warning("Binance handler not initialized")
            return

        try:
            logger.info("Fetching latest BTC data...")

            # Fetch recent data (last 500 bars of 5min data)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=2)

            df = self.data_manager.fetch_binance_data(
                symbol="BTCUSDT",
                interval="5m",
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d"),
            )

            df = self.data_manager.clean_data(df)

            if len(df) < 200:
                logger.warning("Insufficient data for BTC analysis")
                return

            # Get aggregated signal
            signal, details = self.aggregator.get_aggregated_signal(df)

            logger.info(f"BTC Signal Details: {details}")

            # Execute trade if signal is valid
            if signal != 0 and self.check_trading_limits():
                current_price = self.binance_handler.get_current_price()

                if self.binance_handler.execute_signal(signal, current_price):
                    self.trade_count_today += 1
                    logger.info(
                        f"BTC trade executed. Daily count: {self.trade_count_today}"
                    )
                else:
                    logger.warning("Failed to execute BTC trade")
            else:
                logger.info("No BTC trade executed (HOLD signal or limits reached)")

        except Exception as e:
            logger.error(f"Error in BTC trading: {e}", exc_info=True)

    def trade_gold(self):
        """Execute Gold trading logic"""
        if not self.mt5_handler:
            logger.warning("MT5 handler not initialized")
            return

        try:
            logger.info("Fetching latest Gold data...")

            # Fetch recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=2)

            df = self.data_manager.fetch_mt5_data(
                symbol="XAUUSD",
                timeframe="M5",
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d"),
            )

            df = self.data_manager.clean_data(df)

            if len(df) < 200:
                logger.warning("Insufficient data for Gold analysis")
                return

            # Get aggregated signal
            signal, details = self.aggregator.get_aggregated_signal(df)

            logger.info(f"Gold Signal Details: {details}")

            # Execute trade if signal is valid
            if signal != 0 and self.check_trading_limits():
                if self.mt5_handler.execute_signal(signal):
                    self.trade_count_today += 1
                    logger.info(
                        f"Gold trade executed. Daily count: {self.trade_count_today}"
                    )
                else:
                    logger.warning("Failed to execute Gold trade")
            else:
                logger.info("No Gold trade executed (HOLD signal or limits reached)")

        except Exception as e:
            logger.error(f"Error in Gold trading: {e}", exc_info=True)

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("=" * 60)
        logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        self.reset_daily_counters()

        # Trade BTC
        self.trade_btc()

        # Small delay between trades
        time.sleep(2)

        # Trade Gold
        self.trade_gold()

        logger.info("Trading cycle complete")
        logger.info("=" * 60)

    def start(self):
        """Start the trading bot"""
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTING")
        logger.info("=" * 60)

        # Initialize exchanges
        self.initialize_exchanges()

        # Load models
        self.load_models()

        # Schedule trading cycles
        # Run every 5 minutes (adjust based on your timeframe)
        schedule.every(5).minutes.do(self.run_trading_cycle)

        self.is_running = True
        logger.info("Trading bot is now running...")
        logger.info("Press Ctrl+C to stop")

        try:
            # Run first cycle immediately
            self.run_trading_cycle()

            # Enter main loop
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received...")
            self.stop()

    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False
        self.data_manager.shutdown()
        logger.info("Trading bot stopped successfully")


def main():
    """Main entry point"""
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Check if models exist
    if not Path("models/mean_reversion_btc.pkl").exists():
        logger.error("Trained models not found. Please run train.py first.")
        sys.exit(1)

    # Initialize and start bot
    bot = TradingBot()
    bot.start()


if __name__ == "__main__":
    main()
