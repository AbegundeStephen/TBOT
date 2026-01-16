#!/usr/bin/env python3
"""
Fixed BTC Futures Execution Test Script
"""

import subprocess
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from binance.client import Client

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.execution.binance_handler import BinanceExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.database.database_manager import TradingDatabaseManager
from src.execution.binance_futures import enable_futures_for_binance_handler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("TEST_SCRIPT")


def load_config():
    """Load configuration"""
    config_path = Path("config/config.json")
    if not config_path.exists():
        logger.warning("config.json not found, using template")
        config_path = Path("config/config.template.json")

    with open(config_path, "r") as f:
        return json.load(f)


def run_test():
    logger.info("=" * 60)
    logger.info("🚀 STARTING BTC FUTURES EXECUTION TEST")
    logger.info("=" * 60)

    # 1. Load Config
    config = load_config()

    if not config["assets"]["BTC"].get("enabled", False):
        logger.error("❌ BTC is disabled in config!")
        return

    # 2. Initialize Database (Optional)
    try:
        db_config = config.get("database", {})
        if db_config.get("enabled"):
            db_manager = TradingDatabaseManager(
                supabase_url=db_config["supabase_url"],
                supabase_key=db_config["supabase_key"],
            )
            logger.info("✅ Database Manager initialized")
        else:
            db_manager = None
            logger.info("⚠️ Database disabled")
    except Exception as e:
        logger.warning(f"⚠️ Database init failed: {e}")
        db_manager = None

    # 3. Initialize Binance Client (Futures)
    api_config = config["api"]["binance_futures"]
    try:
        # ✅ FIX: Use Futures client from the start
        client = Client(
            api_key=api_config["api_key"],
            api_secret=api_config["api_secret"],
            testnet=api_config.get("testnet", False),
        )

        # Test Futures connection
        client.futures_ping()
        logger.info("✅ Binance Futures Client connected")
    except Exception as e:
        logger.critical(f"❌ Failed to connect to Binance Futures: {e}")
        return

    # 4. Initialize Portfolio Manager
    try:
        portfolio_manager = PortfolioManager(
            config=config, binance_client=client, db_manager=db_manager
        )
        logger.info("✅ Portfolio Manager initialized")
        logger.info(f"   Capital: ${portfolio_manager.current_capital:,.2f}")
    except Exception as e:
        logger.critical(f"❌ Portfolio Manager failed: {e}")
        return

    # 5. Initialize Execution Handler
    try:
        # ✅ CRITICAL: Disable auto-sync temporarily
        original_auto_sync = config["trading"].get("auto_sync_on_startup", True)
        config["trading"]["auto_sync_on_startup"] = False

        handler = BinanceExecutionHandler(
            client=client,
            config=config,
            portfolio_manager=portfolio_manager,
            data_manager=db_manager,
        )

        # Restore original setting
        config["trading"]["auto_sync_on_startup"] = original_auto_sync

        logger.info("✅ Binance Execution Handler initialized")
    except Exception as e:
        logger.critical(f"❌ Handler initialization failed: {e}")
        return

    # 6. ENABLE FUTURES (Before sync!)
    logger.info("⚙️  Integrating Futures Handler...")
    if config["assets"]["BTC"].get("enable_futures", False):
        success = enable_futures_for_binance_handler(handler)
        if success:
            logger.info("✅ Futures Enabled for Handler")
        else:
            logger.error("❌ Failed to enable Futures!")
            return
    else:
        logger.error("❌ 'enable_futures' is FALSE in BTC config")
        return

    # 7. NOW do position sync (after Futures is enabled)
    if original_auto_sync and not portfolio_manager.is_paper_mode:
        logger.info("🔄 Syncing positions with Binance...")
        handler.sync_positions_with_binance("BTC")

    # 8. EXECUTE TEST TRADE
    logger.info("\n" + "-" * 60)
    logger.info("🧪 EXECUTING TEST LONG POSITION")
    logger.info("-" * 60)

    # Get current price
    try:
        current_price = handler.get_current_price("BTCUSDT")
        logger.info(f"💰 Current BTC Price: ${current_price:,.2f}")
    except Exception as e:
        logger.error(f"❌ Could not fetch price: {e}")
        return

    # ✅ FIX: Create proper signal_details with all required fields
    signal_details = {
        "signal_quality": 0.95,
        "regime": "🚀 BULL",
        "regime_confidence": 0.85,
        "reasoning": "TEST SCRIPT EXECUTION",
        "aggregator_mode": "council",  # Test council mode
        "mode_confidence": 0.90,
        "regime_analysis": {
            "regime_type": "trending_bull",
            "trend_strength": "strong",
            "trend_direction": "up",
            "adx": 35.0,
            "volatility_regime": "normal",
            "volatility_ratio": 1.0,
            "price_clarity": "clear",
            "momentum_aligned": True,
            "at_key_level": False,
        },
        "ai_validation": {
            "pattern_name": "Test Pattern",
            "pattern_confidence": 0.99,
            "validation_passed": True,
        },
    }

    # Execute the trade
    try:
        success = handler.execute_signal(
            signal=1,  # BUY/LONG
            current_price=current_price,
            asset_name="BTC",
            confidence_score=0.95,
            market_condition="bull",
            signal_details=signal_details,  # ✅ Pass complete details
        )

        if success:
            logger.info("\n✅ ✅ TRADE EXECUTION SUCCESSFUL! ✅ ✅")
            logger.info("Check your Binance Futures Open Positions.")

            # Show position details
            positions = portfolio_manager.get_asset_positions("BTC")
            if positions:
                pos = positions[-1]
                logger.info(f"\nPosition Created:")
                logger.info(f"  ID:          {pos.position_id}")
                logger.info(f"  Side:        {pos.side.upper()}")
                logger.info(f"  Entry:       ${pos.entry_price:,.2f}")
                logger.info(f"  Quantity:    {pos.quantity:.6f} BTC")

                # ✅ FIX: Handle None values safely
                if pos.stop_loss:
                    logger.info(f"  Stop Loss:   ${pos.stop_loss:,.2f}")
                else:
                    logger.info(f"  Stop Loss:   VTM Managed")

                if pos.take_profit:
                    logger.info(f"  Take Profit: ${pos.take_profit:,.2f}")
                else:
                    logger.info(f"  Take Profit: VTM Managed")

                logger.info(f"  Futures:     {getattr(pos, 'is_futures', 'Unknown')}")
                logger.info(f"  Leverage:    {getattr(pos, 'leverage', 'Unknown')}x")
        else:
            logger.error("\n❌ TRADE EXECUTION FAILED (Handler returned False)")
            logger.error("Check logs above for specific failure reason")

    except Exception as e:
        logger.error(f"\n❌ CRASH DURING EXECUTION: {e}", exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_test()
