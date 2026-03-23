#!/usr/bin/env python3
"""
Multi-Asset Margin Trading Test Suite for MetaTrader 5
Tests the entire trading pipeline: LONG → HOLD → SHORT → CLOSE

Usage:
  python test_gold_margin_execution.py --asset USTEC --mode live    # Execute real trades
  python test_gold_margin_execution.py --asset EURJPY --mode paper  # Simulated trades
  python test_gold_margin_execution.py --test long                 # Test only LONG
"""

import subprocess
import json
import logging
import sys
import time
import argparse
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path

# Load environment variables
load_dotenv()

import MetaTrader5 as mt5

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.execution.mt5_handler import MT5ExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.database.database_manager import TradingDatabaseManager
from src.data.data_manager import DataManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MT5_TEST_SUITE")


def load_config():
    """Load configuration"""
    config_path = Path("config/config.json")
    if not config_path.exists():
        logger.warning("config.json not found, using template")
        config_path = Path("config/config.template.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    
    # ✨  Override config with environment variables for security
    if os.getenv("SUPABASE_URL"):
        config.setdefault("database", {})["supabase_url"] = os.getenv("SUPABASE_URL")
    if os.getenv("SUPABASE_KEY"):
        config.setdefault("database", {})["supabase_key"] = os.getenv("SUPABASE_KEY")
    
    if os.getenv("MT5_LOGIN"):
        config.setdefault("api", {}).setdefault("mt5", {})["login"] = int(os.getenv("MT5_LOGIN"))
    if os.getenv("MT5_PASSWORD"):
        config.setdefault("api", {}).setdefault("mt5", {})["password"] = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        config.setdefault("api", {}).setdefault("mt5", {})["server"] = os.getenv("MT5_SERVER")
        
    return config


def initialize_mt5(config):
    """Initialize connection to MetaTrader 5"""
    mt5_config = config["api"]["mt5"]
    if not mt5.initialize(
        login=mt5_config["login"],
        password=mt5_config["password"],
        server=mt5_config["server"],
        path=mt5_config["path"],
    ):
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info("✅ MetaTrader 5 connection initialized")
    return True


def create_signal_details(asset_name: str, signal_type: str, confidence: float = 0.85):
    """
    Create realistic signal_details matching what your strategies generate.
    """
    regime_map = {
        "buy": {
            "type": "trending_bull",
            "direction": "up",
            "emoji": "🚀 BULL",
            "adx": 35.0,
        },
        "sell": {
            "type": "trending_bear",
            "direction": "down",
            "emoji": "🐻 BEAR",
            "adx": 32.0,
        },
        "hold": {
            "type": "neutral",
            "direction": "sideways",
            "emoji": "😐 NEUTRAL",
            "adx": 18.0,
        }
    }
    
    regime = regime_map.get(signal_type, regime_map["hold"])
    
    return {
        "signal_quality": confidence,
        "regime": regime["emoji"],
        "regime_confidence": confidence,
        "reasoning": f"TEST SCRIPT: Simulated {signal_type.upper()} signal for {asset_name} validation",
        "aggregator_mode": "council",
        "mode_confidence": confidence,
        "regime_analysis": {
            "regime_type": regime["type"],
            "trend_strength": "strong" if confidence > 0.7 else "moderate",
            "trend_direction": regime["direction"],
            "adx": regime["adx"],
        },
        "ai_validation": {"validation_passed": True, "action": "approved"},
        "council_scores": {
            "buy_score": 4.5 if signal_type == "buy" else 1.5,
            "sell_score": 4.5 if signal_type == "sell" else 1.5,
        },
        "timestamp": datetime.now().isoformat(),
        "test_mode": True,
    }


def check_mt5_account_info() -> dict:
    """Check MT5 account info"""
    try:
        info = mt5.account_info()
        if not info:
            logger.error("Could not get MT5 account info")
            return {'balance': 0, 'equity': 0, 'margin': 0, 'margin_free': 0}
            
        return {
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'margin_free': info.margin_free,
        }
    except Exception as e:
        logger.error(f"Could not fetch MT5 account info: {e}")
        return {'balance': 0, 'equity': 0, 'margin': 0, 'margin_free': 0}


def test_long_position(handler, asset_name, current_price, config):
    """Test opening a LONG position"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🧪 TEST 1: {asset_name} LONG POSITION (BUY SIGNAL)")
    logger.info("=" * 80)
    
    signal_details = create_signal_details(asset_name, "buy", confidence=0.85)
    
    logger.info(f"📊 Signal Details:")
    logger.info(f"  Quality:     {signal_details['signal_quality']:.0%}")
    logger.info(f"  Regime:      {signal_details['regime']}")
    
    try:
        success = handler.execute_signal(
            signal=1,  # BUY
            asset_name=asset_name,
            confidence_score=0.85,
            market_condition="bull",
            signal_details=signal_details,
        )
        
        if success:
            logger.info(f"\n✅ {asset_name} LONG POSITION OPENED SUCCESSFULLY")
            positions = handler.portfolio_manager.get_asset_positions(asset_name)
            if positions:
                pos = positions[-1]
                logger.info(f"\n📈 Position Details:")
                logger.info(f"  ID:          {pos.position_id}")
                logger.info(f"  Side:        {pos.side.upper()}")
                logger.info(f"  Entry:       ${pos.entry_price:,.5f}")
                logger.info(f"  Quantity:    {pos.quantity} units")
                logger.info(f"  Size:        ${(pos.quantity * pos.entry_price):,.2f}")
                logger.info(f"  MT5 Ticket:  {pos.mt5_ticket}")
                
                if pos.trade_manager:
                    logger.info(f"  VTM:         ACTIVE ✓")
                    vtm_status = pos.get_vtm_status()
                    logger.info(f"    SL: ${vtm_status.get('stop_loss', 0):,.5f}")
                    logger.info(f"    TP: ${vtm_status.get('take_profit', 0):,.5f}")
                return True
        else:
            logger.error(f"\n❌ {asset_name} LONG POSITION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def test_hold_signal(handler, asset_name):
    """Test HOLD signal (checks SL/TP on existing positions)"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🧪 TEST 2: {asset_name} HOLD SIGNAL (Position Management)")
    logger.info("=" * 80)
    
    if not handler.portfolio_manager.get_asset_positions(asset_name):
        logger.warning(f"⚠️  No {asset_name} positions to manage - skipping HOLD test")
        return True # Not a failure
    
    try:
        success = handler.execute_signal(
            signal=0,  # HOLD
            asset_name=asset_name,
        )
        logger.info(f"\n✅ {asset_name} HOLD SIGNAL PROCESSED (Checked SL/TP)")
        return True
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION during HOLD for {asset_name}: {e}", exc_info=True)
        return False


def test_short_position(handler, asset_name, current_price, config):
    """Test opening a SHORT position"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🧪 TEST 3: {asset_name} SHORT POSITION (SELL SIGNAL)")
    logger.info("=" * 80)
    
    signal_details = create_signal_details(asset_name, "sell", confidence=0.88)
    
    logger.info(f"📊 Signal Details:")
    logger.info(f"  Quality:      {signal_details['signal_quality']:.0%}")
    logger.info(f"  Regime:       {signal_details['regime']}")

    try:
        success = handler.execute_signal(
            signal=-1,  # SELL
            asset_name=asset_name,
            confidence_score=0.88,
            market_condition="bear",
            signal_details=signal_details,
        )
        
        if success:
            logger.info(f"\n✅ {asset_name} SHORT POSITION OPENED SUCCESSFULLY")
            positions = handler.portfolio_manager.get_asset_positions(asset_name)
            shorts = [p for p in positions if p.side == "short"]
            if shorts:
                pos = shorts[-1]
                logger.info(f"\n📉 Position Details:")
                logger.info(f"  ID:          {pos.position_id}")
                logger.info(f"  Side:        {pos.side.upper()}")
                logger.info(f"  Entry:       ${pos.entry_price:,.5f}")
                logger.info(f"  Quantity:    {pos.quantity} units")
                logger.info(f"  Size:        ${(pos.quantity * pos.entry_price):,.2f}")
                logger.info(f"  MT5 Ticket:  {pos.mt5_ticket}")

                if pos.trade_manager:
                    logger.info(f"  VTM:         ACTIVE ✓")
                    vtm_status = pos.get_vtm_status()
                    logger.info(f"    SL: ${vtm_status.get('stop_loss', 0):,.5f}")
                    logger.info(f"    TP: ${vtm_status.get('take_profit', 0):,.5f}")
                return True
        else:
            logger.error(f"\n❌ {asset_name} SHORT POSITION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def test_close_all(handler, asset_name, current_price):
    """Test closing all positions"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🧪 TEST 4: CLOSE ALL {asset_name} POSITIONS")
    logger.info("=" * 80)
    
    positions = handler.portfolio_manager.get_asset_positions(asset_name)
    if not positions:
        logger.warning(f"⚠️  No {asset_name} positions to close")
        return True
    
    # Use portfolio_manager's dedicated method to close all positions for the asset.
    closed_positions_results = handler.portfolio_manager.close_all_positions_for_asset(
        asset=asset_name,
        exit_price=current_price,
        reason="test_suite_close_all"
    )

    remaining = handler.portfolio_manager.get_asset_positions(asset_name)
    if not remaining:
        logger.info(f"\n✅ ALL {asset_name} POSITIONS CLOSED SUCCESSFULLY")
        success = len(closed_positions_results) == len(positions)
    else:
        logger.error(f"\n❌ FAILED TO CLOSE ALL {asset_name} POSITIONS. {len(remaining)} remaining.")
        success = False

    return success


def run_full_test_suite(asset_name="GOLD", mode="paper", test_type="full"):
    """Run complete test suite for an asset on MT5"""
    logger.info("=" * 80)
    logger.info(f"🚀 {asset_name} MARGIN COMPLETE TEST SUITE (MT5)")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Test Type: {test_type.upper()}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    config = load_config()
    config["trading"]["mode"] = mode
    
    if not config["assets"].get(asset_name, {}).get("enabled", False):
        logger.error(f"❌ {asset_name} is disabled in config!")
        return False

    if not initialize_mt5(config):
        return False
        
    db_manager = None
    if config.get("database", {}).get("enabled"):
        try:
            db_manager = TradingDatabaseManager(
                supabase_url=config["database"]["supabase_url"],
                supabase_key=config["database"]["supabase_key"],
            )
            logger.info("✅ Database Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️  Database init failed: {e}")

    try:
        data_manager = DataManager(config=config)
        data_manager.initialize_mt5()
        logger.info("✅ Data Manager initialized")
    except Exception as e:
        logger.critical(f"❌ Data Manager failed: {e}")
        mt5.shutdown()
        return False

    account_info = check_mt5_account_info()
    logger.info(f"\n💰 MT5 Account Info:")
    logger.info(f"  Balance:      ${account_info['balance']:,.2f}")
    logger.info(f"  Equity:       ${account_info['equity']:,.2f}")
    logger.info(f"  Free Margin:  ${account_info['margin_free']:,.2f}")
    
    if mode == "live" and account_info['margin_free'] < 50:
        logger.error("\n❌ INSUFFICIENT MARGIN! Need at least $50 free margin for live testing.")
        mt5.shutdown()
        return False

    # Initialize Portfolio Manager and Execution Handler
    try:
        portfolio_manager = PortfolioManager(
            config=config,
            db_manager=db_manager
        )
        logger.info("✅ Portfolio Manager initialized (Stage 1)")

        handler = MT5ExecutionHandler(
            config=config,
            portfolio_manager=portfolio_manager,
            data_manager=data_manager,
        )
        logger.info("✅ MT5 Execution Handler initialized")

        portfolio_manager.mt5_handler = handler
        portfolio_manager.execution_handlers = {'mt5': handler}
        logger.info("✅ Portfolio Manager linked with MT5 Handler (Stage 2)")

        portfolio_manager.refresh_capital(force=True)
        logger.info(f"   Live Capital Refreshed: ${portfolio_manager.current_capital:,.2f}")

    except Exception as e:
        logger.critical(f"❌ Portfolio/Handler initialization failed: {e}", exc_info=True)
        mt5.shutdown()
        return False

    symbol = config["assets"][asset_name]["symbol"]
    try:
        current_price = handler.get_current_price(symbol)
        logger.info(f"\n💰 Current {asset_name} ({symbol}) Price: ${current_price:,.5f}")
    except Exception as e:
        logger.error(f"❌ Could not fetch price for {symbol}: {e}")
        mt5.shutdown()
        return False

    results = {"long": False, "hold": False, "short": False, "close": False}
    
    logger.info(f"Attempting to close any existing {asset_name} positions before starting test...")
    test_close_all(handler, asset_name, current_price)


    try:
        if test_type in ["long", "full"]:
            results["long"] = test_long_position(handler, asset_name, current_price, config)
            time.sleep(2)
        
        if test_type == "full":
            current_price = handler.get_current_price(symbol)
            results["hold"] = test_hold_signal(handler, asset_name)
            time.sleep(2)
        
        if test_type in ["short", "full"]:
            current_price = handler.get_current_price(symbol)
            results["short"] = test_short_position(handler, asset_name, current_price, config)
            time.sleep(2)
        
        if test_type == "full":
            current_price = handler.get_current_price(symbol)
            config['trading']['allow_simultaneous_long_short'] = False # Force close
            logger.info("Temporarily disabling hedging to ensure shorts close longs.")
            handler.config = config 
            results["close"] = test_close_all(handler, asset_name, current_price)
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Test interrupted by user")
    
    finally:
        logger.info("Shutting down MT5 connection...")
        mt5.shutdown()

    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUITE SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, result in results.items():
        if test_type == "full" or test_type == test_name:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {test_name.upper()}")
            if not result: all_passed = False
    
    positions = handler.portfolio_manager.get_asset_positions(asset_name)
    logger.info(f"\n📈 Final State:")
    logger.info(f"  Open Positions: {len(positions)}")
    if positions:
        for pos in positions:
            logger.info(f"    {pos.position_id}: {pos.side.upper()} @ ${pos.entry_price:,.5f}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 Margin Trading Test Suite")
    parser.add_argument("--asset", default="GOLD", help="Asset to test (GOLD, USTEC, EURJPY, EURUSD)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument("--test", choices=["long", "short", "full"], default="full", help="Test type")
    
    args = parser.parse_args()
    
    if args.mode == "live":
        logger.warning("\n" + "!" * 80)
        logger.warning("⚠️  LIVE MODE SELECTED - REAL MONEY WILL BE USED ON MT5!")
        logger.warning("!" * 80)
        response = input("\nType 'CONFIRM' to proceed with live trading: ")
        if response != "CONFIRM":
            logger.info("Test cancelled")
            sys.exit(0)
    
    success = run_full_test_suite(asset_name=args.asset.upper(), mode=args.mode, test_type=args.test)
    sys.exit(0 if success else 1)
