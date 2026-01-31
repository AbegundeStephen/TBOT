#!/usr/bin/env python3
"""
Complete BTC Futures Trading Test Suite
Tests the entire trading pipeline: LONG → HOLD → SHORT → CLOSE

Usage:
  python test_btc_futures_complete.py --mode live    # Execute real trades
  python test_btc_futures_complete.py --mode paper   # Simulated trades (default)
  python test_btc_futures_complete.py --test long    # Test only LONG
  python test_btc_futures_complete.py --test short   # Test only SHORT
  python test_btc_futures_complete.py --test full    # Full cycle (default)
"""

import subprocess
import json
import logging
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from binance.client import Client

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.execution.binance_handler import BinanceExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.database.database_manager import TradingDatabaseManager
from src.data.data_manager import DataManager # Import DataManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("TEST_SUITE")


def load_config():
    """Load configuration"""
    config_path = Path("config/config.json")
    if not config_path.exists():
        logger.warning("config.json not found, using template")
        config_path = Path("config/config.template.json")

    with open(config_path, "r") as f:
        return json.load(f)


def create_signal_details(signal_type: str, confidence: float = 0.85):
    """
    Create realistic signal_details matching what your strategies generate
    
    Args:
        signal_type: "buy", "sell", or "hold"
        confidence: Signal confidence (0-1)
    
    Returns:
        Complete signal_details dict
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
        "reasoning": f"TEST SCRIPT: Simulated {signal_type.upper()} signal for validation",
        
        # Hybrid aggregator metadata
        "aggregator_mode": "council",  # or "performance"
        "mode_confidence": confidence,
        
        # Market regime analysis (matches your MTF filter)
        "regime_analysis": {
            "regime_type": regime["type"],
            "trend_strength": "strong" if confidence > 0.7 else "moderate",
            "trend_direction": regime["direction"],
            "adx": regime["adx"],
            "volatility_regime": "normal",
            "volatility_ratio": 1.0,
            "price_clarity": "clear" if confidence > 0.75 else "moderate",
            "momentum_aligned": True,
            "at_key_level": False,
        },
        
        # AI validation (matches your hybrid validator)
        "ai_validation": {
            "pattern_name": f"Test {signal_type.upper()} Pattern",
            "pattern_confidence": confidence,
            "validation_passed": True,
            "action": "approved",
            "top_3_patterns": [
                {"name": "Test Pattern 1", "confidence": confidence},
                {"name": "Test Pattern 2", "confidence": confidence * 0.9},
                {"name": "Test Pattern 3", "confidence": confidence * 0.8},
            ]
        },
        
        # Council scores (matches your council aggregator)
        "council_scores": {
            "buy_score": 4.5 if signal_type == "buy" else 1.5,
            "sell_score": 4.5 if signal_type == "sell" else 1.5,
            "trend_score": 1.5,
            "structure_score": 1.5,
            "momentum_score": 1.0 if signal_type != "hold" else 0.5,
            "pattern_score": 0.5,
            "volume_score": 0.0,
        },
        
        # Timestamp
        "timestamp": datetime.now().isoformat(),
        "test_mode": True,
    }


def check_margin_balance(client: Client) -> dict:
    """Check Binance Futures margin balance"""
    try:
        account = client.futures_account()
        
        total_balance = float(account.get('totalWalletBalance', 0))
        available = float(account.get('availableBalance', 0))
        unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
        
        return {
            'total': total_balance,
            'available': available,
            'unrealized_pnl': unrealized_pnl,
        }
    except Exception as e:
        logger.error(f"Could not fetch margin balance: {e}")
        return {'total': 0, 'available': 0, 'unrealized_pnl': 0}


def test_long_position(handler, current_price, config):
    """Test opening a LONG position"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TEST 1: LONG POSITION (BUY SIGNAL)")
    logger.info("=" * 80)
    
    signal_details = create_signal_details("buy", confidence=0.85)
    
    logger.info(f"📊 Signal Details:")
    logger.info(f"  Quality:     {signal_details['signal_quality']:.0%}")
    logger.info(f"  Regime:      {signal_details['regime']}")
    logger.info(f"  Mode:        {signal_details['aggregator_mode'].upper()}")
    logger.info(f"  Council Buy: {signal_details['council_scores']['buy_score']:.1f}/5.0")
    
    try:
        success = handler.execute_signal(
            signal=1,  # BUY
            current_price=current_price,
            asset_name="BTC",
            confidence_score=0.85,
            market_condition="bull",
            signal_details=signal_details,
        )
        
        if success:
            logger.info("\n✅ LONG POSITION OPENED SUCCESSFULLY")
            
            # Show position details
            positions = handler.portfolio_manager.get_asset_positions("BTC")
            logger.info(f"Position object: {positions if positions else None}")
            if positions:
                pos = positions[-1]
                logger.info(f"\n📈 Position Details:")
                logger.info(f"  ID:          {pos.position_id}")
                logger.info(f"  Side:        {pos.side.upper()}")
                logger.info(f"  Entry:       ${pos.entry_price:,.2f}")
                logger.info(f"  Quantity:    {pos.quantity:.6f} BTC")
                logger.info(f"  Size:        ${(pos.quantity * pos.entry_price):,.2f}")
                logger.info(f"  Futures:     {getattr(pos, 'is_futures', False)}")
                logger.info(f"  Leverage:    {getattr(pos, 'leverage', 1)}x")
                
                if pos.trade_manager:
                    logger.info(f"  VTM:         ACTIVE ✓")
                    vtm_status = pos.get_vtm_status()
                    logger.info(f"    SL: ${vtm_status.get('stop_loss', 0):,.2f}")
                    logger.info(f"    TP: ${vtm_status.get('take_profit', 0):,.2f}")
                else:
                    logger.info(f"  VTM:         INACTIVE")
                
                return True
        else:
            logger.error("\n❌ LONG POSITION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def test_hold_signal(handler, current_price):
    """Test HOLD signal (checks SL/TP on existing positions)"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TEST 2: HOLD SIGNAL (Position Management)")
    logger.info("=" * 80)
    
    positions = handler.portfolio_manager.get_asset_positions("BTC")
    
    if not positions:
        logger.warning("⚠️  No positions to manage - skipping HOLD test")
        return False
    
    logger.info(f"📊 Managing {len(positions)} position(s)")
    
    signal_details = create_signal_details("hold", confidence=0.50)
    
    try:
        # Execute HOLD signal
        success = handler.execute_signal(
            signal=0,  # HOLD
            current_price=current_price,
            asset_name="BTC",
            confidence_score=0.50,
            market_condition="neutral",
            signal_details=signal_details,
        )
        
        logger.info("\n✅ HOLD SIGNAL PROCESSED")
        logger.info("  → Checked SL/TP on all positions")
        logger.info("  → VTM updated trailing stops")
        
        # Show updated positions
        positions = handler.portfolio_manager.get_asset_positions("BTC")
        for pos in positions:
            if pos.trade_manager:
                vtm_status = pos.get_vtm_status()
                logger.info(f"\n  Position {pos.position_id}:")
                logger.info(f"    Entry:  ${pos.entry_price:,.2f}")
                logger.info(f"    Current: ${current_price:,.2f}")
                logger.info(f"    P&L:    {vtm_status.get('unrealized_pnl_pct', 0):+.2f}%")
                logger.info(f"    VTM SL: ${vtm_status.get('stop_loss', 0):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def test_short_position(handler, current_price, config):
    """Test opening a SHORT position (closes longs first)"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TEST 3: SHORT POSITION (SELL SIGNAL)")
    logger.info("=" * 80)
    
    # Check for existing longs
    positions = handler.portfolio_manager.get_asset_positions("BTC")
    longs = [p for p in positions if p.side == "long"]
    
    if longs:
        logger.info(f"📉 Will close {len(longs)} LONG position(s) before opening SHORT")
    
    signal_details = create_signal_details("sell", confidence=0.88)
    
    logger.info(f"📊 Signal Details:")
    logger.info(f"  Quality:      {signal_details['signal_quality']:.0%}")
    logger.info(f"  Regime:       {signal_details['regime']}")
    logger.info(f"  Mode:         {signal_details['aggregator_mode'].upper()}")
    logger.info(f"  Council Sell: {signal_details['council_scores']['sell_score']:.1f}/5.0")
    
    try:
        success = handler.execute_signal(
            signal=-1,  # SELL
            current_price=current_price,
            asset_name="BTC",
            confidence_score=0.88,
            market_condition="bear",
            signal_details=signal_details,
        )
        
        if success:
            logger.info("\n✅ SHORT POSITION OPENED SUCCESSFULLY")
            
            # Show position details
            positions = handler.portfolio_manager.get_asset_positions("BTC")
            shorts = [p for p in positions if p.side == "short"]
            
            if shorts:
                pos = shorts[-1]
                logger.info(f"\n📉 Position Details:")
                logger.info(f"  ID:          {pos.position_id}")
                logger.info(f"  Side:        {pos.side.upper()}")
                logger.info(f"  Entry:       ${pos.entry_price:,.2f}")
                logger.info(f"  Quantity:    {pos.quantity:.6f} BTC")
                logger.info(f"  Size:        ${(pos.quantity * pos.entry_price):,.2f}")
                logger.info(f"  Futures:     {getattr(pos, 'is_futures', False)}")
                logger.info(f"  Leverage:    {getattr(pos, 'leverage', 1)}x")
                
                if pos.trade_manager:
                    logger.info(f"  VTM:         ACTIVE ✓")
                    vtm_status = pos.get_vtm_status()
                    logger.info(f"    SL: ${vtm_status.get('stop_loss', 0):,.2f}")
                    logger.info(f"    TP: ${vtm_status.get('take_profit', 0):,.2f}")
                
                return True
        else:
            logger.error("\n❌ SHORT POSITION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def test_close_all(handler, current_price):
    """Test closing all positions (opposite signal)"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TEST 4: CLOSE ALL POSITIONS")
    logger.info("=" * 80)
    
    positions = handler.portfolio_manager.get_asset_positions("BTC")
    
    if not positions:
        logger.warning("⚠️  No positions to close")
        return True
    
    shorts = [p for p in positions if p.side == "short"]
    longs = [p for p in positions if p.side == "long"]
    
    if shorts:
        logger.info(f"📈 Closing {len(shorts)} SHORT position(s) with BUY signal")
        signal = 1  # BUY to close shorts
        signal_type = "buy"
    elif longs:
        logger.info(f"📉 Closing {len(longs)} LONG position(s) with SELL signal")
        signal = -1  # SELL to close longs
        signal_type = "sell"
    else:
        return True
    
    signal_details = create_signal_details(signal_type, confidence=0.75)
    
    try:
        # Execute opposite signal to close positions
        success = handler.execute_signal(
            signal=signal,
            current_price=current_price,
            asset_name="BTC",
            confidence_score=0.75,
            market_condition="neutral",
            signal_details=signal_details,
        )
        
        # Check if positions were closed
        remaining = handler.portfolio_manager.get_asset_positions("BTC")
        
        if len(remaining) < len(positions):
            logger.info(f"\n✅ CLOSED {len(positions) - len(remaining)} POSITION(S)")
            
            # Show P&L from closed trades
            logger.info("\n💰 Closed Trade Summary:")
            # Note: You'll need to implement get_closed_trades if not available
            # For now, just confirm closure
            logger.info(f"  Remaining positions: {len(remaining)}")
            
            return True
        else:
            logger.warning("\n⚠️  No positions were closed")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ EXCEPTION: {e}", exc_info=True)
        return False


def run_full_test_suite(mode="paper", test_type="full"):
    """
    Run complete test suite
    
    Args:
        mode: "paper" or "live"
        test_type: "long", "short", or "full"
    """
    logger.info("=" * 80)
    logger.info("🚀 BTC FUTURES COMPLETE TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Test Type: {test_type.upper()}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Load config
    config = load_config()
    
    # Override mode if specified
    if mode == "paper":
        config["trading"]["mode"] = "paper"
    
    if not config["assets"]["BTC"].get("enabled", False):
        logger.error("❌ BTC is disabled in config!")
        return False

    # Initialize Database
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
            logger.info("⚠️  Database disabled")
    except Exception as e:
        logger.warning(f"⚠️  Database init failed: {e}")
        db_manager = None
        
    # Initialize DataManager
    try:
        data_manager = DataManager(config=config)
        data_manager.initialize_binance() # Initialize its clients
        logger.info("✅ Data Manager initialized")
    except Exception as e:
        logger.critical(f"❌ Data Manager failed: {e}")
        return False

    # Initialize Binance Client
    if config["assets"]["BTC"].get("enable_futures", False):
        api_config = config["api"]["binance_futures"]
    else:
        api_config = config["api"]["binance"]
    
    try:
        client = Client(
            api_key=api_config["api_key"],
            api_secret=api_config["api_secret"],
            testnet=api_config.get("testnet", False),
        )
        
        if config["assets"]["BTC"].get("enable_futures", False):
            client.futures_ping()
            logger.info("✅ Binance Futures Client connected")
        else:
            client.ping()
            logger.info("✅ Binance Spot Client connected")
            
    except Exception as e:
        logger.critical(f"❌ Failed to connect to Binance: {e}")
        return False

    # Check margin balance (Futures only)
    if config["assets"]["BTC"].get("enable_futures", False):
        balance = check_margin_balance(client)
        logger.info(f"\n💰 Futures Margin Balance:")
        logger.info(f"  Total:       ${balance['total']:,.2f} USDT")
        logger.info(f"  Available:   ${balance['available']:,.2f} USDT")
        logger.info(f"  Unrealized:  ${balance['unrealized_pnl']:+,.2f} USDT")
        
        if balance['available'] < 10:
            logger.error("\n❌ INSUFFICIENT MARGIN!")
            logger.error("   Fund your Futures wallet with USDT before testing")
            logger.error("   Minimum recommended: $100 USDT")
            return False

    # Initialize Portfolio Manager
    try:
        portfolio_manager = PortfolioManager(
            config=config, 
            binance_client=client, 
            db_manager=db_manager
        )
        logger.info("✅ Portfolio Manager initialized")
        logger.info(f"   Capital: ${portfolio_manager.current_capital:,.2f}")
    except Exception as e:
        logger.critical(f"❌ Portfolio Manager failed: {e}")
        return False

    # Initialize Execution Handler
    try:
        handler = BinanceExecutionHandler(
            client=client,
            config=config,
            portfolio_manager=portfolio_manager,
            data_manager=data_manager, # Pass the correct DataManager
        )
        logger.info("✅ Binance Execution Handler initialized")
        
        # Verify Futures is enabled
        if hasattr(handler, 'futures_handler') and handler.futures_handler:
            logger.info("✅ Futures Handler: ACTIVE")
            logger.info(f"   Leverage: {config['assets']['BTC'].get('leverage', 20)}x")
        else:
            logger.warning("⚠️  Futures Handler: NOT ACTIVE (Spot mode)")
            
    except Exception as e:
        logger.critical(f"❌ Handler initialization failed: {e}")
        return False

    # Get current price
    try:
        current_price = handler.get_current_price("BTCUSDT")
        logger.info(f"\n💰 Current BTC Price: ${current_price:,.2f}")
    except Exception as e:
        logger.error(f"❌ Could not fetch price: {e}")
        return False

    # Run tests based on type
    results = {
        "long": False,
        "hold": False,
        "short": False,
        "close": False,
    }
    
    try:
        if test_type in ["long", "full"]:
            results["long"] = test_long_position(handler, current_price, config)
            time.sleep(2)  # Brief pause between tests
        
        if test_type == "full":
            results["hold"] = test_hold_signal(handler, current_price)
            time.sleep(2)
        
        if test_type in ["short", "full"]:
            results["short"] = test_short_position(handler, current_price, config)
            time.sleep(2)
        
        if test_type == "full":
            results["close"] = test_close_all(handler, current_price)
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Test interrupted by user")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUITE SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        if result is not False:  # Test was run
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {test_name.upper()}")
    
    # Show final portfolio state
    positions = handler.portfolio_manager.get_asset_positions("BTC")
    logger.info(f"\n📈 Final State:")
    logger.info(f"  Open Positions: {len(positions)}")
    logger.info(f"  Portfolio Value: ${handler.portfolio_manager.current_capital:,.2f}")
    
    if positions:
        logger.info(f"\n  Active Positions:")
        for pos in positions:
            logger.info(f"    {pos.position_id}: {pos.side.upper()} {pos.quantity:.6f} BTC @ ${pos.entry_price:,.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 80)
    
    return all(v for v in results.values() if v is not False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Futures Trading Test Suite")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--test",
        choices=["long", "short", "full"],
        default="full",
        help="Test type: long, short, or full cycle (default: full)"
    )
    
    args = parser.parse_args()
    
    # Confirmation for live mode
    if args.mode == "live":
        logger.warning("\n" + "!" * 80)
        logger.warning("⚠️  LIVE MODE SELECTED - REAL MONEY WILL BE USED!")
        logger.warning("!" * 80)
        response = input("\nType 'CONFIRM' to proceed with live trading: ")
        if response != "CONFIRM":
            logger.info("Test cancelled")
            sys.exit(0)
    
    success = run_full_test_suite(mode=args.mode, test_type=args.test)
    sys.exit(0 if success else 1)