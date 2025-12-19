#!/usr/bin/env python3
"""
Multi-Timeframe Historical Data Downloader
Downloads 4h and 15m data for BTC (Binance) and GOLD (MT5)
Saves to data folder for Analyst and Sniper strategies
"""

import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

import pandas as pd

from src.data.data_manager import DataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/download_multi_tf_data.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def download_btc_data(data_manager: DataManager, interval: str, lookback_days: int, output_file: str) -> bool:
    """
    Download BTC data from Binance for specified interval
    
    Args:
        data_manager: Initialized DataManager instance
        interval: Binance interval (e.g., '4h', '15m')
        lookback_days: Number of days to look back
        output_file: Path to save the CSV file
    
    Returns:
        bool: True if successful
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Downloading BTC {interval} Data from Binance")
    logger.info("=" * 70)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"Symbol: BTCUSDT")
    logger.info(f"Interval: {interval}")
    logger.info(f"Start Date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Lookback: {lookback_days} days")
    
    try:
        df = data_manager.fetch_binance_data(
            symbol="BTCUSDT",
            interval=interval,
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        if df.empty:
            logger.error(f"❌ No data received from Binance for {interval}")
            return False
        
        logger.info(f"✓ Fetched {len(df)} bars from Binance")
        
        # Clean data
        logger.info("Cleaning data...")
        df = data_manager.clean_data(df)
        logger.info(f"✓ After cleaning: {len(df)} bars")
        
        # Validate data quality
        if len(df) < 100:
            logger.error(f"❌ Insufficient data: {len(df)} bars (need at least 100)")
            return False
        
        # Save to CSV
        df.to_csv(output_file)
        logger.info(f"✅ Saved to: {output_file}")
        logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"   Total bars: {len(df)}")
        logger.info(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading BTC {interval} data: {e}", exc_info=True)
        return False


def find_gold_symbol(data_manager) -> str:
    """Find available gold symbol on MT5 account"""
    try:
        import MetaTrader5 as mt5
        
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error("Could not retrieve MT5 symbols")
            return None
        
        priority_patterns = ["XAUUSD", "XAU/USD", "GOLD"]
        found_symbols = []
        
        for symbol in symbols:
            symbol_name = symbol.name
            symbol_upper = symbol_name.upper()
            
            # Exclude BTC symbols
            if "BTC" not in symbol_upper:
                if any(pattern.upper() in symbol_upper for pattern in priority_patterns):
                    found_symbols.append(symbol.name)
        
        if found_symbols:
            logger.info(f"📋 Found {len(found_symbols)} gold symbols:")
            for sym in found_symbols:
                logger.info(f"   • {sym}")
            
            # Prefer XAUUSD variant
            for sym in found_symbols:
                if "XAUUSD" in sym.upper():
                    logger.info(f"✓ Using symbol: {sym} (preferred XAUUSD variant)")
                    return sym
            
            logger.info(f"✓ Using symbol: {found_symbols[0]}")
            return found_symbols[0]
        else:
            logger.warning("No gold symbols found on this account")
            return None
            
    except Exception as e:
        logger.error(f"Error finding gold symbol: {e}")
        return None


def download_gold_data(data_manager: DataManager, timeframe: str, lookback_days: int, output_file: str) -> bool:
    """
    Download Gold data from MT5 for specified timeframe
    
    Args:
        data_manager: Initialized DataManager instance
        timeframe: MT5 timeframe (e.g., 'H4', 'M15')
        lookback_days: Number of days to look back
        output_file: Path to save the CSV file
    
    Returns:
        bool: True if successful
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Downloading Gold {timeframe} Data from MT5")
    logger.info("=" * 70)
    
    # Find available gold symbol
    gold_symbol = find_gold_symbol(data_manager)
    if not gold_symbol:
        logger.error("❌ No gold symbols available on this MT5 account")
        return False
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info(f"Symbol: {gold_symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Start Date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Lookback: {lookback_days} days")
    
    try:
        df = data_manager.fetch_mt5_data(
            symbol=gold_symbol,
            timeframe=timeframe,
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        if df.empty:
            logger.error(f"❌ No data received from MT5 for {timeframe}")
            return False
        
        logger.info(f"✓ Fetched {len(df)} bars from MT5")
        
        # Clean data
        logger.info("Cleaning data...")
        df = data_manager.clean_data(df)
        logger.info(f"✓ After cleaning: {len(df)} bars")
        
        # Validate data quality
        if len(df) < 100:
            logger.error(f"❌ Insufficient data: {len(df)} bars (need at least 100)")
            return False
        
        # Save to CSV
        df.to_csv(output_file)
        logger.info(f"✅ Saved to: {output_file}")
        logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"   Total bars: {len(df)}")
        logger.info(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading Gold {timeframe} data: {e}", exc_info=True)
        return False


def main():
    """Main download pipeline"""
    logger.info("=" * 70)
    logger.info("MULTI-TIMEFRAME HISTORICAL DATA DOWNLOADER")
    logger.info("Timeframes: 4h (Analyst) + 15m (Sniper)")
    logger.info("Assets: BTC (Binance) + GOLD (MT5)")
    logger.info("=" * 70)
    
    # Load config
    config_path = Path("config/config.json")
    if not config_path.exists():
        logger.error("❌ config.json not found!")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize data manager
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Initializing Data Sources")
    logger.info("=" * 70)
    
    data_manager = DataManager(config)
    
    # Initialize Binance
    logger.info("\nInitializing Binance API...")
    if not data_manager.initialize_binance():
        logger.error("❌ Failed to initialize Binance")
        return
    logger.info("✓ Binance API ready")
    
    # Initialize MT5
    logger.info("\nInitializing MT5...")
    mt5_available = data_manager.initialize_mt5()
    if not mt5_available:
        logger.warning("⚠ MT5 not available - will skip Gold data")
    else:
        logger.info("✓ MT5 ready")
    
    # Download configuration
    # Optimized for TensorFlow/ML training
    # Recommendations based on timeframe and ML requirements:
    # - 4h: 365-730 days (2,190-4,380 candles) - captures multiple market cycles
    # - 15m: 180-365 days (17,280-35,040 candles) - rich intraday patterns
    # More data = better generalization, especially for deep learning
    
    downloads = [
        # BTC 4h (Analyst) - 730 days (~2 years) for robust ML training
        {
            "asset": "BTC",
            "source": "Binance",
            "interval": "4h",
            "lookback_days": 730,  # ~4,380 candles
            "output_file": "data/train_data_btc_4h.csv",
            "download_func": download_btc_data,
        },
        # BTC 15m (Sniper) - 365 days (1 year) for deep pattern learning
        {
            "asset": "BTC",
            "source": "Binance",
            "interval": "15m",
            "lookback_days": 365,  # ~35,040 candles
            "output_file": "data/train_data_btc_15m.csv",
            "download_func": download_btc_data,
        },
        # Gold 4h (Analyst) - 730 days
        {
            "asset": "GOLD",
            "source": "MT5",
            "timeframe": "H4",
            "lookback_days": 730,  # ~4,380 candles
            "output_file": "data/train_data_gold_4h.csv",
            "download_func": download_gold_data,
            "requires_mt5": True,
        },
        # Gold 15m (Sniper) - 365 days
        {
            "asset": "GOLD",
            "source": "MT5",
            "timeframe": "M15",
            "lookback_days": 365,  # ~35,040 candles
            "output_file": "data/train_data_gold_15m.csv",
            "download_func": download_gold_data,
            "requires_mt5": True,
        },
    ]
    
    # Download all datasets
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Downloading Historical Data")
    logger.info("=" * 70)
    
    results = {}
    
    for i, download_config in enumerate(downloads, 1):
        asset = download_config["asset"]
        source = download_config["source"]
        
        logger.info(f"\n[{i}/{len(downloads)}] {asset} from {source}")
        
        # Skip MT5 downloads if not available
        if download_config.get("requires_mt5") and not mt5_available:
            logger.warning(f"⚠ Skipping {asset} - MT5 not available")
            results[download_config["output_file"]] = False
            continue
        
        # Download based on source
        if source == "Binance":
            success = download_config["download_func"](
                data_manager=data_manager,
                interval=download_config["interval"],
                lookback_days=download_config["lookback_days"],
                output_file=download_config["output_file"],
            )
        else:  # MT5
            success = download_config["download_func"](
                data_manager=data_manager,
                timeframe=download_config["timeframe"],
                lookback_days=download_config["lookback_days"],
                output_file=download_config["output_file"],
            )
        
        results[download_config["output_file"]] = success
    
    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE - SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\n📊 Downloaded Files:")
    logger.info("─" * 70)
    
    success_count = 0
    for file_path, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {file_path}")
        if success:
            success_count += 1
            # Show file size
            file_size = Path(file_path).stat().st_size / 1024  # KB
            logger.info(f"   Size: {file_size:.1f} KB")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Success Rate: {success_count}/{len(results)}")
    logger.info("=" * 70)
    
    if success_count > 0:
        logger.info("\n✅ Downloaded files are ready for training!")
        logger.info("\nNext steps:")
        logger.info("  1. Review the CSV files in the data/ folder")
        logger.info("  2. Train your Analyst models with 4h data")
        logger.info("  3. Train your Sniper models with 15m data")
        logger.info("  4. Update your training script to use these files")
    else:
        logger.error("\n❌ No files were downloaded successfully!")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check Binance API configuration in config.json")
        logger.info("  2. Ensure MT5 is running and logged in")
        logger.info("  3. Verify MT5 credentials in config.json")
        logger.info("  4. Check logs/download_multi_tf_data.log for details")
    
    # Cleanup
    data_manager.shutdown()
    
    logger.info("\n✅ Download pipeline completed!")


if __name__ == "__main__":
    # Create required directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Check dependencies
    try:
        from binance.client import Client
    except ImportError:
        logger.error("python-binance not installed!")
        logger.info("Install with: pip install python-binance")
        sys.exit(1)
    
    try:
        import MetaTrader5 as mt5
        logger.info("MT5 module available")
    except ImportError:
        logger.warning("MetaTrader5 not installed (Gold data will be skipped)")
        logger.info("Install with: pip install MetaTrader5")
    
    main()