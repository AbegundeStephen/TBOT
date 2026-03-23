#!/usr/bin/env python3
"""
Multi-Timeframe Historical Data Downloader - INSTITUTIONAL UPGRADE
Downloads 15m (Sniper), 1H (Classic), 4H (Analyst), and 1D (Governor) data
Saves to data/raw/ folder for AI training and strategy development
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


def download_btc_data(
    data_manager: DataManager, interval: str, lookback_days: int, output_file: str
) -> bool:
    """Download BTC data from Binance"""
    logger.info(f"Downloading BTC {interval} from Binance ({lookback_days} days)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = data_manager.fetch_binance_data(
            symbol="BTCUSDT",
            interval=interval,
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if df is None or df.empty:
            logger.warning(f"⚠ No data received for BTC {interval}")
            return False

        df = data_manager.clean_data(df)
        df.to_csv(output_file)
        logger.info(f"✅ Saved {len(df)} bars to {output_file}")
        return True

    except Exception as e:
        logger.error(f"❌ Error downloading BTC {interval}: {e}")
        return False


def find_mt5_symbol(data_manager, base_name: str) -> str:
    """Find available MT5 symbol matching base name"""
    try:
        import MetaTrader5 as mt5
        symbols = mt5.symbols_get()
        if symbols is None:
            return None
        for symbol in symbols:
            if base_name.upper() in symbol.name.upper():
                return symbol.name
        return None
    except Exception:
        return None


def download_mt5_asset_data(
    data_manager: DataManager, symbol: str, timeframe: str, lookback_days: int, output_file: str
) -> bool:
    """Download asset data from MT5"""
    logger.info(f"Downloading {symbol} {timeframe} from MT5 ({lookback_days} days)")

    actual_symbol = find_mt5_symbol(data_manager, symbol)
    if not actual_symbol:
        logger.error(f"❌ Symbol {symbol} not found")
        return False

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = data_manager.fetch_mt5_data(
            symbol=actual_symbol,
            timeframe=timeframe,
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if df is None or df.empty:
            logger.warning(f"⚠ No data received for {symbol} {timeframe}")
            return False

        df = data_manager.clean_data(df)
        df.to_csv(output_file)
        logger.info(f"✅ Saved {len(df)} bars to {output_file}")
        return True

    except Exception as e:
        logger.error(f"❌ Error downloading {symbol} {timeframe}: {e}")
        return False


def main():
    """Main download pipeline"""
    logger.info("=" * 70)
    logger.info("INSTITUTIONAL DATA HARVEST - PHASE 5")
    logger.info("=" * 70)

    config_path = Path("config/config.json")
    if not config_path.exists():
        config_path = Path("config/config.template.json") # Fallback

    with open(config_path) as f:
        config = json.load(f)

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    data_manager = DataManager(config)
    
    # Try to initialize but don't hard fail if one source is down
    binance_ok = False
    try:
        binance_ok = data_manager.initialize_binance()
    except Exception as e:
        logger.error(f"Binance init crash: {e}")
        
    if not binance_ok:
        logger.warning("⚠ Binance initialization failed - BTC will be skipped")
        
    mt5_ok = False
    try:
        mt5_ok = data_manager.initialize_mt5()
    except Exception as e:
        logger.error(f"MT5 init crash: {e}")
        
    if not mt5_ok:
        logger.warning("⚠ MT5 initialization failed - MT5 assets will be skipped")

    if not binance_ok and not mt5_ok:
        logger.error("❌ Both data sources failed to initialize. Aborting.")
        return

    # Target: ~10,000 hours for 15m data = 150 days
    LOOKBACK = 450

    tasks = []
    
    # BTC Tasks
    if binance_ok:
        tasks.append({"asset": "BTC", "source": "Binance", "tf": "15m", "file": "data/raw/BTCUSDT_15m.csv"})
        tasks.append({"asset": "BTC", "source": "Binance", "tf": "1h", "file": "data/raw/BTCUSDT_1h.csv"})
        tasks.append({"asset": "BTC", "source": "Binance", "tf": "4h", "file": "data/raw/BTCUSDT_4h.csv"})
        tasks.append({"asset": "BTC", "source": "Binance", "tf": "1d", "file": "data/raw/BTCUSDT_1d.csv"})

    # MT5 Tasks
    if mt5_ok:
        mt5_assets = ["XAUUSDm", "USTECm", "EURJPYm", "EURUSDm"]
        for asset in mt5_assets:
            tasks.append({"asset": asset, "source": "MT5", "tf": "M15", "file": f"data/raw/{asset}_15m.csv"})
            tasks.append({"asset": asset, "source": "MT5", "tf": "H1", "file": f"data/raw/{asset}_1h.csv"})
            tasks.append({"asset": asset, "source": "MT5", "tf": "H4", "file": f"data/raw/{asset}_4h.csv"})
            tasks.append({"asset": asset, "source": "MT5", "tf": "D1", "file": f"data/raw/{asset}_1d.csv"})

    results = {}
    for task in tasks:
        if task["source"] == "Binance":
            results[task["file"]] = download_btc_data(data_manager, task["tf"], LOOKBACK, task["file"])
        else:
            results[task["file"]] = download_mt5_asset_data(data_manager, task["asset"], task["tf"], LOOKBACK, task["file"])

    logger.info("\n" + "=" * 70)
    logger.info("HARVEST SUMMARY")
    logger.info("=" * 70)
    for f, s in results.items():
        logger.info(f"{'✅' if s else '❌'} {f}")

    data_manager.shutdown()


if __name__ == "__main__":
    main()
