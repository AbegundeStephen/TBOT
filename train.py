#!/usr/bin/env python3
"""
 Training Script - Generates More Actionable Signals
Properly loads config.json values
NOW WITH EMA STRATEGY INTEGRATION
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
import numpy as np

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training_multi_source.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def validate_data_quality(df: pd.DataFrame, asset_name: str, min_bars: int) -> bool:
    """Validate data quality and sufficiency"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Data Quality Check - {asset_name}")
    logger.info("=" * 60)

    if df.empty:
        logger.error("❌ DataFrame is empty")
        return False

    logger.info(f"Total bars: {len(df)}")
    if len(df) < min_bars:
        logger.error(f"❌ Insufficient data: {len(df)} bars (need {min_bars})")
        return False
    logger.info(f"✓ Sufficient bars: {len(df)} >= {min_bars}")

    date_range = df.index[-1] - df.index[0]
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Time span: {date_range.days} days")

    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"⚠ Missing values: {missing}")
    else:
        logger.info("✓ No missing values")

    zero_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if zero_prices > 0:
        logger.error(f"❌ Found {zero_prices} bars with zero/negative prices")
        return False
    logger.info("✓ All prices positive")

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()

    if invalid_ohlc > 0:
        logger.error(f"❌ Found {invalid_ohlc} bars with invalid OHLC")
        return False
    logger.info("✓ OHLC logic valid")

    logger.info(f"\nPrice Statistics:")
    logger.info(f"  Min: ${df['close'].min():,.2f}")
    logger.info(f"  Max: ${df['close'].max():,.2f}")
    logger.info(f"  Mean: ${df['close'].mean():,.2f}")
    logger.info(f"  Latest: ${df['close'].iloc[-1]:,.2f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Data quality check PASSED for {asset_name}")
    logger.info("=" * 60)

    return True


def fetch_binance_training_data(data_manager: DataManager, config: dict) -> pd.DataFrame:
    """Fetch training data from Binance"""
    logger.info("\n" + "=" * 70)
    logger.info(f"Fetching {config['symbol']} Data from Binance")
    logger.info("=" * 70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["lookback_days"])

    try:
        df = data_manager.fetch_binance_data(
            symbol=config["symbol"],
            interval=config["interval"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            logger.error(f"❌ No data received for {config['symbol']}")
            return pd.DataFrame()

        df = data_manager.clean_data(df)
        return df

    except Exception as e:
        logger.error(f"❌ Error fetching {config['symbol']} data: {e}")
        return pd.DataFrame()


def verify_mt5_symbol(symbol: str) -> bool:
    """Verify if symbol is available on MT5 account"""
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"❌ Symbol {symbol} not found on this MT5 account")
            return False
        
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"❌ Failed to select symbol {symbol}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error verifying symbol {symbol}: {e}")
        return False


def fetch_mt5_training_data(data_manager: DataManager, config: dict) -> pd.DataFrame:
    """Fetch training data from MT5"""
    symbol = config["symbol"]
    logger.info("\n" + "=" * 70)
    logger.info(f"Fetching {symbol} Data from MT5")
    logger.info("=" * 70)

    if not verify_mt5_symbol(symbol):
        return pd.DataFrame()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["lookback_days"])

    try:
        df = data_manager.fetch_mt5_data(
            symbol=symbol,
            timeframe=config["timeframe"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            logger.error(f"❌ No data received for {symbol}")
            return pd.DataFrame()

        df = data_manager.clean_data(df)
        return df

    except Exception as e:
        logger.error(f"❌ Error fetching {symbol} data: {e}")
        return pd.DataFrame()


def train_asset_strategies(
    asset_key: str, train_df: pd.DataFrame, config: dict
) -> dict:
    """
    Train ALL THREE strategies for an asset using config.json values
    """
    results = {
        "asset": asset_key,
        "train_bars": len(train_df),
        "mean_reversion": {},
        "trend_following": {},
        "ema_strategy": {},
    }

    logger.info("\n" + "=" * 70)
    logger.info(f"Training Models for {asset_key}")
    logger.info("=" * 70)

    # Get configs from config.json
    try:
        mr_config = config["strategy_configs"]["mean_reversion"][asset_key]
        tf_config = config["strategy_configs"]["trend_following"][asset_key]
        ema_config = config["strategy_configs"]["exponential_moving_averages"][asset_key]
    except KeyError as e:
        logger.error(f"❌ Missing strategy configuration for {asset_key}: {e}")
        return results

    # =====================================================
    # 1. Train Mean Reversion
    # =====================================================
    try:
        mr_strategy = MeanReversionStrategy(mr_config)
        mr_model_path = f"models/mean_reversion_{asset_key.lower()}.pkl"
        mr_metrics = mr_strategy.train_model(train_df, mr_model_path)
        results["mean_reversion"] = mr_metrics
    except Exception as e:
        logger.error(f"❌ Mean Reversion training failed for {asset_key}: {e}")

    # =====================================================
    # 2. Train Trend Following
    # =====================================================
    try:
        tf_strategy = TrendFollowingStrategy(tf_config)
        tf_model_path = f"models/trend_following_{asset_key.lower()}.pkl"
        tf_metrics = tf_strategy.train_model(train_df, tf_model_path)
        results["trend_following"] = tf_metrics
    except Exception as e:
        logger.error(f"❌ Trend Following training failed for {asset_key}: {e}")

    # =====================================================
    # 3. Train EMA Strategy
    # =====================================================
    try:
        ema_strategy = EMAStrategy(ema_config)
        ema_model_path = f"models/ema_strategy_{asset_key.lower()}.pkl"
        ema_metrics = ema_strategy.train_model(train_df, ema_model_path)
        results["ema_strategy"] = ema_metrics
    except Exception as e:
        logger.error(f"❌ EMA Strategy training failed for {asset_key}: {e}")

    return results


def calculate_data_correlation(data_map: dict) -> Optional[pd.DataFrame]:
    """Calculate correlation matrix between all assets"""
    try:
        closes = {}
        for asset, df in data_map.items():
            s = df["close"].copy()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            closes[asset] = s

        merged = pd.DataFrame(closes).dropna()
        if len(merged) < 50:
            return None

        returns = merged.pct_change().dropna()
        correlation = returns.corr()
        
        logger.info(f"\n{'='*70}")
        logger.info("Multi-Asset Correlation Matrix")
        logger.info("=" * 70)
        logger.info(f"\n{correlation}")
        
        return correlation

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return None


def main():
    """Main training pipeline"""
    logger.info("=" * 70)
    logger.info("MULTI-ASSET TRADING BOT TRAINING ENGINE")
    logger.info("Strategies: Mean Reversion + Trend Following + EMA Crossover")
    logger.info("=" * 70)

    config_path = Path("config/config.json")
    if not config_path.exists():
        logger.error("config.json not found!")
        return

    with open(config_path) as f:
        config = json.load(f)

    data_manager = DataManager(config)
    binance_ok = data_manager.initialize_binance()
    mt5_ok = data_manager.initialize_mt5()

    asset_data = {}
<<<<<<< Updated upstream

    # Asset configs to process
    assets_to_train = ["BTC", "GOLD", "USTEC", "EURJPY", "EURUSD"]
    
    for asset_key in assets_to_train:
        if asset_key not in config["assets"]:
            logger.warning(f"Asset {asset_key} not in config, skipping.")
            continue
            
        asset_cfg = config["assets"][asset_key]
        exchange = asset_cfg.get("exchange", "binance")
        
        # Try local raw data first if fetch fails or for speed
        raw_file = f"data/raw/{asset_cfg.get('symbol', asset_key)}_1h.csv"
        if asset_key == "BTC": raw_file = "data/raw/BTCUSDT_1h.csv"
        elif asset_key == "GOLD": raw_file = "data/raw/XAUUSDm_1h.csv"
        else: raw_file = f"data/raw/{asset_key}m_1h.csv"
        
        df = pd.DataFrame()
        if Path(raw_file).exists():
            logger.info(f"Loading {asset_key} from local file: {raw_file}")
            df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
        
        if df.empty:
            if exchange == "binance" and binance_ok:
                fetch_cfg = {
                    "symbol": asset_cfg["symbol"],
                    "interval": asset_cfg.get("interval", "1h"),
                    "lookback_days": asset_cfg["lookback_days"],
                    "min_bars_required": asset_cfg["min_bars_training"],
                }
                df = fetch_btc_data(data_manager, fetch_cfg)
            elif exchange == "mt5" and mt5_ok:
                fetch_cfg = {
                    "symbol": asset_cfg["symbol"],
                    "timeframe": asset_cfg.get("timeframe", "H1"),
                    "lookback_days": asset_cfg["lookback_days"],
                    "min_bars_required": asset_cfg["min_bars_training"],
                }
                df = fetch_gold_data(data_manager, fetch_cfg) # fetch_gold_data uses find_available_gold_symbol, might need fix for generic mt5
        
        if not df.empty and validate_data_quality(df, asset_key, asset_cfg["min_bars_training"]):
            asset_data[asset_key] = df
            logger.info(f"✅ {asset_key} data ready for training")
        else:
            logger.error(f"❌ {asset_key} data failed validation or could not be fetched")
=======
    enabled_assets = [a for a, cfg in config["assets"].items() if cfg.get("enabled", False)]
    
    for asset_key in enabled_assets:
        asset_cfg = config["assets"][asset_key]
        exchange = asset_cfg.get("exchange", "binance")
        symbol = asset_cfg.get("symbol")
        
        # Try local raw data first
        raw_file = f"data/raw/{symbol}_1h.csv"
        df = pd.DataFrame()
        
        if Path(raw_file).exists():
            logger.info(f"Loading {asset_key} from local file: {raw_file}")
            df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
        
        if df.empty:
            fetch_cfg = {
                "symbol": symbol,
                "lookback_days": asset_cfg.get("lookback_days", 730),
                "interval": asset_cfg.get("interval", "1h"),
                "timeframe": asset_cfg.get("timeframe", "H1"),
            }
            if exchange == "binance" and binance_ok:
                df = fetch_binance_training_data(data_manager, fetch_cfg)
            elif exchange == "mt5" and mt5_ok:
                df = fetch_mt5_training_data(data_manager, fetch_cfg)
        
        if not df.empty and validate_data_quality(df, asset_key, asset_cfg.get("min_bars_training", 1000)):
            asset_data[asset_key] = df
            logger.info(f"✅ {asset_key} data ready")
>>>>>>> Stashed changes

    if not asset_data:
        logger.error("❌ No data available for training!")
        return

    # Correlation
    correlation_matrix = calculate_data_correlation(asset_data)

    # Train/Test Split & Train
    training_results = {}
    train_test_stats = {}
    
    for asset_key, df in asset_data.items():
        train_df, test_df = data_manager.split_train_test(df, train_pct=0.8)
        
        # Save split data
        train_df.to_csv(f"data/train_data_{asset_key.lower()}.csv")
        test_df.to_csv(f"data/test_data_{asset_key.lower()}.csv")
        
        results = train_asset_strategies(asset_key, train_df, config)
        training_results[asset_key] = results
        
        train_test_stats[asset_key] = {
            "total_bars": len(df),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "range": f"{df.index[0]} to {df.index[-1]}"
        }

    # Metadata & Report
    metadata = {
        "training_date": datetime.now().isoformat(),
        "assets_trained": list(training_results.keys()),
        "results": training_results,
        "data_stats": train_test_stats
    }

    with open("models/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    for asset, res in training_results.items():
        logger.info(f"\n{asset}:")
        for strat in ["mean_reversion", "trend_following", "ema_strategy"]:
            metrics = res.get(strat, {})
            status = "✅" if metrics.get("success") else "❌"
            acc = metrics.get("cv_mean_accuracy", 0)
            logger.info(f"  {status} {strat:18}: {acc:.2%}")

    data_manager.shutdown()
    logger.info("\n✅ Multi-asset training pipeline completed!")
