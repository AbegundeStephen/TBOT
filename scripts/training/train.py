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
import os
from dotenv import load_dotenv

# ✅ FIX: Add project root to sys.path (relative to this script)
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ✅ FIX: Load .env from project root
load_dotenv(project_root / ".env")

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
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "training_multi_source.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def validate_data_quality(df: pd.DataFrame, asset_name: str, min_bars: int) -> bool:
    """Validate data quality and sufficiency"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Data Quality Check - {asset_name}")
    logger.info("=" * 60)

    if df is None or df.empty:
        logger.error("❌ DataFrame is empty")
        return False

    logger.info(f"Total bars: {len(df)}")
    if len(df) < min_bars:
        logger.error(f"❌ Insufficient data: {len(df)} bars (need {min_bars})")
        return False
    logger.info(f"✓ Sufficient bars: {len(df)} >= {min_bars}")

    # Ensure index is datetime-like
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            logger.error(f"❌ Could not convert index to datetime: {e}")
            return False

    date_range = df.index[-1] - df.index[0]
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Time span: {date_range.days} days")

    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"⚠ Missing values: {missing}")
    else:
        logger.info("✓ No missing values")

    # Ensure OHLC are numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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


def fetch_binance_asset_data(data_manager: DataManager, config: dict) -> pd.DataFrame:
    """Fetch asset data from Binance"""
    logger.info("\n" + "=" * 70)
    logger.info(f"Fetching {config['symbol']} Data from Binance")
    logger.info("=" * 70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["lookback_days"])

    logger.info(f"Symbol: {config['symbol']}")
    logger.info(f"Interval: {config.get('interval', '1h')}")
    logger.info(
        f"Start Date: {start_date.strftime('%Y-%m-%d')} (going back {config['lookback_days']} days)"
    )
    logger.info(f"End Date: {end_date.strftime('%Y-%m-%d')} (today)")

    try:
        df = data_manager.fetch_binance_data(
            symbol=config["symbol"],
            interval=config.get("interval", "1h"),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            logger.error("❌ No data received from Binance")
            return pd.DataFrame()

        logger.info(f"✓ Fetched {len(df)} bars from Binance")

        logger.info("\nCleaning data...")
        df = data_manager.clean_data(df)
        logger.info(f"✓ After cleaning: {len(df)} bars")

        return df

    except Exception as e:
        logger.error(f"❌ Error fetching Binance data: {e}", exc_info=True)
        return pd.DataFrame()


def find_mt5_symbol(base_name: str) -> str:
    """Find available MT5 symbol matching base name (e.g., BTCUSDT or BTCUSDTm)"""
    try:
        import MetaTrader5 as mt5
        symbols = mt5.symbols_get()
        if symbols is None:
            return None
            
        # Try exact match first
        for s in symbols:
            if s.name.upper() == base_name.upper():
                return s.name
                
        # Try contains match (handle suffixes like 'm')
        for s in symbols:
            if base_name.upper() in s.name.upper():
                return s.name
        return None
    except Exception:
        return None


def fetch_mt5_asset_data(data_manager: DataManager, config: dict) -> pd.DataFrame:
    """Fetch asset data from MT5"""
    logger.info("\n" + "=" * 70)
    logger.info(f"Fetching {config['symbol']} Data from MT5")
    logger.info("=" * 70)

    # Resolve actual symbol (handle suffixes)
    actual_symbol = find_mt5_symbol(config["symbol"])
    if not actual_symbol:
        logger.error(f"❌ Symbol {config['symbol']} not found on this MT5 account")
        return pd.DataFrame()

    logger.info(f"Resolved Symbol: {actual_symbol}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["lookback_days"])

    logger.info(f"Timeframe: {config.get('timeframe', 'H1')}")
    logger.info(
        f"Start Date: {start_date.strftime('%Y-%m-%d')} (going back {config['lookback_days']} days)"
    )
    logger.info(f"End Date: {end_date.strftime('%Y-%m-%d')} (today)")

    try:
        df = data_manager.fetch_mt5_data(
            symbol=actual_symbol,
            timeframe=config.get("timeframe", "H1"),
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if df.empty:
            logger.error(f"❌ No data received from MT5 for {actual_symbol}")
            return pd.DataFrame()

        logger.info(f"✓ Fetched {len(df)} bars from MT5")

        logger.info("\nCleaning data...")
        df = data_manager.clean_data(df)
        logger.info(f"✓ After cleaning: {len(df)} bars")

        return df

    except Exception as e:
        logger.error(f"❌ Error fetching MT5 data: {e}", exc_info=True)
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
    mr_config = config["strategy_configs"]["mean_reversion"][asset_key]
    tf_config = config["strategy_configs"]["trend_following"][asset_key]
    ema_config = config["strategy_configs"]["exponential_moving_averages"][asset_key]

    # Log the actual config being used
    logger.info(f"\n📋 Configuration for {asset_key}:")
    logger.info(f"  Mean Reversion:")
    logger.info(f"    RSI: {mr_config['rsi_oversold']}/{mr_config['rsi_overbought']}")
    logger.info(
        f"    BB: {mr_config['bb_lower_threshold']}/{mr_config['bb_upper_threshold']}"
    )
    logger.info(f"    Min Return: {mr_config['min_return_threshold']:.3f}")
    logger.info(f"    Min Conditions: {mr_config['min_conditions']}")
    logger.info(f"  Trend Following:")
    logger.info(f"    MA: {tf_config['fast_ma']}/{tf_config['slow_ma']}")
    logger.info(f"    ADX Threshold: {tf_config['adx_threshold']}")
    logger.info(f"    Min Return: {tf_config['min_return_threshold']:.3f}")
    logger.info(f"  EMA Strategy:")
    logger.info(f"    EMA: {ema_config['ema_fast']}/{ema_config['ema_slow']}")
    logger.info(f"    Min Distance: {ema_config['min_distance_pct']}%")
    logger.info(f"    Min Return: {ema_config['min_return_threshold']:.3f}")
    logger.info(f"    Min Conditions: {ema_config['min_conditions']}")

    # =====================================================
    # 1. Train Mean Reversion
    # =====================================================
    logger.info(f"\n{'─'*70}")
    logger.info("1. Mean Reversion Strategy")
    logger.info("─" * 70)

    try:
        mr_strategy = MeanReversionStrategy(mr_config)
        mr_model_path = project_root / "models" / f"mean_reversion_{asset_key.lower()}.pkl"

        logger.info(f"Training on {len(train_df)} bars...")
        mr_metrics = mr_strategy.train_model(train_df, mr_model_path)

        if mr_metrics.get("success"):
            logger.info(f"\n✅ Mean Reversion - {asset_key} TRAINED")
            logger.info(
                f"   CV Accuracy: {mr_metrics.get('cv_mean_accuracy', 0):.2%} ± {mr_metrics.get('cv_std_accuracy', 0):.2%}"
            )
            logger.info(f"   Train Samples: {mr_metrics.get('train_samples', 0)}")
            logger.info(f"   Features: {mr_metrics.get('n_features', 0)}")
            logger.info(f"   Model saved: {mr_model_path}")
        else:
            logger.error(f"\n❌ Mean Reversion - {asset_key} FAILED")
            logger.error(f"   Error: {mr_metrics.get('error', 'Unknown')}")

        results["mean_reversion"] = mr_metrics

    except Exception as e:
        logger.error(f"❌ Exception training Mean Reversion: {e}", exc_info=True)
        results["mean_reversion"] = {"success": False, "error": str(e)}

    # =====================================================
    # 2. Train Trend Following
    # =====================================================
    logger.info(f"\n{'─'*70}")
    logger.info("2. Trend Following Strategy")
    logger.info("─" * 70)

    try:
        tf_strategy = TrendFollowingStrategy(tf_config)
        tf_model_path = project_root / "models" / f"trend_following_{asset_key.lower()}.pkl"

        logger.info(f"Training on {len(train_df)} bars...")
        tf_metrics = tf_strategy.train_model(train_df, tf_model_path)

        if tf_metrics.get("success"):
            logger.info(f"\n✅ Trend Following - {asset_key} TRAINED")
            logger.info(
                f"   CV Accuracy: {tf_metrics.get('cv_mean_accuracy', 0):.2%} ± {tf_metrics.get('cv_std_accuracy', 0):.2%}"
            )
            logger.info(f"   Train Samples: {tf_metrics.get('train_samples', 0)}")
            logger.info(f"   Features: {tf_metrics.get('n_features', 0)}")
            logger.info(f"   Model saved: {tf_model_path}")
        else:
            logger.error(f"\n❌ Trend Following - {asset_key} FAILED")
            logger.error(f"   Error: {tf_metrics.get('error', 'Unknown')}")

        results["trend_following"] = tf_metrics

    except Exception as e:
        logger.error(f"❌ Exception training Trend Following: {e}", exc_info=True)
        results["trend_following"] = {"success": False, "error": str(e)}

    # =====================================================
    # 3. Train EMA Strategy (NEW!)
    # =====================================================
    logger.info(f"\n{'─'*70}")
    logger.info("3. EMA Crossover Strategy")
    logger.info("─" * 70)

    try:
        ema_strategy = EMAStrategy(ema_config)
        ema_model_path = project_root / "models" / f"ema_strategy_{asset_key.lower()}.pkl"

        logger.info(f"Training on {len(train_df)} bars...")
        ema_metrics = ema_strategy.train_model(train_df, ema_model_path)

        if ema_metrics.get("success"):
            logger.info(f"\n✅ EMA Strategy - {asset_key} TRAINED")
            logger.info(
                f"   CV Accuracy: {ema_metrics.get('cv_mean_accuracy', 0):.2%} ± {ema_metrics.get('cv_std_accuracy', 0):.2%}"
            )
            logger.info(f"   Train Samples: {ema_metrics.get('train_samples', 0)}")
            logger.info(f"   Features: {ema_metrics.get('n_features', 0)}")
            logger.info(f"   Model saved: {ema_model_path}")
        else:
            logger.error(f"\n❌ EMA Strategy - {asset_key} FAILED")
            logger.error(f"   Error: {ema_metrics.get('error', 'Unknown')}")

        results["ema_strategy"] = ema_metrics

    except Exception as e:
        logger.error(f"❌ Exception training EMA Strategy: {e}", exc_info=True)
        results["ema_strategy"] = {"success": False, "error": str(e)}

    return results


def calculate_data_correlation(btc_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """
    Calculate correlation between BTC and Gold
    Handles timezone-aware and timezone-naive dataframes
    """
    try:
        # : Normalize timezones before merging
        btc_close = btc_df[["close"]].rename(columns={"close": "BTC"})
        gold_close = gold_df[["close"]].rename(columns={"close": "GOLD"})

        # Remove timezone if present (normalize to tz-naive)
        if (
            isinstance(btc_close.index, pd.DatetimeIndex)
            and btc_close.index.tz is not None
        ):
            btc_close.index = btc_close.index.tz_localize(None)

        if (
            isinstance(gold_close.index, pd.DatetimeIndex)
            and gold_close.index.tz is not None
        ):
            gold_close.index = gold_close.index.tz_localize(None)

        # Now merge on timezone-naive indices
        merged = pd.merge(
            btc_close, gold_close, left_index=True, right_index=True, how="inner"
        )

        if len(merged) < 50:
            logger.warning(
                f"Insufficient overlapping data for correlation: {len(merged)} bars"
            )
            return None

        returns = merged.pct_change().dropna()
        correlation = returns["BTC"].corr(returns["GOLD"])

        btc_vol = returns["BTC"].std() * np.sqrt(24)  # Annualized for hourly data
        gold_vol = returns["GOLD"].std() * np.sqrt(24)

        logger.info(f"\n{'='*70}")
        logger.info("Asset Correlation Analysis")
        logger.info("=" * 70)
        logger.info(f"Overlapping bars: {len(merged)}")
        logger.info(f"BTC-GOLD correlation: {correlation:.3f}")
        logger.info(f"BTC annualized volatility: {btc_vol:.1%}")
        logger.info(f"GOLD annualized volatility: {gold_vol:.1%}")

        if correlation > 0.7:
            logger.info("  ⚠ High positive correlation - reduce concurrent positions")
        elif correlation < -0.5:
            logger.info("  ℹ Negative correlation - good diversification")
        else:
            logger.info("  ✓ Low correlation - good diversification")

        return {
            "correlation": correlation,
            "overlapping_bars": len(merged),
            "btc_volatility": btc_vol,
            "gold_volatility": gold_vol,
        }

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return None


def main():
    """Main training pipeline"""
    logger.info("=" * 70)
    logger.info("MULTI-SOURCE TRADING BOT TRAINING")
    logger.info("BTC: Binance | GOLD: MT5")
    logger.info("Strategies: Mean Reversion + Trend Following + EMA Crossover")
    logger.info("=" * 70)

    # Load config from config.json
    config_path = project_root / "config" / "config.json"
    if not config_path.exists():
        logger.error(f"config.json not found at {config_path}!")
        return

    with open(config_path) as f:
        config = json.load(f)

    # ✅ FIX: Override with credentials from .env
    if os.getenv("MT5_LOGIN"):
        config.setdefault("api", {}).setdefault("mt5", {})["login"] = os.getenv("MT5_LOGIN")
    if os.getenv("MT5_PASSWORD"):
        config.setdefault("api", {}).setdefault("mt5", {})["password"] = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        config.setdefault("api", {}).setdefault("mt5", {})["server"] = os.getenv("MT5_SERVER")

    # Initialize data manager
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Initializing Data Sources")
    logger.info("=" * 70)

    data_manager = DataManager(config)

    # Initialize Binance
    logger.info("\nInitializing Binance API...")
    binance_ok = data_manager.initialize_binance()
    if not binance_ok:
        logger.warning("⚠️  Failed to initialize Binance. Will rely on local files for Binance assets.")
    else:
        logger.info("✓ Binance API ready")

    # Initialize MT5
    logger.info("\nInitializing MT5...")
    mt5_ok = data_manager.initialize_mt5()
    if not mt5_ok:
        logger.warning("⚠️  Failed to initialize MT5. Will rely on local files for MT5 assets.")
    else:
        logger.info("✓ MT5 ready")

    # Fetch data for all assets
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Fetching Historical Data")
    logger.info("=" * 70)

    asset_data = {}

    # Asset configs to process
    assets_to_train = ["BTC", "GOLD", "USTEC", "EURJPY", "EURUSD"]
    
    for asset_key in assets_to_train:
        if asset_key not in config["assets"]:
            logger.warning(f"Asset {asset_key} not in config, skipping.")
            continue
            
        asset_cfg = config["assets"][asset_key]
        exchange = asset_cfg.get("exchange", "binance")
        
        # Try local raw data first if fetch fails or for speed
        raw_file = project_root / "data" / "raw" / f"{asset_cfg.get('symbol', asset_key)}_1h.csv"
        if asset_key == "BTC": raw_file = project_root / "data" / "raw" / "BTCUSDT_1h.csv"
        elif asset_key == "GOLD": raw_file = project_root / "data" / "raw" / "XAUUSDm_1h.csv"
        else: raw_file = project_root / "data" / "raw" / f"{asset_key}m_1h.csv"
        
        df = pd.DataFrame()
        if raw_file.exists():
            logger.info(f"Loading {asset_key} from local file: {raw_file}")
            try:
                # Try reading with first column as index and automatic date parsing
                df = pd.read_csv(raw_file)
                
                # Robust date column identification
                date_cols = [c for x in ['timestamp', 'date', 'time', 'datetime', 'open_time'] for c in df.columns if x in c.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                    df.set_index(date_col, inplace=True)
                else:
                    # Fallback to index_col=0 if no date-like column found
                    df = pd.read_csv(raw_file, index_col=0)
                    df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
                
                # Drop rows with invalid index
                if df.index.isnull().any():
                    invalid_count = df.index.isnull().sum()
                    logger.warning(f"⚠️  Removing {invalid_count} rows with invalid timestamps for {asset_key}")
                    df = df[df.index.notnull()]
                
                # Ensure index is sorted and name is clean
                df.index.name = 'timestamp'
                df.sort_index(inplace=True)

                # Check if local data is sufficient
                if len(df) < asset_cfg["min_bars_training"]:
                    logger.warning(f"⚠️  Local data for {asset_key} is insufficient ({len(df)} < {asset_cfg['min_bars_training']}).")
                    logger.info(f"🔄 Attempting to fetch fresh data from API...")
                    df = pd.DataFrame() # Reset to trigger fetch logic below
            except Exception as e:
                logger.error(f"❌ Error loading local file {raw_file}: {e}")
                df = pd.DataFrame()
        
        if df.empty:
            if exchange == "binance" and binance_ok:
                fetch_cfg = {
                    "symbol": asset_cfg["symbol"],
                    "interval": asset_cfg.get("interval", "1h"),
                    "lookback_days": asset_cfg["lookback_days"],
                    "min_bars_required": asset_cfg["min_bars_training"],
                }
                df = fetch_binance_asset_data(data_manager, fetch_cfg)
            elif exchange == "mt5" and mt5_ok:
                fetch_cfg = {
                    "symbol": asset_cfg["symbol"],
                    "timeframe": asset_cfg.get("timeframe", "H1"),
                    "lookback_days": asset_cfg["lookback_days"],
                    "min_bars_required": asset_cfg["min_bars_training"],
                }
                df = fetch_mt5_asset_data(data_manager, fetch_cfg) 
        
        if not df.empty and validate_data_quality(df, asset_key, asset_cfg["min_bars_training"]):
            asset_data[asset_key] = df
            logger.info(f"✅ {asset_key} data ready for training")
        else:
            logger.error(f"❌ {asset_key} data failed validation or could not be fetched")

    if not asset_data:
        logger.error("❌ No valid data available for training!")
        return

    # Correlation analysis
    if "BTC" in asset_data and "GOLD" in asset_data:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Cross-Asset Analysis")
        logger.info("=" * 70)
        correlation_stats = calculate_data_correlation(
            asset_data["BTC"], asset_data["GOLD"]
        )
    else:
        correlation_stats = None

    # Split train/test
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Splitting Train/Test Sets")
    logger.info("=" * 70)

    train_test_data = {}
    for asset_key, df in asset_data.items():
        logger.info(f"\n{asset_key}:")
        train_df, test_df = data_manager.split_train_test(df, train_pct=0.8)

        # Save both train and test for backtest
        train_file = project_root / "data" / f"train_data_{asset_key.lower()}.csv"
        test_file = project_root / "data" / f"test_data_{asset_key.lower()}.csv"

        train_df.to_csv(train_file)
        test_df.to_csv(test_file)

        train_test_data[asset_key] = {"train": train_df, "test": test_df}

        logger.info(f"✓ Train data saved: {train_file}")
        logger.info(f"✓ Test data saved: {test_file}")

    # Train models using config.json
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Training Models (ALL 3 STRATEGIES)")
    logger.info("=" * 70)

    training_results = {}
    for asset_key, data in train_test_data.items():
        results = train_asset_strategies(asset_key, data["train"], config)
        training_results[asset_key] = results

    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - FINAL REPORT")
    logger.info("=" * 70)

    logger.info("\n📊 Model Performance Summary:")
    logger.info("─" * 70)

    for asset_key, results in training_results.items():
        logger.info(f"\n{asset_key}:")

        mr = results["mean_reversion"]
        tf = results["trend_following"]
        ema = results["ema_strategy"]

        if mr.get("success"):
            logger.info(
                f"  ✅ Mean Reversion:   {mr.get('cv_mean_accuracy', 0):.2%} ± {mr.get('cv_std_accuracy', 0):.2%}"
            )
        else:
            logger.info(f"  ❌ Mean Reversion:   FAILED")

        if tf.get("success"):
            logger.info(
                f"  ✅ Trend Following:  {tf.get('cv_mean_accuracy', 0):.2%} ± {tf.get('cv_std_accuracy', 0):.2%}"
            )
        else:
            logger.info(f"  ❌ Trend Following:  FAILED")

        if ema.get("success"):
            logger.info(
                f"  ✅ EMA Strategy:     {ema.get('cv_mean_accuracy', 0):.2%} ± {ema.get('cv_std_accuracy', 0):.2%}"
            )
        else:
            logger.info(f"  ❌ EMA Strategy:     FAILED")

    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_sources": {"BTC": "Binance", "GOLD": "MT5"},
        "strategies": ["mean_reversion", "trend_following", "ema_strategy"],
        "training_results": training_results,
        "correlation": correlation_stats,
        "data_stats": {
            asset_key: {
                "total_bars": len(data["train"]) + len(data["test"]),
                "train_bars": len(data["train"]),
                "test_bars": len(data["test"]),
                "date_range": f"{data['train'].index[0]} to {data['test'].index[-1]}",
            }
            for asset_key, data in train_test_data.items()
        },
    }

    metadata_file = project_root / "models" / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"\n✓ Metadata saved: {metadata_file}")

    logger.info("\n📁 Trained Models:")
    logger.info("─" * 70)
    for asset_key in training_results.keys():
        logger.info(f"  • models/mean_reversion_{asset_key.lower()}.pkl")
        logger.info(f"  • models/trend_following_{asset_key.lower()}.pkl")
        logger.info(f"  • models/ema_strategy_{asset_key.lower()}.pkl")

    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS:")
    logger.info("=" * 70)
    logger.info("  1. Review performance metrics above")
    logger.info("  2. Check model files in models/ directory")
    logger.info("  3. Run backtest: python backtest.py --asset BTC --preset balanced")
    logger.info("  4. If no trades: python backtest.py --asset BTC --preset aggressive")
    logger.info("  5. Paper trade: python main.py --mode paper")
    logger.info("\n  Note: You now have 3 strategies generating signals!")
    logger.info("        Mean Reversion + Trend Following + EMA Crossover")

    # Cleanup
    data_manager.shutdown()

    logger.info("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    (project_root / "models").mkdir(exist_ok=True)
    (project_root / "data").mkdir(exist_ok=True)
    (project_root / "logs").mkdir(exist_ok=True)

    try:
        from binance.client import Client
    except ImportError:
        logger.error("python-binance not installed!")
        logger.info("Install with: pip install python-binance")
        sys.exit(1)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.warning("MetaTrader5 not installed (Gold trading disabled)")
        logger.info("Install with: pip install MetaTrader5")

    main()