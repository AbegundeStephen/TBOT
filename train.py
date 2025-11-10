#**train.py**

#!/usr/bin/env python3
"""
Training Script - Train both ML strategies
"""

import json
import logging
from pathlib import Path
import sys

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    logger.info("="*60)
    logger.info("Starting Training Pipeline")
    logger.info("="*60)
    
    # Load configuration
    config_path = Path('config/config.json')
    if not config_path.exists():
        logger.error("config.json not found. Copy from config.template.json")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Initialize Binance for BTC data
    if not data_manager.initialize_binance():
        logger.error("Failed to initialize Binance")
        return
    
    # Fetch historical data
    logger.info("Fetching BTC historical data...")
    df_btc = data_manager.fetch_binance_data(
        symbol='BTCUSDT',
        interval='5m',
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    # Clean data
    df_btc = data_manager.clean_data(df_btc)
    
    # Split train/test
    train_df, test_df = data_manager.split_train_test(df_btc, train_pct=0.8)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Train Mean Reversion Strategy
    logger.info("\n" + "="*60)
    logger.info("Training Mean Reversion Strategy")
    logger.info("="*60)
    
    mr_config = {
        **config['strategies']['mean_reversion'],
        **config['ml'],
        'model_params': config['ml']['model_params']
    }
    
    mean_reversion = MeanReversionStrategy(mr_config)
    mr_metrics = mean_reversion.train_model(
        train_df,
        'models/mean_reversion_btc.pkl'
    )
    
    logger.info(f"Mean Reversion Training Metrics: {mr_metrics}")
    
    # Train Trend Following Strategy
    logger.info("\n" + "="*60)
    logger.info("Training Trend Following Strategy")
    logger.info("="*60)
    
    tf_config = {
        **config['strategies']['trend_following'],
        **config['ml'],
        'model_params': config['ml']['model_params']
    }
    
    trend_following = TrendFollowingStrategy(tf_config)
    tf_metrics = trend_following.train_model(
        train_df,
        'models/trend_following_btc.pkl'
    )
    
    logger.info(f"Trend Following Training Metrics: {tf_metrics}")
    
    # Save test data for backtesting
    test_df.to_csv('data/test_data_btc.csv')
    logger.info("Test data saved to data/test_data_btc.csv")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info("Next steps:")
    logger.info("1. Run backtesting: python backtest.py")
    logger.info("2. Deploy live: python main.py")


if __name__ == "__main__":
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    main()
