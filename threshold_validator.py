#!/usr/bin/env python3
"""
Quick Threshold Validator
Run this BEFORE full training to verify signal distribution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler
import json
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_test(asset_name, strategy, df, expected_signal_pct=(15, 40)):
    """
    Quick test of strategy signal distribution
    
    Args:
        asset_name: 'BTC' or 'GOLD'
        strategy: Strategy instance
        df: Historical data
        expected_signal_pct: (min, max) acceptable % of BUY+SELL signals
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing {strategy.name} on {asset_name}")
    logger.info('='*70)
    
    
    # Generate features and labels
    df_features = strategy.generate_features(df)
    labels = strategy.generate_labels(df_features)
    
    # Count distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    sell = dist.get(-1, 0)
    hold = dist.get(0, 0)
    buy = dist.get(1, 0)
    total = len(labels)
    
    buy_pct = (buy / total) * 100
    sell_pct = (sell / total) * 100
    signal_pct = buy_pct + sell_pct
    
    print(f"\nLabel Distribution:")
    print(f"  SELL:  {sell:>6} ({sell_pct:>5.2f}%)")
    print(f"  HOLD:  {hold:>6} ({(hold/total)*100:>5.2f}%)")
    print(f"  BUY:   {buy:>6} ({buy_pct:>5.2f}%)")
    print(f"  TOTAL: {total:>6}")
    print(f"\nTradeable Signals: {signal_pct:.1f}%")
    
    # Validation
    min_expected, max_expected = expected_signal_pct
    
    if signal_pct < min_expected:
        print(f"\n❌ FAIL: Too few signals ({signal_pct:.1f}% < {min_expected}%)")
        print(f"   Likely cause: Thresholds too strict")
        return False
    elif signal_pct > max_expected:
        print(f"\n⚠️  WARNING: Many signals ({signal_pct:.1f}% > {max_expected}%)")
        print(f"   This might be okay, but verify quality")
        return True
    else:
        print(f"\n✅ PASS: Good signal distribution ({signal_pct:.1f}%)")
        return True
    
    # Check balance
    if buy > 0 and sell > 0:
        ratio = max(buy, sell) / min(buy, sell)
        if ratio > 3:
            print(f"\n⚠️  WARNING: Imbalanced signals (ratio: {ratio:.1f}:1)")
            print(f"   Consider adjusting thresholds")

def initialize_exchanges(self):
        """Initialize exchange connections based on enabled assets"""
        logger.info("Initializing exchange connections...")

        # Check if BTC is enabled (requires Binance)
        if self.config["assets"]["BTC"].get("enabled", False):
            if self.data_manager.initialize_binance():
                self.binance_handler = BinanceExecutionHandler(
                    self.config, 
                    self.data_manager.binance_client,
                    self.portfolio_manager
                )
                logger.info("Binance handler initialized")
            else:
                logger.error("Failed to initialize Binance - BTC trading will be disabled")

        # Check if GOLD is enabled (requires MT5)
        if self.config["assets"]["GOLD"].get("enabled", False):
            if self.data_manager.initialize_mt5():
                self.mt5_handler = MT5ExecutionHandler(
                    self.config,
                    self.portfolio_manager
                )
                logger.info("MT5 handler initialized")
            else:
                logger.error("Failed to initialize MT5 - GOLD trading will be disabled")


def main():
    """Test both assets with new thresholds"""
    
    print("="*70)
    print("THRESHOLD VALIDATION - NEW STRATEGY CONFIGS")
    print("="*70)
    
    # Initialize data manager
 # Load config
config_path = Path('config/config.json')
with open(config_path) as f:
    config = json.load(f)

    data_manager = DataManager(config=config)
    
    results = {}
    
    # ========================================================================
    # TEST BTC
    # ========================================================================
    if data_manager.initialize_binance():
        print("\n" + "="*70)
        print("TESTING GOLD")
        print("="*70)
    print("\n" + "="*70)
    print("TESTING BTC")
    print("="*70)
    
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months
    
    btc_df = data_manager.fetch_binance_data(
            symbol='BTCUSDT',
            interval='1h',
           start_date=start_date.strftime("%Y-%m-%d"),
           end_date=end_date.strftime("%Y-%m-%d"),
    )
    
    if not btc_df.empty:
        print(f"Fetched {len(btc_df)} bars for BTC")
        
        # Test Mean Reversion
        mr_config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 65,
            'rsi_oversold': 35,
            'bb_lower_threshold': 0.30,
            'bb_upper_threshold': 0.70,
            'min_return_threshold': 0.003,
            'min_conditions': 1,
            'stoch_k': 14,
            'stoch_d': 3,
            'reversion_window': 5
        }
        mr_strategy = MeanReversionStrategy(mr_config)
        btc_mr_pass = quick_test('BTC', mr_strategy, btc_df, expected_signal_pct=(15, 40))
        
        # Test Trend Following
        tf_config = {
            'fast_ma': 20,
            'slow_ma': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 15,
            'require_adx': False,
            'min_return_threshold': 0.003,
            'min_conditions': 2
        }
        tf_strategy = TrendFollowingStrategy(tf_config)
        btc_tf_pass = quick_test('BTC', tf_strategy, btc_df, expected_signal_pct=(20, 45))
        
        results['BTC'] = {
            'mean_reversion': btc_mr_pass,
            'trend_following': btc_tf_pass
        }
    else:
        print("❌ Could not fetch BTC data")
        results['BTC'] = {'mean_reversion': False, 'trend_following': False}
    
    # ========================================================================
    # TEST GOLD
    # ========================================================================
    if data_manager.initialize_mt5():
        print("\n" + "="*70)
        print("TESTING GOLD")
        print("="*70)
        
        gold_df = data_manager.fetch_mt5_data(
            symbol='XAUUSDm',  # Adjust to your symbol
            timeframe='H1',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not gold_df.empty:
            print(f"Fetched {len(gold_df)} bars for GOLD")
            
            # Test Mean Reversion
            mr_config = {
                'bb_period': 20,
                'bb_std': 2.0,
                'rsi_period': 14,
                'rsi_overbought': 60,
                'rsi_oversold': 40,
                'bb_lower_threshold': 0.30,
                'bb_upper_threshold': 0.70,
                'min_return_threshold': 0.002,
                'min_conditions': 1,
                'stoch_k': 14,
                'stoch_d': 3,
                'reversion_window': 5
            }
            mr_strategy = MeanReversionStrategy(mr_config)
            gold_mr_pass = quick_test('GOLD', mr_strategy, gold_df, expected_signal_pct=(15, 40))
            
            # Test Trend Following
            tf_config = {
                'fast_ma': 20,
                'slow_ma': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'adx_period': 14,
                'adx_threshold': 12,
                'require_adx': False,
                'min_return_threshold': 0.002,
                'min_conditions': 2
            }
            tf_strategy = TrendFollowingStrategy(tf_config)
            gold_tf_pass = quick_test('GOLD', tf_strategy, gold_df, expected_signal_pct=(15, 40))
            
            results['GOLD'] = {
                'mean_reversion': gold_mr_pass,
                'trend_following': gold_tf_pass
            }
        else:
            print("❌ Could not fetch GOLD data")
            results['GOLD'] = {'mean_reversion': False, 'trend_following': False}
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    all_passed = True
    for asset, strategies in results.items():
        print(f"\n{asset}:")
        for strategy, passed in strategies.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {strategy:20s}: {status}")
            if not passed:
                all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - READY FOR FULL TRAINING")
        print("\nNext steps:")
        print("  1. Run: python train_multi_source.py")
        print("  2. Check training logs for accuracy metrics")
        print("  3. Run backtest to validate performance")
    else:
        print("❌ SOME TESTS FAILED - ADJUST THRESHOLDS")
        print("\nTroubleshooting:")
        print("  - If too few signals: Lower thresholds further")
        print("  - If too many signals: Increase min_return_threshold")
        print("  - Check logs above for specific issues")
    
    print("="*70)
    
    data_manager.shutdown()


if __name__ == "__main__":
    main()