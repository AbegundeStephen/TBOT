#!/usr/bin/env python3
"""
Quick test script to verify EMA signal generation
Run this BEFORE full training to check if signals are being generated
"""

import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.strategies.ema_strategy import EMAStrategy

def test_ema_signals():
    """Test EMA strategy signal generation"""
    
    # Load config
    with open('config/config.json') as f:
        config = json.load(f)
    
    # Test both assets
    for asset in ['BTC', 'GOLD']:
        print(f"\n{'='*70}")
        print(f"Testing EMA Strategy - {asset}")
        print('='*70)
        
        # Load training data
        data_file = f'data/train_data_{asset.lower()}.csv'
        if not Path(data_file).exists():
            print(f"❌ {data_file} not found. Run train.py first to generate data.")
            continue
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} bars from {data_file}")
        
        # Get config for this asset
        ema_config = config['strategy_configs']['exponential_moving_averages'][asset]
        
        print(f"\n📋 Config:")
        print(f"  EMA Fast/Slow: {ema_config['ema_fast']}/{ema_config['ema_slow']}")
        print(f"  Min Distance: {ema_config['min_distance_pct']}%")
        print(f"  Min Return: {ema_config['min_return_threshold']:.4%}")
        print(f"  Min Conditions: {ema_config['min_conditions']}")
        
        # Initialize strategy
        strategy = EMAStrategy(ema_config)
        
        # Generate features
        print(f"\n🔧 Generating features...")
        df_features = strategy.generate_features(df)
        print(f"✓ Generated {len(df_features.columns)} features")
        print(f"✓ Features have {len(df_features)} rows after generation")
        
        # Check for NaN in critical columns
        critical_cols = ['ema_fast', 'ema_slow', 'ema_diff_pct', 'rsi', 'macd_hist']
        for col in critical_cols:
            if col in df_features.columns:
                nan_count = df_features[col].isna().sum()
                print(f"  - {col}: {nan_count} NaN values")
        
        # Generate labels
        print(f"\n🏷️  Generating labels...")
        labels = strategy.generate_labels(df_features)
        
        # Analyze results
        if len(labels) == 0:
            print(f"\n❌ ERROR: No labels generated (empty series)")
            print(f"   This means all rows were filtered out during label generation")
            continue
            
        buy_signals = (labels == 1).sum()
        sell_signals = (labels == -1).sum()
        hold_signals = (labels == 0).sum()
        total = len(labels)
        
        print(f"\n📊 Signal Summary:")
        print(f"  BUY:  {buy_signals:>5} ({buy_signals/total*100:>5.2f}%)")
        print(f"  SELL: {sell_signals:>5} ({sell_signals/total*100:>5.2f}%)")
        print(f"  HOLD: {hold_signals:>5} ({hold_signals/total*100:>5.2f}%)")
        
        # Verdict
        print(f"\n{'='*70}")
        if buy_signals >= 100 and sell_signals >= 100:
            print(f"✅ {asset}: EXCELLENT - Sufficient signals for training")
        elif buy_signals >= 50 and sell_signals >= 50:
            print(f"✓ {asset}: GOOD - Should train successfully")
        elif buy_signals >= 20 and sell_signals >= 20:
            print(f"⚠ {asset}: MARGINAL - May train but consider lowering thresholds")
        else:
            print(f"❌ {asset}: INSUFFICIENT - Need to lower min_return_threshold or min_conditions")
            print(f"\n💡 Suggestions:")
            print(f"  1. Lower min_return_threshold to 0.00005 (0.005%)")
            print(f"  2. Set min_conditions to 1 (already at minimum)")
            print(f"  3. Check if data quality is good (enough bars, no gaps)")
        print('='*70)
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print("If signals look good, proceed with: python train.py")
    print('='*70)

if __name__ == "__main__":
    test_ema_signals()