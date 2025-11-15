#!/usr/bin/env python3
"""
Quick test script to verify Binance data fetching
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_manager import DataManager

def test_fetch():
    """Test Binance data fetching"""
    
    # Load config
    with open('config/config.json') as f:
        config = json.load(f)
    
    # Initialize
    dm = DataManager(config)
    
    print("Initializing Binance API...")
    if not dm.initialize_binance():
        print("❌ Failed to initialize Binance")
        return
    print("✓ Binance ready\n")
    
    # Test parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    
    # Test different time ranges
    tests = [
        ("7 days", 7),
        ("30 days", 30),
        ("90 days", 90),
        ("180 days", 180),
        ("365 days", 365),
    ]
    
    for test_name, days in tests:
        print(f"\n{'='*60}")
        print(f"TEST: Fetching {test_name} of data")
        print('='*60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Start: {start_date.strftime('%Y-%m-%d')}")
        print(f"End: {end_date.strftime('%Y-%m-%d')}")
        print(f"Expected bars: ~{days * 24}")
        
        df = dm.fetch_binance_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not df.empty:
            actual_days = (df.index[-1] - df.index[0]).days
            coverage = (actual_days / days) * 100
            
            print(f"\n✓ Result:")
            print(f"  Bars fetched: {len(df)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Days covered: {actual_days} / {days} ({coverage:.1f}%)")
            print(f"  First close: ${df['close'].iloc[0]:,.2f}")
            print(f"  Last close: ${df['close'].iloc[-1]:,.2f}")
            
            if coverage < 95:
                print(f"  ⚠ Warning: Only {coverage:.1f}% coverage")
            else:
                print(f"  ✓ Good coverage: {coverage:.1f}%")
        else:
            print("❌ No data received")
        
        # Don't spam the API
        if test_name != tests[-1][0]:
            print("\nWaiting 2 seconds...")
            import time
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print('='*60)

if __name__ == "__main__":
    test_fetch()