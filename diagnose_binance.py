#!/usr/bin/env python3
"""
Diagnose Binance API and data availability
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from binance.client import Client
import pandas as pd

def diagnose():
    """Diagnose Binance setup"""
    
    print("="*70)
    print("BINANCE API DIAGNOSTICS")
    print("="*70)
    
    # Load config
    with open('config/config.json') as f:
        config = json.load(f)
    
    api_config = config['api']['binance']
    api_key = api_config.get('api_key', '')
    api_secret = api_config.get('api_secret', '')
    
    print(f"\n1. API Configuration:")
    print(f"   API Key: {api_key[:20]}... (truncated)")
    print(f"   Testnet: {api_config.get('testnet', True)}")
    print(f"   API Base: {api_config.get('api_base', 'default')}")
    
    # Initialize client
    print(f"\n2. Initializing Client...")
    try:
        if api_config.get('testnet', True):
            client = Client(api_key, api_secret, testnet=True, tld='com')
            client.API_URL = api_config.get('api_base', 'https://testnet.binance.vision/api')
            print(f"   ✓ Testnet client created")
            print(f"   URL: {client.API_URL}")
        else:
            client = Client(api_key, api_secret)
            print(f"   ✓ Production client created")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test connection
    print(f"\n3. Testing Connection...")
    try:
        server_time = client.get_server_time()
        server_dt = pd.to_datetime(server_time['serverTime'], unit='ms')
        print(f"   ✓ Connected successfully")
        print(f"   Server time: {server_dt}")
        print(f"   Your time: {datetime.now()}")
        
        time_diff = abs((datetime.now() - server_dt.replace(tzinfo=None)).total_seconds())
        if time_diff > 60:
            print(f"   ⚠ WARNING: Time difference of {time_diff:.0f} seconds!")
            print(f"   This could cause issues with timestamp-based queries")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # Check symbol info
    print(f"\n4. Checking BTCUSDT Symbol...")
    try:
        symbol_info = client.get_symbol_info('BTCUSDT')
        if symbol_info:
            print(f"   ✓ Symbol exists")
            print(f"   Status: {symbol_info['status']}")
            print(f"   Base Asset: {symbol_info['baseAsset']}")
            print(f"   Quote Asset: {symbol_info['quoteAsset']}")
        else:
            print(f"   ❌ Symbol not found")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Get latest price
    print(f"\n5. Getting Current Price...")
    try:
        ticker = client.get_symbol_ticker(symbol='BTCUSDT')
        print(f"   ✓ Current price: ${float(ticker['price']):,.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test data fetch with different methods
    print(f"\n6. Testing Historical Data Availability...")
    
    # Method 1: Get latest 1000 bars
    print(f"\n   Method 1: Latest 1000 bars (no date filter)")
    try:
        klines = client.get_klines(symbol='BTCUSDT', interval='1h', limit=1000)
        if klines:
            first_dt = pd.to_datetime(klines[0][0], unit='ms')
            last_dt = pd.to_datetime(klines[-1][0], unit='ms')
            print(f"   ✓ Got {len(klines)} bars")
            print(f"   Range: {first_dt} to {last_dt}")
            print(f"   Days covered: {(last_dt - first_dt).days}")
        else:
            print(f"   ❌ No data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 2: Request specific date range (1 year ago)
    print(f"\n   Method 2: Specific date range (last 365 days)")
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)
        
        print(f"   Requesting: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        klines = client.get_klines(
            symbol='BTCUSDT',
            interval='1h',
            startTime=int(start_dt.timestamp() * 1000),
            endTime=int(end_dt.timestamp() * 1000),
            limit=1000
        )
        
        if klines:
            first_dt = pd.to_datetime(klines[0][0], unit='ms')
            last_dt = pd.to_datetime(klines[-1][0], unit='ms')
            print(f"   ✓ Got {len(klines)} bars")
            print(f"   Range: {first_dt} to {last_dt}")
            
            if first_dt > start_dt:
                missing_days = (first_dt - start_dt).days
                print(f"   ⚠ Missing {missing_days} days at start")
                print(f"   Earliest available: {first_dt}")
        else:
            print(f"   ❌ No data returned")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Try even older data
    print(f"\n   Method 3: Testing data availability (2 years ago)")
    try:
        end_dt = datetime.now() - timedelta(days=365)
        start_dt = end_dt - timedelta(days=365)
        
        print(f"   Requesting: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        klines = client.get_klines(
            symbol='BTCUSDT',
            interval='1h',
            startTime=int(start_dt.timestamp() * 1000),
            endTime=int(end_dt.timestamp() * 1000),
            limit=100
        )
        
        if klines and len(klines) > 0:
            first_dt = pd.to_datetime(klines[0][0], unit='ms')
            print(f"   ✓ Historical data available from: {first_dt}")
        else:
            print(f"   ⚠ No data available for this period")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    print(f"\n📋 SUMMARY:")
    print(f"   If you're seeing limited data (only ~155 bars),")
    print(f"   possible causes:")
    print(f"   1. Testnet has limited historical data")
    print(f"   2. System clock issues causing date mismatches")
    print(f"   3. API endpoint restrictions")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Try switching to production API (set testnet: false)")
    print(f"   2. Check your system time is correct")
    print(f"   3. Use shorter lookback periods (90 days instead of 365)")

if __name__ == "__main__":
    diagnose()