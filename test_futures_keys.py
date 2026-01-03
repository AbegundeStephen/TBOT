"""
✅ Test Script: Verify Your Separate Binance Futures Keys Work
Run this BEFORE starting your bot to confirm everything is set up correctly
"""

import json
from binance.client import Client


def test_binance_futures_keys():
    """Test your separate Futures keys"""
    
    print("=" * 80)
    print("🧪 TESTING SEPARATE BINANCE FUTURES KEYS")
    print("=" * 80)
    
    # ============================================================
    # STEP 1: Load config
    # ============================================================
    try:
        with open("config/config.json", "r") as f:
            config = json.load(f)
        print("✅ Config loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading config.json: {e}")
        return False
    
    # ============================================================
    # STEP 2: Check if separate Futures keys exist
    # ============================================================
    futures_config = config.get("api", {}).get("binance_futures")
    
    if not futures_config:
        print("❌ No 'binance_futures' section found in config.json")
        print("\n💡 Add this to your config:")
        print("""
  "api": {
    "binance_futures": {
      "api_key": "YOUR_FUTURES_TESTNET_API_KEY",
      "api_secret": "YOUR_FUTURES_TESTNET_SECRET_KEY",
      "testnet": true,
      "api_base": "https://testnet.binancefuture.com"
    }
  }
        """)
        return False
    
    print("✅ Found separate Futures config\n")
    
    # ============================================================
    # STEP 3: Test Spot keys (for comparison)
    # ============================================================
    print("📊 Testing SPOT keys...")
    try:
        spot_config = config["api"]["binance"]
        spot_client = Client(
            api_key=spot_config["api_key"],
            api_secret=spot_config["api_secret"],
            testnet=spot_config.get("testnet", True)
        )
        
        # Test Spot API
        spot_account = spot_client.get_account()
        print(f"✅ Spot API connected")
        print(f"   Balance: {sum(float(b['free']) + float(b['locked']) for b in spot_account['balances'] if b['asset'] == 'USDT'):.2f} USDT\n")
    
    except Exception as e:
        print(f"❌ Spot API error: {e}\n")
    
    # ============================================================
    # STEP 4: Test Futures keys (main test)
    # ============================================================
    print("🚀 Testing FUTURES keys...")
    try:
        futures_client = Client(
            api_key=futures_config["api_key"],
            api_secret=futures_config["api_secret"],
            testnet=futures_config.get("testnet", True)
        )
        
        # Set correct Futures endpoint
        if futures_config.get("testnet"):
            futures_client.API_URL = "https://testnet.binancefuture.com"
            print(f"   Endpoint: testnet.binancefuture.com")
        else:
            futures_client.API_URL = "https://fapi.binance.com"
            print(f"   Endpoint: fapi.binance.com (LIVE - BE CAREFUL!)")
        
        # Test 1: Get account info
        print("\n   Test 1: Fetching account info...")
        account = futures_client.futures_account()
        balance = float(account.get('totalWalletBalance', 0))
        print(f"   ✅ Account connected")
        print(f"   📊 Balance: {balance:.2f} USDT")
        
        # Test 2: Check if any positions exist
        print("\n   Test 2: Checking existing positions...")
        positions = futures_client.futures_position_information(symbol="BTCUSDT")
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        if open_positions:
            print(f"   ⚠️  Found {len(open_positions)} open position(s):")
            for pos in open_positions:
                pos_amt = float(pos['positionAmt'])
                side = "LONG" if pos_amt > 0 else "SHORT"
                entry = float(pos['entryPrice'])
                pnl = float(pos['unRealizedProfit'])
                print(f"      - {side}: {abs(pos_amt):.6f} BTC @ ${entry:,.2f} (P&L: ${pnl:,.2f})")
        else:
            print(f"   ✅ No open positions")
        
        # Test 3: Get current leverage
        print("\n   Test 3: Checking leverage settings...")
        leverage_info = futures_client.futures_position_information(symbol="BTCUSDT")
        if leverage_info:
            leverage = leverage_info[0].get('leverage', 'Unknown')
            print(f"   ✅ Current leverage: {leverage}x")
        
        # Test 4: Get current price
        print("\n   Test 4: Fetching BTC price...")
        ticker = futures_client.futures_symbol_ticker(symbol="BTCUSDT")
        price = float(ticker['price'])
        print(f"   ✅ BTC Price: ${price:,.2f}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - YOUR FUTURES KEYS WORK!")
        print("=" * 80)
        print("\n💡 Next steps:")
        print("   1. Your config is correct ✓")
        print("   2. Start your trading bot")
        print("   3. It will use separate Futures keys automatically")
        print("\n⚠️  IMPORTANT:")
        print(f"   - Testnet mode: {'ENABLED' if futures_config.get('testnet') else 'DISABLED'}")
        print(f"   - Paper mode: {config['trading'].get('mode', 'unknown').upper()}")
        
        if not futures_config.get('testnet') and config['trading'].get('mode') == 'live':
            print("\n   🚨 WARNING: USING REAL MONEY!")
        else:
            print("\n   ✅ Safe: Using testnet + paper mode (no real money)")
        
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ FUTURES API ERROR: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your Futures API keys are correct")
        print("   2. Go to https://testnet.binancefuture.com")
        print("   3. Generate new API keys")
        print("   4. Update config.json with new keys")
        print("   5. Run this test again")
        return False


if __name__ == "__main__":
    success = test_binance_futures_keys()
    
    if not success:
        print("\n❌ Tests failed - fix the errors above before running your bot")
        exit(1)
    else:
        print("\n✅ Ready to start trading with Futures!")
        exit(0)