"""
🔍 Diagnostic Script: Figure out why sync is failing
Run this to see which API endpoints you're hitting
"""

import json
from binance.client import Client


def diagnose_sync_issue():
    """Diagnose which APIs are being used and why sync fails"""
    
    print("=" * 80)
    print("🔍 SYNC ISSUE DIAGNOSTIC")
    print("=" * 80)
    
    # Load config
    with open("config/config.json", "r") as f:
        config = json.load(f)
    
    # Get API configs
    spot_config = config["api"]["binance"]
    futures_config = config["api"].get("binance_futures")
    btc_config = config["assets"]["BTC"]
    
    print("\n📋 Configuration:")
    print(f"  Trading mode:     {config['trading']['mode']}")
    print(f"  Futures enabled:  {btc_config.get('enable_futures', False)}")
    print(f"  Separate keys:    {'Yes' if futures_config else 'No'}")
    print(f"  Testnet:          {spot_config.get('testnet', True)}")
    
    # ============================================================
    # TEST 1: Spot API
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 1: SPOT API")
    print("=" * 80)
    
    try:
        spot_client = Client(
            spot_config["api_key"],
            spot_config["api_secret"],
            testnet=spot_config.get("testnet", True)
        )
        spot_client.API_URL = "https://testnet.binance.vision/api"
        
        print("✅ Spot client created")
        print(f"   Endpoint: {spot_client.API_URL}")
        
        # Test Spot balance
        account = spot_client.get_account()
        btc_balance = 0
        usdt_balance = 0
        
        for bal in account["balances"]:
            if bal["asset"] == "BTC":
                btc_balance = float(bal["free"]) + float(bal["locked"])
            elif bal["asset"] == "USDT":
                usdt_balance = float(bal["free"]) + float(bal["locked"])
        
        print(f"✅ Spot balance fetched")
        print(f"   BTC:  {btc_balance:.6f}")
        print(f"   USDT: {usdt_balance:.2f}")
        
    except Exception as e:
        print(f"❌ Spot API error: {e}")
    
    # ============================================================
    # TEST 2: Futures API (if configured)
    # ============================================================
    if futures_config:
        print("\n" + "=" * 80)
        print("TEST 2: FUTURES API")
        print("=" * 80)
        
        try:
            futures_client = Client(
                futures_config["api_key"],
                futures_config["api_secret"],
                testnet=futures_config.get("testnet", True)
            )
            futures_client.API_URL = "https://testnet.binancefuture.com"
            
            print("✅ Futures client created")
            print(f"   Endpoint: {futures_client.API_URL}")
            
            # Test Futures balance
            account = futures_client.futures_account()
            
            for asset in account.get("assets", []):
                if asset["asset"] == "USDT":
                    balance = float(asset.get("availableBalance", 0))
                    print(f"✅ Futures balance fetched")
                    print(f"   USDT: {balance:.2f}")
                    break
            
            # Test Futures positions
            positions = futures_client.futures_position_information(symbol="BTCUSDT")
            active_positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
            
            print(f"✅ Futures positions checked")
            print(f"   Open: {len(active_positions)}")
            
            if active_positions:
                for pos in active_positions:
                    pos_amt = float(pos["positionAmt"])
                    side = "LONG" if pos_amt > 0 else "SHORT"
                    print(f"   - {side}: {abs(pos_amt):.6f} BTC @ ${float(pos['entryPrice']):,.2f}")
        
        except Exception as e:
            print(f"❌ Futures API error: {e}")
    
    # ============================================================
    # RECOMMENDATIONS
    # ============================================================
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    mode = config["trading"]["mode"]
    futures_enabled = btc_config.get("enable_futures", False)
    has_separate_keys = futures_config is not None
    
    if mode == "paper":
        print("\n✅ PAPER MODE:")
        print("   - Sync should be skipped (simulated positions only)")
        print("   - No API calls needed")
        print("   - If sync is running, check is_paper_mode flag")
    
    elif futures_enabled and has_separate_keys:
        print("\n🚀 FUTURES MODE (Separate Keys):")
        print("   - BinanceHandler should use Futures client")
        print("   - Sync should use futures_position_information()")
        print("   - Portfolio Manager should use Spot client for balance")
        print("\n   Code structure:")
        print("   1. data_manager.binance_client → Spot (for OHLCV)")
        print("   2. data_manager.futures_client → Futures (for trading)")
        print("   3. binance_handler.futures_handler → Futures positions")
        print("   4. portfolio_manager.binance_client → Spot (for balance)")
    
    elif futures_enabled:
        print("\n🚀 FUTURES MODE (Same Keys):")
        print("   - Using Spot keys for both Spot and Futures")
        print("   - Sync should detect Futures via futures_handler")
        print("   - Balance check uses Spot API")
    
    else:
        print("\n📊 SPOT MODE:")
        print("   - Sync uses get_account() for BTC balance")
        print("   - Only LONG positions supported")
        print("   - Everything uses Spot API")
    
    print("\n" + "=" * 80)
    print("🔧 TO FIX YOUR ERROR:")
    print("=" * 80)
    print("\n1. Replace sync_positions_with_binance() with the fixed version")
    print("2. Make sure binance_handler.client uses CORRECT keys:")
    print("   - If using Futures: Pass futures_client")
    print("   - If using Spot: Pass spot_client")
    print("\n3. In main.py, initialization should be:")
    print("""
    # Get correct client for trading
    if config['assets']['BTC'].get('enable_futures'):
        trading_client = data_manager.get_futures_client()
    else:
        trading_client = data_manager.binance_client
    
    binance_handler = BinanceExecutionHandler(
        config=config,
        client=trading_client,  # ← Important!
        portfolio_manager=portfolio_manager,
        data_manager=data_manager
    )
    """)
    
    print("\n✅ Run this diagnosis again after fixes to verify")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_sync_issue()