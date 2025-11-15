"""
MT5 Connection Test Script
Run this to diagnose MT5 connection issues
"""

import MetaTrader5 as mt5
from datetime import datetime

print("=" * 60)
print("MT5 CONNECTION TEST")
print("=" * 60)

# Your credentials
LOGIN = 297047456
PASSWORD = "Stepheng01@"
SERVER = "Exness-MT5Trial9"
PATH = "C:\\Program Files\\MetaTrader 5"  # Without terminal64.exe

print(f"\nCredentials:")
print(f"  Login: {LOGIN}")
print(f"  Server: {SERVER}")
print(f"  Path: {PATH}")

print("\n" + "-" * 60)
print("TEST 1: Initialize MT5 (with path)")
print("-" * 60)

result = mt5.initialize(path=PATH)
if not result:
    error = mt5.last_error()
    print(f"❌ FAILED: {error}")
    print("\nTrying without path...")
    
    print("\n" + "-" * 60)
    print("TEST 2: Initialize MT5 (auto-detect)")
    print("-" * 60)
    
    result = mt5.initialize()
    if not result:
        error = mt5.last_error()
        print(f"❌ FAILED: {error}")
        print("\n⚠ CRITICAL ERROR: Cannot initialize MT5")
        print("\nCHECKLIST:")
        print("  [ ] Is MetaTrader 5 OPEN and RUNNING?")
        print("  [ ] Are you logged into MT5 manually?")
        print("  [ ] Is Python running as Administrator?")
        print("  [ ] Is antivirus blocking Python?")
        input("\nPress Enter to exit...")
        exit(1)

print("✓ Initialize successful!")

# Get terminal info
print("\n" + "-" * 60)
print("TEST 3: Get Terminal Info")
print("-" * 60)

terminal_info = mt5.terminal_info()
if terminal_info:
    print(f"✓ Terminal Info Retrieved:")
    print(f"  Company: {terminal_info.company}")
    print(f"  Name: {terminal_info.name}")
    print(f"  Path: {terminal_info.path}")
    print(f"  Connected: {terminal_info.connected}")
else:
    print("❌ Cannot get terminal info")

# Try login
print("\n" + "-" * 60)
print("TEST 4: Login to Account")
print("-" * 60)

authorized = mt5.login(login=LOGIN, password=PASSWORD, server=SERVER)

if not authorized:
    error = mt5.last_error()
    print(f"❌ Login FAILED: {error}")
    print("\nPOSSIBLE ISSUES:")
    print("  1. Wrong password (check for typos)")
    print("  2. Wrong server name")
    print("  3. Account not activated")
    print("  4. Demo account expired")
    print("\nTO VERIFY SERVER NAME:")
    print("  Open MT5 > Tools > Options > Server tab")
    mt5.shutdown()
    input("\nPress Enter to exit...")
    exit(1)

print("✓ Login successful!")

# Get account info
print("\n" + "-" * 60)
print("TEST 5: Get Account Info")
print("-" * 60)

account_info = mt5.account_info()
if account_info:
    print(f"✓ Account Info Retrieved:")
    print(f"  Account: {account_info.login}")
    print(f"  Name: {account_info.name}")
    print(f"  Server: {account_info.server}")
    print(f"  Balance: ${account_info.balance:.2f}")
    print(f"  Leverage: 1:{account_info.leverage}")
    print(f"  Currency: {account_info.currency}")
    print(f"  Trade Allowed: {account_info.trade_allowed}")
else:
    print("❌ Cannot get account info")

# Test symbol access
print("\n" + "-" * 60)
print("TEST 6: Test Symbol Access (XAUUSD)")
print("-" * 60)

symbol_info = mt5.symbol_info("XAUUSD")
if symbol_info:
    print(f"✓ XAUUSD accessible:")
    print(f"  Bid: {symbol_info.bid}")
    print(f"  Ask: {symbol_info.ask}")
    print(f"  Spread: {symbol_info.spread}")
else:
    print("❌ Cannot access XAUUSD")
    print("  (This might be normal if not available on your account)")

# Test data fetch
print("\n" + "-" * 60)
print("TEST 7: Fetch Historical Data")
print("-" * 60)

rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 10)
if rates is not None and len(rates) > 0:
    print(f"✓ Successfully fetched {len(rates)} bars")
    print(f"  Latest close: {rates[-1]['close']}")
else:
    print("❌ Cannot fetch historical data")

# Cleanup
mt5.shutdown()

print("\n" + "=" * 60)
print("✓✓✓ ALL TESTS COMPLETED ✓✓✓")
print("=" * 60)
print("\nMT5 connection is working! Your bot should work now.")
input("\nPress Enter to exit...")