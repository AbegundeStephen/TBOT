#!/usr/bin/env python3
"""
Debug script to diagnose why main.py won't start
"""

import sys
import json
from pathlib import Path

print("=" * 70)
print("TRADING BOT STARTUP DIAGNOSTICS")
print("=" * 70)

# 1. Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  WARNING: Python 3.8+ recommended")

# 2. Check config file
print("\n2. Checking config file...")
config_path = Path("config/config.json")
if config_path.exists():
    print(f"   ✓ Found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("   ✓ Valid JSON")
        print(f"   - Trading mode: {config.get('trading', {}).get('mode', 'N/A')}")
        print(f"   - BTC enabled: {config.get('assets', {}).get('BTC', {}).get('enabled', False)}")
        print(f"   - GOLD enabled: {config.get('assets', {}).get('GOLD', {}).get('enabled', False)}")
    except json.JSONDecodeError as e:
        print(f"   ✗ INVALID JSON: {e}")
    except Exception as e:
        print(f"   ✗ ERROR reading config: {e}")
else:
    print(f"   ✗ NOT FOUND: {config_path}")
    print("   → Create config/config.json first")

# 3. Check directories
print("\n3. Checking directories...")
dirs = {
    "config": Path("config"),
    "logs": Path("logs"),
    "models": Path("models"),
    "src": Path("src"),
}

for name, path in dirs.items():
    if path.exists():
        print(f"   ✓ {name}/")
    else:
        print(f"   ✗ {name}/ (missing)")
        if name in ["logs"]:
            print(f"     → Will be created automatically")

# 4. Check for trained models
print("\n4. Checking for trained models...")
models_dir = Path("models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    if model_files:
        print(f"   ✓ Found {len(model_files)} model(s):")
        for model in model_files:
            print(f"     - {model.name}")
    else:
        print("   ✗ NO MODELS FOUND")
        print("     → Run: python train.py")
        print("     → Bot will EXIT without models")
else:
    print("   ✗ models/ directory missing")

# 5. Check required modules
print("\n5. Checking required modules...")
required_modules = [
    "schedule",
    "src.data.data_manager",
    "src.strategies.mean_reversion",
    "src.strategies.trend_following",
    "src.execution.signal_aggregator",
    "src.execution.binance_handler",
    "src.execution.mt5_handler",
    "src.portfolio.portfolio_manager",
    "src.utils.market_hours",
    "src.telegram_bot",
]

missing_modules = []
for module_name in required_modules:
    try:
        __import__(module_name)
        print(f"   ✓ {module_name}")
    except ImportError as e:
        print(f"   ✗ {module_name}: {e}")
        missing_modules.append(module_name)

if missing_modules:
    print(f"\n   ⚠️  Missing {len(missing_modules)} module(s)")
    print("   → Run: pip install -r requirements.txt")

# 6. Check credentials (if config exists)
if config_path.exists():
    print("\n6. Checking credentials...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Binance
        binance_key = config.get("exchanges", {}).get("binance", {}).get("api_key", "")
        if binance_key and binance_key != "your_binance_api_key":
            print("   ✓ Binance API key configured")
        else:
            print("   ⚠️  Binance API key not set")
        
        # MT5
        mt5_login = config.get("exchanges", {}).get("mt5", {}).get("login", "")
        if mt5_login:
            print("   ✓ MT5 login configured")
        else:
            print("   ⚠️  MT5 login not set")
        
        # Telegram
        telegram_token = config.get("telegram", {}).get("bot_token", "")
        if telegram_token and telegram_token != "your_telegram_bot_token":
            print("   ✓ Telegram bot token configured")
        else:
            print("   ⚠️  Telegram bot token not set")
            
    except Exception as e:
        print(f"   ✗ Error checking credentials: {e}")

# 7. Try to import main
print("\n7. Testing main.py import...")
try:
    import main
    print("   ✓ main.py imports successfully")
except Exception as e:
    print(f"   ✗ ERROR importing main.py:")
    print(f"     {type(e).__name__}: {e}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if not config_path.exists():
    print("⚠️  CRITICAL: config/config.json missing")
elif not Path("models").exists() or not list(Path("models").glob("*.pkl")):
    print("⚠️  CRITICAL: No trained models found (run train.py)")
elif missing_modules:
    print("⚠️  CRITICAL: Missing required modules")
else:
    print("✓ All checks passed - bot should start")
    print("\nIf bot still won't start, run with verbose output:")
    print("  python main.py 2>&1 | tee startup.log")

print("=" * 70)