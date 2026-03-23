import requests
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables from .env
load_dotenv()

# Get keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    print("❌ Error: BINANCE_API_KEY or BINANCE_API_SECRET not found in .env file.")
    exit(1)

# Step 1: Get your current public IP
try:
    public_ip = requests.get("https://api.ipify.org", timeout=5).text
    print(f"🌐 Your current public IP: {public_ip}")
except Exception as e:
    print("❌ Could not fetch public IP:", e)
    public_ip = None

# Step 2: Initialize Binance client
client = Client(API_KEY, API_SECRET, testnet=False)  # set testnet=True if using testnet

# Step 3: Test Futures account access
try:
    balances = client.futures_account_balance()
    print("✅ API key is valid and Futures access works!")
    print("Futures Balances:")
    # Filter for non-zero balances to avoid clutter
    for asset in balances:
        if float(asset['balance']) > 0:
            print(f"{asset['asset']}: {asset['balance']}")
except BinanceAPIException as e:
    print("❌ Binance API error:")
    print(f"Code: {e.code}, Message: {e.message}")
    if e.code == -2015:
        print("→ Invalid API key, IP not allowed, or Futures permission missing.")
        if public_ip:
            print(
                f"→ Make sure this IP ({public_ip}) is whitelisted in your Binance API settings."
            )
