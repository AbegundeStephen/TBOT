import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Replace with your Binance API key and secret
API_KEY = "7U9E8zvO1zdPFnHaQum0nVceOp3C0H5O1Yw0MlblSL9g9mzuxIPK8OXtksl2GUXW"
API_SECRET = "FwAb7uNbQ3NWCyF87cIriEI3NvE1oXgFdUjS3oBu9KZ7usALquxsJH5vlZ98g0Vq"

# Step 1: Get your current public IP
try:
    public_ip = requests.get("https://api.ipify.org").text
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
    for asset in balances:
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
