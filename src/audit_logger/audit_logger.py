import json
import time
import os
from pathlib import Path

LOG_FILE = "logs/trade_audit.log"

def log_trade(data: dict):
    """
    Append a machine-readable trade event to the audit log.
    Data should include: event, asset, side, price, size, pnl, etc.
    """
    try:
        # Ensure logs directory exists
        Path(LOG_FILE).parent.mkdir(exist_ok=True)
        
        # Add standard timestamp if missing
        if "timestamp" not in data:
            data["timestamp"] = time.time()
            data["datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
    except Exception as e:
        # We don't want audit logging to crash the main bot
        print(f"Error writing to audit log: {e}")
