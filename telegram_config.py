#!/usr/bin/env python3
"""
Telegram Bot Configuration
Store your Telegram bot token and admin user IDs here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_CONFIG = {
    # Enable/disable Telegram bot
    "enabled": True,
    
    # Your Telegram bot token from @BotFather
    # Get it by talking to @BotFather on Telegram
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    
    # List of authorized Telegram user IDs (admins)
    # Get your ID by messaging @userinfobot on Telegram
    # Example: [123456789, 987654321]
    "admin_ids": [
        int(id_str) for id_str in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",")
        if id_str.strip()
    ],
    
    # Notification settings
    "notifications": {
        # Send notification when trade is opened
        "notify_trade_open": True,
        
        # Send notification when trade is closed
        "notify_trade_close": True,
        
        # Send notification on errors
        "notify_errors": True,
        
        # Send daily performance summary
        "notify_daily_summary": True,
        
        # Time to send daily summary (24-hour format)
        "daily_summary_time": "23:00",
        
        # Send market status changes (e.g., GOLD market closed)
        "notify_market_status": False,  # Can be spammy
    },
    
    # Rate limiting (prevent notification spam)
    "rate_limit": {
        # Maximum error notifications per hour
        "max_errors_per_hour": 10,
        
        # Cooldown between similar notifications (seconds)
        "notification_cooldown": 60,
    }
}


# Validation
def validate_config():
    """Validate Telegram configuration"""
    if not TELEGRAM_CONFIG["enabled"]:
        return True, "Telegram bot is disabled"
    
    if not TELEGRAM_CONFIG["bot_token"]:
        return False, "TELEGRAM_BOT_TOKEN not set in .env file"
    
    if not TELEGRAM_CONFIG["admin_ids"]:
        return False, "TELEGRAM_ADMIN_IDS not set in .env file"
    
    return True, "Configuration valid"


if __name__ == "__main__":
    # Test configuration
    is_valid, message = validate_config()
    
    print("=" * 70)
    print("TELEGRAM CONFIGURATION TEST")
    print("=" * 70)
    print(f"Enabled: {TELEGRAM_CONFIG['enabled']}")
    print(f"Bot Token: {'*' * 20 if TELEGRAM_CONFIG['bot_token'] else 'NOT SET'}")
    print(f"Admin IDs: {TELEGRAM_CONFIG['admin_ids'] if TELEGRAM_CONFIG['admin_ids'] else 'NOT SET'}")
    print(f"\nStatus: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"Message: {message}")
    print("=" * 70)
    
    if not is_valid:
        print("\n⚠️ Please configure your .env file with:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("   TELEGRAM_ADMIN_IDS=123456789,987654321")
        print("\n📖 See .env.example for template")