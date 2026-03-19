import requests
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("ALERT_MANAGER")

def send_alert(message: str):
    """
    Emergency alert sender using raw requests.
    Designed to be fast and fail-safe (short timeout, no blocking).
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    admin_ids = os.getenv("TELEGRAM_ADMIN_IDS")

    if not token or not admin_ids:
        logger.warning("[ALERT] Telegram credentials missing in environment")
        return

    # Send to first admin ID
    target_id = admin_ids.split(",")[0].strip()

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": target_id,
            "text": f"🚨 *CRITICAL ALERT*\n\n{message}",
            "parse_mode": "Markdown"
        }
        # Low timeout to ensure it doesn't hang the bot
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        logger.debug(f"[ALERT] Emergency alert failed: {e}")
