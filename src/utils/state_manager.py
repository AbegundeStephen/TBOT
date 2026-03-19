import json
import os
import logging
from pathlib import Path

logger = logging.getLogger("STATE_MANAGER")

STATE_FILE = "data/system_state.json"

def save_system_state(data: dict):
    """
    Persist critical system metrics to JSON.
    """
    try:
        # Ensure directory exists
        Path(STATE_FILE).parent.mkdir(exist_ok=True)
        
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"[STATE] System state saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"[STATE] Failed to save state: {e}")

def load_system_state() -> dict:
    """
    Restore critical system metrics from JSON.
    """
    if not os.path.exists(STATE_FILE):
        logger.info("[STATE] No system state file found.")
        return {}
        
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        logger.info(f"[STATE] System state restored from {STATE_FILE}")
        return state
    except Exception as e:
        logger.error(f"[STATE] Failed to load state: {e}")
        return {}
