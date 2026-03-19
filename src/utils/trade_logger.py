import logging
import time
from typing import Dict, Any
import json
from src.audit_logger.audit_logger import log_trade

logger = logging.getLogger("TRADE_EVENT")


def log_trade_event(event_type: str, data: Dict[str, Any]):
    """
    Standardized trade logger for structured analysis.
    Supported event_types: ENTRY, EXIT, SL_HIT, TP_HIT, PYRAMID, REJECTION
    """
    event_payload = {
        "event": event_type.upper(),
        "symbol": data.get("symbol"),
        "asset": data.get("asset"),
        "side": data.get("side", "").upper(),
        "type": data.get("trade_type", data.get("type", "UNKNOWN")),
        "price": data.get("price"),
        "size": data.get("size", data.get("quantity")),
        "reason": data.get("reason"),
        "pnl": data.get("pnl"),
        "pnl_pct": data.get("pnl_pct"),
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "position_id": data.get("position_id"),
    }

    # 1. Log to main bot log (structured text)
    logger.info(f"[TRADE_EVENT] {json.dumps(event_payload)}")

    # 2. Log to machine-readable audit log (JSON lines)
    try:
        log_trade(event_payload)
    except Exception as e:
        logger.error(f"Failed to write to audit log: {e}")
