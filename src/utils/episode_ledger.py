"""
DATA-1 ITEM 6: shared episode-ledger writer.

One JSONL row per closed episode (live or shadow), the join target for
episode_id across the funnel, trade events, move ledger and shadow/live
records. Needs to be callable from two different classes in two different
modules (PortfolioManager.close_position and ShadowTradingEngine._archive) --
extracted as a standalone function rather than duplicated or reached via a
cross-class back-reference, same reasoning as write_json_atomic's extraction
into run_status.py.

Deliberately JSONL and local -- same pattern as the shadow and move ledgers,
which have both proven durable. No database: the dashboard's connection
already drops often enough.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_episode(record: dict) -> None:
    """Append one closed episode to today's daily ledger file."""
    try:
        _dir = Path("logs/episodes")
        _dir.mkdir(parents=True, exist_ok=True)
        _path = _dir / f"episodes_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"[EPISODE] {record.get('episode_id')} closed and written")
    except Exception as e:
        logger.warning(f"[EPISODE] could not write episode record: {e}")
