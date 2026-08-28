"""
DATA-4 ITEM 4: path capture.

Desire's ruling, 28 Aug: build this only if the offline replayer fails
calibration on 15m bars. It did -- confirmed via extensive investigation
(tools/replayer.py's own commit history) that the real exit-management
system has multiple interacting mechanisms (soft risk-cut, intermediate
trail, R-lock, ATR-based breakeven, the runner trail itself, plus a
Livermore-state-gated trend-invalidation exit), evaluated at 1H-bar-close
granularity -- not reliably reconstructable after the fact from 15m/1H
CSVs, however faithfully each individual formula is replicated.

Rather than keep refining a post-hoc reconstruction, the bot now records
the real observed price directly, once per ~5-minute trading cycle, for
every currently open position -- live and shadow. No reconstruction
needed for any future replay: the path IS the record. Keyed by
episode_id, same join key as the episode/gate ledgers.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_path_point(episode_id, asset, price, source):
    """Append one price observation for one open episode. Never raises.

    source: "live" | "shadow" -- same tagging convention as the episode
    ledger, since both feed the eventual replayer/learner.
    """
    if not episode_id or price is None:
        return
    try:
        _dir = Path("logs/paths")
        _dir.mkdir(parents=True, exist_ok=True)
        _path = _dir / f"paths_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        _row = {
            "ts": datetime.now().isoformat(),
            "episode_id": episode_id,
            "asset": asset,
            "source": source,
            "price": price,
        }
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_row, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[PATH-LEDGER] write failed: {e}")
