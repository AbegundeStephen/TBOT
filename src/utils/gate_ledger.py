"""DATA-3 ITEM 4: per-gate decision ledger.

The funnel stores one final stage per evaluation and mislabels it (64% of
"blocked_low_score" on 27 Aug was actually the cost gate). This records every
gate that evaluated a signal, in order, with the numbers it decided on.

Keyed by episode_id so a blocked signal's gate history joins to its shadow
outcome -- which is how "was this gate right to block?" becomes answerable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_gate_decision(episode_id, asset, gate, verdict, numbers=None):
    """Append one gate decision. Never raises."""
    try:
        _dir = Path("logs/gates")
        _dir.mkdir(parents=True, exist_ok=True)
        _path = _dir / f"gates_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        _row = {
            "ts": datetime.now().isoformat(),
            "episode_id": episode_id,
            "asset": asset,
            "gate": gate,
            "verdict": verdict,
            "numbers": numbers or {},
        }
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_row, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[GATE-LEDGER] write failed: {e}")
