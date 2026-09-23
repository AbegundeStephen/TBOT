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

from src.utils.instance_paths import suffixed_path as _suffixed_path

logger = logging.getLogger(__name__)


def write_episode(record: dict) -> None:
    """Append one closed episode to today's daily ledger file."""
    try:
        record.setdefault("schema_version", 2)   # DIARY-1: v1 = pre-15 Sep rows; v2 = management path + state_age + pair fields
        # B10 D1: live rows wrote box-local times with no zone; practice rows write
        # UTC with +00:00, so the two sorted two hours apart. Every time goes out
        # in UTC. An unmarked time means box-local on a live row and UTC on a
        # practice row (the practice engine stamps with utcnow).
        from datetime import timezone as _tz_d1
        _naive_is_local = record.get("source") != "shadow"
        for _k in ("entry_time", "close_time", "open_time", "exit_time"):
            _v = record.get(_k)
            try:
                if isinstance(_v, datetime):
                    _dt = _v
                elif isinstance(_v, str) and _v:
                    _dt = datetime.fromisoformat(_v.replace("Z", "+00:00"))
                else:
                    continue
                if _dt.tzinfo is None:
                    _dt = _dt.astimezone() if _naive_is_local else _dt.replace(tzinfo=_tz_d1.utc)
                record[_k] = _dt.astimezone(_tz_d1.utc).isoformat()
            except Exception:
                pass
        # HF-2 I3: logs/episodes/ was the known gap in B11's path-suffixing --
        # instance B's episodes would otherwise land in the SAME daily ledger
        # file the live instance reads, corrupting both instances' diaries.
        _dir = Path(_suffixed_path("logs/episodes"))
        _dir.mkdir(parents=True, exist_ok=True)
        # B9 S9: the market snapshot is 96% of every row and repeats across rows
        # from the same cycle. Store it once under its own hash, keep a reference
        # in the row. Nothing is lost -- the snapshot file holds the full copy.
        _day = datetime.now().strftime('%Y-%m-%d')
        _path = _dir / f"episodes_{_day}.jsonl"
        try:
            _cs = record.get("composite_state")
            if _cs:
                import hashlib as _hl
                _blob = json.dumps(_cs, default=str, sort_keys=True)
                _ref = _hl.sha1(_blob.encode("utf-8")).hexdigest()[:16]
                _snap = _dir / f"snapshots_{_day}.jsonl"
                _seen = globals().setdefault("_B9_SNAP_SEEN", set())
                if _ref not in _seen:
                    with open(_snap, "a", encoding="utf-8") as _sf:
                        _sf.write(json.dumps({"ref": _ref, "composite_state": _cs}, default=str) + "\n")
                    _seen.add(_ref)
                record.pop("composite_state", None)
                record["composite_state_ref"] = _ref
        except Exception as _snap_err:
            logger.warning(f"[EPISODE] snapshot split skipped ({_snap_err})")
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"[EPISODE] {record.get('episode_id')} closed and written")
    except Exception as e:
        logger.warning(f"[EPISODE] could not write episode record: {e}")
