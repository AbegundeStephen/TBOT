"""
OutcomeTracker — the real shared pipeline behind Tier 6.2.

One class, one name, serving X.2's softening events, near-miss tracking,
and RLHF's human labels — instead of each one quietly inventing its own
logger. Injected into every class that needs to remember "I did X, check
later if X was right" so they all write to this same notebook.
"""


class OutcomeTracker:
    def __init__(self):
        self._pending = []
        self._records = []

    def tag(self, asset, direction, price, timestamp, event_type="softening"):
        self._pending.append({
            "asset": asset,
            "direction": direction,
            "price": price,
            "timestamp": timestamp,
            "event_type": event_type,
            "resolved": False,
        })

    def check_due(self, price_lookup_fn, bars_ahead=6):
        for entry in self._pending:
            if entry["resolved"]:
                continue
            later_price = price_lookup_fn(entry["asset"], entry["timestamp"], bars_ahead)
            if later_price is None:
                continue
            entry["moved_as_expected"] = (
                (later_price > entry["price"]) if entry["direction"] == "bullish"
                else (later_price < entry["price"])
            )
            entry["resolved"] = True

    # B8-5 (Desire, 22 Sep, option A): trade records and your Telegram answers
    # are written to logs/trade_labels.jsonl as well as kept in memory, so a
    # restart no longer wipes them and the replayer can join each answer to
    # its trade by episode id ("python tools/replayer.py --by label").
    _LABELS_PATH = "logs/trade_labels.jsonl"

    def _append(self, row):
        try:
            import json as _json, os as _os
            from datetime import datetime as _dt, timezone as _tz
            row = dict(row, ts=_dt.now(_tz.utc).isoformat(timespec="seconds"))
            _os.makedirs(_os.path.dirname(self._LABELS_PATH) or ".", exist_ok=True)
            with open(self._LABELS_PATH, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(row, default=str) + "\n")
            return True
        except Exception:
            return False

    def record(self, position, vtm, reason, human_label=None):
        rec = {
            "trade_id": getattr(position, "db_trade_id", None),
            "episode_id": getattr(position, "episode_id", None),
            "asset": getattr(vtm, "asset", None) or getattr(position, "asset", None),
            "exit_reason": reason,
            "human_label": human_label,
        }
        self._records.append(rec)
        self._append(dict(rec, type="record"))

    def attach_human_label(self, trade_id, label):
        episode_id = None
        for rec in self._records:
            if rec["trade_id"] == trade_id:
                rec["human_label"] = label
                episode_id = rec.get("episode_id")
        # Saved even when the in-memory record is gone after a restart -- the
        # replayer links it to its trade through the matching "record" row.
        return self._append({"type": "label", "trade_id": trade_id,
                             "episode_id": episode_id, "label": label})
