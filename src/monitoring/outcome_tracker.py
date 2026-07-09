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

    def record(self, position, vtm, reason, human_label=None):
        self._records.append({
            "trade_id": getattr(position, "db_trade_id", None),
            "asset": getattr(vtm, "asset", None),
            "exit_reason": reason,
            "human_label": human_label,
        })

    def attach_human_label(self, trade_id, label):
        for rec in self._records:
            if rec["trade_id"] == trade_id:
                rec["human_label"] = label
                return True
        return False
