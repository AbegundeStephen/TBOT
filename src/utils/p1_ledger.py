"""MEASURE M2: structured break-magnitude refusal ledger.

The [P1-MAGNITUDE] log line has carried every number needed since 11-Aug, but
only as text -- answering "was this refused level later born as a setup?"
required manually cross-referencing two greps by timestamp. One row per
refusal, keyed by asset+ref, makes it a query.

Mirrors path_ledger.py: append-only, daily file, never raises into the caller.
"""

import json
import os
from datetime import datetime, timezone

_DIR = os.path.join("logs", "p1_refusals")


def write_refusal(asset, kind, direction, ref, price, dist, band,
                  tier, mult, scale, classification, atr=None):
    """Append one refusal row. Silent on failure -- telemetry must never
    break a trading cycle."""
    try:
        os.makedirs(_DIR, exist_ok=True)
        _now = datetime.now(timezone.utc)
        _row = {
            "ts": _now.isoformat(),
            "asset": asset,
            "kind": kind,
            "dir": int(direction),
            "ref": float(ref),
            "price": float(price),
            "dist": float(dist) if dist is not None else None,
            "band": float(band),
            "ratio": (float(dist) / float(band)) if (dist and band) else None,
            "tier": tier,
            "mult": float(mult),
            "scale": float(scale),
            "atr": float(atr) if atr else None,
            "classification": classification,
        }
        _path = os.path.join(_DIR, f"p1_{_now.strftime('%Y-%m-%d')}.jsonl")
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_row) + "\n")
    except Exception:
        pass
