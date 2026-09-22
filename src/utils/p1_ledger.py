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


def write_break_check(asset, kind, direction, h, close, band, dist_atr, tier,
                      result, gear, tf, candle_ts=None):
    """B8-7 (Desire, 22 Sep, option B): one row per stage-1 break check --
    breaks AND near-misses, with the distance each time -- so the 0.10 break
    band can later be judged on complete data. Same daily file and folder as
    write_refusal above (its old caller, the break-size filter, was removed
    by CU-1 on 14 Sep). Silent on failure: telemetry must never break a cycle."""
    try:
        os.makedirs(_DIR, exist_ok=True)
        _now = datetime.now(timezone.utc)
        _row = {
            "ts": _now.isoformat(),
            "event": "break_check",
            "asset": asset,
            "kind": kind,
            "dir": int(direction),
            "h": float(h),
            "close": float(close),
            "band": float(band),
            "dist_atr4": round(float(dist_atr), 4) if dist_atr is not None else None,
            "tier": tier,
            "result": result,
            "gear": gear,
            "tf": tf,
            "candle_ts": str(candle_ts) if candle_ts is not None else None,
        }
        _path = os.path.join(_DIR, f"p1_{_now.strftime('%Y-%m-%d')}.jsonl")
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_row) + "\n")
    except Exception:
        pass
