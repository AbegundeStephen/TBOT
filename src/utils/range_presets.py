"""
Shared range-preset vocabulary for the dashboard's Data section and Backtest
launch panel. Pure data + one pure function, no I/O -- importable by
backtest.py, scripts/refresh_data.py, and src/dashboard/server.py without
import-order or circular-import concerns.

The six keys match exactly what was asked for: 3/6 months, 2/3/4/5 years.
No 1y/all key is invented -- "no preset selected" continues to mean
"full history" (today's existing behavior), not a seventh preset.
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

RANGE_PRESETS = {
    "3m": {"label": "3 months", "months": 3},
    "6m": {"label": "6 months", "months": 6},
    "2y": {"label": "2 years", "months": 24},
    "3y": {"label": "3 years", "months": 36},
    "4y": {"label": "4 years", "months": 48},
    "5y": {"label": "5 years", "months": 60},
}


def resolve_cutoff(anchor_dt: datetime, preset_key: str) -> datetime:
    """Return the earliest datetime to keep, `preset_key`'s window back from anchor_dt."""
    months = RANGE_PRESETS[preset_key]["months"]
    return anchor_dt - relativedelta(months=months)


def resolve_lookback_days(preset_key: str) -> int:
    """Convert a preset to an integer day count, for callers that need lookback_days
    rather than a cutoff datetime (e.g. refresh_data.py's fetch functions)."""
    anchor = datetime(2000, 1, 1)
    cutoff = resolve_cutoff(anchor, preset_key)
    return (anchor - cutoff).days
