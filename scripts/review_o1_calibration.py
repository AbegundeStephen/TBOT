"""
O-Track calibration review — run manually against Supabase once 30+ trades
have accumulated under each O1 signal. Breaks down P&L by which orphan
signal fired, so the starting multipliers can be replaced with real numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.database.database_manager import TradingDatabaseManager as DatabaseManager


def review_o1_calibration(db: DatabaseManager, min_samples: int = 20):
    result = (
        db.supabase.table("trades")
        .select("metadata, pnl")
        .eq("status", "closed")
        .execute()
    )
    rows = result.data or []

    buckets = {
        "rejection_fired":      [],
        "rejection_not_fired":  [],
        "parabolic_locked":     [],
        "parabolic_not_locked": [],
        "absorption_fired":     [],
        "absorption_not_fired": [],
    }

    for row in rows:
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                continue
        if not meta:
            continue
        pnl = row.get("pnl")
        if pnl is None:
            continue

        o1 = meta.get("o1_vtm_telemetry") or {}
        bucket_key = lambda fired, label: f"{label}_{'fired' if fired else 'not_fired'}"
        buckets[bucket_key(o1.get("rejection_fired"), "rejection")].append(pnl)
        buckets[bucket_key(o1.get("parabolic_locked"), "parabolic")].append(pnl)
        buckets[bucket_key(o1.get("absorption_fired"), "absorption")].append(pnl)

    print(f"\n{'='*70}\nO1 CALIBRATION REVIEW\n{'='*70}")
    for name, pnls in buckets.items():
        if len(pnls) < min_samples:
            print(f"\n{name}: {len(pnls)} samples (need {min_samples}+) — not enough data yet")
            continue
        wins = [p for p in pnls if p > 0]
        print(
            f"\n{name}: {len(pnls)} trades\n"
            f"  Win rate: {len(wins)/len(pnls)*100:.1f}%\n"
            f"  Total P&L: ${sum(pnls):,.2f}\n"
            f"  Avg P&L/trade: ${sum(pnls)/len(pnls):.2f}"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    db = DatabaseManager(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_KEY"],
    )
    review_o1_calibration(db)
