"""
Wave L4 calibration review — run after 30+ days.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.database.database_manager import TradingDatabaseManager as DatabaseManager


def review_l4_calibration(db: DatabaseManager, min_samples: int = 20):
    result = (
        db.supabase.table("trades")
        .select("metadata, pnl")
        .eq("status", "closed")
        .execute()
    )
    rows = result.data or []
    buckets = {
        "L7_bull_confirm": [], "L7_bear_confirm": [], "L7_no_fire": [],
        "L9_livermore_derived": [], "L9_adx_fallback": [],
        "L10_late_stage_fired": [], "L10_late_stage_not_fired": [],
    }
    for row in rows:
        meta = row.get("metadata")
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except Exception: continue
        if not meta: continue
        pnl = row.get("pnl")
        if pnl is None: continue
        telem = meta.get("livermore_telemetry") or {}
        l7 = telem.get("L7_trend_judge")
        if l7 and l7.get("fired"):
            buckets[f"L7_{l7['direction']}"].append(pnl)
        elif l7 is not None:
            buckets["L7_no_fire"].append(pnl)
        l9 = telem.get("L9_lifecycle_source")
        if l9:
            key = f"L9_{l9['source']}"
            if key in buckets: buckets[key].append(pnl)
        for key in ("L10_tf_tag", "L10_ema_tag"):
            l10 = telem.get(key)
            if l10 and l10.get("late_stage_penalty_fired"):
                buckets["L10_late_stage_fired"].append(pnl)
            elif l10 is not None:
                buckets["L10_late_stage_not_fired"].append(pnl)
    print(f"\n{'='*70}\nWAVE L4 CALIBRATION REVIEW\n{'='*70}")
    for name, pnls in buckets.items():
        if len(pnls) < min_samples:
            print(f"\n{name}: {len(pnls)} samples — not enough yet")
            continue
        wins = [p for p in pnls if p > 0]
        print(f"\n{name}: {len(pnls)} trades\n"
              f"  Win rate: {len(wins)/len(pnls)*100:.1f}%\n"
              f"  Total P&L: ${sum(pnls):,.2f}\n"
              f"  Avg/trade: ${sum(pnls)/len(pnls):.2f}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    db = DatabaseManager(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_KEY"],
    )
    review_l4_calibration(db)