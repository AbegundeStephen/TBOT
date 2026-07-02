"""
MR Mode calibration review — run after 30+ days.
Breaks down performance by which MR mode fired the trade.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.database.database_manager import TradingDatabaseManager as DatabaseManager


def review_mr_modes(db: DatabaseManager, min_samples: int = 10):
    result = (
        db.supabase.table("trades")
        .select("metadata, pnl, strategy")
        .eq("status", "closed")
        .execute()
    )
    rows = result.data or []
    print(f"[DEBUG] Total closed trades fetched: {len(rows)}")
    mr_rows = [r for r in rows if r.get("strategy") == "REVERSION"]
    print(f"[DEBUG] REVERSION strategy rows: {len(mr_rows)}")
    buckets = {
        "mode1_pullback": [],
        "mode2_counter": [],
        "mode3_climax": [],
        "other": [],
    }
    for row in rows:
        meta = row.get("metadata")
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except Exception: continue
        if not meta: continue
        pnl = row.get("pnl")
        if pnl is None: continue
        ct = meta.get("confluence_telemetry") or {}
        lsm = ct.get("livermore_state_1h", "")
        entry_type = meta.get("entry_type", "")
        if row.get("strategy") != "REVERSION":
            continue
        if "NATURAL_RETRACEMENT" in lsm or "mode1" in entry_type.lower():
            buckets["mode1_pullback"].append(pnl)
        elif "SECONDARY" in lsm or "mode2" in entry_type.lower():
            buckets["mode2_counter"].append(pnl)
        elif "MAIN_" in lsm or "mode3" in entry_type.lower():
            buckets["mode3_climax"].append(pnl)
        else:
            buckets["other"].append(pnl)
    print(f"\n{'='*70}\nMR MODE PERFORMANCE REVIEW\n{'='*70}")
    for name, pnls in buckets.items():
        if len(pnls) < min_samples:
            print(f"\n{name}: {len(pnls)} trades — not enough yet")
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
    review_mr_modes(db)