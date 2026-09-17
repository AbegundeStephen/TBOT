"""
B4 W7: one-time backfill of the 17 Sep GBPAUD trade the dead VTM management
thread never registered (open and close both fell inside the 16 Sep 20:01 -
17 Sep ~16:28 hang window; broker reconciliation lives in that thread, so
neither the open nor the close was ever recorded).

Broker record (ticket 129765239): short, open 10:03:30 @ 1.88115, close
10:10:43 @ 1.88203, -$0.63, 0.01 lots. Times are local (UTC+2), per this
whole batch's documented log-time convention -- converted to UTC below.

Idempotent: checks the target day's episode file for this ticket first, so
running it twice does not write a duplicate row.

Run once, on whichever machine holds the real gap (the VPS):
    python scripts/reconstruct_w7_gbpaud_trade.py
"""

import glob
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.episode_ledger import write_episode
from src.utils.instance_paths import suffixed_path

MT5_TICKET = 129765239


def already_written() -> bool:
    for f in glob.glob(f"{suffixed_path('logs/episodes')}/*.jsonl"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("mt5_ticket") == MT5_TICKET:
                        print(f"already present in {f} -- nothing to do")
                        return True
        except Exception:
            continue
    return False


def main():
    if already_written():
        return

    local_tz = timezone(timedelta(hours=2))
    entry_local = datetime(2026, 9, 17, 10, 3, 30, tzinfo=local_tz)
    exit_local = datetime(2026, 9, 17, 10, 10, 43, tzinfo=local_tz)

    record = {
        "episode_id": "GBPAUD_short_129765239_reconstructed",
        "source": "live",
        "asset": "GBPAUD",
        "side": "short",
        "entry_price": 1.88115,
        "exit_price": 1.88203,
        "close_price": 1.88203,
        "quantity": 0.01,
        "entry_time": entry_local.astimezone(timezone.utc).isoformat(),
        "exit_time": exit_local.astimezone(timezone.utc).isoformat(),
        "close_time": exit_local.astimezone(timezone.utc).isoformat(),
        "exit_event_type": "closed_on_exchange",
        "exit_reason": "closed_on_exchange_unreconciled",
        "close_reason": "closed_on_exchange_unreconciled",
        "pnl": -0.63,
        "mt5_ticket": MT5_TICKET,
        "net_pnl_r": None,
        "gross_r": None,
        "pnl_pct": None,
        "reconstruction_note": (
            "B4 W7: manually reconstructed from the MT5 broker record after "
            "the VTM management thread was found hung silently 16 Sep 20:01 "
            "- 17 Sep ~16:28. This position opened and closed entirely "
            "inside that window; broker reconciliation lives in the dead "
            "thread, so the bot never registered the open or the close. "
            "Fields not derivable from the broker record alone (stop-loss, "
            "R-multiple, ATR, composite_state, sl_path) are intentionally "
            "left null rather than invented."
        ),
    }
    write_episode(record)
    print("wrote reconstruction row for ticket", MT5_TICKET)


if __name__ == "__main__":
    main()
