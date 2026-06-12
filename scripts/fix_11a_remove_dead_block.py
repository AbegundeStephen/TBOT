"""
Fix 11a — Remove dead duplicate block in portfolio_manager.py.

The block from line 2837 through 3464 (inclusive) is a dead zone:
nine methods are defined twice; Python last-def-wins so the earlier
definitions (2837-3464) are never executed. Delete them in one pass.

Run from TBOT root:
    python scripts/fix_11a_remove_dead_block.py
"""

import sys
from pathlib import Path

TARGET = Path("src/portfolio/portfolio_manager.py")
DEAD_START = 2837   # first line of dead @handle_errors decorator
DEAD_END   = 3464   # last line of dead get_portfolio_status closing brace

def main():
    lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)
    print(f"File has {total} lines.")

    if total < DEAD_END:
        print(f"ERROR: file only has {total} lines — wrong file or already patched?")
        sys.exit(1)

    # Verify the boundary looks right
    first_dead = lines[DEAD_START - 1]        # 0-indexed
    first_live = lines[DEAD_END]              # line after dead block
    print(f"Line {DEAD_START} (first dead): {first_dead.rstrip()}")
    print(f"Line {DEAD_END+1} (first live): {first_live.rstrip()}")

    confirm = input("Delete lines 2837-3464? [yes/N]: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    # Backup
    backup = TARGET.with_suffix(".py.bak_fix11a")
    backup.write_bytes(TARGET.read_bytes())
    print(f"Backup written to {backup}")

    # Remove dead lines (2837-3464 are indices 2836-3463 in 0-indexed list)
    kept = lines[:DEAD_START - 1] + lines[DEAD_END:]
    TARGET.write_text("".join(kept), encoding="utf-8")
    print(f"Done. Removed {DEAD_END - DEAD_START + 1} lines.")
    print(f"New line count: {len(kept)}")

    # Quick sanity: count definitions of each duplicate method
    result = "".join(kept)
    for name in [
        "def close_position",
        "def reconcile_positions",
        "def update_positions",
        "def get_open_positions_count",
        "def get_position",
        "def has_position",
        "def reset_daily_pnl",
        "def start_trading_session",
        "def get_portfolio_status",
    ]:
        count = result.count(name)
        status = "✅" if count == 1 else f"❌ ({count} found)"
        print(f"  {status}  {name}")

if __name__ == "__main__":
    main()
