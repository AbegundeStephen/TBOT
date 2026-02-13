I have implemented enhancements to provide better visibility into how VTM's dynamic profit locking and adaptive trailing stops are behaving, both via the Telegram `/vtm` command and through the bot's logs.

**Key Changes Implemented:**

1.  **Veteran Trade Manager (`src/execution/veteran_trade_manager.py`) Enhancements:**
    *   **Initialization (`__init__`):**
        *   Added `self.current_early_lock_threshold_pct` and `self.current_runner_trail_pct` as instance variables to store the currently active dynamic percentages.
        *   Enhanced the initialization log output to clearly indicate whether the "Early Lock" and "Runner Trail" are configured as "Dynamic" (with their ATR multipliers) or "Fixed" (with their initial percentage values).
    *   **`check_exit` Method:**
        *   Modified the log message when the early profit lock is triggered (`[VTM] 🛡️ Break-even`) to now include the `actual_early_lock_threshold` that caused the lock, providing context on the dynamic threshold used.
        *   The `actual_early_lock_threshold` is now stored in `self.current_early_lock_threshold_pct` when the lock is triggered.
    *   **`update_with_current_price` Method:**
        *   Added explicit log messages (`[VTM] 🏃 Trailing SL updated`) when the adaptive trailing stop moves, showing the new and old stop loss prices and the `current_runner_trail_pct` (the dynamic percentage used for trailing).
        *   The `dynamic_runner_trail_pct` (or fixed `runner_trail_pct`) is now stored in `self.current_runner_trail_pct` when the stop loss is updated.
    *   **`get_current_levels` Method:**
        *   Modified to return additional dynamic VTM parameters: `early_lock_atr_multiplier`, `runner_trail_atr_multiplier`, `current_early_lock_threshold_pct`, and `current_runner_trail_pct`. These values are essential for providing comprehensive status via the Telegram command.

2.  **Telegram Bot (`src/telegram/__init__.py`) Enhancements:**
    *   **`cmd_VTM_status` Method:**
        *   Updated the message formatting for the `/vtm` command to display the newly exposed dynamic VTM parameters from `vtm_status`. This now includes:
            *   Whether "Early Lock" is Dynamic (with its ATR multiplier) and its current effective threshold.
            *   Whether "Runner Trail" is Dynamic (with its ATR multiplier) and its current effective trailing percentage.
            *   Falls back to displaying fixed percentages if dynamic multipliers are not configured.

These modifications ensure that users can clearly see how the VTM is behaving regarding dynamic profit locking and adaptive trailing stops, both in the detailed log output when positions are managed and in a concise summary via the Telegram `/vtm` command.

The implementation is complete.