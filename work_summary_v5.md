I have fixed the `IndentationError` in `src/execution/veteran_trade_manager.py` at line 289.

The issue was caused by an incorrect indentation and logic flow within the `__init__` method, specifically where general parameters were being set and where `trade_type`-specific constraints were applied. The previous `replace` operations had inadvertently misaligned these blocks.

**Changes Made:**

*   **`src/execution/veteran_trade_manager.py`:**
    *   The `__init__` method was refactored to correctly separate the initialization of general parameters (`enable_early_profit_lock`, `early_lock_threshold_pct`, `runner_trail_pct`, `early_lock_atr_multiplier`, `runner_trail_atr_multiplier`, `current_early_lock_threshold_pct`, `current_runner_trail_pct`) from the `trade_type`-specific adjustments (for `SCALP` vs. `TREND`).
    *   The indentation of the `else: # TREND` block and subsequent code was corrected to align properly with its `if` statement.

This fix ensures that the `VeteranTradeManager`'s `__init__` method now adheres to Python's indentation rules and correctly initializes all its parameters, resolving the `IndentationError`. The bot should now be able to start without this traceback.
