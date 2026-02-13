I have implemented significant changes to the Telegram bot integration to address the issue of the Telegram loop stopping during bot resyncs and causing errors. The core strategy was to make the Telegram bot more resilient and self-healing within its own dedicated thread, minimizing the need for full restarts from the main bot.

**Summary of Implemented Changes:**

1.  **Enhanced `TradingTelegramBot` Resilience (`src/telegram/__init__.py`):**
    *   **New `_start_polling()` Method:** Extracted the core polling initiation logic into its own asynchronous method. This allows `TradingTelegramBot` to independently start or restart the Telegram API polling.
    *   **Refactored `initialize()` Method:** Now calls `_start_polling()` to initiate polling, centralizing this responsibility.
    *   **Resilient `run_polling()` Method:** This method (which runs in the dedicated Telegram thread) was completely refactored. It now contains a continuous loop that repeatedly attempts to call `_start_polling()`. It includes exponential backoff for retries (starting at 5 seconds, maxing at 60 seconds) if polling fails or stops, ensuring the Telegram bot thread remains active and self-healing.
    *   **New `_stop_polling()` Method:** Introduced an asynchronous method to gracefully stop the `telegram.ext.Updater` and the `telegram.ext.Application` components.
    *   **Simplified `shutdown()` Method:** Now utilizes the new `_stop_polling()` method for a cleaner and more centralized shutdown process for the Telegram application.

2.  **Refined Main Bot Interaction (`main.py`):**
    *   **Simplified `_run_telegram_loop()`:** The `_run_telegram_loop()` method (which is the target for the dedicated Telegram thread launched by the main bot) is now much simpler. Its primary role is to ensure the `TradingTelegramBot`'s dedicated `asyncio` loop is running and then to schedule `self.telegram_bot.run_polling()` within that loop. This delegates the continuous management of Telegram polling and its resilience entirely to the `TradingTelegramBot` class.
    *   **Robust `_restart_telegram_thread()`:** This method, acting as a last-resort recovery mechanism when the entire `telegram_thread` has died, was made more robust. It now correctly signals the old `TradingTelegramBot` instance for graceful shutdown, explicitly waits for the old thread to terminate, then creates a completely new `TradingTelegramBot` instance and starts a fresh dedicated thread for it. This ensures a clean slate when a full thread restart is unavoidable.
    *   **Adjusted `start()` Method:** The initial startup of the Telegram bot's dedicated thread in the `start()` method was refined to reflect the new structure and to gracefully handle initialization timeouts, including signaling the internal shutdown of the `TradingTelegramBot` instance if it fails to become ready.

**Impact and Resolution of Concern:**

With these changes, the Telegram bot is now significantly more fault-tolerant.
*   **Reduced Disruptions:** The `TradingTelegramBot`'s `run_polling()` method will now proactively attempt to restart polling if it ceases, largely eliminating the "window of unavailability" that caused command errors during bot resyncs.
*   **Improved Command Processing:** Commands sent to the bot should be processed more reliably, as the underlying Telegram `Application` is continuously running or actively recovering its polling mechanism within its persistent thread.
*   **Cleaner Recovery:** Even in the event of a critical thread failure, the `_restart_telegram_thread()` mechanism in `main.py` is now designed to perform a cleaner and more controlled full restart.

This comprehensive refactoring ensures a more stable and reliable Telegram integration, allowing for uninterrupted command processing and notification delivery even during periods of network instability or bot resyncs.
