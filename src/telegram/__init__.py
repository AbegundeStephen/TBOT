#!/usr/bin/env python3
"""
 Telegram Bot Interface for Trading Bot
Provides notifications and remote control capabilities
"""

import logging
import asyncio
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from telegram.request import HTTPXRequest

matplotlib.use("Agg")  # Non-interactive backend

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut, RetryAfter, TelegramError

logger = logging.getLogger(__name__)


class SignalMonitoringIntegration:
    """
    Signal monitoring features for Telegram bot
    Integrates with PerformanceWeightedAggregator to track signals in real-time
    """

    def __init__(self, max_history: int = 100):
        self.signal_history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = max_history
        self.regime_tracking: Dict[str, Dict] = defaultdict(
            lambda: {"current": None, "changes": [], "change_count": 0}
        )
        self.override_tracking: Dict[str, List[Dict]] = defaultdict(list)
        logger.info(
            f"SignalMonitoringIntegration initialized (max_history={max_history})"
        )

    def record_signal(
        self,
        asset: str,
        signal: int,
        details: Dict,
        price: float,
        timestamp: datetime = None,
    ):
        """Record a signal for monitoring"""
        if timestamp is None:
            timestamp = datetime.now()

        entry = {
            "timestamp": timestamp,
            "signal": signal,
            "price": price,
            "regime": details.get("regime"),
            "is_bull": details.get("is_bull_market"),
            "quality": details.get("signal_quality", 0),
            "reasoning": details.get("reasoning"),
            "mr_signal": details.get("mean_reversion_signal"),
            "mr_conf": details.get("mean_reversion_confidence", 0),
            "tf_signal": details.get("trend_following_signal"),
            "tf_conf": details.get("trend_following_confidence", 0),
            "ema_signal": details.get("ema_signal"),
            "ema_conf": details.get("ema_confidence", 0),
            "regime_changed": details.get("regime_changed", False),
        }

        self.signal_history[asset].append(entry)

        if len(self.signal_history[asset]) > self.max_history:
            self.signal_history[asset].pop(0)

        if entry["regime_changed"]:
            self.regime_tracking[asset]["changes"].append(
                {
                    "timestamp": timestamp,
                    "regime": entry["regime"],
                    "price": price,
                    "ema_conf": entry["ema_conf"],
                }
            )
            self.regime_tracking[asset]["change_count"] += 1

        if "override" in entry["reasoning"].lower():
            self.override_tracking[asset].append(
                {
                    "timestamp": timestamp,
                    "price": price,
                    "ema_conf": entry["ema_conf"],
                    "quality": entry["quality"],
                }
            )

    def get_last_signals(self, asset: str, n: int = 5) -> List[Dict]:
        """Get last N signals for an asset"""
        return self.signal_history[asset][-n:] if asset in self.signal_history else []

    def get_signal_statistics(self, asset: str) -> Dict:
        """Get signal statistics for an asset"""
        if asset not in self.signal_history or not self.signal_history[asset]:
            return {}

        signals = self.signal_history[asset]
        buy_count = sum(1 for s in signals if s["signal"] == 1)
        sell_count = sum(1 for s in signals if s["signal"] == -1)
        hold_count = sum(1 for s in signals if s["signal"] == 0)

        qualities = [s["quality"] for s in signals if s["signal"] != 0]

        return {
            "total_signals": len(signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "buy_pct": (buy_count / len(signals) * 100) if signals else 0,
            "sell_pct": (sell_count / len(signals) * 100) if signals else 0,
            "hold_pct": (hold_count / len(signals) * 100) if signals else 0,
            "avg_quality": np.mean(qualities) if qualities else 0,
            "high_quality_count": sum(1 for q in qualities if q >= 0.65),
        }

    def get_regime_info(self, asset: str) -> Dict:
        """Get regime information for an asset"""
        if asset not in self.regime_tracking:
            return {}

        tracking = self.regime_tracking[asset]

        return {
            "change_count": tracking["change_count"],
            "last_changes": tracking["changes"][-5:],
        }

    def get_override_info(self, asset: str) -> Dict:
        """Get override event information"""
        if asset not in self.override_tracking:
            return {"total": 0, "last_events": []}

        events = self.override_tracking[asset]
        avg_quality = np.mean([e["quality"] for e in events]) if events else 0

        return {
            "total": len(events),
            "avg_quality": avg_quality,
            "last_events": events[-5:],
        }


def admin_only(func):
    """Decorator to restrict commands to admin users only"""

    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in self.admin_ids:
            await update.message.reply_text(
                "🚫 *Access Denied*\n\n"
                "This command is restricted to authorized users only.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        return await func(self, update, context)

    return wrapper


class TradingTelegramBot:
    """
    Telegram bot interface for trading bot
    Handles notifications and user commands properly
    """

    def __init__(self, token: str, admin_ids: List[int], trading_bot):
        self.token = token
        self.admin_ids = admin_ids
        self.trading_bot = trading_bot
        self.application = None
        self.is_running = False

        self.signal_monitor = SignalMonitoringIntegration(max_history=100)

        self._is_ready = False
        self._init_lock = asyncio.Lock()
        self._message_queue = []

        # Track shutdown state
        self._shutdown_event = asyncio.Event()
        self._shutdown_complete = False

        # ✨ NEW: Network error tracking
        self._network_error_count = 0
        self._last_network_error = None
        self._max_consecutive_errors = 5
        self._reconnect_delay = 5  # seconds

        # Track notification counts to avoid spam
        self.notification_counts = {}
        self.last_daily_summary = None

        logger.info(f"TelegramBot initialized - Admins: {admin_ids}")

    async def initialize(self):
        """Initialize with improved error handling and retry logic"""
        request = HTTPXRequest(
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
            pool_timeout=60,
        )
        try:
            async with self._init_lock:
                self._shutdown_event.clear()
                self._shutdown_complete = False

                logger.info("[TELEGRAM] Building application...")

                # ✨ IMPROVED: Longer timeouts and connection pooling
                self.application = (
                    Application.builder()
                    .token(self.token)
                    .connect_timeout(60)  # Increased from 30
                    .read_timeout(60)  # Increased from 30
                    .write_timeout(60)  # Increased from 30
                    .pool_timeout(60)  # Increased from 30
                    .get_updates_read_timeout(60)  # Increased from 30
                    .connection_pool_size(8)  # ✨ NEW: Connection pooling
                    .build()
                )

                logger.info("[TELEGRAM] Registering command handlers...")
                self._register_handlers()

                logger.info("[TELEGRAM] Initializing application...")
                await self.application.initialize()

                logger.info("[TELEGRAM] Starting application...")
                await self.application.start()

                logger.info("[TELEGRAM] Starting update polling...")

                # ✨ IMPROVED: Better polling configuration
                await self.application.updater.start_polling(
                    poll_interval=2.0,  # Increased from 1.0 for stability
                    timeout=60,  # Increased from 30
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES,
                    bootstrap_retries=10,  # Increased from 5
                )

                # Wait for polling to stabilize
                await asyncio.sleep(3)  # Increased from 2

                if await self._verify_bot_ready():
                    self._is_ready = True
                    self.is_running = True
                    self._network_error_count = 0  # ✨ Reset error count
                    logger.info("[TELEGRAM] ✅ Bot fully operational")

                    await self.send_notification(
                        "🤖 *Trading Bot Started*\n\n"
                        f"Bot initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        "Type /help for available commands"
                    )

                    await self._process_message_queue()
                else:
                    logger.error("[TELEGRAM] ❌ Bot verification failed")
                    raise Exception("Bot failed to become ready")

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}", exc_info=True)
            self._is_ready = False
            raise

    async def _keepalive_monitor(self):
        """Monitor bot health and restart if needed"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                if self._is_ready:
                    # Ping the bot
                    await self.application.bot.get_me()
                    logger.debug("[TELEGRAM] Keepalive check passed")
            except Exception as e:
                logger.warning(f"[TELEGRAM] Keepalive failed: {e}")
                await self._attempt_reconnection()

    async def _verify_bot_ready(self) -> bool:
        """Verify bot can send messages with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                bot_info = await self.application.bot.get_me()
                logger.info(f"[TELEGRAM] Bot verified: @{bot_info.username}")
                return True
            except Exception as e:
                logger.warning(
                    f"[TELEGRAM] Verification attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        return False

    async def _process_message_queue(self):
        """Send any queued messages from before initialization"""
        if not self._message_queue:
            return

        logger.info(f"[TELEGRAM] Processing {len(self._message_queue)} queued messages")
        for msg in self._message_queue:
            await self.send_notification(msg, disable_preview=True)
        self._message_queue.clear()

    def _register_handlers(self):
        """Register all command handlers"""
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(
            CommandHandler("performance", self.cmd_performance)
        )
        self.application.add_handler(CommandHandler("signals", self.cmd_signals))
        self.application.add_handler(CommandHandler("stats", self.cmd_signal_stats))
        self.application.add_handler(CommandHandler("regimes", self.cmd_regimes))
        self.application.add_handler(CommandHandler("overrides", self.cmd_overrides))
        self.application.add_handler(
            CommandHandler("start_trading", self.cmd_start_trading)
        )
        self.application.add_handler(
            CommandHandler("stop_trading", self.cmd_stop_trading)
        )
        self.application.add_handler(CommandHandler("presets", self.cmd_presets))
        self.application.add_handler(CommandHandler("debug", self.cmd_debug_positions))
        self.application.add_handler(CommandHandler("close_all", self.cmd_close_all))
        self.application.add_handler(CommandHandler("close", self.cmd_close_asset))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

        # ✨ NEW: Add error handler
        self.application.add_error_handler(self.error_handler)

    async def error_handler(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        ✨ NEW: Global error handler for Telegram errors
        Handles network issues gracefully without crashing
        """
        error = context.error

        # Log all errors
        logger.error(
            f"[TELEGRAM] Error handler caught: {error}", exc_info=context.error
        )

        # Handle specific Telegram errors
        if isinstance(error, NetworkError):
            self._network_error_count += 1
            self._last_network_error = datetime.now()
            logger.warning(
                f"[TELEGRAM] Network error #{self._network_error_count}: {error}"
            )

            # If too many consecutive errors, attempt reconnection
            if self._network_error_count >= self._max_consecutive_errors:
                logger.error(
                    f"[TELEGRAM] Too many network errors ({self._network_error_count}), "
                    f"attempting reconnection..."
                )
                await self._attempt_reconnection()

        elif isinstance(error, TimedOut):
            logger.warning(f"[TELEGRAM] Request timed out: {error}")
            # Don't count timeouts as critical errors

        elif isinstance(error, RetryAfter):
            retry_after = error.retry_after
            logger.warning(f"[TELEGRAM] Rate limited, retry after {retry_after}s")
            await asyncio.sleep(retry_after)

        elif isinstance(error, TelegramError):
            logger.error(f"[TELEGRAM] Telegram error: {error}")

        else:
            logger.error(f"[TELEGRAM] Unexpected error: {error}")

        # Don't let errors crash the bot
        return None

    async def _attempt_reconnection(self):
        """
        ✨ NEW: Attempt to reconnect after network issues
        """
        try:
            logger.info(
                f"[TELEGRAM] Waiting {self._reconnect_delay}s before reconnection..."
            )
            await asyncio.sleep(self._reconnect_delay)

            logger.info("[TELEGRAM] Testing connection...")
            if await self._verify_bot_ready():
                logger.info("[TELEGRAM] ✅ Reconnection successful")
                self._network_error_count = 0
                self._last_network_error = None
            else:
                logger.error("[TELEGRAM] ❌ Reconnection failed")
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, 60
                )  # Exponential backoff

        except Exception as e:
            logger.error(f"[TELEGRAM] Reconnection error: {e}")

    async def run_polling(self):
        """
        ✨ IMPROVED: Keep the bot running with better error recovery
        """
        try:
            logger.info("[TELEGRAM] Starting polling loop...")
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    # ✨ NEW: Periodic health check
                    if self._network_error_count > 0:
                        time_since_error = (
                            (datetime.now() - self._last_network_error).total_seconds()
                            if self._last_network_error
                            else 999
                        )

                        # Reset error count if no errors for 5 minutes
                        if time_since_error > 300:
                            logger.info(
                                "[TELEGRAM] Resetting error count after recovery period"
                            )
                            self._network_error_count = 0
                            self._reconnect_delay = 5

                    await asyncio.sleep(1)

                except asyncio.CancelledError:
                    logger.info("[TELEGRAM] Polling cancelled")
                    raise
                except Exception as e:
                    logger.error(f"[TELEGRAM] Error in polling loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying

            logger.info("[TELEGRAM] Polling loop ended")

        except asyncio.CancelledError:
            logger.info("[TELEGRAM] Polling cancelled")
            raise
        except Exception as e:
            logger.error(f"[TELEGRAM] Fatal error in polling loop: {e}", exc_info=True)

    async def shutdown(self):
        """
        ✨ IMPROVED: Gracefully shutdown with better task cleanup
        """
        if self._shutdown_complete:
            logger.info("[TELEGRAM] Already shut down, skipping")
            return

        logger.info(
            "[TELEGRAM] ==================== SHUTDOWN STARTED ===================="
        )

        self._shutdown_event.set()
        self.is_running = False
        self._is_ready = False

        try:
            # Try to send shutdown notification
            if self.application and self._is_ready:
                try:
                    await asyncio.wait_for(
                        self.send_notification(
                            "🛑 *Trading Bot Stopped*\n\n"
                            f"Bot shutdown at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        ),
                        timeout=3.0,
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(
                        f"[TELEGRAM] Could not send shutdown notification: {e}"
                    )

            if self.application:
                # ✨ IMPROVED: More aggressive task cancellation
                # Cancel all pending tasks first
                try:
                    tasks = [t for t in asyncio.all_tasks() if not t.done()]
                    if tasks:
                        logger.info(
                            f"[TELEGRAM] Cancelling {len(tasks)} pending tasks..."
                        )
                        for task in tasks:
                            if not task.done():
                                task.cancel()

                        # Wait for cancellation with timeout
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=2.0,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("[TELEGRAM] Task cancellation timeout")
                except Exception as e:
                    logger.warning(f"[TELEGRAM] Error cancelling tasks: {e}")

                # Stop the updater
                if hasattr(self.application, "updater") and self.application.updater:
                    logger.info("[TELEGRAM] Stopping updater...")
                    try:
                        if self.application.updater.running:
                            await asyncio.wait_for(
                                self.application.updater.stop(), timeout=3.0
                            )
                            logger.info("[TELEGRAM] ✅ Updater stopped")
                    except asyncio.TimeoutError:
                        logger.warning("[TELEGRAM] ⚠️ Updater stop timed out")
                    except Exception as e:
                        logger.warning(f"[TELEGRAM] ⚠️ Updater stop error: {e}")

                # Stop the application
                logger.info("[TELEGRAM] Stopping application...")
                try:
                    if self.application.running:
                        await asyncio.wait_for(self.application.stop(), timeout=3.0)
                        logger.info("[TELEGRAM] ✅ Application stopped")
                except asyncio.TimeoutError:
                    logger.warning("[TELEGRAM] ⚠️ Application stop timed out")
                except Exception as e:
                    logger.warning(f"[TELEGRAM] ⚠️ Application stop error: {e}")

                # Shutdown the application
                logger.info("[TELEGRAM] Shutting down application...")
                try:
                    await asyncio.wait_for(self.application.shutdown(), timeout=3.0)
                    logger.info("[TELEGRAM] ✅ Application shutdown complete")
                except asyncio.TimeoutError:
                    logger.warning("[TELEGRAM] ⚠️ Application shutdown timed out")
                except Exception as e:
                    logger.warning(f"[TELEGRAM] ⚠️ Application shutdown error: {e}")

                self.application = None

            self._shutdown_complete = True
            logger.info(
                "[TELEGRAM] ==================== SHUTDOWN COMPLETE ===================="
            )

        except Exception as e:
            logger.error(f"[TELEGRAM] ❌ Error during shutdown: {e}", exc_info=True)
            self._shutdown_complete = True

    # ==================== COMMAND HANDLERS ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        is_admin = user_id in self.admin_ids

        welcome_msg = (
            "🤖 *Welcome to Trading Bot Control*\n\n"
            f"👤 User: @{username}\n"
            f"🆔 ID: `{user_id}`\n"
            f"🔐 Access: {'✅ Admin' if is_admin else '❌ Guest'}\n\n"
        )

        if is_admin:
            welcome_msg += (
                "You have full access to all bot commands.\n"
                "Use /help to see available commands."
            )
        else:
            welcome_msg += (
                "⚠️ You are not authorized to control this bot.\n"
                "Contact the bot administrator for access."
            )

        await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"User {user_id} ({username}) started bot - Admin: {is_admin}")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_id = update.effective_user.id
        is_admin = user_id in self.admin_ids

        help_text = (
            "📋 *Available Commands*\n\n"
            "*📊 Information Commands:*\n"
            "/status - Current bot status and portfolio\n"
            "/positions - View open positions with P&L\n"
            "/history - Recent trade history\n"
            "/performance - Performance metrics\n"
            "/presets - View current aggregator presets\n"
            "/signals - Latest trading signals\n"
            "/stats - Signal statistics\n"
            "/regimes - Market regime tracking\n"
            "/overrides - Golden Cross overrides\n"
            "/help - Show this help message\n\n"
        )

        if is_admin:
            help_text += (
                "*🎮 Control Commands (Admin Only):*\n"
                "/start\\_trading - Resume trading operations\n"
                "/stop\\_trading - Pause trading (keep positions)\n"
                "/close\\_all - Close all open positions\n"
                "/close BTC - Close BTC position\n"
                "/close GOLD - Close GOLD position\n\n"
                "⚠️ *Control commands are restricted to authorized users*"
            )
        else:
            help_text += (
                "🔒 Control commands are restricted to admins.\n"
                "You have read-only access."
            )

        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            status_icon = "🟢" if self.trading_bot.is_running else "🔴"

            total_value = portfolio_status.get("total_value", 0)
            cash = portfolio_status.get("cash", 0)
            open_positions = portfolio_status.get("open_positions", 0)
            daily_pnl = portfolio_status.get("daily_pnl", 0)

            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_sign = "+" if daily_pnl >= 0 else ""

            # Check market status
            btc_status = "✅ 24/7 Open"
            gold_status = self._get_gold_market_status()

            status_msg = (
                f"{status_icon} *Bot Status*\n\n"
                f"🤖 Trading: {'Running' if self.trading_bot.is_running else 'Stopped'}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"💵 Cash: ${cash:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n\n"
                f"*Market Status:*\n"
                f"₿ BTC: {btc_status}\n"
                f"🥇 GOLD: {gold_status}\n\n"
            )

            # ✨ NEW: Add preset info
            if (
                hasattr(self.trading_bot, "selected_presets")
                and self.trading_bot.selected_presets
            ):
                status_msg += "*Current Presets:*\n"
                for asset, preset in self.trading_bot.selected_presets.items():
                    emoji = "₿" if asset == "BTC" else "🥇"
                    status_msg += f"{emoji} {asset}: `{preset.upper()}`\n"
                status_msg += "\n"

            status_msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            # Add inline keyboard for quick actions
            keyboard = [
                [
                    InlineKeyboardButton("📊 Positions", callback_data="positions"),
                    InlineKeyboardButton("📜 History", callback_data="history"),
                ],
                [
                    InlineKeyboardButton("⚙️ Presets", callback_data="presets"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="status")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                status_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_status: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Error fetching status. Please try again."
            )

    async def _send_status_message(self, query):
        """Send status message (for callback)"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            status_icon = "🟢" if self.trading_bot.is_running else "🔴"

            total_value = portfolio_status.get("total_value", 0)
            cash = portfolio_status.get("cash", 0)
            open_positions = portfolio_status.get("open_positions", 0)
            daily_pnl = portfolio_status.get("daily_pnl", 0)

            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_sign = "+" if daily_pnl >= 0 else ""

            btc_status = "✅ 24/7 Open"
            gold_status = self._get_gold_market_status()

            status_msg = (
                f"{status_icon} *Bot Status*\n\n"
                f"🤖 Trading: {'Running' if self.trading_bot.is_running else 'Stopped'}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"💵 Cash: ${cash:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n\n"
                f"*Market Status:*\n"
                f"₿ BTC: {btc_status}\n"
                f"🥇 GOLD: {gold_status}\n\n"
            )

            # ✨ NEW: Add preset info
            if (
                hasattr(self.trading_bot, "selected_presets")
                and self.trading_bot.selected_presets
            ):
                status_msg += "*Current Presets:*\n"
                for asset, preset in self.trading_bot.selected_presets.items():
                    emoji = "₿" if asset == "BTC" else "🥇"
                    status_msg += f"{emoji} {asset}: `{preset.upper()}`\n"
                status_msg += "\n"

            status_msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Positions", callback_data="positions"),
                    InlineKeyboardButton("📜 History", callback_data="history"),
                ],
                [
                    InlineKeyboardButton("⚙️ Presets", callback_data="presets"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="status")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                status_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_status_message: {e}")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - Show open positions with P&L, entry price, and current price"""
        try:
            # Fetch the latest prices for all assets
            current_prices = {}
            for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.trading_bot.binance_handler
                    if exchange == "binance"
                    else self.trading_bot.mt5_handler
                )

                if handler:
                    try:
                        price = handler.get_current_price()
                        if price and price > 0:
                            current_prices[asset_name] = price
                            logger.debug(
                                f"[TELEGRAM] {asset_name} price: ${price:,.2f}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to get {asset_name} price: {e}")

            # Update positions with the latest prices (this updates MT5/Binance profit)
            self.trading_bot.portfolio_manager.update_positions(current_prices)

            # Get the updated portfolio status with current prices
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status(
                current_prices
            )
            positions = portfolio_status.get("positions", {})

            if not positions:
                await update.message.reply_text(
                    "📭 *No Open Positions*\n\nCurrently no active trades.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            positions_msg = "📊 *Open Positions*\n\n"

            for asset, pos in positions.items():
                side = pos["side"].upper()
                side_icon = "🟢" if side == "LONG" else "🔴"

                # Get accurate values
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                quantity = pos.get("quantity", 0)
                current_value = pos.get("current_value", 0)

                # Get P&L - prioritize exchange-reported profit
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0) * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                positions_msg += (
                    f"{side_icon} *{asset} - {side}*\n"
                    f"📍 Entry Price: ${entry_price:,.2f}\n"
                    f"💹 Current Price: ${current_price:,.2f}\n"
                    f"📦 Quantity: {quantity:.6f}\n"
                    f"💰 Current Value: ${current_value:,.2f}\n"
                    f"{pnl_icon} P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                )

                # Show exchange-specific tracking if available
                if pos.get("mt5_ticket"):
                    positions_msg += f"🎫 MT5 Ticket: {pos['mt5_ticket']}\n"
                    if pos.get("mt5_profit") is not None:
                        mt5_profit = pos["mt5_profit"]
                        positions_msg += f"   MT5 P&L: ${mt5_profit:,.2f}\n"

                if pos.get("binance_order_id"):
                    positions_msg += f"🎫 Binance Order: {pos['binance_order_id']}\n"
                    if pos.get("binance_profit") is not None:
                        binance_profit = pos["binance_profit"]
                        positions_msg += f"   Binance P&L: ${binance_profit:,.2f}\n"

                # Show stop loss and take profit
                if pos.get("stop_loss"):
                    sl = pos["stop_loss"]
                    sl_dist = abs(current_price - sl)
                    sl_pct = (sl_dist / current_price * 100) if current_price > 0 else 0
                    positions_msg += f"🛑 Stop Loss: ${sl:,.2f} ({sl_pct:.2f}% away)\n"

                if pos.get("take_profit"):
                    tp = pos["take_profit"]
                    tp_dist = abs(tp - current_price)
                    tp_pct = (tp_dist / current_price * 100) if current_price > 0 else 0
                    positions_msg += (
                        f"🎯 Take Profit: ${tp:,.2f} ({tp_pct:.2f}% away)\n"
                    )

                positions_msg += "\n"

            # Add summary
            total_exposure = portfolio_status.get("total_exposure", 0)
            total_unrealized = portfolio_status.get("total_unrealized_pnl", 0)
            unrealized_icon = "🟢" if total_unrealized >= 0 else "🔴"
            unrealized_sign = "+" if total_unrealized >= 0 else ""

            positions_msg += (
                f"*Portfolio Summary*\n"
                f"💼 Total Exposure: ${total_exposure:,.2f}\n"
                f"{unrealized_icon} Total realized P&L: {unrealized_sign}${total_unrealized:,.2f}\n\n"
            )

            positions_msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            # Add inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                positions_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_positions: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching positions.")

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        try:
            closed_positions = self.trading_bot.portfolio_manager.closed_positions

            if not closed_positions:
                await update.message.reply_text(
                    "📭 *No Trade History*\n\n" "No completed trades yet.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # Get last 10 trades
            recent_trades = closed_positions[-10:]

            history_msg = "📜 *Recent Trade History*\n\n"

            for trade in reversed(recent_trades):
                asset = trade["asset"]
                side = trade["side"].upper()
                pnl = trade["pnl"]
                pnl_pct = trade["pnl_pct"] * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                exit_time = trade["exit_time"].strftime("%m/%d %H:%M")
                reason = trade["reason"].replace("_", " ").title()

                history_msg += (
                    f"{pnl_icon} *{asset} {side}*\n"
                    f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                    f"Exit: {exit_time} | {reason}\n\n"
                )

            await update.message.reply_text(history_msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in cmd_history: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching history.")

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        try:
            closed_positions = self.trading_bot.portfolio_manager.closed_positions
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            if not closed_positions:
                await update.message.reply_text(
                    "📊 *No Performance Data*\n\n"
                    "Not enough trades for performance metrics.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # Calculate metrics
            total_trades = len(closed_positions)
            winning_trades = sum(1 for t in closed_positions if t["pnl"] > 0)
            losing_trades = sum(1 for t in closed_positions if t["pnl"] < 0)

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t["pnl"] for t in closed_positions)
            avg_win = (
                sum(t["pnl"] for t in closed_positions if t["pnl"] > 0) / winning_trades
                if winning_trades > 0
                else 0
            )
            avg_loss = (
                sum(t["pnl"] for t in closed_positions if t["pnl"] < 0) / losing_trades
                if losing_trades > 0
                else 0
            )

            profit_factor = (
                abs(avg_win * winning_trades / (avg_loss * losing_trades))
                if losing_trades > 0 and avg_loss != 0
                else 0
            )

            initial_capital = self.trading_bot.portfolio_manager.initial_capital
            current_equity = portfolio_status.get("equity", initial_capital)
            total_return = (current_equity - initial_capital) / initial_capital * 100

            perf_icon = "🟢" if total_return >= 0 else "🔴"

            performance_msg = (
                f"{perf_icon} *Performance Metrics*\n\n"
                f"📊 Total Trades: {total_trades}\n"
                f"🟢 Winning: {winning_trades} ({win_rate:.1f}%)\n"
                f"🔴 Losing: {losing_trades}\n\n"
                f"💰 Total P&L: ${total_pnl:,.2f}\n"
                f"📈 Avg Win: ${avg_win:,.2f}\n"
                f"📉 Avg Loss: ${avg_loss:,.2f}\n"
                f"⚖️ Profit Factor: {profit_factor:.2f}\n\n"
                f"💵 Initial Capital: ${initial_capital:,.2f}\n"
                f"💎 Current Equity: ${current_equity:,.2f}\n"
                f"📊 Total Return: {total_return:+.2f}%\n"
            )

            await update.message.reply_text(
                performance_msg, parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error in cmd_performance: {e}", exc_info=True)
            await update.message.reply_text("❌ Error calculating performance.")

    @admin_only
    async def cmd_start_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /start_trading command"""
        if self.trading_bot.is_running:
            await update.message.reply_text("ℹ️ Trading is already running.")
        else:
            self.trading_bot.is_running = True
            await update.message.reply_text(
                "🟢 *Trading Resumed*\n\n" "Bot will now process trading signals.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info("Trading resumed via Telegram command")

    @admin_only
    async def cmd_stop_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /stop_trading command"""
        if not self.trading_bot.is_running:
            await update.message.reply_text("ℹ️ Trading is already stopped.")
        else:
            self.trading_bot.is_running = False
            await update.message.reply_text(
                "🔴 *Trading Stopped*\n\n"
                "Bot will not open new positions.\n"
                "Existing positions remain open.\n\n"
                "Use /close\\_all to close all positions.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info("Trading stopped via Telegram command")

    @admin_only
    async def cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close_all command"""
        try:
            # Get current prices
            prices = {}
            if self.trading_bot.binance_handler:
                btc_price = self.trading_bot.binance_handler.get_current_price()
                if btc_price:
                    prices["BTC"] = btc_price

            if self.trading_bot.mt5_handler:
                gold_price = self.trading_bot.mt5_handler.get_current_price()
                if gold_price:
                    prices["GOLD"] = gold_price

            # Close all positions
            self.trading_bot.portfolio_manager.close_all_positions(prices)

            await update.message.reply_text(
                "✅ *All Positions Closed*\n\n"
                "All open positions have been closed.\n"
                "Check /history for trade results.",
                parse_mode=ParseMode.MARKDOWN,
            )

            logger.info("All positions closed via Telegram command")

        except Exception as e:
            logger.error(f"Error closing all positions: {e}", exc_info=True)
            await update.message.reply_text("❌ Error closing positions. Check logs.")

    @admin_only
    async def cmd_close_asset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close <asset> command"""
        try:
            if not context.args:
                await update.message.reply_text("⚠️ Usage: /close BTC or /close GOLD")
                return

            asset = context.args[0].upper()

            if asset not in ["BTC", "GOLD"]:
                await update.message.reply_text(
                    "⚠️ Invalid asset. Use: /close BTC or /close GOLD"
                )
                return

            # Check if position exists
            if not self.trading_bot.portfolio_manager.has_position(asset):
                await update.message.reply_text(f"ℹ️ No open position for {asset}")
                return

            # Get current price
            if asset == "BTC" and self.trading_bot.binance_handler:
                current_price = self.trading_bot.binance_handler.get_current_price()
            elif asset == "GOLD" and self.trading_bot.mt5_handler:
                current_price = self.trading_bot.mt5_handler.get_current_price()
            else:
                await update.message.reply_text(f"❌ Cannot get price for {asset}")
                return

            # Close position
            result = self.trading_bot.portfolio_manager.close_position(
                asset=asset, exit_price=current_price, reason="manual_telegram"
            )

            if result:
                pnl = result["pnl"]
                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                await update.message.reply_text(
                    f"{pnl_icon} *{asset} Position Closed*\n\n"
                    f"P&L: {pnl_sign}${pnl:,.2f}\n"
                    f"Reason: Manual close via Telegram",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(f"❌ Failed to close {asset} position")

        except Exception as e:
            logger.error(f"Error closing asset: {e}", exc_info=True)
            await update.message.reply_text("❌ Error closing position.")

    async def cmd_presets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /presets command - Show current aggregator presets"""
        try:
            if not hasattr(self.trading_bot, "selected_presets"):
                await update.message.reply_text(
                    "❌ Preset information not available.\n"
                    "Presets are determined during bot startup."
                )
                return

            presets = self.trading_bot.selected_presets

            if not presets:
                await update.message.reply_text(
                    "ℹ️ No presets configured yet.\n" "Bot may still be initializing."
                )
                return

            # Check if auto-mode was used
            aggregator_cfg = self.trading_bot.config.get("aggregator_settings", {})
            preset_mode = aggregator_cfg.get("preset", "auto")

            msg = "⚙️ *Aggregator Preset Configuration*\n\n"

            if preset_mode == "auto":
                msg += "🤖 *Mode:* AUTO-SELECT\n"
                msg += "Presets are automatically selected based on current market conditions.\n\n"
            else:
                msg += f"🔧 *Mode:* MANUAL ({preset_mode.upper()})\n"
                msg += "All assets use the same preset.\n\n"

            msg += "*Current Presets:*\n"

            # BTC Preset
            if "BTC" in presets:
                preset = presets["BTC"]
                msg += f"\n₿ *BTC:* `{preset.upper()}`\n"
                msg += self._get_preset_description(preset)

            # GOLD Preset
            if "GOLD" in presets:
                preset = presets["GOLD"]
                msg += f"\n🥇 *GOLD:* `{preset.upper()}`\n"
                msg += self._get_preset_description(preset)

            msg += f"\n\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            # Add inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="presets")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_presets: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching preset information")

    def _get_preset_description(self, preset: str) -> str:
        """Get description for a preset"""
        descriptions = {
            "conservative": "  • Low risk, high thresholds\n  • Best for stable markets",
            "balanced": "  • Moderate risk/reward\n  • Default for most conditions",
            "aggressive": "  • Higher frequency trading\n  • Best for trending markets",
            "scalper": "  • Maximum activity\n  • Best for high volatility",
        }
        return descriptions.get(preset, "  • Unknown preset")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()

        callback_data = query.data

        if callback_data == "status":
            await self._send_status_message(query)
        elif callback_data == "positions":
            await self._send_positions_message(query)
        elif callback_data == "history":
            await self._send_history_message(query)
        elif callback_data == "presets":  # ✨ NEW
            await self._send_presets_message(query)
        elif callback_data == "signals":
            await self._send_signals_message(query)
        # ==================== NOTIFICATION METHODS ====================

    async def _send_presets_message(self, query):
        """Send presets message (for callback)"""
        try:
            if not hasattr(self.trading_bot, "selected_presets"):
                await query.edit_message_text(
                    "❌ Preset information not available.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            presets = self.trading_bot.selected_presets

            if not presets:
                await query.edit_message_text(
                    "ℹ️ No presets configured yet.", parse_mode=ParseMode.MARKDOWN
                )
                return

            aggregator_cfg = self.trading_bot.config.get("aggregator_settings", {})
            preset_mode = aggregator_cfg.get("preset", "auto")

            msg = "⚙️ *Aggregator Preset Configuration*\n\n"

            if preset_mode == "auto":
                msg += "🤖 *Mode:* AUTO-SELECT\n\n"
            else:
                msg += f"🔧 *Mode:* MANUAL ({preset_mode.upper()})\n\n"

            msg += "*Current Presets:*\n"

            if "BTC" in presets:
                preset = presets["BTC"]
                msg += f"\n₿ *BTC:* `{preset.upper()}`\n"
                msg += self._get_preset_description(preset)

            if "GOLD" in presets:
                preset = presets["GOLD"]
                msg += f"\n🥇 *GOLD:* `{preset.upper()}`\n"
                msg += self._get_preset_description(preset)

            msg += f"\n\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="presets")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_presets_message: {e}")

    async def _send_signals_message(self, query):
        """Send signals message (for callback)"""
        try:
            if not hasattr(self, "signal_monitor"):
                await query.edit_message_text("❌ Signal monitoring not initialized")
                return

            btc_signals = self.signal_monitor.get_last_signals("BTC", n=3)
            gold_signals = self.signal_monitor.get_last_signals("GOLD", n=3)

            msg = "📡 *Latest Trading Signals*\n\n"

            msg += "*₿ BTC*\n"
            if btc_signals:
                for sig in reversed(btc_signals):
                    msg += self._format_signal_entry(sig)
            else:
                msg += "No signals yet\n"

            msg += "\n*🥇 GOLD*\n"
            if gold_signals:
                for sig in reversed(gold_signals):
                    msg += self._format_signal_entry(sig)
            else:
                msg += "No signals yet\n"

            msg += f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Stats", callback_data="stats"),
                    InlineKeyboardButton("⚙️ Presets", callback_data="presets"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="signals")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_signals_message: {e}")

    async def send_notification(self, message: str, disable_preview: bool = True):
        """
        ✨ IMPROVED: Send notification with retry logic and error handling
        """
        # If not ready, queue the message
        if not self._is_ready:
            logger.warning("[TELEGRAM] Bot not ready, queuing message")
            self._message_queue.append(message)
            return

        if not self.is_running or not self.application:
            logger.warning("[TELEGRAM] Bot not running, cannot send notification")
            return

        success_count = 0
        max_retries = 3

        for admin_id in self.admin_ids:
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(
                        self.application.bot.send_message(
                            chat_id=admin_id,
                            text=message,
                            parse_mode=ParseMode.MARKDOWN,
                            disable_web_page_preview=disable_preview,
                        ),
                        timeout=10.0,  # ✨ NEW: Timeout for send operation
                    )
                    success_count += 1
                    logger.debug(f"[TELEGRAM] Notification sent to {admin_id}")
                    break  # Success, no need to retry

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[TELEGRAM] Timeout sending to {admin_id} (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)

                except (NetworkError, TimedOut) as e:
                    logger.warning(
                        f"[TELEGRAM] Network error sending to {admin_id} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"[TELEGRAM] Failed to send to {admin_id}: {e}")
                    break  # Don't retry on non-network errors

        if success_count == 0:
            logger.error("[TELEGRAM] Failed to send notification to any admin")
        else:
            logger.info(
                f"[TELEGRAM] Notification sent to {success_count}/{len(self.admin_ids)} admins"
            )

    async def notify_trade_opened(
        self, asset: str, side: str, price: float, size: float, sl: float, tp: float
    ):
        """Notify when a trade is opened"""
        side_icon = "🟢" if side.lower() == "long" else "🔴"

        msg = (
            f"{side_icon} *Trade Opened: {asset}*\n\n"
            f"Side: {side.upper()}\n"
            f"Entry Price: ${price:,.2f}\n"
            f"Size: ${size:,.2f}\n"
            f"🛑 Stop Loss: ${sl:,.2f}\n"
            f"🎯 Take Profit: ${tp:,.2f}\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_notification(msg)

    async def notify_trade_closed(
        self, asset: str, side: str, pnl: float, pnl_pct: float, reason: str
    ):
        """Notify when a trade is closed"""
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""

        reason_formatted = reason.replace("_", " ").title()

        msg = (
            f"{pnl_icon} *Trade Closed: {asset}*\n\n"
            f"Side: {side.upper()}\n"
            f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"Reason: {reason_formatted}\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_notification(msg)

    async def notify_error(self, error_msg: str):
        """Notify about errors"""
        msg = (
            f"⚠️ *Error Alert*\n\n"
            f"{error_msg}\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_notification(msg)

    async def send_daily_summary(self):
        """Send end-of-day performance summary - PROPERLY ASYNC"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            daily_pnl = portfolio_status.get("daily_pnl", 0)
            total_value = portfolio_status.get("total_value", 0)
            open_positions = portfolio_status.get("open_positions", 0)

            # Get today's trades
            closed_positions = self.trading_bot.portfolio_manager.closed_positions
            today = datetime.now().date()
            today_trades = [
                t for t in closed_positions if t["exit_time"].date() == today
            ]

            winning_today = sum(1 for t in today_trades if t["pnl"] > 0)
            losing_today = sum(1 for t in today_trades if t["pnl"] < 0)

            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_sign = "+" if daily_pnl >= 0 else ""

            msg = (
                f"📊 *Daily Summary - {today.strftime('%Y-%m-%d')}*\n\n"
                # f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n\n"
                f"📊 Today's Trades: {len(today_trades)}\n"
                f"🟢 Winning: {winning_today}\n"
                f"🔴 Losing: {losing_today}\n"
            )

            await self.send_notification(msg)
            self.last_daily_summary = datetime.now()
            logger.info("[TELEGRAM] Daily summary sent successfully")

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}", exc_info=True)

    # ==================== HELPER METHODS ====================

    def _get_gold_market_status(self) -> str:
        """Get GOLD market status from trading bot"""
        try:
            from src.utils.market_hours import MarketHours, should_trade_gold

            is_open = should_trade_gold()
            status, message = MarketHours.get_market_status("gold")

            if is_open:
                return "✅ Open"
            else:
                return f"🔴 Closed - {message.split('-')[1].strip() if '-' in message else 'Weekend'}"
        except:
            return "❓ Unknown"

    async def _send_status_message(self, query):
        """Send status message (for callback)"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            status_icon = "🟢" if self.trading_bot.is_running else "🔴"

            total_value = portfolio_status.get("total_value", 0)
            cash = portfolio_status.get("cash", 0)
            open_positions = portfolio_status.get("open_positions", 0)
            daily_pnl = portfolio_status.get("daily_pnl", 0)

            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_sign = "+" if daily_pnl >= 0 else ""

            btc_status = "✅ 24/7 Open"
            gold_status = self._get_gold_market_status()

            status_msg = (
                f"{status_icon} *Bot Status*\n\n"
                f"🤖 Trading: {'Running' if self.trading_bot.is_running else 'Stopped'}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"💵 Cash: ${cash:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n"
                # f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n\n"
                f"*Market Status:*\n"
                f"₿ BTC: {btc_status}\n"
                f"🥇 GOLD: {gold_status}\n\n"
                f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"
            )

            keyboard = [
                [
                    InlineKeyboardButton("📊 Positions", callback_data="positions"),
                    InlineKeyboardButton("📜 History", callback_data="history"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="status")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                status_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_status_message: {e}")

    async def _send_positions_message(self, query):
        """Send positions message (for callback) - FIXED VERSION"""
        try:
            # Fetch current prices
            current_prices = {}
            for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.trading_bot.binance_handler
                    if exchange == "binance"
                    else self.trading_bot.mt5_handler
                )

                if handler:
                    try:
                        price = handler.get_current_price()
                        if price and price > 0:
                            current_prices[asset_name] = price
                    except Exception as e:
                        logger.debug(f"Failed to get {asset_name} price: {e}")

            # Update positions with latest prices
            self.trading_bot.portfolio_manager.update_positions(current_prices)

            # Get updated status
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status(
                current_prices
            )
            positions = portfolio_status.get("positions", {})

            if not positions:
                await query.edit_message_text(
                    "📭 *No Open Positions*\n\nCurrently no active trades.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            positions_msg = "📊 *Open Positions*\n\n"

            for asset, pos in positions.items():
                side = pos["side"].upper()
                side_icon = "🟢" if side == "LONG" else "🔴"

                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                quantity = pos.get("quantity", 0)
                current_value = pos.get("current_value", 0)
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0) * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                positions_msg += (
                    f"{side_icon} *{asset} - {side}*\n"
                    f"📍 Entry: ${entry_price:,.2f}\n"
                    f"💹 Current: ${current_price:,.2f}\n"
                    f"📦 Qty: {quantity:.6f}\n"
                    f"💰 Value: ${current_value:,.2f}\n"
                    f"{pnl_icon} P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                )

                if pos.get("stop_loss"):
                    positions_msg += f"🛑 SL: ${pos['stop_loss']:,.2f}\n"
                if pos.get("take_profit"):
                    positions_msg += f"🎯 TP: ${pos['take_profit']:,.2f}\n"

                positions_msg += "\n"

            # Add summary
            total_unrealized = portfolio_status.get("total_unrealized_pnl", 0)
            unrealized_icon = "🟢" if total_unrealized >= 0 else "🔴"
            unrealized_sign = "+" if total_unrealized >= 0 else ""

            positions_msg += (
                f"{unrealized_icon} Total Unrealized: {unrealized_sign}${total_unrealized:,.2f}\n"
                f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"
            )

            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                positions_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_positions_message: {e}", exc_info=True)

    async def cmd_debug_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Debug command to show raw position data"""
        try:
            # Get current prices
            current_prices = {}
            for asset_name in ["BTC", "GOLD"]:
                asset_cfg = self.trading_bot.config["assets"].get(asset_name, {})
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.trading_bot.binance_handler
                    if exchange == "binance"
                    else self.trading_bot.mt5_handler
                )

                if handler:
                    try:
                        current_prices[asset_name] = handler.get_current_price()
                    except:
                        pass

            # Update positions
            self.trading_bot.portfolio_manager.update_positions(current_prices)

            # Get raw position data
            positions = self.trading_bot.portfolio_manager.positions

            msg = "🔍 *Debug: Raw Position Data*\n\n"

            for asset, pos in positions.items():
                msg += f"*{asset}*\n"
                msg += f"Entry Price: {pos.entry_price}\n"
                msg += f"Quantity: {pos.quantity}\n"
                msg += f"Side: {pos.side}\n"
                msg += f"MT5 Ticket: {pos.mt5_ticket}\n"
                msg += f"MT5 Profit: {pos.mt5_profit}\n"
                msg += f"Binance Order: {pos.binance_order_id}\n"
                msg += f"Binance Profit: {pos.binance_profit}\n"

                if asset in current_prices:
                    msg += f"Current Price (fetched): ${current_prices[asset]:,.2f}\n"
                    calc_pnl = pos.get_pnl(current_prices[asset])
                    msg += f"Calculated P&L: ${calc_pnl:,.2f}\n"

                msg += "\n"

            # Get portfolio status
            status = self.trading_bot.portfolio_manager.get_portfolio_status(
                current_prices
            )
            msg += f"*Portfolio Status P&L*\n"
            msg += f"Daily P&L: ${status.get('daily_pnl', 0):,.2f}\n"
            msg += f"Unrealized P&L: ${status.get('total_unrealized_pnl', 0):,.2f}\n"
            msg += f"Realized Today: ${status.get('realized_pnl_today', 0):,.2f}\n"

            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in cmd_debug_positions: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Debug error: {str(e)}")

    async def _send_history_message(self, query):
        """Send history message (for callback)"""
        try:
            closed_positions = self.trading_bot.portfolio_manager.closed_positions

            if not closed_positions:
                await query.edit_message_text(
                    "📭 *No Trade History*\n\n" "No completed trades yet.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            recent_trades = closed_positions[-10:]

            history_msg = "📜 *Recent Trade History*\n\n"

            for trade in reversed(recent_trades):
                asset = trade["asset"]
                side = trade["side"].upper()
                pnl = trade["pnl"]
                pnl_pct = trade["pnl_pct"] * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                exit_time = trade["exit_time"].strftime("%m/%d %H:%M")
                reason = trade["reason"].replace("_", " ").title()

                history_msg += (
                    f"{pnl_icon} *{asset} {side}*\n"
                    f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                    f"Exit: {exit_time} | {reason}\n\n"
                )

            await query.edit_message_text(history_msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in _send_history_message: {e}")

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command - Show latest signals"""
        try:
            if not hasattr(self, "signal_monitor"):
                await update.message.reply_text("❌ Signal monitoring not initialized")
                return

            # Get signals for both assets
            btc_signals = self.signal_monitor.get_last_signals("BTC", n=3)
            gold_signals = self.signal_monitor.get_last_signals("GOLD", n=3)

            msg = "📡 *Latest Trading Signals*\n\n"

            # BTC Signals
            msg += "*₿ BTC*\n"
            if btc_signals:
                for sig in reversed(btc_signals):
                    msg += self._format_signal_entry(sig)
            else:
                msg += "No signals yet\n"

            msg += "\n*🥇 GOLD*\n"
            if gold_signals:
                for sig in reversed(gold_signals):
                    msg += self._format_signal_entry(sig)
            else:
                msg += "No signals yet\n"

            msg += f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 BTC Stats", callback_data="btc_stats"),
                    InlineKeyboardButton("🥇 GOLD Stats", callback_data="gold_stats"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="signals")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_signals: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching signals")

    async def cmd_signal_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /stats command - Show signal statistics"""
        try:
            if not hasattr(self, "signal_monitor"):
                await update.message.reply_text("❌ Signal monitoring not initialized")
                return

            btc_stats = self.signal_monitor.get_signal_statistics("BTC")
            gold_stats = self.signal_monitor.get_signal_statistics("GOLD")

            msg = "📊 *Signal Statistics*\n\n"

            # BTC Stats
            if btc_stats:
                msg += "*₿ BTC*\n"
                msg += f"Total: {btc_stats['total_signals']}\n"
                msg += f"🟢 BUY: {btc_stats['buy_signals']} ({btc_stats['buy_pct']:.1f}%)\n"
                msg += f"🔴 SELL: {btc_stats['sell_signals']} ({btc_stats['sell_pct']:.1f}%)\n"
                msg += f"⚪ HOLD: {btc_stats['hold_signals']} ({btc_stats['hold_pct']:.1f}%)\n"
                msg += f"⭐ Avg Quality: {btc_stats['avg_quality']:.2f}\n"
                msg += f"✨ High Quality: {btc_stats['high_quality_count']}\n\n"

            # GOLD Stats
            if gold_stats:
                msg += "*🥇 GOLD*\n"
                msg += f"Total: {gold_stats['total_signals']}\n"
                msg += f"🟢 BUY: {gold_stats['buy_signals']} ({gold_stats['buy_pct']:.1f}%)\n"
                msg += f"🔴 SELL: {gold_stats['sell_signals']} ({gold_stats['sell_pct']:.1f}%)\n"
                msg += f"⚪ HOLD: {gold_stats['hold_signals']} ({gold_stats['hold_pct']:.1f}%)\n"
                msg += f"⭐ Avg Quality: {gold_stats['avg_quality']:.2f}\n"
                msg += f"✨ High Quality: {gold_stats['high_quality_count']}\n"

            keyboard = [
                [InlineKeyboardButton("📡 Recent Signals", callback_data="signals")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="stats")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_signal_stats: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching statistics")

    async def cmd_regimes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regimes command - Show regime information"""
        try:
            if not hasattr(self, "signal_monitor"):
                await update.message.reply_text("❌ Signal monitoring not initialized")
                return

            btc_regime = self.signal_monitor.get_regime_info("BTC")
            gold_regime = self.signal_monitor.get_regime_info("GOLD")

            msg = "🔄 *Market Regimes*\n\n"

            # BTC Regime
            msg += "*₿ BTC*\n"
            msg += f"Regime Changes: {btc_regime.get('change_count', 0)}\n"
            if btc_regime.get("last_changes"):
                msg += "Recent Changes:\n"
                for change in btc_regime["last_changes"][-3:]:
                    timestamp = change["timestamp"].strftime("%H:%M")
                    regime = "🚀 BULL" if "BULL" in change["regime"] else "⚖️ BEAR"
                    msg += f"  {timestamp}: {regime} @ ${change['price']:,.2f}\n"
            msg += "\n"

            # GOLD Regime
            msg += "*🥇 GOLD*\n"
            msg += f"Regime Changes: {gold_regime.get('change_count', 0)}\n"
            if gold_regime.get("last_changes"):
                msg += "Recent Changes:\n"
                for change in gold_regime["last_changes"][-3:]:
                    timestamp = change["timestamp"].strftime("%H:%M")
                    regime = "🚀 BULL" if "BULL" in change["regime"] else "⚖️ BEAR"
                    msg += f"  {timestamp}: {regime} @ ${change['price']:,.2f}\n"

            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="regimes")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_regimes: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching regime data")

    async def cmd_overrides(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /overrides command - Show Golden Cross override events"""
        try:
            if not hasattr(self, "signal_monitor"):
                await update.message.reply_text("❌ Signal monitoring not initialized")
                return

            btc_overrides = self.signal_monitor.get_override_info("BTC")
            gold_overrides = self.signal_monitor.get_override_info("GOLD")

            msg = "🔒 *Golden Cross Overrides*\n"
            msg += "(Sells blocked during bull market)\n\n"

            # BTC Overrides
            msg += "*₿ BTC*\n"
            msg += f"Total Overrides: {btc_overrides['total']}\n"
            if btc_overrides["total"] > 0:
                msg += f"Avg Quality: {btc_overrides['avg_quality']:.2f}\n"
                if btc_overrides["last_events"]:
                    msg += "Recent Blocks:\n"
                    for event in btc_overrides["last_events"][-3:]:
                        timestamp = event["timestamp"].strftime("%H:%M")
                        msg += f"  {timestamp}: @ ${event['price']:,.2f} (Quality: {event['quality']:.2f})\n"
            msg += "\n"

            # GOLD Overrides
            msg += "*🥇 GOLD*\n"
            msg += f"Total Overrides: {gold_overrides['total']}\n"
            if gold_overrides["total"] > 0:
                msg += f"Avg Quality: {gold_overrides['avg_quality']:.2f}\n"
                if gold_overrides["last_events"]:
                    msg += "Recent Blocks:\n"
                    for event in gold_overrides["last_events"][-3:]:
                        timestamp = event["timestamp"].strftime("%H:%M")
                        msg += f"  {timestamp}: @ ${event['price']:,.2f} (Quality: {event['quality']:.2f})\n"

            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="overrides")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in cmd_overrides: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching override data")

    def _format_signal_entry(self, signal_entry: Dict) -> str:
        """Format a signal entry for display"""
        timestamp = signal_entry["timestamp"].strftime("%H:%M:%S")
        price = signal_entry["price"]
        signal = signal_entry["signal"]
        regime = signal_entry["regime"]
        quality = signal_entry["quality"]
        reasoning = signal_entry["reasoning"].replace("_", " ").title()

        # Signal icon
        if signal == 1:
            signal_icon = "🟢 BUY"
        elif signal == -1:
            signal_icon = "🔴 SELL"
        else:
            signal_icon = "⚪ HOLD"

        # Quality indicator
        quality_icon = "★" if quality >= 0.65 else "•"

        entry = (
            f"  {timestamp} | {signal_icon} | ${price:,.2f}\n"
            f"    {regime} | Quality: {quality_icon} {quality:.2f}\n"
            f"    {reasoning}\n"
        )

        return entry
