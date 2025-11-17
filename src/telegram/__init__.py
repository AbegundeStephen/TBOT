#!/usr/bin/env python3
"""
Fixed Telegram Bot Interface for Trading Bot
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

logger = logging.getLogger(__name__)


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
    Handles notifications and user commands
    """

    def __init__(self, token: str, admin_ids: List[int], trading_bot):
        """
        Initialize Telegram bot

        Args:
            token: Telegram bot token
            admin_ids: List of authorized Telegram user IDs
            trading_bot: Reference to main trading bot instance
        """
        self.token = token
        self.admin_ids = admin_ids
        self.trading_bot = trading_bot
        self.application = None
        self.is_running = False

        self._is_ready = False
        self._init_lock = asyncio.Lock()
        self._message_queue = []  # Queue for messages during init

        # Track notification counts to avoid spam
        self.notification_counts = {}
        self.last_daily_summary = None

        logger.info(f"TelegramBot initialized - Admins: {admin_ids}")

    async def initialize(self):
            """Initialize with proper error handling and verification"""
            try:
                async with self._init_lock:
                    logger.info("[TELEGRAM] Building application...")
                    self.application = (
                        Application.builder()
                        .token(self.token)
                        .connect_timeout(30)
                        .read_timeout(30)
                        .write_timeout(30)
                        .pool_timeout(30)
                        .get_updates_read_timeout(30)
                        .build()
                    )

                    # Register handlers
                    logger.info("[TELEGRAM] Registering command handlers...")
                    self._register_handlers()

                    logger.info("[TELEGRAM] Initializing application...")
                    await self.application.initialize()

                    logger.info("[TELEGRAM] Starting application...")
                    await self.application.start()

                    logger.info("[TELEGRAM] Starting update polling...")
                    await self.application.updater.start_polling(
                        poll_interval=1.0,
                        timeout=30,
                        drop_pending_updates=True,
                        allowed_updates=Update.ALL_TYPES,
                        bootstrap_retries=5,
                    )

                    # CRITICAL: Wait for polling to actually start
                    await asyncio.sleep(2)

                    # Verify bot is responsive
                    if await self._verify_bot_ready():
                        self._is_ready = True
                        self.is_running = True
                        logger.info("[TELEGRAM] ✅ Bot fully operational")

                        # Send startup notification
                        await self.send_notification(
                            "🤖 *Trading Bot Started*\n\n"
                            f"Bot initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            "Type /help for available commands"
                        )

                        # Process queued messages
                        await self._process_message_queue()
                    else:
                        logger.error("[TELEGRAM] ❌ Bot verification failed")
                        raise Exception("Bot failed to become ready")

            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}", exc_info=True)
                self._is_ready = False
                raise

    async def _verify_bot_ready(self) -> bool:
            """Verify bot can send messages"""
            try:
                # Try to get bot info
                bot_info = await self.application.bot.get_me()
                logger.info(f"[TELEGRAM] Bot verified: @{bot_info.username}")
                return True
            except Exception as e:
                logger.error(f"[TELEGRAM] Verification failed: {e}")
                return False

    async def _process_message_queue(self):
            """Send any queued messages from before initialization"""
            if not self._message_queue:
                return

            logger.info(
                f"[TELEGRAM] Processing {len(self._message_queue)} queued messages"
            )
            for msg in self._message_queue:
                await self.send_notification(msg, disable_preview=True)
            self._message_queue.clear()

    def _register_handlers(self):
            """Register all command handlers"""
            self.application.add_handler(CommandHandler("start", self.cmd_start))
            self.application.add_handler(CommandHandler("help", self.cmd_help))
            self.application.add_handler(CommandHandler("status", self.cmd_status))
            self.application.add_handler(
                CommandHandler("positions", self.cmd_positions)
            )
            self.application.add_handler(CommandHandler("history", self.cmd_history))
            self.application.add_handler(
                CommandHandler("performance", self.cmd_performance)
            )
            self.application.add_handler(
                CommandHandler("start_trading", self.cmd_start_trading)
            )
            self.application.add_handler(
                CommandHandler("stop_trading", self.cmd_stop_trading)
            )
            self.application.add_handler(
                CommandHandler("close_all", self.cmd_close_all)
            )
            self.application.add_handler(CommandHandler("close", self.cmd_close_asset))
            self.application.add_handler(CallbackQueryHandler(self.button_callback))

    async def run_polling(self):
        """Keep the bot running with polling - FIXED VERSION"""
        try:
            logger.info("[TELEGRAM] Starting polling loop...")

            # The bot will keep running as long as this doesn't exit
            while self.is_running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("[TELEGRAM] Polling cancelled")
        except Exception as e:
            logger.error(f"[TELEGRAM] Error in polling loop: {e}", exc_info=True)

    async def shutdown(self):
        """Shutdown the bot gracefully"""
        try:
            self.is_running = False

            await self.send_notification(
                "🛑 *Trading Bot Stopped*\n\n"
                f"Bot shutdown at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if self.application:
                logger.info("[TELEGRAM] Stopping updater...")
                if self.application.updater and self.application.updater.running:
                    await self.application.updater.stop()

                logger.info("[TELEGRAM] Stopping application...")
                await self.application.stop()

                logger.info("[TELEGRAM] Shutting down application...")
                await self.application.shutdown()

            logger.info("[TELEGRAM] Shutdown complete")
        except Exception as e:
            logger.error(f"Error during Telegram bot shutdown: {e}")

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
                f"📈 Open Positions: {open_positions}\n"
                f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n\n"
                f"*Market Status:*\n"
                f"₿ BTC: {btc_status}\n"
                f"🥇 GOLD: {gold_status}\n\n"
                f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"
            )

            # Add inline keyboard for quick actions
            keyboard = [
                [
                    InlineKeyboardButton("📊 Positions", callback_data="positions"),
                    InlineKeyboardButton("📜 History", callback_data="history"),
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

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()
            positions = portfolio_status.get("positions", {})

            if not positions:
                await update.message.reply_text(
                    "📭 *No Open Positions*\n\n" "Currently no active trades.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            positions_msg = "📊 *Open Positions*\n\n"

            for asset, pos in positions.items():
                side = pos["side"].upper()
                side_icon = "🟢" if side == "LONG" else "🔴"

                entry_price = pos["entry_price"]
                current_value = pos["current_value"]
                pnl = pos["pnl"]
                pnl_pct = (pnl / current_value * 100) if current_value > 0 else 0

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                positions_msg += (
                    f"{side_icon} *{asset} - {side}*\n"
                    f"Entry: ${entry_price:,.2f}\n"
                    f"Size: ${current_value:,.2f}\n"
                    f"{pnl_icon} P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                )

                if pos.get("stop_loss"):
                    positions_msg += f"🛑 SL: ${pos['stop_loss']:,.2f}\n"
                if pos.get("take_profit"):
                    positions_msg += f"🎯 TP: ${pos['take_profit']:,.2f}\n"

                positions_msg += "\n"

            positions_msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            await update.message.reply_text(
                positions_msg, parse_mode=ParseMode.MARKDOWN
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

    # ==================== NOTIFICATION METHODS ====================

    async def send_notification(self, message: str, disable_preview: bool = True):
        """Send notification with queuing support"""
        # If not ready, queue the message
        if not self._is_ready:
            logger.warning("[TELEGRAM] Bot not ready, queuing message")
            self._message_queue.append(message)
            return

        if not self.is_running or not self.application:
            logger.warning("[TELEGRAM] Bot not running, cannot send notification")
            return

        success_count = 0
        for admin_id in self.admin_ids:
            try:
                await self.application.bot.send_message(
                    chat_id=admin_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=disable_preview,
                )
                success_count += 1
                logger.debug(f"[TELEGRAM] Notification sent to {admin_id}")
            except Exception as e:
                logger.error(f"[TELEGRAM] Failed to send to {admin_id}: {e}")

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
        """Send end-of-day performance summary"""
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
                f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n\n"
                f"📊 Today's Trades: {len(today_trades)}\n"
                f"🟢 Winning: {winning_today}\n"
                f"🔴 Losing: {losing_today}\n"
            )

            await self.send_notification(msg)
            self.last_daily_summary = datetime.now()

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
                f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n\n"
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
        """Send positions message (for callback)"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()
            positions = portfolio_status.get("positions", {})

            if not positions:
                await query.edit_message_text(
                    "📭 *No Open Positions*\n\n" "Currently no active trades.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            positions_msg = "📊 *Open Positions*\n\n"

            for asset, pos in positions.items():
                side = pos["side"].upper()
                side_icon = "🟢" if side == "LONG" else "🔴"

                entry_price = pos["entry_price"]
                current_value = pos["current_value"]
                pnl = pos["pnl"]
                pnl_pct = (pnl / current_value * 100) if current_value > 0 else 0

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                positions_msg += (
                    f"{side_icon} *{asset} - {side}*\n"
                    f"Entry: ${entry_price:,.2f}\n"
                    f"Size: ${current_value:,.2f}\n"
                    f"{pnl_icon} P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                )

                if pos.get("stop_loss"):
                    positions_msg += f"🛑 SL: ${pos['stop_loss']:,.2f}\n"
                if pos.get("take_profit"):
                    positions_msg += f"🎯 TP: ${pos['take_profit']:,.2f}\n"

                positions_msg += "\n"

            positions_msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            await query.edit_message_text(positions_msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in _send_positions_message: {e}")

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
