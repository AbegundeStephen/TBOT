#!/usr/bin/env python3
"""
Enhanced Telegram Bot Debug Module
Tests Telegram bot connectivity and command handling
"""

import logging
import asyncio
from telegram_config import TELEGRAM_CONFIG
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test command handler"""
    await update.message.reply_text("✅ Bot is working! Commands are being received.")
    logger.info(f"Test command received from {update.effective_user.id}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test status command"""
    await update.message.reply_text(
        "📊 *Bot Status Test*\n\n"
        "✅ Telegram bot is responsive\n"
        "✅ Commands are working\n"
        "✅ Connection is stable",
        parse_mode='Markdown'
    )


async def main():
    """Test Telegram bot connectivity"""
    logger.info("=" * 70)
    logger.info("TELEGRAM BOT DEBUG TEST")
    logger.info("=" * 70)
    
    # Check config
    if not TELEGRAM_CONFIG.get("enabled"):
        logger.error("❌ Telegram is disabled in config")
        return
    
    token = TELEGRAM_CONFIG.get("bot_token")
    admin_ids = TELEGRAM_CONFIG.get("admin_ids", [])
    
    if not token:
        logger.error("❌ No bot token found")
        return
    
    if not admin_ids:
        logger.error("❌ No admin IDs configured")
        return
    
    logger.info(f"✅ Bot token found: {token[:10]}...")
    logger.info(f"✅ Admin IDs: {admin_ids}")
    
    # Build application
    logger.info("\n📡 Building Telegram application...")
    application = Application.builder().token(token).build()
    
    # Add test handlers
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(CommandHandler("status", status_command))
    
    logger.info("✅ Handlers registered")
    
    # Initialize
    logger.info("\n🚀 Initializing bot...")
    await application.initialize()
    logger.info("✅ Bot initialized")
    
    # Start
    logger.info("\n🟢 Starting bot...")
    await application.start()
    logger.info("✅ Bot started")
    
    # Get bot info
    bot_info = await application.bot.get_me()
    logger.info(f"\n🤖 Bot Info:")
    logger.info(f"   Name: {bot_info.first_name}")
    logger.info(f"   Username: @{bot_info.username}")
    logger.info(f"   ID: {bot_info.id}")
    
    # Send test message to admins
    logger.info("\n📤 Sending test message to admins...")
    for admin_id in admin_ids:
        try:
            await application.bot.send_message(
                chat_id=admin_id,
                text=(
                    "🧪 *Debug Test Message*\n\n"
                    "✅ Bot is running\n"
                    "✅ Connection successful\n\n"
                    "Try these commands:\n"
                    "/test - Test command\n"
                    "/status - Status test"
                ),
                parse_mode='Markdown'
            )
            logger.info(f"✅ Test message sent to {admin_id}")
        except Exception as e:
            logger.error(f"❌ Failed to send to {admin_id}: {e}")
    
    # Start polling
    logger.info("\n🔄 Starting update polling...")
    logger.info("Bot is now running. Send /test or /status to test.")
    logger.info("Press Ctrl+C to stop\n")
    
    try:
        # This is the key part - keeps the bot alive and responsive
        await application.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopping bot...")
    finally:
        logger.info("🛑 Shutting down...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Debug test stopped")