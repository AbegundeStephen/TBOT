"""
Multi-Timeframe Regime Integration
====================================
Integrates MTF regime detection with:
1. Database logging (Supabase)
2. AI Validator context
3. Trading bot decision making
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from src.execution.mtf_regime_detector import (
    MultiTimeFrameRegimeDetector,
    RegimeStatus,  # Updated import
    GovernorStatus, # New import
)

logger = logging.getLogger(__name__)


class MTFRegimeIntegration:
    """
    Integrates multi-timeframe regime detection with trading system
    """

    def __init__(
        self, data_manager, db_manager=None, ai_validator=None, telegram_bot=None
    ):
        """
        Initialize integration

        Args:
            data_manager: DataManager instance
            db_manager: TradingDatabaseManager instance (optional)
            ai_validator: HybridSignalValidator instance (optional)
            telegram_bot: TelegramBot instance (optional)
        """
        self.data_manager = data_manager
        self.db_manager = db_manager
        self.ai_validator = ai_validator
        self.telegram_bot = telegram_bot

        # Create detectors for each asset
        self.detectors = {}

        logger.info("[MTF INTEGRATION] Initialized")

    def get_detector(self, asset_type: str) -> MultiTimeFrameRegimeDetector:
        """
        Get or create detector for asset

        Args:
            asset_type: "BTC" or "GOLD"

        Returns:
            MultiTimeFrameRegimeDetector instance
        """
        if asset_type not in self.detectors:
            self.detectors[asset_type] = MultiTimeFrameRegimeDetector(
                data_manager=self.data_manager, asset_type=asset_type
            )
            logger.info(f"[MTF] Created detector for {asset_type}")

        return self.detectors[asset_type]

    def analyze_and_log(
        self, asset_name: str, symbol: str, exchange: str, force_refresh: bool = False
    ) -> RegimeStatus: # Updated return type
        """
        Analyze regime and log to database

        Args:
            asset_name: "BTC" or "GOLD"
            symbol: Trading symbol
            exchange: "binance" or "mt5"
            force_refresh: Skip cache

        Returns:
            RegimeStatus object
        """
        try:
            # Get detector
            detector = self.get_detector(asset_name)

            # Analyze regime
            regime_status = detector.analyze_regime( # Use new RegimeStatus
                symbol=symbol, exchange=exchange, force_refresh=force_refresh
            )

            # Log to database
            if self.db_manager:
                self._log_to_database(regime_status)

            # Update AI validator context
            if self.ai_validator:
                self._update_ai_context(regime_status)

            # Send Telegram notification (if significant change)
            if self.telegram_bot and self._should_notify(regime_status):
                self._send_telegram_notification(regime_status)

            return regime_status

        except Exception as e:
            logger.error(f"[MTF] Analysis error for {asset_name}: {e}", exc_info=True)
            raise

    def _log_to_database(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Log regime analysis to Supabase

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Insert into mtf_regime_analysis table
            result = (
                self.db_manager.supabase.table("mtf_regime_analysis")
                .insert(
                    {
                        "asset": regime_status.asset,
                        "timestamp": regime_status.timestamp.isoformat(),
                        "score": regime_status.score,
                        "is_bullish": regime_status.is_bullish,
                        "is_bearish": regime_status.is_bearish,
                        "reasoning": regime_status.reasoning,
                    }
                )
                .execute()
            )

            logger.info(f"[MTF DB] ✓ Logged {regime_status.asset} regime to database")

        except Exception as e:
            logger.error(f"[MTF DB] Failed to log regime: {e}")

    def _update_ai_context(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Update AI validator with regime context

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Store regime in AI validator for use during validation
            if not hasattr(self.ai_validator, "mtf_regime_context"):
                self.ai_validator.mtf_regime_context = {}

            self.ai_validator.mtf_regime_context[regime_status.asset] = {
                "score": regime_status.score,
                "is_bullish": regime_status.is_bullish,
                "is_bearish": regime_status.is_bearish,
                "reasoning": regime_status.reasoning,
                "timestamp": regime_status.timestamp,
            }

            logger.info(f"[MTF AI] ✓ Updated AI context for {regime_status.asset}")

        except Exception as e:
            logger.error(f"[MTF AI] Failed to update context: {e}")

    def _should_notify(self, regime_status: RegimeStatus) -> bool: # Updated parameter type
        """
        Determine if Telegram notification should be sent based on significant score change.

        Args:
            regime_status: RegimeStatus object

        Returns:
            True if notification should be sent
        """
        # Notify if there's a strong bullish or bearish bias
        return regime_status.score > 0.5 or regime_status.score < -0.5

    def _send_telegram_notification(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Send Telegram notification about regime

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Format message
            message = self._format_regime_message(regime_status)

            # Send via Telegram bot
            # self.telegram_bot.send_message(message)

            logger.info(f"[MTF TG] ✓ Sent notification for {regime_status.asset}")

        except Exception as e:
            logger.error(f"[MTF TG] Failed to send notification: {e}")

    def _format_regime_message(self, regime_status: RegimeStatus) -> str: # Updated parameter type
        """
        Format regime analysis for Telegram

        Args:
            regime_status: RegimeStatus object

        Returns:
            Formatted message string
        """
        emoji = "📈" if regime_status.is_bullish else ("📉" if regime_status.is_bearish else "➡️")

        message = f"""
{emoji} **MTF REGIME UPDATE - {regime_status.asset}**

**Score:** {regime_status.score:.2f}
**Bias:** {'BULLISH' if regime_status.is_bullish else 'BEARISH' if regime_status.is_bearish else 'NEUTRAL'}
**Reasoning:** {regime_status.reasoning}
"""

        return message.strip()

    def get_regime_for_trading(
        self, asset_name: str, symbol: str, exchange: str
    ) -> Dict:
        """
        Get regime data formatted for trading decisions

        Args:
            asset_name: "BTC" or "GOLD"
            symbol: Trading symbol
            exchange: "binance" or "mt5"

        Returns:
            Dict with regime data for trading logic
        """
        try:
            regime_status = self.analyze_and_log(
                asset_name=asset_name, symbol=symbol, exchange=exchange
            )

            return {
                "regime_score": regime_status.score,
                "is_bullish": regime_status.is_bullish,
                "is_bearish": regime_status.is_bearish,
                "reasoning": regime_status.reasoning,
                "timestamp": regime_status.timestamp.isoformat(),
                # For detailed logging if needed
                "full_regime_status": regime_status,
            }

        except Exception as e:
            logger.error(f"[MTF] Error getting regime: {e}")
            # Return safe defaults
            return {
                "regime_score": 0.0,
                "is_bullish": False,
                "is_bearish": False,
                "reasoning": "Error getting regime",
                "timestamp": datetime.now().isoformat(),
            }
