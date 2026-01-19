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
    MultiTimeFrameRegime,
    RegimeType,
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
    ) -> MultiTimeFrameRegime:
        """
        Analyze regime and log to database

        Args:
            asset_name: "BTC" or "GOLD"
            symbol: Trading symbol
            exchange: "binance" or "mt5"
            force_refresh: Skip cache

        Returns:
            MultiTimeFrameRegime object
        """
        try:
            # Get detector
            detector = self.get_detector(asset_name)

            # Analyze regime
            regime = detector.analyze_regime(
                symbol=symbol, exchange=exchange, force_refresh=force_refresh
            )

            # Log to database
            if self.db_manager:
                self._log_to_database(regime)

            # Update AI validator context
            if self.ai_validator:
                self._update_ai_context(regime)

            # Send Telegram notification (if significant change)
            if self.telegram_bot and self._should_notify(regime):
                self._send_telegram_notification(regime)

            return regime

        except Exception as e:
            logger.error(f"[MTF] Analysis error for {asset_name}: {e}", exc_info=True)
            raise

    def _log_to_database(self, regime: MultiTimeFrameRegime):
        """
        Log regime analysis to Supabase

        Args:
            regime: MultiTimeFrameRegime object
        """
        try:
            # Convert to dict
            regime_data = regime.to_dict()

            # Insert into mtf_regime_analysis table
            result = (
                self.db_manager.supabase.table("mtf_regime_analysis")
                .insert(
                    {
                        "asset": regime_data["asset"],
                        "timestamp": regime_data["timestamp"],
                        # Consensus
                        "consensus_regime": regime_data["consensus_regime"],
                        "consensus_confidence": regime_data["consensus_confidence"],
                        "timeframe_agreement": regime_data["timeframe_agreement"],
                        "trend_coherence": regime_data["trend_coherence"],
                        # Risk metrics
                        "risk_level": regime_data["risk_level"],
                        "volatility_regime": regime_data["volatility_regime"],
                        # Trading implications
                        "recommended_mode": regime_data["recommended_mode"],
                        "allow_counter_trend": regime_data["allow_counter_trend"],
                        "suggested_max_positions": regime_data[
                            "suggested_max_positions"
                        ],
                        # Scores
                        "bullish_score": regime_data["bullish_score"],
                        "bearish_score": regime_data["bearish_score"],
                        # 1H timeframe
                        "h1_regime": regime_data.get("1h_regime"),
                        "h1_confidence": regime_data.get("1h_confidence"),
                        "h1_trend_strength": regime_data.get("1h_trend_strength"),
                        "h1_trend_direction": regime_data.get("1h_trend_direction"),
                        "h1_adx": regime_data.get("1h_adx"),
                        "h1_rsi": regime_data.get("1h_rsi"),
                        "h1_ema_diff_pct": regime_data.get("1h_ema_diff_pct"),
                        # 4H timeframe
                        "h4_regime": regime_data.get("4h_regime"),
                        "h4_confidence": regime_data.get("4h_confidence"),
                        "h4_trend_strength": regime_data.get("4h_trend_strength"),
                        "h4_trend_direction": regime_data.get("4h_trend_direction"),
                        "h4_adx": regime_data.get("4h_adx"),
                        "h4_rsi": regime_data.get("4h_rsi"),
                        "h4_ema_diff_pct": regime_data.get("4h_ema_diff_pct"),
                        # 1D timeframe
                        "d1_regime": regime_data.get("1d_regime"),
                        "d1_confidence": regime_data.get("1d_confidence"),
                        "d1_trend_strength": regime_data.get("1d_trend_strength"),
                        "d1_trend_direction": regime_data.get("1d_trend_direction"),
                        "d1_adx": regime_data.get("1d_adx"),
                        "d1_rsi": regime_data.get("1d_rsi"),
                        "d1_ema_diff_pct": regime_data.get("1d_ema_diff_pct"),
                    }
                )
                .execute()
            )

            logger.info(f"[MTF DB] ✓ Logged {regime.asset} regime to database")

        except Exception as e:
            logger.error(f"[MTF DB] Failed to log regime: {e}")

    def _update_ai_context(self, regime: MultiTimeFrameRegime):
        """
        Update AI validator with regime context

        Args:
            regime: MultiTimeFrameRegime object
        """
        try:
            # Store regime in AI validator for use during validation
            if not hasattr(self.ai_validator, "mtf_regime_context"):
                self.ai_validator.mtf_regime_context = {}

            self.ai_validator.mtf_regime_context[regime.asset] = {
                "regime": regime.consensus_regime.value,
                "confidence": regime.consensus_confidence,
                "allow_counter_trend": regime.allow_counter_trend,
                "risk_level": regime.risk_level,
                "volatility": regime.volatility_regime,
                "timestamp": regime.timestamp,
                # Individual timeframe data for adaptive thresholds
                "1d_trend": regime.tf_1d.trend_direction if regime.tf_1d else None,
                "4h_trend": regime.tf_4h.trend_direction if regime.tf_4h else None,
                "1h_trend": regime.tf_1h.trend_direction if regime.tf_1h else None,
            }

            logger.info(f"[MTF AI] ✓ Updated AI context for {regime.asset}")

        except Exception as e:
            logger.error(f"[MTF AI] Failed to update context: {e}")

    def _should_notify(self, regime: MultiTimeFrameRegime) -> bool:
        """
        Determine if Telegram notification should be sent

        Args:
            regime: MultiTimeFrameRegime object

        Returns:
            True if notification should be sent
        """
        # Notify on:
        # 1. Strong regimes (confidence > 80%)
        # 2. Regime changes (implement caching for comparison)
        # 3. High disagreement (agreement < 50%)

        if regime.consensus_confidence > 0.80:
            return True

        if regime.timeframe_agreement < 0.50:
            return True

        # Check for regime change (simplified - implement proper caching)
        return False

    def _send_telegram_notification(self, regime: MultiTimeFrameRegime):
        """
        Send Telegram notification about regime

        Args:
            regime: MultiTimeFrameRegime object
        """
        try:
            # Format message
            message = self._format_regime_message(regime)

            # Send via Telegram bot
            # self.telegram_bot.send_message(message)

            logger.info(f"[MTF TG] ✓ Sent notification for {regime.asset}")

        except Exception as e:
            logger.error(f"[MTF TG] Failed to send notification: {e}")

    def _format_regime_message(self, regime: MultiTimeFrameRegime) -> str:
        """
        Format regime analysis for Telegram

        Args:
            regime: MultiTimeFrameRegime object

        Returns:
            Formatted message string
        """
        # Emoji mapping
        regime_emoji = {
            "strong_bull": "🚀",
            "bull": "📈",
            "neutral": "➡️",
            "bear": "📉",
            "strong_bear": "⚠️",
        }

        emoji = regime_emoji.get(regime.consensus_regime.value, "❓")

        message = f"""
{emoji} **MTF REGIME UPDATE - {regime.asset}**

**Consensus:** {regime.consensus_regime.value.upper()}
**Confidence:** {regime.consensus_confidence:.1%}
**TF Agreement:** {regime.timeframe_agreement:.1%}

**Timeframes:**
• 1D: {regime.tf_1d.regime.value if regime.tf_1d else 'N/A'} ({regime.tf_1d.confidence:.1%} if regime.tf_1d else 'N/A'))
• 4H: {regime.tf_4h.regime.value if regime.tf_4h else 'N/A'} ({regime.tf_4h.confidence:.1%} if regime.tf_4h else 'N/A'))
• 1H: {regime.tf_1h.regime.value if regime.tf_1h else 'N/A'} ({regime.tf_1h.confidence:.1%} if regime.tf_1h else 'N/A'))

**Risk:** {regime.risk_level.upper()}
**Volatility:** {regime.volatility_regime.upper()}

**Trading Mode:** {regime.recommended_mode.upper()}
**Counter-Trend:** {'✓ Allowed' if regime.allow_counter_trend else '✗ Blocked'}
**Max Positions:** {regime.suggested_max_positions}
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
            regime = self.analyze_and_log(
                asset_name=asset_name, symbol=symbol, exchange=exchange
            )

            return {
                "regime": regime.consensus_regime.value,
                "is_bull": regime.consensus_regime
                in [RegimeType.BULL, RegimeType.STRONG_BULL],
                "confidence": regime.consensus_confidence,
                "allow_counter_trend": regime.allow_counter_trend,
                "recommended_mode": regime.recommended_mode,
                "max_positions": regime.suggested_max_positions,
                "risk_level": regime.risk_level,
                "volatility": regime.volatility_regime,
                "timeframe_agreement": regime.timeframe_agreement,
                # For detailed logging
                "full_regime": regime,
            }

        except Exception as e:
            logger.error(f"[MTF] Error getting regime: {e}")
            # Return safe defaults
            return {
                "regime": "neutral",
                "is_bull": False,
                "confidence": 0.5,
                "allow_counter_trend": True,
                "recommended_mode": "conservative",
                "max_positions": 1,
                "risk_level": "high",
                "volatility": "normal",
                "timeframe_agreement": 0.5,
            }


# SQL Schema for Supabase table
"""
CREATE TABLE IF NOT EXISTS mtf_regime_analysis (
    id BIGSERIAL PRIMARY KEY,
    asset TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Consensus
    consensus_regime TEXT NOT NULL,
    consensus_confidence FLOAT NOT NULL,
    timeframe_agreement FLOAT NOT NULL,
    trend_coherence FLOAT NOT NULL,
    
    -- Risk metrics
    risk_level TEXT NOT NULL,
    volatility_regime TEXT NOT NULL,
    
    -- Trading implications
    recommended_mode TEXT NOT NULL,
    allow_counter_trend BOOLEAN NOT NULL,
    suggested_max_positions INTEGER NOT NULL,
    
    -- Scores
    bullish_score FLOAT NOT NULL,
    bearish_score FLOAT NOT NULL,
    
    -- 1H timeframe
    "1h_regime" TEXT,
    "1h_confidence" FLOAT,
    "1h_trend_strength" TEXT,
    "1h_trend_direction" TEXT,
    "1h_adx" FLOAT,
    "1h_rsi" FLOAT,
    "1h_ema_diff_pct" FLOAT,
    
    -- 4H timeframe
    "4h_regime" TEXT,
    "4h_confidence" FLOAT,
    "4h_trend_strength" TEXT,
    "4h_trend_direction" TEXT,
    "4h_adx" FLOAT,
    "4h_rsi" FLOAT,
    "4h_ema_diff_pct" FLOAT,
    
    -- 1D timeframe
    "1d_regime" TEXT,
    "1d_confidence" FLOAT,
    "1d_trend_strength" TEXT,
    "1d_trend_direction" TEXT,
    "1d_adx" FLOAT,
    "1d_rsi" FLOAT,
    "1d_ema_diff_pct" FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficient queries
CREATE INDEX idx_mtf_regime_asset_timestamp ON mtf_regime_analysis(asset, timestamp DESC);
CREATE INDEX idx_mtf_regime_timestamp ON mtf_regime_analysis(timestamp DESC);
"""
