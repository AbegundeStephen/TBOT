import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.execution.auto_preset_selector import DynamicPresetSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("REGIME_ALIGNMENT_TEST")

def validate_regime_alignment():
    logger.info("\n" + "="*60)
    logger.info("📐 STARTING REGIME + PRESET ALIGNMENT VALIDATION")
    logger.info("="*60 + "\n")

    # 1. SETUP
    config = {
        "assets": {
            "BTC": {
                "symbol": "BTCUSDT",
                "weight": 1.0,
                "fixed_risk_usd": {"TREND": 100, "SCALP": 50}
            }
        },
        "trading": {"mode": "paper"},
        "aggregator_presets": {
            "BTC": {
                "mr": {"score_threshold": 3.0},
                "conservative": {"score_threshold": 3.5},
                "balanced": {"score_threshold": 3.5},
                "aggressive": {"score_threshold": 3.5}
            }
        }
    }
    
    mock_dm = MagicMock()
    # Disable cooldown for testing
    with patch('src.execution.auto_preset_selector.DynamicPresetSelector._can_switch_preset', return_value=True):
        selector = DynamicPresetSelector(mock_dm, config)
        asset = "BTC"

        # ============================================================================
        # PHASE 1: TREND MARKET (High ADX)
        # ============================================================================
        logger.info("📈 PHASE 1: Simulating TREND Market (ADX = 35)")
        
        metrics_trend = {
            'adx': 35.0,
            'adx_series': pd.Series([35.0]*10),
            'volatility_ratio': 1.2,
            'volume_trend': 10.0,
            'price_momentum': 2.0,
            'current_price': 50000.0,
            'ema_200': 48000.0,
            'trend_direction': 'BULL'
        }

        # Case A: High score should map to 'aggressive' (Trend)
        with patch.object(selector, '_calculate_regime_metrics', return_value=metrics_trend):
            preset_a = selector.update_market_regime(asset, MagicMock())
            logger.info(f"   Metrics (ADX 35, Score High) -> Preset: {preset_a}")
            
            if preset_a in ["balanced", "aggressive"]:
                logger.info("✅ SUCCESS: Trend preset allowed in TREND regime.")
            else:
                logger.error(f"❌ FAILED: Trend preset blocked in TREND regime. Got: {preset_a}")

        # Case B: Low score (MR candidate) should be blocked in TREND
        # Score = 50 (base) + 25 (ADX 30) - 30 (Vol 5.0) - 15 (Mom 0.1) - 15 (Strong Bear) - 8 (Vol Trend -30) = 7
        metrics_forced_mr_in_trend = {
            'adx': 30.0, # TREND regime
            'adx_series': pd.Series([30.0]*10),
            'volatility_ratio': 5.0, # -30
            'price_momentum': 0.1,  # -15
            'current_price': 51000.0,
            'ema_200': 50000.0, 
            'trend_direction': 'BEAR', # -15 (ema_distance 2% > 1.5%)
            'volume_trend': -30.0, # -8
        }
        
        logger.info("🧪 Testing: MR trades blocked in TREND regime")
        with patch.object(selector, '_calculate_regime_metrics', return_value=metrics_forced_mr_in_trend):
             preset_b = selector.update_market_regime(asset, MagicMock())
             logger.info(f"   Metrics (ADX 30, Score 7) -> Preset: {preset_b}")
             if preset_b is None:
                 logger.info("✅ SUCCESS: MR preset correctly blocked in TREND regime.")
             elif preset_b != 'mr':
                 logger.info(f"✅ SUCCESS: MR preset avoided in TREND regime (got {preset_b}).")
             else:
                 logger.error("❌ FAILED: MR preset allowed in TREND regime!")

        # ============================================================================
        # PHASE 2: RANGE MARKET (Low ADX)
        # ============================================================================
        logger.info("\n↔️ PHASE 2: Simulating RANGE Market (ADX = 15)")
        
        metrics_range = {
            'adx': 15.0,
            'adx_series': pd.Series([15.0]*10),
            'volatility_ratio': 1.0, 
            'price_momentum': 0.0, 
            'current_price': 50000.0,
            'ema_200': 50000.0, 
            'trend_direction': 'BULL',
            'volume_trend': 0
        }
        
        # Case C: Low score should map to 'mr' (Mean Reversion)
        # Score: 50 (base) + 15 (Low Vol) - 20 (ADX < 20) - 15 (Stagnant) = 30 -> CONSERVATIVE
        # To get < 20: 
        # Score: 50 (base) + 0 (Mod Vol) - 20 (ADX) - 15 (Stagnant) = 15 -> MR
        metrics_mr_in_range = metrics_range.copy()
        metrics_mr_in_range['volatility_ratio'] = 1.3 # Moderate Vol (0)
        
        with patch.object(selector, '_calculate_regime_metrics', return_value=metrics_mr_in_range):
            preset_c = selector.update_market_regime(asset, MagicMock())
            logger.info(f"   Metrics (Score 15, ADX 15) -> Preset: {preset_c}")
            
            if preset_c == "mr":
                logger.info("✅ SUCCESS: MR preset allowed in RANGE regime.")
            else:
                # Check for Dirty Range Override which also forces MR
                # Needs ADX < 20 for 4 bars, vol [0.45, 1.0], vol_trend <= 5
                logger.error(f"❌ FAILED: MR preset blocked in RANGE regime. Got: {preset_c}")

        # Case D: High score (Trend candidate) should be blocked in RANGE
        # To bypass Dirty Range Override: volatility_ratio > 1.0
        # To get score >= 40 (maps to balanced/aggressive):
        # Base 50 + Mod Vol (0) - 20 (ADX 15) + Strong Mom (+15) + Strong Trend (+15) = 60 (BALANCED)
        metrics_trend_in_range = metrics_range.copy()
        metrics_trend_in_range['volatility_ratio'] = 1.1 # Mod Vol (0)
        metrics_trend_in_range['price_momentum'] = 10.0 # Strong Mom (+15)
        metrics_trend_in_range['current_price'] = 60000.0
        metrics_trend_in_range['ema_200'] = 50000.0 # Strong Bull (+15)
        
        with patch.object(selector, '_calculate_regime_metrics', return_value=metrics_trend_in_range):
            preset_d = selector.update_market_regime(asset, MagicMock())
            logger.info(f"   Metrics (Score 60, ADX 15) -> Preset: {preset_d}")
            
            if preset_d is None:
                logger.info("✅ SUCCESS: Trend preset correctly blocked in RANGE regime.")
            else:
                logger.error(f"❌ FAILED: Trend preset allowed in RANGE regime. Got: {preset_d}")


    logger.info("\n" + "="*60)
    logger.info("✨ REGIME ALIGNMENT VALIDATION COMPLETE! ✨")
    logger.info("="*60)

if __name__ == "__main__":
    validate_regime_alignment()
