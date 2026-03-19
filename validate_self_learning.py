import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.analytics.performance_tracker import PerformanceTracker
from src.execution.council_aggregator import InstitutionalCouncilAggregator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LEARNING_VALIDATION")

def validate_self_learning():
    logger.info("\n" + "="*60)
    logger.info("🧠 STARTING SELF-LEARNING BEHAVIOR VALIDATION")
    logger.info("="*60 + "\n")

    # 1. SETUP
    tracker = PerformanceTracker()
    
    # Mock strategies with proper return values
    mock_strat = MagicMock()
    mock_strat.generate_signal.return_value = (0, 0.5) # Signal, Confidence
    
    # Mock MTF Governor to allow all trades
    mock_governor = MagicMock()
    mock_governor.get_regime_consensus.return_value = {
        "regime": "NEUTRAL",
        "is_bull": True,
        "is_bear": False,
        "is_ranging": False,
        "score": 0.5,
        "status": {"overall_regime": "NEUTRAL", "is_bull": True}
    }
    
    aggregator = InstitutionalCouncilAggregator(
        mock_strat, mock_strat, mock_strat, 
        asset_type="BTC",
        performance_tracker=tracker,
        mtf_integration=mock_governor
    )
    
    # Mock DataFrame for signal generation
    df = pd.DataFrame({
        "open": [50000.0] * 100,
        "high": [50100.0] * 100,
        "low": [49900.0] * 100,
        "close": [50000.0] * 100,
        "volume": [100.0] * 100
    }, index=pd.date_range(start="2026-03-18 12:00:00", periods=100, freq="h")) # Afternoon to avoid Asian session

    # ============================================================================
    # PHASE 1: REVERSION SUPPRESSION (10 LOSES)
    # ============================================================================
    logger.info("📉 PHASE 1: Simulating 10 consecutive losses for REVERSION...")
    for _ in range(10):
        tracker.record_trade("REVERSION", -100.0) # Loss
    
    wr_rev = tracker.get_winrate("REVERSION")
    logger.info(f"   Current REVERSION Winrate: {wr_rev:.1%}")

    # Mock MTF Regime for passing to aggregator
    mtf_regime = {
        "regime": "NEUTRAL",
        "is_bull": True,
        "is_bear": False,
        "is_ranging": False,
        "score": 0.5,
        "status": {"overall_regime": "NEUTRAL", "is_bull": True}
    }

    # Patch EVERYTHING that could veto
    exp = {'buy': '✅ TEST', 'sell': '✅ TEST'}
    
    with patch.object(aggregator, '_check_macro_regime', return_value="NEUTRAL"), \
         patch.object(aggregator, '_check_governor_filter', return_value=(True, "REVERSION")), \
         patch.object(aggregator, '_check_profit_economics_adaptive', return_value=True), \
         patch.object(aggregator, '_check_sniper_filter', return_value=(True, {})), \
         patch('src.execution.council_aggregator.validate_candle_structure', return_value=True), \
         patch.object(aggregator, '_judge_trend_bidirectional', return_value=(0.0, 0.0, exp)), \
         patch.object(aggregator, '_judge_structure_bidirectional', return_value=(1.5, 1.5, exp)), \
         patch.object(aggregator, '_judge_reversion_bidirectional', return_value=(1.5, 1.5, exp)), \
         patch.object(aggregator, '_judge_volume_bidirectional', return_value=(0.5, 0.5, exp)), \
         patch.object(aggregator, '_detect_breakout_state', return_value=False): # Force Reversion path
        
        signal, details = aggregator.get_aggregated_signal(
            df, 
            current_regime="NEUTRAL", 
            is_bull_market=True, 
            governor_data=mtf_regime
        )
        
        logger.info(f"   Aggregator Result: Signal={signal}, Decision={details.get('decision_type')}")
        
        if signal == 0 and "Circuit Breaker" in details.get('decision_type', ''):
            logger.info("✅ SUCCESS: REVERSION strategy correctly suppressed after 10 losses.")
        else:
            logger.error("❌ FAILED: REVERSION strategy was not suppressed.")

    # ============================================================================
    # PHASE 2: TREND REWARDING (WINNING STREAK)
    # ============================================================================
    logger.info("\n📈 PHASE 2: Simulating winning streak for TREND...")
    for _ in range(5):
        tracker.record_trade("TREND", 200.0) # Win
    
    wr_trend = tracker.get_winrate("TREND")
    logger.info(f"   Current TREND Winrate: {wr_trend:.1%}")

    # Patch judges to return 3.0 base score
    with patch.object(aggregator, '_check_macro_regime', return_value="NEUTRAL"), \
         patch.object(aggregator, '_check_governor_filter', return_value=(True, "TREND")), \
         patch.object(aggregator, '_check_profit_economics_adaptive', return_value=True), \
         patch.object(aggregator, '_check_sniper_filter', return_value=(True, {})), \
         patch('src.execution.council_aggregator.validate_candle_structure', return_value=True), \
         patch.object(aggregator, '_judge_trend_bidirectional', return_value=(1.5, 1.5, exp)), \
         patch.object(aggregator, '_judge_structure_bidirectional', return_value=(1.5, 1.5, exp)), \
         patch.object(aggregator, '_judge_momentum_bidirectional', return_value=(0.0, 0.0, exp)), \
         patch.object(aggregator, '_judge_volume_bidirectional', return_value=(0.0, 0.0, exp)), \
         patch.object(aggregator, '_detect_breakout_state', return_value=True): # Force Trend path
        
        signal, details = aggregator.get_aggregated_signal(
            df, 
            current_regime="NEUTRAL", 
            is_bull_market=True, 
            governor_data=mtf_regime
        )
        
        logger.info(f"   Aggregator Result: Signal={signal}, Score={details.get('total_score')}")
        
        # Base was 3.0. Winrate 100% -> Multiplier 1.5x -> Score 4.5
        if details.get('total_score', 0) > 3.0:
            logger.info(f"✅ SUCCESS: TREND confidence boosted ({details.get('total_score'):.2f} > 3.0 base).")
        else:
            logger.error(f"❌ FAILED: TREND confidence was not boosted.")

    logger.info("\n" + "="*60)
    logger.info("✨ SELF-LEARNING VALIDATION COMPLETE! ✨")
    logger.info("="*60)

if __name__ == "__main__":
    validate_self_learning()
