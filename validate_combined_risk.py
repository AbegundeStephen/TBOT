import logging
import time
import pandas as pd
import numpy as np
from src.portfolio.portfolio_manager import PortfolioManager
from src.analytics.performance_tracker import PerformanceTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("COMBINED_RISK_TEST")

def validate_combined_risk():
    logger.info("\n" + "="*60)
    logger.info("⚖️ STARTING COMBINED RISK SYSTEM VALIDATION")
    logger.info("="*60 + "\n")

    # 1. SETUP: Mock Config
    config = {
        "portfolio": {
            "initial_capital": 10000,
            "target_risk_per_trade": 0.02,
            "max_drawdown": 0.20,
            "max_portfolio_exposure": 5.0,
            "max_total_open_risk": 0.1
        },
        "assets": {
            "BTC": {"enabled": True, "symbol": "BTCUSDT", "exchange": "binance", "weight": 1.0},
            "EURUSD": {"enabled": True, "symbol": "EURUSDm", "exchange": "mt5", "weight": 1.0},
            "GOLD": {"enabled": True, "symbol": "XAUUSDm", "exchange": "mt5", "weight": 1.0}
        },
        "trading": {
            "mode": "paper",
            "allow_simultaneous_long_short": True,
            "max_positions_per_asset": 3
        }
    }

    pm = PortfolioManager(config=config)
    pm.is_paper_mode = True
    pm.performance_tracker = PerformanceTracker()
    
    logger.info("✅ System Initialized (Paper Mode)")

    # ============================================================================
    # PHASE 1: USD CORRELATION SHIELD
    # ============================================================================
    logger.info("\n📉 PHASE 1: Testing USD Correlation Shield")
    
    # 1. Open BTC trade (Baseline)
    # entry_price=50000, stop_loss=49000 (2% dist)
    # risk = 2% of 10k = $200. Size = $200 / 0.02 = $10,000
    size_btc = pm.calculate_position_size("BTC", 50000.0, 49000.0, "BINANCE")
    logger.info(f"   BTC Base Size: ${size_btc:,.2f}")
    
    pm.add_position("BTC", "BTCUSDT", "long", 50000.0, size_btc)
    logger.info("   ✓ BTC Position Opened")

    # 2. Trigger EURUSD trade (Correlated asset)
    # entry_price=1.10, stop_loss=1.089 ( ~1% dist)
    # Expected: Size should be reduced by 50%
    size_eur = pm.calculate_position_size("EURUSD", 1.10, 1.089, "MT5")
    logger.info(f"   EURUSD Size (Shield Active): ${size_eur:,.2f}")
    
    # Simple check: size_eur should be significantly smaller than if no shield
    # Max size for 1% risk on 10k is $20,000. With 50% reduction = $10,000.
    if size_eur <= 10500: # Allowing for some rounding/margin room
        logger.info("✅ SUCCESS: Correlation shield reduced EURUSD position size.")
    else:
        logger.error(f"❌ FAILED: EURUSD size not reduced correctly. Size: {size_eur}")

    # ============================================================================
    # PHASE 2: COMBINED LOSS STREAK HALT
    # ============================================================================
    logger.info("\n🚫 PHASE 2: Testing Consecutive Loss Streak Halt")
    
    # Close existing positions as losses
    pids = list(pm.positions.keys())
    for pid in pids:
        pm.close_position(pid, exit_price=pm.positions[pid].entry_price * 0.95, reason="manual_loss")
    
    logger.info(f"   Current Loss Streak: {pm.loss_streak}")
    
    # Simulate one more loss to hit 3
    # Manually increment since we close everything at once above
    while pm.loss_streak < 3:
        pm.loss_streak += 1
        logger.info(f"   Simulating loss... Streak: {pm.loss_streak}")

    # Now attempt a new trade
    halted, reason = pm.check_circuit_breaker()
    logger.info(f"   Circuit Breaker Result: Halted={halted}, Reason='{reason}'")
    
    if halted and "streak" in reason:
        logger.info("✅ SUCCESS: System halted trading after consecutive losses.")
    else:
        logger.error("❌ FAILED: System allowed trading despite loss streak.")

    logger.info("\n" + "="*60)
    logger.info("✨ COMBINED RISK VALIDATION COMPLETE! ✨")
    logger.info("="*60)

if __name__ == "__main__":
    validate_combined_risk()
