import logging
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.portfolio.portfolio_manager import PortfolioManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PROFIT_LOCK_TEST")

def validate_profit_lock():
    logger.info("\n" + "="*60)
    logger.info("🔒 STARTING PROFIT LOCK VALIDATION")
    logger.info("="*60 + "\n")

    # 1. SETUP
    config = {
        "portfolio": {
            "initial_capital": 10000,
            "max_drawdown": 0.20
        },
        "assets": {
            "BTC": {"enabled": True, "symbol": "BTCUSDT", "exchange": "binance"}
        },
        "trading": {"mode": "paper"}
    }
    
    pm = PortfolioManager(config=config)
    pm.is_paper_mode = True
    pm.paper_capital = 10000
    pm.equity = 10000
    pm.peak_equity = 10000
    pm.session_start_equity = 10000

    # ============================================================================
    # PHASE 1: EQUITY GROWTH
    # ============================================================================
    logger.info("📈 PHASE 1: Simulating Equity Growth (10k -> 15k)")
    pm.equity = 15000
    pm.peak_equity = 15000
    
    halted, reason = pm.check_circuit_breaker()
    logger.info(f"   Equity: ${pm.equity:,.2f} | Peak: ${pm.peak_equity:,.2f}")
    logger.info(f"   Halted: {halted}")
    
    if not halted:
        logger.info("✅ SUCCESS: Trading allowed during growth phase.")
    else:
        logger.error(f"❌ FAILED: Trading halted during growth! Reason: {reason}")

    # ============================================================================
    # PHASE 2: TRIGGER DRAWDOWN > 5% FROM PEAK
    # ============================================================================
    # 15,000 * 0.94 = 14,100 (6% drawdown)
    logger.info("\n📉 PHASE 2: Simulating Drawdown > 5% from peak (15k -> 14k)")
    pm.equity = 14000 
    
    # Calculate expected drawdown for log
    drawdown = (pm.peak_equity - pm.equity) / pm.peak_equity
    logger.info(f"   Equity: ${pm.equity:,.2f} | Peak: ${pm.peak_equity:,.2f} | Drawdown: {drawdown:.1%}")

    with patch('src.portfolio.portfolio_manager.send_alert') as mock_alert:
        halted, reason = pm.check_circuit_breaker()
        logger.info(f"   Halted: {halted} | Reason: {reason}")
        
        if halted and "PROFIT LOCK" in reason:
            logger.info("✅ SUCCESS: Profit Lock triggered. Trading paused.")
        else:
            logger.error(f"❌ FAILED: Profit Lock NOT triggered! Reason: {reason}")

    logger.info("\n" + "="*60)
    logger.info("✨ PROFIT LOCK VALIDATION COMPLETE! ✨")
    logger.info("="*60)

if __name__ == "__main__":
    validate_profit_lock()
