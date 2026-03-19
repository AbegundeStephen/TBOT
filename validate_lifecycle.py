import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LIFECYCLE_TEST")

# Import bot components
from src.portfolio.portfolio_manager import PortfolioManager
from src.execution.binance_handler import BinanceExecutionHandler
from src.utils.trade_logger import log_trade_event
from src.analytics.performance_tracker import PerformanceTracker
from src.monitoring.health_monitor import HealthMonitor

def validate_lifecycle():
    logger.info("🚀 STARTING END-TO-END LIFECYCLE VALIDATION")
    
    # 1. SETUP: Mock Config & Components
    config = {
        "portfolio": {
            "initial_capital": 10000,
            "target_risk_per_trade": 0.01,
            "max_drawdown": 0.20,
            "max_portfolio_exposure": 5.0,
            "max_total_open_risk": 0.1
        },
        "assets": {
            "BTC": {
                "enabled": True,
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "weight": 1.0,
                "fixed_risk_usd": {"TREND": 100},
                "risk": {
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.04,
                    "atr_multiplier": 1.5
                }
            }
        },
        "trading": {
            "mode": "paper",
            "allow_simultaneous_long_short": True,
            "max_positions_per_asset": 3
        }
    }

    # Initialize Portfolio Manager
    pm = PortfolioManager(config=config)
    pm.is_paper_mode = True
    pm.performance_tracker = PerformanceTracker()
    
    # Initialize Health Monitor
    health = HealthMonitor()
    
    logger.info("✅ Components Initialized")

    # 2. SIGNAL GENERATION (Simulated)
    asset = "BTC"
    signal = 1 # BUY
    current_price = 50000.0
    details = {
        "signal_quality": 0.85, # High quality to pass gates
        "trade_type": "TREND",
        "regime": "🚀 BULL"
    }
    
    logger.info(f"📡 [SIGNAL] Generated {asset} BUY at ${current_price}")

    # 3. SAFETY GATE CHECK
    # Check Health
    if not health.is_healthy():
        logger.error("❌ FAILED: Health Monitor unhealthy")
        return
    
    # Check Circuit Breaker
    halted, reason = pm.check_circuit_breaker()
    if halted:
        logger.error(f"❌ FAILED: Circuit Breaker triggered: {reason}")
        return
        
    # Check Quality
    if details["signal_quality"] < 0.65:
        logger.error("❌ FAILED: Quality gate blocked trade")
        return
        
    logger.info("🛡️ [SAFETY] All gates passed")

    # 4. EXECUTION (Simulated add_position)
    # Mock OHLC data for VTM (ATR calculation)
    ohlc = {
        "high": np.array([50100] * 20),
        "low": np.array([49900] * 20),
        "close": np.array([50000] * 20),
        "volume": np.array([100] * 20)
    }
    
    success = pm.add_position(
        asset=asset,
        symbol="BTCUSDT",
        side="long",
        entry_price=current_price,
        position_size_usd=1000.0,
        ohlc_data=ohlc,
        signal_details=details
    )
    
    if not success or not pm.has_position(asset):
        logger.error("❌ FAILED: Position not created in portfolio")
        return
        
    position_id = list(pm.positions.keys())[0]
    logger.info(f"💼 [PORTFOLIO] Position created: {position_id}")

    # 5. MANAGEMENT (Simulate Price Move to TP)
    logger.info("⌛ [MANAGEMENT] Simulating price move to Take Profit...")
    tp_price = 53000.0 # Hits 4% TP
    
    # Update with new bar
    exit_info = pm.positions[position_id].update_with_new_bar(
        high=tp_price, 
        low=tp_price-100, 
        close=tp_price
    )
    
    if exit_info and exit_info.get("reason"):
        logger.info(f"🎯 [VTM] Exit triggered: {exit_info['reason']}")
        
        # 6. CLOSURE
        result = pm.close_position(
            position_id=position_id,
            exit_price=tp_price,
            reason=exit_info["reason"]
        )
        
        if result:
            logger.info("✅ [EXIT] Trade closed successfully")
            
            # Verify Tracker
            wr = pm.performance_tracker.get_winrate("TREND")
            stats = pm.performance_tracker.get_all_stats()["TREND"]
            
            logger.info(f"📊 [STATS] TREND Winrate: {wr:.0%}, Wins: {stats['wins']}, Losses: {stats['losses']}")
            
            if stats['wins'] == 1:
                logger.info("\n" + "="*40)
                logger.info("✨ END-TO-END LIFECYCLE VALIDATED! ✨")
                logger.info("="*40)
                return True
    
    logger.error("❌ FAILED: Lifecycle interrupted or TP not hit")
    return False

if __name__ == "__main__":
    validate_lifecycle()
