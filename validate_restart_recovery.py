import logging
import os
import pickle
from datetime import datetime
from unittest.mock import MagicMock, patch
from src.portfolio.portfolio_manager import PortfolioManager, Position
from src.utils.state_manager import STATE_FILE

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("RESTART_RECOVERY_TEST")

def validate_restart_recovery():
    logger.info("\n" + "="*60)
    logger.info("🔄 STARTING RESTART RECOVERY VALIDATION")
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
        "trading": {"mode": "live"} # Set to live to enable state saving logic
    }
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Cleanup previous state if any
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    
    pm = PortfolioManager(config=config)
    pm.state_file = MagicMock()
    pm.state_file.exists.return_value = True
    pm.state_file.with_suffix.return_value = MagicMock()
    pm.state_file.parent = MagicMock()
    
    # Mock the actual state file path for pickle
    REAL_STATE_FILE = "data/portfolio_state_test.pkl"
    if os.path.exists(REAL_STATE_FILE):
        os.remove(REAL_STATE_FILE)
    
    # We will manually patch the state_file property or just use the real one for the test
    pm.state_file = MagicMock()
    pm.state_file.exists.side_effect = lambda: os.path.exists(REAL_STATE_FILE)
    pm.state_file.with_suffix.return_value = MagicMock()
    pm.state_file.with_suffix.return_value.rename = lambda new_name: os.replace(REAL_STATE_FILE + ".tmp", REAL_STATE_FILE)
    pm.state_file.with_suffix.return_value.exists.side_effect = lambda: os.path.exists(REAL_STATE_FILE + ".tmp")
    pm.state_file.with_suffix.return_value.unlink = lambda: os.remove(REAL_STATE_FILE + ".tmp") if os.path.exists(REAL_STATE_FILE + ".tmp") else None
    
    # Simplified approach: Patch the open call and Path in the PortfolioManager for this test
    
    # Let's set some state
    pm.is_paper_mode = False # Enable saving
    pm.peak_equity = 15000
    pm.loss_streak = 2
    pm.realized_pnl_today = 500.0
    
    # Create a dummy position
    pos = Position(
        asset="BTC",
        symbol="BTCUSDT",
        side="long",
        entry_price=50000.0,
        quantity=0.1,
        entry_time=datetime.now(),
        risk_config={}, # Added missing arg
        use_dynamic_management=False
    )
    pm.positions = {pos.position_id: pos}
    
    logger.info(f"📍 Initial State:")
    logger.info(f"   Positions:   {len(pm.positions)} ({list(pm.positions.keys())[0]})")
    logger.info(f"   Peak Equity: ${pm.peak_equity:,.2f}")
    logger.info(f"   Loss Streak: {pm.loss_streak}")
    logger.info(f"   Today's P&L: ${pm.realized_pnl_today:,.2f}")

    # 2. SAVE STATE
    logger.info("\n💾 Saving state...")
    
    # Patch the save_portfolio_state to use our real test file instead of MagicMock
    with patch.object(pm, 'state_file') as mock_path:
        from pathlib import Path
        mock_path.with_suffix.return_value = Path(REAL_STATE_FILE + ".tmp")
        mock_path.parent = Path("data")
        # We need to trick the rename to work
        # Actually let's just use a real Path object for pm.state_file
        pm.state_file = Path(REAL_STATE_FILE)
        pm.save_portfolio_state()
    
    # 3. RESTART (New Instance)
    logger.info("\n🔄 Simulating Restart...")
    pm_new = PortfolioManager(config=config)
    pm_new.is_paper_mode = False
    pm_new.state_file = Path(REAL_STATE_FILE)
    
    # Confirm it's empty
    assert len(pm_new.positions) == 0
    assert pm_new.peak_equity == 0
    
    # 4. LOAD STATE
    logger.info("📂 Loading state...")
    mock_dm = MagicMock()
    # We need to mock fetch_binance_data/fetch_mt5_data if load_portfolio_state calls them
    # Based on previous read, it does call them to re-init VTM
    
    with patch('src.portfolio.portfolio_manager.Position.update_with_new_bar', return_value=None):
        pm_new.load_portfolio_state(mock_dm)
    
    # 5. VERIFY
    logger.info("\n📍 Restored State:")
    logger.info(f"   Positions:   {len(pm_new.positions)}")
    if len(pm_new.positions) > 0:
        logger.info(f"   Position ID: {list(pm_new.positions.keys())[0]}")
    logger.info(f"   Peak Equity: ${pm_new.peak_equity:,.2f}")
    logger.info(f"   Loss Streak: {pm_new.loss_streak}")
    logger.info(f"   Today's P&L: ${pm_new.realized_pnl_today:,.2f}")
    
    success = True
    if len(pm_new.positions) != 1: success = False
    if pm_new.peak_equity != 15000: success = False
    if pm_new.loss_streak != 2: success = False
    if pm_new.realized_pnl_today != 500.0: success = False
    
    if success:
        logger.info("\n✅ SUCCESS: All state components restored correctly.")
    else:
        logger.error("\n❌ FAILED: State restoration mismatch.")
        
    # Cleanup
    if os.path.exists(REAL_STATE_FILE):
        os.remove(REAL_STATE_FILE)
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    logger.info("\n" + "="*60)
    logger.info("✨ RESTART RECOVERY VALIDATION COMPLETE! ✨")
    logger.info("="*60)

if __name__ == "__main__":
    validate_restart_recovery()
