import logging
import json
import io
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.portfolio.portfolio_manager import PortfolioManager, Position
from src.global_error_handler import GlobalErrorHandler, ErrorSeverity

def validate_observability():
    print("\n" + "="*60)
    print("👁️ STARTING OBSERVABILITY VALIDATION")
    print("="*60 + "\n")

    # 1. SETUP
    mock_bot = MagicMock()
    mock_bot.send_notification = MagicMock(side_effect=lambda msg, **kwargs: asyncio.Future())
    # Mocking the async future to be completed
    fut = asyncio.Future()
    fut.set_result(True)
    mock_bot.send_notification.return_value = fut
    
    # Also need _current_loop for close_position notification
    mock_bot._current_loop = MagicMock()

    config = {
        "portfolio": {
            "initial_capital": 10000, 
            "max_drawdown": 0.20,
            "target_risk_per_trade": 0.015,
            "max_portfolio_exposure": 5.0,
            "max_total_open_risk": 0.1
        },
        "assets": {"BTC": {"enabled": True, "symbol": "BTCUSDT", "exchange": "binance", "weight": 1.0}},
        "trading": {"mode": "paper"}
    }
    
    pm = PortfolioManager(config=config, telegram_bot=mock_bot)
    error_handler = GlobalErrorHandler(telegram_bot=mock_bot)
    pm.error_handler = error_handler

    # ============================================================================
    # 2. TRIGGER: TRADE ENTRY
    # ============================================================================
    print("📥 Triggering TRADE ENTRY...")
    # Mock OHLC data for VTM (expects numpy arrays)
    import numpy as np
    mock_ohlc = {
        "high": np.array([50100.0]*100), 
        "low": np.array([49900.0]*100), 
        "close": np.array([50000.0]*100)
    }
    
    # Patch the direct import in portfolio_manager.py
    with patch('src.portfolio.portfolio_manager.log_trade_event') as mock_log_event:
        pm.add_position(
            asset="BTC",
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            position_size_usd=1000.0,
            entry_time=datetime.now(),
            ohlc_data=mock_ohlc,
            signal_details={"trade_type": "TREND"}
        )

        # Verify log event was called
        if any("ENTRY" in str(call) for call in mock_log_event.call_args_list):
            print("✅ SUCCESS: Trade Entry event triggered.")
        else:
            print("❌ FAILED: Trade Entry event NOT triggered.")

    # ============================================================================
    # 3. TRIGGER: TRADE EXIT
    # ============================================================================
    print("\n📤 Triggering TRADE EXIT...")
    pos_id = list(pm.positions.keys())[0]
    
    with patch('src.portfolio.portfolio_manager.log_trade_event') as mock_log_event:
        with patch('asyncio.run_coroutine_threadsafe'): # Prevent the error from PM
            pm.close_position(position_id=pos_id, exit_price=51000.0, reason="TP_HIT")

        # Verify log event was called
        if any("EXIT" in str(call) or "TP_HIT" in str(call) or "SL_HIT" in str(call) for call in mock_log_event.call_args_list):
            print("✅ SUCCESS: Trade Exit event triggered.")
        else:
            print("❌ FAILED: Trade Exit event NOT triggered.")
        
    # Verify Telegram notification was called
    if mock_bot.notify_trade_closed.called:
         print("✅ SUCCESS: Telegram notification sent for trade exit.")
    else:
         print("❌ FAILED: Telegram notification NOT sent for trade exit.")

    # ============================================================================
    # 4. TRIGGER: CIRCUIT BREAKER
    # ============================================================================
    print("\n🛑 Triggering CIRCUIT BREAKER (Profit Lock)...")
    pm.peak_equity = 20000
    pm.equity = 18000 # 10% drawdown (trigges PROFIT LOCK layer)
    
    with patch('src.portfolio.portfolio_manager.send_alert') as mock_alert:
        halted, reason = pm.check_circuit_breaker()
        print(f"   Result: Halted={halted}, Reason='{reason}'")
        
        if halted and "PROFIT LOCK" in reason and mock_alert.called:
            print("✅ SUCCESS: Profit Lock triggered and emergency alert sent.")
        else:
            print("❌ FAILED: Circuit breaker observability failed.")

    # ============================================================================
    # 5. TRIGGER: ERROR
    # ============================================================================
    print("\n⚠️ Triggering ERROR...")
    try:
        raise ValueError("Simulated system error")
    except Exception as e:
        error_handler.handle_error(e, component="TEST_SUITE", severity=ErrorSeverity.ERROR)
    
    if mock_bot.send_notification.called:
        print("✅ SUCCESS: Telegram notification sent for Error.")
    else:
        print("❌ FAILED: Telegram notification NOT sent for Error.")

    print("\n" + "="*60)
    print("✨ OBSERVABILITY VALIDATION COMPLETE! ✨")
    print("="*60)

if __name__ == "__main__":
    validate_observability()
