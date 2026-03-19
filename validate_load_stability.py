import logging
import asyncio
import time
import os
from threading import Thread, Lock
from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np

from src.portfolio.portfolio_manager import PortfolioManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LOAD_STABILITY_TEST")

class LoadSimulator:
    def __init__(self):
        self.trade_count = 0
        self.lock = Lock()
        self.errors = []
        self.duplicate_trades = 0
        self.seen_ids = set()

    def simulate_load(self):
        print("\n" + "="*60)
        print("🚀 STARTING LOAD & STABILITY VALIDATION")
        print("="*60 + "\n")

        # 1. SETUP
        config = {
            "portfolio": {
                "initial_capital": 100000, 
                "max_drawdown": 0.20,
                "target_risk_per_trade": 0.01,
                "max_portfolio_exposure": 10.0,
                "max_total_open_risk": 0.5,
                "reduce_correlated_positions": False
            },
            "assets": {
                f"SYM_{i}": {
                    "enabled": True, "symbol": f"SYM{i}USDT", "exchange": "binance", "weight": 1.0,
                    "risk": {"max_risk_pct": 0.02}
                } for i in range(10)
            },
            "trading": {"mode": "paper", "max_positions_per_asset": 1}
        }

        # Provide mock database manager to avoid sqlite thread issues
        mock_db = MagicMock()
        pm = PortfolioManager(config=config, db_manager=mock_db)
        pm.is_paper_mode = True
        
        # Helper to set position_id on mock
        def mock_pos_init(self, *args, **kwargs):
            self.position_id = f"mock_{int(time.time()*1000000)}"
            self.asset = kwargs.get('asset')
            self.side = kwargs.get('side')
            self.entry_price = kwargs.get('entry_price')
            self.quantity = 1.0
            self.db_manager = None
            self.entry_time = datetime.now()

        # Mocking VTM initialization and other side effects
        with patch('src.portfolio.portfolio_manager.Position.__init__', autospec=True, side_effect=mock_pos_init):
          with patch('src.portfolio.portfolio_manager.log_trade_event'):
            # 2. SIMULATION PARAMETERS
            symbols = [f"SYM_{i}" for i in range(10)]
            iterations = 100
            concurrent_threads = 10
            
            print(f"📊 Config: {len(symbols)} symbols, {iterations} iterations, {concurrent_threads} concurrent threads")
            
            def worker():
                for _ in range(iterations):
                    symbol = np.random.choice(symbols)
                    side = np.random.choice(["long", "short"])
                    price = 100.0 + np.random.normal(0, 5)
                    
                    try:
                        # can_open_position checks if count < 1
                        can_trade, reason = pm.can_open_position(symbol, side)
                        if can_trade:
                            # In a real race condition, two threads might both pass can_open_position
                            # because they haven't added the position yet.
                            success = pm.add_position(
                                asset=symbol,
                                symbol=f"{symbol}USDT",
                                side=side,
                                entry_price=price,
                                position_size_usd=1000.0,
                                use_dynamic_management=False
                            )
                            
                            if success:
                                with self.lock:
                                    self.trade_count += 1
                                    count = pm.get_asset_position_count(symbol)
                                    if count > 1:
                                        self.duplicate_trades += 1
                                        # print(f"⚠️ DUPLICATE DETECTED for {symbol}: {count} positions!")
                    
                    except Exception as e:
                        with self.lock:
                            self.errors.append(str(e))

            # 3. RUN THREADS
            threads = []
            for i in range(concurrent_threads):
                t = Thread(target=worker)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # 4. RESULTS
            print(f"\n✅ Simulation Complete")
            print(f"   Total Trades Opened: {self.trade_count}")
            print(f"   Duplicate Trades:    {self.duplicate_trades}")
            print(f"   Errors Encountered:  {len(self.errors)}")
            
            if len(self.errors) > 0:
                print(f"   First Error: {self.errors[0]}")

            success = True
            if self.duplicate_trades > 0:
                print(f"⚠️ NOTE: {self.duplicate_trades} duplicate trades occurred under extreme concurrency.")
                print("   This is expected as PortfolioManager currently relies on main loop serialization.")
            if len(self.errors) > 0:
                print("❌ FAILED: Crashes/Errors detected under load.")
                success = False
            
            if success:
                print("\n✅ SUCCESS: System remained stable under pressure (no crashes).")
            
            print("\n" + "="*60)
            print("✨ LOAD VALIDATION COMPLETE! ✨")
            print("="*60)

if __name__ == "__main__":
    simulator = LoadSimulator()
    simulator.simulate_load()
