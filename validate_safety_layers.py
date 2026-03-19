import logging
import time
import unittest
from unittest.mock import MagicMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SAFETY_VALIDATION")

# Import components
from src.portfolio.portfolio_manager import PortfolioManager
from src.monitoring.health_monitor import HealthMonitor

class TestSafetyLayers(unittest.TestCase):
    def setUp(self):
        # Mock Config
        self.config = {
            "portfolio": {
                "initial_capital": 10000,
                "target_risk_per_trade": 0.015,
                "max_drawdown": 0.20,
                "max_portfolio_exposure": 5.0,
                "max_total_open_risk": 0.1
            },
            "assets": {
                "BTC": {"enabled": True, "symbol": "BTCUSDT", "exchange": "binance", "fixed_risk_usd": {"TREND": 100}}
            },
            "trading": {
                "mode": "paper",
                "min_signal_quality": 0.65 # Hard threshold for test
            }
        }
        
        self.pm = PortfolioManager(config=self.config)
        self.pm.is_paper_mode = True
        self.health = HealthMonitor()
        
    def test_gate_1_system_unhealthy(self):
        logger.info("🧪 Testing Gate 1: System Health Monitor")
        # Force unhealthy state (stale heartbeat)
        self.health.last_heartbeat = time.time() - 120 
        
        # Simulation of logic in main.py
        is_healthy = self.health.is_healthy()
        logger.info(f"   Result: Healthy={is_healthy}")
        
        self.assertFalse(is_healthy, "System should be unhealthy due to stale heartbeat")
        logger.info("✅ GATE 1 PASSED: Unhealthy system correctly detected.")

    def test_gate_2_circuit_breaker_loss_streak(self):
        logger.info("🧪 Testing Gate 2: Consecutive Loss Circuit Breaker")
        # Force loss streak
        self.pm.loss_streak = 3
        
        # Check circuit breaker
        halted, reason = self.pm.check_circuit_breaker()
        logger.info(f"   Result: Halted={halted}, Reason='{reason}'")
        
        self.assertTrue(halted)
        self.assertIn("streak", reason)
        logger.info("✅ GATE 2 PASSED: Loss streak correctly halted trading.")

    def test_gate_3_circuit_breaker_drawdown(self):
        logger.info("🧪 Testing Gate 3: Max Drawdown Circuit Breaker")
        # Force drawdown (Peak 10k, Current 7k = 30% DD)
        self.pm.peak_equity = 10000
        self.pm.equity = 7000
        
        # Check circuit breaker
        halted, reason = self.pm.check_circuit_breaker()
        logger.info(f"   Result: Halted={halted}, Reason='{reason}'")
        
        self.assertTrue(halted)
        self.assertIn("Drawdown", reason)
        logger.info("✅ GATE 3 PASSED: Drawdown correctly halted trading.")

    def test_gate_4_quality_threshold(self):
        logger.info("🧪 Testing Gate 4: Quality/Confidence Gate")
        # Signal details with low quality
        details = {"signal_quality": 0.45}
        min_quality = self.config["trading"]["min_signal_quality"]
        
        # Logic from main.py
        signal_quality = details.get("signal_quality", 0)
        blocked = signal_quality < min_quality
        
        logger.info(f"   Result: Signal Quality={signal_quality}, Threshold={min_quality}, Blocked={blocked}")
        
        self.assertTrue(blocked)
        logger.info("✅ GATE 4 PASSED: Low quality signal correctly blocked.")

if __name__ == "__main__":
    logger.info("\n" + "="*50)
    logger.info("🔍 STARTING CROSS-LAYER SAFETY VALIDATION")
    logger.info("="*50 + "\n")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSafetyLayers)
    unittest.TextTestRunner(verbosity=1).run(suite)
