import logging

logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.stats = {
            "TREND": {"wins": 0, "losses": 0},
            "REVERSION": {"wins": 0, "losses": 0},
            "EMA": {"wins": 0, "losses": 0}
        }

    def record_trade(self, trade_type: str, pnl: float):
        """Record trade result for a specific strategy type"""
        tt_upper = trade_type.upper()
        if tt_upper not in self.stats:
            self.stats[tt_upper] = {"wins": 0, "losses": 0}
            
        if pnl > 0:
            self.stats[tt_upper]["wins"] += 1
        else:
            self.stats[tt_upper]["losses"] += 1
            
        logger.info(f"[PERFORMANCE] Recorded {tt_upper} trade: {'WIN' if pnl > 0 else 'LOSS'} (${pnl:,.2f})")

    def get_winrate(self, trade_type: str) -> float:
        """Calculate win rate for a specific strategy type"""
        tt_upper = trade_type.upper()
        if tt_upper not in self.stats:
            return 0.5
            
        s = self.stats[tt_upper]
        total = s["wins"] + s["losses"]
        return s["wins"] / total if total > 0 else 0.5

    def get_all_stats(self) -> dict:
        """Return full statistics for all strategies"""
        return self.stats
