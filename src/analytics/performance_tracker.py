import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks per-strategy results.

    Phase 5.3 (council fix #6): previously only win/loss COUNTS were kept, so the
    council's dynamic weighting could only use win rate — which ignores payoff and
    rewards high-hit-rate / negative-expectancy strategies. record_trade already
    receives the trade P&L; we now retain its magnitude so profit factor and
    expectancy are available. Fully backward-compatible: wins/losses still present.
    """

    def __init__(self):
        # Each bucket keeps counts AND P&L magnitudes.
        self.stats = {
            "TREND": self._empty(),
            "REVERSION": self._empty(),
            "EMA": self._empty(),
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "wins": 0,
            "losses": 0,
            "gross_profit": 0.0,   # sum of winning P&L (positive)
            "gross_loss": 0.0,     # sum of |losing P&L| (positive)
            "pnl_sum": 0.0,        # net P&L
        }

    def record_trade(self, trade_type: str, pnl: float):
        """Record trade result (count + P&L magnitude) for a strategy type."""
        tt_upper = trade_type.upper()
        if tt_upper not in self.stats:
            self.stats[tt_upper] = self._empty()

        s = self.stats[tt_upper]
        try:
            pnl = float(pnl)
        except Exception:
            pnl = 0.0

        if pnl > 0:
            s["wins"] += 1
            s["gross_profit"] += pnl
        else:
            s["losses"] += 1
            s["gross_loss"] += abs(pnl)
        s["pnl_sum"] += pnl

        logger.info(
            f"[PERFORMANCE] Recorded {tt_upper} trade: "
            f"{'WIN' if pnl > 0 else 'LOSS'} (${pnl:,.2f})"
        )

    def get_winrate(self, trade_type: str) -> float:
        """Win rate for a strategy type (0.5 when no data)."""
        s = self.stats.get(trade_type.upper())
        if not s:
            return 0.5
        total = s["wins"] + s["losses"]
        return s["wins"] / total if total > 0 else 0.5

    def get_total_trades(self, trade_type: str) -> int:
        s = self.stats.get(trade_type.upper())
        if not s:
            return 0
        return s["wins"] + s["losses"]

    def get_profit_factor(self, trade_type: str):
        """
        Gross profit / gross loss. Returns None when there isn't enough data to
        be meaningful (no losses yet, or no trades). PF > 1 = net winning.
        """
        s = self.stats.get(trade_type.upper())
        if not s:
            return None
        if s["gross_loss"] <= 0:
            # No losses recorded: PF is undefined/infinite — treat as "no usable
            # signal yet" rather than a free pass.
            return None
        return s["gross_profit"] / s["gross_loss"]

    def get_expectancy(self, trade_type: str):
        """Average net P&L per trade. None when no trades."""
        s = self.stats.get(trade_type.upper())
        if not s:
            return None
        total = s["wins"] + s["losses"]
        if total == 0:
            return None
        return s["pnl_sum"] / total

    def get_all_stats(self) -> dict:
        """Full statistics for all strategies (counts + magnitudes)."""
        return self.stats
