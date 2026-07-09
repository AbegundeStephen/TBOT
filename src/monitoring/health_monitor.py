import time
import logging

logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.start_time = time.time()
        # Part 1.8 (Brain Rebuild): expected minutes-per-cycle per asset,
        # used by check_cadence/check_decision_completeness below.
        self.asset_cadence_baseline_min = {
            "GBPAUD": 1.1, "BTC": 5.3,
            "EURUSD": 5.3, "USTEC": 5.3, "GOLD": 6.0,
            "USOIL": 6.7,
        }

    def heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = time.time()

    def record_error(self):
        """Increment the error count"""
        self.error_count += 1
        logger.warning(f"[HEALTH] Error recorded. Current count: {self.error_count}")

    def is_healthy(self):
        """
        Check if the system is healthy based on:
        1. Last heartbeat within 60 seconds
        2. Error count below 10
        """
        now = time.time()
        heartbeat_ok = (now - self.last_heartbeat) < 60
        errors_ok = self.error_count < 10
        
        healthy = heartbeat_ok and errors_ok
        
        if not healthy:
            if not heartbeat_ok:
                logger.error(f"[HEALTH] System Unhealthy: Heartbeat stale ({now - self.last_heartbeat:.1f}s)")
            if not errors_ok:
                logger.error(f"[HEALTH] System Unhealthy: Too many errors ({self.error_count})")
                
        return healthy

    def get_status(self):
        """Get detailed health status"""
        return {
            "healthy": self.is_healthy(),
            "uptime": time.time() - self.start_time,
            "last_heartbeat_ago": time.time() - self.last_heartbeat,
            "error_count": self.error_count
        }

    # ── Part 1.8 (Brain Rebuild): judge/cadence/completeness diagnostics ──
    # All four methods below take main.py's self.aggregators dict directly.
    # Each per-asset entry is either a plain aggregator (performance mode)
    # or a dict wrapper (hybrid/council mode) — score_history only exists on
    # InstitutionalCouncilAggregator, so the dict form is unwrapped to its
    # "council" entry; performance-mode assets simply report "no_history".

    @staticmethod
    def _unwrap_council(agg_entry):
        return agg_entry.get("council", agg_entry) if isinstance(agg_entry, dict) else agg_entry

    def check_judge_liveness(self, aggregators: dict, window: int = 20) -> dict:
        """
        Per-judge fired/dead/pinned diagnostics from each asset's recent
        score_history. A judge that never fires (DEAD) or always fires at a
        nonzero score (SUSPICIOUSLY_PINNED) over the window is surfaced so a
        silently broken judge doesn't go unnoticed indefinitely.
        """
        report = {}
        for asset, agg_entry in aggregators.items():
            agg = self._unwrap_council(agg_entry)
            history = getattr(agg, "score_history", None)
            if not history:
                report[asset] = {"status": "no_history"}
                continue
            recent = list(history)[-window:]
            judge_names = set()
            for cycle in recent:
                judge_names |= set(cycle.get("buy_scores", {}).keys())
            per_judge = {}
            for j in judge_names:
                fired = sum(
                    1 for c in recent
                    if c.get("buy_scores", {}).get(j, 0) > 0
                    or c.get("sell_scores", {}).get(j, 0) > 0
                )
                pct = fired / len(recent)
                flag = "DEAD" if pct == 0 else "SUSPICIOUSLY_PINNED" if pct > 0.95 else "OK"
                per_judge[j] = {"fired_pct": round(pct, 2), "flag": flag}
            report[asset] = per_judge
        return report

    def check_cadence(self, aggregators: dict) -> dict:
        """Flags assets whose council hasn't produced a new cycle recently."""
        report = {}
        for asset, agg_entry in aggregators.items():
            agg = self._unwrap_council(agg_entry)
            history = getattr(agg, "score_history", None)
            if not history or len(history) < 2:
                continue
            last_ts = history[-1].get("wall_clock_ts")
            baseline_min = self.asset_cadence_baseline_min.get(asset, 5.5)
            if isinstance(last_ts, (int, float)):
                age_min = (time.time() - last_ts) / 60
                report[asset] = {
                    "minutes_since_last_cycle": round(age_min, 1),
                    "baseline_min": baseline_min,
                    "late": age_min > baseline_min * 3,
                }
        return report

    def check_decision_completeness(self, aggregators: dict, window_minutes: int = 60) -> dict:
        """Compares actual vs. expected cycle count over the trailing window."""
        report = {}
        now = time.time()
        for asset, agg_entry in aggregators.items():
            agg = self._unwrap_council(agg_entry)
            history = getattr(agg, "score_history", None)
            if not history:
                continue
            cutoff = now - (window_minutes * 60)
            actual = sum(
                1 for c in history
                if isinstance(c.get("wall_clock_ts"), (int, float)) and c["wall_clock_ts"] >= cutoff
            )
            baseline_min = self.asset_cadence_baseline_min.get(asset, 5.5)
            expected = max(1, int(window_minutes / baseline_min))
            completeness = actual / expected
            report[asset] = {
                "actual_cycles": actual, "expected_cycles": expected,
                "completeness_pct": round(completeness * 100, 1),
                "flag": "GAPS_DETECTED" if completeness < 0.7 else "OK",
            }
        return report

    def check_connections(self, mt5_module, binance_handler, cvd_consumer) -> dict:
        return {
            "mt5_connected": mt5_module.terminal_info() is not None,
            "binance_stale": binance_handler.is_stale() if binance_handler else None,
            "cvd_stale": cvd_consumer.is_stale() if cvd_consumer else None,
        }
