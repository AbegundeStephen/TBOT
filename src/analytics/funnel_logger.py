"""
Signal Funnel + AI-Filter A/B Logger  (Remediation Phase 2.1 / 2.2)
====================================================================
Purpose: make the signal pipeline OBSERVABLE before any alpha tuning.

For every aggregator evaluation we record which stage a potential trade
reached — generated → raw signal → survived the veto stack → passed to
execution → actually executed — so we can finally SEE how many real
opportunities each veto is killing (Audit §5). It also logs every signal the
AI validator rejected, with the would-be entry, so the AI filter's marginal
value can be measured instead of assumed (Audit §12.3 / §5).

This module is PURELY observational — it never changes a trading decision.
Output: append-only JSONL under logs/funnel/ plus a periodic INFO summary.
"""

import os
import json
import logging
from datetime import datetime, timezone
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class FunnelLogger:
    def __init__(self, log_dir: str = "logs/funnel", summary_every: int = 50):
        self.log_dir = log_dir
        self.summary_every = max(1, int(summary_every))
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"[FUNNEL] Could not create log dir {self.log_dir}: {e}")
        self._lock = Lock()
        # counters keyed by (utc_date, asset) -> {stage: count}
        self._counts = defaultdict(lambda: defaultdict(int))
        self._since_summary = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _append(self, day: str, kind: str, rec: dict) -> None:
        try:
            path = os.path.join(self.log_dir, f"{kind}_{day}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            logger.debug(f"[FUNNEL] append failed ({kind}): {e}")

    @staticmethod
    def _raw_fired(details: dict) -> bool:
        return any(
            details.get(k, 0) for k in ("mr_signal", "tf_signal", "ema_signal")
        )

    def _classify_stage(self, signal: int, details: dict) -> str:
        """Bucket the evaluation outcome into one funnel stage."""
        if signal != 0:
            return "passed_to_execution"

        # Ground-truth AI rejection first: the validator actively changed a real
        # signal to a hold. We do NOT use `validation_passed` (several paths set
        # it to simply `signal != 0`, which would mislabel every block as AI).
        if details.get("ai_modified"):
            return "blocked_ai_validation"

        reason = str(details.get("reasoning", "")).lower()
        # Map known veto reasons (set across both aggregators) to families.
        # Order matters — first match wins. "quality" precedes "score"/"rejected"
        # so "REJECTED (Quality: ...)" lands in low_quality, while
        # "REJECTED (Score: ...)" lands in low_score (a consensus-threshold miss,
        # NOT an AI rejection).
        families = {
            "stale_price": "blocked_stale_price",
            "flash_veto": "blocked_flash",
            "ny_open": "blocked_ny_open",
            "econ_calendar": "blocked_calendar",
            "no_governor": "blocked_no_governor",
            "silent zone": "blocked_silent_zone",
            "silent_zone": "blocked_silent_zone",
            "hard_veto": "blocked_hard_veto",
            "warmup": "blocked_warmup",
            "blocked_trading_limits": "blocked_trading_limits",
            "blocked_cooldown": "blocked_cooldown",
            "blocked_natural_cycle": "blocked_natural_cycle",
            "blocked_same_direction": "blocked_same_direction",
            "quality": "blocked_low_quality",
            "score": "blocked_low_score",
            "rejected": "blocked_low_score",
        }
        for key, fam in families.items():
            if key in reason:
                return fam

        if not self._raw_fired(details):
            return "no_raw_signal"
        return "blocked_other"

    # ── public API ───────────────────────────────────────────────────────
    def record(self, asset: str, signal: int, details: dict) -> None:
        """Record one aggregator evaluation. Never raises."""
        try:
            if not isinstance(details, dict):
                return
            with self._lock:
                day = self._today()
                ck = (day, asset)
                self._counts[ck]["evaluations"] += 1
                if self._raw_fired(details):
                    self._counts[ck]["raw_signal"] += 1
                stage = self._classify_stage(signal, details)
                self._counts[ck][stage] += 1

                self._append(day, "funnel", {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "asset": asset,
                    "final_signal": signal,
                    "mr": details.get("mr_signal", 0),
                    "tf": details.get("tf_signal", 0),
                    "ema": details.get("ema_signal", 0),
                    "quality": details.get("signal_quality", details.get("signal_quality", 0)),
                    "reasoning": details.get("reasoning", ""),
                    "stage": stage,
                })

                # Phase 2.2: AI A/B — capture genuinely AI-rejected signals only
                # (validator actively changed a real signal → hold), keyed off
                # `ai_modified`, not the unreliable `validation_passed`.
                ai = details.get("ai_validation") or {}
                if details.get("ai_modified") and signal == 0:
                    self._append(day, "ai_rejects", {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "asset": asset,
                        "intended_signal": (
                            details.get("original_signal")
                            or details.get("mr_signal")
                            or details.get("tf_signal")
                            or details.get("ema_signal")
                        ),
                        "would_be_price": details.get("entry_price")
                        or details.get("current_price"),
                        "pattern": ai.get("pattern_name"),
                        "pattern_confidence": ai.get("pattern_confidence"),
                        "sr_action": ai.get("action"),
                        "reasoning": details.get("reasoning", ""),
                    })

                self._since_summary += 1
                if self._since_summary >= self.summary_every:
                    self._since_summary = 0
                    self._log_summary_locked(day)
        except Exception as e:
            logger.debug(f"[FUNNEL] record failed: {e}")

    def mark_executed(self, asset: str) -> None:
        """Call when an order was actually placed for this asset."""
        try:
            with self._lock:
                day = self._today()
                self._counts[(day, asset)]["executed"] += 1
                self._append(day, "funnel", {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "asset": asset,
                    "stage": "executed",
                })
        except Exception:
            pass

    def _log_summary_locked(self, day: str) -> None:
        for (d, a), c in sorted(self._counts.items()):
            if d != day:
                continue
            blocks = " ".join(
                f"{k}={v}" for k, v in sorted(c.items())
                if k.startswith("blocked") or k == "no_raw_signal"
            )
            logger.info(
                f"[FUNNEL] {a} {d}: eval={c.get('evaluations', 0)} "
                f"raw={c.get('raw_signal', 0)} "
                f"exec_candidate={c.get('passed_to_execution', 0)} "
                f"executed={c.get('executed', 0)} | {blocks}"
            )

    def log_summary(self) -> None:
        with self._lock:
            self._log_summary_locked(self._today())
