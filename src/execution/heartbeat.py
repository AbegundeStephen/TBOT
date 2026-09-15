"""
DIARY-1 D5: heartbeat monitor.

This project's real failures have been silent -- a method never called
([LIFECYCLE]), a log at the wrong level ([GATE-ID-TRACE]), a value stuck at
zero (_atr4), a config key nobody reads (use_ema_structure). Every component
registers a promise in config/heartbeat.json; this checks each one every
five minutes and turns a broken promise into a loud ERROR line plus one
Telegram message, instead of a silence nobody notices for weeks.

What it catches: absence and wrong shape. What it does NOT catch: wrong
values (R1 six pips behind H needed a human) -- that stays the replayer's
job.
"""

import glob
import json
import logging
import os
import re
import time
from collections import deque

logger = logging.getLogger("HEARTBEAT")

_REGISTRY_PATH = "config/heartbeat.json"
_CHECK_INTERVAL_S = 300          # 5 minutes
_ERROR_REPEAT_S = 3600           # at most hourly while still failing
_OK_SUMMARY_S = 3600             # once an hour


class _TailHandler(logging.Handler):
    """Appends every formatted log record to a bounded deque -- the
    heartbeat's own view of the log, independent of file rotation/rereads."""

    def __init__(self, maxlen=50000):
        super().__init__()
        self.buffer = deque(maxlen=maxlen)

    def emit(self, record):
        try:
            self.buffer.append((record.created, record.getMessage()))
        except Exception:
            pass


class HeartbeatMonitor:
    """One instance lives on the bot for its whole runtime. Call `tick()`
    from the same periodic timer that runs the [VALIDATOR] report -- it
    self-paces to every 5 minutes internally, so calling it more often
    (e.g. every trading cycle) is harmless."""

    def __init__(self, config=None, registry_path=_REGISTRY_PATH, telegram_bot=None):
        self.config = config or {}
        self.registry_path = registry_path
        self.telegram_bot = telegram_bot
        self.assets = list(self.config.get("assets", {}).keys())
        self.handler = _TailHandler()
        logging.getLogger().addHandler(self.handler)
        self._promises = self._load_registry()
        self._state = {}   # promise_id -> {"failing": bool, "last_error_ts": float}
        self._last_check_ts = 0.0
        self._last_ok_summary_ts = 0.0
        logger.info("[HEARTBEAT] loaded %d enabled promises from %s",
                    len(self._promises), self.registry_path)

    def _load_registry(self):
        try:
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)
            return [p for p in data.get("promises", []) if p.get("enabled", True)]
        except Exception as e:
            logger.error("[HEARTBEAT] could not load registry %s: %s", self.registry_path, e)
            return []

    # ── log-line evaluation (cadence, set) ──────────────────────────────

    def _lines_in_window(self, window_min):
        cutoff = time.time() - window_min * 60
        return [msg for ts, msg in self.handler.buffer if ts >= cutoff]

    def _count_tag(self, tag, window_min, asset=None):
        lines = self._lines_in_window(window_min)
        try:
            pattern = re.compile(tag)
        except re.error:
            pattern = re.compile(re.escape(tag))
        n = 0
        for msg in lines:
            if pattern.search(msg) and (asset is None or asset in msg):
                n += 1
        return n

    def _check_cadence(self, p):
        tag = p["tag"]
        window_min = p.get("window_min", 60)
        _min, _max = p.get("min"), p.get("max")
        if p.get("per_asset"):
            bad = []
            for a in self.assets:
                if not self.config.get("assets", {}).get(a, {}).get("enabled", False):
                    continue
                n = self._count_tag(tag, window_min, asset=a)
                if _min is not None and n < _min:
                    bad.append(f"{a}={n}<{_min}")
                if _max is not None and n > _max:
                    bad.append(f"{a}={n}>{_max}")
            return (not bad), (f"{tag}: " + ", ".join(bad) if bad else "")
        n = self._count_tag(tag, window_min)
        if _min is not None and n < _min:
            return False, f"{tag}: {n} < min {_min} in {window_min}min"
        if _max is not None and n > _max:
            return False, f"{tag}: {n} > max {_max} in {window_min}min"
        return True, ""

    def _check_set(self, p):
        tag = p["tag"]
        extract = p.get("extract")
        allowed = p.get("allowed")
        forbidden = p.get("forbidden")
        window_min = p.get("window_min", 1440)
        lines = self._lines_in_window(window_min)
        bad = []
        for msg in lines:
            if tag not in msg or extract not in msg:
                continue
            val = msg.split(extract, 1)[1].split()[0].strip(",")
            if allowed is not None and val not in [str(a) for a in allowed]:
                bad.append(val)
            if forbidden is not None and val in forbidden:
                bad.append(val)
        if bad:
            return False, f"{tag} {extract}: unexpected value(s) {sorted(set(bad))}"
        return True, ""

    # ── direct-state evaluation (non_null, ledger-backed set) ───────────

    def _recent_episode_rows(self, limit=50):
        rows = []
        for f in sorted(glob.glob("logs/episodes/*.jsonl"))[-2:]:
            try:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                continue
        return rows[-limit:]

    def _check_non_null(self, p):
        if "config_path" in p:
            path = p["config_path"]
            parts = path.split(".")
            if "*" in parts:
                star_idx = parts.index("*")
                sub = self.config
                for k in parts[:star_idx]:
                    sub = sub.get(k, {}) if isinstance(sub, dict) else {}
                rest = parts[star_idx + 1:]
                bad = []
                for asset_key, asset_val in (sub or {}).items():
                    v = asset_val
                    for k in rest:
                        v = v.get(k) if isinstance(v, dict) else None
                    if v is None:
                        bad.append(asset_key)
                return (not bad), (f"{path}: missing for {bad}" if bad else "")
            v = self.config
            for k in parts:
                v = v.get(k) if isinstance(v, dict) else None
            return (v is not None), (f"{path}: missing" if v is None else "")
        if "file" in p:
            ok = os.path.exists(p["file"]) and os.path.getsize(p["file"]) > 0
            return ok, (f"{p['file']}: missing or empty" if not ok else "")
        if "ledger_field" in p:
            field = p["ledger_field"]
            rows = self._recent_episode_rows(limit=50)
            if not rows:
                return True, ""   # nothing to check yet -- not a failure
            missing = sum(1 for r in rows if r.get(field) is None or r.get(field) == "")
            ok = missing == 0
            return ok, (f"ledger field {field}: missing on {missing}/{len(rows)} recent rows" if not ok else "")
        return True, ""

    def _check_set_ledger(self, p):
        field = p["ledger_field"]
        allowed = p.get("allowed") or []
        rows = self._recent_episode_rows(limit=50)
        if not rows:
            return True, ""
        bad = [r.get(field) for r in rows if r.get(field) not in allowed]
        if bad:
            return False, f"ledger field {field}: unexpected value(s) {sorted(set(map(str, bad)))}"
        return True, ""

    def _evaluate(self, p):
        try:
            ptype = p.get("type")
            if ptype == "cadence":
                return self._check_cadence(p)
            if ptype == "set":
                if "ledger_field" in p:
                    return self._check_set_ledger(p)
                return self._check_set(p)
            if ptype == "non_null":
                return self._check_non_null(p)
        except Exception as e:
            return False, f"checker error: {e}"
        return True, ""

    def _notify(self, message):
        try:
            if self.telegram_bot and getattr(self.telegram_bot, "_current_loop", None):
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.send_notification(message), self.telegram_bot._current_loop,
                )
        except Exception as e:
            logger.debug("[HEARTBEAT] telegram notify failed: %s", e)

    def tick(self):
        """Self-paced to _CHECK_INTERVAL_S -- safe to call as often as the
        caller's own timer fires."""
        now = time.time()
        if now - self._last_check_ts < _CHECK_INTERVAL_S:
            return
        self._last_check_ts = now

        ok_count = 0
        for p in self._promises:
            pid = p["id"]
            passed, detail = self._evaluate(p)
            st = self._state.setdefault(pid, {"failing": False, "last_error_ts": 0.0})
            if passed:
                if st["failing"]:
                    logger.info("[HEARTBEAT] RECOVERED %s", pid)
                st["failing"] = False
                ok_count += 1
            else:
                was_failing = st["failing"]
                st["failing"] = True
                if not was_failing or (now - st["last_error_ts"]) >= _ERROR_REPEAT_S:
                    st["last_error_ts"] = now
                    logger.error("[HEARTBEAT] FAIL %s: %s", pid, detail)
                    if not was_failing:
                        self._notify(f"⚠️ HEARTBEAT FAIL: {pid} — {detail}")

        if now - self._last_ok_summary_ts >= _OK_SUMMARY_S:
            self._last_ok_summary_ts = now
            logger.info("[HEARTBEAT] OK %d/%d promises", ok_count, len(self._promises))
