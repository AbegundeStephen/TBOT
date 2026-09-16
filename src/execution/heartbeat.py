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
import threading
import time
from collections import deque

logger = logging.getLogger("HEARTBEAT")

_REGISTRY_PATH = "config/heartbeat.json"
_CHECK_INTERVAL_S = 300          # 5 minutes
_ERROR_REPEAT_S = 3600           # at most hourly while still failing
_OK_SUMMARY_S = 3600             # once an hour


class _TailHandler(logging.Handler):
    """Appends every formatted log record to a bounded deque -- the
    heartbeat's own view of the log, independent of file rotation/rereads.

    HOTFIX (found within minutes of DIARY-1 going live): every check used to
    iterate self.buffer directly while other threads were still calling
    emit() on it -- a live deque mutated during iteration raises
    "RuntimeError: deque mutated during iteration", which the per-promise
    except in _evaluate() was catching and reporting as a FAIL for whatever
    promise happened to be running at that instant. Random promise names,
    same underlying error, every few minutes -- nothing about those
    promises was actually failing. Fixed by never iterating the live
    buffer: emit() appends under a lock, and snapshot() copies it out under
    the same lock once per tick, before any promise is evaluated.

    HF-2 H1: never count yourself -- a HEARTBEAT FAIL/OK/RECOVERED line is
    itself a log record; without this, one promise's line can satisfy (or
    inflate the count toward) another cadence check purely because the
    checker logged about it.
    """

    def __init__(self, maxlen=50000):
        super().__init__()
        self.buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.started_at = time.time()   # HF-2 H4: warm-up window anchor

    def emit(self, record):
        try:
            if record.name == "HEARTBEAT" or "[HEARTBEAT]" in record.getMessage():
                return
            with self._lock:
                self.buffer.append((record.created, record.getMessage()))
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            return list(self.buffer)


# HF-2 H3: root logger, exactly once, process-wide -- a second
# HeartbeatMonitor construction (e.g. across whatever hot-reload path
# recreates parts of TradingBot) must reuse the same handler rather than
# attaching a second one, which would double-count every line.
_shared_handler = None
_shared_handler_lock = threading.Lock()


def _get_tail_handler():
    global _shared_handler
    with _shared_handler_lock:
        if _shared_handler is None:
            _shared_handler = _TailHandler()
            # Remove any stray attachment to a named (non-root) logger from
            # an older version of this code -- only root sees every record.
            for name, obj in list(logging.Logger.manager.loggerDict.items()):
                if isinstance(obj, logging.Logger):
                    for h in list(obj.handlers):
                        if isinstance(h, _TailHandler):
                            obj.removeHandler(h)
            root = logging.getLogger()
            for h in list(root.handlers):
                if isinstance(h, _TailHandler) and h is not _shared_handler:
                    root.removeHandler(h)
            root.addHandler(_shared_handler)
        return _shared_handler


class HeartbeatMonitor:
    """One instance lives on the bot for its whole runtime. Call `tick()`
    from the same periodic timer that runs the [VALIDATOR] report -- it
    self-paces to every 5 minutes internally, so calling it more often
    (e.g. every trading cycle) is harmless."""

    def __init__(self, config=None, registry_path=_REGISTRY_PATH, telegram_bot=None, bot=None):
        self.config = config or {}
        self.registry_path = registry_path
        self.telegram_bot = telegram_bot
        self.bot = bot   # HF-2 H6: source of bot.market_status for "market_open"
        self.assets = list(self.config.get("assets", {}).keys())
        self.handler = _get_tail_handler()
        self._promises = self._load_registry()
        self._state = {}   # promise_id -> {"failing": bool, "last_error_ts": float}
        self._warmup_logged = set()     # HF-2 H4: promise ids already logged once
        self._market_skip_logged = set()  # HF-2 H6: (promise_id, asset) already logged once
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
    # All of these take `snap` -- one immutable list captured by tick() via
    # self.handler.snapshot() before any promise is evaluated. Never touch
    # self.handler.buffer directly here (see _TailHandler's HOTFIX note).

    @staticmethod
    def _lines_in_window(snap, window_min):
        cutoff = time.time() - window_min * 60
        return [msg for ts, msg in snap if ts >= cutoff]

    @staticmethod
    def _tag_matcher(tag):
        """
        HOTFIX (hotfix 2): most registry tags are literal log prefixes like
        "[KILL-R1]", and [ / ] are regex character-class delimiters.
        re.compile("[KILL-R1]") does NOT raise (so the old re.error fallback
        never triggered); it silently compiles to a character class matching
        any ONE of {K,I,L,R,1,-} -- which matches nearly every log line,
        wildly inflating every count. Only treat a tag as a real regex when
        it visibly asks for one (contains ".*" -- every actual regex tag in
        the registry, e.g. "Fetching .* H4", uses this).

        HF-2 H2: never a bare substring search either, for a bracketed
        literal tag -- "[KILL-R1]" must anchor the bracketed token itself
        (start of message; every log call in this codebase is
        logger.info("[TAG] ...", ...), so the tag is always the first
        thing in record.getMessage()), not match "[KILL-R1]" appearing
        incidentally inside some unrelated line's text. A free-text tag
        with no brackets (e.g. "Traceback", "No fresh tick data") still
        searches anywhere in the line -- it was never meant to be anchored.
        """
        if ".*" in tag:
            # HF-2 follow-up: several regex tags mix a literal bracket WITH
            # the ".*" wildcard ("[EPISODE] .* closed and written",
            # "[PERSIST] .* restored", "[VTM] .* management paused") --
            # compiling the tag raw hits the exact same character-class bug
            # as a pure-literal bracket tag (confirmed: re.compile("[VTM] .*
            # management paused") matches any line with a bare "T" ... far
            # too permissive). Escape the whole tag, then restore only the
            # ".*" sequences to real wildcards.
            try:
                pattern = re.compile(re.escape(tag).replace(re.escape(".*"), ".*"))
            except re.error:
                pattern = re.compile(re.escape(tag))
            return pattern.search
        if tag.startswith("[") and tag.endswith("]"):
            return lambda msg: msg.startswith(tag)
        return lambda msg: tag in msg

    def _count_tag(self, snap, tag, window_min, asset=None):
        lines = self._lines_in_window(snap, window_min)
        match = self._tag_matcher(tag)
        n = 0
        for msg in lines:
            if match(msg) and (asset is None or asset in msg):
                n += 1
        return n

    def _is_market_closed(self, asset):
        """HF-2 H6: reads the status Group M writes to bot.market_status.
        Unknown/absent status is treated as OPEN -- never silently skip a
        check just because the field hasn't been wired somewhere yet."""
        status_map = getattr(self.bot, "market_status", None) if self.bot else None
        if not status_map:
            return False
        entry = status_map.get(asset)
        return bool(entry and entry[0] == "CLOSED")

    def _check_cadence(self, p, snap):
        pid = p["id"]
        tag = p["tag"]
        window_min = p.get("window_min", 60)
        _min, _max = p.get("min"), p.get("max")

        # HF-2 H4: warm-up -- a promise whose window is longer than the
        # process (or the tail handler) has been alive can only ever read
        # as a false absence. Skip silently after the first log.
        uptime_s = time.time() - self.handler.started_at
        if window_min * 60 > uptime_s:
            if pid not in self._warmup_logged:
                self._warmup_logged.add(pid)
                logger.info("[HEARTBEAT] WARMUP skip %s (uptime %.0fmin < window %dmin)",
                            pid, uptime_s / 60, window_min)
            return True, ""

        _market_gated = p.get("when") == "market_open"

        if p.get("per_asset"):
            bad = []
            for a in self.assets:
                if not self.config.get("assets", {}).get(a, {}).get("enabled", False):
                    continue
                if _market_gated and self._is_market_closed(a):
                    if (pid, a) not in self._market_skip_logged:
                        self._market_skip_logged.add((pid, a))
                        logger.info("[HEARTBEAT] SKIP %s (%s closed)", pid, a)
                    continue
                else:
                    self._market_skip_logged.discard((pid, a))
                n = self._count_tag(snap, tag, window_min, asset=a)
                if _min is not None and n < _min:
                    bad.append(f"{a}={n}<{_min}")
                if _max is not None and n > _max:
                    bad.append(f"{a}={n}>{_max}")
            return (not bad), (f"{tag}: " + ", ".join(bad) if bad else "")
        n = self._count_tag(snap, tag, window_min)
        if _min is not None and n < _min:
            return False, f"{tag}: {n} < min {_min} in {window_min}min"
        if _max is not None and n > _max:
            return False, f"{tag}: {n} > max {_max} in {window_min}min"
        return True, ""

    def _check_set(self, p, snap):
        tag = p["tag"]
        extract = p.get("extract")
        allowed = p.get("allowed")
        forbidden = p.get("forbidden")
        window_min = p.get("window_min", 1440)
        lines = self._lines_in_window(snap, window_min)
        match = self._tag_matcher(tag)
        bad = []
        for msg in lines:
            if not match(msg) or extract not in msg:
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
        # HF-2 I3: not in the batch's own file list, but this runs live
        # inside each instance's process -- unsuffixed, instance B's
        # heartbeat would read instance A's episode ledger.
        from src.utils.instance_paths import suffixed_path as _p_inst_ep
        rows = []
        for f in sorted(glob.glob(f"{_p_inst_ep('logs/episodes')}/*.jsonl"))[-2:]:
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

    def _episode_rows_in_window(self, window_min):
        """B3 GATE-1 G6: unlike _recent_episode_rows (fixed row count, used
        for schema/id spot-checks), gate coverage needs an actual TIME
        window -- reads enough daily files to cover it, filtered by each
        row's close_time (falling back to entry_time)."""
        from src.utils.instance_paths import suffixed_path as _p_inst_ep
        import datetime as _dt
        days_back = max(1, int(window_min / 1440) + 2)
        cutoff = time.time() - window_min * 60
        rows = []
        for f in sorted(glob.glob(f"{_p_inst_ep('logs/episodes')}/*.jsonl"))[-days_back:]:
            try:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    _ts_raw = row.get("close_time") or row.get("entry_time")
                    if not _ts_raw:
                        continue
                    try:
                        _ts = _dt.datetime.fromisoformat(str(_ts_raw).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
                    if _ts >= cutoff:
                        rows.append(row)
            except Exception:
                continue
        return rows

    def _check_ledger_seen(self, p):
        """B3 GATE-1 G6: counts episode rows with `ledger_field` == `equals`
        within window_min. One promise per gate, generated from
        config/gates.json."""
        field = p["ledger_field"]
        target = p["equals"]
        window_min = p.get("window_min", 10080)
        rows = self._episode_rows_in_window(window_min)
        n = sum(1 for r in rows if r.get(field) == target)
        _min = p.get("min", 1)
        if n < _min:
            return False, f"{field}=={target}: {n} < min {_min} in {window_min}min"
        return True, ""

    def _check_non_null(self, p, snap):
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
        if "tag" in p and "extract" in p:
            # HOTFIX: reload.keys (tag+extract+min_value) and brc.confirmed_fields
            # (tag+extract list) were falling through to the unconditional
            # `return True, ""` below -- silently passing forever, never
            # actually checking anything.
            win = p.get("window_min", 1440)
            lines = self._lines_in_window(snap, win)
            tag = p["tag"]
            extracts = p["extract"] if isinstance(p["extract"], list) else [p["extract"]]
            tag_match = self._tag_matcher(tag)
            match_line = None
            for msg in reversed(lines):
                if tag_match(msg):
                    match_line = msg
                    break
            if match_line is None:
                return True, ""   # tag hasn't appeared in-window yet -- not a failure
            if "min_value" in p:
                ex = extracts[0]
                if ex not in match_line:
                    return False, f"{tag}: '{ex}' not found in matching line"
                try:
                    val = int(match_line.split(ex, 1)[1].split()[0].strip(","))
                except Exception:
                    return False, f"{tag}: could not parse value after '{ex}'"
                ok = val >= p["min_value"]
                return ok, (f"{tag} {ex}: {val} < min_value {p['min_value']}" if not ok else "")
            missing = [ex for ex in extracts if ex not in match_line]
            return (not missing), (f"{tag}: missing field(s) {missing}" if missing else "")
        return True, ""

    def _check_set_ledger(self, p):
        # HF-2A A2: missing_as lets a promise treat a MISSING field as if it
        # held a given value -- episode.schema's real need: pre-DIARY-1 rows
        # have no schema_version at all, and "None is not in [1, 2]" would
        # fail forever on those old rows even though they are legitimately
        # schema 1 (schema_version didn't exist yet).
        field = p["ledger_field"]
        allowed = p.get("allowed") or []
        missing_as = p.get("missing_as")
        rows = self._recent_episode_rows(limit=50)
        if not rows:
            return True, ""
        bad = []
        for r in rows:
            v = r.get(field)
            if v is None and missing_as is not None:
                v = missing_as
            if v not in allowed:
                bad.append(v)
        if bad:
            return False, f"ledger field {field}: unexpected value(s) {sorted(set(map(str, bad)))}"
        return True, ""

    def _check_file_fresh(self, p):
        """B3 REPLAYER-2 R5 / RL-1: file(s) touched within window_min. `file`
        may be a glob pattern (e.g. logs/replayer_arms_*.json); passes if the
        newest match's mtime is inside the window. No matches at all is not
        a failure -- these promises are 'enabled once threshold first met',
        so an offline tool that hasn't run yet must not page anyone."""
        pattern = p["file"]
        win_s = p.get("window_min", 10080) * 60
        matches = glob.glob(pattern)
        if not matches:
            return True, ""
        newest = max(os.path.getmtime(m) for m in matches)
        ok = (time.time() - newest) <= win_s
        return ok, ("" if ok else f"{pattern}: newest match is stale (> {p.get('window_min')} min)")

    def _evaluate(self, p, snap):
        """Returns (passed, detail) where passed is True/False for a real
        promise result, or None if the checker itself errored -- a checker
        crash is NOT a promise failure (HOTFIX, see _TailHandler)."""
        try:
            ptype = p.get("type")
            if ptype == "cadence":
                return self._check_cadence(p, snap)
            if ptype == "set":
                if "ledger_field" in p:
                    return self._check_set_ledger(p)
                return self._check_set(p, snap)
            if ptype == "non_null":
                return self._check_non_null(p, snap)
            if ptype == "ledger_seen":
                return self._check_ledger_seen(p)
            if ptype == "file_fresh":
                return self._check_file_fresh(p)
        except Exception as e:
            return None, f"checker error: {e}"
        return True, ""

    def _notify(self, message):
        try:
            if self.telegram_bot and getattr(self.telegram_bot, "_current_loop", None):
                import asyncio
                # HF-2 H5: explicit plain text. send_notification's own
                # parse_mode default is Markdown, and promise ids like
                # "kill.once_per_bar" carry underscores -- an odd count
                # left an unclosed italics span and Telegram silently
                # rejected the whole message.
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.send_notification(message, parse_mode=None),
                    self.telegram_bot._current_loop,
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

        # HOTFIX: one snapshot for the whole tick, taken under the handler's
        # lock -- every promise evaluates against this same immutable list,
        # never against the live buffer other threads keep appending to.
        snap = self.handler.snapshot()

        ok_count = 0
        for p in self._promises:
            pid = p["id"]
            passed, detail = self._evaluate(p, snap)
            st = self._state.setdefault(pid, {"failing": False, "last_error_ts": 0.0})
            if passed is None:
                # Checker itself errored -- not a promise failure. Loud once,
                # never a FAIL log, never a Telegram, never counted either way.
                logger.error("[HEARTBEAT] CHECKER ERROR %s: %s", pid, detail)
                continue
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
