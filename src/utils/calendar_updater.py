"""
Economic Calendar Updater
=========================
Fetches high-impact events from Forex Factory's public JSON feed (no API key
required), merges them with a hardcoded 2026 anchor schedule for FOMC / NFP /
CPI / ECB, and writes the result to config/economic_calendar.json.

The signal aggregator calls reload_calendar() immediately after each write so
the bot picks up new data without restarting.

Background thread:  run_daily_loop()   → refresh at startup + 00:05 UTC daily
One-shot call:      update()           → fetch + write + reload aggregator
"""

import json
import logging
import time
import threading
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, List, Dict, Optional

from typing import Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FOREX FACTORY endpoints  (public, no auth, returns this/next week)
# ─────────────────────────────────────────────────────────────────────────────
FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

# Currencies the bot trades — only block on events for these
RELEVANT_CURRENCIES = {"USD", "EUR", "JPY"}

# ─────────────────────────────────────────────────────────────────────────────
# ANCHOR SCHEDULE  (hard-coded 2026 dates; FF data overrides where available)
#
# Times are UTC.  EDT = UTC-4 (Mar 8 – Nov 1),  EST = UTC-5 otherwise.
# FOMC 14:00 ET → 18:00 UTC (EDT) / 19:00 UTC (EST)
# NFP  08:30 ET → 12:30 UTC (EDT) / 13:30 UTC (EST)
# CPI  08:30 ET → 12:30 UTC (EDT) / 13:30 UTC (EST)
# ECB  12:15 CET → 11:15 UTC (CEST summer) / 11:15 UTC (CET winter same)
# ─────────────────────────────────────────────────────────────────────────────
_ANCHOR: List[Dict] = [
    # ── FOMC ──────────────────────────────────────────────────────────────
    {"datetime": "2026-04-29T18:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-06-17T18:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-07-29T18:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-09-16T18:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-10-28T18:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-12-16T19:00:00Z", "event": "FOMC Rate Decision",    "currency": "USD", "block_hours_before": 2},
    # ── NFP ───────────────────────────────────────────────────────────────
    {"datetime": "2026-05-01T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-06-05T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-07-03T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-08-07T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-09-04T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-10-02T12:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-11-06T13:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-12-04T13:30:00Z", "event": "Non-Farm Payrolls",     "currency": "USD", "block_hours_before": 2},
    # ── CPI ───────────────────────────────────────────────────────────────
    {"datetime": "2026-05-12T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-06-10T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-07-14T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-08-11T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-09-09T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-10-13T12:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-11-10T13:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-12-08T13:30:00Z", "event": "US CPI",                "currency": "USD", "block_hours_before": 2},
    # ── ECB ───────────────────────────────────────────────────────────────
    {"datetime": "2026-06-04T11:15:00Z", "event": "ECB Rate Decision",     "currency": "EUR", "block_hours_before": 2},
    {"datetime": "2026-07-23T11:15:00Z", "event": "ECB Rate Decision",     "currency": "EUR", "block_hours_before": 2},
    {"datetime": "2026-09-10T11:15:00Z", "event": "ECB Rate Decision",     "currency": "EUR", "block_hours_before": 2},
    {"datetime": "2026-10-29T11:15:00Z", "event": "ECB Rate Decision",     "currency": "EUR", "block_hours_before": 2},
    {"datetime": "2026-12-17T11:15:00Z", "event": "ECB Rate Decision",     "currency": "EUR", "block_hours_before": 2},
    # ── US GDP (quarterly advance) ────────────────────────────────────────
    {"datetime": "2026-04-29T12:30:00Z", "event": "US GDP (Q1 Advance)",   "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-07-29T12:30:00Z", "event": "US GDP (Q2 Advance)",   "currency": "USD", "block_hours_before": 2},
    {"datetime": "2026-10-29T12:30:00Z", "event": "US GDP (Q3 Advance)",   "currency": "USD", "block_hours_before": 2},
    # ── ADP Employment ────────────────────────────────────────────────────
    {"datetime": "2026-05-06T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-06-03T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-07-01T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-08-05T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-09-02T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-10-07T12:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-11-04T13:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
    {"datetime": "2026-12-02T13:15:00Z", "event": "ADP Employment",        "currency": "USD", "block_hours_before": 1},
]


class CalendarUpdater:
    """
    Fetches economic calendar data from Forex Factory and maintains a
    merged, up-to-date event list in config/economic_calendar.json.

    Usage in main.py:
        updater = CalendarUpdater(config, reload_callback=_reload_all_calendars)
        thread = updater.start_background_thread()
    """

    def __init__(
        self,
        config: Dict,
        calendar_path: str = "config/economic_calendar.json",
        reload_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Parameters
        ----------
        config          : bot config dict
        calendar_path   : path to economic_calendar.json
        reload_callback : called after every successful write so all running
                          aggregators pick up the new events immediately.
                          In main.py wire this to reload all self.aggregators.
        """
        self.config = config
        self.calendar_path = Path(calendar_path)
        self.reload_callback = reload_callback
        self.relevant_currencies = set(
            config.get("calendar_currencies", list(RELEVANT_CURRENCIES))
        )
        self._last_updated: Optional[datetime] = None
        self._stop_event = threading.Event()

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def update(self) -> bool:
        """
        Fetch live data, merge with anchor schedule, write file, notify aggregator.
        Returns True on success.
        """
        try:
            logger.info("[CALENDAR] 🔄 Refreshing economic calendar...")
            now = datetime.now(timezone.utc)

            # 1. Fetch from Forex Factory (best-effort; may return empty list)
            live_events = self._fetch_ff_events()

            # 2. Filter anchor schedule to future events only
            anchor = []
            for ev in _ANCHOR:
                try:
                    dt = datetime.fromisoformat(ev["datetime"].replace("Z", "+00:00"))
                    if dt >= now - timedelta(hours=24):
                        anchor.append({**ev, "impact": "HIGH", "source": "schedule"})
                except Exception:
                    continue

            # 3. Merge (live takes priority over anchor for same day + event name)
            merged = self._merge(live_events, anchor)

            # 4. Keep only upcoming events, sort soonest first
            upcoming = [
                e for e in merged
                if datetime.fromisoformat(e["datetime"].replace("Z", "+00:00"))
                >= now - timedelta(hours=24)
            ]
            upcoming.sort(key=lambda e: e["datetime"])

            # 5. Write to file
            payload = {
                "_comment": (
                    "Auto-maintained by CalendarUpdater. "
                    "Live data from Forex Factory + anchor schedule. "
                    "Do not edit manually — changes will be overwritten on next refresh."
                ),
                "_updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "_source": "forex_factory + schedule",
                "_live_count": sum(1 for e in upcoming if e.get("source") == "forex_factory"),
                "_anchor_count": sum(1 for e in upcoming if e.get("source") == "schedule"),
                "events": upcoming,
            }
            self.calendar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calendar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self._last_updated = now
            logger.info(
                f"[CALENDAR] ✅ {len(upcoming)} events written "
                f"({payload['_live_count']} live, {payload['_anchor_count']} from schedule)"
            )

            # 6. Hot-reload all running aggregators via callback
            if self.reload_callback is not None:
                try:
                    self.reload_callback()
                except Exception as _cb_e:
                    logger.warning(f"[CALENDAR] reload_callback failed: {_cb_e}")

            return True

        except Exception as e:
            logger.error(f"[CALENDAR] Update failed: {e}", exc_info=True)
            return False

    def start_background_thread(self) -> threading.Thread:
        """Start the daily-refresh daemon thread and return it."""
        t = threading.Thread(
            target=self._daily_loop,
            daemon=True,
            name="CalendarUpdater",
        )
        t.start()
        logger.info("[CALENDAR] Background refresh thread started")
        return t

    def stop(self):
        """Signal the background thread to exit cleanly."""
        self._stop_event.set()

    # ─────────────────────────────────────────────────────────────────────
    # BACKGROUND LOOP
    # ─────────────────────────────────────────────────────────────────────

    def _daily_loop(self):
        """Run update() at startup, then every day at 00:05 UTC."""
        # Initial fetch
        self.update()

        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

            wait_secs = (next_run - now).total_seconds()
            logger.info(
                f"[CALENDAR] Next refresh in "
                f"{wait_secs / 3600:.1f}h "
                f"({next_run.strftime('%Y-%m-%d %H:%M UTC')})"
            )

            # Sleep in 60-second chunks so stop() is responsive
            elapsed = 0
            while elapsed < wait_secs and not self._stop_event.is_set():
                time.sleep(min(60, wait_secs - elapsed))
                elapsed += 60

            if not self._stop_event.is_set():
                self.update()

    # ─────────────────────────────────────────────────────────────────────
    # FOREX FACTORY FETCH
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_ff_events(self) -> List[Dict]:
        """Fetch this week + next week from Forex Factory. Returns [] on failure."""
        events: List[Dict] = []
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        for url in FF_URLS:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"[CALENDAR] FF {url.split('/')[-1]}: HTTP {resp.status_code}")
                    continue

                raw = resp.json()
                if not isinstance(raw, list):
                    logger.warning(f"[CALENDAR] FF unexpected response type: {type(raw)}")
                    continue

                parsed = self._parse_ff_response(raw)
                events.extend(parsed)
                logger.info(
                    f"[CALENDAR] FF {url.split('/')[-1]}: "
                    f"{len(parsed)} HIGH-impact events"
                )

            except requests.Timeout:
                logger.warning(f"[CALENDAR] FF timeout: {url}")
            except Exception as e:
                logger.warning(f"[CALENDAR] FF fetch error ({url.split('/')[-1]}): {e}")

        return events

    def _parse_ff_response(self, raw: List[Dict]) -> List[Dict]:
        """
        Parse FF JSON into standard event dicts.
        FF format: {"title", "country", "date", "time", "impact", "forecast", "previous"}
        Date: "Apr 23, 2026"  |  Time: "8:30am" (Eastern Time)
        """
        from zoneinfo import ZoneInfo  # Python 3.9+

        et_tz = ZoneInfo("America/New_York")
        events: List[Dict] = []

        for item in raw:
            try:
                impact = (item.get("impact") or "").strip().lower()
                if impact != "high":
                    continue

                country = (item.get("country") or "").strip().upper()
                if country not in self.relevant_currencies:
                    continue

                title = (item.get("title") or "").strip()
                date_str = (item.get("date") or "").strip()
                time_str = (item.get("time") or "").strip().lower()

                if not title or not date_str:
                    continue

                # Skip all-day / tentative events (no precise block window possible)
                if time_str in ("all day", "tentative", ""):
                    continue

                # Parse "Apr 23, 2026 8:30am" as Eastern Time
                dt_naive = datetime.strptime(
                    f"{date_str} {time_str}", "%b %d, %Y %I:%M%p"
                )
                dt_et = dt_naive.replace(tzinfo=et_tz)
                dt_utc = dt_et.astimezone(timezone.utc)

                events.append({
                    "datetime": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "event": title,
                    "currency": country,
                    "impact": "HIGH",
                    "block_hours_before": 2,
                    "source": "forex_factory",
                    "forecast": item.get("forecast", ""),
                    "previous": item.get("previous", ""),
                })

            except Exception:
                continue

        return events

    # ─────────────────────────────────────────────────────────────────────
    # MERGE LOGIC
    # ─────────────────────────────────────────────────────────────────────

    def _merge(self, live: List[Dict], anchor: List[Dict]) -> List[Dict]:
        """
        Live events take priority.  Anchor events are included only when no live
        event covers the same date + similar event name (prefix match).
        """
        # Build a set of (date, name_prefix) from live events
        live_keys: set = set()
        for e in live:
            day = e["datetime"][:10]
            # Use first 12 chars of lowercased name as a fuzzy key
            prefix = e["event"].lower().replace(" ", "").replace("-", "")[:12]
            live_keys.add((day, prefix))

        merged = list(live)
        for e in anchor:
            day = e["datetime"][:10]
            prefix = e["event"].lower().replace(" ", "").replace("-", "")[:12]
            if (day, prefix) not in live_keys:
                merged.append(e)

        return merged
