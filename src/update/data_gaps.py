"""B14 Part 2 (Desire 16 Oct 2026): the bot's own missing-candle check.

The old check (HistoricalDataUpdater.verify_data_integrity) was never called from anywhere, so a missing candle in a
price file was only ever found by chance. This one runs after every hourly update of the price files, reads the 1H
and 4H files of each market and logs one "[DATA-GAP] ..." warning per problem; the watchdog (config/heartbeat.json,
b14.data_gap) alarms on that line. It only reads files. It never changes them and never raises.

What counts as missing -- a candle the market should have made, by its schedule in NEW YORK time, learnt from
2.5 years of the box's own hourly candles (1H files to 12 Oct 2026, every market):
  BTC        every hour, every day
  EURUSD, GBPAUD, EURJPY, AUDJPY      Sunday 17:00 to Friday 17:00 New York, no daily pause
  GOLD, SILVER, USTEC, USOIL, JP225   Sunday 18:00 to Friday 17:00 New York, paused 17:00-18:00 every evening;
                                      USTEC and JP225 often have no candle in Friday's last hour (16:00) -- not counted
  holidays (no candle is expected, nothing is reported): Christmas (24-26 Dec) and New Year (31 Dec 17:00 to 1 Jan)
             for every market except BTC; for the five CFDs also the US market holidays they close early or shut for
             (MLK day, Presidents' day, Good Friday, Memorial day, Juneteenth, 3 and 4 July, Labor day, Thanksgiving and
             the Friday after), from 17:00 New York the evening before to the end of the day.
Replayed over that whole history (28 Mar 2024 to 12 Oct 2026), the rule reports 8 problems, about 3 a year:
  AUDJPY 1-11 Oct 2026 (168 hourly candles: MT5's own hourly history has the hole, the daily candles are there),
  BTC 23 Aug 2025 10:00 (1), USOIL 15 Oct 2024, 8 Nov 2024, 2 Dec 2024, 11 Jun 2025 (1 each), SILVER 13 Oct 2025
  01:00-05:00 UTC (a late Sunday open), and USTEC 9 Jan 2025 (US markets shut that day for a national day of mourning
  -- a one-off no fixed calendar can know).

Two kinds of report, each logged once per run of the bot:
  hole   expected candles missing between two saved candles, inside the last WINDOW_DAYS days
  late   the newest LATE_AFTER or more expected candles are not in the file (one missing newest candle is normal:
         the hourly update can run a moment before MT5 has it, and the next update re-reads the last 6 candles)
"""
import datetime as dt
import functools

from src.utils.market_hours import MarketHours

CFD = ("GOLD", "SILVER", "USTEC", "USOIL", "JP225")
FRI_LAST_HOUR_OPTIONAL = ("USTEC", "JP225")
WINDOW_DAYS = 7
LATE_AFTER = 2


def group_of(asset, asset_cfg=None):
    """crypto (24/7), cfd or forex."""
    a = str(asset).upper()
    if "BTC" in a or str((asset_cfg or {}).get("asset_type", "")).lower() == "crypto":
        return "crypto"
    return "cfd" if a in CFD else "forex"


def _nth_weekday(y, m, wd, n):
    """The n-th weekday wd (0 = Monday) of month m; n = -1 is the last one."""
    if n > 0:
        d = dt.date(y, m, 1)
        return d + dt.timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))
    d = (dt.date(y, m + 1, 1) if m < 12 else dt.date(y + 1, 1, 1)) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - wd) % 7)


def _easter(y):
    """Easter Sunday (the Gregorian rule)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    return dt.date(y, (h + l_ - 7 * m + 114) // 31, (h + l_ - 7 * m + 114) % 31 + 1)


def _observed(d):
    """A Saturday holiday is kept on the Friday before, a Sunday one on the Monday after."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


@functools.lru_cache(maxsize=16)
def us_holidays(y):
    """The US market holidays the CFDs close early or shut for, as New York dates. 3 July is in when 4 July falls on a
    Tuesday to Friday (US stocks close early that day: USTEC had no candles from 13:00 New York on 3 July 2024 and 2025)."""
    tg = _nth_weekday(y, 11, 3, 4)
    days = {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _easter(y) - dt.timedelta(days=2),
            _nth_weekday(y, 5, 0, -1), _observed(dt.date(y, 6, 19)), _observed(dt.date(y, 7, 4)),
            _nth_weekday(y, 9, 0, 1), tg, tg + dt.timedelta(days=1)}
    if 1 <= dt.date(y, 7, 4).weekday() <= 4:
        days.add(dt.date(y, 7, 3))
    return tuple(sorted(days))


def _holiday(group, ny):
    """True inside a holiday window (ny = New York local time)."""
    if group == "crypto":
        return False
    y = ny.year
    wins = [(dt.datetime(y, 12, 24), dt.datetime(y, 12, 27)),
            (dt.datetime(y, 12, 31, 17), dt.datetime(y + 1, 1, 2)),
            (dt.datetime(y - 1, 12, 31, 17), dt.datetime(y, 1, 2))]
    if group == "cfd":
        for h in us_holidays(y):
            wins.append((dt.datetime.combine(h - dt.timedelta(days=1), dt.time(17)),
                         dt.datetime.combine(h + dt.timedelta(days=1), dt.time(0))))
    return any(a <= ny < b for a, b in wins)


def hour_state(group, asset, h_utc):
    """The hour that starts at h_utc (naive UTC): 'open', 'closed', 'optional' (may have no candle) or 'skip'."""
    if group == "crypto":
        return "open"
    ny = h_utc + dt.timedelta(hours=MarketHours.ny_offset_hours(h_utc))
    if _holiday(group, ny):
        return "skip"
    wd, hr = ny.weekday(), ny.hour
    if wd == 5 or (wd == 4 and hr >= 17):
        return "closed"
    if wd == 6:
        return "open" if hr >= (18 if group == "cfd" else 17) else "closed"
    if group == "cfd" and hr == 17:
        return "closed"
    if wd == 4 and hr == 16 and str(asset).upper() in FRI_LAST_HOUR_OPTIONAL:
        return "optional"
    return "open"


def candle_expected(group, asset, t_utc, tf_h):
    """True when the market had to make the candle that starts at t_utc (1H or 4H)."""
    states = [hour_state(group, asset, t_utc + dt.timedelta(hours=k)) for k in range(tf_h)]
    return "skip" not in states and "open" in states


def find_gaps(times, asset, group, tf_h, now_utc, days=WINDOW_DAYS):
    """times: the saved candle starts (naive UTC). Returns a list of
    {"kind": "hole" | "late", "start", "end", "n", "newest"} for the last `days` days up to the newest closed candle."""
    step = dt.timedelta(hours=tf_h)
    now = now_utc.replace(minute=0, second=0, microsecond=0)
    now = now.replace(hour=(now.hour // tf_h) * tf_h)
    last = now - step                                        # the newest candle that has closed by now
    t = last - dt.timedelta(days=days)
    t = t.replace(hour=(t.hour // tf_h) * tf_h)
    have = set(times)
    expected = []
    while t <= last:
        if candle_expected(group, asset, t, tf_h):
            expected.append(t)
        t += step
    newest = max(times) if times else None
    out, run = [], []
    for i, c in enumerate(expected):
        if c in have:
            if run:
                out.append({"kind": "hole", "start": run[0], "end": run[-1], "n": len(run), "newest": newest})
                run = []
        else:
            run.append(c)
    if run and len(run) >= LATE_AFTER:
        out.append({"kind": "late", "start": run[0], "end": run[-1], "n": len(run), "newest": newest})
    return out


def report_line(asset, tf_name, g):
    """The log line for one problem (always starts with [DATA-GAP] -- the watchdog's tag)."""
    f = "%Y-%m-%d %H:%M"
    if g["kind"] == "hole":
        return ("[DATA-GAP] %s %s: %d candle(s) missing, %s to %s UTC, while the market was open -- the Saturday "
                "repair refills them if MT5 has them" % (asset, tf_name, g["n"], g["start"].strftime(f), g["end"].strftime(f)))
    return ("[DATA-GAP] %s %s: the newest %d closed candle(s) are not in the file (newest saved: %s UTC) -- the price "
            "updates are behind" % (asset, tf_name, g["n"], g["newest"].strftime(f) if g["newest"] else "none"))
