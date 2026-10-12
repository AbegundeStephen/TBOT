"""
B11-NS -- the new proof engine, and the shared per-market trade rules.

Desire's tested rules (research 25 Sep 2026: tools/final_test.py, realistic_test.py,
target_test.py). Every number below was measured; change none of them without a re-test.

  R2      = the last 4H swing CLOSE (a 4H close higher/lower than the 2 closes on each side).
            Its zone runs from that close to the wick (the highest high / lowest low of
            those 5 candles).
  R1      = the opposite swing that confirms after it -- the kill line.
  setup   = born when R1 confirms, if R2 and R1 are at least 1 x 4H ATR apart.
  break   = a 4H close past R2's close, within 10 days.
  peak    = the best 1H close of the break's 4 hours, then ratcheting until the retest.
  retest  = a 1H low (high, for shorts) within 1 x 1H ATR of the zone's wick edge, within 7 days.
  entry E = the first 1H close past the peak after the retest, within 7 days.
  entry B = the 4H break close itself (GOLD, USOIL, USTEC).
  dead    = a 4H close back past R1; retired when a wait runs out.
  fresh   = the entry no more than 2.5 x 1H ATR from R2, otherwise retired at once.
  stop    = R2's close - 0.3 x 1H ATR (floor: min_sl_pct; cap: 5 x 1H ATR).
  "one move" = the 1H ATR: the simple 14-candle average of the true range.

The Livermore brains never kill a proof (Desire's ruling 5). They are drawn on the
ladder as context only.

One module, used by the builder (proofs), the trade manager (stop, target, runner exit),
pre-order sizing and the practice lane -- so the four can never drift apart.
"""
import copy
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- the tested rules ------------------------------------------------------------------
K = 2
MIN_SWING_ATR4 = 1.0
TOUCH_ATR = 1.0
ALLOW_ATR = 0.3
FRESH_ATR = 2.5
PKG_NEAR = 0.5                 # B13 (Desire 2 Oct): a bigger 4H line this close ahead (4H moves) -> the package
PKG_AHEAD_DAYS = 60            # B13: bigger 4H lines are looked for this far back
AHEAD_ON_WICKS = True          # B13 item 33 (Desire 6 Oct): the old high/low in front is judged on 4H WICKS (False = closes)
DIAG_LIFE = 60                 # B13 item 32 (Desire 6 Oct): 4H candles a 3-touch diagonal is watched for a break
DIAG_RECENT_H = 24             # B13 item 32: a diagonal broken the same way this many hours before the signal
STOP_CAP_ATR = 5.0
GATE_TARGET_ATR = 2.5          # the R:R screen the tests used, for every market
WAIT_BREAK = timedelta(days=10)
WAIT_TOUCH = timedelta(days=7)
WAIT_TRIGGER = timedelta(days=7)
SPIKE = 0.85                   # a trigger candle closing in its top 15% (bottom, for shorts)
TRADE_BARS = 168               # 7 days of 1H candles
STALE_EMIT = timedelta(hours=2)   # HF-2: a proof older than this is never live (weekend / outage catch-up)
APLUS_NEAR_ATR = 0.5           # B12 label: entry within half a move of a 4H brain level or an open 1H gap
FVG_1H_BARS = 240              # B12: an open 1H fair-value gap counts for 10 days
BRAIN_HIST_DAYS = 30           # B12: brain levels remembered for the labels and the bounce pockets
LEVEL_NEAR_ATR = 0.5           # B12 (Tests 7a/8a/8c): "at a level" = within half a typical move of it
BIG_CANDLE_ATR = 1.6           # B12 (Test 7c): a big entry candle spans 1.6+ typical moves
BOUNCE_TOL_ATR = 0.25          # B12 exploration: a touch = within a quarter move of the level
BOUNCE_TARGET_ATR = 2.5        # B12 exploration: the bounce trade's target (Test 1c)
REV_K = 6                      # B12 (decision 35 A): the last 6 hourly candles are re-read every call
REV_TOL = 1e-4                 # ...and a change of more than 0.01% counts as a revised candle
_quiet_log = logging.getLogger(__name__ + ".explore")   # B12: exploration engines stay silent...
_quiet_log.setLevel(logging.WARNING)                     # ...except their own [NS-EXPLORE] lines
BRAIN4_KEEP = 400              # B15: the 4H brain's dated states kept for the phase tag (~66 days; the box test keeps all)
REPLAY_DAYS = 30               # first run after a start: rebuild the setups from 30 days
LAYER1_DAYS = 30
LAYER2_DAYS = {"1H": 10, "4H": 30}
STALE_DAYS = {"1H": 5, "4H": 20}
UP_STATES = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
# B14 item 1.3 (Desire 8 Oct): the engine's memory is stamped with the rules it was built under. A different stamp
# (a code change below, or a change to this market's settings) rebuilds the memory from history on the next look.
RULES_V = "B15"
# B14 (Desire 7-9 Oct): the new entry and exit rules. Off until phase_config.b14_rules_enabled is switched on after
# the 16-month box test; with it off the engine trades exactly as B13. The knobs are for the box test only.
# B15 (Desire 12 Oct): the Phase 2 redesign changes the defaults (B14's value in brackets):
#   sig_break "mark" [b121]: the 3/4-mark break on every line, the signal line too;
#   clock "size": a big line (a major 4H swing, known by then) keeps the hourly proof, a small line is proven on the 4H
#     clock (the package's 4H/4H); "1h" = B14's all-hourly proofs; "phase_all" / "phase_small" = box-test rows S8/S9;
#   wall_route False [True]: a wall ahead no longer sends a signal to the package; lock False [True]; stair True;
#   pkg_majority False [True]: no bigger-timeframe vote (box-test row S7 brings it back);
#   pkg_wait "setup": a big-line signal sent to wait keeps the setup's 7-day waits ("24h" = row S10);
#   trigger_clear 0: the entry close must beat the peak by this many of its own clock's moves (rows S11/S12);
#   ch_edge "sphere": a channel target at the near edge of the other side's sphere ("line" = row S4).
#   wide "rearm" (D8, Desire 12 Oct): a 4H entry with its stop too wide (> 2.5 moves) or its reward:risk too low is a
#        failed attempt -- re-armed, a fresh 4H retest needed (option A); "wait" = no trade at that close, keep waiting
#        for the next 4H close past the same level (option B, row S13); "end" = the setup ends (option C, row S14).
B14_KNOBS = {"witness_h": 12, "ma_witness": True, "runner": True, "runner_dip": True, "turn": "sphere",
             "strong": True, "release": "c", "rearm": True, "rearm_max": 5, "rearm_after_majority": True,
             "pkg_majority": False, "sig_break": "mark", "channel_target": True, "room_r": 0.5, "tp_line_only": True,
             "clock": "size", "wall_route": False, "lock": False, "stair": True, "pkg_wait": "setup",
             "trigger_clear": 0.0, "ch_edge": "sphere", "wide": "rearm"}

DEFAULT_MARKETS = {
    "GOLD":   dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "USOIL":  dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "USTEC":  dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "EURUSD": dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True),
    "GBPAUD": dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True),
    # B14 item 6.1 (Desire 9 Oct): BTC aims for 4 hourly moves with a fixed exit, like the others -- the built-in
    # default now agrees with the switch-on config (config.json decides while BTC's block is there)
    "BTC":    dict(entry="E", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    # B12 new markets (decisions 2, 3, 5, 7) -- the profiles that qualified in Test 6
    "JP225":  dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "EURJPY": dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "SILVER": dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "AUDJPY": dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True),     # paper only
}
KIND_MAP = {"continuation": "TF_CONT", "reversal": "MR_REV"}


def market_settings(asset, pcfg=None):
    """This market's settings: the tested defaults, overridden by phase_config.ns_markets.<ASSET>.
    None for a market the new strategy does not trade."""
    a = str(asset or "").upper()
    over = (((pcfg or {}).get("ns_markets") or {}).get(a) or {})
    if a not in DEFAULT_MARKETS and not over:
        return None
    cfg = dict(DEFAULT_MARKETS.get(a, DEFAULT_MARKETS["EURUSD"]))
    cfg.update(over)
    cfg["min_sl_pct"] = float(cfg.get("min_sl_pct", 0.0) or 0.0)
    cfg["min_rr"] = float(cfg.get("min_rr", 0.5) or 0.5)
    # B12.1 (Desire 30 Sep, decisions 48 and 50): your break rules. Every default = the old B12 rule.
    cfg["quality_lines"] = bool(cfg.get("quality_lines", False))      # only lines that held before or are brain-marked
    cfg["break_rule"] = str(cfg.get("break_rule", "any") or "any")    # "any" | "big_or_1h_hold" (1b) | "big_or_confirmed" (4H)
    cfg["big_break_atr"] = float(cfg.get("big_break_atr", 0.25) or 0.25)
    cfg["room_atr"] = float(cfg.get("room_atr", 1.0) or 1.0)
    cfg["room_required"] = bool(cfg.get("room_required", False))      # decision 48: a tag, unless switched on
    cfg["reversal_at_4h_brain"] = bool(cfg.get("reversal_at_4h_brain", False))   # decision 50 (kept = reversal_rule "brain4")
    # B12.1 (Desire 30 Sep, decisions 50 A and 54): which reversals may live -- "none" | "brain4" (the turning point on
    # a 4H brain level) | "vote2" (BTC: at least 2 of -- 4H brain level / held before / yesterday's low (buy) or high (sell))
    cfg["reversal_rule"] = str(cfg.get("reversal_rule") or ("brain4" if cfg["reversal_at_4h_brain"] else "none"))
    # B14: the master switch (phase_config.b14_rules_enabled), a market's own "b14" key wins; the box-test knobs
    _pc = pcfg or {}
    cfg["b14"] = bool(over.get("b14", _pc.get("b14_rules_enabled", False)))
    cfg["b14_knobs"] = dict(B14_KNOBS, **(_pc.get("b14_knobs") or {}))
    cfg["b15_small_only"] = bool(_pc.get("b15_small_4h_only", False))   # box-test row S6 only: B13 + the 4H clock
    cfg["builtin"] = not bool(over)          # B14 item 7.8: True = no config block, the code's built-in settings
    return cfg


def rules_stamp(cfg, variant=None):
    """B14 item 1.3: what the engine's memory was built under -- the code's rules version plus every setting of this
    market. Any difference rebuilds the memory. A practice-only exploration engine (variant) never uses the B14
    rules, so the B14 switch and its knobs are left out of its stamp."""
    import json as _json
    keep = {k: v for k, v in (cfg or {}).items()
            if k != "builtin" and not (variant and k in ("b14", "b14_knobs"))}
    return "%s|%s|%s" % (RULES_V, variant or "-", _json.dumps(keep, sort_keys=True, default=str))


# ---- shared building blocks -------------------------------------------------------------
def close_times(df, hours):
    """Close times (tz-naive UTC) of a candle frame indexed (or stamped) by open time."""
    if df is None:
        return pd.DatetimeIndex([])
    raw = df["timestamp"] if "timestamp" in df.columns else df.index
    idx = pd.DatetimeIndex(pd.to_datetime(raw))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx + pd.Timedelta(hours=hours)


def atr14(high, low, close):
    """The tests' ATR: simple 14-candle average of the true range."""
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    return pd.Series(tr).rolling(14).mean().values


def swings_4h(t4, hi4, lo4, c4):
    """Every confirmed 4H swing: (confirmed_at, "H"|"L", close, wick edge)."""
    out = []
    n = len(c4)
    for i in range(K, n - K):
        conf = t4[i + K]
        win = c4[i - K:i + K + 1]
        if c4[i] == win.max():
            out.append((conf, "H", float(c4[i]), float(hi4[i - K:i + K + 1].max())))
        if c4[i] == win.min():
            out.append((conf, "L", float(c4[i]), float(lo4[i - K:i + K + 1].min())))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def wick_swings_4h(t4, hi4, lo4):
    """B13 item 33 (Desire 6 Oct): every confirmed 4H swing by its WICK -- (confirmed_at, "H"|"L", wick, wick).
    Used only to find the old high (low) just in front of a signal; the lines the bot trades stay on closes."""
    out = []
    n = len(hi4)
    for i in range(K, n - K):
        conf = t4[i + K]
        if hi4[i] == hi4[i - K:i + K + 1].max():
            out.append((conf, "H", float(hi4[i]), float(hi4[i])))
        if lo4[i] == lo4[i - K:i + K + 1].min():
            out.append((conf, "L", float(lo4[i]), float(lo4[i])))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def diagonal_breaks_4h(t4, a4, c4):
    """B13 items 31-32 (Desire 6 Oct): every break of a 3-touch diagonal on the 4H closes -- (direction, break time).
    Exactly the finder the 16-month tests used: lines through consecutive swing closes (falling through highs for a
    buy, rising through lows for a sell), valid once a third close touched it within half a 4H move (an earlier swing,
    or a later close), watched for DIAG_LIFE 4H candles or until the next swing; a break is a 4H close at least a
    quarter of a 4H move past it."""
    n = len(c4)
    piv = {"H": [], "L": []}
    for i in range(K, n - K):
        w = c4[i - K:i + K + 1]
        if c4[i] == w.max():
            piv["H"].append(i)
        if c4[i] == w.min():
            piv["L"].append(i)
    out = []
    for d, typ in ((1, "H"), (-1, "L")):
        P = piv[typ]
        for j in range(1, len(P)):
            i1, i2 = P[j - 1], P[j]
            v1, v2 = c4[i1], c4[i2]
            if (d == 1 and not v2 < v1) or (d == -1 and not v2 > v1):
                continue
            slope = (v2 - v1) / (i2 - i1)
            nxt = P[j + 1] if j + 1 < len(P) else n
            start = i2 + K
            end = min(n, start + DIAG_LIFE, nxt + K)
            earlier = P[max(0, j - 5):j - 1]
            vf = start if any(a4[p] == a4[p] and abs(c4[p] - (v1 + slope * (p - i1))) <= 0.5 * a4[p]
                              for p in earlier) else None
            for x in range(start, end):
                lv = v1 + slope * (x - i1)
                if vf is None:
                    if x >= i2 + 3 and a4[x] == a4[x] and abs(c4[x] - lv) <= 0.5 * a4[x]:
                        vf = x + 1
                    continue
                if x < vf or not (a4[x] == a4[x] and a4[x] > 0):
                    continue
                if d * (c4[x] - lv) >= 0.25 * a4[x]:
                    out.append((d, t4[x]))
                    break
    return out


def hourly_to_4h(df1):
    """B12 (Desire 28 Sep, decision 35 A): the engine's own 4H candles, built from the hourly candles it already reads
    -- a 4H close IS the close of its last hour. MT5's own H4 series can still be catching up a minute after a 4H
    close: on 28 Sep it handed the bot 84,414 for a BTC candle that closed at 83,325, and a real break was missed.
    Only COMPLETE blocks are returned: a block counts once its end is at or before the last hourly close (so a block
    cut short by a market close completes when the next hour arrives)."""
    d = df1[["open", "high", "low", "close"]].astype(float)
    g = d.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    last_close = df1.index[-1] + pd.Timedelta(hours=1)
    return g[g.index + pd.Timedelta(hours=4) <= last_close]


def ns_levels(d, entry, r2, atr1, min_sl_pct=0.0, target_atr=None):
    """The tested stop and target for one trade. Returns (stop, target) -- target None for
    the runner -- or (None, None) when no valid stop exists."""
    try:
        entry, r2, atr1 = float(entry), float(r2), float(atr1)
    except (TypeError, ValueError):
        return None, None
    if not (entry > 0 and r2 > 0 and atr1 == atr1 and atr1 > 0):
        return None, None
    stop = r2 - d * ALLOW_ATR * atr1
    floor = float(min_sl_pct or 0.0) * entry
    if abs(entry - stop) < floor:
        stop = entry - d * floor
    if abs(entry - stop) > STOP_CAP_ATR * atr1:
        stop = entry - d * STOP_CAP_ATR * atr1
    if (d == 1 and stop >= entry) or (d == -1 and stop <= entry):
        return None, None
    target = (entry + d * float(target_atr) * atr1) if target_atr else None
    return stop, target


def gate_rr_ok(entry, stop, atr1, min_rr):
    """The tests' R:R screen: a 2.5-move reward against the stop, for every market."""
    risk = abs(float(entry) - float(stop))
    return risk > 0 and GATE_TARGET_ATR * float(atr1) / risk >= float(min_rr)


# B14.1 (Desire 11 Oct, option ii) -- the package's stop. B15: shared by the package (its own records) and the engine's
# 4H/4H entries, so the two can never drift apart. Values exactly as B14.1.
PKG_FLOOR_ATR = 1.0            # breathing space: a package stop is never closer than 1 move
PKG_STOP_MAX = 2.5             # a package stop wider than this (1H moves) = no trade at the smallest lot
PKG_SWING_BUF = 0.3            # 0.3 of a move behind the last 1H swing


def package_stop(d, q, t1, hi1, lo1, j1, t_touch, atr, min_sl_pct=0.0):
    """B14.1: 0.3 of a 1H move behind the LAST 1H SWING before the entry -- the last dip for a buy (a 1H low below both
    neighbours' lows, confirmed by the candle after it), the last bounce for a sell -- looked for from the retest
    candle to the last closed 1H candle (index j1), on the far side of the entry q. With no such swing: the push's own
    1H extreme since the retest. Never closer than 1 move nor min_sl_pct. t1 = 1H close times (DatetimeIndex).
    -> (stop, swing, j_sw, too_wide) -- too_wide is the skip text when the stop is wider than PKG_STOP_MAX moves."""
    j_touch = int(np.searchsorted(t1.values, np.datetime64(pd.Timestamp(t_touch))))
    j_touch = min(max(j_touch, 1), j1)
    ext1 = lo1 if d == 1 else hi1
    swing, j_sw = None, None
    for j in range(j1 - 1, j_touch - 1, -1):           # newest first; j + 1 <= j1 so the swing is confirmed
        if d * (ext1[j - 1] - ext1[j]) > 0 and d * (ext1[j + 1] - ext1[j]) > 0 and d * (q - ext1[j]) > 0:
            swing, j_sw = float(ext1[j]), j
            break
    if swing is None:
        swing = float(ext1[j_touch:j1 + 1].min() if d == 1 else ext1[j_touch:j1 + 1].max())
    stop = swing - d * PKG_SWING_BUF * atr
    if d * (q - stop) < PKG_FLOOR_ATR * atr:
        stop = q - d * PKG_FLOOR_ATR * atr             # breathing space: never closer than 1 move
    if d * (q - stop) < min_sl_pct * q:
        stop = q - d * min_sl_pct * q
    too_wide = None
    if d * (q - stop) > PKG_STOP_MAX * atr:
        too_wide = "stop too wide at the smallest lot: %.1f moves (limit %g) to the last 1H swing %.5g%s -- skipped" % (
            d * (q - stop) / atr, PKG_STOP_MAX, swing,
            (" at %s" % str(t1[j_sw])[:16]) if j_sw is not None else " (the push's extreme)")
    return stop, swing, j_sw, too_wide


def candle_strength(d, high, low, close):
    """Where the candle closed in its range, in the trade's direction (1.0 = at the extreme)."""
    rng = float(high) - float(low)
    if not (rng == rng) or rng <= 0:
        return 0.5
    return (float(close) - float(low)) / rng if d == 1 else (float(high) - float(close)) / rng


def pivot_flags(hi, lo, cl, atr):
    """The runner trail's swing points: 4 candles on the left, 2 on the right, >= 0.3 ATR deep."""
    hi, lo, cl, atr = (np.asarray(x, dtype=float) for x in (hi, lo, cl, atr))
    n = len(lo)
    pl, ph = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(6, n - 2):
        a = atr[i] if atr[i] == atr[i] else 0.0
        if all(lo[i] < lo[i - k] for k in (1, 2, 3, 4)) and lo[i] < lo[i + 1] and lo[i] < lo[i + 2] \
                and min(cl[i - 4:i]) - lo[i] >= 0.3 * a:
            pl[i] = True
        if all(hi[i] > hi[i - k] for k in (1, 2, 3, 4)) and hi[i] > hi[i + 1] and hi[i] > hi[i + 2] \
                and hi[i] - max(cl[i - 4:i]) >= 0.3 * a:
            ph[i] = True
    return pl, ph


def runner_step(d, entry, stop, risk0, hi, lo, cl, bars, armed, peak, r_trigger=1.0, r_lock=0.2):
    """One closed 1H candle of the tested runner exit (BTC).
    Lock +0.2R once price has gone +1R; then, after 10 candles and while the close is in
    profit, trail behind the latest confirmed 1H swing minus 0.3 x 1H ATR. The arrays end
    at the candle just closed and may include candles from before the entry.
    Returns (new_stop or None, armed, peak, reason)."""
    hi, lo, cl = (np.asarray(x, dtype=float) for x in (hi, lo, cl))
    j = len(cl) - 1
    if j < 14:
        return None, armed, peak, None
    atr = atr14(hi, lo, cl)
    a = atr[j]
    if not (a == a):
        return None, armed, peak, None
    peak = (max(peak, hi[j]) if d == 1 else min(peak, lo[j])) if peak is not None else (hi[j] if d == 1 else lo[j])
    prof = d * (peak - entry)
    best, why = stop, None
    if prof >= r_trigger * risk0:
        armed = True
        lock = entry + d * r_lock * risk0
        if (d == 1 and lock > best) or (d == -1 and lock < best):
            best, why = lock, "ns_runner_lock"
    if armed and bars >= 10 and d * (cl[j] - entry) > 0:
        pl, ph = pivot_flags(hi, lo, cl, atr)
        piv = pl if d == 1 else ph
        lo_i = max(6, j - 200)
        idx = np.nonzero(piv[lo_i:j - 1])[0] + lo_i
        if len(idx):
            lv = (lo[idx] - 0.3 * a) if d == 1 else (hi[idx] + 0.3 * a)
            ok = lv[lv < cl[j]] if d == 1 else lv[lv > cl[j]]
            if len(ok):
                px = float(ok.max() if d == 1 else ok.min())
                if (d == 1 and px > best) or (d == -1 and px < best):
                    best, why = px, "ns_runner_trail"
    return (best if why else None), armed, peak, why


# ---- the engine ----------------------------------------------------------------------------
class NSEngine:
    """One per market. Stateless between calls: everything lives in the `st` dict the builder
    persists (its _STATE_KEYS). Each call processes every closed 1H candle since the last one,
    oldest first -- so a restart catches up by itself. Only a trigger on the NEWEST candle
    becomes a live proof; triggers found while catching up are logged, never traded."""

    def __init__(self, asset, variant=None):
        self.asset = str(asset).upper()
        self._first_call = True
        # B12: None = the tested rules (the only engine that can trade). "1H_BREAK" / "ALL_PROOFS" = exploration
        # engines for the practice lane: their proofs never reach state.proofs or the council.
        self.variant = variant
        self._log = _quiet_log if variant else logger

    @staticmethod
    def new_state():
        return {"v": 1, "last_1h": None, "hist": {"H": [], "L": []}, "hist_upto": None,
                "setups": [], "seen": [], "next_id": 1, "levels2": {}}

    # -- public ---------------------------------------------------------------------------
    def update(self, st, df1, df4, pcfg=None, cs=None, now=None, mp=None):
        out = {"state": st, "proofs": [], "head": None, "ladder": [], "setups": [], "brains": {},
               "processed": 0, "missed": [], "pkg4": [], "pkg_events": []}   # B15: 4H-clock entries and their events
        cfg = market_settings(self.asset, pcfg)
        self._main_cfg = cfg
        if cfg is not None and self.variant == "ALL_PROOFS":
            cfg = dict(cfg, cont_only=False, no_spike=False)     # B12 exploration: the filters' rejects as well
        self._cs_now = cs                       # display only: the brains at the moment of a proof
        if not isinstance(st, dict) or st.get("v") != 1:
            st = self.new_state()
        if cfg is not None:
            st = self._rules_check(st, cfg)          # B14 item 1.3: rebuild the memory after a rules change
        out["state"] = st
        if cfg is None or df1 is None or len(df1) < 30:
            return out
        self._map = mp if mp is not None else (_line_map_latest(self.asset) if cfg.get("b14") else None)   # B14
        self._nomap_n = 0                         # B14: signals judged without a map in this look (said once, below)
        df4 = hourly_to_4h(df1)                 # B12 (decision 35 A): never MT5's own, possibly stale, H4 series
        if len(df4) < 2 * K + 2:
            return out
        t1 = close_times(df1, 1)
        hi1, lo1, c1 = (df1[c].astype(float).values for c in ("high", "low", "close"))
        a1 = atr14(hi1, lo1, c1)
        self._lab = (t1, hi1, lo1, a1)               # B12: for the A+ labels (read-only)
        t4 = close_times(df4, 4)
        hi4, lo4, c4 = (df4[c].astype(float).values for c in ("high", "low", "close"))
        a4 = atr14(hi4, lo4, c4)
        self._lab4 = (t4, hi4, lo4)                  # B12: for the 4H-gap watch label (read-only)
        self._t4a4 = (t4, a4)                        # B12.1: the 4H move size at an entry (room check)
        if self.variant is None:
            self._brain4_step(st, t4, hi4, lo4, c4, self._lab[0][-1] if len(self._lab[0]) else None)  # B12.1 (57 B)
        close4 = dict(zip(t4, c4))
        atr4_at = dict(zip(t4, a4))
        bar4 = {t: (h, l, c) for t, h, l, c in zip(t4, hi4, lo4, c4)}
        self._close4, self._bar4, self._c4 = close4, bar4, c4      # B15: the 4H clock reads these
        sw = swings_4h(t4, hi4, lo4, c4)
        self._sw4 = sw                               # B13: the package's 'bigger line just ahead' check (read-only)
        self._sw4w = wick_swings_4h(t4, hi4, lo4)    # B13 item 33: the old high/low in front, judged on wicks
        self._dbrk4 = diagonal_breaks_4h(t4, a4, c4) if self.variant is None else []   # B13 items 31-32

        n1 = len(t1)
        _rev = self._revised(st, t1, hi1, lo1, c1)
        if _rev is not None:
            t_r, old_c, new_c = _rev
            snap = (st.get("snaps") or {}).get(str(t_r))
            if snap is not None:
                keep = {k: st.get(k) for k in ("snaps", "recent")}
                st.clear()
                st.update(copy.deepcopy(snap))
                st["snaps"] = {k: v for k, v in (keep["snaps"] or {}).items() if pd.Timestamp(k) < t_r}
                st["recent"] = [x for x in (keep["recent"] or []) if pd.Timestamp(x[0]) < t_r]
                out["revised"] = str(t_r)
                logger.warning("[NS-DATA] %s: the candle closing %s changed after it was used (close %.6g -> %.6g) -- "
                               "re-checking from there", self.asset, t_r, old_c, new_c)
            else:
                logger.warning("[NS-DATA] %s: the candle closing %s changed after it was used (close %.6g -> %.6g) -- "
                               "too old to re-check", self.asset, t_r, old_c, new_c)
        # HF-2 (28 Sep): a proof is only live if its candle closed within the last 2 hours. After a weekend or an
        # outage the newest candle can be days old -- its trigger is logged as missed, never traded late.
        if now is None:
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        fresh_emit = (now - t1[-1]) <= STALE_EMIT
        if not fresh_emit:
            self._log.info("[NS] %s: newest candle closed %s (%.1f h ago) -- nothing on it can be traded now",
                        self.asset, t1[-1], (now - t1[-1]).total_seconds() / 3600)
        if st["last_1h"] is None:
            first_i = max(int(t1.searchsorted(t1[-1] - timedelta(days=REPLAY_DAYS), side="left")), 14)
            cutoff = t1[min(first_i, n1 - 1)]
            for s in sw:                       # swings from before the window: history only
                if s[0] < cutoff:
                    self._register(st, s, atr4_at, create=False, emit=False)
        else:
            first_i = int(t1.searchsorted(st["last_1h"], side="right"))
        ptr = 0
        while ptr < len(sw) and st["hist_upto"] is not None and (sw[ptr][0], sw[ptr][1]) <= st["hist_upto"]:
            ptr += 1

        snaps = st.setdefault("snaps", {})
        for i in range(first_i, n1):
            t = t1[i]
            if i >= n1 - REV_K:                  # B12 (35 A): the state just before this candle, to rewind to
                snaps[str(t)] = copy.deepcopy({k: v for k, v in st.items() if k not in ("snaps", "recent")})
            emit = (i == n1 - 1) and fresh_emit
            if a1[i] == a1[i]:
                self._candle(st, cfg, i, t, emit, t1, hi1, lo1, c1, a1, close4, bar4, atr4_at, out)
            while ptr < len(sw) and sw[ptr][0] <= t:
                self._register(st, sw[ptr], atr4_at, create=True, emit=emit)
                ptr += 1
            st["last_1h"] = t
            out["processed"] += 1

        for k in sorted(snaps, key=lambda x: pd.Timestamp(x))[:-REV_K]:
            del snaps[k]
        st["recent"] = [[str(t1[j]), float(hi1[j]), float(lo1[j]), float(c1[j])] for j in range(max(0, n1 - REV_K), n1)]
        live = st["setups"]
        if out["processed"]:
            st["proofs_live"] = list(out["proofs"])
        else:
            # HF-2: a proof kept from an earlier call is only live while its candle is fresh -- a Friday proof must
            # not come back to life at the Sunday open (or after an outage)
            out["proofs"] = list(st.get("proofs_live", [])) if fresh_emit else []
        if out["processed"]:
            n_stage = [sum(1 for s in live if s["stage"] == k) for k in (0, 1, 2)]
            self._log.info("[NS] %s: %d candle(s) processed to %s | live setups %d (waiting break %d, retest %d, "
                        "trigger %d) | proofs %d", self.asset, out["processed"], t1[-1], len(live),
                        n_stage[0], n_stage[1], n_stage[2], len(out["proofs"]))
        if self._first_call:
            self._first_call = False
            self._log.info("[DEPLOY-HYGIENE] %s: new engine caught up %d candle(s); %d setup(s) live; "
                        "%d trigger(s) found while catching up were logged, not traded",
                        self.asset, out["processed"], len(live), len(out["missed"]))
        _a1l = float(a1[-1]) if a1[-1] == a1[-1] else None
        _a4l = float(a4[-1]) if a4[-1] == a4[-1] else None
        out["setups"] = [self._public(s, float(c1[-1]), _a1l, float(c4[-1]), _a4l) for s in live]
        out["head"] = self._head_fields(live, t1[-1])
        out["ladder"] = self._ladder(st, t1[-1], cs)
        out["proofs_hist"] = list(st.get("proofs_hist", []))
        out["brains"] = self._brains(cs, float(c1[-1]), float(a1[-1]) if a1[-1] == a1[-1] else None,
                                     float(a4[-1]) if a4[-1] == a4[-1] else None)
        return out

    # B14 item 1.3 (Desire 8 Oct): at the B13 restart every market kept memory built by the old rules ("caught up 0
    # candle(s)"), so GOLD's retired setup stayed dead. The memory now carries the stamp of the rules it was built
    # under; a different stamp starts it again from history (REPLAY_DAYS of candles; triggers found while catching up
    # are logged, never traded). Kept: the 4H brain (rule-free), its level history, the package's records, its
    # waiting entries and the chart's recent proofs.
    _KEEP_ON_REBUILD = ("brain4_m", "brain4_t", "brain4_prev", "brain_hist", "levels2", "pkg", "pkg_ready",
                        "proofs_hist", "brain4_states")

    def _rules_check(self, st, cfg):
        stamp = rules_stamp(cfg, self.variant)
        old = st.get("rules")
        if old == stamp:
            return st
        if old is None and (st.get("last_1h") is None or self.variant is not None):
            # a brand-new memory: nothing to rebuild. A practice-only exploration engine's rules are not changed by B14
            # (it keeps 25 days and the B13 rules), so its memory is only stamped; a later settings change rebuilds it.
            st["rules"] = stamp
            return st
        fresh = self.new_state()
        for k in self._KEEP_ON_REBUILD:
            if k in st:
                fresh[k] = st[k]
        fresh["next_id"] = max(int(st.get("next_id", 1) or 1), 1)
        fresh["rules"] = stamp
        fresh["rebuilt"] = str(pd.Timestamp.now(tz="UTC").tz_localize(None))
        # B15 Part 3 (Desire 12 Oct): a package record started under the old rules is closed with a clear line at the
        # switch; this rebuild re-creates its setup from history under the B15 rules (a small line: the 4H clock)
        if self._b15(cfg) is not None:
            for rec in fresh.get("pkg") or []:
                if rec.get("status") == "watching":
                    rec["status"], rec["why_done"] = "cancelled", "closed at the switch to the B15 rules"
                    self._log.warning("[PKG-CANCEL] %s: dir=%+d R2=%.5g (signal %s) -- closed at the switch to the B15 "
                                      "rules; the engine rebuilds this setup from history under B15",
                                      rec.get("asset", self.asset), int(rec.get("d", 0) or 0),
                                      float(rec.get("r2", 0.0) or 0.0), rec.get("tE"))
        self._rebuilt_from = ("its memory is from before B14 (no rules stamp)" if old is None else
                              "the rules or this market's settings changed since its memory was built")
        self._log.warning("[NS-REBUILD] %s: %s -- rebuilding the memory from the last %d days of candles "
                          "(triggers found while catching up are logged, never traded)",
                          self.asset, self._rebuilt_from, REPLAY_DAYS)
        self._first_call = True                          # so the catch-up count is logged ([DEPLOY-HYGIENE])
        return fresh

    @staticmethod
    def _revised(st, t1, hi1, lo1, c1):
        """(earliest revised candle time, old close, new close) or None -- a candle already used that now reads
        differently (B12, decision 35 A)."""
        for tr, h, l, c in st.get("recent") or []:
            t = pd.Timestamp(tr)
            j = int(t1.searchsorted(t, side="left"))
            if j >= len(t1) or t1[j] != t:
                continue
            for old, new in ((c, c1[j]), (h, hi1[j]), (l, lo1[j])):
                if abs(float(new) - float(old)) > REV_TOL * max(abs(float(new)), 1e-12):
                    return t, float(c), float(c1[j])
        return None

    # -- setups ---------------------------------------------------------------------------
    def _register(self, st, s, atr4_at, create, emit):
        conf, typ, lvl, edge = s
        hist = st["hist"]
        hist[typ].append((conf, lvl, edge))
        if len(hist[typ]) > 60:
            hist[typ] = hist[typ][-60:]
        st["hist_upto"] = (conf, typ)
        if not create:
            return
        if typ == "L" and len(hist["H"]) and hist["H"][-1][1] > lvl:
            H = hist["H"][-1]
            prev = hist["H"][-2][1] if len(hist["H"]) > 1 else None
            kind = "unknown" if prev is None else ("reversal" if prev > H[1] else "continuation")
            d, r2, edge2, r1 = 1, H[1], H[2], lvl
        elif typ == "H" and len(hist["L"]) and hist["L"][-1][1] < lvl:
            L = hist["L"][-1]
            prev = hist["L"][-2][1] if len(hist["L"]) > 1 else None
            kind = "unknown" if prev is None else ("reversal" if prev < L[1] else "continuation")
            d, r2, edge2, r1 = -1, L[1], L[2], lvl
        else:
            return
        if [d, round(r2, 6)] in st["seen"]:
            return
        a4 = atr4_at.get(conf)
        if a4 is None or not (a4 == a4) or a4 <= 0 or abs(r2 - r1) < MIN_SWING_ATR4 * a4:
            return
        _cfg = getattr(self, "_main_cfg", None) or {}
        if self.variant is None and (_cfg.get("quality_lines") or _cfg.get("reversal_rule", "none") != "none"):
            _why = self._line_check(st, _cfg, d, r2, r1, kind, conf)
            if _why:
                if emit:
                    self._log.info("[SETUP-SKIP] %s: NS %s dir=%+d R2=%.5g R1=%.5g -- %s", self.asset,
                                   KIND_MAP.get(kind, "TF_CONT"), d, r2, r1, _why)
                return
        sid = st["next_id"]
        st["next_id"] += 1
        st["setups"].append(dict(id=sid, d=d, r2=r2, edge=edge2, r1=r1, kind=kind, conf=conf, atr4=float(a4),
                                 stage=0, t_break=None, h2=None, t_touch=None,
                                 checks=getattr(self, "_checks", None) if self.variant is None else None))
        if emit:
            self._log.info("[SETUP-BORN] %s: NS %s dir=%+d R2=%.5g (zone to %.5g) R1=%.5g -- waiting for a 4H close past R2",
                        self.asset, KIND_MAP.get(kind, "TF_CONT"), d, r2, edge2, r1)
            self._log.info("[R1-ORIGIN] %s: NS %s dir=%+d H=%.5g R1=%.5g tag=SWING_4H tests=0 gap=%.2fATR4 (%s)",
                        self.asset, KIND_MAP.get(kind, "TF_CONT"), d, r2, r1, abs(r2 - r1) / a4, kind)
            if st["setups"][-1].get("checks"):
                self._log.info("[NS-QUALITY] %s: R2=%.5g -- %s", self.asset, r2, st["setups"][-1]["checks"])

    def _brain4_step(self, st, t4, hi4, lo4, c4, t_last1h=None):
        """B12.1 (Desire 30 Sep, decision 57 B): the engine keeps its OWN 4H brain -- the fixed Livermore machine,
        fed every closed 4H candle once, saved with the engine's memory (so restarts keep it) -- and records every
        level it sets, dated by the candle that set it, exactly as the research does (Tests 7a/8a, rules 48/50/54).
        After a reset it replays all the 4H candles on hand, so there is no blind month. A 4H candle is fed only once
        the NEXT hourly candle has closed (settled), so a stale candle corrected on the next read (35 A) never
        reaches the brain -- a level becomes usable one hour later; its date stays the candle that set it."""
        lsm, piv = _lsm_mod()
        if lsm is None or not len(t4):
            return
        m, last = st.get("brain4_m"), st.get("brain4_t")
        if m is None:
            m = lsm.make_livermore_pair(self.asset, piv.get(self.asset, {}))[0]
            st["brain4_m"], st["brain4_prev"], last = m, {}, None
            st["brain_hist"] = [x for x in st.get("brain_hist", []) if x[0] != "4H"]
        fr = pd.DataFrame({"high": np.asarray(hi4, float), "low": np.asarray(lo4, float), "close": np.asarray(c4, float)},
                          index=pd.DatetimeIndex(t4))
        atr = lsm.atr14(fr).values
        prev = st.setdefault("brain4_prev", {})
        new = 0
        for k in range(len(t4)):
            tk = pd.Timestamp(t4[k])
            if last is not None and tk <= last:
                continue
            if t_last1h is not None and tk >= pd.Timestamp(t_last1h):
                break                                   # not settled yet: the next hourly candle hasn't closed
            last = tk
            try:
                s_ = m.update(float(c4[k]), float(atr[k]))
            except Exception:
                continue
            # B15 (Desire 12 Oct, "test it alongside"): the brain's state after each 4H candle, dated by that candle, so a
            # break can be tagged with the phase the bot knew at that moment (box test and live alike)
            _hs = st.setdefault("brain4_states", [])
            _hs.append((str(tk), str(getattr(s_.state, "value", s_.state))))
            if len(_hs) > BRAIN4_KEEP:
                del _hs[:-BRAIN4_KEEP]
            up = s_.state in UP4
            for key, v in (("up", s_.anchor_main_up_max if up else None), ("down", None if up else s_.anchor_main_down_min),
                           ("nlow", s_.anchor_natural_low), ("nhigh", s_.anchor_natural_high)):
                if v is not None and v == v and prev.get(key) != v:
                    st.setdefault("brain_hist", []).append(("4H", float(v), tk))
                    prev[key] = v
                    new += 1
        st["brain4_t"] = last

    def _line_check(self, st, cfg, d, r2, r1, kind, conf):
        """B12.1 (decisions 48 / 50): None if the setup may live, else why not. Only levels known before the setup.
        Quality line = within half a 1H move of an EARLIER 4H top (buy) / bottom (sell) of the last 30 days, or of a
        4H brain level seen in the last 30 days. Switch (50): a reversal must turn (R1) at a 4H brain level."""
        t1, _h, _l, a1 = self._lab
        self._checks = None
        conf = pd.Timestamp(conf)
        ic = int(t1.searchsorted(conf, side="left"))
        a1c = float(a1[ic]) if 2 <= ic < len(t1) else float("nan")
        if not (a1c == a1c and a1c > 0):
            return "no 1H move size at the setup's birth"
        band = LEVEL_NEAR_ATR * a1c
        brain = [float(v) for tf_, v, since in st.get("brain_hist", [])
                 if tf_ == "4H" and pd.Timestamp(since) < conf and conf - pd.Timestamp(since) <= timedelta(days=30)]
        notes = []
        if cfg.get("quality_lines"):
            same = "H" if d == 1 else "L"
            held = any(pd.Timestamp(c) < conf and conf - pd.Timestamp(c) <= timedelta(days=30)
                       and 1e-9 * max(1.0, abs(r2)) < abs(float(lv) - r2) <= band for c, lv, _e in st["hist"][same])
            onb = any(abs(v - r2) <= band for v in brain)
            if not (held or onb):
                return "not a quality line (no earlier 4H %s and no 4H brain level within %.5g)" % (
                    "top" if d == 1 else "bottom", band)
            notes.append("quality line: %s" % " + ".join(x for x, y in (("held before", held), ("4H brain level", onb)) if y))
        rule = cfg.get("reversal_rule", "none")
        if cfg.get("b14") and self.variant is None:     # the practice-only exploration engines keep the B13 check
            # B14 Q6 A / Q7 A (Desire 9 Oct): a reversal is no longer thrown away at birth. Its turning point is judged
            # at the signal on the line map (inside the sphere of a major line or an average), and a reversal that
            # fails goes to the package instead. BTC uses the same rule (its 2-of-3 check goes).
            rule = "none"
        if kind == "reversal" and rule in ("brain4", "vote2"):
            c1_ = any(abs(v - r1) <= band for v in brain)                                  # on a 4H brain level
            if rule == "brain4":
                if not c1_:
                    return "a reversal whose turning point (R1) is not at a 4H brain level"
                notes.append("reversal: turning point on a 4H brain level")
            else:                                                                          # BTC (decision 54): 2 of 3
                ty = "L" if d == 1 else "H"                                                # R1 = a swing low (buy) / high (sell)
                c2_ = any(pd.Timestamp(c) < conf and conf - pd.Timestamp(c) <= timedelta(days=30)
                          and abs(float(lv) - r1) <= band for c, lv, _e in st["hist"][ty])  # the turning point held before
                y = _yday_side(t1, _h, _l, conf, d)                                         # yesterday's low (buy) / high (sell)
                c3_ = y is not None and abs(y - r1) <= band
                n = int(c1_) + int(c2_) + int(c3_)
                txt = "4H brain level %s, held before %s, yesterday's %s %s" % (
                    "yes" if c1_ else "no", "yes" if c2_ else "no", "low" if d == 1 else "high", "yes" if c3_ else "no")
                if n < 2:
                    return "a reversal whose turning point passes %d of 3 checks (%s) -- needs 2" % (n, txt)
                notes.append("reversal: %d of 3 (%s)" % (n, txt))
        self._checks = "; ".join(notes) or None
        return None

    def _room_ok(self, st, cfg, d, e, t, atr1):
        """B12.1 (decision 48): True if no prominent level -- a 4H brain level, a level that held twice (2+ 4H
        swings of the last 30 days within half a 1H move) or yesterday's low (sell) / high (buy) -- lies within
        room_atr 4H moves ahead of the entry."""
        t4, a4 = self._t4a4
        T = pd.Timestamp(t)
        k4 = int(t4.searchsorted(T, side="right")) - 1
        a4t = float(a4[k4]) if k4 >= 0 else float("nan")
        if not (a4t == a4t and a4t > 0):
            return True
        w = float(cfg.get("room_atr", 1.0)) * a4t
        lo, hi = (e - w, e) if d == -1 else (e, e + w)
        inside = (lambda x: lo <= x < hi) if d == -1 else (lambda x: lo < x <= hi)
        for tf_, v, since in st.get("brain_hist", []):
            if tf_ == "4H" and pd.Timestamp(since) < T and T - pd.Timestamp(since) <= timedelta(days=30) and inside(float(v)):
                return False
        be = LEVEL_NEAR_ATR * float(atr1) if atr1 == atr1 and atr1 > 0 else 0.0
        lv = [float(x) for c, x, _e in st["hist"]["L" if d == -1 else "H"]
              if pd.Timestamp(c) < T and T - pd.Timestamp(c) <= timedelta(days=30)]
        if any(inside(x) and sum(1 for y in lv if abs(y - x) <= be) >= 2 for x in lv):
            return False
        t1, hi1, lo1, _a = self._lab
        days = (pd.DatetimeIndex(t1) - pd.Timedelta(hours=1)).normalize()
        cand = days[days <= T.normalize() - pd.Timedelta(days=1)]
        if len(cand):
            m = np.asarray(days == cand.max())
            if inside(float(lo1[m].min()) if d == -1 else float(hi1[m].max())):
                return False
        return True

    def _end(self, st, s, reason, emit, t):
        st["setups"] = [x for x in st["setups"] if x["id"] != s["id"]]
        if emit:
            self._log.info("[MEASURE-8.4-DEATH] %s: kind=%s dir=%+d age_at_death=%s reason=%s",
                        self.asset, KIND_MAP.get(s["kind"], "TF_CONT"), s["d"],
                        int((t - s["conf"]).total_seconds() // 3600), reason)

    def _candle(self, st, cfg, i, t, emit, t1, hi1, lo1, c1, a1, close4, bar4, atr4_at, out):
        c, atr = float(c1[i]), float(a1[i])
        r4 = close4.get(t)
        if r4 is None and t.hour % 4 == 0:
            r4 = c                              # the 4H candle closing now closes at this 1H close
        killed = []
        for s in sorted(list(st["setups"]), key=lambda x: (x["conf"], x["id"])):
            if not any(x["id"] == s["id"] for x in st["setups"]):
                continue
            d, r2, r1 = s["d"], s["r2"], s["r1"]
            if r4 is not None and ((d == 1 and r4 < r1) or (d == -1 and r4 > r1)):
                killed.append("%+d R1=%.5g" % (d, r1))
                if s.get("clock") == "4H":
                    self._note4(s, "cancelled", "setup died: a 4H close past its origin %.5g" % r1, emit, out)
                self._end(st, s, "REF_INVALIDATED", emit, t)
                continue
            if s.get("clock") == "4H":                   # B15: this setup is proven on the 4H clock
                self._candle4(st, cfg, s, i, t, emit, t1, hi1, lo1, a1, r4, atr4_at, out)
                continue
            if s["stage"] == 0:
                if t - s["conf"] > WAIT_BREAK:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                rb = c if self.variant == "1H_BREAK" else r4     # B12 exploration: the break on any 1H close
                _h1 = self.variant is None and cfg.get("break_rule") == "big_or_1h_hold"
                _conf1h = False
                if _h1 and rb is None and s.get("pend1h") is not None:
                    # B12.1 (Desire 30 Sep, 1b): the NEXT 1H candle after a small break must close past the line --
                    # then the break counts at this 1H close; if not, that small break doesn't count (the line stays).
                    s.pop("pend1h", None)
                    rb, a4b = c, (s.get("atr4") or 0.0)
                    broke = d * (c - r2) > 0
                    how = ("BREAK (confirmed: the next 1H candle held past the line)" if broke
                           else "no (the next 1H candle closed back inside the line)")
                    s["break_kind"] = "small, confirmed by the next 1H candle" if broke else None
                    _conf1h = broke
                    if emit:
                        self._log.info("[COUNT-1-CHECK] %s NS dir=%+d tf=1H close=%.5g H=%.5g -> %s", self.asset, d, c, r2, how)
                    if not broke:
                        continue
                elif rb is None:
                    continue
                # B12.1 (Desire 30 Sep, decision 48): a BIG break (a 4H close at least big_break_atr 4H moves past R2)
                # counts now; a SMALL one counts only if the NEXT 4H candle closes further past. Live engine only --
                # the practice-lane ideas keep their tested rule. (Replaces the 29 Sep clear-break margin.)
                a4b = atr4_at.get(t) or s.get("atr4") or 0.0
                _kn14 = cfg.get("b14_knobs") if (cfg.get("b14") and self.variant is None) else None
                if _kn14 and _kn14.get("sig_break") == "mark" and getattr(self, "_map", None) is not None:
                    # box test only (4.5 / 9.1): the signal line breaks at the 3/4 mark too -- a quarter of a 4H move
                    # past a major 4H line, an eighth past any other
                    from src.execution import line_map as LM
                    _big_l = self._map.is_major_4h(d, r2, s["conf"], t)
                    _mk = (LM.SPH_BIG if _big_l else LM.SPH_SMALL) * LM.BREAK_FRAC * a4b
                    broke = a4b > 0 and d * (rb - r2) >= _mk
                    how = "BREAK (past the 3/4 mark)" if broke else "no (not past the 3/4 mark)"
                    s["break_kind"] = "3/4 mark" if broke else None
                elif _conf1h:
                    pass                                             # confirmed on the 1H candle above
                elif _h1:
                    if s.pop("pend1h", None) is not None and emit:   # a small break never got its next 1H candle (data gap)
                        self._log.info("[COUNT-1-CHECK] %s NS: the small break's next 1H candle never came -- it doesn't count", self.asset)
                    if a4b > 0 and d * (rb - r2) >= float(cfg.get("big_break_atr", 0.25)) * a4b:
                        broke, how = True, "BREAK (big)"
                        s["break_kind"] = "big"
                    else:
                        broke, how = False, "no"
                        if d * (rb - r2) > 0:
                            s["pend1h"] = float(rb)
                            how = "no -- small break: waiting for the next 1H candle to hold past the line"
                elif self.variant is None and cfg.get("break_rule") == "big_or_confirmed":
                    _p = s.pop("pend", None)
                    if _p is not None and d * (rb - _p) > 0:
                        broke, how = True, "BREAK (confirmed: this 4H candle closed further past than the small break)"
                    elif a4b > 0 and d * (rb - r2) >= float(cfg.get("big_break_atr", 0.25)) * a4b:
                        broke, how = True, "BREAK (big)"
                    else:
                        broke = False
                        how = "no (the next 4H candle did not close further past)" if _p is not None else "no"
                        if d * (rb - r2) > 0:
                            s["pend"] = float(rb)
                            how += " -- small break: waiting for the next 4H candle to close further past"
                else:
                    broke = d * (rb - r2) > 0
                    how = "BREAK" if broke else "no"
                if emit and not _conf1h:
                    a4 = atr4_at.get(t) or s["atr4"]
                    self._log.info("[COUNT-1-CHECK] %s NS dir=%+d tf=4H close=%.5g H=%.5g dist=%.2fATR4 "
                                "tier=SWING_4H -> %s", self.asset, d, rb, r2, d * (rb - r2) / a4, how)
                if not broke:
                    continue
                if [d, round(r2, 6)] in st["seen"]:
                    self._end(st, s, "NS_DUPLICATE", emit, t)
                    continue
                st["seen"].append([d, round(r2, 6)])
                if len(st["seen"]) > 400:
                    st["seen"] = st["seen"][-400:]
                j = i if self.variant == "1H_BREAK" else max(0, i - 3)
                s.update(stage=1, t_break=t, h2=float(c1[j:i + 1].max() if d == 1 else c1[j:i + 1].min()))
                _k = int(np.argmax(c1[j:i + 1]) if d == 1 else np.argmin(c1[j:i + 1]))      # display only below
                s.update(b_t=t, b_px=float(rb), b_a4=float(a4b), h2_t=t1[j + _k], gap2=bool(d * (rb - s["edge"]) <= 0))   # B12.1: + the break's 4H ATR (display)
                _kn15 = self._b15(cfg)
                if _kn15 is not None:                 # B15 (Desire 12 Oct): the line decides the clock
                    self._b15_clock(st, cfg, _kn15, s, i, t, float(rb), emit, out)
                    if s.get("clock") == "4H":
                        continue
                if cfg["entry"] == "B":
                    b = bar4.get(t)
                    strength = candle_strength(d, *b) if b else 0.5
                    self._candidate(st, cfg, s, "B", i, t, c, atr, strength, emit, out)
                    self._end(st, s, "NS_ENTERED", False, t)
                continue
            if cfg["entry"] == "E2":
                # B12 (Desire 28 Sep, decision 9): BTC's pause-then-turn entry, exactly as tested
                # (realistic_test.py:225-231). After the break, a candle closing against the trade marks a pause; the
                # first later close past R2 AND past the three closes before it is the entry. Checked before the
                # waits run out, in the same order as the research.
                if i >= 1 and d * (c - float(c1[i - 1])) < 0:
                    s["paused"] = True
                if s.get("paused") and i >= 3 and d * (c - r2) > 0 and \
                        d * (c - (float(np.max(c1[i - 3:i])) if d == 1 else float(np.min(c1[i - 3:i])))) > 0:
                    strength = candle_strength(d, hi1[i], lo1[i], c1[i])
                    self._candidate(st, cfg, s, "E2", i, t, c, atr, strength, emit, out)
                    self._end(st, s, "NS_ENTERED", False, t)
                    continue
            if s["stage"] == 1:
                if t - s["t_break"] > WAIT_TOUCH:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                if (d == 1 and lo1[i] <= s["edge"] + TOUCH_ATR * atr) or \
                        (d == -1 and hi1[i] >= s["edge"] - TOUCH_ATR * atr):
                    s.update(stage=2, t_touch=t, h2_frozen=s["h2"], touch_px=float(lo1[i] if d == 1 else hi1[i]),
                             gap1=bool(t - s["t_break"] <= timedelta(hours=1)))          # gap1: display only
                    if emit:
                        self._log.info("[COUNT-2] %s NS dir=%+d RETEST low/high=%.5g edge=%.5g peak=%.5g",
                                    self.asset, d, lo1[i] if d == 1 else hi1[i], s["edge"], s["h2"])
                    continue
                _kn14 = cfg.get("b14_knobs") if (cfg.get("b14") and self.variant is None and cfg["entry"] == "E"
                                                  and bool(cfg.get("package"))) else None
                if _kn14 and _kn14.get("runner") and getattr(self, "_map", None) is not None:
                    # B14 item 5.7 (Desire 8-9 Oct): the RUNNER entry -- price broke out and ran without a retest:
                    # 3 higher 1H closes (one dip of any size may sit between them; the run ends on a higher close),
                    # every candle clear of the line (a low within 1 move of it is a retest: the normal entry takes
                    # over above). Then the normal trigger: a 1H close past the best close so far. It needs a witness
                    # that broke the same way in the last 12 hours AND was retested -- the witness supplies the retest.
                    # With no witness the setup keeps waiting for a normal retest.
                    if s.get("run_ok") and d * (c - s["h2"]) > 0:
                        _w = self._map.witnesses(d, t, hours=float(_kn14["witness_h"]), need_retest=True)
                        if not _kn14.get("ma_witness", True):
                            _w = [w for w in _w if w["kind"] == "diagonal"]
                        if _w:
                            strength = candle_strength(d, hi1[i], lo1[i], c1[i])
                            if not self._candidate(st, cfg, s, "E", i, t, c, atr, strength, emit, out, runner_wits=_w):
                                self._end(st, s, "NS_ENTERED", False, t)
                            continue
                        if emit:
                            self._log.info("[RUNNER-WAIT] %s: dir=%+d R2=%.5g close=%.5g -- a run with no retest, but "
                                           "no witness broke and was retested in the last %gh -- waiting for a normal "
                                           "retest", self.asset, d, r2, c, float(_kn14["witness_h"]))
                    rn = s.setdefault("rn", {"last": None, "n": 0, "dip": False})
                    if rn["last"] is None or d * (c - rn["last"]) > 0:
                        rn["last"], rn["n"] = c, rn["n"] + 1
                    elif _kn14.get("runner_dip", True) and not rn["dip"] and rn["n"] < 3:
                        rn["dip"] = True                    # the one dip allowed between the three closes
                    elif rn["n"] < 3:
                        s["rn"] = {"last": c, "n": 1, "dip": False}     # a second dip: the run starts again here
                        rn = s["rn"]
                    if rn["n"] >= 3 and d * (c - rn["last"]) >= 0 and not s.get("run_ok"):
                        s["run_ok"] = True
                        if emit:
                            self._log.info("[RUNNER-RUN] %s: dir=%+d R2=%.5g -- 3 1H closes running the trade's way "
                                           "(%s each time) with no retest (last %.5g) -- a 1H close past the best close "
                                           "takes it, with a witness", self.asset, d, r2,
                                           "higher" if d == 1 else "lower", c)
                if d * (c - s["h2"]) > 0:
                    s["h2"] = c
                    s["h2_t"] = t                   # display only
                continue
            if s["stage"] == 2:
                if t - s["t_touch"] > WAIT_TRIGGER:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                # B15 rows S11/S12 (Desire 12 Oct, "consider it"): the close must beat the peak by trigger_clear moves
                _clr = float((self._b15(cfg) or {}).get("trigger_clear", 0.0) or 0.0)
                trig = d * (c - s["h2"]) > _clr * atr
                if emit:
                    self._log.info("[COUNT-3-CHECK] %s NS dir=%+d tf=1H close=%.5g H2=%.5g tol=%g dist=%.2fATR -> %s",
                                self.asset, d, c, s["h2"], _clr, d * (c - s["h2"]) / atr, "PROOF" if trig else "no")
                if trig:
                    strength = candle_strength(d, hi1[i], lo1[i], c1[i])
                    if not self._candidate(st, cfg, s, "E", i, t, c, atr, strength, emit, out):
                        self._end(st, s, "NS_ENTERED", False, t)
        if emit and killed:
            self._log.info("[KILL-R1] %s NS close4=%.5g -> %d setup(s) dead (%s)", self.asset, r4, len(killed),
                        ", ".join(killed))

    def _candidate(self, st, cfg, s, style, i, t, e, atr, strength, emit, out, runner_wits=None):
        d, r2 = s["d"], s["r2"]
        why = None
        if cfg["entry"] != style:
            return
        label = "RUNNER" if runner_wits else None     # B14 item 5.7: a run without a retest, taken with a witness
        too_far = d * (e - r2) / atr > FRESH_ATR
        if too_far:
            why = "too far from R2 (%.2f moves)" % (d * (e - r2) / atr)
        stop, target = ns_levels(d, e, r2, atr, cfg["min_sl_pct"], cfg["target_atr"])
        if why is None and (stop is None or not gate_rr_ok(e, stop, atr, cfg["min_rr"])):
            why = "no valid stop / R:R below %.2f" % cfg["min_rr"]
        f_cont = bool(cfg.get("cont_only") and s["kind"] != "continuation")
        f_spike = bool(cfg.get("no_spike") and style != "B" and strength > SPIKE)
        if why is None and f_cont:
            why = "filter: %s (continuations only)" % s["kind"]
        if why is None and f_spike:
            why = "filter: spike candle (closed at %.0f%% of its range)" % (100 * strength)
        # B13 (Desire 2 Oct): the package. A small line broken with a bigger 4H line just ahead, or a signal the
        # engine calls too far, is handed over instead of traded or retired. Everything else is unchanged.
        pkg_route = None
        diag_agrees, diag_t = self._diag_recent(d, t)           # B13 items 31-32 (Desire 6 Oct)
        b14 = None
        _b14_on = self.variant is None and style == "E" and bool(cfg.get("package")) and bool(cfg.get("b14"))
        if _b14_on and getattr(self, "_map", None) is None:
            _b14_on = False                                      # rule 13: never silent -- said once per look
            self._nomap_n = getattr(self, "_nomap_n", 0) + 1
            if self._nomap_n == 1:
                self._log.warning("[NS-MAP] %s: no line map -- the B14 checks can't run; signals in this look follow "
                                  "the B13 rules (see the [MAP] lines)", self.asset)
        if _b14_on and not f_cont and not f_spike:
            # B14 section 5 (Desire 8-9 Oct): every reason to wait is collected from the one line map; any reason to
            # wait wins (Q1 A) and the signal goes to the package. Otherwise it is taken at once.
            b14 = self._b14_route(cfg, s, t, e, atr, stop, target, runner_wits)
            if too_far:
                pkg_route, why = "far", None
            elif why is None and b14["wait"]:
                pkg_route = "wait"
            if why is None and not too_far:
                target = b14["target"]
        elif self.variant is None and style == "E" and bool(cfg.get("package")) and not f_cont and not f_spike:
            if too_far:
                pkg_route, why = "far", None
            elif why is None:
                _lvl, _edg, _dist = self._bigger_ahead(d, e, t)
                if _lvl is not None and _dist <= PKG_NEAR:
                    if diag_agrees:
                        # B13 item 32: the diagonal confirmation rule -- a small line broken with an old high (low) just
                        # ahead, but a 3-touch diagonal broke the same way in the last DIAG_RECENT_H hours: taken at once
                        if emit:
                            self._log.info("[DIAG-CONFIRM] %s: %s dir=%+d R2=%.5g entry=%.5g -- old 4H %s %.5g just ahead "
                                           "(%.2f 4H moves), but a diagonal broke the same way at %s -- taken at once",
                                           self.asset, style, d, r2, e, "high" if d == 1 else "low", _lvl, _dist, diag_t)
                    else:
                        pkg_route = "near"
                elif emit:
                    self._log.info("[PKG-ROUTE] %s: %s dir=%+d R2=%.5g entry=%.5g -- %s -- taken at once (space to run)",
                                   self.asset, style, d, r2, e,
                                   ("nearest old 4H %s %.5g is %.2f 4H moves ahead" % ("high" if d == 1 else "low", _lvl, _dist))
                                   if _lvl is not None else self._ahead_days_text(t))   # B14 item 1.2b
        room = None
        if self.variant is None and (cfg.get("quality_lines") or cfg.get("room_required")):
            room = self._room_ok(st, cfg, d, e, t, atr)          # B12.1 (decision 48): room ahead of the entry?
            if why is None and room is False and cfg.get("room_required"):
                why = "no room: a prominent level within %.2g 4H moves ahead (room switch on)" % cfg["room_atr"]
        if why is not None:
            if emit:
                self._log.info("[NS-SKIP] %s: %s dir=%+d R2=%.5g entry=%.5g -- %s -- retired",
                            self.asset, style, d, r2, e, why)
            return
        if self.variant == "ALL_PROOFS":
            _mc = getattr(self, "_main_cfg", None) or cfg
            _dropped = bool((_mc.get("cont_only") and s["kind"] != "continuation") or
                            (_mc.get("no_spike") and style != "B" and strength > SPIKE))
            if not _dropped:
                return                                   # the tested engine takes this one itself
        kind = KIND_MAP.get(s["kind"], "TF_CONT")
        tier = "BREAK" if style == "B" else "RETEST"
        peak = float(s.get("h2_frozen", s["h2"])) if style == "E" else float(e)
        a4 = float(s["atr4"])
        depth = None
        if style == "E" and abs(peak - s["r1"]) > 0:
            depth = abs(peak - float(s.get("touch_px", peak))) / abs(peak - s["r1"])
        fields = {
            "setup_active": True, "setup_kind": kind, "setup_dir": d,
            "setup_age": int((t - s["conf"]).total_seconds() // 3600), "setup_energy_trend": None,
            "setup_ref": float(r2), "setup_ref_tier": "SWING_4H", "setup_ref_tests": 0,
            "ref_1": float(s["r1"]), "ref_1_tests": 0, "brc_r1": float(s["r1"]), "brc_r1_tag": "SWING_4H",
            "brc_ref_tier": "SWING_4H", "brc_confirmed": True, "brc_direction": d, "brc_kind": kind,
            "brc_tier": tier, "brc_count": 3, "brc_h2": peak, "ref_h": peak,
            "brc_proof_dist_atr": d * (e - (peak if style == "E" else r2)) / a4,
            "brc_retest_depth": depth, "brc_age": 0, "brc_first_confirmed_ts": str(t), "brc_gear": "4H/1H",
            "ns_entry": style, "ns_exit": cfg["exit"], "ns_target_atr": cfg["target_atr"],
            "ns_r2": float(r2), "ns_edge": float(s["edge"]), "ns_r1": float(s["r1"]), "ns_atr1": float(atr),
            "ns_stop": float(stop), "ns_target": (float(target) if target else None), "ns_close": float(e),
            "ns_kind_raw": s["kind"], "ns_strength": float(strength), "ns_setup_id": int(s["id"]),
            "ns_diag_agrees": bool(diag_agrees), "ns_diag_break_t": diag_t,      # B13 item 31: the tag
            "ns_conf": str(s["conf"]), "ns_candle": str(t),
            # display only -- the proof card (steps, checks, brains at entry); never read by trading code
            "ns_b_t": str(s.get("b_t")), "ns_b_px": s.get("b_px"), "ns_b_atr4": s.get("b_a4"), "ns_break_kind": s.get("break_kind"), "ns_checks": s.get("checks"), "ns_h2_t": str(s.get("h2_t")),
            "ns_touch_t": str(s.get("t_touch")) if s.get("t_touch") is not None else None,
            "ns_touch_px": s.get("touch_px"), "ns_gap1_instant_retest": bool(s.get("gap1", False)),
            "ns_gap2_break_inside_zone": bool(s.get("gap2", False)),
            "ns_brain_4h": _cs_get(getattr(self, "_cs_now", None), "livermore_state_4h"),
            "ns_brain_4h_age_days": round(float(_cs_get(getattr(self, "_cs_now", None), "livermore_state_age_4h") or 0) * 4 / 24.0, 1),
            "ns_brain_1h": _cs_get(getattr(self, "_cs_now", None), "livermore_state_1h"),
            "ns_brain_1h_age_days": round(float(_cs_get(getattr(self, "_cs_now", None), "livermore_state_age_1h") or 0) / 24.0, 1),
        }
        fields.update(self._labels(st, i, d, e, atr, style, strength, s))   # B12: tracking labels only
        if room is not None:                           # B12.1 (decision 48): room is a tag unless switched on
            fields["ns_room"] = bool(room)
            (fields["ns_aplus"] if room else fields["ns_watch"]).append("room ahead" if room else "no room ahead")
        fields["ns_explore"] = self.variant
        if b14 is not None:
            # B14: what the line map said (the card, the weekly trace) and the switch the trade manager reads for the
            # lock and the staircase (section 6)
            from src.execution import line_map as LM
            _wb = self._map.walls_between(b14["b"], d, e, target) if target is not None else \
                self._map.walls_between(b14["b"], d, e, None, reach=4.0)
            fields.update({
                "ns_b14": True, "ns_b14_exits": True, "ns_label": label,
                "ns_wait": list(b14["wait"]), "ns_notes": list(b14["notes"]),
                "ns_route": "; ".join(b14["notes"]) or "space to run",
                "ns_witness": ("%s broke %s" % (b14["wits"][0]["name"], str(b14["wits"][0]["t_break"])))
                if b14["wits"] else None,
                "ns_target_kind": b14["target_kind"], "ns_wild": b14["wild"],
                "ns_walls": [{"near": B["near"], "s_near": B["s_near"], "s_far": B["s_far"], "label": B["label"]}
                             for B in _wb[:4]],
                "ns_lock_at": LM.lock_level(_wb, d, e),
            })
            _kb = cfg.get("b14_knobs") or {}
            if s.get("b15") and not s.get("big") and _kb.get("clock", "size") == "size":   # B15 Part 3: must never happen
                self._log.error("[B15-NEVER] %s: dir=%+d R2=%.5g -- a small line reached the hourly entry while B15 is on "
                                "(it belongs on the 4H clock) -- tell Claude", self.asset, s["d"], s["r2"])
            fields.update({                      # B15 (Desire 12 Oct): the exits switches, the path and the tags
                "ns_lock_on": bool(_kb.get("lock", True)), "ns_stair_on": bool(_kb.get("stair", True)),
                "ns_path": ("%s -- the hourly proof" % ("a big line (a major 4H swing)" if s.get("big") else "a small line"))
                if s.get("b15") else "the hourly proof (B14 rules)",
                "ns_clock": "1H", "ns_phase_4h": s.get("phase4"), "ns_stop_rule": "0.3 of a move behind the line",
            })
            if not _kb.get("lock", True):
                fields["ns_lock_at"] = None
            if label:
                fields["ns_aplus"].append("RUNNER entry (%s)" % fields["ns_witness"])
        if kind == "MR_REV":
            fields.update({"setup_active_mr": True, "setup_kind_mr": kind, "setup_dir_mr": d,
                           "setup_age_mr": fields["setup_age"], "setup_ref_mr": float(r2),
                           "setup_ref_tier_mr": "SWING_4H", "setup_ref_tests_mr": 0})
        proof = {"key": (self.asset, kind, d, round(float(r2), 8)), "kind": kind, "dir": d, "ref": float(r2),
                 "tests": 0, "first_ts": str(t), "fields": fields}
        if pkg_route is not None and b14 is not None and self._b15(cfg) is not None:
            # B15 (Desire 12 Oct): a big-line signal with a reason to wait is proven on the 4H clock (the package's
            # 4H/4H) by this engine -- the setup lives on (True = the caller must not end it)
            return self._route4(st, cfg, s, i, t, e, atr, b14, pkg_route, emit, out)
        if pkg_route is not None and not emit:
            # found while catching up -- never traded, exactly as before: a too-far signal was simply retired, a near
            # one is recorded as missed by the original path below
            if pkg_route == "far":
                return
            pkg_route = None
        _tier_log = "RUNNER" if label else tier            # B14 item 5.7: the "RUNNER" label in every market
        if pkg_route is not None:
            _lvl, _edg, _dist = self._bigger_ahead(d, e, t)
            rec = {"id": int(s["id"]), "asset": self.asset, "d": int(d), "kind": kind, "route": pkg_route,
                   "r2": float(r2), "r1": float(s["r1"]), "edge": float(s["edge"]), "trig": float(peak),
                   # (a RUNNER entry has no retest: its push is measured from the break)
                   "t_touch": str(s.get("t_touch") if s.get("t_touch") is not None else s.get("t_break")),
                   "tE": str(t), "eE": float(e), "atrE": float(atr),
                   "big": (float(_lvl) if _lvl is not None else None), "big_dist": (float(_dist) if _dist is not None else None),
                   "status": "watching", "last_t": None, "maj": None, "announced": False, "fields": fields}
            if b14 is not None:
                # B14: everything the package needs to judge the release (5.1 option c, the old line, the channel room)
                # and to re-arm the setup after a cancel (5.10 option A)
                _far = ["%.2f moves past the line (too far to take at once)" % (d * (e - r2) / atr)] if pkg_route == "far" else []
                rec.update({"b14": True, "why": _far + list(b14["wait"]), "label": label,
                            "walls": [{"near": B["near"], "far": B["far"], "s_near": B["s_near"], "mark": B["mark"],
                                       "label": B["label"]} for B in b14["walls"]],
                            "setup": copy.deepcopy(s),     # with its own types: a re-armed setup goes straight back
                            "rearm_n": int(s.get("rearms", 0) or 0)})
            st.setdefault("pkg", []).append(rec)
            self._log.info("[PKG-HANDOVER] %s: %s dir=%+d R2=%.5g entry=%.5g -- %s -- the package takes over",
                           self.asset, _tier_log, d, r2, e,
                           "; ".join(rec["why"]) if rec.get("b14") else
                           ("old 4H %s %.5g just ahead (%.2f 4H moves)" % ("high" if d == 1 else "low", _lvl, _dist))
                           if pkg_route == "near"
                           else "%.2f moves past the line (too far to take at once)" % (d * (e - r2) / atr))
            return
        if b14 is not None and emit:
            self._log.info("[NS-ROUTE] %s: %s dir=%+d R2=%.5g entry=%.5g -- taken at once: %s | target %s (%s)",
                           self.asset, _tier_log, d, r2, e, "; ".join(b14["notes"]) or "space to run",
                           ("%.5g" % target) if target is not None else "none", b14["target_kind"])
        _hist = st.setdefault("proofs_hist", [])                   # display only: the chart's recent proofs
        _hist.append(dict(proof, missed=not emit))
        if len(_hist) > 30:
            del _hist[:-30]
        if emit and self.variant:
            out["proofs"].append(proof)
            logger.info("[NS-EXPLORE] %s: %s -- %s %s dir=%+d R2=%.5g entry=%.5g stop=%.5g target=%s (practice lane only)",
                        self.asset, self.variant, tier, kind, d, r2, e, stop, ("%.5g" % target) if target else "runner")
        elif emit:
            out["proofs"].append(proof)
            self._log.info("[NS-PROOF] %s: %s %s dir=%+d R2=%.5g entry=%.5g stop=%.5g target=%s exit=%s (%s, %s)",
                        self.asset, _tier_log, kind, d, r2, e, stop,
                        ("%.5g" % target) if target else "none (runner)", cfg["exit"], s["kind"], str(t))
        else:
            out["missed"].append(proof)
            self._log.info("[NS-MISSED] %s: %s dir=%+d R2=%.5g at %s -- found while catching up, not traded",
                        self.asset, tier, d, r2, str(t))

    # -- B15 (Desire 12 Oct): the 4H clock -- the package's 4H/4H, proven inside the engine ----------------------------
    # Inside the engine (not as package records) so a waiting setup survives a restart, a rebuild and a replay exactly
    # like an hourly one. A live 4H entry still goes on the package's board, so the package's machinery places and
    # reports it: [PKG-ENTER] -> [PKG-OUTCOME], the freshness exemption, the trade manager's package stop.
    _DOWN15 = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")     # livermore_state_machine.DOWN_STATES

    def _b15(self, cfg):
        """The knobs when the B15 rules apply to this engine, else None: the live engine with entry E and the package,
        and either the B14 switch on with a clock other than "1h" (B14's all-hourly proofs, kept for the box test), or
        box-test row S6 (the B13 rules + the 4H clock for small lines, nothing else)."""
        if self.variant is not None or cfg is None or cfg.get("entry") != "E" or not cfg.get("package"):
            return None
        if cfg.get("b14"):
            kn = cfg.get("b14_knobs") or {}
            return kn if kn.get("clock", "size") != "1h" else None
        if cfg.get("b15_small_only"):
            return dict(B14_KNOBS, clock="size", channel_target=False, _small_only=True)
        return None

    @staticmethod
    def _phase_at(st, t):
        """The engine's own 4H brain state as the bot knew it at time t: the state after the last 4H candle fed before
        t (a 4H candle is fed an hour after it closes, so at a 4H close the newest state is the candle before)."""
        hs = st.get("brain4_states") or []
        if not hs:
            return None
        T = pd.Timestamp(t)
        try:                                       # the history is in time order: a binary search (same answer, fast)
            ks = pd.DatetimeIndex([x[0] for x in hs])
            if ks.is_monotonic_increasing:
                j = int(ks.searchsorted(T, side="left"))
                return hs[j - 1][1] if j > 0 else None
        except Exception:
            pass
        for tk, sv in reversed(hs):                # (out of time order -- never expected): the plain scan
            if pd.Timestamp(tk) < T:
                return sv
        return None

    def _b15_clock(self, st, cfg, kn, s, i, t, rb, emit, out):
        """At the break the line decides the clock. A big line -- a major 4H swing (the highest/lowest 4H close for 6
        candles each side), known by now -- keeps the hourly proof. A small line is proven on the 4H clock: a 4H retest,
        then a 4H close past the peak. Box-test rows S8/S9: the 4H brain's phase chooses instead (with the trend = an
        UP state for a buy, a DOWN state for a sell; anything else counts as against)."""
        d, r2 = s["d"], s["r2"]
        if getattr(self, "_map", None) is None:
            self._nomap_n = getattr(self, "_nomap_n", 0) + 1
            if self._nomap_n == 1:
                self._log.warning("[NS-MAP] %s: no line map -- the B15 clock can't be chosen; breaks in this look keep "
                                  "the hourly proof (see the [MAP] lines)", self.asset)
            return
        big = bool(self._map.is_major_4h(d, r2, s["conf"], t))
        ph = self._phase_at(st, t)
        withp = bool((d == 1 and ph in UP_STATES) or (d == -1 and ph in self._DOWN15))
        mode = kn.get("clock", "size")
        if mode == "phase_all":
            four = not withp
        elif mode == "phase_small":
            four = (not big) and (not withp)
        else:
            four = not big
        size = "a big line (a major 4H swing)" if big else "a small line"
        path = "%s -- %s" % (size, "the 4H clock (a 4H retest, then a 4H close past the peak)" if four
                             else "the hourly proof")
        if mode in ("phase_all", "phase_small"):
            path += " | 4H brain %s, %s the trade" % (ph or "unknown", "with" if withp else "against")
        s.update(b15=True, big=big, phase4=ph, phase_with=withp)
        if emit:
            self._log.info("[NS-PATH] %s: dir=%+d R2=%.5g broke at %s -- %s", self.asset, d, r2, str(t), path)
        if not four:
            return
        s.update(clock="4H", stage=1, h2=float(rb), h2_t=t, fails=0, route4="small" if not big else "phase",
                 why4=[path], path4=path)
        for k in ("h2_frozen", "t_touch", "touch_px", "pend1h", "pend", "rn", "run_ok", "gap1"):
            s.pop(k, None)
        if kn.get("pkg_majority"):
            n = self._vote4(d, t)                       # box-test row S7: the bigger-timeframe vote, once
            s["maj4"] = n
            if n < 2:
                self._end4(st, s, "no bigger-timeframe majority (%d of 3) -- not proven (box-test row S7)" % n,
                           emit, t, out)
                return
        _lvl, _edg, _dist = self._bigger_ahead(d, float(rb), t)
        if _lvl is not None and _dist is not None and _dist <= PKG_NEAR:
            s["ahead4"] = float(_lvl)                 # D5 (Desire 2 Oct, kept): the entry close must clear it too
        self._handover4(s, emit, out, "line %.5g broken -- %s%s" % (
            r2, size, ("; old 4H %s %.5g just ahead: the entry close must clear it too" % (
                "high" if d == 1 else "low", s["ahead4"])) if s.get("ahead4") is not None else ""))

    def _vote4(self, d, t):
        """Box-test row S7 only: the package's bigger-timeframe check (weekly / daily / 4H, 2 of 3 WITH the trade)."""
        from src.execution import ns_package as NP
        mp, n = self._map, 0
        for fr, sp in ((getattr(mp, "w1", None), NP._SPAN["W1"]), (getattr(mp, "d1", None), NP._SPAN["D1"])):
            try:
                n += int(fr is not None and NP._bigger_read(fr, sp, t, d) == 1)
            except Exception:
                pass
        try:
            n += int(bool(NP._strong_4h(pd.DatetimeIndex(self._t4a4[0]), np.asarray(self._c4, float), t, d)))
        except Exception:
            pass
        return n

    def _route4(self, st, cfg, s, i, t, e, atr, b14, pkg_route, emit, out):
        """A big-line signal with a reason to wait (too far, an old high/low just ahead with no witness, a reversal not
        turning at a major line, a wild market with no witness, a channel too tight) is proven on the 4H clock: a 4H
        close past the peak -- and past the old high/low, for that reason -- with no 4H close back past the line first.
        The hourly retest counts as the 4H retest when its 4H candle closed on the right side of the line. No late
        entries: a 4H close that already beat the peak before this moment only raises the peak."""
        d, r2 = s["d"], s["r2"]
        why = (["%.2f moves past the line (too far to take at once)" % (d * (e - r2) / atr)] if pkg_route == "far"
               else []) + list(b14.get("wait") or [])
        close4 = getattr(self, "_close4", None) or {}
        tb, T = pd.Timestamp(s["t_break"]), pd.Timestamp(t)
        tt = pd.Timestamp(s["t_touch"]) if s.get("t_touch") is not None else None
        t4r = ((tt - pd.Timedelta(hours=1)).floor("4h") + pd.Timedelta(hours=4)) if tt is not None else None
        prior = [float(v) for k, v in close4.items() if tb <= k and (t4r is None or k < t4r) and k <= T]
        peak4 = (max(prior) if d == 1 else min(prior)) if prior else float(s.get("b_px") or s["h2"])
        s.update(clock="4H", b15=True, big=True, route4=pkg_route, why4=why, routed_t=T, fails=0, h2=peak4, h2_t=tb,
                 path4="a big line (a major 4H swing) -- sent to wait (%s) -- the 4H clock (a 4H close past the peak)"
                 % "; ".join(why))
        for k in ("h2_frozen", "t_touch4", "rn", "run_ok"):
            s.pop(k, None)
        stage = 1
        if t4r is not None and t4r <= T and t4r in close4:
            if d * (float(close4[t4r]) - r2) > 0:
                stage, s["t_touch4"], s["h2_frozen"] = 2, t4r, peak4
            else:
                s["fails"] = 1                         # the retest's own 4H candle closed back past the line
        s["stage"] = stage
        for k in sorted(x for x in close4 if (t4r if t4r is not None else tb) < x <= T):
            v = float(close4[k])
            if d * (v - r2) < 0:
                s["fails"] = int(s.get("fails", 0)) + 1
                s["stage"] = 1
                s.pop("t_touch4", None)
            elif d * (v - s["h2"]) > 0:
                s["h2"] = v                            # already past: a fresh 4H close past it is needed
        _lvl, _edg, _dist = self._bigger_ahead(d, e, t)
        if _lvl is not None and any(str(w).startswith("old 4H") for w in why):
            s["ahead4"] = float(_lvl)
        if emit:
            self._log.info("[NS-PATH] %s: dir=%+d R2=%.5g -- %s", self.asset, d, r2, s["path4"])
        self._handover4(s, emit, out, "big line %.5g -- %s" % (r2, "; ".join(why)))
        return True

    def _handover4(self, s, emit, out, text):
        """The package takes over on the 4H clock: a log line and a Telegram (live only; a replay stays silent)."""
        if not emit:
            return
        from src.execution import ns_package as NP
        d = s["d"]
        self._log.info("[PKG-HANDOVER] %s: 4H/4H dir=%+d R2=%.5g -- %s -- the 4H clock takes over (up to 7 "
                       "days)", self.asset, d, s["r2"], text)
        NP._emit(out.setdefault("pkg_events", []), {"type": "handover", "asset": self.asset, "dir": d, "text":
                 "<b>PACKAGE TAKES OVER -- %s %s (4H clock)</b>\n%s.\nProving on the 4H clock: a 4H retest, then a 4H "
                 "close past %.5g -- up to 7 days." % (self.asset, "BUY" if d == 1 else "SELL", text, s["h2"])})

    def _note4(self, s, kind, text, emit, out):
        """A line for a 4H-clock setup: 'watch' (each 4H close, log only), 'rearm' and 'cancelled' (log + Telegram).
        Every line carries '(signal <break time>)' like the package's own lines, so the watchdog can follow them."""
        if not emit:
            return
        tag = {"watch": "PKG-4H", "rearm": "PKG-REARM", "cancelled": "PKG-CANCEL"}[kind]
        self._log.info("[%s] %s: dir=%+d R2=%.5g (signal %s) -- %s%s", tag, self.asset, s["d"], s["r2"],
                       str(s.get("t_break")), text, " -- 4H clock decided" if kind == "cancelled" else "")
        if kind == "watch":
            return
        from src.execution import ns_package as NP
        NP._emit(out.setdefault("pkg_events", []), {
            "type": "cancelled" if kind == "cancelled" else "rearm", "asset": self.asset, "dir": s["d"],
            "text": "<b>PACKAGE %s -- %s %s (4H clock)</b>\n%s" % (
                "CANCELLED" if kind == "cancelled" else "ATTEMPT FAILED", self.asset, "BUY" if s["d"] == 1 else "SELL",
                text)})

    def _end4(self, st, s, why, emit, t, out):
        self._note4(s, "cancelled", why, emit, out)
        self._end(st, s, "PKG4_END", emit, t)

    def _fail4(self, st, s, kn, why, emit, t, out):
        """Re-arm on the 4H clock (your 9 Oct 'retest' version): the attempt fails; the setup waits for a fresh 4H
        retest, then a 4H close past the best close so far -- at most rearm_max times."""
        n = int(s.get("fails", 0) or 0) + 1
        mx = int(kn.get("rearm_max", 5))
        if not kn.get("rearm", True) or n > mx:
            self._end4(st, s, "%s -- %s" % (why, "not re-armed (box-test switch)" if not kn.get("rearm", True)
                                            else "all %d re-arms used" % mx), emit, t, out)
            return
        s.update(stage=1, fails=n)
        for k in ("t_touch4", "h2_frozen"):
            s.pop(k, None)
        self._note4(s, "rearm", "%s -- re-armed (%d of %d): waiting for a fresh 4H retest, then a 4H close past %.5g" % (
            why, n, mx, s["h2"]), emit, out)

    def _skip4(self, s, why, emit, out):
        """D8 option B (box-test row S13): no trade at this 4H close. The setup keeps waiting for the next 4H close past
        the same level -- no fresh retest, not counted as a failed attempt; the 7-day limit from the retest still runs."""
        self._note4(s, "watch", "%s -- no trade at this close; still waiting for the next 4H close past the peak %.5g"
                    % (why, s["h2"]), emit, out)

    @staticmethod
    def _level4(s, kn, a4t):
        """The 4H entry level: the peak, plus the clearance knob (rows S11/S12, in 4H moves), and past the old 4H
        high/low just ahead when there was one."""
        d = s["d"]
        lvl = s["h2"] + d * float(kn.get("trigger_clear", 0.0) or 0.0) * a4t
        if s.get("ahead4") is not None:
            lvl = max(lvl, s["ahead4"]) if d == 1 else min(lvl, s["ahead4"])
        return lvl

    def _candle4(self, st, cfg, s, i, t, emit, t1, hi1, lo1, a1, r4, atr4_at, out):
        """One setup on the 4H clock. Only 4H closes count: no hourly close, no 30m push, no bigger-timeframe vote.
        Retest = a later 4H candle whose low (high, for a sell) comes within 1 move of the zone AND that closes on the
        right side of the line. Peak = the best 4H close from the break up to the retest. Entry = the first 4H close
        past the peak (plus the clearance knob, and past the old 4H high/low just ahead when there was one). A 4H close
        back past the line fails the attempt and re-arms. Waits: the setup's own -- 7 days to the retest, 7 days to the
        entry (box-test row S10: 24 hours for a signal sent to wait)."""
        if r4 is None:
            return
        kn = self._b15(cfg) or cfg.get("b14_knobs") or {}
        d, r2 = s["d"], s["r2"]
        b = (getattr(self, "_bar4", None) or {}).get(t)
        if b is None:                          # a 4H block the 4H builder left out (a data gap): its last 4 hours
            b = (float(np.max(hi1[max(0, i - 3):i + 1])), float(np.min(lo1[max(0, i - 3):i + 1])), float(r4))
        h4, l4, _c = b
        atr = float(a1[i])
        if not (atr == atr and atr > 0):
            return
        a4t = float(atr4_at.get(t) or s.get("atr4") or 0.0)
        T = pd.Timestamp(t)

        def day(t0):
            return int((T - pd.Timestamp(t0)).total_seconds() // 86400) + 1
        if s.get("routed_t") is not None and kn.get("pkg_wait") == "24h" and \
                T - pd.Timestamp(s["routed_t"]) > timedelta(hours=24):
            self._end4(st, s, "not proven within 24 hours of being sent to wait (box-test row S10)", emit, t, out)
            return
        if s["stage"] == 1 and T - pd.Timestamp(s["t_break"]) > WAIT_TOUCH:
            self._end4(st, s, "no 4H retest within 7 days of the break", emit, t, out)
            return
        if s["stage"] == 2 and T - pd.Timestamp(s["t_touch4"]) > WAIT_TRIGGER:
            self._end4(st, s, "no 4H close past %.5g within 7 days of the retest" % self._level4(s, kn, a4t), emit, t, out)
            return
        if T <= pd.Timestamp(s["t_break"]):
            return                                   # the break candle itself is never its own retest
        if d * (r4 - r2) < 0:
            self._fail4(st, s, kn, "a 4H close back past the line %.5g (close %.5g)" % (r2, r4), emit, t, out)
            return
        if s["stage"] == 1:
            reach = s["edge"] + d * TOUCH_ATR * atr
            if (d == 1 and l4 <= reach) or (d == -1 and h4 >= reach):
                s.update(stage=2, t_touch4=T, t_touch=T - timedelta(hours=3), touch_px=float(l4 if d == 1 else h4),
                         h2_frozen=s["h2"])
                self._note4(s, "watch", "4H retest: the 4H candle reached %.5g and closed %.5g, on the right side of the "
                            "line -- now waiting for a 4H close past %.5g (up to 7 days)" % (
                                l4 if d == 1 else h4, r4, self._level4(s, kn, a4t)), emit, out)
                return
            if d * (r4 - s["h2"]) > 0:
                s["h2"], s["h2_t"] = float(r4), T     # the peak ratchets until the retest
            self._note4(s, "watch", "waiting for a 4H retest (a 4H candle within 1 move of %.5g that closes on the right "
                        "side); peak %.5g; day %d of 7" % (s["edge"], s["h2"], day(s["t_break"])), emit, out)
            return
        lvl = self._level4(s, kn, a4t)
        if d * (r4 - lvl) <= 0:
            self._note4(s, "watch", "waiting for a 4H close past %.5g (this one %.5g); day %d of 7 since the retest" % (
                lvl, r4, day(s["t_touch4"])), emit, out)
            return
        self._enter4(st, cfg, kn, s, i, t, float(r4), atr, candle_strength(d, h4, l4, r4), emit, t1, hi1, lo1, out)

    def _enter4(self, st, cfg, kn, s, i, t, q, atr, strength, emit, t1, hi1, lo1, out):
        """The 4H entry: the B14.1 package stop (behind the last 1H swing since the retest, 1-move floor, skipped over
        2.5 moves), the reward:risk gate, the normal target or a channel's other side (0.5R of room, else keep
        waiting). Live: onto the package's board (placed and reported like any package entry). Catching up: missed."""
        d, r2 = s["d"], s["r2"]
        stop, swing, j_sw, wide = package_stop(d, q, t1, hi1, lo1, i, s.get("t_touch") or s["t_break"], atr,
                                               cfg["min_sl_pct"])
        why = wide or (None if gate_rr_ok(q, stop, atr, cfg["min_rr"])
                       else "reward:risk below %.2f at the entry" % cfg["min_rr"])
        if why:
            mode = kn.get("wide", "rearm")              # D8 (Desire 12 Oct: test all three)
            if mode == "wait":
                self._skip4(s, why, emit, out)
            elif mode == "end":
                self._end4(st, s, "%s -- no trade, and the setup ends (D8 option C)" % why, emit, t, out)
            else:
                self._fail4(st, s, kn, why, emit, t, out)
            return
        tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr
        tkind = "normal"
        mp = getattr(self, "_map", None)
        if tgt is not None and kn.get("channel_target", True) and not kn.get("_small_only") and mp is not None:
            from src.execution import line_map as LM
            b = mp.bucket(t)
            lvl_c, ch = mp.channel_target(b, d, q)
            if lvl_c is not None and kn.get("ch_edge") == "line":
                lvl_c = lvl_c + d * LM.SPH_BIG * mp.a4_at(b)          # box-test row S4: the other side's line itself
            if lvl_c is not None and d * (lvl_c - tgt) < 0:
                room = d * (lvl_c - q) / max(abs(q - stop), 1e-12)
                if room < float(kn.get("room_r", 0.5)):
                    self._note4(s, "watch", "a 4H close past the peak at %.5g, but the %s's other side (%.5g) is only "
                                "%.2fR away -- needs %.1fR of room; still waiting" % (
                                    q, ch["kind"], lvl_c, max(room, 0.0), float(kn.get("room_r", 0.5))), emit, out)
                    return
                tgt, tkind = lvl_c, "the %s's other side (%s)" % (
                    ch["kind"], "the line itself" if kn.get("ch_edge") == "line" else "near edge of its sphere")
        fields = self._fields4(st, cfg, kn, s, i, t, q, atr, strength, stop, tgt, tkind, swing, j_sw, t1)
        kind = KIND_MAP.get(s["kind"], "TF_CONT")
        proof = {"key": (self.asset, kind, d, round(float(r2), 8)), "kind": kind, "dir": d, "ref": float(r2),
                 "tests": 0, "first_ts": str(t), "fields": fields}
        _hist = st.setdefault("proofs_hist", [])                   # display only: the chart's recent proofs
        _hist.append(dict(proof, missed=not emit))
        if len(_hist) > 30:
            del _hist[:-30]
        txt = ("4H/4H: the 4H close %.5g beat %.5g after the 4H retest of %s; stop %.5g (%.1f moves, behind the last 1H "
               "swing %.5g), target %s%s" % (q, self._level4(s, kn, 0.0), str(s.get("t_touch4"))[:16], stop,
                                              abs(q - stop) / atr, swing, ("%.5g" % tgt) if tgt is not None else "runner",
                                              (" (%s)" % tkind) if tkind != "normal" else ""))
        if emit:
            from src.execution import ns_package as NP
            NP.board_add(st, proof, t, self.asset, d, float(r2))
            self._log.info("[PKG-ENTER] %s: dir=%+d R2=%.5g (signal %s) -- %s -- 4H clock decided", self.asset, d, r2,
                           str(s.get("t_break")), txt)
            NP._emit(out.setdefault("pkg_events", []), {"type": "entered", "asset": self.asset, "dir": d, "text":
                     "<b>PACKAGE ENTERED -- %s %s (4H clock)</b>\n%s" % (self.asset, "BUY" if d == 1 else "SELL", txt)})
            out.setdefault("pkg4", []).append(proof)
        else:
            out["missed"].append(proof)
            self._log.info("[NS-MISSED] %s: 4H/4H dir=%+d R2=%.5g at %s -- found while catching up, not traded",
                           self.asset, d, r2, str(t))
        self._end(st, s, "NS_ENTERED", False, t)

    def _fields4(self, st, cfg, kn, s, i, t, q, atr, strength, stop, tgt, tkind, swing, j_sw, t1):
        """The proof fields of a 4H entry -- the same keys as an hourly entry (_candidate), marked as a package entry
        (ns_entry "PKG": its own stop in the trade manager, exempt from the 2.5-move check at sending)."""
        d, r2 = s["d"], s["r2"]
        kind = KIND_MAP.get(s["kind"], "TF_CONT")
        peak = float(s.get("h2_frozen", s["h2"]))
        a4 = float(s["atr4"])
        depth = None
        if abs(peak - s["r1"]) > 0 and s.get("touch_px") is not None:
            depth = abs(peak - float(s["touch_px"])) / abs(peak - s["r1"])
        diag_agrees, diag_t = self._diag_recent(d, t)
        wait_h = round((pd.Timestamp(t) - pd.Timestamp(s["t_break"])).total_seconds() / 3600.0, 1)
        fields = {
            "setup_active": True, "setup_kind": kind, "setup_dir": d,
            "setup_age": int((t - s["conf"]).total_seconds() // 3600), "setup_energy_trend": None,
            "setup_ref": float(r2), "setup_ref_tier": "SWING_4H", "setup_ref_tests": 0,
            "ref_1": float(s["r1"]), "ref_1_tests": 0, "brc_r1": float(s["r1"]), "brc_r1_tag": "SWING_4H",
            "brc_ref_tier": "SWING_4H", "brc_confirmed": True, "brc_direction": d, "brc_kind": kind,
            "brc_tier": "RETEST", "brc_count": 3, "brc_h2": peak, "ref_h": peak,
            "brc_proof_dist_atr": d * (q - peak) / a4 if a4 else None,
            "brc_retest_depth": depth, "brc_age": 0, "brc_first_confirmed_ts": str(t), "brc_gear": "4H/4H",
            "ns_entry": "PKG", "ns_exit": cfg["exit"], "ns_target_atr": cfg["target_atr"],
            "ns_r2": float(r2), "ns_edge": float(s["edge"]), "ns_r1": float(s["r1"]), "ns_atr1": float(atr),
            "ns_stop": float(stop), "ns_target": (float(tgt) if tgt is not None else None), "ns_close": float(q),
            "ns_kind_raw": s["kind"], "ns_strength": float(strength), "ns_setup_id": int(s["id"]),
            "ns_diag_agrees": bool(diag_agrees), "ns_diag_break_t": diag_t,
            "ns_conf": str(s["conf"]), "ns_candle": str(t),
            "ns_b_t": str(s.get("b_t")), "ns_b_px": s.get("b_px"), "ns_b_atr4": s.get("b_a4"),
            "ns_break_kind": s.get("break_kind"), "ns_checks": s.get("checks"), "ns_h2_t": str(s.get("h2_t")),
            "ns_touch_t": str(s.get("t_touch4")) if s.get("t_touch4") is not None else None,
            "ns_touch_px": s.get("touch_px"), "ns_gap1_instant_retest": False,
            "ns_gap2_break_inside_zone": bool(s.get("gap2", False)),
            "ns_brain_4h": _cs_get(getattr(self, "_cs_now", None), "livermore_state_4h"),
            "ns_brain_4h_age_days": round(float(_cs_get(getattr(self, "_cs_now", None), "livermore_state_age_4h") or 0) * 4 / 24.0, 1),
            "ns_brain_1h": _cs_get(getattr(self, "_cs_now", None), "livermore_state_1h"),
            "ns_brain_1h_age_days": round(float(_cs_get(getattr(self, "_cs_now", None), "livermore_state_age_1h") or 0) / 24.0, 1),
        }
        fields.update(self._labels(st, i, d, q, atr, "E", strength, s))
        if cfg.get("quality_lines") or cfg.get("room_required"):
            room = self._room_ok(st, cfg, d, q, t, atr)
            fields["ns_room"] = bool(room)
            (fields["ns_aplus"] if room else fields["ns_watch"]).append("room ahead" if room else "no room ahead")
        fields["ns_explore"] = None
        from src.execution import line_map as LM
        mp = getattr(self, "_map", None)
        _wb = (mp.walls_between(mp.bucket(t), d, q, tgt) if tgt is not None else
               mp.walls_between(mp.bucket(t), d, q, None, reach=4.0)) if mp is not None else []
        fields.update({
            "ns_b14": True, "ns_b14_exits": not kn.get("_small_only"), "ns_label": None,
            "ns_lock_on": bool(kn.get("lock", True)), "ns_stair_on": bool(kn.get("stair", True)),
            "ns_wait": list(s.get("why4") or []), "ns_notes": [],
            "ns_route": "package (4H clock): " + "; ".join(s.get("why4") or []),
            "ns_witness": None, "ns_target_kind": tkind, "ns_wild": None,
            "ns_walls": [{"near": B["near"], "s_near": B["s_near"], "s_far": B["s_far"], "label": B["label"]}
                         for B in _wb[:4]],
            "ns_lock_at": LM.lock_level(_wb, d, q) if kn.get("lock", True) else None,
            "ns_path": s.get("path4") or "the 4H clock", "ns_clock": "4H", "ns_phase_4h": s.get("phase4"),
            "ns_stop_rule": "0.3 of a move behind the last 1H swing %.5g%s (B14.1)" % (
                swing, (" at %s" % str(t1[j_sw])[:16]) if j_sw is not None else " (the push's extreme)"),
            "pkg_route": s.get("route4"), "pkg_how": "4H/4H", "pkg_wait_h": wait_h, "pkg_bigger": s.get("ahead4"),
            "pkg_majority": s.get("maj4"), "pkg_signal_t": str(s.get("t_break")), "pkg_signal_close": s.get("b_px"),
        })
        fields["ns_aplus"] = list(fields.get("ns_aplus") or []) + ["package 4H/4H, after %.1fh" % wait_h]
        if kind == "MR_REV":
            fields.update({"setup_active_mr": True, "setup_kind_mr": kind, "setup_dir_mr": d,
                           "setup_age_mr": fields["setup_age"], "setup_ref_mr": float(r2),
                           "setup_ref_tier_mr": "SWING_4H", "setup_ref_tests_mr": 0})
        return fields

    def _ahead_days_text(self, t):
        """B14 item 1.2b (Desire 8 Oct): the line used to say "in the last 60 days" when the engine only had 25 days
        of candles. It now says how many days were really searched."""
        t1 = getattr(self, "_lab", (None,))[0]
        have = (pd.Timestamp(t) - pd.Timestamp(t1[0])).total_seconds() / 86400.0 if t1 is not None and len(t1) else 0.0
        if have >= PKG_AHEAD_DAYS:
            return "no old 4H high/low ahead in the last %d days" % PKG_AHEAD_DAYS
        return "no old 4H high/low ahead in the %.0f days of candles on hand (%d wanted)" % (have, PKG_AHEAD_DAYS)

    def _b14_route(self, cfg, s, t, e, atr, stop, target, runner_wits=None):
        """B14 section 5 (Desire 8-9 Oct): read the one line map at the signal and list every reason to wait.
          - a big wall within half a 4H move ahead, unless the signal is STRONG (2 of the 3 kinds of line broke the
            same way and all are big -- a big horizontal with a big diagonal or a big average -- or all 3 kinds broke);
          - a reversal whose turning point is not inside the sphere of a major line or an average (Q6 A; BTC too, Q7 A);
          - an old 4H high/low within half a move ahead and no witness (with a witness: the fast lane);
          - a wild market (the 4H move at least 1.5x its 60-day average) and no witness;
          - inside a channel, the other side comes before the normal target with less than 0.5R of room.
        A witness = a diagonal or average that broke the same way (a 4H close past its 3/4 mark) in the 12 hours before
        the signal and has had no 1H close back across since. -> dict(wait, notes, wits, walls, strong, target, ...)"""
        from src.execution import line_map as LM
        mp, kn, d = self._map, cfg["b14_knobs"], s["d"]
        b = mp.bucket(t)
        a4 = mp.a4_at(b)
        wits = list(runner_wits) if runner_wits else mp.witnesses(d, t, hours=float(kn["witness_h"]))
        if not kn.get("ma_witness", True):                       # box test only: diagonal witnesses only
            wits = [w for w in wits if w["kind"] == "diagonal"]
        walls = mp.walls_ahead(b, d, e, LM.WALL_REACH) if a4 == a4 and a4 > 0 else []
        wait, notes, strong = [], [], None
        if walls and kn.get("wall_route", True):    # B15 (Desire 12 Oct): walls no longer route a signal
            strong, why_s = LM.strong_through_walls(mp, b, d, s["r2"], s["conf"], t, wits)
            if not kn.get("strong", True):                       # box test only: every big wall goes to the package
                strong, why_s = False, "strong buys switched off (box test)"
            W = walls[0]
            txt = "%s %.5g, %.2f 4H moves ahead" % (W["label"], W["near"], d * (W["near"] - e) / a4)
            if strong:
                notes.append("strong through the wall (%s; wall: %s)" % (why_s, txt))
            else:
                wait.append("big wall ahead (%s) and not strong: %s" % (txt, why_s))
        if s["kind"] == "reversal":
            if kn.get("turn") == "half1h":                      # box test: the tested half-hourly-move reading
                ok_t, nm = self._turn_half1h(mp, d, s["r1"], s["conf"])
            else:
                ok_t, nm = mp.turn_at(d, s["r1"], s["conf"])
            if ok_t:
                notes.append("reversal turned at a %s" % nm)
            else:
                wait.append("a reversal whose turning point %.5g is not inside the sphere of a major line or an "
                            "average" % s["r1"])
        _lvl, _edg, _dist = self._bigger_ahead(d, e, t)
        if _lvl is not None and _dist <= PKG_NEAR:
            _o = "old 4H %s %.5g, %.2f 4H moves ahead" % ("high" if d == 1 else "low", _lvl, _dist)
            if wits:
                notes.append("fast lane: %s, witness %s" % (_o, wits[0]["name"]))
            else:
                wait.append("%s, and no witness" % _o)
        wild, ratio = mp.wild(t)
        if wild:
            if wits:
                notes.append("wild market (4H move %.1fx its 60-day average), witness %s" % (ratio, wits[0]["name"]))
            else:
                wait.append("wild market (4H move %.1fx its 60-day average) and no witness" % ratio)
        tgt, tgt_kind = target, "normal"
        if target is not None and stop is not None and a4 == a4 and a4 > 0:
            if kn.get("channel_target", True):
                lvl_c, ch = mp.channel_target(b, d, e)
                if lvl_c is not None and kn.get("ch_edge") == "line":
                    lvl_c = lvl_c + d * LM.SPH_BIG * a4                 # box-test row S4: the other side's line itself
                if lvl_c is not None and d * (lvl_c - target) < 0:      # the channel's other side comes first
                    room = d * (lvl_c - e) / max(abs(e - stop), 1e-12)
                    if room < float(kn.get("room_r", 0.5)):
                        wait.append("the %s's other side (%.5g) is only %.2fR away -- needs %.1fR of room" % (
                            ch["kind"], lvl_c, max(room, 0.0), float(kn.get("room_r", 0.5))))
                    else:
                        tgt, tgt_kind = lvl_c, "the %s's other side (near edge of its sphere)" % ch["kind"]
            if not kn.get("tp_line_only", True) and tgt_kind == "normal":
                # box test only: take profit at the near edge of a big line's sphere that holds the normal target
                for B in mp.walls_between(b, d, e, target):
                    if d * (target - B["s_near"]) >= 0 and d * (target - B["mark"]) < 0:
                        tgt, tgt_kind = B["s_near"], "near edge of the %s's sphere" % B["label"]
                        break
        return dict(wait=wait, notes=notes, wits=wits, walls=walls, strong=strong, target=tgt, target_kind=tgt_kind,
                    wild=ratio, b=b, a4=a4)

    @staticmethod
    def _turn_half1h(mp, d, r1, conf):
        """Box test only (9.1 row "turning points"): the tested reading -- the turning point within half an hourly
        move of a major line or an average."""
        b = mp.bucket(conf)
        if b < 0:
            return False, ""
        m = mp.members(b)
        j = int(np.searchsorted(mp.t1.values, np.datetime64(pd.Timestamp(conf)), side="right")) - 1
        a1 = atr14(mp.hi1[max(0, j - 30):j + 1], mp.lo1[max(0, j - 30):j + 1], mp.c1[max(0, j - 30):j + 1])[-1] \
            if j >= 14 else float("nan")
        if not (a1 == a1 and a1 > 0):
            return False, ""
        own = 1 if d == 1 else 0                      # H4L / H4H codes in line_map
        for k in range(len(m["lv"])):
            cd = int(m["code"][k])
            if abs(m["lv"][k] - r1) > 0.5 * a1 or (cd == own and abs(m["lv"][k] - r1) <= 1e-9 * max(1.0, abs(r1))):
                continue
            if cd in (16, 17):
                continue
            if m["big"][k] or m["maj"][k] or 6 <= cd <= 11:
                return True, "line within half an hourly move"
        return False, ""

    def _bigger_ahead(self, d, e, t):
        """B13 (Desire 2 Oct): the nearest bigger 4H line ahead of price -- a 4H swing close from the last
        PKG_AHEAD_DAYS days, confirmed by time t -- and its distance in 4H moves. (None, None, None) when none."""
        sw = (getattr(self, "_sw4w", None) if AHEAD_ON_WICKS else getattr(self, "_sw4", None)) or []   # B13 item 33
        t4, a4 = getattr(self, "_t4a4", (None, None))
        if not sw or t4 is None or len(t4) == 0:
            return None, None, None
        tt = pd.Timestamp(t)
        k = int(np.searchsorted(pd.DatetimeIndex(t4).values, np.datetime64(tt), side="right")) - 1
        a = float(a4[k]) if k >= 0 else float("nan")
        if not (a == a and a > 0):
            return None, None, None
        typ, t0, best = ("H" if d == 1 else "L"), tt - pd.Timedelta(days=PKG_AHEAD_DAYS), None
        for cf, ty, lv, ed in sw:
            if ty != typ or cf > tt or cf < t0:
                continue
            if d * (lv - e) > 0 and (best is None or d * (lv - e) < d * (best[0] - e)):
                best = (lv, ed)
        if best is None:
            return None, None, None
        return best[0], best[1], d * (best[0] - e) / a

    def _diag_recent(self, d, t):
        """B13 items 31-32 (Desire 6 Oct): did a 3-touch diagonal break the same way in the DIAG_RECENT_H hours before
        t? Returns (True/False, the latest such break time or None)."""
        tt, best = pd.Timestamp(t), None
        for dd, tb in (getattr(self, "_dbrk4", None) or []):
            tb = pd.Timestamp(tb)
            if dd == d and tt - pd.Timedelta(hours=DIAG_RECENT_H) <= tb <= tt and (best is None or tb > best):
                best = tb
        return best is not None, (str(best) if best is not None else None)

    def _labels(self, st, i, d, e, atr, style, strength, s=None):
        """B12 (Desire 28 Sep, ruling 2): tracking labels -- never read by trading code.
        A+ = the entry sits within half a move of a 4H brain level known before this candle (30 days), or of an
        open 1H fair-value gap in the trade's direction (10 days). Watch = the two groups that lost in both
        halves of the tests: GOLD shorts, and BTC entries that were not spike candles."""
        aplus, watch = [], []
        try:
            t1, hi1, lo1 = self._lab[:3]
            t_prev = t1[i - 1] if i > 0 else t1[i]
            band = APLUS_NEAR_ATR * float(atr)
            for tf, v, since in st.get("brain_hist", []):
                if tf == "4H" and since <= t_prev and (t_prev - since) <= timedelta(days=BRAIN_HIST_DAYS) \
                        and abs(float(e) - float(v)) <= band:
                    aplus.append("4H brain level")
                    break
            for k in range(max(2, i - FVG_1H_BARS), i):
                if d == 1 and lo1[k] > hi1[k - 2]:
                    zl, zh = float(hi1[k - 2]), float(lo1[k])
                    if (lo1[k + 1:i] <= zl).any():
                        continue
                elif d == -1 and hi1[k] < lo1[k - 2]:
                    zl, zh = float(hi1[k]), float(lo1[k - 2])
                    if (hi1[k + 1:i] >= zh).any():
                        continue
                else:
                    continue
                if zl - band <= float(e) <= zh + band:
                    aplus.append("open 1H FVG")
                    break
        except Exception:
            pass
        # B12 (Desire 28 Sep, rulings 27, 28, 30, 32, 34): the Test 7-8 findings, measured exactly as tested.
        yday = None
        try:
            t1, hi1, lo1, a1 = self._lab
            if float(hi1[i]) - float(lo1[i]) >= BIG_CANDLE_ATR * float(atr):
                aplus.append("big entry candle")                                        # 30 A (Test 7c)
            if s is not None and s.get("kind") in ("reversal", "continuation"):
                ic = int(t1.searchsorted(pd.Timestamp(s["conf"]), side="left"))
                ac = float(a1[ic]) if 2 <= ic < len(t1) else float("nan")
                if ac == ac and ac > 0:
                    tp, hb = t1[ic - 1], LEVEL_NEAR_ATR * ac

                    def _brain(tf, days, x):
                        return any(tf_ == tf and since <= tp and (tp - since) <= timedelta(days=days)
                                   and abs(float(x) - float(v)) <= hb for tf_, v, since in st.get("brain_hist", []))
                    # B12.1 (Desire 29 Sep, decision 46 A): re-tested on the FIXED brain -- labels 27 ("reversal at 1H
                    # brain level") and 32 ("reversal breaks 4H brain level") no longer held and are gone; these two did:
                    if s["kind"] == "continuation":
                        if _brain("4H", 30, s["r2"]):
                            aplus.append("continuation breaks 4H brain level")          # 46 A (Test 8a, fixed brain)
                        raise StopIteration                    # the checks below are for reversals only
                    # B13 item 23 (Desire 5 Oct): "reversal at 4H brain level" retired -- the reversal rule already requires
                    # it, so it was true on every reversal and told nothing
                    t4, hi4, lo4 = self._lab4
                    k4 = int(t4.searchsorted(tp, side="right")) - 1
                    if k4 >= 2 and _open_4h_gap_near(hi4, lo4, k4, d == 1, float(s["r1"]), hb):
                        watch.append("reversal inside open 4H gap")                     # 28 A (Test 7a)
                    yday = _yday_hl_near(t1, hi1, lo1, pd.Timestamp(s["conf"]), float(s["r1"]), hb)   # 34 A
        except StopIteration:
            pass
        except Exception as _lab_err:
            if not getattr(self, "_lab_warned", False):          # rule 13: a broken label check must show up once
                self._lab_warned = True
                self._log.warning("[NS-LABELS] %s: label check failed (labels incomplete, trading unaffected): %s",
                                  self.asset, _lab_err)
        if self.asset == "GOLD" and d == -1:
            watch.append("GOLD short")
        if self.asset == "BTC" and style != "B" and float(strength) <= SPIKE:
            watch.append("BTC non-spike")
        return {"ns_aplus": aplus, "ns_watch": watch, "ns_yday_hl": yday}

    # -- outputs for the state, charts and dashboard --------------------------------------
    @staticmethod
    def _public(s, c1_last=None, a1_last=None, c4_last=None, a4_last=None):
        """Dashboard/scanner note: dist_atr/next_stage are display-only estimates
        computed from the last CLOSED candle -- they read the same numbers the
        [COUNT-1-CHECK]/[COUNT-2]/[COUNT-3-CHECK] log lines use, but never feed
        back into any trading decision (setup birth/stage/death logic above is
        untouched)."""
        d = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in s.items()}
        try:
            if s.get("clock") == "4H":                   # B15: a setup on the 4H clock -- its distances in 4H moves
                _a4 = float(s.get("atr4") or 0) or a4_last
                if s["stage"] == 1 and c4_last is not None and _a4:
                    d["dist_atr"] = round(s["d"] * (c4_last - s["edge"]) / _a4, 2)
                    d["next_stage"] = "4H retest"
                elif s["stage"] == 2 and c4_last is not None and _a4:
                    d["dist_atr"] = round(s["d"] * (s["h2"] - c4_last) / _a4, 2)
                    d["next_stage"] = "4H close past the peak"
                d["clock"] = "4H"
                return d
            if s["stage"] == 0 and c4_last is not None and a4_last:
                d["dist_atr"] = round(s["d"] * (s["r2"] - c4_last) / a4_last, 2)
                d["next_stage"] = "break"
            elif s["stage"] == 1 and c1_last is not None and a1_last:
                d["dist_atr"] = round(s["d"] * (c1_last - s["edge"]) / a1_last, 2)
                d["next_stage"] = "retest"
            elif s["stage"] == 2 and c1_last is not None and a1_last:
                d["dist_atr"] = round(s["d"] * (s["h2"] - c1_last) / a1_last, 2)
                d["next_stage"] = "trigger"
        except Exception:
            pass
        return d

    @staticmethod
    def _head_fields(live, now):
        if not live:
            return None
        s = sorted(live, key=lambda x: (x["stage"], x["conf"]))[-1]
        kind = KIND_MAP.get(s["kind"], "TF_CONT")
        return {"setup_active": True, "setup_kind": kind, "setup_dir": s["d"],
                "setup_age": int((now - s["conf"]).total_seconds() // 3600), "setup_ref": float(s["r2"]),
                "setup_ref_tier": "SWING_4H", "setup_ref_tests": 0, "ref_1": float(s["r1"]),
                "brc_r1": float(s["r1"]), "brc_r1_tag": "SWING_4H", "brc_ref_tier": "SWING_4H",
                "brc_count": int(s["stage"]), "brc_h2": s["h2"], "ref_h": s["h2"]}

    def _ladder(self, st, now, cs):
        rungs = []
        for typ in ("H", "L"):
            for conf, lvl, edge in st["hist"][typ]:
                if now - conf <= timedelta(days=LAYER1_DAYS):
                    rungs.append({"layer": 1, "tf": "4H", "type": typ, "close": float(lvl),
                                  "edge": float(edge), "since": str(conf)})
        lv2 = st.setdefault("levels2", {})
        for tf, suf in (("1H", "_1h"), ("4H", "")):
            for name in ("main_up_max", "main_down_min", "natural_high", "natural_low"):
                v = _cs_get(cs, "livermore_anchor_%s%s" % (name, suf))
                key = "%s %s" % (tf, name)
                if v is None or not (v == v):
                    continue
                if key not in lv2 or lv2[key][0] != float(v):
                    lv2[key] = (float(v), now)
                    if tf == "1H" or self.variant is not None:     # B12.1 (57 B): 4H levels come from _brain4_step
                        st.setdefault("brain_hist", []).append((tf, float(v), now))     # B12: labels and bounces
        st["brain_hist"] = [x for x in st.get("brain_hist", []) if now - x[2] <= timedelta(days=BRAIN_HIST_DAYS)]
        for key, (v, since) in list(lv2.items()):
            tf = key.split()[0]
            if now - since <= timedelta(days=LAYER2_DAYS[tf]):
                rungs.append({"layer": 2, "tf": tf, "type": key.split()[1], "close": v, "edge": v,
                              "since": str(since)})
        return rungs

    def _brains(self, cs, price, atr1, atr4):
        out = {}
        for tf, suf, atr, per_bar_h in (("1H", "_1h", atr1, 1), ("4H", "", atr4, 4)):
            state = _cs_get(cs, "livermore_state%s" % ("_1h" if tf == "1H" else "_4h"))
            age = _cs_get(cs, "livermore_state_age%s" % ("_1h" if tf == "1H" else "_4h")) or 0
            if state is None:
                continue
            up = state in UP_STATES
            above = _cs_get(cs, "livermore_anchor_%s%s" % ("main_up_max" if up else "natural_high", suf))
            below = _cs_get(cs, "livermore_anchor_%s%s" % ("natural_low" if up else "main_down_min", suf))
            days = float(age) * per_bar_h / 24.0

            def _mv(lvl):
                return (abs(lvl - price) / atr) if (lvl is not None and atr) else None
            out[tf] = {"state": state, "age_days": round(days, 1), "stale": days > STALE_DAYS[tf],
                       "flips_above": above, "moves_above": _mv(above),
                       "flips_below": below, "moves_below": _mv(below)}
        return out


def _open_4h_gap_near(hi4, lo4, k4, bull, x, band):
    """Test 7a's open 4H gap: formed in the last 180 4H candles up to k4, in the trade's direction, not filled by k4,
    and x within `band` of it."""
    for k in range(max(2, k4 - 180), k4 + 1):
        if bull and lo4[k] > hi4[k - 2]:
            zl, zh = float(hi4[k - 2]), float(lo4[k])
            if k < k4 and (lo4[k + 1:k4 + 1] <= zl).any():
                continue
        elif (not bull) and hi4[k] < lo4[k - 2]:
            zl, zh = float(hi4[k]), float(lo4[k - 2])
            if k < k4 and (hi4[k + 1:k4 + 1] >= zh).any():
                continue
        else:
            continue
        if zl - band <= x <= zh + band:
            return True
    return False


UP4 = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")     # as the research's UP
_LSM_CACHE = {}


def _lsm_mod():
    """B12.1 (57 B): the fixed brain module and its per-market settings (loaded once)."""
    if "m" not in _LSM_CACHE:
        try:
            from src.execution import livermore_state_machine as _m
            import json as _json
            import os as _os
            _p = _json.load(open(_os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig")).get("LIVERMORE_PIVOTS", {})
            _LSM_CACHE["m"], _LSM_CACHE["p"] = _m, _p
        except Exception as _e:
            logging.getLogger(__name__).warning("[NS-BRAIN] the brain module could not be loaded -- brain checks see no levels: %s", _e)
            _LSM_CACHE["m"], _LSM_CACHE["p"] = None, {}
    return _LSM_CACHE["m"], _LSM_CACHE["p"]


def _yday_side(t1, hi1, lo1, conf, d):
    """B12.1 (decision 54): the previous UTC day's LOW for a buy (d=+1) / HIGH for a sell (days by candle open time,
    the latest day with candles on or before yesterday -- as the research). None if unknown."""
    days = (t1 - pd.Timedelta(hours=1)).normalize()
    prev = pd.Timestamp(conf).normalize() - pd.Timedelta(days=1)
    m = days <= prev
    if not m.any():
        return None
    sel = days == days[m].max()
    return float(lo1[sel].min()) if d == 1 else float(hi1[sel].max())


def _yday_hl_near(t1, hi1, lo1, conf, x, band):
    """Test 8c: x within `band` of the previous UTC day's high or low (days by candle open time). None if unknown."""
    days = (t1 - pd.Timedelta(hours=1)).normalize()
    prev = conf.normalize() - pd.Timedelta(days=1)
    m = days <= prev
    if not m.any():
        return None
    sel = days == days[m].max()
    return bool(abs(x - float(hi1[sel].max())) <= band or abs(x - float(lo1[sel].min())) <= band)


def _line_map_latest(asset):
    """B14: the market's newest line map (built once per look by the builder; see line_map.build_live)."""
    try:
        from src.execution import line_map as _lm
        return _lm.latest(asset)
    except Exception as _e:
        logger.warning("[NS-MAP] %s: line map unavailable: %s", asset, _e)
        return None


def _cs_get(cs, name):
    if cs is None:
        return None
    return cs.get(name) if isinstance(cs, dict) else getattr(cs, name, None)


# -- B12 exploration: the bounce pockets from Test 1c, on paper only ----------------------------------------------
def _fvgs_open(hi, lo, upto, max_bars, bull):
    """Open fair-value gaps formed at or before index `upto` and not traded through by it, in one direction."""
    out = []
    for k in range(max(2, upto - max_bars), upto + 1):
        if bull and lo[k] > hi[k - 2]:
            zl, zh = float(hi[k - 2]), float(lo[k])
            if not (lo[k + 1:upto + 1] <= zl).any():
                out.append((zl, zh))
        elif (not bull) and hi[k] < lo[k - 2]:
            zl, zh = float(hi[k]), float(lo[k - 2])
            if not (hi[k + 1:upto + 1] >= zh).any():
                out.append((zl, zh))
    return out


def bounce_candidates(asset, df1, df4, st_main, bst, pcfg=None, pockets=None, now=None):
    """B12 (Desire 28 Sep, rulings A+B): Test 1c's bounce, for the chosen market/level pockets, on the newest
    closed 1H candle only. Price came from the right side, touched a level known before this candle (within a
    quarter move, or into its zone) and closed back on the right side -> one PRACTICE trade: stop 0.3 moves beyond
    the candle's extreme, target 2.5 moves, out after 7 days. One per pocket per day. Returns (proofs, bst)."""
    out, bst = [], dict(bst or {})
    try:
        cfg = market_settings(asset, pcfg)
        want = list((pockets or {}).get(str(asset).upper(), []) or [])
        if cfg is None or not want or df1 is None or len(df1) < 220:
            return out, bst
        t1 = close_times(df1, 1)
        hi1, lo1, c1 = (df1[c].astype(float).values for c in ("high", "low", "close"))
        a1 = atr14(hi1, lo1, c1)
        i = len(t1) - 1
        if now is None:
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if (now - t1[i]) > STALE_EMIT or bst.get("last_t") == t1[i]:
            return out, bst
        bst["last_t"] = t1[i]
        atr = float(a1[i])
        if not (atr == atr and atr > 0):
            return out, bst
        t_prev = t1[i - 1]
        df4 = hourly_to_4h(df1)                 # B12 (35 A): the same 4H candles as the engine
        t4 = close_times(df4, 4)
        hi4, lo4 = df4["high"].astype(float).values, df4["low"].astype(float).values
        k4 = int(t4.searchsorted(t_prev, side="right")) - 1
        ema20 = float(pd.Series(c1[:i]).ewm(span=20, adjust=False).mean().iloc[-1])
        bh = (st_main or {}).get("brain_hist", []) if isinstance(st_main, dict) else []
        fired = bst.setdefault("fired", {})
        tol = BOUNCE_TOL_ATR * atr
        for want_low in (True, False):
            d = 1 if want_low else -1
            bands = []
            if "EMA 20 (1H)" in want:
                bands.append(("EMA 20 (1H)", ema20, ema20))
            for btf, days, lab in (("1H", 10, "1H brain level"), ("4H", 30, "4H brain level")):
                if lab in want:
                    bands += [(lab, float(v), float(v)) for (tf, v, since) in bh
                              if tf == btf and since <= t_prev and (t_prev - since) <= timedelta(days=days)]
            if "open FVG (4H)" in want and k4 >= 2:
                bands += [("open FVG (4H)", zl, zh) for (zl, zh) in _fvgs_open(hi4, lo4, k4, 180, want_low)]
            for (typ, l, h) in bands:
                if want_low:
                    ok = c1[i - 1] > h and lo1[i] <= h + tol and c1[i] > h and lo1[i] >= l - 2 * atr
                else:
                    ok = c1[i - 1] < l and hi1[i] >= l - tol and c1[i] < l and hi1[i] <= h + 2 * atr
                if not ok:
                    continue
                key = typ                                 # one practice bounce per pocket per day (Test 1c: one
                if key in fired and (t1[i] - fired[key]) < timedelta(hours=24):   # at a time per level type)
                    continue
                e, ext = float(c1[i]), float(lo1[i] if d == 1 else hi1[i])
                stop, target = ns_levels(d, e, ext, atr, cfg["min_sl_pct"], BOUNCE_TARGET_ATR)
                if stop is None or not gate_rr_ok(e, stop, atr, cfg["min_rr"]):
                    continue
                fired[key] = t1[i]
                lvl = float(h if d == 1 else l)
                fields = {"setup_active": True, "setup_kind": "BOUNCE", "setup_dir": d, "setup_ref": lvl,
                          "ns_entry": "BOUNCE", "ns_exit": "FIXED", "ns_target_atr": BOUNCE_TARGET_ATR,
                          "ns_r2": ext, "ns_edge": ext, "ns_r1": None, "ns_atr1": atr, "ns_stop": float(stop),
                          "ns_target": float(target), "ns_close": e, "ns_candle": str(t1[i]),
                          "ns_explore": "BOUNCE", "ns_bounce_level": typ, "ns_bounce_band": [float(l), float(h)]}
                out.append({"key": (str(asset).upper(), "BOUNCE", d, round(lvl, 8)), "kind": "BOUNCE", "dir": d,
                            "ref": lvl, "tests": 0, "first_ts": str(t1[i]), "fields": fields})
                logger.info("[NS-EXPLORE] %s: BOUNCE off %s dir=%+d level=%.5g-%.5g entry=%.5g stop=%.5g "
                            "target=%.5g (practice lane only)", str(asset).upper(), typ, d, l, h, e, stop, target)
        for k in [k for k, v in fired.items() if (t1[i] - v) > timedelta(days=2)]:
            del fired[k]
    except Exception as ex:
        logger.warning("[NS-EXPLORE] %s: bounce check failed: %s", asset, ex)
    return out, bst
