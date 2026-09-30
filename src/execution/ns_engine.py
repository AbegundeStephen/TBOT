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
REPLAY_DAYS = 30               # first run after a start: rebuild the setups from 30 days
LAYER1_DAYS = 30
LAYER2_DAYS = {"1H": 10, "4H": 30}
STALE_DAYS = {"1H": 5, "4H": 20}
UP_STATES = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")

DEFAULT_MARKETS = {
    "GOLD":   dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "USOIL":  dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "USTEC":  dict(entry="B", target_atr=4.0, exit="FIXED", cont_only=False, no_spike=False),
    "EURUSD": dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True),
    "GBPAUD": dict(entry="E", target_atr=2.5, exit="FIXED", cont_only=True, no_spike=True),
    "BTC":    dict(entry="E2", target_atr=None, exit="RUNNER", cont_only=False, no_spike=False),   # B12 (decision 9)
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
    return cfg


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
    def update(self, st, df1, df4, pcfg=None, cs=None, now=None):
        out = {"state": st, "proofs": [], "head": None, "ladder": [], "setups": [], "brains": {},
               "processed": 0, "missed": []}
        cfg = market_settings(self.asset, pcfg)
        self._main_cfg = cfg
        if cfg is not None and self.variant == "ALL_PROOFS":
            cfg = dict(cfg, cont_only=False, no_spike=False)     # B12 exploration: the filters' rejects as well
        self._cs_now = cs                       # display only: the brains at the moment of a proof
        if not isinstance(st, dict) or st.get("v") != 1:
            st = self.new_state()
        out["state"] = st
        if cfg is None or df1 is None or len(df1) < 30:
            return out
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
        sw = swings_4h(t4, hi4, lo4, c4)

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
                self._end(st, s, "REF_INVALIDATED", emit, t)
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
                if _conf1h:
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
                if d * (c - s["h2"]) > 0:
                    s["h2"] = c
                    s["h2_t"] = t                   # display only
                continue
            if s["stage"] == 2:
                if t - s["t_touch"] > WAIT_TRIGGER:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                trig = d * (c - s["h2"]) > 0
                if emit:
                    self._log.info("[COUNT-3-CHECK] %s NS dir=%+d tf=1H close=%.5g H2=%.5g tol=0 dist=%.2fATR -> %s",
                                self.asset, d, c, s["h2"], d * (c - s["h2"]) / atr, "PROOF" if trig else "no")
                if trig:
                    strength = candle_strength(d, hi1[i], lo1[i], c1[i])
                    self._candidate(st, cfg, s, "E", i, t, c, atr, strength, emit, out)
                    self._end(st, s, "NS_ENTERED", False, t)
        if emit and killed:
            self._log.info("[KILL-R1] %s NS close4=%.5g -> %d setup(s) dead (%s)", self.asset, r4, len(killed),
                        ", ".join(killed))

    def _candidate(self, st, cfg, s, style, i, t, e, atr, strength, emit, out):
        d, r2 = s["d"], s["r2"]
        why = None
        if cfg["entry"] != style:
            return
        if d * (e - r2) / atr > FRESH_ATR:
            why = "too far from R2 (%.2f moves)" % (d * (e - r2) / atr)
        stop, target = ns_levels(d, e, r2, atr, cfg["min_sl_pct"], cfg["target_atr"])
        if why is None and (stop is None or not gate_rr_ok(e, stop, atr, cfg["min_rr"])):
            why = "no valid stop / R:R below %.2f" % cfg["min_rr"]
        if why is None and cfg.get("cont_only") and s["kind"] != "continuation":
            why = "filter: %s (continuations only)" % s["kind"]
        if why is None and cfg.get("no_spike") and style != "B" and strength > SPIKE:
            why = "filter: spike candle (closed at %.0f%% of its range)" % (100 * strength)
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
        if kind == "MR_REV":
            fields.update({"setup_active_mr": True, "setup_kind_mr": kind, "setup_dir_mr": d,
                           "setup_age_mr": fields["setup_age"], "setup_ref_mr": float(r2),
                           "setup_ref_tier_mr": "SWING_4H", "setup_ref_tests_mr": 0})
        proof = {"key": (self.asset, kind, d, round(float(r2), 8)), "kind": kind, "dir": d, "ref": float(r2),
                 "tests": 0, "first_ts": str(t), "fields": fields}
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
                        self.asset, tier, kind, d, r2, e, stop,
                        ("%.5g" % target) if target else "none (runner)", cfg["exit"], s["kind"], str(t))
        else:
            out["missed"].append(proof)
            self._log.info("[NS-MISSED] %s: %s dir=%+d R2=%.5g at %s -- found while catching up, not traded",
                        self.asset, tier, d, r2, str(t))

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
                    if _brain("4H", 30, s["r1"]):
                        aplus.append("reversal at 4H brain level")                      # 46 A (Test 7a, fixed brain)
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
