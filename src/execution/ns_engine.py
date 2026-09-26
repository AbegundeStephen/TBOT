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
    "BTC":    dict(entry="E", target_atr=None, exit="RUNNER", cont_only=False, no_spike=False),
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

    def __init__(self, asset):
        self.asset = str(asset).upper()
        self._first_call = True

    @staticmethod
    def new_state():
        return {"v": 1, "last_1h": None, "hist": {"H": [], "L": []}, "hist_upto": None,
                "setups": [], "seen": [], "next_id": 1, "levels2": {}}

    # -- public ---------------------------------------------------------------------------
    def update(self, st, df1, df4, pcfg=None, cs=None):
        out = {"state": st, "proofs": [], "head": None, "ladder": [], "setups": [], "brains": {},
               "processed": 0, "missed": []}
        cfg = market_settings(self.asset, pcfg)
        if not isinstance(st, dict) or st.get("v") != 1:
            st = self.new_state()
        out["state"] = st
        if cfg is None or df1 is None or df4 is None or len(df1) < 30 or len(df4) < 2 * K + 2:
            return out
        t1 = close_times(df1, 1)
        hi1, lo1, c1 = (df1[c].astype(float).values for c in ("high", "low", "close"))
        a1 = atr14(hi1, lo1, c1)
        t4 = close_times(df4, 4)
        hi4, lo4, c4 = (df4[c].astype(float).values for c in ("high", "low", "close"))
        a4 = atr14(hi4, lo4, c4)
        close4 = dict(zip(t4, c4))
        atr4_at = dict(zip(t4, a4))
        bar4 = {t: (h, l, c) for t, h, l, c in zip(t4, hi4, lo4, c4)}
        sw = swings_4h(t4, hi4, lo4, c4)

        n1 = len(t1)
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

        for i in range(first_i, n1):
            t = t1[i]
            emit = (i == n1 - 1)
            if a1[i] == a1[i]:
                self._candle(st, cfg, i, t, emit, t1, hi1, lo1, c1, a1, close4, bar4, atr4_at, out)
            while ptr < len(sw) and sw[ptr][0] <= t:
                self._register(st, sw[ptr], atr4_at, create=True, emit=emit)
                ptr += 1
            st["last_1h"] = t
            out["processed"] += 1

        live = st["setups"]
        if out["processed"]:
            st["proofs_live"] = list(out["proofs"])
        else:
            out["proofs"] = list(st.get("proofs_live", []))
        if out["processed"]:
            n_stage = [sum(1 for s in live if s["stage"] == k) for k in (0, 1, 2)]
            logger.info("[NS] %s: %d candle(s) processed to %s | live setups %d (waiting break %d, retest %d, "
                        "trigger %d) | proofs %d", self.asset, out["processed"], t1[-1], len(live),
                        n_stage[0], n_stage[1], n_stage[2], len(out["proofs"]))
        if self._first_call:
            self._first_call = False
            logger.info("[DEPLOY-HYGIENE] %s: new engine caught up %d candle(s); %d setup(s) live; "
                        "%d trigger(s) found while catching up were logged, not traded",
                        self.asset, out["processed"], len(live), len(out["missed"]))
        _a1l = float(a1[-1]) if a1[-1] == a1[-1] else None
        _a4l = float(a4[-1]) if a4[-1] == a4[-1] else None
        out["setups"] = [self._public(s, float(c1[-1]), _a1l, float(c4[-1]), _a4l) for s in live]
        out["head"] = self._head_fields(live, t1[-1])
        out["ladder"] = self._ladder(st, t1[-1], cs)
        out["brains"] = self._brains(cs, float(c1[-1]), float(a1[-1]) if a1[-1] == a1[-1] else None,
                                     float(a4[-1]) if a4[-1] == a4[-1] else None)
        return out

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
        sid = st["next_id"]
        st["next_id"] += 1
        st["setups"].append(dict(id=sid, d=d, r2=r2, edge=edge2, r1=r1, kind=kind, conf=conf, atr4=float(a4),
                                 stage=0, t_break=None, h2=None, t_touch=None))
        if emit:
            logger.info("[SETUP-BORN] %s: NS %s dir=%+d R2=%.5g (zone to %.5g) R1=%.5g -- waiting for a 4H close past R2",
                        self.asset, KIND_MAP.get(kind, "TF_CONT"), d, r2, edge2, r1)
            logger.info("[R1-ORIGIN] %s: NS %s dir=%+d H=%.5g R1=%.5g tag=SWING_4H tests=0 gap=%.2fATR4 (%s)",
                        self.asset, KIND_MAP.get(kind, "TF_CONT"), d, r2, r1, abs(r2 - r1) / a4, kind)

    def _end(self, st, s, reason, emit, t):
        st["setups"] = [x for x in st["setups"] if x["id"] != s["id"]]
        if emit:
            logger.info("[MEASURE-8.4-DEATH] %s: kind=%s dir=%+d age_at_death=%s reason=%s",
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
                if r4 is None:
                    continue
                broke = d * (r4 - r2) > 0
                if emit:
                    a4 = atr4_at.get(t) or s["atr4"]
                    logger.info("[COUNT-1-CHECK] %s NS dir=%+d tf=4H close=%.5g H=%.5g band=0 dist=%.2fATR4 "
                                "tier=SWING_4H -> %s", self.asset, d, r4, r2, d * (r4 - r2) / a4,
                                "BREAK" if broke else "no")
                if not broke:
                    continue
                if [d, round(r2, 6)] in st["seen"]:
                    self._end(st, s, "NS_DUPLICATE", emit, t)
                    continue
                st["seen"].append([d, round(r2, 6)])
                if len(st["seen"]) > 400:
                    st["seen"] = st["seen"][-400:]
                j = max(0, i - 3)
                s.update(stage=1, t_break=t, h2=float(c1[j:i + 1].max() if d == 1 else c1[j:i + 1].min()))
                if cfg["entry"] == "B":
                    b = bar4.get(t)
                    strength = candle_strength(d, *b) if b else 0.5
                    self._candidate(st, cfg, s, "B", i, t, c, atr, strength, emit, out)
                    self._end(st, s, "NS_ENTERED", False, t)
                continue
            if s["stage"] == 1:
                if t - s["t_break"] > WAIT_TOUCH:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                if (d == 1 and lo1[i] <= s["edge"] + TOUCH_ATR * atr) or \
                        (d == -1 and hi1[i] >= s["edge"] - TOUCH_ATR * atr):
                    s.update(stage=2, t_touch=t, h2_frozen=s["h2"], touch_px=float(lo1[i] if d == 1 else hi1[i]))
                    if emit:
                        logger.info("[COUNT-2] %s NS dir=%+d RETEST low/high=%.5g edge=%.5g peak=%.5g",
                                    self.asset, d, lo1[i] if d == 1 else hi1[i], s["edge"], s["h2"])
                    continue
                if d * (c - s["h2"]) > 0:
                    s["h2"] = c
                continue
            if s["stage"] == 2:
                if t - s["t_touch"] > WAIT_TRIGGER:
                    self._end(st, s, "NS_EXPIRED", emit, t)
                    continue
                trig = d * (c - s["h2"]) > 0
                if emit:
                    logger.info("[COUNT-3-CHECK] %s NS dir=%+d tf=1H close=%.5g H2=%.5g tol=0 dist=%.2fATR -> %s",
                                self.asset, d, c, s["h2"], d * (c - s["h2"]) / atr, "PROOF" if trig else "no")
                if trig:
                    strength = candle_strength(d, hi1[i], lo1[i], c1[i])
                    self._candidate(st, cfg, s, "E", i, t, c, atr, strength, emit, out)
                    self._end(st, s, "NS_ENTERED", False, t)
        if emit and killed:
            logger.info("[KILL-R1] %s NS close4=%.5g -> %d setup(s) dead (%s)", self.asset, r4, len(killed),
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
        if why is not None:
            if emit:
                logger.info("[NS-SKIP] %s: %s dir=%+d R2=%.5g entry=%.5g -- %s -- retired",
                            self.asset, style, d, r2, e, why)
            return
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
        }
        if kind == "MR_REV":
            fields.update({"setup_active_mr": True, "setup_kind_mr": kind, "setup_dir_mr": d,
                           "setup_age_mr": fields["setup_age"], "setup_ref_mr": float(r2),
                           "setup_ref_tier_mr": "SWING_4H", "setup_ref_tests_mr": 0})
        proof = {"key": (self.asset, kind, d, round(float(r2), 8)), "kind": kind, "dir": d, "ref": float(r2),
                 "tests": 0, "first_ts": str(t), "fields": fields}
        if emit:
            out["proofs"].append(proof)
            logger.info("[NS-PROOF] %s: %s %s dir=%+d R2=%.5g entry=%.5g stop=%.5g target=%s exit=%s (%s, %s)",
                        self.asset, tier, kind, d, r2, e, stop,
                        ("%.5g" % target) if target else "none (runner)", cfg["exit"], s["kind"], str(t))
        else:
            out["missed"].append(proof)
            logger.info("[NS-MISSED] %s: %s dir=%+d R2=%.5g at %s -- found while catching up, not traded",
                        self.asset, tier, d, r2, str(t))

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


def _cs_get(cs, name):
    if cs is None:
        return None
    return cs.get(name) if isinstance(cs, dict) else getattr(cs, name, None)
