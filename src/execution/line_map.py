"""B14 (Desire 7-9 Oct 2026) -- THE LINE MAP: one map of every line per market, read by the engine, the package,
the trade manager, the chart and the card, so they can never disagree.

What is on the map (each line with its kind, its size and its sphere):
  4H swing closes (highs and lows)          small; "major" = the highest/lowest 4H close for 6 candles each side
  daily swing closes                        big when major (the highest/lowest daily close for 3 candles each side)
  weekly swing closes                       big
  the 50 and 200 EMAs on 4H, daily, weekly  the 4H 50 is small; the 4H 200 and every daily/weekly average is big
  diagonals (falling through swing highs,   big = a long line with 3+ touches over 2+ days; small = a short, steep
  rising through swing lows)                2-touch line; kept for good; awake while price came within 1 4H move of
                                            it in the last 18 4H candles (3 trading days); drawn per candle, so a
                                            weekend leaves no gap
  channel edges                             a diagonal channel or a horizontal range around price; its edges are walls

Spheres: half a 4H move around a big line, a quarter around a small one; entering a sphere counts as a touch.
A map line breaks on a 4H close past the three-quarter mark of its sphere (big: a quarter move past the line; small:
an eighth). Bands: the nearest line ahead and every line within half a 4H move beyond it form one band. A band is a
BIG WALL when it holds a big line, 2+ averages, 3 kinds of line, or a channel edge (4H swing lines are never big
walls on their own).

Pure numpy/pandas: no MT5, no logging side effects in the queries -- the live bot and the 16-month box test call the
same functions. Live, `build_live()` fetches the candles and keeps the newest map per market in LATEST.
"""
import collections
import logging
import threading

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- the agreed numbers ---------------------------------------------------------------------------------------------
K = 2                    # a swing close: higher/lower than the 2 closes on each side (the engine's rule)
MAJOR_K = 6              # a major 4H line: the highest/lowest 4H close for 6 candles each side (14.2 B)
D1_MAJOR_K = 3           # a major daily line: the highest/lowest daily close for 3 candles each side
LOOKBACK_DAYS = 365      # diagonals and daily/weekly lines are remembered for 1 year (4.7)
OLD_HIGH_DAYS = 60       # the engine's old-high check keeps its 60 days (it lives in ns_engine.py)
MAP_WIN = 4.0            # the map keeps lines within 4 4H moves of price
BAND = 0.5               # a band: the nearest line ahead + every line within half a 4H move beyond it
STACK_N = 3              # a stack = 3 different kinds of line in one band
SPH_BIG, SPH_SMALL = 0.5, 0.25   # sphere half-widths in 4H moves
BREAK_FRAC = 0.5         # a break is a close past the line + half its sphere (the three-quarter mark)
DG_TOUCH = 0.5           # a 4H close within half a 4H move of a diagonal is a touch (as tested)
DG_BIG_TOUCHES = 3       # big diagonal: 3+ touches ...
DG_BIG_SPAN = 12         # ... spread over 12+ 4H candles (2 days)
DG_SMALL_SPAN = 12       # small diagonal: its 2 swing points at most 12 4H candles apart ...
DG_STEEP = 0.15          # ... and rising/falling at least 0.15 of a 4H move per 4H candle
DG_AWAKE = 1.0           # awake while price came within 1 4H move of the line ...
DG_AWAKE_BARS = 18       # ... in the last 18 4H candles
DG_SAME = 0.15           # near-identical diagonals (within 0.15 of a move) are one line
WITNESS_H = 12           # a witness broke the same way in the 12 hours before the trigger
WALL_REACH = 0.5         # a wall = a big band within half a 4H move ahead
NEAR_REACH = 1.0         # "near" = within 1 4H move
WILD_K, WILD_DAYS = 1.5, 60       # wild market: the 4H move is at least 1.5 x its 60-day average
CH_MAX = 6.0             # a channel is at most 6 4H moves wide
CH_SLOPE = 0.3           # its two edges have nearly the same slope (within 30%)

# line kinds (the box test's codes, plus the two channel-edge kinds)
H4H, H4L, D1H, D1L, W1H, W1L = 0, 1, 2, 3, 4, 5
MA4_50, MA4_200, MAD_50, MAD_200, MAW_50, MAW_200 = 6, 7, 8, 9, 10, 11
DSH, DSL, DBH, DBL = 12, 13, 14, 15            # small falling (through highs) / small rising / big falling / big rising
CHT, CHB = 16, 17                              # channel top / channel bottom
NAME = {0: "4H high", 1: "4H low", 2: "daily high", 3: "daily low", 4: "weekly high", 5: "weekly low",
        6: "4H 50 MA", 7: "4H 200 MA", 8: "daily 50 MA", 9: "daily 200 MA", 10: "weekly 50 MA", 11: "weekly 200 MA",
        12: "small falling diagonal", 13: "small rising diagonal", 14: "big falling diagonal", 15: "big rising diagonal",
        16: "channel top", 17: "channel bottom"}
FAMILY = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 15: 3,
          16: 4, 17: 4}                          # 4H lines, daily/weekly lines, averages, diagonals, channel edges
MA_DEF = {MA4_50: ("4H", 50), MA4_200: ("4H", 200), MAD_50: ("1D", 50), MAD_200: ("1D", 200),
          MAW_50: ("1W", 50), MAW_200: ("1W", 200)}
BIG_MA = (MA4_200, MAD_50, MAD_200, MAW_50, MAW_200)   # 4.3 (9 Oct): the 4H 50 is small, the rest are big

LATEST = {}              # live: asset -> the newest MarketMap (read by the trade manager and the chart)
_LOCK = threading.Lock()


def ema(x, n):
    return pd.Series(np.asarray(x, dtype=float)).ewm(span=n, adjust=False).mean().values


def atr14(h, l, c):
    """The engine's ATR: simple 14-candle average of the true range."""
    h, l, c = (np.asarray(x, dtype=float) for x in (h, l, c))
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    return pd.Series(tr).rolling(14).mean().values


def to_4h(df1):
    """The engine's own 4H candles from hourly candles (ns_engine.hourly_to_4h): complete blocks only."""
    d = df1[["open", "high", "low", "close"]].astype(float)
    g = d.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    last_close = df1.index[-1] + pd.Timedelta(hours=1)
    return g[g.index + pd.Timedelta(hours=4) <= last_close]


def _tf_swings(df, span, code_h, code_l, major_k):
    """Swing closes (2 candles each side) of a daily or weekly frame: (confirmed at, level, code, major, major known at).
    major_k None = every swing counts as major (weekly)."""
    if df is None or len(df) < 5:
        e = np.array([], dtype="datetime64[ns]")
        return e, np.array([]), np.array([], dtype=np.int8), np.array([], dtype=bool), e
    c = df["close"].astype(float).values
    end = (df.index + span).values
    rows = []
    for i in range(2, len(c) - 2):
        w = c[i - 2:i + 3]
        for hit, cd in ((c[i] == w.max(), code_h), (c[i] == w.min(), code_l)):
            if not hit:
                continue
            if major_k is None:
                mj, mt = True, end[i + 2]
            elif i + major_k < len(c):
                ww = c[max(0, i - major_k):i + major_k + 1]
                mj, mt = bool(c[i] == (ww.max() if cd == code_h else ww.min())), end[i + major_k]
            else:
                mj, mt = False, np.datetime64("2200-01-01")
            rows.append((end[i + 2], float(c[i]), cd, mj, mt))
    rows.sort(key=lambda r: r[0])
    return (np.array([r[0] for r in rows], dtype="datetime64[ns]"), np.array([r[1] for r in rows], dtype=float),
            np.array([r[2] for r in rows], dtype=np.int8), np.array([r[3] for r in rows], dtype=bool),
            np.array([r[4] for r in rows], dtype="datetime64[ns]"))


class Diagonal:
    """One diagonal through two neighbouring 4H swing closes. Its level for 4H candle x is v1 + slope * (x - i1)
    (per candle, so weekends leave no gap). Arrays run from `start` (the 4H candle after the second swing is confirmed)
    to the end of the data."""
    __slots__ = ("ln", "d", "i1", "v1", "slope", "start", "cls", "awake", "touches", "breaks", "role", "i2")

    def level(self, x):
        return self.v1 + self.slope * (x - self.i1)

    def cls_at(self, x):          # 0 = not drawn (neither big nor small), 1 = small, 2 = big
        m = x - self.start
        return int(self.cls[m]) if 0 <= m < len(self.cls) else 0

    def awake_at(self, x):
        m = x - self.start
        return bool(self.awake[m]) if 0 <= m < len(self.awake) else False


def _diagonals(hi4, lo4, c4, a4, role):
    """Every diagonal of one role. role 'break' = the breakable kind (falling through highs = a ceiling for buys, d=+1;
    rising through lows = a floor for sells, d=-1) -- the hybrid diagonals of the 8 Oct tests. role 'channel' = the
    other two kinds (rising through highs = a rising channel's top; falling through lows = a falling channel's bottom),
    used for channels only. Touches count until the line first breaks."""
    n = len(c4)
    piv = {"H": [], "L": []}
    for i in range(K, n - K):
        w = c4[i - K:i + K + 1]
        if c4[i] == w.max():
            piv["H"].append(i)
        if c4[i] == w.min():
            piv["L"].append(i)
    okA = (a4 == a4) & (a4 > 0)
    aa = np.where(okA, a4, np.nan)
    out, ln = [], 0
    specs = ((1, "H", "falling"), (-1, "L", "rising")) if role == "break" else ((1, "H", "rising"), (-1, "L", "falling"))
    for d, typ, shape in specs:
        P = piv[typ]
        for j in range(1, len(P)):
            i1, i2 = P[j - 1], P[j]
            v1, v2 = float(c4[i1]), float(c4[i2])
            if (shape == "falling" and not v2 < v1) or (shape == "rising" and not v2 > v1):
                continue
            slope = (v2 - v1) / (i2 - i1)
            start = i2 + K
            if start >= n:
                continue
            x = np.arange(start, n)
            lv = v1 + slope * (x - i1)
            with np.errstate(invalid="ignore", divide="ignore"):
                dist = (c4[x] - lv) / aa[x]
                gap = np.maximum(np.maximum(lo4[x] - lv, lv - hi4[x]), 0.0) / aa[x]
            fin = ~np.isnan(dist)
            past = d if role == "break" else (1 if typ == "H" else -1)     # the side a break goes to
            sd = np.where(fin, past * dist, 0.0)
            ne, first = 0, i1
            if role == "break":
                for p in P[max(0, j - 5):j - 1]:          # earlier swings on the line count as touches (as tested)
                    if okA[p] and abs(c4[p] - (v1 + slope * (p - i1))) <= DG_TOUCH * a4[p]:
                        ne += 1
                        first = min(first, p)
            crossed = fin & (sd >= SPH_SMALL * BREAK_FRAC)  # the earliest a break can happen (a small line's mark)
            fc = int(np.argmax(crossed)) if crossed.any() else len(sd)
            near = fin & (np.abs(np.where(fin, dist, 99.0)) <= DG_TOUCH) & (sd < SPH_SMALL * BREAK_FRAC) & \
                (x >= i2 + 3) & (np.arange(len(sd)) < fc)
            ev = near & ~np.r_[False, near[:-1]]         # each new visit before the first break is one touch
            touches = 2 + ne + np.cumsum(ev)
            span_ = np.maximum.accumulate(np.where(ev, x, i2)) - first
            big = (touches >= DG_BIG_TOUCHES) & (span_ >= DG_BIG_SPAN)
            steep = bool(okA[i2] and abs(slope) >= DG_STEEP * a4[i2] and (i2 - i1) <= DG_SMALL_SPAN)
            cls = np.where(big, 2, 1 if steep else 0).astype(np.int8)
            ce = np.where(np.isnan(gap), False, gap <= DG_AWAKE)
            cs = np.cumsum(ce)
            prev = np.r_[np.zeros(DG_AWAKE_BARS, dtype=cs.dtype), cs[:-DG_AWAKE_BARS]] if len(cs) > DG_AWAKE_BARS \
                else np.zeros_like(cs)
            awake = (cs - prev) > 0
            g = Diagonal()
            g.ln, g.d, g.i1, g.i2, g.v1, g.slope, g.start, g.role = ln, d, i1, i2, v1, slope, start, role
            g.cls, g.awake, g.touches = cls, awake, touches
            # breaks (only for the breakable kind): a 4H close past the three-quarter mark while awake and drawn;
            # each break holds until a 4H close back across (the 1H hold for witnesses is checked separately)
            br = []
            if role == "break":
                back = np.nonzero(fin & (sd < 0))[0]
                m_min = 0
                for m in range(len(sd)):
                    if m < m_min or not fin[m] or m < 1:
                        continue
                    c0 = cls[m - 1]
                    if c0 == 0 or not awake[m - 1]:
                        continue
                    mark = (SPH_BIG if c0 == 2 else SPH_SMALL) * BREAK_FRAC
                    if sd[m] >= mark and sd[m - 1] < mark:
                        pos = int(np.searchsorted(back, m, side="right"))
                        hend = int(back[pos]) if pos < len(back) else None
                        br.append((start + m, (start + hend) if hend is not None else n, int(c0)))
                        m_min = (hend + 1) if hend is not None else len(sd)
            g.breaks = br
            out.append(g)
            ln += 1
    return out


class MarketMap:
    """Every line of one market, worked out once from the candles. Query it at any 4H close b (no look-ahead):
    members(b), bands(), walls_ahead(), walls_between(), witnesses(), turn_at(), channels(), wild()."""

    def __init__(self, asset, df1, d1=None, w1=None, built_at=None):
        self.asset = str(asset).upper()
        df1 = df1[["open", "high", "low", "close"]].astype(float)
        self.df1 = df1
        self.t1 = df1.index + pd.Timedelta(hours=1)                 # 1H close times
        self.c1, self.hi1, self.lo1 = (df1[c].values for c in ("close", "high", "low"))
        d4 = to_4h(df1)
        self.d4 = d4
        self.t4 = (d4.index + pd.Timedelta(hours=4)).values         # 4H close times
        self.o4, self.hi4, self.lo4, self.c4 = (d4[c].values for c in ("open", "high", "low", "close"))
        self.a4 = atr14(self.hi4, self.lo4, self.c4)
        self.n = n = len(self.c4)
        self.built_at = built_at if built_at is not None else (pd.Timestamp(self.t4[-1]) if n else None)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if d1 is None or len(d1) < 5:
            d1 = df1.resample("1D").agg(agg).dropna()
        if w1 is None or len(w1) < 5:
            w1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        self.d1, self.w1 = d1, w1
        # 4H swing closes, with "major" (MAJOR_K) and when that became known
        c4 = self.c4
        sw = []
        for i in range(K, n - K):
            w = c4[i - K:i + K + 1]
            for typ, hit in (("H", c4[i] == w.max()), ("L", c4[i] == w.min())):
                if not hit:
                    continue
                if i + MAJOR_K < n:
                    ww = c4[max(0, i - MAJOR_K):i + MAJOR_K + 1]
                    mj, mt = bool(c4[i] == (ww.max() if typ == "H" else ww.min())), self.t4[i + MAJOR_K]
                else:
                    mj, mt = False, np.datetime64("2200-01-01")
                sw.append((self.t4[i + K], H4H if typ == "H" else H4L, float(c4[i]), i, mj, mt))
        sw.sort(key=lambda x: x[0])
        self.sw_conf = np.array([x[0] for x in sw], dtype="datetime64[ns]")
        self.sw_code = np.array([x[1] for x in sw], dtype=np.int8)
        self.sw_lv = np.array([x[2] for x in sw], dtype=float)
        self.sw_i = np.array([x[3] for x in sw], dtype=int)
        self.sw_maj = np.array([x[4] for x in sw], dtype=bool)
        self.sw_majt = np.array([x[5] for x in sw], dtype="datetime64[ns]")
        self.dsw = _tf_swings(d1, pd.Timedelta(days=1), D1H, D1L, D1_MAJOR_K)
        self.wsw = _tf_swings(w1, pd.Timedelta(days=7), W1H, W1L, None)
        # averages per 4H close (daily/weekly from the candle closed by then)
        dend = (d1.index + pd.Timedelta(days=1)).values
        wend = (w1.index + pd.Timedelta(days=7)).values
        kd = np.searchsorted(dend, self.t4, side="right") - 1
        kw = np.searchsorted(wend, self.t4, side="right") - 1
        dcl, wcl = d1["close"].astype(float).values, w1["close"].astype(float).values
        self.ma_at = {}
        self.ma_missing = []
        for cd, (tf, span) in MA_DEF.items():
            if tf == "4H":
                v = ema(c4, span) if n else np.array([])
                v = np.where(np.arange(n) >= span, v, np.nan)
            else:
                k_, src = (kd, dcl) if tf == "1D" else (kw, wcl)
                if len(src) < span:
                    self.ma_missing.append(NAME[cd])          # 4.7: not enough candles -> left out (logged by the caller)
                e_ = ema(src, span) if len(src) else np.array([np.nan])
                v = np.where(k_ >= span - 1, e_[np.clip(k_, 0, len(e_) - 1)], np.nan)
            self.ma_at[cd] = v
        self.diags = _diagonals(self.hi4, self.lo4, c4, self.a4, "break") if n > 10 else []
        self.chlines = _diagonals(self.hi4, self.lo4, c4, self.a4, "channel") if n > 10 else []
        self._m = {}
        self._ch = {}

    # -- time helpers ------------------------------------------------------------------------------------------------
    def bucket(self, t):
        """Index of the last 4H candle closed at or before t (-1 if none)."""
        return int(np.searchsorted(self.t4, np.datetime64(pd.Timestamp(t)), side="right")) - 1

    def a4_at(self, b):
        return float(self.a4[b]) if 0 <= b < self.n and self.a4[b] == self.a4[b] else float("nan")

    # -- the map at one 4H close -------------------------------------------------------------------------------------
    def members(self, b):
        """Every line on the map at 4H close b, valid for the next 4H candle: dict of arrays lv, code, big (a big line),
        maj (a major line, for turning points), sph (sphere half-width in price), fam, plus a (the 4H move), p, t."""
        got = self._m.get(b)
        if got is not None:
            return got
        a = self.a4_at(b)
        if not (a == a and a > 0):
            e = np.array([])
            got = dict(lv=e, code=e.astype(np.int8), big=e.astype(bool), maj=e.astype(bool), sph=e, fam=e.astype(np.int8),
                       ln=e.astype(int), a=float("nan"), p=float("nan"), t=None)
            self._m[b] = got
            return got
        t, p = self.t4[b], float(self.c4[b])
        lim = MAP_WIN * a
        LV, CD, BG, MJ, LN = [], [], [], [], []
        lo = int(np.searchsorted(self.sw_conf, t - np.timedelta64(LOOKBACK_DAYS, "D"), side="left"))
        hi = int(np.searchsorted(self.sw_conf, t, side="right"))
        v = self.sw_lv[lo:hi]
        m = np.abs(v - p) <= lim
        LV.append(v[m]), CD.append(self.sw_code[lo:hi][m]), BG.append(np.zeros(int(m.sum()), dtype=bool))
        MJ.append((self.sw_maj[lo:hi] & (self.sw_majt[lo:hi] <= t))[m]), LN.append(np.full(int(m.sum()), -1))
        for (cf, lv_, cd, mj, mt) in (self.dsw, self.wsw):
            if not len(cf):
                continue
            lo = int(np.searchsorted(cf, t - np.timedelta64(LOOKBACK_DAYS, "D"), side="left"))
            hi = int(np.searchsorted(cf, t, side="right"))
            v = lv_[lo:hi]
            m = np.abs(v - p) <= lim
            bg = (mj[lo:hi] & (mt[lo:hi] <= t))[m]
            LV.append(v[m]), CD.append(cd[lo:hi][m]), BG.append(bg), MJ.append(bg), LN.append(np.full(int(m.sum()), -1))
        ml, mc, mb, mm, mn = [], [], [], [], []
        for cd, arr in self.ma_at.items():
            x = arr[b] if b < len(arr) else float("nan")
            if x == x and abs(x - p) <= lim:
                ml.append(float(x)), mc.append(cd), mb.append(cd in BIG_MA), mm.append(True), mn.append(-1)
        keep = []
        for g in self.diags:
            if b < g.start or g.start < b - 6 * LOOKBACK_DAYS:
                continue
            c0, aw = g.cls_at(b), g.awake_at(b)
            if c0 == 0 or not aw:
                continue
            lvn = g.level(b + 1)                         # its level for the next 4H candle
            if abs(lvn - p) > lim:
                continue
            code = (DBH if g.d == 1 else DBL) if c0 == 2 else (DSH if g.d == 1 else DSL)
            keep.append((lvn, code, g.ln))
        keep.sort(key=lambda e: (e[1] not in (DBH, DBL), e[0]))      # big lines first: near-identical lines are one
        seen = []
        for lvn, code, ln in keep:
            hi_ = code in (DSH, DBH)
            if any((k[1] in (DSH, DBH)) == hi_ and abs(k[0] - lvn) <= DG_SAME * a for k in seen):
                continue
            seen.append((lvn, code, ln))
            ml.append(float(lvn)), mc.append(code), mb.append(code in (DBH, DBL)), mm.append(code in (DBH, DBL)), mn.append(ln)
        LV.append(np.array(ml, dtype=float)), CD.append(np.array(mc, dtype=np.int8))
        BG.append(np.array(mb, dtype=bool)), MJ.append(np.array(mm, dtype=bool)), LN.append(np.array(mn, dtype=int))
        lv = np.concatenate(LV)
        code = np.concatenate(CD).astype(np.int8)
        big = np.concatenate(BG).astype(bool)
        maj = np.concatenate(MJ).astype(bool)
        lns = np.concatenate(LN).astype(int)
        o = np.argsort(lv, kind="stable")                # the range finder below needs the lines in price order
        lv, code, big, maj, lns = lv[o], code[o], big[o], maj[o], lns[o]
        # channel edges (9 Oct: channel edges count as walls)
        chs = self.channels(b, _members=dict(lv=lv, code=code, a=a, p=p))
        el, ec = [], []
        for ch in chs:
            el += [ch["top"], ch["bottom"]]
            ec += [CHT, CHB]
        if el:
            lv = np.concatenate([lv, np.array(el, dtype=float)])
            code = np.concatenate([code, np.array(ec, dtype=np.int8)])
            big = np.concatenate([big, np.ones(len(el), dtype=bool)])
            maj = np.concatenate([maj, np.zeros(len(el), dtype=bool)])
            lns = np.concatenate([lns, np.full(len(el), -1)])
        o = np.argsort(lv, kind="stable")
        lv, code, big, maj, lns = lv[o], code[o], big[o], maj[o], lns[o]
        sph = np.where(big | np.isin(code, BIG_MA), SPH_BIG, SPH_SMALL) * a
        fam = np.array([FAMILY[int(c)] for c in code], dtype=np.int8)
        got = dict(lv=lv, code=code, big=big, maj=maj, sph=sph, fam=fam, ln=lns, a=a, p=p, t=pd.Timestamp(t))
        self._m[b] = got
        return got

    @staticmethod
    def _ahead(m, d, p):
        lv = m["lv"]
        if d == 1:
            return list(range(int(np.searchsorted(lv, p, side="right")), len(lv)))
        return list(range(int(np.searchsorted(lv, p, side="left")) - 1, -1, -1))

    def bands(self, b, d, p, reach=MAP_WIN, limit=None):
        """The bands ahead of price p (above for a buy, below for a sell), nearest first, whose first line is within
        `reach` 4H moves (and before `limit`). Each: near/far (first and last line), s_near/s_far (the sphere edges:
        the edge facing the trade and the edge beyond the wall), mark (the three-quarter mark past the far line, which
        clears the wall), big (a big wall), kinds, label."""
        if b < 0:
            return []
        m = self.members(b)
        lv, code, big, sph, fam, a = m["lv"], m["code"], m["big"], m["sph"], m["fam"], m["a"]
        if not (a == a and a > 0):
            return []
        idx = self._ahead(m, d, p)
        out, k, N = [], 0, len(idx)
        while k < N:
            i = idx[k]
            near = lv[i]
            if d * (near - p) > reach * a or (limit is not None and d * (near - limit) >= 0):
                break
            mem = []
            while k < N and d * (lv[idx[k]] - near) <= BAND * a:
                mem.append(idx[k])
                k += 1
            cds = code[mem]
            fams = set(int(f) for f in fam[mem])
            nma = int(((cds >= MA4_50) & (cds <= MAW_200)).sum())
            bigl = bool(big[mem].any())
            chan = bool(((cds == CHT) | (cds == CHB)).any())
            isbig = bigl or nma >= 2 or len(fams - {4}) >= STACK_N or chan
            fi = mem[-1]
            out.append(dict(near=float(near), far=float(lv[fi]), mem=mem, big=isbig, chan=chan, nma=nma, fams=fams,
                            s_near=float(near - d * sph[mem[0]]), s_far=float(lv[fi] + d * sph[fi]),
                            mark=float(lv[fi] + d * BREAK_FRAC * sph[fi]),
                            kinds=[NAME[int(c)] for c in cds], label=self._label(cds, big[mem], fams, nma, chan)))
        return out

    @staticmethod
    def _label(cds, bigs, fams, nma, chan):
        dg = bool(((cds >= DSH) & (cds <= DBL)).any()) if len(cds) else False
        bigH = bool((bigs & (cds <= W1L)).any())
        if chan:
            return "channel edge"
        if dg and bigH:
            return "meeting point (diagonal + big line)"
        if nma >= 2:
            return "MA band (2+ averages together)"
        if len(fams - {4}) >= STACK_N:
            return "stack (3 kinds of line together)"
        if bigH:
            return "big: weekly / major daily line"
        if np.isin(cds, BIG_MA).any():
            return "big: an average"
        if ((cds == DBH) | (cds == DBL)).any():
            return "big: a big diagonal"
        names = {0: "4H lines", 1: "daily/weekly lines", 2: "4H 50 MA", 3: "small diagonal", 4: "channel edge"}
        return " + ".join(names[f] for f in sorted(fams)) + (" (only)" if len(fams) == 1 else "")

    def walls_ahead(self, b, d, p, reach=WALL_REACH):
        """Big walls whose first line is within `reach` 4H moves ahead of p (nearest first)."""
        return [B for B in self.bands(b, d, p, reach) if B["big"]]

    def walls_between(self, b, d, entry, target, reach=MAP_WIN):
        """Big walls between the entry and the target (target None: within `reach` 4H moves), nearest first -- the
        walls the lock and the staircase work on."""
        return [B for B in self.bands(b, d, entry, reach, limit=target) if B["big"]]

    def band_holding(self, b, p):
        """The band (either side) whose lines straddle price p -- 'in the middle of a wall' (5.1 option c)."""
        m = self.members(b)
        lv, a = m["lv"], m["a"]
        if not len(lv) or not (a == a and a > 0):
            return None
        for d in (1, -1):
            for B in self.bands(b, -d, p + d * BAND * a * 1.01, reach=2 * BAND):
                lo_, hi_ = min(B["near"], B["far"]), max(B["near"], B["far"])
                if lo_ < p < hi_ and B["big"]:
                    return B
        return None

    # -- channels ----------------------------------------------------------------------------------------------------
    def channels(self, b, _members=None):
        """Channels around price at 4H close b: a diagonal channel (a top line above and a bottom line below with
        nearly the same slope, at most 6 4H moves apart) and a horizontal range (2+ daily/weekly swing highs close
        together above and 2+ swing lows below, at most 6 moves apart). -> [dict(kind, top, bottom, slope)]"""
        if _members is None:
            cached = self._ch.get(b)
            if cached is not None:
                return cached
        if b < 0 or b >= self.n:
            return []
        a = self.a4_at(b)
        if not (a == a and a > 0):
            return []
        p = float(self.c4[b])
        tops, bots = [], []
        for g in self.diags + self.chlines:
            if b < g.start:
                continue
            if g.cls_at(b) == 0 or not g.awake_at(b):
                continue
            lvn = g.level(b + 1)
            if abs(lvn - p) > MAP_WIN * a:
                continue
            is_top = (g.role == "break" and g.d == 1) or (g.role == "channel" and g.d == 1)
            (tops if is_top else bots).append((lvn, g.slope))
        found = []
        best = None
        for vt, st in tops:
            if vt <= p:
                continue
            for vb, sb in bots:
                if vb >= p or st * sb <= 0 or vt - vb > CH_MAX * a:
                    continue
                if abs(st - sb) <= CH_SLOPE * max(abs(st), abs(sb)):
                    cand = (vt - vb, vt, vb, (st + sb) / 2.0)
                    if best is None or cand[0] < best[0]:
                        best = cand
        if best is not None:
            found.append(dict(kind="diagonal channel (%s)" % ("rising" if best[3] > 0 else "falling"),
                              top=float(best[1]), bottom=float(best[2]), slope=float(best[3])))
        m = _members if _members is not None else self.members(b)
        lv, code = m["lv"], m["code"]
        top = bot = None
        hi_ = [i for i in range(int(np.searchsorted(lv, p, side="right")), len(lv)) if code[i] in (D1H, W1H)]
        for q, i in enumerate(hi_):
            if sum(1 for j in hi_[q:] if lv[j] - lv[i] <= BAND * a) >= 2:
                top = float(lv[i])
                break
        lo_ = [i for i in range(int(np.searchsorted(lv, p, side="left")) - 1, -1, -1) if code[i] in (D1L, W1L)]
        for q, i in enumerate(lo_):
            if sum(1 for j in lo_[q:] if lv[i] - lv[j] <= BAND * a) >= 2:
                bot = float(lv[i])
                break
        if top is not None and bot is not None and top - bot <= CH_MAX * a:
            found.append(dict(kind="horizontal range", top=top, bottom=bot, slope=0.0))
        if _members is None:
            self._ch[b] = found
        return found

    def channel_target(self, b, d, entry):
        """6.2: inside a channel the target is the near edge of the other side's sphere (the side ahead). -> (level,
        channel) or (None, None) when price is not inside a channel."""
        best = None
        a = self.a4_at(b)
        if not (a == a and a > 0):
            return None, None
        for ch in self.channels(b):
            if not (ch["bottom"] < entry < ch["top"]):
                continue
            other = ch["top"] if d == 1 else ch["bottom"]
            lvl = other - d * SPH_BIG * a                 # channel edges are walls: a big line's sphere
            if best is None or d * (lvl - best[0]) < 0:
                best = (lvl, ch)
        return best if best is not None else (None, None)

    # -- witnesses, major lines, turning points, wild market --------------------------------------------------------
    def _hold_end_1h(self, line_level_at_x, d, t_break):
        """The first 1H close after t_break that is back across the line (None if none) -- 'no 1H close back across
        it since' (5.2)."""
        j0 = int(np.searchsorted(self.t1.values, np.datetime64(pd.Timestamp(t_break)), side="right"))
        for j in range(j0, len(self.c1)):
            x = int(np.searchsorted(self.t4, np.datetime64(self.t1[j] - pd.Timedelta(hours=1)), side="right"))
            lvl = line_level_at_x(x)
            if lvl == lvl and d * (self.c1[j] - lvl) < 0:
                return pd.Timestamp(self.t1[j])
        return None

    def witnesses(self, d, t_trigger, hours=WITNESS_H, need_retest=False):
        """5.2: the diagonals and averages that broke the same way (a 4H close past the three-quarter mark) in the
        `hours` before the trigger, still holding (no 1H close back across since; a dip into the sphere is fine).
        need_retest (the runner entry): the witness must also have been retested -- price entered its sphere after the
        break. -> [dict(kind, name, big, t_break, level)] newest first."""
        T = pd.Timestamp(t_trigger)
        lo_t = T - pd.Timedelta(hours=hours)
        bT = self.bucket(T)
        out = []
        for g in self.diags:
            if g.d != d:
                continue
            for x, hend, c0 in g.breaks:
                tb = pd.Timestamp(self.t4[x])
                if not (lo_t <= tb <= T) or x > bT:
                    continue
                he = self._hold_end_1h(g.level, d, tb)
                if he is not None and he <= T:
                    continue
                if need_retest and not self._retested(g.level, d, tb, T, (SPH_BIG if c0 == 2 else SPH_SMALL)):
                    continue
                out.append(dict(kind="diagonal", name=("big" if c0 == 2 else "small") + (" falling" if d == 1 else " rising")
                                + " diagonal", big=c0 == 2, t_break=tb, level=float(g.level(x))))
        ok = (self.a4 == self.a4) & (self.a4 > 0)
        for cd, v in self.ma_at.items():
            frac = (SPH_BIG if cd in BIG_MA else SPH_SMALL) * BREAK_FRAC
            k0 = max(1, self.bucket(lo_t))
            for x in range(k0, bT + 1):
                if not (ok[x] and v[x] == v[x] and v[x - 1] == v[x - 1] and ok[x - 1]):
                    continue
                tb = pd.Timestamp(self.t4[x])
                if not (lo_t <= tb <= T):
                    continue
                now_ = d * (self.c4[x] - v[x]) >= frac * self.a4[x]
                was = d * (self.c4[x - 1] - v[x - 1]) < frac * self.a4[x - 1]
                if not (now_ and was):
                    continue
                he = self._hold_end_1h(lambda xx, _v=v: float(_v[min(max(xx - 1, 0), len(_v) - 1)]), d, tb)
                if he is not None and he <= T:
                    continue
                if need_retest and not self._retested(lambda xx, _v=v: float(_v[min(max(xx - 1, 0), len(_v) - 1)]), d, tb,
                                                      T, SPH_BIG if cd in BIG_MA else SPH_SMALL):
                    continue
                out.append(dict(kind="average", name=NAME[cd], big=cd in BIG_MA, t_break=tb, level=float(v[x])))
        out.sort(key=lambda w: w["t_break"], reverse=True)
        return out

    def _retested(self, level_at, d, t_break, T, sph_moves):
        """Price came back into the line's sphere after the break (a 1H low for a buy / high for a sell), by T."""
        j0 = int(np.searchsorted(self.t1.values, np.datetime64(pd.Timestamp(t_break)), side="right"))
        j1 = int(np.searchsorted(self.t1.values, np.datetime64(pd.Timestamp(T)), side="right"))
        for j in range(j0, j1):
            x = int(np.searchsorted(self.t4, np.datetime64(self.t1[j] - pd.Timedelta(hours=1)), side="right"))
            lvl = level_at(x)
            a = self.a4_at(min(max(x - 1, 0), self.n - 1))
            if not (lvl == lvl and a == a and a > 0):
                continue
            ext = self.lo1[j] if d == 1 else self.hi1[j]
            if d * (ext - lvl) <= sph_moves * a:
                return True
        return False

    def is_major_4h(self, d, r2, conf, t):
        """14.2 B: is the signal's line a major 4H swing (the highest/lowest 4H close for 6 candles each side), known by
        time t? The line's own swing is the latest swing of its type at that level confirmed by the setup's birth."""
        code = H4H if d == 1 else H4L
        conf64, t64 = np.datetime64(pd.Timestamp(conf)), np.datetime64(pd.Timestamp(t))
        cand = np.nonzero((self.sw_code == code) & (np.abs(self.sw_lv - float(r2)) <= 1e-9 * max(1.0, abs(float(r2))))
                          & (self.sw_conf <= conf64))[0]
        if not len(cand):
            return False
        q = int(cand[-1])
        return bool(self.sw_maj[q] and self.sw_majt[q] <= t64)

    def turn_at(self, d, r1, conf):
        """Q6 A (9 Oct): did the setup's turning point (R1) sit inside the sphere of a major line or an average at the
        setup's birth? Major = a major 4H line, a weekly or major daily swing line, a big diagonal, or any of the six
        averages. -> (True/False, name)."""
        b = self.bucket(conf)
        if b < 0:
            return False, "no 4H move at the setup's birth"
        m = self.members(b)
        own = H4L if d == 1 else H4H
        for i in range(len(m["lv"])):
            cd = int(m["code"][i])
            if abs(m["lv"][i] - r1) > m["sph"][i]:
                continue
            if cd == own and abs(m["lv"][i] - r1) <= 1e-9 * max(1.0, abs(r1)):
                continue                                   # the turning point's own swing
            if cd in (CHT, CHB):
                continue
            if m["big"][i] or m["maj"][i] or MA4_50 <= cd <= MAW_200:
                return True, NAME[cd]
        return False, ""

    def wild(self, t):
        """5.3: the 4H move at t is at least 1.5 x its 60-day average. -> (True/False, ratio)."""
        b = self.bucket(t)
        if b < 0:
            return False, float("nan")
        t0 = np.datetime64(pd.Timestamp(self.t4[b]) - pd.Timedelta(days=WILD_DAYS))
        k0 = int(np.searchsorted(self.t4, t0, side="left"))
        win = self.a4[k0:b + 1]
        win = win[win == win]
        if len(win) < 60 or not (self.a4[b] == self.a4[b]):
            return False, float("nan")
        r = float(self.a4[b] / win.mean())
        return r >= WILD_K, r

    # -- for the chart -----------------------------------------------------------------------------------------------
    def chart_diagonals(self, b_from, b_to):
        """Every drawn diagonal between two 4H closes, as segments with their state at each candle: 'awake' (solid
        black), 'asleep' (dashed), 'broken' (solid dark red, from the first break on). -> [dict(ln, big, segs)] where segs
        = [(state, [x...], [level...])]."""
        out = []
        for g in self.diags:
            x0 = max(g.start, b_from)
            if x0 > b_to:
                continue
            brk0 = g.breaks[0][0] if g.breaks else None
            segs, cur, xs, ys = [], None, [], []
            anyd = False
            for x in range(x0, b_to + 1):
                c0 = g.cls_at(x)
                if c0 == 0 and (brk0 is None or x < brk0):
                    if xs:
                        segs.append((cur, xs, ys))
                        cur, xs, ys = None, [], []
                    continue
                anyd = True
                stt = "broken" if (brk0 is not None and x >= brk0) else ("awake" if g.awake_at(x) else "asleep")
                if stt != cur and xs:
                    segs.append((cur, xs + [x], ys + [g.level(x)]))
                    xs, ys = [], []
                cur = stt
                xs.append(x)
                ys.append(g.level(x))
            if xs:
                segs.append((cur, xs, ys))
            if anyd and segs:
                out.append(dict(ln=g.ln, big=bool(g.cls_at(b_to) == 2 or (g.cls[:max(1, b_to - g.start + 1)] == 2).any()),
                                segs=segs, d=g.d, i1=g.i1, v1=g.v1, slope=g.slope,
                                broken_at=brk0, open_end=(brk0 is None and g.cls_at(b_to) != 0)))
        return out

    def describe(self, b, around=2.5):
        """The map near price at one 4H close, band by band (logs, card, replays)."""
        m = self.members(b)
        if not len(m["lv"]):
            return ["(no lines)"]
        p, a = m["p"], m["a"]
        out = []
        for d in (1, -1):
            for B in self.bands(b, d, p, around):
                names = ["%.6g %s" % (m["lv"][i], NAME[int(m["code"][i])]) for i in B["mem"]
                         if m["code"][i] > H4L or m["maj"][i]]
                n4 = sum(1 for i in B["mem"] if m["code"][i] <= H4L and not m["maj"][i])
                if n4:
                    names.append("%d other 4H line%s" % (n4, "s" if n4 > 1 else ""))
                out.append((-B["near"], "%+5.2f moves  %.6g..%.6g  %s   [%s%s]" % (
                    (B["near"] - p) / a, min(B["near"], B["far"]), max(B["near"], B["far"]), ", ".join(names[:8]),
                    "BIG WALL: " if B["big"] else "", B["label"])))
        out.sort()
        return [x[1] for x in out] or ["(no lines within %.1f 4H moves)" % around]


# ---- the rules the engine, the package and the trade manager share --------------------------------------------------
def strong_through_walls(mp, b, d, r2, conf, t_trigger, wits):
    """5.1 (Desire 9 Oct): a signal with big walls ahead buys straight away when it is STRONG: 2 of the 3 kinds of line
    (horizontal, diagonal, average) broke the same way and all of them are big (the horizontal = a major 4H line), or
    all 3 kinds broke (any size). -> (True/False, reason)."""
    big_h = mp.is_major_4h(d, r2, conf, t_trigger)
    dg = [w for w in wits if w["kind"] == "diagonal"]
    ma = [w for w in wits if w["kind"] == "average"]
    if dg and ma:
        return True, "3 of 3 (line, %s, %s)" % (dg[0]["name"], ma[0]["name"])
    if big_h and any(w["big"] for w in dg):
        return True, "2 of 3, all big (major 4H line + %s)" % next(w["name"] for w in dg if w["big"])
    if big_h and any(w["big"] for w in ma):
        return True, "2 of 3, all big (major 4H line + %s)" % next(w["name"] for w in ma if w["big"])
    why = []
    if not big_h:
        why.append("the broken line is not a major 4H line")
    if not dg and not ma:
        why.append("no diagonal or average broke the same way")
    elif not any(w["big"] for w in dg + ma):
        why.append("only small lines broke with it")
    return False, "; ".join(why) or "not 2 of 3 all big"


def lock_level(walls, d, entry):
    """6.4: the stop goes to the entry when price first touches the near edge of the first big wall's sphere -- or,
    when the entry is already inside that sphere, the wall line itself. -> trigger price or None."""
    if not walls:
        return None
    B = walls[0]
    if d * (entry - B["s_near"]) >= 0:                 # already inside the first wall's sphere
        return B["near"]
    return B["s_near"]


def stair_step(walls, d, close_1h, cur_stop):
    """6.5: a 1H close beyond a big wall's sphere moves the stop to just behind that sphere (its edge facing the
    trade). Never back. -> new stop or None."""
    best = None
    for B in walls:
        if d * (close_1h - B["s_far"]) > 0:
            lvl = B["s_near"]
            if d * (lvl - cur_stop) > 0 and (best is None or d * (lvl - best) > 0):
                best = lvl
    return best


# ---- live: fetch the candles, build, keep the newest map per market -------------------------------------------------
SYMBOLS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}


def _mt5_frame(symbol, tf_name, n):
    import MetaTrader5 as mt5
    tf = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1}[tf_name]
    r = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if r is None or len(r) == 0:
        return None
    f = pd.DataFrame(r)
    f.index = pd.to_datetime(f["time"], unit="s")
    f = f[["open", "high", "low", "close"]].astype(float)
    span = {"H1": pd.Timedelta(hours=1), "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(days=7)}[tf_name]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return f[f.index + span <= now]                       # closed candles only


def build_live(asset, df1_engine=None):
    """Live: one year of hourly candles (MT5), daily and weekly candles (MT5; the weekly 200 needs ~4 years and is
    left out -- and logged -- when MT5 has fewer). The newest engine candles are appended so the map is never behind
    the engine. Returns the map (also kept in LATEST) or None."""
    a = str(asset).upper()
    sym = SYMBOLS.get(a)
    if sym is None:
        return None
    _last = None
    if df1_engine is not None and len(df1_engine):
        _last = pd.Timestamp(df1_engine.index[-1])
        if _last.tzinfo is not None:
            _last = _last.tz_convert("UTC").tz_localize(None)
        with _LOCK:
            _old = LATEST.get(a)
        if _old is not None and getattr(_old, "_engine_last", None) == _last:
            return _old                                   # nothing new has closed since this map was built
    try:
        h1 = _mt5_frame(sym, "H1", 8800)                  # a year of hourly candles (BTC trades 24/7)
        d1 = _mt5_frame(sym, "D1", 420)
        w1 = _mt5_frame(sym, "W1", 260)
    except Exception as e:
        logger.warning("[MAP] %s: candles from MT5 failed -- map not rebuilt this look (%s)", a, e)
        return LATEST.get(a)
    if h1 is None or len(h1) < 400:
        logger.warning("[MAP] %s: too few hourly candles from MT5 (%s) -- map not rebuilt this look", a,
                       0 if h1 is None else len(h1))
        return LATEST.get(a)
    if df1_engine is not None and len(df1_engine):
        e = df1_engine[["open", "high", "low", "close"]].astype(float)
        idx = e.index
        if getattr(idx, "tz", None) is not None:
            e = e.set_axis(idx.tz_convert("UTC").tz_localize(None))
        h1 = pd.concat([h1[h1.index < e.index[0]], e])
        h1 = h1[~h1.index.duplicated(keep="last")].sort_index()
    mp = MarketMap(a, h1, d1, w1)
    mp._engine_last = _last
    with _LOCK:
        LATEST[a] = mp
    try:
        b = mp.n - 1
        m = mp.members(b)
        n_awake = sum(1 for g in mp.diags if g.cls_at(b) and g.awake_at(b))
        logger.info("[MAP] %s: built from %d hourly candles (%s to %s), %d daily, %d weekly -- %d lines near price, "
                    "%d diagonals awake, %d channel(s)", a, len(h1), str(h1.index[0])[:16], str(h1.index[-1])[:16],
                    len(d1) if d1 is not None else 0, len(w1) if w1 is not None else 0, len(m["lv"]), n_awake,
                    len(mp.channels(b)))
    except Exception as _e:
        logger.warning("[MAP] %s: built, but its summary was not written: %s", a, _e)
    if mp.ma_missing:
        logger.info("[MAP] %s: left out for lack of MT5 history: %s", a, ", ".join(mp.ma_missing))
    return mp


def latest(asset):
    with _LOCK:
        return LATEST.get(str(asset).upper())
