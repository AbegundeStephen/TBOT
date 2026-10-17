"""B14 (Desire, 15 Oct 2026): his additions to the B13 rules, all in one place -- the engine, the trade manager, the
practice trader, the box test and the weekly job call these same functions, so they cannot drift apart.

What B14 adds to B13 (everything else is B13 as today; off = B13 exactly):
  1. the clearance: the close-through (the 1H close past the peak) must beat the peak by CLEAR_ATR1 of a 1H move;
  2. the widened confirmation: a signal with a bigger 4H line just ahead trades at once (instead of going to the
     package) when a WITNESS broke the same way in the last WITNESS_H hours -- a diagonal through 2+ swing closes, or
     one of the six MAs, broken by a 4H close at least BREAK_ATR4 of a 4H move past it (B13 item 32 only knew 3-touch
     diagonals). The witness never blocks a trade; it is recorded on every trade;
  3. the target: on the nearest line 2-3R from the entry (R = the stop distance); none there: the nearest out to 3.5R;
     none: today's target (BTC, which has none: BTC_FALLBACK_ATR1 1H moves). The lines are the map's: big 4H lines,
     daily and weekly close lines (last LINE_DAYS days), the six MAs, the big awake diagonals;
  4. the ladder: once price is 1R ahead the stop goes to +0.2R; at 1.6R to +0.8R; at 2R to +1.5R. Never back.
     BTC runs this too (no runner under B14: Desire 15 Oct, "general plan");
  5. the record: the witness, the target kind, the clearance, the lines ahead and the ladder steps on every trade.
Numbers: Desire 15 Oct 2026 (clearance 0.15; the ladder; the target 2-3R / 3.5R; the witness 2+ touches, MAs too)."""
import numpy as np
import pandas as pd

# ---- his numbers ------------------------------------------------------------------------------------------------------
CLEAR_ATR1 = 0.15                       # the close-through must beat the peak by this many 1H moves
LADDER = ((1.0, 0.2), (1.6, 0.8), (2.0, 1.5))   # (price this far ahead, in R) -> the stop moves to this (in R)
TARGET_FROM, TARGET_TO, TARGET_EXT = 2.0, 3.0, 3.5   # the target: a line 2-3R away; else out to 3.5R; else as today
BTC_FALLBACK_ATR1 = 4.0                 # BTC has no target of its own today: 4 hourly moves (Desire 9 Oct)
WITNESS_H = 24                          # a witness break counts for this many hours (B13 item 32's window)
DIAG_TOUCHES = 2                        # a diagonal through 2 swing closes counts (B13's finder wanted a third touch)
BREAK_ATR4 = 0.25                       # a witness break: a 4H close at least a quarter of a 4H move past the line
LINE_DAYS = 60                          # the horizontal lines come from the last 60 days
SPH_BIG, SPH_4H50 = 0.5, 0.25           # a line's band: half a 4H move each side (the 4H 50 MA: a quarter)
EPS = 1e-9
TF_RANK = {"1W": 0, "1D": 1, "4H": 2, "MA": 3, "DG": 4}
RULES_TAG = "B14"
# Re-arm (Desire 9 Oct, option A "retest"; kept in this batch 16 Oct): after a package cancel the setup goes back to
# the engine -- a fresh retest, then a 1H close past the best close so far (ns_package._rearm). The box test sets these
# per row; live they are as written here.
REARM = True
REARM_MAX = 5                  # second chances per setup
REARM_AFTER_MAJORITY = False   # Desire 16 Oct, option B: no re-arm after a "no bigger-timeframe majority" refusal
MAX_RUNGS = 6                  # a tuned ladder may have 1 to 6 rungs (the trade manager's guard knows all six)
# The springboard (Desire 16 Oct: "make sure a wall ahead has been broken and turned into a springboard before it enters,
# instead of entering into it" -- live, behind the B14 switch, walls closer than 1R): a package entry with a B14 line
# (the target's line set) less than SPRING_DIST_R ahead does not enter; it waits up to SPRING_WAIT_H hours for a 1H close
# past the line's band, a 1H candle back into the band that closes on the far side, then a 1H close past that candle's
# high (a sell: its low) -- and enters there (ns_package._spring_step).
SPRING = True
SPRING_DIST_R = 1.0            # Desire 16 Oct: 1R (the test also tried 0.5R)
SPRING_WAIT_H = 48
FAR_MAX = None                 # test rows only (Desire 16 Oct, the BTC 15 Oct short): a package entry confirmed more than
                               # this many 1H moves past its line is not taken. None = no limit (live)
                               # (it beat re-arming after every cancel on both samples: +3.0R and +1.2R)


def _lm():
    from src.execution import line_map as LM
    return LM


def _ne():
    from src.execution import ns_engine as NE
    return NE


# ---- 1. the clearance --------------------------------------------------------------------------------------------------
def clearance(d, close, peak, atr1):
    """How far the close beat the peak, in 1H moves (negative = not past it)."""
    return d * (float(close) - float(peak)) / float(atr1) if atr1 and atr1 == atr1 and atr1 > 0 else float("nan")


def clearance_ok(d, close, peak, atr1):
    c = clearance(d, close, peak, atr1)
    return c == c and c >= CLEAR_ATR1 - EPS


# ---- the lines (the map's set) ---------------------------------------------------------------------------------------
def _turn_arrays(mp, tf):
    cache = mp.__dict__.setdefault("_b14n_turns", {})
    if tf in cache:
        return cache[tf]
    rows = mp._close_turns(tf)
    never = np.datetime64("2200-01-01")
    arr = (np.array([r[0] for r in rows], dtype="datetime64[ns]"), np.array([r[1] for r in rows], dtype="datetime64[ns]"),
           np.array([r[2] for r in rows], dtype=float), np.array([r[3] == "close high" for r in rows], dtype=bool),
           np.array([r[4] if r[4] is not None else never for r in rows], dtype="datetime64[ns]"))
    cache[tf] = arr
    return arr


def hz_lines(mp, t):
    """The horizontal lines known at time t (a 1H close): big 4H lines (a major 4H swing close, 6 candles each side),
    daily close lines that beat 3 days each side, weekly close lines -- all from the last LINE_DAYS days. Known only
    once the candles after them have closed (no look-ahead). Lowest first; equal levels are one line."""
    b = mp.bucket(t)
    cache = mp.__dict__.setdefault("_b14n_hz", {})
    got = cache.get(b)
    if got is not None:
        return got
    if b < 0:
        return []
    LM = _lm()
    T = mp.t4[b]
    t0 = T - np.timedelta64(LINE_DAYS, "D")
    out = []
    m = mp.sw_maj & (mp.sw_majt <= T) & (mp.sw_i >= LM.MAJOR_K)
    for q in np.nonzero(m)[0]:
        ts = mp.t4[int(mp.sw_i[q])]
        if ts < t0 or ts > T:
            continue
        hi = int(mp.sw_code[q]) == LM.H4H
        dt = str(pd.Timestamp(ts))[:16]
        out.append(dict(id="4H|%s|%s" % (dt, "h" if hi else "l"), lv=float(mp.sw_lv[q]), tf="4H", kind="high" if hi else "low",
                        date=dt, fam="line", sph=SPH_BIG, name="big 4H %s, the 4H close of %s" % ("high" if hi else "low", dt)))
    for tf in ("1D", "1W"):
        kn, st, lv, hi, bk = _turn_arrays(mp, tf)
        if not len(kn):
            continue
        for q in np.nonzero((bk <= T) & (st >= t0))[0]:
            dt = str(pd.Timestamp(st[q]))[:10]
            hq = bool(hi[q])
            out.append(dict(id="%s|%s|%s" % (tf, dt, "h" if hq else "l"), lv=float(lv[q]), tf=tf, kind="high" if hq else "low",
                            date=dt, fam="line", sph=SPH_BIG,
                            name=("weekly close %s, week of %s" if tf == "1W" else "daily close %s, %s") % ("high" if hq else "low", dt)))
    out.sort(key=lambda L: (TF_RANK[L["tf"]], L["date"]))
    keep = []
    for L in out:
        if any(abs(K["lv"] - L["lv"]) <= 1e-9 * max(1.0, abs(L["lv"])) for K in keep):
            continue
        keep.append(L)
    keep.sort(key=lambda L: L["lv"])
    cache[b] = keep
    return keep


def moving_lines(mp, x):
    """The six MAs and the big awake diagonals for the map's 4H candle x (in progress): an MA at its value after
    candle x-1, a diagonal at its level on candle x. Cached per x."""
    cache = mp.__dict__.setdefault("_b14n_mv", {})
    got = cache.get(x)
    if got is not None:
        return got
    LM = _lm()
    out = []
    if 1 <= x <= mp.n:
        for cd, arr in mp.ma_at.items():
            v = float(arr[x - 1]) if x - 1 < len(arr) else float("nan")
            if v == v:
                out.append(dict(id="MA|%d" % int(cd), lv=v, fam="average", name=LM.NAME[int(cd)],
                                sph=SPH_4H50 if int(cd) == LM.MA4_50 else SPH_BIG, code=int(cd)))
        for g in mp.diags:
            if g.start > x - 1:
                continue
            if g.cls_at(x - 1) == 2 and g.awake_at(x - 1):
                ta, tb = str(pd.Timestamp(mp.t4[int(g.i1)]))[:16], str(pd.Timestamp(mp.t4[int(g.i2)]))[:16]
                out.append(dict(id="DG|%s|%s|%+d" % (ta, tb, int(g.d)), lv=float(g.level(x)), fam="diagonal",
                                name="big %s diagonal" % ("falling" if g.d == 1 else "rising"), sph=SPH_BIG,
                                code=int(LM.DBH if g.d == 1 else LM.DBL)))
    cache[x] = out
    return out


def all_lines(mp, t, a4):
    """Every B14 line at time t (a 1H close): the horizontal lines known by t, the MAs and the big awake diagonals,
    each with w = its band's half-width in price. Lowest first."""
    out = [dict(L, w=float(L["sph"]) * float(a4)) for L in hz_lines(mp, t)]
    for L in moving_lines(mp, mp.bucket(t) + 1):
        out.append(dict(L, w=float(L["sph"]) * float(a4)))
    out.sort(key=lambda L: L["lv"])
    return out


def line_text(L):
    return "none" if not L else "%.6g (%s)" % (float(L["lv"]), L.get("name", "?"))


# ---- 3. the target -----------------------------------------------------------------------------------------------------
def line_target(d, e, risk, lines, fallback, fallback_text="today's target"):
    """The nearest line TARGET_FROM-TARGET_TO R from the entry; none: the nearest out to TARGET_EXT R; none: the
    fallback (today's target; for BTC the 4-move target). -> (target, the line or None, kind, text).
    kind: "line" / "line_ext" / "fallback"."""
    e, risk = float(e), float(risk)
    if risk <= 0:
        return fallback, None, "fallback", fallback_text
    ahead = [(d * (float(L["lv"]) - e) / risk, L) for L in lines if d * (float(L["lv"]) - e) > 0]
    for kind, lo, hi in (("line", TARGET_FROM, TARGET_TO), ("line_ext", TARGET_TO, TARGET_EXT)):
        cand = [(r, L) for r, L in ahead if lo - EPS <= r <= hi + EPS]
        if cand:
            r, L = min(cand, key=lambda z: z[0])
            return float(L["lv"]), L, kind, "on %s, %.2fR away" % (line_text(L), r)
    return fallback, None, "fallback", fallback_text + " (no line 2 to 3.5R away)"


def wall_ahead(d, e, risk, lines, dist_r):
    """The springboard's wall (Desire 16 Oct): the nearest line ahead of the entry e, less than dist_r R away. None: no
    wall (or no risk to measure by)."""
    if risk <= 0:
        return None
    ahead = [(d * (float(L["lv"]) - e), L) for L in lines if 0 < d * (float(L["lv"]) - e) <= dist_r * risk]
    return min(ahead, key=lambda z: z[0])[1] if ahead else None


def lines_ahead(d, e, risk, lines, n=3):
    """The first n lines ahead of the entry, as (distance in R, text) -- for the record and the cards."""
    ahead = sorted(((d * (float(L["lv"]) - e) / risk, L) for L in lines if d * (float(L["lv"]) - e) > 0), key=lambda z: z[0])
    return [("%.2fR" % r, line_text(L)) for r, L in ahead[:n]] if risk > 0 else []


# ---- 2. the witness ------------------------------------------------------------------------------------------------------
def diag_breaks(t4, a4, c4, min_touch=DIAG_TOUCHES):
    """ns_engine.diagonal_breaks_4h with the touch count as a parameter: lines through consecutive 4H swing closes
    (falling through highs for a buy, rising through lows for a sell); with min_touch 3 the line must be touched a
    third time first (B13 item 32, today's finder); with 2 the two swings alone make the line. Watched for DIAG_LIFE
    4H candles or until the next swing; a break = a 4H close at least BREAK_ATR4 of a 4H move past it.
    -> [(direction, break time (a 4H close), name)]"""
    NE = _ne()
    K, LIFE = NE.K, NE.DIAG_LIFE
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
            end = min(n, start + LIFE, nxt + K)
            if min_touch <= 2:
                vf = start
            else:
                earlier = P[max(0, j - 5):j - 1]
                vf = start if any(a4[p] == a4[p] and abs(c4[p] - (v1 + slope * (p - i1))) <= 0.5 * a4[p] for p in earlier) else None
            for x in range(start, end):
                lv = v1 + slope * (x - i1)
                if vf is None:
                    if x >= i2 + 3 and a4[x] == a4[x] and abs(c4[x] - lv) <= 0.5 * a4[x]:
                        vf = x + 1
                    continue
                if x < vf or not (a4[x] == a4[x] and a4[x] > 0):
                    continue
                if d * (c4[x] - lv) >= BREAK_ATR4 * a4[x] - EPS:
                    out.append((d, pd.Timestamp(t4[x]), "%s diagonal through %s and %s" % (
                        "falling" if d == 1 else "rising", str(pd.Timestamp(t4[i1]))[:16], str(pd.Timestamp(t4[i2]))[:16])))
                    break
    return out


def ma_breaks(mp):
    """The six MAs, each broken by a 4H close at least BREAK_ATR4 of a 4H move past it when the close before was not
    (a fresh cross), on the map's own 4H candles. -> [(direction, break time (a 4H close), name)]"""
    NE, LM = _ne(), _lm()
    c4 = np.asarray(mp.c4, dtype=float)
    a4 = NE.atr14(np.asarray(mp.hi4, dtype=float), np.asarray(mp.lo4, dtype=float), c4)
    out = []
    for cd, arr in mp.ma_at.items():
        arr = np.asarray(arr, dtype=float)
        nm = LM.NAME[int(cd)]
        for x in range(2, min(len(arr), len(c4))):
            m_now, m_prev = arr[x - 1], arr[x - 2]
            if not (m_now == m_now and m_prev == m_prev and a4[x] == a4[x] and a4[x] > 0 and a4[x - 1] == a4[x - 1]):
                continue
            for d in (1, -1):
                if d * (c4[x] - m_now) >= BREAK_ATR4 * a4[x] - EPS and d * (c4[x - 1] - m_prev) < BREAK_ATR4 * a4[x - 1] - EPS:
                    out.append((d, pd.Timestamp(mp.t4[x]), nm))
    return out


def witness_breaks(mp, t4, a4, c4):
    """Every witness break the engine can see: the 2+-touch diagonals on the engine's own 4H candles (t4, a4, c4 --
    exactly what diagonal_breaks_4h gets) and the six MAs on the map. Sorted by time."""
    out = diag_breaks(t4, a4, c4, DIAG_TOUCHES) + (ma_breaks(mp) if mp is not None else [])
    out.sort(key=lambda z: z[1])
    return out


def witness_at(breaks, d, t, hours=WITNESS_H):
    """The latest witness break the same way in the `hours` before t (inclusive). -> (found, name, time or None)"""
    tt = pd.Timestamp(t)
    best = None
    for dd, tb, nm in breaks:
        if dd == d and tt - pd.Timedelta(hours=hours) <= tb <= tt and (best is None or tb > best[1]):
            best = (nm, tb)
    return (best is not None, best[0] if best else None, str(best[1]) if best else None)


# ---- 4. the ladder --------------------------------------------------------------------------------------------------------
def exit_init(d, e, stop0, target, ladder=None):
    """The trade's B14 exit state (a plain dict, saved with the trade). `ladder` = the rungs the proof was made with
    (fields ns_b14n_ladder_def); the trade keeps them even if the settings change later (Desire 16 Oct)."""
    return {"risk0": abs(float(e) - float(stop0)), "step": 0, "target": (float(target) if target is not None else None),
            "best": float(e), "ladder": [[float(a), float(b)] for a, b in (ladder or LADDER)]}


def ladder_stop(st, d, e, px, cur_stop):
    """A price (a tick live; a candle's best price in the box test): the R-ladder. The stop moves to LADDER[k][1] R
    once price has been LADDER[k][0] R ahead. Never back; a stop that is not tighter is not returned.
    -> (new stop or None, reason)"""
    e, px = float(e), float(px)
    R = float(st["risk0"])
    if R <= 0:
        return None, None
    st["best"] = max(float(st["best"]), px) if d == 1 else min(float(st["best"]), px)
    best, why, base = None, None, float(cur_stop)
    for k, (trig_r, lock_r) in enumerate(st.get("ladder") or LADDER):    # the trade's own rungs (older states: today's)
        if d * (float(st["best"]) - e) >= trig_r * R - EPS:
            lk = e + d * lock_r * R
            if d * (lk - base) > 0:
                best, why, base = lk, "b14_ladder%d" % (k + 1), lk
                st["step"] = max(int(st["step"]), k + 1)
    return best, why


STOP_REASONS = tuple("b14_ladder%d" % (k + 1) for k in range(MAX_RUNGS))   # the trade manager's guard lets these through


# ---- 5. the record ------------------------------------------------------------------------------------------------------
def fields(witness, target_kind, target_text, clear, ahead, lines_n):
    """The B14 facts on a proof: witness (found, name, time), target kind/text, the clearance in 1H moves, the lines
    ahead at entry, how many lines the map had."""
    found, nm, tb = witness
    return {"ns_b14n": True, "ns_b14n_witness": bool(found), "ns_b14n_witness_name": nm, "ns_b14n_witness_t": tb,
            "ns_b14n_target_kind": target_kind, "ns_b14n_target_text": target_text,
            "ns_b14n_clearance": (round(float(clear), 4) if clear == clear else None),
            "ns_b14n_ahead": ["%s %s" % x for x in ahead], "ns_b14n_lines": int(lines_n), "ns_b14n_ladder": 0,
            "ns_b14n_ladder_def": [[float(a), float(b)] for a, b in LADDER]}       # the rungs this trade keeps


def vtm_fields(kind, target_atr):
    """The two proof fields the trade manager reads to set the target (veteran_trade_manager ~2234-2246, unchanged):
    a line target -> ns_target_kind "B14 line" / "B14 line_ext" (the manager uses ns_target as it is); the fallback ->
    no ns_target_kind (the manager recomputes entry + ns_target_atr x 1H move from the fill, as today), with
    ns_target_atr = BTC_FALLBACK_ATR1 for a market whose own setting is None (BTC: 4 moves, no runner)."""
    if kind in ("line", "line_ext"):
        return {"ns_target_kind": "B14 %s" % kind}
    return {"ns_target_atr": float(target_atr) if target_atr is not None else float(BTC_FALLBACK_ATR1)}


def live_rules_text(pc):
    """The start-up line: the B14 rules as the settings read them."""
    on = bool((pc or {}).get("b14_new_rules_enabled", False))
    if not on:
        return "B14 rules OFF (phase_config.b14_new_rules_enabled) -- the B13 rules trade as today"
    lad = ", ".join("%.3gR -> +%.3gR" % (a, b) for a, b in LADDER)
    return ("B14 rules ON (phase_config.b14_new_rules_enabled) | B13 as today, plus: the close-through must beat the peak by "
            "%.2f of a 1H move | a signal with a bigger 4H line just ahead trades at once when a witness (a 2+-touch "
            "diagonal or one of the six MAs) broke the same way in the last %d h, else the package as today | target on "
            "the nearest line %.3g-%.3gR away, else out to %.3gR, else today's target (BTC: %.3g 1H moves) | the ladder: "
            "%s | BTC on the general plan (no runner) | %s | every trade records its witness, target kind, clearance and "
            "ladder steps" % (CLEAR_ATR1, WITNESS_H, TARGET_FROM, TARGET_TO, TARGET_EXT, BTC_FALLBACK_ATR1, lad,
                              ("a package entry with a line less than %gR ahead waits for the springboard (a close past it, "
                               "a retest that holds, a close-through; %gh)" % (SPRING_DIST_R, SPRING_WAIT_H))
                              if SPRING else "no springboard"))


# ---- the settings the learner may tune (Desire 16 Oct) -----------------------------------------------------------------
# Today's values are the ones written at the top. phase_config.b14_tuning, read ONCE at start-up (main.py calls
# apply_settings before the first look), may set any of these names; a name it does not give -- or a value that fails
# its check -- keeps today's value, and a refused value is said as an ERROR at start-up (never silent). An open trade
# keeps the ladder it opened with (it rides on the proof: ns_b14n_ladder_def). The learner (tools) writes suggestions
# only; Desire decides; the apply tool writes the approved value into phase_config.b14_tuning with a backup.
TUNABLE = ("CLEAR_ATR1", "LADDER", "TARGET_FROM", "TARGET_TO", "TARGET_EXT", "BTC_FALLBACK_ATR1",
           "REARM", "REARM_MAX", "REARM_AFTER_MAJORITY", "SPRING", "SPRING_DIST_R", "SPRING_WAIT_H", "FAR_MAX")
DEFAULTS = {k: globals()[k] for k in TUNABLE}        # today's values, kept for every reset


def _num(v, lo, hi, name, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, "%s must be a number" % name
    v = float(v)
    if not (v == v) or v > hi or v < lo or (lo_open and v <= lo):
        return None, "%s=%g outside %s%g..%g]" % (name, v, "(" if lo_open else "[", lo, hi)
    return v, None


def check_setting(name, v):
    """One setting -> (the value as the rules use it, None) or (None, why it is refused)."""
    if name == "LADDER":
        if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= MAX_RUNGS:
            return None, "LADDER must be a list of 1 to %d [trigger R, stop R] pairs" % MAX_RUNGS
        out, last_t, last_l = [], 0.0, -1.0
        for p in v:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return None, "each LADDER rung is a pair [trigger R, stop R]"
            a, why = _num(p[0], 0.0, 10.0, "trigger", lo_open=True)
            if why:
                return None, why
            b, why = _num(p[1], 0.0, 10.0, "stop")
            if why:
                return None, why
            if not (a > last_t and b < a and b >= last_l):
                return None, "LADDER rungs must climb: each trigger higher, each stop below its trigger and not lower than the last"
            out.append((a, b))
            last_t, last_l = a, b
        return tuple(out), None
    if name == "CLEAR_ATR1":
        return _num(v, 0.0, 1.0, name)
    if name in ("TARGET_FROM", "TARGET_TO", "TARGET_EXT"):
        return _num(v, 0.5, 10.0, name, lo_open=True)
    if name == "BTC_FALLBACK_ATR1":
        return _num(v, 0.5, 20.0, name, lo_open=True)
    if name in ("REARM", "REARM_AFTER_MAJORITY"):
        return (v, None) if isinstance(v, bool) else (None, "%s must be true or false" % name)
    if name == "REARM_MAX":
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 10:
            return None, "REARM_MAX must be a whole number 0..10"
        return int(v), None
    if name == "SPRING":
        return (v, None) if isinstance(v, bool) else (None, "SPRING must be true or false")
    if name == "SPRING_DIST_R":
        return _num(v, 0.1, 5.0, name)
    if name == "SPRING_WAIT_H":
        return _num(v, 4.0, 168.0, name)
    if name == "FAR_MAX":
        return (None, None) if v is None else _num(v, 0.5, 20.0, name)
    return None, "not a setting the learner may change"


def apply_settings(tuning):
    """tuning = phase_config.b14_tuning (a dict; empty or None = today's values). Resets every setting to today's value,
    then applies the valid ones. -> (values in force, [(name, today's, new)], [(name, why refused)])"""
    g = globals()
    for k in TUNABLE:
        g[k] = DEFAULTS[k]
    changed, refused = [], []
    if tuning is not None and not isinstance(tuning, dict):
        refused.append(("(the whole block)", "must be a block of settings {name: value}, not a %s (every setting)"
                        % type(tuning).__name__))
        tuning = None
    for k, v in (tuning or {}).items():
        val, why = check_setting(k, v)
        if why:
            refused.append((k, why))
            continue
        g[k] = val
    if not (g["TARGET_FROM"] < g["TARGET_TO"] <= g["TARGET_EXT"]):
        refused.append(("TARGET_FROM/TO/EXT", "must rise (from < to <= extension): %g / %g / %g -- all three kept at "
                        "today's" % (g["TARGET_FROM"], g["TARGET_TO"], g["TARGET_EXT"])))
        for k in ("TARGET_FROM", "TARGET_TO", "TARGET_EXT"):
            g[k] = DEFAULTS[k]
    for k in TUNABLE:
        if g[k] != DEFAULTS[k]:
            changed.append((k, DEFAULTS[k], g[k]))
    return {k: g[k] for k in TUNABLE}, changed, refused


def settings_text(values=None):
    """The settings in force, as one short line (the start-up line, the learner's report)."""
    v = values or {k: globals()[k] for k in TUNABLE}
    return ("clearance %.3g | ladder %s | target %.3g-%.3gR, out to %.3gR | BTC fallback %.3g moves | re-arm %s (max %d, "
            "after a bigger-TF refusal: %s)" % (v["CLEAR_ATR1"], ", ".join("+%gR->+%gR" % (a, b) for a, b in v["LADDER"]),
                                                v["TARGET_FROM"], v["TARGET_TO"], v["TARGET_EXT"], v["BTC_FALLBACK_ATR1"],
                                                "on" if v["REARM"] else "off", v["REARM_MAX"],
                                                "yes" if v["REARM_AFTER_MAJORITY"] else "no")
            + (" | springboard: walls < %gR, %gh" % (v["SPRING_DIST_R"], v["SPRING_WAIT_H"]) if v["SPRING"] else " | springboard off")
            + (" | far limit %g moves" % v["FAR_MAX"] if v["FAR_MAX"] is not None else ""))
