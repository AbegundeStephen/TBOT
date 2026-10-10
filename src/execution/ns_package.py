"""B13 (Desire 2 Oct 2026): THE PACKAGE -- judges the E signals the engine hands over.

The engine (ns_engine.py) hands over an E signal (4H break, 1H retest, 1H close past the peak) when it broke a small
line with a bigger 4H line just ahead (within half a 4H move), or when it is too far from the line to take at once.
Everything else is traded by the engine as before.

The package decides with the evidence -- version A since B13 item 29 (Desire 5 Oct): proof by closes only.
  1. Bigger timeframes first (once, at the signal): weekly and daily count as WITH when 2 of 3 agree -- price past the
     50 EMA with the 50 past the 200; swing structure (higher highs and lows for buys); a diagonal through the last two
     swing points, confirmed by a third touch. The 4H counts when its close is past the 20 EMA and the 20 past the 50.
     2 of the 3 needed -- otherwise no trade.
  2. The push, at every 30m close: STRONG = the 1H and the last two 30m closes past the trigger (and past the bigger
     line), the newest the furthest -> enter. While dithering, the re-read: the forming 4H candle strong (top third),
     the 30m lows stepping up, price above a rising 30m 20 EMA, the 1H past the trigger -- and only once the newest 30m
     close is already past the wall (the bigger line, or the trigger when there is none) -> enter. WEAK = a 1H close, or
     two 30m closes, back past the line -> cancel. Otherwise keep watching.
  3. 24 hours at most; a 4H close past the origin kills it.
  4. Stop: 0.3 of a 1H move behind the push's extreme since the retest, never closer than 1 move (breathing space).
     Far entries (over 2.5 moves from the line): behind the last 30m higher low, never closer than 1 move; no trade if
     that needs more than 3 moves. Target: the market's normal target. The engine's reward:risk gate applies.
Read-only towards MT5 (candles only). Every decision is logged ([PKG-...]) and returned as an event for Telegram.

B14 (Desire 8-9 Oct):
  1.1 An entry is kept on a board (st["pkg_ready"]) until the trading look uses it -- once, and only until the next
      30-minute close -- so an entry made during the first (recording) look is no longer lost. Each decision goes to
      Telegram the moment it is made (add_sink). main.py states what the trading look did with every entry
      ([PKG-OUTCOME]); an entry no look reached is said loudly when it runs out.
  With the B14 rules on (records marked "b14"):
  5.1 option (c): a signal held back by a big wall buys only after its push has beaten the first wall (a close past the
      wall's three-quarter mark), and never while price sits inside a wall; re-checked at the entry.
  6.2 inside a channel the target is the near edge of the other side's sphere when that comes first, with at least
      0.5R of room -- otherwise the package keeps waiting.
  5.10 re-arm (option A, "retest"): after a cancel the setup goes back to the engine, waiting for a fresh retest and
      then a 1H close past the best close so far -- until a 4H close past its kill line, 5 re-arms, or the 7-day waits.
"""
import copy
import logging

import numpy as np
import pandas as pd

from src.execution.ns_engine import (STOP_CAP_ATR, atr14, close_times, gate_rr_ok, hourly_to_4h,
                                     market_settings)

RE_READ = False                # B13 item 29 (Desire 5 Oct): version A -- the re-read shortcut is off (True = middle way)
logger = logging.getLogger(__name__)

WAIT_H = 24          # Desire 2 Oct: 24 hours
NEAR = 0.5           # the engine's PKG_NEAR: a bigger 4H line this close ahead must be cleared too
FAR = 2.5            # an entry further than this from the line (1H moves) is a far entry
FLOOR = 1.0          # breathing space: a stop is never closer than 1 move
CAP = 3.0            # a far entry needing a wider stop than this is not taken
KEEP_DONE = 20       # finished package records kept for display
READY_FOR = pd.Timedelta(minutes=30)    # B14 item 1.1: an entry waits for the trading look until the next 30m close
KEEP_READY = 20
SYMBOLS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}
_SPAN = {"M30": pd.Timedelta(minutes=30), "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(days=7)}

_SINKS = []          # B14 item 1.1b: called with every event the moment it happens (main.py: Telegram)


def add_sink(fn):
    """B14 item 1.1b: main.py registers its Telegram sender here, so a package decision reaches Telegram from
    whichever look makes it (before B14 only the trading look's decisions were sent)."""
    if fn not in _SINKS:
        _SINKS.append(fn)


def _emit(events, ev):
    events.append(ev)
    for fn in list(_SINKS):
        try:
            fn(ev)
        except Exception as _e:
            logger.warning("[PKG] %s: Telegram notice failed: %s", ev.get("asset"), _e)


def _now_utc():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _mt5_frame(asset, tf_name, n, now):
    """Closed candles straight from MT5 (count-based: UTC, the same clock as the bot's 1H file)."""
    import MetaTrader5 as mt5
    tf = {"M30": mt5.TIMEFRAME_M30, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1}[tf_name]
    r = mt5.copy_rates_from_pos(SYMBOLS[asset], tf, 0, n)
    if r is None or len(r) == 0:
        return None
    f = pd.DataFrame(r)
    f.index = pd.to_datetime(f["time"], unit="s")
    f = f[["open", "high", "low", "close"]].astype(float)
    return f[f.index + _SPAN[tf_name] <= now]          # never a candle still forming


def _bigger_read(df, span, t, d):
    """+1 when this weekly/daily frame is WITH the trade, -1 against, 0 mixed (2 of 3 pieces of evidence)."""
    f = df[df.index + span <= pd.Timestamp(t)]
    if len(f) < 31:
        return 0
    c, h, l = f["close"].values, f["high"].values, f["low"].values
    n, last = len(c) - 1, c[-1]
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values[-1]
    e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values[-1]
    ema = 1 if (last > e50 and e50 > e200) else -1 if (last < e50 and e50 < e200) else 0
    sw = []
    for i in range(2, len(c) - 2):
        w = c[i - 2:i + 3]
        if c[i] == w.max():
            sw.append((i, float(c[i]), "H"))
        if c[i] == w.min():
            sw.append((i, float(c[i]), "L"))
    H = [x for x in sw if x[2] == "H"]
    L = [x for x in sw if x[2] == "L"]
    hor = 0
    if len(H) >= 2 and len(L) >= 2:
        up = (H[-1][1] > H[-2][1] and L[-1][1] > L[-2][1]) or last > H[-1][1]
        dn = (H[-1][1] < H[-2][1] and L[-1][1] < L[-2][1]) or last < L[-1][1]
        hor = 1 if up and not dn else -1 if dn and not up else 0
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    a = pd.Series(np.r_[h[0] - l[0], tr]).rolling(14, min_periods=1).mean().values[-1]

    def diag(pts, kind):
        if len(pts) < 3:
            return None
        (i1, v1, _), (i2, v2, _) = pts[-2], pts[-1]
        if (kind == "L" and v2 <= v1) or (kind == "H" and v2 >= v1) or i2 == i1:
            return None
        slope = (v2 - v1) / (i2 - i1)
        third = any(abs(v - (v1 + slope * (i - i1))) <= 0.5 * a for i, v, _ in pts[:-2][-4:]) or \
            any(abs(c[j] - (v1 + slope * (j - i1))) <= 0.5 * a for j in range(i2 + 3, n + 1))
        return (v1 + slope * (n - i1)) if third else None
    sup, res = diag(L, "L"), diag(H, "H")
    dia = 1 if (sup is not None and last > sup) else -1 if (res is not None and last < res) else 0
    v = [ema, hor, dia] if d == 1 else [-ema, -hor, -dia]
    return 1 if v.count(1) >= 2 else -1 if v.count(-1) >= 2 else 0


def _strong_4h(t4, c4, t, d):
    k = int(np.searchsorted(t4.values, np.datetime64(pd.Timestamp(t)), side="right")) - 1
    if k < 50:
        return False
    s = pd.Series(c4[:k + 1])
    e20, e50 = s.ewm(span=20, adjust=False).mean().values[-1], s.ewm(span=50, adjust=False).mean().values[-1]
    return (c4[k] > e20 and e20 > e50) if d == 1 else (c4[k] < e20 and e20 < e50)


def _proof(rec, q, tau, stop, tgt, atr, how, wait_h, extra=None):
    f = dict(rec["fields"])
    for key in ("ns_aplus", "ns_watch"):
        f[key] = list(f.get(key) or [])
    f.update(ns_entry="PKG", ns_close=float(q), ns_stop=float(stop), ns_target=(float(tgt) if tgt is not None else None),
             ns_atr1=float(atr), ns_candle=str(tau), brc_first_confirmed_ts=str(tau),
             pkg_route=rec["route"], pkg_how=how, pkg_wait_h=round(float(wait_h), 1), pkg_bigger=rec.get("big"),
             pkg_majority=rec.get("maj"), pkg_signal_t=rec["tE"], pkg_signal_close=rec["eE"])
    if extra:
        f.update(extra)
    f["ns_aplus"].append("package: %s, after %.1fh" % (how, wait_h))
    return {"key": (rec["asset"], rec["kind"], rec["d"], round(float(rec["r2"]), 8)), "kind": rec["kind"],
            "dir": rec["d"], "ref": float(rec["r2"]), "tests": 0, "first_ts": str(tau), "fields": f}


def _finish(rec, status, why, events, tau, st=None, proof=None, note=None):
    rec.update(status=status, why_done=why, done_t=str(tau))
    if "why" not in rec:
        rec["why"] = why                          # B13 records kept their reason in "why"
    logger.info("[PKG-%s] %s: dir=%+d R2=%.5g (signal %s) -- %s", "ENTER" if status == "entered" else "CANCEL",
                rec["asset"], rec["d"], rec["r2"], rec["tE"], why)
    if status == "entered" and st is not None and proof is not None:
        # B14 item 1.1a: on the board until the trading look uses it, once, and only until the next 30m close
        st.setdefault("pkg_ready", []).append({"proof": proof, "t": str(tau), "until": str(pd.Timestamp(tau) + READY_FOR),
                                               "used": False, "asset": rec["asset"], "dir": rec["d"], "r2": rec["r2"]})
        st["pkg_ready"] = st["pkg_ready"][-KEEP_READY:]
    _emit(events, {"type": status, "asset": rec["asset"], "dir": rec["d"],
                   "text": "<b>PACKAGE %s -- %s %s</b>\n%s%s" % ("ENTERED" if status == "entered" else "CANCELLED",
                                                                rec["asset"], "BUY" if rec["d"] == 1 else "SELL", why,
                                                                ("\n" + note) if note else "")})


def ready_proofs(st, now=None, asset=None):
    """B14 item 1.1a: entries on the board, not used yet and before their next 30m close. An entry that ran out with
    no trading look reaching it is said once, loudly."""
    out = []
    now = pd.Timestamp(now) if now is not None else _now_utc()
    for r in (st or {}).get("pkg_ready", []):
        if r.get("used"):
            continue
        if now >= pd.Timestamp(r["until"]):
            r["used"] = True
            r["outcome"] = "expired"
            logger.warning("[PKG-OUTCOME] %s: dir=%+d R2=%.5g (entry %s) -- EXPIRED: no trading look reached it before "
                           "the next 30-minute close -- not traded", r.get("asset", asset), r["dir"], r["r2"], r["t"])
            continue
        out.append(r["proof"])
    return out


def settle(st, asset, outcome, detail, now=None):
    """B14 item 1.1c: called by main.py after every trading look. Every entry still on the board for this market has
    now been used once: it either became an order or was refused, and the [PKG-OUTCOME] line says which and why."""
    n = 0
    now = pd.Timestamp(now) if now is not None else _now_utc()
    for r in (st or {}).get("pkg_ready", []):
        if r.get("used") or now >= pd.Timestamp(r["until"]):
            continue
        r["used"] = True
        r["outcome"] = outcome
        n += 1
        (logger.info if outcome == "order" else logger.warning)(
            "[PKG-OUTCOME] %s: dir=%+d R2=%.5g (entry %s) -- %s: %s", asset, r["dir"], r["r2"], r["t"],
            "ORDER SENT" if outcome == "order" else "REFUSED", detail)
    return n


def _rearm(st, rec, why, tau, df1, kn, events):
    """B14 item 5.10 (Desire 9 Oct, option A "retest"): after a cancel the setup goes back to the engine at stage 1 --
    it needs a fresh retest, then a 1H close past the best close so far. It still ends on a 4H close past its kill
    line, after 5 re-arms, or after the 7-day waits. Returns the note for the Telegram, or None."""
    s0 = rec.get("setup")
    if not s0 or not kn.get("rearm", True):
        return None
    if why.startswith("setup died"):
        return None
    if why.startswith("no bigger-timeframe majority") and not kn.get("rearm_after_majority", True):
        return None
    n = int(rec.get("rearm_n", 0) or 0) + 1
    if n > int(kn.get("rearm_max", 5)):
        logger.info("[PKG-REARM] %s: dir=%+d R2=%.5g -- not re-armed: %d re-arms used", rec["asset"], rec["d"], rec["r2"],
                    n - 1)
        return "not re-armed: all %d second chances used" % (n - 1)
    s = copy.deepcopy(s0)
    d = int(s["d"])
    t1 = close_times(df1, 1)
    c1 = df1["close"].astype(float).values
    tb = pd.Timestamp(s.get("t_break") or rec["tE"])
    m = (t1 > tb) & (t1 <= pd.Timestamp(tau))
    best = float(s.get("h2") or rec["trig"])
    if m.any():
        best = max(best, float(c1[m].max())) if d == 1 else min(best, float(c1[m].min()))
    for k in ("pend1h", "pend", "rn", "run_ok", "h2_frozen", "t_touch", "touch_px", "gap1"):
        s.pop(k, None)
    s.update(stage=1, t_break=pd.Timestamp(tau), h2=best, rearms=n)
    if any(x.get("id") == s["id"] for x in st.get("setups", [])):
        return None
    st.setdefault("setups", []).append(s)
    logger.info("[PKG-REARM] %s: dir=%+d R2=%.5g -- re-armed after \"%s\" (%d of %d): waiting for a fresh retest, then "
                "a 1H close past %.5g", rec["asset"], d, rec["r2"], why, n, int(kn.get("rearm_max", 5)), best)
    return "re-armed (%d of %d): waiting for a fresh retest, then a 1H close past %.5g" % (n, int(kn.get("rearm_max", 5)), best)


def _b14_target(mp, b, d, q, stop, tgt, kn):
    """B14 6.1-6.3 at the package's entry: the normal target, or the near edge of a channel's other side when that
    comes first. -> (target, kind, wait reason or None)"""
    if mp is None or tgt is None or not kn.get("channel_target", True):
        return tgt, "normal", None
    lvl_c, ch = mp.channel_target(b, d, q)
    if lvl_c is None or d * (lvl_c - tgt) >= 0:
        return tgt, "normal", None
    room = d * (lvl_c - q) / max(abs(q - stop), 1e-12)
    if room < float(kn.get("room_r", 0.5)):
        return tgt, "normal", "the %s's other side (%.5g) is only %.2fR away" % (ch["kind"], lvl_c, max(room, 0.0))
    return lvl_c, "the %s's other side (near edge of its sphere)" % ch["kind"], None


def step(asset, st, df1, pcfg, now=None, frames=None, mp=None):
    """Judge every watching package record of this market. Returns (proofs to trade now, events for Telegram).
    `frames` (tests only) = {"M30":..., "D1":..., "W1":...}; live, they are read from MT5. `mp` (tests only) = the line
    map; live, the newest map from line_map."""
    st = st if st is not None else {}
    now = pd.Timestamp(now) if now is not None else _now_utc()
    ready = ready_proofs(st, now, asset)            # B14 item 1.1a: entries waiting for the trading look
    recs = [r for r in st.get("pkg", []) if r.get("status") == "watching"]
    if not recs:
        return ready, []
    cfg = market_settings(asset, pcfg)
    kn = (cfg or {}).get("b14_knobs") or {}
    if mp is None and any(r.get("b14") for r in recs):
        try:
            from src.execution import line_map as _LM
            mp = _LM.latest(asset)
        except Exception:
            mp = None
    get = (lambda k, n: frames.get(k)) if frames is not None else (lambda k, n: _mt5_frame(asset, k, n, now))
    m30 = get("M30", 700)
    if m30 is None or len(m30) < 60:
        logger.warning("[PKG] %s: no 30m candles from MT5 -- %d signal(s) wait for the next cycle", asset, len(recs))
        return ready, []
    t30, o30, h30, l30, c30 = (m30.index + pd.Timedelta(minutes=30)), m30["open"].values, m30["high"].values, \
        m30["low"].values, m30["close"].values
    e30 = pd.Series(c30).ewm(span=20, adjust=False).mean().values
    t1 = close_times(df1, 1)
    c1 = df1["close"].astype(float).values
    a1 = atr14(df1["high"].astype(float).values, df1["low"].astype(float).values, c1)
    df4 = hourly_to_4h(df1)
    t4, c4 = close_times(df4, 4), df4["close"].astype(float).values
    proofs, events = [], []
    for rec in recs:
        d, r2, r1, trig = rec["d"], rec["r2"], rec["r1"], rec["trig"]
        start = pd.Timestamp(rec["tE"])
        b14 = bool(rec.get("b14"))
        if not rec.get("announced"):
            rec["announced"] = True
            _emit(events, {"type": "handover", "asset": asset, "dir": d, "text":
                           "<b>PACKAGE TAKES OVER -- %s %s%s</b>\nline %.5g broken; %s. Watching the 30m/1H closes (up to %dh)." % (
                               asset, "BUY" if d == 1 else "SELL", (" (%s)" % rec["label"]) if rec.get("label") else "", r2,
                               ("; ".join(rec.get("why") or []) if b14 else
                                ("bigger 4H line %.5g just ahead (%.2f 4H moves)" % (rec["big"], rec["big_dist"]))
                                if rec["route"] == "near" else "signal too far from the line to take at once"), WAIT_H)})
        if rec.get("maj") is None:
            if b14 and not kn.get("pkg_majority", True):
                rec["maj"] = -1                               # box test only: the bigger-timeframe check switched off
                logger.info("[PKG-EVIDENCE] %s: dir=%+d bigger timeframes not checked (box-test switch)", asset, d)
            else:
                w1, d1 = get("W1", 400), get("D1", 400)
                n = (_bigger_read(w1, _SPAN["W1"], start, d) == 1 if w1 is not None else False) + \
                    (_bigger_read(d1, _SPAN["D1"], start, d) == 1 if d1 is not None else False) + _strong_4h(t4, c4, start, d)
                rec["maj"] = int(n)
                logger.info("[PKG-EVIDENCE] %s: dir=%+d bigger timeframes WITH the trade: %d of 3 (weekly/daily/4H)", asset, d, n)
                if n < 2:
                    _why = "no bigger-timeframe majority (%d of 3)" % n
                    _note = _rearm(st, rec, _why, now, df1, kn, events) if b14 else None
                    _finish(rec, "cancelled", _why, events, now, note=_note)
                    continue
        level = trig
        if rec.get("big") is not None and rec.get("big_dist") is not None and rec["big_dist"] <= NEAR:
            level = max(trig, rec["big"]) if d == 1 else min(trig, rec["big"])     # the push must clear it too
        k_from = int(np.searchsorted(t30.values, np.datetime64(start))) if rec.get("last_t") is None else \
            int(np.searchsorted(t30.values, np.datetime64(pd.Timestamp(rec["last_t"])), side="right"))
        decided = False
        for k in range(max(3, k_from), len(c30)):
            tau = t30[k]
            if tau > now:
                break
            rec["last_t"] = str(tau)
            if tau - start > pd.Timedelta(hours=WAIT_H):
                _why = "still dithering after %d hours" % WAIT_H
                _note = _rearm(st, rec, _why, tau, df1, kn, events) if b14 else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
                decided = True
                break
            j1 = int(np.searchsorted(t1.values, np.datetime64(tau), side="right")) - 1
            if j1 < 1:
                continue
            last1h, t_last1h = c1[j1], t1[j1]
            k4 = int(np.searchsorted(t4.values, np.datetime64(tau), side="right")) - 1
            if k4 >= 0 and t4[k4] > start and d * (c4[k4] - r1) < 0:
                _finish(rec, "cancelled", "setup died: a 4H close past its origin %.5g" % r1, events, tau)
                decided = True
                break
            p, q = c30[k - 1], c30[k]
            if (t_last1h > start and d * (last1h - r2) < 0) or (d * (p - r2) < 0 and d * (q - r2) < 0):
                _why = "weak: closed back past the line %.5g" % r2
                _note = _rearm(st, rec, _why, tau, df1, kn, events) if b14 else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
                decided = True
                break
            lvl_k, wall_txt = level, None
            if b14 and mp is not None:
                # B14 5.1 option (c): the push must also beat the first big wall ahead of the signal (a close past the
                # wall's three-quarter mark), read from the current map (diagonals move); option (a), box test only:
                # no big wall within half a move ahead of the entry at all
                bq = mp.bucket(tau)
                if rec.get("walls") and kn.get("release", "c") == "c":
                    from src.execution import line_map as _LM
                    _w = mp.walls_ahead(bq, d, rec["eE"], _LM.WALL_REACH)
                    if _w:
                        lvl_k = max(lvl_k, _w[0]["mark"]) if d == 1 else min(lvl_k, _w[0]["mark"])
                        wall_txt = "past the %s at %.5g" % (_w[0]["label"], _w[0]["near"])
            how = None
            if d * (last1h - level) > 0 and d * (p - lvl_k) > 0 and d * (q - lvl_k) > 0 and d * (q - p) >= 0:
                how = "strong push" + ((" " + wall_txt) if wall_txt else "")
            else:
                st4 = tau.floor("4h")
                if st4 == tau:
                    st4 = tau - pd.Timedelta(hours=4)
                klo = int(np.searchsorted(m30.index.values, np.datetime64(st4)))
                if k - klo >= 1:
                    o4, hh4, ll4 = o30[klo], h30[klo:k + 1].max(), l30[klo:k + 1].min()
                    rng = max(hh4 - ll4, 1e-12)
                    s4 = (q > o4 and (q - ll4) / rng >= 0.67) if d == 1 else (q < o4 and (hh4 - q) / rng >= 0.67)
                    ls = l30[k - 2:k + 1] if d == 1 else h30[k - 2:k + 1]
                    stair = (ls[0] < ls[1] < ls[2]) if d == 1 else (ls[0] > ls[1] > ls[2])
                    above = d * (q - e30[k]) > 0 and d * (e30[k] - e30[k - 2]) > 0
                    if RE_READ and s4 and stair and above and d * (last1h - trig) > 0 and d * (q - lvl_k) > 0:
                        how = "re-read past the wall (4H candle, 30m staircase, 20 EMA)"     # middle way
            if how is None:
                continue
            if b14 and mp is not None:
                bq = mp.bucket(tau)
                if rec.get("walls") and kn.get("release", "c") == "a":
                    from src.execution import line_map as _LM
                    if mp.walls_ahead(bq, d, q, _LM.WALL_REACH):
                        continue                              # option (a), box test: wait for an open road
                elif rec.get("walls") and _in_wall(mp, bq, d, rec["eE"], q):
                    continue                                  # never in the middle of a wall (re-checked at entry)
            atr = float(a1[j1])
            if not (atr == atr and atr > 0):
                continue
            klo2 = int(np.searchsorted(m30.index.values, np.datetime64(pd.Timestamp(rec["t_touch"]) - pd.Timedelta(hours=1))))
            seg = l30[klo2:k + 1] if d == 1 else h30[klo2:k + 1]
            stop = (seg.min() if d == 1 else seg.max()) - d * 0.3 * atr
            if d * (q - r2) / atr > FAR:
                piv = [seg[i] for i in range(1, len(seg) - 1) if ((seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]) if d == 1
                                                                   else (seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]))]
                if piv:
                    stop = piv[-1] - d * 0.3 * atr
                if d * (q - stop) < FLOOR * atr:
                    stop = q - d * FLOOR * atr
                if d * (q - stop) > CAP * atr:
                    _why = "far entry would need a stop wider than %g moves" % CAP
                    _note = _rearm(st, rec, _why, tau, df1, kn, events) if b14 else None
                    _finish(rec, "cancelled", _why, events, tau, note=_note)
                    decided = True
                    break
            if d * (q - stop) < FLOOR * atr:
                stop = q - d * FLOOR * atr                 # breathing space: never closer than 1 move
            if d * (q - stop) < cfg["min_sl_pct"] * q:
                stop = q - d * cfg["min_sl_pct"] * q
            if d * (q - stop) > STOP_CAP_ATR * atr:
                stop = q - d * STOP_CAP_ATR * atr
            if not gate_rr_ok(q, stop, atr, cfg["min_rr"]):
                _why = "reward:risk below %.2f at the entry" % cfg["min_rr"]
                _note = _rearm(st, rec, _why, tau, df1, kn, events) if b14 else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
                decided = True
                break
            tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr
            extra, tkind = None, "normal"
            if b14:
                tgt, tkind, _room_wait = _b14_target(mp, mp.bucket(tau) if mp is not None else -1, d, q, stop, tgt, kn)
                if _room_wait:
                    continue                                  # 6.2: the channel's other side is too close -- wait
                from src.execution import line_map as _LM
                _wb = mp.walls_between(mp.bucket(tau), d, q, tgt) if (mp is not None and tgt is not None) else []
                extra = {"ns_b14": True, "ns_b14_exits": True, "ns_target_kind": tkind, "ns_label": rec.get("label"),
                         "ns_walls": [{"near": B["near"], "s_near": B["s_near"], "s_far": B["s_far"], "label": B["label"]}
                                      for B in _wb[:4]],
                         "ns_lock_at": _LM.lock_level(_wb, d, q), "ns_route": "package: " + "; ".join(rec.get("why") or [])}
            wait = (tau - start).total_seconds() / 3600.0
            _pf = _proof(rec, q, tau, stop, tgt, atr, how, wait, extra)
            proofs.append(_pf)
            _finish(rec, "entered", "%s at %.5g after %.1fh; stop %.5g (%.1f moves), target %s%s" % (
                how, q, wait, stop, abs(q - stop) / atr, ("%.5g" % tgt) if tgt is not None else "runner",
                (" (%s)" % tkind) if tkind != "normal" else ""), events, tau, st=st, proof=_pf)
            decided = True
            break
        if not decided:
            logger.info("[PKG-WATCH] %s: dir=%+d R2=%.5g -- dithering, %.1fh of %dh used", asset, d, r2,
                        (now - start).total_seconds() / 3600.0, WAIT_H)
    done = [r for r in st["pkg"] if r.get("status") != "watching"]
    st["pkg"] = [r for r in st["pkg"] if r.get("status") == "watching"] + done[-KEEP_DONE:]
    return ready + proofs, events


def _in_wall(mp, b, d, e_sig, q):
    """B14 5.1 option (c): is q inside a big wall (past the edge of its sphere facing the trade but not yet past its
    three-quarter mark)? Walls are read ahead of the signal's entry."""
    for B in mp.bands(b, d, e_sig, 4.0):
        if B["big"] and d * (q - B["s_near"]) >= 0 and d * (q - B["mark"]) < 0:
            return True
    return False
