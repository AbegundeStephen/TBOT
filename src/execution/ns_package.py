"""B13 (Desire 2 Oct 2026): THE PACKAGE -- judges the E signals the engine hands over.

The engine (ns_engine.py) hands over an E signal (4H break, 1H retest, 1H close past the peak) when it broke a small
line with a bigger 4H line just ahead (within half a 4H move), or when it is too far from the line to take at once.
Everything else is traded by the engine as before.

The package decides with the evidence -- the middle way, as ruled on 2 Oct (B's re-read, but only past the wall):
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
"""
import logging

import numpy as np
import pandas as pd

from src.execution.ns_engine import (STOP_CAP_ATR, atr14, close_times, gate_rr_ok, hourly_to_4h,
                                     market_settings)

logger = logging.getLogger(__name__)

WAIT_H = 24          # Desire 2 Oct: 24 hours
NEAR = 0.5           # the engine's PKG_NEAR: a bigger 4H line this close ahead must be cleared too
FAR = 2.5            # an entry further than this from the line (1H moves) is a far entry
FLOOR = 1.0          # breathing space: a stop is never closer than 1 move
CAP = 3.0            # a far entry needing a wider stop than this is not taken
KEEP_DONE = 20       # finished package records kept for display
SYMBOLS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}
_SPAN = {"M30": pd.Timedelta(minutes=30), "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(days=7)}


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


def _proof(rec, q, tau, stop, tgt, atr, how, wait_h):
    f = dict(rec["fields"])
    for key in ("ns_aplus", "ns_watch"):
        f[key] = list(f.get(key) or [])
    f.update(ns_entry="PKG", ns_close=float(q), ns_stop=float(stop), ns_target=(float(tgt) if tgt is not None else None),
             ns_atr1=float(atr), ns_candle=str(tau), brc_first_confirmed_ts=str(tau),
             pkg_route=rec["route"], pkg_how=how, pkg_wait_h=round(float(wait_h), 1), pkg_bigger=rec.get("big"),
             pkg_majority=rec.get("maj"), pkg_signal_t=rec["tE"], pkg_signal_close=rec["eE"])
    f["ns_aplus"].append("package: %s, after %.1fh" % (how, wait_h))
    return {"key": (rec["asset"], rec["kind"], rec["d"], round(float(rec["r2"]), 8)), "kind": rec["kind"],
            "dir": rec["d"], "ref": float(rec["r2"]), "tests": 0, "first_ts": str(tau), "fields": f}


def _finish(rec, status, why, events, tau):
    rec.update(status=status, why=why, done_t=str(tau))
    logger.info("[PKG-%s] %s: dir=%+d R2=%.5g (signal %s) -- %s", "ENTER" if status == "entered" else "CANCEL",
                rec["asset"], rec["d"], rec["r2"], rec["tE"], why)
    events.append({"type": status, "asset": rec["asset"], "dir": rec["d"],
                   "text": "<b>PACKAGE %s -- %s %s</b>\n%s" % ("ENTERED" if status == "entered" else "CANCELLED",
                                                              rec["asset"], "BUY" if rec["d"] == 1 else "SELL", why)})


def step(asset, st, df1, pcfg, now=None, frames=None):
    """Judge every watching package record of this market. Returns (proofs to trade now, events for Telegram).
    `frames` (tests only) = {"M30":..., "D1":..., "W1":...}; live, they are read from MT5."""
    recs = [r for r in (st or {}).get("pkg", []) if r.get("status") == "watching"]
    if not recs:
        return [], []
    cfg = market_settings(asset, pcfg)
    now = pd.Timestamp(now) if now is not None else _now_utc()
    get = (lambda k, n: frames.get(k)) if frames is not None else (lambda k, n: _mt5_frame(asset, k, n, now))
    m30 = get("M30", 700)
    if m30 is None or len(m30) < 60:
        logger.warning("[PKG] %s: no 30m candles from MT5 -- %d signal(s) wait for the next cycle", asset, len(recs))
        return [], []
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
        if not rec.get("announced"):
            rec["announced"] = True
            events.append({"type": "handover", "asset": asset, "dir": d, "text":
                           "<b>PACKAGE TAKES OVER -- %s %s</b>\nline %.5g broken; %s. Watching the 30m/1H closes (up to %dh)." % (
                               asset, "BUY" if d == 1 else "SELL", r2,
                               ("bigger 4H line %.5g just ahead (%.2f 4H moves)" % (rec["big"], rec["big_dist"]))
                               if rec["route"] == "near" else "signal too far from the line to take at once", WAIT_H)})
        if rec.get("maj") is None:
            w1, d1 = get("W1", 400), get("D1", 400)
            n = (_bigger_read(w1, _SPAN["W1"], start, d) == 1 if w1 is not None else False) + \
                (_bigger_read(d1, _SPAN["D1"], start, d) == 1 if d1 is not None else False) + _strong_4h(t4, c4, start, d)
            rec["maj"] = int(n)
            logger.info("[PKG-EVIDENCE] %s: dir=%+d bigger timeframes WITH the trade: %d of 3 (weekly/daily/4H)", asset, d, n)
            if n < 2:
                _finish(rec, "cancelled", "no bigger-timeframe majority (%d of 3)" % n, events, now)
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
                _finish(rec, "cancelled", "still dithering after %d hours" % WAIT_H, events, tau)
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
                _finish(rec, "cancelled", "weak: closed back past the line %.5g" % r2, events, tau)
                decided = True
                break
            how = None
            if d * (last1h - level) > 0 and d * (p - level) > 0 and d * (q - level) > 0 and d * (q - p) >= 0:
                how = "strong push"
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
                    if s4 and stair and above and d * (last1h - trig) > 0 and d * (q - level) > 0:
                        how = "re-read past the wall (4H candle, 30m staircase, 20 EMA)"     # middle way
            if how is None:
                continue
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
                    _finish(rec, "cancelled", "far entry would need a stop wider than %g moves" % CAP, events, tau)
                    decided = True
                    break
            if d * (q - stop) < FLOOR * atr:
                stop = q - d * FLOOR * atr                 # breathing space: never closer than 1 move
            if d * (q - stop) < cfg["min_sl_pct"] * q:
                stop = q - d * cfg["min_sl_pct"] * q
            if d * (q - stop) > STOP_CAP_ATR * atr:
                stop = q - d * STOP_CAP_ATR * atr
            if not gate_rr_ok(q, stop, atr, cfg["min_rr"]):
                _finish(rec, "cancelled", "reward:risk below %.2f at the entry" % cfg["min_rr"], events, tau)
                decided = True
                break
            tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr
            wait = (tau - start).total_seconds() / 3600.0
            proofs.append(_proof(rec, q, tau, stop, tgt, atr, how, wait))
            _finish(rec, "entered", "%s at %.5g after %.1fh; stop %.5g (%.1f moves), target %s" % (
                how, q, wait, stop, abs(q - stop) / atr, ("%.5g" % tgt) if tgt is not None else "runner"), events, tau)
            decided = True
            break
        if not decided:
            logger.info("[PKG-WATCH] %s: dir=%+d R2=%.5g -- dithering, %.1fh of %dh used", asset, d, r2,
                        (now - start).total_seconds() / 3600.0, WAIT_H)
    done = [r for r in st["pkg"] if r.get("status") != "watching"]
    st["pkg"] = [r for r in st["pkg"] if r.get("status") == "watching"] + done[-KEEP_DONE:]
    return proofs, events
