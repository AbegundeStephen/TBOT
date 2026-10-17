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
  4. Stop -- B14.1 (Desire 11 Oct, "the stop has to be close", option ii): 0.3 of a 1H move behind the LAST 1H SWING
     before the entry (the last dip for a buy, the last bounce for a sell, by wick), never closer than 1 move
     (breathing space). If that stop would be wider than STOP_MAX moves the trade is skipped and said so: at the
     smallest lot the account cannot carry it (8 Oct SILVER: 4.5 moves = $88 on a $122 account). Before B14.1 the
     stop sat behind the push's whole extreme since the retest (up to 5 moves; far entries 3).
     Target: the market's normal target. The engine's reward:risk gate applies.
Read-only towards MT5 (candles only). Every decision is logged ([PKG-...]) and returned as an event for Telegram.

B14 (Desire 8-9 Oct):
  1.1 An entry is kept on a board (st["pkg_ready"]) until the trading look uses it -- once, and only until the next
      30-minute close -- so an entry made during the first (recording) look is no longer lost. Each decision goes to
      Telegram the moment it is made (add_sink). main.py states what the trading look did with every entry
      ([PKG-OUTCOME]); an entry no look reached is said loudly when it runs out.
  The new rules (records marked "b14n", Desire 15-16 Oct): the entry close must beat the signal's peak by 0.15 of a 1H
  move (else the record is cancelled), the target is the map's first line 2-3R ahead (else out to 3.5R, else the
  normal target), and after a cancel the setup is re-armed (9 Oct option A, "retest"; not after a "bigger timeframes
  disagree" refusal -- Desire 16 Oct, option B): it goes back to the engine,
  waiting for a fresh retest and then a 1H close past the best close so far -- until a 4H close past its kill line,
  5 re-arms, or the 7-day waits.
  The springboard (Desire 16 Oct, live behind the B14 switch): a B14 record whose entry has a line (the target's line set)
  less than B14.SPRING_DIST_R (1R) ahead does not enter; it waits (up to B14.SPRING_WAIT_H, 48 h) for a 1H close past the
  line's band, a 1H candle back into the band that closes on the far side, then a 1H close past that candle's high (a
  sell: its low), and enters there -- stop behind the last 1H swing from that retest, target on the lines from the new
  entry. A close back through the wall after its break, a close back past the signal's own line, or the time limit
  cancels it (re-armed, option B); a 4H close past the origin kills it. _spring_start / _spring_step.
"""
import copy
import html
import logging

import numpy as np
import pandas as pd

from src.execution import b14_rules as B14     # B14 (Desire 15 Oct): the clearance and the line target at the confirmation
from src.execution.ns_engine import (PKG_FLOOR_ATR, PKG_STOP_MAX, atr14, close_times, gate_rr_ok, hourly_to_4h,
                                     market_settings, package_stop)

RE_READ = False                # B13 item 29 (Desire 5 Oct): version A -- the re-read shortcut is off (True = middle way)
logger = logging.getLogger(__name__)

WAIT_H = 24          # Desire 2 Oct: 24 hours
NEAR = 0.5           # the engine's PKG_NEAR: a bigger 4H line this close ahead must be cleared too
FAR = 2.5            # an entry further than this from the line (1H moves) is a far entry
FLOOR = PKG_FLOOR_ATR      # breathing space: a stop is never closer than 1 move (one value, in ns_engine)
STOP_MAX = PKG_STOP_MAX    # B14.1: a package stop wider than this (1H moves) = no trade (one value, in ns_engine)
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


def _rearm(st, rec, why, tau, df1, events):
    """Re-arm (Desire 9 Oct option A "retest", kept 16 Oct): after a cancel the setup goes back to the engine at stage 1 --
    it needs a fresh retest, then a 1H close past the best close so far. It still ends on a 4H close past its kill
    line, after 5 re-arms, or after the 7-day waits. Returns the note for the Telegram, or None."""
    s0 = rec.get("setup")
    if not s0 or not B14.REARM:
        return None
    if why.startswith("setup died"):
        return None
    if why.startswith("no bigger-timeframe majority") and not B14.REARM_AFTER_MAJORITY:
        return None
    n = int(rec.get("rearm_n", 0) or 0) + 1
    if n > B14.REARM_MAX:
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
    for k in ("pend1h", "pend", "h2_frozen", "t_touch", "touch_px", "gap1"):
        s.pop(k, None)
    s.update(stage=1, t_break=pd.Timestamp(tau), h2=best, rearms=n)
    if any(x.get("id") == s["id"] for x in st.get("setups", [])):
        return None
    st.setdefault("setups", []).append(s)
    logger.info("[PKG-REARM] %s: dir=%+d R2=%.5g -- re-armed after \"%s\" (%d of %d): waiting for a fresh retest, then "
                "a 1H close past %.5g", rec["asset"], d, rec["r2"], why, n, B14.REARM_MAX, best)
    return "re-armed (%d of %d): waiting for a fresh retest, then a 1H close past %.5g" % (n, B14.REARM_MAX, best)


def _spring_start(rec, wall, q, risk, tau, events):
    """B14 (Desire 16 Oct): the entry has a wall just ahead -- the record keeps watching, now for the springboard."""
    d = rec["d"]
    lv, w = float(wall["lv"]), float(wall["w"])
    sp = {"lv": lv, "w": w, "name": str(wall.get("name", "?")), "t0": str(tau), "q0": float(q),
          "dist_r": round(d * (lv - float(q)) / risk, 2) if risk > 0 else 0.0, "state": 0, "ext": None, "t_touch": None,
          "last_t": str(tau)}
    rec["spring"] = sp
    logger.info("[PKG-SPRING] %s: dir=%+d R2=%.5g -- waits at the wall %.5g (%s), %.2fR ahead of the entry %.5g: it enters "
                "only after a 1H close past it, a retest that holds and a close-through (up to %gh)", rec["asset"], d,
                rec["r2"], lv, sp["name"], sp["dist_r"], q, B14.SPRING_WAIT_H)
    _emit(events, {"type": "spring", "asset": rec["asset"], "dir": d, "text":
                   "<b>PACKAGE WAITS AT A WALL -- %s %s</b>\nthe entry %.5g has a line %.5g (%s) %.2fR ahead. It enters only "
                   "once price closes past that line, comes back to it and holds, then closes on through (up to %gh)." % (
                       rec["asset"], "BUY" if d == 1 else "SELL", q, lv, html.escape(sp["name"]), sp["dist_r"],
                       B14.SPRING_WAIT_H)})


def _spring_step(asset, st, rec, df1, t1, c1, h1, l1, a1, t4, c4, now, events, cfg, mp, wit14n, proofs):
    """B14 (Desire 16 Oct): the springboard, on the 1H closes since the last look -- (1) a close past the wall's band (the
    break), (2) a candle back into the band that closes on the far side of the line (the retest holds), (3) a close past
    that candle's high (a sell: its low) -> the entry at that close: stop behind the last 1H swing from the retest (the
    package's rule, too wide = cancelled), the reward:risk gate, target on the lines from the new entry. A close back
    through the wall after its break, a close back past the signal's own line, or SPRING_WAIT_H hours -> cancelled and
    re-armed (option B); a 4H close past the origin -> the setup died."""
    sp, d, r1, r2 = rec["spring"], rec["d"], rec["r1"], rec["r2"]
    start, t0 = pd.Timestamp(rec["tE"]), pd.Timestamp(sp["t0"])
    lv, w = float(sp["lv"]), float(sp["w"])
    far, near = lv + d * w, lv - d * w

    def _cancel(why, tj, rearm=True):
        _note = _rearm(st, rec, why, tj, df1, events) if rearm else None
        _finish(rec, "cancelled", why, events, tj, note=_note)

    j0 = int(np.searchsorted(t1.values, np.datetime64(pd.Timestamp(sp.get("last_t") or sp["t0"])), side="right"))
    for j in range(j0, len(c1)):
        tj = t1[j]
        if tj > now:
            break
        sp["last_t"] = str(tj)
        if tj - t0 > pd.Timedelta(hours=float(B14.SPRING_WAIT_H)):
            _cancel("B14 springboard: the wall %.5g was not broken, retested and closed through within %gh" % (
                lv, B14.SPRING_WAIT_H), tj)
            return
        k4 = int(np.searchsorted(t4.values, np.datetime64(tj), side="right")) - 1
        if k4 >= 0 and t4[k4] > start and d * (c4[k4] - r1) < 0:
            _cancel("setup died: a 4H close past its origin %.5g" % r1, tj, rearm=False)
            return
        c, hi, lo = float(c1[j]), float(h1[j]), float(l1[j])
        if d * (c - r2) < 0:
            _cancel("weak: closed back past the line %.5g (while waiting at the wall %.5g)" % (r2, lv), tj)
            return
        if sp["state"] >= 1 and d * (c - near) < 0:
            _cancel("B14 springboard: the break of the wall %.5g failed (a 1H close back through it)" % lv, tj)
            return
        if sp["state"] == 0:
            if d * (c - far) > 0:
                sp["state"] = 1
                logger.info("[PKG-SPRING] %s: dir=%+d -- the wall %.5g broken (1H close %.5g at %s); waiting for the retest",
                            asset, d, lv, c, tj)
            continue
        back = (lo <= far) if d == 1 else (hi >= far)
        if back and d * (c - lv) > 0:
            sp.update(state=2, ext=(hi if d == 1 else lo), t_touch=str(tj))
            logger.info("[PKG-SPRING] %s: dir=%+d -- the wall %.5g retested and held at %s; the entry: a 1H close past %.5g",
                        asset, d, lv, tj, sp["ext"])
            continue
        if sp["state"] == 2 and d * (c - float(sp["ext"])) > 0:
            atr = float(a1[j])
            if not (atr == atr and atr > 0):
                continue
            stop, _sw, _jsw, wide = package_stop(d, c, t1, h1, l1, j, sp["t_touch"], atr, cfg["min_sl_pct"])
            if wide:
                _cancel(wide, tj)
                return
            if not gate_rr_ok(c, stop, atr, cfg["min_rr"]):
                _cancel("reward:risk below %.2f at the springboard entry" % cfg["min_rr"], tj)
                return
            tgt0 = None if cfg["target_atr"] is None else c + d * float(cfg["target_atr"]) * atr
            _pk = rec["fields"].get("brc_h2")
            _pk = None if _pk in (None, "None") else float(_pk)
            _a4 = float(rec["fields"].get("ns_b_atr4") or 0) or atr * 4
            plan = _b14n_plan(asset, mp, d, c, stop, tgt0, atr, _a4, tj, _pk if _pk is not None else c, wit14n, cfg)
            extra = dict(plan["fields"], ns_exit="FIXED", ns_b14n_spring="after the springboard at %.6g (%s), %.2fR ahead of "
                         "the first confirmation %.6g" % (lv, sp["name"], sp["dist_r"], sp["q0"]))
            wait = (tj - start).total_seconds() / 3600.0
            pf = _proof(rec, c, tj, stop, plan["target"], atr, "springboard", wait, extra)
            proofs.append(pf)
            _finish(rec, "entered", "springboard at %.5g after %.1fh (the wall %.5g broken, retested, closed through); stop "
                    "%.5g (%.1f moves), target %.5g (%s)" % (c, wait, lv, stop, abs(c - stop) / atr, plan["target"],
                                                            plan["kind"]), events, tj, st=st, proof=pf)
            return
    logger.info("[PKG-SPRING] %s: dir=%+d R2=%.5g -- waiting at the wall %.5g (%s), %.1fh of %gh", asset, d, r2, lv,
                ("not broken yet", "broken, waiting for the retest", "retested, waiting for the close-through")[int(sp["state"])],
                (now - t0).total_seconds() / 3600.0, B14.SPRING_WAIT_H)


def _b14n_plan(asset, mp, d, q, stop, today, atr, a4, tau, peak, wit14n, cfg):
    """B14 at the package's entry: the target on the map's lines and the record facts -- the same rule as the hourly
    entry (ns_engine._b14n_plan), with the package's own entry price, stop and time. -> {"target", "kind", "fields"}"""
    risk = abs(float(q) - float(stop))
    if today is None:                                         # a runner market (BTC): no runner under B14
        fb, fb_text = float(q) + d * B14.BTC_FALLBACK_ATR1 * float(atr), "%g 1H moves (no runner under B14)" % B14.BTC_FALLBACK_ATR1
    else:
        fb, fb_text = float(today), "today's target"
    if mp is None:
        logger.warning("[B14] %s: no line map at the package's entry -- %s is used (see the [MAP] lines)", asset, fb_text)
        lines = []
    else:
        lines = B14.all_lines(mp, tau, a4)
    tgt, L, kind, text = B14.line_target(d, q, risk, lines, fb, fb_text)
    wit = B14.witness_at(wit14n or [], d, tau)
    f = B14.fields(wit, kind, text, B14.clearance(d, q, peak, atr), B14.lines_ahead(d, q, risk, lines), len(lines))
    f.update(B14.vtm_fields(kind, cfg["target_atr"]))        # what the trade manager reads (see ns_engine._b14n_plan)
    return {"target": float(tgt), "kind": "B14 target %s: %s" % (kind, text), "fields": f}


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
    if mp is None and any(r.get("b14n") for r in recs):
        # B14: the line target and the MA witness need the newest line map (the box test passes its own; before
        # 16 Oct only old-B14 records fetched it, so live a B14 package entry would have had no map)
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
    h1, l1 = df1["high"].astype(float).values, df1["low"].astype(float).values     # B14.1: for the last 1H swing
    a1 = atr14(h1, l1, c1)
    df4 = hourly_to_4h(df1)
    t4, c4 = close_times(df4, 4), df4["close"].astype(float).values
    wit14n = None
    if any(r.get("b14n") for r in recs):
        # B14: the witness list (2+-touch diagonals on these 4H candles, the map's six MAs) for the record's witness fact,
        # judged at the confirmation time exactly as the research did; a failure is said and the fact is left empty
        try:
            _h4, _l4 = df4["high"].astype(float).values, df4["low"].astype(float).values
            wit14n = B14.witness_breaks(mp, t4, atr14(_h4, _l4, c4), c4)
        except Exception as _we:
            logger.warning("[B14] %s: package witness list not built (%s)", asset, _we)
    proofs, events = [], []
    for rec in recs:
        d, r2, r1, trig = rec["d"], rec["r2"], rec["r1"], rec["trig"]
        start = pd.Timestamp(rec["tE"])
        b14n = bool(rec.get("b14n"))                     # B14: handed over with the B14 rules on
        if not rec.get("announced"):
            rec["announced"] = True
            _emit(events, {"type": "handover", "asset": asset, "dir": d, "text":
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
                _why = "no bigger-timeframe majority (%d of 3)" % n
                _note = _rearm(st, rec, _why, now, df1, events) if b14n else None
                _finish(rec, "cancelled", _why, events, now, note=_note)
                continue
        if rec.get("spring"):                            # B14 (Desire 16 Oct): waiting at a wall for the springboard
            _spring_step(asset, st, rec, df1, t1, c1, h1, l1, a1, t4, c4, now, events, cfg, mp, wit14n, proofs)
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
                _note = _rearm(st, rec, _why, tau, df1, events) if b14n else None
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
                _note = _rearm(st, rec, _why, tau, df1, events) if b14n else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
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
                    if RE_READ and s4 and stair and above and d * (last1h - trig) > 0 and d * (q - level) > 0:
                        how = "re-read past the wall (4H candle, 30m staircase, 20 EMA)"     # middle way
            if how is None:
                continue
            atr = float(a1[j1])
            if not (atr == atr and atr > 0):
                continue
            # B14.1 (Desire 11 Oct, option ii): the stop sits 0.3 of a move behind the LAST 1H SWING before the entry --
            # the last dip for a buy (a 1H low below both neighbours' lows, confirmed by the candle after it), the last
            # bounce for a sell -- looked for from the retest candle to the last closed 1H candle, on the far side of
            # the entry. With no such swing: the push's own 1H extreme since the retest. Wider than STOP_MAX moves =
            # no trade (said as a cancel). The far-entry rule (last 30m pivot, 3 moves) is replaced by this one rule.
            stop, swing, j_sw, _wide = package_stop(d, q, t1, h1, l1, j1, rec["t_touch"], atr, cfg["min_sl_pct"])
            if _wide:
                _why = _wide
                _note = _rearm(st, rec, _why, tau, df1, events) if b14n else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
                decided = True
                break
            if not gate_rr_ok(q, stop, atr, cfg["min_rr"]):
                _why = "reward:risk below %.2f at the entry" % cfg["min_rr"]
                _note = _rearm(st, rec, _why, tau, df1, events) if b14n else None
                _finish(rec, "cancelled", _why, events, tau, note=_note)
                decided = True
                break
            tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr
            extra, tkind = None, "normal"
            if b14n:
                # B14 (Desire 15 Oct): the package's entry obeys the same two rules as an hourly entry -- the close must
                # beat the signal's peak by CLEAR_ATR1 of a 1H move (else the record is cancelled, as the research
                # dropped such trades), and the target is the map's first line 2-3R ahead (else out to 3.5R, else
                # today's; BTC: else 4 1H moves, FIXED exit -- no runner under B14)
                _pk = rec["fields"].get("brc_h2")
                _pk = None if _pk in (None, "None") else float(_pk)
                if _pk is not None and not B14.clearance_ok(d, q, _pk, atr):
                    _why = "B14: the push closed only %.2f of a 1H move past the signal's peak %.5g (%.2f needed)" % (
                        B14.clearance(d, q, _pk, atr), _pk, B14.CLEAR_ATR1)
                    _note = _rearm(st, rec, _why, tau, df1, events)
                    _finish(rec, "cancelled", _why, events, tau, note=_note)
                    decided = True
                    break
                _a4 = float(rec["fields"].get("ns_b_atr4") or 0) or float(atr) * 4
                _plan = _b14n_plan(asset, mp, d, q, stop, tgt, atr, _a4, tau, _pk if _pk is not None else q, wit14n, cfg)
                tgt, tkind = _plan["target"], _plan["kind"]
                extra = dict(extra or {}, ns_exit="FIXED", **_plan["fields"])
                if B14.FAR_MAX is not None and d * (q - r2) > float(B14.FAR_MAX) * atr:
                    # test rows only (FAR_MAX is None live): the BTC 15 Oct short -- confirmed far past its line
                    _why = "B14: the push closed %.1f 1H moves past its line %.5g (limit %g)" % (d * (q - r2) / atr, r2, B14.FAR_MAX)
                    _note = _rearm(st, rec, _why, tau, df1, events)
                    _finish(rec, "cancelled", _why, events, tau, note=_note)
                    decided = True
                    break
                if B14.SPRING:
                    # B14 (Desire 16 Oct): a wall less than SPRING_DIST_R ahead -> no entry yet; wait for the springboard
                    _risk = abs(q - stop)
                    _wall = B14.wall_ahead(d, q, _risk, B14.all_lines(mp, tau, _a4) if mp is not None else [], B14.SPRING_DIST_R)
                    if _wall is not None:
                        _spring_start(rec, _wall, q, _risk, tau, events)
                        decided = True
                        break
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
