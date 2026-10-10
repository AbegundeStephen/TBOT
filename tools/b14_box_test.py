"""B14 BOX TEST (Desire 9 Oct, section 9.1) -- READ-ONLY. Runs the B14 code itself (ns_engine, ns_package, line_map:
the very files the bot runs) over the saved price files, every market, about 16 months, once per test row, and prints
one table. It changes nothing: it reads data\\raw\\*_1h.csv, config\\config.json, the logs (spreads) and MT5's own
30-minute / daily / weekly candles (read-only), and writes only its report logs\\b14_box_test_<date>.txt.

How it runs each row (so the numbers can be checked):
  1. the line map is built once per market from the hourly file (+ MT5's daily/weekly candles);
  2. the engine replays every closed 1H candle in one pass (every candle counted as "live", as each one is live at the
     look right after it closes);
  3. every hand-over goes to the package, judged on MT5's 30-minute closes exactly as live (24 h at most);
     a re-arm puts the setup back and replays it from the cancel;
  4. trades are booked one position per market at a time and run on the 30-minute candles: stop, target, the 7-day
     exit, and -- under the B14 rules -- the wall lock and the staircase (line_map.lock_level / stair_step, the same
     functions the trade manager calls). Money = the smallest lot, minus the spread the bot really paid.
Rows: R0 = the B13 rules on the B14 code (the switch off) -- checked IDENTICAL against the B13 files in
tools\\b14_b13ref; R1 = B14 as approved; R2.. = R1 with one thing changed. Then: the guard check (no lock or staircase
move held as [NS-STOP-HOLD]), T4 re-run with spheres (what price does at each kind of line), and the replays.

usage (from C:\\TradingBot\\TBOT, with the bot's python):
  python tools\\b14_box_test.py                      all rows, all markets (an hour or more -- run it at the weekend)
  python tools\\b14_box_test.py --rows R0,R1 --markets GOLD,BTC     a quick look
  python tools\\b14_box_test.py --months 16 --replay-from 2026-09-22"""
import argparse
import collections
import copy
import glob
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import types

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from src.execution import line_map as LM          # noqa: E402
from src.execution import ns_engine as NE         # noqa: E402
from src.execution import ns_package as NP        # noqa: E402
for _n in ("src.execution.ns_engine", "src.execution.ns_engine.explore", "src.execution.ns_package",
           "src.execution.line_map", "src.execution.livermore_state_machine"):
    logging.getLogger(_n).setLevel(logging.ERROR)

MARKETS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}
FALLBACK_USD = {"BTC": 0.01, "GOLD": 1.0, "SILVER": 50.0, "USOIL": 10.0, "USTEC": 0.05, "JP225": 0.02,
                "EURUSD": 1000.0, "GBPAUD": 660.0, "EURJPY": 6.7, "AUDJPY": 6.7}
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


# ---- data ------------------------------------------------------------------------------------------------------------
def load_1h(sym):
    p = os.path.join("data", "raw", "%s_1h.csv" % sym)
    if not os.path.exists(p):
        return None, "no price file %s" % p
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamp", "time", "datetime", "date", "open_time") if c in df.columns), df.columns[0])
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        idx = pd.to_datetime(ts, unit="s" if float(ts.max()) < 1e11 else "ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True, errors="coerce")
    df.index = pd.DatetimeIndex(idx).tz_convert("UTC").tz_localize(None)
    if not all(c in df.columns for c in ("open", "high", "low", "close")):
        return None, "price file %s has no open/high/low/close columns" % p
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df, "%d hourly candles %s .. %s" % (len(df), str(df.index[0])[:16], str(df.index[-1])[:16])


def mt5_frames(sym):
    """30-minute, daily and weekly candles straight from MT5 (count-based: no local-time conversion)."""
    out = {}
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return out
        mt5.symbol_select(sym, True)
        for name, tf, n in (("M30", mt5.TIMEFRAME_M30, 30000), ("D1", mt5.TIMEFRAME_D1, 2500),
                            ("W1", mt5.TIMEFRAME_W1, 700)):
            r = mt5.copy_rates_from_pos(sym, tf, 0, n)
            if r is None or len(r) == 0:
                continue
            f = pd.DataFrame(r)
            f.index = pd.to_datetime(f["time"], unit="s")
            out[name] = f[["open", "high", "low", "close"]].astype(float)
    except Exception:
        pass
    return out


def m30_from_1h(df):
    """Only when MT5 gives no 30-minute candles: two halves per hour (APPROXIMATE -- the report says so)."""
    rows = []
    for t, (o, h, l, c) in zip(df.index, df[["open", "high", "low", "close"]].values):
        m = (o + c) / 2.0
        rows.append((t, o, max(o, m), min(o, m), m))
        rows.append((t + pd.Timedelta(minutes=30), m, max(m, h), min(m, l), c))
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).set_index("t")


def lot_values():
    """US dollars per 1.0 price move at the smallest lot (MT5; fallback table) and the live spreads."""
    out, spreads = {}, {}
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError(mt5.last_error())
        for a, sym in MARKETS.items():
            mt5.symbol_select(sym, True)
            si = mt5.symbol_info(sym)
            if si is None:
                continue
            tk0 = mt5.symbol_info_tick(sym)
            if tk0 is not None and tk0.bid > 0 and tk0.ask > tk0.bid:
                spreads[a] = (tk0.ask - tk0.bid) / ((tk0.ask + tk0.bid) / 2.0)
            units, ccy, rate = float(si.volume_min) * float(si.trade_contract_size), si.currency_profit, 1.0
            if ccy != "USD":
                rate = None
                for s2, inv in ((ccy + "USDm", False), ("USD" + ccy + "m", True)):
                    mt5.symbol_select(s2, True)
                    tk = mt5.symbol_info_tick(s2)
                    if tk is not None and tk.bid > 0:
                        rate = (1.0 / tk.bid) if inv else tk.bid
                        break
            if rate:
                out[a] = units * rate
    except Exception as e:
        say("   MT5 not reachable (%s) -- the fallback lot table is used" % e)
    for a in MARKETS:
        out.setdefault(a, FALLBACK_USD[a])
    return out, spreads


def log_spreads():
    """The spreads the bot really paid ([FRICTION] ... used=), median per market, from the newest logs."""
    fr = collections.defaultdict(list)
    rx = re.compile(r"\[FRICTION\] (\w+)\b.*?used=([0-9.]+)")
    for f in sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime)[-5:]:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "[FRICTION]" in line:
                        m = rx.search(line)
                        if m:
                            fr[m.group(1)].append(float(m.group(2)))
        except OSError:
            continue
    return {a: float(np.median(v)) for a, v in fr.items() if v}


# ---- the engine, replayed --------------------------------------------------------------------------------------------
NE.REPLAY_DAYS = 100000          # this process only: replay the whole file in one pass


class BoxEngine(NE.NSEngine):
    """The live engine; every candle counts as live (as it is at the look right after it closes)."""
    def _candle(self, st, cfg, i, t, emit, *a, **k):
        return NE.NSEngine._candle(self, st, cfg, i, t, True, *a, **k)


class ReplayEngine(BoxEngine):
    """A re-armed setup replayed from its cancel: no new setups are born in this run."""
    def _register(self, st, s, atr4_at, create, emit):
        return NE.NSEngine._register(self, st, s, atr4_at, False, False)


def load_b13_ref():
    """The B13 engine and package (commit 82d77e8) from tools\\b14_b13ref -- for the R0 identity check."""
    d = os.path.join("tools", "b14_b13ref")
    pe, pp = os.path.join(d, "ns_engine.py"), os.path.join(d, "ns_package.py")
    if not (os.path.exists(pe) and os.path.exists(pp)):
        return None, None
    spec = importlib.util.spec_from_file_location("b14ref_ns_engine", pe)
    me = importlib.util.module_from_spec(spec)
    sys.modules["b14ref_ns_engine"] = me
    spec.loader.exec_module(me)
    me.REPLAY_DAYS = 100000
    src = open(pp, encoding="utf-8").read().replace("from src.execution.ns_engine import", "from b14ref_ns_engine import")
    mp_ = types.ModuleType("b14ref_ns_package")
    exec(compile(src, pp, "exec"), mp_.__dict__)
    for _n in ("b14ref_ns_engine", "b14ref_ns_engine.explore"):
        logging.getLogger(_n).setLevel(logging.ERROR)
    mp_.logger.setLevel(logging.ERROR)

    class RefEngine(me.NSEngine):
        def _candle(self, st, cfg, i, t, emit, *a, **k):
            return me.NSEngine._candle(self, st, cfg, i, t, True, *a, **k)
    return RefEngine, mp_


def run_signals(asset, df1, frames, mp, pcfg, eng_cls=BoxEngine, pkg=NP, ref=False):
    """One market, one row: the engine pass, the package on every hand-over, re-arms replayed. -> (entries, log)"""
    end = df1.index[-1] + pd.Timedelta(hours=1)
    eng = eng_cls(asset)
    if ref:
        r = eng.update(None, df1, None, pcfg, None, now=end)
    else:
        r = eng.update(None, df1, None, pcfg, None, now=end, mp=mp)
    st = r["state"]
    entries, log = [], collections.Counter()
    for p in r["proofs"]:
        entries.append(dict(src="engine", proof=p))
        log["taken at once"] += 1
    todo = [rec for rec in (st.get("pkg") or []) if rec.get("status") == "watching"]
    seen_ids = set()
    while todo:
        rec = todo.pop(0)
        log["hand-overs"] += 1
        now = min(pd.Timestamp(rec["tE"]) + pd.Timedelta(hours=26), end)
        sub = {"pkg": [rec], "setups": [], "pkg_ready": []}
        if ref:
            prs, _ev = pkg.step(asset, sub, df1, pcfg, now=now, frames=frames)
        else:
            prs, _ev = pkg.step(asset, sub, df1, pcfg, now=now, frames=frames, mp=mp)
        for p in prs:
            entries.append(dict(src="package", proof=p))
        log["package %s" % rec.get("status")] += 1
        if rec.get("status") == "cancelled":
            _w = re.sub(r"[-+]?\d[\d.,]*(e[-+]?\d+)?", "", str(rec.get("why_done") or rec.get("why")))
            log["cancel: %s" % re.sub(r"\s+", " ", re.sub(r"[()]", "", _w).replace(" of ", " ")).strip()[:44]] += 1
        for s in sub.get("setups") or []:                          # re-armed: replay it from the cancel
            if (s["id"], s.get("rearms")) in seen_ids:
                continue
            seen_ids.add((s["id"], s.get("rearms")))
            log["re-arms"] += 1
            st2 = NE.NSEngine.new_state()
            st2["setups"] = [s]
            st2["next_id"] = int(s["id"]) + 1
            t_c = pd.Timestamp(s["t_break"]).floor("h")
            st2["last_1h"] = t_c
            st2["rules"] = NE.rules_stamp(NE.market_settings(asset, pcfg), None)
            e2 = ReplayEngine(asset)
            r2 = e2.update(st2, df1, None, pcfg, None, now=end, mp=mp)
            for p in r2["proofs"]:
                entries.append(dict(src="re-armed", proof=p))
                log["re-armed: taken at once"] += 1
            todo += [x for x in (r2["state"].get("pkg") or []) if x.get("status") == "watching"]
    return entries, log


# ---- the trades ------------------------------------------------------------------------------------------------------
def simulate(e, m30, mp, cost, usd, exits_on, h1=None):
    """One trade on the 30-minute candles: stop, target, 7 days; BTC's runner (today's exit) trails on each 1H close
    with the engine's own runner_step; with the B14 exits: the wall lock and the staircase.
    -> dict or None (no candles after the entry)"""
    f = e["proof"]["fields"]
    d = int(e["proof"]["dir"])
    q, stop, tgt = float(f["ns_close"]), float(f["ns_stop"]), f.get("ns_target")
    tgt = None if tgt in (None, "None") else float(tgt)
    t_in = pd.Timestamp(f["ns_candle"])
    k0 = int(np.searchsorted(m30.index.values, np.datetime64(t_in), side="left"))
    if k0 >= len(m30):
        return None
    risk0 = abs(q - stop)
    cur, locked, how, steps = stop, False, None, 0
    b14x = exits_on and bool(f.get("ns_b14_exits"))
    runner = f.get("ns_exit") == "RUNNER" and h1 is not None
    armed, peak = False, None
    if runner:
        t1c = h1.index + pd.Timedelta(hours=1)
        hi1a, lo1a, cl1a = h1["high"].values, h1["low"].values, h1["close"].values
    O, Hh, L, C = (m30[c].values for c in ("open", "high", "low", "close"))
    ti = m30.index
    k_end, px = None, None
    for k in range(k0, len(m30)):
        hi, lo = Hh[k], L[k]
        if (d == 1 and lo <= cur) or (d == -1 and hi >= cur):
            k_end, px = k, cur
            how = "stop" if not locked else ("lock / staircase stop" if d * (cur - q) >= 0 else "stop")
            break
        if tgt is not None and ((d == 1 and hi >= tgt) or (d == -1 and lo <= tgt)):
            k_end, px, how = k, tgt, "target"
            break
        new = None
        if b14x and mp is not None:
            b = mp.bucket(ti[k])
            walls = mp.walls_between(b, d, q, tgt) if tgt is not None else mp.walls_between(b, d, q, None, reach=4.0)
            if not locked:
                lk = LM.lock_level(walls, d, q)
                if lk is not None and ((d == 1 and hi >= lk) or (d == -1 and lo <= lk)):
                    locked = True
                    if d * (q - cur) > 0:
                        new = q
            if locked and (ti[k] + pd.Timedelta(minutes=30)).minute == 0:          # a 1H close
                st_ = LM.stair_step(walls, d, C[k], new if new is not None else cur)
                if st_ is not None:
                    new, steps = st_, steps + 1
        if runner and (ti[k] + pd.Timedelta(minutes=30)).minute == 0:                    # BTC's runner, as live
            j = int(np.searchsorted(t1c.values, np.datetime64(ti[k] + pd.Timedelta(minutes=30)), side="right")) - 1
            if j >= 0:
                a0 = max(0, j - 260)
                bars = int((ti[k] + pd.Timedelta(minutes=30) - t_in).total_seconds() // 3600)
                rn, armed, peak, _w = NE.runner_step(d, q, cur, risk0, hi1a[a0:j + 1], lo1a[a0:j + 1], cl1a[a0:j + 1],
                                                     bars, armed, peak if peak is not None else q)
                if rn is not None and (new is None or d * (rn - new) > 0):
                    new = rn
        if new is not None and d * (new - cur) > 0:
            cur = new
        if (ti[k] + pd.Timedelta(minutes=30) - t_in) >= pd.Timedelta(hours=NE.TRADE_BARS):
            k_end, px, how = k, C[k], "7 days"
            break
    if k_end is None:
        return None
    pts = d * (px - q) - cost * q
    return dict(t_in=t_in, t_out=ti[k_end] + pd.Timedelta(minutes=30), d=d, q=q, px=px, how=how, R=pts / risk0 if risk0 else 0.0,
                usd=pts * usd, locked=locked, steps=steps, src=e["src"], label=f.get("ns_label"),
                route=f.get("ns_route"), key=e["proof"]["key"])


def book(entries, m30, mp, cost, usd, exits_on, h1=None):
    """One position per market at a time, in time order (the live bot never adds to an open trade)."""
    rows, free = [], pd.Timestamp("1900-01-01")
    for e in sorted(entries, key=lambda x: pd.Timestamp(x["proof"]["fields"]["ns_candle"])):
        t_in = pd.Timestamp(e["proof"]["fields"]["ns_candle"])
        if t_in < free:
            continue
        r = simulate(e, m30, mp, cost, usd, exits_on, h1=h1)
        if r is None:
            continue
        rows.append(r)
        free = r["t_out"]
    return rows


def stats(rows, weeks):
    if not rows:
        return dict(n=0, win=0.0, R=0.0, usd=0.0, wk=0.0, dd=0.0, avgR=0.0)
    usd = np.array([r["usd"] for r in sorted(rows, key=lambda x: x["t_out"])])
    eq = np.cumsum(usd)
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]))
    R = np.array([r["R"] for r in rows])
    return dict(n=len(rows), win=100.0 * float((R > 0).mean()), R=float(R.sum()), usd=float(usd.sum()),
                wk=float(usd.sum()) / max(weeks, 1e-9), dd=dd, avgR=float(R.mean()))


# ---- the rows --------------------------------------------------------------------------------------------------------
def rows_table(pc0):
    """Each row = the config the bot reads, with one change. R0 = the switch off; R1 = B14 as approved."""
    def mk(on, knobs=None, btc=None, exits=True):
        pc = copy.deepcopy(pc0)
        pc["b14_rules_enabled"] = bool(on)
        pc["b14_knobs"] = dict(knobs or {})
        nm = pc.setdefault("ns_markets", {})
        if on and "BTC" in nm:
            nm["BTC"] = dict(nm["BTC"], **(btc or {"target_atr": 4.0, "exit": "FIXED"}))   # 6.1 (Desire 9 Oct)
        return dict(pc=pc, exits=exits)
    return collections.OrderedDict([
        ("R0", ("B13 rules on the B14 code (switch off)", mk(False))),
        ("R1", ("B14 as approved", mk(True))),
        ("R2", ("witness window 8 h", mk(True, {"witness_h": 8}))),
        ("R3", ("witness window 24 h", mk(True, {"witness_h": 24}))),
        ("R4", ("diagonal witnesses only (no averages)", mk(True, {"ma_witness": False}))),
        ("R5", ("runner entry with no dip", mk(True, {"runner_dip": False}))),
        ("R6", ("no runner entry", mk(True, {"runner": False}))),
        ("R7", ("no re-arm", mk(True, {"rearm": False}))),
        ("R8", ("re-arm, but not after a bigger-timeframe refusal", mk(True, {"rearm_after_majority": False}))),
        ("R9", ("the signal line breaks at the 3/4 mark too", mk(True, {"sig_break": "mark"}))),
        ("R10", ("no strong buys: every big wall goes to the package", mk(True, {"strong": False}))),
        ("R11", ("package option (a): wait for an open road", mk(True, {"release": "a"}))),
        ("R12", ("turning point within half an hourly move (as tested)", mk(True, {"turn": "half1h"}))),
        ("R13", ("BTC on 2.5 hourly moves", mk(True, btc={"target_atr": 2.5, "exit": "FIXED"}))),
        ("R14", ("BTC on its runner (today's exit)", mk(True, btc={"target_atr": None, "exit": "RUNNER"}))),
        ("R15", ("take profit at the sphere for every target", mk(True, {"tp_line_only": False}))),
        ("R16", ("no channel targets", mk(True, {"channel_target": False}))),
        ("R17", ("package without its bigger-timeframe check", mk(True, {"pkg_majority": False}))),
        ("R18", ("no wall lock and no staircase", mk(True, exits=False))),
    ])


def guard_check():
    """9.1 hard check: the trade manager's guard lets the lock and the staircase through (no [NS-STOP-HOLD])."""
    try:
        from src.execution.veteran_trade_manager import VeteranTradeManager as V
        held = []

        class Hd(logging.Handler):
            def emit(self, r):
                if "[NS-STOP-HOLD]" in r.getMessage():
                    held.append(r.getMessage())
        lg = logging.getLogger("src.execution.veteran_trade_manager")
        lg.addHandler(Hd())
        lg.setLevel(logging.INFO)
        res = []
        for reason, side, e, s0, new in (("ns_wall_lock", "long", 100.0, 99.0, 100.0), ("ns_wall_stair", "long", 100.0, 100.0, 100.4),
                                         ("ns_wall_lock", "short", 100.0, 101.0, 100.0), ("ns_wall_stair", "short", 100.0, 100.0, 99.6)):
            v = object.__new__(V)
            v.asset, v.side, v.entry_price, v.current_stop_loss, v.initial_stop_loss = "TEST", side, e, s0, s0
            v.risk_config, v._sl_lock, v.sl_path, v._pending_moves = {"phase_config": {}}, threading.RLock(), [], []
            v.ns_exit, v.ns_r2, v.ns_atr1 = "FIXED", 99.5, 0.5
            res.append(bool(v._propose_stop(new, reason)) and abs(v.current_stop_loss - new) < 1e-12)
        ok = all(res) and not held
        return ("PASS -- the lock and the staircase moves all went through (4 of 4); none held as [NS-STOP-HOLD]" if ok else
                "FAIL -- %s; held: %s" % (res, held[:2]))
    except Exception as ex:
        return "FAIL -- the check could not run: %s" % ex


def t4_spheres(asset, mp, acc, acc_old):
    """14.4 A: T4 re-run with spheres -- price reaching each kind of line (entering its sphere from at least a quarter
    move outside, not already there in the 3 candles before): BROKE = a 4H close past its three-quarter mark first;
    HELD = a 4H close a full 4H move back from the sphere first; within 30 4H candles. The old point version beside it."""
    c4, hi4, lo4 = mp.c4, mp.hi4, mp.lo4
    n = mp.n
    for x in range(250, n - 1):
        b = x - 1
        m = mp.members(b)
        if not len(m["lv"]):
            continue
        a, p = m["a"], m["p"]
        for d in (1, -1):
            for B in mp.bands(b, d, p, LM.MAP_WIN):
                lab = B["label"] if not B["big"] else "BIG WALL: " + B["label"]
                r = _approach_sphere(c4, hi4, lo4, x, d, B, a)
                if r is not None:
                    acc[lab][r] += 1
                    if B["big"]:
                        acc["(any BIG WALL)"][r] += 1
                r0 = _approach_point(c4, hi4, lo4, x, B["near"], a)
                if r0 is not None:
                    acc_old[lab][r0] += 1
            for off in (0.7, 1.3, 1.9, 2.6):                         # the control: levels at fixed distances
                lv = p + d * off * a
                B = dict(near=lv, s_near=lv - d * 0.5 * a, mark=lv + d * 0.25 * a)
                r = _approach_sphere(c4, hi4, lo4, x, d, B, a)
                if r is not None:
                    acc["control: a level at random"][r] += 1


def _approach_sphere(c4, hi4, lo4, x, d, B, a):
    sn, mk = B["s_near"], B["mark"]
    if not (d * (sn - c4[x - 1]) >= 0.25 * a and ((d == 1 and hi4[x] >= sn) or (d == -1 and lo4[x] <= sn))):
        return None
    for z in (x - 1, x - 2, x - 3):
        if z >= 0 and ((d == 1 and hi4[z] >= sn) or (d == -1 and lo4[z] <= sn)):
            return None
    for y in range(x, min(len(c4), x + 30)):
        if d * (c4[y] - mk) > 0:
            return "broke"
        if d * (sn - c4[y]) >= 1.0 * a:
            return "held"
    return "neither"


def _approach_point(c4, hi4, lo4, x, lv, a):
    """The 8 Oct version (a line, no sphere), for comparison."""
    if c4[x - 1] < lv - 0.25 * a and hi4[x] >= lv - 0.1 * a:
        side = 1
    elif c4[x - 1] > lv + 0.25 * a and lo4[x] <= lv + 0.1 * a:
        side = -1
    else:
        return None
    for z in (x - 1, x - 2, x - 3):
        if z >= 0 and lo4[z] <= lv + 0.1 * a and hi4[z] >= lv - 0.1 * a:
            return None
    for y in range(x, min(len(c4), x + 30)):
        r = side * (c4[y] - lv) / a
        if r >= 1.0:
            return "broke"
        if r <= -1.0:
            return "held"
    return "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all")
    ap.add_argument("--markets", default="all")
    ap.add_argument("--months", type=float, default=16.0)
    ap.add_argument("--replay-from", default="2026-09-22")
    ap.add_argument("--no-t4", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%d_%H%M")
    say("=" * 110)
    say("B14 BOX TEST -- %s (read-only)" % stamp)
    say("=" * 110)
    cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    pc0 = cfg.get("phase_config", {}) or {}
    markets = list(MARKETS) if args.markets == "all" else [m.strip().upper() for m in args.markets.split(",")]
    table = rows_table(pc0)
    want = list(table) if args.rows == "all" else [r.strip().upper() for r in args.rows.split(",")]
    usd, spreads = lot_values()
    paid = log_spreads()
    try:
        from src.execution.shadow_trader import FRICTION_PENALTIES as _FP, _DEFAULT_FRICTION as _DF
    except Exception:
        _FP, _DF = {}, 0.0003
    RefEngine, RefPkg = load_b13_ref()
    data = {}
    for a in markets:
        df, note = load_1h(MARKETS[a])
        if df is None:
            say("%-7s skipped: %s" % (a, note))
            continue
        df = df[df.index >= df.index[-1] - pd.Timedelta(days=int(args.months * 30.4))]
        fr = mt5_frames(MARKETS[a])
        approx = "M30" not in fr
        m30 = fr.get("M30") if not approx else m30_from_1h(df)
        m30 = m30[m30.index >= df.index[0]]
        d1, w1 = fr.get("D1"), fr.get("W1")
        mp = LM.MarketMap(a, df, d1, w1)
        frames = {"M30": m30, "D1": d1 if d1 is not None else mp.d1, "W1": w1 if w1 is not None else mp.w1}
        cost = paid.get(a, spreads.get(a, _FP.get(a, _DF)))
        weeks = (df.index[-1] - df.index[0]).total_seconds() / (7 * 86400.0)
        data[a] = dict(df=df, m30=m30, mp=mp, frames=frames, cost=cost, usd=usd[a], weeks=weeks, approx=approx)
        say("%-7s %s | 30m: %s | map: %d 4H candles, %d diagonals%s | spread %.5f%% | $%.4g per 1.0 at the smallest lot" % (
            a, note, "MT5" if not approx else "APPROXIMATE (made from the hourly file -- MT5 gave none)", mp.n,
            len(mp.diags), (", left out: " + ", ".join(mp.ma_missing)) if mp.ma_missing else "", 100 * cost, usd[a]))
    if not data:
        say("no market could be loaded -- nothing to test")
        return
    res = collections.OrderedDict()
    sig_cache = {}
    for rid in want:
        if rid not in table:
            continue
        name, spec = table[rid]
        res[rid] = dict(name=name, per={}, log=collections.Counter(), rows=[])
        for a, D in data.items():
            pcfg = spec["pc"]
            key = json.dumps({"m": a, "pc": NE.rules_stamp(NE.market_settings(a, pcfg)),
                              "pk": pcfg.get("b14_knobs"), "on": pcfg.get("b14_rules_enabled")}, sort_keys=True, default=str)
            if key not in sig_cache:
                sig_cache[key] = run_signals(a, D["df"], D["frames"], D["mp"], pcfg)
            entries, lg = sig_cache[key]
            rows = book(entries, D["m30"], D["mp"], D["cost"], D["usd"], spec["exits"], h1=D["df"])
            res[rid]["per"][a] = (stats(rows, D["weeks"]), rows)
            res[rid]["log"].update(lg)
            res[rid]["rows"] += rows
        say("   %s done (%.0f s)" % (rid, time.time() - t0))
    weeks = float(np.mean([D["weeks"] for D in data.values()]))
    # ---- R0 identity against the B13 files
    say("\n" + "=" * 110 + "\nFIRST CHECK: the B14 code with its switch off against the B13 engine and package (tools\\b14_b13ref)\n" + "=" * 110)
    if RefEngine is None:
        say("   NOT RUN: tools\\b14_b13ref\\ns_engine.py and ns_package.py are missing")
    elif "R0" in res:
        bad = []
        for a, D in data.items():
            ref_entries, _lg = run_signals(a, D["df"], D["frames"], D["mp"], table["R0"][1]["pc"], eng_cls=RefEngine,
                                           pkg=RefPkg, ref=True)
            k_ref = sorted((str(e["proof"]["key"]), e["proof"]["fields"]["ns_candle"], round(e["proof"]["fields"]["ns_stop"], 8))
                           for e in ref_entries)
            r0_entries = sig_cache[json.dumps({"m": a, "pc": NE.rules_stamp(NE.market_settings(a, table["R0"][1]["pc"])),
                                               "pk": table["R0"][1]["pc"].get("b14_knobs"),
                                               "on": table["R0"][1]["pc"].get("b14_rules_enabled")}, sort_keys=True, default=str)][0]
            k_r0 = sorted((str(e["proof"]["key"]), e["proof"]["fields"]["ns_candle"], round(e["proof"]["fields"]["ns_stop"], 8))
                          for e in r0_entries)
            same = k_ref == k_r0
            say("   %-7s %s  (%d entries B13, %d entries B14 switch off)" % (a, "IDENTICAL" if same else "DIFFERENT", len(k_ref), len(k_r0)))
            if not same:
                bad.append(a)
                for x in sorted(set(k_ref) ^ set(k_r0))[:5]:
                    say("           only in %s: %s" % ("B13" if x in k_ref else "B14", x))
        say("   RESULT: %s" % ("PASS -- with the switch off the B14 code trades exactly as B13" if not bad else
                               "FAIL for %s -- tell Claude before anything is switched on" % ", ".join(bad)))
    # ---- the table
    say("\n" + "=" * 110 + "\nTHE ROWS (all markets together; %.0f weeks; money = the smallest lot after the spread)\n" % weeks + "=" * 110)
    say("%-4s %-52s %6s %6s %8s %8s %9s %9s" % ("row", "what changes", "trades", "win%", "total R", "$ total", "$ / week", "worst $"))
    base = None
    for rid, r in res.items():
        s = stats(r["rows"], weeks)
        if rid == "R1":
            base = s
        say("%-4s %-52s %6d %6.1f %8.1f %8.1f %9.2f %9.1f" % (rid, r["name"][:52], s["n"], s["win"], s["R"], s["usd"], s["wk"], s["dd"]))
    say("\n   R2..R18 change ONE thing from R1. Compare each with R1 (B14 as approved) and R0 (today's rules).")
    for rid in ("R0", "R1"):
        if rid not in res:
            continue
        say("\n--- %s per market ---" % rid)
        for a, (s, rows) in res[rid]["per"].items():
            hows = collections.Counter(x["how"] for x in rows)
            srcs = collections.Counter(x["src"] for x in rows)
            say("   %-7s trades %3d  win %5.1f%%  R %6.1f  $ %8.1f  ($%.2f/wk)  worst $%.1f | exits %s | from %s" % (
                a, s["n"], s["win"], s["R"], s["usd"], s["wk"], s["dd"], dict(hows), dict(srcs)))
        lg = res[rid]["log"]
        say("   signals: %s" % ", ".join("%s %d" % kv for kv in sorted(lg.items())))
        if rid == "R1":
            rr = res[rid]["rows"]
            say("   locks fired: %d of %d trades; staircase steps: %d; RUNNER entries: %d" % (
                sum(1 for x in rr if x["locked"]), len(rr), sum(x["steps"] for x in rr), sum(1 for x in rr if x["label"] == "RUNNER")))
    say("\n" + "=" * 110 + "\nGUARD CHECK (9.1): " + guard_check())
    # ---- T4 with spheres
    if not args.no_t4:
        say("\n" + "=" * 110 + "\nT4 RE-RUN WITH SPHERES (14.4 A): what price does when it reaches each kind of line\n" + "=" * 110)
        acc, acc_old = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        for a, D in data.items():
            t4_spheres(a, D["mp"], acc, acc_old)
        say("%-58s %7s %7s %7s %7s | %s" % ("kind of line", "visits", "held%", "broke%", "neither", "the 8 Oct point version: held%"))
        for k in sorted(acc, key=lambda k_: (k_.startswith("control"), k_.startswith("(any"), -sum(acc[k_].values()))):
            n = sum(acc[k].values())
            no = sum(acc_old[k].values())
            say("%-58s %7d %6.1f%% %6.1f%% %7d | %s" % (k[:58], n, 100.0 * acc[k]["held"] / max(n, 1), 100.0 * acc[k]["broke"] / max(n, 1),
                                                     acc[k]["neither"], ("%.1f%% of %d" % (100.0 * acc_old[k]["held"] / no, no)) if no else "-"))
        say("   A line that means nothing scores about the control's held%. Spheres: half a 4H move (big), a quarter (small).")
    # ---- replays
    say("\n" + "=" * 110 + "\nREPLAYS (9.2): every trade since %s, B13 rules (R0) and B14 (R1)\n" % args.replay_from + "=" * 110)
    t_from = pd.Timestamp(args.replay_from)
    for a in ("BTC", "GBPAUD", "USOIL", "USTEC"):
        if a not in data:
            continue
        for rid in ("R0", "R1"):
            if rid not in res:
                continue
            rows = [x for x in res[rid]["per"].get(a, (None, []))[1] if x["t_in"] >= t_from]
            say("   %s %s: %d trade(s)" % (a, rid, len(rows)))
            for x in rows:
                say("      %s %s %.6g -> %.6g %s  %+.2fR  $%+.1f  %s%s" % (
                    str(x["t_in"])[:16], "BUY " if x["d"] == 1 else "SELL", x["q"], x["px"], x["how"], x["R"], x["usd"],
                    x["src"], ("  | " + str(x["route"])[:70]) if x.get("route") else ""))
    say("   The live USOIL sell of 6 Oct (MT5 #141483213, signal at the 07:00 UTC 1H close): look for USOIL 2026-10-06 "
        "in both lists -- R0 shows what B13 did with it, R1 what B14 would do.")
    say("\nfinished in %.0f s" % (time.time() - t0))
    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", "b14_box_test_%s.txt" % stamp)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print("report written:", out)


if __name__ == "__main__":
    main()
