"""B14 BOX TEST (Desire 15-16 Oct 2026) -- READ-ONLY. Run with the bot running or stopped: it reads the price files,
the config, the logs (the spreads the bot paid) and MT5's own 30-minute / daily / weekly candles (read-only, or the
export files of tools\\b15_export.py when MT5 is not reachable); it never sends an order and changes no file of the bot.
It writes only its report logs\\b14_box_test_<stamp>.txt (+ .pkl with every booked trade).

  python tools\\b14_box_test.py                      -- 27 months: the last 16 months AND the 11 months before them
                                                       (the second sample)
  python tools\\b14_box_test.py --months 16          -- the last 16 months only (quicker)
  python tools\\b14_box_test.py --markets BTC,GOLD   -- some markets only
  python tools\\b14_box_test.py --no-ref             -- skip the identity check (quicker)

It runs the bot's own files (ns_engine, ns_package, line_map, b14_rules), so it can never drift from them:
  1. the line map is built once per market from the hourly file (+ the daily/weekly candles);
  2. the engine replays every closed 1H candle in one pass (each candle counted as live, as it is at the look right
     after it closes);
  3. every hand-over goes to the package, judged on the 30-minute closes exactly as live (24 h at most); a re-armed
     setup is put back and replayed from its cancel;
  4. trades are booked one position per market at a time and run on the 30-minute candles: stop, target, the 7-day
     exit (168 closed 1H candles), BTC's runner on the B13 rows, the R-ladder on the new-B14 rows. Money = the
     smallest lot, minus the spread the bot really paid.

What it prints:
  FIRST CHECK  the new code with the B14 switch OFF against the frozen engine and package the box ran on 15 Oct
               (tools\\b14_ref, commit a5e55ce): every engine AND package entry must be IDENTICAL on every market.
  THE TABLE    per window: S0 = the B13 rules (switch off); B14 = the new B14 rules with the settings in force
               (phase_config.b14_tuning; none = the rules file's own); B14-NR = without the re-arm; B14-ALL = the
               re-arm after every cancel -- each with trades, win rate, R, money, R per trade, PF, worst drop, the
               longest run of losses.
  B14 FACTS    the witness split, where the targets came from, how trades ended, the ladder rungs reached, package
               entries, signals retired for the clearance, the re-arm counts, the wider witness's takes.
It is also the learner's library (tools\\b14_rl.py): prepare(), run_row(), book_row(), stats()."""
import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import re
import sys
import time
import types

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
logging.basicConfig(level=logging.ERROR, format="%(message)s")
from src.execution import line_map as LM          # noqa: E402
from src.execution import ns_engine as NE         # noqa: E402
from src.execution import ns_package as NP        # noqa: E402
from src.execution import b14_rules as R14        # noqa: E402
for _n in ("src.execution.ns_engine.explore", "src.execution.line_map", "src.execution.livermore_state_machine"):
    logging.getLogger(_n).setLevel(logging.ERROR)

MARKETS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}
FALLBACK_USD = {"BTC": 0.01, "GOLD": 1.0, "SILVER": 50.0, "USOIL": 10.0, "USTEC": 0.05, "JP225": 0.02,
                "EURUSD": 1000.0, "GBPAUD": 660.0, "EURJPY": 6.7, "AUDJPY": 6.7}
OUT = []
STAMP = time.strftime("%Y-%m-%d_%H%M")
QUIET = {"on": False}
LOG_N = collections.Counter()        # the engine's and the package's own lines, counted per row while they run


def say(s=""):
    OUT.append(s)
    if not QUIET["on"]:
        print(s, flush=True)


class _H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if m.startswith("[DIAG-CONFIRM]"):
            LOG_N[(_H.row, "confirm")] += 1
        elif m.startswith("[PKG-CANCEL]") and "B14: the push closed only" in m:
            LOG_N[(_H.row, "clearance cancel")] += 1
        elif m.startswith("[PKG-ENTER]"):
            LOG_N[(_H.row, "package entries")] += 1
            if "springboard at" in m:
                LOG_N[(_H.row, "spring entries")] += 1
        elif m.startswith("[PKG-SPRING]") and "waits at the wall" in m:
            LOG_N[(_H.row, "spring waits")] += 1
        elif m.startswith("[PKG-CANCEL]") and ("B14 springboard" in m or "while waiting at the wall" in m):
            LOG_N[(_H.row, "spring cancels")] += 1
        elif m.startswith("[NS-SKIP]") and "B14: the close beat the peak" in m:
            LOG_N[(_H.row, "clearance skip")] += 1
        elif m.startswith("[PKG-REARM]") and "re-armed after" in m:
            LOG_N[(_H.row, "re-arms")] += 1


_H.row = "-"
for _nm in ("src.execution.ns_engine", "src.execution.ns_package"):
    _l = logging.getLogger(_nm)
    _l.setLevel(logging.INFO)
    _l.addHandler(_H())
    _l.propagate = False


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


def _export_frames(sym):
    """The candles tools\\b15_export.py saved from MT5 (data\\b15_export\\<sym>_30m.csv / _1d.csv / _1w.csv), used only
    when MT5 itself is not reachable."""
    out = {}
    for name, suf in (("M30", "30m"), ("D1", "1d"), ("W1", "1w")):
        p = os.path.join("data", "b15_export", "%s_%s.csv" % (sym, suf))
        if not os.path.exists(p):
            continue
        try:
            f = pd.read_csv(p, parse_dates=[0], index_col=0)
            f.index = pd.DatetimeIndex(f.index).tz_localize(None) if getattr(f.index, "tz", None) is not None else pd.DatetimeIndex(f.index)
            out[name] = f[["open", "high", "low", "close"]].astype(float).sort_index()
            out["_src"] = "export"
        except Exception:
            continue
    return out


def mt5_frames(sym):
    """30-minute, daily and weekly candles straight from MT5 (count-based: no local-time conversion); without MT5,
    the export files if they are there."""
    out = {}
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return _export_frames(sym)
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
    return out or _export_frames(sym)


def m30_from_1h(df):
    """Only when there are no 30-minute candles: two halves per hour (APPROXIMATE -- the report says so)."""
    rows = []
    for t, (o, h, l, c) in zip(df.index, df[["open", "high", "low", "close"]].values):
        m = (o + c) / 2.0
        rows.append((t, o, max(o, m), min(o, m), m))
        rows.append((t + pd.Timedelta(minutes=30), m, max(m, h), min(m, l), c))
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).set_index("t")


def lot_values():
    """US dollars per 1.0 price move at the smallest lot (MT5; the export's copy; a fallback table) and the spreads."""
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
        _pj = os.path.join("data", "b15_export", "lot_values.json")
        try:
            _j = json.load(open(_pj, encoding="utf-8"))
            out, spreads = dict(_j.get("usd") or {}), dict(_j.get("spreads") or {})
            say("   MT5 not reachable (%s) -- lot values and spreads from the export %s (%s)" % (e, _pj, _j.get("exported")))
        except Exception:
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


LAST_STATE = {}      # asset -> the hourly engine's memory after the last run_signals (only the last run is kept)


def run_signals(asset, df1, frames, mp, pcfg, eng_cls=BoxEngine, pkg=NP, ref=False):
    """One market, one row: the engine pass, the package on every hand-over, re-arms replayed from their cancel.
    ref=True runs the frozen 15 Oct copy (it fetches no map: with the old switches off it never used one).
    -> (entries [{src, proof}], log Counter)"""
    end = df1.index[-1] + pd.Timedelta(hours=1)
    eng = eng_cls(asset)
    r = eng.update(None, df1, None, pcfg, None, now=end) if ref else eng.update(None, df1, None, pcfg, None, now=end, mp=mp)
    st = r["state"]
    LAST_STATE[asset] = st          # the engine's memory at the end of the data (the weekly report's waiting list)
    entries, log = [], collections.Counter()
    for p in r["proofs"]:
        entries.append(dict(src="hourly", proof=p))
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
        if not ref and rec.get("status") == "watching" and rec.get("spring"):
            # B14 (Desire 16 Oct): waiting at a wall -- follow it to the end of the springboard (as the live looks do)
            now2 = min(pd.Timestamp(rec["spring"]["t0"]) + pd.Timedelta(hours=float(R14.SPRING_WAIT_H) + 2), end)
            prs2, _ev2 = pkg.step(asset, sub, df1, pcfg, now=now2, frames=frames, mp=mp)
            for p in prs2:
                entries.append(dict(src="package", proof=p))
        log["package %s" % rec.get("status")] += 1
        for s in sub.get("setups") or []:                          # re-armed: replay it from the cancel
            if (s["id"], s.get("rearms")) in seen_ids:
                continue
            seen_ids.add((s["id"], s.get("rearms")))
            log["re-arms"] += 1
            st2 = NE.NSEngine.new_state()
            st2["setups"] = [s]
            st2["next_id"] = int(s["id"]) + 1
            st2["last_1h"] = pd.Timestamp(s["t_break"]).floor("h")
            st2["rules"] = NE.rules_stamp(NE.market_settings(asset, pcfg), None)
            e2 = ReplayEngine(asset)
            r2 = e2.update(st2, df1, None, pcfg, None, now=end, mp=mp)
            for p in r2["proofs"]:
                entries.append(dict(src="re-armed", proof=p))
                log["re-armed: taken at once"] += 1
            todo += [x for x in (r2["state"].get("pkg") or []) if x.get("status") == "watching"]
    return entries, log


# ---- the trades ------------------------------------------------------------------------------------------------------
def simulate(e, m30, cost, usd, exits_on, h1=None, ladder=None):
    """One trade on the 30-minute candles: stop, target, the 7 days (168 1H candles CLOSED after the entry, as the
    live trade manager counts them); BTC's runner (the B13 exit) trails on each 1H close with the engine's own
    runner_step; a new-B14 proof (FIXED exit) runs the R-ladder -- the same b14_rules.ladder_stop the trade manager
    and the practice trader run -- on each candle's best price, after that candle's stop and target.
    `ladder` (the learner only) replaces the proof's rungs, to test another ladder on the same trades.
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
    cur, how = stop, None
    runner = f.get("ns_exit") == "RUNNER" and h1 is not None
    nb14 = exits_on and bool(f.get("ns_b14n")) and f.get("ns_exit") == "FIXED"
    st14 = R14.exit_init(d, q, stop, tgt, ladder=(ladder or f.get("ns_b14n_ladder_def"))) if nb14 else None
    armed, peak = False, None
    if runner:
        t1c = h1.index + pd.Timedelta(hours=1)
        hi1a, lo1a, cl1a = h1["high"].values, h1["low"].values, h1["close"].values
    Hh, L, C = (m30[c].values for c in ("high", "low", "close"))
    ti = m30.index
    k_end, px = None, None
    n_h1 = 0                                      # 1H candles closed since the entry (the live trade manager's count)
    for k in range(k0, len(m30)):
        hi, lo = Hh[k], L[k]
        if (d == 1 and lo <= cur) or (d == -1 and hi >= cur):
            k_end, px = k, cur
            how = "ladder stop" if (st14 and st14["step"]) and d * (cur - q) >= 0 else "stop"
            break
        if tgt is not None and ((d == 1 and hi >= tgt) or (d == -1 and lo <= tgt)):
            k_end, px, how = k, tgt, "target"
            break
        new = None
        if runner and (ti[k] + pd.Timedelta(minutes=30)).minute == 0:                    # BTC's runner, as live
            j = int(np.searchsorted(t1c.values, np.datetime64(ti[k] + pd.Timedelta(minutes=30)), side="right")) - 1
            if j >= 0:
                a0 = max(0, j - 260)
                bars = int((ti[k] + pd.Timedelta(minutes=30) - t_in).total_seconds() // 3600)
                rn, armed, peak, _w = NE.runner_step(d, q, cur, risk0, hi1a[a0:j + 1], lo1a[a0:j + 1], cl1a[a0:j + 1],
                                                     bars, armed, peak if peak is not None else q)
                if rn is not None and (new is None or d * (rn - new) > 0):
                    new = rn
        if nb14:
            l14, _w14 = R14.ladder_stop(st14, d, q, hi if d == 1 else lo, new if new is not None else cur)
            if l14 is not None and (new is None or d * (l14 - new) > 0):
                new = l14
        if new is not None and d * (new - cur) > 0:
            cur = new
        if (ti[k] + pd.Timedelta(minutes=30)).minute == 0 and ti[k] + pd.Timedelta(minutes=30) > t_in:
            n_h1 += 1                             # a 1H candle closed
        if n_h1 >= NE.TRADE_BARS:
            k_end, px, how = k, C[k], "7 days"
            break
    if k_end is None:
        return None
    pts = d * (px - q) - cost * q
    return dict(t_in=t_in, t_out=ti[k_end] + pd.Timedelta(minutes=30), d=d, q=q, px=px, how=how,
                R=pts / risk0 if risk0 else 0.0, usd=pts * usd, src=e["src"],
                ladder=int(st14["step"]) if st14 else 0, key=e["proof"]["key"], f=f)


def book(entries, D, exits_on, ladder=None):
    """One position per market at a time, in time order (the live bot never adds to an open trade)."""
    rows, free = [], pd.Timestamp("1900-01-01")
    for e in sorted(entries, key=lambda x: pd.Timestamp(x["proof"]["fields"]["ns_candle"])):
        t_in = pd.Timestamp(e["proof"]["fields"]["ns_candle"])
        if t_in < free:
            continue
        r = simulate(e, D["m30"], D["cost"], D["usd"], exits_on, h1=D["df"], ladder=ladder)
        if r is None:
            continue
        rows.append(r)
        free = r["t_out"]
    return rows


def stats(rows):
    rs = sorted(rows, key=lambda x: x["t_out"])
    if not rs:
        return dict(n=0)
    R = np.array([x["R"] for x in rs])
    U = np.array([x["usd"] for x in rs])
    eqU, eqR = np.cumsum(U), np.cumsum(R)
    ddU = float(np.max(np.maximum.accumulate(np.r_[0.0, eqU]) - np.r_[0.0, eqU]))
    ddR = float(np.max(np.maximum.accumulate(np.r_[0.0, eqR]) - np.r_[0.0, eqR]))
    g, l = R[R > 0].sum(), -R[R < 0].sum()
    run, lrun = 0, 0
    for x in rs:
        run = run + 1 if x["R"] < 0 else 0
        lrun = max(lrun, run)
    return dict(n=len(rs), win=100 * (R > 0).mean(), R=R.sum(), usd=U.sum(), avgR=R.mean(),
                pf=(g / l) if l > 0 else float("inf"), ddR=ddR, ddU=ddU, lrun=lrun)


def row_line(name, s):
    if s["n"] == 0:
        return "   %-46s      0 trades" % name
    return "   %-46s %5d trades  %5.1f%% won  %+8.1fR  $%+7.0f  %+.3fR/trade  PF %s  worst drop %5.1fR / $%4.0f  run %d" % (
        name, s["n"], s["win"], s["R"], s["usd"], s["avgR"], ("inf" if math.isinf(s["pf"]) else "%.2f" % s["pf"]),
        s["ddR"], s["ddU"], s["lrun"])


# ---- the identity reference ------------------------------------------------------------------------------------------
def load_ref():
    """The engine and package the box ran on 15 Oct (commit a5e55ce), frozen in tools\\b14_ref -- the identity check."""
    d = os.path.join("tools", "b14_ref")
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


def entry_sig(e):
    """What must be identical: the key, the candle, the stop, the close and the target of every entry."""
    f = e["proof"]["fields"]
    tg = f.get("ns_target")
    return (str(e["proof"]["key"]), f["ns_candle"], f.get("ns_entry"), round(float(f["ns_stop"]), 8),
            round(float(f["ns_close"]), 8), None if tg in (None, "None") else round(float(tg), 8))


def fp(path):
    try:
        return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16].upper()
    except Exception:
        return "MISSING"


# ---- the library the learner calls -----------------------------------------------------------------------------------
def prepare(months=27.0, markets="all"):
    """Load every market once: the hourly file (from `months` before the earliest file end), the 30-minute / daily /
    weekly candles, the line map, the spread, the lot value. -> dict(data, windows, pc0, t_end)"""
    cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    pc0 = dict(cfg.get("phase_config", {}) or {})
    names = list(MARKETS) if markets == "all" else [m.strip().upper() for m in str(markets).split(",")]
    usd, spreads = lot_values()
    paid = log_spreads()
    try:
        from src.execution.shadow_trader import FRICTION_PENALTIES as _FP, _DEFAULT_FRICTION as _DF
    except Exception:
        _FP, _DF = {}, 0.0003
    raw = {}
    for a in names:
        df, note = load_1h(MARKETS[a])
        if df is None:
            say("%-7s skipped: %s" % (a, note))
            continue
        raw[a] = (df, note)
    if not raw:
        return None
    t_end = min(df.index[-1] for df, _n in raw.values())
    t_start = t_end - pd.Timedelta(days=int(months * 30.4))
    first = max(df.index[0] for df, _n in raw.values())
    say("every market starts on %s (%.0f months before the earliest file end %s); the latest-starting file begins %s" % (
        str(t_start)[:16], months, str(t_end)[:16], str(first)[:16]))
    w16 = (t_end - pd.Timedelta(days=int(16 * 30.4)), t_end + pd.Timedelta(days=1))
    windows = [("the last 16 months", w16)]
    if months >= 26:
        windows.append(("the 11 months before (second sample)", (t_start, w16[0])))
        late = [a for a, (df, _n) in raw.items() if df.index[0] > t_start + pd.Timedelta(days=7)]
        if late:
            say("   second sample: these files start later than %s, so their part of it is shorter: %s" % (
                str(t_start)[:10], ", ".join("%s from %s" % (a, str(raw[a][0].index[0])[:10]) for a in late)))
    else:
        say("   second sample NOT run: --months is below 26 (use --months 27 for both windows)")
    say("   NOTE: a longer replay changes what the engine knows at the start of the last 16 months (older lines, diagonals,"
        " brain history), so the 16-month numbers of a 27-month run differ a little from a 16-month run -- compare like with like")
    data = {}
    for a, (df, note) in raw.items():
        df = df[df.index >= t_start]
        fr = mt5_frames(MARKETS[a])
        approx = "M30" not in fr
        m30 = fr.get("M30") if not approx else m30_from_1h(df)
        m30 = m30[m30.index >= df.index[0]]
        d1, w1 = fr.get("D1"), fr.get("W1")
        mp = LM.MarketMap(a, df, d1, w1)
        frames = {"M30": m30, "D1": d1 if d1 is not None else mp.d1, "W1": w1 if w1 is not None else mp.w1}
        cost = paid.get(a, spreads.get(a, _FP.get(a, _DF)))
        data[a] = dict(df=df, m30=m30, mp=mp, frames=frames, cost=cost, usd=usd[a], approx=approx)
        say("%-7s %s | 30m: %s | map: %d 4H candles, %d daily, %d weekly | spread %.5f%% | $%.4g per 1.0 at the smallest lot" % (
            a, note, ("MT5" if fr.get("_src") != "export" else "the export files") if not approx else
            "APPROXIMATE (made from the hourly file)", mp.n, len(mp.d1) if mp.d1 is not None else 0,
            len(mp.w1) if mp.w1 is not None else 0, 100 * cost, usd[a]))
    return dict(data=data, windows=windows, pc0=pc0, t_end=t_end)


def run_row(data, pc, settings=None, row="-"):
    """One row on every market: the new rules' settings set for this row (b14_rules.apply_settings; None = the
    rules file's own), the signals, then the settings put back. -> dict(entries={asset: [...]}, log=Counter)"""
    _H.row = row
    keep = {k: getattr(R14, k) for k in R14.TUNABLE}
    _v, _chg, bad = R14.apply_settings(settings)
    if bad:
        say("   row %s: settings refused -- %s" % (row, "; ".join("%s (%s)" % x for x in bad)))
    try:
        ent, lg = {}, collections.Counter()
        for a, D in data.items():
            ent[a], l1 = run_signals(a, D["df"], D["frames"], D["mp"], pc)
            lg.update(l1)
        return dict(entries=ent, log=lg)
    finally:
        for k, v in keep.items():
            setattr(R14, k, v)


def book_row(data, entries, exits_on, ladder=None):
    """Book every market's entries (one position per market at a time). -> rows, each with its asset."""
    rows = []
    for a, D in data.items():
        for x in book(entries.get(a) or [], D, exits_on, ladder=ladder):
            x["asset"] = a
            rows.append(x)
    return rows


def in_window(rows, w):
    lo, hi = w
    return [x for x in rows if lo <= x["t_in"] < hi]


# ---- the report ------------------------------------------------------------------------------------------------------
def report_rows(pc0, more=False):
    """The rows of the report: the B13 rules (the switch off), the new B14 rules with the settings in force, and the new
    rules without the springboard. more=True (--more) adds the decided re-arm rows and the far-route test rows (Desire
    16 Oct: a limit on how far past its line a package entry may be confirmed, with the springboard on)."""
    live = dict(pc0.get("b14_tuning") or {})
    pc_off = dict(pc0, b14_new_rules_enabled=False)
    pc_on = dict(pc0, b14_new_rules_enabled=True)
    rows = [
        ("S0", ("B13 rules (the B14 switch off) -- today", pc_off, live, False)),
        ("B14", ("B14 rules (on, the settings in force)", pc_on, live, True)),
        ("B14-NOSPR", ("B14 rules without the springboard", pc_on, dict(live, SPRING=False), True)),
    ]
    if more:
        rows += [
            ("B14-NR", ("B14 rules without the re-arm", pc_on, dict(live, REARM=False), True)),
            ("B14-ALL", ("B14, re-arm after every cancel", pc_on, dict(live, REARM_AFTER_MAJORITY=True), True)),
            ("B14-FAR2.5", ("B14, far limit 2.5 moves past the line", pc_on, dict(live, FAR_MAX=2.5), True)),
            ("B14-FAR3.5", ("B14, far limit 3.5 moves past the line", pc_on, dict(live, FAR_MAX=3.5), True)),
            ("B14-FAR5", ("B14, far limit 5 moves past the line", pc_on, dict(live, FAR_MAX=5.0), True)),
        ]
    return collections.OrderedDict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="all")
    ap.add_argument("--months", type=float, default=27.0)
    ap.add_argument("--no-ref", action="store_true", help="skip the identity check (faster)")
    ap.add_argument("--more", action="store_true", help="also the re-arm rows and the far-route test rows")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    pc0 = dict(cfg.get("phase_config", {}) or {})
    _vals, _chg, _bad = R14.apply_settings(pc0.get("b14_tuning"))
    say("=" * 118)
    say("B14 BOX TEST -- %s (read-only) | engine rules %s | the new B14 settings in force: %s%s" % (
        STAMP, NE.RULES_V, R14.settings_text(_vals), (" | REFUSED in config: %s" % "; ".join("%s (%s)" % x for x in _bad))
        if _bad else ""))
    say("files tested: " + " | ".join("%s %s" % (p, fp(p)) for p in (
        "src/execution/ns_engine.py", "src/execution/ns_package.py", "src/execution/line_map.py",
        "src/execution/b14_rules.py", "tools/b14_box_test.py", "tools/b14_ref/ns_engine.py",
        "tools/b14_ref/ns_package.py")))
    say("=" * 118)
    say("switch in config.json: b14_new_rules_enabled=%s | ns_paper_markets=%s (the rows below set the switch themselves; "
        "everything else is read from config.json)" % (pc0.get("b14_new_rules_enabled", "(absent)"), pc0.get("ns_paper_markets")))
    P = prepare(args.months, args.markets)
    if P is None:
        say("no market could be loaded -- nothing to test")
        return 1
    data, windows = P["data"], P["windows"]
    rows = report_rows(pc0, more=args.more)
    res = {}
    for rid, (name, pc, settings, exits_on) in rows.items():
        r = run_row(data, pc, settings, row=rid)
        res[rid] = dict(name=name, entries=r["entries"], log=r["log"], rows=book_row(data, r["entries"], exits_on))
        say("   %s done (%.0f s)" % (rid, time.time() - t0))
    # ---- FIRST CHECK: the switch off against the frozen a5e55ce engine and package
    say("\n" + "=" * 118 + "\nFIRST CHECK: the new code with the B14 switch OFF (S0) against the engine and package the box "
        "ran on 15 Oct (tools\\b14_ref, a5e55ce)\n" + "=" * 118)
    RefEngine, RefPkg = (None, None) if args.no_ref else load_ref()
    verdict = "NOT RUN"
    if RefEngine is None:
        say("   NOT RUN: %s" % ("--no-ref" if args.no_ref else "tools\\b14_ref\\ns_engine.py and ns_package.py are missing"))
    else:
        bad = []
        pc_off = rows["S0"][1]
        for a, D in data.items():
            ref_entries, _lg = run_signals(a, D["df"], D["frames"], D["mp"], pc_off, eng_cls=RefEngine, pkg=RefPkg, ref=True)
            s_ref = sorted(entry_sig(e) for e in ref_entries)
            s_new = sorted(entry_sig(e) for e in res["S0"]["entries"][a])
            same = s_ref == s_new
            n_pk = sum(1 for x in s_new if x[2] == "PKG")
            say("   %-7s %s  (%d entries: %d engine, %d package)" % (a, "IDENTICAL" if same else "DIFFERENT",
                                                                     len(s_new), len(s_new) - n_pk, n_pk))
            if not same:
                bad.append(a)
                for x in sorted(set(s_ref) ^ set(s_new))[:5]:
                    say("           only in %s: %s" % ("the 15 Oct code" if x in s_ref else "the new code", x))
        verdict = ("PASS -- with the switch off the new code's entries (engine and package) are exactly the 15 Oct code's"
                   if not bad else "FAIL for %s -- tell Claude before the switch is touched" % ", ".join(bad))
        say("   RESULT: %s" % verdict)
    # ---- the tables, per window
    pk = {"windows": {}, "stamp": STAMP}
    paper = [str(m).upper() for m in (pc0.get("ns_paper_markets") or [])]
    for wname, w in windows:
        say("\n" + "=" * 118 + "\nTHE TABLE -- %s: %s to %s (every market, one trade at a time, at the smallest lot)\n" % (
            wname, str(w[0])[:10], str(w[1])[:10]) + "=" * 118)
        W = {rid: in_window(res[rid]["rows"], w) for rid in rows}
        for rid in rows:
            say(row_line(rows[rid][0], stats(W[rid])))
        say("   per market (S0 -> B14):")
        for a in data:
            s0 = stats([x for x in W["S0"] if x["asset"] == a])
            s1 = stats([x for x in W["B14"] if x["asset"] == a])
            say("      %-7s %3d trades %+7.1fR $%+6.0f  ->  %3d trades %+7.1fR $%+6.0f%s" % (
                a, s0["n"], s0.get("R", 0), s0.get("usd", 0), s1["n"], s1.get("R", 0), s1.get("usd", 0),
                "   (paper market: not traded live)" if a in paper else ""))
        b14 = W["B14"]
        wit1 = [x for x in b14 if x["f"].get("ns_b14n_witness")]
        wit0 = [x for x in b14 if not x["f"].get("ns_b14n_witness")]
        s1, s0 = stats(wit1), stats(wit0)
        say("   B14 facts:")
        say("      witness in the 24 h before the entry: WITH %d trades %.1f%% won %+.1fR (%+.3f/trade) | WITHOUT %d trades %.1f%% won %+.1fR (%+.3f/trade)" % (
            s1["n"], s1.get("win", 0), s1.get("R", 0), s1.get("avgR", 0), s0["n"], s0.get("win", 0), s0.get("R", 0), s0.get("avgR", 0)))
        kinds = collections.Counter(str(x["f"].get("ns_b14n_target_kind")) for x in b14)
        say("      targets: %s" % ", ".join("%s %d" % (k, n) for k, n in sorted(kinds.items())))
        say("      how they ended: %s" % ", ".join("%s %d" % (k, n) for k, n in sorted(collections.Counter(x["how"] for x in b14).items())))
        rung = collections.Counter(int(x.get("ladder", 0)) for x in b14)
        say("      ladder rungs reached: %s" % ", ".join("%d rung%s: %d" % (k, "" if k == 1 else "s", n) for k, n in sorted(rung.items())))
        say("      package entries: %d (S0 %d) | signals retired for the clearance (whole replay): %d at the hourly entry, %d at the package's entry" % (
            sum(1 for x in b14 if x["src"] == "package"), sum(1 for x in W["S0"] if x["src"] == "package"),
            LOG_N[("B14", "clearance skip")], LOG_N[("B14", "clearance cancel")]))
        say("      re-arm (whole replay): %d setups re-armed after a package cancel, %d of them then taken at once by the "
            "engine (the rest went back to the package or ran out)%s" % (
                res["B14"]["log"]["re-arms"], res["B14"]["log"]["re-armed: taken at once"],
                (" | re-arm after every cancel: %d re-arms" % res["B14-ALL"]["log"]["re-arms"]) if "B14-ALL" in res else ""))
        say("      springboard (whole replay): %d package entries waited at a wall less than %gR ahead; %d entered after the "
            "springboard, %d were cancelled (re-armed by option B)" % (
                LOG_N[("B14", "spring waits")], R14.SPRING_DIST_R, LOG_N[("B14", "spring entries")], LOG_N[("B14", "spring cancels")]))
        pk["windows"][wname] = {rid: W[rid] for rid in rows}
    say("\n   taken at once on a witness ([DIAG-CONFIRM], whole replay): S0 %d (3-touch diagonals) -> B14 %d (2+-touch diagonals and the six MAs)" % (
        LOG_N[("S0", "confirm")], LOG_N[("B14", "confirm")]))
    say("\nIDENTITY: %s" % verdict)
    say("done in %.0f s" % (time.time() - t0))
    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", "b14_box_test_%s.txt" % STAMP)
    open(out, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
    pk["entries"] = {rid: res[rid]["entries"] for rid in rows}
    pickle.dump(pk, open(os.path.join("logs", "b14_box_test_%s.pkl" % STAMP), "wb"))
    print("written: %s (+ .pkl)" % out)
    return 0 if verdict.startswith(("PASS", "NOT RUN")) else 1


if __name__ == "__main__":
    sys.exit(main())
