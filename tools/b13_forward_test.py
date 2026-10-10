# TBOT B13 WEEKLY FORWARD TEST (6 Oct 2026) -- read-only. Rebuilds the SILVER long of 5 Oct 10:00 UTC with the bot's own code
# and candles, then tests two fixes over 16 months: lines judged on wicks, and the majority check on trades taken at once.
# Desire's option 1 (2 Oct): the package WITHOUT the bigger-timeframe majority -- only the 30m/1H closes and the
# bigger-line rule decide. 'near' rows send only signals with a bigger 4H line just ahead (or that the engine calls too
# far) to the package and take the rest at once; 'all' rows judge every signal. E + skip-near alongside, same pricing.
import collections, glob, hashlib, importlib.util, json, logging, os, re, sys, time, types
from datetime import timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
ENGINE = os.path.join("src", "execution", "ns_engine.py")
ENGINE_SHA = "849177CE5E31979E3BF0402460CF72249D387E63342C41B829B70866F581E153"   # B14 (9 Oct) engine
MARKETS = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "USOIL": "USOILm", "EURUSD": "EURUSDm",
           "GBPAUD": "GBPAUDm", "JP225": "JP225m", "EURJPY": "EURJPYm", "SILVER": "XAGUSDm", "AUDJPY": "AUDJPYm"}
FALLBACK_USD = {"BTC": 0.01, "GOLD": 1.0, "SILVER": 50.0, "USOIL": 10.0, "USTEC": 0.05, "JP225": 0.02,
                "EURUSD": 1000.0, "GBPAUD": 660.0, "EURJPY": 6.7, "AUDJPY": 6.7}
B12_LIVE, B121_LIVE = pd.Timestamp("2026-09-28 21:50"), pd.Timestamp("2026-09-30 16:30")   # UTC (box 23:50 / 18:30)
STOPS_RETEST = [("today", None), ("low+0.3", 0.3), ("low+0.5", 0.5), ("low+0.75", 0.75)]
STOPS_BREAK = [("today", None), ("room0.5", 0.5), ("room0.75", 0.75)]
T0 = time.time()


def say(s=""):
    print(s, flush=True)


def stub_brain():
    """Load the brain module by path so the engine's brain checks work without importing the whole bot."""
    p = os.path.join("src", "execution", "livermore_state_machine.py")
    if not os.path.exists(p):
        return "brain file not found -- brain-level checks see no levels"
    try:
        for name in ("src", "src.execution"):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = [os.path.join(*name.split("."))]
                sys.modules[name] = mod
        spec = importlib.util.spec_from_file_location("src.execution.livermore_state_machine", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        sys.modules["src.execution"].livermore_state_machine = m
        return "brain module loaded"
    except Exception as e:
        return "brain module failed to load (%s) -- brain-level checks see no levels" % e


def load_engine(tag, hourly):
    spec = importlib.util.spec_from_file_location("ns_engine_" + tag.replace("/", "_"), ENGINE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.REPLAY_DAYS = 100000                      # replay the whole history, not just 30 days
    if hourly:                                  # 1H/1H: hourly candles stand in for the 4H ones
        _ct = m.close_times
        m.close_times = lambda df, hours: _ct(df, 1)
        m.hourly_to_4h = lambda df1: df1[["open", "high", "low", "close"]].astype(float)
        m.WAIT_BREAK, m.WAIT_TOUCH, m.WAIT_TRIGGER = timedelta(hours=60), timedelta(hours=42), timedelta(hours=42)
    return m


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
    return df, "%d hourly candles %s .. %s" % (len(df), df.index[0], df.index[-1])


def read_logs():
    """Spreads the bot actually used ([FRICTION] ... used=) and the live proofs ([NS-PROOF]) since B12."""
    fr, proofs = collections.defaultdict(list), []
    rx_f = re.compile(r"\[FRICTION\] (\w+)\b.*?used=([0-9.]+)")
    rx_p = re.compile(r"\[NS-PROOF\] (\w+): \w+ \w+ dir=([+-]\d) R2=([-0-9.e+]+) entry=([-0-9.e+]+).*\((\w+), "
                      r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\)")
    files = sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime)[-5:]
    for f in files:
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "[FRICTION]" in line:
                    m = rx_f.search(line)
                    if m:
                        fr[m.group(1)].append(float(m.group(2)))
                elif "[NS-PROOF]" in line:
                    m = rx_p.search(line)
                    if m:
                        proofs.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), pd.Timestamp(m.group(6))))
    return {a: float(np.median(v)) for a, v in fr.items() if v}, sorted(set(proofs), key=lambda x: x[4])


def lot_values():
    """US dollars per 1.0 price move at the smallest lot, read from MT5 (fallback table if MT5 is not reachable),
    and each market's live spread as a fraction of price (used when the log has no [FRICTION] lines for it)."""
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
                out[a] = (units * rate, "MT5 %g lot x %g = %g units, profit in %s" % (si.volume_min, si.trade_contract_size, units, ccy))
        mt5.shutdown()
    except Exception as e:
        say("   MT5 not reachable (%s) -- using the fallback lot table" % e)
    for a in MARKETS:
        out.setdefault(a, (FALLBACK_USD[a], "fallback table"))
    return out, spreads


def run_engine(M, asset, df1, pcfg):
    """One full replay. Returns (the engine's own entries, every setup's life as it ended, setups still alive)."""
    eng = M.NSEngine(asset)
    ends = []
    orig = M.NSEngine._end

    def _end(self, st, s, reason, emit, t):
        ends.append((dict(s), reason, t))
        return orig(self, st, s, reason, emit, t)
    eng._end = types.MethodType(_end, eng)
    out = eng.update(M.NSEngine.new_state(), df1, None, pcfg=pcfg, now=df1.index[-1] + pd.Timedelta(hours=1))
    alive = [dict(s) for s in out["state"]["setups"]]
    return list(out.get("missed") or []) + list(out.get("proofs") or []), ends, alive


def arrays(df1, hourly, M):
    t1 = df1.index + pd.Timedelta(hours=1)
    A = dict(t1=t1, op=df1["open"].values, hi=df1["high"].values, lo=df1["low"].values, cl=df1["close"].values, n=len(df1))
    A["pos"] = dict(zip(t1, range(len(t1))))
    A["kill"] = (lambda i: True) if hourly else (lambda i: t1[i].hour % 4 == 0)
    A["atr"] = M.atr14(A["hi"], A["lo"], A["cl"])          # the same hourly move size the engine uses
    return A


def stop_for(M, cfg, d, e, r2, atr, struct, k):
    """Today's stop (k None) or the structure stop: behind the structure point by k moves; same floor and cap."""
    if k is None:
        return M.ns_levels(d, e, r2, atr, cfg["min_sl_pct"], cfg["target_atr"])
    stop = struct - d * k * atr
    floor = float(cfg["min_sl_pct"] or 0.0) * e
    if abs(e - stop) < floor:
        stop = e - d * floor
    if abs(e - stop) > M.STOP_CAP_ATR * atr:
        stop = e - d * M.STOP_CAP_ATR * atr
    if (d == 1 and stop >= e) or (d == -1 and stop <= e):
        return None, None
    tgt = (e + d * float(cfg["target_atr"]) * atr) if cfg["target_atr"] else None
    return stop, tgt


def gated(M, cfg, s, style, e, atr, stop, strength):
    """The engine's own entry gates (_candidate), applied to every entry with its own stop."""
    d, r2 = s["d"], s["r2"]
    if not (atr == atr and atr > 0) or d * (e - r2) / atr > M.FRESH_ATR:
        return False
    if stop is None or not M.gate_rr_ok(e, stop, atr, cfg["min_rr"]):
        return False
    if cfg.get("cont_only") and s["kind"] != "continuation":
        return False
    if cfg.get("no_spike") and style != "B" and strength > M.SPIKE:
        return False
    return True


def retest_triggers(M, s, A):
    """From the retest touch on: E (close past the peak), CT (close past the retest candle), H (first close the
    trade's way). Stops when the setup dies (a 4H close past R1) or the trigger wait runs out."""
    tt = s.get("t_touch")
    it = A["pos"].get(pd.Timestamp(tt)) if tt is not None else None
    if it is None:
        return {}
    d, r1, r2, edge = s["d"], s["r1"], s["r2"], s["edge"]
    h2 = float(s.get("h2_frozen", s.get("h2")))
    op, hi, lo, cl, t1, n = A["op"], A["hi"], A["lo"], A["cl"], A["t1"], A["n"]
    got, k = {}, it
    if d * (cl[it] - r2) > 0:                    # RC1: the first retest candle closed past the line -- the break held
        got["RC1"] = it
    for i in range(it, n):
        if i > it:
            if A["kill"](i) and ((d == 1 and cl[i] < r1) or (d == -1 and cl[i] > r1)):
                break
            if t1[i] - pd.Timestamp(tt) > M.WAIT_TRIGGER:
                break
        c = cl[i]
        if "H" not in got and d * (c - op[i]) > 0:
            got["H"] = i
        if "RC" not in got and d * (c - r2) > 0 and ((d == 1 and lo[i] <= edge + M.TOUCH_ATR * A["atr"][i]) or
                                                     (d == -1 and hi[i] >= edge - M.TOUCH_ATR * A["atr"][i])):
            got["RC"] = i                        # RC: a retest candle (back at the zone) that closed past the line
        if i > it:
            if "E" not in got and d * (c - h2) > 0:
                got["E"] = i
            if (lo[i] < lo[k]) if d == 1 else (hi[i] > hi[k]):
                k = i
            elif "CT" not in got and d * (c - (hi[k] if d == 1 else lo[k])) > 0:
                got["CT"] = i
        if all(x in got for x in ("H", "E", "CT", "RC")):
            break
    return {st: (i, (float(lo[it:i + 1].min()) if d == 1 else float(hi[it:i + 1].max()))) for st, i in got.items()}


def sim_trade(M, A, d, i, e, stop, target, struct, runner):
    hi, lo, cl, n = A["hi"], A["lo"], A["cl"], A["n"]
    risk0, cur, armed, peak, broke = abs(e - stop), stop, False, None, False
    last = min(n - 1, i + M.TRADE_BARS)
    for j in range(i + 1, last + 1):
        if (d == 1 and lo[j] < struct) or (d == -1 and hi[j] > struct):
            broke = True
        if (d == 1 and lo[j] <= cur) or (d == -1 and hi[j] >= cur):
            return j, cur, "stop", broke, risk0
        if target is not None and ((d == 1 and hi[j] >= target) or (d == -1 and lo[j] <= target)):
            return j, target, "target", broke, risk0
        if runner:
            a0 = max(0, j - 260)
            ns_, armed, peak, _w = M.runner_step(d, e, cur, risk0, hi[a0:j + 1], lo[a0:j + 1], cl[a0:j + 1], j - i, armed, peak)
            if ns_ is not None:
                cur = ns_
    return last, float(cl[last]), "time", broke, risk0


def book(M, A, cands, runner, cost, usd):
    """One position at a time: take each entry only after the previous trade has closed."""
    rows, free_at = [], -1
    for i, e, stop, tgt, struct, d in sorted(cands, key=lambda x: x[0]):
        if i <= free_at:
            continue
        j, px, why, broke, risk0 = sim_trade(M, A, d, i, e, stop, tgt, struct, runner)
        pts = d * (px - e) - cost * e
        rows.append(dict(t=A["t1"][j], money=pts * usd, R=pts / risk0 if risk0 > 0 else 0.0, inside=not broke,
                         noise=(why == "stop" and not broke)))
        free_at = j
    return rows


def stats(rows, t0, t1):
    rows = sorted(rows, key=lambda r: r["t"])
    mid = t0 + (t1 - t0) / 2
    weeks = max((t1 - t0).total_seconds() / 604800.0, 1e-9)
    m = [r["money"] for r in rows]
    eq = np.concatenate([[0.0], np.cumsum(m)]) if m else np.array([0.0])
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    run = worst = 0.0
    for x in m:
        run = run + x if x < 0 else 0.0
        worst = min(worst, run)
    h1 = sum(r["money"] for r in rows if r["t"] < mid)
    h2 = sum(r["money"] for r in rows if r["t"] >= mid)
    losers = [r for r in rows if r["money"] < 0]
    return dict(n=len(rows), win=(100.0 * sum(1 for x in m if x > 0) / len(m)) if m else 0.0, net=sum(m) / weeks,
                lost=sum(-x for x in m if x < 0) / weeks, big=min(m + [0.0]), run=worst, dd=dd, h1=h1, h2=h2,
                inside=(100.0 * sum(1 for r in rows if r["inside"]) / len(rows)) if rows else 0.0,
                noise=sum(1 for r in losers if r["noise"]), losers=len(losers),
                R=(sum(r["R"] for r in rows) / len(rows)) if rows else 0.0, ok=bool(rows) and h1 > 0 and h2 > 0)


def line(name, s):
    return ("%-15s %5d %5.0f%% %+9.2f %9.2f %9.2f %10.2f %8.2f %+10.2f %+10.2f %6.0f%% %+6.2f  %s" %
            (name, s["n"], s["win"], s["net"], s["lost"], s["big"], s["run"], s["dd"], s["h1"], s["h2"], s["inside"],
             s["R"], "PASS" if s["ok"] else "-"))


HEAD = ("variant         trades  win   net$/wk  lost$/wk  biggest$  worstrun$  maxDD$   1st-half$  2nd-half$ inside   R/tr  "
        "(PASS = made money in both halves)")



PIV = {}

NEAR = 0.5            # a bigger 4H level this close ahead (in 4H moves): the push must also close past it
AHEAD_DAYS = 60
MAJOR_K = 6
FAR = 2.5             # an entry further than this from the broken line (in 1H moves) is a far entry
HL_FLOOR = 1.0        # far-entry stop (last 30m higher low): never closer than 1 move ...
HL_CAP = 3.0          # ... and a far entry whose stop would be wider than 3 moves is not taken
ROWS = [  # (name, routing, version, hours, far stop, majority, 1-move floor, lines on wicks, majority on trades at once)
    ("Package A (B13)", "near", "A", 24, "HL", True, True, False, False)]
CEIL_BACK = {"D1": 180, "W1": 3 * 365}
ORDER = ["TODAY", "E", "E skip-near"] + [r[0] for r in ROWS]
FOCUS = ["E", "E skip-near", "Package A (B13)"]


def mt5_frames(sym):
    """30-minute, daily and weekly candles straight from MT5 (count-based: no local-time conversion). Empty when MT5
    is not reachable -- daily/weekly are then built from the hourly file and the package runs on 1H closes only."""
    out = {}
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return out
        mt5.symbol_select(sym, True)
        for name, tf, n in (("M30", mt5.TIMEFRAME_M30, 30000), ("D1", mt5.TIMEFRAME_D1, 2500), ("W1", mt5.TIMEFRAME_W1, 700)):
            r = mt5.copy_rates_from_pos(sym, tf, 0, n)
            if r is None or len(r) == 0:
                continue
            f = pd.DataFrame(r)
            f.index = pd.to_datetime(f["time"], unit="s")
            out[name] = f[["open", "high", "low", "close"]].astype(float)
    except Exception:
        pass
    return out


class TF:
    """One timeframe's candles, with what the tide check needs worked out once."""
    def __init__(self, M, df, hours):
        self.df = df
        self.end_ts = df.index + pd.Timedelta(hours=hours)          # each candle's close time
        self.end = self.end_ts.values
        self.cl, self.hi, self.lo = df["close"].values, df["high"].values, df["low"].values
        self.atr = M.atr14(self.hi, self.lo, self.cl)
        self.ema = {n: pd.Series(self.cl).ewm(span=n, adjust=False).mean().values for n in (20, 50, 200)}

    def last_closed(self, ts):
        return int(np.searchsorted(self.end, np.datetime64(ts), side="right")) - 1

    def barriers(self, d, e, ts):
        """Lines and EMAs of this timeframe sitting just ahead of the trade -- within one of its own moves.
        None = not enough candles to judge."""
        k = self.last_closed(ts)
        if k < 30:
            return None
        a = float(self.atr[k])
        if not (a == a and a > 0):
            return None
        hits = []
        for n in (20, 50, 200):
            if k + 1 >= n:
                v = float(self.ema[n][k])
                if 0 < d * (v - e) <= a:
                    hits.append("EMA%d" % n)
        cl = self.cl
        for x in range(k - 2, max(1, k - 60), -1):               # swing closes confirmed by candle k, newest first
            w = cl[x - 2:x + 3]
            if len(w) == 5 and ((d == 1 and cl[x] == w.max()) or (d == -1 and cl[x] == w.min())) \
                    and 0 < d * (cl[x] - e) <= a:
                hits.append("line")
                break
        return hits


class Levels:
    """The engine's own 4H swing closes, with what the two bigger-level rules need."""
    def __init__(self, M, d4):
        self.end = (d4.index + pd.Timedelta(hours=4)).values
        self.c4 = d4["close"].values
        self.atr4 = M.atr14(d4["high"].values, d4["low"].values, self.c4)
        K, c = M.K, self.c4
        self.sw, self.bylevel = [], collections.defaultdict(list)
        for i in range(K, len(c) - K):
            w = c[i - K:i + K + 1]
            for typ, hit in (("H", c[i] == w.max()), ("L", c[i] == w.min())):
                if hit:
                    edge = float(d4["high"].values[i - K:i + K + 1].max() if typ == "H" else d4["low"].values[i - K:i + K + 1].min())
                    self.sw.append((self.end[i + K], typ, float(c[i]), i, edge))
                    self.bylevel[(typ, round(float(c[i]), 8))].append((pd.Timestamp(self.end[i + K]), i))
        self.conf = np.array([x[0] for x in self.sw])
        # swing highs / lows on the WICKS (the high / low itself, 2 candles either side)
        h4, l4 = d4["high"].values, d4["low"].values
        self.swh = []
        for i in range(K, len(c) - K):
            if h4[i] == h4[i - K:i + K + 1].max():
                self.swh.append((self.end[i + K], "H", float(h4[i]), i))
            if l4[i] == l4[i - K:i + K + 1].min():
                self.swh.append((self.end[i + K], "L", float(l4[i]), i))
        self.confh = np.array([x[0] for x in self.swh])

    def ahead_wick(self, d, e, t):
        """The nearest swing HIGH (a buy) / LOW (a sell) by wick ahead of price, known by time t: (level, distance in 4H moves)."""
        typ = "H" if d == 1 else "L"
        top = int(np.searchsorted(self.confh, np.datetime64(t), side="right"))
        t0 = np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=AHEAD_DAYS))
        a = self.atr_at(t)
        best = None
        for x in range(top - 1, -1, -1):
            cf, ty, lv, _i = self.swh[x]
            if cf < t0:
                break
            if ty == typ and d * (lv - e) > 0 and (best is None or d * (lv - e) < d * (best - e)):
                best = lv
        if best is None or not (a == a and a > 0):
            return None, None
        return best, d * (best - e) / a

    def atr_at(self, t):
        k = int(np.searchsorted(self.end, np.datetime64(t), side="right")) - 1
        return float(self.atr4[k]) if k >= 0 else float("nan")

    def ahead(self, d, e, t):
        """Distance (in 4H moves) to the nearest bigger 4H level ahead of price, known by time t (None if none)."""
        typ = "H" if d == 1 else "L"
        top = int(np.searchsorted(self.conf, np.datetime64(t), side="right"))
        t0 = np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=AHEAD_DAYS))
        best = None
        for x in range(top - 1, -1, -1):
            cf, ty, lv, _i, _e = self.sw[x]
            if cf < t0:
                break
            if ty == typ and d * (lv - e) > 0 and (best is None or d * (lv - e) < best):
                best = d * (lv - e)
        a = self.atr_at(t)
        return None if best is None or not (a == a and a > 0) else best / a

    def ahead_level(self, d, e, t, beyond=0.0):
        """The nearest bigger 4H level ahead of price (known by time t): (level, wick edge, distance in 4H moves)."""
        typ = "H" if d == 1 else "L"
        top = int(np.searchsorted(self.conf, np.datetime64(t), side="right"))
        t0 = np.datetime64(pd.Timestamp(t) - pd.Timedelta(days=AHEAD_DAYS))
        a = self.atr_at(t)
        best = None
        for x in range(top - 1, -1, -1):
            cf, ty, lv, _i, ed = self.sw[x]
            if cf < t0:
                break
            if ty == typ and d * (lv - e) > beyond * (a if a == a else 0) and (best is None or d * (lv - e) < d * (best[0] - e)):
                best = (lv, ed)
        if best is None or not (a == a and a > 0):
            return None, None, None
        return best[0], best[1], d * (best[0] - e) / a

    def major(self, conf, d, r2, t):
        """Is the line a major swing (highest/lowest 4H close for MAJOR_K candles each side), known by time t?
        A setup is born when the opposite swing confirms (its `conf`), so the line's own swing is the latest swing of
        the line's type at exactly that level confirmed no later than that."""
        cands = [x for x in self.bylevel.get(("H" if d == 1 else "L", round(float(r2), 8)), []) if x[0] <= pd.Timestamp(conf)]
        if not cands:
            return None
        i = max(cands, key=lambda x: x[0])[1]
        if i + MAJOR_K >= len(self.c4) or self.end[i + MAJOR_K] > np.datetime64(t):
            return False                                   # not proven major by the time of the signal
        w = self.c4[max(0, i - MAJOR_K):i + MAJOR_K + 1]
        return bool(self.c4[i] == (w.max() if d == 1 else w.min()))


def atr41(M, A, i):
    return float(M.atr14(A["hi"][max(0, i - 40):i + 1], A["lo"][max(0, i - 40):i + 1], A["cl"][max(0, i - 40):i + 1])[-1])


class Bigger:
    """Weekly or daily direction evidence (Desire, 2 Oct): EMA 50/200 (no 20), swing structure (the 4H lines' swing
    rule on this timeframe's closes) and a diagonal line through the last two swing points, valid once a third point
    touched it. The timeframe is WITH the trade when 2 of the 3 agree."""
    def __init__(self, df, step):
        self.end = (df.index + step).values
        self.c, self.h, self.l = df["close"].values, df["high"].values, df["low"].values
        s = pd.Series(self.c)
        self.e50 = s.ewm(span=50, adjust=False).mean().values
        self.e200 = s.ewm(span=200, adjust=False).mean().values
        h, l, c = self.h, self.l, self.c
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        self.atr = pd.Series(np.r_[h[0] - l[0], tr]).rolling(14, min_periods=1).mean().values
        self.sw = []
        for i in range(2, len(c) - 2):
            w = c[i - 2:i + 3]
            if c[i] == w.max():
                self.sw.append((i, float(c[i]), "H"))
            if c[i] == w.min():
                self.sw.append((i, float(c[i]), "L"))

    def read(self, t, d):
        n = int(np.searchsorted(self.end, np.datetime64(pd.Timestamp(t)), side="right")) - 1
        if n < 30:
            return 0
        c = self.c
        last = c[n]
        ema = 1 if (last > self.e50[n] and self.e50[n] > self.e200[n]) else -1 if (last < self.e50[n] and self.e50[n] < self.e200[n]) else 0
        sw = [x for x in self.sw if x[0] + 2 <= n]
        H = [x for x in sw if x[2] == "H"]
        L = [x for x in sw if x[2] == "L"]
        hor = 0
        if len(H) >= 2 and len(L) >= 2:
            up = (H[-1][1] > H[-2][1] and L[-1][1] > L[-2][1]) or last > H[-1][1]
            dn = (H[-1][1] < H[-2][1] and L[-1][1] < L[-2][1]) or last < L[-1][1]
            hor = 1 if up and not dn else -1 if dn and not up else 0
        a = self.atr[n]

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



class Ceilings:
    """E4 (Desire, 2 Oct morning: the barrier check): daily or weekly lines and averages sitting right in the way.
    Swing closes (2 candles either side, highs for a buy and lows for a sell) and the 50/200 EMAs, known by time t."""
    def __init__(self, df, span, back_days):
        self.end = (df.index + span).values
        c = df["close"].astype(float).values
        sr = pd.Series(c)
        self.e50 = sr.ewm(span=50, adjust=False).mean().values
        self.e200 = sr.ewm(span=200, adjust=False).mean().values
        self.back = pd.Timedelta(days=back_days)
        self.sw = []
        for i in range(2, len(c) - 2):
            w = c[i - 2:i + 3]
            if c[i] == w.max():
                self.sw.append((self.end[i + 2], "H", float(c[i])))
            if c[i] == w.min():
                self.sw.append((self.end[i + 2], "L", float(c[i])))
        self.conf = np.array([x[0] for x in self.sw]) if self.sw else np.array([], dtype="datetime64[ns]")

    def ahead(self, d, e, t):
        tt = np.datetime64(pd.Timestamp(t))
        out = []
        k = int(np.searchsorted(self.end, tt, side="right")) - 1
        if k >= 200:
            for v in (self.e50[k], self.e200[k]):
                if d * (v - e) > 0:
                    out.append(float(v))
        top = int(np.searchsorted(self.conf, tt, side="right"))
        t0 = np.datetime64(pd.Timestamp(t) - self.back)
        typ = "H" if d == 1 else "L"
        for x in range(top - 1, -1, -1):
            cf, ty, lv = self.sw[x]
            if cf < t0:
                break
            if ty == typ and d * (lv - e) > 0:
                out.append(lv)
        return out


class Thirty:
    """MT5 30-minute candles (UTC, same clock as the bot's 1H file)."""
    def __init__(self, df):
        self.open_t = df.index.values
        self.t = df.index + pd.Timedelta(minutes=30)
        self.tv = self.t.values
        self.o, self.h, self.l, self.c = (df[x].values for x in ("open", "high", "low", "close"))
        self.e20 = pd.Series(self.c).ewm(span=20, adjust=False).mean().values
        self.n = len(self.c)


def strong4h(F4, t, d):
    k = F4.last_closed(t)
    if k < 50:
        return False
    c, e20, e50 = F4.cl[k], F4.ema[20][k], F4.ema[50][k]
    return (c > e20 and e20 > e50) if d == 1 else (c < e20 and e20 < e50)


def judge(A, T, F4, d, start, r2, r1, trig, level, ver, hours, double):
    """The package on the lower timeframes, checked at every 30m close from `start`. -> (outcome, 30m index, how)"""
    t1v, c1 = A["t1"].values, A["cl"]
    start = pd.Timestamp(start)
    k0 = int(np.searchsorted(T.tv, np.datetime64(start)))
    for k in range(max(3, k0), T.n):
        tau = T.t[k]
        if tau - start > pd.Timedelta(hours=hours):
            return "expired", None, ""
        j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
        if j1 < 1:
            continue
        last1h, t_last1h = c1[j1], A["t1"][j1]
        k4 = F4.last_closed(tau)
        if k4 >= 0 and F4.end_ts[k4] > start and d * (F4.cl[k4] - r1) < 0:
            return "died", None, ""
        p, q = T.c[k - 1], T.c[k]
        if (t_last1h > start and d * (last1h - r2) < 0) or (d * (p - r2) < 0 and d * (q - r2) < 0):
            return "weak", None, ""
        two = (not double) or d * (c1[j1 - 1] - level) > 0
        if d * (last1h - level) > 0 and d * (p - level) > 0 and d * (q - level) > 0 and d * (q - p) >= 0 and two:
            return "enter", k, "strict"
        if ver in ("B", "C", "MID"):
            st4 = tau.floor("4h")
            if st4 == tau:
                st4 = tau - pd.Timedelta(hours=4)
            klo = int(np.searchsorted(T.open_t, np.datetime64(st4)))
            if k - klo >= 1:
                o4, h4, l4 = T.o[klo], T.h[klo:k + 1].max(), T.l[klo:k + 1].min()
                rng = max(h4 - l4, 1e-12)
                s4 = (q > o4 and (q - l4) / rng >= 0.67) if d == 1 else (q < o4 and (h4 - q) / rng >= 0.67)
                ls = T.l[k - 2:k + 1] if d == 1 else T.h[k - 2:k + 1]
                stair = (ls[0] < ls[1] < ls[2]) if d == 1 else (ls[0] > ls[1] > ls[2])
                above = d * (q - T.e20[k]) > 0 and d * (T.e20[k] - T.e20[k - 2]) > 0
                need = r2 if ver == "C" else trig
                ok1 = d * (last1h - need) > 0 and ((not double) or d * (c1[j1 - 1] - need) > 0)
                if ver == "MID":
                    ok1 = ok1 and d * (q - level) > 0      # middle way: only once past the wall
                if s4 and stair and above and ok1:
                    return "enter", k, "re-read"
    return "watching", None, ""


def place_stop(M, T, cfg, d, k, q, r2, t_touch, atr, far_stop, floor1=False):
    """0.3 of a 1H move behind the push's extreme since the retest; a far entry may use the last 30m higher low
    (never closer than HL_FLOOR moves; not taken if wider than HL_CAP moves)."""
    k_lo = int(np.searchsorted(T.open_t, np.datetime64(pd.Timestamp(t_touch) - pd.Timedelta(hours=1))))
    seg = T.l[k_lo:k + 1] if d == 1 else T.h[k_lo:k + 1]
    if len(seg) == 0:
        return None, "no structure"
    stop = (seg.min() if d == 1 else seg.max()) - d * 0.3 * atr
    if d * (q - r2) / atr > FAR and far_stop == "HL":
        piv = [seg[i] for i in range(1, len(seg) - 1)
               if ((seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]) if d == 1 else (seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]))]
        if piv:
            stop = piv[-1] - d * 0.3 * atr
        if d * (q - stop) < HL_FLOOR * atr:
            stop = q - d * HL_FLOOR * atr
        if d * (q - stop) > HL_CAP * atr:
            return None, "far entry, stop wider than 3 moves"
    if floor1 and d * (q - stop) < HL_FLOOR * atr:
        stop = q - d * HL_FLOOR * atr                 # breathing space: never closer than 1 move (live since B13)
    floor = float(cfg.get("min_sl_pct") or 0.0) * q
    if d * (q - stop) < floor:
        stop = q - d * floor
    if d * (q - stop) > M.STOP_CAP_ATR * atr:
        stop = q - d * M.STOP_CAP_ATR * atr
    if d * (q - stop) <= 0:
        return None, "stop on the wrong side"
    return stop, ""


def sim30(M, A, T, d, k, q, stop, tgt, runner):
    """One trade on the 30m candles: stop, target, the 7-day limit; BTC's runner trails on each 1H close."""
    risk0, cur, armed, peak = abs(q - stop), stop, False, None
    t0 = T.t[k]
    for kk in range(k + 1, T.n):
        hi, lo = T.h[kk], T.l[kk]
        if (d == 1 and lo <= cur) or (d == -1 and hi >= cur):
            return kk, cur, risk0
        if tgt is not None and ((d == 1 and hi >= tgt) or (d == -1 and lo <= tgt)):
            return kk, tgt, risk0
        tau = T.t[kk]
        hrs = (tau - t0).total_seconds() / 3600.0
        if hrs >= M.TRADE_BARS:
            return kk, T.c[kk], risk0
        if runner and tau.minute == 0:
            j = A["pos"].get(tau)
            if j is not None:
                a0 = max(0, j - 260)
                ns_, armed, peak, _w = M.runner_step(d, q, cur, risk0, A["hi"][a0:j + 1], A["lo"][a0:j + 1], A["cl"][a0:j + 1],
                                                     int(hrs), armed, peak)
                if ns_ is not None:
                    cur = ns_
    return T.n - 1, T.c[-1], risk0


def book30(M, A, T, cands, runner, cost, usd):
    """One position per market at a time, priced on the 30m candles."""
    rows, free = [], -1
    for c in sorted(cands, key=lambda x: x["k"]):
        if c["k"] <= free:
            continue
        kk, px, risk0 = sim30(M, A, T, c["d"], c["k"], c["q"], c["stop"], c["tgt"], runner)
        pts = c["d"] * (px - c["q"]) - cost * c["q"]
        rows.append(dict(t=T.t[kk], money=pts * usd, R=pts / risk0 if risk0 > 0 else 0.0, inside=False, noise=False,
                         delay=c.get("delay"), give=c.get("give")))
        free = kk
    return rows


def run(data, frames, pcfg, costs, lots, live_proofs):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nTHE LIVE PACKAGE (B13-PKG) -- every market, judged on the real 30m and 1H closes\n" + "=" * 110)
    allrows, spans, per = collections.defaultdict(list), [], {}
    tally = collections.defaultdict(collections.Counter)
    e_match = e_eng = e_mine = 0
    for a, df1 in data.items():
        t_a = time.time()
        cfg = M.market_settings(a, pcfg)
        if cfg is None:
            continue
        fr = frames.get(a, {})
        if fr.get("M30") is None or len(fr["M30"]) < 1000:
            say("   %-7s no 30m candles from MT5 -- the package can't be judged here; market skipped" % a)
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd = costs.get(a, 0.0005), lots[a][0]
        runner = cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        live, _e, _a = run_engine(M, a, df1, pcfg)
        engE, ends, alive = run_engine(M, a, df1, pE)
        cfgE = M.market_settings(a, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        CD, CW = Ceilings(D1, pd.Timedelta(days=1), CEIL_BACK["D1"]), Ceilings(W1, pd.Timedelta(days=7), CEIL_BACK["W1"])

        def dw_ceil(d0, e0, t0):
            """Daily/weekly ceilings within half a 4H move ahead of price."""
            a4 = L.atr_at(t0)
            if not (a4 == a4 and a4 > 0):
                return []
            return [v for v in CD.ahead(d0, e0, t0) + CW.ahead(d0, e0, t0) if d0 * (v - e0) <= NEAR * a4]
        V = collections.defaultdict(list)
        eng_by = {int(p["fields"]["ns_setup_id"]): p["fields"] for p in engE}

        def at_once(f):
            tc = pd.Timestamp(f["ns_candle"])
            k = int(np.searchsorted(T.tv, np.datetime64(tc)))
            if k >= T.n or T.t[k] != tc or f.get("ns_stop") is None:
                return None
            return dict(k=k, q=float(f["ns_close"]), stop=float(f["ns_stop"]), tgt=f.get("ns_target"), d=int(f["setup_dir"]))
        for name, fl in (("TODAY", [p["fields"] for p in live]), ("E", [p["fields"] for p in engE])):
            for f in fl:
                c = at_once(f)
                if c is None:
                    tally[name]["no 30m candle at the entry"] += 1
                else:
                    V[name].append(c)
        for f in (p["fields"] for p in engE):
            tc, d0, e0 = pd.Timestamp(f["ns_candle"]), int(f["setup_dir"]), float(f["ns_close"])
            lvl, _ed, near = L.ahead_level(d0, e0, tc)
            n4 = lvl is not None and near is not None and near <= NEAR
            wl0, wn0 = L.ahead_wick(d0, e0, tc)
            nw = wl0 is not None and wn0 is not None and wn0 <= NEAR
            c = at_once(f)
            for nm, skip in (("E skip-near", n4), ("E skip wicks", nw)):
                if skip:
                    tally[nm]["skipped (a 4H line just ahead%s)" % (", by wicks" if nm != "E skip-near" else "")] += 1
                elif c is not None:
                    V[nm].append(c)
                    tally[nm]["taken at once"] += 1
        mineE = {}
        t1v = A["t1"].values
        for sid, s in setups.items():
            d = s["d"]
            trig = retest_triggers(M, s, A)
            if "E" in trig:
                i, struct = trig["E"]
                e = float(A["cl"][i])
                atr = atr41(M, A, i)
                strength = M.candle_strength(d, A["hi"][i], A["lo"][i], A["cl"][i])
                stopE, _t = stop_for(M, cfgE, d, e, s["r2"], atr, struct, None)
                if gated(M, cfgE, s, "E", e, atr, stopE, strength):
                    mineE[sid] = i
            if "E" not in trig or (cfg.get("cont_only") and s["kind"] != "continuation"):
                continue
            iE = trig["E"][0]
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tt = s.get("t_touch")
            if tt is None:
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            trg = float(s.get("h2_frozen", s.get("h2")))
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            lvl, _ed, near = L.ahead_level(d, eE, tE)
            is_near = lvl is not None and near is not None and near <= NEAR
            level = (max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg
            f = eng_by.get(sid)
            took = f is not None and A["pos"].get(pd.Timestamp(f["ns_candle"])) == iE
            aE = atr41(M, A, iE)
            far_ref = (not took) and aE == aE and aE > 0 and d * (eE - r2) / aE > M.FRESH_ATR
            nmaj = None
            wl, wn = L.ahead_wick(d, eE, tE)
            near_w = wl is not None and wn is not None and wn <= NEAR
            level_w = (max(trg, wl) if d == 1 else min(trg, wl)) if near_w else trg
            for name, scope, ver, hrs, fstop, maj, floor1, wick, majall in ROWS:
                nr, lev = (near_w, level_w) if wick else (is_near, level)
                tally[name]["signals"] += 1
                if scope == "near" and not nr and not far_ref:
                    c = at_once(f) if took else None
                    if c is None:
                        tally[name]["refused by the engine"] += 1
                        continue
                    if majall:
                        if nmaj is None:
                            nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                        if nmaj < 2:
                            tally[name]["space to run, but no bigger-timeframe majority -- not taken"] += 1
                            continue
                    V[name].append(c)
                    tally[name]["taken at once (space to run)"] += 1
                    continue
                tally[name]["judged by the package"] += 1
                if maj:
                    if nmaj is None:
                        nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                    if nmaj < 2:
                        tally[name]["no bigger-timeframe majority"] += 1
                        continue
                out, k, how = judge(A, T, F4, d, tE, r2, r1, trg, lev, ver, hrs, False)
                if out != "enter":
                    tally[name][{"weak": "cancelled (weak)", "died": "cancelled (setup died)", "expired": "expired (still dithering)",
                                 "watching": "still watching at the end"}[out]] += 1
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr = atr41(M, A, j1)
                if not (atr == atr and atr > 0):
                    tally[name]["no move size"] += 1
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr, fstop, floor1)
                if stop is None:
                    tally[name]["not taken: " + why] += 1
                    continue
                if not M.gate_rr_ok(q, stop, atr, cfg["min_rr"]):
                    tally[name]["not taken: reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr
                far = d * (q - r2) / atr > FAR
                tally[name]["entered (%s%s)" % (how, ", far" if far else "")] += 1
                V[name].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d, delay=(tau - tE).total_seconds() / 3600.0,
                                    give=d * (q - eE) / atr))
        eng_ids = {int(p["fields"]["ns_setup_id"]): A["pos"].get(pd.Timestamp(p["fields"]["ns_candle"])) for p in engE}
        e_eng += len(eng_ids)
        e_mine += len(mineE)
        e_match += sum(1 for sid, i in eng_ids.items() if mineE.get(sid) == i)
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        say("   %-7s replayed in %.0f s | today %d | E %d | package A %d | A wicks + maj %d | 30m candles from %s" % (
            a, time.time() - t_a, len(V.get("TODAY", [])), len(V.get("E", [])), len(V.get("Package A (B13)", [])),
            0, str(T.t[0])[:10]))
    if not spans:
        return
    T0_, T1_ = min(s[0] for s in spans), max(s[1] for s in spans)
    say("\n   CHECK 1 -- my close-through entries vs the engine's own: %d of %d engine entries matched exactly (mine: %d)%s" % (
        e_match, e_eng, e_mine, "" if e_match == e_eng == e_mine else "   <-- MISMATCH: results below don't count -- tell Claude"))
    say("\n   ALL MARKETS TOGETHER (one position per market at a time; every row priced on the 30m candles):")
    say("   " + HEAD)
    tab = {v: stats(r, T0_, T1_) for v, r in allrows.items()}
    for v in ORDER:
        say("   " + (line(v, tab[v]) if v in tab else "%-15s (no trades)" % v))
    say("\n   SHARPNESS (package trades only): average hours from the E signal to the entry, and the price given up vs the E close"
        " in 1H moves")
    for v in ORDER[3:]:
        r = [x for x in allrows.get(v, []) if x.get("delay") is not None]
        if r:
            say("   %-20s %5d trades | wait %5.1f h | gave up %+5.2f moves" % (v, len(r), np.mean([x["delay"] for x in r]),
                                                                         np.mean([x["give"] for x in r])))
    say("\n   WHAT THE PACKAGE DECIDED (all markets):")
    for v in ORDER[2:]:
        c = tally[v]
        say("   %-20s %s" % (v, " | ".join("%s %d" % (k, n) for k, n in sorted(c.items(), key=lambda kv: -kv[1]))))
    say("\n   EACH MARKET:")
    for a in per:
        say("   " + a)
        for v in ["TODAY"] + FOCUS:
            say("     " + (line(v, per[a][v]) if v in per[a] else "%-15s (no trades)" % v))



# =====================================================================================================================
# D2 (Desire 5 Oct): THE LIVERMORE TIMEFRAME TEST -- which timeframe's Livermore reading fits best: 1H, 4H, 1D or 1W.
# The bot's own Livermore code and each market's own settings, replayed candle by candle on each timeframe.
# =====================================================================================================================
UP_STATES = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
LSM_TFS = ("1H", "4H", "1D", "1W")


def lsm_signs(a, df, span):
    """(close times, +1 in an up state / -1 in a down state / 0 unknown) for every closed candle of this timeframe."""
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())             # run from C:\\TradingBot\\TBOT, so the bot's own code can be read
    from src.execution.livermore_state_machine import LivermoreStateMachine, atr14 as lsm_atr
    p = (PIV or {}).get(a, {}) or {}
    lsm = LivermoreStateMachine(asset=a, timeframe="TEST", major_mult=p.get("major_mult", 3.5),
                                minor_mult=p.get("minor_mult", 1.0), dual_confirm=p.get("dual_confirm", 2),
                                atr_period=p.get("atr_period", 14))
    d = df[["high", "low", "close"]].astype(float)
    atr = lsm_atr(d, int(p.get("atr_period", 14))).values
    c = d["close"].values
    sg = np.zeros(len(c))
    for i in range(len(c)):
        lsm.update(float(c[i]), float(atr[i]))
        st = str(lsm.state() if callable(getattr(lsm, "state", None)) else getattr(lsm, "state", ""))
        sg[i] = 1 if st in UP_STATES else (-1 if st else 0)
    return (df.index + span).values, sg


def lsm_frames(M, a, df1, fr):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    d1 = fr.get("D1") if fr.get("D1") is not None else df1.resample("1D").agg(agg).dropna()
    w1 = fr.get("W1") if fr.get("W1") is not None else df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
    return {"1H": (df1, pd.Timedelta(hours=1)), "4H": (M.hourly_to_4h(df1), pd.Timedelta(hours=4)),
            "1D": (d1, pd.Timedelta(days=1)), "1W": (w1, pd.Timedelta(days=7))}


def sign_at(tt, sg, t):
    k = int(np.searchsorted(tt, np.datetime64(pd.Timestamp(t)), side="right")) - 1
    return sg[k] if k >= 0 else 0


def book_tag(M, A, T, cands, runner, cost, usd):
    """book30, keeping each trade's own details (entry time, direction, tags)."""
    out, free = [], -1
    for c in sorted(cands, key=lambda x: x["k"]):
        if c["k"] <= free:
            continue
        kk, px, risk0 = sim30(M, A, T, c["d"], c["k"], c["q"], c["stop"], c["tgt"], runner)
        pts = c["d"] * (px - c["q"]) - cost * c["q"]
        out.append(dict(c, t=T.t[kk], t_in=T.t[c["k"]], money=pts * usd, R=pts / risk0 if risk0 > 0 else 0.0,
                        inside=False, noise=False))
        free = kk
    return out


def e_cands(M, a, df1, pcfg, T):
    pE = json.loads(json.dumps(pcfg or {}, default=str))
    pE.setdefault("ns_markets", {}).setdefault(a, {})
    pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
    engE, _e, _a = run_engine(M, a, df1, pE)
    out = []
    for p in engE:
        f = p["fields"]
        tc = pd.Timestamp(f["ns_candle"])
        k = int(np.searchsorted(T.tv, np.datetime64(tc)))
        if k < T.n and T.t[k] == tc and f.get("ns_stop") is not None:
            out.append(dict(k=k, q=float(f["ns_close"]), stop=float(f["ns_stop"]), tgt=f.get("ns_target"), d=int(f["setup_dir"])))
    return out


def run_lsm_test(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nD2 -- THE LIVERMORE TIMEFRAME TEST: which timeframe's Livermore reading fits best (the bot's own brain code"
        " and settings, replayed on 1H, 4H, daily and weekly candles)\n" + "=" * 110)
    dirn = {tf: [0, 0, 0.0] for tf in LSM_TFS}                 # readings, right, sum of signed moves
    trades = {tf: {1: [], -1: []} for tf in LSM_TFS}           # agree / against -> R list
    permk = {}
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        sig = {tf: lsm_signs(a, f, sp) for tf, (f, sp) in lsm_frames(M, a, df1, fr).items()}
        # A. direction: at every 4H close, the state against the next 24 hours of price, in 4H moves
        d4 = M.hourly_to_4h(df1)
        t4 = (d4.index + pd.Timedelta(hours=4)).values
        a4 = M.atr14(d4["high"].values, d4["low"].values, d4["close"].values)
        t1v, c1 = A["t1"].values, A["cl"]
        pm = {tf: [0, 0] for tf in LSM_TFS}
        for i in range(60, len(t4)):
            j0 = int(np.searchsorted(t1v, t4[i], side="right")) - 1
            j1 = int(np.searchsorted(t1v, t4[i] + np.timedelta64(24, "h"), side="right")) - 1
            if j0 < 0 or j1 <= j0 or pd.Timestamp(t1v[j1]) - pd.Timestamp(t4[i]) < pd.Timedelta(hours=20):
                continue
            if not (a4[i] == a4[i] and a4[i] > 0):
                continue
            mv = (c1[j1] - c1[j0]) / a4[i]
            for tf in LSM_TFS:
                s = sign_at(sig[tf][0], sig[tf][1], pd.Timestamp(t4[i]))
                if s == 0:
                    continue
                dirn[tf][0] += 1
                dirn[tf][1] += int(s * mv > 0)
                dirn[tf][2] += s * mv
                pm[tf][0] += 1
                pm[tf][1] += int(s * mv > 0)
        # B. trades: every E trade, split by whether each timeframe agreed with it at the entry
        rows = book_tag(M, A, T, e_cands(M, a, df1, pcfg, T), runner, cost, usd)
        for r in rows:
            for tf in LSM_TFS:
                s = sign_at(sig[tf][0], sig[tf][1], r["t_in"])
                if s != 0:
                    trades[tf][1 if s == r["d"] else -1].append(r["R"])
        permk[a] = (pm, rows, sig)
        say("   %-7s %d E trades | readings per timeframe: %s" % (a, len(rows), " ".join("%s %d" % (tf, pm[tf][0]) for tf in LSM_TFS)))
    say("\n   A. DIRECTION -- at every 4H close: did each timeframe's Livermore state point the way price went over the next 24 hours?")
    say("   timeframe   readings   right   average move WITH the state (4H moves)")
    for tf in LSM_TFS:
        n, ok, sm = dirn[tf]
        say("   %-9s %9d   %4.1f%%   %+8.3f" % (tf, n, 100.0 * ok / n if n else 0.0, sm / n if n else 0.0))
    say("\n   B. TRADES -- every E trade (the live entry), split by whether each timeframe's state AGREED with the trade at entry")
    say("   timeframe   agreed: trades  win  R/trade   |   against: trades  win  R/trade   |   difference R/trade")
    for tf in LSM_TFS:
        ag, ot = trades[tf][1], trades[tf][-1]
        f = lambda xs: (len(xs), 100.0 * sum(1 for x in xs if x > 0) / len(xs) if xs else 0.0, sum(xs) / len(xs) if xs else 0.0)
        g, o = f(ag), f(ot)
        say("   %-9s %15d %4.0f%% %+8.2f   | %16d %4.0f%% %+8.2f   | %+10.2f" % (tf, g[0], g[1], g[2], o[0], o[1], o[2], g[2] - o[2]))
    say("\n   EACH MARKET -- direction right% per timeframe (1H / 4H / 1D / 1W), and trades agreeing per timeframe (n, R/trade)")
    for a, (pm, rows, sig) in permk.items():
        parts = []
        for tf in LSM_TFS:
            ag = [r["R"] for r in rows if sign_at(sig[tf][0], sig[tf][1], r["t_in"]) == r["d"]]
            parts.append("%s %4.1f%% | %3d %+5.2f" % (tf, 100.0 * pm[tf][1] / pm[tf][0] if pm[tf][0] else 0.0, len(ag),
                                                   sum(ag) / len(ag) if ag else 0.0))
        say("   %-7s %s" % (a, "   ".join(parts)))
    say("\n   READ: the best timeframe points the right way most often (A) AND separates winning from losing trades (B: a big"
        " positive difference). If no timeframe does both clearly, none should be used globally yet.")


# =====================================================================================================================
# D1 (Desire 1 and 5 Oct): THE DIAGONAL-LINES TEST -- diagonal lines mirror the horizontal lines' rules; a break on a
# diagonal alone may trade. Lines through two 4H swing closes (two lower highs for a buy line, two higher lows for a
# sell line), valid once a third point touches (within half a 4H move) -- the same 3-touch rule the package reads.
# Break: a 4H close past the line by 0.25 of a 4H move (big), or a smaller close past it held by the next 1H close.
# Retest: a later 4H candle back to the line (within 0.25 of a move) without a close past the origin; a close back
# through the line needs a later 4H close past it again (the recovery). Entry: the first 1H close past the highest
# (lowest, for sells) close since the break. Stop and target: the engine's own (behind the line at the retest).
# =====================================================================================================================
DIAG_LIFE = 60          # 4H candles a line is watched for a break
DIAG_WAIT = 30          # 4H candles after the break that a setup may take to retest and trigger
DIAG_WITH_H = 12        # hours: a diagonal entry within this of a horizontal E entry, same direction, is "with horizontal"


def diag_setups(M, a, df1, A, cfg):
    d4 = M.hourly_to_4h(df1)
    t4 = (d4.index + pd.Timedelta(hours=4))
    hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
    a4 = M.atr14(hi4, lo4, c4)
    n4, KK = len(c4), M.K
    t1v, c1, atr1 = A["t1"].values, A["cl"], A["atr"]
    piv = {"H": [], "L": []}
    for i in range(KK, n4 - KK):
        w = c4[i - KK:i + KK + 1]
        if c4[i] == w.max():
            piv["H"].append(i)
        if c4[i] == w.min():
            piv["L"].append(i)
    out, seen = [], set()
    for d, typ in ((1, "H"), (-1, "L")):
        P = piv[typ]
        for j in range(1, len(P)):
            i1, i2 = P[j - 1], P[j]
            v1, v2 = c4[i1], c4[i2]
            if i2 == i1 or (d == 1 and not v2 < v1) or (d == -1 and not v2 > v1):
                continue
            slope = (v2 - v1) / (i2 - i1)
            L = lambda x: v1 + slope * (x - i1)
            nxt = P[j + 1] if j + 1 < len(P) else n4      # a newer pivot of the same kind replaces this line
            start = i2 + KK                                # the second pivot is confirmed here
            earlier = P[max(0, j - 5):j - 1]
            valid = any(abs(c4[p] - L(p)) <= 0.5 * a4[p] for p in earlier if a4[p] == a4[p])
            brk = None
            for x in range(start, min(n4, start + DIAG_LIFE, nxt + KK)):
                if not valid and x >= i2 + 3 and a4[x] == a4[x] and abs(c4[x] - L(x)) <= 0.5 * a4[x]:
                    valid = True
                    continue
                if not valid or not (a4[x] == a4[x] and a4[x] > 0):
                    continue
                past = d * (c4[x] - L(x))
                if past >= 0.25 * a4[x]:
                    brk = x
                    break
                if past > 0:
                    jn = int(np.searchsorted(t1v, t4[x].to_datetime64(), side="right"))   # the next 1H close
                    if jn < len(c1) and d * (c1[jn] - L(x)) > 0:
                        brk = x
                        break
            if brk is None:
                continue
            origin = (c4[i2:brk + 1].min() if d == 1 else c4[i2:brk + 1].max())
            ret, peak = None, c4[brk]
            pend = False
            for y in range(brk + 1, min(n4, brk + 1 + DIAG_WAIT)):
                if d * (c4[y] - origin) < 0:
                    break                                   # the setup died
                peak = max(peak, c4[y]) if d == 1 else min(peak, c4[y])
                lv = L(y)
                touch = (lo4[y] <= lv + 0.25 * a4[y]) if d == 1 else (hi4[y] >= lv - 0.25 * a4[y])
                if pend and d * (c4[y] - lv) >= 0:
                    ret = y
                    break
                if touch:
                    if d * (c4[y] - lv) >= 0:
                        ret = y
                        break
                    pend = True                              # closed back through the line: needs a recovery close
            if ret is None:
                continue
            peak = (c4[brk:ret + 1].max() if d == 1 else c4[brk:ret + 1].min())
            r2 = L(ret)
            tr = t4[ret]
            j0 = int(np.searchsorted(t1v, tr.to_datetime64(), side="right"))
            tend = t4[min(n4 - 1, brk + DIAG_WAIT)]
            for jj in range(j0, len(c1)):
                if pd.Timestamp(t1v[jj]) > tend:
                    break
                k4 = int(np.searchsorted(t4.values, t1v[jj], side="right")) - 1
                if k4 >= 0 and d * (c4[k4] - origin) < 0:
                    break
                if d * (c1[jj] - peak) > 0:
                    e, at = float(c1[jj]), float(atr1[jj])
                    if not (at == at and at > 0) or d * (e - r2) / at > M.FRESH_ATR:
                        break
                    stop, tgt = M.ns_levels(d, e, r2, at, cfg["min_sl_pct"], cfg["target_atr"])
                    if stop is None or not M.gate_rr_ok(e, stop, at, cfg["min_rr"]):
                        break
                    key = (d, pd.Timestamp(t1v[jj]))
                    if key not in seen:
                        seen.add(key)
                        out.append(dict(d=d, t=pd.Timestamp(t1v[jj]), e=e, stop=float(stop), tgt=tgt))
                    break
    return out


def run_diag_test(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nD1 -- THE DIAGONAL-LINES TEST: breaks of 3-touch diagonal lines, traded break -> retest -> 1H close-through,"
        " like the horizontal lines\n" + "=" * 110)
    names = ["E (horizontal, live)", "DIAG all", "DIAG alone", "DIAG with horizontal", "E + DIAG alone"]
    allrows, spans = collections.defaultdict(list), []
    per = {}
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        ec = e_cands(M, a, df1, pcfg, T)
        e_times = [(c["d"], T.t[c["k"]]) for c in ec]
        dc = []
        for s in diag_setups(M, a, df1, A, cfg):
            k = int(np.searchsorted(T.tv, np.datetime64(s["t"])))
            if k >= T.n or T.t[k] != s["t"]:
                continue
            withh = any(dd == s["d"] and abs((tt - s["t"]).total_seconds()) <= DIAG_WITH_H * 3600 for dd, tt in e_times)
            dc.append(dict(k=k, q=s["e"], stop=s["stop"], tgt=s["tgt"], d=s["d"], withh=withh))
        V = {"E (horizontal, live)": ec, "DIAG all": dc, "DIAG alone": [c for c in dc if not c["withh"]],
             "DIAG with horizontal": [c for c in dc if c["withh"]], "E + DIAG alone": ec + [c for c in dc if not c["withh"]]}
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v in names:
            rows = book_tag(M, A, T, V[v], runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        say("   %-7s diagonal entries %d (alone %d, with a horizontal E %d) | horizontal E entries %d" % (
            a, len(dc), sum(1 for c in dc if not c["withh"]), sum(1 for c in dc if c["withh"]), len(ec)))
    if not spans:
        say("   no market had 30m candles -- nothing to report")
        return
    T0_, T1_ = min(s[0] for s in spans), max(s[1] for s in spans)
    say("\n   ALL MARKETS TOGETHER (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in names:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   EACH MARKET:")
    for a, pv in per.items():
        say("   %s" % a)
        for v in names:
            say("     " + line(v[:15], pv[v]))
    say("\n   READ: 'DIAG alone' is what your 1 Oct ruling adds (a break only on a diagonal line). It should only go live if it"
        " makes money in both halves and 'E + DIAG alone' beats 'E' without a much bigger drop.")



# =====================================================================================================================
# THE SILVER LONG OF 5 OCT 2026, REBUILT: the engine's own proof, its setup, the 4H candles of the break, every line
# ahead (on 4H closes, as the bot judges, and on wicks, as the eye reads the chart), the bigger-timeframe votes, what
# package A would have done had the trade been handed over, and the live trade's path to its stop.
# =====================================================================================================================
CASE = dict(asset="SILVER", d=1, r2=61.278, day="2026-10-05", fill=61.709, stop=61.152, target=63.39,
            filled="2026-10-05 10:09")


def trade_case(data, frames, pcfg):
    a, d = CASE["asset"], CASE["d"]
    say("\n" + "=" * 110 + "\nTHE SILVER LONG OF 5 OCT, REBUILT WITH THE BOT'S OWN CODE AND CANDLES (all times MT5 server time, as the bot's candles)\n" + "=" * 110)
    if a not in data:
        say("   %s: no candles -- the trade can't be rebuilt" % a)
        return
    M = load_engine("4H/1H", False)
    df1, fr = data[a], frames.get(a, {})
    cfg = M.market_settings(a, pcfg)
    A = arrays(df1, False, M)
    pE = json.loads(json.dumps(pcfg or {}, default=str))
    pE.setdefault("ns_markets", {}).setdefault(a, {})
    pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
    engE, ends, alive = run_engine(M, a, df1, pE)
    setups = {s["id"]: s for s, _r, _t in ends}
    for s in alive:
        setups[s["id"]] = s
    day0 = pd.Timestamp(CASE["day"])
    near_day = [p["fields"] for p in engE if day0 - pd.Timedelta(days=3) <= pd.Timestamp(p["fields"]["ns_candle"]) <= day0 + pd.Timedelta(days=1)]
    hit = [f for f in near_day if int(f["setup_dir"]) == d and abs(float(f["ns_r2"]) - CASE["r2"]) < 0.02]
    say("   the engine's SILVER proofs of 2-5 Oct: " + (" | ".join("%s dir %+d line %.3f close %.3f" % (
        f["ns_candle"], int(f["setup_dir"]), float(f["ns_r2"]), float(f["ns_close"])) for f in near_day) or "none"))
    if not hit:
        say("   no proof at line %.3f -- the rebuild stops here (tell Claude)" % CASE["r2"])
        return
    f = hit[-1]
    sid = int(f["ns_setup_id"])
    s = setups.get(sid, {})
    tE, eE, r2 = pd.Timestamp(f["ns_candle"]), float(f["ns_close"]), float(f["ns_r2"])
    say("\n   1. THE PROOF: 1H candle %s closed %.3f | the line (R2) %.3f | stop %s | target %s | setup %d" % (
        tE, eE, r2, f.get("ns_stop"), f.get("ns_target"), sid))
    say("      the setup as the engine recorded it:")
    for k in sorted(s):
        v = s[k]
        if isinstance(v, (int, float, str, bool, type(None), pd.Timestamp, np.floating, np.integer)):
            say("        %-16s %s" % (k, v))
    d4 = M.hourly_to_4h(df1)
    t4c = d4.index + pd.Timedelta(hours=4)
    h4, l4, c4, o4 = (d4[x].astype(float).values for x in ("high", "low", "close", "open"))
    a4 = M.atr14(h4, l4, c4)
    say("\n   2. THE 4H CANDLES, 28 Sep - 5 Oct (close time): open / high / low / close | close past the line, in 4H moves")
    for i in range(len(d4)):
        if day0 - pd.Timedelta(days=7) <= t4c[i] <= day0 + pd.Timedelta(hours=16):
            say("      %s  %.3f  %.3f  %.3f  %.3f | %+.2f%s" % (t4c[i], o4[i], h4[i], l4[i], c4[i], d * (c4[i] - r2) / a4[i],
                "   <- the proof's hour is inside this candle" if t4c[i] - pd.Timedelta(hours=4) < tE <= t4c[i] else ""))
    L = Levels(M, d4)
    aE = L.atr_at(tE)
    say("\n   3. LINES AHEAD AT THE PROOF (close %.3f; a 4H move = %.3f, so 'just ahead' = within %.3f, up to %.3f)" % (
        eE, aE, NEAR * aE, eE + d * NEAR * aE))
    typ = "H" if d == 1 else "L"
    for lab, rows in (("on 4H CLOSES (how the bot judges)", [(lv, cf) for cf, ty, lv, _i, _ed in L.sw if ty == typ]),
                      ("on 4H WICKS (how the eye reads the chart)", [(lv, cf) for cf, ty, lv, _i in L.swh if ty == typ])):
        rr = sorted({(round(lv, 3), pd.Timestamp(cf)) for lv, cf in rows
                     if pd.Timestamp(cf) <= tE and pd.Timestamp(cf) >= tE - pd.Timedelta(days=AHEAD_DAYS)
                     and d * (lv - eE) > -1.5 * aE and d * (lv - eE) < 6 * aE}, key=lambda x: d * x[0])
        say("      swing lines %s, from -1.5 to +6 4H moves around the proof:" % lab)
        for lv, cf in rr:
            dist = d * (lv - eE) / aE
            say("        %.3f  confirmed %s  %+.2f moves%s" % (lv, cf, dist, "   <- JUST AHEAD" if 0 < dist <= NEAR else
                                                                  ("   (ahead)" if dist > 0 else "   (already passed)")))
    lc, _ec, nc = L.ahead_level(d, eE, tE)
    lw, nw = L.ahead_wick(d, eE, tE)
    say("      VERDICT on closes (the bot): nearest line ahead %s -> %s" % (
        ("%.3f, %.2f moves" % (lc, nc)) if lc is not None else "none",
        "the PACKAGE" if (lc is not None and nc <= NEAR) else "taken at once (space to run)"))
    say("      VERDICT on wicks:            nearest high ahead %s -> %s" % (
        ("%.3f, %.2f moves" % (lw, nw)) if lw is not None else "none",
        "the PACKAGE" if (lw is not None and nw <= NEAR) else "taken at once (space to run)"))
    F4 = TF(M, d4, 4)
    D1, W1 = fr.get("D1"), fr.get("W1")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if D1 is None:
        D1 = df1.resample("1D").agg(agg).dropna()
    if W1 is None:
        W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
    BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
    vw, vd, v4 = BW.read(tE, d), BD.read(tE, d), strong4h(F4, tE, d)
    nm = (vw == 1) + (vd == 1) + int(bool(v4))
    word = {1: "WITH the buy", -1: "AGAINST", 0: "mixed"}
    say("\n   4. BIGGER TIMEFRAMES AT THE PROOF (the package's 2-of-3 check): weekly %s | daily %s | 4H %s -> %d of 3%s" % (
        word.get(vw, vw), word.get(vd, vd), "strong WITH" if v4 else "not strong", nm,
        "" if nm >= 2 else "  -> the package would have said NO MAJORITY (no trade)"))
    if fr.get("M30") is not None and len(fr["M30"]) > 100:
        T = Thirty(fr["M30"])
        trg = float(s.get("h2_frozen", s.get("h2", eE)))
        r1 = float(s.get("r1", r2))
        say("\n   5. IF IT HAD BEEN HANDED TO PACKAGE A (trigger %.3f):" % trg)
        for lab, lv in (("closes", lc if (lc is not None and nc <= NEAR) else None),
                        ("wicks", lw if (lw is not None and nw <= NEAR) else None)):
            if lv is None:
                say("      on %s: not handed over (no line just ahead)" % lab)
                continue
            if nm < 2:
                say("      on %s: line %.3f -- stopped at the majority check (no trade)" % (lab, lv))
                continue
            level = max(trg, lv) if d == 1 else min(trg, lv)
            out, k, how = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
            say("      on %s: must close past %.3f -> %s%s" % (lab, level, out.upper(),
                (" at %s, price %.3f" % (T.t[k], T.c[k])) if out == "enter" and k is not None else ""))
        tf = pd.Timestamp(CASE["filled"])
        k0 = int(np.searchsorted(T.tv, np.datetime64(tf)))
        hi_max, hit_t = -1e18, None
        for k in range(k0, min(T.n, k0 + 200)):
            hi_max = max(hi_max, float(T.h[k])) if d == 1 else hi_max
            if (d == 1 and T.l[k] <= CASE["stop"]) or (d == -1 and T.h[k] >= CASE["stop"]):
                hit_t = T.t[k]
                break
        say("\n   6. THE LIVE TRADE (filled %.3f at %s): highest price before the stop %.3f | stop %.3f hit in the 30m candle of %s"
            % (CASE["fill"], CASE["filled"], hi_max, CASE["stop"], hit_t))
        say("      30m candles from 2 hours before the fill to the stop (time / open / high / low / close):")
        for k in range(max(k0 - 4, 0), min(T.n, k0 + 12)):
            say("        %s  %.3f  %.3f  %.3f  %.3f" % (T.t[k], T.o[k] if hasattr(T, "o") else float("nan"), T.h[k], T.l[k], T.c[k]))
            if hit_t is not None and T.t[k] >= hit_t:
                break
    else:
        say("   no 30m candles -- parts 5 and 6 skipped")



# =====================================================================================================================
# CLOSE-THROUGH QUALITY (Desire 5 Oct, after the SILVER long): judge the quality of the close-through (and the retest)
# instead of adding vetoes. Each variant changes WHEN the E signal fires; everything after it is the live system after
# B13: a signal with a bigger 4H line (on closes) within half a move ahead, or too far, goes to package A (2-of-3
# majority, strict 1H + two 30m closes past the trigger and the line, 24 h); the rest are taken at once with the engine's
# own stop, target and gates. Every variant runs through the same code, so the rows compare like with like.
#   E       today: the first 1H close past the peak
#   MARGIN  ...by at least 0.15 of a 4H move (Desire's 14 Sep proof tolerance)
#   STRONG  ...and the close-through candle closes the trade's way, in the top third of its range
#   HOLD    ...and the next 1H close also holds past the peak (entry on that second close)
#   M+S     MARGIN and STRONG together
#   REAL RT the retest must come back to the line itself (within the touch allowance of R2), not just to the wick edge
#           of the line's swing; the peak keeps rising until that real retest; then E
#   RT+S    REAL RT, then STRONG
# =====================================================================================================================
CT_VARIANTS = ["E", "MARGIN", "STRONG", "HOLD", "M+S", "REAL RT", "RT+S"]
CT_MARGIN = 0.15


def ct_trigger(M, s, A, v, a4_at):
    """The 1H index of the entry for variant v, the peak it had to beat, and the retest index (or None)."""
    tt = s.get("t_touch")
    it = A["pos"].get(pd.Timestamp(tt)) if tt is not None else None
    if it is None:
        return None
    d, r1, r2 = s["d"], s["r1"], s["r2"]
    h2 = float(s.get("h2_frozen", s.get("h2")))
    op, hi, lo, cl, t1, n, atr = A["op"], A["hi"], A["lo"], A["cl"], A["t1"], A["n"], A["atr"]
    real = v in ("REAL RT", "RT+S")
    start = it
    if real:
        ir = None
        for i in range(it, n):
            if i > it and (A["kill"](i) and ((d == 1 and cl[i] < r1) or (d == -1 and cl[i] > r1))):
                return None
            if t1[i] - pd.Timestamp(tt) > M.WAIT_TRIGGER:
                return None
            if (d == 1 and lo[i] <= r2 + M.TOUCH_ATR * atr[i]) or (d == -1 and hi[i] >= r2 - M.TOUCH_ATR * atr[i]):
                ir = i
                break
            if i > it and d * (cl[i] - h2) > 0:
                h2 = float(cl[i])               # the peak keeps rising until the real retest
        if ir is None:
            return None
        start = ir
    for i in range(start + 1, n):
        if A["kill"](i) and ((d == 1 and cl[i] < r1) or (d == -1 and cl[i] > r1)):
            return None
        if t1[i] - pd.Timestamp(tt) > M.WAIT_TRIGGER:
            return None
        c = cl[i]
        if d * (c - h2) <= 0:
            continue
        if v in ("MARGIN", "M+S"):
            a4 = a4_at(t1[i])
            if not (a4 == a4 and a4 > 0) or d * (c - h2) < CT_MARGIN * a4:
                continue
        if v in ("STRONG", "M+S", "RT+S"):
            rng = hi[i] - lo[i]
            if rng <= 0 or d * (c - op[i]) <= 0:
                continue
            pos = (c - lo[i]) / rng if d == 1 else (hi[i] - c) / rng
            if pos < 2.0 / 3.0:
                continue
        if v == "HOLD":
            if i + 1 >= n or d * (cl[i + 1] - h2) <= 0:
                continue
            return i + 1, h2, start
        return i, h2, start
    return None


def run_ct(data, frames, pcfg, costs, lots, case_r2=61.278):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nCLOSE-THROUGH QUALITY -- the live system after B13, with the E signal judged more strictly in each row\n" + "=" * 110)
    allrows = {v: [] for v in CT_VARIANTS}
    per, spans = {}, []
    tally = {v: collections.Counter() for v in CT_VARIANTS}
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        t1v = A["t1"].values
        V = {v: [] for v in CT_VARIANTS}
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            is_case = (a == "SILVER" and d == 1 and abs(r2 - case_r2) < 0.02 and
                       pd.Timestamp("2026-10-04") <= pd.Timestamp(tt) <= pd.Timestamp("2026-10-06"))
            for v in CT_VARIANTS:
                got = ct_trigger(M, s, A, v, L.atr_at)
                if got is None:
                    tally[v]["no close-through"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-8s no close-through -> NO TRADE" % v)
                    continue
                iE, trg, istart = got
                if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                    tally[v]["spike candle"] += 1
                    continue
                tally[v]["signals"] += 1
                tE, eE = A["t1"][iE], float(A["cl"][iE])
                atr = atr41(M, A, iE)
                if not (atr == atr and atr > 0):
                    continue
                lvl, _ed, near = L.ahead_level(d, eE, tE)
                is_near = lvl is not None and near is not None and near <= NEAR
                far = d * (eE - r2) / atr > M.FRESH_ATR
                if not is_near and not far:
                    struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                    stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                    strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                    if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                        tally[v]["refused by the engine's gates"] += 1
                        continue
                    k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                    if k >= T.n or T.t[k] != tE:
                        continue
                    V[v].append(dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d))
                    tally[v]["taken at once (space to run)"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-8s close-through %s at %.3f (peak %.3f) -> taken at once, stop %.3f" % (
                            v, tE, eE, trg, stop))
                    continue
                level = (max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg
                nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                if nmaj < 2:
                    tally[v]["package: no bigger-timeframe majority"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-8s close-through %s at %.3f -> package: no majority -> NO TRADE" % (v, tE, eE))
                    continue
                out, k, how = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
                if out != "enter":
                    tally[v]["package: " + out] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-8s close-through %s -> package: %s -> NO TRADE" % (v, tE, out))
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr2 = atr41(M, A, j1)
                if not (atr2 == atr2 and atr2 > 0):
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
                if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                    tally[v]["package: stop / reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
                V[v].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d))
                tally[v]["package: entered"] += 1
                if is_case:
                    say("   SILVER 5 Oct, %-8s close-through %s -> package entered %s at %.3f" % (v, tE, tau, q))
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        say("   %-7s %s" % (a, " | ".join("%s %d" % (v, len(V[v])) for v in CT_VARIANTS)))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   ALL MARKETS TOGETHER (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in CT_VARIANTS:
        say("   " + line(v, stats(allrows[v], T0_, T1_)))
    say("\n   WHAT HAPPENED (all markets):")
    for v in CT_VARIANTS:
        say("   %-8s %s" % (v, " | ".join("%s %d" % (k, n) for k, n in tally[v].most_common())))
    say("\n   EACH MARKET:")
    for a in per:
        say("   %s" % a)
        for v in CT_VARIANTS:
            say("     " + line(v, per[a][v]))
    say("\n   E here should sit close to the 'Package A (B13)' row above (same system, rebuilt in one place). MARGIN = 0.15 of a"
        " 4H move | STRONG = closes the trade's way, top third of its range | HOLD = the next 1H close holds too | REAL RT ="
        " the retest reaches the line itself")



# =====================================================================================================================
# THE LIVERMORE BRAIN AS BREAKOUT CONFIRMATION (Desire 5 Oct): the 4H brain splits the chart into six segments (its
# states). A real breakout should move it into another segment, toward the trade; a weak one leaves it where it was.
# Two readings, each used two ways, all on the live system after B13 (package A, lines on closes, today's E):
#   WITH   the brain sits in the trade's family at the close-through (buy: MAIN_UP, NATURAL_RETRACEMENT,
#          SECONDARY_RETRACEMENT; sell: the mirror)
#   MOVED  the brain moved at least one segment toward the trade between the last 4H close before the break and the
#          close-through. Segment order for a buy: MAIN_DOWN < NATURAL_REBOUND < SECONDARY_REBOUND <
#          SECONDARY_RETRACEMENT < NATURAL_RETRACEMENT < MAIN_UP (mirror for a sell)
#   soft   not confirmed -> handed to package A to prove itself (your ruling 5: the brain never kills a proof)
#   hard   not confirmed -> no trade (for comparison only)
# The bot's own Livermore code and each market's own brain settings, fed every closed 4H candle.
# =====================================================================================================================
BR_ORD = {"MAIN_DOWN": 0, "NATURAL_REBOUND": 1, "SECONDARY_REBOUND": 2, "SECONDARY_RETRACEMENT": 3,
          "NATURAL_RETRACEMENT": 4, "MAIN_UP": 5}
BR_ROWS = [("Package A now", None, None), ("WITH soft", "WITH", "soft"), ("MOVED soft", "MOVED", "soft"),
           ("WITH hard", "WITH", "hard"), ("MOVED hard", "MOVED", "hard")]


def lsm_states(a, df, span):
    """(close times, state names) of the bot's own Livermore brain for every closed candle of this timeframe."""
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from src.execution.livermore_state_machine import LivermoreStateMachine, atr14 as lsm_atr
    p = (PIV or {}).get(a, {}) or {}
    lsm = LivermoreStateMachine(asset=a, timeframe="TEST", major_mult=p.get("major_mult", 3.5),
                                minor_mult=p.get("minor_mult", 1.0), dual_confirm=p.get("dual_confirm", 2),
                                atr_period=p.get("atr_period", 14))
    d = df[["high", "low", "close"]].astype(float)
    atr = lsm_atr(d, int(p.get("atr_period", 14))).values
    c = d["close"].values
    out = []
    for i in range(len(c)):
        lsm.update(float(c[i]), float(atr[i]))
        out.append(str(lsm.state() if callable(getattr(lsm, "state", None)) else getattr(lsm, "state", "")))
    return (df.index + span).values, out


def run_brain(data, frames, pcfg, costs, lots, case_r2=61.278):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nTHE LIVERMORE BRAIN AS BREAKOUT CONFIRMATION -- the live system after B13, routed by the 4H brain in each row\n" + "=" * 110)
    names = [r[0] for r in BR_ROWS]
    allrows = {v: [] for v in names}
    per, spans = {}, []
    tally = {v: collections.Counter() for v in names}
    conf = collections.Counter()
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        try:
            bt, bs = lsm_states(a, d4, pd.Timedelta(hours=4))
        except Exception as ex:
            say("   %-7s the brain could not be run: %s -- market skipped" % (a, ex))
            continue
        t1v = A["t1"].values

        def brain_at(t):
            k = int(np.searchsorted(bt, np.datetime64(pd.Timestamp(t)), side="right")) - 1
            return bs[k] if k >= 0 else ""
        V = {v: [] for v in names}
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            atr = atr41(M, A, iE)
            if not (atr == atr and atr > 0):
                continue
            st_now = brain_at(tE)
            tb = s.get("b_t") or s.get("t_break")
            st_before = brain_at(pd.Timestamp(tb) - pd.Timedelta(hours=4)) if tb is not None else ""
            o_now = BR_ORD.get(st_now)
            o_bef = BR_ORD.get(st_before)
            if o_now is not None and d == -1:
                o_now = 5 - o_now
            if o_bef is not None and d == -1:
                o_bef = 5 - o_bef
            ok = {"WITH": o_now is not None and o_now >= 3,
                  "MOVED": o_now is not None and o_bef is not None and o_now > o_bef}
            conf["signals"] += 1
            conf["brain WITH the trade at the close-through"] += ok["WITH"]
            conf["brain MOVED toward the trade since the break"] += ok["MOVED"]
            is_case = (a == "SILVER" and d == 1 and abs(r2 - case_r2) < 0.02 and
                       pd.Timestamp("2026-10-04") <= pd.Timestamp(tt) <= pd.Timestamp("2026-10-06"))
            if is_case:
                say("   SILVER 5 Oct: the brain before the break %s -> at the close-through %s | WITH %s | MOVED %s" % (
                    st_before or "?", st_now or "?", "yes" if ok["WITH"] else "no", "yes" if ok["MOVED"] else "no"))
            lvl, _ed, near = L.ahead_level(d, eE, tE)
            is_near = lvl is not None and near is not None and near <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            nmaj = None
            for name, rule, how in BR_ROWS:
                if rule is not None and not ok[rule] and how == "hard":
                    tally[name]["not confirmed by the brain -- no trade"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-14s -> NO TRADE" % name)
                    continue
                to_pkg = is_near or far or (rule is not None and not ok[rule] and how == "soft")
                if not to_pkg:
                    struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                    stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                    strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                    if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                        tally[name]["refused by the engine's gates"] += 1
                        continue
                    k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                    if k >= T.n or T.t[k] != tE:
                        continue
                    V[name].append(dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d))
                    tally[name]["taken at once"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-14s -> taken at once at %.3f" % (name, eE))
                    continue
                if rule is not None and not ok[rule] and not is_near and not far:
                    tally[name]["sent to the package by the brain"] += 1
                level = (max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg
                if nmaj is None:
                    nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                if nmaj < 2:
                    tally[name]["package: no bigger-timeframe majority"] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-14s -> package: no majority -> NO TRADE" % name)
                    continue
                out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
                if out != "enter":
                    tally[name]["package: " + out] += 1
                    if is_case:
                        say("   SILVER 5 Oct, %-14s -> package: %s -> NO TRADE" % (name, out))
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr2 = atr41(M, A, j1)
                if not (atr2 == atr2 and atr2 > 0):
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
                if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                    tally[name]["package: stop / reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
                V[name].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d))
                tally[name]["package: entered"] += 1
                if is_case:
                    say("   SILVER 5 Oct, %-14s -> package entered %s at %.3f" % (name, tau, q))
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        say("   %-7s %s" % (a, " | ".join("%s %d" % (v, len(V[v])) for v in names)))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   HOW OFTEN THE BRAIN CONFIRMED (all markets): " + " | ".join("%s %d" % (k, n) for k, n in conf.items()))
    say("\n   ALL MARKETS TOGETHER (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in names:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   WHAT HAPPENED (all markets):")
    for v in names:
        say("   %-14s %s" % (v, " | ".join("%s %d" % (k, n) for k, n in tally[v].most_common())))
    say("\n   EACH MARKET:")
    for a in per:
        say("   %s" % a)
        for v in names:
            say("     " + line(v[:15], per[a][v]))
    say("\n   'Package A now' here should equal the 'Package A (B13)' row above (same system, rebuilt in one place).")



# =====================================================================================================================
# ZONES, LINES AND DIAGONALS (Desire 5 Oct). Part 1: the Livermore brain's lines (both brains: the main up-leg high,
# the main down-leg low, the natural high and the natural low) -- when a 4H close crosses one, how often does price go
# on 2 more 4H moves before closing back across it, against ordinary 4H swing lines? Part 2: our trades by the zone
# they were taken in. Part 3: the live system after B13 with idea 1 (a diagonal broke the same way in the 24 h before
# the signal -> taken at once even with a line just ahead), idea 2 (an unbroken 3-touch diagonal within half a move
# ahead -> the package must clear it), both, and the same two ideas with the brain's lines in place of diagonals.
# =====================================================================================================================
ZN_KEYS = (("up", "MAIN UP max"), ("down", "MAIN DOWN min"), ("nhigh", "natural high"), ("nlow", "natural low"))
ZN_FWD, ZN_GO = 30, 2.0
ZN_ROWS = ["Package A now", "1 diag fast lane", "2 diag ceiling", "1+2 diagonals", "brain line ahead", "brain cross fast"]


def brain_lines(a, df, tf):
    """The bot's own Livermore brain on this timeframe: close times, states and the four lines after every candle."""
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from src.execution.livermore_state_machine import make_livermore_pair, atr14 as lsm_atr
    m4, m1 = make_livermore_pair(a, (PIV or {}).get(a, {}) or {})
    m = m4 if tf == "4H" else m1
    d = df[["high", "low", "close"]].astype(float)
    atr = lsm_atr(d, int(((PIV or {}).get(a, {}) or {}).get("atr_period", 14))).values
    c = d["close"].values
    out = {k: np.full(len(c), np.nan) for k, _ in ZN_KEYS}
    st = []
    for i in range(len(c)):
        try:
            s_ = m.update(float(c[i]), float(atr[i]))
        except Exception:
            st.append("")
            continue
        st.append(str(s_.state))
        for k, v in (("up", s_.anchor_main_up_max), ("down", s_.anchor_main_down_min),
                     ("nhigh", s_.anchor_natural_high), ("nlow", s_.anchor_natural_low)):
            if v is not None and v == v:
                out[k][i] = float(v)
    return (df.index + (pd.Timedelta(hours=4) if tf == "4H" else pd.Timedelta(hours=1))).values, st, out


def diag_lines(M, d4):
    """Every 3-touch diagonal (the D1 finder): falling lines through swing-high closes (ceilings for a buy, d=1) and
    rising lines through swing-low closes (floors for a sell, d=-1), with the 4H candle it became valid and broke."""
    hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
    a4 = M.atr14(hi4, lo4, c4)
    n4, KK = len(c4), M.K
    piv = {"H": [], "L": []}
    for i in range(KK, n4 - KK):
        w = c4[i - KK:i + KK + 1]
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
            nxt = P[j + 1] if j + 1 < len(P) else n4
            start = i2 + KK
            end = min(n4, start + DIAG_LIFE, nxt + KK)
            earlier = P[max(0, j - 5):j - 1]
            vf = start if any(abs(c4[p] - (v1 + slope * (p - i1))) <= 0.5 * a4[p] for p in earlier if a4[p] == a4[p]) else None
            brk = None
            for x in range(start, end):
                lv = v1 + slope * (x - i1)
                if vf is None:
                    if x >= i2 + 3 and a4[x] == a4[x] and abs(c4[x] - lv) <= 0.5 * a4[x]:
                        vf = x + 1
                    continue
                if x < vf or not (a4[x] == a4[x] and a4[x] > 0):
                    continue
                if d * (c4[x] - lv) >= 0.25 * a4[x]:
                    brk = x
                    break
            if vf is not None:
                out.append(dict(d=d, i1=i1, v1=v1, slope=slope, vf=vf, brk=brk, end=end))
    return out


def run_zones(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nZONES, LINES AND DIAGONALS -- the brain's lines as gates, our trades by zone, and the two diagonal ideas\n" + "=" * 110)
    cross = collections.Counter()
    allrows = {v: [] for v in ZN_ROWS}
    zone_rows = collections.defaultdict(list)
    per, spans = {}, []
    tally = {v: collections.Counter() for v in ZN_ROWS}
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        t4c = (d4.index + pd.Timedelta(hours=4)).values
        hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
        a4 = M.atr14(hi4, lo4, c4)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        try:
            b4t, _b4s, b4 = brain_lines(a, d4, "4H")
            b1t, _b1s, b1 = brain_lines(a, df1, "1H")
        except Exception as ex:
            say("   %-7s the brain could not be run: %s -- market skipped" % (a, ex))
            continue
        DL = diag_lines(M, d4)
        dbreaks = [(x["d"], pd.Timestamp(t4c[x["brk"]])) for x in DL if x["brk"] is not None]
        t1v = A["t1"].values
        # ---- part 1: crossings of the brain's lines vs ordinary 4H swing lines --------------------------------
        for x in range(1, len(c4)):
            if not (a4[x] == a4[x] and a4[x] > 0):
                continue
            k1 = int(np.searchsorted(b1t, t4c[x - 1], side="right")) - 1
            lines = [("4H " + nm, b4[k][x - 1]) for k, nm in ZN_KEYS] + \
                    ([("1H " + nm, b1[k][k1]) for k, nm in ZN_KEYS] if k1 >= 0 else [])
            top = int(np.searchsorted(L.conf, t4c[x - 1], side="right"))
            t0 = t4c[x - 1] - np.timedelta64(AHEAD_DAYS, "D")
            for y in range(top - 1, -1, -1):
                cf, ty, lv, _i, _ed = L.sw[y]
                if cf < t0:
                    break
                lines.append(("ordinary 4H swing line", lv))
            for nm, v in lines:
                if not (v == v):
                    continue
                d = 1 if c4[x - 1] <= v < c4[x] else (-1 if c4[x - 1] >= v > c4[x] else 0)
                if d == 0:
                    continue
                res = "neither"
                for y in range(x + 1, min(len(c4), x + 1 + ZN_FWD)):
                    if d * (c4[y] - v) < 0:
                        res = "back"
                        break
                    if d * ((hi4[y] if d == 1 else lo4[y]) - v) >= ZN_GO * a4[x]:
                        res = "on"
                        break
                cross[(nm, "up" if d == 1 else "down", res)] += 1
        # ---- parts 2 and 3: the live system with the diagonal and brain-line ideas --------------------------
        V = {v: [] for v in ZN_ROWS}

        def diag_ahead(d, e, kx):
            best = None
            for ln in DL:
                if ln["d"] != d or ln["vf"] > kx or kx >= ln["end"] or (ln["brk"] is not None and ln["brk"] <= kx):
                    continue
                lv = ln["v1"] + ln["slope"] * (kx - ln["i1"])
                if d * (lv - e) > 0 and (best is None or d * (lv - e) < d * (best - e)):
                    best = lv
            return best

        def brain_now(t):
            kx4 = int(np.searchsorted(b4t, np.datetime64(pd.Timestamp(t)), side="right")) - 1
            kx1 = int(np.searchsorted(b1t, np.datetime64(pd.Timestamp(t)), side="right")) - 1
            return ({nm: b4[k][kx4] for k, nm in ZN_KEYS} if kx4 >= 0 else {},
                    {nm: b1[k][kx1] for k, nm in ZN_KEYS} if kx1 >= 0 else {})
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            atr = atr41(M, A, iE)
            aE = L.atr_at(tE)
            if not (atr == atr and atr > 0 and aE == aE and aE > 0):
                continue
            kx = int(np.searchsorted(t4c, np.datetime64(tE), side="right")) - 1
            lvl, _ed, near = L.ahead_level(d, eE, tE)
            is_near = lvl is not None and near is not None and near <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            dbr = any(dd == d and tE - pd.Timedelta(hours=24) <= tb <= tE for dd, tb in dbreaks)
            dah = diag_ahead(d, eE, kx)
            dnear = dah is not None and d * (dah - eE) <= NEAR * aE
            bl4, bl1 = brain_now(tE)
            blines = [v for v in list(bl4.values()) + list(bl1.values()) if v == v]
            bahead = [v for v in blines if 0 < d * (v - eE) <= NEAR * aE]
            tb = s.get("b_t") or s.get("t_break")
            bcross = False
            if tb is not None:
                pb, _ = brain_now(pd.Timestamp(tb) - pd.Timedelta(hours=4))
                kb = int(np.searchsorted(t4c, np.datetime64(pd.Timestamp(tb)), side="right")) - 1
                if kb >= 1:
                    bcross = any(v == v and d * (c4[kb - 1] - v) <= 0 < d * (c4[kb] - v) for v in pb.values())
            # the zone for part 2: the highest 4H brain line the entry is past (buy: above; sell: below)
            past = [(nm, v) for nm, v in bl4.items() if v == v and d * (eE - v) > 0]
            zone = (max(past, key=lambda x: d * x[1])[0] if past else "none") + (", line just ahead" if any(
                0 < d * (v - eE) <= NEAR * aE for v in bl4.values() if v == v) else "")
            info = tally["Package A now"]
            info["(info) signals"] += 1
            info["(info) a diagonal broke the same way in the 24 h before"] += dbr
            info["(info) an unbroken diagonal within half a move ahead"] += dnear
            info["(info) a brain line within half a move ahead"] += bool(bahead)
            info["(info) the break crossed a 4H brain line"] += bcross
            for name in ZN_ROWS:
                nr, lev = is_near, ((max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg)
                if name in ("2 diag ceiling", "1+2 diagonals") and dnear and not is_near:
                    nr, lev = True, (max(trg, dah) if d == 1 else min(trg, dah))
                if name == "brain line ahead" and bahead and not is_near:
                    bl = min(bahead, key=lambda v: d * (v - eE))
                    nr, lev = True, (max(trg, bl) if d == 1 else min(trg, bl))
                fast = (name in ("1 diag fast lane", "1+2 diagonals") and dbr) or (name == "brain cross fast" and bcross)
                if not far and (not nr or fast):
                    struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                    stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                    strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                    if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                        tally[name]["refused by the engine's gates"] += 1
                        continue
                    k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                    if k >= T.n or T.t[k] != tE:
                        continue
                    c = dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d)
                    V[name].append(c)
                    tally[name]["taken at once" + (" (fast lane)" if (fast and nr) else "")] += 1
                    if name == "Package A now":
                        zone_rows[zone].append((a, c))
                    continue
                nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                if nmaj < 2:
                    tally[name]["package: no bigger-timeframe majority"] += 1
                    continue
                out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, lev, "A", 24, False)
                if out != "enter":
                    tally[name]["package: " + out] += 1
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr2 = atr41(M, A, j1)
                if not (atr2 == atr2 and atr2 > 0):
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
                if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                    tally[name]["package: stop / reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
                c = dict(k=k, q=q, stop=stop, tgt=tgt, d=d)
                V[name].append(c)
                tally[name]["package: entered"] += 1
                if name == "Package A now":
                    zone_rows[zone + " (package)"].append((a, c))
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        for z in list(zone_rows):
            mine = [c for aa, c in zone_rows[z] if aa == a]
            if mine:
                zone_rows[z] = [x for x in zone_rows[z] if x[0] != a] + [("_done", r) for r in book30(M, A, T, mine, runner, cost, usd)]
        say("   %-7s %s" % (a, " | ".join("%s %d" % (v, len(V[v])) for v in ZN_ROWS)))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   PART 1 -- WHEN A 4H CLOSE CROSSES A LINE: 'went on' = price reached 2 more 4H moves past it before a 4H close back"
        " across it (within %d 4H candles); 'back' = closed back across first" % ZN_FWD)
    say("   %-34s %-5s %8s %9s %8s %9s" % ("line", "way", "crosses", "went on", "back", "neither"))
    for nm in sorted({k[0] for k in cross}, key=lambda x: (x.startswith("ordinary"), x)):
        for way in ("up", "down"):
            n_on, n_back, n_nei = cross[(nm, way, "on")], cross[(nm, way, "back")], cross[(nm, way, "neither")]
            n = n_on + n_back + n_nei
            if n:
                say("   %-34s %-5s %8d %8.0f%% %7.0f%% %8.0f%%" % (nm, way, n, 100.0 * n_on / n, 100.0 * n_back / n, 100.0 * n_nei / n))
    say("\n   PART 2 -- OUR TRADES (package A now) BY ZONE: the highest 4H brain line the entry had passed, and whether another"
        " sat within half a move ahead")
    say("   " + HEAD)
    for z in sorted(zone_rows):
        rows = [r for aa, r in zone_rows[z] if aa == "_done"]
        if rows:
            say("   " + line(z[:15], stats(rows, T0_, T1_)) + "   <- " + z)
    say("\n   PART 3 -- THE LIVE SYSTEM AFTER B13 WITH EACH IDEA (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in ZN_ROWS:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   WHAT HAPPENED (all markets):")
    for v in ZN_ROWS:
        say("   %-17s %s" % (v, " | ".join("%s %d" % (k, n) for k, n in tally[v].most_common())))
    say("\n   EACH MARKET:")
    for a in per:
        say("   %s" % a)
        for v in ZN_ROWS:
            say("     " + line(v[:15], per[a][v]))
    say("\n   'Package A now' should equal the 'Package A (B13)' row above. Ideas: 1 = a 3-touch diagonal broke the same way in the"
        " 24 h before the signal -> taken at once even with a line just ahead | 2 = an unbroken 3-touch diagonal within half a"
        " move ahead -> the package must clear it | brain line ahead = a 4H or 1H brain line within half a move ahead -> the"
        " package must clear it | brain cross fast = the break's 4H close crossed a 4H brain line -> taken at once")



# =====================================================================================================================
# MOVING AVERAGES AS DIAGONAL LINES (Desire 5 Oct): the fast lane (B13 item 32) with a moving average in place of the
# 3-touch diagonal. A break of an MA = a 4H close at least 0.25 of a 4H move past it, the previous 4H close not; the
# daily MAs are read at their last finished daily close. If the MA broke the same way in the 24 h before a horizontal
# E signal that has an old high just ahead, the signal is taken at once instead of waiting in the package -- exactly as
# the diagonal fast lane. Everything else is the live system after B13 (package A, lines on closes).
# =====================================================================================================================
MA_LINES = (("4H 50 EMA", "4H", 50), ("4H 200 EMA", "4H", 200), ("1D 50 EMA", "1D", 50), ("1D 200 EMA", "1D", 200))
MA_ROWS = ["Package A now", "diagonal lane"] + ["%s lane" % m[0] for m in MA_LINES] + ["any MA lane"]


def run_ma(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nMOVING AVERAGES AS DIAGONAL LINES -- the fast lane with the 4H and daily 50/200 EMAs in place of the diagonal\n" + "=" * 110)
    allrows = {v: [] for v in MA_ROWS}
    lane_rows = {v: [] for v in MA_ROWS}
    per, spans = {}, []
    tally = {v: collections.Counter() for v in MA_ROWS}
    info = collections.Counter()
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        t4c = (d4.index + pd.Timedelta(hours=4)).values
        hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
        a4 = M.atr14(hi4, lo4, c4)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        DL = diag_lines(M, d4)
        breaks = {"diagonal": [(x["d"], pd.Timestamp(t4c[x["brk"]])) for x in DL if x["brk"] is not None]}
        dend = (D1.index + pd.Timedelta(days=1)).values
        dcl = D1["close"].astype(float)
        for nm, tf, span in MA_LINES:
            if tf == "4H":
                ema = pd.Series(c4).ewm(span=span, adjust=False).mean().values
                ok = np.arange(len(c4)) >= span
            else:
                ed = dcl.ewm(span=span, adjust=False).mean().values
                kd = np.searchsorted(dend, t4c, side="right") - 1
                ema = np.where(kd >= 0, ed[np.clip(kd, 0, None)], np.nan)
                ok = kd >= span
            ev = []
            for x in range(1, len(c4)):
                if not (ok[x] and ok[x - 1] and a4[x] == a4[x] and a4[x] > 0 and a4[x - 1] == a4[x - 1]):
                    continue
                for d in (1, -1):
                    if d * (c4[x] - ema[x]) >= 0.25 * a4[x] and d * (c4[x - 1] - ema[x - 1]) < 0.25 * a4[x - 1]:
                        ev.append((d, pd.Timestamp(t4c[x])))
            breaks[nm] = ev
        t1v = A["t1"].values
        V = {v: [] for v in MA_ROWS}
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            atr = atr41(M, A, iE)
            if not (atr == atr and atr > 0):
                continue
            lvl, _ed, near = L.ahead_level(d, eE, tE)
            is_near = lvl is not None and near is not None and near <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            hit = {nm: any(dd == d and tE - pd.Timedelta(hours=24) <= tb <= tE for dd, tb in ev) for nm, ev in breaks.items()}
            hit["any MA"] = any(hit[m[0]] for m in MA_LINES)
            info["signals"] += 1
            for nm in ["diagonal"] + [m[0] for m in MA_LINES] + ["any MA"]:
                info["%s broke the same way in the 24 h before" % nm] += hit[nm]
            nmaj = None
            for name in MA_ROWS:
                key = None if name == "Package A now" else ("diagonal" if name == "diagonal lane" else name[:-5])
                fast = key is not None and hit[key]
                if not far and (not is_near or fast):
                    struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                    stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                    strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                    if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                        tally[name]["refused by the engine's gates"] += 1
                        continue
                    k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                    if k >= T.n or T.t[k] != tE:
                        continue
                    c = dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d)
                    V[name].append(c)
                    if fast and is_near:
                        tally[name]["taken at once by the fast lane"] += 1
                        lane_rows[name].append((a, c))
                    else:
                        tally[name]["taken at once (space to run)"] += 1
                    continue
                if nmaj is None:
                    nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                if nmaj < 2:
                    tally[name]["package: no bigger-timeframe majority"] += 1
                    continue
                level = (max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg
                out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
                if out != "enter":
                    tally[name]["package: " + out] += 1
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr2 = atr41(M, A, j1)
                if not (atr2 == atr2 and atr2 > 0):
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
                if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                    tally[name]["package: stop / reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
                V[name].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d))
                tally[name]["package: entered"] += 1
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
            mine = [c for aa, c in lane_rows[v] if aa == a]
            if mine:
                lane_rows[v] = [x for x in lane_rows[v] if x[0] != a] + [("_done", r) for r in book30(M, A, T, mine, runner, cost, usd)]
        say("   %-7s %s" % (a, " | ".join("%s %d" % (v, len(V[v])) for v in MA_ROWS)))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   HOW OFTEN EACH LINE BROKE THE SAME WAY IN THE 24 H BEFORE A SIGNAL (all markets): " +
        " | ".join("%s %d" % (k, n) for k, n in info.items()))
    say("\n   THE LIVE SYSTEM AFTER B13 WITH EACH FAST LANE (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in MA_ROWS:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   THE FAST-LANE TRADES ON THEIR OWN (the signals each lane took at once that would otherwise have waited):")
    say("   " + HEAD)
    for v in MA_ROWS[1:]:
        rows = [r for aa, r in lane_rows[v] if aa == "_done"]
        say("   " + (line(v[:15], stats(rows, T0_, T1_)) if rows else "%-15s (none)" % v[:15]))
    say("\n   WHAT HAPPENED (all markets):")
    for v in MA_ROWS:
        say("   %-15s %s" % (v[:15], " | ".join("%s %d" % (k, n) for k, n in tally[v].most_common())))
    say("\n   EACH MARKET:")
    for a in per:
        say("   %s" % a)
        for v in MA_ROWS:
            say("     " + line(v[:15], per[a][v]))
    say("\n   'Package A now' should equal the 'Package A (B13)' row above; 'diagonal lane' should equal the zones test's"
        " '1 diag fast lane'.")



# =====================================================================================================================
# THE SCENARIOS TEST (Desire 6 Oct) -- on the live system after B13 (package A, lines on closes, the 24 h diagonal lane):
#  1. diagonal MEMORY: the lane with the diagonal broken within 3 days, 7 days, or at any time while price has not
#     closed back across it (scenarios 2, 3, 6); lane trades split into continuations and reversals
#  2. SAME MOVE: the diagonal broke within one 4H candle of the horizontal break (scenarios 1 and 9)
#  3. THE SPRING (scenario 4): a 4H close through a support line (a 4H swing-low close), a 4H close back over it within
#     6 candles, and a falling diagonal broken upward in the 24 h before the reclaim -> buy at the reclaim close, stop a
#     tenth of a 4H move under the trap's low (mirror for sells), the market's own target and reward:risk gate
#  4. THE EARLY EXIT (scenario 7): get out at the first 4H close back across the broken line
#  +  YOUR QUESTION: big lines with space to run but no diagonal sent through the package instead of bought at once
# =====================================================================================================================
SC_ROWS = ["B13 (24h lane)", "lane 3 days", "lane 7 days", "lane while holding", "lane same move",
           "B13 + early exit", "B13 + spring", "no-diag big->pkg"]
SPRING_WAIT = 6


def book30x(M, A, T, cands, runner, cost, usd, is4):
    """book30, plus the early exit: out at the first 4H close back across the broken line (candidates with 'x')."""
    rows, free, early = [], -1, 0
    for c in sorted(cands, key=lambda x: x["k"]):
        if c["k"] <= free:
            continue
        kk, px, risk0 = sim30(M, A, T, c["d"], c["k"], c["q"], c["stop"], c["tgt"], runner)
        if c.get("x") is not None:
            for k2 in range(c["k"] + 1, min(kk, T.n)):
                if is4[k2] and c["d"] * (T.c[k2] - c["x"]) < 0:
                    kk, px = k2, float(T.c[k2])
                    early += 1
                    break
        pts = c["d"] * (px - c["q"]) - cost * c["q"]
        rows.append(dict(t=T.t[kk], money=pts * usd, R=pts / risk0 if risk0 > 0 else 0.0, inside=False, noise=False,
                         delay=None, give=None))
        free = kk
    return rows, early


def run_scen(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nTHE SCENARIOS TEST -- diagonal memory, same move, the spring, the early exit, and space-to-run without a diagonal\n" + "=" * 110)
    allrows = {v: [] for v in SC_ROWS}
    lane = collections.defaultdict(list)
    spr = {"spring + diagonal": [], "spring alone": []}
    per, spans = {}, []
    tally = {v: collections.Counter() for v in SC_ROWS}
    info = collections.Counter()
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        t4c = (d4.index + pd.Timedelta(hours=4)).values
        hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
        a4 = M.atr14(hi4, lo4, c4)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        is4 = np.isin(T.tv, t4c)
        DL = diag_lines(M, d4)
        dl = []                                   # (d, break index, holds until index)
        for ln in DL:
            if ln["brk"] is None:
                continue
            b, d = ln["brk"], ln["d"]
            hold = len(c4)
            for y in range(b + 1, min(len(c4), b + 1 + DIAG_LIFE)):
                if d * (c4[y] - (ln["v1"] + ln["slope"] * (y - ln["i1"]))) < 0:
                    hold = y
                    break
            else:
                hold = min(len(c4), b + 1 + DIAG_LIFE)
            dl.append((d, b, hold))
        t1v = A["t1"].values
        # ---- the spring (scenario 4) and its mirror ----------------------------------------------------------
        S = {"spring + diagonal": [], "spring alone": []}
        seen = set()
        for x in range(2, len(c4)):
            if not (a4[x] == a4[x] and a4[x] > 0):
                continue
            top = int(np.searchsorted(L.conf, t4c[x - 1], side="right"))
            t0 = t4c[x - 1] - np.timedelta64(AHEAD_DAYS, "D")
            for y in range(top - 1, -1, -1):
                cf, ty, lv, _i, _ed = L.sw[y]
                if cf < t0:
                    break
                d = 1 if ty == "L" else -1            # a buy springs from a support line; a sell from a resistance line
                if d * (c4[x] - lv) <= 0 or d * (c4[x - 1] - lv) > 0:
                    continue                          # this close must be back over the line, the last one under it
                xs = None
                for z in range(x - 1, max(0, x - 1 - SPRING_WAIT), -1):
                    if d * (c4[z] - lv) < 0 and (z == 0 or d * (c4[z - 1] - lv) >= 0):
                        xs = z
                        break
                if xs is None or (d, round(lv, 8), xs) in seen:
                    continue
                seen.add((d, round(lv, 8), xs))
                info["springs: a close through a line and back within 6 candles"] += 1
                trap = lo4[xs:x + 1].min() if d == 1 else hi4[xs:x + 1].max()
                e = float(c4[x])
                tt = pd.Timestamp(t4c[x])
                j = int(np.searchsorted(t1v, np.datetime64(tt), side="right")) - 1
                atr1 = float(A["atr"][j]) if j >= 0 else float("nan")
                if not (atr1 == atr1 and atr1 > 0):
                    continue
                stop = float(trap - d * 0.1 * a4[x])
                if d * (e - stop) < 0.3 * atr1:
                    stop = e - d * 0.3 * atr1
                if d * (e - stop) > 5 * atr1 or not M.gate_rr_ok(e, stop, atr1, cfg["min_rr"]):
                    continue
                tgt = None if cfg["target_atr"] is None else e + d * float(cfg["target_atr"]) * atr1
                k = int(np.searchsorted(T.tv, np.datetime64(tt)))
                if k >= T.n or T.t[k] != tt:
                    continue
                c = dict(k=k, q=e, stop=stop, tgt=tgt, d=d)
                S["spring alone"].append(c)
                if any(dd == d and 0 <= x - b <= 6 for dd, b, _h in dl):
                    S["spring + diagonal"].append(c)
                    info["springs with a diagonal broken the same way in the 24 h before the reclaim"] += 1
        # ---- the horizontal signals on the live system -------------------------------------------------------
        V = {v: [] for v in SC_ROWS}
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            atr = atr41(M, A, iE)
            if not (atr == atr and atr > 0):
                continue
            kx = int(np.searchsorted(t4c, np.datetime64(tE), side="right")) - 1
            tb = s.get("b_t") or s.get("t_break")
            kb = int(np.searchsorted(t4c, np.datetime64(pd.Timestamp(tb)), side="right")) - 1 if tb is not None else -99
            lvl, _ed, near = L.ahead_level(d, eE, tE)
            is_near = lvl is not None and near is not None and near <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            hit = {"24h": any(dd == d and tE - pd.Timedelta(hours=24) <= pd.Timestamp(t4c[b]) <= tE for dd, b, _h in dl),
                   "3d": any(dd == d and tE - pd.Timedelta(days=3) <= pd.Timestamp(t4c[b]) <= tE for dd, b, _h in dl),
                   "7d": any(dd == d and tE - pd.Timedelta(days=7) <= pd.Timestamp(t4c[b]) <= tE for dd, b, _h in dl),
                   "hold": any(dd == d and b <= kx < h for dd, b, h in dl),
                   "same": any(dd == d and abs(b - kb) <= 1 for dd, b, _h in dl)}
            info["signals"] += 1
            for kname in ("24h", "3d", "7d", "hold", "same"):
                info["diagonal %s" % kname] += hit[kname]
            kind = "continuation" if s.get("kind") == "continuation" else "reversal"
            nmaj = None
            for name in SC_ROWS:
                cond = {"lane 3 days": "3d", "lane 7 days": "7d", "lane while holding": "hold",
                        "lane same move": "same"}.get(name, "24h")
                fast = hit[cond]
                if name == "no-diag big->pkg":
                    at_once = not far and fast
                else:
                    at_once = not far and (not is_near or fast)
                if at_once:
                    struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                    stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                    strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                    if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                        tally[name]["refused by the engine's gates"] += 1
                        continue
                    k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                    if k >= T.n or T.t[k] != tE:
                        continue
                    c = dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d, x=(r2 if name == "B13 + early exit" else None))
                    V[name].append(c)
                    if fast and is_near:
                        tally[name]["taken at once by the lane"] += 1
                        if name in ("B13 (24h lane)", "lane 3 days", "lane 7 days", "lane while holding", "lane same move"):
                            lane[(name, "all")].append((a, c))
                            lane[(name, kind)].append((a, c))
                    else:
                        tally[name]["taken at once" + (" (space to run)" if not is_near else "")] += 1
                    continue
                if nmaj is None:
                    nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
                if nmaj < 2:
                    tally[name]["package: no bigger-timeframe majority"] += 1
                    continue
                level = (max(trg, lvl) if d == 1 else min(trg, lvl)) if is_near else trg
                out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
                if out != "enter":
                    tally[name]["package: " + out] += 1
                    continue
                q, tau = float(T.c[k]), T.t[k]
                j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
                atr2 = atr41(M, A, j1)
                if not (atr2 == atr2 and atr2 > 0):
                    continue
                stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
                if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                    tally[name]["package: stop / reward:risk"] += 1
                    continue
                tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
                V[name].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d, x=(r2 if name == "B13 + early exit" else None)))
                tally[name]["package: entered"] += 1
        V["B13 + spring"] = V["B13 + spring"] + S["spring + diagonal"]
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows, early = book30x(M, A, T, cands, runner, cost, usd, is4)
            if early:
                tally[v]["(early exits)"] += early
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        for key in list(lane):
            mine = [c for aa, c in lane[key] if aa == a]
            if mine:
                lane[key] = [x for x in lane[key] if x[0] != a] + [("_done", r) for r in book30x(M, A, T, mine, runner, cost, usd, is4)[0]]
        for nm, cands in S.items():
            spr[nm] += book30x(M, A, T, cands, runner, cost, usd, is4)[0]
        say("   %-7s %s | springs %d (with a diagonal %d)" % (a, " | ".join("%s %d" % (v, len(V[v])) for v in SC_ROWS),
                                                         len(S["spring alone"]), len(S["spring + diagonal"])))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   HOW OFTEN (all markets): " + " | ".join("%s %d" % (k, n) for k, n in info.items()))
    say("\n   THE LIVE SYSTEM WITH EACH CHANGE (one position per market at a time; priced on the 30m candles):")
    say("   " + HEAD)
    for v in SC_ROWS:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   THE LANE TRADES ON THEIR OWN (signals with a line just ahead that each lane took at once), all / continuations /"
        " reversals:")
    say("   " + HEAD)
    for v in SC_ROWS[:5]:
        for kind in ("all", "continuation", "reversal"):
            rows = [r for aa, r in lane.get((v, kind), []) if aa == "_done"]
            lab = (v if kind == "all" else "  " + kind)[:15]
            say("   " + (line(lab, stats(rows, T0_, T1_)) if rows else "%-15s (none)" % lab) + ("   <- " + v if kind == "all" else ""))
    say("\n   THE SPRING ON ITS OWN (scenario 4):")
    say("   " + HEAD)
    for nm, rows in spr.items():
        say("   " + (line(nm[:15], stats(rows, T0_, T1_)) if rows else "%-15s (none)" % nm[:15]) + "   <- " + nm)
    say("\n   WHAT HAPPENED (all markets):")
    for v in SC_ROWS:
        say("   %-17s %s" % (v[:17], " | ".join("%s %d" % (k, n) for k, n in tally[v].most_common())))
    say("\n   EACH MARKET:")
    for a in per:
        say("   %s" % a)
        for v in SC_ROWS:
            say("     " + line(v[:15], per[a][v]))
    say("\n   'B13 (24h lane)' should equal the MA test's 'diagonal lane' row. Lanes: 3 days / 7 days = the diagonal broke the same"
        " way within that time before the signal | while holding = it broke and no 4H close has gone back across it since"
        " (up to 10 days) | same move = within one 4H candle of the horizontal break | no-diag big->pkg = space-to-run"
        " signals without a diagonal in the last 24 h go through the package")



# =====================================================================================================================
# B13 RESEARCH RUNS (Desire 6 Oct) -- read-only, run at the start of B13 while the rest is built.
#  R1  THE TIGHT SPRING: only quality supports (a swing-low close that held before -- an earlier swing low within a
#      quarter of a 4H move -- or that sits within a quarter move of a 4H brain line); a SHALLOW sweep (the closes
#      through it stay within one 4H move of it); a QUICK reclaim (a 4H close back over it within 3 candles); with and
#      without a 3-touch diagonal broken the same way in the 24 h before the reclaim. Mirror for sells. Shown on its
#      own and added to B13 FINAL (package A + wick lines + the diagonal confirmation rule) -- the first run of
#      B13's strategy changes all together.
#  R2  DAILY AND WEEKLY BRAIN CALIBRATION: for each market, 40 brain settings on the daily and on the weekly candles;
#      the best on the FIRST half (how often the brain's side -- up or down family -- matched the next 5 days /
#      4 weeks), then checked on the SECOND half against the market's current settings; plus how B13-final trades
#      did when the calibrated daily/weekly brain agreed with them vs not.
# =====================================================================================================================
UPS = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
DOWNS = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
CAL_GRID = [(mj, mn, du) for mj in (2.0, 2.5, 3.0, 3.5, 4.0) for mn in (0.5, 0.75, 1.0, 1.5) for du in (1, 2)]
CAL_H = {"D1": 5, "W1": 4}
R_ROWS = ["B13 final", "B13 final + spring"]


def fam_series(df, major, minor, dual):
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from src.execution.livermore_state_machine import LivermoreStateMachine, atr14 as lsm_atr
    d = df[["high", "low", "close"]].astype(float)
    atr = lsm_atr(d, 14).values
    c = d["close"].values
    m = LivermoreStateMachine(asset="CAL", timeframe="TEST", major_mult=major, minor_mult=minor, dual_confirm=dual, atr_period=14)
    out = np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        try:
            st = str(m.update(float(c[i]), float(atr[i])).state)
        except Exception:
            st = ""
        out[i] = 1 if st in UPS else (-1 if st in DOWNS else 0)
    return out


def hit_rate(fam, c, H, lo, hi):
    n_hit = n = 0
    for i in range(max(lo, 0), min(hi, len(c) - H)):
        r = c[i + H] - c[i]
        if fam[i] != 0 and r != 0:
            n += 1
            n_hit += (np.sign(r) == fam[i])
    return (100.0 * n_hit / n if n else float("nan")), n


def run_research(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nB13 RESEARCH RUNS -- R1 the tight spring (and B13 final all together) | R2 daily/weekly brain calibration\n" + "=" * 110)
    allrows = {v: [] for v in R_ROWS}
    spr = {"tight spring alone": [], "tight spring + diagonal": []}
    trades = []                                   # (asset, entry time, direction, R) of B13-final trades, for R2
    per, spans = {}, []
    tally = {v: collections.Counter() for v in R_ROWS}
    info = collections.Counter()
    cal_rows = []
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        t4c = (d4.index + pd.Timedelta(hours=4)).values
        hi4, lo4, c4 = (d4[x].astype(float).values for x in ("high", "low", "close"))
        a4 = M.atr14(hi4, lo4, c4)
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        dbr = [(x["d"], x["brk"]) for x in diag_lines(M, d4) if x["brk"] is not None]
        try:
            b4t, _b4s, b4 = brain_lines(a, d4, "4H")
        except Exception as ex:
            b4t, b4 = None, None
            say("   %-7s the 4H brain could not be run (%s) -- R1 uses 'held before' only" % (a, ex))
        t1v = A["t1"].values
        # ---- R1: the tight spring ------------------------------------------------------------------------------
        S = {"tight spring alone": [], "tight spring + diagonal": []}
        seen = set()
        for x in range(2, len(c4)):
            if not (a4[x] == a4[x] and a4[x] > 0):
                continue
            top = int(np.searchsorted(L.conf, t4c[x - 1], side="right"))
            t0 = t4c[x - 1] - np.timedelta64(AHEAD_DAYS, "D")
            for y in range(top - 1, -1, -1):
                cf, ty, lv, _i, _ed = L.sw[y]
                if cf < t0:
                    break
                d = 1 if ty == "L" else -1
                if d * (c4[x] - lv) <= 0 or d * (c4[x - 1] - lv) > 0:
                    continue
                xs = None
                for z in range(x - 1, max(0, x - 1 - 3), -1):          # QUICK: the sweep began within 3 candles
                    if d * (c4[z] - lv) < 0 and (z == 0 or d * (c4[z - 1] - lv) >= 0):
                        xs = z
                        break
                if xs is None or (d, round(lv, 8), xs) in seen:
                    continue
                if max(d * (lv - c4[z]) for z in range(xs, x)) > 1.0 * a4[x]:
                    continue                                          # SHALLOW: a deeper drop is a real breakdown
                held = any(ty2 == ty and abs(lv2 - lv) <= 0.25 * a4[x] and cf2 < cf and cf2 >= cf - np.timedelta64(AHEAD_DAYS, "D")
                           for cf2, ty2, lv2, _i2, _e2 in L.sw[max(0, y - 40):y])
                onbrain = False
                if b4 is not None:
                    kb = int(np.searchsorted(b4t, t4c[x - 1], side="right")) - 1
                    if kb >= 0:
                        onbrain = any(b4[k][kb] == b4[k][kb] and abs(b4[k][kb] - lv) <= 0.25 * a4[x] for k, _nm in ZN_KEYS)
                if not (held or onbrain):
                    continue                                          # QUALITY supports only
                seen.add((d, round(lv, 8), xs))
                info["R1 tight springs found"] += 1
                trap = lo4[xs:x + 1].min() if d == 1 else hi4[xs:x + 1].max()
                e = float(c4[x])
                tt = pd.Timestamp(t4c[x])
                j = int(np.searchsorted(t1v, np.datetime64(tt), side="right")) - 1
                atr1 = float(A["atr"][j]) if j >= 0 else float("nan")
                if not (atr1 == atr1 and atr1 > 0):
                    continue
                stop = float(trap - d * 0.1 * a4[x])
                if d * (e - stop) < 0.3 * atr1:
                    stop = e - d * 0.3 * atr1
                if d * (e - stop) > 5 * atr1 or not M.gate_rr_ok(e, stop, atr1, cfg["min_rr"]):
                    continue
                tgt = None if cfg["target_atr"] is None else e + d * float(cfg["target_atr"]) * atr1
                k = int(np.searchsorted(T.tv, np.datetime64(tt)))
                if k >= T.n or T.t[k] != tt:
                    continue
                c = dict(k=k, q=e, stop=stop, tgt=tgt, d=d)
                S["tight spring alone"].append(c)
                if any(dd == d and 0 <= x - b <= 6 for dd, b in dbr):
                    S["tight spring + diagonal"].append(c)
                    info["R1 tight springs with a diagonal"] += 1
        # ---- B13 final: package A + wick lines + the diagonal confirmation rule --------------------------------
        V = {v: [] for v in R_ROWS}
        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                continue
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            atr = atr41(M, A, iE)
            if not (atr == atr and atr > 0):
                continue
            lw, nw = L.ahead_wick(d, eE, tE)
            is_near = lw is not None and nw is not None and nw <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            lane = any(dd == d and tE - pd.Timedelta(hours=24) <= pd.Timestamp(t4c[b]) <= tE for dd, b in dbr)
            info["B13 final signals"] += 1
            info["... with an old high/low in front (wick)"] += is_near
            info["... taken at once by the diagonal rule"] += bool(is_near and lane and not far)
            if not far and (not is_near or lane):
                struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                    continue
                k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                if k >= T.n or T.t[k] != tE:
                    continue
                for v in R_ROWS:
                    V[v].append(dict(k=k, q=eE, stop=float(stop), tgt=tgt, d=d))
                continue
            nmaj = (BW.read(tE, d) == 1) + (BD.read(tE, d) == 1) + strong4h(F4, tE, d)
            if nmaj < 2:
                continue
            level = (max(trg, lw) if d == 1 else min(trg, lw)) if is_near else trg
            out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, level, "A", 24, False)
            if out != "enter":
                continue
            q, tau = float(T.c[k]), T.t[k]
            j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
            atr2 = atr41(M, A, j1)
            if not (atr2 == atr2 and atr2 > 0):
                continue
            stop, why = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
            if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                continue
            tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
            for v in R_ROWS:
                V[v].append(dict(k=k, q=q, stop=stop, tgt=tgt, d=d))
        V["B13 final + spring"] = V["B13 final + spring"] + S["tight spring + diagonal"]
        t0, t1 = A["t1"][30], A["t1"][-1]
        spans.append((t0, t1))
        per[a] = {}
        for v, cands in V.items():
            rows = book30(M, A, T, cands, runner, cost, usd)
            per[a][v] = stats(rows, t0, t1)
            allrows[v] += rows
        # the B13-final trades with their entry times, for R2's separation check
        free = -1
        for c in sorted(V["B13 final"], key=lambda x: x["k"]):
            if c["k"] <= free:
                continue
            kk, px, risk0 = sim30(M, A, T, c["d"], c["k"], c["q"], c["stop"], c["tgt"], runner)
            pts = c["d"] * (px - c["q"]) - cost * c["q"]
            trades.append((a, T.t[c["k"]], c["d"], pts / risk0 if risk0 > 0 else 0.0))
            free = kk
        for nm, cands in S.items():
            spr[nm] += book30(M, A, T, cands, runner, cost, usd)
        # ---- R2: calibrate the daily and weekly brains ------------------------------------------------------
        p0 = (PIV or {}).get(a, {}) or {}
        dflt = (p0.get("major_mult", 3.5), p0.get("minor_mult", 1.0), p0.get("dual_confirm", 2))
        for tf, df in (("D1", D1), ("W1", W1)):
            if df is None or len(df) < 120:
                continue
            c = df["close"].astype(float).values
            n, H = len(c), CAL_H[tf]
            half = n // 2
            best = None
            for g in CAL_GRID:
                fam = fam_series(df, *g)
                flips = int(np.sum((fam[1:half] != fam[:half - 1]) & (fam[1:half] != 0)))
                if flips < 4:
                    continue
                h1, n1 = hit_rate(fam, c, H, 0, half)
                if best is None or h1 > best[1]:
                    best = (g, h1, fam)
            famd = fam_series(df, *dflt)
            hd2, nd2 = hit_rate(famd, c, H, half, n)
            if best is None:
                continue
            hb2, nb2 = hit_rate(best[2], c, H, half, n)
            end = (df.index + (pd.Timedelta(days=1) if tf == "D1" else pd.Timedelta(days=7))).values
            cal_rows.append(dict(a=a, tf=tf, best=best[0], h1=best[1], hb2=hb2, nb2=nb2, dflt=dflt, hd2=hd2, nd2=nd2,
                                 end=end, fam=best[2], famd=famd))
        say("   %-7s B13 final %d | + spring %d | tight springs %d (with a diagonal %d)" % (
            a, len(V["B13 final"]), len(V["B13 final + spring"]), len(S["tight spring alone"]), len(S["tight spring + diagonal"])))
    if not spans:
        return
    T0_, T1_ = min(x[0] for x in spans), max(x[1] for x in spans)
    say("\n   HOW OFTEN (all markets): " + " | ".join("%s %d" % (k, n) for k, n in info.items()))
    say("\n   R1 + B13 FINAL -- the live system after B13 (package A, wick lines, the diagonal confirmation rule), and with the tight spring added:")
    say("   " + HEAD)
    for v in R_ROWS:
        say("   " + line(v[:15], stats(allrows[v], T0_, T1_)))
    say("\n   R1 -- THE TIGHT SPRING ON ITS OWN:")
    say("   " + HEAD)
    for nm, rows in spr.items():
        say("   " + (line(nm[:15], stats(rows, T0_, T1_)) if rows else "%-15s (none)" % nm[:15]) + "   <- " + nm)
    say("\n   EACH MARKET (B13 final / + spring):")
    for a in per:
        say("   %s" % a)
        for v in R_ROWS:
            say("     " + line(v[:15], per[a][v]))
    say("\n   R2 -- DAILY AND WEEKLY BRAIN CALIBRATION: how often the brain's side matched the next %d days (daily) / %d weeks"
        " (weekly). Best setting chosen on the FIRST half, judged on the SECOND half, against the current settings:" % (CAL_H["D1"], CAL_H["W1"]))
    say("   %-7s %-3s %-22s %8s %8s %6s | %-16s %8s" % ("market", "tf", "best (major/minor/dual)", "1st half", "2nd half", "n", "current", "2nd half"))
    for r in cal_rows:
        say("   %-7s %-3s %-22s %7.1f%% %7.1f%% %6d | %-16s %7.1f%%" % (
            r["a"], r["tf"], "%.2g / %.2g / %d" % r["best"], r["h1"], r["hb2"], r["nb2"], "%.2g / %.2g / %d" % r["dflt"], r["hd2"]))
    for tf in ("D1", "W1"):
        rr = [r for r in cal_rows if r["tf"] == tf and r["hb2"] == r["hb2"] and r["hd2"] == r["hd2"]]
        if rr:
            say("   %s average on the SECOND half: calibrated %.1f%% vs current %.1f%% (%d markets)" % (
                tf, float(np.mean([r["hb2"] for r in rr])), float(np.mean([r["hd2"] for r in rr])), len(rr)))
    say("\n   R2 -- B13-FINAL TRADES WHEN THE CALIBRATED / CURRENT DAILY AND WEEKLY BRAIN AGREED vs NOT (R per trade):")
    for tf in ("D1", "W1"):
        for lab, key in (("calibrated", "fam"), ("current", "famd")):
            agree, against = [], []
            for a, te, d, R in trades:
                r = next((x for x in cal_rows if x["a"] == a and x["tf"] == tf), None)
                if r is None:
                    continue
                kx = int(np.searchsorted(r["end"], np.datetime64(pd.Timestamp(te)), side="right")) - 1
                if kx < 0:
                    continue
                f = r[key][kx]
                (agree if f == d else against if f == -d else []).append(R)
            if agree or against:
                say("   %s %-10s agreed: %4d trades, %+.2fR each | against: %4d trades, %+.2fR each" % (
                    tf, lab, len(agree), float(np.mean(agree)) if agree else float("nan"),
                    len(against), float(np.mean(against)) if against else float("nan")))



# =====================================================================================================================
# B13 WEEKLY FORWARD TEST (items 24 and 28, Desire 5 Oct): all 10 markets, TODAY's rules -- entry E, package A, wick
# lines, the diagonal confirmation rule -- replayed on the real candles; lists every trade the rules took in the last
# FW_DAYS days (priced on 30m candles, smallest lot) next to what MT5 actually traded in the same days.
# =====================================================================================================================
FW_DAYS = int(os.environ.get("FW_DAYS", "7"))


B13PKG_SINCE = pd.Timestamp("2026-10-02 21:50")     # entry E + the package went live with the restart at 23:52 box time
                                                    # on 2 Oct = 21:52 MT5 time (the box clock runs 2 h ahead: the 2 Oct
                                                    # close-all logs at 16:19, MT5 shows 14:19) -- Stephen's 5 Oct check
TRACE_BEFORE_H = 26                                 # a live entry can come up to ~25 h after its signal (package)


def _votes(BW, BD, F4, tE, d):
    w, dd, h4 = BW.read(tE, d), BD.read(tE, d), strong4h(F4, tE, d)
    lab = {1: "with", -1: "against", 0: "mixed"}
    n = (w == 1) + (dd == 1) + int(bool(h4))
    return n, "weekly %s, daily %s, 4H %s" % (lab.get(w, w), lab.get(dd, dd), "strong" if h4 else "not strong")


def run_forward(data, frames, pcfg, costs, lots):
    M = load_engine("4H/1H", False)
    say("\n" + "=" * 110 + "\nB13 WEEKLY FORWARD TEST -- today's rules on all markets, last %d days, vs what MT5 traded\n" % FW_DAYS + "=" * 110)
    if not data:
        say("   NO PRICE FILES COULD BE READ (data\\raw\\*_1h.csv) -- nothing to replay; refresh the files and run again")
        return
    now = max(df.index[-1] for df in data.values()) + pd.Timedelta(hours=1)
    since = now - pd.Timedelta(days=FW_DAYS)
    rows_all, trace = [], []
    for a, df1 in data.items():
        cfg = M.market_settings(a, pcfg)
        fr = frames.get(a, {})
        if cfg is None or fr.get("M30") is None or len(fr["M30"]) < 1000:
            say("   %-7s skipped (no settings or no 30m candles)" % a)
            continue
        A = arrays(df1, False, M)
        T = Thirty(fr["M30"])
        cost, usd, runner = costs.get(a, 0.0005), lots[a][0], cfg["exit"] == "RUNNER"
        pE = json.loads(json.dumps(pcfg or {}, default=str))
        pE.setdefault("ns_markets", {}).setdefault(a, {})
        pE["ns_markets"][a] = dict(pE["ns_markets"][a], entry="E")
        # B14: this replay re-runs the B13 entry rules (decide() below) -- with the B14 switch on it would look for a
        # line map that only the live bot builds. What the B14 rules did live is listed under B14 TRACKING.
        pE["b14_rules_enabled"] = False
        cfgE = M.market_settings(a, pE)
        engE, ends, alive = run_engine(M, a, df1, pE)
        setups = {s["id"]: s for s, _r, _t in ends}
        for s in alive:
            setups[s["id"]] = s
        d4 = M.hourly_to_4h(df1)
        t4c = (d4.index + pd.Timedelta(hours=4)).values
        L = Levels(M, d4)
        F4 = TF(M, d4, 4)
        D1, W1 = fr.get("D1"), fr.get("W1")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if D1 is None:
            D1 = df1.resample("1D").agg(agg).dropna()
        if W1 is None:
            W1 = df1.resample("W-SUN", label="left", closed="left").agg(agg).dropna()
        BW, BD = Bigger(W1, pd.Timedelta(days=7)), Bigger(D1, pd.Timedelta(days=1))
        dbr = [(x["d"], x["brk"]) for x in diag_lines(M, d4) if x["brk"] is not None]
        t1v = A["t1"].values
        C = []

        def decide(s, d, r2, r1, tt, iE, trg, istart, tE, eE, atr, rules):
            """What one rule set does with one entry-E signal -> (text, candidate or None).
            B13  = old high/low in front judged on WICKS, the diagonal confirmation rule, package version A
            LIVE = what ran 2-6 Oct: old high/low on CLOSES, no diagonal rule, package middle way, far package
                   entries refused at sending (the freshness check B13 item 30 removes)"""
            if rules == "B13":
                lv, dist = L.ahead_wick(d, eE, tE)
            else:
                lv, _ed, dist = L.ahead_level(d, eE, tE)
            near = lv is not None and dist is not None and dist <= NEAR
            far = d * (eE - r2) / atr > M.FRESH_ATR
            brk = [pd.Timestamp(t4c[b]) for dd, b in dbr if dd == d and tE - pd.Timedelta(hours=24) <= pd.Timestamp(t4c[b]) <= tE]
            lane = rules == "B13" and bool(brk)
            hl = "high" if d == 1 else "low"
            if not far and (not near or lane):
                struct = float(A["lo"][istart:iE + 1].min()) if d == 1 else float(A["hi"][istart:iE + 1].max())
                stop, tgt = stop_for(M, cfgE, d, eE, r2, atr, struct, None)
                strength = M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE])
                if stop is None or not gated(M, cfgE, s, "E", eE, atr, stop, strength):
                    return "REFUSED by the engine's own gates (stop / reward:risk / strength)", None
                why = ("diagonal rule: old %s %.6g just ahead, but a diagonal broke %s" % (hl, lv, str(max(brk))[5:16])
                       if near else ("space to run (nearest old %s %.6g, %.2f moves)" % (hl, lv, dist) if lv is not None
                                     else "space to run (no old %s ahead)" % hl))
                k = int(np.searchsorted(T.tv, np.datetime64(tE)))
                if k < T.n and T.t[k] == tE:
                    return "TAKEN AT ONCE at %.6g -- %s" % (eE, why), dict(
                        k=k, q=eE, stop=float(stop), tgt=tgt, d=d, kind=s.get("kind"),
                        how=("diagonal rule" if near else "at once"))
                return "TAKEN AT ONCE -- %s (no 30m candle to price it)" % why, None
            why = ("FAR entry (%.1f moves from the line)" % (d * (eE - r2) / atr)) if far else \
                  ("old %s %.6g just ahead (%.2f moves, by %s)" % (hl, lv, dist, "wick" if rules == "B13" else "close"))
            nmaj, vtxt = _votes(BW, BD, F4, tE, d)
            if nmaj < 2:
                return "PACKAGE (%s) -> NO TRADE: bigger timeframes %d of 3 (%s)" % (why, nmaj, vtxt), None
            level = (max(trg, lv) if d == 1 else min(trg, lv)) if near else trg
            out, k, how2 = judge(A, T, F4, d, tE, r2, r1, trg, level, "A" if rules == "B13" else "MID", 24, False)
            if out != "enter":
                said = {"weak": "CANCELLED (the closes fell back)", "expired": "EXPIRED (24 h, never proven)",
                        "died": "DIED (a 4H close past the setup's origin)",
                        "watching": "still WATCHING at the end of the data"}.get(out, out)
                return "PACKAGE (%s; votes %d of 3) -> %s" % (why, nmaj, said), None
            q, tau = float(T.c[k]), T.t[k]
            j1 = int(np.searchsorted(t1v, np.datetime64(tau), side="right")) - 1
            atr2 = atr41(M, A, j1)
            if not (atr2 == atr2 and atr2 > 0):
                return "PACKAGE (%s) -> proven, but no move size to place a stop" % why, None
            stop, why2 = place_stop(M, T, cfg, d, k, q, r2, tt, atr2, "HL", True)
            if stop is None or not M.gate_rr_ok(q, stop, atr2, cfg["min_rr"]):
                return "PACKAGE (%s) -> proven %s, but NOT TAKEN: %s" % (why, str(tau)[5:16], why2 or "reward:risk"), None
            if rules == "LIVE" and far:
                return ("PACKAGE (%s) -> proven %s at %.6g, but REFUSED AT SENDING (freshness check -- B13 item 30 "
                        "removes this)" % (why, str(tau)[5:16], q)), None
            tgt = None if cfg["target_atr"] is None else q + d * float(cfg["target_atr"]) * atr2
            return "PACKAGE (%s; votes %d of 3) -> ENTERED %s at %.6g (%s)" % (why, nmaj, str(tau)[5:16], q, how2), dict(
                k=k, q=q, stop=stop, tgt=tgt, d=d, kind=s.get("kind"), how="package (%s)" % ("far" if far else "old high ahead"))

        for sid, s in setups.items():
            d = s["d"]
            if cfg.get("cont_only") and s["kind"] != "continuation":
                continue
            r2, r1 = float(s["r2"]), float(s["r1"])
            tt = s.get("t_touch")
            if tt is None:
                continue
            got = ct_trigger(M, s, A, "E", L.atr_at)
            if got is None:
                continue
            iE, trg, istart = got
            tE, eE = A["t1"][iE], float(A["cl"][iE])
            if tE < since - pd.Timedelta(days=2):
                continue
            if cfg.get("no_spike") and M.candle_strength(d, A["hi"][iE], A["lo"][iE], A["cl"][iE]) > M.SPIKE:
                if tE >= since - pd.Timedelta(hours=TRACE_BEFORE_H):
                    trace.append(dict(a=a, d=d, t=tE, r2=r2, e=eE, kind=s.get("kind"), b13="SKIPPED: spike candle",
                                      live="SKIPPED: spike candle"))
                continue
            atr = atr41(M, A, iE)
            if not (atr == atr and atr > 0):
                continue
            txt_b, cand = decide(s, d, r2, r1, tt, iE, trg, istart, tE, eE, atr, "B13")
            txt_l, _c = decide(s, d, r2, r1, tt, iE, trg, istart, tE, eE, atr, "LIVE")
            if cand is not None:
                C.append(cand)
            if tE >= since - pd.Timedelta(hours=TRACE_BEFORE_H):
                trace.append(dict(a=a, d=d, t=tE, r2=r2, e=eE, kind=s.get("kind"), b13=txt_b, live=txt_l))
        free = -1
        for c in sorted(C, key=lambda x: x["k"]):
            if c["k"] <= free:
                continue
            kk, px, risk0 = sim30(M, A, T, c["d"], c["k"], c["q"], c["stop"], c["tgt"], runner)
            free = kk
            if T.t[c["k"]] < since:
                continue
            pts = c["d"] * (px - c["q"]) - cost * c["q"]
            rows_all.append((a, "BUY" if c["d"] == 1 else "SELL", T.t[c["k"]], c["q"], c["stop"], T.t[kk], px,
                             pts / risk0 if risk0 > 0 else 0.0, pts * usd, c["how"], c.get("kind")))
    say("\n   THE RULES' TRADES, %s to %s (entries in the window; exits may be later or still open at the end of data):"
        % (str(since)[:16], str(now)[:16]))
    say("   %-7s %-4s %-16s %-11s %-11s %-16s %-11s %6s %9s  %s" % ("market", "side", "entry time", "entry", "stop",
                                                                 "exit time", "exit", "R", "$", "how / setup"))
    for r in sorted(rows_all, key=lambda x: x[2]):
        say("   %-7s %-4s %-16s %-11.6g %-11.6g %-16s %-11.6g %+6.2f %+9.2f  %s / %s" % (
            r[0], r[1], str(r[2])[:16], r[3], r[4], str(r[5])[:16], r[6], r[7], r[8], r[9], r[10]))
    say("   TOTAL: %d trades | %+.2f R | %+.2f $ at the smallest lot" % (
        len(rows_all), sum(r[7] for r in rows_all), sum(r[8] for r in rows_all)))
    live_pos = []
    try:
        import MetaTrader5 as _mt5
        _mt5.initialize()
        deals = _mt5.history_deals_get(since.to_pydatetime(), (now + pd.Timedelta(days=1)).to_pydatetime()) or []
        pos = {}
        for dl in deals:
            p = pos.setdefault(dl.position_id, {"sym": dl.symbol, "in": None, "out": None, "profit": 0.0})
            p["profit"] += float(dl.profit) + float(getattr(dl, "commission", 0.0)) + float(getattr(dl, "swap", 0.0))
            if dl.entry == 0:
                p["in"] = dl
            elif dl.entry == 1:
                p["out"] = dl
        say("\n   WHAT MT5 ACTUALLY TRADED in the same days (closed positions):")
        tot = 0.0
        for pid, p in sorted(pos.items(), key=lambda kv: (kv[1]["in"].time if kv[1]["in"] else 0)):
            if p["in"] is None:
                continue
            tot += p["profit"] if p["out"] is not None else 0.0
            t_in = pd.Timestamp(p["in"].time, unit="s")
            live_pos.append((p["sym"], 1 if p["in"].type == 0 else -1, t_in, float(p["in"].price), p["profit"]))
            say("   %-10s %-4s opened %s at %-11.6g %s  profit %+.2f" % (
                p["sym"], "BUY" if p["in"].type == 0 else "SELL", t_in.strftime("%Y-%m-%d %H:%M"), p["in"].price,
                ("closed %s at %.6g" % (pd.Timestamp(p["out"].time, unit="s").strftime("%Y-%m-%d %H:%M"), p["out"].price))
                if p["out"] is not None else "STILL OPEN", p["profit"]))
        say("   MT5 TOTAL (closed): %+.2f $" % tot)
    except Exception as ex:
        say("   (MT5 history not available here: %s)" % ex)
    # ---- the trace (Desire 7 Oct): for every live trade, which rule passed on it or took it, and why ----------------
    sym2a = {v: k for k, v in MARKETS.items()}
    sym2a.update({v.rstrip("m"): k for k, v in MARKETS.items()})
    say("\n   TRACE 1 -- EVERY LIVE MT5 TRADE AND WHAT EACH RULE SET DID WITH ITS SIGNAL (signals up to %d h before the open)"
        % TRACE_BEFORE_H)
    say("   B13 = after B13 (wick lines, diagonal rule, package A) | LIVE = the rules that ran 2-6 Oct (closes, middle way)")
    if not live_pos:
        say("   (no MT5 history here)")
    for sym, d, t_in, px, prof in live_pos:
        a = sym2a.get(sym, sym2a.get(str(sym).rstrip("m"), sym))
        say("   %s %s opened %s at %.6g (%+.2f)" % (a, "BUY" if d == 1 else "SELL", t_in.strftime("%m-%d %H:%M"), px, prof))
        hits = [r for r in trace if r["a"] == a and r["d"] == d and t_in - pd.Timedelta(hours=TRACE_BEFORE_H) <= r["t"] <= t_in + pd.Timedelta(hours=1)]
        if not hits:
            say("      no entry-E signal from the engine in the %d h before -- %s" % (
                TRACE_BEFORE_H, "the live bot used the OLD entries then (before the switch to entry E on 2 Oct)"
                if t_in < B13PKG_SINCE else "this trade did not come from an entry-E signal: check the log at that time"))
        for r in hits:
            say("      signal %s  line %.6g  E close %.6g  (%s)" % (r["t"].strftime("%m-%d %H:%M"), r["r2"], r["e"], r["kind"]))
            say("        B13 : %s" % r["b13"])
            say("        LIVE: %s" % r["live"])
    say("\n   TRACE 2 -- EVERY ENTRY-E SIGNAL IN THE WINDOW (both rule sets):")
    for r in sorted(trace, key=lambda x: x["t"]):
        if r["t"] < since:
            continue
        say("   %-7s %-4s %s  line %-10.6g E %-10.6g %s" % (r["a"], "BUY" if r["d"] == 1 else "SELL", r["t"].strftime("%m-%d %H:%M"),
                                                          r["r2"], r["e"], r["kind"]))
        say("        B13 : %s" % r["b13"])
        say("        LIVE: %s" % r["live"])
    say("\n   READ: a live trade with no B13 entry shows which B13 rule passed on it; 'LIVE: TAKEN' with no MT5 trade means a"
        " live-only check stopped it (health, session, council, MT5) -- look in the log at that time.")


# =====================================================================================================================
# B14 item 9.3 (Desire 9 Oct): the witness, the lock and the staircase, from the bot's own log lines
# =====================================================================================================================
B14_TAGS = ("[NS-ROUTE]", "[PKG-HANDOVER]", "[PKG-ENTER]", "[PKG-CANCEL]", "[PKG-REARM]", "[PKG-OUTCOME]", "[NS-PROOF]",
            "[NS-LOCK]", "[NS-STAIR]", "[NS-STAIR-FAIL]", "[NS-WALLS]", "[RUNNER-RUN]", "[NS-REBUILD]", "[NS-MAP]",
            "[NS-STOP-HOLD]")


def b14_tracking(days, pcfg):
    """Reads the last `days` of logs\trading_bot.log* and lists, market by market and in time order, what the B14 rules
    did: each signal taken at once (with its witness), each hand-over (why it waited), each package outcome, each
    re-arm, each RUNNER run, each wall lock and staircase step -- and any line saying one of them failed."""
    say("\n" + "=" * 110 + "\nB14 TRACKING -- the witness, the lock and the staircase (the bot's own log lines, last %d days, box time)\n" % days + "=" * 110)
    say("   new rules switch (phase_config.b14_rules_enabled): %s" % ("ON" if (pcfg or {}).get("b14_rules_enabled") else
                                                                       "OFF -- no witness / lock / staircase lines are expected yet"))
    since = pd.Timestamp.now() - pd.Timedelta(days=days)
    rx = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    ev = collections.defaultdict(list)
    cnt = collections.Counter()
    for f in sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime):
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "[N" not in line and "[P" not in line and "[R" not in line:
                    continue
                m = rx.match(line)
                if not m:
                    continue
                t = pd.Timestamp(m.group(1))
                if t < since:
                    continue
                msg = line.split(" - ", 3)[-1].strip()
                for tag in B14_TAGS:
                    if msg.startswith(tag):
                        a = msg[len(tag):].strip().split(":", 1)[0].split()[0] if ":" in msg else "?"
                        ev[a].append((t, tag, msg))
                        cnt[tag] += 1
                        break
    if not ev:
        say("   no B14 lines in the last %d days" % days)
        return
    say("   counts: " + ", ".join("%s %d" % kv for kv in sorted(cnt.items())))
    bad = cnt["[NS-STAIR-FAIL]"] + cnt["[NS-WALLS]"] + cnt["[NS-MAP]"] + sum(
        1 for a in ev for _t, tag, msg in ev[a] if tag == "[NS-STOP-HOLD]" and "ns_wall" in msg)
    say("   failures (stair not fed / no map / a lock or step held by the guard): %d %s" % (bad, "" if not bad else "<-- tell Claude"))
    for a in sorted(ev):
        say("\n   %s" % a)
        for t, tag, msg in sorted(ev[a]):
            if tag == "[NS-STOP-HOLD]" and "ns_wall" not in msg:
                continue
            say("      %s  %s" % (str(t)[5:16], msg[:200]))


def main():
    say("=== TBOT B13 WEEKLY FORWARD TEST (read-only) -- today's rules, all markets, last %d days vs MT5 ===" % FW_DAYS)
    try:
        sha = hashlib.sha256(open(ENGINE, "rb").read().replace(b"\r\n", b"\n")).hexdigest().upper()   # same fingerprint whatever the line endings
    except OSError as e:
        say("   cannot read %s (%s) -- run this from C:\\TradingBot\\TBOT" % (ENGINE, e))
        return
    say("   engine: %s" % ("the B14 engine (9 Oct) -- the same code this test was built against" if sha == ENGINE_SHA else
                           "DIFFERENT from the version this test was built against (%s...) -- results may not match the bot; tell Claude" % sha[:16]))
    say("   %s" % stub_brain())
    try:
        pcfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig")).get("phase_config", {})
        say("   live settings: config/config.json phase_config (%d market override(s))" % len(pcfg.get("ns_markets") or {}))
    except Exception as e:
        pcfg = {}
        say("   config/config.json not read (%s) -- engine defaults used" % e)
    if pcfg.get("b14_rules_enabled"):
        say("   NOTE: the B14 rules are ON live. The replay below re-runs the B13 entry rules for comparison; what the B14"
            " rules did (routes, witnesses, package outcomes, locks, staircase steps) is listed under B14 TRACKING at the end.")
    costs, live_proofs = read_logs()
    lots, spreads = lot_values()
    data, frames = {}, {}
    for a, sym in MARKETS.items():
        if a not in costs and spreads.get(a):
            costs[a] = spreads[a]
        df, note = load_1h(sym)
        fr = mt5_frames(sym)
        frames[a] = fr
        cov = " | ".join("%s %d from %s" % (k, len(v), str(v.index[0])[:10]) for k, v in fr.items()) or "MT5 frames not available"
        say("   %-7s %s | cost %.5f | lot $%.4g per 1.0 | %s" % (a, note, costs.get(a, 0.0005), lots[a][0], cov))
        if df is not None and len(df) > 500:
            data[a] = df
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass
    global PIV
    try:
        PIV = json.load(open(os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig")).get("LIVERMORE_PIVOTS") or {}
    except Exception:
        PIV = {}
    run_forward(data, frames, pcfg, costs, lots)
    say("\n   RULES: entry E = 4H break, 1H retest, first 1H close past the peak | at once = no old 4H high/low within half a 4H"
        " move ahead (B13: by wick; LIVE: by close) | diagonal rule (B13) = a 3-touch diagonal broke the same way in the 24 h"
        " before -> taken at once anyway | package = 2 of 3 bigger timeframes, then the 1H and two 30m closes past the old"
        " high/low within 24 h (B13: version A; LIVE: middle way) | priced on 30m candles at the smallest lot")
    b14_tracking(FW_DAYS, pcfg)                                   # B14 item 9.3
    say("\n=== END OF THE B13 WEEKLY FORWARD TEST (%.0f s) ===" % (time.time() - T0))


if __name__ == "__main__":
    main()
