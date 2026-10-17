"""
REALISTIC TEST -- the same setups as tools/final_test.py, but trading the way the live bot can: ONE POSITION PER
MARKET. A trade is only taken if that market has no trade still open. Read-only.

  TODAY   today's rule (E) + the simple exit, every market
  A       each market's own settings, no filters
  D       A + continuations only + no spike entries, every market
  MIX     the recommended mix:
            EURUSD, GBPAUD      today's rule + simple exit + continuations only + no spikes
            GOLD, USOIL, USTEC  break-close entry + 4-ATR target, no filters
            BTC                 today's rule + runner exit, no filters
  MIX-noBTC  the same, without BTC

Each setup is shown twice: "counted" = every trade (as in the earlier tests), "one at a time" = only trades the
bot could actually hold. AGREED IN ADVANCE: a setup replaces today's only if, ONE AT A TIME, it beats TODAY
(also one at a time) after costs in BOTH halves AND makes more R a week.
Built-in checks (counted): TODAY 574 trades +0.09R (+0.85R a week), A +3.22R a week, D +1.76R a week.
Still before the council and the $5 minimum; slippage not modelled. Chosen on this same history -- the final
proof is new weeks of data.
"""
import os, sys, json, glob, re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYM = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm", "EURUSD": "EURUSDm", "USOIL": "USOILm", "GBPAUD": "GBPAUDm"}
START, END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-09-24")
SPLIT = pd.Timestamp("2026-02-10")
BOX_OFFSET_H = 2
K, MIN_SWING, TOL, ALLOW, GATE, TGT, STOP_CAP = 2, 1.0, 1.0, 0.3, 2.5, 2.5, 5.0
WAIT_BREAK, WAIT_RETEST, WAIT_THROUGH, TRADE_DAYS = 10, 7, 7, 7
MIN_RR = {"BTC": 0.6, "GOLD": 0.48, "USTEC": 0.67, "EURUSD": 0.5, "USOIL": 0.48, "GBPAUD": 0.5}
MIN_SL_PCT = {"BTC": 0.006, "GOLD": 0.0035, "USTEC": 0.0025, "EURUSD": 0.0012, "USOIL": 0.0055, "GBPAUD": 0.0025}
LIVE = {a: dict(be_bars=2, be_pct=0.01, tsb=72) for a in SYM}
PHASE = dict(r_trigger=1.0, r_lock=0.2, max_tgt=2.5, struct=True)
try:
    _c = json.load(open("config/config.json", encoding="utf-8-sig"))
    _pc = _c.get("phase_config") or {}
    PHASE.update(r_trigger=float(_pc.get("r_breakeven_trigger", 1.0)), r_lock=float(_pc.get("r_breakeven_lock", 0.2)),
                 max_tgt=float(_pc.get("max_target_atr", 2.5)), struct=bool(_pc.get("structural_trailing_enabled", True)))
    for a in SYM:
        r = (((_c.get("assets") or {}).get(a) or {}).get("risk") or {})
        MIN_RR[a] = float(r.get("min_rr", MIN_RR[a]))
        MIN_SL_PCT[a] = float(r.get("min_sl_pct", MIN_SL_PCT[a]))
        tsb = r.get("time_stop_trend", r.get("time_stop_bars", r.get("max_hold_bars", 72)))
        LIVE[a] = dict(be_bars=int(r.get("breakeven_after_bars") or 2), be_pct=float(r.get("breakeven_profit_threshold") or 0.01),
                       tsb=int(tsb or 72))
except Exception as ex:
    print(f"  (config not read, using defaults: {ex})")

PAT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2}.*?\[FRICTION\] (\w+): used=([0-9.]+) \([^)]*\) observed=([0-9.]+)")
obs = defaultdict(list)
for fpath in sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime):
    try:
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[FRICTION]" not in line or "NO_DATA" in line:
                    continue
                m = PAT.search(line)
                if m and m.group(2) in SYM:
                    h = (datetime.strptime(m.group(1), "%Y-%m-%d %H") - timedelta(hours=BOX_OFFSET_H)).hour
                    obs[(m.group(2), h)].append(float(m.group(4)))
    except Exception as ex:
        print(f"  ({fpath}: {ex})")
def _med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None
COST = {}
for a in SYM:
    allv = [x for (aa, h), v in obs.items() if aa == a for x in v]
    for h in range(24):
        COST[(a, h)] = _med(obs.get((a, h), [])) or _med(allv) or 0.0005


def load(sym, tf):
    df = pd.read_csv(f"data/raw/{sym}_{tf}.csv", parse_dates=[0], index_col=0)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def swings(h4):
    hi, lo, cl, n = h4["high"].values, h4["low"].values, h4["close"].values, len(h4)
    out = []
    for i in range(K, n - K):
        conf = h4.index[i + K] + timedelta(hours=4)
        if cl[i] == cl[i - K:i + K + 1].max():
            out.append((conf, "H", float(cl[i]), float(hi[i - K:i + K + 1].max())))
        if cl[i] == cl[i - K:i + K + 1].min():
            out.append((conf, "L", float(cl[i]), float(lo[i - K:i + K + 1].min())))
    return sorted(out)


def pivots(hi, lo, cl, atr):
    """The live trail's swing points: 4 candles on the left, 2 on the right, at least 0.3 ATR deep."""
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


def exit_run(asset, mode, d, e, stop0, i0, t1, hi1, lo1, c1, a1, piv, atr0):
    """One trade, one way out. Returns (R, reason)."""
    risk0 = abs(e - stop0)
    stop, reason_stop = stop0, "stop"
    live = mode in ("LIVE", "LIVE_NOEARLY", "RUNNER")
    early = mode == "LIVE"
    if mode in ("SIMPLE", "LIVE", "LIVE_NOEARLY"):
        tgt = e + d * min(TGT, PHASE["max_tgt"]) * atr0
    elif mode == "SIMPLE4":
        tgt = e + d * 4.0 * atr0
    else:
        tgt = None
    tsb = LIVE[asset]["tsb"] if mode in ("LIVE", "LIVE_NOEARLY") else 24 * TRADE_DAYS
    extended, armed, peak = False, False, e
    cfg = LIVE[asset]
    last = min(len(t1) - 1, i0 + tsb + 30)
    for j in range(i0 + 1, last + 1):
        bars = j - i0
        # exits first (a stop hit and a target in the same candle count as the stop)
        if (d == 1 and lo1[j] <= stop) or (d == -1 and hi1[j] >= stop):
            return d * (stop - e) / risk0, reason_stop, j
        if tgt is not None and ((d == 1 and hi1[j] >= tgt) or (d == -1 and lo1[j] <= tgt)):
            return d * (tgt - e) / risk0, "target", j
        if not live:
            if bars >= tsb:
                return d * (c1[j] - e) / risk0, "time", j
            continue
        atr = a1[j] if a1[j] == a1[j] else atr0
        peak = max(peak, hi1[j]) if d == 1 else min(peak, lo1[j])
        prof = d * (peak - e)
        cands = []
        if early:
            if prof > 0.75 * atr:
                cands.append((e - d * 0.75 * risk0, "soft risk-cut"))
            if prof > 1.0 * atr:
                cands.append((e - d * max(0.0, risk0 - 0.5 * atr), "intermediate trail"))
            if prof > 1.5 * atr:
                cands.append((e, "ATR break-even"))
            if bars >= cfg["be_bars"] and d * (c1[j] - e) / e >= cfg["be_pct"]:
                cands.append((e, "time break-even"))
        if prof >= PHASE["r_trigger"] * risk0:
            cands.append((e + d * PHASE["r_lock"] * risk0, "R-lock"))
            armed = True
        if armed and PHASE["struct"] and bars >= 10 and d * (c1[j] - e) > 0:
            lo_i = max(6, j - 200)
            idx = np.nonzero(piv[lo_i:j - 1])[0] + lo_i
            if len(idx):
                lv = (lo1[idx] - 0.3 * atr) if d == 1 else (hi1[idx] + 0.3 * atr)
                ok = lv[(lv < c1[j])] if d == 1 else lv[(lv > c1[j])]
                if len(ok):
                    cands.append((ok.max() if d == 1 else ok.min(), "trail"))
        for px, why in cands:
            if (d == 1 and px > stop) or (d == -1 and px < stop):
                stop, reason_stop = px, why
        if bars >= tsb:
            if d * (c1[j] - e) > 0 and not extended and mode != "RUNNER":
                extended, tsb = True, tsb + 24
            else:
                return d * (c1[j] - e) / risk0, "time", j
    return d * (c1[last] - e) / risk0, "time", last


def run_market(asset):
    h1, h4 = load(SYM[asset], "1h"), load(SYM[asset], "4h")
    t1 = h1.index + timedelta(hours=1)
    c1, hi1, lo1, a1 = h1["close"].values, h1["high"].values, h1["low"].values, h1["atr"].values
    pl, ph = pivots(hi1, lo1, c1, a1)
    close4 = {ts + timedelta(hours=4): (float(c), float(a)) for ts, c, a in zip(h4.index, h4["close"], h4["atr"])}
    bar4 = {ts + timedelta(hours=4): (float(h), float(l), float(c))
            for ts, h, l, c in zip(h4.index, h4["high"], h4["low"], h4["close"])}
    hist, setups = {"H": [], "L": []}, []
    for conf, typ, lvl, edge in swings(h4):
        hist[typ].append((conf, lvl, edge))
        if not (START <= conf <= END):
            continue
        if typ == "L" and hist["H"] and hist["H"][-1][1] > lvl:
            H = hist["H"][-1]
            prev = hist["H"][-2][1] if len(hist["H"]) > 1 else None
            kind = "unknown" if prev is None else ("reversal" if prev > H[1] else "continuation")
            setups.append((conf, 1, H[1], H[2], lvl, kind))
        if typ == "H" and hist["L"] and hist["L"][-1][1] < lvl:
            L = hist["L"][-1]
            prev = hist["L"][-2][1] if len(hist["L"]) > 1 else None
            kind = "unknown" if prev is None else ("reversal" if prev < L[1] else "continuation")
            setups.append((conf, -1, L[1], L[2], lvl, kind))
    seen, rows = set(), []
    for conf, d, R2, EDGE, R1, kind in setups:
        if (d, round(R2, 6)) in seen:
            continue
        i0 = int(np.searchsorted(t1.values, np.datetime64(conf), side="right"))
        a4 = [v[1] for k, v in close4.items() if k <= conf][-1:]
        if i0 >= len(t1) or not a4 or not a4[0] or a4[0] != a4[0] or abs(R2 - R1) < MIN_SWING * a4[0]:
            continue
        got, stage, H2, iB, iR, paused = {}, 0, None, None, None, False
        reached = False
        for i in range(i0, len(t1)):
            t, atr = t1[i], a1[i]
            if atr != atr:
                continue
            r4 = close4.get(t)
            if r4 is not None and ((d == 1 and r4[0] < R1) or (d == -1 and r4[0] > R1)):
                break
            if stage == 0:
                if t - conf > timedelta(days=WAIT_BREAK):
                    break
                if r4 is not None and ((d == 1 and r4[0] > R2) or (d == -1 and r4[0] < R2)):
                    stage, iB, reached = 1, i, True
                    j = max(0, i - 3)
                    H2 = float(c1[j:i + 1].max()) if d == 1 else float(c1[j:i + 1].min())
                    got["B"] = (i, float(c1[i]), R2 - d * ALLOW * atr, float(atr))
                continue
            c = float(c1[i])
            if d * (c - c1[i - 1]) < 0:
                paused = True
            if "E2" not in got and paused and i - 3 >= 0 and d * (c - R2) > 0 \
                    and d * (c - (c1[i - 3:i].max() if d == 1 else c1[i - 3:i].min())) > 0:
                got["E2"] = (i, c, R2 - d * ALLOW * atr, float(atr))
            if stage == 1:
                if t - t1[iB] > timedelta(days=WAIT_RETEST):
                    break
                if (d == 1 and lo1[i] <= EDGE + TOL * atr) or (d == -1 and hi1[i] >= EDGE - TOL * atr):
                    stage, iR = 2, i
                    continue
                if d * (c - H2) > 0:
                    H2 = c
                continue
            if stage == 2:
                if t - t1[iR] > timedelta(days=WAIT_THROUGH):
                    break
                if d * (c - H2) > 0:
                    got["E"] = (i, c, R2 - d * ALLOW * atr, float(atr))
                    break
        if reached:
            seen.add((d, round(R2, 6)))
        for style, (i, e, stop, atr) in got.items():
            if d * (e - R2) / atr > GATE:
                continue
            floor = MIN_SL_PCT.get(asset, 0.005) * e
            if abs(e - stop) < floor:
                stop = e - d * floor
            if abs(e - stop) > STOP_CAP * atr:
                stop = e - d * STOP_CAP * atr
            if (d == 1 and stop >= e) or (d == -1 and stop <= e) or TGT * atr / abs(e - stop) < MIN_RR.get(asset, 0.5):
                continue
            cost = COST[(asset, t1[i].hour)] * e / abs(e - stop)
            piv = pl if d == 1 else ph
            hh, ll, cc = bar4.get(t1[i], (np.nan, np.nan, np.nan)) if style == "B" else (hi1[i], lo1[i], c1[i])
            rng_ = hh - ll
            strength = ((cc - ll) / rng_ if d == 1 else (hh - cc) / rng_) if (rng_ == rng_ and rng_ > 0) else 0.5
            for mode in MODES:
                R, why, jx = exit_run(asset, mode, d, e, stop, i, t1, hi1, lo1, c1, a1, piv, atr)
                rows.append(dict(asset=asset, entry=style, mode=mode, t=t1[i], exit_t=t1[jx], R=R, net=R - cost,
                                 kind=kind, strength=float(strength)))
    return rows


MODES = ("SIMPLE", "LIVE", "SIMPLE4", "RUNNER")
MNAME = {"SIMPLE": "simple", "LIVE": "live rules", "SIMPLE4": "4-ATR target", "RUNNER": "runner"}
ENAME = {"E": "E today's rule", "B": "B at the break close", "E2": "E2 pause, then turn"}
SETTINGS = {"GOLD": ("B", "SIMPLE4"), "USOIL": ("B", "SIMPLE4"), "USTEC": ("B", "SIMPLE4"),
            "BTC": ("E", "RUNNER"), "EURUSD": ("E", "SIMPLE"), "GBPAUD": ("E", "SIMPLE")}
FILTERED = ("EURUSD", "GBPAUD")               # the markets the mix filters
SPIKE = 0.85
rows = []
for asset in SYM:
    try:
        rows += run_market(asset)
    except Exception as ex:
        print(f"  ({asset}: {type(ex).__name__}: {ex})")
df = pd.DataFrame(rows)
if df.empty:
    print("no trades found"); sys.exit(1)
weeks = max(1.0, (END - START).total_seconds() / 86400 / 7)


def one_at_a_time(g):
    """Keep a trade only if its market has nothing still open -- the way the live bot trades."""
    keep = []
    for _, ga in g.sort_values("t").groupby("asset"):
        free_at = None
        for idx, r in ga.iterrows():
            if free_at is None or r["t"] >= free_at:
                keep.append(idx)
                free_at = r["exit_t"]
    return g.loc[keep]


def st(g):
    a_, b_ = g[g["t"] < SPLIT]["net"], g[g["t"] >= SPLIT]["net"]
    return dict(n=len(g), wk=len(g) / weeks, r=g.net.mean() if len(g) else float("nan"),
                h1=a_.mean() if len(a_) else float("nan"), h2=b_.mean() if len(b_) else float("nan"),
                rw=g.net.sum() / weeks)


today = df[(df.entry == "E") & (df["mode"] == "SIMPLE")]
mask = pd.Series(False, index=df.index)
for a, (e, m) in SETTINGS.items():
    mask |= (df.asset == a) & (df.entry == e) & (df["mode"] == m)
prof = df[mask]
spike = (prof.entry != "B") & (prof.strength > SPIKE)
cont = prof.kind == "continuation"
filt = prof.asset.isin(FILTERED)
mix = prof[~filt | (cont & ~spike)]
CONFIGS = [("TODAY      E + simple exit, every market", today),
           ("A          each market's own settings", prof),
           ("D          A + both filters, every market", prof[cont & ~spike]),
           ("MIX        the recommended mix", mix),
           ("MIX-noBTC  the mix without BTC", mix[mix.asset != "BTC"])]
print(f"window {START:%d %b %Y} -> {END:%d %b %Y} ({weeks:.0f} weeks). R after the spread the bot saw.\n")
print(f"   {'setup':<46}{'':<14}{'trades':>7}{'a week':>7}{'R a trade':>10}{'1st half':>10}{'2nd half':>10}{'R a week':>10}")
base = st(one_at_a_time(today))
res = {}
for name, g in CONFIGS:
    s_all, s_one = st(g), st(one_at_a_time(g))
    res[name] = s_one
    for lab, s in (("counted", s_all), ("one at a time", s_one)):
        print(f"   {name if lab == 'counted' else '':<46}{lab:<14}{s['n']:>7}{s['wk']:>7.1f}{s['r']:>+10.2f}{s['h1']:>+10.2f}"
              f"{s['h2']:>+10.2f}{s['rw']:>+10.2f}")
    if not name.startswith("TODAY"):
        ok = s_one["h1"] > base["h1"] and s_one["h2"] > base["h2"] and s_one["rw"] > base["rw"]
        print(f"   {'':<46}{'-> vs TODAY, one at a time: ' + ('PASSES' if ok else 'fails')}")
    print()
print("   CHECKS (counted): TODAY 574 trades +0.09R (+0.85R a week); A +3.22R a week; D +1.76R a week.")
print("   RULE (agreed): replaces today's only if, one at a time, it beats TODAY in BOTH halves AND makes more R a week.\n")

print("THE MIX, MARKET BY MARKET, one at a time (trades, R a trade, 1st / 2nd half, R a week):")
m1 = one_at_a_time(mix)
t1_ = one_at_a_time(today)
for a in SYM:
    s, b0 = st(m1[m1.asset == a]), st(t1_[t1_.asset == a])
    e, m = SETTINGS[a]
    extra = " + continuations + no spikes" if a in FILTERED else ""
    print(f"   {a:<7} {e}+{m.lower()}{extra:<30} {s['n']:>4} tr {s['r']:+.2f} ({s['h1']:+.2f}/{s['h2']:+.2f}) {s['rw']:+.2f}/wk"
          f"    today {b0['r']:+.2f} ({b0['h1']:+.2f}/{b0['h2']:+.2f}) {b0['rw']:+.2f}/wk")
print("\nStill before the council and the 5-dollar minimum; slippage not modelled. The settings were chosen on this")
print("same history -- the final proof is new weeks of data (the weekly re-run, or the practice lane).")
