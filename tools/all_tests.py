"""
ALL TESTS -- one file, one run, read-only. Nothing here changes the bot.

TEST 1   LEVELS: where do 1H turning points land -- the ladder's 4H swing zones, both Livermore brains' levels,
         moving averages (EMA 20/50/200 on the 1H, EMA 50 on the 4H) and open fair-value gaps (1H, 4H) -- against
         chance? (Replaces the 25 Sep level test's Part A, now with MAs and FVGs added.)
TEST 1c  LEVEL EDGE: trade a bounce off each level type -- does it make money after costs, in both halves? And do
         those levels near our own proofs make the proofs better? Writes a gallery of example charts to go through
         by eye: data/research/level_gallery.html
TEST 2   FORWARD TEST: the B11 rules frozen as tested (window ended 24 Sep 2026), run only on setups born after
         it, against what the backtest promised; plus the bot's real trades. Keeps a copy of the price files used.
TEST 3   FLIP TEST (the second record): every signal and its mirror, with what was known at entry.
TEST 3b  YOUR VERSION: the opposite strategy, the losers' record (hindsight flip against what the opposite really
         paid) and the loss fingerprint.
TEST 4   GEAR TEST (gear_sim rebuilt): a break on a 1H close against the tested 4H close.
TEST 5   SECOND LOOKS: BTC's pause-then-turn entry + runner; the runner exit on GOLD and USTEC -- by halves.
TEST 6   NEW MARKETS, SAME RULES: silver, US30, ETH, USDJPY, EURJPY -- fetched from MT5 (15 months), 3 profiles each.
PAPER    every run (and --weekly): ideas 2 to 6 measured on candles the rules have never seen.
WEEKLY   --weekly : the forward test + proof supply + council outliers (Saturday's run).
All tests run on the research engine already verified on this box (tools/realistic_test.py, fingerprint checked).
Run:  python tools/all_tests.py            (everything)
      python tools/all_tests.py --weekly   (Saturday)
"""
import os, sys, re, json, glob, hashlib, shutil, importlib.util
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================================================
# shared: the research engine already verified on this box
# ============================================================================================================
RESEARCH = os.path.join("tools", "realistic_test.py")
RESEARCH_SHA = "5924BF474DC91BEDB91759CAD15EB31574D6D81749CAD4B4B6E010ABE1FAE1BD"
FREEZE = pd.Timestamp("2026-09-24")      # the tested window ended here; every setup after it is new to the rules
MIX = {"GOLD": ("B", "SIMPLE4", False), "USOIL": ("B", "SIMPLE4", False), "USTEC": ("B", "SIMPLE4", False),
       "BTC": ("E", "RUNNER", False), "EURUSD": ("E", "SIMPLE", True), "GBPAUD": ("E", "SIMPLE", True)}
SPIKE = 0.85
UP = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")


def research_ns(variant=None):
    raw = open(RESEARCH, "rb").read()
    got = hashlib.sha256(raw).hexdigest().upper()
    if got != RESEARCH_SHA:
        raise RuntimeError("%s is not the verified version (%s...) -- tests 2 and 3 need it" % (RESEARCH, got[:12]))
    src = raw.decode("utf-8")
    src = src[:src.index("SETTINGS = {")]
    old = "kind=kind, strength=float(strength)))"
    if src.count(old) != 1:
        raise RuntimeError("research engine layout not recognised -- tests 2 and 3 skipped")
    src = src.replace(old, "kind=kind, strength=float(strength), d=d, R2=R2, R1=R1, e=e, stop=stop, i=i, "
                           "conf=conf, atr=atr))")
    if variant == "1H":      # TEST 4: the break confirmed on a 1H close past R2 (the kill stays on the 4H close)
        brk = "                if r4 is not None and ((d == 1 and r4[0] > R2) or (d == -1 and r4[0] < R2)):"
        if src.count(brk) != 1 or src.count("                    j = max(0, i - 3)") != 1:
            raise RuntimeError("research engine layout not recognised -- the gear test is skipped")
        src = src.replace(brk, "                if ((d == 1 and c1[i] > R2) or (d == -1 and c1[i] < R2)):")
        src = src.replace("                    j = max(0, i - 3)", "                    j = i")
    ns = {"__name__": "research"}
    exec(compile(src, "research", "exec"), ns)
    return ns


def mix_rows(df):
    parts = []
    for a, (e, m, filt) in MIX.items():
        g = df[(df.asset == a) & (df.entry == e) & (df["mode"] == m)]
        if filt:
            g = g[(g.kind == "continuation") & ~((g.entry != "B") & (g.strength > SPIKE))]
        parts.append(g)
    return pd.concat(parts) if parts else df.iloc[0:0]


def one_at_a_time(g):
    """One position per market, as the bot trades. Fixed tie-break: time, then R2, then entry type."""
    keep = []
    g = g.sort_values(["asset", "t", "R2", "entry"], kind="mergesort")
    for _, ga in g.groupby("asset", sort=True):
        free = None
        for idx, r in ga.iterrows():
            if free is None or r["t"] >= free:
                keep.append(idx)
                free = r["exit_t"]
    return g.loc[keep]


def run_window(ns, start, end):
    ns["START"], ns["END"] = start, end
    rows = []
    for a in MIX:
        try:
            rows += ns["run_market"](a)
        except Exception as ex:
            print("   (%s: %s: %s)" % (a, type(ex).__name__, ex))
    return pd.DataFrame(rows)


# ============================================================================================================
# TEST 2 -- the forward test
# ============================================================================================================
def test2(ns, bt):
    print("\n" + "=" * 110)
    print("TEST 2 -- THE FORWARD TEST: the B11 rules frozen as tested (window ended %s), run ONLY on setups born after it"
          % FREEZE.strftime("%d %b %Y"))
    print("=" * 110)
    snap = os.path.join("data", "research", "frozen_" + datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(snap, exist_ok=True)
    last_close, stale = {}, []
    for a in MIX:
        sym = ns["SYM"][a]
        for tf in ("1h", "4h"):
            p = os.path.join("data", "raw", "%s_%s.csv" % (sym, tf))
            if os.path.exists(p):
                shutil.copy2(p, snap)
        h1 = ns["load"](sym, "1h")
        last_close[a] = h1.index[-1] + pd.Timedelta(hours=1)
        age_h = (pd.Timestamp.now() - last_close[a]).total_seconds() / 3600
        if age_h > 72:
            stale.append("%s (last candle %s)" % (a, last_close[a].strftime("%d %b %H:%M")))
    print("   data copied for the record: %s" % snap)
    print("   newest candle per market: " + ", ".join("%s %s" % (a, t.strftime("%d %b %H:%M")) for a, t in last_close.items()))
    if stale:
        print("   !! STALE DATA -- these price files have not been updated for 3+ days: " + "; ".join(stale))
        print("   !! the forward test can only see what is in data/raw -- refresh the files, then re-run")

    bm = one_at_a_time(mix_rows(bt))
    wk_bt = max(1.0, (ns_end_bt - ns_start_bt).total_seconds() / 86400 / 7)
    exp = {}
    for a in MIX:
        g = bm[bm.asset == a]["net"]
        exp[a] = (len(g) / wk_bt, g.mean() if len(g) else float("nan"), g.std(ddof=1) if len(g) > 1 else float("nan"))
    g = bm["net"]
    exp["ALL"] = (len(g) / wk_bt, g.mean(), g.std(ddof=1))

    fdf = run_window(ns, FREEZE + pd.Timedelta(minutes=1), pd.Timestamp("2100-01-01"))
    fwd_weeks = max(1e-9, (max(last_close.values()) - FREEZE).total_seconds() / 86400 / 7)
    print("   forward window: %s -> %s (%.1f weeks)" % (FREEZE.strftime("%d %b"), max(last_close.values()).strftime("%d %b %H:%M"), fwd_weeks))
    fm = pd.DataFrame()
    if not fdf.empty:
        fm = one_at_a_time(mix_rows(fdf))
        fm = fm.assign(open=[(r.exit_t >= last_close[r.asset]) and (r.exit_t - r.t < pd.Timedelta(days=7))
                             for r in fm.itertuples()])
    print("\n   %-8s%10s%10s%12s%13s%24s   %s" % ("market", "finished", "open", "R a trade", "expected", "95% band", "verdict so far"))
    tot = []
    for a in list(MIX) + ["ALL"]:
        rows = fm if a == "ALL" else (fm[fm.asset == a] if len(fm) else fm)
        done = rows[~rows["open"]] if len(rows) else rows
        n, n_open = len(done), (int(rows["open"].sum()) if len(rows) else 0)
        per_wk, mu, sd = exp[a]
        m = done["net"].mean() if n else float("nan")
        if n and sd == sd:
            lo_b, hi_b = mu - 1.96 * sd / np.sqrt(n), mu + 1.96 * sd / np.sqrt(n)
            band = "%+.2f to %+.2f" % (lo_b, hi_b)
            if n < 10:
                verdict = "too early (n=%d)" % n
            elif m < lo_b:
                verdict = "BELOW the band"
            elif m > hi_b:
                verdict = "above the band"
            else:
                verdict = "inside the band -- on track"
        else:
            band, verdict = "n/a", "no finished trades yet"
        print("   %-8s%10d%10d%12s%+13.2f%24s   %s" % (a, n, n_open, ("%+.2f" % m) if n else "-", mu, band, verdict))
        if a != "ALL":
            print("   %-8s%10s%10s%12s%13s   trades a week: forward %.1f, backtest %.1f" % ("", "", "", "", "", (n + n_open) / fwd_weeks, per_wk))
    done_all = fm[~fm["open"]] if len(fm) else fm
    print("\n   RULE (agreed in advance, printed every run): at 30+ finished forward trades in total --")
    print("     forward mean below 0R            -> STOP and investigate before anything else changes")
    print("     below the band but above 0R      -> watch; re-check at 60 trades")
    print("     inside or above the band         -> on track")
    if len(done_all) < 30:
        print("   status: %d finished forward trades -- the rule applies from 30." % len(done_all))

    # the bot's real trades over the same days (whatever system was running)
    live = []
    for f in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        try:
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("source") != "live":
                    continue
                ct = r.get("close_time") or r.get("exit_time") or r.get("closed_at")
                try:
                    t = pd.Timestamp(ct)
                    t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
                except Exception:
                    continue
                if t <= FREEZE:
                    continue
                rv = next((r.get(k) for k in ("net_pnl_r", "pnl_r", "r_multiple", "gross_r") if r.get(k) is not None), None)
                live.append((str(r.get("asset")), None if rv is None else float(rv)))
        except Exception:
            continue
    print("\n   REAL TRADES since %s (from the diary; B11 or whatever was running):" % FREEZE.strftime("%d %b"))
    if not live:
        print("     none found")
    for a in sorted({x[0] for x in live}):
        rr = [x[1] for x in live if x[0] == a and x[1] is not None]
        nr = sum(1 for x in live if x[0] == a and x[1] is None)
        print("     %-8s %d trade(s)%s%s" % (a, sum(1 for x in live if x[0] == a),
                                           (", mean %+.2fR" % (sum(rr) / len(rr))) if rr else "",
                                           (" (%d without an R field)" % nr) if nr else ""))
    return fdf


# ============================================================================================================
# TEST 3 -- the flip test (the second record)
# ============================================================================================================
def _brain_states(asset, df):
    try:
        spec = importlib.util.spec_from_file_location("lsm_flip", os.path.join("src", "execution", "livermore_state_machine.py"))
        lsm = importlib.util.module_from_spec(spec)
        sys.modules["lsm_flip"] = lsm
        spec.loader.exec_module(lsm)
        piv = json.load(open(os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig"))["LIVERMORE_PIVOTS"]
        m4, m1 = lsm.make_livermore_pair(asset, piv.get(asset, {}))
        return m4, m1, lsm
    except Exception:
        return None, None, None


def _replay(machine, lsm, df):
    out = []
    if machine is None:
        return [None] * len(df)
    atr = lsm.atr14(df)
    for c, a in zip(df["close"].values, atr.values):
        try:
            out.append(machine.update(float(c), float(a)).state)
        except Exception:
            out.append(out[-1] if out else None)
    return out


def test3(ns, bt, fdf):
    print("\n" + "=" * 110)
    print("TEST 3 -- THE FLIP TEST (the second record): every tested signal and its MIRROR -- same candle, opposite")
    print("          direction, same stop and target distances -- with what was known at entry")
    print("=" * 110)
    sig = pd.concat([bt.assign(window="backtest"), (fdf.assign(window="forward") if len(fdf) else bt.iloc[0:0])])
    sig = mix_rows(sig)
    split = ns["SPLIT"]
    recs = []
    for a in MIX:
        g = sig[sig.asset == a]
        if not len(g):
            continue
        sym = ns["SYM"][a]
        h1, h4 = ns["load"](sym, "1h"), ns["load"](sym, "4h")
        t1 = h1.index + pd.Timedelta(hours=1)
        c1, hi1, lo1, a1 = h1["close"].values, h1["high"].values, h1["low"].values, h1["atr"].values
        pl, ph = ns["pivots"](hi1, lo1, c1, a1)
        t4 = h4.index + pd.Timedelta(hours=4)
        sma4 = h4["close"].rolling(50).mean().values
        m4, m1, lsm = _brain_states(a, h1)
        st1, st4 = _replay(m1, lsm, h1), _replay(m4, lsm, h4)
        mode = MIX[a][1]
        for r in g.itertuples():
            i, d, e, stop = int(r.i), int(r.d), float(r.e), float(r.stop)
            risk = abs(e - stop)
            if risk <= 0:
                continue
            dm = -d
            Rm, why_m, jm = ns["exit_run"](a, mode, dm, e, e + d * risk, i, t1, hi1, lo1, c1, a1,
                                          pl if dm == 1 else ph, float(r.atr))
            cost = ns["COST"][(a, t1[i].hour)] * e / risk
            k4 = int(t4.searchsorted(t1[i], side="right")) - 1
            k4c = int(t4.searchsorted(pd.Timestamp(r.conf), side="right")) - 1
            a4 = float(h4["atr"].values[k4c]) if k4c >= 0 else float("nan")
            fresh = d * (e - float(r.R2)) / float(r.atr)
            swing = abs(float(r.R2) - float(r.R1)) / a4 if a4 == a4 and a4 > 0 else float("nan")
            hr = t1[i].hour
            s1 = st1[i] if i < len(st1) else None
            s4 = st4[k4] if 0 <= k4 < len(st4) else None
            recs.append(dict(
                window=r.window, half="1st" if r.t < split else "2nd", asset=a, t=r.t, dir="long" if d == 1 else "short",
                style=r.entry, kind=r.kind, R2=float(r.R2), R1=float(r.R1), entry=e, stop=stop, risk=risk,
                orig=float(r.net), mirror=float(Rm - cost), mirror_exit=why_m,
                spike="spike" if r.strength > SPIKE else "normal",
                fresh="<0.5 moves" if fresh < 0.5 else ("0.5-1.5 moves" if fresh < 1.5 else "1.5-2.5 moves"),
                swing="small (<2 moves)" if swing < 2 else ("medium (2-4)" if swing < 4 else "big (4+)"),
                session="00-07" if hr < 7 else ("07-13" if hr < 13 else ("13-21" if hr < 21 else "21-24")),
                weekday=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][pd.Timestamp(r.t).dayofweek],
                trend4=("4H above its 50-avg" if (k4 >= 0 and sma4[k4] == sma4[k4] and h4["close"].values[k4] > sma4[k4])
                        else "4H below its 50-avg") if k4 >= 0 else "n/a",
                brain1=("no reading" if s1 is None else ("1H brain agrees" if (s1 in UP) == (d == 1) else "1H brain disagrees")),
                brain4=("no reading" if s4 is None else ("4H brain agrees" if (s4 in UP) == (d == 1) else "4H brain disagrees")),
            ))
    df = pd.DataFrame(recs)
    if df.empty:
        print("   no signals to flip")
        return df
    os.makedirs(os.path.join("data", "research"), exist_ok=True)
    out = os.path.join("data", "research", "flip_record.csv")
    df.to_csv(out, index=False)
    b = df[df.window == "backtest"]
    print("   the second record: %d signals (%d backtest, %d forward) -> %s" % (len(df), len(b), len(df) - len(b), out))
    print("   OVERALL (backtest, every signal, R a trade after costs):  original %+.2f   mirror %+.2f   "
          "(a mirror of a winning strategy should lose -- this is the sanity check)" % (b.orig.mean(), b.mirror.mean()))

    feats = ["style", "kind", "spike", "fresh", "swing", "session", "weekday", "trend4", "brain1", "brain4", "dir"]
    groups = []
    for f in feats:
        for v, g in b.groupby(f):
            groups.append(("all markets", "%s = %s" % (f, v), g))
    for a, ga in b.groupby("asset"):
        groups.append((a, "every signal", ga))
        for f in feats:
            for v, g in ga.groupby(f):
                groups.append((a, "%s = %s" % (f, v), g))
    flips, skips = [], []
    for mk, lab, g in groups:
        h1g, h2g = g[g.half == "1st"], g[g.half == "2nd"]
        if len(g) < 20 or len(h1g) < 8 or len(h2g) < 8:
            continue
        o1, o2, m1_, m2_ = h1g.orig.mean(), h2g.orig.mean(), h1g.mirror.mean(), h2g.mirror.mean()
        row = (mk, lab, len(g), g.orig.mean(), o1, o2, g.mirror.mean(), m1_, m2_)
        if m1_ > 0.10 and m2_ > 0.10 and g.mirror.mean() - g.orig.mean() >= 0.20:
            flips.append(row)
        if o1 < 0 and o2 < 0:
            skips.append(row)
    tested = sum(1 for mk, lab, g in groups if len(g) >= 20 and (g.half == "1st").sum() >= 8 and (g.half == "2nd").sum() >= 8)
    print("   groups tested: %d (every one with 20+ signals and 8+ in each half)" % tested)
    print("   AGREED IN ADVANCE -- a FLIP candidate: the mirror makes over +0.10R in BOTH halves and beats the original by")
    print("   0.20R+; a SKIP candidate: the original loses in BOTH halves. With this many groups some pass by luck --")
    print("   nothing here is a rule until the forward test confirms it.")
    hdr = "   %-12s%-34s%6s%10s%16s%10s%16s" % ("market", "group", "n", "orig", "(1st / 2nd)", "mirror", "(1st / 2nd)")
    for title, lst in (("FLIP CANDIDATES (the mirror wins)", flips), ("SKIP CANDIDATES (the original loses in both halves)", skips)):
        print("\n   " + title + (": none" if not lst else ""))
        if lst:
            print(hdr)
            for r in sorted(lst, key=lambda x: -(x[6] - x[3]))[:20]:
                print("   %-12s%-34s%6d%+10.2f%16s%+10.2f%16s" % (r[0], r[1][:33], r[2], r[3], "(%+.2f / %+.2f)" % (r[4], r[5]),
                                                              r[6], "(%+.2f / %+.2f)" % (r[7], r[8])))
    fw = df[df.window == "forward"]
    if len(fw):
        print("\n   forward signals so far: %d -- original %+.2f, mirror %+.2f (too few to judge; kept in the record)"
              % (len(fw), fw.orig.mean(), fw.mirror.mean()))
    return df


# ============================================================================================================
# ============================================================================================================
# TEST 3b -- Desire's version: flip the losers, and study what the losers had in common
# ============================================================================================================
FEATS = ["style", "kind", "spike", "fresh", "swing", "session", "weekday", "trend4", "brain1", "brain4", "dir"]


def test3b(df):
    print("\n" + "=" * 110)
    print("TEST 3b -- YOUR VERSION: flip the losers into wins, find the opposite's edge, and reverse-engineer the losers")
    print("=" * 110)
    if df is None or df.empty:
        print("   no record to study")
        return
    b = df[df.window == "backtest"].copy()
    b["lost"] = b["orig"] < 0
    print("   PART 1 -- THE OPPOSITE STRATEGY (every signal flipped), market by market, R a trade after costs:")
    print("   %-8s%6s%12s%18s%12s%18s%14s" % ("market", "n", "original", "(1st / 2nd)", "opposite", "(1st / 2nd)", "opposite wins"))
    for a, g in list(b.groupby("asset")) + [("ALL", b)]:
        h1g, h2g = g[g.half == "1st"], g[g.half == "2nd"]
        print("   %-8s%6d%+12.2f%18s%+12.2f%18s%13.0f%%" % (
            a, len(g), g.orig.mean(), "(%+.2f / %+.2f)" % (h1g.orig.mean(), h2g.orig.mean()), g.mirror.mean(),
            "(%+.2f / %+.2f)" % (h1g.mirror.mean(), h2g.mirror.mean()), 100.0 * (g.mirror > 0).mean()))
    L = b[b.lost]
    W = b[~b.lost]
    print("\n   PART 2 -- THE LOSERS' RECORD: %d losing signals out of %d (%.0f%%)" % (len(L), len(b), 100.0 * len(L) / max(1, len(b))))
    print("     flip their result on paper (the losses turned into wins):  %+.2fR a trade  <- HINDSIGHT: nobody knows at entry" % (-L.orig.mean() if len(L) else 0))
    print("     actually trade the opposite on those same signals:          %+.2fR a trade  <- what the market really paid" % (L.mirror.mean() if len(L) else 0))
    print("     the gap between the two is the price of not knowing in advance -- the patterns below try to close it.")
    out = os.path.join("data", "research", "losers_record.csv")
    L.assign(hindsight_flip=-L.orig).to_csv(out, index=False)
    print("     saved: %s" % out)
    print("\n   PART 3 -- THE LOSS FINGERPRINT: what was more common at entry on the LOSERS than on the winners?")
    print("   lift = (share of losers with it) / (share of winners with it). AGREED IN ADVANCE: a loser marker needs lift 1.3+")
    print("   in BOTH halves and 15+ losers; the last column shows whether trading the opposite on that group actually paid.")
    rows = []
    for f in FEATS:
        for v in b[f].dropna().unique():
            lifts = []
            for h in ("1st", "2nd"):
                Lh, Wh = L[L.half == h], W[W.half == h]
                pl_ = (Lh[f] == v).mean() if len(Lh) else 0.0
                pw_ = (Wh[f] == v).mean() if len(Wh) else 0.0
                lifts.append(pl_ / pw_ if pw_ > 0 else float("inf") if pl_ > 0 else float("nan"))
            nl = int((L[f] == v).sum())
            grp = b[b[f] == v]
            if nl >= 15 and all(x == x and x >= 1.3 for x in lifts):
                rows.append((f, v, nl, lifts[0], lifts[1], grp.orig.mean(), grp.mirror.mean(), len(grp)))
    if not rows:
        print("     no loser markers pass -- the losers look like the winners at entry (so far)")
    else:
        print("   %-10s%-24s%8s%16s%12s%12s%8s" % ("feature", "value", "losers", "lift 1st/2nd", "original", "opposite", "n"))
        for r in sorted(rows, key=lambda x: -min(x[3], x[4]))[:20]:
            print("   %-10s%-24s%8d%16s%+12.2f%+12.2f%8d" % (r[0], str(r[1])[:23], r[2], "%.2f / %.2f" % (r[3], r[4]), r[5], r[6], r[7]))
    print("   a marker whose group loses (original < 0) is a SKIP candidate; if the opposite pays there too, a FLIP")
    print("   candidate. Either way: forward test first, then a rule.")


# ============================================================================================================
# TEST 4 -- the gear test (gear_sim rebuilt for the new engine): break on a 1H close against a 4H close
# ============================================================================================================
def test4(ns4, bt4):
    print("\n" + "=" * 110)
    print("TEST 4 -- THE GEAR TEST (gear_sim, rebuilt for the new engine): a break on a 1H close past R2, against the")
    print("          tested 4H close. Everything else identical; one position per market; after costs.")
    print("=" * 110)
    try:
        ns1 = research_ns(variant="1H")
    except Exception as ex:
        print("   skipped: %s" % ex)
        return
    ns1["START"], ns1["END"] = ns4["START"], ns4["END"]
    bt1 = run_window(ns1, ns4["START"], ns4["END"])
    weeks = max(1.0, (ns4["END"] - ns4["START"]).total_seconds() / 86400 / 7)
    m4, m1 = one_at_a_time(mix_rows(bt4)), one_at_a_time(mix_rows(bt1))
    split = ns4["SPLIT"]

    def st(g):
        a_, b_ = g[g["t"] < split]["net"], g[g["t"] >= split]["net"]
        return (len(g), g.net.mean() if len(g) else float("nan"), a_.mean() if len(a_) else float("nan"),
                b_.mean() if len(b_) else float("nan"), g.net.sum() / weeks)
    print("   AGREED IN ADVANCE: the 1H break replaces the 4H break in a market only if it beats it in BOTH halves AND")
    print("   makes more R a week there.")
    print("   %-8s%-10s%8s%11s%18s%11s   %s" % ("market", "break", "trades", "R a trade", "(1st / 2nd)", "R a week", "verdict"))
    for a in list(MIX) + ["ALL"]:
        g4 = m4 if a == "ALL" else m4[m4.asset == a]
        g1 = m1 if a == "ALL" else m1[m1.asset == a]
        s4, s1 = st(g4), st(g1)
        ok = s1[2] > s4[2] and s1[3] > s4[3] and s1[4] > s4[4]
        for lab, s in (("4H close", s4), ("1H close", s1)):
            print("   %-8s%-10s%8d%+11.2f%18s%+11.2f   %s" % (a if lab == "4H close" else "", lab, s[0], s[1],
                                                           "(%+.2f / %+.2f)" % (s[2], s[3]), s[4],
                                                           ("-> SWITCH to the 1H break" if ok else "-> keep the 4H break") if lab == "1H close" else ""))


# ============================================================================================================
# WEEKLY -- proof supply and council outliers (the B11 versions), for Saturday's run
# ============================================================================================================
def weekly_supply(days=7):
    print("\n" + "=" * 110)
    print("WEEKLY -- PROOF SUPPLY: the new engine's funnel per market over the last %d days (from the bot's log)" % days)
    print("=" * 110)
    since = datetime.now() - timedelta(days=days)
    tags = [("born", "[SETUP-BORN]"), ("broke", "-> BREAK"), ("retested", "[COUNT-2]"), ("proofs", "[NS-PROOF]"),
            ("retired", "[NS-SKIP]"), ("missed", "[NS-MISSED]"), ("entered", "[NS-LEVELS]"), ("pause", "[NS-PAUSE]")]
    cnt = defaultdict(lambda: defaultdict(int))
    skip_why = defaultdict(int)
    files = sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime)
    for fpath in files:
        if datetime.fromtimestamp(os.path.getmtime(fpath)) < since:
            continue
        try:
            for line in open(fpath, encoding="utf-8", errors="replace"):
                if len(line) < 19 or line[:19] < since.strftime("%Y-%m-%d %H:%M:%S"):
                    continue
                for key, tag in tags:
                    if tag in line and (key != "broke" or "[COUNT-1-CHECK]" in line):
                        m = re.search(r"\] (\w+)[: ]", line[line.index(tag if key != "broke" else "[COUNT-1-CHECK]"):])
                        cnt[m.group(1) if m else "?"][key] += 1
                        if key == "retired":
                            skip_why[line.split(" -- ")[1].split("(")[0].strip() if " -- " in line else "?"] += 1
        except Exception:
            continue
    if not cnt:
        print("   no new-engine lines in the log for this period (B11 not running yet, or the log has rolled)")
        return
    keys = [k for k, _ in tags]
    print("   %-8s" % "market" + "".join("%10s" % k for k in keys))
    for a in sorted(cnt):
        print("   %-8s" % a + "".join("%10d" % cnt[a][k] for k in keys))
    if skip_why:
        print("   retired because: " + "; ".join("%s x%d" % (k, v) for k, v in sorted(skip_why.items(), key=lambda x: -x[1])))


def weekly_council(days=28):
    print("\n" + "=" * 110)
    print("WEEKLY -- COUNCIL OUTLIERS: does the council's score predict the result? (diary, last %d days)" % days)
    print("=" * 110)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows, seen_keys = [], None
    for f in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ct = r.get("close_time") or r.get("exit_time") or r.get("closed_at")
            try:
                t = pd.Timestamp(ct)
                t = t.tz_localize("UTC") if t.tzinfo is None else t
            except Exception:
                continue
            if t < since:
                continue
            seen_keys = seen_keys or sorted(r.keys())
            sc = next((r.get(k) for k in ("council_score", "final_score", "aggregator_score", "score", "confidence")
                       if isinstance(r.get(k), (int, float))), None)
            rv = next((r.get(k) for k in ("net_pnl_r", "pnl_r", "r_multiple", "gross_r") if isinstance(r.get(k), (int, float))), None)
            if sc is not None and rv is not None:
                rows.append((str(r.get("asset")), str(r.get("source")), float(sc), float(rv)))
    if not rows:
        print("   no rows with both a council score and an R -- the diary's field names (tell Claude):")
        print("   " + (", ".join(seen_keys) if seen_keys else "no diary rows in the period"))
        return
    d = pd.DataFrame(rows, columns=["asset", "source", "score", "R"])
    d["band"] = pd.qcut(d["score"].rank(method="first"), q=min(4, len(d)), labels=False)
    print("   score quarter (0 = lowest)   trades   mean R   win %")
    for q, g in d.groupby("band"):
        print("   %20s%10d%+9.2f%8.0f%%" % (int(q), len(g), g.R.mean(), 100.0 * (g.R > 0).mean()))
    top = d.sort_values("score").tail(3)
    bot = d.sort_values("score").head(3)
    print("   highest-scored: " + "; ".join("%s %.2f -> %+.2fR" % (r.asset, r.score, r.R) for r in top.itertuples()))
    print("   lowest-scored:  " + "; ".join("%s %.2f -> %+.2fR" % (r.asset, r.score, r.R) for r in bot.itertuples()))
    print("   if the top quarter does no better than the bottom quarter, the council's score is not adding anything yet.")


# ============================================================================================================
# TEST 1 -- LEVELS: where do 1H turning points land? (ladder zones, brain levels, moving averages, FVGs vs chance)
# TEST 1c -- LEVEL EDGE: does trading a bounce off each level make money, and do they help our proofs?
# ============================================================================================================
LEVEL_TYPES = ["4H swing zone", "1H brain level", "4H brain level", "EMA 20 (1H)", "EMA 50 (1H)", "EMA 200 (1H)",
               "EMA 50 (4H)", "open FVG (1H)", "open FVG (4H)"]
_MKT_CACHE = {}


def _brain_levels(asset, df, hours):
    """Every level a Livermore brain sets, with the time it became known (the close of that candle)."""
    m4, m1, lsm = _brain_states(asset, df)
    m = m4 if hours == 4 else m1
    if m is None:
        return np.array([], dtype="datetime64[ns]"), np.array([])
    atr = lsm.atr14(df)
    ts, vs, last = [], [], {}
    for t, c, a in zip(df.index, df["close"].values, atr.values):
        try:
            s = m.update(float(c), float(a))
        except Exception:
            continue
        up = s.state in UP
        for key, v in (("up", s.anchor_main_up_max if up else None), ("down", None if up else s.anchor_main_down_min),
                       ("nlow", s.anchor_natural_low), ("nhigh", s.anchor_natural_high)):
            if v is not None and v == v and last.get(key) != v:
                ts.append(np.datetime64(t + pd.Timedelta(hours=hours)))
                vs.append(float(v))
                last[key] = v
    o = np.argsort(np.array(ts, dtype="datetime64[ns]"), kind="mergesort")
    return np.array(ts, dtype="datetime64[ns]")[o], np.array(vs)[o]


def _fvg_list(hi, lo):
    """(formed_at_index, zone_low, zone_high, bullish, filled_at_index)."""
    out, n = [], len(hi)
    for k in range(2, n):
        if lo[k] > hi[k - 2]:
            zl, zh, bull = float(hi[k - 2]), float(lo[k]), True
        elif hi[k] < lo[k - 2]:
            zl, zh, bull = float(hi[k]), float(lo[k - 2]), False
        else:
            continue
        j = k + 1
        while j < n and not ((bull and lo[j] <= zl) or ((not bull) and hi[j] >= zh)):
            j += 1
        out.append((k, zl, zh, bull, j))
    return out


def _market(ns, a):
    """Everything the level tests need for one market, computed once."""
    if a in _MKT_CACHE:
        return _MKT_CACHE[a]
    sym = ns["SYM"][a]
    h1, h4 = ns["load"](sym, "1h"), ns["load"](sym, "4h")
    M = {"h1": h1, "h4": h4, "t1": h1.index + pd.Timedelta(hours=1), "t4": h4.index + pd.Timedelta(hours=4)}
    M["c1"], M["hi1"], M["lo1"], M["a1"] = h1["close"].values, h1["high"].values, h1["low"].values, h1["atr"].values
    M["pl"], M["ph"] = ns["pivots"](M["hi1"], M["lo1"], M["c1"], M["a1"])
    M["ema"] = {p: h1["close"].ewm(span=p, adjust=False).mean().values for p in (20, 50, 200)}
    M["ema4"] = h4["close"].ewm(span=50, adjust=False).mean().values
    M["zones"] = [(np.datetime64(conf), min(lvl, edge), max(lvl, edge), typ) for conf, typ, lvl, edge in ns["swings"](h4)]
    M["b1"] = _brain_levels(a, h1, 1)
    M["b4"] = _brain_levels(a, h4, 4)
    M["f1"] = _fvg_list(M["hi1"], M["lo1"])
    M["f4"] = _fvg_list(h4["high"].values, h4["low"].values)
    M["f1k"] = np.array([f[0] for f in M["f1"]], dtype=int)
    M["f4k"] = np.array([f[0] for f in M["f4"]], dtype=int)
    M["zt"] = np.array([z[0] for z in M["zones"]], dtype="datetime64[ns]")
    _MKT_CACHE[a] = M
    return M


def _levels_at(M, i, want_low):
    """Every level known before 1H candle i (using nothing from candle i onwards), as (type, lo, hi) bands."""
    t_prev = np.datetime64(M["t1"][i - 1])
    k4 = int(M["t4"].searchsorted(M["t1"][i - 1], side="right")) - 1
    out = []
    z0 = int(M["zt"].searchsorted(t_prev - np.timedelta64(30, "D"), side="left"))
    z1 = int(M["zt"].searchsorted(t_prev, side="right"))
    for conf, zl, zh, typ in M["zones"][z0:z1]:
        out.append(("4H swing zone", zl, zh))
    for lab, (ts, vs), days in (("1H brain level", M["b1"], 10), ("4H brain level", M["b4"], 30)):
        lo_i = ts.searchsorted(t_prev - np.timedelta64(days, "D"), side="left")
        hi_i = ts.searchsorted(t_prev, side="right")
        out += [(lab, float(v), float(v)) for v in vs[lo_i:hi_i]]
    for p in (20, 50, 200):
        out.append(("EMA %d (1H)" % p, float(M["ema"][p][i - 1]), float(M["ema"][p][i - 1])))
    if k4 >= 0:
        out.append(("EMA 50 (4H)", float(M["ema4"][k4]), float(M["ema4"][k4])))
    a0, a1_ = int(M["f1k"].searchsorted(i - 240, side="left")), int(M["f1k"].searchsorted(i - 1, side="right"))
    for (k, zl, zh, bull, filled) in M["f1"][a0:a1_]:
        if filled > i - 1 and bull == want_low:
            out.append(("open FVG (1H)", zl, zh))
    if k4 >= 0:
        b0, b1_ = int(M["f4k"].searchsorted(k4 - 180, side="left")), int(M["f4k"].searchsorted(k4, side="right"))
        for (k, zl, zh, bull, filled) in M["f4"][b0:b1_]:
            if filled > k4 and bull == want_low:
                out.append(("open FVG (4H)", zl, zh))
    return out


def test1_levels(ns):
    print("\n" + "=" * 110)
    print("TEST 1 -- LEVELS: where do 1H turning points land? The ladder (4H swing zones, both brains' levels), moving")
    print("          averages and open fair-value gaps -- against chance.")
    print("=" * 110)
    print("   turning point = a 1H swing low / high (4 candles left, 2 right, 0.3+ moves deep); levels = only those known")
    print("   BEFORE the turning candle; hit = the turn's extreme within a quarter of a typical move of the level (or inside")
    print("   its zone); chance = the same levels at 20 random prices within 3 moves of the real turn.")
    print("   AGREED IN ADVANCE: a level type 'matters' if turns land on it 1.5x+ as often as chance in BOTH halves.")
    rng = np.random.default_rng(7)
    split = ns["SPLIT"]
    agg = defaultdict(lambda: [0, 0.0, 0])
    for a in MIX:
        M = _market(ns, a)
        for i in range(210, len(M["c1"]) - 2):
            if not (M["pl"][i] or M["ph"][i]) or not (ns_start_bt <= M["t1"][i] <= ns_end_bt):
                continue
            atr = M["a1"][i]
            if not (atr == atr and atr > 0):
                continue
            low_turn = bool(M["pl"][i])
            x = float(M["lo1"][i] if low_turn else M["hi1"][i])
            tol = 0.25 * atr
            half = "1st" if M["t1"][i] < split else "2nd"
            lv = _levels_at(M, i, low_turn)
            rnd = x + rng.uniform(-3, 3, 20) * atr
            for typ in LEVEL_TYPES:
                bands = [(l, h) for (tt, l, h) in lv if tt == typ]
                hit = any(l - tol <= x <= h + tol for l, h in bands)
                rh = float(np.mean([any(l - tol <= r <= h + tol for l, h in bands) for r in rnd])) if bands else 0.0
                for key in ((a, typ, half), ("ALL", typ, half)):
                    s = agg[key]
                    s[0] += int(hit)
                    s[1] += rh
                    s[2] += 1
    print("\n   %-8s%-16s%16s%13s%14s   %s" % ("market", "level type", "turns 1st/2nd", "hit % real", "hit % chance", "ratio 1st / 2nd -> verdict"))
    for a in ["ALL"] + list(MIX):
        for typ in LEVEL_TYPES:
            s1, s2 = agg[(a, typ, "1st")], agg[(a, typ, "2nd")]
            if s1[2] + s2[2] == 0:
                continue
            r1 = s1[0] / s1[1] if s1[1] > 0 else float("nan")
            r2 = s2[0] / s2[1] if s2[1] > 0 else float("nan")
            ok = r1 == r1 and r2 == r2 and r1 >= 1.5 and r2 >= 1.5 and s1[2] >= 30 and s2[2] >= 30
            print("   %-8s%-16s%16s%12.1f%%%13.1f%%   %s / %s -> %s" % (
                a, typ, "%d / %d" % (s1[2], s2[2]), 100.0 * (s1[0] + s2[0]) / (s1[2] + s2[2]),
                100.0 * (s1[1] + s2[1]) / (s1[2] + s2[2]), ("%.2f" % r1) if r1 == r1 else "n/a",
                ("%.2f" % r2) if r2 == r2 else "n/a", "MATTERS" if ok else "not shown"))
        if a == "ALL":
            print()


GALLERY = []      # (title, market, i0, i1, level band, entry, stop, target, R, direction)


def test1c_edge(ns, bt):
    print("\n" + "=" * 110)
    print("TEST 1c -- LEVEL EDGE: trade a bounce off each level type -- does it make money? And do moving averages / FVGs")
    print("           near our own proofs make those proofs better?")
    print("=" * 110)
    print("   the bounce: price comes from the right side, touches the level (within a quarter move, or into its zone) and")
    print("   the same 1H candle closes back on the right side -> enter at that close. Stop 0.3 moves beyond the candle's")
    print("   extreme; target 2.5 moves; out after 7 days. Same stop floor, R:R minimum and costs as every other test.")
    print("   One trade per level type per market at a time; one per level per day.")
    print("   AGREED IN ADVANCE: 'promising' = after costs over +0.10R a trade in BOTH halves with 30+ trades in each half.")
    print("   That earns a proper design and test -- it is not a rule.")
    split = ns["SPLIT"]
    rows = []
    for a in MIX:
        M = _market(ns, a)
        c1, hi1, lo1, a1, t1 = M["c1"], M["hi1"], M["lo1"], M["a1"], M["t1"]
        last_lvl = {}
        for i in range(210, len(c1) - 1):
            if not (ns_start_bt <= t1[i] <= ns_end_bt):
                continue
            atr = a1[i]
            if not (atr == atr and atr > 0):
                continue
            tol = 0.25 * atr
            for want_low in (True, False):
                d = 1 if want_low else -1
                for (typ, l, h) in _levels_at(M, i, want_low):
                    if want_low:
                        ok = c1[i - 1] > h and lo1[i] <= h + tol and c1[i] > h and lo1[i] >= l - 2 * atr
                    else:
                        ok = c1[i - 1] < l and hi1[i] >= l - tol and c1[i] < l and hi1[i] <= h + 2 * atr
                    if not ok:
                        continue
                    key = (typ, d, round(l, 6), round(h, 6))
                    if key in last_lvl and (t1[i] - last_lvl[key]) < pd.Timedelta(hours=24):
                        continue
                    last_lvl[key] = t1[i]
                    e = float(c1[i])
                    stop = float(lo1[i] - 0.3 * atr) if d == 1 else float(hi1[i] + 0.3 * atr)
                    floor = ns["MIN_SL_PCT"].get(a, 0.0) * e
                    if abs(e - stop) < floor:
                        stop = e - d * floor
                    if abs(e - stop) > 5 * atr:
                        stop = e - d * 5 * atr
                    risk = abs(e - stop)
                    if risk <= 0 or 2.5 * atr / risk < ns["MIN_RR"].get(a, 0.5):
                        continue
                    R, why, jx = ns["exit_run"](a, "SIMPLE", d, e, stop, i, t1, hi1, lo1, c1, a1,
                                               M["pl"] if d == 1 else M["ph"], float(atr))
                    cost = ns["COST"][(a, t1[i].hour)] * e / risk
                    rows.append(dict(asset=a, type=typ, t=t1[i], exit_t=t1[jx], d=d, i=i, j=jx, e=e, stop=stop,
                                     tgt=e + d * 2.5 * atr, lvl_lo=l, lvl_hi=h, net=R - cost, R2=round(l, 6), entry=typ))
    df = pd.DataFrame(rows)
    if df.empty:
        print("   no bounces found")
        return
    weeks = max(1.0, (ns_end_bt - ns_start_bt).total_seconds() / 86400 / 7)
    kept = []
    for (a, typ), g in df.groupby(["asset", "type"]):
        g = g.sort_values(["t", "R2"], kind="mergesort")
        free = None
        for r in g.itertuples():
            if free is None or r.t >= free:
                kept.append(r.Index)
                free = r.exit_t
    df = df.loc[kept]
    print("\n   %-8s%-16s%8s%9s%11s%18s%10s   %s" % ("market", "level type", "trades", "win %", "R a trade", "(1st / 2nd)", "R a week", "verdict"))
    promising = []
    for a in ["ALL"] + list(MIX):
        for typ in LEVEL_TYPES:
            g = df[df.type == typ] if a == "ALL" else df[(df.asset == a) & (df.type == typ)]
            if not len(g):
                continue
            g1, g2 = g[g.t < split], g[g.t >= split]
            m1_ = g1.net.mean() if len(g1) else float("nan")
            m2_ = g2.net.mean() if len(g2) else float("nan")
            ok = len(g1) >= 30 and len(g2) >= 30 and m1_ > 0.10 and m2_ > 0.10
            if ok:
                promising.append((a, typ))
            print("   %-8s%-16s%8d%8.0f%%%+11.2f%18s%+10.2f   %s" % (a, typ, len(g), 100.0 * (g.net > 0).mean(), g.net.mean(),
                                                               "(%+.2f / %+.2f)" % (m1_, m2_), g.net.sum() / weeks,
                                                               "PROMISING" if ok else ""))
        if a == "ALL":
            print()
    print("   level types x markets checked: %d -- with this many, one or two can look good by luck." % (len(LEVEL_TYPES) * (len(MIX) + 1)))
    print("   promising: " + (", ".join("%s %s" % p for p in promising) if promising else "none"))
    for typ in LEVEL_TYPES:
        g = df[df.type == typ].sort_values("net")
        for r in list(g.tail(2).itertuples()) + list(g.head(2).itertuples()):
            GALLERY.append(("%s bounce -- %s %s -- %+.2fR" % (typ, r.asset, "long" if r.d == 1 else "short", r.net), r.asset,
                            int(r.i), int(r.j), (r.lvl_lo, r.lvl_hi), r.e, r.stop, r.tgt, r.net, r.d))

    # do moving averages / FVGs / brain levels near our own proofs help those proofs?
    print("\n   OUR PROOFS WITH AND WITHOUT A LEVEL NEARBY (the tested mix, every signal, after costs):")
    sig = mix_rows(bt)
    feat = defaultdict(lambda: {"yes": [], "no": []})
    for r in sig.itertuples():
        M = _market(ns, r.asset)
        i, d, e = int(r.i), int(r.d), float(r.e)
        atr = float(r.atr)
        lv = _levels_at(M, i, d == 1)
        half = "1st" if r.t < split else "2nd"
        for typ in LEVEL_TYPES:
            near = any(l - 0.5 * atr <= e <= h + 0.5 * atr for (tt, l, h) in lv if tt == typ)
            feat[typ]["yes" if near else "no"].append((half, float(r.net), r))
    print("   %-16s%18s%18s%22s%22s" % ("level near entry", "with: n / R", "without: n / R", "with (1st / 2nd)", "without (1st / 2nd)"))
    for typ in LEVEL_TYPES:
        y, n_ = feat[typ]["yes"], feat[typ]["no"]

        def mm(lst, h=None):
            v = [x[1] for x in lst if h is None or x[0] == h]
            return (sum(v) / len(v)) if v else float("nan")
        print("   %-16s%18s%18s%22s%22s" % (typ, "%d / %+.2f" % (len(y), mm(y)), "%d / %+.2f" % (len(n_), mm(n_)),
                                         "(%+.2f / %+.2f)" % (mm(y, "1st"), mm(y, "2nd")),
                                         "(%+.2f / %+.2f)" % (mm(n_, "1st"), mm(n_, "2nd"))))
        best = sorted(y, key=lambda x: x[1])
        for (h, net, r) in best[-1:] + best[:1]:
            M = _market(ns, r.asset)
            GALLERY.append(("our proof with %s nearby -- %s %s -- %+.2fR" % (typ, r.asset, "long" if int(r.d) == 1 else "short", net),
                            r.asset, int(r.i), min(len(M["c1"]) - 1, int(r.i) + 60), None, float(r.e), float(r.stop),
                            None, net, int(r.d)))
    print("   a level type that helps our proofs in BOTH halves is a FILTER candidate (forward test first).")


def write_gallery(ns, path=os.path.join("data", "research", "level_gallery.html")):
    """Example charts to go through by eye: best and worst bounces per level type, and our proofs near each level."""
    if not GALLERY:
        return None
    divs, scripts = [], []
    for n, (title, a, i0, i1, band, e, stop, tgt, net, d) in enumerate(GALLERY[:80]):
        M = _market(ns, a)
        lo_i, hi_i = max(0, i0 - 48), min(len(M["c1"]) - 1, max(i1, i0) + 12)
        x = [str(t)[:16] for t in M["h1"].index[lo_i:hi_i + 1]]
        tr = [{"type": "candlestick", "x": x, "open": M["h1"]["open"].values[lo_i:hi_i + 1].round(6).tolist(),
               "high": M["hi1"][lo_i:hi_i + 1].round(6).tolist(), "low": M["lo1"][lo_i:hi_i + 1].round(6).tolist(),
               "close": M["c1"][lo_i:hi_i + 1].round(6).tolist(), "name": a, "showlegend": False}]
        for p, col in ((50, "#ab47bc"), (200, "#6d4c41")):
            tr.append({"type": "scatter", "mode": "lines", "x": x, "y": M["ema"][p][lo_i:hi_i + 1].round(6).tolist(),
                       "name": "EMA %d" % p, "line": {"color": col, "width": 1}})
        shapes = []
        if band is not None:
            shapes.append({"type": "rect", "xref": "x", "yref": "y", "x0": x[0], "x1": x[-1], "y0": band[0],
                           "y1": band[1] if band[1] > band[0] else band[0] * 1.00001,
                           "fillcolor": "rgba(0,176,255,0.18)", "line": {"color": "#00b0ff", "width": 1}})
        ex = str(M["h1"].index[i0])[:16]
        tr.append({"type": "scatter", "mode": "markers", "x": [ex], "y": [e], "name": "entry",
                   "marker": {"symbol": "triangle-up" if d == 1 else "triangle-down", "size": 13,
                              "color": "#00c853" if net > 0 else "#d50000", "line": {"color": "black", "width": 1}}})
        for yv, col, nm in ((stop, "#d50000", "stop"), (tgt, "#00c853", "target")):
            if yv is not None:
                shapes.append({"type": "line", "xref": "x", "yref": "y", "x0": ex, "x1": x[-1], "y0": yv, "y1": yv,
                               "line": {"color": col, "width": 1, "dash": "dot"}})
        lay = {"title": {"text": title, "font": {"size": 12}}, "height": 360, "margin": {"l": 50, "r": 10, "t": 36, "b": 30},
               "xaxis": {"rangeslider": {"visible": False}, "type": "category", "nticks": 8}, "shapes": shapes,
               "legend": {"orientation": "h", "y": -0.12}}
        divs.append('<div id="g%d" class="g"></div>' % n)
        scripts.append("Plotly.newPlot('g%d', %s, %s, {responsive: true, displaylogo: false});" % (
            n, json.dumps(tr, default=str), json.dumps(lay, default=str)))
    page = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Level gallery</title><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
            "<style>body{font-family:sans-serif;margin:8px}.g{width:100%%;max-width:900px;margin:0 auto 18px}</style></head><body>"
            "<h2>Level gallery -- best and worst examples, to go through by eye</h2>"
            "<p>Blue band = the level (zone, FVG, MA or brain level). Triangle = the entry (green won, red lost). Dotted = stop"
            " and target. Purple / brown = EMA 50 / 200 (1H). From the backtest window. Sample, not proof.</p>%s<script>%s</script>"
            "</body></html>") % ("".join(divs), "".join(scripts))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


# ============================================================================================================
# TEST 5 -- second looks: BTC's pause-then-turn entry; the runner exit on GOLD and USTEC (idea 2 and idea 5)
# ============================================================================================================
T5 = [("BTC", ("E", "RUNNER"), ("E2", "RUNNER"), "BTC: pause-then-turn entry + runner (idea 2)"),
      ("GOLD", ("B", "SIMPLE4"), ("B", "RUNNER"), "GOLD: runner exit instead of the 4-move target (idea 5)"),
      ("USTEC", ("B", "SIMPLE4"), ("B", "RUNNER"), "USTEC: runner exit on the break entry (idea 5)"),
      ("USTEC", ("B", "SIMPLE4"), ("E", "RUNNER"), "USTEC: today's entry + runner (idea 5)")]


def _stat(g, split, weeks):
    a_, b_ = g[g["t"] < split]["net"], g[g["t"] >= split]["net"]
    return (len(g), g.net.mean() if len(g) else float("nan"), a_.mean() if len(a_) else float("nan"),
            b_.mean() if len(b_) else float("nan"), g.net.sum() / weeks, len(a_), len(b_))


def _pick(df, a, e, m):
    return one_at_a_time(df[(df.asset == a) & (df.entry == e) & (df["mode"] == m)])


def test5(ns, bt):
    print("\n" + "=" * 110)
    print("TEST 5 -- SECOND LOOKS (ideas 2 and 5): each candidate against today's setting in that market, one position at")
    print("          a time, after costs. AGREED IN ADVANCE: a candidate replaces today's only if it beats it in BOTH")
    print("          halves AND makes more R a week there.")
    print("=" * 110)
    weeks = max(1.0, (ns_end_bt - ns_start_bt).total_seconds() / 86400 / 7)
    split = ns["SPLIT"]
    print("   %-58s%8s%11s%18s%10s" % ("", "trades", "R a trade", "(1st / 2nd)", "R a week"))
    for a, cur, cand, lab in T5:
        s0, s1 = _stat(_pick(bt, a, *cur), split, weeks), _stat(_pick(bt, a, *cand), split, weeks)
        ok = s1[2] > s0[2] and s1[3] > s0[3] and s1[4] > s0[4]
        print("   %s" % lab)
        for nm, s in (("today: %s + %s" % (cur[0], cur[1].lower()), s0), ("candidate: %s + %s" % (cand[0], cand[1].lower()), s1)):
            print("     %-56s%8d%+11.2f%18s%+10.2f" % (nm, s[0], s[1], "(%+.2f / %+.2f)" % (s[2], s[3]), s[4]))
        print("     -> %s" % ("PASSES -- a candidate for your ruling (forward test first)" if ok else "today's setting stays"))


# ============================================================================================================
# TEST 6 -- new markets, same rules (idea 1): fetch 15 months of candles from MT5, then the three tested profiles
# ============================================================================================================
NEW_MKTS = {"SILVER": ("XAGUSDm", 0.0035), "US30": ("US30m", 0.0025), "ETH": ("ETHUSDm", 0.006),
            "USDJPY": ("USDJPYm", 0.0012), "EURJPY": ("EURJPYm", 0.0012)}
PROFILES = [("E", "SIMPLE", "today's entry + 2.5-move target"), ("B", "SIMPLE4", "break entry + 4-move target"),
            ("E", "RUNNER", "today's entry + runner")]


def _fetch(sym):
    """Price files for a market the bot does not trade: written once to data/raw, never overwriting anything.
    Returns (status, spread as a fraction of price or None)."""
    p1, p4 = os.path.join("data", "raw", "%s_1h.csv" % sym), os.path.join("data", "raw", "%s_4h.csv" % sym)
    spread = None
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            info = mt5.symbol_info(sym)
            if info is None:
                return "not offered by the broker", None
            mt5.symbol_select(sym, True)
            tick = mt5.symbol_info_tick(sym)
            px = float(tick.bid) if tick is not None and tick.bid else None
            if px:
                spread = float(info.spread) * float(info.point) / px
            if not (os.path.exists(p1) and os.path.exists(p4)):
                for tf, path, min_bars in ((mt5.TIMEFRAME_H1, p1, 3000), (mt5.TIMEFRAME_H4, p4, 700)):
                    r = mt5.copy_rates_range(sym, tf, datetime(2025, 6, 1), datetime.now())
                    if r is None or len(r) < min_bars:
                        return "not enough history at the broker (%s bars)" % (0 if r is None else len(r)), None
                    d = pd.DataFrame(r)
                    d["time"] = pd.to_datetime(d["time"], unit="s")
                    d = d.rename(columns={"tick_volume": "volume"})[["time", "open", "high", "low", "close", "volume"]]
                    d.to_csv(path, index=False)
                return "fetched from MT5", spread
    except ImportError:
        pass
    except Exception as ex:
        return "MT5 error: %s" % ex, None
    if os.path.exists(p1) and os.path.exists(p4):
        return "on file", spread
    return "no MT5 connection and no price file", None


def test6(ns):
    print("\n" + "=" * 110)
    print("TEST 6 -- NEW MARKETS, SAME RULES (idea 1): %s" % ", ".join(NEW_MKTS))
    print("          the three tested profiles, no filters, one position at a time, after the spread the broker quotes now.")
    print("          AGREED IN ADVANCE: a market qualifies for a live trial at the smallest size if one profile makes over")
    print("          +0.10R a trade in BOTH halves with 30+ trades in each half. 5 markets x 3 profiles = 15 tries, so a")
    print("          pass is a candidate, not proof -- the live trial is the proof.")
    print("=" * 110)
    weeks = max(1.0, (ns_end_bt - ns_start_bt).total_seconds() / 86400 / 7)
    split = ns["SPLIT"]
    known_costs = [v for v in ns["COST"].values() if v == v]
    base_cost = float(np.median(known_costs)) if known_costs else 0.0002
    live_tpl = dict(next(iter(ns["LIVE"].values()))) if ns.get("LIVE") else {}
    qualified = []
    for a, (sym, slp) in NEW_MKTS.items():
        status, sp = _fetch(sym)
        if status not in ("fetched from MT5", "on file"):
            print("   %-8s %-9s skipped: %s" % (a, sym, status))
            continue
        ns["SYM"][a] = sym
        ns["MIN_RR"][a] = 0.5
        ns["MIN_SL_PCT"][a] = slp
        if live_tpl:
            ns["LIVE"][a] = dict(live_tpl)
        for h in range(24):
            ns["COST"][(a, h)] = sp if sp else base_cost
        try:
            rows = pd.DataFrame(ns["run_market"](a))
        except Exception as ex:
            print("   %-8s %-9s failed: %s: %s" % (a, sym, type(ex).__name__, ex))
            continue
        print("   %-8s %-9s (%s; spread %s)" % (a, sym, status, ("%.4f%%" % (100 * sp)) if sp else "median of our markets"))
        for e, m, lab in PROFILES:
            s = _stat(_pick(rows, a, e, m), split, weeks) if len(rows) else (0, float("nan"), float("nan"), float("nan"), 0.0, 0, 0)
            ok = s[5] >= 30 and s[6] >= 30 and s[2] > 0.10 and s[3] > 0.10
            if ok:
                qualified.append((a, lab))
            print("     %-36s%7d tr%+9.2f R%18s%+9.2f R/wk   %s" % (lab, s[0], s[1], "(%+.2f / %+.2f)" % (s[2], s[3]), s[4],
                                                               "QUALIFIES" if ok else ""))
    print("   qualified: " + (", ".join("%s (%s)" % q for q in qualified) if qualified else "none"))


# ============================================================================================================
# WEEKLY -- the paper ideas on candles they have never seen (ideas 2 to 6)
# ============================================================================================================
VARIANTS = [("BTC pause-then-turn + runner (idea 2)", ("BTC", "E2", "RUNNER")),
            ("GOLD runner exit (idea 5)", ("GOLD", "B", "RUNNER")),
            ("USTEC runner exit (idea 5)", ("USTEC", "B", "RUNNER")),
            ("EURUSD every proof -- reversals and spikes back in (idea 4)", ("EURUSD", "E", "SIMPLE")),
            ("GBPAUD every proof -- reversals and spikes back in (idea 4)", ("GBPAUD", "E", "SIMPLE"))]
POCKETS = [("USTEC", "open FVG (4H)"), ("USTEC", "4H brain level"), ("GOLD", "1H brain level"), ("GOLD", "EMA 20 (1H)")]


def _scan_bounces(ns, start, pockets):
    """The Test 1c bounce, only for the chosen market / level pockets, on candles after `start`."""
    rows = []
    for a in sorted({p[0] for p in pockets}):
        M = _market(ns, a)
        c1, hi1, lo1, a1, t1 = M["c1"], M["hi1"], M["lo1"], M["a1"], M["t1"]
        want = {typ for (mk, typ) in pockets if mk == a}
        last_lvl, free = {}, {}
        for i in range(210, len(c1) - 1):
            if t1[i] <= start:
                continue
            atr = a1[i]
            if not (atr == atr and atr > 0):
                continue
            tol = 0.25 * atr
            for want_low in (True, False):
                d = 1 if want_low else -1
                for (typ, l, h) in _levels_at(M, i, want_low):
                    if typ not in want:
                        continue
                    if free.get(typ) is not None and t1[i] < free[typ]:
                        continue
                    if want_low:
                        ok = c1[i - 1] > h and lo1[i] <= h + tol and c1[i] > h and lo1[i] >= l - 2 * atr
                    else:
                        ok = c1[i - 1] < l and hi1[i] >= l - tol and c1[i] < l and hi1[i] <= h + 2 * atr
                    if not ok:
                        continue
                    key = (typ, d, round(l, 6), round(h, 6))
                    if key in last_lvl and (t1[i] - last_lvl[key]) < pd.Timedelta(hours=24):
                        continue
                    last_lvl[key] = t1[i]
                    e = float(c1[i])
                    stop = float(lo1[i] - 0.3 * atr) if d == 1 else float(hi1[i] + 0.3 * atr)
                    floor = ns["MIN_SL_PCT"].get(a, 0.0) * e
                    if abs(e - stop) < floor:
                        stop = e - d * floor
                    if abs(e - stop) > 5 * atr:
                        stop = e - d * 5 * atr
                    risk = abs(e - stop)
                    if risk <= 0 or 2.5 * atr / risk < ns["MIN_RR"].get(a, 0.5):
                        continue
                    R, why, jx = ns["exit_run"](a, "SIMPLE", d, e, stop, i, t1, hi1, lo1, c1, a1,
                                               M["pl"] if d == 1 else M["ph"], float(atr))
                    cost = ns["COST"][(a, t1[i].hour)] * e / risk
                    open_ = jx >= len(c1) - 1 and (t1[jx] - t1[i]) < pd.Timedelta(days=7)
                    rows.append(dict(asset=a, type=typ, t=t1[i], net=R - cost, open=open_))
                    free[typ] = t1[jx]
    return pd.DataFrame(rows)


def variants_forward(ns, fdf):
    print("\n" + "=" * 110)
    print("PAPER IDEAS ON NEW CANDLES (ideas 2 to 6) -- same forward window as Test 2; measured, never traded")
    print("=" * 110)
    print("   %-60s%10s%8s%12s" % ("idea", "finished", "open", "R a trade"))

    def show(lab, g):
        g_done = g[~g["open"]] if len(g) and "open" in g else g
        print("   %-60s%10d%8d%12s" % (lab[:59], len(g_done), (len(g) - len(g_done)),
                                     ("%+.2f" % g_done.net.mean()) if len(g_done) else "-"))
    lc = {}
    for a in MIX:
        h1 = ns["load"](ns["SYM"][a], "1h")
        lc[a] = h1.index[-1] + pd.Timedelta(hours=1)

    def _mark_open(g):
        if not len(g):
            return g.assign(open=[])
        return g.assign(open=[(r.exit_t >= lc.get(r.asset, r.exit_t)) and (r.exit_t - r.t < pd.Timedelta(days=7))
                              for r in g.itertuples()])
    for lab, (a, e, m) in VARIANTS:
        g = one_at_a_time(fdf[(fdf.asset == a) & (fdf.entry == e) & (fdf["mode"] == m)]) if len(fdf) else fdf
        show(lab, _mark_open(g) if len(g) else pd.DataFrame(columns=["net", "open"]))
    try:
        ns1 = research_ns(variant="1H")
        f1 = run_window(ns1, FREEZE + pd.Timedelta(minutes=1), pd.Timestamp("2100-01-01"))
        g = one_at_a_time(mix_rows(f1)) if len(f1) else f1
        show("1H break, every market (idea 3)", _mark_open(g) if len(g) else pd.DataFrame(columns=["net", "open"]))
    except Exception as ex:
        print("   1H break (idea 3): skipped -- %s" % ex)
    b = _scan_bounces(ns, FREEZE, POCKETS)
    for (a, typ) in POCKETS:
        g = b[(b.asset == a) & (b.type == typ)] if len(b) else pd.DataFrame(columns=["net", "open"])
        show("%s bounce off %s (idea 6)" % (a, typ), g)
    print("   each idea graduates to a proper test only after 20+ finished forward trades that stay positive.")


# ============================================================================================================
def main():
    weekly = "--weekly" in sys.argv
    print("ALL TESTS -- %s -- %s -- read-only, nothing here changes the bot"
          % (datetime.now().strftime("%d %b %Y %H:%M"), "WEEKLY run (forward test + supply + council)" if weekly else "full research run"))
    global ns_start_bt, ns_end_bt
    try:
        ns = research_ns()
    except Exception as ex:
        print("\nALL TESTS SKIPPED: %s" % ex)
        return
    ns_start_bt, ns_end_bt = ns["START"], ns["END"]
    print("\n   (building the tested backtest record: %s -> %s ...)" % (ns_start_bt.strftime("%d %b %Y"), ns_end_bt.strftime("%d %b %Y")))
    bt = run_window(ns, ns_start_bt, ns_end_bt)
    if not weekly:
        for label, fn, args in (("TEST 1", test1_levels, (ns,)), ("TEST 1c", test1c_edge, (ns, bt))):
            try:
                fn(*args)
            except Exception as ex:
                print("%s FAILED: %s: %s" % (label, type(ex).__name__, ex))
    fdf = pd.DataFrame()
    try:
        fdf = test2(ns, bt)
    except Exception as ex:
        print("TEST 2 FAILED: %s: %s" % (type(ex).__name__, ex))
    ns["START"], ns["END"] = ns_start_bt, ns_end_bt
    try:
        variants_forward(ns, fdf if fdf is not None else pd.DataFrame())
    except Exception as ex:
        print("PAPER IDEAS FAILED: %s: %s" % (type(ex).__name__, ex))
    if weekly:
        for fn in (weekly_supply, weekly_council):
            try:
                fn()
            except Exception as ex:
                print("%s FAILED: %s: %s" % (fn.__name__, type(ex).__name__, ex))
        return
    rec = None
    try:
        rec = test3(ns, bt, fdf if fdf is not None else pd.DataFrame())
    except Exception as ex:
        print("TEST 3 FAILED: %s: %s" % (type(ex).__name__, ex))
    try:
        test3b(rec)
    except Exception as ex:
        print("TEST 3b FAILED: %s: %s" % (type(ex).__name__, ex))
    try:
        test4(ns, bt)
    except Exception as ex:
        print("TEST 4 FAILED: %s: %s" % (type(ex).__name__, ex))
    try:
        test5(ns, bt)
    except Exception as ex:
        print("TEST 5 FAILED: %s: %s" % (type(ex).__name__, ex))
    try:
        test6(ns)
    except Exception as ex:
        print("TEST 6 FAILED: %s: %s" % (type(ex).__name__, ex))
    try:
        p = write_gallery(ns)
        print("\nGALLERY -- example charts to go through: %s (%d charts)" % (p, min(80, len(GALLERY))) if p else "\nGALLERY -- nothing to draw")
    except Exception as ex:
        print("GALLERY FAILED: %s: %s" % (type(ex).__name__, ex))
    print("\nDone. Before the council, the 5-dollar minimum and slippage. Everything except the forward test was measured on")
    print("the backtest window -- the forward test is the only part the rules have never seen.")


ns_start_bt = ns_end_bt = None

if __name__ == "__main__":
    main()
