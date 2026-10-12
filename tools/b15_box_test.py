"""B15 BOX TEST (Desire 12 Oct, rule set Part 4) -- READ-ONLY, built on the B14 box test. It runs the B15 code itself
(ns_engine, ns_package, line_map: the very files the bot runs) over the saved price files, every market, about 16
months, once per row, and prints one table. It changes nothing: it reads what tools\\b14_box_test.py reads
(data\\raw\\*_1h.csv, config\\config.json, the logs for the spreads, MT5's own 30-minute / daily / weekly candles,
read-only) and writes only its report logs\\b15_box_test_<date>.txt.

It uses the B14 test's loaders and trade simulation as they are (so the two can never drift apart), and adds:
  - rows S0-S14 (rule set 4.1; S13-S14 = decision D8, Desire 12 Oct 13:24 "test all 3"). S0 = the switch off (B13): its engine entries are checked IDENTICAL against the B13
    files in tools\\b14_b13ref; package entries are listed beside the check (the B14.1 stop);
  - the 4H-clock entries (small lines, and big-line signals sent to wait);
  - per path: breaks, entries, failed attempts, ends and why (e.g. a stop too wide); staircase steps; channel targets
    hit; locks (must be 0 under B15) (4.3);
  - beside every row: the worst trade and the worst run of losses (4.3);
  - the brain table: every S0 and S1 trade tagged with the engine's own 4H brain state at its break -- per state,
    with/against the trend, main states only; a group needs 50+ trades before it can earn a role (4.2);
  - every market starts on the same date, and the tags are compared with a brain started 8 weeks later (the trap in
    4.2: a tag that depends on where the brain's history starts);
  - the gate (4.4): does S1 beat S0 over the same weeks?

usage (rule set 4.4: on Stephen's side copy of the bot folder, with the bot's python):
  python tools\\b15_box_test.py                         all rows, all markets (an hour or more)
  python tools\\b15_box_test.py --rows S0,S1 --markets GOLD,BTC     a quick look"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
_sp = importlib.util.spec_from_file_location("b14_box_test", os.path.join(ROOT, "tools", "b14_box_test.py"))
B14 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(B14)
NE, NP, LM = B14.NE, B14.NP, B14.LM
say = B14.say
# The brain's whole history: live keeps about 66 days (enough there -- a rebuild goes back REPLAY_DAYS = 30), but
# this test replays ~16 months in one look, and a break months back must still find the state the bot knew then.
NE.BRAIN4_KEEP = 10 ** 7
UP, DOWN = tuple(NE.UP_STATES), tuple(NE.NSEngine._DOWN15)   # the engine's own with-the-trend sets (rule set 4.1)
LATER_WEEKS = 8                                              # the second brain for the tag check starts this late


def _txt(w):
    """A reason without its numbers, for counting."""
    w = re.sub(r"[-+]?\d[\d.,]*(e[-+]?\d+)?", "#", str(w))
    return re.sub(r"\s+", " ", w).strip()[:72]


def _path(s):
    if s.get("clock") == "4H":
        r = s.get("route4")
        return "4H clock: small line" if r == "small" else (
            "4H clock: big line, by the brain's phase" if r == "phase" else "4H clock: big line sent to wait")
    if s.get("b15"):
        return "hourly: big line"
    return "hourly (B13)" if s.get("b_t") is not None else "not broken yet"


GLOSS = {"REF_INVALIDATED": "REF_INVALIDATED (killed: the kill line broke)", "NS_EXPIRED": "NS_EXPIRED (out of time)",
         "NS_DUPLICATE": "NS_DUPLICATE (the same line as another setup)",
         "NS_ENTERED": "NS_ENTERED (entered, or handed to the package)"}


class Box15(B14.BoxEngine):
    """The live engine, every candle counted as live (as the B14 test), and it counts what each path did. Test only:
    each count is taken around the engine's own step, so nothing the engine decides changes."""
    def _c(self):
        return self.__dict__.setdefault("c15", collections.Counter())

    def _b15_clock(self, st, cfg, kn, s, i, t, rb, emit, out):
        r = NE.NSEngine._b15_clock(self, st, cfg, kn, s, i, t, rb, emit, out)
        self._c()["%s | breaks" % _path(s)] += 1
        return r

    def _route4(self, st, cfg, s, i, t, e, atr, b14, pkg_route, emit, out):
        self._c()["4H clock: big line sent to wait | sent (%s)" % (pkg_route or "-")] += 1
        return NE.NSEngine._route4(self, st, cfg, s, i, t, e, atr, b14, pkg_route, emit, out)

    def _fail4(self, st, s, kn, why, emit, t, out):
        self._c()["%s | attempt failed: %s" % (_path(s), _txt(why))] += 1
        return NE.NSEngine._fail4(self, st, s, kn, why, emit, t, out)

    def _skip4(self, s, why, emit, out):
        self._c()["%s | no trade at that close, still waiting: %s" % (_path(s), _txt(why))] += 1
        return NE.NSEngine._skip4(self, s, why, emit, out)

    def _end4(self, st, s, why, emit, t, out):
        self._c()["%s | ended: %s" % (_path(s), _txt(why))] += 1
        return NE.NSEngine._end4(self, st, s, why, emit, t, out)

    def _end(self, st, s, reason, emit, t):
        if reason != "PKG4_END":                                  # (a 4H end is counted with its reason above)
            self._c()["%s | ended: %s" % (_path(s), GLOSS.get(reason, _txt(reason)))] += 1
        return NE.NSEngine._end(self, st, s, reason, emit, t)


class Replay15(Box15):
    """A re-armed package setup replayed from its cancel: no new setups are born in this run (as the B14 test)."""
    def _register(self, st, s, atr4_at, create, emit):
        return NE.NSEngine._register(self, st, s, atr4_at, False, False)


def run_signals15(asset, df1, frames, mp, pcfg):
    """One market, one row: tools\\b14_box_test.run_signals, plus the 4H-clock entries and the path counts.
    -> (entries, log, path counts, the engine's final state)"""
    end = df1.index[-1] + pd.Timedelta(hours=1)
    eng = Box15(asset)
    r = eng.update(None, df1, None, pcfg, None, now=end, mp=mp)
    st = r["state"]
    entries, log, cnt = [], collections.Counter(), collections.Counter(eng._c())
    for p in r["proofs"]:
        entries.append(dict(src="hourly", proof=p))
        log["taken at once (hourly)"] += 1
    for p in r.get("pkg4") or []:
        entries.append(dict(src="4H clock", proof=p))
        log["taken on the 4H clock"] += 1
    todo = [rec for rec in (st.get("pkg") or []) if rec.get("status") == "watching"]
    seen_ids = set()
    while todo:                                                    # the B13/B14 package (S0, S6): as the B14 test
        rec = todo.pop(0)
        log["package hand-overs"] += 1
        now = min(pd.Timestamp(rec["tE"]) + pd.Timedelta(hours=26), end)
        sub = {"pkg": [rec], "setups": [], "pkg_ready": []}
        prs, _ev = NP.step(asset, sub, df1, pcfg, now=now, frames=frames, mp=mp)
        for p in prs:
            entries.append(dict(src="package", proof=p))
        log["package %s" % rec.get("status")] += 1
        if rec.get("status") == "cancelled":
            log["package cancel: %s" % _txt(rec.get("why_done") or rec.get("why"))[:44]] += 1
        for s in sub.get("setups") or []:
            if (s["id"], s.get("rearms")) in seen_ids:
                continue
            seen_ids.add((s["id"], s.get("rearms")))
            log["package re-arms"] += 1
            st2 = NE.NSEngine.new_state()
            st2["setups"] = [s]
            st2["next_id"] = int(s["id"]) + 1
            st2["last_1h"] = pd.Timestamp(s["t_break"]).floor("h")
            st2["rules"] = NE.rules_stamp(NE.market_settings(asset, pcfg), None)
            e2 = Replay15(asset)
            r2 = e2.update(st2, df1, None, pcfg, None, now=end, mp=mp)
            cnt.update(e2._c())
            for p in r2["proofs"]:
                entries.append(dict(src="re-armed", proof=p))
                log["re-armed: taken at once"] += 1
            for p in r2.get("pkg4") or []:
                entries.append(dict(src="4H clock", proof=p))
                log["re-armed: taken on the 4H clock"] += 1
            todo += [x for x in (r2["state"].get("pkg") or []) if x.get("status") == "watching"]
    return entries, log, cnt, st


def book15(entries, D, exits_on):
    """tools\\b14_box_test.book (one position per market at a time, in time order), keeping each trade's fields."""
    rows, free = [], pd.Timestamp("1900-01-01")
    for e in sorted(entries, key=lambda x: pd.Timestamp(x["proof"]["fields"]["ns_candle"])):
        t_in = pd.Timestamp(e["proof"]["fields"]["ns_candle"])
        if t_in < free:
            continue
        r = B14.simulate(e, D["m30"], D["mp"], D["cost"], D["usd"], exits_on, h1=D["df"])
        if r is None:
            continue
        r["f"] = e["proof"]["fields"]
        rows.append(r)
        free = r["t_out"]
    return rows


def stats15(rows, weeks):
    s = B14.stats(rows, weeks)
    s.update(worst_usd=0.0, worst_R=0.0, lrun=0, lrun_usd=0.0)
    if rows:
        srt = sorted(rows, key=lambda x: x["t_out"])
        w = min(srt, key=lambda x: x["usd"])
        s["worst_usd"], s["worst_R"] = w["usd"], w["R"]
        run, run_usd = 0, 0.0
        for x in srt:                                  # the longest run of losing trades (and what it cost)
            if x["R"] < 0:
                run, run_usd = run + 1, run_usd + x["usd"]
                if run > s["lrun"] or (run == s["lrun"] and run_usd < s["lrun_usd"]):
                    s["lrun"], s["lrun_usd"] = run, run_usd
            else:
                run, run_usd = 0, 0.0
    return s


def rows_table15(pc0):
    """Each row = the config the bot reads, with one change (rule set 4.1). S1 = B15 as ruled."""
    def mk(on, knobs=None, btc=None, extra=None):
        pc = copy.deepcopy(pc0)
        pc["b14_rules_enabled"] = bool(on)
        pc["b14_knobs"] = dict(knobs or {})
        pc.pop("b15_small_4h_only", None)
        pc.update(extra or {})
        nm = pc.setdefault("ns_markets", {})
        if on and "BTC" in nm:
            nm["BTC"] = dict(nm["BTC"], **(btc or {"target_atr": 4.0, "exit": "FIXED"}))   # 2.3 (as B14 6.1)
        return dict(pc=pc, exits=True)
    return collections.OrderedDict([
        ("S0", ("B13 rules (the switch off) -- today", mk(False))),
        ("S1", ("B15 as ruled", mk(True))),
        ("S2", ("S1 without the staircase", mk(True, {"stair": False}))),
        ("S3", ("S1 without channel targets", mk(True, {"channel_target": False}))),
        ("S4", ("S1, channel targets at the line itself", mk(True, {"ch_edge": "line"}))),
        ("S5", ("S1 with BTC on its runner", mk(True, btc={"target_atr": None, "exit": "RUNNER"}))),
        ("S6", ("B13 + only the small-line 4H/4H path", mk(False, extra={"b15_small_4h_only": True}))),
        ("S7", ("S1 + the bigger-timeframe vote (package)", mk(True, {"pkg_majority": True}))),
        ("S8", ("S1, brain phase picks the clock (all lines)", mk(True, {"clock": "phase_all"}))),
        ("S9", ("S1, brain phase picks the clock (small)", mk(True, {"clock": "phase_small"}))),
        ("S10", ("S1, signals sent to wait: 24 h at most", mk(True, {"pkg_wait": "24h"}))),
        ("S11", ("S1, entry close beats the peak by 1/4 move", mk(True, {"trigger_clear": 0.25}))),
        ("S12", ("S1, entry close beats the peak by 1/2 move", mk(True, {"trigger_clear": 0.5}))),
        ("S13", ("S1, stop too wide: no trade, keep waiting (D8 B)", mk(True, {"wide": "wait"}))),
        ("S14", ("S1, stop too wide: the setup ends (D8 C)", mk(True, {"wide": "end"}))),
    ])


def sig_key(a, pcfg):
    return json.dumps({"m": a, "ms": NE.market_settings(a, pcfg),
                       "pc": {k: pcfg.get(k) for k in ("b14_rules_enabled", "b14_knobs", "b15_small_4h_only")},
                       "nm": (pcfg.get("ns_markets") or {}).get(a)}, sort_keys=True, default=str)


# ---- the brain tags (rule set 4.2) ------------------------------------------------------------------------------------
def brain_alone(asset, df1):
    """The engine's 4H brain on its own, fed exactly as update() feeds it -> its dated states."""
    eng = NE.NSEngine(asset)
    st = NE.NSEngine.new_state()
    df4 = NE.hourly_to_4h(df1)
    t4 = NE.close_times(df4, 4)
    hi4, lo4, c4 = (df4[c].astype(float).values for c in ("high", "low", "close"))
    eng._brain4_step(st, t4, hi4, lo4, c4, NE.close_times(df1, 1)[-1])
    return {"brain4_states": st.get("brain4_states") or []}


def tag(hist, f):
    tb = f.get("ns_b_t")
    if tb in (None, "None", "", "NaT"):
        return None
    return NE.NSEngine._phase_at(hist, tb)


def with_trend(d, ph):
    return (d == 1 and ph in UP) or (d == -1 and ph in DOWN)


def main_only(d, ph):
    return (d == 1 and ph == "MAIN_UP") or (d == -1 and ph == "MAIN_DOWN")


def _line(name, rows):
    if not rows:
        return "   %-44s %6d" % (name, 0)
    R = np.array([x["R"] for x in rows])
    return "   %-44s %6d %6.1f %8.1f %7.3f %9.1f" % (name, len(rows), 100.0 * float((R > 0).mean()), float(R.sum()),
                                                     float(R.mean()), sum(x["usd"] for x in rows))


def brain_table(rid, rows, later):
    say("\n--- %s: every trade by the engine's 4H brain state at its break ---" % rid)
    say("   %-44s %6s %6s %8s %7s %9s" % ("group", "trades", "win%", "total R", "avg R", "$ total"))
    by = collections.defaultdict(list)
    for x in rows:
        by[x["ph"] or "unknown (no state at the break)"].append(x)
    for k in sorted(by, key=lambda k_: -len(by[k_])):
        say(_line(k, by[k]))
    known = [x for x in rows if x["ph"]]
    w = [x for x in known if with_trend(x["d"], x["ph"])]
    ag = [x for x in known if not with_trend(x["d"], x["ph"])]
    mw = [x for x in known if main_only(x["d"], x["ph"])]
    mn = [x for x in known if not main_only(x["d"], x["ph"])]
    say(_line("with the trend", w))
    say(_line("against the trend", ag))
    say(_line("with the trend -- main states only", mw))
    say(_line("everything else (vs main states only)", mn))
    for nm, a_, b_ in (("with / against", w, ag), ("main only / the rest", mw, mn)):
        ok = len(a_) >= 50 and len(b_) >= 50
        gap = (np.mean([x["R"] for x in a_]) - np.mean([x["R"] for x in b_])) if a_ and b_ else float("nan")
        say("   %s: %d / %d trades -- %s; avg R gap %+.3f" % (
            nm, len(a_), len(b_), "both groups have 50+" if ok else "NOT ENOUGH (a group needs 50+ before the brain "
            "can earn a role)", gap))
    if rid == "S1":                                     # the live tag (ns_phase_4h) must equal this table's tag
        eng_t = [x for x in rows if x["f"].get("ns_phase_4h") is not None]
        same = sum(1 for x in eng_t if x["f"]["ns_phase_4h"] == x["ph"])
        say("   the engine's own tag on the trade (ns_phase_4h, what live records) = this table's tag: %d of %d trades%s"
            % (same, len(eng_t), "" if same == len(eng_t) else "  <-- DIFFERENT: tell Claude"))
    lt = [x for x in rows if x.get("ph_later", "-") != "-"]
    if lt:
        s_ = sum(1 for x in lt if x["ph_later"] == x["ph"])
        sw = sum(1 for x in lt if x["ph"] and x["ph_later"] and with_trend(x["d"], x["ph"]) == with_trend(x["d"], x["ph_later"]))
        say("   brain started %d weeks later (the 4.2 trap): the same state on %d of %d trades after %s, the same "
            "with/against verdict on %d" % (LATER_WEEKS, s_, len(lt), later, sw))


def fp(path):
    try:
        b = open(path, "rb").read().replace(b"\r\n", b"\n")
        return hashlib.sha256(b).hexdigest()[:16].upper()
    except OSError:
        return "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all")
    ap.add_argument("--markets", default="all")
    ap.add_argument("--months", type=float, default=16.0)
    ap.add_argument("--replay-from", default="2026-09-22")
    args = ap.parse_args()
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%d_%H%M")
    say("=" * 118)
    say("B15 BOX TEST -- %s (read-only) | rules %s" % (stamp, NE.RULES_V))
    say("files tested: " + " | ".join("%s %s" % (os.path.basename(p), fp(p)) for p in (
        "src/execution/ns_engine.py", "src/execution/ns_package.py", "src/execution/line_map.py",
        "tools/b14_box_test.py", "tools/b15_box_test.py", "config/config.json")))
    say("=" * 118)
    cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    pc0 = cfg.get("phase_config", {}) or {}
    markets = list(B14.MARKETS) if args.markets == "all" else [m.strip().upper() for m in args.markets.split(",")]
    table = rows_table15(pc0)
    want = list(table) if args.rows == "all" else [r.strip().upper() for r in args.rows.split(",")]
    usd, spreads = B14.lot_values()
    paid = B14.log_spreads()
    try:
        from src.execution.shadow_trader import FRICTION_PENALTIES as _FP, _DEFAULT_FRICTION as _DF
    except Exception:
        _FP, _DF = {}, 0.0003
    RefEngine, RefPkg = B14.load_b13_ref()
    raw = {}
    for a in markets:
        df, note = B14.load_1h(B14.MARKETS[a])
        if df is None:
            say("%-7s skipped: %s" % (a, note))
            continue
        raw[a] = (df, note)
    if not raw:
        say("no market could be loaded -- nothing to test")
        return
    # 4.2: every market's brain (and everything else) starts on the same date
    t_start = min(df.index[-1] for df, _n in raw.values()) - pd.Timedelta(days=int(args.months * 30.4))
    say("every market starts on %s (%.0f months before the earliest file end)" % (str(t_start)[:16], args.months))
    data = {}
    for a, (df, note) in raw.items():
        df = df[df.index >= t_start]
        late = df.index[0] > t_start + pd.Timedelta(days=1)
        fr = B14.mt5_frames(B14.MARKETS[a])
        approx = "M30" not in fr
        m30 = fr.get("M30") if not approx else B14.m30_from_1h(df)
        m30 = m30[m30.index >= df.index[0]]
        d1, w1 = fr.get("D1"), fr.get("W1")
        mp = LM.MarketMap(a, df, d1, w1)
        frames = {"M30": m30, "D1": d1 if d1 is not None else mp.d1, "W1": w1 if w1 is not None else mp.w1}
        cost = paid.get(a, spreads.get(a, _FP.get(a, _DF)))
        weeks = (df.index[-1] - df.index[0]).total_seconds() / (7 * 86400.0)
        t_later = df.index[0] + pd.Timedelta(weeks=LATER_WEEKS)
        data[a] = dict(df=df, m30=m30, mp=mp, frames=frames, cost=cost, usd=usd[a], weeks=weeks, approx=approx,
                       hist=None, later=brain_alone(a, df[df.index >= t_later]), t_later=t_later)
        say("%-7s %s%s | 30m: %s | map: %d 4H candles | spread %.5f%% | $%.4g per 1.0 at the smallest lot" % (
            a, note, (" | STARTS LATER: its file begins %s" % str(df.index[0])[:16]) if late else "",
            ("MT5" if fr.get("_src") != "export" else "the export files") if not approx else
            "APPROXIMATE (made from the hourly file -- MT5 gave none)", mp.n, 100 * cost, usd[a]))
    res = collections.OrderedDict()
    sig_cache = {}
    for rid in want:
        if rid not in table:
            continue
        name, spec = table[rid]
        res[rid] = dict(name=name, per={}, log=collections.Counter(), cnt=collections.Counter(), rows=[])
        for a, D in data.items():
            pcfg = spec["pc"]
            key = sig_key(a, pcfg)
            if key not in sig_cache:
                sig_cache[key] = run_signals15(a, D["df"], D["frames"], D["mp"], pcfg)
            entries, lg, cnt, st = sig_cache[key]
            if D["hist"] is None:                        # one brain history per market (the brain has no rules)
                D["hist"] = {"brain4_states": list(st.get("brain4_states") or [])}
            rows = book15(entries, D, spec["exits"])
            for x in rows:
                x["asset"] = a
                x["ph"] = tag(D["hist"], x["f"])
                x["ph_later"] = tag(D["later"], x["f"]) if x["t_in"] >= D["t_later"] + pd.Timedelta(weeks=2) else "-"
            res[rid]["per"][a] = (stats15(rows, D["weeks"]), rows)
            res[rid]["log"].update(lg)
            res[rid]["cnt"].update(cnt)
            res[rid]["rows"] += rows
        say("   %s done (%.0f s)" % (rid, time.time() - t0))
    weeks = float(np.mean([D["weeks"] for D in data.values()]))
    # ---- first check: S0 against the B13 files
    say("\n" + "=" * 118 + "\nFIRST CHECK: the B15 code with its switch off (S0) against the B13 engine and package "
        "(tools\\b14_b13ref)\n" + "=" * 118)
    if RefEngine is None:
        say("   NOT RUN: tools\\b14_b13ref\\ns_engine.py and ns_package.py are missing")
    elif "S0" not in res:
        say("   NOT RUN: row S0 was not asked for")
    else:
        def _split(entries):
            eng, pk = [], []
            for e in entries:
                f = e["proof"]["fields"]
                (pk if f.get("ns_entry") == "PKG" else eng).append((str(e["proof"]["key"]), f["ns_candle"],
                                                                     round(f["ns_stop"], 8)))
            return sorted(eng), sorted(pk)
        bad = []
        for a, D in data.items():
            ref_entries, _lg = B14.run_signals(a, D["df"], D["frames"], D["mp"], table["S0"][1]["pc"],
                                               eng_cls=RefEngine, pkg=RefPkg, ref=True)
            s0_entries = sig_cache[sig_key(a, table["S0"][1]["pc"])][0]
            four = sum(1 for e in s0_entries if e["src"] == "4H clock")
            e_ref, p_ref = _split(ref_entries)
            e_s0, p_s0 = _split(s0_entries)
            same = e_ref == e_s0 and four == 0
            kt_ref, kt_s0 = {x[:2] for x in p_ref}, {x[:2] for x in p_s0}
            say("   %-7s %s  (engine entries: %d B13, %d B15 switch off | 4H-clock entries: %d (must be 0) | package "
                "entries: %d B13, %d B14.1 -- %d not taken, %d with a closer stop)" % (
                    a, "IDENTICAL" if same else "DIFFERENT", len(e_ref), len(e_s0), four, len(p_ref), len(p_s0),
                    len(kt_ref - kt_s0), sum(1 for x in p_ref if x[:2] in kt_s0 and x not in p_s0)))
            if not same:
                bad.append(a)
                for x in sorted(set(e_ref) ^ set(e_s0))[:5]:
                    say("           only in %s: %s" % ("B13" if x in e_ref else "B15", x))
            if kt_s0 - kt_ref:
                say("           package entries only in B14.1 (unexpected): %s" % sorted(kt_s0 - kt_ref)[:3])
        say("   RESULT: %s" % ("PASS -- with the switch off the B15 code's engine entries are exactly B13's (package "
                               "entries differ only by the B14.1 stop rule)" if not bad else
                               "FAIL for %s -- tell Claude before anything is switched on" % ", ".join(bad)))
    # ---- the rows
    say("\n" + "=" * 118 + "\nTHE ROWS (all markets together; %.0f weeks; money = the smallest lot after the spread)\n"
        % weeks + "=" * 118)
    say("%-4s %-44s %6s %6s %8s %8s %8s %8s | %-16s | %s" % ("row", "what changes", "trades", "win%", "total R", "$ total",
                                                             "$/week", "max drop", "worst trade", "worst losing run"))
    for rid, r in res.items():
        s = stats15(r["rows"], weeks)
        say("%-4s %-44s %6d %6.1f %8.1f %8.1f %8.2f %8.1f | $%7.1f (%+.2fR) | %d in a row, $%.1f" % (
            rid, r["name"][:44], s["n"], s["win"], s["R"], s["usd"], s["wk"], s["dd"], s["worst_usd"], s["worst_R"],
            s["lrun"], s["lrun_usd"]))
    say("\n   Each row from S2 on changes ONE thing from S1. S0 = today's rules. 'max drop' = the biggest fall of the "
        "running total, in $.")
    if "S0" in res and "S1" in res:                     # 4.4 the gate
        a0, a1 = stats15(res["S0"]["rows"], weeks), stats15(res["S1"]["rows"], weeks)
        say("\nGATE (4.4): S1 %s S0 on money ($%.1f vs $%.1f) and %s on R (%.1f vs %.1f), same %d markets, same weeks. "
            "If S1 doesn't beat S0, B15 stays off and we look at why." % (
                "BEATS" if a1["usd"] > a0["usd"] else "does NOT beat", a1["usd"], a0["usd"],
                "beats it" if a1["R"] > a0["R"] else "does NOT beat it", a1["R"], a0["R"], len(data)))
    # ---- per row details
    for rid in [x for x in ("S0", "S1", "S6") if x in res] + [x for x in res if x not in ("S0", "S1", "S6")]:
        r = res[rid]
        say("\n--- %s: %s ---" % (rid, r["name"]))
        if rid in ("S0", "S1", "S6"):
            for a, (s, rows) in r["per"].items():
                hows = collections.Counter(x["how"] for x in rows)
                srcs = collections.Counter(x["src"] for x in rows)
                say("   %-7s trades %3d  win %5.1f%%  R %6.1f  $ %8.1f  max drop $%.1f | exits %s | from %s" % (
                    a, s["n"], s["win"], s["R"], s["usd"], s["dd"], dict(hows), dict(srcs)))
        by_src = collections.defaultdict(list)
        for x in r["rows"]:
            by_src[x["src"]].append(x)
        for k in sorted(by_src):
            say(_line("trades from: %s" % k, by_src[k]))
        rr = r["rows"]
        ch_hit = sum(1 for x in rr if x["how"] == "target" and x["f"].get("ns_target_kind") not in (None, "", "normal"))
        say("   exits: staircase steps %d (in %d trades) | channel targets hit %d | locks %d%s" % (
            sum(x["steps"] for x in rr), sum(1 for x in rr if x["steps"]), ch_hit, sum(1 for x in rr if x["locked"]),
            "  <-- B15 has the lock OFF: must be 0" if rid not in ("S0", "S6") and any(x["locked"] for x in rr) else ""))
        say("   signals: %s" % ", ".join("%s %d" % kv for kv in sorted(r["log"].items())))
        if r["cnt"]:
            say("   per path (what the engine did):")
            for k, v in sorted(r["cnt"].items(), key=lambda kv: (kv[0].split(" | ")[0], -kv[1])):
                say("      %-42s %-72s %6d" % tuple(k.split(" | ", 1) + [v]) if " | " in k else "      %s %d" % (k, v))
    # ---- the brain table
    say("\n" + "=" * 118 + "\nTHE BRAIN TABLE (4.2): S0 and S1 trades by the engine's own 4H brain state at the break\n"
        + "=" * 118)
    say("   with the trend: a buy in MAIN_UP / NATURAL_RETRACEMENT / SECONDARY_RETRACEMENT, a sell in MAIN_DOWN / "
        "NATURAL_REBOUND / SECONDARY_REBOUND; anything else = against.")
    for rid in ("S0", "S1"):
        if rid in res:
            brain_table(rid, res[rid]["rows"], "%s (later for a market whose file starts later)" % str(
                min(D["t_later"] for D in data.values()) + pd.Timedelta(weeks=2))[:10])
    # ---- replays
    say("\n" + "=" * 118 + "\nREPLAYS: every trade since %s, today's rules (S0) and B15 (S1)\n" % args.replay_from
        + "=" * 118)
    t_from = pd.Timestamp(args.replay_from)
    for a in data:
        for rid in ("S0", "S1"):
            if rid not in res:
                continue
            rows = [x for x in res[rid]["per"].get(a, (None, []))[1] if x["t_in"] >= t_from]
            if not rows:
                continue
            say("   %s %s: %d trade(s)" % (a, rid, len(rows)))
            for x in rows:
                say("      %s %s %.6g -> %.6g %-22s %+.2fR $%+.1f  %s | brain %s | %s" % (
                    str(x["t_in"])[:16], "BUY " if x["d"] == 1 else "SELL", x["q"], x["px"], x["how"], x["R"],
                    x["usd"], x["src"], x["ph"] or "unknown", str(x["f"].get("ns_path") or "")[:60]))
    say("\n" + "=" * 118 + "\nGUARD CHECK: " + B14.guard_check())
    say("\nfinished in %.0f s" % (time.time() - t0))
    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", "b15_box_test_%s.txt" % stamp)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(B14.OUT) + "\n")
    print("report written:", out)


if __name__ == "__main__":
    main()
