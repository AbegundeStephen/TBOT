"""THE B14 LEARNER -- RL-1 for the new B14 rules (Desire 16 Oct 2026: "build it now, suggestions only -- a system that
really learns, improves and grows the bot"). READ-ONLY towards the bot: it never changes config.json or any bot file.

What it does every time it runs (the Saturday job runs it after the box test):
  1. REPLAY: the new B14 rules with the settings in force (phase_config.b14_tuning; none = the rules file's own) over
     27 months of saved prices, every market -- the box test's own code (tools/b14_box_test.py), so the learner can
     never drift from the bot. Two samples: the last 16 months, and the 11 months before them.
  2. TRY: versions of the settings, one lever at a time (tools/b14_rl_levers.json): every exit version (the ladder's
     rungs -- the same trades, only the exits re-run: quick) and a few entry versions (the clearance, the target
     window, the re-arms, BTC's fallback -- the whole replay re-run: slow), picked by the learner: untried ones first,
     then the most promising (Thompson sampling: each version's result is drawn from what it believes about it, so
     the test time goes where a win is most likely).
  3. LEARN: each version's record (better or worse than the settings in force, by how much, how sure, on each
     sample) is kept in logs/rl/b14_rl_knowledge.json and updated every run as new weeks of prices come in. A
     version clearly worse on either sample is retired. Around a version that keeps winning, it adds the next step
     further (the search grows toward what works).
  4. SUGGEST -- only when a version beats the settings in force on BOTH samples by more than chance, with the bar
     raised for every version tried this season (so trying many cannot produce a fluke), at least 100 trades in each
     sample, and the worst drop no more than 20% deeper on either. Score = total R (Desire 16 Oct). A suggestion is
     written to logs/rl/b14_rl_suggestions.jsonl with its evidence and the exact config line. Desire decides; the
     apply tool writes it (with a backup); nothing changes by itself. When the settings in force change, a new
     season starts and every version is judged again against them.
Most weeks the answer will be "nothing beats the settings in force" -- that is the learner working, not failing.

  python tools\\b14_rl.py               the weekly run (both samples, every exit version, up to 4 entry versions)
  python tools\\b14_rl.py --full 0      exit versions only (minutes)
  python tools\\b14_rl.py --show        what the learner knows so far -- runs nothing, changes nothing
Writes only under logs\\rl\\: the knowledge file, one line per test in b14_rl_experiments.jsonl, a suggestion line
when one passes, and the report b14_rl_<stamp>.txt."""
import argparse
import collections
import copy
import glob
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from statistics import NormalDist

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
_sp = importlib.util.spec_from_file_location("b14_box_test", os.path.join(ROOT, "tools", "b14_box_test.py"))
BT = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(BT)
R14 = BT.R14

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

LEVERS_FILE = os.path.join("tools", "b14_rl_levers.json")
DIR = os.path.join("logs", "rl")
KNOW = os.path.join(DIR, "b14_rl_knowledge.json")
EXPS = os.path.join(DIR, "b14_rl_experiments.jsonl")
SUGG = os.path.join(DIR, "b14_rl_suggestions.jsonl")
APPLIED = os.path.join(DIR, "b14_rl_applied.jsonl")      # written by tools/b14_rl_apply.py when Desire says yes
UNDO_MIN = 20         # trades since a change before the learner may suggest undoing it
ALPHA = 0.10          # the chance of a false suggestion the learner accepts for a whole season, shared by every try
DD_MAX = 1.20         # a suggestion may not make the worst drop more than 20% deeper, on either sample
MIN_TRADES = 100      # trades needed in each sample before a suggestion
RETIRE_Z = 1.645      # retired when 95% sure it is worse on a sample
MAX_STEPS = 4         # the search grows up to 4 steps away from the settings in force
REAL_MIN = 20         # recorded real-signal trades needed before the real-trade check counts
NEW_MIN = 30          # trades under the new B14 rules (after the install) needed before the forward check counts
GROUPS = ("new B14 (since the install)", "live + lane A (real signals)", "lane C (random entries)", "other lanes")
REPORT = []
TG = []               # the short Telegram summary the Saturday job sends (logs/rl/b14_rl_telegram.txt)


def lower_priority():
    """The learner runs below the live bot's priority (BTC trades at the weekend): best effort, said either way."""
    try:
        if os.name == "nt":
            import ctypes
            ok = ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
            return "below normal" if ok else "unchanged (Windows refused)"
        os.nice(10)
        return "nice +10"
    except Exception as e:
        return "unchanged (%s)" % e


def say(s=""):
    print(s, flush=True)
    REPORT.append(s)


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---- settings and levers ---------------------------------------------------------------------------------------------
def jsonable(settings):
    return {k: ([[float(a), float(b)] for a, b in v] if k == "LADDER" else v) for k, v in settings.items()}


def key_of(settings):
    return hashlib.sha1(json.dumps(jsonable(settings), sort_keys=True).encode()).hexdigest()[:12]


def load_levers():
    d = json.load(open(LEVERS_FILE, encoding="utf-8"))
    out = []
    for lv in d.get("levers") or []:
        base = lv["name"].split(".")[0]
        if base not in R14.TUNABLE:
            say("   lever %s skipped: not a setting of the rules (b14_rules.TUNABLE)" % lv["name"])
            continue
        out.append(lv)
    return out


def get_lever(settings, name):
    if name.startswith("LADDER."):
        part, k = name.split(".")[1][0], int(name.split(".")[1][1:]) - 1
        lad = settings["LADDER"]
        return None if k >= len(lad) else float(lad[k][0 if part == "t" else 1])
    return settings[name]


def set_lever(settings, name, value):
    """A copy of the settings with one lever changed, or None when the result breaks a rule of the settings."""
    s = copy.deepcopy(settings)
    if name.startswith("LADDER."):
        part, k = name.split(".")[1][0], int(name.split(".")[1][1:]) - 1
        lad = [list(x) for x in s["LADDER"]]
        if k >= len(lad):
            return None
        lad[k][0 if part == "t" else 1] = float(value)
        val, why = R14.check_setting("LADDER", lad)
        if why:
            return None
        s["LADDER"] = val
        return s
    val, why = R14.check_setting(name, value)
    if why:
        return None
    s[name] = val
    if not (s["TARGET_FROM"] < s["TARGET_TO"] <= s["TARGET_EXT"]):
        return None
    return s


def fmt(v):
    return ("%g" % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


# ---- the knowledge -------------------------------------------------------------------------------------------------
def new_season(base):
    return {"base_key": key_of(base), "base": jsonable(base), "started": now_utc(), "variants": {}}


def load_knowledge(base):
    k = None
    if os.path.exists(KNOW):
        try:
            k = json.load(open(KNOW, encoding="utf-8"))
        except Exception as e:
            say("   the knowledge file could not be read (%s) -- starting a fresh one (the old one is kept as .bad)" % e)
            os.replace(KNOW, KNOW + ".bad")
    if not k:
        return {"version": 1, "season": new_season(base), "past_seasons": [], "suggestions": [], "runs": 0}, "new"
    if k["season"]["base_key"] != key_of(base):
        k["past_seasons"].append({"base": k["season"]["base"], "started": k["season"]["started"], "ended": now_utc(),
                                  "variants_tried": len(k["season"]["variants"])})
        k["season"] = new_season(base)
        return k, "the settings in force changed -- a new season: every version is judged again against them"
    return k, "season since %s" % k["season"]["started"]


def add_variant(K, base, lv, value, why):
    s = set_lever(base, lv["name"], value)
    if s is None:
        return None
    vid = "%s=%s" % (lv["name"], fmt(value))
    V = K["season"]["variants"]
    if vid in V:
        return None
    b0 = get_lever(base, lv["name"])
    V[vid] = {"lever": lv["name"], "kind": lv["kind"], "value": value, "from": b0, "settings": jsonable(s),
              "steps": int(round(abs(float(value) - float(b0)) / float(lv["step"]))), "added": now_utc(), "why": why,
              "status": "testing", "runs": 0, "last": None, "history": [], "markets": lv.get("markets")}
    return vid


def seed(K, base, levers):
    n = 0
    for lv in levers:
        b0 = get_lever(base, lv["name"])
        if b0 is None:
            continue
        for sgn in (-1, 1):
            v = b0 + sgn * lv["step"]
            v = int(round(v)) if lv.get("int") else round(float(v), 4)
            if lv["min"] <= v <= lv["max"] and add_variant(K, base, lv, v, "first try, one step from the settings in force"):
                n += 1
    return n


# ---- scoring -------------------------------------------------------------------------------------------------------
def week_of(t):
    t = pd.Timestamp(t)
    return (t.normalize() - pd.Timedelta(days=t.weekday()))


def weekly(rows, w):
    lo, hi = w
    idx = pd.date_range(week_of(lo), week_of(hi - pd.Timedelta(seconds=1)), freq="7D")
    s = pd.Series(0.0, index=idx)
    for x in rows:
        if lo <= x["t_in"] < hi:
            s[week_of(x["t_in"])] += float(x["R"])
    return s


def compare(base_rows, var_rows, w):
    """Total R of the version minus the settings in force on one sample, how sure (weekly differences, so trades that
    overlap are not counted as separate evidence), the trades and the worst drops."""
    sb, sv = weekly(base_rows, w), weekly(var_rows, w)
    d = (sv - sb).values
    n = len(d)
    mu = float(d.sum())
    sd = float(np.std(d, ddof=1) * np.sqrt(n)) if n > 1 else 0.0
    stb = BT.stats(BT.in_window(base_rows, w))
    stv = BT.stats(BT.in_window(var_rows, w))
    return {"dR": round(mu, 3), "sd": round(sd, 3), "weeks": n, "n_base": stb["n"], "n_var": stv["n"],
            "R_base": round(float(stb.get("R", 0.0)), 2), "R_var": round(float(stv.get("R", 0.0)), 2),
            "dd_base": round(float(stb.get("ddR", 0.0)), 2), "dd_var": round(float(stv.get("ddR", 0.0)), 2)}


def z_bar(n_tried):
    return NormalDist().inv_cdf(1.0 - ALPHA / max(1, n_tried))


def passes(ev, z):
    """-> (passes, the shortfall text). Both samples: lower edge above zero; enough trades; the worst drop guard."""
    why = []
    for wk in ("w1", "w2"):
        e = ev.get(wk)
        if not e:
            why.append("%s not tested" % wk)
            continue
        lo = e["dR"] - z * e["sd"]
        if lo <= 0:
            why.append("%s: %+.1fR, needs more than %.1fR to be sure" % (wk, e["dR"], z * e["sd"]))
        if min(e["n_base"], e["n_var"]) < MIN_TRADES:
            why.append("%s: only %d trades" % (wk, min(e["n_base"], e["n_var"])))
        if e["dd_base"] > 0 and e["dd_var"] > DD_MAX * e["dd_base"]:
            why.append("%s: worst drop %.1fR vs %.1fR" % (wk, e["dd_var"], e["dd_base"]))
    return (not why), "; ".join(why)


# ---- the forward check: exit versions on the recorded trades ---------------------------------------------------------
def load_diary():
    """Every closed trade in logs/episodes (live and practice), with its proof details joined from the snapshot files
    (the diary keeps them once, by reference). -> (rows, details by reference)"""
    rows = []
    for p in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        with open(p, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    need = {r.get("composite_state_ref") for r in rows if r.get("composite_state_ref") and not r.get("composite_state")}
    snap, rx = {}, re.compile(r'^\{"ref": "([0-9a-f]{16})"')
    if need:
        for p in sorted(glob.glob(os.path.join("logs", "episodes", "snapshots_*.jsonl"))):
            with open(p, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    m = rx.match(ln)
                    if m and m.group(1) in need and m.group(1) not in snap:
                        try:
                            snap[m.group(1)] = json.loads(ln).get("composite_state") or {}
                        except Exception:
                            pass
    return rows, snap


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except Exception:
        return None


def replay_set(rows, snap, data, base_ladder):
    """Each recorded trade as a replayable entry (its own entry, its own first stop, its own target) under the new B14
    exits -- so a ladder version can be compared with the settings in force on real entries and real prices.
    -> ([(group, market, entry)], skipped counts)"""
    out, skipped = [], collections.Counter()
    for r in rows:
        a = str(r.get("asset") or "").upper()
        if a not in data:
            skipped["market not in this run"] += 1
            continue
        side = str(r.get("side") or "").lower()
        d = 1 if side in ("long", "buy") else -1 if side in ("short", "sell") else 0
        q, stop = _f(r.get("entry_price")), _f(r.get("initial_stop_loss") or r.get("stop_loss"))
        try:
            t_in = pd.Timestamp(r.get("entry_time"))
            t_in = t_in.tz_convert("UTC").tz_localize(None) if t_in.tzinfo is not None else t_in
        except Exception:
            t_in = None
        if not d or q is None or stop is None or t_in is None or pd.isna(t_in):
            skipped["no side / entry / stop / time"] += 1
            continue
        if d * (q - stop) <= 0:
            skipped["stop on the wrong side"] += 1
            continue
        cs = r.get("composite_state") or snap.get(r.get("composite_state_ref")) or {}
        tgt = _f(cs.get("ns_target")) if cs.get("ns_target") not in (None, "None") else None
        if tgt is None:
            tgt = _f(r.get("take_profit"))
        if tgt is None and _f(r.get("tp_dist_atr")) and _f(r.get("entry_atr")):
            tgt = q + d * _f(r["tp_dist_atr"]) * _f(r["entry_atr"])
        if tgt is None or d * (tgt - q) <= 0:
            skipped["no usable target"] += 1
            continue
        lane = "live" if r.get("source") == "live" else str(r.get("lane") or "?")
        grp = GROUPS[0] if (cs.get("ns_b14n") or r.get("ns_b14n")) else GROUPS[1] if lane == "live" or lane.startswith("A") \
            else GROUPS[2] if lane.startswith("C") else GROUPS[3]
        e = {"src": grp, "proof": {"dir": d, "key": (a, r.get("episode_id"), str(t_in)), "fields": {
            "ns_close": q, "ns_stop": stop, "ns_target": tgt, "ns_candle": str(t_in), "ns_exit": "FIXED",
            "ns_b14n": True, "ns_b14n_ladder_def": base_ladder}}}
        out.append((grp, a, e))
    return out, skipped


def forward_eval(trades, data, cache, ladder):
    """One ladder version on every recorded trade, paired with the settings in force on the same trade.
    -> {group: {dR, sd, n}}; `cache` keeps the settings in force's own replay of each trade."""
    wk = collections.defaultdict(lambda: collections.defaultdict(float))
    n = collections.Counter()
    for i, (grp, a, e) in enumerate(trades):
        D = data[a]
        if i not in cache:
            cache[i] = BT.simulate(e, D["m30"], D["cost"], D["usd"], True, h1=None)
        rb = cache[i]
        if rb is None:
            continue
        rv = BT.simulate(e, D["m30"], D["cost"], D["usd"], True, h1=None, ladder=ladder)
        if rv is None:
            continue
        wk[grp][week_of(rb["t_in"])] += float(rv["R"]) - float(rb["R"])
        n[grp] += 1
    out = {}
    for g in GROUPS:
        vals = list(wk[g].values())
        out[g] = {"dR": round(float(sum(vals)), 3), "n": int(n[g]),
                  "sd": round(float(np.std(vals, ddof=1) * np.sqrt(len(vals))), 3) if len(vals) > 1 else 0.0}
    return out


def fwd_ok(fw):
    """The forward check for an exit version: never clearly worse on the recorded real-signal trades; once the new
    rules have NEW_MIN trades, not worse on them either. -> (ok, text)"""
    if not fw:
        return True, "no recorded trades to check"
    why = []
    g = fw.get(GROUPS[1]) or {}
    if g.get("n", 0) >= REAL_MIN and g["dR"] + RETIRE_Z * g["sd"] < 0:
        why.append("worse on the recorded real-signal trades (%+.1fR, %d trades)" % (g["dR"], g["n"]))
    g = fw.get(GROUPS[0]) or {}
    if g.get("n", 0) >= NEW_MIN and g["dR"] < 0:
        why.append("worse on the trades since the install (%+.1fR, %d trades)" % (g["dR"], g["n"]))
    return (not why), "; ".join(why)


# ---- findings: recorded facts that win or lose on both samples -> candidate new rules -------------------------------
FINDING_MIN = {"w1": 20, "w2": 15}     # trades a group needs in each sample before it is judged
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _bucket(v, edges, names):
    v = _f(v)
    if v is None:
        return None
    for e, nm in zip(edges, names):
        if v < e:
            return nm
    return names[-1]


def facts_of(x, kind="box"):
    """The recorded facts of one trade -> {fact: value}. kind "box": a booked box-test trade (every proof fact);
    kind "diary": a recorded trade (only what the diary keeps for every lane: market, direction, weekday, hour)."""
    if kind == "diary":
        grp, a, e = x
        t = pd.Timestamp(e["proof"]["fields"]["ns_candle"])
        return {"market": a, "direction": "buy" if e["proof"]["dir"] == 1 else "sell", "weekday": DAYS[t.weekday()],
                "hour (UTC)": _bucket(t.hour, (7, 13, 17, 24), ("00-06", "07-12", "13-16", "17-23"))}
    f = x["f"]
    t = pd.Timestamp(x["t_in"])
    bk = str(f.get("ns_break_kind") or "")
    out = {
        "market": x["asset"],
        "entry path": {"hourly": "taken at once", "package": "via the package", "re-armed": "re-armed"}.get(x["src"], x["src"]),
        "setup kind": f.get("ns_kind_raw"),
        "direction": "buy" if int(f.get("setup_dir", 1) or 1) == 1 else "sell",
        "witness in the 24 h before": "yes" if f.get("ns_b14n_witness") else "no",
        "target from": f.get("ns_b14n_target_kind"),
        "3-touch diagonal agrees": "yes" if f.get("ns_diag_agrees") else "no",
        "break": "big" if "big" in bk else "small, confirmed" if "small" in bk else "plain",
        "weekday": DAYS[t.weekday()],
        "hour (UTC)": _bucket(t.hour, (7, 13, 17, 24), ("00-06", "07-12", "13-16", "17-23")),
        "clearance (1H moves)": _bucket(f.get("ns_b14n_clearance"), (0.3, 0.6, 1.0, 1e9), ("<0.3", "0.3-0.6", "0.6-1.0", "1.0+")),
        "lines ahead at entry": _bucket(len(f.get("ns_b14n_ahead") or []), (1, 3, 1e9), ("none", "1-2", "3+")),
        "entry candle strength": _bucket(f.get("ns_strength"), (0.5, 0.7, 0.85, 1e9), ("<0.5", "0.5-0.7", "0.7-0.85", "0.85+")),
        "retest depth": _bucket(f.get("brc_retest_depth"), (0.25, 0.5, 0.75, 1e9), ("<0.25", "0.25-0.5", "0.5-0.75", "0.75+")),
        "setup age (hours)": _bucket(f.get("setup_age"), (24, 72, 168, 1e9), ("<24", "24-72", "72-168", "168+")),
    }
    return {k: v for k, v in out.items() if v not in (None, "None", "")}


def findings(K, base_rows, W, run_date, trades=None, cache=None):
    """Every fact value with enough trades in both samples: its total R and its R per trade against the rest.
    A CANDIDATE RULE ("skip trades where ...") only when that group loses on BOTH samples beyond chance -- the bar
    raised for every group judged. Patterns (the same lean on both samples, not proven) are kept and counted run after
    run, so a lasting one stands out. -> (candidate rules, patterns)"""
    groups = {}
    for wk, w in W.items():
        for x in BT.in_window(base_rows, w):
            for fa, va in facts_of(x).items():
                groups.setdefault((fa, va), {}).setdefault(wk, []).append(x)
    tested = [(k, g) for k, g in groups.items() if all(len(g.get(wk, [])) >= FINDING_MIN[wk] for wk in W)]
    z = z_bar(len(tested))
    allw = {wk: BT.in_window(base_rows, w) for wk, w in W.items()}
    rec = collections.defaultdict(lambda: [0.0, 0])               # the recorded real-signal trades, per fact value
    if trades and cache is not None:
        for i, tr in enumerate(trades):
            if tr[0] != GROUPS[1] or cache.get(i) is None:
                continue
            for fa, va in facts_of(tr, "diary").items():
                rec[(fa, va)][0] += float(cache[i]["R"])
                rec[(fa, va)][1] += 1
    rules, pats = [], []
    for (fa, va), g in tested:
        ev = {}
        for wk, w in W.items():
            rows = g[wk]
            ids = {id(x) for x in rows}
            rest = [x for x in allw[wk] if id(x) not in ids]
            s = weekly(rows, w)
            tot = float(s.sum())
            ev[wk] = {"n": len(rows), "R": round(tot, 2), "per": round(tot / len(rows), 3),
                      "rest": round(sum(float(x["R"]) for x in rest) / max(1, len(rest)), 3),
                      "sd": round(float(np.std(s.values, ddof=1) * np.sqrt(len(s))) if len(s) > 1 else 0.0, 2)}
        item = {"fact": fa, "value": va, "ev": ev, "rec": rec.get((fa, va))}
        if all(ev[wk]["R"] + z * ev[wk]["sd"] < 0 for wk in W):
            rules.append(item)
        lean = [ev[wk]["per"] - ev[wk]["rest"] for wk in W]
        if all(d > 0 for d in lean) or all(d < 0 for d in lean):
            item["lean"] = min(abs(d) for d in lean) * (1 if lean[0] > 0 else -1)
            pats.append(item)
    pats.sort(key=lambda it: -abs(it["lean"]))
    F = K.setdefault("findings", {})
    for it in pats[:8]:
        k = "%s = %s" % (it["fact"], it["value"])
        F.setdefault(k, {"first": run_date, "runs": 0})
        F[k]["runs"] += 1
        F[k]["last"] = {"date": run_date, **it["ev"]}
        it["runs"] = F[k]["runs"]
    return rules, pats[:8], z, len(tested)


def show_findings(rules, pats, z, n):
    say("\nFINDINGS -- the recorded facts of every trade under the settings in force, %d groups judged (the bar: %.2f of "
        "each group's own spread, on both samples)" % (n, z))
    for it in rules:
        e1, e2 = it["ev"]["w1"], it["ev"]["w2"]
        say("   CANDIDATE RULE (for Desire): skip trades where %s = %s -- they lost on both samples beyond chance: %+.1fR in "
            "%d trades / %+.1fR in %d trades%s" % (it["fact"], it["value"], e1["R"], e1["n"], e2["R"], e2["n"],
                                                   (" | the recorded real-signal trades: %+.1fR in %d" % tuple(it["rec"]))
                                                   if it.get("rec") else ""))
    if not rules:
        say("   candidate rules: none -- no group loses on both samples beyond chance")
    if pats:
        say("   patterns to watch (the same lean on both samples, NOT proven -- R per trade in the group vs the rest):")
    for it in pats:
        e1, e2 = it["ev"]["w1"], it["ev"]["w2"]
        say("      %-44s %+.2f vs %+.2f (%d) | %+.2f vs %+.2f (%d)%s | seen in %d run%s" % (
            "%s = %s" % (it["fact"], it["value"]), e1["per"], e1["rest"], e1["n"], e2["per"], e2["rest"], e2["n"],
            (" | recorded: %+.1fR in %d" % tuple(it["rec"])) if it.get("rec") else "", it.get("runs", 1),
            "" if it.get("runs", 1) == 1 else "s"))


# ---- the follow-up: is an applied change delivering on the new trades? ---------------------------------------------
def follow_up(trades, data, base):
    """Every applied suggestion still in force, judged on the trades made since it was applied (the diary). A ladder
    change: those trades replayed with the ladder before and the ladder now, trade by trade (paired) -- UNDO is
    suggested when, after UNDO_MIN trades, the change is worse beyond chance. Any other change: the trades since cannot
    be paired (their entries would have differed), so their R per trade is shown beside what history promised."""
    recs = []
    if os.path.exists(APPLIED):
        for ln in open(APPLIED, encoding="utf-8"):
            try:
                a = json.loads(ln)
            except Exception:
                continue
            if str(a.get("file", "")).replace("\\", "/").endswith("config/config.json"):
                recs.append(a)
    last = {}
    for a in recs:
        last[a["id"]] = a
    out = []
    for sid, a in last.items():
        if a.get("undo"):
            continue
        cur = get_lever(base, a["lever"])
        if cur is None or abs(float(cur) - float(a["to"])) > 1e-9:
            out.append("   %s (%s %s -> %s): no longer in force -- the setting is %s now" % (
                sid, a["lever"], fmt(a["from"]), fmt(a["to"]), fmt(cur)))
            continue
        t0 = pd.Timestamp(a["date"])
        idx = [i for i, tr in enumerate(trades) if tr[0] == GROUPS[0] and pd.Timestamp(tr[2]["proof"]["fields"]["ns_candle"]) >= t0]
        sub = [trades[i] for i in idx]
        if a["lever"].startswith("LADDER."):
            before = (a.get("before") or {}).get("LADDER") or [list(x) for x in R14.DEFAULTS["LADDER"]]
            fw = forward_eval(sub, data, {}, before)[GROUPS[0]]
            gain, n = -fw["dR"], fw["n"]
            verdict = ("UNDO SUGGESTED -- worse than before beyond chance: python tools\\b14_rl_apply.py undo %s" % sid
                       if n >= UNDO_MIN and gain + RETIRE_Z * fw["sd"] < 0 else
                       "on track" if gain >= 0 else "behind so far, not beyond chance")
            out.append("   %s (%s %s -> %s, applied %s): %d trades since, %+.1fR against the ladder before -- %s" % (
                sid, a["lever"], fmt(a["from"]), fmt(a["to"]), a["date"], n, gain, verdict if n else "no trades yet"))
        else:
            out.append("   %s (%s %s -> %s, applied %s): %d trades since -- an entry change cannot be paired on live trades; "
                       "see the weekly report's R per trade" % (sid, a["lever"], fmt(a["from"]), fmt(a["to"]), a["date"], len(sub)))
    return out


# ---- the run -------------------------------------------------------------------------------------------------------
def evaluate(K, vid, ev, run_date, data_end):
    v = K["season"]["variants"][vid]
    v["runs"] += 1
    v["last"] = {"date": run_date, "data_end": data_end, **ev}
    v["history"].append({"date": run_date, "data_end": data_end, "w1": ev["w1"]["dR"], "w2": ev.get("w2", {}).get("dR")})
    v["history"] = v["history"][-26:]
    with open(EXPS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": run_date, "data_end": data_end, "season": K["season"]["base_key"], "variant": vid,
                            "kind": v["kind"], "ev": ev}) + "\n")


def priority(v, rng):
    if not v["last"]:
        return 1e9 - v["steps"]                      # untried first, nearest first
    w1, w2 = v["last"]["w1"], v["last"].get("w2") or v["last"]["w1"]
    return min(rng.gauss(w1["dR"], max(w1["sd"], 1e-6)), rng.gauss(w2["dR"], max(w2["sd"], 1e-6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=27.0)
    ap.add_argument("--markets", default="all")
    ap.add_argument("--full", type=int, default=4, help="entry versions to re-run this time (each = one whole replay)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    _prio = lower_priority() if not args.show else "-"
    cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    pc0 = dict(cfg.get("phase_config", {}) or {})
    base, _chg, bad = R14.apply_settings(pc0.get("b14_tuning"))
    stamp = time.strftime("%Y-%m-%d_%H%M")
    say("=" * 110)
    say("THE B14 LEARNER (RL-1, suggestions only) -- %s | process priority: %s" % (now_utc(), _prio))
    say("settings in force: %s%s" % (R14.settings_text(base), (" | REFUSED in config: %s" % "; ".join(
        "%s (%s)" % x for x in bad)) if bad else ""))
    K, how = load_knowledge(base)
    say("knowledge: %s | %d versions known this season, %d suggestions ever" % (how, len(K["season"]["variants"]),
                                                                              len(K["suggestions"])))
    say("=" * 110)
    if args.show:
        show(K)
        return 0
    levers = load_levers()
    n_seed = seed(K, base, levers)
    if n_seed:
        say("new versions to try (one step from the settings in force): %d" % n_seed)
    BT.QUIET["on"] = True
    P = BT.prepare(args.months, args.markets)
    BT.QUIET["on"] = False
    if P is None:
        say("no market could be loaded -- nothing learned this time")
        return 1
    data, windows, t_end = P["data"], P["windows"], P["t_end"]
    if len(windows) < 2:
        say("only %d months of prices -- the second sample is needed before anything can be suggested" % args.months)
    W = {"w1": windows[0][1]}
    if len(windows) > 1:
        W["w2"] = windows[1][1]
    data_end, run_date = str(t_end)[:16], now_utc()
    pc_on = dict(pc0, b14_new_rules_enabled=True)
    base_run = BT.run_row(data, pc_on, base, row="base")
    base_rows = BT.book_row(data, base_run["entries"], True)
    for wk, w in W.items():
        st = BT.stats(BT.in_window(base_rows, w))
        say("the settings in force, %s: %d trades, %+.1fR, worst drop %.1fR" % (
            "the last 16 months" if wk == "w1" else "the 11 months before", st["n"], st.get("R", 0), st.get("ddR", 0)))
    V = K["season"]["variants"]
    # the recorded trades (the diary): each one replayed under the new B14 exits, for the forward check
    drows, snap = load_diary()
    base_ladder = [list(x) for x in base["LADDER"]]
    trades, skipped = replay_set(drows, snap, data, base_ladder)
    cache = {}
    cnt = collections.Counter(g for g, _a, _e in trades)
    say("the diary: %d closed trades, %d replayable (%s)%s" % (
        len(drows), len(trades), ", ".join("%s %d" % (g, cnt[g]) for g in GROUPS if cnt[g]) or "none",
        (" | not replayable: " + ", ".join("%s %d" % kv for kv in skipped.most_common())) if skipped else ""))
    fu = follow_up(trades, data, base)
    if fu:
        say("\nFOLLOW-UP of the changes applied (on the trades since):")
        for ln in fu:
            say(ln)
            if "UNDO SUGGESTED" in ln:
                TG.append("UNDO SUGGESTED:" + ln.split(":", 1)[-1][:300])
    # exit versions: all of them, on the same trades (only the exits re-run)
    n_exit = 0
    for vid, v in V.items():
        if v["kind"] != "exit" or v["status"] == "retired":
            continue
        lad = [list(x) for x in v["settings"]["LADDER"]]
        rows = BT.book_row(data, base_run["entries"], True, ladder=lad)
        ev = {wk: compare(base_rows, rows, w) for wk, w in W.items()}
        if trades:
            ev["fwd"] = forward_eval(trades, data, cache, lad)
        evaluate(K, vid, ev, run_date, data_end)
        n_exit += 1
    # entry versions: a few, picked by the learner
    rng = random.Random(int(pd.Timestamp(t_end).strftime("%Y%m%d")))
    todo = sorted([vid for vid, v in V.items() if v["kind"] == "entry" and v["status"] != "retired"],
                  key=lambda vid: -priority(V[vid], rng))[:max(0, args.full)]
    for vid in todo:
        v = V[vid]
        sub = {a: D for a, D in data.items() if not v.get("markets") or a in v["markets"]}
        r = BT.run_row(sub, pc_on, v["settings"], row=vid)
        rows = BT.book_row(sub, r["entries"], True)
        if v.get("markets"):
            rows += [x for x in base_rows if x["asset"] not in sub]
        evaluate(K, vid, {wk: compare(base_rows, rows, w) for wk, w in W.items()}, run_date, data_end)
        say("   entry version %-26s tested (%.0f s)" % (vid, time.time() - t0))
    # learn: retire the clearly worse, grow around the winners
    n_tried = sum(1 for v in V.values() if v["runs"] > 0)
    z = z_bar(n_tried)
    retired, grown = [], []
    lv_by = {lv["name"]: lv for lv in levers}
    for vid, v in list(V.items()):
        L = v["last"]
        if not L or L.get("date") != run_date:
            continue
        if any(L.get(wk) and L[wk]["dR"] + RETIRE_Z * L[wk]["sd"] < 0 for wk in W):
            v["status"] = "retired"
            retired.append(vid)
            continue
        if all(L.get(wk) and L[wk]["dR"] > 0 for wk in W) and v["steps"] < MAX_STEPS and v["lever"] in lv_by:
            lv = lv_by[v["lever"]]
            nxt = v["value"] + (lv["step"] if v["value"] > v["from"] else -lv["step"])
            nxt = int(round(nxt)) if lv.get("int") else round(float(nxt), 4)
            if lv["min"] <= nxt <= lv["max"]:
                g = add_variant(K, base, lv, nxt, "one step further than %s, which won on both samples" % vid)
                if g:
                    grown.append(g)
    # suggest
    cands = []
    for vid, v in V.items():
        L = v["last"]
        if not L or L.get("date") != run_date or v["status"] == "retired" or len(W) < 2:
            continue
        ok, short = passes(L, z)
        if v["kind"] == "exit":
            ok2, short2 = fwd_ok(L.get("fwd"))
            if not ok2:
                ok, short = False, "; ".join(x for x in (short, short2) if x)
        edge = min(L[wk]["dR"] - z * L[wk]["sd"] for wk in W)
        cands.append((edge, vid, ok, short))
    cands.sort(reverse=True)
    K["runs"] += 1
    say("\ntested this run: %d exit versions, %d entry versions | versions tried this season: %d -> to be sure, a version must "
        "beat the settings in force by %.2f of its own spread on both samples (the bar rises with every try)" % (
            n_exit, len(todo), n_tried, z))
    say("\nTHE BEST SO FAR (total R against the settings in force: the last 16 months / the 11 months before)")
    big = sorted(cands, key=lambda c: -min(V[c[1]]["last"][wk]["dR"] for wk in W))[:3]
    near = [c for c in cands if c not in big][:3]
    for title, grp in (("the biggest gains on both samples:", big), ("the closest to the bar:", near)):
        if grp:
            say("  " + title)
        for edge, vid, ok, short in grp:
            L = V[vid]["last"]
            say("   %-30s %+7.1fR / %+7.1fR   %s" % (label(V[vid]), L["w1"]["dR"], L.get("w2", {}).get("dR", float("nan")),
                                                      "PASSES" if ok else "not yet: " + short))
            if L.get("fwd"):
                say("   %-30s recorded trades: %s" % ("", " | ".join("%s %+.1fR (%d)" % (
                    g.split(" (")[0], L["fwd"][g]["dR"], L["fwd"][g]["n"]) for g in GROUPS if L["fwd"][g]["n"])))
    if retired:
        say("\nretired (clearly worse on a sample): %s" % ", ".join(retired))
    if grown:
        say("new versions for next time (the search grows toward what wins): %s" % ", ".join(grown))
    winners = [c for c in cands if c[2]]
    if winners:
        edge, vid, ok, short = winners[0]
        v = V[vid]
        recent = [s for s in K["suggestions"] if s["variant"] == vid and s["season"] == K["season"]["base_key"]]
        if recent:
            say("\nSUGGESTION STILL STANDING: %s (first made %s) -- the evidence still holds" % (label(v), recent[0]["date"]))
        else:
            sid = "S%s-%s" % (time.strftime("%Y%m%d"), vid.replace("=", "-"))
            change = {v["lever"].split(".")[0]: v["settings"][v["lever"].split(".")[0]]}
            sug = {"id": sid, "date": run_date, "season": K["season"]["base_key"], "variant": vid, "lever": v["lever"],
                   "from": v["from"], "to": v["value"], "change": change, "evidence": v["last"], "z": round(z, 3),
                   "tried_this_season": n_tried, "config": {"b14_tuning": dict(K["season"]["base"], **change)},
                   "status": "new -- for Desire to decide"}
            K["suggestions"].append({k: sug[k] for k in ("id", "date", "season", "variant", "lever", "from", "to")})
            with open(SUGG, "a", encoding="utf-8") as f:
                f.write(json.dumps(sug) + "\n")
            v["status"] = "suggested"
            say("\nSUGGESTION %s: %s" % (sid, label(v)))
            say("   the last 16 months: %+.1fR (%d trades); the 11 months before: %+.1fR (%d trades); worst drops %.1fR / %.1fR "
                "(today's %.1fR / %.1fR)" % (v["last"]["w1"]["dR"], v["last"]["w1"]["n_var"], v["last"]["w2"]["dR"],
                                             v["last"]["w2"]["n_var"], v["last"]["w1"]["dd_var"], v["last"]["w2"]["dd_var"],
                                             v["last"]["w1"]["dd_base"], v["last"]["w2"]["dd_base"]))
            say("   the config change, if Desire says yes: phase_config.b14_tuning = %s" % json.dumps(change))
            TG.append("SUGGESTION %s: %s (%+.1fR in the last 16 months, %+.1fR in the 11 months before; passes every check). "
                      "To see it: python tools\\b14_rl_apply.py show %s -- nothing changes unless you say yes" % (
                          sid, label(v), v["last"]["w1"]["dR"], v["last"]["w2"]["dR"], sid))
    else:
        say("\nSUGGESTION: none this time -- nothing beats the settings in force on both samples by more than chance. "
            "That is the learner working: it only speaks when the evidence is strong.")
        _b = sorted([c for c in cands if all(V[c[1]]["last"][wk]["dR"] > 0 for wk in W)],
                    key=lambda c: -min(V[c[1]]["last"][wk]["dR"] for wk in W))
        TG.append("no suggestion this week" + ((" -- best so far, ahead on both samples but not sure yet: %s (%+.1fR / %+.1fR)" % (
            label(V[_b[0][1]]), V[_b[0][1]]["last"]["w1"]["dR"], V[_b[0][1]]["last"]["w2"]["dR"])) if _b else ""))
    if len(W) > 1:
        try:
            _fr = findings(K, base_rows, W, run_date, trades, cache)
            show_findings(*_fr)
            TG.append(("%d candidate rule(s) from the findings: %s" % (len(_fr[0]), "; ".join(
                "skip %s = %s" % (it["fact"], it["value"]) for it in _fr[0]))) if _fr[0] else
                      "findings: no candidate rule; %d patterns to watch" % len(_fr[1]))
        except Exception as _fe:
            say("\nFINDINGS could not be worked out this time (%s) -- tell Claude; the rest of the run stands" % _fe)
    say("forward check: every exit version is also replayed on the diary's recorded trades -- never clearly worse on the "
        "real-signal trades (%d+ needed), and once the new rules have %d trades of their own, not worse on them either; "
        "entry versions cannot be replayed on recorded trades (their entries would differ) -- history only" % (REAL_MIN, NEW_MIN))
    K["season"]["last_run"] = run_date
    json.dump(K, open(KNOW + ".tmp", "w", encoding="utf-8"), indent=1, default=str)
    os.replace(KNOW + ".tmp", KNOW)
    say("\ndone in %.0f s | knowledge: %s" % (time.time() - t0, KNOW))
    open(os.path.join(DIR, "b14_rl_telegram.txt"), "w", encoding="utf-8").write(
        "TBOT learner (B14) %s: %s\n" % (run_date[:10], " | ".join(TG) or "ran, nothing to say"))
    open(os.path.join(DIR, "b14_rl_%s.txt" % stamp), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    return 0


def label(v):
    return "%s %s -> %s" % (v["lever"], fmt(v["from"]), fmt(v["value"]))


def show(K):
    V = K["season"]["variants"]
    say("versions this season: %d (%d tested, %d retired, %d suggested)" % (
        len(V), sum(1 for v in V.values() if v["runs"]), sum(1 for v in V.values() if v["status"] == "retired"),
        sum(1 for v in V.values() if v["status"] == "suggested")))
    for vid, v in sorted(V.items(), key=lambda kv: -(min((kv[1]["last"] or {}).get(w, {}).get("dR", -1e9)
                                                         for w in ("w1", "w2")) if kv[1]["last"] else -1e9)):
        L = v["last"] or {}
        say("   %-30s %-9s runs %2d  last %s  %s" % (label(v), v["status"], v["runs"], L.get("date", "-"),
                                                    ("%+.1fR / %+.1fR" % (L["w1"]["dR"], L.get("w2", {}).get("dR", float("nan"))))
                                                    if L else "not tested yet"))
    for s in K["suggestions"][-10:]:
        say("   suggestion %s (%s): %s %s -> %s" % (s["id"], s["date"], s["lever"], fmt(s["from"]), fmt(s["to"])))


if __name__ == "__main__":
    sys.exit(main())
