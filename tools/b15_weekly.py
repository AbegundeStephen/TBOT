"""B15 WEEKLY REPLAY (Desire 12 Oct, rule set Part 5) -- READ-ONLY. Replaces tools\\b13_forward_test.py as the Saturday
"forward" job (tools\\b14_task.ps1 -Job forward). It is the BOX TEST's own replay (tools\\b15_box_test.py, which runs
the very files the bot runs) over the last few months, and it prints the last FW_DAYS days of it: today's rules (S0,
B13) and the B15 rules (S1) side by side, next to what MT5 actually traded in the same days. One code path, so the
box test and the weekly report can never disagree (closes B14 flag F13).

Sections:
  1. THE TWO RULE SETS' TRADES in the window, per market: entry, stop, exit, R, $, the path and the 4H brain state
  2. WAITING AT THE END: setups still on the 4H clock (what the live bot would be watching)
  3. WHAT MT5 ACTUALLY TRADED -- self-explaining: account, server, dates, how many deals it got; 0 deals says why that
     can happen; the bot's own diary (logs\\episodes) is listed beside it as a second source
  4. THE 30-TRADE RULE on B15's live trades since the switch (decision D3): below 0R at 30+ -> stop and investigate
  5. B15 TRACKING: the bot's own log lines for the week (paths, 4H progress, hand-overs, entries, cancels, re-arms,
     outcomes, staircase steps, rebuilds, must-never-happen lines)
Writes only its output (the scheduled job keeps it in logs\\b15_weekly_<date>.txt).

usage (from C:\\TradingBot\\TBOT, with the bot's python):  python tools\\b15_weekly.py
  FW_DAYS=7 (environment) = the window; --months 4 = how far back the replay starts (the brain and the lines need it)"""
import argparse
import collections
import glob
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
_sp = importlib.util.spec_from_file_location("b15_box_test", os.path.join(ROOT, "tools", "b15_box_test.py"))
B15 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(B15)
B14, NE, NP, LM, say = B15.B14, B15.NE, B15.NP, B15.LM, B15.say
FW_DAYS = int(os.environ.get("FW_DAYS", "7"))
TAGS = ("[NS-PATH]", "[PKG-4H]", "[PKG-HANDOVER]", "[PKG-ENTER]", "[PKG-CANCEL]", "[PKG-REARM]", "[PKG-OUTCOME]",
        "[NS-PROOF]", "[NS-ROUTE]", "[NS-LOCK]", "[NS-STAIR]", "[NS-STAIR-FAIL]", "[NS-WALLS]", "[RUNNER-RUN]",
        "[NS-REBUILD]", "[NS-MAP]", "[NS-STOP-HOLD]", "[B15-NEVER]", "[B15] rules")
T0 = time.time()


def trades_in_window(res, rid, a, since):
    return [x for x in res[rid]["per"].get(a, (None, []))[1] if x["t_in"] >= since]


def show_trade(x):
    f = x["f"]
    return "      %s %s %-10.6g stop %-10.6g -> %s %-10.6g %-22s %+6.2fR $%+8.2f | %s | brain %s" % (
        str(x["t_in"])[:16], "BUY " if x["d"] == 1 else "SELL", x["q"], float(f.get("ns_stop") or 0), str(x["t_out"])[:16],
        x["px"], x["how"], x["R"], x["usd"], str(f.get("ns_path") or ("%s entry" % f.get("ns_entry")))[:70],
        x["ph"] or "unknown")


def mt5_section(since, now, diary_live):
    say("\n" + "=" * 118 + "\n3. WHAT MT5 ACTUALLY TRADED, %s to %s\n" % (str(since)[:16], str(now)[:16]) + "=" * 118)
    pos, info = {}, None
    try:
        import MetaTrader5 as _mt5
        if not _mt5.initialize():
            say("   MT5 could not be started from this python (initialize() failed: %s) -- no MT5 history here. The "
                "diary below is the only record this run has." % str(_mt5.last_error()))
        else:
            ai = _mt5.account_info()
            ti = _mt5.terminal_info()
            info = (getattr(ai, "login", "?"), getattr(ai, "server", "?"), getattr(ai, "name", "?"),
                    getattr(ai, "balance", float("nan")), getattr(ti, "path", "?"))
            say("   account %s on %s (%s), balance %.2f | terminal %s" % info)
            deals = _mt5.history_deals_get(since.to_pydatetime(), (now + pd.Timedelta(days=1)).to_pydatetime()) or []
            say("   history_deals_get(%s -> %s): %d deal(s)" % (str(since)[:16], str(now + pd.Timedelta(days=1))[:16], len(deals)))
            for dl in deals:
                p = pos.setdefault(dl.position_id, {"sym": dl.symbol, "in": None, "out": None, "profit": 0.0})
                p["profit"] += float(dl.profit) + float(getattr(dl, "commission", 0.0)) + float(getattr(dl, "swap", 0.0))
                if dl.entry == 0:
                    p["in"] = dl
                elif dl.entry == 1:
                    p["out"] = dl
            _mt5.shutdown()
    except Exception as ex:
        say("   (MT5 history not available here: %s)" % ex)
    tot, n_closed = 0.0, 0
    for pid, p in sorted(pos.items(), key=lambda kv: (kv[1]["in"].time if kv[1]["in"] else 0)):
        if p["in"] is None:
            continue
        t_in = pd.Timestamp(p["in"].time, unit="s")
        if p["out"] is not None:
            tot += p["profit"]
            n_closed += 1
        say("   %-10s %-4s opened %s at %-11.6g %s  profit %+.2f" % (
            p["sym"], "BUY" if p["in"].type == 0 else "SELL", t_in.strftime("%Y-%m-%d %H:%M"), p["in"].price,
            ("closed %s at %.6g" % (pd.Timestamp(p["out"].time, unit="s").strftime("%Y-%m-%d %H:%M"), p["out"].price))
            if p["out"] is not None else "STILL OPEN", p["profit"]))
    say("   MT5 TOTAL (closed): %+.2f $ from %d closed position(s), %d open" % (tot, n_closed, sum(1 for p in pos.values() if p["in"] is not None and p["out"] is None)))
    if info is not None and not pos:
        say("   0 DEALS: nothing was opened or closed on account %s (%s) between those dates. If the bot DID trade in these "
            "days, then either this python's MT5 is logged into a different account/terminal than the bot's, or the "
            "history window is wrong -- compare with the terminal's History tab and the diary below." % (info[0], info[1]))
    say("\n   the bot's own diary (logs\\episodes), live trades closed in the window -- the second record:")
    if not diary_live:
        say("      none")
    for r in diary_live:
        say("      %-7s %-5s closed %s  %s  R %s  %s" % (
            r.get("asset"), str(r.get("side") or r.get("direction") or "")[:5], str(r.get("close_time") or r.get("exit_time"))[:16],
            ("%+.2f%%" % float(r["_pct"])) if r.get("_pct") is not None else "-", ("%+.2f" % r["_r"]) if r.get("_r") is not None else "-",
            str(((r.get("proof_snapshot") or {}).get("composite_state") or {}).get("ns_path") or "")[:60]))
    if pos and diary_live and abs(len([p for p in pos.values() if p["out"] is not None]) - len(diary_live)) > 0:
        say("   NOTE: MT5 closed %d position(s), the diary has %d closed live trade(s) in the window -- a difference means "
            "trades opened or closed outside the bot (phone, manual) or a diary row that was not written; check." % (
                len([p for p in pos.values() if p["out"] is not None]), len(diary_live)))


def diary_rows(since):
    """Closed trades from the diary (logs/episodes), UTC close time >= since."""
    out = []
    for f in sorted(glob.glob(os.path.join("logs", "episodes", "episodes_*.jsonl"))):
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ct = r.get("close_time") or r.get("exit_time") or r.get("closed_at")
                if not ct:
                    continue
                try:
                    t = pd.Timestamp(ct)
                    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                except Exception:
                    continue
                if t.tz_localize(None) < since:
                    continue
                r["_r"] = next((float(r.get(k)) for k in ("net_pnl_r", "pnl_r", "r_multiple", "gross_r")
                                if isinstance(r.get(k), (int, float))), None)
                r["_pct"] = next((float(r.get(k)) for k in ("net_pnl_pct", "pnl_pct", "realized_pnl_pct", "gross_pnl_pct")
                                  if isinstance(r.get(k), (int, float))), None)
                r["_t"] = t.tz_localize(None)
                out.append(r)
    return out


def thirty_trade_rule(res, weeks):
    """Decision D3: TEST 2's 30-trade rule, on B15's live trades (the diary rows whose proof carries ns_path -- only
    B15 writes it) since the switch; the band comes from this replay's own S1 trades."""
    say("\n" + "=" * 118 + "\n4. THE 30-TRADE RULE on B15's live trades (decision D3; TEST 2 retired)\n" + "=" * 118)
    rows = [r for r in diary_rows(pd.Timestamp("2000-01-01")) if str(r.get("source", "live")) == "live"
            and not r.get("external") and not r.get("adopted")
            and ((r.get("proof_snapshot") or {}).get("composite_state") or {}).get("ns_path")]
    done = [r for r in rows if r.get("_r") is not None]
    s1 = res.get("S1", {}).get("rows") or []
    R = np.array([x["R"] for x in s1]) if s1 else np.array([])
    if len(R) > 1:
        mu, sd = float(R.mean()), float(R.std(ddof=1))
        say("   expected from this replay's B15 trades (S1, %d trades over %.0f weeks): %+.2fR a trade, sd %.2f, %.1f trades a week"
            % (len(R), weeks, mu, sd, len(R) / max(weeks, 1e-9)))
    else:
        mu = sd = float("nan")
    say("   B15 live trades so far: %d closed with an R (%d rows in all)" % (len(done), len(rows)))
    say("   RULE (agreed in advance, printed every run): at 30+ finished B15 live trades --")
    say("     mean below 0R                   -> STOP and investigate before anything else changes")
    say("     below the band but above 0R     -> watch; re-check at 60 trades")
    say("     inside or above the band        -> on track")
    if not done:
        say("   status: no finished B15 live trades yet (the rule applies from 30)")
        return
    m = float(np.mean([r["_r"] for r in done]))
    if len(done) < 30:
        say("   status: %d finished trades, mean %+.2fR -- the rule applies from 30" % (len(done), m))
        return
    if sd == sd:
        lo, hi = mu - 1.96 * sd / np.sqrt(len(done)), mu + 1.96 * sd / np.sqrt(len(done))
        verdict = ("STOP AND INVESTIGATE (below 0R)" if m < 0 else "BELOW the band -- watch" if m < lo else
                   "inside or above the band -- on track")
        say("   status: %d finished trades, mean %+.2fR, band %+.2f to %+.2f -> %s" % (len(done), m, lo, hi, verdict))
    else:
        say("   status: %d finished trades, mean %+.2fR -> %s (no band: the replay had too few trades)" % (
            len(done), m, "STOP AND INVESTIGATE (below 0R)" if m < 0 else "above 0R"))


def tracking(days, pcfg):
    """The bot's own log lines for the week, market by market (tools\\b13_forward_test.b14_tracking, plus B15's tags)."""
    say("\n" + "=" * 118 + "\n5. B15 TRACKING -- the bot's own log lines, last %d days (box time)\n" % days + "=" * 118)
    say("   rules switch (phase_config.b14_rules_enabled): %s" % (
        "ON -- B15 trades" if (pcfg or {}).get("b14_rules_enabled") else "OFF -- B13 trades; no [NS-PATH] / [PKG-4H] lines are expected"))
    since = pd.Timestamp.now() - pd.Timedelta(days=days)
    rx = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    ev, cnt = collections.defaultdict(list), collections.Counter()
    for f in sorted(glob.glob(os.path.join("logs", "trading_bot.log*")), key=os.path.getmtime):
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "[N" not in line and "[P" not in line and "[R" not in line and "[B" not in line:
                    continue
                m = rx.match(line)
                if not m:
                    continue
                t = pd.Timestamp(m.group(1))
                if t < since:
                    continue
                msg = line.split(" - ", 3)[-1].strip()
                for tag in TAGS:
                    if msg.startswith(tag):
                        a = msg[len(tag):].strip().split(":", 1)[0].split()[0] if ":" in msg else "?"
                        ev[a].append((t, tag, msg))
                        cnt[tag] += 1
                        break
    if not ev:
        say("   no engine / package lines in the last %d days -- if the bot was running, tell Claude" % days)
        return
    say("   counts: " + ", ".join("%s %d" % kv for kv in sorted(cnt.items())))
    bad = cnt["[NS-STAIR-FAIL]"] + cnt["[NS-WALLS]"] + cnt["[NS-MAP]"] + cnt["[B15-NEVER]"] + sum(
        1 for a in ev for _t, tag, msg in ev[a] if tag == "[NS-STOP-HOLD]" and "ns_wall" in msg)
    say("   failures (stair not fed / no map / a step held by the guard / a small line on the hourly clock): %d %s" % (
        bad, "" if not bad else "<-- tell Claude"))
    for a in sorted(ev):
        say("\n   %s" % a)
        for t, tag, msg in sorted(ev[a]):
            if tag == "[NS-STOP-HOLD]" and "ns_wall" not in msg:
                continue
            if tag == "[PKG-4H]" and "day" in msg:                 # the 4H progress lines: one per 4H close -- show the last 3
                pass
            say("      %s  %s" % (str(t)[5:16], msg[:210]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=4.0)
    ap.add_argument("--markets", default="all")
    args = ap.parse_args()
    say("=" * 118)
    say("B15 WEEKLY REPLAY -- %s (read-only) | rules %s | the box test's own replay, last %d days, B13 (S0) and B15 (S1) "
        "side by side" % (time.strftime("%Y-%m-%d_%H%M"), NE.RULES_V, FW_DAYS))
    say("files: " + " | ".join("%s %s" % (os.path.basename(p), B15.fp(p)) for p in (
        "src/execution/ns_engine.py", "src/execution/ns_package.py", "src/execution/line_map.py",
        "tools/b15_box_test.py", "tools/b15_weekly.py")))
    say("=" * 118)
    try:
        cfg = json.load(open(os.path.join("config", "config.json"), encoding="utf-8-sig"))
    except Exception as e:
        say("   config/config.json not read (%s) -- cannot replay the live settings; stopping" % e)
        sys.exit(1)
    pc0 = cfg.get("phase_config", {}) or {}
    say("   live switch (phase_config.b14_rules_enabled): %s" % ("ON -- the bot trades B15" if pc0.get("b14_rules_enabled") else "OFF -- the bot trades B13"))
    markets = list(B14.MARKETS) if args.markets == "all" else [m.strip().upper() for m in args.markets.split(",")]
    usd, spreads = B14.lot_values()
    paid = B14.log_spreads()
    try:
        from src.execution.shadow_trader import FRICTION_PENALTIES as _FP, _DEFAULT_FRICTION as _DF
    except Exception:
        _FP, _DF = {}, 0.0003
    raw = {}
    for a in markets:
        df, note = B14.load_1h(B14.MARKETS[a])
        if df is None:
            say("   %-7s skipped: %s" % (a, note))
            continue
        raw[a] = (df, note)
    if not raw:
        say("   NO PRICE FILES COULD BE READ (data\\raw\\*_1h.csv) -- nothing to replay; run tools\\data_repair.py and again")
        sys.exit(1)
    now = max(df.index[-1] for df, _n in raw.values()) + pd.Timedelta(hours=1)
    since = now - pd.Timedelta(days=FW_DAYS)
    t_start = now - pd.Timedelta(days=int(args.months * 30.4))
    stale = [a for a, (df, _n) in raw.items() if now - df.index[-1] > pd.Timedelta(days=3)]
    if stale:
        say("   STALE PRICE FILES (last candle more than 3 days before the newest): %s -- their replay ends early; run "
            "tools\\data_repair.py" % ", ".join(stale))
    data = {}
    for a, (df, note) in raw.items():
        df = df[df.index >= t_start]
        fr = B14.mt5_frames(B14.MARKETS[a])
        approx = "M30" not in fr
        m30 = fr.get("M30") if not approx else B14.m30_from_1h(df)
        m30 = m30[m30.index >= df.index[0]]
        d1, w1 = fr.get("D1"), fr.get("W1")
        mp = LM.MarketMap(a, df, d1, w1)
        frames = {"M30": m30, "D1": d1 if d1 is not None else mp.d1, "W1": w1 if w1 is not None else mp.w1}
        cost = paid.get(a, spreads.get(a, _FP.get(a, _DF)))
        weeks = (df.index[-1] - df.index[0]).total_seconds() / (7 * 86400.0)
        data[a] = dict(df=df, m30=m30, mp=mp, frames=frames, cost=cost, usd=usd[a], weeks=weeks, approx=approx, hist=None)
        say("   %-7s %s | 30m: %s | spread %.5f%% | $%.4g per 1.0" % (
            a, note, ("MT5" if fr.get("_src") != "export" else "the export files") if not approx else
            "APPROXIMATE (from the hourly file)", 100 * cost, usd[a]))
    table = B15.rows_table15(pc0)
    res = collections.OrderedDict()
    waiting = {}
    for rid in ("S0", "S1"):
        name, spec = table[rid]
        res[rid] = dict(name=name, per={}, rows=[], log=collections.Counter(), cnt=collections.Counter())
        for a, D in data.items():
            entries, lg, cnt, st = B15.run_signals15(a, D["df"], D["frames"], D["mp"], spec["pc"])
            if D["hist"] is None:
                D["hist"] = {"brain4_states": list(st.get("brain4_states") or [])}
            rows = B15.book15(entries, D, spec["exits"])
            for x in rows:
                x["asset"], x["ph"] = a, B15.tag(D["hist"], x["f"])
            res[rid]["per"][a] = (B15.stats15(rows, D["weeks"]), rows)
            res[rid]["rows"] += rows
            res[rid]["log"].update(lg)
            res[rid]["cnt"].update(cnt)
            if rid == "S1":
                waiting[a] = [s for s in (st.get("setups") or []) if s.get("stage", 0) >= 1]
    weeks = float(np.mean([D["weeks"] for D in data.values()]))
    say("\n" + "=" * 118 + "\n1. THE TWO RULE SETS' TRADES, %s to %s (entries in the window; exits may be later or still "
        "open at the end of data)\n" % (str(since)[:16], str(now)[:16]) + "=" * 118)
    for a in data:
        for rid in ("S0", "S1"):
            rows = trades_in_window(res, rid, a, since)
            say("   %-7s %s (%s): %d trade(s)" % (a, rid, "B13, today's rules" if rid == "S0" else "B15", len(rows)))
            for x in rows:
                say(show_trade(x))
    for rid in ("S0", "S1"):
        rows = [x for a in data for x in trades_in_window(res, rid, a, since)]
        say("   TOTAL %s: %d trades | %+.2f R | %+.2f $ at the smallest lot" % (rid, len(rows), sum(x["R"] for x in rows), sum(x["usd"] for x in rows)))
    s0, s1 = B15.stats15(res["S0"]["rows"], weeks), B15.stats15(res["S1"]["rows"], weeks)
    say("   whole replay (%.0f weeks, %d markets): S0 %d trades %+.1fR $%+.1f | S1 %d trades %+.1fR $%+.1f" % (
        weeks, len(data), s0["n"], s0["R"], s0["usd"], s1["n"], s1["R"], s1["usd"]))
    say("\n" + "=" * 118 + "\n2. WAITING AT THE END OF THE DATA under B15 (what the live bot should be watching now)\n" + "=" * 118)
    n_w = 0
    for a, ss in waiting.items():
        for s in ss:
            n_w += 1
            say("   %-7s setup #%s %s R2 %.6g -- %s, stage %d (%s), since %s%s" % (
                a, s.get("id"), "BUY" if s["d"] == 1 else "SELL", s["r2"], "the 4H clock" if s.get("clock") == "4H" else "the hourly proof",
                int(s.get("stage", 0)), "waiting for the retest" if s.get("stage") == 1 else "waiting for the entry close",
                str(s.get("t_break"))[:16], (" | failed attempts %d" % s["fails"]) if s.get("fails") else ""))
    if not n_w:
        say("   nothing waiting")
    diary_live = [r for r in diary_rows(since) if str(r.get("source", "live")) == "live" and not r.get("external")]
    mt5_section(since, now, diary_live)
    thirty_trade_rule(res, weeks)
    tracking(FW_DAYS, pc0)
    say("\n   READ: S0 = what today's rules would have done this week, S1 = what B15 would have done, both priced on the "
        "30-minute candles at the smallest lot after the spread. Compare with what MT5 did. Live trades missing from "
        "the rule set that was ON that week mean a live-only check stopped them (health, session, MT5) -- look in the log.")
    say("\n=== END OF THE B15 WEEKLY REPLAY (%.0f s) ===" % (time.time() - T0))


if __name__ == "__main__":
    main()
