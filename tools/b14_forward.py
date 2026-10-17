"""B14 WEEKLY FORWARD REPORT (Desire 16 Oct 2026: "keep all 3") -- READ-ONLY. The three jobs of the old B15 weekly
replay (tools/b15_weekly.py, removed with the B15 rules), for the new B14:
  1. THE WEEK'S TRADES of both rule sets -- the B13 rules (S0, the B14 switch off) and the new B14 rules -- replayed by
     the box test's own code (tools/b14_box_test.py, the very files the bot runs), market by market
  2. WAITING AT THE END: the setups and package entries the rules that trade live are watching at the end of the data
  3. WHAT MT5 ACTUALLY TRADED in the same days, next to the bot's own diary -- a difference means the live bot did
     something the tested code would not (or a trade was opened or closed by hand)
  4. THE 30-TRADE RULE (decision D3, 12 Oct) on the new B14 rules' LIVE trades since the switch: below 0R at 30+
     finished trades -> STOP AND INVESTIGATE (the old B15 version looked for the trades where the diary does not keep
     them -- it would never have counted one)
The bot's own B14 log lines for the week are in tools/b14_weekly.py (same Saturday report). Changes nothing; MT5 is
only read (history), and the account is shown by its last 3 digits only.

usage (from C:\\TradingBot\\TBOT, with the bot's python):  python tools\\b14_forward.py [--months 4] [--markets all]
  FW_DAYS=7 (environment) = the window. Run by the Saturday job (tools\\rl1_weekly.ps1). Writes logs\\b14_forward_<stamp>.txt"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT)


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


BX = _load("b14_box_test", os.path.join("tools", "b14_box_test.py"))
W = _load("b14_weekly", os.path.join("tools", "b14_weekly.py"))
say = BX.say
FW_DAYS = int(os.environ.get("FW_DAYS", "7"))
T0 = time.time()
BAR = "=" * 118


def show_trade(x):
    f = x["f"]
    if f.get("ns_b14n"):
        extra = "B14: target %s, witness %s, ladder rung %d" % (
            f.get("ns_b14n_target_kind") or "-", f.get("ns_b14n_witness_name") or ("yes" if f.get("ns_b14n_witness") else "none"),
            int(x.get("ladder") or 0))
    else:
        extra = "entry %s" % (f.get("ns_entry") or "-")
    return "      %s %s %-10.6g stop %-10.6g -> %s %-10.6g %-12s %+6.2fR $%+8.2f | %-8s | %s" % (
        str(x["t_in"])[:16], "BUY " if x["d"] == 1 else "SELL", x["q"], float(f.get("ns_stop") or 0), str(x["t_out"])[:16],
        x["px"], x["how"], x["R"], x["usd"], x.get("src", ""), extra)


def waiting_section(states, live_name, end):
    say("\n" + BAR + "\n2. WAITING AT THE END OF THE DATA under %s -- what the live bot should be watching now\n" % live_name + BAR)
    n = 0
    for a, st in states.items():
        for s in (st or {}).get("setups") or []:
            if int(s.get("stage", 0) or 0) < 1:
                continue
            n += 1
            say("   %-7s setup #%s %s level %.6g -- stage %d (%s), broke %s%s%s" % (
                a, s.get("id"), "BUY" if s.get("d") == 1 else "SELL", float(s.get("r2") or 0), int(s.get("stage", 0)),
                "waiting for the retest" if s.get("stage") == 1 else "waiting for the entry close", str(s.get("t_break"))[:16],
                (" | failed attempts %d" % s["fails"]) if s.get("fails") else "",
                (" | re-armed %d" % s["rearms"]) if s.get("rearms") else ""))
        for rec in (st or {}).get("pkg") or []:
            if rec.get("status") != "watching" or pd.Timestamp(rec.get("tE")) < end - pd.Timedelta(hours=26):
                continue                      # the package watches 26 hours at most: older records were settled
            n += 1
            say("   %-7s package entry %s handed over %s -- watching the 30-minute candles" % (
                a, "BUY" if rec.get("d", rec.get("dir")) == 1 else "SELL", str(rec.get("tE"))[:16]))
    if not n:
        say("   nothing waiting")


def mt5_section(since, now, diary_live):
    say("\n" + BAR + "\n3. WHAT MT5 ACTUALLY TRADED, %s to %s\n" % (str(since)[:16], str(now)[:16]) + BAR)
    pos, acct = {}, None
    try:
        import MetaTrader5 as _mt5
        if not _mt5.initialize():
            say("   MT5 could not be started from this python (initialize() failed: %s) -- no MT5 history here. The "
                "diary below is the only record this run has." % str(_mt5.last_error()))
        else:
            ai = _mt5.account_info()
            login = str(getattr(ai, "login", "") or "")
            acct = ("...%s" % login[-3:]) if login else "?"
            say("   account %s on %s" % (acct, getattr(ai, "server", "?")))
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
    for _pid, p in sorted(pos.items(), key=lambda kv: (kv[1]["in"].time if kv[1]["in"] else 0)):
        if p["in"] is None:
            continue
        if p["out"] is not None:
            tot += p["profit"]
            n_closed += 1
        say("   %-10s %-4s opened %s at %-11.6g %s  profit %+.2f" % (
            p["sym"], "BUY" if p["in"].type == 0 else "SELL", pd.Timestamp(p["in"].time, unit="s").strftime("%Y-%m-%d %H:%M"),
            p["in"].price, ("closed %s at %.6g" % (pd.Timestamp(p["out"].time, unit="s").strftime("%Y-%m-%d %H:%M"), p["out"].price))
            if p["out"] is not None else "STILL OPEN", p["profit"]))
    if pos:
        say("   MT5 TOTAL (closed): %+.2f $ from %d closed position(s), %d open" % (
            tot, n_closed, sum(1 for p in pos.values() if p["in"] is not None and p["out"] is None)))
    if acct is not None and not pos:
        say("   0 DEALS on account %s between those dates. If the bot DID trade in these days, this python's MT5 is logged "
            "into another account or terminal than the bot's -- compare with the terminal's History tab and the diary." % acct)
    say("\n   the bot's own diary (logs\\episodes), live trades closed in the window -- the second record:")
    if not diary_live:
        say("      none")
    for r in diary_live:
        rr = W.r_of(r)
        say("      %-7s %-5s closed %s  R %s  %s" % (
            r.get("asset"), str(r.get("side") or r.get("direction") or "")[:5], str(r.get("close_time") or r.get("exit_time"))[:16],
            ("%+.2f" % rr) if rr is not None else "-", "new B14 rules" if W.cs(r).get("ns_b14n") else "B13 rules"))
    closed = len([p for p in pos.values() if p["out"] is not None])
    if pos and closed != len(diary_live):
        say("   NOTE: MT5 closed %d position(s), the diary has %d closed live trade(s) in the window -- a difference means "
            "trades opened or closed outside the bot (phone, by hand) or a diary row that was not written; check." % (
                closed, len(diary_live)))


def thirty_trade_rule(b14_rows, weeks):
    say("\n" + BAR + "\n4. THE 30-TRADE RULE on the new B14 rules' LIVE trades (decision D3, 12 Oct)\n" + BAR)
    rows = [r for r in W.rows_since(pd.Timestamp("2000-01-01", tz="UTC").to_pydatetime())
            if r.get("source") == "live" and not r.get("external") and not r.get("adopted")]
    W.load_snapshots([r.get("composite_state_ref") for r in rows])
    rows = [r for r in rows if W.cs(r).get("ns_b14n")]
    done = [r for r in rows if W.r_of(r) is not None]
    R = np.array([x["R"] for x in b14_rows]) if b14_rows else np.array([])
    if len(R) > 1:
        mu, sd = float(R.mean()), float(R.std(ddof=1))
        say("   expected from this replay's B14 trades (%d trades over %.0f weeks): %+.2fR a trade, sd %.2f, %.1f trades a week"
            % (len(R), weeks, mu, sd, len(R) / max(weeks, 1e-9)))
    else:
        mu = sd = float("nan")
    say("   new-B14 live trades so far: %d finished with an R (%d diary rows)" % (len(done), len(rows)))
    say("   RULE (agreed in advance, printed every run): at 30+ finished new-B14 live trades --")
    say("     mean below 0R                   -> STOP and investigate before anything else changes")
    say("     below the band but above 0R     -> watch; re-check at 60 trades")
    say("     inside or above the band        -> on track")
    if not done:
        say("   status: no finished new-B14 live trades yet (the rule applies from 30; the rules trade live only once "
            "Desire switches them on)")
        return
    m = float(np.mean([W.r_of(r) for r in done]))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=4.0)
    ap.add_argument("--markets", default="all")
    args = ap.parse_args()
    del BX.OUT[:]
    BX.QUIET["on"] = False              # printed too: the Saturday job keeps it in its own report
    say(BAR)
    say("B14 WEEKLY FORWARD REPORT -- %s (read-only) | rules %s | the box test's own replay, last %d days, the B13 rules "
        "(S0) and the new B14 rules side by side" % (time.strftime("%Y-%m-%d_%H%M"), BX.NE.RULES_V, FW_DAYS))
    say("files: " + " | ".join("%s %s" % (os.path.basename(p), BX.fp(p)) for p in (
        "src/execution/ns_engine.py", "src/execution/ns_package.py", "src/execution/b14_rules.py", "src/execution/line_map.py",
        "tools/b14_box_test.py", "tools/b14_forward.py")))
    say(BAR)
    P = BX.prepare(args.months, args.markets)
    if P is None:
        say("   NO PRICE FILES COULD BE READ (data\\raw\\*_1h.csv) -- nothing to replay; run tools\\data_repair.py and again")
        sys.exit(1)
    data, pc0 = P["data"], P["pc0"]
    live_on = bool(pc0.get("b14_new_rules_enabled"))
    say("   live switch (phase_config.b14_new_rules_enabled): %s" % (
        "ON -- the bot trades the new B14 rules" if live_on else "OFF -- the bot trades the B13 rules"))
    now = max(D["df"].index[-1] for D in data.values()) + pd.Timedelta(hours=1)
    since = now - pd.Timedelta(days=FW_DAYS)
    stale = [a for a, D in data.items() if now - D["df"].index[-1] > pd.Timedelta(days=3)]
    if stale:
        say("   STALE PRICE FILES (last candle more than 3 days before the newest): %s -- their replay ends early; run "
            "tools\\data_repair.py" % ", ".join(stale))
    weeks = float(np.mean([(D["df"].index[-1] - D["df"].index[0]).total_seconds() / (7 * 86400.0) for D in data.values()]))
    table = BX.report_rows(pc0)
    live_rid = "B14" if live_on else "S0"
    res, waiting = {}, {}
    for rid in (live_rid, "B14" if live_rid == "S0" else "S0"):           # the live rules first: their memory is kept
        name, pc, settings, exits = table[rid]
        out = BX.run_row(data, pc, settings, row=rid)
        if rid == live_rid:
            waiting = {a: BX.LAST_STATE.get(a) for a in data}
        res[rid] = (name, BX.book_row(data, out["entries"], exits))
    say("\n" + BAR + "\n1. THE TWO RULE SETS' TRADES, %s to %s (entries in the window; exits may be later or still open at "
        "the end of data)\n" % (str(since)[:16], str(now)[:16]) + BAR)
    for a in data:
        for rid in ("S0", "B14"):
            rows = [x for x in BX.in_window(res[rid][1], (since, now + pd.Timedelta(days=1))) if x["asset"] == a]
            say("   %-7s %-3s (%s): %d trade(s)" % (a, rid, "the B13 rules" if rid == "S0" else "the new B14 rules", len(rows)))
            for x in rows:
                say(show_trade(x))
    for rid in ("S0", "B14"):
        rows = BX.in_window(res[rid][1], (since, now + pd.Timedelta(days=1)))
        say("   TOTAL %-3s: %d trades | %+.2f R | %+.2f $ at the smallest lot" % (
            rid, len(rows), sum(x["R"] for x in rows), sum(x["usd"] for x in rows)))
    for rid in ("S0", "B14"):
        say(BX.row_line("whole replay %s (%.0f weeks, %d markets)" % (rid, weeks, len(data)), BX.stats(res[rid][1])))
    waiting_section(waiting, "the new B14 rules (trading live now)" if live_on else "the B13 rules (trading live now)", now)
    diary_live = [r for r in W.rows_since(since.tz_localize("UTC").to_pydatetime()) if r.get("source") == "live"
                  and not r.get("external")]
    W.load_snapshots([r.get("composite_state_ref") for r in diary_live])
    mt5_section(since, now, diary_live)
    thirty_trade_rule(res["B14"][1], weeks)
    say("\n   READ: S0 = what the B13 rules would have done this week, B14 = what the new B14 rules would have done, both "
        "priced on the 30-minute candles at the smallest lot after the spread. Compare with what MT5 did: live trades "
        "missing from the rules that were ON that week mean a live-only check stopped them (health, session, MT5).")
    say("   the week's B14 log lines (start-up, ladder moves, clearance, re-arms): the B14 weekly section of this report")
    say("\n=== END OF THE B14 WEEKLY FORWARD REPORT (%.0f s) ===" % (time.time() - T0))
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "b14_forward_%s.txt" % BX.STAMP), "w", encoding="utf-8") as f:
        f.write("\n".join(BX.OUT) + "\n")


if __name__ == "__main__":
    main()
