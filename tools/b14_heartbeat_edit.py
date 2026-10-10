"""B14 (Desire 8-9 Oct) -- edits the watchdog's check list (config/heartbeat.json) IN PLACE, safely: a backup first,
only the named changes, prints what it changed, and running it twice changes nothing more. It includes everything B13
step 11 was meant to do (tools/b13_heartbeat_edit.py was never run on the box), plus B14's own checks.
  B13 item 3 : retire the 4 dead checks + the 'bars passed=' check
  B13 item 6 : the R1-ORIGIN check only counts while markets are open
  B13 item 16: nightly breaks for SILVER and JP225; the ny_open false alarm (B14: it comes from gate.log_vs_ledger,
               which now skips the ny_open gate -- heartbeat.py reads "skip_gates")
  B13 item 25: freshness refusals, package hand-overs decided, Telegram getting through, B12.1's failure lines
  B14 3.6    : the look now comes every 30 minutes -- [COUNCIL-ENTRY] (noproof.per_cycle) and [PERSIST] saved
               (persist.saved) get 35-minute windows
  B14 1.1c   : every package entry ends in an order or a stated refusal within 10 minutes ([PKG-OUTCOME])
  B14 7.7    : no weekly report for 8 days = alarm
  B14 7.9    : every new B14 line that can fail has a check; B14's "must never happen" checks run from the start
               (no 24-hour warm-up). Start-up lines (built-in settings, account metrics) are covered by the start-up
               Telegram and the midnight summary instead -- a watchdog window can't see a line written only at start."""
import json, shutil, time
P = "config/heartbeat.json"
d = json.load(open(P, encoding="utf-8"))
pr = {p.get("id"): p for p in d.get("promises", [])}
changed = []
for pid in ("replayer.runs", "rl1.proposals_file", "b8.p1_ledger_fresh", "reftier.on_birth", "persist.bars_metric"):
    p = pr.get(pid)
    if p is None:
        print("NOT FOUND (left as is):", pid)
    elif p.get("enabled", True) is not False:
        p["enabled"] = False
        p["retired"] = "B13 item 3 (Desire 5 Oct), applied by B14"
        changed.append("retired " + pid)
p = pr.get("r1origin.on_birth")
if p is None:
    print("NOT FOUND (left as is): r1origin.on_birth")
elif p.get("when") != "market_open":
    p["when"] = "market_open"
    changed.append("r1origin.on_birth: only while markets are open")
br = d.get("daily_break_utc") or {"GOLD": ["21:50", "23:15"], "USTEC": ["21:50", "23:15"], "USOIL": ["21:50", "23:15"]}
for a, span in (("SILVER", ["21:50", "23:15"]), ("JP225", ["20:20", "21:40"])):
    if br.get(a) != span:
        br[a] = span
        changed.append("daily break %s %s-%s UTC" % (a, span[0], span[1]))
d["daily_break_utc"] = br
# ny_open: a watchdog check NAMED ny_open (none in the repo copy) is retired; the real source is gate.log_vs_ledger
for p in d.get("promises", []):
    if "ny_open" in (str(p.get("id", "")) + " " + str(p.get("tag", ""))).lower() and p.get("enabled", True) is not False:
        p["enabled"] = False
        p["retired"] = "B13 item 16 (Desire 5 Oct): ny_open false alarm"
        changed.append("retired " + str(p.get("id")))
p = pr.get("gate.log_vs_ledger")
if p is None:
    print("NOT FOUND (left as is): gate.log_vs_ledger")
elif "ny_open" not in (p.get("skip_gates") or []):
    p["skip_gates"] = sorted(set((p.get("skip_gates") or []) + ["ny_open"]))
    p["skip_why"] = "B14 (Desire 8 Oct): ny_open logs a block on cycles with no signal, so no practice row can exist"
    changed.append("gate.log_vs_ledger: ny_open skipped")
# B14 3.6: windows for the 30-minute look
for pid, win in (("noproof.per_cycle", 35), ("persist.saved", 35)):
    p = pr.get(pid)
    if p is None:
        print("NOT FOUND (left as is):", pid)
    elif int(p.get("window_min", 0) or 0) < win:
        changed.append("%s: window %s -> %d min" % (pid, p.get("window_min"), win))
        p["window_min"] = win
        p["b14_note"] = "B14 3.6 (Desire 8 Oct): the look comes every 30 minutes"
NEW = [
    {"id": "b13.freshness_refused", "type": "cadence", "tag": "[FRESHNESS] .* refused before sending", "max": 0,
     "window_min": 1440, "why": "B13 item 25: package entries are exempt -- any send-time refusal is unexpected"},
    {"id": "b13.pkg_decides", "type": "follows", "trigger": "-- the package takes over", "tag": "[PKG-.* (signal",
     "within_min": 1500, "window_min": 2880, "per_asset": True,
     "why": "B13 item 25: every package hand-over must end in ENTER or CANCEL within ~25 h"},
    {"id": "b13.tg_direct_ok", "type": "cadence", "tag": "[TELEGRAM] direct send failed", "max": 0, "window_min": 1440,
     "why": "B13 item 25: the direct Telegram (stops, health, breakers, MT5) must get through"},
    {"id": "b13.pkg_notice_ok", "type": "cadence", "tag": "[PKG] .* Telegram notice failed", "max": 0, "window_min": 1440,
     "why": "B13 item 25: package notices must reach Telegram"},
    {"id": "b121.ns_data", "type": "cadence", "tag": "[NS-DATA] .* changed after it was used", "max": 0,
     "window_min": 1440, "why": "B12.1: a candle the engine used changed afterwards"},
    {"id": "b121.labels", "type": "cadence", "tag": "[NS-LABELS] .* label check failed", "max": 0, "window_min": 1440,
     "why": "B12.1: the proof labels failed"},
    {"id": "b121.card", "type": "cadence", "tag": "[NS-CARD] .* not", "max": 0, "window_min": 1440,
     "why": "B12.1: the proof card was not drawn / written / sent"},
    {"id": "b121.chart", "type": "cadence", "tag": "[NS-CHART] .* not written", "max": 0, "window_min": 1440,
     "why": "B12.1: the interactive chart was not written"},
    {"id": "b121.advisory", "type": "cadence", "tag": "[COUNCIL-ADVISORY] .* failed", "max": 0, "window_min": 1440,
     "why": "B12: the council-advisory step failed"},
    {"id": "b121.pause_file", "type": "cadence", "tag": "[NS-PAUSE] .* failed", "max": 0, "window_min": 1440,
     "why": "B11-NS: the per-market pause file could not be read or saved"},
    {"id": "b13.combined", "type": "cadence", "tag": "[COMBINED-CHART] .* not", "max": 0, "window_min": 1440,
     "why": "B13 item 9: the combined chart was not drawn / sent"},
    # ---- B14 ----
    {"id": "b14.pkg_outcome", "type": "follows", "trigger": "[PKG-ENTER]", "tag": "[PKG-OUTCOME]", "within_min": 10,
     "window_min": 240, "per_asset": True,
     "why": "B14 1.1c: every package entry ends in an order or a stated refusal within 10 minutes"},
    {"id": "b14.pkg_expired", "no_warmup": True, "type": "cadence", "tag": "[PKG-OUTCOME] .* EXPIRED", "max": 0, "window_min": 1440,
     "why": "B14 1.1a: a package entry no trading look reached before the next 30m close"},
    {"id": "b14.weekly_report", "type": "file_fresh", "file": "logs/rl1_weekly_*.txt", "window_min": 11520,
     "why": "B14 7.7: the Saturday job must write its report every week (8-day window)"},
    {"id": "b14.look_runs", "type": "cadence", "tag": "[CYCLE]", "min": 1, "window_min": 35,
     "why": "B14 3.1: a look at every 30-minute close (the [CYCLE] line each look writes; also true on the old timer)"},
    {"id": "b14.look_late", "no_warmup": True, "type": "cadence", "tag": "[LOOK] .* late", "max": 0, "window_min": 1440,
     "why": "B14 3.1: a look that starts more than 5 minutes late"},
    {"id": "b14.map_built", "type": "cadence", "tag": "[MAP] .* built from", "min": 1, "window_min": 180,
     "per_asset": True, "when": "market_open", "why": "B14 4.1: each market's line map is rebuilt after every 1H close"},
    {"id": "b14.map_failed", "no_warmup": True, "type": "cadence", "tag": "[MAP] .* not", "max": 0, "window_min": 1440,
     "why": "B14 4.1: a line map that could not be built or summarised"},
    {"id": "b14.ns_map_missing", "no_warmup": True, "type": "cadence", "tag": "[NS-MAP]", "max": 0, "window_min": 1440,
     "why": "B14: a signal judged without its line map (the B13 rules were used instead)"},
    {"id": "b14.ns_history", "no_warmup": True, "type": "cadence", "tag": "[NS-DATA] .* the usual", "max": 0, "window_min": 1440,
     "why": "B14 1.2a: the engine's 75-day history could not be fetched"},
    {"id": "b14.walls_failed", "no_warmup": True, "type": "cadence", "tag": "[NS-WALLS]", "max": 0, "window_min": 1440,
     "why": "B14 6.4-6.5: the lock / staircase check could not run"},
    {"id": "b14.stair_failed", "no_warmup": True, "type": "cadence", "tag": "[NS-STAIR-FAIL]", "max": 0, "window_min": 1440,
     "why": "B14 6.5: a 1H close did not reach the staircase"},
    {"id": "b14.stop_hold_walls", "no_warmup": True, "type": "cadence", "tag": "[NS-STOP-HOLD] .* ns_wall", "max": 0, "window_min": 1440,
     "why": "B14 6.7: a lock or staircase move held back by the guard (it must reach MT5)"},
    {"id": "b14.telegram_cmds", "no_warmup": True, "type": "cadence", "tag": "[TELEGRAM] queued commands not run", "max": 0,
     "window_min": 1440, "why": "B14 (3.1 knock-on): Telegram close commands must run between looks"},
]
# B14 7.9: B14's "must never happen" checks run from the start (heartbeat.py reads "no_warmup") -- also when they
# were added by an earlier run of this tool
for q in d["promises"]:
    if q.get("id") in {x["id"] for x in NEW if x.get("no_warmup")} and not q.get("no_warmup"):
        q["no_warmup"] = True
        changed.append("%s: checked from the start (no warm-up)" % q.get("id"))
have = {q.get("id") for q in d["promises"]}
for np_ in NEW:
    if np_["id"] not in have:
        np_["enabled"] = True
        d["promises"].append(np_)
        changed.append("added " + np_["id"])
if changed:
    bk = P + ".bak_b14_" + time.strftime("%Y%m%d_%H%M%S")      # the file as it was, before this run's changes
    shutil.copy2(P, bk)
    json.dump(d, open(P, "w", encoding="utf-8"), indent=2)
    print("backup:", bk)
else:
    print("nothing to change (already applied) -- no backup needed")
print("enabled checks now: %d" % sum(1 for q in d["promises"] if q.get("enabled", True)))
print("changed (%d):" % len(changed))
for c in changed:
    print("  " + c)
