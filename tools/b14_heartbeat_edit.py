"""B14 (Desire 15-16 Oct 2026) -- the watchdog checks of the new B14, written into config/heartbeat.json IN PLACE,
safely: a backup first, only the changes listed here, every change printed; running it twice changes nothing more.
Run with the bot STOPPED, at the install, before the bot is started again (the watchdog reads heartbeat.json at
start-up). Everything else in the file is left exactly as it is.

ADDS -- each one watches something that would otherwise fail without a word:
  b14.startup_failed    "[B14] START-UP FAILED" must never appear (the settings step at start-up raised: the rules run on
                        the rules file's own values, whatever phase_config.b14_tuning says)
  b14.settings_refused  "[B14] SETTING REFUSED" -- a value in phase_config.b14_tuning the rules refused -- must never appear
  b14.ladder_failed     "[NS-LADDER-FAIL]" must never appear (the ladder step raised: a trade's stop would stop climbing)
  b14.no_map            "[B14] ... no line map" -- the line target / the MA witness could not read the map: at most 4 a day
                        (said once per look, only while the new rules trade; more = the map is not there when entries
                        need it -- the fallback target is used). Counted from the start, like the "must never" checks.
  NOT added: a "the start-up line appeared" check. The watchdog has no "once since start" check; the clock.startup
  shape (min 1 in 30 days) is skipped for the first 30 days and then fails on day 30 -- it can never say anything
  useful. The install's own box check reads the start-up lines instead.
  b14.weekly_box        the Saturday job's box test writes logs/b14_box_test_*.txt every week (8-day window)
  b14.learner_ran       the Saturday job's learner writes logs/rl/b14_rl_*.txt every week (8-day window)
  b14.clock_wrong       "[CLOCK] ... SERVER OFFSET" must never appear (MT5's clock is off real time: every 4H candle shifts)
  b14.clock_unchecked   "[CLOCK] could not compare" must never appear (the clock check itself failed)
  b14.startup_checks_failed  "[CONFIG] start-up checks failed" must never appear (the start-up safety checks crashed)
  b14.nightly_summary   the nightly alarm summary writes logs/b13_alarm_summary_*.txt every night (2-day window)
  b14.weekly_forward    the Saturday job's weekly forward report writes logs/b14_forward_*.txt every week (8-day window)
  b14.data_gap          (Part 2, Desire 16 Oct) "[DATA-GAP]" must never appear: a 1H or 4H price file is missing candles
                        the market should have made, the newest candles are late, or the check itself failed
                        (src/update/data_gaps.py). Part 1 adds the 11 checks above; running this again in Part 2 adds
                        only this one.
MOVES the nightly pauses to New York time (Desire 16 Oct): "daily_break_utc" (right only in US winter time; JP225's
  wrong in both seasons) is replaced by "daily_break_ny" -- the 5 CFDs, 16:50-18:15 New York (they stop 17:00-18:00)
RETIRES (switched off with a note, not deleted):
  clock.startup         could never fire: skipped for 30 days after every start, and any [CLOCK] line -- even SERVER
                        OFFSET -- counted as fine. b14.clock_wrong and b14.clock_unchecked replace it (Desire 16 Oct).
REMOVES -- they watched code this batch removes (it never ran):
  b14.walls_failed, b14.stair_failed, b14.stop_hold_walls   the old wall lock and staircase
  replayer.runs, rl1.proposals_file                          the old RL-1 arms (switched off since 28 Sep)

  python tools\\b14_heartbeat_edit.py [file]       default: config\\heartbeat.json"""
import json
import shutil
import sys
import time

P = sys.argv[1] if len(sys.argv) > 1 else "config/heartbeat.json"
NEW = [
    {"id": "b14.startup_failed", "no_warmup": True, "type": "cadence", "tag": "[B14] START-UP FAILED", "max": 0,
     "window_min": 1440, "note": "B14 (16 Oct): the settings step at start-up raised -- the rules file's own values are used"},
    {"id": "b14.settings_refused", "no_warmup": True, "type": "cadence", "tag": "[B14] SETTING REFUSED", "max": 0,
     "window_min": 1440, "note": "B14 (16 Oct): a value in phase_config.b14_tuning the rules refused (today's value is used)"},
    {"id": "b14.ladder_failed", "no_warmup": True, "type": "cadence", "tag": "[NS-LADDER-FAIL]", "max": 0, "window_min": 1440,
     "note": "B14 (15 Oct): the ladder step must never raise -- if it did, a trade's stop would quietly stop climbing"},
    {"id": "b14.no_map", "no_warmup": True, "type": "cadence", "tag": "[B14] .*no line map", "max": 4, "window_min": 1440,
     "note": "B14 (15 Oct): the line target or the MA witness could not read the map -- a few at start-up are normal"},
    {"id": "b14.weekly_box", "type": "file_fresh", "file": "logs/b14_box_test_*.txt", "window_min": 11520,
     "note": "B14 (16 Oct): the Saturday job's box test writes its report every week (8-day window)"},
    {"id": "b14.learner_ran", "type": "file_fresh", "file": "logs/rl/b14_rl_*.txt", "window_min": 11520,
     "note": "B14 (16 Oct): the Saturday job's learner writes its report every week (8-day window)"},
    {"id": "b14.clock_wrong", "no_warmup": True, "type": "cadence", "tag": "[CLOCK] .*SERVER OFFSET", "max": 0,
     "window_min": 1440, "note": "B14 (Desire 16 Oct): MT5's clock is more than 15 min off real time -- every 4H candle shifts"},
    {"id": "b14.clock_unchecked", "no_warmup": True, "type": "cadence", "tag": "[CLOCK] could not compare", "max": 0,
     "window_min": 1440, "note": "B14 (Desire 16 Oct): the start-up clock check itself failed -- the clock is unchecked"},
    {"id": "b14.startup_checks_failed", "no_warmup": True, "type": "cadence", "tag": "[CONFIG] start-up checks failed",
     "max": 0, "window_min": 1440, "note": "B14 (Desire 16 Oct): the start-up safety checks crashed (was a log line only)"},
    {"id": "b14.nightly_summary", "type": "file_fresh", "file": "logs/b13_alarm_summary_*.txt", "window_min": 2880,
     "note": "B14 (Desire 16 Oct): the nightly alarm summary (tools/b14_task.ps1, 00:10 box time) writes its file every night"},
    {"id": "b14.weekly_forward", "type": "file_fresh", "file": "logs/b14_forward_*.txt", "window_min": 11520,
     "note": "B14 (Desire 16 Oct): the Saturday job's weekly forward report writes its file every week (8-day window)"},
    {"id": "b14.data_gap", "no_warmup": True, "type": "cadence", "tag": "[DATA-GAP]", "max": 0, "window_min": 1440,
     "note": "B14 Part 2 (Desire 16 Oct): a 1H/4H price file is missing candles the market should have made, or is behind"},
]
GONE = ("b14.walls_failed", "b14.stair_failed", "b14.stop_hold_walls", "replayer.runs", "rl1.proposals_file")
RETIRE = {"clock.startup": "B14 (Desire 16 Oct): could never fire -- skipped for 30 days after every start, and any "
                           "[CLOCK] line (even SERVER OFFSET) counted as fine; b14.clock_wrong and b14.clock_unchecked "
                           "replace it"}

raw = open(P, encoding="utf-8").read()
d = json.loads(raw)
changed = []
have = {q.get("id") for q in d["promises"]}
for q in NEW:
    if q["id"] not in have:
        d["promises"].append(dict(q, enabled=True))
        changed.append("added   %-22s %s" % (q["id"], q.get("tag") or q.get("file")))
for q in d["promises"]:
    if q.get("id") in RETIRE and q.get("enabled", True):
        q["enabled"] = False
        q["retired"] = RETIRE[q["id"]]
        changed.append("retired %-22s %s" % (q["id"], q.get("tag") or q.get("file")))
_ny14 = {a: ["16:50", "18:15"] for a in ("GOLD", "SILVER", "USTEC", "USOIL", "JP225")}
if "daily_break_utc" in d:
    changed.append("removed daily_break_utc %s (UTC, right only in US winter time)" % json.dumps(d.pop("daily_break_utc")))
if d.get("daily_break_ny") != _ny14:
    changed.append("set     daily_break_ny %s (New York time)" % json.dumps(_ny14))
    d["daily_break_ny"] = _ny14
keep = []
for q in d["promises"]:
    if q.get("id") in GONE:
        changed.append("removed %-22s %s" % (q["id"], q.get("tag") or q.get("file")))
        continue
    keep.append(q)
d["promises"] = keep
if changed:
    bk = P + ".bak_b14_" + time.strftime("%Y%m%d_%H%M%S")      # the file as it was, before this run's changes
    shutil.copy2(P, bk)
    tmp = P + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    json.load(open(tmp, encoding="utf-8"))                   # reads back cleanly before it replaces the file
    shutil.move(tmp, P)
    print("backup:", bk)
else:
    print("nothing to change (already applied) -- no backup needed")
print("enabled checks now: %d" % sum(1 for q in d["promises"] if q.get("enabled", True)))
print("changed (%d):" % len(changed))
for c in changed:
    print("  " + c)
