"""B15 (Desire 12 Oct, rule set Part 3) -- adds B15's watchdog checks to config/heartbeat.json IN PLACE, safely: a backup
first, only these additions, prints what it changed; running it twice changes nothing more (same pattern as
tools/b14_heartbeat_edit.py). b13.pkg_decides stays as it is: it now sees only B13 package records (a 4H record's
hand-over line says "-- the 4H clock takes over", which its trigger does not match).
  b15.pkg4_decides  : a 4H record ends in ENTER or CANCEL within its limits (7 days to the retest + 7 to the entry, + a day)
  b15.pkg4_reports  : a waiting 4H record writes where it stands at every 4H close (a follow-up within 5 hours; the short
                      window means a weekend or a daily break is never counted as silence). It needs
                      heartbeat.py's new "after_only" option: the progress line must not count as its own follow-up.
  b15.small_hourly  : must never happen -- a small line on the hourly clock while B15 is on (checked from the start)"""
import json, shutil, time
P = "config/heartbeat.json"
d = json.load(open(P, encoding="utf-8"))
changed = []
NEW = [
    {"id": "b15.pkg4_decides", "type": "follows", "trigger": "-- the 4H clock takes over", "tag": "4H clock decided",
     "within_min": 21600, "window_min": 23040, "per_asset": True,
     "why": "B15 Part 3: a 4H record must end in [PKG-ENTER] or [PKG-CANCEL] within its limits (7 + 7 days, + 1)"},
    {"id": "b15.pkg4_reports", "type": "follows", "trigger": "[PKG-4H]", "tag": "[PKG-.* (signal", "within_min": 300,
     "window_min": 600, "per_asset": True, "after_only": True,
     "why": "B15 Part 3: a waiting 4H record writes where it stands at every 4H close"},
    {"id": "b15.small_hourly", "no_warmup": True, "type": "cadence", "tag": "[B15-NEVER]", "max": 0, "window_min": 1440,
     "why": "B15 Part 3: a small line must never be traded on the hourly clock while B15 is on"},
]
for q in d["promises"]:                      # (a copy added before "after_only" existed gets it)
    if q.get("id") == "b15.pkg4_reports" and not q.get("after_only"):
        q["after_only"] = True
        changed.append("b15.pkg4_reports: the follow-up must come after the progress line")
have = {q.get("id") for q in d["promises"]}
for np_ in NEW:
    if np_["id"] not in have:
        np_["enabled"] = True
        d["promises"].append(np_)
        changed.append("added " + np_["id"])
if changed:
    bk = P + ".bak_b15_" + time.strftime("%Y%m%d_%H%M%S")      # the file as it was, before this run's changes
    shutil.copy2(P, bk)
    json.dump(d, open(P, "w", encoding="utf-8"), indent=2)
    print("backup:", bk)
else:
    print("nothing to change (already applied) -- no backup needed")
print("enabled checks now: %d" % sum(1 for q in d["promises"] if q.get("enabled", True)))
print("changed (%d):" % len(changed))
for c in changed:
    print("  " + c)
