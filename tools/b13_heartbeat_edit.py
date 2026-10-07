"""B13 Part 2c (Desire 5 Oct) -- edits the watchdog's check list (config/heartbeat.json) IN PLACE, safely:
a backup first, only the named changes, prints what it changed, and running it twice changes nothing more.
  item 3 : retire the 4 dead checks + the 'bars passed=' check (false alarms; re-enable once its log format is known)
  item 6 : the R1-ORIGIN check only counts while markets are open (weekend skip is in heartbeat.py)
  item 16: nightly breaks for SILVER and JP225 (their pauses were raising price/tick false alarms)
  item 25: new checks -- any send-time freshness refusal; every package hand-over decided; direct Telegram and
           package notices getting through
  item 16 (v2): the ny_open alarm retired wherever it is a watchdog check"""
import json, shutil, sys, time
P = "config/heartbeat.json"
d = json.load(open(P, encoding="utf-8"))
bk = P + ".bak_b13_" + time.strftime("%Y%m%d_%H%M")
shutil.copy2(P, bk)
pr = {p.get("id"): p for p in d.get("promises", [])}
changed = []
for pid in ("replayer.runs", "rl1.proposals_file", "b8.p1_ledger_fresh", "reftier.on_birth", "persist.bars_metric"):
    p = pr.get(pid)
    if p is None:
        print("NOT FOUND (left as is):", pid)
    elif p.get("enabled", True) is not False:
        p["enabled"] = False
        p["retired"] = "B13 item 3 (Desire 5 Oct)"
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
if "b13.freshness_refused" not in pr:
    d["promises"].append({"id": "b13.freshness_refused", "type": "cadence", "tag": "[FRESHNESS] .* refused before sending",
                          "max": 0, "window_min": 1440, "enabled": True,
                          "why": "B13 item 25: package entries are exempt now -- any send-time refusal is unexpected; surface it"})
    changed.append("added b13.freshness_refused")
# v2 (B13 items 16 and 25): retire the ny_open alarm wherever it is a watchdog check; add B13's own checks
for p in d.get("promises", []):
    if "ny_open" in (str(p.get("id", "")) + " " + str(p.get("tag", ""))).lower() and p.get("enabled", True) is not False:
        p["enabled"] = False
        p["retired"] = "B13 item 16 (Desire 5 Oct): ny_open false alarm"
        changed.append("retired " + str(p.get("id")))
for np_ in ({"id": "b13.pkg_decides", "type": "follows", "trigger": "-- the package takes over", "tag": "[PKG-.* (signal",
             "within_min": 1500, "window_min": 2880, "per_asset": True, "enabled": True,
             "why": "B13 item 25: every package hand-over must end in ENTER or CANCEL within ~25 h -- never stuck silently"},
            {"id": "b13.tg_direct_ok", "type": "cadence", "tag": "[TELEGRAM] direct send failed", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B13 item 25: the direct Telegram (stops, health, breakers, MT5) must get through"},
            {"id": "b13.pkg_notice_ok", "type": "cadence", "tag": "[PKG] .* Telegram notice failed", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B13 item 25: package notices must reach Telegram"},
            # B13 item 25 (5 Oct triage C7): B12.1's new log lines had no checks -- one per failure line
            {"id": "b121.ns_data", "type": "cadence", "tag": "[NS-DATA] .* changed after it was used", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B12.1: a candle the engine used changed afterwards (data integrity)"},
            {"id": "b121.labels", "type": "cadence", "tag": "[NS-LABELS] .* label check failed", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B12.1: the proof labels failed (display only, but must surface)"},
            {"id": "b121.card", "type": "cadence", "tag": "[NS-CARD] .* not", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B12.1: the proof card was not drawn / written / sent"},
            {"id": "b121.chart", "type": "cadence", "tag": "[NS-CHART] .* not written", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B12.1: the interactive chart was not written"},
            {"id": "b121.advisory", "type": "cadence", "tag": "[COUNCIL-ADVISORY] .* failed", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B12: the council-advisory step failed (the council's vote would then stand)"},
            {"id": "b121.pause_file", "type": "cadence", "tag": "[NS-PAUSE] .* failed", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B11-NS: the per-market pause file could not be read or saved"},
            {"id": "b13.combined", "type": "cadence", "tag": "[COMBINED-CHART] .* not", "max": 0, "window_min": 1440,
             "enabled": True, "why": "B13 item 9: the combined chart was not drawn / sent"}):
    if np_["id"] not in {q.get("id") for q in d["promises"]}:
        d["promises"].append(np_)
        changed.append("added " + np_["id"])
json.dump(d, open(P, "w", encoding="utf-8"), indent=2)
print("backup:", bk)
print("changed:", changed if changed else "nothing (already applied)")
