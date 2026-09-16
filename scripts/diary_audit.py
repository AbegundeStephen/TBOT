import glob, json, collections
keys = collections.Counter(); n = 0; versions = collections.Counter(); sources = collections.Counter()
for f in sorted(glob.glob("logs/episodes/*.jsonl")):
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: e = json.loads(line)
        except Exception: continue
        n += 1; versions[e.get("schema_version", 1)] += 1; sources[e.get("source")] += 1
        for k in e.keys(): keys[k] += 1
        cs = e.get("composite_state") or {}
        for k in cs.keys(): keys["composite_state." + k] += 1
print("rows:", n, "| versions:", dict(versions), "| sources:", dict(sources))
for k, c in sorted(keys.items()): print(f"{c:6d}  {k}")

# B3 GATE-1 G6: second table -- gate_id x gate_stage -> rows, sides, with_outcome.
# with_outcome = row has a recorded net_pnl_r (i.e. the shadow position actually
# closed), as opposed to a refusal that was only ever recorded, never opened
# (the G4 pre_direction/no-signal "record, don't shadow" path).
gate_rows = collections.defaultdict(lambda: {"rows": 0, "long": 0, "short": 0, "with_outcome": 0})
seen_gate_ids = set()
for f in sorted(glob.glob("logs/episodes/*.jsonl")):
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: e = json.loads(line)
        except Exception: continue
        gid = e.get("gate_id")
        if not gid:
            continue
        seen_gate_ids.add(gid)
        stat = gate_rows[(gid, e.get("gate_stage", "unknown"))]
        stat["rows"] += 1
        side = e.get("side")
        if side == "long": stat["long"] += 1
        elif side == "short": stat["short"] += 1
        if e.get("net_pnl_r") is not None: stat["with_outcome"] += 1

print("\ngate_id x gate_stage -> rows (L/S) with_outcome")
for (gid, stage), stat in sorted(gate_rows.items()):
    print(f"  {gid:22s} {stage:14s} rows={stat['rows']:5d}  "
          f"L={stat['long']:5d} S={stat['short']:5d}  with_outcome={stat['with_outcome']:5d}")

try:
    registry = json.load(open("config/gates.json", encoding="utf-8")).get("gates", {})
except Exception:
    registry = {}
zero_rows = sorted(g for g in registry if g not in seen_gate_ids)
print("\ngates in gates.json with zero rows ever:", zero_rows if zero_rows else "(none)")
