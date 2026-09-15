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
