import json, shutil, datetime

P = "config/aggregator_presets.json"
shutil.copy(P, P + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M"))
with open(P, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = []
def walk(node, path=""):
    if isinstance(node, dict):
        if "council_trend_aligned" in node:
            aligned = node["council_trend_aligned"]
            old = node.get("council_counter_trend", "(absent)")
            node["council_counter_trend"] = aligned   # set explicitly even if absent,
            changed.append((path, old, aligned))      # so the deep-merge over Python
        for k, v in node.items():                     # defaults is also symmetric
            walk(v, f"{path}/{k}")

walk(data)
with open(P, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Symmetrized {len(changed)} preset blocks:")
for path, old, new in changed:
    print(f"  {path}: counter {old} -> {new}")
