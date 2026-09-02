# TBOT — Batch FIX-1 Report to Desire

**Date:** 2 September 2026
**Batch:** FIX-1 (five segments)
**Commit:** `3e968ca` (pushed to `main`)

---

## Segments applied

| Seg | What | Status |
|---|---|---|
| F1a/b/c | `main.py` ~1214-1290: missing/empty mode → CRITICAL + `RuntimeError` (refuses to start). Invalid mode → same. Non-council mode requires `phase_config.legacy_engine_enabled=true` explicitly (defaults `false`). | Applied |
| F1d | `hybrid_aggregator_selector._default_analysis`: `'performance'` → `'none'` on insufficient data. | Applied |
| F1e | `main.py`: caller now honours `selected_mode == "none"` as a hold (`return 0, {...}`) instead of falling through to the legacy branch. | Applied |
| F2 | `main.py`, right after `self.config = json.load(f)`: validates 10 required top-level sections present; raises `RuntimeError` if any missing. | Applied |
| F3 | `mtf_regime_detector.py:127-132`: hard floor `400 → 200` bars, with an informational log in the 200-399 range. Backfill (Stephen, manual) already brought GOLD/USTEC/EURUSD/USOIL to 900 daily bars each. | Applied (code); backfill done separately |
| F4 | `DynamicThresholds.save_cache()` moved from shutdown-only into the trading cycle, beside the MEASURE-2 S1 shadow save. | Applied (with two corrections — see below) |
| F5 | Config only, `phase_config.legacy_engine_enabled: false` added to all three files. | Applied |

## Marker count

13 (spec expected 8+).

## Config structure check, before / after

Run against `config.prod.json` at each commit:

- **Before** (`210ca9e`, pre-FIX-1): `OK - 21 top-level keys, 87 phase flags`
- **After** (`3e968ca`, post-FIX-1): `OK - 21 top-level keys, 88 phase flags` (the +1 is `legacy_engine_enabled`)

Current live values confirmed:

```
mode: council
legacy_engine_enabled: False
sl_on_exchange: True   tp: True
```

## `_notify_critical`

Does not exist in `main.py`. Used the fallback per the spec's own instruction — the real MT5-outage watchdog's shape, `self.telegram_bot.notify_error(msg)` wrapped in `self._send_telegram_notification(...)`, guarded by `if self.telegram_bot:`. Confirmed `self.telegram_bot` is already constructed (line 916) before F1a's code runs (line 1214+), so the guard is real, not decorative.

## Deviations from the literal spec

Both in F4, both found by reading the actual code before applying rather than trusting the spec's description:

1. **Wrong dict-unwrap key.** The spec's dict-aggregator branch read `dynamic_thresholds` off `_agg.get("council")`. That attribute actually lives on `PerformanceWeightedAggregator` (`signal_aggregator.py:366`), not `InstitutionalCouncilAggregator` — as written, it would have silently returned `None` for every council/hybrid-mode asset, i.e. every asset in production today. Corrected to unwrap via `"performance"`/`"livermore"`, matching the pattern already used at `_shadow_open_blocked` and MEASURE-1's Lane C generator.

2. **Not one shared instance.** The spec assumed one shared `DynamicThresholds` instance ("break after the first save — the cache is shared, not per-asset"). Verified this is false: each asset's aggregator constructs its own independent instance; all six just default to the same cache file path. Breaking after the first would have silently dropped the other five assets' session data on every save. Now saves all distinct instances (id-deduped).

3. **Flagging, not fixing.** `DynamicThresholds.save_cache()` itself still does a whole-dict dump with no merge-on-write, and each instance only loads the shared file once at startup — so even saving all six, whichever asset saves last within a cycle can still clobber another asset's same-cycle update. Real gap, but it's in `DynamicThresholds`'s own design, not something a "call the existing method periodically" fix can resolve. Separate item if it's worth a ruling.

F5's `place_vtm_sl_on_exchange`/`place_vtm_tp_on_exchange` were already `true` in all three files before this batch touched anything — no edit was needed there; only `legacy_engine_enabled` was new.

---

All changes compiled, imported, and JSON-validated; committed as `3e968ca` and pushed to `main`.
