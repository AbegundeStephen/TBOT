# TBOT — 1D zone tier diagnostic: findings from round 1, round 2 shipped

**Date:** 2 September 2026
**To:** Desire Omunakwe
**Re:** zone1d lo=None hi=None diagnostic (`src/execution/composite_state_builder.py`)

---

## What the first capture proved

Your two diagnostic lines came back clean — 14 lines, six assets, both timeframes:

```
[ZONE-DIAG] EURUSD 4H: store=561 cands=48 above=42 below=6 px=1.15869
[ZONE-DIAG] EURUSD 1D: store=561 cands=57 above=48 below=9 px=1.15869
[ZONE-DIAG] USOIL 4H: store=295 cands=41 above=0 below=41 px=89.423
[ZONE-DIAG] USOIL 1D: store=295 cands=48 above=0 below=48 px=89.423
[ZONE-DIAG] BTC 4H: store=602 cands=90 above=34 below=56 px=77130.79
[ZONE-DIAG] BTC 1D: store=602 cands=99 above=42 below=57 px=77130.79
[ZONE-DIAG] GOLD 4H: store=395 cands=40 above=33 below=7 px=4368.447
[ZONE-DIAG] GOLD 1D: store=395 cands=52 above=42 below=10 px=4368.447
[ZONE-DIAG] USTEC 4H: store=440 cands=39 above=32 below=7 px=29130.19
[ZONE-DIAG] USTEC 1D: store=440 cands=48 above=37 below=11 px=29130.19
```

**`_build_zone_view` is not the bug.** Every single 1D call returned real
candidates both above and below price — GOLD 1D: above=42 below=10, BTC 1D:
above=42 below=57, USTEC 1D: above=37 below=11. USOIL's `above=0` is present
identically on *both* 4H and 1D (a legitimate "price above every level" case,
not a 1D-specific failure). No `EARLY RETURN` line fired anywhere in the
capture.

That directly rules out three of your four candidate rows: the store isn't
empty, the filter isn't rejecting everything, and there's no price-comparison
type mismatch. The function, the store, and the filter are all healthy.

## What's left, and round 2

`_build_zone_view` for 1D only ever runs *inside* a gate:

```python
if _df_1d is not None:
    ...
    _v1 = self._build_zone_view(_df_1d, self.asset_type, "1D", _price_now)
    state.zone_1d_current_upper = _v1["current_upper"]
```

Both of your original diagnostic lines live *inside* `_build_zone_view` — so
if `governor_data["df_1d"]` is `None` on a given `_build_composite_state`
call, the entire block is skipped silently, `state.zone_1d_current_upper/lower`
stay at their dataclass default of `None`, and **neither original line can
fire** to tell us so. That reproduces `[M1-CONVERGE] zone1d lo=None hi=None`
without contradicting a single line of what round 1 captured — round 1 could
only ever show us the calls where the gate was already open.

Added a third temporary line — an `else` on that exact gate — plus `id(self)`
on all three lines (including the pre-existing `EARLY RETURN`/`store=`
line), so the next capture can show whether the `None`-gate calls come from
the same aggregator instance as the working ones (a timing/pass issue —
`_build_composite_state` has multiple call sites; some use `_perf_agg`,
others `_lsm_comp`) or a genuinely different instance. Left the permanent
`[M1-CONVERGE]` log itself untouched — out of scope for a temporary
diagnostic — and will cross-reference by timestamp instead.

**Still temporary. Still no behaviour change** — one more `logger.warning`,
nothing else. Revert all three once the cause is confirmed.

## Next

Same as round 1: restart, wait one full cycle, then:

```powershell
Select-String -Path logs\trading_bot.log -Pattern "ZONE-DIAG" | Select -Last 20 | ForEach-Object { ($_.Line -split " - ",4)[3] }
```

If `GATE CLOSED` lines appear, that's confirmed root cause — `governor_data['df_1d']`
is arriving `None` at a specific call site, and the fix is at whichever site's
`mtf_regime`/`governor_data` construction is skipping the cache top-up
(DATA-1 Item 7's pattern — sites 1/2/3 already have it; worth checking whether
there's a 4th site that doesn't).

If no `GATE CLOSED` lines appear at all, the gate is never the problem either,
and the `None`s Desire is seeing must be coming from a stale/cached
`composite_state` read somewhere downstream rather than from `_build_composite_state`
itself — a different investigation, not this one.

---

Compiled, imported, JSON/py_compile clean. Committed and pushed.
