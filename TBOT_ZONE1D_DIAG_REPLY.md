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

## Round 2 result — confirmed, not a guess

The `GATE CLOSED` line fired on every asset, every cycle, with the **same
`id(self)` as that asset's own working 4H line**:

```
[ZONE-DIAG] EURUSD 4H: store=561 cands=48 above=42 below=6 px=1.15869 id=2314298167120
[ZONE-DIAG] EURUSD 1D: GATE CLOSED (id=2314298167120) — governor_data['df_1d'] is None ...
[ZONE-DIAG] BTC 4H: store=602 cands=90 above=34 below=56 px=77130.79 id=2314197735792
[ZONE-DIAG] BTC 1D: GATE CLOSED (id=2314197735792) — governor_data['df_1d'] is None ...
```
(GOLD, USTEC, USOIL identical pattern.)

This is the same aggregator instance, in the same `_build_composite_state()`
call, where `df_4h` arrives populated and `df_1d` does not. That rules out an
instance mismatch entirely — it's the same object.

`_build_composite_state` has two live council-mode call sites, both carrying
an identical "DATA-1 Item 7" top-up (`if mtf_regime.get("df_1d") is None:
mtf_regime["df_1d"] = self._df_1d_cache.get(asset_name)`):

- **site2**, `trade_asset()` (main.py:5654-5674) — runs *after* its own
  fresh `self._df_1d_cache[asset_name] = self._fetch_1d_data(asset_name)`
  fetch in the same method call (main.py:5517).
- **site3**, `_update_asset_signal()` (main.py:7619-7639) — and critically,
  `_update_asset_signal()` runs **before** `trade_asset()` every cycle
  (main.py:3751 vs :3972). If site3 is the one firing, it's reading
  whatever `self._df_1d_cache` held before *this* cycle's fetch ran —
  either stale-by-one-cycle, or genuinely empty if the cache isn't
  surviving between cycles.

Since both sites' top-up code is identical, I can't yet tell from the
`GATE CLOSED` line alone which one is producing it, or whether the cache is
merely stale-by-a-cycle (would resolve as trade_asset catches up) versus
never actually persisting (would need a real fix). This also revisits a
"Batch 610" finding that already tried a similar top-up and believed it
had landed — worth knowing this is the second time this exact symptom has
needed a fix here.

## Round 3 — shipped

Added one more temporary line to each call site (not inside
`_build_composite_state` this time — directly at site2 and site3), tagging
which one fires and showing both `self._df_1d_cache.get(asset_name)` and
`mtf_regime.get("df_1d")` at that exact point:

```
[ZONE-DIAG] <asset> site2(trade_asset): df_1d_cache=<present|None> mtf_df_1d=<present|None>
[ZONE-DIAG] <asset> site3(_update_asset_signal): df_1d_cache=<present|None> mtf_df_1d=<present|None>
```

Reading it:

| Pattern | Meaning |
|---|---|
| site2 shows `df_1d_cache=None` | the fetch/cache write itself is failing for this asset — contradicts the "fetch succeeds" finding and needs re-checking directly |
| site2 shows `df_1d_cache=present` but only site3 logs `GATE CLOSED` | confirms the cross-pass timing gap: `_update_asset_signal` reads before `trade_asset` refreshes the cache each cycle — fix is either reordering the two passes or giving `_update_asset_signal` its own guaranteed-fresh read |
| both sites show `df_1d_cache=present` yet `_build_composite_state` still logs `df_1d is None` | the top-up itself, or the dict identity between top-up and build call, has a bug — narrower fix, inside `_build_composite_state`'s own read |

Same restart-and-capture as before:

```powershell
Select-String -Path logs\trading_bot.log -Pattern "ZONE-DIAG" | Select -Last 30 | ForEach-Object { ($_.Line -split " - ",4)[3] }
```

---

## Round 3 result — a real surprise, and no exceptions

`site3(_update_asset_signal)` fired on every asset, every cycle, and this
time it showed something new: `df_1d_cache=None` — not just the `mtf_regime`
copy, the **actual `self._df_1d_cache` dict itself** is empty when
`_update_asset_signal` reads it. And `site2(trade_asset)` still never
appeared, not once, across two full capture rounds.

Checked the obvious explanation first — is `trade_asset()` silently crashing
before reaching its own council block?

```powershell
Select-String -Path logs\trading_bot.log -Pattern "\[ERROR\].*trade failed" | Select -Last 20
```

Empty. No exceptions. `trade_asset()` runs clean for every asset, every
cycle — it just never takes the branch containing site2.

Traced the dispatch itself (main.py:5640-5648):

```python
if isinstance(aggregator, dict) and aggregator.get("mode") == "hybrid":
    ...
elif isinstance(aggregator, dict) and aggregator.get("mode") == "council":
    # site2 lives here
    ...
else:
    # PERFORMANCE / plain council (non-dict) mode
    signal, details = aggregator.get_aggregated_signal(df, ..., governor_data=mtf_regime, ...)
```

Both places `self.aggregators[asset_name]` gets constructed as council mode
(main.py:1582 and :3473) correctly include `"mode": "council"` in the dict —
verified directly, not assumed. But `main.py:1489` and `:3543` both assign a
**bare** aggregator object (no dict, no "mode" key) to
`self.aggregators[asset_name]` under other conditions (plain "performance"
mode construction, and what looks like the auto-preset reinit fallback).

If `aggregator` is a bare object when `trade_asset()` reads it, dispatch
falls straight to the bottom `else:` — which calls
`aggregator.get_aggregated_signal()` directly and **never rebuilds
composite_state at all**. `InstitutionalCouncilAggregator` has no
`_build_composite_state` method of its own (verified — plain class, no
mixin). That branch just reads back whatever `mtf_regime["composite_state"]`
already holds — which, this cycle, is whatever `_update_asset_signal()`'s
site3 already built, df_1d=None bug and all.

That would explain everything at once: no exception (it's a valid, if
unintended, code path), site2 never firing (it's simply not the branch being
taken), and the None propagating through to real trading decisions instead
of being locally contained.

**Not yet confirmed** — I don't have direct proof `aggregator` really is a
bare object at this read site, only that it's the one explanation consistent
with every piece of evidence so far.

## Round 4 — shipped

One line directly at the trade_asset dispatch point, before the if/elif/else,
logging `type(aggregator).__name__`, whether it's a dict, and its `mode` key
if so:

```
[ZONE-DIAG] <asset> trade_asset dispatch: type=<...> is_dict=<True|False> mode=<...>
```

If `is_dict=False`, that's confirmed — the fix is finding why
`self.aggregators[asset_name]` isn't the council dict it was constructed as
by the time `trade_asset()` reads it (likely the auto-preset reinit path at
main.py:3543, or a per-asset mode override). If `is_dict=True mode=council`,
this whole branch-mismatch theory is wrong and the search moves elsewhere.

Same restart-and-capture:

```powershell
Select-String -Path logs\trading_bot.log -Pattern "ZONE-DIAG" | Select -Last 30 | ForEach-Object { ($_.Line -split " - ",4)[3] }
```

---

## Confirmed root cause — not a permanent bug

Round 4's data showed something new: earlier in the capture window, `site2`
had fired *successfully* (`df_1d_cache=present`, real zone data) for
GOLD/USTEC/EURUSD/USOIL — contradicting the "wrong branch, every cycle"
theory from round 3. That meant intermittent, not deterministic — a timing
question, not a misconfiguration.

Checked the two most likely fetch-failure paths directly against the
existing (non-diagnostic) log lines already in `_fetch_1d_data()`:

```powershell
Select-String -Path logs\trading_bot.log -Pattern "\[1D\].*Insufficient daily data"
Select-String -Path logs\trading_bot.log -Pattern "\[1D\].*Failed to fetch daily data"
```

Both empty. The fetch itself was never failing.

Asked for one more capture, with timestamps, after a genuinely patient wait
(10+ minutes) instead of an immediate restart-and-grep. That settled it:

```
20:13:53  USOIL 1D: store=295 ... (from the tail of an earlier, already-working state)
20:13:54  BTC trade_asset dispatch: type=dict is_dict=True mode=council
20:13:54  BTC site2(trade_asset): df_1d_cache=present mtf_df_1d=present
... (all six assets succeed, first trade_asset pass since the restart)
20:19:33  BTC site3(_update_asset_signal): df_1d_cache=present mtf_df_1d=present
... (stable from here on, both passes healthy every cycle)
```

**Root cause: a cold-start race, not a permanent defect.**
`_update_asset_signal()` runs first every cycle (main.py:3751), before
`trade_asset()` (main.py:3972) — and only `trade_asset()`'s own fetch
populates `self._df_1d_cache`/`self._df_4h_cache` for the first time.
`self._df_1d_cache = {}` starts genuinely empty on every process start (no
cross-restart persistence), so on the very first cycle after *any* restart,
`_update_asset_signal()` legitimately reads `None` — not a bug, a real gap
that self-heals the moment `trade_asset()` completes its first pass, usually
within about five minutes.

This explains every piece of evidence from all four rounds without
contradiction: `_build_zone_view` was always healthy, the store was never
empty, the filter was never wrong, no exception ever fired, and the
"wrong branch" observation in round 3 was real but incomplete — it's the
*expected* state before `trade_asset()`'s first pass, not a permanent
misrouting.

**Why it looked permanent on 1 September specifically**: four restarts that
day. If the bot rarely got much past this ~5-minute warm-up window before
being restarted again, a one-cycle gap would look like an every-cycle
failure to anyone sampling the log at the wrong moment — which is exactly
what happened, repeatedly, across this whole investigation's own early
captures.

## The fix

Closed the gap directly rather than leaving it to self-heal: in
`_update_asset_signal()`'s DATA-1 Item 7 top-up block (main.py, "site 3"),
a cache miss now falls through to a direct `self._fetch_1d_data(asset_name)`
/ `self._fetch_4h_data(asset_name)` call and populates the cache right
there, instead of accepting `None` for that cycle. Applied to both df_1d and
df_4h for symmetry, even though df_4h was never observed failing — same
class of gap, same fix, cheap to close preventively.

Not applied at `trade_asset()`'s own site (site2): its cache is always warm
by the time it reads it, since it does its own fetch moments earlier in the
same call — no evidence it ever needs the fallback, so it wasn't added
there.

All four rounds of temporary `[ZONE-DIAG]` logging have been reverted, per
the original instruction to revert once the cause is known. Verified zero
remaining `ZONE-DIAG` log statements — the only remaining trace is an
explanatory code comment at the fix site.

---

Compiled, imported, JSON/py_compile clean. Committed and pushed.
