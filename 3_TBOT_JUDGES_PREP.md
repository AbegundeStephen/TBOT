# 3 — JUDGES PREP BATCH (eleven fixes)

**Deploy order: FILE 3 of 4. Ships after File 2. Everything here must land before File 4.**

Owner: Desire · Developer: Stephen
**Files touched:** `council_aggregator.py`, `composite_state_builder.py`

---

## WHY THIS BATCH EXISTS

The judges are about to be rebuilt (File 4). Before that happens, the things they *read* have to be correct.

Right now several judges are reading broken or blunt inputs — a direction-blind flag, a number written by the wrong calculation, a field nobody reads. Rebuild the judges on top of those and you've built carefully-weighted scoring on bad data.

**Eleven fixes. All surgical. None change the shape of a judge — they fix what a judge is looking at.**

Nothing here is behind a flag except where noted: these are corrections, not features.

---

## THE ELEVEN

| # | Fix | Judge | What's wrong |
|---|-----|-------|--------------|
| F1 | Directional BOS | STRUCTURE | Scores a BUY on a bearish break |
| F2 | Defense into BOS base | STRUCTURE | A break at a proven level scores same as one in open air |
| F3 | defense_strength clobber | STRUCTURE | Reads a number written by a different EMA |
| F4 | CHoCH existence-only | STRUCTURE | Scores reversal direction — that belongs to REVERSION |
| P1 | Anchored spring | PATTERN | Uses a drifting reference instead of the proven one |
| P2 | EMA-50 six-state | PATTERN | Scores **zero** on trending MT5 |
| M1 | Super-Cycle no early return | MOMENTUM | Stops measuring momentum in strong trends |
| M2 | MACD units | MOMENTUM | Adds a raw number to a weighted score |
| R1 | Spring shared source | registry | (covered in File 2 — verify only) |
| X1 | setup_died wiring | VTM | Computed every cycle, never read |
| X2 | Registry verification | all | Confirm File 2's registry landed |

---

# F1 — STRUCTURE READS DIRECTIONAL BREAKS

**File:** `council_aggregator.py`, `_judge_structure_bidirectional`

**What's wrong, plainly.** The board records *which way* a break went — `bos_bullish` and `bos_bearish` are separate fields. STRUCTURE ignores both and reads the blunt `bos_detected`, which is True for either. So a **bearish** break can hand full credit to the **BUY** side.

The trajectory tracker in the same codebase already does this correctly — it reads `bos_bullish`/`bos_bearish` to decide setup direction. STRUCTURE was simply never updated.

**Find, in the BUY scoring block:**
```python
                if (
                    bos_detected
                    and (is_bullish_regime or is_breakout_mode)
                    and not failed_breakout
                ):
                    buy_score = weight
                    buy_exp = f"STRUCT BUY: ✅ BOS confirmed ({weight:.1f})"
```

**Replace with:**
```python
                # F1: read the DIRECTIONAL break. bos_detected is True for a
                # break either way, so a bearish break was crediting the buy
                # side. The board already records which way it went.
                if (
                    bos_bullish
                    and (is_bullish_regime or is_breakout_mode)
                    and not failed_breakout
                ):
                    buy_score = weight
                    buy_exp = f"STRUCT BUY: ✅ Bullish BOS confirmed ({weight:.1f})"
```

**Add the read near the other `_cs(...)` reads at the top of the block:**
```python
                bos_bullish = bool(_cs("bos_bullish", False))
                bos_bearish = bool(_cs("bos_bearish", False))
```

**Mirror it on the SELL side** — wherever the sell branch uses `bos_detected` for a breakdown, use `bos_bearish`.

⚠️ **Write both sides explicitly. Do not assume symmetry** — this project has been bitten by half-applied directional fixes before.

**CHECK F1:** grep `bos_detected` inside `_judge_structure_bidirectional` → should be 0 (or only in a deliberate non-directional context you flag back to Desire).

---

# F2 — A BREAK AT A PROVEN LEVEL IS WORTH MORE

**File:** `council_aggregator.py`, `_judge_structure_bidirectional`

**What's wrong.** The scoring is a cascade: if BOS fires, it returns flat `weight` and the `elif level_defended:` branch never runs. So a break through a level that has been tested and held four times scores **identically** to a break through empty air.

For a judge whose entire job is structural location, that's the wrong instinct. A break *at proven structure* is the stronger event.

**In the BUY branch, replace the flat `buy_score = weight` with:**
```python
                    # F2: fold defense into the break itself. A break AT a
                    # proven, defended level is stronger evidence than a break
                    # in open air — the old cascade scored them identically
                    # because `elif level_defended` was unreachable once BOS won.
                    _f2_defense = float(_cs("defense_strength", 0.0) or 0.0)
                    _f2_bonus = 0.15 * weight * min(max(_f2_defense, 0.0), 1.0)
                    buy_score = min(weight, weight * 0.85 + _f2_bonus)
                    buy_exp = (
                        f"STRUCT BUY: ✅ Bullish BOS ({buy_score:.2f})"
                        + (f" +defended({_f2_defense:.2f})" if _f2_defense > 0 else " (open air)")
                    )
```

**Mirror on SELL.**

**Note:** a bare break now scores `0.85 × weight` rather than full weight; full weight is reached only when the break happens at a well-defended level. That is deliberate — it creates headroom for the proven-level evidence to matter.

**CHECK F2:** logs show `+defended(0.xx)` on some breaks and `(open air)` on others.

---

# F3 — STOP THE DEFENSE NUMBER BEING OVERWRITTEN ⚠️

**File:** `composite_state_builder.py`, `_update_ma_defense`

**What's wrong — and the code already admits it.** The docstring says:

> *"defense_strength and absorption_detected are NOT parameterized — every call (1H-50, 1H-200, 1D-50, 1D-200) writes the same shared fields, so whichever call runs last wins."*

The builder calls this method **four times**. The **1D-200** call runs last. So `defense_strength` — which STRUCTURE reads to judge how well a *horizontal price level* was defended — actually holds a value describing how the **daily 200 EMA** behaved.

**F2 above makes this worse**, because F2 makes STRUCTURE depend on that number more heavily. F3 must ship in the same batch.

**Fix:** parameterise the two shared fields the same way `ema_{span}_status{suffix}` already is.

**In `_update_ma_defense`, replace the two shared writes:**
```python
                state.defense_strength = min(1.0, _wick / max(_body, 0.0001) / 3.0)
```
**with:**
```python
                # F3: parameterised so the four calls (1H-50, 1H-200, 1D-50,
                # 1D-200) stop clobbering one another. The unsuffixed 1H-50
                # write is preserved as the canonical one STRUCTURE reads.
                _f3_strength = min(1.0, _wick / max(_body, 0.0001) / 3.0)
                setattr(state, f"defense_strength_{span}{suffix}", _f3_strength)
                if span == 50 and suffix == "":
                    state.defense_strength = _f3_strength
```

**And for absorption:**
```python
                    setattr(state, f"absorption_detected_{span}{suffix}", True)
                    if span == 50 and suffix == "":
                        state.absorption_detected = True
```

**Add the new fields to `composite_state.py`** alongside the existing `defense_strength` / `absorption_detected`:
```python
    # F3: per-call defense readings. The unsuffixed fields above stay as the
    # canonical 1H-50 values every existing consumer already reads.
    defense_strength_50: float = 0.0
    defense_strength_200: float = 0.0
    defense_strength_50_1d: float = 0.0
    defense_strength_200_1d: float = 0.0
    absorption_detected_50: bool = False
    absorption_detected_200: bool = False
    absorption_detected_50_1d: bool = False
    absorption_detected_200_1d: bool = False
```

⚠️ **The unsuffixed `defense_strength` and `absorption_detected` must keep working** — VOLUME's absorption segment and other consumers read them. This fix stops them being *overwritten*, it does not remove them.

**CHECK F3:** log `defense_strength` and `defense_strength_200_1d` on the same cycle. They should now differ. Before this fix they were always identical.

---

# F4 — CHoCH: STRUCTURE SCORES THAT IT HAPPENED, NOT WHAT IT MEANS

**File:** `council_aggregator.py`, `_judge_structure_bidirectional`

**What's wrong.** STRUCTURE currently gives `+0.3 × weight` for CHoCH in a directional way — CHoCH in a bull regime credits SELL, CHoCH in a bear regime credits BUY. That is a *reversal thesis*, and under the six-judge design the reversal thesis belongs to REVERSION.

**The split, decided:** STRUCTURE scores CHoCH as a **structural fact** — "character changed here." REVERSION scores the **direction** — "therefore this reverses." Different aspects, so no double-count.

**Replace both CHoCH blocks with a single symmetric one:**
```python
                # F4: CHoCH as existence, not direction. A change of character
                # is a structural fact — STRUCTURE's job. What it MEANS
                # directionally is the reversal thesis, which belongs to
                # REVERSION. Small, symmetric credit to both sides so this
                # judge reports the fact without arguing the case.
                if choch_detected:
                    _f4 = 0.08 * weight
                    buy_score = min(buy_score + _f4, weight)
                    sell_score = min(sell_score + _f4, weight)
                    buy_exp += " +CHoCH(structure)"
                    sell_exp += " +CHoCH(structure)"
```

**Delete** the old `if choch_detected and is_bullish_regime:` and `if choch_detected and is_bearish_regime:` blocks.

**CHECK F4:** grep `is_bullish_regime` / `is_bearish_regime` near CHoCH → 0 hits.

---

# P1 — PATTERN USES THE PROVEN SPRING REFERENCE

**File:** `composite_state_builder.py`, `_compute_institutional_pattern`

**What's wrong.** PATTERN's accumulation/distribution read uses its own basing logic, while MR's spring detector uses the Livermore anchor that BRV already validated. Two different references for the same kind of event.

A prior investigation measured this exact mismatch elsewhere: **26 bars passed the anchored check, only 1 passed the independently-recomputed one.** Same class of problem.

**Wire the anchored detector in.** In `_compute_institutional_pattern`, where ACCUMULATION/DISTRIBUTION is decided, add the anchor as the reference when available:
```python
            # P1: prefer the Livermore anchor — the same level BRV validates
            # and MR's spring check uses — over an independently-derived basing
            # level. Two references for one event is how they drift apart.
            _p1_anchor_lo = getattr(state, "livermore_anchor_natural_low", None)
            _p1_anchor_hi = getattr(state, "livermore_anchor_natural_high", None)
            if _p1_anchor_lo is not None and _basing_bullish:
                _dist_supp = abs(_close - float(_p1_anchor_lo))
            if _p1_anchor_hi is not None and _basing_bearish:
                _dist_res = abs(_close - float(_p1_anchor_hi))
```
placed immediately before the `_conf` calculations that consume `_dist_supp` / `_dist_res`.

**CHECK P1:** ACCUMULATION/DISTRIBUTION still classify; confidence values shift slightly. No crash on `None` anchors (both guarded).

---

# P2 — PATTERN CAN SCORE ON A TRENDING MARKET

**File:** `council_aggregator.py`, `_judge_pattern_bidirectional`

**What's wrong.** PATTERN reads `ema_50_status` but only reacts to `DEFENDED`. The board also produces `EMA_ABOVE` and `EMA_BELOW` — the trend-riding states — and PATTERN ignores them entirely.

**Consequence:** on a trending MT5 asset with no tight range, no squeeze release, and EMA in an unread state, **PATTERN scores zero.** A whole judge silent, most of the time, on most assets.

**In the EMA-50 section, extend the read:**
```python
                # P2: the board publishes six EMA-50 states; this judge only
                # reacted to DEFENDED, so on a trending market — EMA_ABOVE or
                # EMA_BELOW, no squeeze, no tight range — PATTERN scored zero.
                # These two are genuine directional structure.
                _p2_status = (
                    getattr(_gd_composite, "ema_50_status", "UNTESTED")
                    if not isinstance(_gd_composite, dict)
                    else _gd_composite.get("ema_50_status", "UNTESTED")
                )
                if _p2_status == "EMA_ABOVE":
                    buy_score = min(buy_score + 0.30 * weight, weight)
                    buy_exp += " +EMA_ABOVE(trend-ride)"
                elif _p2_status == "EMA_BELOW":
                    sell_score = min(sell_score + 0.30 * weight, weight)
                    sell_exp += " +EMA_BELOW(trend-ride)"
```

**CHECK P2:** on a trending asset, PATTERN's score is no longer always 0.00.

---

# M1 — MOMENTUM KEEPS MEASURING MOMENTUM

**File:** `council_aggregator.py`, `_judge_momentum_bidirectional`

**What's wrong.** When ADX > 32, the Super-Cycle gate awards full weight to whichever direction the **macro regime** favours — and then `return`s immediately. So in a strong trend MOMENTUM stops measuring momentum and just echoes the 1D macro read, **returning before the RSI-divergence engine runs**.

That means the exhaustion signal — a bearish divergence at a blow-off top — is **unreachable exactly when it matters most.** It also borrows direction from another layer, which the judge design explicitly forbids.

**Replace the early return with a score-and-continue:**
```python
            if _adx_for_gate > 32:
                # M1: was an early return, which meant MOMENTUM stopped
                # measuring momentum in exactly the conditions where
                # exhaustion matters — the divergence engine below never ran.
                # Score the super-cycle read, then CONTINUE so divergence can
                # still speak against it.
                _m1_buy_base = weight if is_bull else 0.0
                _m1_sell_base = weight if not is_bull else 0.0
                buy_score = _m1_buy_base
                sell_score = _m1_sell_base
                buy_exp = (
                    f"MOM BUY: ✅ Super-Cycle ({buy_score:.2f}) ADX {_adx_for_gate:.1f}>32"
                    if is_bull else "MOM BUY: ❌ Dead in Bear Super-Cycle"
                )
                sell_exp = (
                    f"MOM SELL: ✅ Super-Cycle ({sell_score:.2f}) ADX {_adx_for_gate:.1f}>32"
                    if not is_bull else "MOM SELL: ❌ Dead in Bull Super-Cycle"
                )
                _m1_supercycle = True
            else:
                _m1_supercycle = False
```

**Then, after the divergence engine has run, allow it to cut the super-cycle score:**
```python
            # M1: divergence against a super-cycle read is the exhaustion tell
            # this judge exists to catch. Cap the cut so a single divergence
            # can't fully erase a genuine strong trend.
            if _m1_supercycle:
                if div_res.type == "BEARISH" and buy_score > 0:
                    buy_score *= 0.60
                    buy_exp += " -divergence(exhaustion)"
                elif div_res.type == "BULLISH" and sell_score > 0:
                    sell_score *= 0.60
                    sell_exp += " -divergence(exhaustion)"
```

⚠️ **Ensure the divergence block is no longer skipped.** The old code returned before reaching it; verify by line order that `div_res` is computed before this new block.

**CHECK M1:** in a strong trend with a divergence present, logs show `-divergence(exhaustion)`. Previously impossible.

---

# M2 — MACD ADDS A WEIGHTED AMOUNT

**File:** `council_aggregator.py`, `_judge_momentum_bidirectional`

**What's wrong.** The MACD confirmation adds a raw `0.2` to a score measured in units of `weight`. When `weight` is 1.0 that's 20%; when it's 0.75 it's 27%. The same evidence changes value depending on the regime, for no reason.

**Find `+ 0.2` in the MACD block and replace with `+ 0.2 * weight`.**

**CHECK M2:** grep the MACD block; every additive term is now `× weight`.

---

# X1 — WIRE THE INVALIDATION SIGNAL

**File:** `veteran_trade_manager.py`

**What's wrong.** `setup_died` and `setup_death_reason` are computed every cycle by the trajectory tracker and read by **nothing**. When a setup dies — the state flipped, the breakout failed, structure broke the other way — that's a real invalidation the exit layer should hear.

**In VTM's `_check_alert_conditions`, add to the signals list:**
```python
        # X1: the trajectory tracker declares a setup dead on real evidence
        # (Livermore state flip, failed breakout, opposing BOS). Nothing read
        # it. If the setup that justified this position has died, that is a
        # genuine signal against the position.
        if getattr(_cs, "setup_died", False):
            _death = getattr(_cs, "setup_death_reason", "unknown")
            signals_fired.append(f"setup invalidated ({_death})")
```

**Note:** this is alert-only. It contributes to the existing human-alert count; it does not close a position by itself.

**CHECK X1:** on a cycle where `[TRAJECTORY] setup DIED` logs, the VTM alert count includes it.

---

# X2 — VERIFY THE REGISTRY

**File:** `council_aggregator.py`

File 2 replaced `JUDGE_SOURCE_REGISTRY`. **Confirm it reads:**
```python
JUDGE_SOURCE_REGISTRY = {
    "trend":     [("independent", 0.80), ("livermore_1h", 0.10), ("brc", 0.10)],
    "momentum":  [("independent", 0.75), ("livermore_1h", 0.15), ("mr_features", 0.10)],
    "pattern":   [("independent", 0.85), ("spring", 0.15)],
    "structure": [("independent", 0.85), ("livermore_1h", 0.15)],
    "volume":    [("independent", 0.85), ("livermore_1h", 0.15)],
    "reversion": [("independent", 0.75), ("spring", 0.15), ("brc", 0.10)],
}
```
If File 2 was applied correctly this is already true. **Verification step only — no edit.**

---

## VERIFICATION RECORD

### PASS 1 — FORWARD

- **F1** makes STRUCTURE directionally honest; the board already carries the fields. ✓
- **F2** rewards a break at proven structure — correct instinct for a location judge; creates headroom by dropping the bare break to 0.85. ✓
- **F3** stops the number F2 depends on being overwritten. **F2 without F3 would be actively harmful** — they must ship together. ✓
- **F4** hands the reversal thesis to REVERSION and keeps the structural fact with STRUCTURE. No double-count. ✓
- **P1/P2** raise PATTERN from "often silent" to "reads real structure." ✓
- **M1** restores exhaustion detection where it matters; **M2** makes MACD regime-independent. ✓
- **X1** connects a computed-but-dark invalidation signal. ✓

### PASS 2 — BACKWARD (against the repo)

- **F1 confirmed live:** STRUCTURE's BUY branch reads `bos_detected`; the trajectory tracker in `composite_state_builder.py` reads `bos_bullish`/`bos_bearish`. Same repo, two standards. ✓
- **F2 confirmed:** the BOS branch returns flat `weight`; `elif level_defended` is unreachable when BOS fires. ✓
- **F3 confirmed by the code's own docstring** — four calls, shared fields, last wins. The 1D-200 call is last in `_build_composite_state`'s sequence. ✓
- **F4 confirmed:** two regime-conditional CHoCH blocks exist, `+0.3 * weight` each. ✓
- **P2 confirmed:** `ema_50_status` supports EMA_ABOVE/EMA_BELOW in the builder; the judge only branches on DEFENDED. ✓
- **M1 confirmed:** `if _adx_for_gate > 32:` returns immediately, before `self.divergence_detector.analyze(df)`. ✓
- **X1 confirmed:** `setup_died` / `setup_death_reason` are written in the trajectory block; grep finds no consumer. ✓
- **Ladder untouched:** no fix here reads or writes `zone_*` fields. No interaction. ✓
- **BRC untouched:** no fix here reads `brc_*`. File 1's work is independent. ✓

---

## BUILD CHECKLIST

- [ ] F1 — directional BOS both sides; grep `bos_detected` in STRUCTURE = 0
- [ ] F2 — defense folded into BOS base, both sides
- [ ] F3 — `_update_ma_defense` parameterised; 8 new fields; unsuffixed canonical write preserved
- [ ] F4 — CHoCH symmetric existence credit; regime-conditional blocks deleted
- [ ] P1 — anchored reference in `_compute_institutional_pattern`
- [ ] P2 — EMA_ABOVE / EMA_BELOW scored
- [ ] M1 — Super-Cycle no longer early-returns; divergence can cut it
- [ ] M2 — MACD `× weight`
- [ ] X1 — `setup_died` wired into VTM alerts
- [ ] X2 — registry verified (no edit)
- [ ] Bot starts clean; scorecards show non-zero PATTERN on trending assets
- [ ] Report exact line numbers back to Desire

⚠️ **F2 and F3 must ship in the same commit.** F2 increases STRUCTURE's dependence on `defense_strength`; F3 is what makes that number real.
