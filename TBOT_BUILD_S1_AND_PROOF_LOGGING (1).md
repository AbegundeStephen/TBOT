# BUILD — LOGGING REPAIR + S1 STRUCTURE + PROOF STANDARD

**Owner:** Desire · **Developer:** Stephen · **Analyst:** Claude
**Date:** 29 Jul 2026 · **Version 3 — SHIPS HOT, no flags**
**Supersedes:** all earlier drafts. Work from this file only.
**File touched:** `src/execution/composite_state_builder.py` — that file only

---

## READ THIS FIRST

**Nine edits. One commit. One file. No flags — everything takes effect on restart.**

- **PART A — logging repair.** Four edits. Changes only what gets written to the log.
- **PART B — S1 structure fix.** Two edits. Changes trading behaviour.
- **PART C — proof standard.** Three edits. Changes trading behaviour.

**Nine edits is a lot for one commit. Apply them in order, tick each one off, and do not batch them from memory.** Several are near-identical in shape and easy to half-apply.

**There is no flag.** Tag the commit before you deploy — see BEFORE YOU DEPLOY. The revert plan is git.

**Expect fewer proofs after this, not more.** That is the intended result, explained in Part C. Do not treat a drop as a bug unless it goes to zero.

**Do not fix anything else while you are in this file.** If you spot something, write it down and tell Desire.

---

## WHAT THIS BUILD FIXES

Three problems that compound each other.

**1. The instruments are lying.** Three log lines are at DEBUG while the live logger runs at INFO — they write nothing. One more over-counts by roughly 23×.

**2. Uptrends are recorded as reversals.** The detector never checks which way the trend is going, so a healthy uptrend (higher high + higher low) sets both a break flag and a reversal flag, and the reversal overwrites the break. Measured: 3,490 of 5,874 GOLD candles.

**3. The proof standard has never actually been enforced.** The retest window looks back 8 candles with no requirement that the touch came *after* the break. Median setup age at death is 1 bar. **So the entire window predates the setup, by arithmetic.** Measured across five assets: roughly 90% of retest evidence draws on pre-break price action, and there are **zero clean continuation proofs** on GOLD and BTC.

Part C is what makes `break → retest → close` mean what it says.

---

# PART A — REPAIR THE LOGGING

## EDIT A1 — add the dedup store

**Location:** `__init__`, immediately after `self._brc_memory = {}`

**Add:**

```python
        # Once-per-candle dedup for the observation logs. This builder runs
        # ~23x per 1H candle (twice per 5-min cycle), so an ungated log line
        # fires ~23 times for one real event — the same cycles-vs-candles trap
        # brc_age was built to avoid. Keyed (asset, tag).
        self._brc_log_ts = {}
        # Part C: break-anchored retest memory. Keyed on the REFERENCE LEVEL,
        # not the setup object — the setup churns (median life 1 bar) but the
        # level it broke does not. {asset: {"ref", "last_ts", "bars"}}
        self._brc_break_ts = {}
```

---

## EDIT A2 — hoist the bar timestamp

`_bar_ts` is currently computed **inside** the `if _retested and _closed_through:` branch. Parts A and C both need it earlier.

**Find, inside the confirmation branch:**

```python
                        try:
                            _bar_ts = (
                                df["timestamp"].iloc[-1] if "timestamp" in df.columns
                                else df.index[-1]
                            )
                        except Exception:
                            _bar_ts = None
```

**Cut it. Paste it immediately AFTER this line:**

```python
                    _brc_close = float(df["close"].iloc[-1])
```

**That placement matters** — it must be above the window slicing, above the MEASURE-8.7 block, and above `if _retested and _closed_through:`. Everything downstream depends on it.

**Watch the indentation** — it moves out one level.

---

## EDIT A3 — the four log calls

### A3a — PROOF-DISTINCT

**Find:**

```python
                        if _age == 0:
                            logger.info(
                                "[PROOF-DISTINCT] %s: %s dir=%+d ref=%.5g close=%.5g ts=%s — new proof.",
                                self.asset_type, _brc_kind, _brc_dir, _brc_ref, _brc_close, _bar_ts,
                            )
```

**Replace with:**

```python
                        if _age == 0:
                            # Once per closed candle: age stays 0 for every
                            # repeat call within the same candle, so an ungated
                            # line logs one proof ~23 times live.
                            _k = (self.asset_type, "DISTINCT")
                            if _bar_ts is not None and self._brc_log_ts.get(_k) != _bar_ts:
                                self._brc_log_ts[_k] = _bar_ts
                                logger.info(
                                    "[PROOF-DISTINCT] %s: %s dir=%+d ref=%.5g close=%.5g ts=%s — new proof.",
                                    self.asset_type, _brc_kind, _brc_dir, _brc_ref, _brc_close, _bar_ts,
                                )
```

### A3b — PROOF-REPEAT

**Find** (note `logger.debug`):

```python
                        else:
                            logger.debug(
                                "[PROOF-REPEAT] %s: %s dir=%+d ref=%.5g age=%d — "
                                "re-confirmation of an already-counted proof.",
                                self.asset_type, _brc_kind, _brc_dir, _brc_ref, _age,
                            )
```

**Replace with:**

```python
                        else:
                            _k = (self.asset_type, "REPEAT")
                            if _bar_ts is not None and self._brc_log_ts.get(_k) != _bar_ts:
                                self._brc_log_ts[_k] = _bar_ts
                                logger.info(
                                    "[PROOF-REPEAT] %s: %s dir=%+d ref=%.5g age=%d — "
                                    "re-confirmation of an already-counted proof.",
                                    self.asset_type, _brc_kind, _brc_dir, _brc_ref, _age,
                                )
```

### A3c — PROOF-NEAR-MISS

**Find** (note `logger.debug`):

```python
                            logger.debug(
                                "[PROOF-NEAR-MISS] %s: %s dir=%+d retested but no "
                                "close-through — close=%.5g ref=%.5g gap=%.5g (%.3f%%).",
                                self.asset_type, _brc_kind, _brc_dir,
                                _brc_close, _brc_ref, _gap, _gap_pct,
                            )
```

**Replace with:**

```python
                            _k = (self.asset_type, "NEARMISS")
                            if _bar_ts is not None and self._brc_log_ts.get(_k) != _bar_ts:
                                self._brc_log_ts[_k] = _bar_ts
                                logger.info(
                                    "[PROOF-NEAR-MISS] %s: %s dir=%+d retested but no "
                                    "close-through — close=%.5g ref=%.5g gap=%.5g (%.3f%%).",
                                    self.asset_type, _brc_kind, _brc_dir,
                                    _brc_close, _brc_ref, _gap, _gap_pct,
                                )
```

### A3d — MEASURE-8.7 — replaced entirely by Part C's version

**This one is also at DEBUG and also writes nothing live.** It is the measurement that found the contamination, so it matters most.

It is **rewritten in EDIT C3** rather than patched here, because its anchor changes from `setup_age` to the new break memory. **Do not edit it separately — C3 replaces the whole block.**

---

# PART B — S1: TREND-CONTEXTUAL STRUCTURE

## Why — in plain English

The bot decides *"is this a reversal?"* without ever checking which way the trend is going.

In an uptrend you get **higher highs** and **higher lows**. The code reads the higher high as a break of structure — correct. It reads the higher low as a **reversal warning** — wrong, that's also just the uptrend. Then the setup-birth logic lets the reversal label overwrite the break label.

**So a clean uptrend gets recorded as a reversal.**

### The evidence

| Market shape | What fires today | Result |
|---|---|---|
| **Uptrend** (higher high + higher low) | bos_bullish + choch_bullish | filed as reversal |
| **Downtrend** (lower high + lower low) | choch_bearish + bos_bearish | filed as reversal |
| **Contracting range** | two CHoCH flags | reversal |
| **Expanding range** | two BOS flags | continuation |

**3,490 dual-signal candles + 683 = 4,173 reversal births.** Exact match to the measured counts.

It also explains the **zero direction flips** across 3,490 dual-signal candles on all five assets. In a trend both flags derive from the same directional move — they cannot disagree. Structurally impossible, not a clean sample.

### The rule being restored

> **A break of structure CONTINUES the prevailing trend. A change of character OPPOSES it.**

---

## EDIT B1 — the detector

**Location:** `_update_structure()`, around line 1562

### Find

```python
            # ── CHoCH / BOS classification — both directions independent ────
            if len(swing_highs) >= 2:
                if swing_highs[0] > swing_highs[1]:
                    state.bos_detected = True    # Higher high — trend continuing
                    state.bos_bullish = True
                elif swing_highs[0] < swing_highs[1]:
                    state.choch_detected = True  # Lower high — reversal warning
                    state.choch_bearish = True

            if len(swing_lows) >= 2:
                if swing_lows[0] < swing_lows[1]:
                    state.bos_detected = True    # Lower low — downtrend continuing
                    state.bos_bearish = True
                elif swing_lows[0] > swing_lows[1]:
                    state.choch_detected = True  # Higher low — reversal warning
                    state.choch_bullish = True
```

### Replace with

```python
            # ── CHoCH / BOS classification ──────────────────────────────────
            # S1: a break of structure CONTINUES the prevailing trend; a change
            # of character OPPOSES it. The old code made both calls without ever
            # asking what the trend was, so a healthy uptrend (higher high +
            # higher low) set bos_bullish AND choch_bullish on the same candle —
            # and the birth logic downstream lets CHoCH overwrite BOS. Every
            # trending candle was therefore recorded as a reversal. Measured:
            # 3,490 of 5,874 GOLD candles, and identically on four other assets.
            _hh = len(swing_highs) >= 2 and swing_highs[0] > swing_highs[1]
            _lh = len(swing_highs) >= 2 and swing_highs[0] < swing_highs[1]
            _ll = len(swing_lows)  >= 2 and swing_lows[0]  < swing_lows[1]
            _hl = len(swing_lows)  >= 2 and swing_lows[0]  > swing_lows[1]

            # Trend source: the Livermore 1H MACHINE, not
            # state.livermore_state_1h. This is not a preference — that field
            # is written at line ~661 and this method runs at line ~297, so it
            # is still None here. Reading it would return None on every call
            # and silently disable the entire fix.
            _UP   = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
            _DOWN = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
            _lsm_state = None
            try:
                if self._livermore_1h is not None:
                    _lsm_state = self._livermore_1h.snapshot().state
            except Exception:
                _lsm_state = None

            if _lsm_state in _UP:
                _trend = 1
            elif _lsm_state in _DOWN:
                _trend = -1
            else:
                # Fallback for warm-up: the swings describe the trend
                # themselves. Both feet stepping up = uptrend.
                _trend = 1 if (_hh and _hl) else (-1 if (_lh and _ll) else 0)

            if _trend == 1:
                # Uptrend. A higher high continues it. A lower high or a lower
                # low opposes it — that is the change of character.
                if _hh:
                    state.bos_detected = True
                    state.bos_bullish = True
                if _lh or _ll:
                    state.choch_detected = True
                    state.choch_bearish = True
            elif _trend == -1:
                # Downtrend. A lower low continues it.
                if _ll:
                    state.bos_detected = True
                    state.bos_bearish = True
                if _hh or _hl:
                    state.choch_detected = True
                    state.choch_bullish = True
            # _trend == 0: no established trend, so no directional structural
            # call. A range's swings are noise, not a break and not a reversal.

            # Once per closed candle — this method runs ~23x per candle.
            try:
                _s1_ts = (
                    df["timestamp"].iloc[-1] if "timestamp" in df.columns
                    else df.index[-1]
                )
            except Exception:
                _s1_ts = None
            _k = (self.asset_type, "S1")
            if _s1_ts is not None and self._brc_log_ts.get(_k) != _s1_ts:
                self._brc_log_ts[_k] = _s1_ts
                logger.info(
                    "[S1-STRUCTURE] %s: trend=%+d (lsm=%s) hh=%s lh=%s ll=%s hl=%s "
                    "-> bos_bull=%s bos_bear=%s choch_bull=%s choch_bear=%s",
                    self.asset_type, _trend, _lsm_state, _hh, _lh, _ll, _hl,
                    state.bos_bullish, state.bos_bearish,
                    state.choch_bullish, state.choch_bearish,
                )
```

**A1 must be applied before this** — B1 uses `self._brc_log_ts`. Without it this raises `AttributeError`, which the surrounding try/except swallows silently, and structure detection quietly stops working.

---

## EDIT B2 — the death condition

**Location:** trajectory block, STEP 2 death check, **lines 1066–1070**

**Why not optional:** with S1, inside an uptrend a lower low surfaces as `choch_bearish` instead of `bos_bearish`. Without this edit a long setup survives price breaking the prior swing low — the exact event that invalidates it.

### Find

```python
                    if _death_reason is None:
                        if _dir == 1 and getattr(state, "bos_bearish", False):
                            _death_reason = "OPPOSING_BOS"
                        elif _dir == -1 and getattr(state, "bos_bullish", False):
                            _death_reason = "OPPOSING_BOS"
```

### Replace with

```python
                    if _death_reason is None:
                        # S1: inside a trend, a break against the setup now
                        # surfaces as the opposing CHoCH rather than an opposing
                        # BOS. Without this the setup outlives its own
                        # invalidation. Separate reason string so the two cases
                        # stay distinguishable. Verified by grep: those were the
                        # only two occurrences of "OPPOSING_BOS" in the repo.
                        if _dir == 1:
                            if getattr(state, "bos_bearish", False):
                                _death_reason = "OPPOSING_BOS"
                            elif getattr(state, "choch_bearish", False):
                                _death_reason = "OPPOSING_CHOCH"
                        elif _dir == -1:
                            if getattr(state, "bos_bullish", False):
                                _death_reason = "OPPOSING_BOS"
                            elif getattr(state, "choch_bullish", False):
                                _death_reason = "OPPOSING_CHOCH"
```

---

# PART C — MAKE THE PROOF STANDARD REAL

## Why — in plain English

The standard is **break → retest → close**. The code has never enforced the ordering.

The retest check asks *"did price touch this level anywhere in the last 8 candles?"* — with **no requirement that the touch came after the break.**

**Trading picture.** BTC chops around 64,140 all day. Dozens of touches, no direction. Then it finally breaks upward. The bot looks back 8 candles, sees the chop, and calls it a retest. **There was no retest.** That was pre-break noise.

**And it is worse than occasional — it is arithmetic.** Median setup age at death is **1 bar**. The window looks back **8 bars**. So the entire window predates the setup, always. Measured across five assets: ~90% of retest evidence is pre-break, and **zero clean continuation proofs exist on GOLD or BTC.**

## Two changes, and they only work together

**1. Anchor the retest to the break.** Only count touches that happened *after* the level was broken.

**2. Widen the window from 8 candles to 28.** A continuation measures against a **4-hour** level but was given only 8 one-hour candles — two 4H bars — to prove itself. 28 hourly candles is seven 4H bars: the same room, in the level's own units. Desire has set both continuation and reversal to 28.

**Widening without anchoring would be actively dangerous** — 28 candles of pre-break noise to draw false retests from. Anchoring without widening leaves too little room for a 4H level to be retested. Ship both.

## The key design decision

**The break memory is keyed on the REFERENCE LEVEL, not the setup object.**

The setup churns — born and killed every bar or two. The level does not. GOLD broke 3,340 once; the tracker's bookkeeping may reset six times before price returns, but 3,340 was still broken at hour one.

Anchoring to `setup_age` would give a 1-bar window and silence the bot. Anchoring to the level is what makes this work.

---

## EDIT C1 — the data guard

**Find:**

```python
            if _brc_active and _brc_dir != 0 and df is not None and len(df) >= 9:
```

**Replace with:**

```python
            # Part C: window widened 8 -> 28 candles, so the data floor moves
            # with it. Miss this and the whole block silently never runs.
            if _brc_active and _brc_dir != 0 and df is not None and len(df) >= 29:
```

---

## EDIT C2 — break memory and the ordered retest

**Find:**

```python
                if _brc_ref is not None and float(_brc_ref) > 0:
                    _brc_ref = float(_brc_ref)
                    _brc_close = float(df["close"].iloc[-1])
                    _brc_win_high = df["high"].iloc[-9:-1].values
                    _brc_win_low  = df["low"].iloc[-9:-1].values

                    if _brc_dir == 1:
                        _retested = any(l <= _brc_ref for l in _brc_win_low)
                        _closed_through = _brc_close > _brc_ref
                    else:
                        _retested = any(h >= _brc_ref for h in _brc_win_high)
                        _closed_through = _brc_close < _brc_ref
```

**Replace with** (note `_bar_ts` is computed here by Edit A2):

```python
                if _brc_ref is not None and float(_brc_ref) > 0:
                    _brc_ref = float(_brc_ref)
                    _brc_close = float(df["close"].iloc[-1])

                    # ── A2 places the _bar_ts computation here ──

                    # ── Part C: break-anchored retest ordering ──────────────
                    # Keyed on the reference LEVEL, not the setup object. The
                    # setup churns (median life 1 bar); the level it broke does
                    # not. A new reference means a new break — start the clock.
                    # The same reference means the same break still standing,
                    # however many times the tracker reset in between.
                    _bt = self._brc_break_ts.get(self.asset_type)
                    if _bt is None or _bt.get("ref") != _brc_ref:
                        self._brc_break_ts[self.asset_type] = {
                            "ref": _brc_ref, "last_ts": _bar_ts, "bars": 0
                        }
                    elif _bar_ts is not None and _bar_ts != _bt.get("last_ts"):
                        # A new candle closed — the break is one bar older.
                        _bt["bars"] = int(_bt.get("bars", 0)) + 1
                        _bt["last_ts"] = _bar_ts
                    _bars_since_break = int(
                        self._brc_break_ts[self.asset_type].get("bars", 0)
                    )

                    # Window widened 8 -> 28 candles (seven 4H bars' worth of
                    # hourly candles) so a 4H level has room to be retested.
                    _WIN = 28
                    _brc_win_high = df["high"].iloc[-(_WIN + 1):-1].values
                    _brc_win_low  = df["low"].iloc[-(_WIN + 1):-1].values

                    # Window position i maps to (WIN - i) bars ago: i=0 is the
                    # oldest bar in the window, i=WIN-1 is one bar back. A touch
                    # is POST-BREAK only when it is more recent than the break
                    # itself — strictly, so the break candle's own wick never
                    # counts as its own retest.
                    if _brc_dir == 1:
                        _retested = any(
                            (v <= _brc_ref) and ((_WIN - i) < _bars_since_break)
                            for i, v in enumerate(_brc_win_low)
                        )
                        _closed_through = _brc_close > _brc_ref
                    else:
                        _retested = any(
                            (v >= _brc_ref) and ((_WIN - i) < _bars_since_break)
                            for i, v in enumerate(_brc_win_high)
                        )
                        _closed_through = _brc_close < _brc_ref
```

---

## EDIT C3 — rewrite MEASURE-8.7 as a regression check

The old block anchored on `setup_age` and logged at DEBUG. Both are now wrong: the anchor is the break, and DEBUG writes nothing live.

**Find the whole existing block** — it starts with the comment `# Measurement 8.7: window index i (0..7) is "9-i bars ago"` and ends with the `logger.debug("[MEASURE-8.7-PRE-BIRTH-RETEST] ...` call including its closing bracket.

**Replace the entire block with:**

```python
                    # MEASURE-8.7 — now a regression check on the ordering fix
                    # above rather than a survey. Counts every touch of the
                    # reference in the window and splits pre-break from
                    # post-break. Before this build ~90% of touches were
                    # pre-break and all of them counted; now only post-break
                    # touches can satisfy _retested. pre_break should stay high
                    # (the market really does touch these levels beforehand)
                    # while retested=True should now only ever appear alongside
                    # post_break > 0. If it does not, the filter is not working.
                    _touch_src = _brc_win_low if _brc_dir == 1 else _brc_win_high
                    _touch_idxs = [
                        i for i, v in enumerate(_touch_src)
                        if (v <= _brc_ref if _brc_dir == 1 else v >= _brc_ref)
                    ]
                    if _touch_idxs:
                        _pre_break  = [i for i in _touch_idxs if (_WIN - i) >= _bars_since_break]
                        _post_break = [i for i in _touch_idxs if (_WIN - i) <  _bars_since_break]
                        _k = (self.asset_type, "ORDERING")
                        if _bar_ts is not None and self._brc_log_ts.get(_k) != _bar_ts:
                            self._brc_log_ts[_k] = _bar_ts
                            logger.info(
                                "[MEASURE-8.7-ORDERING] %s: kind=%s dir=%+d "
                                "bars_since_break=%d touches=%d pre_break=%d "
                                "post_break=%d retested=%s",
                                self.asset_type, _brc_kind, _brc_dir,
                                _bars_since_break, len(_touch_idxs),
                                len(_pre_break), len(_post_break), _retested,
                            )
```

---

# VERIFICATION — TWO PASSES

## PASS 1 — FORWARD (reading the plan into the code)

- **Part A** touches only log level and frequency. No value the bot reads is altered. ✓
- Dedup keys on the bar timestamp — the same value `brc_age` already uses to separate candles from cycles. ✓
- A2's placement puts `_bar_ts` above the window slicing, the 8.7 block and the confirmation branch. All three need it. ✓
- **B1**, uptrend, higher high + higher low → `bos_bullish` only. The overwrite can no longer happen. ✓
- **B1**, downtrend, lower low → `bos_bearish` only. Mirror written explicitly, not assumed. ✓
- **B2** catches structure breaking against a setup under either label. ✓
- **C2** keys the break on the reference level, so it survives the setup churn that would make a setup-anchored version useless. ✓
- **C2** uses strict `<`, so the break candle's own wick can never count as its own retest. ✓
- **C1** moves the data floor with the window. Without it the block silently never runs. ✓
- **C3** logs pre- and post-break counts alongside `retested`, so the filter can be seen working rather than assumed. ✓

## PASS 2 — BACKWARD (tracing from code back to the evidence)

- **`_update_structure` is called at line 297; `state.livermore_state_1h` is written at line 661.** Reading the state field returns `None` every time — reading the machine is mandatory. **Confirmed by grep.** ✓
- **`OPPOSING_BOS` appears exactly twice in the repo**, lines 1068 and 1070, both replaced by B2. **Confirmed by grep.** ✓
- **`_bar_ts` currently sits inside the confirmation branch** — confirmed in the repo, which is why A2 is required before Parts A and C can work. ✓
- **Median setup age at death is 1 bar; the window looked back 8.** Every window position predated the setup by arithmetic — which is why ~90% pre-break contamination was measured, and why anchoring to `setup_age` would not work. ✓
- **Zero clean TF_CONT proofs on GOLD and BTC**, at most 1 on the other three, out of 145–204 distinct proofs each. There is currently no clean evidence that a genuine continuation proof pays — because one has barely ever occurred. ✓
- Arithmetic check on S1: **3,490 + 683 = 4,173** reversal births, exactly as measured. ✓

---

# BEFORE YOU DEPLOY

- [ ] **Tag the current commit.** `git tag pre-s1 && git push --tags`. This is the revert point — there is no flag.
- [ ] Confirm all nine edits are in `composite_state_builder.py` and nothing else was touched: `git diff --stat`.
- [ ] **Confirm A1 was applied before B1 and C2** — both use dicts A1 creates. Missing it fails silently.
- [ ] Note the restart time, for separating before/after in the log.

---

# FIRST HOUR AFTER RESTART

- [ ] Bot starts clean, no exceptions, cycles completing.
- [ ] **`[S1-STRUCTURE]` prints for every asset**, once per candle, with a sane `trend` and `lsm`. If `trend=0` everywhere, Livermore is not warmed — say so immediately.
- [ ] **No candle sets `bos_bullish` and `choch_bullish` together.**
- [ ] **`[PROOF-NEAR-MISS]` and `[MEASURE-8.7-ORDERING]` appear at INFO.** Both were completely absent before.
- [ ] Each tag appears **at most once per asset per hour**, not ~23 times.
- [ ] **`[MEASURE-8.7-ORDERING]` shows `bars_since_break` climbing** on at least one asset. If it is stuck at 0 everywhere, the reference is changing every candle and C2 needs a look.

# FIRST 24 HOURS

- [ ] **`[TRAJECTORY]` shows `TF_CONT` on trending assets.** Today it is almost all `MR_REV`. **Headline check — if this does not change, S1 did not land.**
- [ ] **Median setup age rises above 1 bar**, and `OPPOSING_BOS` falls well below its current 55% share of deaths. **This is a stated prediction. If it does not happen, stop and tell Desire — the model behind this build is wrong.**
- [ ] `[MEASURE-8.7-ORDERING]` shows `retested=True` only alongside `post_break > 0`.
- [ ] **Proof count drops sharply. This is expected** — see below.
- [ ] Report exact line numbers for all nine edits back to Desire.

---

# EXPECTED EFFECTS — none of these are bugs

1. **`[TRAJECTORY]` starts showing `TF_CONT`** on trending assets.
2. **`+CHoCH(structure)` mostly disappears** from scorecards. STRUCTURE loses that 0.08 on most candles — correct, it was being paid for an uptrend.
3. **MR Mode 2 fires even less.** Its `choch_detected` gate was permanently open in any trending market; it becomes a real filter. **MR was already near-silent and this tightens it further.**
4. **`OPPOSING_CHOCH` appears** in death logs.
5. **Proofs drop, possibly sharply.** ~90% of current proofs are contaminated. Removing them leaves few; the wider window should add some back. **A large fall is the fix working.** A fall to exactly zero across all assets for a full day is not — that is the revert trigger below.

---

# REVERT

No flag. To back out:

```
git revert <commit-sha>
```
then restart. Or reset to the `pre-s1` tag.

**Revert triggers — any one, stop and tell Desire:**

- Exceptions in `_update_structure`, the trajectory block, or the BRC block
- `trend=0` on every asset for more than one full cycle after warm-up
- **Zero proofs on every asset for a full day** — the ordering filter is too strict
- `bars_since_break` stuck at 0 everywhere
- Setup deaths running away — every setup dying every candle
- Anything in the logs you cannot explain

**Do not diagnose a live problem by editing more code. Revert first, investigate after.**

---

# NOT IN THIS BUILD

- **GOLD's inverted RSI zones** — confirmed hardcoded at `council_aggregator.py` 240–241, the exact mirror of every other asset, no explanatory comment. Ships separately for attribution.
- **TF's 0.3 counter-trend penalty** against a code default of 2.0. Needs a decision.
- **The two-lane setup tracker.** On hold — 94.8% of blocked births were same-direction and harmless, so the case is much weaker than first thought.
- **MR's compression veto and the Mode 1 / Mode 2 dispatch mismatch.** Untouched. MR stays silent.
- **Flag ratification and EURUSD enabled in live config.** Desire's decisions.

---

# AFTER THIS RUNS — RE-MEASURE EVERYTHING

**Every number in the last report was taken with both bugs live.** Once this has a few days of data, all eight Section 8 measurements need re-running: the labels have changed, the proof standard has changed, and the setup lifespan should have changed.

**Two questions that could not be answered before and should be answerable now:**

- **Do clean continuation proofs exist at all?** There were zero on GOLD and BTC. If there are still zero after this, the standard cannot be met and that is a strategy finding, not a bug.
- **Does TF's own signal agree with the proof direction?** The existing suppression line already prints all four gate values, so grep for `brc_confirmed=True brc_kind=TF_CONT` — every hit is a direction block.
