# BUILD — PROOF LOGGING REPAIR + S1 TREND-CONTEXTUAL STRUCTURE

**Owner:** Desire · **Developer:** Stephen · **Analyst:** Claude
**Date:** 29 Jul 2026 · **Version 2 — SHIPS HOT, no flag**
**Supersedes:** all earlier S1 drafts. Work from this file only.
**File touched:** `src/execution/composite_state_builder.py` — that file only

---

## READ THIS FIRST

**Six edits. One commit. One file. No flags — everything takes effect on restart.**

- **PART A — logging repair.** Three edits. Changes only what gets written to the log, never what the bot does.
- **PART B — S1 structure fix.** Three edits. **This changes trading behaviour immediately.**

**Desire has decided this ships hot.** There is no flag and no inert deploy. The moment you restart, the bot is running the new structure logic.

**That means the revert plan is git, not a config toggle.** Tag the commit before you deploy. See the REVERT section at the bottom — read it before you start, not after.

**Do not fix anything else while you are in this file.** If you spot something, write it down and tell Desire.

---

## WHAT THIS BUILD ACTUALLY DOES

Right now the bot **cannot** trade. Not "rarely" — cannot. Two of six judges score zero on every cycle, so the remaining four cap at 3.5 against thresholds of 3.65–4.35. Arithmetically impossible.

S1 restores continuation labelling, which lets TF fire, which lets TREND score up to 1.5, which lifts the ceiling to 5.0.

**So this build converts a bot that cannot trade into one that can.** Expect trades. That is the point of it.

---

# PART A — REPAIR THE PROOF LOGGING

## Why

**1. Two log lines are at DEBUG and the live logger runs at INFO.** `PROOF-NEAR-MISS` and `PROOF-REPEAT` are writing **nothing at all** — there is not one DEBUG line anywhere in the live log. Near-miss is the input to the close-through question.

**2. `PROOF-DISTINCT` over-counts by roughly 23×.** This file runs about 23 times per 1H candle — twice per 5-minute cycle. `PROOF-DISTINCT` fires on `_age == 0`, and the aging logic deliberately holds age at 0 for every repeat call within the same candle.

That second one is the same cycles-versus-candles trap `brc_age` was built to avoid, reappearing inside the tool built to measure it. In backtest the count was right (one build per candle). Live it is not.

**Note:** `SETUP-*` lines are unaffected — they sit inside the trajectory block, which is already candle-guarded.

---

## EDIT A1 — add the dedup store

**Location:** `__init__`, immediately after `self._brc_memory = {}`

**Add:**

```python
        # Once-per-candle dedup for the PROOF-* and S1-STRUCTURE observation
        # logs. This builder runs ~23x per 1H candle (twice per 5-min cycle),
        # so an ungated log line fires ~23 times for one real event — the same
        # cycles-vs-candles trap brc_age was built to avoid. Keyed (asset, tag).
        self._brc_log_ts = {}
```

---

## EDIT A2 — compute the bar timestamp before the branch

**Location:** the BRC block. `_bar_ts` is currently computed **inside** the `if _retested and _closed_through:` branch, so the near-miss path on the other side has no timestamp to dedup against.

**Find this, inside the confirmation branch:**

```python
                        try:
                            _bar_ts = (
                                df["timestamp"].iloc[-1] if "timestamp" in df.columns
                                else df.index[-1]
                            )
                        except Exception:
                            _bar_ts = None
```

**Cut it, and paste it immediately BEFORE this line:**

```python
                    if _retested and _closed_through:
```

so `_bar_ts` is available to both branches. Everything else in the confirmation branch stays exactly as it is.

**Watch the indentation** — it moves out one level.

---

## EDIT A3 — the three log calls

All three: fire at INFO, once per closed candle.

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

---

# PART B — S1: TREND-CONTEXTUAL STRUCTURE

**No flag. Live on restart.**

## Why — in plain English

The bot decides *"is this a reversal?"* without ever checking which way the trend is going.

In an uptrend you get **higher highs** and **higher lows**. The code reads the higher high as a break of structure — correct. It reads the higher low as a **reversal warning** — wrong, that's also just the uptrend. Then the setup-birth logic lets the reversal label overwrite the break label.

**So a clean uptrend gets recorded as a reversal.**

### The trading picture

GOLD trending up. Swing highs 3,300 → 3,340 (higher high). Swing lows 3,280 → 3,320 (higher low). Textbook uptrend, both feet stepping upward. **The bot records: reversal, long.**

### The evidence

| Market shape | What fires today | Result |
|---|---|---|
| **Uptrend** (higher high + higher low) | bos_bullish + choch_bullish | dual signal → filed as reversal |
| **Downtrend** (lower high + lower low) | choch_bearish + bos_bearish | dual signal → filed as reversal |
| **Contracting range** | two CHoCH flags | reversal |
| **Expanding range** | two BOS flags | continuation |

**3,490 dual-signal candles + 683 = 4,173 reversal births.** Exact match to the measured GOLD backtest counts.

It also explains the **zero direction flips** across 3,490 dual-signal candles. In a trend both flags derive from the same directional move — they cannot disagree. That was not a clean sample; it was structurally impossible.

### The rule being restored

> **A break of structure CONTINUES the prevailing trend. A change of character OPPOSES it.**

You cannot tell them apart without knowing the trend. Right now neither comparison knows the trend at all.

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
            # 3,490 of 5,874 GOLD candles.
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

**Note:** this uses `self._brc_log_ts` from Edit A1. **A1 must be in place first** or this raises `AttributeError` on the first call.

---

## EDIT B2 — the death condition

**Location:** the trajectory block, STEP 2 evidence-based death check, **lines 1066–1070**

### Why this is not optional

Today a long setup dies when `bos_bearish` fires. With S1, inside an uptrend a lower low now surfaces as `choch_bearish` instead.

**Without this edit a long setup would survive price breaking the prior swing low** — the exact event that invalidates it. The setup would outlive its own thesis.

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
                        # stay distinguishable, and so nothing matching on
                        # "OPPOSING_BOS" changes meaning. Verified by grep:
                        # those were the only two occurrences in the repo.
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

# VERIFICATION — TWO PASSES

## PASS 1 — FORWARD (reading the plan into the code)

- **Part A** touches only log level and log frequency. No value the bot reads is altered. ✓
- Dedup keys on the bar timestamp — the same value `brc_age` already uses to distinguish candles from cycles. ✓
- Hoisting `_bar_ts` makes it available to the near-miss branch, which previously had no timestamp. ✓
- **B1**, uptrend, higher high + higher low → `bos_bullish` only. **The overwrite can no longer happen.** ✓
- **B1**, downtrend, lower low → `bos_bearish` only. Mirror written explicitly, not assumed. ✓
- Livermore unwarmed → swing-derived fallback still separates trends from ranges. ✓
- **B2** now catches structure breaking against a setup under either label. ✓
- A1 must precede B1 — B1 uses `self._brc_log_ts`. Ordering noted in both edits. ✓

## PASS 2 — BACKWARD (tracing from code back to the evidence)

- **`_update_structure` is called at line 297; `state.livermore_state_1h` is written at line 661.** Reading the state field would return `None` every time. Reading the machine is mandatory. **Confirmed by grep.** ✓
- **`OPPOSING_BOS` appears exactly twice in the repo** — lines 1068 and 1070, both replaced by B2. Nothing downstream matches. **Confirmed by grep.** ✓
- `self._livermore_1h` exists on the builder and exposes `.snapshot().state` — the same access the trajectory block already uses. ✓
- The six state names match the `_UP`/`_DOWN` sets already defined in the trajectory block in this file. ✓
- Arithmetic check: the four chart shapes explain the measured counts exactly — **3,490 + 683 = 4,173**. Confirmed by real data, not assumed. ✓
- `PROOF-NEAR-MISS` fired 3,153 times in backtest = 2,017 + 1,136 from the report exactly, so the logger is measuring the right population. ✓

---

# BEFORE YOU DEPLOY

- [ ] **Tag the current commit.** `git tag pre-s1 && git push --tags`. This is the revert point — there is no flag.
- [ ] Confirm the six edits are all in `composite_state_builder.py` and nothing else was touched: `git diff --stat`.
- [ ] Note the restart time. You will need it to separate before/after in the log.

---

# FIRST HOUR AFTER RESTART

- [ ] **`[S1-STRUCTURE]` prints for every asset**, once per candle, with a sane `trend` and `lsm`. If `trend=0` everywhere, Livermore is not warmed — say so immediately.
- [ ] **No candle sets `bos_bullish` and `choch_bullish` together.** If any does, B1 did not land correctly.
- [ ] **`[PROOF-NEAR-MISS]` appears at INFO.** It was completely absent before.
- [ ] **Each `PROOF-*` tag appears at most once per asset per hour**, not ~23 times.
- [ ] Bot starts clean, no exceptions, cycles completing.

# FIRST 24 HOURS

- [ ] **`[TRAJECTORY]` shows `TF_CONT` on trending assets.** Today it is almost all `MR_REV`. **This is the headline check — if it does not change, the fix did not land.**
- [ ] `[MEASURE-8.4-DEATH]` starts showing `OPPOSING_CHOCH`.
- [ ] `+CHoCH(structure)` mostly disappears from scorecards. STRUCTURE loses that 0.08 on most candles — correct, it was being paid for an uptrend.
- [ ] TREND scores above 0.00 on at least one asset.
- [ ] **Watch the first trade closely.** The bot has been unable to trade at all; this build is what makes trading possible again.

- [ ] Report exact line numbers for all six edits back to Desire.

---

# REVERT

There is no flag. If this needs backing out:

```
git revert <commit-sha>
```
then restart. Or reset to the `pre-s1` tag.

**Revert triggers — any one of these, stop and tell Desire:**

- Exceptions in `_update_structure` or the trajectory block
- `trend=0` on every asset for more than one full cycle after warm-up
- Setup deaths running away — every setup dying every candle
- Trades firing at a rate that looks wrong for the account size
- Anything in the logs you cannot explain

**Do not diagnose a live problem by editing more code.** Revert first, then investigate.

---

# KNOWN CONDITIONS THIS SHIPS INTO

Desire has reviewed these and accepted them. Recorded so the context is not lost:

- **GOLD's RSI zones are inverted** — hardcoded at `council_aggregator.py` 240–241, the exact mirror of every other asset. MOMENTUM is 1.5 points and it scores backwards on GOLD.
- **TF's counter-trend penalty is 0.3** against a code default of 2.0. It signalled short in an uptrend on 29 Jul.
- **EURUSD is enabled** in the live config, against the ledger's ratified DISABLED.
- **Several phase_config flags are on** without ratification.

None of these block this build. All of them are live while it runs.

---

# NOT IN THIS BUILD

Named so nothing is quietly dropped:

- **GOLD's inverted RSI zones** — ships separately, so we can attribute which change moved the numbers.
- **TF's 0.3 counter-trend penalty** — needs a decision, not a patch.
- **The 28-candle continuation window** and break-anchored retest ordering. Decided, not yet written.
- **The two-lane setup tracker.** On hold — S1 removes most of the overwrite at source, so it may be unnecessary or much smaller. Re-run the backtest first.
- **MR's compression veto and the Mode 1 / Mode 2 dispatch mismatch.** Untouched. MR stays silent.
- **Flag ratification and EURUSD.** Desire's decisions.

---

# THE QUESTION THIS BUILD ANSWERS — NO EXTRA CODE

Once S1 is live, we need to know whether TF's own signal agrees with the proof direction. **The existing log line already answers it** — it prints all four gate values on every suppression.

```powershell
Select-String -Path logs\trading_bot.log -Pattern "brc_confirmed=True brc_kind=TF_CONT"
```

Any hit is a **direction block**: proof present, correctly typed, and TF's own signal pointed the other way. Count those against successful fires.

**This matters because on both assets in the 29 Jul log, TF's signal opposed the Livermore trend state.** If that is systematic, S1 unlocks the label but the direction check still blocks the trade — and we need that from data, not argument.
