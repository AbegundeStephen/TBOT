# BUILD 2 — PROOF KIND, PROOF AGE, QUALITY MATHS + VISIBILITY

**Owner:** Desire · **Developer:** Stephen
**Files touched:** `composite_state.py`, `composite_state_builder.py`, `trend_following.py`, `mean_reversion.py`, `signal_aggregator.py`, `council_aggregator.py`, config

---

## READ THIS FIRST

Five items. **None of them change what the bot does today** — three sit behind flags that are currently off, one is a logging switch, one restores a flag that got turned off by accident, and one adds a counter that starts life very permissive.

**What this build is for:** the proof gates from Build 1 have two holes. They don't check *what kind* of proof they got, and they don't check *how old* it is. This closes both. It also fixes two hardcoded numbers that would quietly break the six-slot flag, and turns on the logging we need to see what the judges are actually doing.

**Order:** do **Item 5 first** — Item 1's gates read the field Item 5 creates. Everything after that is independent.

---

# ITEM 5 — How old is the proof? *(do this one first)*

## Why, in plain terms

BTC breaks 64142. Price comes back, touches it, closes through. **That's the proof** — and the right entry is right there, with a stop just under the level.

But the bot keeps saying "confirmed" for the next several bars while price runs to 64444. Same proof, still being announced 300 points later.

We saw exactly this in the logs — one reference level, confirming across six different closes as price climbed.

**Why it matters once the gates go live:** if TF's score only clears its threshold on bar six, the gate still says "proof confirmed" and TF enters **300 points above where the proof happened**, with its stop still down at the level. That's chasing — the thing break-retest-close exists to prevent.

**The fix isn't to delete the proof.** It's to record how old it is, so each gate can decide for itself how fresh it needs.

## ⚠️ The thing most likely to get built wrong

The bot cycles every ~5 minutes but trades on **1-hour bars**. The same proof gets recomputed roughly **22 times within one bar**.

**Age must count BARS, not cycles.** Count cycles and a proof "ages" 22 times an hour, making the whole field meaningless. That's why the code below keys off the bar timestamp instead of just incrementing.

## 5a — Two new fields

**File:** `src/execution/composite_state.py`, in the BRC block next to `brc_confirmed`:

```python
    # Build 2: how old the proof is, in BARS (not cycles). 0 = completed on this
    # bar. The bot recomputes ~22x per 1H bar, so this must key off the bar
    # timestamp — a naive per-cycle increment would age a proof 22x per hour.
    brc_age: int = 0
    brc_first_confirmed_ts: Optional[object] = None
```

## 5b — Memory on the builder

**File:** `src/execution/composite_state_builder.py`, in `__init__`:

```python
        # Build 2: per-asset BRC proof memory — {ref, first_ts, last_ts, age}.
        # Tracks a single continuously-confirmed proof so brc_age counts bars.
        self._brc_memory = {}
```

## 5c — The age logic

**File:** `src/execution/composite_state_builder.py`, in the BRC block.

**Find the confirmation branch:**
```python
                    if _retested and _closed_through:
                        state.brc_confirmed = True
                        state.brc_direction = _brc_dir
                        state.brc_kind = _brc_kind
                        state.brc_tier = None
                        logger.info(
                            "[BRC] %s: CONFIRMED %s dir=%+d ref=%.5g close=%.5g "
                            "(strict close-through, 8-bar retest)",
                            self.asset_type, _brc_kind, _brc_dir, _brc_ref, _brc_close,
                        )
```

**Replace with:**
```python
                    if _retested and _closed_through:
                        # ── Build 2: age the proof in BARS ────────────────────
                        # Same proof at the same reference across several bars is
                        # ONE proof getting older, not several proofs. A different
                        # reference means a genuinely new proof — reset to 0.
                        try:
                            _bar_ts = df.index[-1]
                        except Exception:
                            _bar_ts = None

                        _mem = self._brc_memory.get(self.asset_type)

                        if _mem is None or _mem.get("ref") != _brc_ref:
                            # New proof at a new level.
                            _age = 0
                            _first_ts = _bar_ts
                        elif _bar_ts is not None and _bar_ts != _mem.get("last_ts"):
                            # Same proof, but a NEW bar has closed — it ages by 1.
                            _age = int(_mem.get("age", 0)) + 1
                            _first_ts = _mem.get("first_ts")
                        else:
                            # Same proof, same bar — another cycle within the bar.
                            # Do NOT age. This is the ~22x-per-bar case.
                            _age = int(_mem.get("age", 0))
                            _first_ts = _mem.get("first_ts")

                        self._brc_memory[self.asset_type] = {
                            "ref": _brc_ref,
                            "first_ts": _first_ts,
                            "last_ts": _bar_ts,
                            "age": _age,
                        }

                        state.brc_confirmed = True
                        state.brc_direction = _brc_dir
                        state.brc_kind = _brc_kind
                        state.brc_tier = None
                        state.brc_age = _age
                        state.brc_first_confirmed_ts = _first_ts

                        logger.info(
                            "[BRC] %s: CONFIRMED %s dir=%+d ref=%.5g close=%.5g "
                            "age=%d bar(s) (strict close-through, 8-bar retest)",
                            self.asset_type, _brc_kind, _brc_dir, _brc_ref,
                            _brc_close, _age,
                        )
                    else:
                        # Proof condition no longer holds — forget it. If it
                        # re-forms later that is a NEW proof starting at age 0.
                        self._brc_memory.pop(self.asset_type, None)
```

⚠️ **Watch the `else:`** — it pairs with `if _retested and _closed_through:` and must sit at that same indent level, inside the `if _brc_ref is not None...` block.

⚠️ **`df.index[-1]`** assumes the dataframe is indexed by bar timestamp. **Confirm that's true.** If the timestamp is a column (e.g. `df["timestamp"]`), use that and tell Desire.

## 5d — Three config keys, set to 20

Add to `phase_config` in **all three** config files:
```json
"brc_max_age_tf": 20,
"brc_max_age_mr": 20,
"brc_max_age_solo": 20
```

**Why 20 and not something tight:** this counter has never run. We have one weekend sample on one asset showing a proof persisting across roughly six closes — not enough to pick a real limit from. **20 is deliberately permissive**: it lets the full distribution show up in the logs so the number can be chosen from evidence later, while still blocking anything genuinely absurd.

Separate keys per gate on purpose — a reversal caught late is worse than a continuation caught late, so these will likely diverge once there's data.

**CHECK 5:** logs show `age=0` on a fresh proof, then `age=1`, `age=2` on following bars — **and the age must NOT change between cycles within the same bar.** Watch a full hour: the same age should repeat ~22 times, then increment once.

---

# ITEM 1 — The gates must check the KIND and the AGE of the proof

## Why the kind matters, in plain terms

Two kinds of trade, two kinds of proof:

- **Continuation** — price breaks *with* the trend, retests, holds. The trend continues. **TF's trade.**
- **Reversal** — price breaks *against* the trend, retests, holds. The trend is turning. **MR's trade.**

The board records which, in `brc_kind` — `"TF_CONT"` or `"MR_REV"`. **The gates don't look.**

**What goes wrong.** BTC is falling. It breaks down through support, retests, closes lower — a **continuation** proof, direction short.

Mode 2 also wants short, but for the opposite reason: it thinks the bounce is exhausted and the market is **reversing**.

The gate sees "proof exists, direction short" and waves it through. **The trade got approved by evidence against itself.**

**Note:** the new bar system already does this correctly (`_bar_trend` checks `brc_kind == "TF_CONT"`). This brings the gates in line with what's already right there.

## 1a — TF gate

**File:** `src/strategies/trend_following.py`, in `_generate_live_signal`, the TF-PROOF block.

**Find:**
```python
                _brc_dir = (
                    composite_state.get("brc_direction", 0)
                    if isinstance(composite_state, dict)
                    else getattr(composite_state, "brc_direction", 0)
                )
                if not (_brc_ok and _brc_dir == signal):
```

**Replace with:**
```python
                _brc_dir = (
                    composite_state.get("brc_direction", 0)
                    if isinstance(composite_state, dict)
                    else getattr(composite_state, "brc_direction", 0)
                )
                # Build 2: require the CONTINUATION proof specifically. A reversal
                # proof (MR_REV) is evidence the trend is TURNING — it must not
                # green-light a continuation entry. TF trades BOS -> retest ->
                # close; that is what TF_CONT means.
                _brc_kind = (
                    composite_state.get("brc_kind", None)
                    if isinstance(composite_state, dict)
                    else getattr(composite_state, "brc_kind", None)
                )
                # Build 2: and require it to be reasonably fresh. 20 is
                # deliberately permissive for now so the age distribution shows
                # up in the logs and the real limit can be set from evidence.
                _brc_age = (
                    composite_state.get("brc_age", 0)
                    if isinstance(composite_state, dict)
                    else getattr(composite_state, "brc_age", 0)
                )
                _brc_max_age = int(_pc.get("brc_max_age_tf", 20))
                if not (
                    _brc_ok
                    and _brc_kind == "TF_CONT"
                    and _brc_dir == signal
                    and _brc_age <= _brc_max_age
                ):
```

**Update the log line below it:**
```python
                    if not silent:
                        logger.info(
                            "[TF] %s: signal=%+d suppressed — no fresh CONTINUATION "
                            "proof (brc_confirmed=%s brc_kind=%s brc_direction=%s "
                            "brc_age=%s max=%s).",
                            getattr(self, "name", "TF"), signal, _brc_ok,
                            _brc_kind, _brc_dir, _brc_age, _brc_max_age,
                        )
```

## 1b — MR Mode 2 gate

**File:** `src/strategies/mean_reversion.py`, in `_mode2_counter_trend`, the FULL-PROOF GATE block.

**Find:**
```python
                _brc_ok2 = getattr(composite_state, "brc_confirmed", False)
                _brc_dir2 = getattr(composite_state, "brc_direction", 0)
                if not (_brc_ok2 and _intended_dir != 0 and _brc_dir2 == _intended_dir):
```

**Replace with:**
```python
                _brc_ok2 = getattr(composite_state, "brc_confirmed", False)
                _brc_dir2 = getattr(composite_state, "brc_direction", 0)
                # Build 2: require the REVERSAL proof specifically. A continuation
                # proof (TF_CONT) is evidence the trend is HOLDING — it must not
                # authorise a counter-trend entry. Mode 2 trades CHoCH -> retest
                # -> close; that is what MR_REV means.
                _brc_kind2 = getattr(composite_state, "brc_kind", None)
                # Build 2: and require freshness. A reversal caught several bars
                # late is not a reversal — it is a chase.
                _brc_age2 = int(getattr(composite_state, "brc_age", 0) or 0)
                _brc_max_age2 = int(_pc2.get("brc_max_age_mr", 20))
                if not (
                    _brc_ok2
                    and _brc_kind2 == "MR_REV"
                    and _intended_dir != 0
                    and _brc_dir2 == _intended_dir
                    and _brc_age2 <= _brc_max_age2
                ):
```

**Update the log line below:**
```python
                    logger.info(
                        "[MR Mode2] %s: no fresh REVERSAL proof dir=%+d "
                        "(brc_confirmed=%s brc_kind=%s brc_direction=%s "
                        "brc_age=%s max=%s) — full-proof gate holds.",
                        self.asset, _intended_dir, _brc_ok2, _brc_kind2,
                        _brc_dir2, _brc_age2, _brc_max_age2,
                    )
```

⚠️ **Expect this gate to stay closed initially.** `MR_REV` proofs have never confirmed live — 252 of 252 were `TF_CONT`. That's the reference bug the other build fixes. Until that lands, flipping `mr_mode2_brc_gate_enabled` mutes Mode 2 entirely. **That's correct** — better silent than firing on the wrong evidence — but don't flip it and then wonder why Mode 2 went quiet.

## 1c — Solo-fire gate (and EMA gets blocked)

**File:** `src/execution/signal_aggregator.py`, the SOLO-FIRE PROOF GATE block.

**Why EMA is blocked:** in this architecture EMA is a **pure confirmer**. It has no thesis of its own — it confirms or tempers from the side. A confirmer firing *alone* has nothing to confirm; it would be acting as an originator, a role it was explicitly not given.

**Find:**
```python
                            _brc_ok_i = bool(_ind_get("brc_confirmed", False))
                            _brc_dir_i = int(_ind_get("brc_direction", 0) or 0)
                            if not (_brc_ok_i and _brc_dir_i == best_signal):
                                logger.info(
                                    "[INDEPENDENT] %s: %s solo fire suppressed — no "
                                    "break-retest-close proof (brc_confirmed=%s "
                                    "brc_direction=%s vs signal=%+d).",
                                    self.asset_type, best_name, _brc_ok_i,
                                    _brc_dir_i, best_signal,
                                )
                                candidates = []
```

**Replace with:**
```python
                            _brc_ok_i = bool(_ind_get("brc_confirmed", False))
                            _brc_dir_i = int(_ind_get("brc_direction", 0) or 0)
                            _brc_kind_i = _ind_get("brc_kind", None)
                            _brc_age_i = int(_ind_get("brc_age", 0) or 0)
                            _brc_max_age_i = int(_ind_pc.get("brc_max_age_solo", 20))

                            # Build 2: a solo fire must carry the proof kind that
                            # matches the engine firing it.
                            #   TF trades continuation -> needs TF_CONT
                            #   MR trades reversal     -> needs MR_REV
                            #   EMA is a pure confirmer with no thesis of its own.
                            #     A confirmer cannot originate a trade — there is
                            #     nothing for it to confirm when it fires alone —
                            #     so it has no valid solo proof and is blocked.
                            _required_kind = {
                                "TF": "TF_CONT",
                                "MR": "MR_REV",
                            }.get(best_name)

                            if _required_kind is None:
                                logger.info(
                                    "[INDEPENDENT] %s: %s solo fire suppressed — "
                                    "confirmer engines cannot originate a trade "
                                    "under the proof gate.",
                                    self.asset_type, best_name,
                                )
                                candidates = []
                            elif not (
                                _brc_ok_i
                                and _brc_kind_i == _required_kind
                                and _brc_dir_i == best_signal
                                and _brc_age_i <= _brc_max_age_i
                            ):
                                logger.info(
                                    "[INDEPENDENT] %s: %s solo fire suppressed — no "
                                    "fresh %s proof (brc_confirmed=%s brc_kind=%s "
                                    "brc_direction=%s brc_age=%s max=%s vs signal=%+d).",
                                    self.asset_type, best_name, _required_kind,
                                    _brc_ok_i, _brc_kind_i, _brc_dir_i,
                                    _brc_age_i, _brc_max_age_i, best_signal,
                                )
                                candidates = []
```

⚠️ **Watch the indentation.** The `if _required_kind is None:` / `elif not (...)` pair replaces a single `if not (...)`. Everything after — the second `if candidates:` block — must stay exactly where it was. **Paste the finished block back to Desire before calling it done.**

**CHECK 1:** grep each file for `brc_kind` and `brc_age` — one new read of each per gate.

---

# ITEM 2 — Two hardcoded `5.0`s that will break the six-slot flag

## Why, in plain terms

The bot scores signals as a **percentage of the best possible score**. With five judges the best possible was 5.0, so someone wrote `5.0` straight into the maths. Fine at the time.

**There are six judges now.** The ceiling becomes ~6.0 once `six_slot_judges_enabled` is on — but these two lines still divide by 5.0, measuring quality against the wrong maximum.

**This is a hard blocker on the six-slot flag.** Must be fixed before that flag is flipped.

**File:** `src/execution/council_aggregator.py`

## 2a — the quality rejection check

**Find:**
```python
                min_quality_threshold = 0.55
                signal_quality = total_score / 5.0
```

**Replace with:**
```python
                min_quality_threshold = 0.55
                # Build 2: was a hardcoded 5.0 — correct only while the weights
                # summed to 5.0. With a sixth judge the ceiling moves, so quality
                # must be measured against the real achievable maximum.
                _sq_denom = _achievable_max if _achievable_max > 0 else 5.0
                signal_quality = total_score / _sq_denom
```

## 2b — the base quality calculation

**Find:**
```python
            base_quality = min(total_score / 5.0, 1.0)
```

**Replace with:**
```python
            # Build 2: same hardcoded-5.0 problem as above.
            _bq_denom = _achievable_max if _achievable_max > 0 else 5.0
            base_quality = min(total_score / _bq_denom, 1.0)
```

⚠️ **Check `_achievable_max` is in scope at BOTH sites.** If either edit sits inside a `try`/`except` or a branch where it wasn't assigned, you get a `NameError`. If unreachable, use `sum(judge_weights.values())` and tell Desire.

**CHECK 2:** with `six_slot_judges_enabled` **off**, print `_sq_denom` for one cycle. Must be **exactly 5.0**. Off by anything, stop and report.

**Heads-up for Desire:** once six-slot is on, the same raw score yields a slightly lower quality figure (against 6.0 instead of 5.0), so the `0.55` gate bites a little sooner. Six contributors should lift raw scores enough to compensate. Leaving `0.55` alone and watching.

---

# ITEM 3 — Turn on detailed logging

## Why

Every judge writes an explanation of its score — `STRUCT BUY: ✅ Bullish BOS (1.28) +defended(0.62)` and so on. **None of it reaches the log**, because those lines only print when detailed logging is on.

The cost: 24 fixes shipped across the judges, and exactly **one** can be proven working. The rest are invisible — we're inferring from arithmetic instead of reading what the code says it did.

## What to do

```powershell
Select-String -Path "src/execution/council_aggregator.py" -Pattern "detailed_logging"
Select-String -Path "config/config.json" -Pattern "detailed_logging"
```

Turn it on — config if config-driven, constructor default if not.

**If config-driven**, add/set in **all three** files:
```json
"detailed_logging": true
```

**If it's a constructor argument or hardcoded**, set the default to `True` and tell Desire where it lives.

**CHECK 3:** after restart the log shows `STRUCT BUY:`, `MOM SELL:`, `PATTERN BUY:` alongside the scorecard bars.

**On volume:** the log gets noticeably bigger. That's the point for now. If it becomes unmanageable, say so and we'll scope it down rather than switch it off.

---

# ITEM 4 — A flag that got turned off, probably by accident

## What happened

`mr_vetoes_as_dampeners_enabled` turns MR's hard blocks into soft penalties — instead of "veto, stop," it becomes "reduce confidence, keep evaluating." Affects **both** MR modes (Mode 1's `vol_down_ratio`, Mode 2's `range_classification`).

**23 July:** logs showed it working — `DAMPEN x0.40 (flag ON)`.
**Now:** both config files say `false`, zero DAMPEN lines, and **392 hard VETOs** in the last 33 hours.

One of the few flags already verified working end to end. Read is that it was flipped off by mistake — possibly a config rebuild from template.

## What to do

```powershell
Select-String -Path "config/config.json" -Pattern "mr_vetoes_as_dampeners_enabled"
Select-String -Path "config/config.prod.json" -Pattern "mr_vetoes_as_dampeners_enabled"
Select-String -Path "config/config.template.json" -Pattern "mr_vetoes_as_dampeners_enabled"
```

**Set to `true` in all three.**

**Report what you found** — `false` everywhere, or only some files? That tells us whether it was deliberate or a template rebuild wiping it, and **whether other flags were reset the same way.**

**CHECK 4:** after restart, `DAMPEN x` lines reappear when MR hits a dampened condition.

---

## VERIFICATION RECORD

### PASS 1 — FORWARD (does it do what we intend?)

- **5** — age keys off the bar timestamp, so intra-bar recomputes don't inflate it. New reference resets to 0. Lapsed condition clears memory so a re-formed proof starts fresh. ✓
- **5d** — 20 is permissive enough to observe the real distribution while blocking the absurd. ✓
- **1a/1b/1c** — each gate requires the proof kind matching the trade it authorises, plus freshness. ✓
- **1c EMA block** — matches EMA's assigned role (pure confirmer, no thesis), not its indicator behaviour. ✓
- **2a/2b** — with six-slot off, `_achievable_max` is 5.0, so both edits are arithmetically identical to current code. ✓
- **3** — logging only, zero decision impact. ✓
- **4** — restores a previously verified flag. ✓

### PASS 2 — BACKWARD (verified against the repo)

- **TF gate confirmed** at `if not (_brc_ok and _brc_dir == signal)` — no kind or age check. ✓
- **Mode 2 gate confirmed** at `if not (_brc_ok2 and _intended_dir != 0 and _brc_dir2 == _intended_dir)` — no kind or age check. ✓
- **Solo-fire gate confirmed** at `if not (_brc_ok_i and _brc_dir_i == best_signal)` — no kind or age check. ✓
- **`_bar_trend` already checks kind** (`brc_kind == "TF_CONT"`) — this aligns the gates to a pattern already correct in the bars. ✓
- **Both divisors confirmed hardcoded.** ✓
- **`brc_kind` confirmed** written by the builder as `"TF_CONT"` / `"MR_REV"`. ✓
- **Latching confirmed from live logs** — one reference (64142) confirmed across six different closes as price ran ~300 points. That is what Item 5 measures. ✓
- **Dampener confirmed** gated at two sites (Mode 1 `_vdr_damp`, Mode 2 `_rc_damp`); both configs currently `false`. ✓
- **`_pc` / `_pc2` / `_ind_pc` confirmed in scope** at each gate — the new `.get("brc_max_age_*")` reads use handles already resolved there. ✓

---

## BUILD CHECKLIST

- [ ] 5a — `brc_age` + `brc_first_confirmed_ts` added to the dataclass
- [ ] 5b — `self._brc_memory = {}` in the builder's `__init__`
- [ ] 5c — age logic in; **`else:` correctly paired and indented**
- [ ] 5c — `df.index[-1]` confirmed as the bar timestamp (or corrected)
- [ ] 5d — three `brc_max_age_*` keys at **20** in all three configs
- [ ] 5 — **age holds steady within a bar, increments once per new bar**
- [ ] 1a — TF gate requires `TF_CONT` + age; log updated
- [ ] 1b — Mode 2 gate requires `MR_REV` + age; log updated
- [ ] 1c — solo-fire maps engine→kind, EMA blocked, age checked; **block pasted back**
- [ ] 2a/2b — both divisors use `_achievable_max`; scope confirmed
- [ ] 2 — with six-slot OFF, `_sq_denom` prints exactly **5.0**
- [ ] 3 — `detailed_logging` on; judge explanations visible
- [ ] 4 — dampener `true` in all three configs; **report what you found**
- [ ] Bot starts clean, no errors
- [ ] Report exact line numbers for every change

---

## WHAT TO SEND BACK

1. Line numbers for every edit
2. The finished solo-fire block, pasted (indentation is the risk)
3. **A log sample spanning at least two full bars**, showing `age=` holding steady within a bar then incrementing — this is what proves Item 5 works
4. `_sq_denom` value with six-slot off (must be 5.0)
5. What the dampener flag was set to in each of the three config files
6. Where `detailed_logging` lives and how you switched it on
7. A short log sample showing judge explanations now printing
