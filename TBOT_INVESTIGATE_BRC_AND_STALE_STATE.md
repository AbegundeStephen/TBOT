# TBOT — INVESTIGATE-TOGETHER: BRC reference, stale MR list, misleading logs

**This one is different from our usual plans.** I'm not handing you finished code to apply. I've done the tracing and I'm fairly sure of the diagnosis — but I want you to *check my work against the live code before you change anything*, because you know this codebase's quirks better than a search does, and a couple of times this week the logs have fooled both of us.

**How to use this doc:**
- Each issue has a **"What I think is happening"**, then **"Verify it yourself"** (commands for you to run), then **"If confirmed, here's the change."**
- Run the verify steps first. If what you see doesn't match what I describe, **stop and tell me** — the fix assumes the diagnosis holds.
- Nothing here is flag-gated except where noted. BRC is observation-only, so a bug in it can't move a trade — but it blocks everything downstream that waits on proof.

---

# ISSUE 1 — BRC can't confirm because its reference is gone by the time it looks

## What I think is happening

BRC proves "break → retest → close." For a **reversal** setup (`MR_REV`, born at a CHoCH), it measures the retest against the Livermore natural anchor — `livermore_anchor_natural_low` or `_high`.

The problem is timing, and it's structural:

1. That anchor only exists while price is in a **NATURAL** state.
2. The instant price breaches it, the state machine flips to **SECONDARY** and *repurposes* the anchor — in `_secondary_retr`, `nl_confirmed` becomes `nl_entry`, and on the next NATURAL cycle it's overwritten.
3. But a `MR_REV` setup is **born at the CHoCH**, which happens *in* the SECONDARY move — **after** the anchor has already been consumed.

So when BRC runs `getattr(state, "livermore_anchor_natural_low", None)`, it's asking for a level that was spent one state-transition ago. It comes back `None` or stale, the `if _brc_ref is not None and > 0` guard fails, and **BRC silently does nothing.**

That's why: **15 hours, 810 MR_REV births, zero `[BRC]` fires.** Not rare — structurally unreachable for the reversal side.

There's corroboration in the code already: BRV (`break_retest.py`) hits this exact wall and has a fallback for it — *"anchor not yet confirmed — using legacy check."* BRC has no such fallback. That asymmetry is the bug.

## Verify it yourself

**Step 1 — confirm the anchor is consumed at the SECONDARY transition:**
```powershell
Select-String -Path "src/execution/livermore_state_machine.py" -Pattern "nl_entry|nl_confirmed" -Context 0,1
```
You're looking for: `_secondary_retr` reads `nl_entry` and never refreshes `nl_confirmed` while in SECONDARY. Confirm the natural anchor isn't maintained once we leave NATURAL.

**Step 2 — confirm MR_REV is born in a state where the anchor is already gone.** Add a one-line temporary debug right after the birth block in `composite_state_builder.py` (STEP 3, where `_born = {"kind": "MR_REV", ...}` is set):
```python
                    if _born is not None and _born.get("kind") == "MR_REV":
                        logger.info(
                            "[BRC-PROBE] %s: MR_REV born in state=%s  nl_anchor=%s  nh_anchor=%s",
                            _asset, _lsm_now,
                            getattr(state, "livermore_anchor_natural_low", None),
                            getattr(state, "livermore_anchor_natural_high", None),
                        )
```
Run for an hour. **My prediction:** you'll see `MR_REV born in state=SECONDARY_*` with `nl_anchor=None` (or a stale value that doesn't match recent price) on most or all births. If the anchor is consistently populated and sane at birth, my diagnosis is wrong — stop and tell me.

**Step 3 — confirm BRC is bailing at the reference guard**, not somewhere else. Temporarily, inside the BRC block right after `_brc_ref` is computed:
```python
                if _brc_kind == "MR_REV":
                    logger.info("[BRC-PROBE] %s: kind=MR_REV dir=%+d ref=%s",
                                self.asset_type, _brc_dir, _brc_ref)
```
**Prediction:** `ref=None` on the MR_REV lines. That's the silent no-op, caught in the act.

---

## If confirmed — the fix, in two parts

### PART A — capture the reference at setup birth (the real fix)

Freeze the anchor onto the setup the moment it's born, so BRC reads the frozen copy instead of the live one that's since been consumed.

**A1 — carry the frozen price on the setup dict.** In `composite_state_builder.py`, STEP 3, where the setup is born. You already build `_born` and then `.update({...})` — add the reference capture there:

```python
                    if _born is not None:
                        # BRC-FIX Part A: freeze the retest reference AT BIRTH.
                        # The Livermore natural anchor is consumed when price
                        # crosses into SECONDARY (nl_confirmed -> nl_entry), which
                        # is BEFORE the CHoCH that births an MR_REV setup. Reading
                        # the live anchor later returns None/stale, so BRC never
                        # fires. Snapshot it now, while (for MR_REV just born off a
                        # fresh CHoCH) it is still the level that actually broke.
                        _ref_price = None
                        if _born["kind"] == "MR_REV":
                            _ref_price = (
                                getattr(state, "livermore_anchor_natural_low", None)
                                if _born["dir"] == 1
                                else getattr(state, "livermore_anchor_natural_high", None)
                            )
                        elif _born["kind"] == "TF_CONT":
                            _ref_price = (
                                getattr(state, "last_swing_high_4h", None)
                                if _born["dir"] == 1
                                else getattr(state, "last_swing_low_4h", None)
                            )
                        _born.update({
                            "age": 0,
                            "born_state": _lsm_now,
                            "born_compression": _comp,
                            "last_compression": _comp,
                            "energy": "HOLDING",
                            "ref_price": _ref_price,   # BRC-FIX: frozen at birth
                        })
                        self._active_setup[_asset] = _born
                        _cur = _born
```

**A2 — publish the frozen ref in STEP 4** so BRC (and later the judges) can read it off `state`:

```python
            if _cur is not None:
                state.setup_active = True
                state.setup_kind = _cur.get("kind")
                state.setup_dir = int(_cur.get("dir", 0))
                state.setup_age = int(_cur.get("age", 0))
                state.setup_energy_trend = _cur.get("energy")
                state.setup_ref_price = _cur.get("ref_price")   # BRC-FIX
```

**A3 — add the field to the dataclass.** In `composite_state.py`, in the TRAJECTORY LAYER block next to `setup_active`:
```python
    # BRC-FIX: the retest reference price, frozen at setup birth. The live
    # Livermore anchor is consumed at the NATURAL->SECONDARY transition, so
    # BRC must measure against this snapshot, not the (by-then gone) anchor.
    setup_ref_price: Optional[float] = None
```

**A4 — BRC reads the frozen ref FIRST.** In the BRC block, replace the reference lookup so it prefers the frozen value and only falls back to the live anchor if the snapshot is missing:

```python
            if _brc_active and _brc_dir != 0 and df is not None and len(df) >= 9:
                # BRC-FIX Part A: prefer the reference frozen at setup birth.
                _brc_ref = getattr(state, "setup_ref_price", None)

                # Part B fallback: if no frozen ref (older setup, or born before
                # this fix shipped), fall back to the live anchor, then to a
                # swing reference — mirrors BRV's legacy fallback so BRC degrades
                # gracefully instead of going silent.
                if _brc_ref is None or float(_brc_ref) <= 0:
                    if _brc_kind == "MR_REV":
                        _brc_ref = (
                            getattr(state, "livermore_anchor_natural_low", None) if _brc_dir == 1
                            else getattr(state, "livermore_anchor_natural_high", None)
                        )
                    elif _brc_kind == "TF_CONT":
                        _brc_ref = (
                            getattr(state, "last_swing_high_4h", None) if _brc_dir == 1
                            else getattr(state, "last_swing_low_4h", None)
                        )
```

Everything below that (`if _brc_ref is not None and > 0:` … the retest/close check) stays exactly as-is.

### PART B — the fallback (already wired above, but here's why it's separate)

Part A fixes setups **born after** this ships. But two cases still need the fallback:
- A setup that was **already alive** when the fix deployed has no `ref_price` — it was born before the capture existed.
- A rare birth where even at CHoCH the anchor hadn't locked yet (`nl_confirmed is None`).

For both, the block above walks: **frozen ref → live anchor → swing reference.** That's the same graceful-degradation BRV already does. Without it, those cases stay silent; with it, BRC still gets a reasonable reference.

**One judgement call for you, and I want your read:** for `MR_REV`, the swing fallback uses `last_swing_high_4h/low_4h` — the *TF* reference. That's not a perfect proxy for a reversal's broken level, but it's a sane "closest structural level" when the natural anchor is truly gone. Alternative: skip the swing tier for MR_REV and just accept BRC stays silent on pre-fix setups (they age out within hours anyway). **Tell me which you prefer** — I lean toward including it because silence is what we're trying to kill, but you may feel the swing level is too loose to represent a reversal retest.

## Verify the fix worked
```powershell
# After deploy + restart, watch for real confirmations:
Select-String -Path "logs/trading_bot.log" -Pattern "\[BRC\] .*CONFIRMED"
```
Expect `[BRC] ... CONFIRMED MR_REV` lines to start appearing at roughly your backtest rate (~1 per 20h per asset). Also confirm `TF_CONT` still fires — Part A changed how it reads its ref too, so check you didn't regress the side that was closer to working.

Remove the three `[BRC-PROBE]` debug lines once satisfied.

---

# ISSUE 2 — one file still thinks MR is silent in NATURAL_REBOUND

## What I think is happening

Unit 2 changed the meaning of NATURAL_REBOUND: with `mr_rebound_short_enabled` ON, MR now produces a **short** there (the mirror of the NATURAL_RETRACEMENT long). It's no longer a "MR has no opinion" state.

But `main.py` still has this, in the minimum-regime-confidence gate:
```python
_MR_SILENT_STATES = {"NATURAL_REBOUND"}   # hard veto always zeroes MR
```
This set drives a **threshold** decision, not a block. When MR is "silent," the code *suppresses a gate-raise* on an opposing signal. The logic: "MR has no view here, so don't make the counter-signal work harder."

With Unit 2 on, that premise is false. MR *does* have a view in NATURAL_REBOUND — it wants short. So when TF/EMA want **long** (chasing the rebound), the bar should rise because they oppose MR. Instead, this stale set suppresses the raise, and **the long gets an easier ride precisely when MR is warning against it.**

To be clear on severity: this is a *threshold mis-calibration*, not a hard veto, and it only bites in NATURAL_REBOUND **with Unit 2 on**. It's real but not urgent.

## Verify it yourself
```powershell
Select-String -Path "src/main.py" -Pattern "_MR_SILENT_STATES" -Context 2,8
```
Confirm: (a) the set contains `NATURAL_REBOUND`, (b) it's used to *skip* a gate-raise (`gate raise suppressed`), and (c) there's no flag check making it Unit-2-aware. If it already reads the flag, this is done — tell me.

## If confirmed — the fix

Make the "silent" classification conditional on the flag. NATURAL_REBOUND is only MR-silent when Unit 2 is **off**.

Find where `_MR_SILENT_STATES` is defined in `main.py` and replace the static set with a flag-aware one:

```python
                # Unit 2 changed NATURAL_REBOUND from "MR silent" to "MR shorts
                # the rebound". When the flag is ON, MR has a directional view
                # here, so an opposing TF/EMA signal SHOULD face the normal
                # counter-MR gate raise. Only treat it as silent when OFF.
                _rebound_short_on = bool(
                    _phase_cfg_for_gate.get("mr_rebound_short_enabled", False)
                )
                _MR_SILENT_STATES = set() if _rebound_short_on else {"NATURAL_REBOUND"}
```

**Where does `_phase_cfg_for_gate` come from?** It needs to be the live phase_config, the same source MR itself reads (`composite_state.phase_config`), **not** `self.config` (which is the preset and never carries phase_config). Check what's already in scope at that point in `main.py`:
```powershell
Select-String -Path "src/main.py" -Pattern "phase_config" -Context 0,1
```
Use whichever live-config handle is already resolved nearby. If there isn't one in scope, pull it from the composite_state you already have in that function (`mtf_regime.get("composite_state")` → `.phase_config`). **Show me the surrounding lines and I'll confirm the right handle before you wire it** — I don't want to guess the variable name and have it silently read `{}`.

## Verify the fix
With Unit 2 ON, in a NATURAL_REBOUND cycle where TF/EMA want long and MR wants short, confirm the log shows the gate *raised* (not "suppressed"). With Unit 2 OFF, confirm behaviour is unchanged.

---

# ISSUE 3 — the logs describe things the code no longer does

## What I think is happening

Three log/label strings describe retired behaviour. None change what the bot does, but they cost real debugging time — I burned a while this week convinced Mode 3 was running because the log kept printing the word "Mode3."

The offenders, all in `mean_reversion.py`'s `generate_signal`:

1. **`_mode_label["NATURAL_REBOUND"] = "SILENT_ZONE"`** — prints "SILENT_ZONE" while the code routes to a short (when Unit 2 is on).
2. **`_mode_label["MAIN_UP"/"MAIN_DOWN"] = "HOLD(Mode3 removed)"`** — the phrase contains "Mode3", which matches naive greps and looks like execution. (The behaviour is correct — it holds — the *word* is the problem.)
3. **The routing docstring** — `NATURAL_REBOUND → zero (MR silent zone; LONGs blocked by Phase 2 HVL)` references the Hard Veto Layer, which was retired.

## Verify it yourself
```powershell
Select-String -Path "src/strategies/mean_reversion.py" -Pattern "_mode_label|_mode_desc|SILENT_ZONE|Phase 2 HVL"
```
Confirm the label dict still hardcodes `SILENT_ZONE` for NATURAL_REBOUND regardless of the flag, and the docstring still mentions HVL.

## If confirmed — the fix

**3a — make the NATURAL_REBOUND label tell the truth about what routed.** Since the label is built before the routing decision, compute it flag-aware:
```python
            _rebound_on = bool(
                (getattr(composite_state, "phase_config", {}) or {}).get(
                    "mr_rebound_short_enabled", False
                )
            ) if composite_state is not None else False

            _mode_label = {
                "NATURAL_RETRACEMENT":   "Mode1(Pullback/LONG)",
                "NATURAL_REBOUND":       "Mode1(Rebound/SHORT)" if _rebound_on else "SILENT_ZONE",
                "SECONDARY_RETRACEMENT": "Mode2(Counter/LONG)",
                "SECONDARY_REBOUND":     "Mode2(Counter/SHORT)",
                "MAIN_UP":               "HOLD(no-fade)",
                "MAIN_DOWN":             "HOLD(no-fade)",
            }
```
Note I also changed the MAIN labels from `HOLD(Mode3 removed)` to `HOLD(no-fade)` — same meaning, but the word "Mode3" is gone so it stops matching greps and confusing us.

**3b — update `_mode_desc` to match**, and drop the "Phase 2 HVL" reference in the docstring (that mechanism is retired). Replace the routing-summary docstring's NATURAL_REBOUND line with:
```
          NATURAL_REBOUND       → Mode 1 SHORT if mr_rebound_short_enabled else 0
```

**These are comment/label-only changes.** Zero behavioural impact. The only goal is that the logs stop lying.

## Verify the fix
```powershell
Select-String -Path "logs/trading_bot.log" -Pattern "Mode1\(Rebound/SHORT\)|HOLD\(no-fade\)"
```
With Unit 2 on, NATURAL_REBOUND cycles should now log `Mode1(Rebound/SHORT)`. No cycle should print the bare word "Mode3" anymore.

---

# ORDER & REPORTING

**Order:**
1. **Issue 1 (BRC)** — the priority. Everything waiting on proof is blocked until BRC fires. Do the verify probes first, confirm my diagnosis, then Part A + B.
2. **Issue 3 (labels)** — do this alongside 1, honestly, because it'll make verifying 1 much less confusing (you'll be able to trust the log lines).
3. **Issue 2 (stale MR list)** — real but least urgent; threshold-only, Unit-2-conditional.

**When you report back, per issue:**
- The output of the verify probes (paste it — did my prediction hold?)
- Line numbers of every change
- For Issue 1: the first few `[BRC] ... CONFIRMED` lines once it's live
- Anything you found that contradicts my diagnosis — I'd rather hear "you were wrong about X" than have it quietly patched over

**Two open questions I want your answer on before or during:**
- Issue 1 Part B: include the swing-level fallback for MR_REV, or let pre-fix setups stay silent? (I lean include.)
- Issue 2: which in-scope handle is the live phase_config at that point in main.py? (Show me the lines.)

Take it apart, check it against the real thing, and tell me where I've got it wrong.
