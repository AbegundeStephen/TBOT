# BUILD 3 — THE OWNERSHIP RULE

**Owner:** Desire · **Developer:** Stephen
**Files touched:** `council_aggregator.py` (main change), `main.py` (cleanup), config (cleanup)

---

## READ THIS FIRST

**This one has no flag.** It's a correction, not a feature — it enforces a rule that was always the design. Shipping it behind a switch would mean the "off" position contradicts the strategy, and that's not a valid mode.

Same reasoning as the eleven judges-prep fixes, which also shipped unflagged.

**It does change behaviour immediately on deploy.** Everything else recently has been flag-gated and inert until flipped; this isn't. Expect fewer signals, not more.

---

# THE PROBLEM

The bot has two **signal engines** — the things that actually spot a setup:

- **TF** — spots continuations (price breaks with the trend, retests, holds)
- **MR** — spots reversals (price breaks against the trend, retests, holds)

And it has **six judges** that score the setup. Two of them read the engines:

- **TREND** reads TF's signal
- **REVERSION** reads MR's signal

The other four — STRUCTURE, MOMENTUM, PATTERN, VOLUME — score independently off the board. They never see an engine signal.

## What goes wrong

Those four judges can clear the threshold **on their own**, with no engine having fired at all.

Real numbers from a USOIL cycle:

```
TREND 1.5 · STRUCTURE 1.0 · MOMENTUM 1.5 · PATTERN 0.5 · VOLUME 0.5 = 6.0
Threshold: 3.20

Evidence judges alone (STRUCTURE+MOMENTUM+PATTERN+VOLUME) max at 3.5
```

**3.5 clears 3.20.** So the bot can open a trade with no break, no retest, no close-through — just four evidence judges agreeing about the board.

That contradicts the rule the whole strategy rests on: *no BOS/CHoCH, no trade. No retest, no trade. No confirming close, no trade.*

**And the proof gates don't catch it.** They gate TF, Mode 2 and solo-fire. None of them gate the council's consensus path — which is the one route that can trade without proof.

---

# THE RULE

**An engine must propose. The judges can only confirm or deny.**

Three parts:

1. **A side can only fire if TF or MR proposed it.** Evidence judges add to a thesis; they can never create one.
2. **The proposing side must also outscore the other side.** If the judges collectively lean the other way, the trade is killed — but the *other* side doesn't fire either, because no engine proposed it.
3. **If both engines fire in opposite directions, hold.** Two valid theses in conflict isn't a close call to settle by adding up. It's the market being unclear, and the bot sits out.

## Why EMA doesn't count

EMA is a **pure confirmer**. It has no thesis of its own — it confirms or tempers from the side. A confirmer that fires alone has nothing to confirm, so it can't own a cycle.

*(Note: this is safe today because `tf_drives_trend_judge_enabled` is ON, so TF is both the thesis and the TREND judge's driver. If Unit 3 is ever switched off, TREND would score on EMA while ownership comes from TF — the two could point opposite ways. Flag it to Desire if that flip is ever considered.)*

## What this replaces

The **MR lean conflict gate**, which raised the required score when the council's direction opposed MR's lean. It was a tax — trade against MR if you score high enough. Its bumps have been softened to 0.25/0.0, which is close to nothing.

The ownership rule answers the same question structurally instead of by calibration. **The gate comes out.**

---

# ITEM 1 — The ownership check

**File:** `src/execution/council_aggregator.py`, in `_get_aggregated_signal_impl`.

**Where exactly:** immediately **before** the decision cascade — the `if _buy_clears and _sell_clears: / elif _buy_clears: / elif _sell_clears:` block. Gate the two `_clears` flags and the existing cascade handles the rest.

**Find the line just above the cascade** (after `_buy_clears` and `_sell_clears` are computed) and insert:

```python
            # ══════════════════════════════════════════════════════════════
            # BUILD 3: OWNERSHIP RULE
            # A side can only fire if a signal ENGINE proposed it. The four
            # evidence judges (STRUCTURE, MOMENTUM, PATTERN, VOLUME) confirm
            # or deny a thesis — they cannot originate one. Without this they
            # can clear the threshold alone (they max at 3.5 vs a 3.20 bar),
            # producing entries with no BOS, no retest and no close-through.
            #
            # EMA is excluded deliberately: it is a pure confirmer with no
            # thesis of its own, so it cannot own a cycle.
            # ══════════════════════════════════════════════════════════════
            _buy_has_thesis  = (tf_signal == 1)  or (mr_signal == 1)
            _sell_has_thesis = (tf_signal == -1) or (mr_signal == -1)

            # Two engines pointing opposite ways is a genuine conflict, not a
            # close call to be settled by adding up scores. Sit it out — this
            # mirrors the existing ambiguous_both_sides_cleared precedent.
            _opposing_theses = (
                (tf_signal == 1 and mr_signal == -1)
                or (tf_signal == -1 and mr_signal == 1)
            )

            # Capture what WOULD have fired, before the gate closes it, so the
            # shadow engine can still track the outcome (Item 2 below).
            _ownership_intended = 0
            if _buy_clears and not _sell_clears:
                _ownership_intended = 1
            elif _sell_clears and not _buy_clears:
                _ownership_intended = -1
            _ownership_blocked_reason = ""

            if _opposing_theses:
                if _buy_clears or _sell_clears:
                    _ownership_blocked_reason = "thesis_conflict"
                    logger.info(
                        "[COUNCIL] %s: HOLD — opposing theses (TF=%+d, MR=%+d). "
                        "Two engines disagree; not arbitrating by score.",
                        self.asset_type, tf_signal, mr_signal,
                    )
                _buy_clears = False
                _sell_clears = False
            else:
                # Thesis required, AND the owning side must outscore the other.
                # A thesis whose own evidence points the other way is not a trade.
                _buy_ok  = _buy_has_thesis  and (buy_total > sell_total)
                _sell_ok = _sell_has_thesis and (sell_total > buy_total)

                if _buy_clears and not _buy_ok:
                    _ownership_blocked_reason = (
                        "no_thesis_backing" if not _buy_has_thesis else "outvoted"
                    )
                    logger.info(
                        "[COUNCIL] %s: BUY killed — %s (TF=%+d MR=%+d, "
                        "buy=%.2f vs sell=%.2f).",
                        self.asset_type, _ownership_blocked_reason,
                        tf_signal, mr_signal, buy_total, sell_total,
                    )
                if _sell_clears and not _sell_ok:
                    _ownership_blocked_reason = (
                        "no_thesis_backing" if not _sell_has_thesis else "outvoted"
                    )
                    logger.info(
                        "[COUNCIL] %s: SELL killed — %s (TF=%+d MR=%+d, "
                        "sell=%.2f vs buy=%.2f).",
                        self.asset_type, _ownership_blocked_reason,
                        tf_signal, mr_signal, sell_total, buy_total,
                    )

                _buy_clears  = _buy_clears  and _buy_ok
                _sell_clears = _sell_clears and _sell_ok
```

⚠️ **Confirm `tf_signal`, `mr_signal`, `buy_total`, `sell_total` are all in scope at that point.** They're computed earlier in the same method — but check, don't assume.

## A note on the structure tie-break

Because the rule requires the owner to outscore the other side, `buy_total > sell_total` and `sell_total > buy_total` can never both be true. **The `elif _buy_clears and _sell_clears:` branch becomes unreachable.**

**Leave it in place.** An exact tie now means neither clears → HOLD, which is the same outcome that branch produced. Add a short comment above it noting it's unreachable under the ownership rule, so nobody later spends an hour working out why it never fires.

---

# ITEM 2 — Keep the shadow engine seeing these blocks ⚠️

**Why this matters:** the shadow engine opens a virtual position for every blocked signal and tracks whether it *would* have won. That's how you find out if a gate is costing you money.

But `original_signal = signal` is captured **after** the cascade. If ownership sets both `_clears` to False, `signal` is already 0, `original_signal` is 0, and `main.py` never opens a shadow. **The rule would block silently and you'd never learn what it cost.**

**Find:**
```python
            # Capture initial consensus before penalties and vetos
            original_signal = signal
```

**Replace with:**
```python
            # Capture initial consensus before penalties and vetos.
            # Build 3: when the ownership rule kills a side, `signal` is already
            # 0 here — so surface the intended direction instead. main.py uses
            # original_signal to decide whether to open a shadow position, and
            # a silent block would leave the rule permanently unmeasured.
            original_signal = signal if signal != 0 else _ownership_intended
```

**Then, where `decision_type` and `reasoning` are set for the HOLD path**, add the ownership case so `main.py` can attribute the block. Insert before the existing HOLD assignment:

```python
            if signal == 0 and _ownership_blocked_reason:
                decision_type = f"BLOCKED (Ownership Rule: {_ownership_blocked_reason})"
                reasoning = (
                    f"Ownership rule: {_ownership_blocked_reason}. "
                    f"TF={tf_signal:+d} MR={mr_signal:+d} "
                    f"buy={buy_total:.2f} sell={sell_total:.2f}."
                )
```

⚠️ **`main.py` parses `decision_type` as `_dt.split("(", 1)[-1].rstrip(")")` to get `block_source`.** The format above yields `Ownership Rule: no_thesis_backing`, which is what you want in the shadow scorecard and the funnel. **Don't change the bracket format.**

**CHECK 2:** with a blocked cycle in the log, confirm the funnel record carries the ownership veto family, and a shadow position opens for the intended direction.

---

# ITEM 3 — Remove the MR lean conflict gate (council)

**File:** `src/execution/council_aggregator.py`

Find the block beginning:
```python
            if (_mr_lean_mode != "off"
                and signal != 0
                and not ((mr_signal == 1 and signal == 1) or (mr_signal == -1 and signal == -1))):
```

**Delete the whole block**, including its `_MR_LEAN_LONG` / `_MR_LEAN_SHORT` sets and any `_mr_lean_*` config reads, and leave this in its place:

```python
            # ── MR lean conflict gate: REMOVED (Build 3) ──────────────────
            # This raised required_score when the council's direction opposed
            # MR's Livermore lean — a tax on trading against MR, tuned to
            # 1.5, then 0.25, then effectively 0.0 for secondary states.
            #
            # The ownership rule above replaces it structurally: the opposing
            # side can no longer fire at all, so there is no bar to price.
            #
            # The two losses this gate was built on are both covered, harder:
            #   USOIL Jun2026 — MR leaned LONG, council took SELL at exactly
            #     threshold, price rose. Under ownership: MR had no signal, so
            #     no owner on the sell side → HOLD.
            #   BTC Jun2026 — MAIN_UP, MR wanted to fade, council bought 4.0
            #     at the top of an extended leg, stopped out. Same → HOLD.
            # ──────────────────────────────────────────────────────────────
```

**Keep that comment.** It's the evidence trail for why the case is handled — without it someone re-adds a tax later thinking it was never covered.

**CHECK 3:** grep `_mr_lean` in `council_aggregator.py` → **0 hits**.

---

# ITEM 4 — Remove the performance-mode mirror

**File:** `src/main.py`

There's a second copy of the same logic on the performance-aggregator path (`_perf_lean_conflict` or similar). Council mode is pinned, so it's already dormant — but two copies of the same rule is how stale code survives, and this project has lost real time to that.

```powershell
Select-String -Path "src/main.py" -Pattern "_perf_lean_conflict|mr_lean" -Context 3,10
```

**Delete the block.** Also remove the now-dead attribution branch:
```python
elif "MR lean conflict" in _dt:
    block_source = "MR Lean Conflict"
```

**CHECK 4:** grep `mr_lean` across `src/` → **0 hits**.

---

# ITEM 5 — Config cleanup

Remove from `phase_config` in **all three** config files:

```json
"council_mr_lean_mode": "soft",
"council_mr_lean_bump_main": 0.25,
"council_mr_lean_bump_secondary": 0.0
```

**CHECK 5:** bot starts clean — no `KeyError`, no missing-config warnings.

---

## VERIFICATION RECORD

### PASS 1 — FORWARD (does it do what we intend?)

- **Thesis check** uses `tf_signal` / `mr_signal` directly. EMA excluded by design. ✓
- **Outscore condition** means a thesis whose own evidence opposes it cannot fire, and the opposing side can't fire either — it has no engine behind it. ✓
- **Opposing theses → HOLD** mirrors the existing `ambiguous_both_sides_cleared` precedent for genuine ambiguity. ✓
- **Item 2** keeps the block visible to the funnel and shadow, so the rule's cost is measurable from day one. ✓
- **Items 3–5** remove the mechanism this replaces, in both copies, plus its config. ✓

### PASS 2 — BACKWARD (verified against the repo)

- **Insertion point confirmed:** the decision cascade is a single `if/elif/elif` on `_buy_clears`/`_sell_clears`; gating those two flags needs no restructuring. ✓
- **Variables confirmed in scope** at that point: `tf_signal`, `mr_signal`, `ema_signal`, `buy_total`, `sell_total`. ✓
- **`original_signal = signal` confirmed** captured after the cascade — which is exactly why Item 2 is needed. ✓
- **`main.py` block_source parsing confirmed** as `_dt.split("(", 1)[-1].rstrip(")")`; the `BLOCKED (Ownership Rule: reason)` format matches. ✓
- **MR lean conflict gate confirmed present** in `council_aggregator.py`, keyed on `_mr_lean_mode` with `_MR_LEAN_LONG`/`_MR_LEAN_SHORT` state sets. ✓
- **Performance-mode mirror confirmed** to exist separately in `main.py`. ✓
- **Structure tie-break confirmed** to become unreachable — same outcome preserved via the tie → HOLD path. ✓
- **`tf_drives_trend_judge_enabled` confirmed ON** in config, so TF is both the thesis and TREND's driver — owner and judge agree by construction. ✓

---

## BUILD CHECKLIST

- [ ] Item 1 — ownership check inserted **before** the decision cascade
- [ ] Item 1 — `tf_signal` / `mr_signal` / `buy_total` / `sell_total` confirmed in scope
- [ ] Item 1 — comment added above the now-unreachable both-clear branch
- [ ] Item 2 — `original_signal` falls back to `_ownership_intended`
- [ ] Item 2 — `decision_type` uses the `BLOCKED (Ownership Rule: reason)` format
- [ ] Item 3 — council lean gate deleted, explanatory comment kept; grep `_mr_lean` = 0
- [ ] Item 4 — performance mirror + attribution branch deleted; grep `mr_lean` in `src/` = 0
- [ ] Item 5 — three config keys removed from all three files
- [ ] Bot starts clean, no errors
- [ ] Report exact line numbers for every change

---

## WHAT TO SEND BACK

1. Line numbers for every edit
2. **A log sample showing an ownership block** — one of `no_thesis_backing`, `outvoted`, or `thesis_conflict`, with the TF/MR values and both totals
3. Confirmation a **shadow position opened** for that blocked signal (`gate_blocked_by` shows the ownership rule)
4. Grep results for `_mr_lean` and `mr_lean` — both should be 0
5. Anything you found that contradicts the reasoning above — I'd rather hear it than have it patched over

---

## WHAT TO EXPECT

**Fewer signals.** A side now needs to clear its threshold *and* beat the other side *and* have an engine behind it. Previously it only needed to clear.

At current score levels this changes almost nothing — nothing has cleared anything in days. It matters once scores lift.

**The most likely block reason at first is `no_thesis_backing`** — TF is proof-gated and mostly silent, MR is failing its compression gate, so both engines are quiet while the evidence judges keep scoring. That's the rule doing exactly its job: refusing to trade on evidence alone.

**Watch the shadow scorecard for the ownership rows.** If they show consistently positive forward P&L, the rule is blocking winners and we revisit. That's the whole reason Item 2 exists.
