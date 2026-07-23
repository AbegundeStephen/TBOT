# 2 — SIX-SLOT JUDGE SPLIT

**Deploy order: FILE 2 of 4. Ships only after File 1 has soaked and its flags are on.**

Owner: Desire · Developer: Stephen · **File touched:** `src/execution/council_aggregator.py`

---

## READ THIS BEFORE YOU TOUCH ANYTHING

**What this does.** The council has five scoring slots. Two judges — MOMENTUM and REVERSION — are fighting over one of them. Only one scores each cycle; the other is discarded. This gives REVERSION its own slot so both score, every cycle.

**Why it matters to the strategy.** REVERSION is the judge that scores reversal setups — the CHoCH → retest → close trade. Because it shares a slot, it gets dropped in exactly the conditions where reversals occur. The bot has been half-blind to one of its two trade types by accident of wiring.

**The line that will break everything if you skip it.** There is a hardcoded `5.0` in the threshold maths. Add a sixth slot without fixing it and every threshold silently gets *easier* — the bot fires on weaker signals across the board and nothing tells you. **Segment 4. Do not skip it.**

**Order.** Segments 1 → 6, one commit. Do not split across commits — the bot will crash between them.

**Prerequisites — must be live AND soaked:**
- File 1 Fix 0, 1, 2 deployed
- The three proof flags flipped on and behaving

⚠️ **If Fix 2 is not live, stop.** REVERSION would get a scoring lane while the silent-zone gate is still zeroing the generator that fills it. An empty slot, and a wasted soak.

**Flag:** `six_slot_judges_enabled`, default `false`. Flag off = byte-identical to today.

---

## SEGMENT 1 — Add the sixth key

**Where:** `_get_aggregated_signal_impl`, the two scorecard dicts.

**Why:** these dicts are the scoreboard. A judge with no key has nowhere to write.

**Find:**
```python
            buy_scores = {
                "trend": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "pattern": 0.0,
                "volume": 0.0,
            }
```

**Replace with:**
```python
            buy_scores = {
                "trend": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "pattern": 0.0,
                "volume": 0.0,
                "reversion": 0.0,   # Six-slot split: REVERSION's own slot
            }
```

**Then do the identical change to `sell_scores` directly below.**

⚠️ **TWO dicts. Change BOTH.** Changing one gives you a bot scoring buys and sells on different scoreboards — it looks like it works, which is the worst kind of bug.

**CHECK 1:** grep `"volume": 0.0,` → exactly two hits, each followed by a `"reversion": 0.0,`.

---

## SEGMENT 2 — Give REVERSION its own weight

**Where:** just after the `w_volume` assignment in the dynamic-weights block.

**Why:** every judge needs a ceiling. REVERSION never had one — it borrowed MOMENTUM's.

**Add after the last `w_volume = ...`:**
```python
            # Six-slot split: REVERSION's own weight. Flag OFF → 0.0, so every
            # sum below is arithmetically identical to today's five-slot maths.
            _six_slot = bool(
                (getattr(_composite_state, "phase_config", {}) or {}).get(
                    "six_slot_judges_enabled", False
                )
            )
            w_reversion = self.w_reversion if _six_slot else 0.0
```

**In `__init__`, alongside the other `self.w_*` defaults:**
```python
        # Six-slot split: REVERSION's standing weight. 1.0 matches MOMENTUM —
        # the two are peers: one scores continuation fuel, one scores reversal
        # conviction.
        self.w_reversion = self.config.get("w_reversion", 1.0)
```

### 2b — The ambiguous-regime trim ⚠️ DECIDED

**Find the SLIGHTLY-regime reweight block:**
```python
            if consensus_regime in ["SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"]:
                w_momentum = 0.75
                w_structure = 1.5
                w_pattern = 0.75
```

**Add one line inside it:**
```python
            if consensus_regime in ["SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"]:
                w_momentum = 0.75
                w_structure = 1.5
                w_pattern = 0.75
                # Six-slot split: REVERSION trims with MOMENTUM in an ambiguous
                # regime. SLIGHTLY means the CONTEXT is less trustworthy — every
                # judge reading that context gets trimmed. Leaving REVERSION at
                # full weight while MOMENTUM drops would quietly make reversal
                # the loudest voice exactly when the picture is least clear.
                if _six_slot:
                    w_reversion = 0.75
```

⚠️ **`_six_slot` must be defined ABOVE this block.** Move the Segment 2 `_six_slot` derivation above the reweight if the ordering doesn't already allow it.

**CHECK 2:** start in paper mode, flag OFF. Runs exactly as before; `w_reversion` = 0.0.

---

## SEGMENT 3 — Call both judges every cycle

**Where:** the momentum/reversion dispatch fork.

**Why — the heart of the change.** Today an `if/else` picks *one*. A trending market sends every cycle to MOMENTUM and REVERSION never runs. Two unconditional calls replace it. MOMENTUM asks "does this move have fuel?" REVERSION asks "has a reversal proven itself?" Different questions; both deserve an answer.

**Replace the whole `if is_breakout_mode or is_trending_regime:` / `else:` block with:**

```python
            # ── Six-slot split: both judges score, every cycle ──────────────
            if _six_slot:
                buy_scores["momentum"], sell_scores["momentum"], momentum_exp = (
                    self._judge_momentum_bidirectional(
                        df, is_bull, is_breakout_mode, w_momentum, adx,
                        governor_data=governor_data,
                    )
                )
                buy_scores["reversion"], sell_scores["reversion"], reversion_exp = (
                    self._judge_reversion_bidirectional(
                        df, w_reversion, governor_data=governor_data,
                        mr_signal=mr_signal, mr_conf=mr_conf,
                    )
                )
                buy_explanations.append(reversion_exp["buy"])
                sell_explanations.append(reversion_exp["sell"])
            else:
                # Legacy five-slot path — unchanged, byte-identical to today.
                if is_breakout_mode or is_trending_regime:
                    momentum_result = self._judge_momentum_bidirectional(
                        df, is_bull, is_breakout_mode, w_momentum, adx,
                        governor_data=governor_data,
                    )
                    buy_scores["momentum"], sell_scores["momentum"], momentum_exp = momentum_result

                    _pc_mrreach = getattr(_composite_state, "phase_config", {}) or {}
                    if _pc_mrreach.get("mr_reversal_scoring_in_trend_enabled", False) and mr_signal != 0:
                        _rev_buy, _rev_sell, _rev_exp = self._judge_reversion_bidirectional(
                            df, w_momentum, governor_data=governor_data,
                            mr_signal=mr_signal, mr_conf=mr_conf,
                        )
                        if _rev_buy > buy_scores["momentum"]:
                            buy_scores["momentum"] = _rev_buy
                            momentum_exp["buy"] = _rev_exp["buy"]
                        if _rev_sell > sell_scores["momentum"]:
                            sell_scores["momentum"] = _rev_sell
                            momentum_exp["sell"] = _rev_exp["sell"]
                else:
                    buy_scores["momentum"], sell_scores["momentum"], momentum_exp = (
                        self._judge_reversion_bidirectional(
                            df, w_momentum, governor_data=governor_data,
                            mr_signal=mr_signal, mr_conf=mr_conf,
                        )
                    )

            buy_explanations.append(momentum_exp["buy"])
            sell_explanations.append(momentum_exp["sell"])
```

⚠️ **The existing `buy_explanations.append(momentum_exp["buy"])` pair that followed the old block is included at the bottom above. Do not leave a duplicate behind.**

**CHECK 3:** flag ON → a `REV BUY:` / `REV SELL:` line **every cycle**, including trending markets. That is the whole point.

---

## SEGMENT 4 — Fix the maximum-score maths ⚠️ CRITICAL

**Why — read carefully.** `_achievable_max` is "the best score possible this cycle." Thresholds are percentages of it. Add a sixth judge without updating this and the bot measures a six-slot score against a five-slot maximum — **every threshold loosens and the bot fires on weaker setups.** Nothing surfaces it.

**Find:**
```python
            _achievable_max = w_trend + w_structure + w_momentum + w_pattern + w_volume
```

**Replace:**
```python
            # w_reversion is 0.0 when the flag is off → arithmetically identical
            # to the five-weight sum in legacy mode.
            _achievable_max = (
                w_trend + w_structure + w_momentum + w_pattern + w_volume + w_reversion
            )
```

**Find:**
```python
            _buy_required_pct  = _buy_threshold  / 5.0
            _sell_required_pct = _sell_threshold / 5.0
```

**Replace:**
```python
            # The hardcoded 5.0 assumed the weights always sum to 5.0 — true by
            # coincidence in five slots, false the moment a sixth weight exists.
            _rq_denom = _achievable_max if _achievable_max > 0 else 5.0
            _buy_required_pct  = _buy_threshold  / _rq_denom
            _sell_required_pct = _sell_threshold / _rq_denom
```

**CHECK 4 — by hand, do not skip.** Flag OFF, print `_achievable_max` for one cycle. Must equal exactly what it did before (5.0 in normal regimes). Off by any amount → stop and report.

---

## SEGMENT 5 — Show the sixth judge

**Why:** the dashboard, Telegram and the funnel logger read these. Miss them and REVERSION scores invisibly — it works, but nobody can see it, which makes the soak worthless.

**`judge_scores` in `details` — add:**
```python
                    "reversion": _judge_scores_src.get("reversion", 0.0),
```

**`judge_weights` in `details` — add:**
```python
                    "reversion": w_reversion,
```

**CHECK 5:** flag ON → the log scorecard shows six judges.

---

## SEGMENT 6 — Correct the correlation registry

**Why:** this registry stops the bot counting the same evidence twice. Two judges reading the same underlying signal are not two independent opinions. Four entries are currently wrong.

**Replace the whole registry:**
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

**What changed:**
- **`spring`** on PATTERN and REVERSION — both read the same spring detector. Marked independent; they aren't.
- **`mr_features`** on MOMENTUM — it borrows the MR strategy's feature pipeline for RSI.
- **`brc`** on TREND and REVERSION — both read the proof. They cannot fire together (the proof is tagged for one or the other), but the registry should say so rather than rely on it.

**CHECK 6:** no other code change — `_effective_vote_count` reads this dict directly.

---

## CONFIG

Add to `phase_config` in **all three** config files:
```json
"six_slot_judges_enabled": false
```

---

## VERIFICATION RECORD

### PASS 1 — FORWARD

- **S1:** every downstream reader uses `.get(key, 0.0)`; a new key cannot raise. ✓
- **S2:** `w_reversion` = 0.0 when flagged off → all sums unchanged. ✓
- **S2b:** REVERSION trims with MOMENTUM in ambiguity — consistent treatment of a less-trustworthy context. ✓
- **S3:** flag ON = both judges unconditionally; flag OFF preserves the existing fork *including* the `mr_reversal_scoring_in_trend_enabled` branch already in the repo. Nothing deleted. ✓
- **S4:** `_achievable_max` grows only when `w_reversion > 0`; `_rq_denom` guards divide-by-zero. ✓
- **S5/S6:** additive only; shares still sum to 1.0 per judge. ✓

### PASS 2 — BACKWARD (against the repo)

- **Five-key sites: FOUR, not three.** `buy_scores`, `sell_scores`, `judge_scores`, `judge_weights`. All four covered. ✓
- **`_effective_vote_count` confirmed** to walk `registry.get(judge, ...)` — picks up `reversion` automatically once it scores. ✓
- **`_buy_contributing` confirmed** built by iterating the registry, not a hardcoded list — no edit needed. ✓
- **`_judge_reversion_bidirectional` signature confirmed** `(df, weight, governor_data, mr_signal, mr_conf)` — Segment 3's call matches. ✓
- **Threshold denominator confirmed** hardcoded `5.0` at two sites; both fixed. ✓
- **SLIGHTLY reweight block confirmed** at `w_momentum=0.75, w_structure=1.5, w_pattern=0.75` — Segment 2b inserts correctly. ✓
- **Registry already contains a `reversion` entry** — it was simply never reachable as a contributing judge. ✓

---

## WHAT TO EXPECT

- **Flag OFF:** nothing changes.
- **Flag ON:** a `REV` line every cycle; totals rise slightly (six contributors); `_achievable_max` ≈ 6.0. **Trade frequency should NOT jump** — the percentage maths holds the bar in place. **If it jumps, Segment 4 was applied wrong. Flag off and report.**
- **Watch for:** does REVERSION score in trending markets? Grep `REV BUY: ✅` / `REV SELL: ✅` in a trending session. Under the old wiring that was structurally impossible.
