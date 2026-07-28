# TBOT — WHAT WE FOUND, AND WHAT WE NEED TO CHECK NEXT

**For:** Stephen · **From:** Desire + Claude · **Date:** 28 Jul 2026
**Replaces:** `TBOT_BACKTEST_ANALYSIS_CONSOLIDATED.md` and `TBOT_FURTHER_INVESTIGATIONS.md` — one document instead of two.
**Built on:** your report *"Why trades dropped, and what's actually gating the system"* (GOLD, 5,874 candles)

---

# READ THIS FIRST

**This is not a list of things to fix.** It is a list of things to *find out*.
Nothing in here should be coded as a fix yet. Section 7 is the only part with
code in it, and that code only adds log lines — it changes no behaviour.

**Your report was good.** It was honest about what it couldn't answer (the n=1
problem), it measured the right things, and its main finding — *MR almost never
proposes anything* — is the correct question to be asking. This document mostly
agrees with you and then digs underneath it.

**Where I was wrong, I've said so.** Section 6 lists my mistakes by name,
including the one where your diagnosis beat mine outright. Please read that
section — it tells you how much to trust the rest.

**How to use this:**
1. Section 5 has four questions you can answer in minutes. Start there.
2. Section 7 is logging to add. It's the actual investigation tool.
3. Section 8 is the measurements to run once the logging has data.
4. If anything here contradicts what you see in the code, **stop and tell us.**
   You know this codebase better than a search does.

**Three labels used throughout:**
- **SURE** — I read it in the actual code or your PDF this session
- **TOLD** — it came from an earlier chat and nobody re-checked it
- **DON'T KNOW** — nobody has looked yet

**How this was checked:** twice, from both ends. Once forwards — reading the code
to see what it's *supposed* to do. Once backwards — taking your report's numbers
and asking what code could have produced them. Both passes were done. The
backwards pass is what found most of Section 4.

**One gap you should know about:** I read the code through the project's file
search, which gives me the file and the exact text but **not line numbers**. So
citations below say "in this file, this block" rather than "line 847". If you
send a repo zip, we can tighten that.

---

# 1. THE ONE-PARAGRAPH SUMMARY

Your report asked why trades dried up and found that MR almost never proposes a
trade. We think we know why, and it isn't a threshold or a tuning problem.
**The bot can only track one setup at a time, and your own numbers show that slot
is occupied on 95% of candles.** When a reversal appears while a continuation
setup is still sitting in the slot, the reversal is never created at all. MR
never sees it. Nothing logs it. On top of that, setups that *do* get created can
be labelled with the wrong type — and sometimes the wrong direction — and the
proof check forgets the retest after 8 candles, which may be doing a lot of the
work behind your "80% fail at close-through" number.

---

# 2. YOUR NUMBERS — ALL CHECKED, ALL CORRECT

**SURE.** I read the PDF directly. Every figure holds up:

| What | Number |
|---|---|
| Sample | 5,874 GOLD candles |
| Anchor fix — GOLD MR_REV confirmations | 64 → 390 (2.5×–6× across all assets) |
| MR_REV confirmations | 390 (62%) |
| TF_CONT confirmations | 237 (38%) |
| TF firing rate | 4.2% |
| TF_CONT proof availability | ~4.0% |
| MR_REV — sample / retest-no-close / confirmed | 4,173 / 2,017 (48%) / 390 (9%) |
| TF_CONT — sample / retest-no-close / confirmed | 1,416 / 1,136 (80%) / 237 (17%) |
| Compression bypass | 1 extra signal; BRV-NONE 524 → 2,256 |

**Two small things about the report itself:**

- **It ends mid-sentence.** The last line stops at *"...loosening compression"*.
  Is there a page 2? If there is, we're working from half a document.
- **It doesn't say which flags were on during the run.** That matters — see
  Section 3, item C.

**The 293-vs-627 puzzle is solved.** An earlier figure said BRC fired 293 times
in 5,918 candles, and your report says 627. Those look contradictory but they're
not. Before the anchor fix: 64 MR_REV + about 229 TF_CONT ≈ 293. After: 390 +
237 = 627. TF_CONT barely moved (229 → 237), which is exactly right — your fix
only touched MR_REV's reference. **The two runs are pre-fix and post-fix of the
same thing.** Can you confirm that in one line?

---

# 3. THREE THINGS IN YOUR REPORT THAT MEAN MORE THAN THEY LOOK

## A. The counts are per-candle, not per-proof — and this changes several conclusions

**SURE.** Your Measurement 3 header says it: *"every cycle with a resolved
reference."* That's the key word. **Cycle, not setup.**

Two things follow.

**First — the slot is almost always full.** 4,173 + 1,416 = **5,589 candles with
a live setup, out of 5,874 candles. That's 95%.** So the "one setup at a time"
limit isn't occasionally getting in the way. It's in the way almost permanently.
This is the strongest single piece of evidence in the whole report and it's
sitting there by accident. More on it in Section 4.

**Second — the confirmation counts are inflated.** BRC re-confirms the same proof
on every candle while it still holds. That's exactly why you built `brc_age` —
we watched one level confirm across six different closes. So **390 and 237 are
"candles where a proof was true", not "number of proofs".** The real number of
distinct proofs is smaller, and we don't know by how much.

**Why this matters practically.** We've been saying "about 6 proof events a day
across six assets — that's a workable rate." That was used to justify how much
score a proof is worth in the judges. If proofs actually last 5 candles on
average, the real rate is closer to one a day, and that justification needs
redoing. **You already have the field that answers this** — count only the
confirmations where `brc_age == 0`. That's one counter (see Section 7, item 6).

## B. Build 3 is live, has no flag, and it changed the funnel

**SURE — I found it in `council_aggregator.py` this session.** The ownership
block is there (`_buy_has_thesis` / `_sell_has_thesis` / `_opposing_theses`, with
`no_thesis_backing` / `outvoted` / `thesis_conflict`), and so is the tombstone
comment where the MR lean conflict gate used to be. The `main.py` mirror is
removed too.

This was intentional — it shipped unflagged because it's a correction, not a
feature. Just flagging it because it's the one recent change that altered live
behaviour immediately, and your report says it's what exposed the MR silence.

**Credit where it's due:** ownership blocks are the *only* part of this whole
chain that says anything out loud. They log the reason, both engine signals, and
both totals, and they reach the funnel and shadow. Everything else in this
document is silent. That's the standard the rest should be held to.

**Still unchecked:** Build 4 (shadow engine alignment). Nobody has verified it in
any session. **DON'T KNOW.**

## C. "TF's gate is working as designed" — this depends on a flag we don't know the state of

Your reading was: TF fires 4.2%, its proof is available ~4.0%, so the gate is
sized right, not too tight.

**If `tf_brc_gate_enabled` was ON during the run**, that comparison can't tell us
anything — TF is *gated on* that proof, so of course the two match. It's like
saying "the door opens exactly as often as it's unlocked." True, but it doesn't
tell you whether the lock is too strict.

There's also a small oddity: **4.2% is higher than ~4.0%.** If the gate was on,
TF shouldn't be able to fire more often than proof is available. Probably
rounding or two different denominators — but worth knowing which.

**If the flag was OFF**, then the match is genuinely interesting: it would mean
TF's indicator setups naturally happen at about the same rate as structural
proof, independently. That's a real observation.

**So: which flags were on?** Full `phase_config` state for the run, plus what the
two percentages are measured against.

**Either way, "not too tight" needs a different measurement:** what did the
signals the gate *blocked* go on to do? That isn't measured anywhere yet.

---

# 4. WHAT WE THINK IS ACTUALLY BROKEN

All of these are **SURE** — read in the code this session, in
`composite_state_builder.py` unless stated.

## 4.1 🔴 The bot can only track one setup at a time — and it's full 95% of the time

**The code.** The birth check runs only when the slot is empty:

```python
# ---- STEP 3: birth check (only if nothing is alive) ----
if _cur is None:
```

A setup only dies for three reasons:
1. Livermore 1H flips to the opposite camp
2. `failed_breakout`
3. An opposing BOS

**A CHoCH against the live setup does not kill it.** And nothing dies of old age.

**What that means in trading terms.** USTEC breaks out upwards. A TF_CONT setup
is born and takes the slot. Price then stalls, rolls over, and prints a clean
lower high — a CHoCH. That is precisely the reversal MR exists to trade. But
Livermore hasn't flipped camp yet and no opposing BOS has printed, so the long
setup keeps the slot. **The reversal setup is never created.** MR doesn't score
it low. It doesn't exist.

**Why we think this is the answer to your question.** Your report asks why MR
almost never proposes. A reversal that can't be born can't propose. And your own
95% occupancy figure says the slot is nearly always taken.

**Nothing logs this.** There is no line anywhere saying "a setup wanted to be
born and couldn't."

## 4.2 🔴 A setup can be given the wrong type — and sometimes the wrong direction

**The code**, in the birth block:

```python
if getattr(state, "bos_bullish", False):
    _born = {"kind": "TF_CONT", "dir": 1}
elif getattr(state, "bos_bearish", False):
    _born = {"kind": "TF_CONT", "dir": -1}
# CHoCH takes precedence when both appear...
if getattr(state, "choch_bullish", False):      # ← plain if, not elif
    _born = {"kind": "MR_REV", "dir": 1}
elif getattr(state, "choch_bearish", False):
    _born = {"kind": "MR_REV", "dir": -1}
```

The second block is a plain `if`. So on any candle where a BOS **and** a CHoCH
both fire, the CHoCH overwrites the BOS entirely.

**Type gets overwritten.** A continuation becomes a reversal.

**And direction can flip.** A *bullish* BOS on a candle that also has a *bearish*
CHoCH ends up as `dir = -1`. The setup is now pointing the opposite way.

**In trading terms.** GOLD is trending up and breaks a swing high — that's a long
continuation. On the same candle a small lower high prints. The bot records: short
reversal. Everything downstream that reads `brc_kind` now believes it's looking at
the opposite trade.

**Why this matters to your parked finding.** You found that the trades Build 2's
kind gate removed had a *better* win rate than the ones that survived. If those
trades were actually good continuations wearing a reversal label, then the gate
isn't wrong — **it's being fed bad labels.** That's a very different fix.

**Your comment says the precedence is deliberate**, and that's fair — CHoCH being
the earlier warning is a reasonable design call. Desire will decide on that. But
**the direction flip isn't defensible under any reading of it.** That part looks
like an accident.

## 4.3 🔴 The proof check forgets the retest after 8 candles

**The code.** The retest window is `iloc[-9:-1]` — the 8 candles before the
current one. It slides forward every candle. It is **not** pinned to where the
break happened.

**In trading terms.** GOLD breaks above 3,340. Two candles later price taps 3,340
and holds — a textbook retest. Then it chops sideways for eleven candles before
finally closing decisively above. Any trader would call that a valid, slow
break-retest-hold. The bot calls it nothing — by then the tap has slid out of the
8-candle window and can never be seen again.

**Combine that with 4.1.** The setup is still alive (nothing killed it), still
holding the only slot, and now permanently unable to prove itself.

**This is our main suspect for your 80% figure.** Your Measurement 3 says 80% of
TF_CONT "retested but never closed through." That could be three different
things:
- The market genuinely rejected the retests — a real strategy finding
- The bot forgot the retest before the close arrived — a bug
- A handful of stuck setups counted once per candle for dozens of candles — a
  counting artefact (see 3A)

**All three produce the same 80%.** They need separating before anyone changes
the close-through rule. Loosening the core proof standard to work around a
forgetting bug would be a bad trade.

## 4.4 🟠 A retest and close inside one candle arrives late — and needs a fourth beat

**First, a good thing, now confirmed.** `main.py` drops the in-progress candle
before signals are generated (*"signal generation must only use confirmed, closed
candle data"*). So BRC only ever reads finished candles. **No mid-hour flicker.**
That's correct and worth knowing.

**But:** the retest window excludes the current candle, so a candle can't check
itself.

- **Hour N** — price wicks down to the level and closes back above it. BRC looks
  at hours N-8 to N-1 for the retest. Hour N isn't in that list. No proof.
- **Hour N+1** — the window slides and now includes hour N, so the wick is finally
  seen. But the close test has moved too: it now tests **hour N+1's** close.

**So the proof isn't lost — it's one candle late, and it quietly requires a
fourth beat: the next candle must also close beyond the level.**

**In trading terms.** GOLD breaks 3,340 and pulls back. At 14:00 it wicks to
3,340 and closes 3,348 — a clean sweep and reclaim. Nothing happens. At 15:00 it
closes 3,352 — now it confirms, but the entry is 12 dollars above where the proof
actually happened, with the stop still down at 3,340. **That's chasing** — the
thing `brc_age` was built to prevent, arriving through a different door.

**And the genuine miss:** if 15:00 closes at 3,336 instead, the pattern completed
perfectly at 14:00 and never registers at all.

**Note this pulls the opposite way to 4.3.** 4.4 wants the current candle
*included*; 4.3 wants the window *pinned to the break* instead of sliding. One
change could address both — which is exactly why we need the measurement in 8.2
before touching it.

## 4.5 🟠 A "retest" can be price action from before the break

The check is a plain `any()` over the window. Nothing requires the touch to have
happened **after** the break.

**In trading terms.** BTC chops around 64,140 for hours, wicking that level over
and over with no direction. Then it finally breaks upward. On the very next
candle the bot looks back 8 candles, sees plenty of touches at 64,140 from the
chop, sees the close above, and declares break-retest-close proven. **There was
no retest.** That was noise from before the break.

**Why this matters:** 4.3 makes the check too strict and 4.5 makes it too loose.
In an aggregate number like "80%" they partly cancel out and hide each other.

## 4.6 🟡 Mode 2 has a hard CHoCH gate that isn't behind any flag

In `mean_reversion.py`, `_mode2_counter_trend` requires `choch_detected` and
returns 0 if it's absent — **before** the flag-gated BRC check runs. So Mode 2's
real requirement is CHoCH *and* (when flagged) an MR_REV proof. Worth knowing
before anyone reasons about MR silence purely from flag settings.

## 4.7 🟡 The freeze-at-birth fix was never built — and that's probably fine

An earlier plan said to capture the retest reference onto the setup at birth
(`setup_ref_price`). The code doesn't do that — `_born.update({...})` has no
`ref_price` field. You solved the problem a different way, with the 1H anchor.

**That was the better call** (see Section 6). We're only raising it because
nothing should quietly disappear off the list. Desire decides whether to retire
it formally. One thing in its favour: freezing the reference at birth would also
protect against 4.3's sliding window.

## 4.8 🟡 Almost none of this makes a sound

For every problem above we asked: does anything alert, or does it just quietly
do nothing?

| Problem | Does anything say so? |
|---|---|
| Wrong type / flipped direction (4.2) | **No** — the log prints the wrong label confidently |
| Setup couldn't be born (4.1) | **No** — completely silent |
| Setup too old to ever confirm (4.3) | **No** — age is tracked, nothing warns |
| Same-candle retest arriving late (4.4) | **No** |
| Retest that predates the break (4.5) | **No** — counts as valid |
| Ownership rule blocks (Build 3) | **YES** — logs reason, both signals, both totals |

**Five silent, one loud.** Build 3 shows what good looks like. Section 7 brings
the rest up to that standard.

---

# 5. FOUR QUESTIONS FOR YOU — MINUTES EACH, DO THESE FIRST

## Q1. How many trades were in the kind-mismatch finding?
Your report says the removed trades had a better win rate and better PnL, but
gives no count anywhere. **Ten trades and two hundred trades mean completely
different things.**

**Send:** number of trades, wins, losses and total PnL — for the removed group
**and** the surviving group.

## Q2. Confirm the two runs
Was 293 fires / 5,918 candles the **pre**-anchor-fix run, and 627 / 5,874 the
**post**-fix run? (The arithmetic works — see Section 2.) One line is enough.

## Q3. Which flags were on during the backtest?
Full `phase_config` state for that run. Plus: what are "4.2%" and "~4.0%"
measured against — every candle, or every evaluated cycle?

## Q4. Is there a page 2?
The PDF stops mid-sentence at *"...loosening compression"*. If there's more,
send it.

---

# 6. WHERE I GOT THINGS WRONG

**You should know how much to trust the rest of this document.**

**1. I was wrong about the anchor, and you were right.** I said the Livermore
anchor gets *consumed* when price crosses from NATURAL to SECONDARY. That was
overcomplicated and wrong. Your explanation is the correct one: those fields are
**4H-only**, and a 4H machine sitting in MAIN_UP or MAIN_DOWN has no natural
anchor at all. Not consumed — never existed. A 1H reversal was being measured
against something that mostly wasn't there. **Your fix and your diagnosis both
beat mine.**

**2. I said same-candle retest-and-close was "invisible."** It isn't. It's one
candle late and needs an extra confirmation (4.4). Overstated.

**3. I called a 2× number gap an unexplained discrepancy.** It reconciles cleanly
as pre-fix vs post-fix (Section 2). I should have done that arithmetic first.

**4. I've been quoting proof frequency as if the counts were distinct events.**
They're per-candle counts (3A). Several sizing arguments were built on that and
now need redoing.

**Three things you got right that are worth recording:**
- **`brc_age` keying off the bar timestamp** instead of just incrementing. The
  bot recomputes ~22 times an hour; a naive counter would have aged every proof
  22× per candle and made the field meaningless. You also handled the backtest's
  RangeIndex-vs-datetime difference. That was better than the spec.
- **Giving the MR solo threshold its own config key** instead of hardcoding it.
- **The anchor diagnosis**, above.

---

# 7. STEP ONE — MAKE THE SILENT THINGS TALK

**This is the actual investigation.** Six log lines. **No behaviour changes, no
flags, nothing gated.** Observation only.

Everything in Section 8 needs the data these produce, so this comes first.

**1. Both signals on one candle.** When `bos_*` and `choch_*` are both true at
birth, log both, plus which one won and what type/direction was recorded.
→ *feeds 8.1*

**2. A setup that couldn't be born.** When a BOS or CHoCH fires while the slot is
occupied, log: what wanted to be born (type + direction), what's currently in the
slot (type + direction), and **how old the incumbent is**.
→ *feeds 8.3 — probably the most important line in this list*

**3. Direction flipped.** When the CHoCH overwrite reverses the direction the BOS
had set, log it as an inversion specifically. Don't bury it in item 1.
→ *feeds 8.1*

**4. Setup past its proof window.** When `setup_age` goes beyond the retest
window, log once: this setup can no longer confirm.
→ *feeds 8.4*

**5. Near-miss on proof.** When the retest passed but the close-through failed,
log the gap between the close and the reference. How close was it?
→ *feeds 8.2*

**6. Distinct proofs vs repeats.** Count confirmations where `brc_age == 0`
separately from re-confirmations of a proof already counted.
→ *feeds 8.5 — this is the one that fixes our frequency numbers*

**Run for a week**, or re-run the backtest with these in place, whichever gets
data faster.

---

# 8. STEP TWO — THE MEASUREMENTS

## 8.1 Is the kind gate wrong, or just fed bad labels?

**Question.** How often do a BOS and a CHoCH land on the same candle? And of the
kind-mismatched trades, how many were born on such a candle? Separately: how
often does the overwrite flip the direction?

**How.** Log at birth (item 1 above), then match against the mismatch list.

**Decided in advance:**
- **Mostly dual-signal candles** → the label is the bug. Fix how precedence
  works. **Keep the gate.**
- **Mostly not** → those trades really were mismatched and still won. Then the
  rule itself — "proof type must match trade type" — needs rethinking from the
  strategy up.
- **Direction flips** are a problem at any rate. Count them either way.

**This is the cheapest test in the document and it settles the biggest fork.**

## 8.2 Is the 80% the market, the sliding window, or the counting?

**Question.** Take TF_CONT's 1,136 "retested, no close-through" candles.
(a) How many **actual setups** is that? (b) Of those setups, how many *did*
eventually close through, but with more than 8 candles between the retest and the
close?

**How.** Per setup, record three candle numbers: where it broke, where it first
retested, where it first closed through (if ever). Then bucket:
- **A** — never closed through → the market really did reject it
- **B** — closed through, but the gap was over 8 candles → **the bot forgot**
- **C** — closed through within 8 candles and still wasn't flagged → a third bug
  we haven't found

**Decided in advance:**
- Mostly A → your reading is right, close-through is a real strategy question
- Meaningful B → **don't touch the close-through rule.** Fix the window, re-run
- Any C at all → stop and trace it

**Nothing about the close-through rule gets decided until this comes back.**

## 8.3 How often does a setup get blocked from being born?

**Question.** Per asset per day: how many times did a BOS or CHoCH fire while the
slot was occupied? Split by whether the blocked setup pointed the **same** way as
the incumbent (harmless) or the **opposite** way (a missed reversal).

**Also capture the incumbent's age.** If the thing holding the slot is routinely
older than 8 candles, then 4.1 and 4.3 are compounding — the slot is being held
by something that can no longer confirm anyway.

**Decided in advance:**
- Rare, mostly same-direction → 4.1 is theoretical, log it and move on
- Common and opposite-direction → the tracker needs either more than one slot, or
  CHoCH-against needs to become a death condition. **That's Desire's call, not
  ours.**

## 8.4 How long do setups actually live?

**Question.** Distribution of `setup_age` at death. What fraction are still alive
past 8, 20, 50 candles? What's the mix of death reasons?

**Why it matters beyond the obvious:** `brc_max_age` was set to 20 deliberately
loose, so the real distribution would show up in the logs and the right number
could be picked from evidence. **That evidence has never been collected.** This
is where it comes from.

## 8.5 How many distinct proofs are there really?

**Question.** For every measurement in the report, what's the count of *distinct
events* alongside the count of candles? Distinct proofs = confirmations at
`brc_age == 0`. Distinct setups = births.

**Why.** See 3A. Every frequency number we've used is a per-candle rate. If the
real event rate is much lower, then how much score a proof is worth in the judges
needs recalculating.

## 8.6 Do the proofs actually make money?

**Question.** For each distinct confirmed proof: enter at the confirming candle's
close, put the stop where the structure says, and check what happened next. Split
by type, by asset, and by `brc_age` at entry.

**Why this is the big one.** The whole design rests on the idea that a completed
break-retest-close is a **profitable** event. We have measured how *often* it
happens. **We have never measured whether it pays.** Three gates, both judges'
proof scoring, and the entire "proof is the trigger" architecture are riding on
an assumption nobody has tested.

**Order matters:** run 8.7 first, or false proofs will pollute the result.

**Bonus:** the `brc_age` split also tells us the right value for `brc_max_age`,
closing 8.4 in the same pass.

## 8.7 How many "proofs" used a retest from before the break?

**Question.** Record the break candle and the candle of the touch that satisfied
the retest. Count where the touch came **first**.

**Decided in advance:** rare → leave it. Common → a chunk of the 627 proofs
aren't proofs, and 8.6's result can't be trusted until they're excluded.

## 8.8 How many TF signals actually became trades?

**Question.** Full funnel counts: signal generated → passed the proof gate →
scored by the council → cleared the threshold → **passed the ownership rule** →
executed.

**Why.** Every frequency number we have is measured *before* the judges. Build 3
is now a mandatory stage in that funnel, and it already logs its blocks — so this
is mostly counting what's already there.

**Watch the ownership rows in the shadow scorecard specifically.** If blocked
signals consistently show positive forward PnL, the rule is killing winners and
we revisit it. That check was built into Build 3 on purpose.

---

# 9. OLDER ITEMS STILL ON THE LIST

Nothing gets dropped just because it's inconvenient.

| Item | Where it stands |
|---|---|
| The rest of the 1H/4H mismatch work | **Partly done.** Your anchor fix was one case of this and was worth 2.5×–6×. Still untouched: the TREND judge reads 1H data to answer a 4H question; REVERSION's dispatch and its judge gate on different timeframes; the 1H Livermore calibration was derived entirely from 4H data. Worth a systematic sweep — for every field a judge reads, which timeframe wrote it vs which one the reader assumes. |
| GOLD's RSI zones look inverted | **TOLD, unchecked — and now more urgent.** GOLD's bullish/bearish RSI bands reportedly mirror every other asset. **GOLD is the backtest asset.** If its momentum judge scores backwards, part of your results are GOLD-specific noise. One side-by-side config table settles it. |
| EMA's hardcoded 0.75 confidence path | **TOLD, unchecked.** Reportedly fires after scoring has already failed. Matters because Build 2 blocks EMA from firing alone and Build 3 says EMA can't own a trade — if a hardcoded path injects confidence upstream of those, both blocks are leakier than designed. |
| Build 4 (shadow engine alignment) | **DON'T KNOW.** Never verified. |
| Build 3 leftover checks | Grep `mr_lean` across `src/` should return 0; the three config keys should be gone from all three files; line numbers for the edits. |
| `mr_reversal_scoring_in_trend_enabled` | **Conflict.** Our ledger says "outside agreed scope, decision pending." Last session's notes say it's **ON**. One of those is wrong. Desire needs to settle it. |
| Monitoring layer / three-lives retry | **On ice** — Desire's call, don't resurface. |
| A3, A11, A12, A13, B2, B6 | **On ice** — same. |
| Weekend patch set, Monday set, trade-management work | **Held** until Desire asks. |
| Pre-live blockers | Still open and still non-negotiable before going live: the silent reconciler pops (the survivorship-bias fix), three orphaned DB rows, the validation script fix, the MT5 outage Telegram alert, and the dead-code bundle. |

---

# 10. FOR DESIRE — SIX DECISIONS

Claude isn't deciding any of these.

1. **CHoCH precedence** — is the overwrite what you want? (The direction flip is a
   separate matter — that looks like a plain bug either way.)
2. **One setup slot** — should the tracker hold more than one, or should a CHoCH
   against a live setup kill it? Architectural.
3. **The freeze-at-birth fix (4.7)** — retire it formally, or keep it as a second
   layer of protection?
4. **`mr_reversal_scoring_in_trend_enabled`** — ratify it, or switch it off until
   you do?
5. **Page 2** — chase it if it exists.
6. **Order of work** — Section 5, then 7, then 8 is our suggestion. Overrule
   freely.

---

# 11. WHAT WOULD HELP MOST

1. **A repo zip** — so every claim in here gets a real line number instead of
   "in this block"
2. **Page 2 of the report**, if there is one
3. **A log with actual trades in it** — every check so far has been code against
   code, never against live behaviour
4. **Anything in here that contradicts what you see.** You know this codebase
   better than a search does, and you've already been right once where we were
   wrong. Say so and we'll fix the document.

---

**Nothing here gets implemented until Desire asks for it.**
