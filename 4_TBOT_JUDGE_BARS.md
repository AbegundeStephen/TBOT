# 4 — THE SIX JUDGE BARS

**Deploy order: FILE 4 of 4. Last. Everything in Files 1–3 must be live and soaked first.**

Owner: Desire · Developer: Stephen · **File touched:** `src/execution/council_aggregator.py`

---

## WHAT THIS IS

Today each judge is a cascade: `if this → score and stop`, `elif that → score and stop`. The first branch that matches wins and everything after it is never considered. A break at a proven level, with a sweep, near a ladder line, scores exactly the same as a bare break — because the first branch already returned.

**A bar is different.** Each judge builds its score from named pieces that accumulate. Evidence adds up. A setup with four things going for it outscores one with a single thing going for it, which is what a judge is supposed to do.

**Everything is a fraction of `weight`** — the judge's ceiling for that cycle. Weights are dynamic (1.0 or 1.5; trimmed in ambiguous regimes), so fractions travel correctly.

**Flag:** `judge_bars_enabled`, default `false`. OFF = today's cascade, byte-identical.

---

## THE PROOF SEGMENT — how BRC enters the judges

The strategy says the bot waits for break → retest → close. Files 1–3 built that proof and put it on the board. **This is where it finally earns score.**

`brc_kind` routes it, and the routing is exclusive:
- **`TF_CONT`** (born at a BOS) → **TREND's** proof segment
- **`MR_REV`** (born at a CHoCH) → **REVERSION's** proof segment

They cannot both fire on the same bar, so there is no double-count between them.

### ⚠️ The MR double-proof, and how it's resolved — DECIDED

MR Mode 1 **already** requires BRV to pass before it emits a signal. BRV checks the same sweep-and-close on the same anchor that BRC checks. So if REVERSION scored `mr_conf` **and** a BRC segment, it would be paid twice for one event.

**Resolution: REVERSION's proof segment only fires when `mr_signal == 0`.**

That turns the overlap into genuinely new information: *the proof completed, but MR's own entry gates didn't clear.* The proof still earns credit; it just never gets paid twice for the same thing.

**TREND has no such overlap** — TF never consults BRV — so its proof segment fires unconditionally.

---

# BAR 1 — TREND

**Role:** the continuation thesis.

| Segment | Fraction | Reads |
|---|---|---|
| Proof of continuation | **0.40** | `brc_confirmed` + `brc_kind=="TF_CONT"` + direction |
| Driver signal × confidence | **0.45** | TF (or EMA when Unit 3 is off) |
| Livermore agreement | ×1.15 capped | `livermore_state_1h` |
| Slope agreement | ×1.10 capped | `slopes_aligned` |
| Conviction dying | ×0.75 | `conviction_dying` |

```python
    def _bar_trend(self, df, weight, ema_signal, ema_conf, tf_signal, tf_conf,
                   governor_data=None):
        """
        TREND bar. Proof-first: an unproven driver signal caps at 0.45 of the
        bar. Proven continuation is the larger share — that is the strategy
        expressed as arithmetic rather than hoped for.
        """
        cs = (governor_data or {}).get("composite_state") if governor_data else None
        def _g(a, d=None):
            if cs is None: return d
            return cs.get(a, d) if isinstance(cs, dict) else getattr(cs, a, d)

        buy = sell = 0.0
        buy_parts, sell_parts = [], []

        # ── Segment 1: proof of continuation (0.40) ──────────────────
        if _g("brc_confirmed", False) and _g("brc_kind", None) == "TF_CONT":
            _d = int(_g("brc_direction", 0) or 0)
            if _d == 1:
                buy += 0.40 * weight; buy_parts.append("proof")
            elif _d == -1:
                sell += 0.40 * weight; sell_parts.append("proof")

        # ── Segment 2: driver signal (0.45) ──────────────────────────
        _tf_drives = bool((_g("phase_config", {}) or {}).get(
            "tf_drives_trend_judge_enabled", False))
        _sig, _conf, _name = (
            (tf_signal, tf_conf, "TF") if _tf_drives else (ema_signal, ema_conf, "EMA")
        )
        if _sig == 1:
            buy += 0.45 * weight * _conf; buy_parts.append(f"{_name}({_conf:.2f})")
        elif _sig == -1:
            sell += 0.45 * weight * _conf; sell_parts.append(f"{_name}({_conf:.2f})")

        # ── Modifiers: capped multipliers, applied after accumulation ──
        _lsm = _g("livermore_state_1h", None)
        _bull = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
        _bear = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
        if buy > 0 and _lsm in _bull:
            buy = min(buy * 1.15, weight); buy_parts.append("livermore")
        if sell > 0 and _lsm in _bear:
            sell = min(sell * 1.15, weight); sell_parts.append("livermore")

        if _g("slopes_aligned", False):
            if buy >= sell and buy > 0:
                buy = min(buy * 1.10, weight); buy_parts.append("slopes")
            elif sell > 0:
                sell = min(sell * 1.10, weight); sell_parts.append("slopes")

        if _g("conviction_dying", False):
            _cd_dir = getattr(self, "phase_config", {}).get(
                "conviction_dying_directional_enabled", False)
            if _cd_dir and _lsm == "NATURAL_REBOUND":
                buy *= 0.75
                if buy > 0: buy_parts.append("-dying")
            else:
                buy *= 0.75; sell *= 0.75
                if buy > 0: buy_parts.append("-dying")
                if sell > 0: sell_parts.append("-dying")

        buy, sell = min(buy, weight), min(sell, weight)
        return buy, sell, {
            "buy":  f"TREND BUY: {buy:.2f} [{'+'.join(buy_parts) or 'none'}]",
            "sell": f"TREND SELL: {sell:.2f} [{'+'.join(sell_parts) or 'none'}]",
        }
```

**Why 0.45 for the driver, not more.** TF fires on indicator alignment, which happens often. Left as the dominant share it would pin TREND near its ceiling most of the time, and a judge that always votes the same way stops being evidence. Proof is the bigger share because proof is what the strategy trusts.

---

# BAR 2 — REVERSION

**Role:** the reversal thesis.

| Segment | Fraction | Reads |
|---|---|---|
| Proof of reversal *(only when `mr_signal == 0`)* | **0.40** | `brc_confirmed` + `MR_REV` + direction |
| MR signal × confidence | **0.45** | `mr_signal`, `mr_conf`, NATURAL states |
| CHoCH direction | **0.15** | `choch_bullish` / `choch_bearish` |

```python
    def _bar_reversion(self, df, weight, governor_data=None,
                       mr_signal=0, mr_conf=0.0):
        """
        REVERSION bar. Mirrors TREND — equal rigor, both directions.

        The proof segment fires ONLY when mr_signal == 0. MR Mode 1 already
        requires BRV, which validates the same sweep-and-close on the same
        anchor BRC uses — so scoring both would pay twice for one event.
        Gating on mr_signal == 0 makes the segment mean something new:
        the proof completed, but MR's own entry gates did not clear.
        """
        cs = (governor_data or {}).get("composite_state") if governor_data else None
        def _g(a, d=None):
            if cs is None: return d
            return cs.get(a, d) if isinstance(cs, dict) else getattr(cs, a, d)

        buy = sell = 0.0
        buy_parts, sell_parts = [], []
        _lsm = _g("livermore_state_1h", None)

        # ── Segment 1: MR signal (0.45) ──────────────────────────────
        if _lsm == "NATURAL_RETRACEMENT" and mr_signal == 1:
            buy += min(0.45 * weight * mr_conf, 0.45 * weight)
            buy_parts.append(f"mr({mr_conf:.2f})")
        elif _lsm == "NATURAL_REBOUND" and mr_signal == -1:
            sell += min(0.45 * weight * mr_conf, 0.45 * weight)
            sell_parts.append(f"mr({mr_conf:.2f})")

        # ── Segment 2: proof of reversal (0.40) — only if MR stayed silent ──
        if mr_signal == 0 and _g("brc_confirmed", False) and _g("brc_kind", None) == "MR_REV":
            _d = int(_g("brc_direction", 0) or 0)
            if _d == 1:
                buy += 0.40 * weight; buy_parts.append("proof(mr-silent)")
            elif _d == -1:
                sell += 0.40 * weight; sell_parts.append("proof(mr-silent)")

        # ── Segment 3: CHoCH direction (0.15) ────────────────────────
        # F4 gave STRUCTURE the existence; the direction lives here.
        if _g("choch_bullish", False):
            buy += 0.15 * weight; buy_parts.append("choch")
        elif _g("choch_bearish", False):
            sell += 0.15 * weight; sell_parts.append("choch")

        buy, sell = min(buy, weight), min(sell, weight)
        return buy, sell, {
            "buy":  f"REV BUY: {buy:.2f} [{'+'.join(buy_parts) or 'none'}]",
            "sell": f"REV SELL: {sell:.2f} [{'+'.join(sell_parts) or 'none'}]",
        }
```

---

# BAR 3 — STRUCTURE

**Role:** where price is, and what it just broke.

| Segment | Fraction | Needs |
|---|---|---|
| Directional break | **0.30** | F1 |
| Proven level + defense | **0.22** | F2, F3 |
| Zone position (ladder) | **0.18** | live |
| Sweep spring / upthrust | **0.15** | live |
| CHoCH existence | **0.08** | F4 |
| VWAP proximity | **0.07** | live |

```python
    def _bar_structure(self, df, weight, governor_data=None,
                       rt_buy=None, rt_sell=None):
        """
        STRUCTURE bar. Accumulates instead of cascading — a break AT a proven
        level, near a ladder line, with a sweep, now outscores a bare break.
        Under the old cascade all four scored identically.
        """
        cs = (governor_data or {}).get("composite_state") if governor_data else None
        def _g(a, d=None):
            if cs is None: return d
            return cs.get(a, d) if isinstance(cs, dict) else getattr(cs, a, d)

        buy = sell = 0.0
        buy_parts, sell_parts = [], []

        # ── 1. Directional break (0.30) — F1 ─────────────────────────
        if _g("bos_bullish", False) and not _g("failed_breakout", False):
            buy += 0.30 * weight; buy_parts.append("BOS")
        if _g("bos_bearish", False):
            sell += 0.30 * weight; sell_parts.append("BOS")
        if _g("failed_breakout", False):
            sell += 0.30 * weight; sell_parts.append("failed-breakout")

        # ── 2. Proven level + defense (0.22) — F2/F3 ─────────────────
        _def = float(_g("defense_strength", 0.0) or 0.0)
        if _g("level_defended", False):
            _lvl = 0.22 * weight * (0.5 + 0.5 * min(max(_def, 0.0), 1.0))
            _close = float(df["close"].iloc[-1]) if len(df) else 0.0
            _ref = _g("nearby_4h_level", None)
            if _ref is not None:
                if _close >= float(_ref):
                    buy += _lvl; buy_parts.append(f"defended({_def:.2f})")
                else:
                    sell += _lvl; sell_parts.append(f"defended({_def:.2f})")

        # ── 3. Zone position — the ladder (0.18) ─────────────────────
        _dist = _g("activity_ladder_dist_atr", None)
        _side = _g("activity_ladder_side", None)
        if _dist is not None and float(_dist) < 1.0:
            _prox = 0.18 * weight * (1.0 - min(float(_dist), 1.0))
            _lo_t = int(_g("zone_4h_current_lower_tests", 0) or 0)
            _up_t = int(_g("zone_4h_current_upper_tests", 0) or 0)
            if _side == "BELOW":
                buy += _prox * (1.0 + min(_lo_t * 0.05, 0.20))
                buy_parts.append(f"ladder({_dist:.2f}ATR,t{_lo_t})")
            elif _side == "ABOVE":
                sell += _prox * (1.0 + min(_up_t * 0.05, 0.20))
                sell_parts.append(f"ladder({_dist:.2f}ATR,t{_up_t})")

        # ── 4. Sweep (0.15) ──────────────────────────────────────────
        _sweep = int(_g("sweep_direction", 0) or 0)
        if _sweep == 1:
            buy += 0.15 * weight; buy_parts.append("spring")
        elif _sweep == -1:
            sell += 0.15 * weight; sell_parts.append("upthrust")

        # ── 5. CHoCH existence (0.08) — F4, symmetric ────────────────
        if _g("choch_detected", False):
            buy += 0.08 * weight; sell += 0.08 * weight
            buy_parts.append("choch"); sell_parts.append("choch")

        # ── 6. VWAP proximity (0.07) ─────────────────────────────────
        _vwap = _g("vwap_price", None)
        _vd = _g("distance_to_vwap_atr", None)
        if _vwap and _vd is not None and float(_vd) < 0.5 and len(df):
            _c = float(df["close"].iloc[-1])
            if _c > float(_vwap):
                buy += 0.07 * weight; buy_parts.append("vwap-support")
            else:
                sell += 0.07 * weight; sell_parts.append("vwap-resistance")

        # ── Retest tier: multiplicative, not additive ────────────────
        _bt = rt_buy.retest_type if rt_buy is not None else None
        _st = rt_sell.retest_type if rt_sell is not None else None
        if buy > 0:
            if _bt == "WICK": buy = min(buy * 1.15, weight); buy_parts.append("wick")
            elif _bt == "CHASE_HARD": buy *= 0.30; buy_parts.append("chase-hard")
            elif _bt == "CHASE_SOFT": buy *= 0.70; buy_parts.append("chase-soft")
        if sell > 0:
            if _st == "WICK": sell = min(sell * 1.15, weight); sell_parts.append("wick")
            elif _st == "CHASE_HARD": sell *= 0.30; sell_parts.append("chase-hard")
            elif _st == "CHASE_SOFT": sell *= 0.70; sell_parts.append("chase-soft")

        buy, sell = min(buy, weight), min(sell, weight)
        return buy, sell, {
            "buy":  f"STRUCT BUY: {buy:.2f} [{'+'.join(buy_parts) or 'none'}]",
            "sell": f"STRUCT SELL: {sell:.2f} [{'+'.join(sell_parts) or 'none'}]",
        }
```

**The ladder finally scores.** `activity_ladder_dist_atr` and the per-line test counts have been computed every cycle and read by no judge. Segment 3 connects them: closer to a tested line scores more, and a line tested four times scores more than a fresh one.

---

# BAR 4 — MOMENTUM

| Segment | Fraction |
|---|---|
| ADX-scaled base | **0.45** |
| RSI zone confirmation | **0.25** |
| Divergence (regular / hidden) | **0.20 / 0.13** |
| MACD | **0.10** |

Keep the existing internals — including **M1** (no early return, divergence can cut a super-cycle read) and **M2** (`× weight`) from File 3. The bar change is that these four accumulate rather than short-circuit, and each is expressed as a fraction of `weight`.

⚠️ **GOLD RSI ZONES — UNRESOLVED, FLAGGED.** GOLD is the only asset whose bullish and bearish RSI zones are inverted relative to every other asset (bullish 35–47, bearish 53–65; every other asset is bullish-high / bearish-low). No comment explains it. **The bar does not change this.** It is either a long-standing typo or a deliberate mean-reversion read for gold, and only Desire can say which. **Do not "fix" it as part of this build.**

---

# BAR 5 — PATTERN

| Segment | Fraction |
|---|---|
| Distribution / Accumulation | **0.35** |
| Compression / coiled | **0.30** |
| EMA-50 six-state (P2) | **0.25** |
| Squeeze strength | **0.10** |

Accumulation stays modest at 0.35 deliberately — it shares the spring detector with REVERSION, and the registry now records that (`spring` shared source). Modest weight plus registry discount is belt-and-braces on the one genuine overlap in the set.

---

# BAR 6 — VOLUME

| Segment | Fraction |
|---|---|
| Volume ratio (BTC) / spread ratio (MT5) | **0.45** |
| OBV divergence + Livermore top-up | **0.25** |
| Absorption (directional) | **0.20** |
| Order-book wall (BTC only) | **0.10** |
| `vpd_diverging` | ×0.70 |

⚠️ **KNOWN LIMITATION, NOT FIXED HERE.** On all seven MT5 assets, tick volume is unreliable, so this judge scores from **spread ratio alone**. Only BTC gets real volume evidence. The bar does not change that — it is honest about what it has.

**Worth Desire knowing:** a fully-built `VolumeOrderFlowStrategy` exists in the repo (OBV / MFI / CMF / surge / divergence, no trained model needed) and is only consumed by the performance aggregator — which you don't run. It is exactly the real volume evidence this judge lacks. **Wiring it is a separate decision, not part of this build.**

---

## WIRING THE BARS IN

Each bar is a new method alongside the existing judge, selected by flag:

```python
            _bars_on = bool(
                (getattr(_composite_state, "phase_config", {}) or {}).get(
                    "judge_bars_enabled", False
                )
            )

            if _bars_on:
                buy_scores["trend"], sell_scores["trend"], trend_exp = \
                    self._bar_trend(df, w_trend, ema_signal, ema_conf,
                                    tf_signal, tf_conf, governor_data=governor_data)
            else:
                # existing cascade path, unchanged
                ...
```

Repeat for all six. **Do not delete the cascade methods** — they are the emergency revert.

---

## VERIFICATION RECORD

### PASS 1 — FORWARD

- **Proof is the largest single segment in both thesis judges** (0.40 vs a 0.45 driver that only reaches full value at confidence 1.0). Unproven signals cannot max a bar. That is the strategy as arithmetic. ✓
- **TREND and REVERSION are symmetric** — 0.40 proof, 0.45 own-signal. Equal rigor, both directions. ✓
- **MR double-proof resolved structurally** — `mr_signal == 0` makes the segment new information rather than a second payment. ✓
- **STRUCTURE accumulates** — break + level + ladder + sweep now outscores a bare break. ✓
- **Ladder connected** for the first time. ✓
- **CHoCH split honoured** — existence in STRUCTURE (0.08, symmetric), direction in REVERSION (0.15). ✓
- **All segments capped at `weight`;** retest tier multiplicative, never additive. ✓

### PASS 2 — BACKWARD (against the repo)

- **Every field read here exists on the board:** `bos_bullish/bearish`, `choch_bullish/bearish`, `level_defended`, `defense_strength`, `nearby_4h_level`, `activity_ladder_dist_atr`, `activity_ladder_side`, `zone_4h_current_*_tests`, `sweep_direction`, `vwap_price`, `distance_to_vwap_atr`, `slopes_aligned`, `conviction_dying`, `livermore_state_1h`. ✓
- **`brc_*` fields come from File 1.** If File 1 is not deployed they read `False`/`0` and the proof segments contribute nothing — the bars degrade safely rather than crash. ✓
- **`w_reversion` comes from File 2.** Bar 2 is only reachable when the sixth slot exists. ✓
- **F1–F4 prerequisites:** Bar 3 reads `bos_bullish` (F1), scaled `defense_strength` (F2/F3), symmetric CHoCH (F4). **Ship File 3 first or Bar 3 reads the wrong values.** ✓
- **Dual-form reads:** every bar uses the dict-or-dataclass `_g()` helper, matching the pattern used by every existing judge. ✓
- **Retest tier semantics preserved** — WICK confirms, CHASE penalises, CLEAN deliberately excluded (it overlaps the defense segment). ✓

---

## CONFIG

```json
"judge_bars_enabled": false
```
All three config files.

---

## BUILD CHECKLIST

- [ ] Six `_bar_*` methods added; six cascade methods **kept** as revert path
- [ ] Flag `judge_bars_enabled` default OFF in all three configs
- [ ] Flag OFF → scores byte-identical to today
- [ ] Flag ON → every explanation shows named parts, e.g. `[BOS+defended(0.62)+ladder(0.31ATR,t3)]`
- [ ] STRUCTURE visibly higher when several things line up than on a bare break
- [ ] TREND cannot exceed ~0.45 of its bar without proof
- [ ] `REV` shows `proof(mr-silent)` only when `mr_signal == 0`
- [ ] Report exact line numbers back to Desire

## WHAT TO WATCH ON THE SOAK

The explanations are the point. `[BOS+defended+ladder+spring]` tells you *why* a judge scored what it did — the cascade never could. If a judge shows a single part on most cycles, that segment is doing all the work and the others are dark. **That is the finding to report.**
