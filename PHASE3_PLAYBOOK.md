# Phase 3 Execution Playbook — data → action
**Companion to:** `AUDIT_2026-06-19.md`, `REMEDIATION_PLAN.md`
**Purpose:** turn the soak into mechanical decisions. After ~1 week of funnel + shadow data, run the analyzer and follow the decision rules below. Every change is flag-gated, default-OFF, backtested before live.

## Step 0 — generate the evidence
```bash
python scripts/analyze_observability.py --days 7 --out logs/obs_report.md
```
Read three blocks: (1) FUNNEL per asset, (2) SHADOW gate scorecard, (3) AI rejections.

Sanity gates before acting:
- Need a meaningful sample. Treat any gate/bucket with **n < 30 closed shadow trades** or **< ~200 evaluations/asset** as inconclusive — let the soak run longer rather than tune on noise.
- Confirm the funnel logger and shadow archive actually populated (`logs/funnel/`, `logs/shadow/closed_*.jsonl`). If empty, fix logging first.

---

## Decision tree

### A. Where is the bottleneck — silent strategies or vetoes?
Look at FUNNEL `raw_signal` rate and the top `blocked_*` / `no_raw_signal` buckets.

- **A1. `no_raw_signal` dominates** (strategies rarely produce a signal) → the bottleneck is the engines, not the vetoes. Prioritise **Action C (unmute MRS)** and verify TF/EMA aren't over-gated. This confirms the audit's "MRS silent" thesis.
- **A2. Raw signals are plentiful but `blocked_*` dominates** → the bottleneck is the veto stack. Prioritise **Action B** on whichever family is largest.
- **A3. `passed_to_execution` is healthy but `executed` is much lower** → downstream execution gates (health/circuit-breaker/quality/cooldown) are the leak; cross-check with the SHADOW scorecard (those gates are exactly what shadow measures).

### B. Which veto to relax — driven by the SHADOW scorecard
For each gate the analyzer flags `>> RELAX?` (n ≥ 10, win_rate ≥ 50%, total_pnl > 0 → it blocked net-winning trades):

| Gate flagged by shadow | Action | File / knob |
|---|---|---|
| `quality_gate` | raise nothing; instead lower `min_signal_quality` one notch and re-test | `config trading.min_signal_quality` (main.py quality check ~5173) |
| `circuit_breaker` | do NOT relax — safety gate; investigate why signals cluster post-loss | — (leave) |
| `natural_cycle_gate` | shorten the cycle-age hold | `signal_aggregator` natural-cycle gate |
| `system_health_veto` | fix the health flapping, don't relax the gate | `health_monitor` |
| AI rejection (from FUNNEL `blocked_ai_validation`, not shadow) | see **Action D** | `hybrid_validator` |

Rule: **only relax gates the data shows are blocking *winners*.** A gate with negative `total_pnl` on blocked trades is doing its job — leave it. Never relax `circuit_breaker` or health.

### C. Unmute the mean-reversion engine (the core alpha fix, Audit §2/§3)
Do these in order, **each flag-gated default-OFF, each backtested separately** (≥12 months, all assets):

1. **Stop netting MR against TF on the same bar** (`signal_aggregator._calculate_score`, ~2222). Interim: drop the MR↔TF opposition penalty when each is in its correct regime. Full: route by regime (trend sleeve in trending/expansion, reversion sleeve in range/MR).
   - Accept if: MR Mode-1/Mode-2 fire rate rises in the funnel AND net expectancy ≥ current AND profit factor not worse.
2. **Relax MRS gates** (`mean_reversion.py`): Mode 2 `optional_min_count` 4→2/3 (drop `ma_proximity`); Mode 1 make BB/KC squeeze a confidence booster, not a hard gate (keep spring + 2/4).
3. **Wire the Mode-1 silent-zone bypass** so NATURAL_RETRACEMENT springs aren't zeroed by the 4H hold (`signal_aggregator` ~3303, `mean_reversion` ~998).
4. **Lower MR independent-override** 0.75→~0.62 (`signal_aggregator` ~272) so a clean reversion can fire alone.
5. **Fold EMA into the trend sleeve** (`signal_aggregator` weights ~256) so the vote isn't 2-trend-vs-1-reversion by construction.

Backtest gate for the whole of C: equity-curve, profit factor, expectancy, max-DD, trade count before/after. Ship per-asset only where it improves.

### D. The AI filter verdict (Audit §5/§12.3) — needs the A/B
From FUNNEL: how many signals did `blocked_ai_validation` kill? The early n=14 sample showed **100% of BTC raw signals AI-rejected** — if that holds at scale, the filter is the dominant veto and is unvalidated.
- The funnel only counts AI rejects (it can't see their forward P&L — they're zeroed inside the aggregator before the shadow engine).
- To get the true A/B you must first **surface the pre-veto signal to the shadow engine** (the Phase-2.4 "future enhancement"): when AI rejects, open a shadow position so its forward P&L is tracked. Then the shadow scorecard gets an `ai_validation` row and you can judge it like any other gate.
- Decision once data exists: if AI rejects are net-losers → keep/strengthen. If net-winners → demote AI to advisory or retune its thresholds. Do NOT keep an unvalidated 80%+ veto.

### E. Then (and only then) exit tuning — Phase 4
After the entry/alpha side is producing more, healthier signals:
- **4.1 runner trail**: `config risk.runner_trail_atr_multiplier` 1.0 → test 2.5–3.0 (or structure trail). Backtest profit factor — the system is runner-dependent, this is where right-tail lives.
- **4.3 TP1**: re-test `risk.partial_targets`/`partial_sizes` first clip (45% at ~1–1.5R is aggressive de-risk).
- **4.2 reversion SL/TP path** (`veteran_trade_manager._calculate_initial_levels` REVERSION branch): now worth doing because reversion trades actually fire post-C. Give it the TREND R:R pre-flight + partial ladder; use the computed 2.0×ATR reversion floor instead of the hardcoded 0.5×ATR wick stop.

---

## Guardrails (unchanged from the plan)
- One change family at a time; flag default-OFF; backtest before/after; paper/shadow soak; enable per-asset.
- Keep the soak clean: don't stack Phase-3 changes while still collecting the baseline you're comparing against.
- Re-run `scripts/analyze_observability.py` after each shipped change to confirm the funnel moved the way the backtest predicted.

## What NOT to do
- Don't wire the RandomForest models into live signals (Audit §12B): they're unused, likely leaky (MR CV ≈ 0.988). If ML is added later it should be a **meta-labeler on shadow outcomes** (take/skip + size on top of rules), not a direction oracle — and only after C proves a rule edge worth filtering.
- Don't relax safety gates (circuit breaker, health, stale/flash) to chase trade count.
