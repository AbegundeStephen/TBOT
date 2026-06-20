# TBOT Remediation Plan — Phased Fix of All Audit Findings
**Companion to:** `AUDIT_2026-06-19.md`
**Date:** 2026-06-20
**Goal:** make the bot profitable by fixing it in the correct order — *stop the bleeding → restore risk integrity → make the system observable → unmute the alpha → tune for profit → validate & clean up.*

## Implementation status (2026-06-20)
- **Phase 0 — DONE & compiled.** 0.1 risk-cap guard + orphaned key fixed (active cap kept at 0.25 per owner); 0.2 Binance entry price = fill; 0.3 MT5 broker-stop guaranteed (retry → emergency-close).
- **Phase 1 — CODE COMPLETE, needs validation.** 1.1 sizing uses VTM effective ATR multiplier (shared `VeteranTradeManager.compute_effective_atr_multiplier`); 1.2 periodic atomic portfolio-state save in the VTM loop (default 30s, `trading.state_save_interval_seconds`); 1.3 idempotency guards on Binance-futures + MT5 retry paths (detect/adopt a landed position instead of double-sending).
  - **Must validate before live:** run `python -m py_compile` on changed files; **testnet/paper-test 1.3 specifically** (it touches the live order-send path and could not be integration-tested here).
  - **Known residuals (tracked):** REVERSION sizing↔stop alignment deferred to Phase 4.2 (currently errs toward under-risk — safe); spot (non-futures) double-fill not guarded (spot is unleveraged, lower risk); 1.3 relies on `get_position_info`/`positions_get` which can't always distinguish "flat" from "query error" — guard adopts on positive detection only.
  - **Restart required** for all changes to take effect.
- **Phase 2 — DONE (observational, zero behaviour change).** New `src/analytics/funnel_logger.py` (compiled + unit-smoke-tested). 2.1 funnel: every aggregator eval recorded by stage (generated→raw→veto family→passed→executed) to `logs/funnel/funnel_<date>.jsonl` + periodic INFO summary; hooked at the single return of `get_aggregated_signal_hybrid_dynamic` and `mark_executed` at the `if success:` execution block. 2.2 AI A/B: rejected signals + would-be entry written to `logs/funnel/ai_rejects_<date>.jsonl`. 2.3 model-age: startup warns if any `models/*.pkl` older than `ml.max_model_age_days` (default 10).
  - **Next:** let it collect ~1 week of live/paper data, then read the funnel summary to see which vetoes kill the most opportunities and whether AI rejects would have been winners — this DATA gates Phase 3.
- **Phase 2.4 — DONE (durable shadow engine).** The existing `ShadowTradingEngine` already tracked blocked signals' forward P&L by gate (`get_gate_scorecard`), but was wiped on every restart (no load-on-startup; snapshot-only dump). Now: `_archive` appends every closed shadow trade to an append-only `logs/shadow/closed_<date>.jsonl`, and `load_state()` (called at startup, `trading.shadow_archive_lookback_days` default 30) restores the history so the scorecard accumulates across restarts and can reach the 200-trade calibration threshold. Load+scorecard logic unit-tested.
  - **Scope correction:** the funnel logger (2.1/2.2) and the shadow engine are COMPLEMENTARY, not redundant. The shadow engine's `_shadow_open_blocked` no-ops on `signal == 0`, and AI/aggregator vetoes zero the signal *inside* the aggregator — so the shadow engine only sees signals killed by *downstream execution gates* (health, circuit breaker, quality, cooldown). The funnel is the only record of aggregator-internal vetoes (hard-veto, silent-zone, AI rejection). Both kept.
  - **Future enhancement:** to get forward-P&L on AI/aggregator-internal rejects (the true AI A/B test), the aggregator would need to surface the pre-veto signal to the shadow engine. Noted for later.
- **Phase 2.5 — DONE (analysis tooling).** `scripts/analyze_observability.py` — read-only, safe to run on the live box. Turns funnel + shadow archives into one Phase-3 decision report: signal funnel per asset (raw→passed→executed + veto families ranked), shadow gate scorecard ranked by blocked-signal forward P&L (flags gates that vetoed winners as ">> RELAX?"), and AI-rejection summary. Usage: `python scripts/analyze_observability.py --days 7 [--out report.md]`.
  - **Early read (n=14, NOT actionable):** first funnel sample showed 14/14 BTC raw signals killed by `blocked_ai_validation` — consistent with the audit's "AI filter is the dominant veto." The week-long soak will quantify this with a real sample.

## Soak protocol (deployed to Contabo 2026-06-20)
Phases 0–2 are live. Let it run ~1 week WITHOUT further alpha changes (so the data reflects one stable version), then:
1. `python scripts/analyze_observability.py --days 7 --out logs/obs_report.md`
2. Read which veto family blocks most + which gates the shadow scorecard flags ">> RELAX?".
3. Drive Phase 3 from that report (relax the gates that demonstrably block winners; confirm the MRS-silent thesis), backtesting each change.

## Out-of-order work done during the soak (2026-06-20)
While Phase 3 waits on soak data, the soak-independent items were tackled:
- **Phase 6.1 — DONE (retrainer).** Root cause: the autotrainer never completed a cycle — `training_metadata.json` has only `training_date` (manual May-14 run), no `last_trained`. Fixes in `autotrainer.py`: readiness gate now waits only for the exchanges ENABLED assets need (and proceeds with whatever connected after timeout) instead of requiring BOTH binance+mt5; drift calc isolated so its failure can't block a scheduled retrain; per-cycle INFO diagnostics (last_trained/overdue/weekday/drift) so the cause is visible; `_get_last_training_time` falls back to `training_date`. Scheduled retrain still weekend-gated (Sunday) by design. **Deploy after soak** (a retrain swaps models = behaviour change). To confirm cause on Contabo: `grep "AUTO-TRAIN" logs/*.log`.
- **Phase 6.3 — DONE (docs).** `CLAUDE.md` corrected: AI/sniper reality (ML sniper disconnected; Council "sniper" is a separate live displacement gate; `_check_sniper_filter_LEGACY` is dead), risk-cap correction (active `risk_management.max_total_open_risk=0.25` + load guard), per-asset partials/runner reality, and pointers to the audit/plan/observability.
- **Phase 6.4 — DONE (data hygiene).** `clean_data` now WARNS when it ffills gap rows (phantom-candle visibility on the live regime path); startup banner surfaces `trading.mode` vs `binance.testnet` and warns on the live+testnet mismatch.
- **Phase 6.2 — DOCUMENTED, deletion deferred.** Confirmed `_check_sniper_filter_LEGACY` (signal_aggregator.py, ~227 lines) is unreferenced dead code. Physical deletion deferred: low value, and the 4,200-line file can't be compile-verified in the current sandbox — remove it in an environment where `py_compile` can confirm.
- **Phase 5.2 — DEFERRED (not started).** Extracting/reconciling the drifted shared gates (stale 65 vs 30, NY-open lists, calendar fallback) is a behaviour-affecting refactor across two 3–4k-line live-signal files. Too risky to do blind without compile + integration tests; do it with a proper test harness. Spec retained in Phase 5 above.

## Guiding principles
1. **Correctness before cleverness.** Fix the bugs that mis-price risk and leave positions unprotected *before* touching any strategy logic. There is no point tuning signals that get sized and stopped wrong.
2. **Make it observable before you tune it.** You cannot improve a funnel you can't see or an ML filter you can't measure. Observability (Phase 2) precedes alpha changes (Phase 3).
3. **Every behavioural change is flag-gated and backtested.** Default OFF, enable per-asset, A/B against the current behaviour, paper/shadow before live.
4. **One change family per phase.** Each phase is independently shippable and independently revertible.
5. **No strategy change ships without a before/after backtest + a paper-trade soak.**

## Severity & ordering rationale
- **Phase 0** = active money-losers / unprotected-capital risks (ship in hours, no strategy change).
- **Phases 1–2** = risk integrity + observability (days).
- **Phases 3–4** = the actual profitability work (weeks, measured).
- **Phases 5–6** = architecture decision, validation, hygiene.

---

## Phase 0 — Emergency safety & correctness (ship first, ~1 day)
**Objective:** stop the three things that can lose real money right now. No alpha/behaviour change.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 0.1 | §8 risk cap disabled (`max_total_open_risk=45`) | Set to `0.45` now; reset to `0.25` Sunday. Add load-time assert `0 < cap <= 1.0` that refuses to start otherwise. | `config/config.json`; `portfolio_manager.py:1103` + config loader |
| 0.2 | §13.1 Binance registers requested price, not fill | Pass `entry_price=executed_price` to `add_position`; only fall back to `current_price` when fill price is genuinely unavailable. | `binance_handler.py:1284` (compare MT5 correct path :861) |
| 0.3 | §13.4 MT5 stop is best-effort / swallowed failure | After entry, verify `_push_sl_to_exchange` retcode; retry N times; if still unplaced, **emergency-close** the position rather than run naked. Log loudly. | `mt5_handler.py:909-931, 1009` |

**Validation:** unit test the cap assert (rejects 45, accepts 0.25); paper-trade one BTC and one MT5 entry, confirm logged entry price == broker fill and a broker SL exists within seconds of fill; force an SL-push failure (bad price) and confirm emergency-close fires.
**Rollback:** all three are localized; revert the commit. No schema/state migration.
**Exit criteria:** every new position provably carries a broker-side stop and a correct entry price; bot refuses to start with an out-of-range risk cap.

---

## Phase 1 — Risk-budget integrity & state durability (~2–3 days)
**Objective:** make realized per-trade risk equal the intended risk, and make trade state survive restarts.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 1.1 | §13.2 sizing stop ≠ executed stop | Reorder: compute the VTM's *actual* initial stop first (regime-adaptive mult + floors), then size against that exact distance. Single source of truth for stop distance. | `binance_handler.py:986-1048`, `mt5_handler.py:520-640`, `veteran_trade_manager.py:342-355,713` |
| 1.2 | §12.1 VTM rebuilt from scratch on restart | Persist VTM state (`to_dict`) on every mutation (stop move, partial, runner) to disk/DB; on reattach use `from_dict` instead of reconstructing. | `main.py:3444`, `veteran_trade_manager.py:2153/2186` |
| 1.3 | §13.3 non-idempotent retries → double fill | Attach deterministic `newClientOrderId` (Binance) / unique `comment`+pre-send position check (MT5); before re-sending, query exchange for that token. | `binance_handler.py:1140-1196`, `mt5_handler.py:1167-1207` |

**Validation:** backtest unaffected (execution-layer only); paper test — confirm position size × stop distance == risk_pct × balance (±1 tick); kill -9 the process mid-trade and restart, confirm partials/breakeven/runner state restored exactly; simulate a response timeout and confirm no double position.
**Rollback:** 1.1/1.3 are localized reverts. 1.2 adds a state file — on rollback, ignore the file (reconstruct path still exists as fallback).
**Exit criteria:** realized risk matches budget; a restart never loses or resets trade-management state; no double fills under induced network failure.

---

## Phase 2 — Observability (do this BEFORE any alpha change, ~2–3 days)
**Objective:** make the signal funnel and the ML filter measurable so Phase 3/4 decisions are data-driven, not vibes.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 2.1 | §5/§10 invisible veto funnel | Per asset/day, count: signals generated → survived each gate (hard-veto, silent-zone, gatekeeper, consensus, AI, filters) → executed. Emit to a structured log + dashboard table. | new `funnel_logger`, hook in both aggregators' `get_aggregated_signal` |
| 2.2 | §5/§12.3 AI filter unvalidated (~80% reject) | Log every AI-rejected signal with the would-have-been entry, then record the forward outcome (e.g. +N bars / to next opposite signal). Build a weekly "AI filter marginal P&L" report. | `hybrid_validator.py`, shadow logger |
| 2.3 | §12.2 models silently stale | Startup + daily check: warn/alert if any model file age > configured max (e.g. 10 days). | `main.py` init, `autotrainer.py` |

**Validation:** run 1 week paper/live; confirm the funnel report shows realistic counts and the AI A/B log accumulates labelled outcomes. No trading-behaviour change in this phase.
**Rollback:** pure addition; disable the loggers.
**Exit criteria:** you can answer "how many real opportunities did each veto kill last week, and did the AI filter add or destroy edge?" with data. This gates Phase 3.

---

## Phase 3 — Unmute the alpha (~1–2 weeks, measured)
**Objective:** stop the architecture from cancelling its own mean-reversion edge; let valid signals express. Driven by Phase 2 data.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 3.1 | §2 trend & reversion averaged/netted | **Core fix.** Route by regime instead of summing: trend sleeve weighted in trending/expansion regimes, reversion sleeve in range/MR regimes. Minimum interim step: remove the MR↔TF opposition penalty when each is in its correct regime. | `signal_aggregator.py:2222 _calculate_score`; regime from existing MTF/Livermore |
| 3.2 | §3 MRS gates too tight | Mode 2: 4/4 → 2/3 confluences, drop weakest (ma_proximity). Mode 1: make BB/KC squeeze a confidence booster, not a hard gate (keep spring + 2/4 mandatory). | `mean_reversion.py:482,502,589` |
| 3.3 | §3 Mode 1 eaten by 4H silent-zone gate | Wire the promised Mode-1 spring bypass so NATURAL_RETRACEMENT pullback entries aren't zeroed by the 4H silent-zone hold. | `signal_aggregator.py:3303`, `mean_reversion.py:998` |
| 3.4 | §2 MR can't fire alone | Lower MR independent-override threshold 0.75 → ~0.62; raise MR confidence ceiling so a clean reversion can win/fire solo. | `signal_aggregator.py:272`, `mean_reversion.py:528,648` |
| 3.5 | §7 EMA is a 2nd trend vote | Fold EMA into the trend sleeve (don't count it as an independent 3rd voter that guarantees trend bias). | `signal_aggregator.py:256-289` |
| 3.6 | §4 TF tuning | Make the squeeze bonus a direction-gated multiplier (not +1.0 to both sides); review the 2.0 4H counter-penalty once reversion sleeve covers the other regime. | `trend_following.py:494-518` |

**Validation:** for each sub-change — backtest before/after on ≥12 months across all assets (win rate, profit factor, expectancy, max DD, trade count); confirm Phase-2 funnel shows MR Mode 1/2 actually firing now; paper-soak 1–2 weeks before live; enable per-asset.
**Rollback:** every item flag-gated, default OFF; revert flags individually.
**Exit criteria:** MR Mode 1 & 2 fire at a meaningful rate; net expectancy in backtest ≥ current; no single change degrades profit factor without a documented reason.

---

## Phase 4 — Exit & profitability tuning (~1–2 weeks, measured)
**Objective:** let winners pay. The system is currently biased to small wins/scratches; this is where profit factor is won.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 4.1 | §6.2/6.3 runner clipped early | Widen runner trail from 1×ATR → 2.5–3.0×ATR (or structure trail); push breakeven later. Backtest profit factor before/after. | `config risk.runner_trail_atr_multiplier`, `veteran_trade_manager.py` trailing |
| 4.2 | §6.1 REVERSION SL/TP path weak | Give reversion entries the same R:R pre-flight as TREND; add a partial ladder; **fix the unused 2.0×ATR reversion floor** (branch hardcodes 0.5×ATR wick stop). | `veteran_trade_manager.py:346,732-752,211` |
| 4.3 | §6.2 TP1 over-aggressive | Re-test TP1 at 45% off vs a smaller first clip, so expectancy isn't 100% runner-dependent. | `config risk.partial_targets/partial_sizes` |

**Validation:** backtest each change in isolation, then combined; compare profit factor, expectancy, avg-R of winners, and equity-curve smoothness; paper-soak; enable per-asset.
**Rollback:** all config/flag-gated.
**Exit criteria:** measurable improvement in profit factor / expectancy without unacceptable DD increase; runner contribution to P&L is healthy, not the only source.

---

## Phase 5 — Architecture decision & validation (~1 week)
**Objective:** stop hiding attribution; keep the filters that earn their keep.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 5.1 | §10.8 hybrid hides which engine drives P&L | Run Council vs Performance head-to-head on identical periods; pick by net expectancy. Document the decision. | `hybrid_aggregator_selector.py`, both aggregators |
| 5.2 | §12.4 duplicated/drifted safety gates | Extract shared pre-trade gates (stale price, NY-open, econ-calendar, flash) into one module both aggregators import. Reconcile the 65 vs 30 min and asset-list divergences. | `signal_aggregator.py:301,2962+`, `council_aggregator.py:148,942+` |
| 5.3 | §5/§12.3 AI filter | Using Phase-2 A/B data: keep, retrain, or demote the AI validator. If it stays an unvalidated ~80% veto with no positive marginal P&L, demote it to advisory. Re-wire a real model or stop calling it "AI". | `hybrid_validator.py`, `signal_aggregator.py:2472` |

**Validation:** decision backed by the head-to-head and the AI A/B report; after gate-dedup, diff behaviour on a replay to confirm no unintended change.
**Rollback:** keep both aggregators available behind the selector during transition.
**Exit criteria:** one chosen aggregator (or a justified rule for switching); a single shared gate module; an evidence-based verdict on the AI filter.

---

## Phase 6 — Hygiene & sustainability (~3–5 days)
**Objective:** keep the system honest and maintainable.

| ID | Finding | Change | File / ref |
|---|---|---|---|
| 6.1 | §12.2 retrainer enabled but not producing | Diagnose why no fresh models since May; fix the train→promote pipeline; verify the Phase-2 model-age alert clears after a successful run. | `autotrainer.py`, `scripts/training/` |
| 6.2 | §12.3 sniper dead code | Remove or clearly quarantine the disconnected CNN-LSTM sniper paths. | `signal_aggregator.py:2470-2488`, `src/ai/sniper.py` |
| 6.3 | §12.5 doc drift | Update `CLAUDE.md`/README: remove sniper from architecture, correct partials/runner/circuit-breaker specs to match config, note regime-routed sleeves. | `CLAUDE.md`, `README.md` |
| 6.4 | §13.5 data hygiene | Add startup banner/assert for testnet-execution-vs-live-data mismatch; ensure `ffill`'d data never reaches live signal generation. | `data_manager.py:121,632-644` |

**Validation:** a fresh model set is produced and promoted; docs match code; startup banner correctly reports execution vs data source.
**Exit criteria:** no dead code in the live path, docs trustworthy, retrainer demonstrably running.

---

## Findings traceability matrix (every audit finding → phase)
| Audit finding | Phase |
|---|---|
| §2 averaging trend vs reversion | 3.1 |
| §3 MRS silent (gates) | 3.2, 3.3, 3.4 |
| §3 Livermore veto stack / silent-zone | 3.1, 3.3 |
| §4 TF squeeze both-sides / 4H penalty / conf rounding | 3.6 |
| §5 over-vetoing funnel | 2.1, 3.1 |
| §5 AI validator unvalidated | 2.2, 5.3 |
| §6.1 REVERSION SL/TP weak + unused 2.0 floor | 4.2 |
| §6.2 TP1 over-aggressive / runner-dependent | 4.3, 4.1 |
| §6.3 fast BE + 1×ATR runner trail | 4.1 |
| §7 EMA duplicate trend vote | 3.5 |
| §8 max_total_open_risk=45 | 0.1 |
| §10.8 hybrid hides attribution | 5.1 |
| §12.1 VTM rebuilt on restart | 1.2 |
| §12.2 stale models / retrainer | 2.3, 6.1 |
| §12.3 sniper dead code / "AI" naming | 5.3, 6.2 |
| §12.4 duplicated aggregator gates | 5.2 |
| §12.5 doc drift | 6.3 |
| §13.1 Binance entry price bug | 0.2 |
| §13.2 sizing stop ≠ executed stop | 1.1 |
| §13.3 non-idempotent retries | 1.3 |
| §13.4 MT5 best-effort stop | 0.3 |
| §13.5 data testnet/live + ffill | 6.4 |

**Coverage: all 22 finding groups are assigned to a phase.**

---

## Sequencing summary
```
Phase 0  Safety/correctness ............ ~1 day      (no strategy change)
Phase 1  Risk integrity + durability ... ~2–3 days
Phase 2  Observability ................. ~2–3 days   (gates Phase 3)
Phase 3  Unmute alpha ................. ~1–2 weeks   (measured, flag-gated)
Phase 4  Exit/profit tuning ........... ~1–2 weeks   (measured, flag-gated)
Phase 5  Architecture + validation .... ~1 week
Phase 6  Hygiene ...................... ~3–5 days
```

## Cross-cutting guardrails
- Every behavioural flag defaults **OFF**; enable per-asset after backtest + paper soak.
- Keep a one-line changelog entry + the before/after backtest metrics for each shipped item.
- Bot restart required after config/flag changes — confirm the new state loaded (see existing memory notes on restart-required changes).
- Re-run this plan's validation column as the definition of done for each item.
