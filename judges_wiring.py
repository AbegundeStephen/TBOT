HOW TO USE THIS DOCUMENT (read this first)
Each numbered item has: a plain-English box explaining what's wrong and why it matters, the exact file and location, then the code change.
Deploy Batch 1 completely before starting Batch 2. Batch 1 is bug fixes only — nothing about how the bot decides to trade changes. Batch 2 changes actual trading behavior.
Inside Batch 2, follow the numbered order — items 11-16 (retest engine + scoring math) must go in before 17-19 (the judges), because the judges lean on them.
After each item, re-check the file compiles before moving to the next — don't batch-paste everything blind.
BATCH 1 — Fix what's broken. No trading-behavior change.
1.1 — Quarantine deadlock
Plain English for Stephen: after a restart, the bot has a rule that says "don't trade until prices look normal." The problem: checking "do prices look normal" was written so it can only happen after the bot's already allowed to trade — which never happens, because it's stuck waiting on that exact check. It's been permanently frozen since this bug was introduced. This fix makes the check ask directly instead of waiting on itself.
Where: main.py, inside _startup_quarantine_active (line 4340).
Two passes: Forward — read the function, confirmed the circular wait. Backward — confirmed via a real 36-hour production log: the line that should print when a trade is allowed never printed once, across two restarts.
# FIND THIS (inside _startup_quarantine_active, near the end):
if self.mt5_handler is not None:
    sanity = getattr(self.mt5_handler, "_price_sanity_status", {})
    asset_status = sanity.get(asset_name, {})
    if not asset_status.get("passed", False):
        logger.debug(f"[QUARANTINE] {asset_name}: time floor met but price sanity not yet confirmed — holding.")
        return True

# REPLACE WITH:
asset_cfg = self.config.get("assets", {}).get(asset_name, {})
if self.mt5_handler is not None and asset_cfg.get("exchange") == "mt5":
    symbol = asset_cfg.get("symbol", asset_name)
    sanity = getattr(self.mt5_handler, "_price_sanity_status", {})
    asset_status = sanity.get(asset_name, {})
    if not asset_status.get("passed", False):
        try:
            _, is_sane, _ = self.mt5_handler._get_verified_current_price(symbol, asset_name)
        except Exception as _qc_err:
            logger.debug(f"[QUARANTINE] {asset_name}: sanity check errored ({_qc_err}) — holding.")
            return True
        if not is_sane:
            logger.debug(f"[QUARANTINE] {asset_name}: time floor met but price sanity not yet confirmed — holding.")
            return True
Also add — a startup check that runs once before the main loop starts, so the very first cycle already has real data instead of an empty dict. Add this new method anywhere in the same class as _startup_quarantine_active:
def _warm_start_price_sanity_all_assets(self) -> None:
    """Populate price-sanity status for every MT5-routed asset before the
    trading loop starts."""
    if self.mt5_handler is None:
        return
    for asset_name, asset_cfg in self.config.get("assets", {}).items():
        if not asset_cfg.get("enabled", False) or asset_cfg.get("exchange") != "mt5":
            continue
        symbol = asset_cfg.get("symbol", asset_name)
        try:
            self.mt5_handler._get_verified_current_price(symbol, asset_name)
        except Exception as e:
            logger.warning(f"[STARTUP] Price-sanity warm-start failed for {asset_name}: {e}")
Find where self._warm_start_livermore_all_assets() gets called near startup, and add self._warm_start_price_sanity_all_assets() on the line right before it.
1.2 — Stale setting read (lsm_regime_disagreement_gate_enabled)
Plain English: there are two copies of the same on/off switch check in the code. One reads today's setting correctly. This one reads a frozen, outdated copy that never updates. Make it match the correct one.
Where: main.py, line 5118, the "Filter 1" counter-trend block.
Two passes: Forward — confirmed this exact line still reads the stale source. Backward — confirmed the sibling copy elsewhere already reads the correct source and works.
# FIND THIS:
_lsm_gate_enabled_m = bool(self.config.get("phase_config", {}).get(
    "lsm_regime_disagreement_gate_enabled", False
))
if _lsm_gate_enabled_m:
    _agg_m = self.aggregators.get(asset_name) if hasattr(self, "aggregators") else None
    _cs_m = {}
    if _agg_m and getattr(_agg_m, "_cached_composite", None) is not None:
        try:
            _cs_m = _agg_m._cached_composite.to_dict()
        except Exception:
            _cs_m = {}
    _lsm_4h_m = _cs_m.get("livermore_state_4h")

# REPLACE WITH:
_agg_m = self.aggregators.get(asset_name) if hasattr(self, "aggregators") else None
_cs_m = {}
if _agg_m and getattr(_agg_m, "_cached_composite", None) is not None:
    try:
        _cs_m = _agg_m._cached_composite.to_dict()
    except Exception:
        _cs_m = {}
_lsm_gate_enabled_m = bool(
    (_cs_m.get("phase_config", {}) or {}).get("lsm_regime_disagreement_gate_enabled", False)
)
if _lsm_gate_enabled_m:
    _lsm_4h_m = _cs_m.get("livermore_state_4h")
1.3 — Swing-low memory gap
Plain English: the bot's memory of "important price levels" only ever remembers ceilings (resistance). It never remembers floors (support) — the code to save a floor simply doesn't exist. This quietly breaks every long-side trade that needs to check "is there support nearby."
Where: signal_aggregator.py, inside _update_structure_memory (line 1510) — add right after the existing swing-high saving block.
Two passes: Forward — read the function fully, confirmed only a "swing_high" save block exists, no "swing_low" equivalent anywhere in the file. Backward — confirmed via real log: on a genuine BTC long setup, the buy side showed "no level nearby" while the sell side found a real level, in the same cycle.
# ADD this new block immediately after the existing swing-high loop
# (the one that appends {"type": "swing_high", ...}):
for i in range(len(_4h_lows) - 3, 4, -1):
    if _4h_lows[i] < _4h_lows[i - 1] and _4h_lows[i] < _4h_lows[i + 1]:
        _exists = any(
            abs(lvl["price"] - _4h_lows[i]) / _atr < 0.3
            for lvl in self._structure_levels[asset]
        )
        if not _exists:
            self._structure_levels[asset].append({
                "price": _4h_lows[i],
                "tf": "4H",
                "type": "swing_low",
                "tests": 0,
                "age_hours": 0,
            })
        break
1.4 — Connection alarm gated by open positions
Plain English: the "MT5 has disconnected" alarm currently only fires if you happen to have a trade open at that exact moment. No trade open = no alarm, even during a genuine outage. Remove that condition.
Where: main.py, line 6645.
# FIND (paraphrased — check the exact surrounding condition):
_should_alert = (ok is False) and self._has_mt5_positions()

# REPLACE WITH:
_should_alert = (ok is False)
1.5 — MR Mode-3 lean-conflict gap
Plain English: there's a safety check meant to catch when the mean-reversion strategy disagrees with a trade. It's built so it only runs when mean-reversion has no opinion at all. The moment it actually has a real opinion, this exact check turns itself off — backwards from what it should do.
Where: council_aggregator.py, line 3075.
Two passes: Forward — confirmed the condition literally requires mr_signal == 0. Backward — confirmed via real log that a moment where mean-reversion clearly had an active opinion (logged as "Mode3 Fade/SHORT") produced zero output from this conflict check.
# FIND (the outer condition guarding this whole block):
_mr_lean_mode != "off"
and signal != 0
and mr_signal == 0
and mr_conf == 0.0

# REPLACE WITH:
_mr_lean_mode != "off"
and signal != 0
and not (
    (mr_signal == 1 and signal == 1) or (mr_signal == -1 and signal == -1)
)
1.6 — GOLD/USOIL stale-price diagnostic
Plain English: when the bot can't get a fresh price and falls back to "last known price," it never actually asks MT5 why. There's a built-in "tell me why" function already used elsewhere in this exact file for order errors — just never wired in here. This makes future freezes leave a real reason instead of a shrug.
Where: mt5_handler.py, inside get_current_price (line 268), right before the final fallback branch.
# ADD, right before the final "no fresh tick data" log line:
if mt5.last_error()[0] != 0:
    logger.warning(f"[MT5] {symbol}: last_error={mt5.last_error()}")
1.7 — Funnel logger: which gate actually blocks a trade
Plain English: right now the bot logs "772 signals looked good" and "7 actually became trades" — with nothing explaining what happened to the other 765. There are 4-5 separate checkpoints in between that can each silently kill a trade, and none of them currently say so.
Where: main.py — at each of the following gate return-points: the trading-limits check, the minimum-time-between-trades check, the natural-cycle check, and the counter-direction suppression check.
# At EACH of those gates, right before its "return" line, add:
if getattr(self, "funnel_logger", None) is not None:
    try:
        self.funnel_logger.record(asset_name, 0, {"reasoning": "blocked_trading_limits"})
        # swap the reason string per-gate: "blocked_cooldown", "blocked_natural_cycle",
        # "blocked_same_direction" — one distinct label per checkpoint
    except Exception:
        pass
Then in funnel_logger.py, _classify_stage (line 58) needs a matching branch for each new reasoning string so analyze_observability.py can count them separately.
1.8 — Funnel logger: record each judge's own score
Plain English: right now there's no record anywhere of "Trend scored X, Structure scored Y" for real trades — only the final total. Without this, nobody can ever prove which judges are actually earning their weight with real data instead of guesswork.
Where: council_aggregator.py — wherever the final details dict gets built for a cycle (the same dict already passed to funnel_logger.record in main.py), add the per-judge scores into it:
details["judge_scores"] = {
    "trend": trend_score, "structure": structure_score,
    "momentum": momentum_score, "pattern": pattern_score, "volume": volume_score,
}
1.9 — Single-instance lock
Plain English: right now nothing stops two copies of the bot from running at the same time — which is the exact condition believed to have caused the original phantom-position incident. This adds a simple lock file so a second copy refuses to start while one's already running.
Where: main.py, very start of the startup sequence, before anything else runs.
import os, sys

_LOCK_PATH = "logs/bot.lock"
if os.path.exists(_LOCK_PATH):
    with open(_LOCK_PATH) as f:
        _old_pid = f.read().strip()
    if _old_pid.isdigit() and os.path.exists(f"/proc/{_old_pid}"):
        print(f"[STARTUP] Another instance (PID {_old_pid}) is already running. Exiting.")
        sys.exit(1)
with open(_LOCK_PATH, "w") as f:
    f.write(str(os.getpid()))
(Note for Stephen: /proc/{pid} is Linux-style — if this runs on Windows, swap that check for psutil.pid_exists(int(_old_pid)) instead.)
Batch 1, item 10 — disabled-asset enforcement: checked all known candidate sites twice this conversation, all currently clean. No code needed. Closing this one out.
BATCH 2 — Change how the brain decides. Real behavior changes. Follow this order.
2.1 — Retest engine: count real visits, not time spent nearby
Plain English: right now, sitting near a level for 2 hours straight counts as "tested many times," when a trader would call that "tested once, for a while." This makes it require price to actually leave and come back before counting a new test.
Where: signal_aggregator.py, the level-selection block (the same area handling nearby_4h_level selection).
# BEFORE incrementing "tests":
_was_away = best.get("_last_dist_atr", 99) >= 0.5
if best_dist < 0.3 and _was_away:
    best["tests"] = best.get("tests", 0) + 1
best["_last_dist_atr"] = best_dist
2.2 — Retest engine: support/resistance role reversal
Plain English: once a ceiling gets broken and price holds above it, that old ceiling often becomes a new floor (and vice versa). Right now the bot never updates a level's label once it's set.
Where: signal_aggregator.py, same level-tracking area, run every cycle.
for lvl in self._structure_levels[asset]:
    if lvl["type"] == "swing_high" and current_price > lvl["price"] * 1.003:
        lvl["type"] = "swing_low"
    elif lvl["type"] == "swing_low" and current_price < lvl["price"] * 0.997:
        lvl["type"] = "swing_high"
2.3 — Retest engine: proven levels score stronger
Plain English: a level tested 4 times and a level tested for the first time currently score identically. A well-proven level should count for more.
Where: retest_engine.py, inside classify (line 145), the CLEAN tier branch.
_tests = getattr(state, "level_test_count", 0)
_proven_bonus = min(_tests * 0.05, 0.20)
# add _proven_bonus to whatever modifier value CLEAN currently returns
2.4 — Retest engine: WICK tier respects the current role label
Plain English: once 2.2 relabels a level, the sweep-detection logic (WICK tier) needs to actually read that current label, not an old, stale one.
Where: retest_engine.py, WICK tier branch — read lvl["type"] fresh at classification time rather than any cached copy.
2.5 — Achievable-max normalization
Plain English: the current judge weights (Trend 1.5, Structure 1.0, Momentum 1.5, Pattern 0.5, Volume 0.5) already add up cleanly to 5.0, even through the "slightly regime" reweighting — checked this by hand, it holds. The real risk is the next time anyone touches a weight, nobody will notice if it stops adding up. This makes the system calculate its own real ceiling every cycle instead of assuming one.
Where: council_aggregator.py, wherever required_score gets compared against total_score.
_achievable_max = w_trend + w_structure + (w_momentum if is_trending_regime else w_reversion) + w_pattern + w_volume
_score_pct = total_score / _achievable_max if _achievable_max > 0 else 0
_required_pct = required_score / 5.0  # keep existing required_score logic, just express both sides as %
_clears = _score_pct >= _required_pct
2.6 — Continuous scoring with a plain-English tag
Plain English: scores currently jump in steps — barely miss a line, get zero; barely clear it, get full marks. This makes strength scale smoothly, and adds a plain label next to the number so it's readable at a glance.
Where: each judge function, wherever it currently does a hard yes/no cutoff. Example pattern to follow (Structure judge, council_aggregator.py:3566):
_dist_atr = abs(current_price - nearby_level) / atr if nearby_level else 99
if _dist_atr < 0.3:
    tag, structure_score = "CLEAN_BREAK", weight
elif _dist_atr < 1.0:
    tag, structure_score = "MARGINAL_BREAK", weight * (1.0 - _dist_atr)
else:
    tag, structure_score = "NO_SIGNAL", 0.0
2.7 — Pattern judge rebuilt (corrected version — SECONDARY states now included)
Plain English: the Pattern judge's real data source has been broken for a long time, silently falling back to old, weak candlestick math instead. This gives it a real, live data source instead — and it now covers both shallow and deep pullback states, not just the shallow ones (caught this gap by actually simulating it).
Where: new function in signal_aggregator.py, called from _build_composite_state (line 695), right after the Livermore update calls.
def _compute_institutional_pattern(self, df, state) -> None:
    try:
        lsm = getattr(state, "livermore_state_1h", None)
        vol_ratio = df["volume"].iloc[-10:].mean() / max(df["volume"].iloc[-30:-10].mean(), 1e-9)
        range_pct = (df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / df["close"].iloc[-1]
        atr = getattr(state, "atr_fast", None) or df["close"].diff().abs().rolling(14).mean().iloc[-1]
        is_tight_range = range_pct < (2.5 * atr / df["close"].iloc[-1])

        _basing_bull_states = ("NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
        _basing_bear_states = ("NATURAL_REBOUND", "SECONDARY_REBOUND")

        if lsm in _basing_bull_states and is_tight_range and vol_ratio < 0.8:
            state.institutional_pattern = "ACCUMULATION"
        elif lsm in _basing_bear_states and is_tight_range and vol_ratio < 0.8:
            state.institutional_pattern = "DISTRIBUTION"
        elif is_tight_range:
            state.institutional_pattern = "COMPRESSION"
        else:
            state.institutional_pattern = None
    except Exception as e:
        logger.debug(f"[PATTERN] compute error: {e}")
        state.institutional_pattern = None
(Note: the specific 0.8 volume ratio and 2.5×ATR numbers are reasonable starting shapes, not yet validated against real outcomes — flag for review once 1.8's judge-level logging has real data.)
2.8 — Trend judge redefined (corrected — SECONDARY states fixed)
Plain English: stop asking "is price above the average" — Livermore's own state already answers that. Ask "do the 1-hour and 4-hour pictures actually agree" instead. First version of this had the deep-pullback states swapped backwards — caught and corrected below.
Where: council_aggregator.py, _judge_trend_bidirectional (line 3385).
lsm_1h = getattr(state, "livermore_state_1h", None)
lsm_4h = getattr(state, "livermore_state_4h", None)
_bull_states = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
_bear_states = ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
_agree_bull = lsm_1h in _bull_states and lsm_4h in _bull_states
_agree_bear = lsm_1h in _bear_states and lsm_4h in _bear_states
if _agree_bull:
    buy_score, sell_score = weight, 0.0
elif _agree_bear:
    buy_score, sell_score = 0.0, weight
else:
    buy_score = sell_score = weight * 0.3
2.9 — Momentum judge redefined (corrected — capped at its own weight)
Plain English: speed of the current market state matters more than a raw RSI reading. First version of this could accidentally score higher than the judge's own maximum — caught and capped below.
Where: council_aggregator.py, _judge_momentum_bidirectional (line 3829).
lsm_age = getattr(state, "livermore_state_1h_age_bars", 0)
_fresh_transition = lsm_age is not None and lsm_age <= 3
base = weight if _fresh_transition else weight * 0.5
rsi_confirms = (signal > 0 and rsi > 50) or (signal < 0 and rsi < 50)
momentum_score = min(base * (1.15 if rsi_confirms else 0.85), weight)
2.10 — AI validator's structural-anchor check, fixed for real independence
Plain English: this check currently "confirms" a signal by looking for a Livermore anchor — but Structure judge may have already used that exact anchor to create the signal. That's not a second opinion, it's the same one asked twice. This makes it require genuinely separate evidence.
Where: hybrid_validator.py, line 341.
_independent_confirm = False
if composite_state is not None:
    if getattr(composite_state, "order_book_imbalance", None) is not None:
        _independent_confirm = (
            (signal > 0 and composite_state.order_book_imbalance > 0.15) or
            (signal < 0 and composite_state.order_book_imbalance < -0.15)
        )
    else:
        _independent_confirm = (
            (signal > 0 and df["close"].iloc[-1] > df["close"].iloc[-3]) or
            (signal < 0 and df["close"].iloc[-1] < df["close"].iloc[-3])
        )

if _anchor is not None and _independent_confirm:
    pass  # existing approve logic continues here
else:
    return self._reject_signal(
        signal_details, sr_result, pattern_result,
        reason="anchor_without_independent_confirmation", strategy=strategy,
    )
2.11 — The trade_type/Reversion fix (the biggest single change in this batch)
Plain English: there's a switch meant to decide "is this a trend trade or a reversal trade" — and it's been permanently stuck on "trend" for every asset, forever, because it checks a setting that's never actually been set to anything else. This means the more forgiving rules built specifically for reversals have likely never once applied. This ties it to what's actually happening live instead.
Where: council_aggregator.py, line 1849.
# FIND:
preset_name = self.config.get("name", "balanced").lower()
trade_type = "REVERSION" if preset_name == "mr" else "TREND"

# REPLACE WITH:
trade_type = "REVERSION" if not is_trending_regime else "TREND"
2.12 — Post-signal block + MR-lean-conflict made Structure-aware
Plain English: two separate safety checks currently treat a signal the same regardless of why it fired. If Structure already drove the signal, these checks are just re-asking the same question — soften them in that case. If Trend/Momentum drove it, keep them at full strength — that's a genuine second opinion.
Where: both the POST-SIGNAL LIVERMORE COUNTER-TREND BLOCK in main.py, and the lean-conflict block in council_aggregator.py (line 3075 area).
_driven_by_structure = details.get("judge_scores", {}).get("structure", 0) > (0.6 * details.get("total_score", 1))
if _driven_by_structure:
    _bump = _bump * 0.3
2.13 — Small addition: NATURAL_REBOUND missing from the bearish-lean set
Plain English: found this while checking the above — the bearish version of the lean-conflict set is missing one state that its bullish mirror already has. Small, real asymmetry.
Where: council_aggregator.py, near line 3090.
# CURRENT:
_MR_LEAN_SHORT = {"SECONDARY_REBOUND", "MAIN_UP"}
# CONSIDER ADDING "NATURAL_REBOUND" for symmetry with the LONG set — confirm with
# Desire before changing, since this changes live behavior, however small.
2.14 — Stop-loss: math only as a last resort, and only for strong signals
Plain English: stops should sit behind a real level whenever possible. When there genuinely isn't one, only fall back to a plain distance-based stop if the signal is strong enough to justify it — otherwise, skip the trade rather than protect it with a weak stop.
Where: veteran_trade_manager.py, _compute_structural_stop (line 2781).
if nearby_level is None:
    if total_score >= (required_score + 1.0):
        return None  # falls through to the existing distance-based stop, on purpose
    else:
        return "SKIP_TRADE"
2.15 — V-Shape override no longer gets silently cancelled
Plain English: there's a rare, deliberate exception that lets an extremely fast, strong move override the "don't fight the trend" rule. The very next check has no idea that exception was granted and cancels it a moment later. This makes it recognize the exception.
Where: main.py, Filter 1 (near line 5118).
if details.get("trade_type") == "V_SHAPE":
    pass  # already earned an explicit exception — don't re-block it
elif is_counter_trend:
    pass  # existing block logic stays here
2.16 — VTM: counter-trend positions get their own transition branch
Plain English: found this by simulating a real reversal trade start to finish. VTM knows how to manage a normal, trend-aligned trade through a transition. It has no logic at all for a counter-trend trade (exactly the kind item 2.11 is about to make possible) when its own thesis starts confirming — it just does nothing.
Where: veteran_trade_manager.py, _apply_livermore_transition (line 1502), inside the MAIN → NATURAL branch.
if prev_state in _MAIN and new_state in _NATURAL:
    _thesis_confirming = (
        (is_long and new_state == "NATURAL_REBOUND") or
        (not is_long and new_state == "NATURAL_RETRACEMENT")
    )
    if _same_dir:
        pass  # existing logic stays
    elif _thesis_confirming:
        self.runner_trail_atr_multiplier = _config_trail * 1.1
        logger.info("[VTM LIVE LSM] %s: counter-trend thesis confirming — trail eased", self.asset)
2.17 — Shadow trader schema overhaul
Plain English: the shadow tracker that's supposed to prove whether these changes actually work doesn't even have "Council" as a valid category to record — only the old system's labels. This gives it real categories matching everything above.
Where: shadow_trader.py, the ShadowPosition class (line 72).
@dataclass
class ShadowPosition:
    asset: str
    side: str
    strategy_source: str  # now includes "COUNCIL" alongside "TF" | "MR" | "EMA" | "consensus"
    judge_driver: str = "unknown"       # NEW — which judge contributed most
    score_pct_of_max: float = 0.0       # NEW — from item 2.5
    qualify_tag: str = ""               # NEW — from item 2.6
    livermore_state_1h: str = ""        # NEW
    gate_blocked_by: str = ""
    # ... rest of existing fields unchanged
Deployment order recap: Batch 1 fully first. Inside Batch 2: 2.1→2.4, then 2.5→2.6, then 2.7→2.9, then 2.10→2.16 (can go in any order relative to each other), then 2.17 last since it needs the final field names from everything above it.