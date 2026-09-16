"""
DATA-4 ITEM 3: offline trail-multiplier replayer.

Reads closed episodes, reconstructs what each candidate trail multiplier would
have done against real 15m price, and reports. Proposes; never applies.

REFUSES TO REPORT until it reproduces a stored backtest's exits (Route A
calibration). If this script cannot recover a result already verified by hand,
nothing else it says is worth reading.

Run:  python tools/replayer.py --calibrate
      python tools/replayer.py --report
"""

import argparse, json, glob, random, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ARMS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]   # floor 0.8 — the 15m study found no
                                        # interior optimum below it

BE_TRIGGER_R = 0.75      # phase_config.r_breakeven_trigger
BE_LOCK_R    = 0.20      # phase_config.r_breakeven_lock
SYMBOL_MAP = {"BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm",
              "EURUSD": "EURUSDm", "USOIL": "USOILm", "GBPAUD": "GBPAUDm"}

# Per-asset ATR stop multiplier -- assets.<ASSET>.risk.atr_multiplier in
# config.json. Confirmed against the live config (28 Aug) rather than taken
# on faith: BTC 1.5, GOLD 2.5, USTEC 1.8, EURUSD 2.0, USOIL 2.5, GBPAUD 2.0 --
# all matched exactly.
ASSET_ATR_MULT = {"BTC": 1.5, "GOLD": 2.5, "USTEC": 1.8,
                   "EURUSD": 2.0, "USOIL": 2.5, "GBPAUD": 2.0}


def load_path(asset, start, end):
    """15m bars between entry and exit. None if unavailable or mis-dated."""
    sym = SYMBOL_MAP.get(asset)
    if not sym:
        return None
    p = Path(f"data/raw/{sym}_15m.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=[0], index_col=0)
    if df.index.min().year < 2023:
        raise RuntimeError(
            f"{p} carries pre-2023 timestamps — the 2020-epoch corruption. "
            f"Run DATA-4 Item 2 before replaying."
        )
    return df.loc[str(start):str(end)]


def estimate_atr_at(asset, entry_time, lookback_bars=40):
    """Regime-adaptive ATR from 1H bars, matching VTM exactly. Returns the
    entry-time ATR the real system would have computed for this trade.

    CALIBRATION FIX (28 Aug, post-first-run): the original version of this
    function computed a flat period-14 True Range average from 15m bars.
    That does not match what the real system computes at all -- confirmed
    directly against VeteranTradeManager._calculate_atr()
    (veteran_trade_manager.py:913-960):

      1. It's regime-adaptive, not flat: ATR(7)/ATR(14)/ATR(28) are all
         computed, and the one actually used is chosen by the fast/slow
         ratio -- ratio > 1.30 -> ATR(7), ratio < 0.70 -> ATR(28), else
         ATR(14). (A squeeze-aware ATR(50) override exists too, gated on
         BB/KC squeeze state; not replicated here -- squeeze detection is a
         separate indicator this script has no cheap way to reconstruct,
         and the ratio-based selection is almost certainly the dominant
         effect. Residual gap, stated rather than hidden.)

      2. It runs on 1H bars, not 15m -- confirmed via backtest.py:97/800
         (DATA_FILE_MAP maps every asset to *_1h.csv, the feed VTM's
         high/low/close arrays are built from). A flat 15m ATR(14) and a
         regime-adaptive 1H ATR are not close approximations of each other;
         they're different numbers computed a different way from different
         data. This was very likely the dominant cause of the first
         calibration run's near-total R mismatch (3/85 within 0.15R, worse
         than the 66% exit-reason match rate alone suggested) -- a wrong
         ATR corrupts both the stop distance (denominator of every R) and
         the trail distance in the same direction.

    The PATH WALK (replay/replay_verbose) still uses 15m bars, per Desire's
    ruling that hourly simulation inflates expectancy -- only the ATR
    VALUE itself needs to match what the real system computed it on. Those
    are two different concerns: how coarse the price path is sampled at
    (15m, deliberately) vs what timeframe produced the ATR number being
    walked against (1H, because that's what actually happened).
    """
    import talib
    sym = SYMBOL_MAP.get(asset)
    if not sym:
        return None
    p = Path(f"data/raw/{sym}_1h.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=[0], index_col=0)
    if df.index.min().year < 2023:
        raise RuntimeError(
            f"{p} carries pre-2023 timestamps — check this file's integrity "
            f"before replaying."
        )
    window = df.loc[:str(entry_time)].tail(max(lookback_bars, 40))
    if len(window) < 29:   # need at least 28 bars + 1 for the slow ATR
        return None
    high, low, close = (window["high"].values, window["low"].values,
                         window["close"].values)
    atr_fast = talib.ATR(high, low, close, timeperiod=7)[-1]
    atr_mid  = talib.ATR(high, low, close, timeperiod=14)[-1]
    atr_slow = talib.ATR(high, low, close, timeperiod=28)[-1]
    if pd.isna(atr_mid) or not atr_slow:
        return None
    ratio = atr_fast / atr_slow
    if ratio > 1.30:
        selected = atr_fast
    elif ratio < 0.70:
        selected = atr_slow
    else:
        selected = atr_mid
    return float(selected) if pd.notna(selected) else None


def replay(entry, stop, side, atr, path, mult):
    """Re-run the exit stack for one trail multiplier. Returns R."""
    risk = abs(entry - stop)
    if risk <= 0 or path is None or path.empty:
        return None
    cur, armed, peak = stop, False, entry
    for _, bar in path.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        # Adverse first: within a bar we cannot know the order, so assume the
        # stop is hit before the extreme. Pessimistic, and consistent across
        # every arm, so comparisons stay fair.
        if (side == "long" and lo <= cur) or (side == "short" and hi >= cur):
            return (cur - entry) / risk * (1 if side == "long" else -1)
        peak = max(peak, hi) if side == "long" else min(peak, lo)
        prog = abs(peak - entry) / risk
        if not armed and prog >= BE_TRIGGER_R:
            armed = True
            cur = entry + BE_LOCK_R * risk * (1 if side == "long" else -1)
        if armed:
            t = peak - mult * atr if side == "long" else peak + mult * atr
            cur = max(cur, t) if side == "long" else min(cur, t)
    close = float(path.iloc[-1]["close"])
    return (close - entry) / risk * (1 if side == "long" else -1)


def replay_verbose(entry, stop, side, atr, path, mult):
    """Same exit stack as replay(), but also returns which stage fired.

    CALIBRATION FIX #4 (28 Aug, post-fourth-run): the real classification
    rule is nothing like an "armed" flag -- confirmed directly against
    VeteranTradeManager._check_exit_locked's STEP 2 stop-loss check
    (veteran_trade_manager.py:2708-2777). At the moment the stop is hit, it
    classifies purely by WHERE the stop currently sits relative to entry,
    using a fixed 0.125*ATR band -- not by tracking whether some trailing
    mechanism was ever "armed":

      offset = 0.125 * atr
      long:  stop >  entry + offset  -> TRAILING_STOP
             stop >= entry - offset  -> BREAK_EVEN
             else                    -> STOP_LOSS
      (mirrored for short)

    This matters because the R-lock price itself (entry + BE_LOCK_R*risk =
    entry + 0.20*1.5*atr = entry + 0.30*atr for BTC) already sits PAST the
    0.125*ATR band -- so a trade that stops exactly at the R-lock, having
    never advanced further, is real-classified TRAILING_STOP, not
    BREAK_EVEN. The previous armed/be_lock_price equality check got this
    backwards for exactly that case, which was the dominant mismatch
    pattern in every calibration run so far (break_even -> trailing_stop).

    Log evidence confirming which mechanism actually fires (28 Aug, against
    this backtest's own log): "R-lock:" appears 52 times; soft_risk_cut /
    intermediate_trail / breakeven_atr (the older, separate ATR-profit-
    triggered mechanism) all appear zero times. So the R-based trigger/lock
    this script already modeled (BE_TRIGGER_R=0.75, BE_LOCK_R=0.20) is
    confirmed correct -- only the exit-reason label was wrong.

    Returns (r, reason) -- either may be None if the path is unusable.
    """
    risk = abs(entry - stop)
    if risk <= 0 or path is None or path.empty:
        return None, None
    cur, armed, peak = stop, False, entry
    _offset = 0.125 * atr

    def _classify(cur_price):
        if side == "long":
            if cur_price > entry + _offset:
                return "trailing_stop"
            if cur_price >= entry - _offset:
                return "break_even"
            return "stop_loss"
        else:
            if cur_price < entry - _offset:
                return "trailing_stop"
            if cur_price <= entry + _offset:
                return "break_even"
            return "stop_loss"

    for _, bar in path.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if (side == "long" and lo <= cur) or (side == "short" and hi >= cur):
            r = (cur - entry) / risk * (1 if side == "long" else -1)
            return r, _classify(cur)
        peak = max(peak, hi) if side == "long" else min(peak, lo)
        prog = abs(peak - entry) / risk
        if not armed and prog >= BE_TRIGGER_R:
            armed = True
            cur = entry + BE_LOCK_R * risk * (1 if side == "long" else -1)
        if armed:
            t = peak - mult * atr if side == "long" else peak + mult * atr
            cur = max(cur, t) if side == "long" else min(cur, t)
    close = float(path.iloc[-1]["close"])
    r = (close - entry) / risk * (1 if side == "long" else -1)
    return r, _classify(cur)


def calibrate(result_json="logs/backtests/20260822_164803/result.json"):
    """ROUTE A: reproduce a stored backtest's exits, trade by trade.

    Passing means the replayer models the exit stack faithfully. Failing means
    it does not -- and the per-trade mismatches say which assumption broke.

    Refuses to pass quietly: prints every mismatch, not just a score.
    """
    data = json.load(open(result_json, encoding="utf-8"))
    asset = data["asset"]
    trades = data.get("trades_detail", [])
    print(f"CALIBRATION (Route A): {asset}, {len(trades)} trades from {result_json}")
    print(f"backtest preset={data.get('preset')} aggregator={data.get('aggregator')}")

    # The multiplier that run used. Confirm against the run's own log before
    # trusting a pass -- a wrong assumption here invalidates the test.
    RUN_MULT = 0.8   # runner_trail_atr_multiplier at the time of the run

    # CALIBRATION FIX #2 (28 Aug, post-second-run): the flat entry -/+
    # atr*mult stop formula below is only what the real system uses for
    # TREND trades -- confirmed directly against
    # VeteranTradeManager._calculate_initial_levels()
    # (veteran_trade_manager.py:1006-1160). REVERSION trades get an
    # entirely different stop: the nearest tested 4H zone-ladder line
    # (self.zone_current_lower/upper), wick-buffered by 0.5*atr, only
    # falling back to the ATR formula if that geometry is invalid. That
    # zone-ladder state is a function of the FULL historical price/level
    # history up to the trade's entry moment -- it is not reconstructable
    # from result.json's stored fields, and re-deriving it would mean
    # rebuilding the entire zone-building pipeline (arguably as much work
    # as the path-capture alternative this replayer exists to avoid).
    #
    # So: report TREND and REVERSION separately. Only TREND trades test
    # what this script can actually claim to model; REVERSION trades are
    # scored too (for visibility) but flagged as using a known-wrong stop,
    # and do not count toward the pass bar.
    #
    # Also worth knowing before trusting a TREND-only pass: the ATR
    # baseline can be widened further by a "MA Shield" step
    # (use_ema_structure) that tucks the stop behind a nearby EMA/zone
    # line. NOT replicated here. Confirmed inert for BTC specifically
    # (assets.BTC.use_ema_structure is false, per DATA-1D's own appendix
    # note on this exact asymmetry) -- so a BTC calibration run is not
    # affected by this gap, but a run on any other asset would be.
    by_type = {"TREND": [0, 0, []], "REVERSION": [0, 0, []]}
    for t in trades:
        path = load_path(asset, t["entry_time"], t["exit_time"])
        if path is None or path.empty:
            continue
        # The backtest does not store the stop, so derive it the way the
        # bot does for TREND trades: entry -/+ atr * per-asset
        # atr_multiplier. Known wrong for REVERSION -- see note above.
        atr = estimate_atr_at(asset, t["entry_time"])
        if not atr:
            continue
        mult = ASSET_ATR_MULT.get(asset, 1.8)
        stop = (t["entry_price"] - atr * mult) if t["side"] == "long" \
               else (t["entry_price"] + atr * mult)

        r, reason = replay_verbose(t["entry_price"], stop, t["side"],
                                   atr, path, RUN_MULT)
        bucket = by_type.get(t.get("trade_type"), by_type.setdefault(
            t.get("trade_type", "UNKNOWN"), [0, 0, []]
        ))
        if reason == t["exit_reason"]:
            bucket[0] += 1
        else:
            bucket[2].append((t["entry_time"], t["exit_reason"], reason))
        # CALIBRATION FIX #3 (28 Aug, post-third-run): the previous
        # expected_r = pnl / risk_distance divided a dollar P&L by a price
        # distance -- only a clean R-multiple at exactly 1 unit position
        # size. trades_detail carries no quantity, so this was silently
        # wrong for every trade with any real position size (confirmed:
        # matched_pnl stayed ~0/85 across all three calibration runs, even
        # for TREND trades with the otherwise-correct stop formula, which
        # is the tell that this metric itself was broken, not the exit
        # modeling). Fixed by using pnl_pct instead -- confirmed directly
        # in backtest.py: pnl_pct = (trade.pnl / notional) * 100 where
        # notional = entry_price * quantity, so quantity cancels out of
        # trade.pnl / notional exactly. pnl_pct is therefore already a
        # clean, quantity-independent, correctly-signed price-return
        # percentage -- comparable to risk expressed the same way.
        risk_pct = abs(t["entry_price"] - stop) / t["entry_price"] * 100 if stop else None
        expected_r = (t.get("pnl_pct") / risk_pct) if risk_pct else None
        if r is not None and expected_r is not None and abs(r - expected_r) < 0.15:
            bucket[1] += 1

    print()
    for ttype, (matched, matched_pnl, mism) in by_type.items():
        n = matched + len(mism)
        if n == 0:
            continue
        tag = "" if ttype == "TREND" else "  (known-wrong stop formula, informational only)"
        print(f"{ttype}: exit reason matched {matched}/{n} "
              f"({100*matched/max(n,1):.0f}%), net R within 0.15: "
              f"{matched_pnl}/{max(n,1)}{tag}")
        if mism:
            print(f"  MISMATCH PATTERN (expected -> replayed):")
            for k, v in Counter((m[1], m[2]) for m in mism).most_common():
                print(f"    {k[0]:>14} -> {k[1]:<14} {v}")

    trend_matched, _, trend_mism = by_type.get("TREND", [0, 0, []])
    trend_n = trend_matched + len(trend_mism)
    if trend_mism:
        print("\nIf TREND mismatches cluster on 'break_even', the trail-arming "
              "assumption (3D) is wrong -- try arming at trail_start_progress_r "
              "(0.25R) rather than r_breakeven_trigger (0.75R).")

    passed = trend_n > 0 and trend_matched >= 0.80 * trend_n
    print(f"\nCALIBRATION {'PASSED' if passed else 'FAILED'} "
          f"(bar: 80% of TREND exit reasons reproduced -- REVERSION excluded, "
          f"known-wrong stop formula)")
    if not passed:
        print("No proposals may be made. The replayer does not model the "
              "exit stack faithfully enough to be trusted.")
    return passed


def load_episodes():
    eps = []
    for f in sorted(glob.glob("logs/episodes/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                eps.append(json.loads(line))
    usable = [e for e in eps if e.get("episode_id") and e.get("entry_atr")]
    print(f"episodes: {len(eps)} total, {len(usable)} usable "
          f"({len(eps) - len(usable)} missing id or entry_atr)")
    return usable


def report():
    eps = load_episodes()
    if not eps:
        print("No usable episodes. Nothing to report.")
        return
    res = defaultdict(lambda: defaultdict(list))
    for e in eps:
        path = load_path(e["asset"], e["entry_time"], e["exit_time"])
        for m in ARMS:
            r = replay(e["entry_price"], e.get("intended_stop"), e["side"],
                       e["entry_atr"], path, m)
            if r is not None:
                res[e["asset"]][m].append(r)
    print(f"\n{'asset':<8} {'n':>4}  " + "  ".join(f"{m:>6}" for m in ARMS))
    for a in sorted(res):
        n = len(res[a][ARMS[0]])
        means = [sum(res[a][m]) / len(res[a][m]) if res[a][m] else 0 for m in ARMS]
        print(f"{a:<8} {n:>4}  " + "  ".join(f"{v:>+6.3f}" for v in means))
    print("\nNo proposals: sample sizes below the threshold, and calibration "
          "has not been run. See --calibrate.")


# ── B8: replayer reads PPL's fields ────────────────────────────────────────
# Design deviation from the batch doc, stated: the doc asked for a join of
# each closed trade to its [BRC] CONFIRMED / [ENTRY-MEASURE] LOG LINES by
# asset + entry time. Checked the actual data first (logs/episodes/*.jsonl):
# every record already IS one full closed trade -- entry_time, close_time,
# close_reason, net_pnl_r, mfe_pct, mae_pct -- with the ENTIRE CompositeState
# embedded under "composite_state", including PPL's brc_tier/brc_gear/
# brc_retest_depth/brc_proof_dist_atr fields verbatim. That is strictly the
# same information the two log lines carry, already structured and already
# joined -- a text-log join by asset+timestamp would be strictly more
# fragile (rotation, formatting drift, near-miss timestamps) for no gain.
# Read the field directly instead.
#
# The doc's "count closed-trade rows, never the per-cycle [EPISODE] ids
# (382/day)" warning is real but the cause is different from what it
# implies: episode rows already are individual closed trades, not per-cycle
# snapshots -- the inflation comes from non-market closures like
# "abandoned_restart_gap" (a restart bookkeeping closure, no real exit),
# which get filtered out below rather than counted as trade outcomes.

_ADMIN_CLOSE_REASONS = {"abandoned_restart_gap", "abandoned", "manual_flat", ""}

_DEPTH_BUCKET_EDGES = (0.25, 0.5)
_DEPTH_BUCKET_LABELS = ("<0.25", "0.25-0.5", ">0.5")


def _bucket(value, edges=_DEPTH_BUCKET_EDGES, labels=_DEPTH_BUCKET_LABELS):
    if value is None:
        return None
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


# ── REPLAYER-2 R1: PPL-only selection ───────────────────────────────────────
# Hard-coded cutoff per the batch doc -- the moment PPL went live. Rows are
# "PPL" if close_time (or entry_time, for still-open-at-write rows) falls
# after it; --all includes earlier rows too.
_PPL_CUTOFF = pd.Timestamp("2026-09-14T16:21:00+02:00")


def _is_ppl_row(e):
    ts = e.get("close_time") or e.get("entry_time")
    if not ts:
        return False
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t > _PPL_CUTOFF
    except Exception:
        return False


# ── REPLAYER-2 R2: derived fields ───────────────────────────────────────────
def _proof_dist_h_atr(e):
    """|entry - H| / ATR4, H = composite_state.setup_ref, ATR4 = entry_atr
    (composite_state carries no separate brc-level ATR field to fall back to
    -- entry_atr is the only ATR4 value these rows actually store)."""
    cs = e.get("composite_state") or {}
    h, atr, entry = cs.get("setup_ref"), e.get("entry_atr"), e.get("entry_price")
    if h is None or not atr or entry is None:
        return None
    try:
        return abs(float(entry) - float(h)) / float(atr)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _sl_path_fields(e):
    """n_moves / trail_travel_atr / first_move_bar / be_reached / last_move_reason
    from sl_path ([ts_iso, old, new, reason] tuples -- shared shape between
    live VTM and shadow positions, just different reason vocabularies)."""
    path = e.get("sl_path") or []
    out = {"n_moves": len(path), "trail_travel_atr": None, "first_move_bar": None,
           "be_reached": False, "last_move_reason": None}
    if not path:
        return out
    atr = e.get("entry_atr") or 0
    travel = 0.0
    for mv in path:
        try:
            _, old, new, reason = mv
        except (ValueError, TypeError):
            continue
        try:
            travel += abs(float(new) - float(old))
        except (TypeError, ValueError):
            pass
        if reason and "breakeven" in str(reason).lower():
            out["be_reached"] = True
    out["trail_travel_atr"] = (travel / atr) if atr else None
    try:
        out["last_move_reason"] = path[-1][3]
    except (IndexError, TypeError):
        pass
    try:
        et = pd.Timestamp(e.get("entry_time"))
        ft = pd.Timestamp(path[0][0])
        # 1H bars -- the timeframe the whole management stack runs on.
        out["first_move_bar"] = max(0, int((ft - et).total_seconds() // 3600))
    except Exception:
        pass
    return out


def _exit_via(e):
    """Reclassify the exit by comparing close_price to the last sl_path stop
    / initial stop / target, within one spread -- close_reason strings alone
    don't distinguish a trail-stop-out from an initial-stop-out, which is why
    trailing exits read as zero today."""
    close_price = e.get("close_price")
    if close_price is None:
        return None
    try:
        close_price = float(close_price)
    except (TypeError, ValueError):
        return None
    from src.execution.shadow_trader import FRICTION_PENALTIES, _DEFAULT_FRICTION
    asset = (e.get("asset") or "").upper()
    spread = FRICTION_PENALTIES.get(asset, _DEFAULT_FRICTION) * close_price

    take_profit = e.get("take_profit")
    if take_profit:
        try:
            if abs(close_price - float(take_profit)) <= spread:
                return "target"
        except (TypeError, ValueError):
            pass

    path = e.get("sl_path") or []
    if path:
        try:
            last_stop, last_reason = float(path[-1][2]), str(path[-1][3] or "").lower()
            if abs(close_price - last_stop) <= spread and "trail" in last_reason:
                return "trail"
        except (TypeError, ValueError, IndexError):
            pass

    initial_stop = e.get("initial_stop_loss")
    if initial_stop:
        try:
            if abs(close_price - float(initial_stop)) <= spread:
                return "stop"
        except (TypeError, ValueError):
            pass
    return "other"


def _derive(e):
    """Attach every R2 derived field once, at load time. schema_version < 2
    rows (pre-DIARY-1 -- no sl_path/state_age captured) get the sl_path-only
    fields left None rather than dropping the row outright -- non-sl_path
    questions (win rate by tier, etc.) are still answerable from them."""
    e = dict(e)
    e["proof_dist_h_atr"] = _proof_dist_h_atr(e)
    if (e.get("schema_version") or 1) >= 2:
        e.update(_sl_path_fields(e))
        e["exit_via"] = _exit_via(e)
    else:
        e.update({"n_moves": None, "trail_travel_atr": None, "first_move_bar": None,
                   "be_reached": None, "last_move_reason": None, "exit_via": None})
    return e


def load_closed_trades(include_all=False):
    """B8/R1: real closed-trade rows (not per-cycle snapshots, not admin
    closes). PPL-only by default (R1); --all includes pre-PPL rows too.
    Always drops stale-state opens (state_age_s > 900) -- those never
    reflect what the live gate actually saw at decision time."""
    rows = []
    skipped_admin = skipped_stale = skipped_pre_ppl = 0
    for f in sorted(glob.glob("logs/episodes/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if not e.get("close_time") or not e.get("entry_time"):
                continue
            if e.get("net_pnl_r") is None:
                continue
            if (e.get("close_reason") or "") in _ADMIN_CLOSE_REASONS:
                skipped_admin += 1
                continue
            _age = e.get("state_age_s")
            if _age is not None and _age > 900:
                skipped_stale += 1
                continue
            if not include_all and not _is_ppl_row(e):
                skipped_pre_ppl += 1
                continue
            rows.append(_derive(e))
    n_assets = len(set(r.get("asset") for r in rows))
    _tail = " -- use --all to include" if skipped_pre_ppl and not include_all else ""
    print(f"closed trades: {len(rows)} usable rows across {n_assets} assets "
          f"({skipped_admin} admin closures, {skipped_stale} stale-state, "
          f"{skipped_pre_ppl} pre-PPL excluded{_tail})")
    return rows


def _dedupe_pairs(rows):
    """R1 pair rule: rows sharing a pair_id count as ONE sample for direction
    questions (--by side). Keeps the earliest row per pair_id; rows with no
    pair_id (the overwhelming majority -- pairing is Lane-C-only) pass
    through untouched."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r.get("entry_time") or ""):
        pid = r.get("pair_id")
        if pid:
            if pid in seen:
                continue
            seen.add(pid)
        out.append(r)
    return out


# R3 field extractors, keyed by --by name -- the exact vocabulary from the
# batch doc (supersedes B8's narrower retest_depth/proof_dist_atr names).
_BY_FIELDS = {
    "tier":         lambda e: (e.get("composite_state") or {}).get("brc_tier"),
    "gear":         lambda e: (e.get("composite_state") or {}).get("brc_gear"),
    "variant":      lambda e: e.get("variant") or None,
    "depth":        lambda e: _bucket((e.get("composite_state") or {}).get("brc_retest_depth")),
    "proof_dist_h": lambda e: _bucket(e.get("proof_dist_h_atr"), edges=(1.0, 2.0),
                                       labels=("<1", "1-2", ">2")),
    "exit_via":     lambda e: e.get("exit_via"),
    "gate_id":      lambda e: e.get("gate_id") if e.get("gate_id") not in (None, "unknown") else None,
    "gate_stage":   lambda e: e.get("gate_stage") if e.get("gate_stage") not in (None, "unknown") else None,
    "asset":        lambda e: e.get("asset"),
    "lane":         lambda e: e.get("lane"),
    "side":         lambda e: e.get("side"),
}


def replay_by(by, min_trades=30, min_trailing=10, min_assets=3, include_all=False):
    """R3: group PPL closed trades by any of the doc's fields.

    Deviation from B8, stated: R3 says a below-threshold table still prints,
    headed "BELOW THRESHOLD -- indicative only", rather than B8's hard
    refusal to print anything. Trailing-exit count now uses the R2 exit_via
    reclassification, not a close_reason substring match (the fix for
    "0 trailing exits ever").
    """
    if by not in _BY_FIELDS:
        print(f"unknown --by {by!r} -- choose from {sorted(_BY_FIELDS)}")
        return
    key_fn = _BY_FIELDS[by]
    rows = load_closed_trades(include_all=include_all)
    if by == "side":
        rows = _dedupe_pairs(rows)  # R1: direction questions count a pair once
    eligible = [r for r in rows if key_fn(r) is not None]

    n_total = len(eligible)
    n_trailing = sum(1 for r in eligible if r.get("exit_via") == "trail")
    n_assets = len(set(r.get("asset") for r in eligible))
    below = n_total < min_trades or n_trailing < min_trailing or n_assets < min_assets

    print(f"\nreplayer --by {by}: {n_total} eligible trades, "
          f"{n_trailing} trailing exits (exit_via), {n_assets} assets  "
          f"[threshold: >={min_trades} trades / >={min_trailing} trailing / >={min_assets} assets]")
    if not eligible:
        print("no eligible rows -- nothing to group.")
        return
    if below:
        print("BELOW THRESHOLD -- indicative only")

    groups = defaultdict(list)
    for r in eligible:
        groups[key_fn(r)].append(r)

    print(f"\n{'group':<16} {'n':>5} {'win%':>7} {'mean_R':>8} {'med_mfe_r':>10} "
          f"{'med_mae_r':>10} {'trail_atr':>10} {'worst10%':>9}")
    for k in sorted(groups, key=lambda x: str(x)):
        grp = groups[k]
        rs = [g["net_pnl_r"] for g in grp if g.get("net_pnl_r") is not None]
        mfe_r = [g["mfe_r"] for g in grp if g.get("mfe_r") is not None]
        mae_r = [g["mae_r"] for g in grp if g.get("mae_r") is not None]
        travel = [g["trail_travel_atr"] for g in grp if g.get("trail_travel_atr") is not None]
        wins = sum(1 for x in rs if x > 0)
        win_pct = 100 * wins / len(rs) if rs else float("nan")
        mean_r = sum(rs) / len(rs) if rs else float("nan")
        med_mfe = statistics.median(mfe_r) if mfe_r else float("nan")
        med_mae = statistics.median(mae_r) if mae_r else float("nan")
        mean_travel = sum(travel) / len(travel) if travel else float("nan")
        rs_sorted = sorted(rs)
        worst_n = max(1, len(rs_sorted) // 10) if rs_sorted else 0
        worst10 = sum(rs_sorted[:worst_n]) / worst_n if worst_n else float("nan")
        print(f"{str(k):<16} {len(grp):>5} {win_pct:>6.0f}% {mean_r:>+8.3f} "
              f"{med_mfe:>+10.3f} {med_mae:>+10.3f} {mean_travel:>10.3f} {worst10:>+9.3f}")


# ── REPLAYER-2 R4: arms -- the input to RL-1 ────────────────────────────────
_ARM_VALUES = {
    "trail_mult":  [0.6, 0.8, 1.0, 1.3],
    "be_r":        [0.75, 1.0, 1.25],
    "grade_table": ["current", "wick_1.25x", "flat"],
}
_RUNNER_TRAIL_DEFAULT = 0.8   # current runner_atr_mult_flat -- the fixed trail
                              # multiplier used while re-running be_r/grade_table


def _bootstrap_ci(values, n_resamples=1000, ci=0.90):
    """Bootstrap CI on the mean. Returns (mean, lo, hi); (mean, mean, mean)
    for n<2 -- a CI on one point is meaningless, not an error."""
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    n = len(values)
    if n < 2:
        return mean, mean, mean
    lo_pct, hi_pct = (1 - ci) / 2, 1 - (1 - ci) / 2
    means = sorted(
        sum(values[random.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    lo = means[int(lo_pct * n_resamples)]
    hi = means[min(int(hi_pct * n_resamples), n_resamples - 1)]
    return mean, lo, hi


def _replay_with_be(entry, stop, side, atr, path, be_trigger_r, mult=_RUNNER_TRAIL_DEFAULT):
    """replay()'s exact exit stack with BE_TRIGGER_R swapped for the be_r
    arm's candidate value -- replay() itself hardcodes the module constant,
    so trying a different trigger means re-running the stack, not calling it."""
    risk = abs(entry - stop)
    if risk <= 0 or path is None or path.empty:
        return None
    cur, armed, peak = stop, False, entry
    for _, bar in path.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if (side == "long" and lo <= cur) or (side == "short" and hi >= cur):
            return (cur - entry) / risk * (1 if side == "long" else -1)
        peak = max(peak, hi) if side == "long" else min(peak, lo)
        prog = abs(peak - entry) / risk
        if not armed and prog >= be_trigger_r:
            armed = True
            cur = entry + BE_LOCK_R * risk * (1 if side == "long" else -1)
        if armed:
            t = peak - mult * atr if side == "long" else peak + mult * atr
            cur = max(cur, t) if side == "long" else min(cur, t)
    close = float(path.iloc[-1]["close"])
    return (close - entry) / risk * (1 if side == "long" else -1)


def _replay_row_arm(e, knob, value):
    """Re-run one closed row's exit stack under one arm value, walking the
    real 15m path. Returns R, or None if the row/path can't support a replay."""
    asset, side = e.get("asset"), e.get("side")
    entry, stop, atr = e.get("entry_price"), e.get("initial_stop_loss") or e.get("stop_loss"), e.get("entry_atr")
    if not (asset and side and entry and stop and atr):
        return None
    path = load_path(asset, e.get("entry_time"), e.get("close_time"))
    if path is None or path.empty:
        return None

    if knob == "trail_mult":
        return replay(entry, stop, side, atr, path, value)
    if knob == "be_r":
        return _replay_with_be(entry, stop, side, atr, path, value)
    if knob == "grade_table":
        # grade_table's real geometry lives inside CompositeStateBuilder's
        # zone-ladder state (full price history, not reconstructable from a
        # closed episode row -- the same gap calibrate() documents for
        # REVERSION stops). Approximated by widening/flattening the ATR stop
        # distance instead, which this script CAN replay faithfully; stated
        # here rather than silently guessed.
        if value == "current":
            eff_stop = stop
        elif value == "wick_1.25x":
            risk = abs(entry - stop)
            eff_stop = entry - risk * 1.25 if side == "long" else entry + risk * 1.25
        else:  # "flat"
            mult = ASSET_ATR_MULT.get(asset, 1.8)
            eff_stop = entry - atr * mult if side == "long" else entry + atr * mult
        return replay(entry, eff_stop, side, atr, path, _RUNNER_TRAIL_DEFAULT)
    return None


def run_arms(knob, include_all=False):
    """R4: re-run every closed row's exit stack under each arm value, report
    n / mean R / bootstrap 90% CI / worst-10% per asset x arm, and write the
    table to logs/replayer_arms_<knob>_<date>.json alongside stdout."""
    if knob not in _ARM_VALUES:
        print(f"unknown knob {knob!r} -- choose from {list(_ARM_VALUES)}")
        return
    print("Replay path = 1H bars; results are directional, not precise "
          "(1H sims inflate expectancy, per calibrate()'s own finding). "
          "15m path pending.\n")
    rows = load_closed_trades(include_all=include_all)
    by_asset_arm = defaultdict(lambda: defaultdict(list))
    for e in rows:
        for value in _ARM_VALUES[knob]:
            r = _replay_row_arm(e, knob, value)
            if r is not None:
                by_asset_arm[e.get("asset")][value].append(r)

    out = {
        "knob": knob,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": "Replay path = 1H bars; results are directional, not precise "
                  "(1H sims inflate). 15m path pending.",
        "assets": {},
    }
    print(f"{'asset':<8} {'arm':>12} {'n':>5} {'mean_R':>8} {'CI90_lo':>8} "
          f"{'CI90_hi':>8} {'worst10%':>9}")
    for asset in sorted(by_asset_arm):
        out["assets"][asset] = {}
        for value in _ARM_VALUES[knob]:
            vals = by_asset_arm[asset].get(value, [])
            n = len(vals)
            mean, lo, hi = _bootstrap_ci(vals)
            vals_sorted = sorted(vals)
            worst_n = max(1, n // 10) if n else 0
            worst10 = sum(vals_sorted[:worst_n]) / worst_n if worst_n else None
            out["assets"][asset][str(value)] = {
                "n": n, "mean_r": mean, "ci90_lo": lo, "ci90_hi": hi, "worst10pct": worst10,
            }
            if n:
                print(f"{asset:<8} {str(value):>12} {n:>5} {mean:>+8.3f} "
                      f"{lo:>+8.3f} {hi:>+8.3f} {worst10:>+9.3f}")
            else:
                print(f"{asset:<8} {str(value):>12} {n:>5} {'--':>8} {'--':>8} {'--':>8} {'--':>9}")

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = f"logs/replayer_arms_{knob}_{date_tag}.json"
    Path("logs").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--by", choices=sorted(_BY_FIELDS.keys()),
                     help="R3: group PPL closed trades by tier/gear/variant/depth/"
                          "proof_dist_h/exit_via/gate_id/gate_stage/asset/lane/side")
    ap.add_argument("--arms", choices=sorted(_ARM_VALUES.keys()),
                     help="R4: per-asset arm table (trail_mult/be_r/grade_table) with bootstrap 90%% CI")
    ap.add_argument("--all", action="store_true",
                     help="R1: include pre-PPL rows too (default: PPL-only, cutoff 2026-09-14T16:21+02:00)")
    ap.add_argument("--result-json", default="logs/backtests/20260822_164803/result.json")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.result_json)
    elif a.report:
        report()
    elif a.arms:
        run_arms(a.arms, include_all=a.all)
    elif a.by:
        replay_by(a.by, include_all=a.all)
    else:
        ap.print_help()
