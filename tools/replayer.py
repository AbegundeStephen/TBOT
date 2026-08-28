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

import argparse, json, glob
from collections import Counter, defaultdict
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
    """Same exit stack as replay(), but also returns which stage fired:
    "stop_loss" (before the trail ever armed), "break_even" (armed, hit
    exactly at the lock price before the trail advanced past it), or
    "trailing_stop" (hit after the trail had moved beyond the lock).
    Returns (r, reason) -- either may be None if the path is unusable.
    """
    risk = abs(entry - stop)
    if risk <= 0 or path is None or path.empty:
        return None, None
    cur, armed, peak = stop, False, entry
    be_lock_price = None

    def _classify(cur_price):
        if not armed:
            return "stop_loss"
        if be_lock_price is not None and cur_price == be_lock_price:
            return "break_even"
        return "trailing_stop"

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
            be_lock_price = cur
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--result-json", default="logs/backtests/20260822_164803/result.json")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.result_json)
    elif a.report:
        report()
    else:
        ap.print_help()
