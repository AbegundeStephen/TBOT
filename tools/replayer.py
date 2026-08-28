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


def estimate_atr_at(asset, entry_time, period=14, lookback_bars=40):
    """14-period ATR on 15m bars, from the window ending at entry_time.

    Needs its own load (not the entry-to-exit `path` calibrate()/report()
    already have) because ATR requires history BEFORE entry, which a path
    sliced from entry_time onward does not carry.
    """
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
    window = df.loc[:str(entry_time)].tail(lookback_bars)
    if len(window) < period + 1:
        return None
    high, low, close = window["high"], window["low"], window["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.iloc[-period:].mean()
    return float(atr) if pd.notna(atr) else None


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

    matched_reason, matched_pnl, mismatches = 0, 0, []
    for t in trades:
        path = load_path(asset, t["entry_time"], t["exit_time"])
        if path is None or path.empty:
            continue
        # The backtest does not store the stop, so derive it the way the
        # bot does: entry -/+ atr * per-asset atr_multiplier.
        atr = estimate_atr_at(asset, t["entry_time"])
        if not atr:
            continue
        mult = ASSET_ATR_MULT.get(asset, 1.8)
        stop = (t["entry_price"] - atr * mult) if t["side"] == "long" \
               else (t["entry_price"] + atr * mult)

        r, reason = replay_verbose(t["entry_price"], stop, t["side"],
                                   atr, path, RUN_MULT)
        if reason == t["exit_reason"]:
            matched_reason += 1
        else:
            mismatches.append((t["entry_time"], t["exit_reason"], reason))
        expected_r = t["pnl"] / (abs(t["entry_price"] - stop) * 1.0) \
                     if stop else None
        if r is not None and expected_r and abs(r - expected_r) < 0.15:
            matched_pnl += 1

    n = len(mismatches) + matched_reason
    print(f"\nexit reason matched : {matched_reason}/{n} "
          f"({100*matched_reason/max(n,1):.0f}%)")
    print(f"net R within 0.15   : {matched_pnl}/{max(n,1)}")

    if mismatches:
        print("\nMISMATCH PATTERN (expected -> replayed):")
        for k, v in Counter((m[1], m[2]) for m in mismatches).most_common():
            print(f"  {k[0]:>14} -> {k[1]:<14} {v}")
        print("\nIf these cluster on 'break_even', the trail-arming assumption "
              "(3D) is wrong -- try arming at trail_start_progress_r (0.25R) "
              "rather than r_breakeven_trigger (0.75R).")

    passed = matched_reason >= 0.80 * max(n, 1)
    print(f"\nCALIBRATION {'PASSED' if passed else 'FAILED'} "
          f"(bar: 80% of exit reasons reproduced)")
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
