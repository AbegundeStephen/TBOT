"""
TRADE X-RAY — one closed trade, its proof, its levels, its picture.

Read-only. Touches nothing the bot uses: it reads the log files, the trade
events and data/raw/<symbol>_1h.csv, and writes a PNG into logs/xray/.

    python tools\\trade_xray.py BTC
    python tools\\trade_xray.py USOIL
    python tools\\trade_xray.py BTC_long_1790071356      (a position id)
    python tools\\trade_xray.py BTC --tf 15m             (15-minute picture)

What it prints, in the order the proof was built:

    R1   the origin — where the move came from. A 4H close beyond R1 kills the setup.
    H    the reference (ref_2) — the level the setup is built on, and the stop anchor.
    H2   the best close in the trend direction after the break, frozen at the retest.
    c1   the retest close — the pullback that made the setup a RETEST rather than a RUNNER.
    depth |H2 - c1| / |H2 - R1| — how deep the pullback went, as a fraction of the run.
    close-through  stage 3: a close beyond H2 by a tolerance. That is the confirmation.

Then the trade: entry, stop, how the stop was chosen, targets, exit and result.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

SYMBOL = {
    "BTC": "BTCUSDm", "GOLD": "XAUUSDm", "USTEC": "USTECm",
    "EURUSD": "EURUSDm", "USOIL": "USOILm", "GBPAUD": "GBPAUDm",
}

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def log_files():
    fs = sorted(glob.glob("logs/trading_bot.log*"))
    return sorted(fs, key=lambda p: os.path.getmtime(p))


def scan(patterns, asset=None):
    """Every log line matching any pattern, in time order, as (ts, line)."""
    out = []
    rx = [re.compile(p) for p in patterns]
    for f in log_files():
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if asset and asset not in line:
                        continue
                    if any(r.search(line) for r in rx):
                        m = TS.match(line)
                        if m:
                            out.append((m.group(1), line.rstrip()))
        except Exception:
            pass
    return sorted(out)


def trade_events(asset):
    """ENTRY and EXIT events for this asset, newest last."""
    evs = []
    for f in log_files():
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "TRADE_EVENT" not in line or f'"{asset}"' not in line:
                        continue
                    try:
                        d = json.loads(line[line.index("{"):])
                    except Exception:
                        continue
                    if d.get("asset") == asset and d.get("event") in ("ENTRY", "EXIT"):
                        evs.append(d)
        except Exception:
            pass
    evs.sort(key=lambda d: d.get("timestamp") or 0)
    return evs


def pick_trade(asset, want_id=None):
    evs = trade_events(asset)
    pairs = {}
    for e in evs:
        pairs.setdefault(e.get("position_id"), {})[e.get("event")] = e
    closed = [(pid, p) for pid, p in pairs.items() if "ENTRY" in p and "EXIT" in p]
    if want_id:
        for pid, p in closed:
            if pid == want_id:
                return pid, p
        print(f"[XRAY] {want_id} not found among closed trades for {asset}")
        return None, None
    if not closed:
        print(f"[XRAY] no closed trade found for {asset}")
        return None, None
    closed.sort(key=lambda kv: kv[1]["EXIT"].get("timestamp") or 0)
    return closed[-1]


def num(line, key, cast=float):
    m = re.search(re.escape(key) + r"=(-?[\d.]+(?:e-?\d+)?)", line)
    try:
        return cast(m.group(1)) if m else None
    except Exception:
        return None


def before(rows, when, n=1):
    """The last n rows at or before a timestamp string."""
    keep = [r for r in rows if r[0] <= when]
    return keep[-n:] if keep else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="asset (BTC) or position id (BTC_long_1790071356)")
    ap.add_argument("--tf", default="1h", choices=["1h", "15m", "4h"])
    ap.add_argument("--bars", type=int, default=120, help="bars of context before entry")
    args = ap.parse_args()

    if "_" in args.target and args.target.split("_")[0] in SYMBOL:
        asset, want_id = args.target.split("_")[0], args.target
    else:
        asset, want_id = args.target.upper(), None
    if asset not in SYMBOL:
        print(f"[XRAY] unknown asset {asset}. One of: {', '.join(SYMBOL)}")
        return 2

    pid, pair = pick_trade(asset, want_id)
    if not pid:
        return 1
    ent, ext = pair["ENTRY"], pair["EXIT"]
    t_in, t_out = ent["datetime"], ext["datetime"]
    side = (ent.get("side") or "").upper()

    print("=" * 78)
    print(f"TRADE X-RAY  {pid}")
    print("=" * 78)
    print(f"  {side} {asset}   in {t_in} @ {ent.get('price')}   "
          f"out {t_out} @ {ext.get('price')}")
    print(f"  result: {ext.get('pnl')}   reason: {ext.get('reason')}   size: {ent.get('size')}")

    # ---------- the proof, in build order ----------
    brc = before(scan([r"\[BRC\] " + asset + r": CONFIRMED"], asset), t_in, 1)
    born = before(scan([r"\[SETUP-BORN\] " + asset], asset), t_in, 3)
    r1 = before(scan([r"\[R1-ORIGIN\] " + asset], asset), t_in, 3)
    c1 = before(scan([r"\[COUNT-1\] " + asset], asset), t_in, 3)
    c2 = before(scan([r"\[COUNT-2\] " + asset], asset), t_in, 3)
    c3 = before(scan([r"\[COUNT-3-CHECK\] " + asset + r".*-> yes"], asset), t_in, 2)

    print("\n--- THE PROOF ------------------------------------------------------------")
    for label, rows in (("setup born", born), ("origin R1", r1), ("stage 1 break", c1),
                        ("stage 2 retest", c2), ("stage 3 close-through", c3),
                        ("CONFIRMED", brc)):
        if rows:
            for ts, line in rows:
                print(f"  {label:<22} {ts}  {line.split(' - INFO - ')[-1]}")
        else:
            print(f"  {label:<22} (not found in the saved logs)")

    lv = {}
    if brc:
        b = brc[-1][1]
        lv["H (ref)"] = num(b, "ref")
        lv["H2"] = num(b, "h2")
        lv["close at confirm"] = num(b, "close")
        print("\n  reading of the confirm line:")
        print(f"    H (ref) = {lv['H (ref)']}   the level the setup is built on, and the stop anchor")
        print(f"    H2      = {lv['H2']}   best close after the break, frozen at the retest")
        print(f"    close   = {lv['close at confirm']}   the close-through that confirmed it")
        print(f"    tier    = {re.search(r'tier=(\\w+)', b).group(1) if re.search(r'tier=(\\w+)', b) else '?'}"
              "   RETEST = it pulled back and came back; RUNNER = it never pulled back")
        print(f"    depth   = {num(b, 'depth')}   pullback as a fraction of the run")
        print(f"    dist    = {num(b, 'dist')}   |entry - H| in ATR4 — how far price had already travelled")
        print(f"    age     = {num(b, 'age', int)} candles since the break")
    if r1:
        lv["R1 (origin)"] = num(r1[-1][1], "R1")
    if c2:
        lv["c1 (retest close)"] = num(c2[-1][1], "close1")
        if lv.get("H2") is None:
            lv["H2"] = num(c2[-1][1], "H2")

    # ---------- how the stop was chosen ----------
    stop_rows = before(scan([r"\[TIER-STOP\] " + asset, r"\[STOP-PICK\] " + asset,
                             r"\[STOP-FINAL\] " + asset, r"\[MIN-SL\] " + asset], asset),
                       (datetime.strptime(t_in, "%Y-%m-%d %H:%M:%S")
                        + timedelta(seconds=90)).strftime("%Y-%m-%d %H:%M:%S"), 8)
    print("\n--- THE STOP AND TARGETS -------------------------------------------------")
    for ts, line in stop_rows:
        print(f"  {ts}  {line.split(' - INFO - ')[-1]}")
    for ts, line in stop_rows:
        if "[STOP-PICK]" in line:
            lv["stop"] = num(line, "stop")
            lv["entry (bot)"] = num(line, "entry")

    # ---------- the council's own scorecard ----------
    sc = before(scan([r"SCORE: +" + asset], asset), t_in, 1)
    bar = before(scan([r"\[BAR\] " + asset], asset), t_in, 2)
    print("\n--- THE COUNCIL ----------------------------------------------------------")
    for ts, line in (bar + sc):
        print(f"  {ts}  {line.split(' - INFO - ')[-1]}")

    # ---------- what happened after entry ----------
    mgmt = [r for r in scan([r"\[STOP-H\] " + asset, r"manual SL", r"\[VTM\].*" + asset + r".*SL",
                             r"\[RECONCILE\].*" + asset, r"\[CLOSE\].*" + asset], asset)
            if t_in <= r[0] <= t_out]
    print("\n--- AFTER ENTRY ----------------------------------------------------------")
    for ts, line in mgmt[:20]:
        print(f"  {ts}  {line.split(' - ')[-1]}")
    if not mgmt:
        print("  (no stop moves or interventions logged)")

    print("\n--- LEVELS ---------------------------------------------------------------")
    for k in ("R1 (origin)", "H (ref)", "H2", "c1 (retest close)", "close at confirm",
              "entry (bot)", "stop"):
        if lv.get(k) is not None:
            print(f"  {k:<20} {lv[k]}")
    print(f"  {'exit':<20} {ext.get('price')}")

    # ---------- the picture ----------
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        path = f"data/raw/{SYMBOL[asset]}_{args.tf}.csv"
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                    else df.index.tz_convert("UTC"))
        # log stamps are box time (UTC+2); the files are UTC
        t0 = pd.Timestamp(t_in).tz_localize("UTC") - pd.Timedelta(hours=2)
        t1 = pd.Timestamp(t_out).tz_localize("UTC") - pd.Timedelta(hours=2)
        step = df.index.to_series().diff().median()
        win = df.loc[t0 - step * args.bars: t1 + step * 10]
        if win.empty:
            print("\n[XRAY] no price rows in that window — chart skipped")
            return 0

        fig, ax = plt.subplots(figsize=(16, 9))
        w = (step.total_seconds() / 86400.0) * 0.6
        for t, r in win.iterrows():
            up = r["close"] >= r["open"]
            col = "#26a69a" if up else "#ef5350"
            x = mdates.date2num(t)
            ax.plot([x, x], [r["low"], r["high"]], color=col, linewidth=0.8, zorder=2)
            ax.add_patch(plt.Rectangle((x - w / 2, min(r["open"], r["close"])), w,
                                       max(abs(r["close"] - r["open"]), 1e-9),
                                       facecolor=col, edgecolor=col, zorder=3))

        styles = {
            "R1 (origin)": ("#9c27b0", "-."), "H (ref)": ("#1565c0", "-"),
            "H2": ("#ef6c00", "--"), "c1 (retest close)": ("#6d4c41", ":"),
            "entry (bot)": ("#000000", "-"), "stop": ("#c62828", "-"),
        }
        for k, (col, ls) in styles.items():
            if lv.get(k) is not None:
                ax.axhline(lv[k], color=col, linestyle=ls, linewidth=1.4, zorder=4)
                ax.annotate(f"{k}  {lv[k]:.5g}", xy=(1.002, lv[k]),
                            xycoords=("axes fraction", "data"), color=col,
                            fontsize=9, va="center")
        ax.axhline(float(ext.get("price")), color="#2e7d32", linewidth=1.4, zorder=4)
        ax.annotate(f"exit  {ext.get('price')}", xy=(1.002, float(ext.get("price"))),
                    xycoords=("axes fraction", "data"), color="#2e7d32", fontsize=9, va="center")

        for when, label, col in ((t0, f"ENTRY {side}", "#000000"), (t1, "EXIT", "#2e7d32")):
            ax.axvline(when, color=col, linestyle="--", linewidth=1.0, alpha=0.7, zorder=5)
            ax.annotate(label, xy=(when, ax.get_ylim()[1]), rotation=90,
                        fontsize=9, color=col, va="top", ha="right")

        for rows, label, col in ((c1, "break", "#1565c0"), (c2, "retest", "#6d4c41"),
                                 (c3, "close-through", "#ef6c00")):
            if rows:
                when = pd.Timestamp(rows[-1][0]).tz_localize("UTC") - pd.Timedelta(hours=2)
                if win.index.min() <= when <= win.index.max():
                    ax.axvline(when, color=col, linestyle=":", linewidth=1.0, alpha=0.8, zorder=5)
                    ax.annotate(label, xy=(when, ax.get_ylim()[0]), rotation=90,
                                fontsize=8, color=col, va="bottom", ha="right")

        ax.set_title(f"{asset} {side}  {t_in} → {t_out}   result {ext.get('pnl')}   "
                     f"({args.tf} candles, times UTC)", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.subplots_adjust(right=0.86)

        os.makedirs("logs/xray", exist_ok=True)
        out = f"logs/xray/{pid}_{args.tf}.png"
        fig.savefig(out, dpi=130)
        print(f"\n[XRAY] chart written: {out}")
    except ImportError as e:
        print(f"\n[XRAY] chart skipped (missing library: {e}) — the reading above still stands")
    except Exception as e:
        print(f"\n[XRAY] chart failed ({e}) — the reading above still stands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
