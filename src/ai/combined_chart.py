"""B13 item 9A (Desire 1, 5 and 6 Oct 2026): THE COMBINED WHITE CHART -- one picture per market, four panels
(1H, 4H, 1D, 1W), every layer in one place:
  - candles; EMAs 20/50/200 on 1H and 4H, 50/200 on 1D and 1W
  - 4H swing zones (from the swing close to its wick) -- the lines the bot trades
  - wick lines: old 4H highs/lows measured at the wick tip -- DASHED, their own colour (B13 item 33's "old high in front")
  - both Livermore brains' lines (1H and 4H): main up-leg high, main down-leg low, natural high, natural low
  - 3-touch diagonal trendlines (solid while unbroken, faded once broken)
  - live setups (R2 = the line to break, R1 = the kill line) and the latest proof's B / R / E markers
  - a short card: the latest proof or the most advanced setup, with the diagonal tag
1D and 1W carry no Livermore state (the R2 calibration decides that). Display only -- nothing here is read by trading
code. Redrawn only when a candle, a proof or a setup changed. Output: logs/charts/<ASSET>_combined.png

Dashboard item (Oct 2026): write_combined also writes one chart per timeframe (logs/charts/<ASSET>_<TF>.png,
TF in 1H/4H/1D/1W) via write_singles, so the dashboard's Charts tab can toggle between timeframes and load
just the one requested instead of the whole four-panel picture. Same data, same redraw-on-change rule, no
second MT5 fetch -- write_combined hands its own D1/W1 frames straight to write_singles."""
import os
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
K = 2
COL = {"zone": "#9ec5e8", "r2": "#1565c0", "r1": "#c62828", "wick": "#ef6c00", "diag": "#6a1b9a",
       "main": "#2e7d32", "natural": "#f9a825", "brain1h": "#e57373",
       "ema20": "#26a69a", "ema50": "#5c6bc0", "ema200": "#8d6e63", "up": "#2e7d32", "down": "#c62828"}
_MEMO = {}
_MEMO_SINGLE = {}
_TFS = ("1H", "4H", "1D", "1W")


def _frame(df):
    h = df.copy()
    if "timestamp" in h.columns:
        h.index = pd.to_datetime(h["timestamp"])
    h.index = pd.DatetimeIndex(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_convert("UTC").tz_localize(None)
    return h[["open", "high", "low", "close"]].astype(float).sort_index()


def _resample(h1, rule):
    return h1.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def _mt5_frame(symbol, which, n):
    """Daily / weekly candles straight from MT5 (read-only); None if MT5 can't give them."""
    try:
        import MetaTrader5 as mt5
        tf = {"D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1}[which]
        r = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        if r is None or len(r) < 30:
            return None
        df = pd.DataFrame(r)
        df.index = pd.to_datetime(df["time"], unit="s")
        return df[["open", "high", "low", "close"]].astype(float)
    except Exception:
        return None


def _atr14(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).rolling(14, min_periods=1).mean().values


def _swings(t4c, hi, lo, c):
    """4H swings: (confirmed, type, close level, wick edge) on closes, and (confirmed, type, wick) on wicks."""
    sw, sww = [], []
    for i in range(K, len(c) - K):
        conf = t4c[i + K]
        w = c[i - K:i + K + 1]
        if c[i] == w.max():
            sw.append((conf, "H", float(c[i]), float(hi[i - K:i + K + 1].max())))
        if c[i] == w.min():
            sw.append((conf, "L", float(c[i]), float(lo[i - K:i + K + 1].min())))
        if hi[i] == hi[i - K:i + K + 1].max():
            sww.append((conf, "H", float(hi[i])))
        if lo[i] == lo[i - K:i + K + 1].min():
            sww.append((conf, "L", float(lo[i])))
    return sw, sww


def _diagonals(t4o, a4, c4, life=60):
    """3-touch diagonals on the 4H closes -- the same finder the engine's diagonal rule uses (display copy)."""
    n = len(c4)
    piv = {"H": [], "L": []}
    for i in range(K, n - K):
        w = c4[i - K:i + K + 1]
        if c4[i] == w.max():
            piv["H"].append(i)
        if c4[i] == w.min():
            piv["L"].append(i)
    out = []
    for d, typ in ((1, "H"), (-1, "L")):
        P = piv[typ]
        for j in range(1, len(P)):
            i1, i2 = P[j - 1], P[j]
            v1, v2 = c4[i1], c4[i2]
            if (d == 1 and not v2 < v1) or (d == -1 and not v2 > v1):
                continue
            slope = (v2 - v1) / (i2 - i1)
            nxt = P[j + 1] if j + 1 < len(P) else n
            start = i2 + K
            end = min(n, start + life, nxt + K)
            earlier = P[max(0, j - 5):j - 1]
            vf = start if any(a4[p] == a4[p] and abs(c4[p] - (v1 + slope * (p - i1))) <= 0.5 * a4[p] for p in earlier) else None
            brk = None
            for x in range(start, end):
                lv = v1 + slope * (x - i1)
                if vf is None:
                    if x >= i2 + 3 and a4[x] == a4[x] and abs(c4[x] - lv) <= 0.5 * a4[x]:
                        vf = x + 1
                    continue
                if x >= vf and a4[x] == a4[x] and a4[x] > 0 and d * (c4[x] - lv) >= 0.25 * a4[x]:
                    brk = x
                    break
            if vf is not None:
                stop = brk if brk is not None else min(end, n - 1)
                out.append(dict(d=d, x0=t4o[i1], x1=t4o[stop], y0=v1, y1=v1 + slope * (stop - i1), broken=brk is not None))
    return out


def _brain_lines(asset, h1, h4):
    """The bot's own Livermore brains (1H and 4H), replayed on the candles shown: their four lines at the end."""
    out = {}
    try:
        try:
            from src.execution import livermore_state_machine as m
        except Exception:
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("lsm_cc", os.path.join("src", "execution", "livermore_state_machine.py"))
            m = importlib.util.module_from_spec(spec)
            sys.modules["lsm_cc"] = m
            spec.loader.exec_module(m)
        try:
            piv = json.load(open(os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig")).get("LIVERMORE_PIVOTS", {})
        except Exception:
            piv = {}
        m4, m1 = m.make_livermore_pair(asset, piv.get(asset, {}))
        for lab, mach, df in (("4H", m4, h4), ("1H", m1, h1)):
            atr = m.atr14(df)
            snap = None
            for c, a in zip(df["close"].values, atr.values):
                try:
                    snap = mach.update(float(c), float(a))
                except Exception:
                    pass
            if snap is not None:
                out[lab] = {"state": str(snap.state), "MAIN UP max": snap.anchor_main_up_max,
                            "MAIN DOWN min": snap.anchor_main_down_min, "natural high": snap.anchor_natural_high,
                            "natural low": snap.anchor_natural_low}
    except Exception as e:
        logger.debug("[COMBINED-CHART] %s: brains skipped: %s", asset, e)
    return out


def _candles(ax, df, width_days, mdates, Rectangle):
    for ts, r in df.iterrows():
        x = mdates.date2num((ts + pd.Timedelta(days=width_days / 2)).to_pydatetime())
        col = COL["up"] if r["close"] >= r["open"] else COL["down"]
        ax.plot([x, x], [r["low"], r["high"]], color=col, lw=0.7, zorder=2)
        ax.add_patch(Rectangle((x - width_days * 0.35, min(r["open"], r["close"])), width_days * 0.7,
                               max(abs(r["close"] - r["open"]), 1e-12), fc=col, ec=col, zorder=3))


def _prepare(asset, df1, cs, now=None):
    """Everything computed once per asset per redraw -- candles, swings, diagonals, brain lines, proofs/setups.
    Shared by the combined 4-panel chart (render) and every single-timeframe chart (render_single)."""
    h1 = _frame(df1)
    h4 = _resample(h1, "4h")
    h4 = h4[h4.index + pd.Timedelta(hours=4) <= h1.index[-1] + pd.Timedelta(hours=1)]
    now = pd.Timestamp(now) if now is not None else h1.index[-1] + pd.Timedelta(hours=1)
    hi4, lo4, c4 = h4["high"].values, h4["low"].values, h4["close"].values
    a4 = _atr14(hi4, lo4, c4)
    t4c = (h4.index + pd.Timedelta(hours=4)).to_numpy()
    sw, sww = _swings(t4c, hi4, lo4, c4)
    diags = _diagonals(h4.index.to_numpy(), a4, c4)
    brains = _brain_lines(asset, h1[h1.index >= now - pd.Timedelta(days=60)], h4[h4.index >= now - pd.Timedelta(days=60)])
    last = float(h1["close"].iloc[-1])
    a4n = float(a4[-1]) if len(a4) and a4[-1] == a4[-1] else abs(last) * 0.01
    band = (last - 4 * a4n, last + 4 * a4n)
    near = lambda v: v is not None and v == v and band[0] <= float(v) <= band[1]
    cs = cs or {}
    proofs = cs.get("ns_proofs_hist") or []
    setups = [s for s in (cs.get("ns_setups") or []) if s.get("r2") is not None]
    return dict(h1=h1, h4=h4, now=now, sw=sw, sww=sww, diags=diags, brains=brains,
                last=last, band=band, near=near, proofs=proofs, setups=setups)


def _draw_panel(ax, tf, d1, w1, ctx, asset, mdates, Rectangle):
    """Draw one timeframe panel (candles, EMAs, and -- for 1H/4H only -- zones/diagonals/brain lines/setups/
    proof markers) onto `ax`. Returns False (nothing drawn, caller should skip this panel) when there's no
    data in the lookback window for this timeframe yet."""
    h1, h4, now = ctx["h1"], ctx["h4"], ctx["now"]
    spans = {"1H": (h1, 5, 1 / 24, (20, 50, 200)), "4H": (h4, 40, 4 / 24, (20, 50, 200)),
             "1D": (d1, 270, 1.0, (50, 200)), "1W": (w1, 1100, 7.0, (50, 200))}
    df, days, wd, emas = spans[tf]
    full = df
    df = df[df.index >= now - pd.Timedelta(days=days)]
    if len(df) == 0:
        return False
    _candles(ax, df, wd, mdates, Rectangle)
    for span in emas:
        e = full["close"].ewm(span=span, adjust=False).mean()
        e = e[e.index >= df.index[0]]
        ax.plot([mdates.date2num((t + pd.Timedelta(days=wd / 2)).to_pydatetime()) for t in e.index], e.values,
                color=COL["ema%d" % span], lw=1.1, alpha=0.9, zorder=4)
    x0, x1 = df.index[0], now + pd.Timedelta(days=wd * 6)
    ax.set_xlim(x0, x1)
    lo_, hi_ = df["low"].min(), df["high"].max()
    if tf in ("1H", "4H"):
        near = ctx["near"]
        sw, sww, diags, brains, setups, proofs = ctx["sw"], ctx["sww"], ctx["diags"], ctx["brains"], ctx["setups"], ctx["proofs"]
        t_from = np.datetime64(now - pd.Timedelta(days=60))
        for cf, ty, lv, ed in sw:                                       # the lines the bot trades: zone close -> wick
            if cf >= t_from and near(lv):
                ax.fill_between([max(pd.Timestamp(cf), x0), x1], min(lv, ed), max(lv, ed), color=COL["zone"], alpha=0.35, lw=0, zorder=1)
        for cf, ty, lv in sww:                                          # old highs/lows by wick -- DASHED
            if cf >= t_from and near(lv):
                ax.plot([max(pd.Timestamp(cf), x0), x1], [lv, lv], color=COL["wick"], lw=1.0, ls=(0, (6, 4)), zorder=4)
        for dg in diags:
            if pd.Timestamp(dg["x1"]) >= x0 and (near(dg["y0"]) or near(dg["y1"])):
                ax.plot([pd.Timestamp(dg["x0"]), pd.Timestamp(dg["x1"])], [dg["y0"], dg["y1"]], color=COL["diag"],
                        lw=1.6 if not dg["broken"] else 1.0, alpha=1.0 if not dg["broken"] else 0.35, zorder=5)
        for blab, b in brains.items():
            for k, v in b.items():
                if k == "state" or v is None or not near(v):
                    continue
                col = COL["brain1h"] if blab == "1H" else (COL["main"] if k.startswith("MAIN") else COL["natural"])
                ax.plot([x0, x1], [v, v], color=col, lw=1.3, ls=":" if blab == "1H" else "-", zorder=5)
                ax.text(x1, v, " %s (%s)" % (k, blab), color=col, fontsize=8, va="center")
        for s in setups[:3]:
            r2, r1 = float(s["r2"]), float(s.get("r1") or s["r2"])
            ax.plot([x0, x1], [r2, r2], color=COL["r2"], lw=2.0, zorder=6)
            ax.plot([x0, x1], [r1, r1], color=COL["r1"], lw=1.2, ls="--", zorder=6)
        if proofs:
            f = proofs[-1].get("fields", {})
            for tk, pk, lab, col in (("ns_b_t", "ns_b_px", "B", COL["r2"]), ("ns_touch_t", "ns_touch_px", "R", COL["diag"]),
                                     ("ns_candle", "ns_close", "E", "black")):
                try:
                    tm, px = pd.Timestamp(f[tk]), float(f[pk])
                    if tm >= x0:
                        ax.plot([tm], [px], "o", ms=13, mfc="white", mec=col, mew=2, zorder=8)
                        ax.text(tm, px, lab, color=col, fontsize=8, fontweight="bold", ha="center", va="center", zorder=9)
                except Exception:
                    pass
    pad = (hi_ - lo_) * 0.08
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.axhline(ctx["last"], color="#999999", lw=0.8, ls=":")
    ax.grid(alpha=0.25)
    ax.xaxis_date()
    ax.set_title("%s  %s" % (asset, tf) + ("   (4H brain: %s | 1H brain: %s)" % (
        (ctx["brains"].get("4H") or {}).get("state", "-"), (ctx["brains"].get("1H") or {}).get("state", "-")) if tf == "1H" else "")
                 + ("   no Livermore state on this timeframe" if tf in ("1D", "1W") else ""), loc="left", fontsize=11)
    return True


def _panel_lines(asset, ctx):
    """The latest-proof / watching-setup card text, shared by the combined chart and every single chart."""
    g6 = lambda v: ("%.6g" % float(v)) if v not in (None, "None", "") else "-"
    proofs, setups, now = ctx["proofs"], ctx["setups"], ctx["now"]
    lines = ["%s -- %s UTC" % (asset, str(now)[:16]), ""]
    if proofs:
        f = proofs[-1].get("fields", {})
        lines += ["LATEST PROOF: %s %s" % ("BUY" if int(f.get("setup_dir", 1)) == 1 else "SELL", f.get("ns_entry", "")),
                  "  line %s  entry %s" % (g6(f.get("ns_r2")), g6(f.get("ns_close"))),
                  "  stop %s  target %s" % (g6(f.get("ns_stop")), g6(f.get("ns_target"))),
                  "  %s, candle %s" % (f.get("ns_kind_raw", ""), str(f.get("ns_candle"))[:16]),
                  "  diagonal: %s" % ("agrees" if f.get("ns_diag_agrees") else "none in the 24 h before")]
    elif setups:
        s = sorted(setups, key=lambda x: -int(x.get("stage", 0)))[0]
        lines += ["WATCHING SETUP #%s: %s" % (s.get("id"), "BUY" if int(s.get("d", 1)) == 1 else "SELL"),
                  "  line %s  kill %s" % (g6(s.get("r2")), g6(s.get("r1"))), "  stage %s/3" % (int(s.get("stage", 0)) + 1)]
    else:
        lines += ["no live setup"]
    return lines


def _legend_handles():
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, fc=COL["zone"], alpha=0.5), Line2D([], [], color=COL["wick"], ls=(0, (6, 4))),
               Line2D([], [], color=COL["main"]), Line2D([], [], color=COL["natural"]), Line2D([], [], color=COL["brain1h"], ls=":"),
               Line2D([], [], color=COL["diag"], lw=1.6), Line2D([], [], color=COL["r2"], lw=2), Line2D([], [], color=COL["r1"], ls="--"),
               Line2D([], [], color=COL["ema20"]), Line2D([], [], color=COL["ema50"]), Line2D([], [], color=COL["ema200"]),
               Line2D([], [], marker="o", color="w", mec="black", ms=10)]
    labels = ["4H swing zone (close to wick)", "old high/low by WICK (dashed)", "4H brain: main up/down",
              "4H brain: natural high/low", "1H brain lines", "3-touch diagonal", "setup line (R2)", "kill line (R1)",
              "EMA 20", "EMA 50", "EMA 200", "B break  R retest  E entry"]
    return handles, labels


def _daily_weekly_frames(ctx, d1, w1):
    d1f = _frame(d1) if d1 is not None else _resample(ctx["h1"], "1D")      # fallback only (MT5's own daily is used live)
    w1f = _frame(w1) if w1 is not None else ctx["h1"].resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return d1f, w1f


def render(asset, df1, cs, out_png, d1=None, w1=None, now=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    ctx = _prepare(asset, df1, cs, now=now)
    d1f, w1f = _daily_weekly_frames(ctx, d1, w1)

    fig = plt.figure(figsize=(20, 12.5), facecolor="white")
    axes = {"1H": fig.add_axes([0.04, 0.53, 0.47, 0.40]), "4H": fig.add_axes([0.54, 0.53, 0.30, 0.40]),
            "1D": fig.add_axes([0.04, 0.08, 0.38, 0.37]), "1W": fig.add_axes([0.46, 0.08, 0.38, 0.37])}
    for tf, ax in axes.items():
        _draw_panel(ax, tf, d1f, w1f, ctx, asset, mdates, Rectangle)

    fig.text(0.855, 0.93, "\n".join(_panel_lines(asset, ctx)), family="monospace", fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.7", fc="#fafafa", ec="#999999"))
    leg_handles, leg_labels = _legend_handles()
    fig.legend(leg_handles, leg_labels, loc="lower left", bbox_to_anchor=(0.86, 0.10), ncol=1, frameon=False, fontsize=9)

    tmp = out_png + ".tmp.png"
    fig.savefig(tmp, dpi=80, facecolor="white")
    plt.close(fig)
    os.replace(tmp, out_png)
    return out_png


def render_single(asset, which, df1, cs, out_png, d1=None, w1=None, now=None):
    """One timeframe panel, full canvas -- the dashboard's per-timeframe toggle. Same layers as render(),
    just one panel instead of four, with its own copy of the info card and legend so the image is
    self-contained."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    ctx = _prepare(asset, df1, cs, now=now)
    d1f, w1f = _daily_weekly_frames(ctx, d1, w1)

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_axes([0.055, 0.09, 0.66, 0.84])
    if not _draw_panel(ax, which, d1f, w1f, ctx, asset, mdates, Rectangle):
        plt.close(fig)
        return None

    fig.text(0.745, 0.90, "\n".join(_panel_lines(asset, ctx)), family="monospace", fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.7", fc="#fafafa", ec="#999999"))
    leg_handles, leg_labels = _legend_handles()
    fig.legend(leg_handles, leg_labels, loc="lower left", bbox_to_anchor=(0.745, 0.08), ncol=1, frameon=False, fontsize=9)

    tmp = out_png + ".tmp.png"
    fig.savefig(tmp, dpi=90, facecolor="white")
    plt.close(fig)
    os.replace(tmp, out_png)
    return out_png


def write_combined(asset, df1, cs, symbol=None, out_dir=os.path.join("logs", "charts"), now=None):
    """Draw logs/charts/<ASSET>_combined.png when a candle, a proof or a setup changed; returns the path or None.
    Also redraws each single-timeframe panel via write_singles, on the same change signature and the same
    D1/W1 fetch -- the dashboard's per-timeframe Charts toggle stays current without a second trigger."""
    try:
        if df1 is None or len(df1) < 60:
            return None
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "%s_combined.png" % str(asset).upper())
        if cs is not None and not isinstance(cs, dict):          # the bot passes a dict or a CompositeState object
            cs = {"ns_proofs_hist": getattr(cs, "ns_proofs_hist", None), "ns_setups": getattr(cs, "ns_setups", None)}
        cs = cs or {}
        sig = "|".join(str(x) for x in (df1.index[-1] if "timestamp" not in df1.columns else df1["timestamp"].iloc[-1],
                                        len(cs.get("ns_proofs_hist") or []),
                                        [(s.get("id"), s.get("stage")) for s in (cs.get("ns_setups") or [])]))
        if _MEMO.get(asset) == sig and os.path.exists(out_png):
            return out_png
        d1 = _mt5_frame(symbol, "D1", 400) if symbol else None
        w1 = _mt5_frame(symbol, "W1", 260) if symbol else None
        res = render(str(asset).upper(), df1, cs, out_png, d1=d1, w1=w1, now=now)
        _MEMO[asset] = sig
        logger.info("[COMBINED-CHART] %s: drawn (%s)", asset, out_png)
        write_singles(asset, df1, cs, out_dir=out_dir, now=now, _d1=d1, _w1=w1, _sig=sig)
        return res
    except Exception as e:
        logger.warning("[COMBINED-CHART] %s: not drawn: %s", asset, e)
        return None


def write_singles(asset, df1, cs, symbol=None, out_dir=os.path.join("logs", "charts"), now=None,
                   _d1=None, _w1=None, _sig=None):
    """Dashboard item (Oct 2026): one chart per timeframe instead of the combined 1H/4H/1D/1W picture, so
    the Charts tab can toggle between timeframes and load just the one requested. Writes
    logs/charts/<ASSET>_<TF>.png for TF in 1H/4H/1D/1W. Same redraw-on-change rule as write_combined,
    which calls this directly -- passing the D1/W1 frames it already fetched via _d1/_w1, so this never
    fetches from MT5 twice in the same cycle. Callable on its own too (symbol-based fetch as fallback)."""
    try:
        if df1 is None or len(df1) < 60:
            return {}
        os.makedirs(out_dir, exist_ok=True)
        if cs is not None and not isinstance(cs, dict):
            cs = {"ns_proofs_hist": getattr(cs, "ns_proofs_hist", None), "ns_setups": getattr(cs, "ns_setups", None)}
        cs = cs or {}
        sig = _sig or "|".join(str(x) for x in (df1.index[-1] if "timestamp" not in df1.columns else df1["timestamp"].iloc[-1],
                                        len(cs.get("ns_proofs_hist") or []),
                                        [(s.get("id"), s.get("stage")) for s in (cs.get("ns_setups") or [])]))
        paths = {tf: os.path.join(out_dir, "%s_%s.png" % (str(asset).upper(), tf)) for tf in _TFS}
        if _MEMO_SINGLE.get(asset) == sig and all(os.path.exists(p) for p in paths.values()):
            return paths
        d1 = _d1 if _d1 is not None else (_mt5_frame(symbol, "D1", 400) if symbol else None)
        w1 = _w1 if _w1 is not None else (_mt5_frame(symbol, "W1", 260) if symbol else None)
        out = {}
        for tf, out_png in paths.items():
            res = render_single(str(asset).upper(), tf, df1, cs, out_png, d1=d1, w1=w1, now=now)
            if res:
                out[tf] = res
        _MEMO_SINGLE[asset] = sig
        logger.info("[SINGLE-CHART] %s: drawn %s", asset, list(out.keys()))
        return out
    except Exception as e:
        logger.warning("[SINGLE-CHART] %s: not drawn: %s", asset, e)
        return {}
