"""B13 item 9A (Desire 1, 5 and 6 Oct 2026): THE COMBINED WHITE CHART -- one picture per market, four panels
(1H, 4H, 1D, 1W), every layer in one place:
  - candles; EMAs 20/50/200 on 1H and 4H, 50/200 on 1D and 1W
  - 4H swing zones (from the swing close to its wick) -- the lines the bot trades
  - wick lines: old 4H highs/lows measured at the wick tip -- DASHED, their own colour (B13 item 33's "old high in front")
  - both Livermore brains' lines (1H and 4H): main up-leg high, main down-leg low, natural high, natural low
  - live setups (R2 = the line to break, R1 = the kill line) and the latest proof's B / R / E markers
  - a short card: the latest proof or the most advanced setup
1D and 1W carry no Livermore state (the R2 calibration decides that). Display only -- nothing here is read by trading
code. Redrawn only when a candle, a proof or a setup changed. Output: logs/charts/<ASSET>_combined.png

Dashboard item (Oct 2026): write_combined also writes one chart per timeframe (logs/charts/<ASSET>_<TF>.png,
TF in 1H/4H/1D/1W) via write_singles, so the dashboard's Charts tab can toggle between timeframes and load
just the one requested instead of the whole four-panel picture. Same data, same redraw-on-change rule, no
second MT5 fetch -- write_combined hands its own D1/W1 frames straight to write_singles.

B14 section 8 (Desire 7 and 9 Oct): the chart draws THE LINE MAP (line_map.py) -- the very lines the bot reads:
  - diagonals: solid black while awake, dashed while asleep, solid dark red once broken; big lines thicker
  - the map's averages (4H 50/200, daily and weekly 50/200), its channels, and the spheres of the big lines near price
  - every panel is drawn candle after candle, so weekends and nightly breaks leave no gaps and the diagonals run on
    the same 4H candles the bot measures them on (the B13.1 change held back on 7 Oct, redone on this version)
  - the card shows the route, the witness, the lock and the staircase of a B14 trade
Without a map (MT5 down at the look) the chart is drawn without the map layers and says so on the card."""
import os
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
K = 2
COL = {"zone": "#9ec5e8", "r2": "#1565c0", "r1": "#c62828", "wick": "#ef6c00", "diag": "#000000", "broken": "#7f0000",
       "main": "#2e7d32", "natural": "#f9a825", "brain1h": "#e57373",
       "ema20": "#26a69a", "ema50": "#5c6bc0", "ema200": "#8d6e63", "up": "#2e7d32", "down": "#c62828",
       "sphere": "#b39ddb", "channel": "#cfd8dc", "mapma": "#455a64"}
_MEMO = {}
_MEMO_SINGLE = {}
_TFS = ("1H", "4H", "1D", "1W")
WINDOW_DAYS = {"1H": 5, "4H": 40, "1D": 270, "1W": 1100}
BROKEN_SHOW = 60          # a broken diagonal is drawn for 60 4H candles (10 trading days) after its break
MAX_DIAGS = 6             # per direction: the awake/asleep diagonals nearest to price ...
MAX_BROKEN = 3            # ... and the latest broken ones


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


class _Panel:
    """B14 8.2 (the 7 Oct change): one panel drawn candle after candle -- candle k sits at x = k and fills
    [k - 0.5, k + 0.5]; weekends and nightly breaks take no space."""
    def __init__(self, ax, df, span):
        self.ax, self.df, self.span = ax, df, span
        self.t0 = df.index.values
        self.n = len(df)
        self.right = self.n - 1 + max(6.0, 0.08 * self.n)        # room on the right for labels and extended lines

    def x(self, t):
        """A moment in time -> x. An open time -> the candle's left edge; a close time -> its right edge."""
        tt = np.datetime64(pd.Timestamp(t))
        k = int(np.searchsorted(self.t0, tt, side="right")) - 1
        if k < 0:
            return -0.5
        f = (tt - self.t0[k]) / np.timedelta64(int(self.span.total_seconds()), "s")
        return k - 0.5 + float(min(max(f, 0.0), 1.0))

    def candles(self):
        from matplotlib.patches import Rectangle
        ax = self.ax
        for k, (o, h, l, c) in enumerate(self.df[["open", "high", "low", "close"]].values):
            col = COL["up"] if c >= o else COL["down"]
            ax.plot([k, k], [l, h], color=col, lw=0.7, zorder=2)
            ax.add_patch(Rectangle((k - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-12), fc=col, ec=col, zorder=3))

    def ticks(self, fmt, period):
        """Date labels where the day / month / quarter changes; no two labels closer than a label's own width."""
        idx = self.df.index
        keys = idx.to_period(period).astype(str)
        pos = [k for k in range(1, self.n) if keys[k] != keys[k - 1]] or [0]
        if len(pos) >= 3:
            med = float(np.median(np.diff(pos)))
            pos = [p for j, p in enumerate(pos) if j == len(pos) - 1 or pos[j + 1] - p >= max(2.0, med / 3.0)]
        fig = self.ax.figure
        ax_in = self.ax.get_position().width * fig.get_figwidth()
        lab_in = len(idx[0].strftime(fmt)) * 0.62 * 8 / 72.0 + 0.25
        gap = lab_in / max(ax_in / max(self.right + 0.5, 1.0), 1e-9)
        kept = []
        for p in pos:
            if not kept or p - kept[-1] >= gap:
                kept.append(p)
        step = max(1, int(np.ceil(len(kept) / 7.0)))
        kept = kept[::step]
        self.ax.set_xticks(kept)
        self.ax.set_xticklabels([idx[k].strftime(fmt) for k in kept], fontsize=8)


def _get_map(asset):
    try:
        from src.execution import line_map as LM
        return LM.latest(asset)
    except Exception:
        return None


def _prepare(asset, df1, cs, now=None):
    """Everything computed once per asset per redraw -- candles, swings, the line map, brain lines, proofs/setups.
    Shared by the combined 4-panel chart (render) and every single-timeframe chart (render_single)."""
    mp = _get_map(asset)
    h1 = _frame(df1)
    if mp is not None and len(mp.df1) and mp.df1.index[-1] >= h1.index[-1]:
        h1 = mp.df1                                  # the map's own year of candles: the lines sit on them exactly
        h4 = mp.d4
    else:
        mp = None if mp is None or not len(mp.df1) or mp.df1.index[-1] < h1.index[-1] else mp
        h4 = _resample(h1, "4h")
        h4 = h4[h4.index + pd.Timedelta(hours=4) <= h1.index[-1] + pd.Timedelta(hours=1)]
    now = pd.Timestamp(now) if now is not None else h1.index[-1] + pd.Timedelta(hours=1)
    hi4, lo4, c4 = h4["high"].values, h4["low"].values, h4["close"].values
    a4 = _atr14(hi4, lo4, c4)
    t4c = (h4.index + pd.Timedelta(hours=4)).to_numpy()
    sw, sww = _swings(t4c, hi4, lo4, c4)
    brains = _brain_lines(asset, h1[h1.index >= now - pd.Timedelta(days=60)], h4[h4.index >= now - pd.Timedelta(days=60)])
    last = float(h1["close"].iloc[-1])
    a4n = float(a4[-1]) if len(a4) and a4[-1] == a4[-1] else abs(last) * 0.01
    band = (last - 4 * a4n, last + 4 * a4n)
    near = lambda v: v is not None and v == v and band[0] <= float(v) <= band[1]
    cs = cs or {}
    proofs = cs.get("ns_proofs_hist") or []
    setups = [s for s in (cs.get("ns_setups") or []) if s.get("r2") is not None]
    ctx = dict(h1=h1, h4=h4, now=now, sw=sw, sww=sww, brains=brains, last=last, band=band, near=near,
               proofs=proofs, setups=setups, mp=mp, a4n=a4n, diags=[], mas=[], channels=[], spheres=[])
    if mp is not None:
        try:
            ctx.update(_map_layers(mp, band, last))
        except Exception as e:
            logger.warning("[COMBINED-CHART] %s: the map layers were not drawn: %s", asset, e)
            ctx["mp"] = None
    return ctx


def _map_layers(mp, band, last):
    """B14 8.1: the map's layers for the chart -- the diagonals near price (with their awake / asleep / broken
    state per candle), the averages, the channels and the spheres of the big lines near price."""
    from src.execution import line_map as LM
    b = mp.n - 1
    lo, hi = band
    b_from = max(0, b - 6 * WINDOW_DAYS["4H"] - 10)
    keep = {1: [], -1: []}
    brk = {1: [], -1: []}
    for dg in mp.chart_diagonals(b_from, b):
        segs = []
        for stt, xs, ys in dg["segs"]:
            if stt == "broken" and dg["broken_at"] is not None:
                pts = [(x, y) for x, y in zip(xs, ys) if x <= dg["broken_at"] + BROKEN_SHOW]
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            segs.append((stt, xs, ys))
        if not segs or not any(lo <= y <= hi for _s, _x, ys in segs for y in ys):
            continue
        dg = dict(dg, segs=segs)
        lvl_now = dg["v1"] + dg["slope"] * (b + 1 - dg["i1"])
        if dg["broken_at"] is None:
            keep[dg["d"]].append((abs(lvl_now - last), dg))
        else:
            brk[dg["d"]].append((-dg["broken_at"], dg))
    diags = []
    for d in (1, -1):
        diags += [x[1] for x in sorted(keep[d], key=lambda z: z[0])[:MAX_DIAGS]]
        diags += [x[1] for x in sorted(brk[d], key=lambda z: z[0])[:MAX_BROKEN]]
    m = mp.members(b)
    mas = [(float(v), LM.NAME[int(c)], bool(c in LM.BIG_MA)) for v, c in zip(m["lv"], m["code"])
           if LM.MA4_50 <= int(c) <= LM.MAW_200 and lo <= v <= hi]
    channels = [dict(ch) for ch in mp.channels(b)]
    spheres = []
    for i in range(len(m["lv"])):
        cd = int(m["code"][i])
        if (m["big"][i] or cd in LM.BIG_MA) and lo <= m["lv"][i] <= hi and cd not in (LM.CHT, LM.CHB):
            spheres.append((float(m["lv"][i] - m["sph"][i]), float(m["lv"][i] + m["sph"][i]), LM.NAME[cd]))
    return dict(diags=diags, mas=mas, channels=channels, spheres=spheres, b4=b)


def _draw_map_diagonals(P, ctx, kmap):
    """B14 8.1: each diagonal along the 4H candles it is measured on -- solid black awake, dashed black asleep, solid
    dark red once broken; big lines thicker. kmap = each panel candle's position on the map's 4H clock."""
    per = (kmap[-1] - kmap[0]) / max(len(kmap) - 1, 1) if len(kmap) > 1 else 1.0
    lo, hi = ctx["band"]
    for dg in ctx["diags"]:
        v = lambda k, _dg=dg: _dg["v1"] + _dg["slope"] * (k - _dg["i1"])
        lw = 2.2 if dg["big"] else 1.2
        for j, (stt, xs, _ys) in enumerate(dg["segs"]):
            ka, kb = xs[0], xs[-1]
            pts = [(x, v(kmap[x])) for x in range(len(kmap)) if ka <= kmap[x] <= kb]
            last_seg = j == len(dg["segs"]) - 1
            if last_seg and dg["open_end"] and kb >= ctx.get("b4", kb):     # still on the map: carry it to the right
                pts.append((P.right, v(float(kmap[-1]) + (P.right - (len(kmap) - 1)) * per)))
            if len(pts) < 2 or not any(lo <= y <= hi for _x, y in pts):
                continue
            col = COL["broken"] if stt == "broken" else COL["diag"]
            ls = (0, (5, 4)) if stt == "asleep" else "-"
            P.ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=lw, ls=ls, zorder=5)


def _kmap(tf, df, h4):
    """Each panel candle's position on the 4H clock the diagonals are measured on."""
    n4 = len(h4)
    if tf == "4H":
        return np.arange(len(df), dtype=float) + (n4 - len(df))
    # an hour sits where its CLOSE falls: the hour that closes 4H candle k sits at exactly k (the level the engine
    # checks that 4H close against); hours of a 4H candle still forming count as candle n4
    op = pd.DatetimeIndex(df.index)
    bo = op.floor("4h")
    kk = pd.DatetimeIndex(h4.index).get_indexer(bo)
    kk = np.where(kk < 0, n4, kk)
    return kk + np.asarray(((op + pd.Timedelta(hours=1)) - (bo + pd.Timedelta(hours=4))) / pd.Timedelta(hours=4),
                           dtype=float)


def _draw_panel(ax, tf, d1, w1, ctx, asset):
    """Draw one timeframe panel (candles, EMAs, and -- for 1H/4H -- zones, the map, brain lines, setups, proof
    markers) onto `ax`. Returns False when there is no data in this timeframe's window."""
    h1, h4, now = ctx["h1"], ctx["h4"], ctx["now"]
    spec = {"1H": (h1, pd.Timedelta(hours=1), (20, 50, 200), ("%d %b", "D")),
            "4H": (h4, pd.Timedelta(hours=4), (20, 50, 200), ("%d %b", "D")),
            "1D": (d1, pd.Timedelta(days=1), (50, 200), ("%b %y", "M")),
            "1W": (w1, pd.Timedelta(days=7), (50, 200), ("%b %y", "Q"))}
    full, span, emas, (fmt, per) = spec[tf]
    df = full[full.index >= now - pd.Timedelta(days=WINDOW_DAYS[tf])]
    if len(df) == 0:
        return False
    P = _Panel(ax, df, span)
    P.candles()
    for n_ema in emas:
        e = full["close"].ewm(span=n_ema, adjust=False).mean().reindex(df.index)
        ax.plot(np.arange(P.n), e.values, color=COL["ema%d" % n_ema], lw=1.1, alpha=0.9, zorder=4)
    xr = P.right
    ax.set_xlim(-0.5, xr)
    lo_, hi_ = df["low"].min(), df["high"].max()
    if tf in ("1H", "4H"):
        near = ctx["near"]
        t_from = np.datetime64(now - pd.Timedelta(days=60))
        for cf, ty, lv, ed in ctx["sw"]:                                 # the lines the bot trades: zone close -> wick
            if cf >= t_from and near(lv):
                ax.fill_between([max(P.x(cf), -0.5), xr], min(lv, ed), max(lv, ed), color=COL["zone"], alpha=0.35, lw=0, zorder=1)
        for cf, ty, lv in ctx["sww"]:                                    # old highs/lows by wick -- DASHED
            if cf >= t_from and near(lv):
                ax.plot([max(P.x(cf), -0.5), xr], [lv, lv], color=COL["wick"], lw=1.0, ls=(0, (6, 4)), zorder=4)
        if ctx["mp"] is not None:                                        # B14 8.1: the line map
            _draw_map_diagonals(P, ctx, _kmap(tf, df, h4))
            x_m = P.n - 0.5 + 0.35 * (xr - P.n)
            for a_, b_, nm in ctx["spheres"]:                            # spheres of the big lines near price
                ax.fill_between([P.n - 0.5, xr], a_, b_, color=COL["sphere"], alpha=0.18, lw=0, zorder=1)
            for ch in ctx["channels"]:
                ax.fill_between([P.n - 0.5, xr], ch["bottom"], ch["top"], color=COL["channel"], alpha=0.25, lw=0, zorder=0)
                ax.text(xr, ch["top"], " %s top" % ch["kind"], color="#546e7a", fontsize=7, va="bottom", clip_on=True)
                ax.text(xr, ch["bottom"], " %s bottom" % ch["kind"], color="#546e7a", fontsize=7, va="top", clip_on=True)
            for v_, nm, big in ctx["mas"]:
                ax.plot([x_m, xr], [v_, v_], color=COL["mapma"], lw=2.0 if big else 1.0, zorder=6)
                ax.text(xr, v_, " %s" % nm, color=COL["mapma"], fontsize=7, va="center", clip_on=True)
        for blab, b in ctx["brains"].items():                            # Livermore: context only
            for k, v in b.items():
                if k == "state" or v is None or not near(v):
                    continue
                col = COL["brain1h"] if blab == "1H" else (COL["main"] if k.startswith("MAIN") else COL["natural"])
                ax.plot([-0.5, xr], [v, v], color=col, lw=1.3, ls=":" if blab == "1H" else "-", zorder=5)
                ax.text(xr, v, " %s (%s)" % (k, blab), color=col, fontsize=8, va="center")
        for s in ctx["setups"][:3]:
            r2, r1 = float(s["r2"]), float(s.get("r1") or s["r2"])
            ax.plot([-0.5, xr], [r2, r2], color=COL["r2"], lw=2.0, zorder=6)
            ax.plot([-0.5, xr], [r1, r1], color=COL["r1"], lw=1.2, ls="--", zorder=6)
        if ctx["proofs"]:
            f = ctx["proofs"][-1].get("fields", {})
            half = pd.Timedelta(minutes=15 if f.get("ns_entry") == "PKG" else 30)
            for tk, pk, lab, col in (("ns_b_t", "ns_b_px", "B", COL["r2"]), ("ns_touch_t", "ns_touch_px", "R", "#6a1b9a"),
                                     ("ns_candle", "ns_close", "E", "black")):
                try:
                    tm, px = pd.Timestamp(f[tk]), float(f[pk])
                    if tm >= df.index[0]:
                        xm = P.x(tm - (half if lab == "E" else pd.Timedelta(minutes=30)))
                        ax.plot([xm], [px], "o", ms=13, mfc="white", mec=col, mew=2, zorder=8)
                        ax.text(xm, px, lab, color=col, fontsize=8, fontweight="bold", ha="center", va="center", zorder=9)
                except Exception:
                    pass
            for _lv, _lab in ((f.get("ns_lock_at"), " lock"),):          # B14 6.4: where the lock fires
                if _lv not in (None, "None"):
                    ax.plot([P.n - 0.5, xr], [float(_lv), float(_lv)], color="#00838f", lw=1.2, ls=(0, (2, 2)), zorder=7)
                    ax.text(xr, float(_lv), _lab, color="#00838f", fontsize=8, va="center", clip_on=True)
    pad = (hi_ - lo_) * 0.08
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.axhline(ctx["last"], color="#999999", lw=0.8, ls=":")
    ax.grid(alpha=0.25)
    P.ticks(fmt, per)
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
        lines += ["LATEST PROOF: %s %s%s" % ("BUY" if int(f.get("setup_dir", 1)) == 1 else "SELL", f.get("ns_entry", ""),
                                            (" " + f["ns_label"]) if f.get("ns_label") else ""),
                  "  line %s  entry %s" % (g6(f.get("ns_r2")), g6(f.get("ns_close"))),
                  "  stop %s  target %s" % (g6(f.get("ns_stop")), g6(f.get("ns_target"))),
                  "  %s, candle %s" % (f.get("ns_kind_raw", ""), str(f.get("ns_candle"))[:16])]
        if f.get("ns_b14"):
            # B14 8.3 (Desire 9 Oct): the route, the witness, the lock and the staircase
            import textwrap as _tw
            walls = f.get("ns_walls") or []
            lines += _tw.wrap("route: %s" % (f.get("ns_route") or "-"), 40, initial_indent="  ", subsequent_indent="    ")
            lines += _tw.wrap("witness: %s" % (f.get("ns_witness") or "none in the 12 h before"), 40,
                              initial_indent="  ", subsequent_indent="    ")
            lines += [
                      "  target: %s" % (f.get("ns_target_kind") or "normal"),
                      "  lock: stop to entry at %s" % g6(f.get("ns_lock_at")) if walls else "  lock: no big wall before the target"]
            for W in walls[:3]:
                lines.append("  stair: close past %s -> stop %s" % (g6(W.get("s_far")), g6(W.get("s_near"))))
        else:
            lines.append("  diagonal: %s" % ("agrees" if f.get("ns_diag_agrees") else "none in the 24 h before"))
    elif setups:
        s = sorted(setups, key=lambda x: -int(x.get("stage", 0)))[0]
        lines += ["WATCHING SETUP #%s: %s" % (s.get("id"), "BUY" if int(s.get("d", 1)) == 1 else "SELL"),
                  "  line %s  kill %s" % (g6(s.get("r2")), g6(s.get("r1"))), "  stage %s/3" % (int(s.get("stage", 0)) + 1)]
    else:
        lines += ["no live setup"]
    if ctx.get("mp") is None:
        lines += ["", "(no line map this look -- map layers not drawn)"]
    return lines


def _legend_handles():
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, fc=COL["zone"], alpha=0.5), Line2D([], [], color=COL["wick"], ls=(0, (6, 4))),
               Line2D([], [], color=COL["main"]), Line2D([], [], color=COL["natural"]), Line2D([], [], color=COL["brain1h"], ls=":"),
               Line2D([], [], color=COL["diag"], lw=2.2), Line2D([], [], color=COL["diag"], lw=1.2, ls=(0, (5, 4))),
               Line2D([], [], color=COL["broken"], lw=1.6), Rectangle((0, 0), 1, 1, fc=COL["sphere"], alpha=0.4),
               Rectangle((0, 0), 1, 1, fc=COL["channel"], alpha=0.5), Line2D([], [], color=COL["mapma"], lw=2),
               Line2D([], [], color=COL["r2"], lw=2), Line2D([], [], color=COL["r1"], ls="--"),
               Line2D([], [], color=COL["ema20"]), Line2D([], [], color=COL["ema50"]), Line2D([], [], color=COL["ema200"]),
               Line2D([], [], marker="o", color="w", mec="black", ms=10)]
    labels = ["4H swing zone (close to wick)", "old high/low by WICK (dashed)", "4H brain: main up/down (context)",
              "4H brain: natural high/low (context)", "1H brain lines (context)", "diagonal awake (thick = big)",
              "diagonal asleep", "diagonal broken", "sphere of a big line", "channel", "map average (D/W/4H)",
              "setup line (R2)", "kill line (R1)", "EMA 20", "EMA 50", "EMA 200", "B break  R retest  E entry"]
    return handles, labels


def _daily_weekly_frames(ctx, d1, w1):
    mp = ctx.get("mp")
    if d1 is None and mp is not None:
        d1 = mp.d1                                                           # the map's own daily candles
    if w1 is None and mp is not None:
        w1 = mp.w1
    d1f = _frame(d1) if d1 is not None else _resample(ctx["h1"], "1D")      # fallback only (MT5's own daily is used live)
    w1f = _frame(w1) if w1 is not None else ctx["h1"].resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return d1f, w1f


def render(asset, df1, cs, out_png, d1=None, w1=None, now=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = _prepare(asset, df1, cs, now=now)
    d1f, w1f = _daily_weekly_frames(ctx, d1, w1)

    fig = plt.figure(figsize=(20, 12.5), facecolor="white")
    axes = {"1H": fig.add_axes([0.04, 0.53, 0.47, 0.40]), "4H": fig.add_axes([0.54, 0.53, 0.30, 0.40]),
            "1D": fig.add_axes([0.04, 0.08, 0.38, 0.37]), "1W": fig.add_axes([0.46, 0.08, 0.38, 0.37])}
    for tf, ax in axes.items():
        _draw_panel(ax, tf, d1f, w1f, ctx, asset)

    fig.text(0.855, 0.93, "\n".join(_panel_lines(asset, ctx)), family="monospace", fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.7", fc="#fafafa", ec="#999999"))
    leg_handles, leg_labels = _legend_handles()
    fig.legend(leg_handles, leg_labels, loc="lower left", bbox_to_anchor=(0.86, 0.04), ncol=1, frameon=False, fontsize=8.5)

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

    ctx = _prepare(asset, df1, cs, now=now)
    d1f, w1f = _daily_weekly_frames(ctx, d1, w1)

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = fig.add_axes([0.055, 0.09, 0.66, 0.84])
    if not _draw_panel(ax, which, d1f, w1f, ctx, asset):
        plt.close(fig)
        return None

    fig.text(0.745, 0.90, "\n".join(_panel_lines(asset, ctx)), family="monospace", fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.7", fc="#fafafa", ec="#999999"))
    leg_handles, leg_labels = _legend_handles()
    fig.legend(leg_handles, leg_labels, loc="lower left", bbox_to_anchor=(0.745, 0.04), ncol=1, frameon=False, fontsize=8)

    tmp = out_png + ".tmp.png"
    fig.savefig(tmp, dpi=90, facecolor="white")
    plt.close(fig)
    os.replace(tmp, out_png)
    return out_png


def _chart_sig(df1, cs, asset):
    mp = _get_map(asset)
    return "|".join(str(x) for x in (df1.index[-1] if "timestamp" not in df1.columns else df1["timestamp"].iloc[-1],
                                     len(cs.get("ns_proofs_hist") or []),
                                     [(s.get("id"), s.get("stage")) for s in (cs.get("ns_setups") or [])],
                                     getattr(mp, "_engine_last", None) if mp is not None else None))


def write_combined(asset, df1, cs, symbol=None, out_dir=os.path.join("logs", "charts"), now=None):
    """Draw logs/charts/<ASSET>_combined.png when a candle, a proof, a setup or the map changed; returns the path or
    None. Also redraws each single-timeframe panel via write_singles, on the same change signature and the same
    D1/W1 fetch -- the dashboard's per-timeframe Charts toggle stays current without a second trigger."""
    try:
        if df1 is None or len(df1) < 60:
            return None
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "%s_combined.png" % str(asset).upper())
        if cs is not None and not isinstance(cs, dict):          # the bot passes a dict or a CompositeState object
            cs = {"ns_proofs_hist": getattr(cs, "ns_proofs_hist", None), "ns_setups": getattr(cs, "ns_setups", None)}
        cs = cs or {}
        sig = _chart_sig(df1, cs, str(asset).upper())
        if _MEMO.get(asset) == sig and os.path.exists(out_png):
            return out_png
        mp = _get_map(str(asset).upper())
        d1 = mp.d1 if mp is not None else (_mt5_frame(symbol, "D1", 400) if symbol else None)
        w1 = mp.w1 if mp is not None else (_mt5_frame(symbol, "W1", 260) if symbol else None)
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
        sig = _sig or _chart_sig(df1, cs, str(asset).upper())
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
