"""B12.1 (Desire 29 Sep, decision 45): the white proof card -- one plain picture per market, drawn from the real
candles: the line it broke (R2) and its wick zone, the R1 kill line, break / retest / entry markers, stop and
target, both Livermore brains as colour bands, a legend and the proof card. Sent with every trade and by /chart,
and shown on the dashboard. Display only -- nothing here feeds a trading decision."""
import os
import json
import importlib.util
import logging

import pandas as pd

logger = logging.getLogger(__name__)
BRAIN_COL = {"MAIN_UP": "#2e7d32", "NATURAL_RETRACEMENT": "#c0ca33", "SECONDARY_RETRACEMENT": "#ef6c00",
             "MAIN_DOWN": "#c62828", "NATURAL_REBOUND": "#4dd0e1", "SECONDARY_REBOUND": "#8e24aa"}
STYLE = {"B": "break", "E": "retest", "E2": "pause-then-turn"}
STAGE = {0: "waiting for a CLEAR 4H close past R2", 1: "broken -- waiting for the retest", 2: "retested -- waiting for the trigger"}
_LSM = {}


def _lsm():
    """The same brain module and settings the bot runs (loaded once)."""
    if "m" not in _LSM:
        try:
            from src.execution import livermore_state_machine as m      # the module the bot already runs
        except Exception:
            import sys
            spec = importlib.util.spec_from_file_location("lsm_card", os.path.join("src", "execution", "livermore_state_machine.py"))
            m = importlib.util.module_from_spec(spec)
            sys.modules["lsm_card"] = m                                  # dataclasses need the module registered
            spec.loader.exec_module(m)
        try:
            piv = json.load(open(os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig")).get("LIVERMORE_PIVOTS", {})
        except Exception:
            piv = {}
        _LSM["m"], _LSM["piv"] = m, piv
    return _LSM["m"], _LSM["piv"]


def _frame(df1):
    """Hourly candles indexed by OPEN time (tz-naive UTC), columns open/high/low/close."""
    h = df1.copy()
    if "timestamp" in h.columns:
        h.index = pd.to_datetime(h["timestamp"])
    h.index = pd.DatetimeIndex(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_convert("UTC").tz_localize(None)
    return h[["open", "high", "low", "close"]].astype(float).sort_index()


def _brain_bands(asset, h1, start, now):
    """Each brain's state per candle, replayed from 30 days of candles before the window (display only)."""
    m, piv = _lsm()
    out = {}
    src1 = h1[(h1.index >= start - pd.Timedelta(days=30)) & (h1.index + pd.Timedelta(hours=1) <= now)]
    src4 = src1.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    src4 = src4[src4.index + pd.Timedelta(hours=4) <= now]
    for hours, df in ((1, src1), (4, src4)):
        mach = m.make_livermore_pair(asset, piv.get(asset, {}))[0 if hours == 4 else 1]
        atr = m.atr14(df)
        seq = []
        for ts, c, a in zip(df.index, df["close"].values, atr.values):
            try:
                seq.append((ts, mach.update(float(c), float(a)).state))
            except Exception:
                seq.append((ts, seq[-1][1] if seq else None))
        out[hours] = [(ts, s) for ts, s in seq if ts + pd.Timedelta(hours=hours) > start]
    return out


def pick(cs, now=None, fresh_days=7):
    """What to draw: the newest proof (if under fresh_days old), else the most advanced live setup, else nothing."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow().tz_localize(None)
    hist = (cs or {}).get("ns_proofs_hist") or []
    if hist:
        p = hist[-1]
        try:
            if now - pd.Timestamp(p["fields"]["ns_candle"]) <= pd.Timedelta(days=fresh_days):
                return "proof", p
        except Exception:
            pass
    setups = [s for s in ((cs or {}).get("ns_setups") or []) if s.get("r2") is not None]
    if setups:
        return "setup", sorted(setups, key=lambda s: (-int(s.get("stage", 0)), str(s.get("conf"))))[0]
    return "none", None


def render(asset, df1, what, obj, out_png, now=None, margin_atr=None, risk_text=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    h1 = _frame(df1)
    now = pd.Timestamp(now) if now is not None else h1.index[-1] + pd.Timedelta(hours=1)
    T = lambda v: None if v in (None, "None", "NaT", "") else pd.Timestamp(v)
    ctr = lambda t_close: t_close - pd.Timedelta(minutes=30)          # engine times are candle CLOSES
    f = obj.get("fields", {}) if what == "proof" else {}
    if what == "proof":
        d = int(f.get("setup_dir", obj.get("dir", 1)))
        conf, r2, edge, r1 = T(f.get("ns_conf")), float(f["ns_r2"]), float(f.get("ns_edge") or f["ns_r2"]), float(f["ns_r1"])
        tb, tt, te = T(f.get("ns_b_t")), T(f.get("ns_touch_t")), T(f.get("ns_candle"))
    elif what == "setup":
        d = int(obj["d"])
        conf, r2, edge, r1 = T(obj.get("conf")), float(obj["r2"]), float(obj.get("edge") or obj["r2"]), float(obj.get("r1") or obj["r2"])
        tb, tt, te = T(obj.get("t_break")), T(obj.get("t_touch")), None
    else:
        d, conf, r2, edge, r1, tb, tt, te = 1, None, None, None, None, None, None, None
    anchors = [x for x in (conf, tb, te) if x is not None]
    start = (min(anchors) - pd.Timedelta(hours=40)) if anchors else now - pd.Timedelta(days=4)
    start = max(start, now - pd.Timedelta(days=10))
    h = h1[(h1.index + pd.Timedelta(hours=1) > start) & (h1.index + pd.Timedelta(hours=1) <= now)]
    fig = plt.figure(figsize=(18.5, 10.5), facecolor="white")
    ax = fig.add_axes([0.05, 0.25, 0.55, 0.63])
    b1 = fig.add_axes([0.05, 0.165, 0.55, 0.035])
    b4 = fig.add_axes([0.05, 0.12, 0.55, 0.035])
    w = 40.0 / 1440.0
    for ts, r in h.iterrows():
        xm = mdates.date2num((ts + pd.Timedelta(minutes=30)).to_pydatetime())
        col = "#2e7d32" if r["close"] >= r["open"] else "#c62828"
        ax.plot([xm, xm], [r["low"], r["high"]], color=col, lw=0.9, zorder=2)
        ax.add_patch(Rectangle((xm - w / 2, min(r["open"], r["close"])), w, max(abs(r["close"] - r["open"]), 1e-12),
                               fc=col, ec=col, zorder=3))
    ax.xaxis_date()
    x_end = now + pd.Timedelta(hours=2)
    if r2 is not None:
        x0 = max(conf, start) if conf is not None else start
        ax.fill_between([x0, x_end], min(r2, edge), max(r2, edge), color="#9ec5e8", alpha=0.45, zorder=1)
        ax.plot([x0, x_end], [r2, r2], color="#1565c0", lw=2.2, zorder=4)
        ax.text(x0, r2, " R2 (close)", color="#1565c0", fontsize=11, fontweight="bold", va="bottom" if d == 1 else "top")
        ax.plot([x0, x_end], [r1, r1], color="#c62828", lw=1.6, ls="--", zorder=4)
        ax.text(x0, r1, " R1 kill line", color="#c62828", fontsize=10, va="top" if d == 1 else "bottom")
    e = stop = tgt = None
    if what == "proof":
        e, stop = float(f.get("ns_close")), float(f.get("ns_stop"))
        tgt = f.get("ns_target")
        tgt = None if tgt in (None, "None") else float(tgt)
        xs = ctr(te)
        ax.plot([xs, x_end], [stop, stop], color="#e57373", lw=1.8, ls="--")
        ax.text(x_end, stop, " S stop", color="#c62828", fontsize=10, va="center")
        if tgt is not None:
            ax.plot([xs, x_end], [tgt, tgt], color="#81c784", lw=1.8, ls="--")
            ax.text(x_end, tgt, " T target", color="#2e7d32", fontsize=10, va="center")
        marks = [(tb, f.get("ns_b_px"), "B", "#1565c0"), (tt, f.get("ns_touch_px"), "R", "#6a1b9a"), (te, e, "E", "black")]
        if tb is not None and te is not None and tb == te:
            marks = [(te, e, "B/E", "black"), (tt, f.get("ns_touch_px"), "R", "#6a1b9a")]
        for tm, px, lab, col in marks:
            if tm is not None and px not in (None, "None"):
                ax.plot([ctr(tm)], [float(px)], "o", ms=24 if "/" in lab else 20, mfc="white", mec=col, mew=2.2, zorder=6)
                ax.text(ctr(tm), float(px), lab, color=col, fontsize=11, fontweight="bold", ha="center", va="center", zorder=7)
    ax.set_xlim(start, now + pd.Timedelta(hours=10))
    ax.grid(alpha=0.25)
    ax.tick_params(labelbottom=False)
    side = "LONG" if d == 1 else "SHORT"
    if what == "proof":
        title = "%s %s  -  %s entry  -  %s   (entry candle closed %s UTC)" % (
            asset, side, STYLE.get(f.get("ns_entry"), f.get("ns_entry")), f.get("ns_kind_raw", ""), str(te)[:16])
    elif what == "setup":
        title = "%s  -  watching setup #%s: %s %s  -  %s" % (asset, obj.get("id"), side, obj.get("kind", ""),
                                                             STAGE.get(int(obj.get("stage", 0)), ""))
    else:
        title = "%s  -  no live setup" % asset
    fig.suptitle(title, x=0.05, y=0.955, ha="left", fontsize=15)
    try:
        bands = _brain_bands(asset, h1, start, now)
    except Exception as _be:
        bands = {}
        logger.debug("[NS-CARD] %s: brain bands skipped: %s", asset, _be)
    for bx, hours, lab in ((b1, 1, "1H brain"), (b4, 4, "4H brain")):
        bx.set_yticks([])
        bx.xaxis_date()
        bx.set_xlim(ax.get_xlim())
        bx.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10, fontweight="bold")
        seq = bands.get(hours, [])
        for (t0, s0), nxt in zip(seq, [x[0] for x in seq[1:]] + [now]):
            bx.axvspan(max(t0, start), nxt, color=BRAIN_COL.get(s0, "white"), lw=0)
        bx.tick_params(labelbottom=(hours == 4))
    lines = []
    if what == "proof":
        rr = abs((tgt if tgt is not None else e) - e) / max(abs(e - stop), 1e-12)
        bpx, ba4 = f.get("ns_b_px"), f.get("ns_b_atr4")
        lines = ["PROOF CARD - %s %s  -  %s entry  -  %s" % (asset, side, f.get("ns_entry"), str(f.get("ns_kind_raw", "")).upper()), "",
                 "R2  %.6g = 4H swing CLOSE, zone to wick %.6g" % (r2, edge), "    (setup born %s UTC)" % str(conf)[:16],
                 "R1  %.6g = kill line" % r1, ""]
        if tb is not None and bpx not in (None, "None"):
            lines.append("B  break   %s  4H close %.6g" % (str(tb)[:16], float(bpx)))
            if ba4 not in (None, "None") and float(ba4) > 0:
                lines.append("           %.2f of a 4H move past R2 -- %s" % (d * (float(bpx) - r2) / float(ba4),
                             f.get("ns_break_kind") or ("big" if d * (float(bpx) - r2) >= 0.25 * float(ba4) else "small")))
            lines.append("           %s the wick" % ("CLEARED" if d * (float(bpx) - edge) > 0 else "inside"))
        lines.append(("R  retest  %s  %.6g" % (str(tt)[:16], float(f["ns_touch_px"]))) if tt is not None and f.get("ns_touch_px") not in (None, "None")
                     else "R  retest  - (break entry: no retest needed)" if f.get("ns_entry") == "B" else "R  retest  -")
        lines += ["E  entry   %s  close %.6g" % (str(te)[:16], e), "",
                  "S  stop    %.6g" % stop,
                  "T  target  %s   reward:risk %.2f" % (("%.6g" % tgt) if tgt is not None else "runner (trailed)", rr), "",
                  risk_text or "REAL RISK  (not known yet -- drawn before the order)", "",
                  "labels: A+ %s" % (", ".join(f.get("ns_aplus") or []) or "-"),
                  "        watch %s" % (", ".join(f.get("ns_watch") or []) or "-"),
                  "brains at entry: 1H %s | 4H %s" % (f.get("ns_brain_1h") or "-", f.get("ns_brain_4h") or "-"),
                  "checks: %s" % (f.get("ns_checks") or "-")]
    elif what == "setup":
        lines = ["SETUP #%s - %s %s" % (obj.get("id"), side, str(obj.get("kind", "")).upper()), "",
                 "R2  %.6g = the line to break (4H swing CLOSE)" % r2, "    zone to wick %.6g" % edge,
                 "R1  %.6g = kill line" % r1, "    born %s UTC" % str(conf)[:16], "",
                 "stage %d/3: %s" % (int(obj.get("stage", 0)) + 1, STAGE.get(int(obj.get("stage", 0)), "")),
                 "next: %s" % (obj.get("next_stage") or "-"),
                 "distance: %s 4H moves" % (obj.get("dist_atr") if obj.get("dist_atr") is not None else "-")]
    if lines:
        import textwrap
        lines = [w for l in lines for w in (textwrap.wrap(l, 64, subsequent_indent="           ") or [""])]
        fig.text(0.625, 0.88, "\n".join(lines), family="monospace", fontsize=10.5, va="top",
                 bbox=dict(boxstyle="round,pad=0.8", fc="#fafafa", ec="#999999"))
    leg = [Rectangle((0, 0), 1, 1, fc="#9ec5e8"), Line2D([], [], color="#1565c0", lw=2.2), Line2D([], [], color="#c62828", ls="--"),
           Line2D([], [], color="#e57373", ls="--"), Line2D([], [], color="#81c784", ls="--"),
           Line2D([], [], marker="o", color="w", mec="black", ms=12)] + [Rectangle((0, 0), 1, 1, fc=c) for c in BRAIN_COL.values()]
    fig.legend(leg, ["R2 zone (close to wick)", "R2 - the line to break", "R1 kill line", "S stop", "T target", "B break  R retest  E entry"]
               + [k.lower().replace("_", " ") for k in BRAIN_COL], loc="lower left", bbox_to_anchor=(0.05, 0.005), ncol=6,
               frameon=False, fontsize=10)
    fig.text(0.61, 0.135, "brain bands: replayed from\nthe candles shown (display only)", fontsize=8, color="#777777", va="center")
    tmp = out_png + ".tmp.png"
    fig.savefig(tmp, dpi=90, facecolor="white")
    plt.close(fig)
    os.replace(tmp, out_png)
    return out_png


def write_card(asset, df1, cs, out_dir=os.path.join("logs", "charts"), now=None, margin_atr=None, risk_text=None):
    """Draw this market's card to logs/charts/<ASSET>.png; returns the path (None if it could not be drawn)."""
    try:
        if df1 is None or len(df1) < 30:
            return None
        os.makedirs(out_dir, exist_ok=True)
        what, obj = pick(cs, now=now)
        return render(str(asset).upper(), df1, what, obj, os.path.join(out_dir, "%s.png" % str(asset).upper()),
                      now=now, margin_atr=margin_atr, risk_text=risk_text)
    except Exception as e:
        logger.warning("[NS-CARD] %s: card not drawn: %s", asset, e)
        return None
