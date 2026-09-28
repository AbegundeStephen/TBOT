"""
B11-NS -- the live interactive chart, one HTML page per market.

The chart Desire approved on 25 Sep (the overlay charts: 4H structure + both Livermore
brains over the ladder), rebuilt from the bot's LIVE state:
  - 1H and 4H candles
  - the new two-layer ladder: 4H swing zones (close to wick) + both brains' levels
  - both brains over time: up-leg highs / down-leg lows, natural highs / lows, and a
    colour band per brain showing its state candle by candle
  - live setups (R2 + zone, R1, peak, stage) and recent proofs with full proof cards
    (break / peak / retest / entry with times and prices, freshness, stop, target, R:R,
    both brains at entry, and the automatic checks)
  - the brains' labels: state, age, STALE, and the levels that would flip them
  - legend toggles, drag / scroll zoom, and the price axis re-fits itself on zoom

Plotly is loaded from its CDN, so a page is small enough to send on Telegram.
DISPLAY ONLY: nothing in this module feeds any trading decision.
"""
import html
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
STATES = ["MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT", "MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND"]
STATE_COLORS = {"MAIN_UP": "#2e7d32", "NATURAL_RETRACEMENT": "#c0ca33", "SECONDARY_RETRACEMENT": "#ef6c00",
                "MAIN_DOWN": "#c62828", "NATURAL_REBOUND": "#4dd0e1", "SECONDARY_REBOUND": "#8e24aa"}
UP_STATES = ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
STALE_DAYS = {"1H": 5, "4H": 20}
DISPLAY_DAYS = 21
# Desire, 28 Sep (ruling B): moving averages and FVGs stay OFF the chart until tools/all_tests.py (Tests 1 and 1c)
# says they are promising. Flip to True only after that ruling.
SHOW_DYNAMIC_LEVELS = True    # B12 (Desire 28 Sep, ruling 1A): the open 1H gaps are shown (display only)...
SHOW_MAS = False              # ...moving averages stay OFF until a test says they are promising (Tests 1/1c: not)


def _nm(state):
    return str(state or "").replace("_", " ").lower()


def _open_times(df):
    raw = df["timestamp"] if "timestamp" in df.columns else df.index
    idx = pd.DatetimeIndex(pd.to_datetime(raw))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _fmt_t(t):
    try:
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(t)


def _num(v, nd=5):
    try:
        return ("%." + str(nd) + "g") % float(v)
    except (TypeError, ValueError):
        return "n/a"


def _replay_brains(asset, df, tf):
    """Replay a Livermore brain over the candles we have (read-only), for drawing only.
    Returns per-candle state and anchors, or None if the brain module is unavailable."""
    try:
        from src.execution import livermore_state_machine as lsm
        with open(os.path.join("config", "aggregator_presets.json"), encoding="utf-8-sig") as fh:
            pivots = (json.load(fh).get("LIVERMORE_PIVOTS") or {}).get(str(asset).upper(), {})
        m4, m1 = lsm.make_livermore_pair(asset, pivots)
        m = m4 if tf == "4H" else m1
        atr = lsm.atr14(df)
        out = {"state": [], "up": [], "down": [], "nlow": [], "nhigh": []}
        for c, a in zip(df["close"].values, atr.values):
            try:
                s = m.update(float(c), float(a))
            except Exception:
                s = None
            st = getattr(s, "state", None) if s is not None else None
            up = st in UP_STATES
            out["state"].append(st)
            out["up"].append(getattr(s, "anchor_main_up_max", None) if (s is not None and up) else None)
            out["down"].append(getattr(s, "anchor_main_down_min", None) if (s is not None and not up) else None)
            out["nlow"].append(getattr(s, "anchor_natural_low", None) if s is not None else None)
            out["nhigh"].append(getattr(s, "anchor_natural_high", None) if s is not None else None)
        return out
    except Exception as e:
        logger.warning(f"[NS-CHART] {asset}: {tf} brain replay unavailable ({e}) -- chart drawn without brain bands")
        return None


def _clean(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _brain_verdict(d, state, age_days, tf):
    if state is None:
        return "no reading"
    try:
        if float(age_days or 0) > STALE_DAYS[tf]:
            return "STALE"
    except (TypeError, ValueError):
        pass
    agrees = (state in UP_STATES) == (int(d) == 1)
    return "agrees" if agrees else "disagrees"


def proof_card_lines(p, asset=""):
    """The proof card, one line per fact (used in the hover box, the page and Telegram)."""
    f = p.get("fields", p) if isinstance(p, dict) else {}
    d = int(f.get("setup_dir") or f.get("brc_direction") or 1)
    side = "long" if d == 1 else "short"
    atr = _clean(f.get("ns_atr1"))
    e, r2 = _clean(f.get("ns_close")), _clean(f.get("ns_r2"))
    stop, tgt = _clean(f.get("ns_stop")), _clean(f.get("ns_target"))
    fresh = (d * (e - r2) / atr) if (e is not None and r2 is not None and atr) else None
    rr = (abs(tgt - e) / abs(e - stop)) if (tgt is not None and e is not None and stop is not None and e != stop) else None
    lines = [
        "PROOF %s %s %s -- %s%s" % (f.get("brc_tier", "?"), f.get("ns_kind_raw", f.get("setup_kind", "?")), side, asset,
                                    "  [found while catching up -- not traded]" if p.get("missed") else ""),
        "R2 %s (zone to %s)  |  R1 %s" % (_num(r2), _num(f.get("ns_edge")), _num(f.get("ns_r1"))),
        "Break  %s @ %s%s" % (_fmt_t(f.get("ns_b_t")), _num(f.get("ns_b_px")),
                              "  <- CHECK: closed inside the zone (wick not cleared)" if f.get("ns_gap2_break_inside_zone") else ""),
    ]
    if f.get("ns_entry") == "E":
        lines += ["Peak   %s (%s)" % (_num(f.get("brc_h2")), _fmt_t(f.get("ns_h2_t"))),
                  "Retest %s @ %s%s" % (_fmt_t(f.get("ns_touch_t")), _num(f.get("ns_touch_px")),
                                        "  <- CHECK: instant retest (the very next candle)" if f.get("ns_gap1_instant_retest") else "")]
    lines += [
        "Entry  %s @ %s  (%s)" % (_fmt_t(f.get("ns_candle")), _num(e), "at the 4H break" if f.get("ns_entry") == "B" else "close past the peak"),
        "Fresh  %s moves from R2 (limit 2.50)" % (("%.2f" % fresh) if fresh is not None else "n/a"),
        "Stop %s  |  Target %s  |  R:R %s" % (_num(stop), _num(tgt) if tgt is not None else "none (runner)",
                                             ("%.2f" % rr) if rr is not None else "n/a"),
        "Brains at entry: 4H %s (%s)  |  1H %s (%s)" % (
            _nm(f.get("ns_brain_4h")), _brain_verdict(d, f.get("ns_brain_4h"), f.get("ns_brain_4h_age_days"), "4H"),
            _nm(f.get("ns_brain_1h")), _brain_verdict(d, f.get("ns_brain_1h"), f.get("ns_brain_1h_age_days"), "1H")),
        "Outcome: see the trade record (not joined to the card yet)",
        "(times are candle closes, UTC)",
    ]
    return lines


def build_chart_html(asset, df1, df4, cs, days=DISPLAY_DAYS):
    """The whole page as a string. df1/df4 = CLOSED 1H/4H candles; cs = the composite state (dict)."""
    cs = cs or {}
    if hasattr(cs, "to_dict"):
        cs = cs.to_dict()
    asset = str(asset).upper()
    t1_all, t4_all = _open_times(df1), _open_times(df4)
    b1 = _replay_brains(asset, df1, "1H")
    b4 = _replay_brains(asset, df4, "4H")
    start = t1_all[-1] - pd.Timedelta(days=days)
    k1 = int(t1_all.searchsorted(start))
    k4 = int(t4_all.searchsorted(start))
    d1, d4 = df1.iloc[k1:], df4.iloc[k4:]
    t1, t4 = t1_all[k1:], t4_all[k4:]
    x1, x4 = [_fmt_t(t) for t in t1], [_fmt_t(t) for t in t4]
    x_end = _fmt_t(t1[-1] + pd.Timedelta(hours=1))
    price = float(d1["close"].values[-1])
    traces, ann = [], []

    def line_seg(name, pts, color, dash="solid", width=1.5, group=None, show=True, hover=None, visible=True, legend=True):
        xs, ys, hv = [], [], []
        for (xa, xb, y, h) in pts:
            xs += [xa, xb, None]
            ys += [y, y, None]
            hv += [h, h, None]
        tr = {"type": "scatter", "mode": "lines", "name": name, "x": xs, "y": ys,
              "line": {"color": color, "width": width, "dash": dash}, "hoverinfo": "text", "text": hv,
              "showlegend": bool(legend and show)}
        if group:
            tr["legendgroup"] = group
        if visible is not True:
            tr["visible"] = visible
        traces.append(tr)

    # candles
    traces.append({"type": "candlestick", "name": "1H candles", "x": x1, "open": d1["open"].round(6).tolist(),
                   "high": d1["high"].round(6).tolist(), "low": d1["low"].round(6).tolist(),
                   "close": d1["close"].round(6).tolist(), "increasing": {"line": {"color": "#26a69a"}},
                   "decreasing": {"line": {"color": "#ef5350"}}})
    traces.append({"type": "candlestick", "name": "4H candles", "x": x4, "open": d4["open"].round(6).tolist(),
                   "high": d4["high"].round(6).tolist(), "low": d4["low"].round(6).tolist(),
                   "close": d4["close"].round(6).tolist(), "visible": "legendonly",
                   "increasing": {"line": {"color": "#90a4ae"}}, "decreasing": {"line": {"color": "#78909c"}}})

    # layer 1: 4H swing zones (close to wick)
    zx, zy, zc = [], [], []
    for r in cs.get("ns_ladder") or []:
        if r.get("layer") != 1:
            continue
        xa = max(_fmt_t(r.get("since")), x1[0])
        a, b = float(r["close"]), float(r["edge"])
        zx += [xa, x_end, x_end, xa, xa, None]
        zy += [a, a, b, b, a, None]
        zc.append((xa, x_end, a, "4H swing %s: close %s / wick %s (since %s)" % (
            "high" if r.get("type") == "H" else "low", _num(a), _num(b), _fmt_t(r.get("since")))))
    if zx:
        traces.append({"type": "scatter", "mode": "lines", "name": "4H swing zones (close to wick)", "x": zx, "y": zy,
                       "fill": "toself", "fillcolor": "rgba(21,101,192,0.16)", "line": {"width": 0},
                       "hoverinfo": "skip", "legendgroup": "zones"})
        line_seg("swing close", zc, "#1565c0", width=1.2, group="zones", show=False, legend=False)

    # layer 2: both brains' levels (Option A: rungs of their own, never R2)
    for tf, col in (("1H", "#ef6c00"), ("4H", "#8e24aa")):
        pts = []
        for r in cs.get("ns_ladder") or []:
            if r.get("layer") == 2 and r.get("tf") == tf:
                xa = max(_fmt_t(r.get("since")), x1[0])
                pts.append((xa, x_end, float(r["close"]), "%s brain %s %s (since %s)" % (
                    tf, str(r.get("type", "")).replace("_", " "), _num(r["close"]), _fmt_t(r.get("since")))))
        if pts:
            line_seg("%s brain levels (ladder)" % tf, pts, col, dash="dash", width=1.4)

    # both brains over time (replayed for drawing)
    for tf, br, xs in (("4H", b4, x4), ("1H", b1, x1)):
        if not br:
            continue
        kk = k4 if tf == "4H" else k1
        for key, name, col, mode in (("up", "%s up-leg high (close)" % tf, "#2e7d32", "lines"),
                                     ("down", "%s down-leg low (close)" % tf, "#c62828", "lines"),
                                     ("nlow", "%s natural low" % tf, "#2e7d32", "markers"),
                                     ("nhigh", "%s natural high" % tf, "#c62828", "markers")):
            ys = [_clean(v) for v in br[key][kk:]]
            tr = {"type": "scatter", "mode": mode, "name": name, "x": xs, "y": ys, "connectgaps": False,
                  "hovertemplate": name + " %{y}<extra></extra>"}
            if mode == "lines":
                tr["line"] = {"color": col, "width": 2 if tf == "4H" else 1.2, "shape": "hv",
                              "dash": "solid" if tf == "4H" else "dot"}
            else:
                tr["marker"] = {"color": col, "size": 5 if tf == "4H" else 3, "symbol": "line-ew-open"}
            if tf == "1H":
                tr["visible"] = "legendonly" if key in ("nlow", "nhigh") else True
            traces.append(tr)
        # the state band under the chart
        yax = "y2" if tf == "4H" else "y3"
        sts = br["state"][kk:]
        for s in STATES:
            xx = [x for x, v in zip(xs, sts) if v == s]
            traces.append({"type": "bar", "name": _nm(s), "x": xx, "y": [1] * len(xx), "yaxis": yax,
                           "marker": {"color": STATE_COLORS[s]}, "legendgroup": s, "showlegend": tf == "4H",
                           "hovertemplate": "%s brain: %s<extra></extra>" % (tf, _nm(s))})

    if SHOW_DYNAMIC_LEVELS:     # B12: open 1H gaps (display only); moving averages only if SHOW_MAS
        _dynamic_layers(traces, df1, df4, k1, k4, x1, x4, x_end, t1_all, t4_all)

    # the old ladder (the council still reads it) -- hidden by default
    old = cs.get("zone_ladder_4h") or []
    pts2, pts01 = [], []
    for z in old if isinstance(old, list) else []:
        try:
            lvl = float(z.get("price", z.get("level")) if isinstance(z, dict) else z)
            tests = int((z.get("tests", 0) if isinstance(z, dict) else 0) or 0)
        except (TypeError, ValueError):
            continue
        (pts2 if tests >= 2 else pts01).append((x1[0], x_end, lvl, "old ladder %s (tested %d)" % (_num(lvl), tests)))
    if pts2:
        line_seg("bot's old ladder, tested 2+ (council)", pts2, "rgba(97,97,97,0.8)", width=1, visible="legendonly")
    if pts01:
        line_seg("bot's old ladder, tested 0-1", pts01, "rgba(158,158,158,0.6)", width=1, visible="legendonly")

    # live setups: the proofs FORMING, stage by stage (colour = stage; markers = the steps already done)
    stage_txt = ("waiting for the break", "broken -- waiting for the retest", "retested -- waiting for the trigger")
    stage_col = {0: "#78909c", 1: "#fb8c00", 2: "#43a047"}
    badges, zsx, zsy, shown = [], [], [], set()
    for s_ in cs.get("ns_setups") or []:
        try:
            d = int(s_["d"])
            st = min(2, max(0, int(s_.get("stage", 0))))
            xa = max(_fmt_t(s_.get("conf")), x1[0])
            col = stage_col[st]
            if st == 0:
                nxt = "a 4H close %s %s" % ("above" if d == 1 else "below", _num(s_["r2"]))
            elif st == 1:
                nxt = "a 1H %s within 1 move of %s (the zone's wick edge)" % ("low" if d == 1 else "high", _num(s_["edge"]))
            else:
                nxt = "a 1H close %s %s (the peak) -> PROOF" % ("above" if d == 1 else "below", _num(s_.get("h2")))
            dist = s_.get("dist_atr")
            label = "#%s %s %s -- STAGE %d/3: %s -- next: %s%s" % (
                s_.get("id"), "LONG" if d == 1 else "SHORT", s_.get("kind"), st + 1, stage_txt[st], nxt,
                (" (%s moves away)" % dist) if dist is not None else "")
            badges.append('<span style="color:%s">&#9632;</span> %s' % (col, html.escape(label)))
            traces.append({"type": "scatter", "mode": "lines", "name": "stage %d/3: %s" % (st + 1, stage_txt[st]),
                           "x": [xa, x_end], "y": [float(s_["r2"])] * 2, "line": {"color": col, "width": 3},
                           "legendgroup": "stage%d" % st, "showlegend": st not in shown, "hoverinfo": "text",
                           "text": [label, label]})
            shown.add(st)
            traces.append({"type": "scatter", "mode": "lines", "name": "R1 (kill line)", "x": [xa, x_end],
                           "y": [float(s_["r1"])] * 2, "line": {"color": "#757575", "width": 1.3, "dash": "dot"},
                           "legendgroup": "stage%d" % st, "showlegend": False, "hoverinfo": "text",
                           "text": ["R1 (kill line) of #%s: %s" % (s_.get("id"), _num(s_["r1"]))] * 2})
            zsx += [xa, x_end, x_end, xa, xa, None]
            zsy += [float(s_["r2"]), float(s_["r2"]), float(s_["edge"]), float(s_["edge"]), float(s_["r2"]), None]
            steps = []
            if s_.get("b_t") and s_.get("b_px") is not None:
                steps.append((s_["b_t"], s_["b_px"], "circle", "BREAK close %s" % _num(s_["b_px"])))
            if s_.get("h2_t") and s_.get("h2") is not None:
                steps.append((s_["h2_t"], s_["h2"], "diamond", "PEAK close %s" % _num(s_["h2"])))
            if s_.get("t_touch") and s_.get("touch_px") is not None:
                steps.append((s_["t_touch"], s_["touch_px"], "square", "RETEST %s%s" % (
                    _num(s_["touch_px"]), " (instant retest)" if s_.get("gap1") else "")))
            for (tt, px, sym, txt) in steps:
                traces.append({"type": "scatter", "mode": "markers", "name": "steps done", "showlegend": False,
                               "legendgroup": "stage%d" % st,
                               "x": [_fmt_t(pd.Timestamp(tt) - pd.Timedelta(hours=1))], "y": [float(px)],
                               "marker": {"symbol": sym, "size": 11, "color": col, "line": {"color": "black", "width": 1}},
                               "hoverinfo": "text", "text": ["#%s %s @ %s" % (s_.get("id"), txt, _fmt_t(tt))]})
            if st == 2 and s_.get("h2") is not None:
                xa2 = _fmt_t(pd.Timestamp(s_["t_touch"]) - pd.Timedelta(hours=1)) if s_.get("t_touch") else xa
                traces.append({"type": "scatter", "mode": "lines", "name": "trigger line", "showlegend": False,
                               "legendgroup": "stage2", "x": [max(xa2, x1[0]), x_end], "y": [float(s_["h2"])] * 2,
                               "line": {"color": "#43a047", "width": 1.5, "dash": "dash"}, "hoverinfo": "text",
                               "text": ["#%s trigger: a 1H close past %s makes the proof" % (s_.get("id"), _num(s_["h2"]))] * 2})
        except Exception:
            continue
    if zsx:
        traces.append({"type": "scatter", "mode": "lines", "name": "live setups: R2 zone", "x": zsx, "y": zsy,
                       "fill": "toself", "fillcolor": "rgba(0,176,255,0.12)", "line": {"width": 0},
                       "hoverinfo": "skip", "legendgroup": "setups"})
    r2p = [(None, None, None, b) for b in badges]
    if badges:
        ann.append({"x": 0.995, "y": 0.995, "xref": "paper", "yref": "paper", "xanchor": "right", "yanchor": "top",
                    "align": "left", "showarrow": False, "text": "<b>PROOFS FORMING</b><br>" + "<br>".join(badges[:8]),
                    "font": {"size": 11}, "bgcolor": "rgba(255,255,255,0.88)", "bordercolor": "#9e9e9e", "borderwidth": 1})

    # proofs (live + recent), with the full proof card on hover
    cards = []
    for p in (cs.get("ns_proofs_hist") or [])[-20:]:
        f = p.get("fields", {})
        try:
            d = int(f.get("setup_dir", 1))
            ex = _fmt_t(pd.Timestamp(f.get("ns_candle")) - pd.Timedelta(hours=1))   # engine times = candle CLOSE; plot on that candle
            lines = proof_card_lines(p, asset)
            cards.append(lines)
            traces.append({"type": "scatter", "mode": "markers", "name": "proof", "showlegend": False,
                           "legendgroup": "proofs", "x": [ex], "y": [float(f.get("ns_close"))],
                           "marker": {"symbol": "triangle-up" if d == 1 else "triangle-down", "size": 13,
                                      "color": "#9e9e9e" if p.get("missed") else ("#00c853" if d == 1 else "#d50000"),
                                      "line": {"color": "black", "width": 1}},
                           "hoverinfo": "text", "text": ["<br>".join(html.escape(x) for x in lines)]})
            if f.get("ns_stop") is not None:
                line_seg("proof stop", [(ex, x_end, float(f["ns_stop"]), "stop %s" % _num(f["ns_stop"]))],
                         "#d50000", dash="dot", width=1, group="proofs", legend=False)
            if f.get("ns_target"):
                line_seg("proof target", [(ex, x_end, float(f["ns_target"]), "target %s" % _num(f["ns_target"]))],
                         "#00c853", dash="dot", width=1, group="proofs", legend=False)
        except Exception:
            continue
    if cards:
        traces.append({"type": "scatter", "mode": "markers", "name": "proofs (hover for the card)", "x": [None],
                       "y": [None], "legendgroup": "proofs", "marker": {"symbol": "triangle-up", "color": "#00c853"}})

    # labels on the chart (prototype style) + the brains' tag box
    brains = cs.get("ns_brains") or {}
    tag_lines = []
    for tf in ("4H", "1H"):
        b = brains.get(tf)
        if not b:
            continue
        ma, mb = b.get("moves_above"), b.get("moves_below")
        tag_lines.append("<b>%s brain</b>: %s, %.1f days%s | flips above %s (%s) / below %s (%s)" % (
            tf, _nm(b.get("state")), float(b.get("age_days") or 0), " <b>(STALE)</b>" if b.get("stale") else "",
            _num(b.get("flips_above")), ("%.1f moves" % ma) if ma is not None else "n/a",
            _num(b.get("flips_below")), ("%.1f moves" % mb) if mb is not None else "n/a"))
        for key, lab in (("flips_above", "flips above"), ("flips_below", "flips below")):
            v = _clean(b.get(key))
            if v is not None:
                ann.append({"x": x_end, "y": v, "xref": "x", "yref": "y", "text": "%s brain %s %s" % (tf, lab, _num(v)),
                            "showarrow": False, "xanchor": "left", "font": {"size": 10,
                            "color": "#8e24aa" if tf == "4H" else "#ef6c00"}})
    if tag_lines:
        ann.append({"x": 0.005, "y": 0.995, "xref": "paper", "yref": "paper", "xanchor": "left", "yanchor": "top",
                    "align": "left", "showarrow": False, "text": "<br>".join(tag_lines), "font": {"size": 11},
                    "bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#9e9e9e", "borderwidth": 1})
    ann.append({"x": 0, "y": 0.145, "xref": "paper", "yref": "paper", "text": "<b>4H brain</b>", "showarrow": False,
                "xanchor": "left", "yanchor": "bottom", "font": {"size": 10}})
    ann.append({"x": 0, "y": 0.065, "xref": "paper", "yref": "paper", "text": "<b>1H brain</b>", "showarrow": False,
                "xanchor": "left", "yanchor": "bottom", "font": {"size": 10}})

    now_4h = brains.get("4H", {}).get("state")
    title = "<b>%s</b> - 4H structure + Livermore 4H and 1H brains + new ladder - now: 4H %s, 1H %s - live setups %d - price %s - updated %s UTC" % (
        asset, _nm(now_4h), _nm(brains.get("1H", {}).get("state")), len(cs.get("ns_setups") or []), _num(price),
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    layout = {
        "title": {"text": title, "font": {"size": 13}, "x": 0.01}, "autosize": True, "hovermode": "closest",
        "dragmode": "zoom", "barmode": "stack", "bargap": 0, "margin": {"l": 60, "r": 150, "t": 50, "b": 40},
        "legend": {"orientation": "h", "y": -0.06, "yanchor": "top", "x": 0, "font": {"size": 11},
                   "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        "xaxis": {"type": "date", "rangeslider": {"visible": False}, "anchor": "y3"},
        "yaxis": {"domain": [0.16, 1.0], "title": {"text": asset}, "side": "left"},
        "yaxis2": {"domain": [0.085, 0.145], "showticklabels": False, "range": [0, 1], "fixedrange": True},
        "yaxis3": {"domain": [0.005, 0.065], "showticklabels": False, "range": [0, 1], "fixedrange": True},
        "annotations": ann,
    }
    fit_ms = [int(pd.Timestamp(t).value // 10 ** 6) for t in t1]
    fit = {"l": d1["low"].round(6).tolist(), "h": d1["high"].round(6).tolist()}
    cards_html = "".join('<div class="card"><pre>%s</pre></div>' % html.escape("\n".join(c)) for c in reversed(cards))
    setups_html = "".join("<li>%s</li>" % t[3] for t in r2p) or "<li>none right now</li>"
    return PAGE.format(asset=html.escape(asset), cdn=PLOTLY_CDN, traces=json.dumps(traces, allow_nan=False, default=str),
                       layout=json.dumps(layout, default=str), fit_ms=json.dumps(fit_ms), fit=json.dumps(fit),
                       setups=setups_html, cards=cards_html or "<p>no proofs in the recent history</p>")


def _dynamic_layers(traces, df1, df4, k1, k4, x1, x4, x_end, t1_all, t4_all):
    """Display only. B12 (Desire 28 Sep): the open 1H fair-value gaps are drawn -- price turns inside them 1.5x more
    often than chance, in both halves (Test 1) -- but bouncing off them loses after costs (Test 1c). Moving averages
    and 4H gaps stay off until a test says they are promising (Tests 1/1c, 28 Sep: they are not)."""
    if SHOW_MAS:
        for p_, vis, col in ((20, "legendonly", "#26c6da"), (50, True, "#ab47bc"), (200, True, "#6d4c41")):
            ev = df1["close"].ewm(span=p_, adjust=False).mean().values[k1:]
            traces.append({"type": "scatter", "mode": "lines", "name": "EMA %d (1H)" % p_, "x": x1,
                           "y": [round(float(v), 6) for v in ev], "line": {"width": 1.3, "color": col}, "visible": vis,
                           "legendgroup": "ma", "hovertemplate": "EMA %d (1H) %%{y}<extra></extra>" % p_})
        ev4 = df4["close"].ewm(span=50, adjust=False).mean().values[k4:]
        traces.append({"type": "scatter", "mode": "lines", "name": "EMA 50 (4H)", "x": x4,
                       "y": [round(float(v), 6) for v in ev4], "line": {"width": 2, "color": "#8d6e63", "shape": "hv"},
                       "visible": "legendonly", "legendgroup": "ma", "hovertemplate": "EMA 50 (4H) %{y}<extra></extra>"})

    # open fair-value gaps (3-candle imbalances not yet traded through) -- DISPLAY ONLY
    def _open_fvgs(df, max_bars):
        hi_, lo_ = df["high"].values, df["low"].values
        out_ = []
        for k in range(max(2, len(hi_) - max_bars), len(hi_)):
            if lo_[k] > hi_[k - 2]:
                zl, zh, bull = float(hi_[k - 2]), float(lo_[k]), True
            elif hi_[k] < lo_[k - 2]:
                zl, zh, bull = float(hi_[k]), float(lo_[k - 2]), False
            else:
                continue
            if (bull and (lo_[k + 1:] <= zl).any()) or ((not bull) and (hi_[k + 1:] >= zh).any()):
                continue
            out_.append((k, zl, zh, bull))
        return out_
    # B12 (ruling 1A): the open 1H gaps only (Test 1: 1.5x chance, both halves); the 4H gaps stay off (1.35x)
    for tf, dfx, tx, mb, vis in (("1H", df1, t1_all, 240, True),):
        for bull, col, nm in ((True, "rgba(0,200,83,0.16)", "up"), (False, "rgba(213,0,0,0.13)", "down")):
            fx, fy = [], []
            for (k, zl, zh, bb) in _open_fvgs(dfx, mb):
                if bb != bull:
                    continue
                xa = max(_fmt_t(tx[k]), x1[0])
                fx += [xa, x_end, x_end, xa, xa, None]
                fy += [zl, zl, zh, zh, zl, None]
            if fx:
                traces.append({"type": "scatter", "mode": "lines", "name": "open FVG %s (%s) -- display only: bouncing off these loses" % (nm, tf), "x": fx, "y": fy,
                               "fill": "toself", "fillcolor": col, "line": {"width": 0.5, "color": col}, "visible": vis,
                               "legendgroup": "fvg", "hoverinfo": "name"})


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{asset} -- live chart</title>
<script src="{cdn}"></script>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #fafafa; }}
#chart {{ width: 100%; height: 86vh; }}
.note {{ font-size: 12px; color: #616161; padding: 4px 10px; }}
h3 {{ margin: 14px 10px 6px; font-size: 15px; }}
ul {{ margin: 0 10px 10px 28px; font-size: 13px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 0 10px 20px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 6px 10px; font-size: 12px; }}
.card pre {{ margin: 0; white-space: pre-wrap; }}
</style></head>
<body>
<div id="chart"></div>
<div class="note">click legend items to show/hide - drag to zoom - scroll to zoom - double-click to reset - hover a triangle for its proof card - hover a stage marker for when it happened</div>
<div class="note" id="live">snapshot (open /charts on the dashboard for the live, self-refreshing view)</div>
<h3>Live setups</h3><ul>{setups}</ul>
<h3>Proof cards (newest first)</h3><div class="cards">{cards}</div>
<script>
const gd = document.getElementById("chart");
const TRACES = {traces};
const LAYOUT = {layout};
const FIT_MS = {fit_ms};
const FIT = {fit};
function toMs(v) {{ return (typeof v === "number") ? v : Date.parse(String(v).replace(" ", "T") + "Z"); }}
Plotly.newPlot(gd, TRACES, LAYOUT, {{responsive: true, displaylogo: false, scrollZoom: true}});
function fitY() {{
  const r = gd.layout.xaxis.range; if (!r) return;
  const a = toMs(r[0]), b = toMs(r[1]);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < FIT_MS.length; i++) {{
    if (FIT_MS[i] >= a && FIT_MS[i] <= b && FIT.l[i] !== null) {{ lo = Math.min(lo, FIT.l[i]); hi = Math.max(hi, FIT.h[i]); }}
  }}
  if (lo < hi) {{ const p = (hi - lo) * 0.06; Plotly.relayout(gd, {{"yaxis.range": [lo - p, hi + p]}}); }}
}}
gd.on("plotly_relayout", ev => {{
  if (ev["xaxis.range[0]"] !== undefined || ev["xaxis.range"] !== undefined || ev["xaxis.autorange"] !== undefined) setTimeout(fitY, 0);
}});
setTimeout(fitY, 0);
if (location.protocol.indexOf("http") === 0) {{
  const KEY = "nschart_range_" + location.pathname;
  try {{ const r = JSON.parse(sessionStorage.getItem(KEY) || "null"); if (r) Plotly.relayout(gd, {{"xaxis.range": r}}); }} catch (e) {{}}
  gd.on("plotly_relayout", () => {{ try {{ if (gd.layout.xaxis.range) sessionStorage.setItem(KEY, JSON.stringify(gd.layout.xaxis.range)); }} catch (e) {{}} }});
  document.getElementById("live").textContent = "LIVE -- this page refreshes itself every minute (your zoom is kept)";
  setTimeout(() => location.reload(), 60000);
}}
</script>
</body></html>
"""


def write_chart(asset, df1, df4, cs, out_dir=os.path.join("logs", "charts")):
    """Build and save logs/charts/<ASSET>.html. Returns the path, or None (and a WARNING) on failure."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "%s.html" % str(asset).upper())
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(build_chart_html(asset, df1, df4, cs))
        os.replace(tmp, path)
        return path
    except Exception as e:
        logger.warning(f"[NS-CHART] {asset}: interactive chart failed: {e}")
        return None
