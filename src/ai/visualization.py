#!/usr/bin/env python3
"""
AI Decision Visualization System -
Generates detailed charts showing:
- Candlestick patterns with 1H data
- Support/Resistance levels from 4H analysis
- Pattern detection overlays
- AI validation decisions
- Signal aggregation visualization

FIXES:
- Removed tight_layout warning by adjusting subplot spacing
- Replaced unicode emoji with ASCII equivalents
- Better error handling for missing data
"""

import logging
import io
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter, HourLocator

# Suppress specific matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = logging.getLogger(__name__)


class AIVisualizationGenerator:
    """
    Generates comprehensive visualizations of AI trading decisions
    Shows exactly what the AI "sees" at decision time
    """

    def __init__(self, analyst, sniper=None, ai_validator=None):
        """
        Args:
            analyst: DynamicAnalyst instance (4H S/R detection)
            sniper: unused — PAT-4: pattern module removed
            ai_validator: HybridSignalValidator instance
        """
        self.analyst = analyst
        self.sniper = None  # PAT-4: pattern module removed
        self.ai_validator = ai_validator

        # Set matplotlib style for dark theme
        plt.style.use("dark_background")

        logger.info("[VIZ] AI Visualization Generator initialized")

    def generate_decision_chart(
        self,
        asset_name: str,
        df_15min: pd.DataFrame,
        df_4h: pd.DataFrame,
        signal: int,
        details: Dict,
        current_price: float,
        save_path: Optional[str] = None,
        position: Optional[Dict] = None,
    ) -> Optional[io.BytesIO]:
        """
        Generate comprehensive decision visualization

        Args:
            asset_name: Asset symbol (BTC/GOLD)
            df_15min: primary-timeframe candle data (1H — every asset's config.json timeframe is H1; the param name is legacy)
            df_4h: 4-hour candle data (for S/R analysis)
            signal: Final aggregated signal (-1, 0, 1)
            details: Signal details from aggregator
            current_price: Current market price
            save_path: Optional path to save image file
            position: Optional open-position summary for entry/SL/TP markers
                on the main chart -- {"side", "entry_price", "stop_loss",
                "take_profit"}. None (default) renders exactly as before;
                the Telegram /chart caller never passes this, so its output
                is unchanged. Dashboard-driven calls pass it when a position
                is open on this asset.

        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            # Create figure with explicit spacing ( prevents tight_layout warning)
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(
                4,
                2,
                hspace=0.35,
                wspace=0.35,
                left=0.08,
                right=0.95,
                top=0.92,
                bottom=0.08,
            )

            # Main chart: 15min candlesticks with overlays
            ax_main = fig.add_subplot(gs[0:2, :])
            self._plot_main_chart(ax_main, df_15min, asset_name, current_price, details, position=position)

            # Support/Resistance levels from 4H
            ax_sr = fig.add_subplot(gs[2, 0])
            self._plot_sr_levels(ax_sr, df_4h, current_price, details)

            # Pattern detection — PAT-4: removed, slot repurposed as blank
            ax_pattern = fig.add_subplot(gs[2, 1])
            ax_pattern.text(0.5, 0.5, "Pattern Module: DISABLED", ha="center", va="center", fontsize=11)
            ax_pattern.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
            ax_pattern.axis("off")

            # Signal aggregation breakdown
            ax_signals = fig.add_subplot(gs[3, 0])
            self._plot_signal_breakdown(ax_signals, details)

            # AI Validation summary
            ax_ai = fig.add_subplot(gs[3, 1])
            self._plot_ai_validation(ax_ai, details, current_price)

            # Add title with timestamp ( No unicode emoji)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            signal_type = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
            signal_color = (
                "lime" if signal == 1 else "red" if signal == -1 else "yellow"
            )

            # Use ASCII symbols instead of unicode
            signal_symbol = "[+]" if signal == 1 else "[-]" if signal == -1 else "[=]"

            fig.suptitle(
                f"{asset_name} - AI Trading Decision: {signal_symbol} {signal_type}\n"
                f"Quality: {details.get('signal_quality', 0):.2%} | {timestamp}",
                fontsize=16,
                fontweight="bold",
                color=signal_color,
            )

            #  Don't use tight_layout (already handled by gridspec)

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(
                buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1e1e1e"
            )
            buf.seek(0)

            # Optionally save to file
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    save_path,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                    facecolor="#1e1e1e",
                )
                logger.info(f"[VIZ] Chart saved to {save_path}")

            plt.close(fig)

            logger.info(f"[VIZ] Generated decision chart for {asset_name}")
            return buf

        except Exception as e:
            logger.error(f"[VIZ] Chart generation failed: {e}", exc_info=True)
            if "fig" in locals():
                plt.close(fig)
            return None

    def _plot_main_chart(
        self, ax, df: pd.DataFrame, asset_name: str, current_price: float, details: Dict,
        position: Optional[Dict] = None,
    ):
        """Plot main 1H candlestick chart with overlays"""
        try:
            # Make a copy to avoid modifying original
            df = df.copy()

            # Ensure datetime index
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            else:
                # Create synthetic date index
                df["date"] = pd.date_range(
                    end=datetime.now(), periods=len(df), freq="1h"
                )

            # Limit to last 100 candles for clarity
            df_plot = df.tail(100).copy()

            if len(df_plot) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No candle data available",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(
                    f"{asset_name} - 1H Candlesticks", fontsize=12, fontweight="bold"
                )
                return

            # Plot candlesticks manually (more reliable than mplfinance)
            self._plot_candlesticks(ax, df_plot)

            # Nearest real zone-ladder level, not the legacy sr_analysis
            # heuristic -- same composite_state.zone_4h_current_upper/lower
            # the S/R panel and live scoring both read.
            _cs = details.get("composite_state") if isinstance(details, dict) else None
            if isinstance(_cs, dict):
                _upper = _cs.get("zone_4h_current_upper")
                _lower = _cs.get("zone_4h_current_lower")
                _candidates = [
                    (abs(lvl - current_price), lvl, "Resistance" if lvl >= current_price else "Support")
                    for lvl in (_upper, _lower) if lvl is not None
                ]
                if _candidates:
                    _, nearest_level, level_type = min(_candidates, key=lambda c: c[0])
                    ax.axhline(
                        y=nearest_level,
                        color="orange",
                        linestyle="--",
                        linewidth=2,
                        label=f"{level_type}: ${nearest_level:,.2f}",
                    )

            # Mark current price
            ax.axhline(
                y=current_price,
                color="white",
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                label=f"Current: ${current_price:,.2f}",
            )

            # Open-position markers (dashboard only -- Telegram's /chart never
            # passes `position`, so this block is a no-op there).
            if position:
                _side = position.get("side", "?")
                _entry = position.get("entry_price")
                _sl = position.get("stop_loss")
                _tp = position.get("take_profit")
                if _entry is not None:
                    ax.axhline(
                        y=_entry, color="0.8", linestyle="--", linewidth=1.5,
                        label=f"Entry ({_side}): ${_entry:,.2f}",
                    )
                if _sl is not None:
                    ax.axhline(
                        y=_sl, color="red", linestyle="--", linewidth=1.5,
                        label=f"SL: ${_sl:,.2f}",
                    )
                if _tp is not None:
                    ax.axhline(
                        y=_tp, color="lime", linestyle="--", linewidth=1.5,
                        label=f"TP: ${_tp:,.2f}",
                    )

            # ✨ NEW: Advanced Overlays (Divergence, B&R, Liquidity)
            viz_overlay = details.get("viz_overlay", {})
            if viz_overlay:
                # 1. RSI Divergence Overlay
                div = viz_overlay.get("divergence")
                if div and div.type != "NONE":
                    color = "lime" if "BULLISH" in div.type else "red"
                    marker = "^" if "BULLISH" in div.type else "v"
                    ax.annotate(
                        f"DIV: {div.type}",
                        xy=(len(df_plot)-1, current_price),
                        xytext=(len(df_plot)-10, current_price * (1.005 if color == "lime" else 0.995)),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=5),
                        fontsize=9, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec=color, alpha=0.7)
                    )

                # 2. Break-and-Retest Overlay
                br = viz_overlay.get("break_retest")
                if br and br.is_valid:
                    ax.axhline(y=br.level, color="cyan", linestyle=":", alpha=0.6)
                    ax.scatter(len(df_plot)-1, br.level, color="cyan", s=100, marker="*", label="Retest Zone", zorder=5)

                # 3. Liquidity Sweep Marker (Wick Trap)
                latest = df_plot.iloc[-1]
                body = abs(latest['close'] - latest['open'])
                high_wick = latest['high'] - max(latest['open'], latest['close'])
                low_wick = min(latest['open'], latest['close']) - latest['low']

                if (high_wick > 2.0 * body) or (low_wick > 2.0 * body):
                    ax.scatter(len(df_plot)-1, latest['high'] if high_wick > low_wick else latest['low'], 
                               color="orange", s=150, facecolors='none', edgecolors='orange', 
                               linewidth=2, label="Liquidity Sweep")

            # Add EMAs if available - try multiple common column naming conventions
            ema_defs = [
                (["ema_20", "EMA_20", "EMA20", "ema20"], "yellow",   1.2, "EMA 20"),
                (["ema_50", "EMA_50", "EMA50", "ema50"], "cyan",     1.5, "EMA 50"),
                (["ema_200","EMA_200","EMA200","ema200"],"magenta",  1.5, "EMA 200"),
            ]
            for col_names, color, lw, label in ema_defs:
                col = next((c for c in col_names if c in df_plot.columns and df_plot[c].notna().any()), None)
                if col:
                    ax.plot(
                        range(len(df_plot)),
                        df_plot[col],
                        color=color,
                        linewidth=lw,
                        label=label,
                        alpha=0.7,
                    )

            ax.set_title(
                f"{asset_name} - 1H Candlesticks (Last 100 candles)",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Candle Index", fontsize=10)
            ax.set_ylabel("Price ($)", fontsize=10)

        except Exception as e:
            logger.error(f"[VIZ] Main chart error: {e}", exc_info=True)
            ax.text(
                0.5,
                0.5,
                f"Chart Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_candlesticks(self, ax, df):
        """Manually plot candlesticks (more reliable than mplfinance)"""
        try:
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values

            # Plot wicks
            for i in range(len(df)):
                color = "lime" if closes[i] >= opens[i] else "red"
                ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)

            # Plot bodies
            for i in range(len(df)):
                body_height = abs(closes[i] - opens[i])
                body_bottom = min(opens[i], closes[i])
                color = "lime" if closes[i] >= opens[i] else "red"

                rect = Rectangle(
                    (i - 0.3, body_bottom),
                    0.6,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.8,
                )
                ax.add_patch(rect)

            # Set limits
            ax.set_xlim(-1, len(df))
            price_range = df[["high", "low"]].values.flatten()
            ax.set_ylim(price_range.min() * 0.998, price_range.max() * 1.002)

        except Exception as e:
            logger.error(f"[VIZ] Candlestick plot error: {e}")

    def _plot_sr_levels(
        self, ax, df_4h: pd.DataFrame, current_price: float, details: Dict
    ):
        """
        Plot Support/Resistance levels -- the real zone ladder
        (_structure_levels/_zone_levels, role-flip and test-count tracked),
        the same levels the BRC proof pipeline actually scores against.
        The legacy ai_validation.sr_analysis heuristic panel this used to
        fall back to has been removed entirely -- it drew a different,
        older S/R system than the one live trading reads, so a fallback to
        it was worse than an honest empty state.
        """
        _cs = details.get("composite_state") if isinstance(details, dict) else None
        _ladder = (_cs or {}).get("zone_ladder_4h", []) if isinstance(_cs, dict) else []
        self._plot_real_zone_ladder(ax, df_4h, current_price, _ladder, _cs)

    def _plot_real_zone_ladder(
        self, ax, df_4h: pd.DataFrame, current_price: float,
        ladder: List[Dict], composite_state: Optional[Dict],
    ):
        """Plot the REAL zone ladder -- _structure_levels/_zone_levels, the
        role-flip/test-count-tracked levels this session's BRC proof
        pipeline (ZONE_LADDER tier) actually reads.
        """
        try:
            _have_4h = df_4h is not None and len(df_4h) >= 10
            if _have_4h:
                df_plot = df_4h.tail(60).copy()
                x_range = range(len(df_plot))
                closes = df_plot["close"].values
                highs = df_plot["high"].values
                lows = df_plot["low"].values
                ax.plot(x_range, closes, color="cyan", linewidth=2, alpha=0.8,
                         label="4H Close", zorder=5)
                ax.fill_between(x_range, lows, highs, color="gray", alpha=0.15, zorder=1)
                price_min = float(df_plot[["high", "low"]].values.min())
                price_max = float(df_plot[["high", "low"]].values.max())
            else:
                # No 4H data to plot candles against -- still show the ladder
                # itself, scaled to the levels and current price.
                x_range = range(2)
                prices = [l["price"] for l in ladder] + [current_price]
                price_min, price_max = min(prices), max(prices)

            price_range = max(price_max - price_min, 1e-9)
            buffer = price_range * 0.1

            # Each level: role-colored (resistance=red, support=lime), line
            # weight and alpha scale with test count so a level that's earned
            # real history reads as more significant than a fresh one. Zone
            # band width is the REAL dedup tolerance stored at the level's
            # creation time (composite_state_builder._update_zone_levels) --
            # not a cosmetic guess. Levels created before that field existed
            # get width=0 and fall back to a thin line, honestly.
            for lvl in ladder:
                price = lvl["price"]
                if not (price_min - buffer <= price <= price_max + buffer):
                    continue
                tests = int(lvl.get("tests", 0) or 0)
                width = float(lvl.get("zone_width", 0) or 0)
                is_resistance = lvl.get("type") == "swing_high"
                color = "red" if is_resistance else "lime"
                role_label = "Resistance" if is_resistance else "Support"
                flipped = " (flipped)" if lvl.get("role_flipped_at") else ""
                if width > 0:
                    ax.axhspan(
                        price - width, price + width, color=color,
                        alpha=min(0.10 + 0.03 * tests, 0.30), zorder=2,
                    )
                ax.axhline(
                    y=price, color=color, linestyle="--",
                    linewidth=min(1.0 + 0.5 * tests, 3.5),
                    alpha=min(0.5 + 0.1 * tests, 0.95),
                    label=f"{role_label}{flipped}: ${price:,.4g} ({tests}x)",
                    zorder=3,
                )

            # Real trendlines -- connect the two most recent swing highs
            # (resistance trendline) and the two most recent swing lows
            # (support trendline) actually present in this visible 4H
            # window, using the same 3-bar-fractal definition of a swing
            # point _update_zone_levels uses. Computed fresh from the
            # plotted candles, not read from a stored/fabricated source --
            # skipped entirely if the window doesn't have two of a kind.
            if _have_4h and len(df_plot) >= 5:
                def _swings(arr, is_high):
                    pts = []
                    for i in range(2, len(arr) - 2):
                        if is_high and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                            pts.append((i, arr[i]))
                        elif not is_high and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                            pts.append((i, arr[i]))
                    return pts

                _last_x = len(df_plot) - 1
                _high_swings = _swings(highs, True)
                _low_swings = _swings(lows, False)

                if len(_high_swings) >= 2:
                    (x1, y1), (x2, y2) = _high_swings[-2], _high_swings[-1]
                    if x2 > x1:
                        slope = (y2 - y1) / (x2 - x1)
                        y_end = y2 + slope * (_last_x - x2)
                        ax.plot(
                            [x1, _last_x], [y1, y_end], color="orange",
                            linestyle="-", linewidth=1.6, alpha=0.85,
                            label="Trendline (highs)", zorder=6,
                        )

                if len(_low_swings) >= 2:
                    (x1, y1), (x2, y2) = _low_swings[-2], _low_swings[-1]
                    if x2 > x1:
                        slope = (y2 - y1) / (x2 - x1)
                        y_end = y2 + slope * (_last_x - x2)
                        ax.plot(
                            [x1, _last_x], [y1, y_end], color="dodgerblue",
                            linestyle="-", linewidth=1.6, alpha=0.85,
                            label="Trendline (lows)", zorder=6,
                        )

            ax.axhline(
                y=current_price, color="white", linestyle="-", linewidth=2.5,
                alpha=0.9, label=f"Current: ${current_price:,.4g}", zorder=10,
            )

            ax.set_xlim(-1, max(len(x_range) - 1, 1))
            ax.set_ylim(price_min - buffer, price_max + buffer)
            ax.set_xlabel("4H Candles (Recent 60)", fontsize=10)
            ax.set_ylabel("Price", fontsize=10)
            ax.set_title("Zone Ladder (4H) — role/test-tracked", fontsize=11, fontweight="bold")
            # Fixed corner, not "best" -- "best" kept picking upper-left,
            # the same corner as the info box below, so the two boxes
            # overlapped whenever the price line left that corner empty.
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.3, which="both")

            _upper = (composite_state or {}).get("zone_4h_current_upper")
            _lower = (composite_state or {}).get("zone_4h_current_lower")
            info_text = (
                f"Ladder levels: {len(ladder)}\n"
                f"Nearest above: {_upper:,.4g}" if _upper is not None else "Nearest above: none"
            )
            if _lower is not None:
                info_text += f"\nNearest below: {_lower:,.4g}"
            ax.text(
                0.02, 0.98, info_text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="black", edgecolor="orange",
                          alpha=0.8, linewidth=2),
            )
        except Exception as e:
            logger.error(f"[VIZ] Zone ladder plot error: {e}", exc_info=True)
            ax.text(0.5, 0.5, "Error displaying zone ladder", ha="center", va="center", fontsize=10)
            ax.set_title("Zone Ladder (4H)", fontsize=11, fontweight="bold")

    def _plot_pattern_analysis(self, ax, df: pd.DataFrame, details: Dict):
        """PAT-4: pattern module removed — no-op stub kept for call-site compatibility."""
        ax.text(0.5, 0.5, "Pattern Module: DISABLED", ha="center", va="center", fontsize=11)
        ax.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
        ax.axis("off")

    def _plot_signal_breakdown(self, ax, details: Dict):
        """Plot signal aggregation breakdown — handles both performance and council modes"""
        try:
            agg_mode = details.get("aggregator_mode", "performance")
            final_quality = details.get("signal_quality", 0)
            regime = details.get("regime", "UNKNOWN")
            trade_type = details.get("trade_type", "")
            adx_val = details.get("adx", None)

            # ── Info box text ─────────────────────────────────────────────
            info_lines = [f"Quality: {final_quality:.1%}", f"Regime: {regime}"]
            if trade_type:
                info_lines.append(f"Mode: {trade_type}")
            if adx_val is not None:
                adx_label = "Strong" if adx_val >= 40 else "Trending" if adx_val >= 25 else "Weak"
                info_lines.append(f"ADX: {adx_val:.1f} ({adx_label})")

            # ── Council aggregator mode ───────────────────────────────────
            if agg_mode == "council":
                total_score = details.get("total_score", 0) or 0
                decision_type = details.get("decision_type", "N/A")

                # Prefer buy_scores or sell_scores depending on the final signal
                buy_scores = details.get("buy_scores", {}) or {}
                sell_scores = details.get("sell_scores", {}) or {}

                # Use whichever side has the higher total
                buy_total = sum(buy_scores.values()) if buy_scores else 0
                sell_total = sum(sell_scores.values()) if sell_scores else 0
                judge_scores = sell_scores if sell_total >= buy_total else buy_scores
                side_label = "SELL" if sell_total >= buy_total else "BUY"

                judge_names = list(judge_scores.keys())
                judge_values = [judge_scores[k] for k in judge_names]

                if not judge_names:
                    # Fallback: just show total score bar
                    judge_names = ["Council Score"]
                    judge_values = [min(total_score / 5.0, 1.0)]

                colors = ["lime" if side_label == "BUY" else "red"] * len(judge_names)
                y_pos = range(len(judge_names))

                ax.barh(y_pos, judge_values, color=colors, alpha=0.7, edgecolor="white")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([n.replace("_", " ").title() for n in judge_names], fontsize=8)
                ax.set_xlabel("Judge Score", fontsize=10)
                ax.set_xlim(0, max(max(judge_values, default=1) * 1.2, 1.5))

                # Value labels
                for i, val in enumerate(judge_values):
                    ax.text(val + 0.02, i, f"{val:.2f}", va="center", fontsize=8)

                ax.set_title(
                    f"Council Judges — {decision_type}  ({side_label} lean: {total_score:.2f}/5.0)",
                    fontsize=11, fontweight="bold"
                )
                info_lines.insert(0, f"Engine: COUNCIL")
                info_lines.insert(1, f"Decision: {decision_type}")

            else:
                # ── Performance-weighted aggregator ───────────────────────
                # Try both short-form and long-form key names (aggregator may use either)
                mr_signal = details.get("mean_reversion_signal") or details.get("mr_signal", 0)
                mr_conf   = details.get("mean_reversion_confidence") or details.get("mr_confidence", 0) or 0
                tf_signal = details.get("trend_following_signal") or details.get("tf_signal", 0)
                tf_conf   = details.get("trend_following_confidence") or details.get("tf_confidence", 0) or 0
                ema_signal = details.get("ema_signal", 0)
                ema_conf   = details.get("ema_confidence") or details.get("ema_conf", 0) or 0

                strategies  = ["Mean\nReversion", "Trend\nFollowing", "EMA\nRegime"]
                signals     = [mr_signal, tf_signal, ema_signal]
                confidences = [mr_conf, tf_conf, ema_conf]

                colors = []
                for sig in signals:
                    if sig == 1:
                        colors.append("lime")
                    elif sig == -1:
                        colors.append("red")
                    else:
                        colors.append("gray")

                x = range(len(strategies))
                bars = ax.bar(x, confidences, color=colors, alpha=0.7, edgecolor="white")

                for bar, sig, conf in zip(bars, signals, confidences):
                    if conf > 0.05:
                        signal_text = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            conf / 2,
                            f"{signal_text}\n{conf:.2%}",
                            ha="center", va="center",
                            fontsize=9, fontweight="bold",
                        )

                ax.set_xticks(x)
                ax.set_xticklabels(strategies, fontsize=9)
                ax.set_ylabel("Confidence", fontsize=10)
                ax.set_ylim(0, 1)
                ax.set_title("Individual Strategy Signals", fontsize=11, fontweight="bold")
                info_lines.insert(0, "Engine: PERF-WEIGHTED")

            ax.grid(True, alpha=0.3, axis="y" if agg_mode != "council" else "x")

            # Info box
            ax.text(
                0.02, 0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
            )

        except Exception as e:
            logger.error(f"[VIZ] Signal breakdown error: {e}", exc_info=True)
            ax.text(0.5, 0.5, f"Breakdown Error: {str(e)}", ha="center", va="center")

    def _plot_ai_validation(self, ax, details: Dict, current_price: float = None):
        """
        AI validation summary with comprehensive data display
        """
        try:
            ax.axis("off")

            # Extract AI details
            ai_details = details.get("ai_validation", {})

            # Handle different response formats
            if isinstance(ai_details, str):
                summary_text = (
                    "=== AI VALIDATION ===\n\n"
                    f"{ai_details.upper()}\n\n"
                    "Status: Disabled or Error"
                )
                ax.text(
                    0.05,
                    0.95,
                    summary_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    family="monospace",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="#1e1e1e",
                        edgecolor="yellow",
                        linewidth=2,
                        alpha=0.9,
                    ),
                )
                ax.set_title("AI Validation Summary", fontsize=11, fontweight="bold")
                return

            if not ai_details or not isinstance(ai_details, dict):
                summary_text = (
                    "=== AI VALIDATION ===\n\n"
                    "[!] NOT AVAILABLE\n\n"
                    "AI validation may be disabled\n"
                    "or initialization failed."
                )
                ax.text(
                    0.05,
                    0.95,
                    summary_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    family="monospace",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="#1e1e1e",
                        edgecolor="red",
                        linewidth=2,
                        alpha=0.9,
                    ),
                )
                ax.set_title("AI Validation Summary", fontsize=11, fontweight="bold")
                return

            # Build comprehensive summary
            summary_lines = []
            summary_lines.append("=== AI VALIDATION RESULTS ===\n")

            # Pattern Detection Section
            pattern_detected = ai_details.get("pattern_detected", False)
            pattern_name = ai_details.get("pattern_name", "None")
            pattern_conf = ai_details.get("pattern_confidence", 0)

            if pattern_name != "None":
                summary_lines.append(f"[PATTERN DETECTED]")
                summary_lines.append(f"  Name:       {pattern_name}")
                summary_lines.append(f"  Confidence: {pattern_conf:.1%}")

                # Show top 3 if available
                top3 = ai_details.get("top3_patterns", [])
                top3_conf = ai_details.get("top3_confidences", [])
                if len(top3) > 1:  # More than just the detected pattern
                    summary_lines.append(f"\n  Top Alternatives:")
                    for i in range(min(3, len(top3))):
                        if i < len(top3_conf):
                            summary_lines.append(
                                f"    {i+1}. {top3[i]}: {top3_conf[i]:.1%}"
                            )
            else:
                summary_lines.append(f"[NO PATTERN]")
                if pattern_conf > 0:
                    summary_lines.append(
                        f"  Highest: {pattern_name} ({pattern_conf:.1%})"
                    )
                else:
                    summary_lines.append(f"  No significant patterns detected")

            summary_lines.append("")

            # Zone Ladder Section — real _zone_levels (role-flip/test-count
            # tracked), the same levels the BRC proof pipeline scores
            # against, not the legacy sr_analysis heuristic.
            _cs = details.get("composite_state") if isinstance(details, dict) else None
            _ladder = (_cs or {}).get("zone_ladder_4h", []) if isinstance(_cs, dict) else []
            _candidates = []
            if isinstance(_cs, dict):
                _upper = _cs.get("zone_4h_current_upper")
                _lower = _cs.get("zone_4h_current_lower")
                for lvl in (_upper, _lower):
                    if lvl is not None:
                        _dist = (
                            abs(lvl - current_price) / current_price * 100
                            if current_price else None
                        )
                        _candidates.append((lvl, "RESISTANCE" if lvl >= (current_price or lvl) else "SUPPORT", _dist))

            summary_lines.append(f"[ZONE LADDER]")
            if _candidates:
                nearest, level_type, distance = min(
                    _candidates, key=lambda c: c[2] if c[2] is not None else float("inf")
                )
                summary_lines.append(f"  Near {level_type}:  ${nearest:,.4g}")
                if distance is not None:
                    summary_lines.append(f"  Distance:   {distance:.2f}%")
                summary_lines.append(f"  Total Levels: {len(_ladder)}")
            else:
                summary_lines.append(f"  Status: No nearby levels")
                if _ladder:
                    summary_lines.append(f"  ({len(_ladder)} levels found)")

            summary_lines.append("")

            # Validation Decision Section
            validation_passed = ai_details.get("validation_passed", False)
            action = ai_details.get("action", "unknown")

            if validation_passed:
                if "bypass" in action.lower():
                    summary_lines.append(f"[RESULT: BYPASSED]")
                    summary_lines.append(f"  Strong signal quality")
                    summary_lines.append(f"  AI validation skipped")
                else:
                    summary_lines.append(f"[RESULT: APPROVED]")
                    summary_lines.append(f"  All layers passed")
                    summary_lines.append(f"  Signal confirmed by AI")
            elif action in ("hold", "none", "ai_disabled"):
                # HOLD — score below threshold, not an AI rejection
                summary_lines.append(f"[RESULT: HOLD]")
                if action == "ai_disabled":
                    summary_lines.append(f"  AI validation disabled")
                else:
                    summary_lines.append(f"  Score below signal threshold")
                    summary_lines.append(f"  No trade action taken")
            else:
                # Genuine rejection: AI validator blocked a non-zero signal
                summary_lines.append(f"[RESULT: REJECTED]")
                reasons = ai_details.get("rejection_reasons", [])
                # Fallback: single-string reason from hybrid_validator
                if not reasons:
                    single = ai_details.get("ai_rejection_reason", "")
                    if single:
                        reasons = [single.replace("_", " ").title()]
                if reasons:
                    summary_lines.append(f"  Reasons:")
                    for reason in reasons[:3]:
                        summary_lines.append(f"    - {reason}")
                else:
                    summary_lines.append(f"  Signal blocked by AI filter")

            # Check for errors
            if ai_details.get("error"):
                summary_lines.append(f"\n[ERROR]")
                summary_lines.append(f"  {ai_details['error'][:40]}")

            # ── Engine context (trade type, ADX, regime, council score) ──
            summary_lines.append("")
            summary_lines.append("[ENGINE CONTEXT]")

            agg_mode   = details.get("aggregator_mode", "performance")
            trade_type = details.get("trade_type", "")
            adx_val    = details.get("adx", None)
            regime_str = details.get("regime", "")

            summary_lines.append(f"  Engine: {'COUNCIL' if agg_mode == 'council' else 'PERF-WEIGHTED'}")

            if trade_type:
                risk_map = {"TREND": "2.0% risk", "SCALP": "1.0% risk"}
                risk_str = risk_map.get(trade_type, "1.5% risk")
                summary_lines.append(f"  Type:   {trade_type} ({risk_str})")

            if adx_val is not None:
                adx_label = "Strong" if adx_val >= 40 else "Trending" if adx_val >= 25 else "Weak"
                summary_lines.append(f"  ADX:    {adx_val:.1f} ({adx_label})")

            if regime_str:
                summary_lines.append(f"  Regime: {regime_str}")

            # Council score if available
            if agg_mode == "council":
                total_score = details.get("total_score")
                decision    = details.get("decision_type", "")
                if total_score is not None:
                    summary_lines.append(f"  Score:  {total_score:.2f}/5.0  [{decision}]")

            # Display the summary
            summary_text = "\n".join(summary_lines)

            # Color code border based on result
            if validation_passed:
                if "bypass" in action.lower():
                    border_color = "yellow"
                else:
                    border_color = "lime"
            elif action in ("hold", "none", "ai_disabled"):
                border_color = "gray"       # HOLD — neutral, not a failure
            else:
                border_color = "red"        # REJECTED — AI blocked a real signal

            ax.text(
                0.05,
                0.95,
                summary_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox=dict(
                    boxstyle="round",
                    facecolor="#1e1e1e",
                    edgecolor=border_color,
                    linewidth=2,
                    alpha=0.9,
                ),
            )

            ax.set_title("AI Validation Summary", fontsize=11, fontweight="bold")

        except Exception as e:
            logger.error(f"[VIZ] AI validation plot error: {e}", exc_info=True)
            ax.text(
                0.5,
                0.5,
                f"Error displaying validation:\n{str(e)[:50]}",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )
            ax.set_title("AI Validation Summary", fontsize=11, fontweight="bold")
            ax.axis("off")


class TelegramChartSender:
    """
    Integration layer to send AI visualization charts via Telegram
    """

    def __init__(self, telegram_bot, visualizer: AIVisualizationGenerator):
        """
        Args:
            telegram_bot: TradingTelegramBot instance
            visualizer: AIVisualizationGenerator instance
        """
        self.telegram_bot = telegram_bot
        self.visualizer = visualizer
        logger.info("[TELEGRAM] Chart sender initialized")

    async def send_decision_chart(
        self,
        asset_name: str,
        df_15min: pd.DataFrame,
        df_4h: pd.DataFrame,
        signal: int,
        details: Dict,
        current_price: float,
    ):
        """
        Generate and send AI decision chart via Telegram

        Args:
            asset_name: Asset symbol
            df_15min: primary-timeframe data (1H)
            df_4h: 4-hour data
            signal: Final signal
            details: Signal details
            current_price: Current price
        """
        try:
            if not self.telegram_bot or not self.telegram_bot._is_ready:
                logger.debug("[TELEGRAM] Bot not ready, skipping chart")
                return

            # Generate chart
            logger.info(f"[TELEGRAM] Generating chart for {asset_name}...")
            chart_buffer = self.visualizer.generate_decision_chart(
                asset_name=asset_name,
                df_15min=df_15min,
                df_4h=df_4h,
                signal=signal,
                details=details,
                current_price=current_price,
            )

            if not chart_buffer:
                logger.error("[TELEGRAM] Chart generation failed")
                return

            # Prepare caption (ASCII symbols) - escape Markdown special chars
            signal_type = (
                "[+] BUY" if signal == 1 else "[-] SELL" if signal == -1 else "[=] HOLD"
            )
            quality = details.get("signal_quality", 0)
            regime = details.get("regime", "UNKNOWN")
            reasoning = details.get("reasoning", "N/A")

            # Escape Markdown special characters in dynamic content
            def escape_markdown(text: str) -> str:
                """Escape special Markdown characters"""
                special_chars = [
                    "_",
                    "*",
                    "[",
                    "]",
                    "(",
                    ")",
                    "~",
                    "`",
                    ">",
                    "#",
                    "+",
                    "-",
                    "=",
                    "|",
                    "{",
                    "}",
                    ".",
                    "!",
                ]
                for char in special_chars:
                    text = text.replace(char, f"\\{char}")
                return text

            asset_escaped = escape_markdown(asset_name)
            regime_escaped = escape_markdown(regime)
            reasoning_escaped = escape_markdown(reasoning[:100])

            caption = (
                f"*{asset_escaped} AI Decision Chart*\n\n"
                f"Signal: {signal_type}\n"
                f"Quality: {quality:.1%}\n"
                f"Regime: {regime_escaped}\n"
                f"Reasoning: {reasoning_escaped}\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Send to all admins — seek(0) before each send so the buffer
            # is always at the start (a consumed buffer produces a blank image
            # for the 2nd+ admin if only reset once at the end)
            for admin_id in self.telegram_bot.admin_ids:
                try:
                    chart_buffer.seek(0)
                    await self.telegram_bot.application.bot.send_photo(
                        chat_id=admin_id,
                        photo=chart_buffer,
                        caption=caption,
                        parse_mode="Markdown",
                    )
                    logger.info(f"[TELEGRAM] Chart sent to admin {admin_id}")
                except Exception as e:
                    logger.error(f"[TELEGRAM] Failed to send to {admin_id}: {e}")

        except Exception as e:
            logger.error(f"[TELEGRAM] Chart sending error: {e}", exc_info=True)


# ============================================================================
# INTEGRATION HELPER FUNCTIONS
# ============================================================================


def create_visualization_system(telegram_bot, analyst, sniper, ai_validator):
    """
    Create complete visualization system

    Args:
        telegram_bot: TradingTelegramBot instance
        analyst: DynamicAnalyst instance
        sniper: OHLCSniper instance
        ai_validator: HybridSignalValidator instance

    Returns:
        TelegramChartSender instance ready to use
    """
    try:
        visualizer = AIVisualizationGenerator(
            analyst=analyst, sniper=sniper, ai_validator=ai_validator
        )

        chart_sender = TelegramChartSender(
            telegram_bot=telegram_bot, visualizer=visualizer
        )

        logger.info("[VIZ] Visualization system created successfully")
        return chart_sender

    except Exception as e:
        logger.error(f"[VIZ] System creation failed: {e}")
        return None


def should_send_chart(signal: int, details: Dict, config: Dict) -> bool:
    """
    Determine if chart should be sent based on configuration

    Args:
        signal: Trading signal
        details: Signal details
        config: Bot configuration

    Returns:
        True if chart should be sent
    """
    # Check if visualization is enabled
    viz_config = config.get("telegram", {}).get("visualization", {})
    if not viz_config.get("enabled", True):
        return False

    # Always send on BUY/SELL signals
    if signal != 0:
        return True

    # For HOLD signals, check if high quality
    if viz_config.get("send_high_quality_holds", False):
        quality = details.get("signal_quality", 0)
        threshold = viz_config.get("high_quality_threshold", 0.65)
        return quality >= threshold

    return False
