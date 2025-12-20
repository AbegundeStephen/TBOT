#!/usr/bin/env python3
"""
AI Decision Visualization System - FIXED
Generates detailed charts showing:
- Candlestick patterns with 15min data
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

    def __init__(self, analyst, sniper, ai_validator):
        """
        Args:
            analyst: DynamicAnalyst instance (4H S/R detection)
            sniper: OHLCSniper instance (15min pattern detection)
            ai_validator: HybridSignalValidator instance
        """
        self.analyst = analyst
        self.sniper = sniper
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
    ) -> Optional[io.BytesIO]:
        """
        Generate comprehensive decision visualization

        Args:
            asset_name: Asset symbol (BTC/GOLD)
            df_15min: 15-minute candle data (for pattern detection)
            df_4h: 4-hour candle data (for S/R analysis)
            signal: Final aggregated signal (-1, 0, 1)
            details: Signal details from aggregator
            current_price: Current market price
            save_path: Optional path to save image file

        Returns:
            BytesIO buffer containing the chart image
        """
        try:
            # Create figure with explicit spacing (FIXED: prevents tight_layout warning)
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
            self._plot_main_chart(ax_main, df_15min, asset_name, current_price, details)

            # Support/Resistance levels from 4H
            ax_sr = fig.add_subplot(gs[2, 0])
            self._plot_sr_levels(ax_sr, df_4h, current_price, details)

            # Pattern detection heatmap
            ax_pattern = fig.add_subplot(gs[2, 1])
            self._plot_pattern_analysis(ax_pattern, df_15min, details)

            # Signal aggregation breakdown
            ax_signals = fig.add_subplot(gs[3, 0])
            self._plot_signal_breakdown(ax_signals, details)

            # AI Validation summary
            ax_ai = fig.add_subplot(gs[3, 1])
            self._plot_ai_validation(ax_ai, details)

            # Add title with timestamp (FIXED: No unicode emoji)
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

            # FIXED: Don't use tight_layout (already handled by gridspec)

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
        self, ax, df: pd.DataFrame, asset_name: str, current_price: float, details: Dict
    ):
        """Plot main 15min candlestick chart with overlays"""
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
                    end=datetime.now(), periods=len(df), freq="15min"
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
                    f"{asset_name} - 15min Candlesticks", fontsize=12, fontweight="bold"
                )
                return

            # Plot candlesticks manually (more reliable than mplfinance)
            self._plot_candlesticks(ax, df_plot)

            # Overlay pattern detection zones
            ai_details = details.get("ai_validation", {})
            if isinstance(ai_details, dict) and ai_details.get("pattern_detected"):
                pattern_name = ai_details.get("pattern_name", "Unknown")
                confidence = ai_details.get("pattern_confidence", 0)

                # Highlight the last 15 candles (pattern window)
                pattern_start = len(df_plot) - 15
                if pattern_start >= 0:
                    ax.axvspan(
                        pattern_start,
                        len(df_plot),
                        alpha=0.2,
                        color="cyan",
                        label=f"Pattern Window: {pattern_name} ({confidence:.0%})",
                    )

            # Show support/resistance levels
            if isinstance(ai_details, dict):
                sr_details = ai_details.get("sr_analysis", {})
                if isinstance(sr_details, dict) and sr_details.get("near_sr_level"):
                    nearest_level = sr_details.get("nearest_level", current_price)
                    level_type = sr_details.get("level_type", "Unknown")

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

            # Add EMAs if available
            if "ema_50" in df_plot.columns and df_plot["ema_50"].notna().any():
                ax.plot(
                    range(len(df_plot)),
                    df_plot["ema_50"],
                    color="cyan",
                    linewidth=1.5,
                    label="EMA 50",
                    alpha=0.7,
                )
            if "ema_200" in df_plot.columns and df_plot["ema_200"].notna().any():
                ax.plot(
                    range(len(df_plot)),
                    df_plot["ema_200"],
                    color="magenta",
                    linewidth=1.5,
                    label="EMA 200",
                    alpha=0.7,
                )

            ax.set_title(
                f"{asset_name} - 15min Candlesticks (Last 100 candles)",
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
        Plot Support/Resistance levels as trend chart with overlaid S/R lines
        """
        try:
            # ✅ FIX: Defensive extraction of AI details
            ai_details = details.get("ai_validation", {})

            # Handle both dict and string formats
            if isinstance(ai_details, str):
                ax.text(
                    0.5,
                    0.5,
                    f"AI Status: {ai_details}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(
                    "4H Support/Resistance Analysis", fontsize=11, fontweight="bold"
                )
                return

            # If dict is empty or no SR data
            if not ai_details or not isinstance(ai_details, dict):
                ax.text(
                    0.5,
                    0.5,
                    "AI Validation: Not Available",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(
                    "4H Support/Resistance Analysis", fontsize=11, fontweight="bold"
                )
                return

            # Extract SR analysis
            sr_analysis = ai_details.get("sr_analysis", {})

            if not sr_analysis or not sr_analysis.get("levels"):
                ax.text(
                    0.5,
                    0.5,
                    "No S/R Levels Found",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(
                    "4H Support/Resistance Analysis", fontsize=11, fontweight="bold"
                )
                return

            # Check if we have 4H dataframe
            if df_4h is None or len(df_4h) == 0:
                # Fallback to bar chart if no 4H data
                self._plot_sr_levels_bars(ax, sr_analysis, current_price)
                return

            levels = sr_analysis.get("levels", [])

            # Take last 60 candles for clarity
            df_plot = df_4h.tail(60).copy()

            if len(df_plot) < 10:
                # Not enough data, use bar chart
                self._plot_sr_levels_bars(ax, sr_analysis, current_price)
                return

            # Plot price line (close prices)
            x_range = range(len(df_plot))
            closes = df_plot["close"].values

            # Main price line
            ax.plot(
                x_range,
                closes,
                color="cyan",
                linewidth=2,
                alpha=0.8,
                label="4H Close",
                zorder=5,
            )

            # Fill between high/low for context
            highs = df_plot["high"].values
            lows = df_plot["low"].values
            ax.fill_between(x_range, lows, highs, color="gray", alpha=0.15, zorder=1)

            # Get price range for visualization
            price_min = df_plot[["high", "low"]].values.min()
            price_max = df_plot[["high", "low"]].values.max()
            price_range = price_max - price_min
            buffer = price_range * 0.1

            # Plot S/R levels as horizontal lines
            for level in levels:
                # Only show levels within visible range
                if price_min - buffer <= level <= price_max + buffer:
                    if level < current_price:
                        # Support
                        ax.axhline(
                            y=level,
                            color="lime",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label=f"Support: ${level:,.2f}",
                            zorder=3,
                        )
                    else:
                        # Resistance
                        ax.axhline(
                            y=level,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label=f"Resistance: ${level:,.2f}",
                            zorder=3,
                        )

            # Mark current price with prominent line
            ax.axhline(
                y=current_price,
                color="white",
                linestyle="-",
                linewidth=2.5,
                alpha=0.9,
                label=f"Current: ${current_price:,.2f}",
                zorder=10,
            )

            # Mark nearest level if available
            if sr_analysis.get("near_sr_level"):
                nearest = sr_analysis.get("nearest_level")
                if nearest and price_min - buffer <= nearest <= price_max + buffer:
                    level_type = sr_analysis.get("level_type", "Level")
                    ax.axhline(
                        y=nearest,
                        color="orange",
                        linestyle=":",
                        linewidth=3,
                        alpha=0.9,
                        label=f"Nearest {level_type}: ${nearest:,.2f}",
                        zorder=8,
                    )

                    # Highlight zone around nearest level
                    zone_size = price_range * 0.01  # 1% zone
                    ax.axhspan(
                        nearest - zone_size,
                        nearest + zone_size,
                        color="orange",
                        alpha=0.15,
                        zorder=2,
                    )

            # Set axis limits
            ax.set_xlim(-1, len(df_plot))
            ax.set_ylim(price_min - buffer, price_max + buffer)

            # Labels and styling
            ax.set_xlabel("4H Candles (Recent 60)", fontsize=10)
            ax.set_ylabel("Price ($)", fontsize=10)
            ax.set_title(
                "4H S/R Analysis - Analyst View", fontsize=11, fontweight="bold"
            )
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, which="both")

            # Add distance info box
            distance_pct = sr_analysis.get("distance_pct", 0)
            if isinstance(distance_pct, (int, float)) and distance_pct > 0:
                level_type = sr_analysis.get("level_type", "Level")
                info_text = (
                    f"Distance to {level_type}: {distance_pct:.2f}%\n"
                    f"Total levels found: {len(levels)}"
                )
                ax.text(
                    0.02,
                    0.98,
                    info_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="black",
                        edgecolor="orange",
                        alpha=0.8,
                        linewidth=2,
                    ),
                )

        except Exception as e:
            logger.error(f"[VIZ] S/R plot error: {e}", exc_info=True)
            ax.text(
                0.5,
                0.5,
                f"Error displaying S/R levels",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_title(
                "4H Support/Resistance Analysis", fontsize=11, fontweight="bold"
            )

    def _plot_sr_levels_bars(self, ax, sr_analysis: Dict, current_price: float):
        """
        Fallback: Plot S/R levels as horizontal bars (original method)
        Used when 4H dataframe is not available
        """
        try:
            levels = sr_analysis.get("levels", [])
            levels_sorted = sorted(levels)

            # Create horizontal bar chart
            y_positions = range(len(levels_sorted))
            colors = []

            for level in levels_sorted:
                if level < current_price:
                    colors.append("lime")  # Support
                else:
                    colors.append("red")  # Resistance

            ax.barh(
                y_positions, levels_sorted, color=colors, alpha=0.7, edgecolor="white"
            )

            # Mark current price
            ax.axvline(
                x=current_price,
                color="white",
                linestyle="--",
                linewidth=2,
                label=f"Current: ${current_price:,.2f}",
            )

            # Mark nearest level
            if sr_analysis.get("near_sr_level"):
                nearest = sr_analysis.get("nearest_level")
                if nearest:
                    ax.axvline(
                        x=nearest,
                        color="orange",
                        linestyle="--",
                        linewidth=2,
                        label=f"Nearest: ${nearest:,.2f}",
                    )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"L{i+1}" for i in y_positions])
            ax.set_xlabel("Price Level ($)", fontsize=10)
            ax.set_title("4H S/R Levels (Bar View)", fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")

        except Exception as e:
            logger.error(f"[VIZ] S/R bar plot error: {e}")

    def _plot_pattern_analysis(self, ax, df: pd.DataFrame, details: Dict):
        """
        Plot pattern detection with defensive checks
        """
        try:
            # ✅ FIX: Defensive extraction
            ai_details = details.get("ai_validation", {})

            # Handle string format
            if isinstance(ai_details, str):
                ax.text(
                    0.5,
                    0.5,
                    f"AI Status: {ai_details}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
                ax.axis("off")
                return

            # Check if dict is valid
            if not ai_details or not isinstance(ai_details, dict):
                ax.text(
                    0.5,
                    0.5,
                    "Pattern Detection: Not Available",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
                ax.axis("off")
                return

            # Check if pattern was detected
            if not ai_details.get("pattern_detected"):
                ax.text(
                    0.5,
                    0.5,
                    "No Pattern Detected",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
                ax.axis("off")
                return

            # Get pattern info
            pattern_name = ai_details.get("pattern_name", "Unknown")
            confidence = ai_details.get("pattern_confidence", 0)
            top3_patterns = ai_details.get("top3_patterns", [])
            top3_confidences = ai_details.get("top3_confidences", [])

            # Create bar chart of top 3 patterns
            if top3_patterns and len(top3_patterns) > 0:
                y_pos = range(len(top3_patterns))
                colors = [
                    "lime" if i == 0 else "orange" if i == 1 else "yellow"
                    for i in range(len(top3_patterns))
                ]

                ax.barh(
                    y_pos, top3_confidences, color=colors, alpha=0.7, edgecolor="white"
                )
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top3_patterns)
                ax.set_xlabel("Confidence", fontsize=10)
                ax.set_xlim(0, 1)

                # Add confidence values
                for i, conf in enumerate(top3_confidences):
                    ax.text(conf + 0.02, i, f"{conf:.1%}", va="center", fontsize=9)
            else:
                # Fallback: show single pattern
                ax.text(
                    0.5,
                    0.5,
                    f"{pattern_name}\nConfidence: {confidence:.1%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            ax.set_title(
                f"Pattern: {pattern_name} ({confidence:.1%})",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="x")

            # Add validation status
            validation_passed = ai_details.get("validation_passed", False)
            validation_status = (
                "[OK] VALIDATED" if validation_passed else "[X] REJECTED"
            )
            status_color = "lime" if validation_passed else "red"

            ax.text(
                0.98,
                0.02,
                validation_status,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=status_color,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

        except Exception as e:
            logger.error(f"[VIZ] Pattern plot error: {e}", exc_info=True)
            ax.text(
                0.5,
                0.5,
                "Error displaying pattern",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_title("AI Pattern Detection", fontsize=11, fontweight="bold")
            ax.axis("off")

    def _plot_signal_breakdown(self, ax, details: Dict):
        """Plot signal aggregation breakdown"""
        try:
            # Get individual strategy signals
            mr_signal = details.get("mr_signal", 0)
            mr_conf = details.get("mr_confidence", 0)
            tf_signal = details.get("tf_signal", 0)
            tf_conf = details.get("tf_confidence", 0)
            ema_signal = details.get("ema_signal", 0)
            ema_conf = details.get("ema_confidence", 0)

            # Create stacked bar chart
            strategies = ["Mean\nReversion", "Trend\nFollowing", "EMA\nRegime"]
            signals = [mr_signal, tf_signal, ema_signal]
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

            # Add signal labels on bars
            for i, (bar, sig, conf) in enumerate(zip(bars, signals, confidences)):
                signal_text = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    conf / 2,
                    f"{signal_text}\n{conf:.2%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(strategies, fontsize=9)
            ax.set_ylabel("Confidence", fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title("Individual Strategy Signals", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            # Add final aggregated signal
            final_quality = details.get("signal_quality", 0)
            regime = details.get("regime", "UNKNOWN")

            ax.text(
                0.02,
                0.98,
                f"Final Quality: {final_quality:.1%}\nRegime: {regime}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            )

        except Exception as e:
            logger.error(f"[VIZ] Signal breakdown error: {e}", exc_info=True)
            ax.text(0.5, 0.5, f"Breakdown Error: {str(e)}", ha="center", va="center")

    def _plot_ai_validation(self, ax, details: Dict):
        """
        FIXED: AI validation summary with comprehensive data display
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

            # S/R Analysis Section
            sr_analysis = ai_details.get("sr_analysis", {})
            if sr_analysis and sr_analysis.get("near_sr_level"):
                level_type = sr_analysis.get("level_type", "Level").upper()
                nearest = sr_analysis.get("nearest_level")
                distance = sr_analysis.get("distance_pct")
                total_levels = sr_analysis.get("total_levels_found", 0)

                summary_lines.append(f"[S/R ANALYSIS]")
                summary_lines.append(f"  Near {level_type}:  ${nearest:,.2f}")
                if distance is not None:
                    summary_lines.append(f"  Distance:   {distance:.2f}%")
                summary_lines.append(f"  Total Levels: {total_levels}")
            else:
                summary_lines.append(f"[S/R ANALYSIS]")
                summary_lines.append(f"  Status: No nearby levels")
                total_levels = sr_analysis.get("total_levels_found", 0)
                if total_levels > 0:
                    summary_lines.append(f"  ({total_levels} levels found)")

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
            else:
                summary_lines.append(f"[RESULT: REJECTED]")
                reasons = ai_details.get("rejection_reasons", [])
                if reasons:
                    summary_lines.append(f"  Reasons:")
                    for reason in reasons[:3]:  # Show max 3
                        summary_lines.append(f"    - {reason}")
                else:
                    summary_lines.append(f"  No specific reasons logged")

            # Check for errors
            if ai_details.get("error"):
                summary_lines.append(f"\n[ERROR]")
                summary_lines.append(f"  {ai_details['error'][:40]}")

            # Display the summary
            summary_text = "\n".join(summary_lines)

            # Color code border based on result
            if validation_passed:
                if "bypass" in action.lower():
                    border_color = "yellow"
                else:
                    border_color = "lime"
            else:
                border_color = "red"

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
            df_15min: 15-minute data
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

            # Prepare caption (ASCII symbols)
            signal_type = (
                "[+] BUY" if signal == 1 else "[-] SELL" if signal == -1 else "[=] HOLD"
            )
            quality = details.get("signal_quality", 0)
            regime = details.get("regime", "UNKNOWN")
            reasoning = details.get("reasoning", "N/A")

            caption = (
                f"*{asset_name} AI Decision Chart*\n\n"
                f"Signal: {signal_type}\n"
                f"Quality: {quality:.1%}\n"
                f"Regime: {regime}\n"
                f"Reasoning: {reasoning[:100]}\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Send to all admins
            for admin_id in self.telegram_bot.admin_ids:
                try:
                    await self.telegram_bot.application.bot.send_photo(
                        chat_id=admin_id,
                        photo=chart_buffer,
                        caption=caption,
                        parse_mode="Markdown",
                    )
                    logger.info(f"[TELEGRAM] Chart sent to admin {admin_id}")
                except Exception as e:
                    logger.error(f"[TELEGRAM] Failed to send to {admin_id}: {e}")

            # Reset buffer for next use
            chart_buffer.seek(0)

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
