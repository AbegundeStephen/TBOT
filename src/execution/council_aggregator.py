"""
Institutional Council Aggregator - Bidirectional Version
Supports both BUY and SELL signals with symmetric logic
✨ ENHANCED: Integrated with World-Class Asymmetric Hedging Filters (1D Governor, Volatility Gate, Sniper Lock).
"""

import pandas as pd
import numpy as np
import talib as ta  # ✨ Added for Volatility/ATR checks
import logging
from typing import Dict, Tuple, Optional
from collections import deque
from datetime import datetime
from src.utils.trap_filter import validate_candle_structure
from src.indicators.divergence import RSIDivergenceDetector
from src.analysis.break_retest import BreakRetestValidator
from src.execution.transition_detector import TransitionDetector
from src.strategies.trend_following import compute_adx_slope

logger = logging.getLogger(__name__)


class InstitutionalCouncilAggregator:
    """
    "BlackRock-Style" Weighted Council with Bidirectional Signals

    Council Members (Judges):
    1. TREND (1.5 pts)     - The Boss: EMA alignment
    2. STRUCTURE (1.5 pts) - The Location: S/R + AI pivots
    3. MOMENTUM (1.0 pt)   - The Fuel: RSI + MACD
    4. PATTERN (0.5 pt)    - The Trigger: AI candlestick patterns
    5. VOLUME (0.5 pt)     - The Validator: Volume confirmation

    Total: 5.0 points
    Trade Threshold: 3.0 / 5.0 (60%)

    Regime Rules:
    - Trend-aligned: Need 3.0+ (simple majority)
    - Counter-trend: Need 3.5+ (unanimous overrule)

    NEW: Symmetric scoring for both BUY and SELL signals
    ✨ NEW: Asymmetric Output (TREND vs SCALP) based on MTF Governor
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        asset_type: str = "BTC",
        ai_validator=None,
        enable_detailed_logging: bool = False,
        # Council thresholds
        trend_aligned_threshold: float = 3.0,
        counter_trend_threshold: float = 3.5,
        # Judge weights (must sum to 5.0)
        weight_trend: float = 1.5,
        weight_structure: float = 1.0,
        weight_momentum: float = 1.5,
        weight_pattern: float = 0.5,
        weight_volume: float = 0.5,
        # Asset-specific tuning
        config: Optional[Dict] = None,
        mtf_integration=None,  # ✨ INJECTED: The Governor
        performance_tracker=None,  # ✨ INJECTED: Performance Analytics
        use_macro_governor: bool = True,
        use_gatekeeper: bool = True,
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()
        self.ai_validator = ai_validator
        self.detailed_logging = enable_detailed_logging
        self.mtf_integration = mtf_integration
        self.performance_tracker = performance_tracker
        # L7: telemetry tag for the most recent TREND judge / Livermore agreement
        # check — read by funnel/shadow logging, not behavior-critical itself.
        self._last_trend_judge_tag: str = "LSM_UNAVAILABLE"
        # L9: telemetry tag for the lifecycle-phase gate's Livermore overlay —
        # read by funnel/shadow logging, not behavior-critical itself.
        self._last_lifecycle_tag: str = "LSM_LIFECYCLE_UNAVAILABLE"
        self.use_macro_governor = use_macro_governor
        self.use_gatekeeper = use_gatekeeper

        # Configuration merge
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # ✨ NEW: World-Class Filter Thresholds (Symmetric Logic)
        self.filter_thresholds = {
            "min_sniper_conf": self.config.get("ai", {}).get(
                "min_sniper_confidence", 0.65
            ),
        }

        # Dynamic threshold loading
        self.trend_aligned_threshold = self.config.get(
            "trend_aligned_threshold",
            self.config.get("council_trend_aligned", trend_aligned_threshold),
        )
        self.counter_trend_threshold = self.config.get(
            "counter_trend_threshold",
            self.config.get("council_counter_trend", counter_trend_threshold),
        )

        # Weights
        self.w_trend = weight_trend
        self.w_structure = weight_structure
        self.w_momentum = weight_momentum
        self.w_pattern = weight_pattern
        self.w_volume = weight_volume

        # Validate weights sum to 5.0
        total_weight = sum(
            [
                self.w_trend,
                self.w_structure,
                self.w_momentum,
                self.w_pattern,
                self.w_volume,
            ]
        )
        if abs(total_weight - 5.0) > 0.01:
            logger.warning(f"[COUNCIL] Weights sum to {total_weight:.2f}, not 5.0")

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "trend_aligned_buys": 0,
            "trend_aligned_sells": 0,
            "counter_trend_buys": 0,
            "counter_trend_sells": 0,
            "avg_score_on_trade": [],
            "avg_score_on_hold": [],
        }

        # Decision history
        self.decision_history = deque(maxlen=100)

        # T4.0: Score trajectory history — per-judge, per-side RAW scores
        # (captured before the trajectory modifier itself runs) from each
        # cycle, so the next cycle can tell a judge that is "forming"
        # (rising) from one "losing steam" (falling) instead of scoring
        # blind off a single-bar snapshot every time.
        self.score_history = deque(maxlen=20)

        # Regime tracking
        self.previous_regime = None
        self.regime_initialized = False

        # T1.5: Stale price detection — tracks (last_price, last_change_time) per asset
        self._last_prices = {}
        self._stale_threshold_minutes = 30

        # T3.4: Economic calendar — loaded at startup, hot-reloadable
        self._econ_cal_path = "config/economic_calendar.json"
        self._econ_events = []
        self._load_calendar_file()

        # ✨ NEW: Advanced Confluence Engines
        self.divergence_detector = RSIDivergenceDetector(pivot_window=5)
        self.break_retest_validator = BreakRetestValidator(lookback=50)

        # Transition evidence — same detector used by Performance's gatekeeper
        self._transition_detector = TransitionDetector()

        self._log_initialization()

    def _get_default_config(self) -> Dict:
        """
        Asset-specific configurations.

        RSI zone logic:
          bullish_zone = (low, high) — RSI must be IN this range to score bullish momentum
          bearish_zone = (low, high) — RSI must be IN this range to score bearish momentum
          oversold_bonus  — RSI below this level gets a mean-reversion BUY bonus
          overbought_bonus — RSI above this level gets a mean-reversion SELL bonus

        Asset classes:
          BTC          — Crypto: high volatility, strong trends, wide RSI range
          GOLD/XAUUSD  — Commodity: moderate volatility, mix of trend and range
          USTEC/US100  — Index: high momentum, RSI stays elevated longer in trends
          USOIL        — Oil: commodity, wide swings, mean-reverting within trends
          FX majors    — EURUSD, GBPUSD, USDJPY: lower volatility, tighter RSI ranges
          FX crosses   — GBPAUD, EURJPY: higher volatility, wider RSI tolerance
        """
        a = self.asset_type.upper()

        # ── Crypto ────────────────────────────────────────────────────────────
        if "BTC" in a:
            return {
                "rsi_bullish_zone": (40, 65),
                "rsi_bearish_zone": (35, 60),
                "rsi_oversold_bonus": 30,
                "rsi_overbought_bonus": 70,
                "volume_ma_period": 20,
                "pattern_confidence_min": 0.65,
                "macd_confirmation": True,
                "trend_safety_threshold": 0.50,
            }

        # ── Gold / Precious Metals ─────────────────────────────────────────────
        if "GOLD" in a or "XAU" in a:
            return {
                "rsi_bullish_zone": (35, 60),
                "rsi_bearish_zone": (40, 65),
                "rsi_oversold_bonus": 25,
                "rsi_overbought_bonus": 75,
                "volume_ma_period": 20,
                "pattern_confidence_min": 0.65,
                "macd_confirmation": True,
                "trend_safety_threshold": 0.50,
            }

        # ── Tech / Equity Indices ──────────────────────────────────────────────
        # USTEC / US100 / NAS100 trend hard — RSI stays elevated; use wider zones
        # and a higher overbought threshold so bull runs aren't prematurely blocked.
        if any(x in a for x in ("USTEC", "US100", "NAS", "US30", "SPX")):
            return {
                "rsi_bullish_zone": (45, 70),
                "rsi_bearish_zone": (30, 55),
                "rsi_oversold_bonus": 30,
                "rsi_overbought_bonus": 75,
                "volume_ma_period": 20,
                "pattern_confidence_min": 0.65,
                "macd_confirmation": True,
                "trend_safety_threshold": 0.50,
            }

        # ── Oil / Energy Commodities ───────────────────────────────────────────
        # USOIL swings wide; symmetric RSI zones, slightly relaxed extremes.
        if "OIL" in a or "WTI" in a or "BRENT" in a:
            return {
                "rsi_bullish_zone": (35, 65),
                "rsi_bearish_zone": (35, 65),
                "rsi_oversold_bonus": 28,
                "rsi_overbought_bonus": 72,
                "volume_ma_period": 20,
                "pattern_confidence_min": 0.60,
                "macd_confirmation": True,
                "trend_safety_threshold": 0.50,
            }

        # ── FX Crosses (GBPAUD, EURJPY) ────────────────────────────────────────
        # Cross pairs carry two economies — bigger swings, wider RSI tolerance.
        # Lower pattern_confidence_min because AI patterns are noisier on crosses.
        if any(x in a for x in ("GBPAUD", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY")):
            return {
                "rsi_bullish_zone": (35, 65),
                "rsi_bearish_zone": (35, 65),
                "rsi_oversold_bonus": 25,
                "rsi_overbought_bonus": 75,
                "volume_ma_period": 20,
                "pattern_confidence_min": 0.60,
                "macd_confirmation": True,
                "trend_safety_threshold": 0.55,
            }

        # ── FX Majors (EURUSD, GBPUSD, USDJPY, etc.) ─────────────────────────
        # Major pairs are more mean-reverting with tighter RSI ranges.
        # Higher trend_safety_threshold — TF must be more confident to veto.
        return {
            "rsi_bullish_zone": (40, 62),
            "rsi_bearish_zone": (38, 60),
            "rsi_oversold_bonus": 30,
            "rsi_overbought_bonus": 70,
            "volume_ma_period": 20,
            "pattern_confidence_min": 0.65,
            "macd_confirmation": True,
            "trend_safety_threshold": 0.55,
        }

    def _log_initialization(self):
        """Log startup configuration"""
        logger.info("=" * 80)
        logger.info(f"🏛️  INSTITUTIONAL COUNCIL AGGREGATOR - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("")
        logger.info("   COUNCIL MEMBERS (Judges):")
        logger.info(f"   1. TREND      ({self.w_trend:.1f} pts) - EMA alignment")
        logger.info(f"   2. STRUCTURE  ({self.w_structure:.1f} pts) - S/R + AI pivots")
        logger.info(f"   3. MOMENTUM   ({self.w_momentum:.1f} pt)  - RSI + MACD")
        logger.info(f"   4. PATTERN    ({self.w_pattern:.1f} pt)  - AI candlesticks")
        logger.info(f"   5. VOLUME     ({self.w_volume:.1f} pt)  - Volume confirmation")
        logger.info("")
        logger.info("   DECISION RULES (Bidirectional):")
        logger.info(f"   • Trend-aligned:  ≥ {self.trend_aligned_threshold:.1f} / 5.0")
        logger.info(f"   • Counter-trend:  ≥ {self.counter_trend_threshold:.1f} / 5.0")
        logger.info("")
        logger.info(
            f"   AI Validation: {'ENABLED' if self.ai_validator else 'DISABLED'}"
        )
        logger.info(
            f"   Governor MTF:  {'ENABLED' if self.mtf_integration else 'DISABLED'}"
        )
        logger.info("=" * 80)
        logger.info("")

    # ========================================================================
    # T3.4: Economic Calendar helpers
    # ========================================================================

    def _load_calendar_file(self):
        """Load economic events from the JSON calendar file."""
        try:
            import json as _json

            with open(self._econ_cal_path, encoding="utf-8") as _f:
                self._econ_events = _json.load(_f).get("events", [])
            logger.info(
                f"[CALENDAR] Loaded {len(self._econ_events)} events from {self._econ_cal_path}"
            )
        except Exception as _e:
            logger.warning(f"[CALENDAR] Could not load {self._econ_cal_path}: {_e}")
            self._econ_events = []

    def reload_calendar(self):
        """Hot-reload the economic calendar (called by CalendarUpdater after each write)."""
        self._load_calendar_file()
        logger.info(
            f"[CALENDAR] 🔄 Hot-reloaded — {len(self._econ_events)} active events in memory"
        )

    # ========================================================================
    # ✨ WORLD-CLASS FILTERS (Asymmetric Logic)
    # ========================================================================

    def _check_governor_filter(
        self,
        df: pd.DataFrame,
        signal: int,
        governor_data: Optional[Dict] = None,
        preset_trade_type: str = "TREND",
    ) -> Tuple[bool, str]:
        """
        Check the 1D Macro Trend via pre-injected Governor data.
        ✅ INSTITUTIONAL: Strict TREND enforcement. Supports REVERSION gating.
        """
        # 1. FAIL-SAFE: If no Governor data, return NO TRADE (Strict macro dependency)
        if not governor_data:
            logger.warning(
                "[GOV] ❌ BLOCKED - No MTF Governor data available. Blocking trade (Strict Macro Rule)."
            )
            return False, "NEUTRAL"

        governor = governor_data.get("governor") or governor_data.get(
            "full_regime_status"
        )

        if not governor:
            logger.warning(
                "[GOV] ❌ BLOCKED - No Governor status object found in data. Blocking trade."
            )
            return False, "NEUTRAL"

        try:
            # Extract regime context
            regime_name = getattr(
                governor, "consensus_regime", governor_data.get("regime", "NEUTRAL")
            )
            is_bullish = getattr(
                governor, "is_bullish", governor_data.get("is_bullish", False)
            )
            is_bearish = getattr(
                governor, "is_bearish", governor_data.get("is_bearish", False)
            )

            # 2. MRS §6 Phase 0 — TRANSITION path permanently removed.
            # NEUTRAL regime passes through at normal scoring. Structural gating
            # is handled by the Livermore Hard Veto (is_silent_zone) which blocks
            # entries in NATURAL_RETRACEMENT / NATURAL_REBOUND states — the only
            # states where NEUTRAL regime entries were genuinely dangerous.
            # Raising required_score +0.75 for MTF NEUTRAL was the single largest
            # source of rejected valid setups in the pre-v3 system.
            if regime_name == "NEUTRAL" and preset_trade_type == "TREND":
                logger.debug(
                    f"[GOV] NEUTRAL regime — passing at standard score threshold."
                )
                return True, "TREND"

            # 3. ASSET-DNA Gating & Trade Alignment
            asset = self.asset_type.upper()

            if preset_trade_type == "REVERSION":
                # --- REVERSION GATING (DNA) ---
                if "BTC" in asset or "USTEC" in asset:
                    # Block MR during BULLISH or SLIGHTLY_BULLISH
                    if regime_name in ["BULLISH", "SLIGHTLY_BULLISH"]:
                        logger.info(
                            f"[GOV] ❌ BLOCKED - MR forbidden in {regime_name} regime for {asset}"
                        )
                        return False, "REVERSION"
                    # Allow MR Buys only in BEARISH or NEUTRAL
                    if signal == -1:  # MR Short
                        logger.info(
                            f"[GOV] ❌ BLOCKED - MR Shorts forbidden for {asset}"
                        )
                        return False, "REVERSION"
                    # Buys in BEARISH or NEUTRAL are allowed
                    return True, "REVERSION"

                elif "GOLD" in asset:
                    # Allow MR Buys in BEARISH
                    if signal == 1:
                        if regime_name == "BEARISH":
                            return True, "REVERSION"
                        else:
                            logger.info(
                                f"[GOV] ❌ BLOCKED - MR Buys only allowed in BEARISH for {asset} (Current: {regime_name})"
                            )
                            return False, "REVERSION"
                    # Block MR Shorts in BULLISH
                    elif signal == -1:
                        if regime_name == "BULLISH":
                            logger.info(
                                f"[GOV] ❌ BLOCKED - MR Shorts forbidden in BULLISH for {asset}"
                            )
                            return False, "REVERSION"
                        else:
                            # Implies allowed in BEARISH or NEUTRAL
                            return True, "REVERSION"

                # EURUSD / EURJPY allow symmetric MR (no extra blocks here)
                return True, "REVERSION"

            else:
                # --- TREND GATING (STRICT) ---
                if (is_bullish and signal == -1) or (is_bearish and signal == 1):
                    # ✨ Explosive Momentum Overrule (V-Shape Reversal)
                    if self._is_explosive_momentum(df, signal):
                        logger.info(
                            f"[GOV] 🚀 EXPLOSIVE MOMENTUM - Overruling Macro Veto ({regime_name})"
                        )
                        return True, "V_SHAPE"

                    # ✨ SLIGHTLY regimes are ambiguous — soft block instead of hard block.
                    # Returns SLIGHTLY_COUNTER so the caller raises required_score by +0.5
                    # (needs ≥4.0) rather than rejecting outright.  Full BEARISH/BULLISH
                    # counter-trend signals are still hard-blocked below.
                    if regime_name in ("SLIGHTLY_BEARISH", "SLIGHTLY_BULLISH"):
                        direction = "Short" if signal == -1 else "Long"
                        logger.info(
                            f"[GOV] ⚠️ SLIGHTLY_COUNTER — {direction} in {regime_name}: "
                            f"allowing at raised score threshold (+0.5)"
                        )
                        return True, "SLIGHTLY_COUNTER"

                    logger.info(
                        f"[GOV] ❌ BLOCKED - {('Short' if signal == -1 else 'Long')} attempt in Macro {('BULLISH' if is_bullish else 'BEARISH')} regime ({regime_name})"
                    )
                    return False, "TREND"

                return True, "TREND"

        except Exception as e:
            logger.error(f"[GOV] Error processing Governor data: {e}", exc_info=True)
            return False, "NEUTRAL"

    def _check_volatility_gate_adaptive(
        self, df: pd.DataFrame, atr_fast: float, atr_slow: float
    ) -> bool:
        """Blocks trades in dead markets (atr_fast < 0.5 * atr_slow)."""
        try:
            # Coiled Spring Tracker
            if len(df) >= 30:
                high, low, close = (
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                )
                atr_f_series = ta.ATR(high, low, close, timeperiod=14)
                atr_s_series = ta.ATR(high, low, close, timeperiod=100)

                atr_ratio_series = atr_f_series / atr_s_series

                # Check last 20 bars for extreme compression
                if np.max(atr_ratio_series[-20:]) < 0.65:
                    logger.info(
                        "[VOLATILITY] Coiled Spring Detected - Breakout readiness high"
                    )
                    return True

            if atr_fast < (0.5 * atr_slow):
                logger.info(
                    f"[VOLATILITY] ❌ BLOCKED - Dead Market (ATR Fast: {atr_fast:.4f} < 0.5 * ATR Slow: {atr_slow:.4f})"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"[VOLATILITY] Error: {e}")
            return True

    def _check_sniper_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, Dict]:
        """
        Hybrid Confirmation: AI Pattern OR Momentum Impulse.
        ✅ INSTITUTIONAL UPGRADE: Mandatory Displacement Fork (Binance vs Exness).
        """
        try:
            latest = df.iloc[-1]
            reasons = []

            # ================================================================
            # 0. Institutional Displacement Fork (MANDATORY)
            # ================================================================
            # Reason: Proves institutional conviction vs broker tick noise.
            body = abs(latest["close"] - latest["open"])
            high, low, close_vals = (
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            atr_fast = ta.ATR(high, low, close_vals, timeperiod=14)[-1]
            volume_rolling_avg = (
                df["volume"].iloc[-21:-1].mean() if "volume" in df.columns else 0
            )

            displacement_passed = False
            displacement_reason = ""
            hard_blocked = False

            # --- Staircase Bypass ---
            # Compute NET displacement across last three candles (A -> C)
            # Reason: Proves sustained directional move rather than choppy whipsaw.
            net_displacement = df["close"].iloc[-1] - df["open"].iloc[-3]

            # Check if net move is in signal direction and size is significant
            if signal == 1 and net_displacement > (1.2 * atr_fast):
                displacement_passed = True
                displacement_reason = f"Staircase Bypass: 3-bar net UP displacement {net_displacement:.2f} > 1.2 ATR"
            elif signal == -1 and net_displacement < -(1.2 * atr_fast):
                displacement_passed = True
                displacement_reason = f"Staircase Bypass: 3-bar net DOWN displacement {abs(net_displacement):.2f} > 1.2 ATR"

            # --- ✅ News Exception FIX (T11) ---
            # If candle size is extreme, reject unless institutional volume confirms
            if body > (2.5 * atr_fast):
                volume_average = volume_rolling_avg
                volume = latest.get("volume", 0)
                if volume > (4.5 * volume_average) and volume_average > 0:
                    displacement_passed = True
                    displacement_reason = (
                        "News Exception: Institutional volume confirmed huge candle."
                    )
                else:
                    hard_blocked = True  # 🚨 SET HARD BLOCK
                    displacement_passed = False
                    displacement_reason = (
                        "News Exception: Huge candle without institutional volume."
                    )

            # --- Coiled Spring Detection ---
            # Measure volatility compression
            high_arr, low_arr, close_arr = (
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            atr_fast_series = ta.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            atr_slow_series = ta.ATR(high_arr, low_arr, close_arr, timeperiod=100)
            atr_ratio_series = pd.Series(atr_fast_series / atr_slow_series)

            conviction_score = 0.0
            if atr_ratio_series.iloc[-20:].max() < 0.65:
                conviction_score += 1.0
                logger.info(
                    f"[SNIPER] 🌀 Coiled Spring detected: Compression < 0.65. Conviction +1.0"
                )

            if conviction_score >= 1.0:
                displacement_passed = True  # Override for coiled spring breakout

            # If none of the institutional rules passed, fallback to standard momentum
            # ✅ FIX: Added 'not hard_blocked' guard
            if not displacement_passed and not hard_blocked:
                candle_range = latest.get("high", 0) - latest.get("low", 0)
                if body > (0.5 * atr_fast) or candle_range > (1.0 * atr_fast):
                    displacement_passed = True
                else:
                    displacement_reason = (
                        "Standard: Displacement < 0.5 ATR and range < 1.0 ATR"
                    )

            if not displacement_passed:
                if self.detailed_logging:
                    logger.info(f"[SNIPER] ❌ BLOCKED - {displacement_reason}")
                return False, {
                    "trigger_type": "DISPLACEMENT",
                    "reason": displacement_reason,
                }

            # ================================================================
            # 1. AI Pattern Confidence — PAT-3: removed (pattern module disabled)
            # ================================================================

            # ================================================================
            # 2. Institutional Displacement Confirmation
            # ================================================================
            # Reason: Signal already passed mandatory displacement fork.
            # We record it here as a confirmed trigger for the audit trail.
            if "BTC" in self.asset_type:
                reasons.append(
                    {
                        "passed": True,
                        "trigger_type": "VOLUME_SURGE_INSTITUTIONAL",
                        "volume": latest.get("volume", 0),
                        "surge_factor": latest.get("volume", 0) / volume_rolling_avg,
                    }
                )
            else:
                reasons.append(
                    {
                        "passed": True,
                        "trigger_type": "MOMENTUM_DISPLACEMENT_INSTITUTIONAL",
                        "body": body,
                        "atr_multiplier": body / atr_fast if atr_fast > 0 else 0,
                    }
                )

            # Check if we have enough data for rolling indicators (Donchian, Bollinger Bands)
            # Need 20 periods + current, so at least 21 bars
            if len(df) < 21:
                if reasons:
                    if self.detailed_logging:
                        logger.info(
                            f"[SNIPER] ✅ PASSED - Trigger(s): {[r['trigger_type'] for r in reasons]} (Partial checks due to insufficient data)"
                        )
                    return True, reasons[0]
                else:
                    logger.warning(
                        f"[SNIPER] ❌ BLOCKED - Insufficient data for full institutional checks (need 21 bars, have {len(df)})."
                    )
                    return False, {
                        "trigger_type": None,
                        "reason": f"Insufficient data for full checks (have {len(df)})",
                    }

            # ================================================================
            # 3. Turtle Breakout (20-period Donchian Channel)
            # ================================================================
            # Reason: Detects that price has moved into a new volatility regime.
            close_rolling_mean = df["close"].iloc[-21:-1].mean()
            close_rolling_std = df["close"].iloc[-21:-1].std()

            if close_rolling_std > 0:
                upper_band = close_rolling_mean + (2.0 * close_rolling_std)
                lower_band = close_rolling_mean - (2.0 * close_rolling_std)

                if signal == 1 and latest["close"] > upper_band:
                    reasons.append(
                        {
                            "passed": True,
                            "trigger_type": "VOLATILITY_BREACH",
                            "band": "upper",
                            "price": latest["close"],
                        }
                    )
                elif signal == -1 and latest["close"] < lower_band:
                    reasons.append(
                        {
                            "passed": True,
                            "trigger_type": "VOLATILITY_BREACH",
                            "band": "lower",
                            "price": latest["close"],
                        }
                    )

            # ================================================================
            # Final Decision
            # ================================================================
            if reasons:
                # Log all triggers that passed
                trigger_types = [r["trigger_type"] for r in reasons]
                logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {trigger_types}")
                # Return the details of the first trigger found
                return True, reasons[0]

            logger.info(f"[SNIPER] ❌ BLOCKED - No institutional edge confirmed.")
            return False, {
                "trigger_type": None,
                "reason": "No confirmation criteria met",
            }

        except Exception as e:
            logger.error(
                f"[SNIPER] Error in institutional edge check: {e}", exc_info=True
            )
            # Fail-open: If the filter fails, we allow the trade to avoid blocking valid signals due to code errors.
            return True, {"trigger_type": "ERROR_FALLBACK", "reason": str(e)}

    def _check_profit_economics_adaptive(
        self,
        entry: float,
        stop_loss: float,
        atr_fast: float,
        first_tp_mult: float = 1.5,
    ) -> bool:
        """
        The 'Worth It' Check. Validates if potential RR covers fees using ATR scaling.
        ✅ FIXED: Corrected mathematical impossibility (1.5 < 0.5)
        """
        try:
            risk = abs(entry - stop_loss)
            if risk <= 0:
                return True

            expected_reward = risk * first_tp_mult
            min_required = 0.5 * atr_fast

            if expected_reward < min_required:
                logger.info(
                    f"[PROFIT GATE] ❌ Blocked - Low Reward (reward {expected_reward:.4f} < {min_required:.4f})"
                )
                return False

            # Minimum 1.2:1 R:R check
            return (expected_reward / risk) >= 1.2

        except Exception as e:
            logger.error(f"[PROFIT] Error: {e}")
            return True

    def _check_recent_momentum_alignment(
        self,
        df: pd.DataFrame,
        signal: int,
        atr_fast: float,
        n_candles: int = 3,
        min_agreement: int = 2,
    ) -> Tuple[bool, str]:
        """
        Candle Momentum Alignment Gate.

        Blocks a trend-aligned signal when the most recent closed 1H candles
        show clear momentum in the *opposite* direction — i.e. price is actively
        bouncing against the proposed trade.

        Examples of what this catches:
          • SELL proposed in a BEARISH regime but the last 3 candles are all
            bullish (price rallying) — don't add short exposure into a bounce.
          • BUY proposed in a BULLISH regime but the last 3 candles are all
            bearish (price pulling back hard) — don't add long exposure into a
            sharp sell-off.

        Returns:
            (passed: bool, reason: str)
            passed=True  → gate cleared, signal may proceed
            passed=False → gate failed, signal should be blocked
        """
        try:
            if len(df) < n_candles + 2:
                return True, "insufficient data"

            # Use rows [-n_candles-1 : -1] to get n_candles confirmed-closed
            # bars (exclude the most-recent potentially-forming candle).
            _recent = df.iloc[-(n_candles + 1) : -1]
            _opens = _recent["open"].values
            _closes = _recent["close"].values

            _bullish = int((_closes > _opens).sum())
            _bearish = int((_closes < _opens).sum())

            # Require that candle bodies are meaningful (not tiny dojis).
            _avg_body = float(abs(_closes - _opens).mean())
            _min_body = atr_fast * 0.08  # 8% of ATR — filters out doji noise

            if _avg_body < _min_body:
                return True, "candle bodies too small to determine momentum"

            if signal == -1 and _bullish >= min_agreement:
                reason = (
                    f"{_bullish}/{n_candles} recent candles are bullish "
                    f"(avg body {_avg_body:.4f}) — momentum opposing SELL"
                )
                logger.info(
                    f"[CMR] ⛔ SELL blocked — recent {_bullish}/{n_candles} "
                    f"candles bullish, market bouncing against short entry."
                )
                return False, reason

            if signal == 1 and _bearish >= min_agreement:
                reason = (
                    f"{_bearish}/{n_candles} recent candles are bearish "
                    f"(avg body {_avg_body:.4f}) — momentum opposing BUY"
                )
                logger.info(
                    f"[CMR] ⛔ BUY blocked — recent {_bearish}/{n_candles} "
                    f"candles bearish, market falling against long entry."
                )
                return False, reason

            return True, "momentum aligned"

        except Exception as e:
            logger.warning(f"[CMR] Check failed, allowing signal: {e}")
            return True, f"check error: {e}"

    def _check_macro_regime(self, asset: str) -> str:
        """
        Extract macro regime from MTF integration or current state.
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if self.mtf_integration and hasattr(
            self.mtf_integration, "_current_regime_data"
        ):
            regime_data = self.mtf_integration._current_regime_data.get(asset.upper())
            if regime_data:
                governor = regime_data.get("governor") or regime_data.get(
                    "full_regime_status"
                )
                if governor:
                    if getattr(governor, "is_bullish", False):
                        return "BULLISH"
                    if getattr(governor, "is_bearish", False):
                        return "BEARISH"
        return "NEUTRAL"

    def _check_lifecycle_phase(
        self,
        df: pd.DataFrame,
        signal: int,
        adx: float,
        current_required_score: float,
        governor_data: Optional[Dict] = None,
    ) -> Tuple[float, str]:
        """
        Lifecycle / Phase-Awareness Gate
        ─────────────────────────────────
        The Council judges score *current conditions* but have no memory of where
        in the trend cycle the market is.  This method adds that memory.

        Returns (adjusted_required_score, phase_label) where phase_label is one of:
          • "EXHAUSTED"    — trend is old, ADX declining, price overextended + RSI divergence
          • "EXTENDED"     — price stretched but no confirmed divergence yet
          • "ESTABLISHING" — fresh breakout, ADX just crossed 20 and still rising
          • "HEALTHY"      — normal mid-trend, no adjustment

        Score adjustments (never blocks outright):
          EXHAUSTED    → required_score += 0.75
          EXTENDED     → required_score += 0.35
          ESTABLISHING → required_score -= 0.25  (reward fresh momentum)
          HEALTHY      → no change

        L9: on top of this ADX-based read, _apply_lifecycle_livermore_overlay()
        applies a small additive nudge sourced from the *already* Livermore-
        derived composite_state.lifecycle_phase (PICKUP/CONFIRMATION/ESTABLISHED/
        FADING/EXHAUSTION — see signal_aggregator.py's LSM state→phase mapping).
        Gated by phase_config.council_lifecycle_livermore_guard_enabled (default
        False); never changes the returned phase_label, only the score.
        """
        try:
            # C2: extract composite_state so the RSI divergence block below can
            # write back to it. Without this, composite_state is undefined and
            # the divergence write throws a NameError caught silently by the
            # outer except, meaning divergence_detected stays False forever.
            composite_state = (governor_data or {}).get("composite_state")

            if len(df) < 60:
                return current_required_score, "HEALTHY"

            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            # ── 1. ADX SLOPE: is the trend gaining or losing strength? ────────
            adx_series = ta.ADX(high, low, close, timeperiod=14)
            valid_adx = adx_series[~np.isnan(adx_series)]
            if len(valid_adx) < 10:
                return current_required_score, "HEALTHY"

            adx_now = valid_adx[-1]
            adx_5ago = valid_adx[-5]
            adx_peak = float(np.max(valid_adx[-20:]))  # rolling 20-bar peak
            adx_rising = adx_now > adx_5ago  # simple slope check

            # Peak-decline: how far has ADX fallen from its recent high?
            adx_peak_decline_pct = (
                (adx_peak - adx_now) / adx_peak if adx_peak > 0 else 0.0
            )

            # ── 2. PRICE Z-SCORE: how far is price from its 50-bar mean? ─────
            close_series = pd.Series(close)
            mean_50 = close_series.rolling(50).mean().iloc[-1]
            std_50 = close_series.rolling(50).std().iloc[-1]
            z_score = (close[-1] - mean_50) / std_50 if std_50 and std_50 > 0 else 0.0
            # Directional z-score: positive means price extended in the signal direction
            directional_z = z_score if signal == 1 else -z_score

            # ── 3. RSI DIVERGENCE: price made new extreme but RSI didn't ─────
            rsi_series = ta.RSI(close, timeperiod=14)
            valid_rsi = rsi_series[~np.isnan(rsi_series)]
            rsi_divergence = False
            if len(valid_rsi) >= 10:
                rsi_now = valid_rsi[-1]
                rsi_5ago = valid_rsi[-5]
                price_now = close[-1]
                price_5ago = close[-5]
                # Bearish divergence: price higher but RSI lower (exhaustion on longs)
                if signal == 1 and price_now > price_5ago and rsi_now < rsi_5ago - 2:
                    rsi_divergence = True
                # Bullish divergence: price lower but RSI higher (exhaustion on shorts)
                elif signal == -1 and price_now < price_5ago and rsi_now > rsi_5ago + 2:
                    rsi_divergence = True

            # Share divergence findings with CompositeState so MarketWatcher,
            # VTM, and the Confluence scoring block can all benefit from
            # council's RSI analysis instead of each having to recompute it.
            # Guards against composite_state being None (performance-mode calls
            # to this function don't always have it available).
            if rsi_divergence and composite_state is not None:
                try:
                    if hasattr(composite_state, "divergence_detected"):
                        composite_state.divergence_detected = True
                        # Only upgrade strength, never downgrade — performance
                        # mode may have already set a higher value from its
                        # own mean-reversion divergence analysis.
                        existing_strength = getattr(
                            composite_state, "divergence_strength", 0.0
                        ) or 0.0
                        composite_state.divergence_strength = max(existing_strength, 0.5)
                except Exception:
                    pass

            # ── 4. CLASSIFY PHASE ─────────────────────────────────────────────
            # ESTABLISHING: ADX was below 20 recently and is now rising cleanly
            adx_was_low = (
                float(np.min(valid_adx[-8:-3])) < 20
            )  # dipped below 20 in last 3–8 bars
            if adx_was_low and adx_rising and adx_now > 20:
                adj_score = max(current_required_score - 0.25, 2.5)
                logger.info(
                    f"[LIFECYCLE] 🌱 ESTABLISHING — ADX just crossed 20 and rising "
                    f"({adx_now:.1f}). required_score ↓ {current_required_score:.2f} → {adj_score:.2f}"
                )
                return self._apply_lifecycle_livermore_overlay(
                    adj_score, "ESTABLISHING", governor_data
                )

            # EXHAUSTED: ADX declined significantly from peak + price overextended + divergence
            if (
                adx_peak_decline_pct >= 0.15  # ADX fell ≥15% from recent peak
                and directional_z >= 1.8  # price stretched ≥1.8σ in signal dir
                and rsi_divergence
            ):  # confirmed RSI divergence
                adj_score = min(current_required_score + 0.75, 4.75)
                logger.info(
                    f"[LIFECYCLE] 🔴 EXHAUSTED — ADX peak-decline {adx_peak_decline_pct:.0%}, "
                    f"z-score {directional_z:.2f}σ, RSI divergence confirmed. "
                    f"required_score ↑ {current_required_score:.2f} → {adj_score:.2f}"
                )
                return self._apply_lifecycle_livermore_overlay(
                    adj_score, "EXHAUSTED", governor_data
                )

            # EXTENDED: price stretched + ADX declining (but no divergence confirmation yet)
            if adx_peak_decline_pct >= 0.12 and directional_z >= 1.5 and not adx_rising:
                adj_score = min(current_required_score + 0.35, 4.5)
                logger.info(
                    f"[LIFECYCLE] 🟡 EXTENDED — ADX peak-decline {adx_peak_decline_pct:.0%}, "
                    f"z-score {directional_z:.2f}σ (no divergence). "
                    f"required_score ↑ {current_required_score:.2f} → {adj_score:.2f}"
                )
                return self._apply_lifecycle_livermore_overlay(
                    adj_score, "EXTENDED", governor_data
                )

            logger.debug(
                f"[LIFECYCLE] ✅ HEALTHY — ADX={adx_now:.1f} peak-decline={adx_peak_decline_pct:.0%} "
                f"z={directional_z:.2f}σ divergence={rsi_divergence}. No score adjustment."
            )
            return self._apply_lifecycle_livermore_overlay(
                current_required_score, "HEALTHY", governor_data
            )

        except Exception as e:
            logger.debug(f"[LIFECYCLE] Phase check error: {e}")
            return current_required_score, "HEALTHY"

    def _apply_lifecycle_livermore_overlay(
        self,
        adj_score: float,
        phase_label: str,
        governor_data: Optional[Dict] = None,
    ) -> Tuple[float, str]:
        """
        L9: nudges the ADX-based lifecycle required_score using the *already*
        Livermore-derived composite_state.lifecycle_phase (PICKUP/CONFIRMATION/
        ESTABLISHED/FADING/EXHAUSTION — see signal_aggregator.py's LSM
        state-age mapping). Purely additive on top of whatever the ADX-based
        classifier above produced; never changes phase_label so the existing
        4-value contract for downstream consumers (e.g. funnel logging at
        'lifecycle_phase') stays intact. Gated by
        phase_config.council_lifecycle_livermore_guard_enabled (default False).
        """
        self._last_lifecycle_tag = "LSM_LIFECYCLE_UNAVAILABLE"
        try:
            cs = (governor_data or {}).get("composite_state") if governor_data else None
            _phase_cfg = (
                (governor_data or {}).get("phase_config") if governor_data else None
            )
            if _phase_cfg is None:
                _phase_cfg = (
                    (
                        cs.get("phase_config")
                        if isinstance(cs, dict)
                        else getattr(cs, "phase_config", None)
                    )
                    if cs is not None
                    else None
                )
            _phase_cfg = _phase_cfg or {}

            if not _phase_cfg.get("council_lifecycle_livermore_guard_enabled", False):
                self._last_lifecycle_tag = "LSM_LIFECYCLE_DISABLED"
                return adj_score, phase_label

            if cs is None:
                return adj_score, phase_label

            lsm_phase = (
                cs.get("lifecycle_phase")
                if isinstance(cs, dict)
                else getattr(cs, "lifecycle_phase", None)
            )
            if not lsm_phase:
                return adj_score, phase_label

            _phase_adjustments = {
                "PICKUP": -0.20,
                "CONFIRMATION": -0.10,
                "ESTABLISHED": 0.0,
                "FADING": 0.30,
                "EXHAUSTION": 0.60,
            }
            delta = _phase_adjustments.get(lsm_phase)
            if delta is None:
                self._last_lifecycle_tag = f"LSM_LIFECYCLE_UNKNOWN({lsm_phase})"
                return adj_score, phase_label

            new_score = max(2.0, min(adj_score + delta, 5.0))
            if abs(delta) > 1e-9:
                logger.info(
                    f"[LIFECYCLE] 🧭 Livermore overlay: lsm_phase={lsm_phase} adx_phase={phase_label} "
                    f"required_score {adj_score:.2f} → {new_score:.2f} (Δ{delta:+.2f})"
                )
            self._last_lifecycle_tag = f"LSM_LIFECYCLE_{lsm_phase}"
            return new_score, phase_label
        except Exception as e:
            logger.debug(f"[LIFECYCLE] L9 Livermore overlay skipped: {e}")
            return adj_score, phase_label

    def _is_explosive_momentum(self, df: pd.DataFrame, signal: int) -> bool:
        """
        Detects 'V-Shape' or 'Parabolic' price action that overrules macro bias.
        Criteria:
        1. ADX > 22 (Meaningful trend — lowered from 30 to catch geopolitical/supply-shock moves)
        2. Velocity: Last 6 bars move > 1.5 * ATR14 (lowered from 2.0 to be less restrictive)
        3. Alignment: Price > EMA20 > EMA50 (for Longs)
        """
        try:
            if len(df) < 50:
                return False

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # 1. Trend Strength (22 catches strong momentum without requiring parabolic ADX)
            adx = ta.ADX(high, low, close, timeperiod=14)[-1]
            if adx < 22:
                return False

            # 2. ATR-Scaled Velocity (1.5x ATR over last 6 bars)
            atr = ta.ATR(high, low, close, timeperiod=14)[-1]
            move = close[-1] - close[-6]
            velocity_ratio = abs(move) / (atr if atr > 0 else 1)

            if velocity_ratio < 1.5:
                return False

            # 3. Local Alignment
            ema20 = ta.EMA(close, timeperiod=20)[-1]
            ema50 = ta.EMA(close, timeperiod=50)[-1]

            if signal == 1:  # Buying into a bear regime
                if move > 0 and close[-1] > ema20 > ema50:
                    return True
            elif signal == -1:  # Selling into a bull regime
                if move < 0 and close[-1] < ema20 < ema50:
                    return True

            return False
        except Exception as e:
            logger.debug(f"[MOMENTUM] Overrule check error: {e}")
            return False

    def _get_aggregated_signal_impl(
        self,
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",
        is_bull_market: bool = True,
        governor_data: Optional[Dict] = None,
        live_price: Optional[float] = None,
    ) -> Tuple[int, Dict]:
        """
        Main council decision logic with bidirectional support
        ✅ INSTITUTIONAL PHASE 4: Dynamic Weights & Penalty Shift
        """
        self.stats["total_evaluations"] += 1
        timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

        # AI-5: Clear pattern cache at the start of each cycle so all internal
        # calls to _check_pattern() within this evaluation share the first result.
        if self.ai_validator and hasattr(self.ai_validator, "clear_pattern_cache"):
            self.ai_validator.clear_pattern_cache()

        # ════════════════════════════════════════════════════════════════════
        # T1.5: STALE PRICE DETECTION
        # Blocks evaluation when price has not moved by even 1 pip in >30 min.
        # ════════════════════════════════════════════════════════════════════
        from datetime import datetime as _dt

        # Use live_price if provided (from exchange), fallback to last closed bar
        _current_price = (
            live_price
            if live_price is not None
            else (float(df["close"].iloc[-1]) if len(df) > 0 else 0.0)
        )
        _now = _dt.now()
        _last = self._last_prices.get(self.asset_type)
        if _last:
            _last_price, _last_time = _last
            _minutes_since_move = (_now - _last_time).total_seconds() / 60
            _price_moved = (
                abs(_current_price - _last_price) / max(_last_price, 1) > 0.00001
            )
            if not _price_moved:
                if _minutes_since_move > self._stale_threshold_minutes:
                    logger.warning(
                        f"[COUNCIL] ⏸️ Stale price: {self.asset_type} frozen at "
                        f"{_current_price} for {_minutes_since_move:.0f}min — blocking"
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "signal": 0,
                        "asset": self.asset_type,
                        "reasoning": f"stale_price_{_minutes_since_move:.0f}min",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0,
                        "mr_confidence": 0.0,
                        "tf_signal": 0,
                        "tf_confidence": 0.0,
                        "ema_signal": 0,
                        "ema_confidence": 0.0,
                    }
                # ✅ IMPORTANT: Even if price didn't move, if we successfully fetched data
                # that matches the current anchor, we don't update the anchor.
                # However, if we WANT to reset the timer because we verified the "stillness"
                # is from fresh data, we would update the time.
                # BUT the user reports the fetched close prices ARE moving.
                # If they move > 0.001% (0.00001), they will hit the update block below.
                # If they move LESS than that, the timer keeps ticking.

            else:
                # Price moved! Update anchor immediately
                self._last_prices[self.asset_type] = (_current_price, _now)
        else:
            # First run for this asset
            self._last_prices[self.asset_type] = (_current_price, _now)

        # ════════════════════════════════════════════════════════════════════
        # T3.3: NY OPEN HOUR BLOCK (13:00–13:59 UTC)
        # USTEC/GOLD/USOIL/GBPAUD — FX majors excluded (13:00 UTC is their
        # best liquidity hour). USOIL and GBPAUD added: same stop-hunting
        # behaviour as GOLD/USTEC at NY open (53% WR, -21% P&L in that hour).
        # ════════════════════════════════════════════════════════════════════
        _hour_utc = _dt.utcnow().hour
        if _hour_utc == 13 and self.asset_type in (
            "USTEC",
            "US100",
            "NAS100",
            "GOLD",
            "XAUUSD",
            "USOIL",
            "OIL",
            "GBPAUD",
        ):
            logger.info(
                f"[COUNCIL] ⏸️ NY open hour block — no new entries for {self.asset_type}"
            )
            return 0, {
                "timestamp": timestamp,
                "signal": 0,
                "asset": self.asset_type,
                "reasoning": "ny_open_block",
                "final_signal": 0,
                "signal_quality": 0.0,
                "mr_signal": 0,
                "mr_confidence": 0.0,
                "tf_signal": 0,
                "tf_confidence": 0.0,
                "ema_signal": 0,
                "ema_confidence": 0.0,
            }

        # ════════════════════════════════════════════════════════════════════
        # T3.4: ECONOMIC CALENDAR BLOCK
        # Block N hours before each high-impact event that affects this asset.
        # ════════════════════════════════════════════════════════════════════
        if self._econ_events:
            from datetime import timezone as _tz, timedelta as _td

            _utc_now = _dt.now(_tz.utc)
            for _evt in self._econ_events:
                try:
                    _evt_time = _dt.fromisoformat(
                        _evt["datetime"].replace("Z", "+00:00")
                    )
                    _hours_before = _evt.get("block_hours_before", 2)
                    _block_start = _evt_time - _td(hours=_hours_before)
                    if _block_start <= _utc_now < _evt_time:
                        _affected = _evt.get("currencies", [])
                        _blocked = (
                            (
                                self.asset_type in ("BTC", "BTCUSDT")
                                and "USD" in _affected
                            )
                            or (
                                self.asset_type in ("GOLD", "XAUUSD")
                                and "USD" in _affected
                            )
                            or (
                                self.asset_type == "EURUSD"
                                and ("EUR" in _affected or "USD" in _affected)
                            )
                            or (
                                self.asset_type == "EURJPY"
                                and ("EUR" in _affected or "JPY" in _affected)
                            )
                            or (
                                self.asset_type in ("USTEC", "US100", "NAS100")
                                and "USD" in _affected
                            )
                        )
                        if _blocked:
                            _mins_to_evt = (_evt_time - _utc_now).total_seconds() / 60
                            logger.warning(
                                f"[COUNCIL] ⛔ Calendar block: {_evt.get('event', 'HIGH IMPACT')} "
                                f"in {_mins_to_evt:.0f}min"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "signal": 0,
                                "asset": self.asset_type,
                                "reasoning": f"econ_calendar_{_evt.get('event','').replace(' ','_')}",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "mr_signal": 0,
                                "mr_confidence": 0.0,
                                "tf_signal": 0,
                                "tf_confidence": 0.0,
                                "ema_signal": 0,
                                "ema_confidence": 0.0,
                            }
                except Exception:
                    continue

        # ====================================================================
        # ⚡ THE FLASH VETO: Volatility Circuit Breaker
        # ====================================================================
        # Reason: Detects Black Swan crashes in real-time before EMAs flip.
        try:
            if len(df) >= 20:
                latest = df.iloc[-1]
                high, low, close = (
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                )
                atr_20 = ta.ATR(high, low, close, timeperiod=20)[-1]
                vol_avg = (
                    df["volume"].iloc[-21:-1].mean() if "volume" in df.columns else 0
                )

                candle_body = (
                    latest["close"] - latest["open"]
                )  # Positive for bull, negative for bear
                candle_size = abs(candle_body)
                vol_raw = latest.get("volume", 0)
                vol_ratio = (vol_raw / vol_avg) if vol_avg > 0 and vol_raw > 0 else 1.0

                # ✅ TASK 19: Calibrated Flash Veto (Phase 3)
                # Reason: 2.5x ATR was too tight for CPI/FOMC; 3.0x Volume missed real institutional moves.
                if candle_body < 0 and candle_size > (2.8 * atr_20) and vol_ratio > 2.5:
                    logger.warning(
                        f"[FLASH VETO] 🚨 BLACK SWAN DETECTED: Velocity {candle_size/atr_20:.1f}x ATR + Volume {vol_ratio:.1f}x AVG. Blocking all trades."
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "signal": 0,
                        "asset": self.asset_type,
                        "decision_type": "BLOCKED (Flash Veto)",
                        "reasoning": "black_swan_circuit_breaker",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0,
                        "mr_confidence": 0.0,
                        "tf_signal": 0,
                        "tf_confidence": 0.0,
                        "ema_signal": 0,
                        "ema_confidence": 0.0,
                    }
        except Exception as e:
            logger.debug(f"[COUNCIL] Flash Veto check skipped: {e}")

        # ====================================================================
        # STEP 1 — Governor-First Protocol
        # ====================================================================
        # Move macro regime check to the very first step.
        if self.use_macro_governor:
            macro_regime = self._check_macro_regime(self.asset_type)

            # Determine preliminary signal from Trend Following to enable immediate veto
            # Reason: Must check veto BEFORE computing RSI, ATR, or AI validation.
            try:
                # We use a fast, low-compute check of the Trend Following strategy
                prelim_signal, _ = self.s_trend_following.generate_signal(
                    df, silent=True
                )

                # T1.3: Smart Gatekeeper — preliminary TF check is advisory only.
                # The authoritative regime gate is _check_governor_filter() called
                # after full council scoring. Hard-vetoing here based on a single
                # strategy's prelim signal was blocking valid counter-trend signals
                # that the full scorecard would have rejected anyway.
                if macro_regime == "BEARISH" and prelim_signal == 1:
                    logger.info(
                        "[COUNCIL] ⚠️ Governor pre-check: Bearish regime vs LONG prelim — proceeding to full scoring."
                    )
                if macro_regime == "BULLISH" and prelim_signal == -1:
                    logger.info(
                        "[COUNCIL] ⚠️ Governor pre-check: Bullish regime vs SHORT prelim — proceeding to full scoring."
                    )
            except Exception as e:
                logger.debug(f"[COUNCIL] Governor-First check skipped: {e}")

        try:
            # ================================================================
            # VOLATILITY & REGIME CONTEXT
            # ================================================================
            high, low, close_vals = (
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )
            atr_fast = ta.ATR(high, low, close_vals, timeperiod=14)[-1]
            atr_slow = ta.ATR(high, low, close_vals, timeperiod=100)[-1]
            adx = ta.ADX(high, low, close_vals, timeperiod=14)[-1]

            # ✅ FIXED: Use the highly-accurate MTF data if provided, otherwise fallback
            mr_signal, mr_conf = 0, 0.0
            tf_signal, tf_conf = 0, 0.0
            ema_signal, ema_conf = 0, 0.0

            if governor_data:
                is_bull = is_bull_market
                regime_name = current_regime
                regime_conf = governor_data.get("confidence", 0.5)
            else:
                # Get regime context
                is_bull, regime_conf = self._detect_regime(df)
                regime_name = "🚀 BULL" if is_bull else "🐻 BEAR"

            # ================================================================
            # ⚖️ DYNAMIC COUNCIL WEIGHTS (Phase 4)
            # ================================================================
            w_trend = self.w_trend
            w_structure = self.w_structure
            w_momentum = self.w_momentum
            w_pattern = self.w_pattern
            w_volume = self.w_volume

            consensus_regime = (
                governor_data.get("consensus_regime", "NEUTRAL")
                if governor_data
                else "NEUTRAL"
            )

            if consensus_regime in ["SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"]:
                w_momentum = 0.75  # ✨ Balanced momentum points
                w_structure = 1.5  # ✨ Standard structure weight
                w_pattern = 0.75  # ✨ Balanced pattern weight
                if self.detailed_logging:
                    logger.info(
                        f"[COUNCIL] ⚖️ DYNAMIC WEIGHTS APPLIED: {consensus_regime}"
                    )

            # Pass 4H context and composite_state (Livermore) from governor_data.
            # composite_state is built by the LSM companion PerformanceWeightedAggregator
            # in main.py before the council aggregator is called — without it, MR strategy
            # always falls back to LEGACY(warmup) mode and Phase 3A/3B gates never fire.
            df_4h = governor_data.get("df_4h") if governor_data else None
            _composite_state = (governor_data or {}).get("composite_state")

            if self.s_mean_reversion:
                try:
                    mr_signal, mr_conf = self.s_mean_reversion.generate_signal(
                        df, df_4h=df_4h, composite_state=_composite_state
                    )
                except Exception as e:
                    logger.error(f"[COUNCIL] MR signal error: {e}")

            if self.s_trend_following:
                try:
                    # L10: pass composite_state through so the TF strategy's
                    # Livermore awareness nudge (flag-gated) can see live LSM state.
                    tf_signal, tf_conf = self.s_trend_following.generate_signal(
                        df, df_4h=df_4h, composite_state=_composite_state
                    )
                except Exception as e:
                    logger.error(f"[COUNCIL] TF signal error: {e}")

            if self.s_ema:
                try:
                    # L10: same wiring as TF above.
                    ema_signal, ema_conf = self.s_ema.generate_signal(
                        df, df_4h=df_4h, composite_state=_composite_state
                    )
                except Exception as e:
                    logger.error(f"[COUNCIL] EMA signal error: {e}")

            # ================================================================
            # BIDIRECTIONAL SCORING: Evaluate both BUY and SELL
            # ================================================================

            # BUY scorecard
            buy_scores = {
                "trend": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "pattern": 0.0,
                "volume": 0.0,
            }
            buy_explanations = []

            # SELL scorecard
            sell_scores = {
                "trend": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "pattern": 0.0,
                "volume": 0.0,
            }
            sell_explanations = []

            # ✨ NEW: Detect Breakout State to enable adaptive logic
            is_breakout_mode = self._detect_breakout_state(df)

            # Run all judges for both directions
            buy_scores["trend"], sell_scores["trend"], trend_exp = (
                self._judge_trend_bidirectional(
                    df, is_bull, w_trend, consensus_regime, governor_data=governor_data
                )
            )
            buy_explanations.append(trend_exp["buy"])
            sell_explanations.append(trend_exp["sell"])

            # Pass breakout flag and ADX to adaptive judges
            buy_scores["structure"], sell_scores["structure"], structure_exp = (
                self._judge_structure_bidirectional(
                    df, is_breakout_mode, w_structure, adx, governor_data=governor_data
                )
            )
            buy_explanations.append(structure_exp["buy"])
            sell_explanations.append(structure_exp["sell"])

            # Use trend-aligned momentum scoring for non-neutral regimes.
            # The reversion judge (RSI extreme crosses) is wrong for trend-continuation
            # setups — it scores 0 when RSI is 50-65, which is exactly where a
            # healthy trend lives. The trend momentum judge correctly awards points
            # when RSI is in the directional zone (e.g. 40-65 for bearish GOLD).
            # This was causing all MT5 trend signals to score only 2.25/5.0 max
            # (TREND 1.5 + PATTERN 0.5 + VOLUME 0.25) regardless of how aligned
            # TF/EMA were, because is_breakout_mode was always False for MT5 assets
            # (MT5 tick volume is unreliable so the volume-surge gate never triggered).
            is_trending_regime = consensus_regime not in ["NEUTRAL", "UNKNOWN"]
            # Fallback: MTF regime detector often returns NEUTRAL (0% confidence) for all
            # assets because the MTF data feed is stale or the regime boundary is ambiguous.
            # In that case, use local ADX as a tie-breaker — if ADX > 20 the market is
            # directional regardless of what the regime string says, so switch to the
            # trend-aligned momentum judge to avoid permanently scoring only 1.75/5.0.
            ADX_TRENDING_THRESHOLD = 20
            if not is_trending_regime and adx > ADX_TRENDING_THRESHOLD:
                is_trending_regime = True
                logger.info(
                    f"[MOMENTUM] MTF regime='{consensus_regime}' but local ADX={adx:.1f} > {ADX_TRENDING_THRESHOLD} "
                    f"— overriding to trend-aligned momentum judge"
                )
            if is_breakout_mode or is_trending_regime:
                buy_scores["momentum"], sell_scores["momentum"], momentum_exp = (
                    self._judge_momentum_bidirectional(
                        df,
                        is_bull,
                        is_breakout_mode,
                        w_momentum,
                        adx,
                        governor_data=governor_data,
                    )
                )
            else:
                buy_scores["momentum"], sell_scores["momentum"], momentum_exp = (
                    self._judge_reversion_bidirectional(df, w_momentum)
                )

            buy_explanations.append(momentum_exp["buy"])
            sell_explanations.append(momentum_exp["sell"])

            buy_scores["pattern"], sell_scores["pattern"], pattern_exp = (
                self._judge_pattern_bidirectional(
                    df, w_pattern, governor_data=governor_data
                )
            )
            buy_explanations.append(pattern_exp["buy"])
            sell_explanations.append(pattern_exp["sell"])

            buy_scores["volume"], sell_scores["volume"], volume_exp = (
                self._judge_volume_bidirectional(df, w_volume, governor_data=governor_data)
            )
            buy_explanations.append(volume_exp["buy"])
            sell_explanations.append(volume_exp["sell"])

            # ════════════════════════════════════════════════════════════════
            # T2.6: CONSECUTIVE CANDLE COUNTER + ADX GUARD
            # Applied as a momentum score modifier after all judges run.
            # BTC: 3+ consecutive same-direction bars + ADX < 25 → MR setup
            # GOLD/USTEC: 5+ consecutive bars → trend continuation boost
            # ════════════════════════════════════════════════════════════════
            try:
                _closes = df["close"].values
                _consec = 0
                for _ci in range(len(_closes) - 1, max(len(_closes) - 10, 0), -1):
                    if _ci == 0:
                        break
                    if _closes[_ci] > _closes[_ci - 1]:
                        if _consec >= 0:
                            _consec += 1
                        else:
                            break
                    else:
                        if _consec <= 0:
                            _consec -= 1
                        else:
                            break

                if (
                    self.asset_type in ("BTC", "BTCUSDT")
                    and abs(_consec) >= 3
                    and adx < 25
                ):
                    # Bearish streak → MR long signal; bullish streak → MR short
                    _boost = min(0.15 * w_momentum, 0.25)
                    if _consec < 0:  # consecutive bearish → boost BUY momentum
                        buy_scores["momentum"] = min(
                            w_momentum, buy_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[CANDLE] BTC {abs(_consec)}-bar bear streak + ADX={adx:.1f}<25: BUY momentum +{_boost:.2f}"
                        )
                    elif _consec > 0:  # consecutive bullish → boost SELL momentum
                        sell_scores["momentum"] = min(
                            w_momentum, sell_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[CANDLE] BTC {_consec}-bar bull streak + ADX={adx:.1f}<25: SELL momentum +{_boost:.2f}"
                        )

                if (
                    self.asset_type in ("GOLD", "XAUUSD", "USTEC", "US100", "NAS100")
                    and abs(_consec) >= 5
                ):
                    _boost = min(0.2 * w_momentum, 0.3)
                    if _consec > 0:
                        buy_scores["momentum"] = min(
                            w_momentum, buy_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[CANDLE] {self.asset_type} {_consec}-bar bull streak: BUY momentum +{_boost:.2f}"
                        )
                    elif _consec < 0:
                        sell_scores["momentum"] = min(
                            w_momentum, sell_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[CANDLE] {self.asset_type} {abs(_consec)}-bar bear streak: SELL momentum +{_boost:.2f}"
                        )
            except Exception:
                pass  # Bonus only — never block on failure

            # ════════════════════════════════════════════════════════════════
            # T3.5: BTC FUNDING RATE Z-SCORE MOMENTUM MODIFIER
            # Extreme funding rates (|Z| >= 2.0) → crowded positioning → MR edge
            # ════════════════════════════════════════════════════════════════
            _funding_z = (
                governor_data.get("funding_rate_zscore", 0.0) if governor_data else 0.0
            )
            if self.asset_type in ("BTC", "BTCUSDT") and abs(_funding_z) >= 2.0:
                _boost = min(0.15 * w_momentum, 0.25)
                if _funding_z > 0:  # over-leveraged longs → MR short is high prob
                    sell_scores["momentum"] = min(
                        w_momentum, sell_scores["momentum"] + _boost
                    )
                    logger.info(
                        f"[FUNDING] Extreme long positioning (Z={_funding_z:+.1f}): SELL momentum +{_boost:.2f}"
                    )
                else:  # over-leveraged shorts → MR long is high prob
                    buy_scores["momentum"] = min(
                        w_momentum, buy_scores["momentum"] + _boost
                    )
                    logger.info(
                        f"[FUNDING] Extreme short positioning (Z={_funding_z:+.1f}): BUY momentum +{_boost:.2f}"
                    )

            # ════════════════════════════════════════════════════════════════
            # T3.6: DXY PROXY MOMENTUM MODIFIER
            # Rising EUR/USD = falling dollar = tailwind for GOLD/USTEC/EURJPY
            # ════════════════════════════════════════════════════════════════
            _dxy_falling = governor_data.get("dxy_falling") if governor_data else None
            if _dxy_falling is not None:
                _boost = min(0.10 * w_momentum, 0.15)
                if self.asset_type in ("GOLD", "XAUUSD"):
                    if _dxy_falling:
                        buy_scores["momentum"] = min(
                            w_momentum, buy_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[DXY] Weak dollar: GOLD BUY momentum +{_boost:.2f}"
                        )
                    else:
                        sell_scores["momentum"] = min(
                            w_momentum, sell_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[DXY] Strong dollar: GOLD SELL momentum +{_boost:.2f}"
                        )
                elif self.asset_type in ("USTEC", "US100", "NAS100"):
                    if _dxy_falling:
                        buy_scores["momentum"] = min(
                            w_momentum, buy_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[DXY] Weak dollar: USTEC BUY momentum +{_boost:.2f}"
                        )
                elif self.asset_type == "EURJPY":
                    if _dxy_falling:
                        buy_scores["momentum"] = min(
                            w_momentum, buy_scores["momentum"] + _boost
                        )
                        logger.debug(
                            f"[DXY] Weak dollar: EURJPY BUY momentum +{_boost:.2f}"
                        )

            # ════════════════════════════════════════════════════════════════
            # T4.0: SCORE TRAJECTORY AWARENESS (Momentum-of-Conviction)
            # Reason: every judge above recomputes purely from the current bar.
            # A side that just flipped from near-zero to a high score this one
            # cycle looks IDENTICAL to the council as a side that has been
            # building steadily for 10 cycles — there was no memory of last
            # cycle's scores anywhere in this class (self.decision_history is
            # write-only/log-only; self.previous_regime is dead state). That's
            # exactly the "doesn't know what has been happening" gap.
            #
            # Fix: compare each judge's RAW score (pre-modifier, this cycle)
            # against its own reading from the previous cycle. A judge that
            # rose gets a small same-direction boost (forming); a judge that
            # fell gets a small same-direction penalty (losing steam). Capped
            # per-judge at that judge's own weight ceiling, symmetric, and
            # applied the same additive way as the existing ADX-slope /
            # funding / DXY modifiers above — it nudges the scorecard, it
            # never blocks a trade outright on its own.
            # ════════════════════════════════════════════════════════════════
            _raw_buy_scores = dict(buy_scores)
            _raw_sell_scores = dict(sell_scores)
            _TRAJ_GAIN = 0.15  # 15% of a judge's own cycle-over-cycle delta
            _judge_weight_cap = {
                "trend": w_trend,
                "structure": min(
                    w_structure, 1.0
                ),  # STRUCTURE judge hard-caps its own return at 1.0
                "momentum": w_momentum,
                "pattern": w_pattern,
                "volume": w_volume,
            }

            _prev_cycle = self.score_history[-1] if self.score_history else None
            if _prev_cycle:
                _prev_buy = _prev_cycle.get("buy_scores", {})
                _prev_sell = _prev_cycle.get("sell_scores", {})
                for _j in list(buy_scores.keys()):
                    _cap = _judge_weight_cap.get(_j, 5.0)
                    _pb = _prev_buy.get(_j)
                    if _pb is not None:
                        _adj = (_raw_buy_scores[_j] - _pb) * _TRAJ_GAIN
                        if _adj != 0.0:
                            buy_scores[_j] = float(
                                np.clip(buy_scores[_j] + _adj, 0.0, _cap)
                            )
                    _ps = _prev_sell.get(_j)
                    if _ps is not None:
                        _adj = (_raw_sell_scores[_j] - _ps) * _TRAJ_GAIN
                        if _adj != 0.0:
                            sell_scores[_j] = float(
                                np.clip(sell_scores[_j] + _adj, 0.0, _cap)
                            )

            # Persist this cycle's RAW (pre-trajectory-adjustment) scores for next
            # cycle's comparison — diffing against already-adjusted values would
            # compound the nudge instead of measuring fresh judge-level momentum.
            self.score_history.append(
                {
                    "timestamp": timestamp,
                    "buy_scores": _raw_buy_scores,
                    "sell_scores": _raw_sell_scores,
                    "buy_total": sum(_raw_buy_scores.values()),
                    "sell_total": sum(_raw_sell_scores.values()),
                }
            )

            # Calculate total scores
            buy_total = sum(buy_scores.values())
            sell_total = sum(sell_scores.values())

            # ── MR Direct Routing ─────────────────────────────────────────────
            # MR signal is collected above but never reached the council total.
            # In MR's primary states (NATURAL_RETRACEMENT = long spring,
            # NATURAL_REBOUND = short spring) the signal is directionally
            # authoritative — add it directly so the judges' trend-following
            # bias cannot silence a correct MR call. Weight 0.75 is additive
            # only — MR alone cannot clear the 2.75 trend threshold.
            _lsm_1h_for_mr = (
                getattr(_composite_state, "livermore_state_1h", None)
                if _composite_state is not None
                else None
            )
            _MR_LONG_STATES  = ("NATURAL_RETRACEMENT",)
            _MR_SHORT_STATES = ("NATURAL_REBOUND",)
            _MR_WEIGHT = 0.75
            if mr_signal == 1 and _lsm_1h_for_mr in _MR_LONG_STATES:
                buy_total += mr_conf * _MR_WEIGHT
                logger.info(
                    f"[COUNCIL MR ROUTE] {self.asset_type}: Mode1 long "
                    f"(conf={mr_conf:.2f}) added {mr_conf * _MR_WEIGHT:.2f} to buy_total "
                    f"→ buy_total now {buy_total:.2f}"
                )
            elif mr_signal == -1 and _lsm_1h_for_mr in _MR_SHORT_STATES:
                sell_total += mr_conf * _MR_WEIGHT
                logger.info(
                    f"[COUNCIL MR ROUTE] {self.asset_type}: Mode1 short "
                    f"(conf={mr_conf:.2f}) added {mr_conf * _MR_WEIGHT:.2f} to sell_total "
                    f"→ sell_total now {sell_total:.2f}"
                )

            # ── Item 19b: 4H Livermore / macro-regime disagreement gate ──────
            # Flag-gated (phase_config.lsm_regime_disagreement_gate_enabled,
            # default False — same on/off switch used in signal_aggregator.py
            # and main.py for this item). When enabled and the live 4H
            # Livermore state's lean disagrees with the EMA-derived `is_bull`
            # macro lean, the decision block below is skipped entirely this
            # cycle (HOLD) rather than trading on a regime label the structural
            # state machine actively contradicts. Mirrors "Council fix #4"'s
            # self.config.get('phase_config', {}) pattern just below.
            _lsm_4h_for_regime_c = (
                getattr(_composite_state, "livermore_state_4h", None)
                if _composite_state is not None
                else None
            )
            _lsm_lean_c = (
                "bullish"
                if _lsm_4h_for_regime_c
                in ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
                else (
                    "bearish"
                    if _lsm_4h_for_regime_c
                    in ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
                    else None
                )
            )
            _regime_lean_c = "bullish" if is_bull else "bearish"
            _lsm_gate_enabled_c = bool(
                (self.config.get("phase_config", {}) or {}).get(
                    "lsm_regime_disagreement_gate_enabled", False
                )
            )
            _regime_lsm_disagree = bool(
                _lsm_gate_enabled_c
                and _lsm_lean_c is not None
                and _lsm_lean_c != _regime_lean_c
            )

            # ================================================================
            # DECISION LOGIC: Choose strongest direction
            # ================================================================
            signal = 0
            total_score = 0.0
            required_score = self.trend_aligned_threshold
            chosen_scores = {}
            decision_type = "HOLD"
            lifecycle_phase = (
                "HEALTHY"  # Updated by _check_lifecycle_phase when signal != 0
            )

            # Determine preliminary trade type from preset
            preset_name = self.config.get("name", "balanced").lower()
            trade_type = "REVERSION" if preset_name == "mr" else "TREND"

            # Apply the same Livermore-state threshold-raise performance mode already
            # gets (signal_aggregator.py STEP 5B, Layer 1 only — no Retest Engine
            # layer here, that's performance-mode-specific), so council can't fire
            # more easily than performance just because it was never wired to this
            # safety adjustment.
            _rsm_state_council = (
                getattr(_composite_state, "livermore_state_4h", None)
                if _composite_state
                else None
            )
            if not hasattr(self, "_rsm_table_council"):
                import json as _json_rsm_c

                try:
                    with open("config/aggregator_presets.json") as _rsm_f_c:
                        _rsm_cfg_c = _json_rsm_c.load(_rsm_f_c)
                    self._rsm_table_council = _rsm_cfg_c.get(
                        "REQUIRED_SCORE_MODIFIER", {}
                    ).get("state_modifiers", {})
                    self._rsm_cap_council = _rsm_cfg_c.get(
                        "REQUIRED_SCORE_MODIFIER", {}
                    ).get("modifier_cap", 1.50)
                except Exception as _rsm_cfg_err_c:
                    logger.debug("[RSM][COUNCIL] config load error: %s", _rsm_cfg_err_c)
                    self._rsm_table_council = {}
                    self._rsm_cap_council = 1.50
            _rsm_delta_council = self._rsm_table_council.get(_rsm_state_council, 0.0)
            _effective_trend_threshold = min(
                self.trend_aligned_threshold + self._rsm_cap_council,
                self.trend_aligned_threshold + _rsm_delta_council,
            )

            # SLIGHTLY regime: raise the counter-trend bar to match signal_aggregator's 0.50
            # confidence penalty, preventing low-conviction counter-trend trades in ambiguous regimes.
            # Now also folds in the Livermore RSM delta above, same as trend threshold.
            _is_slightly = consensus_regime in ("SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH")
            _effective_counter_threshold = min(
                self.counter_trend_threshold + self._rsm_cap_council,
                (
                    self.counter_trend_threshold + 0.50
                    if _is_slightly
                    else self.counter_trend_threshold
                )
                + _rsm_delta_council,
            )
            # Counter-trend ceiling (phase_config.council_counter_trend_cap, default
            # 4.0 = legacy). Funnel soak (6/20-6/24) showed the effective counter-trend
            # bar climbing to 4.0 (base 3.5 + RSM +0.5), rejecting BTC counter-trend
            # signals scoring 3.50. Capping it stops the RSM raise from pushing the
            # counter-trend requirement past this value. Trend-aligned bar is untouched.
            _ct_cap = float(
                (self.config.get("phase_config", {}) or {}).get(
                    "council_counter_trend_cap", 4.0
                )
            )
            _effective_counter_threshold = min(_effective_counter_threshold, _ct_cap)
            if _is_slightly and self.detailed_logging:
                logger.debug(
                    f"[COUNCIL] SLIGHTLY regime — counter_trend_threshold raised: "
                    f"{self.counter_trend_threshold:.2f} → {_effective_counter_threshold:.2f}"
                )

            # Determine preliminary signal
            if _regime_lsm_disagree:
                logger.info(
                    f"[COUNCIL] {self.asset_type} 4H Livermore lean ({_lsm_lean_c}) disagrees "
                    f"with macro regime lean ({_regime_lean_c}) — HOLD "
                    f"(lsm_regime_disagreement_gate_enabled, buy={buy_total:.2f} sell={sell_total:.2f})"
                )
            elif is_bull and buy_total >= _effective_trend_threshold:
                signal = 1
                total_score = buy_total
                required_score = _effective_trend_threshold
                chosen_scores = buy_scores
                # ══════════════════════════════════════════════════════════════════
            # GATE A: STRUCTURAL LOCATION (RetestEngine)
            # ══════════════════════════════════════════════════════════════════
            # Runs before the final decision so statistics can only confirm a
            # structurally sound entry — they cannot override structural reality.
            #
            # CHASE_HARD: absolute veto. Price is too far from any structural
            #             reference. No council score can override this.
            # All other tiers adjust the threshold up or down based on the
            # quality of the structural location.
            # ══════════════════════════════════════════════════════════════════
            try:
                if not hasattr(self, "_council_retest_engine"):
                    import json as _j_re
                    try:
                        with open("config/aggregator_presets.json") as _f_re:
                            _re_cfg = _j_re.load(_f_re)
                        from src.analysis.retest_engine import RetestEngine as _RE
                        self._council_retest_engine = _RE(
                            _re_cfg.get("RETEST_ENGINE", {})
                        )
                        logger.info(
                            "[COUNCIL GATE] RetestEngine loaded for council mode"
                        )
                    except Exception as _re_load_err:
                        logger.warning(
                            "[COUNCIL GATE] RetestEngine failed to load: %s",
                            _re_load_err,
                        )
                        self._council_retest_engine = None

                _cre = getattr(self, "_council_retest_engine", None)
                if _cre is not None and _composite_state is not None:
                    _likely_dir = 1 if buy_total >= sell_total else -1
                    _rt_buy  = _cre.classify(df, _composite_state, self.asset_type, direction=+1)
                    _rt_sell = _cre.classify(df, _composite_state, self.asset_type, direction=-1)

                    _buy_type  = _rt_buy.retest_type  if _rt_buy  is not None else "UNKNOWN"
                    _sell_type = _rt_sell.retest_type if _rt_sell is not None else "UNKNOWN"

                    logger.info(
                        "[COUNCIL GATE] %s structural classification → "
                        "BUY=%s SELL=%s",
                        self.asset_type, _buy_type, _sell_type,
                    )

                    # ── ABSOLUTE VETO ───────────────────────────────────────
                    if _buy_type == "CHASE_HARD" and _sell_type == "CHASE_HARD":
                        logger.info(
                            "[COUNCIL GATE] %s CHASE_HARD both directions — "
                            "structural veto: price extended from all levels",
                            self.asset_type,
                        )
                        signal        = 0
                        decision_type = "HOLD (structural_chase_hard)"
                    elif _buy_type == "CHASE_HARD" and _likely_dir == 1:
                        logger.info(
                            "[COUNCIL GATE] %s BUY direction CHASE_HARD — "
                            "structural veto on long",
                            self.asset_type,
                        )
                        buy_total = 0.0
                    elif _sell_type == "CHASE_HARD" and _likely_dir == -1:
                        logger.info(
                            "[COUNCIL GATE] %s SELL direction CHASE_HARD — "
                            "structural veto on short",
                            self.asset_type,
                        )
                        sell_total = 0.0

                    # ── THRESHOLD MODIFIERS ─────────────────────────────────
                    # CLEAN (-0.20): at a defended level — easier to enter
                    # WICK  ( 0.00): spring/upthrust recovery — standard
                    # BREAKOUT (+0.10 to +0.40): fresh state, slight caution
                    # CHASE_SOFT (+0.75): extended, needs strong conviction
                    if _buy_type != "CHASE_HARD" and _rt_buy is not None:
                        _effective_trend_threshold   += _rt_buy.modifier
                        _effective_counter_threshold += _rt_buy.modifier
                        if _rt_buy.modifier != 0.0:
                            logger.debug(
                                "[COUNCIL GATE] %s threshold adjusted by %.2f "
                                "(structural=%s)",
                                self.asset_type, _rt_buy.modifier, _buy_type,
                            )

            except Exception as _gate_err:
                logger.debug(
                    "[COUNCIL GATE] Gate error (non-blocking): %s", _gate_err
                )
            # ══════════════════════════════════════════════════════════════════

            # ══════════════════════════════════════════════════════════════════
            # GATE B: DUAL TIMEFRAME CONFIRMATION
            # ══════════════════════════════════════════════════════════════════
            # When the 4H and 1H Livermore states agree on direction, the brain
            # has full structural conviction across both timeframes.
            # Ease the threshold slightly — the market is proving itself at two
            # zoom levels simultaneously. Conservative when timeframes disagree,
            # flexible when both are aligned. Floor at 2.0 ensures dual
            # confirmation alone can never open a trade.
            # ══════════════════════════════════════════════════════════════════
            if (
                _composite_state is not None
                and getattr(_composite_state, "livermore_dual_confirmation", False)
            ):
                _ease = 0.15
                _effective_trend_threshold   = max(2.0, _effective_trend_threshold   - _ease)
                _effective_counter_threshold = max(2.0, _effective_counter_threshold - _ease)
                logger.debug(
                    "[COUNCIL GATE] %s dual_confirmation — thresholds eased "
                    "by %.2f → trend=%.2f counter=%.2f",
                    self.asset_type, _ease,
                    _effective_trend_threshold,
                    _effective_counter_threshold,
                )
            # ══════════════════════════════════════════════════════════════════
            elif not is_bull and buy_total >= _effective_counter_threshold:
                signal = 1
                total_score = buy_total
                required_score = _effective_counter_threshold
                chosen_scores = buy_scores
            elif not is_bull and sell_total >= _effective_trend_threshold:
                signal = -1
                total_score = sell_total
                required_score = _effective_trend_threshold
                chosen_scores = sell_scores
            elif is_bull and sell_total >= _effective_counter_threshold:
                signal = -1
                total_score = sell_total
                required_score = _effective_counter_threshold
                chosen_scores = sell_scores

            # Capture initial consensus before penalties and vetos
            original_signal = signal

            # ── Council fix #4: CONVICTION MARGIN (flag-gated, default 0.0) ──────
            # Audit §12C.4: the decision uses one side's absolute total and ignores
            # the spread between BUY and SELL, so buy=3.1 / sell=3.0 trades as a
            # confident BUY (a coin-flip). When council_min_score_margin > 0 we
            # require the chosen side to beat the other by that gap, else HOLD.
            # Default 0.0 == current behaviour (no change until backtested).
            _pc_cfg = self.config.get("phase_config", {}) or {}
            _min_margin = float(_pc_cfg.get("council_min_score_margin", 0.0))
            if signal != 0 and _min_margin > 0.0:
                _gap = (
                    (buy_total - sell_total)
                    if signal == 1
                    else (sell_total - buy_total)
                )
                if _gap < _min_margin:
                    logger.info(
                        f"[COUNCIL] Conviction margin {_gap:.2f} < {_min_margin:.2f} "
                        f"(buy={buy_total:.2f} sell={sell_total:.2f}) — HOLD (chop)"
                    )
                    signal = 0
                    decision_type = f"HOLD (margin {_gap:.2f}<{_min_margin:.2f})"

            # ====================================================================
            # 🛡️ THE INTERCEPTOR: ABSOLUTE VETO (Phase 4)
            # ====================================================================
            entry_price = 0.0
            stop_loss = 0.0

            if signal != 0:
                # Pre-calculate entry and stop loss for gates and penalties
                try:
                    entry_price = float(df["close"].iloc[-1])
                    risk_cfg = self.config.get("risk", {})
                    sl_mult = risk_cfg.get("atr_multiplier", 1.5)
                    # 7.2 LIVE: mirror VTM's Livermore overlay so the gate judges the
                    # stop the trade will ACTUALLY get. Reads from _composite_state (in scope).
                    if getattr(_composite_state, "phase_config", {}).get(
                        "rr_gate_consistency_enabled", False
                    ):
                        _lv4h = (
                            getattr(_composite_state, "livermore_state_4h", None)
                            if _composite_state
                            else None
                        )
                        _adj = risk_cfg.get("livermore_atr_adjustments", {})
                        if _lv4h in ("MAIN_UP", "MAIN_DOWN"):
                            sl_mult += _adj.get("main_stop_add", 0.3)
                        elif _lv4h in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
                            sl_mult = max(
                                1.5, sl_mult - _adj.get("secondary_stop_sub", 0.2)
                            )
                    sl_dist = atr_fast * sl_mult
                    stop_loss = (
                        entry_price - sl_dist if signal == 1 else entry_price + sl_dist
                    )
                except Exception as e:
                    logger.warning(f"[COUNCIL] Initial price calculation failed: {e}")

                # 0. OPPOSITE TREND BLOCK (SAFETY VETO)
                # Reason: Prevents Council from "fighting" a strong trend (e.g., buying while TF is screaming SELL)
                if self.use_gatekeeper:
                    base_threshold = self.config.get("trend_safety_threshold", 0.50)

                    # Regime-adaptive threshold: in ambiguous regimes TF must be
                    # much more confident to override the Council's consensus.
                    if consensus_regime == "NEUTRAL":
                        safety_threshold = max(base_threshold, 0.75)
                    elif consensus_regime in ("SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"):
                        safety_threshold = max(base_threshold, 0.65)
                    else:
                        # STRONGLY trending — keep original tight threshold
                        safety_threshold = base_threshold

                    # High-conviction Council override: when Council ≥ 4.0/5.0
                    # in a non-strongly-trending regime, a borderline TF signal
                    # should not be able to veto it.
                    if total_score >= 4.0 and consensus_regime not in (
                        "STRONGLY_BULLISH",
                        "STRONGLY_BEARISH",
                    ):
                        safety_threshold = max(safety_threshold, 0.80)

                    # Near-unanimous Council (≥ 4.5/5.0) in NEUTRAL/SLIGHTLY regime:
                    # Council + AI are both signalling reversal — TF momentum lag
                    # should not be able to override a near-unanimous Council.
                    # Effectively disable the veto by requiring TF to be > 100%.
                    if total_score >= 4.5 and consensus_regime in (
                        "NEUTRAL",
                        "SLIGHTLY_BULLISH",
                        "SLIGHTLY_BEARISH",
                    ):
                        safety_threshold = 1.01  # impossible to exceed — veto bypassed

                    if self.detailed_logging:
                        logger.debug(
                            f"[VETO] Opposite-Trend gate: regime={consensus_regime}, "
                            f"council={total_score:.2f}/5.0, tf_conf={tf_conf:.2f}, "
                            f"threshold={safety_threshold:.2f}"
                        )

                    # ✅ FIX: Only apply the TF veto when the council's signal is
                    # COUNTER-TREND. The veto was designed to stop the council from
                    # fighting strong trends (BUY in BEARISH, SELL in BULLISH).
                    # Previously it fired symmetrically, also blocking trend-aligned
                    # council trades (e.g., SELL in BEARISH) whenever TF picked up
                    # short-term counter-trend momentum. A TF counter-trend reading
                    # should NOT override a high-conviction trend-aligned council —
                    # it's noise, not a regime change.
                    _council_counter_trend = (
                        signal == 1 and not is_bull
                    ) or (  # BUY in BEARISH regime
                        signal == -1 and is_bull
                    )  # SELL in BULLISH regime

                    if not _council_counter_trend:
                        # Trend-aligned council — log TF opposition for visibility
                        # but bypass the veto entirely.
                        if (signal == 1 and tf_signal == -1) or (
                            signal == -1 and tf_signal == 1
                        ):
                            logger.info(
                                f"[VETO] ⚡ TF counter-trend opposition noted "
                                f"(tf_conf={tf_conf:.2f}, threshold={safety_threshold:.2f}) "
                                f"but council is TREND-ALIGNED — veto bypassed."
                            )
                    elif (
                        signal == 1 and tf_signal == -1 and tf_conf >= safety_threshold
                    ) or (
                        signal == -1 and tf_signal == 1 and tf_conf >= safety_threshold
                    ):
                        logger.info(
                            f"[VETO] ❌ BLOCKED - Opposite Trend: TF signals strong opposition ({tf_conf:.2f} >= {safety_threshold}) while Council disagrees."
                        )
                        return 0, {
                            "timestamp": timestamp,
                            "signal": 0,
                            "asset": self.asset_type,
                            "decision_type": f"BLOCKED (Opposite Trend)",
                            "action": "rejected",
                            "original_signal": signal,
                            "reasoning": "blocked_by_opposite_trend",
                            "final_signal": 0,
                            "signal_quality": 0.0,
                            "total_score": total_score,
                            "scores": chosen_scores,
                            "buy_total": buy_total,
                            "sell_total": sell_total,
                            "regime": regime_name,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                        }

                # 1. MACRO GOVERNOR (ABSOLUTE VETO)
                # Reason: Proves macro alignment (1D 200 EMA). Sacrosanct macro rule.
                if self.use_macro_governor:
                    gov_passed, trade_type = self._check_governor_filter(
                        df, signal, governor_data, trade_type
                    )
                    if not gov_passed:
                        logger.info(f"[VETO] ❌ BLOCKED - Macro Regime Conflict.")
                        return 0, {
                            "timestamp": timestamp,
                            "signal": 0,
                            "asset": self.asset_type,
                            "decision_type": "BLOCKED (Macro Regime Conflict)",
                            "action": "rejected",
                            "original_signal": signal,
                            "reasoning": "blocked_by_macro_governor",
                            "final_signal": 0,
                            "signal_quality": 0.0,
                            "total_score": total_score,
                            "scores": chosen_scores,
                            "buy_total": buy_total,
                            "sell_total": sell_total,
                            "regime": regime_name,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                        }
                    # MRS §6: TRANSITION path removed — no score raise for NEUTRAL regime.
                    # trade_type arrives as "TREND" for NEUTRAL regime (see governor check above).

                    # SLIGHTLY_COUNTER — ambiguous regime counter-trend, raise required_score.
                    # TransitionDetector can soften the +0.50 raise if ≥2 of 4 sources
                    # confirm a reversal is building (momentum, S/R, order flow, candle).
                    # Matches Performance aggregator's gatekeeper behaviour in SLIGHTLY regimes.
                    elif trade_type == "SLIGHTLY_COUNTER":
                        trade_type = "TREND"  # Restore for downstream gates
                        _te_raise = 0.50
                        try:
                            from types import SimpleNamespace as _NS

                            # Use the real composite state now that council has
                            # access to it via governor_data. Fall back to the
                            # original stub only if it's genuinely unavailable.
                            _real_cs = (governor_data or {}).get("composite_state")
                            if _real_cs is not None and hasattr(
                                _real_cs, "nearby_4h_level"
                            ):
                                # Real CompositeState object — use directly.
                                _dummy_state = _real_cs
                            elif (
                                isinstance(_real_cs, dict)
                                and "nearby_4h_level" in _real_cs
                            ):
                                # Serialised dict — wrap in SimpleNamespace so
                                # _check_structure's `state.nearby_4h_level` attr
                                # access works without modification.
                                _dummy_state = _NS(
                                    **{
                                        k: v
                                        for k, v in _real_cs.items()
                                        if not k.startswith("_")
                                    }
                                )
                            else:
                                # No real state available — original fallback.
                                # cvd_trend and order_book_imbalance come from
                                # governor_data the same way they always did.
                                _ob_raw   = float(governor_data.get("order_book_imbalance", 0.0)) if governor_data else 0.0
                                _ob_valid = bool(governor_data.get("order_book_imbalance_valid", False)) if governor_data else False
                                _dummy_state = _NS(
                                    nearby_4h_level=None,
                                    level_test_count=0,
                                    level_defended=False,
                                    cvd_trend=(
                                        int(governor_data.get("cvd_trend", 0))
                                        if governor_data
                                        else 0
                                    ),
                                    order_book_imbalance=_ob_raw if _ob_valid else 0.0,
                                )
                            _depth = (
                                governor_data.get("depth_data")
                                if governor_data
                                else None
                            )
                            _te = self._transition_detector.collect_evidence(
                                asset=self.asset_type,
                                regime=regime_name,
                                df_4h=df_4h if df_4h is not None else pd.DataFrame(),
                                df_1h=df,
                                composite_state=_dummy_state,
                                cvd_trend=_dummy_state.cvd_trend,
                                order_book_imbalance=_dummy_state.order_book_imbalance,
                                depth_data=_depth,
                            )
                            if _te.conditions_met >= 2:
                                # Reduce raise proportionally to evidence strength; floor at 0.20
                                _te_raise = max(
                                    0.20, 0.50 - abs(_te.total_score) * 0.50
                                )
                                logger.info(
                                    f"[COUNCIL] 🔄 TRANSITION evidence softens SLIGHTLY_COUNTER raise: "
                                    f"0.50 → {_te_raise:.2f} "
                                    f"(score={_te.total_score:+.3f}, conditions={_te.conditions_met}/4, "
                                    f"dir={_te.direction}) [{self.asset_type}]"
                                )
                            else:
                                logger.debug(
                                    f"[COUNCIL] SLIGHTLY_COUNTER: insufficient transition evidence "
                                    f"({_te.conditions_met}/4 conditions) — full +0.50 raise applies "
                                    f"[{self.asset_type}]"
                                )
                        except Exception as _te_err:
                            logger.debug(
                                f"[COUNCIL] Transition evidence error in SLIGHTLY_COUNTER: {_te_err}"
                            )
                        # Honour the same counter-trend ceiling computed above so the
                        # SLIGHTLY_COUNTER raise can't reintroduce the 4.0 bar.
                        required_score = min(required_score + _te_raise, _ct_cap)
                        logger.info(
                            f"[GOV] ⚠️ SLIGHTLY_COUNTER: required score raised to {required_score:.2f}"
                        )

                # 2. ATR WICK TRAP (ABSOLUTE VETO) — T2.3: pass regime context
                # Fix #16 (council): Neutral regime SHORT bias — when is_bull=False and regime is NEUTRAL,
                # counter-trend LONGs were incorrectly flagged as "not regime-aligned", applying tighter
                # wick-trap thresholds. Neutral means no directional bias; all signals are equally aligned.
                _regime_is_bullish = (
                    governor_data.get("is_bullish", False) if governor_data else False
                )
                _regime_is_bearish = (
                    governor_data.get("is_bearish", False) if governor_data else False
                )
                _is_neutral_regime = not _regime_is_bullish and not _regime_is_bearish
                _trap_regime_aligned = (
                    _is_neutral_regime
                    or (signal == 1 and is_bull)
                    or (signal == -1 and not is_bull)
                )
                if not validate_candle_structure(
                    df,
                    self.asset_type,
                    direction="long" if signal == 1 else "short",
                    regime_confidence=regime_conf,
                    regime_aligned=_trap_regime_aligned,
                ):
                    logger.info(f"[VETO] ❌ BLOCKED - Institutional Wick Trap.")
                    return 0, {
                        "timestamp": timestamp,
                        "signal": 0,
                        "asset": self.asset_type,
                        "decision_type": "BLOCKED (Institutional Wick Trap)",
                        "action": "rejected",
                        "original_signal": signal,
                        "reasoning": "blocked_by_trap_filter",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "total_score": total_score,
                        "scores": chosen_scores,
                        "buy_total": buy_total,
                        "sell_total": sell_total,
                        "regime": regime_name,
                        "mr_signal": mr_signal,
                        "mr_confidence": mr_conf,
                        "tf_signal": tf_signal,
                        "tf_confidence": tf_conf,
                        "ema_signal": ema_signal,
                        "ema_confidence": ema_conf,
                    }

                # 3. DEAD VOLATILITY GATE (ABSOLUTE VETO)
                if not self._check_volatility_gate_adaptive(df, atr_fast, atr_slow):
                    logger.info(f"[VETO] ❌ BLOCKED - Dead Market Volatility.")
                    return 0, {
                        "timestamp": timestamp,
                        "signal": 0,
                        "asset": self.asset_type,
                        "decision_type": "BLOCKED (Dead Market Volatility)",
                        "action": "rejected",
                        "original_signal": signal,
                        "reasoning": "low_volatility_veto",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "total_score": total_score,
                        "scores": chosen_scores,
                        "buy_total": buy_total,
                        "sell_total": sell_total,
                        "regime": regime_name,
                        "mr_signal": mr_signal,
                        "mr_confidence": mr_conf,
                        "tf_signal": tf_signal,
                        "tf_confidence": tf_conf,
                        "ema_signal": ema_signal,
                        "ema_confidence": ema_conf,
                    }

                # 4. RISK/REWARD GATE (ECONOMIC VETO)
                # Reason: Ensures trade has sufficient economic potential before execution.
                try:
                    distance_to_sl = abs(entry_price - stop_loss)

                    if trade_type == "REVERSION":
                        # ⚡ EMA 20 as Take-Profit Magnet
                        ema_20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]

                        # Directional Guard: Ensure TP is in the right direction
                        if signal == 1 and ema_20 <= entry_price:
                            logger.info(
                                f"[COUNCIL] ❌ BLOCKED - Inverted TP Magnet (Long): EMA-20 {ema_20:.2f} <= Entry {entry_price:.2f}"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "signal": 0,
                                "asset": self.asset_type,
                                "decision_type": "BLOCKED (Inverted TP Magnet)",
                                "action": "rejected",
                                "original_signal": signal,
                                "reasoning": "inverted_mr_magnet_long",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "total_score": total_score,
                                "scores": chosen_scores,
                                "mr_signal": mr_signal,
                                "mr_confidence": mr_conf,
                                "tf_signal": tf_signal,
                                "tf_confidence": tf_conf,
                                "ema_signal": ema_signal,
                                "ema_confidence": ema_conf,
                            }
                        if signal == -1 and ema_20 >= entry_price:
                            logger.info(
                                f"[COUNCIL] ❌ BLOCKED - Inverted TP Magnet (Short): EMA-20 {ema_20:.2f} >= Entry {entry_price:.2f}"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "signal": 0,
                                "asset": self.asset_type,
                                "decision_type": "BLOCKED (Inverted TP Magnet)",
                                "action": "rejected",
                                "original_signal": signal,
                                "reasoning": "inverted_mr_magnet_short",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "total_score": total_score,
                                "scores": chosen_scores,
                                "mr_signal": mr_signal,
                                "mr_confidence": mr_conf,
                                "tf_signal": tf_signal,
                                "tf_confidence": tf_conf,
                                "ema_signal": ema_signal,
                                "ema_confidence": ema_conf,
                            }

                        distance_to_tp = abs(entry_price - ema_20)
                        take_profit = ema_20
                    else:
                        # Simulate Take Profit for TREND (Using first partial target or default)
                        risk_cfg = self.config.get("risk", {})
                        tp_mult_raw = risk_cfg.get("partial_targets", [2.0])[0]
                        tp_dist = atr_fast * tp_mult_raw
                        take_profit = (
                            entry_price + tp_dist
                            if signal == 1
                            else entry_price - tp_dist
                        )
                        distance_to_tp = abs(take_profit - entry_price)

                    # Compute R/R
                    rr_ratio = (
                        distance_to_tp / distance_to_sl if distance_to_sl > 0 else 0
                    )

                    # Strategy Rules
                    if trade_type == "TREND" and rr_ratio < 1.0:
                        logger.info(
                            f"[COUNCIL] R/R Gate: Trend trade rejected (R/R: {rr_ratio:.2f} < 1.0)"
                        )
                        return 0, {
                            "timestamp": timestamp,
                            "action": "rejected",
                            "original_signal": signal,
                            "reasoning": "rr_gate_rejected_trend",
                            "signal": 0,
                            "rr_ratio": rr_ratio,
                            "total_score": total_score,
                            "scores": chosen_scores,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                        }

                    if trade_type == "REVERSION" and rr_ratio < 0.6:
                        logger.info(
                            f"[COUNCIL] MR Trade rejected due to poor R/R (Magnet: {ema_20:.2f}, R/R: {rr_ratio:.2f} < 0.6)."
                        )
                        return 0, {
                            "timestamp": timestamp,
                            "action": "rejected",
                            "original_signal": signal,
                            "reasoning": "rr_gate_rejected_reversion",
                            "signal": 0,
                            "rr_ratio": rr_ratio,
                            "total_score": total_score,
                            "scores": chosen_scores,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                        }

                except Exception as e:
                    logger.warning(f"[COUNCIL] Risk/Reward Gate simulation failed: {e}")

                # 5. CANDLE MOMENTUM ALIGNMENT GATE (ABSOLUTE VETO)
                # Reason: Prevents adding trend-aligned exposure when short-term
                # price action is clearly moving against the proposed direction.
                # Addresses "9 SELL signals while BTC bounces" — macro regime is
                # sticky (BEARISH for hours) but recent 1H candles may show a
                # clear bullish rebound, making a new short entry very poor timing.
                #
                # Only applied to TREND-ALIGNED signals (counter-trend signals
                # already require a higher council score; MR setups may legitimately
                # trade against recent momentum).
                _cmr_trend_aligned = (
                    signal == -1 and not is_bull
                ) or (  # SELL in BEARISH regime
                    signal == 1 and is_bull
                )  # BUY  in BULLISH regime
                if _cmr_trend_aligned:
                    _cmr_cfg = self.config.get("momentum_alignment", {})
                    _cmr_candles = _cmr_cfg.get("candles", 3)
                    _cmr_min_agree = _cmr_cfg.get("min_agreement", 2)
                    _cmr_enabled = _cmr_cfg.get("enabled", True)

                    if _cmr_enabled:
                        _cmr_passed, _cmr_reason = (
                            self._check_recent_momentum_alignment(
                                df,
                                signal,
                                atr_fast,
                                n_candles=_cmr_candles,
                                min_agreement=_cmr_min_agree,
                            )
                        )
                        if not _cmr_passed:
                            logger.info(
                                f"[VETO] ❌ BLOCKED - Candle Momentum Reversal: {_cmr_reason}"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "signal": 0,
                                "asset": self.asset_type,
                                "decision_type": "BLOCKED (Candle Momentum Reversal)",
                                "action": "rejected",
                                "original_signal": signal,
                                "reasoning": "blocked_by_candle_momentum_reversal",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "total_score": total_score,
                                "scores": chosen_scores,
                                "buy_total": buy_total,
                                "sell_total": sell_total,
                                "regime": regime_name,
                                "mr_signal": mr_signal,
                                "mr_confidence": mr_conf,
                                "tf_signal": tf_signal,
                                "tf_confidence": tf_conf,
                                "ema_signal": ema_signal,
                                "ema_confidence": ema_conf,
                                "cmr_reason": _cmr_reason,
                            }

            # ====================================================================
            # 📉 MINOR FAILURES: SCORING PENALTIES (Phase 4)
            # ====================================================================
            if signal != 0:
                penalty = 0.0

                # A. SNIPER LOCK — MRS §6 Phase 0: gate removed.
                # CNN-LSTM sniper disconnected. Displacement check retained as
                # informational log only — no score penalty applied.
                sniper_passed, sniper_details = self._check_sniper_filter(df, signal)
                if not sniper_passed:
                    logger.debug(
                        f"[DISPLACEMENT] Low momentum candle — informational only, no penalty "
                        f"(MRS Phase 0 sniper gate removed)"
                    )

                # B. PROFIT ECONOMICS
                # ✅ FIXED: Using corrected method from Task 10
                if not self._check_profit_economics_adaptive(
                    entry_price, stop_loss, atr_fast
                ):
                    penalty += 1.0
                    logger.info(f"[PENALTY] ⚠️ Low profit economics: -1.0")

                # Apply penalties
                total_score -= penalty

                # C. SESSION LIQUIDITY PENALTY (Extended to all MT5 Assets)
                try:
                    from src.utils.market_hours import MarketHours

                    _hour_utc_s = _dt.utcnow().hour

                    # 1. BTC (Binance) is 24/7 - only check for global liquidity lows
                    if "BTC" in self.asset_type:
                        session_quality = MarketHours.get_btc_session_quality()
                        if session_quality == "LOW":
                            required_score += 0.5
                            logger.info(
                                f"[SESSION] ⚠️ BTC low liquidity: required score +0.5 → {required_score:.1f}"
                            )

                    # 2. MT5/Exness Assets - Apply Session Penalties
                    #
                    # NOTE: This used to hardcode its own hour ranges per asset,
                    # duplicating (and drifting from) MarketHours.PREFERRED_SESSIONS
                    # in market_hours.py. Found 2026-06-16: OIL here was stuck at a
                    # stale "13-19 UTC" guess while market_hours.py had been updated
                    # to the data-derived (6,20) window — the same asset/cycle could
                    # be judged in-session by one gate and off-session by this one.
                    # Now delegates to MarketHours so there's a single source of
                    # truth; only the category-matching (which canonical key this
                    # asset maps to) stays local, since self.asset_type spelling
                    # varies (e.g. "GOLD" vs "XAUUSD").
                    else:
                        is_off_session = False
                        asset = self.asset_type.upper()

                        _session_key = None
                        if any(
                            x in asset
                            for x in ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")
                        ):
                            # Each FX pair has its own window in PREFERRED_SESSIONS;
                            # use the exact pair if recognized, else fall back to
                            # the generic EURUSD window.
                            _session_key = (
                                asset
                                if asset in MarketHours.PREFERRED_SESSIONS
                                else "EURUSD"
                            )
                        elif "GOLD" in asset or "XAU" in asset:
                            _session_key = "GOLD"
                        elif any(
                            x in asset for x in ("USTEC", "US100", "NAS", "US30", "SPX")
                        ):
                            _session_key = "USTEC"
                        elif "OIL" in asset:
                            _session_key = "USOIL"

                        if _session_key and not MarketHours.is_preferred_session(
                            _session_key
                        ):
                            is_off_session = True
                            _window = MarketHours.PREFERRED_SESSIONS.get(_session_key)
                            logger.info(
                                f"[SESSION] ⚠️ {_session_key} off-session "
                                f"({_hour_utc_s}:00 UTC, window={_window})"
                            )

                        if is_off_session:
                            required_score += 0.5
                            logger.info(
                                f"[SESSION] Required score +0.5 → {required_score:.1f}"
                            )

                except Exception as e:
                    logger.warning(f"[SESSION] Gate calculation failed: {e}")

                # ✨ NEW: Dynamic Strategy Confidence Weighting
                # Reason: Boost high-performing strategies and penalize failing ones based on live history.
                if self.performance_tracker:
                    try:
                        winrate = self.performance_tracker.get_winrate(trade_type)

                        # A. PERFORMANCE CIRCUIT BREAKER (HARD VETO)
                        # Reason: Pause strategies that are statistically proven to be losing.
                        stats = self.performance_tracker.get_all_stats().get(
                            trade_type, {}
                        )
                        total_trades = stats.get("wins", 0) + stats.get("losses", 0)

                        if total_trades >= 10 and winrate < 0.35:
                            logger.warning(
                                f"[VETO] 🛑 Strategy Circuit Breaker: {trade_type} winrate {winrate:.1%} "
                                f"after {total_trades} trades is below 35% threshold. Blocking trade."
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "signal": 0,
                                "asset": self.asset_type,
                                "decision_type": f"BLOCKED (Strategy Circuit Breaker: {winrate:.1%})",
                                "action": "rejected",
                                "original_signal": signal,
                                "reasoning": "strategy_circuit_breaker",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "mr_signal": mr_signal,
                                "mr_confidence": mr_conf,
                                "tf_signal": tf_signal,
                                "tf_confidence": tf_conf,
                                "ema_signal": ema_signal,
                                "ema_confidence": ema_conf,
                            }

                        # B. DYNAMIC WEIGHTING (SOFT ADJUSTMENT)
                        # Requires the same minimum sample as the circuit breaker (10 trades)
                        # before adjusting scores. Below that threshold a 100% win rate is
                        # statistical noise from 1-2 trades — applying a 1.5× multiplier
                        # defeats the raised thresholds the Governor set for NEUTRAL regimes.
                        # Adjustment: Winrate 50% -> 1.0x, Winrate 80% -> 1.3x, Winrate 20% -> 0.7x
                        # Council fix #6 (flag-gated): expectancy/profit-factor
                        # weighting instead of win rate. Audit §12C.6: win rate
                        # ignores payoff — it throttles profitable low-hit-rate
                        # systems and boosts high-hit-rate negative-expectancy
                        # ones. When council_expectancy_weighting is on and a
                        # profit factor is available, map PF→multiplier
                        # (PF 1.0→1.0x, 1.5→1.3x, 0.5→0.7x, clamped 0.7–1.3).
                        # Default OFF == current win-rate behaviour.
                        _pc_dw = self.config.get("phase_config", {}) or {}
                        _DW_MIN_TRADES = int(
                            _pc_dw.get("council_dyn_weight_min_trades", 10)
                        )
                        _use_expectancy = bool(
                            _pc_dw.get("council_expectancy_weighting", False)
                        )
                        _pf = None
                        if _use_expectancy and hasattr(
                            self.performance_tracker, "get_profit_factor"
                        ):
                            _pf = self.performance_tracker.get_profit_factor(trade_type)

                        if (
                            total_trades >= _DW_MIN_TRADES
                            and _use_expectancy
                            and _pf is not None
                        ):
                            weight_multiplier = float(
                                min(1.3, max(0.7, 1.0 + 0.6 * (_pf - 1.0)))
                            )
                            old_score = total_score
                            total_score *= weight_multiplier
                            if abs(weight_multiplier - 1.0) > 0.05:
                                logger.info(
                                    f"[DYNAMIC WEIGHT] {trade_type} PF={_pf:.2f} "
                                    f"({total_trades} trades) -> "
                                    f"Multiplier {weight_multiplier:.2f}x | Score: {old_score:.2f} -> {total_score:.2f}"
                                )
                        elif total_trades >= _DW_MIN_TRADES:
                            weight_multiplier = 0.5 + winrate
                            old_score = total_score
                            total_score *= weight_multiplier
                            if abs(weight_multiplier - 1.0) > 0.05:
                                logger.info(
                                    f"[DYNAMIC WEIGHT] {trade_type} winrate {winrate:.1%} "
                                    f"({total_trades} trades) -> "
                                    f"Multiplier {weight_multiplier:.2f}x | Score: {old_score:.2f} -> {total_score:.2f}"
                                )
                        else:
                            logger.debug(
                                f"[DYNAMIC WEIGHT] {trade_type} skipped — only {total_trades}/{_DW_MIN_TRADES} "
                                f"trades (winrate {winrate:.1%} unreliable at this sample size)"
                            )
                    except Exception as e:
                        logger.warning(f"[DYNAMIC WEIGHT] Failed: {e}")

                # ════════════════════════════════════════════════════════════
                # LIFECYCLE / PHASE-AWARENESS GATE
                # Adjusts required_score based on where in the trend cycle the
                # market is.  ESTABLISHING lowers the bar; EXHAUSTED / EXTENDED
                # raise it.  Never blocks outright — that's the veto layer's job.
                # ════════════════════════════════════════════════════════════
                required_score, lifecycle_phase = self._check_lifecycle_phase(
                    df=df,
                    signal=signal,
                    adx=adx,
                    current_required_score=required_score,
                    governor_data=governor_data,
                )

                # Final execution check
                # Quality gate aligned with the council's own score threshold.
                # Previously 0.65 (requiring total_score ≥ 3.25) which was STRICTER
                # than the trend_aligned_threshold (3.0), making any signal that
                # barely passed the threshold still get rejected. Lowered to 0.55
                # so the score threshold (3.0) is the authoritative gate.
                min_quality_threshold = 0.55
                signal_quality = total_score / 5.0

                if total_score < required_score:
                    logger.info(
                        f"[SIGNAL] ❌ REJECTED - Score after penalties ({total_score:.2f}) < {required_score:.2f}"
                    )
                    signal = 0
                    decision_type = f"REJECTED (Score: {total_score:.2f})"
                elif signal_quality < min_quality_threshold:
                    logger.info(
                        f"[SIGNAL] ❌ REJECTED - Low Quality: {signal_quality:.2f} < {min_quality_threshold:.2f}"
                    )
                    signal = 0
                    decision_type = f"REJECTED (Quality: {signal_quality:.2f})"
                else:
                    decision_type = f"{'BUY' if signal == 1 else 'SELL'} (Confirmed)"

            # Map back chosen details
            if signal == 1:
                chosen_scores = buy_scores
                chosen_explanations = buy_explanations
            elif signal == -1:
                chosen_scores = sell_scores
                chosen_explanations = sell_explanations
            else:
                chosen_scores = {"buy": buy_scores, "sell": sell_scores}
                chosen_explanations = buy_explanations + sell_explanations
                if total_score == 0:
                    total_score = max(buy_total, sell_total)

            # Update statistics based on FINAL signal
            if signal == 1:
                self.stats["buy_signals"] += 1
                if is_bull:
                    self.stats["trend_aligned_buys"] += 1
                else:
                    self.stats["counter_trend_buys"] += 1
                self.stats["avg_score_on_trade"].append(total_score)
            elif signal == -1:
                self.stats["sell_signals"] += 1
                if not is_bull:
                    self.stats["trend_aligned_sells"] += 1
                else:
                    self.stats["counter_trend_sells"] += 1
                self.stats["avg_score_on_trade"].append(total_score)
            else:
                self.stats["hold_signals"] += 1
                self.stats["avg_score_on_hold"].append(total_score)

            # Calculate signal quality
            base_quality = min(total_score / 5.0, 1.0)
            if signal != 0:
                judge_agreement = sum(1 for s in chosen_scores.values() if s > 0) / len(
                    chosen_scores
                )
            else:
                judge_agreement = 0.5
            signal_quality = base_quality * (0.8 + 0.2 * judge_agreement)
            signal_quality = min(signal_quality, 1.0)

            # ── MR lean conflict gate ────────────────────────────────────────────
            # When MR's Livermore routing wanted to go the OPPOSITE direction but
            # was blocked by a mode gate (returned 0, 0.0), the council needs higher
            # conviction before going against MR's structural read.
            #
            # Root cause of USOIL June-2026 loss: SECONDARY_RETRACEMENT→MR lean LONG,
            # council SELL scored exactly 3.50 (threshold=3.50) → trade executed.
            # Price went up; the bounce MR identified was real.
            #
            # Root cause of BTC June-2026 loss: MAIN_UP → MR Mode3 (climax fade/SHORT).
            # No reversal candle found so MR returned 0, but price WAS overextended.
            # Council scored BUY 4.0 (trend+structure+momentum all max, zero pattern/volume)
            # and entered long at the top of an overextended leg → hit stop loss.
            # The prior comment "council going WITH the trend is correct" was wrong:
            # MAIN_UP means the leg is already extended; MR failing to find a reversal
            # candle means "not yet" for the fade, NOT "safe to buy."
            #
            # States and conflict bumps:
            #   SECONDARY_RETRACEMENT → MR leans LONG  → council SELL needs +0.5
            #   NATURAL_RETRACEMENT   → MR leans LONG  → council SELL needs +0.5
            #   SECONDARY_REBOUND     → MR leans SHORT → council BUY  needs +0.5
            #   MAIN_UP               → MR leans SHORT → council BUY  needs +1.5
            #   MAIN_DOWN             → MR leans LONG  → council SELL needs +1.5
            # Excluded: NATURAL_REBOUND (MR silent by design).
            # The +1.5 bump for MAIN states (vs +0.5 for secondary) reflects that
            # MAIN_UP/DOWN represents a more extreme overextension — only truly
            # exceptional council scores (≥ base + 1.5) justify going with the trend.
            # Mode-driven (phase_config.council_mr_lean_mode):
            #   "off"    → gate skipped entirely. The council still consumes the
            #              Livermore read directly elsewhere (livermore_state_1h is
            #              used by the trend-judge confirmation, lifecycle guard, and
            #              regime-disagreement gate), so removing THIS veto does not
            #              blind the council to structure.
            #   "strict" → legacy hard veto (bumps 1.5 MAIN / 0.5 secondary).
            #   "soft"   → relaxed bumps from phase_config (default 0.25 / 0.0).
            _pc_mrlean = self.config.get("phase_config", {}) or {}
            _mr_lean_mode = (
                str(_pc_mrlean.get("council_mr_lean_mode", "soft")).strip().lower()
            )
            if (
                _mr_lean_mode != "off"
                and signal != 0
                and mr_signal == 0
                and mr_conf == 0.0
            ):
                _lsm_lean = (
                    getattr(_composite_state, "livermore_state_1h", None)
                    if _composite_state
                    else None
                )
                _MR_LEAN_LONG = {
                    "SECONDARY_RETRACEMENT",
                    "NATURAL_RETRACEMENT",
                    "MAIN_DOWN",
                }
                _MR_LEAN_SHORT = {"SECONDARY_REBOUND", "MAIN_UP"}
                _lean_conflict = (signal == -1 and _lsm_lean in _MR_LEAN_LONG) or (
                    signal == +1 and _lsm_lean in _MR_LEAN_SHORT
                )
                if _lean_conflict:
                    _lean_dir = "LONG" if _lsm_lean in _MR_LEAN_LONG else "SHORT"
                    _council_dir = "SELL" if signal == -1 else "BUY"
                    # Use fixed base threshold (trend_aligned_threshold), not the
                    # already-adjusted required_score — prevents lifecycle and other
                    # adjustments from stacking and compounding the raise.
                    # MAIN states get a larger bump: price is more overextended and
                    # the historical failure rate is higher (observed June-2026).
                    # FIX 2026-06-16 (first pass): swapped from inverted (0.0 for
                    # MAIN, 0.50 for non-MAIN) to (0.50 for MAIN, 0.0 for non-MAIN).
                    # FIX 2026-06-16 (second pass): first pass only fixed which side
                    # got a bump, not the magnitudes — the comment block above (see
                    # "States and conflict bumps") documents a 0.5 / 1.5 split, not
                    # 0.0 / 0.5. Caught live: USTEC MAIN_UP conflict scored 4.25,
                    # cleared the old 3.5 bar (3.0+0.5) and executed, but would have
                    # failed the documented 4.5 bar (3.0+1.5). Now matches the spec.
                    _is_main_state = _lsm_lean in ("MAIN_UP", "MAIN_DOWN")
                    if _mr_lean_mode == "strict":
                        # Legacy hard-veto bumps (pre-2026-06-24 behaviour).
                        _bump = 1.50 if _is_main_state else 0.50
                    else:
                        # "soft" (default): relaxed, tunable bumps. Funnel+shadow soak
                        # (6/20-6/24) showed the +1.5 MAIN bump (bar = 2.75+1.5 = 4.25)
                        # blocking GOLD/USOIL/BTC trend signals scoring 4.0-4.3 that the
                        # shadow engine forward-tracked as winners. Default 0.25 MAIN
                        # keeps a light structural caution without amputating them.
                        _bump = (
                            float(_pc_mrlean.get("council_mr_lean_bump_main", 0.25))
                            if _is_main_state
                            else float(
                                _pc_mrlean.get("council_mr_lean_bump_secondary", 0.0)
                            )
                        )
                    _conflict_req = min(self.trend_aligned_threshold + _bump, 5.0)
                    if total_score < _conflict_req:
                        logger.warning(
                            f"[COUNCIL] 🛑 MR lean conflict: LSM={_lsm_lean} → MR leans "
                            f"{_lean_dir} but council wants {_council_dir}. "
                            f"Score {total_score:.2f} < {_conflict_req:.2f} "
                            f"(bump={_bump:+.1f}, main_state={_is_main_state}) — signal blocked."
                        )
                        signal = 0
                        signal_quality = 0.0
                        decision_type = f"HOLD (MR lean conflict — {_lsm_lean})"
                    else:
                        logger.info(
                            f"[COUNCIL] ⚠️ MR lean conflict: LSM={_lsm_lean} → MR leans "
                            f"{_lean_dir} vs council {_council_dir} — cleared elevated bar "
                            f"{total_score:.2f} ≥ {_conflict_req:.2f} (bump={_bump:+.1f})."
                        )

            # Enhance reasoning with specific bonuses
            main_reasoning = (
                f"{decision_type} (Score: {total_score:.2f}/{required_score:.1f})"
            )
            bonus_tags = []
            for exp in chosen_explanations:
                if "✨" in exp or "🚀" in exp:
                    tag = exp.split(":")[-1].split("(")[0].strip()
                    bonus_tags.append(tag)

            if bonus_tags:
                main_reasoning += " | " + " | ".join(bonus_tags[:2])

            # Build details dict
            details = {
                "timestamp": timestamp,
                "signal": signal,
                "asset": self.asset_type,
                "trade_type": trade_type,
                "strategy": trade_type,
                "decision_type": decision_type,
                "total_score": total_score,
                "required_score": required_score,
                "scores": chosen_scores,
                "buy_scores": buy_scores,
                "sell_scores": sell_scores,
                "buy_total": buy_total,
                "sell_total": sell_total,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "explanations": chosen_explanations,
                "signal_quality": signal_quality,
                "reasoning": main_reasoning,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
                "ema_signal": ema_signal,
                "ema_confidence": ema_conf,
                "buy_score": buy_total,
                "sell_score": sell_total,
                "aggregator_type": "council",
                "judge_agreement": judge_agreement,
                "atr_fast": atr_fast,
                "atr_slow": atr_slow,
                "lifecycle_phase": lifecycle_phase,
                "governor_data": governor_data,
                "livermore_state_1h": (
                    getattr(
                        governor_data.get("composite_state"), "livermore_state_1h", None
                    )
                    if governor_data
                    else None
                ),
                "livermore_state_4h": (
                    getattr(
                        governor_data.get("composite_state"), "livermore_state_4h", None
                    )
                    if governor_data
                    else None
                ),
                "viz_overlay": {
                    "divergence": (
                        self.divergence_detector.analyze(df)
                        if hasattr(self, "divergence_detector")
                        else None
                    ),
                    "break_retest": (
                        self.break_retest_validator.validate(df, self.asset_type)
                        if hasattr(self, "break_retest_validator")
                        else None
                    ),
                    "adx": adx,
                },
            }

            # Log decision
            if self.detailed_logging or signal != 0:
                self._log_decision_bidirectional(details)

            # Store history
            self.decision_history.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "score": total_score,
                    "regime": regime_name,
                }
            )

            # AI validation
            if self.ai_validator and signal != 0:
                original_signal = signal
                details["original_signal"] = original_signal

                validated_signal, ai_details = self.ai_validator.validate_signal(
                    signal=signal,
                    signal_details=details,
                    df=df,
                    composite_state=_composite_state,
                )

                if validated_signal != signal:
                    logger.warning(f"[AI] Overruled: {signal} → {validated_signal}")
                    signal = validated_signal
                    details["ai_modified"] = True
                    details["signal"] = signal

                try:
                    formatted_ai = self._format_ai_validation_for_viz(
                        final_signal=signal, details=details.copy(), df=df
                    )
                    details["ai_validation"] = formatted_ai
                except Exception as e:
                    logger.error(f"[COUNCIL] AI formatting failed: {e}")
                    details["ai_validation"] = {
                        "pattern_detected": False,
                        "pattern_name": "Error",
                        "pattern_confidence": 0.0,
                        "validation_passed": signal != 0,
                        "action": "error_formatting",
                        "error": str(e),
                    }

            elif self.ai_validator and signal == 0:
                try:
                    details["ai_validation"] = self._format_ai_validation_for_viz(
                        final_signal=signal, details=details.copy(), df=df
                    )
                except Exception as e:
                    logger.error(f"[COUNCIL] AI formatting for hold signal failed: {e}")
                    details["ai_validation"] = {
                        "pattern_detected": False,
                        "pattern_name": "None",
                        "pattern_confidence": 0.0,
                        "validation_passed": True,
                        "action": "hold",
                        "error": str(e),
                    }

            return signal, details

        except Exception as e:
            logger.error(f"[COUNCIL] Error: {e}", exc_info=True)
            return 0, {
                "error": str(e),
                "timestamp": timestamp,
                "signal": 0,
                "total_score": 0.0,
                "signal_quality": 0.0,
                "mr_signal": 0,
                "mr_confidence": 0.0,
                "tf_signal": 0,
                "tf_confidence": 0.0,
                "ema_signal": 0,
                "ema_confidence": 0.0,
                "reasoning": f"error: {str(e)[:50]}",
            }

    def _build_ai_validation_stub(self, signal: int, details: dict) -> dict:
        """
        Lightweight ai_validation stub for paths where the AI validator never ran
        (early-return vetos: CMR, governor, MR lean conflict, opposite-trend, etc.).
        Built purely from the details dict that's already been computed — no AI calls.
        """
        decision_type = details.get("decision_type", "")
        is_blocked = "BLOCKED" in decision_type or "HOLD" in decision_type
        action = "hold" if signal == 0 else "approved"

        # Extract a human-readable rejection reason from decision_type when blocked
        rejection_reasons: list = []
        if is_blocked and decision_type:
            # Strip wrapper text: "BLOCKED (Candle Momentum Reversal)" → the inner label
            inner = decision_type
            if "(" in inner:
                inner = inner.split("(", 1)[-1].rstrip(")")
            if inner and inner != decision_type:
                rejection_reasons = [inner]

        return {
            "pattern_detected": False,
            "pattern_name": "None",
            "pattern_confidence": 0.0,
            "validation_passed": signal != 0,
            "action": action,
            "rejection_reasons": rejection_reasons,
            "top3_patterns": [],
            "top3_confidences": [],
            "sr_analysis": {},
        }

    def get_aggregated_signal(
        self,
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",
        is_bull_market: bool = True,
        governor_data: Optional[Dict] = None,
        live_price: Optional[float] = None,
    ) -> Tuple[int, Dict]:
        """
        Public entry point — guaranteed to always return details with 'ai_validation'.
        Delegates to _get_aggregated_signal_impl; early-return veto paths inside that
        method skip the AI validator, so we patch in a lightweight stub here rather
        than letting main.py re-run _format_ai_validation_for_viz as a fallback.
        """
        signal, details = self._get_aggregated_signal_impl(
            df,
            current_regime=current_regime,
            is_bull_market=is_bull_market,
            governor_data=governor_data,
            live_price=live_price,
        )
        if not isinstance(details.get("ai_validation"), dict):
            details["ai_validation"] = self._build_ai_validation_stub(signal, details)
        return signal, details

    # ========================================================================
    # BIDIRECTIONAL JUDGES
    # ========================================================================

    def _judge_trend_bidirectional(
        self,
        df: pd.DataFrame,
        is_bull: bool,
        weight: float,
        consensus_regime: str = "NEUTRAL",
        governor_data: Optional[Dict] = None,
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 1: TREND (Bidirectional)

        L7: optionally confirms/contradicts the EMA-based trend read against the
        Livermore 4H state carried on governor_data["composite_state"]. Gated by
        phase_config.council_trend_judge_livermore_confirmation_enabled (default
        False) — purely additive scoring nudge, never flips a 0 score to nonzero
        and never lets a confirmed score exceed `weight`.
        """
        try:
            features = self.s_trend_following.generate_features(df.tail(250))
            if features.empty:
                return 0.0, 0.0, {"buy": "TREND: No data", "sell": "TREND: No data"}

            latest = features.iloc[-1]
            price = latest["close"]
            ema_20 = latest.get("ema_fast", 0)
            ema_50 = latest.get("ema_slow", 0)
            ema_200 = latest.get("ema_200", 0)

            buy_score = 0.0
            sell_score = 0.0

            # BUY scoring
            if price > ema_50:
                if ema_20 > ema_50:
                    buy_score = weight
                    buy_exp = f"TREND BUY: ✅ Full ({weight:.1f}) - Price > EMA50, EMA20 > EMA50"
                else:
                    buy_score = weight * 0.5
                    buy_exp = f"TREND BUY: ⚠️ Partial ({buy_score:.1f}) - Price > EMA50 but EMA20 < EMA50"
            elif (
                consensus_regime == "SLIGHTLY_BULLISH"
                and price < ema_50
                and price > ema_200
            ):
                buy_score = weight * 0.5
                buy_exp = f"TREND BUY: 🌊 Pullback ({buy_score:.1f}) - Slight Bullish regime, Price > EMA200"
            else:
                buy_exp = "TREND BUY: ❌ No credit - Price < EMA50"

            # SELL scoring
            if price < ema_50:
                if ema_20 < ema_50:
                    sell_score = weight
                    sell_exp = f"TREND SELL: ✅ Full ({weight:.1f}) - Price < EMA50, EMA20 < EMA50"
                else:
                    sell_score = weight * 0.5
                    sell_exp = f"TREND SELL: ⚠️ Partial ({sell_score:.1f}) - Price < EMA50 but EMA20 > EMA50"
            elif (
                consensus_regime == "SLIGHTLY_BEARISH"
                and price > ema_50
                and price < ema_200
            ):
                sell_score = weight * 0.5
                sell_exp = f"TREND SELL: 🌊 Pullback ({sell_score:.1f}) - Slight Bearish regime, Price < EMA200"
            else:
                sell_exp = "TREND SELL: ❌ No credit - Price > EMA50"

            # ── L7: Livermore 4H confirmation nudge ──────────────────────────
            self._last_trend_judge_tag = "LSM_UNAVAILABLE"
            try:
                _phase_cfg = (governor_data or {}).get("phase_config")
                if _phase_cfg is None:
                    _cs_probe = (governor_data or {}).get("composite_state")
                    _phase_cfg = getattr(_cs_probe, "phase_config", {}) or {}
                if _phase_cfg.get(
                    "council_trend_judge_livermore_confirmation_enabled", False
                ):
                    cs = (governor_data or {}).get("composite_state")
                    if cs is not None:
                        _lsm_4h = (
                            cs.get("livermore_state_4h")
                            if isinstance(cs, dict)
                            else getattr(cs, "livermore_state_4h", None)
                        )
                        if _lsm_4h:
                            _bullish_states = (
                                "MAIN_UP",
                                "NATURAL_RETRACEMENT",
                                "SECONDARY_RETRACEMENT",
                            )
                            _bearish_states = (
                                "MAIN_DOWN",
                                "NATURAL_REBOUND",
                                "SECONDARY_REBOUND",
                            )
                            _lsm_bull = _lsm_4h in _bullish_states
                            _lsm_bear = _lsm_4h in _bearish_states

                            if buy_score > 0:
                                if _lsm_bull:
                                    buy_score = min(buy_score + 0.15 * weight, weight)
                                    buy_exp += f" +LSM_CONFIRMED({_lsm_4h})"
                                    self._last_trend_judge_tag = "LSM_CONFIRMED"
                                elif _lsm_bear:
                                    buy_score = max(buy_score - 0.20 * weight, 0.0)
                                    buy_exp += f" -LSM_CONTRARIAN({_lsm_4h})"
                                    self._last_trend_judge_tag = "LSM_CONTRARIAN"
                                else:
                                    self._last_trend_judge_tag = "LSM_NEUTRAL"

                            if sell_score > 0:
                                if _lsm_bear:
                                    sell_score = min(sell_score + 0.15 * weight, weight)
                                    sell_exp += f" +LSM_CONFIRMED({_lsm_4h})"
                                    self._last_trend_judge_tag = "LSM_CONFIRMED"
                                elif _lsm_bull:
                                    sell_score = max(sell_score - 0.20 * weight, 0.0)
                                    sell_exp += f" -LSM_CONTRARIAN({_lsm_4h})"
                                    self._last_trend_judge_tag = "LSM_CONTRARIAN"
                                else:
                                    self._last_trend_judge_tag = "LSM_NEUTRAL"
            except Exception as _lsm_e:
                logger.debug(f"[TREND] L7 Livermore confirmation skipped: {_lsm_e}")

            # ── O2a: Orphan signal wiring — slopes_aligned, conviction_dying ──
            # Pre-computed Confluence signals that the trend judge was ignoring
            # while independently re-deriving similar information.
            if governor_data is not None:
                _cs_t = (governor_data or {}).get("composite_state")
                _slopes_aligned = bool(
                    getattr(_cs_t, "slopes_aligned", False) if not isinstance(_cs_t, dict)
                    else _cs_t.get("slopes_aligned", False)
                )
                _conviction_dying = bool(
                    getattr(_cs_t, "conviction_dying", False) if not isinstance(_cs_t, dict)
                    else _cs_t.get("conviction_dying", False)
                )

                # slopes_aligned = 1H and 4H EMA slopes agree in direction.
                # The field is a boolean with no direction stored, so we infer
                # direction from the trend judge's own EMA scores: if buy_score
                # > sell_score the EMA picture is bullish, so aligned slopes
                # confirm the bull — and vice versa. This prevents bearish
                # slope alignment from boosting a buy signal.
                if _slopes_aligned:
                    _slopes_bullish = buy_score > sell_score
                    if _slopes_bullish and buy_score > 0:
                        buy_score = min(weight, buy_score * 1.10)
                        buy_exp  += " +slopes_aligned_bull"
                    elif not _slopes_bullish and sell_score > 0:
                        sell_score = min(weight, sell_score * 1.10)
                        sell_exp  += " +slopes_aligned_bear"

                # conviction_dying = candle body momentum fading. Discount
                # both directions — a fading trend is worth less than a fresh
                # one regardless of which way it's going.
                if _conviction_dying:
                    buy_score  = buy_score  * 0.75
                    sell_score = sell_score * 0.75
                    if buy_score > 0:
                        buy_exp  += " -conviction_dying"
                    if sell_score > 0:
                        sell_exp += " -conviction_dying"

            return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

        except Exception as e:
            logger.error(f"[TREND] Error: {e}")
            return 0.0, 0.0, {"buy": f"TREND: Error", "sell": f"TREND: Error"}

    def _judge_structure_bidirectional(
        self,
        df: pd.DataFrame,
        is_breakout_mode: bool,
        weight: float,
        adx: float = 20.0,
        governor_data: Optional[Dict] = None,
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 2: STRUCTURE (Bidirectional & Adaptive)
        Phase 3B: Reads pre-computed CompositeState fields when available.
        MRS: "STRUCTURE judge reads bos_detected, level_defended,
              rejection_at_level, failed_breakout from CompositeState
              directly. No redundant raw-OHLCV recomputation in the judge."
        Fallback: legacy BreakRetestValidator + raw OHLCV when composite_state absent.
        """
        try:
            buy_score, sell_score = 0.0, 0.0
            buy_exp = "STRUCT BUY: ❌ No signal"
            sell_exp = "STRUCT SELL: ❌ No signal"

            cs = (governor_data or {}).get("composite_state") if governor_data else None

            if cs:
                # ── Phase 3B: CompositeState reads ────────────────────────────
                # composite_state may be a CompositeState dataclass or a dict;
                # use getattr with fallback to handle both.
                def _cs(attr, default=False):
                    if isinstance(cs, dict):
                        return cs.get(attr, default)
                    return getattr(cs, attr, default)

                bos_detected = bool(_cs("bos_detected", False))
                choch_detected = bool(_cs("choch_detected", False))
                level_defended = bool(_cs("level_defended", False))
                rej_at_level = bool(_cs("rejection_at_level", False))
                failed_breakout = bool(_cs("failed_breakout", False))
                sweep_direction = int(_cs("sweep_direction", 0))
                is_bullish_regime = bool((governor_data or {}).get("is_bullish", False))

                # ── BUY scoring ───────────────────────────────────────────────
                if (
                    bos_detected
                    and (is_bullish_regime or is_breakout_mode)
                    and not failed_breakout
                ):
                    buy_score = weight
                    buy_exp = f"STRUCT BUY: ✅ BOS confirmed ({weight:.1f})"
                elif level_defended:
                    # Scale credit to defense strength when available.
                    # Previously all level defenses got the same 0.7× credit
                    # regardless of how hard the level was defended.
                    _def_strength = float(
                        getattr(cs, "defense_strength", 0.0) if not isinstance(cs, dict)
                        else cs.get("defense_strength", 0.0)
                    )
                    if _def_strength >= 0.75:
                        buy_score = weight * 0.9   # strong defense → near-full credit
                        buy_exp   = f"STRUCT BUY: ✅ Strong defense (str={_def_strength:.2f}, {buy_score:.1f})"
                    elif _def_strength >= 0.50:
                        buy_score = weight * 0.7   # current behaviour unchanged
                        buy_exp   = f"STRUCT BUY: ⚠️ Level defended (str={_def_strength:.2f}, {buy_score:.1f})"
                    else:
                        buy_score = weight * 0.5   # weak defense → reduced credit
                        buy_exp   = f"STRUCT BUY: ⚠️ Weak defense (str={_def_strength:.2f}, {buy_score:.1f})"
                elif is_breakout_mode and not failed_breakout and len(df) >= 21:
                    # Donchian breakout detected but no SMC BOS yet — partial credit
                    _h20 = df["high"].iloc[-21:-1].max()
                    if float(df["close"].iloc[-1]) > _h20:
                        buy_score = weight * 0.8
                        buy_exp = f"STRUCT BUY: ⚡ Donchian breakout, no SMC BOS ({buy_score:.1f})"
                    else:
                        buy_exp = "STRUCT BUY: ❌ No structural signal"
                else:
                    buy_exp = "STRUCT BUY: ❌ No structural signal"
                # Bullish spring bonus (wick swept lows, recovered above level)
                if sweep_direction == 1 and buy_score < weight:
                    buy_score = min(buy_score + 0.3 * weight, weight)
                    buy_exp += " +spring"

                # ── SELL scoring ──────────────────────────────────────────────
                if failed_breakout:
                    sell_score = weight
                    sell_exp = f"STRUCT SELL: ✅ Failed breakout ({weight:.1f})"
                elif rej_at_level:
                    sell_score = weight * 0.7
                    sell_exp = f"STRUCT SELL: ⚠️ Rejected at level ({sell_score:.1f})"
                elif is_breakout_mode and len(df) >= 21:
                    # Donchian breakdown detected but no SMC failed_breakout yet — partial credit
                    _l20 = df["low"].iloc[-21:-1].min()
                    if float(df["close"].iloc[-1]) < _l20:
                        sell_score = weight * 0.8
                        sell_exp = f"STRUCT SELL: ⚡ Donchian breakdown, no SMC BOS ({sell_score:.1f})"
                    else:
                        sell_exp = "STRUCT SELL: ❌ No structural signal"
                else:
                    sell_exp = "STRUCT SELL: ❌ No structural signal"
                # Bearish upthrust bonus (wick swept highs, rejected back below)
                if sweep_direction == -1 and sell_score < weight:
                    sell_score = min(sell_score + 0.3 * weight, weight)
                    sell_exp += " +upthrust"
                # Change of character in bull regime = potential top
                if choch_detected and is_bullish_regime:
                    sell_score = min(sell_score + 0.3 * weight, weight)
                    sell_exp += " +CHoCH"

            else:
                # ── Fallback: legacy BreakRetestValidator + raw OHLCV ─────────
                br_res = self.break_retest_validator.validate(df, self.asset_type)
                if br_res.is_valid:
                    if br_res.type == "BULLISH_RETEST":
                        buy_score = min(buy_score + 0.4 * weight, weight)
                        buy_exp = f"STRUCT BUY: ✨ {br_res.explanation} (Bonus)"
                    elif br_res.type == "BEARISH_RETEST":
                        sell_score = min(sell_score + 0.4 * weight, weight)
                        sell_exp = f"STRUCT SELL: ✨ {br_res.explanation} (Bonus)"

                current_price = float(df["close"].iloc[-1])
                if is_breakout_mode:
                    if len(df) < 21:
                        return (
                            0.0,
                            0.0,
                            {
                                "buy": "STRUCT: Need 21 bars",
                                "sell": "STRUCT: Need 21 bars",
                            },
                        )
                    high_20 = df["high"].iloc[-21:-1].max()
                    low_20 = df["low"].iloc[-21:-1].min()
                    if current_price > high_20:
                        buy_score = weight
                        buy_exp = f"STRUCT BUY: ✅ Breakout ({weight:.1f}) - Price > 20-bar high ${high_20:.2f}"
                    if current_price < low_20:
                        sell_score = weight
                        sell_exp = f"STRUCT SELL: ✅ Breakdown ({weight:.1f}) - Price < 20-bar low ${low_20:.2f}"
                else:
                    if not self.ai_validator:
                        return (
                            0.0,
                            0.0,
                            {
                                "buy": "STRUCT: AI disabled",
                                "sell": "STRUCT: AI disabled",
                            },
                        )
                    high, low, close = (
                        df["high"].values,
                        df["low"].values,
                        df["close"].values,
                    )
                    atr_fast = ta.ATR(high, low, close, timeperiod=14)[-1]
                    multiplier = 2.5 if adx > 25 else 1.5
                    threshold_pct = (multiplier * atr_fast) / current_price
                    sr_buy = self.ai_validator._check_support_resistance_fixed(
                        asset=self.asset_type,
                        df=df,
                        current_price=current_price,
                        signal=1,
                        threshold=threshold_pct,
                    )
                    if sr_buy.get("near_level"):
                        lvl = sr_buy.get("nearest_level", 0)
                        buy_score = weight
                        buy_exp = f"STRUCT BUY: ✅ At Support ({weight:.1f}) - Near level ${lvl:.2f} (±{multiplier}*ATR)"
                    else:
                        buy_exp = "STRUCT BUY: ❌ No support nearby"
                    sr_sell = self.ai_validator._check_support_resistance_fixed(
                        asset=self.asset_type,
                        df=df,
                        current_price=current_price,
                        signal=-1,
                        threshold=threshold_pct,
                    )
                    if sr_sell.get("near_level"):
                        lvl = sr_sell.get("nearest_level", 0)
                        sell_score = weight
                        sell_exp = f"STRUCT SELL: ✅ At Resistance ({weight:.1f}) - Near level ${lvl:.2f} (±{multiplier}*ATR)"
                    else:
                        sell_exp = "STRUCT SELL: ❌ No resistance nearby"

            # ── O2c: VWAP as institutional reference level ────────────────
            # VWAP direction matters: price above VWAP = VWAP is support
            # (confirms buys). Price below VWAP = VWAP is resistance
            # (confirms sells). The original code boosted both blindly,
            # which could boost a buy when price is pressing into VWAP
            # resistance from below. Fixed to check side.
            _vwap = float(
                getattr(cs, "vwap_price", 0.0) if not isinstance(cs, dict)
                else cs.get("vwap_price", 0.0)
            ) if cs else 0.0
            _vwap_dist_atr = float(
                getattr(cs, "distance_to_vwap_atr", 999.0) if not isinstance(cs, dict)
                else cs.get("distance_to_vwap_atr", 999.0)
            ) if cs else 999.0
            if _vwap > 0 and _vwap_dist_atr < 0.5:
                _current_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
                _price_above_vwap = _current_close > _vwap
                # Only boost the direction VWAP supports given price location.
                if _price_above_vwap and buy_score > 0:
                    buy_score = min(buy_score + 0.15 * weight, weight)
                    buy_exp  += f" +above_VWAP_support({_vwap_dist_atr:.2f}ATR)"
                elif not _price_above_vwap and sell_score > 0:
                    sell_score = min(sell_score + 0.15 * weight, weight)
                    sell_exp  += f" +below_VWAP_resistance({_vwap_dist_atr:.2f}ATR)"

            # Cap at 1.0 — the scorecard declares STRUCTURE max as 1.0/1.0.
            # When w_structure=1.5 (SLIGHTLY_BEARISH dynamic weight) the raw scores
            # overflow to 1.5 which inflates the total past what the UI shows and
            # allows trades to clear thresholds they shouldn't. Hard cap here.
            buy_score = min(buy_score, 1.0)
            sell_score = min(sell_score, 1.0)
            return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

        except Exception as e:
            logger.error(f"[STRUCTURE] Error: {e}", exc_info=True)
            return 0.0, 0.0, {"buy": "STRUCT: Error", "sell": "STRUCT: Error"}

    def _judge_momentum_bidirectional(
        self,
        df: pd.DataFrame,
        is_bull: bool,
        is_breakout_mode: bool,
        weight: float,
        adx: float,
        governor_data: Optional[Dict] = None,
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 3: MOMENTUM (Bidirectional & Adaptive)
        """
        try:
            if weight == 0:
                return 0.0, 0.0, {"buy": "MOM: Disabled", "sell": "MOM: Disabled"}

            # ✅ TASK 19: Super-Cycle Recalibration (Phase 3)
            # Reason: Real BTC super-trends start at ADX 30-32. 35 is too late.
            if adx > 32:
                buy_score = weight if is_bull else 0.0
                sell_score = weight if not is_bull else 0.0
                buy_exp = (
                    f"MOM BUY: ✅ Super-Cycle ({buy_score:.1f}) - ADX {adx:.1f} > 32"
                    if is_bull
                    else "MOM BUY: ❌ Dead in Bear Super-Cycle"
                )
                sell_exp = (
                    f"MOM SELL: ✅ Super-Cycle ({sell_score:.1f}) - ADX {adx:.1f} > 32"
                    if not is_bull
                    else "MOM SELL: ❌ Dead in Bull Super-Cycle"
                )
                return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

            features_mr = self.s_mean_reversion.generate_features(df.tail(100))
            if features_mr.empty:
                return 0.0, 0.0, {"buy": "MOM: No data", "sell": "MOM: No data"}

            rsi = features_mr.iloc[-1].get("rsi", 50)

            # Config values
            bullish_min, bullish_max = self.config["rsi_bullish_zone"]
            bearish_min, bearish_max = self.config["rsi_bearish_zone"]
            oversold = self.config["rsi_oversold_bonus"]
            overbought = self.config["rsi_overbought_bonus"]

            buy_score = 0.0
            sell_score = 0.0
            buy_exp = f"MOM BUY: ❌ No credit - RSI {rsi:.1f}"
            sell_exp = f"MOM SELL: ❌ No credit - RSI {rsi:.1f}"

            # ✨ NEW: RSI Divergence Engine
            div_res = self.divergence_detector.analyze(df)
            if div_res.type != "NONE":
                if div_res.type == "BULLISH":
                    buy_score = min(buy_score + 0.3 * weight, weight)
                    buy_exp += f" | ✨ {div_res.explanation}"
                elif div_res.type == "BEARISH":
                    sell_score = min(sell_score + 0.3 * weight, weight)
                    sell_exp += f" | ✨ {div_res.explanation}"
                elif div_res.type == "HIDDEN_BULLISH":
                    buy_score = min(buy_score + 0.2 * weight, weight)
                    buy_exp += f" | 🚀 {div_res.explanation}"
                elif div_res.type == "HIDDEN_BEARISH":
                    sell_score = min(sell_score + 0.2 * weight, weight)
                    sell_exp += f" | 🚀 {div_res.explanation}"

            if is_breakout_mode:
                # --- BREAKOUT LOGIC (Momentum Continuation) ---
                if rsi < oversold:
                    sell_score = weight
                    sell_exp = f"MOM SELL: ✅ Breakout ({weight:.1f}) - RSI {rsi:.1f} shows downside momentum"
                if rsi > overbought:
                    buy_score = weight
                    buy_exp = f"MOM BUY: ✅ Breakout ({weight:.1f}) - RSI {rsi:.1f} shows upside momentum"

            else:
                # --- NORMAL LOGIC (Trend Alignment) ---
                # Award points for price being in the 'value' zone relative to RSI
                if bullish_min <= rsi <= bullish_max:
                    buy_score = weight
                    buy_exp = f"MOM BUY: ✅ Full ({weight:.1f}) - RSI {rsi:.1f} in bullish zone"

                if bearish_min <= rsi <= bearish_max:
                    sell_score = weight
                    sell_exp = f"MOM SELL: ✅ Full ({weight:.1f}) - RSI {rsi:.1f} in bearish zone"

            # MACD confirmation
            if self.config["macd_confirmation"]:
                macd = features_mr.iloc[-1].get("macd", 0)
                macd_signal = features_mr.iloc[-1].get("macd_signal", 0)

                if buy_score > 0 and macd > macd_signal:
                    buy_score = min(buy_score + 0.2, weight)
                    buy_exp += " +MACD"

                if sell_score > 0 and macd < macd_signal:
                    sell_score = min(sell_score + 0.2, weight)
                    sell_exp += " +MACD"

            # ── Phase 3B: CompositeState momentum enrichment ───────────────────
            # MRS: "conviction_dying reduces momentum score regardless of direction.
            #       vpd_diverging reduces score in trend direction.
            #       cvd_trend gives a small directional boost."
            cs = (governor_data or {}).get("composite_state") if governor_data else None
            if cs:

                def _cs(attr, default=False):
                    if isinstance(cs, dict):
                        return cs.get(attr, default)
                    return getattr(cs, attr, default)

                conviction_dying = bool(_cs("conviction_dying", False))
                vpd_diverging = bool(_cs("vpd_diverging", False))
                cvd_trend = int(_cs("cvd_trend", 0))

                if conviction_dying:
                    # Shrinking candle bodies = conviction dying; penalise both
                    penalty = 0.20 * weight
                    buy_score = max(0.0, buy_score - penalty)
                    sell_score = max(0.0, sell_score - penalty)
                    buy_exp += " -conviction_dying"
                    sell_exp += " -conviction_dying"

                if vpd_diverging:
                    # Price moves but volume disagrees — penalise trend direction
                    div_penalty = 0.15 * weight
                    if is_bull:
                        buy_score = max(0.0, buy_score - div_penalty)
                        buy_exp += " -vpd_div"
                    else:
                        sell_score = max(0.0, sell_score - div_penalty)
                        sell_exp += " -vpd_div"

                if cvd_trend > 0:
                    buy_score = min(buy_score + 0.15 * weight, weight)
                    buy_exp += " +cvd"
                elif cvd_trend < 0:
                    sell_score = min(sell_score + 0.15 * weight, weight)
                    sell_exp += " +cvd"

            # ── ADX slope modifier ────────────────────────────────────────────
            # The raw ADX value (passed in as `adx`) is a snapshot.  The slope
            # tells us whether trend strength is building or decaying — critical
            # information for momentum scoring.
            #
            # Impact table (applied as a multiplier on the DOMINANT side only,
            # capped so the score never exceeds the judge weight):
            #   RISING_FAST  → +20% of weight  (trend accelerating)
            #   RISING       → +10% of weight  (trend building)
            #   FLAT         →   0%            (no change)
            #   FALLING      → −15% of weight  (trend losing steam)
            #   FALLING_FAST → −25% of weight  (trend collapsing)
            #
            # Also: when ADX is below 20 AND falling, both directions are
            # penalised (choppy market, signals are noise).
            # ─────────────────────────────────────────────────────────────────
            try:
                _adx_col = df["adx"].dropna().values if "adx" in df.columns else None
                _slope = compute_adx_slope(_adx_col)
                _regime = _slope["regime"]

                _SLOPE_DELTA = {
                    "RISING_FAST": +0.20 * weight,
                    "RISING": +0.10 * weight,
                    "FLAT": 0.0,
                    "FALLING": -0.15 * weight,
                    "FALLING_FAST": -0.25 * weight,
                }
                _delta = _SLOPE_DELTA.get(_regime, 0.0)

                # ── Livermore-aware ADX slope interpretation ──────────────────
                # The default delta table assumes MAIN states where rising ADX
                # aligns with the trade direction.  In SECONDARY and NATURAL
                # states the relationship inverts or becomes ambiguous.
                #
                # SECONDARY_RETRACEMENT / SECONDARY_REBOUND (counter-trend setup):
                #   Rising ADX = the ADVERSE move is accelerating → flip delta so
                #   rising becomes a penalty and falling becomes a boost (exhaustion
                #   is what you want before a counter-trend entry).
                #
                # NATURAL_RETRACEMENT / NATURAL_REBOUND (pullback within trend):
                #   Rising ADX during a retracement is ambiguous — could be the
                #   correction intensifying OR the trend reasserting.  Cap delta
                #   to 0 (FLAT behaviour) to avoid false confidence either way.
                #
                # MAIN_UP / MAIN_DOWN: leave as-is — rising ADX is unambiguously
                # bullish for the trend direction.
                # ─────────────────────────────────────────────────────────────
                _lsm_1h = None
                if cs:
                    _lsm_1h = _cs("livermore_state_1h", None)

                _lsm_context = "MAIN"  # default
                if _lsm_1h in ("SECONDARY_RETRACEMENT", "SECONDARY_REBOUND"):
                    _lsm_context = "SECONDARY"
                elif _lsm_1h in ("NATURAL_RETRACEMENT", "NATURAL_REBOUND"):
                    _lsm_context = "NATURAL"

                if _lsm_context == "SECONDARY":
                    # Flip: rising becomes penalty, falling becomes boost
                    _delta = -_delta
                elif _lsm_context == "NATURAL":
                    # Ambiguous: neutralise any boost, keep penalties
                    if _delta > 0:
                        _delta = 0.0

                _lsm_tag = (
                    f"|lsm={_lsm_1h}" if _lsm_1h and _lsm_context != "MAIN" else ""
                )

                if _delta != 0.0:
                    if buy_score >= sell_score and buy_score > 0:
                        buy_score = float(np.clip(buy_score + _delta, 0.0, weight))
                        _sign = "+" if _delta > 0 else ""
                        buy_exp += (
                            f" ADX-slope:{_regime}{_lsm_tag}"
                            f"(s={_slope['short_slope']:+.1f}"
                            f" m={_slope['med_slope']:+.1f}"
                            f" {_sign}{_delta:.2f})"
                        )
                    elif sell_score > buy_score and sell_score > 0:
                        sell_score = float(np.clip(sell_score + _delta, 0.0, weight))
                        _sign = "+" if _delta > 0 else ""
                        sell_exp += (
                            f" ADX-slope:{_regime}{_lsm_tag}"
                            f"(s={_slope['short_slope']:+.1f}"
                            f" m={_slope['med_slope']:+.1f}"
                            f" {_sign}{_delta:.2f})"
                        )

                # Choppy-market penalty: ADX low AND falling → both sides hurt
                if adx < 20 and _regime in ("FALLING", "FALLING_FAST"):
                    _chop_penalty = 0.10 * weight
                    buy_score = max(0.0, buy_score - _chop_penalty)
                    sell_score = max(0.0, sell_score - _chop_penalty)
                    buy_exp += f" -chop(ADX{adx:.0f}↓)"
                    sell_exp += f" -chop(ADX{adx:.0f}↓)"

            except Exception:
                pass  # ADX slope is non-critical; never block a trade on failure

            return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

        except Exception as e:
            logger.error(f"[MOMENTUM] Error: {e}", exc_info=True)
            return 0.0, 0.0, {"buy": "MOM: Error", "sell": "MOM: Error"}

    def _judge_pattern_bidirectional(
        self, df: pd.DataFrame, weight: float, governor_data: Optional[Dict] = None
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 4: PATTERN (Bidirectional)
        Phase 3B: Reads institutional_pattern from performance aggregator first.
        MRS: "PATTERN judge reads institutional_pattern from performance aggregator.
              ACCUMULATION confirmed = PATTERN judge scores it.
              This connection was completely absent."
        Fallback: ai_validator._check_pattern() when institutional_pattern is None.
        """
        try:
            buy_score = 0.0
            sell_score = 0.0

            # ── Phase 3B: institutional_pattern from performance aggregator ────
            _gd_composite = (
                (governor_data or {}).get("composite_state") if governor_data else None
            )
            if isinstance(_gd_composite, dict):
                inst_pattern = _gd_composite.get("institutional_pattern")
            else:
                inst_pattern = getattr(_gd_composite, "institutional_pattern", None)
            if inst_pattern is not None:
                if inst_pattern == "ACCUMULATION":
                    buy_score = weight
                    return (
                        buy_score,
                        sell_score,
                        {
                            "buy": f"PATTERN BUY: ✅ ACCUMULATION ({weight:.1f}) — institutional buying",
                            "sell": "PATTERN SELL: ❌ No pattern (ACCUMULATION context)",
                        },
                    )
                elif inst_pattern == "DISTRIBUTION":
                    sell_score = weight
                    return (
                        buy_score,
                        sell_score,
                        {
                            "buy": "PATTERN BUY: ❌ No pattern (DISTRIBUTION context)",
                            "sell": f"PATTERN SELL: ✅ DISTRIBUTION ({weight:.1f}) — institutional selling",
                        },
                    )
                elif inst_pattern in ("COMPRESSION", "CONSOLIDATION"):
                    buy_score = weight * 0.5
                    sell_score = weight * 0.5
                    return (
                        buy_score,
                        sell_score,
                        {
                            "buy": f"PATTERN BUY: ⚠️ {inst_pattern} ({buy_score:.1f}) — range bound",
                            "sell": f"PATTERN SELL: ⚠️ {inst_pattern} ({sell_score:.1f}) — range bound",
                        },
                    )
                # Unknown institutional_pattern string → fall through to ai_validator

            # ── Fallback: ai_validator pattern check (legacy path) ─────────────
            if not self.ai_validator:
                return (
                    0.0,
                    0.0,
                    {"buy": "PATTERN: AI disabled", "sell": "PATTERN: AI disabled"},
                )

            # Check bullish pattern
            pattern_buy = self.ai_validator._check_pattern(
                df=df,
                signal=1,
                min_confidence=self.config["pattern_confidence_min"],
            )

            if pattern_buy.get("pattern_confirmed"):
                conf = pattern_buy.get("confidence", 0)
                name = pattern_buy.get("pattern_name", "Unknown")

                if conf > 0.75:
                    buy_score = weight
                    buy_exp = (
                        f"PATTERN BUY: ✅ Full ({weight:.1f}) - {name} ({conf:.0%})"
                    )
                else:
                    buy_score = weight * 0.8
                    buy_exp = f"PATTERN BUY: ⚠️ Partial ({buy_score:.1f}) - {name} ({conf:.0%})"
            else:
                buy_exp = "PATTERN BUY: ❌ No pattern"

            # Check bearish pattern
            pattern_sell = self.ai_validator._check_pattern(
                df=df,
                signal=-1,
                min_confidence=self.config["pattern_confidence_min"],
            )

            if pattern_sell.get("pattern_confirmed"):
                conf = pattern_sell.get("confidence", 0)
                name = pattern_sell.get("pattern_name", "Unknown")

                if conf > 0.75:
                    sell_score = weight
                    sell_exp = (
                        f"PATTERN SELL: ✅ Full ({weight:.1f}) - {name} ({conf:.0%})"
                    )
                else:
                    sell_score = weight * 0.8
                    sell_exp = f"PATTERN SELL: ⚠️ Partial ({sell_score:.1f}) - {name} ({conf:.0%})"
            else:
                sell_exp = "PATTERN SELL: ❌ No pattern"

            return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return 0.0, 0.0, {"buy": "PATTERN: Error", "sell": "PATTERN: Error"}

    def _judge_volume_bidirectional(
        self, df: pd.DataFrame, weight: float, governor_data: Optional[Dict] = None
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 5: VOLUME (Same for both directions)
        """
        if self.asset_type in ["GOLD", "EURUSD", "EURJPY", "USTEC", "USOIL", "GBPAUD"]:
            return (
                0.5 * weight,
                0.5 * weight,
                {
                    "buy": "VOL: Neutral (MT5 tick volume unreliable)",
                    "sell": "VOL: Neutral (MT5 tick volume unreliable)",
                },
            )

        try:
            if "volume" not in df.columns:
                return 0.0, 0.0, {"buy": "VOL: No data", "sell": "VOL: No data"}

            volume_ma_period = self.config["volume_ma_period"]
            current_volume = df["volume"].iloc[-1]
            volume_ma = df["volume"].rolling(volume_ma_period).mean().iloc[-1]

            vol_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0

            # Same scoring for both directions
            if vol_ratio > 1.5:
                score = weight
                exp = f"VOLUME: ✅ Strong ({weight:.1f}) - {vol_ratio:.1f}x avg"
            elif vol_ratio > 1.0:
                score = weight * 0.7
                exp = f"VOLUME: ⚠️ Partial ({score:.1f}) - {vol_ratio:.1f}x avg"
            else:
                score = 0.0
                exp = f"VOLUME: ❌ Below avg ({vol_ratio:.1f}x)"

            # ── O2b: Orphan signal wiring — absorption_detected, vpd_diverging ──
            _cs_v = (governor_data or {}).get("composite_state") if governor_data else None
            if _cs_v is not None:
                _absorption = bool(
                    getattr(_cs_v, "absorption_detected", False) if not isinstance(_cs_v, dict)
                    else _cs_v.get("absorption_detected", False)
                )
                _vpd_diverging = bool(
                    getattr(_cs_v, "vpd_diverging", False) if not isinstance(_cs_v, dict)
                    else _cs_v.get("vpd_diverging", False)
                )

                # absorption_detected = institutional orders absorbing at price.
                # High-conviction Wyckoff signal regardless of raw volume ratio.
                # Boosts both directions equally since absorption confirms
                # institutional interest without specifying which way.
                if _absorption:
                    score = min(weight, score + 0.30 * weight)
                    exp  += " +absorption"

                # vpd_diverging = volume not supporting price movement.
                # Warning that the current move lacks conviction behind it.
                # Penalise both directions.
                if _vpd_diverging:
                    score = score * 0.70
                    exp  += " -vpd_diverging"

                # order_book_wall_detected = large resting orders at price.
                # Indicates institutional positioning. Modest buy/sell boost
                # since a wall can be either supply or demand depending on
                # which side it sits — treat as generic institutional interest.
                _ob_wall = bool(
                    getattr(_cs_v, "order_book_wall_detected", False) if not isinstance(_cs_v, dict)
                    else _cs_v.get("order_book_wall_detected", False)
                )
                if _ob_wall:
                    score = min(weight, score + 0.20 * weight)
                    exp  += " +ob_wall"

            return score, score, {"buy": exp, "sell": exp}

        except Exception as e:
            logger.error(f"[VOLUME] Error: {e}")
            return 0.0, 0.0, {"buy": "VOL: Error", "sell": "VOL: Error"}

    def _judge_reversion_bidirectional(
        self, df: pd.DataFrame, weight: float
    ) -> Tuple[float, float, Dict]:
        """
        JUDGE 6: REVERSION (Structural Mean Reversion)
        Objective: Prevents catching falling knives by requiring RSI reversal + price break.
        """
        try:
            if len(df) < 5:
                return 0.0, 0.0, {"buy": "REV: No data", "sell": "REV: No data"}

            # Calculate RSI
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            rsi_series = ta.RSI(close, timeperiod=14)
            current_rsi = rsi_series[-1]
            previous_rsi = rsi_series[-2]

            current_close = close[-1]
            previous_high = high[-2]
            previous_low = low[-2]

            buy_score = 0.0
            sell_score = 0.0
            buy_exp = "REV BUY: ❌ No structural reversal"
            sell_exp = "REV SELL: ❌ No structural reversal"

            # LONG ENTRY CONDITIONS
            if (
                previous_rsi < 30
                and current_rsi >= 30
                and current_close > previous_high
            ):
                buy_score = weight
                buy_exp = f"REV BUY: ✅ Structural Reversal ({weight:.1f}) - RSI {previous_rsi:.1f}->{current_rsi:.1f} + Price > Prev High"

            # SHORT ENTRY CONDITIONS
            if previous_rsi > 70 and current_rsi <= 70 and current_close < previous_low:
                sell_score = weight
                sell_exp = f"REV SELL: ✅ Structural Reversal ({weight:.1f}) - RSI {previous_rsi:.1f}->{current_rsi:.1f} + Price < Prev Low"

            return buy_score, sell_score, {"buy": buy_exp, "sell": sell_exp}

        except Exception as e:
            logger.error(f"[REVERSION] Error: {e}")
            return 0.0, 0.0, {"buy": "REV: Error", "sell": "REV: Error"}

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Leverage existing EMA strategy for regime detection"""
        try:
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            is_bull = ema_signal >= 0
            return is_bull, ema_conf
        except Exception as e:
            logger.error(f"[REGIME] Error: {e}")
            return False, 0.5

    def _detect_breakout_state(
        self,
        df: pd.DataFrame,
        adx_threshold: int = 25,
        volume_surge_factor: float = 1.5,
        donchian_period: int = 20,
    ) -> bool:
        """
        Detects a market breakout state based on a confluence of indicators.
        A breakout is confirmed if ALL of the following conditions are true:
        1. Strength: ADX is above a specified threshold (e.g., 25), indicating strong trend.
        2. Participation: Volume is significantly higher than its rolling average, showing conviction.
        3. Structure: Price has broken a recent high or low, confirming a structural shift.

        This method provides a binary flag (is_breakout_mode) to switch the logic of other judges.

        Args:
            df (pd.DataFrame): The market data.
            adx_threshold (int): The ADX value required to confirm trend strength.
            volume_surge_factor (float): The multiplier for volume vs. its rolling average.
            donchian_period (int): The lookback period for the Donchian channel breakout.

        Returns:
            bool: True if the market is in a breakout state, False otherwise.
        """
        try:
            if len(df) < (donchian_period + 1):
                return False  # Not enough data to determine breakout state

            # 1. Strength Check: ADX > 25
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            adx = ta.ADX(
                highs, lows, closes, timeperiod=14
            )  # Standard ADX period is 14
            latest_adx = adx[-1]
            is_strong_trend = latest_adx > adx_threshold

            if not is_strong_trend:
                if self.detailed_logging:
                    logger.info(
                        f"[BREAKOUT] Condition not met: ADX {latest_adx:.1f} <= {adx_threshold}"
                    )
                return False

            # 2. Participation Check: Volume >= 1.5x average
            volume_ma = df["volume"].rolling(donchian_period).mean().iloc[-1]
            current_volume = df["volume"].iloc[-1]
            is_volume_surge = current_volume >= (volume_ma * volume_surge_factor)

            if not is_volume_surge:
                if self.detailed_logging:
                    logger.info(
                        f"[BREAKOUT] Condition not met: Volume {current_volume:.0f} < {volume_ma * volume_surge_factor:.0f}"
                    )
                return False

            # 3. Structure Check: Price breaks Donchian High/Low
            # ✅ PHASE 4: STOP-HUNT CEILING LOGIC
            highs_20 = df["high"].iloc[-donchian_period - 1 : -1]
            closes_20 = df["close"].iloc[-donchian_period - 1 : -1]
            lows_20 = df["low"].iloc[-donchian_period - 1 : -1]

            highest_wick = highs_20.max()
            highest_close = closes_20.max()
            lowest_wick = lows_20.min()
            lowest_close = closes_20.min()

            atr = ta.ATR(highs, lows, closes, timeperiod=14)[-1]
            latest_close = closes[-1]

            is_structure_broken = False

            # Bullish Breakout Check
            if (highest_wick - highest_close) < (0.25 * atr):
                # CEILING DETECTED: Must exceed the wick
                if latest_close > highest_wick:
                    is_structure_broken = True
            else:
                # NORM: Exceeding highest close is sufficient
                if latest_close > highest_close:
                    is_structure_broken = True

            # Bearish Breakdown Check (Symmetric)
            if not is_structure_broken:
                if (lowest_close - lowest_wick) < (0.25 * atr):
                    # FLOOR DETECTED: Must exceed the wick
                    if latest_close < lowest_wick:
                        is_structure_broken = True
                else:
                    if latest_close < lowest_close:
                        is_structure_broken = True

            if not is_structure_broken:
                if self.detailed_logging:
                    logger.info(f"[BREAKOUT] Condition not met: Structure holding.")
                return False

            # If all conditions are met, we are in a breakout state.
            logger.info(
                f"🔥 BREAKOUT STATE DETECTED: ADX={latest_adx:.1f}, Vol Ratio={current_volume/volume_ma:.1f}x, Price broke structure."
            )
            return True

        except Exception as e:
            logger.error(
                f"[BREAKOUT] Error detecting breakout state: {e}", exc_info=True
            )
            return False  # Fail-safe to False

    def _log_decision_bidirectional(self, details: Dict):
        """Log council decision with bidirectional breakdown"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🏛️  COUNCIL DECISION - {details['regime']}")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {details['timestamp']}")
        logger.info(f"")

        # Show both BUY and SELL scores
        logger.info(f"BUY SCORECARD (Total: {details['buy_total']:.2f}/5.0):")
        for judge, score in details["buy_scores"].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")

        logger.info(f"")
        logger.info(f"SELL SCORECARD (Total: {details['sell_total']:.2f}/5.0):")
        for judge, score in details["sell_scores"].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")

        logger.info(f"")
        logger.info(f"DECISION: {details['decision_type']}")
        logger.info(f"SIGNAL:   {details['signal']:+2d}")
        logger.info(
            f"SCORE:    {details['total_score']:.2f} / {details['required_score']:.2f}"
        )
        logger.info("=" * 80)
        logger.info("")

    def _format_ai_validation_for_viz(
        self, final_signal: int, details: dict, df: pd.DataFrame
    ) -> dict:
        """Format AI validation results for visualization"""
        try:
            viz_data = {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "None",
                "pattern_id": None,
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "action": "none",
                "rejection_reasons": [],
                "error": None,
            }

            if not self.ai_validator:
                viz_data["action"] = "ai_disabled"
                return viz_data

            current_price = float(df["close"].iloc[-1])

            # S/R Analysis
            try:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type,
                    df=df,
                    current_price=current_price,
                    signal=final_signal,
                    threshold=self.ai_validator.current_sr_threshold,
                )

                viz_data["sr_analysis"] = {
                    "near_sr_level": sr_result.get("near_level", False),
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }
            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")

            # Pattern Detection — PAT-3: module removed, static stubs
            viz_data["pattern_detected"] = False
            viz_data["pattern_name"] = None
            viz_data["pattern_id"] = None
            viz_data["pattern_confidence"] = 0.0

            # Validation Status
            original_signal = details.get("original_signal", final_signal)

            if final_signal == 0 and original_signal != 0:
                viz_data["validation_passed"] = False
                viz_data["action"] = "rejected"

                reasons = []
                if not viz_data["sr_analysis"]["near_sr_level"]:
                    reasons.append("No nearby S/R level")
                if not viz_data["pattern_detected"]:
                    reasons.append("No pattern detected")
                if (
                    viz_data["pattern_confidence"]
                    < self.ai_validator.current_pattern_threshold
                ):
                    reasons.append(
                        f"Low confidence ({viz_data['pattern_confidence']:.1%})"
                    )
                viz_data["rejection_reasons"] = reasons
            elif final_signal != 0:
                viz_data["validation_passed"] = True
                viz_data["action"] = "approved"
            else:
                viz_data["action"] = "hold"

            return viz_data

        except Exception as e:
            logger.error(f"[VIZ] AI formatting failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "error": str(e),
                "action": "error",
            }

    def get_statistics(self) -> Dict:
        """Return aggregator statistics"""
        total = max(self.stats["total_evaluations"], 1)

        return {
            **self.stats,
            "buy_rate": (self.stats["buy_signals"] / total) * 100,
            "sell_rate": (self.stats["sell_signals"] / total) * 100,
            "hold_rate": (self.stats["hold_signals"] / total) * 100,
            "avg_score_on_trade": (
                np.mean(self.stats["avg_score_on_trade"])
                if self.stats["avg_score_on_trade"]
                else 0.0
            ),
            "avg_score_on_hold": (
                np.mean(self.stats["avg_score_on_hold"])
                if self.stats["avg_score_on_hold"]
                else 0.0
            ),
        }
