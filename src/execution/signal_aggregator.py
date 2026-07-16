"""
Enhanced PerformanceWeightedAggregator with AI Safety Features
===============================================================
IMPROVEMENTS:
- AI circuit breaker to prevent over-filtering
- Regime context passed to AI validator
- Better cold-start handling for regime detection
- AI performance tracking
- Graceful degradation if AI fails
"""

import pandas as pd
import logging
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta, timezone
from src.utils.trap_filter import validate_candle_structure
from src.indicators.divergence import RSIDivergenceDetector
from src.analysis.break_retest import BreakRetestValidator
from src.execution.transition_detector import TransitionDetector

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
    Enhanced Signal Aggregator with World-Class Filters
    ====================================================
    Adds Governor + Volatility + Sniper checks to existing aggregator
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        volume_flow_strategy=None,
        asset_type: str = "BTC",
        config: Dict = None,
        ai_validator=None,
        mtf_integration=None,  # For Governor access
        enable_world_class_filters: bool = True,
        enable_ai_circuit_breaker: bool = False,
        enable_detailed_logging: bool = False,
        strong_signal_bypass_threshold: float = 0.70,
        use_macro_governor: bool = True,
        use_gatekeeper: bool = True,
        state_persistence_path: str = "data/aggregator_state.json",
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.s_volume_flow = volume_flow_strategy
        self.asset_type = asset_type.upper()
        self.use_macro_governor = use_macro_governor
        self.use_gatekeeper = use_gatekeeper

        # Initialize regime tracking
        self.previous_regime = None
        self.regime_initialized = False

        # Logging and Thresholds
        self.detailed_logging = enable_detailed_logging
        self.strong_signal_bypass = strong_signal_bypass_threshold

        # ================================================================
        # AI VALIDATOR SETUP
        # ================================================================
        self.ai_validator = None
        self.ai_enabled = True

        # ✨ NEW: Store MTF integration for Governor
        self.mtf_integration = mtf_integration
        self.enable_filters = enable_world_class_filters

        # ✨ NEW: Initialize filter thresholds
        # Volatility gate: asset-class-specific defaults so FX pairs (EURJPY ATR ~0.07%)
        # aren't permanently blocked by a crypto-calibrated 0.35% threshold.
        #   FX  (EUR*, GBP*, USD*, JPY*, CHF*, AUD*, NZD*, CAD*) → 0.03% (0.0003)
        #   Metals / Indices (XAU, GOLD, USTEC, NAS*, SP5*, GER*) → 0.10% (0.0010)
        #   Crypto (BTC*, ETH*, BNB*)                             → 0.20% (0.0020)
        # Config override still wins if explicitly set.
        _asset_upper = asset_type.upper()
        _is_fx = (
            any(
                fx in _asset_upper
                for fx in ("EUR", "GBP", "USD", "JPY", "CHF", "AUD", "NZD", "CAD")
            )
            and "BTC" not in _asset_upper
            and "ETH" not in _asset_upper
        )
        _is_crypto = any(c in _asset_upper for c in ("BTC", "ETH", "BNB", "SOL", "XRP"))
        _is_metals_indices = any(
            m in _asset_upper
            for m in ("XAU", "GOLD", "USTEC", "NAS", "SP5", "GER", "UK1", "NDX")
        )
        if _is_fx:
            _default_vol_threshold = 0.0003  # 0.03% — FX pairs
        elif _is_metals_indices:
            _default_vol_threshold = 0.0010  # 0.10% — metals / indices
        elif _is_crypto:
            _default_vol_threshold = (
                0.0020  # 0.20% — crypto (relaxed from original 0.35%)
            )
        else:
            _default_vol_threshold = 0.0010  # 0.10% — safe generic fallback

        self.filter_thresholds = {
            "volatility_gate": config.get("world_class_filters", {}).get(
                "volatility_gate_threshold", _default_vol_threshold
            ),
            "min_profit": config.get("world_class_filters", {}).get(
                "min_profit_potential", 0.005
            ),
        }
        # NOTE: sniper_confidence threshold removed — CNN-LSTM sniper disconnected
        # from scoring pipeline in Phase 0B. See MRS §6 Phase 0.

        if self.enable_filters:
            logger.info(f"[FILTERS] World-Class Filters ENABLED for {asset_type}")
            logger.info(
                f"  Volatility Gate: {self.filter_thresholds['volatility_gate']:.3%}"
            )
            logger.info(
                f"  Min Profit:      {self.filter_thresholds['min_profit']:.2%}"
            )
            logger.info(f"  Sniper:          DISCONNECTED (Phase 0B)")

        if ai_validator is not None:
            try:
                # Validate AI pattern miner is properly initialized.
                # Sniper assertions removed — sniper is disconnected from pipeline.
                assert hasattr(
                    ai_validator, "pattern_id_map"
                ), "Pattern mapping missing"
                # FIX (2026-06-24): was `assert len(ai_validator.pattern_id_map) > 0`.
                # hybrid_validator.py's __init__ now hardcodes
                # `self.pattern_id_map = {}` (PAT-1: pattern detection module
                # removed). That made this assertion fail unconditionally —
                # self.ai_validator silently stayed None for EVERY
                # PerformanceWeightedAggregator instance, on every asset, with
                # only a startup-time logger.error/warning ("Pattern mapping
                # empty" / "Continuing without AI validation") and no visible
                # runtime symptom until STEP 8 below: with self.ai_validator
                # falsy, neither the real-validator branch nor the cosmetic
                # _format_ai_validation_for_viz branch ever ran, so
                # ai_validation_details stayed `{}` and main.py's hybrid block
                # reported all 8 required fields "missing" on every single
                # PERFORMANCE-mode cycle (the "[HYBRID] ⚠️ AI validation
                # missing fields" / "Regenerating..." loop). council_aggregator.py
                # has no equivalent pattern_id_map assertion, which is why this
                # was isolated to performance-mode assets. An empty pattern map
                # is the correct, intentional state post-PAT-1..7, not an error.
                self.ai_validator = ai_validator
                self.ai_enabled = True

                logger.info(f"[AGGREGATOR] AI validation: ✓ ENABLED")
                logger.info(
                    f"[AGGREGATOR] Patterns loaded: {len(ai_validator.pattern_id_map)}"
                )

            except (AssertionError, AttributeError) as e:
                logger.error(f"[AGGREGATOR] AI validation setup failed: {e}")
                logger.warning("[AGGREGATOR] Continuing without AI validation")
                self.ai_validator = None
                self.ai_enabled = False

        # AI statistics tracking
        if self.ai_enabled:
            self.ai_stats = {
                "mr_signals_checked": 0,
                "mr_approved": 0,
                "mr_rejected": 0,
                "tf_signals_checked": 0,
                "tf_approved": 0,
                "tf_rejected": 0,
                "bypassed_strong_signal": 0,
            }

            # Circuit breaker configuration
            self.enable_circuit_breaker = enable_ai_circuit_breaker
            self.ai_rejection_window = deque(maxlen=50)
            self.ai_bypass_active = False
            self.ai_bypass_threshold = 0.85
            self.ai_bypass_cooldown = 0

            logger.info(
                f"[AGGREGATOR] AI circuit breaker: {'ENABLED' if enable_ai_circuit_breaker else 'DISABLED'}"
            )
            logger.info(
                f"[AGGREGATOR] Strong signal bypass: {self.strong_signal_bypass:.2%}"
            )
            logger.info(
                f"[AGGREGATOR] Detailed logging: {'ENABLED' if self.detailed_logging else 'DISABLED'}"
            )

        # ✨ NEW: Advanced Confluence Engines
        self.divergence_detector = RSIDivergenceDetector(pivot_window=5)
        self.break_retest_validator = BreakRetestValidator(lookback=50)

        # Strategy weights — read from config, used for priority when multiple strategies fire
        # NOT for consensus voting. Hardcoded 0.50/0.50 was ignoring mean_reversion_weight: 0.0
        # in all presets, causing MR opposition penalty to bleed into every BTC TF score.
        # Will be updated after config merge below so we use a temporary default here.
        self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}

        # ================================================================
        # CONFIGURATION MERGE (Safety Fix)
        # ================================================================
        # 1. Define Defaults first (guarantees all keys exist)
        _is_fx_asset = (
            any(
                fx in self.asset_type.upper()
                for fx in ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")
            )
            and "BTC" not in self.asset_type.upper()
        )

        if self.asset_type == "BTC":
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.26,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.25,
                "bull_sell_penalty": 0.20,
                "bear_sell_boost": 0.25,
                "bear_buy_penalty": 0.30,
                "min_confidence_to_use": 0.08,
                "min_signal_quality": 0.28,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }
        elif _is_fx_asset:
            # FX pairs (EURUSD, EURJPY, GBPUSD, etc.) move in smaller, more
            # gradual increments than BTC or GOLD. Lowering thresholds prevents
            # valid setups from being blocked by score calculations tuned for
            # higher-volatility assets.
            # single_override_threshold 0.60 vs 0.72: FX strategies are configured
            # with min_confidence=0.45 — a 0.72 bar for independent firing almost
            # never gets reached, silently killing solo TF/EMA signals.
            self.config = {
                "buy_threshold": 0.26,
                "sell_threshold": 0.22,
                "two_strategy_bonus": 0.22,
                "three_strategy_bonus": 0.28,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.20,
                "bull_sell_penalty": 0.12,
                "bear_sell_boost": 0.20,
                "bear_buy_penalty": 0.22,
                "min_confidence_to_use": 0.05,
                "min_signal_quality": 0.22,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.35,
                "single_override_threshold": 0.60,
                "allow_single_override": True,
            }
        else:  # GOLD, USTEC, indices (Default)
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.24,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.22,
                "bull_sell_penalty": 0.15,
                "bear_sell_boost": 0.22,
                "bear_buy_penalty": 0.28,
                "min_confidence_to_use": 0.06,
                "min_signal_quality": 0.25,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }

        # 2. Update with passed config (Merge instead of Overwrite)
        if config is not None:
            # This ensures keys missing from 'config' are filled by defaults above
            self.config.update(config)

        # 3. Wire strategy weights from merged config (T1.2 fix)
        # Previously hardcoded to 0.50/0.50, ignoring mean_reversion_weight: 0.0 in presets.
        # EMA weight now included so all three strategies contribute to consensus scoring.
        self.weights = {
            "mean_reversion": self.config.get("mean_reversion_weight", 0.50),
            "trend_following": self.config.get("trend_following_weight", 0.50),
            "ema": self.config.get("ema_weight", 0.40),
        }
        logger.info(
            f"[AGGREGATOR] Strategy weights: MR={self.weights['mean_reversion']:.2f}, "
            f"TF={self.weights['trend_following']:.2f}, "
            f"EMA={self.weights['ema']:.2f}"
        )

        # 4. Independent strategy thresholds (T1.1 fix)
        # allow_single_override and single_override_threshold exist in presets but were
        # never read by this class — orphaned config keys. Now wired.
        #
        # MR was given its own fallback default here (0.75 vs TF/EMA's 0.72)
        # but all three read the SAME config key ("single_override_threshold"),
        # so MR's distinct default never actually applied whenever a preset
        # was in play — it just silently inherited TF/EMA's number. That
        # matters because MR's own confidence formulas (Mode1/2/3 combined)
        # empirically top out at 0.67-0.70 across every asset tested — below
        # every preset's threshold except "scalper" (0.65). MR was therefore
        # structurally incapable of ever winning the independent-fire contest
        # under "balanced" or "aggressive", regardless of signal quality.
        # Give it its own config key, calibrated to its real achievable range,
        # so a genuinely fired MR signal gets a chance to compete.
        self.independent_thresholds = {
            "trend_following": self.config.get("single_override_threshold", 0.72),
            "mean_reversion": self.config.get("mr_independent_threshold", 0.60),
            "ema": self.config.get("single_override_threshold", 0.72),
        }
        self.allow_independent = self.config.get("allow_single_override", True)
        logger.info(
            f"[AGGREGATOR] Independent firing: {'ENABLED' if self.allow_independent else 'DISABLED'} "
            f"(TF≥{self.independent_thresholds['trend_following']:.2f}, "
            f"MR≥{self.independent_thresholds['mean_reversion']:.2f})"
        )

        self.stats = {
            "total_evaluations": 0,
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "bull_regime_count": 0,
            "bear_regime_count": 0,
            "regime_changes": 0,
            "consensus_signals": 0,
            "single_strategy_signals": 0,
            "regime_detection_failures": 0,
        }

        # T1.5: Stale price detection state
        # Tracks (last_price, last_change_time) per asset to catch frozen data feeds.
        # MT5 assets trade market hours only — 90 min avoids false stale alerts
        # across the overnight close gap. Crypto is 24/7 so 30 min is tight enough.
        self._last_prices = {}
        self._stale_threshold_minutes = 65  # default — exceeds 1H candle duration
        self._stale_thresholds = (
            {  # per-asset overrides (MT5 = 90 min, crypto = default 65)
                "GOLD": 90,
                "USTEC": 90,
                "EURUSD": 90,
                "EURJPY": 90,
                "USOIL": 90,
                "GBPAUD": 90,
                "GBPUSD": 90,
                "USDJPY": 90,
            }
        )

        # T3.4: Economic calendar — loaded at startup, hot-reloaded by CalendarUpdater
        self._econ_cal_path = "config/economic_calendar.json"
        self._econ_events = []
        self._load_calendar_file()

        # ── CONTEXT ENGINE: new infrastructure ──────────────────────────────
        # B.3: Dynamic thresholds
        from src.utils.dynamic_thresholds import DynamicThresholds

        self.dynamic_thresholds = DynamicThresholds(lookback=100, min_samples=5)

        # Composite state builder — owns Livermore + builds CompositeState.
        # Shares dynamic_thresholds so persisted threshold history stays a
        # single source (see _persist_state / _load_persisted_state).
        from src.execution.composite_state_builder import CompositeStateBuilder
        self._cs_builder = CompositeStateBuilder(
            asset_type=self.asset_type,
            mean_reversion_strategy=mean_reversion_strategy,
            config=config,
        )
        self._cs_builder.dynamic_thresholds = self.dynamic_thresholds

        # D.1: Trend Lifecycle tracking
        self._previous_regime = {}  # {asset: regime_name}
        self._regime_start_time = {}  # {asset: datetime}
        self._regime_durations = {}  # {asset: [list of durations in hours]}
        self._transition_counts = {}  # {asset: {(from, to): count}}

        # E.2: MTF Structure Memory
        self._structure_levels = {}  # {asset: [{price, tf, type, age_hours, tests}]}

        # G.1: Liquidity sweep tracking
        self._pdh = {}  # {asset: price}
        self._pdl = {}  # {asset: price}
        self._asian_high = {}  # {asset: price}
        self._asian_low = {}  # {asset: price}
        self._pdh_date = None

        # G.5: Last loss tracking (populated externally by trade result callback)
        self._last_loss_time = {}  # {asset: datetime}

        # E.5: Squeeze state tracking
        self._squeeze_was_active = {}  # {asset: bool}

        # B.2: State cache slots (populated in get_aggregated_signal)
        self._cached_composite = None
        self._last_state_candle_time = None

        # F.7: Spread history for MT5 assets (per asset, last 20 values)
        self._spread_history = {}

        # B.4: State persistence — survive restarts.
        # Backtests must never share this path with live trading: backtest.py
        # passes an isolated per-asset/per-run path (or None to disable
        # persistence outright) so replaying a year of history can't read
        # stale/foreign state from — or overwrite — the live bot's real
        # calibration file.
        self._state_persistence_path = state_persistence_path
        if self._state_persistence_path:
            self._load_persisted_state()
        # ────────────────────────────────────────────────────────────────────

        self._log_initialization()

    # ── B.4: State Persistence ───────────────────────────────────────────────

    def _load_persisted_state(self):
        """Load cached state from disk to survive restarts."""
        try:
            import json, os

            if not os.path.exists(self._state_persistence_path):
                logger.info("[STATE] No persisted state file found — starting fresh.")
                return

            with open(self._state_persistence_path) as f:
                saved = json.load(f)

            # Restore dynamic threshold distributions
            if hasattr(self, "dynamic_thresholds"):
                for key_str, values in saved.get("threshold_cache", {}).items():
                    parts = key_str.split("|")
                    if len(parts) == 2:
                        self.dynamic_thresholds._cache[tuple(parts)] = values

            # Restore structure memory
            # Relocated: _structure_levels etc. now live on the builder
            # (see Step 3/composite_state_builder.py) — read/write there so
            # persisted state actually reaches the code that reads it.
            self._cs_builder._structure_levels = saved.get("structure_levels", {})
            self._cs_builder._zone_levels = saved.get("zone_levels", {})

            # Restore regime tracking
            self._cs_builder._previous_regime = saved.get("previous_regime", {})
            self._cs_builder._regime_start_time = {}
            for k, v in saved.get("regime_start_times", {}).items():
                try:
                    from datetime import datetime as _dtp

                    self._cs_builder._regime_start_time[k] = _dtp.fromisoformat(v)
                except Exception:
                    pass
            self._cs_builder._regime_durations = saved.get("regime_durations", {})

            # Restore transition counts (convert string keys back to tuples)
            saved_tc = saved.get("transition_counts", {})
            self._cs_builder._transition_counts = {}
            for asset, counts in saved_tc.items():
                self._cs_builder._transition_counts[asset] = {}
                if isinstance(counts, dict):
                    for k_str, v in counts.items():
                        if "|" in k_str:
                            parts = k_str.split("|")
                            self._cs_builder._transition_counts[asset][tuple(parts)] = v
                        else:
                            self._cs_builder._transition_counts[asset][k_str] = v

            # Restore sweep levels
            self._cs_builder._pdh = saved.get("pdh", {})
            self._cs_builder._pdl = saved.get("pdl", {})
            self._cs_builder._asian_high = saved.get("asian_high", {})
            self._cs_builder._asian_low = saved.get("asian_low", {})

            # Restore squeeze tracking
            self._cs_builder._squeeze_was_active = saved.get("squeeze_was_active", {})

            # Restore spread history (F.7)
            self._cs_builder._spread_history = saved.get("spread_history", {})

            _n_levels = sum(
                len(v)
                for v in self._cs_builder._structure_levels.values()
                if isinstance(v, (list, dict))
            )
            _n_thresh = len(saved.get("threshold_cache", {}))
            logger.info(
                f"[STATE] ✅ Loaded persisted state: "
                f"{_n_levels} structure levels, "
                f"{_n_thresh} threshold distributions, "
                f"{len(self._cs_builder._previous_regime)} regime histories"
            )
        except Exception as e:
            logger.warning(
                f"[STATE] Could not load persisted state: {e}. Starting fresh."
            )

    def _persist_state(self):
        """Save critical state to disk. Called once per candle close."""
        if not self._state_persistence_path:
            return
        try:
            import json, os

            # Convert tuple keys to pipe-separated strings for JSON
            _tc = {}
            if hasattr(self, "dynamic_thresholds"):
                for key_tuple, values in self.dynamic_thresholds._cache.items():
                    if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                        _tc[f"{key_tuple[0]}|{key_tuple[1]}"] = list(values)[-100:]

            from datetime import datetime as _dtj

            # Relocated: these now live on the builder (see _load_persisted_state).
            _csb = self._cs_builder
            state_data = {
                "threshold_cache": _tc,
                "structure_levels": getattr(_csb, "_structure_levels", {}),
                "zone_levels": getattr(_csb, "_zone_levels", {}),
                "previous_regime": getattr(_csb, "_previous_regime", {}),
                "regime_start_times": {
                    k: v.isoformat()
                    for k, v in getattr(_csb, "_regime_start_time", {}).items()
                },
                "regime_durations": getattr(_csb, "_regime_durations", {}),
                "transition_counts": {
                    asset: {f"{k[0]}|{k[1]}": v for k, v in counts.items()}
                    for asset, counts in getattr(_csb, "_transition_counts", {}).items()
                },
                "pdh": getattr(_csb, "_pdh", {}),
                "pdl": getattr(_csb, "_pdl", {}),
                "asian_high": getattr(_csb, "_asian_high", {}),
                "asian_low": getattr(_csb, "_asian_low", {}),
                "squeeze_was_active": getattr(_csb, "_squeeze_was_active", {}),
                "spread_history": getattr(_csb, "_spread_history", {}),
                "saved_at": _dtj.now().isoformat(),
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(self._state_persistence_path)),
                exist_ok=True,
            )
            tmp = self._state_persistence_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state_data, f, default=str)
            os.replace(tmp, self._state_persistence_path)
            logger.debug("[STATE] Persisted state to disk.")
        except Exception as e:
            logger.warning(f"[STATE] Persist failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯   PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("   ✓ STRICT MODE: Counter-trend trades blocked > 50% conf")
        logger.info("   ✓ RANGING SAFEGUARD: Max 1 trade when trend is weak")
        logger.info("   ✓ DYNAMIC threshold adjustment")
        if self.ai_enabled:
            logger.info("   ✓ AI VALIDATION: Active with circuit breaker")
        else:
            logger.info("   ⚠ AI VALIDATION: Disabled")
        logger.info("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # CALENDAR HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _load_calendar_file(self):
        """Load economic events from the JSON file on disk."""
        try:
            import json as _json

            with open(self._econ_cal_path, encoding="utf-8") as _f:
                self._econ_events = _json.load(_f).get("events", [])
            logger.info(
                f"[CALENDAR] Loaded {len(self._econ_events)} events "
                f"from {self._econ_cal_path}"
            )
        except Exception as _e:
            logger.warning(f"[CALENDAR] Could not load {self._econ_cal_path}: {_e}")
            self._econ_events = []

    def reload_calendar(self):
        """
        Hot-reload the economic calendar from disk.
        Called by CalendarUpdater after each successful write so the
        aggregator picks up fresh data without a bot restart.
        """
        self._load_calendar_file()
        logger.info(
            f"[CALENDAR] 🔄 Hot-reloaded — "
            f"{len(self._econ_events)} active events in memory"
        )

    # ══════════════════════════════════════════════════════════════════════
    # CONTEXT ENGINE — Build composite state (called once per candle close)
    # ══════════════════════════════════════════════════════════════════════

    def _build_composite_state(self, df, df_4h, governor_data: dict):
        """Relocated to CompositeStateBuilder. Forwarder kept so existing
        callers work unchanged: main.py (2027, 4855, 6534), backtest.py (887),
        and STEP 6B below (net_conviction consolidation)."""
        return self._cs_builder._build_composite_state(df, df_4h, governor_data)

    def warm_start_livermore(self, df_4h, df_1h) -> None:
        """Relocated to CompositeStateBuilder. Forwarder kept for
        main.py:7281's _warm_start_livermore_all_assets()."""
        return self._cs_builder.warm_start_livermore(df_4h, df_1h)

    @property
    def _livermore_warmed(self):
        """main.py reads/writes this at 7 sites (2034, 3059, 3128, 3193,
        3236, 4862, 6541). Forwards to the builder so those work untouched."""
        return self._cs_builder._livermore_warmed

    @_livermore_warmed.setter
    def _livermore_warmed(self, value):
        self._cs_builder._livermore_warmed = value

    @property
    def _livermore_4h(self):
        return self._cs_builder._livermore_4h

    @_livermore_4h.setter
    def _livermore_4h(self, value):
        """SETTER REQUIRED — main.py:3125/3190/3233 assign warmed Livermore
        state across aggregator swaps on preset change. A getter-only
        property raises AttributeError there."""
        self._cs_builder._livermore_4h = value

    @property
    def _livermore_1h(self):
        return self._cs_builder._livermore_1h

    @_livermore_1h.setter
    def _livermore_1h(self, value):
        self._cs_builder._livermore_1h = value

    @property
    def _livermore_last_4h_ts(self):
        return self._cs_builder._livermore_last_4h_ts

    @_livermore_last_4h_ts.setter
    def _livermore_last_4h_ts(self, value):
        self._cs_builder._livermore_last_4h_ts = value

    @property
    def phase_config(self):
        """Discovered during relocation (not in the original plan):
        _build_composite_state reads getattr(self, "phase_config", {}) as its
        very first real line. main.py/backtest.py set .phase_config directly
        onto this shell instance at 8+ call sites after construction (e.g.
        main.py:1396,1464-1465,1537-1538,3136-3137,3205-3206,3244;
        backtest.py:673,711). Without this forwarder, _build_composite_state
        (now on the builder) would never see those externally-set values —
        state.phase_config would silently stay {} forever, disabling every
        phase_config-gated behavior in the system with no error raised."""
        return getattr(self._cs_builder, "phase_config", {})

    @phase_config.setter
    def phase_config(self, value):
        self._cs_builder.phase_config = value

    def _score_confluence(self, state, tf_conf: float, mr_conf: float, signal: int = 0):
        """
        The Brain. Reads the complete state and applies adjustments
        based on PATTERNS first, individual evidence second.
        signal: the current trade direction (+1 long, -1 short, 0 unknown)
        """

        # ─── STEP 1: INSTITUTIONAL PATTERN RECOGNITION ───────────────────

        # ─── PATTERN DIAGNOSTICS (Issue 1, Step 1) ──────────────────────
        # Shows exactly which conditions pass/fail for each pattern.
        # MISSING fields = upstream module not writing to CompositeState (bug).
        # ❌ fields = market conditions don't match (working as intended).
        _diag_fields = {
            "lifecycle_phase": state.lifecycle_phase,
            "regime_age_ratio": f"{state.regime_age_ratio:.2f}",
            "choch_detected": state.choch_detected,
            "structural_decay": state.structural_decay,
            "absorption_detected": state.absorption_detected,
            "conviction_dying": state.conviction_dying,
            "distance_zscore": f"{state.distance_zscore:.2f}",
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            "effort_result_zscore": f"{state.effort_result_zscore:.2f}",
            "outside_bar": state.outside_bar,
            "failed_breakout": state.failed_breakout,
            "coiled_spring": state.coiled_spring,
            "ema_50_status": state.ema_50_status,
            "ema_50_reclassified": state.ema_50_reclassified,
        }
        logger.info(
            f"[PATTERN DIAG] {self.asset_type} state: "
            + " | ".join(f"{k}={v}" for k, v in _diag_fields.items())
        )

        # Evaluate each pattern and log which condition blocks it
        _dist_checks = {
            "phase∈ESTABLISHED/FADING": state.lifecycle_phase
            in ("ESTABLISHED", "FADING"),
            f"age_ratio>{1.3}": state.regime_age_ratio > 1.3,
            "choch_or_decay": state.choch_detected or state.structural_decay,
            "absorption_or_dying": state.absorption_detected or state.conviction_dying,
            f"dist_z>{1.5}": state.distance_zscore > 1.5,
        }
        _accum_checks = {
            "phase∈PICKUP/CONFIRM": state.lifecycle_phase in ("PICKUP", "CONFIRMATION"),
            f"age_ratio<{0.8}": state.regime_age_ratio < 0.8,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "no_absorption": not state.absorption_detected,
        }
        _liq_checks = {
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            f"effort_z>{2.0}": state.effort_result_zscore > 2.0,
            "outside_or_failed": state.outside_bar or state.failed_breakout,
        }
        _spring_checks = {
            "coiled_spring": state.coiled_spring,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
        }
        _ma_checks = {
            "ema50=DEFENDED/ABOVE": state.ema_50_status in ("DEFENDED", "EMA_ABOVE"),
            "ema50=SUPPORT": state.ema_50_reclassified == "SUPPORT",
            "phase∈CONFIRM/ESTAB": state.lifecycle_phase
            in ("CONFIRMATION", "ESTABLISHED"),
            f"age_ratio<{1.5}": state.regime_age_ratio < 1.5,
        }

        for _pname, _pchecks in [
            ("DISTRIBUTION", _dist_checks),
            ("ACCUMULATION", _accum_checks),
            ("LIQUIDITY_HUNT", _liq_checks),
            ("SPRING_BREAKOUT", _spring_checks),
            ("MA_DEFENSE", _ma_checks),
        ]:
            _passed = sum(_pchecks.values())
            _total = len(_pchecks)
            _blocker = next((k for k, v in _pchecks.items() if not v), None)
            _status = (
                "✅ MATCHED" if all(_pchecks.values()) else f"❌ {_passed}/{_total}"
            )
            logger.info(
                f"[PATTERN CHECK] {self.asset_type} {_pname}: {_status}"
                + (f" — first blocker: {_blocker}" if _blocker else "")
            )

        # PATTERN A: Institutional Distribution
        if all(_dist_checks.values()):
            tf_conf *= 0.45
            mr_conf *= 1.25
            state.institutional_pattern = "DISTRIBUTION"

        # PATTERN B: Institutional Accumulation
        elif all(_accum_checks.values()):
            tf_conf *= 1.30
            mr_conf *= 0.65
            state.institutional_pattern = "ACCUMULATION"

        # PATTERN C: Liquidity Hunt → Reversal
        elif all(_liq_checks.values()):
            mr_conf *= 1.35
            tf_conf *= 0.60
            state.institutional_pattern = "LIQUIDITY_HUNT"

        # PATTERN D: Coiled Spring Breakout
        elif state.coiled_spring and state.bos_detected and state.slopes_aligned:
            tf_conf *= 1.25
            mr_conf *= 0.70
            state.institutional_pattern = "SPRING_BREAKOUT"

        # PATTERN E: MA Defense → Continuation (or EMA_ABOVE trend ride)
        elif (
            state.ema_50_status in ("DEFENDED", "EMA_ABOVE")
            and state.ema_50_reclassified == "SUPPORT"
            and state.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED")
            and state.regime_age_ratio < 1.5
        ):
            tf_conf *= 1.20
            state.institutional_pattern = "MA_DEFENSE"

        # ─── STEP 2: ADDITIVE CONFLUENCE (fallback if no pattern matched) ─
        else:
            # Item 2.7: don't blank institutional_pattern here. None of these
            # five stricter patterns matched, but _compute_institutional_pattern
            # already set a live ACCUMULATION/DISTRIBUTION/COMPRESSION/None
            # classification earlier in _build_composite_state — resetting to
            # None here would silently erase it on every cycle this stricter
            # classifier doesn't fire (the common case), sending the Pattern
            # judge back to its legacy candlestick fallback for no reason.
            pass

            # A12: net_conviction computation moved to _compute_net_conviction,
            # relocated onto CompositeStateBuilder along with
            # _build_composite_state (both are composite-state logic). This
            # method (_score_confluence) stays on the shell, so it reaches
            # across to the builder instance rather than calling a sibling
            # method directly. Same call, same result — just relocated.
            _net = self._cs_builder._compute_net_conviction(state, tf_conf=tf_conf, signal=signal)

            if _net > 0:
                _boost = min(1.35, 1.0 + (_net * 0.05))
                tf_conf *= _boost
            elif _net < 0:
                _discount = max(0.40, 1.0 + (_net * 0.07))
                tf_conf *= _discount

        # Friday PM flag for VTM
        state.friday_tighten = state.is_friday_pm

        logger.info(
            f"[CONFLUENCE] {self.asset_type}: Phase={state.lifecycle_phase} "
            f"Pattern={state.institutional_pattern} "
            f"Net={state.net_conviction:.1f} "
            f"TF={tf_conf:.3f} MR={mr_conf:.3f}"
        )

        # ✅ M-1 FIX: Confluence multipliers (1.10–1.30×) can push confidence
        # above 1.0, making downstream percentage calculations nonsensical.
        tf_conf = max(0.0, min(1.0, tf_conf))
        mr_conf = max(0.0, min(1.0, mr_conf))

        return tf_conf, mr_conf, state

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics"""
        total = max(self.stats["total_evaluations"], 1)
        base_stats = {
            **self.stats,
            "signal_rate": (self.stats["signals_generated"] / total) * 100,
            "buy_rate": (self.stats["buy_signals"] / total) * 100,
            "sell_rate": (self.stats["sell_signals"] / total) * 100,
            "bull_regime_pct": (self.stats["bull_regime_count"] / total) * 100,
            "bear_regime_pct": (self.stats["bear_regime_count"] / total) * 100,
        }

        # Add AI statistics
        if self.ai_enabled and hasattr(self, "ai_stats"):
            mr_total = self.ai_stats["mr_signals_checked"]
            tf_total = self.ai_stats["tf_signals_checked"]

            base_stats["ai_validation"] = {
                "enabled": True,
                "circuit_breaker_active": self.ai_bypass_active,
                "mr_checked": mr_total,
                "mr_approved": self.ai_stats["mr_approved"],
                "mr_rejected": self.ai_stats["mr_rejected"],
                "mr_rejection_rate": (
                    (self.ai_stats["mr_rejected"] / mr_total * 100)
                    if mr_total > 0
                    else 0
                ),
                "tf_checked": tf_total,
                "tf_approved": self.ai_stats["tf_approved"],
                "tf_rejected": self.ai_stats["tf_rejected"],
                "tf_rejection_rate": (
                    (self.ai_stats["tf_rejected"] / tf_total * 100)
                    if tf_total > 0
                    else 0
                ),
            }

        return base_stats

    def _check_ai_circuit_breaker(self) -> bool:
        """
        Check if AI is rejecting too many signals
        Returns True if AI should be bypassed
        """
        if not self.enable_circuit_breaker or len(self.ai_rejection_window) < 20:
            return False

        # Calculate rejection rate (True = rejected, False = approved)
        rejection_rate = sum(self.ai_rejection_window) / len(self.ai_rejection_window)

        if rejection_rate > self.ai_bypass_threshold:
            if not self.ai_bypass_active:
                logger.warning("")
                logger.warning("=" * 70)
                logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
                logger.warning(
                    f"   Rejection rate: {rejection_rate:.0%} (threshold: {self.ai_bypass_threshold:.0%})"
                )
                logger.warning(
                    f"   AI validation temporarily DISABLED for next 10 signals"
                )
                logger.warning("=" * 70)
                logger.warning("")
                self.ai_bypass_active = True
                self.ai_bypass_cooldown = 10  # Bypass next 10 signals

            return True

        # Check if cooldown expired
        if self.ai_bypass_active and self.ai_bypass_cooldown <= 0:
            logger.info("🔄 AI circuit breaker reset - validation RE-ENABLED")
            self.ai_bypass_active = False
            self.ai_rejection_window.clear()  # Reset tracking

        return self.ai_bypass_active

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Multi-factor regime detection with cold-start handling
        Returns: (is_bull, confidence)
        """
        try:
            MIN_DATA_POINTS = 50

            # ===============================
            # 1️⃣ Cold-start & data sufficiency
            # ===============================
            if len(df) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for regime detection: {len(df)} rows"
                )
                self.stats["regime_detection_failures"] += 1

                if self.previous_regime is not None:
                    return self.previous_regime, 0.3

                if len(df) >= 20:
                    recent_momentum = (
                        df["close"].iloc[-1] - df["close"].iloc[-20]
                    ) / df["close"].iloc[-20]
                    emergency_regime = recent_momentum > 0
                    logger.info(
                        f"[REGIME] Emergency mode: {'BULL' if emergency_regime else 'BEAR'} "
                        f"(20-day momentum: {recent_momentum:.2%})"
                    )
                    return emergency_regime, 0.3

                logger.warning(
                    "[REGIME] Insufficient data - defaulting to BEAR (conservative)"
                )
                return False, 0.3

            # ===============================
            # 2️⃣ Feature generation
            # ===============================
            features_df = self.s_ema.generate_features(df.tail(250))
            if features_df.empty or len(features_df) < MIN_DATA_POINTS:
                logger.warning(f"EMA features insufficient: {len(features_df)} rows")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            latest = features_df.iloc[-1]

            ema_fast = latest.get("ema_fast", np.nan)
            ema_slow = latest.get("ema_slow", np.nan)
            ema_diff_pct = latest.get("ema_diff_pct", 0.0)

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                logger.warning("Invalid EMA values")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # ====================================================================
            # 3️⃣ Thresholds (Rolling Quantile) - ✅ TASK 21 (Phase 3)
            # ====================================================================
            # Reason: Fixed thresholds fail in different volatility regimes.
            # We use the last 100 bars to find the 65th/35th percentiles.
            ema_diff_series = features_df["ema_diff_pct"].tail(100).dropna()

            if len(ema_diff_series) >= 50:  # Minimum bars for meaningful quantile
                # Calculate percentiles
                BULLISH_THRESHOLD = ema_diff_series.quantile(0.65)
                BEARISH_THRESHOLD = ema_diff_series.quantile(0.35)

                # Clamp to institutional bounds [0.05, 0.40]
                BULLISH_THRESHOLD = max(0.05, min(0.40, BULLISH_THRESHOLD))
                BEARISH_THRESHOLD = min(-0.05, max(-0.40, BEARISH_THRESHOLD))
            else:
                # Fallback to defaults
                BULLISH_THRESHOLD = 0.15 if self.asset_type == "BTC" else 0.10
                BEARISH_THRESHOLD = -0.15 if self.asset_type == "BTC" else -0.10

            close_prices = features_df["close"].values

            ret_20 = (
                (close_prices[-1] - close_prices[-20]) / close_prices[-20]
                if len(close_prices) >= 20
                else 0.0
            )
            ret_50 = (
                (close_prices[-1] - close_prices[-50]) / close_prices[-50]
                if len(close_prices) >= 50
                else 0.0
            )

            if len(close_prices) >= 21:
                returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                vol_20 = np.std(returns) * np.sqrt(252)
            else:
                vol_20 = 0.2

            adx = latest.get("adx", 20)
            macd_hist = latest.get("macd_hist", 0)
            rsi = latest.get("rsi", 50)

            # Asset-specific ADX threshold
            adx_threshold = getattr(self.s_trend_following, "adx_threshold", 25)

            # ===============================
            # 4️⃣ Multi-factor scoring
            # ===============================
            bullish_score = 0
            bearish_score = 0

            # EMA positioning (dominant factor)
            if ema_diff_pct > BULLISH_THRESHOLD:
                bullish_score += 3
            elif ema_diff_pct < BEARISH_THRESHOLD:
                bearish_score += 3

            # Short-term momentum
            if ret_20 > 0.02:
                bullish_score += 2
            elif ret_20 < -0.02:
                bearish_score += 2

            # Medium-term momentum
            if ret_50 > 0.05:
                bullish_score += 2
            elif ret_50 < -0.05:
                bearish_score += 2

            # MACD
            if macd_hist > 0:
                bullish_score += 1
            elif macd_hist < 0:
                bearish_score += 1

            # ADX trend strength
            if adx > adx_threshold:
                if ema_diff_pct > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            # RSI
            if rsi > 60:
                bullish_score += 1
            elif rsi < 40:
                bearish_score += 1

            # ===============================
            # 5️⃣ Hysteresis-based decision
            # ===============================
            if self.previous_regime is None:
                is_bull = bullish_score > bearish_score
            else:
                if self.previous_regime:
                    is_bull = not (bearish_score > bullish_score + 2)
                else:
                    is_bull = bullish_score > bearish_score + 2

            # ===============================
            # 6️⃣ Confidence scoring
            # ===============================
            confidence = 0.5

            if abs(ema_diff_pct) > 0.5:
                confidence += 0.15

            if (is_bull and ret_20 > 0.03) or (not is_bull and ret_20 < -0.03):
                confidence += 0.15

            if adx > 25:
                confidence += 0.1

            if abs(bullish_score - bearish_score) >= 4:
                confidence += 0.1

            confidence = min(1.0, max(0.3, confidence))

            # ===============================
            # 7️⃣ Logging & stats
            # ===============================
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
                logger.info(
                    f"⚡ REGIME FLIP → {'BULL' if is_bull else 'BEAR'} | "
                    f"Scores B:{bullish_score} / R:{bearish_score} | "
                    f"Confidence: {confidence:.2f}"
                )

            elif not self.regime_initialized:
                logger.info(
                    f"🎬 INITIAL REGIME → {'BULL' if is_bull else 'BEAR'} | "
                    f"Confidence: {confidence:.2f}"
                )
                self.regime_initialized = True

            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            return is_bull, confidence

        # ======================================================
        # 8️⃣ HARD FALLBACK: EMA-only regime detection
        # ======================================================
        except Exception as e:
            logger.error(f"Primary regime detection failed: {e}", exc_info=True)
            self.stats["regime_detection_failures"] += 1

            try:
                ema_signal, ema_conf = self.s_ema.generate_signal(df)
                is_bull = ema_signal >= 0

                self.previous_regime = is_bull
                if is_bull:
                    self.stats["bull_regime_count"] += 1
                else:
                    self.stats["bear_regime_count"] += 1

                return is_bull, ema_conf

            except Exception as e:
                logger.error(f"EMA fallback failed: {e}", exc_info=True)
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

    def calculate_regime_adjusted_thresholds(
        self, is_bull: bool, regime_confidence: float
    ) -> Tuple[float, float]:
        """
        Dynamically adjust thresholds based on regime strength
        """
        base_buy = self.config["buy_threshold"]
        base_sell = self.config["sell_threshold"]

        # Fix E: proportional adjustments (percentage of base) instead of fixed offsets.
        # Fixed offsets were regime-blind: a 0.10 offset on a 0.23 scalper threshold
        # is a 43% swing, while the same offset on a 0.33 conservative is only 30%.
        strength = (regime_confidence - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
        strength = max(0.0, min(1.0, strength))

        if is_bull:
            # Bull: ease buy gate by up to 18%, tighten sell gate by up to 15%
            adjusted_buy = base_buy * (1.0 - 0.18 * strength)
            adjusted_sell = base_sell * (1.0 + 0.15 * strength)
        else:
            # Bear: tighten buy gate by up to 20%, ease sell gate by up to 18%
            adjusted_buy = base_buy * (1.0 + 0.20 * strength)
            adjusted_sell = base_sell * (1.0 - 0.18 * strength)

        # Safety bounds
        adjusted_buy = max(0.15, min(0.60, adjusted_buy))
        adjusted_sell = max(0.15, min(0.60, adjusted_sell))

        # Log significant changes
        if abs(adjusted_buy - base_buy) > 0.05:
            logger.debug(
                f"[THRESHOLD] Buy: {base_buy:.2f}→{adjusted_buy:.2f} ({'BULL' if is_bull else 'BEAR'}, conf:{regime_confidence:.2f})"
            )

        return adjusted_buy, adjusted_sell

    def _format_ai_validation_for_viz(
        self, final_signal: int, details: dict, df: pd.DataFrame
    ) -> dict:
        """
        CRITICAL FIX: Format AI validation results for visualization
        ✅ FIXED: Proper type conversions for pattern_detected and near_sr_level
        """
        try:
            # Initialize with safe defaults
            viz_data = {
                "pattern_detected": False,  # ← Must be bool
                "validation_passed": False,
                "pattern_name": "None",
                "pattern_id": None,
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,  # ← Must be bool
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

            # Check if AI validator exists
            if not self.ai_validator or not self.ai_enabled:
                viz_data["action"] = "ai_disabled"
                return viz_data

            current_price = float(df["close"].iloc[-1])

            # ================================================================
            # STEP 1: Get S/R Analysis
            # ================================================================
            try:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type,
                    df=df,
                    current_price=current_price,
                    signal=final_signal,
                    threshold=self.ai_validator.current_sr_threshold,
                )

                # ✅ FIX: Convert numpy.bool to Python bool
                near_level = sr_result.get("near_level", False)
                if isinstance(near_level, np.bool_):
                    near_level = bool(near_level)

                viz_data["sr_analysis"] = {
                    "near_sr_level": near_level,  # ← Now guaranteed Python bool
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }

            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")
                viz_data["error"] = f"S/R error: {str(e)}"

            # ================================================================
            # STEP 2: Pattern Detection — PAT-2: module removed, static stubs
            # ================================================================
            viz_data["pattern_detected"] = False
            viz_data["pattern_name"] = None
            viz_data["pattern_id"] = None
            viz_data["pattern_confidence"] = 0.0

            # ================================================================
            # STEP 3: Determine Validation Status
            # ================================================================
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
                # Item 18c: removed dead "ai_bypassed"/quality-guess branch — that
                # flag was never set anywhere in the codebase, and the quality-vs-
                # strong_signal_bypass check below it was guessing a dashboard label
                # without reflecting whether the real validator (now wired in STEP 8,
                # flag-gated) actually ran. "approved" is the correct label whenever
                # this cosmetic formatter sees a non-zero signal.
                viz_data["validation_passed"] = True
                viz_data["action"] = "approved"
            else:
                viz_data["action"] = "hold"

            # ================================================================
            # ✅ FINAL TYPE VALIDATION
            # ================================================================
            # Ensure all bools are Python bool, not numpy.bool
            viz_data["pattern_detected"] = bool(viz_data["pattern_detected"])
            viz_data["validation_passed"] = bool(viz_data["validation_passed"])
            viz_data["sr_analysis"]["near_sr_level"] = bool(
                viz_data["sr_analysis"]["near_sr_level"]
            )

            return viz_data

        except Exception as e:
            logger.error(f"[VIZ] AI formatting failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "ERROR",
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
                "action": "error",
                "error": str(e),
            }

    def _calculate_score(
        self,
        df: pd.DataFrame,
        target_signal: int,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        ema_signal: int,
        ema_conf: float,
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """Calculate aggregated score for all three strategies (MR + TF + EMA)."""
        components = []
        total_score = 0.0
        agreement_count = 0
        min_conf = self.config["min_confidence_to_use"]
        hold_contrib = self.config["hold_contribution_pct"]
        opposition_penalty = self.config["opposition_penalty"]

        # Mean Reversion contribution
        if mr_signal == target_signal:
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_agree:{contribution:.3f}")
            agreement_count += 1
        elif mr_signal == 0:
            effective_conf = max(mr_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "mean_reversion"
            ]
            total_score += contribution
            components.append(f"MR_hold:{contribution:.3f}")
        else:
            effective_conf = max(mr_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "mean_reversion"
            ]
            total_score -= penalty
            components.append(f"MR_oppose:-{penalty:.3f}")

        # Trend Following contribution
        if tf_signal == target_signal:
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_agree:{contribution:.3f}")
            agreement_count += 1
        elif tf_signal == 0:
            effective_conf = max(tf_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "trend_following"
            ]
            total_score += contribution
            components.append(f"TF_hold:{contribution:.3f}")
        else:
            effective_conf = max(tf_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "trend_following"
            ]
            total_score -= penalty
            components.append(f"TF_oppose:-{penalty:.3f}")

        # EMA contribution (previously excluded — now a full voting member)
        if ema_signal == target_signal:
            effective_conf = max(ema_conf, min_conf)
            contribution = effective_conf * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_agree:{contribution:.3f}")
            agreement_count += 1
        elif ema_signal == 0:
            effective_conf = max(ema_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_hold:{contribution:.3f}")
        else:
            effective_conf = max(ema_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights["ema"]
            total_score -= penalty
            components.append(f"EMA_oppose:-{penalty:.3f}")

        # --- VolumeFlow vote ---
        if self.s_volume_flow is not None:
            try:
                vf_signal, vf_conf = self.s_volume_flow.generate_signal(df)
                if vf_signal == target_signal and vf_conf >= min_conf:
                    effective_conf = max(vf_conf, min_conf)
                    contribution = effective_conf * (
                        1 - self.config.get("opposition_penalty", 0.40)
                    )
                    total_score += contribution
                    components.append(f"VF_agree:{contribution:.3f}")
                    agreement_count += 1
                elif vf_signal != 0 and vf_signal != target_signal:
                    penalty = vf_conf * self.config.get("opposition_penalty", 0.40)
                    total_score -= penalty
                    components.append(f"VF_oppose:-{penalty:.3f}")
                elif vf_signal == 0:
                    effective_conf = max(vf_conf, min_conf)
                    hold_contribution = effective_conf * hold_contrib
                    total_score += hold_contribution
                    if hold_contribution > 0:
                        components.append(f"VF_hold:{hold_contribution:.3f}")
            except Exception as _vf_e:
                logger.debug(f"[AGG] VolumeFlow signal error: {_vf_e}")

        explanation = " + ".join(components) if components else "no_agreement"

        # Agreement bonus — tiered (two_strategy_bonus and three_strategy_bonus now both active)
        if agreement_count == 4:
            bonus = self.config.get(
                "four_strategy_bonus", self.config.get("three_strategy_bonus", 0.35)
            )
            total_score += bonus
            explanation += f" + bonus4({bonus:.2f})"
        elif agreement_count == 3:
            bonus = self.config.get(
                "three_strategy_bonus", self.config["two_strategy_bonus"]
            )
            total_score += bonus
            explanation += f" + bonus3({bonus:.2f})"
        elif agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus2({bonus:.2f})"

        # Regime context
        if target_signal == 1:  # BUY
            if is_bull:
                regime_adj = self.config["bull_buy_boost"]
                total_score += regime_adj
                explanation += f" + bull({regime_adj:.2f})"
            else:
                # ✨ NEW: Explosive Momentum Overrule
                if self._is_explosive_momentum(df, target_signal):
                    logger.info(
                        "[MOMENTUM] Skipping bear-regime penalty due to explosive BUY momentum"
                    )
                    regime_adj = 0
                    explanation += " + V-Shape Overrule"
                else:
                    regime_adj = -self.config["bear_buy_penalty"]
                    total_score = max(0.0, total_score + regime_adj)
                    explanation += f" - bear({abs(regime_adj):.2f})"
        else:  # SELL
            if is_bull:
                # ✨ NEW: Explosive Momentum Overrule
                if self._is_explosive_momentum(df, target_signal):
                    logger.info(
                        "[MOMENTUM] Skipping bull-regime penalty due to explosive SELL momentum"
                    )
                    regime_adj = 0
                    explanation += " + V-Shape Overrule"
                else:
                    regime_adj = -self.config["bull_sell_penalty"]
                    total_score = max(0.0, total_score + regime_adj)
                    explanation += f" - bull({abs(regime_adj):.2f})"
            else:
                regime_adj = self.config["bear_sell_boost"]
                total_score += regime_adj
                explanation += f" + bear({regime_adj:.2f})"

        total_score = max(0.0, total_score)
        return total_score, explanation, agreement_count

    def _check_governor_filter(
        self, df: pd.DataFrame, signal: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Filter 1: Governor (Daily 200 EMA) Check

        Returns:
            (passed, trade_type)
        """
        if not self.use_macro_governor:
            return True, "TREND"

        if not self.enable_filters or not self.mtf_integration:
            return True, "TREND"  # Skip if disabled

        try:
            # Get Governor analysis from MTF
            regime_data = self.mtf_integration._current_regime_data.get(self.asset_type)

            if not regime_data:
                logger.debug(f"[GOV] No data for {self.asset_type}, allowing trade")
                return True, "TREND"

            # ✨ IMPROVED: Robust key check
            governor = regime_data.get("governor") or regime_data.get(
                "full_regime_status"
            )

            if not governor:
                logger.debug(
                    f"[GOV] No governor object for {self.asset_type}, allowing trade"
                )
                return True, "TREND"

            # ✨ IMPROVED: Handle Enum vs String vs Attribute
            raw_trade_type = getattr(governor, "trade_type", None)
            if raw_trade_type is None:
                # Fallback to consensus_regime if trade_type is missing
                regime_name = getattr(governor, "consensus_regime", "NEUTRAL")
                trade_type = "NEUTRAL" if regime_name == "NEUTRAL" else "TREND"
            else:
                trade_type = getattr(raw_trade_type, "value", str(raw_trade_type))

            # TRANSITION path removed (Phase 0B — MRS §6).
            # NEUTRAL regime no longer maps to a half-size TRANSITION entry.
            # NEUTRAL trades pass through at full sizing — the gatekeeper
            # already handles NEUTRAL as "all strategies allowed."
            # Future: Phase 2 Hard Veto Layer will gate on Livermore state
            # (NATURAL/SECONDARY) rather than MTF regime, providing structural
            # context that MTF NEUTRAL cannot supply.
            if trade_type == "NEUTRAL":
                logger.debug("[GOV] NEUTRAL regime — passing at full size")
                return True, "NEUTRAL"

            return True, trade_type

        except Exception as e:
            logger.error(f"[GOV] Error: {e}")
            return True, "TREND"  # Fail-open

    def _check_volatility_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 2: Volatility Gate

        Returns:
            (passed, atr_pct)
        """
        if not self.enable_filters:
            return True, 0.005

        try:
            if len(df) < 20:
                return True, 0.005

            # Calculate ATR
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

            current_price = df["close"].iloc[-1]
            atr_pct = atr / current_price

            threshold = self.filter_thresholds["volatility_gate"]
            passed = atr_pct >= threshold

            if not passed:
                logger.info(f"[VOL] ❌ BLOCKED - ATR {atr_pct:.3%} < {threshold:.3%}")

            return passed, atr_pct

        except Exception as e:
            logger.error(f"[VOL] Error: {e}")
            return True, 0.005

    def _check_sniper_filter(
        self, df: pd.DataFrame, signal: int, governor_data: Dict = None
    ) -> Tuple[bool, Dict]:
        """
        DISCONNECTED — Phase 0B (MRS §6 Phase 0).
        CNN-LSTM sniper removed from scoring pipeline.
        This method is retained as dead code for reference only.
        It is no longer called from get_aggregated_signal().

        Returns (True, {}) unconditionally if somehow invoked.
        """
        return True, {"trigger_type": "SNIPER_DISCONNECTED"}

    def _check_sniper_filter_LEGACY(
        self, df: pd.DataFrame, signal: int, governor_data: Dict = None
    ) -> Tuple[bool, Dict]:
        """
        LEGACY — kept for reference only. Not called anywhere.
        Original Filter 3: Sniper Lock - Institutional Edge Confirmation.
        Removed in Phase 0B: CNN-LSTM trained on 15-min data, received 1H data.
        Peak accuracy ~0.70 adds no edge over structural rules.
        """
        if not self.enable_filters:
            return True, {"trigger_type": "DISABLED"}

        try:
            latest = df.iloc[-1]
            reasons = []

            # ================================================================
            # 1. AI Pattern Confidence — PAT-2: removed (pattern module disabled)
            # ================================================================

            # ================================================================
            # 2. Momentum Candle
            # ================================================================
            # Reason: Confirms strong conviction from buyers or sellers in the current period.
            body = abs(latest["close"] - latest["open"])
            total_range = latest["high"] - latest["low"]
            if total_range > 0:
                body_ratio = body / total_range
                if body_ratio >= 0.60:
                    is_bullish_candle = latest["close"] > latest["open"]
                    if (signal == 1 and is_bullish_candle) or (
                        signal == -1 and not is_bullish_candle
                    ):
                        reasons.append(
                            {
                                "passed": True,
                                "trigger_type": "MOMENTUM_CANDLE",
                                "body_ratio": body_ratio,
                            }
                        )

            # ================================================================
            # 3. Trend Momentum (Institutional Continuity)
            # ================================================================
            # Reason: If the macro regime and 1H momentum are both strong and
            # aligned, we allow entry even without a classic breakout or pattern.
            # Fix #18: Use "consensus_regime" key (backtest) falling back to "regime"
            # (live). Also derive h1_momentum_dir from df when not supplied by
            # governor (backtest governor doesn't compute it).
            if governor_data:
                _regime = governor_data.get(
                    "consensus_regime", governor_data.get("regime", "NEUTRAL")
                )
                _is_bull = "BULL" in _regime.upper()
                _is_bear = "BEAR" in _regime.upper()

                # Derive 1H momentum from df close slope when governor doesn't supply it
                _h1_dir = governor_data.get("h1_momentum_dir", "")
                if not _h1_dir and len(df) >= 5:
                    _slope = df["close"].iloc[-1] - df["close"].iloc[-5]
                    _atr_est = (
                        (df["high"].iloc[-14:] - df["low"].iloc[-14:]).mean()
                        if len(df) >= 14
                        else abs(_slope)
                    )
                    # Require slope to exceed 0.25 ATR to be directional (filters noise)
                    if _slope > _atr_est * 0.25:
                        _h1_dir = "UP"
                    elif _slope < -_atr_est * 0.25:
                        _h1_dir = "DOWN"
                    else:
                        _h1_dir = "FLAT"

                _regime_aligned = (signal == 1 and _is_bull) or (
                    signal == -1 and _is_bear
                )
                _h1_aligned = (signal == 1 and _h1_dir == "UP") or (
                    signal == -1 and _h1_dir == "DOWN"
                )

                if _regime_aligned and _h1_aligned:
                    reasons.append(
                        {
                            "passed": True,
                            "trigger_type": "TREND_MOMENTUM",
                            "regime": _regime,
                            "h1_dir": _h1_dir,
                        }
                    )

            # Check if we have enough data for rolling indicators
            if len(df) < 21:  # Need 20 periods + current
                if reasons:
                    logger.info(
                        f"[SNIPER] ✅ PASSED - Trigger(s): {[r['trigger_type'] for r in reasons]}"
                    )
                    return True, reasons[0]
                else:
                    logger.warning(
                        f"[SNIPER] ❌ BLOCKED - Insufficient data for full institutional checks (need 21 bars, have {len(df)})."
                    )
                    return False, {
                        "trigger_type": None,
                        "reason": f"Insufficient data for breakouts (have {len(df)})",
                    }

            # ================================================================
            # 4. Turtle Breakout (20-period Donchian Channel)
            # ================================================================
            # Reason: Captures classic institutional breakout entries.
            # We look at the previous 20 candles to define the channel *before* the current candle.
            high_20 = df["high"].iloc[-21:-1].max()
            low_20 = df["low"].iloc[-21:-1].min()

            if signal == 1 and latest["close"] > high_20:
                reasons.append(
                    {
                        "passed": True,
                        "trigger_type": "TURTLE_BREAKOUT",
                        "breakout_level": high_20,
                        "price": latest["close"],
                    }
                )
            elif signal == -1 and latest["close"] < low_20:
                reasons.append(
                    {
                        "passed": True,
                        "trigger_type": "TURTLE_BREAKOUT",
                        "breakout_level": low_20,
                        "price": latest["close"],
                    }
                )

            # ================================================================
            # 5. Volume Surge
            # ================================================================
            # Reason: Confirms institutional participation and conviction behind a move.
            volume_rolling_avg = df["volume"].iloc[-21:-1].mean()
            if volume_rolling_avg > 0 and latest["volume"] >= (
                volume_rolling_avg * 1.5
            ):
                reasons.append(
                    {
                        "passed": True,
                        "trigger_type": "VOLUME_SURGE",
                        "volume": latest["volume"],
                        "avg_volume": volume_rolling_avg,
                        "surge_factor": (
                            latest["volume"] / volume_rolling_avg
                            if volume_rolling_avg > 0
                            else 0
                        ),
                    }
                )

            # ================================================================
            # 6. Volatility Breach (Bollinger Bands)
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
            # 7. Established Trend + BOS / EMA Continuation (Institutional Continuation)
            # ================================================================
            # Reason: When the lifecycle is ESTABLISHED/CONFIRMATION with a fresh
            # Break-of-Structure and aligned slopes, this is a classic institutional
            # trend-continuation setup. Also fires when price is cleanly above EMA50
            # with aligned slopes (EMA_ABOVE) — this covers assets in a parabolic
            # rally where lower swing highs create CHoCH (not BOS) but the overall
            # trend and EMAs are clearly bullish. Without this path these setups are
            # silently blocked for assets like GOLD in a strong uptrend.
            _cs = getattr(self, "_cached_composite", None)
            if _cs is not None:
                _bos_continuation = (
                    _cs.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED")
                    and _cs.bos_detected
                    and _cs.slopes_aligned
                    and not _cs.structural_decay
                    and not _cs.absorption_detected
                    and _cs.regime_age_ratio < 2.0
                )
                _ema_above_continuation = (
                    _cs.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED")
                    and _cs.ema_50_status in ("EMA_ABOVE",)
                    and _cs.slopes_aligned
                    and not _cs.structural_decay
                    and _cs.regime_age_ratio < 1.8
                    and signal == 1  # only valid as a long trigger
                )
                _ema_below_continuation = (
                    _cs.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED")
                    and _cs.ema_50_status in ("EMA_BELOW",)
                    and _cs.slopes_aligned
                    and not _cs.structural_decay
                    and _cs.regime_age_ratio < 1.8
                    and signal == -1  # only valid as a short trigger
                )
                if _bos_continuation:
                    reasons.append(
                        {
                            "passed": True,
                            "trigger_type": "ESTABLISHED_BOS",
                            "phase": _cs.lifecycle_phase,
                            "age_ratio": round(_cs.regime_age_ratio, 2),
                        }
                    )
                elif _ema_above_continuation or _ema_below_continuation:
                    reasons.append(
                        {
                            "passed": True,
                            "trigger_type": "EMA_TREND_CONTINUATION",
                            "phase": _cs.lifecycle_phase,
                            "ema_status": _cs.ema_50_status,
                            "age_ratio": round(_cs.regime_age_ratio, 2),
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
            return True, {"trigger_type": "ERROR_FALLBACK"}

    def _check_profit_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 4: Minimum Profit Potential

        Returns:
            (passed, potential_pct)
        """
        if not self.enable_filters:
            return True, 0.01

        try:
            if len(df) < 20:
                return True, 0.01

            # Use ATR as proxy for potential move
            high_low = df["high"] - df["low"]
            atr = high_low.rolling(14).mean().iloc[-1]

            current_price = df["close"].iloc[-1]
            potential_pct = atr / current_price

            threshold = self.filter_thresholds["min_profit"]
            passed = potential_pct >= threshold

            if not passed:
                logger.info(
                    f"[PROFIT] ❌ BLOCKED - Potential {potential_pct:.2%} < {threshold:.2%}"
                )

            return passed, potential_pct

        except Exception as e:
            logger.error(f"[PROFIT] Error: {e}")
            return True, 0.01

    def _check_atr_expansion_filter(self, df: pd.DataFrame, trade_type: str) -> bool:
        """
        Fix C: Replaced ATR Expansion (candle_range >= 1.5*ATR) with ADX Trend Confirmation.

        Old logic required the latest candle's range to exceed 1.5× ATR. This blocked
        valid signals in slow-grinding trends (GOLD, EURUSD) where candles are small but
        direction is clear. The 1.5× bar was consistently failing even when ADX showed a
        strong trend (ADX > 25).

        New logic: confirm a trend is in force (ADX > 18). This threshold is intentionally
        low — 18 separates genuine trend from pure noise without demanding strong momentum.
        Counter-trend and REVERSION trades bypass the check (trade_type != "TREND").
        """
        if trade_type != "TREND":
            return True

        try:
            if len(df) < 20:
                return True

            # Calculate ADX (14)
            try:
                import talib

                adx_series = talib.ADX(
                    df["high"].values,
                    df["low"].values,
                    df["close"].values,
                    timeperiod=14,
                )
                adx = adx_series[-1]
            except Exception:
                # Manual ADX fallback: use DM-based approximation via TR rolling
                high_low = df["high"] - df["low"]
                high_close = np.abs(df["high"] - df["close"].shift())
                low_close = np.abs(df["low"] - df["close"].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean()
                dm_plus = (df["high"].diff()).clip(lower=0)
                dm_minus = (-df["low"].diff()).clip(lower=0)
                # Use only the dominant direction
                dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
                dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
                di_plus = 100 * dm_plus.rolling(14).mean() / atr14
                di_minus = 100 * dm_minus.rolling(14).mean() / atr14
                dx = (
                    100
                    * np.abs(di_plus - di_minus)
                    / (di_plus + di_minus).replace(0, np.nan)
                )
                adx = dx.rolling(14).mean().iloc[-1]

            if pd.isna(adx):
                return True

            ADX_MIN = 18
            passed = adx >= ADX_MIN

            if not passed:
                logger.info(
                    f"[ADX_TREND] ❌ BLOCKED - ADX {adx:.1f} < {ADX_MIN} (insufficient trend strength)"
                )
            else:
                logger.debug(f"[ADX_TREND] ✅ PASSED - ADX {adx:.1f}")

            return passed

        except Exception as e:
            logger.error(f"[ADX_TREND] Error: {e}")
            return True  # Fail-open

    def _is_explosive_momentum(self, df: pd.DataFrame, signal: int) -> bool:
        """
        Detects 'V-Shape' or 'Parabolic' price action that overrules macro bias.
        Criteria:
        1. ADX > 30 (Strong immediate trend)
        2. Velocity: Last 6 bars move > 2.0 * ATR14
        3. Alignment: Price > EMA20 > EMA50 (for Longs)
        """
        try:
            if len(df) < 50:
                return False

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # 1. Trend Strength
            adx = ta.ADX(high, low, close, timeperiod=14)[-1]
            if adx < 30:
                return False

            # 2. ATR-Scaled Velocity
            atr = ta.ATR(high, low, close, timeperiod=14)[-1]
            move = close[-1] - close[-6]
            velocity_ratio = abs(move) / (atr if atr > 0 else 1)

            if velocity_ratio < 2.0:
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

    def get_aggregated_signal(
        self,
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",
        is_bull_market: bool = True,
        governor_data: Dict = None,
        live_price: Optional[float] = None,  # ✨ NEW: For accurate staleness check
    ) -> Tuple[int, Dict]:
        """
        Main aggregation logic with AI validation and external regime context.
        """
        self.stats["total_evaluations"] += 1
        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

            # AI-5: Clear per-cycle pattern cache so sniper filter and format_viz share results.
            if self.ai_validator and hasattr(self.ai_validator, "clear_pattern_cache"):
                self.ai_validator.clear_pattern_cache()

            # ═══════════════════════════════════════════════════════════════
            # T1.5: STALE PRICE DETECTION
            # Gold was frozen at 5021.08 for 47.6 hours (March 13–15), firing
            # 12 SELL signals on dead data. Block evaluation if price has not
            # moved by even 1 pip in over 30 minutes.
            # ═══════════════════════════════════════════════════════════════
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
                _stale_limit = self._stale_thresholds.get(
                    self.asset_type, self._stale_threshold_minutes
                )
                if not _price_moved and _minutes_since_move > _stale_limit:
                    logger.warning(
                        f"[STALE] ❌ {self.asset_type} price frozen at {_current_price} "
                        f"for {_minutes_since_move:.0f}min — blocking signal evaluation"
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "regime": "UNKNOWN",
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
            # Update last-seen price only when it actually moves
            if not _last or abs(_current_price - _last[0]) / max(_last[0], 1) > 0.00001:
                self._last_prices[self.asset_type] = (_current_price, _now)

            # ═══════════════════════════════════════════════════════════════
            # B.2: STATE CACHE — Heavy calculations run ONCE per candle close.
            # The 5-second loop reads the cached state for micro-execution checks only.
            # ═══════════════════════════════════════════════════════════════
            _candle_time = df.index[-1] if not df.empty else None
            _state_is_fresh = (
                _candle_time is not None
                and getattr(self, "_last_state_candle_time", None) == _candle_time
            )

            if not _state_is_fresh and _candle_time is not None:
                # New candle closed — rebuild the full composite state.
                # Item 1: previously, any exception here left the rebuild
                # half-done with no warning — the cycle would silently keep
                # trading on yesterday's cached state forever. Now it logs
                # loudly and explicitly keeps the last known good state.
                try:
                    _new_state = self._build_composite_state(
                        df,
                        governor_data.get("df_4h") if governor_data else None,
                        governor_data or {},
                    )
                    self._cached_composite = _new_state
                    self._last_state_candle_time = _candle_time
                    self._persist_state()
                    logger.debug(
                        f"[STATE] Rebuilt composite state for {self.asset_type} at {_candle_time}"
                    )
                except Exception as e:
                    logger.error(
                        f"[STATE] Composite rebuild failed for {self.asset_type}, "
                        f"keeping last known state: {e}"
                    )

            # Use cached state for all downstream logic
            state = getattr(self, "_cached_composite", None)

            # ═══════════════════════════════════════════════════════════════
            # FLASH VETO — abnormal candle body detection
            # Hard-block above 5× ATR14; soft-discount (−40% quality) at 3–5×.
            # ═══════════════════════════════════════════════════════════════
            _flash_discount = 1.0
            try:
                if len(df) >= 15:
                    import numpy as _fnp

                    _hi = df["high"].values
                    _lo = df["low"].values
                    _cl = df["close"].values
                    _op = df["open"].values
                    _tr = _fnp.maximum(
                        _hi[1:] - _lo[1:],
                        _fnp.abs(_hi[1:] - _cl[:-1]),
                        _fnp.abs(_lo[1:] - _cl[:-1]),
                    )
                    _atr14 = float(_fnp.nanmean(_tr[-14:])) if len(_tr) >= 14 else 0.0
                    _last_body = abs(float(_cl[-1]) - float(_op[-1]))
                    if _atr14 > 0:
                        _body_ratio = _last_body / _atr14
                        if _body_ratio > 5.0:
                            logger.warning(
                                f"[FLASH] ⛔ Hard-veto: candle body {_body_ratio:.1f}× ATR "
                                f"— news spike detected, blocking signal"
                            )
                            return 0, {
                                "timestamp": timestamp,
                                "regime": "UNKNOWN",
                                "reasoning": f"flash_veto_{_body_ratio:.1f}x_atr",
                                "final_signal": 0,
                                "signal_quality": 0.0,
                                "mr_signal": 0,
                                "mr_confidence": 0.0,
                                "tf_signal": 0,
                                "tf_confidence": 0.0,
                                "ema_signal": 0,
                                "ema_confidence": 0.0,
                            }
                        elif _body_ratio > 3.0:
                            logger.warning(
                                f"[FLASH] ⚠️ Soft-veto: candle body {_body_ratio:.1f}× ATR "
                                f"— quality discounted 40%"
                            )
                            _flash_discount = 0.60
            except Exception:
                _flash_discount = 1.0

            # ═══════════════════════════════════════════════════════════════
            # T3.3: NY OPEN HOUR BLOCK (13:00–13:59 UTC)
            # TF signals at NY open: 53% WR, -21.2% P&L (stop-hunting territory).
            # Trades 1–2 hours later: 60% WR, +101.5% P&L.
            # BTC trades 24/7 — only block market-hours assets.
            # NOTE: FX pairs (EURUSD, EURJPY) are intentionally excluded.
            # 13:00 UTC = London/NY overlap — the highest-liquidity, most
            # directional hour of the FX session. Blocking it kills best entries.
            # The stop-hunt data that justified this block was from USTEC/GOLD.
            # ═══════════════════════════════════════════════════════════════
            _hour_utc = _dt.utcnow().hour
            if _hour_utc == 13 and self.asset_type in (
                "USTEC",
                "GOLD",
                "USOIL",
                "GBPAUD",
            ):
                logger.info(
                    f"[SESSION] ⏸️ NY open hour block — no new entries for {self.asset_type}"
                )
                return 0, {
                    "timestamp": timestamp,
                    "regime": "UNKNOWN",
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

            # ═══════════════════════════════════════════════════════════════
            # T3.4: ECONOMIC CALENDAR BLOCK
            # Trading through NFP/FOMC/CPI on a 1H timeframe is gambling.
            # Block N hours before each high-impact event.
            # ═══════════════════════════════════════════════════════════════
            if self._econ_events:
                from datetime import timezone as _tz, timedelta as _td

                _utc_now = _dt.now(_tz.utc)
                _asset = self.asset_type
                for _evt in self._econ_events:
                    try:
                        _evt_time = _dt.fromisoformat(
                            _evt["datetime"].replace("Z", "+00:00")
                        )
                        _hours_before = _evt.get("block_hours_before", 2)
                        _hours_after = _evt.get("block_hours_after", 0)
                        _block_start = _evt_time - _td(hours=_hours_before)
                        _block_end = _evt_time + _td(hours=_hours_after)
                        if _block_start <= _utc_now < _block_end:
                            _affected = _evt.get("currencies", _evt.get("currency"))
                            if isinstance(_affected, str):
                                _affected = [_affected]
                            _affected = _affected or []
                            _blocked = (
                                (_asset in ("BTC", "BTCUSDT") and "USD" in _affected)
                                or (_asset in ("GOLD", "XAUUSD") and "USD" in _affected)
                                or (
                                    _asset == "EURUSD"
                                    and ("EUR" in _affected or "USD" in _affected)
                                )
                                or (
                                    _asset == "EURJPY"
                                    and ("EUR" in _affected or "JPY" in _affected)
                                )
                                or (
                                    _asset in ("USTEC", "US100", "NAS100")
                                    and "USD" in _affected
                                )
                                or (
                                    _asset == "GBPAUD"
                                    and ("GBP" in _affected or "AUD" in _affected)
                                )
                                or (
                                    _asset in ("USOIL", "USOILM")
                                    and ("USD" in _affected or "OIL" in _affected)
                                )
                                or (
                                    not _affected
                                )  # fallback: block all if no currencies listed
                            )
                            if _blocked:
                                _mins_to_evt = (
                                    _evt_time - _utc_now
                                ).total_seconds() / 60
                                logger.warning(
                                    f"[CALENDAR] ⏸️ Blocking {_asset} — "
                                    f"{_evt['event']} in {_mins_to_evt:.0f}min"
                                )
                                return 0, {
                                    "timestamp": timestamp,
                                    "regime": "UNKNOWN",
                                    "reasoning": f"econ_calendar_{_evt['event'].replace(' ', '_')}",
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

            # Step 1: Prepare context
            is_bull = is_bull_market
            regime_conf = governor_data.get("confidence", 0.5) if governor_data else 0.5
            regime_name = (
                governor_data.get("regime", "NEUTRAL") if governor_data else "NEUTRAL"
            )

            # ✨ NEW: Advanced Confluence Overlays
            div_res = self.divergence_detector.analyze(df)
            br_res = self.break_retest_validator.validate(df, self.asset_type)

            # D.1: Update trend lifecycle in composite state
            # Extract bar timestamp from governor_data when available.  In a
            # backtest this is the historical bar's datetime; in live trading it
            # is the current wall-clock time — both are correct for their context.
            _bar_dt = None
            if governor_data:
                _ts = governor_data.get("timestamp")
                if _ts:
                    try:
                        from datetime import datetime as _dtp

                        _bar_dt = (
                            _dtp.fromisoformat(_ts) if isinstance(_ts, str) else _ts
                        )
                    except Exception:
                        _bar_dt = None

            if state is not None:
                # Relocated to CompositeStateBuilder along with _build_composite_state
                # — get_aggregated_signal stays on the shell, so it reaches across to
                # the builder instance rather than calling a sibling method directly.
                self._cs_builder._update_trend_lifecycle(state, regime_name, current_dt=_bar_dt)
                # regime_age_ratio is now fully populated inside _update_trend_lifecycle;
                # the separate re-calculation block below is no longer needed.

            # Update stats based on provided regime
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            # STEP 2: Get strategy signals
            # Pass 4H context to strategies if available
            df_4h = governor_data.get("df_4h") if governor_data else None
            logger.debug(
                f"[MR INPUT] {self.asset_type}: df_4h={'present, ' + str(len(df_4h)) + ' bars' if df_4h is not None else 'MISSING'}"
            )

            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(
                df, df_4h=df_4h, composite_state=state
            )
            # L10: pass composite_state through so TF/EMA's Livermore awareness
            # nudge (flag-gated, see base_strategy.py) can see live LSM state.
            tf_signal, tf_conf = self.s_trend_following.generate_signal(
                df, df_4h=df_4h, composite_state=state
            )
            ema_signal, ema_conf = self.s_ema.generate_signal(
                df, df_4h=df_4h, composite_state=state
            )

            # Store originals for logging
            mr_original = mr_signal
            tf_original = tf_signal

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: LIVERMORE HARD VETO LAYER — RETIRED (2026-07-01)
            # Blocks A-D previously ran here, duplicating what main.py's
            # POST-SIGNAL LIVERMORE COUNTER-TREND BLOCK (4H-aware, all aggregator
            # types) now handles fully and correctly. The old blocks were:
            #   A — NATURAL_REBOUND + any LONG
            #   B — SECONDARY_REBOUND + LONG without dual confirmation
            #   C — MR counter-trend during NATURAL states
            #   D — TF/EMA shorts during NATURAL_RETRACEMENT
            # All of these are now covered by main.py, which runs after all
            # aggregators have resolved their final signal, is 4H-aware (uses
            # both livermore_state_1h and livermore_state_4h), and covers council
            # mode too (Blocks A-D only ran in the Performance path). The
            # reasoning tags produced by main.py's replacement are registered in
            # system_validator.py's HARD_VETO_LAYER liveness check so the health
            # metric continues to fire correctly.
            # ─────────────────────────────────────────────────────────────

            # ═══════════════════════════════════════════════════════════════
            # T3.5: BTC FUNDING RATE Z-SCORE CONFIDENCE MULTIPLIER
            # Extreme funding rates (Z ≥ 2.0) indicate crowded positioning.
            # Over-leveraged longs → MR short setups become highest probability.
            # Z-score adapts to sustained bull runs; static threshold doesn't.
            # ═══════════════════════════════════════════════════════════════
            _funding_z = (
                governor_data.get("funding_rate_zscore", 0.0) if governor_data else 0.0
            )
            if self.asset_type in ("BTC", "BTCUSDT") and abs(_funding_z) >= 2.0:
                if mr_signal != 0:
                    mr_conf = min(1.0, mr_conf * 1.15)
                    logger.info(
                        f"[FUNDING] Extreme positioning (Z={_funding_z:+.1f}): "
                        f"MR conf boosted to {mr_conf:.2f}"
                    )

            # ═══════════════════════════════════════════════════════════════
            # T3.6: DXY PROXY CONFIDENCE MULTIPLIER
            # Rising EUR/USD = falling dollar = bullish for GOLD/USTEC/EURJPY.
            # Computed from already-traded EUR/USD data — zero API cost.
            # ═══════════════════════════════════════════════════════════════
            _dxy_falling = governor_data.get("dxy_falling") if governor_data else None
            if _dxy_falling is not None and self.asset_type in (
                "GOLD",
                "USTEC",
                "EURJPY",
                "USOIL",
            ):
                if self.asset_type == "GOLD":
                    # Dollar weakness → gold strength
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(
                            f"[DXY] Weak dollar: GOLD TF BUY conf boosted to {tf_conf:.2f}"
                        )
                    elif not _dxy_falling and tf_signal == -1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(
                            f"[DXY] Strong dollar: GOLD TF SELL conf boosted to {tf_conf:.2f}"
                        )
                elif self.asset_type == "USTEC":
                    # Dollar weakness generally supportive of risk assets
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.05)
                        logger.debug(
                            f"[DXY] Weak dollar: USTEC TF BUY conf boosted to {tf_conf:.2f}"
                        )
                elif self.asset_type == "USOIL":
                    # Dollar weakness = oil strength (inverse correlation)
                    if _dxy_falling and tf_signal == 1:  # Weak dollar + BUY oil
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(
                            f"[DXY] Weak dollar: USOIL TF BUY conf boosted to {tf_conf:.2f}"
                        )
                    elif (
                        not _dxy_falling and tf_signal == -1
                    ):  # Strong dollar + SELL oil
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(
                            f"[DXY] Strong dollar: USOIL TF SELL conf boosted to {tf_conf:.2f}"
                        )

            # ═══════════════════════════════════════════════════════════════
            # T2.6: CONSECUTIVE CANDLE CONFIDENCE MULTIPLIER
            # BTC after 3 consecutive same-direction bars + low ADX: 66% MR WR
            # GOLD after 5 consecutive bars: 85% TF continue rate
            # ADX guard prevents counter-trend fading during strong momentum
            # (MR fading streaks in high ADX: 33% WR on GOLD, 56% on BTC).
            # This is a confidence bonus, not a new gate — fails silently.
            # ═══════════════════════════════════════════════════════════════
            try:
                _closes = df["close"].values
                _consec = 0
                for _i in range(len(_closes) - 1, max(len(_closes) - 10, 0), -1):
                    if _i == 0:
                        break
                    if _closes[_i] > _closes[_i - 1]:
                        if _consec >= 0:
                            _consec += 1
                        else:
                            break
                    else:
                        if _consec <= 0:
                            _consec -= 1
                        else:
                            break

                # Compute ADX for the guard
                _adx_guard = 25.0  # default if calculation fails
                try:
                    import talib as _talib_c

                    _adx_raw = _talib_c.ADX(
                        df["high"].values, df["low"].values, _closes, timeperiod=14
                    )[-1]
                    if not np.isnan(_adx_raw):
                        _adx_guard = _adx_raw
                except Exception:
                    pass

                # BTC: boost MR when price has made 3+ consecutive candles in one
                # direction AND momentum is low — classic mean reversion setup
                if self.asset_type == "BTC" and abs(_consec) >= 3 and _adx_guard < 25:
                    if mr_signal != 0:
                        mr_conf = min(1.0, mr_conf * 1.20)
                        logger.debug(
                            f"[CANDLE] BTC {_consec}-bar streak + low ADX ({_adx_guard:.0f}): "
                            f"MR conf boosted to {mr_conf:.2f}"
                        )

                # GOLD: boost TF when riding a 5+ bar streak — trend continuation
                if self.asset_type == "GOLD" and abs(_consec) >= 5:
                    if tf_signal != 0:
                        tf_conf = min(1.0, tf_conf * 1.15)
                        logger.debug(
                            f"[CANDLE] GOLD {_consec}-bar streak: "
                            f"TF conf boosted to {tf_conf:.2f}"
                        )
            except Exception:
                pass  # Bonus only — never block execution on failure

            # Extract regime score for Gatekeeper (Phase 3)
            regime_score = (
                governor_data.get("regime_score", 0.0) if governor_data else 0.0
            )
            regime_is_bullish = (
                governor_data.get("is_bullish", False) if governor_data else False
            )
            regime_is_bearish = (
                governor_data.get("is_bearish", False) if governor_data else False
            )

            # ── Item 19b: 4H Livermore / macro-regime disagreement gate ──────
            # Flag-gated (phase_config.lsm_regime_disagreement_gate_enabled,
            # default False — same switch used in council_aggregator.py and
            # main.py for this item). When enabled and the live 4H Livermore
            # state's lean disagrees with the EMA-derived macro regime lean,
            # strip that lean's directional gating power here so the
            # gatekeeper below falls back through to its NEUTRAL branch
            # instead of hard-blocking/penalising counter-trend signals on a
            # regime label the structural state machine actively contradicts.
            _lsm_gate_enabled = bool(
                getattr(self, "phase_config", {}).get(
                    "lsm_regime_disagreement_gate_enabled", False
                )
            )
            if _lsm_gate_enabled:
                _lsm_4h_for_regime = (
                    getattr(state, "livermore_state_4h", None)
                    if state is not None
                    else None
                )
                _lsm_lean_sa = (
                    "bullish"
                    if _lsm_4h_for_regime
                    in ("MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT")
                    else (
                        "bearish"
                        if _lsm_4h_for_regime
                        in ("MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND")
                        else None
                    )
                )
                if _lsm_lean_sa is not None:
                    if regime_is_bullish and _lsm_lean_sa != "bullish":
                        logger.info(
                            f"[GATEKEEPER] {self.asset_type} 4H Livermore lean ({_lsm_lean_sa}) "
                            f"disagrees with bullish regime — stripping bullish lean (lsm_regime_disagreement_gate_enabled)"
                        )
                        regime_is_bullish = False
                    if regime_is_bearish and _lsm_lean_sa != "bearish":
                        logger.info(
                            f"[GATEKEEPER] {self.asset_type} 4H Livermore lean ({_lsm_lean_sa}) "
                            f"disagrees with bearish regime — stripping bearish lean (lsm_regime_disagreement_gate_enabled)"
                        )
                        regime_is_bearish = False

            # ═══════════════════════════════════════════════════════════════
            # ENHANCED GATEKEEPER — Confidence Scaling + Transition Evidence
            # ═══════════════════════════════════════════════════════════════
            # FULL regimes (|regime_score| >= 1.0): hard block counter-trend.
            # SLIGHTLY regimes (|regime_score| < 1.0): penalise confidence,
            #   with penalty modulated by TransitionEvidence (2+ conditions
            #   required before any reduction is applied).
            # NEUTRAL: all strategies fire freely.
            # Explosive momentum overrule preserved for full-regime hard blocks.
            # ═══════════════════════════════════════════════════════════════
            if self.use_gatekeeper:
                # ─────────────────────────────────────────────────────────────
                # PHASE 2: LIVERMORE STRUCTURAL HOLD — 4H NATURAL STATES
                # When the 4H macro state is NATURAL (silent zone), no new entries.
                # NATURAL_RETRACEMENT and NATURAL_REBOUND are the two highest-
                # value waiting periods in Livermore's system — these are where
                # the trend breathes before continuation. Entering here was the
                # primary losing pattern in the pre-v3 bot.
                #
                # SECONDARY states are handled via Required Score Modifier (+0.40)
                # which raises the entry bar — entries still allowed but harder.
                # Phase 3A MR Mode 1 will add specific NATURAL_RETRACEMENT re-entry
                # logic (spring detection) that bypasses this hold for that one case.
                # ─────────────────────────────────────────────────────────────
                if state is not None and state.is_silent_zone:
                    if mr_signal != 0 or tf_signal != 0 or ema_signal != 0:
                        logger.info(
                            "[GATEKEEPER] %s 4H Livermore=%s (silent zone) → HOLD, no new entries",
                            self.asset_type,
                            state.livermore_state_4h or "NATURAL",
                        )
                        mr_signal = 0
                        mr_conf = 0.0
                        tf_signal = 0
                        tf_conf = 0.0
                        ema_signal = 0
                        ema_conf = 0.0
                # ─────────────────────────────────────────────────────────────

                # FAIL-CLOSED guard: no governor data = no regime context.
                # Trading without regime context risks entering during high-volatility
                # regime transitions where direction is unknown. Log a warning and skip.
                # Council already fails-closed; this aligns Performance with that posture.
                # NOTE: Only applies after the gatekeeper is enabled — early startup cycles
                # that haven't yet received governor data will be caught here and logged,
                # not silently treated as NEUTRAL.
                if not governor_data:
                    logger.warning(
                        f"[GATEKEEPER] ⚠️ No governor data for {self.asset_type} — "
                        f"fail-closed (no regime context). Skipping signal."
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "regime": "UNKNOWN",
                        "reasoning": "no_governor_data",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0,
                        "mr_confidence": 0.0,
                        "tf_signal": 0,
                        "tf_confidence": 0.0,
                        "ema_signal": 0,
                        "ema_confidence": 0.0,
                    }

                is_neutral = (regime_score == 0.0) or (
                    not regime_is_bullish and not regime_is_bearish
                )
                regime_strength = abs(regime_score)  # 0.5 for SLIGHTLY, 1.0 for full

                # Pull transition evidence if available
                _te = getattr(state, "_transition_evidence", None) if state else None
                _transition_score = _te.total_score if _te else 0.0
                _transition_conditions = _te.conditions_met if _te else 0

                if is_neutral:
                    # NEUTRAL: all strategies allowed in any direction.
                    # If TransitionDetector fired (NEUTRAL+TRANSITION trade), log the
                    # directional tilt so it's visible in logs/dashboard for calibration.
                    # No hard block or boost — NEUTRAL stays permissive by design.
                    if _te and _transition_conditions >= 2:
                        _tilt = (
                            f"BULLISH tilt ({_transition_score:+.3f})"
                            if _transition_score > 0.15
                            else (
                                f"BEARISH tilt ({_transition_score:+.3f})"
                                if _transition_score < -0.15
                                else f"no clear tilt ({_transition_score:+.3f})"
                            )
                        )
                        logger.info(
                            f"[GATEKEEPER] NEUTRAL+TRANSITION — all strategies allowed, "
                            f"evidence {_tilt} ({_transition_conditions}/4 conditions) [{self.asset_type}]"
                        )
                    else:
                        logger.debug(
                            f"[GATEKEEPER] NEUTRAL — all strategies allowed ({self.asset_type})"
                        )

                elif regime_is_bullish:
                    if regime_strength >= 1.0:
                        # FULL BULLISH: hard block counter-trend shorts.
                        # Exception 1: explosive momentum (existing)
                        # Exception 2 (TASK-8): strong transition evidence (≥3/4 sources,
                        #   score < -0.30) softens the hard block to a steep penalty.
                        _strong_bearish_reversal = (
                            _te is not None
                            and _transition_conditions >= 3
                            and _transition_score < -0.30
                        )
                        if _strong_bearish_reversal:
                            _full_bull_penalty = max(
                                0.35, 0.55 + _transition_score * 0.5
                            )
                            logger.info(
                                f"[GATEKEEPER] ⚡ FULL BULLISH softened by transition evidence "
                                f"({_transition_conditions}/4 conditions, score={_transition_score:+.3f}) → "
                                f"applying penalty {_full_bull_penalty:.2f} instead of hard block "
                                f"[{self.asset_type}]"
                            )
                            if tf_signal < 0:
                                tf_conf *= _full_bull_penalty
                                logger.info(
                                    f"[GATEKEEPER] ⚠️ PENALIZED SHORT (TF): Full bullish+evidence — "
                                    f"conf reduced to {tf_conf:.2f}"
                                )
                            if ema_signal < 0:
                                ema_signal = 0
                                ema_conf = 0.0
                            if mr_signal < 0:
                                mr_conf *= min(_full_bull_penalty + 0.10, 0.80)
                                logger.info(
                                    f"[GATEKEEPER] ⚠️ PENALIZED SHORT (MR): Full bullish+evidence — "
                                    f"conf reduced to {mr_conf:.2f}"
                                )
                            elif mr_signal > 0:
                                logger.info(
                                    f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in bullish+evidence for {self.asset_type}"
                                )
                        else:
                            if tf_signal < 0:
                                if self._is_explosive_momentum(df, -1):
                                    logger.info(
                                        f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bullish block for SHORT (TF)"
                                    )
                                else:
                                    logger.info(
                                        f"[GATEKEEPER] ❌ BLOCKED SHORT (TF): Strong bullish for {self.asset_type}"
                                    )
                                    tf_signal = 0
                                    tf_conf = 0.0
                            if ema_signal < 0:
                                if self._is_explosive_momentum(df, -1):
                                    logger.info(
                                        f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bullish block for SHORT (EMA)"
                                    )
                                else:
                                    logger.info(
                                        f"[GATEKEEPER] ❌ BLOCKED SHORT (EMA): Strong bullish for {self.asset_type}"
                                    )
                                    ema_signal = 0
                                    ema_conf = 0.0
                            if mr_signal < 0:
                                logger.info(
                                    f"[GATEKEEPER] ❌ BLOCKED SHORT (MR): Counter-trend in strong Bullish for {self.asset_type}"
                                )
                                mr_signal = 0
                                mr_conf = 0.0
                            elif mr_signal > 0:
                                logger.info(
                                    f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in strong Bullish for {self.asset_type}"
                                )

                    else:
                        # SLIGHTLY BULLISH: penalise shorts, don't kill them
                        # Bearish reversal evidence in a slightly bullish zone reduces penalty
                        _penalty = 0.50  # base: halve confidence
                        if _transition_conditions >= 2 and _transition_score < -0.15:
                            _penalty = max(0.30, _penalty + _transition_score)
                            logger.info(
                                f"[GATEKEEPER] TRANSITION evidence reduces SHORT penalty: "
                                f"{_penalty:.2f} (score={_transition_score:+.3f}, "
                                f"conditions={_transition_conditions}/4)"
                            )
                        if tf_signal < 0:
                            tf_conf *= _penalty
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED SHORT (TF): Slightly bullish — "
                                f"conf reduced to {tf_conf:.2f}"
                            )
                        if ema_signal < 0:
                            # EMA is a slow-trend follower — still zero in counter trend
                            ema_signal = 0
                            ema_conf = 0.0
                        if mr_signal < 0:
                            mr_conf *= min(
                                _penalty + 0.10, 0.80
                            )  # MR slightly less penalised
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED SHORT (MR): Slightly bullish — "
                                f"conf reduced to {mr_conf:.2f}"
                            )
                        elif mr_signal > 0:
                            logger.info(
                                f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in slightly Bullish for {self.asset_type}"
                            )

                elif regime_is_bearish:
                    if regime_strength >= 1.0:
                        # FULL BEARISH: hard block counter-trend longs.
                        # Exception 1: explosive momentum (existing)
                        # Exception 2 (TASK-8): strong transition evidence (≥3/4 sources,
                        #   score > 0.30) softens the hard block to a steep penalty instead.
                        #   This handles the "GOLD stuck at BEARISH all day while price
                        #   rallied 1%+" scenario where the day-open regime snapshot is stale.
                        _strong_bullish_reversal = (
                            _te is not None
                            and _transition_conditions >= 3
                            and _transition_score > 0.30
                        )
                        if _strong_bullish_reversal:
                            # Treat like a SLIGHTLY_BEARISH with extra caution
                            _full_bear_penalty = max(
                                0.35, 0.55 - _transition_score * 0.5
                            )
                            logger.info(
                                f"[GATEKEEPER] ⚡ FULL BEARISH softened by transition evidence "
                                f"({_transition_conditions}/4 conditions, score={_transition_score:+.3f}) → "
                                f"applying penalty {_full_bear_penalty:.2f} instead of hard block "
                                f"[{self.asset_type}]"
                            )
                            if tf_signal > 0:
                                tf_conf *= _full_bear_penalty
                                logger.info(
                                    f"[GATEKEEPER] ⚠️ PENALIZED LONG (TF): Full bearish+evidence — "
                                    f"conf reduced to {tf_conf:.2f}"
                                )
                            if ema_signal > 0:
                                # EMA is slow — zero it even with evidence; TF covers the bullish case
                                ema_signal = 0
                                ema_conf = 0.0
                            if mr_signal > 0:
                                mr_conf *= min(_full_bear_penalty + 0.10, 0.80)
                                logger.info(
                                    f"[GATEKEEPER] ⚠️ PENALIZED LONG (MR): Full bearish+evidence — "
                                    f"conf reduced to {mr_conf:.2f}"
                                )
                            elif mr_signal < 0:
                                logger.info(
                                    f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in bearish+evidence for {self.asset_type}"
                                )
                        else:
                            if tf_signal > 0:
                                if self._is_explosive_momentum(df, 1):
                                    logger.info(
                                        f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bearish block for LONG (TF)"
                                    )
                                else:
                                    logger.info(
                                        f"[GATEKEEPER] ❌ BLOCKED LONG (TF): Strong bearish for {self.asset_type}"
                                    )
                                    tf_signal = 0
                                    tf_conf = 0.0
                            if ema_signal > 0:
                                if self._is_explosive_momentum(df, 1):
                                    logger.info(
                                        f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bearish block for LONG (EMA)"
                                    )
                                else:
                                    logger.info(
                                        f"[GATEKEEPER] ❌ BLOCKED LONG (EMA): Strong bearish for {self.asset_type}"
                                    )
                                    ema_signal = 0
                                    ema_conf = 0.0
                            if mr_signal > 0:
                                logger.info(
                                    f"[GATEKEEPER] ❌ BLOCKED LONG (MR): Counter-trend in strong Bearish for {self.asset_type}"
                                )
                                mr_signal = 0
                                mr_conf = 0.0
                            elif mr_signal < 0:
                                logger.info(
                                    f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in strong Bearish for {self.asset_type}"
                                )

                    else:
                        # SLIGHTLY BEARISH: penalise longs, don't kill them
                        # Bullish reversal evidence in a slightly bearish zone reduces penalty
                        _penalty = 0.50
                        if _transition_conditions >= 2 and _transition_score > 0.15:
                            _penalty = max(0.30, _penalty - _transition_score)
                            logger.info(
                                f"[GATEKEEPER] TRANSITION evidence reduces LONG penalty: "
                                f"{_penalty:.2f} (score={_transition_score:+.3f}, "
                                f"conditions={_transition_conditions}/4)"
                            )
                        if tf_signal > 0:
                            tf_conf *= _penalty
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED LONG (TF): Slightly bearish — "
                                f"conf reduced to {tf_conf:.2f}"
                            )
                        if ema_signal > 0:
                            ema_signal = 0
                            ema_conf = 0.0
                        if mr_signal > 0:
                            mr_conf *= min(_penalty + 0.10, 0.80)
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED LONG (MR): Slightly bearish — "
                                f"conf reduced to {mr_conf:.2f}"
                            )
                        elif mr_signal < 0:
                            logger.info(
                                f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in slightly Bearish for {self.asset_type}"
                            )
            # --- End Enhanced Gatekeeper ---

            # Initialize core variables for details building (prevents UnboundLocalError if we skip)
            buy_score = 0.0
            sell_score = 0.0
            signal_quality = 0.0
            ai_validation_details = {}
            original_signal = 0
            final_signal = 0
            reasoning = "hold (no strategy agreement)"
            trade_type = "TREND"

            # COMPUTATIONAL OPTIMIZATION: If all signals are zero, skip heavy validation
            if mr_signal == 0 and tf_signal == 0 and ema_signal == 0:
                logger.debug(
                    f"[AGGREGATOR] {self.asset_type}: No signals to validate, skipping to end."
                )
                # We can skip to building the details dictionary
            else:
                # Ranging Detection — keeps position limits, counter-trend blocking
                # now handled exclusively by the Smart Gatekeeper above (T1.3 fix).
                is_ranging = regime_conf <= 0.50
                max_trades_override = None
                filter_reason = ""
                if is_ranging:
                    max_trades_override = 1
                    filter_reason = "Ranging Mode (Max 1 Trade)"

                signal_quality = max(mr_conf, tf_conf)

                # --- Directional Trap Filter Veto (T2.3: regime-aware) ---
                if mr_signal != 0 or tf_signal != 0 or ema_signal != 0:
                    test_direction = (
                        "long"
                        if (mr_signal > 0 or tf_signal > 0 or ema_signal > 0)
                        else "short"
                    )
                    # regime_aligned: signal direction matches macro regime.
                    # Fix #16: NEUTRAL regime has no directional opinion — both
                    # directions are valid so treat as aligned for both sides.
                    # Without this, LONG signals in NEUTRAL are always "not aligned"
                    # which triggers the 1.5× BTC volume check that doesn't apply
                    # to SHORT, creating a permanent short-bias in NEUTRAL.
                    _is_neutral_regime = (regime_score == 0.0) or (
                        not regime_is_bullish and not regime_is_bearish
                    )
                    _trap_aligned = (
                        _is_neutral_regime
                        or (test_direction == "long" and is_bull)
                        or (test_direction == "short" and not is_bull)
                    )
                    if not validate_candle_structure(
                        df,
                        self.asset_type,
                        direction=test_direction,
                        regime_confidence=regime_conf,
                        regime_aligned=_trap_aligned,
                    ):
                        logger.info(
                            f"[TRAP] VETO - Candidate rejected by structure check."
                        )
                        # Pass the REAL strategy signals through so the shadow trader
                        # can record and learn from trap-filter blocks (Bug 2 fix).
                        # Zeroing these out was hiding ~47 signals/cycle from the
                        # gate scorecard (76.6% WR, +13.3% P&L invisible to ML labels).
                        return 0, {
                            "timestamp": timestamp,
                            "regime": regime_name,
                            "reasoning": "blocked_by_trap_filter",
                            "final_signal": 0,
                            "original_signal": mr_signal
                            or tf_signal
                            or ema_signal,  # Pass the intended direction
                            "signal_quality": 0.0,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                            # Raw pre-gatekeeper values for shadow trader gate scoring
                            "mr_signal_raw": mr_original,
                            "tf_signal_raw": tf_original,
                        }

                # STEP 3: PRE-SCORE AI VALIDATION — DISABLED (T2.2)
                # Previously killed individual MR/TF votes before scoring, destroying
                # the consensus and independent evaluation pipeline.
                # Blocked 13 signals with 92.3% WR and +17.2% P&L.
                # AI validation now runs post-score via hybrid_validator.py.
                # The circuit breaker and stats objects are preserved for post-score use.
                ai_bypass = False
                ai_validation_details = {}

                # STEP 4: Calculate scores (MR + TF + EMA all contribute)
                buy_score, buy_explanation, buy_agreement = self._calculate_score(
                    df,
                    1,
                    mr_signal,
                    mr_conf,
                    tf_signal,
                    tf_conf,
                    ema_signal,
                    ema_conf,
                    is_bull,
                )
                sell_score, sell_explanation, sell_agreement = self._calculate_score(
                    df,
                    -1,
                    mr_signal,
                    mr_conf,
                    tf_signal,
                    tf_conf,
                    ema_signal,
                    ema_conf,
                    is_bull,
                )

                # STEP 5: Dynamic thresholds
                adj_buy_thresh, adj_sell_thresh = (
                    self.calculate_regime_adjusted_thresholds(is_bull, regime_conf)
                )

                # STEP 5B: REQUIRED SCORE MODIFIER (Livermore state) + Retest Engine
                # Layer 1: Livermore RSM — state-conditional base threshold delta.
                #   SECONDARY states +0.40; NATURAL states 0.00 (gated before here).
                # Layer 2: Retest Engine — entry context tier (CLEAN / BREAKOUT / WICK /
                #   CHASE_SOFT / CHASE_HARD / NO_LEVEL_NEARBY).
                # Both layers are additive; combined modifier capped at rsm_cap (1.50).
                # All numeric values in aggregator_presets.json — no magic numbers.
                _pending_retest_buy = None  # RetestResult for LONG; set below
                _pending_retest_sell = None  # RetestResult for SHORT; set below
                try:
                    _rsm_state = state.livermore_state_4h if state is not None else None
                    if _rsm_state is not None:
                        # ── Load config tables once (cached on instance) ────────────
                        if not hasattr(self, "_rsm_table"):
                            import json as _json_rsm

                            try:
                                with open("config/aggregator_presets.json") as _rsm_f:
                                    _rsm_cfg = _json_rsm.load(_rsm_f)
                                self._rsm_table = _rsm_cfg.get(
                                    "REQUIRED_SCORE_MODIFIER", {}
                                ).get("state_modifiers", {})
                                self._rsm_cap = _rsm_cfg.get(
                                    "REQUIRED_SCORE_MODIFIER", {}
                                ).get("modifier_cap", 1.50)
                                # Instantiate RetestEngine with its config section
                                from src.analysis.retest_engine import (
                                    RetestEngine as _RE,
                                )

                                self._retest_engine = _RE(
                                    _rsm_cfg.get("RETEST_ENGINE", {})
                                )
                            except Exception as _cfg_err:
                                logger.debug("[RSM] config load error: %s", _cfg_err)
                                self._rsm_table = {}
                                self._rsm_cap = 1.50
                                self._retest_engine = None

                        # ── Layer 1: Livermore RSM delta ───────────────────────────
                        _rsm_delta = self._rsm_table.get(_rsm_state, 0.0)

                        # ── Layer 2: Retest Engine (directional) ───────────────────
                        _re = getattr(self, "_retest_engine", None)
                        if _re is not None and state is not None:
                            try:
                                _pending_retest_buy = _re.classify(
                                    df, state, self.asset_type, direction=+1
                                )
                                _pending_retest_sell = _re.classify(
                                    df, state, self.asset_type, direction=-1
                                )
                            except Exception as _re_err:
                                logger.debug(
                                    "[RETEST] classify error (non-blocking): %s",
                                    _re_err,
                                )

                        # Fix 6a: Mode1 spring exemption. Mode 1 (Pullback
                        # Completion, mean_reversion.py) fires on 1H
                        # NATURAL_RETRACEMENT only after its own mandatory
                        # compression + spring + 2-of-4 optional gates already
                        # passed. That spring is local 1H structure and routinely
                        # has no nearby 4H level, so the Retest Engine's
                        # NO_LEVEL_NEARBY tier (+0.35/+0.40) was stacking on top
                        # of Mode1's own conviction check and raising
                        # adj_buy_thresh enough that the signal never cleared it
                        # — Mode 1 effectively never fired live. Widen the
                        # exemption: a LONG retest classified NO_LEVEL_NEARBY
                        # while Mode1 is the active 1H state pays no penalty.
                        _is_mode1_spring_buy = (
                            state is not None
                            and getattr(state, "livermore_state_1h", None)
                            == "NATURAL_RETRACEMENT"
                            and mr_signal == 1
                            and _pending_retest_buy is not None
                            and _pending_retest_buy.retest_type == "NO_LEVEL_NEARBY"
                        )
                        if _is_mode1_spring_buy:
                            logger.debug(
                                "[RETEST] %s Mode1 spring exemption — NO_LEVEL_NEARBY penalty waived",
                                self.asset_type,
                            )

                        # ── Combine and apply ──────────────────────────────────────
                        _retest_buy_delta = (
                            0.0
                            if _is_mode1_spring_buy
                            else (
                                _pending_retest_buy.modifier
                                if _pending_retest_buy is not None
                                else 0.0
                            )
                        )
                        _retest_sell_delta = (
                            _pending_retest_sell.modifier
                            if _pending_retest_sell is not None
                            else 0.0
                        )
                        _total_buy_delta = _rsm_delta + _retest_buy_delta
                        _total_sell_delta = _rsm_delta + _retest_sell_delta

                        if _total_buy_delta != 0.0 or _total_sell_delta != 0.0:
                            _base_buy = self.config["buy_threshold"]
                            _base_sell = self.config["sell_threshold"]
                            adj_buy_thresh = min(
                                _base_buy + self._rsm_cap,
                                adj_buy_thresh + _total_buy_delta,
                            )
                            adj_sell_thresh = min(
                                _base_sell + self._rsm_cap,
                                adj_sell_thresh + _total_sell_delta,
                            )
                            _buy_rt = (
                                _pending_retest_buy.retest_type
                                if _pending_retest_buy is not None
                                else "N/A"
                            )
                            _sell_rt = (
                                _pending_retest_sell.retest_type
                                if _pending_retest_sell is not None
                                else "N/A"
                            )
                            logger.info(
                                "[RSM+RETEST] %s Livermore=%s rsm=%.2f | "
                                "buy=%s(\u0394%.2f) sell=%s(\u0394%.2f) | "
                                "buy_thresh=%.2f sell_thresh=%.2f",
                                self.asset_type,
                                _rsm_state,
                                _rsm_delta,
                                _buy_rt,
                                _retest_buy_delta,
                                _sell_rt,
                                _retest_sell_delta,
                                adj_buy_thresh,
                                adj_sell_thresh,
                            )
                except Exception as _rsm_err:
                    logger.debug("[RSM] modifier error (non-blocking): %s", _rsm_err)

                # STEP 6: Make decision
                if buy_score >= adj_buy_thresh and buy_score > sell_score:
                    final_signal = 1
                elif sell_score >= adj_sell_thresh and sell_score > buy_score:
                    final_signal = -1

                reasoning = (
                    f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})"
                    if final_signal == 1
                    else (
                        f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})"
                        if final_signal == -1
                        else f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                    )
                )
                original_signal = final_signal

                # ── STEP 6B: Confluence-adjusted re-score ─────────────────────
                # Run the Confluence Engine NOW (with the known signal direction)
                # so pattern/regime adjustments to tf_conf/mr_conf feed back into
                # the scoring and can change the decision. Previously this ran
                # post-decision and only affected the response dict (dead code).
                if final_signal != 0 and state is not None:
                    try:
                        _tf_adj, _mr_adj, state = self._score_confluence(
                            state, tf_conf, mr_conf, signal=final_signal
                        )
                        if abs(_tf_adj - tf_conf) > 1e-4 or abs(_mr_adj - mr_conf) > 1e-4:
                            _b_adj, _, _ = self._calculate_score(
                                df, 1, mr_signal, _mr_adj, tf_signal, _tf_adj,
                                ema_signal, ema_conf, is_bull,
                            )
                            _s_adj, _, _ = self._calculate_score(
                                df, -1, mr_signal, _mr_adj, tf_signal, _tf_adj,
                                ema_signal, ema_conf, is_bull,
                            )
                            if _b_adj >= adj_buy_thresh and _b_adj > _s_adj:
                                final_signal = 1
                            elif _s_adj >= adj_sell_thresh and _s_adj > _b_adj:
                                final_signal = -1
                            else:
                                final_signal = 0
                            buy_score, sell_score = _b_adj, _s_adj
                            if final_signal != original_signal:
                                reasoning = (
                                    f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})"
                                    if final_signal == 1
                                    else (
                                        f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})"
                                        if final_signal == -1
                                        else (
                                            f"hold — confluence overrode {original_signal:+d} "
                                            f"(buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                                        )
                                    )
                                )
                        tf_conf, mr_conf = _tf_adj, _mr_adj
                    except Exception as _ce:
                        logger.debug(f"[CONFLUENCE] Re-score failed (non-blocking): {_ce}")

                # Write entry_type to CompositeState for VTM routing (Phase 3B).
                # Directional: uses the retest result that matched final_signal direction.
                if state is not None:
                    try:
                        if final_signal == 1 and _pending_retest_buy is not None:
                            state.entry_type = _pending_retest_buy.entry_type
                            state.range_high = _pending_retest_buy.range_high
                            state.range_low = _pending_retest_buy.range_low
                        elif final_signal == -1 and _pending_retest_sell is not None:
                            state.entry_type = _pending_retest_sell.entry_type
                            state.range_high = _pending_retest_sell.range_high
                            state.range_low = _pending_retest_sell.range_low
                        else:
                            state.entry_type = None
                            state.range_high = None
                            state.range_low = None
                    except Exception:
                        pass

                # ── CANDLE MOMENTUM REVERSAL GATE ───────────────────────────
                # Mirror of the CMR veto in council_aggregator. Blocks a
                # trend-aligned signal when the most recent closed 1H candles
                # are unanimously moving AGAINST the proposed direction.
                # "3 green candles → don't add a new short" principle.
                # Only applied to trend-aligned signals (SELL in BEARISH,
                # BUY in BULLISH). Counter-trend setups are exempt — they
                # intentionally trade against recent momentum.
                if final_signal != 0:
                    _cmr_trend_aligned = (final_signal == -1 and not is_bull) or (
                        final_signal == 1 and is_bull
                    )
                    if _cmr_trend_aligned:
                        try:
                            _cmr_cfg = self.config.get("momentum_alignment", {})
                            _cmr_enabled = _cmr_cfg.get("enabled", True)
                            _cmr_candles = _cmr_cfg.get("candles", 3)
                            _cmr_agree = _cmr_cfg.get("min_agreement", 3)

                            if _cmr_enabled and len(df) >= _cmr_candles + 2:
                                _cmr_recent = df.iloc[-(_cmr_candles + 1) : -1]
                                _cmr_opens = _cmr_recent["open"].values
                                _cmr_closes = _cmr_recent["close"].values
                                _cmr_bull = int((_cmr_closes > _cmr_opens).sum())
                                _cmr_bear = int((_cmr_closes < _cmr_opens).sum())

                                # Reuse _atr14 from flash-veto block above if present
                                _cmr_atr = locals().get("_atr14", 0.0)
                                if _cmr_atr <= 0:
                                    try:
                                        import numpy as _cnp

                                        _hi = df["high"].values
                                        _lo = df["low"].values
                                        _cl = df["close"].values
                                        _tr = _cnp.maximum(
                                            _hi[1:] - _lo[1:],
                                            _cnp.abs(_hi[1:] - _cl[:-1]),
                                            _cnp.abs(_lo[1:] - _cl[:-1]),
                                        )
                                        _cmr_atr = (
                                            float(_cnp.nanmean(_tr[-14:]))
                                            if len(_tr) >= 14
                                            else 0.0
                                        )
                                    except Exception:
                                        _cmr_atr = 0.0

                                _avg_body = float(abs(_cmr_closes - _cmr_opens).mean())
                                _min_body = _cmr_atr * 0.08

                                # Net displacement (first open → last close) in ATRs.
                                # Requires a genuine sustained move, not just a majority
                                # of tiny candles pointing the wrong way.
                                _cmr_net = float(_cmr_closes[-1] - _cmr_opens[0])
                                _cmr_adv_mult = float(_cmr_cfg.get("adverse_atr_mult", 0.30))

                                if _avg_body >= _min_body:
                                    _cmr_atr_pos = _cmr_atr if _cmr_atr > 0 else 1e-8
                                    if (
                                        final_signal == -1
                                        and _cmr_bull >= _cmr_agree
                                        and _cmr_net >= _cmr_adv_mult * _cmr_atr_pos
                                    ):
                                        _cmr_reason = (
                                            f"{_cmr_bull}/{_cmr_candles} recent candles bullish "
                                            f"AND net +{_cmr_net / _cmr_atr_pos:.2f} ATR "
                                            f"— momentum opposing SELL"
                                        )
                                        logger.info(
                                            f"[CMR] ⛔ {self.asset_type} SELL blocked — "
                                            f"{_cmr_bull}/{_cmr_candles} candles bullish, "
                                            f"net {_cmr_net / _cmr_atr_pos:+.2f} ATR against short entry."
                                        )
                                        return 0, {
                                            "timestamp": timestamp,
                                            "regime": regime_name,
                                            "reasoning": "blocked_by_candle_momentum_reversal",
                                            "final_signal": 0,
                                            "original_signal": final_signal,
                                            "signal_quality": 0.0,
                                            "mr_signal": mr_signal,
                                            "mr_confidence": mr_conf,
                                            "tf_signal": tf_signal,
                                            "tf_confidence": tf_conf,
                                            "ema_signal": ema_signal,
                                            "ema_confidence": ema_conf,
                                            "cmr_reason": _cmr_reason,
                                        }
                                    elif (
                                        final_signal == 1
                                        and _cmr_bear >= _cmr_agree
                                        and -_cmr_net >= _cmr_adv_mult * _cmr_atr_pos
                                    ):
                                        _cmr_reason = (
                                            f"{_cmr_bear}/{_cmr_candles} recent candles bearish "
                                            f"AND net {_cmr_net / _cmr_atr_pos:.2f} ATR "
                                            f"— momentum opposing BUY"
                                        )
                                        logger.info(
                                            f"[CMR] ⛔ {self.asset_type} BUY blocked — "
                                            f"{_cmr_bear}/{_cmr_candles} candles bearish, "
                                            f"net {_cmr_net / _cmr_atr_pos:+.2f} ATR against long entry."
                                        )
                                        return 0, {
                                            "timestamp": timestamp,
                                            "regime": regime_name,
                                            "reasoning": "blocked_by_candle_momentum_reversal",
                                            "final_signal": 0,
                                            "original_signal": final_signal,
                                            "signal_quality": 0.0,
                                            "mr_signal": mr_signal,
                                            "mr_confidence": mr_conf,
                                            "tf_signal": tf_signal,
                                            "tf_confidence": tf_conf,
                                            "ema_signal": ema_signal,
                                            "ema_confidence": ema_conf,
                                            "cmr_reason": _cmr_reason,
                                        }
                        except Exception as _cmr_exc:
                            logger.debug(
                                f"[CMR] Check failed, allowing signal: {_cmr_exc}"
                            )
                # ── END CMR GATE ─────────────────────────────────────────────

                # Fix F: removed hard cap at 0.7 — score can now reflect true 3-strategy consensus
                raw_quality = max(buy_score, sell_score)
                if buy_agreement < 2 and sell_agreement < 2:
                    raw_quality *= 0.7
                if (final_signal == 1 and is_bull) or (
                    final_signal == -1 and not is_bull
                ):
                    raw_quality *= 1.15
                signal_quality = min(raw_quality, 1.0)

                # Section 2.4B: Boost quality when transition evidence strongly agrees
                if (
                    state
                    and hasattr(state, "_transition_evidence")
                    and state._transition_evidence
                ):
                    if state._transition_evidence.conditions_met >= 3:
                        _te_boost = abs(state._transition_evidence.total_score) * 0.15
                        _te_dir = state._transition_evidence.direction
                        if (final_signal == 1 and _te_dir == "BULLISH_REVERSAL") or (
                            final_signal == -1 and _te_dir == "BEARISH_REVERSAL"
                        ):
                            signal_quality = min(
                                1.0, signal_quality * (1.0 + _te_boost)
                            )
                            logger.debug(
                                f"[QUALITY] Transition evidence boost: "
                                f"×{1.0 + _te_boost:.3f} → {signal_quality:.2f}"
                            )

                # ── O3b: Liquidity and overextension quality gate ─────────────────
                # Two orphaned signals that were computed but never fed into the
                # quality pipeline that already has the flash-crash discount.
                if state is not None and final_signal != 0:
                    _spread_spike = bool(getattr(state, "spread_velocity_spike", False))
                    _dist_z       = float(getattr(state, "distance_zscore", 0.0))

                    # spread_velocity_spike = liquidity withdrawing fast.
                    # Discount signal quality — entering into a spread spike risks
                    # filling at a far worse price than the model expects.
                    if _spread_spike:
                        signal_quality = round(signal_quality * 0.80, 4)
                        logger.debug(
                            f"[QUALITY] {self.asset_type}: spread_velocity_spike — "
                            f"quality discounted to {signal_quality:.3f}"
                        )

                    # distance_zscore > 2.5 = statistically overextended.
                    # Reduces quality proportionally. Complements the existing
                    # is_parabolic check (which is a harder threshold) with a
                    # graduated quality reduction at lower Z-scores.
                    if _dist_z > 2.5 and not bool(getattr(state, "is_parabolic", False)):
                        _overext_discount = min(0.85, 1.0 - ((_dist_z - 2.5) * 0.06))
                        signal_quality = round(signal_quality * _overext_discount, 4)
                        logger.debug(
                            f"[QUALITY] {self.asset_type}: distance_zscore={_dist_z:.2f} — "
                            f"quality discounted to {signal_quality:.3f}"
                        )

                if (
                    final_signal != 0
                    and signal_quality < self.config["min_signal_quality"]
                ):
                    final_signal = 0
                    reasoning = f"hold_lowquality (original:{reasoning}, quality:{signal_quality:.2f})"

                # ═══════════════════════════════════════════════════════════
                # INDEPENDENT STRATEGY EVALUATION (T1.1 fix)
                # Consensus failed (final_signal still 0). Check if any single
                # strategy has enough individual confidence to fire alone.
                # Priority: TF > EMA > MR (based on solo P&L simulation data).
                # allow_single_override and single_override_threshold are config
                # keys that existed in presets but were never read — now wired.
                # ═══════════════════════════════════════════════════════════
                if final_signal == 0 and self.allow_independent:
                    candidates = []

                    # TF: use post-gatekeeper signal (consistent with MR/EMA treatment).
                    # tf_original pre-bypass was causing asymmetric gatekeeper application.
                    if (
                        tf_signal != 0
                        and tf_conf >= self.independent_thresholds["trend_following"]
                    ):
                        candidates.append(("TF", tf_signal, tf_conf))

                    # EMA: evaluated post-gatekeeper (gatekeeper treats EMA same as TF)
                    if (
                        ema_signal != 0
                        and ema_conf >= self.independent_thresholds["ema"]
                    ):
                        candidates.append(("EMA", ema_signal, ema_conf))

                    # MR: use post-gatekeeper signal (Smart Gatekeeper already filtered it)
                    if (
                        mr_signal != 0
                        and mr_conf >= self.independent_thresholds["mean_reversion"]
                    ):
                        candidates.append(("MR", mr_signal, mr_conf))

                    if candidates:
                        # Sort by confidence descending; TF wins ties (listed first)
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        best_name, best_signal, best_conf = candidates[0]
                        final_signal = best_signal
                        signal_quality = (
                            best_conf * 0.85
                        )  # Solo signals get a small quality discount

                        # Multi-strategy confirmation bonus: any agreeing strategy lifts quality
                        agreeing = [c for c in candidates if c[1] == best_signal]
                        if len(agreeing) >= 2:
                            signal_quality = min(1.0, best_conf * 1.1)

                        reasoning = (
                            f"{'BUY' if final_signal == 1 else 'SELL'} "
                            f"(independent:{best_name}, conf:{best_conf:.2f}, "
                            f"confirmations:{len(agreeing)})"
                        )
                        logger.info(
                            f"[INDEPENDENT] {self.asset_type}: {best_name} fires alone "
                            f"(conf={best_conf:.2f}, aligned={len(agreeing)} strategies)"
                        )

                        # O3b parity: the overextension discount above (lines ~4733-4743)
                        # only ever applied to consensus signals — signal_quality gets
                        # fully overwritten by the independent-fire block, so a solo TF/EMA/MR
                        # fire chasing an already-extreme move never got discounted at all.
                        # Apply the same guard here, directionally: only when the fired
                        # signal chases the extension (LONG into EMA_ABOVE_FAR, SHORT into
                        # EMA_BELOW_FAR) rather than fading it.
                        if state is not None:
                            _is_extreme = bool(getattr(state, "is_parabolic", False))
                            _ema_status = getattr(state, "ema_50_status", None)
                            _chasing_extension = (
                                (final_signal == 1 and _ema_status == "EMA_ABOVE_FAR")
                                or (final_signal == -1 and _ema_status == "EMA_BELOW_FAR")
                            )
                            if _is_extreme and _chasing_extension:
                                _dist_z = float(getattr(state, "distance_zscore", 0.0))
                                _overext_discount = min(0.85, 1.0 - max(0.0, _dist_z - 2.5) * 0.06)
                                signal_quality = round(signal_quality * _overext_discount, 4)
                                logger.debug(
                                    f"[QUALITY] {self.asset_type}: independent fire chasing "
                                    f"extension (ema={_ema_status}, z={_dist_z:.2f}) — "
                                    f"quality discounted to {signal_quality:.3f}"
                                )

                        if final_signal != 0 and signal_quality < self.config["min_signal_quality"]:
                            final_signal = 0
                            reasoning = f"hold_lowquality_independent (quality:{signal_quality:.2f})"

                        # BTC-specific: independent-fire LONGs on BTC showed no
                        # discernible edge across a full trending year (worse
                        # take-profit:stop-loss hit ratio than SHORT, and no
                        # discriminating entry-quality signal found across three
                        # tested hypotheses — volatility-expansion tag, ADX slope,
                        # overextension/parabolic z-score). Suppress independent
                        # LONG fires for BTC only; SHORT and consensus-path (2+
                        # strategy agreement) LONGs are unaffected. Toggleable via
                        # phase_config so its effect can be isolated from other
                        # gates (e.g. ai_validator_gates_performance_mode) during
                        # backtest soak comparisons — default True preserves the
                        # validated behavior.
                        _btc_veto_enabled = getattr(self, "phase_config", {}).get(
                            "btc_independent_long_veto_enabled", True
                        )
                        if self.asset_type == "BTC" and final_signal == 1 and _btc_veto_enabled:
                            logger.info(
                                f"[INDEPENDENT-VETO] BTC: suppressed independent LONG "
                                f"from {best_name} (conf={best_conf:.2f})"
                            )
                            final_signal = 0
                            reasoning = (
                                f"hold_btc_independent_long_suppressed "
                                f"(was {best_name}, conf:{best_conf:.2f})"
                            )

                # Update original_signal to capture any consensus OR independent signal
                # before final filters (volatility, governor, etc) are applied.
                original_signal = final_signal

                # World-Class Filters
                # Fix D: profit filter removed — it duplicated the volatility filter (both
                # measured ATR/price%) while adding an independent failure point that blocked
                # valid signals in low-ATR trending regimes (e.g. GOLD steady grind moves).
                # Fix C: ATR expansion filter replaced with ADX trend confirmation (see method).
                if final_signal != 0 and self.enable_filters:
                    gov_passed, trade_type = self._check_governor_filter(
                        df, final_signal
                    )
                    if not gov_passed:
                        final_signal = 0
                        reasoning = "blocked_by_governor"
                    else:
                        vol_passed, _ = self._check_volatility_filter(df)
                        if not vol_passed:
                            final_signal = 0
                            reasoning = "low_volatility"
                        else:
                            # Sniper filter removed (Phase 0B) — CNN-LSTM disconnected.
                            # Filter chain: Governor → Volatility → ATR Expansion.
                            if final_signal != 0:
                                atr_exp_passed = self._check_atr_expansion_filter(
                                    df, trade_type
                                )
                                if not atr_exp_passed:
                                    # Same advisory logic for ADX filter
                                    if signal_quality >= self.strong_signal_bypass:
                                        signal_quality *= (
                                            0.85  # −15% quality, still trades
                                        )
                                        reasoning += "+adx_warning"
                                        logger.info(
                                            f"[ADX_TREND] ⚠️ Advisory downgrade "
                                            f"(quality={signal_quality:.2f})"
                                        )
                                    else:
                                        final_signal = 0
                                        reasoning = "insufficient_trend_strength"
                                else:
                                    # Error 7: Profit Economics Monitor (non-blocking log)
                                    try:
                                        if final_signal != 0 and len(df) >= 14:
                                            import numpy as _pm_np

                                            _pm_tr = _pm_np.maximum(
                                                df["high"].values[1:]
                                                - df["low"].values[1:],
                                                _pm_np.abs(
                                                    df["high"].values[1:]
                                                    - df["close"].values[:-1]
                                                ),
                                                _pm_np.abs(
                                                    df["low"].values[1:]
                                                    - df["close"].values[:-1]
                                                ),
                                            )
                                            _pm_atr = float(
                                                _pm_np.nanmean(_pm_tr[-14:])
                                            )
                                            if _pm_atr > 0:
                                                _pm_rr = (2.5 * _pm_atr) / (
                                                    1.5 * _pm_atr
                                                )
                                                if _pm_rr < 1.5:
                                                    logger.warning(
                                                        f"[PROFIT] ⚠️ Low R:R {_pm_rr:.2f} — monitor only"
                                                    )
                                    except Exception:
                                        pass

                # Apply flash veto soft-discount to final quality score
                if _flash_discount < 1.0 and final_signal != 0:
                    signal_quality = round(signal_quality * _flash_discount, 4)
                    reasoning += f" [flash_discount:{_flash_discount:.0%}]"

                # ═══════════════════════════════════════════════════════════
                # C. SESSION LIQUIDITY PENALTY (Extended to all MT5 Assets)
                # ═══════════════════════════════════════════════════════════
                try:
                    if final_signal != 0:
                        from src.utils.market_hours import MarketHours

                        _hour_utc_s = _dt.utcnow().hour

                        # 1. BTC (Binance) is 24/7 - only check for global liquidity lows
                        if "BTC" in self.asset_type:
                            session_quality = MarketHours.get_btc_session_quality()
                            if session_quality == "LOW":
                                signal_quality *= 0.85
                                reasoning += " [session:LOW_LIQ]"
                                logger.info(
                                    f"[SESSION] ⚠️ BTC low liquidity: quality discounted"
                                )

                        # 2. MT5/Exness Assets - Apply Session Penalties
                        else:
                            is_off_session = False
                            asset = self.asset_type.upper()

                            if any(
                                x in asset
                                for x in (
                                    "EUR",
                                    "GBP",
                                    "JPY",
                                    "CHF",
                                    "AUD",
                                    "NZD",
                                    "CAD",
                                )
                            ):
                                if _hour_utc_s < 7 or _hour_utc_s >= 20:
                                    is_off_session = True
                                    logger.info(
                                        f"[SESSION] ⚠️ FX off-session ({_hour_utc_s}:00 UTC)"
                                    )

                            elif "GOLD" in asset or "XAU" in asset:
                                if _hour_utc_s < 7 or _hour_utc_s >= 20:
                                    is_off_session = True
                                    logger.info(
                                        f"[SESSION] ⚠️ GOLD off-session ({_hour_utc_s}:00 UTC)"
                                    )

                            elif any(
                                x in asset
                                for x in ("USTEC", "US100", "NAS", "US30", "SPX")
                            ):
                                if _hour_utc_s < 13 or _hour_utc_s >= 21:
                                    is_off_session = True
                                    logger.info(
                                        f"[SESSION] ⚠️ INDEX off-session ({_hour_utc_s}:00 UTC)"
                                    )

                            elif "OIL" in asset:
                                if _hour_utc_s < 13 or _hour_utc_s >= 19:
                                    is_off_session = True
                                    logger.info(
                                        f"[SESSION] ⚠️ OIL off-session ({_hour_utc_s}:00 UTC)"
                                    )

                            if is_off_session:
                                # In Performance mode, we discount the final quality score
                                signal_quality *= 0.80
                                reasoning += " [session:OFF]"
                                logger.info(
                                    f"[SESSION] Off-session discount applied to {asset}"
                                )

                except Exception as e:
                    logger.warning(f"[SESSION] Gate calculation failed: {e}")

            # ── CONTEXT ENGINE WIRING ─────────────────────────────────────
            # F.3: MR Divergence Cross-Signal (reads from MR strategy if available)
            if state is not None:
                try:
                    _mr_details = {}
                    if hasattr(self.s_mean_reversion, "_last_divergence_info"):
                        _mr_details = self.s_mean_reversion._last_divergence_info or {}
                    if _mr_details.get("divergence_detected"):
                        state.divergence_detected = True
                        state.divergence_strength = float(
                            _mr_details.get("divergence_strength", 0.5)
                        )
                    if state.is_parabolic and state.divergence_detected:
                        state.reversal_imminent = True
                except Exception:
                    pass

            # ─────────────────────────────────────────────────────────────

            # STEP 7: Build base response
            # ✨ NEW: Confluence Reasoning Enhancement
            bonus_tags = []
            if div_res and div_res.type != "NONE":
                # Only add if aligned with signal
                if (final_signal == 1 and "BULLISH" in div_res.type) or (
                    final_signal == -1 and "BEARISH" in div_res.type
                ):
                    tag = div_res.explanation.split(":")[-1].split("(")[0].strip()
                    bonus_tags.append(f"✨ {tag}")

            if br_res and br_res.is_valid:
                if (final_signal == 1 and br_res.type == "BULLISH_RETEST") or (
                    final_signal == -1 and br_res.type == "BEARISH_RETEST"
                ):
                    bonus_tags.append(f"🚀 {br_res.type.replace('_', ' ').title()}")

            if bonus_tags:
                reasoning += " | " + " | ".join(bonus_tags[:2])

            # ✅ FIX: If a filter (sniper, volatility, governor, ATR) zeroed the
            # If a filter zeroed the signal, reset quality to 0.0.
            # Previously signal_quality was set before the filter chain, so a
            # 1.0-quality signal blocked by e.g. low_volatility or governor
            # still reported "Signal Quality: 100%" — contradictory and misleading.
            if final_signal == 0 and original_signal != 0:
                signal_quality = 0.0

            # ── ATR-14 for downstream stop-loss sizing ─────────────────────────
            # mt5_handler and binance_handler look for signal_details["atr_fast"]
            # to compute the ATR-based SL distance.  The council aggregator always
            # includes this key; the performance aggregator was omitting it, causing
            # a fallback to a static percentage SL (wrong stop distance).
            _atr_fast_for_sl = None
            try:
                import talib as _ta_atr

                _atr_result = _ta_atr.ATR(
                    df["high"].values.astype(float),
                    df["low"].values.astype(float),
                    df["close"].values.astype(float),
                    timeperiod=14,
                )
                _last = float(_atr_result[-1])
                if not np.isnan(_last) and _last > 0:
                    _atr_fast_for_sl = _last
            except Exception:
                pass

            # Extract Livermore 1H state for main.py Livermore block
            # (composite_state may or may not be in governor_data depending on path)
            _lsm_1h_for_details = None
            _lsm_4h_for_details = None
            try:
                _cs_for_details = (
                    governor_data.get("composite_state") if governor_data else None
                )
                if _cs_for_details is None:
                    _cs_for_details = getattr(self, "_cached_composite", None)
                if _cs_for_details is not None:
                    _lsm_1h_for_details = getattr(
                        _cs_for_details, "livermore_state_1h", None
                    )
                    _lsm_4h_for_details = getattr(
                        _cs_for_details, "livermore_state_4h", None
                    )
            except Exception:
                pass

            details = {
                "timestamp": timestamp,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "original_signal": original_signal,
                "final_signal": final_signal,
                "reasoning": reasoning,
                "signal_quality": signal_quality,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
                "ema_signal": ema_signal,
                "ema_confidence": ema_conf,
                "atr_fast": _atr_fast_for_sl,
                "governor_data": governor_data,  # Pass governor data through
                "ai_validation": ai_validation_details,
                "trade_type": trade_type,
                "livermore_state_1h": _lsm_1h_for_details,  # for main.py Livermore block
                "livermore_state_4h": _lsm_4h_for_details,
                "viz_overlay": {"divergence": div_res, "break_retest": br_res},
            }

            # STEP 8 (item 18c): Real AI validation, flag-gated OFF by default.
            # Performance-weighted mode previously NEVER called the real
            # validate_signal() — only the cosmetic _format_ai_validation_for_viz
            # formatter ran (council_aggregator.py is the only live caller of the
            # real validator today). phase_config.ai_validator_gates_performance_mode
            # defaults to False, which preserves that exact pre-existing behavior;
            # flip to True only after a backtest/paper-soak watch window, per
            # standing flag-gated rollout policy.
            _ai_gates_perf_mode = getattr(self, "phase_config", {}).get(
                "ai_validator_gates_performance_mode", False
            )
            if self.ai_validator and final_signal != 0 and _ai_gates_perf_mode:
                try:
                    _validated_signal, ai_validation_details = (
                        self.ai_validator.validate_signal(
                            signal=final_signal,
                            signal_details=details,
                            df=df,
                            composite_state=state,
                        )
                    )
                    if _validated_signal != final_signal:
                        logger.warning(
                            f"[AI VALIDATOR] {self.asset_type} signal overruled: "
                            f"{final_signal} -> {_validated_signal} "
                            f"({ai_validation_details.get('action', 'rejected')})"
                        )
                        final_signal = _validated_signal
                        signal = _validated_signal
                        details["ai_modified"] = True
                        details["final_signal"] = final_signal
                        details["reasoning"] = (
                            f"ai_validator_rejected ({ai_validation_details.get('action', 'unknown')})"
                        )
                except Exception as e:
                    logger.error(f"[AGGREGATOR] AI validation failed: {e}")
                    ai_validation_details = {}
            elif self.ai_validator:
                # Flag off (default) or final_signal already 0 — unchanged
                # cosmetic-only formatting, exactly as before item 18c.
                try:
                    # Pass copies to avoid accidental modification
                    ai_validation_details = self._format_ai_validation_for_viz(
                        final_signal=final_signal, details={**details}, df=df
                    )
                except Exception as e:
                    logger.error(f"[AGGREGATOR] AI formatting failed: {e}")

            # STEP 9: Final Response update
            # Derive the ai_validated boolean from the action field so the DB
            # and dashboard always have a correct True/False value.
            # "approved"/"bypassed*" → AI allowed the signal through.
            # "rejected" → AI blocked it.
            # "skipped*"/"none"/"ai_disabled"/"hold" → AI was not in the loop.
            _ai_action = (
                ai_validation_details.get("action", "")
                if isinstance(ai_validation_details, dict)
                else ""
            )
            _ai_validated = _ai_action == "approved" or _ai_action.startswith(
                "bypassed"
            )

            details.update(
                {
                    "ai_validation": ai_validation_details,
                    "ai_validated": _ai_validated,
                    "mr_signal_raw": mr_original,  # Ensure originals are present
                    "tf_signal_raw": tf_original,
                    # Composite state — used by VTM pattern-aware exits and shadow trader
                    "institutional_pattern": (
                        state.institutional_pattern if state else None
                    ),
                    "friday_tighten": state.friday_tighten if state else False,
                    "composite_state": state.to_dict() if state else {},
                }
            )

            return final_signal, details

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {
                "error": str(e),
                "timestamp": timestamp,
                "reasoning": f"error: {str(e)[:50]}",
                "signal_quality": 0.0,
                "final_signal": 0,
                "mr_signal": 0,
                "mr_confidence": 0.0,
                "tf_signal": 0,
                "tf_confidence": 0.0,
                "ema_signal": 0,
                "ema_confidence": 0.0,
            }
