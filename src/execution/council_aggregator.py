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
        weight_structure: float = 1.5,
        weight_momentum: float = 1.0,
        weight_pattern: float = 0.5,
        weight_volume: float = 0.5,
        
        # Asset-specific tuning
        config: Optional[Dict] = None,
        mtf_integration=None # ✨ INJECTED: The Governor
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_type = asset_type.upper()
        self.ai_validator = ai_validator
        self.detailed_logging = enable_detailed_logging
        self.mtf_integration = mtf_integration 
        
        # Configuration merge
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # ✨ NEW: World-Class Filter Thresholds (Symmetric Logic)
        self.filter_thresholds = {
            "min_sniper_conf": self.config.get("ai", {}).get("min_sniper_confidence", 0.65),
        }

        # Dynamic threshold loading
        self.trend_aligned_threshold = self.config.get(
            'council_trend_aligned', trend_aligned_threshold
        )
        self.counter_trend_threshold = self.config.get(
            'council_counter_trend', counter_trend_threshold
        )
        
        # Weights
        self.w_trend = weight_trend
        self.w_structure = weight_structure
        self.w_momentum = weight_momentum
        self.w_pattern = weight_pattern
        self.w_volume = weight_volume
        
        # Validate weights sum to 5.0
        total_weight = sum([
            self.w_trend, self.w_structure, self.w_momentum,
            self.w_pattern, self.w_volume
        ])
        if abs(total_weight - 5.0) > 0.01:
            logger.warning(f"[COUNCIL] Weights sum to {total_weight:.2f}, not 5.0")
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'trend_aligned_buys': 0,
            'trend_aligned_sells': 0,
            'counter_trend_buys': 0,
            'counter_trend_sells': 0,
            'avg_score_on_trade': [],
            'avg_score_on_hold': [],
        }
        
        # Decision history
        self.decision_history = deque(maxlen=100)
        
        # Regime tracking
        self.previous_regime = None
        self.regime_initialized = False
        
        self._log_initialization()
    
    def _get_default_config(self) -> Dict:
        """Asset-specific configurations"""
        if self.asset_type == "BTC":
            return {
                'rsi_bullish_zone': (40, 65),
                'rsi_bearish_zone': (35, 60),
                'rsi_oversold_bonus': 30,
                'rsi_overbought_bonus': 70,
                'volume_ma_period': 20,
                'pattern_confidence_min': 0.60,
                'macd_confirmation': True,
            }
        else:  # GOLD
            return {
                'rsi_bullish_zone': (35, 60),
                'rsi_bearish_zone': (40, 65),
                'rsi_oversold_bonus': 25,
                'rsi_overbought_bonus': 75,
                'volume_ma_period': 20,
                'pattern_confidence_min': 0.65,
                'macd_confirmation': True,
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
        logger.info(f"   AI Validation: {'ENABLED' if self.ai_validator else 'DISABLED'}")
        logger.info(f"   Governor MTF:  {'ENABLED' if self.mtf_integration else 'DISABLED'}")
        logger.info("=" * 80)
        logger.info("")

    # ========================================================================
    # ✨ WORLD-CLASS FILTERS (Asymmetric Logic)
    # ========================================================================

    def _check_governor_filter(self, df: pd.DataFrame, signal: int, governor_data: Optional[Dict] = None, preset_trade_type: str = "TREND") -> Tuple[bool, str]:
        """
        Check the 1D Macro Trend via pre-injected Governor data.
        ✅ INSTITUTIONAL: Strict TREND enforcement. Supports REVERSION gating.
        """
        # 1. FAIL-SAFE: If no Governor data, return NO TRADE (Strict macro dependency)
        if not governor_data:
            logger.warning("[GOV] ❌ BLOCKED - No MTF Governor data available. Blocking trade (Strict Macro Rule).")
            return False, "NEUTRAL"
            
        governor = governor_data.get('governor') or governor_data.get('full_regime_status')
        
        if not governor:
            logger.warning("[GOV] ❌ BLOCKED - No Governor status object found in data. Blocking trade.")
            return False, "NEUTRAL"

        try:
            # Extract regime context
            regime_name = getattr(governor, 'consensus_regime', governor_data.get('regime', "NEUTRAL"))
            is_bullish = getattr(governor, 'is_bullish', governor_data.get('is_bullish', False))
            is_bearish = getattr(governor, 'is_bearish', governor_data.get('is_bearish', False))

            # 2. Block neutral or mixed markets (Except for REVERSION)
            if regime_name == "NEUTRAL" and preset_trade_type == "TREND":
                logger.info(f"[GOV] ❌ BLOCKED - Market is Neutral/Mixed (Institutional Caution)")
                return False, "NEUTRAL"

            # 3. ASSET-DNA Gating & Trade Alignment
            asset = self.asset_type.upper()

            if preset_trade_type == "REVERSION":
                # --- REVERSION GATING (DNA) ---
                if "BTC" in asset or "USTEC" in asset:
                    # Block MR during BULLISH or SLIGHTLY_BULLISH
                    if regime_name in ["BULLISH", "SLIGHTLY_BULLISH"]:
                        logger.info(f"[GOV] ❌ BLOCKED - MR forbidden in {regime_name} regime for {asset}")
                        return False, "REVERSION"
                    # Allow MR Buys only in BEARISH or NEUTRAL
                    if signal == -1: # MR Short
                        logger.info(f"[GOV] ❌ BLOCKED - MR Shorts forbidden for {asset}")
                        return False, "REVERSION"
                    # Buys in BEARISH or NEUTRAL are allowed
                    return True, "REVERSION"
                
                elif "GOLD" in asset:
                    # Allow MR Buys in BEARISH
                    if signal == 1:
                        if regime_name == "BEARISH":
                            return True, "REVERSION"
                        else:
                            logger.info(f"[GOV] ❌ BLOCKED - MR Buys only allowed in BEARISH for {asset} (Current: {regime_name})")
                            return False, "REVERSION"
                    # Block MR Shorts in BULLISH
                    elif signal == -1:
                        if regime_name == "BULLISH":
                            logger.info(f"[GOV] ❌ BLOCKED - MR Shorts forbidden in BULLISH for {asset}")
                            return False, "REVERSION"
                        else:
                            # Implies allowed in BEARISH or NEUTRAL
                            return True, "REVERSION"

                # EURUSD / EURJPY allow symmetric MR (no extra blocks here)
                return True, "REVERSION"

            else:
                # --- TREND GATING (STRICT) ---
                if is_bullish and signal == -1:
                    logger.info(f"[GOV] ❌ BLOCKED - Short attempt in Macro BULLISH regime ({regime_name})")
                    return False, "TREND"
                if is_bearish and signal == 1:
                    logger.info(f"[GOV] ❌ BLOCKED - Long attempt in Macro BEARISH regime ({regime_name})")
                    return False, "TREND"

                return True, "TREND"

        except Exception as e:
            logger.error(f"[GOV] Error processing Governor data: {e}", exc_info=True)
            return False, "NEUTRAL"

    def _check_volatility_gate_adaptive(self, atr_fast: float, atr_slow: float) -> bool:
        """Blocks trades in dead markets (atr_fast < 0.5 * atr_slow)."""
        try:
            if atr_fast < (0.5 * atr_slow):
                logger.info(f"[VOLATILITY] ❌ BLOCKED - Dead Market (ATR Fast: {atr_fast:.4f} < 0.5 * ATR Slow: {atr_slow:.4f})")
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
            body = abs(latest['close'] - latest['open'])
            high, low, close_vals = df['high'].values, df['low'].values, df['close'].values
            atr_fast = ta.ATR(high, low, close_vals, timeperiod=14)[-1]
            
            displacement_passed = False
            displacement_reason = ""
            
            if 'BTC' in self.asset_type:
                # BINANCE: Real volume is reliable.
                volume_rolling_avg = df['volume'].iloc[-21:-1].mean()
                if volume_rolling_avg > 0 and latest['volume'] >= (volume_rolling_avg * 1.5):
                    displacement_passed = True
                else:
                    displacement_ratio = latest['volume'] / volume_rolling_avg if volume_rolling_avg > 0 else 0
                    displacement_reason = f"BTC Volume Surge < 1.5x ({displacement_ratio:.2f}x)"
            else:
                # EXNESS/FOREX: ATR Displacement is King (Ignore tick volume)
                candle_range = latest['high'] - latest['low']
                if body > (0.5 * atr_fast) or candle_range > (1.0 * atr_fast):
                    displacement_passed = True
                else:
                    displacement_reason = f"Forex Displacement < 0.5*ATR ({body:.4f} < {0.5 * atr_fast:.4f}) AND Range < 1.0*ATR ({candle_range:.4f} < {1.0 * atr_fast:.4f})"
            
            if not displacement_passed:
                logger.info(f"[SNIPER] ❌ BLOCKED - {displacement_reason}")
                return False, {'trigger_type': None, 'reason': displacement_reason}

            # ================================================================
            # 1. AI Pattern Confidence
            # ================================================================
            # Reason: The AI model has already encoded a multi-factor edge.
            if self.ai_validator:
                try:
                    pattern_result = self.ai_validator._check_pattern(
                        df=df,
                        signal=signal,
                        min_confidence=self.filter_thresholds['min_sniper_conf']
                    )
                    if pattern_result.get('pattern_confirmed'):
                        reasons.append({
                            'passed': True,
                            'trigger_type': 'AI_PATTERN',
                            'pattern_name': pattern_result.get('pattern_name'),
                            'confidence': pattern_result.get('confidence'),
                        })
                except Exception as e:
                    logger.debug(f"[SNIPER] AI Pattern check failed: {e}")

            # ================================================================
            # 2. Institutional Displacement Confirmation
            # ================================================================
            # Reason: Signal already passed mandatory displacement fork.
            # We record it here as a confirmed trigger for the audit trail.
            if 'BTC' in self.asset_type:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'VOLUME_SURGE_INSTITUTIONAL',
                    'volume': latest['volume'],
                    'surge_factor': latest['volume'] / volume_rolling_avg,
                })
            else:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'MOMENTUM_DISPLACEMENT_INSTITUTIONAL',
                    'body': body,
                    'atr_multiplier': body / atr_fast if atr_fast > 0 else 0,
                })

            # Check if we have enough data for rolling indicators (Donchian, Bollinger Bands)
            # Need 20 periods + current, so at least 21 bars
            if len(df) < 21:
                if reasons:
                    if self.detailed_logging: logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {[r['trigger_type'] for r in reasons]} (Partial checks due to insufficient data)")
                    return True, reasons[0]
                else:
                    logger.warning(f"[SNIPER] ❌ BLOCKED - Insufficient data for full institutional checks (need 21 bars, have {len(df)}).")
                    return False, {'trigger_type': None, 'reason': f'Insufficient data for full checks (have {len(df)})'}

            # ================================================================
            # 3. Turtle Breakout (20-period Donchian Channel)
            # ================================================================
            # Reason: Detects that price has moved into a new volatility regime.
            close_rolling_mean = df['close'].iloc[-21:-1].mean()
            close_rolling_std = df['close'].iloc[-21:-1].std()
            
            if close_rolling_std > 0:
                upper_band = close_rolling_mean + (2.0 * close_rolling_std)
                lower_band = close_rolling_mean - (2.0 * close_rolling_std)

                if signal == 1 and latest['close'] > upper_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'upper',
                        'price': latest['close'],
                    })
                elif signal == -1 and latest['close'] < lower_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'lower',
                        'price': latest['close'],
                    })
            
            # ================================================================
            # Final Decision
            # ================================================================
            if reasons:
                # Log all triggers that passed
                trigger_types = [r['trigger_type'] for r in reasons]
                logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {trigger_types}")
                # Return the details of the first trigger found
                return True, reasons[0]

            logger.info(f"[SNIPER] ❌ BLOCKED - No institutional edge confirmed.")
            return False, {'trigger_type': None, 'reason': 'No confirmation criteria met'}

        except Exception as e:
            logger.error(f"[SNIPER] Error in institutional edge check: {e}", exc_info=True)
            # Fail-open: If the filter fails, we allow the trade to avoid blocking valid signals due to code errors.
            return True, {'trigger_type': 'ERROR_FALLBACK', 'reason': str(e)}

    def _check_profit_economics_adaptive(self, atr_fast: float) -> bool:
        """The 'Worth It' Check. Validates if potential RR covers fees using ATR scaling."""
        try:
            # Using 1.5 ATR as a proxy for the expected first target distance
            expected_tp_distance = atr_fast * 1.5
            
            # BLOCK trade IF: expected_tp_distance < (0.5 * atr_fast)
            if expected_tp_distance < (0.5 * atr_fast):
                logger.info(f"[PROFIT] ❌ BLOCKED - Low Reward (Expected TP Distance: {expected_tp_distance:.4f} < 0.5 * ATR Fast: {0.5 * atr_fast:.4f})")
                return False
            return True
        except Exception as e:
            logger.error(f"[PROFIT] Error: {e}")
            return True

    
    def get_aggregated_signal(
        self, 
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",  # ✨ NEW: Accepted from main.py
        is_bull_market: bool = True,      # ✨ NEW: Accepted from main.py
        governor_data: Optional[Dict] = None # ✨ NEW: Accepted from main.py
    ) -> Tuple[int, Dict]:
        """
        Main council decision logic with bidirectional support
        ✅ INSTITUTIONAL PHASE 4: Dynamic Weights & Penalty Shift
        """
        self.stats['total_evaluations'] += 1
        timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"
        
        try:
            # ================================================================
            # VOLATILITY & REGIME CONTEXT
            # ================================================================
            high, low, close_vals = df['high'].values, df['low'].values, df['close'].values
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
                regime_conf = governor_data.get('confidence', 0.5)
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
            
            consensus_regime = governor_data.get("consensus_regime", "NEUTRAL") if governor_data else "NEUTRAL"
            
            if consensus_regime in ["SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"]:
                w_momentum = 0.5  # ✨ Allow partial momentum points
                w_structure = 1.5 # ✨ Standard structure weight
                w_pattern = 1.0   # ✨ Increased pattern weight for trigger clarity
                if self.detailed_logging: logger.info(f"[COUNCIL] ⚖️ DYNAMIC WEIGHTS APPLIED: {consensus_regime}")

            if self.s_mean_reversion:
                try:
                    mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] MR signal error: {e}")
            
            if self.s_trend_following:
                try:
                    tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] TF signal error: {e}")
            
            if self.s_ema:
                try:
                    ema_signal, ema_conf = self.s_ema.generate_signal(df)
                except Exception as e:
                    logger.error(f"[COUNCIL] EMA signal error: {e}")
            
            # ================================================================
            # BIDIRECTIONAL SCORING: Evaluate both BUY and SELL
            # ================================================================
            
            # BUY scorecard
            buy_scores = {
                'trend': 0.0,
                'structure': 0.0,
                'momentum': 0.0,
                'pattern': 0.0,
                'volume': 0.0,
            }
            buy_explanations = []
            
            # SELL scorecard
            sell_scores = {
                'trend': 0.0,
                'structure': 0.0,
                'momentum': 0.0,
                'pattern': 0.0,
                'volume': 0.0,
            }
            sell_explanations = []
            
            # ✨ NEW: Detect Breakout State to enable adaptive logic
            is_breakout_mode = self._detect_breakout_state(df)
            
            # Run all judges for both directions
            buy_scores['trend'], sell_scores['trend'], trend_exp = self._judge_trend_bidirectional(df, is_bull, w_trend, consensus_regime)
            buy_explanations.append(trend_exp['buy'])
            sell_explanations.append(trend_exp['sell'])
            
            # Pass breakout flag and ADX to adaptive judges
            buy_scores['structure'], sell_scores['structure'], structure_exp = self._judge_structure_bidirectional(df, is_breakout_mode, w_structure, adx)
            buy_explanations.append(structure_exp['buy'])
            sell_explanations.append(structure_exp['sell'])
            
            buy_scores['momentum'], sell_scores['momentum'], momentum_exp = self._judge_momentum_bidirectional(df, is_bull, is_breakout_mode, w_momentum, adx)
            buy_explanations.append(momentum_exp['buy'])
            sell_explanations.append(momentum_exp['sell'])
            
            buy_scores['pattern'], sell_scores['pattern'], pattern_exp = self._judge_pattern_bidirectional(df, w_pattern)
            buy_explanations.append(pattern_exp['buy'])
            sell_explanations.append(pattern_exp['sell'])
            
            buy_scores['volume'], sell_scores['volume'], volume_exp = self._judge_volume_bidirectional(df, w_volume)
            buy_explanations.append(volume_exp['buy'])
            sell_explanations.append(volume_exp['sell'])
            
            # Calculate total scores
            buy_total = sum(buy_scores.values())
            sell_total = sum(sell_scores.values())
            
            # ================================================================
            # DECISION LOGIC: Choose strongest direction
            # ================================================================
            signal = 0
            total_score = 0.0
            required_score = self.trend_aligned_threshold
            chosen_scores = {}
            decision_type = "HOLD"
            
            # Determine preliminary trade type from preset
            preset_name = self.config.get('name', 'balanced').lower()
            trade_type = "REVERSION" if preset_name == "mr" else "TREND"
            
            # Determine preliminary signal
            if is_bull and buy_total >= self.trend_aligned_threshold:
                signal = 1
                total_score = buy_total
                required_score = self.trend_aligned_threshold
                chosen_scores = buy_scores
            elif not is_bull and buy_total >= self.counter_trend_threshold:
                signal = 1
                total_score = buy_total
                required_score = self.counter_trend_threshold
                chosen_scores = buy_scores
            elif not is_bull and sell_total >= self.trend_aligned_threshold:
                signal = -1
                total_score = sell_total
                required_score = self.trend_aligned_threshold
                chosen_scores = sell_scores
            elif is_bull and sell_total >= self.counter_trend_threshold:
                signal = -1
                total_score = sell_total
                required_score = self.counter_trend_threshold
                chosen_scores = sell_scores
            
            # ====================================================================
            # 🛡️ THE INTERCEPTOR: ABSOLUTE VETO (Phase 4)
            # ====================================================================
            if signal != 0:
                # 1. MACRO GOVERNOR (ABSOLUTE VETO)
                # Reason: Proves macro alignment (1D 200 EMA). Sacrosanct macro rule.
                gov_passed, trade_type = self._check_governor_filter(df, signal, governor_data, trade_type)
                if not gov_passed:
                    logger.info(f"[VETO] ❌ BLOCKED - Macro Regime Conflict.")
                    return 0, {
                        'timestamp': timestamp,
                        'signal': 0,
                        'asset': self.asset_type,
                        'decision_type': "BLOCKED (Macro Regime Conflict)",
                        'reasoning': "blocked_by_macro_governor",
                        'final_signal': 0,
                        'signal_quality': 0.0,
                        'total_score': total_score,
                        'scores': chosen_scores,
                        'buy_total': buy_total,
                        'sell_total': sell_total,
                        'regime': regime_name,
                    }

                # 2. ATR WICK TRAP (ABSOLUTE VETO)
                if not validate_candle_structure(df, self.asset_type, direction="long" if signal == 1 else "short"):
                    logger.info(f"[VETO] ❌ BLOCKED - Institutional Wick Trap.")
                    return 0, {
                        'timestamp': timestamp,
                        'signal': 0,
                        'asset': self.asset_type,
                        'decision_type': "BLOCKED (Institutional Wick Trap)",
                        'reasoning': "blocked_by_trap_filter",
                        'final_signal': 0,
                        'signal_quality': 0.0,
                        'total_score': total_score, # ✨ ADDED
                        'scores': chosen_scores,     # ✨ ADDED
                        'buy_total': buy_total,
                        'sell_total': sell_total,
                        'regime': regime_name,
                    }

                # 3. DEAD VOLATILITY GATE (ABSOLUTE VETO)
                if not self._check_volatility_gate_adaptive(atr_fast, atr_slow):
                    logger.info(f"[VETO] ❌ BLOCKED - Dead Market Volatility.")
                    return 0, {
                        'timestamp': timestamp,
                        'signal': 0,
                        'asset': self.asset_type,
                        'decision_type': "BLOCKED (Dead Market Volatility)",
                        'reasoning': "low_volatility_veto",
                        'final_signal': 0,
                        'signal_quality': 0.0,
                        'total_score': total_score, # ✨ ADDED
                        'scores': chosen_scores,     # ✨ ADDED
                        'buy_total': buy_total,
                        'sell_total': sell_total,
                        'regime': regime_name,
                    }

            # ====================================================================
            # 📉 MINOR FAILURES: SCORING PENALTIES (Phase 4)
            # ====================================================================
            if signal != 0:
                penalty = 0.0
                
                # A. SNIPER LOCK
                sniper_passed, sniper_details = self._check_sniper_filter(df, signal)
                if not sniper_passed:
                    penalty += 1.0
                    logger.info(f"[PENALTY] ⚠️ Sniper confirmation failure: -1.0")

                # B. PROFIT ECONOMICS
                if not self._check_profit_economics_adaptive(atr_fast):
                    penalty += 1.0
                    logger.info(f"[PENALTY] ⚠️ Low profit economics: -1.0")

                # Apply penalties
                total_score -= penalty
                
                # Final execution check
                if total_score < required_score:
                    logger.info(f"[SIGNAL] ❌ REJECTED - Score after penalties ({total_score:.2f}) < {required_score:.2f}")
                    signal = 0
                    decision_type = f"REJECTED (Score: {total_score:.2f})"
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
                chosen_scores = {'buy': buy_scores, 'sell': sell_scores}
                chosen_explanations = buy_explanations + sell_explanations
                if total_score == 0: total_score = max(buy_total, sell_total)
            
            # Update statistics based on FINAL signal
            if signal == 1:
                self.stats['buy_signals'] += 1
                if is_bull: self.stats['trend_aligned_buys'] += 1
                else: self.stats['counter_trend_buys'] += 1
                self.stats['avg_score_on_trade'].append(total_score)
            elif signal == -1:
                self.stats['sell_signals'] += 1
                if not is_bull: self.stats['trend_aligned_sells'] += 1
                else: self.stats['counter_trend_sells'] += 1
                self.stats['avg_score_on_trade'].append(total_score)
            else:
                self.stats['hold_signals'] += 1
                self.stats['avg_score_on_hold'].append(total_score)
            
            # Calculate signal quality
            base_quality = min(total_score / 5.0, 1.0)
            if signal != 0:
                judge_agreement = sum(1 for s in chosen_scores.values() if s > 0) / len(chosen_scores)
            else:
                judge_agreement = 0.5
            signal_quality = base_quality * (0.8 + 0.2 * judge_agreement)
            signal_quality = min(signal_quality, 1.0)
            
            # Build details dict
            details = {
                'timestamp': timestamp,
                'signal': signal,
                'asset': self.asset_type,
                'trade_type': trade_type, 
                'decision_type': decision_type,
                'total_score': total_score,
                'required_score': required_score,
                'scores': chosen_scores,
                'buy_scores': buy_scores,
                'sell_scores': sell_scores,
                'buy_total': buy_total,
                'sell_total': sell_total,
                'regime': regime_name,
                'regime_confidence': regime_conf,
                'explanations': chosen_explanations,
                'signal_quality': signal_quality,
                'reasoning': f"{decision_type} (Score: {total_score:.2f}/{required_score:.1f})",
                'mr_signal': mr_signal,
                'mr_confidence': mr_conf,
                'tf_signal': tf_signal,
                'tf_confidence': tf_conf,
                'ema_signal': ema_signal,
                'ema_confidence': ema_conf,
                'buy_score': buy_total,
                'sell_score': sell_total,
                'aggregator_type': 'council',
                'judge_agreement': judge_agreement,
                'atr_fast': atr_fast,
                'atr_slow': atr_slow,
                'governor_data': governor_data
            }
            
            # Log decision
            if self.detailed_logging or signal != 0:
                self._log_decision_bidirectional(details)
            
            # Store history
            self.decision_history.append({
                'timestamp': timestamp,
                'signal': signal,
                'score': total_score,
                'regime': regime_name,
            })
            
            # AI validation
            if self.ai_validator and signal != 0:
                original_signal = signal
                
                validated_signal, ai_details = self.ai_validator.validate_signal(
                    signal=signal,
                    signal_details=details,
                    df=df,
                )
                
                if validated_signal != signal:
                    logger.warning(f"[AI] Overruled: {signal} → {validated_signal}")
                    signal = validated_signal
                    details['ai_modified'] = True
                    details['signal'] = signal
                
                try:
                    formatted_ai = self._format_ai_validation_for_viz(
                        final_signal=signal,
                        details=details.copy(),
                        df=df
                    )
                    details['ai_validation'] = formatted_ai
                except Exception as e:
                    logger.error(f"[COUNCIL] AI formatting failed: {e}")
                    details['ai_validation'] = {
                        "pattern_detected": False,
                        "pattern_name": "Error",
                        "pattern_confidence": 0.0,
                        "validation_passed": signal != 0,
                        "action": "error_formatting",
                        "error": str(e),
                    }
            
            elif self.ai_validator and signal == 0:
                try:
                    details['ai_validation'] = self._format_ai_validation_for_viz(
                        final_signal=signal,
                        details=details.copy(),
                        df=df
                    )
                except Exception as e:
                    logger.error(f"[COUNCIL] AI formatting for hold signal failed: {e}")
                    details['ai_validation'] = {
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
                'error': str(e),
                'timestamp': timestamp,
                'signal': 0,
                'total_score': 0.0,
                'signal_quality': 0.0,
                'mr_signal': 0,
                'mr_confidence': 0.0,
                'tf_signal': 0,
                'tf_confidence': 0.0,
                'ema_signal': 0,
                'ema_confidence': 0.0,
                'reasoning': f"error: {str(e)[:50]}",
            }
    
    # ========================================================================
    # BIDIRECTIONAL JUDGES
    # ========================================================================
    
    def _judge_trend_bidirectional(self, df: pd.DataFrame, is_bull: bool, weight: float, consensus_regime: str = "NEUTRAL") -> Tuple[float, float, Dict]:
        """
        JUDGE 1: TREND (Bidirectional)
        """
        try:
            features = self.s_trend_following.generate_features(df.tail(250))
            if features.empty:
                return 0.0, 0.0, {'buy': "TREND: No data", 'sell': "TREND: No data"}
            
            latest = features.iloc[-1]
            price = latest['close']
            ema_20 = latest.get('ema_fast', 0)
            ema_50 = latest.get('ema_slow', 0)
            ema_200 = latest.get('ema_200', 0)
            
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
            elif consensus_regime == "SLIGHTLY_BULLISH" and price < ema_50 and price > ema_200:
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
            elif consensus_regime == "SLIGHTLY_BEARISH" and price > ema_50 and price < ema_200:
                sell_score = weight * 0.5
                sell_exp = f"TREND SELL: 🌊 Pullback ({sell_score:.1f}) - Slight Bearish regime, Price < EMA200"
            else:
                sell_exp = "TREND SELL: ❌ No credit - Price > EMA50"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[TREND] Error: {e}")
            return 0.0, 0.0, {'buy': f"TREND: Error", 'sell': f"TREND: Error"}
    
    def _judge_structure_bidirectional(self, df: pd.DataFrame, is_breakout_mode: bool, weight: float, adx: float = 20.0) -> Tuple[float, float, Dict]:
        """
        JUDGE 2: STRUCTURE (Bidirectional & Adaptive)
        """
        try:
            current_price = float(df['close'].iloc[-1])
            buy_score, sell_score = 0.0, 0.0
            buy_exp, sell_exp = "STRUCT BUY: ❌ No signal", "STRUCT SELL: ❌ No signal"

            if is_breakout_mode:
                # --- BREAKOUT LOGIC: Has structure been broken? ---
                if len(df) < 21:
                    return 0.0, 0.0, {'buy': "STRUCT: Need 21 bars for breakout", 'sell': "STRUCT: Need 21 bars for breakout"}
                
                high_20 = df['high'].iloc[-21:-1].max()
                low_20 = df['low'].iloc[-21:-1].min()

                if current_price > high_20:
                    buy_score = weight
                    buy_exp = f"STRUCT BUY: ✅ Breakout ({weight:.1f}) - Price > 20-bar high ${high_20:.2f}"
                
                if current_price < low_20:
                    sell_score = weight
                    sell_exp = f"STRUCT SELL: ✅ Breakdown ({weight:.1f}) - Price < 20-bar low ${low_20:.2f}"

            else:
                # --- NORMAL LOGIC: Is price reacting to an S/R level? ---
                if not self.ai_validator:
                    return 0.0, 0.0, {'buy': "STRUCT: AI disabled", 'sell': "STRUCT: AI disabled"}

                # ✨ ADAPTIVE: ATR-based proximity scaling
                high, low, close = df['high'].values, df['low'].values, df['close'].values
                atr_fast = ta.ATR(high, low, close, timeperiod=14)[-1]
                
                # If trending (ADX > 25), be more lenient with distance
                multiplier = 2.5 if adx > 25 else 1.5
                threshold_val = (multiplier * atr_fast)
                threshold_pct = threshold_val / current_price
                
                # Check for reaction at a SUPPORT level (for BUY)
                sr_buy = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type, df=df, current_price=current_price, signal=1, threshold=threshold_pct
                )
                if sr_buy.get('near_level'):
                    level = sr_buy.get('nearest_level', 0)
                    buy_score = weight
                    buy_exp = f"STRUCT BUY: ✅ At Support ({weight:.1f}) - Near level ${level:.2f} (±{multiplier}*ATR)"
                else:
                    buy_exp = "STRUCT BUY: ❌ No support nearby"
                
                # Check for reaction at a RESISTANCE level (for SELL)
                sr_sell = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type, df=df, current_price=current_price, signal=-1, threshold=threshold_pct
                )
                if sr_sell.get('near_level'):
                    level = sr_sell.get('nearest_level', 0)
                    sell_score = weight
                    sell_exp = f"STRUCT SELL: ✅ At Resistance ({weight:.1f}) - Near level ${level:.2f} (±{multiplier}*ATR)"
                else:
                    sell_exp = "STRUCT SELL: ❌ No resistance nearby"

            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}

        except Exception as e:
            logger.error(f"[STRUCTURE] Error: {e}", exc_info=True)
            return 0.0, 0.0, {'buy': "STRUCT: Error", 'sell': "STRUCT: Error"}
    
    def _judge_momentum_bidirectional(self, df: pd.DataFrame, is_bull: bool, is_breakout_mode: bool, weight: float, adx: float) -> Tuple[float, float, Dict]:
        """
        JUDGE 3: MOMENTUM (Bidirectional & Adaptive)
        """
        try:
            if weight == 0:
                return 0.0, 0.0, {'buy': "MOM: Disabled", 'sell': "MOM: Disabled"}

            # ✅ SUPER-CYCLE OVERRIDE (ADX > 35)
            # In a super-trend, we award full weight to the trend-aligned direction
            # and bypass oscillator (RSI) noise entirely.
            if adx > 35:
                buy_score = weight if is_bull else 0.0
                sell_score = weight if not is_bull else 0.0
                buy_exp = f"MOM BUY: ✅ Super-Cycle ({buy_score:.1f}) - ADX {adx:.1f} > 35" if is_bull else "MOM BUY: ❌ Dead in Bear Super-Cycle"
                sell_exp = f"MOM SELL: ✅ Super-Cycle ({sell_score:.1f}) - ADX {adx:.1f} > 35" if not is_bull else "MOM SELL: ❌ Dead in Bull Super-Cycle"
                return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}

            features_mr = self.s_mean_reversion.generate_features(df.tail(100))
            if features_mr.empty:
                return 0.0, 0.0, {'buy': "MOM: No data", 'sell': "MOM: No data"}
            
            rsi = features_mr.iloc[-1].get('rsi', 50)
            
            # Config values
            bullish_min, bullish_max = self.config['rsi_bullish_zone']
            bearish_min, bearish_max = self.config['rsi_bearish_zone']
            oversold = self.config['rsi_oversold_bonus']
            overbought = self.config['rsi_overbought_bonus']
            
            buy_score = 0.0
            sell_score = 0.0
            buy_exp = f"MOM BUY: ❌ No credit - RSI {rsi:.1f}"
            sell_exp = f"MOM SELL: ❌ No credit - RSI {rsi:.1f}"

            if is_breakout_mode:
                # --- BREAKOUT LOGIC (Momentum Continuation) ---
                if rsi < oversold:
                    sell_score = weight
                    sell_exp = f"MOM SELL: ✅ Breakout ({weight:.1f}) - RSI {rsi:.1f} shows downside momentum"
                if rsi > overbought:
                    buy_score = weight
                    buy_exp = f"MOM BUY: ✅ Breakout ({weight:.1f}) - RSI {rsi:.1f} shows upside momentum"

            else:
                # --- NORMAL LOGIC (Mean Reversion) ---
                # Check for Super-Cycle Override (ADX > 35)
                # If ADX is high, we don't penalize mean-reversion entries for being "overextended"
                # RSI points are granted normally.
                if bullish_min <= rsi <= bullish_max:
                    buy_score = weight
                    buy_exp = f"MOM BUY: ✅ Full ({weight:.1f}) - RSI {rsi:.1f} in bullish zone"
                elif rsi < oversold:
                    buy_score = weight
                    buy_exp = f"MOM BUY: ✅ Oversold ({weight:.1f}) - RSI {rsi:.1f}"

                if bearish_min <= rsi <= bearish_max:
                    sell_score = weight
                    sell_exp = f"MOM SELL: ✅ Full ({weight:.1f}) - RSI {rsi:.1f} in bearish zone"
                elif rsi > overbought:
                    sell_score = weight
                    sell_exp = f"MOM SELL: ✅ Overbought ({weight:.1f}) - RSI {rsi:.1f}"
            
            # MACD confirmation
            if self.config['macd_confirmation']:
                macd = features_mr.iloc[-1].get('macd', 0)
                macd_signal = features_mr.iloc[-1].get('macd_signal', 0)
                
                if buy_score > 0 and macd > macd_signal:
                    buy_score = min(buy_score + 0.2, weight)
                    buy_exp += " +MACD"
                
                if sell_score > 0 and macd < macd_signal:
                    sell_score = min(sell_score + 0.2, weight)
                    sell_exp += " +MACD"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[MOMENTUM] Error: {e}", exc_info=True)
            return 0.0, 0.0, {'buy': "MOM: Error", 'sell': "MOM: Error"}
    
    def _judge_pattern_bidirectional(self, df: pd.DataFrame, weight: float) -> Tuple[float, float, Dict]:
        """
        JUDGE 4: PATTERN (Bidirectional)
        """
        try:
            if not self.ai_validator:
                return 0.0, 0.0, {'buy': "PATTERN: AI disabled", 'sell': "PATTERN: AI disabled"}
            
            buy_score = 0.0
            sell_score = 0.0
            
            # Check bullish pattern
            pattern_buy = self.ai_validator._check_pattern(
                df=df,
                signal=1,
                min_confidence=self.config['pattern_confidence_min'],
            )
            
            if pattern_buy.get('pattern_confirmed'):
                conf = pattern_buy.get('confidence', 0)
                name = pattern_buy.get('pattern_name', 'Unknown')
                
                if conf > 0.75:
                    buy_score = weight
                    buy_exp = f"PATTERN BUY: ✅ Full ({weight:.1f}) - {name} ({conf:.0%})"
                else:
                    buy_score = weight * 0.8
                    buy_exp = f"PATTERN BUY: ⚠️ Partial ({buy_score:.1f}) - {name} ({conf:.0%})"
            else:
                buy_exp = "PATTERN BUY: ❌ No pattern"
            
            # Check bearish pattern
            pattern_sell = self.ai_validator._check_pattern(
                df=df,
                signal=-1,
                min_confidence=self.config['pattern_confidence_min'],
            )
            
            if pattern_sell.get('pattern_confirmed'):
                conf = pattern_sell.get('confidence', 0)
                name = pattern_sell.get('pattern_name', 'Unknown')
                
                if conf > 0.75:
                    sell_score = weight
                    sell_exp = f"PATTERN SELL: ✅ Full ({weight:.1f}) - {name} ({conf:.0%})"
                else:
                    sell_score = weight * 0.8
                    sell_exp = f"PATTERN SELL: ⚠️ Partial ({sell_score:.1f}) - {name} ({conf:.0%})"
            else:
                sell_exp = "PATTERN SELL: ❌ No pattern"
            
            return buy_score, sell_score, {'buy': buy_exp, 'sell': sell_exp}
            
        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return 0.0, 0.0, {'buy': "PATTERN: Error", 'sell': "PATTERN: Error"}
    
    def _judge_volume_bidirectional(self, df: pd.DataFrame, weight: float) -> Tuple[float, float, Dict]:
        """
        JUDGE 5: VOLUME (Same for both directions)
        """
        try:
            if 'volume' not in df.columns:
                return 0.0, 0.0, {'buy': "VOL: No data", 'sell': "VOL: No data"}
            
            volume_ma_period = self.config['volume_ma_period']
            current_volume = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(volume_ma_period).mean().iloc[-1]
            
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
            
            return score, score, {'buy': exp, 'sell': exp}
            
        except Exception as e:
            logger.error(f"[VOLUME] Error: {e}")
            return 0.0, 0.0, {'buy': "VOL: Error", 'sell': "VOL: Error"}
    
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

    def _detect_breakout_state(self, df: pd.DataFrame, adx_threshold: int = 25, volume_surge_factor: float = 1.5, donchian_period: int = 20) -> bool:
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
                return False # Not enough data to determine breakout state

            # 1. Strength Check: ADX > 25
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            adx = ta.ADX(highs, lows, closes, timeperiod=14) # Standard ADX period is 14
            latest_adx = adx[-1]
            is_strong_trend = latest_adx > adx_threshold
            
            if not is_strong_trend:
                if self.detailed_logging: logger.info(f"[BREAKOUT] Condition not met: ADX {latest_adx:.1f} <= {adx_threshold}")
                return False

            # 2. Participation Check: Volume >= 1.5x average
            volume_ma = df['volume'].rolling(donchian_period).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            is_volume_surge = current_volume >= (volume_ma * volume_surge_factor)

            if not is_volume_surge:
                if self.detailed_logging: logger.info(f"[BREAKOUT] Condition not met: Volume {current_volume:.0f} < {volume_ma * volume_surge_factor:.0f}")
                return False

            # 3. Structure Check: Price breaks Donchian High/Low
            # ✅ PHASE 4: STOP-HUNT CEILING LOGIC
            highs_20 = df['high'].iloc[-donchian_period-1:-1]
            closes_20 = df['close'].iloc[-donchian_period-1:-1]
            lows_20 = df['low'].iloc[-donchian_period-1:-1]
            
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
                if self.detailed_logging: logger.info(f"[BREAKOUT] Condition not met: Structure holding.")
                return False

            # If all conditions are met, we are in a breakout state.
            logger.info(f"🔥 BREAKOUT STATE DETECTED: ADX={latest_adx:.1f}, Vol Ratio={current_volume/volume_ma:.1f}x, Price broke structure.")
            return True

        except Exception as e:
            logger.error(f"[BREAKOUT] Error detecting breakout state: {e}", exc_info=True)
            return False # Fail-safe to False
    
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
        for judge, score in details['buy_scores'].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")
        
        logger.info(f"")
        logger.info(f"SELL SCORECARD (Total: {details['sell_total']:.2f}/5.0):")
        for judge, score in details['sell_scores'].items():
            max_score = getattr(self, f"w_{judge}")
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            logger.info(f"  {judge.upper():12s} [{bar}] {score:.2f}/{max_score:.1f}")
        
        logger.info(f"")
        logger.info(f"DECISION: {details['decision_type']}")
        logger.info(f"SIGNAL:   {details['signal']:+2d}")
        logger.info(f"SCORE:    {details['total_score']:.2f} / {details['required_score']:.2f}")
        logger.info("=" * 80)
        logger.info("")
    
    def _format_ai_validation_for_viz(self, final_signal: int, details: dict, df: pd.DataFrame) -> dict:
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

            # Pattern Detection
            try:
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=final_signal,
                    min_confidence=self.ai_validator.current_pattern_threshold,
                )

                viz_data["pattern_detected"] = pattern_result.get("pattern_confirmed", False)
                viz_data["pattern_name"] = pattern_result.get("pattern_name", "None")
                viz_data["pattern_id"] = pattern_result.get("pattern_id")
                viz_data["pattern_confidence"] = pattern_result.get("confidence", 0.0)

                if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                    try:
                        snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                        first_open = snippet[0, 0]

                        if first_open > 0:
                            snippet_norm = snippet / first_open - 1
                            snippet_input = snippet_norm.reshape(1, 15, 4)
                            predictions = self.ai_validator.sniper.model.predict(snippet_input, verbose=0)[0]

                            top3_indices = predictions.argsort()[-3:][::-1]
                            top3_confidences = predictions[top3_indices]

                            top3_patterns = []
                            for idx in top3_indices:
                                pattern_name = self.ai_validator.reverse_pattern_map.get(idx, f"Pattern_{idx}")
                                top3_patterns.append(pattern_name)

                            viz_data["top3_patterns"] = top3_patterns
                            viz_data["top3_confidences"] = top3_confidences.tolist()
                    except Exception as e:
                        logger.debug(f"[VIZ] Top3 patterns failed: {e}")
            except Exception as e:
                logger.error(f"[VIZ] Pattern detection failed: {e}")

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
                if viz_data["pattern_confidence"] < self.ai_validator.current_pattern_threshold:
                    reasons.append(f"Low confidence ({viz_data['pattern_confidence']:.1%})")
                
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
        total = max(self.stats['total_evaluations'], 1)
        
        return {
            **self.stats,
            'buy_rate': (self.stats['buy_signals'] / total) * 100,
            'sell_rate': (self.stats['sell_signals'] / total) * 100,
            'hold_rate': (self.stats['hold_signals'] / total) * 100,
            'avg_score_on_trade': (
                np.mean(self.stats['avg_score_on_trade']) 
                if self.stats['avg_score_on_trade'] else 0.0
            ),
            'avg_score_on_hold': (
                np.mean(self.stats['avg_score_on_hold']) 
                if self.stats['avg_score_on_hold'] else 0.0
            ),
        }