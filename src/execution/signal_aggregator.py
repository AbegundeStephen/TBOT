"""
Production Improvements for BullMarketFilteredAggregator
Addresses issues seen in live logs and adds monitoring utilities
"""

import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from src.strategies.base_strategy import BaseStrategy
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BullMarketFilteredAggregator:
    """
    IMPROVED Version with:
    - Better signal threshold handling
    - Regime change detection (bull->bear transitions)
    - Signal quality scoring
    - Confidence normalization
    - Performance tracking
    """
    
    def __init__(
        self,
        mean_reversion_strategy: BaseStrategy,
        trend_following_strategy: BaseStrategy,
        ema_strategy: BaseStrategy,
        confidence_config: Optional[Dict] = None,
        asset_name: str = "UNKNOWN"
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.asset_name = asset_name
        
        default_config = {
            # Entry Rules
            "min_confidence": 0.45,
            "require_both_buy": True,
            "require_both_sell": True,
            
            # Weights
            "mean_reversion_weight": 1.0,
            "trend_following_weight": 1.0,
            "ema_weight": 1.5,
            
            # Bull Market Filter
            "enable_bull_filter": True,
            "block_sells_in_bull": True,
            "relax_entries_in_bull": False,
            
            # IMPROVEMENTS: Signal Quality & Thresholds
            "signal_quality_threshold": 0.50,  # Min quality to consider signal
            "high_confidence_threshold": 0.65,  # For premium signals
            "regime_confirmation_bars": 2,     # Bars needed to confirm regime change
            "confidence_normalization": True,   # Normalize confidence 0-1
        }
        
        self.config = confidence_config or {}
        for key, value in default_config.items():
            self.config.setdefault(key, value)
        
        # State tracking
        self.consecutive_losses = 0
        self.in_cooldown = False
        self.last_trade_result = None
        self.trade_count = 0
        
        # IMPROVEMENTS: Regime tracking
        self.previous_regime = None
        self.regime_change_count = 0
        self.regime_confirmed = False
        self.signal_history = []  # Track last N signals
        self.max_history = 10
        
        self._log_initialization()
    
    def _log_initialization(self):
        """Log aggregator initialization details"""
        logger.info("=" * 70)
        logger.info(f"BullMarketFilteredAggregator initialized for {self.asset_name}")
        logger.info("=" * 70)
        logger.info(f"  Regime Filter (General): EMA Strategy (50/200 SMA Golden Cross)")
        logger.info(f"  Confirmation Signals (Soldiers):")
        logger.info(f"    - Scout: Mean Reversion (Bollinger Bands)")
        logger.info(f"    - Confirmor: Trend Following (Fast MA Crossover)")
        logger.info("")
        logger.info("SNIPER BOT CONFIGURATION:")
        logger.info(f"  Min Confidence Threshold: {self.config['min_confidence']:.2f}")
        logger.info(f"  Signal Quality Threshold: {self.config['signal_quality_threshold']:.2f}")
        logger.info(f"  High Confidence Threshold: {self.config['high_confidence_threshold']:.2f}")
        logger.info(f"  Strict Entry Rules:")
        logger.info(f"    - Require BOTH for BUY: {self.config['require_both_buy']}")
        logger.info(f"    - Require BOTH for SELL: {self.config['require_both_sell']}")
        logger.info(f"  Bull Market Filter:")
        logger.info(f"    - Enabled: {self.config['enable_bull_filter']}")
        logger.info(f"    - Block Sells in Bull: {self.config['block_sells_in_bull']}")
        logger.info("=" * 70)
    
    def _normalize_confidence(self, confidence: float) -> float:
        """
        Normalize confidence scores to 0-1 range
        Handles cases where model returns values outside 0-1
        """
        if not self.config['confidence_normalization']:
            return confidence
        
        # If already in 0-1 range, return as-is
        if 0 <= confidence <= 1:
            return confidence
        
        # If out of range, clip to 0-1
        logger.warning(
            f"[{self.asset_name}] Confidence out of bounds: {confidence:.3f}, clipping to [0, 1]"
        )
        return np.clip(confidence, 0, 1)
    
    def _evaluate_signal_quality(
        self,
        signal: int,
        confidence: float
    ) -> float:
        """
        Evaluate quality of a signal
        Returns 0-1 score: how reliable is this signal?
        """
        if signal == 0:  # No signal
            return 0.0
        
        # Normalize confidence first
        confidence = self._normalize_confidence(confidence)
        
        # Base quality is the confidence
        quality = confidence
        
        # Bonus if high confidence
        if confidence >= self.config['high_confidence_threshold']:
            quality = min(1.0, quality * 1.1)
        
        return quality
    
    def _detect_regime_change(self, current_regime: bool) -> Tuple[bool, str]:
        """
        Detect if market regime has changed (bull <-> bear)
        Returns: (regime_confirmed, description)
        """
        if self.previous_regime is None:
            self.previous_regime = current_regime
            return False, "regime_initialization"
        
        # Check if regime changed
        if current_regime != self.previous_regime:
            self.regime_change_count += 1
            
            # Require N bars of confirmation before accepting regime change
            if self.regime_change_count >= self.config['regime_confirmation_bars']:
                self.previous_regime = current_regime
                self.regime_confirmed = True
                direction = "BULL" if current_regime else "BEAR"
                
                logger.warning(
                    f"[{self.asset_name}] ⚠️  REGIME CHANGE CONFIRMED → {direction}"
                )
                return True, f"regime_changed_to_{direction}"
            else:
                return False, f"regime_change_pending_{self.regime_change_count}"
        else:
            # Regime stable
            self.regime_change_count = 0
            return True, "regime_stable"
    
    def _track_signal_history(self, signal: int, confidence: float, reasoning: str):
        """Track signal history for analysis and debugging"""
        entry = {
            "timestamp": datetime.now(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        self.signal_history.append(entry)
        
        # Keep only last N signals
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
    
    def get_aggregated_signal(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Main signal aggregation with improved handling
        """
        try:
            # Step 1: Get signals with confidence normalization
            ema_signal, ema_conf = self.s_ema.generate_signal(df)
            ema_conf = self._normalize_confidence(ema_conf)
            is_bull_market = ema_signal == 1
            
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df)
            mr_conf = self._normalize_confidence(mr_conf)
            
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df)
            tf_conf = self._normalize_confidence(tf_conf)
            
            # Step 2: Detect regime changes
            regime_changed, regime_status = self._detect_regime_change(is_bull_market)
            
            # Step 3: Apply Sniper Logic
            final_signal, reasoning, decision_path = self._apply_sniper_logic(
                mr_signal, mr_conf,
                tf_signal, tf_conf,
                is_bull_market, ema_conf
            )
            
            # Step 4: Evaluate signal quality
            signal_quality = self._evaluate_signal_quality(final_signal, 
                max(mr_conf, tf_conf) if final_signal != 0 else 0)
            
            # Step 5: Track history
            self._track_signal_history(final_signal, signal_quality, reasoning)
            
            # Step 6: Package Details
            signal_details = {
                # Market Regime
                "regime": "🚀 BULL" if is_bull_market else "⚖️  NEUTRAL/BEAR",
                "is_bull_market": is_bull_market,
                "regime_changed": regime_changed,
                "regime_status": regime_status,
                
                # Individual Strategy Signals
                "mean_reversion_signal": mr_signal,
                "mean_reversion_confidence": mr_conf,
                "trend_following_signal": tf_signal,
                "trend_following_confidence": tf_conf,
                "ema_signal": ema_signal,
                "ema_confidence": ema_conf,
                
                # Final Decision
                "final_signal": final_signal,
                "reasoning": reasoning,
                "decision_path": decision_path,
                "signal_quality": signal_quality,
                
                # Quality Metrics
                "signal_quality_pass": signal_quality >= self.config['signal_quality_threshold'],
                "high_confidence": signal_quality >= self.config['high_confidence_threshold'],
                
                # Metadata
                "aggregator_mode": "sniper_strict",
                "asset": self.asset_name,
                "timestamp": (
                    df.index[-1].isoformat()
                    if hasattr(df.index[-1], "isoformat")
                    else str(df.index[-1])
                ),
            }
            
            self._log_decision(
                mr_signal, tf_signal, is_bull_market,
                final_signal, reasoning, signal_quality, regime_changed
            )
            
            return final_signal, signal_details
        
        except Exception as e:
            logger.error(f"[{self.asset_name}] Error in get_aggregated_signal: {e}", exc_info=True)
            return 0, {
                "final_signal": 0,
                "reasoning": "error_in_aggregation",
                "error": str(e)
            }
    
    def _apply_sniper_logic(
        self,
        mr_sig: int, mr_conf: float,
        tf_sig: int, tf_conf: float,
        is_bull: bool, ema_conf: float
    ) -> Tuple[int, str, str]:
        """
        MASTER DECISION LOGIC with improved thresholds
        """
        min_conf = self.config['min_confidence']
        
        # IMPROVEMENT: Better confidence filtering
        # Check signal quality, not just min confidence
        mr_quality = self._evaluate_signal_quality(mr_sig, mr_conf)
        tf_quality = self._evaluate_signal_quality(tf_sig, tf_conf)
        
        mr_valid = mr_quality >= self.config['signal_quality_threshold'] and mr_conf >= min_conf
        tf_valid = tf_quality >= self.config['signal_quality_threshold'] and tf_conf >= min_conf
        
        decision_path = []
        
        # === BUY LOGIC ===
        if mr_sig == 1 and tf_sig == 1 and mr_valid and tf_valid:
            decision_path.append(f"✓ BOTH_BUY (MR:{mr_conf:.2f} TF:{tf_conf:.2f})")
            
            if is_bull and self.config['enable_bull_filter']:
                decision_path.append("→ IN_BULL")
                decision_path.append("→ BUY_ALLOWED")
                reasoning = "buy_perfect_setup_bull_market"
                return 1, reasoning, " ".join(decision_path)
            else:
                decision_path.append("→ IN_NEUTRAL_BEAR")
                decision_path.append("→ BUY_ALLOWED")
                reasoning = "buy_perfect_setup_standard"
                return 1, reasoning, " ".join(decision_path)
        
        # === SELL LOGIC (The Core Override) ===
        elif mr_sig == -1 and tf_sig == -1 and mr_valid and tf_valid:
            decision_path.append(f"✓ BOTH_SELL (MR:{mr_conf:.2f} TF:{tf_conf:.2f})")
            
            if is_bull and self.config['enable_bull_filter'] and self.config['block_sells_in_bull']:
                decision_path.append("→ IN_BULL")
                decision_path.append("⚠️  OVERRIDE")
                decision_path.append("→ HOLD")
                reasoning = "sell_blocked_golden_cross_override"
                
                logger.warning(
                    f"[{self.asset_name}] ⚠️  OVERRIDE: Sell blocked (Golden Cross active)"
                )
                return 0, reasoning, " ".join(decision_path)
            else:
                decision_path.append("→ IN_NEUTRAL_BEAR")
                decision_path.append("→ SELL_ALLOWED")
                reasoning = "sell_perfect_setup_standard"
                return -1, reasoning, " ".join(decision_path)
        
        # === PARTIAL SIGNALS ===
        elif (mr_sig == 1 or tf_sig == 1) and not (mr_sig == 1 and tf_sig == 1):
            decision_path.append(f"⚠️  PARTIAL_BUY (MR:{mr_sig} TF:{tf_sig})")
            decision_path.append("→ WAIT_AGREEMENT")
            reasoning = "wait_both_soldiers_must_agree_buy"
            return 0, reasoning, " ".join(decision_path)
        
        elif (mr_sig == -1 or tf_sig == -1) and not (mr_sig == -1 and tf_sig == -1):
            decision_path.append(f"⚠️  PARTIAL_SELL (MR:{mr_sig} TF:{tf_sig})")
            decision_path.append("→ WAIT_AGREEMENT")
            reasoning = "wait_both_soldiers_must_agree_sell"
            return 0, reasoning, " ".join(decision_path)
        
        # === NO SIGNALS ===
        else:
            decision_path.append(f"NO_SIGNAL (MR:{mr_sig} TF:{tf_sig})")
            decision_path.append("→ HOLD_CASH")
            reasoning = "no_action_wait_for_setup"
            return 0, reasoning, " ".join(decision_path)
    
    def _log_decision(
        self,
        mr_sig: int,
        tf_sig: int,
        is_bull: bool,
        final_signal: int,
        reasoning: str,
        signal_quality: float,
        regime_changed: bool
    ):
        """Enhanced logging with signal quality and regime info"""
        mr_name = "BUY" if mr_sig == 1 else "SELL" if mr_sig == -1 else "HOLD"
        tf_name = "BUY" if tf_sig == 1 else "SELL" if tf_sig == -1 else "HOLD"
        regime = "🚀 BULL" if is_bull else "⚖️  NEUTRAL/BEAR"
        final_name = "→ BUY" if final_signal == 1 else "→ SELL" if final_signal == -1 else "→ HOLD"
        regime_mark = "⚡" if regime_changed else " "
        quality_mark = "★" if signal_quality >= 0.65 else "•"
        
        logger.info(
            f"[{self.asset_name}] {regime_mark} {regime} | "
            f"Scout(MR): {mr_name:6} | Confirmor(TF): {tf_name:6} | "
            f"{final_name:7} | Quality: {quality_mark} {signal_quality:.2f} ({reasoning})"
        )
    
    def get_regime_status(self, df: pd.DataFrame) -> Dict:
        """Get current market regime status"""
        try:
            ema_signal, ema_confidence = self.s_ema.generate_signal(df)
            ema_confidence = self._normalize_confidence(ema_confidence)
            is_bull = ema_signal == 1
            
            return {
                "regime": "BULL_MARKET" if is_bull else "NEUTRAL_OR_BEAR",
                "golden_cross_active": is_bull,
                "ema_confidence": ema_confidence,
                "override_active": is_bull and self.config['block_sells_in_bull'],
                "previous_regime": "BULL" if self.previous_regime else "BEAR" if self.previous_regime is False else "UNKNOWN",
                "regime_confirmed": self.regime_confirmed,
                "description": (
                    f"Golden Cross ACTIVE - Selling BLOCKED (Confidence: {ema_confidence:.2f})"
                    if is_bull else
                    f"Standard operation - Both long and short allowed (Confidence: {ema_confidence:.2f})"
                )
            }
        except Exception as e:
            logger.error(f"[{self.asset_name}] Error getting regime status: {e}")
            return {"regime": "UNKNOWN", "error": str(e)}
    
    def get_signal_statistics(self) -> Dict:
        """Get statistics from signal history"""
        if not self.signal_history:
            return {"message": "No signal history yet"}
        
        signals = [s['signal'] for s in self.signal_history]
        confidences = [s['confidence'] for s in self.signal_history]
        
        return {
            "total_signals": len(self.signal_history),
            "buy_signals": signals.count(1),
            "sell_signals": signals.count(-1),
            "hold_signals": signals.count(0),
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "last_signal": self.signal_history[-1] if self.signal_history else None
        }
    
    def load_models(
        self,
        mean_reversion_path: str,
        trend_following_path: str,
        ema_path: str
    ) -> bool:
        """Load all three strategy models"""
        logger.info(f"[{self.asset_name}] Loading strategy models...")
        
        mr_loaded = self.s_mean_reversion.load_model(mean_reversion_path)
        tf_loaded = self.s_trend_following.load_model(trend_following_path)
        ema_loaded = self.s_ema.load_model(ema_path)
        
        if mr_loaded and tf_loaded and ema_loaded:
            logger.info(f"[{self.asset_name}] ✅ All three models loaded")
            return True
        else:
            logger.error(f"[{self.asset_name}] ❌ Failed to load one or more models")
            return False
    
    def get_signal_breakdown(self, df: pd.DataFrame) -> Dict:
        """Get detailed signal breakdown for analysis"""
        try:
            signal, details = self.get_aggregated_signal(df)
            
            return {
                "timestamp": details.get("timestamp"),
                "regime": details.get("regime"),
                "regime_changed": details.get("regime_changed"),
                "strategies": {
                    "mean_reversion": {
                        "signal": details.get("mean_reversion_signal"),
                        "confidence": details.get("mean_reversion_confidence")
                    },
                    "trend_following": {
                        "signal": details.get("trend_following_signal"),
                        "confidence": details.get("trend_following_confidence")
                    },
                    "ema_golden_cross": {
                        "signal": details.get("ema_signal"),
                        "confidence": details.get("ema_confidence")
                    }
                },
                "final_decision": {
                    "signal": signal,
                    "quality": details.get("signal_quality"),
                    "high_confidence": details.get("high_confidence"),
                    "reasoning": details.get("reasoning"),
                    "decision_path": details.get("decision_path")
                }
            }
        except Exception as e:
            logger.error(f"[{self.asset_name}] Error in get_signal_breakdown: {e}")
            return {"error": str(e)}