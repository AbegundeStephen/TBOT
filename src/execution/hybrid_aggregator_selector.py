"""
Hybrid Aggregator Selector - Dynamic Mode Switching
====================================================
Intelligently switches between Performance-Weighted and Council aggregators
based on real-time market conditions, with adaptive TP/SL management.

DECISION FRAMEWORK:
-------------------
- COUNCIL MODE: Institutional approach for high-confidence, clear trends
  → Use when: Strong directional bias, low noise, institutional opportunities
  
- PERFORMANCE MODE: Statistical approach for complex/ranging markets  
  → Use when: Mixed signals, high volatility, mean-reversion conditions

"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Analyzes market microstructure to determine optimal aggregator mode
    """
    
    def __init__(self):
        self.analysis_history = deque(maxlen=100)
        
    def analyze_market_state(self, df: pd.DataFrame, asset_name: str) -> Dict:
        """
        Comprehensive market analysis for aggregator selection
        
        Returns dict with:
        - recommended_mode: 'council' or 'performance'
        - confidence: 0-1 (how confident in the recommendation)
        - regime_type: 'trending', 'ranging', 'volatile', 'reversal'
        - metrics: all calculated indicators
        """
        try:
            if len(df) < 200:
                return self._default_analysis()
            
            metrics = {}
            
            # ============================================================
            # 1. TREND STRENGTH (ADX + Directional Movement)
            # ============================================================
            adx, plus_di, minus_di = self._calculate_adx_full(df)
            metrics['adx'] = adx
            metrics['plus_di'] = plus_di
            metrics['minus_di'] = minus_di
            
            # Strong trend = ADX > 25 with clear direction
            trend_strength = 'strong' if adx > 25 else 'weak'
            trend_direction = 'bull' if plus_di > minus_di else 'bear'
            
            # ============================================================
            # 2. VOLATILITY REGIME (ATR Analysis)
            # ============================================================
            atr_current = self._calculate_atr(df, period=14)
            atr_slow = self._calculate_atr(df, period=50)
            volatility_ratio = atr_current / atr_slow if atr_slow > 0 else 1.0
            
            metrics['atr_current'] = atr_current
            metrics['atr_ratio'] = volatility_ratio
            
            # High volatility = ratio > 1.3
            volatility_regime = (
                'high' if volatility_ratio > 1.3 else
                'low' if volatility_ratio < 0.8 else
                'normal'
            )
            
            # ============================================================
            # 3. PRICE ACTION CLARITY (Candle Analysis)
            # ============================================================
            # Count indecision candles (doji, spinning tops)
            recent_candles = df.tail(20)
            body_sizes = abs(recent_candles['close'] - recent_candles['open'])
            total_ranges = recent_candles['high'] - recent_candles['low']
            
            body_ratios = body_sizes / total_ranges
            indecision_count = (body_ratios < 0.3).sum()
            
            metrics['indecision_pct'] = (indecision_count / 20) * 100
            
            # Clear price action = < 30% indecision candles
            price_clarity = (
                'clear' if indecision_count < 6 else
                'noisy' if indecision_count > 12 else
                'mixed'
            )
            
            # ============================================================
            # 4. MOMENTUM ALIGNMENT (RSI + MACD)
            # ============================================================
            rsi = self._calculate_rsi(df, period=14)
            macd, macd_signal, macd_hist = self._calculate_macd(df)
            
            metrics['rsi'] = rsi
            metrics['macd_hist'] = macd_hist
            
            # Check if momentum indicators align
            momentum_aligned = (
                (rsi > 50 and macd_hist > 0) or  # Both bullish
                (rsi < 50 and macd_hist < 0)     # Both bearish
            )
            
            # ============================================================
            # 5. SUPPORT/RESISTANCE PROXIMITY
            # ============================================================
            current_price = df['close'].iloc[-1]
            sr_levels = self._find_key_levels(df)
            
            nearest_level, distance_pct = self._find_nearest_level(
                current_price, sr_levels
            )
            
            metrics['nearest_sr_level'] = nearest_level
            metrics['distance_to_sr_pct'] = distance_pct
            
            # At key level if within 1.5%
            at_key_level = distance_pct < 1.5 if distance_pct is not None else False
            
            # ============================================================
            # 6. MARKET REGIME CLASSIFICATION
            # ============================================================
            
            if trend_strength == 'strong' and price_clarity == 'clear':
                if volatility_regime == 'normal':
                    regime_type = 'trending_clean'
                else:
                    regime_type = 'trending_volatile'
                    
            elif trend_strength == 'weak' and volatility_regime == 'low':
                regime_type = 'ranging_quiet'
                
            elif volatility_regime == 'high':
                regime_type = 'volatile_choppy'
                
            elif at_key_level and momentum_aligned:
                regime_type = 'reversal_setup'
                
            else:
                regime_type = 'mixed_signals'
            
            metrics['regime_type'] = regime_type
            
            # ============================================================
            # 7. AGGREGATOR RECOMMENDATION
            # ============================================================
            
            recommendation = self._select_aggregator_mode(
                regime_type=regime_type,
                trend_strength=trend_strength,
                adx=adx,
                volatility_ratio=volatility_ratio,
                price_clarity=price_clarity,
                momentum_aligned=momentum_aligned,
                at_key_level=at_key_level,
            )
            
            # Store analysis
            analysis = {
                'timestamp': datetime.now(),
                'asset': asset_name,
                'recommended_mode': recommendation['mode'],
                'confidence': recommendation['confidence'],
                'regime_type': regime_type,
                'trend': {
                    'strength': trend_strength,
                    'direction': trend_direction,
                    'adx': adx,
                },
                'volatility': {
                    'regime': volatility_regime,
                    'ratio': volatility_ratio,
                },
                'price_action': {
                    'clarity': price_clarity,
                    'indecision_pct': metrics['indecision_pct'],
                },
                'momentum_aligned': momentum_aligned,
                'at_key_level': at_key_level,
                'metrics': metrics,
                'reasoning': recommendation['reasoning'],
            }
            
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"[HYBRID] Analysis error: {e}", exc_info=True)
            return self._default_analysis()
    
    def _select_aggregator_mode(
        self,
        regime_type: str,
        trend_strength: str,
        adx: float,
        volatility_ratio: float,
        price_clarity: str,
        momentum_aligned: bool,
        at_key_level: bool,
    ) -> Dict:
        """
        Core decision logic for aggregator selection
        """
        
        # ============================================================
        # COUNCIL MODE CRITERIA (Institutional Approach)
        # ============================================================
        # Best for: Clean trends, clear setups, institutional-grade signals
        
        council_score = 0
        council_reasons = []
        
        # Strong directional trend (Council's TREND judge thrives here)
        if trend_strength == 'strong' and adx > 30:
            council_score += 3
            council_reasons.append(f"Strong trend (ADX={adx:.1f})")
        
        # Clear price action (Council's STRUCTURE judge needs this)
        if price_clarity == 'clear':
            council_score += 2
            council_reasons.append("Clear price action")
        
        # At key S/R level (Council's STRUCTURE judge validates)
        if at_key_level:
            council_score += 2
            council_reasons.append("Near key S/R level")
        
        # Normal volatility (Council prefers stability)
        if 0.9 <= volatility_ratio <= 1.2:
            council_score += 1
            council_reasons.append("Stable volatility")
        
        # Momentum alignment (All judges agree)
        if momentum_aligned:
            council_score += 1
            council_reasons.append("Momentum aligned")
        
        # ============================================================
        # PERFORMANCE MODE CRITERIA (Statistical Approach)
        # ============================================================
        # Best for: Complex markets, mixed signals, mean-reversion setups
        
        performance_score = 0
        performance_reasons = []
        
        # Weak trend / ranging (Performance's mean-reversion shines)
        if trend_strength == 'weak':
            performance_score += 3
            performance_reasons.append("Weak/ranging market")
        
        # High volatility (Performance handles uncertainty better)
        if volatility_ratio > 1.15:  
            performance_score += 2
            performance_reasons.append(f"High volatility ({volatility_ratio:.2f}x)")
        
        # Noisy price action (Performance uses statistical filters)
        if price_clarity == 'noisy':
            performance_score += 2
            performance_reasons.append("Noisy price action")
        
        # Mixed signals (Performance weights strategies dynamically)
        if not momentum_aligned:
            performance_score += 1
            performance_reasons.append("Mixed momentum signals")
        
        # Mean-reversion setup (Performance's strength)
        if regime_type in ['ranging_quiet', 'reversal_setup']:
            performance_score += 2
            performance_reasons.append(f"Mean-reversion setup ({regime_type})")
        
        # ============================================================
        # DECISION
        # ============================================================
        
        max_score = max(council_score, performance_score)
        
        if council_score > performance_score:
            mode = 'council'
            confidence = min(council_score / 9.0, 1.0)  # Max 9 points
            reasoning = " | ".join(council_reasons)
            
        elif performance_score > council_score:
            mode = 'performance'
            confidence = min(performance_score / 10.0, 1.0)  # Max 10 points
            reasoning = " | ".join(performance_reasons)
            
        else:
            # Tie-breaker: Use regime type
            if regime_type in ['trending_clean', 'trending_volatile']:
                mode = 'council'
                confidence = 0.6
                reasoning = "Tie-breaker: Trending market favors Council"
            else:
                mode = 'performance'
                confidence = 0.6
                reasoning = "Tie-breaker: Complex market favors Performance"
        
        return {
            'mode': mode,
            'confidence': confidence,
            'reasoning': reasoning,
            'council_score': council_score,
            'performance_score': performance_score,
        }
    
    def _calculate_adx_full(self, df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
        """Calculate ADX with +DI and -DI"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean().iloc[-1]
            
            return (
                adx if not np.isnan(adx) else 20.0,
                plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0,
                minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0,
            )
        except:
            return 20.0, 0.0, 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            return atr if not np.isnan(atr) else 0.0
        except:
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate MACD"""
        try:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            
            return (
                macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0,
                signal.iloc[-1] if not np.isnan(signal.iloc[-1]) else 0.0,
                hist.iloc[-1] if not np.isnan(hist.iloc[-1]) else 0.0,
            )
        except:
            return 0.0, 0.0, 0.0
    
    def _find_key_levels(self, df: pd.DataFrame, lookback: int = 50) -> list:
        """Find key support/resistance levels"""
        try:
            recent = df.tail(lookback)
            highs = recent['high'].values
            lows = recent['low'].values
            
            levels = []
            
            # Pivot highs
            for i in range(2, len(highs) - 2):
                if highs[i] == max(highs[i-2:i+3]):
                    levels.append(highs[i])
            
            # Pivot lows
            for i in range(2, len(lows) - 2):
                if lows[i] == min(lows[i-2:i+3]):
                    levels.append(lows[i])
            
            # Cluster nearby levels
            if levels:
                levels = sorted(levels)
                clustered = [levels[0]]
                
                for level in levels[1:]:
                    if abs(level - clustered[-1]) / clustered[-1] > 0.02:  # 2% apart
                        clustered.append(level)
                
                return clustered
            
            return []
        except:
            return []
    
    def _find_nearest_level(self, price: float, levels: list) -> Tuple[Optional[float], Optional[float]]:
        """Find nearest S/R level and distance"""
        if not levels:
            return None, None
        
        distances = [abs(price - level) for level in levels]
        min_idx = distances.index(min(distances))
        nearest = levels[min_idx]
        distance_pct = (abs(price - nearest) / nearest) * 100
        
        return nearest, distance_pct
    
    def _default_analysis(self) -> Dict:
        """Fallback analysis when data is insufficient"""
        return {
            'timestamp': datetime.now(),
            'recommended_mode': 'performance',  # Conservative default
            'confidence': 0.3,
            'regime_type': 'unknown',
            'reasoning': 'Insufficient data - defaulting to Performance mode',
            'metrics': {},
        }


class AdaptiveTPSLManager:
    """
    Dynamically adjusts TP/SL based on aggregator mode and market conditions
    """
    
    def __init__(self, asset_type: str = "BTC"):
        self.asset_type = asset_type
        
        # Base configurations (in ATR multiples)
        self.base_configs = {
            'BTC': {
                'council': {
                    'tp_multiplier': 2.5,   # Council: Ride trends
                    'sl_multiplier': 1.2,
                    'trailing_start': 1.5,  # Start trailing at 1.5 ATR profit
                    'trailing_distance': 1.0,
                },
                'performance': {
                    'tp_multiplier': 2.0,   # Performance: Quicker exits
                    'sl_multiplier': 1.0,
                    'trailing_start': 1.2,
                    'trailing_distance': 0.8,
                },
            },
            'GOLD': {
                'council': {
                    'tp_multiplier': 2.0,
                    'sl_multiplier': 1.0,
                    'trailing_start': 1.3,
                    'trailing_distance': 0.8,
                },
                'performance': {
                    'tp_multiplier': 1.5,
                    'sl_multiplier': 0.8,
                    'trailing_start': 1.0,
                    'trailing_distance': 0.6,
                },
            },
        }
    
    def calculate_tp_sl(
        self,
        entry_price: float,
        signal: int,  # 1=long, -1=short
        atr: float,
        mode: str,  # 'council' or 'performance'
        confidence: float,  # 0-1
        volatility_ratio: float,
        at_key_level: bool = False,
    ) -> Dict:
        """
        Calculate adaptive TP/SL levels
        
        Returns:
            {
                'stop_loss': float,
                'take_profit': float,
                'trailing_start': float,
                'trailing_distance': float,
                'risk_reward_ratio': float,
            }
        """
        
        # Get base config for asset and mode
        config = self.base_configs.get(self.asset_type, self.base_configs['BTC'])
        mode_config = config.get(mode, config['performance'])
        
        # ============================================================
        # ADAPTIVE ADJUSTMENTS
        # ============================================================
        
        # 1. Confidence-based scaling
        # Higher confidence = wider TP, tighter SL
        confidence_factor = 0.8 + (confidence * 0.4)  # Range: 0.8 to 1.2
        
        tp_mult = mode_config['tp_multiplier'] * confidence_factor
        sl_mult = mode_config['sl_multiplier'] / confidence_factor
        
        # 2. Volatility adjustment
        # High volatility = wider stops to avoid noise
        if volatility_ratio > 1.3:
            sl_mult *= 1.2
            tp_mult *= 1.1
        elif volatility_ratio < 0.8:
            sl_mult *= 0.9
            tp_mult *= 0.9
        
        # 3. Key level adjustment
        # At S/R = tighter stops (expecting rejection/bounce)
        if at_key_level:
            sl_mult *= 0.85
            tp_mult *= 1.15
        
        # 4. Mode-specific fine-tuning
        if mode == 'council':
            # Council: Institutional approach - ride winners
            if confidence > 0.75:
                tp_mult *= 1.2  # Let winners run
        
        elif mode == 'performance':
            # Performance: Statistical approach - quick profits
            if confidence < 0.6:
                tp_mult *= 0.9  # Take profits earlier in uncertain conditions
        
        # ============================================================
        # CALCULATE LEVELS
        # ============================================================
        
        if signal == 1:  # LONG
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
            trailing_start = entry_price + (atr * mode_config['trailing_start'])
            trailing_distance = atr * mode_config['trailing_distance']
            
        else:  # SHORT
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)
            trailing_start = entry_price - (atr * mode_config['trailing_start'])
            trailing_distance = atr * mode_config['trailing_distance']
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_start': trailing_start,
            'trailing_distance': trailing_distance,
            'risk_reward_ratio': rr_ratio,
            'atr_used': atr,
            'tp_multiplier': tp_mult,
            'sl_multiplier': sl_mult,
        }


class HybridAggregatorSelector:
    """
    Main controller for dynamic aggregator switching
    """
    
    def __init__(self, data_manager, config,telegram_bot=None):
        self.data_manager = data_manager
        self.config = config
        self.telegram_bot = telegram_bot 
        
        # Initialize analyzers
        self.market_analyzer = MarketRegimeAnalyzer()
        self.tpsl_managers = {
            'BTC': AdaptiveTPSLManager('BTC'),
            'GOLD': AdaptiveTPSLManager('GOLD'),
        }
        
        # Track current modes per asset
        self.current_modes = {}
        self.mode_history = {}
        
        # Cooldown to prevent excessive switching
        self.last_switch_time = {}
        self.min_switch_interval = timedelta(minutes=30)
        
        # Statistics
        self.stats = {
            'total_switches': 0,
            'council_signals': 0,
            'performance_signals': 0,
            'mode_duration': {},  # Track time spent in each mode
        }
    
    def get_optimal_mode(
        self,
        asset_name: str,
        df: pd.DataFrame,
    ) -> Dict:
        """
        Main method: Determine optimal aggregator mode for current market
        
        Returns:
            {
                'mode': 'council' or 'performance',
                'confidence': 0-1,
                'switch_occurred': bool,
                'analysis': full market analysis dict,
            }
        """
        try:
            # Analyze market
            analysis = self.market_analyzer.analyze_market_state(df, asset_name)
            
            recommended_mode = analysis['recommended_mode']
            confidence = analysis['confidence']
            
            # Get current mode
            current_mode = self.current_modes.get(asset_name)
            
            # Check if switch is needed
            switch_occurred = False
            
            if current_mode != recommended_mode:
                # Check cooldown
                if self._can_switch(asset_name):
                    # Only switch if confidence is high enough
                    if confidence >= 0.55:
                        self._execute_mode_switch(
                            asset=asset_name,
                            old_mode=current_mode,
                            new_mode=recommended_mode,
                            analysis=analysis,
                        )
                        switch_occurred = True
                    else:
                        logger.debug(
                            f"[HYBRID] {asset_name}: Switch suggested but confidence "
                            f"too low ({confidence:.2%})"
                        )
                else:
                    logger.debug(
                        f"[HYBRID] {asset_name}: In cooldown period "
                        f"({current_mode} → {recommended_mode} blocked)"
                    )
            
            return {
                'mode': self.current_modes.get(asset_name, recommended_mode),
                'confidence': confidence,
                'switch_occurred': switch_occurred,
                'analysis': analysis,
            }
        
        except Exception as e:
            logger.error(f"[HYBRID] Mode selection error: {e}", exc_info=True)
            return {
                'mode': self.current_modes.get(asset_name, 'performance'),
                'confidence': 0.3,
                'switch_occurred': False,
                'analysis': self.market_analyzer._default_analysis(),
            }
        
    
    def calculate_tp_sl(
        self,
        asset_name: str,
        entry_price: float,
        signal: int,
        df: pd.DataFrame,
        mode: str,
        confidence: float,
    ) -> Dict:
        """
        Calculate adaptive TP/SL for the current trade
        """
        try:
            asset_type = 'BTC' if 'BTC' in asset_name.upper() else 'GOLD'
            manager = self.tpsl_managers.get(asset_type)
            
            if not manager:
                raise ValueError(f"No TP/SL manager for {asset_type}")
            
            # Get market metrics
            atr = manager._calculate_atr(df) if hasattr(manager, '_calculate_atr') else \
                  self.market_analyzer._calculate_atr(df)
            
            analysis = self.market_analyzer.analyze_market_state(df, asset_name)
            volatility_ratio = analysis['metrics'].get('atr_ratio', 1.0)
            at_key_level = analysis.get('at_key_level', False)
            
            # Calculate levels
            tp_sl = manager.calculate_tp_sl(
                entry_price=entry_price,
                signal=signal,
                atr=atr,
                mode=mode,
                confidence=confidence,
                volatility_ratio=volatility_ratio,
                at_key_level=at_key_level,
            )
            
            return tp_sl
        
        except Exception as e:
            logger.error(f"[HYBRID] TP/SL calculation error: {e}")
            # Fallback: basic ATR-based calculation
            atr = self.market_analyzer._calculate_atr(df)
            return {
                'stop_loss': entry_price - (atr * 1.0) if signal == 1 else entry_price + (atr * 1.0),
                'take_profit': entry_price + (atr * 2.0) if signal == 1 else entry_price - (atr * 2.0),
                'trailing_start': entry_price + (atr * 1.5) if signal == 1 else entry_price - (atr * 1.5),
                'trailing_distance': atr * 0.8,
                'risk_reward_ratio': 2.0,
            }
    
    def _can_switch(self, asset_name: str) -> bool:
        """Check if cooldown period has passed"""
        last_switch = self.last_switch_time.get(asset_name)
        
        if last_switch is None:
            return True
        
        elapsed = datetime.now() - last_switch
        return elapsed >= self.min_switch_interval
    
    def _execute_mode_switch(
        self,
        asset: str,
        old_mode: Optional[str],
        new_mode: str,
        analysis: Dict,
    ):
        """
        Execute aggregator mode switch with Telegram notification
        """
        
        # Console logging
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[HYBRID] MODE SWITCH - {asset}")
        logger.info(f"{'=' * 70}")
        logger.info(f"OLD MODE: {old_mode or 'None'}")
        logger.info(f"NEW MODE: {new_mode.upper()}")
        logger.info(f"CONFIDENCE: {analysis['confidence']:.0%}")
        logger.info(f"\nREGIME: {analysis['regime_type']}")
        logger.info(f"REASONING: {analysis['reasoning']}")
        logger.info(f"\nTREND:")
        logger.info(f"  Strength:  {analysis['trend']['strength']}")
        logger.info(f"  Direction: {analysis['trend']['direction']}")
        logger.info(f"  ADX:       {analysis['trend']['adx']:.1f}")
        logger.info(f"\nVOLATILITY:")
        logger.info(f"  Regime: {analysis['volatility']['regime']}")
        logger.info(f"  Ratio:  {analysis['volatility']['ratio']:.2f}x")
        logger.info(f"\nPRICE ACTION:")
        logger.info(f"  Clarity:       {analysis['price_action']['clarity']}")
        logger.info(f"  Indecision %:  {analysis['price_action']['indecision_pct']:.0f}%")
        logger.info(f"{'=' * 70}\n")
        
        # Update state
        self.current_modes[asset] = new_mode
        self.last_switch_time[asset] = datetime.now()
        
        # Track statistics
        self.stats['total_switches'] += 1
        if new_mode == 'council':
            self.stats['council_signals'] += 1
        else:
            self.stats['performance_signals'] += 1
        
        # Store in history
        if asset not in self.mode_history:
            self.mode_history[asset] = []
        
        switch_record = {
            'timestamp': datetime.now(),
            'old_mode': old_mode,
            'new_mode': new_mode,
            'confidence': analysis['confidence'],
            'regime_type': analysis['regime_type'],
            'reasoning': analysis['reasoning'],
            'trend': analysis['trend'],
            'volatility': analysis['volatility'],
            'price_action': analysis['price_action'],
        }
        
        self.mode_history[asset].append(switch_record)
        
        # ✅ NEW: Send Telegram notification
        if self.telegram_bot:
            try:
                self._send_switch_notification(asset, switch_record)
            except Exception as e:
                logger.error(f"[HYBRID] Failed to send Telegram notification: {e}")
    
    def _send_switch_notification(self, asset: str, switch_record: Dict):
        """
        ✅ NEW: Send rich Telegram notification for mode switch
        """
        old_mode = switch_record['old_mode'] or 'None'
        new_mode = switch_record['new_mode']
        confidence = switch_record['confidence']
        
        # Emoji for asset
        asset_emoji = '₿' if asset == 'BTC' else '🥇'
        
        # Mode emojis
        mode_emoji = {
            'council': '🏛️',
            'performance': '📊',
        }
        
        # Regime emoji
        regime = switch_record['regime_type']
        regime_emoji = {
            'trending_clean': '📈',
            'trending_volatile': '⚡',
            'ranging_quiet': '➡️',
            'volatile_choppy': '🌊',
            'reversal_setup': '🔄',
            'mixed_signals': '❓',
        }.get(regime, '📊')
        
        # Build message
        msg = (
            f"{asset_emoji} *AGGREGATOR MODE SWITCH: {asset}*\n\n"
            f"{mode_emoji.get(old_mode, '❓')} `{old_mode.upper()}` → "
            f"{mode_emoji.get(new_mode, '❓')} `{new_mode.upper()}`\n"
            f"*Confidence:* {confidence:.0%}\n\n"
            f"{regime_emoji} *Market Regime:* {regime.replace('_', ' ').title()}\n"
            f"*Reasoning:* {switch_record['reasoning']}\n\n"
            f"*Trend Analysis:*\n"
            f"  • Strength: {switch_record['trend']['strength'].title()}\n"
            f"  • Direction: {switch_record['trend']['direction'].title()}\n"
            f"  • ADX: {switch_record['trend']['adx']:.1f}\n\n"
            f"*Volatility:*\n"
            f"  • Regime: {switch_record['volatility']['regime'].title()}\n"
            f"  • Ratio: {switch_record['volatility']['ratio']:.2f}x\n\n"
            f"*Price Action:*\n"
            f"  • Clarity: {switch_record['price_action']['clarity'].title()}\n"
            f"  • Indecision: {switch_record['price_action']['indecision_pct']:.0f}%\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Send via Telegram bot's notification method
        import asyncio
        
        if hasattr(self.telegram_bot, '_send_telegram_notification'):
            # If using the trading bot's method
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.telegram_bot.send_notification(msg, disable_preview=True)
                    )
                else:
                    asyncio.run(
                        self.telegram_bot.send_notification(msg, disable_preview=True)
                    )
            except:
                # Fallback
                self.telegram_bot.send_notification(msg, disable_preview=True)
        else:
            # Direct call
            asyncio.run(
                self.telegram_bot.send_notification(msg, disable_preview=True)
            )
    
    def get_statistics(self) -> Dict:
        """Get statistics on mode switches"""
        return {
            **self.stats,
            'current_modes': self.current_modes.copy(),
            'mode_history': {
                asset: len(history)
                for asset, history in self.mode_history.items()
            },
        }
    
    def get_mode_history(self, asset: str, n: int = 5) -> list:
        """
        ✅ NEW: Get recent mode history for an asset
        
        Args:
            asset: Asset name (BTC/GOLD)
            n: Number of recent switches to return
            
        Returns:
            List of switch records
        """
        if asset not in self.mode_history:
            return []
        
        return self.mode_history[asset][-n:]



