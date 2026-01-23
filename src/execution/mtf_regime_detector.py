"""
Multi-Timeframe Regime Detector - ENHANCED WITH GOVERNOR
==========================================================
Analyzes market regime across 1H, 4H, and 1D timeframes.
✨ NEW: Daily 200 EMA Governor logic for macro trend filtering

Governor Rules:
1. TREND MODE: Price > 200 EMA AND Slope > 0 → Safe for big trades
2. SCALP MODE: Price choppy/flat → Small trades only
3. V-SHAPE OVERRIDE: Price > 200 EMA, Slope < 0, RSI > 60, Volume spike → Allow recovery trades
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Supported timeframes"""

    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"


class RegimeType(Enum):
    """Regime classifications"""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    
class TradeType(Enum):
    """Trade type based on Governor analysis"""
    TREND = "TREND"      # Macro trend aligned - use 2% risk
    SCALP = "SCALP"      # Choppy/ranging - use 1% risk
    V_SHAPE = "V_SHAPE"  # Recovery play - use 1.5% risk


@dataclass
class GovernorAnalysis:
    """
    ✨ Governor (Daily 200 EMA) analysis results
    """
    # Raw data
    current_price: float
    ema_200: float
    ema_slope: float
    
    # Position relative to EMA
    price_above_ema: bool
    distance_from_ema_pct: float
    
    # Slope analysis
    slope_positive: bool
    slope_strength: str
    
    # V-Shape detection
    rsi: float
    volume_ratio: float
    volume_spike: bool
    v_shape_conditions_met: bool
    
    # Final classification
    regime: str
    trade_type: TradeType
    confidence: float
    
    # Explanation
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dict for logging"""
        return {
            "current_price": self.current_price,
            "ema_200": self.ema_200,
            "ema_slope": self.ema_slope,
            "price_above_ema": self.price_above_ema,
            "distance_from_ema_pct": self.distance_from_ema_pct,
            "slope_positive": self.slope_positive,
            "slope_strength": self.slope_strength,
            "rsi": self.rsi,
            "volume_ratio": self.volume_ratio,
            "volume_spike": self.volume_spike,
            "v_shape_detected": self.v_shape_conditions_met,
            "regime": self.regime,
            "trade_type": self.trade_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }

@dataclass
class TimeFrameAnalysis:
    """Analysis results for a single timeframe"""
    timeframe: str
    regime: RegimeType
    confidence: float
    
    # Trend metrics
    ema_diff_pct: float
    price_vs_ema50: float
    adx: float
    trend_strength: str
    trend_direction: str
    
    # Momentum metrics
    rsi: float
    macd_histogram: float
    momentum_aligned: bool
    
    # Volatility metrics
    atr_pct: float
    volatility_regime: str
    
    # Price action
    returns_20: float
    returns_50: float
    higher_highs: bool
    higher_lows: bool
    
    # Supporting data
    timestamp: datetime
    weight: float = 1.0


@dataclass
class MultiTimeFrameRegime:
    """Aggregated regime across multiple timeframes"""
    asset: str
    timestamp: datetime
    
    # Individual timeframe results
    tf_1h: Optional[TimeFrameAnalysis]
    tf_4h: Optional[TimeFrameAnalysis]
    tf_1d: Optional[TimeFrameAnalysis]
    
    # ✨ Governor analysis
    governor: Optional[GovernorAnalysis]
    
    # Consensus regime
    consensus_regime: RegimeType
    consensus_confidence: float
    
    # Alignment metrics
    timeframe_agreement: float
    trend_coherence: float
    
    # Risk assessment
    risk_level: str
    volatility_regime: str
    
    # Trading implications
    recommended_mode: str
    allow_counter_trend: bool
    suggested_max_positions: int
    
    # ✨ Trade type from Governor
    trade_type: TradeType
    
    # Raw scores
    bullish_score: float
    bearish_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        result = {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "consensus_regime": self.consensus_regime.value,
            "consensus_confidence": self.consensus_confidence,
            "timeframe_agreement": self.timeframe_agreement,
            "trend_coherence": self.trend_coherence,
            "risk_level": self.risk_level,
            "volatility_regime": self.volatility_regime,
            "recommended_mode": self.recommended_mode,
            "allow_counter_trend": self.allow_counter_trend,
            "suggested_max_positions": self.suggested_max_positions,
            "trade_type": self.trade_type.value,
            "bullish_score": self.bullish_score,
            "bearish_score": self.bearish_score,
        }
        
        # Add Governor data
        if self.governor:
            result["governor"] = self.governor.to_dict()
        
        # Add individual timeframe data
        for tf_name, tf_data in [("1h", self.tf_1h), ("4h", self.tf_4h), ("1d", self.tf_1d)]:
            if tf_data:
                result[f"{tf_name}_regime"] = tf_data.regime.value
                result[f"{tf_name}_confidence"] = tf_data.confidence
                result[f"{tf_name}_trend_strength"] = tf_data.trend_strength
                result[f"{tf_name}_trend_direction"] = tf_data.trend_direction
                result[f"{tf_name}_adx"] = tf_data.adx
                result[f"{tf_name}_rsi"] = tf_data.rsi
                result[f"{tf_name}_ema_diff_pct"] = tf_data.ema_diff_pct
                result[f"{tf_name}_volatility"] = tf_data.volatility_regime
                result[f"{tf_name}_returns_20"] = tf_data.returns_20
        
        return result


class MultiTimeFrameRegimeDetector:
    """
    Analyzes market regime across multiple timeframes.
    ✨ ENHANCED: Now includes Daily 200 EMA Governor logic
    
    Weighting:
    - 1D: 50% (primary trend + Governor)
    - 4H: 30% (intermediate trend)
    - 1H: 20% (short-term noise filter)
    """
    
    def __init__(self, data_manager, asset_type: str = "BTC"):
        """Initialize detector"""
        self.data_manager = data_manager
        self.asset_type = asset_type.upper()
        
        # Timeframe weights
        self.weights = {
            TimeFrame.ONE_DAY: 0.50,
            TimeFrame.FOUR_HOUR: 0.30,
            TimeFrame.ONE_HOUR: 0.20,
        }
        
        # Governor thresholds
        self.governor_thresholds = {
            "ema_slope_strong": 0.005,
            "ema_slope_moderate": 0.002,
            "ema_slope_negative": -0.002,
            "v_shape_rsi_min": 60,
            "v_shape_volume_min": 1.5,
            "distance_danger_pct": 0.10,
            "min_required_bars": 220,  # ✨ NEW: Make configurable
        }
        
        # Asset-specific thresholds
        if self.asset_type == "BTC":
            self.thresholds = {
                "ema_bull": 0.15,
                "ema_bear": -0.15,
                "adx_strong": 25,
                "adx_weak": 20,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volatility_high": 0.40,
                "volatility_low": 0.15,
            }
        else:  # GOLD
            self.thresholds = {
                "ema_bull": 0.10,
                "ema_bear": -0.10,
                "adx_strong": 25,
                "adx_weak": 20,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volatility_high": 0.30,
                "volatility_low": 0.10,
            }
        
        # Cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info(f"[MTF + GOVERNOR] Initialized for {asset_type}")
        logger.info(f"  Weights: 1D={self.weights[TimeFrame.ONE_DAY]:.0%} (+ Governor), "
                    f"4H={self.weights[TimeFrame.FOUR_HOUR]:.0%}, "
                   f"1H={self.weights[TimeFrame.ONE_HOUR]:.0%}")
        logger.info(f"  Governor EMA: 200-period Daily")
        logger.info(f"  Governor Modes: TREND / SCALP / V_SHAPE")
    
    def analyze_regime(
        self, 
        symbol: str, 
        exchange: str = "binance", 
        force_refresh: bool = False
    ) -> MultiTimeFrameRegime:
        """
        ✨ FIXED: Graceful Governor fallback when data insufficient
        """
        # Check cache
        cache_key = f"{symbol}_{exchange}"
        if not force_refresh and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                logger.debug(f"[MTF] Using cached result for {symbol}")
                return cached_result
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[MTF + GOVERNOR] Analyzing {self.asset_type} ({symbol})")
        logger.info(f"{'='*70}")
        
        # ✨ STEP 1: Analyze Governor (with graceful fallback)
        governor_analysis = None
        try:
            governor_analysis = self._analyze_governor(symbol, exchange)
            
            logger.info(f"\n[GOVERNOR] Daily 200 EMA Analysis:")
            logger.info(f"  Regime:         {governor_analysis.regime}")
            logger.info(f"  Trade Type:     {governor_analysis.trade_type.value}")
            logger.info(f"  Confidence:     {governor_analysis.confidence:.2%}")
            logger.info(f"  Reasoning:      {governor_analysis.reasoning}")
        
        except ValueError as e:
            # ✨ GRACEFUL FALLBACK: Insufficient data
            if "Insufficient daily data" in str(e):
                logger.warning(f"[GOVERNOR] {e}")
                logger.warning(f"[GOVERNOR] Using fallback: SCALP mode (conservative)")
                
                # Create fallback Governor analysis
                governor_analysis = GovernorAnalysis(
                    current_price=0.0,
                    ema_200=0.0,
                    ema_slope=0.0,
                    price_above_ema=False,
                    distance_from_ema_pct=0.0,
                    slope_positive=False,
                    slope_strength="unknown",
                    rsi=50.0,
                    volume_ratio=1.0,
                    volume_spike=False,
                    v_shape_conditions_met=False,
                    regime="SCALP_MODE",
                    trade_type=TradeType.SCALP,
                    confidence=0.50,
                    reasoning="Insufficient data - using conservative SCALP mode",
                    timestamp=datetime.now(),
                )
            else:
                raise
        
        except Exception as e:
            logger.error(f"[GOVERNOR] Analysis failed: {e}", exc_info=True)
            # Create error fallback
            governor_analysis = GovernorAnalysis(
                current_price=0.0,
                ema_200=0.0,
                ema_slope=0.0,
                price_above_ema=False,
                distance_from_ema_pct=0.0,
                slope_positive=False,
                slope_strength="error",
                rsi=50.0,
                volume_ratio=1.0,
                volume_spike=False,
                v_shape_conditions_met=False,
                regime="SCALP_MODE",
                trade_type=TradeType.SCALP,
                confidence=0.50,
                reasoning=f"Error in analysis - using conservative SCALP mode: {str(e)}",
                timestamp=datetime.now(),
            )
        
        # STEP 2: Analyze each timeframe
        tf_results = {}
        for timeframe in [TimeFrame.ONE_HOUR, TimeFrame.FOUR_HOUR, TimeFrame.ONE_DAY]:
            try:
                analysis = self._analyze_timeframe(symbol, timeframe, exchange)
                tf_results[timeframe] = analysis
                
                logger.info(f"\n[{timeframe.value.upper()}] Analysis:")
                logger.info(f"  Regime:     {analysis.regime.value.upper()}")
                logger.info(f"  Confidence: {analysis.confidence:.2%}")
            
            except Exception as e:
                logger.error(f"[MTF] Error analyzing {timeframe.value}: {e}")
                tf_results[timeframe] = None
        
        # STEP 3: Aggregate results
        result = self._aggregate_timeframes(
            asset=self.asset_type,
            tf_1h=tf_results.get(TimeFrame.ONE_HOUR),
            tf_4h=tf_results.get(TimeFrame.FOUR_HOUR),
            tf_1d=tf_results.get(TimeFrame.ONE_DAY),
            governor=governor_analysis,
        )
        
        # Log consensus
        logger.info(f"\n{'='*70}")
        logger.info(f"[CONSENSUS] {result.consensus_regime.value.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"  Trade Type:    {result.trade_type.value}")
        logger.info(f"  Confidence:    {result.consensus_confidence:.2%}")
        logger.info(f"{'='*70}\n")
        
        # Cache result
        self.cache[cache_key] = (result, datetime.now())
        
        return result
    
    def _analyze_governor(self, symbol: str, exchange: str) -> GovernorAnalysis:
        """
        ✅ FIXED: Uses CSV files for daily data (full history)
        """
        # ✅ CHANGED: Use CSV instead of API
        df_daily = self._fetch_data_from_csv(symbol, TimeFrame.ONE_DAY, exchange)
        
        min_bars = self.governor_thresholds["min_required_bars"]
        
        if len(df_daily) < min_bars:
            raise ValueError(
                f"Insufficient daily data: {len(df_daily)} bars (need {min_bars}+). "
                f"CSV file exists but needs more data. Run: python download_multi_tf_data.py"
            )
        
        # Calculate 200 EMA
        ema_200 = df_daily['close'].ewm(span=200, adjust=False).mean()
        current_price = float(df_daily['close'].iloc[-1])
        current_ema = float(ema_200.iloc[-1])
        
        # Calculate EMA slope
        ema_20_days_ago = float(ema_200.iloc[-20])
        ema_slope = (current_ema - ema_20_days_ago) / ema_20_days_ago
        
        # Price position
        price_above_ema = current_price > current_ema
        distance_pct = (current_price - current_ema) / current_ema
        
        # Slope classification
        if ema_slope > self.governor_thresholds["ema_slope_strong"]:
            slope_strength = "strong"
            slope_positive = True
        elif ema_slope > self.governor_thresholds["ema_slope_moderate"]:
            slope_strength = "moderate"
            slope_positive = True
        elif ema_slope > self.governor_thresholds["ema_slope_negative"]:
            slope_strength = "weak"
            slope_positive = True
        else:
            slope_strength = "negative"
            slope_positive = False
        
        # RSI
        delta = df_daily['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1])
        
        # Volume analysis
        current_volume = float(df_daily['volume'].iloc[-1])
        volume_ma = float(df_daily['volume'].rolling(20).mean().iloc[-1])
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
        volume_spike = volume_ratio > self.governor_thresholds["v_shape_volume_min"]
        
        # Decision Logic
        if price_above_ema and slope_positive:
            regime = "TREND_MODE"
            trade_type = TradeType.TREND
            confidence = 0.85
            reasoning = f"Price above 200 EMA with {slope_strength} positive slope"
        
        elif (price_above_ema and 
              not slope_positive and 
              rsi > self.governor_thresholds["v_shape_rsi_min"] and 
              volume_spike):
            regime = "V_SHAPE_OVERRIDE"
            trade_type = TradeType.V_SHAPE
            confidence = 0.70
            reasoning = f"V-shape recovery: RSI={rsi:.0f}, Volume {volume_ratio:.1f}x"
        
        else:
            regime = "SCALP_MODE"
            trade_type = TradeType.SCALP
            confidence = 0.60
            
            if not price_above_ema:
                reasoning = "Price below 200 EMA - Defensive mode"
            elif not slope_positive:
                reasoning = "Price above EMA but slope negative - Choppy"
            else:
                reasoning = "Unclear trend - Conservative mode"
        
        # Adjust confidence based on distance
        if abs(distance_pct) > self.governor_thresholds["distance_danger_pct"]:
            confidence *= 0.85
        
        v_shape_met = (
            price_above_ema and 
            not slope_positive and 
            rsi > self.governor_thresholds["v_shape_rsi_min"] and 
            volume_spike
        )
        
        return GovernorAnalysis(
            current_price=current_price,
            ema_200=current_ema,
            ema_slope=ema_slope,
            price_above_ema=price_above_ema,
            distance_from_ema_pct=distance_pct,
            slope_positive=slope_positive,
            slope_strength=slope_strength,
            rsi=rsi,
            volume_ratio=volume_ratio,
            volume_spike=volume_spike,
            v_shape_conditions_met=v_shape_met,
            regime=regime,
            trade_type=trade_type,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
        )

    def _analyze_timeframe(
        self, symbol: str, timeframe: TimeFrame, exchange: str
    ) -> TimeFrameAnalysis:
        """
        Analyze a single timeframe

        Args:
            symbol: Trading symbol
            timeframe: TimeFrame enum
            exchange: "binance" or "mt5"

        Returns:
            TimeFrameAnalysis object
        """
        # Fetch data
        df = self._fetch_data(symbol, timeframe, exchange)

        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} rows")

        # Calculate indicators
        df = self._calculate_indicators(df)

        latest = df.iloc[-1]

        # Extract metrics
        close = latest["close"]
        ema_20 = latest["ema_20"]
        ema_50 = latest["ema_50"]
        ema_diff_pct = (ema_20 - ema_50) / ema_50
        price_vs_ema50 = (close - ema_50) / ema_50

        adx = latest["adx"]
        rsi = latest["rsi"]
        macd_hist = latest["macd_hist"]
        atr = latest["atr"]
        atr_pct = atr / close

        # Calculate returns
        returns_20 = (
            (close - df["close"].iloc[-20]) / df["close"].iloc[-20]
            if len(df) >= 20
            else 0.0
        )
        returns_50 = (
            (close - df["close"].iloc[-50]) / df["close"].iloc[-50]
            if len(df) >= 50
            else 0.0
        )

        # Price action analysis
        highs = df["high"].tail(20).values
        lows = df["low"].tail(20).values
        higher_highs = len(highs) >= 10 and highs[-1] > np.max(highs[:-5])
        higher_lows = len(lows) >= 10 and lows[-1] > np.min(lows[:-5])

        # Determine trend strength and direction
        if adx > self.thresholds["adx_strong"]:
            trend_strength = "strong"
        elif adx > self.thresholds["adx_weak"]:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        if ema_diff_pct > 0.01:
            trend_direction = "up"
        elif ema_diff_pct < -0.01:
            trend_direction = "down"
        else:
            trend_direction = "sideways"

        # Momentum alignment
        momentum_aligned = (ema_diff_pct > 0 and macd_hist > 0) or (
            ema_diff_pct < 0 and macd_hist < 0
        )

        # Volatility regime
        if atr_pct > self.thresholds["volatility_high"]:
            volatility_regime = "high"
        elif atr_pct < self.thresholds["volatility_low"]:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"

        # Determine regime
        regime, confidence = self._classify_regime(
            ema_diff_pct=ema_diff_pct,
            price_vs_ema50=price_vs_ema50,
            adx=adx,
            rsi=rsi,
            macd_hist=macd_hist,
            returns_20=returns_20,
            returns_50=returns_50,
            trend_strength=trend_strength,
            momentum_aligned=momentum_aligned,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
        )

        return TimeFrameAnalysis(
            timeframe=timeframe.value,
            regime=regime,
            confidence=confidence,
            ema_diff_pct=ema_diff_pct,
            price_vs_ema50=price_vs_ema50,
            adx=adx,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            rsi=rsi,
            macd_histogram=macd_hist,
            momentum_aligned=momentum_aligned,
            atr_pct=atr_pct,
            volatility_regime=volatility_regime,
            returns_20=returns_20,
            returns_50=returns_50,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            timestamp=datetime.now(),
        )

    def _classify_regime(
        self,
        ema_diff_pct: float,
        price_vs_ema50: float,
        adx: float,
        rsi: float,
        macd_hist: float,
        returns_20: float,
        returns_50: float,
        trend_strength: str,
        momentum_aligned: bool,
        higher_highs: bool,
        higher_lows: bool,
    ) -> Tuple[RegimeType, float]:
        """
        Classify regime based on multiple factors

        Returns:
            (regime, confidence)
        """
        # Initialize scores
        bullish_score = 0
        bearish_score = 0

        # EMA alignment (3 points)
        if ema_diff_pct > self.thresholds["ema_bull"]:
            bullish_score += 3
        elif ema_diff_pct < self.thresholds["ema_bear"]:
            bearish_score += 3

        # Price vs EMA50 (2 points)
        if price_vs_ema50 > 0.02:
            bullish_score += 2
        elif price_vs_ema50 < -0.02:
            bearish_score += 2

        # Returns (2 points)
        if returns_20 > 0.03:
            bullish_score += 2
        elif returns_20 < -0.03:
            bearish_score += 2

        # MACD (1 point)
        if macd_hist > 0:
            bullish_score += 1
        elif macd_hist < 0:
            bearish_score += 1

        # ADX with trend (1 point)
        if adx > self.thresholds["adx_strong"]:
            if ema_diff_pct > 0:
                bullish_score += 1
            else:
                bearish_score += 1

        # RSI (1 point)
        if rsi > 60:
            bullish_score += 1
        elif rsi < 40:
            bearish_score += 1

        # Price action (1 point)
        if higher_highs and higher_lows:
            bullish_score += 1
        elif not higher_highs and not higher_lows:
            bearish_score += 1

        # Total possible: 10 points
        total_score = bullish_score + bearish_score

        # Classify regime
        if bullish_score >= 7:
            regime = RegimeType.STRONG_BULL
            confidence = bullish_score / 10.0
        elif bullish_score >= 5:
            regime = RegimeType.BULL
            confidence = bullish_score / 10.0
        elif bearish_score >= 7:
            regime = RegimeType.STRONG_BEAR
            confidence = bearish_score / 10.0
        elif bearish_score >= 5:
            regime = RegimeType.BEAR
            confidence = bearish_score / 10.0
        else:
            regime = RegimeType.NEUTRAL
            confidence = 0.5 + abs(bullish_score - bearish_score) / 20.0

        # Adjust confidence based on trend strength
        if trend_strength == "strong":
            confidence *= 1.15
        elif trend_strength == "weak":
            confidence *= 0.85

        # Adjust for momentum alignment
        if momentum_aligned:
            confidence *= 1.10

        confidence = min(1.0, confidence)

        return regime, confidence

    def _aggregate_timeframes(
        self,
        asset: str,
        tf_1h: Optional[TimeFrameAnalysis],
        tf_4h: Optional[TimeFrameAnalysis],
        tf_1d: Optional[TimeFrameAnalysis],
        governor: Optional[GovernorAnalysis],  # ✨ NEW
    ) -> MultiTimeFrameRegime:
        """
        Aggregate timeframe analyses + Governor into consensus regime
        
        Args:
            asset: Asset name
            tf_1h: 1H analysis
            tf_4h: 4H analysis
            tf_1d: 1D analysis
            governor: Governor (Daily 200 EMA) analysis
        
        Returns:
            MultiTimeFrameRegime object
        """
        # [Existing aggregation logic...]
        # Calculate weighted scores (keeping existing code)
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        timeframes = [
            (TimeFrame.ONE_HOUR, tf_1h),
            (TimeFrame.FOUR_HOUR, tf_4h),
            (TimeFrame.ONE_DAY, tf_1d),
        ]
        
        for tf_enum, tf_data in timeframes:
            if tf_data:
                weight = self.weights[tf_enum]
                
                if tf_data.regime == RegimeType.STRONG_BULL:
                    bullish_score += 10 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.BULL:
                    bullish_score += 7 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.STRONG_BEAR:
                    bearish_score += 10 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.BEAR:
                    bearish_score += 7 * weight * tf_data.confidence
                
                total_weight += weight
        
        # Normalize if not all timeframes available
        if total_weight < 1.0 and total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
        
        # Determine consensus regime
        if bullish_score >= 7.0:
            consensus = RegimeType.STRONG_BULL
            consensus_conf = min(1.0, bullish_score / 10.0)
        elif bullish_score >= 5.0:
            consensus = RegimeType.BULL
            consensus_conf = min(1.0, bullish_score / 10.0)
        elif bearish_score >= 7.0:
            consensus = RegimeType.STRONG_BEAR
            consensus_conf = min(1.0, bearish_score / 10.0)
        elif bearish_score >= 5.0:
            consensus = RegimeType.BEAR
            consensus_conf = min(1.0, bearish_score / 10.0)
        else:
            consensus = RegimeType.NEUTRAL
            consensus_conf = 0.5
        
        # Calculate agreement
        regimes = [tf.regime for tf in [tf_1h, tf_4h, tf_1d] if tf]
        if len(regimes) >= 2:
            agreement = sum(1 for r in regimes if r == consensus) / len(regimes)
        else:
            agreement = 1.0 if len(regimes) == 1 else 0.5
        
        # Calculate trend coherence
        directions = [tf.trend_direction for tf in [tf_1h, tf_4h, tf_1d] if tf]
        if directions:
            most_common = max(set(directions), key=directions.count)
            coherence = directions.count(most_common) / len(directions)
        else:
            coherence = 0.5
        
        # Determine volatility
        volatility = (tf_1d or tf_4h or tf_1h).volatility_regime if any([tf_1d, tf_4h, tf_1h]) else "normal"
        
        # Risk assessment
        if volatility == "high" or agreement < 0.5:
            risk_level = "high"
        elif volatility == "low" and agreement > 0.8:
            risk_level = "low"
        else:
            risk_level = "medium"
        
        # ✨ NEW: Determine trade type from Governor
        if governor:
            trade_type = governor.trade_type
        else:
            # Fallback if Governor failed
            trade_type = TradeType.SCALP
        
        # Trading implications
        recommended_mode, allow_counter, max_positions = self._determine_trading_mode(
            consensus=consensus,
            confidence=consensus_conf,
            agreement=agreement,
            volatility=volatility,
            risk_level=risk_level,
            trade_type=trade_type,  # ✨ NEW: Pass trade type
        )
        
        return MultiTimeFrameRegime(
            asset=asset,
            timestamp=datetime.now(),
            tf_1h=tf_1h,
            tf_4h=tf_4h,
            tf_1d=tf_1d,
            governor=governor,  # ✨ NEW
            consensus_regime=consensus,
            consensus_confidence=consensus_conf,
            timeframe_agreement=agreement,
            trend_coherence=coherence,
            risk_level=risk_level,
            volatility_regime=volatility,
            recommended_mode=recommended_mode,
            allow_counter_trend=allow_counter,
            suggested_max_positions=max_positions,
            trade_type=trade_type,  # ✨ NEW
            bullish_score=bullish_score,
            bearish_score=bearish_score,
        )
    
    def _determine_trading_mode(
        self,
        consensus: RegimeType,
        confidence: float,
        agreement: float,
        volatility: str,
        risk_level: str,
        trade_type: TradeType,  # ✨ NEW
    ) -> Tuple[str, bool, int]:
        """
        Determine optimal trading mode based on regime + Governor
        
        Returns:
            (recommended_mode, allow_counter_trend, max_positions)
        """
        # ✨ NEW: Governor override for TREND mode
        if trade_type == TradeType.TREND:
            # Macro trend is strong - be aggressive
            if consensus in [RegimeType.STRONG_BULL, RegimeType.STRONG_BEAR]:
                mode = "aggressive"
                allow_counter = False
                max_pos = 3
            else:
                mode = "balanced"
                allow_counter = False
                max_pos = 2
        
        elif trade_type == TradeType.V_SHAPE:
            # Recovery play - moderate aggression
            mode = "balanced"
            allow_counter = False
            max_pos = 2
        
        else:  # SCALP mode
            # Conservative by default
            mode = "scalper"
            allow_counter = True
            max_pos = 1
            
            # Allow slight increase if conditions are good
            if volatility == "low" and agreement > 0.6:
                max_pos = 2
        
        # Adjust for risk
        if risk_level == "high":
            if mode == "aggressive":
                mode = "balanced"
            max_pos = max(1, max_pos - 1)
        
        return mode, allow_counter, max_pos

    def _fetch_data_from_csv(
        self, symbol: str, timeframe: TimeFrame, exchange: str
    ) -> pd.DataFrame:
        """
        ✅ NEW: Fetch data from local CSV files (much faster + full history)
        
        Falls back to API if CSV doesn't exist or is stale.
        
        Args:
            symbol: Trading symbol
            timeframe: TimeFrame enum
            exchange: "binance" or "mt5"
        
        Returns:
            DataFrame with OHLCV data
        """
        # Determine CSV file path
        if exchange == "binance":
            asset = "btc"
            if timeframe == TimeFrame.ONE_HOUR:
                csv_file = Path("data/train_data_btc_1h.csv")
            elif timeframe == TimeFrame.FOUR_HOUR:
                csv_file = Path("data/train_data_btc_4h.csv")
            else:  # ONE_DAY
                csv_file = Path("data/train_data_btc_1d.csv")
        else:  # mt5
            asset = "gold"
            if timeframe == TimeFrame.ONE_HOUR:
                csv_file = Path("data/train_data_gold_1h.csv")
            elif timeframe == TimeFrame.FOUR_HOUR:
                csv_file = Path("data/train_data_gold_4h.csv")
            else:  # ONE_DAY
                csv_file = Path("data/train_data_gold_1d.csv")
        
        # Try to read from CSV
        if csv_file.exists():
            try:
                logger.info(f"[CSV] Reading {asset.upper()} {timeframe.value} from {csv_file}")
                
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                
                # Ensure timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Check data freshness (should be within last 24 hours)
                latest_date = df.index[-1]
                hours_old = (pd.Timestamp.now(tz='UTC') - latest_date).total_seconds() / 3600
                
                if hours_old > 24:
                    logger.warning(f"[CSV] Data is {hours_old:.1f} hours old - consider updating")
                
                logger.info(f"[CSV] ✓ Loaded {len(df)} bars from CSV")
                logger.info(f"[CSV]   Range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"[CSV]   Age: {hours_old:.1f} hours old")
                
                return df
                
            except Exception as e:
                logger.warning(f"[CSV] Failed to read {csv_file}: {e}")
                logger.info(f"[CSV] Falling back to API fetch")
        
        else:
            logger.info(f"[CSV] File not found: {csv_file}")
            logger.info(f"[CSV] Falling back to API fetch")
        
        # Fallback: Use original API fetch method
        return self._fetch_data(symbol, timeframe, exchange)
    
    def _fetch_data(
        self, symbol: str, timeframe: TimeFrame, exchange: str
    ) -> pd.DataFrame:
        """
        Fetch historical data for the specified timeframe

        Args:
            symbol: Trading symbol
            timeframe: TimeFrame enum
            exchange: "binance" or "mt5"

        Returns:
            DataFrame with OHLCV data
        """
        end_time = datetime.now(timezone.utc)

        # Determine lookback based on timeframe
        if timeframe == TimeFrame.ONE_HOUR:
            lookback_days = 30
            interval = "1h"
            mt5_timeframe = "H1"
        elif timeframe == TimeFrame.FOUR_HOUR:
            lookback_days = 60
            interval = "4h"
            mt5_timeframe = "H4"
        else:  # ONE_DAY
            lookback_days = 180
            interval = "1d"
            mt5_timeframe = "D1"

        start_time = end_time - timedelta(days=lookback_days)

        if exchange == "binance":
            df = self.data_manager.fetch_binance_data(
                symbol=symbol,
                interval=interval,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:  # mt5
            df = self.data_manager.fetch_mt5_data(
                symbol=symbol,
                timeframe=mt5_timeframe,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        return self.data_manager.clean_data(df)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators

        Args:
            df: OHLCV dataframe

        Returns:
            DataFrame with indicators added
        """
        # EMAs
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ADX
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(14).mean()

        # ATR
        df["atr"] = true_range.rolling(14).mean()

        return df

    def get_regime_summary(self, regime: MultiTimeFrameRegime) -> str:
        """
        Generate human-readable summary

        Args:
            regime: MultiTimeFrameRegime object

        Returns:
            Formatted string summary
        """
        # Emoji mapping for regimes
        regime_emoji = {
            RegimeType.STRONG_BULL: "🚀",
            RegimeType.BULL: "📈",
            RegimeType.NEUTRAL: "➡️",
            RegimeType.BEAR: "📉",
            RegimeType.STRONG_BEAR: "⚠️",
        }

        emoji = regime_emoji.get(regime.consensus_regime, "❓")

        summary = [
            f"\n{'='*70}",
            f"{emoji} MULTI-TIMEFRAME REGIME ANALYSIS - {regime.asset}",
            f"{'='*70}",
            f"Timestamp: {regime.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"CONSENSUS REGIME: {regime.consensus_regime.value.upper()}",
            f"  Confidence:       {regime.consensus_confidence:.2%}",
            f"  TF Agreement:     {regime.timeframe_agreement:.2%}",
            f"  Trend Coherence:  {regime.trend_coherence:.2%}",
            f"",
            f"INDIVIDUAL TIMEFRAMES:",
        ]

        # Add individual timeframe details
        for tf_name, tf_data in [
            ("1D", regime.tf_1d),
            ("4H", regime.tf_4h),
            ("1H", regime.tf_1h),
        ]:
            if tf_data:
                summary.extend(
                    [
                        f"",
                        f"  {tf_name} Analysis:",
                        f"    Regime:     {tf_data.regime.value.upper()}",
                        f"    Confidence: {tf_data.confidence:.2%}",
                        f"    Trend:      {tf_data.trend_strength} {tf_data.trend_direction}",
                        f"    ADX:        {tf_data.adx:.1f}",
                        f"    RSI:        {tf_data.rsi:.1f}",
                        f"    EMA Diff:   {tf_data.ema_diff_pct:+.2%}",
                        f"    Volatility: {tf_data.volatility_regime.upper()}",
                        f"    Returns:    {tf_data.returns_20:+.2%} (20-period)",
                    ]
                )
            else:
                summary.append(f"  {tf_name}: No data available")

        # Add trading implications
        summary.extend(
            [
                f"",
                f"RISK ASSESSMENT:",
                f"  Risk Level:       {regime.risk_level.upper()}",
                f"  Volatility:       {regime.volatility_regime.upper()}",
                f"",
                f"TRADING IMPLICATIONS:",
                f"  Recommended Mode: {regime.recommended_mode.upper()}",
                f"  Counter-Trend:    {'✓ Allowed' if regime.allow_counter_trend else '✗ Blocked'}",
                f"  Max Positions:    {regime.suggested_max_positions}",
                f"",
                f"SCORES:",
                f"  Bullish Score:    {regime.bullish_score:.2f}/10.0",
                f"  Bearish Score:    {regime.bearish_score:.2f}/10.0",
                f"{'='*70}",
            ]
        )

        return "\n".join(summary)

    def clear_cache(self):
        """Clear the analysis cache (useful for forcing fresh analysis)"""
        self.cache.clear()
        logger.info("[MTF REGIME] Cache cleared")

    def get_cached_regime(
        self, symbol: str, exchange: str
    ) -> Optional[MultiTimeFrameRegime]:
        """
        Get cached regime if available and not stale

        Args:
            symbol: Trading symbol
            exchange: "binance" or "mt5"

        Returns:
            Cached MultiTimeFrameRegime or None
        """
        cache_key = f"{symbol}_{exchange}"
        if cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()

            if age < self.cache_duration:
                logger.debug(f"[MTF REGIME] Cache hit for {symbol} (age: {age:.0f}s)")
                return cached_result
            else:
                logger.debug(
                    f"[MTF REGIME] Cache expired for {symbol} (age: {age:.0f}s)"
                )

        return None

    def get_statistics(self) -> Dict:
        """
        Get detector statistics

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_analyses": len(self.cache),
            "cache_duration_seconds": self.cache_duration,
            "asset_type": self.asset_type,
            "timeframe_weights": {
                "1h": self.weights[TimeFrame.ONE_HOUR],
                "4h": self.weights[TimeFrame.FOUR_HOUR],
                "1d": self.weights[TimeFrame.ONE_DAY],
            },
        }
