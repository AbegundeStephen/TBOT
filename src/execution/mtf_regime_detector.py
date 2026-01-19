"""
Multi-Timeframe Regime Detector
=================================
Analyzes market regime across 1H, 4H, and 1D timeframes for robust regime detection.
Provides weighted consensus and stores detailed analysis in database.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

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
    trend_strength: str  # "strong", "moderate", "weak"
    trend_direction: str  # "up", "down", "sideways"

    # Momentum metrics
    rsi: float
    macd_histogram: float
    momentum_aligned: bool

    # Volatility metrics
    atr_pct: float
    volatility_regime: str  # "high", "normal", "low"

    # Price action
    returns_20: float  # 20-period return
    returns_50: float  # 50-period return (if available)
    higher_highs: bool
    higher_lows: bool

    # Supporting data
    timestamp: datetime
    weight: float = 1.0  # For multi-timeframe weighting


@dataclass
class MultiTimeFrameRegime:
    """Aggregated regime across multiple timeframes"""

    asset: str
    timestamp: datetime

    # Individual timeframe results
    tf_1h: Optional[TimeFrameAnalysis]
    tf_4h: Optional[TimeFrameAnalysis]
    tf_1d: Optional[TimeFrameAnalysis]

    # Consensus regime
    consensus_regime: RegimeType
    consensus_confidence: float

    # Alignment metrics
    timeframe_agreement: float  # 0-1, how aligned timeframes are
    trend_coherence: float  # 0-1, how coherent trend is across TFs

    # Risk assessment
    risk_level: str  # "low", "medium", "high"
    volatility_regime: str

    # Trading implications
    recommended_mode: str  # "aggressive", "balanced", "conservative", "scalper"
    allow_counter_trend: bool
    suggested_max_positions: int

    # Raw scores for decision making
    bullish_score: float  # 0-10
    bearish_score: float  # 0-10

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
            "bullish_score": self.bullish_score,
            "bearish_score": self.bearish_score,
        }

        # Add individual timeframe data
        for tf_name, tf_data in [
            ("1h", self.tf_1h),
            ("4h", self.tf_4h),
            ("1d", self.tf_1d),
        ]:
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
    Analyzes market regime across multiple timeframes for robust detection.

    Weighting:
    - 1D: 50% (primary trend)
    - 4H: 30% (intermediate trend)
    - 1H: 20% (short-term noise filter)
    """

    def __init__(self, data_manager, asset_type: str = "BTC"):
        """
        Initialize detector

        Args:
            data_manager: DataManager instance for fetching historical data
            asset_type: "BTC" or "GOLD"
        """
        self.data_manager = data_manager
        self.asset_type = asset_type.upper()

        # Timeframe weights (must sum to 1.0)
        self.weights = {
            TimeFrame.ONE_DAY: 0.50,  # Primary trend
            TimeFrame.FOUR_HOUR: 0.30,  # Intermediate
            TimeFrame.ONE_HOUR: 0.20,  # Short-term
        }

        # Asset-specific thresholds
        if self.asset_type == "BTC":
            self.thresholds = {
                "ema_bull": 0.15,  # EMA diff for bullish
                "ema_bear": -0.15,  # EMA diff for bearish
                "adx_strong": 25,  # Strong trend
                "adx_weak": 20,  # Weak trend
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

        # Cache for recent analyses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        logger.info(f"[MTF REGIME] Initialized for {asset_type}")
        logger.info(
            f"  Weights: 1D={self.weights[TimeFrame.ONE_DAY]:.0%}, "
            f"4H={self.weights[TimeFrame.FOUR_HOUR]:.0%}, "
            f"1H={self.weights[TimeFrame.ONE_HOUR]:.0%}"
        )

    def analyze_regime(
        self, symbol: str, exchange: str = "binance", force_refresh: bool = False
    ) -> MultiTimeFrameRegime:
        """
        Analyze regime across all timeframes

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "XAUUSD")
            exchange: "binance" or "mt5"
            force_refresh: Skip cache and force new analysis

        Returns:
            MultiTimeFrameRegime object with complete analysis
        """
        # Check cache
        cache_key = f"{symbol}_{exchange}"
        if not force_refresh and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                logger.debug(f"[MTF REGIME] Using cached result for {symbol}")
                return cached_result

        logger.info(f"\n{'='*70}")
        logger.info(f"[MTF REGIME] Analyzing {self.asset_type} ({symbol})")
        logger.info(f"{'='*70}")

        # Analyze each timeframe
        tf_results = {}

        for timeframe in [TimeFrame.ONE_HOUR, TimeFrame.FOUR_HOUR, TimeFrame.ONE_DAY]:
            try:
                analysis = self._analyze_timeframe(symbol, timeframe, exchange)
                tf_results[timeframe] = analysis

                logger.info(f"\n[{timeframe.value.upper()}] Analysis:")
                logger.info(f"  Regime:     {analysis.regime.value.upper()}")
                logger.info(f"  Confidence: {analysis.confidence:.2%}")
                logger.info(
                    f"  Trend:      {analysis.trend_strength} {analysis.trend_direction}"
                )
                logger.info(f"  ADX:        {analysis.adx:.1f}")
                logger.info(f"  RSI:        {analysis.rsi:.1f}")
                logger.info(f"  EMA Diff:   {analysis.ema_diff_pct:+.2%}")

            except Exception as e:
                logger.error(f"[MTF REGIME] Error analyzing {timeframe.value}: {e}")
                tf_results[timeframe] = None

        # Aggregate results
        result = self._aggregate_timeframes(
            asset=self.asset_type,
            tf_1h=tf_results.get(TimeFrame.ONE_HOUR),
            tf_4h=tf_results.get(TimeFrame.FOUR_HOUR),
            tf_1d=tf_results.get(TimeFrame.ONE_DAY),
        )

        # Log consensus
        logger.info(f"\n{'='*70}")
        logger.info(f"[CONSENSUS] {result.consensus_regime.value.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"  Confidence:       {result.consensus_confidence:.2%}")
        logger.info(f"  TF Agreement:     {result.timeframe_agreement:.2%}")
        logger.info(f"  Trend Coherence:  {result.trend_coherence:.2%}")
        logger.info(f"  Risk Level:       {result.risk_level.upper()}")
        logger.info(f"  Recommended Mode: {result.recommended_mode.upper()}")
        logger.info(
            f"  Counter-Trend:    {'✓ Allowed' if result.allow_counter_trend else '✗ Blocked'}"
        )
        logger.info(f"{'='*70}\n")

        # Cache result
        self.cache[cache_key] = (result, datetime.now())

        return result

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
    ) -> MultiTimeFrameRegime:
        """
        Aggregate timeframe analyses into consensus regime

        Args:
            asset: Asset name
            tf_1h: 1H analysis
            tf_4h: 4H analysis
            tf_1d: 1D analysis

        Returns:
            MultiTimeFrameRegime object
        """
        # Calculate weighted scores
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0

        timeframes = [
            (TimeFrame.ONE_HOUR, tf_1h),
            (TimeFrame.FOUR_HOUR, tf_4h),
            (TimeFrame.ONE_DAY, tf_1d),
        ]

        valid_count = 0
        for tf_enum, tf_data in timeframes:
            if tf_data:
                weight = self.weights[tf_enum]

                # Convert regime to score
                if tf_data.regime == RegimeType.STRONG_BULL:
                    bullish_score += 10 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.BULL:
                    bullish_score += 7 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.STRONG_BEAR:
                    bearish_score += 10 * weight * tf_data.confidence
                elif tf_data.regime == RegimeType.BEAR:
                    bearish_score += 7 * weight * tf_data.confidence

                total_weight += weight
                valid_count += 1

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

        # Calculate agreement (how aligned are the timeframes?)
        regimes = [tf.regime for tf in [tf_1h, tf_4h, tf_1d] if tf]
        if len(regimes) >= 2:
            # Count how many agree with consensus
            agreement = sum(1 for r in regimes if r == consensus) / len(regimes)
        else:
            agreement = 1.0 if len(regimes) == 1 else 0.5

        # Calculate trend coherence (all pointing same direction?)
        directions = [tf.trend_direction for tf in [tf_1h, tf_4h, tf_1d] if tf]
        if directions:
            most_common = max(set(directions), key=directions.count)
            coherence = directions.count(most_common) / len(directions)
        else:
            coherence = 0.5

        # Determine volatility regime (use 1D if available, else 4H)
        volatility = (
            (tf_1d or tf_4h or tf_1h).volatility_regime
            if any([tf_1d, tf_4h, tf_1h])
            else "normal"
        )

        # Risk assessment
        if volatility == "high" or agreement < 0.5:
            risk_level = "high"
        elif volatility == "low" and agreement > 0.8:
            risk_level = "low"
        else:
            risk_level = "medium"

        # Trading implications
        recommended_mode, allow_counter, max_positions = self._determine_trading_mode(
            consensus=consensus,
            confidence=consensus_conf,
            agreement=agreement,
            volatility=volatility,
            risk_level=risk_level,
        )

        return MultiTimeFrameRegime(
            asset=asset,
            timestamp=datetime.now(),
            tf_1h=tf_1h,
            tf_4h=tf_4h,
            tf_1d=tf_1d,
            consensus_regime=consensus,
            consensus_confidence=consensus_conf,
            timeframe_agreement=agreement,
            trend_coherence=coherence,
            risk_level=risk_level,
            volatility_regime=volatility,
            recommended_mode=recommended_mode,
            allow_counter_trend=allow_counter,
            suggested_max_positions=max_positions,
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
    ) -> Tuple[str, bool, int]:
        """
        Determine optimal trading mode based on regime

        Returns:
            (recommended_mode, allow_counter_trend, max_positions)
        """
        # Base decision on consensus and confidence
        if consensus in [RegimeType.STRONG_BULL, RegimeType.STRONG_BEAR]:
            if confidence > 0.80 and agreement > 0.75:
                mode = "aggressive"
                allow_counter = False
                max_pos = 3
            else:
                mode = "balanced"
                allow_counter = False
                max_pos = 2

        elif consensus in [RegimeType.BULL, RegimeType.BEAR]:
            if confidence > 0.70:
                mode = "balanced"
                allow_counter = confidence < 0.75
                max_pos = 2
            else:
                mode = "conservative"
                allow_counter = True
                max_pos = 1

        else:  # NEUTRAL
            if volatility == "low" and agreement > 0.6:
                mode = "scalper"
                allow_counter = True
                max_pos = 2
            else:
                mode = "conservative"
                allow_counter = True
                max_pos = 1

        # Adjust for risk
        if risk_level == "high":
            if mode == "aggressive":
                mode = "balanced"
            max_pos = max(1, max_pos - 1)

        return mode, allow_counter, max_pos

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
