"""
Multi-Timeframe Regime Detector - REFACTORED FOR STRICT OPTION 2
================================================================
Enforces strict trend hierarchy using Dual-Structure Logic Gates.
Focuses on 1H and 4H timeframes, guided by explicit EMA relationships.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

# Centralized Indicator Constants
FAST_EMA = 10
SLOW_EMA = 50
BASELINE_EMA = 200


@dataclass
class GovernorStatus:
    """Simplified Governor status for the Constitution Gate."""
    is_bullish: bool
    is_bearish: bool
    reasoning: str


@dataclass
class RegimeStatus:
    """Simplified aggregated regime status."""
    asset: str
    score: float
    is_bullish: bool
    is_bearish: bool
    reasoning: str
    timestamp: datetime
    consensus_regime: str # NEW: Add human-readable regime string


class MultiTimeFrameRegimeDetector:
    """
    Analyzes market regime across 1H and 4H timeframes using Dual-Structure Logic Gates.
    """

    def __init__(self, data_manager, asset_type: str = "BTC"):
        """Initialize detector"""
        self.data_manager = data_manager
        self.asset_type = asset_type.upper()

        # Governor thresholds (simplified, only relevant for _analyze_governor for Constitution Gate)
        self.governor_thresholds = {
            "min_required_bars_1d": BASELINE_EMA + 20, # At least 220 bars for 200 EMA + 20 for slope
            "ema_slope_positive": 0.0005, # Positive slope threshold for 200 EMA
        }

        # Cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        logger.info(f"[MTF REGIME] Initialized for {asset_type} (Strict Option 2)")
        logger.info(f"  EMAs: FAST={FAST_EMA}, SLOW={SLOW_EMA}, BASELINE={BASELINE_EMA}")
        logger.info(f"  Logic Gates: Constitution (4H {BASELINE_EMA} EMA), General (4H {SLOW_EMA} EMA), Captain (1H {SLOW_EMA} EMA)")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate essential EMA indicators for trend following.
        """
        if df.empty:
            return pd.DataFrame()

        df_copy = df.copy()
        df_copy[f"ema_{FAST_EMA}"] = df_copy["close"].ewm(span=FAST_EMA, adjust=False).mean()
        df_copy[f"ema_{SLOW_EMA}"] = df_copy["close"].ewm(span=SLOW_EMA, adjust=False).mean()
        df_copy[f"ema_{BASELINE_EMA}"] = df_copy["close"].ewm(span=BASELINE_EMA, adjust=False).mean()
        return df_copy

    def _fetch_data_from_csv(
        self, symbol: str, timeframe_str: str, exchange: str
    ) -> pd.DataFrame:
        """
        Fetches data from the 'data/raw/' local directory.
        Falls back to API if CSV doesn't exist or is stale.
        """
        # Determine CSV file path based on the new data/raw/ structure
        asset_prefix = "BTCUSDT" if exchange == "binance" else "XAUUSDm"
        csv_file = Path(f"data/raw/{asset_prefix}_{timeframe_str}.csv")

        # Try to read from CSV
        if csv_file.exists():
            try:
                logger.info(f"[CSV] Reading {self.asset_type} {timeframe_str} from {csv_file}")
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
        return self._fetch_data(symbol, timeframe_str, exchange)

    def _fetch_data(
        self, symbol: str, timeframe_str: str, exchange: str
    ) -> pd.DataFrame:
        """
        Fetch historical data for the specified timeframe from API.
        """
        end_time = datetime.now(timezone.utc)

        # Determine lookback based on timeframe string
        if timeframe_str == "1h":
            lookback_days = 30
        elif timeframe_str == "4h":
            lookback_days = 60
        elif timeframe_str == "1d":
            lookback_days = 180
        else:
            logger.error(f"Unsupported timeframe_str: {timeframe_str}")
            return pd.DataFrame() # Return empty for unsupported

        start_time = end_time - timedelta(days=lookback_days)

        if exchange == "binance":
            df = self.data_manager.fetch_binance_data(
                symbol=symbol,
                interval=timeframe_str,
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:  # mt5
            df = self.data_manager.fetch_mt5_data(
                symbol=symbol,
                timeframe=timeframe_str.upper().replace('H', 'H'), # e.g., '1h' -> 'H1'
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        return self.data_manager.clean_data(df)


    def _analyze_governor(self, symbol: str, exchange: str) -> GovernorStatus:
        """
        Constitution Gate: Analyzes the Daily 200 EMA for macro trend.
        """
        try:
            df_daily = self._fetch_data_from_csv(symbol, "1d", exchange)

            min_bars = self.governor_thresholds["min_required_bars_1d"]

            if len(df_daily) < min_bars:
                logger.warning(
                    f"[CONSTITUTION] Insufficient daily data: {len(df_daily)} bars (need {min_bars}+). "
                    "Cannot establish macro trend. Defaulting to neutral."
                )
                return GovernorStatus(
                    is_bullish=False, is_bearish=False, reasoning="Insufficient 1D data"
                )

            df_daily = self._calculate_indicators(df_daily)
            latest = df_daily.iloc[-1]
            current_price = latest["close"]
            ema_200 = latest[f"ema_{BASELINE_EMA}"]

            if pd.isna(ema_200):
                 logger.warning("[CONSTITUTION] 1D 200 EMA is NaN. Defaulting to neutral.")
                 return GovernorStatus(
                    is_bullish=False, is_bearish=False, reasoning="1D 200 EMA is NaN"
                 )

            # Check slope of 200 EMA (over last 20 bars)
            if len(df_daily) < BASELINE_EMA + 20: # Ensure enough data for slope
                ema_slope = 0.0 # Cannot calculate slope
            else:
                ema_200_series = df_daily[f"ema_{BASELINE_EMA}"]
                ema_slope = (ema_200_series.iloc[-1] - ema_200_series.iloc[-20]) / ema_200_series.iloc[-20]


            is_bullish = (current_price > ema_200) and (ema_slope > self.governor_thresholds["ema_slope_positive"])
            is_bearish = (current_price < ema_200) and (ema_slope < -self.governor_thresholds["ema_slope_positive"])

            reasoning = (
                f"Price {'above' if is_bullish else 'below' if is_bearish else 'near'} 200 EMA "
                f"with {'positive' if ema_slope > 0 else 'negative' if ema_slope < 0 else 'flat'} slope."
            )

            return GovernorStatus(
                is_bullish=is_bullish, is_bearish=is_bearish, reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"[CONSTITUTION] Error analyzing 1D Governor: {e}", exc_info=True)
            return GovernorStatus(
                is_bullish=False, is_bearish=False, reasoning=f"Error: {str(e)}"
            )

    def get_aggregated_regime_score(
        self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, governor_status: GovernorStatus, asset_type: str
    ) -> RegimeStatus:
        """
        Implements the Dual-Structure Logic Gates to determine the aggregated regime score.

        Gates:
        1. Constitution (4H BASELINE_EMA): Determines macro trend territory.
        2. General (4H SLOW_EMA): Defines the intermediate trend direction.
        3. Captain (1H SLOW_EMA): Gives permission for execution based on short-term trend.

        Returns:
            RegimeStatus: Object indicating bullish/bearish score and reasoning.
        """
        score = 0.0
        reasons = []

        # Ensure dataframes are not empty
        if df_1h.empty or df_4h.empty:
            reasons.append("Insufficient 1H or 4H data.")
            return RegimeStatus(
                asset=asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=", ".join(reasons), timestamp=datetime.now(timezone.utc)
            )

        # Calculate indicators for both timeframes
        df_1h_with_ema = self._calculate_indicators(df_1h)
        df_4h_with_ema = self._calculate_indicators(df_4h)

        if df_1h_with_ema.empty or df_4h_with_ema.empty:
            reasons.append("Failed to calculate EMAs due to insufficient data.")
            return RegimeStatus(
                asset=asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=", ".join(reasons), timestamp=datetime.now(timezone.utc)
            )

        latest_1h = df_1h_with_ema.iloc[-1]
        latest_4h = df_4h_with_ema.iloc[-1]

        # Get current prices and EMAs
        price_1h = latest_1h["close"]
        ema_1h_slow = latest_1h[f"ema_{SLOW_EMA}"]

        price_4h = latest_4h["close"]
        ema_4h_slow = latest_4h[f"ema_{SLOW_EMA}"]
        ema_4h_baseline = latest_4h[f"ema_{BASELINE_EMA}"]

        # --- Gate 1: Constitution (4H BASELINE_EMA) ---
        # Macro trend territory from the Governor
        if governor_status.is_bullish:
            score += 1.0 # Strong bullish macro environment
            reasons.append("Constitution: 1D 200 EMA is bullish.")
        elif governor_status.is_bearish:
            score -= 1.0 # Strong bearish macro environment
            reasons.append("Constitution: 1D 200 EMA is bearish.")
        else:
            reasons.append("Constitution: 1D 200 EMA is neutral/unclear.")

        # --- Gate 2: General (4H SLOW_EMA) ---
        # Intermediate trend direction
        if price_4h > ema_4h_baseline: # Only consider General if above Constitution line
            if price_4h > ema_4h_slow:
                score += 1.0
                reasons.append("General: 4H price > 4H 50 EMA.")
            elif price_4h < ema_4h_slow:
                score -= 0.5 # Slight negative if price below 50 EMA but still above 200
                reasons.append("General: 4H price < 4H 50 EMA (but above 200).")
        else: # Below Constitution line, General can only be neutral or bearish
            if price_4h < ema_4h_slow:
                score -= 1.0
                reasons.append("General: 4H price < 4H 50 EMA (below 200).")
            else:
                score -= 0.5 # Price above 50 EMA but below 200
                reasons.append("General: 4H price > 4H 50 EMA (but below 200).")

        # --- Gate 3: Captain (1H SLOW_EMA) ---
        # Short-term execution permission
        if price_1h > ema_1h_slow:
            score += 0.5
            reasons.append("Captain: 1H price > 1H 50 EMA.")
        elif price_1h < ema_1h_slow:
            score -= 0.5
            reasons.append("Captain: 1H price < 1H 50 EMA.")
        else:
            reasons.append("Captain: 1H price near 1H 50 EMA (neutral).")

        # Final decision based on score
        is_bullish = score > 0.5 # Threshold for bullish bias
        is_bearish = score < -0.5 # Threshold for bearish bias

        if is_bullish:
            consensus_regime = "BULLISH"
        elif is_bearish:
            consensus_regime = "BEARISH"
        else:
            consensus_regime = "NEUTRAL"

        final_reasoning = f"Aggregated score: {score:.2f}. " + ", ".join(reasons)

        return RegimeStatus(
            asset=asset_type,
            score=score,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            reasoning=final_reasoning,
            timestamp=datetime.now(timezone.utc),
            consensus_regime=consensus_regime
        )

    def analyze_regime(
        self,
        symbol: str,
        exchange: str = "binance",
        force_refresh: bool = False
    ) -> RegimeStatus:
        """
        Analyzes market regime across 1H, 4H, and 1D timeframes using Dual-Structure Logic Gates.
        Returns a simplified RegimeStatus object.
        """
        cache_key = f"{symbol}_{exchange}"
        if not force_refresh and cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                logger.debug(f"[MTF] Using cached result for {symbol}")
                return cached_result

        logger.info(f"\n{'='*70}")
        logger.info(f"[MTF REGIME] Analyzing {self.asset_type} ({symbol}) - Strict Option 2")
        logger.info(f"{'='*70}")

        # --- STEP 1: Constitution Gate (Daily 200 EMA) ---
        governor_status = self._analyze_governor(symbol, exchange)
        logger.info(f"[CONSTITUTION] Status: {'Bullish' if governor_status.is_bullish else 'Bearish' if governor_status.is_bearish else 'Neutral'}. Reason: {governor_status.reasoning}")

        # --- STEP 2: Fetch 1H and 4H data ---
        df_1h = pd.DataFrame()
        df_4h = pd.DataFrame()
        
        try:
            df_1h = self._fetch_data_from_csv(symbol, "1h", exchange)
            if df_1h.empty or len(df_1h) < SLOW_EMA + 1:
                raise ValueError("Insufficient 1H data for EMA calculation.")
        except Exception as e:
            logger.error(f"[MTF] Failed to get 1H data: {e}", exc_info=True)
            return RegimeStatus(asset=self.asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=f"Insufficient 1H data: {str(e)}", timestamp=datetime.now(timezone.utc))

        try:
            df_4h = self._fetch_data_from_csv(symbol, "4h", exchange)
            if df_4h.empty or len(df_4h) < BASELINE_EMA + 1:
                raise ValueError("Insufficient 4H data for EMA calculation.")
        except Exception as e:
            logger.error(f"[MTF] Failed to get 4H data: {e}", exc_info=True)
            return RegimeStatus(asset=self.asset_type, score=0.0, is_bullish=False, is_bearish=False, reasoning=f"Insufficient 4H data: {str(e)}", timestamp=datetime.now(timezone.utc))

        # --- STEP 3: Get Aggregated Regime Score ---
        regime_status = self.get_aggregated_regime_score(df_1h, df_4h, governor_status, self.asset_type)

        logger.info(f"\n[FINAL REGIME] Score: {regime_status.score:.2f}")
        logger.info(f"  Bullish: {regime_status.is_bullish}, Bearish: {regime_status.is_bearish}")
        logger.info(f"  Reasoning: {regime_status.reasoning}")
        logger.info(f"{'='*70}\n")

        self.cache[cache_key] = (regime_status, datetime.now(timezone.utc))
        return regime_status
