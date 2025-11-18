# src/indicators/ema_crossover.py
"""
EMA Crossover Indicator for Trading Bot
Implements EMA 50/200 crossover strategy with filters
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EMACrossoverIndicator:
    """
    Exponential Moving Average Crossover Strategy
    
    Signals:
    - Golden Cross (EMA 50 > EMA 200): BUY signal
    - Death Cross (EMA 50 < EMA 200): SELL signal
    - Trend following: Stay in position during strong trends
    """
    
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        min_distance_pct: float = 0.3,
        use_price_confirmation: bool = True,
        use_volume_filter: bool = False,
        volume_multiplier: float = 1.2,
    ):
        """
        Args:
            fast_period: Period for fast EMA (default: 50)
            slow_period: Period for slow EMA (default: 200)
            min_distance_pct: Minimum % distance between EMAs for signal (default: 0.3%)
            use_price_confirmation: Require price to confirm signal (default: True)
            use_volume_filter: Filter by volume (default: False)
            volume_multiplier: Volume must be X times average (default: 1.2x)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_distance_pct = min_distance_pct
        self.use_price_confirmation = use_price_confirmation
        self.use_volume_filter = use_volume_filter
        self.volume_multiplier = volume_multiplier
        
        self.name = f"EMA_{fast_period}_{slow_period}"
        
        # State tracking for crossover detection
        self.prev_fast_ema = None
        self.prev_slow_ema = None
        self.prev_signal = 0
        
        logger.info(f"EMA Crossover Indicator initialized: {self.name}")
        logger.info(f"  Min distance: {min_distance_pct}%")
        logger.info(f"  Price confirmation: {use_price_confirmation}")
        logger.info(f"  Volume filter: {use_volume_filter}")
    
    def calculate_ema(
        self, 
        data: pd.DataFrame, 
        period: int, 
        column: str = 'close'
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        EMA = Price(t) × k + EMA(y) × (1 - k)
        where k = 2 / (N + 1)
        """
        return data[column].ewm(span=period, adjust=False).mean()
    
    def calculate_signal(
        self, 
        data: pd.DataFrame,
        asset: str = "UNKNOWN"
    ) -> Tuple[int, Dict]:
        """
        Calculate EMA crossover signal with filters
        
        Args:
            data: DataFrame with OHLCV data (must have 'close' column)
            asset: Asset name for logging
        
        Returns:
            signal (int): 1 (BUY), -1 (SELL), 0 (HOLD)
            metadata (dict): Signal details for logging/analysis
        """
        try:
            # Validate data
            if len(data) < self.slow_period:
                logger.warning(
                    f"{asset}: Insufficient data - need {self.slow_period} candles, got {len(data)}"
                )
                return 0, {"error": f"Need at least {self.slow_period} candles"}
            
            # Calculate EMAs
            fast_ema = self.calculate_ema(data, self.fast_period)
            slow_ema = self.calculate_ema(data, self.slow_period)
            
            # Get current values
            current_fast = fast_ema.iloc[-1]
            current_slow = slow_ema.iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Initialize previous values on first run
            if self.prev_fast_ema is None:
                if len(fast_ema) > 1:
                    self.prev_fast_ema = fast_ema.iloc[-2]
                    self.prev_slow_ema = slow_ema.iloc[-2]
                else:
                    self.prev_fast_ema = current_fast
                    self.prev_slow_ema = current_slow
            
            # Calculate metrics
            ema_distance = current_fast - current_slow
            ema_distance_pct = (ema_distance / current_slow) * 100 if current_slow != 0 else 0
            price_to_fast_dist = current_price - current_fast
            price_to_fast_pct = (price_to_fast_dist / current_fast) * 100 if current_fast != 0 else 0
            
            # Detect crossovers
            golden_cross = (
                current_fast > current_slow and 
                self.prev_fast_ema <= self.prev_slow_ema
            )
            
            death_cross = (
                current_fast < current_slow and 
                self.prev_fast_ema >= self.prev_slow_ema
            )
            
            # Initialize signal
            signal = 0
            signal_reason = "HOLD"
            confidence = 0.0
            
            # === SIGNAL GENERATION ===
            
            # 1. GOLDEN CROSS - BUY Signal
            if golden_cross:
                signal = 1
                signal_reason = "GOLDEN_CROSS"
                confidence = min(abs(ema_distance_pct) / 2.0, 1.0)  # 0-1 based on distance
                
                # Apply filters
                if self.use_price_confirmation and current_price < current_fast:
                    signal = 0
                    signal_reason = "GOLDEN_CROSS_NO_PRICE_CONFIRM"
                    confidence = 0.3
                
                if abs(ema_distance_pct) < self.min_distance_pct:
                    signal = 0
                    signal_reason = "GOLDEN_CROSS_TOO_CLOSE"
                    confidence = 0.2
            
            # 2. DEATH CROSS - SELL Signal
            elif death_cross:
                signal = -1
                signal_reason = "DEATH_CROSS"
                confidence = min(abs(ema_distance_pct) / 2.0, 1.0)
            
            # 3. TREND CONTINUATION - Stay Long
            elif current_fast > current_slow and current_price > current_fast:
                # Check trend strength
                if abs(ema_distance_pct) >= self.min_distance_pct:
                    signal = 1
                    signal_reason = "UPTREND_STRONG"
                    confidence = min(abs(ema_distance_pct) / 2.0, 1.0)
                else:
                    signal = 0
                    signal_reason = "UPTREND_WEAK"
                    confidence = 0.5
            
            # 4. DOWNTREND - Exit/Avoid
            elif current_fast < current_slow:
                if current_price < current_fast:
                    signal = -1
                    signal_reason = "DOWNTREND_STRONG"
                    confidence = 0.8
                else:
                    signal = 0
                    signal_reason = "DOWNTREND"
                    confidence = 0.3
            
            # 5. SIDEWAYS - No signal
            else:
                signal = 0
                signal_reason = "SIDEWAYS"
                confidence = 0.0
            
            # === VOLUME FILTER ===
            volume_ok = True
            if self.use_volume_filter and 'volume' in data.columns:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                
                if current_volume < avg_volume * self.volume_multiplier:
                    if signal == 1:  # Only filter BUY signals
                        signal = 0
                        signal_reason = f"{signal_reason}_LOW_VOLUME"
                        confidence *= 0.5
                        volume_ok = False
            
            # Update state for next iteration
            self.prev_fast_ema = current_fast
            self.prev_slow_ema = current_slow
            self.prev_signal = signal
            
            # Prepare metadata
            metadata = {
                "ema_fast": float(current_fast),
                "ema_slow": float(current_slow),
                "price": float(current_price),
                "ema_distance": float(ema_distance),
                "ema_distance_pct": float(ema_distance_pct),
                "price_to_fast_pct": float(price_to_fast_pct),
                "signal_reason": signal_reason,
                "confidence": float(confidence),
                "golden_cross": golden_cross,
                "death_cross": death_cross,
                "volume_ok": volume_ok,
            }
            
            # Log significant signals
            if signal != 0 or golden_cross or death_cross:
                logger.info(
                    f"[{asset}] EMA Signal: {signal:+d} ({signal_reason})\n"
                    f"  Price: ${current_price:,.2f}\n"
                    f"  EMA {self.fast_period}: ${current_fast:,.2f}\n"
                    f"  EMA {self.slow_period}: ${current_slow:,.2f}\n"
                    f"  Distance: {ema_distance_pct:+.2f}%\n"
                    f"  Confidence: {confidence:.1%}"
                )
            
            return signal, metadata
        
        except Exception as e:
            logger.error(f"Error calculating EMA signal for {asset}: {e}", exc_info=True)
            return 0, {"error": str(e)}
    
    def get_indicator_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA indicators to dataframe for analysis/plotting
        
        Returns:
            DataFrame with EMA columns added
        """
        df = data.copy()
        
        df[f'ema_{self.fast_period}'] = self.calculate_ema(data, self.fast_period)
        df[f'ema_{self.slow_period}'] = self.calculate_ema(data, self.slow_period)
        df['ema_distance'] = df[f'ema_{self.fast_period}'] - df[f'ema_{self.slow_period}']
        df['ema_distance_pct'] = (df['ema_distance'] / df[f'ema_{self.slow_period}']) * 100
        
        # Add trend classification
        df['ema_trend'] = np.where(
            df[f'ema_{self.fast_period}'] > df[f'ema_{self.slow_period}'],
            'BULL',
            'BEAR'
        )
        
        return df
    
    def calculate_dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        entry_price: float,
        position_side: str,
        buffer_pct: float = 0.02
    ) -> Optional[float]:
        """
        Calculate dynamic stop loss based on EMA
        
        Args:
            data: OHLCV data
            entry_price: Position entry price
            position_side: 'long' or 'short'
            buffer_pct: Buffer below/above EMA (default: 2%)
        
        Returns:
            Stop loss price or None
        """
        try:
            fast_ema = self.calculate_ema(data, self.fast_period)
            current_ema = fast_ema.iloc[-1]
            
            if position_side == 'long':
                # Stop loss below EMA 50
                stop_loss = current_ema * (1 - buffer_pct)
                # Don't move stop loss down
                return max(stop_loss, entry_price * 0.97)  # At least 3% below entry
            
            else:  # short
                # Stop loss above EMA 50
                stop_loss = current_ema * (1 + buffer_pct)
                # Don't move stop loss up
                return min(stop_loss, entry_price * 1.03)  # At least 3% above entry
        
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=300, freq='1H')
    np.random.seed(42)
    
    # Simulate price movement with trend
    price = 50000
    prices = [price]
    for i in range(299):
        change = np.random.normal(0.001, 0.02)  # Slight upward bias
        price = price * (1 + change)
        prices.append(price)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 300)
    })
    
    # Initialize indicator
    ema_indicator = EMACrossoverIndicator(
        fast_period=50,
        slow_period=200,
        min_distance_pct=0.3,
        use_price_confirmation=True
    )
    
    # Calculate signal
    signal, metadata = ema_indicator.calculate_signal(data, asset="BTC")
    
    print("\n" + "="*60)
    print("EMA CROSSOVER SIGNAL")
    print("="*60)
    print(f"Signal: {signal:+d}")
    print(f"Reason: {metadata['signal_reason']}")
    print(f"Confidence: {metadata['confidence']:.1%}")
    print(f"\nPrice: ${metadata['price']:,.2f}")
    print(f"EMA 50: ${metadata['ema_fast']:,.2f}")
    print(f"EMA 200: ${metadata['ema_slow']:,.2f}")
    print(f"Distance: {metadata['ema_distance_pct']:+.2f}%")
    print("="*60)
    
    # Get full dataframe with indicators
    df_with_emas = ema_indicator.get_indicator_dataframe(data)
    print("\nLast 5 rows with EMA indicators:")
    print(df_with_emas[['close', 'ema_50', 'ema_200', 'ema_trend']].tail())