
"""
Testing Script for AI Layer
Usage: python test_ai_layer.py
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.ai.analyst import DynamicAnalyst
from src.ai.sniper import OHLCSniper
from src.ai.hybrid_validator import HybridSignalValidator


def load_trained_model():
    """Load the trained Sniper model and mappings"""
    models_dir = Path("models/ai")
    
    # Load pattern mapping
    mapping_path = models_dir / "pattern_mapping.pkl"
    with open(mapping_path, 'rb') as f:
        pattern_map = pickle.load(f)
    
    # Load config
    config_path = models_dir / "training_config.pkl"
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Initialize and load model
    sniper = OHLCSniper(
        input_shape=(15, 4),
        num_classes=config['num_classes']
    )
    
    model_path = models_dir / "sniper.weights.h5"
    sniper.load_model(str(model_path))
    
    logger.info(f"✓ Model loaded from {model_path}")
    logger.info(f"✓ Validation accuracy: {config['validation_accuracy']:.2%}")
    logger.info(f"✓ Patterns: {len(pattern_map)}")
    
    return sniper, pattern_map, config


def test_analyst():
    """Test the Analyst (S/R detection)"""
    logger.info("\n" + "="*70)
    logger.info("TESTING ANALYST (Support/Resistance Detection)")
    logger.info("="*70)
    
    # Create synthetic price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    
    # Create OHLC
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices
    })
    
    # Extract pivots (simple method)
    highs = df['high'].values
    lows = df['low'].values
    pivots = []
    
    for i in range(5, len(df)-5):
        if highs[i] == max(highs[i-5:i+6]):
            pivots.append(highs[i])
        if lows[i] == min(lows[i-5:i+6]):
            pivots.append(lows[i])
    
    pivots = np.array(pivots)
    
    # Test Analyst
    analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
    levels = analyst.get_support_resistance_levels(
        pivot_points=pivots,
        highs=df['high'].values,
        lows=df['low'].values,
        closes=df['close'].values,
        n_levels=5
    )
    
    logger.info(f"✓ Identified {len(levels)} S/R levels:")
    for i, level in enumerate(levels, 1):
        logger.info(f"  Level {i}: ${level:.2f}")
    
    current_price = df['close'].iloc[-1]
    logger.info(f"\n✓ Current price: ${current_price:.2f}")
    
    # Check distance to nearest level
    distances = [abs(current_price - l) / current_price for l in levels]
    nearest_idx = np.argmin(distances)
    nearest_level = levels[nearest_idx]
    nearest_dist = distances[nearest_idx]
    
    logger.info(f"✓ Nearest level: ${nearest_level:.2f} ({nearest_dist:.2%} away)")
    logger.info(f"✓ Within 0.5% threshold: {'YES' if nearest_dist < 0.005 else 'NO'}")
    
    return analyst


def test_sniper(sniper, pattern_map):
    """Test the Sniper (Pattern recognition)"""
    logger.info("\n" + "="*70)
    logger.info("TESTING SNIPER (Pattern Recognition)")
    logger.info("="*70)
    
    # Create sample patterns manually
    test_cases = [
        {
            'name': 'Bullish Engulfing (manual)',
            'ohlc': np.array([
                [100, 102, 99, 99.5],   # Small red candle
                [99.5, 103, 99, 102]    # Large green engulfing
            ])
        },
        {
            'name': 'Hammer (manual)',
            'ohlc': np.array([
                [100, 100.5, 95, 99.8]  # Long lower wick, close near high
            ])
        }
    ]
    
    reverse_map = {v: k for k, v in pattern_map.items()}
    
    # Test each case
    for test in test_cases:
        logger.info(f"\nTesting: {test['name']}")
        
        # Pad to 15 candles if needed
        ohlc = test['ohlc']
        if len(ohlc) < 15:
            padding = np.tile(ohlc[0], (15 - len(ohlc), 1))
            ohlc = np.vstack([padding, ohlc])
        
        # Predict
        pattern_id, confidence = sniper.predict(ohlc)
        pattern_name = reverse_map.get(pattern_id, 'Noise')
        
        logger.info(f"  Detected: {pattern_name}")
        logger.info(f"  Confidence: {confidence:.2%}")
        logger.info(f"  Pattern ID: {pattern_id}")
    
    logger.info("\n✓ Sniper testing complete")
    return True


def test_integration():
    """Test the full integration"""
    logger.info("\n" + "="*70)
    logger.info("TESTING FULL INTEGRATION")
    logger.info("="*70)
    
    # Load trained model
    sniper, pattern_map, config = load_trained_model()
    
    # Initialize components
    analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
    
    validator = HybridSignalValidator(
        analyst=analyst,
        sniper=sniper,
        pattern_id_map=pattern_map,
        sr_threshold_pct=0.005,
        pattern_confidence_min=0.70,
        use_ai_validation=True
    )
    
    # Create test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(prices))
    }, index=pd.date_range('2024-01-01', periods=len(prices), freq='1h'))
    
    # Test validation
    test_signals = [
        {'signal': 1, 'name': 'BUY'},
        {'signal': -1, 'name': 'SELL'},
        {'signal': 0, 'name': 'HOLD'}
    ]
    
    for test in test_signals:
        signal = test['signal']
        signal_details = {
            'reasoning': 'test_signal',
            'signal_quality': 0.75
        }
        
        logger.info(f"\nTesting {test['name']} signal...")
        
        final_signal, enhanced_details = validator.validate_signal(
            signal=signal,
            signal_details=signal_details,
            df=df
        )
        
        logger.info(f"  Original: {signal}")
        logger.info(f"  Final: {final_signal}")
        logger.info(f"  Validation: {enhanced_details.get('ai_validation', 'N/A')}")
        
        if 'ai_sr_check' in enhanced_details:
            sr = enhanced_details['ai_sr_check']
            logger.info(f"  Near S/R: {sr.get('near_level', False)}")
        
        if 'ai_pattern_check' in enhanced_details:
            pat = enhanced_details['ai_pattern_check']
            logger.info(f"  Pattern: {pat.get('pattern_name', 'N/A')} ({pat.get('confidence', 0):.0%})")
    
    logger.info("\n✓ Integration testing complete")
    
    # Print statistics
    stats = validator.get_statistics()
    logger.info("\nValidator Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("AI LAYER TESTING SUITE")
    logger.info("="*70)
    
    try:
        # Test 1: Analyst
        analyst = test_analyst()
        
        # Test 2: Load and test Sniper
        sniper, pattern_map, config = load_trained_model()
        test_sniper(sniper, pattern_map)
        
        # Test 3: Full integration
        test_integration()
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*70)
        logger.info("\nThe AI layer is ready for integration.")
        logger.info("Next: Update your signal aggregator (see integration guide)")
        
    except FileNotFoundError as e:
        logger.error("\n❌ ERROR: Trained model not found!")
        logger.error("Please run 'python train_ai_layer.py' first")
        logger.error(f"Details: {e}")
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()