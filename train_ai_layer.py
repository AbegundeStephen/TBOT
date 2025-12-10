"""
Training Script for AI Layer
Usage: python train_ai_layer.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
import logging
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import AI components
from src.ai.pattern_miner import PatternMiner
from src.ai.sniper import OHLCSniper


def train_sniper_model(
    num_samples=20000,
    epochs=100,
    batch_size=64,
    validation_split=0.2
):
    """
    Complete training pipeline for the Sniper model
    
    Steps:
    1. Mine patterns from synthetic data
    2. Split into train/validation sets
    3. Train CNN-LSTM model
    4. Save trained model and pattern mappings
    """
    
    logger.info("="*70)
    logger.info("AI LAYER TRAINING PIPELINE")
    logger.info("="*70)
    
    # Create models directory
    models_dir = Path("models/ai")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Generate Training Data
    # ========================================================================
    logger.info("\nSTEP 1: Mining candlestick patterns...")
    
    miner = PatternMiner(sequence_length=15)
    X, y, pattern_map = miner.mine_patterns(
        num_samples=num_samples,
        use_synthetic=True
    )
    
    logger.info(f"✓ Generated {len(X)} pattern samples")
    logger.info(f"✓ Patterns: {list(pattern_map.keys())}")
    logger.info(f"✓ Data shape: X={X.shape}, y={y.shape}")
    
    # ========================================================================
    # STEP 2: Add "Noise" Class
    # ========================================================================
    logger.info("\nSTEP 2: Adding noise samples...")
    
    # Generate random "no-pattern" samples (label 0)
    num_noise = int(len(X) * 0.3)  # 30% noise samples
    noise_X = []
    
    for _ in range(num_noise):
        o, h, l, c = miner.generate_synthetic_market(length=100)
        snippet = np.stack([
            o[-15:], h[-15:], l[-15:], c[-15:]
        ], axis=1)
        if snippet[0, 0] > 0:
            noise_X.append(snippet / snippet[0, 0] - 1)
    
    noise_X = np.array(noise_X)
    noise_y = np.zeros(len(noise_X), dtype=int)  # Label 0 for noise
    
    # Combine pattern + noise data
    X_combined = np.vstack([X, noise_X])
    y_combined = np.concatenate([y, noise_y])
    
    logger.info(f"✓ Added {len(noise_X)} noise samples")
    logger.info(f"✓ Total dataset: {len(X_combined)} samples")
    
    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
    logger.info("\nSTEP 3: Splitting train/validation...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_combined,
        test_size=validation_split,
        random_state=42,
        stratify=y_combined
    )
    
    logger.info(f"✓ Training: {len(X_train)} samples")
    logger.info(f"✓ Validation: {len(X_val)} samples")
    
    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    logger.info("\nSTEP 4: Training Sniper model...")
    
    num_classes = len(pattern_map) + 1  # +1 for noise class
    sniper = OHLCSniper(input_shape=(15, 4), num_classes=num_classes)
    
    history = sniper.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # ========================================================================
    # STEP 5: Evaluate Model
    # ========================================================================
    logger.info("\nSTEP 5: Evaluating model...")
    
    val_loss, val_acc = sniper.model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"✓ Validation Accuracy: {val_acc:.2%}")
    logger.info(f"✓ Validation Loss: {val_loss:.4f}")
    
    # ========================================================================
    # STEP 6: Save Everything
    # ========================================================================
    logger.info("\nSTEP 6: Saving trained model...")
    
    # Save model weights
    model_path = models_dir / "sniper.weights.h5"
    sniper.save_model(str(model_path))
    
    # Save pattern mapping
    mapping_path = models_dir / "pattern_mapping.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(pattern_map, f)
    
    # Save training config
    config = {
        'num_samples': num_samples,
        'sequence_length': 15,
        'num_classes': num_classes,
        'patterns': list(pattern_map.keys()),
        'validation_accuracy': float(val_acc),
        'validation_loss': float(val_loss)
    }
    
    config_path = models_dir / "training_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    logger.info(f"✓ Model saved: {model_path}")
    logger.info(f"✓ Mapping saved: {mapping_path}")
    logger.info(f"✓ Config saved: {config_path}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Validation Accuracy: {val_acc:.2%}")
    logger.info(f"Model location: {model_path}")
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_ai_layer.py")
    logger.info("2. Integrate into your signal aggregator")
    logger.info("="*70)
    
    return sniper, pattern_map, history


if __name__ == "__main__":
    # Run training with default parameters
    # Adjust these based on your needs:
    # - num_samples: More = better accuracy, but slower training
    # - epochs: More = better fit, but risk overfitting
    # - batch_size: Larger = faster, but needs more RAM
    
    train_sniper_model(
        num_samples=10000,  # Start with 10k, increase to 50k for production
        epochs=100,
        batch_size=64,
        validation_split=0.2
    )