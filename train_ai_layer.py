"""
IMPROVED Production Training Script - Fixes for 14% accuracy issue
Key improvements:
1. Better class balancing with capped weights
2. Stronger pattern filtering (min 50 samples)
3. Balanced noise generation
4. Longer training with better early stopping
5. Data augmentation improvements
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import AI components
from src.ai.pattern_miner import PatternMiner
from src.ai.sniper import OHLCSniper


def plot_training_history(history, save_path='models/ai/training_history.png'):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"✓ Training plots saved: {save_path}")


def balance_classes_with_oversampling(X, y, target_samples_per_class=None):
    """
    Balance classes by oversampling minority classes with augmentation
    """
    from scipy.ndimage import shift
    
    class_counts = Counter(y)
    
    if target_samples_per_class is None:
        # Target is 80% of the largest class
        target_samples_per_class = int(max(class_counts.values()) * 0.8)
    
    logger.info(f"\n[BALANCING] Target: {target_samples_per_class} samples per class")
    
    X_balanced = []
    y_balanced = []
    
    for class_id in sorted(class_counts.keys()):
        # Get samples for this class
        mask = (y == class_id)
        X_class = X[mask]
        current_count = len(X_class)
        
        # Add original samples
        X_balanced.append(X_class)
        y_balanced.append(np.full(current_count, class_id))
        
        # Oversample if needed
        if current_count < target_samples_per_class:
            needed = target_samples_per_class - current_count
            
            # Randomly sample and augment
            for _ in range(needed):
                idx = np.random.randint(0, current_count)
                sample = X_class[idx].copy()
                
                # Apply random augmentation
                aug_type = np.random.choice(['noise', 'shift', 'scale', 'none'])
                
                if aug_type == 'noise':
                    # Add small noise
                    sample += np.random.normal(0, 0.003, sample.shape)
                elif aug_type == 'shift':
                    # Tiny temporal shift
                    shift_amount = np.random.randint(-1, 2)
                    if shift_amount != 0:
                        sample = shift(sample, (shift_amount, 0), mode='nearest')
                elif aug_type == 'scale':
                    # Slight scaling
                    scale = np.random.uniform(0.995, 1.005)
                    sample *= scale
                
                X_balanced.append(sample.reshape(1, *sample.shape))
                y_balanced.append(np.array([class_id]))
            
            logger.info(f"  Class {class_id}: {current_count} → {target_samples_per_class} (+{needed} augmented)")
        else:
            logger.info(f"  Class {class_id}: {current_count} samples (no augmentation needed)")
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    logger.info(f"[BALANCING] Final dataset: {len(X_balanced)} samples")
    
    return X_balanced, y_balanced


def train_production_model(
    # DATA SOURCES
    assets=['btc', 'gold'],
    csv_files=None,
    
    # TRAINING PARAMETERS - IMPROVED DEFAULTS
    samples_per_pattern=2000,  # Increased from 1500
    min_samples_per_class=50,  # Increased from 10 to filter weak patterns
    epochs=300,  # Increased from 150
    batch_size=64,
    validation_split=0.2,
    use_augmentation=True,
    balance_classes=True,  # NEW: Balance classes with smart oversampling
    
    # MODEL SETTINGS
    use_class_weights=True,
    max_class_weight=10.0,  # NEW: Cap extreme weights
    
    # OUTPUT
    model_name='sniper_production_v2',
    save_plots=True
):
    """
    IMPROVED production training pipeline
    """
    
    logger.info("="*70)
    logger.info("IMPROVED PRODUCTION AI TRAINING")
    logger.info("="*70)
    
    models_dir = Path("models/ai")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load & Mine Historical Data
    # ========================================================================
    logger.info("\nSTEP 1: Mining patterns from historical data...")
    
    miner = PatternMiner(sequence_length=15)
    
    try:
        if csv_files:
            logger.info(f"Loading specific files: {csv_files}")
            df = miner.load_multiple_sources(csv_files)
        else:
            logger.info(f"Loading TRAINING data for assets: {assets}")
            all_dfs = []
            for asset in assets:
                try:
                    train_path = f"data/train_data_{asset.lower()}.csv"
                    if not Path(train_path).exists():
                        logger.warning(f"  ✗ {train_path} not found, skipping")
                        continue
                    
                    df_asset = miner.load_csv_data(train_path)
                    all_dfs.append(df_asset)
                    logger.info(f"  ✓ Loaded {asset} training data: {len(df_asset)} candles")
                except Exception as e:
                    logger.warning(f"  ✗ Skipped {asset}: {e}")
            
            if not all_dfs:
                raise ValueError("No asset data loaded. Check your data/ folder.")
            
            df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"✓ Combined {len(df)} total candles from {len(all_dfs)} assets")
        
        # Mine patterns
        X, y, pattern_map = miner.mine_from_dataframe(
            df,
            samples_per_pattern=samples_per_pattern,
            use_augmentation=use_augmentation
        )
        
    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        raise
    
    # Check distribution
    class_counts = Counter(y)
    logger.info(f"✓ Mined {len(X)} pattern samples")
    logger.info(f"✓ Initial class distribution:")
    for pattern_name, pattern_id in pattern_map.items():
        count = class_counts.get(pattern_id, 0)
        logger.info(f"    {pattern_name}: {count} samples")
    
    # ========================================================================
    # STEP 2: Remove weak patterns (STRICTER FILTER)
    # ========================================================================
    logger.info(f"\nSTEP 2: Removing patterns with < {min_samples_per_class} samples...")
    
    classes_to_keep = [pid for pid, count in class_counts.items() 
                       if count >= min_samples_per_class]
    
    patterns_to_remove = []
    new_pattern_map = {}
    next_id = 1
    
    for pattern_name, pattern_id in pattern_map.items():
        if pattern_id in classes_to_keep:
            new_pattern_map[pattern_name] = next_id
            next_id += 1
        else:
            patterns_to_remove.append(pattern_name)
            logger.info(f"  ✗ Removing '{pattern_name}': only {class_counts[pattern_id]} samples")
    
    if patterns_to_remove:
        old_to_new = {old_id: new_pattern_map[name] 
                      for name, old_id in pattern_map.items() 
                      if name in new_pattern_map}
        
        mask = np.isin(y, list(old_to_new.keys()))
        X = X[mask]
        y_old = y[mask]
        y = np.array([old_to_new[old_id] for old_id in y_old])
        pattern_map = new_pattern_map
        
        logger.info(f"✓ Kept {len(pattern_map)} patterns, removed {len(patterns_to_remove)}")
        logger.info(f"✓ Remaining samples: {len(X)}")
    
    # ========================================================================
    # STEP 3: Generate BALANCED noise class
    # ========================================================================
    logger.info("\nSTEP 3: Generating balanced noise class...")
    
    # Use median count for noise (not average, to avoid outlier bias)
    pattern_counts = [class_counts[pid] for pid in new_pattern_map.values()]
    noise_target = int(np.median(pattern_counts))
    
    logger.info(f"  Target noise samples: {noise_target} (median of pattern counts)")
    
    noise_X = miner.generate_noise_samples(df, num_samples=noise_target)
    noise_y = np.zeros(len(noise_X), dtype=int)
    
    logger.info(f"✓ Generated {len(noise_X)} noise samples")
    
    # ========================================================================
    # STEP 4: Combine and optionally balance
    # ========================================================================
    logger.info("\nSTEP 4: Combining and balancing dataset...")
    
    X_combined = np.vstack([X, noise_X])
    y_combined = np.concatenate([y, noise_y])
    
    if balance_classes:
        X_combined, y_combined = balance_classes_with_oversampling(
            X_combined, y_combined
        )
    else:
        # Just shuffle
        shuffle_idx = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
    
    final_counts = Counter(y_combined)
    logger.info(f"✓ Total samples: {len(X_combined)}")
    logger.info(f"✓ Final class distribution:")
    logger.info(f"    Noise (0): {final_counts[0]} samples")
    for pattern_name, pattern_id in sorted(pattern_map.items(), key=lambda x: x[1]):
        count = final_counts.get(pattern_id, 0)
        logger.info(f"    {pattern_name} ({pattern_id}): {count} samples")
    
    # ========================================================================
    # STEP 5: Train/Val Split
    # ========================================================================
    logger.info("\nSTEP 5: Splitting dataset...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_combined,
        test_size=validation_split,
        random_state=42,
        stratify=y_combined
    )
    
    logger.info(f"✓ Training: {len(X_train)} samples")
    logger.info(f"✓ Validation: {len(X_val)} samples")
    
    # ========================================================================
    # STEP 6: Train with IMPROVED parameters
    # ========================================================================
    logger.info("\nSTEP 6: Training improved model...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Early stopping patience: 30")
    logger.info(f"  Class weights: {use_class_weights} (capped at {max_class_weight})")
    
    num_classes = len(pattern_map) + 1
    sniper = OHLCSniper(input_shape=(15, 4), num_classes=num_classes)
    
    # MODIFIED: Update the model's callbacks before training
    import tensorflow as tf
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # Increased from 15
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Increased from 5
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Compute capped class weights
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        # Cap weights at max_class_weight
        weights_capped = np.minimum(weights, max_class_weight)
        class_weights = {int(cls): float(w) for cls, w in zip(classes, weights_capped)}
        
        logger.info(f"[TRAINING] Using capped class weights (max={max_class_weight}):")
        for cls, weight in sorted(class_weights.items()):
            original = weights[list(classes).index(cls)]
            if original > max_class_weight:
                logger.info(f"    Class {cls}: {weight:.3f} (capped from {original:.3f})")
            else:
                logger.info(f"    Class {cls}: {weight:.3f}")
    
    # Train with custom callbacks
    history = sniper.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    logger.info(f"[TRAINING] ✓ Complete (train: {final_acc:.2%}, val: {final_val_acc:.2%})")
    
    # ========================================================================
    # STEP 7: Evaluation
    # ========================================================================
    logger.info("\nSTEP 7: Evaluating model...")
    
    val_loss, val_acc = sniper.model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(sniper.model.predict(X_val, verbose=0), axis=1)
    
    logger.info(f"✓ Validation Accuracy: {val_acc:.2%}")
    logger.info(f"✓ Validation Loss: {val_loss:.4f}")
    
    # Per-class metrics
    reverse_map = {v: k for k, v in pattern_map.items()}
    reverse_map[0] = 'Noise'
    target_names = [reverse_map.get(i, f'Class_{i}') for i in range(num_classes)]
    
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*70)
    print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))
    
    # Confusion analysis
    cm = confusion_matrix(y_val, y_pred)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 5:
                confused_pairs.append((target_names[i], target_names[j], cm[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info("\n" + "="*70)
    logger.info("MOST CONFUSED PAIRS")
    logger.info("="*70)
    for true_class, pred_class, count in confused_pairs[:10]:
        logger.info(f"  {true_class} → {pred_class}: {count} times")
    
    # ========================================================================
    # STEP 8: Save Everything
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 8: Saving model...")
    logger.info("="*70)
    
    model_path = models_dir / f"{model_name}.weights.h5"
    sniper.save_model(str(model_path))
    
    mapping_path = models_dir / f"{model_name}_mapping.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(pattern_map, f)
    
    config = {
        'model_name': model_name,
        'assets_used': assets if not csv_files else 'custom',
        'num_samples': len(X_combined),
        'samples_per_pattern': samples_per_pattern,
        'min_samples_per_class': min_samples_per_class,
        'sequence_length': 15,
        'num_classes': num_classes,
        'patterns': list(pattern_map.keys()),
        'removed_patterns': patterns_to_remove,
        'validation_accuracy': float(val_acc),
        'validation_loss': float(val_loss),
        'balance_classes': balance_classes,
        'max_class_weight': max_class_weight,
        'epochs_trained': len(history.history['loss']),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    config_path = models_dir / f"{model_name}_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    logger.info(f"✓ Model: {model_path}")
    logger.info(f"✓ Mapping: {mapping_path}")
    logger.info(f"✓ Config: {config_path}")
    
    if save_plots:
        plot_path = models_dir / f"{model_name}_history.png"
        plot_training_history(history, save_path=str(plot_path))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("🎯 TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"✅ Validation Accuracy: {val_acc:.2%}")
    logger.info(f"✅ Improvement: {val_acc*100 - 14.05:.1f} percentage points vs previous")
    logger.info(f"✅ Model: {model_path}")
    logger.info(f"✅ Active patterns: {len(pattern_map)}")
    if patterns_to_remove:
        logger.info(f"⚠️  Excluded: {', '.join(patterns_to_remove)}")
    
    if val_acc < 0.50:
        logger.warning("\n⚠️  STILL LOW ACCURACY - NEXT STEPS:")
        logger.warning("  1. Add more assets (ETH, EUR/USD, SPX)")
        logger.warning("  2. Increase min_samples_per_class to 100")
        logger.warning("  3. Check data quality (are patterns actually present?)")
    elif val_acc < 0.70:
        logger.info("\n✅ Better! Further improvements:")
        logger.info("  - Add 2-3 more diverse assets")
        logger.info("  - Increase samples_per_pattern to 3000")
    else:
        logger.info("\n🚀 EXCELLENT! Model ready for production!")
    
    logger.info("\n" + "="*70)
    
    return sniper, pattern_map, history, config


if __name__ == "__main__":
    
    # Run improved training
    train_production_model(
        assets=['btc', 'gold'],
        samples_per_pattern=2000,      # Increased
        min_samples_per_class=50,      # Much stricter filtering
        epochs=300,                     # More training
        batch_size=64,
        balance_classes=True,           # NEW: Balance with augmentation
        use_class_weights=True,
        max_class_weight=10.0,          # NEW: Cap extreme weights
        model_name='sniper_btc_gold_v2'
    )