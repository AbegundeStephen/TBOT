"""
The Sniper: CNN-LSTM Pattern Recognition Model
Identifies candlestick patterns from OHLC sequences
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger(__name__)


class OHLCSniper:
    """
    Deep learning model for candlestick pattern recognition
    Architecture: Conv1D → MaxPooling → LSTM → Dense
    """
    
    def __init__(self, input_shape=(15, 4), num_classes=21):
        """
        Args:
            input_shape: (sequence_length, 4) for OHLC
            num_classes: Number of patterns + 1 (Noise class)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
        logger.info(f"[SNIPER] Model initialized: {input_shape} → {num_classes} classes")

    def _build_model(self):
        """
        Build CNN-LSTM hybrid model
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Feature extraction (learn pattern shapes)
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Sequence understanding (temporal relationships)
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.4),
            
            # Decision layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"[SNIPER] Model compiled ({model.count_params():,} parameters)")
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the pattern recognition model
        
        Args:
            X_train: Training data (N, 15, 4)
            y_train: Training labels (N,)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info(f"[SNIPER] Training on {len(X_train)} samples...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        final_acc = history.history['accuracy'][-1]
        logger.info(f"[SNIPER] ✓ Training complete (final accuracy: {final_acc:.2%})")
        
        return history

    def predict(self, ohlc_sequence):
        """
        Predict pattern from recent candles
        
        Args:
            ohlc_sequence: Array of shape (15, 4) - last 15 OHLC candles
            
        Returns:
            pattern_id: Predicted pattern ID (0 = Noise)
            confidence: Prediction confidence (0-1)
        """
        # Normalize exactly as in training
        if ohlc_sequence[0, 0] > 0:
            normalized = ohlc_sequence / ohlc_sequence[0, 0] - 1
        else:
            normalized = ohlc_sequence.copy()
        
        # Reshape for model input
        normalized = normalized.reshape(1, 15, 4)
        
        # Get prediction
        prediction = self.model.predict(normalized, verbose=0)
        pattern_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return pattern_id, confidence

    def save_model(self, filepath):
        """Save trained model weights"""
        self.model.save_weights(filepath)
        logger.info(f"[SNIPER] Model saved to {filepath}")

    def load_model(self, filepath):
        """Load pre-trained model weights"""
        self.model.load_weights(filepath)
        logger.info(f"[SNIPER] Model loaded from {filepath}")