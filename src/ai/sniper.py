"""
IMPROVED Sniper: Enhanced CNN-LSTM Pattern Recognition Model
Key improvements:
1. Deeper architecture with residual connections
2. Better regularization (dropout + L2)
3. Attention mechanism for pattern focus
4. Improved feature extraction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


class OHLCSniperV2:
    """
    IMPROVED deep learning model for candlestick pattern recognition
    Architecture: Multi-scale Conv1D → Attention → Bidirectional LSTM → Dense
    """
    
    def __init__(self, input_shape=(15, 4), num_classes=21):
        """
        Args:
            input_shape: (sequence_length, 4) for OHLC
            num_classes: Number of patterns + 1 (Noise class)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_improved_model()
        
        logger.info(f"[SNIPER V2] Model initialized: {input_shape} → {num_classes} classes")

    def _build_improved_model(self):
        """
        Build improved CNN-LSTM hybrid with attention
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # ==================================================================
        # MULTI-SCALE FEATURE EXTRACTION
        # Extract features at different time scales (short, medium, long)
        # ==================================================================
        
        # Short-term patterns (3-candle)
        conv1_short = layers.Conv1D(
            64, 3, activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )(inputs)
        conv1_short = layers.BatchNormalization()(conv1_short)
        
        # Medium-term patterns (5-candle)
        conv1_medium = layers.Conv1D(
            64, 5, activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )(inputs)
        conv1_medium = layers.BatchNormalization()(conv1_medium)
        
        # Combine multi-scale features
        conv1_combined = layers.Concatenate()([conv1_short, conv1_medium])
        conv1_combined = layers.Dropout(0.2)(conv1_combined)
        
        # ==================================================================
        # DEEPER FEATURE LEARNING
        # ==================================================================
        
        conv2 = layers.Conv1D(
            128, 3, activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )(conv1_combined)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.3)(conv2)
        
        conv3 = layers.Conv1D(
            128, 3, activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        )(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        
        # Residual connection
        conv3 = layers.Add()([conv2, conv3])
        conv3 = layers.Dropout(0.3)(conv3)
        
        # ==================================================================
        # ATTENTION MECHANISM
        # Learn which parts of the sequence are important
        # ==================================================================
        
        # Simple attention layer
        attention = layers.Dense(1, activation='tanh')(conv3)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention weights
        conv_attended = layers.Multiply()([conv3, attention])
        
        # ==================================================================
        # TEMPORAL MODELING with Bidirectional LSTM
        # ==================================================================
        
        lstm = layers.Bidirectional(
            layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
        )(conv_attended)
        lstm = layers.Dropout(0.4)(lstm)
        
        # ==================================================================
        # DECISION LAYERS
        # ==================================================================
        
        dense1 = layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        )(lstm)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        )(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(dense2)
        
        # ==================================================================
        # BUILD AND COMPILE
        # ==================================================================
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Use Adam with learning rate schedule
        initial_lr = 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"[SNIPER V2] Model compiled ({model.count_params():,} parameters)")
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, use_class_weights=False):
        """
        Train the pattern recognition model
        
        Args:
            X_train: Training data (N, 15, 4)
            y_train: Training labels (N,)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Training epochs
            batch_size: Batch size
            use_class_weights: If True, balance classes using computed weights
            
        Returns:
            Training history
        """
        logger.info(f"[SNIPER V2] Training on {len(X_train)} samples...")
        
        # Compute class weights if requested
        class_weights = None
        if use_class_weights:
            try:
                classes = np.unique(y_train)
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y_train
                )
                
                class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
                
                logger.info(f"[SNIPER V2] Using class weights:")
                for cls, weight in sorted(class_weights.items()):
                    logger.info(f"    Class {cls}: {weight:.3f}")
                    
            except Exception as e:
                logger.warning(f"[SNIPER V2] Failed to compute class weights: {e}")
                class_weights = None
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        final_acc = history.history['accuracy'][-1]
        val_info = ""
        if X_val is not None:
            final_val_acc = history.history['val_accuracy'][-1]
            val_info = f", validation: {final_val_acc:.2%}"
        
        logger.info(f"[SNIPER V2] ✓ Training complete (train: {final_acc:.2%}{val_info})")
        
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

    def predict_batch(self, ohlc_sequences):
        """
        Predict patterns for multiple sequences at once
        
        Args:
            ohlc_sequences: Array of shape (N, 15, 4)
            
        Returns:
            pattern_ids: Array of predicted pattern IDs (N,)
            confidences: Array of confidences (N,)
        """
        # Normalize all sequences
        normalized = np.zeros_like(ohlc_sequences)
        for i, seq in enumerate(ohlc_sequences):
            if seq[0, 0] > 0:
                normalized[i] = seq / seq[0, 0] - 1
            else:
                normalized[i] = seq.copy()
        
        # Get predictions
        predictions = self.model.predict(normalized, verbose=0)
        pattern_ids = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return pattern_ids, confidences

    def save_model(self, filepath):
        """Save trained model weights"""
        self.model.save_weights(filepath)
        logger.info(f"[SNIPER V2] Model saved to {filepath}")

    def load_model(self, filepath):
        """Load pre-trained model weights"""
        self.model.load_weights(filepath)
        logger.info(f"[SNIPER V2] Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model architecture summary"""
        return self.model.summary()


# Alias for backward compatibility
OHLCSniper = OHLCSniperV2