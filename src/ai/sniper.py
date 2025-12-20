# ==============================================================================
# 2.  SNIPER.PY
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


class OHLCSniper:
    """
    Pattern recognition model trained on 15min candles
    Client Requirement: Must detect patterns from 15min timeframe
    """

    def __init__(self, input_shape=(15, 4), num_classes=17, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.timeframe = "15min"
        self.timeframe = "15min"  # Explicit timeframe tracking
        self.model = self._build_model()

        logger.info(
            f"[SNIPER] Initialized for {self.timeframe} candles: "
            f"{input_shape} → {num_classes} classes"
        )

    def _build_model(self):
        """Build enhanced CNN-LSTM architecture"""
        inputs = layers.Input(shape=self.input_shape)

        # Multi-scale feature extraction
        conv1_short = layers.Conv1D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        conv1_short = layers.BatchNormalization()(conv1_short)

        conv1_medium = layers.Conv1D(
            64,
            5,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        conv1_medium = layers.BatchNormalization()(conv1_medium)

        conv1_long = layers.Conv1D(
            64,
            7,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.001),
        )(inputs)
        conv1_long = layers.BatchNormalization()(conv1_long)

        conv1_combined = layers.Concatenate()([conv1_short, conv1_medium, conv1_long])
        conv1_combined = layers.Dropout(self.dropout_rate * 0.7)(conv1_combined)

        # Deeper layers with residuals
        conv2 = layers.Conv1D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.001),
        )(conv1_combined)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(self.dropout_rate)(conv2)

        conv3 = layers.Conv1D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.001),
        )(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Add()([conv2, conv3])
        conv3 = layers.Dropout(self.dropout_rate)(conv3)

        # Attention mechanism
        attention = layers.Dense(1, activation="tanh")(conv3)
        attention = layers.Flatten()(attention)
        attention = layers.Activation("softmax")(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        conv_attended = layers.Multiply()([conv3, attention])

        # Bidirectional LSTM
        lstm = layers.Bidirectional(
            layers.LSTM(
                64,
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=0.2,
            )
        )(conv_attended)
        lstm = layers.Dropout(self.dropout_rate * 1.2)(lstm)

        # Decision layers
        dense1 = layers.Dense(
            256, activation="relu", kernel_regularizer=regularizers.l2(0.001)
        )(lstm)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(self.dropout_rate * 1.3)(dense1)

        dense2 = layers.Dense(
            128, activation="relu", kernel_regularizer=regularizers.l2(0.001)
        )(dense1)
        dense2 = layers.Dropout(self.dropout_rate)(dense2)

        outputs = layers.Dense(self.num_classes, activation="softmax")(dense2)

        model = models.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        #  Use SparseTopKCategoricalAccuracy for sparse labels
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            ],
        )

        logger.info(f"[SNIPER] Model built ({model.count_params():,} parameters)")
        return model

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=300,
        batch_size=64,
        use_class_weights=True,
        max_class_weight=10.0,
        early_stopping_patience=30,
        verbose=1,
    ):
        """Train on 15min pattern data"""
        logger.info(f"[SNIPER] Training on {len(X_train)} {self.timeframe} samples...")

        # Compute class weights
        class_weights = None
        if use_class_weights:
            try:
                classes = np.unique(y_train)
                weights = compute_class_weight("balanced", classes=classes, y=y_train)
                weights_capped = np.minimum(weights, max_class_weight)
                class_weights = {
                    int(cls): float(w) for cls, w in zip(classes, weights_capped)
                }
                logger.info(
                    f"[SNIPER] Using capped class weights (max={max_class_weight})"
                )
            except Exception as e:
                logger.warning(f"[SNIPER] Class weights failed: {e}")

        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy" if X_val is not None else "accuracy",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=verbose,
        )

        final_acc = history.history["accuracy"][-1]
        if X_val is not None:
            final_val_acc = history.history["val_accuracy"][-1]
            logger.info(
                f"[SNIPER] ✓ Complete: train={final_acc:.2%}, val={final_val_acc:.2%}"
            )

        return history

    def predict(self, ohlc_sequence_15min):
        """
        Predict pattern from last 15 candles of 15min data

        Args:
            ohlc_sequence_15min: (15, 4) array of 15min OHLC candles

        Returns:
            pattern_id, confidence, extra_info
        """
        if ohlc_sequence_15min.shape != (15, 4):
            raise ValueError(
                f"Expected (15, 4) shape, got {ohlc_sequence_15min.shape}. "
                f"Sniper requires 15 consecutive {self.timeframe} candles."
            )

        if ohlc_sequence_15min[0, 0] > 0:
            normalized = ohlc_sequence_15min / ohlc_sequence_15min[0, 0] - 1
        else:
            normalized = ohlc_sequence_15min.copy()

        normalized = normalized.reshape(1, 15, 4)

        prediction = self.model.predict(normalized, verbose=0)
        pattern_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        top3_ids = np.argsort(prediction[0])[-3:][::-1]
        top3_conf = prediction[0][top3_ids]

        return (
            pattern_id,
            confidence,
            {"top3_ids": top3_ids.tolist(), "top3_confidences": top3_conf.tolist()},
        )

    def predict_single(self, ohlc_sequence_normalized):
        """
        ADDED: Predict from pre-normalized input (for HybridSignalValidator)

        Args:
            ohlc_sequence_normalized: Already normalized (1, 15, 4) array

        Returns:
            pattern_id (int), confidence (float)
        """
        # Validate input shape
        if len(ohlc_sequence_normalized.shape) != 3:
            raise ValueError(
                f"Expected (1, 15, 4) shape, got {ohlc_sequence_normalized.shape}"
            )

        if ohlc_sequence_normalized.shape != (1, 15, 4):
            raise ValueError(
                f"Expected (1, 15, 4) shape, got {ohlc_sequence_normalized.shape}. "
                f"Input should be pre-normalized and batched."
            )

        # Make prediction
        prediction = self.model.predict(ohlc_sequence_normalized, verbose=0)
        pattern_id = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        return pattern_id, confidence

    def save_model(self, filepath):
        """Save weights"""
        self.model.save_weights(filepath)
        logger.info(f"[SNIPER] Saved to {filepath}")

    def load_model(self, filepath):
        """Load weights"""
        self.model.load_weights(filepath)
        logger.info(f"[SNIPER] Loaded from {filepath}")
