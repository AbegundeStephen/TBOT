# The `BaseStrategy` class provides a framework for trading strategies with machine learning
# integration, allowing for the generation of features, training of models, and generation of trade
# signals with raw confidence scores.
"""
 Base Strategy - Returns raw confidence scores without filtering
Lets the aggregator make the final decision
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies with ML integration"""

    def __init__(self, config: Dict, name: str):
        self.config = config
        self.name = name
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns = []

        # Keep for logging but DON'T use for filtering in generate_signal
        self.min_confidence = config.get("min_confidence", 0.55)
        logger.info(
            f"[{self.name}] Confidence threshold (info only): {self.min_confidence:.2f}"
        )

    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy-specific features"""
        pass

    @abstractmethod
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate training labels based on strategy logic"""
        pass

    def get_warmup_period(self) -> int:
        """Calculate the minimum warmup period needed for indicators"""
        periods = [
            self.config.get("fast_ma", 20),
            self.config.get("slow_ma", 50),
            self.config.get("bb_period", 20),
            self.config.get("macd_slow", 26),
            self.config.get("adx_period", 14),
        ]
        return max([p for p in periods if p is not None])

    def remove_data_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that could cause data leakage"""
        logger.info(f"[{self.name}] Pre-cleaning rows: {len(df)}")

        df_clean = df.dropna().copy()
        warmup_period = self.get_warmup_period()

        if len(df_clean) > warmup_period:
            df_clean = df_clean.iloc[warmup_period:].copy()

        logger.info(f"[{self.name}] Post-cleaning rows: {len(df_clean)}")
        return df_clean

    def remove_outliers(
        self, df: pd.DataFrame, std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove statistical outliers using IQR method (more robust for financial data)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "label"]

        if len(numeric_cols) == 0:
            logger.info(f"[{self.name}] No numeric columns for outlier detection")
            return df.copy()

        df_clean = df.copy()

        # Use IQR method instead of z-score for financial data (more robust)
        Q1 = df_clean[numeric_cols].quantile(0.25)
        Q3 = df_clean[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds using IQR (less aggressive than z-score)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Create mask for rows within bounds
        outlier_mask = (
            (df_clean[numeric_cols] >= lower_bound)
            & (df_clean[numeric_cols] <= upper_bound)
        ).all(axis=1)

        removed = len(df_clean) - outlier_mask.sum()

        # Safety check: don't remove too many samples
        if removed > len(df_clean) * 0.5:  # If removing more than 50%
            logger.warning(
                f"[{self.name}] Outlier removal would remove {removed}/{len(df_clean)} samples ({removed/len(df_clean)*100:.1f}%)"
            )
            logger.warning(f"[{self.name}] Using less aggressive bounds (5x IQR)")

            # Use less aggressive bounds
            lower_bound = Q1 - 5 * IQR
            upper_bound = Q3 + 5 * IQR
            outlier_mask = (
                (df_clean[numeric_cols] >= lower_bound)
                & (df_clean[numeric_cols] <= upper_bound)
            ).all(axis=1)
            removed = len(df_clean) - outlier_mask.sum()

            # If still too aggressive, disable outlier removal
            if removed > len(df_clean) * 0.3:  # Still removing more than 30%
                logger.warning(
                    f"[{self.name}] Still removing {removed} samples, disabling outlier removal"
                )
                return df.copy()

        logger.info(f"[{self.name}] Removed {removed} outlier rows")

        result = df_clean[outlier_mask].copy()

        # Final safety check
        if len(result) < 100:
            logger.warning(
                f"[{self.name}] Very few samples remaining ({len(result)}), disabling outlier removal"
            )
            return df.copy()

        return result

    def train_model(self, df: pd.DataFrame, model_path: str) -> Dict:
        logger.info(f"[{self.name}] Starting model training...")
        df_features = self.generate_features(df)
        df_features["label"] = self.generate_labels(df_features)
        df_clean = self.remove_data_leakage(df_features)
        if self.config.get("remove_outliers", True):
            df_clean = self.remove_outliers(
                df_clean, self.config.get("outlier_std", 3.0)
            )

        # ONLY take numeric columns for features
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        self.feature_columns = [
            col for col in numeric_cols if col not in ["label", "timestamp", "time", "date"]
        ]
        
        X = df_clean[self.feature_columns].values
        y = df_clean["label"].values

        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"[{self.name}] Class distribution: {dict(zip(unique, counts))}")

        if len(unique) < 2:
            logger.error(f"[{self.name}] Insufficient class diversity")
            return {"success": False, "error": "Insufficient class diversity"}

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        model_params = self.config.get("model_params", {})

        # Ensure class_weight is correctly defined
        if "class_weight" not in model_params:
            model_params["class_weight"] = "balanced"
        else:
            # Convert string keys to integers if necessary
            class_weight = model_params["class_weight"]
            if isinstance(class_weight, dict):
                class_weight = {int(k): v for k, v in class_weight.items()}
                model_params["class_weight"] = class_weight

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            temp_model = RandomForestClassifier(**model_params)
            temp_model.fit(X_train, y_train)
            score = temp_model.score(X_val, y_val)
            cv_scores.append(score)
            logger.info(f"[{self.name}] Fold {fold+1} accuracy: {score:.4f}")

        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X_scaled, y)

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
            },
            model_path,
        )

        metrics = {
            "success": True,
            "cv_mean_accuracy": np.mean(cv_scores),
            "cv_std_accuracy": np.std(cv_scores),
            "train_samples": len(X),
            "n_features": len(self.feature_columns),
        }
        logger.info(f"[{self.name}] Training complete: {metrics}")
        return metrics

    def load_model(self, model_path: str) -> bool:
        """Load trained model from disk"""
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.feature_columns = saved_data["feature_columns"]
            logger.info(f"[{self.name}] Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            return False

    def generate_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate trade signal WITH RAW CONFIDENCE SCORE

         NO FILTERING HERE - Let aggregator decide!

        Returns: (signal, confidence)
        - signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        - confidence: 0.0 to 1.0 (RAW probability from model)
        """
        if self.model is None:
            logger.error(f"[{self.name}] Model not loaded")
            return 0, 0.0

        try:
            df_features = self.generate_features(df)
            latest = df_features.iloc[[-1]][self.feature_columns]

            if latest.isnull().any().any():
                logger.warning(f"[{self.name}] NaN values in features, returning HOLD")
                return 0, 0.0

            X_scaled = self.scaler.transform(latest.values)

            # Get probabilities for all classes
            probas = self.model.predict_proba(X_scaled)[0]

            # Get predicted class and its probability
            predicted_class = self.model.predict(X_scaled)[0]

            # CRITICAL : Map class index to confidence correctly
            # RandomForest classes_ attribute tells us the order
            class_to_idx = {cls: idx for idx, cls in enumerate(self.model.classes_)}

            # Get confidence for the predicted class
            predicted_idx = class_to_idx.get(predicted_class, 0)
            confidence = probas[predicted_idx]

            # Map class label to signal
            # Handle both -1/0/1 and 0/1/2 encoding
            if predicted_class in [-1, 0, 1]:
                signal = int(predicted_class)
            else:
                # If classes are encoded as 0/1/2
                signal_map = {0: 0, 1: 1, 2: -1}
                signal = signal_map.get(predicted_class, 0)

            # REMOVED: No confidence filtering here!
            # Just return raw signal and confidence

            logger.debug(
                f"[{self.name}] Signal: {signal}, Confidence: {confidence:.3f}, All probas: {probas}"
            )
            return signal, confidence

        except Exception as e:
            logger.error(f"[{self.name}] Error in generate_signal: {e}", exc_info=True)
            return 0, 0.0

    def predict_signal(self, df: pd.DataFrame) -> int:
        """
        Backward compatibility wrapper
        Returns only the signal (for existing code)
        """
        signal, _ = self.generate_signal(df)
        return signal

    def get_signal_with_details(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced method that returns full signal details
        Useful for debugging and monitoring
        """
        signal, confidence = self.generate_signal(df)

        try:
            df_features = self.generate_features(df)
            latest = df_features.iloc[[-1]][self.feature_columns]
            X_scaled = self.scaler.transform(latest.values)

            # Get all class probabilities
            probas = self.model.predict_proba(X_scaled)[0]
            class_probs = {
                str(cls): float(prob) for cls, prob in zip(self.model.classes_, probas)
            }

            return {
                "signal": signal,
                "confidence": confidence,
                "class_probabilities": class_probs,
                "strategy": self.name,
                "timestamp": (
                    df.index[-1]
                    if hasattr(df.index[-1], "isoformat")
                    else str(df.index[-1])
                ),
            }
        except Exception as e:
            logger.error(f"[{self.name}] Error getting signal details: {e}")
            return {"signal": signal, "confidence": confidence, "strategy": self.name}
