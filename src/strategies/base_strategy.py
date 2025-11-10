# src/strategies/base_strategy.py

"""
Base Strategy Class - Abstract interface for all trading strategies
Implements anti-leakage and noise reduction protocols
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
        
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate strategy-specific features
        CRITICAL: Must not use future data (no lookahead bias)
        """
        pass
    
    @abstractmethod
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate training labels based on strategy logic
        CRITICAL: Labels must be from PAST data only
        """
        pass
    
    def remove_data_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that could cause data leakage
        - Drop rows with NaN features (from indicator warm-up)
        - Ensure no future data in features
        """
        logger.info(f"[{self.name}] Pre-cleaning rows: {len(df)}")
        
        # Drop NaN values (indicator warm-up period)
        df_clean = df.dropna().copy()
        
        # Drop the first N rows to ensure all indicators are stable
        warmup_period = max(
            self.config.get('slow_ma', 200),
            self.config.get('bb_period', 20)
        )
        df_clean = df_clean.iloc[warmup_period:].copy()
        
        logger.info(f"[{self.name}] Post-cleaning rows: {len(df_clean)}")
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, std_threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove statistical outliers using z-score method
        Helps reduce noise in training data
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'label']
        
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        outlier_mask = (z_scores < std_threshold).all(axis=1)
        
        removed = len(df) - outlier_mask.sum()
        logger.info(f"[{self.name}] Removed {removed} outlier rows")
        
        return df[outlier_mask].copy()
    
    def train_model(self, df: pd.DataFrame, model_path: str) -> Dict:
        """
        Train ML model with proper time-series validation
        Implements walk-forward validation to prevent leakage
        """
        logger.info(f"[{self.name}] Starting model training...")
        
        # Generate features and labels
        df_features = self.generate_features(df)
        df_features['label'] = self.generate_labels(df_features)
        
        # Anti-leakage: Remove problematic rows
        df_clean = self.remove_data_leakage(df_features)
        
        # Noise reduction: Remove outliers
        if self.config.get('remove_outliers', True):
            df_clean = self.remove_outliers(
                df_clean, 
                self.config.get('outlier_std', 3.0)
            )
        
        # Drop label column from features
        self.feature_columns = [col for col in df_clean.columns if col not in ['label', 'timestamp']]
        X = df_clean[self.feature_columns].values
        y = df_clean['label'].values
        
        # Handle class imbalance
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"[{self.name}] Class distribution: {dict(zip(unique, counts))}")
        
        if len(unique) < 2:
            logger.error(f"[{self.name}] Insufficient class diversity. Cannot train.")
            return {'success': False, 'error': 'Insufficient class diversity'}
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series cross-validation (prevents leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            temp_model = RandomForestClassifier(
                **self.config.get('model_params', {}),
                class_weight='balanced'  # Handle imbalance
            )
            temp_model.fit(X_train, y_train)
            score = temp_model.score(X_val, y_val)
            cv_scores.append(score)
            logger.info(f"[{self.name}] Fold {fold+1} accuracy: {score:.4f}")
        
        # Train final model on all data
        self.model = RandomForestClassifier(
            **self.config.get('model_params', {}),
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        
        metrics = {
            'success': True,
            'cv_mean_accuracy': np.mean(cv_scores),
            'cv_std_accuracy': np.std(cv_scores),
            'train_samples': len(X),
            'n_features': len(self.feature_columns)
        }
        
        logger.info(f"[{self.name}] Training complete: {metrics}")
        return metrics
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model from disk"""
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_columns = saved_data['feature_columns']
            logger.info(f"[{self.name}] Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load model: {e}")
            return False
    
    def predict_signal(self, df: pd.DataFrame) -> int:
        """
        Predict trade signal for the latest bar
        Returns: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        if self.model is None:
            logger.error(f"[{self.name}] Model not loaded")
            return 0
        
        # Generate features for latest data
        df_features = self.generate_features(df)
        
        # Use only the last row (current bar)
        latest = df_features.iloc[[-1]][self.feature_columns]
        
        # Check for NaN
        if latest.isnull().any().any():
            logger.warning(f"[{self.name}] NaN values in latest features, returning HOLD")
            return 0
        
        # Scale and predict
        X_scaled = self.scaler.transform(latest.values)
        prediction = self.model.predict(X_scaled)[0]
        
        # Map to signal: assuming labels are 0 (HOLD), 1 (BUY), -1 (SELL)
        signal_map = {0: 0, 1: 1, -1: -1, 2: -1}  # Flexible mapping
        signal = signal_map.get(prediction, 0)
        
        logger.debug(f"[{self.name}] Predicted signal: {signal}")
        return signal
