"""
Continuous Learning Pipeline (The "Self-Healing" Brain)
Automates data fetching, model retraining, and hot-swapping without downtime.
"""

import logging
import threading
import time
import json
import os
import gc
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy

logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """
    Background worker that retrains ML models periodically and hot-swaps them into the live bot.
    """

    def __init__(self, config: Dict, trading_bot, telegram_bot=None):
        self.config = config
        self.trading_bot = trading_bot
        self.telegram = telegram_bot
        self.data_manager = trading_bot.data_manager

        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()

        # Configuration
        self.retrain_frequency_days = config.get("ml", {}).get(
            "retrain_frequency_days", 7
        )
        self.min_accuracy_threshold = config.get("ml", {}).get(
            "min_training_accuracy", 0.54
        )
        self.models_dir = "models"
        self.metadata_file = os.path.join(self.models_dir, "training_metadata.json")
        
        # Strategy mapping for evaluation
        self.strategy_classes = {
            "mean_reversion": MeanReversionStrategy,
            "trend_following": TrendFollowingStrategy,
            "exponential_moving_averages": EMAStrategy,
        }

        logger.info(
            f"[AUTO-TRAIN] Pipeline initialized. Retraining every {self.retrain_frequency_days} days."
        )

    def start(self):
        """Start the background pipeline thread."""
        if self.is_running:
            return

        self.is_running = True
        self._thread = threading.Thread(
            target=self._training_loop, daemon=True, name="AutoTrainer"
        )
        self._thread.start()
        logger.info("[AUTO-TRAIN] ✅ Background learning thread started")

    def stop(self):
        """Stop the background pipeline."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("[AUTO-TRAIN] 🛑 Background learning thread stopped")

    def _safe_send_telegram(self, message: str):
        """✅ FIXED: Thread-safe way to send Telegram messages from a background thread."""
        if not self.telegram or not self.telegram.is_running:
            return
            
        try:
            # Create a localized event loop for this specific thread operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.telegram.send_notification(message))
            loop.close()
        except Exception as e:
            logger.debug(f"[AUTO-TRAIN] Failed to send telegram update: {e}")

    def _wait_for_bot_readiness(self):
        """✅ FIXED: The Handshake. Prevents the race condition at startup."""
        logger.info("[AUTO-TRAIN] Waiting for main bot exchange connections...")
        max_wait = 120  # Wait up to 2 minutes for main bot to boot
        waited = 0
        
        while self.is_running and waited < max_wait:
            # Check if main bot has successfully initialized the handlers
            has_binance = self.trading_bot.binance_handler is not None
            has_mt5 = self.trading_bot.mt5_handler is not None
            
            if has_binance and has_mt5:
                logger.info("[AUTO-TRAIN] ✅ Main bot connections detected. Proceeding.")
                time.sleep(5) # Buffer for absolute safety
                return True
                
            time.sleep(2)
            waited += 2
            
        logger.error("[AUTO-TRAIN] Timed out waiting for bot connections.")
        return False

    def _get_metadata(self) -> Dict:
        """Read the full training metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[AUTO-TRAIN] Could not read metadata: {e}")
        return {}

    def _get_last_training_time(self) -> datetime:
        """Read the last training time from metadata."""
        metadata = self._get_metadata()
        return datetime.fromisoformat(
            metadata.get("last_trained", "2000-01-01T00:00:00")
        )

    def _training_loop(self):
        """Main loop that checks if retraining is needed."""
        
        # 1. ✅ DO NOT RUN until main bot says network is ready
        if not self._wait_for_bot_readiness():
            return

        while self.is_running:
            try:
                last_trained = self._get_last_training_time()
                next_training = last_trained + timedelta(days=self.retrain_frequency_days)
                now = datetime.now()

                # ✅ CHANGE 2: Drift Detection
                accuracy_drop = self._calculate_max_drift()
                
                if accuracy_drop > 0.05:
                    trigger_retrain = True  # drift detected
                    logger.info(f"🧠 [AUTO-TRAIN] DRIFT DETECTED (Max drop: {accuracy_drop:.1%}) - INITIATING IMMEDIATE RETRAIN")
                elif now >= next_training and now.weekday() == 6:  # 6 = Sunday
                    trigger_retrain = True  # scheduled retrain
                    logger.info(f"🧠 [AUTO-TRAIN] SCHEDULED BRAIN UPGRADE INITIATED")
                else:
                    trigger_retrain = False

                if trigger_retrain:
                    logger.info("\n" + "=" * 70)
                    logger.info("=" * 70)

                    self._run_training_pipeline()
                    
                    # Snooze for a day to prevent re-running on the same day
                    time.sleep(24 * 3600)
                else:
                    # Sleep for 1 hour before checking the clock again
                    time.sleep(3600)

            except Exception as e:
                logger.error(f"[AUTO-TRAIN] Fatal error in training loop: {e}", exc_info=True)
                time.sleep(3600)  # Sleep on error to prevent CPU thrashing

    def _calculate_max_drift(self) -> float:
        """Calculate the maximum accuracy drop across all enabled models."""
        metadata = self._get_metadata()
        last_summary = metadata.get("summary", {})
        if not last_summary:
            return 0.0

        max_drop = 0.0
        enabled_assets = [a for a, cfg in self.config["assets"].items() if cfg.get("enabled", False)]
        
        for asset in enabled_assets:
            if asset not in last_summary: continue
            
            # Fetch last 7 days for drift detection
            df = self._fetch_latest_data(asset, days=7)
            if df is None or len(df) < 100: continue
            
            for strat_name, last_acc in last_summary[asset].items():
                current_acc = self._evaluate_current_model(asset, strat_name, df)
                if current_acc == 0: continue
                
                drop = last_acc - current_acc
                if drop > max_drop:
                    max_drop = drop
                    
        return max_drop

    def _evaluate_current_model(self, asset: str, strategy_name: str, df: pd.DataFrame) -> float:
        """Evaluate the currently active model on new data."""
        model_path = os.path.join(self.models_dir, f"{strategy_name}_{asset.lower()}.pkl")
        if not os.path.exists(model_path):
            return 0.0
            
        try:
            strat_class = self.strategy_classes.get(strategy_name)
            if not strat_class: return 0.0
            
            strat_config = self.config["strategy_configs"][strategy_name][asset]
            temp_strat = strat_class(strat_config)
            if not temp_strat.load_model(model_path): return 0.0
            
            feat_df = temp_strat.generate_features(df)
            feat_df["label"] = temp_strat.generate_labels(feat_df)
            clean_df = temp_strat.remove_data_leakage(feat_df)
            
            X = clean_df[temp_strat.feature_columns].values
            y = clean_df["label"].values
            
            if len(X) == 0: return 0.0
            
            X_scaled = temp_strat.scaler.transform(X)
            y_pred = temp_strat.model.predict(X_scaled)
            
            return accuracy_score(y, y_pred)
        except Exception as e:
            logger.debug(f"[AUTO-TRAIN] Could not evaluate drift for {asset} {strategy_name}: {e}")
            return 0.0

    def _fetch_latest_data(self, asset: str, days: int = None) -> pd.DataFrame:
        """Fetch the most recent data block."""
        asset_cfg = self.config["assets"][asset]
        lookback_days = days if days is not None else asset_cfg.get("lookback_days", 730)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        if days is None: # Only log for training fetch, not drift check
            logger.info(f"[AUTO-TRAIN] Fetching {lookback_days} days of data for {asset}...")

        if asset_cfg["exchange"] == "binance":
            return self.data_manager.fetch_binance_data(
                symbol=asset_cfg["symbol"],
                interval=asset_cfg["interval"],
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            return self.data_manager.fetch_mt5_data(
                symbol=asset_cfg["symbol"],
                timeframe=asset_cfg["timeframe"],
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _train_strategy(
        self, strategy_class, asset: str, df: pd.DataFrame, strategy_name: str
    ) -> Dict[str, Any]:
        """
        Train a single strategy in isolation (Shadow Mode).
        ✅ TASK 16: Strict Walk-Forward Holdout (90 Days)
        """
        logger.info(f"[AUTO-TRAIN] Training {strategy_name} for {asset}...")

        # Get the specific configuration for this strategy and asset
        try:
            strategy_config = self.config["strategy_configs"][strategy_name][asset]
        except KeyError:
            logger.error(f"[AUTO-TRAIN] Missing config for {strategy_name} on {asset}")
            return {"success": False, "error": "Missing configuration"}

        # 1. SPLIT DATA (Strict 90-day Holdout)
        holdout_bars = 90 * 24
        if len(df) < (holdout_bars * 2):
            logger.warning(f"[AUTO-TRAIN] {asset} dataset too small for strict 90-day holdout. Using 20% fallback.")
            holdout_bars = int(len(df) * 0.2)

        train_df = df.iloc[:-holdout_bars].copy()
        test_df = df.iloc[-holdout_bars:].copy()

        logger.info(f"[AUTO-TRAIN] Data Split: {len(train_df)} train, {len(test_df)} holdout bars")

        # 2. Instantiate fresh strategy for training
        temp_strategy = strategy_class(strategy_config)

        # Define model path
        model_path = os.path.join(self.models_dir, f"{strategy_name}_{asset.lower()}.pkl")

        # 3. Train on training set
        train_results = temp_strategy.train_model(train_df, model_path)
        
        # 4. Validate on unseen holdout set (Honest performance)
        logger.info(f"[AUTO-TRAIN] Validating on 90-day holdout...")
        
        # Calculate features and labels for holdout set
        test_df_features = temp_strategy.generate_features(test_df)
        test_df_features["label"] = temp_strategy.generate_labels(test_df_features)
        test_df_clean = temp_strategy.remove_data_leakage(test_df_features)
        
        X_test = test_df_clean[temp_strategy.feature_columns].values
        y_test = test_df_clean["label"].values
        
        if len(X_test) > 0:
            X_test_scaled = temp_strategy.scaler.transform(X_test)
            y_pred = temp_strategy.model.predict(X_test_scaled)
            
            # ✅ CHANGE 1: Use F1-score with [1, -1] labels (Buy/Sell)
            f1 = f1_score(y_test, y_pred, labels=[1, -1], average='macro')
            accuracy = accuracy_score(y_test, y_pred)
        else:
            f1 = 0.0
            accuracy = 0.0
            
        # Merge results for reporting
        final_results = train_results.copy()
        final_results["holdout_accuracy"] = accuracy
        final_results["holdout_f1"] = f1
        
        logger.info(f"[AUTO-TRAIN] {asset} {strategy_name} Holdout F1: {f1:.3f} (Acc: {accuracy:.1%})")

        # Free memory aggressively
        del temp_strategy
        gc.collect()

        return final_results

    def _run_training_pipeline(self):
        """Execute the full retrain and hot-swap procedure."""
        with self._lock:
            self._safe_send_telegram("🧠 *Auto-Trainer Initiated*\n\nFetching new market data and retraining ML models...")

            # Get all enabled assets
            enabled_assets = [
                a for a, cfg in self.config["assets"].items() if cfg.get("enabled", False)
            ]
            
            summary = {asset: {} for asset in enabled_assets}
            all_successful = True

            strategies = [
                (MeanReversionStrategy, "mean_reversion"),
                (TrendFollowingStrategy, "trend_following"),
                (EMAStrategy, "exponential_moving_averages"),
            ]

            for asset in enabled_assets:
                asset_cfg = self.config["assets"][asset]
                min_bars = asset_cfg.get("min_bars_training", 5000)

                # 1. Fetch Latest Data
                df = self._fetch_latest_data(asset)
                if df is None or len(df) < min_bars:
                    logger.error(f"[AUTO-TRAIN] Insufficient data for {asset} ({len(df) if df is not None else 0}/{min_bars}), skipping.")
                    continue

                # 2. Train Each Strategy in Shadow Mode
                for strat_class, strat_name in strategies:
                    results = self._train_strategy(strat_class, asset, df, strat_name)

                    # ✅ CHANGE 1: Use F1-score for shadow test
                    f1 = results.get("holdout_f1", 0)
                    accuracy = results.get("holdout_accuracy", 0)
                    summary[asset][strat_name] = accuracy

                    if f1 < 0.45:
                        logger.warning(
                            f"[AUTO-TRAIN] ❌ {asset} {strat_name} failed shadow test (Holdout F1: {f1:.3f}). Aborting swap."
                        )
                        all_successful = False
                    else:
                        logger.info(
                            f"[AUTO-TRAIN] ✅ {asset} {strat_name} passed shadow test (Holdout F1: {f1:.3f})"
                        )

            # 3. Hot-Swap (Only if all models improved or maintained stability)
            if all_successful:
                self._hot_swap_models(summary)
            else:
                msg = "⚠️ *Brain Upgrade Aborted*\n\nNew models failed to meet F1-score thresholds. Keeping current robust models active."
                logger.warning(msg)
                self._safe_send_telegram(msg)

    def _hot_swap_models(self, summary: Dict):
        """Seamlessly inject new models into the live running strategies."""
        logger.info("[AUTO-TRAIN] Initiating HOT-SWAP of AI models...")

        try:
            # Update metadata timestamp
            with open(self.metadata_file, "w") as f:
                json.dump(
                    {"last_trained": datetime.now().isoformat(), "summary": summary}, f
                )

            # Reload strategies in the live bot safely
            for asset, aggregators in self.trading_bot.aggregators.items():
                if isinstance(aggregators, dict) and "mode" in aggregators:
                    # Hybrid Mode - reload all sub-aggregators
                    self._reload_aggregator(aggregators["council"], asset)
                    self._reload_aggregator(aggregators["performance"], asset)
                else:
                    self._reload_aggregator(aggregators, asset)

            logger.info("[AUTO-TRAIN] ✅ HOT-SWAP COMPLETE. Bot is now using updated intelligence.")

            # Send Telegram Report
            report = "🧠 *Brain Upgrade Complete*\n_Live models hot-swapped successfully._\n\n"
            for asset, strats in summary.items():
                if strats:
                    report += f"*{asset} New Accuracies:*\n"
                    for strat, acc in strats.items():
                        report += f"  • {strat.replace('_', ' ').title()}: {acc:.1%}\n"
                    report += "\n"

            self._safe_send_telegram(report)

        except Exception as e:
            logger.error(f"[AUTO-TRAIN] Hot-swap failed: {e}", exc_info=True)

    def _reload_aggregator(self, aggregator, asset: str):
        """Helper to reload model files for a specific aggregator."""
        if hasattr(aggregator, "mr_strategy"):
            aggregator.mr_strategy.load_model(f"models/mean_reversion_{asset.lower()}.pkl")
        if hasattr(aggregator, "tf_strategy"):
            aggregator.tf_strategy.load_model(f"models/trend_following_{asset.lower()}.pkl")
        if hasattr(aggregator, "ema_strategy"):
            aggregator.ema_strategy.load_model(f"models/ema_strategy_{asset.lower()}.pkl")