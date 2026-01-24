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
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any

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
        self.retrain_frequency_days = config.get("ml", {}).get("retrain_frequency_days", 7)
        self.min_accuracy_threshold = config.get("ml", {}).get("min_training_accuracy", 0.55)
        self.models_dir = "models"
        self.metadata_file = os.path.join(self.models_dir, "training_metadata.json")

        logger.info(f"[AUTO-TRAIN] Pipeline initialized. Retraining every {self.retrain_frequency_days} days.")

    def start(self):
        """Start the background pipeline thread."""
        if self.is_running:
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._training_loop, daemon=True, name="AutoTrainer")
        self._thread.start()
        logger.info("[AUTO-TRAIN] ✅ Background learning thread started")

    def stop(self):
        """Stop the background pipeline."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("[AUTO-TRAIN] 🛑 Background learning thread stopped")

    def _get_last_training_time(self) -> datetime:
        """Read the last training time from metadata."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    metadata = json.load(f)
                    last_time = datetime.fromisoformat(metadata.get("last_trained", "2000-01-01T00:00:00"))
                    return last_time
        except Exception as e:
            logger.warning(f"[AUTO-TRAIN] Could not read metadata: {e}")
        return datetime.min

    def _training_loop(self):
        """Main loop that checks if retraining is needed."""
        while self.is_running:
            try:
                last_trained = self._get_last_training_time()
                next_training = last_trained + timedelta(days=self.retrain_frequency_days)
                now = datetime.now()

                # Trigger training if it's time (usually scheduled for weekends/low-volatility periods)
                if now >= next_training and now.weekday() == 6:  # 6 = Sunday
                    logger.info("\n" + "="*70)
                    logger.info(f"🧠 [AUTO-TRAIN] SCHEDULED BRAIN UPGRADE INITIATED")
                    logger.info("="*70)
                    
                    self._run_training_pipeline()
                
                # Sleep for 1 hour before checking again
                time.sleep(3600)

            except Exception as e:
                logger.error(f"[AUTO-TRAIN] Fatal error in training loop: {e}", exc_info=True)
                time.sleep(3600) # Sleep on error

    def _fetch_latest_data(self, asset: str) -> pd.DataFrame:
        """Fetch the most recent data block for training."""
        asset_cfg = self.config["assets"][asset]
        lookback_days = asset_cfg.get("lookback_days", 730)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        logger.info(f"[AUTO-TRAIN] Fetching {lookback_days} days of data for {asset}...")
        
        if asset_cfg["exchange"] == "binance":
            df = self.data_manager.fetch_binance_data(
                symbol=asset_cfg["symbol"],
                interval=asset_cfg["interval"],
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df = self.data_manager.fetch_mt5_data(
                symbol=asset_cfg["symbol"],
                timeframe=asset_cfg["timeframe"],
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        return df

    def _train_strategy(self, strategy_class, asset: str, df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """Train a single strategy in isolation (Shadow Mode)."""
        logger.info(f"[AUTO-TRAIN] Training {strategy_name} for {asset}...")
        
        # Instantiate fresh strategy for training (won't affect live bot)
        temp_strategy = strategy_class(asset_type=asset, config=self.config)
        
        # Train
        results = temp_strategy.train(df)
        
        # Free memory
        del temp_strategy
        gc.collect()
        
        return results

    def _run_training_pipeline(self):
        """Execute the full retrain and hot-swap procedure."""
        with self._lock:
            if self.telegram:
                asyncio.run(self.telegram.send_notification("🧠 *Auto-Trainer Initiated*\n\nFetching new market data and retraining ML models..."))

            summary = {"BTC": {}, "GOLD": {}}
            all_successful = True

            strategies = [
                (MeanReversionStrategy, "mean_reversion"),
                (TrendFollowingStrategy, "trend_following"),
                (EMAStrategy, "ema_strategy")
            ]

            for asset in ["BTC", "GOLD"]:
                if not self.config["assets"][asset]["enabled"]:
                    continue

                # 1. Fetch Latest Data
                df = self._fetch_latest_data(asset)
                if df is None or len(df) < 5000:
                    logger.error(f"[AUTO-TRAIN] Insufficient data for {asset}, skipping.")
                    continue

                # 2. Train Each Strategy in Shadow Mode
                for strat_class, strat_name in strategies:
                    results = self._train_strategy(strat_class, asset, df, strat_name)
                    
                    accuracy = results.get("cv_mean_accuracy", 0)
                    summary[asset][strat_name] = accuracy

                    # SHADOW TEST: Check if the new model meets minimum standards
                    if accuracy < self.min_accuracy_threshold:
                        logger.warning(f"[AUTO-TRAIN] ❌ {asset} {strat_name} failed shadow test (Acc: {accuracy:.1%}). Aborting swap.")
                        all_successful = False
                    else:
                        logger.info(f"[AUTO-TRAIN] ✅ {asset} {strat_name} passed shadow test (Acc: {accuracy:.1%})")

            # 3. Hot-Swap (Only if all models improved or maintained stability)
            if all_successful:
                self._hot_swap_models(summary)
            else:
                msg = "⚠️ *Brain Upgrade Aborted*\n\nNew models failed to meet accuracy thresholds. Keeping current robust models active."
                logger.warning(msg)
                if self.telegram:
                    asyncio.run(self.telegram.send_notification(msg))

    def _hot_swap_models(self, summary: Dict):
        """Seamlessly inject new models into the live running strategies."""
        logger.info("[AUTO-TRAIN] Initiating HOT-SWAP of AI models...")

        try:
            # Update metadata timestamp
            with open(self.metadata_file, "w") as f:
                json.dump({"last_trained": datetime.now().isoformat(), "summary": summary}, f)

            # Reload strategies in the live bot
            for asset, aggregators in self.trading_bot.aggregators.items():
                if isinstance(aggregators, dict) and 'mode' in aggregators:
                    # Hybrid Mode - reload all sub-aggregators
                    self._reload_aggregator(aggregators['council'], asset)
                    self._reload_aggregator(aggregators['performance'], asset)
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
            
            if self.telegram:
                asyncio.run(self.telegram.send_notification(report))

        except Exception as e:
            logger.error(f"[AUTO-TRAIN] Hot-swap failed: {e}", exc_info=True)

    def _reload_aggregator(self, aggregator, asset: str):
        """Helper to reload model files for a specific aggregator."""
        if hasattr(aggregator, 'mr_strategy'):
            aggregator.mr_strategy.load_model(f"models/mean_reversion_{asset.lower()}.pkl")
        if hasattr(aggregator, 'tf_strategy'):
            aggregator.tf_strategy.load_model(f"models/trend_following_{asset.lower()}.pkl")
        if hasattr(aggregator, 'ema_strategy'):
            aggregator.ema_strategy.load_model(f"models/ema_strategy_{asset.lower()}.pkl")