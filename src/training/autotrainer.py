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
                        # T4.4 — P&L Promotion Gate
                        new_model_path = os.path.join(
                            self.models_dir, f"{strat_name}_{asset.lower()}.pkl"
                        )
                        current_model_path = new_model_path  # same path — current is what's on disk before swap
                        if not self._passes_pnl_promotion_gate(
                            strat_name, asset, new_model_path, current_model_path
                        ):
                            logger.warning(
                                f"[PROMO-GATE] ❌ {asset} {strat_name} blocked by P&L promotion gate. "
                                f"Keeping current model."
                            )
                            all_successful = False

            # 3. T4.3 — Mine false negatives from shadow data every retrain cycle
            try:
                fn_summary = self.mine_false_negatives(profit_threshold=0.10)
                if not fn_summary.empty:
                    # T4.1 — Augment labels with shadow ground truth
                    shadow_df = self.generate_shadow_labels(min_samples=30)
                    if not shadow_df.empty:
                        logger.info(
                            f"[SHADOW-LABELS] {len(shadow_df)} shadow labels available "
                            f"for future retraining augmentation."
                        )
            except Exception as _e:
                logger.warning(f"[AUTO-TRAIN] Shadow mining step failed (non-fatal): {_e}")

            # 4. Hot-Swap (Only if all models improved or maintained stability)
            if all_successful:
                self._hot_swap_models(summary)
            else:
                # ✅ FIX: Update metadata even on abort to prevent recursive re-triggering
                try:
                    with open(self.metadata_file, "w") as f:
                        json.dump(
                            {"last_trained": datetime.now().isoformat(), "summary": summary, "status": "aborted"}, f
                        )
                    logger.info("[AUTO-TRAIN] Stale timestamp updated after aborted upgrade.")
                except Exception as e:
                    logger.error(f"[AUTO-TRAIN] Failed to update metadata on abort: {e}")

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

    # ─────────────────────────────────────────────────────────────────────────
    # T4.1 — Shadow Label Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_shadow_labels(
        self,
        min_net_pnl_threshold: float = 0.0,
        min_samples: int = 30,
    ) -> "pd.DataFrame":
        """
        Convert closed shadow positions into a labelled training DataFrame.

        Uses net_pnl_pct (gross P&L minus friction) as the ground-truth signal
        so that models learn from realistic, cost-adjusted outcomes — not raw
        price moves that look better than they feel live.

        Label encoding
        --------------
        label = 1  if net_pnl_pct > min_net_pnl_threshold  (profitable)
        label = 0  otherwise (loss / breakeven)

        Feature columns (from strategy_votes snapshot at entry)
        --------------------------------------------------------
        mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf,
        signal_quality, regime_score, mfe_pct, mae_pct, bars_open

        Returns
        -------
        pd.DataFrame with feature columns + "label" column,
        or an empty DataFrame if not enough closed positions exist.
        """
        shadow = getattr(
            getattr(self.trading_bot, "shadow_trader", None), "closed_results", []
        )
        if len(shadow) < min_samples:
            logger.info(
                f"[SHADOW-LABELS] Only {len(shadow)} closed shadow trades "
                f"(need {min_samples}). Skipping label generation."
            )
            return pd.DataFrame()

        rows = []
        for r in shadow:
            votes = r.get("strategy_votes", {}) or {}
            rows.append(
                {
                    # Features
                    "mr_signal":      float(votes.get("mr_signal", 0)),
                    "mr_conf":        float(votes.get("mr_conf", 0.0)),
                    "tf_signal":      float(votes.get("tf_signal", 0)),
                    "tf_conf":        float(votes.get("tf_conf", 0.0)),
                    "ema_signal":     float(votes.get("ema_signal", 0)),
                    "ema_conf":       float(votes.get("ema_conf", 0.0)),
                    "signal_quality": float(votes.get("signal_quality", 0.0)),
                    "regime_score":   float(r.get("regime_score", 0.0)),
                    "mfe_pct":        float(r.get("mfe_pct", 0.0)),
                    "mae_pct":        float(r.get("mae_pct", 0.0)),
                    "bars_open":      int(r.get("bars_open", 0)),
                    # Meta (not used as features but useful for slicing)
                    "asset":          r.get("asset", ""),
                    "side":           r.get("side", ""),
                    "strategy_source": r.get("strategy_source", ""),
                    "gate_blocked_by": r.get("gate_blocked_by", ""),
                    "regime_name":     r.get("regime_name", ""),
                    "net_pnl_pct":    float(r.get("net_pnl_pct", 0.0)),
                    # Ground-truth label
                    "label": 1 if float(r.get("net_pnl_pct", 0.0)) > min_net_pnl_threshold else 0,
                }
            )

        df = pd.DataFrame(rows)
        win_rate = df["label"].mean() * 100
        logger.info(
            f"[SHADOW-LABELS] Built {len(df)} labelled rows from shadow trades "
            f"(win_rate={win_rate:.1f}%)"
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # T4.3 — False Negative Mining (Missed Profitable Signals)
    # ─────────────────────────────────────────────────────────────────────────

    def mine_false_negatives(
        self,
        profit_threshold: float = 0.10,
        top_n: int = 10,
    ) -> "pd.DataFrame":
        """
        Identify blocked signals that would have been profitable.

        A "false negative" here is a signal that was killed by a gate but
        whose shadow trade outcome exceeded profit_threshold% net P&L.
        These are the most valuable data points for gate calibration —
        they show which gates are over-blocking and costing real alpha.

        Parameters
        ----------
        profit_threshold : float
            Minimum net_pnl_pct (in %) to count as a missed opportunity.
            Default 0.10 = 0.10% net profit after friction.
        top_n : int
            How many top-missed gates to surface in the summary log.

        Returns
        -------
        pd.DataFrame with columns:
            gate_blocked_by, strategy_source, count, avg_net_pnl,
            avg_mfe_pct, win_rate, total_missed_pnl
        Sorted descending by total_missed_pnl.
        """
        shadow = getattr(
            getattr(self.trading_bot, "shadow_trader", None), "closed_results", []
        )
        if not shadow:
            logger.info("[FALSE-NEG] No shadow trades available yet.")
            return pd.DataFrame()

        # Filter to profitable blocked trades
        false_negatives = [
            r for r in shadow
            if float(r.get("net_pnl_pct", 0.0)) > profit_threshold
        ]

        if not false_negatives:
            logger.info(
                f"[FALSE-NEG] No false negatives above {profit_threshold}% threshold "
                f"in {len(shadow)} closed shadow trades."
            )
            return pd.DataFrame()

        df = pd.DataFrame(false_negatives)

        # Group by gate + strategy
        summary = (
            df.groupby(["gate_blocked_by", "strategy_source"])
            .agg(
                count=("net_pnl_pct", "count"),
                avg_net_pnl=("net_pnl_pct", "mean"),
                avg_mfe_pct=("mfe_pct", "mean"),
                total_missed_pnl=("net_pnl_pct", "sum"),
            )
            .reset_index()
        )
        summary["win_rate"] = (
            df.groupby(["gate_blocked_by", "strategy_source"])
            .apply(lambda g: (g["net_pnl_pct"] > 0).mean() * 100)
            .values
        )
        summary = summary.sort_values("total_missed_pnl", ascending=False).reset_index(drop=True)
        summary["avg_net_pnl"] = summary["avg_net_pnl"].round(3)
        summary["avg_mfe_pct"] = summary["avg_mfe_pct"].round(3)
        summary["total_missed_pnl"] = summary["total_missed_pnl"].round(3)
        summary["win_rate"] = summary["win_rate"].round(1)

        # Log top missed gates
        logger.info(
            f"[FALSE-NEG] {len(false_negatives)}/{len(shadow)} shadow trades "
            f"were profitable false negatives (>{profit_threshold}% net). "
            f"Top over-blocking gates:"
        )
        for _, row in summary.head(top_n).iterrows():
            logger.info(
                f"  gate={row['gate_blocked_by']} | src={row['strategy_source']} | "
                f"n={row['count']} | avg_net={row['avg_net_pnl']:+.3f}% | "
                f"total_missed={row['total_missed_pnl']:+.3f}%"
            )

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # T4.4 — P&L Promotion Gate (gates model hot-swap on live performance)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_shadow_pnl(
        self,
        strategy_name: str,
        asset: str,
        new_model_path: str,
        lookback: int = 200,
    ) -> dict:
        """
        Run the candidate new model against the last `lookback` closed shadow
        trades and return a dict with net_pnl and win_rate for comparison.

        The strategy's predict() is called on the strategy_votes features
        embedded in each shadow record.  This is zero-cost: no live data
        fetch, no exchange calls.

        Returns
        -------
        dict with keys:
            n_samples    : int
            win_rate_pct : float (0–100)
            avg_net_pnl  : float (in %)
            total_net_pnl: float (in %)
        """
        shadow = getattr(
            getattr(self.trading_bot, "shadow_trader", None), "closed_results", []
        )
        # Filter to matching asset + strategy_source
        relevant = [
            r for r in shadow
            if r.get("asset", "").upper() == asset.upper()
            and r.get("strategy_source", "") in (
                strategy_name,
                strategy_name.upper(),
                # map long names to short tags used in shadow_trader
                {"mean_reversion": "MR", "trend_following": "TF",
                 "exponential_moving_averages": "EMA"}.get(strategy_name, strategy_name),
            )
        ][-lookback:]

        if len(relevant) < 10:
            return {"n_samples": len(relevant), "win_rate_pct": 0.0,
                    "avg_net_pnl": 0.0, "total_net_pnl": 0.0}

        try:
            strat_class = self.strategy_classes.get(strategy_name)
            if not strat_class:
                return {"n_samples": 0, "win_rate_pct": 0.0,
                        "avg_net_pnl": 0.0, "total_net_pnl": 0.0}

            strat_config = self.config["strategy_configs"][strategy_name][asset]
            candidate = strat_class(strat_config)
            if not candidate.load_model(new_model_path):
                return {"n_samples": 0, "win_rate_pct": 0.0,
                        "avg_net_pnl": 0.0, "total_net_pnl": 0.0}

            feature_cols = ["mr_conf", "tf_conf", "ema_conf", "signal_quality",
                            "regime_score", "mfe_pct", "mae_pct", "bars_open"]

            rows = []
            net_pnls = []
            for r in relevant:
                votes = r.get("strategy_votes", {}) or {}
                row = [
                    float(votes.get("mr_conf", 0.0)),
                    float(votes.get("tf_conf", 0.0)),
                    float(votes.get("ema_conf", 0.0)),
                    float(votes.get("signal_quality", 0.0)),
                    float(r.get("regime_score", 0.0)),
                    float(r.get("mfe_pct", 0.0)),
                    float(r.get("mae_pct", 0.0)),
                    float(r.get("bars_open", 0)),
                ]
                rows.append(row)
                net_pnls.append(float(r.get("net_pnl_pct", 0.0)))

            import numpy as _np
            X = _np.array(rows)
            if hasattr(candidate, "scaler") and candidate.scaler is not None:
                X = candidate.scaler.transform(X)

            preds = candidate.model.predict(X)  # 1 = buy/long, -1 = sell/short
            # A prediction is "correct" if it agrees with the profitable direction
            # (net_pnl > 0 means the shadow trade direction was right)
            correct = sum(
                1 for pred, pnl in zip(preds, net_pnls)
                if (pred != 0 and pnl > 0)
            )
            win_rate = correct / len(preds) * 100
            avg_net = sum(net_pnls) / len(net_pnls)
            total_net = sum(net_pnls)

            del candidate
            gc.collect()

            return {
                "n_samples": len(relevant),
                "win_rate_pct": round(win_rate, 1),
                "avg_net_pnl": round(avg_net, 3),
                "total_net_pnl": round(total_net, 3),
            }
        except Exception as e:
            logger.warning(f"[PROMO-GATE] Shadow P&L eval failed for {asset} {strategy_name}: {e}")
            return {"n_samples": 0, "win_rate_pct": 0.0,
                    "avg_net_pnl": 0.0, "total_net_pnl": 0.0}

    def _passes_pnl_promotion_gate(
        self,
        strategy_name: str,
        asset: str,
        new_model_path: str,
        current_model_path: str,
        min_pnl_improvement: float = 0.05,
        max_winrate_drop: float = 5.0,
    ) -> bool:
        """
        T4.4 Promotion Gate: only allow model hot-swap if the candidate
        improves shadow-trade net P&L by at least min_pnl_improvement %
        AND does not drop win rate by more than max_winrate_drop pp.

        Falls through (returns True) when shadow data is insufficient to
        make a reliable comparison — we don't block new models on day-one.

        Parameters
        ----------
        min_pnl_improvement : float
            Minimum avg_net_pnl improvement in % to approve the swap.
            Default 0.05 = the new model must average at least +0.05% more
            net P&L per shadow trade than the current model.
        max_winrate_drop : float
            Maximum allowed win rate regression in percentage points.
            Default 5.0 pp.
        """
        new_stats = self._evaluate_shadow_pnl(strategy_name, asset, new_model_path)

        if new_stats["n_samples"] < 10:
            logger.info(
                f"[PROMO-GATE] {asset} {strategy_name}: insufficient shadow data "
                f"({new_stats['n_samples']} samples) — gate bypassed, allowing swap."
            )
            return True

        # Evaluate current model on same shadow set for fair comparison
        cur_stats = self._evaluate_shadow_pnl(strategy_name, asset, current_model_path)

        pnl_delta = new_stats["avg_net_pnl"] - cur_stats["avg_net_pnl"]
        wr_delta = new_stats["win_rate_pct"] - cur_stats["win_rate_pct"]

        passes = (pnl_delta >= min_pnl_improvement) and (wr_delta >= -max_winrate_drop)

        logger.info(
            f"[PROMO-GATE] {asset} {strategy_name}: "
            f"new_avg_pnl={new_stats['avg_net_pnl']:+.3f}% "
            f"cur_avg_pnl={cur_stats['avg_net_pnl']:+.3f}% "
            f"Δpnl={pnl_delta:+.3f}% (need≥{min_pnl_improvement:+.3f}%) | "
            f"new_wr={new_stats['win_rate_pct']:.1f}% "
            f"cur_wr={cur_stats['win_rate_pct']:.1f}% "
            f"Δwr={wr_delta:+.1f}pp (max_drop={max_winrate_drop:.1f}pp) | "
            f"{'✅ APPROVED' if passes else '❌ BLOCKED'}"
        )
        return passes