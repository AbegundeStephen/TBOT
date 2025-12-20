"""
Integration: Veteran Trade Manager with Backtrader
Replace simple stop loss with professional risk management
"""

import backtrader as bt
import numpy as np
import pandas as pd
import logging
import json
from src.execution.veteran_trade_manager import VeteranTradeManager, ExitReason

logger = logging.getLogger(__name__)


class MLStrategyWithVTM(bt.Strategy):
    """
    Enhanced ML Strategy using Veteran Trade Manager
    for professional risk management
    """

    params = (
        # Signal generation params (keep your existing)
        ("lookback", 100),
        ("aggregator_preset", "balanced"),
        ("use_ai_validation", True),
        # AI validation params (from your MLStrategy)
        ("ai_sr_threshold", 0.015),
        ("ai_pattern_confidence", 0.50),
        ("ai_enable_adaptive", True),
        ("ai_strong_signal_bypass", 0.85),
        ("ai_circuit_breaker_threshold", 0.70),
        ("atr_period", 14),
        # VTM Risk Management Parameters
        ("account_risk", 0.015),  # 1.5% risk per trade
        ("partial_targets", [1.5, 2.5, 4.0]),  # R:R ratios
        ("partial_sizes", [0.33, 0.33, 0.34]),  # Exit 33%/33%/34%
        ("min_stop_distance_pct", 0.12),  # 12% min stop
        ("max_stop_distance_pct", 0.20),  # 20% max stop
        ("runner_trail_pct", 0.25),  # 25% trail for runner
        ("use_ema_exit", True),
        ("ema_period", 21),
        ("time_stop_bars", 20),
        # Backtrader integration
        ("max_position_pct", 0.95),
    )

    def __init__(self):
        # Set asset key (keep your existing logic)
        self.asset_key = getattr(self.__class__, "asset_key", "btc").upper()

        # ===================================================================
        # COPY YOUR ENTIRE MLStrategy.__init__() CODE HERE
        # ===================================================================

        # Load config
        with open("config/config.json") as f:
            config = json.load(f)
        self.config = config

        # Set model paths dynamically based on asset
        mean_rev_model_path = f"models/mean_reversion_{self.asset_key.lower()}.pkl"
        trend_model_path = f"models/trend_following_{self.asset_key.lower()}.pkl"
        ema_model_path = f"models/ema_strategy_{self.asset_key.lower()}.pkl"

        # Load strategy configs from config.json
        mr_config = config["strategy_configs"]["mean_reversion"][self.asset_key]
        tf_config = config["strategy_configs"]["trend_following"][self.asset_key]
        ema_config = config["strategy_configs"]["exponential_moving_averages"][
            self.asset_key
        ]

        # ATR for VTM and your existing logic
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.params.atr_period if hasattr(self.params, "atr_period") else 14,
        )
        self.atr_indicator = self.atr  # Alias for compatibility

        # Initialize strategies
        from src.strategies.mean_reversion import MeanReversionStrategy
        from src.strategies.trend_following import TrendFollowingStrategy
        from src.strategies.ema_strategy import EMAStrategy

        self.mean_reversion = MeanReversionStrategy(mr_config)
        self.trend_following = TrendFollowingStrategy(tf_config)
        self.ema_strategy = EMAStrategy(ema_config)

        # Initialize AI validator if enabled
        self.ai_validator = None
        if self.params.use_ai_validation:
            self.ai_validator = self._initialize_ai_layer()

        # Load trained models
        mr_loaded = self.mean_reversion.load_model(mean_rev_model_path)
        tf_loaded = self.trend_following.load_model(trend_model_path)
        ema_loaded = self.ema_strategy.load_model(ema_model_path)

        if not (mr_loaded and tf_loaded and ema_loaded):
            raise RuntimeError("Failed to load one or more strategy models")

        # Get asset-specific preset configuration
        from backtest import AGGREGATOR_PRESETS  # Import your presets

        preset_name = self.params.aggregator_preset
        if preset_name not in AGGREGATOR_PRESETS[self.asset_key]:
            logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset_name = "balanced"

        confidence_config = AGGREGATOR_PRESETS[self.asset_key][preset_name].copy()

        # Initialize PerformanceWeightedAggregator
        from src.execution.signal_aggregator import PerformanceWeightedAggregator

        self.aggregator = PerformanceWeightedAggregator(
            mean_reversion_strategy=self.mean_reversion,
            trend_following_strategy=self.trend_following,
            ema_strategy=self.ema_strategy,
            asset_type=self.asset_key,
            config=confidence_config,
            ai_validator=self.ai_validator if self.params.use_ai_validation else None,
            enable_ai_circuit_breaker=True,
            enable_detailed_logging=True,
            strong_signal_bypass_threshold=getattr(
                self.params, "ai_strong_signal_bypass", 0.85
            ),
        )

        # Original MLStrategy tracking variables
        self.next_call_count = 0
        self.signal_log = []
        self.ai_stats = {
            "total_signals": 0,
            "ai_approved": 0,
            "ai_rejected": 0,
            "rejected_no_sr": 0,
            "rejected_no_pattern": 0,
        }

        # ===================================================================
        # END OF COPIED CODE
        # ===================================================================

        # VTM-specific initialization
        # Trade manager instance (created on each trade)
        self.trade_manager = None

        # Track orders
        self.order = None
        self.partial_orders = []  # Track multiple exit orders

        # Statistics
        self.vtm_stats = {
            "total_trades": 0,
            "partial_exits": 0,
            "stop_loss_exits": 0,
            "take_profit_exits": 0,
            "time_stop_exits": 0,
            "structure_exits": 0,
        }

        logger.info("=" * 70)
        logger.info("🎯 VETERAN TRADE MANAGER INTEGRATION")
        logger.info("=" * 70)
        logger.info(f"Asset: {self.asset_key}")
        logger.info(f"Preset: {preset_name}")
        logger.info(
            f"AI Validation: {'ENABLED' if self.params.use_ai_validation else 'DISABLED'}"
        )
        logger.info(f"─" * 70)
        logger.info(f"VTM Settings:")
        logger.info(f"  Partial Targets: {self.params.partial_targets}")
        logger.info(f"  Partial Sizes: {self.params.partial_sizes}")
        logger.info(
            f"  Stop Range: {self.params.min_stop_distance_pct:.0%} - {self.params.max_stop_distance_pct:.0%}"
        )
        logger.info(f"  Runner Trail: {self.params.runner_trail_pct:.0%}")
        logger.info(f"  Account Risk: {self.params.account_risk:.1%} per trade")
        logger.info("=" * 70)

    def _initialize_ai_layer(self):
        """Copy from your MLStrategy (lines 121-200 in your backtest.py)"""
        try:
            import pickle
            from pathlib import Path
            from src.ai import DynamicAnalyst, OHLCSniper, HybridSignalValidator

            models_dir = Path("models/ai")
            model_path = models_dir / "sniper_btc_gold_v2.weights.h5"
            mapping_path = models_dir / "sniper_btc_gold_v2_mapping.pkl"
            config_path = models_dir / "sniper_btc_gold_v2_config.pkl"

            if not model_path.exists():
                logger.warning(f"[AI] Model not found: {model_path}")
                logger.warning("[AI] Backtesting WITHOUT AI validation")
                return None

            # Load mappings
            with open(mapping_path, "rb") as f:
                pattern_map = pickle.load(f)
            with open(config_path, "rb") as f:
                ai_config = pickle.load(f)

            logger.info(f"[AI] Loaded {len(pattern_map)} patterns")
            logger.info(f"[AI] Model accuracy: {ai_config['validation_accuracy']:.2%}")

            # Initialize components
            analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
            sniper = OHLCSniper(
                input_shape=(15, 4), num_classes=ai_config["num_classes"]
            )
            sniper.load_model(str(model_path))

            # Use params or defaults
            sr_threshold = getattr(self.params, "ai_sr_threshold", 0.015)
            pattern_conf = getattr(self.params, "ai_pattern_confidence", 0.50)
            strong_bypass = getattr(self.params, "ai_strong_signal_bypass", 0.85)
            circuit_breaker = getattr(self.params, "ai_circuit_breaker_threshold", 0.70)

            validator = HybridSignalValidator(
                analyst=analyst,
                sniper=sniper,
                pattern_id_map=pattern_map,
                sr_threshold_pct=sr_threshold,
                pattern_confidence_min=pattern_conf,
                use_ai_validation=True,
                enable_adaptive_thresholds=getattr(
                    self.params, "ai_enable_adaptive", True
                ),
                strong_signal_bypass_threshold=strong_bypass,
                circuit_breaker_threshold=circuit_breaker,
                enable_detailed_logging=False,
            )

            logger.info("[AI] ✓ Enhanced validation layer initialized")
            logger.info(f"  S/R Threshold: {sr_threshold:.2%}")
            logger.info(f"  Pattern Confidence: {pattern_conf:.0%}")
            logger.info(f"  Strong Signal Bypass: {strong_bypass:.0%}")

            return validator

        except Exception as e:
            logger.error(f"[AI] Failed to initialize: {e}")
            logger.warning("[AI] Backtesting WITHOUT AI validation")
            return None

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                logger.info(
                    f"✅ BUY EXECUTED @ ${order.executed.price:.2f} "
                    f"| Size: {order.executed.size:.6f}"
                )
            elif order.issell():
                # Check if this was a partial exit
                is_partial = order in self.partial_orders
                exit_type = "PARTIAL" if is_partial else "FULL"

                logger.info(
                    f"✅ {exit_type} SELL @ ${order.executed.price:.2f} "
                    f"| Size: {order.executed.size:.6f}"
                )

                # Remove from partial tracking
                if is_partial and order in self.partial_orders:
                    self.partial_orders.remove(order)

                # If no position left, clean up
                if not self.position:
                    self._cleanup_trade_manager()

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"⚠️  Order {order.ref} - Status: {order.getstatusname()}")

        # Clear order reference if it was the main order
        if order == self.order:
            self.order = None

    def notify_trade(self, trade):
        """Track completed trades"""
        if trade.isclosed:
            pnl_pct = (trade.pnl / trade.value) * 100 if trade.value else 0
            self.vtm_stats["total_trades"] += 1

            logger.info(
                f"💰 TRADE CLOSED | "
                f"P&L: ${trade.pnl:.2f} ({pnl_pct:+.2f}%) | "
                f"Net: ${trade.pnlcomm:.2f}"
            )

    def _cleanup_trade_manager(self):
        """Clean up after trade closes"""
        if self.trade_manager:
            logger.info(
                f"[VTM] Trade completed | "
                f"Partials hit: {len(self.trade_manager.partials_hit)}/{len(self.trade_manager.partial_targets)} | "
                f"Bars: {self.trade_manager.bars_in_trade}"
            )

        self.trade_manager = None
        self.partial_orders = []

    def _initialize_trade_manager(self, entry_price: float, signal_direction: int):
        """
        Initialize VTM when entering a trade

        Args:
            entry_price: Price at which we're entering
            signal_direction: 1 for long, -1 for short
        """
        try:
            # Prepare historical data for VTM
            lookback = 100
            high = np.array([x for x in self.data.high.get(size=lookback)])
            low = np.array([x for x in self.data.low.get(size=lookback)])
            close = np.array([x for x in self.data.close.get(size=lookback)])

            # Get account balance
            account_balance = self.broker.getcash()

            # Determine side
            side = "long" if signal_direction == 1 else "short"

            # Create VTM instance
            self.trade_manager = VeteranTradeManager(
                entry_price=entry_price,
                side=side,
                high=high,
                low=low,
                close=close,
                account_balance=account_balance,
                account_risk=self.params.account_risk,
                partial_targets=list(self.params.partial_targets),
                partial_sizes=list(self.params.partial_sizes),
                min_stop_distance_pct=self.params.min_stop_distance_pct,
                max_stop_distance_pct=self.params.max_stop_distance_pct,
                runner_trail_pct=self.params.runner_trail_pct,
                use_ema_exit=self.params.use_ema_exit,
                ema_period=self.params.ema_period,
                time_stop_bars=self.params.time_stop_bars,
            )

            logger.info(
                f"[VTM] Initialized | "
                f"Entry: ${entry_price:.2f} | "
                f"SL: ${self.trade_manager.initial_stop_loss:.2f} | "
                f"Targets: {[f'${t:.2f}' for t in self.trade_manager.take_profit_levels]}"
            )

            return True

        except Exception as e:
            logger.error(f"[VTM] Initialization failed: {e}", exc_info=True)
            return False

    def _update_trade_manager(self):
        """
        Update VTM on each new bar
        Returns exit_info dict if exit triggered
        """
        if not self.trade_manager or not self.position:
            return None

        try:
            # Update VTM with new bar
            exit_info = self.trade_manager.on_new_bar(
                new_high=self.data.high[0],
                new_low=self.data.low[0],
                new_close=self.data.close[0],
            )

            return exit_info

        except Exception as e:
            logger.error(f"[VTM] Update error: {e}", exc_info=True)
            return None

    def _execute_vtm_exit(self, exit_info: dict):
        """
        Execute exit based on VTM signal

        Args:
            exit_info: {'reason': ExitReason, 'price': float, 'size': float}
        """
        try:
            reason = exit_info["reason"]
            size_pct = exit_info["size"]
            current_price = exit_info["price"]

            # Calculate actual size to close
            current_position_size = abs(self.position.size)
            close_size = current_position_size * size_pct

            # Track exit type
            if reason == ExitReason.STOP_LOSS:
                self.vtm_stats["stop_loss_exits"] += 1
                logger.info(f"🛑 VTM: Stop Loss triggered @ ${current_price:.2f}")

            elif reason in [
                ExitReason.TAKE_PROFIT_1,
                ExitReason.TAKE_PROFIT_2,
                ExitReason.TAKE_PROFIT_3,
            ]:
                self.vtm_stats["partial_exits"] += 1
                self.vtm_stats["take_profit_exits"] += 1
                logger.info(
                    f"🎯 VTM: {reason.value} hit @ ${current_price:.2f} "
                    f"| Closing {size_pct:.0%}"
                )

            elif reason == ExitReason.TRAILING_STOP:
                self.vtm_stats["stop_loss_exits"] += 1
                logger.info(f"📉 VTM: Runner trailing stop @ ${current_price:.2f}")

            elif reason == ExitReason.STRUCTURE_BREAK:
                self.vtm_stats["structure_exits"] += 1
                logger.info(f"📊 VTM: Structure break (EMA) @ ${current_price:.2f}")

            elif reason == ExitReason.TIME_STOP:
                self.vtm_stats["time_stop_exits"] += 1
                logger.info(f"⏰ VTM: Time stop (dead trade) @ ${current_price:.2f}")

            # Execute the close
            if size_pct >= 0.99:  # Close entire position
                order = self.close()
            else:  # Partial close
                order = (
                    self.sell(size=close_size)
                    if self.position.size > 0
                    else self.buy(size=close_size)
                )
                self.partial_orders.append(order)

            return order

        except Exception as e:
            logger.error(f"[VTM] Exit execution error: {e}", exc_info=True)
            return None

    def next(self):
        """Main strategy logic - UPDATED with VTM"""

        # Skip if pending order
        if self.order:
            return

        # Minimum data check
        if len(self.data) < self.params.lookback:
            return

        try:
            current_price = self.data.close[0]

            # === IF IN POSITION: Update VTM ===
            if self.position and self.trade_manager:
                exit_info = self._update_trade_manager()

                if exit_info:
                    self._execute_vtm_exit(exit_info)
                    return

            # === IF NOT IN POSITION: Check for entry signals ===
            if not self.position:
                # Prepare data for signal generation (your existing code)
                df = pd.DataFrame(
                    {
                        "open": [
                            x for x in self.data.open.get(size=self.params.lookback)
                        ],
                        "high": [
                            x for x in self.data.high.get(size=self.params.lookback)
                        ],
                        "low": [
                            x for x in self.data.low.get(size=self.params.lookback)
                        ],
                        "close": [
                            x for x in self.data.close.get(size=self.params.lookback)
                        ],
                        "volume": [
                            x for x in self.data.volume.get(size=self.params.lookback)
                        ],
                    }
                )

                # Get signal from aggregator (your existing code)
                signal, details = self.aggregator.get_aggregated_signal(df)

                # Execute BUY signal
                if signal == 1:
                    # Initialize VTM FIRST to calculate position size
                    if self._initialize_trade_manager(current_price, signal):
                        # Use VTM's calculated position size
                        size = self.trade_manager.position_size

                        # Adjust for max position constraint
                        max_value = self.broker.getcash() * self.params.max_position_pct
                        max_size = max_value / current_price
                        size = min(size, max_size)

                        if size > 0:
                            self.order = self.buy(size=size)

                            logger.info(
                                f"🟢 ENTRY SIGNAL | "
                                f"${current_price:.2f} | "
                                f"Size: {size:.6f} | "
                                f"Reason: {details.get('reasoning', 'N/A')}"
                            )

        except Exception as e:
            logger.error(f"❌ Error in next(): {e}", exc_info=True)

    def stop(self):
        """Print strategy statistics at end"""
        logger.info("=" * 70)
        logger.info("🏁 VTM STRATEGY COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Trades: {self.vtm_stats['total_trades']}")
        logger.info(f"Partial Exits: {self.vtm_stats['partial_exits']}")
        logger.info(f"Stop Loss Exits: {self.vtm_stats['stop_loss_exits']}")
        logger.info(f"Take Profit Exits: {self.vtm_stats['take_profit_exits']}")
        logger.info(f"Structure Exits: {self.vtm_stats['structure_exits']}")
        logger.info(f"Time Stop Exits: {self.vtm_stats['time_stop_exits']}")

        if self.trade_manager:
            logger.info(f"\nLast Trade Status:")
            logger.info(f"  Partials Hit: {len(self.trade_manager.partials_hit)}")
            logger.info(f"  Bars in Trade: {self.trade_manager.bars_in_trade}")

        logger.info("=" * 70)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Replace your existing MLStrategy with MLStrategyWithVTM

    In backtest.py, change:
        cerebro.addstrategy(MLStrategy, ...)

    To:
        cerebro.addstrategy(MLStrategyWithVTM, ...)
    """

    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(
        """
    
    ╔════════════════════════════════════════════════════════════════╗
    ║  VETERAN TRADE MANAGER - BACKTRADER INTEGRATION                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    CHANGES REQUIRED:
    
    1. Save VTM code as: veteran_trade_manager.py
    
    2. In backtest.py, replace:
       
       class MLStrategy(bt.Strategy):
           ...
       
       With:
       
       from vtm_integration import MLStrategyWithVTM
       
       class MLStrategy(MLStrategyWithVTM):
           # Keep your __init__ for aggregator/AI setup
           # Remove manual stop loss logic from next()
    
    3. Run backtest:
       python backtest.py --asset BTC --preset balanced
    
    ════════════════════════════════════════════════════════════════
    
    WHAT CHANGES:
    
    ✅ BEFORE (Simple):
       -  % stop loss (e.g., -2%)
       -  % take profit (e.g., +8%)
       - All-or-nothing exits
       - No partial profits
       - No trailing stops
    
    ✅ AFTER (VTM):
       - Structure-based initial stop (12-20%)
       - Partial exits: 33% @ 1.5R, 33% @ 2.5R, 34% runner
       - Runner trails with 25% stop
       - Time-based exit for dead trades
       - EMA-based structure exits
       - Locks profit after 1st partial
    
    ════════════════════════════════════════════════════════════════
    
    EXPECTED IMPROVEMENTS:
    
    📈 Win Rate: May drop slightly (wider stops)
    📈 Avg Win: Should increase significantly (partials + runner)
    📉 Avg Loss: May increase slightly (wider stops)
    📈 Profit Factor: Should improve (bigger winners)
    📉 Max Drawdown: Should decrease (profit locking)
    📈 Sharpe Ratio: Should improve (better risk/reward)
    
    ════════════════════════════════════════════════════════════════
    """
    )
