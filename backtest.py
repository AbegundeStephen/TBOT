#!/usr/bin/env python3
"""
Backtesting Script - Properly integrates BullMarketFilteredAggregator with risk management
"""
import json
import logging
import argparse
from pathlib import Path
import sys
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy
from src.execution.signal_aggregator import BullMarketFilteredAggregator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Preset configurations for different trading styles
AGGREGATOR_PRESETS = {
    "conservative": {
        "mean_reversion_weight": 0.6,
        "trend_following_weight": 1.2,
        "ema_weight": 1.2,
        "buy_score_threshold": 0.45,
        "sell_score_threshold": 0.50,
        "perfect_agreement_bonus": 0.15,
        "allow_single_mr_signal": False,
        "allow_single_tf_signal": True,
        "allow_single_ema_signal": True,
        "single_tf_threshold": 0.65,
        "single_ema_threshold": 0.65,
        "enable_bull_filter": True,
        "block_sells_in_bull": False,
        "boost_buys_in_bull": 0.10,
        "sell_penalty_in_bull": 0.15,
        "regime_confirmation_bars": 3,
        "regime_cooldown_hours": 6,
        "verbose_logging": True,
    },
    "balanced": {
        "mean_reversion_weight": 0.7,
        "trend_following_weight": 1.1,
        "ema_weight": 1.2,
        "buy_score_threshold": 0.35,
        "sell_score_threshold": 0.40,
        "perfect_agreement_bonus": 0.15,
        "allow_single_mr_signal": False,
        "allow_single_tf_signal": True,
        "allow_single_ema_signal": True,
        "single_tf_threshold": 0.60,
        "single_ema_threshold": 0.60,
        "enable_bull_filter": True,
        "block_sells_in_bull": False,
        "boost_buys_in_bull": 0.15,
        "sell_penalty_in_bull": 0.20,
        "regime_confirmation_bars": 2,
        "regime_cooldown_hours": 6,
        "verbose_logging": True,
    },
    "aggressive": {
        "mean_reversion_weight": 0.8,
        "trend_following_weight": 1.0,
        "ema_weight": 1.1,
        "buy_score_threshold": 0.30,
        "sell_score_threshold": 0.35,
        "perfect_agreement_bonus": 0.12,
        "allow_single_mr_signal": False,
        "allow_single_tf_signal": True,
        "allow_single_ema_signal": True,
        "single_tf_threshold": 0.55,
        "single_ema_threshold": 0.55,
        "enable_bull_filter": True,
        "block_sells_in_bull": False,
        "boost_buys_in_bull": 0.15,
        "sell_penalty_in_bull": 0.15,
        "regime_confirmation_bars": 1,
        "regime_cooldown_hours": 3,
        "verbose_logging": True,
    },
}


class MLStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper using BullMarketFilteredAggregator with risk management
    """

    params = (
        ("stop_loss_pct", 0.03),  # 3% stop (was 1% - too tight!)
        ("take_profit_pct", 0.06),  # 6% target (2:1 reward/risk)
        ("trailing_stop_pct", 0.02),  # 2% trailing stop
        ("risk_per_trade", 0.02),  # 2% account risk per trade
        # Position sizing
        ("max_position_pct", 0.95),  # Max 95% of capital
        ("use_atr_sizing", True),  # Volatility-adjusted sizing
        ("atr_period", 14),
        ("atr_multiplier", 1.5),  # ATR-based stop distance
        # Strategy config
        ("lookback", 100),
        ("aggregator_mode", "weighted_voting"),
        ("aggregator_preset", "balanced"),
        # Exit management
        ("use_trailing_stop", True),
        ("exit_on_opposite_signal", True),
    )

    def __init__(self):
        # Set asset key from class attribute
        self.asset_key = getattr(self.__class__, "asset_key", "btc").lower()

        # Load config
        with open("config/config.json") as f:
            config = json.load(f)

        self.config = config
        # Set model paths dynamically based on asset
        mean_rev_model_path = f"models/mean_reversion_{self.asset_key}.pkl"
        trend_model_path = f"models/trend_following_{self.asset_key}.pkl"
        ema_model_path = f"models/ema_strategy_{self.asset_key}.pkl"

        # Load strategy configs from config.json
        mr_config = config["strategy_configs"]["mean_reversion"][self.asset_key.upper()]
        tf_config = config["strategy_configs"]["trend_following"][
            self.asset_key.upper()
        ]
        ema_config = config["strategy_configs"]["exponential_moving_averages"][
            self.asset_key.upper()
        ]

        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.trailing_stop_price = None
        self.highest_price_since_entry = None
        # Initialize strategies
        self.mean_reversion = MeanReversionStrategy(mr_config)
        self.trend_following = TrendFollowingStrategy(tf_config)
        self.ema_strategy = EMAStrategy(ema_config)

        # Load trained models
        mr_loaded = self.mean_reversion.load_model(mean_rev_model_path)
        tf_loaded = self.trend_following.load_model(trend_model_path)
        ema_loaded = self.ema_strategy.load_model(ema_model_path)

        if not (mr_loaded and tf_loaded and ema_loaded):
            raise RuntimeError("Failed to load one or more strategy models")

        # Get aggregator configuration from preset
        preset_name = self.params.aggregator_preset
        if preset_name not in AGGREGATOR_PRESETS:
            logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset_name = "balanced"

        confidence_config = AGGREGATOR_PRESETS[preset_name].copy()

        # Initialize BullMarketFilteredAggregator with weighted voting
        self.aggregator = BullMarketFilteredAggregator(
            mean_reversion_strategy=self.mean_reversion,
            trend_following_strategy=self.trend_following,
            ema_strategy=self.ema_strategy,
            confidence_config=confidence_config,
            asset_name=self.asset_key.upper(),
        )

        self.order = None
        self.trade_count = 0
        self.signal_log = []
        self.next_call_count = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

        logger.info(f"=" * 70)
        logger.info(f" Strategy Configuration for {self.asset_key.upper()}")
        logger.info(f"=" * 70)
        logger.info(f"Risk Management:")
        logger.info(f"  Stop-Loss: {self.params.stop_loss_pct * 100}% (: was 1%)")
        logger.info(f"  Take-Profit: {self.params.take_profit_pct * 100}%")
        logger.info(
            f"  Reward/Risk: {self.params.take_profit_pct/self.params.stop_loss_pct:.1f}:1"
        )
        logger.info(f"  Risk per Trade: {self.params.risk_per_trade * 100}%")

        if self.params.use_trailing_stop:
            logger.info(f"  Trailing Stop: {self.params.trailing_stop_pct * 100}%")
        logger.info(f"Position Sizing:")
        logger.info(f"  ATR-based: {self.params.use_atr_sizing}")
        if self.params.use_atr_sizing:
            logger.info(f"  ATR Period: {self.params.atr_period}")
            logger.info(f"  ATR Multiplier: {self.params.atr_multiplier}x")
        logger.info(f"=" * 70)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                logger.info(
                    f"✅ BUY EXECUTED - Price: ${order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.8f}"
                )
            elif order.issell():
                logger.info(
                    f"✅ SELL EXECUTED - Price: ${order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.8f}"
                )
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                self.trailing_stop_price = None
                self.highest_price_since_entry = None

            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl_pct = (trade.pnl / trade.value) * 100 if trade.value else 0
            logger.info(
                f"💰 TRADE CLOSED - PnL: ${trade.pnl:.2f} ({pnl_pct:+.2f}%), "
                f"Net: ${trade.pnlcomm:.2f}"
            )
            self.trade_count += 1

    def next(self):
        self.next_call_count += 1

        if self.order:
            return

        if len(self.data) < self.params.lookback:
            return

        try:
            current_price = self.data.close[0]

            # Check stops ONLY if position exists
            if self.position and self.position.size > 0:
                # Update trailing stop
                if self.params.use_trailing_stop:
                    self.update_trailing_stop()

                # Check all exit conditions
                hit_stop_loss = self.stop_loss and current_price <= self.stop_loss
                hit_take_profit = self.take_profit and current_price >= self.take_profit
                hit_trailing_stop = (
                    self.params.use_trailing_stop
                    and self.trailing_stop_price
                    and current_price <= self.trailing_stop_price
                )

                if hit_stop_loss:
                    self.order = self.close()
                    pct_loss = (
                        (current_price - self.entry_price) / self.entry_price
                    ) * 100
                    logger.info(
                        f"🛑 STOP-LOSS at ${current_price:.2f} "
                        f"({pct_loss:+.2f}%) | Entry: ${self.entry_price:.2f}"
                    )
                    return
                elif hit_take_profit:
                    self.order = self.close()
                    pct_gain = (
                        (current_price - self.entry_price) / self.entry_price
                    ) * 100
                    logger.info(
                        f"🎯 TAKE-PROFIT at ${current_price:.2f} "
                        f"({pct_gain:+.2f}%) | Entry: ${self.entry_price:.2f}"
                    )
                    return
                elif hit_trailing_stop:
                    self.order = self.close()
                    pct_change = (
                        (current_price - self.entry_price) / self.entry_price
                    ) * 100
                    logger.info(
                        f"📉 TRAILING STOP at ${current_price:.2f} "
                        f"({pct_change:+.2f}%) | High: ${self.highest_price_since_entry:.2f}"
                    )
                    return

            # Prepare data
            df = pd.DataFrame(
                {
                    "open": [x for x in self.data.open.get(size=self.params.lookback)],
                    "high": [x for x in self.data.high.get(size=self.params.lookback)],
                    "low": [x for x in self.data.low.get(size=self.params.lookback)],
                    "close": [
                        x for x in self.data.close.get(size=self.params.lookback)
                    ],
                    "volume": [
                        x for x in self.data.volume.get(size=self.params.lookback)
                    ],
                }
            )

            if len(df) < self.params.lookback:
                return

            # Get signal
            signal, details = self.aggregator.get_aggregated_signal(df)

            # Log periodically
            if self.next_call_count % 5 == 0:
                self.signal_log.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": current_price,
                        "signal": signal,
                        "details": details,
                    }
                )

                buy_score = details.get("buy_score", 0)
                sell_score = details.get("sell_score", 0)
                regime = details.get("regime", "UNKNOWN")

                logger.info(
                    f"📍 {self.data.datetime.date(0)} | "
                    f"${current_price:.2f} | {regime} | "
                    f"Score B/S: {buy_score:.2f}/{sell_score:.2f} | "
                    f"Signal: {signal:>2} | "
                    f"Quality: {details.get('signal_quality', 0):.2f}"
                )

            # Execute trades
            if not self.position:
                if signal == 1:  # BUY
                    size = self.calculate_position_size(signal)
                    if size > 0:
                        self.order = self.buy(size=size)

                        # Set stops based on ATR if enabled
                        if self.params.use_atr_sizing:
                            atr_value = self.atr[0]
                            stop_distance = atr_value * self.params.atr_multiplier
                            self.stop_loss = current_price - stop_distance
                            self.take_profit = current_price + (
                                stop_distance * 2
                            )  # 2:1 R/R
                        else:
                            self.stop_loss = current_price * (
                                1 - self.params.stop_loss_pct
                            )
                            self.take_profit = current_price * (
                                1 + self.params.take_profit_pct
                            )

                        self.trailing_stop_price = None
                        self.highest_price_since_entry = None

                        logger.info(
                            f"🟢 BUY at ${current_price:.2f} | Size: {size:.8f} | "
                            f"SL: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f} | "
                            f"Reason: {details['reasoning']}"
                        )
            else:
                # Exit on opposite signal
                if self.params.exit_on_opposite_signal and signal == -1:
                    self.order = self.close()
                    logger.info(
                        f"🔵 EXIT on opposite signal at ${current_price:.2f} | "
                        f"Reason: {details['reasoning']}"
                    )

        except Exception as e:
            logger.error(f"❌ Error in next(): {e}", exc_info=True)

    def calculate_position_size(self, signal_direction):
        """Calculate position size with ATR-based risk management"""
        current_price = self.data.close[0]
        cash = self.broker.getcash()

        if self.params.use_atr_sizing:
            # ATR-based stop distance
            atr_value = self.atr[0]
            stop_distance = atr_value * self.params.atr_multiplier
            stop_distance_pct = stop_distance / current_price

            # Position size based on risk and ATR stop
            risk_amount = cash * self.params.risk_per_trade
            position_value = risk_amount / stop_distance_pct
            size = position_value / current_price

            # Cap at max position percentage
            max_position_value = cash * self.params.max_position_pct
            max_size = max_position_value / current_price
            size = min(size, max_size)

            logger.debug(
                f"ATR Position Sizing: "
                f"ATR={atr_value:.2f}, "
                f"Stop Distance={stop_distance_pct*100:.2f}%, "
                f"Size={size:.8f}"
            )
        else:
            #  percentage stop
            risk_amount = cash * self.params.risk_per_trade
            position_value = risk_amount / self.params.stop_loss_pct
            size = position_value / current_price

            max_position_value = cash * self.params.max_position_pct
            max_size = max_position_value / current_price
            size = min(size, max_size)

        return max(size, 0)

    def update_trailing_stop(self):
        """Update trailing stop for long positions"""
        if not self.position or self.position.size <= 0:
            return

        current_price = self.data.close[0]

        # Track highest price since entry
        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = current_price
        else:
            self.highest_price_since_entry = max(
                self.highest_price_since_entry, current_price
            )

        # Calculate trailing stop
        new_trailing_stop = self.highest_price_since_entry * (
            1 - self.params.trailing_stop_pct
        )

        # Update if higher than current trailing stop
        if (
            self.trailing_stop_price is None
            or new_trailing_stop > self.trailing_stop_price
        ):
            self.trailing_stop_price = new_trailing_stop
            logger.debug(
                f"Trailing stop updated: ${self.trailing_stop_price:.2f} "
                f"(High: ${self.highest_price_since_entry:.2f})"
            )

    def stop(self):
        logger.info(f"=" * 70)
        logger.info(f"🛑 Strategy stopped")
        logger.info(f"=" * 70)
        logger.info(f"Total bars processed: {self.next_call_count}")
        logger.info(f"Total signals logged: {len(self.signal_log)}")
        if self.signal_log:
            # Analyze signal distribution
            signal_counts = {-1: 0, 0: 0, 1: 0}
            reasoning_counts = {}
            for log in self.signal_log:
                sig = log["signal"]
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
                reason = log["details"].get("reasoning", "unknown")
                reasoning_counts[reason] = reasoning_counts.get(reason, 0) + 1
            total_signals = len(self.signal_log)
            logger.info(f"Signal distribution:")
            logger.info(
                f"  SELL (-1): {signal_counts[-1]:>4} ({signal_counts[-1]/total_signals*100:>5.1f}%)"
            )
            logger.info(
                f"  HOLD ( 0): {signal_counts[0]:>4} ({signal_counts[0]/total_signals*100:>5.1f}%)"
            )
            logger.info(
                f"  BUY  ( 1): {signal_counts[1]:>4} ({signal_counts[1]/total_signals*100:>5.1f}%)"
            )
            logger.info(f"\nTop signal reasoning:")
            sorted_reasons = sorted(
                reasoning_counts.items(), key=lambda x: x[1], reverse=True
            )
            for reason, count in sorted_reasons[:5]:
                logger.info(f"  {reason}: {count} ({count/total_signals*100:.1f}%)")
            logger.info(f"\nSample signals (first 5):")
            for i, log in enumerate(self.signal_log[:5]):
                logger.info(
                    f"  {log['date']}: Signal={log['signal']:>2}, "
                    f"Price=${log['price']:.2f}, "
                    f"Reason={log['details']['reasoning']}"
                )
        else:
            logger.warning("⚠️ NO SIGNALS WERE GENERATED!")
            logger.warning(
                "This is unexpected - check model loading and feature generation"
            )


def run_backtest(
    asset_key, aggregator_mode="weighted_voting", aggregator_preset="balanced"
):
    """
    Run backtest with configurable aggregator settings
    Args:
        asset_key: 'BTC' or 'GOLD'
        aggregator_mode: 'weighted_voting' (only supported mode currently)
        aggregator_preset: 'conservative', 'balanced', or 'aggressive'
    """
    logger.info("=" * 70)
    logger.info(f"🚀 STARTING BACKTEST FOR {asset_key.upper()}")
    logger.info("=" * 70)
    logger.info(f"Aggregator Mode: {aggregator_mode}")
    logger.info(f"Aggregator Preset: {aggregator_preset}")
    logger.info("=" * 70)
    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("❌ config/config.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config/config.json: {e}")
        sys.exit(1)

    cerebro = bt.Cerebro()

    # Load test dataset
    test_path = f"data/test_data_{asset_key.lower()}.csv"
    try:
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        test_df.columns = test_df.columns.str.lower()
    except FileNotFoundError:
        logger.error(f"❌ Test data not found: {test_path}")
        sys.exit(1)

    # Try to load training data for more context
    train_path = f"data/train_data_{asset_key.lower()}.csv"
    try:
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        train_df.columns = train_df.columns.str.lower()
        # Combine train and test data
        df = pd.concat([train_df, test_df]).drop_duplicates().sort_index()
        logger.info(f"✅ Combined train + test data: {len(df)} bars")
    except FileNotFoundError:
        logger.warning(f"⚠️ Train data not found, using only test data")
        df = test_df

    logger.info(f"📊 Backtest Data Summary:")
    logger.info(f"  Total bars: {len(df)}")
    logger.info(f"  Date range: {df.index[0]} → {df.index[-1]}")
    logger.info(f"  Price range: ${df['close'].min():.2f} → ${df['close'].max():.2f}")
    logger.info(f"  Mean price: ${df['close'].mean():.2f}")

    # Create Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )
    cerebro.adddata(data)

    # Add strategy with custom parameters
    MLStrategy.asset_key = asset_key.lower()
    cerebro.addstrategy(
        MLStrategy, aggregator_mode=aggregator_mode, aggregator_preset=aggregator_preset
    )

    # Broker settings
    initial_capital = config["backtesting"]["initial_capital"]
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=config["backtesting"]["commission_pct"])

    # Add analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run backtest
    logger.info(f"💵 Starting Portfolio Value: ${initial_capital:,.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    # Display results
    logger.info("=" * 70)
    logger.info(f"📊 BACKTEST RESULTS FOR {asset_key.upper()}")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
    total_return = (final_value - initial_capital) / initial_capital * 100
    logger.info(f"Total Return: {total_return:+.2f}%")

    # Sharpe Ratio
    sharpe_dict = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_dict.get("sharperatio", None)
    if sharpe:
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    else:
        logger.info("Sharpe Ratio: N/A (insufficient trades)")

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    logger.info(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    # Trade statistics
    trades = strat.analyzers.trades.get_analysis()
    try:
        closed = trades.total.closed if hasattr(trades, "total") else 0
        won = trades.won.total if hasattr(trades, "won") else 0
        lost = trades.lost.total if hasattr(trades, "lost") else 0
        if closed > 0:
            win_rate = won / closed * 100
            logger.info(f"─" * 70)
            logger.info(f"Trade Statistics:")
            logger.info(f"  Total Trades: {closed}")
            logger.info(f"  Winning Trades: {won}")
            logger.info(f"  Losing Trades: {lost}")
            logger.info(f"  Win Rate: {win_rate:.2f}%")
            if hasattr(trades, "pnl") and hasattr(trades.pnl, "net"):
                logger.info(f"  Total Net PnL: ${trades.pnl.net.total:.2f}")
                if hasattr(trades.pnl.net, "average"):
                    logger.info(
                        f"  Average PnL per Trade: ${trades.pnl.net.average:.2f}"
                    )
        else:
            logger.warning("=" * 70)
            logger.warning("⚠️ NO TRADES EXECUTED!")
            logger.warning("=" * 70)
            logger.warning("Possible reasons:")
            logger.warning("  1. All signals below confidence thresholds")
            logger.warning("  2. Strategies not agreeing on direction")
            logger.warning("  3. Model predictions too uncertain")
            logger.warning("")
            logger.warning("Recommendations:")
            logger.warning("  1. Try 'aggressive' preset: --preset aggressive")
            logger.warning("  2. Check model training logs for accuracy")
            logger.warning("  3. Review signal logs above for reasoning patterns")
            logger.warning("=" * 70)
    except Exception as e:
        logger.error(f"❌ Error extracting trade statistics: {e}")
        logger.warning("⚠️ Could not extract trade statistics (likely no trades)")

    logger.info("=" * 70)
    logger.info(f"✅ BACKTEST FOR {asset_key.upper()} COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with BullMarketFilteredAggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balanced approach (default)
  python backtest.py --asset BTC

  # More trades (aggressive)
  python backtest.py --asset GOLD --preset aggressive

  # Fewer but higher quality trades
  python backtest.py --asset BTC --preset conservative
        """,
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        choices=["BTC", "GOLD"],
        help="Asset to backtest (BTC or GOLD)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="weighted_voting",
        choices=["weighted_voting"],
        help="Signal aggregation mode (currently only weighted_voting supported)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive", "scalper", "sniper"],
        help="Confidence threshold preset",
    )
    args = parser.parse_args()
    run_backtest(
        asset_key=args.asset, aggregator_mode=args.mode, aggregator_preset=args.preset
    )
