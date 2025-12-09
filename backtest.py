#!/usr/bin/env python3
"""
Backtesting Script - Properly integrates PerformanceWeightedAggregator
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
from src.execution.signal_aggregator import PerformanceWeightedAggregator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Asset-specific presets based on actual training performance
AGGREGATOR_PRESETS = {
            "BTC": {
                "conservative": {
                    "buy_threshold": 0.35,
                    "sell_threshold": 0.40,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.20,
                    "bull_buy_boost": 0.10,
                    "bull_sell_penalty": 0.12,
                    "bear_sell_boost": 0.10,
                    "bear_buy_penalty": 0.12,
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.32,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.30,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.22,
                    "bull_buy_boost": 0.11,
                    "bull_sell_penalty": 0.11,
                    "bear_sell_boost": 0.11,
                    "bear_buy_penalty": 0.11,
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.17,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.25,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.12,
                    "bull_sell_penalty": 0.12,
                    "bear_sell_boost": 0.12,
                    "bear_buy_penalty": 0.12,
                    "min_confidence_to_use": 0.09,
                    "min_signal_quality": 0.25,
                    "hold_contribution_pct": 0.18,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "verbose": False,
                },
                "scalper": {
                    "buy_threshold": 0.24,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.25,
                    "three_strategy_bonus": 0.30,
                    "bull_buy_boost": 0.15,
                    "bull_sell_penalty": 0.10,
                    "bear_sell_boost": 0.15,
                    "bear_buy_penalty": 0.10,
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.20,
                    "hold_contribution_pct": 0.20,
                    "allow_single_override": True,
                    "single_override_threshold": 0.65,
                    "verbose": False,
                    "min_quality_margin": 0.05,
                },
            },
            "GOLD": {
                "conservative": {
                    "buy_threshold": 0.38,
                    "sell_threshold": 0.42,
                    "two_strategy_bonus": 0.18,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.07,
                    "bull_sell_penalty": 0.09,
                    "bear_sell_boost": 0.07,
                    "bear_buy_penalty": 0.09,
                    "min_confidence_to_use": 0.12,
                    "min_signal_quality": 0.30,
                    "hold_contribution_pct": 0.12,
                    "allow_single_override": True,
                    "single_override_threshold": 0.75,
                    "verbose": False,
                },
                "balanced": {
                    "buy_threshold": 0.33,
                    "sell_threshold": 0.36,
                    "two_strategy_bonus": 0.20,
                    "three_strategy_bonus": 0.25,
                    "bull_buy_boost": 0.08,
                    "bull_sell_penalty": 0.08,
                    "bear_sell_boost": 0.08,
                    "bear_buy_penalty": 0.08,
                    "min_confidence_to_use": 0.10,
                    "min_signal_quality": 0.28,
                    "hold_contribution_pct": 0.14,
                    "allow_single_override": True,
                    "single_override_threshold": 0.72,
                    "verbose": False,
                },
                "aggressive": {
                    "buy_threshold": 0.28,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.22,
                    "three_strategy_bonus": 0.30,
                    "bull_buy_boost": 0.09,
                    "bull_sell_penalty": 0.09,
                    "bear_sell_boost": 0.09,
                    "bear_buy_penalty": 0.09,
                    "min_confidence_to_use": 0.08,
                    "min_signal_quality": 0.22,
                    "hold_contribution_pct": 0.15,
                    "allow_single_override": True,
                    "single_override_threshold": 0.70,
                    "verbose": False,
                },
                "scalper": {
                    "buy_threshold": 0.23,
                    "sell_threshold": 0.30,
                    "two_strategy_bonus": 0.25,
                    "three_strategy_bonus": 0.35,
                    "bull_buy_boost": 0.12,
                    "bull_sell_penalty": 0.08,
                    "bear_sell_boost": 0.12,
                    "bear_buy_penalty": 0.08,
                    "min_confidence_to_use": 0.06,
                    "min_signal_quality": 0.18,
                    "hold_contribution_pct": 0.18,
                    "allow_single_override": True,
                    "single_override_threshold": 0.65,
                    "verbose": False,
                    "min_quality_margin": 0.06,
                },
            },
}

class MLStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper using PerformanceWeightedAggregator
    """

    params = (
        ("stop_loss_pct", 0.004),
        ("take_profit_pct", 0.08),
        ("trailing_stop_pct", 0.015),
        ("risk_per_trade", 0.10),
        ("max_position_pct", 0.95),
        ("use_atr_sizing", True),
        ("atr_period", 14),
        ("atr_multiplier", 1.2),
        ("lookback", 100),
        ("aggregator_preset", "balanced"),
        ("use_trailing_stop", True),
        ("exit_on_opposite_signal", True),
    )

    def __init__(self):
        # Set asset key from class attribute
        self.asset_key = getattr(self.__class__, "asset_key", "btc").upper()

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

        # Get asset-specific preset configuration
        preset_name = self.params.aggregator_preset
        if preset_name not in AGGREGATOR_PRESETS[self.asset_key]:
            logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset_name = "balanced"

        confidence_config = AGGREGATOR_PRESETS[self.asset_key][preset_name].copy()

        # Initialize PerformanceWeightedAggregator
        self.aggregator = PerformanceWeightedAggregator(
            mean_reversion_strategy=self.mean_reversion,
            trend_following_strategy=self.trend_following,
            ema_strategy=self.ema_strategy,
            asset_type=self.asset_key,
            config=confidence_config,
        )

        self.order = None
        self.trade_count = 0
        self.signal_log = []
        self.next_call_count = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

        logger.info(f"=" * 70)
        logger.info(f" Strategy Configuration for {self.asset_key}")
        logger.info(f"=" * 70)
        logger.info(f"Preset: {preset_name}")
        logger.info(f"Risk Management:")
        logger.info(f"  Stop-Loss: {self.params.stop_loss_pct * 100}%")
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

            # Get signal from PerformanceWeightedAggregator
            signal, details = self.aggregator.get_aggregated_signal(df)

            # Log periodically
            if self.next_call_count % 10 == 0:
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
                quality = details.get("signal_quality", 0)

                logger.info(
                    f"📍 {self.data.datetime.date(0)} | "
                    f"${current_price:.2f} | {regime} | "
                    f"B/S: {buy_score:.2f}/{sell_score:.2f} | "
                    f"Sig: {signal:>2} | Q: {quality:.2f}"
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
                            self.take_profit = current_price + (stop_distance * 2)
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
                            f"Reason: {details.get('reasoning', 'N/A')}"
                        )
            else:
                # Exit on opposite signal
                if self.params.exit_on_opposite_signal and signal == -1:
                    self.order = self.close()
                    logger.info(
                        f"🔵 EXIT on opposite signal at ${current_price:.2f} | "
                        f"Reason: {details.get('reasoning', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"❌ Error in next(): {e}", exc_info=True)

    def calculate_position_size(self, signal_direction):
        """Calculate position size with ATR-based risk management"""
        current_price = self.data.close[0]
        cash = self.broker.getcash()

        if self.params.use_atr_sizing:
            atr_value = self.atr[0]
            stop_distance = atr_value * self.params.atr_multiplier
            stop_distance_pct = stop_distance / current_price

            risk_amount = cash * self.params.risk_per_trade
            position_value = risk_amount / stop_distance_pct
            size = position_value / current_price

            max_position_value = cash * self.params.max_position_pct
            max_size = max_position_value / current_price
            size = min(size, max_size)
        else:
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

        if self.highest_price_since_entry is None:
            self.highest_price_since_entry = current_price
        else:
            self.highest_price_since_entry = max(
                self.highest_price_since_entry, current_price
            )

        new_trailing_stop = self.highest_price_since_entry * (
            1 - self.params.trailing_stop_pct
        )

        if (
            self.trailing_stop_price is None
            or new_trailing_stop > self.trailing_stop_price
        ):
            self.trailing_stop_price = new_trailing_stop

    def stop(self):
        logger.info(f"=" * 70)
        logger.info(f"🛑 Strategy stopped - {self.asset_key}")
        logger.info(f"=" * 70)
        logger.info(f"Total bars processed: {self.next_call_count}")
        logger.info(f"Total signals logged: {len(self.signal_log)}")

        if self.signal_log:
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
        else:
            logger.warning("⚠️ NO SIGNALS WERE GENERATED!")

        # Print aggregator statistics
        stats = self.aggregator.get_statistics()
        logger.info(f"\n📊 Aggregator Statistics:")
        logger.info(f"  Signal Rate: {stats['signal_rate']:.2f}%")
        logger.info(f"  Buy Rate: {stats['buy_rate']:.2f}%")
        logger.info(f"  Sell Rate: {stats['sell_rate']:.2f}%")
        logger.info(f"  Bull Regime: {stats['bull_regime_pct']:.2f}%")
        logger.info(f"  Bear Regime: {stats['bear_regime_pct']:.2f}%")
        logger.info(f"  Regime Changes: {stats['regime_changes']}")
        logger.info(f"  Consensus Signals: {stats['consensus_rate']:.2f}%")


def run_backtest(asset_key, aggregator_preset="balanced"):
    """Run backtest with PerformanceWeightedAggregator"""
    logger.info("=" * 70)
    logger.info(f"🚀 STARTING BACKTEST FOR {asset_key.upper()}")
    logger.info("=" * 70)
    logger.info(f"Aggregator Preset: {aggregator_preset}")
    logger.info("=" * 70)

    try:
        with open("config/config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("❌ config/config.json not found")
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
        df = pd.concat([train_df, test_df]).drop_duplicates().sort_index()
        logger.info(f"✅ Combined train + test data: {len(df)} bars")
    except FileNotFoundError:
        logger.warning(f"⚠️ Train data not found, using only test data")
        df = test_df

    logger.info(f"📊 Backtest Data Summary:")
    logger.info(f"  Total bars: {len(df)}")
    logger.info(f"  Date range: {df.index[0]} → {df.index[-1]}")
    logger.info(f"  Price range: ${df['close'].min():.2f} → ${df['close'].max():.2f}")

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

    # Add strategy
    MLStrategy.asset_key = asset_key.upper()
    cerebro.addstrategy(MLStrategy, aggregator_preset=aggregator_preset)

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
        logger.info("Sharpe Ratio: N/A")

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
                    logger.info(f"  Avg PnL/Trade: ${trades.pnl.net.average:.2f}")
        else:
            logger.warning("=" * 70)
            logger.warning("⚠️ NO TRADES EXECUTED!")
            logger.warning("=" * 70)
            logger.warning(
                "Try: python backtest.py --asset " + asset_key + " --preset aggressive"
            )
            logger.warning(
                "Or:  python backtest.py --asset " + asset_key + " --preset scalper"
            )
    except Exception as e:
        logger.error(f"❌ Error extracting trade statistics: {e}")

    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run backtest with PerformanceWeightedAggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balanced approach (default)
  python backtest.py --asset BTC

  # More trades
  python backtest.py --asset GOLD --preset aggressive

  # High frequency
  python backtest.py --asset BTC --preset scalper

  # Quality over quantity
  python backtest.py --asset GOLD --preset conservative
        """,
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        choices=["BTC", "GOLD"],
        help="Asset to backtest",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive", "scalper"],
        help="Signal threshold preset",
    )

    args = parser.parse_args()
    run_backtest(asset_key=args.asset, aggregator_preset=args.preset)
