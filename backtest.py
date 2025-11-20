#!/usr/bin/env python3
"""
Fixed Backtesting Script - Properly integrates BullMarketFilteredAggregator
"""
import json
import logging
import argparse
from pathlib import Path
import sys
import pandas as pd
import backtrader as bt
from datetime import datetime
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
        "buy_score_threshold": 0.50,
        "sell_score_threshold": 0.50,
        "perfect_agreement_bonus": 0.15,
        "mean_reversion_weight": 1.3,
        "trend_following_weight": 0.4,
        "allow_single_mr_signal": True,
        "single_mr_threshold": 0.70,
        "allow_single_tf_signal": False,
        "min_confidence": 0.45,
        "signal_quality_threshold": 0.55,
        "high_confidence_threshold": 0.70,
        "confidence_normalization": True,
        "enable_bull_filter": True,
        "block_sells_in_bull": True,
        "boost_buys_in_bull": 0.05,
        "regime_confirmation_bars": 3,
        "verbose_logging": True,
    },
    "balanced": {
        "buy_score_threshold": 0.40,
        "sell_score_threshold": 0.40,
        "perfect_agreement_bonus": 0.15,
        "mean_reversion_weight": 1.2,
        "trend_following_weight": 0.5,
        "allow_single_mr_signal": True,
        "single_mr_threshold": 0.60,
        "allow_single_tf_signal": False,
        "min_confidence": 0.35,
        "signal_quality_threshold": 0.40,
        "high_confidence_threshold": 0.65,
        "confidence_normalization": True,
        "enable_bull_filter": True,
        "block_sells_in_bull": True,
        "boost_buys_in_bull": 0.10,
        "regime_confirmation_bars": 2,
        "verbose_logging": True,
    },
    "aggressive": {
        "buy_score_threshold": 0.35,
        "sell_score_threshold": 0.35,
        "perfect_agreement_bonus": 0.12,
        "mean_reversion_weight": 1.2,
        "trend_following_weight": 0.6,
        "allow_single_mr_signal": True,
        "single_mr_threshold": 0.55,
        "allow_single_tf_signal": False,
        "min_confidence": 0.30,
        "signal_quality_threshold": 0.35,
        "high_confidence_threshold": 0.60,
        "confidence_normalization": True,
        "enable_bull_filter": True,
        "block_sells_in_bull": False,
        "boost_buys_in_bull": 0.12,
        "regime_confirmation_bars": 1,
        "verbose_logging": True,
    },
    "scalper": {
        "buy_score_threshold": 0.30,
        "sell_score_threshold": 0.30,
        "perfect_agreement_bonus": 0.10,
        "mean_reversion_weight": 1.5,
        "trend_following_weight": 0.4,
        "allow_single_mr_signal": True,
        "single_mr_threshold": 0.50,
        "allow_single_tf_signal": False,
        "min_confidence": 0.25,
        "signal_quality_threshold": 0.30,
        "high_confidence_threshold": 0.55,
        "confidence_normalization": True,
        "enable_bull_filter": True,
        "block_sells_in_bull": False,
        "boost_buys_in_bull": 0.08,
        "regime_confirmation_bars": 1,
        "verbose_logging": True,
    },
    "sniper": {
        "buy_score_threshold": 0.60,
        "sell_score_threshold": 0.60,
        "perfect_agreement_bonus": 0.20,
        "mean_reversion_weight": 1.5,
        "trend_following_weight": 0.3,
        "allow_single_mr_signal": True,
        "single_mr_threshold": 0.75,
        "allow_single_tf_signal": False,
        "min_confidence": 0.50,
        "signal_quality_threshold": 0.60,
        "high_confidence_threshold": 0.75,
        "confidence_normalization": True,
        "enable_bull_filter": True,
        "block_sells_in_bull": True,
        "boost_buys_in_bull": 0.05,
        "regime_confirmation_bars": 4,
        "verbose_logging": True,
    },
}


class MLStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper using BullMarketFilteredAggregator
    """

    params = (
        ("stop_loss_pct", 0.015),
        ("take_profit_pct", 0.03),
        ("lookback", 100),
        ("aggregator_mode", "weighted_voting"),
        ("aggregator_preset", "balanced"),
    )

    def __init__(self):
        # Set asset key from class attribute
        self.asset_key = getattr(self.__class__, 'asset_key', 'btc').lower()
        
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
        tf_config = config["strategy_configs"]["trend_following"][self.asset_key.upper()]
        ema_config = config["strategy_configs"]["exponential_moving_averages"][self.asset_key.upper()]

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
            asset_name=self.asset_key.upper()
        )

        self.order = None
        self.trade_count = 0
        self.signal_log = []
        self.next_call_count = 0

        logger.info(f"=" * 70)
        logger.info(f"Strategy initialized for {self.asset_key.upper()}")
        logger.info(f"=" * 70)
        logger.info(f"Lookback period: {self.params.lookback}")
        logger.info(f"Aggregator mode: {self.params.aggregator_mode}")
        logger.info(f"Aggregator preset: {preset_name}")
        logger.info(f"=" * 70)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(
                    f"✅ BUY EXECUTED - Price: ${order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.8f}, "
                    f"Cost: ${order.executed.value:.2f}"
                )
            elif order.issell():
                logger.info(
                    f"✅ SELL EXECUTED - Price: ${order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.8f}, "
                    f"Value: ${order.executed.value:.2f}"
                )
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"⚠️ Order {order.status}: Status={order.status}")
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

        # Log first call
        if self.next_call_count == 1:
            logger.info(
                f"📊 First next() call - Data length: {len(self.data)}, "
                f"Lookback needed: {self.params.lookback}"
            )

        # Don't interfere with pending orders
        if self.order:
            return

        # Wait for sufficient data
        if len(self.data) < self.params.lookback:
            if self.next_call_count % 100 == 0:
                logger.debug(
                    f"⏳ Warming up: {len(self.data)}/{self.params.lookback} bars"
                )
            return

        try:
            # Prepare data for models
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

            # Get aggregated signal using weighted voting system
            signal, details = self.aggregator.get_aggregated_signal(df)
            
            # Extract scoring details
            buy_score = details.get('buy_score', 0)
            sell_score = details.get('sell_score', 0)
            signal_quality = details.get('signal_quality', 0)

            # Log signals periodically (every 5 bars)
            if self.next_call_count % 5 == 0:
                self.signal_log.append(
                    {
                        "date": self.data.datetime.date(0),
                        "price": self.data.close[0],
                        "signal": signal,
                        "details": details,
                    }
                )

                regime = details.get('regime', 'UNKNOWN')
                mr_sig = details.get('mean_reversion_signal', 0)
                mr_conf = details.get('mean_reversion_confidence', 0)
                tf_sig = details.get('trend_following_signal', 0)
                tf_conf = details.get('trend_following_confidence', 0)

                logger.info(
                    f"📍 Bar {self.next_call_count} | "
                    f"Date: {self.data.datetime.date(0)} | "
                    f"Price: ${self.data.close[0]:.2f} | "
                    f"{regime} | "
                    f"MR: {mr_sig:>2}({mr_conf:.2f}) | "
                    f"TF: {tf_sig:>2}({tf_conf:.2f}) | "
                    f"Score(B/S): {buy_score:.2f}/{sell_score:.2f} | "
                    f"Signal: {signal:>2} | "
                    f"Quality: {signal_quality:.2f}"
                )

            # Execute trades based on signals
            current_price = self.data.close[0]

            if not self.position:
                # No position - look for entry
                if signal == 1:  # BUY signal
                    cash = self.broker.getcash()
                    size = (cash * 0.95) / current_price
                    if size > 0:
                        self.order = self.buy(size=size)
                        logger.info(
                            f"🟢 BUY SIGNAL TRIGGERED at ${current_price:.2f} | "
                            f"Size: {size:.8f} | BuyScore: {buy_score:.2f} | "
                            f"Quality: {signal_quality:.2f} | "
                            f"Reason: {details['reasoning']}"
                        )

                elif signal == -1:  # SELL signal (short)
                    cash = self.broker.getcash()
                    size = (cash * 0.95) / current_price
                    if size > 0:
                        self.order = self.sell(size=size)
                        logger.info(
                            f"🔴 SELL SIGNAL TRIGGERED at ${current_price:.2f} | "
                            f"Size: {size:.8f} | SellScore: {sell_score:.2f} | "
                            f"Quality: {signal_quality:.2f} | "
                            f"Reason: {details['reasoning']}"
                        )

            else:
                # Have position - look for exit
                if signal == -1 and self.position.size > 0:  # Close long
                    self.order = self.close()
                    logger.info(
                        f"🔵 CLOSE LONG at ${current_price:.2f} | "
                        f"Reason: {details['reasoning']}"
                    )

                elif signal == 1 and self.position.size < 0:  # Close short
                    self.order = self.close()
                    logger.info(
                        f"🔵 CLOSE SHORT at ${current_price:.2f} | "
                        f"Reason: {details['reasoning']}"
                    )

        except Exception as e:
            logger.error(f"❌ Error in next() call {self.next_call_count}: {e}")
            import traceback
            traceback.print_exc()

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
        choices=["conservative", "balanced", "aggressive"],
        help="Confidence threshold preset",
    )

    args = parser.parse_args()

    run_backtest(
        asset_key=args.asset, aggregator_mode=args.mode, aggregator_preset=args.preset
    )