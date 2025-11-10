# **backtest.py**

#!/usr/bin/env python3
"""
Backtesting Script using Backtrader
"""

import json
import logging
from pathlib import Path
import sys
import pandas as pd
import backtrader as bt
from datetime import datetime

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.execution.signal_aggregator import SignalAggregator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLStrategy(bt.Strategy):
    """
    Backtrader strategy wrapper for our ML models
    """

    params = (
        ("mean_rev_model_path", "models/mean_reversion_btc.pkl"),
        ("trend_model_path", "models/trend_following_btc.pkl"),
        ("stop_loss_pct", 0.015),
        ("take_profit_pct", 0.03),
    )

    def __init__(self):
        # Load models
        with open("config/config.json") as f:
            config = json.load(f)

        mr_config = {**config["strategies"]["mean_reversion"], **config["ml"]}
        tf_config = {**config["strategies"]["trend_following"], **config["ml"]}

        self.mean_reversion = MeanReversionStrategy(mr_config)
        self.trend_following = TrendFollowingStrategy(tf_config)

        self.mean_reversion.load_model(self.params.mean_rev_model_path)
        self.trend_following.load_model(self.params.trend_model_path)

        self.aggregator = SignalAggregator(self.mean_reversion, self.trend_following)

        self.order = None
        self.trade_count = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            elif order.issell():
                logger.info(f"SELL EXECUTED, Price: {order.executed.price:.2f}")
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.info(f"TRADE CLOSED, PnL: {trade.pnl:.2f}")
            self.trade_count += 1

    def next(self):
        if self.order:
            return

        # Convert current data to DataFrame
        df = pd.DataFrame(
            {
                "open": [x for x in self.data.open.get(size=200)],
                "high": [x for x in self.data.high.get(size=200)],
                "low": [x for x in self.data.low.get(size=200)],
                "close": [x for x in self.data.close.get(size=200)],
                "volume": [x for x in self.data.volume.get(size=200)],
            }
        )

        if len(df) < 200:
            return

        # Get aggregated signal
        signal, details = self.aggregator.get_aggregated_signal(df)

        # Execute trades
        if not self.position:
            if signal == 1:  # BUY
                self.order = self.buy()
                # Set stop loss and take profit
                self.sell(
                    exectype=bt.Order.Stop,
                    price=self.data.close[0] * (1 - self.params.stop_loss_pct),
                )
                self.sell(
                    exectype=bt.Order.Limit,
                    price=self.data.close[0] * (1 + self.params.take_profit_pct),
                )
            elif signal == -1:  # SELL (Short)
                self.order = self.sell()
        else:
            if signal == -1 and self.position.size > 0:  # Close long
                self.order = self.close()
            elif signal == 1 and self.position.size < 0:  # Close short
                self.order = self.close()


def run_backtest():
    """Run comprehensive backtest"""
    logger.info("=" * 60)
    logger.info("Starting Backtest")
    logger.info("=" * 60)

    # Load configuration
    with open("config/config.json") as f:
        config = json.load(f)

    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Load test data
    df = pd.read_csv("data/test_data_btc.csv", index_col=0, parse_dates=True)

    # Create data feed
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(MLStrategy)

    # Set initial capital
    initial_capital = config["backtesting"]["initial_capital"]
    cerebro.broker.setcash(initial_capital)

    # Set commission
    cerebro.broker.setcommission(commission=config["backtesting"]["commission"])

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run backtest
    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()

    # Print results
    logger.info("=" * 60)
    logger.info("Backtest Results")
    logger.info("=" * 60)
    logger.info(f"Final Portfolio Value: ${final_value:.2f}")
    logger.info(
        f"Total Return: {((final_value - initial_capital) / initial_capital * 100):.2f}%"
    )

    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    logger.info(f'Sharpe Ratio: {sharpe.get("sharperatio", 0):.2f}')

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    logger.info(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    # Trade Analysis
    trades = strat.analyzers.trades.get_analysis()
    logger.info(f"Total Trades: {trades.total.closed if trades.total.closed else 0}")
    logger.info(f'Won Trades: {trades.won.total if hasattr(trades, "won") else 0}')
    logger.info(f'Lost Trades: {trades.lost.total if hasattr(trades, "lost") else 0}')

    if hasattr(trades, "won") and hasattr(trades, "lost"):
        win_rate = (
            trades.won.total / trades.total.closed * 100
            if trades.total.closed > 0
            else 0
        )
        logger.info(f"Win Rate: {win_rate:.2f}%")

    # Plot results
    # cerebro.plot()  # Uncomment to see visual plots

    logger.info("=" * 60)


if __name__ == "__main__":
    run_backtest()
