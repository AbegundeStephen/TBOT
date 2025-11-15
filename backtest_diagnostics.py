#!/usr/bin/env python3
"""
Backtest Diagnostics - Analyze why your strategy is losing money
Run this BEFORE making changes to understand the problems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class BacktestDiagnostics:
    """Comprehensive backtest analysis"""
    
    def __init__(self, asset_key: str):
        self.asset_key = asset_key
        self.data_path = f"data/test_data_{asset_key.lower()}.csv"
        self.model_path = f"models"
        
    def load_data(self):
        """Load price data"""
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.df.columns = self.df.columns.str.lower()
        print(f"Loaded {len(self.df)} bars for {self.asset_key}")
        
    def analyze_market_conditions(self):
        """Understand the market you're trading"""
        print("\n" + "="*70)
        print("MARKET CONDITION ANALYSIS")
        print("="*70)
        
        # Price statistics
        returns = self.df['close'].pct_change()
        
        print(f"\nPrice Stats:")
        print(f"  Start: ${self.df['close'].iloc[0]:.2f}")
        print(f"  End: ${self.df['close'].iloc[-1]:.2f}")
        print(f"  Buy & Hold Return: {(self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100:.2f}%")
        print(f"  Max Price: ${self.df['close'].max():.2f}")
        print(f"  Min Price: ${self.df['close'].min():.2f}")
        
        print(f"\nVolatility:")
        print(f"  Daily Volatility: {returns.std() * 100:.2f}%")
        print(f"  Average Daily Return: {returns.mean() * 100:.3f}%")
        print(f"  Max Drawdown: {self._calculate_max_drawdown(self.df['close']) * 100:.2f}%")
        
        # Trending vs Ranging
        sma_50 = self.df['close'].rolling(50).mean()
        sma_200 = self.df['close'].rolling(200).mean()
        
        trending_days = (sma_50 > sma_200).sum()
        ranging_days = len(sma_50) - trending_days
        
        print(f"\nMarket Regime:")
        print(f"  Trending (SMA50 > SMA200): {trending_days/len(sma_50)*100:.1f}%")
        print(f"  Ranging (SMA50 < SMA200): {ranging_days/len(sma_50)*100:.1f}%")
        
        # Opportunity analysis
        from talib import RSI, BBANDS
        
        rsi = RSI(self.df['close'].values, timeperiod=14)
        bb_upper, bb_middle, bb_lower = BBANDS(
            self.df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        oversold_count = (rsi < 30).sum()
        overbought_count = (rsi > 70).sum()
        bb_breakout_lower = (self.df['close'] < bb_lower).sum()
        bb_breakout_upper = (self.df['close'] > bb_upper).sum()
        
        print(f"\nTrading Opportunities:")
        print(f"  RSI Oversold (<30): {oversold_count} bars ({oversold_count/len(self.df)*100:.1f}%)")
        print(f"  RSI Overbought (>70): {overbought_count} bars ({overbought_count/len(self.df)*100:.1f}%)")
        print(f"  Below BB Lower: {bb_breakout_lower} bars ({bb_breakout_lower/len(self.df)*100:.1f}%)")
        print(f"  Above BB Upper: {bb_breakout_upper} bars ({bb_breakout_upper/len(self.df)*100:.1f}%)")
        
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    def simulate_simple_strategies(self):
        """Test simple rule-based strategies without ML"""
        print("\n" + "="*70)
        print("SIMPLE STRATEGY SIMULATION (No ML)")
        print("="*70)
        
        from talib import RSI, BBANDS, MACD, SMA
        
        close = self.df['close'].values
        
        # Strategy 1: RSI Mean Reversion
        rsi = RSI(close, timeperiod=14)
        buys_rsi = (rsi < 30)
        sells_rsi = (rsi > 70)
        
        pnl_rsi = self._simulate_trades(close, buys_rsi, sells_rsi, "RSI Mean Reversion")
        
        # Strategy 2: Bollinger Bands
        bb_upper, bb_middle, bb_lower = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        buys_bb = (close < bb_lower)
        sells_bb = (close > bb_upper)
        
        pnl_bb = self._simulate_trades(close, buys_bb, sells_bb, "Bollinger Bands")
        
        # Strategy 3: MA Crossover
        sma_fast = SMA(close, timeperiod=20)
        sma_slow = SMA(close, timeperiod=50)
        
        buys_ma = (sma_fast > sma_slow) & (np.roll(sma_fast <= sma_slow, 1))
        sells_ma = (sma_fast < sma_slow) & (np.roll(sma_fast >= sma_slow, 1))
        
        pnl_ma = self._simulate_trades(close, buys_ma, sells_ma, "MA Crossover")
        
        print(f"\n⚠️  If simple strategies are profitable, ML is HURTING performance!")
        print(f"⚠️  If simple strategies lose money, the market might not be tradeable!")
        
    def _simulate_trades(self, prices, buy_signals, sell_signals, strategy_name):
        """Simulate a simple trading strategy"""
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(len(prices)):
            if position == 0 and buy_signals[i]:
                position = 1
                entry_price = prices[i]
            elif position == 1 and sell_signals[i]:
                pnl = (prices[i] - entry_price) / entry_price
                trades.append(pnl)
                position = 0
        
        if len(trades) == 0:
            print(f"\n{strategy_name}: NO TRADES")
            return 0
        
        total_return = np.sum(trades) * 100
        win_rate = (np.array(trades) > 0).sum() / len(trades) * 100
        avg_win = np.array(trades)[np.array(trades) > 0].mean() * 100 if (np.array(trades) > 0).any() else 0
        avg_loss = np.array(trades)[np.array(trades) < 0].mean() * 100 if (np.array(trades) < 0).any() else 0
        
        print(f"\n{strategy_name}:")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Trades: {len(trades)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Win: {avg_win:.2f}%")
        print(f"  Avg Loss: {avg_loss:.2f}%")
        
        return total_return
    
    def analyze_model_labels(self):
        """Analyze the labels your model was trained on"""
        print("\n" + "="*70)
        print("MODEL LABEL ANALYSIS")
        print("="*70)
        
        # This would require loading your trained data
        # For now, provide instructions
        print("\n⚠️  TO DO:")
        print("1. Save labels during training: df['label'].to_csv(f'analysis/labels_{asset}.csv')")
        print("2. Check label distribution:")
        print("   - If < 5% BUY signals: Criteria too strict")
        print("   - If > 40% BUY signals: Criteria too loose")
        print("   - Ideal: 10-20% of bars labeled as trades")
        
    def analyze_transaction_costs(self):
        """Calculate impact of fees and slippage"""
        print("\n" + "="*70)
        print("TRANSACTION COST ANALYSIS")
        print("="*70)
        
        # Assume your backtest numbers
        num_trades = 93  # From your BTC results
        commission = 0.001  # 0.1%
        slippage = 0.0005  # 0.05%
        
        total_cost_per_trade = (commission + slippage) * 2  # Entry + Exit
        total_cost = num_trades * total_cost_per_trade * 100
        
        print(f"\nFor {num_trades} trades:")
        print(f"  Commission per trade: {commission*100:.2f}%")
        print(f"  Slippage per trade: {slippage*100:.2f}%")
        print(f"  Total cost per round-trip: {total_cost_per_trade*100:.2f}%")
        print(f"  Total costs: {total_cost:.2f}%")
        print(f"\n⚠️  You lost 15.73% but paid ~{total_cost:.2f}% in fees!")
        print(f"⚠️  Even a profitable strategy would struggle with {num_trades} trades!")
        
    def generate_recommendations(self):
        """Provide actionable recommendations"""
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        print("\n🔧 IMMEDIATE FIXES:")
        print("1. Reduce number of trades by 50%")
        print("   → Raise min_confidence from 0.60 to 0.75")
        print("   → Switch aggregator to 'strict' mode")
        
        print("\n2. Fix label generation")
        print("   → Use stop-loss/take-profit simulation (see artifact)")
        print("   → Current method: predicting average future price")
        print("   → Better method: predicting trade outcomes")
        
        print("\n3. Add confidence filtering")
        print("   → Use predict_proba() instead of predict()")
        print("   → Only trade when model is >70% confident")
        
        print("\n4. Widen stops if needed")
        print("   → BTC: Try 5% SL, 10% TP")
        print("   → GOLD: Try 3% SL, 6% TP")
        
        print("\n📊 ANALYSIS TO RUN:")
        print("1. Plot equity curve to find when losses occur")
        print("2. Check if losses cluster in ranging/trending markets")
        print("3. Analyze losing trades: Are stops too tight?")
        print("4. Compare to buy-and-hold: Are you beating it?")
        
        print("\n🎯 SUCCESS CRITERIA:")
        print("Target metrics for next iteration:")
        print("  • Win Rate: >50%")
        print("  • Total Trades: 20-40 (not 90+)")
        print("  • Sharpe Ratio: >0.5")
        print("  • Max Drawdown: <10%")
        print("  • Total Return: >0%")
        

def main():
    """Run full diagnostic"""
    import sys
    
    asset = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    
    print(f"\n{'='*70}")
    print(f"BACKTEST DIAGNOSTICS FOR {asset}")
    print(f"{'='*70}\n")
    
    diag = BacktestDiagnostics(asset)
    
    try:
        diag.load_data()
        diag.analyze_market_conditions()
        diag.simulate_simple_strategies()
        diag.analyze_model_labels()
        diag.analyze_transaction_costs()
        diag.generate_recommendations()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTICS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()