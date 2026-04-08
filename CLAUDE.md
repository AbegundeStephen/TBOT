# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TBOT is a Python 3.10+ AI-powered multi-asset algorithmic trading bot targeting BTC (Binance Futures/Spot), Gold, EURUSD, EURJPY, and USTEC (MetaTrader 5). It uses triple-strategy signal aggregation with ML-based validation and real-time execution.

## Commands

### Setup
```bash
python -m venv venv
source venv/Scripts/activate          # Windows Git Bash
pip install -r requirements.txt
cp config/config.template.json config/config.json
```

### Run
```bash
python main.py           # Live/paper trading bot
python backtest.py       # Backtesting (supports --start-date, --end-date, --initial-capital)
```

### Tests
```bash
pytest tests/
python tests/test_connection.py
python tests/test_btc_futures_execution.py
python tests/test_gold_margin_execution.py
python tests/test_risk_management.py
```

### Logs & Monitoring
```bash
tail -f logs/trading_bot.log          # Live log stream
# Web dashboard at http://localhost:5000 (Flask)
```

## Architecture

### Signal Flow (main loop runs every ~5 minutes)
```
DataManager (1H + 4H OHLCV)
    → MTFRegimeDetector (4H trend/range context)
    → PerformanceWeightedAggregator
        ├── MeanReversionStrategy  (Bollinger + RSI + Stochastic)
        ├── TrendFollowingStrategy (EMA + MACD + ADX)
        └── EMAStrategy            (Fast/Slow EMA crossovers)
    → HybridSignalValidator (AI filtering: OHLCSniper + PatternMiner)
    → Risk gating (circuit breakers, position sizing, portfolio correlation)
    → BinanceHandler (BTC) | MT5Handler (Gold/FX/Indices)
    → TradingDatabaseManager (Supabase) + TradingTelegramBot (alerts)
```

### Key Source Modules

| Path | Purpose |
|---|---|
| `main.py` | Main `TradingBot` orchestrator; entry point |
| `backtest.py` | Backtesting engine |
| `src/execution/signal_aggregator.py` | `PerformanceWeightedAggregator` — primary signal fusion |
| `src/execution/council_aggregator.py` | `InstitutionalCouncilAggregator` — advanced alternative aggregator |
| `src/strategies/` | Three concrete strategies inheriting `BaseStrategy` |
| `src/ai/hybrid_validator.py` | `HybridSignalValidator` — AI-based signal accept/reject |
| `src/ai/sniper.py` | `OHLCSniper` — price action entry pattern detection |
| `src/execution/mt5_handler.py` | All MetaTrader 5 order execution |
| `src/execution/binance_handler.py` | Binance Spot & Futures execution |
| `src/data/data_manager.py` | Hybrid live/testnet OHLCV fetching |
| `src/portfolio/portfolio_manager.py` | Position tracking, correlation filtering |
| `src/execution/veteran_trade_manager.py` | Advanced exit logic (trailing stops, partial TPs) |
| `src/execution/mtf_integration.py` | 4H multi-timeframe regime context |
| `src/training/autotrainer.py` | `ContinuousLearningPipeline` — 7-day model retraining |
| `src/dashboard/server.py` | Flask web UI |
| `config/config.json` | Runtime config (assets, strategies, risk params, API keys) |

### Configuration System

All runtime behaviour lives in `config/config.json` (generated from `config.template.json`). Credentials and mode flags are loaded from `.env`:
- `TRADING_MODE` — `"live"` or `"paper"`
- `BINANCE_TESTNET` — `true`/`false`
- Binance API keys, MT5 credentials, Supabase URL/key, Telegram token

Per-asset strategy parameters (leverage, stop-loss tiers, signal thresholds) are nested under each asset key in `config.json`.

### Risk Management Rules (hardcoded behaviour to be aware of)
- **Circuit breaker**: stops all trading if daily loss ≥ 3% or drawdown ≥ 15%
- **Position sizing**: 1–2% risk per trade of account balance
- **Partial exits**: three TP tiers at 45% / 30% / 25% of position
- **Trailing stop**: activates after 2% unrealised profit
- **MTF gate**: 1H entry signals rejected when 4H regime is counter-trend

### Adding a New Strategy
1. Inherit from `src/strategies/base_strategy.py::BaseStrategy`
2. Implement `generate_signal(df) -> dict` returning `{"signal": int, "confidence": float, ...}`
3. Register in `src/execution/signal_aggregator.py` alongside the existing three strategies
4. Add per-asset config block in `config.json`
