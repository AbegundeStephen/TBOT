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
    → Aggregator (hybrid: picks PerformanceWeighted or InstitutionalCouncil per asset)
        ├── MeanReversionStrategy  (Livermore-state-routed 3-mode: pullback / counter-trend / climax fade)
        ├── TrendFollowingStrategy (EMA + MACD + slope-aware ADX)
        └── EMAStrategy            (Fast/Slow EMA crossovers)
    → HybridSignalValidator (S/R proximity + candlestick patterns; see AI note below)
    → Risk gating (circuit breakers, position sizing, portfolio correlation)
    → BinanceHandler (BTC) | MT5Handler (Gold/FX/Indices)
    → TradingDatabaseManager (Supabase) + TradingTelegramBot (alerts)
```

> **AI validation note (2026-06 audit):** the CNN-LSTM `OHLCSniper` ML model is
> **disconnected** — `PerformanceWeightedAggregator._check_sniper_filter` returns
> `True` unconditionally and the model is not in the scoring pipeline.
> `HybridSignalValidator` today is **S/R-proximity + candlestick-pattern heuristics**,
> not an ML edge model. The Council's `_check_sniper_filter` is a *different* thing —
> a live price-action displacement/momentum gate that merely shares the "sniper"
> name. `signal_aggregator.py::_check_sniper_filter_LEGACY` is dead code (unreferenced;
> remove when the file can be compile-verified).

### Key Source Modules

| Path | Purpose |
|---|---|
| `main.py` | Main `TradingBot` orchestrator; entry point |
| `backtest.py` | Backtesting engine |
| `src/execution/signal_aggregator.py` | `PerformanceWeightedAggregator` — primary signal fusion |
| `src/execution/council_aggregator.py` | `InstitutionalCouncilAggregator` — advanced alternative aggregator |
| `src/strategies/` | Three concrete strategies inheriting `BaseStrategy` |
| `src/ai/hybrid_validator.py` | `HybridSignalValidator` — AI-based signal accept/reject |
| `src/ai/sniper.py` | `OHLCSniper` (CNN-LSTM) — **disconnected**, not in the live scoring pipeline |
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

### Risk Management Rules (behaviour to be aware of)
- **Circuit breaker**: stops all trading if daily loss ≥ 3% or drawdown ≥ 15% (`risk_management.*`)
- **Aggregate risk cap**: `risk_management.max_total_open_risk` (currently `0.25` = 25% of equity at risk across all open positions). A load-time guard in `main.py` refuses to start if any `max_total_open_risk` is outside `(0, 1.0]`. (Note: `portfolio.max_total_open_risk` is an unused legacy mirror — keep it ≤ 1.0 too.)
- **Position sizing**: 1–2% risk per trade; sized against the VTM's *effective* regime-adaptive ATR stop (`VeteranTradeManager.compute_effective_atr_multiplier`), not the raw config base.
- **Partial exits / targets**: per-asset in `config.json` (`risk.partial_targets` / `risk.partial_sizes`), e.g. BTC `[1.5,2.5,4.0]` @ `[0.45,0.30,0.25]`; ADX-conditioned at runtime. Not a fixed 45/30/25.
- **Runner trail**: `risk.runner_trail_atr_multiplier` (currently 1×ATR — flagged as too tight in the audit; Phase 4 candidate).
- **Trailing/breakeven**: regime- and Livermore-aware; early-lock + breakeven managed by the VTM.
- **MTF gate**: 1H entry signals rejected when 4H regime is counter-trend; plus Livermore hard-veto and 4H silent-zone holds.

### Audit & remediation (2026-06)
A full audit and phased fix plan live in the repo root:
- `AUDIT_2026-06-19.md` — findings (strategy netting, MRS-silent, over-veto, execution bugs).
- `REMEDIATION_PLAN.md` — phased plan with implementation-status block. Phases 0–2 done & deployed; Phase 3 (alpha) is gated on a ~1-week funnel/shadow soak.
- Observability: `logs/funnel/` (signal funnel + AI rejects), `logs/shadow/` (durable shadow gate scorecard). Analyze with `python scripts/analyze_observability.py --days 7`.

### Adding a New Strategy
1. Inherit from `src/strategies/base_strategy.py::BaseStrategy`
2. Implement `generate_signal(df) -> dict` returning `{"signal": int, "confidence": float, ...}`
3. Register in `src/execution/signal_aggregator.py` alongside the existing three strategies
4. Add per-asset config block in `config.json`
