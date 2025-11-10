**README.md**

```markdown
# Advanced Multi-Asset AI Trading Bot

AI-powered trading bot for BTC (Binance) and Gold (Exness MT5) using dual-strategy signal aggregation with rigorous data integrity controls.

## Features

- **Dual Strategy System**: Mean reversion (ranging markets) + Trend following (trending markets)
- **Signal Confirmation Filter**: Only trades when both strategies agree
- **Anti-Leakage Protocols**: Strict time-series validation, no future data in training
- **Noise Reduction**: Statistical outlier removal, robust feature engineering
- **Multi-Asset Support**: BTC via Binance API, Gold via MT5
- **Comprehensive Risk Management**: Position sizing, stop-loss, take-profit
- **Full Backtesting**: Realistic simulation with slippage and commissions

## Prerequisites

- Python 3.10+
- Binance API account (testnet recommended for testing)
- Exness MT5 account
- MetaTrader 5 installed (for Gold trading)

## Installation

### 1. Clone and Setup

```bash
git clone  crypto-gold-trading-bot
cd crypto-gold-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install TA-Lib

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ta-lib
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
Download pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install:
```bash
pip install TA_Lib‑0.4.XX‑cpXX‑cpXXm‑win_amd64.whl
```

### 3. Configure Settings

```bash
cp config/config.template.json config/config.json
```

Edit `config/config.json` with your API credentials:

```json
{
  "api": {
    "binance": {
      "api_key": "YOUR_BINANCE_API_KEY",
      "api_secret": "YOUR_BINANCE_SECRET",
      "testnet": true
    },
    "mt5": {
      "login": 12345678,
      "password": "YOUR_MT5_PASSWORD",
      "server": "Exness-MT5Trial",
      "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    }
  }
}
```

## Usage

### Step 1: Train Models

Train both AI strategies on historical data:

```bash
python train.py
```

**Expected Output:**
- Trained models saved to `models/` directory
- Training metrics displayed (accuracy, CV scores)
- Test data saved for backtesting

**Training incorporates:**
- ✅ Time-series cross-validation (prevents data leakage)
- ✅ Outlier removal (noise reduction)
- ✅ Feature scaling and normalization
- ✅ Class imbalance handling
- ✅ No lookahead bias in label generation

### Step 2: Run Backtest

Validate strategy performance on out-of-sample data:

```bash
python backtest.py
```

**Key Metrics:**
- **Sharpe Ratio** (Target > 1.0)
- **Maximum Drawdown** (Target < 10%)
- **Win Rate**
- **Total Trades**
- **Cumulative Return**

### Step 3: Deploy Live (Paper Trading)

Start the bot in paper trading mode:

```bash
python main.py
```

**The bot will:**
1. Connect to Binance (testnet) and MT5
2. Load trained ML models
3. Fetch real-time market data every 5 minutes
4. Generate signals using both strategies
5. Execute trades only when both strategies agree
6. Apply strict risk management (stop-loss, take-profit)

**To stop:** Press `Ctrl+C`

## Architecture

### Modular Design (Future-Proof for Microservices)

```
┌─────────────────────────────────────────────────┐
│            Signal Aggregator (Master)           │
│         ┌──────────────────────────────┐        │
│         │  Confirmation Filter Logic   │        │
│         │  (Both strategies must agree)│        │
│         └──────────────────────────────┘        │
└────────┬────────────────────────┬────────────────┘
         │                        │
    ┌────▼─────┐            ┌────▼─────┐
    │  Mean    │            │  Trend   │
    │Reversion │            │Following │
    │ Strategy │            │ Strategy │
    │ (Range)  │            │ (Trend)  │
    └────┬─────┘            └────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │ Data Manager│
              │ (Binance +  │
              │     MT5)    │
              └──────┬──────┘
                     │
         ┌───────────┴───────────┐
         │                       │
   ┌─────▼─────┐          ┌─────▼─────┐
   │  Binance  │          │    MT5    │
   │  Handler  │          │  Handler  │
   │   (BTC)   │          │  (Gold)   │
   └───────────┘          └───────────┘
```

## Anti-Leakage & Noise Reduction Features

### 1. **Time-Series Cross-Validation**
- Uses `TimeSeriesSplit` for walk-forward validation
- Never trains on future data
- Respects temporal order

### 2. **Label Generation Without Lookahead Bias**
- Labels created from PAST outcomes only
- Forward-looking offset properly implemented
- Last N bars excluded from training

### 3. **Feature Engineering**
- All indicators use historical data only
- Proper warm-up periods enforced
- No future data leakage in calculations

### 4. **Outlier Removal**
- Z-score based statistical outlier detection
- Configurable threshold (default: 3 standard deviations)
- Reduces training noise

### 5. **Data Cleaning**
- Duplicate removal
- OHLC validation
- Missing value handling (forward-fill only)

## Risk Management

- **Position Sizing**: 1-2% of account equity per trade
- **Stop Loss**: 1.5% from entry (configurable)
- **Take Profit**: 3% from entry (configurable)
- **Max Daily Trades**: 10 (prevents overtrading)
- **Max Open Positions**: 2 (limits exposure)

## Configuration Parameters

### Strategy Parameters

**Mean Reversion:**
- Bollinger Bands period: 20
- BB standard deviation: 2.0
- RSI period: 14
- Stochastic K/D: 14/3

**Trend Following:**
- Fast MA: 50
- Slow MA: 200
- MACD: 12/26/9
- ADX period: 14
- ADX threshold: 25

### ML Parameters

- Model: RandomForestClassifier
- Trees: 200
- Max depth: 10
- Class weight: balanced (handles imbalance)
- CV splits: 5

## Logs and Monitoring

All activities are logged to:
- `logs/training.log` - Training process
- `logs/trading_bot.log` - Live trading operations

**Log entries include:**
- API requests/responses
- Strategy signals (Range, Trend, Final)
- Order placements and fills
- Errors and warnings

## Testing

Run unit tests:
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

**1. TA-Lib import error**
```bash
# Ensure TA-Lib C library is installed first, then:
pip install TA-Lib
```

**2. MT5 connection fails**
- Verify MT5 is installed and running
- Check login credentials in config.json
- Ensure server name is correct

**3. Binance API errors**
- Verify API keys are correct
- Check if testnet is enabled (`"testnet": true`)
- Ensure IP is whitelisted (if required)

**4. Insufficient training data**
- Increase date range in train.py
- Ensure stable internet connection during data fetch

## Production Deployment

### For Live Trading (Real Money)

**⚠️ WARNING: Only deploy to production after thorough testing! ⚠️**

1. **Update config.json:**
   ```json
   "testnet": false
   ```

2. **Deploy to VPS:**
   ```bash
   # Use screen or tmux for persistent sessions
   screen -S trading-bot
   python main.py
   # Detach: Ctrl+A then D
   ```

3. **Monitor logs:**
   ```bash
   tail -f logs/trading_bot.log
   ```

4. **Setup systemd service (Linux):**
   ```bash
   sudo nano /etc/systemd/system/trading-bot.service
   ```
   
   ```ini
   [Unit]
   Description=AI Trading Bot
   After=network.target

   [Service]
   Type=simple
   User=your-user
   WorkingDirectory=/path/to/crypto-gold-trading-bot
   ExecStart=/path/to/venv/bin/python main.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl enable trading-bot
   sudo systemctl start trading-bot
   ```

## Performance Targets

- **Sharpe Ratio**: > 1.0
- **Max Drawdown**: < 10%
- **Win Rate**: > 50%
- **Profit Factor**: > 1.5

## License

MIT License - See LICENSE file

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies and forex involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## Support

For issues and questions:
- Open an issue on GitHub
- Check logs for error details
- Ensure all dependencies are properly installed

---

**Built with ❤️ for algorithmic traders**
```

---

## 13. Git Workflow

```bash
# Initialize repository
git init
git add .
git commit -m "feat: Initial implementation of multi-asset trading bot

- Dual strategy system (mean reversion + trend following)
- Anti-leakage protocols for ML training
- Binance and MT5 integration
- Comprehensive backtesting framework
- Risk management and position sizing"

# Create .gitignore (already covered above)

# Optional: Push to remote
git remote add origin <your-repo-url>
git push -u origin main
```

---

## 14. Quick Start Commands

```bash
# Complete setup
cd crypto-gold-trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp config/config.template.json config/config.json
# Edit config.json with your credentials

# Train models
python train.py

# Backtest
python backtest.py

# Deploy (paper trading)
python main.py
```

---

## Summary

This implementation provides:

✅ **Complete modular architecture** (easy microservice transition)
✅ **Strict anti-leakage protocols** (time-series CV, no lookahead)
✅ **Noise reduction** (outlier removal, feature scaling)
✅ **Dual-strategy confirmation filter** (high-confidence trades only)
✅ **Multi-asset support** (BTC via Binance, Gold via MT5)
✅ **Comprehensive risk management**
✅ **Full backtesting with realistic simulation**
✅ **Production-ready logging and monitoring**
✅ **Git configuration and best practices**

The bot is designed to be deployed immediately for paper trading and can be easily scaled to live trading after validation.   ├── __init__.py
│       └── backtest_engine.py
├── models/
├── data/
├── logs/
├── tests/
├── notebooks/
├── train.py
├── backtest.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
EOF
```

---