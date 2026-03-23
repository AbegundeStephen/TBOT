# Production Veteran Trade Manager (VTM)
## Complete Documentation & Usage Guide

---

## 📋 Table of Contents
1. [What It Does](#what-it-does)
2. [How It Works](#how-it-works)
3. [Asset Profiles](#asset-profiles)
4. [Profit Locking Comparison](#profit-locking-comparison)
5. [API Reference](#api-reference)
6. [Live Trading Examples](#live-trading-examples)
7. [Best Practices](#best-practices)

---

## 🎯 What It Does

The Production VTM is an **intelligent trade management system** that automates risk management for live trading. It replaces manual stop-loss and take-profit decisions with a sophisticated system that:

### Core Functions:
1. **Sets Smart Stop Losses** - Uses Gemini's structure-based pivot detection + ATR buffers
2. **Takes Partial Profits** - Exits in 3 stages to lock gains while letting winners run
3. **Trails Stops** - Automatically moves stops to protect profits on the final portion
4. **Locks Profits** - Moves stop to break-even after gains are achieved
5. **Adapts to Assets** - Different strategies for BTC (volatile) vs GOLD (stable)

### What Makes It "Veteran"?
- **Not a rookie system** - Avoids tight stops that get you shaken out
- **Based on actual market structure** - Finds real support/resistance (pivot points)
- **Lets winners run** - Uses partials + trailing stops instead of  targets
- **Asset-aware** - BTC needs 8-15% stops, GOLD needs 2-5% stops
- **Production-ready** - Real-time updates, comprehensive logging, error handling

---

## 🔧 How It Works

### Step 1: Entry & Initialization
```python
manager = VeteranTradeManager(
    entry_price=50000,      # Your entry price
    side='long',            # 'long' or 'short'
    asset='BTC',            # 'BTC' or 'GOLD'
    high=price_highs,       # Last 50-100 bars of high prices
    low=price_lows,         # Last 50-100 bars of low prices
    close=price_closes,     # Last 50-100 bars of close prices
    account_balance=10000,  # Your account size
    account_risk=0.015      # Risk 1.5% per trade
)
```

**What happens during initialization:**

1. **Loads Asset Profile**
   - BTC: Wide stops (8-15%), aggressive targets, 20% trail
   - GOLD: Tight stops (2-5%), conservative targets, 12% trail

2. **Finds Structure-Based Stop (Gemini's Method)**
   ```
   Scans backwards 20-30 bars looking for:
   - LONG: V-shape pivot low (left > center < right)
   - SHORT: Λ-shape pivot high (left < center > right)
   
   Example for LONG:
   Bar -5: Low = $49,500 ← left neighbor
   Bar -4: Low = $48,000 ← CENTER (V-shape pivot!) ✓
   Bar -3: Low = $49,200 ← right neighbor
   
   This $48,000 is REAL support (price bounced here)
   ```

3. **Applies ATR Buffer**
   ```
   ATR = $1,200 (current market volatility)
   Buffer = ATR × 1.5 = $1,800
   
   Stop = Pivot Low - Buffer
   Stop = $48,000 - $1,800 = $46,200
   ```

4. **Enforces Min/Max Constraints**
   ```
   BTC limits: 8% min, 15% max from entry
   Entry: $50,000
   Min stop: $50,000 × (1 - 0.15) = $42,500
   Max stop: $50,000 × (1 - 0.08) = $46,000
   
   Final stop: $46,200 (within range ✓)
   ```

5. **Calculates Profit Targets**
   ```
   Risk = Entry - Stop = $50,000 - $46,200 = $3,800
   
   Target 1: Entry + (Risk × 1.5) = $50,000 + $5,700 = $55,700
   Target 2: Entry + (Risk × 2.5) = $50,000 + $9,500 = $59,500
   Target 3: Entry + (Risk × 4.0) = $50,000 + $15,200 = $65,200
   ```

6. **Calculates Position Size**
   ```
   Account: $10,000
   Risk per trade: 1.5% = $150
   Trade risk: $3,800 per BTC
   
   Position size = $150 / $3,800 = 0.0395 BTC
   
   If BTC drops to stop at $46,200:
   Loss = 0.0395 BTC × $3,800 = $150 ✓
   ```

---

### Step 2: Ongoing Management

#### **Every Price Update (Real-Time)**
```python
exit_info = manager.update_with_current_price(52000)
```

Checks:
1. **Stop Loss** - Is price below stop?
2. **Partial Targets** - Did we hit $55,700 / $59,500 / $65,200?
3. **Profit Lock** - If enabled and +1% gain, move stop to break-even
4. **Trailing Stop** - If runner active, trail stop below recent high

#### **Every Candle Close**
```python
exit_info = manager.update_with_new_bar(
    new_high=52500,
    new_low=51800,
    new_close=52000
)
```

Same checks as above, plus updates the price history for ATR calculations.

---

### Step 3: Exit Scenarios

#### **Scenario A: Price Rises (Winner)**
```
Entry: $50,000
Stop: $46,200

Price → $55,700: 🎯 Target 1 Hit!
  - Exit 33% of position
  - Keep 67% running
  - Move stop to $50,050 (break-even)
  - You've locked profit on 33%, can't lose on rest

Price → $59,500: 🎯 Target 2 Hit!
  - Exit another 33% (now 66% total out)
  - Keep 34% running
  - Activate 20% trailing stop

Price → $67,000: 📈 New high!
  - Trail stop updates: $67,000 × (1 - 0.20) = $53,600
  - Your 34% runner is protected at $53,600

Price → $54,000: 🏃 Trailing Stop Hit!
  - Exit final 34%
  
TOTAL RESULT:
  - 33% exited @ +11.4% = +3.76%
  - 33% exited @ +19.0% = +6.27%
  - 34% exited @ +8.0% = +2.72%
  - TOTAL: +12.75% on full position
```

#### **Scenario B: Price Falls (Loser)**
```
Entry: $50,000
Stop: $46,200

Price → $46,200: 🛑 Stop Hit!
  - Exit 100% of position
  - Loss: -7.6%
  - Account loss: $150 (1.5% of $10,000)
```

#### **Scenario C: Price Rises Then Falls (Profit Lock)**
```
Entry: $50,000
Stop: $46,200
Early Profit Lock: Enabled at 1%

Price → $50,500 (+1%): 🔒 Profit Locked!
  - Stop moves: $46,200 → $50,100
  - Now protected at break-even

Price → $52,000: Still safe

Price → $50,100: 🔒 Stopped at Break-Even
  - Exit 100%
  - P&L: +0.2% (minimal gain, but no loss!)
```

---

## 📊 Asset Profiles

### Bitcoin (BTC)
```python
{
    'min_stop_pct': 0.08,          # 8% minimum (BTC can swing 5-10% daily)
    'max_stop_pct': 0.15,          # 15% maximum
    'atr_multiplier': 2.0,         # Wide ATR buffer (volatile)
    'pivot_lookback': 30,          # Look back 30 bars for pivots
    'partial_targets': [1.5, 2.5, 4.0],  # Aggressive targets
    'partial_sizes': [0.33, 0.33, 0.34],  # Exit evenly
    'runner_trail_pct': 0.20,      # 20% trail (room to breathe)
    'time_stop_bars': 30,          # Give 30 bars before timing out
    'use_ema_structure': False     # BTC too whippy for EMA exits
}
```

**Why these settings:**
- BTC can drop 10% and bounce back the same day
- Tight stops = death by 1000 cuts
- Wide stops + big targets = catch the 50-100% moves

### Gold (GOLD/XAUUSD)
```python
{
    'min_stop_pct': 0.02,          # 2% minimum (Gold is stable)
    'max_stop_pct': 0.05,          # 5% maximum
    'atr_multiplier': 1.5,         # Tighter ATR buffer
    'pivot_lookback': 20,          # Look back 20 bars
    'partial_targets': [1.5, 2.5, 3.5],  # More conservative
    'partial_sizes': [0.40, 0.30, 0.30],  # Take more profit early
    'runner_trail_pct': 0.12,      # 12% trail (tighter)
    'time_stop_bars': 20,          # Exit dead trades faster
    'use_ema_structure': True      # Gold respects EMA better
}
```

**Why these settings:**
- Gold rarely moves more than 3-5% in a day
- Tighter stops work because volatility is lower
- Take 40% profit at first target (vs 33% for BTC)
- Shorter trails prevent giving back gains

---

## 🔒 Profit Locking Comparison

### Example Setup
- **Entry:** $50,000 BTC
- **Stop:** $46,000 (8% below)
- **Account:** $10,000
- **Risk:** 1.5% = $150

### Option 1: No Early Lock (Default)
```
Time    Price     Event                      Stop Location
────────────────────────────────────────────────────────────
00:00   $50,000   Entry                      $46,000
01:00   $50,500   +1% gain                   $46,000 (no change)
02:00   $51,000   +2% gain                   $46,000 (no change)
03:00   $55,700   +11.4% → Target 1 hit!     $50,050 (break-even)
04:00   $57,000   +14% gain                  $50,050
05:00   $59,500   +19% → Target 2 hit!       $50,050
        
        Runner now trails with 20% stop
        
06:00   $65,000   +30% (new high)            $52,000 (trails up)
07:00   $52,500   Drop                       $52,000 (trail hit!)

RESULT:
  33% @ $55,700 = +$188 (+11.4%)
  33% @ $59,500 = +$313 (+19.0%)
  34% @ $52,500 = +$83  (+5.0%)
  TOTAL: +$584 (+11.7% on full position)
```

### Option 2: Early Lock at 1% Gain
```
Time    Price     Event                      Stop Location
────────────────────────────────────────────────────────────
00:00   $50,000   Entry                      $46,000
01:00   $50,500   +1% → PROFIT LOCKED!       $50,100 (break-even)
02:00   $51,000   +2% gain                   $50,100
03:00   $55,700   +11.4% → Target 1 hit!     $50,050 (already at BE)
04:00   $57,000   +14% gain                  $50,050
05:00   $59,500   +19% → Target 2 hit!       $50,050
        
        Runner trails
        
06:00   $65,000   +30% (new high)            $52,000
07:00   $52,500   Drop                       $52,000 (trail hit!)

RESULT: SAME as Option 1 (+$584)

BUT if price had dropped after 01:00:
  Option 1: Would hit $46,000 stop = -$158 loss (-7.6%)
  Option 2: Would hit $50,100 stop = +$3 gain (+0.2%) ✓
```

### Option 3: Early Lock at 0.5% Gain (Too Aggressive)
```
Time    Price     Event                      Stop Location
────────────────────────────────────────────────────────────
00:00   $50,000   Entry                      $46,000
00:30   $50,250   +0.5% → PROFIT LOCKED!     $50,100 (break-even)
00:45   $50,100   Minor dip                  $50,100 (STOPPED OUT!)

RESULT: +$3 (+0.2%)

BUT price continued:
03:00   $55,700   Would have hit Target 1    (missed +11.4%)
05:00   $59,500   Would have hit Target 2    (missed +19%)

Locked profit TOO EARLY, missed the real move!
```

---

## 🔑 Recommendations

### For Gold Scalping (High Frequency)
```python
manager = VeteranTradeManager(
    asset='GOLD',
    enable_early_profit_lock=True,
    early_lock_threshold_pct=0.01,  # Lock at 1%
    ...
)
```
✅ Protects your 84% win rate  
✅ Prevents winners from becoming losers  
✅ Good for 1-5% target moves

### For BTC Position Trading (Low Frequency)
```python
manager = VeteranTradeManager(
    asset='BTC',
    enable_early_profit_lock=False,  # Default
    ...
)
```
✅ Gives trades room to develop  
✅ Doesn't get shaken out by volatility  
✅ Catches 20-50% moves

### For BTC Day Trading (Medium Frequency)
```python
manager = VeteranTradeManager(
    asset='BTC',
    enable_early_profit_lock=True,
    early_lock_threshold_pct=0.02,  # Lock at 2%
    ...
)
```
✅ Balance between protection and profit  
✅ Survives overnight volatility  
✅ Good for 5-15% target moves

---

## 📚 API Reference

### Initialization
```python
VeteranTradeManager(
    entry_price: float,
    side: str,  # 'long' or 'short'
    asset: str,  # 'BTC' or 'GOLD'
    high: np.ndarray,  # Historical highs
    low: np.ndarray,   # Historical lows
    close: np.ndarray, # Historical closes
    account_balance: float,
    account_risk: float = 0.015,  # Default 1.5%
    atr_period: int = 14,
    custom_profile: Optional[Dict] = None,
    enable_early_profit_lock: bool = False,
    early_lock_threshold_pct: float = 0.01
)
```

### Main Methods

#### `update_with_current_price(price: float) → Optional[Dict]`
Call this **every minute** or on every tick for real-time monitoring.

Returns:
```python
{
    'reason': ExitReason.TAKE_PROFIT_1,  # Or STOP_LOSS, etc.
    'price': 55700.00,
    'size': 0.33  # 33% of position
}
```

#### `update_with_new_bar(high, low, close) → Optional[Dict]`
Call this **when candle closes** (hourly, daily, etc.).

Same return format as above.

#### `get_current_levels() → Dict`
Get all current trade information:
```python
{
    'entry_price': 50000.00,
    'current_price': 52000.00,
    'stop_loss': 50050.00,
    'initial_stop': 46000.00,
    'next_target': 55700.00,
    'all_targets': [55700, 59500, 65200],
    'remaining_position_pct': 1.0,  # 100%
    'pnl_pct': 4.0,  # +4%
    'bars_in_trade': 12,
    'partials_hit': 0,
    'runner_active': False,
    'highest_reached': 52500.00,
    'lowest_reached': 49800.00
}
```

---

## 💻 Live Trading Examples

### Example 1: CCXT Integration (Crypto Exchange)
```python
import ccxt
from production_vtm import VeteranTradeManager

class CryptoBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': 'YOUR_KEY',
            'secret': 'YOUR_SECRET'
        })
        self.manager = None
        
    def enter_trade(self, symbol='BTC/USDT'):
        # Get historical data
        ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
        high = np.array([x[2] for x in ohlcv])
        low = np.array([x[3] for x in ohlcv])
        close = np.array([x[4] for x in ohlcv])
        
        current_price = close[-1]
        
        # Initialize VTM
        self.manager = VeteranTradeManager(
            entry_price=current_price,
            side='long',
            asset='BTC',
            high=high,
            low=low,
            close=close,
            account_balance=self.get_balance(),
            account_risk=0.015,
            enable_early_profit_lock=False  # Position trading
        )
        
        # Execute entry
        size = self.manager.position_size
        self.exchange.create_market_buy_order(symbol, size)
        
        print(f"✅ Entered BTC LONG @ ${current_price:,.2f}")
        print(f"   Size: {size:.6f} BTC")
        print(f"   Stop: ${self.manager.initial_stop_loss:,.2f}")
        print(f"   Targets: {self.manager.take_profit_levels}")
        
    def monitor_loop(self, symbol='BTC/USDT'):
        """Run this in a loop every 1 minute"""
        while self.manager:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Check for exits
            exit_info = self.manager.update_with_current_price(current_price)
            
            if exit_info:
                self.execute_exit(symbol, exit_info)
            
            time.sleep(60)  # Check every minute
    
    def execute_exit(self, symbol, exit_info):
        reason = exit_info['reason']
        size_pct = exit_info['size']
        price = exit_info['price']
        
        # Calculate exit size
        current_position = self.get_position_size(symbol)
        exit_amount = current_position * size_pct
        
        # Execute
        self.exchange.create_market_sell_order(symbol, exit_amount)
        
        print(f"🚪 EXIT: {reason.value}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Size: {size_pct:.0%} ({exit_amount:.6f} BTC)")
        
        if size_pct >= 0.99:  # Full exit
            self.manager = None
```

### Example 2: MetaTrader 5 Integration (Forex/Gold)
```python
import MetaTrader5 as mt5
from production_vtm import VeteranTradeManager

class MT5Bot:
    def __init__(self):
        mt5.initialize()
        self.manager = None
        
    def enter_gold_trade(self):
        symbol = "XAUUSD"
        
        # Get historical data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        high = np.array([x['high'] for x in rates])
        low = np.array([x['low'] for x in rates])
        close = np.array([x['close'] for x in rates])
        
        current_price = close[-1]
        
        # Initialize VTM for GOLD
        self.manager = VeteranTradeManager(
            entry_price=current_price,
            side='long',
            asset='GOLD',
            high=high,
            low=low,
            close=close,
            account_balance=mt5.account_info().balance,
            account_risk=0.015,
            enable_early_profit_lock=True,  # Scalping mode
            early_lock_threshold_pct=0.01   # Lock at 1%
        )
        
        # Calculate lot size
        lot_size = self.manager.position_size / 100  # MT5 uses lots
        
        # Place order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": current_price,
            "sl": self.manager.initial_stop_loss,
            "tp": self.manager.take_profit_levels[0],  # First partial
            "deviation": 10,
            "magic": 123456,
            "comment": "VTM Entry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        print(f"✅ GOLD LONG @ ${current_price:,.2f}")
```

---

## ✅ Best Practices

### 1. Data Quality
```python
# ❌ BAD: Not enough history
high = np.array([last_10_bars])  # Too short!

# ✅ GOOD: Sufficient history for ATR and pivots
high = np.array([last_100_bars])  # Enough for calculations
```

### 2. Real-Time Updates
```python
# ❌ BAD: Only checking on candle close
def on_new_bar():
    check_exit()  # Only runs once per hour!

# ✅ GOOD: Check frequently + on candle close
def every_minute():
    manager.update_with_current_price(price)  # Real-time
    
def on_new_bar():
    manager.update_with_new_bar(h, l, c)  # Also on close
```

### 3. Position Sizing
```python
# ❌ BAD: Ignoring VTM's calculated size
size = 0.1  # Arbitrary

# ✅ GOOD: Use VTM's risk-calculated size
size = manager.position_size  # Calculated for 1.5% risk
```

### 4. Exit Execution
```python
# ❌ BAD: Ignoring partial exits
if exit_info:
    close_all()  # Loses the partial profit strategy!

# ✅ GOOD: Respect the exit size
if exit_info:
    size_to_close = position * exit_info['size']
    close_partial(size_to_close)
```

### 5. Error Handling
```python
# ❌ BAD: No error handling
manager = VeteranTradeManager(...)  # Could fail

# ✅ GOOD: Catch and handle errors
try:
    manager = VeteranTradeManager(...)
except Exception as e:
    logger.error(f"VTM init failed: {e}")
    # Use fallback stop loss
    stop_loss = entry_price * 0.98
```

---

## 🎓 Summary

**Production VTM** is a complete trade management solution that:

1. **Calculates stops intelligently** using market structure (pivots) + volatility (ATR)
2. **Takes partial profits** at 3 levels to lock gains progressively
3. **Trails stops** on the final portion to capture big moves
4. **Locks profits** early (optional) or after first partial (default)
5. **Adapts to assets** with BTC and GOLD-specific profiles
6. **Provides real-time updates** for live trading monitoring

**Use it when:**
- You have good entry signals (from your aggregator/AI)
- You want to automate exit management
- You're tired of manually adjusting stops
- You want to lock profits without cutting winners short

**Don't use it for:**
- Backtesting (it's too sophisticated, kills trade frequency)
- Assets you don't understand (stick to BTC/GOLD profiles)
- Ultra-tight scalping (use early profit lock if you must)

**Bottom line:** This is production-grade risk management that lets you focus on finding good entries while it handles the exits professionally.