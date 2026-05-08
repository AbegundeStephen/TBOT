# TBOT Implementation Guide: Pattern Diagnostics, MR Fix, BTC Stale Fix, Data Sources

**Date:** 2026-05-07  
**Scope:** Four issues identified from live logs, mapped to exact code locations.

---

## ISSUE 1: Pattern=None — Diagnostics & Fix

**Root Cause:** The 5 institutional patterns (DISTRIBUTION, ACCUMULATION, LIQUIDITY_HUNT, SPRING_BREAKOUT, MA_DEFENSE) in `_score_confluence()` at line 1012-1055 of `signal_aggregator.py` are AND chains. Every condition must pass. Right now they never fire because the conditions are strict and there's zero visibility into which specific conditions fail.

**What to do:** Add diagnostic logging directly inside `_score_confluence()`. Not a new file — inline it where the patterns are evaluated.

### Step 1: Add diagnostic logging to _score_confluence()

**File:** `src/execution/signal_aggregator.py`  
**Location:** Line 1012, inside `_score_confluence()`, before the pattern evaluation block.

**Insert this block BEFORE line 1014 (`# PATTERN A: Institutional Distribution`):**

```python
        # ─── PATTERN DIAGNOSTICS ──────────────────────────────────────
        # Shows exactly which conditions pass/fail for each pattern.
        # MISSING fields = upstream module not writing to CompositeState (bug).
        # ❌ fields = market conditions don't match (working as intended).
        _diag_fields = {
            "lifecycle_phase": state.lifecycle_phase,
            "regime_age_ratio": f"{state.regime_age_ratio:.2f}",
            "choch_detected": state.choch_detected,
            "structural_decay": state.structural_decay,
            "absorption_detected": state.absorption_detected,
            "conviction_dying": state.conviction_dying,
            "distance_zscore": f"{state.distance_zscore:.2f}",
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            "effort_result_zscore": f"{state.effort_result_zscore:.2f}",
            "outside_bar": state.outside_bar,
            "failed_breakout": state.failed_breakout,
            "coiled_spring": state.coiled_spring,
            "ema_50_status": state.ema_50_status,
            "ema_50_reclassified": state.ema_50_reclassified,
        }
        logger.info(
            f"[PATTERN DIAG] {self.asset_type} state: "
            + " | ".join(f"{k}={v}" for k, v in _diag_fields.items())
        )

        # Evaluate each pattern and log which condition blocks it
        _dist_checks = {
            "phase∈ESTABLISHED/FADING": state.lifecycle_phase in ("ESTABLISHED", "FADING"),
            f"age_ratio>{1.3}": state.regime_age_ratio > 1.3,
            "choch_or_decay": state.choch_detected or state.structural_decay,
            "absorption_or_dying": state.absorption_detected or state.conviction_dying,
            f"dist_z>{1.5}": state.distance_zscore > 1.5,
        }
        _accum_checks = {
            "phase∈PICKUP/CONFIRM": state.lifecycle_phase in ("PICKUP", "CONFIRMATION"),
            f"age_ratio<{0.8}": state.regime_age_ratio < 0.8,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "no_absorption": not state.absorption_detected,
        }
        _liq_checks = {
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            f"effort_z>{2.0}": state.effort_result_zscore > 2.0,
            "outside_or_failed": state.outside_bar or state.failed_breakout,
        }
        _spring_checks = {
            "coiled_spring": state.coiled_spring,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
        }
        _ma_checks = {
            "ema50=DEFENDED": state.ema_50_status == "DEFENDED",
            "ema50=SUPPORT": state.ema_50_reclassified == "SUPPORT",
            "phase∈CONFIRM/ESTAB": state.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED"),
            f"age_ratio<{1.5}": state.regime_age_ratio < 1.5,
        }

        for _pname, _pchecks in [
            ("DISTRIBUTION", _dist_checks),
            ("ACCUMULATION", _accum_checks),
            ("LIQUIDITY_HUNT", _liq_checks),
            ("SPRING_BREAKOUT", _spring_checks),
            ("MA_DEFENSE", _ma_checks),
        ]:
            _passed = sum(_pchecks.values())
            _total = len(_pchecks)
            _blocker = next((k for k, v in _pchecks.items() if not v), None)
            _status = "✅ MATCHED" if all(_pchecks.values()) else f"❌ {_passed}/{_total}"
            logger.info(
                f"[PATTERN CHECK] {self.asset_type} {_pname}: {_status}"
                + (f" — first blocker: {_blocker}" if _blocker else "")
            )
```

**Do NOT change the actual pattern evaluation logic below (lines 1014-1059).** The diagnostics sit above and log what's happening. The pattern evaluation itself stays identical.

### Step 2: Fix regime_age_ratio population

**Problem identified:** `regime_age_ratio` defaults to 0.0 in CompositeState and is only updated if a regime transition has occurred. Check `_update_trend_lifecycle()` at line 741.

**File:** `src/execution/signal_aggregator.py`  
**Location:** After `_update_trend_lifecycle()` is called and before `_score_confluence()`.

Find where `_update_trend_lifecycle` is called (search for `_update_trend_lifecycle` in the file). After that call, add:

```python
        # Ensure regime_age_ratio is always populated
        _start = self._regime_start_time.get(self.asset_type)
        if _start:
            _age_hours = (datetime.now() - _start).total_seconds() / 3600
            state.regime_age_hours = _age_hours
            _median = state.median_regime_duration or 12.0
            state.regime_age_ratio = _age_hours / _median
```

This ensures DISTRIBUTION can fire (needs `regime_age_ratio > 1.3`) and MA_DEFENSE can fire (needs `regime_age_ratio < 1.5`).

### Step 3: Evaluate

After deploying, run for 2-3 cycles and check logs. You should see output like:

```
[PATTERN DIAG] USTEC state: lifecycle_phase=ESTABLISHED | regime_age_ratio=0.00 | choch_detected=False | ...
[PATTERN CHECK] USTEC DISTRIBUTION: ❌ 1/5 — first blocker: age_ratio>1.3
[PATTERN CHECK] USTEC ACCUMULATION: ❌ 0/5 — first blocker: phase∈PICKUP/CONFIRM
[PATTERN CHECK] USTEC LIQUIDITY_HUNT: ❌ 0/4 — first blocker: sweep_detected
[PATTERN CHECK] USTEC SPRING_BREAKOUT: ❌ 0/3 — first blocker: coiled_spring
[PATTERN CHECK] USTEC MA_DEFENSE: ❌ 1/4 — first blocker: ema50=DEFENDED
```

If `regime_age_ratio=0.00` appears after the fix in Step 2, then `_regime_start_time` isn't being set — trace back to `_update_trend_lifecycle`.

---

## ISSUE 2: MR=0.000 Across All Assets

**Root Cause:** MR's scorecard in `mean_reversion.py` lines 508-605 requires **score ≥ 3** from 4 pillars (max 5 points) OR micro-reversion trigger. The pillars are:

| Pillar | Points | Condition | Why it's probably failing |
|--------|--------|-----------|--------------------------|
| STRETCH | 2 | Price > 2×ATR from EMA50 OR outside BB | Markets are trending but not extreme enough |
| DIVERGENCE | 1 | RSI divergence over 60-bar window | Requires sustained price/RSI disagreement |
| LIQUIDITY SWEEP | 1 | Price broke 100-bar low/high | Requires a genuine range break |
| EXHAUSTION | 1 | Candlestick pattern at BB extreme (<0.15 or >0.85) | Rare by definition |

So MR needs STRETCH (2pts) + at least one other. In the current market conditions (USTEC bullish but not parabolic, USOIL bearish but not crashed, EURJPY slightly bearish), none of the assets are at extreme enough levels for the STRETCH to fire.

**This is partially by design — MR is supposed to be rare.** But it's TOO rare. The `2.0 × ATR` stretch requirement is very tight for 1H candles.

### Step 1: Add MR diagnostic logging

**File:** `src/strategies/mean_reversion.py`  
**Location:** After line 531 (after all scorecard pillars are evaluated, before micro-reversion section).

**Insert:**

```python
            # ================================================================
            # DIAGNOSTIC: Log scorecard breakdown every cycle
            # ================================================================
            logger.info(
                f"[MR SCORECARD] {self.asset}:\n"
                f"  STRETCH:   long={stretch_long} short={stretch_short} "
                f"(dist_from_ema50={abs(ema_50[-1] - current_close):.2f}, "
                f"2xATR={2.0 * atr[-1]:.2f}, "
                f"bb_pos={bb_pos[-1]:.3f})\n"
                f"  DIVERGENCE: long={div_long} short={div_short}\n"
                f"  SWEEP:     long={sweep_long} short={sweep_short} "
                f"(100bar_low={np.min(lookback_100) if len(lookback_100) > 0 else 'N/A':.2f}, "
                f"100bar_high={np.max(highback_100) if len(highback_100) > 0 else 'N/A':.2f})\n"
                f"  SCORES:    long={score_long}/5 short={score_short}/5 "
                f"(need ≥3)"
            )
```

### Step 2: Consider loosening the STRETCH threshold

**File:** `src/strategies/mean_reversion.py`  
**Location:** Lines 514-515.

**Current:**
```python
stretch_long = (ema_50[-1] - current_close > 2.0 * atr[-1]) or (current_close < bb_lower[-1])
stretch_short = (current_close - ema_50[-1] > 2.0 * atr[-1]) or (current_close > bb_upper[-1])
```

**Option A — Lower to 1.5× ATR (moderate loosening):**
```python
stretch_long = (ema_50[-1] - current_close > 1.5 * atr[-1]) or (current_close < bb_lower[-1])
stretch_short = (current_close - ema_50[-1] > 1.5 * atr[-1]) or (current_close > bb_upper[-1])
```

**Option B — Add a "near stretch" half-point (graduated scoring):**
```python
# Full stretch = 2 pts
stretch_long = (ema_50[-1] - current_close > 2.0 * atr[-1]) or (current_close < bb_lower[-1])
stretch_short = (current_close - ema_50[-1] > 2.0 * atr[-1]) or (current_close > bb_upper[-1])
if stretch_long: score_long += 2
if stretch_short: score_short += 2

# Half stretch = 1 pt (new — catches moderate pullbacks)
if not stretch_long and (ema_50[-1] - current_close > 1.2 * atr[-1]):
    score_long += 1
if not stretch_short and (current_close - ema_50[-1] > 1.2 * atr[-1]):
    score_short += 1
```

This means a moderate pullback (1.2× ATR) + divergence + candlestick pattern = 3 points = signal. Without loosening, only an extreme pullback (2× ATR) + one other indicator fires.

**Recommendation:** Deploy the diagnostic logging first (Step 1). Run for 24-48 hours. If the logs show STRETCH consistently failing while other pillars pass, apply Option B. If everything fails, the problem is different.

### Step 3: Check the micro-reversion path

The micro-reversion trigger (lines 554-574) requires 4H data. Check that `df_4h` is actually being passed to MR's `generate_signal()`. In the logs, you see `df_4h` being fetched for all MT5 assets but the question is whether it reaches MR.

**File:** `src/execution/signal_aggregator.py`  
**Location:** Line 2360.

```python
df_4h = governor_data.get('df_4h') if governor_data else None
```

Verify that `governor_data` contains `df_4h` for MT5 assets. Add a one-liner after line 2362:

```python
logger.debug(f"[MR INPUT] {self.asset_type}: df_4h={'present, ' + str(len(df_4h)) + ' bars' if df_4h is not None else 'MISSING'}")
```

If `df_4h=MISSING`, the micro-reversion path is dead for that asset — it can never fire.

---

## ISSUE 3: BTC Stale Price Detection False Positive

**Root Cause:** The stale detector at line 2137 reads `df["close"].iloc[-1]` — the last **completed** hourly candle's close. Within an hour, this value doesn't change because Binance kline API returns only closed candles. With a 30-minute threshold, the detector triggers every hour roughly 30 minutes after the candle closes, blocking BTC for ~50% of all time.

The logs confirm this: BTC shows cached prices updating (79759, 79835, 79906, 79948) but the stale check is stuck comparing against 79801.08 — the close of a specific completed candle that remains the most recent completed candle for the duration of the hour.

### Fix Option A: Use current tick price instead of candle close (recommended)

**File:** `src/execution/signal_aggregator.py`  
**Location:** Lines 2136-2164.

**Replace the `_current_price` source for BTC with the cached price from the price cache (which updates from API ticker/book data), not the candle close:**

```python
            # T1.5: STALE PRICE DETECTION
            from datetime import datetime as _dt
            _now = _dt.now()

            # For BTC: use the price cache (updated from recent kline fetches across
            # different timeframes), not df close which only changes on candle close.
            if self.asset_type in ("BTC", "BTCUSDT"):
                _current_price = getattr(self, '_price_cache', {}).get(
                    self.asset_type, float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
                )
            else:
                _current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
```

Then ensure the price cache is updated. Find where BTC data is fetched in the main loop and add:

```python
# After any Binance fetch for BTC, update the price cache
if not hasattr(self, '_price_cache'):
    self._price_cache = {}
self._price_cache["BTC"] = float(df["close"].iloc[-1])
```

### Fix Option B: Increase BTC stale threshold to 65 minutes (simpler)

Since hourly candles can remain unchanged for up to 59 minutes, set the BTC threshold above 60.

**File:** `src/execution/signal_aggregator.py`  
**Location:** Line 291.

**Change:**
```python
self._stale_threshold_minutes = 30   # default (crypto / Binance)
```

**To:**
```python
self._stale_threshold_minutes = 65   # default — must exceed 1H candle duration
```

This is the fastest fix. BTC will still be checked for stale data, but only if the same candle close persists for over 65 minutes (which would genuinely indicate a frozen feed).

### Fix Option C: Compare against the most recent TWO candle closes (best)

The real question isn't "has the last candle close changed?" but "has a new candle appeared?" Compare the candle timestamp, not the price:

**Replace lines 2136-2164 with:**

```python
            # T1.5: STALE PRICE DETECTION (candle-age based)
            from datetime import datetime as _dt
            _now = _dt.now()
            if len(df) > 0:
                _last_candle_time = df.index[-1]
                if hasattr(_last_candle_time, 'timestamp'):
                    _candle_age_minutes = (_now.timestamp() - _last_candle_time.timestamp()) / 60
                else:
                    _candle_age_minutes = 0

                _stale_limit = self._stale_thresholds.get(
                    self.asset_type, self._stale_threshold_minutes
                )
                # For 1H candles, a candle older than 120 min means no new candle arrived
                # (should be ~60 min max if data is fresh)
                _effective_limit = max(_stale_limit, 120) if self.asset_type in ("BTC", "BTCUSDT") else _stale_limit

                if _candle_age_minutes > _effective_limit:
                    logger.warning(
                        f"[STALE] ❌ {self.asset_type} last candle is {_candle_age_minutes:.0f}min old "
                        f"— blocking signal evaluation"
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "regime": "UNKNOWN",
                        "reasoning": f"stale_candle_{_candle_age_minutes:.0f}min",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0, "mr_confidence": 0.0,
                        "tf_signal": 0, "tf_confidence": 0.0,
                        "ema_signal": 0, "ema_confidence": 0.0,
                    }
```

**Recommendation:** Fix Option B is a one-line change that unblocks BTC immediately. Apply it now. Implement Option C when you next do a proper release.

---

## ISSUE 4: Alternative Data Sources for BTC Real-Time Pricing

**The problem:** Binance REST API klines return completed candles only. You're fetching 1H candles, so the "latest price" only updates once per hour. The rate limiting (1200 requests/min for IP-based, 6000 weight/min for keyed requests) isn't the bottleneck — the issue is architectural: kline endpoints don't give you a current price.

### Option A: Use the existing WebSocket for current price (zero cost, zero new dependency)

You already have `cvd_consumer.py` running a WebSocket connection to Binance aggTrade stream. It's receiving every BTC trade in real time. Add a `last_price` field to CVDConsumer.

**File:** `src/execution/cvd_consumer.py`

**Add to `__init__`:**
```python
        self._last_price = 0.0
```

**Add inside the `async for msg in stream:` loop (line 32-39), after the CVD update:**
```python
                    self._last_price = float(msg['p'])  # 'p' = price field in aggTrade
```

**Add a getter:**
```python
    def get_last_price(self) -> float:
        """Returns the most recent trade price from WebSocket."""
        if self.is_stale():
            return 0.0
        return self._last_price
```

Then in `signal_aggregator.py`, use this for the stale check instead of `df["close"].iloc[-1]`:

```python
            # For BTC, use WebSocket last trade price instead of candle close
            if self.asset_type in ("BTC", "BTCUSDT") and governor_data:
                _ws_price = governor_data.get("last_trade_price", 0.0)
                if _ws_price > 0:
                    _current_price = _ws_price
                else:
                    _current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
            else:
                _current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
```

And in `main.py`, feed it into governor_data wherever you build that dict:

```python
governor_data["last_trade_price"] = cvd_consumer.get_last_price()
```

**Cost:** Free. Already connected. One field addition.

### Option B: Add Binance ticker endpoint (1 extra REST call per cycle)

The `/api/v3/ticker/price` endpoint returns the current price and costs weight 2 per call. One call every 5 minutes = negligible.

**File:** `src/data/data_manager.py`

**Add method:**
```python
    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Fetch current price via ticker endpoint. Weight: 2."""
        try:
            client = self._get_data_client(prefer_live=True)
            ticker = client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"[TICKER] Failed to get {symbol} price: {e}")
            return 0.0
```

**Cost:** Free, weight 2 per call, well within limits.

### Option C: Free alternatives to Binance for price data

If you want to reduce Binance dependency entirely:

| Source | Type | Cost | Latency | Notes |
|--------|------|------|---------|-------|
| **CoinGecko API** | REST | Free (30 calls/min) | ~5s | `/simple/price?ids=bitcoin&vs_currencies=usd` — good for stale check, not for order flow |
| **Kraken REST** | REST | Free (15 calls/min) | ~1s | `/0/public/Ticker?pair=XBTUSD` — reliable backup |
| **CryptoCompare** | REST + WS | Free (100k calls/month) | ~2s | WebSocket available for real-time |
| **Bybit REST** | REST | Free | ~1s | `/v5/market/tickers?category=linear&symbol=BTCUSDT` — good reliability |
| **Binance WebSocket** | WS | Free | <100ms | **Already connected.** The `aggTrade` stream you're using IS real-time BTC price. |

**Recommendation:** Option A. You're literally already receiving every BTC trade via WebSocket. Adding `self._last_price = float(msg['p'])` to CVDConsumer gives you a sub-100ms real-time price feed with zero additional API calls, zero rate limit impact, and zero new dependencies. The stale price problem disappears entirely.

For the order book / limit order tracking question: the `bookTicker` stream you already have (`_book_listener`) gives best bid/ask but not depth. If you need full L2 depth (to see where limit orders cluster), add:

```python
    async def _depth_listener(self, bm):
        """L2 depth stream — top 20 levels."""
        try:
            async with bm.depth_socket("btcusdt", depth=20) as stream:
                async for msg in stream:
                    if not self._running:
                        break
                    self._bids = msg.get('bids', [])  # [[price, qty], ...]
                    self._asks = msg.get('asks', [])
                    self._depth_last_update = datetime.now()
        except Exception as e:
            logger.debug(f"[L2] Depth error: {e}")
```

This is free (it's a WebSocket stream, not a REST call). It gives you the order book snapshot at each update, which you can use to find large limit order clusters.

---

## DEPLOYMENT ORDER

1. **BTC Stale Fix (Option B — one line)** — Unblocks BTC signals immediately
2. **Pattern Diagnostics (Issue 1, Step 1)** — Gives visibility into why patterns don't fire
3. **MR Diagnostics (Issue 2, Step 1)** — Shows exactly why MR returns 0
4. **regime_age_ratio fix (Issue 1, Step 2)** — Unblocks DISTRIBUTION and MA_DEFENSE patterns
5. **CVD last_price addition (Issue 4, Option A)** — Permanent fix for BTC stale detection
6. **MR threshold adjustment (Issue 2, Step 2)** — Only after 24-48h of diagnostic data

## HOW TO EVALUATE

After each deployment step, check 3 consecutive cycles in the logs:

| Step | What to look for in logs |
|------|--------------------------|
| BTC stale fix | `[STALE]` should NOT appear for BTC. BTC should show `Signal: +1` or `-1` instead of `Signal: 0 (Quality: 0.00)` |
| Pattern diagnostics | `[PATTERN CHECK]` lines for every asset every cycle, showing pass/fail counts |
| MR diagnostics | `[MR SCORECARD]` lines showing which pillars pass/fail and the actual numeric distances |
| regime_age_ratio | `regime_age_ratio` in `[PATTERN DIAG]` should show non-zero values (e.g., 0.45, 1.72) |
| CVD last_price | `[STALE]` completely gone for BTC. Price in stale check should change every few seconds, not every hour |
| MR threshold | MR starts generating non-zero signals. Track win rate in shadow trader for 1 week before going live |
