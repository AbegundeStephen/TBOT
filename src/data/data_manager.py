"""
Data Manager -  VERSION
Corrects date handling for Binance API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from typing import Optional, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DataManager:
    """Unified data manager for multiple exchanges"""

    def __init__(self, config: Dict):
        self.config = config
        self.binance_client = None
        self.mt5_initialized = False

    def initialize_binance(self) -> bool:
        """Initialize Binance API connection with dynamic API base URL."""
        try:
            api_config = self.config["api"]["binance"]
            api_key = api_config.get("api_key", "")
            api_secret = api_config.get("api_secret", "")

            logger.info(f"Initializing Binance API...")

            if not api_key or not api_secret or api_key == "YOUR_BINANCE_API_KEY":
                logger.warning("Binance API keys not configured, using public API only")
                self.binance_client = Client("", "")
            else:
                # Check if Futures trading is enabled
                enable_futures = self.config.get("assets", {}).get("BTC", {}).get("enable_futures", False)

                if api_config.get("testnet", True):
                    # Set the API base URL based on whether Futures is enabled
                    if enable_futures:
                        api_base = api_config.get("futures_api_base", "https://testnet.binancefuture.com")
                        logger.info(f"Using Futures Testnet API: {api_base}")
                        self.binance_client = Client(api_key, api_secret, testnet=True, tld="com")
                        self.binance_client.API_URL = api_base
                    else:
                        api_base = api_config.get("api_base", "https://testnet.binance.vision/api")
                        logger.info(f"Using Spot Testnet API: {api_base}")
                        self.binance_client = Client(api_key, api_secret, testnet=True, tld="com")
                        self.binance_client.API_URL = api_base
                else:
                    # Live trading
                    self.binance_client = Client(api_key, api_secret)

            # Verify the connection
            self.binance_client.ping()
            logger.info("Binance API initialized successfully")
            return True

        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Binance API: {e}")
            return False


    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection with detailed diagnostics"""
        try:
            import MetaTrader5 as mt5

            if "api" not in self.config or "mt5" not in self.config["api"]:
                logger.warning(
                    "MT5 configuration not found in config. Skipping MT5 initialization."
                )
                return False

            mt5_config = self.config["api"]["mt5"]

            path = mt5_config.get("path")
            login = mt5_config.get("login")
            password = mt5_config.get("password")
            server = mt5_config.get("server")

            if path == "null" or path == "None":
                path = None

            if not all([login, password, server]):
                logger.warning(
                    "MT5 credentials incomplete. Skipping MT5 initialization."
                )
                return False

            logger.info("=" * 60)
            logger.info("MT5 INITIALIZATION DIAGNOSTICS")
            logger.info("=" * 60)
            logger.info(f"MT5 Path: {path if path else 'Auto-detect'}")
            logger.info(f"Login: {login}")
            logger.info(f"Server: {server}")
            logger.info(
                f"MT5 Python Package Version: {mt5.__version__ if hasattr(mt5, '__version__') else 'Unknown'}"
            )

            logger.info("\nStep 1: Initializing MT5 terminal...")

            logger.info(
                "Attempting initialize without path (auto-detect running MT5)..."
            )
            init_result = mt5.initialize()

            logger.info("\nStep 1: Initializing MT5 terminal...")

            if not path:
                logger.info("No MT5 path provided, attempting auto-detect")
                init_result = mt5.initialize()
            else:
                logger.info(f"Initializing MT5 using explicit executable path: {path}")
                init_result = mt5.initialize(
                    path=path,
                    login=int(login),
                    password=str(password),
                    server=str(server),
                )

            if not init_result:
                error = mt5.last_error()
                logger.error(f"MT5 initialize() failed: {error}")
                return False


            if not init_result:
                error = mt5.last_error()
                logger.error(f"MT5 initialize() failed: {error}")
                logger.error("\nTROUBLESHOOTING STEPS:")
                logger.error("1. Make sure MetaTrader 5 is RUNNING and logged in")
                logger.error("2. Check if path is correct (should be folder, not .exe)")
                logger.error("3. Run Python as Administrator")
                logger.error("4. Check Windows Firewall settings")
                logger.error("5. Try closing MT5 and reopening it")
                return False

            logger.info("MT5 terminal initialized successfully")

            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info(f"\nMT5 Terminal Info:")
                logger.info(f"  Company: {terminal_info.company}")
                logger.info(f"  Name: {terminal_info.name}")
                logger.info(f"  Path: {terminal_info.path}")
                logger.info(f"  Connected: {terminal_info.connected}")
                logger.info(f"  Trade Allowed: {terminal_info.trade_allowed}")

            logger.info(f"\nStep 2: Logging into account {login}...")

            authorized = mt5.login(
                login=int(login), password=str(password), server=str(server)
            )

            if not authorized:
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                logger.error("\nPOSSIBLE CAUSES:")
                logger.error("1. Wrong login/password/server combination")
                logger.error("2. Account not approved for trading")
                logger.error(
                    "3. Server name incorrect (check in MT5: Tools > Options > Server)"
                )
                logger.error("4. Internet connection issue")

                account_info = mt5.account_info()
                if account_info:
                    logger.info(
                        f"Current account: {account_info.login} (trying to switch)"
                    )

                mt5.shutdown()
                return False

            logger.info("Login successful")

            logger.info("\nStep 3: Verifying account access...")
            account_info = mt5.account_info()

            if account_info is None:
                logger.error("Connected but cannot get account info")
                mt5.shutdown()
                return False

            logger.info("\n" + "=" * 60)
            logger.info("MT5 ACCOUNT INFORMATION")
            logger.info("=" * 60)
            logger.info(f"Account: {account_info.login}")
            logger.info(f"Name: {account_info.name}")
            logger.info(f"Server: {account_info.server}")
            logger.info(f"Balance: ${account_info.balance:.2f}")
            logger.info(f"Equity: ${account_info.equity:.2f}")
            logger.info(f"Margin: ${account_info.margin:.2f}")
            logger.info(f"Free Margin: ${account_info.margin_free:.2f}")
            logger.info(f"Leverage: 1:{account_info.leverage}")
            logger.info(f"Currency: {account_info.currency}")
            logger.info(f"Trade Mode: {account_info.trade_mode}")
            logger.info(f"Trade Allowed: {account_info.trade_allowed}")
            logger.info("=" * 60)

            logger.info("\nStep 4: Testing symbol access...")
            test_symbol = "XAUUSD"
            symbol_info = mt5.symbol_info(test_symbol)

            if symbol_info is None:
                logger.warning(f"Cannot access symbol {test_symbol}")
                logger.warning(
                    "This might be normal if symbol doesn't exist on this account"
                )
            else:
                logger.info(f"Symbol {test_symbol} accessible")
                logger.info(f"  Bid: {symbol_info.bid}")
                logger.info(f"  Ask: {symbol_info.ask}")
                logger.info(f"  Spread: {symbol_info.spread}")

            logger.info("\n" + "=" * 60)
            logger.info("MT5 INITIALIZATION COMPLETE")
            logger.info("=" * 60 + "\n")

            self.mt5_initialized = True
            return True

        except ImportError:
            logger.error("MetaTrader5 module not installed")
            logger.error("Install with: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during MT5 initialization: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_klines_with_retry(
        self, symbol: str, interval: str, startTime: int, endTime: int, limit: int
    ):
        """Helper method to fetch klines with retry logic."""
        return self.binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
        )

    def fetch_binance_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance with robust date handling and retry logic.

         Properly handles datetime strings with time components
        """
        if self.binance_client is None:
            raise RuntimeError("Binance client not initialized")

        try:
            #  Parse datetime strings that may include time
            # This handles both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" formats
            start_dt = pd.to_datetime(start_date)

            # If timezone-naive, assume UTC
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize("UTC")
            else:
                start_dt = start_dt.tz_convert("UTC")

            #  Handle end_date properly
            if end_date:
                end_dt = pd.to_datetime(end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize("UTC")
                else:
                    end_dt = end_dt.tz_convert("UTC")
            else:
                # If no end_date provided, use current time
                end_dt = pd.Timestamp.now(tz="UTC")

            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            logger.info(f"Fetching {symbol} data from {start_dt} to {end_dt} (UTC)")
            logger.info(f"Start timestamp: {start_ts} ({start_dt})")
            logger.info(f"End timestamp: {end_ts} ({end_dt})")

            all_klines = []
            current_start = start_ts
            max_iterations = 100  # Safety limit to prevent infinite loops
            iteration = 0

            while current_start < end_ts and iteration < max_iterations:
                iteration += 1

                try:
                    klines = self._fetch_klines_with_retry(
                        symbol=symbol,
                        interval=interval,
                        startTime=current_start,
                        endTime=end_ts,
                        limit=limit,
                    )
                except Exception as e:
                    logger.error(f"Error fetching batch {iteration}: {e}")
                    break

                if not klines:
                    logger.info(f"No more data available at timestamp {current_start}")
                    break

                all_klines.extend(klines)

                # Get the timestamp of the last candle
                last_timestamp = klines[-1][0]

                # If we got the same timestamp as before, we're stuck
                if last_timestamp <= current_start:
                    logger.warning(
                        f"Duplicate timestamp detected, stopping to prevent infinite loop"
                    )
                    break

                # Move to next batch (add 1ms to avoid duplicates)
                current_start = last_timestamp + 1

                logger.info(
                    f"Batch {iteration}: Fetched {len(klines)} bars, total: {len(all_klines)}"
                )

                # If we got fewer bars than the limit, we've reached the end
                if len(klines) < limit:
                    logger.info("Reached end of available data (partial batch)")
                    break

                # If we've reached the end timestamp, stop
                if last_timestamp >= end_ts:
                    logger.info("Reached requested end timestamp")
                    break

            if iteration >= max_iterations:
                logger.warning(
                    f"Stopped after {max_iterations} iterations (safety limit)"
                )

            if not all_klines:
                logger.error(
                    f"No data received for {symbol} from {start_date} to {end_date}"
                )
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.set_index("timestamp")

            # Remove duplicates (keep first occurrence)
            df = df[~df.index.duplicated(keep="first")]

            # Sort by timestamp
            df = df.sort_index()

            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            # Validate data range
            days_requested = (end_dt - start_dt).days
            days_received = (df.index[-1] - df.index[0]).days

            if df.index[0] > start_dt:
                missing_days = (df.index[0] - start_dt).days
                logger.warning(
                    f"Data starts at {df.index[0]}, but requested {start_dt}. "
                    f"Missing {missing_days} days of data at the beginning."
                )

            if df.index[-1] < end_dt:
                missing_hours = (end_dt - df.index[-1]).total_seconds() / 3600
                if missing_hours > 2:  # Only warn if significantly behind
                    logger.warning(
                        f"Data ends at {df.index[-1]}, but requested {end_dt}. "
                        f"Missing {missing_hours:.1f} hours of data at the end."
                    )

            logger.info(
                f"Coverage: {days_received}/{days_requested} days ({days_received/max(days_requested,1)*100:.1f}%)"
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_mt5_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
        count: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5

         Properly handles datetime strings with time components
        """
        if not self.mt5_initialized:
            raise RuntimeError("MT5 not initialized")

        try:
            import MetaTrader5 as mt5

            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }

            tf = timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

            #  Handle datetime strings with time components
            start_dt = pd.to_datetime(start_date)
            if end_date:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = datetime.now()

            # Convert to Python datetime (MT5 requires this)
            start_dt = (
                start_dt.to_pydatetime()
                if hasattr(start_dt, "to_pydatetime")
                else start_dt
            )
            end_dt = (
                end_dt.to_pydatetime() if hasattr(end_dt, "to_pydatetime") else end_dt
            )

            logger.info(f"Fetching {symbol} from MT5: {start_dt} to {end_dt}")

            rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

            if rates is None or len(rates) == 0:
                logger.error(f"No data received from MT5 for {symbol}")
                error = mt5.last_error()
                logger.error(f"MT5 Error: {error}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            df = df.set_index("timestamp")

            logger.info(f"Fetched {len(df)} bars for {symbol} from MT5")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except ImportError:
            logger.error("MetaTrader5 module not installed")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching MT5 data: {e}", exc_info=True)
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        if df.empty:
            logger.warning("Cannot clean empty DataFrame")
            return df

        df = df.copy()
        initial_len = len(df)

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
            df = df.ffill()
            df = df.dropna()

        invalid_bars = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["close"] <= 0)
            | (df["volume"] < 0)
        )

        if invalid_bars.any():
            invalid_count = invalid_bars.sum()
            logger.warning(f"Found {invalid_count} invalid OHLC bars, removing...")
            df = df[~invalid_bars]

        final_len = len(df)
        removed = initial_len - final_len

        if removed > 0:
            logger.info(
                f"Cleaned data: removed {removed} bars ({removed/initial_len*100:.1f}%)"
            )

        return df

    def split_train_test(
        self, df: pd.DataFrame, train_pct: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically"""
        if df.empty:
            logger.error("Cannot split empty DataFrame")
            return pd.DataFrame(), pd.DataFrame()

        split_idx = int(len(df) * train_pct)

        if split_idx < 100:
            logger.warning(f"Training set very small ({split_idx} bars)")

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train set: {len(train_df)} bars ({train_pct*100:.0f}%)")
        logger.info(f"Test set: {len(test_df)} bars ({(1-train_pct)*100:.0f}%)")
        logger.info(f"Train period: {train_df.index[0]} to {train_df.index[-1]}")
        logger.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")

        return train_df, test_df

    def get_latest_data(
        self, symbol: str, interval: str, lookback_bars: int = 500
    ) -> pd.DataFrame:
        """Get latest data for live trading"""
        try:
            interval_map = {
                "1m": timedelta(minutes=lookback_bars),
                "5m": timedelta(minutes=5 * lookback_bars),
                "15m": timedelta(minutes=15 * lookback_bars),
                "1h": timedelta(hours=lookback_bars),
                "4h": timedelta(hours=4 * lookback_bars),
                "1d": timedelta(days=lookback_bars),
            }

            delta = interval_map.get(interval, timedelta(hours=lookback_bars))
            start_date = (datetime.now(timezone.utc) - delta).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            df = self.fetch_binance_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )

            df = self.clean_data(df)

            if len(df) > lookback_bars:
                df = df.iloc[-lookback_bars:]

            return df

        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()

    def shutdown(self):
        """Cleanup connections"""
        if self.mt5_initialized:
            try:
                import MetaTrader5 as mt5

                mt5.shutdown()
                logger.info("MT5 shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down MT5: {e}")

        self.binance_client = None
