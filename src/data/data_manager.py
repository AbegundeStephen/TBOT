#**src/data/data_manager.py**


"""
Data Manager - Handles data retrieval from Binance and MT5
Implements strict anti-leakage protocols
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import MetaTrader5 as mt5
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class DataManager:
    """
    Unified data manager for multiple exchanges
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.binance_client = None
        self.mt5_initialized = False
        
    def initialize_binance(self) -> bool:
        """Initialize Binance API connection"""
        try:
            api_key = self.config['api']['binance']['api_key']
            api_secret = self.config['api']['binance']['api_secret']
            
            if self.config['api']['binance'].get('testnet', True):
                self.binance_client = Client(api_key, api_secret, testnet=True)
            else:
                self.binance_client = Client(api_key, api_secret)
            
            # Test connection
            self.binance_client.ping()
            logger.info("Binance API initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Binance API: {e}")
            return False
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            mt5_config = self.config['api']['mt5']
            
            if not mt5.initialize(
                path=mt5_config.get('path'),
                login=mt5_config['login'],
                password=mt5_config['password'],
                server=mt5_config['server']
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            logger.info("MT5 initialized successfully")
            self.mt5_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MT5: {e}")
            return False
    
    def fetch_binance_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance
        """
        if self.binance_client is None:
            raise RuntimeError("Binance client not initialized")
        
        try:
            klines = self.binance_client.get_historical_klines(
                symbol,
                interval,
                start_date,
                end_date
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp')
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            raise
    