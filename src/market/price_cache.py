# src/market/price_cache.py

import time
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class PriceCache:
    """
    A simple in-memory cache for the latest asset prices to minimize API calls.
    Stores prices per symbol.
    """
    def __init__(self, ttl: float = 300.0): # Default TTL is 300 seconds (5 minutes)
        """
        Initializes the cache.
        Args:
            ttl (float): Time-to-live in seconds for the cached price.
        """
        self._prices: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl
        logger.info(f"[PRICE_CACHE] Initialized with TTL: {self._ttl}s.")

    def set(self, symbol: str, price: float):
        """
        Update the cache with a new price for a specific symbol.
        Args:
            symbol (str): The asset symbol (e.g., 'BTCUSDT', 'XAUUSDm').
            price (float): The new price to cache.
        """
        self._prices[symbol] = price
        self._timestamps[symbol] = time.monotonic()
        logger.debug(f"[PRICE_CACHE] Price for {symbol} set to {price} at {self._timestamps[symbol]}")

    def get(self, symbol: str) -> Optional[float]:
        """
        Get the price for a specific symbol from the cache if it's not expired.
        Args:
            symbol (str): The asset symbol.
        Returns:
            Optional[float]: The price if it's within the TTL, otherwise None.
        """
        if symbol not in self._prices or self._prices[symbol] is None:
            return None
        
        if (time.monotonic() - self._timestamps[symbol]) < self._ttl:
            return self._prices[symbol]
        
        logger.debug(f"[PRICE_CACHE] Cached price for {symbol} is stale (TTL: {self._ttl}s).")
        return None # Return None if stale to force a refresh (or use get_last_known as fallback)

    def get_last_known(self, symbol: str) -> Optional[float]:
        """
        Get the last known price for a specific symbol from the cache, even if it's expired.
        Args:
            symbol (str): The asset symbol.
        Returns:
            Optional[float]: The last price that was set for the symbol, regardless of TTL.
        """
        return self._prices.get(symbol)

# Singleton instance to be used across the application
price_cache = PriceCache()
