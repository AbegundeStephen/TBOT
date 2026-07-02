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

    # Lower number = higher authority. A 4H/1D historical fetch's last close
    # is hours/a day stale relative to a 1H or finer fetch — it must never
    # overwrite a fresher price just because it happened to complete last.
    _TIMEFRAME_PRIORITY = {
        "tick": 0, "m1": 1, "1m": 1, "m5": 2, "5m": 2, "m15": 3, "15m": 3,
        "m30": 4, "30m": 4, "h1": 5, "1h": 5, "h4": 6, "4h": 6, "d1": 7, "1d": 7,
    }
    _DEFAULT_PRIORITY = 99  # unknown timeframe — never assume it's authoritative

    def __init__(self, ttl: float = 300.0): # Default TTL is 300 seconds (5 minutes)
        """
        Initializes the cache.
        Args:
            ttl (float): Time-to-live in seconds for the cached price.
        """
        self._prices: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._priorities: Dict[str, int] = {}
        self._ttl = ttl
        logger.info(f"[PRICE_CACHE] Initialized with TTL: {self._ttl}s.")

    def set(self, symbol: str, price: float, timeframe: Optional[str] = None) -> bool:
        """
        Update the cache with a new price for a specific symbol.
        Returns True if the write was accepted, False if rejected by priority.

        Args:
            symbol (str): The asset symbol (e.g., 'BTCUSDT', 'XAUUSDm').
            price (float): The new price to cache.
            timeframe (str): Source timeframe (e.g. '1h', '4h', '1d'). A write
                from a coarser timeframe than the one already cached is
                ignored — last-writer-wins only applies within the same
                priority tier, never across tiers (see _TIMEFRAME_PRIORITY).
        """
        new_priority = self._TIMEFRAME_PRIORITY.get(
            (timeframe or "").lower(), self._DEFAULT_PRIORITY
        )
        current_priority = self._priorities.get(symbol, self._DEFAULT_PRIORITY + 1)
        if new_priority > current_priority:
            logger.debug(
                f"[PRICE_CACHE] Ignoring {symbol} price {price} from "
                f"coarser timeframe '{timeframe}' (priority {new_priority} > "
                f"cached priority {current_priority})"
            )
            return False
        self._prices[symbol] = price
        self._timestamps[symbol] = time.monotonic()
        self._priorities[symbol] = new_priority
        logger.debug(f"[PRICE_CACHE] Price for {symbol} set to {price} at {self._timestamps[symbol]} (tf={timeframe})")
        return True

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
