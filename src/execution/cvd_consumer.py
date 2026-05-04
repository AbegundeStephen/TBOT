"""
Cumulative Volume Delta consumer for BTC.
Single WebSocket, single running float, heartbeat guard.
"""
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CVDConsumer:
    def __init__(self):
        self._cvd = 0.0
        self._last_update = datetime.min
        self._running = False
        # F.6: L2 Order Book Imbalance
        self._best_bid_qty = 0.0
        self._best_ask_qty = 0.0
        self._book_last_update = datetime.min

    async def start(self):
        """Start the WebSocket consumer in a background task."""
        try:
            from binance import AsyncClient, BinanceSocketManager
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            self._running = True

            async with bm.aggtrade_socket("btcusdt") as stream:
                asyncio.create_task(self._book_listener(bm))
                async for msg in stream:
                    if not self._running:
                        break
                    qty = float(msg['q'])
                    if msg['m']:  # Seller was maker = buyer aggressed
                        self._cvd += qty
                    else:
                        self._cvd -= qty
                    self._last_update = datetime.now()
        except Exception as e:
            logger.error(f"[CVD] WebSocket error: {e}")
            self._running = False

    async def _book_listener(self, bm):
        """L2 best bid/ask imbalance — piggybacks on existing connection."""
        try:
            async with bm.symbol_book_ticker_socket("btcusdt") as stream:
                async for msg in stream:
                    if not self._running:
                        break
                    self._best_bid_qty = float(msg.get('B', 0))
                    self._best_ask_qty = float(msg.get('A', 0))
                    self._book_last_update = datetime.now()
        except Exception as e:
            logger.debug(f"[L2] Book ticker error: {e}")

    def get_order_book_imbalance(self) -> float:
        """Returns -1.0 to +1.0. Positive = bid heavy."""
        age = (datetime.now() - self._book_last_update).total_seconds()
        if age > 60:
            return 0.0
        total = self._best_bid_qty + self._best_ask_qty
        if total < 1e-8:
            return 0.0
        return (self._best_bid_qty - self._best_ask_qty) / total

    def is_wall_detected(self, threshold: float = 5.0) -> bool:
        """True if one side has 5x+ more quantity than the other."""
        if self._best_bid_qty < 1e-8 or self._best_ask_qty < 1e-8:
            return False
        ratio = max(self._best_bid_qty, self._best_ask_qty) /                 min(self._best_bid_qty, self._best_ask_qty)
        return ratio > threshold

    def get_trend(self) -> int:
        """Returns +1 (buyers dominant), -1 (sellers dominant), 0 (neutral/stale)."""
        # Heartbeat guard: if no update in 60s, data is stale
        age = (datetime.now() - self._last_update).total_seconds()
        if age > 60:
            return 0

        if self._cvd > 100:   # Threshold tuned per asset
            return 1
        elif self._cvd < -100:
            return -1
        return 0

    def is_stale(self) -> bool:
        return (datetime.now() - self._last_update).total_seconds() > 60

    def daily_reset(self):
        """Call at 00:00 UTC to prevent drift."""
        self._cvd = 0.0

    def stop(self):
        self._running = False
