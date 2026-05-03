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

    async def start(self):
        """Start the WebSocket consumer in a background task."""
        try:
            from binance import AsyncClient, BinanceSocketManager
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            self._running = True

            async with bm.aggtrade_socket("btcusdt") as stream:
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
