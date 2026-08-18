"""
BTC FLOW HARVESTER (17-Aug ratification).
Public, keyless Binance market data for the one asset whose reference
market publishes its order flow for free. REST only — no websockets, no
API keys, shared-hosting friendly. Every call is timeout-capped and
failure-tolerant: a dead feed degrades the bot to exactly today's
behavior, announced by a log line, never an exception.

Design rule, non-negotiable: harvests and computes; never decides. No
judge, gate, or VTM logic imports this module directly — it feeds the
frame (data_manager.py's volume overlay) and composite state; consumption
beyond the volume column goes through the offline bench first.

Harvested per 1H close (klines call) and per 15-min cache (futures):
  volume            real BTCUSDT spot volume        (klines idx 5)
  taker_buy_ratio   aggressive-buyer share of vol   (klines idx 9 / idx 5)
  oi, oi_delta_pct  open interest + 1h change       (futures/data/openInterestHist)
  funding_rate      current funding                 (fapi premiumIndex)
  basis_pct         perp mark vs spot close          (premiumIndex vs kline)

Appends every reading to data/btc_flow_1h.csv so history survives
restarts and offline studies can grade candidate features against the
proof population before anything touches a score.
"""
from __future__ import annotations

import time
import csv
import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_SPOT = "https://api.binance.com/api/v3/klines"
_PREM = "https://fapi.binance.com/fapi/v1/premiumIndex"
_OIH = "https://fapi.binance.com/futures/data/openInterestHist"
_CSV = "data/btc_flow_1h.csv"
_TIMEOUT = 5


class BTCFlowHarvester:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_bar_ts = None
        self._cache = {}  # latest computed fields
        self._cache_ts = 0.0
        self._fail_streak = 0

    # ── public API ──────────────────────────────────────────────────────
    def refresh(self) -> dict:
        """Called from the data-refresh step each cycle. Cheap when the
        1H bar hasn't rolled; one klines call when it has; futures fields
        on a 15-min cache. Returns the latest field dict (possibly stale,
        with age); {} while disabled or before first success."""
        if not self.enabled:
            return {}
        try:
            now = time.time()
            k = self._klines()
            if k and k["bar_ts"] != self._last_bar_ts:
                self._last_bar_ts = k["bar_ts"]
                # Pass the just-fetched spot close through explicitly — the
                # cache still holds the PREVIOUS cycle's close_spot at this
                # point (k hasn't been merged in yet), so basis_pct would
                # otherwise be computed against a stale spot price on every
                # fresh-bar cycle, the only time this branch runs.
                fut = self._futures(spot_hint=k.get("close_spot"))
                rec = {**k, **fut, "fetched_at": now, "src": "binance"}
                self._cache = rec
                self._cache_ts = now
                self._fail_streak = 0
                self._append_csv(rec)
                logger.info(
                    f"[BTC-FLOW] bar={rec['bar_ts']} vol={rec['volume']:.0f} "
                    f"taker={rec.get('taker_buy_ratio', float('nan')):.2f} "
                    f"oiΔ={rec.get('oi_delta_pct', float('nan')):+.2f}% "
                    f"fund={rec.get('funding_rate', float('nan')):+.4f}% "
                    f"basis={rec.get('basis_pct', float('nan')):+.3f}%"
                )
            elif now - self._cache_ts > 900 and self._cache:
                fut = self._futures()
                if fut:
                    self._cache.update(fut)
                    self._cache_ts = now
            out = dict(self._cache)
            if out:
                out["age_min"] = (now - out.get("fetched_at", now)) / 60.0
            return out
        except Exception as e:
            self._fail_streak += 1
            if self._fail_streak in (2, 10):
                logger.warning(f"[BTC-FLOW] degraded (streak {self._fail_streak}): {e}")
            return dict(self._cache) if self._cache else {}

    def hourly_volume(self, bar_ts_utc) -> Optional[float]:
        """Volume for the given closed UTC hour, or None if we don't have
        exactly that bar — the overlay must never guess."""
        c = self._cache
        if c and c.get("bar_ts") == bar_ts_utc:
            return c.get("volume")
        return None

    # ── internals ───────────────────────────────────────────────────────
    def _klines(self):
        r = requests.get(
            _SPOT, params={"symbol": "BTCUSDT", "interval": "1h", "limit": 2}, timeout=_TIMEOUT
        )
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 2:
            return None
        b = rows[-2]  # last CLOSED bar
        vol = float(b[5])
        taker = float(b[9])
        return {
            "bar_ts": int(b[0] // 1000 // 3600 * 3600),  # UTC hour epoch
            "volume": vol,
            "close_spot": float(b[4]),
            "taker_buy_ratio": (taker / vol) if vol > 0 else None,
        }

    def _futures(self, spot_hint: Optional[float] = None):
        out = {}
        try:
            p = requests.get(_PREM, params={"symbol": "BTCUSDT"}, timeout=_TIMEOUT).json()
            out["funding_rate"] = float(p.get("lastFundingRate", 0)) * 100
            mark = float(p.get("markPrice", 0))
            spot = spot_hint or self._cache.get("close_spot") or mark
            if spot:
                out["basis_pct"] = (mark - spot) / spot * 100
        except Exception:
            pass
        try:
            h = requests.get(
                _OIH, params={"symbol": "BTCUSDT", "period": "1h", "limit": 2}, timeout=_TIMEOUT
            ).json()
            if isinstance(h, list) and len(h) >= 2:
                a, b = float(h[-2]["sumOpenInterest"]), float(h[-1]["sumOpenInterest"])
                out["oi"] = b
                out["oi_delta_pct"] = (b - a) / a * 100 if a else None
        except Exception:
            pass
        return out

    def _append_csv(self, rec):
        try:
            os.makedirs("data", exist_ok=True)
            new = not os.path.exists(_CSV)
            with open(_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(
                        ["bar_ts", "volume", "taker_buy_ratio", "oi", "oi_delta_pct", "funding_rate", "basis_pct"]
                    )
                w.writerow(
                    [
                        rec.get(k)
                        for k in (
                            "bar_ts",
                            "volume",
                            "taker_buy_ratio",
                            "oi",
                            "oi_delta_pct",
                            "funding_rate",
                            "basis_pct",
                        )
                    ]
                )
        except Exception as e:
            logger.warning(f"[BTC-FLOW] csv append failed: {e}")
