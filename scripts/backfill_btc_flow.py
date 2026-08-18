#!/usr/bin/env python3
"""
backfill_btc_flow.py — BTC Flow Batch (17-Aug ratification), Segment 3c.

One-shot backfill of BTC-FLOW's bench dataset: 15 months of 1H BTCUSDT
volume + taker-buy split, full funding-rate history over the same window,
and open interest history (Binance caps OI history at ~30 days regardless
of what's requested — collected anyway, capped window noted in the CSV
header).

This is collection only, same "harvest and compute, never decide" rule as
btc_flow_harvester.py — nothing here reads a config flag or touches a
score. Output goes to data/btc_flow_backfill.csv for Desire's offline
study: taker/OI/funding conditioning must clear the standard IS/OOS
qualification bar against the BTC proof population before any of it is
proposed for a live judge or gate.

Usage:
  python scripts/backfill_btc_flow.py [--months 15] [--out data/btc_flow_backfill.csv]

Public, keyless Binance endpoints only — no API keys created, stored, or
read anywhere in this script.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_OIH = "https://fapi.binance.com/futures/data/openInterestHist"
_TIMEOUT = 10
_OI_CAP_DAYS = 30  # Binance's own retention limit for this endpoint — not our choice


def _get(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] {url} failed after {retries} attempts: {e}", file=sys.stderr)
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_klines(start_ms: int, end_ms: int) -> dict:
    """1H BTCUSDT klines -> {bar_ts_epoch_sec: {volume, taker_buy_ratio, close}}."""
    out = {}
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        rows = _get(
            _SPOT_KLINES,
            {"symbol": "BTCUSDT", "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        calls += 1
        if not rows:
            break
        for b in rows:
            vol = float(b[5])
            taker = float(b[9])
            bar_ts = int(b[0] // 1000 // 3600 * 3600)
            out[bar_ts] = {
                "volume": vol,
                "taker_buy_ratio": (taker / vol) if vol > 0 else None,
                "close": float(b[4]),
            }
        last_open = rows[-1][0]
        next_cursor = last_open + 3600_000  # advance one hour past the last bar
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < 1000:
            break
    print(f"  klines: {len(out)} bars over {calls} calls")
    return out


def fetch_funding(start_ms: int, end_ms: int) -> dict:
    """Funding history -> {bar_ts_epoch_sec (hour-floored): funding_rate_pct}."""
    out = {}
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        rows = _get(
            _FUNDING,
            {"symbol": "BTCUSDT", "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        calls += 1
        if not rows:
            break
        for r in rows:
            ts_sec = int(r["fundingTime"]) // 1000
            bar_ts = ts_sec // 3600 * 3600
            out[bar_ts] = float(r["fundingRate"]) * 100
        last_time = rows[-1]["fundingTime"]
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < 1000:
            break
    print(f"  funding: {len(out)} events over {calls} calls")
    return out


def fetch_oi(end_ms: int) -> dict:
    """Open interest history — Binance caps this at ~30 days regardless of
    the requested window, so this only ever covers the tail of the range."""
    out = {}
    start_ms = end_ms - _OI_CAP_DAYS * 24 * 3600_000
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        rows = _get(
            _OIH,
            {"symbol": "BTCUSDT", "period": "1h", "startTime": cursor, "endTime": end_ms, "limit": 500},
        )
        calls += 1
        if not rows:
            break
        for r in rows:
            bar_ts = int(r["timestamp"]) // 1000 // 3600 * 3600
            out[bar_ts] = float(r["sumOpenInterest"])
        last_time = rows[-1]["timestamp"]
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) < 500:
            break
    print(f"  open interest: {len(out)} bars over {calls} calls (capped at ~{_OI_CAP_DAYS}d by Binance)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=15)
    ap.add_argument("--out", type=str, default="data/btc_flow_backfill.csv")
    args = ap.parse_args()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Backfilling BTC-FLOW: {start_dt.date()} -> {end_dt.date()} ({args.months} months)")

    print("Fetching klines (volume + taker split)...")
    klines = fetch_klines(start_ms, end_ms)

    print("Fetching funding history...")
    funding = fetch_funding(start_ms, end_ms)

    print("Fetching open interest history (capped window)...")
    oi = fetch_oi(end_ms)
    oi_sorted_ts = sorted(oi.keys())
    oi_start = datetime.fromtimestamp(oi_sorted_ts[0], tz=timezone.utc) if oi_sorted_ts else None

    all_bars = sorted(klines.keys())
    if not all_bars:
        print("No klines fetched — aborting, nothing written.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# BTC-FLOW backfill: {start_dt.isoformat()} to {end_dt.isoformat()}"])
        w.writerow([
            f"# open_interest coverage: "
            + (f"{oi_start.isoformat()} to {end_dt.isoformat()} (Binance {_OI_CAP_DAYS}d retention cap — "
               f"the rest of this file's oi/oi_delta_pct columns are blank by design, not missing data)"
               if oi_start else "no OI data returned")
        ])
        w.writerow(["bar_ts", "close", "volume", "taker_buy_ratio", "funding_rate", "oi"])
        for bar_ts in all_bars:
            k = klines[bar_ts]
            fund = funding.get(bar_ts)
            cur_oi = oi.get(bar_ts)
            w.writerow([bar_ts, k["close"], k["volume"], k["taker_buy_ratio"], fund, cur_oi])

    print(f"Wrote {len(all_bars)} rows to {args.out}")
    print("This file is bench material — see Segment 3's docstring: taker/OI/funding")
    print("must clear the standard IS/OOS qualification bar before any of it reaches a score.")


if __name__ == "__main__":
    main()
