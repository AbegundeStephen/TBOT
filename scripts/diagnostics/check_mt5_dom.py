#!/usr/bin/env python3
"""
MT5 Depth-of-Market (DOM) availability check.

Context: TBOT's TransitionDetector FLOW component (order_flow_score) is only
ever populated for BTC, because BTC is the only asset with a real order-flow
data source wired up (Binance CVD/L2 via src/execution/cvd_consumer.py).
Every MT5-routed asset (GOLD, USOIL, EURUSD, EURJPY, USTEC, GBPAUD, GBPUSD,
USDJPY) defaults cvd_trend/order_book_imbalance/depth_data to 0/0.0/None,
so FLOW is structurally guaranteed to read 0.00 for these assets.

This script checks whether the broker actually exposes real Level-2 depth
for these symbols via MetaTrader5's market_book_get() API. Many brokers do
NOT populate DOM for FX/CFD/synthetic instruments they don't originate —
if this returns empty/None for all symbols, the honest conclusion is that
there is no real order-flow source available for non-BTC assets, and the
right fix is reweighting TransitionDetector.WEIGHTS to stop giving FLOW
0.25 of the vote when it can never contribute evidence — not building a
fake proxy.

Usage:
  python scripts/diagnostics/check_mt5_dom.py
  python scripts/diagnostics/check_mt5_dom.py --symbols XAUUSDm USTECm USOILm

Must be run on the machine with the MT5 terminal installed (Windows) —
the MetaTrader5 Python package talks to a local terminal process, it has
no remote/headless mode.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed, or not running on Windows "
          "with a local MT5 terminal. This check cannot run in a Linux sandbox.")
    sys.exit(1)


def load_mt5_config():
    config_path = Path("config/config.json")
    if not config_path.exists():
        config_path = Path("config/config.template.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    mt5_cfg = config.get("api", {}).get("mt5", {})
    if os.getenv("MT5_LOGIN"):
        mt5_cfg["login"] = int(os.getenv("MT5_LOGIN"))
    if os.getenv("MT5_PASSWORD"):
        mt5_cfg["password"] = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        mt5_cfg["server"] = os.getenv("MT5_SERVER")

    # Collect every MT5 symbol configured for the bot, for convenience.
    symbols = []
    for asset_name, asset_cfg in config.get("assets", {}).items():
        if asset_cfg.get("exchange") == "mt5" and asset_cfg.get("symbol"):
            symbols.append((asset_name, asset_cfg["symbol"]))

    return mt5_cfg, symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Specific MT5 symbols to check (default: every MT5 asset in config.json)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Seconds to keep the book subscription open per symbol, to let snapshots arrive",
    )
    args = parser.parse_args()

    mt5_cfg, configured_symbols = load_mt5_config()

    if not mt5.initialize(
        login=mt5_cfg.get("login"),
        password=mt5_cfg.get("password"),
        server=mt5_cfg.get("server"),
        path=mt5_cfg.get("path"),
    ):
        print(f"MT5 initialize() failed, error = {mt5.last_error()}")
        sys.exit(1)

    print("MT5 connection OK\n")

    if args.symbols:
        targets = [(s, s) for s in args.symbols]
    else:
        targets = configured_symbols

    results = {}
    for asset_name, symbol in targets:
        print(f"--- {asset_name} ({symbol}) ---")

        if not mt5.symbol_select(symbol, True):
            print(f"  symbol_select failed: {mt5.last_error()}")
            results[symbol] = "symbol_select_failed"
            continue

        subscribed = mt5.market_book_add(symbol)
        if not subscribed:
            print(f"  market_book_add failed (broker likely doesn't support DOM "
                  f"for this symbol): {mt5.last_error()}")
            results[symbol] = "book_add_failed"
            continue

        time.sleep(args.duration)
        book = mt5.market_book_get(symbol)
        mt5.market_book_release(symbol)

        if not book:
            print("  market_book_get returned empty — no real DOM data for this symbol.")
            results[symbol] = "empty"
        else:
            print(f"  market_book_get returned {len(book)} levels:")
            for level in book[:10]:
                side = "BID" if level.type == mt5.BOOK_TYPE_BUY else "ASK"
                print(f"    {side} price={level.price} volume={level.volume}")
            results[symbol] = f"{len(book)}_levels"
        print()

    mt5.shutdown()

    print("=== SUMMARY ===")
    for symbol, outcome in results.items():
        verdict = "REAL DOM AVAILABLE" if outcome.endswith("_levels") else "no usable DOM"
        print(f"  {symbol}: {outcome}  →  {verdict}")


if __name__ == "__main__":
    main()
