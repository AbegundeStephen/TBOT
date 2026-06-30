"""
Tick vs bar latency diagnostic.

Compares MT5's live tick timestamp against wall-clock time, and the latest
closed 1H bar's timestamp against wall-clock time, for each enabled asset.

Answers one question: is the lag in MT5's bar-finalization pipeline
(tick is fresh, bar is stale -> fixable by aggregating bars from ticks
locally) or in the broker's feed itself (tick is ALSO stale -> no amount
of local aggregation fixes it, need a different feed or broker)?

Run from the project root with the project's venv:
    venv\\Scripts\\python.exe scripts\\diagnostics\\check_feed_latency.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import MetaTrader5 as mt5

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"


def init_mt5():
    cfg = json.loads(CONFIG_PATH.read_text())
    mt5_cfg = cfg["api"]["mt5"]
    path = mt5_cfg.get("path")
    login = os.getenv("MT5_LOGIN") or mt5_cfg.get("login")
    password = os.getenv("MT5_PASSWORD") or mt5_cfg.get("password")
    server = os.getenv("MT5_SERVER") or mt5_cfg.get("server")

    if path in ("null", "None", "", None):
        ok = mt5.initialize()
    else:
        ok = mt5.initialize(
            path=path, login=int(login), password=str(password), server=str(server)
        )
    if not ok:
        print(f"mt5.initialize() failed: {mt5.last_error()}")
        sys.exit(1)

    if not mt5.login(login=int(login), password=str(password), server=str(server)):
        print(f"mt5.login() failed: {mt5.last_error()}")
        sys.exit(1)

    acc = mt5.account_info()
    print(f"Connected: {acc.login} on {server}\n")
    return cfg


def main():
    cfg = init_mt5()
    now = datetime.now(timezone.utc)

    symbols = []
    for name, a in cfg["assets"].items():
        if a.get("enabled") and a.get("exchange") == "mt5":
            symbols.append(a.get("symbol", name))

    print(f"Wall-clock UTC now: {now.isoformat()}")
    print("-" * 100)
    print(f"{'Symbol':<12}{'Tick Time (UTC)':<26}{'Tick Age':<12}{'Last 1H Bar (UTC)':<26}{'Bar Age':<10}")
    print("-" * 100)

    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            print(f"{symbol:<12} symbol_select failed: {mt5.last_error()}")
            continue

        tick = mt5.symbol_info_tick(symbol)
        tick_str, tick_age_str = "N/A", "N/A"
        if tick is not None and tick.time > 0:
            tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            tick_age = (now - tick_time).total_seconds() / 60.0
            tick_str = tick_time.strftime("%Y-%m-%d %H:%M:%S")
            tick_age_str = f"{tick_age:.1f} min"

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)
        bar_str, bar_age_str = "N/A", "N/A"
        if rates is not None and len(rates) > 0:
            bar_time = datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc)
            bar_age = (now - bar_time).total_seconds() / 3600.0
            bar_str = bar_time.strftime("%Y-%m-%d %H:%M:%S")
            bar_age_str = f"{bar_age:.2f} h"

        print(f"{symbol:<12}{tick_str:<26}{tick_age_str:<12}{bar_str:<26}{bar_age_str:<10}")

    print("-" * 100)
    print(
        "\nInterpretation:\n"
        "  Tick age should be seconds, not minutes, during market hours.\n"
        "  Bar age 1.0-2.0h near an hour boundary is NORMAL (last closed bar).\n"
        "  If tick age is fresh (seconds) but bar age is consistently >2.5h:\n"
        "    -> bug is in MT5/broker bar finalization, fixable by building bars\n"
        "       from live ticks locally instead of waiting on copy_rates_*.\n"
        "  If tick age itself is stale (minutes) during active market hours:\n"
        "    -> the broker's feed itself is lagging the real market; no local\n"
        "       fix helps, need a different data source or broker.\n"
    )

    mt5.shutdown()


if __name__ == "__main__":
    main()
