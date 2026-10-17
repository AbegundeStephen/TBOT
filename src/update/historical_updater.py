# src/data/historical_updater.py

import os
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HistoricalDataUpdater:
    """
    Automatically updates CSV files with latest candle data across multiple timeframes.
    Supports 1H (Sniper), 4H (Analyst), and 1D (Governor) data.
    Prevents duplicates and maintains data continuity.
    """

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        # ✅ HARMONIZED: Use data/raw/ directory for all historical data
        self.historical_dir = Path("data/raw")
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # Define timeframe configurations
        # Lookback windows are sized so every indicator always has enough history:
        #   1H  → 365 days = 8,760 bars  (200 EMA needs 200 bars; strategies use 100-bar lookback)
        #   4H  → 730 days = 4,380 bars  (200 EMA on 4H needs 200×4h = 33 days; 730 gives 10× margin)
        #   1D  → 1095 days = 3 years    (200-day EMA needs 200 bars; governor/macro filter needs full history)
        self.timeframes = {
            '1h': {
                'binance_interval': '1h',
                'mt5_timeframe': 'H1',
                'suffix': '1h',
                'lookback_days': 365,   # 1 year — was 90 (too short for regime context)
                'min_candles': 500,
            },
            '4h': {
                'binance_interval': '4h',
                'mt5_timeframe': 'H4',
                'suffix': '4h',
                'lookback_days': 730,   # 2 years — unchanged
                'min_candles': 500,
            },
            '1d': {
                'binance_interval': '1d',
                'mt5_timeframe': 'D1',
                'suffix': '1d',
                'lookback_days': 1095,  # 3 years — was 730 (needed 200-day EMA with headroom)
                'min_candles': 250,
            },
        }
        
        logger.info("[MTF UPDATER] Initialized for 1H + 4H + 1D timeframes")

    def _get_filename(self, asset_name: str, timeframe: str) -> str:
        """
        Generate filename for asset and timeframe
        
        Args:
            asset_name: "BTC", "GOLD", etc.
            timeframe: "1h", "4h", "1d"
        
        Returns:
            CSV filename
        """
        filename_map = {
            "BTC": f"BTCUSDT_{timeframe}.csv",
            "GOLD": f"XAUUSDm_{timeframe}.csv",
            "XAU": f"XAUUSDm_{timeframe}.csv",
        }
        return filename_map.get(asset_name.upper(), f"{asset_name}_{timeframe}.csv")

    def update_asset_timeframe(
        self,
        asset_name: str,
        timeframe: str,
        force_full_refresh: bool = False,
        lookback_days_override: Optional[int] = None,
    ) -> bool:
        """
        Update historical data for a specific asset and timeframe.

        Args:
            asset_name: Asset name (e.g., "BTC", "GOLD")
            timeframe: Timeframe key ("1h", "4h", "1d")
            force_full_refresh: If True, re-download all data
            lookback_days_override: Dashboard "Data" section wiring -- when set,
                used in place of this timeframe's hardcoded lookback_days for a
                full/new-file refresh (see the branch below). Only meaningful
                together with force_full_refresh=True (or a brand-new file) --
                an incremental update always starts from "last existing date +
                1 period" and never reaches the lookback branch at all, so this
                param has no effect there. None (default) preserves today's
                behavior for every existing non-dashboard caller.

        Returns:
            True if successful
        """
        try:
            # ✅ Check Market Hours
            from src.utils.market_hours import MarketHours
            # BTC/Crypto is 24/7, ignore the institutional weekend gate for data updates.
            # A forced full refresh is a historical backfill (copy_rates_range), not a live
            # feed poll, so it must also bypass the gate — MT5 returns past bars regardless
            # of whether the market is open right now.
            if "BTC" in asset_name.upper() or force_full_refresh:
                market_open = True
            else:
                market_open = MarketHours.should_trade(asset_name)

            if not market_open:
                logger.debug(f"[UPDATE] Skipping {asset_name} {timeframe.upper()} - Market is CLOSED")
                return True

            # Get timeframe config
            tf_config = self.timeframes.get(timeframe)
            if not tf_config:
                logger.error(f"[UPDATE] Unknown timeframe: {timeframe}")
                return False
            
            # Get asset config
            asset_cfg = self.config["assets"].get(asset_name)
            if not asset_cfg:
                logger.error(f"[UPDATE] Unknown asset: {asset_name}")
                return False
            
            symbol = asset_cfg["symbol"]
            exchange = asset_cfg.get("exchange", "binance")

            # ✅ FIX: When the exchange is MT5, use mt5_symbol if available.
            # Previously the updater always used asset_cfg["symbol"] (e.g. "BTCUSDT"
            # for BTC) even when exchange="mt5", but MT5 has "BTCUSDm". The fetch
            # returned "No MT5 data for BTCUSDT" every cycle and the CSVs were never
            # updated. mt5_symbol is the correct identifier for the MT5 broker feed.
            if exchange == "mt5":
                symbol = asset_cfg.get("mt5_symbol", symbol)

            # Derive CSV filename from the actual trading symbol so it matches
            # the path the MTF regime detector reads from (data/raw/{symbol}_{tf}.csv).
            # Previously _get_filename() had a hardcoded map ("BTC" → "BTCUSDT_*.csv")
            # which diverged from the MTF detector's path ("BTCUSDm_*.csv"), meaning
            # the historical updater and regime detector operated on separate files.
            csv_filename = f"{symbol}_{timeframe}.csv"
            csv_path = self.historical_dir / csv_filename

            logger.info(f"\n{'='*70}")
            logger.info(f"[UPDATE] {asset_name} {timeframe.upper()} Data")
            logger.info(f"{'='*70}")
            logger.info(f"File:     {csv_filename}")
            logger.info(f"Exchange: {exchange}")
            logger.info(f"Symbol:   {symbol}")
            
            # Determine date range
            end_time = datetime.now(timezone.utc)
            existing_df = None
            start_time = None
            
            if csv_path.exists() and not force_full_refresh:
                # Load existing data
                try:
                    existing_df = pd.read_csv(csv_path)
                    
                    # Find timestamp column
                    timestamp_col = self._find_timestamp_column(existing_df)
                    
                    if timestamp_col:
                        existing_df[timestamp_col] = pd.to_datetime(
                            existing_df[timestamp_col], utc=True, errors='coerce'
                        )
                        
                        last_date = existing_df[timestamp_col].max()
                        
                        if pd.notna(last_date):
                            logger.info(f"Last Date: {last_date}")
                            
                            # B12 (Desire 28 Sep, decision 40 A): re-read the last 6 candles every run. MT5 revises
                            # recent candles after they close (around midnight most of all); starting after the last
                            # saved candle meant a wrong one was never looked at again.
                            if timeframe == '1h':
                                _period_b12 = timedelta(hours=1)
                            elif timeframe == '4h':
                                _period_b12 = timedelta(hours=4)
                            else:  # 1d
                                _period_b12 = timedelta(days=1)
                            start_time = last_date - 6 * _period_b12
                        else:
                            logger.warning(f"[UPDATE] No valid dates in existing file")
                            existing_df = None
                    else:
                        logger.warning(f"[UPDATE] No timestamp column found")
                        existing_df = None
                        
                except Exception as e:
                    logger.error(f"[UPDATE] Error reading existing file: {e}")
                    existing_df = None
            
            # If no existing data or forced refresh, use lookback
            if start_time is None:
                lookback_days = lookback_days_override if lookback_days_override is not None else tf_config['lookback_days']
                start_time = end_time - timedelta(days=lookback_days)
                logger.info(f"Status:   {'Full refresh' if force_full_refresh else 'Creating new file'}")
                logger.info(f"Lookback: {lookback_days} days")
            else:
                logger.info(f"Status:   Incremental update")
            
            logger.info(f"Range:    {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            
            # Fetch new data based on exchange
            if exchange == "binance":
                new_df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=tf_config['binance_interval'],
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:  # MT5
                new_df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=tf_config['mt5_timeframe'],
                    start_date=start_time.strftime("%Y-%m-%d %H:%M:%S"),   # B12 (40 A): the exact time
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            
            # Check if new data was fetched
            if new_df is None or len(new_df) == 0:
                logger.info(f"[UPDATE] No new data available")
                return True  # Not an error, just no updates needed
            
            logger.info(f"Fetched:  {len(new_df)} new candles")
            
            # Normalize datetime column to 'date'
            new_df = self._normalize_datetime(new_df)
            # B12 (decision 40 A): never save the candle that is still forming; say when a saved candle was corrected
            try:
                _period_b12 = {'1h': timedelta(hours=1), '4h': timedelta(hours=4)}.get(timeframe, timedelta(days=1))
                new_df = new_df[new_df['date'] + _period_b12 <= end_time]
                if existing_df is not None and len(new_df):
                    _tc_b12 = self._find_timestamp_column(existing_df)
                    if _tc_b12:
                        _old_b12 = existing_df.set_index(pd.to_datetime(existing_df[_tc_b12], utc=True, errors='coerce'))['close']
                        _old_b12 = _old_b12[~_old_b12.index.duplicated(keep='last')].astype(float)
                        _new_b12 = new_df.set_index('date')['close'].astype(float)
                        _both_b12 = _old_b12.index.intersection(_new_b12.index)
                        _rev_b12 = int(((_old_b12.loc[_both_b12] - _new_b12.loc[_both_b12]).abs()
                                        > 1e-4 * _new_b12.loc[_both_b12].abs()).sum())
                        if _rev_b12:
                            logger.warning(f"[UPDATE-REVISED] {asset_name} {timeframe.upper()}: {_rev_b12} saved candle(s) corrected from MT5")
            except Exception as _b12_err:
                logger.warning(f"[UPDATE] {asset_name} {timeframe.upper()}: forming-candle/revision step skipped: {_b12_err}")
            if len(new_df) == 0:
                logger.info("[UPDATE] No closed candles to save yet")
                return True
            
            # Merge with existing data if available
            if existing_df is not None:
                # Normalize existing data
                timestamp_col = self._find_timestamp_column(existing_df)
                if timestamp_col and timestamp_col != 'date':
                    existing_df.rename(columns={timestamp_col: 'date'}, inplace=True)
                
                # Combine and deduplicate
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                
                new_rows = len(combined_df) - len(existing_df)
                logger.info(f"Added:    {new_rows} new candles")
                logger.info(f"Total:    {len(combined_df)} candles")
            else:
                combined_df = new_df
                logger.info(f"Created:  {len(combined_df)} candles")
            
            # Validate minimum candles
            min_candles = tf_config['min_candles']
            if len(combined_df) < min_candles:
                logger.warning(
                    f"[UPDATE] Only {len(combined_df)} candles (need {min_candles}+ for {timeframe})"
                )
            
            # Save to CSV
            columns_to_save = ['date', 'open', 'high', 'low', 'close', 'volume']
            columns_to_save = [col for col in columns_to_save if col in combined_df.columns]
            
            combined_df[columns_to_save].to_csv(csv_path, index=False)
            
            logger.info(f"✅ Saved to: {csv_path}")
            logger.info(f"Date Range: {combined_df['date'].min()} → {combined_df['date'].max()}")
            logger.info(f"Latest Price: ${combined_df['close'].iloc[-1]:,.2f}")
            logger.info(f"{'='*70}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"[UPDATE] Failed to update {asset_name} {timeframe}: {e}", exc_info=True)
            return False

    def update_asset_history(
        self,
        asset_name: str,
        force_full_refresh: bool = False,
        lookback_days_override: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Update all timeframes for a specific asset.

        Args:
            asset_name: Asset name (e.g., "BTC", "GOLD")
            force_full_refresh: If True, re-download all data
            lookback_days_override: Applied uniformly across 1h/4h/1d -- see
                update_asset_timeframe's docstring.

        Returns:
            Dict mapping timeframe to success status
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"# UPDATING {asset_name} - ALL TIMEFRAMES")
        logger.info(f"{'#'*70}")

        results = {}

        for timeframe in ['1h', '4h', '1d']:
            success = self.update_asset_timeframe(
                asset_name=asset_name,
                timeframe=timeframe,
                force_full_refresh=force_full_refresh,
                lookback_days_override=lookback_days_override,
            )
            results[timeframe] = success

        # B14 Part 2 (Desire 16 Oct): the bot's own missing-candle check, right after this market's files are updated.
        # It only reads the 1H and 4H files and logs "[DATA-GAP] ..." (the watchdog alarms on it). It never raises.
        try:
            self._b14_gap_check(asset_name)
        except Exception as _gap_err:
            logger.warning(f"[DATA-GAP] check failed for {asset_name}: {_gap_err}")

        # Summary
        successful = sum(1 for s in results.values() if s)
        logger.info(f"\n[SUMMARY] {asset_name}: {successful}/3 timeframes updated")

        return results

    def _b14_gap_check(self, asset_name: str, now: Optional[datetime] = None) -> int:
        """B14 Part 2 (Desire 16 Oct): the missing-candle check (the rules are in src/update/data_gaps.py). Reads this
        market's 1H and 4H files -- the same paths update_asset_timeframe writes -- and logs one "[DATA-GAP] ..."
        warning per problem it has not reported yet in this run of the bot. Returns how many it logged."""
        from src.update import data_gaps as _dg
        asset_cfg = self.config["assets"].get(asset_name) or {}
        symbol = asset_cfg.get("symbol", asset_name)
        if asset_cfg.get("exchange", "binance") == "mt5":
            symbol = asset_cfg.get("mt5_symbol", symbol)
        group = _dg.group_of(asset_name, asset_cfg)
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        seen = self.__dict__.setdefault("_b14_gap_seen", set())
        if len(seen) > 2000:
            seen.clear()
        logged = 0
        for tf, tf_h in (("1h", 1), ("4h", 4)):
            path = self.historical_dir / f"{symbol}_{tf}.csv"
            try:
                if not path.exists():
                    if (asset_name, tf, "nofile") not in seen:
                        seen.add((asset_name, tf, "nofile"))
                        logged += 1
                        logger.warning(f"[DATA-GAP] {asset_name} {tf.upper()}: price file not found ({path})")
                    continue
                ts = pd.to_datetime(pd.read_csv(path, usecols=[0]).iloc[:, 0], utc=True, errors="coerce").dropna()
                ts = ts.dt.tz_localize(None)
                recent = ts[ts >= pd.Timestamp(now) - pd.Timedelta(days=_dg.WINDOW_DAYS + 2)]
                times = [x.to_pydatetime() for x in (recent if len(recent) else ts.nlargest(1))]
                for g in _dg.find_gaps(times, asset_name, group, tf_h, now):
                    # a hole is known by its LAST missing candle: its first one moves while the 7-day window slides past
                    # it (keyed on that, an old hole was logged again every hour). After a restart a hole still inside
                    # the window is reported once more.
                    key = (asset_name, tf, g["kind"], g["end"] if g["kind"] == "hole" else g["newest"])
                    if key in seen:
                        continue
                    seen.add(key)
                    logged += 1
                    logger.warning(_dg.report_line(asset_name, tf.upper(), g))
            except Exception as e:
                logger.warning(f"[DATA-GAP] check failed for {asset_name} {tf.upper()}: {e}")
        return logged

    def update_all_enabled_assets(
        self,
        force_full_refresh: bool = False,
        lookback_days_override: Optional[int] = None,
        assets: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, bool]]:
        """
        Update historical data for all enabled assets across all timeframes.

        Args:
            force_full_refresh: If True, re-download all data
            lookback_days_override: Applied uniformly across 1h/4h/1d -- see
                update_asset_timeframe's docstring.
            assets: Restrict to this subset of enabled asset names (dashboard
                "single asset" refresh). None = every enabled asset (unchanged
                default).

        Returns:
            Nested dict: {asset: {timeframe: success}}
        """
        enabled = [
            name for name, cfg in self.config["assets"].items()
            if cfg.get("enabled", False)
        ]
        if assets:
            enabled = [a for a in enabled if a in assets]

        logger.info(f"\n{'='*70}")
        logger.info(f"MULTI-TIMEFRAME HISTORICAL DATA UPDATE")
        logger.info(f"{'='*70}")
        logger.info(f"Assets:     {', '.join(enabled)}")
        logger.info(f"Timeframes: 1H, 4H, 1D")
        logger.info(f"Mode:       {'Full Refresh' if force_full_refresh else 'Incremental'}")
        logger.info(f"{'='*70}\n")

        all_results = {}

        for asset_name in enabled:
            results = self.update_asset_history(
                asset_name=asset_name,
                force_full_refresh=force_full_refresh,
                lookback_days_override=lookback_days_override,
            )
            all_results[asset_name] = results
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info(f"UPDATE COMPLETE - FINAL SUMMARY")
        logger.info(f"{'='*70}")
        
        for asset_name, results in all_results.items():
            successful = sum(1 for s in results.values() if s)
            status = "✅" if successful == 3 else "⚠️"
            logger.info(f"{status} {asset_name}: {successful}/3 timeframes")
            
            for tf, success in results.items():
                status_icon = "✅" if success else "❌"
                logger.info(f"   {status_icon} {tf.upper()}")
        
        logger.info(f"{'='*70}\n")
        
        return all_results

    def verify_data_integrity(self, asset_name: str) -> Dict[str, Dict]:
        """
        Check for gaps and issues in all timeframes for an asset.
        
        Args:
            asset_name: Asset name
        
        Returns:
            Dict mapping timeframe to integrity report
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA INTEGRITY CHECK - {asset_name}")
        logger.info(f"{'='*70}\n")
        
        reports = {}
        
        for timeframe in ['1h', '4h', '1d']:
            csv_filename = self._get_filename(asset_name, timeframe)
            csv_path = self.historical_dir / csv_filename
            
            if not csv_path.exists():
                logger.warning(f"[{timeframe.upper()}] File does not exist: {csv_filename}")
                reports[timeframe] = {'exists': False}
                continue
            
            try:
                df = pd.read_csv(csv_path)
                timestamp_col = self._find_timestamp_column(df)
                
                if not timestamp_col:
                    logger.error(f"[{timeframe.upper()}] No timestamp column")
                    reports[timeframe] = {'exists': True, 'error': 'no_timestamp'}
                    continue
                
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values(timestamp_col)
                
                # Check for gaps
                df['time_diff'] = df[timestamp_col].diff()
                
                # Define expected intervals
                expected_intervals = {
                    '1h': pd.Timedelta(hours=1),
                    '4h': pd.Timedelta(hours=4),
                    '1d': pd.Timedelta(days=1),
                }
                
                expected_interval = expected_intervals[timeframe]
                tolerance = expected_interval * 2  # 2x tolerance
                
                gaps = df[df['time_diff'] > tolerance]
                
                # Generate report
                report = {
                    'exists': True,
                    'total_candles': len(df),
                    'date_range': {
                        'start': str(df[timestamp_col].min()),
                        'end': str(df[timestamp_col].max()),
                    },
                    'latest_candle': str(df[timestamp_col].max()),
                    'gaps_found': len(gaps),
                }
                
                reports[timeframe] = report
                
                # Log results
                logger.info(f"[{timeframe.upper()}] {csv_filename}")
                logger.info(f"  Total Candles: {len(df)}")
                logger.info(f"  Date Range:    {df[timestamp_col].min()} → {df[timestamp_col].max()}")
                logger.info(f"  Latest Candle: {df[timestamp_col].max()}")
                
                if len(gaps) > 0:
                    logger.warning(f"  ⚠️  Gaps Found:   {len(gaps)}")
                    for idx, row in gaps.head(5).iterrows():
                        logger.warning(f"     Gap at {row[timestamp_col]}: {row['time_diff']}")
                else:
                    logger.info(f"  ✅ No Gaps")
                
                logger.info("")
                
            except Exception as e:
                logger.error(f"[{timeframe.upper()}] Verification failed: {e}")
                reports[timeframe] = {'exists': True, 'error': str(e)}
        
        return reports

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in dataframe"""
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df.columns:
                return col
        return None

    def _normalize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dataframe to have 'date' column with datetime
        
        Args:
            df: Input dataframe
        
        Returns:
            Normalized dataframe with 'date' column
        """
        # Handle DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            
        # Find and rename timestamp column
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
            
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            
        else:
            raise ValueError(
                f"No time information found in data: "
                f"index={type(df.index)}, cols={df.columns.tolist()}"
            )
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        return df

    def get_file_info(self, asset_name: str, timeframe: str) -> Optional[Dict]:
        """
        Get information about a specific data file.
        
        Args:
            asset_name: Asset name
            timeframe: Timeframe ("1h", "4h", "1d")
        
        Returns:
            Dict with file information or None if file doesn't exist
        """
        csv_filename = self._get_filename(asset_name, timeframe)
        csv_path = self.historical_dir / csv_filename
        
        if not csv_path.exists():
            return None
        
        try:
            df = pd.read_csv(csv_path)
            timestamp_col = self._find_timestamp_column(df)
            
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
                return {
                    'filename': csv_filename,
                    'path': str(csv_path),
                    'size_kb': csv_path.stat().st_size / 1024,
                    'total_candles': len(df),
                    'date_start': str(df[timestamp_col].min()),
                    'date_end': str(df[timestamp_col].max()),
                    'latest_price': float(df['close'].iloc[-1]),
                }
            
        except Exception as e:
            logger.error(f"Error reading file info: {e}")
        
        return None