# src/data/historical_updater.py

import os
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class HistoricalDataUpdater:
    """
    Automatically updates CSV files with latest candle data.
    Prevents duplicates and maintains data continuity.
    """

    def __init__(self, data_manager, config):
        self.data_manager = data_manager
        self.config = config
        self.historical_dir = Path("data/raw")
        self.historical_dir.mkdir(parents=True, exist_ok=True)

    def update_asset_history(self, asset_name):
        """
        Fetches latest data and appends to CSV if new candles exist.
        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            symbol = asset_cfg["symbol"]
            exchange = asset_cfg.get("exchange", "binance")
            
            # Determine filename
            filename_map = {
                "BTC": "BTCUSDT_1h.csv",
                "GOLD": "XAUUSDm_1h.csv",
                "XAU": "XAUUSDm_1h.csv",
            }
            csv_filename = filename_map.get(asset_name.upper(), f"{asset_name}_1h.csv")
            csv_path = self.historical_dir / csv_filename

            # Check if file exists and get last timestamp
            if csv_path.exists():
                existing_df = pd.read_csv(csv_path)
                
                # Find timestamp column
                timestamp_col = None
                for col in ['date', 'timestamp', 'time', 'datetime']:
                    if col in existing_df.columns:
                        timestamp_col = col
                        break
                
                if not timestamp_col:
                    logger.warning(f"[UPDATE] No timestamp column in {csv_filename}, skipping")
                    return
                
                existing_df[timestamp_col] = pd.to_datetime(
                existing_df[timestamp_col],
                utc=True,
                errors="coerce"
            )

                last_date = existing_df[timestamp_col].max()
                
                logger.info(f"[UPDATE] {asset_name} CSV last date: {last_date}")
                
                # Fetch data from last_date to now
                end_time = datetime.now(timezone.utc)

                if pd.isna(last_date):
                    logger.warning(f"[UPDATE] {asset_name} CSV has no valid dates — full backfill")
                    start_time = None
                else:
                    start_time = last_date + timedelta(hours=1)

                
            else:
                # File doesn't exist, fetch last 30 days
                logger.info(f"[UPDATE] Creating new CSV for {asset_name}")
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=30)
                existing_df = None

            # Fetch new data
            if exchange == "binance":
                new_df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=asset_cfg.get("interval", "1h"),
                    start_date=start_time.strftime("%Y-%m-%d") if start_time else None,
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                new_df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=asset_cfg.get("timeframe", "H1"),
                    start_date=start_time.strftime("%Y-%m-%d") if start_time else None,
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            if new_df is None or len(new_df) == 0:
                logger.debug(f"[UPDATE] No new data for {asset_name}")
                return

            # Ensure consistent column names
            # Normalize datetime into `date`
            if isinstance(new_df.index, pd.DatetimeIndex):
                new_df = new_df.reset_index()
                new_df.rename(columns={new_df.columns[0]: 'date'}, inplace=True)
                new_df['date'] = pd.to_datetime(new_df['date'], utc=True, errors='coerce')
            elif 'date' in new_df.columns:
                new_df['date'] = pd.to_datetime(new_df['date'], utc=True, errors='coerce')
            elif 'timestamp' in new_df.columns:
                new_df['date'] = pd.to_datetime(new_df['timestamp'], utc=True, errors='coerce')
            elif 'time' in new_df.columns:
                new_df['date'] = pd.to_datetime(new_df['time'], utc=True, errors='coerce')
            else:
                raise ValueError(
                    f"No time information found in new data: index={type(new_df.index)}, cols={new_df.columns.tolist()}"
                )


            new_df = new_df.dropna(subset=['date'])



            if existing_df is not None:
                # Merge and remove duplicates
                existing_df.rename(columns={timestamp_col: 'date'}, inplace=True)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates based on date
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                
                new_rows = len(combined_df) - len(existing_df)
                logger.info(f"[UPDATE] {asset_name}: Added {new_rows} new candles")
            else:
                combined_df = new_df
                logger.info(f"[UPDATE] {asset_name}: Created with {len(combined_df)} candles")

            # Save to CSV
            # Keep only necessary columns
            columns_to_save = ['date', 'open', 'high', 'low', 'close', 'volume']
            columns_to_save = [col for col in columns_to_save if col in combined_df.columns]
            
            combined_df[columns_to_save].to_csv(csv_path, index=False)
            
            logger.info(f"[UPDATE] ✅ {asset_name} saved to {csv_path}")
            logger.info(f"[UPDATE] Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

        except Exception as e:
            logger.error(f"[UPDATE] Failed to update {asset_name}: {e}", exc_info=True)

    def update_all_enabled_assets(self):
        """
        Updates historical data for all enabled assets.
        """
        enabled = [
            name
            for name, cfg in self.config["assets"].items()
            if cfg.get("enabled", False)
        ]

        logger.info(f"[UPDATE] Updating historical data for: {', '.join(enabled)}")
        
        for asset_name in enabled:
            self.update_asset_history(asset_name)

    def verify_data_integrity(self, asset_name):
        """
        Checks for gaps in the data and reports issues.
        """
        filename_map = {
            "BTC": "BTCUSDT_1h.csv",
            "GOLD": "XAUUSDm_1h.csv",
            "XAU": "XAUUSDm_1h.csv",
        }
        csv_filename = filename_map.get(asset_name.upper(), f"{asset_name}_1h.csv")
        csv_path = self.historical_dir / csv_filename

        if not csv_path.exists():
            logger.warning(f"[VERIFY] {csv_filename} does not exist")
            return

        try:
            df = pd.read_csv(csv_path)
            
            # Find timestamp column
            timestamp_col = None
            for col in ['date', 'timestamp', 'time', 'datetime']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if not timestamp_col:
                logger.error(f"[VERIFY] No timestamp column in {csv_filename}")
                return

            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)

            # Check for gaps (assuming 1h candles)
            df['time_diff'] = df[timestamp_col].diff()
            gaps = df[df['time_diff'] > pd.Timedelta(hours=2)]  # Allow 2h tolerance

            if len(gaps) > 0:
                logger.warning(f"[VERIFY] {asset_name}: Found {len(gaps)} gaps in data")
                for idx, row in gaps.iterrows():
                    logger.warning(f"  Gap at {row[timestamp_col]}: {row['time_diff']}")
            else:
                logger.info(f"[VERIFY] ✅ {asset_name}: No gaps detected")

            # Report summary
            logger.info(f"[VERIFY] {asset_name}:")
            logger.info(f"  Total candles: {len(df)}")
            logger.info(f"  Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
            logger.info(f"  Latest candle: {df[timestamp_col].max()}")

        except Exception as e:
            logger.error(f"[VERIFY] Failed to verify {asset_name}: {e}")