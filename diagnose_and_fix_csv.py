#!/usr/bin/env python3
"""
CSV Diagnostic and Repair Tool
First diagnoses the CSV structure, then fixes it
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def diagnose_csv(filepath: str):
    """Show detailed CSV structure"""
    logger.info(f"\n{'='*70}")
    logger.info(f"DIAGNOSING: {filepath}")
    logger.info("=" * 70)

    try:
        df = pd.read_csv(filepath, nrows=10)

        logger.info(f"\n📋 Columns: {list(df.columns)}")
        logger.info(f"📊 Shape: {df.shape}")
        logger.info(f"\n🔍 First 3 rows:")
        print(df.head(3))

        logger.info(f"\n🔍 Column Data Types:")
        print(df.dtypes)

        logger.info(f"\n🔍 NaN Count per Column:")
        full_df = pd.read_csv(filepath)
        print(full_df.isnull().sum())

        logger.info(f"\n📈 Total Rows: {len(full_df)}")

        return full_df

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return None


def fix_csv_smart(input_file: str, output_file: str = None) -> bool:
    """
    Smart CSV fixer that adapts to the actual data format
    """
    if output_file is None:
        output_file = input_file

    logger.info(f"\n{'='*70}")
    logger.info(f"FIXING: {input_file}")
    logger.info("=" * 70)

    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        logger.info(f"✓ Loaded {initial_rows} rows")

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        logger.info(f"✓ Columns: {list(df.columns)}")

        # Check for required OHLC columns
        required = ["open", "high", "low", "close"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            logger.error(f"❌ Missing required columns: {missing}")
            return False

        # Step 1: Handle timestamp/date - BE FLEXIBLE
        has_date = False

        if "timestamp" in df.columns:
            logger.info("Found 'timestamp' column")
            # Try to clean it - remove completely invalid rows only
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            valid_ts = df["timestamp"].notna()
            logger.info(f"  Valid timestamps: {valid_ts.sum()} / {len(df)}")

            if valid_ts.sum() > 0:
                df = df[valid_ts].copy()
                df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
                has_date = True

        if "date" in df.columns and not has_date:
            logger.info("Found 'date' column")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            valid_date = df["date"].notna()
            logger.info(f"  Valid dates: {valid_date.sum()} / {len(df)}")

            if valid_date.sum() > 0:
                df = df[valid_date].copy()
                has_date = True

        if not has_date:
            logger.warning(
                "⚠ No valid timestamp/date column - will create index-based date"
            )
            df["date"] = pd.date_range(
                start="2020-01-01", periods=len(df), freq="15min"
            )

        # Step 2: Clean OHLC data
        logger.info("\n🧹 Cleaning OHLC data...")

        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows where ANY OHLC is NaN
        before = len(df)
        df = df.dropna(subset=required)
        after = len(df)
        logger.info(f"  Removed {before - after} rows with NaN in OHLC")

        # Validate OHLC relationships
        valid_ohlc = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["open"] > 0)
            & (df["close"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
        )

        invalid_count = (~valid_ohlc).sum()
        if invalid_count > 0:
            logger.warning(
                f"  Removing {invalid_count} rows with invalid OHLC relationships"
            )
            df = df[valid_ohlc].copy()

        # Step 3: Sort by date
        if has_date:
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"✓ Sorted by date")

        # Step 4: Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=required).reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.info(f"  Removed {before - after} duplicate rows")

        # Step 5: Validate final dataset
        if len(df) < 100:
            logger.error(
                f"❌ Insufficient data after cleaning: {len(df)} rows (need ≥100)"
            )
            return False

        # Step 6: Save cleaned data
        # Keep all original columns plus date
        df.to_csv(output_file, index=False)

        logger.info(f"\n✅ SUCCESS!")
        logger.info(f"  Input rows: {initial_rows}")
        logger.info(f"  Output rows: {len(df)}")
        logger.info(
            f"  Removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)"
        )
        logger.info(f"  Saved to: {output_file}")

        if has_date:
            logger.info(f"\n📅 Date Range:")
            logger.info(f"  Start: {df['date'].iloc[0]}")
            logger.info(f"  End: {df['date'].iloc[-1]}")

        logger.info(f"\n💰 Price Statistics:")
        logger.info(f"  Min: ${df['close'].min():.2f}")
        logger.info(f"  Max: ${df['close'].max():.2f}")
        logger.info(f"  Latest: ${df['close'].iloc[-1]:.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return False


def main():
    """Main diagnostic and repair pipeline"""

    logger.info("=" * 70)
    logger.info("MULTI-ASSET CSV DIAGNOSTIC & REPAIR TOOL")
    logger.info("=" * 70)

    # Load config to get enabled assets
    config_path = Path("config/config.json")
    enabled_assets = []
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                enabled_assets = [a.lower() for a, cfg in config.get("assets", {}).items() if cfg.get("enabled")]
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    data_dir = Path("data")
    
    # Dynamically find files for enabled assets
    # Pattern: train_data_[asset]_[timeframe].csv
    files = []
    if enabled_assets:
        for asset in enabled_assets:
            # Check for common timeframes used in training
            for tf in ["15m", "4h", "1h"]:
                filename = f"train_data_{asset}_{tf}.csv"
                if (data_dir / filename).exists():
                    files.append(filename)
    
    # Fallback to general scan if no config-based files found
    if not files:
        logger.info("No specific asset files found via config, scanning data directory...")
        files = [f.name for f in data_dir.glob("train_data_*.csv")]

    if not files:
        logger.error("❌ No CSV files found to process in data/ directory.")
        return

    logger.info(f"Found {len(files)} files to process: {files}")

    # PHASE 1: Diagnose first file
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DIAGNOSIS")
    logger.info("=" * 70)

    first_file = data_dir / files[0]
    df_sample = diagnose_csv(str(first_file))

    if df_sample is not None:
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSIS COMPLETE - See data structure above")
        logger.info("=" * 70)
        
        # Non-interactive skip for automation
        logger.info("Proceeding with repair in 3 seconds...")
        time.sleep(3)

    # PHASE 2: Repair all files
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: REPAIR")
    logger.info("=" * 70)

    results = {}

    for filename in files:
        filepath = data_dir / filename

        # Create backup
        backup_path = data_dir / f"{filename}.backup"
        import shutil

        shutil.copy(filepath, backup_path)
        logger.info(f"\n📦 Backup created for: {filename}")

        # Fix the file
        success = fix_csv_smart(str(filepath))
        results[filename] = success

    # PHASE 3: Summary
    logger.info("\n" + "=" * 70)
    logger.info("REPAIR SUMMARY")
    logger.info("=" * 70)

    for filename, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {filename}")

    success_count = sum(results.values())
    total = len(results)

    logger.info("\n" + "=" * 70)
    logger.info(f"Success Rate: {success_count}/{total}")
    logger.info("=" * 70)

    if success_count == total:
        logger.info("\n🎉 All files processed successfully!")
    elif success_count > 0:
        logger.info(f"\n⚠ Partial success: {success_count}/{total} files processed")
    else:
        logger.error("\n❌ All processing failed!")


if __name__ == "__main__":
    import json
    main()


if __name__ == "__main__":
    main()
