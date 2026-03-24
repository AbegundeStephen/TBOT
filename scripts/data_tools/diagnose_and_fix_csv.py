#!/usr/bin/env python3
"""
CSV Diagnostic and Repair Tool
First diagnoses the CSV structure, then fixes it
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
import sys

# ✅ FIX: Add project root to sys.path (relative to this script)
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

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

    # Load config to get enabled assets and their symbols
    config_path = project_root / "config" / "config.json"
    enabled_assets = []
    active_symbols = []
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                for a, cfg in config.get("assets", {}).items():
                    if cfg.get("enabled"):
                        enabled_assets.append(a.lower())
                        if cfg.get("symbol"):
                            active_symbols.append(cfg.get("symbol"))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    # Search in both data/ and data/raw/
    search_dirs = [project_root / "data", project_root / "data" / "raw"]
    files_to_process = []
    
    for d in search_dirs:
        if not d.exists():
            continue
            
        # Scan all CSVs in the directory
        for csv_file in d.glob("*.csv"):
            filename = csv_file.name.upper()
            
            # Match if filename contains the asset key (BTC) or the symbol (BTCUSDT)
            is_match = False
            for asset in enabled_assets:
                if asset.upper() in filename:
                    is_match = True
                    break
            
            for symbol in active_symbols:
                if symbol.upper() in filename:
                    is_match = True
                    break
            
            if is_match and ".backup" not in filename:
                files_to_process.append(csv_file)

    if not files_to_process:
        logger.error("❌ No CSV files found to process in data/ or data/raw/.")
        return

    logger.info(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        logger.info(f"  • {f}")

    # PHASE 1: Diagnose first file
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DIAGNOSIS")
    logger.info("=" * 70)

    first_file = files_to_process[0]
    df_sample = diagnose_csv(str(first_file))

    if df_sample is not None:
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSIS COMPLETE")
        logger.info("=" * 70)
        logger.info("Proceeding with multi-file repair in 3 seconds...")
        time.sleep(3)

    # PHASE 2: Repair all files
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: REPAIR")
    logger.info("=" * 70)

    results = {}

    for filepath in files_to_process:
        filename = filepath.name
        
        # Create backup
        backup_path = filepath.with_suffix('.csv.backup')
        import shutil
        shutil.copy(filepath, backup_path)
        logger.info(f"\n📦 Backup created: {backup_path}")

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
    logger.info(f"Final Result: {success_count}/{total} files repaired.")
    logger.info("=" * 70)

    if success_count == total:
        logger.info("\n🎉 All Multi-Asset files processed successfully!")
    elif success_count > 0:
        logger.info(f"\n⚠ Partial success: {success_count}/{total} files processed.")
    else:
        logger.error("\n❌ All processing failed!")
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
    main()
