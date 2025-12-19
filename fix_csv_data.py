#!/usr/bin/env python3
"""
CSV Data Cleaner
Fixes NaN issues and validates OHLC data before training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_and_validate_csv(input_file: str, output_file: str = None) -> bool:
    """
    Clean CSV file by:
    1. Removing rows with NaN values
    2. Validating OHLC relationships
    3. Ensuring proper timestamp format
    4. Sorting chronologically
    
    Args:
        input_file: Path to input CSV
        output_file: Path to save cleaned CSV (defaults to overwriting input)
    
    Returns:
        bool: True if successful
    """
    if output_file is None:
        output_file = input_file
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Cleaning: {input_file}")
    logger.info('='*70)
    
    try:
        # Load CSV
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        logger.info(f"✓ Loaded {initial_rows} rows")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.error(f"❌ Missing columns: {missing}")
            logger.info(f"Available columns: {list(df.columns)}")
            return False
        
        # Handle timestamp/date
        if 'timestamp' in df.columns:
            # Remove NaN timestamps first
            df = df.dropna(subset=['timestamp'])
            # Convert to datetime
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        elif 'date' in df.columns:
            df = df.dropna(subset=['date'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with NaN in date
        if 'date' in df.columns:
            df = df.dropna(subset=['date'])
        
        # Remove NaN in OHLC columns
        df = df.dropna(subset=required)
        logger.info(f"✓ After removing NaN: {len(df)} rows ({initial_rows - len(df)} removed)")
        
        # Convert OHLC to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN after conversion
        df = df.dropna(subset=required)
        
        # Validate OHLC relationships
        valid_mask = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['open'] > 0) &
            (df['close'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0)
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"⚠ Removing {invalid_count} rows with invalid OHLC")
            df = df[valid_mask]
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            logger.info(f"✓ Sorted chronologically")
            logger.info(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        
        # Remove duplicate rows
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Final validation
        if len(df) < 100:
            logger.error(f"❌ Insufficient data after cleaning: {len(df)} rows")
            return False
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved {len(df)} clean rows to {output_file}")
        
        # Show statistics
        logger.info(f"\nData Statistics:")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cleaning {input_file}: {e}", exc_info=True)
        return False


def main():
    """Clean all training data files"""
    
    logger.info("="*70)
    logger.info("CSV DATA CLEANER")
    logger.info("="*70)
    
    data_dir = Path("data")
    
    # Files to clean
    files_to_clean = [
        "train_data_btc_15m.csv",
        "train_data_btc_4h.csv",
        "train_data_gold_15m.csv",
        "train_data_gold_4h.csv"
    ]
    
    results = {}
    
    for filename in files_to_clean:
        filepath = data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"⚠ File not found: {filepath}")
            results[filename] = False
            continue
        
        # Create backup
        backup_path = data_dir / f"{filename}.backup"
        import shutil
        shutil.copy(filepath, backup_path)
        logger.info(f"📦 Backup created: {backup_path}")
        
        # Clean the file
        success = clean_and_validate_csv(str(filepath))
        results[filename] = success
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("CLEANING SUMMARY")
    logger.info("="*70)
    
    for filename, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {filename}")
    
    success_count = sum(results.values())
    total = len(results)
    
    logger.info("\n" + "="*70)
    logger.info(f"Success Rate: {success_count}/{total}")
    logger.info("="*70)
    
    if success_count == total:
        logger.info("\n✅ All files cleaned successfully!")
        logger.info("You can now run: python train_ai_layer.py")
    else:
        logger.warning("\n⚠ Some files failed to clean")
        logger.info("Check the logs above for details")


if __name__ == "__main__":
    main()