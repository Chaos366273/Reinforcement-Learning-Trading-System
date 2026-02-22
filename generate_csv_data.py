"""
Generate CSV Data - One-time script to create immutable CSV dataset

This script:
1. Downloads raw OHLCV data
2. Calculates indicators per ticker
3. Validates data quality
4. Saves to immutable CSV
5. Creates manifest with hash

Run this ONCE to create your frozen dataset.

Ticker source (in order of precedence):
- --tickers-from-csv PATH: use unique ticker/Company column from that CSV (all companies).
- config.DATA_CONFIG['tickers']: single source of truth when expanding universe.
- Fallback: 11-ticker list (README default).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ppo.data.csv_data_manager import CSVDataManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Fallback if config not used (same as README default)
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD', 'KO', 'JNJ', 'AMZN', 'XLE', 'TLT']


def get_tickers_from_config():
    """Single source of truth: config.py DATA_CONFIG['tickers']."""
    try:
        import config as cfg
        tickers = cfg.DATA_CONFIG.get('tickers', DEFAULT_TICKERS)
        return list(tickers) if tickers else DEFAULT_TICKERS
    except Exception:
        return DEFAULT_TICKERS


def get_tickers_from_csv(csv_path: str):
    """Load unique tickers from a CSV (column 'Company' or 'Ticker' or 'tic')."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker CSV not found: {path}")
    import pandas as pd
    # Peek columns then read only the ticker column for full unique list
    peek = pd.read_csv(path, nrows=1)
    for col in ('Company', 'Ticker', 'tic', 'Symbol', 'symbol'):
        if col in peek.columns:
            df = pd.read_csv(path, usecols=[col])
            tickers = df[col].dropna().astype(str).str.strip().str.upper().unique().tolist()
            skip = {'BTC-USD', 'ETH-USD'}
            tickers = [t for t in tickers if t and t not in skip]
            if tickers:
                logger.info("Loaded %s tickers from CSV column '%s': %s", len(tickers), col, csv_path)
                return sorted(tickers)
    raise ValueError(f"No ticker column (Company/Ticker/tic) in {path}")


def main():
    """Generate CSV data for PPO trading agent."""
    parser = argparse.ArgumentParser(description="Generate immutable CSV for PPO (all tickers from config or from a CSV).")
    parser.add_argument(
        "--tickers-from-csv",
        type=str,
        default=None,
        help="Use all unique tickers from this CSV (column Company or Ticker). E.g. ppo/saved_data/csv_data/stock_details_5_years.csv"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CSV DATA GENERATION - PPO Trading Agent")
    logger.info("=" * 80)

    if args.tickers_from_csv:
        tickers = get_tickers_from_csv(args.tickers_from_csv)
    else:
        tickers = get_tickers_from_config()

    try:
        import config as cfg
        start_date = cfg.DATA_CONFIG.get('start_date', '2020-01-01')
        end_date = cfg.DATA_CONFIG.get('end_date', '2024-10-17')
    except Exception:
        start_date = '2020-01-01'
        end_date = '2024-10-17'

    logger.info(f"Tickers: {len(tickers)} symbols (config or --tickers-from-csv)")
    logger.info(f"First 15: {tickers[:15]}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Create CSV manager
    manager = CSVDataManager()
    
    # Check if CSV already exists
    if manager.processed_csv_path.exists():
        logger.warning(f"⚠️ Processed CSV already exists: {manager.processed_csv_path}")
        response = input("Delete and regenerate? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted. Existing CSV preserved.")
            return
        # Delete existing files
        manager.processed_csv_path.unlink()
        if manager.raw_csv_path.exists():
            manager.raw_csv_path.unlink()
        if manager.manifest_path.exists():
            manager.manifest_path.unlink()
        logger.info("Deleted existing CSV files")
    
    try:
        # Process data (download + calculate + validate + save)
        df = manager.process_and_save(tickers, start_date, end_date, force=True)
        
        logger.info("=" * 80)
        logger.info("✅ CSV GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Processed CSV: {manager.processed_csv_path}")
        logger.info(f"Raw CSV: {manager.raw_csv_path}")
        logger.info(f"Manifest: {manager.manifest_path}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total tickers: {len(df['Ticker'].unique())}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"CSV Hash: {manager.file_hash(manager.processed_csv_path)[:16]}...")
        logger.info("=" * 80)
        logger.info("✅ Your data is now frozen and ready for PPO training!")
        logger.info("   The CSV is immutable - PPO will load from this file only.")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ CSV generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
