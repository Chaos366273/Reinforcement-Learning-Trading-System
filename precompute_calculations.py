"""
Pre-Compute Runtime Calculations - Speed Up Training

This script pre-computes all runtime calculations and caches them,
significantly speeding up PPO training by eliminating redundant calculations.

Run this ONCE before training to create the calculation cache.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ppo.data.csv_data_manager import CSVDataManager
from ppo.data.csv_loader_adapter import CSVLoaderAdapter
from ppo.data.precompute_runtime_calculations import RuntimeCalculationsCache
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Pre-compute all runtime calculations from CSV data."""
    logger.info("=" * 80)
    logger.info("PRE-COMPUTING RUNTIME CALCULATIONS")
    logger.info("=" * 80)
    
    # Step 1: Load CSV data
    logger.info("Step 1: Loading CSV data...")
    csv_manager = CSVDataManager()
    
    if not csv_manager.processed_csv_path.exists():
        logger.error(f"❌ Processed CSV not found: {csv_manager.processed_csv_path}")
        logger.error("   Run generate_csv_data.py first to create CSV")
        return
    
    df = csv_manager.load_processed_data()
    logger.info(f"✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    
    # Step 2: Convert to matrices
    logger.info("Step 2: Converting to matrices...")
    adapter = CSVLoaderAdapter(csv_manager)
    
    tickers = sorted(df['Ticker'].unique())
    feature_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                   'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
    
    data_dict = adapter.load_and_convert_to_matrices(tickers, feature_cols, train_split=1.0)
    train_data = data_dict['train_data']
    
    logger.info(f"✅ Converted to matrices:")
    logger.info(f"   Price matrix: {train_data['price_matrix'].shape}")
    logger.info(f"   Feature matrix: {train_data['feature_matrix'].shape}")
    logger.info(f"   Volume matrix: {train_data['volume_matrix'].shape}")
    
    # Step 3: Pre-compute all calculations
    logger.info("Step 3: Pre-computing runtime calculations...")
    calculator = RuntimeCalculationsCache()
    
    cache = calculator.precompute_all(
        price_matrix=train_data['price_matrix'],
        volume_matrix=train_data['volume_matrix'],
        feature_matrix=train_data['feature_matrix']
    )
    
    # Step 4: Save cache
    logger.info("Step 4: Saving calculation cache...")
    cache_file = calculator.save_cache(cache)
    
    logger.info("=" * 80)
    logger.info("✅ PRE-COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Cache file: {cache_file}")
    logger.info(f"Cache directory: {calculator.cache_dir}")
    logger.info("")
    logger.info("Pre-computed calculations:")
    logger.info(f"  - VWAP: {cache['vwap'].shape}")
    logger.info(f"  - Returns: {cache['returns'].shape}")
    logger.info(f"  - Volatility: {cache['volatility'].shape}")
    logger.info(f"  - ATR: {cache['atr'].shape}")
    logger.info(f"  - Market Regime: {cache['market_regime'].shape}")
    logger.info(f"  - Volume Ratios: {cache['volume_ratios'].shape}")
    logger.info(f"  - Technical Signals: {len(cache['technical_signals'])} signal types")
    logger.info("")
    logger.info("✅ Your training will now be faster!")
    logger.info("   The environment will load these pre-computed values instead of calculating them.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
