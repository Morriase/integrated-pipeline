"""
Complete SMC Data Pipeline Workflow
====================================

This script runs the complete end-to-end pipeline:
1. Consolidate MT5 exports → consolidated_ohlc_data.csv
2. Process SMC features → processed_smc_data.csv + train/val/test splits

Usage:
    python run_complete_pipeline.py

Configurations available:
- Full pipeline (all symbols, all timeframes)
- Quick test (single symbol: EURUSD, timeframes: M15, H1, H4)
- Custom (specify your own symbols and timeframes)
"""

from consolidate_mt5_data import consolidate_mt5_exports
from data_preparation_pipeline import SMCDataPipeline
from pathlib import Path
import time


def run_full_pipeline(config='full'):
    """
    Run complete SMC data preparation pipeline

    Args:
        config: 'full', 'test', or 'custom'
    """
    print("\n" + "=" * 80)
    print("COMPLETE SMC DATA PIPELINE WORKFLOW")
    print("=" * 80)

    start_time = time.time()

    # ========================================================================
    # STEP 1: CONSOLIDATE MT5 EXPORTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: CONSOLIDATING MT5 EXPORT FILES")
    print("=" * 80)

    if config == 'full':
        # Full pipeline: All symbols, all timeframes
        print("\n📊 Configuration: FULL (All symbols, all timeframes)")
        consolidated_df = consolidate_mt5_exports(
            input_dir='Docs/Data/mt5_exports',
            output_file='Docs/Data/consolidated_ohlc_data.csv',
            symbols=None,  # All symbols
            timeframes=None  # All timeframes
        )

        input_file = 'Docs/Data/consolidated_ohlc_data.csv'
        output_file = 'Docs/Data/processed_smc_data.csv'

    elif config == 'test':
        # Quick test: Single symbol, 3 timeframes
        print("\n📊 Configuration: TEST (EURUSD only, M15/H1/H4)")
        consolidated_df = consolidate_mt5_exports(
            input_dir='Docs/Data/mt5_exports',
            output_file='Docs/Data/consolidated_eurusd_test.csv',
            symbols=['EURUSD'],
            timeframes=['M15', 'H1', 'H4']
        )

        input_file = 'Docs/Data/consolidated_eurusd_test.csv'
        output_file = 'Docs/Data/processed_eurusd_test.csv'

    elif config == 'custom':
        # Custom: Specify your own symbols and timeframes
        print("\n📊 Configuration: CUSTOM")

        # CUSTOMIZE THESE:
        custom_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        custom_timeframes = ['M15', 'H1', 'H4']

        print(f"  Symbols: {custom_symbols}")
        print(f"  Timeframes: {custom_timeframes}")

        consolidated_df = consolidate_mt5_exports(
            input_dir='Docs/Data/mt5_exports',
            output_file='Docs/Data/consolidated_custom.csv',
            symbols=custom_symbols,
            timeframes=custom_timeframes
        )

        input_file = 'Docs/Data/consolidated_custom.csv'
        output_file = 'Docs/Data/processed_custom.csv'

    else:
        raise ValueError(
            f"Unknown config: {config}. Use 'full', 'test', or 'custom'")

    step1_time = time.time() - start_time
    print(f"\n⏱️ Step 1 completed in {step1_time:.2f} seconds")

    # ========================================================================
    # STEP 2: PROCESS SMC FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PROCESSING SMC FEATURES WITH FUZZY LOGIC")
    print("=" * 80)

    # Initialize pipeline with parameters optimized for MORE LABELED DATA
    pipeline = SMCDataPipeline(
        base_timeframe='M15',           # Primary trading timeframe
        higher_timeframes=['H1', 'H4'],  # Higher TFs for confluence
        atr_period=14,                  # ATR calculation period
        rr_ratio=2.0,                   # 1:2 R:R (easier to hit, more labels)
        lookforward=20,                 # 20 candles (faster outcomes, more labels)
        fuzzy_quality_threshold=0.2     # Lower threshold (accept more setups, more labels)
    )

    # Run pipeline
    processed_df = pipeline.run_pipeline(
        input_path=input_file,
        output_path=output_file,
        save_splits=True  # Create train/val/test splits
    )

    step2_time = time.time() - start_time - step1_time
    total_time = time.time() - start_time

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\n⏱️ Execution Time:")
    print(f"  Step 1 (Consolidation): {step1_time:.2f}s")
    print(f"  Step 2 (SMC Processing): {step2_time:.2f}s")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    print(f"\n📁 Output Files:")
    output_dir = Path(output_file).parent
    base_name = Path(output_file).stem

    print(f"\n  Consolidated Data:")
    print(f"    {input_file}")

    print(f"\n  Processed Data:")
    print(f"    {output_file}")
    print(f"    {output_dir / f'{base_name}_train.csv'}")
    print(f"    {output_dir / f'{base_name}_val.csv'}")
    print(f"    {output_dir / f'{base_name}_test.csv'}")
    print(f"    {output_dir / f'{base_name}_feature_list.txt'}")
    print(f"    {output_dir / f'{base_name}_quality_stats.txt'}")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Review the summary report above")
    print(f"  2. Check feature_list.txt for ML features")
    print(f"  3. Load train/val/test splits for model training")
    print(f"  4. Train your ML model (Random Forest, XGBoost, Neural Network)")
    print(f"  5. Predict TBM_Label (-1=Loss, 0=Timeout, 1=Win)")

    print("\n✅ All done! Your data is ready for ML training.")

    return processed_df


if __name__ == "__main__":
    """
    Run the complete pipeline

    Choose your configuration:
    - 'full': All symbols, all timeframes (slowest, most complete)
    - 'test': EURUSD only, M15/H1/H4 (fastest, for testing)
    - 'custom': Specify your own symbols/timeframes in the code above
    """

    # ========================================================================
    # CHOOSE YOUR CONFIGURATION HERE:
    # ========================================================================

    # Option 1: Quick test (recommended for first run)
    # processed_data = run_full_pipeline(config='test')

    # Option 2: Full pipeline (all data - takes longer)
    processed_data = run_full_pipeline(config='full')

    # Option 3: Custom configuration (edit symbols/timeframes in function above)
    # processed_data = run_full_pipeline(config='custom')
