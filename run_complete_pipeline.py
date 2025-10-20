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
import pandas as pd


def run_full_pipeline(data_dir='Data/mt5_exports', output_dir='Data'):
    """
    Run complete SMC data preparation pipeline on FULL dataset

    This combines ALL symbols and timeframes into one massive training dataset
    for better generalization across market conditions.

    Args:
        data_dir: Directory containing MT5 export files
        output_dir: Directory to save processed data
    """
    print("\n" + "=" * 80)
    print("COMPLETE SMC DATA PIPELINE - FULL DATASET")
    print("=" * 80)
    print("\n🎯 Strategy: Combine ALL data for maximum generalization")
    print("   - All symbols (EURUSD, GBPUSD, USDJPY, etc.)")
    print("   - All timeframes (M15, H1, H4)")
    print("   - Single unified training dataset")

    start_time = time.time()

    # ========================================================================
    # STEP 1: CONSOLIDATE MT5 EXPORTS (ALL DATA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: CONSOLIDATING ALL MT5 EXPORT FILES")
    print("=" * 80)

    print("\n📊 Configuration: FULL DATASET")
    print("   - Processing ALL symbols")
    print("   - Processing ALL timeframes")
    print("   - Creating unified dataset")

    consolidated_df = consolidate_mt5_exports(
        input_dir=data_dir,
        output_file=f'{output_dir}/consolidated_ohlc_data.csv',
        symbols=None,  # ALL symbols
        timeframes=None  # ALL timeframes
    )

    input_file = f'{output_dir}/consolidated_ohlc_data.csv'
    output_file = f'{output_dir}/processed_smc_data.csv'

    step1_time = time.time() - start_time
    print(f"\n⏱️ Step 1 completed in {step1_time:.2f} seconds")

    # Print dataset statistics
    print(f"\n📊 Consolidated Dataset Statistics:")
    print(f"   Total rows: {len(consolidated_df):,}")
    print(f"   Symbols: {consolidated_df['symbol'].nunique()}")
    print(f"   Timeframes: {consolidated_df['timeframe'].nunique()}")
    print(
        f"   Date range: {consolidated_df['time'].min()} to {consolidated_df['time'].max()}")

    # ========================================================================
    # STEP 2: PROCESS SMC FEATURES (FULL DATASET)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PROCESSING SMC FEATURES - FULL DATASET")
    print("=" * 80)
    print("\n🔧 Feature Engineering based on:")
    print("   Institutional-Grade_Quantification_and_Normalization_of_Smart_Money_Concepts_(SMC)")

    # Initialize pipeline with optimized parameters
    pipeline = SMCDataPipeline(
        base_timeframe='M15',           # Primary trading timeframe
        higher_timeframes=['H1', 'H4'],  # Higher TFs for confluence
        atr_period=14,                  # ATR calculation period
        rr_ratio=2.0,                   # 1:2 R:R (balanced risk/reward)
        lookforward=20,                 # 20 candles lookforward
        fuzzy_quality_threshold=0.2     # Quality threshold for setups
    )

    print(f"\n📋 Pipeline Configuration:")
    print(f"   Base Timeframe: M15")
    print(f"   Higher Timeframes: H1, H4")
    print(f"   ATR Period: 14")
    print(f"   Risk:Reward Ratio: 1:2.0")
    print(f"   Lookforward Period: 20 candles")
    print(f"   Quality Threshold: 0.2")

    # Run pipeline on FULL dataset
    processed_df = pipeline.run_pipeline(
        input_path=input_file,
        output_path=output_file,
        save_splits=True  # Create train/val/test splits
    )

    step2_time = time.time() - start_time - step1_time
    total_time = time.time() - start_time

    # Print processing statistics
    print(f"\n📊 Processed Dataset Statistics:")
    print(f"   Total samples: {len(processed_df):,}")
    print(f"   Features: {len(processed_df.columns)}")
    print(f"   Symbols: {processed_df['symbol'].nunique()}")

    # Label distribution
    if 'TBM_Label' in processed_df.columns:
        label_counts = processed_df['TBM_Label'].value_counts()
        print(f"\n📈 Label Distribution:")
        print(
            f"   Win (1):     {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(processed_df)*100:.1f}%)")
        print(
            f"   Loss (-1):   {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/len(processed_df)*100:.1f}%)")
        print(
            f"   Timeout (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(processed_df)*100:.1f}%)")

    # ========================================================================
    # SUMMARY & NEXT STEPS
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE - FULL DATASET READY!")
    print("=" * 80)

    print(f"\n⏱️ Execution Time:")
    print(
        f"  Step 1 (Consolidation): {step1_time:.2f}s ({step1_time/60:.1f} min)")
    print(
        f"  Step 2 (SMC Processing): {step2_time:.2f}s ({step2_time/60:.1f} min)")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    print(f"\n📁 Output Files:")
    output_path = Path(output_file).parent
    base_name = Path(output_file).stem

    print(f"\n  1. Consolidated OHLC Data:")
    print(f"     {input_file}")
    print(f"     ({len(consolidated_df):,} rows)")

    print(f"\n  2. Processed SMC Features:")
    print(f"     {output_file}")
    print(
        f"     ({len(processed_df):,} samples, {len(processed_df.columns)} features)")

    print(f"\n  3. Train/Val/Test Splits:")
    train_file = output_path / f'{base_name}_train.csv'
    val_file = output_path / f'{base_name}_val.csv'
    test_file = output_path / f'{base_name}_test.csv'

    if train_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        print(f"     Train: {train_file} ({len(train_df):,} samples)")
        print(f"     Val:   {val_file} ({len(val_df):,} samples)")
        print(f"     Test:  {test_file} ({len(test_df):,} samples)")

        # Split ratios
        total_samples = len(train_df) + len(val_df) + len(test_df)
        print(f"\n     Split Ratios:")
        print(f"     Train: {len(train_df)/total_samples*100:.1f}%")
        print(f"     Val:   {len(val_df)/total_samples*100:.1f}%")
        print(f"     Test:  {len(test_df)/total_samples*100:.1f}%")

    print(f"\n  4. Feature Documentation:")
    print(f"     {output_path / f'{base_name}_feature_list.txt'}")
    print(f"     {output_path / f'{base_name}_quality_stats.txt'}")

    print(f"\n🎯 NEXT STEPS - Model Training:")
    print(f"\n  1. Review Data Quality:")
    print(f"     - Check {base_name}_quality_stats.txt")
    print(f"     - Verify label distribution is balanced")
    print(f"     - Confirm sufficient samples per symbol")

    print(f"\n  2. Train Models:")
    print(f"     python train_all_models.py")
    print(f"     ")
    print(f"     This will train:")
    print(f"     - RandomForest (with cross-validation)")
    print(f"     - XGBoost (with early stopping)")
    print(f"     - Neural Network (simplified architecture)")
    print(f"     - LSTM (removed due to instability)")

    print(f"\n  3. Model Outputs:")
    print(f"     - Trained models: models/trained/*.pkl")
    print(f"     - Training metrics: models/trained/training_results.json")
    print(f"     - Overfitting report: models/trained/overfitting_report.md")
    print(f"     - Deployment manifest: models/trained/deployment_manifest.json")

    print(f"\n  4. Prediction Target:")
    print(f"     TBM_Label:")
    print(f"     - Win (1):     Trade hits take profit")
    print(f"     - Loss (-1):   Trade hits stop loss")
    print(f"     - Timeout (0): Trade expires without hitting either")

    print(f"\n📚 Feature Engineering Details:")
    print(f"   Based on: Institutional-Grade SMC Quantification")
    print(f"   - Order Blocks (OB): Institutional accumulation zones")
    print(f"   - Fair Value Gaps (FVG): Market imbalances")
    print(f"   - Break of Structure (BOS): Trend continuation")
    print(f"   - Change of Character (ChoCH): Trend reversal")
    print(f"   - All features normalized by ATR for stationarity")

    print("\n✅ Dataset ready for training!")
    print("=" * 80)

    return processed_df


if __name__ == "__main__":
    """
    Run the complete pipeline on FULL dataset

    This processes ALL MT5 export data into a unified training dataset
    for maximum generalization across symbols and market conditions.
    """
    import sys

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Check if running on Kaggle
    if Path('/kaggle/input').exists():
        print("🔍 Detected Kaggle environment")
        DATA_DIR = '/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports'
        OUTPUT_DIR = '/kaggle/working/Data-output'
    else:
        # Local paths (adjust for your system)
        DATA_DIR = 'Data/mt5_exports'
        OUTPUT_DIR = 'Data'

    print(f"\n📂 Data Directory: {DATA_DIR}")
    print(f"� Output Dirrectory: {OUTPUT_DIR}")
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")

    # Verify data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\n❌ Error: Data directory not found: {DATA_DIR}")
        print(f"\nPlease update DATA_DIR in the script or create the directory.")
        sys.exit(1)

    # ========================================================================
    # RUN PIPELINE
    # ========================================================================

    print("\n" + "="*80)
    print("STARTING FULL DATASET PIPELINE")
    print("="*80)
    print("\n⚠️  This will process ALL symbols and timeframes")
    print("   Estimated time: 5-15 minutes depending on data size")
    print("\nPress Ctrl+C to cancel...")

    try:
        import time
        time.sleep(2)  # Give user time to cancel

        # Run the full pipeline
        processed_data = run_full_pipeline(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR
        )

        print("\n" + "="*80)
        print("SUCCESS! Pipeline completed successfully")
        print("="*80)
        print(f"\nProcessed {len(processed_data):,} samples")
        print(f"Ready for model training with train_all_models.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
