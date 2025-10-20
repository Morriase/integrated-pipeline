"""
Kaggle-Optimized Pipeline Runner
=================================

This script is specifically configured for Kaggle notebooks with:
- Input: /kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports
- Data Output: /kaggle/working/Data-output
- Model Output: /kaggle/working/Model-output

Usage in Kaggle Notebook:
    !python run_kaggle_pipeline.py
    
Or run in cells:
    %run run_kaggle_pipeline.py
"""

from pathlib import Path
import sys
import time
from datetime import datetime

# Kaggle paths
KAGGLE_INPUT = '/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports'
KAGGLE_DATA_OUTPUT = '/kaggle/working/Data-output'
KAGGLE_MODEL_OUTPUT = '/kaggle/working/Model-output'


def setup_kaggle_environment():
    """Setup Kaggle environment and verify paths"""
    print("=" * 80)
    print("KAGGLE ENVIRONMENT SETUP")
    print("=" * 80)
    
    # Verify input data exists
    input_path = Path(KAGGLE_INPUT)
    if not input_path.exists():
        print(f"❌ ERROR: Input data not found at {KAGGLE_INPUT}")
        print("\nPlease ensure the dataset is attached to your Kaggle notebook:")
        print("  1. Click 'Add Data' in Kaggle")
        print("  2. Search for 'ob-ai-model-2-dataset'")
        print("  3. Add to notebook")
        sys.exit(1)
    
    print(f"✓ Input data found: {KAGGLE_INPUT}")
    
    # Count input files
    csv_files = list(input_path.glob('*.csv'))
    print(f"  Found {len(csv_files)} CSV files")
    
    # Create output directories
    Path(KAGGLE_DATA_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(KAGGLE_MODEL_OUTPUT).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Data output directory: {KAGGLE_DATA_OUTPUT}")
    print(f"✓ Model output directory: {KAGGLE_MODEL_OUTPUT}")
    
    return True


def run_data_pipeline():
    """Run data preparation pipeline"""
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    from consolidate_mt5_data import consolidate_mt5_exports
    from data_preparation_pipeline import SMCDataPipeline
    
    start_time = time.time()
    
    # Consolidate MT5 exports
    print("\n📊 Consolidating MT5 export files...")
    consolidated_df = consolidate_mt5_exports(
        input_dir=KAGGLE_INPUT,
        output_file=f'{KAGGLE_DATA_OUTPUT}/consolidated_ohlc_data.csv',
        symbols=None,  # ALL symbols
        timeframes=None  # ALL timeframes
    )
    
    consolidation_time = time.time() - start_time
    print(f"✓ Consolidation complete: {consolidation_time:.1f}s")
    
    # Process SMC features
    print("\n🔧 Processing SMC features...")
    pipeline = SMCDataPipeline(
        base_timeframe='M15',
        higher_timeframes=['H1', 'H4'],
        atr_period=14,
        rr_ratio=2.0,
        lookforward=20,
        fuzzy_quality_threshold=0.2
    )
    
    processed_df = pipeline.run_pipeline(
        input_path=f'{KAGGLE_DATA_OUTPUT}/consolidated_ohlc_data.csv',
        output_path=f'{KAGGLE_DATA_OUTPUT}/processed_smc_data.csv',
        save_splits=True
    )
    
    total_time = time.time() - start_time
    print(f"\n✓ Data pipeline complete: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return processed_df


def run_model_training(include_lstm=False):
    """Run model training pipeline"""
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    from train_all_models import UnifiedModelTrainer
    
    start_time = time.time()
    
    # Initialize trainer
    trainer = UnifiedModelTrainer(
        data_dir=KAGGLE_DATA_OUTPUT,
        output_dir=KAGGLE_MODEL_OUTPUT,
        include_lstm=include_lstm
    )
    
    # Check data availability
    if not trainer.check_data_availability():
        print("\n❌ Training data not found!")
        print("Please run data pipeline first.")
        sys.exit(1)
    
    # Train all models
    results = trainer.train_all_models()
    
    # Generate summary
    trainer.generate_summary_report()
    
    total_time = time.time() - start_time
    print(f"\n✓ Model training complete: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return results


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("KAGGLE SMC MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Setup environment
    if not setup_kaggle_environment():
        sys.exit(1)
    
    # Run data pipeline
    try:
        processed_df = run_data_pipeline()
        print(f"\n✅ Data pipeline successful: {len(processed_df):,} samples")
    except Exception as e:
        print(f"\n❌ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run model training
    try:
        # Check if user wants LSTM (can be set via environment variable)
        include_lstm = '--include-lstm' in sys.argv
        
        results = run_model_training(include_lstm=include_lstm)
        print(f"\n✅ Model training successful")
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final summary
    overall_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 Output Files:")
    print(f"\nData Output ({KAGGLE_DATA_OUTPUT}):")
    print("  - consolidated_ohlc_data.csv")
    print("  - processed_smc_data.csv")
    print("  - processed_smc_data_train.csv")
    print("  - processed_smc_data_val.csv")
    print("  - processed_smc_data_test.csv")
    
    print(f"\nModel Output ({KAGGLE_MODEL_OUTPUT}):")
    print("  - UNIFIED_RandomForest.pkl")
    print("  - UNIFIED_XGBoost.pkl")
    print("  - UNIFIED_NeuralNetwork.pkl")
    if include_lstm:
        print("  - UNIFIED_LSTM.pkl")
    print("  - training_results.json")
    
    print("\n🎉 All done! Models are ready for inference.")


if __name__ == "__main__":
    """
    Run complete pipeline on Kaggle
    
    Usage:
        # Without LSTM (recommended)
        !python run_kaggle_pipeline.py
        
        # With LSTM (experimental)
        !python run_kaggle_pipeline.py --include-lstm
    """
    main()
