"""
RUN ON KAGGLE - Complete SMC Pipeline
======================================

Run this script after cloning the repo to Kaggle.

Prerequisites:
1. Dataset 'ob-ai-model-2-dataset' must be attached to notebook
2. Repository cloned to /kaggle/working/integrated-pipeline
3. Dependencies installed (torch, scikit-fuzzy, xgboost, etc.)

This script will:
1. Run data consolidation and processing
2. Train all models (RF, XGBoost, NN, LSTM)
3. Generate results summary
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
pipeline_dir = Path('/kaggle/working/integrated-pipeline')
if not pipeline_dir.exists():
    print("❌ Error: Repository not found at /kaggle/working/integrated-pipeline")
    print("Please clone the repository first:")
    print("  cd /kaggle/working")
    print("  git clone https://github.com/YOUR_USERNAME/integrated-pipeline.git")
    sys.exit(1)

# Add to path
sys.path.insert(0, str(pipeline_dir))

print("\n" + "="*80)
print("SMC PIPELINE - KAGGLE EXECUTION")
print("="*80)

# ============================================================================
# STEP 1: RUN DATA PIPELINE
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA CONSOLIDATION & PROCESSING")
print("="*80)

from run_complete_pipeline import run_full_pipeline

try:
    processed_data = run_full_pipeline(config='full')
    print("\n✅ Data pipeline completed successfully")
except Exception as e:
    print(f"\n❌ Data pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: TRAIN ALL MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TRAINING ALL MODELS")
print("="*80)

from train_all_models import SMCModelTrainer

try:
    trainer = SMCModelTrainer()
    
    # Check data availability
    if not trainer.check_data_availability():
        print("❌ Processed data not found. Pipeline may have failed.")
        sys.exit(1)
    
    # Get available symbols
    symbols = trainer.get_available_symbols()
    print(f"\n📊 Training models for {len(symbols)} symbols: {symbols}")
    
    # Train all models for all symbols
    results = trainer.train_all_symbols(
        symbols=symbols,
        exclude_timeout=True  # Focus on Win vs Loss (binary classification)
    )
    
    print("\n✅ Model training completed successfully")
    
except Exception as e:
    print(f"\n❌ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: RESULTS SUMMARY")
print("="*80)

trainer.print_summary()

# Save results to JSON
results_file = '/kaggle/working/training_results.json'
import json
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n💾 Results saved to: {results_file}")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\n📁 Output files in /kaggle/working/:")
print("  - consolidated_ohlc_data.csv (consolidated OHLC data)")
print("  - processed_smc_data*.csv (processed features + labels)")
print("  - *.pkl (trained model files)")
print("  - training_results.json (performance metrics)")

print("\n🎯 Next steps:")
print("  1. Review training_results.json for model performance")
print("  2. Download trained models (.pkl files) for deployment")
print("  3. Use best performing model for live trading")

print("\n✅ All done!")
