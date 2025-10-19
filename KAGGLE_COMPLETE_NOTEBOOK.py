"""
KAGGLE COMPLETE NOTEBOOK - SMC Pipeline
========================================

Complete end-to-end execution for Kaggle:
1. Clone repository from GitHub
2. Install dependencies
3. Run data pipeline
4. Train all models
5. Generate results

Repository: https://github.com/Morriase/integrated-pipeline.git

Prerequisites:
- Kaggle notebook with dataset 'ob-ai-model-2-dataset' attached
- Internet enabled (for git clone)
"""

import subprocess
import sys
import os
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_URL = "https://github.com/Morriase/integrated-pipeline.git"
REPO_DIR = "/kaggle/working/integrated-pipeline"
DATASET_PATH = "/kaggle/input/ob-ai-model-2-dataset"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd, description, check=True):
    """Run shell command and handle errors"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if check and result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"✅ {description} completed")
    return result

def check_dataset():
    """Verify dataset is attached"""
    print(f"\n{'='*80}")
    print("📂 Checking Dataset")
    print(f"{'='*80}")
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("\nPlease attach the dataset 'ob-ai-model-2-dataset' to this notebook:")
        print("  1. Click 'Add Data' in the right panel")
        print("  2. Search for 'ob-ai-model-2-dataset'")
        print("  3. Click 'Add'")
        sys.exit(1)
    
    # Check for mt5_exports directory
    mt5_dir = Path(DATASET_PATH) / "Data" / "mt5_exports"
    if not mt5_dir.exists():
        print(f"❌ MT5 exports not found at {mt5_dir}")
        sys.exit(1)
    
    csv_files = list(mt5_dir.glob("*.csv"))
    print(f"✅ Dataset found: {len(csv_files)} CSV files in mt5_exports")
    return True

# ============================================================================
# STEP 1: CLONE REPOSITORY
# ============================================================================

print("\n" + "="*80)
print("KAGGLE SMC PIPELINE - COMPLETE EXECUTION")
print("="*80)
print(f"Repository: {REPO_URL}")
print(f"Target: {REPO_DIR}")

# Check if already cloned
if Path(REPO_DIR).exists():
    print(f"\n⚠️  Repository already exists at {REPO_DIR}")
    print("Removing old version...")
    run_command(f"rm -rf {REPO_DIR}", "Removing old repository")

# Clone repository
run_command(
    f"cd /kaggle/working && git clone {REPO_URL}",
    "Cloning repository from GitHub"
)

# Verify clone
if not Path(REPO_DIR).exists():
    print(f"❌ Clone failed - directory not found: {REPO_DIR}")
    sys.exit(1)

print(f"\n✅ Repository cloned successfully")

# ============================================================================
# STEP 2: VERIFY DATASET
# ============================================================================

check_dataset()

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

run_command(
    "pip install -q torch scikit-fuzzy xgboost scikit-learn pandas numpy joblib matplotlib seaborn",
    "Installing Python dependencies"
)

# ============================================================================
# STEP 4: SETUP PYTHON PATH
# ============================================================================

print(f"\n{'='*80}")
print("🐍 Setting up Python environment")
print(f"{'='*80}")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Python path includes: {REPO_DIR}")
print("✅ Environment ready")

# ============================================================================
# STEP 5: RUN DATA PIPELINE
# ============================================================================

print(f"\n{'='*80}")
print("📊 RUNNING DATA PIPELINE")
print(f"{'='*80}")

try:
    from run_complete_pipeline import run_full_pipeline
    
    # Run full pipeline (all symbols, all timeframes)
    processed_data = run_full_pipeline(config='full')
    
    print("\n✅ Data pipeline completed successfully")
    print(f"   Processed {len(processed_data):,} rows")
    
except Exception as e:
    print(f"\n❌ Data pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 6: TRAIN ALL MODELS
# ============================================================================

print(f"\n{'='*80}")
print("🤖 TRAINING ALL MODELS")
print(f"{'='*80}")

try:
    from train_all_models import SMCModelTrainer
    
    # Initialize trainer
    trainer = SMCModelTrainer(
        data_dir='/kaggle/working',
        output_dir='/kaggle/working'
    )
    
    # Check data availability
    if not trainer.check_data_availability():
        print("❌ Processed data not found")
        sys.exit(1)
    
    # Get available symbols
    symbols = trainer.get_available_symbols()
    print(f"\n📊 Training models for {len(symbols)} symbols")
    
    # Train all models for all symbols
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'#'*80}")
        print(f"# SYMBOL: {symbol}")
        print(f"{'#'*80}")
        
        try:
            symbol_results = trainer.train_all_for_symbol(
                symbol=symbol,
                exclude_timeout=True,  # Binary: Win vs Loss
                models=['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']
            )
            all_results[symbol] = symbol_results
            
        except Exception as e:
            print(f"\n⚠️  Error training {symbol}: {e}")
            all_results[symbol] = {'error': str(e)}
            continue
    
    print("\n✅ Model training completed")
    
except Exception as e:
    print(f"\n❌ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7: GENERATE RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("📈 RESULTS SUMMARY")
print(f"{'='*80}")

# Print summary table
print(f"\n{'Symbol':<10} {'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Status':<15}")
print("-" * 75)

for symbol, symbol_results in all_results.items():
    if 'error' in symbol_results:
        print(f"{symbol:<10} {'ALL':<20} {'ERROR':<12} {'ERROR':<12} {'❌ Failed':<15}")
        continue
    
    for model_name, result in symbol_results.items():
        if 'error' in result:
            print(f"{symbol:<10} {model_name:<20} {'ERROR':<12} {'ERROR':<12} {'❌ Failed':<15}")
            continue
        
        val_acc = result.get('val_metrics', {}).get('accuracy', 0)
        test_acc = result.get('test_metrics', {}).get('accuracy', 0)
        status = '✅ Success' if test_acc > 0.5 else '⚠️  Low Acc'
        
        print(f"{symbol:<10} {model_name:<20} {val_acc:<12.3f} {test_acc:<12.3f} {status:<15}")

# Save results to JSON
results_file = '/kaggle/working/training_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n💾 Results saved to: {results_file}")

# ============================================================================
# STEP 8: LIST OUTPUT FILES
# ============================================================================

print(f"\n{'='*80}")
print("📁 OUTPUT FILES")
print(f"{'='*80}")

output_dir = Path('/kaggle/working')
files = {
    'Data Files': list(output_dir.glob('*.csv')),
    'Model Files': list(output_dir.glob('*.pkl')),
    'Result Files': list(output_dir.glob('*.json')) + list(output_dir.glob('*.txt'))
}

for category, file_list in files.items():
    print(f"\n{category}:")
    if file_list:
        for f in sorted(file_list):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
    else:
        print("  (none)")

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*80}")
print("✅ PIPELINE COMPLETE!")
print(f"{'='*80}")

print("\n🎯 Next Steps:")
print("  1. Review training_results.json for model performance")
print("  2. Download best performing models (.pkl files)")
print("  3. Check the Output tab to download files")
print("  4. Deploy models to your trading system")

print("\n📊 Quick Stats:")
print(f"  Symbols trained: {len(symbols)}")
print(f"  Models per symbol: 4 (RF, XGBoost, NN, LSTM)")
print(f"  Total models: {len(symbols) * 4}")

print("\n✨ All done! Check /kaggle/working/ for all outputs.")
