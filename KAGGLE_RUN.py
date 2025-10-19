"""
KAGGLE SELF-CONTAINED RUNNER
=============================

Complete pipeline execution - no external file dependencies.
Just run this single script in Kaggle!

Repository: https://github.com/Morriase/integrated-pipeline.git
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
# STEP 1: CLONE REPOSITORY
# ============================================================================

print("\n" + "="*80)
print("🚀 KAGGLE SMC PIPELINE - COMPLETE EXECUTION")
print("="*80)

# Remove old clone if exists
if Path(REPO_DIR).exists():
    print(f"\n⚠️  Removing existing repository...")
    subprocess.run(f"rm -rf {REPO_DIR}", shell=True)

# Clone repository
print(f"\n📥 Cloning repository from GitHub...")
print(f"   URL: {REPO_URL}")
print(f"   Target: {REPO_DIR}")

result = subprocess.run(
    f"cd /kaggle/working && git clone {REPO_URL}",
    shell=True,
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"❌ Clone failed!")
    print(result.stderr)
    sys.exit(1)

if not Path(REPO_DIR).exists():
    print(f"❌ Repository not found after clone: {REPO_DIR}")
    sys.exit(1)

print(f"✅ Repository cloned successfully")

# ============================================================================
# STEP 2: VERIFY DATASET
# ============================================================================

print(f"\n{'='*80}")
print("📂 Verifying Dataset")
print(f"{'='*80}")

if not Path(DATASET_PATH).exists():
    print(f"❌ Dataset not found: {DATASET_PATH}")
    print("\nPlease attach 'ob-ai-model-2-dataset' to this notebook:")
    print("  1. Click '+ Add Data' in right panel")
    print("  2. Search for 'ob-ai-model-2-dataset'")
    print("  3. Click 'Add'")
    sys.exit(1)

mt5_dir = Path(DATASET_PATH) / "Data" / "mt5_exports"
if not mt5_dir.exists():
    print(f"❌ MT5 exports not found: {mt5_dir}")
    sys.exit(1)

csv_files = list(mt5_dir.glob("*.csv"))
print(f"✅ Dataset verified: {len(csv_files)} CSV files found")

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

print(f"\n{'='*80}")
print("📦 Installing Dependencies")
print(f"{'='*80}")

# Install packages one by one to handle failures gracefully
packages = [
    "torch",
    "scikit-fuzzy", 
    "xgboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "matplotlib",
    "seaborn"
]

failed_packages = []
for package in packages:
    result = subprocess.run(
        f"pip install -q {package}",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        failed_packages.append(package)
        print(f"⚠️  {package} - failed")
    else:
        print(f"✅ {package} - installed")

if failed_packages:
    print(f"\n⚠️  Failed packages: {', '.join(failed_packages)}")
    print("Continuing anyway - some may already be installed...")
else:
    print(f"\n✅ All dependencies installed")

# ============================================================================
# STEP 4: SETUP ENVIRONMENT
# ============================================================================

print(f"\n{'='*80}")
print("🐍 Setting up Python Environment")
print(f"{'='*80}")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Python path: {REPO_DIR}")
print("✅ Environment ready")

# ============================================================================
# STEP 5: RUN DATA PIPELINE
# ============================================================================

print(f"\n{'='*80}")
print("📊 RUNNING DATA PIPELINE")
print(f"{'='*80}")

try:
    from run_complete_pipeline import run_full_pipeline
    
    # Run full pipeline
    processed_data = run_full_pipeline(config='full')
    
    print(f"\n✅ Data pipeline completed")
    print(f"   Processed: {len(processed_data):,} rows")
    
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
    
    # Check data
    if not trainer.check_data_availability():
        print("❌ Processed data not found")
        sys.exit(1)
    
    # Get symbols
    symbols = trainer.get_available_symbols()
    print(f"\n📊 Training {len(symbols)} symbols: {symbols}")
    
    # Train all models
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'#'*80}")
        print(f"# TRAINING: {symbol}")
        print(f"{'#'*80}")
        
        try:
            symbol_results = trainer.train_all_for_symbol(
                symbol=symbol,
                exclude_timeout=True,
                models=['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']
            )
            all_results[symbol] = symbol_results
            
        except Exception as e:
            print(f"\n⚠️  Error training {symbol}: {e}")
            all_results[symbol] = {'error': str(e)}
            continue
    
    print("\n✅ Model training completed")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7: RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("📈 RESULTS SUMMARY")
print(f"{'='*80}")

# Print table
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

# Save results
results_file = '/kaggle/working/training_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n💾 Results saved: {results_file}")

# ============================================================================
# STEP 8: OUTPUT FILES
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
print("  1. Review training_results.json")
print("  2. Download .pkl model files from Output tab")
print("  3. Deploy best models to trading system")

print(f"\n📊 Summary:")
print(f"  Symbols: {len(symbols)}")
print(f"  Models per symbol: 4")
print(f"  Total models: {len(symbols) * 4}")

print("\n✨ Check /kaggle/working/ for all outputs!")
