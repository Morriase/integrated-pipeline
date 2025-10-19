"""
KAGGLE TRAIN ONLY - Skip Data Pipeline
=======================================

Use this when you already have processed data and just want to train models.

Prerequisites:
- Processed data files already exist in /kaggle/working/ or /kaggle/input/
- Files needed: processed_smc_data_train.csv, _val.csv, _test.csv
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

# ============================================================================
# STEP 1: CLONE REPOSITORY
# ============================================================================

print("\n" + "="*80)
print("🚀 KAGGLE TRAINING ONLY - SKIP DATA PIPELINE")
print("="*80)

# Remove old clone if exists
if Path(REPO_DIR).exists():
    print(f"\n⚠️  Removing existing repository...")
    subprocess.run(f"rm -rf {REPO_DIR}", shell=True)

# Clone repository
print(f"\n📥 Cloning repository from GitHub...")
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

print(f"✅ Repository cloned")

# ============================================================================
# STEP 2: INSTALL DEPENDENCIES
# ============================================================================

print(f"\n{'='*80}")
print("📦 Installing Dependencies")
print(f"{'='*80}")

packages = [
    "torch",
    "scikit-fuzzy", 
    "xgboost",
    "scikit-learn",
    "imbalanced-learn",
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
    print("Continuing anyway...")

# ============================================================================
# STEP 3: SETUP ENVIRONMENT
# ============================================================================

print(f"\n{'='*80}")
print("🐍 Setting up Environment")
print(f"{'='*80}")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

print(f"Working directory: {os.getcwd()}")
print("✅ Environment ready")

# ============================================================================
# STEP 4: VERIFY DATA FILES
# ============================================================================

print(f"\n{'='*80}")
print("📂 Verifying Data Files")
print(f"{'='*80}")

# Check common locations for data
data_locations = [
    "/kaggle/working",
    "/kaggle/input/processed-smc-data",
    "/kaggle/input"
]

data_dir = None
for location in data_locations:
    train_file = Path(location) / "processed_smc_data_train.csv"
    if train_file.exists():
        data_dir = location
        print(f"✅ Found data in: {location}")
        break

if data_dir is None:
    print("❌ Processed data not found!")
    print("\nSearched locations:")
    for loc in data_locations:
        print(f"  - {loc}")
    print("\nPlease ensure you have:")
    print("  - processed_smc_data_train.csv")
    print("  - processed_smc_data_val.csv")
    print("  - processed_smc_data_test.csv")
    print("\nEither:")
    print("  1. Run the full pipeline first (KAGGLE_RUN.py)")
    print("  2. Upload processed data as a Kaggle dataset")
    sys.exit(1)

# Verify all required files
required_files = [
    "processed_smc_data_train.csv",
    "processed_smc_data_val.csv",
    "processed_smc_data_test.csv"
]

missing_files = []
for filename in required_files:
    filepath = Path(data_dir) / filename
    if not filepath.exists():
        missing_files.append(filename)
    else:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✅ {filename} ({size_mb:.2f} MB)")

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    sys.exit(1)

print(f"\n✅ All data files verified")

# ============================================================================
# STEP 5: TRAIN ALL MODELS
# ============================================================================

print(f"\n{'='*80}")
print("🤖 TRAINING ALL MODELS")
print(f"{'='*80}")

try:
    from train_all_models import SMCModelTrainer
    
    # Initialize trainer with data location
    trainer = SMCModelTrainer(
        data_dir=data_dir,
        output_dir='/kaggle/working'
    )
    
    # Check data
    if not trainer.check_data_availability():
        print("❌ Data check failed")
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
# STEP 6: RESULTS SUMMARY
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
# STEP 7: OUTPUT FILES
# ============================================================================

print(f"\n{'='*80}")
print("📁 OUTPUT FILES")
print(f"{'='*80}")

output_dir = Path('/kaggle/working')
model_files = list(output_dir.glob('*.pkl'))

print(f"\nTrained Models ({len(model_files)} files):")
if model_files:
    for f in sorted(model_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
else:
    print("  (none)")

# ============================================================================
# COMPLETION
# ============================================================================

print(f"\n{'='*80}")
print("✅ TRAINING COMPLETE!")
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
