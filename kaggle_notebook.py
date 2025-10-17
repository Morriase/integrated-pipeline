"""
Kaggle Notebook Script - Complete SMC Pipeline
===============================================

This script runs the complete pipeline on Kaggle:
1. Clone the repository
2. Install dependencies
3. Run data consolidation and processing
4. Train all models
5. Generate results

Usage in Kaggle:
- Add this as a notebook cell
- Ensure the dataset 'ob-ai-model-2-dataset' is attached
"""

import json
from train_all_models import SMCModelTrainer
from run_complete_pipeline import run_full_pipeline
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print output"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        sys.exit(1)
    return result


# Step 1: Clone repository (if not already done)
if not Path('/kaggle/working/integrated-pipeline').exists():
    run_command(
        'cd /kaggle/working && git clone https://github.com/YOUR_USERNAME/integrated-pipeline.git',
        "Cloning repository"
    )

# Step 2: Install dependencies
run_command(
    'pip install -q torch scikit-fuzzy xgboost scikit-learn pandas numpy joblib',
    "Installing dependencies"
)

# Step 3: Change to pipeline directory
os.chdir('/kaggle/working/integrated-pipeline')
sys.path.insert(0, '/kaggle/working/integrated-pipeline')

# Step 4: Run complete pipeline
print("\n" + "="*80)
print("RUNNING COMPLETE DATA PIPELINE")
print("="*80)


# Run pipeline (this will create processed data in /kaggle/working/)
processed_data = run_full_pipeline(config='full')

# Step 5: Train all models
print("\n" + "="*80)
print("TRAINING ALL MODELS")
print("="*80)


trainer = SMCModelTrainer()

# Check data availability
if not trainer.check_data_availability():
    print("❌ Data not available. Pipeline may have failed.")
    sys.exit(1)

# Get available symbols
symbols = trainer.get_available_symbols()

# Train all models for all symbols
results = trainer.train_all_symbols(
    symbols=symbols,
    exclude_timeout=True  # Focus on Win vs Loss
)

# Step 6: Display results
print("\n" + "="*80)
print("TRAINING COMPLETE - RESULTS SUMMARY")
print("="*80)

trainer.print_summary()

# Save results to file
results_file = '/kaggle/working/training_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {results_file}")
print("\n🎯 Check /kaggle/working/ for:")
print("  - Trained model files (.pkl)")
print("  - Training results (training_results.json)")
print("  - Processed data files (processed_smc_data_*.csv)")
