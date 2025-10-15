#!/usr/bin/env python3
"""
Colab-specific training script for Integrated Advanced SMC Pipeline
Handles Google Drive paths and environment setup
"""
import sys
from pathlib import Path
import os

print("="*80)
print("COLAB TRAINING SETUP")
print("="*80)

# Detect if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("✓ Running locally")

# Get the correct paths
current_file = Path(__file__).resolve()
python_dir = current_file.parent
project_root = python_dir.parent

print(f"\n📁 Path Configuration:")
print(f"   Current file: {current_file}")
print(f"   Python dir: {python_dir}")
print(f"   Project root: {project_root}")

# Add Python directory to path
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
    print(f"   Added to sys.path: {python_dir}")

# Check for required data file
data_file = project_root / 'Data' / 'mt5_features_institutional_regime_filtered.csv'
print(f"\n📊 Data File Check:")
print(f"   Looking for: {data_file}")
print(f"   Exists: {data_file.exists()}")

if not data_file.exists():
    print(f"\n❌ ERROR: Required data file not found!")
    print(f"\n   Please run the feature engineering script first:")
    print(f"   python {project_root}/Python/feature_engineering_smc_institutional.py")
    print(f"\n   Or check if the file is in a different location.")
    
    # Try to find it
    print(f"\n🔍 Searching for CSV files in Data directory...")
    data_dir = project_root / 'Data'
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            print(f"   Found {len(csv_files)} CSV files:")
            for f in sorted(csv_files)[:10]:  # Show first 10
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"     • {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"   No CSV files found in {data_dir}")
    else:
        print(f"   Data directory not found: {data_dir}")
    
    sys.exit(1)

# File exists, proceed with training
print(f"\n✅ Data file found!")
size_mb = data_file.stat().st_size / (1024 * 1024)
print(f"   Size: {size_mb:.2f} MB")

# Import and run the integrated pipeline
print(f"\n🚀 Starting Integrated Advanced SMC Pipeline...")
print("="*80)

# Change to Python directory for imports
os.chdir(python_dir)

# Now import and run
from integrated_advanced_pipeline import IntegratedSMCSystem

# Configuration
config = {
    'data_path': str(data_file),
    'sequence_length': 20,
    'prediction_horizon': 8,
    'save_path': python_dir / 'Model_output'
}

print(f"\n⚙️  Configuration:")
print(f"   Data: {config['data_path']}")
print(f"   Sequence length: {config['sequence_length']}")
print(f"   Prediction horizon: {config['prediction_horizon']}")
print(f"   Save path: {config['save_path']}")

# Initialize and run
try:
    system = IntegratedSMCSystem(config)
    
    # Prepare dataset
    print("\n" + "="*80)
    print("PREPARING DATASET")
    print("="*80)
    dataset = system.prepare_comprehensive_dataset(config['data_path'])
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Base samples: {len(dataset['base_data']['train']['features']):,}")
    print(f"   Temporal sequences: {len(dataset['temporal_data']['train']['sequences']):,}")
    print(f"   Features: {dataset['metadata']['n_features']}")
    print(f"   Classes: {dataset['metadata']['n_classes']}")
    
    # Continue with full pipeline...
    print(f"\n💡 Dataset ready for training!")
    print(f"   To continue with full training, uncomment the training code below.")
    
    # Uncomment to run full training:
    # from integrated_advanced_pipeline import main
    # system, results = main()
    
except Exception as e:
    print(f"\n❌ ERROR during execution:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ COLAB SETUP COMPLETE")
print("="*80)
