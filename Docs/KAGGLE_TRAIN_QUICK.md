# Kaggle Training Only - Quick Start

Use this when you **already have processed data** and want to skip the data pipeline.

## Prerequisites

You need these files in `/kaggle/working/` or `/kaggle/input/`:
- `processed_smc_data_train.csv`
- `processed_smc_data_val.csv`
- `processed_smc_data_test.csv`

## Single Code Cell - Copy & Run

```python
"""
SMC Training Only - Skip Data Pipeline
Repository: https://github.com/Morriase/integrated-pipeline.git
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/Morriase/integrated-pipeline.git"
REPO_DIR = "/kaggle/working/integrated-pipeline"

# Clone repository
print("🔄 Cloning repository...")
if Path(REPO_DIR).exists():
    subprocess.run(f"rm -rf {REPO_DIR}", shell=True)

result = subprocess.run(f"cd /kaggle/working && git clone {REPO_URL}", shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print(f"❌ Clone failed: {result.stderr}")
    sys.exit(1)

# Install dependencies
print("📦 Installing dependencies...")
packages = ["torch", "scikit-fuzzy", "xgboost", "scikit-learn", "imbalanced-learn", "pandas", "numpy", "joblib", "matplotlib", "seaborn"]
for pkg in packages:
    subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True)
print("✅ Dependencies ready")

# Setup environment
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Run training only
print("🚀 Starting training (skipping data pipeline)...")
exec(open('KAGGLE_TRAIN_ONLY.py').read())
```

## What This Does

1. ✅ Clones repository
2. ✅ Installs dependencies
3. ✅ **Skips data consolidation and processing** (saves 15-30 min)
4. ✅ Verifies processed data exists
5. ✅ Trains all models
6. ✅ Saves results

## Expected Runtime

- **Total: 30-60 minutes** (vs 45-90 min for full pipeline)
  - Setup: 1-2 min
  - Model training: 30-60 min

## Where to Put Your Data

The script will search for data in these locations (in order):

1. `/kaggle/working/` - If you ran the pipeline before
2. `/kaggle/input/processed-smc-data/` - If you uploaded as dataset
3. `/kaggle/input/` - Any other input dataset

## If You Don't Have Processed Data Yet

You have two options:

### Option 1: Run Full Pipeline First
Use `KAGGLE_RUN.py` to generate the data:
```python
exec(open('KAGGLE_RUN.py').read())
```

### Option 2: Upload Processed Data as Dataset
1. Process data locally or in another notebook
2. Upload to Kaggle as a dataset
3. Attach dataset to your notebook
4. Run this training-only script

## Output Files

After completion:
- `training_results.json` - Performance metrics
- `*.pkl` files - Trained models (44 files)
- Models saved to `/kaggle/working/`

## Advantages

- ⚡ **Faster** - Saves 15-30 minutes
- 🔄 **Reusable** - Train multiple times on same data
- 🧪 **Experimentation** - Test different hyperparameters
- 💰 **Cost-effective** - Less compute time on Kaggle

## Next Steps

1. Review `training_results.json` for model performance
2. Download best `.pkl` models from Output tab
3. Deploy to your trading system

---

**Ready? Copy the code above and run it!** 🚀
