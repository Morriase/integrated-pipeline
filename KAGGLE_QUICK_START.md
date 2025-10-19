# Kaggle Quick Start Guide

## Step-by-Step Instructions

### 1. Create New Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click **"Code"** → **"New Notebook"**
3. Enable **GPU** (optional, helps with deep learning)
4. Add dataset: **"ob-ai-model-2-dataset"**
   - Click **"+ Add Data"** in right panel
   - Search for your dataset
   - Click **"Add"**

### 2. Copy and Run This Code

Create a **single code cell** and paste this:

```python
"""
SMC Pipeline - Complete Execution
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

# Install dependencies (skip if already installed)
print("📦 Installing dependencies...")
packages = ["torch", "scikit-fuzzy", "xgboost", "scikit-learn", "imbalanced-learn", "pandas", "numpy", "joblib", "matplotlib", "seaborn"]
for pkg in packages:
    subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True)
print("✅ Dependencies ready")

# Setup environment
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Run pipeline
print("🚀 Running complete pipeline...")
exec(open('KAGGLE_RUN.py').read())
```

### 3. Run the Cell

Click **"Run"** or press **Shift+Enter**

The pipeline will:
- ✅ Clone your repository
- ✅ Install all dependencies
- ✅ Consolidate MT5 data (~5-10 min)
- ✅ Process SMC features (~10-20 min)
- ✅ Train all models (~30-60 min)
- ✅ Generate results

### 4. Monitor Progress

Watch the output for:
- Data consolidation progress
- Feature engineering stats
- Model training metrics
- Final results summary

### 5. Download Results

After completion, go to **Output** tab and download:
- `training_results.json` - Performance metrics
- `*.pkl` files - Trained models
- `processed_smc_data*.csv` - Processed datasets

---

## Alternative: Manual Step-by-Step

If you prefer more control, run these in separate cells:

### Cell 1: Clone Repository
```bash
%%bash
cd /kaggle/working
git clone https://github.com/Morriase/integrated-pipeline.git
```

### Cell 2: Install Dependencies
```bash
%%bash
pip install -q torch scikit-fuzzy xgboost scikit-learn pandas numpy joblib matplotlib seaborn
```

### Cell 3: Run Pipeline
```python
import sys
import os

# Setup environment
os.chdir('/kaggle/working/integrated-pipeline')
sys.path.insert(0, '/kaggle/working/integrated-pipeline')

# Run complete pipeline
exec(open('KAGGLE_RUN.py').read())
```

---

## Expected Timeline

| Step | Duration | Output |
|------|----------|--------|
| Clone repo | 10-30 sec | Repository files |
| Install deps | 30-60 sec | Python packages |
| Consolidate data | 5-10 min | consolidated_ohlc_data.csv |
| Process features | 10-20 min | processed_smc_data*.csv |
| Train models | 30-60 min | *.pkl model files |
| **Total** | **45-90 min** | Complete results |

---

## Expected Output Files

```
/kaggle/working/
├── consolidated_ohlc_data.csv              # ~666K rows, all symbols
├── processed_smc_data.csv                  # Full dataset with features
├── processed_smc_data_train.csv            # Training split (70%)
├── processed_smc_data_val.csv              # Validation split (15%)
├── processed_smc_data_test.csv             # Test split (15%)
├── processed_smc_data_feature_list.txt     # Feature names
├── processed_smc_data_quality_stats.txt    # Quality statistics
├── training_results.json                   # Model performance
└── [SYMBOL]_[MODEL].pkl                    # Trained models (44 files)
```

---

## Troubleshooting

### Error: "Dataset not found"
**Solution**: Make sure you added the dataset to your notebook
- Click "+ Add Data" → Search → Add

### Error: "Repository not found"
**Solution**: Check the repository URL is correct
- Verify: https://github.com/Morriase/integrated-pipeline.git

### Error: "ModuleNotFoundError"
**Solution**: Dependencies not installed
- Re-run the pip install command

### Low accuracy (< 50%)
**Possible causes**:
- Not enough training data
- Imbalanced classes
- Need more epochs for deep learning

**Solutions**:
- Check class distribution in output
- Increase training epochs
- Try different hyperparameters

---

## Configuration Options

### Quick Test (Single Symbol)

Edit `run_complete_pipeline.py` before running:

```python
# Line 47: Change config
processed_data = run_full_pipeline(config='test')  # EURUSD only
```

### Custom Symbols

Edit `run_complete_pipeline.py`:

```python
# Lines 75-76
custom_symbols = ['EURUSD', 'GBPUSD']
custom_timeframes = ['M15', 'H1', 'H4']
```

Then use `config='custom'`

---

## Support

If you encounter issues:

1. Check error messages carefully
2. Verify dataset is attached
3. Ensure internet is enabled (for git clone)
4. Check execution logs for specific errors
5. Try running in separate cells for better debugging

---

## Success Indicators

You'll know it worked when you see:

✅ "Repository cloned successfully"
✅ "Data pipeline completed successfully"
✅ "Model training completed"
✅ "PIPELINE COMPLETE!"

And you have:
- 44+ `.pkl` model files
- `training_results.json` with metrics
- Processed CSV files with features

---

**Ready to start? Copy the code from Step 2 and run it!** 🚀
