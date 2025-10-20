# Kaggle Notebook Setup Instructions

## 📋 Prerequisites

1. **Kaggle Account** - Sign up at https://www.kaggle.com
2. **Dataset Attached** - The dataset `ob-ai-model-2-dataset` must be attached to your notebook

---

## 🚀 Quick Start (Copy-Paste into Kaggle Notebook)

### Cell 1: Verify Environment
```python
import os
from pathlib import Path

# Check if dataset is attached
input_path = Path('/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports')
if input_path.exists():
    print("✅ Dataset found!")
    csv_files = list(input_path.glob('*.csv'))
    print(f"   Found {len(csv_files)} CSV files")
else:
    print("❌ Dataset not found!")
    print("   Please attach 'ob-ai-model-2-dataset' to this notebook")
```

### Cell 2: Install Dependencies (if needed)
```python
# Most packages are pre-installed on Kaggle
# Only install if you get import errors

# Uncomment if needed:
# !pip install xgboost scikit-learn torch pandas numpy
```

### Cell 3: Run Complete Pipeline
```python
# Run the entire pipeline (data prep + model training)
!python run_kaggle_pipeline.py

# Or with LSTM (experimental):
# !python run_kaggle_pipeline.py --include-lstm
```

---

## 📂 Kaggle Path Configuration

The codebase is now configured with these paths:

### Input (Read-Only)
```
/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports/
├── EURUSD_M15.csv
├── GBPUSD_H1.csv
└── ... (all MT5 export files)
```

### Output (Writable)
```
/kaggle/working/
├── Data-output/
│   ├── consolidated_ohlc_data.csv
│   ├── processed_smc_data.csv
│   ├── processed_smc_data_train.csv
│   ├── processed_smc_data_val.csv
│   └── processed_smc_data_test.csv
│
└── Model-output/
    ├── UNIFIED_RandomForest.pkl
    ├── UNIFIED_XGBoost.pkl
    ├── UNIFIED_NeuralNetwork.pkl
    ├── UNIFIED_LSTM.pkl (if enabled)
    └── training_results.json
```

---

## 🔧 Alternative: Step-by-Step Execution

If you want more control, run each step separately:

### Step 1: Data Preparation
```python
from pathlib import Path
from consolidate_mt5_data import consolidate_mt5_exports
from data_preparation_pipeline import SMCDataPipeline

# Setup paths
INPUT_DIR = '/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports'
OUTPUT_DIR = '/kaggle/working/Data-output'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Consolidate data
consolidated_df = consolidate_mt5_exports(
    input_dir=INPUT_DIR,
    output_file=f'{OUTPUT_DIR}/consolidated_ohlc_data.csv'
)

# Process SMC features
pipeline = SMCDataPipeline(
    base_timeframe='M15',
    higher_timeframes=['H1', 'H4'],
    atr_period=14,
    rr_ratio=2.0,
    lookforward=20
)

processed_df = pipeline.run_pipeline(
    input_path=f'{OUTPUT_DIR}/consolidated_ohlc_data.csv',
    output_path=f'{OUTPUT_DIR}/processed_smc_data.csv',
    save_splits=True
)

print(f"✅ Data preparation complete: {len(processed_df):,} samples")
```

### Step 2: Model Training
```python
from pathlib import Path
from train_all_models import UnifiedModelTrainer

# Setup paths
DATA_DIR = '/kaggle/working/Data-output'
MODEL_DIR = '/kaggle/working/Model-output'
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Initialize trainer
trainer = UnifiedModelTrainer(
    data_dir=DATA_DIR,
    output_dir=MODEL_DIR,
    include_lstm=False  # Set to True for LSTM
)

# Check data
if not trainer.check_data_availability():
    raise ValueError("Training data not found!")

# Train models
results = trainer.train_all_models()

# Generate report
trainer.generate_summary_report()

print("✅ Model training complete!")
```

### Step 3: View Results
```python
import json
import pandas as pd

# Load training results
with open('/kaggle/working/Model-output/training_results.json', 'r') as f:
    results = json.load(f)

# Display summary
for model_name, model_results in results['training_results']['UNIFIED'].items():
    if 'error' in model_results:
        print(f"\n{model_name}: ❌ ERROR")
        continue
    
    test_metrics = model_results.get('test_metrics', {})
    trading_metrics = test_metrics
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {test_metrics.get('accuracy', 0):.3f}")
    print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.1%}")
    print(f"  Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}")
    print(f"  EV/Trade: {trading_metrics.get('expected_value_per_trade', 0):.2f}R")
```

---

## ⏱️ Expected Execution Time

On Kaggle (with GPU enabled):
- **Data Preparation:** 5-10 minutes
- **Model Training:** 10-20 minutes
- **Total:** 15-30 minutes

---

## 💾 Saving Outputs

### Download Trained Models
```python
# In Kaggle notebook, files in /kaggle/working/ are automatically saved
# You can download them from the "Output" tab on the right

# Or create a zip file:
!zip -r models.zip /kaggle/working/Model-output/
```

### Save to Kaggle Dataset
```python
# After training, you can save outputs as a new Kaggle dataset
# 1. Go to "Output" tab
# 2. Click "Save Version"
# 3. Your models will be saved as a dataset for future use
```

---

## 🐛 Troubleshooting

### Issue: Dataset not found
**Solution:** 
1. Click "Add Data" in Kaggle
2. Search for "ob-ai-model-2-dataset"
3. Click "Add" to attach to notebook

### Issue: Out of memory
**Solution:**
1. Enable GPU: Settings → Accelerator → GPU
2. Or reduce data: Edit `run_kaggle_pipeline.py` to filter symbols

### Issue: Timeout (9 hours limit)
**Solution:**
1. Run data prep and training in separate sessions
2. Save intermediate results
3. Use Kaggle's "Save & Run All" feature

### Issue: Import errors
**Solution:**
```python
!pip install xgboost scikit-learn torch pandas numpy
```

---

## 📊 Monitoring Progress

### View Console Output
All progress is printed to console:
- Data consolidation progress
- Feature engineering status
- Model training metrics
- Final results summary

### Check Output Files
```python
# List data outputs
!ls -lh /kaggle/working/Data-output/

# List model outputs
!ls -lh /kaggle/working/Model-output/

# View training results
!cat /kaggle/working/Model-output/training_results.json
```

---

## 🎯 Success Criteria

Training is successful when you see:

```
✅ Data pipeline successful: 2,847 samples
✅ Model training successful

📁 Output Files:
Data Output (/kaggle/working/Data-output):
  - consolidated_ohlc_data.csv
  - processed_smc_data.csv
  - processed_smc_data_train.csv
  - processed_smc_data_val.csv
  - processed_smc_data_test.csv

Model Output (/kaggle/working/Model-output):
  - UNIFIED_RandomForest.pkl
  - UNIFIED_XGBoost.pkl
  - UNIFIED_NeuralNetwork.pkl
  - training_results.json

🎉 All done! Models are ready for inference.
```

---

## 📚 Next Steps

After successful training:

1. **Review Results:** Check `training_results.json`
2. **Download Models:** Save from Output tab
3. **Test Inference:** Load models and make predictions
4. **Deploy:** Use models in production trading system

---

## 🔗 Useful Kaggle Commands

```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# Check disk space
!df -h /kaggle/working

# Monitor memory usage
!free -h

# View file sizes
!du -sh /kaggle/working/*

# Compress outputs
!tar -czf outputs.tar.gz /kaggle/working/
```

---

## ✅ Complete Kaggle Notebook Template

```python
# Cell 1: Setup
import os
from pathlib import Path

print("🔍 Checking environment...")
input_path = Path('/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports')
if input_path.exists():
    print(f"✅ Dataset found: {len(list(input_path.glob('*.csv')))} files")
else:
    print("❌ Dataset not found! Please attach dataset.")

# Cell 2: Run Pipeline
print("\n🚀 Starting complete pipeline...")
!python run_kaggle_pipeline.py

# Cell 3: View Results
import json
with open('/kaggle/working/Model-output/training_results.json', 'r') as f:
    results = json.load(f)

print("\n📊 Training Results:")
for model_name, model_results in results['training_results']['UNIFIED'].items():
    if 'error' not in model_results:
        test = model_results.get('test_metrics', {})
        print(f"\n{model_name}:")
        print(f"  Accuracy: {test.get('accuracy', 0):.3f}")
        print(f"  Win Rate: {test.get('win_rate', 0):.1%}")
        print(f"  Profit Factor: {test.get('profit_factor', 0):.2f}")

print("\n✅ Pipeline complete!")
```

---

## 🎉 You're Ready!

Copy the template above into a new Kaggle notebook and run it. The entire pipeline will execute automatically with the correct paths configured for Kaggle.
