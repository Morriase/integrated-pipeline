# Kaggle Quick Start - Copy & Paste

## 🚀 One-Command Execution

Copy this into a Kaggle notebook cell:

```python
!python run_kaggle_pipeline.py
```

That's it! The pipeline will:
1. ✅ Detect Kaggle environment automatically
2. ✅ Read from `/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports`
3. ✅ Save data to `/kaggle/working/Data-output`
4. ✅ Save models to `/kaggle/working/Model-output`
5. ✅ Train 3 models (RF, XGB, NN) on unified dataset

---

## 📂 Configured Paths

### ✅ Input (Automatic)
```
/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports/
```

### ✅ Data Output (Automatic)
```
/kaggle/working/Data-output/
├── consolidated_ohlc_data.csv
├── processed_smc_data.csv
├── processed_smc_data_train.csv
├── processed_smc_data_val.csv
└── processed_smc_data_test.csv
```

### ✅ Model Output (Automatic)
```
/kaggle/working/Model-output/
├── UNIFIED_RandomForest.pkl
├── UNIFIED_XGBoost.pkl
├── UNIFIED_NeuralNetwork.pkl
└── training_results.json
```

---

## 🎯 Modified Files

These files now auto-detect Kaggle and use correct paths:

1. ✅ `run_complete_pipeline.py` - Auto-detects Kaggle, uses `/kaggle/working/Data-output`
2. ✅ `train_all_models.py` - Auto-detects Kaggle, uses `/kaggle/working/Model-output`
3. ✅ `run_kaggle_pipeline.py` - NEW: Kaggle-optimized launcher

---

## 📝 Complete Kaggle Notebook Template

```python
# ============================================================
# CELL 1: Verify Dataset
# ============================================================
from pathlib import Path

input_path = Path('/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports')
if input_path.exists():
    csv_files = list(input_path.glob('*.csv'))
    print(f"✅ Dataset found: {len(csv_files)} CSV files")
else:
    print("❌ Dataset not found!")
    print("Please attach 'ob-ai-model-2-dataset' to this notebook")

# ============================================================
# CELL 2: Run Complete Pipeline
# ============================================================
!python run_kaggle_pipeline.py

# ============================================================
# CELL 3: View Results
# ============================================================
import json

with open('/kaggle/working/Model-output/training_results.json', 'r') as f:
    results = json.load(f)

print("\n📊 Model Performance Summary:\n")
print(f"{'Model':<20} {'Accuracy':<12} {'Win Rate':<12} {'Profit Factor':<15} {'EV/Trade':<12}")
print("=" * 80)

for model_name, model_results in results['training_results']['UNIFIED'].items():
    if 'error' in model_results:
        print(f"{model_name:<20} {'ERROR':<12}")
        continue
    
    test = model_results.get('test_metrics', {})
    acc = test.get('accuracy', 0)
    wr = test.get('win_rate', 0)
    pf = test.get('profit_factor', 0)
    ev = test.get('expected_value_per_trade', 0)
    
    print(f"{model_name:<20} {acc:<12.3f} {wr:<12.1%} {pf:<15.2f} {ev:<12.2f}R")

print("\n✅ Training complete! Models saved to /kaggle/working/Model-output/")
```

---

## ⚡ With LSTM (Optional)

```python
!python run_kaggle_pipeline.py --include-lstm
```

---

## 💾 Download Models

After training, download from Kaggle's Output tab, or:

```python
# Create zip file
!zip -r trained_models.zip /kaggle/working/Model-output/

# Download link will appear in output
```

---

## ⏱️ Expected Time

- Data Preparation: 5-10 minutes
- Model Training: 10-20 minutes
- **Total: 15-30 minutes**

---

## 🎉 That's It!

The codebase is now fully configured for Kaggle. Just:
1. Create new Kaggle notebook
2. Attach `ob-ai-model-2-dataset`
3. Copy the template above
4. Run!

All paths are handled automatically. No manual configuration needed.
