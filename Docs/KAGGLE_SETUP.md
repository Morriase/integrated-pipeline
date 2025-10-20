# Kaggle Setup Guide

## Quick Start

### 1. Create Kaggle Notebook

1. Go to Kaggle.com and create a new notebook
2. Enable GPU/TPU (optional, but recommended for deep learning)
3. Add the dataset: `ob-ai-model-2-dataset`

### 2. Install Dependencies

Run this in a notebook cell:

```bash
%%bash
pip install -q torch scikit-fuzzy xgboost scikit-learn pandas numpy joblib
```

### 3. Clone Repository

```bash
%%bash
cd /kaggle/working
git clone https://github.com/YOUR_USERNAME/integrated-pipeline.git
```

### 4. Run Pipeline

```python
import sys
sys.path.insert(0, '/kaggle/working/integrated-pipeline')

# Run the complete pipeline
exec(open('/kaggle/working/integrated-pipeline/RUN_ON_KAGGLE.py').read())
```

## What Happens

The pipeline will:

1. **Consolidate MT5 Data** (5-10 minutes)
   - Reads 68 CSV files from `/kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports/`
   - Combines into single file: `/kaggle/working/consolidated_ohlc_data.csv`
   - ~666K rows across 11 symbols and 7 timeframes

2. **Process SMC Features** (10-20 minutes)
   - Detects order blocks, fair value gaps, liquidity zones
   - Applies fuzzy logic quality scoring
   - Generates labels (Win/Loss/Timeout)
   - Creates train/val/test splits
   - Output: `/kaggle/working/processed_smc_data*.csv`

3. **Train Models** (20-40 minutes)
   - Random Forest (fast, interpretable)
   - XGBoost (high accuracy)
   - Neural Network (deep learning)
   - LSTM (temporal patterns)
   - Trains per symbol (11 symbols × 4 models = 44 models)

4. **Generate Results**
   - Performance metrics (accuracy, precision, recall, F1)
   - Win rate analysis
   - Feature importance
   - Saves to `/kaggle/working/training_results.json`

## Expected Output

```
/kaggle/working/
├── consolidated_ohlc_data.csv          # Consolidated OHLC data
├── processed_smc_data.csv              # Full processed dataset
├── processed_smc_data_train.csv        # Training split (70%)
├── processed_smc_data_val.csv          # Validation split (15%)
├── processed_smc_data_test.csv         # Test split (15%)
├── processed_smc_data_feature_list.txt # Feature names
├── processed_smc_data_quality_stats.txt # Quality statistics
├── training_results.json               # Model performance
└── *.pkl                               # Trained model files
```

## Troubleshooting

### Error: "Read-only file system"

**Problem**: Trying to write to `/kaggle/input/`

**Solution**: All scripts now write to `/kaggle/working/` - this should be fixed

### Error: "ModuleNotFoundError: No module named 'models'"

**Problem**: Python can't find the models directory

**Solution**: The scripts now add the pipeline directory to sys.path - this should be fixed

### Error: "No such file or directory: processed_smc_data_train.csv"

**Problem**: Data pipeline didn't complete successfully

**Solution**: 
1. Check if consolidation step completed
2. Look for error messages in Step 1 output
3. Verify dataset is attached correctly

### Low Win Rates (< 50%)

**Problem**: Not enough labeled data for deep learning

**Solution**: The pipeline now uses:
- Lower quality threshold (0.2 instead of 0.4)
- Shorter lookforward (20 instead of 30)
- Lower R:R ratio (2.0 instead of 2.5)

This should generate more labeled samples.

## Performance Expectations

Based on previous runs:

| Model | Expected Win Rate | Training Time |
|-------|------------------|---------------|
| Random Forest | 55-60% | 2-5 min/symbol |
| XGBoost | 55-60% | 3-7 min/symbol |
| Neural Network | 50-55% | 5-10 min/symbol |
| LSTM | 50-55% | 10-20 min/symbol |

**Note**: Deep learning models need 10,000+ labeled samples to perform well. If you have fewer samples, Random Forest and XGBoost will likely perform better.

## Next Steps After Training

1. **Review Results**
   ```python
   import json
   with open('/kaggle/working/training_results.json') as f:
       results = json.load(f)
   print(json.dumps(results, indent=2))
   ```

2. **Download Best Models**
   - Go to Output tab in Kaggle
   - Download `.pkl` files for best performing models
   - Use in your trading system

3. **Deploy to Production**
   - Load model: `model = joblib.load('model.pkl')`
   - Prepare features from live data
   - Predict: `prediction = model.predict(features)`
   - Execute trades based on predictions

## Configuration Options

### Quick Test (Single Symbol)

Edit `RUN_ON_KAGGLE.py` line 47:

```python
processed_data = run_full_pipeline(config='test')  # EURUSD only
```

### Custom Symbols/Timeframes

Edit `run_complete_pipeline.py` lines 75-76:

```python
custom_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
custom_timeframes = ['M15', 'H1', 'H4']
```

Then use:
```python
processed_data = run_full_pipeline(config='custom')
```

## Support

If you encounter issues:

1. Check the error message carefully
2. Verify all paths use `/kaggle/working/` for outputs
3. Ensure dataset is attached correctly
4. Check that dependencies are installed
5. Review the execution logs for specific errors
