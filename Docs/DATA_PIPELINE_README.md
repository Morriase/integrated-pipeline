# SMC Data Preparation Pipeline - Quick Start Guide

## 📋 Overview

This pipeline transforms raw MT5 OHLC data into institutional-grade SMC features ready for machine learning.

## 🚀 Quick Start (3 Steps)

### Step 1: Verify Your Data

Your MT5 exports should be in `Data/mt5_exports/` with this format:
```
Data/mt5_exports/
├── EURUSD_M15.csv
├── EURUSD_H1.csv
├── EURUSD_H4.csv
├── GBPUSD_M15.csv
└── ...
```

Each CSV should have these columns:
```
time,open,high,low,close,tick_volume,spread,real_volume
```

✅ **Your data is already in the correct format!**

### Step 2: Run the Pipeline

**Option A: Quick Test (Recommended for first run)**
```bash
python run_complete_pipeline.py
```
This processes EURUSD only (M15, H1, H4) - takes ~2-5 minutes.

**Option B: Full Pipeline (All symbols)**
Edit `run_complete_pipeline.py` line 165:
```python
# Change from:
processed_data = run_full_pipeline(config='test')

# To:
processed_data = run_full_pipeline(config='full')
```
Then run:
```bash
python run_complete_pipeline.py
```
This processes all symbols and timeframes - takes ~15-30 minutes.

### Step 3: Use the Output

The pipeline creates these files in `Data/`:

**Processed Data:**
- `processed_smc_data.csv` - Full dataset with SMC features
- `processed_smc_data_train.csv` - Training set (70%)
- `processed_smc_data_val.csv` - Validation set (15%)
- `processed_smc_data_test.csv` - Test set (15%)

**Documentation:**
- `processed_smc_data_feature_list.txt` - List of all ML features
- `processed_smc_data_quality_stats.txt` - Quality distribution stats

## 📊 What the Pipeline Does

### Phase 1: Data Consolidation
- Reads all MT5 export files
- Adds `symbol` and `timeframe` columns
- Combines into single CSV file
- Validates data format

### Phase 2: SMC Feature Engineering
1. **Order Block Detection** (with fuzzy logic)
   - Bullish/Bearish OBs
   - Displacement validation
   - Quality scoring

2. **Fair Value Gap Detection**
   - 3-candle FVG patterns
   - Mitigation tracking
   - Fuzzy size classification

3. **Market Structure**
   - Break of Structure (BOS)
   - Change of Character (ChoCH)
   - Commitment flags

4. **Multi-Timeframe Confluence**
   - HTF OB/FVG alignment
   - Trend alignment
   - Fuzzy proximity scoring

5. **Regime Classification**
   - Trend filters (EMA-based)
   - Momentum filters (RSI)
   - Volatility regimes

6. **Triple Barrier Method Labeling**
   - Win/Loss/Timeout labels
   - 1:3 Risk-Reward ratio
   - 20-candle lookforward

7. **Feature Normalization**
   - ATR normalization
   - Z-score standardization
   - Cross-asset comparability

## 🎯 Output Features

The pipeline generates **100+ features** across categories:

### Order Block Features
- `OB_Bullish`, `OB_Bearish` - Binary flags
- `OB_Size_ATR` - Size normalized by ATR
- `OB_Displacement_ATR` - Displacement strength
- `OB_Quality_Fuzzy` - Fuzzy quality score (0-1)
- `OB_Age`, `OB_Mitigated` - Tracking features

### Fair Value Gap Features
- `FVG_Bullish`, `FVG_Bearish` - Binary flags
- `FVG_Depth_ATR` - Gap size normalized
- `FVG_Quality_Fuzzy` - Fuzzy quality score
- `FVG_Mitigated` - Mitigation status

### Market Structure Features
- `BOS_Wick_Confirm`, `BOS_Close_Confirm` - Structure breaks
- `BOS_Commitment_Flag` - High-conviction signals
- `ChoCH_Detected` - Reversal signals

### Multi-Timeframe Features
- `HTF_OB_Confluence` - HTF OB alignment count
- `HTF_FVG_Confluence` - HTF FVG alignment count
- `HTF_Trend_Alignment` - HTF trend confirmation
- `HTF_Confluence_Quality` - Overall fuzzy quality

### Regime Features
- `EMA_50`, `EMA_200` - Trend indicators
- `RSI`, `RSI_Normalized` - Momentum
- `Volatility_Regime_Fuzzy` - Market state
- `Trend_Bias_Indicator` - Directional bias

### Target Variable
- `TBM_Label` - **Your ML target**
  - `-1` = Loss (hit stop loss)
  - `0` = Timeout (neither hit)
  - `+1` = Win (hit take profit)

## 🔧 Customization

### Change Base Timeframe
Edit `run_complete_pipeline.py` line 103:
```python
pipeline = SMCDataPipeline(
    base_timeframe='M15',  # Change to 'M5', 'M30', 'H1', etc.
    higher_timeframes=['H1', 'H4'],  # Adjust accordingly
    ...
)
```

### Change Risk-Reward Ratio
Edit line 106:
```python
rr_ratio=3.0,  # Change to 2.0, 4.0, etc. (1:X ratio)
```

### Change Fuzzy Quality Threshold
Edit line 108:
```python
fuzzy_quality_threshold=0.3,  # Lower = more signals, Higher = fewer but higher quality
```

### Process Specific Symbols
Edit `run_complete_pipeline.py` line 68:
```python
custom_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Add/remove symbols
custom_timeframes = ['M15', 'H1', 'H4']  # Add/remove timeframes
```

Then run with:
```python
processed_data = run_full_pipeline(config='custom')
```

## 📈 Expected Output

### Console Output
```
================================================================================
COMPLETE SMC DATA PIPELINE WORKFLOW
================================================================================

STEP 1: CONSOLIDATING MT5 EXPORT FILES
  ✓ EURUSD    M15 : 10,000 rows | 2024-01-01 to 2025-01-01
  ✓ EURUSD    H1  : 2,500 rows | 2024-01-01 to 2025-01-01
  ✓ EURUSD    H4  : 625 rows | 2024-01-01 to 2025-01-01

STEP 2: PROCESSING SMC FEATURES WITH FUZZY LOGIC
  Processing EURUSD M15...
    ✓ Valid entries: 150, Labeled: 120

📊 INSTITUTIONAL-GRADE SMC PIPELINE SUMMARY
  Order Blocks (Bullish): 75
  Order Blocks (Bearish): 80
  Fair Value Gaps (Bullish): 120
  Fair Value Gaps (Bearish): 115
  
🎯 Triple Barrier Method (TBM) Labels:
  Win     :    45 (37.5%)
  Loss    :    50 (41.7%)
  Timeout :    25 (20.8%)
  
  Win Rate (excl. timeouts): 47.4%

✅ Pipeline complete! Data ready for ML training.
```

## 🐛 Troubleshooting

### Error: "Missing required columns"
**Solution:** Check your CSV files have: `time,open,high,low,close`

### Error: "Insufficient data"
**Solution:** Each symbol-timeframe needs at least 100 rows. Check your MT5 exports.

### Warning: "No data was successfully processed"
**Solution:** 
1. Verify files are in `Data/mt5_exports/`
2. Check filename format: `SYMBOL_TIMEFRAME.csv`
3. Ensure CSV files are not empty

### Pipeline runs but no labels
**Solution:** Increase `lookforward` parameter (default: 20 candles) or decrease `rr_ratio`

## 📚 Next Steps

### 1. Explore the Data
```python
import pandas as pd

# Load training data
train = pd.read_csv('Data/processed_smc_data_train.csv')

# Check features
print(train.columns.tolist())

# Check label distribution
print(train['TBM_Label'].value_counts())

# Check feature statistics
print(train.describe())
```

### 2. Train a Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
train = pd.read_csv('Data/processed_smc_data_train.csv')
test = pd.read_csv('Data/processed_smc_data_test.csv')

# Get feature columns (read from feature_list.txt)
with open('Data/processed_smc_data_feature_list.txt') as f:
    features = [line.strip() for line in f if not line.startswith('#') and line.strip()]

# Prepare data
X_train = train[features].fillna(0)
y_train = train['TBM_Label']
X_test = test[features].fillna(0)
y_test = test['TBM_Label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 3. Backtest
Use the predictions with your trading strategy to simulate performance.

## 🎓 Understanding Fuzzy Logic

The pipeline uses **fuzzy logic** instead of rigid thresholds:

**Traditional Approach:**
```
if displacement >= 1.5 ATR:
    valid_order_block = True
else:
    valid_order_block = False
```

**Fuzzy Logic Approach:**
```
displacement = 1.4 ATR
- Weak: 0.2 (20% membership)
- Moderate: 0.6 (60% membership)
- Strong: 0.2 (20% membership)
→ Quality Score: 0.8 (weighted average)
```

This creates **smooth transitions** and **quality scoring** instead of binary yes/no decisions.

## 📞 Support

If you encounter issues:
1. Check the console output for specific error messages
2. Verify your data format matches the requirements
3. Try the 'test' configuration first before 'full'
4. Review the generated `quality_stats.txt` for data quality insights

## 🎉 Success Indicators

You'll know the pipeline worked when you see:
- ✅ All validation checks passed
- ✅ TBM labels generated (Win/Loss/Timeout distribution shown)
- ✅ Train/val/test CSV files created
- ✅ Feature list file generated
- ✅ No error messages in console

Happy trading! 🚀
