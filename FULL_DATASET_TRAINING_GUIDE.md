# Full Dataset Training Guide

## Overview

This guide explains how to train models on the COMPLETE dataset (all symbols, all timeframes) for maximum generalization.

## Why Use Full Dataset?

### Benefits:
1. **Better Generalization**: Model learns patterns across all market conditions
2. **More Training Data**: Larger dataset = better model performance
3. **Cross-Symbol Learning**: Patterns from EURUSD help predict GBPUSD, etc.
4. **Robust Features**: ATR normalization makes features comparable across symbols
5. **Single Model**: One model works for all symbols (simpler deployment)

### Previous Approach (Per-Symbol):
- 11 symbols × 4 models = 44 models to manage
- Each symbol had 178-295 samples (too small!)
- High overfitting due to limited data
- Models didn't generalize well

### New Approach (Full Dataset):
- 1 unified dataset with 2,000-3,000+ samples
- 3 models total (RandomForest, XGBoost, NeuralNetwork)
- Better generalization across symbols
- Easier to deploy and maintain

## Step-by-Step Process

### Step 1: Prepare Full Dataset

```bash
# Run the complete pipeline
python run_complete_pipeline.py
```

**What it does:**
1. Consolidates ALL MT5 exports from `Data/mt5_exports/`
2. Processes SMC features (OB, FVG, BOS, ChoCH)
3. Normalizes features by ATR (cross-symbol compatibility)
4. Creates train/val/test splits (70/15/15)
5. Generates feature documentation

**Output Files:**
```
Data/
├── consolidated_ohlc_data.csv          # All OHLC data combined
├── processed_smc_data.csv              # Full processed dataset
├── processed_smc_data_train.csv        # Training split (70%)
├── processed_smc_data_val.csv          # Validation split (15%)
├── processed_smc_data_test.csv         # Test split (15%)
├── processed_smc_data_feature_list.txt # Feature documentation
└── processed_smc_data_quality_stats.txt # Quality metrics
```

**Expected Output:**
```
📊 Consolidated Dataset Statistics:
   Total rows: 50,000+
   Symbols: 11
   Timeframes: 3 (M15, H1, H4)
   Date range: 2020-01-01 to 2024-12-31

📊 Processed Dataset Statistics:
   Total samples: 2,500+
   Features: 57
   Symbols: 11

📈 Label Distribution:
   Win (1):     800 (32%)
   Loss (-1):   900 (36%)
   Timeout (0): 800 (32%)
```

### Step 2: Train Models

```bash
# Train all models on full dataset
python train_all_models.py
```

**What it does:**
1. Loads train/val/test splits
2. Trains 3 models:
   - RandomForest (with cross-validation)
   - XGBoost (with early stopping)
   - NeuralNetwork (simplified architecture)
3. Evaluates on validation and test sets
4. Generates comprehensive reports

**Output Files:**
```
models/trained/
├── UNIFIED_RandomForest.pkl            # Trained RandomForest
├── UNIFIED_XGBoost.pkl                 # Trained XGBoost
├── UNIFIED_NeuralNetwork.pkl           # Trained Neural Network
├── training_results.json               # All metrics
├── overfitting_report.md               # Overfitting analysis
├── deployment_manifest.json            # Best model selection
└── reports/
    └── UNIFIED_training_report.md      # Detailed report
```

## Feature Engineering

All features are engineered according to:
**Institutional-Grade_Quantification_and_Normalization_of_Smart_Money_Concepts_(SMC)**

### Key Features (57 total):

#### 1. Order Block (OB) Features:
- `OB_Size_ATR`: Block size normalized by ATR
- `OB_Age`: Time since block formation
- `OB_Bullish_Valid`, `OB_Bearish_Valid`: Validity flags

#### 2. Fair Value Gap (FVG) Features:
- `FVG_Depth_ATR`: Gap size normalized by ATR
- `FVG_Distance_to_Price_ATR`: Distance to entry
- `FVG_Quality_Fuzzy`: Quality score (0-1)
- `FVG_Bullish_Valid`, `FVG_Bearish_Valid`: Validity flags

#### 3. Structure Features:
- `BOS_Commitment_Flag`: Break of structure strength
- `BOS_Close_Confirm`, `BOS_Wick_Confirm`: Confirmation types
- `ChoCH_Detected`, `ChoCH_Direction`: Change of character

#### 4. Trade-Based Metrics (TBM):
- `TBM_Bars_to_Hit`: Time to outcome
- `TBM_Risk_Per_Trade_ATR`: Risk normalized by ATR
- `TBM_Reward_Per_Trade_ATR`: Reward normalized by ATR

#### 5. Normalized Features (Z-Score):
- `ATR_ZScore`: Volatility standardized
- `FVG_Depth_ATR_ZScore`: Gap depth standardized
- `BOS_Dist_ATR_ZScore`: Structure distance standardized

### Why ATR Normalization?

**Problem**: Raw pip values don't work across symbols
- 50 pips on EURUSD ≠ 50 pips on USDJPY
- Different volatility regimes
- Non-stationary data

**Solution**: Normalize by ATR
- `Feature_ATR = Raw_Value / ATR(14)`
- Makes features comparable across symbols
- Handles different volatility regimes
- Creates stationary features

**Example**:
```python
# EURUSD: 50 pip move, ATR = 100 pips
FVG_Depth_ATR = 50 / 100 = 0.5

# USDJPY: 50 pip move, ATR = 50 pips  
FVG_Depth_ATR = 50 / 50 = 1.0

# Now comparable! USDJPY move is 2x more significant
```

## Model Training Details

### RandomForest:
```python
Parameters:
- n_estimators: 200
- max_depth: 15 (anti-overfitting)
- min_samples_split: 20 (anti-overfitting)
- min_samples_leaf: 10 (anti-overfitting)
- max_samples: 0.8 (bootstrap sampling)
- Cross-validation: 5-fold stratified
```

### XGBoost:
```python
Parameters:
- max_depth: 3 (aggressive regularization)
- min_child_weight: 10 (anti-overfitting)
- learning_rate: 0.01 (slow learning)
- subsample: 0.6 (randomness)
- colsample_bytree: 0.6 (feature sampling)
- early_stopping_rounds: 20 (prevent overtraining)
```

### Neural Network:
```python
Architecture:
- Input: 57 features
- Hidden: [128, 64] (simplified)
- Output: 3 classes (Win/Loss/Timeout)
- Dropout: 0.5 (regularization)
- Weight Decay: 0.01 (L2 regularization)
- Early Stopping: patience=30
```

## Expected Results

### With Full Dataset:

| Metric | Target | Previous (Per-Symbol) |
|--------|--------|----------------------|
| Train-Val Gap | <15% | 20-60% |
| Validation Accuracy | >65% | 45-78% |
| Test Accuracy | >60% | 50-81% |
| Training Time | ~5 min | ~3 min |
| Models to Manage | 3 | 44 |
| Overfitting Warnings | 0 | 24 |

### Success Indicators:

✅ **Good**:
- Train-val gap < 15%
- Test accuracy > 60%
- Consistent performance across symbols
- No exploding gradient warnings

⚠️ **Needs Review**:
- Train-val gap 15-20%
- Test accuracy 55-60%
- Some symbols perform poorly

❌ **Problem**:
- Train-val gap > 20%
- Test accuracy < 55%
- High variance across symbols

## Deployment

### Single Model for All Symbols:

```python
# Load best model
import pickle

with open('models/trained/UNIFIED_RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for any symbol
def predict_trade(symbol, features):
    """
    Predict trade outcome for any symbol
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        features: Feature vector (57 features)
    
    Returns:
        prediction: -1 (Loss), 0 (Timeout), 1 (Win)
        probability: Confidence scores
    """
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    return prediction, probabilities

# Works for all symbols!
eurusd_pred = predict_trade('EURUSD', eurusd_features)
gbpusd_pred = predict_trade('GBPUSD', gbpusd_features)
usdjpy_pred = predict_trade('USDJPY', usdjpy_features)
```

### Advantages:
1. **Simple**: One model file to deploy
2. **Consistent**: Same logic for all symbols
3. **Maintainable**: Update one model, affects all symbols
4. **Scalable**: Add new symbols without retraining

## Troubleshooting

### Issue: "Data directory not found"
**Solution**: Update `DATA_DIR` in `run_complete_pipeline.py`
```python
DATA_DIR = 'C:/Users/Morris/Desktop/Python/Data/mt5_exports'
```

### Issue: "Not enough samples"
**Solution**: Check if all MT5 exports are present
```bash
# Should have files like:
# EURUSD_M15.csv
# EURUSD_H1.csv
# EURUSD_H4.csv
# GBPUSD_M15.csv
# etc.
```

### Issue: "High overfitting"
**Solutions**:
1. Increase regularization in config.py
2. Reduce feature count (feature selection)
3. Add more data (longer time period)
4. Use data augmentation

### Issue: "Low accuracy"
**Solutions**:
1. Check label distribution (should be balanced)
2. Review feature importance (remove weak features)
3. Adjust R:R ratio in pipeline
4. Increase lookforward period

## Next Steps

1. ✅ Run `python run_complete_pipeline.py`
2. ✅ Verify output files created
3. ✅ Check quality stats
4. ✅ Run `python train_all_models.py`
5. ✅ Review training reports
6. ✅ Select best model from deployment manifest
7. ✅ Backtest on test set
8. ✅ Deploy to production

## Files Summary

### Input:
- `Data/mt5_exports/*.csv` - Raw MT5 exports

### Pipeline Output:
- `Data/consolidated_ohlc_data.csv` - Combined OHLC
- `Data/processed_smc_data*.csv` - Processed features + splits

### Training Output:
- `models/trained/*.pkl` - Trained models
- `models/trained/*.json` - Metrics and manifests
- `models/trained/*.md` - Reports

### Documentation:
- `Data/processed_smc_data_feature_list.txt` - Feature descriptions
- `models/trained/overfitting_report.md` - Overfitting analysis
- `models/trained/reports/UNIFIED_training_report.md` - Full report

## Key Takeaways

1. **Full dataset = Better generalization**
2. **ATR normalization = Cross-symbol compatibility**
3. **Fewer models = Easier deployment**
4. **More data = Less overfitting**
5. **Single model = Consistent predictions**

Good luck with your training! 🚀
