# Anti-Overfitting Quick Reference

## Problem Identified

Training results showed severe overfitting across all models:
- **Train-Val gaps:** 25-69% (target: <15%)
- **XGBoost:** 100% train accuracy (perfect memorization)
- **LSTM:** Exploding gradients (10-88k norm)

## Solution Applied

Aggressive regularization across all 4 model types.

## Quick Commands

### Test Fixes (Single Symbol)
```bash
python test_aggressive_fixes.py
```
Tests all 4 models on EURUSD with new parameters (~5 minutes)

### Full Training (All Symbols)
```bash
python train_all_models.py --symbols all --models all
```
Trains all 44 models (11 symbols × 4 models) (~5 minutes on Kaggle GPU)

### Single Model Test
```python
from models.random_forest_model import RandomForestSMCModel

model = RandomForestSMCModel(symbol='EURUSD')
# ... load data ...
history = model.train(X_train, y_train, X_val, y_val)
# Uses new aggressive defaults automatically
```

## New Default Parameters

### RandomForest
```python
max_depth=8              # Was: 10
min_samples_split=30     # Was: 20
min_samples_leaf=15      # Was: 10
max_samples=0.6          # Was: 0.7
ccp_alpha=0.02           # NEW: Cost complexity pruning
```

### XGBoost
```python
max_depth=3              # Was: 4
learning_rate=0.01       # Was: 0.1 (10x slower)
subsample=0.6            # Was: 0.7
colsample_bytree=0.6     # Was: 0.7
min_child_weight=10      # Was: 5
reg_alpha=0.5            # Was: 0.2
reg_lambda=3.0           # Was: 2.0
max_delta_step=1         # NEW: Limit weight updates
```

### Neural Network
```python
hidden_dims=[128,64,32]  # Was: [256,128,64]
dropout=0.5              # Was: 0.4
learning_rate=0.001      # Was: 0.005
weight_decay=0.01        # Was: 0.001 (10x more L2)
```

### LSTM
```python
hidden_dim=32            # Was: 64
dropout=0.6              # Was: 0.5
learning_rate=0.0001     # Was: 0.0005 (5x slower)
weight_decay=0.1         # Was: 0.05 (2x more L2)
max_grad_norm=0.5        # Was: 1.0 (stricter clipping)
```

## Success Criteria

✅ **Good:** Train-Val gap < 15%
✅ **Acceptable:** Train-Val gap < 20%
✅ **Critical:** No exploding gradients in LSTM
✅ **Critical:** XGBoost train accuracy < 95%

## Expected Changes

### Before (Old Parameters)
- RandomForest: 94-99% train, 53-78% val, 25-42% gap
- XGBoost: 99-100% train, 45-78% val, 25-55% gap
- Neural Network: 88-95% train, 45-70% val, 24-44% gap
- LSTM: 78-93% train, 16-56% val, 24-69% gap + explosions

### After (New Parameters - Expected)
- RandomForest: 75-85% train, 55-75% val, <20% gap
- XGBoost: 80-90% train, 50-75% val, <20% gap
- Neural Network: 75-85% train, 50-70% val, <20% gap
- LSTM: 65-80% train, 45-65% val, <25% gap, NO explosions

## Troubleshooting

### If train accuracy too low (<60%)
- Regularization may be too aggressive
- Try increasing learning rate slightly
- Try reducing dropout by 0.1

### If still overfitting (gap >25%)
- Increase regularization further
- Reduce model capacity (smaller networks)
- Add more data augmentation

### If LSTM still exploding
- Reduce learning rate to 0.00005
- Reduce max_grad_norm to 0.3
- Simplify architecture further

## Files Modified

1. `models/random_forest_model.py` - Lines ~50-60, ~120-130
2. `models/xgboost_model.py` - Lines ~50-70, ~100-120
3. `models/neural_network_model.py` - Lines ~80-100, ~150-160
4. `models/lstm_model.py` - Lines ~80-100, ~150-170

## Rollback

If results are worse:
```bash
git checkout models/random_forest_model.py
git checkout models/xgboost_model.py
git checkout models/neural_network_model.py
git checkout models/lstm_model.py
```

## Documentation

- `TASK_14_AGGRESSIVE_ANTI_OVERFITTING.md` - Problem analysis
- `AGGRESSIVE_FIXES_APPLIED.md` - Implementation details
- `ANTI_OVERFITTING_QUICK_REFERENCE.md` - This file
- `test_aggressive_fixes.py` - Test script

## Next Steps

1. Run `python test_aggressive_fixes.py` to verify fixes
2. If successful, run full training on Kaggle
3. Compare results with `Docs/training_progress.txt`
4. Document improvements
5. Adjust parameters if needed based on results
