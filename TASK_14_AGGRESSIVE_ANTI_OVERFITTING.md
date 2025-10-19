# Task 14: Aggressive Anti-Overfitting Fixes

## Problem Analysis

Based on training results from `Docs/training_progress.txt`, all models show severe overfitting:

### Current Issues by Model Type

**RandomForest:**
- Train accuracy: 94-99%
- Val accuracy: 53-78%
- Train-Val gaps: 25-42% (target: <15%)
- Issue: Trees too deep, insufficient regularization

**XGBoost:**
- Train accuracy: 99-100% (CRITICAL - perfect memorization)
- Val accuracy: 45-78%
- Train-Val gaps: 25-55%
- Issue: Learning rate too high, trees too deep

**Neural Network:**
- Train accuracy: 88-95%
- Val accuracy: 45-70%
- Train-Val gaps: 24-44%
- Issue: Network too large, dropout insufficient

**LSTM (WORST):**
- Train accuracy: 78-93%
- Val accuracy: 16-56%
- Train-Val gaps: 24-69%
- Exploding gradients: 10-88k norm!
- Issue: Architecture too complex, learning rate too high

## Solution: Aggressive Regularization

### 1. RandomForest Changes

**Current:**
```python
max_depth=10
min_samples_split=20
min_samples_leaf=10
max_samples=0.7
```

**New (AGGRESSIVE):**
```python
max_depth=8              # Shallower trees
min_samples_split=30     # More samples required to split
min_samples_leaf=15      # Larger leaves
max_samples=0.6          # Less bootstrap sampling
ccp_alpha=0.02           # Cost complexity pruning
```

### 2. XGBoost Changes

**Current:**
```python
max_depth=4
learning_rate=0.1
subsample=0.7
min_child_weight=5
reg_alpha=0.2
reg_lambda=2.0
```

**New (AGGRESSIVE):**
```python
max_depth=3              # Even shallower
learning_rate=0.01       # 10x slower learning
subsample=0.6            # More regularization
colsample_bytree=0.6     # More feature sampling
min_child_weight=10      # Larger leaves
reg_alpha=0.5            # 2.5x more L1
reg_lambda=3.0           # 1.5x more L2
max_delta_step=1         # Limit weight updates
```

### 3. Neural Network Changes

**Current:**
```python
hidden_dims=[256, 128, 64]
dropout=0.4
learning_rate=0.005
weight_decay=0.001
```

**New (AGGRESSIVE):**
```python
hidden_dims=[128, 64, 32]  # Smaller network
dropout=0.5                 # More dropout
learning_rate=0.001         # Slower learning
weight_decay=0.01           # 10x more L2
label_smoothing=0.2         # Reduce overconfidence
```

### 4. LSTM Changes (CRITICAL)

**Current:**
```python
hidden_dim=64
num_layers=1
dropout=0.5
learning_rate=0.0005
weight_decay=0.05
```

**New (AGGRESSIVE):**
```python
hidden_dim=32              # Smaller LSTM
num_layers=1               # Keep simple
dropout=0.6                # More dropout
learning_rate=0.0001       # 5x slower
weight_decay=0.1           # 2x more L2
max_grad_norm=0.5          # Stricter gradient clipping
```

## Expected Improvements

### Target Metrics
- Train-Val gap: < 15% (currently 25-69%)
- Validation accuracy: Stable across folds
- No exploding gradients
- Smoother learning curves

### Trade-offs
- Training accuracy will decrease (60-80% range)
- Validation accuracy should improve or stay stable
- Better generalization to test set
- Longer training time (slower learning rates)

## Implementation Plan

1. Update `models/random_forest_model.py` - default parameters
2. Update `models/xgboost_model.py` - default parameters
3. Update `models/neural_network_model.py` - architecture and training
4. Update `models/lstm_model.py` - architecture and training
5. Test on one symbol (EURUSD) to verify improvements
6. Run full training on all symbols

## Verification Criteria

✅ Train-Val gap < 20% (acceptable)
✅ Train-Val gap < 15% (good)
✅ No exploding gradients in LSTM
✅ XGBoost train accuracy < 95%
✅ Validation accuracy stable across epochs
✅ Test accuracy within 5% of validation accuracy

## Files to Modify

- `models/random_forest_model.py` - Line ~50-60 (default params)
- `models/xgboost_model.py` - Line ~50-70 (default params)
- `models/neural_network_model.py` - Line ~80-100 (architecture), ~150-160 (training)
- `models/lstm_model.py` - Line ~80-100 (architecture), ~150-170 (training)
