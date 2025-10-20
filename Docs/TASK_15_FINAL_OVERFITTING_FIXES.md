# Task 15: Final Overfitting Fixes + LSTM Removal

## Changes Being Made:

### 1. Feature Selection (57 → 25 features)
Based on RandomForest importance analysis, keeping only top features:
- TBM_Bars_to_Hit, TBM_Risk_Per_Trade_ATR, TBM_Reward_Per_Trade_ATR
- Distance_to_Entry_ATR, OB_Age
- FVG_Distance_to_Price_ATR, FVG_Depth_ATR, FVG_Quality_Fuzzy, FVG_Size_Fuzzy_Score
- ATR_ZScore, BOS_Dist_ATR_ZScore, FVG_Distance_to_Price_ATR_ZScore, FVG_Depth_ATR_ZScore
- ChoCH_Detected, ChoCH_Direction, BOS_Commitment_Flag, BOS_Close_Confirm, BOS_Wick_Confirm
- OB_Bullish_Valid, OB_Bearish_Valid, FVG_Bullish_Valid, FVG_Bearish_Valid
- Trend_Bias_Indicator, Trend_Strength, atr

### 2. XGBoost Aggressive Regularization
```python
params = {
    'max_depth': 3,              # Was: 6
    'min_child_weight': 10,      # Was: 3
    'subsample': 0.6,            # Was: 0.8
    'colsample_bytree': 0.6,     # Was: 0.8
    'learning_rate': 0.01,       # Was: 0.1
    'n_estimators': 200,
    'early_stopping_rounds': 20  # NEW!
}
```

### 3. Neural Network Simplification
```python
# OLD: 57 -> 512 -> 256 -> 128 -> 64 -> 3
# NEW: 25 -> 128 -> 64 -> 3
layers = [
    nn.Linear(input_size, 128),
    nn.Dropout(0.5),  # Was: 0.3
    nn.Linear(128, 64),
    nn.Dropout(0.5),  # Was: 0.3
    nn.Linear(64, num_classes)
]
weight_decay = 0.01  # NEW L2 regularization
```

### 4. LSTM Removal
- Remove from train_all_models.py
- Remove from ensemble_model.py
- Keep lstm_model.py for reference but don't use it

### 5. Ensemble Weights Adjustment
```python
# OLD: Equal weights or dynamic
# NEW: Favor RandomForest
weights = {
    'RandomForest': 0.5,
    'XGBoost': 0.3,
    'NeuralNetwork': 0.2
}
```

## Expected Improvements:
- Train-val gap: < 15% (from 20-60%)
- Validation accuracy: > 65% (from 45-78%)
- Test accuracy: > 60% (from 50-81%)
- Zero exploding gradient warnings
- Faster training (no LSTM)
