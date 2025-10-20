# Testing Phase 1 Fixes

## Quick Test Commands

### 1. Test Core Models (Recommended)
```bash
python train_all_models.py
```

**Expected Output Changes:**

#### Feature Selection:
```
🔍 Feature Selection: Analyzing 102 features...
  Computing Random Forest importance...
  Computing mutual information...
  After importance filtering: 45 features (threshold: 0.0234)
  After correlation filtering: 38 features (removed 7 correlated)
  ✅ Final selection: 38 features (37.3% of original)
```

#### Class Weights:
```
# Neural Network / LSTM will show:
  Calculated class weights: [1.2, 0.8, 1.0]
```

#### Trading Metrics:
```
📊 Evaluating on Test set...

  Classification Metrics:
    Accuracy: 0.642
    Precision (macro): 0.618
    Recall (macro): 0.635
    F1-Score (macro): 0.626

  Trading Metrics (1:2 R:R):
    Win Rate: 55.2% (138/250 trades)
    Profit Factor: 1.52
    Expected Value/Trade: 0.20R
    Trade Accuracy: 62.4%
    Timeouts: 45 (15.3%)
    💰 PROFITABLE STRATEGY (EV > 0)
```

---

### 2. Test with LSTM (Experimental)
```bash
python train_all_models.py --include-lstm
```

**Additional Output:**
```
⚠️  LSTM ENABLED (Experimental):
   - May show overfitting and gradient issues
   - Included for research/comparison purposes
   - Not recommended for production use

🔄 Starting LSTM training (EXPERIMENTAL)...
⚠️  WARNING: LSTM has shown instability in previous tests:
   - Severe overfitting (39-61% train-val gap)
   - Gradient explosions (24 warnings)
   - Poor test accuracy (16-46%)
   - Training divergence
```

---

## Validation Checklist

After running training, verify:

### ✅ Feature Selection Working
- [ ] Console shows "Feature Selection: Analyzing X features..."
- [ ] Final feature count is reduced (e.g., 102 → 38)
- [ ] Training uses selected features only

### ✅ Class Weights Applied
- [ ] Neural Network shows class weights calculation
- [ ] LSTM shows class weights calculation
- [ ] Confusion matrix is more balanced

### ✅ Trading Metrics Displayed
- [ ] "Trading Metrics" section appears in evaluation
- [ ] Win Rate, Profit Factor, EV/Trade shown
- [ ] Profitability assessment (💰 or ⚠️) displayed

### ✅ LSTM Properly Labeled
- [ ] LSTM warnings appear if enabled
- [ ] Training monitor tracks gradient issues
- [ ] Experimental status clearly communicated

---

## Expected Performance Improvements

### Before Fixes:
- Training on 100+ features
- Overfitting common (train-val gap >20%)
- No visibility into trading viability
- Class imbalance causing bias

### After Fixes:
- Training on 30-50 selected features
- Reduced overfitting (train-val gap <15%)
- Clear trading metrics and profitability
- Balanced predictions across classes

---

## Troubleshooting

### If feature selection fails:
```python
# Check if feature_selector is being created
# Should see in console: "🔍 Feature Selection: Analyzing..."
```

### If class weights not applied:
```python
# Check PyTorch models show:
# "Calculated class weights: [...]"
```

### If trading metrics missing:
```python
# Check evaluate() output includes:
# "Trading Metrics (1:2 R:R):"
```

---

## Success Criteria

Training is successful if:
1. ✅ All 3 core models train without errors
2. ✅ Feature selection reduces dimensionality
3. ✅ Trading metrics show EV > 0 for at least 1 model
4. ✅ No critical warnings (NaN loss, severe overfitting)
5. ✅ Models saved to `models/trained/`

---

## Next Steps After Successful Test

1. Review `models/trained/training_results.json`
2. Compare trading metrics across models
3. Select best model based on:
   - Highest EV/Trade
   - Best Profit Factor
   - Acceptable Win Rate (>50%)
4. Proceed to Phase 2 improvements
