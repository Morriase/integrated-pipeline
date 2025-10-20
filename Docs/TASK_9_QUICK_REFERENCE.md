# Task 9 Quick Reference: Cross-Validation Workflow

## Quick Start

### Train a Model with Cross-Validation

```python
from train_all_models import SMCModelTrainer

# Initialize trainer
trainer = SMCModelTrainer(data_dir='Data', output_dir='models/trained')

# Train with CV
result = trainer.train_with_cross_validation(
    symbol='EURUSD',
    model_type='RandomForest',  # or 'XGBoost', 'NeuralNetwork', 'LSTM'
    exclude_timeout=False
)
```

## Understanding the Output

### Console Output

```
🔄 Performing 5-fold stratified cross-validation...

📈 Cross-Validation Results:
  Mean Accuracy: 0.7500
  Std Accuracy:  0.0800
  Fold Accuracies: ['0.7200', '0.7600', '0.7800', '0.7400', '0.7500']

✅ Model shows stable performance across folds (std < 0.15)
```

### Stability Indicators

- **✅ Stable (std < 0.15):** Model performs consistently across folds
- **⚠️ Unstable (std > 0.15):** High variance detected, needs attention

### Warning for Unstable Models

```
⚠️  WARNING: Model shows high variance (std > 0.15)
  This indicates unstable performance across folds.
  Consider:
    - Collecting more training data
    - Simplifying model architecture
    - Applying stronger regularization
```

## Accessing Results

### CV Results Structure

```python
result['cv_results'] = {
    'mean_accuracy': 0.75,
    'std_accuracy': 0.08,
    'fold_accuracies': [0.72, 0.76, 0.78, 0.74, 0.75],
    'is_stable': True,
    'n_folds': 5
}
```

### Training History (includes CV)

```python
result['history'] = {
    'train_accuracy': 0.85,
    'val_accuracy': 0.78,
    'train_val_gap': 0.07,
    'cv_mean_accuracy': 0.75,      # Added
    'cv_std_accuracy': 0.08,        # Added
    'cv_fold_accuracies': [...],    # Added
    'cv_is_stable': True            # Added
}
```

## Summary Report

### Generate Report

```python
# Store results
trainer.results['EURUSD'] = {'RandomForest': result}

# Generate summary
trainer.generate_summary_report()
```

### Report Format

```
📊 EURUSD:
  Model                CV Mean±Std          Val Acc    Test Acc   Stability
  ---------------------------------------------------------------------------
  RandomForest         0.7500±0.0800        0.780      0.760      ✅ Stable
  NeuralNetwork        0.7000±0.1800        0.780      0.760      ⚠️ Unstable
```

## Interpreting Stability

### Stable Model (std < 0.15)
- **Meaning:** Consistent performance across different data splits
- **Action:** Safe to deploy, model generalizes well
- **Example:** std = 0.08 → Very stable

### Unstable Model (std > 0.15)
- **Meaning:** Performance varies significantly across folds
- **Action:** Investigate and improve before deployment
- **Recommendations:**
  1. Collect more training data
  2. Simplify model (reduce complexity)
  3. Apply stronger regularization
  4. Check for data quality issues

### Borderline (std ≈ 0.15)
- **Meaning:** Moderate variance, acceptable but monitor
- **Action:** Consider improvements if possible

## Model-Specific Parameters

### Random Forest (Anti-Overfitting)
```python
max_depth=15              # Reduced from 20
min_samples_split=20      # Increased from 10
min_samples_leaf=10       # Increased from 5
```

### Neural Network (Anti-Overfitting)
```python
hidden_dims=[256, 128, 64]  # Reduced from [512, 256, 128, 64]
dropout=0.5                  # Increased from 0.4
learning_rate=0.005          # Reduced from 0.01
batch_size=64                # Increased from 32
patience=20                  # Increased from 15
weight_decay=0.1             # Increased from 0.01
```

## Common Use Cases

### 1. Train All Models with CV

```python
trainer = SMCModelTrainer(data_dir='Data', output_dir='models/trained')
symbols = trainer.get_available_symbols()

for symbol in symbols:
    for model_type in ['RandomForest', 'NeuralNetwork']:
        result = trainer.train_with_cross_validation(
            symbol=symbol,
            model_type=model_type
        )
        
        # Store results
        if symbol not in trainer.results:
            trainer.results[symbol] = {}
        trainer.results[symbol][model_type] = result

# Generate comprehensive report
trainer.generate_summary_report()
```

### 2. Check Model Stability

```python
result = trainer.train_with_cross_validation(
    symbol='EURUSD',
    model_type='RandomForest'
)

if not result['cv_results']['is_stable']:
    print("⚠️ Model is unstable!")
    print(f"Std: {result['cv_results']['std_accuracy']:.4f}")
    print("Consider retraining with different parameters")
```

### 3. Compare CV vs Test Performance

```python
cv_acc = result['cv_results']['mean_accuracy']
test_acc = result['test_metrics']['accuracy']
gap = abs(cv_acc - test_acc)

print(f"CV Accuracy:   {cv_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Gap:           {gap:.4f}")

if gap > 0.05:
    print("⚠️ Significant gap between CV and test performance")
```

## Troubleshooting

### High Variance (std > 0.15)

**Possible Causes:**
- Insufficient training data
- Model too complex
- Data quality issues
- Class imbalance

**Solutions:**
1. Increase training data size
2. Apply data augmentation
3. Reduce model complexity
4. Use stronger regularization
5. Check for outliers or errors in data

### Low CV Accuracy

**Possible Causes:**
- Model underfitting
- Poor feature engineering
- Inappropriate hyperparameters

**Solutions:**
1. Increase model complexity
2. Add more features
3. Tune hyperparameters
4. Try different model architecture

### CV Accuracy >> Test Accuracy

**Possible Causes:**
- Data leakage
- Distribution shift
- Overfitting to training distribution

**Solutions:**
1. Check for data leakage
2. Verify train/test split
3. Apply stronger regularization
4. Use more diverse training data

## Best Practices

1. **Always use CV** for model evaluation before deployment
2. **Monitor stability** - flag models with std > 0.15
3. **Compare CV vs test** - large gaps indicate issues
4. **Document results** - save CV metrics with models
5. **Iterate** - use CV feedback to improve models

## Integration with Existing Workflow

The CV workflow integrates seamlessly with:
- ✅ Existing training methods
- ✅ Overfitting report generation
- ✅ Model saving and loading
- ✅ Feature importance analysis
- ✅ Evaluation metrics

## Performance Notes

- **CV Time:** ~5x single training time (5 folds)
- **Memory:** Same as single training
- **Recommended:** Use for final model evaluation
- **Optional:** Can skip CV for quick experiments

## Related Documentation

- `TASK_9_IMPLEMENTATION_SUMMARY.md` - Full implementation details
- `.kiro/specs/anti-overfitting-enhancement/requirements.md` - Requirements
- `.kiro/specs/anti-overfitting-enhancement/design.md` - Design document
- `test_cv_integration_unit.py` - Unit tests and examples
