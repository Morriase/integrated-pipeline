# Cross-Validation Stability Metrics - Quick Reference

## Overview
Enhanced CV system that identifies unstable models through comprehensive stability metrics.

## Key Thresholds

| Std Dev | Status | Flag | Action |
|---------|--------|------|--------|
| ≤ 0.10 | ✅ STABLE | `is_unstable=False` | Safe to deploy |
| 0.10 - 0.15 | ⚠️ UNSTABLE | `is_unstable=True` | Use with caution |
| > 0.15 | ❌ REJECTED | `is_rejected=True` | DO NOT DEPLOY |

## New Metrics

```python
cv_results = model.cross_validate(X, y, n_folds=5)

# Stability metrics
cv_results['std_accuracy']      # Standard deviation of fold accuracies
cv_results['min_accuracy']      # Minimum fold accuracy
cv_results['max_accuracy']      # Maximum fold accuracy
cv_results['is_unstable']       # True if std > 0.10
cv_results['is_rejected']       # True if std > 0.15
cv_results['poor_folds']        # List of underperforming folds
```

## Usage Example

```python
# Run CV
cv_results = model.cross_validate(X_train, y_train, n_folds=5)

# Check stability
if cv_results['is_rejected']:
    print("❌ Model rejected - too unstable")
elif cv_results['is_unstable']:
    print("⚠️ Model unstable - investigate poor folds")
    for fold in cv_results['poor_folds']:
        print(f"  Fold {fold['fold']}: {fold['accuracy']:.3f}")
else:
    print("✅ Model is stable")
```

## Poor Fold Structure

```python
{
    'fold': 3,                    # Fold number (1-indexed)
    'accuracy': 0.520,            # Fold accuracy
    'deviation': -0.130           # Deviation from mean
}
```

## Console Output Examples

### Stable Model
```
Cross-Validation Results:
  Mean Accuracy: 0.720 ± 0.045
  Min Accuracy:  0.680
  Max Accuracy:  0.760
  Range:         0.080
  ✅ STABLE: Std dev 0.045 ≤ 0.10
```

### Unstable Model
```
Cross-Validation Results:
  Mean Accuracy: 0.650 ± 0.120
  Min Accuracy:  0.520
  Max Accuracy:  0.780
  Range:         0.260
  ⚠️ UNSTABLE: Std dev 0.120 > 0.10
  Model shows high variance - use with caution

  Poor-Performing Folds (below mean - std):
    Fold 2: 0.520 (deviation: -0.130)
```

### Rejected Model
```
Cross-Validation Results:
  Mean Accuracy: 0.600 ± 0.180
  ❌ MODEL REJECTED: Std dev 0.180 > 0.15
  Model is highly unstable - DO NOT DEPLOY
```

## Integration with Model Selection

```python
# In ModelSelector
for model_name, metrics in symbol_results.items():
    cv_results = metrics.get('cv_results', {})
    
    # Reject unstable models
    if cv_results.get('is_rejected', False):
        continue  # Skip this model
    
    # Penalize unstable models
    if cv_results.get('is_unstable', False):
        score -= 0.1
```

## Troubleshooting

### High Variance (std > 0.10)
**Possible Causes:**
- Insufficient training data
- Data quality issues in specific folds
- Model too complex for dataset size
- Class imbalance in folds

**Solutions:**
1. Check poor folds for data issues
2. Increase regularization
3. Simplify model architecture
4. Use stratified splits (already default)
5. Increase training data

### Poor-Performing Folds
**Investigation Steps:**
1. Check fold data distribution
2. Look for outliers or anomalies
3. Verify feature quality in that fold
4. Check for temporal dependencies

## Requirements Satisfied

- ✅ 6.1: Calculate std dev of fold accuracies
- ✅ 6.2: Flag if std > 0.10
- ✅ 6.3: Reject if std > 0.15
- ✅ 6.4: Report min, max, mean, std
- ✅ 6.5: Identify poor-performing folds

## Testing

Run tests:
```bash
python test_cv_stability.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```
