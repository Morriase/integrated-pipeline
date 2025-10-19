# Model Selector - Quick Reference Guide

## What It Does
Automatically selects the best-performing model per symbol based on quality criteria, filtering out overfitting and low-accuracy models.

## Quick Start

```python
from train_all_models import ModelSelector

# Initialize with default thresholds
selector = ModelSelector()

# Or customize thresholds
selector = ModelSelector(
    max_gap=0.15,           # Stricter: 15% max gap
    min_accuracy=0.60,      # Higher bar: 60% min accuracy
    stability_threshold=0.03 # Tighter: 3% max val-test diff
)

# Select best models from training results
selections = selector.select_best_models(results)

# Save deployment manifest
manifest = selector.save_deployment_manifest(
    selections,
    'models/trained/deployment_manifest.json'
)
```

## Default Thresholds

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `max_gap` | 20% | Maximum train-val gap (overfitting limit) |
| `min_accuracy` | 55% | Minimum test accuracy (quality floor) |
| `stability_threshold` | 5% | Maximum val-test difference (consistency check) |

## Scoring Formula

```
score = test_accuracy - (train_val_gap × 0.5)
```

**Example:**
- Model A: 75% accuracy, 10% gap → score = 0.75 - 0.05 = 0.70
- Model B: 70% accuracy, 5% gap → score = 0.70 - 0.025 = 0.675
- **Winner:** Model A (higher score)

## Selection Logic

1. **Filter** models by quality criteria
2. **Score** remaining candidates
3. **Select** highest-scoring model
4. **Identify** alternatives
5. **Flag** symbols with no qualifying models

## Output Structure

### Successful Selection
```json
{
  "selected_model": "XGBoost",
  "test_accuracy": 0.72,
  "train_val_gap": 0.13,
  "score": 0.655,
  "reason": "Best score (0.655) with gap 13.0%",
  "alternatives": ["NeuralNetwork"]
}
```

### Manual Review Required
```json
{
  "selected_model": null,
  "reason": "No models met quality criteria",
  "action": "MANUAL_REVIEW_REQUIRED",
  "rejected_models": [...]
}
```

## Common Rejection Reasons

| Reason | Cause | Solution |
|--------|-------|----------|
| Train-val gap too high | Overfitting | Increase regularization, reduce model complexity |
| Test accuracy too low | Poor performance | Collect more data, improve features |
| Val-test inconsistency | Unstable model | Use cross-validation, check data distribution |
| Training error | Failed training | Check logs, fix data issues |

## Deployment Manifest

The manifest includes:
- **Timestamp**: When selection was performed
- **Criteria**: Thresholds used for selection
- **Selections**: Per-symbol results
- **Summary**: Overall statistics

### Summary Statistics
```json
{
  "total_symbols": 11,
  "models_selected": 9,
  "manual_review_needed": 2,
  "selected_model_types": {
    "XGBoost": 5,
    "NeuralNetwork": 4
  }
}
```

## Integration Example

```python
# Complete workflow
trainer = SMCModelTrainer()
selector = ModelSelector()

# Train all models
results = {}
for symbol in symbols:
    results[symbol] = trainer.train_all_for_symbol(symbol)

# Select best models
selections = selector.select_best_models(results)

# Save manifest
manifest = selector.save_deployment_manifest(
    selections,
    'deployment_manifest.json'
)

# Deploy selected models
for symbol, selection in selections.items():
    if selection['selected_model']:
        model_name = selection['selected_model']
        print(f"✅ Deploy {model_name} for {symbol}")
    else:
        print(f"⚠️  Manual review needed for {symbol}")
```

## Customization Tips

### Stricter Quality Control
```python
selector = ModelSelector(
    max_gap=0.10,      # Only 10% gap allowed
    min_accuracy=0.70, # 70% minimum accuracy
    stability_threshold=0.02  # 2% max difference
)
```

### More Lenient (Development)
```python
selector = ModelSelector(
    max_gap=0.30,      # Allow 30% gap
    min_accuracy=0.50, # 50% minimum accuracy
    stability_threshold=0.10  # 10% max difference
)
```

### Production Recommended
```python
selector = ModelSelector(
    max_gap=0.15,      # 15% gap limit
    min_accuracy=0.65, # 65% minimum accuracy
    stability_threshold=0.05  # 5% max difference
)
```

## Troubleshooting

### No Models Selected for Any Symbol
**Problem:** All models rejected across all symbols

**Solutions:**
1. Lower thresholds temporarily
2. Check training data quality
3. Review model hyperparameters
4. Increase training data size

### Too Many Manual Reviews
**Problem:** Many symbols flagged for manual review

**Solutions:**
1. Adjust thresholds based on data characteristics
2. Improve data augmentation
3. Tune model regularization
4. Collect more training data

### Same Model Always Selected
**Problem:** One model type dominates selections

**Solutions:**
1. Check if other models are training correctly
2. Review hyperparameters for diversity
3. Consider ensemble approaches
4. Analyze per-symbol characteristics

## Testing

Run unit tests:
```bash
python test_model_selector.py
```

Expected output:
```
✅ ALL TESTS PASSED
```

## Files

| File | Purpose |
|------|---------|
| `train_all_models.py` | Contains ModelSelector class |
| `test_model_selector.py` | Unit tests |
| `deployment_manifest.json` | Output manifest |
| `TASK_7_MODEL_SELECTOR_SUMMARY.md` | Detailed documentation |

## Requirements Satisfied

- ✅ 5.1: Rank models by test accuracy
- ✅ 5.2: Exclude models with gap >20%
- ✅ 5.3: Exclude models with accuracy <55%
- ✅ 5.4: Prefer val-test consistency within 5%
- ✅ 5.5: Flag symbols for manual review
- ✅ 5.6: Save deployment manifest JSON

## Next Steps

After model selection:
1. Review manifest summary
2. Investigate manual review cases
3. Deploy selected models
4. Monitor production performance
5. Iterate on thresholds based on results
