# Train All Models Orchestrator - Quick Reference

## Overview
The `train_all_models.py` orchestrator trains all SMC models with automatic model selection, comprehensive error handling, and enhanced reporting.

## Key Features

### 🎯 Automatic Model Selection
- Filters models by quality criteria
- Selects best model per symbol
- Generates deployment manifest

### 🛡️ Comprehensive Error Handling
- Continues training on individual failures
- Logs detailed error information
- Graceful degradation of components

### 📊 Enhanced Reporting
- Inline warnings in performance tables
- Model selection results
- Aggregated warnings summary
- Complete JSON output

## Usage

```bash
# Run complete training pipeline
python train_all_models.py
```

## Model Selection Criteria

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| Train-Val Gap | ≤ 20% | Prevent overfitting |
| Test Accuracy | ≥ 55% | Ensure minimum quality |
| Val-Test Diff | ≤ 5% | Ensure stability |

**Scoring Formula:** `test_accuracy - (train_val_gap * 0.5)`

## Output Files

### 1. training_results.json
Complete training results with selections and warnings:
```json
{
  "training_results": {...},
  "model_selections": {...},
  "warnings": [...],
  "timestamp": "..."
}
```

### 2. deployment_manifest.json
Model selection manifest for deployment:
```json
{
  "timestamp": "...",
  "selection_criteria": {...},
  "selections": {
    "EURUSD": {
      "selected_model": "RandomForest",
      "test_accuracy": 0.623,
      "train_val_gap": 0.12,
      "reason": "...",
      "alternatives": [...]
    }
  },
  "summary": {...}
}
```

### 3. overfitting_report.md
Detailed overfitting analysis with visualizations

### 4. overfitting_analysis.png
Visual charts of overfitting metrics

## Warning Types

### 🔴 Training Failures
- **Cause:** Model training crashed
- **Action:** Check error logs and traceback
- **Impact:** Model unavailable for selection

### 🟠 Overfitting Detected
- **Cause:** Train-val gap > 15%
- **Action:** Apply stronger regularization
- **Impact:** Model may not generalize well

### 🟡 CV Instability
- **Cause:** High variance across CV folds
- **Action:** Collect more data or simplify model
- **Impact:** Unreliable performance estimates

### 🟢 Low Test Accuracy
- **Cause:** Test accuracy < 55%
- **Action:** Improve features or model architecture
- **Impact:** Model below quality threshold

## Error Handling Flow

```
Train Symbol
├── Train RandomForest
│   ├── Success → Store results + duration
│   └── Failure → Log error, continue
├── Train XGBoost
│   ├── Success → Store results + duration
│   └── Failure → Log error, continue
├── Train NeuralNetwork
│   ├── Success → Store results + duration
│   └── Failure → Log error, continue
└── Train LSTM
    ├── Success → Store results + duration
    └── Failure → Log error, continue

Select Best Models
├── Success → Generate manifest
└── Failure → Log error, continue

Generate Reports
├── Overfitting Report
│   ├── Success → Save report
│   └── Failure → Log warning, continue
└── Summary Report
    ├── Success → Save results
    └── Failure → Log error
```

## Model Selection Logic

### Step 1: Filter Models
```python
# Reject if:
- train_val_gap > 0.20  # Too much overfitting
- test_accuracy < 0.55  # Too low accuracy
- val_test_diff > 0.05  # Unstable performance
```

### Step 2: Score Candidates
```python
score = test_accuracy - (train_val_gap * 0.5)
```

### Step 3: Select Best
```python
selected_model = max(candidates, key=lambda x: x['score'])
```

### Step 4: Handle No Candidates
```python
if no_candidates:
    flag_for_manual_review()
```

## Summary Report Structure

```
TRAINING SUMMARY REPORT
├── Per-Symbol Performance
│   ├── Model name
│   ├── CV Mean±Std (if available)
│   ├── Val/Test accuracy
│   ├── Stability status
│   └── Warnings (inline)
│
├── MODEL SELECTION RESULTS
│   ├── Selected model per symbol
│   ├── Test accuracy & gap
│   ├── Selection reason
│   └── Alternative models
│
├── WARNINGS SUMMARY
│   ├── Total warning count
│   └── Detailed warning list
│
└── Final Statistics
    ├── Total symbols/models
    ├── Success/failure counts
    ├── Models selected
    └── Training duration
```

## Interpreting Results

### ✅ Healthy Model
```
Model: RandomForest
CV: 0.6234±0.0123
Val: 0.625  Test: 0.623
Stability: ✅ Stable
Warnings: None
```

### ⚠️ Problematic Model
```
Model: NeuralNetwork
CV: 0.7123±0.1567
Val: 0.712  Test: 0.598
Stability: ⚠️ Unstable
Warnings: Overfitting (gap=23%); Unstable CV (std=0.157)
```

### ❌ Failed Model
```
Model: LSTM
Status: ERROR
Warnings: Training failed: RuntimeError
```

## Customizing Selection Criteria

Edit the main execution block:

```python
selector = ModelSelector(
    max_gap=0.15,           # Stricter: 15% max gap
    min_accuracy=0.60,      # Higher: 60% min accuracy
    stability_threshold=0.03  # Tighter: 3% max diff
)
```

## Troubleshooting

### No Models Selected for Symbol
**Symptom:** `selected_model: None` in manifest

**Causes:**
1. All models have high train-val gap (> 20%)
2. All models have low test accuracy (< 55%)
3. All models show val-test inconsistency (> 5%)

**Solutions:**
1. Review overfitting report
2. Apply anti-overfitting techniques
3. Collect more training data
4. Adjust selection criteria if too strict

### High Failure Rate
**Symptom:** Many models show ERROR status

**Causes:**
1. Missing dependencies (XGBoost, PyTorch)
2. Insufficient training data
3. Data quality issues
4. Memory constraints

**Solutions:**
1. Check error logs for specific issues
2. Verify data availability and quality
3. Install missing dependencies
4. Reduce model complexity or batch size

### Inconsistent Performance
**Symptom:** High CV std, unstable models

**Causes:**
1. Small dataset size
2. Imbalanced classes
3. High feature noise
4. Overly complex models

**Solutions:**
1. Collect more data
2. Apply data augmentation
3. Feature selection/engineering
4. Simplify model architecture

## Best Practices

1. **Review Warnings:** Always check warnings summary
2. **Validate Selections:** Review deployment manifest before production
3. **Monitor Trends:** Track overfitting metrics over time
4. **Iterate Criteria:** Adjust thresholds based on results
5. **Document Decisions:** Keep notes on manual reviews

## Integration with Pipeline

```bash
# Complete workflow
python run_complete_pipeline.py  # Prepare data
python train_all_models.py       # Train & select models
python ensemble_model.py          # Create ensemble (if needed)
python backtest.py                # Validate on test set
```

## Quick Checks

```bash
# Check if training completed
ls models/trained/training_results.json

# Check selected models
cat models/trained/deployment_manifest.json | grep selected_model

# Count warnings
cat models/trained/training_results.json | grep -c "warning"

# View overfitting summary
head -n 20 models/trained/overfitting_report.md
```

## Status Indicators

| Symbol | Meaning |
|--------|---------|
| ✅ | Healthy, no issues |
| ⚠️ | Warning, review recommended |
| ❌ | Error, action required |
| 🎯 | Selected for deployment |
| 🔄 | Manual review needed |

## Next Steps After Training

1. ✅ Review `training_results.json`
2. ✅ Check `deployment_manifest.json`
3. ✅ Read `overfitting_report.md`
4. ⚠️ Address flagged warnings
5. 🎯 Deploy selected models
6. 📊 Monitor production performance
