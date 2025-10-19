# Performance Reporting System - Quick Reference

## Overview

The performance reporting system generates comprehensive training reports for each symbol, including model comparisons, overfitting analysis, cross-validation stability, feature importance, confusion matrices, and deployment recommendations.

## Usage

### Automatic Generation

Reports are automatically generated when running the training orchestrator:

```python
python train_all_models.py
```

### Manual Generation

```python
from train_all_models import SMCModelTrainer, ModelSelector

# Initialize trainer
trainer = SMCModelTrainer()

# Train models (results stored in trainer.results)
trainer.train_all_for_symbol('EURUSD')

# Optional: Select best models
selector = ModelSelector()
model_selections = selector.select_best_models(trainer.results)

# Generate reports
trainer.generate_summary_report(model_selections=model_selections)
```

## Output Files

### 1. Console Summary
Real-time output showing:
- Model performance table per symbol
- Cross-validation results
- Model selection results
- Warnings summary

### 2. Per-Symbol Markdown Reports
**Location:** `models/trained/reports/{SYMBOL}_training_report.md`

Each report contains:
- 🎯 Deployment Recommendation
- 📊 Model Performance Comparison
- 🔍 Overfitting Analysis
- 📈 Cross-Validation Stability
- 🎯 Feature Importance (Top 10)
- 📉 Confusion Matrices
- ⚠️ Warnings and Issues

### 3. Training Results JSON
**Location:** `models/trained/training_results.json`

Contains:
- Complete training results for all models
- Model selection results
- All warnings
- Timestamp

## Report Sections Explained

### 🎯 Deployment Recommendation (Requirement 8.7)

Shows the best model for deployment with:
- Test accuracy and train-val gap
- Selection score
- Justification
- Alternative models
- Manual review flag if no model meets criteria

**Example:**
```markdown
**Recommended Model:** XGBoost

- **Test Accuracy:** 0.700 (70.0%)
- **Train-Val Gap:** 13.0%
- **Selection Score:** 0.635
- **Reason:** Best score with acceptable gap
```

### 📊 Model Performance Comparison (Requirements 8.2, 8.3, 8.4)

Comprehensive table showing:
- Train/Val/Test accuracy
- Train-Val gap (overfitting indicator)
- CV Mean±Std (cross-validation results)
- Stability flag (✅ Stable / ⚠️ Unstable)
- Status (✅ / ⚠️ Overfit / ⚠️ Low Acc / ⚠️ Unstable)

**Example:**
```markdown
| Model | Train Acc | Val Acc | Test Acc | Train-Val Gap | CV Mean±Std | CV Stable | Status |
|-------|-----------|---------|----------|---------------|-------------|-----------|--------|
| XGBoost | 0.850 | 0.720 | 0.700 | 13.0% | 0.720±0.040 | ✅ | ✅ |
```

### 🔍 Overfitting Analysis (Requirement 8.3)

Detailed analysis of models with train-val gap > 15%:
- Severity classification:
  - 🔴 Severe: gap > 25%
  - 🟡 Moderate: gap > 20%
  - 🟢 Mild: gap > 15%
- Train and validation accuracy
- Specific recommendations

**Example:**
```markdown
### XGBoost

- **Severity:** 🟡 Moderate
- **Train Accuracy:** 0.950
- **Val Accuracy:** 0.720
- **Gap:** 23.0%
- **Recommendation:** Increase regularization or add more training data
```

### 📈 Cross-Validation Stability (Requirement 8.4)

Shows cross-validation results:
- Mean accuracy and standard deviation
- Stability assessment (stable if std < 0.10)
- Individual fold accuracies
- Min/Max fold performance
- Instability warnings

**Example:**
```markdown
### XGBoost

- **Mean Accuracy:** 0.7200
- **Std Deviation:** 0.0400
- **Stability:** ✅ Stable
- **Fold Accuracies:** 0.700, 0.720, 0.710, 0.740, 0.730
- **Min/Max:** 0.700 / 0.740
```

### 🎯 Feature Importance (Top 10) (Requirement 8.5)

Top 10 most important features for tree-based models:
- Ranked by importance
- Feature name and importance value
- Only available for RandomForest and XGBoost

**Example:**
```markdown
### XGBoost

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | OB_Age | 0.1200 |
| 2 | TBM_Risk_Per_Trade_ATR | 0.0800 |
| 3 | FVG_Volume_Ratio | 0.0650 |
```

### 📉 Confusion Matrices (Requirement 8.6)

Visual representation of predictions:
- ASCII confusion matrix
- Calculated precision, recall, F1-score
- Available for all model types

**Example:**
```markdown
### XGBoost

```
              Predicted
              0    1
Actual  0     14    6
        1      7   19
```

- **Precision:** 0.760
- **Recall:** 0.731
- **F1-Score:** 0.745
```

### ⚠️ Warnings and Issues

Comprehensive list of issues:
- Training failures
- High overfitting (gap > 20%)
- Low test accuracy (< 55%)
- Unstable cross-validation (std > 0.10)
- Training warnings from monitoring system

**Example:**
```markdown
### XGBoost

- ⚠️ High overfitting (gap=23.0%)
- ⚠️ Unstable cross-validation (std=0.120)
```

## Interpreting Results

### Good Model Indicators
- ✅ Test accuracy > 65%
- ✅ Train-val gap < 15%
- ✅ CV std < 0.10 (stable)
- ✅ Val-test difference < 5%

### Warning Signs
- ⚠️ Train-val gap > 20% (overfitting)
- ⚠️ Test accuracy < 55% (poor performance)
- ⚠️ CV std > 0.10 (unstable)
- ⚠️ Val-test difference > 5% (inconsistent)

### Action Required
- 🔴 Gap > 25%: Increase regularization significantly
- 🟡 Gap > 20%: Increase regularization or add data
- 🟢 Gap > 15%: Minor tuning recommended
- ⚠️ No model selected: Manual review required

## Model Selection Criteria

The ModelSelector uses these thresholds:
- **Max Train-Val Gap:** 20%
- **Min Test Accuracy:** 55%
- **Max Val-Test Diff:** 5%

Models are scored as: `test_accuracy - (train_val_gap * 0.5)`

This penalizes overfitting while rewarding high accuracy.

## Example Workflow

```python
# 1. Train models
trainer = SMCModelTrainer()
symbols = trainer.get_available_symbols()

for symbol in symbols:
    trainer.train_all_for_symbol(symbol)

# 2. Select best models
selector = ModelSelector(
    max_gap=0.20,           # 20% max train-val gap
    min_accuracy=0.55,      # 55% min test accuracy
    stability_threshold=0.05 # 5% max val-test diff
)
model_selections = selector.select_best_models(trainer.results)

# 3. Save deployment manifest
selector.save_deployment_manifest(
    model_selections, 
    'models/trained/deployment_manifest.json'
)

# 4. Generate comprehensive reports
trainer.generate_summary_report(model_selections=model_selections)
```

## Output Locations

```
models/trained/
├── reports/
│   ├── EURUSD_training_report.md
│   ├── GBPUSD_training_report.md
│   └── ...
├── training_results.json
└── deployment_manifest.json
```

## Testing

Run the test script to verify reporting functionality:

```bash
# Simple test
python test_summary_report_simple.py

# Comprehensive test
python test_summary_report.py
```

## Requirements Mapping

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| 8.1 - Generate markdown report per symbol | `_generate_symbol_markdown_report()` | Lines 1306-1595 |
| 8.2 - Train/val/test accuracy comparison | Model Performance Comparison table | Lines 1360-1398 |
| 8.3 - Overfitting metrics | Overfitting Analysis section | Lines 1400-1433 |
| 8.4 - Cross-validation stability | CV Stability section | Lines 1435-1471 |
| 8.5 - Feature importance | Feature Importance section | Lines 1473-1497 |
| 8.6 - Confusion matrices | Confusion Matrices section | Lines 1499-1541 |
| 8.7 - Deployment recommendation | Deployment Recommendation section | Lines 1330-1358 |

## Related Files

- **train_all_models.py** - Main implementation
- **TASK_12_PERFORMANCE_REPORTING_VERIFICATION.md** - Verification report
- **TASK_11.3_COMPLETE.md** - Original implementation summary
- **SUMMARY_REPORT_QUICK_REFERENCE.md** - Alternative reference guide
- **test_summary_report_simple.py** - Simple test
- **test_summary_report.py** - Comprehensive test

## Tips

1. **Always review the deployment recommendation** before deploying models
2. **Check the warnings section** for potential issues
3. **Compare CV stability** across models to identify reliable performers
4. **Use feature importance** to understand model decisions
5. **Monitor train-val gaps** to detect overfitting early
6. **Review confusion matrices** to understand prediction patterns

## Troubleshooting

### No reports generated
- Check that `trainer.results` contains data
- Verify output directory exists and is writable
- Check console for error messages

### Missing sections in report
- Some sections only appear for certain model types
- Feature importance: tree-based models only
- CV results: only if cross-validation was performed

### Incorrect metrics
- Ensure models were trained successfully
- Check for training errors in results
- Verify data quality and preprocessing

## Support

For issues or questions:
1. Check test scripts for examples
2. Review implementation in `train_all_models.py`
3. See related documentation files
