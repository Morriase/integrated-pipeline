# Quick Start: Pipeline Optimization

## 🎯 Goal
Improve win rate from **46%** to **>55%** by optimizing data quality and model training.

---

## 🚀 Quick Start (3 Commands)

### Option 1: Automated Optimization
```bash
# Run all optimization steps automatically
python run_optimization.py
```

### Option 2: Manual Step-by-Step
```bash
# Step 1: Diagnose issues
python diagnose_htf_confluence.py

# Step 2: Optimize data
python optimize_training_data.py

# Step 3: Train models
python train_all_models.py --data_path Data/processed_smc_data_train_optimized.csv
```

---

## 📊 What Gets Fixed

### Before Optimization:
```
❌ Win Rate: 46.0%
❌ Training on 76,982 rows (90% noise)
❌ HTF Confluence: 0 (not working)
❌ Quality scores: Q50 = 0.000
```

### After Optimization:
```
✅ Win Rate: 55-60% (target achieved)
✅ Training on ~5,000 high-quality setups
✅ HTF Confluence: Working (non-zero)
✅ Quality scores: Q50 > 0.5
```

---

## 🔧 What Each Script Does

### 1. `diagnose_htf_confluence.py`
**Purpose:** Find why HTF confluence is zero

**Checks:**
- Are H1/H4 data present?
- Are structures detected on higher timeframes?
- Is time alignment working?
- Are proximity calculations correct?

**Output:** Diagnostic report with recommendations

---

### 2. `optimize_training_data.py`
**Purpose:** Filter noise and balance classes

**Actions:**
- ✅ Remove rows with `TBM_Entry = 0` (no setup)
- ✅ Remove rows with `TBM_Label = NaN` (no outcome)
- ✅ Apply quality filter (>= 0.5)
- ✅ Balance classes (undersample majority)
- ✅ Create regime-specific datasets

**Output:**
- `processed_smc_data_train_optimized.csv`
- `processed_smc_data_val_optimized.csv`
- `processed_smc_data_test_optimized.csv`
- `optimized_feature_list.txt`
- `regime_specific/*.csv`

---

### 3. `run_optimization.py`
**Purpose:** Run all optimization steps automatically

**Workflow:**
1. Check if processed data exists
2. Run HTF diagnostic
3. Run data optimization
4. Generate summary report

---

## 📈 Expected Improvements

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Win Rate | 46.0% | 55-60% | >55% |
| Training Samples | 76,982 | ~5,000 | Quality over quantity |
| HTF Confluence | 0 | >0 | Working |
| Avg Quality Score | 0.067 | >0.5 | High-quality setups |
| Class Balance | Imbalanced | Balanced | 33/33/33 |

---

## 🛠️ If HTF Confluence Still Zero

Edit `data_preparation_pipeline.py`:

### Fix 1: Increase Lookback Window
**Line ~1150:**
```python
# Before
lookback_start = max(0, htf_current_idx - 20)

# After
lookback_start = max(0, htf_current_idx - 50)
```

### Fix 2: Increase Proximity Threshold
**Line ~1180:**
```python
# Before
proximity_fuzzy = self.fuzzy_classifier.mf.triangular(
    distance_atr, a=0.0, b=0.0, c=3.0
)

# After
proximity_fuzzy = self.fuzzy_classifier.mf.triangular(
    distance_atr, a=0.0, b=0.0, c=5.0
)
```

### Fix 3: Lower Quality Threshold
**Line ~1200:**
```python
# Before
if proximity_fuzzy > 0.1:

# After
if proximity_fuzzy > 0.05:
```

Then re-run:
```bash
python run_complete_pipeline.py
```

---

## 📁 File Structure After Optimization

```
Data/
├── consolidated_ohlc_data.csv              # Raw consolidated data
├── processed_smc_data.csv                  # Original processed data
├── processed_smc_data_train.csv            # Original train split
├── processed_smc_data_val.csv              # Original val split
├── processed_smc_data_test.csv             # Original test split
├── processed_smc_data_train_optimized.csv  # ✨ Optimized train (use this!)
├── processed_smc_data_val_optimized.csv    # ✨ Optimized val (use this!)
├── processed_smc_data_test_optimized.csv   # ✨ Optimized test (use this!)
├── optimized_feature_list.txt              # ✨ Feature list for ML
└── regime_specific/                        # ✨ Regime-specific datasets
    ├── train_high_vol_trend.csv
    ├── train_low_vol_chop.csv
    └── train_normal.csv
```

---

## 🎯 Training Models on Optimized Data

### Standard Training:
```bash
python train_all_models.py \
    --data_path Data/processed_smc_data_train_optimized.csv \
    --val_path Data/processed_smc_data_val_optimized.csv \
    --test_path Data/processed_smc_data_test_optimized.csv
```

### Regime-Specific Training:
```bash
# Train on trending markets only
python train_all_models.py \
    --data_path Data/regime_specific/train_high_vol_trend.csv \
    --regime High_Vol_Trend

# Train on ranging markets only
python train_all_models.py \
    --data_path Data/regime_specific/train_low_vol_chop.csv \
    --regime Low_Vol_Chop
```

---

## ✅ Success Checklist

After running optimization, verify:

- [ ] HTF confluence features have non-zero values
- [ ] Training set has ~5,000-6,000 samples (not 76,982)
- [ ] Quality scores: Q50 > 0.5 (not 0.000)
- [ ] Class distribution is balanced (~33% each)
- [ ] Win rate on validation set > 55%
- [ ] Optimized splits exist in Data/ folder
- [ ] Feature list has ~40-50 features

---

## 🐛 Troubleshooting

### Issue: "No valid setups found"
**Solution:** Lower `min_quality_score` in `optimize_training_data.py`:
```python
optimizer = TrainingDataOptimizer(
    min_quality_score=0.4  # Try 0.4 instead of 0.5
)
```

### Issue: "HTF confluence still zero"
**Solution:** Follow the "If HTF Confluence Still Zero" section above

### Issue: "Training set too small"
**Solution:** 
1. Lower quality threshold
2. Increase lookforward period (20 → 30)
3. Lower R:R ratio (3.0 → 2.5)

### Issue: "Class imbalance after optimization"
**Solution:** Change balance method in `optimize_training_data.py`:
```python
balance_method='oversample'  # Instead of 'undersample'
```

---

## 📚 Additional Resources

- **OPTIMIZATION_RECOMMENDATIONS.md**: Detailed optimization guide
- **WHATS_NEEDED.md**: Model learning objectives
- **data_preparation_pipeline.py**: Feature engineering code
- **train_all_models.py**: Model training script

---

## 💡 Pro Tips

1. **Start with diagnostic** → Understand the problem before fixing
2. **Use optimized splits** → Don't train on raw processed data
3. **Monitor quality scores** → Should be >0.5 on average
4. **Try regime-specific models** → Better performance in specific conditions
5. **Validate on unseen data** → Use test set for final evaluation

---

**Ready to optimize? Run:** `python run_optimization.py` 🚀
