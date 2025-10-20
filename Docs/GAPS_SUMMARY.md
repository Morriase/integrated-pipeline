# SMC Training Pipeline - Functional Gaps Summary

## Quick Reference

**Total Gaps Identified:** 16
- 🔴 **High Priority:** 4 gaps
- 🟡 **Medium Priority:** 4 gaps  
- 🟢 **Low Priority:** 8 gaps

---

## 🔴 CRITICAL GAPS (Fix Before Production)

### GAP 4.2: Feature Selection Not Enabled
**Location:** `train_all_models.py` → model training calls

**Problem:**
- Models train on 100+ features without selection
- `FeatureSelector` class exists but `apply_feature_selection=False` by default
- High dimensionality → overfitting risk

**Impact:** 
- Curse of dimensionality
- Models memorize noise
- Poor generalization

**Fix:**
```python
# In train_all_models.py, update all prepare_features() calls:
X_train, y_train = model.prepare_features(
    train_df, 
    fit_scaler=True,
    apply_feature_selection=True  # ADD THIS
)
```

**Effort:** 5 minutes (1 line change per model)

---

### GAP 4.5: LSTM Data Representation Mismatch
**Location:** `models/lstm_model.py` vs other models

**Problem:**
- LSTM uses sequences (10 candles lookback)
- RF/XGB/NN use single candles
- Different effective dataset sizes:
  - RF/XGB/NN: 2,500 samples
  - LSTM: 2,490 samples (loses first 10)
- Unfair comparison

**Impact:**
- Can't fairly compare LSTM to other models
- LSTM sees temporal patterns, others don't
- May explain LSTM's poor performance

**Fix Options:**

**Option A: Add temporal features to other models**
```python
# In data_preparation_pipeline.py, add:
def add_temporal_features(df, lookback=10):
    """Add lagged features for temporal context"""
    for lag in range(1, lookback+1):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    return df
```

**Option B: Keep LSTM disabled (current approach)**
- LSTM is already optional (`--include-lstm` flag)
- Document that it's experimental
- Focus on stable models (RF, XGB, NN)

**Recommendation:** Option B (current approach is good)

**Effort:** 0 minutes (already implemented)

---

### GAP 5.1: No Business Metrics
**Location:** `base_model.py` → `evaluate()` method

**Problem:**
- Only classification metrics (accuracy, F1, precision, recall)
- No trading-specific metrics:
  - Win rate
  - Average R:R achieved
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio

**Impact:**
- Can't assess real trading viability
- High accuracy ≠ profitable trading
- Example: 90% accuracy but all wins are 1R, all losses are 3R = losing strategy

**Fix:**
```python
# Add to base_model.py:
def calculate_trading_metrics(self, y_true, y_pred, y_proba=None):
    """Calculate trading-specific metrics"""
    
    # Win rate
    wins = np.sum(y_pred == 1)
    losses = np.sum(y_pred == -1)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Profit factor (assuming 1:2 R:R)
    rr_ratio = 2.0
    total_profit = wins * rr_ratio  # Each win = 2R
    total_loss = losses * 1.0  # Each loss = 1R
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    # Expected value per trade
    ev_per_trade = (win_rate * rr_ratio) - ((1 - win_rate) * 1.0)
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expected_value_per_trade': ev_per_trade,
        'total_wins': wins,
        'total_losses': losses
    }
```

**Effort:** 30 minutes

---

### GAP 3.2: Label Imbalance Not Addressed
**Location:** `data_preparation_pipeline.py` → TBM labeling

**Problem:**
- Timeout (0) labels often dominate (30-40% of data)
- Win/Loss ratio may be skewed
- No class balancing in training

**Impact:**
- Model biased toward majority class
- Poor performance on minority class
- May predict "Timeout" for everything

**Fix Option A: Class Weights**
```python
# In RandomForest/XGBoost training:
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# RandomForest
model = RandomForestClassifier(class_weight=class_weight_dict)

# XGBoost
scale_pos_weight = class_weights[1] / class_weights[0]
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
```

**Fix Option B: SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Recommendation:** Option A (class weights) - simpler, no synthetic data

**Effort:** 15 minutes

---

## 🟡 MEDIUM PRIORITY GAPS

### GAP 1.1: No Data Quality Validation
**Location:** `consolidate_mt5_data.py`

**Problem:**
- No checks for duplicate timestamps
- No detection of missing candles (gaps)
- No price anomaly detection (spikes, zeros)
- No timezone consistency validation

**Impact:** Corrupted data propagates through entire pipeline

**Fix:**
```python
def validate_data_quality(df):
    """Validate OHLC data quality"""
    issues = []
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['time', 'symbol', 'timeframe'])
    if duplicates.any():
        issues.append(f"Found {duplicates.sum()} duplicate timestamps")
    
    # Check for missing candles
    for symbol in df['symbol'].unique():
        for tf in df['timeframe'].unique():
            subset = df[(df['symbol'] == symbol) & (df['timeframe'] == tf)]
            time_diff = subset['time'].diff()
            expected_diff = pd.Timedelta(minutes=15)  # Adjust per timeframe
            gaps = time_diff[time_diff > expected_diff * 1.5]
            if len(gaps) > 0:
                issues.append(f"{symbol} {tf}: {len(gaps)} missing candles")
    
    # Check for price anomalies
    if (df['close'] <= 0).any():
        issues.append("Found zero or negative prices")
    
    return issues
```

**Effort:** 1 hour

---

### GAP 2.2: No Feature Importance Tracking
**Location:** `data_preparation_pipeline.py`

**Problem:**
- Pipeline generates 100+ features
- No tracking of which features are predictive
- Can't identify redundant features
- Can't optimize feature engineering

**Impact:** Unclear which SMC concepts are useful

**Fix:**
```python
# After training all models, generate report:
def generate_feature_importance_report(models, output_path):
    """Aggregate feature importance across models"""
    
    importance_df = pd.DataFrame()
    
    for model_name, model in models.items():
        if hasattr(model, 'get_feature_importance'):
            fi = model.get_feature_importance(top_n=50)
            importance_df[model_name] = fi['importance']
    
    # Average importance across models
    importance_df['avg_importance'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    # Save report
    importance_df.to_csv(output_path, index=True)
    
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10))
```

**Effort:** 30 minutes

---

### GAP 5.2: No Confidence Calibration
**Location:** `base_model.py` → prediction methods

**Problem:**
- Models output probabilities
- No calibration check (are 70% predictions actually 70% accurate?)
- No confidence thresholding
- Can't filter low-confidence predictions

**Impact:** May take bad trades with false confidence

**Fix:**
```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# After training, check calibration:
def check_calibration(y_true, y_proba):
    """Check probability calibration"""
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba[:, 1], n_bins=10
    )
    
    # Plot calibration curve
    import matplotlib.pyplot as plt
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.savefig('calibration_curve.png')
    
    return prob_true, prob_pred

# Apply calibration if needed:
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
```

**Effort:** 1 hour

---

### GAP 6.1: No Model Versioning
**Location:** `base_model.py` → `save_model()`

**Problem:**
- Models overwrite previous versions
- No version tracking
- Can't rollback to previous model
- Can't compare model versions

**Impact:** Risk of losing good models

**Fix:**
```python
import datetime

def save_model(self, output_dir: str):
    """Save model with version timestamp"""
    
    # Create version timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    version_dir = Path(output_dir) / f"v_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = version_dir / f"{self.symbol}_{self.model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'training_history': self.training_history,
            'version': timestamp,
            'hyperparameters': self.get_hyperparameters()
        }, f)
    
    # Save metadata
    metadata = {
        'version': timestamp,
        'model_name': self.model_name,
        'symbol': self.symbol,
        'training_date': datetime.datetime.now().isoformat(),
        'hyperparameters': self.get_hyperparameters(),
        'performance': self.training_history
    }
    
    with open(version_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create symlink to latest
    latest_link = Path(output_dir) / 'latest'
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(version_dir)
```

**Effort:** 45 minutes

---

## 🟢 LOW PRIORITY GAPS

### GAP 2.1: Feature Explosion Without Selection
**Status:** Addressed by GAP 4.2 fix

### GAP 2.3: Lookback Window Inconsistency
**Status:** Acceptable (LSTM is optional/experimental)

### GAP 3.1: No Symbol-Level Stratification
**Impact:** Minor - unified approach already helps

**Fix:** Stratify splits by symbol in addition to time

**Effort:** 30 minutes

### GAP 4.1: No Data Augmentation
**Impact:** Minor - 2,500+ samples is reasonable

**Fix:** Add SMOTE or time-series augmentation

**Effort:** 1 hour

### GAP 4.3: Inconsistent Scaling
**Status:** Acceptable - tree models don't need scaling, NN/LSTM do

### GAP 4.4: No Batch Normalization
**Impact:** Minor - dropout is working

**Fix:** Add BatchNorm layers to neural network

**Effort:** 30 minutes

### GAP 4.6: LSTM Instability Issues
**Status:** Addressed by making LSTM optional

### GAP 6.2: No Model Metadata
**Status:** Addressed by GAP 6.1 fix

---

## Action Plan

### Phase 1: Critical Fixes (Before Next Training)
**Time Required:** ~1 hour

1. ✅ Enable feature selection (5 min)
2. ✅ Add class weights for imbalance (15 min)
3. ✅ Add business metrics calculation (30 min)
4. ✅ Document LSTM as experimental (already done)

### Phase 2: Quality Improvements (Next Sprint)
**Time Required:** ~3 hours

5. Add data quality validation (1 hour)
6. Generate feature importance report (30 min)
7. Add confidence calibration (1 hour)
8. Implement model versioning (45 min)

### Phase 3: Optimizations (Future)
**Time Required:** ~2 hours

9. Add batch normalization to NN (30 min)
10. Symbol-level stratification (30 min)
11. Data augmentation (1 hour)

---

## Quick Wins (Do These Now)

### 1. Enable Feature Selection
```bash
# Edit train_all_models.py
# Change: apply_feature_selection=False
# To:     apply_feature_selection=True
```

### 2. Add Class Weights
```python
# In random_forest_model.py and xgboost_model.py
# Add class_weight='balanced' parameter
```

### 3. Calculate Win Rate
```python
# In base_model.py evaluate() method
wins = np.sum(y_pred == 1)
losses = np.sum(y_pred == -1)
win_rate = wins / (wins + losses)
print(f"Win Rate: {win_rate:.2%}")
```

---

## Conclusion

**Current Status:** Pipeline is functional but needs optimization

**Critical Issues:** 4 gaps that should be fixed before production
**Medium Issues:** 4 gaps that improve reliability
**Low Issues:** 8 gaps that are nice-to-have

**Recommendation:** 
1. Fix the 4 critical gaps (1 hour work)
2. Run training with fixes
3. Evaluate results
4. Then address medium priority gaps

**Total Effort to Production-Ready:** ~4 hours of development work
