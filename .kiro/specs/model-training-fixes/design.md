# Design Document: Model Training Reliability & Performance Fixes

## Overview

This design addresses critical failures and performance issues in the model training pipeline. The solution implements systematic fixes across four model types (RandomForest, XGBoost, Neural Network, LSTM) to achieve reliable training with <15% overfitting and >65% test accuracy.

**Key Design Principles:**
1. **Fail-Safe Serialization** - Convert all numpy types before JSON saving
2. **Aggressive Regularization** - Stronger constraints than current implementation
3. **Smart Data Augmentation** - Adaptive strategies based on dataset size
4. **Architecture Simplification** - Reduce complexity for small datasets
5. **Automated Quality Control** - Real-time monitoring and model selection

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Orchestrator                      │
│                  (train_all_models.py)                       │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────────────────────────────────────────┐
             │                                                   │
    ┌────────▼────────┐                              ┌──────────▼─────────┐
    │  Data Pipeline  │                              │  Model Trainers    │
    │                 │                              │                    │
    │ • Load & Split  │                              │ • RandomForest     │
    │ • Augmentation  │                              │ • XGBoost          │
    │ • Validation    │                              │ • NeuralNetwork    │
    └────────┬────────┘                              │ • LSTM             │
             │                                       └──────────┬─────────┘
             │                                                  │
    ┌────────▼────────────────────────────────────────────────▼─────────┐
    │                    Quality Control Layer                           │
    │                                                                    │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
    │  │ Overfitting  │  │ Stability    │  │ Model        │           │
    │  │ Monitor      │  │ Checker      │  │ Selector     │           │
    │  └──────────────┘  └──────────────┘  └──────────────┘           │
    └────────────────────────────────────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │  Output Layer   │
    │                 │
    │ • Saved Models  │
    │ • Reports       │
    │ • Manifests     │
    └─────────────────┘
```

## Components and Interfaces

### 1. JSON Serialization Fix (Requirement 1)

**Location:** `models/base_model.py` - `save_model()` method

**Implementation:**
```python
def _convert_to_json_serializable(self, obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [self._convert_to_json_serializable(item) for item in obj]
    return obj

def save_model(self, output_dir: str):
    """Save model with safe JSON serialization"""
    # ... existing pickle save code ...
    
    # Convert metadata before JSON save
    metadata_safe = self._convert_to_json_serializable(metadata)
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata_safe, f, indent=2)
    except TypeError as e:
        logger.error(f"JSON serialization failed for field: {e}")
        # Save what we can
        safe_metadata = {k: v for k, v in metadata_safe.items() 
                        if not isinstance(v, (np.ndarray, np.generic))}
        with open(metadata_path, 'w') as f:
            json.dump(safe_metadata, f, indent=2)
```

**Interface:**
- Input: Model metadata dictionary (may contain numpy types)
- Output: JSON-serializable dictionary
- Error Handling: Graceful degradation - saves partial metadata if full save fails

### 2. Enhanced Regularization (Requirement 2)

**Location:** Individual model files

**RandomForest Changes:**
```python
# models/random_forest_model.py
def train(self, ...):
    self.model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,              # REDUCED from 15
        min_samples_split=20,      # INCREASED from 10
        min_samples_leaf=10,       # INCREASED from 5
        max_features='sqrt',
        max_samples=0.7,           # REDUCED from 0.8
        class_weight='balanced',
        random_state=42
    )
```

**XGBoost Changes:**
```python
# models/xgboost_model.py
def train(self, ...):
    self.model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,               # REDUCED from 6
        learning_rate=0.1,
        subsample=0.7,             # REDUCED from 0.8
        colsample_bytree=0.7,      # REDUCED from 0.8
        min_child_weight=5,        # INCREASED from 3
        reg_alpha=0.2,             # INCREASED from 0.1
        reg_lambda=2.0,            # INCREASED from 1.0
        ...
    )
```

**Neural Network Changes:**
```python
# models/neural_network_model.py
def train(self, ...):
    # Architecture: Reduce complexity
    hidden_dims=[256, 128, 64],    # REDUCED from [512, 256, 128, 64]
    dropout=0.4,                   # INCREASED from 0.3
    weight_decay=0.001,            # INCREASED from 0.0001
    label_smoothing=0.2,           # INCREASED from 0.15
```

**LSTM Changes:**
```python
# models/lstm_model.py
def train(self, ...):
    hidden_dim=64,                 # REDUCED from 128
    num_layers=1,                  # REDUCED from 2
    lookback=10,                   # REDUCED from 20
    dropout=0.5,                   # INCREASED from 0.3
    batch_size=16,                 # REDUCED from 32
    learning_rate=0.0005,          # REDUCED from 0.001
    patience=15,                   # REDUCED from 30
```

### 3. Adaptive Data Augmentation (Requirement 3)

**Location:** `models/data_augmentation.py`

**New Implementation:**
```python
class DataAugmenter:
    def augment(self, X, y, threshold=300):
        """Adaptive augmentation based on dataset size"""
        n_samples = len(X)
        
        if n_samples >= threshold:
            return X, y
        
        # Determine augmentation factor
        if n_samples < 200:
            target_size = n_samples * 3  # 3x for very small
        else:
            target_size = n_samples * 2  # 2x for small
        
        # Apply multiple augmentation techniques
        X_aug, y_aug = X.copy(), y.copy()
        
        # 1. Gaussian noise (increased magnitude)
        X_noise = self.add_gaussian_noise(X, std=0.15)  # INCREASED from 0.1
        
        # 2. Time-shift augmentation (new)
        X_shifted = self.time_shift(X, max_shift=2)
        
        # 3. Feature dropout (new)
        X_dropout = self.feature_dropout(X, dropout_rate=0.1)
        
        # 4. SMOTE for class balance
        X_combined = np.vstack([X_aug, X_noise, X_shifted, X_dropout])
        y_combined = np.hstack([y_aug, y, y, y])
        
        X_final, y_final = self.apply_smote(X_combined, y_combined)
        
        # Trim to target size
        if len(X_final) > target_size:
            indices = np.random.choice(len(X_final), target_size, replace=False)
            X_final, y_final = X_final[indices], y_final[indices]
        
        # Validate label distribution (within 5%)
        self._validate_distribution(y, y_final, tolerance=0.05)
        
        return X_final, y_final
    
    def time_shift(self, X, max_shift=2):
        """Shift features by ±max_shift timesteps"""
        X_shifted = X.copy()
        shift = np.random.randint(-max_shift, max_shift + 1, size=len(X))
        for i in range(len(X)):
            if shift[i] != 0:
                X_shifted[i] = np.roll(X[i], shift[i])
        return X_shifted
    
    def feature_dropout(self, X, dropout_rate=0.1):
        """Randomly zero out features"""
        X_dropout = X.copy()
        mask = np.random.binomial(1, 1-dropout_rate, X.shape)
        X_dropout *= mask
        return X_dropout
    
    def _validate_distribution(self, y_original, y_augmented, tolerance=0.05):
        """Ensure augmented data preserves label distribution"""
        orig_dist = np.bincount(y_original) / len(y_original)
        aug_dist = np.bincount(y_augmented) / len(y_augmented)
        
        diff = np.abs(orig_dist - aug_dist)
        if np.any(diff > tolerance):
            logger.warning(f"Label distribution shifted by {diff.max():.2%}")
```

### 4. Model Selection System (Requirement 5)

**Location:** `train_all_models.py` - new class

**Implementation:**
```python
class ModelSelector:
    """Automatic selection of best-performing models"""
    
    def __init__(self, max_gap=0.20, min_accuracy=0.55, stability_threshold=0.05):
        self.max_gap = max_gap
        self.min_accuracy = min_accuracy
        self.stability_threshold = stability_threshold
    
    def select_best_models(self, results: Dict) -> Dict:
        """
        Select best model per symbol based on criteria
        
        Returns:
            {
                'symbol': {
                    'selected_model': 'XGBoost',
                    'test_accuracy': 0.72,
                    'train_val_gap': 0.12,
                    'reason': 'Best test accuracy with acceptable gap',
                    'alternatives': ['NeuralNetwork']
                }
            }
        """
        selections = {}
        
        for symbol, symbol_results in results.items():
            candidates = []
            
            for model_name, metrics in symbol_results.items():
                if 'error' in metrics:
                    continue
                
                # Extract metrics
                test_acc = metrics['test_metrics']['accuracy']
                train_val_gap = metrics['history'].get('train_val_gap', 0)
                val_acc = metrics['val_metrics']['accuracy']
                
                # Apply filters
                if train_val_gap > self.max_gap:
                    continue  # Reject overfitting models
                
                if test_acc < self.min_accuracy:
                    continue  # Reject low accuracy models
                
                # Check val-test consistency (within 5%)
                val_test_diff = abs(val_acc - test_acc)
                if val_test_diff > self.stability_threshold:
                    continue  # Reject unstable models
                
                candidates.append({
                    'model': model_name,
                    'test_accuracy': test_acc,
                    'train_val_gap': train_val_gap,
                    'val_test_diff': val_test_diff,
                    'score': test_acc - (train_val_gap * 0.5)  # Penalize overfitting
                })
            
            if not candidates:
                selections[symbol] = {
                    'selected_model': None,
                    'reason': 'No models met quality criteria',
                    'action': 'MANUAL_REVIEW_REQUIRED'
                }
                continue
            
            # Select best by score
            best = max(candidates, key=lambda x: x['score'])
            alternatives = [c['model'] for c in candidates if c['model'] != best['model']]
            
            selections[symbol] = {
                'selected_model': best['model'],
                'test_accuracy': best['test_accuracy'],
                'train_val_gap': best['train_val_gap'],
                'reason': f"Best score ({best['score']:.3f}) with gap {best['train_val_gap']:.1%}",
                'alternatives': alternatives
            }
        
        return selections
    
    def save_deployment_manifest(self, selections: Dict, output_path: str):
        """Save deployment manifest JSON"""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'selection_criteria': {
                'max_train_val_gap': self.max_gap,
                'min_test_accuracy': self.min_accuracy,
                'max_val_test_diff': self.stability_threshold
            },
            'selections': selections,
            'summary': {
                'total_symbols': len(selections),
                'models_selected': sum(1 for s in selections.values() if s['selected_model']),
                'manual_review_needed': sum(1 for s in selections.values() if not s['selected_model'])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
```

### 5. Early Warning System (Requirement 9)

**Location:** `models/base_model.py` - new mixin class

**Implementation:**
```python
class TrainingMonitor:
    """Real-time training monitoring and alerts"""
    
    def __init__(self):
        self.warnings = []
        self.should_stop = False
    
    def check_overfitting(self, train_acc, val_acc, epoch):
        """Check for severe overfitting"""
        if train_acc > 0.95 and val_acc < 0.60:
            warning = f"Epoch {epoch}: Severe overfitting detected (train={train_acc:.2f}, val={val_acc:.2f})"
            self.warnings.append(warning)
            logger.warning(warning)
            return True
        return False
    
    def check_divergence(self, val_losses, patience=10):
        """Check for validation loss divergence"""
        if len(val_losses) < patience:
            return False
        
        recent_losses = val_losses[-patience:]
        if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
            warning = f"Validation loss increasing for {patience} consecutive epochs"
            self.warnings.append(warning)
            logger.warning(warning)
            return True
        return False
    
    def check_nan_loss(self, loss, epoch):
        """Check for NaN loss"""
        if np.isnan(loss):
            warning = f"Epoch {epoch}: NaN loss detected - stopping training"
            self.warnings.append(warning)
            logger.error(warning)
            self.should_stop = True
            return True
        return False
    
    def check_exploding_gradients(self, grad_norm, threshold=10.0, epoch=None):
        """Check for exploding gradients"""
        if grad_norm > threshold:
            warning = f"Epoch {epoch}: Exploding gradient detected (norm={grad_norm:.2f})"
            self.warnings.append(warning)
            logger.warning(warning)
            return True
        return False
    
    def check_timeout(self, start_time, max_minutes=10):
        """Check for training timeout"""
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        if elapsed > max_minutes:
            warning = f"Training exceeded {max_minutes} minutes"
            self.warnings.append(warning)
            logger.warning(warning)
            return True
        return False
    
    def get_warnings(self):
        """Get all warnings"""
        return self.warnings
```

## Data Models

### Training Result Schema

```python
{
    "symbol": "EURUSD",
    "model_name": "XGBoost",
    "timestamp": "2025-10-19T12:00:00",
    "history": {
        "train_accuracy": 0.85,
        "val_accuracy": 0.72,
        "test_accuracy": 0.70,
        "train_val_gap": 0.13,
        "overfitting_detected": false,
        "cv_mean_accuracy": 0.71,
        "cv_std_accuracy": 0.04,
        "cv_is_stable": true,
        "cv_fold_accuracies": [0.68, 0.72, 0.70, 0.73, 0.72]
    },
    "val_metrics": {
        "accuracy": 0.72,
        "precision": 0.70,
        "recall": 0.69,
        "f1_score": 0.69,
        "confusion_matrix": [[15, 5], [6, 20]]
    },
    "test_metrics": {
        "accuracy": 0.70,
        "precision": 0.68,
        "recall": 0.67,
        "f1_score": 0.67
    },
    "feature_importance": [
        {"feature": "OB_Age", "importance": 0.12},
        {"feature": "TBM_Risk_Per_Trade_ATR", "importance": 0.08}
    ],
    "warnings": [],
    "training_duration_seconds": 45.2
}
```

### Deployment Manifest Schema

```python
{
    "timestamp": "2025-10-19T12:00:00",
    "selection_criteria": {
        "max_train_val_gap": 0.20,
        "min_test_accuracy": 0.55,
        "max_val_test_diff": 0.05
    },
    "selections": {
        "EURUSD": {
            "selected_model": "XGBoost",
            "test_accuracy": 0.72,
            "train_val_gap": 0.13,
            "reason": "Best score (0.655) with gap 13.0%",
            "alternatives": ["NeuralNetwork"]
        },
        "GBPUSD": {
            "selected_model": null,
            "reason": "No models met quality criteria",
            "action": "MANUAL_REVIEW_REQUIRED"
        }
    },
    "summary": {
        "total_symbols": 11,
        "models_selected": 9,
        "manual_review_needed": 2
    }
}
```

## Error Handling

### 1. JSON Serialization Errors
- **Strategy:** Graceful degradation
- **Action:** Save partial metadata, log failed fields
- **Recovery:** Model pickle still saved, can be loaded

### 2. Training Failures
- **Strategy:** Continue with other models
- **Action:** Log error, save error state in results
- **Recovery:** Other models for same symbol still train

### 3. Data Augmentation Failures
- **Strategy:** Fall back to original data
- **Action:** Log warning, proceed without augmentation
- **Recovery:** Training continues with smaller dataset

### 4. Cross-Validation Failures
- **Strategy:** Skip CV, proceed to training
- **Action:** Log warning, mark CV results as unavailable
- **Recovery:** Model still trains, just without CV metrics

## Testing Strategy

### Unit Tests

1. **JSON Serialization Test**
```python
def test_json_serialization():
    metadata = {
        'accuracy': np.float64(0.85),
        'is_overfitting': np.bool_(True),
        'confusion_matrix': np.array([[10, 5], [3, 12]])
    }
    safe_metadata = _convert_to_json_serializable(metadata)
    json.dumps(safe_metadata)  # Should not raise
```

2. **Data Augmentation Test**
```python
def test_adaptive_augmentation():
    X = np.random.rand(150, 57)
    y = np.random.randint(0, 2, 150)
    
    augmenter = DataAugmenter()
    X_aug, y_aug = augmenter.augment(X, y, threshold=300)
    
    assert len(X_aug) >= len(X) * 2  # At least 2x
    assert len(X_aug) <= len(X) * 3  # At most 3x
    
    # Check label distribution preserved
    orig_dist = np.bincount(y) / len(y)
    aug_dist = np.bincount(y_aug) / len(y_aug)
    assert np.all(np.abs(orig_dist - aug_dist) < 0.05)
```

3. **Model Selection Test**
```python
def test_model_selection():
    results = {
        'EURUSD': {
            'XGBoost': {
                'test_metrics': {'accuracy': 0.72},
                'val_metrics': {'accuracy': 0.70},
                'history': {'train_val_gap': 0.13}
            },
            'RandomForest': {
                'test_metrics': {'accuracy': 0.65},
                'val_metrics': {'accuracy': 0.50},
                'history': {'train_val_gap': 0.25}  # Too high
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    assert selections['EURUSD']['selected_model'] == 'XGBoost'
    assert 'RandomForest' not in selections['EURUSD']['alternatives']
```

### Integration Tests

1. **End-to-End Training Test**
```python
def test_full_training_pipeline():
    trainer = SMCModelTrainer()
    results = trainer.train_all_for_symbol('EURUSD', models=['XGBoost'])
    
    assert 'XGBoost' in results
    assert 'error' not in results['XGBoost']
    assert results['XGBoost']['test_metrics']['accuracy'] > 0.55
    assert results['XGBoost']['history']['train_val_gap'] < 0.20
```

2. **Model Saving/Loading Test**
```python
def test_model_persistence():
    model = XGBoostSMCModel('EURUSD')
    # ... train model ...
    model.save_model('test_output')
    
    # Check files exist
    assert Path('test_output/EURUSD_XGBoost.pkl').exists()
    assert Path('test_output/EURUSD_XGBoost_metadata.json').exists()
    
    # Load and verify
    loaded_model = XGBoostSMCModel('EURUSD')
    loaded_model.load_model('test_output')
    assert loaded_model.is_trained
```

## Performance Considerations

### Memory Optimization
- **Data Augmentation:** Process in batches to avoid memory spikes
- **LSTM Sequences:** Use generators for large lookback windows
- **Model Storage:** Compress large models (>100MB) with joblib compression

### Training Speed
- **Parallel Training:** Train different symbols in parallel (not implemented in Phase 1)
- **GPU Acceleration:** Use GPU for Neural Network and LSTM when available
- **Early Stopping:** Aggressive patience values to avoid wasted epochs

### Disk Usage
- **Model Pruning:** Delete intermediate checkpoints after training
- **Report Compression:** Compress large JSON reports
- **Cleanup:** Remove augmented data after training

## Deployment Considerations

### Model Versioning
- Include timestamp in model filenames
- Save hyperparameters in metadata
- Track data version used for training

### Monitoring
- Log all warnings to file
- Generate summary reports after each run
- Alert on critical failures (email/Slack integration in future)

### Rollback Strategy
- Keep previous model versions
- Include model comparison in deployment manifest
- Automated A/B testing framework (future enhancement)

## Future Enhancements

1. **Hyperparameter Tuning** (Requirement 10)
   - Implement Optuna for automated search
   - Symbol-specific tuning for poor performers
   - Bayesian optimization for efficiency

2. **Ensemble Models**
   - Weighted voting based on validation performance
   - Stacking with meta-learner
   - Dynamic ensemble selection per symbol

3. **Online Learning**
   - Incremental updates with new data
   - Concept drift detection
   - Adaptive retraining triggers

4. **Distributed Training**
   - Multi-GPU support for deep learning
   - Parallel symbol training
   - Cloud-based training orchestration
