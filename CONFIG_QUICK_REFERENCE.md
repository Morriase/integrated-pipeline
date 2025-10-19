# Anti-Overfitting Configuration Quick Reference

Quick reference guide for using the anti-overfitting configuration system.

## Quick Start

```python
from models.config import load_config

# Load configuration
config = load_config('models/anti_overfitting_config.json')

# Use in training
model.train(
    X_train, y_train, X_val, y_val,
    max_depth=config.rf_config.max_depth,
    dropout=config.nn_config.dropout,
    # ... other parameters
)
```

## Common Tasks

### 1. Load Default Configuration

```python
from models.config import load_config

config = load_config()  # Uses defaults
```

### 2. Load from Custom File

```python
config = load_config('path/to/custom_config.json')
```

### 3. Modify Configuration

```python
config = load_config()

# Modify values
config.rf_config.max_depth = 20
config.nn_config.dropout = 0.6

# Validate
if config.validate():
    print("Configuration is valid")
```

### 4. Save Configuration

```python
config.save('models/my_config.json')
```

### 5. Create Default Config File

```python
from models.config import create_default_config_file

create_default_config_file('models/anti_overfitting_config.json')
```

## Key Configuration Values

### Random Forest (Reduce Overfitting)

```python
config.rf_config.max_depth = 15          # Lower = less overfitting
config.rf_config.min_samples_split = 20  # Higher = less overfitting
config.rf_config.min_samples_leaf = 10   # Higher = less overfitting
config.rf_config.max_samples = 0.8       # Lower = more diversity
```

### Neural Network (Increase Regularization)

```python
config.nn_config.hidden_dims = [256, 128, 64]  # Smaller = less overfitting
config.nn_config.dropout = 0.5                  # Higher = more regularization
config.nn_config.weight_decay = 0.1             # Higher = more L2 regularization
config.nn_config.learning_rate = 0.005          # Lower = more stable
```

### Feature Selection

```python
config.feature_selection_config.enabled = True
config.feature_selection_config.min_features = 30
config.feature_selection_config.correlation_threshold = 0.9
```

### Data Augmentation

```python
config.augmentation_config.enabled = True
config.augmentation_config.threshold = 300  # Augment if samples < 300
config.augmentation_config.noise_std = 0.01
```

### Cross-Validation

```python
config.cv_config.enabled = True
config.cv_config.n_folds = 5
config.cv_config.stability_threshold = 0.15
```

### Overfitting Monitor

```python
config.monitor_config.warning_threshold = 0.15  # Flag if gap > 15%
config.monitor_config.generate_curves = True
config.monitor_config.save_metrics = True
```

## Configuration Presets

### Maximum Anti-Overfitting

```python
config.rf_config.max_depth = 10
config.rf_config.min_samples_split = 30
config.nn_config.dropout = 0.6
config.nn_config.weight_decay = 0.15
```

### Balanced (Default)

Use default configuration file.

### Minimum Constraints

```python
config.rf_config.max_depth = 20
config.rf_config.min_samples_split = 10
config.nn_config.dropout = 0.3
config.nn_config.weight_decay = 0.05
```

## Troubleshooting

### Still Overfitting?

1. Increase `dropout` (NN) or reduce `max_depth` (RF)
2. Increase `weight_decay` (NN) or increase `min_samples_split` (RF)
3. Enable feature selection
4. Reduce model complexity

### Underfitting?

1. Decrease `dropout` (NN) or increase `max_depth` (RF)
2. Decrease `weight_decay` (NN) or decrease `min_samples_split` (RF)
3. Increase model complexity
4. Check feature selection isn't too aggressive

### High Variance?

1. Increase dataset size (enable augmentation)
2. Simplify model
3. Increase `n_folds` for cross-validation

## Validation

Always validate after modifications:

```python
if not config.validate():
    print("Configuration has errors!")
    # Check logs for details
```

## Legacy Support

For backward compatibility, dictionary exports are available:

```python
from models.config import (
    RF_CONFIG,
    NN_CONFIG,
    FEATURE_SELECTION_CONFIG,
    AUGMENTATION_CONFIG,
    CV_CONFIG,
    MONITOR_CONFIG
)

# Use as dictionaries
max_depth = RF_CONFIG['max_depth']
dropout = NN_CONFIG['dropout']
```

## Files

- **Configuration Module**: `models/config.py`
- **Default Config File**: `models/anti_overfitting_config.json`
- **Full Documentation**: `models/CONFIG_DOCUMENTATION.md`
- **Test Script**: `test_config.py`

## Testing

Run tests to verify configuration system:

```bash
python test_config.py
```

## Support

For detailed documentation, see `models/CONFIG_DOCUMENTATION.md`
