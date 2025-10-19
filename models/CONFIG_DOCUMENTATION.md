# Anti-Overfitting Configuration Documentation

This document provides comprehensive documentation for all anti-overfitting configuration options available in the SMC trading model training pipeline.

## Overview

The configuration management system provides centralized control over all anti-overfitting features including:
- Random Forest constraints
- Neural Network regularization
- Feature selection
- Data augmentation
- Cross-validation
- Overfitting monitoring

## Configuration File Location

Default configuration file: `models/anti_overfitting_config.json`

## Usage

### Loading Configuration

```python
from models.config import load_config, AntiOverfittingConfig

# Load from file
config = load_config('models/anti_overfitting_config.json')

# Use default configuration
config = load_config()

# Or create directly
config = AntiOverfittingConfig()
```

### Accessing Configuration Values

```python
# Access Random Forest settings
max_depth = config.rf_config.max_depth

# Access Neural Network settings
dropout = config.nn_config.dropout

# Access Feature Selection settings
min_features = config.feature_selection_config.min_features
```

### Modifying Configuration

```python
# Modify values
config.rf_config.max_depth = 20
config.nn_config.dropout = 0.6

# Validate changes
if config.validate():
    print("Configuration is valid")

# Save to file
config.save('models/custom_config.json')
```

### Creating Custom Configuration File

```python
from models.config import create_default_config_file

# Create default config file
create_default_config_file('models/my_config.json')
```

## Configuration Sections

### 1. Random Forest Configuration (`rf_config`)

Controls Random Forest model hyperparameters to prevent overfitting.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `n_estimators` | int | 200 | 10-1000 | Number of trees in the forest |
| `max_depth` | int | 15 | 5-50 | Maximum depth of trees (reduced from 20 to prevent overfitting) |
| `min_samples_split` | int | 20 | 2-100 | Minimum samples required to split a node (increased from 10) |
| `min_samples_leaf` | int | 10 | 1-50 | Minimum samples required at leaf node (increased from 5) |
| `max_features` | str | 'sqrt' | 'sqrt', 'log2', 'auto', None, or numeric | Number of features to consider for best split |
| `max_samples` | float | 0.8 | 0.0-1.0 | Fraction of samples to use for each tree (bootstrap sampling control) |
| `class_weight` | str | 'balanced' | - | Weight balancing strategy |
| `random_state` | int | 42 | - | Random seed for reproducibility |

**Purpose:** These constraints prevent Random Forest models from memorizing training data by limiting tree complexity and requiring more samples for splits.

**Impact on Overfitting:**
- Lower `max_depth` prevents deep trees that memorize noise
- Higher `min_samples_split` and `min_samples_leaf` ensure splits are statistically significant
- `max_samples` < 1.0 introduces diversity through bootstrap sampling

---

### 2. Neural Network Configuration (`nn_config`)

Controls Neural Network architecture and regularization parameters.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `hidden_dims` | list[int] | [256, 128, 64] | Each: 8-2048 | Hidden layer dimensions (reduced from [512, 256, 128, 64]) |
| `dropout` | float | 0.5 | 0.0-1.0 | Dropout rate for regularization (increased from 0.4) |
| `learning_rate` | float | 0.005 | 0.0-1.0 | Initial learning rate (reduced from 0.01) |
| `batch_size` | int | 64 | 1-1024 | Training batch size (increased from 32) |
| `epochs` | int | 200 | 1-1000 | Maximum training epochs |
| `weight_decay` | float | 0.1 | 0.0-1.0 | L2 regularization strength (increased from 0.01) |
| `label_smoothing` | float | 0.2 | 0.0-1.0 | Label smoothing factor (increased from 0.15) |
| `patience` | int | 20 | 1-100 | Early stopping patience in epochs (increased from 15) |
| `lr_scheduler_patience` | int | 5 | - | Epochs to wait before reducing learning rate |
| `lr_scheduler_factor` | float | 0.5 | 0.0-1.0 | Factor to reduce learning rate by |
| `min_lr` | float | 1e-6 | - | Minimum learning rate threshold |

**Purpose:** Enhanced regularization to improve generalization and reduce train-validation gap.

**Impact on Overfitting:**
- Smaller `hidden_dims` reduce model capacity
- Higher `dropout` increases regularization
- Lower `learning_rate` prevents overfitting to training data
- Larger `batch_size` provides more stable gradients
- Higher `weight_decay` strengthens L2 regularization
- Higher `label_smoothing` prevents overconfident predictions
- Higher `patience` allows more time to find optimal weights

---

### 3. Feature Selection Configuration (`feature_selection_config`)

Controls dimensionality reduction to focus on informative features.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `methods` | list[str] | ['importance', 'correlation', 'mutual_info'] | - | Feature selection methods to use |
| `importance_threshold_percentile` | int | 25 | 0-100 | Percentile threshold for feature importance |
| `correlation_threshold` | float | 0.9 | 0.0-1.0 | Correlation threshold for redundancy removal |
| `min_features` | int | 30 | ≥1 | Minimum number of features to keep |
| `enabled` | bool | true | - | Whether feature selection is enabled |

**Valid Methods:**
- `'importance'`: Random Forest feature importance
- `'correlation'`: Correlation-based redundancy removal
- `'mutual_info'`: Mutual information with target

**Purpose:** Reduce feature dimensionality to prevent fitting to noise and improve model focus.

**Impact on Overfitting:**
- Removes low-importance features (below 25th percentile)
- Removes redundant features (correlation > 0.9)
- Maintains minimum features to preserve information
- Reduces model complexity and training time

---

### 4. Data Augmentation Configuration (`augmentation_config`)

Controls synthetic data generation for small datasets.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `threshold` | int | 300 | ≥10 | Minimum samples before augmentation is applied |
| `noise_std` | float | 0.01 | 0.0-1.0 | Standard deviation for Gaussian noise |
| `smote_k_neighbors` | int | 5 | 1-20 | Number of neighbors for SMOTE |
| `max_augmentation_ratio` | float | 2.0 | 1.0-10.0 | Maximum ratio of augmented to original data |
| `enabled` | bool | true | - | Whether data augmentation is enabled |

**Purpose:** Generate synthetic training samples to improve model generalization on small datasets.

**Impact on Overfitting:**
- Triggers when training samples < 300
- Adds controlled noise to create variations
- Balances class distributions using SMOTE
- Prevents models from memorizing limited training examples

**Augmentation Techniques:**
1. **Gaussian Noise**: Adds small random variations to features
2. **SMOTE**: Synthetic Minority Over-sampling Technique for class balancing

---

### 5. Cross-Validation Configuration (`cv_config`)

Controls k-fold cross-validation for robust model evaluation.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `n_folds` | int | 5 | 2-20 | Number of cross-validation folds |
| `stratified` | bool | true | - | Whether to use stratified splitting |
| `stability_threshold` | float | 0.15 | 0.0-1.0 | Standard deviation threshold for stability check |
| `enabled` | bool | true | - | Whether cross-validation is enabled |
| `random_state` | int | 42 | - | Random seed for reproducibility |

**Purpose:** Evaluate model performance across multiple data splits to detect overfitting and instability.

**Impact on Overfitting:**
- Provides robust performance estimates
- Detects models with high variance (unstable)
- Flags models with std > 0.15 as unstable
- Helps select hyperparameters that generalize well

**Cross-Validation Process:**
1. Split data into k folds
2. Train on k-1 folds, validate on 1 fold
3. Repeat k times
4. Report mean and standard deviation of metrics

---

### 6. Overfitting Monitor Configuration (`monitor_config`)

Controls overfitting detection and reporting during training.

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `warning_threshold` | float | 0.15 | 0.0-1.0 | Train-val gap threshold for overfitting warning (15%) |
| `generate_curves` | bool | true | - | Whether to generate learning curve plots |
| `save_metrics` | bool | true | - | Whether to save metrics to JSON |
| `curves_dir` | str | 'models/trained' | - | Directory to save learning curves |
| `metrics_dir` | str | 'models/trained' | - | Directory to save metrics |

**Purpose:** Track and report overfitting metrics during and after training.

**Impact on Overfitting:**
- Calculates train-validation accuracy gap
- Flags models with gap > 15%
- Generates learning curves for visual inspection
- Saves metrics for historical tracking

**Overfitting Detection:**
- Gap = Train Accuracy - Validation Accuracy
- Gap > 15% indicates overfitting
- Learning curves show divergence between train and validation

---

## Configuration Validation

All configuration parameters are validated when loaded or modified. Validation checks include:

- **Range checks**: Ensures values are within valid ranges
- **Type checks**: Ensures correct data types
- **Logical checks**: Ensures parameters make sense together

### Validation Example

```python
config = load_config('models/anti_overfitting_config.json')

if config.validate():
    print("Configuration is valid")
else:
    print("Configuration has errors - check logs")
```

### Common Validation Errors

1. **Out of range values**: e.g., `dropout = 1.5` (must be 0.0-1.0)
2. **Invalid method names**: e.g., `methods = ['invalid_method']`
3. **Logical inconsistencies**: e.g., `min_features > total_features`

---

## Best Practices

### 1. Start with Defaults

The default configuration is tuned to reduce overfitting based on empirical testing. Start with defaults and adjust incrementally.

### 2. Monitor Train-Val Gap

Always check the train-validation accuracy gap:
- Gap < 10%: Good generalization
- Gap 10-15%: Acceptable
- Gap > 15%: Overfitting detected

### 3. Adjust Based on Dataset Size

**Small datasets (< 300 samples):**
- Enable data augmentation
- Use stronger regularization
- Reduce model complexity

**Large datasets (> 1000 samples):**
- Can use larger models
- May reduce regularization slightly
- Feature selection becomes more important

### 4. Cross-Validation is Essential

Always use cross-validation to:
- Get robust performance estimates
- Detect unstable models
- Select hyperparameters

### 5. Iterative Tuning

1. Start with default configuration
2. Train and evaluate models
3. Check overfitting metrics
4. Adjust one configuration section at a time
5. Re-validate and compare results

---

## Configuration Presets

### Conservative (Maximum Anti-Overfitting)

```json
{
  "rf_config": {
    "max_depth": 10,
    "min_samples_split": 30,
    "min_samples_leaf": 15,
    "max_samples": 0.7
  },
  "nn_config": {
    "hidden_dims": [128, 64],
    "dropout": 0.6,
    "weight_decay": 0.15,
    "label_smoothing": 0.25
  }
}
```

### Balanced (Default)

Use the default configuration provided in `models/anti_overfitting_config.json`.

### Aggressive (Minimum Constraints)

```json
{
  "rf_config": {
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_samples": 0.9
  },
  "nn_config": {
    "hidden_dims": [512, 256, 128],
    "dropout": 0.3,
    "weight_decay": 0.05,
    "label_smoothing": 0.1
  }
}
```

---

## Troubleshooting

### Problem: Models still overfitting (gap > 15%)

**Solutions:**
1. Increase regularization:
   - RF: Reduce `max_depth`, increase `min_samples_split`
   - NN: Increase `dropout`, increase `weight_decay`
2. Enable feature selection if disabled
3. Reduce model complexity:
   - RF: Reduce `n_estimators`
   - NN: Reduce `hidden_dims`

### Problem: Models underfitting (low train and val accuracy)

**Solutions:**
1. Reduce regularization:
   - RF: Increase `max_depth`, reduce `min_samples_split`
   - NN: Reduce `dropout`, reduce `weight_decay`
2. Increase model complexity:
   - RF: Increase `n_estimators`
   - NN: Increase `hidden_dims`
3. Check if feature selection is too aggressive

### Problem: High cross-validation variance (std > 0.15)

**Solutions:**
1. Increase dataset size through augmentation
2. Simplify model to reduce variance
3. Increase `n_folds` for more robust estimates
4. Check for data quality issues

### Problem: Training too slow

**Solutions:**
1. Reduce `n_estimators` for Random Forest
2. Reduce `epochs` or `batch_size` for Neural Networks
3. Disable cross-validation during development
4. Reduce `n_folds` from 5 to 3

---

## Integration with Existing Code

### Random Forest Model

```python
from models.config import load_config
from models.random_forest_model import RandomForestSMCModel

config = load_config()
rf_config = config.rf_config

model = RandomForestSMCModel(symbol='EURUSD')
model.train(
    X_train, y_train, X_val, y_val,
    n_estimators=rf_config.n_estimators,
    max_depth=rf_config.max_depth,
    min_samples_split=rf_config.min_samples_split,
    min_samples_leaf=rf_config.min_samples_leaf,
    max_features=rf_config.max_features,
    max_samples=rf_config.max_samples
)
```

### Neural Network Model

```python
from models.config import load_config
from models.neural_network_model import NeuralNetworkSMCModel

config = load_config()
nn_config = config.nn_config

model = NeuralNetworkSMCModel(symbol='EURUSD')
model.train(
    X_train, y_train, X_val, y_val,
    hidden_dims=nn_config.hidden_dims,
    dropout=nn_config.dropout,
    learning_rate=nn_config.learning_rate,
    batch_size=nn_config.batch_size,
    epochs=nn_config.epochs,
    weight_decay=nn_config.weight_decay,
    patience=nn_config.patience
)
```

### Feature Selection

```python
from models.config import load_config
from models.base_model import FeatureSelector

config = load_config()
fs_config = config.feature_selection_config

if fs_config.enabled:
    selector = FeatureSelector(methods=fs_config.methods)
    selector.fit(X_train, y_train, feature_names)
    X_train_selected = selector.transform(X_train)
```

---

## Version History

### Version 1.0 (Current)
- Initial configuration management system
- All six configuration sections implemented
- Validation and error handling
- JSON file support
- Comprehensive documentation

---

## Support

For questions or issues with configuration:
1. Check validation error messages
2. Review this documentation
3. Examine default configuration file
4. Test with `python models/config.py`

---

## References

- Design Document: `.kiro/specs/anti-overfitting-enhancement/design.md`
- Requirements Document: `.kiro/specs/anti-overfitting-enhancement/requirements.md`
- Implementation Tasks: `.kiro/specs/anti-overfitting-enhancement/tasks.md`
