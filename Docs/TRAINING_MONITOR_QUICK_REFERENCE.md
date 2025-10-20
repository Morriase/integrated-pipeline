# TrainingMonitor Quick Reference

## Overview
The `TrainingMonitor` class provides real-time early warning detection during model training. It monitors for common training issues and can automatically stop training when critical problems are detected.

## Location
`models/base_model.py` - `TrainingMonitor` class

## Features

### 1. Overfitting Detection
**Method:** `check_overfitting(train_acc, val_acc, epoch=None)`

Detects severe overfitting when:
- Training accuracy > 95% AND
- Validation accuracy < 60%

```python
monitor = TrainingMonitor()
if monitor.check_overfitting(train_acc=0.96, val_acc=0.55, epoch=10):
    print("Severe overfitting detected!")
```

### 2. Validation Loss Divergence
**Method:** `check_divergence(val_losses, patience=10)`

Detects when validation loss increases for N consecutive epochs (default: 10).

```python
val_losses = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
if monitor.check_divergence(val_losses, patience=10):
    print("Validation loss is diverging!")
```

### 3. NaN/Inf Loss Detection (Critical)
**Method:** `check_nan_loss(loss, epoch=None)`

Detects NaN or Inf loss values and sets `should_stop=True` flag.

```python
if monitor.check_nan_loss(loss, epoch=5):
    print("Critical: NaN loss detected!")
    if monitor.should_stop:
        break  # Stop training immediately
```

### 4. Exploding Gradients
**Method:** `check_exploding_gradients(grad_norm, threshold=10.0, epoch=None)`

Detects when gradient norm exceeds threshold (default: 10.0).

```python
if monitor.check_exploding_gradients(grad_norm=15.0, epoch=5):
    print("Exploding gradients detected!")
```

### 5. Training Timeout
**Method:** `check_timeout(start_time, max_minutes=10)`

Detects when training exceeds maximum time limit (default: 10 minutes).

```python
from datetime import datetime

start_time = datetime.now()
# ... training loop ...
if monitor.check_timeout(start_time, max_minutes=10):
    print("Training timeout!")
```

## Usage Pattern

### Basic Usage
```python
from datetime import datetime
from models.base_model import TrainingMonitor

# Initialize monitor
monitor = TrainingMonitor()
start_time = datetime.now()

# Training loop
val_losses = []
for epoch in range(num_epochs):
    # ... training code ...
    
    # Check for issues
    monitor.check_overfitting(train_acc, val_acc, epoch)
    monitor.check_nan_loss(train_loss, epoch)
    monitor.check_exploding_gradients(grad_norm, epoch=epoch)
    
    val_losses.append(val_loss)
    monitor.check_divergence(val_losses)
    monitor.check_timeout(start_time)
    
    # Stop if critical issue detected
    if monitor.should_stop:
        print("Training stopped due to critical issue")
        break

# Get all warnings
warnings = monitor.get_warnings()
print(f"Training completed with {len(warnings)} warnings")
```

### Integration with Model Training
```python
class MyModel:
    def train(self, X_train, y_train, X_val, y_val):
        monitor = TrainingMonitor()
        start_time = datetime.now()
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training step
            train_loss, train_acc = self._train_epoch(X_train, y_train)
            
            # Validation step
            val_loss, val_acc = self._validate(X_val, y_val)
            val_losses.append(val_loss)
            
            # Monitor checks
            monitor.check_nan_loss(train_loss, epoch)
            monitor.check_overfitting(train_acc, val_acc, epoch)
            monitor.check_divergence(val_losses)
            monitor.check_timeout(start_time)
            
            # For PyTorch models, check gradients
            if hasattr(self, 'model'):
                grad_norm = self._compute_grad_norm()
                monitor.check_exploding_gradients(grad_norm, epoch=epoch)
            
            # Critical stop
            if monitor.should_stop:
                print(f"Training stopped at epoch {epoch}")
                break
        
        # Store warnings in training history
        self.training_history['warnings'] = monitor.get_warnings()
        self.training_history['has_critical_warnings'] = monitor.has_critical_warnings()
```

## API Reference

### Initialization
```python
monitor = TrainingMonitor()
```

### Check Methods
All check methods return `bool`:
- `True` if issue detected
- `False` if no issue

| Method | Parameters | Returns | Sets should_stop |
|--------|-----------|---------|------------------|
| `check_overfitting()` | train_acc, val_acc, epoch | bool | No |
| `check_divergence()` | val_losses, patience | bool | No |
| `check_nan_loss()` | loss, epoch | bool | **Yes** |
| `check_exploding_gradients()` | grad_norm, threshold, epoch | bool | No |
| `check_timeout()` | start_time, max_minutes | bool | No |

### Utility Methods
```python
# Get all warnings
warnings = monitor.get_warnings()  # Returns List[str]

# Check if critical warnings exist
is_critical = monitor.has_critical_warnings()  # Returns bool

# Reset for new training run
monitor.reset()
```

## Warning Messages

### Overfitting
```
"Epoch 10: Severe overfitting detected (train=96.00%, val=55.00%)"
```

### Divergence
```
"Validation loss divergence: increasing for 10 consecutive epochs"
```

### NaN Loss
```
"Epoch 5: NaN/Inf loss detected - STOPPING TRAINING"
```

### Exploding Gradients
```
"Epoch 5: Exploding gradient detected (norm=15.00 > 10.0)"
```

### Timeout
```
"Training timeout: exceeded 10 minutes (elapsed: 11.0 min)"
```

## Requirements Mapping

| Requirement | Method | Status |
|-------------|--------|--------|
| 9.1 - Overfitting alert | `check_overfitting()` | ✅ Implemented |
| 9.2 - Divergence alert | `check_divergence()` | ✅ Implemented |
| 9.3 - NaN loss stop | `check_nan_loss()` | ✅ Implemented |
| 9.4 - Gradient explosion | `check_exploding_gradients()` | ✅ Implemented |
| 9.5 - Timeout alert | `check_timeout()` | ✅ Implemented |

## Testing

Run unit tests:
```bash
python test_training_monitor.py
```

All tests verify:
- ✅ Correct detection of issues
- ✅ No false positives
- ✅ Edge case handling
- ✅ Warning storage and retrieval
- ✅ Critical flag management
- ✅ Reset functionality

## Next Steps

1. **Task 10.1**: Integrate into Neural Network training
   - Add monitor to `models/neural_network_model.py`
   - Check after each epoch
   - Stop on critical warnings

2. **Task 10.2**: Integrate into LSTM training
   - Add monitor to `models/lstm_model.py`
   - Check after each epoch
   - Stop on critical warnings

## Notes

- The monitor is **non-invasive** - it only observes and warns
- Only `check_nan_loss()` sets the critical `should_stop` flag
- All other checks are warnings that should be logged but don't force stop
- Warnings are accumulated and can be retrieved at any time
- Use `reset()` when starting a new training run with the same monitor instance
