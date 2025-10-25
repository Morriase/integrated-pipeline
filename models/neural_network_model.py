"""
Neural Network (MLP) Model for SMC Trade Outcome Prediction

Key Learning Objectives (from WHATS_NEEDED.md):
- Learn non-linear feature interactions
- Capture complex decision boundaries
- Model multiplicative effects
- Extract high-level patterns from raw features
"""

import os
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler
from models.base_model import BaseSMCModel, TrainingMonitor
from models.data_augmentation import DataAugmenter
from models.overfitting_monitor import OverfittingMonitor


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification with Batch Normalization"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 3,
                 dropout: float = 0.2):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # BatchNorm with higher momentum for smoother training
            layers.append(nn.BatchNorm1d(
                hidden_dim, 
                momentum=0.1,  # Smoother running stats updates
                eps=1e-5,
                track_running_stats=True
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        
        # He initialization for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform to prevent exploding gradients"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # The network automatically handles training vs eval mode for batch norm
        # When model.train() is called, batch norm uses batch statistics
        # When model.eval() is called, batch norm uses running statistics
        return self.network(x)


class NeuralNetworkSMCModel(BaseSMCModel):
    """
    Neural Network (MLP) classifier for SMC trade prediction

    Strengths:
    - Learns complex non-linear patterns
    - Captures feature interactions
    - Flexible architecture
    - GPU acceleration
    """

    def __init__(self, symbol: str, target_col: str = 'TBM_Label'):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        super().__init__('NeuralNetwork', symbol, target_col)
        self.scaler = StandardScaler()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              hidden_dims: List[int] = [128, 64, 32],  # Deeper network for better learning
              dropout: float = 0.2,  # Lower dropout for smoother training
              learning_rate: float = 0.003,  # Higher initial LR with scheduler
              batch_size: int = 64,  # Larger batches for stable gradients
              epochs: int = 200,  # Maximum epochs
              patience: int = 25,  # More patience for convergence
              weight_decay: float = 0.0005,  # Light L2 regularization
              **kwargs) -> Dict:
        """
        Train Neural Network model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum epochs
            patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        print(f"\n🧠 Training Neural Network for {self.symbol}...")
        print(f"  Device: {self.device}")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Check for problematic features
        constant_features = []
        for i in range(X_train.shape[1]):
            if np.std(X_train[:, i]) == 0:
                constant_features.append(i)
        
        if constant_features:
            print(f"  ⚠️ WARNING: {len(constant_features)} constant features detected!")
            print(f"     These features will not help learning.")
        
        # Check for NaN or inf
        if np.isnan(X_train).any():
            nan_count = np.isnan(X_train).sum()
            print(f"  ⚠️ WARNING: {nan_count} NaN values in training data!")
        
        if np.isinf(X_train).any():
            inf_count = np.isinf(X_train).sum()
            print(f"  ⚠️ WARNING: {inf_count} infinite values in training data!")
        
        print(f"  Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, hidden_dims))} -> 3")

        # Data augmentation for small datasets
        if len(X_train) < 300:
            print(f"\n  Dataset has < 300 samples. Applying data augmentation...")
            augmenter = DataAugmenter(noise_std=0.01, smote_k_neighbors=5)
            X_train, y_train = augmenter.augment(X_train, y_train, threshold=300)
            print(f"  Augmented training samples: {len(X_train):,}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Check class distribution
        unique_labels = np.unique(y_train)
        print(f"\n  Class distribution:")
        for label in unique_labels:
            count = np.sum(y_train == label)
            pct = count / len(y_train) * 100
            print(f"    {label:>4}: {count:>6} ({pct:>5.1f}%)")
        
        # Map labels for training
        num_classes = len(unique_labels)
        sorted_labels = sorted(unique_labels)
        
        # Create mapping: input labels -> 0, 1, 2...
        label_map = {label: i for i, label in enumerate(sorted_labels)}
        
        # Reverse mapping for predictions
        # If cleaned data (0, 1), map back to original (-1, 1)
        # If original data (-1, 0, 1), keep as is
        if set(sorted_labels) == {0, 1}:
            # Binary cleaned data: 0=Loss, 1=Win -> map to -1, 1
            self.label_map_reverse = {0: -1, 1: 1}
            print(f"\n  Detected cleaned binary data (0, 1)")
            print(f"  Will map predictions back to original labels (-1, 1)")
        else:
            # Original labels or 3-class
            self.label_map_reverse = {i: label for label, i in label_map.items()}
        
        output_dim = num_classes
        
        print(f"\n  Using {num_classes}-class classification")
        print(f"  Training label mapping: {label_map}")
        print(f"  Prediction label mapping: {self.label_map_reverse}")
        
        y_train_mapped = np.array([label_map[y] for y in y_train])

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_mapped)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model with He initialization (for ReLU)
        self.model = MLPClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            output_dim=output_dim,  # Use detected number of classes
            dropout=dropout
        ).to(self.device)

        # Apply He (Kaiming) initialization for ReLU networks
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.model.apply(init_weights)

        # Calculate class weights for imbalanced data
        unique_labels, counts = np.unique(y_train_mapped, return_counts=True)
        class_weights = len(y_train_mapped) / (len(unique_labels) * counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"  Class distribution: {dict(zip(unique_labels, counts))}")
        print(f"  Class weights: {dict(zip(unique_labels, class_weights))}")
        
        # Loss with light label smoothing + class weights
        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,  # Light smoothing for better convergence
            weight=class_weights_tensor  # Handle class imbalance
        )
        
        # AdamW with gradient clipping for stable training
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Standard Adam betas
            eps=1e-8
        )

        # Cosine annealing with warm restarts for smooth convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        # label_map_reverse already set earlier - don't overwrite it!

        # Initialize overfitting monitor
        monitor = OverfittingMonitor(warning_threshold=0.15)
        
        # Initialize training monitor for early warnings (Task 10.1)
        from datetime import datetime
        training_monitor = TrainingMonitor()
        training_start_time = datetime.now()

        # Training loop
        history = {'train_loss': [], 'train_acc': [],
                   'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_model_state = None  # Initialize best model state

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(
                    self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stable training
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                
                # Warn on high gradients
                if grad_norm > 10.0:
                    training_monitor.check_exploding_gradients(grad_norm.item(), threshold=10.0, epoch=epoch + 1)
                
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Check for NaN loss (critical - stop immediately)
            if training_monitor.check_nan_loss(train_loss, epoch + 1):
                print(f"\n  ❌ Training stopped due to NaN loss at epoch {epoch+1}")
                break

            # Validation phase and monitoring
            if X_val is not None and y_val is not None:
                # Filter validation data to match training classes
                mask = np.isin(y_val, list(label_map.keys()))
                X_val_filtered = X_val[mask]
                y_val_filtered = y_val[mask]
                
                if len(y_val_filtered) < len(y_val):
                    removed = len(y_val) - len(y_val_filtered)
                    print(f"  Filtered {removed} validation samples not in training classes")
                
                X_val_scaled = self.scaler.transform(X_val_filtered)
                y_val_mapped = np.array([label_map[y] for y in y_val_filtered])

                self.model.eval()
                with torch.no_grad():
                    val_X = torch.FloatTensor(X_val_scaled).to(self.device)
                    val_y = torch.LongTensor(y_val_mapped).to(self.device)

                    val_outputs = self.model(val_X)
                    val_loss = criterion(val_outputs, val_y).item()

                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted ==
                               val_y).sum().item() / len(val_y)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Update overfitting monitor
                monitor.update(
                    epoch=epoch,
                    train_metrics={'accuracy': train_acc, 'loss': train_loss},
                    val_metrics={'accuracy': val_acc, 'loss': val_loss}
                )
                
                # Training monitor checks (Task 10.1 - Requirements 9.1, 9.2, 9.3, 9.4, 9.5)
                # Check for severe overfitting
                training_monitor.check_overfitting(train_acc, val_acc, epoch + 1)
                
                # Check for validation loss divergence
                training_monitor.check_divergence(history['val_loss'], patience=10)
                
                # Check for NaN validation loss
                if training_monitor.check_nan_loss(val_loss, epoch + 1):
                    print(f"\n  ❌ Training stopped due to NaN validation loss at epoch {epoch+1}")
                    break
                
                # Check for training timeout
                if training_monitor.check_timeout(training_start_time, max_minutes=10):
                    print(f"\n  ⏱️ Training stopped due to timeout at epoch {epoch+1}")
                    break
                
                # Stop on critical warnings
                if training_monitor.has_critical_warnings():
                    print(f"\n  ❌ Training stopped due to critical warnings at epoch {epoch+1}")
                    break

                # Step scheduler (CosineAnnealingWarmRestarts steps every epoch)
                scheduler.step()

                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
                    # Restore best model if it was saved
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")

        self.training_history = history
        self.is_trained = True
        
        # Store training warnings in history
        history['training_warnings'] = training_monitor.get_warnings()

        # Generate and save learning curves to Kaggle-friendly path
        if Path('/kaggle/working').exists():
            curves_dir = '/kaggle/working/Training_Images'
        else:
            curves_dir = 'Training_Images'
        
        os.makedirs(curves_dir, exist_ok=True)
        curves_path = os.path.join(curves_dir, f'{self.symbol}_NN_learning_curves.png')
        monitor.generate_learning_curves(curves_path)

        # Save overfitting metrics to JSON
        metrics_path = os.path.join(curves_dir, f'{self.symbol}_NN_overfitting_metrics.json')
        monitor.save_to_json(metrics_path)

        # Print overfitting summary
        monitor.print_summary()
        
        # Print training monitor summary
        warnings = training_monitor.get_warnings()
        if warnings:
            print(f"\n  ⚠️ Training Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"    - {warning}")

        print(f"\n  Final Train Accuracy: {history['train_acc'][-1]:.3f}")
        if history['val_acc']:
            print(f"  Final Val Accuracy:   {history['val_acc'][-1]:.3f}")
            gap = history['train_acc'][-1] - history['val_acc'][-1]
            print(f"  Train-Val Gap:        {gap:.3f} ({gap*100:.1f}%)")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_mapped = predicted.cpu().numpy()

        # Convert back to original labels using stored mapping
        # label_map_reverse maps: model output (0,1) -> original labels (-1, 1)
        y_pred = np.array([self.label_map_reverse[int(y)] for y in y_pred_mapped])
        
        # Verify predictions are in expected range
        unique_preds = np.unique(y_pred)
        if len(unique_preds) > 2 or (0 in unique_preds and len(unique_preds) > 1):
            print(f"  ⚠️ WARNING: Unexpected predictions: {unique_preds}")
            print(f"  label_map_reverse: {self.label_map_reverse}")
            print(f"  Raw model outputs (first 10): {y_pred_mapped[:10]}")

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distributions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba


# Example usage
if __name__ == "__main__":
    """
    Example: Train Neural Network for EURUSD
    """

    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
        exit(1)

    # Initialize model
    model = NeuralNetworkSMCModel(symbol='EURUSD')

    # Load data
    train_df, val_df, test_df = model.load_data(
        train_path='Data/processed_smc_data_train.csv',
        val_path='Data/processed_smc_data_val.csv',
        test_path='Data/processed_smc_data_test.csv',
        exclude_timeout=False
    )

    # Prepare features
    X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
    X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
    X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        patience=15
    )

    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')

    # Save model
    model.save_model('models/trained')

    print("\n✅ Neural Network training complete!")
