"""
Production-Ready Ensemble Modeling Pipeline
- Integrates with existing multi-timeframe SMC pipeline
- Implements multiple ensemble strategies with proper validation
- Includes model persistence and inference optimization
"""

from enhanced_multitf_pipeline import EnhancedSMC_MLP, load_and_engineer_multitf_features, standardize_features
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import os
import platform
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced model


def detect_device():
    """
    Detect Intel integrated GPU (optional) and determine optimal device for LightGBM
    """
    processor_info = platform.processor().lower()
    platform_info = platform.platform().lower()

    if "intel" in processor_info or "uhd" in platform_info or "iris" in platform_info:
        print(
            "[LightGBM] Intel integrated GPU detected, defaulting to CPU optimization.")
        return "cpu"  # Intel iGPU performance better on CPU mode

    return "gpu" if os.environ.get("LGBM_USE_GPU", "0") == "1" else "cpu"


class EnsembleManager:
    """
    Manages multiple models and ensemble strategies for SMC prediction
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.ensemble_weights = {}
        self.feature_scalers = {}
        self.performance_history = {}

    def create_model_variants(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Create multiple model variants with different architectures and parameters
        """
        models = {}

        # Neural Network Variants
        nn_configs = [
            {'name': 'deep_nn', 'hidden_dims': [
                256, 128, 64, 32], 'dropout': 0.4, 'lr': 1e-3},
            {'name': 'wide_nn', 'hidden_dims': [
                512, 256, 128], 'dropout': 0.3, 'lr': 8e-4},
            {'name': 'compact_nn', 'hidden_dims': [
                128, 64], 'dropout': 0.2, 'lr': 1e-3},
            {'name': 'regularized_nn', 'hidden_dims': [
                128, 96, 64, 32], 'dropout': 0.5, 'lr': 5e-4}
        ]

        for config in nn_configs:
            model = EnhancedSMC_MLP(
                input_dim=features.shape[1],
                num_classes=len(np.unique(labels)),
                hidden_dims=config['hidden_dims'],
                dropout_p=config['dropout']
            )
            models[config['name']] = {
                'model': model,
                'type': 'neural_network',
                'config': config
            }

        # Detect optimal device for LightGBM
        device_type = detect_device()

        # Traditional ML Models
        traditional_models = {
            'random_forest_deep': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'random_forest_wide': RandomForestClassifier(
                n_estimators=500, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=123
            ),
            'gradient_boosting': LGBMClassifier(
                n_estimators=500,         # number of boosting rounds
                learning_rate=0.05,       # step size shrinkage
                max_depth=4,              # limit tree depth
                subsample=0.8,            # sample rows
                colsample_bytree=0.8,     # sample features
                n_jobs=-1,                # use all CPU cores
                device=device_type,       # auto-select based on system
                random_state=42,
                verbose=-1                # suppress training output
            ),
            'logistic_regression': LogisticRegression(
                max_iter=2000, C=0.1, penalty='l2', random_state=789
            )
        }

        for name, model in traditional_models.items():
            models[name] = {
                'model': model,
                'type': 'sklearn',
                'config': {'name': name}
            }

        return models

    def train_neural_network(self, model: EnhancedSMC_MLP, features: np.ndarray,
                             labels: np.ndarray, config: Dict) -> Tuple[float, Dict]:
        """
        Train a neural network with proper validation
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Convert to tensors
        X = torch.tensor(features, dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)

        # Training setup
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        # Class weights for imbalanced data
        class_counts = np.bincount(labels)
        class_weights = len(labels) / (len(class_counts) * class_counts)
        class_weights = torch.tensor(
            class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training parameters
        epochs = 50
        batch_size = 256
        best_val_acc = 0
        patience_counter = 0

        # Split for validation
        n = len(features)
        val_size = int(0.2 * n)
        train_size = n - val_size

        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        training_history = {'train_acc': [], 'val_acc': [],
                            'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0

            # Shuffle training data
            perm = torch.randperm(train_size, device=device)
            X_train_shuf = X_train[perm]
            y_train_shuf = y_train[perm]

            for i in range(0, train_size, batch_size):
                batch_X = X_train_shuf[i:i+batch_size]
                batch_y = y_train_shuf[i:i+batch_size]

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for i in range(0, val_size, batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]

                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)

                    val_loss += loss.item()
                    val_correct += (logits.argmax(dim=1) ==
                                    batch_y).sum().item()

            # Calculate metrics
            train_acc = train_correct / train_size
            val_acc = val_correct / val_size
            avg_train_loss = train_loss / (train_size // batch_size + 1)
            avg_val_loss = val_loss / (val_size // batch_size + 1)

            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 15:
                break

        return best_val_acc, training_history

    def train_all_models(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
        """
        Train all model variants and collect performance metrics
        """
        print("Training ensemble of models...")

        # Standardize features
        features_norm, feature_mean, feature_std = standardize_features(
            features)
        self.feature_scalers['mean'] = feature_mean
        self.feature_scalers['std'] = feature_std

        # Create model variants
        model_variants = self.create_model_variants(features_norm, labels)

        results = {}

        for model_name, model_info in model_variants.items():
            print(f"\nTraining {model_name}...")

            if model_info['type'] == 'neural_network':
                # Train neural network
                accuracy, history = self.train_neural_network(
                    model_info['model'], features_norm, labels, model_info['config']
                )
                results[model_name] = {
                    'model': model_info['model'],
                    'type': 'neural_network',
                    'accuracy': accuracy,
                    'history': history,
                    'config': model_info['config']
                }

            elif model_info['type'] == 'sklearn':
                # Train sklearn model
                model = model_info['model']

                # Special handling for LightGBM
                if isinstance(model, LGBMClassifier):
                    print(f"Training LightGBM (device={model.device})...")
                    start_time = time.time()

                    # Cross-validation for LightGBM
                    cv_scores = []
                    skf = StratifiedKFold(
                        n_splits=5, shuffle=True, random_state=42)

                    for train_idx, val_idx in skf.split(features_norm, labels):
                        X_train_cv, X_val_cv = features_norm[train_idx], features_norm[val_idx]
                        y_train_cv, y_val_cv = labels[train_idx], labels[val_idx]

                        model.fit(X_train_cv, y_train_cv)
                        val_pred = model.predict(X_val_cv)
                        cv_scores.append(accuracy_score(y_val_cv, val_pred))

                    # Final training on full dataset
                    model.fit(features_norm, labels)

                    training_time = time.time() - start_time
                    print(
                        f"✅ LightGBM finished in {training_time:.2f}s using {model.device.upper()}")

                    # Save model for later use
                    try:
                        # Ensure directory exists before saving
                        save_path = "Python/Model_output/gradient_boosting.txt"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        model.booster_.save_model(save_path)
                        print(f"→ LightGBM model saved to {save_path}")
                    except Exception as e:
                        print(f"→ Warning: Could not save LightGBM model: {e}")

                else:
                    # Standard sklearn model training
                    cv_scores = []
                    skf = StratifiedKFold(
                        n_splits=5, shuffle=True, random_state=42)

                    for train_idx, val_idx in skf.split(features_norm, labels):
                        X_train_cv, X_val_cv = features_norm[train_idx], features_norm[val_idx]
                        y_train_cv, y_val_cv = labels[train_idx], labels[val_idx]

                        model.fit(X_train_cv, y_train_cv)
                        val_pred = model.predict(X_val_cv)
                        cv_scores.append(accuracy_score(y_val_cv, val_pred))

                    # Final training on full dataset
                    model.fit(features_norm, labels)

                results[model_name] = {
                    'model': model,
                    'type': 'sklearn' if not isinstance(model, LGBMClassifier) else 'tree_ensemble',
                    'accuracy': np.mean(cv_scores),
                    'cv_scores': cv_scores,
                    'config': model_info['config']
                }

            print(f"  → {model_name}: {results[model_name]['accuracy']:.4f}")

        self.models = results
        return results

    def calculate_ensemble_weights(self, validation_features: np.ndarray,
                                   validation_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate optimal ensemble weights based on validation performance
        """
        print("\nCalculating ensemble weights...")

        # Normalize validation features
        val_features_norm = (
            validation_features - self.feature_scalers['mean']) / self.feature_scalers['std']

        # Get predictions from all models
        predictions = {}
        accuracies = {}

        for model_name, model_info in self.models.items():
            if model_info['type'] == 'neural_network':
                model = model_info['model']
                model.eval()
                
                # Get the device the model is on
                device = next(model.parameters()).device
                
                with torch.no_grad():
                    X_tensor = torch.tensor(
                        val_features_norm, dtype=torch.float32, device=device)
                    logits = model(X_tensor)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                model = model_info['model']
                
                # Use numpy arrays directly (models were trained with numpy arrays)
                # Suppress sklearn feature name warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                    
                    preds = model.predict(val_features_norm)
                    probs = model.predict_proba(val_features_norm)

            predictions[model_name] = {'preds': preds, 'probs': probs}
            accuracies[model_name] = accuracy_score(validation_labels, preds)

        # Calculate weights based on performance (softmax of accuracies)
        acc_values = np.array(list(accuracies.values()))
        weights = np.exp(acc_values * 10) / \
            np.sum(np.exp(acc_values * 10))  # Temperature scaling

        ensemble_weights = dict(zip(accuracies.keys(), weights))

        print("Model weights:")
        for model_name, weight in ensemble_weights.items():
            print(
                f"  {model_name}: {weight:.4f} (acc: {accuracies[model_name]:.4f})")

        self.ensemble_weights = ensemble_weights
        return ensemble_weights

    def predict_ensemble(self, features: np.ndarray, method: str = 'weighted_average') -> np.ndarray:
        """
        Make ensemble predictions using specified method
        """
        # Normalize features
        features_norm = (
            features - self.feature_scalers['mean']) / self.feature_scalers['std']

        # Collect predictions from all models
        all_probs = []
        model_names = []

        for model_name, model_info in self.models.items():
            if model_info['type'] == 'neural_network':
                model = model_info['model']
                model.eval()
                
                # Get the device the model is on
                device = next(model.parameters()).device
                
                with torch.no_grad():
                    X_tensor = torch.tensor(features_norm, dtype=torch.float32, device=device)
                    logits = model(X_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                model = model_info['model']
                
                # Use numpy arrays directly (models were trained with numpy arrays)
                # Suppress sklearn feature name warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                    
                    probs = model.predict_proba(features_norm)

            all_probs.append(probs)
            model_names.append(model_name)

        # Shape: (n_models, n_samples, n_classes)
        all_probs = np.array(all_probs)

        if method == 'simple_average':
            # Simple averaging
            ensemble_probs = np.mean(all_probs, axis=0)

        elif method == 'weighted_average':
            # Weighted averaging based on validation performance
            weights = np.array([self.ensemble_weights[name]
                               for name in model_names])
            weights = weights.reshape(-1, 1, 1)  # Broadcast shape
            ensemble_probs = np.sum(all_probs * weights, axis=0)

        elif method == 'majority_vote':
            # Majority voting on predictions
            # Shape: (n_models, n_samples)
            all_preds = np.argmax(all_probs, axis=2)
            ensemble_preds = []
            for i in range(all_preds.shape[1]):
                votes = all_preds[:, i]
                ensemble_preds.append(np.bincount(votes).argmax())
            return np.array(ensemble_preds)

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return np.argmax(ensemble_probs, axis=1)

    def evaluate_ensemble(self, test_features: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance using different methods
        """
        results = {}

        methods = ['simple_average', 'weighted_average', 'majority_vote']

        for method in methods:
            preds = self.predict_ensemble(test_features, method=method)
            accuracy = accuracy_score(test_labels, preds)
            results[method] = accuracy
            print(f"{method}: {accuracy:.4f}")

        return results

    def save_ensemble(self, save_path: Path):
        """
        Save the entire ensemble to disk
        """
        save_path.mkdir(parents=True, exist_ok=True)

        # Save ensemble metadata
        metadata = {
            'ensemble_weights': self.ensemble_weights,
            'feature_scalers': {
                'mean': self.feature_scalers['mean'].tolist(),
                'std': self.feature_scalers['std'].tolist()
            },
            'model_configs': {name: info['config'] for name, info in self.models.items()}
        }

        with open(save_path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save individual models
        for model_name, model_info in self.models.items():
            if model_info['type'] == 'neural_network':
                torch.save(model_info['model'].state_dict(),
                           save_path / f'{model_name}_state_dict.pth')
            else:
                with open(save_path / f'{model_name}_sklearn.pkl', 'wb') as f:
                    pickle.dump(model_info['model'], f)

        print(f"Ensemble saved to {save_path}")


def main():
    """
    Main execution pipeline for ensemble modeling
    """
    print("=== Production Ensemble Pipeline ===")

    # Configuration
    config = {
        'data_path': 'merged_M15_multiTF.csv',
        'ensemble_methods': ['simple_average', 'weighted_average', 'majority_vote'],
        'save_path': Path('Files/black_ice/ensemble_models')
    }

    # Load and prepare data
    print("Loading multi-timeframe data...")
    features, labels, symbols = load_and_engineer_multitf_features(
        config['data_path'])

    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print(f"Label distribution: {np.bincount(labels)}")

    # Split data for ensemble training and evaluation
    n = len(features)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    train_features = features[:train_size]
    train_labels = labels[:train_size]
    val_features = features[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_features = features[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Initialize ensemble manager
    ensemble = EnsembleManager(config)

    # Train all models
    model_results = ensemble.train_all_models(train_features, train_labels)

    # Calculate ensemble weights using validation set
    ensemble_weights = ensemble.calculate_ensemble_weights(
        val_features, val_labels)

    # Evaluate ensemble on test set
    print(f"\n=== Ensemble Evaluation on Test Set ===")
    ensemble_results = ensemble.evaluate_ensemble(test_features, test_labels)

    # Compare with individual model performance
    print(f"\n=== Individual Model Performance ===")
    for model_name, model_info in ensemble.models.items():
        print(f"{model_name}: {model_info['accuracy']:.4f}")

    # Save ensemble
    ensemble.save_ensemble(config['save_path'])

    print(f"\n=== Summary ===")
    print(
        f"Best individual model: {max(model_results.items(), key=lambda x: x[1]['accuracy'])}")
    print(
        f"Best ensemble method: {max(ensemble_results.items(), key=lambda x: x[1])}")

    return ensemble, ensemble_results


if __name__ == "__main__":
    ensemble, results = main()
