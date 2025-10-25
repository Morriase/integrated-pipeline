"""
Base Model Class for SMC Trade Outcome Prediction
Provides common interface and utilities for all model types
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection class for dimensionality reduction
    
    Uses multiple methods to identify and select the most important features:
    - Random Forest feature importance
    - Mutual information
    - Correlation-based redundancy removal
    """
    
    def __init__(self, methods: List[str] = ['importance', 'correlation', 'mutual_info'],
                 importance_threshold_percentile: int = 25,
                 correlation_threshold: float = 0.9,
                 min_features: int = 30):
        """
        Initialize FeatureSelector
        
        Args:
            methods: List of selection methods to use
            importance_threshold_percentile: Percentile threshold for feature importance
            correlation_threshold: Correlation threshold for redundancy removal
            min_features: Minimum number of features to keep
        """
        self.methods = methods
        self.importance_threshold_percentile = importance_threshold_percentile
        self.correlation_threshold = correlation_threshold
        self.min_features = min_features
        
        self.selected_features_ = None
        self.selected_indices_ = None
        self.feature_scores_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> 'FeatureSelector':
        """
        Analyze features and determine selection criteria
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names
            
        Returns:
            self
        """
        print(f"\n🔍 Feature Selection: Analyzing {X.shape[1]} features...")
        
        self.feature_names_ = feature_names
        n_features = X.shape[1]
        
        # Initialize scores dictionary
        scores_dict = {name: 0.0 for name in feature_names}
        
        # Method 1: Random Forest feature importance
        if 'importance' in self.methods:
            print("  Computing Random Forest importance...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # Normalize to 0-1 range
            rf_importance = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min() + 1e-10)
            
            for i, name in enumerate(feature_names):
                scores_dict[name] += rf_importance[i]
        
        # Method 2: Mutual information
        if 'mutual_info' in self.methods:
            print("  Computing mutual information...")
            mi_scores = mutual_info_classif(X, y, random_state=42)
            
            # Normalize to 0-1 range
            mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
            
            for i, name in enumerate(feature_names):
                scores_dict[name] += mi_scores[i]
        
        # Average scores across methods
        n_methods = len([m for m in self.methods if m in ['importance', 'mutual_info']])
        if n_methods > 0:
            for name in scores_dict:
                scores_dict[name] /= n_methods
        
        # Convert to DataFrame for easier manipulation
        self.feature_scores_ = pd.DataFrame({
            'feature': feature_names,
            'score': [scores_dict[name] for name in feature_names]
        })
        
        # Step 1: Remove features below importance threshold
        threshold_value = np.percentile(self.feature_scores_['score'], self.importance_threshold_percentile)
        selected_features = self.feature_scores_[self.feature_scores_['score'] >= threshold_value]['feature'].tolist()
        
        print(f"  After importance filtering: {len(selected_features)} features (threshold: {threshold_value:.4f})")
        
        # Step 2: Remove correlated features
        if 'correlation' in self.methods and len(selected_features) > self.min_features:
            print("  Removing correlated features...")
            
            # Get indices of selected features
            selected_indices = [feature_names.index(f) for f in selected_features]
            X_selected = X[:, selected_indices]
            
            # Skip correlation if only 1 feature
            if X_selected.shape[1] <= 1:
                print(f"  Skipping correlation (only {X_selected.shape[1]} feature)")
            else:
                # Compute correlation matrix
                corr_matrix = np.corrcoef(X_selected.T)
                
                # Find correlated pairs
                to_remove = set()
                for i in range(len(selected_features)):
                    if selected_features[i] in to_remove:
                        continue
                    for j in range(i + 1, len(selected_features)):
                        if selected_features[j] in to_remove:
                            continue
                        if abs(corr_matrix[i, j]) > self.correlation_threshold:
                            # Remove the feature with lower importance score
                            score_i = scores_dict[selected_features[i]]
                            score_j = scores_dict[selected_features[j]]
                            if score_i < score_j:
                                to_remove.add(selected_features[i])
                            else:
                                to_remove.add(selected_features[j])
                
                selected_features = [f for f in selected_features if f not in to_remove]
                print(f"  After correlation filtering: {len(selected_features)} features (removed {len(to_remove)} correlated)")
        
        # Step 3: Enforce minimum feature threshold
        if len(selected_features) < self.min_features:
            print(f"  ⚠️ Only {len(selected_features)} features selected, enforcing minimum of {self.min_features}")
            # Select top features by score
            top_features = self.feature_scores_.nlargest(self.min_features, 'score')['feature'].tolist()
            selected_features = top_features
        
        # Store selected features and their indices
        self.selected_features_ = selected_features
        self.selected_indices_ = [feature_names.index(f) for f in selected_features]
        self.is_fitted_ = True
        
        print(f"  ✅ Final selection: {len(self.selected_features_)} features ({len(self.selected_features_)/n_features*100:.1f}% of original)")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection to dataset
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: List of feature names
            
        Returns:
            Transformed feature matrix with selected features only
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names
        
        Returns:
            List of selected feature names
        """
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
        
        return self.selected_features_
    
    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get importance scores for all features
        
        Returns:
            DataFrame with feature names and scores, sorted by score descending
        """
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
        
        return self.feature_scores_.sort_values('score', ascending=False).reset_index(drop=True)


class TrainingMonitor:
    """
    Real-time training monitoring and early warning system
    
    Monitors training progress and detects issues:
    - Severe overfitting (train >95%, val <60%)
    - Validation loss divergence (increasing for 10+ epochs)
    - NaN loss (immediate stop)
    - Exploding gradients (norm >10)
    - Training timeout (>10 minutes per symbol)
    """
    
    def __init__(self):
        """Initialize training monitor with empty warning storage"""
        self.warnings = []
        self.should_stop = False
    
    def check_overfitting(self, train_acc: float, val_acc: float, epoch: int = None) -> bool:
        """
        Check for severe overfitting (Requirement 9.1)
        
        Args:
            train_acc: Training accuracy (0-1 range)
            val_acc: Validation accuracy (0-1 range)
            epoch: Current epoch number (optional)
            
        Returns:
            True if severe overfitting detected, False otherwise
        """
        if train_acc > 0.95 and val_acc < 0.60:
            epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
            warning = f"{epoch_str}Severe overfitting detected (train={train_acc:.2%}, val={val_acc:.2%})"
            self.warnings.append(warning)
            print(f"⚠️ {warning}")
            return True
        return False
    
    def check_divergence(self, val_losses: List[float], patience: int = 10) -> bool:
        """
        Check for validation loss divergence (Requirement 9.2)
        
        Args:
            val_losses: List of validation losses (most recent last)
            patience: Number of consecutive increasing epochs to trigger warning
            
        Returns:
            True if divergence detected, False otherwise
        """
        if len(val_losses) < patience:
            return False
        
        recent_losses = val_losses[-patience:]
        # Check if all recent losses are increasing
        is_diverging = all(recent_losses[i] > recent_losses[i-1] 
                          for i in range(1, len(recent_losses)))
        
        if is_diverging:
            warning = f"Validation loss divergence: increasing for {patience} consecutive epochs"
            self.warnings.append(warning)
            print(f"⚠️ {warning}")
            return True
        return False
    
    def check_nan_loss(self, loss: float, epoch: int = None) -> bool:
        """
        Check for NaN loss - critical error (Requirement 9.3)
        
        Args:
            loss: Loss value to check
            epoch: Current epoch number (optional)
            
        Returns:
            True if NaN detected, False otherwise
        """
        if np.isnan(loss) or np.isinf(loss):
            epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
            warning = f"{epoch_str}NaN/Inf loss detected - STOPPING TRAINING"
            self.warnings.append(warning)
            print(f"❌ {warning}")
            self.should_stop = True
            return True
        return False
    
    def check_exploding_gradients(self, grad_norm: float, threshold: float = 10.0, 
                                  epoch: int = None) -> bool:
        """
        Check for exploding gradients (Requirement 9.4)
        
        Args:
            grad_norm: Gradient norm value
            threshold: Threshold for gradient explosion (default: 10.0)
            epoch: Current epoch number (optional)
            
        Returns:
            True if exploding gradients detected, False otherwise
        """
        if grad_norm > threshold:
            epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
            warning = f"{epoch_str}Exploding gradient detected (norm={grad_norm:.2f} > {threshold})"
            self.warnings.append(warning)
            print(f"⚠️ {warning}")
            return True
        return False
    
    def check_timeout(self, start_time, max_minutes: int = 10) -> bool:
        """
        Check for training timeout (Requirement 9.5)
        
        Args:
            start_time: Training start time (datetime object)
            max_minutes: Maximum allowed training time in minutes
            
        Returns:
            True if timeout exceeded, False otherwise
        """
        from datetime import datetime
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        if elapsed > max_minutes:
            warning = f"Training timeout: exceeded {max_minutes} minutes (elapsed: {elapsed:.1f} min)"
            self.warnings.append(warning)
            print(f"⚠️ {warning}")
            return True
        return False
    
    def get_warnings(self) -> List[str]:
        """
        Get all warnings collected during training
        
        Returns:
            List of warning messages
        """
        return self.warnings
    
    def reset(self):
        """Reset monitor state for new training run"""
        self.warnings = []
        self.should_stop = False
    
    def has_critical_warnings(self) -> bool:
        """
        Check if any critical warnings were raised
        
        Returns:
            True if should_stop flag is set, False otherwise
        """
        return self.should_stop


class BaseSMCModel(ABC):
    """
    Abstract base class for SMC trade prediction models
    
    All models must implement:
    - train(): Train the model
    - predict(): Make predictions
    - predict_proba(): Get probability distributions
    """
    
    def __init__(self, model_name: str, symbol: str, target_col: str = 'TBM_Entry'):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model (e.g., 'RandomForest', 'XGBoost')
            symbol: Trading symbol (e.g., 'EURUSD')
            target_col: Target column name (default: 'TBM_Entry' - predicts BUY/SELL/HOLD directly)
        """
        self.model_name = model_name
        self.symbol = symbol
        self.target_col = target_col
        self.model = None
        self.feature_cols = None
        self.scaler = None
        self.is_trained = False
        self.training_history = {}
        self.feature_importance = None
        self.feature_selector = None
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict:
        """Train the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distributions - must be implemented by subclasses"""
        pass
    
    def load_data(self, train_path: str, val_path: str, test_path: str,
                  exclude_timeout: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/val/test data from CSV files
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            exclude_timeout: Whether to exclude timeout samples (TBM_Label == 0)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n📂 Loading data for {self.symbol}...")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Filter by symbol
        train_df = train_df[train_df['symbol'] == self.symbol].copy()
        val_df = val_df[val_df['symbol'] == self.symbol].copy()
        test_df = test_df[test_df['symbol'] == self.symbol].copy()
        
        # Remove rows with NaN labels (critical fix)
        train_df = train_df[train_df[self.target_col].notna()].copy()
        val_df = val_df[val_df[self.target_col].notna()].copy()
        test_df = test_df[test_df[self.target_col].notna()].copy()
        
        # NEW: Train on ALL data (no filtering)
        # Models learn to predict BUY/SELL/HOLD from full market context
        # exclude_timeout parameter is now ignored - we want models to learn when to HOLD
        if exclude_timeout:
            print(f"  ⚠️ exclude_timeout=True ignored - training on ALL data for better context learning")
        
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False, 
                        apply_feature_selection: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit scaler (True for training data)
            apply_feature_selection: Whether to apply feature selection (default: False for backward compatibility)
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Get feature columns (exclude metadata and target)
        if self.feature_cols is None:
            exclude_patterns = [
                'time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close',
                'TBM_Label', 'TBM_Entry_Price', 'TBM_Stop_Loss', 'TBM_Take_Profit',
                'TBM_Hit_Candle', 'OB_Open', 'OB_Close', 'FVG_Top', 'FVG_Bottom',
                'ChoCH_Level', 'ChoCH_BrokenIndex', 'FVG_MitigatedIndex',
                'OB_High', 'OB_Low', 'EMA_50', 'EMA_200', 'ATR_MA'
            ]
            
            self.feature_cols = []
            for col in df.columns:
                if any(pattern in col for pattern in exclude_patterns):
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.feature_cols.append(col)
            
            print(f"  Selected {len(self.feature_cols)} features")
        
        # Extract features and target
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        
        # CRITICAL FIX: Remove rows with NaN labels FIRST
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"  ⚠️ Removing {n_invalid:,} samples with NaN labels ({n_invalid/len(y)*100:.1f}%)")
            X = X[valid_mask]
            y = y[valid_mask]
        
        # FIX 1: Handle NaN/inf values in features BEFORE feature selection
        # Replace NaN with 0, inf with large values (simple approach)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values to prevent overflow
        X = np.clip(X, -1e10, 1e10)
        
        # FIX 2: Remap labels for binary classification (when timeout excluded)
        # XGBoost expects [0, 1] not [-1, 1]
        unique_labels = np.unique(y)
        if len(unique_labels) == 2 and -1 in unique_labels and 1 in unique_labels:
            # Binary case: Loss=-1, Win=1 → Loss=0, Win=1
            y = np.where(y == -1, 0, 1).astype(int)
            if fit_scaler:
                print(f"  Remapped labels: {np.unique(y)}")
        
        # Apply feature selection FIRST (before scaling/imputation)
        if apply_feature_selection:
            if fit_scaler:
                # Fit feature selector on training data
                self.feature_selector = FeatureSelector()
                X = self.feature_selector.fit_transform(X, y, self.feature_cols)
                # Update feature_cols to selected features only
                self.feature_cols = self.feature_selector.get_selected_features()
            elif self.feature_selector is not None:
                # Transform using fitted selector
                X = self.feature_selector.transform(X)
            else:
                print("  ⚠️ Warning: Feature selection requested but no selector fitted")
        
        # Apply imputation AFTER feature selection (so dimensions match)
        from sklearn.impute import SimpleImputer
        if fit_scaler:
            self.imputer = SimpleImputer(strategy='median')
            X = self.imputer.fit_transform(X)
        elif hasattr(self, 'imputer'):
            X = self.imputer.transform(X)
        
        # Apply scaling AFTER feature selection and imputation (for neural networks)
        if fit_scaler and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        
        # FIX 3: Final validation - check for any remaining NaN/inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("  ⚠️ Warning: NaN/inf detected after processing, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X, y
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_folds: int = 5, stratified: bool = True) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            X: Feature array
            y: Target labels
            n_folds: Number of folds (default: 5)
            stratified: Whether to use stratified splits (default: True)
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"\n🔄 Performing {n_folds}-fold cross-validation...")
        
        # Initialize cross-validator
        if stratified:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create a temporary model instance for this fold
            # Note: Subclasses should implement _create_model() if needed
            temp_model = self._clone_model()
            
            # Train on fold (disable nested cross-validation to avoid recursion)
            temp_model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                           use_cross_validation=False)
            
            # Evaluate on validation fold
            y_pred = temp_model.predict(X_val_fold)
            
            fold_acc = accuracy_score(y_val_fold, y_pred)
            fold_prec = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
            fold_rec = recall_score(y_val_fold, y_pred, average='macro', zero_division=0)
            fold_f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
            
            fold_accuracies.append(fold_acc)
            fold_precisions.append(fold_prec)
            fold_recalls.append(fold_rec)
            fold_f1s.append(fold_f1)
            
            print(f"  Fold {fold_idx}/{n_folds}: Accuracy={fold_acc:.3f}, F1={fold_f1:.3f}")
        
        # Calculate statistics (Requirement 6.1)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        min_accuracy = np.min(fold_accuracies)
        max_accuracy = np.max(fold_accuracies)
        
        # Stability flagging logic (Requirements 6.2, 6.3)
        is_unstable = std_accuracy > 0.10
        is_rejected = std_accuracy > 0.15
        
        # Identify poor-performing folds (Requirement 6.5)
        poor_folds = []
        threshold = mean_accuracy - std_accuracy
        for i, acc in enumerate(fold_accuracies, 1):
            if acc < threshold:
                poor_folds.append({
                    'fold': i,
                    'accuracy': acc,
                    'deviation': acc - mean_accuracy
                })
        
        cv_results = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'mean_precision': np.mean(fold_precisions),
            'std_precision': np.std(fold_precisions),
            'mean_recall': np.mean(fold_recalls),
            'std_recall': np.std(fold_recalls),
            'mean_f1': np.mean(fold_f1s),
            'std_f1': np.std(fold_f1s),
            'fold_accuracies': fold_accuracies,
            'is_stable': not is_unstable,  # Add is_stable flag (inverse of is_unstable)
            'is_unstable': is_unstable,
            'is_rejected': is_rejected,
            'poor_folds': poor_folds,
            'n_folds': n_folds
        }
        
        # Detailed CV reporting (Requirements 6.4, 6.5)
        print(f"\n  Cross-Validation Results:")
        print(f"    Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"    Min Accuracy:  {min_accuracy:.3f}")
        print(f"    Max Accuracy:  {max_accuracy:.3f}")
        print(f"    Range:         {max_accuracy - min_accuracy:.3f}")
        print(f"    Mean F1-Score: {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        
        # Stability assessment with detailed warnings
        if is_rejected:
            print(f"    ❌ MODEL REJECTED: Std dev {std_accuracy:.3f} > 0.15")
            print(f"    Model is highly unstable - DO NOT DEPLOY")
        elif is_unstable:
            print(f"    ⚠️ UNSTABLE: Std dev {std_accuracy:.3f} > 0.10")
            print(f"    Model shows high variance - use with caution")
        else:
            print(f"    ✅ STABLE: Std dev {std_accuracy:.3f} ≤ 0.10")
        
        # Report poor-performing folds
        if poor_folds:
            print(f"\n    Poor-Performing Folds (below mean - std):")
            for fold_info in poor_folds:
                print(f"      Fold {fold_info['fold']}: {fold_info['accuracy']:.3f} "
                      f"(deviation: {fold_info['deviation']:.3f})")
        
        return cv_results
    
    def _clone_model(self):
        """
        Create a clone of this model for cross-validation
        Subclasses should override if special initialization is needed
        
        Returns:
            New instance of the same model class
        """
        # Try to create new instance with same parameters
        # Handle different constructor signatures
        try:
            cloned = self.__class__(self.model_name, self.symbol, self.target_col)
        except TypeError:
            # If that fails, try with just symbol
            try:
                cloned = self.__class__(self.symbol)
                cloned.model_name = self.model_name
                cloned.target_col = self.target_col
            except TypeError:
                # Last resort: try with no arguments
                cloned = self.__class__()
                cloned.model_name = self.model_name
                cloned.symbol = self.symbol
                cloned.target_col = self.target_col
        
        cloned.feature_cols = self.feature_cols
        cloned.scaler = self.scaler
        return cloned
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  rr_ratio: float = 2.0) -> Dict:
        """
        Calculate trading-specific business metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            rr_ratio: Risk-Reward ratio (default: 2.0 for 1:2 R:R)
            
        Returns:
            Dictionary of trading metrics
        """
        # Count wins and losses (exclude timeouts if present)
        wins = np.sum(y_pred == 1)
        losses = np.sum(y_pred == -1)
        timeouts = np.sum(y_pred == 0)
        total_trades = wins + losses
        
        # Win rate (excluding timeouts)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Profit factor (assuming R:R ratio)
        total_profit = wins * rr_ratio  # Each win = rr_ratio * R
        total_loss = losses * 1.0  # Each loss = 1R
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Expected value per trade
        ev_per_trade = (win_rate * rr_ratio) - ((1 - win_rate) * 1.0)
        
        # Accuracy on actual trades (excluding timeouts)
        actual_trades_mask = (y_true != 0) & (y_pred != 0)
        if np.sum(actual_trades_mask) > 0:
            trade_accuracy = accuracy_score(
                y_true[actual_trades_mask], 
                y_pred[actual_trades_mask]
            )
        else:
            trade_accuracy = 0.0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expected_value_per_trade': ev_per_trade,
            'total_wins': int(wins),
            'total_losses': int(losses),
            'total_timeouts': int(timeouts),
            'total_trades': int(total_trades),
            'trade_accuracy': trade_accuracy,
            'risk_reward_ratio': rr_ratio
        }
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                 dataset_name: str = 'Test') -> Dict:
        """
        Evaluate model performance with both classification and trading metrics
        
        Args:
            X: Feature array
            y_true: True labels
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating on {dataset_name} set...")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Determine label mapping (binary vs 3-class)
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels
        
        if is_binary:
            # Binary classification: 0=Loss, 1=Win
            label_map = {0: 'Loss', 1: 'Win'}
            cm_labels = [0, 1]
        else:
            # 3-class: -1=Loss, 0=Timeout, 1=Win
            label_map = {-1: 'Loss', 0: 'Timeout', 1: 'Win'}
            cm_labels = [-1, 0, 1]
        
        # Per-class metrics
        for label, label_name in label_map.items():
            if label in y_true:
                metrics[f'precision_{label_name}'] = precision_score(
                    y_true, y_pred, labels=[label], average='macro', zero_division=0
                )
                metrics[f'recall_{label_name}'] = recall_score(
                    y_true, y_pred, labels=[label], average='macro', zero_division=0
                )
                metrics[f'f1_{label_name}'] = f1_score(
                    y_true, y_pred, labels=[label], average='macro', zero_division=0
                )
        
        # Win rate
        if is_binary:
            # Binary: Win = 1
            win_rate = (y_pred == 1).sum() / len(y_pred)
            actual_win_rate = (y_true == 1).sum() / len(y_true)
        else:
            # 3-class: exclude timeouts
            decisive_mask = y_true != 0
            if decisive_mask.sum() > 0:
                y_true_decisive = y_true[decisive_mask]
                y_pred_decisive = y_pred[decisive_mask]
                win_rate = (y_pred_decisive == 1).sum() / len(y_pred_decisive)
                actual_win_rate = (y_true_decisive == 1).sum() / len(y_true_decisive)
            else:
                win_rate = 0.0
                actual_win_rate = 0.0
        
        metrics['win_rate_predicted'] = win_rate
        metrics['win_rate_actual'] = actual_win_rate
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate trading metrics
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred, rr_ratio=2.0)
        metrics.update(trading_metrics)
        
        # Print classification results
        print(f"\n  Classification Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    Precision (macro): {metrics['precision_macro']:.3f}")
        print(f"    Recall (macro): {metrics['recall_macro']:.3f}")
        print(f"    F1-Score (macro): {metrics['f1_macro']:.3f}")
        
        # Print trading metrics
        print(f"\n  Trading Metrics (1:{trading_metrics['risk_reward_ratio']} R:R):")
        print(f"    Win Rate: {trading_metrics['win_rate']:.1%} ({trading_metrics['total_wins']}/{trading_metrics['total_trades']} trades)")
        print(f"    Profit Factor: {trading_metrics['profit_factor']:.2f}")
        print(f"    Expected Value/Trade: {trading_metrics['expected_value_per_trade']:.2f}R")
        print(f"    Trade Accuracy: {trading_metrics['trade_accuracy']:.1%}")
        if trading_metrics['total_timeouts'] > 0:
            print(f"    Timeouts: {trading_metrics['total_timeouts']} ({trading_metrics['total_timeouts']/(trading_metrics['total_trades']+trading_metrics['total_timeouts'])*100:.1f}%)")
        
        # Profitability assessment
        if trading_metrics['expected_value_per_trade'] > 0:
            print(f"    💰 PROFITABLE STRATEGY (EV > 0)")
        else:
            print(f"    ⚠️ LOSING STRATEGY (EV < 0)")
        
        print(f"\n  Confusion Matrix:")
        if is_binary:
            print(f"              Pred Loss  Pred Win")
            print(f"  True Loss      {cm[0,0]:6d}    {cm[0,1]:6d}")
            print(f"  True Win       {cm[1,0]:6d}    {cm[1,1]:6d}")
        else:
            print(f"              Pred Loss  Pred Timeout  Pred Win")
            print(f"  True Loss      {cm[0,0]:6d}      {cm[0,1]:6d}    {cm[0,2]:6d}")
            print(f"  True Timeout   {cm[1,0]:6d}      {cm[1,1]:6d}    {cm[1,2]:6d}")
            print(f"  True Win       {cm[2,0]:6d}      {cm[2,1]:6d}    {cm[2,2]:6d}")
        
        # Calculate train-val gap if training history is available
        if hasattr(self, 'training_history') and self.training_history:
            # Try to get training accuracy from history
            train_acc = None
            
            # Check different possible keys for training accuracy
            if 'train_accuracy' in self.training_history:
                train_acc = self.training_history['train_accuracy']
                if isinstance(train_acc, list):
                    train_acc = train_acc[-1]  # Get final epoch
            elif 'final_train_accuracy' in self.training_history:
                train_acc = self.training_history['final_train_accuracy']
            
            # If we have training accuracy and this is validation set, calculate gap
            if train_acc is not None and dataset_name in ['Validation', 'Val']:
                train_val_gap = train_acc - metrics['accuracy']
                metrics['train_val_gap'] = train_val_gap
                metrics['overfitting_detected'] = train_val_gap > 0.15
                
                print(f"\n  Overfitting Analysis:")
                print(f"    Training Accuracy: {train_acc:.3f}")
                print(f"    Validation Accuracy: {metrics['accuracy']:.3f}")
                print(f"    Train-Val Gap: {train_val_gap:.3f}")
                
                if metrics['overfitting_detected']:
                    print(f"    ⚠️ OVERFITTING DETECTED (gap > 15%)")
                else:
                    print(f"    ✅ No significant overfitting")
        
        return metrics
    
    def _convert_to_json_serializable(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization
        
        Args:
            obj: Object to convert (can be dict, list, numpy type, etc.)
            
        Returns:
            JSON-serializable version of the object
        """
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
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For unknown types, try to convert to string as fallback
            try:
                return str(obj)
            except:
                return None
    
    def save_model(self, output_dir: str):
        """
        Save trained model and metadata with safe JSON serialization
        
        Args:
            output_dir: Directory to save model files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model pickle (this should always work)
        model_file = output_path / f"{self.symbol}_{self.model_name}.pkl"
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\n💾 Model pickle saved to {model_file}")
        except Exception as e:
            print(f"\n❌ Error saving model pickle: {e}")
            raise
        
        # Prepare metadata
        metadata = {
            'model_name': self.model_name,
            'symbol': self.symbol,
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        
        # Convert metadata to JSON-serializable format
        metadata_safe = self._convert_to_json_serializable(metadata)
        
        # Save metadata with error handling
        metadata_file = output_path / f"{self.symbol}_{self.model_name}_metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata_safe, f, indent=2)
            print(f"💾 Metadata saved to {metadata_file}")
        except (TypeError, ValueError) as e:
            print(f"\n⚠️ Warning: JSON serialization failed: {e}")
            print(f"   Attempting to save partial metadata...")
            
            # Try to save what we can - exclude problematic fields
            safe_metadata = {}
            for key, value in metadata_safe.items():
                try:
                    # Test if this field is serializable
                    json.dumps({key: value})
                    safe_metadata[key] = value
                except (TypeError, ValueError) as field_error:
                    print(f"   Skipping field '{key}': {field_error}")
            
            # Save partial metadata
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(safe_metadata, f, indent=2)
                print(f"💾 Partial metadata saved to {metadata_file}")
                print(f"   Saved fields: {list(safe_metadata.keys())}")
            except Exception as final_error:
                print(f"❌ Error saving even partial metadata: {final_error}")
                # Don't raise - model pickle is more important
        
        # Save scaler if exists
        if self.scaler is not None:
            scaler_file = output_path / f"{self.symbol}_{self.model_name}_scaler.pkl"
            try:
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
                print(f"💾 Scaler saved to {scaler_file}")
            except Exception as e:
                print(f"⚠️ Warning: Error saving scaler: {e}")
        
        # Save feature selector if exists
        if self.feature_selector is not None:
            selector_file = output_path / f"{self.symbol}_{self.model_name}_feature_selector.pkl"
            try:
                with open(selector_file, 'wb') as f:
                    pickle.dump(self.feature_selector, f)
                print(f"💾 Feature selector saved to {selector_file}")
            except Exception as e:
                print(f"⚠️ Warning: Error saving feature selector: {e}")
        
        print(f"\n✅ Model save complete: {output_path}")
    
    def load_model(self, model_dir: str):
        """
        Load trained model and metadata
        
        Args:
            model_dir: Directory containing model files
        """
        model_path = Path(model_dir)
        
        # Load model
        model_file = model_path / f"{self.symbol}_{self.model_name}.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_file = model_path / f"{self.symbol}_{self.model_name}_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.feature_cols = metadata['feature_cols']
        self.is_trained = metadata['is_trained']
        self.training_history = metadata['training_history']
        self.feature_importance = np.array(metadata['feature_importance']) if metadata['feature_importance'] else None
        
        # Load scaler if exists
        scaler_file = model_path / f"{self.symbol}_{self.model_name}_scaler.pkl"
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print(f"\n📂 Model loaded from {model_path}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance is None:
            print("⚠️ Feature importance not available for this model")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
