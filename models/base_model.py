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


class BaseSMCModel(ABC):
    """
    Abstract base class for SMC trade prediction models
    
    All models must implement:
    - train(): Train the model
    - predict(): Make predictions
    - predict_proba(): Get probability distributions
    """
    
    def __init__(self, model_name: str, symbol: str, target_col: str = 'TBM_Label'):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model (e.g., 'RandomForest', 'XGBoost')
            symbol: Trading symbol (e.g., 'EURUSD')
            target_col: Target column name (default: 'TBM_Label')
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
        
        # Exclude timeout samples if requested
        if exclude_timeout:
            train_df = train_df[train_df[self.target_col] != 0].copy()
            val_df = val_df[val_df[self.target_col] != 0].copy()
            test_df = test_df[test_df[self.target_col] != 0].copy()
            print(f"  Excluding timeout samples (TBM_Label == 0)")
        
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
        
        # FIX 1: Handle NaN/inf values BEFORE any processing
        # Replace NaN with column median, inf with large values
        from sklearn.impute import SimpleImputer
        if fit_scaler:
            self.imputer = SimpleImputer(strategy='median')
            X = self.imputer.fit_transform(X)
        elif hasattr(self, 'imputer'):
            X = self.imputer.transform(X)
        else:
            # Fallback if no imputer fitted
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
        
        # Optional scaling (for neural networks)
        if fit_scaler and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Apply feature selection if enabled
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
        
        # Calculate statistics
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        is_stable = std_accuracy < 0.15
        
        cv_results = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_precision': np.mean(fold_precisions),
            'std_precision': np.std(fold_precisions),
            'mean_recall': np.mean(fold_recalls),
            'std_recall': np.std(fold_recalls),
            'mean_f1': np.mean(fold_f1s),
            'std_f1': np.std(fold_f1s),
            'fold_accuracies': fold_accuracies,
            'is_stable': is_stable,
            'n_folds': n_folds
        }
        
        print(f"\n  Cross-Validation Results:")
        print(f"    Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"    Mean F1-Score: {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        
        if not is_stable:
            print(f"    ⚠️ Warning: High variance detected (std={std_accuracy:.3f} > 0.15)")
            print(f"    Model may be unstable - consider more data or simpler architecture")
        else:
            print(f"    ✅ Model is stable across folds")
        
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
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                 dataset_name: str = 'Test') -> Dict:
        """
        Evaluate model performance
        
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
        
        # Print results
        print(f"\n  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.3f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.3f}")
        print(f"  F1-Score (macro): {metrics['f1_macro']:.3f}")
        
        if 'win_rate_predicted' in metrics:
            print(f"\n  Win Rate (excl. timeouts):")
            print(f"    Predicted: {metrics['win_rate_predicted']:.1%}")
            print(f"    Actual:    {metrics['win_rate_actual']:.1%}")
        
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
    
    def save_model(self, output_dir: str):
        """
        Save trained model and metadata
        
        Args:
            output_dir: Directory to save model files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / f"{self.symbol}_{self.model_name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'symbol': self.symbol,
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        
        metadata_file = output_path / f"{self.symbol}_{self.model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scaler if exists
        if self.scaler is not None:
            scaler_file = output_path / f"{self.symbol}_{self.model_name}_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        print(f"\n💾 Model saved to {output_path}")
    
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
