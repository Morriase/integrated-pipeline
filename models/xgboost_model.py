"""
XGBoost Model for SMC Trade Outcome Prediction

Key Learning Objectives (from WHATS_NEEDED.md):
- Learn gradient-based patterns
- Handle feature importance ranking
- Capture residual patterns missed by other models
- Optimize for classification metrics
"""

import numpy as np
from typing import Dict, Optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

from models.base_model import BaseSMCModel


class XGBoostSMCModel(BaseSMCModel):
    """
    XGBoost classifier for SMC trade prediction
    
    Strengths:
    - Gradient boosting for high accuracy
    - Built-in regularization
    - Handles imbalanced classes
    - Feature importance
    - Fast training with GPU support
    """
    
    def __init__(self, symbol: str, target_col: str = 'TBM_Label'):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        super().__init__('XGBoost', symbol, target_col)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              n_estimators: int = 200,
              max_depth: int = 3,  # AGGRESSIVE: Reduced from 4
              learning_rate: float = 0.01,  # AGGRESSIVE: Reduced from 0.1 (10x slower)
              subsample: float = 0.6,  # AGGRESSIVE: Reduced from 0.7
              colsample_bytree: float = 0.6,  # AGGRESSIVE: Reduced from 0.7
              min_child_weight: int = 10,  # AGGRESSIVE: Increased from 5
              reg_alpha: float = 0.5,  # AGGRESSIVE: Increased from 0.2
              reg_lambda: float = 3.0,  # AGGRESSIVE: Increased from 2.0
              max_delta_step: int = 1,  # AGGRESSIVE: Limit weight updates
              scale_pos_weight: Optional[float] = None,
              early_stopping_rounds: int = 20,
              use_gpu: bool = False,
              **kwargs) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (default: 3 for aggressive regularization)
            learning_rate: Learning rate (eta) (default: 0.01 for slower learning)
            subsample: Subsample ratio of training instances (default: 0.6)
            colsample_bytree: Subsample ratio of columns (default: 0.6)
            min_child_weight: Minimum sum of instance weight in a child (default: 10)
            reg_alpha: L1 regularization (default: 0.5)
            reg_lambda: L2 regularization (default: 3.0)
            max_delta_step: Maximum delta step for weight updates (default: 1)
            scale_pos_weight: Balancing of positive/negative weights
            early_stopping_rounds: Early stopping patience
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Training history dictionary
        """
        print(f"\n🚀 Training XGBoost for {self.symbol}...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Check if labels are already remapped (binary: 0,1) or original (3-class: -1,0,1)
        unique_labels = np.unique(y_train)
        is_binary = len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels
        
        if is_binary:
            # Binary classification: labels already [0, 1]
            y_train_mapped = y_train.astype(int)
            num_classes = 2
            objective = 'binary:logistic'
            eval_metric = 'logloss'
            
            # Calculate class weights
            if scale_pos_weight is None:
                unique, counts = np.unique(y_train_mapped, return_counts=True)
                scale_pos_weight = counts[0] / counts[1] if len(counts) == 2 else 1.0
        else:
            # 3-class: Convert labels to 0, 1, 2
            label_map = {-1: 0, 0: 1, 1: 2}
            y_train_mapped = np.array([label_map[y] for y in y_train])
            num_classes = 3
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            
            # Calculate class weights
            if scale_pos_weight is None:
                unique, counts = np.unique(y_train_mapped, return_counts=True)
                scale_pos_weight = counts[0] / counts[2] if len(counts) > 2 else 1.0
        
        # Set device
        tree_method = 'gpu_hist' if use_gpu else 'hist'
        
        # Initialize model with AGGRESSIVE regularization
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=max_delta_step,  # AGGRESSIVE: Limit weight updates
            scale_pos_weight=scale_pos_weight,
            tree_method=tree_method,
            objective=objective,
            num_class=num_classes if objective == 'multi:softmax' else None,
            eval_metric=eval_metric,
            random_state=42,
            n_jobs=-1
        )
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train_mapped)]
        if X_val is not None and y_val is not None:
            if is_binary:
                y_val_mapped = y_val.astype(int)
            else:
                y_val_mapped = np.array([label_map[y] for y in y_val])
            eval_set.append((X_val, y_val_mapped))
        
        # Store mapping info for predictions
        self.is_binary = is_binary
        if not is_binary:
            self.label_map = label_map
            self.reverse_map = {v: k for k, v in label_map.items()}
        
        # Train (simplified - no early stopping to avoid API issues)
        self.model.fit(
            X_train, y_train_mapped,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store reverse label map for predictions (only for 3-class)
        if not is_binary:
            self.label_map_reverse = {0: -1, 1: 0, 2: 1}
        else:
            self.label_map_reverse = None  # Binary: no remapping needed
        
        # Extract feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train_mapped).mean()
        
        self.training_history = {
            'train_accuracy': train_accuracy,
            'n_estimators': self.model.n_estimators,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else n_estimators,
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val_mapped).mean()
            self.training_history['val_accuracy'] = val_accuracy
            print(f"\n  Train accuracy: {train_accuracy:.3f}")
            print(f"  Val accuracy:   {val_accuracy:.3f}")
            print(f"  Best iteration: {self.training_history['best_iteration']}")
        else:
            print(f"\n  Train accuracy: {train_accuracy:.3f}")
        
        self.is_trained = True
        
        # Print top features
        print(f"\n  Top 10 Most Important Features:")
        top_features = self.get_feature_importance(top_n=10)
        for idx, row in top_features.iterrows():
            print(f"    {row['feature']:<40s}: {row['importance']:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Predict with mapped labels
        y_pred_mapped = self.model.predict(X)
        
        # Convert back to original labels (only for 3-class)
        if self.label_map_reverse is not None:
            y_pred = np.array([self.label_map_reverse[y] for y in y_pred_mapped])
        else:
            # Binary: already in correct format [0, 1]
            y_pred = y_pred_mapped.astype(int)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distributions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)


# Example usage
if __name__ == "__main__":
    """
    Example: Train XGBoost for EURUSD
    """
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available. Install with: pip install xgboost")
        exit(1)
    
    # Initialize model
    model = XGBoostSMCModel(symbol='EURUSD')
    
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
    
    # Train model (using enhanced regularization defaults)
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        n_estimators=200,
        # Enhanced regularization parameters (defaults):
        # max_depth=4, min_child_weight=5, subsample=0.7
        # reg_alpha=0.2, reg_lambda=2.0
        early_stopping_rounds=20,
        use_gpu=False
    )
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')
    
    # Save model
    model.save_model('models/trained')
    
    print("\n✅ XGBoost training complete!")
