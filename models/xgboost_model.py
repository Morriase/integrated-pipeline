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

from base_model import BaseSMCModel


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
              max_depth: int = 6,
              learning_rate: float = 0.1,
              subsample: float = 0.8,
              colsample_bytree: float = 0.8,
              reg_alpha: float = 0.1,
              reg_lambda: float = 1.0,
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
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            scale_pos_weight: Balancing of positive/negative weights
            early_stopping_rounds: Early stopping patience
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Training history dictionary
        """
        print(f"\n🚀 Training XGBoost for {self.symbol}...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Convert labels to 0, 1, 2 (XGBoost requires non-negative)
        label_map = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = np.array([label_map[y] for y in y_train])
        
        # Calculate class weights if not provided
        if scale_pos_weight is None:
            unique, counts = np.unique(y_train_mapped, return_counts=True)
            scale_pos_weight = counts[0] / counts[2] if len(counts) > 2 else 1.0
        
        # Set device
        tree_method = 'gpu_hist' if use_gpu else 'hist'
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            tree_method=tree_method,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train_mapped)]
        if X_val is not None and y_val is not None:
            y_val_mapped = np.array([label_map[y] for y in y_val])
            eval_set.append((X_val, y_val_mapped))
        
        # Train (simplified - no early stopping to avoid API issues)
        self.model.fit(
            X_train, y_train_mapped,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store reverse label map for predictions
        self.label_map_reverse = {0: -1, 1: 0, 2: 1}
        
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
        
        # Convert back to original labels
        y_pred = np.array([self.label_map_reverse[y] for y in y_pred_mapped])
        
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
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=20,
        use_gpu=False
    )
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')
    
    # Save model
    model.save_model('models/trained')
    
    print("\n✅ XGBoost training complete!")
