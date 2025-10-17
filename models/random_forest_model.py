"""
Random Forest Model for SMC Trade Outcome Prediction

Key Learning Objectives (from WHATS_NEEDED.md):
- Identify most important features
- Learn threshold-based rules
- Capture feature interactions
- Provide interpretable decision paths
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from models.base_model import BaseSMCModel


class RandomForestSMCModel(BaseSMCModel):
    """
    Random Forest classifier for SMC trade prediction
    
    Strengths:
    - Feature importance ranking
    - Handles non-linear relationships
    - Robust to outliers
    - No feature scaling needed
    - Interpretable decision trees
    """
    
    def __init__(self, symbol: str, target_col: str = 'TBM_Label'):
        super().__init__('RandomForest', symbol, target_col)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              n_estimators: int = 200,
              max_depth: Optional[int] = 10,  # REDUCED from 20 to prevent overfitting
              min_samples_split: int = 50,  # INCREASED from 10 for more regularization
              min_samples_leaf: int = 25,  # INCREASED from 5 for more regularization
              max_features: str = 'sqrt',
              class_weight: str = 'balanced',
              use_grid_search: bool = False,
              **kwargs) -> Dict:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            class_weight: Class weighting strategy
            use_grid_search: Whether to use grid search for hyperparameters
            
        Returns:
            Training history dictionary
        """
        print(f"\n🌲 Training Random Forest for {self.symbol}...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        
        if use_grid_search and X_val is not None:
            print("\n  Running grid search for hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='f1_macro',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"\n  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.3f}")
            
        else:
            # Train with specified parameters
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            self.model.fit(X_train, y_train)
        
        # Extract feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        self.training_history = {
            'train_accuracy': train_score,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
        }
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history['val_accuracy'] = val_score
            print(f"\n  Train accuracy: {train_score:.3f}")
            print(f"  Val accuracy:   {val_score:.3f}")
        else:
            print(f"\n  Train accuracy: {train_score:.3f}")
        
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
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distributions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)


# Example usage
if __name__ == "__main__":
    """
    Example: Train Random Forest for EURUSD
    """
    
    # Initialize model
    model = RandomForestSMCModel(symbol='EURUSD')
    
    # Load data
    train_df, val_df, test_df = model.load_data(
        train_path='Data/processed_smc_data_train.csv',
        val_path='Data/processed_smc_data_val.csv',
        test_path='Data/processed_smc_data_test.csv',
        exclude_timeout=False  # Include all classes
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
        max_depth=20,
        use_grid_search=False
    )
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')
    
    # Save model
    model.save_model('models/trained')
    
    print("\n✅ Random Forest training complete!")
