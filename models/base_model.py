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
import warnings
warnings.filterwarnings('ignore')


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
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit scaler (True for training data)
            
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
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Optional scaling (for neural networks)
        if fit_scaler and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X, y
    
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
        
        # Per-class metrics
        for label in [-1, 0, 1]:
            label_name = {-1: 'Loss', 0: 'Timeout', 1: 'Win'}[label]
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
        
        # Win rate (excluding timeouts)
        decisive_mask = y_true != 0
        if decisive_mask.sum() > 0:
            y_true_decisive = y_true[decisive_mask]
            y_pred_decisive = y_pred[decisive_mask]
            
            win_rate = (y_pred_decisive == 1).sum() / len(y_pred_decisive)
            actual_win_rate = (y_true_decisive == 1).sum() / len(y_true_decisive)
            
            metrics['win_rate_predicted'] = win_rate
            metrics['win_rate_actual'] = actual_win_rate
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
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
        print(f"              Pred Loss  Pred Timeout  Pred Win")
        print(f"  True Loss      {cm[0,0]:6d}      {cm[0,1]:6d}    {cm[0,2]:6d}")
        print(f"  True Timeout   {cm[1,0]:6d}      {cm[1,1]:6d}    {cm[1,2]:6d}")
        print(f"  True Win       {cm[2,0]:6d}      {cm[2,1]:6d}    {cm[2,2]:6d}")
        
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
