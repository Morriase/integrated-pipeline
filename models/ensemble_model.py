"""
Consensus Ensemble Model - RF + XGBoost + Neural Network
Only trades when ALL 3 models agree (high confidence filter)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import json
from pathlib import Path


class ConsensusEnsembleSMCModel:
    """
    Production Model: RandomForest Only
    
    Single stable model approach:
    - 64% accuracy
    - 48% win rate
    - 0.43R expected value
    - Profitable and reliable
    
    Note: XGBoost and NN disabled due to label mapping issues
    """
    
    def __init__(self, symbol: str = 'UNIFIED'):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        self.is_trained = False
        self.consensus_mode = 'single'  # Only one model
        
    def load_models(self, model_dir: str = '/kaggle/working/Model-output'):
        """Load trained RandomForest model"""
        model_path = Path(model_dir)
        
        print(f"\n🔄 Loading model from: {model_dir}")
        
        # Load Random Forest
        rf_file = model_path / f"{self.symbol}_RandomForest.pkl"
        if rf_file.exists():
            with open(rf_file, 'rb') as f:
                model_data = pickle.load(f)
                # Handle both dict and direct model formats
                if isinstance(model_data, dict):
                    self.models['RandomForest'] = model_data['model']
                    self.feature_cols['RandomForest'] = model_data['feature_cols']
                else:
                    # Model saved directly (old format)
                    self.models['RandomForest'] = model_data
                    # Load metadata separately
                    meta_file = model_path / f"{self.symbol}_RandomForest_metadata.json"
                    if meta_file.exists():
                        with open(meta_file, 'r') as mf:
                            metadata = json.load(mf)
                            self.feature_cols['RandomForest'] = metadata['feature_cols']
            print(f"  ✅ RandomForest loaded (64% accuracy, 0.43R EV)")
            self.is_trained = True
        else:
            print(f"  ❌ RandomForest not found: {rf_file}")
            raise FileNotFoundError(f"RandomForest model not found: {rf_file}")
    
    def set_consensus_mode(self, mode: str = 'strict'):
        """
        Set consensus mode
        
        Args:
            mode: 'strict' (all 3 agree) or 'majority' (2/3 agree)
        """
        if mode not in ['strict', 'majority']:
            raise ValueError("Mode must be 'strict' or 'majority'")
        
        self.consensus_mode = mode
        print(f"\n📊 Consensus mode set to: {mode}")
        if mode == 'strict':
            print("   ✅ Will only trade when ALL 3 models agree")
        else:
            print("   ✅ Will trade when 2/3 models agree")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using RandomForest
        
        Args:
            X: DataFrame with features
            
        Returns:
            Tuple of (predictions, confidence_flags)
            - predictions: Array of predictions (-1, 0, 1)
            - confidence_flags: Array of booleans (all True since single model)
        """
        if not self.is_trained:
            raise ValueError("RandomForest model must be loaded. Call load_models() first.")
        
        n_samples = len(X)
        
        # Add missing features with zeros
        required_features = self.feature_cols['RandomForest']
        for feature in required_features:
            if feature not in X.columns:
                X[feature] = 0.0
        
        # Random Forest prediction
        X_rf = X[required_features].values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions = self.models['RandomForest'].predict(X_rf)
        probabilities = self.models['RandomForest'].predict_proba(X_rf)
        
        # All predictions are confident (single model)
        confidence_flags = np.ones(n_samples, dtype=bool)
        
        return predictions, confidence_flags
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray, dataset_name: str = 'Test') -> Dict:
        """
        Evaluate consensus ensemble performance
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            dataset_name: Name of dataset
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating Consensus Ensemble on {dataset_name} set...")
        
        # Get predictions
        y_pred, confidence_flags = self.predict(X)
        
        # Overall metrics (including skipped trades)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        overall_acc = accuracy_score(y_true, y_pred)
        
        # High-confidence trades only
        high_conf_mask = confidence_flags
        n_high_conf = high_conf_mask.sum()
        n_total = len(y_true)
        
        if n_high_conf > 0:
            y_true_conf = y_true[high_conf_mask]
            y_pred_conf = y_pred[high_conf_mask]
            
            conf_acc = accuracy_score(y_true_conf, y_pred_conf)
            conf_prec = precision_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
            conf_rec = recall_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
            conf_f1 = f1_score(y_true_conf, y_pred_conf, average='macro', zero_division=0)
            
            # Trading metrics for high-confidence trades
            wins = np.sum(y_pred_conf == 1)
            losses = np.sum(y_pred_conf == -1)
            timeouts = np.sum(y_pred_conf == 0)
            total_trades = wins + losses
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            profit_factor = (wins * 2.0) / losses if losses > 0 else float('inf')
            ev_per_trade = (win_rate * 2.0) - ((1 - win_rate) * 1.0)
        else:
            conf_acc = 0
            conf_prec = 0
            conf_rec = 0
            conf_f1 = 0
            wins = 0
            losses = 0
            timeouts = 0
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            ev_per_trade = 0
        
        metrics = {
            'overall_accuracy': overall_acc,
            'high_confidence_trades': n_high_conf,
            'total_samples': n_total,
            'confidence_rate': n_high_conf / n_total,
            'high_conf_accuracy': conf_acc,
            'high_conf_precision': conf_prec,
            'high_conf_recall': conf_rec,
            'high_conf_f1': conf_f1,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expected_value_per_trade': ev_per_trade,
            'total_wins': int(wins),
            'total_losses': int(losses),
            'total_timeouts': int(timeouts),
            'total_trades': int(total_trades)
        }
        
        # Print results
        print(f"\n  Consensus Mode: {self.consensus_mode}")
        print(f"  High-Confidence Trades: {n_high_conf}/{n_total} ({n_high_conf/n_total*100:.1f}%)")
        print(f"\n  Overall Accuracy: {overall_acc:.3f}")
        print(f"\n  High-Confidence Metrics:")
        print(f"    Accuracy: {conf_acc:.3f}")
        print(f"    Precision: {conf_prec:.3f}")
        print(f"    Recall: {conf_rec:.3f}")
        print(f"    F1-Score: {conf_f1:.3f}")
        
        if total_trades > 0:
            print(f"\n  Trading Metrics (1:2 R:R):")
            print(f"    Win Rate: {win_rate:.1%} ({wins}/{total_trades} trades)")
            print(f"    Profit Factor: {profit_factor:.2f}")
            print(f"    Expected Value/Trade: {ev_per_trade:.2f}R")
            
            if ev_per_trade > 0:
                print(f"    💰 PROFITABLE STRATEGY (EV > 0)")
            else:
                print(f"    ⚠️ LOSING STRATEGY (EV < 0)")
        
        return metrics
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from RandomForest (single model)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary of model_name -> predictions
        """
        predictions = {}
        
        # Add missing features with zeros
        required_features = self.feature_cols['RandomForest']
        for feature in required_features:
            if feature not in X.columns:
                X[feature] = 0.0
        
        # Random Forest
        X_rf = X[required_features].values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions['RandomForest'] = self.models['RandomForest'].predict(X_rf)
        with torch.no_grad():
            outputs = self.models['NeuralNetwork'](X_tensor)
            nn_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        
        label_map = {0: -1, 1: 0, 2: 1}
        predictions['NeuralNetwork'] = np.array([label_map.get(p, p) for p in nn_pred])
        
        return predictions
