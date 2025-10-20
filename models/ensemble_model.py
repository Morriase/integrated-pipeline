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
    Consensus Ensemble: RandomForest + XGBoost + NeuralNetwork
    
    Strategy: ONLY TRADE WHEN ALL 3 MODELS AGREE
    - High confidence filter
    - Reduces trade count
    - Increases win rate
    - More reliable predictions
    
    Models:
    - RandomForest:  Stable, feature importance
    - XGBoost:       Highest accuracy (72.2%)
    - NeuralNetwork: Best EV (0.72R), highest win rate (57.2%)
    """
    
    def __init__(self, symbol: str = 'UNIFIED'):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        self.is_trained = False
        self.consensus_mode = 'strict'  # 'strict' = all agree, 'majority' = 2/3 agree
        
    def load_models(self, model_dir: str = '/kaggle/working/Model-output'):
        """Load trained RF, XGBoost, and NN models"""
        model_path = Path(model_dir)
        
        print(f"\n🔄 Loading models from: {model_dir}")
        
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
            print(f"  ✅ RandomForest loaded")
        else:
            print(f"  ❌ RandomForest not found: {rf_file}")
        
        # Load XGBoost
        xgb_file = model_path / f"{self.symbol}_XGBoost.pkl"
        if xgb_file.exists():
            with open(xgb_file, 'rb') as f:
                model_data = pickle.load(f)
                # Handle both dict and direct model formats
                if isinstance(model_data, dict):
                    self.models['XGBoost'] = model_data['model']
                    self.feature_cols['XGBoost'] = model_data['feature_cols']
                else:
                    # Model saved directly (old format)
                    self.models['XGBoost'] = model_data
                    # Load metadata separately
                    meta_file = model_path / f"{self.symbol}_XGBoost_metadata.json"
                    if meta_file.exists():
                        with open(meta_file, 'r') as mf:
                            metadata = json.load(mf)
                            self.feature_cols['XGBoost'] = metadata['feature_cols']
            print(f"  ✅ XGBoost loaded")
        else:
            print(f"  ❌ XGBoost not found: {xgb_file}")
        
        # Load Neural Network
        nn_file = model_path / f"{self.symbol}_NeuralNetwork.pkl"
        if nn_file.exists():
            with open(nn_file, 'rb') as f:
                model_data = pickle.load(f)
                # Handle both dict and direct model formats
                if isinstance(model_data, dict):
                    self.models['NeuralNetwork'] = model_data['model']
                    self.feature_cols['NeuralNetwork'] = model_data['feature_cols']
                    if 'scaler' in model_data:
                        self.scalers['NeuralNetwork'] = model_data['scaler']
                else:
                    # Model saved directly (old format)
                    self.models['NeuralNetwork'] = model_data
                    # Load metadata and scaler separately
                    meta_file = model_path / f"{self.symbol}_NeuralNetwork_metadata.json"
                    if meta_file.exists():
                        with open(meta_file, 'r') as mf:
                            metadata = json.load(mf)
                            self.feature_cols['NeuralNetwork'] = metadata['feature_cols']
                    
                    scaler_file = model_path / f"{self.symbol}_NeuralNetwork_scaler.pkl"
                    if scaler_file.exists():
                        with open(scaler_file, 'rb') as sf:
                            self.scalers['NeuralNetwork'] = pickle.load(sf)
            print(f"  ✅ NeuralNetwork loaded")
        else:
            print(f"  ❌ NeuralNetwork not found: {nn_file}")
        
        self.is_trained = len(self.models) == 3
        
        if self.is_trained:
            print(f"\n✅ All 3 models loaded successfully for {self.symbol}")
            print(f"   Consensus mode: {self.consensus_mode}")
        else:
            print(f"\n⚠️ Only {len(self.models)}/3 models loaded")
            print(f"   Required: RandomForest, XGBoost, NeuralNetwork")
    
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
        Make consensus predictions - only trade when models agree
        
        Args:
            X: DataFrame with features
            
        Returns:
            Tuple of (predictions, confidence_flags)
            - predictions: Array of predictions (-1, 0, 1)
            - confidence_flags: Array of booleans (True = high confidence, False = skip trade)
        """
        if not self.is_trained:
            raise ValueError("All 3 models must be loaded. Call load_models() first.")
        
        n_samples = len(X)
        predictions = {}
        probabilities = {}
        
        # Random Forest prediction
        X_rf = X[self.feature_cols['RandomForest']].values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions['RandomForest'] = self.models['RandomForest'].predict(X_rf)
        probabilities['RandomForest'] = self.models['RandomForest'].predict_proba(X_rf)
        
        # XGBoost prediction
        X_xgb = X[self.feature_cols['XGBoost']].values
        X_xgb = np.nan_to_num(X_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions['XGBoost'] = self.models['XGBoost'].predict(X_xgb)
        probabilities['XGBoost'] = self.models['XGBoost'].predict_proba(X_xgb)
        
        # Neural Network prediction
        X_nn = X[self.feature_cols['NeuralNetwork']].values
        X_nn = np.nan_to_num(X_nn, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Scale for NN
        if 'NeuralNetwork' in self.scalers:
            X_nn = self.scalers['NeuralNetwork'].transform(X_nn)
        
        # Convert to torch and predict
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X_nn).to(device)
        
        self.models['NeuralNetwork'].eval()
        with torch.no_grad():
            outputs = self.models['NeuralNetwork'](X_tensor)
            probabilities['NeuralNetwork'] = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions['NeuralNetwork'] = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Convert XGBoost predictions back to original labels if needed
        # XGBoost uses 0,1,2 but we need -1,0,1
        label_map = {0: -1, 1: 0, 2: 1}
        predictions['XGBoost'] = np.array([label_map.get(p, p) for p in predictions['XGBoost']])
        predictions['NeuralNetwork'] = np.array([label_map.get(p, p) for p in predictions['NeuralNetwork']])
        
        # Consensus voting
        consensus_pred = np.zeros(n_samples, dtype=int)
        confidence_flags = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            rf_pred = predictions['RandomForest'][i]
            xgb_pred = predictions['XGBoost'][i]
            nn_pred = predictions['NeuralNetwork'][i]
            
            if self.consensus_mode == 'strict':
                # All 3 must agree
                if rf_pred == xgb_pred == nn_pred:
                    consensus_pred[i] = rf_pred
                    confidence_flags[i] = True
                else:
                    consensus_pred[i] = 0  # No trade (timeout)
                    confidence_flags[i] = False
            
            elif self.consensus_mode == 'majority':
                # 2 out of 3 agree
                votes = [rf_pred, xgb_pred, nn_pred]
                unique, counts = np.unique(votes, return_counts=True)
                
                if counts.max() >= 2:
                    consensus_pred[i] = unique[counts.argmax()]
                    confidence_flags[i] = True
                else:
                    consensus_pred[i] = 0  # No trade
                    confidence_flags[i] = False
        
        return consensus_pred, confidence_flags
    
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
        Get predictions from each individual model (for debugging)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary of model_name -> predictions
        """
        predictions = {}
        
        # Random Forest
        X_rf = X[self.feature_cols['RandomForest']].values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions['RandomForest'] = self.models['RandomForest'].predict(X_rf)
        
        # XGBoost
        X_xgb = X[self.feature_cols['XGBoost']].values
        X_xgb = np.nan_to_num(X_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        xgb_pred = self.models['XGBoost'].predict(X_xgb)
        label_map = {0: -1, 1: 0, 2: 1}
        predictions['XGBoost'] = np.array([label_map.get(p, p) for p in xgb_pred])
        
        # Neural Network
        X_nn = X[self.feature_cols['NeuralNetwork']].values
        X_nn = np.nan_to_num(X_nn, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if 'NeuralNetwork' in self.scalers:
            X_nn = self.scalers['NeuralNetwork'].transform(X_nn)
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X_nn).to(device)
        
        self.models['NeuralNetwork'].eval()
        with torch.no_grad():
            outputs = self.models['NeuralNetwork'](X_tensor)
            nn_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        
        label_map = {0: -1, 1: 0, 2: 1}
        predictions['NeuralNetwork'] = np.array([label_map.get(p, p) for p in nn_pred])
        
        return predictions
