"""
Ensemble Model - Combines Random Forest + Neural Network
Uses weighted voting based on validation performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
import json
from pathlib import Path


class EnsembleSMCModel:
    """
    Ensemble combining Random Forest and Neural Network
    
    Strategy:
    - Random Forest: 69.5% avg accuracy (primary model)
    - Neural Network: 61.2% avg accuracy (secondary model)
    - Weighted voting based on validation performance
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def load_models(self, model_dir: str = '/kaggle/working'):
        """Load trained RF and NN models"""
        model_path = Path(model_dir)
        
        # Load Random Forest
        rf_file = model_path / f"{self.symbol}_RandomForest.pkl"
        if rf_file.exists():
            with open(rf_file, 'rb') as f:
                self.models['RandomForest'] = pickle.load(f)
            
            # Load metadata
            rf_meta = model_path / f"{self.symbol}_RandomForest_metadata.json"
            with open(rf_meta, 'r') as f:
                rf_metadata = json.load(f)
                self.rf_feature_cols = rf_metadata['feature_cols']
        
        # Load Neural Network
        nn_file = model_path / f"{self.symbol}_NeuralNetwork.pkl"
        if nn_file.exists():
            with open(nn_file, 'rb') as f:
                self.models['NeuralNetwork'] = pickle.load(f)
            
            # Load metadata
            nn_meta = model_path / f"{self.symbol}_NeuralNetwork_metadata.json"
            with open(nn_meta, 'r') as f:
                nn_metadata = json.load(f)
                self.nn_feature_cols = nn_metadata['feature_cols']
            
            # Load scaler
            nn_scaler = model_path / f"{self.symbol}_NeuralNetwork_scaler.pkl"
            if nn_scaler.exists():
                with open(nn_scaler, 'rb') as f:
                    self.nn_scaler = pickle.load(f)
        
        self.is_trained = len(self.models) > 0
        print(f"✅ Loaded {len(self.models)} models for {self.symbol}")
        
    def set_weights(self, val_accuracies: Dict[str, float]):
        """
        Set ensemble weights based on validation performance
        
        Args:
            val_accuracies: Dict of model_name -> validation accuracy
        """
        # Softmax weighting (exponential of accuracies)
        total = sum(np.exp(acc * 10) for acc in val_accuracies.values())
        self.weights = {
            name: np.exp(acc * 10) / total 
            for name, acc in val_accuracies.items()
        }
        
        print(f"\n📊 Ensemble weights for {self.symbol}:")
        for name, weight in self.weights.items():
            print(f"  {name:15s}: {weight:.1%}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions using weighted voting
        
        Args:
            X: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        predictions = {}
        probabilities = {}
        
        # Random Forest prediction
        if 'RandomForest' in self.models:
            X_rf = X[self.rf_feature_cols].values
            # Handle NaN
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_rf = imputer.fit_transform(X_rf)
            
            predictions['RandomForest'] = self.models['RandomForest'].predict(X_rf)
            probabilities['RandomForest'] = self.models['RandomForest'].predict_proba(X_rf)
        
        # Neural Network prediction
        if 'NeuralNetwork' in self.models:
            X_nn = X[self.nn_feature_cols].values
            # Handle NaN
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_nn = imputer.fit_transform(X_nn)
            X_nn = np.clip(X_nn, -1e10, 1e10)
            
            # Scale
            if hasattr(self, 'nn_scaler'):
                X_nn = self.nn_scaler.transform(X_nn)
            
            # Convert to torch and predict
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(X_nn).to(device)
            
            self.models['NeuralNetwork'].eval()
            with torch.no_grad():
                outputs = self.models['NeuralNetwork'](X_tensor)
                probabilities['NeuralNetwork'] = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions['NeuralNetwork'] = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Weighted probability voting
        if len(probabilities) > 1:
            # Average probabilities weighted by model performance
            weighted_probs = np.zeros_like(list(probabilities.values())[0])
            
            for name, probs in probabilities.items():
                weight = self.weights.get(name, 1.0 / len(probabilities))
                weighted_probs += weight * probs
            
            # Final prediction from weighted probabilities
            ensemble_pred = np.argmax(weighted_probs, axis=1)
        else:
            # Only one model available
            ensemble_pred = list(predictions.values())[0]
        
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        y_pred = self.predict(X)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y, y_pred, average='macro', zero_division=0)
        }
        
        # Win rate
        if len(np.unique(y)) == 2:  # Binary
            win_rate = (y_pred == 1).sum() / len(y_pred)
            actual_win_rate = (y == 1).sum() / len(y)
        else:
            win_rate = (y_pred == 1).sum() / len(y_pred)
            actual_win_rate = (y == 1).sum() / len(y)
        
        metrics['win_rate_predicted'] = win_rate
        metrics['win_rate_actual'] = actual_win_rate
        
        return metrics


def create_ensemble_for_all_symbols(model_dir='/kaggle/working', 
                                     results_file='/kaggle/working/all_models_results.json'):
    """
    Create ensemble models for all symbols and evaluate
    """
    import json
    
    # Load training results to get validation accuracies
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load test data
    test_df = pd.read_csv('/kaggle/working/processed_smc_data_test.csv')
    test_df = test_df[test_df['TBM_Label'] != 0]  # Exclude timeout
    
    ensemble_results = {}
    
    for symbol in results.keys():
        print(f"\n{'='*80}")
        print(f"Creating Ensemble for {symbol}")
        print(f"{'='*80}")
        
        # Create ensemble
        ensemble = EnsembleSMCModel(symbol)
        ensemble.load_models(model_dir)
        
        # Set weights based on validation performance
        val_accuracies = {}
        if 'RandomForest' in results[symbol]:
            val_acc = results[symbol]['RandomForest'].get('val_metrics', {}).get('accuracy', 0.5)
            val_accuracies['RandomForest'] = val_acc
        
        if 'NeuralNetwork' in results[symbol]:
            val_acc = results[symbol]['NeuralNetwork'].get('val_metrics', {}).get('accuracy', 0.5)
            val_accuracies['NeuralNetwork'] = val_acc
        
        ensemble.set_weights(val_accuracies)
        
        # Evaluate on test set
        symbol_test = test_df[test_df['symbol'] == symbol].copy()
        
        if len(symbol_test) > 0:
            y_test = symbol_test['TBM_Label'].values
            # Remap labels if needed
            if -1 in y_test:
                y_test = np.where(y_test == -1, 0, y_test)
            
            metrics = ensemble.evaluate(symbol_test, y_test)
            
            print(f"\n📊 Ensemble Test Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.1%}")
            print(f"  Precision: {metrics['precision']:.1%}")
            print(f"  Recall:    {metrics['recall']:.1%}")
            print(f"  F1-Score:  {metrics['f1']:.1%}")
            print(f"  Win Rate:  {metrics['win_rate_predicted']:.1%} (actual: {metrics['win_rate_actual']:.1%})")
            
            # Compare to individual models
            rf_acc = results[symbol].get('RandomForest', {}).get('test_metrics', {}).get('accuracy', 0)
            nn_acc = results[symbol].get('NeuralNetwork', {}).get('test_metrics', {}).get('accuracy', 0)
            
            print(f"\n  vs Random Forest:   {rf_acc:.1%}")
            print(f"  vs Neural Network:  {nn_acc:.1%}")
            
            improvement = metrics['accuracy'] - max(rf_acc, nn_acc)
            print(f"  Improvement: {improvement:+.1%}")
            
            ensemble_results[symbol] = metrics
    
    # Overall summary
    print(f"\n{'='*80}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*80}")
    
    avg_accuracy = np.mean([m['accuracy'] for m in ensemble_results.values()])
    print(f"\nAverage Ensemble Accuracy: {avg_accuracy:.1%}")
    
    # Save results
    with open(f'{model_dir}/ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f, indent=2, default=str)
    
    print(f"\n✅ Ensemble results saved to {model_dir}/ensemble_results.json")
    
    return ensemble_results


if __name__ == "__main__":
    # Create ensembles for all symbols
    results = create_ensemble_for_all_symbols()
