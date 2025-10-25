"""
Ensemble Predictor - Combines RandomForest, XGBoost, and Neural Network
Uses weighted voting based on model performance
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EnsemblePredictor:
    """
    Ensemble predictor combining RF, XGBoost, and NN
    
    Voting strategies:
    - 'weighted': Weight by test accuracy
    - 'soft': Average predicted probabilities
    - 'hard': Majority vote
    """
    
    def __init__(self, models_dir: str = 'models/trained'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.weights = {}
        self.metadata = {}
        
    def load_models(self):
        """Load all trained models and their metadata"""
        print("🔄 Loading ensemble models...")
        
        # Load training results for weights
        results_path = self.models_dir / 'training_results.json'
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        model_results = results['training_results']['UNIFIED']
        
        # Load each model
        for model_name in ['RandomForest', 'XGBoost', 'NeuralNetwork']:
            model_path = self.models_dir / f'UNIFIED_{model_name}.pkl'
            metadata_path = self.models_dir / f'UNIFIED_{model_name}_metadata.json'
            
            # Load model
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata[model_name] = json.load(f)
            
            # Set weight based on test accuracy
            test_acc = model_results[model_name]['test_metrics']['accuracy']
            self.weights[model_name] = test_acc
            
            print(f"  ✅ {model_name}: Test Acc = {test_acc:.4f}")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"\n📊 Normalized weights:")
        for model, weight in self.weights.items():
            print(f"  {model}: {weight:.4f}")
    
    def predict(self, X: np.ndarray, strategy: str = 'weighted') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Feature matrix
            strategy: 'weighted', 'soft', or 'hard'
            
        Returns:
            Predicted labels
        """
        if not self.models:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            pred = model.predict(X)
            predictions[model_name] = pred
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities[model_name] = model.predict_proba(X)
        
        if strategy == 'weighted':
            return self._weighted_vote(predictions)
        elif strategy == 'soft':
            return self._soft_vote(probabilities)
        elif strategy == 'hard':
            return self._hard_vote(predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted probability predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.models:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        probabilities = {}
        
        # Get probabilities from each model
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probabilities[model_name] = model.predict_proba(X)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(probabilities['RandomForest'])
        for model_name, proba in probabilities.items():
            weighted_proba += proba * self.weights[model_name]
        
        return weighted_proba
    
    def _weighted_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted voting based on model accuracy"""
        n_samples = len(predictions['RandomForest'])
        weighted_votes = np.zeros(n_samples)
        
        for model_name, pred in predictions.items():
            weighted_votes += pred * self.weights[model_name]
        
        # Round to nearest class
        return np.round(weighted_votes).astype(int)
    
    def _soft_vote(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Soft voting using weighted probability averaging"""
        weighted_proba = np.zeros_like(probabilities['RandomForest'])
        
        for model_name, proba in probabilities.items():
            weighted_proba += proba * self.weights[model_name]
        
        return np.argmax(weighted_proba, axis=1)
    
    def _hard_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard majority voting"""
        pred_array = np.array([pred for pred in predictions.values()])
        # Get mode (most common prediction) for each sample
        from scipy import stats
        return stats.mode(pred_array, axis=0)[0].flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, strategy: str = 'weighted') -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            X: Feature matrix
            y: True labels
            strategy: Voting strategy
            
        Returns:
            Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Get predictions
        y_pred = self.predict(X, strategy=strategy)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='macro', zero_division=0
        )
        cm = confusion_matrix(y, y_pred)
        
        # Individual model predictions for comparison
        individual_accs = {}
        for model_name, model in self.models.items():
            pred = model.predict(X)
            individual_accs[model_name] = accuracy_score(y, pred)
        
        results = {
            'ensemble_accuracy': accuracy,
            'ensemble_precision': precision,
            'ensemble_recall': recall,
            'ensemble_f1': f1,
            'confusion_matrix': cm.tolist(),
            'strategy': strategy,
            'individual_accuracies': individual_accs,
            'improvement': accuracy - max(individual_accs.values())
        }
        
        return results
    
    def get_model_agreement(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check agreement between models
        
        Args:
            X: Feature matrix
            
        Returns:
            agreement_rate: Fraction of models agreeing per sample
            unanimous: Boolean array of unanimous predictions
        """
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        pred_array = np.array([pred for pred in predictions.values()])
        
        # Count agreements
        n_models = len(self.models)
        agreement_counts = np.zeros(len(X))
        
        for i in range(len(X)):
            # Count most common prediction
            unique, counts = np.unique(pred_array[:, i], return_counts=True)
            agreement_counts[i] = counts.max()
        
        agreement_rate = agreement_counts / n_models
        unanimous = agreement_counts == n_models
        
        return agreement_rate, unanimous


def main():
    """Demo: Load ensemble and evaluate"""
    print("=" * 80)
    print("ENSEMBLE PREDICTOR DEMO")
    print("=" * 80)
    
    # Initialize ensemble
    ensemble = EnsemblePredictor()
    
    # Load models
    ensemble.load_models()
    
    print("\n" + "=" * 80)
    print("ENSEMBLE READY FOR PREDICTIONS")
    print("=" * 80)
    print("\nUsage:")
    print("  predictions = ensemble.predict(X, strategy='weighted')")
    print("  probabilities = ensemble.predict_proba(X)")
    print("  metrics = ensemble.evaluate(X_test, y_test)")
    print("  agreement, unanimous = ensemble.get_model_agreement(X)")
    print("\nStrategies:")
    print("  - 'weighted': Weight by test accuracy (recommended)")
    print("  - 'soft': Average probabilities")
    print("  - 'hard': Majority vote")
    
    return ensemble


if __name__ == "__main__":
    ensemble = main()
