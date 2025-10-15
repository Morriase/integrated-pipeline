"""
Temporal Cross-Validation Module
Implements time-aware k-fold validation for robust ensemble evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

class TemporalValidator:
    """
    Handles temporal cross-validation for time-series aware model evaluation
    """
    
    def __init__(self, save_dir: str = "Model_output/robustness"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fold_results = []
    
    def create_temporal_splits(self, timestamps: np.ndarray, k: int = 5, 
                              validation_ratio: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-based splits for cross-validation
        
        Args:
            timestamps: Array of timestamps or indices
            k: Number of folds
            validation_ratio: Ratio of data to use for validation in each fold
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(timestamps)
        fold_size = n_samples // k
        val_size = int(fold_size * validation_ratio)
        
        splits = []
        
        for i in range(k):
            # Calculate fold boundaries
            fold_start = i * fold_size
            fold_end = min((i + 1) * fold_size, n_samples)
            
            if i == k - 1:  # Last fold gets remaining samples
                fold_end = n_samples
            
            # Training data: everything before validation period
            train_end = fold_end - val_size
            train_indices = np.arange(0, train_end)
            
            # Validation data: last portion of current fold
            val_indices = np.arange(train_end, fold_end)
            
            if len(val_indices) > 0 and len(train_indices) > 0:
                splits.append((train_indices, val_indices))
        
        return splits
    
    def evaluate_fold(self, fold_idx: int, train_data: Dict, val_data: Dict, 
                     ensemble_manager, system) -> Dict[str, float]:
        """
        Evaluate a single fold
        
        Args:
            fold_idx: Fold index
            train_data: Training data dictionary
            val_data: Validation data dictionary
            ensemble_manager: EnsembleManager instance
            system: IntegratedSMCSystem instance
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n--- Evaluating Fold {fold_idx + 1} ---")
        
        # Train models on fold training data
        fold_models = ensemble_manager.train_all_models(
            train_data['features'], 
            train_data['labels']
        )
        
        # Calculate ensemble weights on fold validation data
        fold_weights = ensemble_manager.calculate_ensemble_weights(
            val_data['features'][:len(val_data['features'])//2],  # Use first half for weights
            val_data['labels'][:len(val_data['labels'])//2]
        )
        
        # Evaluate on second half of validation data
        test_features = val_data['features'][len(val_data['features'])//2:]
        test_labels = val_data['labels'][len(val_data['labels'])//2:]
        
        # Get ensemble predictions
        ensemble_preds = ensemble_manager.predict_ensemble(
            test_features, method='weighted_average'
        )
        
        # Calculate metrics
        ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
        
        # Individual model accuracies
        individual_accuracies = {}
        for model_name, model_info in fold_models.items():
            if model_info['type'] == 'neural_network':
                model = model_info['model']
                model.eval()
                
                import torch
                device = next(model.parameters()).device
                with torch.no_grad():
                    X_tensor = torch.tensor(test_features, dtype=torch.float32, device=device)
                    logits = model(X_tensor)
                    preds = logits.argmax(dim=1).cpu().numpy()
            else:
                model = model_info['model']
                # Normalize features
                test_features_norm = (test_features - ensemble_manager.feature_scalers['mean']) / ensemble_manager.feature_scalers['std']
                
                # Use numpy arrays directly (models were trained with numpy arrays)
                # Suppress sklearn feature name warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                    
                    preds = model.predict(test_features_norm)
            
            individual_accuracies[model_name] = accuracy_score(test_labels, preds)
        
        fold_result = {
            'fold': fold_idx,
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies,
            'ensemble_weights': fold_weights,
            'n_train': len(train_data['features']),
            'n_val': len(test_features)
        }
        
        print(f"Fold {fold_idx + 1} Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        return fold_result
    
    def run_temporal_validation(self, dataset: Dict, ensemble_manager, system, 
                               k: int = 5) -> Dict[str, Any]:
        """
        Run complete temporal cross-validation
        
        Args:
            dataset: Complete dataset dictionary
            ensemble_manager: EnsembleManager instance
            system: IntegratedSMCSystem instance
            k: Number of folds
        
        Returns:
            Validation results dictionary
        """
        print(f"\n=== Running Temporal {k}-Fold Cross-Validation ===")
        
        # Get base data for temporal splits
        base_data = dataset['base_data']
        all_features = np.vstack([
            base_data['train']['features'],
            base_data['val']['features'],
            base_data['test']['features']
        ])
        all_labels = np.hstack([
            base_data['train']['labels'],
            base_data['val']['labels'],
            base_data['test']['labels']
        ])
        
        # Create temporal splits
        timestamps = np.arange(len(all_features))  # Use indices as timestamps
        splits = self.create_temporal_splits(timestamps, k=k)
        
        print(f"Created {len(splits)} temporal folds")
        
        # Evaluate each fold
        fold_results = []
        for i, (train_idx, val_idx) in enumerate(splits):
            train_data = {
                'features': all_features[train_idx],
                'labels': all_labels[train_idx]
            }
            val_data = {
                'features': all_features[val_idx],
                'labels': all_labels[val_idx]
            }
            
            fold_result = self.evaluate_fold(i, train_data, val_data, ensemble_manager, system)
            fold_results.append(fold_result)
        
        # Calculate summary statistics
        ensemble_scores = [r['ensemble_accuracy'] for r in fold_results]
        
        validation_summary = {
            'fold_results': fold_results,
            'ensemble_mean': np.mean(ensemble_scores),
            'ensemble_std': np.std(ensemble_scores),
            'ensemble_scores': ensemble_scores,
            'n_folds': len(fold_results)
        }
        
        print(f"\n=== Temporal Validation Summary ===")
        print(f"Ensemble Accuracy: {validation_summary['ensemble_mean']:.4f} ± {validation_summary['ensemble_std']:.4f}")
        print(f"Score Range: [{min(ensemble_scores):.4f}, {max(ensemble_scores):.4f}]")
        
        # Generate plots
        self.plot_robustness_results(validation_summary)
        
        return validation_summary
    
    def plot_robustness_results(self, validation_summary: Dict[str, Any]):
        """
        Generate robustness plots from validation results
        """
        fold_results = validation_summary['fold_results']
        ensemble_scores = validation_summary['ensemble_scores']
        
        # Create comprehensive robustness plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Fold accuracy plot with error bars
        folds = range(1, len(ensemble_scores) + 1)
        mean_score = validation_summary['ensemble_mean']
        std_score = validation_summary['ensemble_std']
        
        ax1.bar(folds, ensemble_scores, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax1.axhline(y=mean_score + std_score, color='orange', linestyle=':', alpha=0.7, label=f'±1 Std: {std_score:.4f}')
        ax1.axhline(y=mean_score - std_score, color='orange', linestyle=':', alpha=0.7)
        ax1.set_title('Ensemble Accuracy Across Folds', fontweight='bold')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, score in enumerate(ensemble_scores):
            ax1.text(i + 1, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Distribution of ensemble scores
        ax2.hist(ensemble_scores, bins=max(3, len(ensemble_scores)//2), alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(x=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax2.set_title('Distribution of Ensemble Scores', fontweight='bold')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual model performance across folds
        model_names = list(fold_results[0]['individual_accuracies'].keys())
        model_scores = {name: [] for name in model_names}
        
        for fold_result in fold_results:
            for name, score in fold_result['individual_accuracies'].items():
                model_scores[name].append(score)
        
        # Plot top 5 models to avoid clutter
        top_models = sorted(model_names, key=lambda x: np.mean(model_scores[x]), reverse=True)[:5]
        
        for i, model_name in enumerate(top_models):
            scores = model_scores[model_name]
            ax3.plot(folds, scores, marker='o', label=model_name, linewidth=2)
        
        ax3.set_title('Top 5 Individual Model Performance', fontweight='bold')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Accuracy')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Ensemble vs best individual model
        best_individual_scores = []
        for fold_result in fold_results:
            best_score = max(fold_result['individual_accuracies'].values())
            best_individual_scores.append(best_score)
        
        x = np.arange(len(folds))
        width = 0.35
        
        ax4.bar(x - width/2, ensemble_scores, width, label='Ensemble', alpha=0.8, color='gold')
        ax4.bar(x + width/2, best_individual_scores, width, label='Best Individual', alpha=0.8, color='lightcoral')
        
        ax4.set_title('Ensemble vs Best Individual Model', fontweight='bold')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Accuracy')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Fold {i}' for i in folds])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / 'temporal_validation_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"→ Robustness plots saved: {save_path}")
        
        # Save detailed results
        results_path = self.save_dir / 'temporal_validation_results.txt'
        with open(results_path, 'w') as f:
            f.write("=== Temporal Cross-Validation Results ===\n\n")
            f.write(f"Number of Folds: {validation_summary['n_folds']}\n")
            f.write(f"Ensemble Mean Accuracy: {validation_summary['ensemble_mean']:.4f}\n")
            f.write(f"Ensemble Std Accuracy: {validation_summary['ensemble_std']:.4f}\n")
            f.write(f"Ensemble Score Range: [{min(ensemble_scores):.4f}, {max(ensemble_scores):.4f}]\n\n")
            
            for i, fold_result in enumerate(fold_results):
                f.write(f"--- Fold {i+1} ---\n")
                f.write(f"Ensemble Accuracy: {fold_result['ensemble_accuracy']:.4f}\n")
                f.write(f"Training Samples: {fold_result['n_train']}\n")
                f.write(f"Validation Samples: {fold_result['n_val']}\n")
                f.write("Individual Model Accuracies:\n")
                for model_name, acc in fold_result['individual_accuracies'].items():
                    f.write(f"  {model_name}: {acc:.4f}\n")
                f.write("\n")
        
        print(f"→ Detailed results saved: {results_path}")
        
        return save_path