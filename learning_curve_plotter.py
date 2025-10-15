"""
Learning Curve Plotting Module
Generates and saves training/validation plots for all models
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LearningCurvePlotter:
    """
    Handles plotting and saving of learning curves for all model types
    """
    
    def __init__(self, save_dir: str = "Model_output/learning_curves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_neural_network_curves(self, model_name: str, history: Dict[str, List[float]]):
        """
        Plot training curves for neural networks (loss and accuracy)
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add best validation accuracy annotation
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        ax2.annotate(f'Best: {best_val_acc:.4f} (Epoch {best_epoch})', 
                    xy=(best_epoch, best_val_acc), 
                    xytext=(best_epoch + len(epochs)*0.1, best_val_acc + 0.01),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f'{model_name}_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"→ Learning curves saved: {save_path}")
        return save_path
    
    def plot_ensemble_comparison(self, model_results: Dict[str, Dict]):
        """
        Plot comparison of all models' final validation accuracies
        """
        model_names = []
        accuracies = []
        model_types = []
        
        for name, result in model_results.items():
            model_names.append(name)
            accuracies.append(result['accuracy'])
            
            # Determine model type for coloring
            if result['type'] == 'neural_network':
                model_types.append('Neural Network')
            elif result['type'] in ['sklearn', 'tree_ensemble']:
                model_types.append('Tree/Classical')
            else:
                model_types.append('Other')
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Create color map
        colors = {'Neural Network': 'skyblue', 'Tree/Classical': 'lightgreen', 'Other': 'lightcoral'}
        bar_colors = [colors[mt] for mt in model_types]
        
        bars = plt.bar(range(len(model_names)), accuracies, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Customize plot
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[mt], alpha=0.8) 
                          for mt in colors.keys()]
        plt.legend(legend_elements, colors.keys(), loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"→ Model comparison saved: {save_path}")
        return save_path
    
    def plot_ensemble_weights(self, ensemble_weights: Dict[str, float]):
        """
        Plot ensemble weights as a pie chart and bar chart
        """
        if not ensemble_weights:
            return None
            
        names = list(ensemble_weights.keys())
        weights = list(ensemble_weights.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        wedges, texts, autotexts = ax1.pie(weights, labels=names, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Ensemble Weights Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(names, weights, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Ensemble Weights', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Weight')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / 'ensemble_weights.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"→ Ensemble weights plot saved: {save_path}")
        return save_path


def integrate_learning_curves(model_results: Dict, ensemble_weights: Dict = None):
    """
    Main function to generate all learning curve plots
    """
    print("\n=== Generating Learning Curves ===")
    
    plotter = LearningCurvePlotter()
    
    # Plot individual model curves
    for model_name, result in model_results.items():
        if result['type'] == 'neural_network' and 'history' in result:
            plotter.plot_neural_network_curves(model_name, result['history'])
    
    # Plot model comparison
    plotter.plot_ensemble_comparison(model_results)
    
    # Plot ensemble weights if available
    if ensemble_weights:
        plotter.plot_ensemble_weights(ensemble_weights)
    
    print("✅ All learning curves generated successfully!")
    return plotter.save_dir