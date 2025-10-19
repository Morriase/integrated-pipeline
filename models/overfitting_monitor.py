"""
Overfitting Monitor for Training Metrics Tracking

Monitors training and validation metrics during model training to detect
and report overfitting issues. Provides visualization and structured logging.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class OverfittingMonitor:
    """
    Monitors training metrics to detect and report overfitting.
    
    Tracks train/val accuracy and loss at each epoch, calculates the
    train-validation gap, and generates learning curves for visualization.
    """
    
    def __init__(self, warning_threshold: float = 0.15):
        """
        Initialize the overfitting monitor.
        
        Args:
            warning_threshold: Gap threshold above which overfitting is flagged (default 15%)
        """
        self.warning_threshold = warning_threshold
        self.history = {
            'epoch': [],
            'train_accuracy': [],
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        self.start_time = datetime.now()
    
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """
        Record metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary with 'accuracy' and 'loss' keys for training
            val_metrics: Dictionary with 'accuracy' and 'loss' keys for validation
        """
        self.history['epoch'].append(epoch)
        self.history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        self.history['train_loss'].append(train_metrics.get('loss', 0.0))
        self.history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        self.history['val_loss'].append(val_metrics.get('loss', 0.0))
    
    def calculate_gap(self) -> float:
        """
        Calculate train-validation accuracy gap.
        
        Returns:
            The difference between final train and validation accuracy
        """
        if not self.history['train_accuracy'] or not self.history['val_accuracy']:
            return 0.0
        
        train_acc = self.history['train_accuracy'][-1]
        val_acc = self.history['val_accuracy'][-1]
        return train_acc - val_acc
    
    def is_overfitting(self) -> bool:
        """
        Check if overfitting threshold is exceeded.
        
        Returns:
            True if train-val gap exceeds warning threshold
        """
        gap = self.calculate_gap()
        return gap > self.warning_threshold
    
    def generate_learning_curves(self, save_path: str):
        """
        Create visualization of train/val curves.
        
        Args:
            save_path: Path where the plot image will be saved
        """
        if not self.history['epoch']:
            print("No training history to plot")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history['epoch']
        
        # Plot accuracy
        ax1.plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, self.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Add overfitting warning if detected
        if self.is_overfitting():
            gap = self.calculate_gap()
            ax1.text(0.5, 0.95, f'⚠ Overfitting Detected (Gap: {gap:.1%})',
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Plot loss
        ax2.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Learning curves saved to: {save_path}")
    
    def get_summary(self) -> Dict:
        """
        Return overfitting summary statistics.
        
        Returns:
            Dictionary containing comprehensive overfitting metrics
        """
        if not self.history['epoch']:
            return {
                'status': 'no_data',
                'message': 'No training history available'
            }
        
        gap = self.calculate_gap()
        is_overfitting = self.is_overfitting()
        
        # Find best epoch (highest validation accuracy)
        best_val_idx = np.argmax(self.history['val_accuracy'])
        best_epoch = self.history['epoch'][best_val_idx]
        
        summary = {
            'status': 'overfitting' if is_overfitting else 'healthy',
            'train_val_gap': gap,
            'is_overfitting': is_overfitting,
            'warning_threshold': self.warning_threshold,
            'final_metrics': {
                'train_accuracy': self.history['train_accuracy'][-1],
                'val_accuracy': self.history['val_accuracy'][-1],
                'train_loss': self.history['train_loss'][-1],
                'val_loss': self.history['val_loss'][-1]
            },
            'best_epoch': {
                'epoch': best_epoch,
                'train_accuracy': self.history['train_accuracy'][best_val_idx],
                'val_accuracy': self.history['val_accuracy'][best_val_idx],
                'train_loss': self.history['train_loss'][best_val_idx],
                'val_loss': self.history['val_loss'][best_val_idx]
            },
            'total_epochs': len(self.history['epoch']),
            'training_duration': str(datetime.now() - self.start_time)
        }
        
        return summary
    
    def save_to_json(self, save_path: str):
        """
        Save overfitting metrics to structured JSON log file.
        
        Args:
            save_path: Path where the JSON file will be saved
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'history': self.history
        }
        
        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Overfitting metrics saved to: {save_path}")
    
    def print_summary(self):
        """Print a formatted summary of overfitting metrics to console."""
        summary = self.get_summary()
        
        if summary['status'] == 'no_data':
            print(summary['message'])
            return
        
        print("\n" + "="*60)
        print("OVERFITTING MONITOR SUMMARY")
        print("="*60)
        
        status_symbol = "⚠ WARNING" if summary['is_overfitting'] else "✓ HEALTHY"
        print(f"Status: {status_symbol}")
        print(f"Train-Val Gap: {summary['train_val_gap']:.2%} (Threshold: {summary['warning_threshold']:.0%})")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Training Duration: {summary['training_duration']}")
        
        print("\nFinal Metrics:")
        print(f"  Train Accuracy: {summary['final_metrics']['train_accuracy']:.4f}")
        print(f"  Val Accuracy:   {summary['final_metrics']['val_accuracy']:.4f}")
        print(f"  Train Loss:     {summary['final_metrics']['train_loss']:.4f}")
        print(f"  Val Loss:       {summary['final_metrics']['val_loss']:.4f}")
        
        print("\nBest Epoch (Highest Val Accuracy):")
        print(f"  Epoch:          {summary['best_epoch']['epoch']}")
        print(f"  Train Accuracy: {summary['best_epoch']['train_accuracy']:.4f}")
        print(f"  Val Accuracy:   {summary['best_epoch']['val_accuracy']:.4f}")
        print(f"  Train Loss:     {summary['best_epoch']['train_loss']:.4f}")
        print(f"  Val Loss:       {summary['best_epoch']['val_loss']:.4f}")
        
        print("="*60 + "\n")
