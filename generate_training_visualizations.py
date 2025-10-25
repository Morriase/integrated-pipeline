"""
Generate comprehensive training visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_results():
    """Load training results"""
    if Path('/kaggle/working/Model-output').exists():
        results_path = '/kaggle/working/Model-output/training_results.json'
        output_dir = '/kaggle/working/Training_Images'
    else:
        results_path = 'models/trained/training_results.json'
        output_dir = 'Training_Images'
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results, output_dir


def plot_model_comparison(results, output_dir):
    """Compare all models performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = []
    train_acc = []
    val_acc = []
    test_acc = []
    
    for model_name, model_data in results.items():
        if 'error' not in model_data:
            models.append(model_name)
            train_acc.append(model_data.get('train_accuracy', 0))
            val_acc.append(model_data.get('val_accuracy', 0))
            test_acc.append(model_data.get('test_accuracy', 0))
    
    x = np.arange(len(models))
    width = 0.25
    
    # Accuracy comparison
    axes[0, 0].bar(x - width, train_acc, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x, val_acc, width, label='Validation', alpha=0.8)
    axes[0, 0].bar(x + width, test_acc, width, label='Test', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Trading metrics
    win_rates = []
    profit_factors = []
    expected_values = []
    
    for model_name in models:
        model_data = results[model_name]
        test_metrics = model_data.get('test_metrics', {})
        win_rates.append(test_metrics.get('win_rate', 0))
        pf = test_metrics.get('profit_factor', 0)
        profit_factors.append(min(pf, 5))  # Cap at 5 for visualization
        expected_values.append(test_metrics.get('expected_value_per_trade', 0))
    
    # Win Rate
    axes[0, 1].bar(models, win_rates, alpha=0.8, color='green')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].set_title('Win Rate Comparison')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Profit Factor
    axes[1, 0].bar(models, profit_factors, alpha=0.8, color='blue')
    axes[1, 0].set_ylabel('Profit Factor (capped at 5)')
    axes[1, 0].set_title('Profit Factor Comparison')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', label='Break-even')
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Expected Value
    axes[1, 1].bar(models, expected_values, alpha=0.8, color='purple')
    axes[1, 1].set_ylabel('Expected Value (R)')
    axes[1, 1].set_title('Expected Value per Trade')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', label='Break-even')
    axes[1, 1].set_xticklabels(models, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results, output_dir):
    """Plot confusion matrices for all models"""
    models = [m for m in results.keys() if 'error' not in results[m]]
    
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
    
    for idx, model_name in enumerate(models):
        model_data = results[model_name]
        cm = np.array(model_data['test_metrics']['confusion_matrix'])
        
        # Determine if binary or 3-class
        if cm.shape[0] == 2:
            labels = ['Loss', 'Win']
        else:
            labels = ['Loss', 'Timeout', 'Win']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    output_path = f'{output_dir}/confusion_matrices.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_training_summary(results, output_dir):
    """Create comprehensive training summary"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Training Summary Dashboard', fontsize=18, fontweight='bold')
    
    models = [m for m in results.keys() if 'error' not in results[m]]
    
    # 1. Accuracy Metrics
    ax1 = fig.add_subplot(gs[0, :])
    metrics_data = []
    for model in models:
        data = results[model]
        metrics_data.append([
            data.get('train_accuracy', 0),
            data.get('val_accuracy', 0),
            data.get('test_accuracy', 0)
        ])
    
    x = np.arange(len(models))
    width = 0.25
    ax1.bar(x - width, [m[0] for m in metrics_data], width, label='Train', alpha=0.8)
    ax1.bar(x, [m[1] for m in metrics_data], width, label='Val', alpha=0.8)
    ax1.bar(x + width, [m[2] for m in metrics_data], width, label='Test', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. Trading Performance
    ax2 = fig.add_subplot(gs[1, 0])
    win_rates = [results[m]['test_metrics']['win_rate'] for m in models]
    colors = ['green' if wr > 0.5 else 'red' for wr in win_rates]
    ax2.barh(models, win_rates, color=colors, alpha=0.7)
    ax2.set_xlabel('Win Rate')
    ax2.set_title('Win Rate (Test Set)')
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 3. Profit Factor
    ax3 = fig.add_subplot(gs[1, 1])
    pfs = [min(results[m]['test_metrics']['profit_factor'], 10) for m in models]
    colors = ['green' if pf > 1 else 'red' for pf in pfs]
    ax3.barh(models, pfs, color=colors, alpha=0.7)
    ax3.set_xlabel('Profit Factor (capped at 10)')
    ax3.set_title('Profit Factor (Test Set)')
    ax3.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Expected Value
    ax4 = fig.add_subplot(gs[1, 2])
    evs = [results[m]['test_metrics']['expected_value_per_trade'] for m in models]
    colors = ['green' if ev > 0 else 'red' for ev in evs]
    ax4.barh(models, evs, color=colors, alpha=0.7)
    ax4.set_xlabel('Expected Value (R)')
    ax4.set_title('Expected Value per Trade')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Time
    ax5 = fig.add_subplot(gs[2, 0])
    times = [results[m].get('training_time', 0) for m in models]
    ax5.bar(models, times, alpha=0.7, color='orange')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Training Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Overfitting Analysis
    ax6 = fig.add_subplot(gs[2, 1:])
    gaps = []
    for model in models:
        data = results[model]
        gap = data.get('train_accuracy', 0) - data.get('val_accuracy', 0)
        gaps.append(gap * 100)
    
    colors = ['green' if abs(g) < 5 else 'orange' if abs(g) < 15 else 'red' for g in gaps]
    ax6.bar(models, gaps, color=colors, alpha=0.7)
    ax6.set_ylabel('Train-Val Gap (%)')
    ax6.set_title('Overfitting Analysis (Train - Val Accuracy)')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    ax6.axhline(y=-15, color='red', linestyle='--', alpha=0.5)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    output_path = f'{output_dir}/training_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations"""
    print("=" * 80)
    print("GENERATING TRAINING VISUALIZATIONS")
    print("=" * 80)
    
    # Load results
    results, output_dir = load_results()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Output directory: {output_dir}")
    print(f"📊 Models found: {len([m for m in results.keys() if 'error' not in results[m]])}")
    
    # Generate plots
    print(f"\n🎨 Generating visualizations...")
    
    try:
        plot_model_comparison(results, output_dir)
    except Exception as e:
        print(f"  ⚠️ Model comparison failed: {e}")
    
    try:
        plot_confusion_matrices(results, output_dir)
    except Exception as e:
        print(f"  ⚠️ Confusion matrices failed: {e}")
    
    try:
        plot_training_summary(results, output_dir)
    except Exception as e:
        print(f"  ⚠️ Training summary failed: {e}")
    
    print(f"\n✅ Visualization generation complete!")
    print(f"\n📁 Files saved to: {output_dir}")
    print(f"   - model_comparison.png")
    print(f"   - confusion_matrices.png")
    print(f"   - training_summary.png")
    print(f"   - UNIFIED_NN_learning_curves.png (already exists)")


if __name__ == "__main__":
    main()
