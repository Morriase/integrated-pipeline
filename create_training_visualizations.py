"""
Create Comprehensive Training Visualizations
Generates all training curves and comparison charts
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
if Path('/kaggle/working').exists():
    OUTPUT_DIR = Path('/kaggle/working/Training_Images')
    RESULTS_FILE = Path('/kaggle/working/Model-output/training_results.json')
else:
    OUTPUT_DIR = Path('Training_Images')
    RESULTS_FILE = Path('models/trained/training_results.json')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_results():
    """Load training results JSON"""
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def plot_model_comparison(results):
    """Create model comparison charts"""
    print("\n📊 Creating model comparison charts...")
    
    models_data = results['training_results']['UNIFIED']
    
    # Extract metrics
    models = []
    test_acc = []
    win_rates = []
    profit_factors = []
    ev_per_trade = []
    
    for model_name, model_results in models_data.items():
        if 'error' in model_results:
            continue
        
        test_metrics = model_results.get('test_metrics', {})
        
        models.append(model_name)
        test_acc.append(test_metrics.get('accuracy', 0))
        win_rates.append(test_metrics.get('win_rate', 0))
        profit_factors.append(test_metrics.get('profit_factor', 0))
        ev_per_trade.append(test_metrics.get('expected_value_per_trade', 0))
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    # 1. Test Accuracy
    bars1 = ax1.bar(models, test_acc, color=colors[:len(models)])
    ax1.set_title('Test Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, test_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.1%}', ha='center', fontweight='bold')
    
    # 2. Win Rate
    bars2 = ax2.bar(models, win_rates, color=colors[:len(models)])
    ax2.set_title('Win Rate (Excluding Timeouts)', fontweight='bold')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Break-even')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    for i, (bar, val) in enumerate(zip(bars2, win_rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.1%}', ha='center', fontweight='bold')
    
    # 3. Profit Factor
    bars3 = ax3.bar(models, profit_factors, color=colors[:len(models)])
    ax3.set_title('Profit Factor (1:2 R:R)', fontweight='bold')
    ax3.set_ylabel('Profit Factor')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()
    for i, (bar, val) in enumerate(zip(bars3, profit_factors)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05, 
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # 4. Expected Value per Trade
    bars4 = ax4.bar(models, ev_per_trade, color=colors[:len(models)])
    ax4.set_title('Expected Value per Trade', fontweight='bold')
    ax4.set_ylabel('EV (R)')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    for i, (bar, val) in enumerate(zip(bars4, ev_per_trade)):
        color = 'green' if val > 0 else 'red'
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02 if val > 0 else val - 0.05, 
                f'{val:.2f}R', ha='center', fontweight='bold', color=color)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results):
    """Create confusion matrix visualizations"""
    print("\n📊 Creating confusion matrices...")
    
    models_data = results['training_results']['UNIFIED']
    
    # Filter models with valid results
    valid_models = {name: data for name, data in models_data.items() 
                   if 'error' not in data and 'test_metrics' in data}
    
    n_models = len(valid_models)
    if n_models == 0:
        print("  ⚠️ No valid models to plot")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
    
    for idx, (model_name, model_data) in enumerate(valid_models.items()):
        cm = np.array(model_data['test_metrics']['confusion_matrix'])
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Loss', 'Timeout', 'Win'],
                   yticklabels=['Loss', 'Timeout', 'Win'],
                   cbar=False)
        axes[idx].set_title(f'{model_name}', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_training_summary(results):
    """Create overall training summary"""
    print("\n📊 Creating training summary...")
    
    models_data = results['training_results']['UNIFIED']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    models = []
    metrics_data = {
        'Test Accuracy': [],
        'Win Rate': [],
        'Profit Factor': [],
        'EV/Trade': []
    }
    
    for model_name, model_results in models_data.items():
        if 'error' in model_results:
            continue
        
        test_metrics = model_results.get('test_metrics', {})
        
        models.append(model_name)
        metrics_data['Test Accuracy'].append(test_metrics.get('accuracy', 0) * 100)
        metrics_data['Win Rate'].append(test_metrics.get('win_rate', 0) * 100)
        metrics_data['Profit Factor'].append(test_metrics.get('profit_factor', 0) * 50)  # Scale for visibility
        metrics_data['EV/Trade'].append(test_metrics.get('expected_value_per_trade', 0) * 100)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        offset = width * (idx - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[idx])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Scaled Metrics', fontweight='bold')
    ax.set_title('Training Summary - All Metrics (Scaled for Comparison)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add note
    note = "Note: Profit Factor scaled by 50, EV/Trade scaled by 100 for visibility"
    ax.text(0.5, -0.15, note, transform=ax.transAxes, 
           ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def create_all_visualizations():
    """Create all training visualizations"""
    print("=" * 80)
    print("CREATING TRAINING VISUALIZATIONS")
    print("=" * 80)
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    
    # Load results
    results = load_training_results()
    if results is None:
        print("\n❌ Cannot create visualizations without training results")
        return
    
    # Create visualizations
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_training_summary(results)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("=" * 80)
    print(f"\n📁 Saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {file.name}")
    
    print("\n🎉 Visualization complete!")


if __name__ == "__main__":
    create_all_visualizations()



def plot_training_curves_comparison(results):
    """Create combined training curves for all models"""
    print("\n📊 Creating training curves comparison...")
    
    models_data = results['training_results']['UNIFIED']
    
    # Collect models with training history
    models_with_history = {}
    for model_name, model_results in models_data.items():
        if 'error' in model_results:
            continue
        history = model_results.get('history', {})
        if history and ('train_loss' in history or 'train_accuracy' in history):
            models_with_history[model_name] = history
    
    if not models_with_history:
        print("  ⚠️ No training history available")
        return
    
    # Create figure with 2 rows: Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Training Curves Comparison (All Models)', fontsize=16, fontweight='bold')
    
    colors = {'RandomForest': '#2ecc71', 'XGBoost': '#3498db', 
              'NeuralNetwork': '#e74c3c', 'LSTM': '#f39c12'}
    
    # Plot Loss curves
    for model_name, history in models_with_history.items():
        if 'train_loss' in history and 'val_loss' in history:
            train_loss = history['train_loss']
            val_loss = history['val_loss']
            
            if isinstance(train_loss, list) and len(train_loss) > 0:
                epochs = list(range(1, len(train_loss) + 1))
                color = colors.get(model_name, '#95a5a6')
                
                ax1.plot(epochs, train_loss, '-', color=color, alpha=0.7, 
                        linewidth=2, label=f'{model_name} (Train)')
                ax1.plot(epochs, val_loss, '--', color=color, alpha=0.7, 
                        linewidth=2, label=f'{model_name} (Val)')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy curves
    for model_name, history in models_with_history.items():
        if 'train_acc' in history and 'val_acc' in history:
            train_acc = history['train_acc']
            val_acc = history['val_acc']
            
            if isinstance(train_acc, list) and len(train_acc) > 0:
                epochs = list(range(1, len(train_acc) + 1))
                color = colors.get(model_name, '#95a5a6')
                
                ax2.plot(epochs, train_acc, '-', color=color, alpha=0.7, 
                        linewidth=2, label=f'{model_name} (Train)')
                ax2.plot(epochs, val_acc, '--', color=color, alpha=0.7, 
                        linewidth=2, label=f'{model_name} (Val)')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_individual_training_curves():
    """Plot individual training curves for each model"""
    print("\n📊 Checking for individual training curves...")
    
    # Check for NN learning curves (already generated during training)
    nn_curves = OUTPUT_DIR / 'UNIFIED_NN_learning_curves.png'
    
    # Also check old location
    old_nn_curves = Path('models/trained/UNIFIED_NN_learning_curves.png')
    
    if nn_curves.exists():
        print(f"  ✅ Neural Network curves found: {nn_curves}")
    elif old_nn_curves.exists():
        # Copy from old location to new location
        import shutil
        shutil.copy(old_nn_curves, nn_curves)
        print(f"  ✅ Neural Network curves copied from old location")
    else:
        print(f"  ℹ️ Neural Network curves will be generated during next training")
    
    # Note: RF and XGB don't have epoch-based training, so no curves for them
    print("  ℹ️ RandomForest and XGBoost don't have epoch-based training curves")


def plot_overfitting_analysis(results):
    """Create overfitting analysis visualization"""
    print("\n📊 Creating overfitting analysis...")
    
    models_data = results['training_results']['UNIFIED']
    
    # Extract train-val gaps
    models = []
    train_acc = []
    val_acc = []
    gaps = []
    
    for model_name, model_results in models_data.items():
        if 'error' in model_results:
            continue
        
        history = model_results.get('history', {})
        val_metrics = model_results.get('val_metrics', {})
        
        # Get training accuracy
        if 'train_accuracy' in history:
            t_acc = history['train_accuracy']
            if isinstance(t_acc, list):
                t_acc = t_acc[-1]  # Last epoch
        elif 'final_train_accuracy' in history:
            t_acc = history['final_train_accuracy']
        else:
            continue
        
        v_acc = val_metrics.get('accuracy', 0)
        gap = t_acc - v_acc
        
        models.append(model_name)
        train_acc.append(t_acc)
        val_acc.append(v_acc)
        gaps.append(gap)
    
    if not models:
        print("  ⚠️ No overfitting data available")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Overfitting Analysis', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.35
    
    # Plot 1: Train vs Val Accuracy
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#3498db')
    bars2 = ax1.bar(x + width/2, val_acc, width, label='Val Accuracy', color='#e74c3c')
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Train vs Validation Accuracy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Train-Val Gap
    colors_gap = ['#2ecc71' if g < 0.15 else '#f39c12' if g < 0.30 else '#e74c3c' 
                  for g in gaps]
    bars3 = ax2.bar(models, gaps, color=colors_gap)
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Train-Val Gap', fontweight='bold')
    ax2.set_title('Overfitting Gap (Train - Val)', fontweight='bold')
    ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Warning (15%)')
    ax2.axhline(y=0.30, color='red', linestyle='--', alpha=0.5, label='Critical (30%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels and status (use text instead of emoji to avoid font warnings)
    for bar, gap in zip(bars3, gaps):
        height = bar.get_height()
        status = 'OK' if gap < 0.15 else 'WARN' if gap < 0.30 else 'BAD'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{status}\n{gap:.1%}', ha='center', va='bottom', fontsize=9,
               fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'overfitting_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


# Update the main function
def create_all_visualizations_enhanced():
    """Create all training visualizations including curves"""
    print("=" * 80)
    print("CREATING COMPREHENSIVE TRAINING VISUALIZATIONS")
    print("=" * 80)
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    
    # Load results
    results = load_training_results()
    if results is None:
        print("\n❌ Cannot create visualizations without training results")
        return
    
    # Create all visualizations
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_training_summary(results)
    plot_training_curves_comparison(results)
    plot_individual_training_curves()
    plot_overfitting_analysis(results)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("=" * 80)
    print(f"\n📁 Saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {file.name}")
    
    print("\n🎉 Visualization complete!")
    print("\nTo view in Kaggle, run:")
    print("  from IPython.display import Image, display")
    print(f"  display(Image('{OUTPUT_DIR}/model_comparison.png'))")


if __name__ == "__main__":
    create_all_visualizations_enhanced()
