"""
Visualize Training Curves for Deep Learning Models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(results_file='/kaggle/working/all_models_results.json', 
                         output_dir='/kaggle/working/plots'):
    """Plot training curves for all models"""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot for each symbol
    for symbol, models in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} - Training Curves', fontsize=16)
        
        # Neural Network
        if 'NeuralNetwork' in models and 'history' in models['NeuralNetwork']:
            history = models['NeuralNetwork']['history']
            
            # Loss curves
            axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_title('Neural Network - Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy curves
            axes[0, 1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
            axes[0, 1].plot(history['val_acc'], label='Val Acc', alpha=0.7)
            axes[0, 1].set_title('Neural Network - Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add test accuracy line
            if 'test_metrics' in models['NeuralNetwork']:
                test_acc = models['NeuralNetwork']['test_metrics']['accuracy']
                axes[0, 1].axhline(y=test_acc, color='r', linestyle='--', 
                                  label=f'Test Acc: {test_acc:.3f}', alpha=0.7)
                axes[0, 1].legend()
        
        # LSTM
        if 'LSTM' in models and 'history' in models['LSTM']:
            history = models['LSTM']['history']
            
            # Loss curves
            axes[1, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[1, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[1, 0].set_title('LSTM - Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Accuracy curves
            axes[1, 1].plot(history['train_acc'], label='Train Acc', alpha=0.7)
            axes[1, 1].plot(history['val_acc'], label='Val Acc', alpha=0.7)
            axes[1, 1].set_title('LSTM - Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add test accuracy line
            if 'test_metrics' in models['LSTM']:
                test_acc = models['LSTM']['test_metrics']['accuracy']
                axes[1, 1].axhline(y=test_acc, color='r', linestyle='--', 
                                  label=f'Test Acc: {test_acc:.3f}', alpha=0.7)
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{symbol}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved {symbol} training curves")
    
    # Create summary comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison (Test Accuracy)', fontsize=16)
    
    symbols = list(results.keys())
    rf_scores = []
    nn_scores = []
    lstm_scores = []
    
    for symbol in symbols:
        models = results[symbol]
        rf_scores.append(models.get('RandomForest', {}).get('test_metrics', {}).get('accuracy', 0))
        nn_scores.append(models.get('NeuralNetwork', {}).get('test_metrics', {}).get('accuracy', 0))
        lstm_scores.append(models.get('LSTM', {}).get('test_metrics', {}).get('accuracy', 0))
    
    x = np.arange(len(symbols))
    width = 0.25
    
    axes[0].bar(x - width, rf_scores, width, label='Random Forest', alpha=0.8)
    axes[0].bar(x, nn_scores, width, label='Neural Network', alpha=0.8)
    axes[0].bar(x + width, lstm_scores, width, label='LSTM', alpha=0.8)
    axes[0].set_xlabel('Symbol')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy by Symbol')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(symbols, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Baseline')
    
    # Average performance
    avg_scores = [np.mean(rf_scores), np.mean(nn_scores), np.mean(lstm_scores)]
    axes[1].bar(['Random Forest', 'Neural Network', 'LSTM'], avg_scores, alpha=0.8)
    axes[1].set_ylabel('Average Test Accuracy')
    axes[1].set_title('Average Performance Across All Symbols')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(avg_scores):
        axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
    
    # Overfitting analysis (train vs test gap)
    gaps_nn = []
    gaps_lstm = []
    
    for symbol in symbols:
        models = results[symbol]
        
        if 'NeuralNetwork' in models and 'history' in models['NeuralNetwork']:
            train_acc = models['NeuralNetwork']['history']['train_acc'][-1]
            test_acc = models['NeuralNetwork']['test_metrics']['accuracy']
            gaps_nn.append(train_acc - test_acc)
        
        if 'LSTM' in models and 'history' in models['LSTM']:
            train_acc = models['LSTM']['history']['train_acc'][-1]
            test_acc = models['LSTM']['test_metrics']['accuracy']
            gaps_lstm.append(train_acc - test_acc)
    
    axes[2].bar(x - width/2, gaps_nn, width, label='Neural Network', alpha=0.8)
    axes[2].bar(x + width/2, gaps_lstm, width, label='LSTM', alpha=0.8)
    axes[2].set_xlabel('Symbol')
    axes[2].set_ylabel('Train-Test Gap')
    axes[2].set_title('Overfitting Analysis (Train - Test Accuracy)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(symbols, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='High Overfit')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved model comparison plot")
    print(f"\n📊 Summary:")
    print(f"  Random Forest avg: {np.mean(rf_scores):.1%}")
    print(f"  Neural Network avg: {np.mean(nn_scores):.1%}")
    print(f"  LSTM avg: {np.mean(lstm_scores):.1%}")
    print(f"\n  NN overfitting: {np.mean(gaps_nn):.1%}")
    print(f"  LSTM overfitting: {np.mean(gaps_lstm):.1%}")


if __name__ == "__main__":
    plot_training_curves()
