"""
Train Neural Network and LSTM on ALL symbols combined
(Deep learning needs large datasets - 76,982 samples total)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'models'))

from models.neural_network_model import NeuralNetworkSMCModel, TORCH_AVAILABLE
from models.lstm_model import LSTMSMCModel
import pandas as pd
import numpy as np

def train_combined_models():
    """Train NN and LSTM on all symbols combined"""
    
    print("\n" + "="*80)
    print("TRAINING DEEP LEARNING MODELS ON COMBINED DATA")
    print("="*80)
    
    # Load FULL data from processed_smc_data CSV files (Kaggle paths)
    print("\n📂 Loading FULL dataset (ALL SYMBOLS COMBINED)...")
    train_df = pd.read_csv('/kaggle/input/ob-ai-model-2-dataset/Data/processed_smc_data_train.csv')
    val_df = pd.read_csv('/kaggle/input/ob-ai-model-2-dataset/Data/processed_smc_data_val.csv')
    test_df = pd.read_csv('/kaggle/input/ob-ai-model-2-dataset/Data/processed_smc_data_test.csv')
    
    print(f"  Raw train: {len(train_df):,} samples")
    print(f"  Raw val:   {len(val_df):,} samples")
    print(f"  Raw test:  {len(test_df):,} samples")
    
    # Only remove NaN labels (models need labels to train)
    train_df = train_df[train_df['TBM_Label'].notna()].copy()
    val_df = val_df[val_df['TBM_Label'].notna()].copy()
    test_df = test_df[test_df['TBM_Label'].notna()].copy()
    
    print(f"\n  After removing NaN labels:")
    print(f"  Train: {len(train_df):,} samples (ALL SYMBOLS)")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Train Neural Network
    if TORCH_AVAILABLE:
        print("\n" + "="*80)
        print("TRAINING NEURAL NETWORK (ALL SYMBOLS COMBINED)")
        print("="*80)
        
        model = NeuralNetworkSMCModel(symbol='ALL_SYMBOLS')
        
        # Prepare features
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
        
        print(f"\n  Features: {X_train.shape[1]}")
        print(f"  Training samples: {len(X_train):,}")
        
        # Train
        history = model.train(
            X_train, y_train, X_val, y_val,
            hidden_dims=[512, 256, 128, 64],
            dropout=0.4,
            learning_rate=0.01,
            batch_size=64,  # Larger batch for more data
            epochs=200,
            patience=30,
            weight_decay=0.01
        )
        
        # Evaluate
        print("\n📊 Evaluating Neural Network...")
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')
        
        # Save (Kaggle working directory)
        import torch
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'scaler': model.scaler,
            'label_map_reverse': model.label_map_reverse
        }, '/kaggle/working/ALL_SYMBOLS_NeuralNetwork.pth')
        print("\n💾 Model saved to /kaggle/working/ALL_SYMBOLS_NeuralNetwork.pth")
        
        print("\n" + "="*80)
        print("NEURAL NETWORK RESULTS (ALL SYMBOLS)")
        print("="*80)
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Test Accuracy:       {test_metrics['accuracy']:.3f}")
        print(f"  Win Rate:            {test_metrics.get('win_rate', 0):.1f}%")
    
    # Train LSTM
    if TORCH_AVAILABLE:
        print("\n" + "="*80)
        print("TRAINING LSTM (ALL SYMBOLS COMBINED)")
        print("="*80)
        
        model = LSTMSMCModel(symbol='ALL_SYMBOLS', lookback=20)
        
        # Prepare features
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
        
        print(f"\n  Features: {X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]}")
        print(f"  Training sequences: {len(X_train):,}")
        
        # Train
        history = model.train(
            X_train, y_train, X_val, y_val,
            hidden_dim=256,
            num_layers=3,
            dropout=0.4,
            learning_rate=0.01,
            batch_size=32,  # Moderate batch for sequences
            epochs=200,
            patience=30,
            weight_decay=0.01,
            bidirectional=True
        )
        
        # Evaluate
        print("\n📊 Evaluating LSTM...")
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')
        
        # Save (Kaggle working directory)
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'scaler': model.scaler,
            'label_map_reverse': model.label_map_reverse
        }, '/kaggle/working/ALL_SYMBOLS_LSTM.pth')
        print("\n💾 Model saved to /kaggle/working/ALL_SYMBOLS_LSTM.pth")
        
        print("\n" + "="*80)
        print("LSTM RESULTS (ALL SYMBOLS)")
        print("="*80)
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Test Accuracy:       {test_metrics['accuracy']:.3f}")
        print(f"  Win Rate:            {test_metrics.get('win_rate', 0):.1f}%")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nModels trained on ALL symbols combined for maximum data utilization.")
    print("These models can now be used for any symbol (transfer learning).")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot train deep learning models.")
        sys.exit(1)
    
    train_combined_models()
