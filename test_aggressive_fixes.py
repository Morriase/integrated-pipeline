"""
Quick test of aggressive anti-overfitting fixes on EURUSD
Tests all 4 models with new default parameters
"""

import sys
import numpy as np
from models.random_forest_model import RandomForestSMCModel
from models.xgboost_model import XGBoostSMCModel
from models.neural_network_model import NeuralNetworkSMCModel
from models.lstm_model import LSTMSMCModel

def test_model(model_class, model_name, symbol='EURUSD', **kwargs):
    """Test a single model with aggressive regularization"""
    print(f"\n{'='*80}")
    print(f"Testing {model_name} with AGGRESSIVE regularization")
    print(f"{'='*80}")
    
    try:
        # Initialize model
        model = model_class(symbol=symbol)
        
        # Load data
        train_df, val_df, test_df = model.load_data(
            train_path='Data/processed_smc_data_train.csv',
            val_path='Data/processed_smc_data_val.csv',
            test_path='Data/processed_smc_data_test.csv',
            exclude_timeout=False
        )
        
        # Prepare features
        X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
        
        print(f"\n📊 Data loaded:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Train with default parameters (now aggressive)
        history = model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        # Evaluate
        print(f"\n📊 Evaluating on Validation set...")
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        
        print(f"\n📊 Evaluating on Test set...")
        test_metrics = model.evaluate(X_test, y_test, 'Test')
        
        # Check for success criteria
        train_acc = history.get('train_accuracy', history.get('train_acc', [0])[-1] if isinstance(history.get('train_acc'), list) else 0)
        val_acc = history.get('val_accuracy', history.get('val_acc', [0])[-1] if isinstance(history.get('val_acc'), list) else 0)
        
        if isinstance(train_acc, list):
            train_acc = train_acc[-1]
        if isinstance(val_acc, list):
            val_acc = val_acc[-1]
        
        train_val_gap = abs(train_acc - val_acc)
        
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY - {model_name}")
        print(f"{'='*80}")
        print(f"Train Accuracy:     {train_acc:.3f}")
        print(f"Val Accuracy:       {val_acc:.3f}")
        print(f"Train-Val Gap:      {train_val_gap:.3f} ({train_val_gap*100:.1f}%)")
        print(f"Test Accuracy:      {test_metrics['accuracy']:.3f}")
        
        # Success criteria
        success = True
        print(f"\n✅ SUCCESS CRITERIA:")
        
        if train_val_gap < 0.20:
            print(f"  ✅ Train-Val gap < 20%: {train_val_gap*100:.1f}%")
        else:
            print(f"  ❌ Train-Val gap >= 20%: {train_val_gap*100:.1f}%")
            success = False
        
        if train_acc < 0.95:
            print(f"  ✅ Train accuracy < 95%: {train_acc*100:.1f}%")
        else:
            print(f"  ⚠️ Train accuracy >= 95%: {train_acc*100:.1f}% (possible overfitting)")
        
        if abs(val_acc - test_metrics['accuracy']) < 0.10:
            print(f"  ✅ Val-Test gap < 10%: {abs(val_acc - test_metrics['accuracy'])*100:.1f}%")
        else:
            print(f"  ⚠️ Val-Test gap >= 10%: {abs(val_acc - test_metrics['accuracy'])*100:.1f}%")
        
        # Check for training warnings (LSTM)
        if 'training_warnings' in history and history['training_warnings']:
            warnings = history['training_warnings']
            print(f"\n⚠️ Training Warnings ({len(warnings)}):")
            for warning in warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more")
        else:
            print(f"\n✅ No training warnings")
        
        return success, {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_metrics['accuracy'],
            'train_val_gap': train_val_gap,
            'success': success
        }
        
    except Exception as e:
        print(f"\n❌ ERROR testing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Test all models with aggressive regularization"""
    print("\n" + "="*80)
    print("TESTING AGGRESSIVE ANTI-OVERFITTING FIXES")
    print("="*80)
    print("\nThis will test all 4 models on EURUSD with new default parameters")
    print("Expected: Lower train accuracy, similar/better val accuracy, smaller gaps")
    
    results = {}
    
    # Test RandomForest
    success, metrics = test_model(
        RandomForestSMCModel,
        'RandomForest',
        symbol='EURUSD'
    )
    results['RandomForest'] = metrics
    
    # Test XGBoost
    success, metrics = test_model(
        XGBoostSMCModel,
        'XGBoost',
        symbol='EURUSD'
    )
    results['XGBoost'] = metrics
    
    # Test Neural Network
    success, metrics = test_model(
        NeuralNetworkSMCModel,
        'NeuralNetwork',
        symbol='EURUSD',
        epochs=50,  # Reduced for quick test
        patience=10
    )
    results['NeuralNetwork'] = metrics
    
    # Test LSTM
    success, metrics = test_model(
        LSTMSMCModel,
        'LSTM',
        symbol='EURUSD',
        lookback=10,
        epochs=50,  # Reduced for quick test
        patience=10
    )
    results['LSTM'] = metrics
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"\n{model_name}:")
            print(f"  Train: {metrics['train_acc']:.3f}")
            print(f"  Val:   {metrics['val_acc']:.3f}")
            print(f"  Test:  {metrics['test_acc']:.3f}")
            print(f"  Gap:   {metrics['train_val_gap']*100:.1f}%")
            print(f"  Status: {'✅ PASS' if metrics['success'] else '❌ FAIL'}")
    
    # Count successes
    successes = sum(1 for m in results.values() if m and m['success'])
    total = len([m for m in results.values() if m is not None])
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {successes}/{total} models passed success criteria")
    print(f"{'='*80}")
    
    if successes == total:
        print("\n🎉 ALL MODELS PASSED! Aggressive fixes are working!")
        print("\nNext step: Run full training on all symbols")
        print("  python train_all_models.py --symbols all --models all")
    elif successes >= total * 0.75:
        print("\n✅ Most models passed. Fixes are improving overfitting.")
        print("\nConsider running full training to see results across all symbols")
    else:
        print("\n⚠️ Some models still struggling. May need further tuning.")
        print("\nReview individual model results above for specific issues")


if __name__ == "__main__":
    main()
