"""
Test Ensemble Predictor on validation/test data
"""

import numpy as np
import pandas as pd
from ensemble_predictor import EnsemblePredictor
from sklearn.metrics import classification_report, confusion_matrix
import json


def load_test_data():
    """Load and prepare test data"""
    print("📂 Loading test data...")

    # Hardcoded for Kaggle
    data_path = '/kaggle/working/Data-output/processed_smc_data.csv'

    # Load unified dataset
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} samples")
    
    # Load feature names from model metadata to ensure exact match
    metadata_path = '/kaggle/working/Model-output/UNIFIED_RandomForest_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_cols']
    print(f"  Using {len(feature_cols)} features from training")
    
    # Check if all features exist in data
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  ⚠️ Missing features: {missing_features[:5]}...")
        raise ValueError(f"Data is missing {len(missing_features)} features used in training")
    
    X = df[feature_cols].values
    y = df['TBM_Label'].values
    
    # Remove NaN labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"  After removing NaN labels: {len(X):,} samples")
    
    # Filter to only Loss (-1) and Win (1) - remove Timeout (0)
    # Models were trained on binary classification
    binary_mask = (y == -1) | (y == 1)
    X = X[binary_mask]
    y = y[binary_mask]
    print(f"  After filtering to Loss/Win only: {len(X):,} samples")
    print(f"  Label distribution: Loss={np.sum(y==-1):,}, Win={np.sum(y==1):,}")
    
    # Use last 30% as test set (same as training)
    test_size = int(len(X) * 0.3)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")
    
    return X_test, y_test, feature_cols


def evaluate_ensemble():
    """Evaluate ensemble with different strategies"""
    print("=" * 80)
    print("ENSEMBLE EVALUATION")
    print("=" * 80)
    
    # Load ensemble
    ensemble = EnsemblePredictor()
    ensemble.load_models()
    
    # Load test data
    X_test, y_test, feature_names = load_test_data()
    
    print("\n" + "=" * 80)
    print("TESTING VOTING STRATEGIES")
    print("=" * 80)
    
    strategies = ['weighted', 'soft', 'hard']
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Strategy: {strategy.upper()}")
        print("-" * 40)
        
        # Evaluate
        metrics = ensemble.evaluate(X_test, y_test, strategy=strategy)
        results[strategy] = metrics
        
        print(f"  Ensemble Accuracy: {metrics['ensemble_accuracy']:.4f}")
        print(f"  Ensemble Precision: {metrics['ensemble_precision']:.4f}")
        print(f"  Ensemble Recall: {metrics['ensemble_recall']:.4f}")
        print(f"  Ensemble F1: {metrics['ensemble_f1']:.4f}")
        
        print(f"\n  Individual Model Accuracies:")
        for model, acc in metrics['individual_accuracies'].items():
            print(f"    {model}: {acc:.4f}")
        
        improvement = metrics['improvement']
        if improvement > 0:
            print(f"\n  ✅ Improvement over best model: +{improvement:.4f}")
        else:
            print(f"\n  ⚠️ No improvement: {improvement:.4f}")
        
        print(f"\n  Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"    {cm}")
    
    # Model agreement analysis
    print("\n" + "=" * 80)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 80)
    
    agreement_rate, unanimous = ensemble.get_model_agreement(X_test)
    
    print(f"\n  Average agreement: {agreement_rate.mean():.2%}")
    print(f"  Unanimous predictions: {unanimous.sum():,} / {len(X_test):,} ({unanimous.mean():.2%})")
    
    # Accuracy on unanimous vs non-unanimous
    y_pred_weighted = ensemble.predict(X_test, strategy='weighted')
    
    unanimous_acc = (y_pred_weighted[unanimous] == y_test[unanimous]).mean()
    non_unanimous_acc = (y_pred_weighted[~unanimous] == y_test[~unanimous]).mean()
    
    print(f"\n  Accuracy on unanimous predictions: {unanimous_acc:.4f}")
    print(f"  Accuracy on non-unanimous predictions: {non_unanimous_acc:.4f}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output = {
        'strategies': results,
        'agreement_analysis': {
            'average_agreement': float(agreement_rate.mean()),
            'unanimous_count': int(unanimous.sum()),
            'unanimous_percentage': float(unanimous.mean()),
            'unanimous_accuracy': float(unanimous_acc),
            'non_unanimous_accuracy': float(non_unanimous_acc)
        },
        'recommendation': 'weighted'  # Best strategy
    }

    # Hardcoded for Kaggle
    save_path = '/kaggle/working/Model-output/ensemble_evaluation.json'

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  ✅ Saved: {save_path}")
    
    # Print recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    best_strategy = max(results.keys(), key=lambda k: results[k]['ensemble_accuracy'])
    best_acc = results[best_strategy]['ensemble_accuracy']
    
    print(f"\n  🎯 Best Strategy: {best_strategy.upper()}")
    print(f"  📈 Accuracy: {best_acc:.4f}")
    print(f"  💡 Use: ensemble.predict(X, strategy='{best_strategy}')")
    
    return ensemble, results


if __name__ == "__main__":
    ensemble, results = evaluate_ensemble()
