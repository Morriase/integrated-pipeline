"""
Test script to verify RandomForestSMCModel anti-overfitting enhancements
"""

import numpy as np
from models.random_forest_model import RandomForestSMCModel

def test_rf_anti_overfitting():
    """Test the enhanced RandomForestSMCModel with anti-overfitting features"""
    
    print("=" * 80)
    print("Testing RandomForestSMCModel Anti-Overfitting Enhancements")
    print("=" * 80)
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 250  # Small dataset to trigger augmentation
    n_features = 50
    
    # Generate synthetic features
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([0, 1, 2], size=n_samples)
    
    X_val = np.random.randn(100, n_features)
    y_val = np.random.choice([0, 1, 2], size=100)
    
    print(f"\nTest Data:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Features: {n_features}")
    
    # Initialize model
    model = RandomForestSMCModel(symbol='TEST')
    model.feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    # Train with anti-overfitting constraints
    print("\n" + "=" * 80)
    print("Training with Anti-Overfitting Constraints")
    print("=" * 80)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        n_estimators=100,  # Reduced for faster testing
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_samples=0.8,
        use_cross_validation=True
    )
    
    # Verify training history contains expected keys
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)
    
    expected_keys = [
        'train_accuracy',
        'val_accuracy',
        'train_val_gap',
        'overfitting_detected',
        'cv_mean_accuracy',
        'cv_std_accuracy',
        'cv_is_stable',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'max_samples'
    ]
    
    print("\n✓ Checking training history keys:")
    for key in expected_keys:
        if key in history:
            print(f"  ✅ {key}: {history[key]}")
        else:
            print(f"  ❌ {key}: MISSING")
    
    # Verify anti-overfitting constraints
    print("\n✓ Verifying anti-overfitting constraints:")
    print(f"  max_depth = {history.get('max_depth')} (expected: 15)")
    print(f"  min_samples_split = {history.get('min_samples_split')} (expected: 20)")
    print(f"  min_samples_leaf = {history.get('min_samples_leaf')} (expected: 10)")
    print(f"  max_samples = {history.get('max_samples')} (expected: 0.8)")
    
    # Verify cross-validation was performed
    print("\n✓ Cross-validation results:")
    if 'cv_mean_accuracy' in history:
        print(f"  Mean CV Accuracy: {history['cv_mean_accuracy']:.3f} ± {history['cv_std_accuracy']:.3f}")
        print(f"  Model Stable: {history['cv_is_stable']}")
    else:
        print("  ❌ Cross-validation results not found")
    
    # Verify train-val gap calculation
    print("\n✓ Overfitting detection:")
    if 'train_val_gap' in history:
        print(f"  Train-Val Gap: {history['train_val_gap']:.3f}")
        print(f"  Overfitting Detected: {history['overfitting_detected']}")
    else:
        print("  ❌ Train-val gap not calculated")
    
    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    
    return history

if __name__ == "__main__":
    test_rf_anti_overfitting()
