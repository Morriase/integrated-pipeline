"""
Quick test to verify RandomForest KeyError fix
"""

import numpy as np
from models.random_forest_model import RandomForestSMCModel

def test_randomforest_cv():
    """Test RandomForest with cross-validation"""
    
    print("=" * 80)
    print("Testing RandomForest Cross-Validation Fix")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    X_val = np.random.randn(50, n_features)
    y_val = np.random.randint(0, 2, 50)
    
    # Initialize model
    model = RandomForestSMCModel(symbol='TEST')
    
    try:
        # Train with cross-validation (this was failing before)
        print("\n🌲 Training RandomForest with cross-validation...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            n_estimators=50,  # Small for quick test
            max_depth=5,
            use_cross_validation=True
        )
        
        print("\n✅ Training completed successfully!")
        print(f"\nTraining History Keys: {list(history.keys())}")
        
        # Check that CV metrics are present
        cv_keys = ['cv_mean_accuracy', 'cv_std_accuracy', 'cv_is_stable', 'cv_fold_accuracies']
        missing_keys = [k for k in cv_keys if k not in history]
        
        if missing_keys:
            print(f"\n❌ Missing CV keys: {missing_keys}")
            return False
        else:
            print(f"\n✅ All CV metrics present:")
            print(f"   - Mean Accuracy: {history['cv_mean_accuracy']:.3f}")
            print(f"   - Std Accuracy: {history['cv_std_accuracy']:.3f}")
            print(f"   - Is Stable: {history['cv_is_stable']}")
            print(f"   - Fold Accuracies: {[f'{x:.3f}' for x in history['cv_fold_accuracies']]}")
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        predictions = model.predict(X_val)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Unique predictions: {np.unique(predictions)}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - RandomForest fix verified!")
        print("=" * 80)
        return True
        
    except KeyError as e:
        print(f"\n❌ KeyError still present: {e}")
        print("   Fix did not resolve the issue")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_randomforest_cv()
    exit(0 if success else 1)
