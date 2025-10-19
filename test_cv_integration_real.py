"""
Integration test for CV stability metrics with real model classes
Demonstrates the enhanced CV functionality with actual RandomForest and XGBoost models
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.random_forest_model import RandomForestSMCModel
from models.xgboost_model import XGBoostSMCModel


def test_rf_cv_stability():
    """Test CV stability with RandomForest model"""
    print("=" * 70)
    print("INTEGRATION TEST: RandomForest CV Stability")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Create somewhat predictable pattern
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    # Create model
    model = RandomForestSMCModel('TEST')
    model.feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run cross-validation
    cv_results = model.cross_validate(X, y, n_folds=5)
    
    # Display results
    print("\n" + "=" * 70)
    print("CV RESULTS SUMMARY")
    print("=" * 70)
    print(f"Mean Accuracy:     {cv_results['mean_accuracy']:.3f}")
    print(f"Std Accuracy:      {cv_results['std_accuracy']:.3f}")
    print(f"Min Accuracy:      {cv_results['min_accuracy']:.3f}")
    print(f"Max Accuracy:      {cv_results['max_accuracy']:.3f}")
    print(f"Range:             {cv_results['max_accuracy'] - cv_results['min_accuracy']:.3f}")
    print(f"\nStability Status:")
    print(f"  Is Unstable:     {cv_results['is_unstable']}")
    print(f"  Is Rejected:     {cv_results['is_rejected']}")
    print(f"  Poor Folds:      {len(cv_results['poor_folds'])}")
    
    if cv_results['poor_folds']:
        print(f"\nPoor-Performing Folds:")
        for fold in cv_results['poor_folds']:
            print(f"  Fold {fold['fold']}: {fold['accuracy']:.3f} (dev: {fold['deviation']:.3f})")
    
    # Verify all new metrics are present
    assert 'std_accuracy' in cv_results
    assert 'min_accuracy' in cv_results
    assert 'max_accuracy' in cv_results
    assert 'is_unstable' in cv_results
    assert 'is_rejected' in cv_results
    assert 'poor_folds' in cv_results
    
    print("\n✅ RandomForest CV integration test passed!")
    return cv_results


def test_xgboost_cv_stability():
    """Test CV stability with XGBoost model"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: XGBoost CV Stability")
    print("=" * 70)
    
    # Create synthetic data with clear pattern
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Create clear pattern for better stability
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
    
    # Create model
    model = XGBoostSMCModel('TEST')
    model.feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run cross-validation
    cv_results = model.cross_validate(X, y, n_folds=5)
    
    # Display results
    print("\n" + "=" * 70)
    print("CV RESULTS SUMMARY")
    print("=" * 70)
    print(f"Mean Accuracy:     {cv_results['mean_accuracy']:.3f}")
    print(f"Std Accuracy:      {cv_results['std_accuracy']:.3f}")
    print(f"Min Accuracy:      {cv_results['min_accuracy']:.3f}")
    print(f"Max Accuracy:      {cv_results['max_accuracy']:.3f}")
    print(f"Range:             {cv_results['max_accuracy'] - cv_results['min_accuracy']:.3f}")
    print(f"\nStability Status:")
    print(f"  Is Unstable:     {cv_results['is_unstable']}")
    print(f"  Is Rejected:     {cv_results['is_rejected']}")
    print(f"  Poor Folds:      {len(cv_results['poor_folds'])}")
    
    if cv_results['poor_folds']:
        print(f"\nPoor-Performing Folds:")
        for fold in cv_results['poor_folds']:
            print(f"  Fold {fold['fold']}: {fold['accuracy']:.3f} (dev: {fold['deviation']:.3f})")
    
    # Verify all new metrics are present
    assert 'std_accuracy' in cv_results
    assert 'min_accuracy' in cv_results
    assert 'max_accuracy' in cv_results
    assert 'is_unstable' in cv_results
    assert 'is_rejected' in cv_results
    assert 'poor_folds' in cv_results
    
    print("\n✅ XGBoost CV integration test passed!")
    return cv_results


def compare_model_stability():
    """Compare stability between different models"""
    print("\n" + "=" * 70)
    print("MODEL STABILITY COMPARISON")
    print("=" * 70)
    
    # Create same dataset for both models
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Test RandomForest
    rf_model = RandomForestSMCModel('TEST')
    rf_model.feature_cols = [f'feature_{i}' for i in range(n_features)]
    rf_cv = rf_model.cross_validate(X, y, n_folds=5)
    
    # Test XGBoost
    xgb_model = XGBoostSMCModel('TEST')
    xgb_model.feature_cols = [f'feature_{i}' for i in range(n_features)]
    xgb_cv = xgb_model.cross_validate(X, y, n_folds=5)
    
    # Compare
    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'RandomForest':<20} {'XGBoost':<20}")
    print("-" * 70)
    print(f"{'Mean Accuracy':<25} {rf_cv['mean_accuracy']:<20.3f} {xgb_cv['mean_accuracy']:<20.3f}")
    print(f"{'Std Accuracy':<25} {rf_cv['std_accuracy']:<20.3f} {xgb_cv['std_accuracy']:<20.3f}")
    print(f"{'Min Accuracy':<25} {rf_cv['min_accuracy']:<20.3f} {xgb_cv['min_accuracy']:<20.3f}")
    print(f"{'Max Accuracy':<25} {rf_cv['max_accuracy']:<20.3f} {xgb_cv['max_accuracy']:<20.3f}")
    print(f"{'Is Unstable':<25} {str(rf_cv['is_unstable']):<20} {str(xgb_cv['is_unstable']):<20}")
    print(f"{'Is Rejected':<25} {str(rf_cv['is_rejected']):<20} {str(xgb_cv['is_rejected']):<20}")
    print(f"{'Poor Folds':<25} {len(rf_cv['poor_folds']):<20} {len(xgb_cv['poor_folds']):<20}")
    print("-" * 70)
    
    # Determine more stable model
    if rf_cv['std_accuracy'] < xgb_cv['std_accuracy']:
        print(f"\n✅ RandomForest is more stable (std: {rf_cv['std_accuracy']:.3f} vs {xgb_cv['std_accuracy']:.3f})")
    else:
        print(f"\n✅ XGBoost is more stable (std: {xgb_cv['std_accuracy']:.3f} vs {rf_cv['std_accuracy']:.3f})")


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("CV STABILITY INTEGRATION TEST SUITE")
    print("Testing with Real Model Classes")
    print("=" * 70)
    
    try:
        # Test individual models
        rf_results = test_rf_cv_stability()
        xgb_results = test_xgboost_cv_stability()
        
        # Compare models
        compare_model_stability()
        
        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
        print("\nKey Findings:")
        print("  ✓ CV stability metrics work with RandomForest")
        print("  ✓ CV stability metrics work with XGBoost")
        print("  ✓ All new metrics (std, min, max, flags) present")
        print("  ✓ Poor fold identification working")
        print("  ✓ Stability comparison between models possible")
        print("\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
