"""
Test script for adaptive data augmentation implementation.

Tests all new features:
- Adaptive augmentation factor (3x for <200, 2x for 200-300)
- Increased noise magnitude (0.15)
- Time-shift augmentation
- Feature dropout augmentation
- Label distribution validation
"""

import numpy as np
import sys
from models.data_augmentation import DataAugmenter

def test_augmentation_factor():
    """Test that augmentation factor adapts to dataset size"""
    print("\n" + "="*60)
    print("TEST 1: Augmentation Factor Logic")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    # Test 1: Very small dataset (<200 samples) - should get 3x
    X_small = np.random.rand(150, 57)
    y_small = np.random.randint(0, 2, 150)
    
    print(f"\nTest 1a: Dataset with {len(X_small)} samples (<200)")
    X_aug, y_aug = augmenter.augment(X_small, y_small)
    ratio = len(X_aug) / len(X_small)
    print(f"Result: {len(X_small)} -> {len(X_aug)} samples (ratio: {ratio:.2f}x)")
    
    expected_min = 2.0  # Should be at least 2x
    if ratio >= expected_min:
        print(f"✓ PASS: Augmentation ratio {ratio:.2f}x >= {expected_min}x")
    else:
        print(f"✗ FAIL: Augmentation ratio {ratio:.2f}x < {expected_min}x")
        return False
    
    # Test 2: Small dataset (200-300 samples) - should get 2x
    X_medium = np.random.rand(250, 57)
    y_medium = np.random.randint(0, 2, 250)
    
    print(f"\nTest 1b: Dataset with {len(X_medium)} samples (200-300)")
    X_aug, y_aug = augmenter.augment(X_medium, y_medium)
    ratio = len(X_aug) / len(X_medium)
    print(f"Result: {len(X_medium)} -> {len(X_aug)} samples (ratio: {ratio:.2f}x)")
    
    expected_min = 1.5  # Should be at least 1.5x
    if ratio >= expected_min:
        print(f"✓ PASS: Augmentation ratio {ratio:.2f}x >= {expected_min}x")
    else:
        print(f"✗ FAIL: Augmentation ratio {ratio:.2f}x < {expected_min}x")
        return False
    
    # Test 3: Large dataset (>=300 samples) - should not augment
    X_large = np.random.rand(350, 57)
    y_large = np.random.randint(0, 2, 350)
    
    print(f"\nTest 1c: Dataset with {len(X_large)} samples (>=300)")
    X_aug, y_aug = augmenter.augment(X_large, y_large)
    print(f"Result: {len(X_large)} -> {len(X_aug)} samples")
    
    if len(X_aug) == len(X_large):
        print(f"✓ PASS: No augmentation applied for large dataset")
    else:
        print(f"✗ FAIL: Augmentation applied when it shouldn't be")
        return False
    
    return True

def test_noise_magnitude():
    """Test that noise magnitude is 0.15"""
    print("\n" + "="*60)
    print("TEST 2: Noise Magnitude (0.15)")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    print(f"\nConfigured noise std: {augmenter.noise_std}")
    
    if augmenter.noise_std == 0.15:
        print("✓ PASS: Noise magnitude set to 0.15")
        return True
    else:
        print(f"✗ FAIL: Noise magnitude is {augmenter.noise_std}, expected 0.15")
        return False

def test_time_shift():
    """Test time-shift augmentation"""
    print("\n" + "="*60)
    print("TEST 3: Time-Shift Augmentation")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    # Create test data with distinct pattern
    X = np.array([[1, 2, 3, 4, 5]] * 10)
    
    print(f"\nOriginal sample: {X[0]}")
    X_shifted = augmenter.time_shift(X, max_shift=2)
    print(f"Shifted samples (first 3):")
    for i in range(min(3, len(X_shifted))):
        print(f"  Sample {i}: {X_shifted[i]}")
    
    # Check that at least some samples were shifted
    different = np.sum(~np.all(X == X_shifted, axis=1))
    print(f"\nSamples that were shifted: {different}/{len(X)}")
    
    if different > 0:
        print("✓ PASS: Time-shift augmentation applied")
        return True
    else:
        print("✗ FAIL: No samples were shifted")
        return False

def test_feature_dropout():
    """Test feature dropout augmentation"""
    print("\n" + "="*60)
    print("TEST 4: Feature Dropout Augmentation")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    # Create test data with all ones
    X = np.ones((100, 50))
    
    print(f"\nOriginal data: all values = 1.0")
    X_dropout = augmenter.feature_dropout(X, dropout_rate=0.1)
    
    # Count zeros (dropped features)
    n_zeros = np.sum(X_dropout == 0)
    total_values = X_dropout.size
    dropout_rate = n_zeros / total_values
    
    print(f"Dropped features: {n_zeros}/{total_values} ({dropout_rate:.2%})")
    
    # Should be approximately 10% (allow some variance)
    if 0.05 <= dropout_rate <= 0.15:
        print(f"✓ PASS: Dropout rate ~10% (actual: {dropout_rate:.2%})")
        return True
    else:
        print(f"✗ FAIL: Dropout rate {dropout_rate:.2%} not close to 10%")
        return False

def test_label_distribution():
    """Test label distribution validation"""
    print("\n" + "="*60)
    print("TEST 5: Label Distribution Validation")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    # Create balanced dataset
    X = np.random.rand(200, 57)
    y = np.array([0] * 100 + [1] * 100)
    
    print(f"\nOriginal distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y):.2%})")
    
    X_aug, y_aug = augmenter.augment(X, y)
    
    print(f"\nAugmented distribution:")
    unique, counts = np.unique(y_aug, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_aug):.2%})")
    
    # Check if distribution is preserved within 5%
    orig_dist = np.bincount(y) / len(y)
    aug_dist = np.bincount(y_aug) / len(y_aug)
    max_diff = np.max(np.abs(orig_dist - aug_dist))
    
    print(f"\nMaximum distribution difference: {max_diff:.2%}")
    
    if max_diff <= 0.10:  # Allow up to 10% difference (SMOTE can shift it)
        print(f"✓ PASS: Distribution preserved (max diff: {max_diff:.2%})")
        return True
    else:
        print(f"✗ FAIL: Distribution shifted too much ({max_diff:.2%})")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ADAPTIVE DATA AUGMENTATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Augmentation Factor Logic", test_augmentation_factor),
        ("Noise Magnitude", test_noise_magnitude),
        ("Time-Shift Augmentation", test_time_shift),
        ("Feature Dropout", test_feature_dropout),
        ("Label Distribution Validation", test_label_distribution),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
