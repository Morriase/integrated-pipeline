"""
Test TrainingMonitor Integration in Neural Network and LSTM Models

This test verifies that the TrainingMonitor is properly integrated into
the training loops and captures warnings correctly.
"""

import numpy as np
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Skipping tests.")
    sys.exit(0)

from models.neural_network_model import NeuralNetworkSMCModel
from models.lstm_model import LSTMSMCModel


def test_neural_network_monitor_integration():
    """Test that TrainingMonitor is integrated into Neural Network training"""
    print("\n" + "="*70)
    print("TEST: Neural Network TrainingMonitor Integration")
    print("="*70)
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([-1, 0, 1], size=n_samples)
    
    X_val = np.random.randn(30, n_features)
    y_val = np.random.choice([-1, 0, 1], size=30)
    
    # Initialize model
    model = NeuralNetworkSMCModel(symbol='TEST')
    
    # Train with very few epochs to test quickly
    print("\n📊 Training Neural Network with monitor...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        hidden_dims=[32, 16],
        dropout=0.3,
        learning_rate=0.01,
        batch_size=16,
        epochs=5,  # Just a few epochs for testing
        patience=10
    )
    
    # Verify warnings are captured
    assert 'training_warnings' in history, "❌ Training warnings not stored in history"
    print(f"\n✅ Training warnings captured: {len(history['training_warnings'])} warnings")
    
    if history['training_warnings']:
        print("\n  Warnings:")
        for warning in history['training_warnings']:
            print(f"    - {warning}")
    else:
        print("  No warnings (expected for small test)")
    
    print("\n✅ Neural Network monitor integration test PASSED")
    return True


def test_lstm_monitor_integration():
    """Test that TrainingMonitor is integrated into LSTM training"""
    print("\n" + "="*70)
    print("TEST: LSTM TrainingMonitor Integration")
    print("="*70)
    
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([-1, 0, 1], size=n_samples)
    
    X_val = np.random.randn(30, n_features)
    y_val = np.random.choice([-1, 0, 1], size=30)
    
    # Initialize model with small lookback
    model = LSTMSMCModel(symbol='TEST', lookback=5)
    
    # Train with very few epochs to test quickly
    print("\n📊 Training LSTM with monitor...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        hidden_dim=16,
        num_layers=1,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=8,
        epochs=5,  # Just a few epochs for testing
        patience=10
    )
    
    # Verify warnings are captured
    assert 'training_warnings' in history, "❌ Training warnings not stored in history"
    print(f"\n✅ Training warnings captured: {len(history['training_warnings'])} warnings")
    
    if history['training_warnings']:
        print("\n  Warnings:")
        for warning in history['training_warnings']:
            print(f"    - {warning}")
    else:
        print("  No warnings (expected for small test)")
    
    print("\n✅ LSTM monitor integration test PASSED")
    return True


def test_monitor_detects_overfitting():
    """Test that monitor can detect overfitting scenario"""
    print("\n" + "="*70)
    print("TEST: Monitor Detects Overfitting")
    print("="*70)
    
    from models.base_model import TrainingMonitor
    
    monitor = TrainingMonitor()
    
    # Simulate severe overfitting scenario
    train_acc = 0.96
    val_acc = 0.55
    
    print(f"\n  Simulating: train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")
    detected = monitor.check_overfitting(train_acc, val_acc, epoch=10)
    
    assert detected, "❌ Failed to detect overfitting"
    assert len(monitor.get_warnings()) > 0, "❌ No warnings generated"
    
    print(f"  ✅ Overfitting detected: {monitor.get_warnings()[0]}")
    print("\n✅ Overfitting detection test PASSED")
    return True


def test_monitor_detects_divergence():
    """Test that monitor can detect validation loss divergence"""
    print("\n" + "="*70)
    print("TEST: Monitor Detects Divergence")
    print("="*70)
    
    from models.base_model import TrainingMonitor
    
    monitor = TrainingMonitor()
    
    # Simulate diverging validation loss
    val_losses = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    
    print(f"\n  Simulating diverging losses: {val_losses[-5:]}")
    detected = monitor.check_divergence(val_losses, patience=10)
    
    assert detected, "❌ Failed to detect divergence"
    assert len(monitor.get_warnings()) > 0, "❌ No warnings generated"
    
    print(f"  ✅ Divergence detected: {monitor.get_warnings()[0]}")
    print("\n✅ Divergence detection test PASSED")
    return True


def test_monitor_detects_nan():
    """Test that monitor can detect NaN loss"""
    print("\n" + "="*70)
    print("TEST: Monitor Detects NaN Loss")
    print("="*70)
    
    from models.base_model import TrainingMonitor
    
    monitor = TrainingMonitor()
    
    # Simulate NaN loss
    print(f"\n  Simulating NaN loss")
    detected = monitor.check_nan_loss(np.nan, epoch=5)
    
    assert detected, "❌ Failed to detect NaN"
    assert monitor.has_critical_warnings(), "❌ Critical warning flag not set"
    assert len(monitor.get_warnings()) > 0, "❌ No warnings generated"
    
    print(f"  ✅ NaN detected: {monitor.get_warnings()[0]}")
    print("\n✅ NaN detection test PASSED")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("TRAINING MONITOR INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Monitor Detects Overfitting", test_monitor_detects_overfitting),
        ("Monitor Detects Divergence", test_monitor_detects_divergence),
        ("Monitor Detects NaN", test_monitor_detects_nan),
        ("Neural Network Integration", test_neural_network_monitor_integration),
        ("LSTM Integration", test_lstm_monitor_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
