"""
Unit tests for TrainingMonitor class
Tests all early warning system functionality
"""

import numpy as np
from datetime import datetime, timedelta
from models.base_model import TrainingMonitor


def test_overfitting_check():
    """Test overfitting detection (Requirement 9.1)"""
    print("\n🧪 Testing overfitting check...")
    
    monitor = TrainingMonitor()
    
    # Test case 1: Severe overfitting (should trigger)
    result = monitor.check_overfitting(train_acc=0.96, val_acc=0.55, epoch=10)
    assert result == True, "Should detect severe overfitting"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    assert "overfitting" in monitor.warnings[0].lower()
    print("  ✅ Severe overfitting detected correctly")
    
    # Test case 2: No overfitting (should not trigger)
    monitor.reset()
    result = monitor.check_overfitting(train_acc=0.85, val_acc=0.80, epoch=10)
    assert result == False, "Should not detect overfitting"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Normal training not flagged")
    
    # Test case 3: High train but acceptable val (should not trigger)
    monitor.reset()
    result = monitor.check_overfitting(train_acc=0.96, val_acc=0.65, epoch=10)
    assert result == False, "Should not trigger with val > 60%"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ High train with acceptable val not flagged")
    
    print("✅ Overfitting check tests passed!")


def test_divergence_check():
    """Test validation loss divergence detection (Requirement 9.2)"""
    print("\n🧪 Testing divergence check...")
    
    monitor = TrainingMonitor()
    
    # Test case 1: Diverging losses (should trigger)
    diverging_losses = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    result = monitor.check_divergence(diverging_losses, patience=10)
    assert result == True, "Should detect divergence"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    assert "divergence" in monitor.warnings[0].lower()
    print("  ✅ Diverging losses detected correctly")
    
    # Test case 2: Improving losses (should not trigger)
    monitor.reset()
    improving_losses = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    result = monitor.check_divergence(improving_losses, patience=10)
    assert result == False, "Should not detect divergence"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Improving losses not flagged")
    
    # Test case 3: Mixed losses (should not trigger)
    monitor.reset()
    mixed_losses = [1.0, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.5]
    result = monitor.check_divergence(mixed_losses, patience=10)
    assert result == False, "Should not trigger with mixed losses"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Mixed losses not flagged")
    
    # Test case 4: Not enough data (should not trigger)
    monitor.reset()
    short_losses = [1.0, 1.1, 1.2]
    result = monitor.check_divergence(short_losses, patience=10)
    assert result == False, "Should not trigger with insufficient data"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Insufficient data handled correctly")
    
    print("✅ Divergence check tests passed!")


def test_nan_loss_check():
    """Test NaN loss detection (Requirement 9.3)"""
    print("\n🧪 Testing NaN loss check...")
    
    monitor = TrainingMonitor()
    
    # Test case 1: NaN loss (should trigger and set stop flag)
    result = monitor.check_nan_loss(np.nan, epoch=5)
    assert result == True, "Should detect NaN"
    assert monitor.should_stop == True, "Should set stop flag"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    assert "nan" in monitor.warnings[0].lower()
    print("  ✅ NaN loss detected correctly")
    
    # Test case 2: Inf loss (should trigger and set stop flag)
    monitor.reset()
    result = monitor.check_nan_loss(np.inf, epoch=5)
    assert result == True, "Should detect Inf"
    assert monitor.should_stop == True, "Should set stop flag"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    print("  ✅ Inf loss detected correctly")
    
    # Test case 3: Normal loss (should not trigger)
    monitor.reset()
    result = monitor.check_nan_loss(0.5, epoch=5)
    assert result == False, "Should not detect issue"
    assert monitor.should_stop == False, "Should not set stop flag"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Normal loss not flagged")
    
    print("✅ NaN loss check tests passed!")


def test_exploding_gradients_check():
    """Test exploding gradients detection (Requirement 9.4)"""
    print("\n🧪 Testing exploding gradients check...")
    
    monitor = TrainingMonitor()
    
    # Test case 1: Exploding gradient (should trigger)
    result = monitor.check_exploding_gradients(grad_norm=15.0, threshold=10.0, epoch=5)
    assert result == True, "Should detect exploding gradient"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    assert "exploding" in monitor.warnings[0].lower()
    print("  ✅ Exploding gradient detected correctly")
    
    # Test case 2: Normal gradient (should not trigger)
    monitor.reset()
    result = monitor.check_exploding_gradients(grad_norm=5.0, threshold=10.0, epoch=5)
    assert result == False, "Should not detect issue"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Normal gradient not flagged")
    
    # Test case 3: Edge case at threshold (should not trigger)
    monitor.reset()
    result = monitor.check_exploding_gradients(grad_norm=10.0, threshold=10.0, epoch=5)
    assert result == False, "Should not trigger at threshold"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Threshold edge case handled correctly")
    
    print("✅ Exploding gradients check tests passed!")


def test_timeout_check():
    """Test training timeout detection (Requirement 9.5)"""
    print("\n🧪 Testing timeout check...")
    
    monitor = TrainingMonitor()
    
    # Test case 1: Timeout exceeded (should trigger)
    start_time = datetime.now() - timedelta(minutes=11)
    result = monitor.check_timeout(start_time, max_minutes=10)
    assert result == True, "Should detect timeout"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    assert "timeout" in monitor.warnings[0].lower()
    print("  ✅ Timeout detected correctly")
    
    # Test case 2: Within time limit (should not trigger)
    monitor.reset()
    start_time = datetime.now() - timedelta(minutes=5)
    result = monitor.check_timeout(start_time, max_minutes=10)
    assert result == False, "Should not detect timeout"
    assert len(monitor.warnings) == 0, "Should have no warnings"
    print("  ✅ Normal training time not flagged")
    
    # Test case 3: Edge case just over limit (should trigger)
    monitor.reset()
    start_time = datetime.now() - timedelta(minutes=10, seconds=1)
    result = monitor.check_timeout(start_time, max_minutes=10)
    assert result == True, "Should trigger just over limit"
    assert len(monitor.warnings) == 1, "Should have 1 warning"
    print("  ✅ Time limit edge case handled correctly")
    
    print("✅ Timeout check tests passed!")


def test_warning_management():
    """Test warning storage and retrieval"""
    print("\n🧪 Testing warning management...")
    
    monitor = TrainingMonitor()
    
    # Add multiple warnings
    monitor.check_overfitting(0.96, 0.55, epoch=5)
    monitor.check_exploding_gradients(15.0, epoch=10)
    
    # Check warnings
    warnings = monitor.get_warnings()
    assert len(warnings) == 2, "Should have 2 warnings"
    print("  ✅ Multiple warnings stored correctly")
    
    # Check critical warnings flag
    assert monitor.has_critical_warnings() == False, "Should not have critical warnings yet"
    
    # Add critical warning
    monitor.check_nan_loss(np.nan, epoch=15)
    assert monitor.has_critical_warnings() == True, "Should have critical warnings"
    assert len(monitor.get_warnings()) == 3, "Should have 3 warnings total"
    print("  ✅ Critical warnings flagged correctly")
    
    # Test reset
    monitor.reset()
    assert len(monitor.get_warnings()) == 0, "Should have no warnings after reset"
    assert monitor.has_critical_warnings() == False, "Should not have critical warnings after reset"
    print("  ✅ Reset works correctly")
    
    print("✅ Warning management tests passed!")


def test_integration_scenario():
    """Test realistic training scenario with multiple checks"""
    print("\n🧪 Testing integration scenario...")
    
    monitor = TrainingMonitor()
    start_time = datetime.now()
    
    # Simulate training epochs
    val_losses = []
    
    for epoch in range(1, 16):
        # Simulate training metrics
        train_acc = 0.70 + (epoch * 0.02)  # Gradually increasing
        val_acc = 0.65 + (epoch * 0.01)    # Slower increase
        loss = 1.0 - (epoch * 0.05)        # Decreasing
        grad_norm = 2.0 + np.random.randn() * 0.5  # Normal gradients
        
        val_losses.append(loss)
        
        # Run all checks
        monitor.check_overfitting(train_acc, val_acc, epoch)
        monitor.check_nan_loss(loss, epoch)
        monitor.check_exploding_gradients(grad_norm, epoch=epoch)
        monitor.check_divergence(val_losses)
        monitor.check_timeout(start_time, max_minutes=10)
        
        # Check if should stop
        if monitor.should_stop:
            print(f"  Training stopped at epoch {epoch}")
            break
    
    # In this scenario, no warnings should be triggered
    warnings = monitor.get_warnings()
    print(f"  Warnings generated: {len(warnings)}")
    for warning in warnings:
        print(f"    - {warning}")
    
    assert monitor.should_stop == False, "Should not stop in normal scenario"
    print("  ✅ Normal training scenario handled correctly")
    
    # Test problematic scenario
    monitor.reset()
    monitor.check_overfitting(0.98, 0.55, epoch=10)
    diverging = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    monitor.check_divergence(diverging)
    
    warnings = monitor.get_warnings()
    assert len(warnings) == 2, "Should have 2 warnings in problematic scenario"
    print(f"  ✅ Problematic scenario detected {len(warnings)} issues")
    
    print("✅ Integration scenario tests passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("TrainingMonitor Unit Tests")
    print("=" * 60)
    
    try:
        test_overfitting_check()
        test_divergence_check()
        test_nan_loss_check()
        test_exploding_gradients_check()
        test_timeout_check()
        test_warning_management()
        test_integration_scenario()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTrainingMonitor is ready for integration into model training loops.")
        print("\nNext steps:")
        print("  1. Integrate into Neural Network training (Task 10.1)")
        print("  2. Integrate into LSTM training (Task 10.2)")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
