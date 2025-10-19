"""
Test script for anti-overfitting configuration management

This script tests:
1. Configuration loading and saving
2. Configuration validation
3. Invalid configuration detection
4. Dictionary conversion
5. Integration with existing code patterns
"""

import os
import json
import tempfile
from models.config import (
    AntiOverfittingConfig,
    RandomForestConfig,
    NeuralNetworkConfig,
    FeatureSelectionConfig,
    AugmentationConfig,
    CrossValidationConfig,
    MonitorConfig,
    load_config,
    create_default_config_file,
    RF_CONFIG,
    NN_CONFIG,
    FEATURE_SELECTION_CONFIG,
    AUGMENTATION_CONFIG,
    CV_CONFIG,
    MONITOR_CONFIG
)


def test_default_config():
    """Test default configuration creation and validation"""
    print("Test 1: Default Configuration")
    print("-" * 50)
    
    config = AntiOverfittingConfig()
    
    # Validate
    assert config.validate(), "Default configuration should be valid"
    print("✓ Default configuration is valid")
    
    # Check some values
    assert config.rf_config.max_depth == 15, "RF max_depth should be 15"
    assert config.nn_config.dropout == 0.5, "NN dropout should be 0.5"
    assert config.feature_selection_config.min_features == 30, "Min features should be 30"
    print("✓ Default values are correct")
    
    print()


def test_config_save_load():
    """Test saving and loading configuration"""
    print("Test 2: Save and Load Configuration")
    print("-" * 50)
    
    # Create config
    config = AntiOverfittingConfig()
    config.rf_config.max_depth = 12
    config.nn_config.dropout = 0.6
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save(temp_path)
        print(f"✓ Configuration saved to {temp_path}")
        
        # Load from file
        loaded_config = AntiOverfittingConfig.load(temp_path)
        print(f"✓ Configuration loaded from {temp_path}")
        
        # Verify values
        assert loaded_config.rf_config.max_depth == 12, "Loaded max_depth should be 12"
        assert loaded_config.nn_config.dropout == 0.6, "Loaded dropout should be 0.6"
        print("✓ Loaded values match saved values")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print()


def test_config_validation():
    """Test configuration validation"""
    print("Test 3: Configuration Validation")
    print("-" * 50)
    
    # Test valid configuration
    config = AntiOverfittingConfig()
    assert config.validate(), "Valid config should pass validation"
    print("✓ Valid configuration passes validation")
    
    # Test invalid RF config
    config.rf_config.max_depth = 100  # Out of range
    assert not config.validate(), "Invalid config should fail validation"
    print("✓ Invalid RF configuration detected")
    
    # Reset and test invalid NN config
    config = AntiOverfittingConfig()
    config.nn_config.dropout = 1.5  # Out of range
    assert not config.validate(), "Invalid config should fail validation"
    print("✓ Invalid NN configuration detected")
    
    # Reset and test invalid feature selection config
    config = AntiOverfittingConfig()
    config.feature_selection_config.correlation_threshold = 1.5  # Out of range
    assert not config.validate(), "Invalid config should fail validation"
    print("✓ Invalid feature selection configuration detected")
    
    print()


def test_dict_conversion():
    """Test dictionary conversion"""
    print("Test 4: Dictionary Conversion")
    print("-" * 50)
    
    # Create config
    config = AntiOverfittingConfig()
    
    # Convert to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict), "Should convert to dictionary"
    assert 'rf_config' in config_dict, "Should contain rf_config"
    assert 'nn_config' in config_dict, "Should contain nn_config"
    print("✓ Configuration converted to dictionary")
    
    # Create from dict
    new_config = AntiOverfittingConfig.from_dict(config_dict)
    assert new_config.rf_config.max_depth == config.rf_config.max_depth, "Values should match"
    assert new_config.nn_config.dropout == config.nn_config.dropout, "Values should match"
    print("✓ Configuration created from dictionary")
    
    print()


def test_legacy_exports():
    """Test legacy dictionary exports for backward compatibility"""
    print("Test 5: Legacy Dictionary Exports")
    print("-" * 50)
    
    # Check that legacy exports exist
    assert isinstance(RF_CONFIG, dict), "RF_CONFIG should be a dictionary"
    assert isinstance(NN_CONFIG, dict), "NN_CONFIG should be a dictionary"
    assert isinstance(FEATURE_SELECTION_CONFIG, dict), "FEATURE_SELECTION_CONFIG should be a dictionary"
    assert isinstance(AUGMENTATION_CONFIG, dict), "AUGMENTATION_CONFIG should be a dictionary"
    assert isinstance(CV_CONFIG, dict), "CV_CONFIG should be a dictionary"
    assert isinstance(MONITOR_CONFIG, dict), "MONITOR_CONFIG should be a dictionary"
    print("✓ All legacy dictionary exports exist")
    
    # Check some values
    assert RF_CONFIG['max_depth'] == 15, "RF max_depth should be 15"
    assert NN_CONFIG['dropout'] == 0.5, "NN dropout should be 0.5"
    assert FEATURE_SELECTION_CONFIG['min_features'] == 30, "Min features should be 30"
    print("✓ Legacy dictionary values are correct")
    
    print()


def test_load_config_function():
    """Test load_config convenience function"""
    print("Test 6: load_config Function")
    print("-" * 50)
    
    # Load default config
    config = load_config()
    assert config.validate(), "Default config should be valid"
    print("✓ load_config() returns valid default configuration")
    
    # Load from file
    config_path = 'models/anti_overfitting_config.json'
    if os.path.exists(config_path):
        config = load_config(config_path)
        assert config.validate(), "Loaded config should be valid"
        print(f"✓ load_config('{config_path}') returns valid configuration")
    else:
        print(f"⚠ Configuration file not found: {config_path}")
    
    print()


def test_individual_config_validation():
    """Test validation of individual configuration sections"""
    print("Test 7: Individual Configuration Validation")
    print("-" * 50)
    
    # Test RF config
    rf_config = RandomForestConfig()
    errors = rf_config.validate()
    assert len(errors) == 0, "Default RF config should have no errors"
    print("✓ RandomForestConfig validation works")
    
    # Test NN config
    nn_config = NeuralNetworkConfig()
    errors = nn_config.validate()
    assert len(errors) == 0, "Default NN config should have no errors"
    print("✓ NeuralNetworkConfig validation works")
    
    # Test feature selection config
    fs_config = FeatureSelectionConfig()
    errors = fs_config.validate()
    assert len(errors) == 0, "Default FS config should have no errors"
    print("✓ FeatureSelectionConfig validation works")
    
    # Test augmentation config
    aug_config = AugmentationConfig()
    errors = aug_config.validate()
    assert len(errors) == 0, "Default augmentation config should have no errors"
    print("✓ AugmentationConfig validation works")
    
    # Test CV config
    cv_config = CrossValidationConfig()
    errors = cv_config.validate()
    assert len(errors) == 0, "Default CV config should have no errors"
    print("✓ CrossValidationConfig validation works")
    
    # Test monitor config
    mon_config = MonitorConfig()
    errors = mon_config.validate()
    assert len(errors) == 0, "Default monitor config should have no errors"
    print("✓ MonitorConfig validation works")
    
    print()


def test_config_modification():
    """Test modifying configuration values"""
    print("Test 8: Configuration Modification")
    print("-" * 50)
    
    config = AntiOverfittingConfig()
    
    # Modify RF config
    config.rf_config.max_depth = 20
    config.rf_config.min_samples_split = 15
    assert config.rf_config.max_depth == 20, "max_depth should be updated"
    assert config.rf_config.min_samples_split == 15, "min_samples_split should be updated"
    print("✓ RF configuration modified successfully")
    
    # Modify NN config
    config.nn_config.hidden_dims = [512, 256, 128]
    config.nn_config.dropout = 0.4
    assert config.nn_config.hidden_dims == [512, 256, 128], "hidden_dims should be updated"
    assert config.nn_config.dropout == 0.4, "dropout should be updated"
    print("✓ NN configuration modified successfully")
    
    # Validate modified config
    assert config.validate(), "Modified config should still be valid"
    print("✓ Modified configuration is valid")
    
    print()


def test_error_messages():
    """Test that validation provides helpful error messages"""
    print("Test 9: Validation Error Messages")
    print("-" * 50)
    
    config = AntiOverfittingConfig()
    
    # Create multiple errors
    config.rf_config.max_depth = 100  # Out of range
    config.nn_config.dropout = 1.5  # Out of range
    config.feature_selection_config.correlation_threshold = 2.0  # Out of range
    
    # Validate and check errors
    is_valid = config.validate()
    assert not is_valid, "Config with errors should not be valid"
    print("✓ Invalid configuration detected")
    print("✓ Error messages logged (check above)")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("ANTI-OVERFITTING CONFIGURATION TESTS")
    print("=" * 50)
    print()
    
    try:
        test_default_config()
        test_config_save_load()
        test_config_validation()
        test_dict_conversion()
        test_legacy_exports()
        test_load_config_function()
        test_individual_config_validation()
        test_config_modification()
        test_error_messages()
        
        print("=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        return True
        
    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 50)
        return False
    except Exception as e:
        print()
        print("=" * 50)
        print(f"ERROR ✗: {e}")
        print("=" * 50)
        return False


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = run_all_tests()
    exit(0 if success else 1)
