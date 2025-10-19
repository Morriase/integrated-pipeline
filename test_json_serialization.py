"""
Test JSON serialization fix for base model
Tests the _convert_to_json_serializable method with various numpy types
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.base_model import BaseSMCModel


class TestModel(BaseSMCModel):
    """Concrete implementation for testing"""
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True
        return {}
    
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X):
        return np.zeros((len(X), 2))


def test_numpy_type_conversion():
    """Test conversion of various numpy types"""
    print("\n🧪 Testing numpy type conversion...")
    
    model = TestModel('TestModel', 'EURUSD')
    
    # Test data with various numpy types
    test_data = {
        'int64': np.int64(42),
        'int32': np.int32(100),
        'float64': np.float64(3.14159),
        'float32': np.float32(2.71828),
        'bool_': np.bool_(True),
        'array_1d': np.array([1, 2, 3, 4, 5]),
        'array_2d': np.array([[1, 2], [3, 4]]),
        'nested_dict': {
            'accuracy': np.float64(0.85),
            'confusion_matrix': np.array([[10, 5], [3, 12]]),
            'is_overfitting': np.bool_(False)
        },
        'list_with_numpy': [np.int64(1), np.float64(2.5), np.bool_(True)],
        'regular_types': {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None
        }
    }
    
    # Convert to JSON-serializable format
    converted = model._convert_to_json_serializable(test_data)
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(converted, indent=2)
        print("✅ Successfully converted and serialized to JSON")
        print(f"\nSample output:\n{json_str[:500]}...")
        return True
    except (TypeError, ValueError) as e:
        print(f"❌ JSON serialization failed: {e}")
        return False


def test_metadata_with_numpy_types():
    """Test metadata structure similar to actual model metadata"""
    print("\n🧪 Testing realistic metadata structure...")
    
    model = TestModel('RandomForest', 'EURUSD')
    
    # Simulate training history with numpy types
    model.training_history = {
        'train_accuracy': np.float64(0.95),
        'val_accuracy': np.float64(0.72),
        'train_val_gap': np.float64(0.23),
        'confusion_matrix': np.array([[15, 5], [6, 20]]),
        'feature_importance': np.array([0.12, 0.08, 0.15, 0.05]),
        'cv_fold_accuracies': [np.float64(0.68), np.float64(0.72), np.float64(0.70)],
        'is_overfitting': np.bool_(True)
    }
    
    model.feature_importance = np.array([0.12, 0.08, 0.15, 0.05, 0.10])
    model.feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Prepare metadata like in save_model
    metadata = {
        'model_name': model.model_name,
        'symbol': model.symbol,
        'target_col': model.target_col,
        'feature_cols': model.feature_cols,
        'is_trained': model.is_trained,
        'training_history': model.training_history,
        'feature_importance': model.feature_importance.tolist() if model.feature_importance is not None else None
    }
    
    # Convert and serialize
    metadata_safe = model._convert_to_json_serializable(metadata)
    
    try:
        json_str = json.dumps(metadata_safe, indent=2)
        print("✅ Successfully converted metadata to JSON")
        print(f"\nMetadata keys: {list(metadata_safe.keys())}")
        print(f"Training history keys: {list(metadata_safe['training_history'].keys())}")
        return True
    except (TypeError, ValueError) as e:
        print(f"❌ Metadata serialization failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")
    
    model = TestModel('TestModel', 'EURUSD')
    
    # Test with NaN and inf
    edge_cases = {
        'nan_value': np.float64(np.nan),
        'inf_value': np.float64(np.inf),
        'neg_inf_value': np.float64(-np.inf),
        'empty_array': np.array([]),
        'nested_empty': {'data': np.array([])},
    }
    
    converted = model._convert_to_json_serializable(edge_cases)
    
    try:
        json_str = json.dumps(converted, indent=2)
        print("✅ Successfully handled edge cases")
        print(f"\nConverted values:")
        for key, value in converted.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
        return True
    except (TypeError, ValueError) as e:
        print(f"❌ Edge case handling failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("JSON Serialization Fix - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Numpy Type Conversion", test_numpy_type_conversion()))
    results.append(("Realistic Metadata", test_metadata_with_numpy_types()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
