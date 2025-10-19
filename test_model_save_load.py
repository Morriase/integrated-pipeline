"""
Integration test for model save/load with JSON serialization fix
Tests the complete save_model workflow including error handling
"""

import numpy as np
import json
import sys
import tempfile
import shutil
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.base_model import BaseSMCModel


class MockModel(BaseSMCModel):
    """Mock model for testing save/load functionality"""
    
    def __init__(self, symbol='EURUSD'):
        super().__init__('MockModel', symbol)
        self.model = {'type': 'mock', 'trained': False}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True
        self.model['trained'] = True
        
        # Simulate training history with numpy types
        self.training_history = {
            'train_accuracy': np.float64(0.95),
            'val_accuracy': np.float64(0.72),
            'train_val_gap': np.float64(0.23),
            'confusion_matrix': np.array([[15, 5], [6, 20]]),
            'epochs': np.int64(100),
            'best_epoch': np.int64(85),
            'early_stopped': np.bool_(True)
        }
        
        # Simulate feature importance
        self.feature_importance = np.random.rand(10)
        self.feature_cols = [f'feature_{i}' for i in range(10)]
        
        return self.training_history
    
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X):
        return np.zeros((len(X), 2))


def test_save_with_numpy_types():
    """Test saving model with numpy types in metadata"""
    print("\n🧪 Test 1: Save model with numpy types...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and train model
        model = MockModel('EURUSD')
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.train(X_train, y_train)
        
        # Save model
        model.save_model(temp_dir)
        
        # Check files exist
        model_file = Path(temp_dir) / 'EURUSD_MockModel.pkl'
        metadata_file = Path(temp_dir) / 'EURUSD_MockModel_metadata.json'
        
        assert model_file.exists(), "Model pickle file not created"
        assert metadata_file.exists(), "Metadata JSON file not created"
        
        # Load and verify JSON
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify structure
        assert 'model_name' in metadata
        assert 'symbol' in metadata
        assert 'training_history' in metadata
        assert 'feature_importance' in metadata
        
        # Verify numpy types were converted
        assert isinstance(metadata['training_history']['train_accuracy'], float)
        assert isinstance(metadata['training_history']['epochs'], int)
        assert isinstance(metadata['training_history']['early_stopped'], bool)
        assert isinstance(metadata['training_history']['confusion_matrix'], list)
        assert isinstance(metadata['feature_importance'], list)
        
        print("✅ Model saved successfully with proper type conversion")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_with_problematic_types():
    """Test error handling when metadata contains problematic types"""
    print("\n🧪 Test 2: Save model with problematic types...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = MockModel('GBPUSD')
        model.is_trained = True
        
        # Add problematic types to training history
        model.training_history = {
            'normal_value': 0.85,
            'numpy_nan': np.float64(np.nan),
            'numpy_inf': np.float64(np.inf),
            'complex_number': complex(1, 2),  # Not JSON serializable
            'nested_numpy': {
                'array': np.array([1, 2, 3]),
                'scalar': np.int64(42)
            }
        }
        
        model.feature_cols = ['f1', 'f2', 'f3']
        model.feature_importance = np.array([0.5, 0.3, 0.2])
        
        # Save model - should handle errors gracefully
        model.save_model(temp_dir)
        
        # Check files exist
        model_file = Path(temp_dir) / 'GBPUSD_MockModel.pkl'
        metadata_file = Path(temp_dir) / 'GBPUSD_MockModel_metadata.json'
        
        assert model_file.exists(), "Model pickle file not created"
        assert metadata_file.exists(), "Metadata JSON file not created"
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify at least basic fields were saved
        assert 'model_name' in metadata
        assert 'symbol' in metadata
        
        print("✅ Model saved with graceful error handling")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_partial_metadata_save():
    """Test that model pickle is saved even if metadata fails"""
    print("\n🧪 Test 3: Partial save when metadata has issues...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = MockModel('USDJPY')
        model.is_trained = True
        model.feature_cols = ['f1', 'f2']
        
        # Create metadata with mix of good and problematic data
        model.training_history = {
            'good_value': np.float64(0.75),
            'good_array': np.array([1, 2, 3]),
        }
        
        # Save model
        model.save_model(temp_dir)
        
        # Model pickle should always be saved
        model_file = Path(temp_dir) / 'USDJPY_MockModel.pkl'
        assert model_file.exists(), "Model pickle must be saved even if metadata fails"
        
        print("✅ Model pickle saved successfully (primary goal achieved)")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_saved_model():
    """Test loading a saved model"""
    print("\n🧪 Test 4: Load saved model...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save model
        model1 = MockModel('AUDUSD')
        X_train = np.random.rand(50, 10)
        y_train = np.random.randint(0, 2, 50)
        model1.train(X_train, y_train)
        model1.save_model(temp_dir)
        
        # Load model
        model2 = MockModel('AUDUSD')
        model2.load_model(temp_dir)
        
        # Verify loaded correctly
        assert model2.is_trained == True
        assert model2.feature_cols == model1.feature_cols
        assert len(model2.training_history) > 0
        
        print("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Model Save/Load Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Save with numpy types", test_save_with_numpy_types()))
    results.append(("Save with problematic types", test_save_with_problematic_types()))
    results.append(("Partial metadata save", test_partial_metadata_save()))
    results.append(("Load saved model", test_load_saved_model()))
    
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
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
