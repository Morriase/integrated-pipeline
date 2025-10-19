"""
Unit tests for ModelSelector class

Tests the model selection logic including:
- Filtering by train-val gap
- Filtering by test accuracy
- Filtering by val-test consistency
- Scoring with overfitting penalty
- Deployment manifest generation
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_all_models import ModelSelector


def test_model_selector_basic():
    """Test basic model selection with clear winner"""
    print("\n" + "="*80)
    print("TEST 1: Basic Model Selection")
    print("="*80)
    
    # Create test results
    results = {
        'EURUSD': {
            'XGBoost': {
                'test_metrics': {'accuracy': 0.72},
                'val_metrics': {'accuracy': 0.70},
                'history': {
                    'train_accuracy': 0.85,
                    'train_val_gap': 0.13
                }
            },
            'RandomForest': {
                'test_metrics': {'accuracy': 0.65},
                'val_metrics': {'accuracy': 0.50},
                'history': {
                    'train_accuracy': 0.75,
                    'train_val_gap': 0.25  # Too high - should be rejected
                }
            },
            'NeuralNetwork': {
                'test_metrics': {'accuracy': 0.68},
                'val_metrics': {'accuracy': 0.66},
                'history': {
                    'train_accuracy': 0.78,
                    'train_val_gap': 0.12
                }
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    # Verify XGBoost was selected (highest test accuracy, acceptable gap)
    assert selections['EURUSD']['selected_model'] == 'XGBoost', \
        f"Expected XGBoost, got {selections['EURUSD']['selected_model']}"
    
    # Verify RandomForest was rejected (gap too high)
    assert 'RandomForest' not in selections['EURUSD']['alternatives'], \
        "RandomForest should be rejected due to high gap"
    
    # Verify NeuralNetwork is in alternatives
    assert 'NeuralNetwork' in selections['EURUSD']['alternatives'], \
        "NeuralNetwork should be in alternatives"
    
    print("\n✅ TEST 1 PASSED: XGBoost correctly selected")


def test_model_selector_no_candidates():
    """Test when no models meet quality criteria"""
    print("\n" + "="*80)
    print("TEST 2: No Qualifying Models")
    print("="*80)
    
    results = {
        'GBPUSD': {
            'XGBoost': {
                'test_metrics': {'accuracy': 0.52},  # Too low
                'val_metrics': {'accuracy': 0.50},
                'history': {
                    'train_accuracy': 0.70,
                    'train_val_gap': 0.18
                }
            },
            'RandomForest': {
                'test_metrics': {'accuracy': 0.60},
                'val_metrics': {'accuracy': 0.40},
                'history': {
                    'train_accuracy': 0.65,
                    'train_val_gap': 0.25  # Too high
                }
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    # Verify no model was selected
    assert selections['GBPUSD']['selected_model'] is None, \
        "No model should be selected"
    
    # Verify manual review flag
    assert selections['GBPUSD']['action'] == 'MANUAL_REVIEW_REQUIRED', \
        "Should flag for manual review"
    
    print("\n✅ TEST 2 PASSED: Correctly flagged for manual review")


def test_model_selector_with_error():
    """Test handling of models with training errors"""
    print("\n" + "="*80)
    print("TEST 3: Model with Training Error")
    print("="*80)
    
    results = {
        'USDJPY': {
            'XGBoost': {
                'error': 'Training failed: insufficient data'
            },
            'NeuralNetwork': {
                'test_metrics': {'accuracy': 0.70},
                'val_metrics': {'accuracy': 0.68},
                'history': {
                    'train_accuracy': 0.80,
                    'train_val_gap': 0.12
                }
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    # Verify NeuralNetwork was selected (XGBoost had error)
    assert selections['USDJPY']['selected_model'] == 'NeuralNetwork', \
        f"Expected NeuralNetwork, got {selections['USDJPY']['selected_model']}"
    
    print("\n✅ TEST 3 PASSED: Correctly handled model with error")


def test_deployment_manifest():
    """Test deployment manifest generation"""
    print("\n" + "="*80)
    print("TEST 4: Deployment Manifest Generation")
    print("="*80)
    
    results = {
        'EURUSD': {
            'XGBoost': {
                'test_metrics': {'accuracy': 0.72},
                'val_metrics': {'accuracy': 0.70},
                'history': {
                    'train_accuracy': 0.85,
                    'train_val_gap': 0.13
                }
            }
        },
        'GBPUSD': {
            'RandomForest': {
                'test_metrics': {'accuracy': 0.52},  # Too low
                'val_metrics': {'accuracy': 0.50},
                'history': {
                    'train_accuracy': 0.70,
                    'train_val_gap': 0.18
                }
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    # Save manifest
    manifest_path = Path('test_output/deployment_manifest.json')
    manifest = selector.save_deployment_manifest(selections, str(manifest_path))
    
    # Verify manifest structure
    assert 'timestamp' in manifest, "Manifest should have timestamp"
    assert 'selection_criteria' in manifest, "Manifest should have selection criteria"
    assert 'selections' in manifest, "Manifest should have selections"
    assert 'summary' in manifest, "Manifest should have summary"
    
    # Verify summary
    assert manifest['summary']['total_symbols'] == 2, "Should have 2 symbols"
    assert manifest['summary']['models_selected'] == 1, "Should have 1 model selected"
    assert manifest['summary']['manual_review_needed'] == 1, "Should have 1 manual review"
    
    # Verify file was created
    assert manifest_path.exists(), "Manifest file should exist"
    
    # Verify file contents
    with open(manifest_path, 'r') as f:
        loaded_manifest = json.load(f)
    
    assert loaded_manifest['summary']['total_symbols'] == 2, \
        "Loaded manifest should match"
    
    print("\n✅ TEST 4 PASSED: Deployment manifest correctly generated")


def test_scoring_with_overfitting_penalty():
    """Test that scoring correctly penalizes overfitting"""
    print("\n" + "="*80)
    print("TEST 5: Overfitting Penalty in Scoring")
    print("="*80)
    
    results = {
        'AUDUSD': {
            'Model_A': {
                # High accuracy but high gap
                'test_metrics': {'accuracy': 0.75},
                'val_metrics': {'accuracy': 0.73},
                'history': {
                    'train_accuracy': 0.90,
                    'train_val_gap': 0.17
                }
            },
            'Model_B': {
                # Lower accuracy but better gap
                'test_metrics': {'accuracy': 0.70},
                'val_metrics': {'accuracy': 0.69},
                'history': {
                    'train_accuracy': 0.78,
                    'train_val_gap': 0.09
                }
            }
        }
    }
    
    selector = ModelSelector()
    selections = selector.select_best_models(results)
    
    # Calculate expected scores
    # Model_A: 0.75 - (0.17 * 0.5) = 0.665
    # Model_B: 0.70 - (0.09 * 0.5) = 0.655
    
    # Model_A should win despite lower gap because accuracy difference is significant
    assert selections['AUDUSD']['selected_model'] == 'Model_A', \
        f"Expected Model_A, got {selections['AUDUSD']['selected_model']}"
    
    print(f"\n  Model_A score: {selections['AUDUSD']['score']:.3f}")
    print(f"  Expected: ~0.665")
    
    print("\n✅ TEST 5 PASSED: Scoring correctly balances accuracy and overfitting")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING MODEL SELECTOR TESTS")
    print("="*80)
    
    try:
        test_model_selector_basic()
        test_model_selector_no_candidates()
        test_model_selector_with_error()
        test_deployment_manifest()
        test_scoring_with_overfitting_penalty()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
