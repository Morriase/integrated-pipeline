"""
Test script for overfitting report generation

Tests the generate_overfitting_report() method with mock data
"""

import json
import sys
from pathlib import Path

# Add pipeline directory to path
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

from train_all_models import SMCModelTrainer


def create_mock_results():
    """Create mock training results for testing"""
    return {
        'EURUSD': {
            'RandomForest': {
                'model_name': 'RandomForest',
                'symbol': 'EURUSD',
                'history': {
                    'train_accuracy': 0.95,
                    'val_accuracy': 0.78,
                    'train_val_gap': 0.17,
                    'overfitting_detected': True,
                    'cv_mean_accuracy': 0.76,
                    'cv_std_accuracy': 0.05,
                    'cv_is_stable': True
                },
                'val_metrics': {
                    'accuracy': 0.78,
                    'precision': 0.75,
                    'recall': 0.73
                },
                'test_metrics': {
                    'accuracy': 0.76,
                    'precision': 0.74,
                    'recall': 0.72
                }
            },
            'NeuralNetwork': {
                'model_name': 'NeuralNetwork',
                'symbol': 'EURUSD',
                'history': {
                    'train_accuracy': 0.88,
                    'val_accuracy': 0.82,
                    'train_val_gap': 0.06,
                    'overfitting_detected': False
                },
                'val_metrics': {
                    'accuracy': 0.82,
                    'precision': 0.80,
                    'recall': 0.79
                },
                'test_metrics': {
                    'accuracy': 0.81,
                    'precision': 0.79,
                    'recall': 0.78
                }
            }
        },
        'GBPUSD': {
            'RandomForest': {
                'model_name': 'RandomForest',
                'symbol': 'GBPUSD',
                'history': {
                    'train_accuracy': 0.92,
                    'val_accuracy': 0.85,
                    'train_val_gap': 0.07,
                    'overfitting_detected': False,
                    'cv_mean_accuracy': 0.84,
                    'cv_std_accuracy': 0.03,
                    'cv_is_stable': True
                },
                'val_metrics': {
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.82
                },
                'test_metrics': {
                    'accuracy': 0.84,
                    'precision': 0.82,
                    'recall': 0.81
                }
            },
            'NeuralNetwork': {
                'model_name': 'NeuralNetwork',
                'symbol': 'GBPUSD',
                'history': {
                    'train_accuracy': 0.98,
                    'val_accuracy': 0.72,
                    'train_val_gap': 0.26,
                    'overfitting_detected': True
                },
                'val_metrics': {
                    'accuracy': 0.72,
                    'precision': 0.70,
                    'recall': 0.68
                },
                'test_metrics': {
                    'accuracy': 0.70,
                    'precision': 0.68,
                    'recall': 0.66
                }
            }
        }
    }


def test_overfitting_report():
    """Test the overfitting report generation"""
    print("="*80)
    print("Testing Overfitting Report Generation")
    print("="*80)
    
    # Create trainer instance
    trainer = SMCModelTrainer(
        data_dir='Data',
        output_dir='test_output'
    )
    
    # Set mock results
    trainer.results = create_mock_results()
    
    # Generate overfitting report
    print("\n📊 Generating overfitting report with mock data...")
    report = trainer.generate_overfitting_report(output_path='test_output')
    
    # Verify report structure
    print("\n✓ Verifying report structure...")
    assert 'timestamp' in report, "Report missing timestamp"
    assert 'summary' in report, "Report missing summary"
    assert 'all_models' in report, "Report missing all_models"
    assert 'problematic_models' in report, "Report missing problematic_models"
    
    print("  ✓ Report has correct structure")
    
    # Verify summary metrics
    summary = report['summary']
    print("\n✓ Verifying summary metrics...")
    assert summary['total_models'] == 4, f"Expected 4 models, got {summary['total_models']}"
    assert summary['overfitting_models'] == 2, f"Expected 2 overfitting models, got {summary['overfitting_models']}"
    print(f"  ✓ Total models: {summary['total_models']}")
    print(f"  ✓ Overfitting models: {summary['overfitting_models']}")
    print(f"  ✓ Average gap: {summary['average_gap']:.2%}")
    
    # Verify problematic models identified correctly
    print("\n✓ Verifying problematic models...")
    problematic = report['problematic_models']
    assert len(problematic) == 2, f"Expected 2 problematic models, got {len(problematic)}"
    
    # Check that the right models are flagged
    problematic_ids = [(m['symbol'], m['model']) for m in problematic]
    assert ('EURUSD', 'RandomForest') in problematic_ids, "EURUSD-RandomForest should be flagged"
    assert ('GBPUSD', 'NeuralNetwork') in problematic_ids, "GBPUSD-NeuralNetwork should be flagged"
    print(f"  ✓ Correctly identified {len(problematic)} problematic models")
    
    for model in problematic:
        print(f"    - {model['symbol']}-{model['model']}: Gap = {model['train_val_gap']:.2%}")
    
    # Verify files were created
    print("\n✓ Verifying output files...")
    output_path = Path('test_output')
    
    json_file = output_path / 'overfitting_report.json'
    assert json_file.exists(), "JSON report not created"
    print(f"  ✓ JSON report created: {json_file}")
    
    md_file = output_path / 'overfitting_report.md'
    assert md_file.exists(), "Markdown report not created"
    print(f"  ✓ Markdown report created: {md_file}")
    
    viz_file = output_path / 'overfitting_analysis.png'
    assert viz_file.exists(), "Visualization not created"
    print(f"  ✓ Visualization created: {viz_file}")
    
    # Verify JSON file content
    print("\n✓ Verifying JSON file content...")
    with open(json_file, 'r') as f:
        saved_report = json.load(f)
    
    assert saved_report['summary']['total_models'] == 4
    assert saved_report['summary']['overfitting_models'] == 2
    print("  ✓ JSON file contains correct data")
    
    # Verify markdown file content
    print("\n✓ Verifying markdown file content...")
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    assert '# Overfitting Analysis Report' in md_content
    assert 'Models Requiring Attention' in md_content
    assert 'EURUSD - RandomForest' in md_content
    assert 'GBPUSD - NeuralNetwork' in md_content
    print("  ✓ Markdown file contains correct sections")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {json_file}")
    print(f"  - {md_file}")
    print(f"  - {viz_file}")
    print("\nYou can review these files to see the report format.")


if __name__ == "__main__":
    try:
        test_overfitting_report()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
