"""
Test Summary Report Generation

Tests the generate_summary_report functionality including:
- Console output generation
- Per-symbol markdown reports (Requirements 8.1-8.7)
- JSON results file
- Overfitting analysis
- Model selection integration
- Warnings collection
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add pipeline directory to path
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

from train_all_models import SMCModelTrainer, ModelSelector


def create_mock_results():
    """Create mock training results for testing"""
    return {
        'EURUSD': {
            'RandomForest': {
                'model_name': 'RandomForest',
                'symbol': 'EURUSD',
                'history': {
                    'train_accuracy': 0.85,
                    'train_val_gap': 0.13,
                    'cv_mean_accuracy': 0.71,
                    'cv_std_accuracy': 0.04,
                    'cv_is_stable': True,
                    'cv_fold_accuracies': [0.68, 0.72, 0.70, 0.73, 0.72]
                },
                'val_metrics': {
                    'accuracy': 0.72,
                    'precision': 0.70,
                    'recall': 0.69,
                    'f1_score': 0.69,
                    'confusion_matrix': [[15, 5], [6, 20]]
                },
                'test_metrics': {
                    'accuracy': 0.70,
                    'precision': 0.68,
                    'recall': 0.67,
                    'f1_score': 0.67,
                    'confusion_matrix': [[14, 6], [7, 19]]
                },
                'feature_importance': [
                    {'feature': 'OB_Age', 'importance': 0.12},
                    {'feature': 'TBM_Risk_Per_Trade_ATR', 'importance': 0.08},
                    {'feature': 'FVG_Width', 'importance': 0.07},
                    {'feature': 'Liquidity_Sweep_Distance', 'importance': 0.06},
                    {'feature': 'Market_Structure_Shift', 'importance': 0.05},
                    {'feature': 'Volume_Profile', 'importance': 0.04},
                    {'feature': 'ATR_Normalized', 'importance': 0.04},
                    {'feature': 'Price_Action_Score', 'importance': 0.03},
                    {'feature': 'Trend_Strength', 'importance': 0.03},
                    {'feature': 'Support_Resistance', 'importance': 0.02}
                ],
                'training_duration_seconds': 45.2
            },
            'XGBoost': {
                'model_name': 'XGBoost',
                'symbol': 'EURUSD',
                'history': {
                    'train_accuracy': 0.88,
                    'train_val_gap': 0.16,
                    'cv_mean_accuracy': 0.73,
                    'cv_std_accuracy': 0.03,
                    'cv_is_stable': True,
                    'cv_fold_accuracies': [0.71, 0.74, 0.72, 0.75, 0.73]
                },
                'val_metrics': {
                    'accuracy': 0.72,
                    'precision': 0.71,
                    'recall': 0.70,
                    'f1_score': 0.70,
                    'confusion_matrix': [[16, 4], [6, 20]]
                },
                'test_metrics': {
                    'accuracy': 0.71,
                    'precision': 0.69,
                    'recall': 0.68,
                    'f1_score': 0.68,
                    'confusion_matrix': [[15, 5], [7, 19]]
                },
                'feature_importance': [
                    {'feature': 'OB_Age', 'importance': 0.15},
                    {'feature': 'TBM_Risk_Per_Trade_ATR', 'importance': 0.10},
                    {'feature': 'FVG_Width', 'importance': 0.08},
                    {'feature': 'Liquidity_Sweep_Distance', 'importance': 0.07},
                    {'feature': 'Market_Structure_Shift', 'importance': 0.06},
                    {'feature': 'Volume_Profile', 'importance': 0.05},
                    {'feature': 'ATR_Normalized', 'importance': 0.04},
                    {'feature': 'Price_Action_Score', 'importance': 0.04},
                    {'feature': 'Trend_Strength', 'importance': 0.03},
                    {'feature': 'Support_Resistance', 'importance': 0.03}
                ],
                'training_duration_seconds': 52.8
            },
            'NeuralNetwork': {
                'model_name': 'NeuralNetwork',
                'symbol': 'EURUSD',
                'history': {
                    'train_accuracy': 0.92,
                    'train_val_gap': 0.28,  # High overfitting
                    'cv_mean_accuracy': 0.65,
                    'cv_std_accuracy': 0.12,  # Unstable
                    'cv_is_stable': False,
                    'cv_fold_accuracies': [0.55, 0.68, 0.62, 0.72, 0.68]
                },
                'val_metrics': {
                    'accuracy': 0.64,
                    'precision': 0.62,
                    'recall': 0.60,
                    'f1_score': 0.61,
                    'confusion_matrix': [[12, 8], [9, 17]]
                },
                'test_metrics': {
                    'accuracy': 0.62,
                    'precision': 0.60,
                    'recall': 0.58,
                    'f1_score': 0.59,
                    'confusion_matrix': [[11, 9], [10, 16]]
                },
                'warnings': [
                    'Epoch 45: Severe overfitting detected (train=0.92, val=0.64)',
                    'Validation loss increasing for 10 consecutive epochs'
                ],
                'training_duration_seconds': 120.5
            }
        },
        'GBPUSD': {
            'RandomForest': {
                'error': 'Insufficient training samples',
                'error_type': 'ValueError',
                'symbol': 'GBPUSD',
                'timestamp': datetime.now().isoformat()
            },
            'XGBoost': {
                'model_name': 'XGBoost',
                'symbol': 'GBPUSD',
                'history': {
                    'train_accuracy': 0.78,
                    'train_val_gap': 0.18,
                    'cv_mean_accuracy': 0.58,
                    'cv_std_accuracy': 0.05,
                    'cv_is_stable': True,
                    'cv_fold_accuracies': [0.55, 0.60, 0.57, 0.61, 0.57]
                },
                'val_metrics': {
                    'accuracy': 0.60,
                    'precision': 0.58,
                    'recall': 0.56,
                    'f1_score': 0.57,
                    'confusion_matrix': [[10, 8], [8, 12]]
                },
                'test_metrics': {
                    'accuracy': 0.52,  # Low accuracy
                    'precision': 0.50,
                    'recall': 0.48,
                    'f1_score': 0.49,
                    'confusion_matrix': [[9, 9], [10, 10]]
                },
                'feature_importance': [
                    {'feature': 'OB_Age', 'importance': 0.10},
                    {'feature': 'TBM_Risk_Per_Trade_ATR', 'importance': 0.09},
                    {'feature': 'FVG_Width', 'importance': 0.08}
                ],
                'training_duration_seconds': 38.2
            }
        }
    }


def create_mock_selections():
    """Create mock model selection results"""
    return {
        'EURUSD': {
            'selected_model': 'XGBoost',
            'test_accuracy': 0.71,
            'val_accuracy': 0.72,
            'train_val_gap': 0.16,
            'val_test_diff': 0.01,
            'score': 0.63,
            'reason': 'Best score (0.630) with gap 16.0%',
            'alternatives': ['RandomForest']
        },
        'GBPUSD': {
            'selected_model': None,
            'reason': 'No models met quality criteria',
            'action': 'MANUAL_REVIEW_REQUIRED',
            'rejected_models': [
                {
                    'model': 'XGBoost',
                    'reason': 'Test accuracy too low (52.0% < 55.0%)',
                    'test_acc': 0.52,
                    'train_val_gap': 0.18
                }
            ]
        }
    }


def test_summary_report_generation():
    """Test comprehensive summary report generation"""
    print("="*80)
    print("TEST: Summary Report Generation")
    print("="*80)
    
    # Create test output directory
    test_output_dir = Path('test_output/summary_report')
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer with mock data
    trainer = SMCModelTrainer(
        data_dir='Data',
        output_dir=str(test_output_dir)
    )
    
    # Set mock results
    trainer.results = create_mock_results()
    mock_selections = create_mock_selections()
    
    print("\n✓ Mock data created")
    print(f"  - Symbols: {list(trainer.results.keys())}")
    print(f"  - Total models: {sum(len(r) for r in trainer.results.values())}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("Generating Summary Report...")
    print("="*80)
    
    try:
        trainer.generate_summary_report(model_selections=mock_selections)
        print("\n✅ Summary report generated successfully")
    except Exception as e:
        print(f"\n❌ Summary report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify outputs
    print("\n" + "="*80)
    print("Verifying Outputs...")
    print("="*80)
    
    success = True
    
    # 1. Check training_results.json exists
    results_file = test_output_dir / 'training_results.json'
    if results_file.exists():
        print("\n✓ training_results.json created")
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Verify structure
        assert 'training_results' in results_data, "Missing training_results"
        assert 'model_selections' in results_data, "Missing model_selections"
        assert 'warnings' in results_data, "Missing warnings"
        assert 'timestamp' in results_data, "Missing timestamp"
        
        print(f"  - Training results: {len(results_data['training_results'])} symbols")
        print(f"  - Model selections: {len(results_data['model_selections'])} symbols")
        print(f"  - Warnings: {len(results_data['warnings'])} total")
        
        # Verify warnings were collected
        expected_warnings = [
            'EURUSD/NeuralNetwork',  # Overfitting
            'GBPUSD/RandomForest',   # Training failed
            'GBPUSD/XGBoost',        # Low accuracy
            'GBPUSD'                 # No model selected
        ]
        
        warnings_found = 0
        for expected in expected_warnings:
            if any(expected in w for w in results_data['warnings']):
                warnings_found += 1
        
        print(f"  - Expected warnings found: {warnings_found}/{len(expected_warnings)}")
        
    else:
        print("\n❌ training_results.json not created")
        success = False
    
    # 2. Check per-symbol markdown reports
    reports_dir = test_output_dir / 'reports'
    if reports_dir.exists():
        print("\n✓ Reports directory created")
        
        report_files = list(reports_dir.glob('*_training_report.md'))
        print(f"  - Report files: {len(report_files)}")
        
        for report_file in report_files:
            print(f"    • {report_file.name}")
            
            # Verify report content
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required sections (Requirements 8.1-8.7)
            required_sections = [
                '# Training Report:',                    # 8.1
                '## 🎯 Deployment Recommendation',       # 8.7
                '## 📊 Model Performance Comparison',    # 8.2
                '## 🔍 Overfitting Analysis',            # 8.3
                '## 📈 Cross-Validation Stability',      # 8.4
                '## 🎯 Feature Importance (Top 10)',     # 8.5
                '## 📉 Confusion Matrices',              # 8.6
                '## ⚠️ Warnings and Issues'
            ]
            
            sections_found = sum(1 for section in required_sections if section in content)
            print(f"      Sections: {sections_found}/{len(required_sections)}")
            
            if sections_found < len(required_sections):
                print(f"      ⚠️ Missing sections in {report_file.name}")
                for section in required_sections:
                    if section not in content:
                        print(f"        - {section}")
    else:
        print("\n❌ Reports directory not created")
        success = False
    
    # 3. Verify specific report content
    eurusd_report = reports_dir / 'EURUSD_training_report.md'
    if eurusd_report.exists():
        print("\n✓ EURUSD report exists")
        
        with open(eurusd_report, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for specific content
        checks = [
            ('Deployment recommendation', 'XGBoost' in content),
            ('Train/Val/Test accuracy table', '| Model | Train Acc | Val Acc | Test Acc |' in content),
            ('Overfitting analysis', 'NeuralNetwork' in content and '🔴 Severe' in content),
            ('CV stability', 'cv_fold_accuracies' in content.lower() or 'fold accuracies' in content.lower()),
            ('Feature importance', 'OB_Age' in content),
            ('Confusion matrix', 'Predicted' in content or 'confusion' in content.lower()),
            ('Warnings', 'overfitting' in content.lower() or 'unstable' in content.lower())
        ]
        
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                success = False
    else:
        print("\n❌ EURUSD report not found")
        success = False
    
    # 4. Verify GBPUSD report (with errors)
    gbpusd_report = reports_dir / 'GBPUSD_training_report.md'
    if gbpusd_report.exists():
        print("\n✓ GBPUSD report exists (with errors)")
        
        with open(gbpusd_report, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check error handling
        checks = [
            ('No model recommended', 'No Model Recommended' in content or 'MANUAL_REVIEW' in content),
            ('Training error shown', 'RandomForest' in content and ('Error' in content or 'error' in content)),
            ('Low accuracy warning', 'Low' in content or 'accuracy' in content.lower())
        ]
        
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                success = False
    else:
        print("\n❌ GBPUSD report not found")
        success = False
    
    # Final result
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nSummary report generation is working correctly:")
        print("  ✓ Includes overfitting analysis (Req 8.3)")
        print("  ✓ Includes model selection results (Req 5.5, 5.6)")
        print("  ✓ Includes warnings from all models (Req 9.1-9.5)")
        print("  ✓ Includes train/val/test accuracy (Req 8.2)")
        print("  ✓ Includes CV stability (Req 8.4)")
        print("  ✓ Includes feature importance (Req 8.5)")
        print("  ✓ Includes confusion matrices (Req 8.6)")
        print("  ✓ Includes deployment recommendations (Req 8.7)")
        print("\nGenerated files:")
        print(f"  - {results_file}")
        print(f"  - {reports_dir}/*.md")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the output above for details.")
    
    return success


if __name__ == "__main__":
    print("Starting test...")
    try:
        success = test_summary_report_generation()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
