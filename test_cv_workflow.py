"""
Test script for cross-validation workflow integration

Tests the train_with_cross_validation method in SMCModelTrainer
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add pipeline directory to path
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

from train_all_models import SMCModelTrainer


def test_cv_workflow():
    """Test cross-validation workflow"""
    print("="*80)
    print("TESTING CROSS-VALIDATION WORKFLOW")
    print("="*80)
    
    # Initialize trainer
    trainer = SMCModelTrainer(
        data_dir='Data',
        output_dir='models/trained'
    )
    
    # Check data availability
    if not trainer.check_data_availability():
        print("\n❌ Cannot proceed without training data")
        print("\nRun this command first:")
        print("  python run_complete_pipeline.py")
        return False
    
    # Get available symbols
    symbols = trainer.get_available_symbols()
    
    if not symbols:
        print("\n❌ No symbols found in dataset")
        return False
    
    # Test with first symbol and Random Forest model
    test_symbol = symbols[0]
    test_model = 'RandomForest'
    
    print(f"\n🧪 Testing CV workflow with {test_symbol} - {test_model}")
    
    try:
        # Train with cross-validation
        result = trainer.train_with_cross_validation(
            symbol=test_symbol,
            model_type=test_model,
            exclude_timeout=False
        )
        
        # Verify result structure
        print(f"\n✅ Training completed successfully!")
        print(f"\n📋 Verifying result structure...")
        
        required_keys = ['model_name', 'symbol', 'cv_results', 'history', 'val_metrics', 'test_metrics']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"❌ Missing keys in result: {missing_keys}")
            return False
        
        print(f"✅ All required keys present")
        
        # Verify CV results structure
        cv_results = result['cv_results']
        cv_required_keys = ['mean_accuracy', 'std_accuracy', 'fold_accuracies', 'is_stable']
        cv_missing_keys = [key for key in cv_required_keys if key not in cv_results]
        
        if cv_missing_keys:
            print(f"❌ Missing keys in cv_results: {cv_missing_keys}")
            return False
        
        print(f"✅ All CV result keys present")
        
        # Verify CV results are in history
        history = result['history']
        history_cv_keys = ['cv_mean_accuracy', 'cv_std_accuracy', 'cv_fold_accuracies', 'cv_is_stable']
        history_cv_missing = [key for key in history_cv_keys if key not in history]
        
        if history_cv_missing:
            print(f"❌ Missing CV keys in history: {history_cv_missing}")
            return False
        
        print(f"✅ CV results properly integrated into history")
        
        # Verify stability flag logic
        cv_std = cv_results['std_accuracy']
        is_stable = cv_results['is_stable']
        expected_stable = cv_std < 0.15
        
        if is_stable != expected_stable:
            print(f"❌ Stability flag incorrect: std={cv_std:.4f}, is_stable={is_stable}, expected={expected_stable}")
            return False
        
        print(f"✅ Stability flag correctly set (std={cv_std:.4f}, stable={is_stable})")
        
        # Print summary
        print(f"\n📊 Test Results Summary:")
        print(f"  Symbol:            {result['symbol']}")
        print(f"  Model:             {result['model_name']}")
        print(f"  CV Mean Accuracy:  {cv_results['mean_accuracy']:.4f}")
        print(f"  CV Std Accuracy:   {cv_results['std_accuracy']:.4f}")
        print(f"  CV Fold Count:     {len(cv_results['fold_accuracies'])}")
        print(f"  Model Stable:      {is_stable}")
        print(f"  Val Accuracy:      {result['val_metrics']['accuracy']:.4f}")
        print(f"  Test Accuracy:     {result['test_metrics']['accuracy']:.4f}")
        
        # Store result for summary report test
        trainer.results[test_symbol] = {test_model: result}
        
        # Test summary report generation
        print(f"\n📄 Testing summary report generation...")
        trainer.generate_summary_report()
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cv_workflow()
    
    if success:
        print("\n" + "="*80)
        print("✅ CROSS-VALIDATION WORKFLOW TEST PASSED")
        print("="*80)
        exit(0)
    else:
        print("\n" + "="*80)
        print("❌ CROSS-VALIDATION WORKFLOW TEST FAILED")
        print("="*80)
        exit(1)
