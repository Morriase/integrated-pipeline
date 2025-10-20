"""
Test Consensus Ensemble - Only trade when all 3 models agree
"""

import pandas as pd
import numpy as np
from models.ensemble_model import ConsensusEnsembleSMCModel
from pathlib import Path


def test_consensus_ensemble():
    """Test the consensus ensemble on test data"""
    
    print("=" * 80)
    print("CONSENSUS ENSEMBLE EVALUATION")
    print("=" * 80)
    print("\n🎯 Strategy: Only trade when RandomForest + XGBoost + NeuralNetwork ALL agree")
    
    # Initialize ensemble
    ensemble = ConsensusEnsembleSMCModel(symbol='UNIFIED')
    
    # Load models
    model_dir = '/kaggle/working/Model-output'
    if not Path(model_dir).exists():
        model_dir = 'models/trained'  # Fallback for local
    
    ensemble.load_models(model_dir=model_dir)
    
    if not ensemble.is_trained:
        print("\n❌ Failed to load all 3 models")
        return
    
    # Set to strict consensus mode
    ensemble.set_consensus_mode('strict')
    
    # Load test data
    data_dir = '/kaggle/working/Data-output'
    if not Path(data_dir).exists():
        data_dir = 'Data'  # Fallback for local
    
    test_path = Path(data_dir) / 'processed_smc_data_test.csv'
    
    if not test_path.exists():
        print(f"\n❌ Test data not found: {test_path}")
        return
    
    print(f"\n📂 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Remove NaN labels
    original_size = len(test_df)
    test_df = test_df[test_df['TBM_Label'].notna()].copy()
    print(f"  Samples with valid labels: {len(test_df)}/{original_size}")
    
    # Get features and labels
    X_test = test_df
    y_test = test_df['TBM_Label'].values
    
    # Evaluate consensus ensemble
    print("\n" + "=" * 80)
    print("STRICT CONSENSUS MODE (All 3 models must agree)")
    print("=" * 80)
    
    metrics_strict = ensemble.evaluate(X_test, y_test, 'Test')
    
    # Compare with majority voting
    print("\n" + "=" * 80)
    print("MAJORITY CONSENSUS MODE (2 out of 3 models agree)")
    print("=" * 80)
    
    ensemble.set_consensus_mode('majority')
    metrics_majority = ensemble.evaluate(X_test, y_test, 'Test')
    
    # Get individual model predictions for comparison
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PREDICTIONS (for comparison)")
    print("=" * 80)
    
    individual_preds = ensemble.get_individual_predictions(X_test)
    
    print(f"\n  Agreement Analysis:")
    n_samples = len(y_test)
    
    # Count agreements
    all_agree = 0
    two_agree = 0
    none_agree = 0
    
    for i in range(n_samples):
        rf = individual_preds['RandomForest'][i]
        xgb = individual_preds['XGBoost'][i]
        nn = individual_preds['NeuralNetwork'][i]
        
        if rf == xgb == nn:
            all_agree += 1
        elif rf == xgb or rf == nn or xgb == nn:
            two_agree += 1
        else:
            none_agree += 1
    
    print(f"    All 3 agree:  {all_agree}/{n_samples} ({all_agree/n_samples*100:.1f}%)")
    print(f"    2 out of 3:   {two_agree}/{n_samples} ({two_agree/n_samples*100:.1f}%)")
    print(f"    No agreement: {none_agree}/{n_samples} ({none_agree/n_samples*100:.1f}%)")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Strict':<15} {'Majority':<15}")
    print("-" * 60)
    print(f"{'Trade Opportunities':<30} {metrics_strict['high_confidence_trades']:<15} {metrics_majority['high_confidence_trades']:<15}")
    print(f"{'Confidence Rate':<30} {metrics_strict['confidence_rate']:<15.1%} {metrics_majority['confidence_rate']:<15.1%}")
    print(f"{'Accuracy':<30} {metrics_strict['high_conf_accuracy']:<15.3f} {metrics_majority['high_conf_accuracy']:<15.3f}")
    print(f"{'Win Rate':<30} {metrics_strict['win_rate']:<15.1%} {metrics_majority['win_rate']:<15.1%}")
    print(f"{'Profit Factor':<30} {metrics_strict['profit_factor']:<15.2f} {metrics_majority['profit_factor']:<15.2f}")
    print(f"{'EV per Trade':<30} {metrics_strict['expected_value_per_trade']:<15.2f}R {metrics_majority['expected_value_per_trade']:<15.2f}R")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if metrics_strict['expected_value_per_trade'] > metrics_majority['expected_value_per_trade']:
        print("\n✅ Use STRICT consensus mode (all 3 models agree)")
        print(f"   - Higher EV: {metrics_strict['expected_value_per_trade']:.2f}R vs {metrics_majority['expected_value_per_trade']:.2f}R")
        print(f"   - Fewer but higher quality trades")
    else:
        print("\n✅ Use MAJORITY consensus mode (2 out of 3 agree)")
        print(f"   - Higher EV: {metrics_majority['expected_value_per_trade']:.2f}R vs {metrics_strict['expected_value_per_trade']:.2f}R")
        print(f"   - More trade opportunities")
    
    print("\n🎉 Consensus ensemble evaluation complete!")


if __name__ == "__main__":
    test_consensus_ensemble()
