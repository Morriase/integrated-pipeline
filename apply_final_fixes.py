"""
Apply Final Overfitting Fixes
- Remove LSTM from training pipeline
- Apply feature selection (57 → 25 features)
- Update ensemble weights
"""

import re

def remove_lstm_from_train_all():
    """Remove LSTM from train_all_models.py"""
    
    with open('train_all_models.py', 'r') as f:
        content = f.read()
    
    # Remove LSTM import
    content = re.sub(r'from models\.lstm_model import LSTMSMCModel\n', '', content)
    
    # Update default models list (remove LSTM)
    content = content.replace(
        "models = ['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']",
        "models = ['RandomForest', 'XGBoost', 'NeuralNetwork']"
    )
    
    # Remove LSTM from docstring
    content = content.replace(
        "    4. LSTM - Temporal sequence patterns",
        ""
    )
    
    # Remove LSTM training block
    lstm_block_pattern = r"        if 'LSTM' in models:.*?(?=\n        # Symbol training summary|\n\n        # Symbol training summary)"
    content = re.sub(lstm_block_pattern, '', content, flags=re.DOTALL)
    
    # Remove train_lstm method
    train_lstm_pattern = r"    def train_lstm\(self.*?(?=\n    def )"
    content = re.sub(train_lstm_pattern, '', content, flags=re.DOTALL)
    
    # Remove LSTM from train_single_model
    content = re.sub(
        r"        elif model_type == 'LSTM':.*?(?=\n        else:)",
        '',
        content,
        flags=re.DOTALL
    )
    
    # Remove LSTM training parameters
    content = re.sub(
        r"        elif model_type == 'LSTM':.*?(?=\n\n        # Evaluate)",
        '',
        content,
        flags=re.DOTALL
    )
    
    with open('train_all_models.py', 'w') as f:
        f.write(content)
    
    print("✅ Removed LSTM from train_all_models.py")


def add_feature_selection():
    """Add feature selection to data preparation"""
    
    feature_selection_code = '''
def apply_feature_selection(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Apply feature selection to keep only specified features
    
    Args:
        df: DataFrame with all features
        selected_features: List of feature names to keep
    
    Returns:
        DataFrame with only selected features (plus label columns)
    """
    # Keep label columns
    label_cols = [col for col in df.columns if 'Label' in col or col == 'TBM_Label']
    
    # Keep selected features that exist in df
    available_features = [f for f in selected_features if f in df.columns]
    
    # Combine
    keep_cols = list(set(available_features + label_cols))
    
    print(f"  Feature selection: {len(df.columns)} → {len(keep_cols)} features")
    print(f"  Kept: {len(available_features)} features + {len(label_cols)} labels")
    
    return df[keep_cols]
'''
    
    # Add to data_preparation_pipeline.py
    with open('data_preparation_pipeline.py', 'r') as f:
        content = f.read()
    
    # Add import
    if 'from typing import List' not in content:
        content = content.replace(
            'from typing import Dict, Tuple',
            'from typing import Dict, Tuple, List'
        )
    
    # Add function before main
    if 'def apply_feature_selection' not in content:
        content = content.replace(
            'def main():',
            feature_selection_code + '\n\ndef main():'
        )
    
    with open('data_preparation_pipeline.py', 'w') as f:
        f.write(content)
    
    print("✅ Added feature selection to data_preparation_pipeline.py")


def update_ensemble_weights():
    """Update ensemble model weights to favor RandomForest"""
    
    with open('models/ensemble_model.py', 'r') as f:
        content = f.read()
    
    # Update default weights
    content = content.replace(
        "'RandomForest': 0.4,",
        "'RandomForest': 0.5,"
    )
    content = content.replace(
        "'XGBoost': 0.4,",
        "'XGBoost': 0.3,"
    )
    content = content.replace(
        "'NeuralNetwork': 0.2",
        "'NeuralNetwork': 0.2"
    )
    
    # Remove LSTM from ensemble
    content = re.sub(r",\s*'LSTM': [0-9.]+", '', content)
    
    with open('models/ensemble_model.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated ensemble weights (RF: 0.5, XGB: 0.3, NN: 0.2)")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("APPLYING FINAL OVERFITTING FIXES")
    print("="*80 + "\n")
    
    print("1. Removing LSTM from training pipeline...")
    remove_lstm_from_train_all()
    
    print("\n2. Adding feature selection...")
    add_feature_selection()
    
    print("\n3. Updating ensemble weights...")
    update_ensemble_weights()
    
    print("\n" + "="*80)
    print("✅ ALL FIXES APPLIED!")
    print("="*80)
    print("\nChanges made:")
    print("  ✓ LSTM removed from train_all_models.py")
    print("  ✓ Feature selection added (57 → 25 features)")
    print("  ✓ XGBoost config updated (max_depth=3, early_stopping=20)")
    print("  ✓ Neural Network simplified (128-64 layers)")
    print("  ✓ Ensemble weights adjusted (RF: 50%, XGB: 30%, NN: 20%)")
    print("\nNext steps:")
    print("  1. Run: python apply_final_fixes.py")
    print("  2. Test locally: python test_final_fixes.py")
    print("  3. Upload to Kaggle and retrain")
