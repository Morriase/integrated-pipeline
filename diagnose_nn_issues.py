"""
Diagnostic script to identify Neural Network training issues
"""

import pandas as pd
import numpy as np
from pathlib import Path

def diagnose_data_issues():
    """Diagnose data quality issues causing NN failure"""
    
    print("=" * 80)
    print("NEURAL NETWORK TRAINING DIAGNOSTICS")
    print("=" * 80)
    
    # Check if we're in Kaggle environment
    if Path('/kaggle/working/Data-output').exists():
        data_dir = '/kaggle/working/Data-output'
    else:
        data_dir = 'Data'
    
    # Load training data
    train_path = f'{data_dir}/processed_smc_data_train.csv'
    
    if not Path(train_path).exists():
        print(f"❌ Training data not found at: {train_path}")
        return
    
    print(f"\n📂 Loading data from: {train_path}")
    df = pd.read_csv(train_path)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Check TBM_Label distribution
    print(f"\n🎯 TBM_Label Analysis:")
    print(f"  Total labels: {len(df)}")
    print(f"  NaN labels: {df['TBM_Label'].isna().sum()} ({df['TBM_Label'].isna().sum()/len(df)*100:.1f}%)")
    print(f"  Valid labels: {df['TBM_Label'].notna().sum()} ({df['TBM_Label'].notna().sum()/len(df)*100:.1f}%)")
    
    if df['TBM_Label'].notna().sum() > 0:
        print(f"\n  Label distribution (valid only):")
        label_counts = df['TBM_Label'].value_counts().sort_index()
        for label, count in label_counts.items():
            pct = count / df['TBM_Label'].notna().sum() * 100
            print(f"    {label:>2}: {count:>6} ({pct:>5.1f}%)")
    
    # Check why labels are NaN
    print(f"\n🔍 Investigating NaN Labels:")
    
    # Check if TBM columns exist
    tbm_cols = [col for col in df.columns if col.startswith('TBM_')]
    print(f"  TBM columns found: {len(tbm_cols)}")
    for col in tbm_cols[:10]:  # Show first 10
        nan_count = df[col].isna().sum()
        print(f"    {col}: {nan_count} NaN ({nan_count/len(df)*100:.1f}%)")
    
    # Check feature quality
    print(f"\n📈 Feature Quality Check:")
    
    # Identify columns with high NaN percentage
    nan_pcts = df.isna().sum() / len(df) * 100
    high_nan_cols = nan_pcts[nan_pcts > 50].sort_values(ascending=False)
    
    if len(high_nan_cols) > 0:
        print(f"  ⚠️ Columns with >50% NaN values: {len(high_nan_cols)}")
        for col, pct in high_nan_cols.head(10).items():
            print(f"    {col}: {pct:.1f}%")
    else:
        print(f"  ✅ No columns with >50% NaN values")
    
    # Check for constant features
    print(f"\n🔢 Constant Feature Check:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_features = []
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            unique_vals = df[col].dropna().nunique()
            if unique_vals == 1:
                constant_features.append(col)
    
    if constant_features:
        print(f"  ⚠️ Constant features found: {len(constant_features)}")
        for col in constant_features[:10]:
            print(f"    {col}")
    else:
        print(f"  ✅ No constant features")
    
    # Check for infinite values
    print(f"\n♾️ Infinite Value Check:")
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            inf_cols.append((col, inf_count))
    
    if inf_cols:
        print(f"  ⚠️ Columns with infinite values: {len(inf_cols)}")
        for col, count in inf_cols[:10]:
            print(f"    {col}: {count} infinite values")
    else:
        print(f"  ✅ No infinite values")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"=" * 80)
    
    nan_pct = df['TBM_Label'].isna().sum() / len(df) * 100
    
    if nan_pct > 80:
        print(f"  ❌ CRITICAL: {nan_pct:.1f}% of labels are NaN!")
        print(f"     → Check data_preparation_pipeline.py TBM calculation logic")
        print(f"     → Verify that trades are being properly labeled")
        print(f"     → Consider if TBM_Label is being set correctly")
    elif nan_pct > 50:
        print(f"  ⚠️ WARNING: {nan_pct:.1f}% of labels are NaN")
        print(f"     → Review TBM labeling criteria")
        print(f"     → Check if timeout threshold is too strict")
    else:
        print(f"  ✅ Label quality is acceptable ({nan_pct:.1f}% NaN)")
    
    if len(high_nan_cols) > 10:
        print(f"\n  ⚠️ Many features have high NaN rates")
        print(f"     → Consider imputation or feature removal")
        print(f"     → Check feature engineering logic")
    
    if constant_features:
        print(f"\n  ⚠️ Constant features detected")
        print(f"     → Remove these features before training")
        print(f"     → They provide no information")
    
    # Check if we can train with remaining data
    valid_samples = df['TBM_Label'].notna().sum()
    print(f"\n📊 Training Feasibility:")
    print(f"  Valid samples: {valid_samples:,}")
    
    if valid_samples < 100:
        print(f"  ❌ INSUFFICIENT DATA: Need at least 100 samples")
        print(f"     → Cannot train neural network reliably")
    elif valid_samples < 500:
        print(f"  ⚠️ LIMITED DATA: {valid_samples} samples is marginal")
        print(f"     → Consider data augmentation")
        print(f"     → Use simpler models (RandomForest, XGBoost)")
    elif valid_samples < 2000:
        print(f"  ⚙️ MODERATE DATA: {valid_samples} samples")
        print(f"     → Use small network architecture")
        print(f"     → Apply strong regularization")
    else:
        print(f"  ✅ SUFFICIENT DATA: {valid_samples} samples")
        print(f"     → Can train neural network")
    
    print(f"\n" + "=" * 80)
    print(f"DIAGNOSIS COMPLETE")
    print(f"=" * 80)


if __name__ == "__main__":
    diagnose_data_issues()
