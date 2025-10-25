"""
Fix Neural Network Training Issues

This script addresses:
1. Remove constant features (10 features providing no information)
2. Handle high-NaN features (19 features with >50% NaN)
3. Fix severe class imbalance (0.1% Timeout class)
4. Improve network architecture for small dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List


def identify_problematic_features(df: pd.DataFrame, target_col: str = 'TBM_Label') -> dict:
    """
    Identify features that should be removed or handled specially
    
    Returns:
        Dictionary with lists of problematic features
    """
    issues = {
        'constant': [],
        'high_nan': [],
        'all_nan': [],
        'infinite': []
    }
    
    # Get numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    for col in feature_cols:
        # Check for all NaN
        if df[col].isna().all():
            issues['all_nan'].append(col)
            continue
        
        # Check for high NaN (>50%)
        nan_pct = df[col].isna().sum() / len(df)
        if nan_pct > 0.5:
            issues['high_nan'].append(col)
        
        # Check for constant values
        if df[col].notna().sum() > 0:
            unique_vals = df[col].dropna().nunique()
            if unique_vals == 1:
                issues['constant'].append(col)
        
        # Check for infinite values
        if np.isinf(df[col]).any():
            issues['infinite'].append(col)
    
    return issues


def clean_features_for_nn(df: pd.DataFrame, target_col: str = 'TBM_Label') -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean features for Neural Network training
    
    Steps:
    1. Remove constant features
    2. Remove all-NaN features
    3. Handle high-NaN features (impute with median)
    4. Handle infinite values
    5. Remove rows with NaN target
    
    Returns:
        Cleaned dataframe and list of removed features
    """
    print("\n🧹 Cleaning features for Neural Network...")
    
    df_clean = df.copy()
    removed_features = []
    
    # Identify issues
    issues = identify_problematic_features(df_clean, target_col)
    
    # 1. Remove constant features
    if issues['constant']:
        print(f"  Removing {len(issues['constant'])} constant features")
        df_clean = df_clean.drop(columns=issues['constant'])
        removed_features.extend(issues['constant'])
    
    # 2. Remove all-NaN features
    if issues['all_nan']:
        print(f"  Removing {len(issues['all_nan'])} all-NaN features")
        df_clean = df_clean.drop(columns=issues['all_nan'])
        removed_features.extend(issues['all_nan'])
    
    # 3. Handle high-NaN features (impute with median)
    if issues['high_nan']:
        print(f"  Imputing {len(issues['high_nan'])} high-NaN features with median")
        for col in issues['high_nan']:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col].fillna(median_val, inplace=True)
    
    # 4. Handle infinite values
    if issues['infinite']:
        print(f"  Replacing infinite values in {len(issues['infinite'])} features")
        for col in issues['infinite']:
            if col in df_clean.columns:
                # Replace inf with max finite value
                max_val = df_clean[col][np.isfinite(df_clean[col])].max()
                min_val = df_clean[col][np.isfinite(df_clean[col])].min()
                df_clean[col].replace([np.inf], max_val, inplace=True)
                df_clean[col].replace([-np.inf], min_val, inplace=True)
    
    # 5. Remove rows with NaN target
    before_count = len(df_clean)
    df_clean = df_clean[df_clean[target_col].notna()].copy()
    after_count = len(df_clean)
    removed_rows = before_count - after_count
    
    if removed_rows > 0:
        print(f"  Removed {removed_rows:,} rows with NaN target ({removed_rows/before_count*100:.1f}%)")
    
    # 6. Fill any remaining NaN in features with 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    remaining_nans = df_clean[feature_cols].isna().sum().sum()
    if remaining_nans > 0:
        print(f"  Filling {remaining_nans} remaining NaN values with 0")
        df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    print(f"  ✅ Cleaned dataset: {len(df_clean):,} samples, {len(feature_cols)} features")
    
    return df_clean, removed_features


def balance_classes_for_nn(df: pd.DataFrame, target_col: str = 'TBM_Label', 
                           min_samples_per_class: int = 100) -> pd.DataFrame:
    """
    Balance classes by removing timeout class if too small
    
    Args:
        df: Input dataframe
        target_col: Target column name
        min_samples_per_class: Minimum samples required per class
    
    Returns:
        Balanced dataframe
    """
    print("\n⚖️ Balancing classes...")
    
    # Check class distribution
    class_counts = df[target_col].value_counts()
    print(f"  Original distribution:")
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"    {label:>4}: {count:>6} ({pct:>5.1f}%)")
    
    # If timeout class is too small, remove it
    if 0.0 in class_counts.index and class_counts[0.0] < min_samples_per_class:
        print(f"\n  ⚠️ Timeout class has only {class_counts[0.0]} samples (< {min_samples_per_class})")
        print(f"     Removing timeout class for binary classification")
        df_balanced = df[df[target_col] != 0.0].copy()
        
        print(f"\n  New distribution (binary):")
        class_counts = df_balanced[target_col].value_counts()
        for label, count in class_counts.items():
            pct = count / len(df_balanced) * 100
            print(f"    {label:>4}: {count:>6} ({pct:>5.1f}%)")
        
        return df_balanced
    
    return df


def prepare_nn_data(train_path: str, val_path: str, test_path: str,
                    target_col: str = 'TBM_Label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for Neural Network training
    
    Returns:
        Cleaned train, val, test dataframes
    """
    print("=" * 80)
    print("PREPARING DATA FOR NEURAL NETWORK")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Clean features
    train_clean, removed_features = clean_features_for_nn(train_df, target_col)
    
    # Apply same cleaning to val and test
    val_clean = val_df.drop(columns=removed_features, errors='ignore')
    test_clean = test_df.drop(columns=removed_features, errors='ignore')
    
    # Remove NaN targets
    val_clean = val_clean[val_clean[target_col].notna()].copy()
    test_clean = test_clean[test_clean[target_col].notna()].copy()
    
    # Fill NaN in features
    numeric_cols = train_clean.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    val_clean[feature_cols] = val_clean[feature_cols].fillna(0)
    test_clean[feature_cols] = test_clean[feature_cols].fillna(0)
    
    # Balance classes
    train_balanced = balance_classes_for_nn(train_clean, target_col)
    val_balanced = balance_classes_for_nn(val_clean, target_col)
    test_balanced = balance_classes_for_nn(test_clean, target_col)
    
    print(f"\n✅ Data preparation complete!")
    print(f"  Train: {len(train_balanced):,} samples")
    print(f"  Val:   {len(val_balanced):,} samples")
    print(f"  Test:  {len(test_balanced):,} samples")
    
    return train_balanced, val_balanced, test_balanced


if __name__ == "__main__":
    # Check environment
    if Path('/kaggle/working/Data-output').exists():
        data_dir = '/kaggle/working/Data-output'
        output_dir = '/kaggle/working/Data-output-clean'
    else:
        data_dir = 'Data'
        output_dir = 'Data-clean'
    
    # Prepare data
    train_clean, val_clean, test_clean = prepare_nn_data(
        train_path=f'{data_dir}/processed_smc_data_train.csv',
        val_path=f'{data_dir}/processed_smc_data_val.csv',
        test_path=f'{data_dir}/processed_smc_data_test.csv'
    )
    
    # Save cleaned data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving cleaned data to: {output_dir}")
    train_clean.to_csv(f'{output_dir}/processed_smc_data_train_clean.csv', index=False)
    val_clean.to_csv(f'{output_dir}/processed_smc_data_val_clean.csv', index=False)
    test_clean.to_csv(f'{output_dir}/processed_smc_data_test_clean.csv', index=False)
    
    print(f"✅ Cleaned data saved!")
