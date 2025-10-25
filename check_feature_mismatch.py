"""
Check feature mismatch between models and test data
"""

import json
from pathlib import Path

# Load metadata for both models
rf_meta_path = '/kaggle/working/Model-output/UNIFIED_RandomForest_metadata.json'
xgb_meta_path = '/kaggle/working/Model-output/UNIFIED_XGBoost_metadata.json'

with open(rf_meta_path, 'r') as f:
    rf_meta = json.load(f)

with open(xgb_meta_path, 'r') as f:
    xgb_meta = json.load(f)

rf_features = set(rf_meta['feature_cols'])
xgb_features = set(xgb_meta['feature_cols'])

print("=" * 80)
print("FEATURE COMPARISON")
print("=" * 80)

print(f"\nRandomForest features: {len(rf_features)}")
print(f"XGBoost features: {len(xgb_features)}")

if rf_features == xgb_features:
    print(f"✅ Models use the same features")
else:
    print(f"⚠️ Models use DIFFERENT features!")
    
    only_rf = rf_features - xgb_features
    only_xgb = xgb_features - rf_features
    
    if only_rf:
        print(f"\n  Only in RF ({len(only_rf)}):")
        for f in sorted(only_rf)[:10]:
            print(f"    - {f}")
    
    if only_xgb:
        print(f"\n  Only in XGBoost ({len(only_xgb)}):")
        for f in sorted(only_xgb)[:10]:
            print(f"    - {f}")

print(f"\n💡 Solution: Use the SAME feature list for both models")
print(f"   Currently using RF features ({len(rf_features)}) for test data")
print(f"   But XGBoost expects {len(xgb_features)} features")
