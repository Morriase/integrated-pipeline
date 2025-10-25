"""
Debug why XGBoost is getting 0% accuracy
"""

import numpy as np
import pandas as pd
import json
import pickle

# Load test data
data_path = '/kaggle/working/Data-output/processed_smc_data.csv'
df = pd.read_csv(data_path)

# Load feature names
metadata_path = '/kaggle/working/Model-output/UNIFIED_RandomForest_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['feature_cols']
X = df[feature_cols].values
y = df['TBM_Label'].values

# Remove NaN and filter to binary
valid_mask = ~np.isnan(y)
X = X[valid_mask]
y = y[valid_mask]

binary_mask = (y == -1) | (y == 1)
X = X[binary_mask]
y = y[binary_mask]

# Take small test sample
X_test = X[:100]
y_test = y[:100]

print("=" * 80)
print("XGBOOST DEBUG")
print("=" * 80)

print(f"\nTest data:")
print(f"  Shape: {X_test.shape}")
print(f"  Labels: {np.unique(y_test)}")
print(f"  Label counts: Loss={np.sum(y_test==-1)}, Win={np.sum(y_test==1)}")

# Load XGBoost model
xgb_path = '/kaggle/working/Model-output/UNIFIED_XGBoost.pkl'
with open(xgb_path, 'rb') as f:
    xgb_model = pickle.load(f)

print(f"\nXGBoost model:")
print(f"  Type: {type(xgb_model)}")
print(f"  Has predict: {hasattr(xgb_model, 'predict')}")

# Make predictions
try:
    predictions = xgb_model.predict(X_test)
    print(f"\nPredictions:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Unique values: {np.unique(predictions)}")
    print(f"  First 20: {predictions[:20]}")
    print(f"\nTrue labels (first 20): {y_test[:20]}")
    
    # Check accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Check if predictions match label space
    print(f"\nLabel space analysis:")
    print(f"  True labels in: {np.unique(y_test)}")
    print(f"  Predictions in: {np.unique(predictions)}")
    
    if set(np.unique(predictions)) != set(np.unique(y_test)):
        print(f"  ⚠️ MISMATCH! Predictions don't match true label space")
        print(f"  This is why accuracy is 0%")
        
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    print(traceback.format_exc())
