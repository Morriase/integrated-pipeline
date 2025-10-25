"""
Debug label mismatch between NN and test data
"""

import numpy as np
import pandas as pd
import json

# Load test data
data_path = '/kaggle/working/Data-output/processed_smc_data.csv'
df = pd.read_csv(data_path)

# Load feature names from metadata
metadata_path = '/kaggle/working/Model-output/UNIFIED_RandomForest_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['feature_cols']
X = df[feature_cols].values
y = df['TBM_Label'].values

# Remove NaN labels
valid_mask = ~np.isnan(y)
X = X[valid_mask]
y = y[valid_mask]

print("=" * 80)
print("LABEL ANALYSIS")
print("=" * 80)

print(f"\n📊 Test Data Labels:")
print(f"  Unique labels: {np.unique(y)}")
print(f"  Label counts:")
for label in np.unique(y):
    count = np.sum(y == label)
    pct = count / len(y) * 100
    print(f"    {label}: {count:,} ({pct:.1f}%)")

# Check NN metadata
nn_metadata_path = '/kaggle/working/Model-output/UNIFIED_NeuralNetwork_metadata.json'
with open(nn_metadata_path, 'r') as f:
    nn_metadata = json.load(f)

print(f"\n🧠 NN Training Info:")
if 'label_mapping' in nn_metadata:
    print(f"  Label mapping: {nn_metadata['label_mapping']}")
if 'training_history' in nn_metadata:
    hist = nn_metadata['training_history']
    if 'label_map_reverse' in hist:
        print(f"  Reverse mapping: {hist['label_map_reverse']}")

# Load RF/XGBoost to see what they expect
print(f"\n🌲 RandomForest Training Info:")
rf_metadata_path = '/kaggle/working/Model-output/UNIFIED_RandomForest_metadata.json'
with open(rf_metadata_path, 'r') as f:
    rf_metadata = json.load(f)

if 'target_col' in rf_metadata:
    print(f"  Target column: {rf_metadata['target_col']}")

# Test a small prediction
print(f"\n🔬 Testing NN Predictions:")
from ensemble_predictor import EnsemblePredictor
import pickle
import torch

# Load NN model
nn_model_path = '/kaggle/working/Model-output/UNIFIED_NeuralNetwork.pkl'
with open(nn_model_path, 'rb') as f:
    nn_model = pickle.load(f)

# Load scaler
scaler_path = '/kaggle/working/Model-output/UNIFIED_NeuralNetwork_scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Test on first 10 samples
X_test_small = X[:10]
y_test_small = y[:10]

X_scaled = scaler.transform(X_test_small)
device = next(nn_model.parameters()).device
X_tensor = torch.FloatTensor(X_scaled).to(device)

nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_np = predicted.cpu().numpy()

print(f"  True labels: {y_test_small}")
print(f"  NN predictions: {predicted_np}")
print(f"  NN output shape: {outputs.shape}")
print(f"  NN output classes: {outputs.shape[1]}")

# Check if there's a label remapping needed
print(f"\n💡 Analysis:")
print(f"  Test data has labels: {np.unique(y)}")
print(f"  NN outputs classes: 0 to {outputs.shape[1]-1}")
if outputs.shape[1] == 2:
    print(f"  ⚠️ NN is binary (2 classes) but test data may have 3 classes")
elif outputs.shape[1] == 3:
    print(f"  ✅ NN has 3 classes matching test data")
