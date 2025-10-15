"""
Ensemble Modeling Pipeline Scaffold
- Trains multiple models (different seeds, architectures, or data splits)
- Combines predictions via averaging, majority vote, or stacking
- Example with RandomForest and LogisticRegression
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# --- Load data ---
df = pd.read_csv('merged_M15_multiTF.csv')
features = [col for col in df.columns if col not in [
    'target', 'time', 'symbol']]
X, y = df[features], df['target']

# --- Train multiple models ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
models = [
    RandomForestClassifier(n_estimators=100, random_state=1),
    RandomForestClassifier(n_estimators=100, random_state=2),
    LogisticRegression(max_iter=1000, random_state=3)
]
for model in models:
    model.fit(X_train, y_train)

# --- Ensemble: Averaging (for probabilities) ---
probs = np.stack([m.predict_proba(X_test) for m in models])
avg_probs = np.mean(probs, axis=0)
avg_preds = np.argmax(avg_probs, axis=1)

# --- Ensemble: Majority Vote ---
preds = np.stack([m.predict(X_test) for m in models])
vote_preds = mode(preds, axis=0).mode[0]

# --- Evaluate ---
print('Averaging Ensemble Accuracy:', accuracy_score(y_test, avg_preds))
print('Majority Vote Ensemble Accuracy:', accuracy_score(y_test, vote_preds))

# --- Stacking (meta-model) ---
# Example: Use predictions as features for a meta-model
meta_X = preds.T
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_X, y_test)
meta_preds = meta_model.predict(meta_X)
print('Stacking Ensemble Accuracy:', accuracy_score(y_test, meta_preds))
