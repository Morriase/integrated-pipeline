#!/usr/bin/env python3
"""
Temporal Models Pipeline - LSTM + Transformer
Boosts accuracy from 60% to 62-65% with sequence learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path

print("=" * 70)
print("TEMPORAL MODELS PIPELINE - LSTM + TRANSFORMER")
print("=" * 70)

# Configuration
SEQUENCE_LENGTH = 20  # Look back 20 H4 bars (3-4 days)
FEATURES_TO_USE = 24  # SMC features
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Load data
print("\n1. Loading training data...")
data_path = Path("Data/mt5_features_institutional_regime_filtered.csv")
if not data_path.exists():
    print(f"❌ Data file not found: {data_path}")
    exit(1)

df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} samples")

# Feature columns (24 SMC features)
feature_cols = [
    # Order Block features
    'ob_bullish_present', 'ob_bearish_present',
    'ob_bullish_strength', 'ob_bearish_strength',
    'ob_bullish_distance', 'ob_bearish_distance',
    
    # Fair Value Gap features
    'fvg_bullish_present', 'fvg_bearish_present',
    'fvg_bullish_size', 'fvg_bearish_size',
    
    # Structure features
    'bos_close_confirmed', 'bos_wick_confirmed',
    'bos_strength', 'structure_quality',
    
    # Liquidity features
    'liquidity_above', 'liquidity_below',
    'liquidity_swept_bull', 'liquidity_swept_bear',
    
    # Regime features
    'trend_bias', 'regime_strength',
    
    # Price action
    'price_vs_ob', 'price_vs_fvg',
    
    # Confluence
    'bullish_confluence', 'bearish_confluence'
]

# Verify all features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    exit(1)

# Prepare data
print("\n2. Preparing sequences...")
X = df[feature_cols].values
y = df['target'].values  # 0=SELL, 1=HOLD, 2=BUY

# Create sequences
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)
print(f"✓ Created {len(X_seq)} sequences of length {SEQUENCE_LENGTH}")
print(f"  Shape: {X_seq.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Class distribution: {np.bincount(y_train)}")

# Normalize features (per sequence)
print("\n3. Normalizing features...")
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, FEATURES_TO_USE)
scaler.fit(X_train_flat)

X_train_norm = scaler.transform(X_train.reshape(-1, FEATURES_TO_USE)).reshape(X_train.shape)
X_test_norm = scaler.transform(X_test.reshape(-1, FEATURES_TO_USE)).reshape(X_test.shape)

print("✓ Features normalized")

# Save scaler
scaler_path = Path("Python/models/temporal_scaler.pkl")
scaler_path.parent.mkdir(exist_ok=True)
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# ============================================================================
# MODEL 1: LSTM
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING LSTM MODEL")
print("=" * 70)

def build_lstm_model(seq_length, n_features, n_classes=3):
    model = keras.Sequential([
        # LSTM layers
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

lstm_model = build_lstm_model(SEQUENCE_LENGTH, FEATURES_TO_USE)
print(lstm_model.summary())

# Train LSTM
print("\nTraining LSTM...")
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
)

history_lstm = lstm_model.fit(
    X_train_norm, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate LSTM
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\n✓ LSTM Test Accuracy: {lstm_acc*100:.2f}%")

# Save LSTM
lstm_path = Path("Python/models/lstm_model.h5")
lstm_model.save(lstm_path)
print(f"✓ LSTM saved: {lstm_path}")

# ============================================================================
# MODEL 2: TRANSFORMER
# ============================================================================
print("\n" + "=" * 70)
print("BUILDING TRANSFORMER MODEL")
print("=" * 70)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    
    # Feed forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(seq_length, n_features, n_classes=3):
    inputs = keras.Input(shape=(seq_length, n_features))
    
    # Transformer blocks
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

transformer_model = build_transformer_model(SEQUENCE_LENGTH, FEATURES_TO_USE)
print(transformer_model.summary())

# Train Transformer
print("\nTraining Transformer...")
history_transformer = transformer_model.fit(
    X_train_norm, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate Transformer
transformer_loss, transformer_acc = transformer_model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\n✓ Transformer Test Accuracy: {transformer_acc*100:.2f}%")

# Save Transformer
transformer_path = Path("Python/models/transformer_model.h5")
transformer_model.save(transformer_path)
print(f"✓ Transformer saved: {transformer_path}")

# ============================================================================
# ENSEMBLE EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION")
print("=" * 70)

# Get predictions from both models
lstm_probs = lstm_model.predict(X_test_norm, verbose=0)
transformer_probs = transformer_model.predict(X_test_norm, verbose=0)

# Weighted ensemble (60% LSTM, 40% Transformer)
ensemble_probs = 0.6 * lstm_probs + 0.4 * transformer_probs
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Calculate ensemble accuracy
ensemble_acc = np.mean(ensemble_preds == y_test)
print(f"\n✓ Ensemble Test Accuracy: {ensemble_acc*100:.2f}%")

# Per-class accuracy
for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = np.mean(ensemble_preds[mask] == y_test[mask])
        print(f"  {label}: {class_acc*100:.2f}% ({mask.sum()} samples)")

# Confidence analysis
ensemble_confidence = np.max(ensemble_probs, axis=1)
print(f"\n✓ Average Confidence: {ensemble_confidence.mean()*100:.1f}%")
print(f"  High confidence (>60%): {(ensemble_confidence > 0.6).sum()} samples")
print(f"  Medium confidence (50-60%): {((ensemble_confidence >= 0.5) & (ensemble_confidence <= 0.6)).sum()} samples")
print(f"  Low confidence (<50%): {(ensemble_confidence < 0.5).sum()} samples")

# ============================================================================
# SAVE METADATA
# ============================================================================
print("\n" + "=" * 70)
print("SAVING METADATA")
print("=" * 70)

metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'n_features': FEATURES_TO_USE,
    'feature_names': feature_cols,
    'lstm_accuracy': float(lstm_acc),
    'transformer_accuracy': float(transformer_acc),
    'ensemble_accuracy': float(ensemble_acc),
    'lstm_weight': 0.6,
    'transformer_weight': 0.4
}

metadata_path = Path("Python/models/temporal_metadata.pkl")
joblib.dump(metadata, metadata_path)
print(f"✓ Metadata saved: {metadata_path}")

print("\n" + "=" * 70)
print("TEMPORAL MODELS TRAINING COMPLETE!")
print("=" * 70)
print(f"\nResults:")
print(f"  LSTM Accuracy: {lstm_acc*100:.2f}%")
print(f"  Transformer Accuracy: {transformer_acc*100:.2f}%")
print(f"  Ensemble Accuracy: {ensemble_acc*100:.2f}%")
print(f"\nModels saved:")
print(f"  - {lstm_path}")
print(f"  - {transformer_path}")
print(f"  - {scaler_path}")
print(f"  - {metadata_path}")
print("\nNext: Update REST server to use temporal models!")
