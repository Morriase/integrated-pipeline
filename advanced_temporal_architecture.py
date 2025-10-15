"""
Advanced Temporal Architecture Pipeline for SMC Analysis
- LSTM/GRU for temporal sequence modeling
- Transformer architecture for attention-based pattern recognition
- Regime-aware ensemble models
- Production-ready integration with multi-timeframe pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math
from pathlib import Path
import json
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer architecture"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SMC_LSTM(nn.Module):
    """
    LSTM-based architecture for temporal SMC pattern recognition
    Captures sequential dependencies in market structure
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention mechanism for sequence weighting
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights using Xavier initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Normalize input features
        x_norm = x.view(-1, features)
        x_norm = self.input_norm(x_norm)
        x_norm = x_norm.view(batch_size, seq_len, features)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_norm)

        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        # Classification
        logits = self.classifier(attended_output)

        return logits, attention_weights


class SMC_Transformer(nn.Module):
    """
    Transformer-based architecture for SMC pattern recognition
    Uses self-attention to capture complex temporal relationships
    """

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Global pooling and classification
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled)

        return logits


class RegimeClassifier(nn.Module):
    """
    Regime classification network to identify market conditions
    Used for regime-aware ensemble selection
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_regimes: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_regimes)
        )

    def forward(self, x):
        return self.network(x)


class RegimeAwareEnsemble(nn.Module):
    """
    Regime-aware ensemble that combines multiple models based on detected market regime
    """

    def __init__(self, models: Dict[str, nn.Module], regime_classifier: RegimeClassifier):
        super().__init__()

        self.models = nn.ModuleDict(models)
        self.regime_classifier = regime_classifier
        self.model_names = list(models.keys())

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(models)) / len(models))

    def forward(self, x, regime_features=None):
        batch_size = x.shape[0]

        # Get predictions from all models
        model_outputs = []
        for name, model in self.models.items():
            if hasattr(model, 'forward') and len(x.shape) == 3:
                # Temporal models (LSTM, Transformer)
                output, *_ = model(x) if isinstance(model,
                                                    SMC_LSTM) else (model(x),)
            else:
                # Non-temporal models - use last timestep
                if len(x.shape) == 3:
                    x_flat = x[:, -1, :]  # Use last timestep
                else:
                    x_flat = x
                output = model(x_flat)

            model_outputs.append(output)

        # Stack model outputs
        # (batch, num_models, num_classes)
        stacked_outputs = torch.stack(model_outputs, dim=1)

        # Regime-based weighting (if regime features provided)
        if regime_features is not None:
            regime_logits = self.regime_classifier(regime_features)
            regime_probs = F.softmax(regime_logits, dim=1)

            # Map regime probabilities to model weights (simplified)
            # In practice, this would be learned or rule-based
            dynamic_weights = F.softmax(
                self.ensemble_weights.unsqueeze(0).expand(batch_size, -1), dim=1)
        else:
            # Use static ensemble weights
            dynamic_weights = F.softmax(
                self.ensemble_weights.unsqueeze(0).expand(batch_size, -1), dim=1)

        # Weighted ensemble prediction
        ensemble_output = torch.sum(
            stacked_outputs * dynamic_weights.unsqueeze(-1), dim=1)

        return ensemble_output, dynamic_weights


class TemporalDataProcessor:
    """
    Processes multi-timeframe data into temporal sequences for LSTM/Transformer training
    """

    def __init__(self, sequence_length: int = 20, prediction_horizon: int = 8):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def create_sequences(self, features: np.ndarray, labels: np.ndarray,
                         symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create temporal sequences from multi-timeframe data

        Args:
            features: (N, feature_dim) array
            labels: (N,) array  
            symbols: List of symbol names for each sample

        Returns:
            sequences: (N_seq, seq_len, feature_dim) array
            sequence_labels: (N_seq,) array
            sequence_symbols: List of symbols for each sequence
        """

        sequences = []
        sequence_labels = []
        sequence_symbols = []

        # Process each symbol separately to maintain temporal integrity
        unique_symbols = list(set(symbols))

        for symbol in unique_symbols:
            # Get indices for this symbol
            symbol_indices = [i for i, s in enumerate(symbols) if s == symbol]

            if len(symbol_indices) < self.sequence_length + self.prediction_horizon:
                continue

            symbol_features = features[symbol_indices]
            symbol_labels = labels[symbol_indices]

            # Create sequences for this symbol
            for i in range(len(symbol_features) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                seq_features = symbol_features[i:i + self.sequence_length]

                # Label at prediction horizon
                label_idx = i + self.sequence_length + self.prediction_horizon - 1
                seq_label = symbol_labels[label_idx]

                sequences.append(seq_features)
                sequence_labels.append(seq_label)
                sequence_symbols.append(symbol)

        return (np.array(sequences), np.array(sequence_labels), sequence_symbols)

    def create_regime_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract regime-relevant features from sequences for regime classification

        Args:
            sequences: (N, seq_len, feature_dim) array

        Returns:
            regime_features: (N, regime_feature_dim) array
        """

        # Extract statistical features over the sequence
        regime_features = []

        for seq in sequences:
            # Price-based regime features
            returns = seq[:, 0]  # Assuming first feature is returns
            volatility = np.std(returns)
            trend_strength = np.abs(np.mean(returns)) / (volatility + 1e-8)

            # Structure-based regime features
            # Last 10 features assumed to be structure
            structure_breaks = np.sum(seq[:, -10:], axis=0)

            # Combine regime features
            regime_feat = np.concatenate([
                [volatility, trend_strength],
                structure_breaks
            ])

            regime_features.append(regime_feat)

        return np.array(regime_features)


def train_temporal_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                         epochs: int = 50, lr: float = 1e-3, device: str = 'cpu') -> Dict:
    """
    Train temporal model with advanced techniques
    """

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            if isinstance(model, SMC_LSTM):
                outputs, _ = model(batch_features)
            else:
                outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                if isinstance(model, SMC_LSTM):
                    outputs, _ = model(batch_features)
                else:
                    outputs = model(batch_features)

                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc
    }


def main_temporal_pipeline():
    """
    Main pipeline for temporal architecture training and evaluation
    """
    print("=== Advanced Temporal Architecture Pipeline ===")

    # Load multi-timeframe data
    from enhanced_multitf_pipeline import load_and_engineer_multitf_features, standardize_features

    features, labels, symbols = load_and_engineer_multitf_features(
        r'C:\Users\Morris\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5\Experts\Black_Ice_Protocol\Data\merged_M15_multiTF.csv')
    features_norm, _, _ = standardize_features(features)

    # Create temporal sequences
    processor = TemporalDataProcessor(sequence_length=20, prediction_horizon=8)
    sequences, seq_labels, seq_symbols = processor.create_sequences(
        features_norm, labels, symbols)

    print(f"Created {len(sequences)} temporal sequences")
    print(f"Sequence shape: {sequences.shape}")

    # Split data
    n = len(sequences)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_sequences = sequences[:train_size]
    train_labels = seq_labels[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    val_labels = seq_labels[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    test_labels = seq_labels[train_size + val_size:]

    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(
        torch.tensor(train_sequences, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_sequences, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_model = SMC_LSTM(
        input_dim=sequences.shape[2],
        hidden_dim=128,
        num_layers=2,
        num_classes=len(np.unique(labels)),
        dropout=0.3
    )

    lstm_results = train_temporal_model(
        lstm_model, train_loader, val_loader, epochs=30)
    print(
        f"LSTM best validation accuracy: {lstm_results['best_val_accuracy']:.4f}")

    # Train Transformer model
    print("\nTraining Transformer model...")
    transformer_model = SMC_Transformer(
        input_dim=sequences.shape[2],
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=len(np.unique(labels))
    )

    transformer_results = train_temporal_model(
        transformer_model, train_loader, val_loader, epochs=30)
    print(
        f"Transformer best validation accuracy: {transformer_results['best_val_accuracy']:.4f}")

    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LSTM evaluation
    lstm_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(
            test_sequences, dtype=torch.float32, device=device)
        lstm_logits, _ = lstm_model(test_tensor)
        lstm_preds = lstm_logits.argmax(dim=1).cpu().numpy()

    lstm_test_acc = (lstm_preds == test_labels).mean()

    # Transformer evaluation
    transformer_model.eval()
    with torch.no_grad():
        transformer_logits = transformer_model(test_tensor)
        transformer_preds = transformer_logits.argmax(dim=1).cpu().numpy()

    transformer_test_acc = (transformer_preds == test_labels).mean()

    print(f"\n=== Test Set Results ===")
    print(f"LSTM Test Accuracy: {lstm_test_acc:.4f}")
    print(f"Transformer Test Accuracy: {transformer_test_acc:.4f}")

    return {
        'lstm': {'model': lstm_model, 'test_accuracy': lstm_test_acc},
        'transformer': {'model': transformer_model, 'test_accuracy': transformer_test_acc}
    }


if __name__ == "__main__":
    results = main_temporal_pipeline()
