"""
LSTM Model for SMC Trade Outcome Prediction

Key Learning Objectives (from WHATS_NEEDED.md):
- Learn short-to-medium term patterns (10-50 candles)
- Capture momentum shifts
- Model state transitions (regime changes)
- Remember context across sequences
- Recognize setup formation patterns
"""

import numpy as np
from typing import Dict, Optional, List
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler
from models.base_model import BaseSMCModel


class LSTMClassifier(nn.Module):
    """LSTM network for sequence classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 3, dropout: float = 0.3, bidirectional: bool = True):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # BiLSTM with variational dropout (same mask across time steps)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Variational dropout
            bidirectional=bidirectional
        )
        
        # Adjust FC layer for bidirectional output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # For bidirectional, concatenate forward and backward final hidden states
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            # Get last layer's forward and backward hidden states
            forward_hidden = h_n[-2]  # Last layer forward
            backward_hidden = h_n[-1]  # Last layer backward
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            final_hidden = h_n[-1]
        
        out = self.fc(final_hidden)
        
        return out


class LSTMSMCModel(BaseSMCModel):
    """
    LSTM classifier for SMC trade prediction
    
    Strengths:
    - Captures temporal dependencies
    - Learns sequence patterns
    - Models momentum shifts
    - Remembers context across time
    """
    
    def __init__(self, symbol: str, target_col: str = 'TBM_Label', lookback: int = 20):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        super().__init__('LSTM', symbol, target_col)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = lookback
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            
        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape (n_samples, lookback, n_features)
        """
        X_seq, y_seq = [], []
        
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              hidden_dim: int = 256,  # Larger hidden dimension
              num_layers: int = 3,  # Deeper stacked LSTM
              dropout: float = 0.4,  # Variational dropout
              learning_rate: float = 0.01,  # Higher max LR for One-Cycle
              batch_size: int = 16,  # Smaller batch for sequences
              epochs: int = 200,  # More epochs for full One-Cycle
              patience: int = 30,  # More patience
              weight_decay: float = 0.01,  # AdamW weight decay
              bidirectional: bool = True,  # Use BiLSTM
              **kwargs) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"\n🔄 Training LSTM for {self.symbol}...")
        print(f"  Device: {self.device}")
        print(f"  Lookback window: {self.lookback} candles")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        print(f"  Creating sequences...")
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
        print(f"  Sequence shape: {X_train_seq.shape}")
        
        # Convert labels to 0, 1, 2
        label_map = {-1: 0, 0: 1, 1: 2}
        self.label_map_reverse = {0: -1, 1: 0, 2: 1}  # Store for predictions
        y_train_mapped = np.array([label_map[y] for y in y_train_seq])
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.LongTensor(y_train_mapped)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model with BiLSTM
        self.model = LSTMClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=3,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Apply He (Kaiming) initialization for ReLU layers in FC
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
        
        self.model.apply(init_weights)
        
        # Loss and optimizer (AdamW with decoupled weight decay)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # One-Cycle Learning Rate Policy
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Store reverse label map
        self.label_map_reverse = {0: -1, 1: 0, 2: 1}
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_model_state = None  # Initialize best model state
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                # Gradient clipping to prevent NaN (critical for RNNs)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Step after each batch (One-Cycle policy)
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
                y_val_mapped = np.array([label_map[y] for y in y_val_seq])
                
                self.model.eval()
                with torch.no_grad():
                    val_X = torch.FloatTensor(X_val_seq).to(self.device)
                    val_y = torch.LongTensor(y_val_mapped).to(self.device)
                    
                    val_outputs = self.model(val_X)
                    val_loss = criterion(val_outputs, val_y).item()
                    
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == val_y).sum().item() / len(val_y)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping (no scheduler.step here - One-Cycle handles it)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
                
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    # Restore best model if it was saved
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        
        self.training_history = history
        self.is_trained = True
        
        print(f"\n  Final Train Accuracy: {history['train_acc'][-1]:.3f}")
        if history['val_acc']:
            print(f"  Final Val Accuracy:   {history['val_acc'][-1]:.3f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X)))
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_mapped = predicted.cpu().numpy()
        
        # Convert back to original labels
        y_pred = np.array([self.label_map_reverse[y] for y in y_pred_mapped])
        
        # Pad with zeros for first lookback samples
        y_pred_full = np.zeros(len(X))
        y_pred_full[self.lookback:] = y_pred
        
        return y_pred_full
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability distributions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X)))
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Pad with uniform distribution for first lookback samples
        proba_full = np.zeros((len(X), 3))
        proba_full[:self.lookback] = 1/3  # Uniform distribution
        proba_full[self.lookback:] = proba
        
        return proba_full


# Example usage
if __name__ == "__main__":
    """
    Example: Train LSTM for EURUSD
    """
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
        exit(1)
    
    # Initialize model
    model = LSTMSMCModel(symbol='EURUSD', lookback=20)
    
    # Load data
    train_df, val_df, test_df = model.load_data(
        train_path='Data/processed_smc_data_train.csv',
        val_path='Data/processed_smc_data_val.csv',
        test_path='Data/processed_smc_data_test.csv',
        exclude_timeout=False
    )
    
    # Prepare features
    X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
    X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
    X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=15
    )
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')
    
    # Save model
    model.save_model('models/trained')
    
    print("\n✅ LSTM training complete!")
