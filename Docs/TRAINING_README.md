# SMC Model Training Guide

Complete guide for training machine learning models to predict SMC trade outcomes.

## 📋 Overview

This training system implements the specifications from `WHATS_NEEDED.md` to train multiple model types that learn to predict trade outcomes (WIN/LOSS/TIMEOUT) for Smart Money Concepts setups.

## 🎯 What the Models Learn

### 1. **Order Block Quality Assessment**
- High-quality OBs with strong displacement → WIN
- Weak OBs with low displacement → LOSS/TIMEOUT
- Fresh, unmitigated zones → Higher probability

### 2. **Fair Value Gap Significance**
- Large, nearby FVGs → Price reaction
- Small, distant FVGs → Ignore

### 3. **Market Structure Reliability**
- Close breaks with commitment → Trend continuation
- Wick-only breaks → False signals
- ChoCH detection → Potential reversal

### 4. **Multi-Timeframe Confluence**
- HTF alignment → Higher win rate
- Isolated setups → Lower probability

### 5. **Market Regime Adaptation**
- Trending markets → OBs work well
- Ranging markets → FVGs more reliable
- Volatility context → Risk adjustment

## 🚀 Quick Start

### Step 1: Generate Training Data

Run the complete pipeline to process raw MT5 data into ML-ready features:

```bash
python run_complete_pipeline.py
```

This will:
1. Consolidate MT5 data from multiple files
2. Run SMC feature detection (OB, FVG, BOS, ChoCH)
3. Apply fuzzy logic quality scoring
4. Add multi-timeframe confluence
5. Label with Triple Barrier Method
6. Split into train/val/test sets (70/15/15)

**Output files:**
- `Data/processed_smc_data_train.csv` - Training set
- `Data/processed_smc_data_val.csv` - Validation set
- `Data/processed_smc_data_test.csv` - Test set
- `Data/processed_smc_data_feature_list.txt` - Feature documentation

### Step 2: Train All Models

Train all model types for all symbols:

```bash
python train_all_models.py
```

This will train:
- **Random Forest** - Feature importance and threshold rules
- **XGBoost** - Gradient boosting for high accuracy
- **Neural Network** - Non-linear feature interactions
- **LSTM** - Temporal sequence patterns

**Output:**
- Trained models saved in `models/trained/`
- Training results in `models/trained/training_results.json`
- Model metadata and feature importance

## 📊 Model Types

### 1. Random Forest (`random_forest_model.py`)

**Strengths:**
- Identifies most important features
- Learns threshold-based rules
- Captures feature interactions
- Provides interpretable decision paths

**Key Parameters:**
- `n_estimators`: 200 trees
- `max_depth`: 20 levels
- `min_samples_split`: 10 samples
- `class_weight`: 'balanced'

**Expected Learning:**
```python
IF OB_Quality_Fuzzy > 0.8 AND 
   OB_Displacement_ATR > 3.5 AND 
   HTF_Confluence_Quality > 0.7
THEN Probability(WIN) = HIGH
```

### 2. XGBoost (`xgboost_model.py`)

**Strengths:**
- Gradient-based pattern learning
- Built-in regularization
- Handles imbalanced classes
- Fast training with GPU support

**Key Parameters:**
- `n_estimators`: 200 rounds
- `max_depth`: 6 levels
- `learning_rate`: 0.1
- `early_stopping_rounds`: 20

**Expected Learning:**
- Residual patterns missed by other models
- Optimal feature combinations
- Gradient-based decision boundaries

### 3. Neural Network (`neural_network_model.py`)

**Strengths:**
- Learns complex non-linear relationships
- Captures multiplicative effects
- Flexible architecture
- GPU acceleration

**Architecture:**
- Input → 256 → 128 → 64 → 3 classes
- Batch normalization
- Dropout (0.3)
- ReLU activation

**Expected Learning:**
```python
IF (OB_Quality_Fuzzy * HTF_Confluence_Quality > 0.6) AND
   (Trend_Bias > 2.0 AND RSI < 50) AND
   (Volatility_Regime = 'High_Vol_Trend')
THEN Complex high-probability setup → WIN
```

### 4. LSTM (`lstm_model.py`)

**Strengths:**
- Captures temporal dependencies
- Learns sequence patterns
- Models momentum shifts
- Remembers context across time

**Architecture:**
- Lookback window: 20 candles
- LSTM layers: 2 (128 hidden units)
- Dropout: 0.3

**Expected Learning:**
```
Winning Sequence:
t-10: ChoCH_Detected = 1 (reversal)
t-5:  OB_Bullish = 1 (order block forms)
t-3:  OB_Displacement_ATR = 4.5 (strong move)
t-1:  HTF_Confluence_Quality increases
t-0:  Entry → Predict WIN
```

## 📈 Performance Requirements

### Minimum Acceptable Performance

**Classification Metrics:**
- Overall Accuracy: >50% (baseline: 33% random)
- Precision (WIN class): >55%
- Recall (WIN class): >50%
- F1-Score (WIN class): >52%
- Macro F1-Score: >45%

**Trading Metrics:**
- Win Rate (excl. timeouts): >55%
- Risk-Reward Ratio: 1:3
- Sharpe Ratio: >1.5
- Maximum Drawdown: <20%
- Profit Factor: >1.8

## 🔧 Training Individual Models

### Train Random Forest Only

```python
from models.random_forest_model import RandomForestSMCModel

model = RandomForestSMCModel(symbol='EURUSD')

# Load data
train_df, val_df, test_df = model.load_data(
    'Data/processed_smc_data_train.csv',
    'Data/processed_smc_data_val.csv',
    'Data/processed_smc_data_test.csv'
)

# Prepare features
X_train, y_train = model.prepare_features(train_df)
X_val, y_val = model.prepare_features(val_df)
X_test, y_test = model.prepare_features(test_df)

# Train
model.train(X_train, y_train, X_val, y_val)

# Evaluate
test_metrics = model.evaluate(X_test, y_test)

# Save
model.save_model('models/trained')
```

### Train XGBoost with GPU

```python
from models.xgboost_model import XGBoostSMCModel

model = XGBoostSMCModel(symbol='EURUSD')

# ... load and prepare data ...

# Train with GPU
model.train(
    X_train, y_train, X_val, y_val,
    use_gpu=True,
    n_estimators=300,
    learning_rate=0.05
)
```

### Train Neural Network

```python
from models.neural_network_model import NeuralNetworkSMCModel

model = NeuralNetworkSMCModel(symbol='EURUSD')

# ... load and prepare data ...

# Train with custom architecture
model.train(
    X_train, y_train, X_val, y_val,
    hidden_dims=[512, 256, 128, 64],
    dropout=0.4,
    learning_rate=0.0005,
    epochs=150
)
```

### Train LSTM

```python
from models.lstm_model import LSTMSMCModel

model = LSTMSMCModel(symbol='EURUSD', lookback=30)

# ... load and prepare data ...

# Train with longer sequences
model.train(
    X_train, y_train, X_val, y_val,
    hidden_dim=256,
    num_layers=3,
    epochs=150
)
```

## 📊 Feature Importance

After training, check which features the models find most important:

```python
# Get top 20 features
top_features = model.get_feature_importance(top_n=20)
print(top_features)
```

**Expected Top Features:**
1. `OB_Quality_Fuzzy` - Overall OB quality
2. `HTF_Confluence_Quality` - Multi-TF alignment
3. `OB_Displacement_ATR` - Institutional strength
4. `Trend_Bias_Indicator` - Directional bias
5. `FVG_Quality_Fuzzy` - Gap significance
6. `BOS_Commitment_Flag` - Structure conviction
7. `Volatility_Regime_Fuzzy` - Market state
8. `HTF_Trend_Alignment` - HTF confirmation
9. `RSI_Normalized` - Momentum
10. `OB_Age` - Setup freshness

## 🎯 Model Evaluation

Each model is evaluated on:

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: When predicting WIN, how often correct?
- **Recall**: What % of actual WINs are caught?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

### Trading Metrics
- **Win Rate**: % of trades that hit TP (excluding timeouts)
- **Risk-Reward**: Maintained at 1:3
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss

## 🔄 Retraining Models

Models should be retrained periodically as market conditions change:

```bash
# Retrain all models
python train_all_models.py

# Or retrain specific model
python models/random_forest_model.py
```

## 📁 File Structure

```
.
├── run_complete_pipeline.py          # Step 1: Generate training data
├── train_all_models.py               # Step 2: Train all models
├── models/
│   ├── base_model.py                 # Base class for all models
│   ├── random_forest_model.py        # Random Forest implementation
│   ├── xgboost_model.py              # XGBoost implementation
│   ├── neural_network_model.py       # Neural Network implementation
│   ├── lstm_model.py                 # LSTM implementation
│   └── trained/                      # Saved models
│       ├── EURUSD_RandomForest.pkl
│       ├── EURUSD_XGBoost.pkl
│       ├── EURUSD_NeuralNetwork.pkl
│       ├── EURUSD_LSTM.pkl
│       └── training_results.json
├── Data/
│   ├── processed_smc_data_train.csv
│   ├── processed_smc_data_val.csv
│   ├── processed_smc_data_test.csv
│   └── processed_smc_data_feature_list.txt
└── WHATS_NEEDED.md                   # Model specifications
```

## 🐛 Troubleshooting

### "Data files not found"
Run `python run_complete_pipeline.py` first to generate training data.

### "XGBoost not available"
Install with: `pip install xgboost`

### "PyTorch not available"
Install with: `pip install torch`

### "CUDA out of memory"
Reduce batch size or use CPU:
```python
model.train(..., batch_size=32, use_gpu=False)
```

### "Model not converging"
- Increase epochs
- Adjust learning rate
- Check for data quality issues
- Ensure sufficient training samples

## 📚 Dependencies

```bash
pip install pandas numpy scikit-learn xgboost torch
```

## ✅ Success Criteria

Your models are successful when:

✅ Accuracy >55% on test set (excluding timeouts)  
✅ Precision (WIN class) >60%  
✅ Recall (WIN class) >50%  
✅ Win rate >55% (excluding timeouts)  
✅ Performance consistent across symbols  
✅ No overfitting (train-val gap <5%)  
✅ Feature importance makes sense  
✅ Positive expectancy in backtesting  

## 🚀 Next Steps

1. ✅ Generate training data: `python run_complete_pipeline.py`
2. ✅ Train models: `python train_all_models.py`
3. ⏭️ Ensemble predictions (combine all models)
4. ⏭️ Backtest on test set
5. ⏭️ Deploy best models for live trading

---

**Questions?** Check `WHATS_NEEDED.md` for detailed specifications.
