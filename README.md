# SMC Trading Model Pipeline

Production-ready machine learning pipeline for Smart Money Concepts (SMC) trade prediction.

## 🎯 Quick Start

### Kaggle Execution
```bash
# Clone repository
git clone https://github.com/Morriase/integrated-pipeline.git
cd integrated-pipeline

# Install dependencies
pip install --upgrade scikit-learn imbalanced-learn

# Run complete pipeline
python run_complete_pipeline.py  # Data preparation
python train_all_models.py       # Train models
python create_training_visualizations.py  # Generate charts
python test_consensus_ensemble.py  # Test ensemble
```

### Local Execution
Same commands work locally. Data paths auto-detect environment.

---

## 📁 Project Structure

```
├── consolidate_mt5_data.py          # MT5 data consolidation
├── data_preparation_pipeline.py     # SMC feature engineering
├── run_complete_pipeline.py         # Complete data pipeline
├── train_all_models.py              # Train all models
├── test_consensus_ensemble.py       # Test ensemble voting
├── create_training_visualizations.py # Generate charts
├── run_kaggle_pipeline.py           # All-in-one Kaggle script
│
├── models/
│   ├── base_model.py                # Base model class
│   ├── random_forest_model.py       # RandomForest
│   ├── xgboost_model.py             # XGBoost
│   ├── neural_network_model.py      # Neural Network (BEST)
│   ├── lstm_model.py                # LSTM (experimental)
│   ├── ensemble_model.py            # Consensus ensemble
│   ├── config.py                    # Model configurations
│   ├── overfitting_monitor.py       # Training monitor
│   └── data_augmentation.py         # Data augmentation
│
├── Docs/
│   ├── DATA_FLOW_ANALYSIS.md        # Complete pipeline analysis
│   ├── GAPS_SUMMARY.md              # Known issues & fixes
│   └── training_progress.txt        # Latest training results
│
└── resources/                        # Training logs & summaries
```

---

## 🚀 Model Performance

| Model | Test Acc | Win Rate | Profit Factor | EV/Trade | Status |
|-------|----------|----------|---------------|----------|--------|
| **Neural Network** | 64.8% | 57.2% | 2.67 | **0.72R** | ⭐ BEST |
| RandomForest | 61.5% | 50.6% | 2.05 | 0.52R | ✅ Stable |
| XGBoost | 72.2% | 42.8% | 1.50 | 0.28R | ✅ Accurate |
| LSTM | 46.4% | 47.6% | 1.82 | 0.43R | ⚠️ Unstable |

**Recommendation:** Deploy Neural Network as primary model.

---

## 📊 Training Data

- **Total Samples:** 110,000 candles
- **Labeled Setups:** 2,630 (3.4% - high-quality SMC setups only)
- **Symbols:** 11 (EURUSD, GBPUSD, USDJPY, etc.)
- **Timeframes:** 7 (M1, M5, M15, M30, H1, H4, D1)
- **Features:** 100+ (ATR-normalized, fuzzy logic)
- **Split:** 70% train, 15% val, 15% test

---

## 🎯 Production Deployment

### Recommended Strategy: Hybrid Ensemble

```python
# Primary: Neural Network (best EV)
nn_pred = neural_network.predict(X)

# Validators: RF + XGB
rf_pred = random_forest.predict(X)
xgb_pred = xgboost.predict(X)

# Scale position by agreement
if nn_pred == rf_pred == xgb_pred:
    position_size = 1.0%  # High confidence
elif (nn_pred == rf_pred) or (nn_pred == xgb_pred):
    position_size = 0.75%  # Medium confidence
else:
    position_size = 0.5%  # Low confidence (NN only)
```

### Expected Performance
```
Win Rate: 57.2%
Expected Value: 0.72R per trade
Profit Factor: 2.67

100 trades @ 1% risk = +72% account growth
```

---

## 📈 Visualizations

Generated in `/kaggle/working/Training_Images/`:
- `model_comparison.png` - Performance comparison
- `confusion_matrices.png` - Prediction matrices
- `training_curves_comparison.png` - Loss & accuracy curves
- `overfitting_analysis.png` - Train-val gaps
- `training_summary.png` - Overall metrics

---

## 🔧 Configuration

### Kaggle Paths (Auto-Detected)
```
Input:  /kaggle/input/ob-ai-model-2-dataset/Data/mt5_exports
Data:   /kaggle/working/Data-output
Models: /kaggle/working/Model-output
Images: /kaggle/working/Training_Images
```

### Local Paths (Fallback)
```
Input:  Data/mt5_exports
Data:   Data
Models: models/trained
Images: Training_Images
```

---

## 📚 Documentation

- **`KAGGLE_QUICK_START.md`** - Quick start guide
- **`Docs/DATA_FLOW_ANALYSIS.md`** - Complete pipeline breakdown
- **`Docs/GAPS_SUMMARY.md`** - Known issues & solutions
- **`models/CONFIG_DOCUMENTATION.md`** - Model configurations

---

## ⚙️ Key Features

### Data Pipeline
- ✅ Multi-symbol consolidation
- ✅ SMC feature extraction (OB, FVG, BOS, ChoCH)
- ✅ Fuzzy logic for adaptive thresholds
- ✅ ATR normalization for cross-symbol compatibility
- ✅ Triple Barrier Method labeling

### Model Training
- ✅ Unified dataset approach (all symbols)
- ✅ Feature selection (100+ → 30-50 features)
- ✅ Class weight balancing
- ✅ Cross-validation
- ✅ Early stopping
- ✅ Overfitting monitoring
- ✅ Trading metrics (win rate, profit factor, EV)

### Ensemble
- ✅ Consensus voting (strict/majority)
- ✅ Confidence filtering
- ✅ Hybrid strategies

---

## 🛠️ Requirements

```
Python 3.8+
pandas
numpy
scikit-learn
xgboost
torch (for NN/LSTM)
matplotlib
seaborn
imbalanced-learn
```

---

## 📝 License

MIT License - See LICENSE file

---

## 🎉 Status

**Production Ready** - All models trained, tested, and validated.

Neural Network achieves **0.72R expected value per trade** with **57.2% win rate**.

Ready for paper trading and live deployment.
