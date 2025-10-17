# SMC Model Training - Quick Start

## 🎯 Goal
Train ML models to predict trade outcomes (WIN/LOSS/TIMEOUT) for SMC setups with >55% win rate.

## 📝 Two-Step Process

### Step 1: Generate Training Data ⚙️

**File to run:** `run_complete_pipeline.py`

```bash
python run_complete_pipeline.py
```

**What it does:**
- Consolidates MT5 data
- Detects SMC structures (OB, FVG, BOS, ChoCH)
- Applies fuzzy logic quality scoring
- Adds multi-timeframe confluence
- Labels trades with Triple Barrier Method
- Splits into train/val/test (70/15/15)

**Output:**
```
Data/
├── processed_smc_data_train.csv  ← Training data
├── processed_smc_data_val.csv    ← Validation data
├── processed_smc_data_test.csv   ← Test data
└── processed_smc_data_feature_list.txt
```

**Time:** ~5-30 minutes depending on data size

---

### Step 2: Train Models 🚀

**File to run:** `train_all_models.py`

```bash
python train_all_models.py
```

**What it does:**
- Trains 4 model types per symbol:
  - Random Forest (feature importance)
  - XGBoost (gradient boosting)
  - Neural Network (non-linear patterns)
  - LSTM (temporal sequences)
- Evaluates on validation and test sets
- Saves trained models

**Output:**
```
models/trained/
├── EURUSD_RandomForest.pkl
├── EURUSD_XGBoost.pkl
├── EURUSD_NeuralNetwork.pkl
├── EURUSD_LSTM.pkl
├── training_results.json  ← Performance metrics
└── *_metadata.json
```

**Time:** ~10-60 minutes depending on data size and hardware

---

## 📊 Check Results

After training, check performance:

```bash
cat models/trained/training_results.json
```

Look for:
- **Accuracy** >50% (baseline: 33%)
- **Win Rate** >55% (excluding timeouts)
- **Precision (WIN)** >55%
- **F1-Score** >52%

---

## 🔧 Requirements

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost torch
```

### Data Requirements

- Raw MT5 OHLC data in `Data/MT5_Data/`
- Multiple timeframes (M15, H1, H4)
- At least 1000+ candles per symbol

---

## 🎓 What the Models Learn

### Order Block Quality
```
High Quality OB:
- OB_Quality_Fuzzy > 0.7
- OB_Displacement_ATR > 3.0
- OB_Mitigated = 0
- OB_Age < 10
→ Predict WIN
```

### Multi-Timeframe Confluence
```
Strong Confluence:
- HTF_OB_Confluence >= 2
- HTF_Trend_Alignment = 1
- HTF_Confluence_Quality > 0.7
→ Higher probability WIN
```

### Market Regime
```
Trending Market:
- Volatility_Regime = 'High_Vol_Trend'
- Trend_Strength_Fuzzy > 2.0
- OB_Bullish_Valid = 1
→ OB works well → WIN
```

---

## 🐛 Common Issues

### "Data files not found"
**Solution:** Run Step 1 first: `python run_complete_pipeline.py`

### "XGBoost not installed"
**Solution:** `pip install xgboost`

### "PyTorch not installed"
**Solution:** `pip install torch`

### "Insufficient data"
**Solution:** Ensure you have at least 1000+ candles per symbol in `Data/MT5_Data/`

---

## 📈 Expected Performance

| Metric | Target | Baseline |
|--------|--------|----------|
| Accuracy | >50% | 33% (random) |
| Win Rate | >55% | 50% (coin flip) |
| Precision (WIN) | >55% | - |
| F1-Score | >52% | - |

---

## 🚀 Full Workflow

```bash
# 1. Generate training data
python run_complete_pipeline.py

# 2. Train all models
python train_all_models.py

# 3. Check results
cat models/trained/training_results.json

# 4. (Optional) Train specific model
python models/random_forest_model.py
```

---

## 📚 More Information

- **Detailed guide:** `TRAINING_README.md`
- **Model specs:** `WHATS_NEEDED.md`
- **Pipeline details:** `DATA_PIPELINE_README.md`

---

## ✅ Success Checklist

- [ ] Step 1 complete: Training data generated
- [ ] Step 2 complete: Models trained
- [ ] Accuracy >50% on test set
- [ ] Win rate >55% (excluding timeouts)
- [ ] Models saved in `models/trained/`
- [ ] Results look reasonable (no overfitting)

---

**Ready to start?** Run: `python run_complete_pipeline.py`
