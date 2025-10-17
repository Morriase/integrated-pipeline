# SMC Model Training Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: DATA PREPARATION                            │
│                     File: run_complete_pipeline.py                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Raw MT5 OHLC Data             │
                    │   Data/MT5_Data/*.csv           │
                    │   (M15, H1, H4 timeframes)      │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Consolidate Multi-TF Data     │
                    │   consolidate_mt5_data.py       │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   SMC Feature Detection         │
                    │   data_preparation_pipeline.py  │
                    │                                 │
                    │   • Order Blocks (OB)           │
                    │   • Fair Value Gaps (FVG)       │
                    │   • Market Structure (BOS)      │
                    │   • Change of Character (ChoCH) │
                    │   • Fuzzy Logic Quality         │
                    │   • Multi-TF Confluence         │
                    │   • Regime Classification       │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Triple Barrier Labeling       │
                    │   TBM_Label: -1, 0, +1          │
                    │   (LOSS, TIMEOUT, WIN)          │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Train/Val/Test Split          │
                    │   70% / 15% / 15%               │
                    │   (Chronological)               │
                    └─────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │                  OUTPUT: Training Data                  │
        │                                                         │
        │  • processed_smc_data_train.csv  (70%)                 │
        │  • processed_smc_data_val.csv    (15%)                 │
        │  • processed_smc_data_test.csv   (15%)                 │
        │  • processed_smc_data_feature_list.txt                 │
        └─────────────────────────────────────────────────────────┘
                                      │
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 2: MODEL TRAINING                              │
│                      File: train_all_models.py                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Load Training Data            │
                    │   • Extract features (100+)     │
                    │   • Extract target (TBM_Label)  │
                    │   • Handle missing values       │
                    └─────────────────────────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │                  Train 4 Model Types                      │
        └───────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
        │             │               │               │             │
        ▼             ▼               ▼               ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐
│ Random       │ │ XGBoost  │ │ Neural       │ │ LSTM     │
│ Forest       │ │          │ │ Network      │ │          │
│              │ │          │ │              │ │          │
│ • 200 trees  │ │ • 200    │ │ • 256→128→64 │ │ • 20     │
│ • Depth: 20  │ │   rounds │ │ • Dropout    │ │   lookback│
│ • Balanced   │ │ • LR:0.1 │ │   0.3        │ │ • 2 layers│
│              │ │ • GPU    │ │ • BatchNorm  │ │ • 128 dim │
└──────────────┘ └──────────┘ └──────────────┘ └──────────┘
        │             │               │               │
        │             │               │               │
        ▼             ▼               ▼               ▼
┌──────────────────────────────────────────────────────────┐
│              Evaluate on Validation Set                  │
│                                                          │
│  • Accuracy                                              │
│  • Precision / Recall / F1                               │
│  • Win Rate (excl. timeouts)                             │
│  • Confusion Matrix                                      │
└──────────────────────────────────────────────────────────┘
        │             │               │               │
        │             │               │               │
        ▼             ▼               ▼               ▼
┌──────────────────────────────────────────────────────────┐
│              Evaluate on Test Set                        │
│              (Final Performance)                         │
└──────────────────────────────────────────────────────────┘
        │             │               │               │
        │             │               │               │
        ▼             ▼               ▼               ▼
┌──────────────────────────────────────────────────────────┐
│                  Save Trained Models                     │
│                                                          │
│  models/trained/                                         │
│  ├── EURUSD_RandomForest.pkl                            │
│  ├── EURUSD_XGBoost.pkl                                 │
│  ├── EURUSD_NeuralNetwork.pkl                           │
│  ├── EURUSD_LSTM.pkl                                    │
│  ├── *_metadata.json                                    │
│  └── training_results.json                              │
└──────────────────────────────────────────────────────────┘
                                      │
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 3: ENSEMBLE & DEPLOY                           │
│                         (Future Implementation)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Ensemble Predictions          │
                    │   • Weighted average            │
                    │   • Voting classifier           │
                    │   • Stacking                    │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Backtest on Test Set          │
                    │   • Simulate trades             │
                    │   • Calculate metrics           │
                    │   • Risk management             │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Deploy to Production          │
                    │   • Real-time predictions       │
                    │   • Trade execution             │
                    │   • Performance monitoring      │
                    └─────────────────────────────────┘
```

---

## 📊 Data Flow

```
Raw OHLC Data (1000s of candles)
        ↓
100+ SMC Features per candle
        ↓
TBM Labels: -1 (Loss), 0 (Timeout), +1 (Win)
        ↓
Train/Val/Test Split (70/15/15)
        ↓
4 Model Types Learn Patterns
        ↓
Predictions: [P(Loss), P(Timeout), P(Win)]
        ↓
Ensemble → Final Prediction
        ↓
Trade Decision: Enter or Skip
```

---

## 🎯 Feature Flow

```
Order Block Detection
├── OB_Bullish / OB_Bearish
├── OB_Quality_Fuzzy ──────────┐
├── OB_Displacement_ATR ───────┤
├── OB_Age ────────────────────┤
└── OB_Mitigated ──────────────┤
                               │
Fair Value Gap Detection       │
├── FVG_Bullish / FVG_Bearish  │
├── FVG_Quality_Fuzzy ─────────┤
├── FVG_Depth_ATR ─────────────┤
└── FVG_Distance_to_Price ─────┤
                               │
Market Structure               │
├── BOS_Close_Confirm ─────────┤
├── BOS_Commitment_Flag ───────┤
└── ChoCH_Detected ────────────┤
                               │
Multi-Timeframe Confluence     │
├── HTF_OB_Confluence ─────────┤
├── HTF_FVG_Confluence ────────┤
├── HTF_Trend_Alignment ───────┤
└── HTF_Confluence_Quality ────┤
                               │
Regime Features                │
├── Trend_Bias_Indicator ──────┤
├── RSI_Normalized ────────────┤
├── Volatility_Regime_Fuzzy ───┤
└── Trend_Strength_Fuzzy ──────┤
                               │
                               ▼
                    ┌──────────────────┐
                    │  ML Model Input  │
                    │  (100+ features) │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Model Learns:   │
                    │  • OB quality    │
                    │  • FVG strength  │
                    │  • HTF alignment │
                    │  • Regime context│
                    │  • Interactions  │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Prediction:     │
                    │  WIN / LOSS /    │
                    │  TIMEOUT         │
                    └──────────────────┘
```

---

## 🔄 Training Loop

```
For each symbol (EURUSD, GBPUSD, etc.):
    │
    ├─ Load train/val/test data
    │
    ├─ For each model type:
    │   │
    │   ├─ Initialize model
    │   │
    │   ├─ For each epoch:
    │   │   │
    │   │   ├─ Forward pass
    │   │   ├─ Calculate loss
    │   │   ├─ Backward pass (if NN/LSTM)
    │   │   ├─ Update weights
    │   │   └─ Validate
    │   │
    │   ├─ Early stopping check
    │   │
    │   ├─ Evaluate on test set
    │   │
    │   └─ Save model
    │
    └─ Generate summary report
```

---

## 📈 Performance Monitoring

```
Training Phase:
├── Train Loss ↓
├── Train Accuracy ↑
├── Val Loss ↓
└── Val Accuracy ↑

Evaluation Phase:
├── Test Accuracy
├── Precision (WIN)
├── Recall (WIN)
├── F1-Score
├── Win Rate
└── Confusion Matrix

Feature Analysis:
├── Feature Importance
├── Top 20 Features
└── Correlation Analysis
```

---

## 🎯 Decision Flow (Inference)

```
New Candle Arrives
        ↓
Extract 100+ Features
        ↓
┌───────────────────────────────────┐
│  Pass through all 4 models        │
│                                   │
│  Random Forest  → [0.1, 0.3, 0.6] │
│  XGBoost        → [0.2, 0.2, 0.6] │
│  Neural Network → [0.15, 0.25, 0.6]│
│  LSTM           → [0.2, 0.3, 0.5] │
└───────────────────────────────────┘
        ↓
Ensemble (Weighted Average)
        ↓
Final Probabilities: [0.16, 0.26, 0.58]
        ↓
Argmax → Prediction: WIN (+1)
        ↓
Confidence Check: 0.58 > 0.55 threshold?
        ↓
    YES → Enter Trade
    NO  → Skip Setup
```

---

## 🚀 Quick Commands

```bash
# Step 1: Generate training data
python run_complete_pipeline.py

# Step 2: Train all models
python train_all_models.py

# Check results
cat models/trained/training_results.json

# Train specific model
python models/random_forest_model.py
python models/xgboost_model.py
python models/neural_network_model.py
python models/lstm_model.py
```

---

## ✅ Success Checkpoints

```
✓ Step 1 Complete
  └─ Training data generated in Data/

✓ Step 2 Complete
  └─ Models trained and saved in models/trained/

✓ Performance Check
  ├─ Accuracy >50%
  ├─ Win Rate >55%
  └─ No overfitting

✓ Ready for Production
  └─ Deploy best models
```

---

**Next:** See `QUICK_START.md` for step-by-step instructions.
