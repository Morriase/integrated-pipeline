# Black Ice AI - Python Components

## Core Files (Production)

### Training Pipeline
- **`integrated_advanced_pipeline.py`** - Main training pipeline with institutional SMC
- **`feature_engineering_smc_institutional.py`** - Institutional-grade SMC feature engineering
- **`production_ensemble_pipeline.py`** - Ensemble model manager
- **`advanced_temporal_architecture.py`** - LSTM/Transformer models
- **`enhanced_multitf_pipeline.py`** - Multi-timeframe utilities

### Inference & API
- **`model_rest_server.py`** - REST API for model inference
- **`start_rest_server.bat`** - Quick start script for REST server

### Evaluation & Export
- **`model_export.py`** - Export models for MT5
- **`temporal_validation.py`** - Temporal cross-validation
- **`learning_curve_plotter.py`** - Learning curve visualization
- **`recovery_mechanism.py`** - Model recovery system
- **`backtest_ensemble.py`** - Backtesting framework

### Dashboard
- **`dashboard.py`** - Streamlit dashboard
- **`launch_dashboard.py`** - Dashboard launcher
- **`requirements_dashboard.txt`** - Dashboard dependencies

## Usage

### 1. Train Models
```bash
python integrated_advanced_pipeline.py
```

This will:
- Load data from `Data/merged_M15_multiTF.csv`
- Apply institutional SMC feature engineering
- Train ensemble models (Random Forest, Gradient Boosting, Neural Networks)
- Export models to MT5 Files folder

### 2. Start REST Server
```bash
python model_rest_server.py
```

Or use the batch file:
```bash
start_rest_server.bat
```

### 3. Launch Dashboard
```bash
streamlit run dashboard.py
```

Or:
```bash
python launch_dashboard.py
```

## Architecture

### Training Flow
```
Data → Institutional SMC → Features → Models → Export
  ↓         ↓                  ↓         ↓        ↓
CSV    OB/FVG/BOS/Regime   25 features  Ensemble  MT5
```

### Inference Flow
```
MT5 EA → REST API → Feature Calc → Models → Prediction
  ↓         ↓            ↓            ↓         ↓
OHLCV    Python      SMC Features  Ensemble  BUY/SELL/HOLD
```

## Key Features

### Institutional SMC
- Order Block detection (1.5 ATR displacement)
- Fair Value Gap detection (quality scoring)
- Break of Structure (wick vs close)
- Regime classification (Trend/Chop/Volatile)
- Triple Barrier Method labeling
- Quality filtering and regime gating

### Models
- Random Forest (deep & wide variants)
- Gradient Boosting (LightGBM)
- Logistic Regression
- Neural Networks (MLP variants)
- LSTM (temporal sequences)
- Transformer (attention mechanism)

### Ensemble
- Weighted voting
- Regime-aware selection
- Confidence thresholding
- Recovery mechanism

## Files Archived

Obsolete files moved to `archive_obsolete/`:
- Old pipelines (black_ice_pipeline.py, regime_aware_pipeline.py)
- Scaffolds (ensemble_modeling_scaffold.py, temporal_architecture_scaffold.py)
- Unused features (volume_orderflow_features.py, advanced_feature_engineering.py)
- One-time scripts (export_mt5_to_csv.py, merge_multitimeframe_data.py)
- Deprecated (file_bridge.py - replaced by REST API)

## Recent Updates

### 2025-10-13: Institutional SMC Integration
- Integrated institutional-grade SMC feature engineering
- Fixed label threshold (0.6% instead of 0.15%)
- Added quality scoring and regime gating
- Implemented proper Triple Barrier Method
- Added OHLCV-based REST API
- Added AI reasoning and SMC context display

### Key Fixes
- ✅ Fixed 88% HOLD bias (now balanced)
- ✅ Improved model accuracy (>60% expected)
- ✅ Added decision transparency
- ✅ Proper SMC detection with quality control
- ✅ Regime-aware filtering

## Documentation

See `docs/` folder for detailed documentation:
- `INSTITUTIONAL_SMC_INTEGRATED.md` - SMC integration details
- `OHLCV_API_APPROACH.md` - REST API architecture
- `AI_DECISION_TRANSPARENCY.md` - Reasoning implementation
- `HOLD_BIAS_PROBLEM.md` - Problem analysis
- `URGENT_FIX_NEEDED.md` - Label generation fix

## Requirements

```
numpy
pandas
scikit-learn
lightgbm
torch
flask
streamlit
plotly
```

Install:
```bash
pip install -r requirements_dashboard.txt
```

## Support

For issues or questions, check the documentation in `docs/` folder.
