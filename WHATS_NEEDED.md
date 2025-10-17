# What the Models Need to Do - Comprehensive Specification

## 🎯 Executive Summary

The models must learn to **predict trade outcomes** for Smart Money Concepts (SMC) trading setups by analyzing institutional order flow patterns, market structure, and multi-timeframe confluence. This is a **3-class classification problem** predicting WIN (+1), LOSS (-1), or TIMEOUT (0) for each trading opportunity.

---

## 📊 Primary Objective

### **Task: Trade Outcome Prediction**

**Input:** 100+ SMC features from `data_preparation_pipeline.py`  
**Output:** One of 3 classes for each setup:

```
Class -1 (LOSS):    Trade hits stop loss before take profit
Class  0 (TIMEOUT): Neither barrier hit within 20 candles
Class +1 (WIN):     Trade hits take profit before stop loss (1:3 R:R)
```

**Success Metric:** Achieve >55% win rate (excluding timeouts) with 1:3 risk-reward ratio

---

## 🧠 What Models Must Learn

### 1. **Order Block Quality Assessment**

**Objective:** Identify which Order Blocks lead to successful trades

**Key Patterns to Learn:**

- **High-Quality OBs → WIN**
  - `OB_Quality_Fuzzy > 0.7` (strong fuzzy score)
  - `OB_Displacement_ATR > 3.0` (significant institutional move)
  - `OB_Mitigated = 0` (zone still valid)
  - `OB_Age < 10` (fresh institutional interest)
  - `OB_Body_Fuzzy_Score > 0.6` (substantial candle body)

- **Low-Quality OBs → LOSS/TIMEOUT**
  - `OB_Quality_Fuzzy < 0.4` (weak setup)
  - `OB_Displacement_ATR < 1.5` (insufficient move)
  - `OB_Mitigated = 1` (already invalidated)
  - `OB_Age > 20` (stale zone)

**Expected Learning:**
```
IF OB_Quality_Fuzzy > 0.8 AND 
   OB_Displacement_ATR > 3.5 AND 
   OB_Mitigated = 0 AND
   OB_Age < 5
THEN Probability(WIN) = HIGH
```

---

### 2. **Fair Value Gap Significance**

**Objective:** Predict which FVGs will be respected vs. filled

**Key Patterns to Learn:**

- **Significant FVGs → Price Reaction**
  - `FVG_Quality_Fuzzy > 0.7` (institutional imbalance)
  - `FVG_Depth_ATR > 2.0` (large gap)
  - `FVG_Distance_to_Price_ATR < 1.0` (nearby)
  - `FVG_Mitigated = 0` (unfilled)
  - `FVG_Size_Fuzzy_Score > 0.6` (substantial gap)

- **Weak FVGs → Ignore**
  - `FVG_Quality_Fuzzy < 0.4`
  - `FVG_Depth_ATR < 0.5` (too small)
  - `FVG_Distance_to_Price_ATR > 3.0` (too far)
  - `FVG_Mitigated = 1` (already filled)

**Expected Learning:**
```
IF FVG_Quality_Fuzzy > 0.7 AND
   FVG_Depth_ATR > 2.5 AND
   FVG_Distance_to_Price_ATR < 0.5
THEN Price likely to react → Setup valid
```

---

### 3. **Market Structure Break Reliability**

**Objective:** Distinguish real breaks from liquidity sweeps

**Key Patterns to Learn:**

- **Reliable Structure Breaks → Trend Continuation**
  - `BOS_Close_Confirm = 1` (candle closed past level)
  - `BOS_Commitment_Flag = 1` (high conviction)
  - `BOS_Dist_ATR > 2.0` (significant break)
  - `BOS_Wick_Confirm = 1` (wick pushed through)

- **False Breaks → Reversal/Chop**
  - `BOS_Close_Confirm = 0` (only wick, no close)
  - `BOS_Commitment_Flag = 0` (weak conviction)
  - `BOS_Dist_ATR < 0.5` (marginal break)

- **Change of Character → Reversal**
  - `ChoCH_Detected = 1` (trend change signal)
  - `ChoCH_Direction = 1 or -1` (new direction)

**Expected Learning:**
```
IF BOS_Close_Confirm = 1 AND
   BOS_Commitment_Flag = 1 AND
   BOS_Dist_ATR > 2.0
THEN Trend continuation → WIN
ELSE IF ChoCH_Detected = 1
THEN Potential reversal → Caution
```

---

### 4. **Multi-Timeframe Confluence**

**Objective:** Leverage higher timeframe alignment for higher probability setups

**Key Patterns to Learn:**

- **Strong Confluence → Higher Win Rate**
  - `HTF_OB_Confluence >= 2` (multiple HTF OBs aligned)
  - `HTF_FVG_Confluence >= 1` (HTF FVG present)
  - `HTF_Trend_Alignment = 1 or -1` (HTF trend agrees)
  - `HTF_Structure_Alignment != 0` (HTF BOS aligned)
  - `HTF_Confluence_Quality > 0.7` (overall strong alignment)
  - `HTF_OB_Proximity_Fuzzy > 0.5` (close to HTF structure)
  - `HTF_FVG_Proximity_Fuzzy > 0.5` (close to HTF gap)

- **No Confluence → Lower Probability**
  - `HTF_OB_Confluence = 0`
  - `HTF_FVG_Confluence = 0`
  - `HTF_Trend_Alignment = 0` (no HTF trend)
  - `HTF_Confluence_Quality < 0.3`

**Expected Learning:**
```
IF HTF_OB_Confluence >= 2 AND
   HTF_Trend_Alignment = 1 AND
   HTF_Confluence_Quality > 0.8
THEN Institutional alignment → Very high probability WIN

IF HTF_Confluence_Quality < 0.3
THEN Isolated setup → Higher risk → Possible TIMEOUT
```

---

### 5. **Market Regime Adaptation**

**Objective:** Understand that setups work differently in different market conditions

**Key Patterns to Learn:**

- **Trending Markets (High_Vol_Trend)**
  - Order Blocks work well
  - BOS signals reliable
  - Trend continuation setups → WIN
  - Counter-trend setups → LOSS

- **Ranging Markets (Low_Vol_Chop)**
  - FVGs more reliable (mean reversion)
  - OBs less reliable
  - Structure breaks often false
  - Many setups → TIMEOUT

- **High Volatility (High_Vol)**
  - Larger stops needed
  - Faster moves
  - Higher risk of LOSS if wrong

- **Low Volatility (Low_Vol)**
  - Slower moves
  - Higher risk of TIMEOUT
  - Smaller profit potential

**Expected Learning:**
```
IF Volatility_Regime_Fuzzy = 'High_Vol_Trend' AND
   Trend_Strength_Fuzzy > 2.0 AND
   OB_Bullish_Valid = 1
THEN Bullish OB in trending market → WIN

IF Volatility_Regime_Fuzzy = 'Low_Vol_Chop' AND
   OB_Bullish_Valid = 1
THEN OB in ranging market → TIMEOUT or LOSS
```

---

### 6. **Trend and Momentum Context**

**Objective:** Align setups with directional bias and momentum

**Key Patterns to Learn:**

- **Strong Trend + Pullback → WIN**
  - `Trend_Bias_Indicator > 2.0` (strong uptrend)
  - `RSI < 50` (pullback/retracement)
  - `OB_Bullish_Valid = 1` (bullish setup)
  - `Bullish_Trend_Fuzzy > 0.7` (fuzzy confirmation)

- **Weak Trend → TIMEOUT**
  - `Trend_Bias_Indicator` near 0 (no clear trend)
  - `Trend_Strength_Fuzzy < 1.0` (weak trend)

- **Overbought/Oversold → Caution**
  - `Overbought_Fuzzy > 0.7` (RSI > 70)
  - `Oversold_Fuzzy > 0.7` (RSI < 30)

**Expected Learning:**
```
IF Trend_Bias_Indicator > 2.0 AND
   RSI_Normalized < 0 AND
   OB_Bullish_Valid = 1 AND
   Overbought_Fuzzy < 0.3
THEN Pullback in strong trend → WIN

IF Trend_Bias_Indicator near 0 AND
   Trend_Strength_Fuzzy < 0.5
THEN Choppy market → TIMEOUT
```

---

### 7. **Temporal Sequence Patterns (For Transformer/LSTM)**

**Objective:** Learn the sequence of events that lead to wins

**Key Sequences to Learn:**

**Winning Sequence:**
```
t-10: ChoCH_Detected = 1 (reversal signal)
t-8:  Trend_Bias_Indicator starts increasing
t-5:  OB_Bullish = 1 (order block forms)
t-3:  OB_Displacement_ATR = 4.5 (strong move away)
t-2:  Price retraces toward OB
t-1:  HTF_Confluence_Quality increases
t-0:  Entry at OB retest → Predict WIN
```

**Losing Sequence:**
```
t-5:  OB_Bullish = 1 (order block forms)
t-3:  OB_Displacement_ATR = 1.2 (weak move)
t-2:  Volatility_Regime_Fuzzy = 'Low_Vol_Chop'
t-1:  HTF_Confluence_Quality = 0.2 (no alignment)
t-0:  Entry → Predict LOSS or TIMEOUT
```

**Expected Learning:**
- Recognize setup formation patterns
- Understand time decay (fresh vs. old setups)
- Identify momentum shifts
- Detect regime transitions

---

### 8. **Complex Non-Linear Relationships (For Neural Networks)**

**Objective:** Capture interactions between features that aren't obvious

**Key Interactions to Learn:**

- **Multiplicative Effects:**
  - `OB_Quality_Fuzzy * HTF_Confluence_Quality` → Combined strength
  - `FVG_Depth_ATR * Trend_Strength_Fuzzy` → Gap significance in trend

- **Conditional Logic:**
  - `(Trend_Bias > 2.0) AND (RSI < 50)` → Pullback opportunity
  - `(OB_Age < 5) AND (OB_Mitigated = 0)` → Fresh valid zone

- **Relative Comparisons:**
  - `FVG_Depth_ATR / OB_Size_ATR` → Which structure is stronger?
  - `OB_Displacement_ATR / ATR_ZScore` → Displacement relative to volatility

- **Threshold Combinations:**
  - Multiple weak signals combining to strong signal
  - Strong signal negated by conflicting weak signals

**Expected Learning:**
```
IF (OB_Quality_Fuzzy * HTF_Confluence_Quality > 0.6) AND
   (Trend_Bias_Indicator > 2.0 AND RSI_Normalized < 0) AND
   (FVG_Depth_ATR / OB_Size_ATR > 1.5) AND
   (Volatility_Regime_Fuzzy = 'High_Vol_Trend')
THEN Complex high-probability setup → WIN
```

---

## 📈 Performance Requirements

### **Minimum Acceptable Performance:**

**Classification Metrics:**
- Overall Accuracy: >50% (baseline: 33% random)
- Precision (WIN class): >55%
- Recall (WIN class): >50%
- F1-Score (WIN class): >52%
- Macro F1-Score: >45%

**Trading Metrics:**
- Win Rate (excl. timeouts): >55%
- Risk-Reward Ratio: Maintain 1:3
- Sharpe Ratio (backtest): >1.5
- Maximum Drawdown: <20%
- Profit Factor: >1.8

**Robustness:**
- Performance consistent across symbols
- Performance stable across time periods
- No overfitting (train vs. val gap <5%)
- Generalization to unseen data

---

## 🎓 Model-Specific Learning Objectives

### **Transformer Model:**
- Learn long-term dependencies (50-100 candles)
- Capture sequential patterns in setup formation
- Understand temporal decay of signals
- Model attention to key events (ChoCH, BOS, OB formation)

### **LSTM/GRU Model:**
- Learn short-to-medium term patterns (10-50 candles)
- Capture momentum shifts
- Model state transitions (regime changes)
- Remember context across sequences

### **Neural Network (MLP/CNN):**
- Learn non-linear feature interactions
- Capture complex decision boundaries
- Model multiplicative effects
- Extract high-level patterns from raw features

### **Random Forest:**
- Identify most important features
- Learn threshold-based rules
- Capture feature interactions
- Provide interpretable decision paths

### **XGBoost/LightGBM/CatBoost:**
- Learn gradient-based patterns
- Handle feature importance ranking
- Capture residual patterns missed by other models
- Optimize for classification metrics

### **Regime Classifier:**
- Accurately classify market state
- Predict regime transitions
- Filter setups based on regime
- Provide context for other models

### **Ensemble Meta-Model:**
- Learn when each base model is reliable
- Combine predictions optimally
- Resolve conflicts between models
- Improve overall accuracy through diversity

---

## 🔍 Feature Importance Expectations

**Top 20 Most Important Features (Expected):**

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
11. `HTF_OB_Confluence` - HTF OB count
12. `FVG_Depth_ATR` - Gap size
13. `Trend_Strength_Fuzzy` - Trend strength
14. `OB_Mitigated` - Zone validity
15. `BOS_Close_Confirm` - Structure break type
16. `HTF_FVG_Confluence` - HTF FVG count
17. `OB_Body_Fuzzy_Score` - Candle body quality
18. `FVG_Distance_to_Price_ATR` - Gap proximity
19. `ChoCH_Detected` - Reversal signal
20. `ATR_ZScore` - Volatility context

---

## 🎯 Success Criteria

### **Model is Successful When:**

✅ **Accuracy:** Consistently predicts >55% of trades correctly (excl. timeouts)

✅ **Precision:** When it predicts WIN, it's right >60% of the time

✅ **Recall:** Catches >50% of actual winning setups

✅ **Generalization:** Performs similarly on train, val, and test sets

✅ **Robustness:** Works across different symbols and time periods

✅ **Interpretability:** Feature importance makes sense (OB quality, confluence, etc.)

✅ **Trading Viability:** Produces positive expectancy in backtesting

✅ **Risk Management:** Maintains 1:3 risk-reward ratio

✅ **Consistency:** Stable performance over rolling windows

✅ **Adaptability:** Adjusts to different market regimes

---

## 🚫 What Models Should NOT Learn

### **Anti-Patterns to Avoid:**

❌ **Overfitting to Noise:**
- Memorizing specific price levels
- Learning symbol-specific quirks that don't generalize
- Fitting to random market fluctuations

❌ **Look-Ahead Bias:**
- Using future information to predict past
- Leaking test data into training

❌ **Regime Confusion:**
- Applying trending strategies in ranging markets
- Ignoring volatility context

❌ **False Confidence:**
- High confidence on low-quality setups
- Ignoring confluence requirements

❌ **Temporal Decay Ignorance:**
- Treating old OBs same as fresh OBs
- Ignoring setup age

---

## 📊 Data Requirements

### **Input Data:**
- Source: `Data/processed_smc_data_train.csv`, `_val.csv`, `_test.csv`
- Features: 100+ SMC features
- Target: `TBM_Label` (-1, 0, +1)
- Splits: 70% train, 15% val, 15% test (chronological)

### **Per Symbol:**
- Filter: `df[df['symbol'] == 'EURUSD']`
- Base Timeframe: `df[df['timeframe'] == 'M15']`
- HTF Features: Already embedded in data

### **Preprocessing:**
- Handle missing values (forward fill or drop)
- Encode categorical features (`Volatility_Regime_Fuzzy`)
- Normalize if needed (already done via ATR + Z-score)
- Create sequences for Transformer/LSTM (lookback window)

---

## 🔧 Training Requirements

### **Hardware:**
- GPU (CUDA) for deep learning models
- CPU for classical ML models
- Minimum 16GB RAM
- SSD for fast data loading

### **Hyperparameters:**
- Learning rate: 0.001 (adaptive with scheduler)
- Batch size: 32-128
- Epochs: 50-200 (with early stopping)
- Dropout: 0.2-0.4
- L1/L2 regularization: 0.01-1.0
- Patience (early stopping): 10-20 epochs

### **Monitoring:**
- Track train/val loss curves
- Monitor accuracy, precision, recall, F1
- Check for overfitting (train-val gap)
- Visualize learning rate schedule
- Log feature importance

---

## 🎯 Deployment Requirements

### **Model Outputs:**
- Prediction: Class (-1, 0, +1)
- Confidence: Probability distribution [P(LOSS), P(TIMEOUT), P(WIN)]
- Feature importance: Top 10 contributing features
- Explanation: Why this prediction? (SHAP values)

### **Inference Speed:**
- <100ms per prediction (real-time trading)
- Batch inference: <1s for 100 predictions
- ONNX optimization for production

### **Model Persistence:**
- Save trained models: `models/EURUSD_transformer.pth`
- Save scalers/encoders: `models/EURUSD_scaler.pkl`
- Save metadata: `models/EURUSD_metadata.json`
- Version control: Track model versions

---

## 📚 Summary

The models must become **institutional trade selectors** that:

1. ✅ Identify high-probability SMC setups (OB, FVG, BOS)
2. ✅ Filter out low-quality setups
3. ✅ Understand market context (regime, trend, momentum)
4. ✅ Leverage multi-timeframe confluence
5. ✅ Predict trade outcomes with >55% accuracy
6. ✅ Maintain 1:3 risk-reward ratio
7. ✅ Generalize across symbols and time periods
8. ✅ Provide interpretable predictions
9. ✅ Operate in real-time (<100ms inference)
10. ✅ Adapt to changing market conditions

**Ultimate Goal:** Automate the decision-making process of a professional ICT/SMC trader through quantified, data-driven machine learning models that consistently identify high-probability trading opportunities.

---

## 🚀 Next Steps

1. Implement modular model trainers (one file per model type)
2. Create `integrated_advanced_pipeline.py` orchestrator
3. Train models per symbol with CUDA acceleration
4. Evaluate on validation set
5. Ensemble predictions
6. Backtest on test set
7. Deploy best-performing models
8. Monitor live performance
9. Retrain periodically

**End Goal:** Production-ready ML system that trades profitably using Smart Money Concepts! 🎯
