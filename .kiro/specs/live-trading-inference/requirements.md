# Live Trading Inference System - Requirements

## Overview
Build a production-ready REST API server and MT5 Expert Advisor for live trading predictions using trained SMC models.

## Business Requirements

### BR-1: Multi-User REST Server
- Support multiple MT5 terminals connecting simultaneously
- Handle concurrent prediction requests
- Maintain single model instance in memory (efficiency)
- Provide consistent predictions across all clients

### BR-2: Multi-Timeframe Data Processing
- Accept M15, H1, and H4 timeframe data from MT5
- Process exactly as training pipeline does
- Ensure feature engineering matches training
- Return predictions for base timeframe (M15)

### BR-3: Model Inference
- Load trained consensus ensemble (RF + XGBoost + NN)
- Use same preprocessing (scaling, feature selection)
- Translate model outputs to trading signals
- Provide confidence scores and SMC context

### BR-4: Trading Signal Format
- Map predictions: Win(1)→BUY, Loss(-1)→SELL, Timeout(0)→HOLD
- Include confidence threshold filtering
- Provide individual model predictions
- Return SMC context (Order Blocks, FVGs, Structure, Regime)

## Functional Requirements

### FR-1: REST API Server (Python)

#### FR-1.1: Startup
- Load trained models from `models/trained/` directory
- Initialize data preparation pipeline
- Load scalers and feature selectors
- Validate all components loaded successfully
- Listen on configurable host:port (default: 0.0.0.0:5000)

#### FR-1.2: Endpoint: POST /predict
**Input:**
```json
{
  "symbol": "EURUSD",
  "data": {
    "M15": [
      {"time": "2025-10-20 10:00:00", "open": 1.0850, "high": 1.0855, "low": 1.0848, "close": 1.0852, "volume": 1000},
      ...
    ],
    "H1": [...],
    "H4": [...]
  }
}
```

**Output:**
```json
{
  "prediction": 2,
  "signal": "BUY",
  "confidence": 0.75,
  "consensus": true,
  "probabilities": {
    "SELL": 0.10,
    "HOLD": 0.15,
    "BUY": 0.75
  },
  "models": {
    "RandomForest": 1,
    "XGBoost": 1,
    "NeuralNetwork": 1
  },
  "smc_context": {
    "order_blocks": {
      "bullish_present": true,
      "bearish_present": false,
      "quality": 0.82
    },
    "fair_value_gaps": {
      "bullish_present": true,
      "bearish_present": false,
      "depth_atr": 2.3
    },
    "structure": {
      "bos_wick_confirmed": true,
      "bos_close_confirmed": true,
      "choch_detected": false
    },
    "regime": {
      "trend_bias": 0.45,
      "volatility": "Medium",
      "regime_label": "Bullish"
    }
  },
  "timestamp": "2025-10-20T10:15:30Z",
  "processing_time_ms": 45
}
```

#### FR-1.3: Endpoint: GET /health
**Output:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "models": ["RandomForest", "XGBoost", "NeuralNetwork"],
  "uptime_seconds": 3600,
  "total_predictions": 1523,
  "avg_processing_time_ms": 42
}
```

#### FR-1.4: Data Processing Pipeline
1. Receive multi-timeframe JSON data
2. Convert to DataFrame format (like consolidate_mt5_data.py)
3. Run through SMCDataPipeline (same as training)
4. Extract features for latest M15 candle
5. Apply same preprocessing (scaling, feature selection)
6. Make prediction using consensus ensemble
7. Extract SMC context from processed features
8. Format response

#### FR-1.5: Error Handling
- Invalid JSON format → 400 Bad Request
- Missing required fields → 400 Bad Request
- Insufficient data bars → 400 Bad Request
- Processing errors → 500 Internal Server Error
- Return descriptive error messages

#### FR-1.6: Logging
- Log all prediction requests (timestamp, symbol, result)
- Log processing times
- Log errors with stack traces
- Rotate logs daily

### FR-2: MT5 Expert Advisor Modifications

#### FR-2.1: Multi-Timeframe Data Collection
- Collect 100 bars from M15 timeframe
- Collect 100 bars from H1 timeframe
- Collect 100 bars from H4 timeframe
- Ensure time alignment (all data up to current time)

#### FR-2.2: JSON Request Format
```json
{
  "symbol": "EURUSD",
  "data": {
    "M15": [
      {"time": "2025-10-20 10:00:00", "open": 1.0850, "high": 1.0855, "low": 1.0848, "close": 1.0852, "volume": 1000}
    ],
    "H1": [...],
    "H4": [...]
  }
}
```

#### FR-2.3: Response Parsing
- Parse prediction (0=SELL, 1=HOLD, 2=BUY)
- Parse confidence score
- Parse individual model predictions
- Parse SMC context
- Handle errors gracefully

#### FR-2.4: Trading Logic Updates
- Use prediction + confidence for trade decisions
- Display SMC context on chart
- Log all predictions to CSV
- Maintain existing risk management

## Technical Requirements

### TR-1: Python Server Stack
- Framework: FastAPI (async, high performance)
- ASGI Server: Uvicorn
- Dependencies: pandas, numpy, scikit-learn, xgboost, torch
- Python version: 3.8+

### TR-2: Performance
- Model loading: < 5 seconds on startup
- Prediction latency: < 100ms per request
- Support 10+ concurrent requests
- Memory usage: < 2GB

### TR-3: Data Validation
- Validate timeframe data completeness
- Check for NaN/Inf values
- Verify time ordering
- Ensure minimum bars (100 per timeframe)

### TR-4: Model Compatibility
- Use exact same pipeline as training
- Load saved scalers/feature selectors
- Match feature names exactly
- Handle missing features gracefully

### TR-5: Security (Multi-User)
- Optional API key authentication
- Rate limiting (10 requests/minute per client)
- CORS configuration
- Input sanitization

## Non-Functional Requirements

### NFR-1: Reliability
- Graceful error handling
- Automatic recovery from failures
- Model validation on startup
- Health check endpoint

### NFR-2: Maintainability
- Clear code structure
- Comprehensive logging
- Configuration file for settings
- Documentation

### NFR-3: Scalability
- Stateless design (except model cache)
- Support horizontal scaling
- Efficient memory usage
- Connection pooling

### NFR-4: Monitoring
- Track prediction counts
- Monitor processing times
- Log error rates
- Health metrics

## Constraints

### C-1: Data Format Compatibility
- Must match training pipeline exactly
- Cannot modify feature engineering
- Must use same normalization

### C-2: Model Files
- Models must exist in `models/trained/`
- Must include: RandomForest, XGBoost, NeuralNetwork
- Must include scalers and metadata

### C-3: MT5 Limitations
- WebRequest must be enabled
- URL must be whitelisted
- JSON parsing is manual (no library)
- Limited string manipulation

## Success Criteria

1. Server starts and loads models successfully
2. Accepts multi-timeframe data from MT5
3. Returns predictions in < 100ms
4. Predictions match training pipeline output
5. EA successfully sends data and receives predictions
6. SMC context is accurately extracted
7. Multiple MT5 terminals can connect simultaneously
8. Error handling prevents crashes
9. Logging captures all activity
10. Health check confirms system status

## Out of Scope

- Model retraining
- Real-time model updates
- Database storage
- Web dashboard
- Historical prediction storage
- Backtesting integration
- Trade execution (handled by EA)
- Position management (handled by EA)
