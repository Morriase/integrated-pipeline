# Live Trading Inference System - Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  MT5 Terminal #1                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  BlackIce EA                                            │    │
│  │  - Collects M15, H1, H4 data (100 bars each)          │    │
│  │  - Sends HTTP POST to server                           │    │
│  │  - Receives prediction + SMC context                   │    │
│  │  - Executes trades                                     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓ HTTP POST
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  MT5 Terminal #2, #3, ... (Multiple Clients)                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Python REST API Server (FastAPI + Uvicorn)                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  API Layer                                              │    │
│  │  - POST /predict                                        │    │
│  │  - GET /health                                          │    │
│  │  - Request validation                                   │    │
│  │  - Error handling                                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Data Processing Layer                                  │    │
│  │  - JSON → DataFrame conversion                          │    │
│  │  - Multi-timeframe merging                             │    │
│  │  - SMCDataPipeline execution                           │    │
│  │  - Feature extraction                                   │    │
│  └────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Inference Layer                                        │    │
│  │  - Load trained models (cached in memory)              │    │
│  │  - Apply preprocessing (scaling, feature selection)    │    │
│  │  - Consensus ensemble prediction                       │    │
│  │  - SMC context extraction                              │    │
│  └────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Response Layer                                         │    │
│  │  - Prediction translation (Win/Loss/Timeout → BUY/SELL/HOLD) │
│  │  - Confidence calculation                               │    │
│  │  - JSON response formatting                             │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓ HTTP Response
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  MT5 Terminals                                                   │
│  - Receive prediction                                            │
│  - Display SMC context                                           │
│  - Execute trades if confidence high                             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Python REST Server (`live_inference_server.py`)

#### 1.1 Server Initialization
```python
class InferenceServer:
    def __init__(self, model_dir='models/trained', host='0.0.0.0', port=5000):
        self.model_dir = model_dir
        self.host = host
        self.port = port
        
        # Load models once at startup
        self.ensemble = ConsensusEnsembleSMCModel()
        self.pipeline = SMCDataPipeline(
            base_timeframe='M15',
            higher_timeframes=['H1', 'H4']
        )
        
        # Metrics
        self.total_predictions = 0
        self.start_time = time.time()
        self.processing_times = []
```

#### 1.2 Data Processing Flow
```python
def process_request(request_data):
    # 1. Validate input
    validate_request(request_data)
    
    # 2. Convert JSON to DataFrame
    df = json_to_dataframe(request_data)
    
    # 3. Run through pipeline
    processed_df = pipeline.run_pipeline_live(df)
    
    # 4. Extract latest M15 features
    latest_features = processed_df[processed_df['timeframe'] == 'M15'].iloc[-1]
    
    # 5. Make prediction
    prediction, confidence = ensemble.predict(latest_features)
    
    # 6. Extract SMC context
    smc_context = extract_smc_context(latest_features)
    
    # 7. Format response
    return format_response(prediction, confidence, smc_context)
```

#### 1.3 Pipeline Adaptation
```python
class SMCDataPipeline:
    def run_pipeline_live(self, df):
        """
        Live inference version - processes data without saving
        Returns processed DataFrame ready for prediction
        """
        # Process each symbol-timeframe combination
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            # Process base timeframe (M15)
            base_df = self.process_timeframe(symbol_data, 'M15')
            
            # Merge higher timeframes
            base_df = self.merge_higher_timeframe_features(
                base_df, symbol_data, symbol
            )
        
        return base_df
```

### 2. MT5 Expert Advisor Modifications

#### 2.1 Multi-Timeframe Data Collection
```mql5
string PrepareMultiTimeframeData()
{
   string json = "{\"symbol\":\"" + _Symbol + "\",\"data\":{";
   
   // M15 data
   json += "\"M15\":[";
   json += CollectTimeframeData(PERIOD_M15, 100);
   json += "],";
   
   // H1 data
   json += "\"H1\":[";
   json += CollectTimeframeData(PERIOD_H1, 100);
   json += "],";
   
   // H4 data
   json += "\"H4\":[";
   json += CollectTimeframeData(PERIOD_H4, 100);
   json += "]";
   
   json += "}}";
   return json;
}

string CollectTimeframeData(ENUM_TIMEFRAMES tf, int bars)
{
   string data = "";
   for(int i = bars - 1; i >= 0; i--)
   {
      datetime time = iTime(_Symbol, tf, i);
      double open = iOpen(_Symbol, tf, i);
      double high = iHigh(_Symbol, tf, i);
      double low = iLow(_Symbol, tf, i);
      double close = iClose(_Symbol, tf, i);
      long volume = iVolume(_Symbol, tf, i);
      
      data += "{";
      data += "\"time\":\"" + TimeToString(time, TIME_DATE|TIME_MINUTES) + "\",";
      data += "\"open\":" + DoubleToString(open, _Digits) + ",";
      data += "\"high\":" + DoubleToString(high, _Digits) + ",";
      data += "\"low\":" + DoubleToString(low, _Digits) + ",";
      data += "\"close\":" + DoubleToString(close, _Digits) + ",";
      data += "\"volume\":" + IntegerToString(volume);
      data += "}";
      
      if(i > 0) data += ",";
   }
   return data;
}
```

#### 2.2 Response Parsing Enhancement
```mql5
void ParseAndExecute(string response)
{
   // Parse prediction (0=SELL, 1=HOLD, 2=BUY)
   int prediction = ExtractInt(response, "\"prediction\":");
   
   // Parse confidence
   double confidence = ExtractDouble(response, "\"confidence\":");
   
   // Parse signal
   string signal = ExtractString(response, "\"signal\":\"");
   
   // Parse consensus
   bool consensus = ExtractBool(response, "\"consensus\":");
   
   // Parse probabilities
   double prob_sell = ExtractDouble(response, "\"SELL\":");
   double prob_hold = ExtractDouble(response, "\"HOLD\":");
   double prob_buy = ExtractDouble(response, "\"BUY\":");
   
   // Parse SMC context
   ParseSMCContext(response);
   
   // Display and execute
   DisplayPrediction(signal, confidence, prob_sell, prob_hold, prob_buy);
   
   if(confidence >= MinConfidence && EnableTrading)
   {
      ExecuteTrade(prediction, confidence);
   }
}
```

## Data Flow

### Request Flow
```
1. MT5 EA OnTick()
   ↓
2. Check if new bar or timer expired
   ↓
3. Collect M15, H1, H4 data (100 bars each)
   ↓
4. Format as JSON
   ↓
5. HTTP POST to /predict
   ↓
6. Server receives request
   ↓
7. Validate JSON structure
   ↓
8. Convert to DataFrame
   ↓
9. Run SMCDataPipeline
   ↓
10. Extract features for latest M15 candle
   ↓
11. Load models (cached)
   ↓
12. Apply preprocessing
   ↓
13. Make prediction
   ↓
14. Extract SMC context
   ↓
15. Format JSON response
   ↓
16. Return to MT5
   ↓
17. EA parses response
   ↓
18. Display on chart
   ↓
19. Execute trade if conditions met
```

### Data Transformation

**Input (MT5 → Server):**
```json
{
  "symbol": "EURUSD",
  "data": {
    "M15": [{"time": "...", "open": 1.0850, ...}],
    "H1": [...],
    "H4": [...]
  }
}
```

**Internal (Server Processing):**
```python
# Convert to DataFrame
df = pd.DataFrame({
    'time': [...],
    'symbol': ['EURUSD', 'EURUSD', ...],
    'timeframe': ['M15', 'M15', ..., 'H1', 'H1', ..., 'H4', 'H4', ...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...]
})

# Run pipeline
processed_df = pipeline.run_pipeline_live(df)

# Extract latest M15 row
latest = processed_df[processed_df['timeframe'] == 'M15'].iloc[-1]

# Features: ~100 columns (OB_*, FVG_*, BOS_*, HTF_*, etc.)
```

**Output (Server → MT5):**
```json
{
  "prediction": 2,
  "signal": "BUY",
  "confidence": 0.75,
  "consensus": true,
  "probabilities": {"SELL": 0.10, "HOLD": 0.15, "BUY": 0.75},
  "models": {"RandomForest": 1, "XGBoost": 1, "NeuralNetwork": 1},
  "smc_context": {...}
}
```

## API Endpoints

### POST /predict

**Request:**
```json
{
  "symbol": "EURUSD",
  "data": {
    "M15": [{"time": "2025-10-20 10:00:00", "open": 1.0850, "high": 1.0855, "low": 1.0848, "close": 1.0852, "volume": 1000}],
    "H1": [...],
    "H4": [...]
  }
}
```

**Response (Success):**
```json
{
  "prediction": 2,
  "signal": "BUY",
  "confidence": 0.75,
  "consensus": true,
  "probabilities": {"SELL": 0.10, "HOLD": 0.15, "BUY": 0.75},
  "models": {"RandomForest": 1, "XGBoost": 1, "NeuralNetwork": 1},
  "smc_context": {
    "order_blocks": {"bullish_present": true, "bearish_present": false, "quality": 0.82},
    "fair_value_gaps": {"bullish_present": true, "bearish_present": false, "depth_atr": 2.3},
    "structure": {"bos_wick_confirmed": true, "bos_close_confirmed": true, "choch_detected": false},
    "regime": {"trend_bias": 0.45, "volatility": "Medium", "regime_label": "Bullish"}
  },
  "timestamp": "2025-10-20T10:15:30Z",
  "processing_time_ms": 45
}
```

**Response (Error):**
```json
{
  "error": "Insufficient data",
  "message": "M15 timeframe requires at least 100 bars, received 50",
  "timestamp": "2025-10-20T10:15:30Z"
}
```

### GET /health

**Response:**
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

## Error Handling

### Server-Side Errors
```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### EA-Side Error Handling
```mql5
if(res == -1)
{
   int error = GetLastError();
   Print("ERROR: WebRequest failed. Error: ", error);
   
   if(error == 4060)
      Print("  URL not allowed. Add to Tools > Options > Expert Advisors");
   else if(error == 4014)
      Print("  Invalid function parameter");
   else
      Print("  Check server status and network connection");
   
   return;
}

// Check for error in response
if(StringFind(response, "\"error\":") >= 0)
{
   string errorMsg = ExtractString(response, "\"message\":\"");
   Print("SERVER ERROR: ", errorMsg);
   return;
}
```

## Performance Optimization

### Model Caching
```python
# Load models once at startup, keep in memory
class InferenceServer:
    def __init__(self):
        self.ensemble = ConsensusEnsembleSMCModel()
        self.ensemble.load_models('models/trained')
        # Models stay in memory for all requests
```

### Async Processing
```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    # FastAPI handles async automatically
    # Multiple requests processed concurrently
    result = await process_prediction(request)
    return result
```

### Response Caching (Optional)
```python
# Cache predictions for same data (within 1 minute)
from functools import lru_cache

@lru_cache(maxsize=100)
def get_prediction(data_hash):
    # Only recompute if data changed
    return make_prediction(data)
```

## Security

### API Key Authentication (Optional)
```python
API_KEYS = {"client1": "key123", "client2": "key456"}

@app.post("/predict")
async def predict(request: PredictionRequest, api_key: str = Header(None)):
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Process request
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request):
    # Max 10 requests per minute per IP
    pass
```

## Deployment

### Local Development
```bash
# Start server
python live_inference_server.py

# Server runs on http://localhost:5000
```

### Production (VPS/Cloud)
```bash
# Use Gunicorn + Uvicorn workers
gunicorn live_inference_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:5000
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "live_inference_server:app", "--host", "0.0.0.0", "--port", "5000"]
```

## Monitoring

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log all predictions
logger.info(f"Prediction: {symbol} {signal} confidence={confidence:.2f}")
```

### Metrics
```python
class Metrics:
    def __init__(self):
        self.total_predictions = 0
        self.processing_times = []
        self.errors = 0
    
    def record_prediction(self, processing_time):
        self.total_predictions += 1
        self.processing_times.append(processing_time)
    
    def get_stats(self):
        return {
            "total": self.total_predictions,
            "avg_time_ms": np.mean(self.processing_times),
            "error_rate": self.errors / self.total_predictions
        }
```

## Testing Strategy

### Unit Tests
- Test JSON to DataFrame conversion
- Test pipeline processing
- Test prediction formatting
- Test error handling

### Integration Tests
- Test full request/response cycle
- Test multi-timeframe processing
- Test model loading
- Test concurrent requests

### Load Tests
- Simulate 10+ concurrent clients
- Measure response times
- Check memory usage
- Verify no crashes

## File Structure
```
project/
├── live_inference_server.py      # Main FastAPI server
├── inference_utils.py             # Helper functions
├── config.py                      # Configuration
├── requirements_server.txt        # Python dependencies
├── models/
│   └── trained/                   # Trained models
│       ├── UNIFIED_RandomForest.pkl
│       ├── UNIFIED_XGBoost.pkl
│       └── UNIFIED_NeuralNetwork.pkl
├── MQL5/
│   ├── BlackIce_MT5_EA.mq5       # Modified EA
│   └── core_functions.mqh         # Helper functions
└── logs/
    └── inference_server.log       # Server logs
```
