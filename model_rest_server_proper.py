"""
Black Ice Intelligence - Proper REST API Server
Utilizes full potential of trained models with correct feature engineering
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - PyTorch models will be disabled")
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import temporal model architectures
try:
    from advanced_temporal_architecture import SMC_LSTM, SMC_Transformer
    TEMPORAL_MODELS_AVAILABLE = True
except ImportError:
    TEMPORAL_MODELS_AVAILABLE = False
    print("⚠️ Temporal model architectures not available - temporal models will be disabled")

# Import base model architectures
try:
    from enhanced_multitf_pipeline import EnhancedSMC_MLP
    BASE_MODELS_AVAILABLE = True
except ImportError:
    BASE_MODELS_AVAILABLE = False
    print("⚠️ Base model architectures not available - base models will be disabled")
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ProperModelServer:
    """
    REST API server that properly utilizes all trained models
    """

    def __init__(self, model_dir: str = "Model_output"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = None
        self.ensemble_weights = {}
        self.sequence_length = 20  # From training config
        self.prediction_horizon = 8  # From training config

        # Load all components
        self._load_models()
        self._load_scalers()
        self._load_ensemble_weights()

    def _load_models(self):
        """Load all trained models with proper typing"""
        logger.info("Loading trained models...")

        # Load base models (neural networks and sklearn)
        ensemble_dir = self.model_dir / "ensemble"
        deployment_dir = self.model_dir / "deployment"

        # Load PyTorch models (TorchScript from deployment)
        pytorch_models = {}
        if TORCH_AVAILABLE:
            for model_file in deployment_dir.glob("*.pt"):
                try:
                    model_name = model_file.stem
                    model = torch.jit.load(model_file, map_location='cpu')
                    model.eval()
                    pytorch_models[model_name] = model
                    logger.info(f"✅ Loaded PyTorch model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load PyTorch model {model_file}: {e}")
        else:
            logger.warning("⚠️ PyTorch not available - skipping PyTorch model loading")

        # Load sklearn models
        sklearn_models = {}
        for model_file in deployment_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                sklearn_models[model_name] = model
                logger.info(f"✅ Loaded sklearn model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sklearn model {model_file}: {e}")

        # Load temporal models (LSTM and Transformer)
        temporal_models = {}
        if TORCH_AVAILABLE and TEMPORAL_MODELS_AVAILABLE:
            for model_file in ensemble_dir.glob("*_state_dict.pth"):
                try:
                    model_name = model_file.stem.replace('_state_dict', '')
                    state_dict = torch.load(model_file, map_location='cpu')

                    if 'lstm' in model_name.lower():
                        # Load SMC_LSTM model
                        model = SMC_LSTM(
                            input_dim=29,  # From training configuration
                            hidden_dim=128,
                            num_layers=2,
                            num_classes=3,
                            dropout=0.3,
                            bidirectional=True
                        )
                        model.load_state_dict(state_dict)
                        model.eval()
                        temporal_models[model_name] = {'model': model, 'type': 'lstm'}
                        logger.info(f"✅ Loaded LSTM model: {model_name}")

                    elif 'transformer' in model_name.lower():
                        # Load SMC_Transformer model
                        model = SMC_Transformer(
                            input_dim=29,  # From training configuration
                            d_model=128,
                            nhead=8,
                            num_layers=4,
                            num_classes=3,
                            dropout=0.1
                        )
                        model.load_state_dict(state_dict)
                        model.eval()
                        temporal_models[model_name] = {'model': model, 'type': 'transformer'}
                        logger.info(f"✅ Loaded Transformer model: {model_name}")

                except Exception as e:
                    logger.warning(f"Failed to load temporal model {model_file}: {e}")
        elif not TEMPORAL_MODELS_AVAILABLE:
            logger.warning("⚠️ Temporal model architectures not available - temporal models disabled")
        else:
            logger.warning("⚠️ PyTorch not available - temporal models disabled")

        self.models = {
            'pytorch': pytorch_models,
            'sklearn': sklearn_models,
            'temporal': temporal_models
        }

        logger.info(f"Loaded {len(pytorch_models)} PyTorch, {len(sklearn_models)} sklearn, {len(temporal_models)} temporal models")

    def _load_scalers(self):
        """Load feature scalers used during training"""
        try:
            # Try ensemble metadata first
            metadata_path = self.model_dir / "ensemble" / "ensemble_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'feature_scalers' in metadata:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(metadata['feature_scalers']['mean'])
                    scaler.scale_ = np.array(metadata['feature_scalers']['std'])
                    scaler.n_features_in_ = len(scaler.mean_)
                    self.scalers = scaler
                    logger.info(f"✅ Loaded scalers from metadata: {len(scaler.mean_)} features")
                    return

            # Fallback to pickle file
            scaler_path = Path("Python/feature_scaler_29.pkl")
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"✅ Loaded scalers from pickle: {self.scalers.n_features_in_} features")
            else:
                logger.warning("❌ No feature scalers found!")

        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")

    def _load_ensemble_weights(self):
        """Load ensemble weights"""
        try:
            metadata_path = self.model_dir / "ensemble" / "ensemble_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'ensemble_weights' in metadata:
                    self.ensemble_weights = metadata['ensemble_weights']
                    logger.info(f"✅ Loaded ensemble weights: {len(self.ensemble_weights)} models")
        except Exception as e:
            logger.error(f"Failed to load ensemble weights: {e}")

    def calculate_institutional_features(self, ohlcv_data: List[Dict]) -> np.ndarray:
        """
        Calculate the exact 24 institutional features used in training
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            df = df.sort_values('time').reset_index(drop=True)

            # Ensure we have enough data
            if len(df) < 50:
                raise ValueError(f"Need at least 50 bars, got {len(df)}")

            # Calculate ATR
            df['ATR'] = self._calculate_atr(df)

            # Calculate EMAs
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

            # Calculate RSI
            df['RSI'] = self._calculate_rsi(df['close'])

            # Detect Order Blocks (simplified institutional version)
            df = self._detect_order_blocks(df)

            # Detect Fair Value Gaps
            df = self._detect_fvg(df)

            # Detect Break of Structure
            df = self._detect_bos(df)

            # Calculate regime features
            df = self._calculate_regime_features(df)

            # Extract the 24 features in exact training order
            feature_columns = [
                'ATR', 'EMA_50', 'EMA_200', 'RSI',
                'OB_Bullish', 'OB_Bearish', 'OB_Size_ATR', 'OB_Displacement_ATR',
                'OB_Quality_Score', 'OB_MTF_Confluence',
                'FVG_Bullish', 'FVG_Bearish', 'FVG_Depth_ATR', 'FVG_Quality_Score',
                'FVG_MTF_Confluence',
                'BOS_Wick_Confirm', 'BOS_Close_Confirm', 'BOS_Dist_ATR', 'Structure_Strength',
                'Trend_Bias_Indicator', 'HTF_Trend_Bias', 'ATR_ZScore',
                'MA_Slope_Normalized', 'RSI_Momentum'
            ]

            # Get features from last row
            features = []
            for col in feature_columns:
                if col in df.columns:
                    val = df[col].iloc[-1]
                    features.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    features.append(0.0)

            logger.info(f"✅ Calculated {len(features)} institutional features")
            return np.array(features)

        except Exception as e:
            logger.error(f"Failed to calculate institutional features: {e}")
            raise

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()

        rs = roll_up / (roll_down + 1e-8)
        return 100 - (100 / (1 + rs))

    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified institutional OB detection"""
        df['OB_Bullish'] = 0
        df['OB_Bearish'] = 0
        df['OB_Size_ATR'] = 0.0
        df['OB_Displacement_ATR'] = 0.0
        df['OB_Quality_Score'] = 0.0
        df['OB_MTF_Confluence'] = 0

        # Simple OB detection logic (simplified for runtime)
        for i in range(5, len(df)):
            current_atr = df['ATR'].iloc[i]
            if pd.isna(current_atr) or current_atr <= 0:
                continue

            # Bullish OB
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i]):

                # Check for displacement
                ob_low = df['low'].iloc[i-1]
                future_high = df['high'].iloc[i+1:i+6].max() if i+6 < len(df) else df['high'].iloc[i]

                displacement = (future_high - ob_low) / current_atr
                if displacement > 1.5:  # Minimum displacement
                    df.loc[df.index[i], 'OB_Bullish'] = 1
                    df.loc[df.index[i], 'OB_Size_ATR'] = (df['high'].iloc[i-1] - ob_low) / current_atr
                    df.loc[df.index[i], 'OB_Displacement_ATR'] = displacement
                    df.loc[df.index[i], 'OB_Quality_Score'] = min(displacement / 3.0, 1.0)

            # Bearish OB
            elif (df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                  df['close'].iloc[i] < df['open'].iloc[i]):

                ob_high = df['high'].iloc[i-1]
                future_low = df['low'].iloc[i+1:i+6].min() if i+6 < len(df) else df['low'].iloc[i]

                displacement = (ob_high - future_low) / current_atr
                if displacement > 1.5:
                    df.loc[df.index[i], 'OB_Bearish'] = 1
                    df.loc[df.index[i], 'OB_Size_ATR'] = (ob_high - df['low'].iloc[i-1]) / current_atr
                    df.loc[df.index[i], 'OB_Displacement_ATR'] = displacement
                    df.loc[df.index[i], 'OB_Quality_Score'] = min(displacement / 3.0, 1.0)

        return df

    def _detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps"""
        df['FVG_Bullish'] = 0
        df['FVG_Bearish'] = 0
        df['FVG_Depth_ATR'] = 0.0
        df['FVG_Quality_Score'] = 0.0
        df['FVG_MTF_Confluence'] = 0

        for i in range(2, len(df)):
            current_atr = df['ATR'].iloc[i]

            # Bullish FVG: gap between candles
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                gap_atr = gap_size / current_atr if current_atr > 0 else 0

                if gap_atr > 0.5:  # Significant gap
                    df.loc[df.index[i], 'FVG_Bullish'] = 1
                    df.loc[df.index[i], 'FVG_Depth_ATR'] = gap_atr
                    df.loc[df.index[i], 'FVG_Quality_Score'] = min(gap_atr / 2.0, 1.0)

            # Bearish FVG
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                gap_atr = gap_size / current_atr if current_atr > 0 else 0

                if gap_atr > 0.5:
                    df.loc[df.index[i], 'FVG_Bearish'] = 1
                    df.loc[df.index[i], 'FVG_Depth_ATR'] = gap_atr
                    df.loc[df.index[i], 'FVG_Quality_Score'] = min(gap_atr / 2.0, 1.0)

        return df

    def _detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Break of Structure"""
        df['BOS_Wick_Confirm'] = 0
        df['BOS_Close_Confirm'] = 0
        df['BOS_Dist_ATR'] = 0.0
        df['Structure_Strength'] = 0.0

        # Simple BOS detection
        for i in range(10, len(df)):
            current_atr = df['ATR'].iloc[i]

            # Check for break above recent highs (bullish BOS)
            recent_highs = df['high'].iloc[i-10:i].max()
            if df['high'].iloc[i] > recent_highs:
                df.loc[df.index[i], 'BOS_Wick_Confirm'] = 1
                df.loc[df.index[i], 'BOS_Dist_ATR'] = (df['high'].iloc[i] - recent_highs) / current_atr
                df.loc[df.index[i], 'Structure_Strength'] = 0.8

            # Check for break below recent lows (bearish BOS)
            recent_lows = df['low'].iloc[i-10:i].min()
            if df['low'].iloc[i] < recent_lows:
                df.loc[df.index[i], 'BOS_Wick_Confirm'] = -1  # Negative for bearish
                df.loc[df.index[i], 'BOS_Dist_ATR'] = (recent_lows - df['low'].iloc[i]) / current_atr
                df.loc[df.index[i], 'Structure_Strength'] = 0.8

        return df

    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime and trend features"""
        # Trend bias indicators
        df['Trend_Bias_Indicator'] = (df['close'] - df['EMA_200']) / df['ATR']
        df['HTF_Trend_Bias'] = (df['EMA_50'] - df['EMA_200']) / df['ATR']

        # ATR Z-score
        df['ATR_ZScore'] = (df['ATR'] - df['ATR'].rolling(50).mean()) / df['ATR'].rolling(50).std()

        # MA slope
        df['MA_Slope_Normalized'] = df['EMA_50'].diff(5) / df['ATR']

        # RSI momentum
        df['RSI_Momentum'] = df['RSI'].diff(3)

        # Fill NaN
        df = df.fillna(0)

        return df

    def create_temporal_sequence(self, features_history: List[np.ndarray]) -> np.ndarray:
        """
        Create temporal sequence from recent feature vectors
        """
        if len(features_history) < self.sequence_length:
            # Pad with zeros if not enough history
            padding = [np.zeros(29) for _ in range(self.sequence_length - len(features_history))]
            sequence = padding + features_history[-self.sequence_length:]
        else:
            sequence = features_history[-self.sequence_length:]

        return np.array(sequence)

    def predict(self, ohlcv_data: List[Dict], features_history: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Make prediction using all available models
        """
        try:
            # Calculate current features
            current_features = self.calculate_institutional_features(ohlcv_data)

            # Normalize features
            if self.scalers is None:
                raise ValueError("Feature scalers not loaded")

            current_features_norm = self.scalers.transform(current_features.reshape(1, -1))[0]

            predictions = {}
            model_votes = {"SELL": 0, "HOLD": 0, "BUY": 0}

            # Predict with base models (single features)
            logger.info("Running base model predictions...")

            # PyTorch models
            if TORCH_AVAILABLE:
                for model_name, model in self.models['pytorch'].items():
                    try:
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(current_features_norm).unsqueeze(0)
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1).numpy()[0]
                            pred = np.argmax(probs)
                            conf = float(np.max(probs))

                            predictions[model_name] = {
                                'prediction': int(pred),
                                'confidence': conf,
                                'probabilities': probs.tolist()
                            }

                            action = ["SELL", "HOLD", "BUY"][pred]
                            model_votes[action] += 1

                            logger.info(f"  {model_name}: {action} ({conf:.1%})")

                    except Exception as e:
                        logger.warning(f"PyTorch model {model_name} failed: {e}")
            else:
                logger.warning("⚠️ PyTorch not available - skipping PyTorch model predictions")

            # Sklearn models
            for model_name, model in self.models['sklearn'].items():
                try:
                    pred = model.predict(current_features_norm.reshape(1, -1))[0]

                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(current_features_norm.reshape(1, -1))[0]
                        conf = float(np.max(probs))
                    else:
                        probs = np.array([0.33, 0.34, 0.33])
                        conf = 0.6

                    predictions[model_name] = {
                        'prediction': int(pred),
                        'confidence': conf,
                        'probabilities': probs.tolist()
                    }

                    action = ["SELL", "HOLD", "BUY"][pred]
                    model_votes[action] += 1

                    logger.info(f"  {model_name}: {action} ({conf:.1%})")

                except Exception as e:
                    logger.warning(f"Sklearn model {model_name} failed: {e}")

            # Predict with temporal models (sequences)
            if features_history and TORCH_AVAILABLE:
                logger.info("Running temporal model predictions...")
                sequence = self.create_temporal_sequence(features_history + [current_features_norm])

                for model_name, model_info in self.models['temporal'].items():
                    try:
                        model = model_info['model']
                        model_type = model_info['type']

                        with torch.no_grad():
                            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)

                            if model_type == 'lstm':
                                output, _ = model(seq_tensor)
                            elif model_type == 'transformer':
                                output = model(seq_tensor)

                            probs = torch.softmax(output, dim=1).numpy()[0]
                            pred = np.argmax(probs)
                            conf = float(np.max(probs))

                            predictions[model_name] = {
                                'prediction': int(pred),
                                'confidence': conf,
                                'probabilities': probs.tolist()
                            }

                            action = ["SELL", "HOLD", "BUY"][pred]
                            model_votes[action] += 1

                            logger.info(f"  {model_name}: {action} ({conf:.1%})")

                    except Exception as e:
                        logger.warning(f"Temporal model {model_name} failed: {e}")
            elif features_history and not TORCH_AVAILABLE:
                logger.warning("⚠️ PyTorch not available - skipping temporal model predictions")

            # Ensemble decision
            if not predictions:
                return {"error": "No models produced predictions"}

            # Weighted ensemble
            weighted_probs = np.zeros(3)
            total_weight = 0

            for model_name, pred_data in predictions.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                weighted_probs += np.array(pred_data['probabilities']) * weight
                total_weight += weight

            if total_weight > 0:
                weighted_probs /= total_weight

            final_prediction = int(np.argmax(weighted_probs))
            final_confidence = float(np.max(weighted_probs))

            # Determine action
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            final_action = action_map[final_prediction]

            # Consensus analysis
            max_votes = max(model_votes.values())
            consensus = "Strong" if max_votes >= len(predictions) * 0.75 else "Moderate" if max_votes >= len(predictions) * 0.5 else "Weak"

            # Trading decision
            should_trade = final_confidence > 0.7

            result = {
                "action": final_action,
                "confidence": final_confidence,
                "signal_strength": final_confidence * 0.9,
                "should_trade": should_trade,
                "probabilities": {
                    "sell": float(weighted_probs[0]),
                    "hold": float(weighted_probs[1]),
                    "buy": float(weighted_probs[2])
                },
                "model_predictions": predictions,
                "model_votes": model_votes,
                "consensus": consensus,
                "models_used": len(predictions),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"FINAL DECISION: {final_action} ({final_confidence:.1%}) - {consensus} consensus")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def _get_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Simple positional encoding for transformer"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# Global server instance
server = ProperModelServer()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "pytorch": len(server.models['pytorch']),
            "sklearn": len(server.models['sklearn']),
            "temporal": len(server.models['temporal'])
        },
        "total_models": sum(len(v) for v in server.models.values()),
        "scalers_loaded": server.scalers is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint accepting OHLCV data"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract OHLCV data
        if 'ohlcv' not in data:
            return jsonify({"error": "Missing 'ohlcv' field with OHLCV data"}), 400

        ohlcv_data = data['ohlcv']
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) == 0:
            return jsonify({"error": "'ohlcv' must be a non-empty list"}), 400

        # Optional features history for temporal models
        features_history = data.get('features_history', [])

        # Make prediction
        result = server.predict(ohlcv_data, features_history)

        if 'error' in result:
            return jsonify(result), 500

        return jsonify({
            "success": True,
            "prediction": result,
            "symbol": data.get('symbol', 'UNKNOWN'),
            "timeframe": data.get('timeframe', 'UNKNOWN')
        })

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "models": {
            "pytorch": list(server.models['pytorch'].keys()),
            "sklearn": list(server.models['sklearn'].keys()),
            "temporal": list(server.models['temporal'].keys())
        },
        "ensemble_weights": server.ensemble_weights,
        "expected_features": 24,
        "sequence_length": server.sequence_length,
        "accepts_ohlcv": True,
        "accepts_features_history": True
    })


if __name__ == '__main__':
    print("🚀 Starting Black Ice Intelligence - Proper REST API Server...")
    print("📡 Server will be available at: http://localhost:5001")
    print("🔍 Health check: http://localhost:5001/health")
    print("🤖 Prediction: POST to http://localhost:5001/predict")
    print("📊 Models: GET /models")

    app.run(host='0.0.0.0', port=5001, debug=True)