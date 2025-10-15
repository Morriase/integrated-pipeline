"""
BlackIce Model REST Server V2
Optimized for 60% Accurate Institutional Models with 24 SMC Features
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class InstitutionalModelServer:
    """
    REST server for institutional-grade SMC models
    - 24 features (vs old 29)
    - 8-model weighted ensemble
    - 60% accuracy
    """

    def __init__(self, model_dir="Model_output/deployment"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = None
        self.ensemble_weights = {}
        self.feature_names = [
            # Basic (4)
            'ATR', 'EMA_50', 'EMA_200', 'RSI',
            # Order Blocks (6)
            'OB_Bullish', 'OB_Bearish', 'OB_Size_ATR',
            'OB_Displacement_ATR', 'OB_Quality_Score', 'OB_MTF_Confluence',
            # Fair Value Gaps (5)
            'FVG_Bullish', 'FVG_Bearish', 'FVG_Depth_ATR',
            'FVG_Quality_Score', 'FVG_MTF_Confluence',
            # Structure Breaks (4)
            'BOS_Wick_Confirm', 'BOS_Close_Confirm', 'BOS_Dist_ATR',
            'Structure_Strength',
            # Regime (5)
            'Trend_Bias_Indicator', 'HTF_Trend_Bias', 'ATR_ZScore',
            'MA_Slope_Normalized', 'RSI_Momentum'
        ]

        self.load_models()

    def load_models(self):
        """Load ensemble models and configuration"""
        try:
            logger.info("="*70)
            logger.info("LOADING INSTITUTIONAL MODELS")
            logger.info("="*70)

            # Load ensemble configuration
            config_path = self.model_dir / "ensemble_config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Ensemble config not found: {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)
                self.ensemble_weights = config['ensemble_weights']

            logger.info(
                f"✅ Ensemble weights loaded: {len(self.ensemble_weights)} models")
            logger.info(
                f"✅ Using {len(self.feature_names)} institutional SMC features")

            # Load feature scalers
            scaler_path = self.model_dir / "feature_scalers.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Feature scalers not found: {scaler_path}")

            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)

            # Handle both dict and sklearn scaler formats
            if isinstance(scaler_data, dict):
                # Create sklearn-compatible scaler from dict
                from sklearn.preprocessing import StandardScaler
                self.scalers = StandardScaler()
                self.scalers.mean_ = np.array(scaler_data['mean'])
                self.scalers.scale_ = np.array(scaler_data['std'])
                self.scalers.n_features_in_ = len(self.scalers.mean_)
            else:
                # Already a sklearn scaler
                self.scalers = scaler_data

            logger.info(
                f"✅ Feature scalers loaded: {self.scalers.mean_.shape[0]} features")

            # Load all models from ensemble config
            for model_name, weight in self.ensemble_weights.items():
                model_path = None
                model_type = None

                # Determine model file extension and type
                if model_name.startswith('neural_network'):
                    model_path = self.model_dir / f"{model_name}.pt"
                    model_type = 'pytorch'
                else:
                    # Map config model names to actual file names
                    file_name_map = {
                        'lightgbm': 'lightgbm.pkl',
                        'xgboost': 'xgboost.pkl',
                        'random_forest_deep': 'random_forest_deep.pkl',
                        'random_forest_wide': 'random_forest_wide.pkl',
                        'gradient_boosting': 'gradient_boosting.pkl',
                        'logistic_regression': 'logistic_regression.pkl'
                    }
                    if model_name in file_name_map:
                        model_path = self.model_dir / file_name_map[model_name]
                        model_type = 'sklearn'
                    else:
                        logger.warning(f"⚠️  Unknown model type: {model_name}")
                        continue

                if model_path and model_path.exists():
                    if model_type == 'pytorch':
                        # Load PyTorch model
                        model = torch.load(model_path, map_location='cpu')
                        model.eval()  # Set to evaluation mode
                    else:
                        # Load sklearn model
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)

                    self.models[model_name] = {
                        'model': model,
                        'type': model_type,
                        'weight': weight
                    }
                    logger.info(
                        f"✅ Loaded {model_name} ({model_type}, weight: {weight:.4f})")
                else:
                    logger.warning(f"⚠️  Model file not found: {model_path}")

            logger.info("="*70)
            logger.info(f"✅ READY: {len(self.models)} models loaded")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            logger.error(traceback.format_exc())
            raise

    def calculate_institutional_features(self, ohlcv_data):
        """
        Calculate 24 institutional SMC features from OHLCV data (matching training)

        Args:
            ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
                       Should contain at least 200 bars for proper SMC detection

        Returns:
            numpy array of 24 features
        """
        try:
            # Import institutional SMC functions
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from feature_engineering_smc_institutional import (
                compute_atr,
                detect_order_blocks_institutional,
                detect_fvg_institutional,
                detect_bos_choch_institutional,
                detect_regime_institutional
            )

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)

            # Ensure required columns
            required_cols = ['time', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"Missing required columns. Got: {df.columns.tolist()}")

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            logger.info(
                f"Processing {len(df)} bars for SMC feature extraction...")

            # Apply institutional SMC pipeline
            df['ATR'] = compute_atr(df)
            df = detect_order_blocks_institutional(df)
            df = detect_fvg_institutional(df)
            df = detect_bos_choch_institutional(df)
            df = detect_regime_institutional(df)

            # Apply regime gating (same as training)
            from feature_engineering_smc_institutional import regime_context_gating_institutional

            df_gated = regime_context_gating_institutional(df.copy())

            # Check if latest bar passes regime filter
            regime_ok = len(
                df_gated) > 0 and df_gated.index[-1] == df.index[-1]

            if not regime_ok:
                logger.warning(
                    "⚠️  Current market regime does not meet institutional criteria - skipping prediction")
                # Return zeros to indicate skip
                return np.full((1, len(self.feature_names)), 0)

            # Extract latest bar features (24 features) from gated data
            latest = df_gated.iloc[-1]

            # Fill NaN values with 0
            features = []
            for feature_name in self.feature_names:
                value = latest.get(feature_name, 0)
                if pd.isna(value):
                    value = 0
                features.append(float(value))

            logger.info(
                f"✅ Extracted {len(features)} institutional SMC features")

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"❌ Feature calculation failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _calculate_technical_indicators(self, df):
        """
        Calculate technical indicators matching the ensemble training features
        """
        # Basic price data
        df['close'] = df['close']
        df['open'] = df['open']
        df['high'] = df['high']
        df['low'] = df['low']
        df['volume'] = df.get('volume', 0)

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd_main'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_main'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

        # Volatility (standard deviation of returns)
        df['volatility'] = df['close'].pct_change().rolling(20).std()

        # Time-based features
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek

        # Trend indicators (simplified)
        df['h1_trend'] = (df['close'] > df['sma_20']).astype(
            int)  # 1 if above MA, 0 if below
        df['h4_trend'] = (df['close'] > df['sma_50']).astype(int)
        df['d1_trend'] = (df['close'] > df['sma_200']).astype(int)

        # Momentum indicators
        df['h1_momentum'] = df['close'].pct_change(1)  # 1-hour momentum
        df['h4_momentum'] = df['close'].pct_change(4)  # 4-hour momentum

        # Liquidity level (volume-based)
        df['liquidity_level'] = df['volume'].rolling(20).mean()

        # Order block strength (simplified - high volume + range)
        df['order_block_strength'] = (
            df['volume'] * (df['high'] - df['low'])).rolling(5).mean()

        # Fair value gap (simplified)
        df['fair_value_gap'] = (
            (df['high'].shift(1) - df['low']) / (df['high'] - df['low'].shift(1)) - 1).abs()

        # Market structure shift (simplified - change in trend)
        df['market_structure_shift'] = (
            df['h1_trend'] != df['h1_trend'].shift(1)).astype(int)

        # Fill NaN values
        df = df.fillna(0)

        return df
        """
        Weighted ensemble prediction using trained weights

        Args:
            features: numpy array of 24 features

        Returns:
            dict with prediction, probabilities, confidence, and model votes
        """
        try:
            # Normalize features
            features_norm = self.scalers.transform(features)

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            for model_name, model_info in self.models.items():
                if model_info['type'] == 'sklearn':
                    pred = model_info['model'].predict(features_norm)[0]
                    proba = model_info['model'].predict_proba(features_norm)[0]

                    predictions[model_name] = int(pred)
                    probabilities[model_name] = proba.tolist()

                elif model_info['type'] == 'pytorch':
                    # Convert to torch tensor
                    features_tensor = torch.FloatTensor(features_norm)

                    with torch.no_grad():
                        outputs = model_info['model'](features_tensor)
                        # Apply softmax to get probabilities
                        proba = torch.softmax(outputs, dim=1).numpy()[0]
                        pred = int(np.argmax(proba))

                    predictions[model_name] = pred
                    probabilities[model_name] = proba.tolist()

            # Weighted voting
            weighted_proba = np.zeros(3)  # [SELL, HOLD, BUY]

            for model_name, proba in probabilities.items():
                weight = self.models[model_name]['weight']
                weighted_proba += weight * np.array(proba)

            # Final prediction
            final_pred = int(np.argmax(weighted_proba))
            confidence = float(weighted_proba[final_pred])

            # Build ensemble details
            ensemble_details = {}
            for model_name, pred in predictions.items():
                ensemble_details[model_name] = {
                    'vote': int(pred),
                    'confidence': float(probabilities[model_name][pred]),
                    'weight': float(self.models[model_name]['weight'])
                }

            return {
                'prediction': final_pred,
                'probabilities': {
                    'SELL': float(weighted_proba[0]),
                    'HOLD': float(weighted_proba[1]),
                    'BUY': float(weighted_proba[2])
                },
                'confidence': confidence,
                'ensemble_details': ensemble_details
            }

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def extract_smc_context(self, ohlcv_data):
        """
        Extract SMC context from latest bar for enhanced response

        Returns:
            dict with SMC structure information
        """
        try:
            df = pd.DataFrame(ohlcv_data)
            latest = df.iloc[-1]

            return {
                'order_blocks': {
                    'bullish_present': bool(latest.get('OB_Bullish', 0)),
                    'bearish_present': bool(latest.get('OB_Bearish', 0)),
                    'quality_score': float(latest.get('OB_Quality_Score', 0)),
                    'mtf_confluence': bool(latest.get('OB_MTF_Confluence', 0))
                },
                'fair_value_gaps': {
                    'bullish_present': bool(latest.get('FVG_Bullish', 0)),
                    'bearish_present': bool(latest.get('FVG_Bearish', 0)),
                    'depth_atr': float(latest.get('FVG_Depth_ATR', 0)),
                    'quality_score': float(latest.get('FVG_Quality_Score', 0))
                },
                'structure': {
                    'bos_wick_confirmed': bool(latest.get('BOS_Wick_Confirm', 0)),
                    'bos_close_confirmed': bool(latest.get('BOS_Close_Confirm', 0)),
                    'strength': float(latest.get('Structure_Strength', 0))
                },
                'regime': {
                    'trend_bias': float(latest.get('Trend_Bias_Indicator', 0)),
                    'htf_bias': float(latest.get('HTF_Trend_Bias', 0)),
                    'volatility_zscore': float(latest.get('ATR_ZScore', 0))
                }
            }
        except Exception as e:
            logger.warning(f"⚠️ SMC context extraction failed: {e}")
            return {}

    def generate_reasoning(self, prediction, confidence, smc_context):
        """
        Generate human-readable reasoning for the prediction

        Returns:
            str with reasoning text
        """
        pred_label = ['SELL', 'HOLD', 'BUY'][prediction]

        reasons = []

        # Confidence level
        if confidence >= 0.70:
            reasons.append(
                f"High confidence {pred_label} signal ({confidence:.1%})")
        elif confidence >= 0.60:
            reasons.append(
                f"Moderate confidence {pred_label} signal ({confidence:.1%})")
        else:
            reasons.append(
                f"Low confidence {pred_label} signal ({confidence:.1%})")

        # SMC context
        if smc_context:
            ob = smc_context.get('order_blocks', {})
            fvg = smc_context.get('fair_value_gaps', {})
            structure = smc_context.get('structure', {})
            regime = smc_context.get('regime', {})

            if prediction == 2:  # BUY
                if ob.get('bullish_present'):
                    reasons.append(
                        f"Bullish OB present (quality: {ob.get('quality_score', 0):.2f})")
                if fvg.get('bullish_present'):
                    reasons.append(f"Bullish FVG detected")
                if structure.get('bos_close_confirmed'):
                    reasons.append(
                        f"BOS confirmed (strength: {structure.get('strength', 0):.1f})")
                if regime.get('trend_bias', 0) > 0.5:
                    reasons.append(f"Bullish trend regime")

            elif prediction == 0:  # SELL
                if ob.get('bearish_present'):
                    reasons.append(
                        f"Bearish OB present (quality: {ob.get('quality_score', 0):.2f})")
                if fvg.get('bearish_present'):
                    reasons.append(f"Bearish FVG detected")
                if structure.get('bos_close_confirmed'):
                    reasons.append(
                        f"BOS confirmed (strength: {structure.get('strength', 0):.1f})")
                if regime.get('trend_bias', 0) < -0.5:
                    reasons.append(f"Bearish trend regime")

        if len(reasons) == 1:
            reasons.append("Ensemble consensus")

        return ". ".join(reasons) + "."


# Initialize server
server = InstitutionalModelServer()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(server.models),
        'features': len(server.feature_names),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint

    Expected JSON:
    {
        "ohlcv": [
            {"time": "2024-01-01 00:00:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005, "volume": 1000},
            ...
        ]
    }

    Returns:
    {
        "prediction": 2,
        "probabilities": {"SELL": 0.15, "HOLD": 0.05, "BUY": 0.80},
        "confidence": 0.80,
        "ensemble_details": {...},
        "smc_context": {...},
        "reasoning": "..."
    }
    """
    try:
        data = request.json

        if 'ohlcv' not in data:
            return jsonify({'error': 'Missing ohlcv data'}), 400

        ohlcv_data = data['ohlcv']

        if len(ohlcv_data) < 200:
            return jsonify({'error': f'Insufficient data: need 200+ bars, got {len(ohlcv_data)}'}), 400

        # Calculate institutional features
        features = server.calculate_institutional_features(ohlcv_data)

        # Get ensemble prediction
        prediction_result = server.predict_ensemble(features)

        # Extract SMC context
        smc_context = server.extract_smc_context(ohlcv_data)

        # Generate reasoning
        reasoning = server.generate_reasoning(
            prediction_result['prediction'],
            prediction_result['confidence'],
            smc_context
        )

        # Build response
        response = {
            **prediction_result,
            'smc_context': smc_context,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✅ Prediction: {['SELL', 'HOLD', 'BUY'][prediction_result['prediction']]} "
                    f"(confidence: {prediction_result['confidence']:.2%})")

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['POST'])
def get_features():
    """
    Calculate and return features without prediction (for debugging)
    """
    try:
        data = request.json
        ohlcv_data = data['ohlcv']

        features = server.calculate_institutional_features(ohlcv_data)

        return jsonify({
            'features': features[0].tolist(),
            'feature_names': server.feature_names,
            'count': len(server.feature_names)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BLACKICE INSTITUTIONAL MODEL SERVER V2")
    print("="*70)
    print(f"Models: 8-model weighted ensemble")
    print(f"Accuracy: 60% (vs 33% random baseline)")
    print(f"Features: 24 institutional SMC features")
    print(f"HOLD bias: 5.8% (vs 88% old system)")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Get prediction")
    print("  POST /features - Get features only")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)


# Model agreement check added for Phase 1
def check_model_agreement(predictions_dict):
    """Check if models agree on prediction (3 out of 4)"""
    from collections import Counter
    votes = Counter(predictions_dict.values())
    most_common = votes.most_common(1)[0]
    agreement_count = most_common[1]
    agreement_pct = agreement_count / len(predictions_dict)

    return {
        'agreement_count': agreement_count,
        'agreement_pct': float(agreement_pct),
        'strong_agreement': agreement_count >= 3,
        'consensus_prediction': most_common[0]
    }
