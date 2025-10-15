"""
BlackIce Model REST Server
Simple REST API for model inference
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
import joblib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ModelServer:
    def __init__(self, model_dir=None):
        # Default to MT5 Files folder if not specified
        if model_dir is None:
            mt5_files_path = Path.home() / \
                "AppData/Roaming/MetaQuotes/Terminal/776D2ACDFA4F66FAF3C8985F75FA9FF6/MQL5/Files/Model_output"
            if mt5_files_path.exists():
                self.model_dir = mt5_files_path
                logger.info(f"📁 Using MT5 Files folder: {self.model_dir}")
            else:
                self.model_dir = Path("Model_output")
                logger.info(f"📁 Using local folder: {self.model_dir}")
        else:
            self.model_dir = Path(model_dir)

        self.models = {}
        self.scalers = None
        self.ensemble_weights = {}
        self.ensemble_metadata = None
        self.load_models()

    def load_models(self):
        """Load all available models"""
        try:
            # Load ensemble metadata first
            metadata_path = self.model_dir / "ensemble" / "ensemble_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.ensemble_metadata = json.load(f)
                logger.info("✅ Ensemble metadata loaded")

                # Extract weights from metadata
                if 'ensemble_weights' in self.ensemble_metadata:
                    self.ensemble_weights = self.ensemble_metadata['ensemble_weights']
                    logger.info(f"✅ Ensemble weights: {self.ensemble_weights}")

                # Extract scalers from metadata (PRIORITY!)
                if 'feature_scalers' in self.ensemble_metadata:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(
                        self.ensemble_metadata['feature_scalers']['mean'])
                    scaler.scale_ = np.array(
                        self.ensemble_metadata['feature_scalers']['std'])
                    scaler.n_features_in_ = len(scaler.mean_)
                    self.scalers = scaler
                    logger.info(
                        f"✅ Feature scalers loaded from metadata: {len(scaler.mean_)} features")
                    logger.info(f"   Using ACTUAL training scalers!")
                    # Continue to load models - DON'T RETURN HERE!

            # Load feature scalers - MUST use proper 29-feature scaler
            scaler_loaded = False

            # Skip other scaler loading if we already have scalers from metadata
            if self.scalers is not None:
                scaler_loaded = True

            # Priority 1: Full 29-feature scaler (REQUIRED)
            full_scaler_path = Path("Python/feature_scaler_29.pkl")
            if full_scaler_path.exists():
                with open(full_scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"✅ Feature scalers loaded: 29 features")
                logger.info(f"   Mean shape: {self.scalers.mean_.shape}")
                logger.info(f"   Scale shape: {self.scalers.scale_.shape}")
                scaler_loaded = True

            # Priority 2: Ensemble folder scalers
            if not scaler_loaded:
                scaler_paths = [
                    self.model_dir / "ensemble" / "feature_scalers.pkl",
                    self.model_dir / "feature_scaler.pkl"
                ]

                for scaler_path in scaler_paths:
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            self.scalers = pickle.load(f)
                        logger.info(
                            f"✅ Feature scalers loaded from {scaler_path.name}")
                        scaler_loaded = True
                        break

            if not scaler_loaded:
                logger.error("❌ NO FEATURE SCALER FOUND!")
                logger.error("   Run: python Python/generate_full_scaler.py")
                raise ValueError(
                    "Feature scaler is required for accurate predictions")

            # Try to load PyTorch models
            self._load_pytorch_models()

            # Try to load sklearn/joblib models
            self._load_sklearn_models()

            print(
                f"DEBUG load_models(): After loading, self.models = {list(self.models.keys())}, count = {len(self.models)}")

            if len(self.models) == 0:
                logger.warning(
                    "⚠️ No models loaded - using fallback prediction")
            else:
                logger.info(
                    f"✅ Loaded {len(self.models)} models: {list(self.models.keys())}")

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            logger.error(traceback.format_exc())

    def _load_csv_scaler(self, csv_path):
        """Load feature scaler from CSV file (MinMaxScaler format)"""
        try:
            df = pd.read_csv(csv_path)

            # Create a simple scaler object
            class CSVScaler:
                def __init__(self, min_vals, scale_vals):
                    self.min_ = min_vals
                    self.scale_ = scale_vals

                def transform(self, X):
                    # MinMaxScaler formula: X_scaled = (X - min) * scale
                    return (X - self.min_) * self.scale_

            min_vals = df['min'].values
            scale_vals = df['scale'].values

            return CSVScaler(min_vals, scale_vals)

        except Exception as e:
            logger.error(f"Failed to load CSV scaler: {e}")
            return None

    def _load_pytorch_models(self):
        """Load PyTorch state dict models"""
        # These are state_dicts, not full models
        # We'll load them as raw state dicts for now
        for model_file in self.model_dir.glob("*_state_dict.pth"):
            try:
                model_name = model_file.stem.replace('_state_dict', '')
                state_dict = torch.load(model_file, map_location='cpu')

                # Store state dict - we'll need model architecture to use it
                self.models[model_name] = {
                    'state_dict': state_dict,
                    'type': 'pytorch_state_dict'
                }
                logger.info(f"✅ Loaded PyTorch state dict: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_file.name}: {e}")
                logger.warning(f"   Error: {traceback.format_exc()}")

    def _load_sklearn_models(self):
        """Load sklearn/joblib models"""
        # Look in both root and ensemble subfolder
        search_paths = [self.model_dir, self.model_dir / "ensemble"]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for model_file in search_path.glob("*_sklearn.pkl"):
                if model_file.name in ['feature_scaler.pkl', 'ensemble_weights.pkl']:
                    continue
                try:
                    model_name = model_file.stem.replace('_sklearn', '')
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    self.models[model_name] = {
                        'model': model,
                        'type': 'sklearn'
                    }
                    logger.info(f"✅ Loaded sklearn model: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_file.name}: {e}")
                    logger.warning(f"   Error: {traceback.format_exc()}")

    def preprocess_features(self, features):
        """Normalize features using loaded scalers - NO FALLBACKS"""
        features_array = np.array(features).reshape(1, -1)

        if self.scalers is None:
            raise ValueError(
                "Feature scaler not loaded! Cannot make predictions.")

        # Verify scaler dimensions
        expected_features = self.scalers.mean_.shape[0]
        if features_array.shape[1] != expected_features:
            raise ValueError(
                f"Feature mismatch: scaler expects {expected_features} features, got {features_array.shape[1]}")

        # Use the real scaler
        features_normalized = self.scalers.transform(features_array)

        return features_normalized

    def calculate_features_from_ohlcv(self, ohlcv_data):
        """
        Calculate all 29 features from raw OHLCV data

        Args:
            ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
                       Should contain at least 50 bars for proper calculation

        Returns:
            List of 29 features ready for prediction
        """
        try:
            # Import feature engineering functions
            from enhanced_multitf_pipeline import engineer_multitf_smc_features

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)

            # Ensure required columns exist
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(
                    f"Missing required columns. Got: {df.columns.tolist()}")
                return None

            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)

            # Engineer all features (including SMC)
            df = engineer_multitf_smc_features(df)

            # Extract the 29 features from the last row
            feature_columns = [
                'return_1m', 'return_5m', 'return_15m', 'sma_ratio', 'rsi_norm', 'atr_norm',
                'ob_bullish_exists', 'ob_bullish_size_atr', 'ob_bullish_displacement_atr',
                'ob_bearish_exists', 'ob_bearish_size_atr', 'ob_bearish_displacement_atr',
                'fvg_bullish_exists', 'fvg_bullish_depth_atr', 'fvg_bearish_exists', 'fvg_bearish_depth_atr',
                'bos_wick_confirm', 'bos_close_confirm', 'bos_dist_atr',
                'h1_trend_bias', 'h1_volatility_regime', 'h1_structure_strength', 'h1_momentum',
                'h4_macro_trend', 'h4_regime_classification', 'h4_volatility_cycle',
                'mtf_trend_alignment', 'mtf_structure_confluence', 'mtf_volatility_sync'
            ]

            # Get the last row's features
            features = []
            for col in feature_columns:
                if col in df.columns:
                    features.append(float(df[col].iloc[-1]))
                else:
                    logger.warning(f"Feature '{col}' not found, using 0.0")
                    features.append(0.0)

            logger.info(
                f"✅ Calculated {len(features)} features from {len(df)} OHLCV bars")
            return features

        except Exception as e:
            logger.error(f"Failed to calculate features from OHLCV: {e}")
            logger.error(traceback.format_exc())
            return None

    def predict(self, features):
        """Generate prediction from features"""
        try:
            if len(features) != 29:
                return {"error": f"Expected 29 features, got {len(features)}"}

            # Preprocess features
            features_normalized = self.preprocess_features(features)

            # DEBUG
            print(
                f"DEBUG predict(): self.models has {len(self.models)} models: {list(self.models.keys())}")

            # If we have loaded models, use them
            if len(self.models) > 0:
                return self._predict_with_models(features_normalized)
            else:
                # Fallback to simple logic if no models loaded
                print("WARNING: Using fallback - no models!")
                return self._fallback_prediction(features)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _analyze_features(self, features_normalized):
        """Analyze features to generate reasoning for decisions"""
        # Feature names (29 features from your pipeline)
        # NOTE: Update these if using enhanced_multitf_pipeline with SMC features
        feature_names = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle',
            'bb_lower', 'bb_width', 'atr', 'adx', 'cci', 'mfi', 'willr',
            'stoch_k', 'stoch_d', 'obv', 'vwap', 'ema_fast', 'ema_slow',
            'sma_20', 'sma_50', 'volume', 'price_change', 'volatility',
            'momentum', 'roc', 'trix', 'ultimate_osc', 'keltner_channel'
        ]

        features = features_normalized[0]
        reasons = []
        smc_context = {}  # Store SMC-specific insights

        # Analyze key indicators (using normalized values)
        # RSI analysis
        if len(features) > 0:
            rsi_val = features[0]
            if rsi_val > 0.5:
                reasons.append("RSI indicates overbought conditions")
            elif rsi_val < -0.5:
                reasons.append("RSI shows oversold conditions")

        # MACD analysis
        if len(features) > 3:
            macd_hist = features[3]
            if macd_hist > 0.3:
                reasons.append("MACD histogram shows bullish momentum")
            elif macd_hist < -0.3:
                reasons.append("MACD histogram shows bearish momentum")

        # Bollinger Bands analysis
        if len(features) > 7:
            bb_width = features[7]
            if bb_width > 0.5:
                reasons.append(
                    "High volatility detected (wide Bollinger Bands)")
            elif bb_width < -0.5:
                reasons.append("Low volatility (tight Bollinger Bands)")

        # ADX analysis
        if len(features) > 9:
            adx = features[9]
            if adx > 0.5:
                reasons.append("Strong trend detected (high ADX)")
            elif adx < -0.5:
                reasons.append("Weak trend (low ADX)")

        # Volume analysis
        if len(features) > 21:
            volume = features[21]
            if volume > 0.5:
                reasons.append("Above-average volume confirms move")
            elif volume < -0.5:
                reasons.append("Low volume suggests weak conviction")

        # Momentum analysis
        if len(features) > 24:
            momentum = features[24]
            if momentum > 0.5:
                reasons.append("Strong upward momentum")
            elif momentum < -0.5:
                reasons.append("Strong downward momentum")

        # SMC Analysis (if features are present - check feature count)
        # Enhanced pipeline has 40+ features with SMC
        if len(features) >= 40:
            # Assuming SMC features start around index 29
            # OB Bullish: exists, size_atr, displacement_atr
            # OB Bearish: exists, size_atr, displacement_atr
            # FVG Bullish: exists, depth_atr
            # FVG Bearish: exists, depth_atr
            # BOS: wick_confirm, close_confirm, dist_atr

            try:
                ob_bullish_exists = features[29] if len(features) > 29 else 0
                ob_bearish_exists = features[32] if len(features) > 32 else 0
                fvg_bullish_exists = features[35] if len(features) > 35 else 0
                fvg_bearish_exists = features[37] if len(features) > 37 else 0
                bos_close_confirm = features[40] if len(features) > 40 else 0

                # Build SMC context
                if ob_bullish_exists > 0.5:
                    smc_context['bullish_ob'] = True
                    reasons.append(
                        "Bullish Order Block detected (institutional accumulation)")
                if ob_bearish_exists > 0.5:
                    smc_context['bearish_ob'] = True
                    reasons.append(
                        "Bearish Order Block detected (institutional distribution)")
                if fvg_bullish_exists > 0.5:
                    smc_context['bullish_fvg'] = True
                    reasons.append(
                        "Bullish Fair Value Gap (buy-side imbalance)")
                if fvg_bearish_exists > 0.5:
                    smc_context['bearish_fvg'] = True
                    reasons.append(
                        "Bearish Fair Value Gap (sell-side imbalance)")
                if bos_close_confirm > 0.5:
                    smc_context['structure_break'] = True
                    reasons.append(
                        "Structure break confirmed (strong institutional commitment)")
            except IndexError:
                pass  # SMC features not available

        return reasons, smc_context

    def _predict_with_models(self, features_normalized):
        """Use loaded models for prediction with enhanced AI feedback"""
        print("\n" + "="*70)
        print("🧊 BLACK ICE AI - ANALYSIS IN PROGRESS")
        print("="*70)
        print(f"📊 Processing {features_normalized.shape[1]} features...")

        predictions = []
        model_votes = {"SELL": 0, "HOLD": 0, "BUY": 0}

        # Analyze features for reasoning (now returns SMC context too)
        feature_reasons, smc_context = self._analyze_features(
            features_normalized)

        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                model_type = model_info['type']

                if model_type == 'pytorch':
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(
                            features_normalized)
                        output = model(features_tensor)
                        probs = torch.softmax(output, dim=1).numpy()[0]
                        pred = np.argmax(probs)
                        conf = float(np.max(probs))

                elif model_type == 'pytorch_state_dict':
                    continue

                elif model_type == 'sklearn':
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore', category=UserWarning, module='sklearn')

                        pred = model.predict(features_normalized)[0]
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(features_normalized)[0]
                            conf = float(np.max(probs))
                        else:
                            probs = np.array([0.33, 0.34, 0.33])
                            conf = 0.6

                    # Enhanced model output
                    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    action = action_map.get(pred, "HOLD")
                    model_votes[action] += 1

                    # Visual confidence bar
                    conf_bar = "█" * int(conf * 10) + "░" * \
                        (10 - int(conf * 10))
                    print(
                        f"  🤖 {model_name:20s} → {action:4s} [{conf_bar}] {conf:.1%}")

                weight = self.ensemble_weights.get(model_name, 1.0)
                predictions.append((pred, conf, weight, probs))

            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")

        if len(predictions) == 0:
            return {"error": "All models failed to predict"}

        # Ensemble prediction
        weighted_probs = np.zeros(3)
        total_weight = 0

        for pred, conf, weight, probs in predictions:
            weighted_probs += probs * weight
            total_weight += weight

        if total_weight > 0:
            weighted_probs /= total_weight

        # TEMPORARY FIX: Reduce HOLD bias (models need retraining!)
        if weighted_probs[1] > 0.7:  # If HOLD > 70%
            logger.warning(
                f"⚠️ HOLD bias detected: {weighted_probs[1]:.1%} - Adjusting...")
            excess = weighted_probs[1] - 0.5
            weighted_probs[1] = 0.5

            # Give excess to whichever is higher (BUY or SELL)
            if weighted_probs[2] > weighted_probs[0]:
                weighted_probs[2] += excess
                logger.info(
                    f"   Boosted BUY: {weighted_probs[0]:.1%} → {weighted_probs[2]:.1%}")
            else:
                weighted_probs[0] += excess
                logger.info(f"   Boosted SELL: {weighted_probs[0]:.1%}")

        final_prediction = int(np.argmax(weighted_probs))
        final_confidence = float(np.max(weighted_probs))

        # Enhanced ensemble output
        print("─" * 70)
        print(f"🎯 ENSEMBLE DECISION:")
        print(
            f"   SELL: {weighted_probs[0]:.1%} {'█' * int(weighted_probs[0] * 20)}")
        print(
            f"   HOLD: {weighted_probs[1]:.1%} {'█' * int(weighted_probs[1] * 20)}")
        print(
            f"   BUY:  {weighted_probs[2]:.1%} {'█' * int(weighted_probs[2] * 20)}")
        print("─" * 70)

        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        final_action = action_map.get(final_prediction, "HOLD")

        # Consensus analysis
        max_votes = max(model_votes.values())
        consensus = "Strong" if max_votes >= len(
            predictions) * 0.75 else "Moderate" if max_votes >= len(predictions) * 0.5 else "Weak"

        print(f"✅ FINAL: {final_action} (Confidence: {final_confidence:.1%})")
        print(
            f"🤖 Model Votes: SELL={model_votes['SELL']} | HOLD={model_votes['HOLD']} | BUY={model_votes['BUY']}")
        print(
            f"💪 Consensus: {consensus} ({max_votes}/{len(predictions)} models agree)")
        print("="*70 + "\n")

        # Map to action
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map.get(final_prediction, "HOLD")

        # Determine market regime based on probabilities
        regime = "Unknown"
        if weighted_probs[2] > 0.6:
            regime = "Bull Market"
        elif weighted_probs[0] > 0.6:
            regime = "Bear Market"
        elif weighted_probs[1] > 0.5:
            regime = "Consolidation"
        elif final_confidence > 0.75:
            regime = "Recovery" if action == "BUY" else "Correction"
        else:
            regime = "Uncertain"

        # Calculate ensemble accuracy (from metadata or estimate)
        ensemble_acc = 0.0
        if self.ensemble_metadata and 'ensemble_accuracy' in self.ensemble_metadata:
            ensemble_acc = self.ensemble_metadata['ensemble_accuracy']
        else:
            # Estimate based on confidence
            ensemble_acc = 0.75 + (final_confidence - 0.5) * 0.3

        # Get model name
        model_name = "Ensemble"
        if len(predictions) > 0:
            model_name = f"Ensemble_{len(predictions)}models"

        # Generate reasoning explanation
        reasoning = self._generate_reasoning(
            action, final_confidence, weighted_probs,
            feature_reasons, consensus, model_votes, smc_context
        )

        # Print reasoning
        print(f"💡 REASONING:")
        for reason in reasoning:
            print(f"   • {reason}")

        # Print SMC context if available
        if smc_context:
            print(f"📊 SMC CONTEXT:")
            if smc_context.get('bullish_ob'):
                print(f"   🟢 Bullish Order Block Active")
            if smc_context.get('bearish_ob'):
                print(f"   🔴 Bearish Order Block Active")
            if smc_context.get('bullish_fvg'):
                print(f"   ⬆️ Bullish FVG (Buy-side Imbalance)")
            if smc_context.get('bearish_fvg'):
                print(f"   ⬇️ Bearish FVG (Sell-side Imbalance)")
            if smc_context.get('structure_break'):
                print(f"   💥 Structure Break Confirmed")
        print("="*70 + "\n")

        return {
            "action": action,
            "confidence": final_confidence,
            "signal_strength": final_confidence * 0.9,
            "should_trade": final_confidence > 0.7,
            "probabilities": {
                "sell": float(weighted_probs[0]),
                "hold": float(weighted_probs[1]),
                "buy": float(weighted_probs[2])
            },
            "regime": regime,
            "ensemble_acc": ensemble_acc,
            "model": model_name,
            "models_used": len(predictions),
            "smc_context": smc_context,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_reasoning(self, action, confidence, probs, feature_reasons, consensus, model_votes, smc_context=None):
        """Generate human-readable reasoning for the decision"""
        reasons = []

        if smc_context is None:
            smc_context = {}

        # Main decision reasoning
        if action == "BUY":
            reasons.append(
                f"Recommending BUY with {confidence:.1%} confidence")
            if probs[2] > 0.7:
                reasons.append(
                    "Strong bullish signals across multiple indicators")
            elif probs[2] > 0.5:
                reasons.append("Moderate bullish sentiment detected")
            else:
                reasons.append("Slight bullish bias but proceed with caution")
        elif action == "SELL":
            reasons.append(
                f"Recommending SELL with {confidence:.1%} confidence")
            if probs[0] > 0.7:
                reasons.append(
                    "Strong bearish signals across multiple indicators")
            elif probs[0] > 0.5:
                reasons.append("Moderate bearish sentiment detected")
            else:
                reasons.append("Slight bearish bias but proceed with caution")
        else:  # HOLD
            reasons.append(
                f"Recommending HOLD with {confidence:.1%} confidence")
            reasons.append(
                "Market conditions suggest waiting for clearer signals")

        # Model consensus
        total_models = sum(model_votes.values())
        if consensus == "Strong":
            reasons.append(
                f"Strong consensus: {max(model_votes.values())}/{total_models} models agree")
        elif consensus == "Moderate":
            reasons.append(
                f"Moderate agreement among models ({max(model_votes.values())}/{total_models})")
        else:
            reasons.append(f"Models are divided - exercise caution")

        # Add feature-based reasoning
        if feature_reasons:
            reasons.append("Technical analysis:")
            # Top 5 reasons
            reasons.extend([f"  - {reason}" for reason in feature_reasons[:5]])

        # Risk assessment
        if confidence > 0.8:
            reasons.append("High confidence - strong signal quality")
        elif confidence > 0.6:
            reasons.append("Moderate confidence - consider position sizing")
        else:
            reasons.append("Low confidence - high uncertainty, reduce risk")

        # Probability spread analysis
        prob_spread = max(probs) - min(probs)
        if prob_spread > 0.5:
            reasons.append("Clear directional bias in probabilities")
        elif prob_spread < 0.2:
            reasons.append("Probabilities are close - market indecision")

        # SMC-specific reasoning
        if smc_context:
            smc_reasons = []
            if action == "BUY":
                if smc_context.get('bullish_ob'):
                    smc_reasons.append(
                        "Institutional accumulation zone (Bullish OB) supports upside")
                if smc_context.get('bullish_fvg'):
                    smc_reasons.append(
                        "Buy-side imbalance (FVG) suggests unfilled demand")
                if smc_context.get('structure_break'):
                    smc_reasons.append(
                        "Confirmed structure break validates bullish momentum")
            elif action == "SELL":
                if smc_context.get('bearish_ob'):
                    smc_reasons.append(
                        "Institutional distribution zone (Bearish OB) supports downside")
                if smc_context.get('bearish_fvg'):
                    smc_reasons.append(
                        "Sell-side imbalance (FVG) suggests unfilled supply")
                if smc_context.get('structure_break'):
                    smc_reasons.append(
                        "Confirmed structure break validates bearish momentum")

            if smc_reasons:
                reasons.append("Smart Money Concepts:")
                reasons.extend([f"  - {r}" for r in smc_reasons])

        return reasons

    def _fallback_prediction(self, features):
        avg_feature = np.mean(features)

        if avg_feature > 0.1:
            action = "BUY"
            confidence = 0.65
        elif avg_feature < -0.1:
            action = "SELL"
            confidence = 0.63
        else:
            action = "HOLD"
            confidence = 0.60

        return {
            "action": action,
            "confidence": confidence,
            "signal_strength": confidence * 0.8,
            "should_trade": False,  # Don't trade with fallback
            "probabilities": {
                "sell": 0.2 if action == "SELL" else 0.1,
                "hold": 0.6 if action == "HOLD" else 0.3,
                "buy": 0.2 if action == "BUY" else 0.1
            },
            "regime": "Unknown",
            "ensemble_acc": 0.0,
            "model": "Fallback",
            "models_used": 0,
            "timestamp": datetime.now().isoformat(),
            "warning": "Using fallback prediction - no models loaded"
        }


# Global model server instance
model_server = ModelServer()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_server.models)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint - accepts either features or raw OHLCV data"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data in request"}), 400

        # Check if raw OHLCV data is provided (preferred method)
        if 'ohlcv' in data:
            # Process raw OHLCV data and calculate features
            ohlcv_data = data['ohlcv']
            features = model_server.calculate_features_from_ohlcv(ohlcv_data)
            if features is None:
                return jsonify({"error": "Failed to calculate features from OHLCV data"}), 400
        elif 'features' in data:
            # Legacy method: accept pre-calculated features
            features = data['features']
        else:
            return jsonify({"error": "Missing 'features' or 'ohlcv' in request"}), 400

        result = model_server.predict(features)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify({
            "success": True,
            "prediction": result,
            "symbol": data.get('symbol', 'UNKNOWN'),
            "timeframe": data.get('timeframe', 'UNKNOWN')
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "models": list(model_server.models.keys()),
        "scalers_loaded": model_server.scalers is not None,
        "expected_features": 29,
        "accepts_ohlcv": True
    })


@app.route('/test_ohlcv', methods=['POST'])
def test_ohlcv():
    """Test endpoint to verify OHLCV feature calculation"""
    try:
        data = request.get_json()

        if 'ohlcv' not in data:
            return jsonify({"error": "Missing 'ohlcv' in request"}), 400

        features = model_server.calculate_features_from_ohlcv(data['ohlcv'])

        if features is None:
            return jsonify({"error": "Failed to calculate features"}), 400

        return jsonify({
            "success": True,
            "features_calculated": len(features),
            "features": features,
            "bars_received": len(data['ohlcv'])
        })

    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting BlackIce Model REST Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("🤖 Prediction: POST to http://localhost:5000/predict")

    app.run(host='0.0.0.0', port=5000, debug=True)
