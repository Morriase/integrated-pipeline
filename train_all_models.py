"""
Training Orchestrator for All SMC Models

Trains all model types per symbol and evaluates performance
Following specifications in WHATS_NEEDED.md
"""

import warnings
from typing import Dict, List
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add pipeline directory to path FIRST
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

# Now import models
from models.random_forest_model import RandomForestSMCModel
from models.xgboost_model import XGBoostSMCModel, XGBOOST_AVAILABLE
from models.neural_network_model import NeuralNetworkSMCModel, TORCH_AVAILABLE
from models.lstm_model import LSTMSMCModel

warnings.filterwarnings('ignore')

# Import model classes


class ModelSelector:
    """
    Automatic selection of best-performing models per symbol.
    
    Filters models based on quality thresholds:
    - Train-val gap < max_gap (default 20%)
    - Test accuracy > min_accuracy (default 55%)
    - Val-test consistency within stability_threshold (default 5%)
    
    Scores models with overfitting penalty and selects best per symbol.
    """
    
    def __init__(self, max_gap: float = 0.20, min_accuracy: float = 0.55, 
                 stability_threshold: float = 0.05):
        """
        Initialize ModelSelector with quality thresholds.
        
        Args:
            max_gap: Maximum acceptable train-val gap (default 0.20 = 20%)
            min_accuracy: Minimum acceptable test accuracy (default 0.55 = 55%)
            stability_threshold: Maximum acceptable val-test difference (default 0.05 = 5%)
        """
        self.max_gap = max_gap
        self.min_accuracy = min_accuracy
        self.stability_threshold = stability_threshold
        
        print(f"\n🎯 ModelSelector initialized with thresholds:")
        print(f"  Max Train-Val Gap:     {self.max_gap:.1%}")
        print(f"  Min Test Accuracy:     {self.min_accuracy:.1%}")
        print(f"  Max Val-Test Diff:     {self.stability_threshold:.1%}")
    
    def select_best_models(self, results: Dict) -> Dict:
        """
        Select best model per symbol based on quality criteria.
        
        Filtering criteria:
        1. Train-val gap must be <= max_gap
        2. Test accuracy must be >= min_accuracy
        3. Val-test difference must be <= stability_threshold
        
        Scoring: test_accuracy - (train_val_gap * 0.5)
        This penalizes overfitting while rewarding high accuracy.
        
        Args:
            results: Dictionary of training results
                     Format: {symbol: {model_name: metrics_dict}}
        
        Returns:
            Dictionary of selections per symbol
            Format: {
                symbol: {
                    'selected_model': model_name or None,
                    'test_accuracy': float,
                    'train_val_gap': float,
                    'reason': str,
                    'alternatives': [model_names],
                    'action': str (if manual review needed)
                }
            }
        """
        print(f"\n{'='*80}")
        print("MODEL SELECTION ANALYSIS")
        print(f"{'='*80}")
        
        selections = {}
        
        for symbol, symbol_results in results.items():
            print(f"\n📊 Analyzing models for {symbol}...")
            
            candidates = []
            rejected = []
            
            for model_name, metrics in symbol_results.items():
                # Skip models with errors
                if 'error' in metrics:
                    rejected.append({
                        'model': model_name,
                        'reason': f"Training error: {metrics['error']}"
                    })
                    continue
                
                # Extract metrics
                history = metrics.get('history', {})
                val_metrics = metrics.get('val_metrics', {})
                test_metrics = metrics.get('test_metrics', {})
                
                test_acc = test_metrics.get('accuracy', 0)
                val_acc = val_metrics.get('accuracy', 0)
                train_acc = history.get('train_accuracy', 0)
                train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                
                # Apply filters
                rejection_reasons = []
                
                if train_val_gap > self.max_gap:
                    rejection_reasons.append(f"Train-val gap too high ({train_val_gap:.1%} > {self.max_gap:.1%})")
                
                if test_acc < self.min_accuracy:
                    rejection_reasons.append(f"Test accuracy too low ({test_acc:.1%} < {self.min_accuracy:.1%})")
                
                # Check val-test consistency
                val_test_diff = abs(val_acc - test_acc)
                if val_test_diff > self.stability_threshold:
                    rejection_reasons.append(f"Val-test inconsistency ({val_test_diff:.1%} > {self.stability_threshold:.1%})")
                
                if rejection_reasons:
                    rejected.append({
                        'model': model_name,
                        'reason': '; '.join(rejection_reasons),
                        'test_acc': test_acc,
                        'train_val_gap': train_val_gap
                    })
                    continue
                
                # Model passed all filters - add to candidates
                score = test_acc - (train_val_gap * 0.5)  # Penalize overfitting
                
                candidates.append({
                    'model': model_name,
                    'test_accuracy': test_acc,
                    'val_accuracy': val_acc,
                    'train_val_gap': train_val_gap,
                    'val_test_diff': val_test_diff,
                    'score': score
                })
                
                print(f"  ✓ {model_name}: test_acc={test_acc:.3f}, gap={train_val_gap:.1%}, score={score:.3f}")
            
            # Print rejected models
            if rejected:
                print(f"\n  ✗ Rejected models:")
                for r in rejected:
                    print(f"    - {r['model']}: {r['reason']}")
            
            # Select best model or flag for manual review
            if not candidates:
                selections[symbol] = {
                    'selected_model': None,
                    'reason': 'No models met quality criteria',
                    'action': 'MANUAL_REVIEW_REQUIRED',
                    'rejected_models': rejected
                }
                print(f"\n  ⚠️  No models passed quality filters - MANUAL REVIEW REQUIRED")
                continue
            
            # Select best by score
            best = max(candidates, key=lambda x: x['score'])
            alternatives = [c['model'] for c in candidates if c['model'] != best['model']]
            
            selections[symbol] = {
                'selected_model': best['model'],
                'test_accuracy': best['test_accuracy'],
                'val_accuracy': best['val_accuracy'],
                'train_val_gap': best['train_val_gap'],
                'val_test_diff': best['val_test_diff'],
                'score': best['score'],
                'reason': f"Best score ({best['score']:.3f}) with gap {best['train_val_gap']:.1%}",
                'alternatives': alternatives
            }
            
            print(f"\n  🏆 Selected: {best['model']}")
            print(f"     Test Accuracy: {best['test_accuracy']:.3f}")
            print(f"     Train-Val Gap: {best['train_val_gap']:.1%}")
            print(f"     Score: {best['score']:.3f}")
            if alternatives:
                print(f"     Alternatives: {', '.join(alternatives)}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SELECTION SUMMARY")
        print(f"{'='*80}")
        
        total_symbols = len(selections)
        models_selected = sum(1 for s in selections.values() if s['selected_model'])
        manual_review = sum(1 for s in selections.values() if not s['selected_model'])
        
        print(f"\n  Total Symbols:         {total_symbols}")
        print(f"  Models Selected:       {models_selected}")
        print(f"  Manual Review Needed:  {manual_review}")
        
        if manual_review > 0:
            print(f"\n  ⚠️  Symbols requiring manual review:")
            for symbol, selection in selections.items():
                if not selection['selected_model']:
                    print(f"    - {symbol}")
        
        return selections
    
    def save_deployment_manifest(self, selections: Dict, output_path: str) -> Dict:
        """
        Save deployment manifest JSON with selection results.
        
        The manifest includes:
        - Timestamp of selection
        - Selection criteria used
        - Per-symbol selections with metrics
        - Summary statistics
        
        Args:
            selections: Dictionary from select_best_models()
            output_path: Path to save manifest JSON file
        
        Returns:
            Complete manifest dictionary
        """
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'selection_criteria': {
                'max_train_val_gap': self.max_gap,
                'min_test_accuracy': self.min_accuracy,
                'max_val_test_diff': self.stability_threshold
            },
            'selections': selections,
            'summary': {
                'total_symbols': len(selections),
                'models_selected': sum(1 for s in selections.values() if s['selected_model']),
                'manual_review_needed': sum(1 for s in selections.values() if not s['selected_model']),
                'selected_model_types': {}
            }
        }
        
        # Count model types selected
        for symbol, selection in selections.items():
            if selection['selected_model']:
                model_type = selection['selected_model']
                manifest['summary']['selected_model_types'][model_type] = \
                    manifest['summary']['selected_model_types'].get(model_type, 0) + 1
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"\n💾 Deployment manifest saved to: {output_path}")
        print(f"\n📋 Manifest Summary:")
        print(f"  Timestamp: {manifest['timestamp']}")
        print(f"  Total Symbols: {manifest['summary']['total_symbols']}")
        print(f"  Models Selected: {manifest['summary']['models_selected']}")
        print(f"  Manual Review Needed: {manifest['summary']['manual_review_needed']}")
        
        if manifest['summary']['selected_model_types']:
            print(f"\n  Selected Model Types:")
            for model_type, count in manifest['summary']['selected_model_types'].items():
                print(f"    - {model_type}: {count}")
        
        return manifest


class SMCModelTrainer:
    """
    Orchestrates training of all SMC models

    Models trained:
    1. Random Forest - Feature importance and threshold rules
    2. XGBoost - Gradient boosting for high accuracy
    3. Neural Network - Non-linear feature interactions
    4. LSTM - Temporal sequence patterns
    """

    def __init__(self, data_dir: str = '/kaggle/working', output_dir: str = 'models/trained'):
        """
        Initialize trainer

        Args:
            data_dir: Directory containing processed data splits
            output_dir: Directory to save trained models
        """
        # Kaggle paths - data is now in /kaggle/working after pipeline runs
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data paths
        self.train_path = self.data_dir / 'processed_smc_data_train.csv'
        self.val_path = self.data_dir / 'processed_smc_data_val.csv'
        self.test_path = self.data_dir / 'processed_smc_data_test.csv'

        # Results storage
        self.results = {}

    def check_data_availability(self) -> bool:
        """Check if processed data files exist"""
        print("\n🔍 Checking data availability...")

        files_exist = all([
            self.train_path.exists(),
            self.val_path.exists(),
            self.test_path.exists()
        ])

        if files_exist:
            print(f"  ✓ Train data: {self.train_path}")
            print(f"  ✓ Val data:   {self.val_path}")
            print(f"  ✓ Test data:  {self.test_path}")
            return True
        else:
            print(f"  ✗ Data files not found in {self.data_dir}")
            print(
                f"\n  Run 'python run_complete_pipeline.py' first to generate training data")
            return False

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols in the dataset"""
        df = pd.read_csv(self.train_path)
        symbols = df['symbol'].unique().tolist()
        print(f"\n📊 Available symbols: {symbols}")
        return symbols

    def train_random_forest(self, symbol: str, exclude_timeout: bool = False) -> Dict:
        """Train Random Forest model"""
        print(f"\n{'='*80}")
        print(f"Training Random Forest for {symbol}")
        print(f"{'='*80}")

        model = RandomForestSMCModel(symbol=symbol)

        # Load data
        train_df, val_df, test_df = model.load_data(
            str(self.train_path), str(self.val_path), str(self.test_path),
            exclude_timeout=exclude_timeout
        )

        # Prepare features (fit imputer on training data)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train
        history = model.train(
            X_train, y_train, X_val, y_val,
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            use_grid_search=False
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'RandomForest',
            'symbol': symbol,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(top_n=20).to_dict('records')
        }

    def train_xgboost(self, symbol: str, exclude_timeout: bool = False) -> Dict:
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            print(f"\n⚠️ XGBoost not available, skipping...")
            return {}

        print(f"\n{'='*80}")
        print(f"Training XGBoost for {symbol}")
        print(f"{'='*80}")

        model = XGBoostSMCModel(symbol=symbol)

        # Load data
        train_df, val_df, test_df = model.load_data(
            str(self.train_path), str(self.val_path), str(self.test_path),
            exclude_timeout=exclude_timeout
        )

        # Prepare features (fit imputer on training data)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train
        history = model.train(
            X_train, y_train, X_val, y_val,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=20,
            use_gpu=False
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'XGBoost',
            'symbol': symbol,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(top_n=20).to_dict('records')
        }

    def train_neural_network(self, symbol: str, exclude_timeout: bool = False) -> Dict:
        """Train Neural Network model"""
        if not TORCH_AVAILABLE:
            print(f"\n⚠️ PyTorch not available, skipping...")
            return {}

        print(f"\n{'='*80}")
        print(f"Training Neural Network for {symbol}")
        print(f"{'='*80}")

        model = NeuralNetworkSMCModel(symbol=symbol)

        # Load data
        train_df, val_df, test_df = model.load_data(
            str(self.train_path), str(self.val_path), str(self.test_path),
            exclude_timeout=exclude_timeout
        )

        # Prepare features (fit scaler on training data only)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train with optimized parameters (from Optimizing_MLP_and_LSTM.txt)
        history = model.train(
            X_train, y_train, X_val, y_val,
            hidden_dims=[512, 256, 128, 64],  # Deeper architecture
            dropout=0.4,  # Increased dropout
            learning_rate=0.01,  # Higher max LR for One-Cycle
            batch_size=32,  # Smaller batch
            epochs=200,  # More epochs for One-Cycle
            patience=30,  # More patience
            weight_decay=0.01  # AdamW weight decay
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'NeuralNetwork',
            'symbol': symbol,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

    def train_lstm(self, symbol: str, exclude_timeout: bool = False) -> Dict:
        """Train LSTM model"""
        if not TORCH_AVAILABLE:
            print(f"\n⚠️ PyTorch not available, skipping...")
            return {}

        print(f"\n{'='*80}")
        print(f"Training LSTM for {symbol}")
        print(f"{'='*80}")

        model = LSTMSMCModel(symbol=symbol, lookback=20)

        # Load data
        train_df, val_df, test_df = model.load_data(
            str(self.train_path), str(self.val_path), str(self.test_path),
            exclude_timeout=exclude_timeout
        )

        # Prepare features (fit scaler on training data only)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train with optimized parameters (from Optimizing_MLP_and_LSTM.txt)
        history = model.train(
            X_train, y_train, X_val, y_val,
            hidden_dim=256,  # Larger hidden dimension
            num_layers=3,  # Deeper stacked LSTM
            dropout=0.4,  # Variational dropout
            learning_rate=0.01,  # Higher max LR for One-Cycle
            batch_size=16,  # Smaller batch for sequences
            epochs=200,  # More epochs
            patience=30,  # More patience
            weight_decay=0.01,  # AdamW weight decay
            bidirectional=True  # Use BiLSTM
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'LSTM',
            'symbol': symbol,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

    def train_all_for_symbol(self, symbol: str, exclude_timeout: bool = False,
                             models: List[str] = None) -> Dict:
        """
        Train all models for a specific symbol with comprehensive error handling.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            exclude_timeout: Whether to exclude timeout samples
            models: List of model names to train (None = all)

        Returns:
            Dictionary of results for each model
        """
        print(f"\n{'#'*80}")
        print(f"# Training All Models for {symbol}")
        print(f"{'#'*80}")

        if models is None:
            models = ['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']

        results = {}
        training_start_time = datetime.now()

        # Train each model with comprehensive error handling
        if 'RandomForest' in models:
            try:
                print(f"\n🌲 Starting RandomForest training for {symbol}...")
                model_start = datetime.now()
                results['RandomForest'] = self.train_random_forest(
                    symbol, exclude_timeout)
                model_duration = (datetime.now() - model_start).total_seconds()
                results['RandomForest']['training_duration_seconds'] = model_duration
                print(f"✅ RandomForest completed in {model_duration:.1f}s")
            except Exception as e:
                error_msg = f"RandomForest training failed: {str(e)}"
                print(f"\n❌ {error_msg}")
                print(f"   Symbol: {symbol}")
                print(f"   Error Type: {type(e).__name__}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                results['RandomForest'] = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
                # Continue training other models

        if 'XGBoost' in models:
            if not XGBOOST_AVAILABLE:
                print(f"\n⚠️ XGBoost not available - skipping")
                results['XGBoost'] = {
                    'error': 'XGBoost library not available',
                    'error_type': 'DependencyError'
                }
            else:
                try:
                    print(f"\n🚀 Starting XGBoost training for {symbol}...")
                    model_start = datetime.now()
                    results['XGBoost'] = self.train_xgboost(
                        symbol, exclude_timeout)
                    model_duration = (datetime.now() - model_start).total_seconds()
                    results['XGBoost']['training_duration_seconds'] = model_duration
                    print(f"✅ XGBoost completed in {model_duration:.1f}s")
                except Exception as e:
                    error_msg = f"XGBoost training failed: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Error Type: {type(e).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    results['XGBoost'] = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }
                    # Continue training other models

        if 'NeuralNetwork' in models:
            if not TORCH_AVAILABLE:
                print(f"\n⚠️ PyTorch not available - skipping Neural Network")
                results['NeuralNetwork'] = {
                    'error': 'PyTorch library not available',
                    'error_type': 'DependencyError'
                }
            else:
                try:
                    print(f"\n🧠 Starting Neural Network training for {symbol}...")
                    model_start = datetime.now()
                    results['NeuralNetwork'] = self.train_neural_network(
                        symbol, exclude_timeout)
                    model_duration = (datetime.now() - model_start).total_seconds()
                    results['NeuralNetwork']['training_duration_seconds'] = model_duration
                    print(f"✅ Neural Network completed in {model_duration:.1f}s")
                except Exception as e:
                    error_msg = f"Neural Network training failed: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Error Type: {type(e).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    results['NeuralNetwork'] = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }
                    # Continue training other models

        if 'LSTM' in models:
            if not TORCH_AVAILABLE:
                print(f"\n⚠️ PyTorch not available - skipping LSTM")
                results['LSTM'] = {
                    'error': 'PyTorch library not available',
                    'error_type': 'DependencyError'
                }
            else:
                try:
                    print(f"\n🔄 Starting LSTM training for {symbol}...")
                    model_start = datetime.now()
                    results['LSTM'] = self.train_lstm(symbol, exclude_timeout)
                    model_duration = (datetime.now() - model_start).total_seconds()
                    results['LSTM']['training_duration_seconds'] = model_duration
                    print(f"✅ LSTM completed in {model_duration:.1f}s")
                except Exception as e:
                    error_msg = f"LSTM training failed: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Error Type: {type(e).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    results['LSTM'] = {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }
                    # Continue training other models

        # Calculate total training time for symbol
        total_duration = (datetime.now() - training_start_time).total_seconds()
        
        # Count successful vs failed models
        successful_models = sum(1 for r in results.values() if 'error' not in r)
        failed_models = sum(1 for r in results.values() if 'error' in r)
        
        print(f"\n{'='*80}")
        print(f"Symbol {symbol} Training Summary:")
        print(f"  Total Duration: {total_duration:.1f}s")
        print(f"  Successful Models: {successful_models}/{len(results)}")
        print(f"  Failed Models: {failed_models}/{len(results)}")
        print(f"{'='*80}")

        self.results[symbol] = results

        return results

    def train_with_cross_validation(self, symbol: str, model_type: str, 
                                    exclude_timeout: bool = False) -> Dict:
        """
        Train model with cross-validation workflow.
        
        Performs stratified k-fold cross-validation before final training,
        reports mean and std of CV metrics, flags unstable models, and
        includes CV results in training summary.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            model_type: Model type ('RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM')
            exclude_timeout: Whether to exclude timeout samples
            
        Returns:
            Dictionary containing CV results and final model metrics
        """
        print(f"\n{'='*80}")
        print(f"Training {model_type} for {symbol} with Cross-Validation")
        print(f"{'='*80}")
        
        # Initialize model based on type
        if model_type == 'RandomForest':
            model = RandomForestSMCModel(symbol=symbol)
        elif model_type == 'XGBoost':
            if not XGBOOST_AVAILABLE:
                print(f"\n⚠️ XGBoost not available, skipping...")
                return {'error': 'XGBoost not available'}
            model = XGBoostSMCModel(symbol=symbol)
        elif model_type == 'NeuralNetwork':
            if not TORCH_AVAILABLE:
                print(f"\n⚠️ PyTorch not available, skipping...")
                return {'error': 'PyTorch not available'}
            model = NeuralNetworkSMCModel(symbol=symbol)
        elif model_type == 'LSTM':
            if not TORCH_AVAILABLE:
                print(f"\n⚠️ PyTorch not available, skipping...")
                return {'error': 'PyTorch not available'}
            model = LSTMSMCModel(symbol=symbol, lookback=20)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load data
        train_df, val_df, test_df = model.load_data(
            str(self.train_path), str(self.val_path), str(self.test_path),
            exclude_timeout=exclude_timeout
        )
        
        # Prepare features (fit scaler on training data)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
        
        print(f"\n📊 Dataset sizes:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Perform stratified k-fold cross-validation
        print(f"\n🔄 Performing 5-fold stratified cross-validation...")
        cv_results = model.cross_validate(X_train, y_train, n_folds=5, stratified=True)
        
        # Report CV metrics
        print(f"\n📈 Cross-Validation Results:")
        print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f}")
        print(f"  Std Accuracy:  {cv_results['std_accuracy']:.4f}")
        print(f"  Fold Accuracies: {[f'{acc:.4f}' for acc in cv_results['fold_accuracies']]}")
        
        # Flag models with high variance (std > 0.15)
        is_stable = cv_results['std_accuracy'] < 0.15
        cv_results['is_stable'] = is_stable
        
        if not is_stable:
            print(f"\n⚠️  WARNING: Model shows high variance (std > 0.15)")
            print(f"  This indicates unstable performance across folds.")
            print(f"  Consider:")
            print(f"    - Collecting more training data")
            print(f"    - Simplifying model architecture")
            print(f"    - Applying stronger regularization")
        else:
            print(f"\n✅ Model shows stable performance across folds (std < 0.15)")
        
        # Train final model on full training set
        print(f"\n🎯 Training final model on full training set...")
        
        if model_type == 'RandomForest':
            history = model.train(
                X_train, y_train, X_val, y_val,
                n_estimators=200,
                max_depth=15,  # Anti-overfitting constraint
                min_samples_split=20,  # Anti-overfitting constraint
                min_samples_leaf=10,  # Anti-overfitting constraint
                use_grid_search=False
            )
        elif model_type == 'XGBoost':
            history = model.train(
                X_train, y_train, X_val, y_val,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                early_stopping_rounds=20,
                use_gpu=False
            )
        elif model_type == 'NeuralNetwork':
            history = model.train(
                X_train, y_train, X_val, y_val,
                hidden_dims=[256, 128, 64],  # Anti-overfitting: reduced from [512, 256, 128, 64]
                dropout=0.5,  # Anti-overfitting: increased from 0.4
                learning_rate=0.005,  # Anti-overfitting: reduced from 0.01
                batch_size=64,  # Anti-overfitting: increased from 32
                epochs=200,
                patience=20,  # Anti-overfitting: increased from 15
                weight_decay=0.1  # Anti-overfitting: increased from 0.01
            )
        elif model_type == 'LSTM':
            history = model.train(
                X_train, y_train, X_val, y_val,
                hidden_dim=256,
                num_layers=3,
                dropout=0.4,
                learning_rate=0.01,
                batch_size=16,
                epochs=200,
                patience=30,
                weight_decay=0.01,
                bidirectional=True
            )
        
        # Add CV results to training history
        history['cv_mean_accuracy'] = cv_results['mean_accuracy']
        history['cv_std_accuracy'] = cv_results['std_accuracy']
        history['cv_fold_accuracies'] = cv_results['fold_accuracies']
        history['cv_is_stable'] = cv_results['is_stable']
        
        # Evaluate on validation and test sets
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')
        
        # Save model
        model.save_model(str(self.output_dir))
        
        # Compile results
        results = {
            'model_name': model_type,
            'symbol': symbol,
            'cv_results': cv_results,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        # Add feature importance for tree-based models
        if model_type in ['RandomForest', 'XGBoost']:
            results['feature_importance'] = model.get_feature_importance(top_n=20).to_dict('records')
        
        print(f"\n✅ Training complete for {symbol} - {model_type}")
        print(f"  CV Mean Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"  Val Accuracy:      {val_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"  Model Stability:   {'✅ Stable' if cv_results['is_stable'] else '⚠️ Unstable'}")
        
        return results

    def generate_overfitting_report(self, output_path: str = None):
        """
        Generate comprehensive overfitting analysis for all trained models.
        
        Analyzes train-val gaps across all models and symbols, identifies
        problematic models with gap > 15%, and creates detailed reports
        with visualizations.
        
        Args:
            output_path: Base path for saving reports (default: output_dir)
        """
        if output_path is None:
            output_path = str(self.output_dir)
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("OVERFITTING ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Collect overfitting metrics from all models
        overfitting_data = []
        problematic_models = []
        
        for symbol, symbol_results in self.results.items():
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    continue
                
                # Extract metrics
                history = result.get('history', {})
                val_metrics = result.get('val_metrics', {})
                test_metrics = result.get('test_metrics', {})
                
                # Calculate train-val gap
                train_acc = history.get('train_accuracy', 0)
                val_acc = history.get('val_accuracy', val_metrics.get('accuracy', 0))
                train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                
                # Check if overfitting is detected
                is_overfitting = history.get('overfitting_detected', train_val_gap > 0.15)
                
                model_data = {
                    'symbol': symbol,
                    'model': model_name,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_metrics.get('accuracy', 0),
                    'train_val_gap': train_val_gap,
                    'is_overfitting': is_overfitting,
                    'cv_mean_accuracy': history.get('cv_mean_accuracy', None),
                    'cv_std_accuracy': history.get('cv_std_accuracy', None),
                    'cv_is_stable': history.get('cv_is_stable', None)
                }
                
                overfitting_data.append(model_data)
                
                if is_overfitting:
                    problematic_models.append(model_data)
        
        # Print console summary
        print(f"\n📊 Total Models Analyzed: {len(overfitting_data)}")
        print(f"⚠️  Models with Overfitting (gap > 15%): {len(problematic_models)}")
        
        if overfitting_data:
            avg_gap = np.mean([m['train_val_gap'] for m in overfitting_data])
            print(f"📈 Average Train-Val Gap: {avg_gap:.2%}")
        
        # Print detailed table
        print(f"\n{'Symbol':<10} {'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10} {'Status':<15}")
        print("-" * 85)
        
        for data in sorted(overfitting_data, key=lambda x: x['train_val_gap'], reverse=True):
            status = "⚠ OVERFITTING" if data['is_overfitting'] else "✓ Healthy"
            print(f"{data['symbol']:<10} {data['model']:<20} "
                  f"{data['train_accuracy']:<12.3f} {data['val_accuracy']:<12.3f} "
                  f"{data['train_val_gap']:<10.2%} {status:<15}")
        
        # Highlight problematic models
        if problematic_models:
            print(f"\n{'='*80}")
            print("⚠️  MODELS REQUIRING ATTENTION (Gap > 15%)")
            print(f"{'='*80}")
            
            for data in sorted(problematic_models, key=lambda x: x['train_val_gap'], reverse=True):
                print(f"\n{data['symbol']} - {data['model']}:")
                print(f"  Train Accuracy: {data['train_accuracy']:.3f}")
                print(f"  Val Accuracy:   {data['val_accuracy']:.3f}")
                print(f"  Test Accuracy:  {data['test_accuracy']:.3f}")
                print(f"  Train-Val Gap:  {data['train_val_gap']:.2%}")
                
                if data['cv_mean_accuracy'] is not None:
                    print(f"  CV Mean Acc:    {data['cv_mean_accuracy']:.3f} ± {data['cv_std_accuracy']:.3f}")
                    print(f"  CV Stable:      {data['cv_is_stable']}")
        
        # Generate summary visualization
        self._generate_overfitting_visualization(overfitting_data, output_path)
        
        # Save comprehensive JSON report
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(overfitting_data),
                'overfitting_models': len(problematic_models),
                'average_gap': float(np.mean([m['train_val_gap'] for m in overfitting_data])) if overfitting_data else 0,
                'max_gap': float(max([m['train_val_gap'] for m in overfitting_data])) if overfitting_data else 0,
                'min_gap': float(min([m['train_val_gap'] for m in overfitting_data])) if overfitting_data else 0
            },
            'all_models': overfitting_data,
            'problematic_models': problematic_models
        }
        
        json_path = output_path / 'overfitting_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        print(f"\n💾 JSON report saved to: {json_path}")
        
        # Save markdown report
        self._generate_markdown_report(json_report, output_path)
        
        print(f"\n✅ Overfitting analysis complete!")
        
        return json_report
    
    def _generate_overfitting_visualization(self, overfitting_data: List[Dict], output_path: Path):
        """
        Generate summary visualizations comparing models.
        
        Args:
            overfitting_data: List of model metrics dictionaries
            output_path: Directory to save visualizations
        """
        if not overfitting_data:
            print("No data available for visualization")
            return
        
        import matplotlib.pyplot as plt
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overfitting Analysis - All Models', fontsize=16, fontweight='bold')
        
        # Prepare data
        symbols = [d['symbol'] for d in overfitting_data]
        models = [d['model'] for d in overfitting_data]
        labels = [f"{s}-{m}" for s, m in zip(symbols, models)]
        train_accs = [d['train_accuracy'] for d in overfitting_data]
        val_accs = [d['val_accuracy'] for d in overfitting_data]
        gaps = [d['train_val_gap'] for d in overfitting_data]
        
        # 1. Train vs Val Accuracy Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width/2, train_accs, width, label='Train Acc', color='blue', alpha=0.7)
        ax1.bar(x + width/2, val_accs, width, label='Val Acc', color='red', alpha=0.7)
        ax1.set_xlabel('Model', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Train vs Validation Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Train-Val Gap Bar Chart
        ax2 = axes[0, 1]
        colors = ['red' if g > 0.15 else 'green' for g in gaps]
        ax2.bar(x, gaps, color=colors, alpha=0.7)
        ax2.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Threshold (15%)')
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_ylabel('Train-Val Gap', fontsize=10)
        ax2.set_title('Train-Validation Gap by Model', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Gap Distribution Histogram
        ax3 = axes[1, 0]
        ax3.hist(gaps, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0.15, color='red', linestyle='--', linewidth=2, label='Overfitting Threshold')
        ax3.set_xlabel('Train-Val Gap', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Distribution of Train-Val Gaps', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Model Type Comparison
        ax4 = axes[1, 1]
        model_types = {}
        for data in overfitting_data:
            model_type = data['model']
            if model_type not in model_types:
                model_types[model_type] = {'gaps': [], 'train_accs': [], 'val_accs': []}
            model_types[model_type]['gaps'].append(data['train_val_gap'])
            model_types[model_type]['train_accs'].append(data['train_accuracy'])
            model_types[model_type]['val_accs'].append(data['val_accuracy'])
        
        model_names = list(model_types.keys())
        avg_gaps = [np.mean(model_types[m]['gaps']) for m in model_names]
        colors_by_type = ['red' if g > 0.15 else 'green' for g in avg_gaps]
        
        ax4.bar(model_names, avg_gaps, color=colors_by_type, alpha=0.7)
        ax4.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Threshold (15%)')
        ax4.set_xlabel('Model Type', fontsize=10)
        ax4.set_ylabel('Average Train-Val Gap', fontsize=10)
        ax4.set_title('Average Gap by Model Type', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        viz_path = output_path / 'overfitting_analysis.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {viz_path}")
    
    def _generate_markdown_report(self, json_report: Dict, output_path: Path):
        """
        Generate markdown report for overfitting analysis.
        
        Args:
            json_report: Dictionary containing overfitting analysis data
            output_path: Directory to save markdown report
        """
        from datetime import datetime
        
        md_path = output_path / 'overfitting_report.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Overfitting Analysis Report\n\n")
            f.write(f"**Generated:** {json_report['timestamp']}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            summary = json_report['summary']
            f.write(f"- **Total Models Analyzed:** {summary['total_models']}\n")
            f.write(f"- **Models with Overfitting (gap > 15%):** {summary['overfitting_models']}\n")
            f.write(f"- **Average Train-Val Gap:** {summary['average_gap']:.2%}\n")
            f.write(f"- **Maximum Gap:** {summary['max_gap']:.2%}\n")
            f.write(f"- **Minimum Gap:** {summary['min_gap']:.2%}\n\n")
            
            # Problematic models section
            if json_report['problematic_models']:
                f.write("## ⚠️ Models Requiring Attention\n\n")
                f.write("The following models show signs of overfitting (train-val gap > 15%):\n\n")
                
                for data in sorted(json_report['problematic_models'], 
                                 key=lambda x: x['train_val_gap'], reverse=True):
                    f.write(f"### {data['symbol']} - {data['model']}\n\n")
                    f.write(f"- **Train Accuracy:** {data['train_accuracy']:.3f}\n")
                    f.write(f"- **Validation Accuracy:** {data['val_accuracy']:.3f}\n")
                    f.write(f"- **Test Accuracy:** {data['test_accuracy']:.3f}\n")
                    f.write(f"- **Train-Val Gap:** {data['train_val_gap']:.2%} ⚠️\n")
                    
                    if data['cv_mean_accuracy'] is not None:
                        f.write(f"- **Cross-Validation Mean:** {data['cv_mean_accuracy']:.3f} ± {data['cv_std_accuracy']:.3f}\n")
                        f.write(f"- **Cross-Validation Stable:** {data['cv_is_stable']}\n")
                    
                    f.write("\n**Recommendations:**\n")
                    f.write("- Apply stronger regularization\n")
                    f.write("- Reduce model complexity\n")
                    f.write("- Increase training data or apply data augmentation\n")
                    f.write("- Consider feature selection to reduce dimensionality\n\n")
            else:
                f.write("## ✅ All Models Healthy\n\n")
                f.write("No models show significant overfitting. All train-val gaps are below 15%.\n\n")
            
            # All models table
            f.write("## All Models Performance\n\n")
            f.write("| Symbol | Model | Train Acc | Val Acc | Test Acc | Gap | Status |\n")
            f.write("|--------|-------|-----------|---------|----------|-----|--------|\n")
            
            for data in sorted(json_report['all_models'], 
                             key=lambda x: x['train_val_gap'], reverse=True):
                status = "⚠️ Overfitting" if data['is_overfitting'] else "✅ Healthy"
                f.write(f"| {data['symbol']} | {data['model']} | "
                       f"{data['train_accuracy']:.3f} | {data['val_accuracy']:.3f} | "
                       f"{data['test_accuracy']:.3f} | {data['train_val_gap']:.2%} | {status} |\n")
            
            f.write("\n## Visualization\n\n")
            f.write("![Overfitting Analysis](overfitting_analysis.png)\n\n")
            
            f.write("## Interpretation Guide\n\n")
            f.write("- **Train-Val Gap < 10%:** Excellent generalization\n")
            f.write("- **Train-Val Gap 10-15%:** Good generalization, acceptable\n")
            f.write("- **Train-Val Gap > 15%:** Overfitting detected, requires attention\n")
            f.write("- **Train-Val Gap > 25%:** Severe overfitting, immediate action needed\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review models flagged with overfitting\n")
            f.write("2. Apply anti-overfitting techniques (regularization, dropout, data augmentation)\n")
            f.write("3. Consider ensemble methods to improve generalization\n")
            f.write("4. Monitor test set performance to validate improvements\n")
        
        print(f"📄 Markdown report saved to: {md_path}")

    def generate_summary_report(self, model_selections: Dict = None):
        """
        Generate comprehensive training summary with cross-validation results,
        overfitting analysis, and model selection results.
        
        Creates:
        1. Console summary output
        2. Per-symbol markdown reports (Requirements 8.1-8.7)
        3. Overall training results JSON
        
        Args:
            model_selections: Optional dictionary from ModelSelector.select_best_models()
        """
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY REPORT")
        print(f"{'='*80}")

        # Collect warnings for final summary
        all_warnings = []
        
        # Generate per-symbol markdown reports (Requirement 8.1)
        reports_dir = self.output_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, symbol_results in self.results.items():
            print(f"\n📊 {symbol}:")
            
            # Generate markdown report for this symbol
            self._generate_symbol_markdown_report(
                symbol, symbol_results, model_selections, reports_dir
            )
            
            # Check if any model has CV results
            has_cv_results = any(
                'cv_results' in result or result.get('history', {}).get('cv_mean_accuracy') is not None
                for result in symbol_results.values() if 'error' not in result
            )
            
            if has_cv_results:
                print(f"  {'Model':<20} {'CV Mean±Std':<20} {'Val Acc':<10} {'Test Acc':<10} {'Stability':<12} {'Warnings':<30}")
                print(f"  {'-'*105}")
            else:
                print(f"  {'Model':<20} {'Val Acc':<10} {'Test Acc':<10} {'Win Rate':<10} {'Warnings':<30}")
                print(f"  {'-'*80}")

            for model_name, result in symbol_results.items():
                warnings_list = []
                
                if 'error' in result:
                    print(f"  {model_name:<20} {'ERROR':<10}")
                    warnings_list.append(f"Training failed: {result.get('error_type', 'Unknown')}")
                    all_warnings.append(f"{symbol}/{model_name}: Training failed")
                    continue

                val_acc = result.get('val_metrics', {}).get('accuracy', 0)
                test_acc = result.get('test_metrics', {}).get('accuracy', 0)
                
                # Check for overfitting
                history = result.get('history', {})
                train_acc = history.get('train_accuracy', 0)
                train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                
                if train_val_gap > 0.15:
                    warnings_list.append(f"Overfitting (gap={train_val_gap:.1%})")
                    all_warnings.append(f"{symbol}/{model_name}: Overfitting detected")
                
                # Check for CV instability
                cv_results = result.get('cv_results')
                cv_mean = history.get('cv_mean_accuracy') or (cv_results.get('mean_accuracy') if cv_results else None)
                cv_std = history.get('cv_std_accuracy') or (cv_results.get('std_accuracy') if cv_results else None)
                cv_stable = history.get('cv_is_stable') if 'cv_is_stable' in history else (cv_results.get('is_stable') if cv_results else None)
                
                if cv_stable is not None and not cv_stable:
                    warnings_list.append(f"Unstable CV (std={cv_std:.3f})")
                    all_warnings.append(f"{symbol}/{model_name}: Unstable cross-validation")
                
                # Check for low test accuracy
                if test_acc < 0.55:
                    warnings_list.append(f"Low accuracy ({test_acc:.1%})")
                    all_warnings.append(f"{symbol}/{model_name}: Low test accuracy")
                
                # Format warnings
                warnings_str = "; ".join(warnings_list) if warnings_list else "None"
                
                if has_cv_results:
                    if cv_mean is not None and cv_std is not None:
                        cv_str = f"{cv_mean:.4f}±{cv_std:.4f}"
                        stability = "✅ Stable" if cv_stable else "⚠️ Unstable"
                    else:
                        cv_str = "N/A"
                        stability = "N/A"
                    
                    print(f"  {model_name:<20} {cv_str:<20} {val_acc:<10.3f} {test_acc:<10.3f} {stability:<12} {warnings_str:<30}")
                else:
                    win_rate = result.get('test_metrics', {}).get('win_rate_predicted', 0)
                    print(f"  {model_name:<20} {val_acc:<10.3f} {test_acc:<10.3f} {win_rate:<10.1%} {warnings_str:<30}")
        
        # Print model selection results if available
        if model_selections:
            print(f"\n{'='*80}")
            print("MODEL SELECTION RESULTS")
            print(f"{'='*80}")
            
            for symbol, selection in model_selections.items():
                print(f"\n🎯 {symbol}:")
                if selection['selected_model']:
                    print(f"  Selected Model: {selection['selected_model']}")
                    print(f"  Test Accuracy:  {selection['test_accuracy']:.3f}")
                    print(f"  Train-Val Gap:  {selection['train_val_gap']:.1%}")
                    print(f"  Reason: {selection['reason']}")
                    if selection.get('alternatives'):
                        print(f"  Alternatives: {', '.join(selection['alternatives'])}")
                else:
                    print(f"  ⚠️ No model selected - {selection['reason']}")
                    print(f"  Action Required: {selection.get('action', 'MANUAL_REVIEW')}")
                    all_warnings.append(f"{symbol}: No model met quality criteria")
        
        # Print all warnings summary
        if all_warnings:
            print(f"\n{'='*80}")
            print("⚠️ WARNINGS SUMMARY")
            print(f"{'='*80}")
            print(f"\nTotal Warnings: {len(all_warnings)}\n")
            for i, warning in enumerate(all_warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print(f"\n{'='*80}")
            print("✅ NO WARNINGS - All models trained successfully")
            print(f"{'='*80}")

        # Save results to JSON
        results_file = self.output_dir / 'training_results.json'
        summary_data = {
            'training_results': self.results,
            'model_selections': model_selections,
            'warnings': all_warnings,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"📄 Per-symbol reports saved to: {reports_dir}")
        print(f"\n✅ Training complete!")
    
    def _generate_symbol_markdown_report(self, symbol: str, symbol_results: Dict, 
                                         model_selections: Dict, reports_dir: Path):
        """
        Generate detailed markdown report for a single symbol.
        
        Implements Requirements 8.1-8.7:
        - 8.1: Generate markdown report per symbol
        - 8.2: Include train/val/test accuracy for all models
        - 8.3: Include overfitting metrics (train-val gap)
        - 8.4: Include cross-validation stability scores
        - 8.5: Include feature importance top 10
        - 8.6: Include confusion matrices
        - 8.7: Recommend best model for deployment
        
        Args:
            symbol: Trading symbol
            symbol_results: Results dictionary for this symbol
            model_selections: Model selection results (if available)
            reports_dir: Directory to save report
        """
        report_path = reports_dir / f'{symbol}_training_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Training Report: {symbol}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Model Selection Recommendation (Requirement 8.7)
            if model_selections and symbol in model_selections:
                selection = model_selections[symbol]
                f.write("## 🎯 Deployment Recommendation\n\n")
                
                if selection['selected_model']:
                    f.write(f"**Recommended Model:** {selection['selected_model']}\n\n")
                    f.write(f"- **Test Accuracy:** {selection['test_accuracy']:.3f} ({selection['test_accuracy']*100:.1f}%)\n")
                    f.write(f"- **Train-Val Gap:** {selection['train_val_gap']:.1%}\n")
                    f.write(f"- **Selection Score:** {selection.get('score', 0):.3f}\n")
                    f.write(f"- **Reason:** {selection['reason']}\n\n")
                    
                    if selection.get('alternatives'):
                        f.write(f"**Alternative Models:** {', '.join(selection['alternatives'])}\n\n")
                else:
                    f.write(f"**⚠️ No Model Recommended**\n\n")
                    f.write(f"- **Reason:** {selection['reason']}\n")
                    f.write(f"- **Action Required:** {selection.get('action', 'MANUAL_REVIEW')}\n\n")
                
                f.write("---\n\n")
            
            # Model Performance Comparison (Requirements 8.2, 8.3, 8.4)
            f.write("## 📊 Model Performance Comparison\n\n")
            f.write("| Model | Train Acc | Val Acc | Test Acc | Train-Val Gap | CV Mean±Std | CV Stable | Status |\n")
            f.write("|-------|-----------|---------|----------|---------------|-------------|-----------|--------|\n")
            
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    f.write(f"| {model_name} | - | - | - | - | - | - | ❌ Error |\n")
                    continue
                
                history = result.get('history', {})
                val_metrics = result.get('val_metrics', {})
                test_metrics = result.get('test_metrics', {})
                
                train_acc = history.get('train_accuracy', 0)
                val_acc = val_metrics.get('accuracy', 0)
                test_acc = test_metrics.get('accuracy', 0)
                train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                
                # CV results
                cv_results = result.get('cv_results')
                cv_mean = history.get('cv_mean_accuracy') or (cv_results.get('mean_accuracy') if cv_results else None)
                cv_std = history.get('cv_std_accuracy') or (cv_results.get('std_accuracy') if cv_results else None)
                cv_stable = history.get('cv_is_stable') if 'cv_is_stable' in history else (cv_results.get('is_stable') if cv_results else None)
                
                cv_str = f"{cv_mean:.3f}±{cv_std:.3f}" if cv_mean is not None else "N/A"
                cv_stable_str = "✅" if cv_stable else ("⚠️" if cv_stable is not None else "N/A")
                
                # Status
                status = "✅"
                if train_val_gap > 0.20:
                    status = "⚠️ Overfit"
                elif test_acc < 0.55:
                    status = "⚠️ Low Acc"
                elif cv_stable is False:
                    status = "⚠️ Unstable"
                
                f.write(f"| {model_name} | {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f} | "
                       f"{train_val_gap:.1%} | {cv_str} | {cv_stable_str} | {status} |\n")
            
            f.write("\n---\n\n")
            
            # Overfitting Analysis (Requirement 8.3)
            f.write("## 🔍 Overfitting Analysis\n\n")
            
            overfitting_detected = False
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    continue
                
                history = result.get('history', {})
                train_acc = history.get('train_accuracy', 0)
                val_acc = result.get('val_metrics', {}).get('accuracy', 0)
                train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                
                if train_val_gap > 0.15:
                    overfitting_detected = True
                    severity = "🔴 Severe" if train_val_gap > 0.25 else ("🟡 Moderate" if train_val_gap > 0.20 else "🟢 Mild")
                    f.write(f"### {model_name}\n\n")
                    f.write(f"- **Severity:** {severity}\n")
                    f.write(f"- **Train Accuracy:** {train_acc:.3f}\n")
                    f.write(f"- **Val Accuracy:** {val_acc:.3f}\n")
                    f.write(f"- **Gap:** {train_val_gap:.1%}\n")
                    f.write(f"- **Recommendation:** ")
                    
                    if train_val_gap > 0.25:
                        f.write("Increase regularization significantly or reduce model complexity\n")
                    elif train_val_gap > 0.20:
                        f.write("Increase regularization or add more training data\n")
                    else:
                        f.write("Minor tuning may improve generalization\n")
                    
                    f.write("\n")
            
            if not overfitting_detected:
                f.write("✅ No significant overfitting detected in any model.\n\n")
            
            f.write("---\n\n")
            
            # Cross-Validation Stability (Requirement 8.4)
            f.write("## 📈 Cross-Validation Stability\n\n")
            
            cv_available = False
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    continue
                
                history = result.get('history', {})
                cv_results = result.get('cv_results')
                cv_mean = history.get('cv_mean_accuracy') or (cv_results.get('mean_accuracy') if cv_results else None)
                
                if cv_mean is not None:
                    cv_available = True
                    cv_std = history.get('cv_std_accuracy') or cv_results.get('std_accuracy')
                    cv_stable = history.get('cv_is_stable') if 'cv_is_stable' in history else cv_results.get('is_stable')
                    cv_fold_accs = history.get('cv_fold_accuracies') or cv_results.get('fold_accuracies', [])
                    
                    f.write(f"### {model_name}\n\n")
                    f.write(f"- **Mean Accuracy:** {cv_mean:.4f}\n")
                    f.write(f"- **Std Deviation:** {cv_std:.4f}\n")
                    f.write(f"- **Stability:** {'✅ Stable' if cv_stable else '⚠️ Unstable'}\n")
                    
                    if cv_fold_accs:
                        f.write(f"- **Fold Accuracies:** {', '.join([f'{acc:.3f}' for acc in cv_fold_accs])}\n")
                        f.write(f"- **Min/Max:** {min(cv_fold_accs):.3f} / {max(cv_fold_accs):.3f}\n")
                    
                    if not cv_stable:
                        f.write(f"- **⚠️ Warning:** High variance across folds suggests model instability\n")
                    
                    f.write("\n")
            
            if not cv_available:
                f.write("ℹ️ Cross-validation results not available for this symbol.\n\n")
            
            f.write("---\n\n")
            
            # Feature Importance (Requirement 8.5)
            f.write("## 🎯 Feature Importance (Top 10)\n\n")
            
            feature_importance_available = False
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    continue
                
                feature_importance = result.get('feature_importance', [])
                if feature_importance:
                    feature_importance_available = True
                    f.write(f"### {model_name}\n\n")
                    f.write("| Rank | Feature | Importance |\n")
                    f.write("|------|---------|------------|\n")
                    
                    for i, feat in enumerate(feature_importance[:10], 1):
                        feat_name = feat.get('feature', 'Unknown')
                        importance = feat.get('importance', 0)
                        f.write(f"| {i} | {feat_name} | {importance:.4f} |\n")
                    
                    f.write("\n")
            
            if not feature_importance_available:
                f.write("ℹ️ Feature importance not available (only for tree-based models).\n\n")
            
            f.write("---\n\n")
            
            # Confusion Matrices (Requirement 8.6)
            f.write("## 📉 Confusion Matrices\n\n")
            
            confusion_available = False
            for model_name, result in symbol_results.items():
                if 'error' in result:
                    continue
                
                test_metrics = result.get('test_metrics', {})
                confusion_matrix = test_metrics.get('confusion_matrix')
                
                if confusion_matrix is not None:
                    confusion_available = True
                    f.write(f"### {model_name}\n\n")
                    f.write("```\n")
                    f.write("              Predicted\n")
                    f.write("              0    1\n")
                    f.write("Actual  0   ")
                    
                    if isinstance(confusion_matrix, list):
                        cm = confusion_matrix
                    else:
                        cm = confusion_matrix.tolist() if hasattr(confusion_matrix, 'tolist') else confusion_matrix
                    
                    f.write(f"{cm[0][0]:4d} {cm[0][1]:4d}\n")
                    f.write(f"        1   {cm[1][0]:4d} {cm[1][1]:4d}\n")
                    f.write("```\n\n")
                    
                    # Calculate metrics from confusion matrix
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f.write(f"- **Precision:** {precision:.3f}\n")
                    f.write(f"- **Recall:** {recall:.3f}\n")
                    f.write(f"- **F1-Score:** {f1:.3f}\n\n")
            
            if not confusion_available:
                f.write("ℹ️ Confusion matrices not available.\n\n")
            
            f.write("---\n\n")
            
            # Warnings and Issues
            f.write("## ⚠️ Warnings and Issues\n\n")
            
            warnings_found = False
            for model_name, result in symbol_results.items():
                model_warnings = []
                
                if 'error' in result:
                    model_warnings.append(f"Training failed: {result.get('error_type', 'Unknown error')}")
                else:
                    history = result.get('history', {})
                    train_acc = history.get('train_accuracy', 0)
                    val_acc = result.get('val_metrics', {}).get('accuracy', 0)
                    test_acc = result.get('test_metrics', {}).get('accuracy', 0)
                    train_val_gap = history.get('train_val_gap', train_acc - val_acc)
                    
                    if train_val_gap > 0.20:
                        model_warnings.append(f"High overfitting (gap={train_val_gap:.1%})")
                    
                    if test_acc < 0.55:
                        model_warnings.append(f"Low test accuracy ({test_acc:.1%})")
                    
                    cv_stable = history.get('cv_is_stable')
                    if cv_stable is False:
                        cv_std = history.get('cv_std_accuracy', 0)
                        model_warnings.append(f"Unstable cross-validation (std={cv_std:.3f})")
                    
                    # Check for training warnings
                    training_warnings = result.get('warnings', [])
                    if training_warnings:
                        model_warnings.extend(training_warnings)
                
                if model_warnings:
                    warnings_found = True
                    f.write(f"### {model_name}\n\n")
                    for warning in model_warnings:
                        f.write(f"- ⚠️ {warning}\n")
                    f.write("\n")
            
            if not warnings_found:
                f.write("✅ No warnings - all models trained successfully.\n\n")
            
            f.write("---\n\n")
            f.write(f"*Report generated by SMC Model Training Pipeline*\n")
        
        print(f"  📄 Report saved: {report_path.name}")


# Main execution
if __name__ == "__main__":
    """
    Train all SMC models with automatic model selection

    Usage:
        python train_all_models.py
    """

    print("="*80)
    print("SMC MODEL TRAINING ORCHESTRATOR")
    print("Following specifications in WHATS_NEEDED.md")
    print("="*80)

    # Initialize trainer
    trainer = SMCModelTrainer(
        data_dir='Data',
        output_dir='models/trained'
    )

    # Check data availability
    if not trainer.check_data_availability():
        print("\n❌ Cannot proceed without training data")
        print("\nRun this command first:")
        print("  python run_complete_pipeline.py")
        exit(1)

    # Get available symbols
    symbols = trainer.get_available_symbols()

    if not symbols:
        print("\n❌ No symbols found in dataset")
        exit(1)

    # Train models for each symbol with comprehensive error handling
    print(f"\n{'='*80}")
    print(f"Training {len(symbols)} symbols with 4 model types each")
    print(f"{'='*80}")
    
    training_start = datetime.now()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            trainer.train_all_for_symbol(
                symbol=symbol,
                exclude_timeout=False,  # Include all classes
                models=['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']
            )
        except Exception as e:
            print(f"\n❌ Critical error training models for {symbol}: {e}")
            print(f"   Error Type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            # Continue with next symbol
            continue
    
    training_duration = (datetime.now() - training_start).total_seconds()
    print(f"\n{'='*80}")
    print(f"All training completed in {training_duration:.1f}s ({training_duration/60:.1f} minutes)")
    print(f"{'='*80}")

    # Initialize ModelSelector and select best models
    print(f"\n{'='*80}")
    print("AUTOMATIC MODEL SELECTION")
    print(f"{'='*80}")
    
    try:
        selector = ModelSelector(
            max_gap=0.20,           # Max 20% train-val gap
            min_accuracy=0.55,      # Min 55% test accuracy
            stability_threshold=0.05  # Max 5% val-test difference
        )
        
        # Select best models per symbol
        model_selections = selector.select_best_models(trainer.results)
        
        # Save deployment manifest
        manifest_path = trainer.output_dir / 'deployment_manifest.json'
        selector.save_deployment_manifest(model_selections, str(manifest_path))
        
    except Exception as e:
        print(f"\n❌ Model selection failed: {e}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        model_selections = None

    # Generate overfitting analysis report
    try:
        trainer.generate_overfitting_report()
    except Exception as e:
        print(f"\n⚠️ Overfitting report generation failed: {e}")
        print(f"   Continuing with summary report...")

    # Generate comprehensive summary report with model selections
    try:
        trainer.generate_summary_report(model_selections=model_selections)
    except Exception as e:
        print(f"\n❌ Summary report generation failed: {e}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review training results in models/trained/training_results.json")
    print("2. Check deployment manifest in models/trained/deployment_manifest.json")
    print("3. Review overfitting analysis in models/trained/overfitting_report.md")
    print("4. Address any warnings flagged in the summary report")
    print("5. Use selected models for ensemble predictions")
    print("6. Backtest on test set")
    print("\nModels saved in: models/trained/")
    
    # Print final statistics
    total_models = sum(len(results) for results in trainer.results.values())
    successful_models = sum(
        1 for results in trainer.results.values() 
        for result in results.values() 
        if 'error' not in result
    )
    failed_models = total_models - successful_models
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"  Total Symbols:        {len(symbols)}")
    print(f"  Total Models Trained: {total_models}")
    print(f"  Successful:           {successful_models}")
    print(f"  Failed:               {failed_models}")
    print(f"  Success Rate:         {successful_models/total_models*100:.1f}%")
    
    if model_selections:
        selected_count = sum(1 for s in model_selections.values() if s['selected_model'])
        print(f"  Models Selected:      {selected_count}/{len(symbols)}")
    
    print(f"  Total Duration:       {training_duration:.1f}s ({training_duration/60:.1f} min)")
    print(f"{'='*80}")
