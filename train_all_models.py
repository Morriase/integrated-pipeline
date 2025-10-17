"""
Training Orchestrator for All SMC Models

Trains all model types per symbol and evaluates performance
Following specifications in WHATS_NEEDED.md
"""

from models.random_forest_model import RandomForestSMCModel
from models.xgboost_model import XGBoostSMCModel, XGBOOST_AVAILABLE
from models.neural_network_model import NeuralNetworkSMCModel, TORCH_AVAILABLE
from models.lstm_model import LSTMSMCModel
import warnings
from typing import Dict, List
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / 'models'))

# Import models from models directory

warnings.filterwarnings('ignore')

# Import model classes


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
        self.output_dir = Path('/kaggle/working')
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

        # Prepare features
        X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
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

        # Prepare features
        X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
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
        Train all models for a specific symbol

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

        # Train each model
        if 'RandomForest' in models:
            try:
                results['RandomForest'] = self.train_random_forest(
                    symbol, exclude_timeout)
            except Exception as e:
                print(f"\n❌ Random Forest training failed: {e}")
                results['RandomForest'] = {'error': str(e)}

        if 'XGBoost' in models and XGBOOST_AVAILABLE:
            try:
                results['XGBoost'] = self.train_xgboost(
                    symbol, exclude_timeout)
            except Exception as e:
                print(f"\n❌ XGBoost training failed: {e}")
                results['XGBoost'] = {'error': str(e)}

        if 'NeuralNetwork' in models and TORCH_AVAILABLE:
            try:
                results['NeuralNetwork'] = self.train_neural_network(
                    symbol, exclude_timeout)
            except Exception as e:
                print(f"\n❌ Neural Network training failed: {e}")
                results['NeuralNetwork'] = {'error': str(e)}

        if 'LSTM' in models and TORCH_AVAILABLE:
            try:
                results['LSTM'] = self.train_lstm(symbol, exclude_timeout)
            except Exception as e:
                print(f"\n❌ LSTM training failed: {e}")
                results['LSTM'] = {'error': str(e)}

        self.results[symbol] = results

        return results

    def generate_summary_report(self):
        """Generate comprehensive training summary"""
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY REPORT")
        print(f"{'='*80}")

        for symbol, symbol_results in self.results.items():
            print(f"\n📊 {symbol}:")
            print(
                f"  {'Model':<20} {'Val Acc':<10} {'Test Acc':<10} {'Win Rate':<10}")
            print(f"  {'-'*50}")

            for model_name, result in symbol_results.items():
                if 'error' in result:
                    print(f"  {model_name:<20} {'ERROR':<10}")
                    continue

                val_acc = result.get('val_metrics', {}).get('accuracy', 0)
                test_acc = result.get('test_metrics', {}).get('accuracy', 0)
                win_rate = result.get('test_metrics', {}).get(
                    'win_rate_predicted', 0)

                print(
                    f"  {model_name:<20} {val_acc:<10.3f} {test_acc:<10.3f} {win_rate:<10.1%}")

        # Save results to JSON
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"\n✅ Training complete!")


# Main execution
if __name__ == "__main__":
    """
    Train all SMC models

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

    # Train models for each symbol
    for symbol in symbols:
        trainer.train_all_for_symbol(
            symbol=symbol,
            exclude_timeout=False,  # Include all classes
            models=['RandomForest', 'XGBoost', 'NeuralNetwork', 'LSTM']
        )

    # Generate summary report
    trainer.generate_summary_report()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review training results in models/trained/training_results.json")
    print("2. Check model performance metrics")
    print("3. Use best models for ensemble predictions")
    print("4. Backtest on test set")
    print("\nModels saved in: models/trained/")
