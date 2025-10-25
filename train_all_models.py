"""
Training Orchestrator for SMC Models - FULL DATASET APPROACH

Trains models on the COMPLETE unified dataset (all symbols combined)
This solves overfitting by providing 2,500+ samples instead of 178-295 per symbol
"""

from models.lstm_model import LSTMSMCModel
from models.neural_network_model import NeuralNetworkSMCModel, TORCH_AVAILABLE
from models.xgboost_model import XGBoostSMCModel, XGBOOST_AVAILABLE
from models.random_forest_model import RandomForestSMCModel
import warnings
from typing import Dict
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add pipeline directory to path
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir))

# Import models

warnings.filterwarnings('ignore')


class UnifiedModelTrainer:
    """
    Trains models on the FULL unified dataset (all symbols combined)

    Benefits:
    - More training data (2,500+ samples vs 178-295 per symbol)
    - Better generalization across symbols
    - Less overfitting
    - Simpler deployment (3 models vs 44 models)
    - ATR normalization makes features comparable across symbols
    """

    def __init__(self, data_dir: str = 'Data', output_dir: str = 'models/trained',
                 include_lstm: bool = False):
        """
        Initialize unified trainer

        Args:
            data_dir: Directory containing processed data splits
            output_dir: Directory to save trained models
            include_lstm: Whether to train LSTM (experimental, may be unstable)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_lstm = include_lstm

        # Data paths - use cleaned data if available
        if (self.data_dir / 'processed_smc_data_train_clean.csv').exists():
            self.train_path = self.data_dir / 'processed_smc_data_train_clean.csv'
            self.val_path = self.data_dir / 'processed_smc_data_val_clean.csv'
            self.test_path = self.data_dir / 'processed_smc_data_test_clean.csv'
            self.using_clean_data = True
        else:
            self.train_path = self.data_dir / 'processed_smc_data_train.csv'
            self.val_path = self.data_dir / 'processed_smc_data_val.csv'
            self.test_path = self.data_dir / 'processed_smc_data_test.csv'
            self.using_clean_data = False

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

            # Print dataset statistics
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)
            test_df = pd.read_csv(self.test_path)

            print(f"\n📊 Dataset Statistics:")
            print(f"  Train samples: {len(train_df):,}")
            print(f"  Val samples:   {len(val_df):,}")
            print(f"  Test samples:  {len(test_df):,}")
            print(
                f"  Total:         {len(train_df) + len(val_df) + len(test_df):,}")
            print(f"  Symbols:       {train_df['symbol'].nunique()}")
            print(f"  Features:      {len(train_df.columns)}")

            return True
        else:
            print(f"  ✗ Data files not found in {self.data_dir}")
            print(
                f"\n  Run 'python run_complete_pipeline.py' first to generate training data")
            return False

    def train_random_forest(self) -> Dict:
        """Train Random Forest on FULL dataset"""
        print(f"\n{'='*80}")
        print(f"Training Random Forest on FULL UNIFIED DATASET")
        print(f"{'='*80}")

        # Use 'UNIFIED' as symbol name
        model = RandomForestSMCModel(symbol='UNIFIED')

        # Load FULL dataset (all symbols)
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        print(f"\n📊 Training on FULL dataset:")
        print(
            f"  Train: {len(train_df):,} samples ({train_df['symbol'].nunique()} symbols)")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")

        # Prepare features (all symbols combined) WITHOUT feature selection for now
        # TODO: Fix feature selection correlation issue with small datasets
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True, 
                                                  apply_feature_selection=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train with cross-validation
        print(f"\n🔄 Performing 5-fold cross-validation...")
        cv_results = model.cross_validate(
            X_train, y_train, n_folds=5, stratified=True)

        print(f"\n📈 Cross-Validation Results:")
        print(
            f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(
            f"  Stability: {'✅ Stable' if cv_results['is_stable'] else '⚠️ Unstable'}")

        # Train final model
        print(f"\n🎯 Training final model...")
        history = model.train(
            X_train, y_train, X_val, y_val,
            n_estimators=200,
            max_depth=15,  # Anti-overfitting
            min_samples_split=20,  # Anti-overfitting
            min_samples_leaf=10,  # Anti-overfitting
            use_grid_search=False
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'RandomForest',
            'symbol': 'UNIFIED',
            'cv_results': cv_results,
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(top_n=20).to_dict('records')
        }

    def train_xgboost(self) -> Dict:
        """Train XGBoost on FULL dataset"""
        if not XGBOOST_AVAILABLE:
            print(f"\n⚠️ XGBoost not available, skipping...")
            return {'error': 'XGBoost not available'}

        print(f"\n{'='*80}")
        print(f"Training XGBoost on FULL UNIFIED DATASET")
        print(f"{'='*80}")

        model = XGBoostSMCModel(symbol='UNIFIED')

        # Load FULL dataset
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        print(f"\n📊 Training on FULL dataset:")
        print(
            f"  Train: {len(train_df):,} samples ({train_df['symbol'].nunique()} symbols)")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")

        # Prepare features WITHOUT feature selection for now
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True,
                                                  apply_feature_selection=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train (uses config defaults: max_depth=3, early_stopping=20, etc.)
        print(f"\n🎯 Training with aggressive regularization...")
        history = model.train(X_train, y_train, X_val, y_val)

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'XGBoost',
            'symbol': 'UNIFIED',
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': model.get_feature_importance(top_n=20).to_dict('records')
        }

    def train_neural_network(self) -> Dict:
        """Train Neural Network on FULL dataset"""
        if not TORCH_AVAILABLE:
            print(f"\n⚠️ PyTorch not available, skipping...")
            return {'error': 'PyTorch not available'}

        print(f"\n{'='*80}")
        print(f"Training Neural Network on FULL UNIFIED DATASET")
        print(f"{'='*80}")

        model = NeuralNetworkSMCModel(symbol='UNIFIED')

        # Load FULL dataset
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        print(f"\n📊 Training on FULL dataset:")
        print(
            f"  Train: {len(train_df):,} samples ({train_df['symbol'].nunique()} symbols)")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")

        # Prepare features WITHOUT feature selection for now
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True,
                                                  apply_feature_selection=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train (uses config defaults: [128, 64] layers, dropout=0.5, etc.)
        print(f"\n🎯 Training with simplified architecture...")
        history = model.train(X_train, y_train, X_val, y_val)

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        return {
            'model_name': 'NeuralNetwork',
            'symbol': 'UNIFIED',
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

    def train_lstm(self) -> Dict:
        """Train LSTM on FULL dataset (EXPERIMENTAL - may be unstable)"""
        if not TORCH_AVAILABLE:
            print(f"\n⚠️ PyTorch not available, skipping LSTM...")
            return {'error': 'PyTorch not available'}

        print(f"\n{'='*80}")
        print(f"Training LSTM on FULL UNIFIED DATASET (EXPERIMENTAL)")
        print(f"{'='*80}")
        print(f"\n⚠️  WARNING: LSTM has shown instability in previous tests:")
        print(f"   - Severe overfitting (39-61% train-val gap)")
        print(f"   - Gradient explosions (24 warnings)")
        print(f"   - Poor test accuracy (16-46%)")
        print(f"   - Training divergence")
        print(f"\n   This model is included for experimental purposes only.")

        model = LSTMSMCModel(symbol='UNIFIED', lookback=10)

        # Load FULL dataset
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        print(f"\n📊 Training on FULL dataset:")
        print(
            f"  Train: {len(train_df):,} samples ({train_df['symbol'].nunique()} symbols)")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        print(f"  Lookback: 10 candles")

        # Prepare features WITHOUT feature selection for now (LSTM needs scaler fitted)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=True,
                                                  apply_feature_selection=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)

        # Train with aggressive regularization
        print(f"\n🎯 Training with AGGRESSIVE regularization...")
        history = model.train(
            X_train, y_train, X_val, y_val,
            hidden_dim=32,  # Small
            num_layers=1,  # Simple
            dropout=0.6,  # High dropout
            learning_rate=0.0001,  # Slow learning
            batch_size=16,
            epochs=200,
            patience=15,
            weight_decay=0.1,  # Strong L2
            bidirectional=True
        )

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, 'Validation')
        test_metrics = model.evaluate(X_test, y_test, 'Test')

        # Save model
        model.save_model(str(self.output_dir))

        # Check for warnings
        warnings = history.get('training_warnings', [])
        stability_status = "⚠️ UNSTABLE" if len(warnings) > 5 else "✅ Stable"

        return {
            'model_name': 'LSTM',
            'symbol': 'UNIFIED',
            'history': history,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_warnings': warnings,
            'stability_status': stability_status,
            'experimental': True
        }

    def train_all_models(self) -> Dict:
        """Train all models on the full dataset"""
        total_models = 4 if self.include_lstm else 3

        print(f"\n{'#'*80}")
        print(f"# Training All Models on UNIFIED DATASET")
        print(f"{'#'*80}")
        print(f"\n🎯 Strategy: Train on ALL data for maximum generalization")
        print(f"   Core Models: RandomForest, XGBoost, NeuralNetwork")
        if self.include_lstm:
            print(f"   Experimental: LSTM (may be unstable)")
        else:
            print(f"   LSTM: Disabled (use --include-lstm to enable)")

        results = {}
        training_start = datetime.now()

        # RandomForest
        try:
            print(f"\n🌲 Starting RandomForest training...")
            model_start = datetime.now()
            results['RandomForest'] = self.train_random_forest()
            model_duration = (datetime.now() - model_start).total_seconds()
            results['RandomForest']['training_duration_seconds'] = model_duration
            print(f"✅ RandomForest completed in {model_duration:.1f}s")
        except Exception as e:
            print(f"\n❌ RandomForest training failed: {e}")
            import traceback
            print(traceback.format_exc())
            results['RandomForest'] = {'error': str(
                e), 'error_type': type(e).__name__}

        # XGBoost
        try:
            print(f"\n🚀 Starting XGBoost training...")
            model_start = datetime.now()
            results['XGBoost'] = self.train_xgboost()
            if 'error' not in results['XGBoost']:
                model_duration = (datetime.now() - model_start).total_seconds()
                results['XGBoost']['training_duration_seconds'] = model_duration
                print(f"✅ XGBoost completed in {model_duration:.1f}s")
        except Exception as e:
            print(f"\n❌ XGBoost training failed: {e}")
            import traceback
            print(traceback.format_exc())
            results['XGBoost'] = {'error': str(
                e), 'error_type': type(e).__name__}

        # Neural Network
        try:
            print(f"\n🧠 Starting Neural Network training...")
            model_start = datetime.now()
            results['NeuralNetwork'] = self.train_neural_network()
            if 'error' not in results['NeuralNetwork']:
                model_duration = (datetime.now() - model_start).total_seconds()
                results['NeuralNetwork']['training_duration_seconds'] = model_duration
                print(f"✅ Neural Network completed in {model_duration:.1f}s")
        except Exception as e:
            print(f"\n❌ Neural Network training failed: {e}")
            import traceback
            print(traceback.format_exc())
            results['NeuralNetwork'] = {'error': str(
                e), 'error_type': type(e).__name__}

        # LSTM (optional/experimental)
        if self.include_lstm:
            try:
                print(f"\n🔄 Starting LSTM training (EXPERIMENTAL)...")
                model_start = datetime.now()
                results['LSTM'] = self.train_lstm()
                if 'error' not in results['LSTM']:
                    model_duration = (
                        datetime.now() - model_start).total_seconds()
                    results['LSTM']['training_duration_seconds'] = model_duration

                    # Check stability
                    warnings = results['LSTM'].get('training_warnings', [])
                    if len(warnings) > 5:
                        print(
                            f"⚠️ LSTM completed in {model_duration:.1f}s but showed {len(warnings)} warnings")
                    else:
                        print(f"✅ LSTM completed in {model_duration:.1f}s")
            except Exception as e:
                print(f"\n❌ LSTM training failed: {e}")
                import traceback
                print(traceback.format_exc())
                results['LSTM'] = {'error': str(
                    e), 'error_type': type(e).__name__}

        # Summary
        total_duration = (datetime.now() - training_start).total_seconds()
        successful = sum(1 for r in results.values() if 'error' not in r)

        print(f"\n{'='*80}")
        print(f"Training Summary:")
        print(
            f"  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"  Successful Models: {successful}/{total_models}")
        print(f"  Failed Models: {total_models - successful}/{total_models}")
        print(f"{'='*80}")

        self.results = {'UNIFIED': results}
        return results

    def generate_summary_report(self):
        """Generate comprehensive training summary"""
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY REPORT - UNIFIED DATASET")
        print(f"{'='*80}")

        results = self.results.get('UNIFIED', {})

        print(f"\n📊 Model Performance:")
        print(
            f"  {'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Status':<20}")
        print(f"  {'-'*100}")

        for model_name, result in results.items():
            if 'error' in result:
                print(f"  {model_name:<20} {'ERROR':<12}")
                continue

            history = result.get('history', {})
            val_metrics = result.get('val_metrics', {})
            test_metrics = result.get('test_metrics', {})

            train_acc = history.get('train_accuracy', 0)
            val_acc = val_metrics.get('accuracy', 0)
            test_acc = test_metrics.get('accuracy', 0)
            train_val_gap = history.get('train_val_gap', train_acc - val_acc)

            # Special handling for LSTM
            if model_name == 'LSTM':
                warnings = result.get('training_warnings', [])
                if len(warnings) > 5:
                    status = f"⚠️ Unstable ({len(warnings)} warnings)"
                elif train_val_gap > 0.30:
                    status = "⚠️ Severe Overfit"
                elif train_val_gap > 0.15:
                    status = "⚠️ Overfit"
                else:
                    status = "✅ Good (Experimental)"
            else:
                status = "✅ Good" if train_val_gap < 0.15 else "⚠️ Overfit"

            print(
                f"  {model_name:<20} {train_acc:<12.3f} {val_acc:<12.3f} {test_acc:<12.3f} {train_val_gap:<10.2%} {status:<20}")

        # Save results
        results_file = self.output_dir / 'training_results.json'
        summary_data = {
            'training_results': self.results,
            'timestamp': datetime.now().isoformat(),
            'approach': 'unified_dataset',
            'total_models': 4 if self.include_lstm else 3,
            'lstm_included': self.include_lstm
        }

        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"\n✅ Training complete!")


# Main execution
if __name__ == "__main__":
    """
    Train models on FULL UNIFIED DATASET

    Usage:
        python train_all_models.py              # Train 3 core models (RF, XGB, NN)
        python train_all_models.py --include-lstm  # Train 4 models (includes LSTM)
    """

    # Parse command line arguments
    include_lstm = '--include-lstm' in sys.argv or '-lstm' in sys.argv

    print("="*80)
    print("SMC MODEL TRAINING - FULL DATASET APPROACH")
    print("="*80)
    print("\n🎯 Training Strategy:")
    print("  - Train on ALL symbols combined (unified dataset)")
    print("  - 2,500+ samples (vs 178-295 per symbol)")
    print(f"  - {4 if include_lstm else 3} models total (vs 44 models)")
    print("  - Better generalization, less overfitting")
    print("  - ATR normalization makes features comparable")

    if include_lstm:
        print("\n⚠️  LSTM ENABLED (Experimental):")
        print("   - May show overfitting and gradient issues")
        print("   - Included for research/comparison purposes")
        print("   - Not recommended for production use")

    # Detect environment and set paths
    if Path('/kaggle/input').exists():
        print("🔍 Detected Kaggle environment")
        # Check for cleaned data first (better for NN)
        if Path('/kaggle/working/Data-output-clean').exists():
            data_dir = '/kaggle/working/Data-output-clean'
            print("  ✓ Using cleaned data (optimized for Neural Network)")
        else:
            data_dir = '/kaggle/working/Data-output'
            print("  ⚠️ Using raw data (run fix_nn_training.py for better NN results)")
        output_dir = '/kaggle/working/Model-output'
    else:
        # Local paths - check for cleaned data
        if Path('Data-clean').exists():
            data_dir = 'Data-clean'
            print("  ✓ Using cleaned data (optimized for Neural Network)")
        else:
            data_dir = 'Data'
            print("  ⚠️ Using raw data (run fix_nn_training.py for better NN results)")
        output_dir = 'models/trained'
    
    print(f"📂 Data Directory: {data_dir}")
    print(f"📂 Model Output Directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = UnifiedModelTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        include_lstm=include_lstm
    )

    # Check data availability
    if not trainer.check_data_availability():
        print("\n❌ Cannot proceed without training data")
        print("\nRun this command first:")
        print("  python run_complete_pipeline.py")
        exit(1)

    # Train all models
    print(f"\n{'='*80}")
    print(f"STARTING UNIFIED TRAINING")
    print(f"{'='*80}")

    training_start = datetime.now()

    try:
        results = trainer.train_all_models()
    except Exception as e:
        print(f"\n❌ Critical error during training: {e}")
        import traceback
        print(traceback.format_exc())
        exit(1)

    training_duration = (datetime.now() - training_start).total_seconds()

    print(f"\n{'='*80}")
    print(
        f"All training completed in {training_duration:.1f}s ({training_duration/60:.1f} minutes)")
    print(f"{'='*80}")

    # Generate summary report
    try:
        trainer.generate_summary_report()
    except Exception as e:
        print(f"\n⚠️ Summary report generation failed: {e}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review training results in models/trained/training_results.json")
    print("2. Check model files: UNIFIED_RandomForest.pkl, UNIFIED_XGBoost.pkl, UNIFIED_NeuralNetwork.pkl")
    print("3. Use unified models for predictions on ANY symbol")
    print("4. Deploy to production")
    print("\nModels saved in: models/trained/")

    # Print final statistics
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = sum(1 for r in results.values() if 'error' in r)
    total_models = 4 if include_lstm else 3

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"  Approach:             Unified Dataset (All Symbols)")
    print(f"  Total Models:         {total_models}")
    print(f"  Successful:           {successful}")
    print(f"  Failed:               {failed}")
    print(f"  Success Rate:         {successful/total_models*100:.1f}%")
    print(
        f"  Total Duration:       {training_duration:.1f}s ({training_duration/60:.1f} min)")
    print(f"  Models per Symbol:    1 (unified model works for all)")
    if include_lstm:
        print(
            f"  LSTM Status:          {'✅ Trained' if 'LSTM' in results and 'error' not in results['LSTM'] else '❌ Failed'}")
    print(f"{'='*80}")

    print("\n🎉 Done! Your unified models are ready for deployment!")

    if include_lstm and 'LSTM' in results and 'error' not in results['LSTM']:
        warnings = results['LSTM'].get('training_warnings', [])
        if len(warnings) > 5:
            print(
                f"\n⚠️  Note: LSTM showed {len(warnings)} training warnings.")
            print(f"   Consider using RF, XGB, or NN for production instead.")
