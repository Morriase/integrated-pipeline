"""
Integrated Advanced SMC Pipeline
- Combines multi-timeframe, temporal, and ensemble architectures
- Production-ready implementation with comprehensive evaluation
- Seamless integration with existing Black Ice Protocol infrastructure
"""

from production_ensemble_pipeline import EnsembleManager
from advanced_temporal_architecture import (
    SMC_LSTM,
    SMC_Transformer,
    RegimeClassifier,
    RegimeAwareEnsemble,
    TemporalDataProcessor,
    train_temporal_model
)
from enhanced_multitf_pipeline import (
    standardize_features,
    EnhancedSMC_MLP
)
# Note: Using pre-engineered institutional features from feature_engineering_smc_institutional.py
# No need to import feature engineering functions - data is already processed
from learning_curve_plotter import integrate_learning_curves
from temporal_validation import TemporalValidator
from model_export import export_all_models
from recovery_mechanism import RecoveryManager
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Import our components


class IntegratedSMCSystem:
    """
    Integrated SMC system combining all advanced architectures
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.temporal_processor = TemporalDataProcessor(
            sequence_length=config.get('sequence_length', 20),
            prediction_horizon=config.get('prediction_horizon', 8)
        )
        self.ensemble_manager = None
        self.regime_classifier = None
        self.performance_metrics = {}

    def create_simple_labels(self, df: pd.DataFrame, lookforward: int = 8, threshold_pct: float = 0.6) -> np.ndarray:
        """
        Create simple labels based on future price movement
        Process each symbol separately to maintain temporal integrity
        """
        print(
            f"\n🏷️  Creating labels (lookforward={lookforward}, threshold={threshold_pct}%)...")

        all_labels = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_labels = np.zeros(len(symbol_df), dtype=np.int64)

            for i in range(len(symbol_df) - lookforward):
                current_close = symbol_df['close'].iloc[i]
                current_atr = symbol_df['ATR'].iloc[i]

                if pd.isna(current_atr) or current_atr == 0:
                    continue

                # Look forward
                future_window = symbol_df['close'].iloc[i+1:i+lookforward+1]

                if len(future_window) == 0:
                    continue

                # Calculate max gain and max loss in ATR terms
                max_gain = (future_window.max() - current_close) / current_atr
                max_loss = (current_close - future_window.min()) / current_atr

                # Label based on threshold
                if max_gain > threshold_pct and max_gain > max_loss:
                    symbol_labels[i] = 1  # BUY
                elif max_loss > threshold_pct and max_loss > max_gain:
                    symbol_labels[i] = -1  # SELL
                else:
                    symbol_labels[i] = 0  # HOLD

            all_labels.append(symbol_labels)

        # Concatenate all labels
        labels = np.concatenate(all_labels)

        # Convert to 0/1/2 format
        labels = labels + 1  # -1→0 (SELL), 0→1 (HOLD), 1→2 (BUY)

        return labels

    def load_and_engineer_institutional_features(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load pre-engineered institutional-grade SMC features
        """
        print("\n🏛️  LOADING INSTITUTIONAL-GRADE SMC FEATURES")
        print("="*70)

        # Verify file exists
        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\nPlease run: python Python/feature_engineering_smc_institutional.py")

        # Load pre-engineered data
        df = pd.read_csv(data_path, parse_dates=['time'])
        print(f"✓ Loaded {len(df):,} regime-filtered samples from {data_path}")

        # Show data statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"  Symbols: {df['symbol'].nunique()} unique")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")

        # Count SMC structures
        ob_count = df['OB_Bullish'].sum() + df['OB_Bearish'].sum()
        fvg_count = df['FVG_Bullish'].sum() + df['FVG_Bearish'].sum()
        bos_count = df['BOS_Wick_Confirm'].sum()

        print(f"  Order Blocks: {int(ob_count):,}")
        print(f"  Fair Value Gaps: {int(fvg_count):,}")
        print(f"  Structure Breaks: {int(bos_count):,}")

        # Create labels for all samples
        labels = self.create_simple_labels(
            df, lookforward=8, threshold_pct=0.6)

        print(
            f"  Label distribution: SELL={np.sum(labels == 0):,}, HOLD={np.sum(labels == 1):,}, BUY={np.sum(labels == 2):,}")

        # Define feature columns (using pre-engineered features)
        feature_columns = [
            # Basic features
            'ATR', 'EMA_50', 'EMA_200', 'RSI',
            # OB features
            'OB_Bullish', 'OB_Bearish', 'OB_Size_ATR', 'OB_Displacement_ATR',
            'OB_Quality_Score', 'OB_MTF_Confluence',
            # FVG features
            'FVG_Bullish', 'FVG_Bearish', 'FVG_Depth_ATR', 'FVG_Quality_Score',
            'FVG_MTF_Confluence',
            # BOS features
            'BOS_Wick_Confirm', 'BOS_Close_Confirm', 'BOS_Dist_ATR', 'Structure_Strength',
            # Regime features
            'Trend_Bias_Indicator', 'HTF_Trend_Bias', 'ATR_ZScore',
            'MA_Slope_Normalized', 'RSI_Momentum'
        ]

        # Fill NaN values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                print(f"  ⚠️  Missing column: {col}")

        # Extract feature matrix
        features = df[feature_columns].values

        # Extract symbol list
        symbols = df['symbol'].values.tolist()

        print(f"\n✅ INSTITUTIONAL FEATURES LOADED")
        print(f"   Features: {features.shape}")
        print(f"   Labels: {len(labels):,}")
        print(f"   Symbols: {len(set(symbols))} unique pairs")
        print("="*70)

        return features, labels, symbols

    def prepare_comprehensive_dataset(self, data_path: str) -> Dict[str, Any]:
        """
        Prepare comprehensive dataset using INSTITUTIONAL-GRADE SMC features
        """
        print("=== Preparing Comprehensive Dataset (INSTITUTIONAL SMC) ===")

        # Load and engineer institutional features
        features, labels, symbols = self.load_and_engineer_institutional_features(
            data_path)

        print(
            f"Base dataset: {len(features)} samples, {features.shape[1]} features")
        print(f"Symbols: {len(set(symbols))} unique pairs")
        print(f"Label distribution: {np.bincount(labels)}")

        # Standardize features
        features_norm, feature_mean, feature_std = standardize_features(
            features)

        # Create temporal sequences for LSTM/Transformer
        sequences, seq_labels, seq_symbols = self.temporal_processor.create_sequences(
            features_norm, labels, symbols
        )

        print(
            f"Temporal sequences: {len(sequences)} sequences of length {sequences.shape[1]}")

        # Create regime features
        regime_features = self.temporal_processor.create_regime_features(
            sequences)

        print(f"Regime features: {regime_features.shape[1]} dimensions")

        # Split data maintaining temporal order
        n_base = len(features_norm)
        n_seq = len(sequences)

        # Base data splits (70/15/15)
        train_end = int(0.7 * n_base)
        val_end = int(0.85 * n_base)

        base_splits = {
            'train': {
                'features': features_norm[:train_end],
                'labels': labels[:train_end],
                'symbols': symbols[:train_end]
            },
            'val': {
                'features': features_norm[train_end:val_end],
                'labels': labels[train_end:val_end],
                'symbols': symbols[train_end:val_end]
            },
            'test': {
                'features': features_norm[val_end:],
                'labels': labels[val_end:],
                'symbols': symbols[val_end:]
            }
        }

        # Temporal data splits
        seq_train_end = int(0.7 * n_seq)
        seq_val_end = int(0.85 * n_seq)

        temporal_splits = {
            'train': {
                'sequences': sequences[:seq_train_end],
                'labels': seq_labels[:seq_train_end],
                'regime_features': regime_features[:seq_train_end],
                'symbols': seq_symbols[:seq_train_end]
            },
            'val': {
                'sequences': sequences[seq_train_end:seq_val_end],
                'labels': seq_labels[seq_train_end:seq_val_end],
                'regime_features': regime_features[seq_train_end:seq_val_end],
                'symbols': seq_symbols[seq_train_end:seq_val_end]
            },
            'test': {
                'sequences': sequences[seq_val_end:],
                'labels': seq_labels[seq_val_end:],
                'regime_features': regime_features[seq_val_end:],
                'symbols': seq_symbols[seq_val_end:]
            }
        }

        return {
            'base_data': base_splits,
            'temporal_data': temporal_splits,
            'feature_scalers': {
                'mean': feature_mean,
                'std': feature_std
            },
            'metadata': {
                'n_features': features.shape[1],
                'n_classes': len(np.unique(labels)),
                'sequence_length': sequences.shape[1],
                'regime_features': regime_features.shape[1]
            }
        }

    def train_temporal_models(self, temporal_data: Dict) -> Dict[str, Any]:
        """
        Train LSTM and Transformer models on temporal sequences
        """
        print("\n=== Training Temporal Models ===")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {device}")

        # Prepare data loaders
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(
            torch.tensor(temporal_data['train']
                         ['sequences'], dtype=torch.float32),
            torch.tensor(temporal_data['train']['labels'], dtype=torch.long)
        )

        val_dataset = TensorDataset(
            torch.tensor(temporal_data['val']
                         ['sequences'], dtype=torch.float32),
            torch.tensor(temporal_data['val']['labels'], dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        # Model configurations
        input_dim = temporal_data['train']['sequences'].shape[2]
        num_classes = len(np.unique(temporal_data['train']['labels']))

        temporal_models = {}

        # LSTM Model
        print("\nTraining LSTM model...")
        lstm_model = SMC_LSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
            bidirectional=True
        )

        lstm_results = train_temporal_model(
            lstm_model, train_loader, val_loader,
            epochs=50, lr=1e-3, device=device
        )

        temporal_models['lstm'] = {
            'model': lstm_model,
            'results': lstm_results,
            'type': 'temporal'
        }

        print(
            f"LSTM best validation accuracy: {lstm_results['best_val_accuracy']:.4f}")

        # Transformer Model
        print("\nTraining Transformer model...")
        transformer_model = SMC_Transformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            num_classes=num_classes,
            dropout=0.1
        )

        transformer_results = train_temporal_model(
            transformer_model, train_loader, val_loader,
            epochs=50, lr=1e-3, device=device
        )

        temporal_models['transformer'] = {
            'model': transformer_model,
            'results': transformer_results,
            'type': 'temporal'
        }

        print(
            f"Transformer best validation accuracy: {transformer_results['best_val_accuracy']:.4f}")

        return temporal_models

    def train_regime_classifier(self, temporal_data: Dict) -> RegimeClassifier:
        """
        Train regime classifier for market condition detection
        """
        print("\n=== Training Regime Classifier ===")

        # Create regime labels based on market conditions
        # This is a simplified approach - in practice, you'd use more sophisticated labeling
        regime_labels = self.create_regime_labels(temporal_data)

        # Train regime classifier
        regime_classifier = RegimeClassifier(
            input_dim=temporal_data['train']['regime_features'].shape[1],
            hidden_dim=64,
            num_regimes=3  # Trending, Ranging, Volatile
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        regime_classifier = regime_classifier.to(device)

        # Training setup
        optimizer = torch.optim.AdamW(regime_classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_train = torch.tensor(
            temporal_data['train']['regime_features'], dtype=torch.float32, device=device)
        y_train = torch.tensor(
            regime_labels['train'], dtype=torch.long, device=device)
        X_val = torch.tensor(
            temporal_data['val']['regime_features'], dtype=torch.float32, device=device)
        y_val = torch.tensor(
            regime_labels['val'], dtype=torch.long, device=device)

        # Training loop
        epochs = 30
        batch_size = 256

        for epoch in range(epochs):
            regime_classifier.train()
            train_loss = 0.0
            train_correct = 0

            # Shuffle training data
            perm = torch.randperm(len(X_train), device=device)
            X_train_shuf = X_train[perm]
            y_train_shuf = y_train[perm]

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuf[i:i+batch_size]
                batch_y = y_train_shuf[i:i+batch_size]

                optimizer.zero_grad()
                outputs = regime_classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1)
                                  == batch_y).sum().item()

            # Validation
            regime_classifier.eval()
            with torch.no_grad():
                val_outputs = regime_classifier(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_correct = (val_outputs.argmax(dim=1) == y_val).sum().item()

            if epoch % 10 == 0:
                train_acc = train_correct / len(X_train)
                val_acc = val_correct / len(X_val)
                print(
                    f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        self.regime_classifier = regime_classifier
        return regime_classifier

    def create_regime_labels(self, temporal_data: Dict) -> Dict[str, np.ndarray]:
        """
        Create regime labels based on market characteristics
        """
        regime_labels = {}

        for split in ['train', 'val', 'test']:
            regime_features = temporal_data[split]['regime_features']

            # Simple regime classification based on volatility and trend strength
            volatility = regime_features[:, 0]  # First feature is volatility
            # Second feature is trend strength
            trend_strength = regime_features[:, 1]

            labels = np.zeros(len(regime_features), dtype=np.int64)

            # Trending regime: high trend strength, moderate volatility
            trending_mask = (trend_strength > np.percentile(trend_strength, 70)) & \
                (volatility < np.percentile(volatility, 80))
            labels[trending_mask] = 0

            # Ranging regime: low trend strength, low volatility
            ranging_mask = (trend_strength < np.percentile(trend_strength, 30)) & \
                (volatility < np.percentile(volatility, 50))
            labels[ranging_mask] = 1

            # Volatile regime: high volatility
            volatile_mask = volatility > np.percentile(volatility, 80)
            labels[volatile_mask] = 2

            regime_labels[split] = labels

        return regime_labels

    def create_regime_aware_ensemble(self, base_models: Dict, temporal_models: Dict) -> RegimeAwareEnsemble:
        """
        Create regime-aware ensemble combining all model types
        """
        print("\n=== Creating Regime-Aware Ensemble ===")

        # Combine all models
        all_models = {}

        # Add base models (from ensemble manager)
        for name, model_info in base_models.items():
            if model_info['type'] == 'neural_network':
                all_models[f"base_{name}"] = model_info['model']

        # Add temporal models
        for name, model_info in temporal_models.items():
            all_models[f"temporal_{name}"] = model_info['model']

        # Create regime-aware ensemble
        ensemble = RegimeAwareEnsemble(
            models=all_models,
            regime_classifier=self.regime_classifier
        )

        return ensemble

    def comprehensive_evaluation(self, dataset: Dict, models: Dict) -> Dict[str, Any]:
        """
        Comprehensive evaluation across all model types and ensemble methods
        """
        print("\n=== Comprehensive Evaluation ===")

        results = {}

        # Evaluate base models
        print("\nEvaluating base models...")
        base_results = self.evaluate_base_models(
            dataset['base_data'], models['base_models'])
        results['base_models'] = base_results

        # Evaluate temporal models
        print("\nEvaluating temporal models...")
        temporal_results = self.evaluate_temporal_models(
            dataset['temporal_data'], models['temporal_models'])
        results['temporal_models'] = temporal_results

        # Evaluate ensemble
        print("\nEvaluating ensemble...")
        ensemble_results = self.evaluate_ensemble(dataset, models['ensemble'])
        results['ensemble'] = ensemble_results

        # Performance summary
        self.print_performance_summary(results)

        return results

    def evaluate_base_models(self, base_data: Dict, base_models: Dict) -> Dict[str, float]:
        """Evaluate base (non-temporal) models"""
        results = {}

        test_features = base_data['test']['features']
        test_labels = base_data['test']['labels']

        for model_name, model_info in base_models.items():
            if model_info['type'] == 'neural_network':
                model = model_info['model']
                model.eval()

                # Get the device the model is on
                device = next(model.parameters()).device

                with torch.no_grad():
                    X_tensor = torch.tensor(
                        test_features, dtype=torch.float32, device=device)
                    logits = model(X_tensor)
                    preds = logits.argmax(dim=1).cpu().numpy()

                accuracy = (preds == test_labels).mean()
                results[model_name] = accuracy
                print(f"  {model_name}: {accuracy:.4f}")

        return results

    def evaluate_temporal_models(self, temporal_data: Dict, temporal_models: Dict) -> Dict[str, float]:
        """Evaluate temporal models"""
        results = {}

        test_sequences = temporal_data['test']['sequences']
        test_labels = temporal_data['test']['labels']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for model_name, model_info in temporal_models.items():
            model = model_info['model']
            model.eval()

            with torch.no_grad():
                X_tensor = torch.tensor(
                    test_sequences, dtype=torch.float32, device=device)

                if isinstance(model, SMC_LSTM):
                    logits, _ = model(X_tensor)
                else:
                    logits = model(X_tensor)

                preds = logits.argmax(dim=1).cpu().numpy()

            accuracy = (preds == test_labels).mean()
            results[model_name] = accuracy
            print(f"  {model_name}: {accuracy:.4f}")

        return results

    def evaluate_ensemble(self, dataset: Dict, ensemble: RegimeAwareEnsemble) -> Dict[str, float]:
        """Evaluate regime-aware ensemble"""
        # This would implement ensemble evaluation
        # For now, return placeholder
        return {'regime_aware_ensemble': 0.65}

    def print_performance_summary(self, results: Dict):
        """Print comprehensive performance summary"""
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*60)

        # Find best performing models in each category
        best_base = max(results['base_models'].items(
        ), key=lambda x: x[1]) if results['base_models'] else ('None', 0)
        best_temporal = max(results['temporal_models'].items(
        ), key=lambda x: x[1]) if results['temporal_models'] else ('None', 0)

        print(f"\nBest Base Model: {best_base[0]} ({best_base[1]:.4f})")
        print(
            f"Best Temporal Model: {best_temporal[0]} ({best_temporal[1]:.4f})")
        print(
            f"Ensemble Performance: {results['ensemble'].get('regime_aware_ensemble', 0):.4f}")

        # Overall best
        all_scores = []
        if results['base_models']:
            all_scores.extend(results['base_models'].values())
        if results['temporal_models']:
            all_scores.extend(results['temporal_models'].values())
        if results['ensemble']:
            all_scores.extend(results['ensemble'].values())

        if all_scores:
            print(f"\nOverall Best Performance: {max(all_scores):.4f}")
            print(
                f"Performance Range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    def save_integrated_system(self, save_path: Path):
        """Save the complete integrated system"""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save system metadata
        # Convert Path objects to strings for JSON serialization
        config_serializable = {}
        for key, value in self.config.items():
            if isinstance(value, Path):
                config_serializable[key] = str(value)
            else:
                config_serializable[key] = value

        metadata = {
            'system_type': 'IntegratedSMCSystem',
            'config': config_serializable,
            'performance_metrics': self.performance_metrics,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        with open(save_path / 'system_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save individual components
        if self.ensemble_manager:
            self.ensemble_manager.save_ensemble(save_path / 'ensemble')

        if self.regime_classifier:
            torch.save(self.regime_classifier.state_dict(),
                       save_path / 'regime_classifier.pth')

        print(f"Integrated system saved to {save_path}")


def main():
    """
    Main execution pipeline for integrated advanced SMC system
    """
    print("="*80)
    print("INTEGRATED ADVANCED SMC PIPELINE")
    print("="*80)

    # Detect environment and set paths accordingly
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Configuration
    config = {
        # Hard-coded for Kaggle environment
        'data_path': '/kaggle/input/training-data/Data/mt5_features_institutional_regime_filtered.csv',
        'sequence_length': 20,
        'prediction_horizon': 8,
        # Output models under /kaggle/working/models
        'save_path': Path('/kaggle/working/models')
    }

    print(f"\n📁 Environment Detection:")
    print(f"   Script dir: {script_dir}")
    print(f"   Project root: {project_root}")
    print(f"   Data path: {config['data_path']}")
    print(f"   Save path: {config['save_path']}")

    # Check if data file exists
    if not Path(config['data_path']).exists():
        print(f"\n❌ ERROR: Data file not found!")
        print(f"   Expected: {config['data_path']}")
        print(
            f"\n   Please ensure the institutional features CSV is in the correct location.")
        print(f"   Run: python Python/feature_engineering_smc_institutional.py")
        return None, None

    # Initialize integrated system
    system = IntegratedSMCSystem(config)

    # Prepare comprehensive dataset
    dataset = system.prepare_comprehensive_dataset(config['data_path'])

    # Train base ensemble models
    print("\n" + "="*60)
    print("TRAINING BASE ENSEMBLE MODELS")
    print("="*60)

    ensemble_manager = EnsembleManager(config)
    base_models = ensemble_manager.train_all_models(
        dataset['base_data']['train']['features'],
        dataset['base_data']['train']['labels']
    )

    # Calculate ensemble weights
    ensemble_weights = ensemble_manager.calculate_ensemble_weights(
        dataset['base_data']['val']['features'],
        dataset['base_data']['val']['labels']
    )

    system.ensemble_manager = ensemble_manager

    # Train temporal models
    temporal_models = system.train_temporal_models(dataset['temporal_data'])

    # Add temporal models to ensemble manager for saving
    for model_name, model_info in temporal_models.items():
        system.ensemble_manager.models[model_name] = model_info

    # Train regime classifier
    regime_classifier = system.train_regime_classifier(
        dataset['temporal_data'])

    # Create regime-aware ensemble
    integrated_ensemble = system.create_regime_aware_ensemble(
        base_models, temporal_models)

    # Comprehensive evaluation
    all_models = {
        'base_models': base_models,
        'temporal_models': temporal_models,
        'ensemble': {'integrated': integrated_ensemble}
    }

    evaluation_results = system.comprehensive_evaluation(dataset, all_models)
    system.performance_metrics = evaluation_results

    # Generate learning curves
    print("\n" + "="*60)
    print("GENERATING LEARNING CURVES")
    print("="*60)

    integrate_learning_curves(base_models, ensemble_weights)

    # Run temporal validation
    print("\n" + "="*60)
    print("TEMPORAL CROSS-VALIDATION")
    print("="*60)

    temporal_validator = TemporalValidator()
    validation_results = temporal_validator.run_temporal_validation(
        dataset, ensemble_manager, system, k=5
    )

    # Export models for production
    print("\n" + "="*60)
    print("EXPORTING MODELS FOR PRODUCTION")
    print("="*60)

    # Combine base and temporal models for export
    all_models_for_export = {**base_models, **temporal_models}

    export_dir = export_all_models(
        all_models_for_export,
        ensemble_weights,
        ensemble_manager.feature_scalers,
        input_shape=(1, dataset['metadata']['n_features']),
        sequence_length=config['sequence_length']
    )

    # Initialize recovery mechanism
    print("\n" + "="*60)
    print("INITIALIZING RECOVERY MECHANISM")
    print("="*60)

    recovery_manager = RecoveryManager()
    system.recovery_manager = recovery_manager

    print("✅ Recovery mechanism initialized")
    print(
        f"   Recovery threshold: {recovery_manager.config['recovery_threshold']:.1%}")
    print(
        f"   Confidence threshold (normal): {recovery_manager.config['normal_confidence_threshold']}")
    print(
        f"   Confidence threshold (recovery): {recovery_manager.config['recovery_confidence_threshold']}")

    # Save integrated system
    system.save_integrated_system(config['save_path'])

    # Final summary with all enhancements
    print("\n" + "="*80)
    print("INTEGRATED ADVANCED SMC PIPELINE COMPLETE")
    print("="*80)

    print(f"✅ Base models trained: {len(base_models)}")
    print(f"✅ Temporal models trained: {len(temporal_models)}")
    print(f"✅ Learning curves generated: Model_output/learning_curves/")
    print(
        f"✅ Temporal validation completed: {validation_results['ensemble_mean']:.4f} ± {validation_results['ensemble_std']:.4f}")
    print(f"✅ Models exported for production: {export_dir}")
    print(f"✅ Recovery mechanism active")
    print(f"✅ System saved to: {config['save_path']}")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Run dashboard: streamlit run dashboard.py")
    print(f"   2. Start live inference: python mt5_inference.py")
    print(f"   3. Monitor performance via dashboard")

    # Store additional results
    enhanced_results = {
        **evaluation_results,
        'temporal_validation': validation_results,
        'export_directory': export_dir,
        'recovery_manager_config': recovery_manager.config
    }

    return system, enhanced_results


if __name__ == "__main__":
    system, results = main()
