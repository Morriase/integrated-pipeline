"""
Model Export Module
Exports trained models to ONNX and TorchScript formats for production deployment
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import warnings


class ModelExporter:
    """
    Handles exporting models to various production-ready formats
    """

    def __init__(self, export_dir: str = "Model_output/deployment"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.exported_models = {}

    def export_pytorch_model(self, model: nn.Module, model_name: str,
                             input_shape: tuple, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Export PyTorch model to ONNX and TorchScript formats

        Args:
            model: PyTorch model to export
            model_name: Name for the exported model
            input_shape: Input tensor shape (batch_size, features)
            metadata: Additional metadata about the model

        Returns:
            Dictionary with export paths
        """
        print(f"\nExporting {model_name} to production formats...")

        device = next(model.parameters()).device
        model.eval()

        export_paths = {}

        # Create dummy input for export
        dummy_input = torch.randn(input_shape).to(device)

        try:
            # Export to ONNX
            onnx_path = self.export_dir / f"{model_name}.onnx"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

            export_paths['onnx'] = str(onnx_path)
            print(f"✅ ONNX export successful: {onnx_path}")

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            export_paths['onnx'] = None

        try:
            # Export to TorchScript
            torchscript_path = self.export_dir / f"{model_name}.pt"

            # Use tracing for better compatibility
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(torchscript_path))

            export_paths['torchscript'] = str(torchscript_path)
            print(f"✅ TorchScript export successful: {torchscript_path}")

        except Exception as e:
            print(f"❌ TorchScript export failed: {e}")
            export_paths['torchscript'] = None

        # Save model metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': 'pytorch_neural_network',
            'input_shape': list(input_shape),
            'export_formats': [k for k, v in export_paths.items() if v is not None],
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'device_trained_on': str(device),
            **metadata
        }

        metadata_path = self.export_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        export_paths['metadata'] = str(metadata_path)
        print(f"✅ Metadata saved: {metadata_path}")

        self.exported_models[model_name] = {
            'paths': export_paths,
            'metadata': model_metadata
        }

        return export_paths

    def export_sklearn_model(self, model, model_name: str,
                             metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Export scikit-learn model to pickle format

        Args:
            model: Scikit-learn model to export
            model_name: Name for the exported model
            metadata: Additional metadata about the model

        Returns:
            Dictionary with export paths
        """
        print(f"\nExporting {model_name} (sklearn) to production format...")

        export_paths = {}

        try:
            # Export to pickle
            pickle_path = self.export_dir / f"{model_name}.pkl"

            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)

            export_paths['pickle'] = str(pickle_path)
            print(f"✅ Pickle export successful: {pickle_path}")

        except Exception as e:
            print(f"❌ Pickle export failed: {e}")
            export_paths['pickle'] = None

        # Save model metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': 'sklearn_model',
            'model_class': model.__class__.__name__,
            'export_formats': [k for k, v in export_paths.items() if v is not None],
            'exported_at': datetime.now(timezone.utc).isoformat(),
            **metadata
        }

        metadata_path = self.export_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        export_paths['metadata'] = str(metadata_path)
        print(f"✅ Metadata saved: {metadata_path}")

        self.exported_models[model_name] = {
            'paths': export_paths,
            'metadata': model_metadata
        }

        return export_paths

    def export_model_architectures(self, model_results: Dict) -> str:
        """
        Export model architecture configurations for reconstruction
        
        Args:
            model_results: Dictionary of trained models
            
        Returns:
            Path to exported architecture config
        """
        print("\nExporting model architecture configurations...")
        
        architectures = {}
        
        for model_name, model_info in model_results.items():
            if model_info['type'] == 'temporal':
                model = model_info['model']
                
                # Extract architecture parameters
                if hasattr(model, 'input_dim'):
                    architectures[model_name] = {
                        'model_class': model.__class__.__name__,
                        'input_dim': model.input_dim,
                        'hidden_dim': getattr(model, 'hidden_dim', None),
                        'num_layers': getattr(model, 'num_layers', None),
                        'd_model': getattr(model, 'd_model', None),
                        'nhead': getattr(model, 'nhead', None) if hasattr(model, 'nhead') else None,
                        'num_classes': getattr(model, 'num_classes', 3),
                        'dropout': getattr(model, 'dropout', 0.3),
                        'bidirectional': getattr(model, 'bidirectional', True) if hasattr(model, 'bidirectional') else None
                    }
                    # Remove None values
                    architectures[model_name] = {k: v for k, v in architectures[model_name].items() if v is not None}
        
        arch_path = self.export_dir / "model_architectures.json"
        
        try:
            with open(arch_path, 'w') as f:
                json.dump(architectures, f, indent=2)
            
            print(f"✅ Model architectures exported: {arch_path}")
            
        except Exception as e:
            print(f"❌ Model architectures export failed: {e}")
            return None
        
        return str(arch_path)
    
    def export_feature_scalers(self, feature_scalers: Dict[str, np.ndarray]) -> str:
        """
        Export feature scalers for preprocessing in production

        Args:
            feature_scalers: Dictionary containing mean and std for feature scaling

        Returns:
            Path to exported scalers
        """
        print("\nExporting feature scalers...")

        scalers_path = self.export_dir / "feature_scalers.pkl"

        try:
            with open(scalers_path, 'wb') as f:
                pickle.dump(feature_scalers, f)

            print(f"✅ Feature scalers exported: {scalers_path}")

            # Also save as JSON for easier inspection
            scalers_json_path = self.export_dir / "feature_scalers.json"
            scalers_serializable = {
                'mean': feature_scalers['mean'].tolist(),
                'std': feature_scalers['std'].tolist()
            }

            with open(scalers_json_path, 'w') as f:
                json.dump(scalers_serializable, f, indent=2)

            print(f"✅ Feature scalers JSON exported: {scalers_json_path}")

        except Exception as e:
            print(f"❌ Feature scalers export failed: {e}")
            return None

        return str(scalers_path)

    def export_ensemble_config(self, ensemble_weights: Dict[str, float],
                               model_configs: Dict[str, Any]) -> str:
        """
        Export ensemble configuration for production inference

        Args:
            ensemble_weights: Dictionary of model weights
            model_configs: Dictionary of model configurations

        Returns:
            Path to exported ensemble config
        """
        print("\nExporting ensemble configuration...")

        ensemble_config = {
            'ensemble_weights': ensemble_weights,
            'model_configs': model_configs,
            'exported_models': {name: info['metadata'] for name, info in self.exported_models.items()},
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        config_path = self.export_dir / "ensemble_config.json"

        try:
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)

            print(f"✅ Ensemble config exported: {config_path}")

        except Exception as e:
            print(f"❌ Ensemble config export failed: {e}")
            return None

        return str(config_path)

    def create_deployment_summary(self) -> str:
        """
        Create a comprehensive deployment summary

        Returns:
            Path to deployment summary file
        """
        print("\nCreating deployment summary...")

        summary = {
            'deployment_info': {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'total_models_exported': len(self.exported_models),
                'export_directory': str(self.export_dir)
            },
            'exported_models': self.exported_models,
            'deployment_instructions': {
                'pytorch_models': "Load using torch.jit.load() for TorchScript or onnxruntime for ONNX",
                'sklearn_models': "Load using pickle.load()",
                'feature_scalers': "Load feature_scalers.pkl and apply: (features - mean) / std",
                'ensemble_inference': "Load ensemble_config.json for model weights and configurations"
            }
        }

        summary_path = self.export_dir / "deployment_summary.json"

        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"✅ Deployment summary created: {summary_path}")

            # Also create a human-readable README
            readme_path = self.export_dir / "README.md"
            self._create_deployment_readme(readme_path, summary)

        except Exception as e:
            print(f"❌ Deployment summary creation failed: {e}")
            return None

        return str(summary_path)

    def _create_deployment_readme(self, readme_path: Path, summary: Dict):
        """Create a human-readable deployment README"""

        with open(readme_path, 'w') as f:
            f.write("# Black Ice AI - Model Deployment Package\n\n")
            f.write(
                f"**Created:** {summary['deployment_info']['created_at']}\n")
            f.write(
                f"**Models Exported:** {summary['deployment_info']['total_models_exported']}\n\n")

            f.write("## Exported Models\n\n")
            for model_name, model_info in summary['exported_models'].items():
                f.write(f"### {model_name}\n")
                f.write(
                    f"- **Type:** {model_info['metadata']['model_type']}\n")
                f.write(
                    f"- **Formats:** {', '.join(model_info['metadata']['export_formats'])}\n")
                if 'input_shape' in model_info['metadata']:
                    f.write(
                        f"- **Input Shape:** {model_info['metadata']['input_shape']}\n")
                f.write("\n")

            f.write("## Quick Start\n\n")
            f.write("### Loading Standard Models (TorchScript)\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write("model = torch.jit.load('model_name.pt')\n")
            f.write("```\n\n")
            
            f.write("### Loading Temporal Models (State Dict)\n")
            f.write("```python\n")
            f.write("from load_temporal_models import load_temporal_model\n\n")
            f.write("# Load LSTM or Transformer\n")
            f.write("model = load_temporal_model('lstm_state_dict.pth')\n")
            f.write("# or\n")
            f.write("model = load_temporal_model('transformer_state_dict.pth')\n")
            f.write("```\n\n")
            
            f.write("### Feature Preprocessing\n")
            f.write("```python\n")
            f.write("import pickle\n")
            f.write("with open('feature_scalers.pkl', 'rb') as f:\n")
            f.write("    scalers = pickle.load(f)\n\n")
            f.write("# Normalize features\n")
            f.write("features_normalized = (features - scalers['mean']) / scalers['std']\n")
            f.write("```\n\n")
            
            f.write("### Making Predictions\n")
            f.write("```python\n")
            f.write("import torch\n\n")
            f.write("# Standard models\n")
            f.write("with torch.no_grad():\n")
            f.write("    output = model(torch.tensor(features_normalized, dtype=torch.float32))\n")
            f.write("    prediction = output.argmax(dim=1).item()\n\n")
            f.write("# Temporal models (need sequences)\n")
            f.write("with torch.no_grad():\n")
            f.write("    # Input shape: (batch, seq_len, features)\n")
            f.write("    sequence = torch.tensor(sequence_data, dtype=torch.float32)\n")
            f.write("    output = model(sequence)\n")
            f.write("    if isinstance(output, tuple):  # LSTM returns (output, attention)\n")
            f.write("        output = output[0]\n")
            f.write("    prediction = output.argmax(dim=1).item()\n")
            f.write("```\n\n")

            f.write("## Files\n\n")
            f.write("- `*.pt` - TorchScript models (standard neural networks)\n")
            f.write("- `*_state_dict.pth` - PyTorch state dicts (temporal models: LSTM, Transformer)\n")
            f.write("- `*.pkl` - Pickle format (sklearn models and scalers)\n")
            f.write("- `model_architectures.json` - Architecture configs for temporal models\n")
            f.write("- `*_metadata.json` - Model metadata and configuration\n")
            f.write("- `ensemble_config.json` - Ensemble weights and configuration\n")
            f.write("- `deployment_summary.json` - Complete deployment information\n")
            f.write("- `load_temporal_models.py` - Helper script to load temporal models\n")

        print(f"✅ Deployment README created: {readme_path}")


def export_all_models(model_results: Dict, ensemble_weights: Dict,
                      feature_scalers: Dict, input_shape: tuple = (1, 29),
                      sequence_length: int = 20) -> str:
    """
    Export all trained models to production formats

    Args:
        model_results: Dictionary of trained models
        ensemble_weights: Dictionary of ensemble weights
        feature_scalers: Feature scaling parameters
        input_shape: Input tensor shape for neural networks
        sequence_length: Sequence length for temporal models

    Returns:
        Path to deployment directory
    """
    print("\n=== Exporting Models for Production Deployment ===")

    exporter = ModelExporter()

    # Export neural network models
    for model_name, model_info in model_results.items():
        if model_info['type'] == 'neural_network':
            metadata = {
                'accuracy': model_info['accuracy'],
                'config': model_info.get('config', {}),
                'training_history': model_info.get('history', {}),
                'model_architecture': 'feedforward'
            }

            exporter.export_pytorch_model(
                model_info['model'],
                model_name,
                input_shape,
                metadata
            )

        elif model_info['type'] == 'temporal':
            # Temporal models - save as state_dict for easier loading
            print(f"\nExporting {model_name} (temporal) as PyTorch state_dict...")
            
            state_dict_path = exporter.export_dir / f"{model_name}_state_dict.pth"
            
            try:
                # Save state dict
                torch.save(model_info['model'].state_dict(), state_dict_path)
                print(f"✅ State dict saved: {state_dict_path}")
                
                # Save model metadata
                metadata = {
                    'model_name': model_name,
                    'model_type': 'temporal',
                    'model_class': model_info['model'].__class__.__name__,
                    'accuracy': model_info['results']['best_val_accuracy'],
                    'config': model_info.get('config', {}),
                    'training_history': model_info['results'],
                    'model_architecture': 'temporal',
                    'sequence_length': sequence_length,
                    'input_shape': [1, sequence_length, input_shape[1]],
                    'exported_at': datetime.now(timezone.utc).isoformat(),
                    'format': 'pytorch_state_dict'
                }
                
                metadata_path = exporter.export_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Metadata saved: {metadata_path}")
                
                exporter.exported_models[model_name] = {
                    'paths': {
                        'state_dict': str(state_dict_path),
                        'metadata': str(metadata_path)
                    },
                    'metadata': metadata
                }
                
            except Exception as e:
                print(f"❌ Temporal model export failed: {e}")

        elif model_info['type'] in ['sklearn', 'tree_ensemble']:
            metadata = {
                'accuracy': model_info['accuracy'],
                'cv_scores': model_info.get('cv_scores', []),
                'config': model_info.get('config', {})
            }

            exporter.export_sklearn_model(
                model_info['model'],
                model_name,
                metadata
            )

    # Export model architectures (for temporal models)
    exporter.export_model_architectures(model_results)
    
    # Export feature scalers
    exporter.export_feature_scalers(feature_scalers)

    # Export ensemble configuration
    model_configs = {name: info.get('config', {})
                     for name, info in model_results.items()}
    exporter.export_ensemble_config(ensemble_weights, model_configs)

    # Create deployment summary
    exporter.create_deployment_summary()

    print(f"\n✅ All models exported successfully to: {exporter.export_dir}")
    return str(exporter.export_dir)
