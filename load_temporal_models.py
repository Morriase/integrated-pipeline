"""
Helper script to load temporal models from state_dict files
"""

import torch
import json
from pathlib import Path
from advanced_temporal_architecture import SMC_LSTM, SMC_Transformer


def load_temporal_model(model_path: str, architecture_config_path: str = None):
    """
    Load a temporal model from state_dict
    
    Args:
        model_path: Path to the state_dict .pth file
        architecture_config_path: Path to model_architectures.json (optional)
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    # Try to find architecture config
    if architecture_config_path is None:
        # Look in same directory
        arch_path = model_path.parent / "model_architectures.json"
        if not arch_path.exists():
            # Try metadata file
            metadata_path = model_path.parent / f"{model_path.stem.replace('_state_dict', '')}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Use metadata to reconstruct
                    return load_from_metadata(model_path, metadata)
            else:
                raise FileNotFoundError(
                    f"Could not find architecture config. Please provide architecture_config_path or ensure model_architectures.json exists"
                )
    else:
        arch_path = Path(architecture_config_path)
    
    # Load architecture config
    with open(arch_path, 'r') as f:
        architectures = json.load(f)
    
    # Get model name from path
    model_name = model_path.stem.replace('_state_dict', '')
    
    if model_name not in architectures:
        raise ValueError(f"Model '{model_name}' not found in architecture config")
    
    arch = architectures[model_name]
    
    # Reconstruct model
    if arch['model_class'] == 'SMC_LSTM':
        model = SMC_LSTM(
            input_dim=arch['input_dim'],
            hidden_dim=arch.get('hidden_dim', 128),
            num_layers=arch.get('num_layers', 2),
            num_classes=arch.get('num_classes', 3),
            dropout=arch.get('dropout', 0.4),
            bidirectional=arch.get('bidirectional', True)
        )
    elif arch['model_class'] == 'SMC_Transformer':
        model = SMC_Transformer(
            input_dim=arch['input_dim'],
            d_model=arch.get('d_model', 128),
            nhead=arch.get('nhead', 8),
            num_layers=arch.get('num_layers', 3),
            num_classes=arch.get('num_classes', 3),
            dropout=arch.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model class: {arch['model_class']}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Loaded {model_name} ({arch['model_class']})")
    
    return model


def load_from_metadata(model_path: Path, metadata: dict):
    """Load model using metadata file"""
    
    model_class = metadata.get('model_class')
    config = metadata.get('config', {})
    
    # Try to extract architecture params from metadata
    if model_class == 'SMC_LSTM':
        model = SMC_LSTM(
            input_dim=config.get('input_dim', 29),  # Default to 29 features
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 3),
            dropout=config.get('dropout', 0.4),
            bidirectional=config.get('bidirectional', True)
        )
    elif model_class == 'SMC_Transformer':
        model = SMC_Transformer(
            input_dim=config.get('input_dim', 29),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 3),
            num_classes=config.get('num_classes', 3),
            dropout=config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Loaded {model_path.stem} ({model_class}) from metadata")
    
    return model


def load_all_temporal_models(deployment_dir: str = "Model_output/deployment"):
    """
    Load all temporal models from deployment directory
    
    Args:
        deployment_dir: Path to deployment directory
        
    Returns:
        Dictionary of loaded models
    """
    deployment_dir = Path(deployment_dir)
    
    if not deployment_dir.exists():
        raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")
    
    # Find all state_dict files
    state_dict_files = list(deployment_dir.glob("*_state_dict.pth"))
    
    if not state_dict_files:
        print("⚠️ No temporal model state_dict files found")
        return {}
    
    # Try to load architecture config
    arch_path = deployment_dir / "model_architectures.json"
    
    models = {}
    
    for state_dict_file in state_dict_files:
        model_name = state_dict_file.stem.replace('_state_dict', '')
        
        try:
            if arch_path.exists():
                model = load_temporal_model(state_dict_file, arch_path)
            else:
                # Try loading from metadata
                metadata_path = deployment_dir / f"{model_name}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model = load_from_metadata(state_dict_file, metadata)
                else:
                    print(f"⚠️ Skipping {model_name}: No architecture config or metadata found")
                    continue
            
            models[model_name] = model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    print(f"\n✅ Loaded {len(models)} temporal models")
    return models


# Example usage
if __name__ == "__main__":
    print("=== Temporal Model Loader ===\n")
    
    # Load all temporal models
    models = load_all_temporal_models("Model_output/deployment")
    
    # Test inference
    if models:
        print("\n=== Testing Inference ===")
        
        import numpy as np
        
        for model_name, model in models.items():
            print(f"\nTesting {model_name}...")
            
            # Create dummy input
            if 'lstm' in model_name.lower():
                # LSTM expects (batch, seq_len, features)
                dummy_input = torch.randn(1, 20, 29)
            elif 'transformer' in model_name.lower():
                # Transformer expects (batch, seq_len, features)
                dummy_input = torch.randn(1, 20, 29)
            else:
                dummy_input = torch.randn(1, 29)
            
            # Run inference
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        output = output[0]  # LSTM returns (output, attention)
                    
                    probs = torch.softmax(output, dim=1)
                    pred = output.argmax(dim=1).item()
                    conf = probs.max().item()
                    
                    print(f"  Prediction: {pred} (confidence: {conf:.4f})")
                    print(f"  Output shape: {output.shape}")
        
        print("\n✅ All models tested successfully!")
