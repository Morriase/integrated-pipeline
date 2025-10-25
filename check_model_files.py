"""
Check if model files exist
"""

from pathlib import Path

model_dir = Path('/kaggle/working/Model-output')

print("=" * 80)
print("MODEL FILES CHECK")
print("=" * 80)

models = ['RandomForest', 'XGBoost', 'NeuralNetwork']

for model_name in models:
    model_path = model_dir / f'UNIFIED_{model_name}.pkl'
    metadata_path = model_dir / f'UNIFIED_{model_name}_metadata.json'
    
    print(f"\n{model_name}:")
    print(f"  Model file: {model_path.exists()} - {model_path}")
    print(f"  Metadata:   {metadata_path.exists()} - {metadata_path}")
    
    if model_path.exists():
        size = model_path.stat().st_size / 1024  # KB
        print(f"  Size: {size:.1f} KB")

print(f"\n📂 All files in {model_dir}:")
for f in sorted(model_dir.glob('*')):
    print(f"  - {f.name}")
