"""
Quick script to pull and verify changes in Kaggle
Copy-paste this into a Kaggle notebook cell and run
"""

import os
import subprocess

# Navigate and pull
os.chdir('/kaggle/working/integrated-pipeline')
print("🔄 Pulling latest changes...")
subprocess.run(['git', 'pull', 'origin', 'main'])

# Quick verification
print("\n✅ Checking Neural Network config:")
os.system("grep 'hidden_dims: List\\[int\\] = \\[' models/neural_network_model.py")
os.system("grep 'dropout: float = 0\\.' models/neural_network_model.py | head -1")
os.system("grep 'batch_size: int = ' models/neural_network_model.py | head -1")

print("\n✅ Checking Random Forest config:")
os.system("grep 'max_depth: Optional\\[int\\] = ' models/random_forest_model.py")
os.system("grep 'min_samples_split: int = ' models/random_forest_model.py | head -1")
os.system("grep 'min_samples_leaf: int = ' models/random_forest_model.py | head -1")

print("\n" + "="*60)
print("✅ Pull complete! Now restart kernel:")
print("   1. Click 'Run' → 'Restart & Run All'")
print("   2. Or press Ctrl+Shift+K (Windows) / Cmd+Shift+K (Mac)")
print("="*60)
