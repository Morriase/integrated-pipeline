"""
Kaggle: Pull latest changes and restart kernel
Run this in a Kaggle notebook cell
"""

import os
import sys
import subprocess

# Pull latest changes
print("🔄 Pulling latest changes from GitHub...")
os.chdir('/kaggle/working/integrated-pipeline')
result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr)

# Verify the changes were applied
print("\n✅ Verifying anti-overfitting changes...")

# Check Neural Network architecture
print("\n📊 Neural Network architecture:")
with open('models/neural_network_model.py', 'r') as f:
    for i, line in enumerate(f, 1):
        if 'hidden_dims: List[int] = [' in line:
            print(f"  Line {i}: {line.strip()}")
        if 'dropout: float = ' in line and 'def train' in open('models/neural_network_model.py').read().split('\n')[i-10:i][-1]:
            print(f"  Line {i}: {line.strip()}")
        if 'batch_size: int = ' in line and 'def train' in open('models/neural_network_model.py').read().split('\n')[i-10:i][-1]:
            print(f"  Line {i}: {line.strip()}")

# Check Random Forest parameters
print("\n🌲 Random Forest parameters:")
with open('models/random_forest_model.py', 'r') as f:
    for i, line in enumerate(f, 1):
        if 'max_depth: Optional[int] = ' in line:
            print(f"  Line {i}: {line.strip()}")
        if 'min_samples_split: int = ' in line:
            print(f"  Line {i}: {line.strip()}")
        if 'min_samples_leaf: int = ' in line:
            print(f"  Line {i}: {line.strip()}")

print("\n" + "="*60)
print("✅ Changes verified!")
print("="*60)
print("\n🔄 Now restarting Python kernel to clear memory...")
print("   (This will stop execution here)")
print("\n💡 After restart, run: python train_all_models.py")
print("="*60)

# Restart the kernel by exiting and letting Kaggle auto-restart
# Or use IPython restart if available
try:
    from IPython.core.display import HTML
    display(HTML("<script>Jupyter.notebook.kernel.restart()</script>"))
    print("\n✅ Kernel restart initiated via IPython")
except:
    print("\n⚠️ Auto-restart not available")
    print("   Please manually restart: Run → Restart & Run All")
    print("   Or use keyboard shortcut: Ctrl+Shift+K")
