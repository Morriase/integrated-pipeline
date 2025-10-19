"""
Run this in a Kaggle Notebook Cell
Copy and paste this entire cell into Kaggle
"""

# Step 1: Remove old repo if exists
import shutil
from pathlib import Path

repo_dir = Path("/kaggle/working/integrated-pipeline")
if repo_dir.exists():
    print("🗑️  Removing old repository...")
    shutil.rmtree(repo_dir)
    print("✅ Removed")

# Step 2: Clone fresh repository
print("\n📥 Cloning repository...")
import subprocess
result = subprocess.run(
    ["git", "clone", "https://github.com/Morriase/integrated-pipeline.git", str(repo_dir)],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("✅ Repository cloned")
else:
    print(f"❌ Clone failed: {result.stderr}")
    raise Exception("Failed to clone repository")

# Step 3: Change to repo directory
import os
os.chdir(repo_dir)
print(f"📂 Changed to: {os.getcwd()}")

# Step 4: Verify the fix is present
print("\n🔍 Verifying RandomForest fix...")
with open("models/random_forest_model.py", "r") as f:
    content = f.read()
    if "cv_results.get('is_stable'" in content:
        print("✅ RandomForest fix is present")
    else:
        print("❌ Fix not found - did you push to GitHub?")
        raise Exception("RandomForest fix not found in repository")

# Step 5: Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([
    "pip", "install", "-q",
    "scikit-fuzzy", "xgboost", "imbalanced-learn"
], check=True)
print("✅ Dependencies installed")

# Step 6: Add to Python path
sys.path.insert(0, str(repo_dir))
print(f"✅ Added to Python path: {repo_dir}")

# Step 7: Run the complete pipeline
print("\n" + "="*80)
print("🚀 STARTING COMPLETE PIPELINE")
print("="*80)
print("\nThis will take approximately 35-40 minutes...")
print("Training 11 symbols × 4 models = 44 models total")
print("\n")

# Import and run
from run_complete_pipeline import main
main()

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE!")
print("="*80)
print("\nCheck the output files in /kaggle/working/")
print("- training_results.json")
print("- Model files: *_XGBoost.pkl, *_RandomForest.pkl, etc.")
print("- Reports in models/trained/reports/")
