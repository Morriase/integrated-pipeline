"""
SMC Pipeline - One Cell Kaggle Execution
Copy and paste this entire cell into a Kaggle notebook
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/Morriase/integrated-pipeline.git"
REPO_DIR = "/kaggle/working/integrated-pipeline"

print("="*80)
print("🚀 SMC PIPELINE - KAGGLE EXECUTION")
print("="*80)

# Step 1: Pull or clone repository
print("\n📦 Setting up repository...")
if Path(REPO_DIR).exists():
    print("  Pulling latest changes...")
    os.chdir(REPO_DIR)
    subprocess.run("git stash", shell=True, capture_output=True)
    result = subprocess.run("git pull origin main", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("  Pull failed, cloning fresh...")
        os.chdir("/kaggle/working")
        subprocess.run(f"rm -rf {REPO_DIR}", shell=True)
        subprocess.run(f"git clone {REPO_URL}", shell=True, check=True)
        os.chdir(REPO_DIR)
    else:
        print("  ✅ Repository updated")
else:
    print("  Cloning repository...")
    os.chdir("/kaggle/working")
    subprocess.run(f"git clone {REPO_URL}", shell=True, check=True)
    os.chdir(REPO_DIR)

# Show latest commit
result = subprocess.run("git log -1 --oneline", shell=True, capture_output=True, text=True)
print(f"  📝 Latest: {result.stdout.strip()}")

# Step 2: Install dependencies
print("\n📦 Installing dependencies...")
packages = ["torch", "scikit-fuzzy", "xgboost", "scikit-learn", "imbalanced-learn", 
            "pandas", "numpy", "joblib", "matplotlib", "seaborn"]
for pkg in packages:
    subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True)
print("  ✅ Dependencies installed")

# Step 3: Setup environment
sys.path.insert(0, REPO_DIR)

# Step 4: Verify Task 12
print("\n🔍 Verifying Task 12 implementation...")
task12_files = ["TASK_12_COMPLETE.md", "PERFORMANCE_REPORTING_QUICK_REFERENCE.md"]
if all(Path(f).exists() for f in task12_files):
    print("  ✅ Task 12 files present")
else:
    print("  ⚠️ Task 12 files missing - using older version")

# Step 5: Run pipeline
print("\n🚀 Running complete pipeline...")
print("="*80 + "\n")

exec(open('KAGGLE_RUN.py').read())

print("\n" + "="*80)
print("✅ EXECUTION COMPLETE")
print("="*80)

# Show outputs
reports_dir = Path("models/trained/reports")
if reports_dir.exists():
    reports = list(reports_dir.glob("*.md"))
    print(f"\n📄 Generated {len(reports)} performance reports")
    for r in reports:
        print(f"  • {r.name}")

if Path("models/trained/deployment_manifest.json").exists():
    print("\n📋 Deployment manifest created")
if Path("models/trained/training_results.json").exists():
    print("📊 Training results saved")

print("\n🎉 Done! Check models/trained/ for outputs")
