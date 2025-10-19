"""
SMC Pipeline - Pull Latest Changes and Run
Repository: https://github.com/Morriase/integrated-pipeline.git

This script:
1. Pulls latest changes from git (or clones if needed)
2. Installs dependencies
3. Verifies Task 12 implementation
4. Runs the complete training pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/Morriase/integrated-pipeline.git"
REPO_DIR = "/kaggle/working/integrated-pipeline"

def run_command(cmd, description, check=True):
    """Run shell command with error handling"""
    print(f"\n{'='*80}")
    print(f"🔄 {description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        if check:
            sys.exit(1)
        return False
    
    print(f"✅ {description} - Complete")
    return True

# Step 1: Clone or pull repository
print("="*80)
print("📦 REPOSITORY SETUP")
print("="*80)

if Path(REPO_DIR).exists():
    print(f"\n📂 Repository exists at {REPO_DIR}")
    print("🔄 Pulling latest changes...")
    
    os.chdir(REPO_DIR)
    
    # Stash any local changes
    subprocess.run("git stash", shell=True, capture_output=True)
    
    # Pull latest
    result = subprocess.run("git pull origin main", shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Repository updated")
        print(result.stdout)
    else:
        print(f"⚠️ Pull failed, trying fresh clone...")
        os.chdir("/kaggle/working")
        subprocess.run(f"rm -rf {REPO_DIR}", shell=True)
        run_command(f"git clone {REPO_URL}", "Cloning repository")
        os.chdir(REPO_DIR)
else:
    print(f"\n📂 Repository not found, cloning...")
    os.chdir("/kaggle/working")
    run_command(f"git clone {REPO_URL}", "Cloning repository")
    os.chdir(REPO_DIR)

# Verify we're in the right place
print(f"\n📍 Current directory: {os.getcwd()}")

# Show latest commit
result = subprocess.run("git log -1 --oneline", shell=True, capture_output=True, text=True)
print(f"📝 Latest commit: {result.stdout.strip()}")

# Step 2: Verify Task 12 files
print("\n" + "="*80)
print("🔍 VERIFYING TASK 12 IMPLEMENTATION")
print("="*80)

task12_files = [
    "TASK_12_COMPLETE.md",
    "TASK_12_PERFORMANCE_REPORTING_VERIFICATION.md",
    "PERFORMANCE_REPORTING_QUICK_REFERENCE.md",
    ".kiro/specs/model-training-fixes/tasks.md"
]

all_files_exist = True
for file in task12_files:
    if Path(file).exists():
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING")
        all_files_exist = False

if all_files_exist:
    print("\n✅ All Task 12 files present")
    
    # Check if reporting methods exist in train_all_models.py
    print("\n🔍 Verifying reporting methods...")
    with open('train_all_models.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    if 'def generate_summary_report' in content:
        print("  ✅ generate_summary_report() method found")
    else:
        print("  ❌ generate_summary_report() method NOT FOUND")
        
    if 'def _generate_symbol_markdown_report' in content:
        print("  ✅ _generate_symbol_markdown_report() method found")
    else:
        print("  ❌ _generate_symbol_markdown_report() method NOT FOUND")
else:
    print("\n⚠️ Some Task 12 files are missing - may be using older version")

# Step 3: Install dependencies
print("\n" + "="*80)
print("📦 INSTALLING DEPENDENCIES")
print("="*80)

packages = [
    "torch",
    "scikit-fuzzy",
    "xgboost",
    "scikit-learn",
    "imbalanced-learn",
    "pandas",
    "numpy",
    "joblib",
    "matplotlib",
    "seaborn"
]

print("\nInstalling packages (this may take a few minutes)...")
for pkg in packages:
    result = subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ {pkg}")
    else:
        print(f"  ⚠️ {pkg} - {result.stderr[:100]}")

print("\n✅ Dependencies installed")

# Step 4: Setup environment
print("\n" + "="*80)
print("🔧 ENVIRONMENT SETUP")
print("="*80)

sys.path.insert(0, REPO_DIR)
print(f"✅ Added {REPO_DIR} to Python path")

# Step 5: Run complete pipeline
print("\n" + "="*80)
print("🚀 RUNNING COMPLETE PIPELINE")
print("="*80)
print("\nThis will:")
print("  1. Run data preparation pipeline")
print("  2. Train all models for all symbols")
print("  3. Select best models per symbol")
print("  4. Generate performance reports (Task 12)")
print("  5. Save deployment manifest")
print("\n" + "="*80 + "\n")

try:
    # Execute the main pipeline
    exec(open('KAGGLE_RUN.py').read())
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    # Show generated reports
    reports_dir = Path("models/trained/reports")
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.md"))
        print(f"\n📄 Generated {len(report_files)} performance reports:")
        for report in report_files:
            print(f"  • {report.name}")
    
    # Show deployment manifest
    manifest_path = Path("models/trained/deployment_manifest.json")
    if manifest_path.exists():
        print(f"\n📋 Deployment manifest: {manifest_path}")
    
    # Show training results
    results_path = Path("models/trained/training_results.json")
    if results_path.exists():
        print(f"📊 Training results: {results_path}")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ PIPELINE EXECUTION FAILED")
    print("="*80)
    print(f"\nError: {str(e)}")
    import traceback
    print("\nTraceback:")
    print(traceback.format_exc())
    sys.exit(1)

print("\n" + "="*80)
print("🎉 ALL DONE!")
print("="*80)
print("\nNext steps:")
print("  1. Review performance reports in models/trained/reports/")
print("  2. Check deployment_manifest.json for model recommendations")
print("  3. Review training_results.json for detailed metrics")
print("\n" + "="*80)
