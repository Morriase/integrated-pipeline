#!/usr/bin/env python3
"""
Kaggle Upload Preparation Script
Prepares all necessary files for uploading to Kaggle
"""

import os
import zipfile
import shutil
from pathlib import Path


def create_kaggle_upload_package():
    """Create a zip package with all files needed for Kaggle training"""

    # Files to include
    required_files = [
        'integrated_advanced_pipeline.py',
        'production_ensemble_pipeline.py',
        'advanced_temporal_architecture.py',
        'enhanced_multitf_pipeline.py',
        'feature_engineering_smc_institutional.py',
        'model_export.py',
        'learning_curve_plotter.py',
        'temporal_validation.py',
        'recovery_mechanism.py',
        'model_rest_server_proper.py',
        'training_data.csv',
        'kaggle_training_notebook.ipynb'
    ]

    # Check if all files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all files are present before uploading to Kaggle.")
        return False

    # Create zip file
    zip_filename = 'kaggle_upload_package.zip'
    print(f"📦 Creating {zip_filename}...")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in required_files:
            print(f"  Adding: {file}")
            zipf.write(file, file)

    # Get file size
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"📦 Package size: {size_mb:.2f} MB")
    print("\n✅ Package created successfully!")
    print("\n📋 Upload Instructions:")
    print("1. Go to https://www.kaggle.com/notebooks")
    print("2. Create a new notebook")
    print("3. Upload the zip file to your notebook environment")
    print("4. Extract the zip file")
    print("5. Run the notebook cells in order")
    print("\n⏱️ Expected training time: 30-60 minutes on Kaggle GPU")
    print("\n📥 After training completes, download the 'trained_models.zip' file")
    print("   and extract it to your local Model_output folder")
    return True


def verify_package_contents():
    """Verify the contents of the created package"""
    zip_filename = 'kaggle_upload_package.zip'

    if not os.path.exists(zip_filename):
        print("❌ Package not found. Run create_kaggle_upload_package() first.")
        return

    print(f"📋 Contents of {zip_filename}:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file in sorted(zipf.namelist()):
            print(f"  - {file}")


if __name__ == "__main__":
    print("🚀 Kaggle Upload Preparation")
    print("=" * 50)

    success = create_kaggle_upload_package()

    if success:
        verify_package_contents()

        print("\n" + "=" * 50)
        print("🎯 Next Steps:")
        print("1. Upload 'kaggle_upload_package.zip' to Kaggle")
        print("2. Extract and run 'kaggle_training_notebook.ipynb'")
        print("3. Download trained models when complete")
        print("=" * 50)
