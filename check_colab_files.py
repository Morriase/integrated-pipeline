#!/usr/bin/env python3
"""Check what files are available in the Colab environment"""
from pathlib import Path
import os

print("="*70)
print("CHECKING COLAB FILE STRUCTURE")
print("="*70)

# Get current directory
cwd = Path.cwd()
print(f"\nCurrent working directory: {cwd}")

# Check for Data directory
data_dir = cwd / 'Data'
print(f"\nData directory exists: {data_dir.exists()}")

if data_dir.exists():
    print(f"\nFiles in Data directory:")
    for file in sorted(data_dir.glob('*.csv')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name} ({size_mb:.2f} MB)")
else:
    print(f"\n❌ Data directory not found at: {data_dir}")

    # Check parent directories
    print(f"\nSearching for Data directory...")
    for parent in [cwd, cwd.parent, cwd.parent.parent]:
        test_data = parent / 'Data'
        if test_data.exists():
            print(f"  ✓ Found at: {test_data}")
            print(f"    CSV files:")
            for file in sorted(test_data.glob('*.csv')):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"      • {file.name} ({size_mb:.2f} MB)")
            break

# Check Python directory
python_dir = cwd / 'Python'
print(f"\nPython directory exists: {python_dir.exists()}")

print("\n" + "="*70)
