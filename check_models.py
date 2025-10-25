"""
Check what model and data files are available
"""

from pathlib import Path
import os

print("=" * 80)
print("CHECKING FILES")
print("=" * 80)

# Check possible locations
locations = [
    '/kaggle/working/Model-output',
    '/kaggle/working/data/processed',
    '/kaggle/working/data',
    '/kaggle/input',
    'data/processed',
]

for loc in locations:
    path = Path(loc)
    print(f"\n📂 Checking: {loc}")
    if path.exists():
        print(f"  ✅ EXISTS")
        # List contents
        try:
            files = list(path.glob('*'))
            if files:
                print(f"  Files found: {len(files)}")
                for f in sorted(files)[:15]:  # Show first 15
                    print(f"    - {f.name}")
            else:
                print(f"  (empty directory)")
        except Exception as e:
            print(f"  Error listing: {e}")
    else:
        print(f"  ❌ NOT FOUND")

# Check /kaggle/input subdirectories
print(f"\n📂 Checking /kaggle/input subdirectories:")
input_path = Path('/kaggle/input')
if input_path.exists():
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            print(f"  📁 {subdir.name}:")
            try:
                files = list(subdir.glob('*.csv'))[:5]
                for f in files:
                    print(f"      - {f.name}")
            except:
                pass
