"""
Check what model files are available
"""

from pathlib import Path
import os

print("=" * 80)
print("CHECKING MODEL FILES")
print("=" * 80)

# Check possible locations
locations = [
    '/kaggle/working/Model-output',
    '/kaggle/working/models/trained',
    'models/trained',
    '/kaggle/input'
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
                for f in sorted(files)[:10]:  # Show first 10
                    print(f"    - {f.name}")
            else:
                print(f"  (empty directory)")
        except Exception as e:
            print(f"  Error listing: {e}")
    else:
        print(f"  ❌ NOT FOUND")

# Check current directory
print(f"\n📂 Current directory: {os.getcwd()}")
print(f"  Contents:")
for item in sorted(os.listdir('.')):
    print(f"    - {item}")
