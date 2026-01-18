"""
Quick test script to verify pipeline setup
"""

import sys
from path lib import Path

print("="*60)
print("Document AI Pipeline - Setup Verification")
print("="*60)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check if all modules can be imported
print("\nChecking core modules...")

modules_to_check = [
    ('utils.preprocessing', 'Preprocessing'),
    ('utils.ocr_module', 'OCR Module'),
    ('utils.vlm_module', 'VLM Module'),
    ('utils.detection_module', 'Detection Module'),
    ('utils.field_extractors', 'Field Extractors'),
    ('utils.validation', 'Validation'),
    ('utils.output_formatter', 'Output Formatter'),
    ('utils.error_analyzer', 'Error Analyzer'),
]

sys.path.append(str(Path(__file__).parent))

all_good = True
for module_name, display_name in modules_to_check:
    try:
        __import__(module_name)
        print(f"  ✓ {display_name}")
    except ImportError as e:
        print(f"  ✗ {display_name}: {e}")
        all_good = False

print("\nChecking dependencies...")

dependencies = [
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('rapidfuzz', 'RapidFuzz'),
]

for module_name, display_name in dependencies:
    try:
        __import__(module_name)
        print(f"  ✓ {display_name}")
    except ImportError:
        print(f"  ✗ {display_name} (not installed - run: pip install -r requirements.txt)")
        all_good = False

# Check optional heavy dependencies
print("\nChecking ML dependencies (these are large downloads)...")

ml_dependencies = [
    ('paddleocr', 'PaddleOCR'),
    ('transformers', 'Transformers'),
    ('torch', 'PyTorch'),
    ('ultralytics', 'Ultralytics (YOLO)'),
]

for module_name, display_name in ml_dependencies:
    try:
        __import__(module_name)
        print(f"  ✓ {display_name}")
    except ImportError:
        print(f"  ⚠ {display_name} (will be installed on first run)")

# Check file structure
print("\nChecking file structure...")

required_files = [
    'execution.py',
    'config.json',
    'requirements.txt',
    'README.md',
    'data/dealer_master.txt',
    'data/asset_master.txt',
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} (missing)")
        all_good = False

print("\n" + "="*60)
if all_good:
    print("✓ All checks passed! Pipeline is ready.")
    print("\nNext steps:")
    print("1. Install dependencies: python install_dependencies.py")
    print("2. Add dataset to a folder (e.g., 'dataset/')")
    print("3. Update master lists in data/")
    print("4. Run pipeline: python execution.py --input dataset/ --output results/")
else:
    print("⚠ Some checks failed. Please address the issues above.")
    print("\nTo install dependencies, run:")
    print("  python install_dependencies.py")
    print("or:")
    print("  pip install -r requirements.txt")

print("="*60)
