"""
Script to install dependencies
Run this before first execution
"""

import subprocess
import sys


def install_dependencies():
    """Install all required dependencies"""
    
    print("="*60)
    print("Installing Document AI Pipeline Dependencies")
    print("="*60)
    print()
    
    print("This will install:")
    print("  - PaddleOCR (multilingual OCR)")
    print("  - Transformers (for Qwen2.5-VL)")
    print("  - Ultralytics (YOLOv8)")
    print("  - OpenCV, Pillow (image processing)")
    print("  - RapidFuzz (fuzzy matching)")
    print("  - and more...")
    print()
    
    response = input("Continue with installation? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    print("\nInstalling dependencies...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("\n" + "="*60)
        print("Installation Complete!")
        print("="*60)
        print("\nYou can now run the pipeline using:")
        print("  python execution.py --input <input_path> --output <output_path>")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        print("Please try installing manually:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()
