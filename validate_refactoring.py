"""Quick validation script to verify rembg integration."""

import sys
from pathlib import Path

def check_imports():
    """Verify all required imports are available."""
    print("=" * 60)
    print("IMPORT CHECK")
    print("=" * 60)
    
    checks = [
        ("PIL.Image", "Pillow"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("rembg", "rembg"),
        ("onnxruntime", "ONNX Runtime"),
    ]
    
    all_ok = True
    for module_name, display_name in checks:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} (run: pip install {display_name.lower().replace(' ', '-')})")
            all_ok = False
    
    return all_ok

def check_code_changes():
    """Verify key code changes were made."""
    print("\n" + "=" * 60)
    print("CODE CHANGES CHECK")
    print("=" * 60)
    
    image_pipeline_path = Path(__file__).parent / "image_pipeline.py"
    
    if not image_pipeline_path.exists():
        print("❌ image_pipeline.py not found!")
        return False
    
    code = image_pipeline_path.read_text()
    
    checks = [
        ("from rembg import remove", "✅ rembg import added"),
        ("def remove_background(" , "✅ remove_background function exists"),
        ("remove(img_rgb)", "✅ rembg.remove() call found"),
        ("requests", "❌ 'requests' library still in code (should be removed)" if "import requests" in code else "✅ 'requests' removed"),
        ("REMOVE_BG_API_KEY", "❌ 'REMOVE_BG_API_KEY' still in code" if "REMOVE_BG_API_KEY" in code else "✅ 'REMOVE_BG_API_KEY' removed"),
    ]
    
    all_ok = True
    for check, result_msg in checks:
        if "❌" not in result_msg:
            if check in code:
                print(result_msg)
            else:
                print(f"⚠️  {result_msg} - NOT FOUND")
                all_ok = False
        else:
            if check not in code:
                print(f"✅ {result_msg}")
            else:
                print(result_msg)
                all_ok = False
    
    return all_ok

def check_files_exist():
    """Check all necessary files exist."""
    print("\n" + "=" * 60)
    print("FILE CHECK")
    print("=" * 60)
    
    base = Path(__file__).parent
    files = [
        ("image_pipeline.py", "Original pipeline (refactored)"),
        ("image_pipeline_optimized.py", "Batch-processing pipeline"),
        ("requirements.txt", "Dependencies list"),
        ("REFACTORING_REMBG.md", "Installation guide"),
    ]
    
    all_ok = True
    for filename, description in files:
        path = base / filename
        if path.exists():
            print(f"✅ {filename}: {description}")
        else:
            print(f"❌ {filename}: NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks."""
    print("\nREMBG REFACTORING VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Code Changes", check_code_changes),
        ("Files", check_files_exist),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n📋 NEXT STEPS:")
        print("  1. pip install rembg[gpu] onnxruntime")
        print("  2. python image_pipeline.py  (or configure INPUT_IMAGE_PATH first)")
        print("  3. For batch processing: python image_pipeline_optimized.py <input_folder> <output_folder>")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - Fix the issues above before running the pipeline")
        return 1

if __name__ == "__main__":
    sys.exit(main())
