"""
Dataset Verification Script
===========================

This script verifies that the dataset is properly structured
and ready for training.
"""

import os
from pathlib import Path


def verify_dataset():
    """
    Verify the dataset structure and provide statistics.
    """
    print("Eye Classification Dataset Verification")
    print("=" * 40)

    # Check archive folder
    archive_path = Path("archive")
    if not archive_path.exists():
        print("❌ Archive folder not found!")
        return False

    dataset_path = archive_path / "dataset_new"
    if not dataset_path.exists():
        print("❌ dataset_new folder not found in archive!")
        return False

    train_path = dataset_path / "train"
    if not train_path.exists():
        print("❌ train folder not found!")
        return False

    # Check class folders
    classes = ['Closed', 'Open', 'no_yawn', 'yawn']
    class_counts = {}

    print("\n📁 Dataset Structure:")
    print(f"Path: {dataset_path.absolute()}")

    for class_name in classes:
        class_path = train_path / class_name
        if class_path.exists():
            # Count image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            count = 0
            for ext in image_extensions:
                count += len(list(class_path.glob(ext)))

            class_counts[class_name] = count
            print(f"  ✅ {class_name}: {count} images")
        else:
            print(f"  ❌ {class_name}: folder not found")
            class_counts[class_name] = 0

    # Summary
    total_images = sum(class_counts.values())
    print(f"\n📊 Total Images: {total_images}")

    # Binary classification mapping
    open_count = class_counts.get('Open', 0)
    closed_count = class_counts.get('Closed', 0)
    binary_total = open_count + closed_count

    print("\n🎯 Binary Classification (Open vs Closed):")
    print(f"  Eyes Open (label=1): {open_count} images")
    print(f"  Eyes Closed (label=0): {closed_count} images")
    print(f"  Total for training: {binary_total} images")

    if binary_total == 0:
        print("❌ No images found for binary classification!")
        return False

    # Check test set
    test_path = dataset_path / "test"
    if test_path.exists():
        test_classes = ['Closed', 'Open', 'no_yawn', 'yawn']
        test_counts = {}

        print("\n🧪 Test Set:")
        for class_name in test_classes:
            class_path = test_path / class_name
            if class_path.exists():
                count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    count += len(list(class_path.glob(ext)))
                test_counts[class_name] = count
                print(f"  {class_name}: {count} images")

        test_total = sum(test_counts.values())
        print(f"  Total test images: {test_total}")
    else:
        print("\n⚠️  Test set not found (optional)")

    print("\n✅ Dataset verification completed!")
    print("Ready to train the model once TensorFlow is installed.")

    return True


def check_dependencies():
    """
    Check if required dependencies are installed.
    """
    print("\n🔍 Checking Dependencies:")

    dependencies = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow')
    ]

    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(name)

    # Special check for TensorFlow
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__}")
        tf_available = True
    except ImportError:
        print("  ❌ TensorFlow (required for training)")
        tf_available = False
        missing.append('TensorFlow')

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies available!")
    return True


if __name__ == "__main__":
    print("Eye Classification Project - Dataset Verification")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Verify dataset
    dataset_ok = verify_dataset()

    # Check dependencies
    deps_ok = check_dependencies()

    print("\n" + "=" * 50)
    if dataset_ok and deps_ok:
        print("🎉 Everything is ready! Run 'python eye_classifier.py' to start training.")
    elif dataset_ok:
        print("📁 Dataset is ready, but dependencies need to be installed.")
        print("Please install TensorFlow-compatible Python version and dependencies.")
    else:
        print("❌ Issues found. Please check dataset structure and dependencies.")