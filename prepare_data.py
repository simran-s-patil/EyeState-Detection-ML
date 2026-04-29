"""
Data Preparation Script
=======================

This script helps prepare the MRL Eye Dataset for training.
It can:
1. Check dataset structure
2. Validate images
3. Split data into train/validation sets
4. Generate dataset statistics
"""

import os
import shutil
from pathlib import Path
import argparse
from PIL import Image
import numpy as np


def check_dataset_structure(data_dir):
    """
    Check if the dataset has the correct structure.

    Args:
        data_dir (str): Path to dataset directory

    Returns:
        dict: Dataset statistics
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    # Check for required subdirectories
    open_dir = data_path / "open"
    closed_dir = data_path / "closed"

    if not open_dir.exists():
        raise ValueError(f"'open' directory not found in {data_dir}")

    if not closed_dir.exists():
        raise ValueError(f"'closed' directory not found in {data_dir}")

    # Count images in each class
    open_images = list(open_dir.glob("*.jpg")) + list(open_dir.glob("*.png")) + list(open_dir.glob("*.jpeg"))
    closed_images = list(closed_dir.glob("*.jpg")) + list(closed_dir.glob("*.png")) + list(closed_dir.glob("*.jpeg"))

    stats = {
        'total_open': len(open_images),
        'total_closed': len(closed_images),
        'total_images': len(open_images) + len(closed_images),
        'open_dir': str(open_dir),
        'closed_dir': str(closed_dir)
    }

    return stats


def validate_images(data_dir):
    """
    Validate all images in the dataset.

    Args:
        data_dir (str): Path to dataset directory

    Returns:
        dict: Validation results
    """
    data_path = Path(data_dir)
    open_dir = data_path / "open"
    closed_dir = data_path / "closed"

    invalid_images = []
    valid_count = 0

    # Check open eyes images
    for img_path in open_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path)
            img.verify()
            valid_count += 1
        except Exception as e:
            invalid_images.append((str(img_path), str(e)))

    for img_path in open_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            img.verify()
            valid_count += 1
        except Exception as e:
            invalid_images.append((str(img_path), str(e)))

    # Check closed eyes images
    for img_path in closed_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path)
            img.verify()
            valid_count += 1
        except Exception as e:
            invalid_images.append((str(img_path), str(e)))

    for img_path in closed_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            img.verify()
            valid_count += 1
        except Exception as e:
            invalid_images.append((str(img_path), str(e)))

    return {
        'valid_images': valid_count,
        'invalid_images': invalid_images
    }


def split_dataset(data_dir, train_ratio=0.8, output_dir=None):
    """
    Split dataset into train and validation sets.

    Args:
        data_dir (str): Path to dataset directory
        train_ratio (float): Ratio of data for training
        output_dir (str): Output directory for split data
    """
    if output_dir is None:
        output_dir = str(Path(data_dir).parent / "dataset_split")

    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "validation"

    # Create directories
    (train_dir / "open").mkdir(parents=True, exist_ok=True)
    (train_dir / "closed").mkdir(parents=True, exist_ok=True)
    (val_dir / "open").mkdir(parents=True, exist_ok=True)
    (val_dir / "closed").mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)

    # Split open eyes images
    open_images = list((data_path / "open").glob("*.jpg")) + \
                  list((data_path / "open").glob("*.png")) + \
                  list((data_path / "open").glob("*.jpeg"))

    np.random.shuffle(open_images)
    split_idx = int(len(open_images) * train_ratio)

    train_open = open_images[:split_idx]
    val_open = open_images[split_idx:]

    # Copy open eyes images
    for img in train_open:
        shutil.copy2(img, train_dir / "open" / img.name)

    for img in val_open:
        shutil.copy2(img, val_dir / "open" / img.name)

    # Split closed eyes images
    closed_images = list((data_path / "closed").glob("*.jpg")) + \
                    list((data_path / "closed").glob("*.png")) + \
                    list((data_path / "closed").glob("*.jpeg"))

    np.random.shuffle(closed_images)
    split_idx = int(len(closed_images) * train_ratio)

    train_closed = closed_images[:split_idx]
    val_closed = closed_images[split_idx:]

    # Copy closed eyes images
    for img in train_closed:
        shutil.copy2(img, train_dir / "closed" / img.name)

    for img in val_closed:
        shutil.copy2(img, val_dir / "closed" / img.name)

    print(f"Dataset split completed:")
    print(f"Training set: {len(train_open)} open, {len(train_closed)} closed")
    print(f"Validation set: {len(val_open)} open, {len(val_closed)} closed")
    print(f"Output directory: {output_dir}")


def print_dataset_stats(stats):
    """
    Print dataset statistics.

    Args:
        stats (dict): Dataset statistics
    """
    print("Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Open eyes: {stats['total_open']}")
    print(f"Closed eyes: {stats['total_closed']}")
    print(".1f")
    print(f"Open directory: {stats['open_dir']}")
    print(f"Closed directory: {stats['closed_dir']}")


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='MRL Eye Dataset Preparation')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--check', action='store_true',
                       help='Check dataset structure and statistics')
    parser.add_argument('--validate', action='store_true',
                       help='Validate all images in dataset')
    parser.add_argument('--split', action='store_true',
                       help='Split dataset into train/validation sets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for split data')

    args = parser.parse_args()

    try:
        if args.check:
            # Check dataset structure
            stats = check_dataset_structure(args.data_dir)
            print_dataset_stats(stats)

        elif args.validate:
            # Validate images
            print("Validating images...")
            validation_results = validate_images(args.data_dir)
            print(f"Valid images: {validation_results['valid_images']}")

            if validation_results['invalid_images']:
                print("Invalid images:")
                for img_path, error in validation_results['invalid_images']:
                    print(f"  {img_path}: {error}")
            else:
                print("All images are valid!")

        elif args.split:
            # Split dataset
            print("Splitting dataset...")
            split_dataset(args.data_dir, args.train_ratio, args.output_dir)

        else:
            print("Please specify an action: --check, --validate, or --split")
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())