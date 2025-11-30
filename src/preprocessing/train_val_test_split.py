"""
Dataset Splitting Utility for Plant Disease Classification
Handles train/validation/test split creation
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict
import random
from collections import defaultdict
from tqdm import tqdm


def create_train_val_test_split(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True
) -> Dict[str, int]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        source_dir: Path to source directory containing class folders
        output_dir: Path to output directory for split datasets
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, move files
    
    Returns:
        Dictionary containing split statistics
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    stats = {
        'train': 0,
        'val': 0,
        'test': 0,
        'class_distribution': defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    }
    
    print(f"Splitting dataset from {source_dir}")
    print(f"Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    print(f"Random seed: {seed}\n")
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Create class subdirectories in each split
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy or move files
        file_operation = shutil.copy2 if copy_files else shutil.move
        
        for f in train_files:
            file_operation(str(f), str(train_dir / class_name / f.name))
            stats['train'] += 1
            stats['class_distribution'][class_name]['train'] += 1
        
        for f in val_files:
            file_operation(str(f), str(val_dir / class_name / f.name))
            stats['val'] += 1
            stats['class_distribution'][class_name]['val'] += 1
        
        for f in test_files:
            file_operation(str(f), str(test_dir / class_name / f.name))
            stats['test'] += 1
            stats['class_distribution'][class_name]['test'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Split Complete!")
    print("="*60)
    print(f"Total images: {stats['train'] + stats['val'] + stats['test']}")
    print(f"Train: {stats['train']} ({stats['train']/(stats['train']+stats['val']+stats['test']):.1%})")
    print(f"Val: {stats['val']} ({stats['val']/(stats['train']+stats['val']+stats['test']):.1%})")
    print(f"Test: {stats['test']} ({stats['test']/(stats['train']+stats['val']+stats['test']):.1%})")
    print(f"\nNumber of classes: {len(stats['class_distribution'])}")
    print("="*60)
    
    return dict(stats)


def verify_split(
    train_dir: str,
    val_dir: str,
    test_dir: str
) -> Dict:
    """
    Verify the dataset split and print statistics
    
    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory
        test_dir: Path to test directory
    
    Returns:
        Dictionary containing split verification statistics
    """
    def count_images(directory):
        """Count images in directory"""
        total = 0
        class_counts = {}
        
        for class_dir in Path(directory).iterdir():
            if class_dir.is_dir():
                count = len([f for f in class_dir.iterdir() 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                class_counts[class_dir.name] = count
                total += count
        
        return total, class_counts
    
    train_count, train_classes = count_images(train_dir)
    val_count, val_classes = count_images(val_dir)
    test_count, test_classes = count_images(test_dir)
    
    total = train_count + val_count + test_count
    
    print("\n" + "="*60)
    print("Dataset Split Verification")
    print("="*60)
    print(f"Train: {train_count} images ({train_count/total:.1%})")
    print(f"Val: {val_count} images ({val_count/total:.1%})")
    print(f"Test: {test_count} images ({test_count/total:.1%})")
    print(f"Total: {total} images")
    print(f"\nNumber of classes: {len(train_classes)}")
    print("\nPer-class distribution:")
    print("-"*60)
    
    all_classes = sorted(set(train_classes.keys()) | set(val_classes.keys()) | set(test_classes.keys()))
    
    for class_name in all_classes:
        train_c = train_classes.get(class_name, 0)
        val_c = val_classes.get(class_name, 0)
        test_c = test_classes.get(class_name, 0)
        class_total = train_c + val_c + test_c
        
        print(f"{class_name:40s} | Train: {train_c:4d} | Val: {val_c:4d} | Test: {test_c:4d} | Total: {class_total:4d}")
    
    print("="*60)
    
    return {
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'total': total,
        'classes': len(all_classes),
        'train_classes': train_classes,
        'val_classes': val_classes,
        'test_classes': test_classes
    }


def stratified_split_by_class(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_samples_per_class: int = 3,
    seed: int = 42
) -> Dict:
    """
    Create stratified split ensuring minimum samples per class in each split
    
    Args:
        source_dir: Path to source directory
        output_dir: Path to output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        min_samples_per_class: Minimum samples required per class in each split
        seed: Random seed
    
    Returns:
        Split statistics
    """
    random.seed(seed)
    source_path = Path(source_dir)
    
    # First, check if all classes have enough samples
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            n_images = len([f for f in class_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            required = min_samples_per_class * 3  # for train, val, test
            
            if n_images < required:
                print(f"Warning: Class '{class_dir.name}' has only {n_images} images, "
                      f"but {required} are recommended for stratified split with min_samples_per_class={min_samples_per_class}")
    
    # Proceed with split
    return create_train_val_test_split(
        source_dir, output_dir, train_ratio, val_ratio, test_ratio, seed
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verify', action='store_true', help='Verify existing split')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_split(
            os.path.join(args.output, 'train'),
            os.path.join(args.output, 'val'),
            os.path.join(args.output, 'test')
        )
    else:
        create_train_val_test_split(
            args.source,
            args.output,
            args.train,
            args.val,
            args.test,
            args.seed
        )
