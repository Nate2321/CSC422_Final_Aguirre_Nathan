"""
Data Loader for Plant Disease Classification
Handles loading and organizing the PlantVillage dataset
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class PlantDiseaseDataset(Dataset):
    """Custom Dataset for Plant Disease Images"""
    
    def __init__(self, root_dir: str, transform=None, split='train'):
        """
        Args:
            root_dir: Directory with all the images organized by class
            transform: Optional transform to be applied on images
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths and create class mappings"""
        classes = sorted([d for d in os.listdir(self.root_dir) 
                         if os.path.isdir(os.path.join(self.root_dir, d))])
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        for class_name in classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_dir / img_name
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset"""
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        train_transform: Transforms for training data
        val_transform: Transforms for validation/test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform, split='train')
    val_dataset = PlantDiseaseDataset(val_dir, transform=val_transform, split='val')
    test_dataset = PlantDiseaseDataset(test_dir, transform=val_transform, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(data_dir: str) -> Dict:
    """
    Calculate dataset statistics (mean, std, class distribution)
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Dictionary containing dataset statistics
    """
    dataset = PlantDiseaseDataset(data_dir, transform=transforms.ToTensor())
    
    # Calculate mean and std
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    print("Calculating dataset statistics...")
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    # Get class distribution
    class_distribution = dataset.get_class_distribution()
    
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'num_classes': len(dataset.class_to_idx),
        'num_images': len(dataset),
        'class_to_idx': dataset.class_to_idx,
        'class_distribution': class_distribution
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data/raw/PlantVillage"
    
    if os.path.exists(data_dir):
        stats = get_dataset_statistics(data_dir)
        print(f"\nDataset Statistics:")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"Total images: {stats['num_images']}")
        print(f"Mean: {stats['mean']}")
        print(f"Std: {stats['std']}")
        print(f"\nClass distribution:")
        for class_name, count in sorted(stats['class_distribution'].items()):
            print(f"  {class_name}: {count}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please download the PlantVillage dataset first.")
