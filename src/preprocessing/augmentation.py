"""
Data Augmentation Pipeline for Plant Disease Classification
Implements training and validation transforms with various augmentation techniques
"""

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple


class AlbumentationsTransform:
    """Wrapper for Albumentations transforms to work with torchvision"""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        # Convert PIL to numpy array
        img = np.array(img)
        # Apply albumentations transform
        augmented = self.transform(image=img)
        return augmented['image']


def get_training_transforms(image_size: int = 224, advanced: bool = True) -> transforms.Compose:
    """
    Get training data augmentation transforms
    
    Args:
        image_size: Target image size (default: 224 for most CNNs)
        advanced: Whether to use advanced augmentations (albumentations)
    
    Returns:
        Composed transforms for training
    """
    if advanced:
        # Advanced augmentations using Albumentations
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.CoarseDropout(
                max_holes=8, 
                max_height=image_size//8, 
                max_width=image_size//8, 
                fill_value=0, 
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return AlbumentationsTransform(train_transform)
    
    else:
        # Basic augmentations using torchvision
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_validation_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation, only normalization)
    
    Args:
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_time_augmentation_transforms(image_size: int = 224, n_augments: int = 5):
    """
    Get test-time augmentation (TTA) transforms for ensemble predictions
    
    Args:
        image_size: Target image size
        n_augments: Number of augmented versions to create
    
    Returns:
        List of transform compositions for TTA
    """
    tta_transforms = []
    
    # Original (no augmentation)
    tta_transforms.append(get_validation_transforms(image_size))
    
    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Vertical flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Rotation 90
    tta_transforms.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Rotation 270
    tta_transforms.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation((270, 270)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    return tta_transforms[:n_augments]


def get_custom_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get custom transforms with user-specified normalization parameters
    
    Args:
        image_size: Target image size
        mean: Custom mean values for normalization
        std: Custom std values for normalization
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


if __name__ == "__main__":
    # Example usage
    print("Training Transforms (Basic):")
    train_transform_basic = get_training_transforms(image_size=224, advanced=False)
    print(train_transform_basic)
    
    print("\n" + "="*50 + "\n")
    
    print("Training Transforms (Advanced):")
    train_transform_advanced = get_training_transforms(image_size=224, advanced=True)
    print("Using Albumentations library for advanced augmentations")
    
    print("\n" + "="*50 + "\n")
    
    print("Validation Transforms:")
    val_transform = get_validation_transforms(image_size=224)
    print(val_transform)
    
    print("\n" + "="*50 + "\n")
    
    print("Test-Time Augmentation Transforms:")
    tta_transforms = get_test_time_augmentation_transforms(image_size=224, n_augments=5)
    print(f"Number of TTA transforms: {len(tta_transforms)}")
