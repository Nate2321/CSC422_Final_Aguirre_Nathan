"""
ResNet50 Model for Plant Disease Classification
Implements transfer learning with ResNet50 pretrained on ImageNet
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet50PlantDisease(nn.Module):
    """ResNet50 model with transfer learning for plant disease classification"""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5
    ):
        """
        Initialize ResNet50 model
        
        Args:
            num_classes: Number of disease classes to predict
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers (only train classifier)
            dropout_rate: Dropout probability for regularization
        """
        super(ResNet50PlantDisease, self).__init__()
        
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get number of features from last layer
        num_features = self.model.fc.in_features
        
        # Replace final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def unfreeze_backbone(self, unfreeze_from_layer: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            unfreeze_from_layer: Layer index from which to unfreeze (None = all)
        """
        if unfreeze_from_layer is None:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze from specific layer onwards
            layers = list(self.model.children())
            for layer in layers[unfreeze_from_layer:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.freeze_backbone = False
        print(f"Backbone unfrozen from layer {unfreeze_from_layer if unfreeze_from_layer else 0}")
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_model_summary(self):
        """Print model architecture summary"""
        trainable, total = self.get_trainable_params()
        print("="*70)
        print("ResNet50 Plant Disease Classification Model")
        print("="*70)
        print(f"Number of classes: {self.num_classes}")
        print(f"Backbone frozen: {self.freeze_backbone}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {total - trainable:,}")
        print(f"Trainable %: {100 * trainable / total:.2f}%")
        print("="*70)


def create_resnet50(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5
) -> ResNet50PlantDisease:
    """
    Factory function to create ResNet50 model
    
    Args:
        num_classes: Number of disease classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone for transfer learning
        dropout_rate: Dropout rate for regularization
    
    Returns:
        ResNet50PlantDisease model
    """
    model = ResNet50PlantDisease(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating ResNet50 model for 38 plant disease classes...\n")
    
    # Create model
    model = create_resnet50(
        num_classes=38,
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=0.5
    )
    
    # Print summary
    model.print_model_summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test unfreezing
    print("\nUnfreezing backbone...")
    model.unfreeze_backbone()
    trainable, total = model.get_trainable_params()
    print(f"Trainable parameters after unfreezing: {trainable:,} ({100 * trainable / total:.2f}%)")
