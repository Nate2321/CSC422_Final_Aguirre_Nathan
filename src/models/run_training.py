"""
Main Training Runner for Plant Disease Classification
Orchestrates the complete training process with configuration management
"""

import torch
import torch.nn as nn
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_model import create_resnet50
from models.train import Trainer, create_optimizer, create_scheduler, save_model, count_parameters
from preprocessing.data_loader import create_data_loaders
from preprocessing.augmentation import get_training_transforms, get_validation_transforms


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path: str = None, **kwargs):
    """
    Main training function
    
    Args:
        config_path: Path to configuration YAML file
        **kwargs: Override config parameters
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Default configuration
        config = {
            'model': {
                'num_classes': 38,
                'pretrained': True,
                'freeze_backbone': True,
                'dropout_rate': 0.5
            },
            'data': {
                'train_dir': 'data/processed/train',
                'val_dir': 'data/processed/val',
                'test_dir': 'data/processed/test',
                'image_size': 224,
                'batch_size': 32,
                'num_workers': 4,
                'pin_memory': True,
                'use_advanced_augmentation': True
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'optimizer': 'adam',
                'scheduler': 'reduce_on_plateau',
                'early_stopping_patience': 10,
                'save_frequency': 5
            },
            'paths': {
                'save_dir': 'results/models',
                'log_dir': 'results/logs'
            }
        }
    
    # Override with kwargs
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'model.name'
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    # Convert relative paths to absolute from project root
    project_root = Path(__file__).parent.parent.parent
    for path_key in ['train_dir', 'val_dir', 'test_dir']:
        if path_key in config['data']:
            path = Path(config['data'][path_key])
            if not path.is_absolute():
                config['data'][path_key] = str(project_root / config['data'][path_key])
            print(f"DEBUG: {path_key} = {config['data'][path_key]}")
    for path_key in ['save_dir', 'log_dir']:
        if path_key in config['paths']:
            path = Path(config['paths'][path_key])
            if not path.is_absolute():
                config['paths'][path_key] = str(project_root / config['paths'][path_key])
    
    print("="*70)
    print("="*70)
    print("PLANT DISEASE CLASSIFICATION - TRAINING (ResNet50)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: ResNet50")
    print(f"  Classes: {config['model']['num_classes']}")
    print(f"  Pretrained: {config['model']['pretrained']}")
    print(f"  Freeze backbone: {config['model']['freeze_backbone']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print("="*70)
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create data transforms
    print("\nCreating data transforms...")
    train_transform = get_training_transforms(
        image_size=config['data']['image_size'],
        advanced=config['data']['use_advanced_augmentation']
    )
    val_transform = get_validation_transforms(
        image_size=config['data']['image_size']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        test_dir=config['data']['test_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    # Create ResNet50 model
    print(f"\nCreating ResNet50 model...")
    model = create_resnet50(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout_rate=config['model'].get('dropout_rate', 0.5)
    )
    # Print model summary if available
    if hasattr(model, 'print_model_summary'):
        model.print_model_summary()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    print(f"\nCreating optimizer: {config['training']['optimizer']}...")
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler_name = config['training'].get('scheduler', 'reduce_on_plateau')
    print(f"Creating scheduler: {scheduler_name}...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=config['paths']['save_dir'],
        log_dir=config['paths']['log_dir'],
        use_tensorboard=True
    )
    
    # Resume from checkpoint if specified
    resume_checkpoint = kwargs.get('resume_checkpoint')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.resume_from_checkpoint(resume_checkpoint)
    
    # Save configuration
    Path(config['paths']['save_dir']).mkdir(parents=True, exist_ok=True)
    config_save_path = os.path.join(config['paths']['save_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")
    
    # Start training
    print("\n")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        save_frequency=config['training']['save_frequency']
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {config['paths']['save_dir']}")
    print(f"Logs saved to: {config['paths']['log_dir']}")
    
    return trainer


def main():
    """Command-line interface for training"""
    parser = argparse.ArgumentParser(
        description='Train Plant Disease Classification Model'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., results/models/checkpoint_epoch_5.pth)'
    )
    
    # Model arguments
    parser.add_argument(
        '--num-classes',
        type=int,
        default=38,
        help='Number of disease classes'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        default=True,
        help='Freeze backbone layers'
    )
    
    # Data arguments
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/processed/train',
        help='Training data directory'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='data/processed/val',
        help='Validation data directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'adamw'],
        help='Optimizer'
    )
    
    args = parser.parse_args()
    
    # Override config with command-line arguments
    kwargs = {
        'model.num_classes': args.num_classes,
        'data.train_dir': args.train_dir,
        'data.val_dir': args.val_dir,
        'data.batch_size': args.batch_size,
        'training.num_epochs': args.epochs,
        'training.learning_rate': args.lr,
        'training.optimizer': args.optimizer,
    }
    
    # Add resume checkpoint if specified
    if args.resume:
        kwargs['resume_checkpoint'] = args.resume
    
    train_model(config_path=args.config, **kwargs)


if __name__ == '__main__':
    main()
