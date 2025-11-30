"""
Main Preprocessing Pipeline for Plant Disease Classification
Orchestrates the complete preprocessing workflow
"""

import os
import json
from pathlib import Path
import argparse
from typing import Dict, Optional
import yaml

from data_loader import get_dataset_statistics, create_data_loaders
from augmentation import get_training_transforms, get_validation_transforms, get_custom_transforms
from train_val_test_split import create_train_val_test_split, verify_split


class PreprocessingPipeline:
    """Main preprocessing pipeline for plant disease dataset"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'data': {
                    'raw_dir': '../../data/raw/PlantVillage',
                    'processed_dir': '../../data/processed',
                    'image_size': 224,
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'seed': 42
                },
                'augmentation': {
                    'use_advanced': True,
                    'use_custom_normalization': False
                },
                'dataloader': {
                    'batch_size': 32,
                    'num_workers': 4,
                    'pin_memory': True
                }
            }
    
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("="*70)
        print("PLANT DISEASE CLASSIFICATION - PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Verify data directory exists
        raw_dir = self.config['data']['raw_dir']
        if not os.path.exists(raw_dir):
            print(f"\n❌ Error: Data directory not found: {raw_dir}")
            print("Please download the PlantVillage dataset first.")
            return
        
        # Step 2: Calculate dataset statistics
        print("\n[1/5] Calculating dataset statistics...")
        stats = self.calculate_statistics(raw_dir)
        
        # Step 3: Create train/val/test split
        print("\n[2/5] Creating train/validation/test split...")
        split_stats = self.create_split()
        
        # Step 4: Verify split
        print("\n[3/5] Verifying split...")
        self.verify_data_split()
        
        # Step 5: Set up data augmentation
        print("\n[4/5] Setting up data augmentation...")
        self.setup_augmentation()
        
        # Step 6: Create sample data loaders
        print("\n[5/5] Creating data loaders...")
        self.create_sample_loaders()
        
        # Save configuration and statistics
        self.save_preprocessing_info(stats, split_stats)
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
    
    def calculate_statistics(self, data_dir: str) -> Dict:
        """Calculate and save dataset statistics"""
        stats = get_dataset_statistics(data_dir)
        
        print(f"\nDataset Statistics:")
        print(f"  - Number of classes: {stats['num_classes']}")
        print(f"  - Total images: {stats['num_images']}")
        print(f"  - Channel means: {[f'{x:.4f}' for x in stats['mean']]}")
        print(f"  - Channel stds: {[f'{x:.4f}' for x in stats['std']]}")
        
        return stats
    
    def create_split(self) -> Dict:
        """Create train/validation/test split"""
        config = self.config['data']
        
        split_stats = create_train_val_test_split(
            source_dir=config['raw_dir'],
            output_dir=config['processed_dir'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            seed=config['seed'],
            copy_files=True  # Copy instead of move to preserve original
        )
        
        return split_stats
    
    def verify_data_split(self):
        """Verify the created split"""
        processed_dir = self.config['data']['processed_dir']
        
        verify_split(
            train_dir=os.path.join(processed_dir, 'train'),
            val_dir=os.path.join(processed_dir, 'val'),
            test_dir=os.path.join(processed_dir, 'test')
        )
    
    def setup_augmentation(self):
        """Set up and demonstrate data augmentation"""
        image_size = self.config['data']['image_size']
        use_advanced = self.config['augmentation']['use_advanced']
        
        train_transform = get_training_transforms(
            image_size=image_size,
            advanced=use_advanced
        )
        val_transform = get_validation_transforms(image_size=image_size)
        
        print(f"\nAugmentation Setup:")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Advanced augmentations: {use_advanced}")
        print(f"  - Training transforms: {'Albumentations' if use_advanced else 'torchvision'}")
        print(f"  - Validation transforms: torchvision (no augmentation)")
    
    def create_sample_loaders(self):
        """Create sample data loaders to verify setup"""
        processed_dir = self.config['data']['processed_dir']
        image_size = self.config['data']['image_size']
        batch_size = self.config['dataloader']['batch_size']
        
        train_transform = get_training_transforms(
            image_size=image_size,
            advanced=self.config['augmentation']['use_advanced']
        )
        val_transform = get_validation_transforms(image_size=image_size)
        
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                train_dir=os.path.join(processed_dir, 'train'),
                val_dir=os.path.join(processed_dir, 'val'),
                test_dir=os.path.join(processed_dir, 'test'),
                train_transform=train_transform,
                val_transform=val_transform,
                batch_size=batch_size,
                num_workers=self.config['dataloader']['num_workers'],
                pin_memory=self.config['dataloader']['pin_memory']
            )
            
            print(f"\nData Loaders Created:")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Train batches: {len(train_loader)}")
            print(f"  - Val batches: {len(val_loader)}")
            print(f"  - Test batches: {len(test_loader)}")
            
        except Exception as e:
            print(f"⚠️  Could not create data loaders: {e}")
    
    def save_preprocessing_info(self, stats: Dict, split_stats: Dict):
        """Save preprocessing information to file"""
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        info = {
            'config': self.config,
            'statistics': stats,
            'split_statistics': split_stats
        }
        
        info_path = processed_dir / 'preprocessing_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n💾 Preprocessing info saved to: {info_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Plant Disease Classification - Preprocessing Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='../../data/raw/PlantVillage',
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='../../data/processed',
        help='Path to output processed data directory'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Target image size'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for data loaders'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PreprocessingPipeline(config_path=args.config)
    
    # Override config with command-line arguments if provided
    if args.raw_dir:
        pipeline.config['data']['raw_dir'] = args.raw_dir
    if args.processed_dir:
        pipeline.config['data']['processed_dir'] = args.processed_dir
    if args.image_size:
        pipeline.config['data']['image_size'] = args.image_size
    if args.batch_size:
        pipeline.config['dataloader']['batch_size'] = args.batch_size
    if args.seed:
        pipeline.config['data']['seed'] = args.seed
    
    # Run pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
