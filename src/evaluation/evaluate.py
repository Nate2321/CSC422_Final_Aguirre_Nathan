"""
Model Evaluation Script for Plant Disease Classification
Comprehensive evaluation on test set with detailed metrics and analysis
"""

import torch
import torch.nn as nn
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_model import create_resnet50
from preprocessing.data_loader import PlantDiseaseDataset
from torch.utils.data import DataLoader
from preprocessing.augmentation import get_validation_transforms
from metrics import MetricsCalculator, evaluate_model, calculate_top_k_accuracy


def load_model(
    num_classes: int,
    checkpoint_path: str,
    device: str = 'cpu'
) -> tuple:
    """
    Load ResNet50 model from checkpoint
    
    Args:
        num_classes: Number of classes
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_data)
    """
    # Create model
    model = create_resnet50(num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    
    return model, checkpoint


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: list,
    save_dir: str
):
    """
    Comprehensive evaluation on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        class_names: List of class names
        save_dir: Directory to save results
    """
    print("="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    model.eval()
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(len(class_names), class_names)
    
    # Additional metrics
    top_5_correct = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("\nRunning inference on test set...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            # Clip predictions to valid class range (in case model outputs more classes than test set has)
            predictions = torch.clamp(predictions, 0, len(class_names) - 1)
            
            # Update metrics
            metrics_calc.update(predictions, targets, probabilities[:, :len(class_names)])
            
            # Top-5 accuracy
            top_5_acc = calculate_top_k_accuracy(outputs[:, :len(class_names)], targets, k=min(5, len(class_names)))
            top_5_correct += top_5_acc * targets.size(0)
            total_samples += targets.size(0)
            
            # Store predictions
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities[:, :len(class_names)].cpu().numpy())
    
    # Compute metrics
    print("\n")
    metrics_calc.print_metrics()
    
    # Top-5 accuracy
    top_5_accuracy = top_5_correct / total_samples
    print(f"\nTop-5 Accuracy: {top_5_accuracy:.4f} ({top_5_accuracy*100:.2f}%)")
    
    # Save metrics
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'test_metrics.json')
    
    metrics = metrics_calc.compute_metrics()
    metrics['top_5_accuracy'] = float(top_5_accuracy)
    metrics['total_samples'] = total_samples
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save classification report
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(metrics_calc.get_classification_report())
    
    print(f"Classification report saved to: {report_path}")
    
    return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate Plant Disease Classification Model (ResNet50)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='../../data/processed/test',
        help='Test data directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='../../results/evaluation',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load test dataset to get class names and count
    print(f"\nLoading test dataset from: {args.test_dir}")
    test_dataset = PlantDiseaseDataset(
        root_dir=args.test_dir,
        transform=get_validation_transforms(image_size=224),
        split='test'
    )
    
    num_classes = len(test_dataset.class_to_idx)
    class_names = [test_dataset.idx_to_class[i] for i in range(num_classes)]
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading ResNet50 model...")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Get num_classes from checkpoint
    temp_ckpt = torch.load(args.checkpoint, map_location=device)
    checkpoint_num_classes = num_classes  # default
    if 'model_state_dict' in temp_ckpt:
        # Infer num_classes from final layer bias shape (last fc layer)
        if 'model.fc.4.bias' in temp_ckpt['model_state_dict']:
            checkpoint_num_classes = temp_ckpt['model_state_dict']['model.fc.4.bias'].shape[0]
            print(f"Model was trained with {checkpoint_num_classes} classes")
    
    model, checkpoint = load_model(
        num_classes=checkpoint_num_classes,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Evaluate
    evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=args.save_dir
    )
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
