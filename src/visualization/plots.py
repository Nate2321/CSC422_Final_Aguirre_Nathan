"""
Visualization Tools for Plant Disease Classification
Plotting utilities for training metrics, confusion matrices, and predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import os


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure (if None, shows plot)
        figsize: Figure size
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = False
):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    # For large number of classes, show without annotations
    if len(class_names) > 20:
        sns.heatmap(
            confusion_matrix,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=False
        )
    else:
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names
        )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(
    metrics: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    top_n: int = 20
):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        metrics: Metrics dictionary from MetricsCalculator
        save_path: Path to save figure
        figsize: Figure size
        top_n: Number of classes to show (sorted by F1-score)
    """
    per_class = metrics['per_class_metrics']
    
    # Extract data
    classes = list(per_class.keys())
    precisions = [per_class[c]['precision'] for c in classes]
    recalls = [per_class[c]['recall'] for c in classes]
    f1_scores = [per_class[c]['f1_score'] for c in classes]
    
    # Sort by F1-score and take top_n
    sorted_indices = np.argsort(f1_scores)[::-1][:top_n]
    classes = [classes[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # Create plot
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Metrics (Top {top_n} by F1-Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(
    class_distribution: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot class distribution bar chart
    
    Args:
        class_distribution: Dictionary mapping class names to counts
        save_path: Path to save figure
        figsize: Figure size
    """
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    plt.figure(figsize=figsize)
    
    bars = plt.bar(range(len(classes)), counts, alpha=0.8)
    
    # Color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    probabilities: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    n_samples: int = 16,
    figsize: Tuple[int, int] = (16, 16)
):
    """
    Plot sample predictions with images
    
    Args:
        images: Batch of images [N, C, H, W]
        predictions: Predicted labels [N]
        targets: True labels [N]
        class_names: List of class names
        probabilities: Prediction probabilities [N, num_classes]
        save_path: Path to save figure
        n_samples: Number of samples to show
        figsize: Figure size
    """
    # Convert to numpy
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    if probabilities is not None:
        probabilities = probabilities.cpu().numpy()
    
    # Denormalize images (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Denormalize image
        img = images[i].transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Get labels
        pred_class = class_names[predictions[i]]
        true_class = class_names[targets[i]]
        correct = predictions[i] == targets[i]
        
        # Title
        color = 'green' if correct else 'red'
        title = f'True: {true_class}\nPred: {pred_class}'
        
        if probabilities is not None:
            conf = probabilities[i][predictions[i]]
            title += f'\nConf: {conf:.2%}'
        
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    n_classes_to_plot: int = 10
):
    """
    Plot ROC curves for top N classes
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities [N, num_classes]
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        n_classes_to_plot: Number of classes to plot
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Sort classes by AUC
    sorted_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)
    
    plt.figure(figsize=figsize)
    
    # Plot top N classes
    for i in sorted_classes[:n_classes_to_plot]:
        plt.plot(
            fpr[i], tpr[i],
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves (Top {n_classes_to_plot} Classes)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_evaluation_report(
    metrics_path: str,
    history_path: str,
    output_dir: str
):
    """
    Create comprehensive evaluation report with all visualizations
    
    Args:
        metrics_path: Path to test_metrics.json
        history_path: Path to training_history.json
        output_dir: Directory to save all visualizations
    """
    print("="*70)
    print("CREATING EVALUATION REPORT")
    print("="*70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Get class names
    class_names = list(metrics['per_class_metrics'].keys())
    
    # 1. Training history
    print("\n1. Plotting training history...")
    plot_training_history(
        history_path,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # 2. Confusion matrix
    print("2. Plotting confusion matrix...")
    conf_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        conf_matrix,
        class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 3. Normalized confusion matrix
    print("3. Plotting normalized confusion matrix...")
    plot_confusion_matrix(
        conf_matrix,
        class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # 4. Per-class metrics
    print("4. Plotting per-class metrics...")
    plot_per_class_metrics(
        metrics,
        save_path=os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    print("\n" + "="*70)
    print("EVALUATION REPORT COMPLETE")
    print("="*70)
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Import functions to create plots and visualizations.")
