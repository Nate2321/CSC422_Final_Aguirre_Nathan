"""
Evaluation Metrics for Plant Disease Classification
Implements comprehensive metrics: accuracy, precision, recall, F1-score, confusion matrix
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score
)
from typing import Dict, List, Tuple, Optional
import json


class MetricsCalculator:
    """Calculate and store evaluation metrics"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: List of class names (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch
        
        Args:
            predictions: Predicted class indices [batch_size]
            targets: True class indices [batch_size]
            probabilities: Prediction probabilities [batch_size, num_classes]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def compute_metrics(self) -> Dict:
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all metrics
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Macro and weighted averages
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(self.num_classes)))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'cohen_kappa': float(kappa),
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
                for i in range(self.num_classes)
            }
        }
        
        # ROC-AUC if probabilities are available
        if len(self.all_probabilities) > 0:
            try:
                y_prob = np.array(self.all_probabilities)
                # One-vs-rest ROC-AUC
                roc_auc_ovr = roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr',
                    average='macro'
                )
                metrics['roc_auc_ovr'] = float(roc_auc_ovr)
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    def get_classification_report(self) -> str:
        """
        Get sklearn classification report
        
        Returns:
            Classification report as string
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            labels=list(range(self.num_classes)),
            zero_division=0
        )
        
        return report
    
    def print_metrics(self):
        """Print all metrics in a formatted way"""
        metrics = self.compute_metrics()
        
        print("="*70)
        print("EVALUATION METRICS")
        print("="*70)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        print(f"\nMacro Average:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")
        
        print(f"\nWeighted Average:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_ovr' in metrics:
            print(f"\nROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
        print("="*70)
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(self.get_classification_report())
    
    def save_metrics(self, save_path: str):
        """
        Save metrics to JSON file
        
        Args:
            save_path: Path to save metrics
        """
        metrics = self.compute_metrics()
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Determine number of classes
    num_classes = model.num_classes if hasattr(model, 'num_classes') else None
    
    # If num_classes not available, try to infer from first batch
    if num_classes is None:
        with torch.no_grad():
            sample_input, _ = next(iter(data_loader))
            sample_output = model(sample_input.to(device))
            num_classes = sample_output.shape[1]
    
    metrics_calc = MetricsCalculator(num_classes, class_names)
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            # Update metrics
            metrics_calc.update(predictions, targets, probabilities)
    
    # Compute and print metrics
    metrics_calc.print_metrics()
    
    return metrics_calc.compute_metrics()


def calculate_top_k_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: True labels [batch_size]
        k: Top-k value
    
    Returns:
        Top-k accuracy
    """
    _, top_k_preds = outputs.topk(k, dim=1)
    targets = targets.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(targets).sum().item()
    total = targets.size(0)
    
    return correct / total


def calculate_class_accuracy(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate per-class accuracy from confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to accuracies
    """
    num_classes = confusion_matrix.shape[0]
    class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    class_accuracies = {}
    
    for i in range(num_classes):
        total = confusion_matrix[i, :].sum()
        if total > 0:
            accuracy = confusion_matrix[i, i] / total
        else:
            accuracy = 0.0
        class_accuracies[class_names[i]] = float(accuracy)
    
    return class_accuracies


def get_misclassified_examples(
    predictions: List[int],
    targets: List[int],
    image_paths: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    top_n: int = 10
) -> List[Dict]:
    """
    Get top misclassified examples
    
    Args:
        predictions: Predicted labels
        targets: True labels
        image_paths: Paths to images (optional)
        class_names: Class names (optional)
        top_n: Number of examples to return
    
    Returns:
        List of misclassified examples
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Find misclassified indices
    misclassified_idx = np.where(predictions != targets)[0]
    
    # Limit to top_n
    if len(misclassified_idx) > top_n:
        misclassified_idx = misclassified_idx[:top_n]
    
    examples = []
    for idx in misclassified_idx:
        example = {
            'index': int(idx),
            'true_label': int(targets[idx]),
            'predicted_label': int(predictions[idx])
        }
        
        if class_names:
            example['true_class'] = class_names[targets[idx]]
            example['predicted_class'] = class_names[predictions[idx]]
        
        if image_paths:
            example['image_path'] = image_paths[idx]
        
        examples.append(example)
    
    return examples


if __name__ == "__main__":
    # Example usage
    print("Metrics module loaded successfully!")
    
    # Simulate some predictions
    num_classes = 5
    num_samples = 100
    
    np.random.seed(42)
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(num_samples, 20, replace=False)
    y_pred[error_indices] = np.random.randint(0, num_classes, 20)
    
    # Create metrics calculator
    class_names = [f"Disease_{i}" for i in range(num_classes)]
    metrics_calc = MetricsCalculator(num_classes, class_names)
    
    # Update with predictions
    metrics_calc.update(y_pred, y_true)
    
    # Print metrics
    metrics_calc.print_metrics()
    
    # Get confusion matrix
    metrics = metrics_calc.compute_metrics()
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
