"""
Training Script for Plant Disease Classification
Implements training loop with validation, checkpointing, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np


def save_model(
    model: nn.Module,
    save_path: str,
    epoch: int,
    optimizer_state: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
):
    """
    Save model checkpoint with metadata
    
    Args:
        model: PyTorch model to save
        save_path: Path to save checkpoint
        epoch: Current epoch number
        optimizer_state: Optimizer state dict (optional)
        metrics: Training metrics (optional)
        config: Model configuration (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to: {save_path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """Main trainer class for plant disease classification"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_dir: str = '../../results/models',
        log_dir: str = '../../results/logs',
        use_tensorboard: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cuda' or 'cpu')
            scheduler: Learning rate scheduler (optional)
            save_dir: Directory to save model checkpoints
            log_dir: Directory for tensorboard logs
            use_tensorboard: Whether to use tensorboard logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Create directories
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.start_epoch = 1
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"Resuming from epoch {checkpoint['epoch']}")
            print(f"  Previous Val Acc: {metrics.get('val_acc', 'N/A')}")
            print(f"  Previous Val Loss: {metrics.get('val_loss', 'N/A')}")
        
        print(f"Will continue training from epoch {self.start_epoch}\n")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        if is_best:
            checkpoint_name = 'best_model.pth'
        
        save_path = os.path.join(self.save_dir, checkpoint_name)
        
        save_model(
            model=self.model,
            save_path=save_path,
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics={
                'train_loss': self.history['train_loss'][-1],
                'train_acc': self.history['train_acc'][-1],
                'val_loss': self.history['val_loss'][-1],
                'val_acc': self.history['val_acc'][-1],
            }
        )
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_frequency: int = 5
    ):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_frequency: Save checkpoint every N epochs
        """
        print("="*70)
        print("TRAINING START")
        print("="*70)
        
        # Model info
        params = count_parameters(self.model)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("="*70)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'  # Maximize validation accuracy
        )
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ New best validation accuracy: {val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Periodic checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if early_stopping(val_acc):
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)
        
        # Save training history
        self.save_history()
        
        if self.writer:
            self.writer.close()
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
    
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'reduce_on_plateau',
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name is None or scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # Example usage
    print("Training script loaded successfully!")
    print("Use this script with a configuration file or import the Trainer class.")
