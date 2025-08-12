"""
Unified Experiment Tracking System

This module provides a consistent experiment tracking system that can be used
by both hierarchical and non-hierarchical approaches. It automatically creates
experiment directories, logs configurations, metrics, and results.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os
import sys

class ExperimentTracker:
    """Unified experiment tracking for both hierarchical and non-hierarchical approaches."""
    
    def __init__(self, experiment_name=None, approach_type="non_hierarchical", custom_config=None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name for this experiment (auto-generated if None)
            approach_type: "hierarchical" or "non_hierarchical"
            custom_config: Dict of config overrides for this experiment
        """
        self.approach_type = approach_type
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{approach_type}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Setup experiment directory
        if approach_type == "hierarchical":
            self.experiment_dir = Path('hierarchy/experiments') / self.experiment_name
        else:
            self.experiment_dir = Path('non_hierarchical/experiments') / self.experiment_name
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / 'model_checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        
        # Start with default config
        self.config = self._get_default_config()
        
        # Apply custom overrides if provided
        if custom_config:
            self.config.update(custom_config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Save config for this experiment
        self._save_experiment_config()
        
        print(f"Starting {approach_type} experiment: {self.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        self._print_config()
        
    def _get_default_config(self):
        """Get default configuration based on approach type."""
        # Import global config here to avoid circular imports
        try:
            from global_config import config as global_config
        except ImportError:
            # Fallback to defaults if global config not available
            if self.approach_type == "hierarchical":
                return {
                    'APPROACH_TYPE': 'hierarchical',
                    'MODEL_TYPE': 'resnet18',
                    'EMBEDDING_DIM': 128,
                    'NUM_EPOCHS': 100,
                    'LEARNING_RATE': 0.001,
                    'BATCH_SIZE': 64,
                    'OPTIMIZER': 'adam',
                    'WEIGHT_DECAY': 1e-4,
                    'SCHEDULER': 'step',
                    'EMBEDDING_LOSS_TYPE': 'inv_corr',
                    'CLASSIFICATION_WEIGHT': 0.1,
                    'EMBEDDING_WEIGHT': 1.0
                }
            else:
                return {
                    'APPROACH_TYPE': 'non_hierarchical',
                    'MODEL_TYPE': 'resnet18',
                    'NUM_EPOCHS': 100,
                    'LEARNING_RATE': 0.001,
                    'BATCH_SIZE': 64,
                    'OPTIMIZER': 'adam',
                    'WEIGHT_DECAY': 1e-4,
                    'SCHEDULER': 'plateau',
                    'DROPOUT_RATE': 0.3,
                    'PRETRAINED': True
                }
        
        # Use global config if available
        return global_config.get_approach_config()
    
    def _save_experiment_config(self):
        """Save the experiment configuration."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Also save to experiments summary
        self._update_experiments_summary()
    
    def _update_experiments_summary(self):
        """Update the master experiments summary CSV."""
        if self.approach_type == "hierarchical":
            summary_path = Path('hierarchy/experiments') / 'experiments_summary.csv'
        else:
            summary_path = Path('non_hierarchical/experiments') / 'experiments_summary.csv'
        
        # Create summary if it doesn't exist
        if not summary_path.exists():
            summary_df = pd.DataFrame(columns=[
                'experiment_name', 'timestamp', 'approach_type', 'model_type',
                'learning_rate', 'batch_size', 'num_epochs', 'best_val_accuracy', 
                'best_val_loss', 'status', 'notes'
            ])
        else:
            summary_df = pd.read_csv(summary_path)
        
        # Add/update current experiment
        experiment_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'approach_type': self.approach_type,
            'model_type': self.config.get('MODEL_TYPE', 'unknown'),
            'learning_rate': self.config.get('LEARNING_RATE', 0.0),
            'batch_size': self.config.get('BATCH_SIZE', 0),
            'num_epochs': self.config.get('NUM_EPOCHS', 0),
            'best_val_accuracy': 0.0,
            'best_val_loss': float('inf'),
            'status': 'running',
            'notes': ''
        }
        
        # Remove existing entry if it exists
        summary_df = summary_df[summary_df['experiment_name'] != self.experiment_name]
        
        # Add new entry
        summary_df = pd.concat([summary_df, pd.DataFrame([experiment_data])], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
    
    def _print_config(self):
        """Print current configuration."""
        print("=" * 50)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 50)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, learning_rate):
        """Log metrics for the current epoch."""
        self.current_epoch = epoch
        
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(learning_rate)
        
        # Update best metrics
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_val_loss = val_loss
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {self.best_val_accuracy:.4f}")
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_path = self.experiment_dir / 'model_checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.experiment_dir / 'model_checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved! Validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Save last model
        last_model_path = self.experiment_dir / 'model_checkpoints' / 'last_model.pth'
        torch.save(checkpoint, last_model_path)
    
    def save_metrics(self):
        """Save training metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
        
        metrics_path = self.experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.experiment_name} ({self.approach_type})', fontsize=16)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, label='Train Accuracy', color='blue')
        axes[0, 1].plot(epochs, self.val_accuracies, label='Val Accuracy', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, label='Learning Rate', color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss vs Accuracy
        axes[1, 1].scatter(self.val_losses, self.val_accuracies, color='red', alpha=0.6)
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / 'plots' / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")
    
    def finish_experiment(self):
        """Mark experiment as completed and save final results."""
        # Final save
        self.save_metrics()
        self.plot_training_curves()
        
        # Update experiments summary
        self._update_experiments_summary()
        
        print(f"Experiment completed! Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Results saved to: {self.experiment_dir}")

def create_experiment_tracker(experiment_name=None, approach_type="non_hierarchical", custom_config=None):
    """Factory function to create experiment tracker."""
    return ExperimentTracker(experiment_name, approach_type, custom_config) 