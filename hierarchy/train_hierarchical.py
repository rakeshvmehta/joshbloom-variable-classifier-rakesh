"""
Hierarchical Galaxy Classification Training Script

This script implements the complete training pipeline for hierarchical galaxy classification,
combining:
- Dual-output CNN architecture (embeddings + classifications)
- Hierarchical loss function (embedding + classification loss)
- Galaxy dataset with proper preprocessing
- Evaluation metrics for both embedding quality and classification accuracy
- Experiment tracking and logging

Based on Sydney's variable classifier training approach but adapted for galaxy morphology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
import pickle
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path to import galaxy modules
sys.path.append('..')
from data_processing.process_galaxy_dataset import get_data_loaders

# Import our hierarchical modules
from hierarchical_galaxy_cnn import create_hierarchical_model
from hierarchical_loss import create_loss_function
from config import config

class HierarchicalTrainer:
    """Complete training pipeline for hierarchical galaxy classification."""
    
    def __init__(self, experiment_name=None, custom_config=None):
        """
        Initialize trainer with global config and optional custom overrides.
        
        Args:
            experiment_name: Name for this experiment (auto-generated if None)
            custom_config: Dict of config overrides for this experiment
        """
        # Start with global config
        self.config = config.to_dict()
        
        # Apply custom overrides if provided
        if custom_config:
            self.config.update(custom_config)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{config.EXPERIMENT_NAME_PREFIX}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Setup experiment directory
        self.experiment_dir = Path('experiments') / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / 'model_checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        
        # Setup device
        self.device = config.get_device()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.embedding_losses = []
        self.classification_losses = []
        self.learning_rates = []
        
        # Save config for this experiment
        self._save_experiment_config()
        
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        config.print_config()
        
    def _save_experiment_config(self):
        """Save the experiment configuration."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Also save to experiments summary
        self._update_experiments_summary()
    
    def _update_experiments_summary(self):
        """Update the master experiments summary CSV."""
        summary_path = Path('experiments') / 'experiments_summary.csv'
        
        # Create summary if it doesn't exist
        if not summary_path.exists():
            summary_df = pd.DataFrame(columns=[
                'experiment_name', 'timestamp', 'learning_rate', 'batch_size', 
                'model_type', 'embedding_dim', 'classification_weight', 
                'embedding_loss_type', 'num_epochs', 'best_val_accuracy', 
                'best_val_loss', 'status', 'notes'
            ])
        else:
            summary_df = pd.read_csv(summary_path)
        
        # Add/update current experiment
        experiment_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'learning_rate': self.config['LEARNING_RATE'],
            'batch_size': self.config['BATCH_SIZE'],
            'model_type': self.config['MODEL_TYPE'],
            'embedding_dim': self.config['EMBEDDING_DIM'],
            'classification_weight': self.config['CLASSIFICATION_WEIGHT'],
            'embedding_loss_type': self.config['EMBEDDING_LOSS_TYPE'],
            'num_epochs': self.config['NUM_EPOCHS'],
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
    
    def load_data(self):
        """Load and prepare the galaxy dataset."""
        print("Loading galaxy dataset...")
        
        # Use your existing data pipeline
        data = get_data_loaders(
            image_dir=self.config['IMAGE_DIR'],
            labels_file=self.config['LABELS_FILE'],
            downsized_dir=self.config['DOWNSIZED_DIR'],
            batch_size=self.config['BATCH_SIZE'],
            num_workers=self.config['NUM_WORKERS'],
            train_ratio=self.config['TRAIN_RATIO'],
            cache_size=self.config['CACHE_SIZE']
        )
        
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        
        # Set number of classes from dataset
        self.config['NUM_CLASSES'] = len(data['class_names'])
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of classes: {self.config['NUM_CLASSES']}")
        
    def create_model(self):
        """Create the hierarchical CNN model."""
        print("Creating hierarchical model...")
        
        self.model = create_hierarchical_model(
            model_type=self.config['MODEL_TYPE'],
            num_classes=self.config['NUM_CLASSES'],
            embedding_dim=self.config['EMBEDDING_DIM']
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def create_loss_function(self):
        """Create the hierarchical loss function."""
        print("Creating hierarchical loss function...")
        
        self.loss_function = create_loss_function(
            embedding_path=self.config['EMBEDDING_PATH'],
            classification_weight=self.config['CLASSIFICATION_WEIGHT'],
            loss_type=self.config['EMBEDDING_LOSS_TYPE'],
            device=self.device
        )
        
    def create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        print("Creating optimizer and scheduler...")
        
        # Create optimizer
        if self.config['OPTIMIZER'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['LEARNING_RATE'],
                weight_decay=self.config['WEIGHT_DECAY']
            )
        elif self.config['OPTIMIZER'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['LEARNING_RATE'],
                weight_decay=self.config['WEIGHT_DECAY'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['OPTIMIZER']}")
        
        # Create scheduler
        if self.config['SCHEDULER'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['SCHEDULER_STEP_SIZE'],
                gamma=self.config['SCHEDULER_GAMMA']
            )
        elif self.config['SCHEDULER'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['NUM_EPOCHS']
            )
        elif self.config['SCHEDULER'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
            
        print(f"Optimizer: {self.config['OPTIMIZER']}")
        print(f"Scheduler: {self.config['SCHEDULER']}")
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_embedding_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings, classifications = self.model(images)
            
            # Compute loss
            loss, loss_components = self.loss_function(embeddings, classifications, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            _, predicted = torch.max(classifications, 1)
            accuracy = (predicted == labels).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            total_embedding_loss += loss_components['embedding_loss'].item()
            total_classification_loss += loss_components['classification_loss'].item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.config['LOG_FREQUENCY'] == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy:.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # Return average metrics
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'embedding_loss': total_embedding_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches
        }
        
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_embedding_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings, classifications = self.model(images)
                
                # Compute loss
                loss, loss_components = self.loss_function(embeddings, classifications, labels)
                
                # Compute accuracy
                _, predicted = torch.max(classifications, 1)
                accuracy = (predicted == labels).float().mean().item()
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                total_embedding_loss += loss_components['embedding_loss'].item()
                total_classification_loss += loss_components['classification_loss'].item()
                num_batches += 1
        
        # Return average metrics
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'embedding_loss': total_embedding_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches
        }
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
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
        
        # Save last model if requested
        if self.config['SAVE_LAST_MODEL']:
            last_model_path = self.experiment_dir / 'model_checkpoints' / 'last_model.pth'
            torch.save(checkpoint, last_model_path)
            
    def save_metrics(self):
        """Save training metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'embedding_losses': self.embedding_losses,
            'classification_losses': self.classification_losses,
            'learning_rates': self.learning_rates
        }
        
        metrics_path = self.experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.experiment_name}', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='red')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Component losses
        axes[1, 0].plot(self.embedding_losses, label='Embedding Loss', color='green')
        axes[1, 0].plot(self.classification_losses, label='Classification Loss', color='orange')
        axes[1, 0].set_title('Component Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.learning_rates, label='Learning Rate', color='purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / 'plots' / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")
        
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Setup
        self.load_data()
        self.create_model()
        self.create_loss_function()
        self.create_optimizer()
        
        # Training loop
        for epoch in range(self.config['NUM_EPOCHS']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['VALIDATION_FREQUENCY'] == 0:
                val_metrics = self.validate_epoch(epoch)
                
                # Update best metrics
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                
                # Print epoch summary
                print(f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Best Val Acc: {self.best_val_accuracy:.4f}")
                
                # Store metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_accuracies.append(train_metrics['accuracy'])
                self.val_accuracies.append(val_metrics['accuracy'])
                self.embedding_losses.append(train_metrics['embedding_loss'])
                self.classification_losses.append(train_metrics['classification_loss'])
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                
                # Save checkpoint
                is_best = val_metrics['accuracy'] == self.best_val_accuracy
                if epoch % self.config['CHECKPOINT_FREQUENCY'] == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                # Update plots
                if epoch % self.config['PLOT_FREQUENCY'] == 0:
                    self.plot_training_curves()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
        
        # Final save
        self.save_metrics()
        self.plot_training_curves()
        self.save_checkpoint(self.config['NUM_EPOCHS'] - 1, False)
        
        # Update experiments summary
        self._update_experiments_summary()
        
        print(f"Training completed! Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Results saved to: {self.experiment_dir}")

def run_experiment(experiment_name=None, custom_config=None):
    """Run a single experiment with the given configuration."""
    trainer = HierarchicalTrainer(experiment_name, custom_config)
    trainer.train()
    return trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hierarchical galaxy classification training')
    parser.add_argument('--name', type=str, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--lr', type=float, help='Learning rate override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--cls_weight', type=float, help='Classification weight override')
    parser.add_argument('--epochs', type=int, help='Number of epochs override')
    
    args = parser.parse_args()
    
    # Build custom config from command line args
    custom_config = {}
    if args.lr:
        custom_config['LEARNING_RATE'] = args.lr
    if args.batch_size:
        custom_config['BATCH_SIZE'] = args.batch_size
    if args.cls_weight:
        custom_config['CLASSIFICATION_WEIGHT'] = args.cls_weight
    if args.epochs:
        custom_config['NUM_EPOCHS'] = args.epochs
    
    # Run experiment
    trainer = run_experiment(args.name, custom_config if custom_config else None) 