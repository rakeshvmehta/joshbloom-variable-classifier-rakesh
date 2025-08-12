"""
Hierarchical Galaxy Classification Training Script

This script implements the complete training pipeline for hierarchical galaxy classification,
combining:
- Dual-output CNN architecture (embeddings + classifications)
- Hierarchical loss function (embedding + classification loss)
- Galaxy dataset with proper preprocessing
- Evaluation metrics for both embedding quality and classification accuracy

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

# Add parent directory to path to import galaxy modules
sys.path.append('..')
from process_galaxy_dataset import get_data_loaders

# Import our hierarchical modules
from hierarchical_galaxy_cnn import create_hierarchical_model
from hierarchical_loss import create_loss_function


class HierarchicalTrainer:
    """Complete training pipeline for hierarchical galaxy classification."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.setup_paths()
        
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
        
    def _setup_device(self):
        """Setup the appropriate device for training."""
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        print(f"Using device: {device}")
        return device
    
    def setup_paths(self):
        """Setup paths for saving models and results."""
        self.save_dir = Path(self.config.get('save_dir', 'hierarchical_results'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.save_dir / 'models').mkdir(exist_ok=True)
        (self.save_dir / 'plots').mkdir(exist_ok=True)
        (self.save_dir / 'logs').mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and prepare the galaxy dataset."""
        print("Loading galaxy dataset...")
        
        # Use your existing data pipeline
        data = get_data_loaders(
            image_dir=self.config.get('image_dir', '../training_images'),
            labels_file=self.config.get('labels_file', '../training_classifications.csv'),
            downsized_dir=self.config.get('downsized_dir', '../downsized_galaxy_images'),
            batch_size=self.config.get('batch_size', 64),
            num_workers=self.config.get('num_workers', 4),
            train_ratio=self.config.get('train_ratio', 0.8),
            cache_size=self.config.get('cache_size', 1000)
        )
        
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
    def create_model(self):
        """Create the hierarchical model."""
        print("Creating hierarchical model...")
        
        model_config = {
            'num_classes': 37,
            'embedding_dim': 37,
            'dropout_rate': self.config.get('dropout_rate', 0.3)
        }
        
        architecture = self.config.get('architecture', 'cnn')
        if architecture == 'resnet':
            model_config['backbone'] = self.config.get('resnet_backbone', 'resnet18')
            model_config['pretrained'] = self.config.get('pretrained', True)
        
        self.model = create_hierarchical_model(architecture, **model_config)
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {type(self.model).__name__}")
        print(f"Total parameters: {total_params:,}")
        
    def create_loss_function(self):
        """Create the hierarchical loss function."""
        print("Setting up hierarchical loss function...")
        
        loss_config = {
            'classification_weight': self.config.get('classification_weight', 0.1),
            'loss_type': self.config.get('loss_type', 'inv_corr'),
            'device': self.device
        }
        
        embedding_path = self.config.get('embedding_path', 'galaxy_hierarchy_embeddings.unitsphere.pickle')
        
        self.criterion = create_loss_function(embedding_path, loss_config)
        self.criterion = self.criterion.to(self.device)
        
    def create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        print("Setting up optimizer...")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get('lr_factor', 0.5),
            patience=self.config.get('lr_patience', 5),
            verbose=True
        )
        
    def convert_labels_to_indices(self, one_hot_labels):
        """Convert one-hot encoded labels to class indices."""
        return torch.argmax(one_hot_labels, dim=1)
    
    def compute_accuracy(self, predictions, targets):
        """Compute classification accuracy."""
        predicted_classes = torch.argmax(predictions, dim=1)
        correct = (predicted_classes == targets).float()
        return correct.mean().item()
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_embedding_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            one_hot_labels = batch['labels'].to(self.device)
            
            # Convert one-hot to class indices for hierarchical loss
            class_indices = self.convert_labels_to_indices(one_hot_labels)
            
            # Forward pass with dual outputs
            pred_embeddings, pred_classifications = self.model(images)
            
            # Compute hierarchical loss
            total_loss, embedding_loss, classification_loss = self.criterion(
                pred_embeddings, pred_classifications, class_indices
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self.compute_accuracy(pred_classifications, class_indices)
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_embedding_loss += embedding_loss.item()
            epoch_classification_loss += classification_loss.item()
            epoch_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Emb': f'{embedding_loss.item():.4f}',
                'Cls': f'{classification_loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
        
        # Average metrics
        return (epoch_loss / num_batches, epoch_embedding_loss / num_batches, 
                epoch_classification_loss / num_batches, epoch_accuracy / num_batches)
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_embedding_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                one_hot_labels = batch['labels'].to(self.device)
                
                # Convert one-hot to class indices
                class_indices = self.convert_labels_to_indices(one_hot_labels)
                
                # Forward pass
                pred_embeddings, pred_classifications = self.model(images)
                
                # Compute loss
                total_loss, embedding_loss, classification_loss = self.criterion(
                    pred_embeddings, pred_classifications, class_indices
                )
                
                # Compute accuracy
                accuracy = self.compute_accuracy(pred_classifications, class_indices)
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_embedding_loss += embedding_loss.item()
                epoch_classification_loss += classification_loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Emb': f'{embedding_loss.item():.4f}',
                    'Cls': f'{classification_loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        # Average metrics
        return (epoch_loss / num_batches, epoch_embedding_loss / num_batches,
                epoch_classification_loss / num_batches, epoch_accuracy / num_batches)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / 'models' / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'models' / 'best_hierarchical_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved! (Val Loss: {self.best_val_loss:.4f})")
    
    def plot_training_metrics(self):
        """Plot and save training metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot total losses
        ax1.plot(epochs, self.train_losses, 'o-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 's-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training and Validation Losses')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'o-', label='Train Accuracy')
        ax2.plot(epochs, self.val_accuracies, 's-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracies')
        ax2.legend()
        ax2.grid(True)
        
        # Plot embedding vs classification losses
        if len(self.embedding_losses) > 0:
            ax3.plot(epochs, self.embedding_losses, 'o-', label='Embedding Loss')
            ax3.plot(epochs, self.classification_losses, 's-', label='Classification Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Embedding vs Classification Losses')
            ax3.legend()
            ax3.grid(True)
        
        # Plot loss ratio
        if len(self.embedding_losses) > 0 and len(self.classification_losses) > 0:
            loss_ratios = [e / (c + 1e-8) for e, c in zip(self.embedding_losses, self.classification_losses)]
            ax4.plot(epochs, loss_ratios, 'o-', label='Embedding/Classification Ratio')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Ratio')
            ax4.set_title('Embedding to Classification Loss Ratio')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'plots' / 'hierarchical_training_metrics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        print("Starting hierarchical training...")
        print("=" * 60)
        
        num_epochs = self.config.get('num_epochs', 20)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_emb_loss, train_cls_loss, train_acc = self.train_epoch()
            
            # Validation phase  
            val_loss, val_emb_loss, val_cls_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.embedding_losses.append(val_emb_loss)
            self.classification_losses.append(val_cls_loss)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Embedding Loss: {val_emb_loss:.4f} | Classification Loss: {val_cls_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Plot metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_metrics()
        
        print("\n" + "=" * 60)
        print("Hierarchical training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Final plots
        self.plot_training_metrics()


def get_default_config():
    """Get default training configuration."""
    return {
        # Data settings
        'image_dir': '../training_images',
        'labels_file': '../training_classifications.csv', 
        'downsized_dir': '../downsized_galaxy_images',
        'embedding_path': 'galaxy_hierarchy_embeddings.unitsphere.pickle',
        
        # Model settings
        'architecture': 'cnn',  # 'cnn' or 'resnet'
        'resnet_backbone': 'resnet18',
        'pretrained': True,
        'dropout_rate': 0.3,
        
        # Training settings
        'batch_size': 64,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        
        # Loss settings
        'classification_weight': 0.1,
        'loss_type': 'inv_corr',
        
        # Scheduler settings
        'lr_factor': 0.5,
        'lr_patience': 5,
        
        # Data loader settings
        'num_workers': 4,
        'train_ratio': 0.8,
        'cache_size': 1000,
        
        # Save settings
        'save_dir': 'hierarchical_results'
    }


def main():
    """Main training function."""
    print("Hierarchical Galaxy Classification Training")
    print("=" * 60)
    
    # Get configuration
    config = get_default_config()
    
    # Initialize trainer
    trainer = HierarchicalTrainer(config)
    
    # Setup training components
    trainer.load_data()
    trainer.create_model()
    trainer.create_loss_function()
    trainer.create_optimizer()
    
    # Save configuration
    config_path = trainer.save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 