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
from data_processing.process_galaxy_dataset import get_data_loaders

# Import our hierarchical modules
from hierarchical_galaxy_cnn import create_hierarchical_model
from hierarchical_loss import create_loss_function

# Import from main global config
import sys
sys.path.append('..')
from global_config import config

class HierarchicalTrainer:
    """Complete training pipeline for hierarchical galaxy classification."""
    
    def __init__(self, custom_config=None):
        """
        Initialize trainer with global config and optional custom overrides.
        
        Args:
            custom_config: Dict of config overrides for this experiment
        """
        # Start with global config
        self.config = config.to_dict()
        
        # Apply custom overrides if provided
        if custom_config:
            self.config.update(custom_config)
        
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
        
        print(f"Starting hierarchical training with config:")
        config.print_config()
        
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
            seed=self.config['SEED'],
            cache_size=self.config['CACHE_SIZE']
        )
        
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        self.class_names = data['class_names']
        
        # Set number of classes in config
        self.config['NUM_CLASSES'] = len(self.class_names)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of classes: {len(self.class_names)}")
        
        return data
    
    def create_model(self):
        """Create the hierarchical model."""
        print(f"Creating hierarchical model...")
        
        model = create_hierarchical_model(
            model_type=self.config['MODEL_TYPE'],
            num_classes=self.config['NUM_CLASSES'],
            embedding_dim=self.config['EMBEDDING_DIM']
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_loss_function(self):
        """Create the hierarchical loss function."""
        print(f"Creating hierarchical loss function...")
        
        loss_function = create_loss_function(
            embedding_path=self.config['EMBEDDING_PATH'],
            classification_weight=self.config['CLASSIFICATION_WEIGHT'],
            loss_type=self.config['EMBEDDING_LOSS_TYPE'],
            device=self.device
        )
        
        print(f"Loss function created successfully!")
        return loss_function
    
    def create_optimizer(self, model):
        """Create optimizer based on configuration."""
        print(f"Creating optimizer...")
        
        if self.config['OPTIMIZER'].lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['LEARNING_RATE'],
                weight_decay=self.config['WEIGHT_DECAY']
            )
        elif self.config['OPTIMIZER'].lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['LEARNING_RATE'],
                weight_decay=self.config['WEIGHT_DECAY'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['OPTIMIZER']}")
        
        print(f"Optimizer: {self.config['OPTIMIZER']}")
        return optimizer
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler based on configuration."""
        if self.config['SCHEDULER'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config['SCHEDULER_STEP_SIZE'],
                gamma=self.config['SCHEDULER_GAMMA']
            )
        elif self.config['SCHEDULER'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['NUM_EPOCHS']
            )
        elif self.config['SCHEDULER'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        else:
            scheduler = None
        
        print(f"Scheduler: {self.config['SCHEDULER']}")
        return scheduler
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_embedding_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            embeddings, classifications = model(images)
            loss, loss_components = criterion(embeddings, classifications, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            _, predicted = torch.max(classifications, 1)
            true_labels = torch.argmax(labels, 1)
            accuracy = (predicted == true_labels).float().mean().item()
            
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
                    'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'embedding_loss': total_embedding_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches
        }
    
    def validate_epoch(self, model, val_loader, criterion, epoch):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_embedding_loss = 0.0
        total_classification_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                embeddings, classifications = model(images)
                loss, loss_components = criterion(embeddings, classifications, labels)
                
                # Compute accuracy
                _, predicted = torch.max(classifications, 1)
                true_labels = torch.argmax(labels, 1)
                accuracy = (predicted == true_labels).float().mean().item()
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                total_embedding_loss += loss_components['embedding_loss'].item()
                total_classification_loss += loss_components['classification_loss'].item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'embedding_loss': total_embedding_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches
        }
    
    def train(self):
        """Main training loop."""
        print("\n🔥 Starting training...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Create loss function
        criterion = self.create_loss_function()
        
        # Create optimizer
        optimizer = self.create_optimizer(model)
        
        # Create scheduler
        scheduler = self.create_scheduler(optimizer)
        
        # Training loop
        for epoch in range(self.config['NUM_EPOCHS']):
            # Train
            train_metrics = self.train_epoch(model, self.train_loader, criterion, optimizer, epoch)
            
            # Validate
            if epoch % self.config['VALIDATION_FREQUENCY'] == 0:
                val_metrics = self.validate_epoch(model, self.val_loader, criterion, epoch)
                
                # Store metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_accuracies.append(train_metrics['accuracy'])
                self.val_accuracies.append(val_metrics['accuracy'])
                self.embedding_losses.append(train_metrics['embedding_loss'])
                self.classification_losses.append(train_metrics['classification_loss'])
                self.learning_rates.append(optimizer.param_groups[0]["lr"])
                
                # Update best metrics
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                
                # Print epoch summary
                print(f"\nEpoch {epoch+1}/{self.config['NUM_EPOCHS']}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Best Val Acc: {self.best_val_accuracy:.4f}")
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
        
        print("\n✅ Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'embedding_losses': self.embedding_losses,
            'classification_losses': self.classification_losses,
            'learning_rates': self.learning_rates,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss
        }

def main():
    """Main function to run hierarchical training."""
    print("🎯 HIERARCHICAL GALAXY CLASSIFICATION TRAINER")
    print("=" * 60)
    
    # Create trainer
    trainer = HierarchicalTrainer()
    
    # Run training
    results = trainer.train()
    
    print(f"\nTraining completed with best validation accuracy: {results['best_val_accuracy']:.4f}")
    print("Use the unified experiment tracking system to save and analyze these results!")

if __name__ == "__main__":
    main() 