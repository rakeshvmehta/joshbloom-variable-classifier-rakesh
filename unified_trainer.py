"""
Unified Trainer for Galaxy Classification

This script can train both hierarchical and non-hierarchical approaches based on
the global configuration in global_config.py.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50

# Add current directory to path
sys.path.append('.')

from global_config import config
from experiment_tracker import create_experiment_tracker
from data_processing.process_galaxy_dataset import get_data_loaders

# Import hierarchical modules
from hierarchy import create_hierarchical_model, create_loss_function

class CNNClassifier(nn.Module):
    """Basic CNN classifier for non-hierarchical approach."""
    
    def __init__(self, num_classes=37, dropout_rate=0.3):
        super(CNNClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
            nn.Sigmoid() if config.USE_SIGMOID else nn.Identity()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetClassifier(nn.Module):
    """ResNet-based classifier for non-hierarchical approach."""
    
    def __init__(self, num_classes=37, model_type='resnet18', pretrained=True, dropout_rate=0.3):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Modify the first convolutional layer to accept 3 input channels
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
            nn.Sigmoid() if config.USE_SIGMOID else nn.Identity()
        )
        
    def forward(self, x):
        return self.resnet(x)

def create_model(num_classes, model_type):
    """Create model based on configuration."""
    if config.APPROACH == 'hierarchical':
        return create_hierarchical_model(
            model_type=model_type,
            num_classes=num_classes,
            embedding_dim=config.EMBEDDING_DIM
        )
    else:
        if model_type.startswith('resnet'):
            return ResNetClassifier(
                num_classes=num_classes,
                model_type=model_type,
                pretrained=config.PRETRAINED,
                dropout_rate=config.DROPOUT_RATE
            )
        elif model_type == 'cnn':
            return CNNClassifier(
                num_classes=num_classes,
                dropout_rate=config.DROPOUT_RATE
            )
        else:
            raise ValueError(f"Unsupported model type for non-hierarchical: {model_type}")

def create_unified_loss_function():
    """Create loss function based on approach."""
    if config.APPROACH == 'hierarchical':
        # Create loss_config dictionary for hierarchical loss
        loss_config = {
            'classification_weight': config.CLASSIFICATION_WEIGHT,
            'loss_type': config.EMBEDDING_LOSS_TYPE,
            'device': config.get_device()
        }
        return create_loss_function(
            embedding_path=config.EMBEDDING_PATH,
            loss_config=loss_config
        )
    else:
        # For non-hierarchical, use standard loss
        if config.USE_SIGMOID:
            return nn.BCELoss()  # Binary Cross Entropy for multi-label
        else:
            return nn.CrossEntropyLoss()  # Cross Entropy for single-label

def create_optimizer(model):
    """Create optimizer based on configuration."""
    if config.OPTIMIZER.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

def create_scheduler(optimizer):
    """Create learning rate scheduler based on configuration."""
    if config.SCHEDULER == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_GAMMA
        )
    elif config.SCHEDULER == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS
        )
    elif config.SCHEDULER == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True
        )
    else:
        return None

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if config.APPROACH == 'hierarchical':
            embeddings, classifications = model(images)
            # Convert one-hot labels to class indices for hierarchical loss
            class_indices = torch.argmax(labels, dim=1)
            loss, embedding_loss, classification_loss = criterion(embeddings, classifications, class_indices)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Compute accuracy
        if config.APPROACH == 'hierarchical':
            _, predicted = torch.max(classifications, 1)
            true_labels = torch.argmax(labels, 1)
            accuracy = (predicted == true_labels).float().mean().item()
        else:
            if config.USE_SIGMOID:
                predicted = (outputs > 0.5).float()
                accuracy = (predicted == labels).float().mean().item()
            else:
                _, predicted = torch.max(outputs, 1)
                true_labels = torch.argmax(labels, 1)
                accuracy = (predicted == true_labels).float().mean().item()
        
        # Update metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # Update progress bar
        if batch_idx % config.LOG_FREQUENCY == 0:
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }

def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if config.APPROACH == 'hierarchical':
                embeddings, classifications = model(images)
                # Convert one-hot labels to class indices for hierarchical loss
                class_indices = torch.argmax(labels, dim=1)
                loss, embedding_loss, classification_loss = criterion(embeddings, classifications, class_indices)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Compute accuracy
            if config.APPROACH == 'hierarchical':
                _, predicted = torch.max(classifications, 1)
                true_labels = torch.argmax(labels, 1)
                accuracy = (predicted == true_labels).float().mean().item()
            else:
                if config.USE_SIGMOID:
                    predicted = (outputs > 0.5).float()
                    accuracy = (predicted == labels).float().mean().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    true_labels = torch.argmax(labels, 1)
                    accuracy = (predicted == true_labels).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }

def main():
    """Main training function."""
    print("UNIFIED GALAXY CLASSIFICATION TRAINER")
    print("=" * 60)
    
    # Print and validate configuration
    config.print_config()
    if not config.validate_config():
        print("❌ Configuration validation failed. Please fix the errors above.")
        return
    
    # Setup device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    data = get_data_loaders(
        image_dir=config.IMAGE_DIR,
        labels_file=config.LABELS_FILE,
        downsized_dir=config.DOWNSIZED_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_ratio=config.TRAIN_RATIO,
        seed=config.SEED,
        cache_size=config.CACHE_SIZE
    )
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    num_classes = len(data['class_names'])
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"\nCreating {config.APPROACH} model...")
    model = create_model(num_classes, config.MODEL_TYPE).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    print(f"\nCreating loss function...")
    criterion = create_unified_loss_function()
    
    # Create optimizer
    print(f"\nCreating optimizer...")
    optimizer = create_optimizer(model)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer)
    print(f"Optimizer: {config.OPTIMIZER}")
    print(f"Scheduler: {config.SCHEDULER}")
    
    # Create experiment tracker
    print(f"\nSetting up experiment tracking...")
    tracker = create_experiment_tracker(
        experiment_name=f"{config.APPROACH}_{config.MODEL_TYPE}",
        approach_type=config.APPROACH,
        custom_config=config.get_approach_config()
    )
    
    # Training loop
    print(f"\nStarting training...")
    print("=" * 60)
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        if epoch % config.VALIDATION_FREQUENCY == 0:
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
            
            # Log metrics to tracker
            current_lr = optimizer.param_groups[0]["lr"]
            tracker.log_epoch(epoch, val_metrics['loss'], val_metrics['loss'], 
                            train_metrics['accuracy'], val_metrics['accuracy'], current_lr)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < tracker.best_val_loss
            if epoch % config.CHECKPOINT_FREQUENCY == 0 or is_best:
                tracker.save_checkpoint(model, optimizer, epoch, is_best)
            
            # Update plots
            if epoch % config.PLOT_FREQUENCY == 0:
                tracker.plot_training_curves()
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
    
    # Finish experiment
    print("\nTraining completed!")
    tracker.finish_experiment()
    print(f"Results saved to: {tracker.experiment_dir}")

if __name__ == "__main__":
    # Import tqdm here to avoid circular imports
    from tqdm import tqdm
    main() 