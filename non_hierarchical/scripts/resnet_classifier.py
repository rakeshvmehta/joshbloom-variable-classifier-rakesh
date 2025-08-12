import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append('..')
from data_processing.process_galaxy_dataset import get_data_loaders
from experiment_tracker import create_experiment_tracker
from torchvision.models import resnet18

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=37, pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 3 input channels
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, experiment_name=None, custom_config=None):
    # Create experiment tracker
    tracker = create_experiment_tracker(
        experiment_name=experiment_name,
        approach_type="non_hierarchical",
        custom_config=custom_config
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    print('\nTraining Progress:')
    print('-' * 30)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        train_pbar = tqdm(train_loader, desc=f'Training')
        
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy Calculation: Top-1 Class Comparison
            predicted_class = outputs.argmax(dim=1)
            true_class = labels.argmax(dim=1)
            correct_predictions += (predicted_class == true_class).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar with batch information
            batch_accuracy = (predicted_class == true_class).float().mean().item()
            train_pbar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_accuracy:.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        print(f'Validating...')
        val_pbar = tqdm(val_loader, desc=f'Validation')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Validation Accuracy: Top-1 Class Comparison
                predicted_class = outputs.argmax(dim=1)
                true_class = labels.argmax(dim=1)
                val_correct += (predicted_class == true_class).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar with batch information
                batch_accuracy = (predicted_class == true_class).float().mean().item()
                val_pbar.set_postfix({
                    'batch': f'{batch_idx+1}/{len(val_loader)}',
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_accuracy:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Log metrics to experiment tracker
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.log_epoch(epoch, avg_train_loss, avg_val_loss, train_accuracy, val_accuracy, current_lr)
        
        # Save checkpoint
        is_best = avg_val_loss < tracker.best_val_loss
        tracker.save_checkpoint(model, optimizer, epoch, is_best)
        
        print('-' * 30)
    
    # Finish experiment
    tracker.finish_experiment()
    
    return tracker

def main():
    # Set device
    device = torch.device('cpu')

    # Check for MPS (Metal Performance Shaders) on Mac first, then CUDA, then fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001  # Lower learning rate for ResNet
    num_epochs = 20
    
    # Custom configuration for experiment tracking
    custom_config = {
        'BATCH_SIZE': batch_size,
        'LEARNING_RATE': learning_rate,
        'NUM_EPOCHS': num_epochs,
        'OPTIMIZER': 'adam',
        'SCHEDULER': 'plateau',
        'DROPOUT_RATE': 0.3,
        'PRETRAINED': True
    }
    
    print(f'\nHyperparameters:')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate}')
    print(f'Number of epochs: {num_epochs}')
    
    # Get data loaders
    print('\nLoading and preparing dataset...')
    data = get_data_loaders(
        image_dir="../training_images",
        labels_file="../training_classifications.csv",
        downsized_dir="../downsized_galaxy_images",
        batch_size=batch_size,
        num_workers=4,
        train_ratio=0.8,
        seed=42,
        cache_size=1000
    )
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    print(f'Dataset loaded successfully!')
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')
    print(f'Total training samples: {len(train_loader.dataset)}')
    print(f'Total validation samples: {len(val_loader.dataset)}')
    
    # Create model
    print('\nInitializing ResNet-18 model...')
    model = ResNetClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model created successfully!')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function and optimizer
    print('\nSetting up loss function and optimizer...')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f'Using MSE Loss and Adam optimizer')
   
    # Train the model with experiment tracking
    print('\nStarting training with experiment tracking...')
    print('=' * 50)
    tracker = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, experiment_name="resnet_baseline", custom_config=custom_config
    )
    print('=' * 50)
    print('Training completed!')
    print(f'Results saved to: {tracker.experiment_dir}')

if __name__ == '__main__':
    main()