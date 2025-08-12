"""
Global Configuration for Hierarchical Galaxy Classification

This file contains all configurable hyperparameters and settings.
Modify values here to run different experiments.
"""

import torch

class GlobalConfig:
    """Global configuration class for all experiments."""
    
    # ===== Data Configuration =====
    IMAGE_DIR = '../training_images'
    LABELS_FILE = '../training_classifications.csv'
    DOWNSIZED_DIR = '../downsized_galaxy_images'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TRAIN_RATIO = 0.8
    CACHE_SIZE = 1000
    
    # ===== Model Configuration =====
    MODEL_TYPE = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50', 'custom'
    EMBEDDING_DIM = 128
    NUM_CLASSES = None  # Will be set automatically from dataset
    
    # ===== Training Configuration =====
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'  # 'adam', 'sgd'
    WEIGHT_DECAY = 1e-4
    SCHEDULER = 'step'  # 'step', 'cosine', 'plateau'
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1
    
    # ===== Loss Configuration =====
    EMBEDDING_LOSS_TYPE = 'inv_corr'  # 'inv_corr', 'squared_distance'
    CLASSIFICATION_WEIGHT = 0.1
    EMBEDDING_WEIGHT = 1.0
    
    # ===== Embedding Configuration =====
    EMBEDDING_PATH = 'galaxy_hierarchy_embeddings.unitsphere.pickle'
    
    # ===== Device Configuration =====
    DEVICE = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    
    # ===== Experiment Configuration =====
    EXPERIMENT_NAME_PREFIX = 'exp'
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    VALIDATION_FREQUENCY = 1  # Validate every N epochs
    CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N epochs
    
    # ===== Logging Configuration =====
    LOG_FREQUENCY = 10  # Print metrics every N batches
    PLOT_FREQUENCY = 5  # Update plots every N epochs
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device for training."""
        if cls.DEVICE == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(cls.DEVICE)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for saving."""
        config_dict = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                if not callable(value):
                    config_dict[attr] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        """Update config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 50)
        print("CURRENT CONFIGURATION")
        print("=" * 50)
        for key, value in cls.to_dict().items():
            print(f"{key}: {value}")
        print("=" * 50)

# Create a global instance
config = GlobalConfig() 