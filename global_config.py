"""
Global Configuration for Galaxy Classification Research

This file contains all configurable parameters for both hierarchical and non-hierarchical approaches.
Modify values here to control which approach to use and set all hyperparameters.
"""

import torch
from pathlib import Path

class GlobalConfig:
    """Global configuration class for all experiments."""
    
    # ===== APPROACH SELECTION =====
    APPROACH = 'hierarchical'  # 'hierarchical' or 'non_hierarchical'
    MODEL_TYPE = 'resnet18'    # 'resnet18', 'resnet34', 'resnet50', 'cnn', 'custom'
    
    # ===== DATA CONFIGURATION =====
    IMAGE_DIR = 'training_images'
    LABELS_FILE = 'training_classifications.csv'
    DOWNSIZED_DIR = 'downsized_galaxy_images'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TRAIN_RATIO = 0.8
    CACHE_SIZE = 1000
    SEED = 42
    
    # ===== HIERARCHICAL CONFIGURATION =====
    # Only used when APPROACH = 'hierarchical'
    EMBEDDING_DIM = 128
    EMBEDDING_PATH = 'hierarchy/galaxy_hierarchy_embeddings.unitsphere.pickle'
    EMBEDDING_LOSS_TYPE = 'inv_corr'  # 'inv_corr', 'squared_distance'
    CLASSIFICATION_WEIGHT = 0.1
    EMBEDDING_WEIGHT = 1.0
    
    # ===== NON-HIERARCHICAL CONFIGURATION =====
    # Only used when APPROACH = 'non_hierarchical'
    DROPOUT_RATE = 0.3
    PRETRAINED = True
    USE_SIGMOID = True  # For multi-label classification
    
    # ===== TRAINING CONFIGURATION =====
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'  # 'adam', 'sgd'
    WEIGHT_DECAY = 1e-4
    SCHEDULER = 'step'  # 'step', 'cosine', 'plateau'
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    
    # ===== EXPERIMENT CONFIGURATION =====
    EXPERIMENT_NAME_PREFIX = 'exp'
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    VALIDATION_FREQUENCY = 1  # Validate every N epochs
    CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N epochs
    
    # ===== LOGGING CONFIGURATION =====
    LOG_FREQUENCY = 10  # Print metrics every N batches
    PLOT_FREQUENCY = 5  # Update plots every N epochs
    
    # ===== ADVANCED CONFIGURATION =====
    GRADIENT_CLIP = 1.0
    LABEL_SMOOTHING = 0.0
    MIXUP_ALPHA = 0.0  # Set to > 0 to enable mixup augmentation
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device for training."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    @classmethod
    def get_approach_config(cls):
        """Get configuration specific to the selected approach."""
        if cls.APPROACH == 'hierarchical':
            return {
                'APPROACH_TYPE': 'hierarchical',
                'MODEL_TYPE': cls.MODEL_TYPE,
                'EMBEDDING_DIM': cls.EMBEDDING_DIM,
                'NUM_EPOCHS': cls.NUM_EPOCHS,
                'LEARNING_RATE': cls.LEARNING_RATE,
                'BATCH_SIZE': cls.BATCH_SIZE,
                'OPTIMIZER': cls.OPTIMIZER,
                'WEIGHT_DECAY': cls.WEIGHT_DECAY,
                'SCHEDULER': cls.SCHEDULER,
                'SCHEDULER_STEP_SIZE': cls.SCHEDULER_STEP_SIZE,
                'SCHEDULER_GAMMA': cls.SCHEDULER_GAMMA,
                'EMBEDDING_LOSS_TYPE': cls.EMBEDDING_LOSS_TYPE,
                'CLASSIFICATION_WEIGHT': cls.CLASSIFICATION_WEIGHT,
                'EMBEDDING_WEIGHT': cls.EMBEDDING_WEIGHT,
                'EMBEDDING_PATH': cls.EMBEDDING_PATH
            }
        else:
            return {
                'APPROACH_TYPE': 'non_hierarchical',
                'MODEL_TYPE': cls.MODEL_TYPE,
                'NUM_EPOCHS': cls.NUM_EPOCHS,
                'LEARNING_RATE': cls.LEARNING_RATE,
                'BATCH_SIZE': cls.BATCH_SIZE,
                'OPTIMIZER': cls.OPTIMIZER,
                'WEIGHT_DECAY': cls.WEIGHT_DECAY,
                'SCHEDULER': cls.SCHEDULER,
                'SCHEDULER_PATIENCE': cls.SCHEDULER_PATIENCE,
                'SCHEDULER_FACTOR': cls.SCHEDULER_FACTOR,
                'DROPOUT_RATE': cls.DROPOUT_RATE,
                'PRETRAINED': cls.PRETRAINED,
                'USE_SIGMOID': cls.USE_SIGMOID
            }
    
    @classmethod
    def get_data_config(cls):
        """Get data configuration."""
        return {
            'IMAGE_DIR': cls.IMAGE_DIR,
            'LABELS_FILE': cls.LABELS_FILE,
            'DOWNSIZED_DIR': cls.DOWNSIZED_DIR,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'NUM_WORKERS': cls.NUM_WORKERS,
            'TRAIN_RATIO': cls.TRAIN_RATIO,
            'CACHE_SIZE': cls.CACHE_SIZE,
            'SEED': cls.SEED
        }
    
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
        print("=" * 60)
        print("GLOBAL CONFIGURATION")
        print("=" * 60)
        print(f"APPROACH: {cls.APPROACH}")
        print(f"MODEL TYPE: {cls.MODEL_TYPE}")
        print(f"DEVICE: {cls.get_device()}")
        print("-" * 60)
        
        # Print approach-specific config
        approach_config = cls.get_approach_config()
        print(f"APPROACH-SPECIFIC CONFIG:")
        for key, value in approach_config.items():
            print(f"  {key}: {value}")
        
        print("-" * 60)
        print(f"DATA CONFIG:")
        data_config = cls.get_data_config()
        for key, value in data_config.items():
            print(f"  {key}: {value}")
        
        print("-" * 60)
        print(f"TRAINING CONFIG:")
        print(f"  NUM_EPOCHS: {cls.NUM_EPOCHS}")
        print(f"  LEARNING_RATE: {cls.LEARNING_RATE}")
        print(f"  BATCH_SIZE: {cls.BATCH_SIZE}")
        print(f"  OPTIMIZER: {cls.OPTIMIZER}")
        print(f"  SCHEDULER: {cls.SCHEDULER}")
        print("=" * 60)
    
    @classmethod
    def validate_config(cls):
        """Validate the current configuration."""
        errors = []
        
        # Check approach
        if cls.APPROACH not in ['hierarchical', 'non_hierarchical']:
            errors.append(f"Invalid APPROACH: {cls.APPROACH}. Must be 'hierarchical' or 'non_hierarchical'")
        
        # Check model type
        valid_models = ['resnet18', 'resnet34', 'resnet50', 'cnn', 'custom']
        if cls.MODEL_TYPE not in valid_models:
            errors.append(f"Invalid MODEL_TYPE: {cls.MODEL_TYPE}. Must be one of {valid_models}")
        
        # Check paths
        if not Path(cls.IMAGE_DIR).exists():
            errors.append(f"IMAGE_DIR does not exist: {cls.IMAGE_DIR}")
        if not Path(cls.LABELS_FILE).exists():
            errors.append(f"LABELS_FILE does not exist: {cls.LABELS_FILE}")
        
        # Check hierarchical-specific paths
        if cls.APPROACH == 'hierarchical':
            if not Path(cls.EMBEDDING_PATH).exists():
                errors.append(f"EMBEDDING_PATH does not exist: {cls.EMBEDDING_PATH}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  ❌ {error}")
            return False
        
        print("✅ Configuration validation passed!")
        return True

# Create a global instance
config = GlobalConfig()

if __name__ == "__main__":
    # Print and validate config
    config.print_config()
    config.validate_config() 