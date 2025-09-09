"""
Hierarchical Galaxy Classification Package

This package contains modules for hierarchical galaxy classification including:
- hierarchical_galaxy_cnn: CNN architecture for hierarchical classification
- hierarchical_loss: Loss functions for hierarchical training
- galaxy_hierarchy: Galaxy hierarchy definitions
"""

# Import key functions for easier access
from .hierarchical_galaxy_cnn import create_hierarchical_model
from .hierarchical_loss import create_loss_function

__all__ = ['create_hierarchical_model', 'create_loss_function']
