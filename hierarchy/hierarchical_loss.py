"""
Hierarchical Loss Functions for Galaxy Classification

This module implements hierarchical loss functions that combine:
1. Embedding loss - measures semantic similarity in embedding space
2. Classification loss - standard cross-entropy for classification

Based on Sydney's variable classifier approach but adapted for galaxy morphology classification.
The key insight is that embedding loss enforces hierarchical relationships while 
classification loss maintains accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from pathlib import Path


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss combining embedding and classification losses.
    
    Loss = embedding_loss + classification_weight * classification_loss
    
    Where:
    - embedding_loss enforces semantic similarity based on hierarchy
    - classification_loss maintains classification accuracy
    """
    
    def __init__(self, embedding_path, classification_weight=0.1, loss_type='inv_corr', device='cpu'):
        """
        Initialize the hierarchical loss function.
        
        Args:
            embedding_path: Path to the unitsphere embedding pickle file
            classification_weight: Weight for classification loss (vs embedding loss)
            loss_type: Type of embedding loss ('inv_corr', 'squared_distance')
            device: Device to run computations on
        """
        super(HierarchicalLoss, self).__init__()
        
        self.classification_weight = classification_weight
        self.loss_type = loss_type
        self.device = device
        
        # Load class embeddings
        self._load_class_embeddings(embedding_path)
        
        # Initialize loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        
    def _load_class_embeddings(self, embedding_path):
        """Load the precomputed class embeddings from pickle file."""
        
        embedding_path = Path(embedding_path)
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        print(f"Loading class embeddings from: {embedding_path}")
        
        with open(embedding_path, 'rb') as f:
            embedding_data = pickle.load(f)
        
        # Extract embeddings and convert to tensor
        self.class_embeddings = torch.tensor(
            embedding_data['embedding'], 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Store metadata
        self.num_classes = self.class_embeddings.shape[0]
        self.embedding_dim = self.class_embeddings.shape[1]
        self.class_names = embedding_data.get('class_names', [])
        
        print(f"Loaded embeddings: {self.class_embeddings.shape}")
        print(f"Classes: {self.num_classes}, Embedding dim: {self.embedding_dim}")
        
    def inverse_correlation_loss(self, pred_embeddings, target_embeddings):
        """
        Inverse correlation loss: 1 - cosine similarity
        
        This is Sydney's preferred embedding loss. It encourages predicted embeddings
        to be similar (high cosine similarity) to target embeddings.
        
        Args:
            pred_embeddings: Predicted embeddings [batch_size, embedding_dim]
            target_embeddings: Target embeddings [batch_size, embedding_dim]
            
        Returns:
            Loss tensor [batch_size]
        """
        # Compute cosine similarity (dot product since embeddings are L2-normalized)
        cosine_sim = torch.sum(pred_embeddings * target_embeddings, dim=1)
        
        # Inverse correlation: 1 - cosine_similarity
        loss = 1.0 - cosine_sim
        
        return loss
    
    def squared_distance_loss(self, pred_embeddings, target_embeddings):
        """
        Squared Euclidean distance loss.
        
        Alternative embedding loss that directly minimizes distance in embedding space.
        
        Args:
            pred_embeddings: Predicted embeddings [batch_size, embedding_dim]
            target_embeddings: Target embeddings [batch_size, embedding_dim]
            
        Returns:
            Loss tensor [batch_size]
        """
        # Compute squared Euclidean distance
        diff = pred_embeddings - target_embeddings
        loss = torch.sum(diff ** 2, dim=1)
        
        return loss
    
    def get_target_embeddings(self, class_indices):
        """
        Get target embeddings for given class indices.
        
        Args:
            class_indices: Class indices tensor [batch_size]
            
        Returns:
            Target embeddings [batch_size, embedding_dim]
        """
        # Ensure class_embeddings is on the same device as class_indices
        if self.class_embeddings.device != class_indices.device:
            self.class_embeddings = self.class_embeddings.to(class_indices.device)
        
        # Index into class embeddings
        target_embeddings = self.class_embeddings[class_indices]
        
        return target_embeddings
    
    def forward(self, pred_embeddings, pred_classifications, true_labels):
        """
        Compute the complete hierarchical loss.
        
        Args:
            pred_embeddings: Predicted embeddings [batch_size, embedding_dim]
            pred_classifications: Predicted classification logits [batch_size, num_classes]
            true_labels: True class indices [batch_size] (NOT one-hot)
            
        Returns:
            total_loss: Combined hierarchical loss
            embedding_loss: Embedding loss component
            classification_loss: Classification loss component
        """
        
        # Get target embeddings for the true classes
        target_embeddings = self.get_target_embeddings(true_labels)
        
        # Compute embedding loss
        if self.loss_type == 'inv_corr':
            embedding_losses = self.inverse_correlation_loss(pred_embeddings, target_embeddings)
        elif self.loss_type == 'squared_distance':
            embedding_losses = self.squared_distance_loss(pred_embeddings, target_embeddings)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Average embedding loss over batch
        embedding_loss = torch.mean(embedding_losses)
        
        # Compute classification loss (standard cross-entropy)
        classification_loss = self.classification_loss_fn(pred_classifications, true_labels)
        
        # Combine losses
        total_loss = embedding_loss + self.classification_weight * classification_loss
        
        return total_loss, embedding_loss, classification_loss


class AdaptiveHierarchicalLoss(HierarchicalLoss):
    """
    Adaptive hierarchical loss that adjusts weights during training.
    
    Starts with higher embedding weight early in training to learn good representations,
    then gradually increases classification weight for better accuracy.
    """
    
    def __init__(self, embedding_path, initial_classification_weight=0.01, 
                 final_classification_weight=0.5, adaptation_epochs=50, **kwargs):
        """
        Initialize adaptive hierarchical loss.
        
        Args:
            embedding_path: Path to embedding file
            initial_classification_weight: Starting weight for classification loss
            final_classification_weight: Final weight for classification loss  
            adaptation_epochs: Number of epochs over which to adapt weights
            **kwargs: Additional arguments for parent class
        """
        super().__init__(embedding_path, classification_weight=initial_classification_weight, **kwargs)
        
        self.initial_weight = initial_classification_weight
        self.final_weight = final_classification_weight
        self.adaptation_epochs = adaptation_epochs
        self.current_epoch = 0
        
    def update_epoch(self, epoch):
        """Update the current epoch and adjust weights accordingly."""
        self.current_epoch = epoch
        
        if epoch < self.adaptation_epochs:
            # Linear interpolation between initial and final weights
            progress = epoch / self.adaptation_epochs
            self.classification_weight = (
                self.initial_weight + 
                progress * (self.final_weight - self.initial_weight)
            )
        else:
            self.classification_weight = self.final_weight


def test_hierarchical_loss():
    """Test function to verify the hierarchical loss works correctly."""
    
    print("Testing Hierarchical Loss Functions...")
    print("=" * 50)
    
    # Test parameters
    batch_size = 8
    num_classes = 37
    embedding_dim = 37
    
    # Create dummy data
    pred_embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    pred_classifications = torch.randn(batch_size, num_classes)
    true_labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Test shapes:")
    print(f"  pred_embeddings: {pred_embeddings.shape}")
    print(f"  pred_classifications: {pred_classifications.shape}")
    print(f"  true_labels: {true_labels.shape}")
    
    # Test hierarchical loss
    try:
        embedding_path = "galaxy_hierarchy_embeddings.unitsphere.pickle"
        hierarchical_loss = HierarchicalLoss(embedding_path, classification_weight=0.1)
        
        total_loss, emb_loss, cls_loss = hierarchical_loss(
            pred_embeddings, pred_classifications, true_labels
        )
        
        print(f"\n✅ HierarchicalLoss test passed!")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Embedding loss: {emb_loss.item():.4f}")
        print(f"  Classification loss: {cls_loss.item():.4f}")
        print(f"  Loss ratio (emb:cls): {emb_loss.item():.3f}:{cls_loss.item():.3f}")
        
    except FileNotFoundError:
        print("⚠️  Warning: Embedding file not found. Please run generate_galaxy_embeddings.py first")
        return False
    
    # Test adaptive loss
    try:
        adaptive_loss = AdaptiveHierarchicalLoss(
            embedding_path, 
            initial_classification_weight=0.01,
            final_classification_weight=0.5,
            adaptation_epochs=10
        )
        
        print(f"\n✅ AdaptiveHierarchicalLoss test:")
        print(f"  Initial weight: {adaptive_loss.classification_weight:.3f}")
        
        # Simulate training epochs
        for epoch in [0, 5, 10, 20]:
            adaptive_loss.update_epoch(epoch)
            print(f"  Epoch {epoch:2d}: weight = {adaptive_loss.classification_weight:.3f}")
            
    except FileNotFoundError:
        print("⚠️  Warning: Could not test adaptive loss (embedding file missing)")
    
    print("\n" + "=" * 50)
    print("✅ Loss function tests completed!")
    
    return True


def create_loss_function(embedding_path="galaxy_hierarchy_embeddings.unitsphere.pickle", 
                        loss_config=None):
    """
    Factory function to create hierarchical loss functions.
    
    Args:
        embedding_path: Path to the embedding file
        loss_config: Dictionary with loss configuration
        
    Returns:
        Configured loss function
    """
    if loss_config is None:
        loss_config = {
            'classification_weight': 0.1,
            'loss_type': 'inv_corr'
        }
    
    if loss_config.get('adaptive', False):
        return AdaptiveHierarchicalLoss(embedding_path, **loss_config)
    else:
        return HierarchicalLoss(embedding_path, **loss_config)


if __name__ == "__main__":
    test_hierarchical_loss() 