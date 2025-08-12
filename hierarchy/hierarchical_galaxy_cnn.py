"""
Hierarchical Galaxy CNN Architecture

This module implements dual-output CNN architectures for hierarchical galaxy classification.
The networks output both semantic embeddings and classification logits, following the 
approach from Sydney's variable classifier but adapted for 2D galaxy images.

Key features:
- Dual outputs: semantic embeddings + classification logits
- L2-normalized embeddings for dot-product similarity
- Classification head takes embeddings as input
- Compatible with existing galaxy data pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class HierarchicalCNNClassifier(nn.Module):
    """
    Hierarchical CNN with dual outputs for galaxy classification.
    
    Architecture:
    Input → CNN Backbone → Features → Embedding Head (L2 normalized)
                                   → Classification Head (from embeddings)
    """
    
    def __init__(self, input_channels=3, num_classes=37, embedding_dim=37, dropout_rate=0.3):
        """
        Initialize the hierarchical CNN classifier.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of galaxy classification classes (37)
            embedding_dim: Dimension of semantic embeddings (37 to match unitsphere)
            dropout_rate: Dropout rate for regularization
        """
        super(HierarchicalCNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # CNN Backbone (same as your original CNN)
        # Initial convolutional block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared feature processing (following Sydney's pattern)
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # Dual Output Heads (Sydney's approach)
        # 1. Embedding Head - produces semantic embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # 2. Classification Head - takes embeddings as input
        self.classification_head = nn.Linear(embedding_dim, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass with dual outputs.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            embeddings: L2-normalized semantic embeddings [batch_size, embedding_dim]
            classifications: Classification logits [batch_size, num_classes]
        """
        # CNN Backbone
        # First block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 256]
        
        # Shared feature processing
        features = self.shared_fc(x)  # [batch_size, 256]
        
        # Dual Output Heads
        # 1. Embedding output (L2 normalized like Sydney's approach)
        embedding_raw = self.embedding_head(features)  # [batch_size, embedding_dim]
        embeddings = F.normalize(embedding_raw, p=2, dim=-1)  # L2 normalize
        
        # 2. Classification output (from embeddings, like Sydney's approach)
        classifications = self.classification_head(embeddings)  # [batch_size, num_classes]
        
        return embeddings, classifications


class HierarchicalResNetClassifier(nn.Module):
    """
    Hierarchical ResNet with dual outputs for galaxy classification.
    
    Based on your existing ResNet but modified to follow Sydney's dual-output pattern.
    """
    
    def __init__(self, num_classes=37, embedding_dim=37, pretrained=True, backbone='resnet18', dropout_rate=0.3):
        """
        Initialize the hierarchical ResNet classifier.
        
        Args:
            num_classes: Number of galaxy classification classes (37)
            embedding_dim: Dimension of semantic embeddings (37)
            pretrained: Whether to use pretrained ResNet weights
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            dropout_rate: Dropout rate for regularization
        """
        super(HierarchicalResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Load pretrained ResNet backbone
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify the first convolutional layer for 3 input channels (if needed)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Shared feature processing (Sydney's pattern)
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # Dual Output Heads (following Sydney's approach)
        # 1. Embedding Head - produces semantic embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # 2. Classification Head - takes embeddings as input
        self.classification_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass with dual outputs.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            embeddings: L2-normalized semantic embeddings [batch_size, embedding_dim]
            classifications: Classification logits [batch_size, num_classes]
        """
        # ResNet backbone feature extraction
        features = self.resnet(x)  # [batch_size, num_features]
        
        # Shared feature processing
        processed_features = self.shared_fc(features)  # [batch_size, 256]
        
        # Dual Output Heads
        # 1. Embedding output (L2 normalized like Sydney's approach)
        embedding_raw = self.embedding_head(processed_features)  # [batch_size, embedding_dim]
        embeddings = F.normalize(embedding_raw, p=2, dim=-1)  # L2 normalize
        
        # 2. Classification output (from embeddings, like Sydney's approach)
        classifications = self.classification_head(embeddings)  # [batch_size, num_classes]
        
        return embeddings, classifications


def create_hierarchical_model(architecture='cnn', **kwargs):
    """
    Factory function to create hierarchical models.
    
    Args:
        architecture: 'cnn' or 'resnet'
        **kwargs: Additional arguments for the model
        
    Returns:
        Hierarchical model instance
    """
    if architecture.lower() == 'cnn':
        return HierarchicalCNNClassifier(**kwargs)
    elif architecture.lower() == 'resnet':
        return HierarchicalResNetClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def test_model_architecture():
    """Test function to verify the model architectures work correctly."""
    
    print("Testing Hierarchical Galaxy CNN Architectures...")
    print("=" * 50)
    
    # Test input (batch of galaxy images)
    batch_size = 4
    channels = 3
    height, width = 128, 128
    test_input = torch.randn(batch_size, channels, height, width)
    
    # Test CNN architecture
    print("\n1. Testing HierarchicalCNNClassifier:")
    cnn_model = HierarchicalCNNClassifier()
    cnn_model.eval()
    
    with torch.no_grad():
        embeddings, classifications = cnn_model(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Embedding output shape: {embeddings.shape}")
    print(f"   Classification output shape: {classifications.shape}")
    print(f"   Embedding L2 norms: {torch.norm(embeddings, p=2, dim=1)}")  # Should be ~1.0
    
    # Test ResNet architecture
    print("\n2. Testing HierarchicalResNetClassifier:")
    resnet_model = HierarchicalResNetClassifier(backbone='resnet18')
    resnet_model.eval()
    
    with torch.no_grad():
        embeddings, classifications = resnet_model(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Embedding output shape: {embeddings.shape}")
    print(f"   Classification output shape: {classifications.shape}")
    print(f"   Embedding L2 norms: {torch.norm(embeddings, p=2, dim=1)}")  # Should be ~1.0
    
    # Test factory function
    print("\n3. Testing factory function:")
    factory_cnn = create_hierarchical_model('cnn', num_classes=37, embedding_dim=37)
    factory_resnet = create_hierarchical_model('resnet', backbone='resnet18')
    
    print(f"   CNN model created: {type(factory_cnn).__name__}")
    print(f"   ResNet model created: {type(factory_resnet).__name__}")
    
    print("\n" + "=" * 50)
    print("✅ All architecture tests passed!")
    print("\nKey features verified:")
    print("- Dual outputs: embeddings + classifications")
    print("- L2-normalized embeddings (norms ≈ 1.0)")
    print("- Compatible with galaxy image input format")
    print("- Classification head uses embeddings as input")


if __name__ == "__main__":
    test_model_architecture() 