"""
SimCLR Model for image-to-image retrieval and self-supervised learning.
"""

import torch
import torch.nn as nn
from .encoders import ImageEncoder


class SimCLRModel(nn.Module):
    """
    SimCLR (Simple Framework for Contrastive Learning of Visual Representations) model.
    
    Uses contrastive learning on augmented image pairs to learn useful image representations.
    The architecture consists of:
    1. Backbone encoder (ResNet-50) for feature extraction
    2. Non-linear projection head for contrastive loss
    
    Args:
        embedding_dim (int): Dimension of projection head output. Default: 512.
        projection_dim (int): Dimension after projection (typically smaller for efficiency). Default: 128.
        image_pretrained (bool): Use pretrained ImageNet weights. Default: True.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        image_pretrained: bool = True,
    ):
        super().__init__()
        
        # Image encoder (backbone)
        self.encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=image_pretrained,
        )
        
        # Projection head for contrastive loss
        # Maps from embedding_dim to projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
    
    def forward(self, x_i, x_j):
        """
        Forward pass with two augmented views of the same images.
        
        Args:
            x_i: First augmentation of images, shape (batch_size, 3, height, width)
            x_j: Second augmentation of images, shape (batch_size, 3, height, width)
            
        Returns:
            z_i: Projections from first view, shape (batch_size, projection_dim)
            z_j: Projections from second view, shape (batch_size, projection_dim)
        """
        # Get embeddings from encoder
        h_i = self.encoder(x_i)  # (batch_size, embedding_dim)
        h_j = self.encoder(x_j)  # (batch_size, embedding_dim)
        
        # Project to contrastive loss space
        z_i = self.projection_head(h_i)  # (batch_size, projection_dim)
        z_j = self.projection_head(h_j)  # (batch_size, projection_dim)
        
        return z_i, z_j
    
    def get_embeddings(self, x):
        """
        Get image embeddings (from encoder, before projection head).
        Useful for downstream tasks like retrieval or clustering.
        
        Args:
            x: Image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def get_projected_embeddings(self, x):
        """
        Get projected embeddings (after projection head).
        
        Args:
            x: Image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            projected_embeddings: Tensor of shape (batch_size, projection_dim)
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
