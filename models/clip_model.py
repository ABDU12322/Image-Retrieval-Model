"""
CLIP Model for text-to-image retrieval.
"""

import torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder


class CLIPModel(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) model.
    
    Combines an image encoder and text encoder to learn joint embeddings
    for text-to-image and image-to-text retrieval.
    
    Args:
        image_embedding_dim (int): Dimension of image embeddings. Default: 512.
        text_embedding_dim (int): Dimension of text embeddings. Default: 512.
        vocab_size (int): Size of text vocabulary. Default: 10000.
        text_max_seq_length (int): Maximum text sequence length. Default: 77.
        num_text_layers (int): Number of transformer layers in text encoder. Default: 12.
        num_text_heads (int): Number of attention heads in text encoder. Default: 8.
        image_pretrained (bool): Use pretrained ImageNet weights. Default: True.
    """
    
    def __init__(
        self,
        image_embedding_dim: int = 512,
        text_embedding_dim: int = 512,
        vocab_size: int = 10000,
        text_max_seq_length: int = 77,
        num_text_layers: int = 12,
        num_text_heads: int = 8,
        image_pretrained: bool = True,
    ):
        super().__init__()
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            pretrained=image_pretrained,
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            max_seq_length=text_max_seq_length,
            num_layers=num_text_layers,
            num_heads=num_text_heads,
        )
        
        # Learnable temperature parameter for scaling logits
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.logit_scale.requires_grad = True
    
    def forward(self, images, text_tokens, text_mask=None):
        """
        Forward pass through CLIP model.
        
        Args:
            images: Image tensor of shape (batch_size, 3, height, width)
            text_tokens: Text token indices of shape (batch_size, max_seq_length)
            text_mask: Optional attention mask for text
            
        Returns:
            image_embeddings: Normalized image embeddings (batch_size, image_embedding_dim)
            text_embeddings: Normalized text embeddings (batch_size, text_embedding_dim)
        """
        # Encode images and text
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_tokens, mask=text_mask)
        
        return image_embeddings, text_embeddings
    
    def get_image_embeddings(self, images):
        """
        Get image embeddings.
        
        Args:
            images: Image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            image_embeddings: Tensor of shape (batch_size, image_embedding_dim)
        """
        return self.image_encoder(images)
    
    def get_text_embeddings(self, text_tokens, text_mask=None):
        """
        Get text embeddings.
        
        Args:
            text_tokens: Text token indices of shape (batch_size, max_seq_length)
            text_mask: Optional attention mask for text
            
        Returns:
            text_embeddings: Tensor of shape (batch_size, text_embedding_dim)
        """
        return self.text_encoder(text_tokens, mask=text_mask)
    
    def compute_similarity(self, image_embeddings, text_embeddings):
        """
        Compute similarity between image and text embeddings.
        
        Args:
            image_embeddings: Tensor of shape (num_images, embedding_dim)
            text_embeddings: Tensor of shape (num_texts, embedding_dim)
            
        Returns:
            similarity: Matrix of shape (num_images, num_texts) with similarity scores
        """
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) * self.logit_scale.exp()
        
        return similarity
