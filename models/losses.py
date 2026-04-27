"""
Loss functions for CLIP model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Contrastive loss for CLIP model (text-to-image and image-to-text).
    
    This loss uses symmetric cross-entropy loss between:
    - Text embeddings vs. image embeddings (image-to-text direction)
    - Image embeddings vs. text embeddings (text-to-image direction)
    
    Uses learnable temperature (logit_scale) from the model for consistency
    between training and inference.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, image_embeddings, text_embeddings, logit_scale=None):
        """
        Args:
            image_embeddings: Tensor of shape (batch_size, embedding_dim)
            text_embeddings: Tensor of shape (batch_size, embedding_dim)
            logit_scale: Learnable temperature parameter from CLIPModel
                        If None, uses default temperature 0.07
            
        Returns:
            loss: Scalar tensor representing the CLIP loss
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Use learnable logit_scale from model for consistency with inference
        # At inference, compute_similarity() uses logit_scale.exp()
        # At training, we should use the same scale
        if logit_scale is not None:
            temperature = logit_scale.exp()
        else:
            temperature = 1.0 / 0.07  # Default: inverse of 0.07
        
        # Compute similarity matrix: (batch_size, batch_size)
        # sim[i, j] = similarity between image_i and text_j
        logits = torch.matmul(
            image_embeddings, text_embeddings.t()
        ) * temperature
        
        # Create labels: diagonal elements should match
        # (image_i should match text_i)
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute cross-entropy loss for both directions
        # Image-to-text: predict which text matches each image
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image: predict which image matches each text
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
