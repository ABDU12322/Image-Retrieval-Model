"""
Loss functions for CLIP and SimCLR models.
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


class SimCLRLoss(nn.Module):
    """
    Contrastive loss for SimCLR model (image-to-image).
    
    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    Expects paired augmented views of the same images.
    
    Args:
        temperature (float): Scaling factor for logits. Default: 0.07.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Embeddings from first augmentation, shape (batch_size, embedding_dim)
            z_j: Embeddings from second augmentation, shape (batch_size, embedding_dim)
            
        Returns:
            loss: Scalar tensor representing the SimCLR loss
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Concatenate embeddings from both views
        # z: (2 * batch_size, embedding_dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        # sim: (2 * batch_size, 2 * batch_size)
        sim = torch.matmul(z, z.t()) / self.temperature
        
        # Create positive pairs mask
        # For each sample i, the positive pair is at index (i + batch_size) % (2 * batch_size)
        # and vice versa
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        mask = ~mask  # We want all non-diagonal elements
        
        # Get positive pairs
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
        
        # Remove diagonal from positive pairs (self-similarity)
        pos_mask = pos_mask & mask
        
        # NT-Xent loss
        # For each sample, compute loss over all negatives and one positive
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()  # For numerical stability
        
        # Exp of similarities
        exp_logits = torch.exp(logits)
        
        # Sum of exponentials for negatives (all positives pairs are also excluded from negatives)
        neg_mask = mask & ~pos_mask
        
        # Compute loss for each sample
        loss = 0.0
        for i in range(2 * batch_size):
            # Positive logits for sample i
            pos_logit = torch.sum(exp_logits[i][pos_mask[i]])
            
            # All logits (positive + negative) for sample i
            all_logit = torch.sum(exp_logits[i][mask[i]])
            
            # NT-Xent loss for sample i
            loss_i = -torch.log(pos_logit / all_logit + 1e-8)
            loss = loss + loss_i
        
        loss = loss / (2 * batch_size)
        
        return loss


class SimCLRLossSimplified(nn.Module):
    """
    Simplified version of SimCLR loss using PyTorch's built-in functions.
    
    This is more memory efficient and cleaner implementation.
    
    Args:
        temperature (float): Scaling factor for logits. Default: 0.07.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Embeddings from first augmentation, shape (batch_size, embedding_dim)
            z_j: Embeddings from second augmentation, shape (batch_size, embedding_dim)
            
        Returns:
            loss: Scalar tensor representing the SimCLR loss
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Concatenate representations: (2*batch_size, embedding_dim)
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix: (2*batch_size, 2*batch_size)
        similarity_matrix = torch.matmul(
            representations, representations.t()
        ) / self.temperature
        
        # Create labels for positive pairs
        # First batch_size samples' positive pair is in indices [batch_size, 2*batch_size)
        # Second batch_size samples' positive pair is in indices [0, batch_size)
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z_i.device),
                torch.arange(batch_size, device=z_i.device),
            ]
        )
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute logits and labels for cross-entropy
        # We want: similarity to positive pair should be highest among all other samples
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
