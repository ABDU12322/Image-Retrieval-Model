"""
Image and Text Encoders for CLIP model.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer


class ImageEncoder(nn.Module):
    """
    Image encoder using ResNet-50 as backbone.
    Projects images to a fixed-dimensional embedding space.
    
    Args:
        embedding_dim (int): Dimension of the output embedding. Default: 512.
        pretrained (bool): Whether to use pretrained ResNet-50. Default: True.
    """
    
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get the feature dimension from ResNet-50 (2048)
        self.feature_dim = 2048
        
        # Projection head to map features to embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, embedding_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Extract features using backbone
        features = self.backbone(x)  # (batch_size, 2048, 1, 1)
        
        # Flatten
        features = torch.flatten(features, 1)  # (batch_size, 2048)
        
        # Project to embedding space
        embeddings = self.projection(features)  # (batch_size, embedding_dim)
        
        return embeddings


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained BERT for better semantic understanding.
    
    Uses the pretrained "bert-base-uncased" model, which has been trained on 
    a large corpus and provides strong semantic representations for text.
    
    Args:
        embedding_dim (int): Dimension of output embeddings. Default: 512.
        max_seq_length (int): Maximum sequence length. Default: 77.
        vocab_size (int): Size of vocabulary (for compatibility, not used). Default: 10000.
        num_layers (int): Number of layers (for compatibility, not used). Default: 12.
        num_heads (int): Number of heads (for compatibility, not used). Default: 8.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        max_seq_length: int = 77,
        vocab_size: int = 10000,
        num_layers: int = 12,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Load pretrained BERT model
        # bert-base-uncased provides strong semantic understanding
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        
        # BERT outputs 768-dimensional vectors
        bert_output_dim = 768
        
        # Trainable projection head to map BERT output to target embedding dimension
        self.projection = nn.Linear(bert_output_dim, embedding_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices of shape (batch_size, seq_length)
            mask: Optional attention mask (1 for valid tokens, 0 for padding)
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Get outputs from pretrained BERT model
        outputs = self.text_model(
            input_ids=x,
            attention_mask=mask,
            return_dict=True,
        )
        
        # Use the [CLS] token output (first token) as the sentence representation
        # This is the standard approach for BERT for classification-like tasks
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Project to target embedding dimension with trainable projection head
        embeddings = self.projection(cls_output)  # (batch_size, embedding_dim)
        
        return embeddings
