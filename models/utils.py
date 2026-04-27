"""
Utility functions for image retrieval models.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def get_image_transforms(image_size: int = 224, augmentation: bool = False):
    """
    Get image preprocessing transforms.
    
    Args:
        image_size (int): Size to resize images to. Default: 224.
        augmentation (bool): Whether to apply data augmentation. Default: False.
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    if augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit vectors.
    
    Args:
        embeddings: Tensor of shape (..., embedding_dim)
        
    Returns:
        normalized: Normalized embeddings
    """
    return torch.nn.functional.normalize(embeddings, dim=-1)


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: Tensor of shape (n, embedding_dim)
        embeddings2: Tensor of shape (m, embedding_dim)
        
    Returns:
        similarity: Tensor of shape (n, m) with cosine similarities
    """
    embeddings1 = normalize_embeddings(embeddings1)
    embeddings2 = normalize_embeddings(embeddings2)
    return torch.matmul(embeddings1, embeddings2.t())


def get_device():
    """
    Get the available device (GPU or CPU).
    
    Returns:
        torch.device: Device to use for training/inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(encoder):
    """
    Freeze encoder parameters (set requires_grad=False).
    
    Args:
        encoder: PyTorch module to freeze
    """
    for param in encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(encoder):
    """
    Unfreeze encoder parameters (set requires_grad=True).
    
    Args:
        encoder: PyTorch module to unfreeze
    """
    for param in encoder.parameters():
        param.requires_grad = True


def tokenize_text(texts, max_length=77):
    """
    Tokenize text captions using BERT tokenizer.
    
    Args:
        texts: String or list of strings to tokenize
        max_length: Maximum sequence length. Default: 77.
        
    Returns:
        dict: Dictionary with 'input_ids' and 'attention_mask' tensors
    """
    from transformers import BertTokenizer
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Handle single string
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize texts
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

