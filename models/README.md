# Image Retrieval Models

This directory contains implementations of two image retrieval models:

1. **CLIP** (Contrastive Language-Image Pre-training) - for text-to-image retrieval
2. **SimCLR** (Simple Framework for Contrastive Learning) - for image-to-image self-supervised learning

## Architecture Overview

### CLIP Model

CLIP learns joint embeddings of images and text through contrastive learning. It consists of:

- **ImageEncoder**: ResNet-50 backbone with projection head to embed images
- **TextEncoder**: Transformer-based encoder to embed text
- **CLIPLoss**: Symmetric cross-entropy loss that matches images with their corresponding captions

**Use Case**: Retrieve images based on text queries (e.g., "a dog playing fetch")

### SimCLR Model

SimCLR learns image representations through self-supervised contrastive learning using augmented image pairs. It consists of:

- **ImageEncoder**: ResNet-50 backbone with projection head
- **Projection Head**: Non-linear head that projects embeddings to contrastive space
- **SimCLRLoss**: NT-Xent (Normalized Temperature-scaled Cross Entropy) loss

**Use Case**: Learn useful image representations without labeled data, useful for downstream tasks like classification or retrieval

## Files Structure

```
models/
├── __init__.py              # Package initialization
├── encoders.py              # ImageEncoder and TextEncoder implementations
├── losses.py                # CLIPLoss and SimCLRLoss implementations
├── clip_model.py            # Full CLIP model
├── simclr_model.py          # Full SimCLR model
├── utils.py                 # Utility functions (transforms, helpers)
└── README.md                # This file
```

## Component Details

### Encoders (encoders.py)

#### ImageEncoder
- Uses ResNet-50 backbone (pretrained on ImageNet by default)
- Projects 2048-dimensional features to configurable embedding dimension
- Input: Images of shape `(batch_size, 3, height, width)`
- Output: Embeddings of shape `(batch_size, embedding_dim)`

```python
from models import ImageEncoder

image_encoder = ImageEncoder(embedding_dim=512, pretrained=True)
images = torch.randn(32, 3, 224, 224)  # Batch of 32 images
embeddings = image_encoder(images)      # (32, 512)
```

#### TextEncoder
- Transformer-based encoder with learnable positional embeddings
- Token embeddings → Positional embeddings → Transformer layers → Layer norm
- Uses [CLS] token (first token) as the representation
- Input: Token indices of shape `(batch_size, max_seq_length)`
- Output: Embeddings of shape `(batch_size, embedding_dim)`

```python
from models import TextEncoder

text_encoder = TextEncoder(vocab_size=10000, embedding_dim=512)
text_tokens = torch.randint(0, 10000, (32, 77))  # Batch of 32 texts, max length 77
embeddings = text_encoder(text_tokens)            # (32, 512)
```

### Loss Functions (losses.py)

#### CLIPLoss
Implements symmetric contrastive loss for text-to-image matching:
- Normalizes embeddings to unit vectors
- Computes similarity matrix between image and text embeddings
- Applies softmax to get probabilities
- Cross-entropy loss where ground truth is the diagonal (matching pairs)
- Returns average of image-to-text and text-to-image losses

```python
from models import CLIPModel, CLIPLoss

model = CLIPModel()
loss_fn = CLIPLoss(temperature=0.07)

images = torch.randn(32, 3, 224, 224)
text_tokens = torch.randint(0, 10000, (32, 77))

img_emb, txt_emb = model(images, text_tokens)
loss = loss_fn(img_emb, txt_emb)
```

#### SimCLRLoss (SimCLRLossSimplified)
Implements NT-Xent loss for contrastive learning:
- Takes embeddings from two augmented views of the same images
- Concatenates them to form `2*batch_size` representations
- Computes similarity matrix
- For each sample, finds the positive pair among all negatives
- Cross-entropy loss treats positive pair as correct class

```python
from models import SimCLRModel, SimCLRLossSimplified

model = SimCLRModel()
loss_fn = SimCLRLossSimplified(temperature=0.07)

x_i = torch.randn(32, 3, 224, 224)  # First augmentation
x_j = torch.randn(32, 3, 224, 224)  # Second augmentation

z_i, z_j = model(x_i, x_j)
loss = loss_fn(z_i, z_j)
```

### Models

#### CLIPModel
Full CLIP model combining image and text encoders.

```python
from models import CLIPModel

model = CLIPModel(
    image_embedding_dim=512,
    text_embedding_dim=512,
    vocab_size=10000,
    text_max_seq_length=77,
    num_text_layers=12,
    num_text_heads=8,
)

# Forward pass
images = torch.randn(32, 3, 224, 224)
text_tokens = torch.randint(0, 10000, (32, 77))
img_emb, txt_emb = model(images, text_tokens)

# Or use individual encoders
img_emb = model.get_image_embeddings(images)
txt_emb = model.get_text_embeddings(text_tokens)

# Compute similarity (for retrieval)
similarity = model.compute_similarity(img_emb, txt_emb)  # (32, 32)
```

#### SimCLRModel
Full SimCLR model with projection head.

```python
from models import SimCLRModel

model = SimCLRModel(
    embedding_dim=512,
    projection_dim=128,
)

# Forward pass with two augmented views
x_i = torch.randn(32, 3, 224, 224)
x_j = torch.randn(32, 3, 224, 224)
z_i, z_j = model(x_i, x_j)  # Projections: (32, 128) each

# Get embeddings for downstream tasks
embeddings = model.get_embeddings(x_i)              # (32, 512)
projected = model.get_projected_embeddings(x_i)    # (32, 128)
```

### Utilities (utils.py)

Helpful functions for data augmentation and training:

- `get_image_transforms()`: Standard image preprocessing with optional augmentation
- `get_simclr_augmentations()`: Two different aggressive augmentation pipelines for SimCLR
- `normalize_embeddings()`: L2 normalization for embeddings
- `compute_cosine_similarity()`: Compute similarity between embedding sets
- `get_device()`: Get available device (GPU/CPU)
- `count_parameters()`: Count trainable parameters
- `freeze_encoder()` / `unfreeze_encoder()`: Freeze/unfreeze parameters

## Quick Start

### 1. Initialize Models

```python
import torch
from models import CLIPModel, SimCLRModel, CLIPLoss, SimCLRLossSimplified

# CLIP for text-to-image retrieval
clip_model = CLIPModel(
    image_embedding_dim=512,
    text_embedding_dim=512,
    vocab_size=10000,
)

# SimCLR for image-to-image self-supervised learning
simclr_model = SimCLRModel(
    embedding_dim=512,
    projection_dim=128,
)

# Loss functions
clip_loss = CLIPLoss(temperature=0.07)
simclr_loss = SimCLRLossSimplified(temperature=0.07)
```

### 2. Training

Use the `CLIPTrainer` and `SimCLRTrainer` classes in `train_template.py`:

```python
from train_template import CLIPTrainer, SimCLRTrainer

# CLIP Training
clip_trainer = CLIPTrainer(
    model=clip_model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    learning_rate=1e-4,
)
clip_trainer.train(num_epochs=100)

# SimCLR Training
simclr_trainer = SimCLRTrainer(
    model=simclr_model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    learning_rate=1e-4,
)
simclr_trainer.train(num_epochs=100)
```

### 3. Data Requirements

#### CLIP DataLoader
Your dataloader should return dictionaries with:
- `images`: Tensor of shape `(batch_size, 3, 224, 224)`
- `text_tokens`: Tensor of shape `(batch_size, max_seq_length)` with token indices
- `text_mask` (optional): Tensor of shape `(batch_size, max_seq_length)` for padding

#### SimCLR DataLoader
Your dataloader should return dictionaries with:
- `x_i`: First augmentation, shape `(batch_size, 3, 224, 224)`
- `x_j`: Second augmentation, shape `(batch_size, 3, 224, 224)`

### 4. Inference

```python
# CLIP: Text-to-image retrieval
text_query = "a cat sitting on a table"
text_tokens = tokenize(text_query)  # Your tokenization

with torch.no_grad():
    text_embedding = clip_model.get_text_embeddings(text_tokens)
    image_embeddings = clip_model.get_image_embeddings(images)
    similarity = clip_model.compute_similarity(image_embeddings, text_embedding)
    
    # Get top-k matching images
    top_k_indices = torch.topk(similarity, k=5, dim=0).indices
```

## Key Hyperparameters

- **temperature** (0.07): Controls sharpness of similarity distributions. Lower = sharper
- **embedding_dim** (512): Dimension of embedding space
- **projection_dim** (128 for SimCLR): Dimension of projection space (usually smaller)
- **learning_rate** (1e-4): Initial learning rate
- **weight_decay** (1e-6): L2 regularization coefficient

## Performance Tips

1. **Use pretrained ResNet-50**: Setting `image_pretrained=True` speeds up convergence
2. **Normalize embeddings**: Always normalize before computing similarities
3. **Batch size**: Larger batches (256-1024) improve contrastive learning performance
4. **Learning rate schedule**: Use cosine annealing for better convergence
5. **Gradient clipping**: Helps stabilize training (max_norm=1.0)
6. **Mixed precision**: Use torch.cuda.amp for faster training on GPUs

## References

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.14030)
- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

## Next Steps

1. Create dataset loaders in a `dataset/` directory
2. Implement the data loading and tokenization
3. Use `CLIPTrainer` or `SimCLRTrainer` to start training
4. Save embeddings for retrieval downstream tasks
5. Fine-tune on your specific domain if needed
