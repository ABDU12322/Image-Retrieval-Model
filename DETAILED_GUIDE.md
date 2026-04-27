# Image Retrieval Models - Complete Detailed Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [File Structure and Detailed Explanations](#file-structure)
4. [Data Flow Walkthrough](#data-flow)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Step-by-Step Code Walkthrough](#code-walkthrough)
7. [Training Process](#training-process)
8. [Inference and Usage](#inference-and-usage)

---

## Project Overview

This project implements two state-of-the-art image retrieval models:

1. **CLIP (Contrastive Language-Image Pre-training)**: Learns joint embeddings of images and text to enable text-to-image retrieval
2. **SimCLR (Simple Framework for Contrastive Learning)**: Learns image representations through self-supervised contrastive learning without labels

### Goal
Enable retrieval of similar images from a database using either:
- Text queries ("a dog playing fetch") for CLIP
- Image queries for SimCLR

---

## Architecture Overview

### High-Level Concept

Both models use **contrastive learning**, which means:
- Pull similar items closer in embedding space
- Push dissimilar items farther apart in embedding space

### CLIP Architecture

```
Image Encoder (ResNet-50 + Projection)  →  Image Embedding (512-dim)
                                              ↓
Input: Image + Text ──────→  Compute Similarity Matrix ──→ Loss Calculation
                                              ↑
Text Encoder (Transformer + Projection) → Text Embedding (512-dim)
```

**Key idea**: An image and its caption should have similar embeddings, while mismatched pairs should be different.

### SimCLR Architecture

```
Image 1 ────→ Encoder ──→ Projection Head ──→ z_i (128-dim)
              (ResNet-50)                        ↓
                                          Contrastive Loss
                                                 ↑
              (ResNet-50)                    z_j (128-dim)
Image 2 ────→ Encoder ──→ Projection Head ──┘
(augmented version of Image 1)
```

**Key idea**: Two augmented versions of the same image should have similar embeddings.

---

## File Structure and Detailed Explanations

### Directory Tree

```
models/
├── __init__.py           # Imports and exposes public API
├── encoders.py           # ImageEncoder and TextEncoder classes
├── losses.py             # CLIPLoss and SimCLRLoss implementations
├── clip_model.py         # Full CLIP model combining encoders
├── simclr_model.py       # Full SimCLR model with projection head
├── utils.py              # Utility functions for training
└── README.md             # API documentation

train_template.py         # Training loops for both models
requirements.txt          # Python dependencies
README.md                 # Project overview
DETAILED_GUIDE.md         # This file
```

---

## encoders.py - Image and Text Encoding

### ImageEncoder Class

**Purpose**: Convert raw images into fixed-dimensional embeddings

**Components**:
1. **ResNet-50 Backbone**: Pre-trained convolutional neural network for feature extraction
2. **Projection Head**: Maps high-dimensional features to lower-dimensional embeddings

#### Detailed Breakdown:

```python
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load ResNet-50 (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove classification layer (we only want features)
        # ResNet-50 has: Conv → ResBlocks → AvgPool → FC(1000)
        # We remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_dim = 2048  # ResNet-50 outputs 2048-dim features
        
        # Create projection head:
        # 2048 → 2048 → 512 (embedding_dim)
        # Why 2 layers? Non-linear transformation helps model learn better
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),  # 2048 → 2048
            nn.ReLU(inplace=True),                          # Non-linearity
            nn.Linear(self.feature_dim, embedding_dim),     # 2048 → 512
        )
```

**Forward Pass (step-by-step)**:

```python
def forward(self, x):
    # Input: x is batch of images
    # Shape: (batch_size=32, channels=3, height=224, width=224)
    
    # Step 1: Extract features using ResNet-50
    features = self.backbone(x)  
    # Output shape: (32, 2048, 1, 1)
    # Explanation: 2048 feature maps, each 1x1 (spatial dimensions reduced)
    
    # Step 2: Flatten to 2D tensor
    features = torch.flatten(features, 1)  
    # Output shape: (32, 2048)
    # Explanation: Convert from (32, 2048, 1, 1) to (32, 2048)
    
    # Step 3: Project to embedding dimension
    embeddings = self.projection(features)  
    # Output shape: (32, 512)
    # Explanation: Map 2048-dim features to 512-dim embeddings
    
    return embeddings
```

**Why this design?**
- **ResNet-50**: Proven architecture for image recognition
- **Pretrained weights**: Already learned good features from ImageNet
- **2-layer projection**: Non-linear transformation improves expressiveness
- **512-dim embeddings**: Balance between expressiveness and memory efficiency

---

### TextEncoder Class

**Purpose**: Convert text sequences into fixed-dimensional embeddings

**Components**:
1. **Token Embeddings**: Convert word indices to vectors
2. **Positional Embeddings**: Add position information to each token
3. **Transformer Encoder**: Multi-head attention layers to process sequences
4. **Layer Normalization**: Stabilize training
5. **[CLS] Token Pooling**: Use first token as sentence representation

#### Detailed Breakdown:

```python
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,           # Vocabulary size
        embedding_dim: int = 512,          # Embedding dimension
        max_seq_length: int = 77,          # Max text length (CLIP uses 77)
        num_layers: int = 12,              # Number of transformer layers
        num_heads: int = 8,                # Attention heads per layer
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Token Embedding Layer
        # Converts token IDs (0-9999) to vectors of size 512
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Shape: (vocab_size, embedding_dim) = (10000, 512)
        
        # Positional Embeddings
        # Learnable vectors indicating position in sequence
        # Each position has unique embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim)
        )
        # Shape: (1, 77, 512)
        # "1" is broadcast to all batch items
        
        # Transformer Encoder Layer (repeated num_layers times)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,           # 512
            nhead=num_heads,                 # 8 attention heads
            dim_feedforward=embedding_dim * 4,  # 512*4=2048 (intermediate FC)
            batch_first=True,                # Input: (batch, seq, features)
            norm_first=True,                 # LayerNorm before attention
        )
        
        # Stack multiple transformer layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,           # 12 stacked layers
        )
        
        # Final layer normalization
        self.ln = nn.LayerNorm(embedding_dim)
```

**Forward Pass (step-by-step)**:

```python
def forward(self, x, mask=None):
    # Input: x is batch of tokenized text
    # Shape: (batch_size=32, max_seq_length=77)
    # Values: token IDs (0-9999)
    # Example: [101, 2054, 2003, 1037, 3899, 102, 0, 0, ...]
    #          (special tokens and padding)
    
    # Step 1: Token Embedding
    embeddings = self.token_embedding(x)
    # Output shape: (32, 77, 512)
    # Explanation: Each token ID mapped to 512-dim vector
    
    # Step 2: Add Positional Embeddings
    # Why? Transformer doesn't know word order without this
    # Position 0 gets positional_embedding[0]
    # Position 1 gets positional_embedding[1], etc.
    embeddings = embeddings + self.positional_embedding[:, :embeddings.size(1), :]
    # Output shape: (32, 77, 512)
    # Each position now contains both token info AND position info
    
    # Step 3: Pass through Transformer
    # Transformer uses self-attention to learn relationships between tokens
    # Example: "The cat sat on the mat"
    # Learns: "cat" attends to "sat", "mat", etc.
    output = self.transformer_encoder(
        embeddings,
        src_key_padding_mask=mask,  # Ignore padding tokens
    )
    # Output shape: (32, 77, 512)
    # Explanation: Contextual representation of each token
    
    # Step 4: Layer Normalization
    output = self.ln(output)
    # Output shape: (32, 77, 512)
    
    # Step 5: [CLS] Token Pooling
    # Use first token as sentence representation
    # (Similar to BERT's approach)
    embeddings = output[:, 0, :]
    # Output shape: (32, 512)
    # Explanation: Take first token from each sequence
    
    return embeddings
```

**Why this design?**
- **Token embeddings**: Standard way to represent discrete tokens
- **Positional embeddings**: Preserves word order information
- **Transformer**: Self-attention captures long-range dependencies
- **[CLS] pooling**: Simple and effective way to aggregate sequence into single vector
- **Layer normalization**: Improves training stability and convergence

---

## losses.py - Loss Functions

### CLIPLoss Class

**Purpose**: Train model to match images with text

**Mathematical Concept**:

The loss encourages:
- Images to have high similarity with their correct caption
- Images to have low similarity with incorrect captions

**Forward Pass (Mathematical)**:

```
Given batch of N images and N captions:
1. Normalize all embeddings to unit vectors
2. Compute similarity matrix S: (N, N)
   - S[i,j] = cos_similarity(image_i, text_j)
   - Diagonal elements S[i,i] should be high
   - Off-diagonal elements should be low

3. Image-to-text loss (temperature scaled):
   - Treat as classification: which text matches image_i?
   - Softmax over all N texts
   - Target: text_i (diagonal)
   - Loss_i2t = CrossEntropy(logits, labels)

4. Text-to-image loss:
   - Reverse: which image matches text_i?
   - Loss_t2i = CrossEntropy(logits.T, labels)

5. Total loss = (Loss_i2t + Loss_t2i) / 2
```

**Code Implementation**:

```python
def forward(self, image_embeddings, text_embeddings):
    # Input:
    # - image_embeddings: (32, 512)
    # - text_embeddings: (32, 512)
    
    # Step 1: L2 normalize embeddings to unit vectors
    # Why? Makes similarity bounded between -1 and 1
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    # Output: (32, 512) with norm=1
    
    # Step 2: Compute similarity matrix
    # Element [i,j] = dot product of image_i and text_j
    logits = torch.matmul(
        image_embeddings, text_embeddings.t()  # (32,512) × (512,32) → (32,32)
    ) / self.temperature  # Divide by 0.07 to scale
    # Output: (32, 32) similarity matrix
    # Example:
    # [[0.85, 0.02, 0.01],    ← image 0 with all texts
    #  [0.03, 0.82, 0.04],    ← image 1 with all texts
    #  [0.02, 0.05, 0.88]]    ← image 2 with all texts
    # Diagonal should be high (correct pairs)
    
    # Step 3: Create target labels
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size)  # [0, 1, 2, ..., 31]
    # Meaning: image_i should match text_i
    
    # Step 4: Image-to-text loss
    # Treat similarity matrix as classification logits
    # Question: "For image_i, which text is it?"
    # Answer should be text_i
    loss_i2t = F.cross_entropy(logits, labels)
    
    # Step 5: Text-to-image loss
    # Transpose the logits
    # Question: "For text_i, which image is it?"
    # Answer should be image_i
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    # Step 6: Average both directions
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss  # Scalar value to minimize
```

**Visual Example**:

```
Say batch_size = 3, and we have 3 images and 3 captions

Images: [dog photo, cat photo, bird photo]
Text:   ["a dog", "a cat", "a bird"]

CORRECT MATCHING:
- dog photo ↔ "a dog"
- cat photo ↔ "a cat"
- bird photo ↔ "a bird"

Similarity matrix (after softmax):
              "a dog"   "a cat"   "a bird"
dog photo   [  0.85,    0.10,     0.05  ]  ← high on diagonal ✓
cat photo   [  0.05,    0.88,     0.07  ]  ← high on diagonal ✓
bird photo  [  0.02,    0.08,     0.90  ]  ← high on diagonal ✓

Loss tries to maximize diagonal, minimize off-diagonal
```

---

### SimCLRLoss Class (SimCLRLossSimplified)

**Purpose**: Train model to match augmented views of same image

**Key Insight**: If we take the same image and apply two different augmentations, the model should produce similar embeddings for both.

**Mathematical Concept** (NT-Xent Loss):

```
Given:
- z_i: embedding of image with augmentation 1
- z_j: embedding of same image with augmentation 2

For each sample i in the batch:
- Its positive pair is z_j[i] (same image, different augmentation)
- Its negative pairs are all other z_j samples and all z_i samples

Loss encourages:
- High similarity between z_i and z_j[i]
- Low similarity between z_i and all other z_j/z_i

Math:
L_i = -log(exp(sim(z_i, z_j[i]) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
where τ is temperature, k ranges over all samples
```

**Code Implementation**:

```python
def forward(self, z_i, z_j):
    # Input:
    # - z_i: embeddings from first augmentation (32, 128)
    # - z_j: embeddings from second augmentation (32, 128)
    # Both from SAME images, just different augmentations
    
    batch_size = z_i.size(0)  # 32
    
    # Step 1: Normalize embeddings
    z_i = F.normalize(z_i, dim=-1)  # (32, 128)
    z_j = F.normalize(z_j, dim=-1)  # (32, 128)
    
    # Step 2: Concatenate both views
    # Combine z_i and z_j to form bigger batch
    z = torch.cat([z_i, z_j], dim=0)  # (64, 128)
    # Order: [z_i[0], z_i[1], ..., z_i[31], z_j[0], z_j[1], ..., z_j[31]]
    
    # Step 3: Compute similarity matrix
    # Each row is similarity between that sample and all 64 samples
    similarity_matrix = torch.matmul(z, z.t()) / self.temperature
    # Shape: (64, 64)
    # Example similarity_matrix[0,:]:
    # [1.0,  0.1,  0.05, ..., 0.92, 0.08, ...]
    #  ↑ self-sim    ↑ other z_i      ↑ z_j[0] (positive pair!) ↑ other z_j
    
    # Step 4: Create labels
    # First 32 samples (z_i) should match with samples 32-64 (z_j)
    # z_i[0] should match z_j[0] → label should point to index 32
    # z_i[1] should match z_j[1] → label should point to index 33, etc.
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),  # [32, 33, 34, ..., 63]
        torch.arange(batch_size),                   # [0, 1, 2, ..., 31]
    ])
    # Meaning:
    # - For z_i[0] (index 0), positive pair is at index 32 (z_j[0])
    # - For z_j[0] (index 32), positive pair is at index 0 (z_i[0])
    
    # Step 5: Mask self-similarity
    # We don't want model to just memorize self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool)
    similarity_matrix.masked_fill_(mask, float('-inf'))
    # Replaces diagonal with -inf so softmax ignores them
    
    # Step 6: Compute cross-entropy loss
    # For each sample, predict which is its positive pair
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss  # Scalar value to minimize
```

**Visual Example**:

```
Given 3 images, each with 2 augmentations:

Image 1: dog photo
  - Augmentation A: dog photo (darker)   → embedding z_i[0]
  - Augmentation B: dog photo (rotated)  → embedding z_j[0]
  These should be similar!

Image 2: cat photo
  - Augmentation A: cat photo (blurred)  → embedding z_i[1]
  - Augmentation B: cat photo (flipped)  → embedding z_j[1]
  These should be similar!

Image 3: bird photo
  - Augmentation A: bird photo (zoomed)  → embedding z_i[2]
  - Augmentation B: bird photo (bright)  → embedding z_j[2]
  These should be similar!

Concatenated: [z_i[0], z_i[1], z_i[2], z_j[0], z_j[1], z_j[2]]

Positive pairs:
- z_i[0] with z_j[0] (indices 0 and 3)
- z_i[1] with z_j[1] (indices 1 and 4)
- z_i[2] with z_j[2] (indices 2 and 5)

Similarity matrix similarities to sample 0:
[ignored, 0.1, 0.08, 0.92, 0.05, 0.03]
         ↑     ↑                ↑
      other  other            z_j[0]
       z_i    z_i            POSITIVE!

Loss makes index 3 (z_j[0]) have highest score for sample 0
```

---

## clip_model.py - Full CLIP Model

**Purpose**: Combine ImageEncoder and TextEncoder into one model

```python
class CLIPModel(nn.Module):
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
        
        # Create image encoder
        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            pretrained=image_pretrained,
        )
        
        # Create text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            max_seq_length=text_max_seq_length,
            num_layers=num_text_layers,
            num_heads=num_text_heads,
        )
        
        # Learnable temperature parameter
        # Controls how sharp the probability distribution is
        # Starts at ~1.0 (log(0.07) ≈ 2.66)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.logit_scale.requires_grad = True
```

**Forward Pass**:

```python
def forward(self, images, text_tokens, text_mask=None):
    # Step 1: Encode images
    image_embeddings = self.image_encoder(images)  # (batch, 512)
    
    # Step 2: Encode text
    text_embeddings = self.text_encoder(text_tokens, mask=text_mask)  # (batch, 512)
    
    # Return both for loss computation
    return image_embeddings, text_embeddings
```

**Utility Methods**:

```python
def compute_similarity(self, image_embeddings, text_embeddings):
    """
    For retrieval: compute similarity between all images and texts
    """
    # Normalize
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    # Multiply by learned temperature scale
    similarity = torch.matmul(
        image_embeddings, text_embeddings.t()
    ) * self.logit_scale.exp()
    
    return similarity  # (num_images, num_texts)
```

---

## simclr_model.py - Full SimCLR Model

**Purpose**: Image encoder with projection head for contrastive learning

```python
class SimCLRModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        image_pretrained: bool = True,
    ):
        super().__init__()
        
        # Backbone encoder
        self.encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=image_pretrained,
        )
        # Output: (batch, 512)
        
        # Projection head for contrastive loss
        # Maps 512 → 512 → 128
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )
        # Output: (batch, 128)
```

**Forward Pass**:

```python
def forward(self, x_i, x_j):
    # Step 1: Encode both augmented views
    h_i = self.encoder(x_i)  # (batch, 512) - features
    h_j = self.encoder(x_j)  # (batch, 512) - features
    
    # Step 2: Project for contrastive loss
    z_i = self.projection_head(h_i)  # (batch, 128)
    z_j = self.projection_head(h_j)  # (batch, 128)
    
    return z_i, z_j
```

**Key Distinction**:
- `encoder(x)` → Returns h (512-dim) for downstream tasks like classification
- `projection_head(h)` → Returns z (128-dim) for contrastive loss during training

This is important because:
- **h embeddings**: Good general features, use for downstream tasks
- **z embeddings**: Optimized for contrastive loss during training

---

## utils.py - Utility Functions

### 1. Image Transformations

```python
def get_image_transforms(image_size: int = 224, augmentation: bool = False):
    """
    Purpose: Prepare images for model input
    
    Without augmentation (inference/validation):
    - Resize to 224x224
    - Convert to tensor
    - Normalize using ImageNet statistics
    
    With augmentation (training):
    - Random crop from random scale
    - Random horizontal flip
    - Color jittering (brightness, contrast, saturation)
    - Random rotation
    - Normalize
    
    ImageNet normalization:
    - mean=[0.485, 0.456, 0.406] ← average RGB values
    - std=[0.229, 0.224, 0.225]  ← color variance
    """
```

### 2. SimCLR Augmentations

```python
def get_simclr_augmentations(image_size: int = 224):
    """
    Purpose: Create two different aggressive augmentation pipelines
    
    Why aggressive? SimCLR needs strong augmentations to learn invariances
    
    Pipeline 1:
    - RandomResizedCrop: Crop and resize with different scales
    - RandomHorizontalFlip: 50% chance to flip
    - ColorJitter: Random brightness/contrast/saturation changes
    - RandomGrayscale: 20% chance to convert to grayscale
    - GaussianBlur: Add blur with random sigma
    
    Pipeline 2: Similar but with RandomRotation instead of GaussianBlur
    
    Why two different pipelines?
    - Each image gets processed twice with DIFFERENT augmentations
    - Ensures model learns true invariances, not just augmentation patterns
    """
```

### 3. Helper Functions

```python
def normalize_embeddings(embeddings):
    """
    Convert embeddings to unit vectors (L2 norm = 1)
    Useful for cosine similarity: sim = dot_product(normalized_emb1, normalized_emb2)
    """

def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute similarity matrix between two sets of embeddings
    Returns: (n, m) matrix with cosine similarities
    """

def get_device():
    """
    Returns: GPU device if available, else CPU
    """

def count_parameters(model):
    """
    Returns: Total number of trainable parameters
    """

def freeze_encoder(encoder):
    """
    Sets requires_grad=False for all parameters
    Use when: Want to keep encoder weights fixed (e.g., using pretrained model as fixed feature extractor)
    """

def unfreeze_encoder(encoder):
    """
    Sets requires_grad=True for all parameters
    Use when: Want to train/fine-tune the encoder
    """
```

---

## train_template.py - Training Infrastructure

### CLIPTrainer Class

**Purpose**: Handle training loop, validation, and checkpointing for CLIP model

```python
class CLIPTrainer:
    def __init__(self, model, train_loader, val_loader=None, learning_rate=1e-4, ...):
        """
        Setup:
        - Model to device
        - Loss function (CLIPLoss)
        - Optimizer (AdamW) - adaptive momentum optimizer
        - Learning rate scheduler (CosineAnnealing) - gradually reduce LR
        """
    
    def train_epoch(self):
        """
        One complete pass through training data:
        
        For each batch:
        1. Move data to GPU
        2. Forward pass: get image and text embeddings
        3. Compute loss: CLIPLoss between embeddings
        4. Backward pass: compute gradients
        5. Clip gradients: prevent exploding gradients
        6. Update weights: optimizer step
        
        Returns: Average loss for epoch
        """
    
    def validate(self):
        """
        Evaluate on validation set (no gradient updates):
        
        For each batch:
        1. Forward pass (no gradients)
        2. Compute loss
        
        Returns: Average validation loss
        
        Used to detect overfitting
        """
    
    def train(self, num_epochs):
        """
        Main training loop:
        
        For each epoch:
        1. train_epoch()
        2. validate()
        3. Step learning rate scheduler
        4. Save checkpoint every 10 epochs
        """
```

### SimCLRTrainer Class

**Similar to CLIPTrainer but**:
- Uses SimCLRLossSimplified instead of CLIPLoss
- Expects batches with two augmented views (x_i, x_j)
- Same training infrastructure and pattern

---

## Data Flow Walkthrough

### Complete CLIP Training Flow

```
1. DATA PREPARATION
   Raw images (e.g., dog.jpg)         Raw text ("a dog")
         ↓                                   ↓
   [Transform: Resize, normalize]    [Tokenize: text → indices]
         ↓                                   ↓
   Tensor (32, 3, 224, 224)          Tensor (32, 77)
   
2. MODEL FORWARD PASS
   Images ──→ ImageEncoder ──→ Image Embeddings (32, 512)
                                        ↓
                                   Normalize (L2)
                                        ↓
                                   (32, 512) normalized
                                        
   Text ──→ TextEncoder ──→ Text Embeddings (32, 512)
                                   ↓
                              Normalize (L2)
                                   ↓
                              (32, 512) normalized

3. LOSS COMPUTATION
   Image emb (32, 512) ──┐
                         ├──→ Compute similarity matrix (32, 32)
   Text emb (32, 512)   ──┘
                              ↓
                         Apply softmax per row
                              ↓
                         Cross-entropy: predict correct text for each image
                              ↓
                         Cross-entropy: predict correct image for each text (transpose)
                              ↓
                         Average both losses
                              ↓
                         Scalar loss value

4. BACKPROPAGATION
   Loss ──→ Compute gradients ──→ All model parameters updated
   
5. REPEAT for next batch/epoch
```

### Complete SimCLR Training Flow

```
1. DATA PREPARATION
   Original image (dog.jpg)
         ↓
   [Create 2 random augmentations]
         ↓
   Augmentation 1 (32, 3, 224, 224)
   Augmentation 2 (32, 3, 224, 224)
   
2. MODEL FORWARD PASS
   Augmentation 1 ──→ Encoder ──→ Features (32, 512)
                                      ↓
                                 Projection Head
                                      ↓
                                   z_i (32, 128)
                                   
   Augmentation 2 ──→ Encoder ──→ Features (32, 512)
                                      ↓
                                 Projection Head
                                      ↓
                                   z_j (32, 128)

3. LOSS COMPUTATION
   z_i (32, 128) ──┐
                   ├──→ Normalize
                   ├──→ Concatenate to (64, 128)
   z_j (32, 128) ──┘       ↓
                      Compute similarity matrix (64, 64)
                           ↓
                      Create positive pair labels
                      [32,33,34,...,63, 0,1,2,...,31]
                           ↓
                      Cross-entropy: for each sample,
                      predict which is its positive pair
                           ↓
                      Scalar loss value

4. BACKPROPAGATION
   Loss ──→ Compute gradients ──→ All model parameters updated
   
5. REPEAT for next batch/epoch
```

---

## Mathematical Foundations

### Temperature in Contrastive Learning

**Purpose**: Control "sharpness" of similarity distribution

```
Without temperature: similarity = dot_product(a, b)
Range: [-1, 1]

With temperature: similarity = dot_product(a, b) / τ
Range depends on τ:
- τ = 0.07 (small): Amplifies differences, sharp distribution
  Example: [0.1, 0.2, 0.7] becomes [1.4, 2.8, 10.0] (more extreme)
- τ = 1.0: No change
- τ = 10.0 (large): Dampens differences, smoother distribution
  Example: [0.1, 0.2, 0.7] becomes [0.01, 0.02, 0.07] (more uniform)
  
CLIP uses 0.07 because:
- Sharp distributions make the model more confident
- Helps learn stronger correspondences
- Prevents mode collapse
```

### Contrastive Learning Mathematics

```
Key insight: Learn to maximize similarity to positive pairs, minimize to negatives

For CLIP:
  L = -log(P(correct text | image))
    = -log(exp(sim(img, txt)) / Σ exp(sim(img, all_txt)))
    
Gradient:
  ∂L/∂sim ∝ (model_prediction - ground_truth)
  
Effect:
  - Increases similarity to correct pairs
  - Decreases similarity to incorrect pairs

For SimCLR:
  L = -log(P(positive pair | sample))
    = -log(exp(sim(z_i, z_j)) / Σ exp(sim(z_i, z_k)))
    
Key difference: z_j is only ONE positive pair (not N like in CLIP)
All other samples are negatives!
```

### Normalization in Embedding Space

```
Why normalize?

Without normalization:
- Embeddings can have arbitrary magnitude
- Model could just "cheat" by making all vectors very large
- Similarity not bounded

After L2 normalization:
- All vectors have norm = 1
- Lie on unit hypersphere
- Cosine similarity = dot_product (much simpler!)
- Similarity bounded in [-1, 1]

Example:
Vector: [3, 4]  →  L2 norm = √(9+16) = 5
Normalized: [3/5, 4/5] = [0.6, 0.8]  →  L2 norm = √(0.36+0.64) = 1 ✓
```

---

## Step-by-Step Code Walkthrough

### Example: Complete CLIP Training Loop Iteration

```python
import torch
from models import CLIPModel, CLIPLoss
from train_template import CLIPTrainer

# 1. INITIALIZATION
model = CLIPModel(image_embedding_dim=512, text_embedding_dim=512)
loss_fn = CLIPLoss(temperature=0.07)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. PREPARE BATCH
batch = {
    'images': torch.randn(32, 3, 224, 224),        # 32 random images
    'text_tokens': torch.randint(0, 10000, (32, 77)),  # 32 random text sequences
    'text_mask': torch.zeros(32, 77, dtype=torch.bool),  # No padding
}

# 3. FORWARD PASS
images = batch['images']  # (32, 3, 224, 224)
text_tokens = batch['text_tokens']  # (32, 77)
text_mask = batch['text_mask']  # (32, 77)

# Step A: Image encoding
# ImageEncoder forward:
features = model.image_encoder.backbone(images)  # (32, 2048, 1, 1)
features = torch.flatten(features, 1)            # (32, 2048)
image_embeddings = model.image_encoder.projection(features)  # (32, 512)

# Step B: Text encoding
# TextEncoder forward:
token_emb = model.text_encoder.token_embedding(text_tokens)  # (32, 77, 512)
pos_emb = model.text_encoder.positional_embedding           # (1, 77, 512)
text_emb = token_emb + pos_emb[:, :77, :]  # (32, 77, 512) + (1, 77, 512) = (32, 77, 512)
text_emb = model.text_encoder.transformer_encoder(text_emb)  # (32, 77, 512)
text_emb = model.text_encoder.ln(text_emb)                   # (32, 77, 512)
text_embeddings = text_emb[:, 0, :]  # (32, 512) - take [CLS] token

# 4. LOSS COMPUTATION
# CLIPLoss forward:
image_embeddings_norm = torch.nn.functional.normalize(image_embeddings, dim=-1)  # (32, 512)
text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, dim=-1)    # (32, 512)

# Compute similarity matrix
logits = torch.matmul(image_embeddings_norm, text_embeddings_norm.t()) / 0.07
# (32, 512) @ (512, 32) = (32, 32)
# Example value at logits[0,0] = 0.95 / 0.07 = 13.57

labels = torch.arange(32)  # [0, 1, 2, ..., 31]

# Image-to-text loss
loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
# For logits[0,:], target is label[0] = 0
# Cross-entropy tries to make logits[0,0] largest

# Text-to-image loss
loss_t2i = torch.nn.functional.cross_entropy(logits.t(), labels)

loss = (loss_i2t + loss_t2i) / 2  # Scalar

# 5. BACKWARD PASS
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute gradients for all parameters
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding
optimizer.step()       # Update all parameters
```

### Example: Complete SimCLR Training Loop Iteration

```python
import torch
from models import SimCLRModel, SimCLRLossSimplified

# 1. INITIALIZATION
model = SimCLRModel(embedding_dim=512, projection_dim=128)
loss_fn = SimCLRLossSimplified(temperature=0.07)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 2. PREPARE BATCH (two augmented views of same images)
x_i = torch.randn(32, 3, 224, 224)  # Augmentation 1
x_j = torch.randn(32, 3, 224, 224)  # Augmentation 2 (same images, different aug)

# 3. FORWARD PASS
# SimCLRModel forward:

# View 1 processing
features_i = model.encoder.backbone(x_i)  # (32, 2048, 1, 1)
features_i = torch.flatten(features_i, 1)  # (32, 2048)
h_i = model.encoder.projection(features_i)  # (32, 512)
z_i = model.projection_head(h_i)  # (32, 128)

# View 2 processing
features_j = model.encoder.backbone(x_j)  # (32, 2048, 1, 1)
features_j = torch.flatten(features_j, 1)  # (32, 2048)
h_j = model.encoder.projection(features_j)  # (32, 512)
z_j = model.projection_head(h_j)  # (32, 128)

# 4. LOSS COMPUTATION
# SimCLRLossSimplified forward:
z_i_norm = torch.nn.functional.normalize(z_i, dim=-1)  # (32, 128)
z_j_norm = torch.nn.functional.normalize(z_j, dim=-1)  # (32, 128)

z = torch.cat([z_i_norm, z_j_norm], dim=0)  # (64, 128)
# Order: [i_0, i_1, ..., i_31, j_0, j_1, ..., j_31]

similarity = torch.matmul(z, z.t()) / 0.07  # (64, 64)

labels = torch.cat([
    torch.arange(32, 64),  # [32, 33, ..., 63]
    torch.arange(32),      # [0, 1, ..., 31]
])

mask = torch.eye(64, dtype=torch.bool)
similarity.masked_fill_(mask, float('-inf'))  # Mask diagonal

loss = torch.nn.functional.cross_entropy(similarity, labels)

# 5. BACKWARD PASS
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Training Process

### Complete Training Workflow

```
1. DATA LOADING
   - Load images from disk
   - For CLIP: Load paired captions
   - For SimCLR: Create augmented pairs
   - Batch into dataloaders

2. MODEL INITIALIZATION
   - Create encoder(s)
   - Load pretrained ResNet-50 (except final layer)
   - Initialize projection heads

3. TRAINING LOOP
   FOR each epoch:
     FOR each batch:
       - Move data to GPU
       - Forward pass (get embeddings)
       - Compute loss
       - Backward pass (compute gradients)
       - Optimizer step (update weights)
       
     - Validation (optional)
     - Learning rate scheduling
     - Checkpoint saving (every 10 epochs)

4. CONVERGENCE
   - Monitor loss (should decrease)
   - Check validation loss (should not diverge too far from training)
   - Stop when loss plateaus
```

### Key Training Parameters

```
learning_rate = 1e-4
  Purpose: Size of weight updates
  Too large: Training diverges
  Too small: Training very slow
  This value: Good balance

weight_decay = 1e-6
  Purpose: L2 regularization (penalize large weights)
  Effect: Prevents overfitting, smoother loss landscape

temperature = 0.07
  Purpose: Control sharpness of contrastive distribution
  Effect: 0.07 means sharper distinctions between positive/negative pairs

batch_size = 32 (typical)
  Purpose: Number of samples per gradient update
  Larger batches: More stable gradients but more memory
  Smaller batches: More frequent updates but noisier gradients

num_epochs = 100
  Purpose: Complete passes through training data
  More epochs: Better convergence but risk of overfitting
  Usually: Use early stopping based on validation loss
```

---

## Inference and Usage

### CLIP Text-to-Image Retrieval

```python
from models import CLIPModel
import torch

# 1. LOAD TRAINED MODEL
model = CLIPModel()
model.load_state_dict(torch.load('clip_checkpoint.pt'))
model.eval()

# 2. ENCODE FULL IMAGE DATABASE
all_images = load_images('dataset/')  # Load all images
with torch.no_grad():
    image_embeddings = model.get_image_embeddings(all_images)
    # Shape: (num_images, 512)
    # Store in database

# 3. QUERY WITH TEXT
text_query = "a dog playing fetch"
text_tokens = tokenize(text_query)  # [101, 1037, 3899, 2062, ...]

with torch.no_grad():
    query_embedding = model.get_text_embeddings(text_tokens)
    # Shape: (1, 512)
    
    # Compute similarity with all images
    similarities = model.compute_similarity(
        image_embeddings,
        query_embedding
    )
    # Shape: (num_images, 1)
    
    # Get top-k matches
    top_k = torch.topk(similarities.squeeze(), k=5)
    best_matches = top_k.indices  # Indices of most similar images
    scores = top_k.values  # Similarity scores
```

### SimCLR Image-to-Image Retrieval

```python
from models import SimCLRModel
import torch

# 1. LOAD TRAINED MODEL
model = SimCLRModel()
model.load_state_dict(torch.load('simclr_checkpoint.pt'))
model.eval()

# 2. ENCODE FULL IMAGE DATABASE
all_images = load_images('dataset/')
with torch.no_grad():
    # Get embeddings (from encoder, NOT projection head)
    database_embeddings = model.get_embeddings(all_images)
    # Shape: (num_images, 512)

# 3. QUERY WITH IMAGE
query_image = load_image('query.jpg')
with torch.no_grad():
    query_embedding = model.get_embeddings(query_image.unsqueeze(0))
    # Shape: (1, 512)
    
    # Compute cosine similarity with all images
    from models.utils import compute_cosine_similarity
    similarities = compute_cosine_similarity(
        query_embedding,
        database_embeddings
    )
    # Shape: (1, num_images)
    
    # Get top-k matches
    top_k = torch.topk(similarities.squeeze(), k=5)
    best_matches = top_k.indices
    scores = top_k.values
```

### Fine-tuning on Custom Dataset

```python
from models import CLIPModel, CLIPLoss
from train_template import CLIPTrainer

# 1. LOAD PRETRAINED MODEL
model = CLIPModel()
model.load_state_dict(torch.load('clip_checkpoint.pt'))

# 2. OPTIONAL: FREEZE ENCODER, TRAIN PROJECTION
from models.utils import freeze_encoder
freeze_encoder(model.image_encoder.backbone)  # Keep ResNet features
# Only train: image projection head, text encoder, temperature

# 3. SETUP TRAINER
trainer = CLIPTrainer(
    model=model,
    train_loader=custom_train_loader,
    val_loader=custom_val_loader,
    learning_rate=1e-5,  # Lower LR for fine-tuning
)

# 4. TRAIN
trainer.train(num_epochs=20)

# 5. SAVE
torch.save(model.state_dict(), 'clip_finetuned.pt')
```

---

## Common Questions & Troubleshooting

### Q: Why normalize embeddings?
A: Makes similarity computation independent of magnitude, bounded in [-1,1], enables efficient computation using dot product instead of euclidean distance.

### Q: Why use temperature in contrastive learning?
A: Controls softness of probability distribution. Small temperature (0.07) makes model more confident about distinctions. Large temperature makes distribution smoother.

### Q: Why does CLIP need text encoder?
A: To bridge gap between vision and language. Allows querying by natural language instead of image-to-image only.

### Q: Why does SimCLR need projection head?
A: Projection head maps embeddings to smaller space (512→128) optimized for contrastive loss. Main encoder embeddings used for downstream tasks.

### Q: What if loss doesn't decrease?
A: Check learning rate (too high: diverges, too low: no change), batch size (too small: noisy), temperature (check it's scaled correctly).

### Q: How to use model for production?
A: Encode your image/text database once, store embeddings, use fast similarity search (FAISS) for queries.

---

## Summary

**CLIP Flow**:
Image → ResNet → Projection → Normalize → Compute Similarity Matrix → Cross-Entropy Loss
Text → Transformer → Projection → Normalize ↗

**SimCLR Flow**:
Image Augmentation 1 → ResNet → Projection → Normalize → Concatenate → Similarity Matrix → NT-Xent Loss
Image Augmentation 2 → ResNet → Projection → Normalize ↗

Both use **contrastive learning**: pull positives closer, push negatives farther in embedding space.
