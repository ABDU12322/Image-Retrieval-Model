# Image-Retrieval-Model

An advanced image retrieval system that retrieves similar images from images and text using:
- **CLIP** (Contrastive Language-Image Pre-training) for text-to-image retrieval
- **SimCLR** (Simple Framework for Contrastive Learning) for image-to-image self-supervised learning

## Architecture

### CLIP Model
- Learns joint embeddings of images and text through contrastive learning
- **Image Encoder**: ResNet-50 backbone with projection head
- **Text Encoder**: Transformer-based encoder
- **Loss**: Symmetric cross-entropy (text-to-image and image-to-text)
- **Use Case**: Query images using natural language descriptions

### SimCLR Model
- Learns image representations through self-supervised contrastive learning
- Uses augmented image pairs to learn useful representations
- **Backbone**: ResNet-50 with projection head
- **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Use Case**: Learn representations without labels for downstream retrieval tasks

## Project Structure

```
Image-Retrieval-Model/
├── models/              # Model implementations
│   ├── encoders.py      # ImageEncoder and TextEncoder
│   ├── losses.py        # CLIPLoss and SimCLRLoss
│   ├── clip_model.py    # CLIP model
│   ├── simclr_model.py  # SimCLR model
│   ├── utils.py         # Utilities (transforms, helpers)
│   └── README.md        # Detailed model documentation
├── dataset/             # Dataset utilities and downloads
├── train_template.py    # Training template for both models
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

See [models/README.md](models/README.md) for detailed documentation and examples.

```python
from models import CLIPModel, SimCLRModel, CLIPLoss, SimCLRLossSimplified

# Initialize models
clip_model = CLIPModel()
simclr_model = SimCLRModel()

# Loss functions
clip_loss = CLIPLoss(temperature=0.07)
simclr_loss = SimCLRLossSimplified(temperature=0.07)
```

### Training

Use the trainer classes from `train_template.py`:

```python
from train_template import CLIPTrainer, SimCLRTrainer

# CLIP Training
clip_trainer = CLIPTrainer(model=clip_model, train_loader=train_loader)
clip_trainer.train(num_epochs=100)

# SimCLR Training
simclr_trainer = SimCLRTrainer(model=simclr_model, train_loader=train_loader)
simclr_trainer.train(num_epochs=100)
```

## Features

### CLIP
- Text-to-image and image-to-text retrieval
- Learnable temperature parameter for dynamic scaling
- Supports attention masks for variable-length text
- Normalized embeddings for efficient similarity computation

### SimCLR
- Self-supervised learning without labels
- Aggressive data augmentation pipelines
- Projection head for contrastive learning
- Methods for both projected and non-projected embeddings

### Utilities
- Multiple image augmentation strategies
- Helper functions for training and inference
- Device management (GPU/CPU)
- Parameter counting and freezing

## References

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.14030)
- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
