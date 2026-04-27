# Image-Retrieval-Model

An advanced image retrieval system that retrieves similar images using CLIP (Contrastive Language-Image Pre-training) with FAISS vector store for efficient similarity search.

## Architecture

### CLIP Model
- Learns joint embeddings of images and text through contrastive learning
- **Image Encoder**: ResNet-50 backbone with projection head
- **Text Encoder**: Transformer-based encoder
- **Loss**: Symmetric cross-entropy (text-to-image and image-to-text)
- **Use Case**: Query images using natural language descriptions or find similar images

### FAISS Integration
- Efficient similarity search with FAISS vector store
- Indexed embeddings for fast retrieval
- Support for various similarity metrics

## Project Structure

```
Image-Retrieval-Model/
├── models/                          # Model implementations
│   ├── clip_model.py               # CLIP model
│   ├── encoders.py                 # ImageEncoder and TextEncoder
│   ├── losses.py                   # CLIPLoss
│   ├── faiss_vector_store.py       # FAISS vector store for efficient retrieval
│   ├── utils.py                    # Utilities (transforms, helpers)
│   └── README.md                   # Detailed model documentation
├── dataset/                         # Dataset utilities and downloads
├── trained_model_clip/             # Pre-trained CLIP model and embeddings
│   ├── clip_encoder.pth
│   ├── config.json
│   └── image_embeddings.index
├── train.py                        # Main training script
├── train_model.py                  # Alternative training script
├── train_with_faiss.py            # Training with FAISS integration
├── evaluate_clip_retrieval.py      # Evaluation script
├── retrieve_images.py              # Image retrieval utilities
├── retrieve_similar_images.py      # Similar image retrieval
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

See [models/README.md](models/README.md) for detailed documentation and examples.

```python
from models import CLIPModel, CLIPLoss

# Initialize model
clip_model = CLIPModel()

# Loss function
clip_loss = CLIPLoss(temperature=0.07)
```

### Training

Run the main training script:

```bash
python train.py
```

For training with FAISS integration:

```bash
python train_with_faiss.py
```

See individual training scripts for configuration options and parameters.

## Features

### CLIP
- Text-to-image and image-to-image retrieval
- Learnable temperature parameter for dynamic scaling
- Supports attention masks for variable-length text
- Normalized embeddings for efficient similarity computation
- Joint image-text embedding space

### FAISS Integration
- Efficient vector indexing and search
- Fast similarity retrieval over large datasets
- Multiple indexing strategies support

### Utilities
- Multiple image augmentation strategies
- Helper functions for training and inference
- Device management (GPU/CPU)
- Parameter counting and freezing
- Embedding extraction and evaluation

## References

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.14030)
- FAISS: [Billion-scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734)
