# Consolidated Workflow Guide

## Overview

Your Image Retrieval Model project has been reorganized into **two main files** for clarity and simplicity:

| File | Purpose | When to Use |
|------|---------|-----------|
| **`train_model.py`** | Train and save your model | When you want to train a new model |
| **`retrieve_similar_images.py`** | Search using trained models | After training, to find similar images |

**Old files** (`train.py`, `train_with_faiss.py`, `train_template.py`) are kept for reference but are no longer needed.

---

## Workflow

### 1. Training (First Time Only)

Run this command to train a model:

```bash
python train_model.py
```

**What happens:**
- Interactive menu to select:
  - Model type (CLIP or SimCLR)
  - Training scale (Small/Medium/Large)
  - Dataset size (Small - 2,000 images / Full - 118,000+ images)
- Model trains and saves automatically to `trained_models/`
- FAISS indices saved to `vector_store/`
- Takes ~minutes to hours depending on scale

**Output:**
```
trained_models/
  ├── clip_20240101_120000/
  │   ├── model_state.pt      ← Model weights
  │   └── metadata.json       ← Config & training info
  ├── simclr_20240101_130000/
  │   ├── model_state.pt
  │   └── metadata.json
  └── ...

vector_store/
  └── faiss_indices/          ← Built during training
```

---

### 2. Searching (Anytime After Training)

Run this command to search for similar images:

```bash
python retrieve_similar_images.py
```

**What happens:**
- Shows list of available trained models
- You select which model to use
- Loads the model and FAISS indices
- Searches for similar images
- Displays results with file paths

**Output:**
```
AVAILABLE MODELS:
1. clip_20240101_120000
   Type: CLIP (Text-to-Image Retrieval)
   Saved: 2024-01-01T12:00:00
   ...

SEARCH RESULTS:
Rank 1:
  Image: image_001.jpg
  Similarity Score: 0.9234
  ✓ Found at: dataset/coco_small/images/image_001.jpg
```

---

## Key Features

### ✅ Model Persistence
- **Before:** Train → Results lost if you close the script
- **Now:** Train → Model saved → Load anytime to get results
- Every trained model saved with:
  - Model weights (`model_state.pt`)
  - Complete configuration (`metadata.json`)
  - Training info (epochs, loss, learning rate, etc.)

### ✅ No Retraining Needed
- Train once: `python train_model.py`
- Search anytime: `python retrieve_similar_images.py`
- Switch between models instantly
- Compare different training configurations

### ✅ Single Purpose Files
- **`train_model.py`** → All training logic (no confusion)
- **`retrieve_similar_images.py`** → All retrieval logic (clear purpose)

---

## Directory Structure

```
Image-Retrieval-Model/
├── train_model.py                 ← Train & save models (RUN THIS FIRST)
├── retrieve_similar_images.py     ← Search & retrieve (RUN THIS SECOND)
│
├── trained_models/                ← Auto-created, stores saved models
│   ├── clip_20240101_120000/
│   │   ├── model_state.pt
│   │   └── metadata.json
│   └── simclr_20240102_150000/
│       ├── model_state.pt
│       └── metadata.json
│
├── vector_store/                  ← Auto-created, FAISS indices
│   └── faiss_indices/
│
├── checkpoints/                   ← Auto-created, epoch checkpoints
│   ├── clip_epoch_5.pt
│   └── clip_epoch_10.pt
│
├── models/                        ← Model architecture code
│   ├── clip_model.py
│   ├── simclr_model.py
│   ├── encoders.py
│   ├── losses.py
│   ├── utils.py
│   └── faiss_vector_store.py
│
└── dataset/                       ← Your training data
    ├── coco_small/
    ├── coco/
    └── download scripts
```

---

## Complete Example

### Step 1: Train a Model (30 seconds for small scale)

```bash
$ python train_model.py

================================================================================
  IMAGE RETRIEVAL MODEL - TRAINING (train_model.py)
================================================================================

This script trains and saves your model for later use.
After training, use retrieve_similar_images.py to search!

================================================================================
  SELECT MODEL TYPE
================================================================================

1. CLIP (Text-to-Image Retrieval)
   Contrastive Language-Image Pre-training
   Capabilities:
     ✓ Text-to-Image retrieval
     ✓ Image-to-Text retrieval
     ✓ Joint image-text embeddings

2. SimCLR (Image-to-Image Retrieval)
   Simple Contrastive Learning of Representations
   Capabilities:
     ✓ Image-to-Image retrieval
     ✓ Self-supervised learning
     ✓ No text labels required

Enter choice (1 or 2): 1
✓ Selected: CLIP (Text-to-Image Retrieval)

[... continue selecting options ...]

Training complete!
============================================================

✓ Model saved to: trained_models/clip_20240101_120000

📝 Next Steps:
   1. Use retrieve_similar_images.py to search with your trained model
   2. No need to retrain - model is saved and ready to use!

💡 Quick Start:
   python retrieve_similar_images.py
```

### Step 2: Search Using Trained Model (Instantly!)

```bash
$ python retrieve_similar_images.py

================================================================================
IMAGE RETRIEVAL - Search Similar Images (retrieve_similar_images.py)
================================================================================

This script searches for similar images using trained models.
No training needed - just load and search!

AVAILABLE MODELS
================================================================================

1. clip_20240101_120000
   Type: CLIP (Text-to-Image Retrieval)
   Saved: 2024-01-01T12:00:00.123456
   Epochs: 2
   Final Loss: 0.1234

Select a model (enter number): 1

STEP 2: Loading model and FAISS indices...
----

✓ Model loaded successfully
✓ FAISS indices loaded successfully

MODEL INFORMATION
================================================================================

Model Type: CLIP (Text-to-Image Retrieval)
Device: cuda

Training Details:
  Epochs: 2
  Batch Size: 16
  Learning Rate: 0.0001
  Data Source: coco_small
  Final Loss: 0.1234

================================================================================

[... search options ...]

SEARCH RESULTS
================================================================================

Rank 1:
  Image: image_001.jpg
  Similarity Score: 0.9234
  Distance: 0.0766
  ✓ Found at: dataset/coco_small/images/image_001.jpg

Rank 2:
  Image: image_002.jpg
  Similarity Score: 0.8912
  Distance: 0.1088
  ✓ Found at: dataset/coco_small/images/image_002.jpg

[... more results ...]

================================================================================

✓ Search complete! Found 10 similar images.
```

---

## Model Metadata Structure

Each saved model includes `metadata.json` with all information:

```json
{
  "model_type": "clip",
  "model_config": {
    "name": "CLIP (Text-to-Image Retrieval)",
    "params": {
      "image_embedding_dim": 512,
      "text_embedding_dim": 512,
      "vocab_size": 10000,
      ...
    }
  },
  "training_info": {
    "num_epochs": 2,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "weight_decay": 1e-06,
    "data_source": "coco_small",
    "training_scale": "small",
    "final_loss": 0.1234,
    "device": "cuda"
  },
  "saved_date": "2024-01-01T12:00:00.123456",
  "pytorch_version": "2.0.0"
}
```

---

## Common Scenarios

### Scenario 1: Train with Small Dataset

```bash
$ python train_model.py
# Select: CLIP, Small Scale, Small Dataset (2,000 images)
# Wait: ~30 seconds - 1 minute
→ saved to: trained_models/clip_20240101_120000/
```

### Scenario 2: Train with Full Dataset

```bash
$ python train_model.py
# Select: CLIP, Large Scale, Full Dataset (118,000+ images)
# Wait: ~1-2 hours
→ saved to: trained_models/clip_20240101_130000/
```

### Scenario 2: Compare Different Models

```bash
# Run retriever and select model 1
$ python retrieve_similar_images.py
→ Search with CLIP model

# Run retriever again and select model 2
$ python retrieve_similar_images.py
→ Search with SimCLR model

# Compare results instantly!
```

### Scenario 3: Use Pre-trained Model

You can also manually load a saved model in your own code:

```python
from train_model import ModelManager
import torch

manager = ModelManager()
model, metadata = manager.load_model('trained_models/clip_20240101_120000')

# Use the model for predictions
embeddings = model.get_image_embeddings(image_tensor)
```

---

## Files to Use / Files to Delete

### ✅ Keep These Files
- ✓ `train_model.py` - NEW consolidated training
- ✓ `retrieve_similar_images.py` - NEW consolidated retrieval
- ✓ `models/` - Model architecture (required)
- ✓ `dataset/` - Your training data (required)
- ✓ `requirements.txt` - Dependencies
- ✓ `README.md` - Main documentation

### ❌ Old Files (No Longer Needed)
- ❌ `train.py` - Replaced by `train_model.py`
- ❌ `train_with_faiss.py` - Replaced by `train_model.py`
- ❌ `train_template.py` - Replaced by `train_model.py`
- ❌ `retrieve_images.py` - Replaced by `retrieve_similar_images.py`

**Note:** Keep old files if you want, but use the new ones. They won't interfere.

---

## Troubleshooting

### Problem: "No saved models found"
**Solution:** Train a model first with `python train_model.py`

### Problem: "FAISS indices not found"
**Solution:** Train a model with `python train_model.py` to generate indices

### Problem: "Data source not found"
**Solution:** Download the dataset:
```bash
python dataset/download.py    # Download COCO dataset (~30GB)
```

### Problem: "CUDA out of memory"
**Solution:** Use small scale training in `train_model.py`

---

## Next Steps

1. **First Time:** Run `python train_model.py` to train and save your first model
2. **After Training:** Run `python retrieve_similar_images.py` to search
3. **Explore:** Check `trained_models/` to see saved models
4. **Compare:** Train multiple models and compare results

---

## Summary

| Task | File | Command |
|------|------|---------|
| Train a new model | `train_model.py` | `python train_model.py` |
| Search for similar images | `retrieve_similar_images.py` | `python retrieve_similar_images.py` |
| List saved models | `retrieve_similar_images.py` | (shown automatically) |
| Load in custom code | `train_model.py` | `ModelManager().load_model(path)` |

**That's it!** 🎉 Your training and retrieval workflows are now consolidated and simple.
