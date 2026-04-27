# FAISS Implementation - Quick Reference

## What Was Implemented

A complete **image embedding storage and retrieval system** using FAISS that integrates with your CLIP training pipeline.

### The Problem You Wanted to Solve

When training your model:
- Images pass through the image encoder → generates embeddings (512-dim vectors)
- You wanted to **store these embeddings** for later searching
- When you have a query image, you want to **find similar images** from your training set
- You need to know **which image file** corresponds to each embedding

### The Solution

**FAISS (Facebook AI Similarity Search)** provides:
- Efficient vector storage in memory or on disk
- Fast similarity search (find top-K similar vectors)
- Metadata mapping (each vector → image filename)

## Files Created

| File | Purpose |
|------|---------|
| `models/faiss_vector_store.py` | Core FAISS management classes |
| `train_with_faiss.py` | Training script with FAISS integration |
| `retrieve_images.py` | Image retrieval utility |
| `example_faiss_workflow.py` | Complete working example |
| `FAISS_INTEGRATION_GUIDE.md` | Detailed documentation |

## Quick Start

### 1. Install FAISS

```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

### 2. Train with Automatic Embedding Storage

```python
from train_with_faiss import CLIPTrainerWithFAISS
from models import CLIPModel

model = CLIPModel()
trainer = CLIPTrainerWithFAISS(
    model=model,
    train_loader=train_loader,
    vector_store_dir='vector_store',
    embedding_dim=512
)

# Train - embeddings are automatically stored in FAISS
trainer.train(num_epochs=10, store_embeddings=True)
```

**What happens:**
- Each batch of images passes through the encoder
- Image embeddings are extracted
- Embeddings are normalized and added to FAISS index
- Image filenames are stored in metadata
- After training, indices are saved to disk

### 3. Search for Similar Images

```python
from retrieve_images import ImageRetriever

retriever = ImageRetriever(
    model_checkpoint='checkpoints/clip_epoch_10.pt',
    vector_store_dir='vector_store',
    image_root_dir='dataset/images'
)

# Search by image file
results = retriever.search_by_image('query_image.jpg', k=10)

# Display results
for result in results:
    print(f"Rank {result['rank']}: {result['image_name']}")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Found at: {result['disk_path']}")
```

## Architecture Overview

```
TRAINING PHASE
─────────────────────────────────────────
Image Input (batch)
        ↓
   [Image Encoder]
        ↓
   Image Embeddings (512-dim)
        ↓
   Normalize L2
        ↓
   [FAISS Index]  ← Accumulated during training
        ↓
   Save to Disk:
   - image_embeddings.index (binary)
   - image_embeddings_metadata.json (image names)

SEARCH PHASE
─────────────────────────────────────────
Query Image
        ↓
   [Trained Image Encoder]
        ↓
   Query Embedding (512-dim)
        ↓
   Normalize L2
        ↓
   [FAISS Search]  ← Fast similarity search
        ↓
   Top-K Results + Image Names
        ↓
   Find Images on Disk ← Map back to file paths
```

## DataLoader Format

Your DataLoader must return batches with this structure:

```python
batch = {
    'images': tensor,              # (batch_size, 3, 224, 224)
    'text_tokens': tensor,         # (batch_size, 77)
    'text_mask': tensor,           # (batch_size, 77)
    'image_names': list            # ['image_1.jpg', 'image_2.jpg', ...]
}
```

## Key Classes and Methods

### FAISSVectorStore
```python
# Create
store = FAISSVectorStore(embedding_dim=512)

# Add embeddings
store.add_embeddings(embeddings, image_names)

# Search
results = store.search(query_embedding, k=10)

# Persist
store.save_index('index.faiss')
store.load_index('index.faiss')
```

### EmbeddingManager
```python
manager = EmbeddingManager(embedding_dim=512)

# Add batch
manager.add_image_embeddings_batch(embeddings, image_names)

# Search
results = manager.search_similar_images(embedding, k=10)

# Save/Load
manager.save_all_indices()
manager.load_all_indices()
```

### CLIPTrainerWithFAISS
```python
trainer = CLIPTrainerWithFAISS(model, train_loader, ...)

# Train with automatic embedding storage
trainer.train(num_epochs=10, store_embeddings=True)

# Search after training
results = trainer.search_similar_images(query_image, k=10)
```

### ImageRetriever
```python
retriever = ImageRetriever(
    model_checkpoint='path/to/model.pt',
    vector_store_dir='vector_store'
)

# Search by image file
results = retriever.search_by_image('image.jpg', k=10)

# Get all indexed images
all_images = retriever.get_all_image_names()

# Statistics
stats = retriever.get_index_stats()
```

## Search Result Format

```python
[
    {
        'rank': 1,
        'image_name': 'image_001.jpg',
        'distance': 0.0234,      # L2 distance (lower = more similar)
        'similarity': 0.977,     # Scaled to 0-1 (higher = more similar)
        'found_on_disk': True,
        'disk_path': '/path/to/image_001.jpg'
    },
    ...
]
```

## Workflow Examples

### Example 1: Train and Save

```python
# Train
trainer = CLIPTrainerWithFAISS(model, train_loader)
trainer.train(num_epochs=10, store_embeddings=True)

# Outputs:
# - checkpoints/clip_epoch_*.pt
# - vector_store/image_embeddings.index
# - vector_store/image_embeddings_metadata.json
```

### Example 2: Load and Search

```python
# Load previously trained model
retriever = ImageRetriever(
    'checkpoints/clip_epoch_10.pt',
    'vector_store'
)

# Search
results = retriever.search_by_image('query.jpg', k=10)
retriever.print_results(results)
```

### Example 3: Real-Time Search

```python
# During inference, search in FAISS without re-training
embedding = model.get_image_embeddings(query_image_tensor)
results = embedding_manager.search_similar_images(embedding, k=5)
```

## Performance

### Memory Usage (per 1M embeddings)
- 512-dim float32 embeddings: ~2GB
- Metadata (filenames): ~50-100MB
- **Total: ~2.1-2.2GB per million images**

### Search Speed
- **IndexFlatL2 (current)**: 
  - 1M images: ~100ms per query (CPU)
  - 1M images: ~10-50ms per query (GPU)
  
- For faster search on billions, use approximate methods (IndexIVFFlat)

## Integration Checklist

- [x] Core FAISS wrapper classes created
- [x] Training integration implemented
- [x] Retrieval utilities created
- [x] Complete example workflow provided
- [x] Documentation written
- [x] Requirements updated

## Next Steps

1. **Install FAISS**: `pip install faiss-cpu` (or faiss-gpu)

2. **Prepare your dataset**: Ensure DataLoader returns correct format

3. **Train your model**:
   ```python
   python train_with_faiss.py
   ```

4. **Search your embeddings**:
   ```python
   from retrieve_images import ImageRetriever
   # Load and search
   ```

5. **Optional - Run example**:
   ```python
   python example_faiss_workflow.py
   ```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Module faiss not found" | Install with: `pip install faiss-cpu` |
| "Embedding dimension mismatch" | Check vector_store embedding_dim matches model |
| "Image not found on disk" | Set correct `image_root_dir` in ImageRetriever |
| "Slow search" | Use GPU version or approximate indices |
| "Memory error" | Use approximate FAISS indices for large datasets |

## Architecture Advantages

✅ **Scalable**: Works with millions of images  
✅ **Fast**: Sub-second search for millions of vectors  
✅ **Integrated**: Automatic storage during training  
✅ **Persistent**: Save/load indices to/from disk  
✅ **Flexible**: Easy to customize for different use cases  
✅ **Production-ready**: Battle-tested technology (Meta/Facebook)  

---

**For detailed information**, see [FAISS_INTEGRATION_GUIDE.md](FAISS_INTEGRATION_GUIDE.md)
