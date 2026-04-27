# FAISS Vector Store Integration Guide

## Overview

This guide explains how to use FAISS (Facebook AI Similarity Search) to store and retrieve image embeddings from your trained CLIP model. The system allows you to:

1. **Store embeddings** during training in a FAISS index
2. **Search for similar images** using vector similarity
3. **Locate images** on disk by their embedding similarity
4. **Scale to millions** of images efficiently

## Architecture

```
Training Flow:
┌─────────────┐
│   Images    │
└──────┬──────┘
       │
┌──────▼──────────────────────┐
│  CLIP Image Encoder         │
│  (ResNet50 + Projection)    │
└──────┬──────────────────────┘
       │
   512-dim embeddings
       │
┌──────▼──────────────────────┐
│  Normalize L2                │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  FAISS Index                 │
│  ├─ Vector: [512-dim]       │
│  └─ Metadata: Image Name     │
└─────────────────────────────┘
       │
   Saved to disk:
   ├─ image_embeddings.index
   └─ image_embeddings_metadata.json

Search Flow:
┌─────────────┐
│Query Image  │
└──────┬──────┘
       │
┌──────▼──────────────────────┐
│  Image Encoder              │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  Normalize L2                │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  FAISS Search (L2)           │
│  Returns: Top-K similar      │
│  vectors + image names       │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  Find on Disk                │
│  └─ Return file paths        │
└─────────────────────────────┘
```

## Components

### 1. `faiss_vector_store.py`

Core FAISS management classes:

#### `FAISSVectorStore`
- Manages FAISS index and metadata
- Stores/retrieves embeddings
- Performs similarity search
- Saves/loads indices to/from disk

```python
from models.faiss_vector_store import FAISSVectorStore

# Create store
store = FAISSVectorStore(embedding_dim=512)

# Add embeddings
embeddings = np.random.randn(100, 512)  # 100 images
image_names = [f'image_{i}.jpg' for i in range(100)]
store.add_embeddings(embeddings, image_names)

# Search
query_emb = np.random.randn(1, 512)
results = store.search(query_emb, k=10)

# Save/Load
store.save_index('faiss_index.index')
store.load_index('faiss_index.index')
```

#### `EmbeddingManager`
- High-level interface for managing embeddings
- Handles multiple stores (image embeddings, etc.)
- Batch operations

```python
from models.faiss_vector_store import EmbeddingManager

manager = EmbeddingManager(embedding_dim=512, index_dir='vector_store')
manager.add_image_embeddings_batch(embeddings, image_names)
manager.save_all_indices()
```

### 2. `train_with_faiss.py`

Training script with integrated FAISS storage:

```python
from train_with_faiss import CLIPTrainerWithFAISS
from models import CLIPModel

# Initialize model and trainer
model = CLIPModel()
trainer = CLIPTrainerWithFAISS(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    vector_store_dir='vector_store',
    embedding_dim=512
)

# Train and store embeddings
trainer.train(
    num_epochs=10,
    store_embeddings=True,  # Store embeddings in FAISS
    save_interval=5
)

# After training, search for similar images
results = trainer.search_similar_images(query_image, k=10)
```

**Key Features:**
- Automatically stores image embeddings in FAISS during training
- Each batch updates the FAISS index
- Saves checkpoint and FAISS indices periodically
- Supports validation loop

### 3. `retrieve_images.py`

Image retrieval utility:

```python
from retrieve_images import ImageRetriever

# Load trained model and FAISS index
retriever = ImageRetriever(
    model_checkpoint='checkpoints/clip_epoch_10.pt',
    vector_store_dir='vector_store',
    image_root_dir='dataset/coco/images'
)

# Search by image file
results = retriever.search_by_image('query_image.jpg', k=10)

# Pretty print results
retriever.print_results(results, include_disk_path=True)

# Get all indexed images
all_images = retriever.get_all_image_names()
print(f"Total indexed images: {len(all_images)}")

# Get statistics
stats = retriever.get_index_stats()
print(stats)
```

## Data Structures

### Output Format - Search Results

Each search returns a list of dictionaries:

```python
[
    {
        'rank': 1,
        'image_name': 'image_001.jpg',
        'distance': 0.0234,  # L2 distance
        'similarity': 0.977,  # Converted to similarity (higher is better)
        'found_on_disk': True,
        'disk_path': '/path/to/image_001.jpg'
    },
    {
        'rank': 2,
        'image_name': 'image_042.jpg',
        'distance': 0.0567,
        'similarity': 0.944,
        'found_on_disk': True,
        'disk_path': '/path/to/image_042.jpg'
    },
    ...
]
```

### Metadata Storage

FAISS metadata is saved in JSON:

```json
{
  "embedding_dim": 512,
  "num_vectors": 118287,
  "image_names": [
    "image_001.jpg",
    "image_002.jpg",
    ...
  ]
}
```

## DataLoader Format

The training script expects batches with this format:

```python
batch = {
    'images': tensor,           # Shape: (batch_size, 3, height, width)
    'text_tokens': tensor,      # Shape: (batch_size, max_seq_length)
    'text_mask': tensor,        # Optional, shape: (batch_size, max_seq_length)
    'image_names': list         # List of image filenames/IDs
}
```

Example custom dataset:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images_dir, captions_file):
        # Load your data
        pass
    
    def __getitem__(self, idx):
        image = load_image(...)  # Return PIL Image or tensor
        captions = load_captions(...)
        image_name = f"image_{idx}.jpg"
        
        return {
            'images': image,
            'text_tokens': caption_tokens,
            'text_mask': mask,
            'image_names': image_name
        }
```

## Workflow Example

### 1. Training with FAISS Storage

```python
from train_with_faiss import CLIPTrainerWithFAISS
from models import CLIPModel
from torch.utils.data import DataLoader

# Prepare your dataset
# Your DataLoader should return batches with format shown above
train_loader = DataLoader(your_dataset, batch_size=32, shuffle=True)

# Create model and trainer
model = CLIPModel(image_embedding_dim=512)
trainer = CLIPTrainerWithFAISS(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    vector_store_dir='vector_store'
)

# Train - embeddings are automatically stored in FAISS
trainer.train(num_epochs=10, store_embeddings=True)

# FAISS indices are saved to:
# - vector_store/image_embeddings.index
# - vector_store/image_embeddings_metadata.json
```

### 2. Searching After Training

```python
from retrieve_images import ImageRetriever

# Load the trained model and FAISS index
retriever = ImageRetriever(
    model_checkpoint='checkpoints/clip_epoch_10.pt',
    vector_store_dir='vector_store',
    image_root_dir='dataset/coco/images'
)

# Search for similar images
query_image = 'path/to/query_image.jpg'
results = retriever.search_by_image(query_image, k=10)

# Display results
retriever.print_results(results)

# Access individual results
for result in results:
    rank = result['rank']
    image_name = result['image_name']
    similarity = result['similarity']
    disk_path = result['disk_path']
    
    print(f"Rank {rank}: {image_name} (similarity: {similarity:.4f})")
    print(f"  Location: {disk_path}")
```

## Performance Characteristics

### FAISS Index Types

Currently using **IndexFlatL2** with L2 distance:

- **Storage**: ~2KB per embedding (512-dim float32)
  - 100,000 images ≈ 200MB
  - 1,000,000 images ≈ 2GB
  
- **Search Speed**: 
  - IndexFlatL2: O(n) - searches all vectors
  - For faster search on millions: Consider IndexIVFFlat or HNSW

- **Memory**: Loaded in RAM during search
  - Recommended: GPU with sufficient VRAM or use CPU for smaller indices

### Optimization Tips

1. **For millions of images**: Switch to approximate index
   ```python
   # In faiss_vector_store.py, replace:
   # self.index = faiss.IndexFlatL2(embedding_dim)
   # With:
   nlist = 100  # Number of clusters
   self.index = faiss.IndexIVFFlat(
       faiss.IndexFlatL2(embedding_dim), 
       embedding_dim, 
       nlist
   )
   ```

2. **GPU Acceleration**: Use GPU indices
   ```python
   res = faiss.StandardGpuResources()
   gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
   ```

3. **Batch Search**: Use `batch_search()` for multiple queries

## Troubleshooting

### Issue: "Embedding dimension mismatch"
- Ensure your model's embedding_dim matches vector store's embedding_dim

### Issue: "Image not found on disk"
- Check that `image_root_dir` is correct
- Verify that image names in FAISS match actual filenames

### Issue: Slow search
- Use approximate indices for large datasets
- Move index to GPU
- Use batch search instead of individual searches

### Issue: Memory error when loading index
- For large indices, use approximate methods
- Consider using GPU for indexing

## Advanced Usage

### Custom Similarity Search

```python
from models.faiss_vector_store import FAISSVectorStore

store = FAISSVectorStore(embedding_dim=512)

# Search with custom K
top_100_results = store.search(query_embedding, k=100)

# Batch search
queries = np.random.randn(10, 512)  # 10 queries
batch_results = store.batch_search(queries, k=10)
```

### Combining with Text Search

```python
# Get image and text embeddings
image_embeddings, text_embeddings = model(images, text_tokens)

# Store only image embeddings in FAISS
# Store text embeddings separately for text search
# Then combine results (top-K from each) for better retrieval
```

## Summary

The FAISS integration provides:

✓ Efficient storage of image embeddings  
✓ Fast similarity search across millions of images  
✓ Easy integration with training loop  
✓ Automatic disk persistence  
✓ Mapping from embeddings back to image files  
✓ Scalable architecture for large-scale retrieval  

This enables your model to not just generate embeddings, but actually retrieve similar images from your dataset efficiently.
