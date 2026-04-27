"""
Complete Example: Training CLIP Model with FAISS Vector Store

This script demonstrates the complete workflow:
1. Create/load a CLIP model
2. Train with automatic FAISS embedding storage
3. Search for similar images using the trained model and FAISS

Usage:
    python example_faiss_workflow.py
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import your models
from models import CLIPModel
from train_with_faiss import CLIPTrainerWithFAISS


# ============================================================================
# STEP 1: Create a Simple Dataset (Example)
# ============================================================================

class ExampleDataset(Dataset):
    """
    Example dataset that returns batches in the correct format.
    Replace this with your actual dataset.
    """
    
    def __init__(self, num_samples=100, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy image
        image = torch.randn(3, *self.image_size)
        
        # Generate dummy text tokens (max_seq_length=77)
        text_tokens = torch.randint(0, 10000, (77,))
        text_mask = torch.ones(77)
        
        # Image name for tracking
        image_name = f"image_{idx:06d}.jpg"
        
        return {
            'images': image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'image_names': image_name
        }


# ============================================================================
# STEP 2: Create DataLoaders
# ============================================================================

def create_dataloaders(batch_size=32, num_train=1000, num_val=100):
    """Create example dataloaders."""
    
    train_dataset = ExampleDataset(num_samples=num_train)
    val_dataset = ExampleDataset(num_samples=num_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for simplicity
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# ============================================================================
# STEP 3: Train Model with FAISS Integration
# ============================================================================

def train_model_with_faiss(
    num_epochs=3,
    batch_size=32,
    learning_rate=1e-4,
    embedding_dim=512,
):
    """
    Train CLIP model with automatic FAISS embedding storage.
    """
    
    print("=" * 70)
    print("STEP 1: Training CLIP Model with FAISS Integration")
    print("=" * 70)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)
    print(f"✓ Created train loader with {len(train_loader)} batches")
    print(f"✓ Created val loader with {len(val_loader)} batches")
    
    # Create model
    print("\nCreating CLIP model...")
    model = CLIPModel(
        image_embedding_dim=embedding_dim,
        text_embedding_dim=embedding_dim,
    )
    print(f"✓ Model created with {embedding_dim}-dim embeddings")
    
    # Create trainer with FAISS
    print("\nCreating trainer with FAISS integration...")
    trainer = CLIPTrainerWithFAISS(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        vector_store_dir='vector_store',
        embedding_dim=embedding_dim,
    )
    print("✓ Trainer initialized")
    
    # Train
    print("\nStarting training...")
    print("-" * 70)
    trainer.train(
        num_epochs=num_epochs,
        store_embeddings=True,  # Store embeddings in FAISS
        save_interval=1
    )
    print("-" * 70)
    
    # Print statistics
    print("\nTraining complete!")
    stats = trainer.get_embedding_stats()
    print(f"✓ Total embeddings stored: {stats['total_image_embeddings']}")
    print(f"✓ Embedding dimension: {stats['embedding_dimension']}")
    print(f"✓ Model saved to: checkpoints/")
    print(f"✓ FAISS indices saved to: vector_store/")
    
    return trainer


# ============================================================================
# STEP 4: Search Using Trained Model and FAISS
# ============================================================================

def search_similar_images(trainer, query_batch_idx=0, k=10):
    """
    Search for similar images using a trained model and FAISS.
    """
    
    print("\n" + "=" * 70)
    print("STEP 2: Searching for Similar Images")
    print("=" * 70)
    
    # Get a query image from validation set
    print(f"\nFetching query image from validation set (batch {query_batch_idx})...")
    _, val_loader = create_dataloaders(batch_size=32)
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx == query_batch_idx:
            query_images = batch['images']
            query_names = batch['image_names']
            break
    
    print(f"✓ Query image batch size: {len(query_images)}")
    
    # Search for first image in batch
    print(f"\nSearching for images similar to: {query_names[0]}")
    print("-" * 70)
    results = trainer.search_similar_images(query_images[0:1], k=k)
    print("-" * 70)
    
    # Print results
    print(f"\nTop {k} Similar Images:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['image_name']}")
        print(f"   Similarity Score: {result['similarity']:.4f}")
        print(f"   Distance (L2): {result['distance']:.4f}")


# ============================================================================
# STEP 5: Load and Use Trained Model Later
# ============================================================================

def load_and_search(
    model_checkpoint='checkpoints/clip_epoch_3.pt',
    vector_store_dir='vector_store',
):
    """
    Load a previously trained model and search in FAISS.
    """
    
    print("\n" + "=" * 70)
    print("STEP 3: Loading Trained Model and Searching FAISS")
    print("=" * 70)
    
    try:
        from retrieve_images import ImageRetriever
        
        print(f"\nLoading model from: {model_checkpoint}")
        print(f"Loading FAISS indices from: {vector_store_dir}")
        
        retriever = ImageRetriever(
            model_checkpoint=model_checkpoint,
            vector_store_dir=vector_store_dir,
            image_root_dir=None,  # Optional: set if you want to find images on disk
            embedding_dim=512,
        )
        
        print("✓ Model and FAISS indices loaded successfully!")
        
        # Get statistics
        stats = retriever.get_index_stats()
        print(f"\nFAISS Index Statistics:")
        print(f"  Total embeddings: {stats['total_image_embeddings']}")
        print(f"  Embedding dimension: {stats['embedding_dimension']}")
        
        # Get sample image names
        all_images = retriever.get_all_image_names()
        print(f"\nSample indexed images:")
        for img_name in all_images[:5]:
            print(f"  - {img_name}")
        
        return retriever
    
    except FileNotFoundError as e:
        print(f"\n⚠ Could not load model/indices: {e}")
        print("Make sure training has completed successfully first.")
        return None


# ============================================================================
# STEP 6: Main Workflow
# ============================================================================

def main():
    """
    Run the complete workflow:
    1. Train model with FAISS integration
    2. Search for similar images
    3. Demonstrate loading and reusing the model
    """
    
    print("\n" + "=" * 70)
    print("FAISS Vector Store Integration - Complete Example")
    print("=" * 70)
    
    # Configuration
    config = {
        'num_epochs': 3,          # Small number for example
        'batch_size': 16,
        'learning_rate': 1e-4,
        'embedding_dim': 512,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Train
    trainer = train_model_with_faiss(**config)
    
    # Step 2: Search
    search_similar_images(trainer, query_batch_idx=0, k=5)
    
    # Step 3: Load and use later
    retriever = load_and_search(
        model_checkpoint='checkpoints/clip_epoch_3.pt',
        vector_store_dir='vector_store',
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)
    print("\nWhat was accomplished:")
    print("1. ✓ Trained CLIP model on sample data")
    print("2. ✓ Stored image embeddings in FAISS during training")
    print("3. ✓ Searched for similar images using FAISS")
    print("4. ✓ Demonstrated loading and reusing the model")
    
    print("\nNext Steps:")
    print("1. Prepare your real dataset in the correct format")
    print("2. Modify ExampleDataset to load your actual images and captions")
    print("3. Train on your full dataset")
    print("4. Use ImageRetriever to search your complete dataset")
    
    print("\nKey Features:")
    print("- Embeddings are automatically stored in FAISS during training")
    print("- Search can be performed without recomputing embeddings")
    print("- Supports millions of images efficiently")
    print("- Easy integration with existing training pipeline")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
