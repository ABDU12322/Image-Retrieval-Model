"""
FAISS Vector Store for managing image embeddings and similarity search.

This module handles:
- Storing image embeddings in FAISS index
- Managing metadata (image names/IDs) associated with vectors
- Performing similarity search to find similar images
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class FAISSVectorStore:
    """
    FAISS-based vector store for image embeddings.
    
    Each image embedding is stored in a FAISS index along with metadata
    (image filename/ID) for later retrieval.
    """
    
    def __init__(self, embedding_dim: int = 512, index_path: Optional[str] = None):
        """
        Initialize FAISS Vector Store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to load existing index (optional)
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = None
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            # Create a new FAISS index
            # Using IndexFlatL2 for L2 distance, or IndexFlatIP for cosine similarity
            # For normalized embeddings, cosine similarity is better
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = []  # List of image names/IDs
    
    def add_embeddings(self, embeddings: np.ndarray, image_names: List[str]) -> None:
        """
        Add embeddings to FAISS index.
        
        Args:
            embeddings: Array of shape (num_images, embedding_dim)
            image_names: List of image names/IDs corresponding to embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(embeddings) != len(image_names):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not match "
                f"number of image names ({len(image_names)})"
            )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata.extend(image_names)
        
        print(f"✓ Added {len(embeddings)} embeddings to FAISS index")
        print(f"  Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search for similar embeddings in FAISS index.
        
        Args:
            query_embedding: Query embedding of shape (1, embedding_dim) or (embedding_dim,)
            k: Number of nearest neighbors to return
            
        Returns:
            List of dicts with keys: 'rank', 'image_name', 'distance'
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Convert to results with metadata
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                results.append({
                    'rank': rank + 1,
                    'image_name': self.metadata[idx],
                    'distance': float(distance),
                    'similarity': 1.0 / (1.0 + float(distance))  # Convert distance to similarity
                })
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple query embeddings.
        
        Args:
            query_embeddings: Array of shape (num_queries, embedding_dim)
            k: Number of nearest neighbors to return
            
        Returns:
            List of results for each query
        """
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        # Convert to results
        all_results = []
        for query_idx in range(len(query_embeddings)):
            results = []
            for rank, (distance, idx) in enumerate(zip(distances[query_idx], indices[query_idx])):
                if idx >= 0:
                    results.append({
                        'rank': rank + 1,
                        'image_name': self.metadata[idx],
                        'distance': float(distance),
                        'similarity': 1.0 / (1.0 + float(distance))
                    })
            all_results.append(results)
        
        return all_results
    
    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON (optional, auto-generated if not provided)
        """
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"✓ Saved FAISS index to: {index_path}")
        
        # Save metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.index', '_metadata.json')
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_vectors': self.index.ntotal,
            'image_names': self.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to: {metadata_path}")
        
        self.metadata_path = metadata_path
    
    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON (optional, auto-generated if not provided)
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index from: {index_path}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.index', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            self.metadata = metadata_dict['image_names']
            print(f"✓ Loaded metadata from: {metadata_path}")
        else:
            print(f"⚠ Warning: Metadata file not found at {metadata_path}")
            self.metadata = []
        
        self.metadata_path = metadata_path
        self.index_path = index_path
    
    def get_size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def get_metadata(self, idx: int) -> str:
        """Get image name/ID for a given index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None


class EmbeddingManager:
    """
    Manager for handling embeddings during training and inference.
    
    Collects embeddings from model outputs and stores them in FAISS.
    """
    
    def __init__(self, embedding_dim: int = 512, index_dir: str = 'vector_store'):
        """
        Initialize Embedding Manager.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_dir: Directory to store FAISS indices
        """
        self.embedding_dim = embedding_dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate stores for different purposes
        self.image_embeddings = FAISSVectorStore(embedding_dim)
        self.image_index_path = self.index_dir / 'image_embeddings.index'
        self.image_metadata_path = self.index_dir / 'image_embeddings_metadata.json'
    
    def add_image_embeddings_batch(
        self,
        embeddings: np.ndarray,
        image_names: List[str]
    ) -> None:
        """
        Add image embeddings from a batch.
        
        Args:
            embeddings: Array of shape (batch_size, embedding_dim)
            image_names: List of image filenames/IDs
        """
        self.image_embeddings.add_embeddings(embeddings, image_names)
    
    def search_similar_images(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Dict]:
        """
        Search for similar images using a query embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            
        Returns:
            List of similar images with distances
        """
        return self.image_embeddings.search(query_embedding, k)
    
    def save_all_indices(self) -> None:
        """Save all FAISS indices to disk."""
        self.image_embeddings.save_index(
            str(self.image_index_path),
            str(self.image_metadata_path)
        )
    
    def load_all_indices(self) -> None:
        """Load all FAISS indices from disk."""
        if self.image_index_path.exists():
            self.image_embeddings.load_index(
                str(self.image_index_path),
                str(self.image_metadata_path)
            )
    
    def get_image_store(self) -> FAISSVectorStore:
        """Get the image embeddings store."""
        return self.image_embeddings
