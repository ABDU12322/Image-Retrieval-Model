"""
Image Retrieval Utility - Search FAISS and retrieve images from disk.

This utility allows you to:
1. Load trained FAISS indices
2. Search for similar images
3. Locate images in the file system
4. Visualize results
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json

from models import CLIPModel
from models.faiss_vector_store import FAISSVectorStore, EmbeddingManager
from models.utils import get_device


class ImageRetriever:
    """
    Image retrieval system using FAISS and trained models.
    """
    
    def __init__(
        self,
        model_checkpoint: str,
        vector_store_dir: str = 'vector_store',
        image_root_dir: Optional[str] = None,
        embedding_dim: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize Image Retriever.
        
        Args:
            model_checkpoint: Path to trained CLIP model checkpoint
            vector_store_dir: Directory containing FAISS indices
            image_root_dir: Root directory for images (used to locate images)
            embedding_dim: Dimension of embeddings
            device: Device to use
        """
        self.device = device or get_device()
        self.embedding_dim = embedding_dim
        self.image_root_dir = Path(image_root_dir) if image_root_dir else None
        
        # Load model
        print(f"Loading model from: {model_checkpoint}")
        self.model = CLIPModel(image_embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_checkpoint, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded")
        
        # Load FAISS embedding manager
        print(f"Loading FAISS indices from: {vector_store_dir}")
        self.embedding_manager = EmbeddingManager(
            embedding_dim=embedding_dim,
            index_dir=vector_store_dir
        )
        self.embedding_manager.load_all_indices()
        print("✓ FAISS indices loaded")
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get embedding for an image from disk.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding as numpy array
        """
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model.get_image_embeddings(image_tensor)
                embedding = embedding.detach().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def search_by_image(self, image_path: str, k: int = 10) -> List[Dict]:
        """
        Search for similar images using a query image file.
        
        Args:
            image_path: Path to query image
            k: Number of results
            
        Returns:
            List of similar images with metadata
        """
        print(f"Searching for images similar to: {image_path}")
        
        # Get query embedding
        query_embedding = self.get_image_embedding(image_path)
        if query_embedding is None:
            return []
        
        # Search FAISS
        results = self.embedding_manager.search_similar_images(query_embedding, k=k)
        
        # Enrich results with file paths
        enriched_results = []
        for result in results:
            result['found_on_disk'] = False
            result['disk_path'] = None
            
            # Try to find image on disk
            if self.image_root_dir:
                image_path_candidates = list(self.image_root_dir.rglob(result['image_name']))
                if image_path_candidates:
                    result['disk_path'] = str(image_path_candidates[0])
                    result['found_on_disk'] = True
            
            enriched_results.append(result)
        
        return enriched_results
    
    def search_by_embedding(self, embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of similar images
        """
        results = self.embedding_manager.search_similar_images(embedding, k=k)
        
        # Enrich with file paths
        enriched_results = []
        for result in results:
            result['found_on_disk'] = False
            result['disk_path'] = None
            
            if self.image_root_dir:
                image_path_candidates = list(self.image_root_dir.rglob(result['image_name']))
                if image_path_candidates:
                    result['disk_path'] = str(image_path_candidates[0])
                    result['found_on_disk'] = True
            
            enriched_results.append(result)
        
        return enriched_results
    
    def print_results(self, results: List[Dict], include_disk_path: bool = True):
        """
        Pretty print search results.
        
        Args:
            results: Search results
            include_disk_path: Whether to show disk paths
        """
        print(f"\n{'='*80}")
        print("SEARCH RESULTS")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\nRank {result['rank']}:")
            print(f"  Image: {result['image_name']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Distance: {result['distance']:.4f}")
            
            if include_disk_path:
                if result['found_on_disk']:
                    print(f"  ✓ Found at: {result['disk_path']}")
                else:
                    print(f"  ✗ Not found on disk")
        
        print(f"\n{'='*80}\n")
    
    def get_all_image_names(self) -> List[str]:
        """Get list of all indexed image names."""
        image_store = self.embedding_manager.get_image_store()
        return image_store.metadata
    
    def get_index_stats(self) -> Dict:
        """Get statistics about FAISS index."""
        stats = self.embedding_manager.get_embedding_stats()
        stats['image_root_dir'] = str(self.image_root_dir) if self.image_root_dir else None
        return stats


def interactive_search():
    """
    Interactive search mode.
    
    Allows user to search for images interactively.
    """
    print("Image Retrieval System - Interactive Mode")
    print("=" * 60)
    print("\nTo use this script:")
    print("1. Make sure you have trained a CLIP model and saved embeddings in FAISS")
    print("2. Modify the paths below to match your setup:")
    print("   - model_checkpoint: path to your trained model")
    print("   - vector_store_dir: directory containing FAISS indices")
    print("   - image_root_dir: root directory of your images")
    print("\nExample:")
    print("  retriever = ImageRetriever(")
    print("      model_checkpoint='checkpoints/clip_epoch_10.pt',")
    print("      vector_store_dir='vector_store',")
    print("      image_root_dir='dataset/coco/images'")
    print("  )")
    print("  results = retriever.search_by_image('path/to/query_image.jpg', k=10)")
    print("  retriever.print_results(results)")


if __name__ == '__main__':
    interactive_search()
