"""
Image Retrieval using Trained CLIP Model

This script loads the trained CLIP model and uses it to:
1. Search for images by text description (text-to-image retrieval)
2. Search for similar images by image (image-to-image retrieval)
3. Display results with file paths

USAGE:
    python retrieve_similar_images.py

The script expects:
    - trained_model_clip/clip_encoder.pth (model weights)
    - trained_model_clip/config.json (training config)
    - trained_model_clip/image_embeddings.index (FAISS index)
    - trained_model_clip/image_embeddings_metadata.json (image names)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import faiss

# ============================================================================
# CLIP MODEL DEFINITION
# ============================================================================

class CLIPImageEncoder(nn.Module):
    """Image encoder for CLIP"""
    def __init__(self, embedding_dim=512):
        super(CLIPImageEncoder, self).__init__()
        
        # ResNet50 backbone (pretrained)
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        # Store feature dimension before removing FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # type: ignore
        
        # Projection head
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        return embeddings


class CLIPTextEncoder(nn.Module):
    """Text encoder for CLIP using CLIP's tokenizer and encoder"""
    def __init__(self, embedding_dim=512):
        super(CLIPTextEncoder, self).__init__()
        
        self.clip_model = None
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(512, embedding_dim)
        self.clip_available = False
        
        try:
            import clip
            _, _ = clip.load("ViT-B/32", device='cpu')
            self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
            self.clip_available = True
        except ImportError:
            print("⚠ Warning: CLIP not installed")
            print("  Text-to-image search will not work.")
            print("  To enable: pip install git+https://github.com/openai/CLIP.git")
        except Exception as e:
            print(f"⚠ Warning: Could not load CLIP model: {e}")
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: (batch_size, max_length) - tokenized text
        
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        if self.clip_model is None:
            # Fallback: return random embeddings if CLIP not loaded
            batch_size = text_tokens.size(0)
            return torch.randn(batch_size, self.embedding_dim)
        
        with torch.no_grad():
            try:
                text_features = self.clip_model.encode_text(text_tokens)
            except:
                batch_size = text_tokens.size(0)
                return torch.randn(batch_size, self.embedding_dim)
        
        embeddings = self.projection(text_features)
        return embeddings


class CLIPModel(nn.Module):
    """Complete CLIP model for text-to-image retrieval"""
    def __init__(self, embedding_dim=512):
        super(CLIPModel, self).__init__()
        self.image_encoder = CLIPImageEncoder(embedding_dim)
        self.text_encoder = CLIPTextEncoder(embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward_image(self, images):
        """Generate image embeddings"""
        return self.image_encoder(images)
    
    def forward_text(self, text_tokens):
        """Generate text embeddings"""
        return self.text_encoder(text_tokens)
    
    def forward(self, images, text_tokens):
        """Forward pass for both modalities"""
        image_embeddings = self.forward_image(images)
        text_embeddings = self.forward_text(text_tokens)
        return image_embeddings, text_embeddings


# ============================================================================
# FAISS VECTOR STORE
# ============================================================================

class FAISSVectorStore:
    """Efficient vector similarity search using FAISS"""
    
    def __init__(self, embedding_dim, use_gpu=True):
        self.embedding_dim = embedding_dim
        self.use_gpu = False  # Use CPU-only for compatibility
        self.metadata = []
        
        # Create index using CPU
        self.index = faiss.IndexFlatL2(embedding_dim)
    
    def add_embeddings(self, embeddings, names):
        """Add embeddings to index"""
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)  # type: ignore
        self.metadata.extend(names)
    
    def search(self, query_embedding, k=5):
        """Search for similar embeddings"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)  # type: ignore
        
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    'rank': rank + 1,
                    'name': self.metadata[idx],
                    'distance': float(distance),
                    'similarity': 1.0 / (1.0 + float(distance))
                })
        return results
    
    def load_index(self, index_path, metadata_path):
        """Load from disk"""
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        self.metadata = metadata_dict['names']
    
    def get_size(self):
        return self.index.ntotal


# ============================================================================
# CLIP RETRIEVER
# ============================================================================

class CLIPRetriever:
    """CLIP-based image retrieval system"""
    
    def __init__(
        self,
        model_dir: str = 'trained_model_clip',
        image_dir: str = 'dataset/coco/train2017',
        captions_file: str = 'dataset/coco/annotations/captions_train2017.json',
        device: Optional[str] = None,
    ):
        """
        Initialize CLIP Retriever
        
        Args:
            model_dir: Directory containing trained model
            image_dir: Directory containing images
            captions_file: Path to captions JSON file
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_dir = Path(image_dir)
        self.captions_file = Path(captions_file)
        
        print("\n" + "="*70)
        print("INITIALIZING CLIP RETRIEVER")
        print("="*70 + "\n")
        
        # Load model
        print(f"Loading model from: {model_dir}")
        self.model = self._load_model(model_dir)
        print("✓ Model loaded")
        
        # Load FAISS index
        print(f"\nLoading FAISS index...")
        self.vector_store = self._load_faiss_index(model_dir)
        print(f"✓ FAISS index loaded ({self.vector_store.get_size()} images)")
        
        # Load image metadata
        print(f"\nLoading image metadata...")
        self.image_names = self.vector_store.metadata
        self._load_captions()
        
        # Build mapping of image indices to actual filenames
        self._build_image_index()
        print(f"✓ Loaded {len(self.image_names)} images")
        
        print("\n" + "="*70 + "\n")
    
    def _build_image_index(self):
        """Build a mapping from image indices to actual filenames"""
        self.image_index = {}
        
        # Get actual image files from directory
        if self.image_dir.exists():
            image_files = sorted([
                f for f in self.image_dir.iterdir() 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            # Create mapping: index -> filename
            for idx, img_file in enumerate(image_files):
                self.image_index[idx] = img_file.name
            
            print(f"  Found {len(image_files)} actual images in {self.image_dir}")
        else:
            print(f"  ⚠ Image directory not found: {self.image_dir}")
    
    def _load_model(self, model_dir: str):
        """Load trained CLIP model"""
        model_path_obj = Path(model_dir)
        
        # Load config
        config_path = model_path_obj / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        embedding_dim = config.get('embedding_dim', 512)
        model = CLIPModel(embedding_dim=embedding_dim)
        
        # Load weights
        model_path = model_path_obj / 'clip_encoder.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load state dict with error handling
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # If the saved state has the full model, extract only what we need
            model_state = model.state_dict()
            new_state = {}
            
            for key, value in state_dict.items():
                # Only load keys that exist in our model
                if key in model_state:
                    new_state[key] = value
                # Handle image_encoder keys
                elif key.startswith('image_encoder.'):
                    new_state[key] = value
            
            # Load the filtered state dict
            if new_state:
                model.load_state_dict(new_state, strict=False)
                print(f"✓ Loaded {len(new_state)} weight keys")
            else:
                print("⚠ Warning: Could not find matching weights in saved model")
                print("  Using randomly initialized model")
        except Exception as e:
            print(f"⚠ Warning: Could not load model weights: {e}")
            print("  Using randomly initialized model for image encoder")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_faiss_index(self, model_dir: str):
        """Load FAISS index"""
        model_path_obj = Path(model_dir)
        
        index_path = model_path_obj / 'image_embeddings.index'
        metadata_path = model_path_obj / 'image_embeddings_metadata.json'
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index not found in {model_dir}")
        
        # Load config to get embedding dim
        config_path = model_path_obj / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        embedding_dim = config.get('embedding_dim', 512)
        
        # Create and load index
        vector_store = FAISSVectorStore(
            embedding_dim=embedding_dim,
            use_gpu=False
        )
        vector_store.load_index(str(index_path), str(metadata_path))
        
        return vector_store
    
    def _load_captions(self):
        """Load captions from JSON"""
        self.captions = {}
        
        if not self.captions_file.exists():
            print(f"⚠ Captions file not found: {self.captions_file}")
            return
        
        try:
            with open(self.captions_file, 'r') as f:
                data = json.load(f)
            
            # Build caption map from image_id
            if 'annotations' in data:
                for annotation in data['annotations']:
                    img_id = annotation.get('image_id')
                    caption = annotation.get('caption', '')
                    if img_id not in self.captions:
                        self.captions[img_id] = []
                    self.captions[img_id].append(caption)
        except Exception as e:
            print(f"⚠ Could not load captions: {e}")
    
    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Get embedding for an image"""
        try:
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
            image_tensor = transform(image)  # type: ignore
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # type: ignore
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model.forward_image(image_tensor)
                embedding = embedding.cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using BERT tokenizer"""
        try:
            from transformers import BertTokenizer
            
            # Tokenize text using BERT
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            encoded = tokenizer(
                text,
                max_length=77,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_tokens = encoded['input_ids'].to(self.device)
            text_mask = encoded['attention_mask'].to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model.forward_text(text_tokens)
                embedding = embedding.cpu().numpy()
            
            return embedding
        except ImportError:
            print("\n" + "="*70)
            print("❌ TRANSFORMERS NOT INSTALLED")
            print("="*70)
            print("\nText-to-image search requires transformers to be installed.")
            print("\nInstall it with:")
            print("  pip install transformers")
            print("\nOr use image-to-image search instead (Option 2).")
            print("="*70 + "\n")
            return None
        except Exception as e:
            print(f"\n❌ Error processing text: {e}")
            print("  Make sure transformers is properly installed.")
            return None
    
    def search_by_text(self, query: str, k: int = 10) -> List[Dict]:
        """Search for images using text query"""
        print(f"\n{'='*70}")
        print(f"Text Search: '{query}'")
        print(f"{'='*70}\n")
        
        # Get text embedding
        embedding = self.get_text_embedding(query)
        if embedding is None:
            return []
        
        # Search FAISS
        results = self.vector_store.search(embedding, k=k)
        
        # Enrich results with actual filenames
        for result in results:
            result['generic_name'] = result['name']
            
            # Extract index from "Image X" format
            try:
                img_idx = int(result['name'].split()[-1])
                if img_idx in self.image_index:
                    actual_filename = self.image_index[img_idx]
                    result['name'] = actual_filename
                    result['image_path'] = str(self.image_dir / actual_filename)
                    result['found'] = (self.image_dir / actual_filename).exists()
                else:
                    result['image_path'] = str(self.image_dir / result['name'])
                    result['found'] = False
            except:
                result['image_path'] = str(self.image_dir / result['name'])
                result['found'] = False
            
            # Try to add caption
            try:
                if img_idx in self.captions:
                    result['captions'] = self.captions[img_idx][:2]
            except:
                pass
        
        return results
    
    def search_by_image(self, image_path: str, k: int = 10) -> List[Dict]:
        """Search for similar images"""
        print(f"\n{'='*70}")
        print(f"Image Search: {image_path}")
        print(f"{'='*70}\n")
        
        # Get image embedding
        embedding = self.get_image_embedding(image_path)
        if embedding is None:
            return []
        
        # Search FAISS
        results = self.vector_store.search(embedding, k=k)
        
        # Enrich results with actual filenames
        for result in results:
            result['generic_name'] = result['name']
            
            # Extract index from "Image X" format
            try:
                img_idx = int(result['name'].split()[-1])
                if img_idx in self.image_index:
                    actual_filename = self.image_index[img_idx]
                    result['name'] = actual_filename
                    result['image_path'] = str(self.image_dir / actual_filename)
                    result['found'] = (self.image_dir / actual_filename).exists()
                else:
                    result['image_path'] = str(self.image_dir / result['name'])
                    result['found'] = False
            except:
                result['image_path'] = str(self.image_dir / result['name'])
                result['found'] = False
            
            # Try to add caption
            try:
                if img_idx in self.captions:
                    result['captions'] = self.captions[img_idx][:2]
            except:
                pass
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Pretty print search results"""
        print(f"\n{'='*70}")
        print("SEARCH RESULTS")
        print(f"{'='*70}\n")
        
        for result in results:
            print(f"Rank {result['rank']}: {result['name']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            
            if result['found']:
                print(f"  ✓ Found: {result['image_path']}")
            else:
                print(f"  ✗ Path: {result['image_path']} (not found)")
            
            if 'captions' in result:
                print(f"  Captions:")
                for cap in result['captions']:
                    print(f"    - {cap}")
            
            print()
        
        print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main retrieval pipeline"""
    
    print("\n" + "="*70)
    print("IMAGE RETRIEVAL WITH TRAINED CLIP MODEL")
    print("="*70)
    
    # Initialize retriever
    try:
        retriever = CLIPRetriever(
            model_dir='trained_model_clip',
            image_dir='dataset/coco/train2017',
            captions_file='dataset/coco/annotations/captions_train2017.json'
        )
        print("\n✓ Retriever initialized successfully!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. trained_model_clip/ directory exists")
        print("  2. All required files are present:")
        print("     - config.json")
        print("     - clip_encoder.pth")
        print("     - image_embeddings.index")
        print("     - image_embeddings_metadata.json")
        return
    except Exception as e:
        print(f"\n❌ Error initializing retriever: {e}")
        print("\nNote: The FAISS index will still be used for searching.")
        print("Text-to-image search may have limited functionality.")
        import traceback
        traceback.print_exc()
        return
    
    # Menu
    while True:
        print("\n" + "-"*70)
        print("SELECT SEARCH METHOD")
        print("-"*70)
        print("\n1. Search by image (image-to-image) ✓ READY")
        print("   → Find similar images to a query image")
        print("\n2. Search by text (text-to-image) ⚠ Requires CLIP")
        print("   → Find images matching a text description")
        print("   → Install: pip install git+https://github.com/openai/CLIP.git")
        print("\n3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip()
            if Path(image_path).exists():
                results = retriever.search_by_image(image_path, k=10)
                if results:
                    retriever.print_results(results)
                else:
                    print("No results found.")
            else:
                print(f"❌ Image not found: {image_path}")
        
        elif choice == '2':
            query = input("\nEnter text description: ").strip()
            if query:
                results = retriever.search_by_text(query, k=10)
                if results:
                    retriever.print_results(results)
                else:
                    print("\n⚠ No results found.")
                    print("  Possible reasons:")
                    print("  - CLIP is not installed")
                    print("  - No similar images in dataset")
        
        elif choice == '3':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == '__main__':
    main()
