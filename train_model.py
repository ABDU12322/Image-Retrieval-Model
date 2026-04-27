"""
Consolidated Training Script - Train CLIP Model with Model Persistence

This is the MAIN training file. It provides:
1. Interactive training interface for CLIP
2. Automatic FAISS vector storage
3. Model persistence (save/load)
4. No need to retrain - just load the saved model!

USAGE:
    python train_model.py

AFTER TRAINING:
    - Trained model will be saved in ./trained_models/
    - Use retrieve_similar_images.py to search using the trained model
    - No need to retrain - just load the saved model!
"""

import torch
import torch.optim as optim
import numpy as np
import json
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import sys
from PIL import Image
import torchvision.transforms as transforms

from models import CLIPModel, CLIPLoss
from models.utils import get_device, count_parameters, tokenize_text
from models.faiss_vector_store import EmbeddingManager


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    'clip': {
        'name': 'CLIP (Text-to-Image Retrieval)',
        'description': 'Contrastive Language-Image Pre-training',
        'capabilities': [
            '✓ Text-to-Image retrieval',
            '✓ Image-to-Text retrieval',
            '✓ Joint image-text embeddings'
        ],
        'params': {
            'image_embedding_dim': 512,
            'text_embedding_dim': 512,
            'vocab_size': 10000,
            'text_max_seq_length': 77,
            'num_text_layers': 12,
            'num_text_heads': 8,
            'image_pretrained': True,
        }
    }
}

TRAINING_CONFIGS = {
    'small': {
        'name': 'Small Scale (Quick Testing)',
        'num_epochs': 2,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
    },
    'medium': {
        'name': 'Medium Scale (Standard)',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
    },
    'large': {
        'name': 'Large Scale (Production)',
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 5e-5,
        'weight_decay': 1e-6,
    }
}


# ============================================================================
# MODEL PERSISTENCE MANAGER
# ============================================================================

class ModelManager:
    """
    Manages saving and loading trained models with complete configuration.
    Saves everything needed to use the model without retraining.
    """
    
    def __init__(self, model_dir='trained_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_type, model_config, training_info, model_name=None):
        """
        Save a trained model with all necessary metadata.
        
        Args:
            model: Trained model instance
            model_type: 'clip'
            model_config: Model configuration dict
            training_info: Dict with training details (epochs, loss, etc.)
            model_name: Custom name for the model (default: auto-generated)
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        # Save model state
        model_state_path = model_path / 'model_state.pt'
        torch.save(model.state_dict(), model_state_path)
        
        # Save configuration and metadata
        metadata = {
            'model_type': model_type,
            'model_config': model_config,
            'training_info': training_info,
            'saved_date': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        metadata_path = model_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"  - Model weights: {model_state_path}")
        print(f"  - Metadata: {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_path, device=None):
        """
        Load a trained model with its configuration.
        
        Args:
            model_path: Path to saved model directory
            device: Device to load model on
            
        Returns:
            model, metadata dict
        """
        model_path = Path(model_path)
        device = device or get_device()
        
        # Load metadata
        metadata_path = model_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata['model_type']
        model_config = metadata['model_config']
        
        # Create model with saved configuration
        if model_type == 'clip':
            model = CLIPModel(
                image_embedding_dim=model_config['params']['image_embedding_dim'],
                text_embedding_dim=model_config['params']['text_embedding_dim'],
                vocab_size=model_config['params']['vocab_size'],
                text_max_seq_length=model_config['params']['text_max_seq_length'],
                num_text_layers=model_config['params']['num_text_layers'],
                num_text_heads=model_config['params']['num_text_heads'],
                image_pretrained=model_config['params']['image_pretrained'],
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model state
        model_state_path = model_path / 'model_state.pt'
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"  Type: {model_config['name']}")
        print(f"  Saved: {metadata['saved_date']}")
        
        return model, metadata
    
    def list_models(self):
        """List all saved models."""
        if not self.model_dir.exists():
            print("No saved models found.")
            return []
        
        models = sorted([d for d in self.model_dir.iterdir() if d.is_dir()])
        
        print("\n" + "="*80)
        print("SAVED MODELS")
        print("="*80)
        
        if not models:
            print("No saved models yet.")
            return []
        
        for i, model_path in enumerate(models, 1):
            metadata_file = model_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"\n{i}. {model_path.name}")
                print(f"   Type: {metadata['model_config']['name']}")
                print(f"   Saved: {metadata['saved_date']}")
                if 'training_info' in metadata:
                    info = metadata['training_info']
                    if 'num_epochs' in info:
                        print(f"   Trained for: {info['num_epochs']} epochs")
                    if 'final_loss' in info:
                        print(f"   Final Loss: {info['final_loss']:.4f}")
        
        print("\n" + "="*80 + "\n")
        return models


# ============================================================================
# TRAINER CLASSES
# ============================================================================

class CLIPTrainerWithFAISS:
    """
    CLIP Trainer with FAISS vector store integration and model persistence.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-6,
        device=None,
        vector_store_dir='vector_store',
        embedding_dim=512,
    ):
        """
        Initialize CLIP trainer with FAISS integration.
        
        Args:
            model: CLIPModel instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to use for training
            vector_store_dir: Directory for FAISS indices
            embedding_dim: Dimension of embeddings
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.embedding_dim = embedding_dim
        
        self.model.to(self.device)
        
        # Loss function (temperature is now passed from model's learnable logit_scale)
        self.criterion = CLIPLoss()
        self.criterion.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
        )
        
        # Initialize FAISS embedding manager
        self.embedding_manager = EmbeddingManager(
            embedding_dim=embedding_dim,
            index_dir=vector_store_dir
        )
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch=0, store_embeddings=True):
        """
        Train for one epoch and optionally store embeddings in FAISS.
        
        Args:
            epoch: Current epoch number
            store_embeddings: Whether to store image embeddings in FAISS
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            images = batch['images'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_mask = batch.get('text_mask', None)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)
            
            # Forward pass
            image_embeddings, text_embeddings = self.model(
                images, text_tokens, text_mask
            )
            
            # Compute loss with learnable logit_scale from model
            # This ensures training and inference use the same temperature
            loss = self.criterion(image_embeddings, text_embeddings, logit_scale=self.model.logit_scale)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Store image embeddings in FAISS if enabled
            if store_embeddings:
                image_embeddings_np = image_embeddings.detach().cpu().numpy()
                image_names = batch.get('image_names', 
                                       [f"image_{batch_idx}_{i}" for i in range(len(images))])
                self.embedding_manager.add_image_embeddings_batch(
                    image_embeddings_np, 
                    image_names
                )
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_batch_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {avg_batch_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def val_epoch(self):
        """
        Validation epoch.
        
        Returns:
            avg_loss: Average validation loss
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                text_mask = batch.get('text_mask', None)
                if text_mask is not None:
                    text_mask = text_mask.to(self.device)
                
                image_embeddings, text_embeddings = self.model(
                    images, text_tokens, text_mask
                )
                
                loss = self.criterion(image_embeddings, text_embeddings)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=10, store_embeddings=True, save_interval=5):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of training epochs
            store_embeddings: Whether to store embeddings in FAISS
            save_interval: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs on device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print()
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss = self.train_epoch(epoch=epoch, store_embeddings=store_embeddings)
            print(f"✓ Training Loss: {train_loss:.4f}")
            
            # Validation
            if self.val_loader is not None:
                val_loss = self.val_epoch()
                print(f"✓ Validation Loss: {val_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        
        # Save FAISS indices at the end
        if store_embeddings:
            print("\nSaving FAISS indices...")
            self.embedding_manager.save_all_indices()
            print("✓ FAISS indices saved")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'clip_epoch_{epoch}.pt'
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_section(text):
    """Print a formatted section."""
    print(f"\n{text}")
    print("-" * 80)

def select_model_type():
    """Get model type (CLIP only)."""
    # CLIP is the only supported model
    return 'clip'

def select_training_scale():
    """Let user select training scale."""
    print_section("SELECT TRAINING SCALE")
    
    for i, (scale_key, config) in enumerate(TRAINING_CONFIGS.items(), 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Learning Rate: {config['learning_rate']}")
    
    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        if choice == '1':
            return 'small'
        elif choice == '2':
            return 'medium'
        elif choice == '3':
            return 'large'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def select_data_source():
    """Let user select data source."""
    print_section("SELECT DATASET SIZE")
    
    print("\n1. Small Dataset - 2,000 images")
    print("   → Good for testing and development")
    print("   → Reasonable training time")
    
    print("\n2. Full Dataset - 118,000+ images")
    print("   → Production-scale training")
    print("   → Significantly longer training time")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1':
            return 'coco_medium'
        elif choice == '2':
            return 'coco_full'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def validate_data_source(data_source):
    """Validate that selected data source exists."""
    data_paths = {

        'coco_medium': Path('./dataset/coco/annotations/cleaned/captions_train2017_sample_2000.json'),
        'coco_full': Path('./dataset/coco/annotations/cleaned/captions_train2017.json'),
    }
    
    path = data_paths.get(data_source)
    if path and path.exists():
        return True
    else:
        print(f"\n⚠ Warning: Data source '{data_source}' not found at {path}")
        print("Please ensure data has been downloaded and processed.")
        return False

def create_clip_model(model_config):
    """Create CLIP model instance."""
    return CLIPModel(
        image_embedding_dim=model_config['params']['image_embedding_dim'],
        text_embedding_dim=model_config['params']['text_embedding_dim'],
        vocab_size=model_config['params']['vocab_size'],
        text_max_seq_length=model_config['params']['text_max_seq_length'],
        num_text_layers=model_config['params']['num_text_layers'],
        num_text_heads=model_config['params']['num_text_heads'],
        image_pretrained=model_config['params']['image_pretrained'],
    )



def print_training_summary(model_type, data_source, training_scale, train_config, model_config):
    """Print a summary of training configuration."""
    print_section("TRAINING SUMMARY")
    
    print(f"\n📋 Configuration:")
    print(f"   Model Type: {MODEL_CONFIGS[model_type]['name']}")
    print(f"   Data Source: {data_source}")
    print(f"   Training Scale: {TRAINING_CONFIGS[training_scale]['name']}")
    
    print(f"\n⚙️  Training Parameters:")
    print(f"   Epochs: {train_config['num_epochs']}")
    print(f"   Batch Size: {train_config['batch_size']}")
    print(f"   Learning Rate: {train_config['learning_rate']}")
    print(f"   Weight Decay: {train_config['weight_decay']}")
    
    print(f"\n📂 Output Directories:")
    print(f"   Trained Models: ./trained_models/")
    print(f"   Checkpoints: ./checkpoints/")
    print(f"   FAISS Indices: ./vector_store/")


# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

def create_data_loaders(data_source, batch_size=32, num_workers=0):
    """
    Create data loaders for training.
    
    Args:
        data_source: 'coco_small', 'coco_medium', or 'coco_full'
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader (DataLoader objects)
    """
    
    class COCODataset(torch.utils.data.Dataset):
        """Simple COCO dataset loader."""
        
        def __init__(self, image_dir, annotations_file, max_samples=None):
            self.image_dir = Path(image_dir)
            
            # Load annotations
            with open(annotations_file, 'r') as f:
                self.data = json.load(f)
            
            # Handle cleaned COCO format (id + captions)
            if 'annotations' in self.data and isinstance(self.data['annotations'], list):
                if len(self.data['annotations']) > 0:
                    first_item = self.data['annotations'][0]
                    if 'captions' in first_item and 'id' in first_item:
                        # Cleaned format: each item has 'id' and 'captions' list
                        self.annotations = self.data['annotations']
                        if max_samples:
                            self.annotations = self.annotations[:max_samples]
                        
                        # Build image info and captions mapping
                        self.img_to_captions = {}
                        for ann in self.annotations:
                            img_id = ann['id']
                            captions = ann.get('captions', [])
                            self.img_to_captions[img_id] = captions
                        
                        self.valid_images = list(self.img_to_captions.keys())
                    else:
                        # Standard COCO format
                        self.images = self.data.get('images', [])
                        annotations = self.data.get('annotations', [])
                        
                        if max_samples:
                            self.images = self.images[:max_samples]
                            annotations = annotations[:min(len(annotations), max_samples * 5)]
                        
                        # Build image to captions mapping
                        self.img_to_captions = {}
                        for ann in annotations:
                            img_id = ann['image_id']
                            if img_id not in self.img_to_captions:
                                self.img_to_captions[img_id] = []
                            self.img_to_captions[img_id].append(ann['caption'])
                        
                        # Filter images that have captions
                        self.valid_images = [img for img in self.images if img['id'] in self.img_to_captions]
                else:
                    raise ValueError("Annotations file is empty")
            else:
                raise ValueError("Unsupported annotations format")
            
            if len(self.valid_images) == 0:
                raise ValueError("No valid images found in annotations")
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        def __len__(self):
            return len(self.valid_images)
        
        def __getitem__(self, idx):
            # For cleaned format, use image ID directly
            if isinstance(self.valid_images[0], (int, np.integer)):
                img_id = self.valid_images[idx]
                file_name = f"{img_id:012d}.jpg"
            else:
                # For standard format, use image dict
                img_info = self.valid_images[idx]
                img_id = img_info['id']
                file_name = img_info['file_name']
            
            # Load image
            img_path = self.image_dir / file_name
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                # Return black image if loading fails
                image = torch.zeros(3, 224, 224)
            
            # Get a random caption
            captions = self.img_to_captions.get(img_id, ["image"])
            caption = captions[0] if captions else "image"
            
            # Tokenize using BERT tokenizer
            tokenized = tokenize_text(caption, max_length=77)
            text_tokens = tokenized['input_ids'][0]  # Remove batch dimension
            text_mask = tokenized['attention_mask'][0]  # Remove batch dimension
            
            return {
                'images': image,
                'text_tokens': text_tokens,
                'text_mask': text_mask,
                'image_names': file_name,
            }
    
    # Determine paths based on data source
    if data_source == 'coco_medium':
        image_dir = Path('dataset/coco/train2017')
        annotations_file = Path('dataset/coco/annotations/cleaned/captions_train2017_sample_2000.json')
        max_samples = 2000
        
    elif data_source == 'coco_full':
        image_dir = Path('dataset/coco/train2017')
        annotations_file = Path('dataset/coco/annotations/cleaned/captions_train2017.json')
        max_samples = None
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    # Verify paths exist
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Create dataset
    dataset = COCODataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
        max_samples=max_samples
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. No valid images found.")
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    
    print_header("IMAGE RETRIEVAL MODEL - TRAINING (train_model.py)")
    print("\nThis script trains and saves your model for later use.")
    print("After training, use retrieve_similar_images.py to search!\n")
    
    # Step 1: Select model type
    model_type = select_model_type()
    model_config = MODEL_CONFIGS[model_type]
    print(f"\n✓ Selected: {model_config['name']}")
    
    # Step 2: Select training scale
    training_scale = select_training_scale()
    train_config = TRAINING_CONFIGS[training_scale]
    print(f"\n✓ Selected: {train_config['name']}")
    
    # Step 3: Select data source
    data_source = select_data_source()
    
    # Validate data source
    if not validate_data_source(data_source):
        print("\n❌ Cannot proceed without valid data source.")
        print("\nPlease run the following to download data:")
        print("  python dataset/download.py  (to download COCO dataset)")
        sys.exit(1)
    
    print(f"\n✓ Selected: {data_source}")
    
    # Print summary
    print_training_summary(model_type, data_source, training_scale, train_config, model_config)
    
    # Confirm before proceeding
    print("\n" + "-" * 80)
    confirm = input("Ready to start training? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Training cancelled.")
        sys.exit(0)
    
    # Step 4: Create model
    print_section("INITIALIZING MODEL")
    
    model = create_clip_model(model_config)
    
    device = get_device()
    print(f"✓ Model created")
    print(f"✓ Device: {device}")
    print(f"✓ Parameters: {count_parameters(model):,}")
    
    # Step 5: Create data loaders
    print_section("PREPARING DATA")
    print(f"Loading data from: {data_source}")
    
    try:
        train_loader, val_loader = create_data_loaders(
            data_source=data_source,
            batch_size=train_config['batch_size'],
            num_workers=0  # Adjust based on your system
        )
        print(f"✓ Data loaded successfully")
        print(f"  - Training batches: {len(train_loader)}")
        if val_loader:
            print(f"  - Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        print("\nPlease ensure:")
        print("  1. Dataset exists at the expected location")
        print("  2. Annotations files are present")
        print("\nTo download data, run:")
        print("  python dataset/download.py             (to download COCO dataset)")
        sys.exit(1)
    
    # Step 6: Train
    print_section("STARTING TRAINING")
    
    try:
        trainer = CLIPTrainerWithFAISS(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            device=device,
        )
        
        trainer.train(
            num_epochs=train_config['num_epochs'],
            store_embeddings=True,
            save_interval=5
        )
        
        # Step 7: Save trained model
        print_section("SAVING MODEL")
        
        model_manager = ModelManager()
        
        training_info = {
            'num_epochs': train_config['num_epochs'],
            'batch_size': train_config['batch_size'],
            'learning_rate': train_config['learning_rate'],
            'data_source': data_source,
            'training_scale': training_scale,
            'final_loss': trainer.train_losses[-1] if trainer.train_losses else None,
            'device': str(device),
        }
        
        model_path = model_manager.save_model(
            model=model,
            model_type=model_type,
            model_config=model_config,
            training_info=training_info,
        )
        
        print_section("TRAINING COMPLETE!")
        print(f"\n✓ Model saved to: {model_path}")
        print(f"\n📝 Next Steps:")
        print(f"   1. Use retrieve_similar_images.py to search with your trained model")
        print(f"   2. No need to retrain - model is saved and ready to use!")
        print(f"\n💡 Quick Start:")
        print(f"   python retrieve_similar_images.py")
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
