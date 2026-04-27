"""
Training script with FAISS vector store integration for image embeddings.

This script demonstrates how to:
1. Train a CLIP model
2. Store image embeddings in FAISS during training
3. Perform similarity search after training
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from models import CLIPModel, CLIPLoss
from models.utils import get_device, count_parameters
from models.faiss_vector_store import EmbeddingManager


class CLIPTrainerWithFAISS:
    """
    CLIP Trainer with FAISS vector store integration.
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
        
        # Loss function
        self.criterion = CLIPLoss(temperature=0.07)
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
            
            # Compute loss
            loss = self.criterion(image_embeddings, text_embeddings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Store image embeddings in FAISS if enabled
            if store_embeddings:
                image_embeddings_np = image_embeddings.detach().cpu().numpy()
                
                # Get image filenames from batch
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
    
    def search_similar_images(self, query_image_tensor, k=10):
        """
        Search for similar images using a query image.
        
        Args:
            query_image_tensor: Query image tensor of shape (3, H, W) or (1, 3, H, W)
            k: Number of similar images to return
            
        Returns:
            List of similar images with distances
        """
        # Ensure batch dimension
        if query_image_tensor.ndim == 3:
            query_image_tensor = query_image_tensor.unsqueeze(0)
        
        # Get image embedding
        self.model.eval()
        with torch.no_grad():
            query_image_tensor = query_image_tensor.to(self.device)
            query_embedding = self.model.get_image_embeddings(query_image_tensor)
            query_embedding = query_embedding.detach().cpu().numpy()
        
        # Search FAISS
        results = self.embedding_manager.search_similar_images(query_embedding, k=k)
        
        return results
    
    def get_embedding_stats(self):
        """Get statistics about stored embeddings."""
        image_store = self.embedding_manager.get_image_store()
        return {
            'total_image_embeddings': image_store.get_size(),
            'embedding_dimension': image_store.embedding_dim,
        }


# Example usage
if __name__ == '__main__':
    # This is a template - adjust based on your actual dataset and configuration
    
    print("CLIP Model Training with FAISS Vector Store")
    print("=" * 60)
    print("\nNote: This is a template. To use it:")
    print("1. Prepare your DataLoader with batch format:")
    print("   {'images': tensor, 'text_tokens': tensor, 'text_mask': tensor, 'image_names': list}")
    print("2. Initialize the trainer")
    print("3. Call trainer.train()")
    print("\nExample:")
    print("  model = CLIPModel()")
    print("  trainer = CLIPTrainerWithFAISS(model, train_loader, val_loader)")
    print("  trainer.train(num_epochs=10, store_embeddings=True)")
    print("\nAfter training, search for similar images:")
    print("  results = trainer.search_similar_images(query_image, k=10)")
