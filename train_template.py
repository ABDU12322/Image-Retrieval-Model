"""
Training template for CLIP and SimCLR models.

This file provides a template for implementing training loops for both:
1. CLIP model (text-to-image retrieval)
2. SimCLR model (image-to-image self-supervised learning)

You can use this as a starting point and customize it according to your needs.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import CLIPModel, SimCLRModel, CLIPLoss, SimCLRLossSimplified
from models.utils import get_device, count_parameters


class CLIPTrainer:
    """
    Trainer for CLIP model (text-to-image retrieval).
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-6,
        device=None,
    ):
        """
        Initialize CLIP trainer.
        
        Args:
            model: CLIPModel instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        
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
        
        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
        )
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            # Expected batch format: (images, text_tokens, text_mask)
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
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """
        Validate the model.
        
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
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting CLIP training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
            
            # Step learning rate scheduler
            self.scheduler.step()
            
            # Save checkpoint (optional)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"clip_checkpoint_epoch_{epoch + 1}.pt")
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")


class SimCLRTrainer:
    """
    Trainer for SimCLR model (image-to-image self-supervised learning).
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-6,
        device=None,
    ):
        """
        Initialize SimCLR trainer.
        
        Args:
            model: SimCLRModel instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        
        self.model.to(self.device)
        
        # Loss function
        self.criterion = SimCLRLossSimplified(temperature=0.07)
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
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            # Expected batch format: dict with 'x_i' and 'x_j' (two augmentations)
            x_i = batch['x_i'].to(self.device)
            x_j = batch['x_j'].to(self.device)
            
            # Forward pass
            z_i, z_j = self.model(x_i, x_j)
            
            # Compute loss
            loss = self.criterion(z_i, z_j)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """
        Validate the model.
        
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
                x_i = batch['x_i'].to(self.device)
                x_j = batch['x_j'].to(self.device)
                
                z_i, z_j = self.model(x_i, x_j)
                loss = self.criterion(z_i, z_j)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting SimCLR training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
            
            # Step learning rate scheduler
            self.scheduler.step()
            
            # Save checkpoint (optional)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"simclr_checkpoint_epoch_{epoch + 1}.pt")
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")


# ============================================================================
# EXAMPLE USAGE (uncomment and modify as needed)
# ============================================================================

if __name__ == "__main__":
    # Example: Create and train CLIP model
    # 
    # from torch.utils.data import DataLoader
    # 
    # # Initialize model
    # clip_model = CLIPModel(
    #     image_embedding_dim=512,
    #     text_embedding_dim=512,
    #     vocab_size=10000,
    # )
    # 
    # # Create dummy dataloaders
    # # Your DataLoader should return batches with:
    # # - 'images': shape (batch_size, 3, 224, 224)
    # # - 'text_tokens': shape (batch_size, max_seq_length)
    # # - 'text_mask' (optional): shape (batch_size, max_seq_length)
    # train_loader = DataLoader(...)  # Your dataset here
    # val_loader = DataLoader(...)    # Your dataset here
    # 
    # # Initialize trainer
    # trainer = CLIPTrainer(
    #     model=clip_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     learning_rate=1e-4,
    # )
    # 
    # # Train
    # trainer.train(num_epochs=100)
    
    pass
