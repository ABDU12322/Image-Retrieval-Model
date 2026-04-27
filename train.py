"""
Interactive Training Script - Train CLIP or SimCLR Model

This script provides an interactive interface to:
1. Choose model type (CLIP or SimCLR)
2. Configure training parameters
3. Train with automatic FAISS vector storage
4. Save checkpoints and embeddings

Usage:
    python train.py
"""

import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
import sys

from models import CLIPModel, SimCLRModel
from train_with_faiss import CLIPTrainerWithFAISS
from models.utils import get_device


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
    },
    'simclr': {
        'name': 'SimCLR (Image-to-Image Retrieval)',
        'description': 'Simple Contrastive Learning of Representations',
        'capabilities': [
            '✓ Image-to-Image retrieval',
            '✓ Self-supervised learning',
            '✓ No text labels required'
        ],
        'params': {
            'embedding_dim': 512,
            'projection_dim': 128,
            'hidden_dim': 2048,
            'num_negative': 4096,
            'temperature': 0.07,
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
    """Let user select model type."""
    print_section("SELECT MODEL TYPE")
    
    for i, (model_key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"\n{i}. {config['name']}")
        print(f"   {config['description']}")
        print(f"   Capabilities:")
        for cap in config['capabilities']:
            print(f"     {cap}")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1':
            return 'clip'
        elif choice == '2':
            return 'simclr'
        else:
            print("Invalid choice. Please enter 1 or 2.")

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
    print_section("SELECT DATA SOURCE")
    
    print("\n1. Small Dataset (coco_small) - 5,000 images")
    print("   → Recommended for testing")
    print("   → Fast training, good for prototyping")
    
    print("\n2. Medium Dataset (coco sampled) - 2,000 images")
    print("   → Good balance for development")
    print("   → Reasonable training time")
    
    print("\n3. Full Dataset (COCO cleaned) - 118,000+ images")
    print("   → Production-scale training")
    print("   → Significantly longer training time")
    
    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        if choice == '1':
            return 'coco_small'
        elif choice == '2':
            return 'coco_medium'
        elif choice == '3':
            return 'coco_full'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def validate_data_source(data_source):
    """Validate that selected data source exists."""
    data_paths = {
        'coco_small': Path('dataset/coco_small/images'),
        'coco_medium': Path('dataset/coco/annotations/cleaned/captions_train2017_sample_2000.json'),
        'coco_full': Path('dataset/coco/annotations/cleaned/captions_train2017.json'),
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

def create_simclr_model(model_config):
    """Create SimCLR model instance."""
    return SimCLRModel(
        embedding_dim=model_config['params']['embedding_dim'],
        projection_dim=model_config['params']['projection_dim'],
        hidden_dim=model_config['params']['hidden_dim'],
        num_negative=model_config['params']['num_negative'],
        temperature=model_config['params']['temperature'],
    )

def save_training_config(model_type, data_source, training_config, model_config, output_dir='training_configs'):
    """Save training configuration for reference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    config = {
        'model_type': model_type,
        'data_source': data_source,
        'training': training_config,
        'model_params': model_config['params'],
        'timestamp': str(Path.cwd()),
    }
    
    config_path = output_dir / f'{model_type}_{data_source}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {config_path}")

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
    
    print(f"\n🧠 Model Parameters:")
    for key, value in model_config['params'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📂 Output Directories:")
    print(f"   Checkpoints: ./checkpoints/")
    print(f"   FAISS Indices: ./vector_store/")
    print(f"   Configs: ./training_configs/")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    
    print_header("IMAGE RETRIEVAL MODEL - INTERACTIVE TRAINING")
    
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
        print("  python dataset/coco_small_download.py  (for small dataset)")
        print("  python dataset/download.py             (for full dataset)")
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
    device = get_device()
    print(f"Device: {device}")
    
    if model_type == 'clip':
        model = create_clip_model(model_config)
        print(f"✓ Created CLIP model")
    else:  # simclr
        model = create_simclr_model(model_config)
        print(f"✓ Created SimCLR model")
    
    # Print model info
    from models.utils import count_parameters
    num_params = count_parameters(model)
    print(f"✓ Total parameters: {num_params:,}")
    
    # Step 5: Create dummy dataloaders for demonstration
    print_section("PREPARING DATA")
    
    # In production, you would load actual data here
    print(f"Data source: {data_source}")
    print("⚠ Note: Using placeholder dataloaders for demonstration.")
    print("   Replace with your actual data loading code.")
    
    # For now, create dummy dataloaders
    from torch.utils.data import TensorDataset
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_texts = torch.randint(0, 10000, (100, 77))
    dummy_dataset = TensorDataset(dummy_images, dummy_texts)
    
    train_loader = DataLoader(dummy_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=train_config['batch_size'], shuffle=False)
    
    print(f"✓ Created data loaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Step 6: Create trainer
    print_section("CREATING TRAINER")
    
    if model_type == 'clip':
        trainer = CLIPTrainerWithFAISS(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            device=device,
            vector_store_dir='vector_store',
            embedding_dim=model_config['params']['image_embedding_dim'],
        )
        print("✓ Created CLIPTrainerWithFAISS")
    else:
        print("⚠ SimCLR trainer not yet implemented")
        print("  Please create SimCLRTrainerWithFAISS class")
        sys.exit(1)
    
    # Step 7: Train
    print_section("STARTING TRAINING")
    
    try:
        trainer.train(
            num_epochs=train_config['num_epochs'],
            store_embeddings=True,
            save_interval=1
        )
        
        print_section("TRAINING COMPLETED")
        print("✓ Model training finished successfully")
        
        # Save configuration
        save_training_config(model_type, data_source, train_config, model_config)
        
        # Print next steps
        print_section("NEXT STEPS")
        print("\n1. Evaluate Model:")
        print("   python retrieve_images.py --model checkpoints/clip_epoch_N.pt")
        
        print("\n2. Search for Similar Images:")
        print("   from retrieve_images import ImageRetriever")
        print("   retriever = ImageRetriever('checkpoints/clip_epoch_N.pt', 'vector_store')")
        print("   results = retriever.search_by_image('query_image.jpg', k=10)")
        
        print("\n3. View Results:")
        print("   retriever.print_results(results)")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print_section("TRAINING FAILED")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Image Retrieval Model (CLIP or SimCLR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python train.py
  
  # Direct configuration
  python train.py --model clip --scale medium --data coco_small
  
  # Non-interactive
  python train.py --model simclr --scale small --data coco_medium --non-interactive
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['clip', 'simclr'],
        help='Model type to train'
    )
    parser.add_argument(
        '--scale',
        choices=['small', 'medium', 'large'],
        help='Training scale'
    )
    parser.add_argument(
        '--data',
        choices=['coco_small', 'coco_medium', 'coco_full'],
        help='Data source'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.model and args.scale and args.data:
        # Non-interactive mode
        print_header("IMAGE RETRIEVAL MODEL - TRAINING")
        
        model_type = args.model
        training_scale = args.scale
        data_source = args.data
        
        model_config = MODEL_CONFIGS[model_type]
        train_config = TRAINING_CONFIGS[training_scale]
        
        print_training_summary(model_type, data_source, training_scale, train_config, model_config)
        
        if not args.non_interactive:
            confirm = input("\nReady to start training? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Training cancelled.")
                sys.exit(0)
    else:
        # Interactive mode
        main()
