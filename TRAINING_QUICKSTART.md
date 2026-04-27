# Quick Training Quickstart

## 30-Second Start (NEW CONSOLIDATED SYSTEM)

```bash
# Start interactive training with consolidated trainer
python train_model.py
```

Follow prompts and start training! Model is saved automatically.

---

## Training in 5 Minutes

### Step 1: Download Data (Pick One)

```bash
# Small dataset (2K images) - FASTEST
cd dataset
python download.py     # Then select small dataset option in train_model.py

# Or full dataset (118K images) - BEST RESULTS
python download.py     # Then select full dataset option in train_model.py
cd ..
```

### Step 2: Start Training

```bash
# Interactive (Recommended)
python train_model.py

# Select:
# 1. Model type (CLIP or SimCLR)
# 2. Training scale (Small, Medium, or Large)
# 3. Dataset size (Small or Full COCO)
```

### Step 3: Use Trained Model

After training, your model is saved automatically:
```bash
# Search for similar images anytime
python retrieve_similar_images.py

# Select which model to use
# No retraining needed!
```

### Step 4: View Results

After training:
```
✓ Model saved to: trained_models/clip_20240101_120000/
✓ Metadata saved to: trained_models/clip_20240101_120000/metadata.json
✓ Embeddings in: vector_store/
✓ Checkpoints in: checkpoints/
```

---

## Three Training Options

### Option 1: Interactive (Easiest)
```bash
python train.py
```
- ✅ Guided step-by-step
- ✅ No coding needed
- ✅ Perfect for beginners

### Option 2: Command Line
```bash
python train.py --model clip --scale medium --data coco_small
```
- ✅ Quick to type
- ✅ Scriptable
- ✅ For experienced users

### Option 3: Python Code
```python
from models import CLIPModel
from train_with_faiss import CLIPTrainerWithFAISS

model = CLIPModel()
trainer = CLIPTrainerWithFAISS(model, train_loader, val_loader)
trainer.train(num_epochs=10, store_embeddings=True)
```
- ✅ Full control
- ✅ Custom loops
- ✅ For developers

---

## Choosing Your Configuration

### Model Type
- **CLIP**: Use text to find images ← **Text-to-Image Search**
- **SimCLR**: Use images to find similar images ← **Image-to-Image Search**

### Training Scale
| Scale | Time | Best For |
|-------|------|----------|
| Small | 5 min | Testing |
| Medium | 30 min | Development |
| Large | 4 hrs | Production |

### Data Size
| Size | Images | Time | Quality |
|------|--------|------|---------|
| Small | 5K | Fast | Good |
| Medium | 2K | Very Fast | Decent |
| Large | 118K | Slow | Best |

---

## What Happens During Training

```
Training Started...
├─ Load data (batches)
├─ For each epoch:
│  ├─ For each batch:
│  │  ├─ Forward pass → embeddings
│  │  ├─ Calculate loss
│  │  ├─ Backward pass
│  │  └─ Update weights
│  ├─ Store embeddings in FAISS
│  └─ Save checkpoint
└─ Done!
```

---

## Output Files

After training, you get:

**Model Weights:**
```
checkpoints/
└── clip_epoch_10.pt     # 512MB (model parameters)
```

**Vector Index:**
```
vector_store/
├── image_embeddings.index          # FAISS vectors (binary)
└── image_embeddings_metadata.json  # Image names
```

**Configuration:**
```
training_configs/
└── clip_coco_small_config.json     # Training parameters
```

---

## Next: Use Your Model

### Search for Similar Images

```python
from retrieve_images import ImageRetriever

# Load model
retriever = ImageRetriever(
    'checkpoints/clip_epoch_10.pt',
    'vector_store'
)

# Search
results = retriever.search_by_image('photo.jpg', k=10)

# View results
retriever.print_results(results)
```

### Output
```
Rank 1: similar_photo_1.jpg (similarity: 0.92)
Rank 2: similar_photo_2.jpg (similarity: 0.88)
...
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use smaller batch size |
| Slow training | Use GPU (check `nvidia-smi`) |
| Data not found | Download first |
| FAISS error | Install: `pip install faiss-cpu` |

---

## Tips for Best Results

1. **More epochs** = Better model
   ```python
   trainer.train(num_epochs=50)  # Instead of 10
   ```

2. **More data** = Better generalization
   - Start with small dataset
   - Graduate to full dataset

3. **Monitor loss**
   - Should decrease over time
   - If not, reduce learning rate

4. **Save intermediate checkpoints**
   - Happens automatically every epoch
   - Choose best one for deployment

---

## Common Commands

```bash
# Full interactive training
python train.py

# Quick test (2 epochs, 16 batch)
python train.py --model clip --scale small --data coco_small

# Development (10 epochs, 32 batch)
python train.py --model clip --scale medium --data coco_small

# Production (50 epochs, full data)
python train.py --model clip --scale large --data coco_full

# Search similar images after training
python retrieve_images.py --model checkpoints/clip_epoch_10.pt
```

---

## That's It!

✅ Data downloaded  
✅ Model training  
✅ Embeddings stored  
✅ Ready to search  

Enjoy! 🎉
