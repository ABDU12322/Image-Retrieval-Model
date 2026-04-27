# Quick Reference - Image Retrieval Training & Search

## 🎯 Two Files to Know

### 1. Training 
**File:** `train_model.py`  
**Run:** `python train_model.py`  
**Does:** Train and save your model (ONE TIME)  
**Saves to:** `trained_models/` (never delete!)  

### 2. Searching
**File:** `retrieve_similar_images.py`  
**Run:** `python retrieve_similar_images.py`  
**Does:** Load saved model and search (ANYTIME after training)  
**Uses:** Saved model from `trained_models/`  

---

## 📋 Complete Workflow

```
┌─────────────────────────────────────┐
│   STEP 1: Train (First Time)        │
│                                     │
│   $ python train_model.py           │
│                                     │
│   • Select model type (CLIP/SimCLR) │
│   • Select training scale           │
│   • Select dataset size             │
│   • Wait for training...            │
│   • Model auto-saved!               │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   STEP 2: Search (Anytime!)         │
│                                     │
│   $ python retrieve_similar_images.py│
│                                     │
│   • Select which model to use       │
│   • Choose search method            │
│   • Get results instantly!          │
│   • No retraining needed!           │
└─────────────────────────────────────┘
```

---

## 🗂️ What Gets Saved (Your Model Archive)

After running `train_model.py`:

```
trained_models/
├── clip_20240101_120000/          ← Your 1st model
│   ├── model_state.pt              (weights)
│   └── metadata.json               (config + training info)
│
├── simclr_20240102_150000/        ← Your 2nd model  
│   ├── model_state.pt
│   └── metadata.json
│
└── ... (more models)
```

**Each model is completely self-contained!**
- Can load anytime
- Includes all config needed
- Shows training details
- Compare results between models

---

## ⚡ Key Improvements

### Before (Confusing)
- ❌ Multiple training files (`train.py`, `train_with_faiss.py`, `train_template.py`)
- ❌ Models not saved properly
- ❌ Had to retrain every time
- ❌ No way to compare trained models
- ❌ Unclear which file to run

### After (Simple!)
- ✅ ONE training file: `train_model.py`
- ✅ ONE retrieval file: `retrieve_similar_images.py`
- ✅ Models auto-saved with all config
- ✅ Reuse trained models instantly
- ✅ Switch between models with one command
- ✅ Clear, simple workflow

---

## 💡 Use Cases

### Use Case 1: Quick Test
```bash
$ python train_model.py
# Select: CLIP, Small Scale, Small Dataset (2,000 images)
# Wait: ~1 minute
$ python retrieve_similar_images.py
# Results instantly!
```

### Use Case 2: Production Model
```bash
$ python train_model.py
# Select: CLIP, Large Scale, Full Dataset (118,000+ images)
# Wait: ~1-2 hours
$ python retrieve_similar_images.py
# Use whenever needed - model is saved!
```

### Use Case 3: Compare Models
```bash
# Train CLIP
$ python train_model.py → clip_model_1

# Train SimCLR
$ python train_model.py → simclr_model_1

# Test CLIP
$ python retrieve_similar_images.py → select clip_model_1 → see results

# Test SimCLR
$ python retrieve_similar_images.py → select simclr_model_1 → see results

# Compare results!
```

---

## 🔧 Model Management

### List Available Models
Run `retrieve_similar_images.py` - shows all saved models

### Load Model in Code
```python
from train_model import ModelManager

manager = ModelManager()
model, metadata = manager.load_model('trained_models/clip_20240101_120000')

# Now use model for predictions
embeddings = model.get_image_embeddings(images)
```

### Delete a Model
```bash
rm -rf trained_models/clip_20240101_120000/  # Remove model directory
```

---

## 📊 Training Parameters

### Model Types
- **CLIP:** Text-to-Image retrieval (recommend if you have text captions)
- **SimCLR:** Image-to-Image retrieval (works without text labels)

### Training Scales
- **Small:** 2 epochs, batch 16 → ~30 sec (testing)
- **Medium:** 10 epochs, batch 32 → ~5-10 min (development)
- **Large:** 50 epochs, batch 64 → ~1-2 hours (production)

### Datasets
- **coco_small:** 5,000 images (quick tests)
- **coco_medium:** 2,000 images (development)
- **coco_full:** 118,000+ images (production)

---

## 🚀 Next Steps

1. **Download Dataset** (if needed):
   ```bash
   python dataset/download.py
   ```

2. **Train First Model**:
   ```bash
   python train_model.py
   ```

3. **Search with Trained Model**:
   ```bash
   python retrieve_similar_images.py
   ```

4. **Experiment**: Train with different settings, compare results!

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No saved models" | Run `train_model.py` first |
| "FAISS indices not found" | Train model to generate indices |
| "Data not found" | Run `dataset/download.py` |
| "Out of memory" | Use Small scale in `train_model.py` |
| "Very slow training" | Use Small dataset + Small scale |

---

## 📝 File Reference

| File | Purpose | Status |
|------|---------|--------|
| `train_model.py` | Train & save models | ✅ USE THIS |
| `retrieve_similar_images.py` | Search & retrieve | ✅ USE THIS |
| `train.py` | Old training script | ❌ Replaced |
| `train_with_faiss.py` | Old FAISS trainer | ❌ Replaced |
| `train_template.py` | Old template | ❌ Replaced |
| `retrieve_images.py` | Old retrieval | ❌ Replaced |

---

## 🎉 You're All Set!

Your consolidated training and retrieval system is ready. Start with:

```bash
python train_model.py
```

Then search anytime with:

```bash
python retrieve_similar_images.py
```

No more confusion about which file to run! 🚀
