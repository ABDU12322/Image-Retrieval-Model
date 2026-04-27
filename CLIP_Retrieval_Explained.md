# CLIP Model — Retrieval Step: Full Detailed Breakdown

This document explains exactly how the trained CLIP model performs retrieval at inference time, covering both **Text-to-Image** and **Image-to-Image** retrieval separately.

---

## 🗂️ Pre-Requisite: What Exists Before Any Query

Before any retrieval can happen, two offline steps must be completed during or after training:

### 1. Image Embedding Index (Built Once After Training)

Every image in the dataset is passed through the **CLIP Image Encoder** to produce a 512-dimensional embedding vector. These vectors are stored in a **FAISS Flat Inner-Product index**.

```python
clip_model.eval()
with torch.no_grad():
    for images, names in eval_loader:
        emb = clip_model.forward_image(images.to(device)).cpu().numpy()
        clip_embs.append(emb)
        clip_names.extend(names)

clip_embs = np.vstack(clip_embs)          # shape: (N_images, 512)
faiss.normalize_L2(clip_embs)            # L2 normalize before indexing
clip_store.index.add(clip_embs)          # add to FAISS inner-product index
```

**What this produces:**
- A FAISS index containing `N` vectors of shape `(512,)`, one per image.
- A parallel metadata list `clip_names` mapping each FAISS row index → image filename.
- Both are saved to disk as `embeddings.index` and `embeddings_meta.json`.

> After L2 normalization, inner-product search is equivalent to cosine similarity search.

---

## 1️⃣ Text-to-Image Retrieval

### Goal
Given a **natural language query** (e.g., `"a dog playing in the park"`), return the top-K most semantically matching images from the database.

### Step-by-Step Pipeline

```
"a dog playing in the park"
        │
        ▼
┌─────────────────────────┐
│   BERT Tokenizer         │  bert_tokenizer(query, max_length=77,
│   (bert-base-uncased)    │    padding='max_length', return_tensors='pt')
└─────────────────────────┘
        │  input_ids: (1, 77)
        │  attention_mask: (1, 77)
        ▼
┌─────────────────────────┐
│   BERT (frozen)          │  BertModel.forward(input_ids, attention_mask)
│   bert-base-uncased      │  → last_hidden_state: (1, 77, 768)
│   12-layer Transformer   │    CLS token (position 0) extracted
└─────────────────────────┘
        │  cls_vec: (1, 768)
        ▼
┌─────────────────────────┐
│   Linear Projection      │  nn.Linear(768, 512)
│   (trained weight)       │  → text_emb: (1, 512)
└─────────────────────────┘
        │
        ▼
   L2 Normalize (inside FAISS search)
        │  query_vec: (1, 512), unit norm
        ▼
┌─────────────────────────┐
│   FAISS Inner-Product    │  index.search(query_vec, k=3)
│   Search                 │  → scores: (1, k)  indices: (1, k)
└─────────────────────────┘
        │
        ▼
   Map indices → image filenames via metadata list
        │
        ▼
   Return: Top-K images ranked by cosine similarity score
```

### Code in the Notebook

```python
# Tokenise the query text
ids, masks = tokenize_text(q)                        # shape (1, 77) each
ids, masks = ids.to(device), masks.to(device)

# Run BERT live + project to 512-dim
txt_emb = clip_model.forward_text(ids, masks)        # (1, 512)
txt_emb_np = txt_emb.cpu().numpy()

# Search the FAISS image index
results = clip_store.search(txt_emb_np, k=3)
# returns: [{'rank':1, 'name':'000000123456.jpg', 'score':0.87}, ...]
```

### What Happens Inside `forward_text()`

```python
def forward(self, input_ids, attention_mask=None):
    out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                    return_dict=True)
    cls = out.last_hidden_state[:, 0, :]   # CLS token → (B, 768)
    return self.projection(cls)            # Linear(768→512) → (B, 512)
```

- BERT processes all 77 tokens in parallel via self-attention.
- Only the **CLS token** (`position 0`) is used — it encodes the global sentence meaning.
- The frozen BERT weights are never updated; only the projection linear layer was trained.

### What Makes the Match Good?

During CLIP training, the contrastive loss forced the image encoder and text encoder to produce **similar 512-dim vectors** for matching image-caption pairs. So at inference, the text query embedding lands near the embeddings of images that are semantically related.

---

## 2️⃣ Image-to-Image Retrieval

### Goal
Given a **query image**, return the top-K visually similar images from the database.

### Step-by-Step Pipeline

```
Query Image (PIL / file path)
        │
        ▼
┌─────────────────────────┐
│   Image Preprocessing   │  Resize(224,224) → ToTensor()
│   (eval transform)       │  → Normalize(mean=[0.485,0.456,0.406],
│                          │               std=[0.229,0.224,0.225])
└─────────────────────────┘
        │  image_tensor: (1, 3, 224, 224)
        ▼
┌─────────────────────────┐
│   ResNet-50 Backbone     │  (pretrained + fine-tuned)
│   (fc replaced with      │  → feature_vec: (1, 2048)
│    nn.Identity)          │
└─────────────────────────┘
        │  (1, 2048)
        ▼
┌─────────────────────────┐
│   MLP Projection Head    │  Linear(2048→1024) → ReLU → Dropout(0.1)
│                          │  → Linear(1024→512)
└─────────────────────────┘
        │  img_emb: (1, 512)
        ▼
   L2 Normalize (inside FAISS search)
        │  query_vec: (1, 512), unit norm
        ▼
┌─────────────────────────┐
│   FAISS Inner-Product    │  index.search(query_vec, k=4)
│   Search                 │  → scores: (1, k)  indices: (1, k)
└─────────────────────────┘
        │
        ▼
   Map indices → image filenames via metadata list
   Skip rank-1 (same image as query)
        │
        ▼
   Return: Top-(K-1) similar images
```

### Code in the Notebook

```python
# Load and preprocess the query image
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
img = _tf(Image.open(path).convert('RGB')).unsqueeze(0).to(device)  # (1,3,224,224)

# Encode with CLIP image encoder
with torch.no_grad():
    emb = clip_model.forward_image(img).cpu().numpy()  # (1, 512)

# Search the FAISS image index
results = clip_store.search(emb, k=4)   # k=4 to account for self-match at rank 1

for r in results[1:]:   # skip rank 1 (query image itself)
    print(f"#{r['rank']-1}  {r['name']}  (score: {r['score']:.4f})")
```

### What Happens Inside `forward_image()`

```python
def forward(self, x):
    features = self.backbone(x)   # ResNet-50 body → (B, 2048)
    return self.head(features)    # MLP 2048→1024→512 → (B, 512)
```

- ResNet-50 extracts hierarchical visual features (edges → textures → objects).
- The MLP head compresses and aligns the features into the shared CLIP embedding space.
- Because CLIP was trained to align image embeddings with their captions, images that look visually similar AND share semantic content end up close in this space.

---

## 🔍 FAISS Search: What Happens Inside `search()`

Both retrieval types end at the same FAISS search step:

```python
def search(self, query: np.ndarray, k=5):
    q = query.copy().astype(np.float32)
    faiss.normalize_L2(q)                        # ensure unit norm
    scores, idxs = self.index.search(q, k)       # inner product = cosine sim
    return [
        {'rank': r+1, 'name': self.metadata[i], 'score': float(s)}
        for r, (i, s) in enumerate(zip(idxs[0], scores[0])) if i >= 0
    ]
```

- **IndexFlatIP** (Flat Inner-Product): exact brute-force search, no approximation.
- After L2 normalization, `inner_product(a, b) = cos(a, b) ∈ [-1, 1]`.
- Score of `1.0` = identical vectors; `0.0` = orthogonal; `-1.0` = opposite.
- Results are always returned in descending score order.

---

## ⚖️ Key Differences: Text-to-Image vs Image-to-Image

| Aspect | Text-to-Image | Image-to-Image |
|---|---|---|
| **Query Encoder** | BERT + Linear Projection | ResNet-50 + MLP |
| **Query Input Shape** | `(1, 77)` token IDs | `(1, 3, 224, 224)` pixels |
| **Query Output Shape** | `(1, 512)` | `(1, 512)` |
| **BERT runs at query time?** | ✅ Yes (live forward pass) | ❌ Not involved |
| **Index searched** | CLIP image FAISS index | CLIP image FAISS index |
| **Self-match issue?** | ❌ No (text ≠ image) | ✅ Yes — skip rank-1 result |
| **Similarity basis** | Semantic (meaning) | Visual + Semantic (CLIP space) |
| **Training signal** | Image-caption contrastive loss | Image-caption contrastive loss |

> **Important:** Both types search the **same FAISS index** of image embeddings. The only difference is how the query vector is produced.

---

## 📐 Embedding Space Geometry

```
               ┌────── 512-dim CLIP Embedding Space ──────┐
               │                                           │
               │   [dog.jpg] ●──────● [cat.jpg]            │
               │               \ 0.72                      │
               │                ●─── [puppy.jpg]           │
               │               /                           │
               │  "a furry dog" ●  (text query)            │
               │         ↕ 0.89                            │
               │   [dog.jpg] ●  (top result)               │
               │                                           │
               └───────────────────────────────────────────┘
```

- All vectors (image and text) live in the same normalized 512-dim space.
- Cosine distance determines similarity.
- The contrastive training ensured that semantically related items cluster together regardless of modality (text or image).

---

*Generated from: `CLIP_SimCLR_Training_v3 (2).ipynb`*
