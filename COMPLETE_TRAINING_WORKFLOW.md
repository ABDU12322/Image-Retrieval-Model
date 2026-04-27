# Complete Training Workflow

## System Architecture with Model Selection

```mermaid
graph TB
    Start["🚀 python train.py<br/>(Start Interactive Training)"]
    
    subgraph UserSelection["👤 USER SELECTION (Interactive)"]
        Q1["1️⃣ Select Model Type"]
        Q2["2️⃣ Choose Training Scale"]
        Q3["3️⃣ Pick Data Source"]
    end
    
    subgraph ModelChoice["🧠 MODEL SELECTION"]
        CLIP["CLIP Model<br/>Text-to-Image<br/>───────────<br/>• ImageEncoder<br/>• TextEncoder<br/>• Joint Loss"]
        SIMCLR["SimCLR Model<br/>Image-to-Image<br/>───────────<br/>• Single Encoder<br/>• Augmentation<br/>• Contrastive Loss"]
        Router["Model Router<br/>Instantiate<br/>Selected Model"]
    end
    
    subgraph DataPrep["📂 DATA PREPARATION"]
        Validate["Validate Data<br/>Exists on Disk"]
        Load["Load DataLoaders<br/>Batch Processing"]
        Config["Create Config<br/>Save to JSON"]
    end
    
    subgraph TrainingPhase["⚙️ TRAINING PHASE"]
        Init["Initialize Model<br/>& Trainer"]
        TrainLoop["🔄 Training Loop"]
        
        subgraph EpochLoop["For each Epoch"]
            BatchLoop["For each Batch"]
            
            subgraph BatchOps["Batch Operations"]
                Forward["Forward Pass<br/>(Get Embeddings)"]
                Loss["Calculate Loss<br/>(Contrastive)"]
                Backward["Backward Pass<br/>(Gradients)"]
                Update["Update Weights<br/>(AdamW)"]
            end
            
            Store["Store Embeddings<br/>in FAISS"]
            Log["Log Metrics"]
            CheckDone{"More<br/>Batches?"}
        end
        
        CheckEpoch{"More<br/>Epochs?"}
        SaveCheckpoint["Save Model<br/>Checkpoint"]
    end
    
    subgraph SavePhase["💾 PERSISTENCE"]
        SaveFAISS["Save FAISS Index<br/>• image_embeddings.index<br/>• image_embeddings_metadata.json"]
        FinalConfig["Save Final Config<br/>training_configs/"]
    end
    
    subgraph Output["📊 OUTPUT FILES"]
        Checkpoints["checkpoints/<br/>clip_epoch_N.pt"]
        VectorStore["vector_store/<br/>index + metadata"]
        Configs["training_configs/<br/>config.json"]
    end
    
    subgraph Retrieval["🔍 RETRIEVAL PHASE"]
        LoadModel["Load Model<br/>from Checkpoint"]
        LoadFAISS["Load FAISS Index<br/>& Metadata"]
        CreateRetriever["Create ImageRetriever"]
        QueryImage["Query Image"]
        FAISSSearch["FAISS Search<br/>L2 Distance → Top-K"]
        Results["Return Results<br/>Ranked by Similarity"]
    end
    
    Start --> Q1
    Q1 --> Q2
    Q2 --> Q3
    
    Q1 --> Router
    Router --> CLIP
    Router --> SIMCLR
    
    CLIP --> Validate
    SIMCLR --> Validate
    Q3 --> Validate
    
    Validate --> Load
    Load --> Config
    Config --> Init
    Q2 --> Init
    
    Init --> TrainLoop
    TrainLoop --> BatchLoop
    BatchLoop --> Forward
    Forward --> Loss
    Loss --> Backward
    Backward --> Update
    Update --> Store
    Store --> Log
    
    Log --> CheckDone
    CheckDone -->|Yes| BatchLoop
    CheckDone -->|No| CheckEpoch
    CheckEpoch -->|No| SaveCheckpoint
    CheckEpoch -->|Yes| SaveCheckpoint
    SaveCheckpoint --> CheckEpoch
    
    SaveCheckpoint --> SaveFAISS
    SaveFAISS --> FinalConfig
    
    FinalConfig --> Checkpoints
    FinalConfig --> VectorStore
    FinalConfig --> Configs
    
    Checkpoints --> LoadModel
    VectorStore --> LoadFAISS
    Configs --> CreateRetriever
    
    LoadModel --> CreateRetriever
    LoadFAISS --> CreateRetriever
    
    CreateRetriever --> QueryImage
    QueryImage --> FAISSSearch
    FAISSSearch --> Results
    
    style Start fill:#4caf50,color:#fff
    style UserSelection fill:#e3f2fd
    style ModelChoice fill:#f3e5f5
    style DataPrep fill:#fff3e0
    style TrainingPhase fill:#fce4ec
    style EpochLoop fill:#fff9c4
    style BatchOps fill:#f0f4c3
    style SavePhase fill:#e8f5e9
    style Output fill:#f1f8e9
    style Retrieval fill:#c8e6c9
```

---

## Detailed Training Loop Sequence

```mermaid
sequenceDiagram
    participant User
    participant Script as train.py
    participant Model as Model<br/>(CLIP/SimCLR)
    participant Trainer as CLIPTrainerWithFAISS
    participant FAISS as FAISS Index
    participant Disk as Disk Storage

    User->>Script: python train.py
    
    Script->>Script: Prompt for Model Type
    Script->>Script: Prompt for Training Scale
    Script->>Script: Prompt for Data Source
    
    Script->>Script: Select Model (CLIP or SimCLR)
    Script->>Model: Instantiate Model
    Script->>Trainer: Create Trainer Instance
    
    Script->>Trainer: trainer.train(num_epochs=10)
    
    loop For each epoch
        Trainer->>Trainer: print(f"Epoch {epoch+1}")
        
        loop For each batch
            Trainer->>Model: Load batch images
            
            alt CLIP Model
                Trainer->>Model: Also load batch text_tokens
                Model->>Model: image_encoder(images) → img_emb
                Model->>Model: text_encoder(text) → txt_emb
            else SimCLR Model
                Trainer->>Model: Also load augmented_images
                Model->>Model: encoder(images) → emb1
                Model->>Model: encoder(augmented) → emb2
            end
            
            Model->>Model: Normalize embeddings (L2)
            Trainer->>Model: Calculate contrastive loss
            Model->>Model: Backward pass (compute gradients)
            Model->>Model: Update weights (AdamW)
            
            Trainer->>FAISS: Add embeddings to index
            Trainer->>FAISS: Store image names metadata
        end
        
        Trainer->>Disk: Save checkpoint (clip_epoch_N.pt)
        Trainer->>Disk: Save FAISS index
    end
    
    Trainer->>Disk: Save final FAISS metadata.json
    Trainer->>User: Training Complete ✓
```

---

## Model-Specific Data Flows

### CLIP Training Flow

```mermaid
graph LR
    subgraph Input["INPUT BATCH"]
        Img["Images<br/>(B, 3, 224, 224)"]
        Txt["Captions<br/>(B, 77)"]
    end
    
    subgraph Encoding["ENCODING"]
        ImgEnc["Image Encoder<br/>ResNet50"]
        TxtEnc["Text Encoder<br/>Transformer"]
    end
    
    subgraph Embeddings["EMBEDDINGS"]
        ImgEmb["Image Embeds<br/>(B, 512)"]
        TxtEmb["Text Embeds<br/>(B, 512)"]
    end
    
    subgraph Loss["LOSS"]
        Norm["Normalize L2"]
        ContrastLoss["Contrastive Loss<br/>τ=0.07"]
    end
    
    subgraph Output["OUTPUT"]
        Grads["Gradients"]
        Storage["Store in FAISS"]
    end
    
    Img --> ImgEnc --> ImgEmb --> Norm --> ContrastLoss --> Grads
    TxtEnc --> TxtEmb --> Norm
    ImgEmb --> Storage
    
    style Input fill:#e3f2fd
    style Encoding fill:#f3e5f5
    style Embeddings fill:#fff3e0
    style Loss fill:#fce4ec
    style Output fill:#e8f5e9
```

### SimCLR Training Flow

```mermaid
graph LR
    subgraph Input["INPUT BATCH"]
        Img1["Image View 1<br/>(B, 3, 224, 224)"]
        Img2["Image View 2<br/>(B, 3, 224, 224)<br/>(Augmented)"]
    end
    
    subgraph Encoding["ENCODING"]
        Enc1["Encoder<br/>ResNet50"]
        Enc2["Encoder<br/>ResNet50"]
    end
    
    subgraph Projection["PROJECTION"]
        Proj1["Project Head<br/>(512→128)"]
        Proj2["Project Head<br/>(512→128)"]
    end
    
    subgraph Loss["LOSS"]
        ContrastLoss["Contrastive Loss<br/>τ=0.07<br/>View1 vs View2"]
    end
    
    subgraph Output["OUTPUT"]
        Grads["Gradients"]
        Storage["Store in FAISS<br/>(512-dim embeddings)"]
    end
    
    Img1 --> Enc1 --> Proj1 --> ContrastLoss --> Grads
    Img2 --> Enc2 --> Proj2 --> ContrastLoss
    
    Enc1 --> Storage
    
    style Input fill:#e3f2fd
    style Encoding fill:#f3e5f5
    style Projection fill:#fff3e0
    style Loss fill:#fce4ec
    style Output fill:#e8f5e9
```

---

## Training Configuration Selection Logic

```mermaid
graph TD
    Start["Select Configuration"]
    
    ScaleQ["What is your<br/>time budget?"]
    
    TimeSmall["< 10 minutes"]
    TimeMed["30-60 minutes"]
    TimeLarge["> 1 hour"]
    
    DataQ["How much data<br/>do you have?"]
    
    DataSmall["Small Dataset<br/>5K images"]
    DataMed["Medium Dataset<br/>2K images"]
    DataLarge["Large Dataset<br/>118K+ images"]
    
    SmallConfig["📊 SMALL SCALE<br/>─────────────<br/>Epochs: 2<br/>Batch: 16<br/>LR: 1e-4<br/>⏱ Time: 5 min<br/>📈 Quality: Good"]
    
    MedConfig["📊 MEDIUM SCALE<br/>─────────────<br/>Epochs: 10<br/>Batch: 32<br/>LR: 1e-4<br/>⏱ Time: 30 min<br/>📈 Quality: Better"]
    
    LargeConfig["📊 LARGE SCALE<br/>─────────────<br/>Epochs: 50<br/>Batch: 64<br/>LR: 5e-5<br/>⏱ Time: 4+ hrs<br/>📈 Quality: Best"]
    
    Start --> ScaleQ
    ScaleQ --> TimeSmall
    ScaleQ --> TimeMed
    ScaleQ --> TimeLarge
    
    TimeSmall --> DataQ
    TimeMed --> DataQ
    TimeLarge --> DataQ
    
    DataQ --> DataSmall
    DataQ --> DataMed
    DataQ --> DataLarge
    
    DataSmall --> SmallConfig
    DataMed --> MedConfig
    DataLarge --> LargeConfig
    
    style SmallConfig fill:#c8e6c9
    style MedConfig fill:#fff9c4
    style LargeConfig fill:#ffccbc
```

---

## Output Directory Structure After Training

```
project_root/
├── checkpoints/                    # 🎯 Model Weights
│   ├── clip_epoch_1.pt
│   ├── clip_epoch_2.pt
│   └── clip_epoch_10.pt            # ← Use this for inference
│
├── vector_store/                   # 🗂️ FAISS Indices
│   ├── image_embeddings.index      # Binary FAISS index
│   └── image_embeddings_metadata.json  # Image name mapping
│       {
│         "embedding_dim": 512,
│         "num_vectors": 2000,
│         "image_names": ["img1.jpg", "img2.jpg", ...]
│       }
│
├── training_configs/               # 📋 Configuration Records
│   └── clip_coco_small_config.json
│       {
│         "model_type": "clip",
│         "training": {
│           "epochs": 10,
│           "batch_size": 32,
│           "learning_rate": 1e-4
│         },
│         "model_params": {...}
│       }
└── logs/                          # 📊 Training Logs (optional)
    └── training_log.txt
```

---

## Complete Training Command Examples

### Example 1: First Time Users (Interactive)
```bash
python train.py
# Follow 3 simple prompts
```

### Example 2: Quick Test
```bash
python train.py --model clip --scale small --data coco_small --non-interactive
```

### Example 3: Development
```bash
python train.py --model clip --scale medium --data coco_small
```

### Example 4: Production
```bash
python train.py --model clip --scale large --data coco_full
```

### Example 5: Alternative Model
```bash
python train.py --model simclr --scale medium --data coco_small
```

---

## After Training: Usage Workflow

```mermaid
graph LR
    Model["✅ Trained Model<br/>checkpoints/"]
    FAISS["✅ FAISS Index<br/>vector_store/"]
    
    Retriever["ImageRetriever<br/>Load Model & Index"]
    
    UseCase1["Search by Image<br/>retrieve_images.py"]
    UseCase2["API Endpoint<br/>FastAPI/Flask"]
    UseCase3["Batch Processing<br/>Script"]
    
    Result1["Similar Images<br/>Top-K Results"]
    Result2["REST API<br/>JSON Response"]
    Result3["Bulk Search<br/>CSV Output"]
    
    Model --> Retriever
    FAISS --> Retriever
    
    Retriever --> UseCase1
    Retriever --> UseCase2
    Retriever --> UseCase3
    
    UseCase1 --> Result1
    UseCase2 --> Result2
    UseCase3 --> Result3
    
    style Model fill:#c8e6c9
    style FAISS fill:#c8e6c9
    style Retriever fill:#fff9c4
    style UseCase1 fill:#f0f4c3
    style UseCase2 fill:#f0f4c3
    style UseCase3 fill:#f0f4c3
```

---

## Key Metrics to Monitor

During training, watch these metrics:

```python
Epoch 1, Batch 1/32, Loss: 3.2145
                      ↑
                  Contrastive Loss
                  Should decrease over batches/epochs
                  
✓ Training Loss: 2.9834       ← Overall epoch loss
✓ Validation Loss: 3.0123     ← Should be similar (no overfitting)

If Training Loss < Validation Loss by a lot → Model is overfitting
If Training Loss doesn't decrease → Learning rate too low
```

---

## Troubleshooting Flowchart

```mermaid
graph TD
    Problem["❌ Training Problem"]
    
    Q1{"Error<br/>message?"}
    
    OutOfMem["Out of Memory"]
    NotFound["Data Not Found"]
    FAISSErr["FAISS Error"]
    LossStalled["Loss Not Decreasing"]
    Slow["Training Too Slow"]
    
    Sol1["Reduce batch_size<br/>train.py --scale small"]
    Sol2["Download data first<br/>cd dataset<br/>python coco_small_download.py"]
    Sol3["Install FAISS<br/>pip install faiss-cpu"]
    Sol4["Increase epochs<br/>or lower learning rate"]
    Sol5["Use GPU<br/>check nvidia-smi"]
    
    Problem --> Q1
    Q1 -->|CUDA| OutOfMem
    Q1 -->|FileNotFound| NotFound
    Q1 -->|ImportError| FAISSErr
    Q1 -->|No Error| Slow
    
    OutOfMem --> Sol1
    NotFound --> Sol2
    FAISSErr --> Sol3
    LossStalled --> Sol4
    Slow --> Sol5
    
    style Problem fill:#ffccbc
    style Sol1 fill:#c8e6c9
    style Sol2 fill:#c8e6c9
    style Sol3 fill:#c8e6c9
    style Sol4 fill:#c8e6c9
    style Sol5 fill:#c8e6c9
```

---

## Summary

✅ **Interactive Training** - `python train.py`  
✅ **Model Selection** - CLIP or SimCLR based on data  
✅ **Flexible Scaling** - Small/Medium/Large configs  
✅ **Automatic FAISS** - Embeddings stored during training  
✅ **Complete Documentation** - Guides for every step  

**Start training now:** `python train.py`
