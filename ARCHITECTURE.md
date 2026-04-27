# Image Retrieval Model - Complete Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Model Selection](#model-selection)
3. [High-Level Architecture](#high-level-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference & Retrieval Pipeline](#inference--retrieval-pipeline)
6. [Data Flow](#data-flow)
7. [Component Details](#component-details)
8. [File Structure](#file-structure)
9. [Integration Points](#integration-points)

---

## System Overview

Your Image Retrieval Model is a **text-to-image and image-to-image retrieval system** built with CLIP (Contrastive Language-Image Pre-training) and enhanced with FAISS vector search.

### Key Capabilities
- **Text-to-Image Retrieval**: Given text, find similar images
- **Image-to-Image Retrieval**: Given an image, find similar images
- **Efficient Search**: FAISS-based fast similarity search over millions of images
- **Scalable**: Can handle large-scale datasets with millions of images

---

## Model Selection

Your system supports **dynamic model selection** based on user requirements:

### CLIP Model (Text-to-Image Retrieval)

**When to Use:**
- You have paired image-text data
- Need to search images by text description
- Want joint image-text embeddings
- Building semantic search systems

**Architecture:**
```
Images ──→ Image Encoder (ResNet50) ──→ Image Embeddings (512-dim)
                                              ↓
                                    Contrastive Loss ← Temperature: 0.07
                                              ↑
Captions → Text Encoder (Transformer) ──→ Text Embeddings (512-dim)
```

**Data Format:**
```python
batch = {
    'images': (B, 3, 224, 224),
    'text_tokens': (B, 77),
    'text_mask': (B, 77),
    'image_names': List[str]
}
```

### SimCLR Model (Image-to-Image Retrieval)

**When to Use:**
- You only have images (no text labels)
- Need self-supervised learning
- Want image similarity search
- Building unsupervised representations

**Architecture:**
```
Images ──→ Encoder ──→ Projections ──→ Contrastive Loss
           ↓              ↓              ↑
        ResNet50       (512→128)    Temperature: 0.07
```

**Data Format:**
```python
batch = {
    'images': (B, 3, 224, 224),
    'augmented_images': (B, 3, 224, 224),  # Augmented view
    'image_names': List[str]
}
```

### Model Selection Decision Tree

```mermaid
graph TD
    Q1["Do you have<br/>paired image-text data?"]
    
    Q1 -->|Yes| Q2["Want text-to-image<br/>retrieval?"]
    Q1 -->|No| Q3["Need labeled data?"]
    
    Q2 -->|Yes| CLIP["🎯 Use CLIP<br/>Text-to-Image Search"]
    Q2 -->|No| BOTH["✓ CLIP also does<br/>Image-to-Image"]
    
    Q3 -->|No| SIMCLR["🎯 Use SimCLR<br/>Self-Supervised"]
    Q3 -->|Yes| SIMCLR2["✓ Still use SimCLR<br/>Label not needed"]
    
    style CLIP fill:#c8e6c9
    style BOTH fill:#c8e6c9
    style SIMCLR fill:#c8e6c9
    style SIMCLR2 fill:#c8e6c9
```

### How to Choose at Runtime

```bash
# Interactive selection
python train.py

# Command line selection
python train.py --model clip --scale medium --data coco_small

# Programmatic selection
from models import CLIPModel, SimCLRModel

# Choose based on data
if has_text_captions:
    model = CLIPModel()
else:
    model = SimCLRModel()
```

---

## High-Level Architecture

```mermaid
graph TB
    subgraph UserInput["👤 User Input"]
        ModelSelect["Select Model<br/>(CLIP or SimCLR)"]
        TrainingScale["Choose Scale<br/>(Small/Medium/Large)"]
        DataSource["Pick Data<br/>(5K/2K/118K)"]
    end
    
    subgraph DataLayer["📦 Data Layer"]
        RawData["Raw Images & Captions<br/>(COCO Dataset)"]
        ProcessedData["Processed Data<br/>(Cleaned Annotations)"]
        SampleData["Sample Data<br/>(2000 images)"]
    end
    
    subgraph DataPrep["🔧 Data Preparation"]
        CleanScript["clean_annotations.py<br/>(Group captions by image)"]
        SampleScript["fetch_sample_captions.py<br/>(Create training set)"]
    end
    
    subgraph ModelSelection["🧠 Model Selection"]
        CLIP["CLIP Model<br/>(Text-Image Joint)"]
        SimCLR["SimCLR Model<br/>(Self-Supervised)"]
        ModelSwitch["Model Router<br/>(Dynamic Selection)"]
    end
    
    subgraph TrainingLayer["⚙️ Training Layer"]
        DataLoader["DataLoader<br/>(Batch Processing)"]
        Trainer["Trainer with FAISS<br/>(Training Loop)"]
        Loss["Loss Function<br/>(Contrastive Loss)"]
    end
    
    subgraph VectorLayer["🗂️ Vector Storage"]
        FAISSIndex["FAISS Index<br/>(512-dim vectors)"]
        Metadata["Metadata<br/>(Image Names)"]
    end
    
    subgraph RetrievalLayer["🔍 Retrieval Layer"]
        Retriever["ImageRetriever<br/>(Load & Search)"]
        Search["FAISS Search<br/>(Similarity)"]
        FileMapper["File Mapper<br/>(Path Resolution)"]
    end
    
    subgraph OutputLayer["📤 Output"]
        Results["Search Results<br/>(Top-K Similar Images)"]
        Images["Image Files<br/>(From Disk)"]
    end
    
    UserInput --> ModelSelect
    ModelSelect --> ModelSwitch
    TrainingScale --> Trainer
    DataSource --> DataPrep
    
    RawData --> CleanScript
    CleanScript --> ProcessedData
    ProcessedData --> SampleScript
    SampleScript --> SampleData
    
    SampleData --> DataLoader
    
    ModelSwitch --> CLIP
    ModelSwitch --> SimCLR
    CLIP --> Trainer
    SimCLR --> Trainer
    
    DataLoader --> Trainer
    Loss --> Trainer
    
    Trainer --> FAISSIndex
    Trainer --> Metadata
    
    FAISSIndex --> Retriever
    Metadata --> Retriever
    
    Retriever --> Search
    Search --> FileMapper
    FileMapper --> Results
    Results --> Images
    
    style UserInput fill:#e3f2fd
    style DataLayer fill:#f3e5f5
    style DataPrep fill:#fff3e0
    style ModelSelection fill:#fce4ec
    style TrainingLayer fill:#e8f5e9
    style VectorLayer fill:#f1f8e9
    style RetrievalLayer fill:#fff9c4
    style OutputLayer fill:#f0f4c3
```

---

## Training Pipeline

### Detailed Training Flow

```mermaid
graph LR
    subgraph Input["INPUT BATCH"]
        Img["Images<br/>(B, 3, H, W)"]
        Cap["Captions<br/>(B, 77)"]
        ImgName["Image Names<br/>(B,)"]
    end
    
    subgraph Processing["ENCODING"]
        ImgEnc["Image Encoder<br/>ResNet50 + Projection"]
        CapEnc["Text Encoder<br/>Transformer"]
    end
    
    subgraph Embeddings["EMBEDDINGS"]
        ImgEmb["Image Embeddings<br/>(B, 512)"]
        CapEmb["Caption Embeddings<br/>(B, 512)"]
    end
    
    subgraph LossComp["LOSS COMPUTATION"]
        Norm["L2 Normalize"]
        ContrastLoss["Contrastive Loss<br/>(Temperature Scaled)"]
    end
    
    subgraph Backprop["BACKPROPAGATION"]
        Backward["Backward Pass<br/>(Gradients)"]
        Update["Update Weights<br/>(AdamW)"]
    end
    
    subgraph Storage["STORAGE"]
        FAISSAdd["Add to FAISS<br/>Index"]
        MetaStore["Store Metadata<br/>(Image Names)"]
    end
    
    Img --> ImgEnc
    Cap --> CapEnc
    
    ImgEnc --> ImgEmb
    CapEnc --> CapEmb
    
    ImgEmb --> Norm
    CapEmb --> Norm
    
    Norm --> ContrastLoss
    ContrastLoss --> Backward
    
    Backward --> Update
    
    ImgEmb --> FAISSAdd
    ImgName --> MetaStore
    
    FAISSAdd --> FAISSAdd
    MetaStore --> MetaStore
    
    style Input fill:#e1f5ff
    style Processing fill:#f3e5f5
    style Embeddings fill:#fff3e0
    style LossComp fill:#fce4ec
    style Backprop fill:#e8f5e9
    style Storage fill:#f1f8e9
```

### Training Iteration Loop

```mermaid
graph TD
    Start["🚀 Start Training"]
    
    For["FOR each epoch"]
    Batch["Get Batch from DataLoader"]
    Forward["Forward Pass<br/>(Images & Captions)"]
    CalcLoss["Calculate Loss"]
    
    BackPass["Backward Pass<br/>(Compute Gradients)"]
    OptStep["Optimizer Step<br/>(Update Weights)"]
    
    StoreEmbed["Store Image Embeddings<br/>in FAISS"]
    StoreMeta["Store Image Names<br/>in Metadata"]
    
    LogLoss["Log Loss"]
    
    NextBatch{More Batches<br/>in Epoch?}
    NextEpoch{More Epochs?}
    
    SaveModel["Save Model<br/>Checkpoint"]
    SaveFAISS["Save FAISS Index<br/>(index + metadata.json)"]
    
    End["✅ Training Complete"]
    
    Start --> For
    For --> Batch
    Batch --> Forward
    Forward --> CalcLoss
    CalcLoss --> BackPass
    BackPass --> OptStep
    OptStep --> StoreEmbed
    StoreEmbed --> StoreMeta
    StoreMeta --> LogLoss
    LogLoss --> NextBatch
    
    NextBatch -->|Yes| Batch
    NextBatch -->|No| NextEpoch
    NextEpoch -->|Yes| For
    NextEpoch -->|No| SaveModel
    SaveModel --> SaveFAISS
    SaveFAISS --> End
    
    style Start fill:#4caf50,color:#fff
    style End fill:#4caf50,color:#fff
    style Batch fill:#e3f2fd
    style Forward fill:#f3e5f5
    style StoreEmbed fill:#fff3e0
    style SaveFAISS fill:#f1f8e9
```

---

## Inference & Retrieval Pipeline

### Image Retrieval Flow

```mermaid
graph LR
    subgraph Query["QUERY PREPARATION"]
        QImg["Query Image<br/>from File"]
        QLoad["Load Image<br/>(PIL)"]
        QProcess["Preprocess<br/>(224x224, Normalize)"]
    end
    
    subgraph Encode["ENCODING"]
        ToTensor["Convert to Tensor<br/>(3, 224, 224)"]
        Batch["Add Batch Dim<br/>(1, 3, 224, 224)"]
        Embed["Image Encoder<br/>Get Embedding"]
    end
    
    subgraph Vector["VECTOR PREP"]
        ToCPU["Move to CPU<br/>as NumPy"]
        Normalize["L2 Normalize<br/>for Cosine Sim"]
    end
    
    subgraph FAISSSearch["FAISS SEARCH"]
        LoadIndex["Load FAISS Index<br/>(512-dim)"]
        LoadMeta["Load Metadata<br/>(Image Names)"]
        Search["FAISS Search<br/>(L2 Distance)"]
        TopK["Get Top-K Results<br/>(k nearest neighbors)"]
    end
    
    subgraph MapFiles["FILE MAPPING"]
        GetNames["Extract Image Names<br/>from Results"]
        FindPaths["Find File Paths<br/>in Directories"]
        CheckDisk["Verify Files<br/>on Disk"]
    end
    
    subgraph Output["OUTPUT"]
        Results["Return Results<br/>(Rank, Name, Similarity, Path)"]
    end
    
    QImg --> QLoad
    QLoad --> QProcess
    QProcess --> ToTensor
    ToTensor --> Batch
    Batch --> Embed
    
    Embed --> ToCPU
    ToCPU --> Normalize
    
    Normalize --> LoadIndex
    LoadIndex --> LoadMeta
    LoadMeta --> Search
    Search --> TopK
    
    TopK --> GetNames
    GetNames --> FindPaths
    FindPaths --> CheckDisk
    
    CheckDisk --> Results
    
    style Query fill:#e3f2fd
    style Encode fill:#f3e5f5
    style Vector fill:#fff3e0
    style FAISSSearch fill:#fce4ec
    style MapFiles fill:#e8f5e9
    style Output fill:#f1f8e9
```

---

## Data Flow

### Complete End-to-End Data Flow

```mermaid
graph TB
    subgraph RawDataSrc["📥 Raw Data Source"]
        CocoImg["COCO Images<br/>(118K+ images)"]
        CocoCap["COCO Captions<br/>(591K+ captions)"]
    end
    
    subgraph DataCleaning["🧹 Data Cleaning"]
        CleanAnn["clean_annotations.py<br/>Groups captions by image_id"]
        GroupResult["Output: 1 Image → 5-7 Captions"]
    end
    
    subgraph SampleCreation["✂️ Sample Creation"]
        FetchSample["fetch_sample_captions.py<br/>Random sample 2000 images"]
        SampleResult["Output: 2000 Images<br/>Each with 5-7 captions"]
    end
    
    subgraph DataPrep["⚙️ Data Preparation"]
        CustomDataset["Custom Dataset Class<br/>Loads images & captions"]
        DataLoader["PyTorch DataLoader<br/>Batch size = 32"]
    end
    
    subgraph Encoding["🧠 Model Encoding"]
        ImgPath["Image Files"]
        CapTexts["Caption Texts"]
        ImgEncoder["Image Encoder<br/>(ResNet50)"]
        TextEncoder["Text Encoder<br/>(Transformer)"]
        ImgEmbed["Image Embeddings<br/>(32, 512)"]
        TextEmbed["Text Embeddings<br/>(32, 512)"]
    end
    
    subgraph Loss["📊 Loss Calculation"]
        Normalize["Normalize Embeddings<br/>L2 norm"]
        CosineSim["Compute Similarity Matrix<br/>(32x32)"]
        ContrastiveLoss["Contrastive Loss<br/>Temperature: 0.07"]
    end
    
    subgraph FAISS["🗂️ FAISS Storage"]
        FAISSIdx["FAISS Index<br/>(L2 Distance)"]
        MetaJSON["Metadata JSON<br/>(Image Names)"]
        SaveDisk["Save to Disk<br/>(After Training)"]
    end
    
    subgraph Retrieval["🔍 Retrieval"]
        QueryImg["Query Image"]
        QueryEmbed["Query Embedding<br/>(1, 512)"]
        FAISSSearch["FAISS Search<br/>Top-K similar"]
        Results["Results:<br/>Rank, Name, Score, Path"]
    end
    
    CocoImg --> CleanAnn
    CocoCap --> CleanAnn
    CleanAnn --> GroupResult
    
    GroupResult --> FetchSample
    FetchSample --> SampleResult
    
    SampleResult --> CustomDataset
    CustomDataset --> DataLoader
    
    DataLoader --> ImgPath
    DataLoader --> CapTexts
    
    ImgPath --> ImgEncoder
    CapTexts --> TextEncoder
    
    ImgEncoder --> ImgEmbed
    TextEncoder --> TextEmbed
    
    ImgEmbed --> Normalize
    TextEmbed --> Normalize
    
    Normalize --> CosineSim
    CosineSim --> ContrastiveLoss
    
    ImgEmbed --> FAISSIdx
    ImgPath --> MetaJSON
    
    FAISSIdx --> SaveDisk
    MetaJSON --> SaveDisk
    
    QueryImg --> QueryEmbed
    QueryEmbed --> FAISSSearch
    
    FAISSIdx --> FAISSSearch
    MetaJSON --> FAISSSearch
    
    FAISSSearch --> Results
    
    style RawDataSrc fill:#e3f2fd
    style DataCleaning fill:#f3e5f5
    style SampleCreation fill:#fff3e0
    style DataPrep fill:#fce4ec
    style Encoding fill:#e8f5e9
    style Loss fill:#f1f8e9
    style FAISS fill:#fff9c4
    style Retrieval fill:#f0f4c3
```

---

## Component Details

### 1. Image Encoder

```mermaid
graph LR
    Input["Image Input<br/>(B, 3, 224, 224)"]
    
    subgraph ResNet["ResNet50 Backbone<br/>(Pretrained)"]
        Conv1["Conv Layer 1"]
        ResBlock1["Residual Block 1"]
        ResBlock2["Residual Block 2"]
        ResBlock3["Residual Block 3"]
        ResBlock4["Residual Block 4"]
        AvgPool["Average Pooling"]
    end
    
    Features["Features<br/>(B, 2048, 1, 1)"]
    Flatten["Flatten<br/>(B, 2048)"]
    
    subgraph Projection["Projection Head"]
        FC1["Linear 2048→2048"]
        ReLU["ReLU Activation"]
        FC2["Linear 2048→512"]
    end
    
    Embedding["Embeddings<br/>(B, 512)"]
    
    Input --> Conv1 --> ResBlock1 --> ResBlock2 --> ResBlock3 --> ResBlock4 --> AvgPool
    AvgPool --> Features
    Features --> Flatten
    Flatten --> FC1 --> ReLU --> FC2
    FC2 --> Embedding
    
    style ResNet fill:#f3e5f5
    style Projection fill:#fff3e0
    style Embedding fill:#e1f5ff
```

### 2. Text Encoder

```mermaid
graph LR
    Tokens["Caption Tokens<br/>(B, 77)"]
    
    subgraph TokenEmb["Token Embedding"]
        Emb["Embedding Layer<br/>vocab=10000 → 512"]
    end
    
    subgraph PosEmb["Positional Embedding"]
        Pos["Add Position Info<br/>(Learnable)"]
    end
    
    Embedded["Embedded Tokens<br/>(B, 77, 512)"]
    
    subgraph Transformer["Transformer Encoder<br/>(12 Layers)"]
        Layer["Attention Layers<br/>(8 heads, 2048 FFN)"]
    end
    
    Encoded["Encoded<br/>(B, 77, 512)"]
    
    Pooling["Pooling Operation<br/>(Take [CLS] or Mean)"]
    
    Embedding["Text Embeddings<br/>(B, 512)"]
    
    Tokens --> Emb
    Emb --> Pos
    Pos --> Embedded
    Embedded --> Layer
    Layer --> Encoded
    Encoded --> Pooling
    Pooling --> Embedding
    
    style TokenEmb fill:#f3e5f5
    style PosEmb fill:#fff3e0
    style Transformer fill:#fce4ec
    style Embedding fill:#e1f5ff
```

### 3. CLIP Model

```mermaid
graph TB
    subgraph Input["Input"]
        Img["Images<br/>(B, 3, 224, 224)"]
        Text["Text Tokens<br/>(B, 77)"]
    end
    
    subgraph Encoders["Encoders"]
        ImgEnc["Image Encoder"]
        TextEnc["Text Encoder"]
    end
    
    subgraph Embeddings["Embeddings"]
        ImgEmb["I_emb<br/>(B, 512)"]
        TextEmb["T_emb<br/>(B, 512)"]
    end
    
    subgraph Normalization["Normalization"]
        L2Norm["L2 Normalize<br/>Unit Vectors"]
    end
    
    subgraph SimilarityComp["Similarity Computation"]
        MatMul["Matrix Multiply<br/>I_emb @ T_emb.T<br/>(B x B)"]
        Temperature["Scale by τ (0.07)"]
    end
    
    subgraph Loss["Contrastive Loss"]
        CrossEnt["Cross-Entropy Loss<br/>Image→Text<br/>+ Text→Image"]
        TotalLoss["Total Loss"]
    end
    
    Img --> ImgEnc
    Text --> TextEnc
    
    ImgEnc --> ImgEmb
    TextEnc --> TextEmb
    
    ImgEmb --> L2Norm
    TextEmb --> L2Norm
    
    L2Norm --> MatMul
    MatMul --> Temperature
    Temperature --> CrossEnt
    CrossEnt --> TotalLoss
    
    style Input fill:#e3f2fd
    style Encoders fill:#f3e5f5
    style Embeddings fill:#fff3e0
    style Normalization fill:#fce4ec
    style SimilarityComp fill:#e8f5e9
    style Loss fill:#f1f8e9
```

### 4. FAISS Vector Store

```mermaid
graph TB
    subgraph Add["Add Embeddings"]
        Embed["Image Embeddings<br/>(B, 512)"]
        Names["Image Names<br/>List"]
        Normalize["Normalize L2"]
        AddToIdx["Add to Index"]
        SaveMeta["Store in Metadata"]
    end
    
    subgraph Index["FAISS Index Structure"]
        VectorData["Vector Data<br/>Binary File<br/>.index"]
        MetaJSON["Metadata<br/>JSON File<br/>_metadata.json"]
    end
    
    subgraph Search["Search Query"]
        Query["Query Embedding<br/>(1, 512)"]
        QueryNorm["Normalize L2"]
        FAISSSearchOp["FAISS Search<br/>L2 Distance<br/>Top-K"]
        RetrieveMeta["Retrieve Metadata<br/>for Top-K"]
    end
    
    subgraph Output["Search Output"]
        Results["Results<br/>Rank, Name, Distance<br/>Similarity Score"]
    end
    
    Embed --> Normalize
    Names --> SaveMeta
    Normalize --> AddToIdx
    AddToIdx --> VectorData
    SaveMeta --> MetaJSON
    
    Query --> QueryNorm
    QueryNorm --> FAISSSearchOp
    VectorData --> FAISSSearchOp
    FAISSSearchOp --> RetrieveMeta
    MetaJSON --> RetrieveMeta
    RetrieveMeta --> Results
    
    style Add fill:#fff3e0
    style Index fill:#fff9c4
    style Search fill:#f0f4c3
    style Output fill:#c8e6c9
```

---

## File Structure

### Project Directory Layout

```mermaid
graph TD
    Root["📁 Image-Retrieval-Model"]
    
    subgraph Scripts["🐍 Scripts"]
        Clean["clean_annotations.py<br/>→ cleaned/"]
        Fetch["fetch_sample_captions.py<br/>→ 2000 images"]
        TrainFAISS["train_with_faiss.py<br/>→ checkpoints/"]
        Retrieve["retrieve_images.py<br/>→ Search"]
        Example["example_faiss_workflow.py<br/>→ Demo"]
    end
    
    subgraph ModelsPkg["models/ 📦"]
        InitMod["__init__.py"]
        Encoders["encoders.py<br/>(ImageEncoder, TextEncoder)"]
        CLIPMod["clip_model.py<br/>(CLIPModel)"]
        SimCLR["simclr_model.py<br/>(SimCLRModel)"]
        Losses["losses.py<br/>(Loss Functions)"]
        Utils["utils.py<br/>(Utilities)"]
        FAISS["faiss_vector_store.py<br/>(FAISSVectorStore, EmbeddingManager)"]
    end
    
    subgraph DatasetDir["dataset/ 📂"]
        COCO["coco/"]
        COCOSmall["coco_small/"]
        Scripts2["download.py, coco_small_download.py"]
        
        subgraph COCOStruct["coco/"]
            Annotations["annotations/"]
            Train["train2017/"]
            Val["val2017/"]
            Cleaned["annotations/cleaned/"]
        end
        
        subgraph CleanedAnn["cleaned/"]
            CCapTrain["captions_train2017.json<br/>(grouped captions)"]
            CCapVal["captions_val2017.json<br/>(grouped captions)"]
            Others["instances_*.json<br/>person_keypoints_*.json"]
        end
    end
    
    subgraph Output["📊 Output/Checkpoints"]
        Checkpoints["checkpoints/"]
        VectorStore["vector_store/"]
        
        subgraph CheckpointFiles["checkpoints/"]
            Epoch1["clip_epoch_1.pt"]
            Epoch2["clip_epoch_N.pt"]
        end
        
        subgraph VectorFiles["vector_store/"]
            FAISSFile["image_embeddings.index"]
            MetaFile["image_embeddings_metadata.json"]
        end
    end
    
    subgraph Docs["📚 Documentation"]
        README["README.md"]
        DetailGuide["DETAILED_GUIDE.md"]
        FAISSGuide["FAISS_INTEGRATION_GUIDE.md"]
        FAISSRef["FAISS_QUICK_REFERENCE.md"]
        ArchDoc["ARCHITECTURE.md<br/>(This file)"]
    end
    
    subgraph Requirements["🔧 Configuration"]
        ReqTxt["requirements.txt<br/>torch, torchvision<br/>numpy, Pillow, tqdm<br/>faiss-cpu"]
        TrainTemplate["train_template.py<br/>(Training template)"]
    end
    
    Root --> Scripts
    Root --> ModelsPkg
    Root --> DatasetDir
    Root --> Output
    Root --> Docs
    Root --> Requirements
    
    DatasetDir --> COCO
    DatasetDir --> COCOSmall
    DatasetDir --> Scripts2
    
    COCO --> COCOStruct
    COCOStruct --> Cleaned
    Cleaned --> CleanedAnn
    
    Checkpoints --> CheckpointFiles
    VectorStore --> VectorFiles
    
    style Root fill:#e3f2fd
    style Scripts fill:#f3e5f5
    style ModelsPkg fill:#fff3e0
    style DatasetDir fill:#fce4ec
    style Output fill:#e8f5e9
    style Docs fill:#f1f8e9
    style Requirements fill:#fff9c4
```

---

## Integration Points

### Training → Inference Pipeline

```mermaid
graph LR
    subgraph Training["⚙️ TRAINING PHASE"]
        TData["Load Data<br/>(2000 images)"]
        TModel["Initialize Model<br/>(CLIP)"]
        TTrainer["Create Trainer<br/>(CLIPTrainerWithFAISS)"]
        TLoop["Training Loop<br/>(num_epochs)"]
        TSaveModel["Save Model<br/>checkpoints/"]
        TSaveFAISS["Save FAISS<br/>vector_store/"]
    end
    
    subgraph Checkpoints["💾 CHECKPOINTS"]
        ModelFile["clip_epoch_N.pt<br/>(Model Weights)"]
        IndexFile["image_embeddings.index<br/>(FAISS Index)"]
        MetaFile["image_embeddings_metadata.json<br/>(Image Names)"]
    end
    
    subgraph Inference["🔍 INFERENCE PHASE"]
        ILoadModel["Load Model<br/>from checkpoint"]
        ILoadFAISS["Load FAISS Index<br/>& Metadata"]
        IRetriever["Create ImageRetriever"]
        IQuery["Query Image"]
        ISearch["Search FAISS"]
        IResults["Get Results"]
    end
    
    subgraph Usage["📤 USAGE"]
        DisplayResults["Display Top-K<br/>Similar Images"]
        LocateFiles["Locate Images<br/>on Disk"]
    end
    
    TData --> TModel
    TModel --> TTrainer
    TTrainer --> TLoop
    TLoop --> TSaveModel
    TLoop --> TSaveFAISS
    
    TSaveModel --> ModelFile
    TSaveFAISS --> IndexFile
    TSaveFAISS --> MetaFile
    
    ModelFile --> ILoadModel
    IndexFile --> ILoadFAISS
    MetaFile --> ILoadFAISS
    
    ILoadModel --> IRetriever
    ILoadFAISS --> IRetriever
    
    IRetriever --> IQuery
    IQuery --> ISearch
    ISearch --> IResults
    
    IResults --> DisplayResults
    IResults --> LocateFiles
    
    style Training fill:#c8e6c9
    style Checkpoints fill:#fff9c4
    style Inference fill:#ffccbc
    style Usage fill:#b3e5fc
```

### Module Dependencies

```mermaid
graph TB
    subgraph External["External Libraries"]
        Torch["PyTorch<br/>torch, torchvision"]
        FAISS["FAISS<br/>faiss-cpu"]
        Utils["Utilities<br/>numpy, PIL, tqdm"]
    end
    
    subgraph ModelCore["Core Models"]
        Encoders["encoders.py<br/>(ImageEncoder, TextEncoder)"]
        Losses["losses.py<br/>(Contrastive Loss)"]
        CLIPModel["clip_model.py<br/>(CLIP Model)"]
    end
    
    subgraph VectorCore["Vector Storage"]
        FAISSStore["faiss_vector_store.py<br/>(FAISSVectorStore)"]
        EmbedMgr["EmbeddingManager"]
    end
    
    subgraph TrainingCore["Training"]
        TrainScript["train_with_faiss.py<br/>(CLIPTrainerWithFAISS)"]
    end
    
    subgraph RetrievalCore["Retrieval"]
        RetrieveScript["retrieve_images.py<br/>(ImageRetriever)"]
    end
    
    subgraph DataCore["Data Processing"]
        CleanScript["clean_annotations.py"]
        FetchScript["fetch_sample_captions.py"]
    end
    
    subgraph ExampleCore["Examples"]
        ExampleScript["example_faiss_workflow.py"]
    end
    
    Torch --> Encoders
    Torch --> Losses
    Torch --> CLIPModel
    Torch --> TrainScript
    Torch --> RetrieveScript
    
    FAISS --> FAISSStore
    Utils --> FAISSStore
    
    Encoders --> CLIPModel
    Losses --> TrainScript
    CLIPModel --> TrainScript
    
    FAISSStore --> EmbedMgr
    EmbedMgr --> TrainScript
    EmbedMgr --> RetrieveScript
    
    CLIPModel --> RetrieveScript
    FAISSStore --> RetrieveScript
    
    Utils --> CleanScript
    Utils --> FetchScript
    
    CLIPModel --> ExampleScript
    TrainScript --> ExampleScript
    RetrieveScript --> ExampleScript
    
    style External fill:#e0e0e0,color:#000
    style ModelCore fill:#f3e5f5
    style VectorCore fill:#fff3e0
    style TrainingCore fill:#fce4ec
    style RetrievalCore fill:#e8f5e9
    style DataCore fill:#f1f8e9
    style ExampleCore fill:#fff9c4
```

---

## Key Workflows

### Workflow 1: End-to-End Training

```mermaid
sequenceDiagram
    participant User
    participant DataCleaning as Data Cleaning
    participant DataSample as Sample Creation
    participant Trainer as Trainer (FAISS)
    participant Model as CLIP Model
    participant FAISS as FAISS Index
    participant Disk as Disk Storage

    User->>DataCleaning: Run clean_annotations.py
    DataCleaning->>Disk: Save grouped captions
    
    User->>DataSample: Run fetch_sample_captions.py
    DataSample->>Disk: Save 2000 sample images
    
    User->>Trainer: trainer.train(num_epochs=10)
    
    loop For each epoch
        Trainer->>Model: Load batch images & captions
        Model->>Model: Image Encoder → Embeddings
        Model->>Model: Text Encoder → Embeddings
        Model->>Model: Calculate Contrastive Loss
        Model->>Model: Backward Pass (Update Weights)
        
        Trainer->>FAISS: Add Image Embeddings
        Trainer->>FAISS: Store Image Names (Metadata)
    end
    
    Trainer->>Disk: Save Model Checkpoint
    Trainer->>Disk: Save FAISS Index (.index)
    Trainer->>Disk: Save Metadata (.json)
    
    Trainer->>User: Training Complete ✓
```

### Workflow 2: Image Search & Retrieval

```mermaid
sequenceDiagram
    participant User
    participant Retriever as ImageRetriever
    participant Model as Model
    participant FAISS as FAISS Index
    participant FileSystem as File System
    participant User2 as Results

    User->>Retriever: Initialize with checkpoint
    Retriever->>Disk: Load model weights
    Retriever->>Disk: Load FAISS index
    Retriever->>Disk: Load metadata
    
    User->>Retriever: search_by_image('query.jpg', k=10)
    
    Retriever->>FileSystem: Load image from disk
    Retriever->>Model: Preprocess image
    Retriever->>Model: Get image embedding
    
    Retriever->>FAISS: search(embedding, k=10)
    FAISS->>FAISS: Compute L2 distances
    FAISS->>FAISS: Get top-10 results
    
    FAISS->>Retriever: Return indices & distances
    Retriever->>Retriever: Get image names from metadata
    Retriever->>FileSystem: Locate images on disk
    
    Retriever->>User2: Return results<br/>(rank, name, path, similarity)
```

---

## Data Structures

### Batch Format (DataLoader)

```python
batch = {
    'images': torch.Tensor,           # Shape: (32, 3, 224, 224)
    'text_tokens': torch.Tensor,      # Shape: (32, 77)
    'text_mask': torch.Tensor,        # Shape: (32, 77)
    'image_names': List[str]          # ['img_001.jpg', 'img_002.jpg', ...]
}
```

### FAISS Metadata Structure

```json
{
  "embedding_dim": 512,
  "num_vectors": 2000,
  "image_names": [
    "image_000001.jpg",
    "image_000002.jpg",
    ...
    "image_002000.jpg"
  ]
}
```

### Search Result Format

```python
result = {
    'rank': 1,                    # Rank in results
    'image_name': 'img_001.jpg',  # Image identifier
    'distance': 0.0234,           # L2 distance (lower = better)
    'similarity': 0.977,          # Normalized similarity (0-1)
    'found_on_disk': True,        # Whether file exists
    'disk_path': '/path/to/img'   # Full file path
}
```

### Model Output

```python
image_embeddings, text_embeddings = model(
    images,        # (B, 3, 224, 224)
    text_tokens,   # (B, 77)
    text_mask      # (B, 77)
)
# image_embeddings: (B, 512)
# text_embeddings: (B, 512)
```

---

## Configuration Summary

### Training Configuration

```python
config = {
    'embedding_dim': 512,
    'text_max_seq_length': 77,
    'vocab_size': 10000,
    'num_text_layers': 12,
    'num_text_heads': 8,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-6,
    'num_epochs': 10,
    'temperature': 0.07,          # For contrastive loss
    'checkpoint_interval': 1,      # Save every N epochs
    'vector_store_dir': 'vector_store'
}
```

### FAISS Configuration

```python
faiss_config = {
    'embedding_dim': 512,
    'index_type': 'IndexFlatL2',   # Can upgrade to IndexIVFFlat
    'index_path': 'vector_store/image_embeddings.index',
    'metadata_path': 'vector_store/image_embeddings_metadata.json',
    'normalize': True              # L2 normalize vectors
}
```

---

## Summary

Your architecture consists of:

1. **Data Pipeline**: Raw COCO → Cleaned → Sampled (2000 images)
2. **Model Pipeline**: Images & Captions → Encoders → 512-dim Embeddings
3. **Loss Pipeline**: Contrastive loss between image and text embeddings
4. **Storage Pipeline**: Embeddings → FAISS Index + Metadata
5. **Retrieval Pipeline**: Query Image → Embedding → FAISS Search → Results

This creates a scalable, efficient image retrieval system that can handle millions of images while maintaining fast search capabilities!
