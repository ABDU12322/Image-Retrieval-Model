# CLIP Training Model v3 Architecture

This document summarizes the architecture defined in `CLIP_SimCLR_Training_v3 (2).ipynb`.

## High-level Design

The notebook defines two model families:

- `CLIPModel` for cross-modal retrieval (text <-> image)
- `SimCLRModel` for image-only contrastive representation learning

This file focuses on the CLIP side because your saved directory is `trained_model_clip`.

## CLIPModel Components

### 1) `CLIPImageEncoder`

- Backbone: `torchvision.models.resnet50(pretrained=True)`
- The final classification layer is removed/replaced with identity.
- A linear projection maps image features to the shared embedding space:
  - `Linear(2048 -> embedding_dim)`
- Output: image embedding vector in the joint space.

### 2) `CLIPTextEncoder`

- Backbone: `BertModel.from_pretrained("bert-base-uncased")`
- The notebook comments indicate BERT is used with cached CLS vectors during training to reduce cost.
- Projection layer:
  - `Linear(768 -> embedding_dim)`
- Output: text embedding vector in the same joint space as images.

### 3) `CLIPModel`

- Contains:
  - `self.image_encoder = CLIPImageEncoder(embedding_dim)`
  - `self.text_encoder = CLIPTextEncoder(embedding_dim)`
- Exposes separate forward paths:
  - image forward (`forward_image`)
  - text forward (`forward_text` / cached variant in notebook flow)

## Training Behavior Noted in v3 Notebook

- Contrastive training with CLIP-style loss (`CLIPLoss`)
- Training optimizer includes:
  - image encoder parameters
  - text projection parameters
  - loss temperature parameter(s)
- Notebook notes mention BERT is frozen while only the small text projection layer is trained in the shown setup.

## SimCLR (Also Defined in v3)

The notebook additionally defines `SimCLRModel` with:

- Image backbone
- Projection head (`nn.Sequential(...)`)
- SimCLR contrastive objective (`SimCLRLoss`)

This is separate from the CLIP retrieval path.
