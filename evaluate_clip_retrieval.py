"""
Evaluate the saved trained_model_clip with retrieval ranking metrics.

This script evaluates:
- Text-to-Image retrieval
- Image-to-Text retrieval

Metrics reported per task:
- R@1, R@5, R@10
- MRR
- MedR
- MeanR
- Random baseline (RndR@K)
- Lift@K = R@K / RndR@K
- Stability across multiple passes (mean/std)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import clip


@dataclass
class Sample:
    image_id: int
    file_name: str
    caption: str


class CLIPImageEncoder(nn.Module):
    """Matches saved model style: ResNet50 + 2-layer head."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class CLIPTextEncoder(nn.Module):
    """Matches saved model style: OpenAI CLIP text encoder + projection."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_model = clip_model
        self.projection = nn.Linear(512, embedding_dim)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        return self.projection(text_features)


class CLIPModel(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.image_encoder = CLIPImageEncoder(embedding_dim=embedding_dim)
        self.text_encoder = CLIPTextEncoder(embedding_dim=embedding_dim)

    def forward_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def forward_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_tokens)


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform: T.Compose):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        image = Image.open(p).convert("RGB")
        return self.transform(image), str(p)


def load_coco_like_annotations(annotations_path: Path) -> Tuple[List[Sample], Dict[int, str]]:
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Sample] = []
    image_id_to_file: Dict[int, str] = {}

    if "images" in data and "annotations" in data:
        for img in data["images"]:
            image_id_to_file[int(img["id"])] = img["file_name"]
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            if image_id in image_id_to_file:
                samples.append(
                    Sample(
                        image_id=image_id,
                        file_name=image_id_to_file[image_id],
                        caption=str(ann.get("caption", "")),
                    )
                )
    elif "annotations" in data and isinstance(data["annotations"], list):
        # Cleaned format with {"id": int, "captions": [...]}
        for item in data["annotations"]:
            image_id = int(item["id"])
            file_name = f"{image_id:012d}.jpg"
            image_id_to_file[image_id] = file_name
            for cap in item.get("captions", []):
                samples.append(Sample(image_id=image_id, file_name=file_name, caption=str(cap)))
    else:
        raise ValueError("Unsupported annotations format.")

    return samples, image_id_to_file


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def image_transform(stochastic: bool) -> T.Compose:
    if stochastic:
        return T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def encode_images(
    model: CLIPModel,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int,
    stochastic: bool,
) -> np.ndarray:
    ds = ImageOnlyDataset(image_paths=image_paths, transform=image_transform(stochastic))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, _ in dl:
            images = images.to(device)
            emb = model.forward_image(images)
            emb = l2_normalize(emb).cpu().numpy()
            all_embs.append(emb)

    if not all_embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def encode_texts(
    model: CLIPModel,
    captions: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_embs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            batch_caps = list(captions[start : start + batch_size])
            text_tokens = clip.tokenize(batch_caps, truncate=True).to(device)
            emb = model.forward_text(text_tokens=text_tokens)
            emb = l2_normalize(emb).cpu().numpy()
            all_embs.append(emb)

    if not all_embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def first_relevant_rank(sorted_indices: np.ndarray, relevant: Set[int]) -> Optional[int]:
    for pos, idx in enumerate(sorted_indices.tolist(), start=1):
        if idx in relevant:
            return pos
    return None


def expected_random_recall(n_candidates: int, n_relevant: int, k: int) -> float:
    if n_candidates <= 0:
        return 0.0
    k = min(k, n_candidates)
    if n_relevant <= 0:
        return 0.0
    if n_relevant >= n_candidates:
        return 1.0
    miss_prob = math.comb(n_candidates - n_relevant, k) / math.comb(n_candidates, k)
    return 1.0 - miss_prob


def compute_metrics(
    similarity: np.ndarray,
    ground_truth: Sequence[Set[int]],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    n_queries, n_candidates = similarity.shape
    recalls = {k: 0 for k in ks}
    rnd_recalls = {k: 0.0 for k in ks}
    reciprocal_ranks: List[float] = []
    ranks: List[int] = []

    for q in range(n_queries):
        rel = ground_truth[q]
        order = np.argsort(-similarity[q])
        rank = first_relevant_rank(order, rel)
        if rank is None:
            rank = n_candidates + 1
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)
        ranks.append(rank)

        for k in ks:
            if rank <= k:
                recalls[k] += 1
            rnd_recalls[k] += expected_random_recall(n_candidates=n_candidates, n_relevant=len(rel), k=k)

    metrics: Dict[str, float] = {}
    for k in ks:
        r_at_k = recalls[k] / n_queries if n_queries else 0.0
        rnd = rnd_recalls[k] / n_queries if n_queries else 0.0
        lift = (r_at_k / rnd) if rnd > 0 else 0.0
        metrics[f"R@{k}"] = r_at_k
        metrics[f"RndR@{k}"] = rnd
        metrics[f"Lift@{k}"] = lift

    metrics["MRR"] = mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    metrics["MedR"] = float(median(ranks)) if ranks else 0.0
    metrics["MeanR"] = float(mean(ranks)) if ranks else 0.0
    return metrics


def _dcg_at_k(gains: np.ndarray, k: int) -> float:
    if gains.size == 0:
        return 0.0
    k = min(k, gains.size)
    if k <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains[:k] * discounts))


def compute_contextual_metrics(
    retrieval_similarity: np.ndarray,
    contextual_relevance: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    relevance_threshold: float = 0.30,
) -> Dict[str, float]:
    """
    Context-oriented metrics where relevance is graded (not exact identity match).

    - nDCG@K over graded gains from contextual_relevance
    - P@K_ctx where candidate is relevant if relevance >= threshold
    - CLIPScore@K = mean contextual relevance in top-K
    """
    n_queries, n_candidates = retrieval_similarity.shape
    out: Dict[str, float] = {}

    ndcg_vals = {k: [] for k in ks}
    p_ctx_vals = {k: [] for k in ks}
    clipscore_vals = {k: [] for k in ks}

    for q in range(n_queries):
        rank_idx = np.argsort(-retrieval_similarity[q])
        rel = contextual_relevance[q]
        ideal_idx = np.argsort(-rel)

        ranked_gains = rel[rank_idx]
        ideal_gains = rel[ideal_idx]

        for k in ks:
            topk = ranked_gains[: min(k, n_candidates)]
            idcg = _dcg_at_k(ideal_gains, k)
            dcg = _dcg_at_k(ranked_gains, k)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_vals[k].append(ndcg)

            if topk.size > 0:
                p_ctx = float(np.mean((topk >= relevance_threshold).astype(np.float32)))
                cscore = float(np.mean(topk))
            else:
                p_ctx = 0.0
                cscore = 0.0
            p_ctx_vals[k].append(p_ctx)
            clipscore_vals[k].append(cscore)

    for k in ks:
        out[f"nDCG@{k}"] = float(mean(ndcg_vals[k])) if ndcg_vals[k] else 0.0
        out[f"P@{k}_ctx"] = float(mean(p_ctx_vals[k])) if p_ctx_vals[k] else 0.0
        out[f"CLIPScore@{k}"] = float(mean(clipscore_vals[k])) if clipscore_vals[k] else 0.0

    return out


def build_contextual_relevance_with_foundation_clip(
    captions: Sequence[str],
    image_paths: Sequence[Path],
    image_to_caption_gt: Sequence[Set[int]],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graded relevance matrices using a frozen pretrained CLIP model.

    Returns:
    - text_to_image_relevance: (num_captions, num_images)
    - image_to_image_relevance: (num_images, num_images)
    """
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Encode images with CLIP foundation model.
    img_ds = ImageOnlyDataset(image_paths=image_paths, transform=clip_preprocess)
    img_dl = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    image_embs: List[np.ndarray] = []
    with torch.no_grad():
        for images, _ in img_dl:
            images = images.to(device)
            emb = clip_model.encode_image(images)
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy()
            image_embs.append(emb)
    image_emb = np.concatenate(image_embs, axis=0).astype(np.float32) if image_embs else np.zeros((0, 512), dtype=np.float32)

    # Encode caption queries with CLIP foundation model.
    text_embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            chunk = list(captions[start : start + batch_size])
            tokens = clip.tokenize(chunk, truncate=True).to(device)
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy()
            text_embs.append(emb)
    text_emb = np.concatenate(text_embs, axis=0).astype(np.float32) if text_embs else np.zeros((0, 512), dtype=np.float32)

    # cosine in [-1, 1], map to [0, 1] for graded gains
    t2i_rel = np.clip((np.matmul(text_emb, image_emb.T) + 1.0) / 2.0, 0.0, 1.0)
    i2i_rel = np.clip((np.matmul(image_emb, image_emb.T) + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(i2i_rel, 0.0)

    # Ensure exact paired positives remain strongly relevant in contextual matrix.
    for img_idx, cap_set in enumerate(image_to_caption_gt):
        for cap_idx in cap_set:
            if cap_idx < t2i_rel.shape[0]:
                t2i_rel[cap_idx, img_idx] = max(t2i_rel[cap_idx, img_idx], 1.0)

    return t2i_rel, i2i_rel


def aggregate_pass_metrics(pass_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted(pass_metrics[0].keys()) if pass_metrics else []
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = [m[key] for m in pass_metrics]
        result[key] = {
            "mean": float(mean(vals)),
            "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
        }
    return result


def resolve_checkpoint(model_dir: Path, checkpoint_name: Optional[str]) -> Path:
    if checkpoint_name:
        path = model_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = ["clip_encoder.pth", "model_state.pt", "clip_model.pth"]
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. Tried: {', '.join(candidates)}"
    )


def pretty_print(task_name: str, aggregated: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'=' * 80}")
    print(f"{task_name}")
    print(f"{'=' * 80}")
    for k, v in aggregated.items():
        print(f"{k:10s} mean={v['mean']:.6f}  std={v['std']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP retrieval metrics.")
    parser.add_argument("--model-dir", type=Path, default=Path("trained_model_clip"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to COCO-style annotations JSON.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory with image files.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, evaluate only first N unique images (faster debug run).",
    )
    parser.add_argument(
        "--max-captions-per-image",
        type=int,
        default=0,
        help="If >0, cap captions per image for faster runs.",
    )
    parser.add_argument(
        "--stochastic-aug",
        action="store_true",
        help="Enable stochastic image augmentation per pass for stability evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("eval_results_clip_metrics.json"),
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=0.30,
        help="Threshold for contextual Precision@K using foundation CLIP relevance.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    config_path = model_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    embedding_dim = int(config.get("embedding_dim", 512))

    checkpoint_path = resolve_checkpoint(model_dir=model_dir, checkpoint_name=args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = CLIPModel(embedding_dim=embedding_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    current_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state.items():
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    model.eval()
    print(f"Loaded {len(compatible_state)} compatible parameters from checkpoint.")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible parameters due to shape/key mismatch.")
    if missing:
        print(f"Missing parameters after load: {len(missing)}")
    if unexpected:
        print(f"Unexpected parameters in checkpoint: {len(unexpected)}")

    samples, image_id_to_file = load_coco_like_annotations(args.annotations)
    if not samples:
        raise ValueError("No caption-image pairs found in annotations.")

    if args.max_captions_per_image > 0:
        per_image_count: Dict[int, int] = {}
        filtered: List[Sample] = []
        for s in samples:
            cnt = per_image_count.get(s.image_id, 0)
            if cnt < args.max_captions_per_image:
                filtered.append(s)
                per_image_count[s.image_id] = cnt + 1
        samples = filtered

    unique_image_ids = sorted(image_id_to_file.keys())
    if args.max_images > 0:
        unique_image_ids = unique_image_ids[: args.max_images]
        allowed = set(unique_image_ids)
        samples = [s for s in samples if s.image_id in allowed]
        image_id_to_file = {k: v for k, v in image_id_to_file.items() if k in allowed}

    image_id_to_index = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    image_paths = [args.images_dir / image_id_to_file[img_id] for img_id in unique_image_ids]

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First few:\n{preview}")

    captions = [s.caption for s in samples]
    print(f"Evaluation set: {len(unique_image_ids)} images, {len(captions)} captions")
    caption_to_image_gt = [{image_id_to_index[s.image_id]} for s in samples]

    image_to_caption_gt: List[Set[int]] = [set() for _ in unique_image_ids]
    for cap_idx, s in enumerate(samples):
        image_to_caption_gt[image_id_to_index[s.image_id]].add(cap_idx)

    print("Building contextual relevance matrix using frozen foundation CLIP...")
    contextual_t2i_rel, contextual_i2i_rel = build_contextual_relevance_with_foundation_clip(
        captions=captions,
        image_paths=image_paths,
        image_to_caption_gt=image_to_caption_gt,
        device=device,
        batch_size=args.batch_size,
    )

    text_to_image_passes: List[Dict[str, float]] = []
    image_to_image_passes: List[Dict[str, float]] = []
    
    # Empty ground truth for exact match since I2I has no predefined class labels
    empty_i2i_gt = [set() for _ in range(len(unique_image_ids))]

    for p in range(args.passes):
        print(f"\nPass {p + 1}/{args.passes}")
        img_emb = encode_images(
            model=model,
            image_paths=image_paths,
            device=device,
            batch_size=args.batch_size,
            stochastic=args.stochastic_aug,
        )
        txt_emb = encode_texts(
            model=model,
            captions=captions,
            device=device,
            batch_size=args.batch_size,
        )

        sim_t2i = np.matmul(txt_emb, img_emb.T)
        sim_i2i = np.matmul(img_emb, img_emb.T)
        np.fill_diagonal(sim_i2i, -1.0) # Prevent finding exact same image

        t2i_metrics = compute_metrics(similarity=sim_t2i, ground_truth=caption_to_image_gt)
        i2i_metrics = compute_metrics(similarity=sim_i2i, ground_truth=empty_i2i_gt)

        t2i_ctx = compute_contextual_metrics(
            retrieval_similarity=sim_t2i,
            contextual_relevance=contextual_t2i_rel,
            ks=(1, 5, 10),
            relevance_threshold=args.context_threshold,
        )
        i2i_ctx = compute_contextual_metrics(
            retrieval_similarity=sim_i2i,
            contextual_relevance=contextual_i2i_rel,
            ks=(1, 5, 10),
            relevance_threshold=args.context_threshold,
        )

        t2i_metrics.update(t2i_ctx)
        i2i_metrics.update(i2i_ctx)

        text_to_image_passes.append(t2i_metrics)
        image_to_image_passes.append(i2i_metrics)

    t2i_agg = aggregate_pass_metrics(text_to_image_passes)
    i2i_agg = aggregate_pass_metrics(image_to_image_passes)

    pretty_print("Text-to-Image Retrieval", t2i_agg)
    pretty_print("Image-to-Image Retrieval", i2i_agg)

    output = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "annotations": str(args.annotations),
        "images_dir": str(args.images_dir),
        "embedding_dim": embedding_dim,
        "passes": args.passes,
        "stochastic_augmentation": bool(args.stochastic_aug),
        "context_threshold": args.context_threshold,
        "text_to_image": {
            "per_pass": text_to_image_passes,
            "aggregate": t2i_agg,
        },
        "image_to_image": {
            "per_pass": image_to_image_passes,
            "aggregate": i2i_agg,
        },
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved detailed results to: {args.output_json}")


if __name__ == "__main__":
    main()
