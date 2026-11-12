#!/usr/bin/env python3
"""Export compact CIFAR-10 embeddings and thumbnails for the static demos.

The atlas and the per-image card share this export.  It loads the model-only
part of a FastJEPA checkpoint, runs the encoder/projector/probe in eval mode on
a balanced subset of the CIFAR-10 test set, and writes:

  assets/atlas/cifar_test.webp   one 32x32 tile per selected test image
  assets/atlas/cifar_views.webp  four 32x32 augmented-view tiles per image
  cifar_atlas/atlas_data.js      window.AtlasData = {...}
  cifar_atlas/atlas_check.json   a handful of values for the PyTorch test

The model definitions below are copied from ``pysrc/lejepa/model.py`` so this
script does not need the fast_JEPA source tree on ``sys.path``.  Everything is
deterministic: the subset, the four augmented views and the projections use
recorded seeds.

Example (from the learn_jepa root):

  ~/projects/fast_jepa/.venv/bin/python cifar_atlas/generate.py \
      --checkpoint checkpoints/cifar_400epochs/ckpt.pt \
      --data-root ../fast_jepa/data

Use ``--skip-tsne`` for a quick PCA-only build.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

try:  # timm is only needed to rebuild the encoder
    import timm
except ImportError as exc:  # pragma: no cover - the error message is the point
    raise SystemExit(
        "timm is required. Use the fast_JEPA virtualenv, e.g. "
        "~/projects/fast_jepa/.venv/bin/python cifar_atlas/generate.py"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
TILE = 32
VIEWS = 4


class LeJEPAEncoder(nn.Module):
    """Same module layout as ``pysrc/lejepa/model.py``."""

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=3,
        drop_path_rate=0.1,
        dynamic_img_size=True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            dynamic_img_size=dynamic_img_size,
        )
        self.proj_head = nn.Linear(embed_dim, 512)

    def forward(self, x):
        return self.proj_head(self.vit(x))


class Projector(nn.Module):
    """Same module layout as ``pysrc/lejepa/model.py``."""

    def __init__(self, in_dim=512, hidden=512, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _solarize(threshold, p):
    random_solarize = getattr(transforms, "RandomSolarize", None)
    if random_solarize is not None:
        return random_solarize(threshold=threshold, p=p)
    return transforms.RandomApply([transforms.Solarize(threshold=threshold)], p=p)


def build_transform(img_size, train):
    """Augmentation recipe from protocol section 5.5 (small-scale M recipe)."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5
                ),
                _solarize(threshold=128, p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def default_checkpoint(repo_root):
    candidates = [
        repo_root / "checkpoints" / "cifar_400epochs" / "ckpt.pt",
        repo_root / "runs" / "checkpoints" / "cifar_400epochs" / "ckpt.pt",
        repo_root / "runs" / "cifar10_400epochs" / "ckpt.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path, repo_root):
    """A portable path for the public data file, not an absolute local path."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(repo_root).resolve())).as_posix()
    except ValueError:
        return Path(path).name


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    enc = LeJEPAEncoder(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        drop_path_rate=cfg["drop_path_rate"],
        dynamic_img_size=cfg["dynamic_img_size"],
    )
    proj = Projector(
        in_dim=512,
        hidden=cfg["proj_hidden"],
        out_dim=cfg["proj_dim"],
    )
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, cfg["num_classes"]))
    enc.load_state_dict(checkpoint["enc"], strict=True)
    proj.load_state_dict(checkpoint["proj"], strict=True)
    probe.load_state_dict(checkpoint["probe"], strict=True)
    for module in (enc, proj, probe):
        module.eval()
        module.to(device)
    return checkpoint, cfg, enc, proj, probe


def choose_subset(dataset, per_class, seed):
    """A balanced, deterministic subset of the test set, sorted by index."""
    rng = np.random.default_rng(seed)
    by_class = {index: [] for index in range(len(CLASS_NAMES))}
    for index in range(len(dataset)):
        _, label = dataset[index]
        by_class[int(label)].append(index)
    chosen = []
    for label in range(len(CLASS_NAMES)):
        pool = np.asarray(by_class[label])
        rng.shuffle(pool)
        chosen.extend(int(index) for index in pool[:per_class])
    chosen.sort()
    return chosen


def denormalize(tensor):
    """Back to an RGB uint8 tile after Normalize()."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    array = (tensor.detach().cpu() * std + mean).clamp(0, 1).mul(255).byte()
    return array.permute(1, 2, 0).numpy()


@torch.no_grad()
def compute_center_embeddings(enc, proj, probe, dataset, indices, device, batch_size=128):
    tf = build_transform(32, train=False)
    cls_rows, head_rows, proj_rows, logits_rows = [], [], [], []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = torch.stack([tf(dataset[index][0]) for index in batch_indices]).to(device)
        feat = enc.vit(images)
        head = enc.proj_head(feat)
        z = proj(head)
        logits = probe(head)
        cls_rows.append(feat.float().cpu())
        head_rows.append(head.float().cpu())
        proj_rows.append(z.float().cpu())
        logits_rows.append(logits.float().cpu())
    return (
        torch.cat(cls_rows).numpy(),
        torch.cat(head_rows).numpy(),
        torch.cat(proj_rows).numpy(),
        torch.cat(logits_rows).numpy(),
    )


@torch.no_grad()
def compute_view_embeddings(enc, proj, dataset, indices, seed, device, tile=32):
    """Four training-recipe augmentations per image, plus display tiles."""
    tf = build_transform(32, train=True)
    tensors = []
    view_tiles = []
    for index in indices:
        image_seed = (seed * 1000003 + int(index) * 7919) % 2147483647
        torch.manual_seed(image_seed)
        random.seed(image_seed)
        views = torch.stack([tf(dataset[index][0]) for _ in range(VIEWS)])
        tensors.append(views)
        for view in range(VIEWS):
            view_tiles.append(denormalize(views[view]))

    rows = []
    for start in range(0, len(tensors), 32):
        batch = torch.stack(tensors[start : start + 32]).to(device)  # (B, V, C, H, W)
        batch = batch.reshape(-1, *batch.shape[2:])
        feat = enc.vit(batch)
        head = enc.proj_head(feat)
        z = proj(head)
        rows.append(z.float().cpu())
    view_z = torch.cat(rows).numpy().reshape(len(indices), VIEWS, -1)
    return view_z, view_tiles


def center_rows(x):
    return x - x.mean(axis=0, keepdims=True)


def pca2(x):
    """Top two principal components, with a deterministic sign convention."""
    x = np.asarray(x, dtype=np.float32)
    x = center_rows(x)
    denom = max(1, x.shape[0] - 1)
    cov = (x.T @ x) / denom
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    for column in range(min(2, evecs.shape[1])):
        pivot = int(np.argmax(np.abs(evecs[:, column])))
        if evecs[pivot, column] < 0:
            evecs[:, column] *= -1
    coords = x @ evecs[:, :2].astype(np.float32)
    total = evals.sum()
    explained = (evals[:2] / total).tolist() if total > 0 else [0.0, 0.0]
    return coords.astype(np.float32), explained


def _tsne_affinities(x, perplexity=30.0, device="cpu"):
    """Row-normalised t-SNE affinities, vectorised over points."""
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[1] > 50:
        cov = x.T @ x / max(1, x.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)[:50]
        x = x @ evecs[:, order]
    dist2 = torch.cdist(x, x, p=2).pow(2)
    n = x.shape[0]
    target = math.log(perplexity)
    beta = torch.ones(n, device=device)
    lo = torch.full((n,), 1e-20, device=device)
    hi = torch.full((n,), 1e20, device=device)
    rows = torch.arange(n, device=device)
    for _ in range(70):
        weights = torch.exp(-dist2 * beta[:, None])
        weights[rows, rows] = 0
        total = weights.sum(dim=1).clamp_min(1e-30)
        entropy = torch.log(total) + beta * (dist2 * weights).sum(dim=1) / total
        too_high = entropy > target
        new_lo = torch.where(too_high, beta, lo)
        new_hi = torch.where(too_high, hi, beta)
        new_beta = torch.where(
            too_high,
            torch.where(hi >= 1e19, beta * 2.0, (beta + hi) / 2.0),
            (beta + lo) / 2.0,
        )
        lo, hi, beta = new_lo, new_hi, new_beta
    weights = torch.exp(-dist2 * beta[:, None])
    weights[rows, rows] = 0
    p = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-30)
    p = (p + p.T) / (2.0 * n)
    return p.clamp_min(1e-12)


def tsne2(x, seed=0, device="cpu", n_iter=900):
    """Small exact t-SNE implementation, good enough for a 2-D atlas."""
    p = _tsne_affinities(x, perplexity=30.0, device=device)
    n = p.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    y = torch.randn((n, 2), generator=generator, device=device, dtype=torch.float32) * 1e-4
    velocity = torch.zeros_like(y)
    learning_rate = 180.0
    momentum = 0.5
    for iteration in range(n_iter):
        exaggeration = 12.0 if iteration < 250 else 1.0
        sum_y = y.square().sum(dim=1)
        dist2 = (sum_y[:, None] + sum_y[None, :] - 2.0 * (y @ y.T)).clamp_min(0)
        inv = 1.0 / (1.0 + dist2)
        inv.fill_diagonal_(0)
        q = (inv / inv.sum().clamp_min(1e-30)).clamp_min(1e-12)
        pq = (p * exaggeration - q) * inv
        gradient = 4.0 * (pq.sum(dim=1, keepdim=True) * y - pq @ y)
        velocity = momentum * velocity - learning_rate * gradient
        y = y + velocity
        y = y - y.mean(dim=0, keepdim=True)
        if iteration == 250:
            momentum = 0.8
    return y.cpu().numpy().astype(np.float32)


def quantize_int8(x):
    x = np.asarray(x, dtype=np.float32)
    scale = np.maximum(np.abs(x).max(axis=0) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale


def int8_payload(x):
    q, scale = quantize_int8(x)
    return {
        "encoding": "int8",
        "rows": int(q.shape[0]),
        "dim": int(q.shape[1]),
        "scale": [round(float(value), 8) for value in scale],
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def float32_payload(x):
    array = np.asarray(x, dtype="<f4")
    return {
        "encoding": "float32",
        "rows": int(array.shape[0]),
        "dim": int(array.shape[-1]),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def make_sprite(tiles, tile, cols, path):
    from PIL import Image

    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile, rows * tile), (244, 246, 248))
    for index, array in enumerate(tiles):
        image = Image.fromarray(np.asarray(array, dtype=np.uint8))
        x = (index % cols) * tile
        y = (index // cols) * tile
        sheet.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "WEBP", quality=87, method=6)
    return {"src": path.name, "cols": cols, "rows": rows, "tile": tile, "count": len(tiles)}


def knn_accuracy(x, labels, k=20, device="cpu"):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    correct = 0
    chunk = 512
    for start in range(0, x.shape[0], chunk):
        sim = x[start : start + chunk] @ x.T
        local = torch.arange(sim.shape[0], device=device)
        sim[local, start + local] = -float("inf")
        neighbours = sim.topk(min(k, x.shape[0] - 1), dim=1).indices
        votes = labels[neighbours]
        prediction = votes.mode(dim=1).values
        correct += int((prediction == labels[start : start + chunk]).sum())
    return correct / max(1, x.shape[0])


def mean_direction_ratio(x):
    x = np.asarray(x, dtype=np.float32)
    mean_norm = float(np.linalg.norm(x.mean(axis=0)))
    row_norm = float(np.linalg.norm(x, axis=1).mean())
    return mean_norm / row_norm if row_norm > 0 else 0.0


def prediction_metrics(view_z, seed, device="cpu"):
    z = torch.as_tensor(view_z, dtype=torch.float32, device=device)
    mu = z.mean(dim=1, keepdim=True)
    paired = float(((mu - z) ** 2).mean())
    generator = torch.Generator(device=device).manual_seed(seed)
    shuffled = torch.stack(
        [z[torch.randperm(z.shape[0], generator=generator, device=device), view] for view in range(z.shape[1])],
        dim=1,
    )
    mu_shuffled = shuffled.mean(dim=1, keepdim=True)
    control = float(((mu_shuffled - shuffled) ** 2).mean())
    return paired, control


def round_coords(coords, digits=4):
    return [[round(float(x), digits), round(float(y), digits)] for x, y in coords]


def write_data_js(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "/* Generated by cifar_atlas/generate.py; do not edit by hand. */\n"
    text += "window.AtlasData = "
    text += json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    text += ";\n"
    path.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("../fast_jepa/data"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--view-seed", type=int, default=4242)
    parser.add_argument("--tsne-seed", type=int, default=7)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    checkpoint_path = (args.checkpoint or default_checkpoint(repo_root)).resolve()
    data_root = args.data_root.resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not (data_root / "cifar-10-batches-py").exists():
        raise SystemExit(f"CIFAR-10 python batches not found under {data_root}")

    device = torch.device(args.device)
    checkpoint, cfg, enc, proj, probe = load_models(checkpoint_path, device)

    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=None)
    indices = choose_subset(dataset, args.per_class, args.subset_seed)
    labels = np.asarray([dataset[index][1] for index in indices], dtype=np.int64)
    print(f"selected {len(indices)} test images ({args.per_class} per class)")

    print("running center-crop embeddings in eval mode ...")
    cls, head, center_z, logits = compute_center_embeddings(
        enc, proj, probe, dataset, indices, device
    )
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    preds = probabilities.argmax(axis=1).astype(np.int64)
    top3 = []
    for row in probabilities:
        order = np.argsort(row)[::-1][:3]
        top3.append([[int(label), round(float(row[label]), 5)] for label in order])
    probe_accuracy = float((preds == labels).mean())

    print("running four augmented views per image ...")
    view_z, view_tiles = compute_view_embeddings(
        enc, proj, dataset, indices, args.view_seed, device, tile=TILE
    )
    print(f"view embeddings: {view_z.shape}")

    print("computing local view layouts ...")
    view2d = np.zeros((len(indices), VIEWS, 2), dtype=np.float32)
    for row in range(len(indices)):
        coords, _ = pca2(view_z[row])
        view2d[row] = coords

    print("computing projections ...")
    projections = {}
    knn_scores = {}
    mean_ratios = {}
    space_arrays = {"cls": cls, "head": head, "proj": center_z}
    for name, raw in space_arrays.items():
        projections[name] = {}
        knn_scores[name] = {}
        mean_ratios[name] = mean_direction_ratio(raw)
        for centering in ("raw", "centered"):
            x = raw if centering == "raw" else center_rows(raw)
            coords, explained = pca2(x)
            entry = {"pca": {"xy": round_coords(coords), "explained": [round(v, 6) for v in explained]}}
            if not args.skip_tsne:
                print(f"  t-SNE {name}/{centering} ...")
                tsne = tsne2(
                    x,
                    seed=args.tsne_seed + len(name) * 17 + int(centering == "centered"),
                    device=str(device),
                )
                entry["tsne"] = {"xy": round_coords(tsne, 3)}
            projections[name][centering] = entry
            knn_scores[name][centering] = round(knn_accuracy(x, labels, k=20, device=str(device)), 4)

    print("computing paired vs shuffled prediction loss ...")
    paired, shuffled = prediction_metrics(view_z, seed=args.view_seed, device=str(device))

    print("writing sprites ...")
    center_tiles = [np.asarray(dataset[index][0], dtype=np.uint8) for index in indices]
    sprite_dir = repo_root / "assets" / "atlas"
    image_sprite = make_sprite(center_tiles, TILE, 64, sprite_dir / "cifar_test.webp")
    view_sprite = make_sprite(view_tiles, TILE, 64, sprite_dir / "cifar_views.webp")

    payload = {
        "meta": {
            "schema": "fastjepa.learning_atlas/1",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_run": display_path(checkpoint_path.parent, repo_root),
            "checkpoint": display_path(checkpoint_path, repo_root),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "git_commit": checkpoint.get("git_commit"),
            "training_epoch": checkpoint.get("epoch"),
            "config": cfg,
            "dataset": "cifar10",
            "split": "test",
            "subset": {
                "images": len(indices),
                "per_class": args.per_class,
                "seed": args.subset_seed,
                "balanced": True,
            },
            "views": VIEWS,
            "view_seed": args.view_seed,
            "eval_mode": "eval",
            "image_size": TILE,
            "class_names": CLASS_NAMES,
            "probe_accuracy": round(probe_accuracy, 4),
            "knn_accuracy": knn_scores,
            "mean_direction_ratio": {name: round(value, 4) for name, value in mean_ratios.items()},
            "prediction_loss": {
                "paired": round(paired, 4),
                "shuffled": round(shuffled, 4),
                "shuffle_seed": args.view_seed,
            },
            "sprite_dir": "../assets/atlas",
            "notes": (
                "Center-crop embeddings use eval mode, so BatchNorm uses running "
                "statistics. The four views use the training augmentation recipe "
                "and are only used for the per-image card."
            ),
        },
        "indices": indices,
        "labels": labels.tolist(),
        "preds": preds.tolist(),
        "top3": top3,
        "spaces": {
            "cls": int8_payload(cls),
            "head": int8_payload(head),
            "proj": float32_payload(center_z),
        },
        "views": {
            "shape": [len(indices), VIEWS, int(center_z.shape[1])],
            "encoding": "float32",
            "data": base64.b64encode(np.asarray(view_z, dtype="<f4").tobytes()).decode("ascii"),
        },
        "view2d": {
            "shape": [len(indices), VIEWS, 2],
            "encoding": "float32",
            "data": base64.b64encode(view2d.astype("<f4").tobytes()).decode("ascii"),
        },
        "projections": projections,
        "sprites": {"images": image_sprite, "views": view_sprite},
    }
    data_path = repo_root / "cifar_atlas" / "atlas_data.js"
    write_data_js(data_path, payload)
    print(f"wrote {data_path} ({data_path.stat().st_size / 1e6:.1f} MB)")

    # A small fixture for tests/test_learning_export.py.  It only contains a few
    # rows, not the whole export.
    sample_rows = sorted({0, len(indices) // 4, len(indices) // 2, (3 * len(indices)) // 4, len(indices) - 1})
    view_sample_rows = sorted({0, len(indices) // 3, (2 * len(indices)) // 3})
    check = {
        "schema": "fastjepa.learning_atlas_check/1",
        "checkpoint_sha256": payload["meta"]["checkpoint_sha256"],
        "indices": [int(indices[row]) for row in sample_rows],
        "eval_samples": [
            {
                "subset_row": int(row),
                "test_index": int(indices[row]),
                "label": int(labels[row]),
                "pred": int(preds[row]),
                "cls": [round(float(value), 6) for value in cls[row]],
                "head": [round(float(value), 6) for value in head[row]],
                "proj": [round(float(value), 6) for value in center_z[row]],
                "logits": [round(float(value), 6) for value in logits[row]],
            }
            for row in sample_rows
        ],
        "view_samples": [
            {
                "subset_row": int(row),
                "test_index": int(indices[row]),
                "seed": int((args.view_seed * 1000003 + int(indices[row]) * 7919) % 2147483647),
                "proj": [[round(float(value), 6) for value in view] for view in view_z[row]],
            }
            for row in view_sample_rows
        ],
    }
    check_path = repo_root / "cifar_atlas" / "atlas_check.json"
    check_path.write_text(json.dumps(check, indent=2) + "\n")
    print(f"wrote {check_path}")
    print(
        f"probe top-1 {probe_accuracy:.2%} | paired {paired:.4f} | "
        f"shuffled {shuffled:.4f} | assets {image_sprite['count']} + {view_sprite['count']} tiles"
    )


if __name__ == "__main__":
    main()
