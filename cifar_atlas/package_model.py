#!/usr/bin/env python3
"""Strip a training checkpoint down to an inference model asset.

The learning pages only need ``enc``, ``proj``, ``probe`` and ``config``.  The
full checkpoint also carries optimizer, scheduler and RNG state, which roughly
triples the file size.  This script keeps the model parts and adds provenance,
so the result can be published as a small GitHub release asset.

Example:

  python cifar_atlas/package_model.py \
      --checkpoint checkpoints/cifar_400epochs/ckpt.pt \
      --out dist/cifar10_lejepa_400ep_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


KEEP = ("enc", "proj", "probe", "config", "git_commit", "epoch", "step", "metrics")
REQUIRED = ("enc", "proj", "probe", "config")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED if key not in checkpoint]
    if missing:
        raise SystemExit(f"checkpoint is missing {missing}")

    payload = {key: checkpoint[key] for key in KEEP if key in checkpoint}
    payload["format"] = "fastjepa.model_only/1"
    payload["source_checkpoint_sha256"] = sha256_file(args.checkpoint)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    print(f"sha256 {sha256_file(args.out)}")


if __name__ == "__main__":
    main()
