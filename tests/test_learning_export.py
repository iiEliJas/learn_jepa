"""Contract and PyTorch-parity checks for the CIFAR learning assets.

The first test runs without PyTorch or the checkpoint: it validates the
generated ``atlas_data.js`` shape, ranges and decoded tensors.  The second test
re-runs a few rows through the real model and compares them with the export.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import random
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "cifar_atlas" / "atlas_data.js"
CHECK_PATH = ROOT / "cifar_atlas" / "atlas_check.json"
CHECKPOINT = ROOT / "checkpoints" / "cifar_400epochs" / "ckpt.pt"
DATA_ROOT = ROOT.parent / "fast_jepa" / "data"


def load_data():
    text = DATA_PATH.read_text()
    prefix = "window.AtlasData = "
    start = text.index(prefix) + len(prefix)
    end = text.rindex(";")
    return json.loads(text[start:end])


def decode_int8(payload):
    raw = base64.b64decode(payload["data"])
    return struct.unpack(f"<{len(raw)}b", raw)


def decode_float32(payload):
    raw = base64.b64decode(payload["data"])
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def decode_plain_float32(text):
    raw = base64.b64decode(text)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


@pytest.fixture(scope="module")
def data():
    assert DATA_PATH.exists(), "run cifar_atlas/generate.py first"
    return load_data()


def test_atlas_export_contract(data):
    meta = data["meta"]
    n = meta["subset"]["images"]
    assert meta["schema"] == "fastjepa.learning_atlas/1"
    assert len(data["indices"]) == n
    assert len(data["labels"]) == n
    assert len(data["preds"]) == n
    assert len(data["top3"]) == n
    assert all(0 <= label < 10 for label in data["labels"])
    assert all(0 <= pred < 10 for pred in data["preds"])
    assert all(0 <= index < 10000 for index in data["indices"])
    assert len(set(data["indices"])) == n
    assert all(len(row) == 3 for row in data["top3"])
    assert all(
        0.0 <= probability <= 1.0 for row in data["top3"] for _, probability in row
    )
    assert meta["probe_accuracy"] == pytest.approx(
        sum(label == pred for label, pred in zip(data["labels"], data["preds"])) / n,
        abs=1e-4,
    )

    for name, expected_dim in (("cls", 192), ("head", 512), ("proj", 16)):
        space = data["spaces"][name]
        assert space["dim"] == expected_dim
        assert space["rows"] == n
        raw = base64.b64decode(space["data"])
        if space["encoding"] == "int8":
            assert len(raw) == n * expected_dim
            assert len(space["scale"]) == expected_dim
        else:
            assert space["encoding"] == "float32"
            assert len(raw) == n * expected_dim * 4
            assert all(value == value for value in decode_float32(space))

    views = data["views"]
    assert views["shape"] == [n, 4, 16]
    assert views["encoding"] == "float32"
    assert len(decode_plain_float32(views["data"])) == n * 4 * 16
    assert all(value == value for value in decode_plain_float32(views["data"]))

    view2d = data["view2d"]
    assert view2d["shape"] == [n, 4, 2]
    assert len(decode_plain_float32(view2d["data"])) == n * 4 * 2
    assert all(value == value for value in decode_plain_float32(view2d["data"]))

    loss = meta["prediction_loss"]
    assert 0.0 < loss["paired"] < loss["shuffled"]
    assert loss["shuffled"] / loss["paired"] > 1.5

    for name in ("cls", "head", "proj"):
        for centering in ("raw", "centered"):
            entry = data["projections"][name][centering]
            assert "pca" in entry
            assert len(entry["pca"]["xy"]) == n
            assert all(
                len(point) == 2 and all(abs(value) < 1e6 for value in point)
                for point in entry["pca"]["xy"]
            )
            if "tsne" in entry:
                assert len(entry["tsne"]["xy"]) == n
                assert all(
                    len(point) == 2 and all(abs(value) < 1e6 for value in point)
                    for point in entry["tsne"]["xy"]
                )

    assert data["sprites"]["images"]["count"] == n
    assert data["sprites"]["views"]["count"] == n * 4
    assert data["sprites"]["images"]["tile"] == 32
    assert data["sprites"]["views"]["tile"] == 32


def test_atlas_export_matches_pytorch(data):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    datasets = pytest.importorskip("torchvision.datasets")
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not available: {CHECKPOINT}")
    if not (DATA_ROOT / "cifar-10-batches-py").exists():
        pytest.skip(f"CIFAR-10 not available: {DATA_ROOT}")
    assert CHECK_PATH.exists(), "run cifar_atlas/generate.py first"
    check = json.loads(CHECK_PATH.read_text())
    assert check["checkpoint_sha256"] == data["meta"]["checkpoint_sha256"]

    spec = importlib.util.spec_from_file_location(
        "learning_generate", ROOT / "cifar_atlas" / "generate.py"
    )
    generate = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(generate)

    device = torch.device("cpu")
    _, _, enc, proj, probe = generate.load_models(CHECKPOINT, device)
    dataset = datasets.CIFAR10(
        root=str(DATA_ROOT), train=False, download=False, transform=None
    )
    eval_transform = generate.build_transform(32, train=False)
    view_transform = generate.build_transform(32, train=True)

    for sample in check["eval_samples"]:
        image = eval_transform(dataset[sample["test_index"]][0]).unsqueeze(0)
        with torch.no_grad():
            feat = enc.vit(image)
            head = enc.proj_head(feat)
            z = proj(head)
            logits = probe(head)
        for exported, actual, tolerance in (
            (sample["cls"], feat[0], 2e-4),
            (sample["head"], head[0], 2e-4),
            (sample["proj"], z[0], 2e-4),
            (sample["logits"], logits[0], 2e-3),
        ):
            assert torch.allclose(
                torch.tensor(exported), actual, atol=tolerance, rtol=0
            ), "exported value differs from a fresh PyTorch forward pass"

        row = sample["subset_row"]
        assert data["indices"][row] == sample["test_index"]
        assert data["labels"][row] == sample["label"]
        assert data["preds"][row] == sample["pred"]
        for name, exported, actual in (
            ("cls", sample["cls"], feat[0]),
            ("head", sample["head"], head[0]),
            ("proj", sample["proj"], z[0]),
        ):
            space = data["spaces"][name]
            if space["encoding"] == "int8":
                values = decode_int8(space)
                scale = space["scale"]
                for dim in range(space["dim"]):
                    quantized = values[row * space["dim"] + dim] * scale[dim]
                    assert abs(quantized - float(actual[dim])) <= scale[dim] + 1e-5
            else:
                values = decode_float32(space)
                for dim in range(space["dim"]):
                    assert (
                        abs(values[row * space["dim"] + dim] - float(actual[dim]))
                        <= 2e-4
                    )

    view_values = decode_plain_float32(data["views"]["data"])
    for sample in check["view_samples"]:
        torch.manual_seed(sample["seed"])
        random.seed(sample["seed"])
        views = torch.stack(
            [view_transform(dataset[sample["test_index"]][0]) for _ in range(4)]
        )
        with torch.no_grad():
            feat = enc.vit(views)
            head = enc.proj_head(feat)
            z = proj(head)
        assert torch.allclose(torch.tensor(sample["proj"]), z, atol=2e-4, rtol=0), (
            "view export differs from a fresh augmented forward pass"
        )
        row = sample["subset_row"]
        for view in range(4):
            for dim in range(16):
                exported = view_values[(row * 4 + view) * 16 + dim]
                assert abs(exported - float(z[view, dim])) <= 2e-4
