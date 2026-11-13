import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_learning_pages_and_assets_exist():
    expected = [
        ROOT / "index.html",
        ROOT / "README.md",
        ROOT / "STARTING_GUIDE.md",
        ROOT / "assets" / "learning.css",
        ROOT / "assets" / "canvas.js",
        ROOT / "assets" / "sigreg_core.js",
        ROOT / "sigreg_lab" / "index.html",
        ROOT / "sigreg_lab" / "app.js",
        ROOT / "loss_calculator" / "index.html",
        ROOT / "loss_calculator" / "app.js",
        ROOT / "loss_calculator" / "loss_core.js",
        ROOT / "assets" / "atlas_core.js",
        ROOT / "assets" / "atlas_ui.js",
        ROOT / "assets" / "projector_core.js",
        ROOT / "assets" / "atlas" / "cifar_test.webp",
        ROOT / "assets" / "atlas" / "cifar_views.webp",
        ROOT / "cifar_atlas" / "generate.py",
        ROOT / "cifar_atlas" / "package_model.py",
        ROOT / "cifar_atlas" / "index.html",
        ROOT / "cifar_atlas" / "app.js",
        ROOT / "cifar_atlas" / "atlas_data.js",
        ROOT / "per_image" / "index.html",
        ROOT / "per_image" / "app.js",
        ROOT / "space_lens" / "index.html",
        ROOT / "space_lens" / "app.js",
        ROOT / "projector_diagnostics" / "index.html",
        ROOT / "projector_diagnostics" / "app.js",
        ROOT / "collapse_sim" / "index.html",
        ROOT / "collapse_sim" / "app.js",
        ROOT / "collapse_sim" / "collapse_core.js",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    assert not missing, f"missing learning files: {missing}"


def test_learning_pages_are_self_contained():
    pages = [
        ROOT / "index.html",
        ROOT / "sigreg_lab" / "index.html",
        ROOT / "loss_calculator" / "index.html",
        ROOT / "cifar_atlas" / "index.html",
        ROOT / "per_image" / "index.html",
        ROOT / "space_lens" / "index.html",
        ROOT / "projector_diagnostics" / "index.html",
        ROOT / "collapse_sim" / "index.html",
    ]
    external = re.compile(r"<(?:script|link|img|iframe)[^>]+(?:src|href)=\"https?://")
    for page in pages:
        text = page.read_text()
        assert not external.search(text), f"{page} loads an external resource"
        assert "learning.css" in text
    for page in pages[1:]:
        assert "<script" in page.read_text()


def test_learning_javascript_suite():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    tests = sorted((ROOT / "tests").glob("*.test.js"))
    assert tests, "no JavaScript tests found"
    result = subprocess.run(
        [node, "--test", *map(str, tests)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
