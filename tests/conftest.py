"""
Pytest fixtures shared across animoflow-app tests.

Key constraint: tests must pass on a CPU laptop with no GPU, no model
weights, and (ideally) no Blender. We mock model inference and skip
end-to-end pipeline tests when blender / weights are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Make app modules importable
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(scope="session", autouse=True)
def _set_test_env():
    """Set test-friendly env vars before any module imports.

    Uses a well-known sub-path of /tmp instead of pytest's tmp_path_factory
    because some FUSE-mounted sandboxes throw on rmdir of session tmpdirs,
    sending pytest's cleanup into recursion. Stable per-test-run path is
    safer and behaves identically across hosts.
    """
    test_root = Path("/tmp/animoflow-app-test")
    os.environ["OUTPUT_DIR"] = str(test_root / "output")
    os.environ.setdefault("WEB_DIR", "/nonexistent")
    os.environ.setdefault("CHECKPOINTS_DIR", str(test_root / "checkpoints"))
    os.environ.setdefault("HEALTH_POLL_INTERVAL", "999999")  # don't poll during tests
    Path(os.environ["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["CHECKPOINTS_DIR"]).mkdir(parents=True, exist_ok=True)
    yield


def _can_import(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001 — we want all import errors to count
        return False
    return True


HAS_GRADIO = _can_import("gradio")
HAS_FASTAPI = _can_import("fastapi")
HAS_HTTPX = _can_import("httpx")
HAS_TORCH = _can_import("torch")
HAS_NUMPY = _can_import("numpy")

ANIMOFLOW_API_API = Path(
    os.environ.get("ANIMOFLOW_API_API_DIR", "/opt/animoflow-api/api")
)
HAS_ANIMOFLOW_API = ANIMOFLOW_API_API.is_dir()

COMFYUI_ANIMOFLOW = Path(
    os.environ.get("COMFYUI_ANIMOFLOW_DIR", "/opt/comfyui-animoflow")
)
HAS_COMFYUI_ANIMOFLOW = COMFYUI_ANIMOFLOW.is_dir()

BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", "/opt/blender/blender"))
HAS_BLENDER = BLENDER_BIN.exists()

