"""
Import-shape tests — verify app.py wires up without crashing on a CPU laptop.

These tests skip cleanly when the sibling repos (animoflow-api,
comfyui-animoflow) aren't cloned at the expected paths. On a real Docker
build they're always present, so this is the canary that catches our own
import-path mistakes.
"""

from __future__ import annotations

import os
import sys

import pytest

from .conftest import HAS_FASTAPI, HAS_GRADIO, HAS_HTTPX, HAS_ANIMOFLOW_API


@pytest.mark.skipif(not HAS_GRADIO, reason="gradio not installed")
def test_spaces_compat_imports():
    """The compat shim must always import on its own, even without `spaces`."""
    import importlib

    import spaces_compat

    importlib.reload(spaces_compat)
    assert callable(spaces_compat.GPU)
    assert isinstance(spaces_compat.HAVE_SPACES, bool)


@pytest.mark.skipif(
    not (HAS_FASTAPI and HAS_GRADIO and HAS_HTTPX and HAS_ANIMOFLOW_API),
    reason="needs fastapi + gradio + httpx + a cloned animoflow-api at /opt/animoflow-api",
)
def test_app_module_imports():
    """app.py should import without crashing — exercises the monkeypatch order."""
    # This is the actual Docker boot path, just minus uvicorn.
    import importlib

    if "app" in sys.modules:
        importlib.reload(sys.modules["app"])
    else:
        import app  # noqa: F401

    import app

    assert app.fastapi_app is not None
    # Verify the monkeypatch happened
    import pipeline as animoflow_api_pipeline
    import pipeline_hf

    assert animoflow_api_pipeline.run is pipeline_hf.run
    assert animoflow_api_pipeline.run_timeline is pipeline_hf.run_timeline


@pytest.mark.skipif(
    not (HAS_FASTAPI and HAS_GRADIO and HAS_HTTPX and HAS_ANIMOFLOW_API),
    reason="needs fastapi + gradio + httpx + animoflow-api",
)
def test_v1_health_route_exists():
    """The /v1/health route from animoflow-api should still be present after
    Gradio is mounted on '/'."""
    import app

    routes = [getattr(r, "path", None) for r in app.fastapi_app.routes]
    assert "/v1/health" in routes, f"missing /v1/health, got: {routes}"
