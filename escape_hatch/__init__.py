"""
Escape hatch for outlier-model venvs.

Models whose deps don't merge cleanly with the orchestrator venv (e.g.
Kimodo's NVIDIA stack, future Tier-2 models) live in their own
`/opt/venvs/comfy-X/` venv inside the same Docker image. The
@spaces.GPU-decorated `run_inference` function in pipeline_hf.py
delegates to these via subprocess — the subprocess inherits CUDA
visibility from the GPU fork and pays cold-start (~15s) for full dep
isolation.

The registry currently carries Kimodo; adding further outlier models
is purely additive — no orchestrator changes.
"""

from .invoke import is_outlier, invoke

__all__ = ["is_outlier", "invoke"]
