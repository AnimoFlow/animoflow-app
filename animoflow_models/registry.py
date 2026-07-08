"""
Warm-model registry: imports upstream model inference modules into the
orchestrator process at module level so the @spaces.GPU fork inherits
them via copy-on-write.

Each model wrapper lives in `comfyui-animoflow/containers/<model>/` and
exposes a class (e.g. `MDMInference`) with a `.generate(prompt, …) ->
(npz_bytes, metadata)` method. We don't modify any of that — we just
import it and hold a singleton.

If a model fails to import (missing weights, dep conflict surfacing
later, etc.), it's logged and removed from the active set — the API
will report it as unavailable instead of crashing the whole orchestrator.

Per [[Wrap, don't fork upstream model repos]] and [[animoflow-app — HF +
OSS deployment wrapper]].
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model definitions — declarative.
# Each entry says where the wrapper module lives, which class to instantiate,
# and what the .generate() signature looks like (so we can adapt args).
# ---------------------------------------------------------------------------

_COMFYUI_ANIMOFLOW = Path(
    os.environ.get("COMFYUI_ANIMOFLOW_DIR", "/opt/comfyui-animoflow")
)

_DEFINITIONS: dict[str, dict[str, Any]] = {
    "mdm": {
        "container_dir": _COMFYUI_ANIMOFLOW / "containers" / "mdm",
        "module_name": "inference",
        "class_name": "MDMInference",
        # MDMInference.generate(prompt, num_frames, seed, guidance_param) -> (bytes, dict)
        "generate_kwargs": {
            "guidance_param_default": 7.5,
            "guidance_param_name": "guidance_param",
        },
    },
    "momask": {
        "container_dir": _COMFYUI_ANIMOFLOW / "containers" / "momask",
        "module_name": "inference_momask",
        "class_name": "MoMaskInference",
        # MoMaskInference.generate uses cond_scale (not guidance_param) and
        # reads CHECKPOINTS_DIR at module import time, so we override the env
        # to point at the momask/ subdir (populated by bootstrap snapshot
        # download from AnimoFlow/animoflow-checkpoints `momask/t2m/**`).
        "generate_kwargs": {
            "guidance_param_default": 5.0,
            "guidance_param_name": "cond_scale",
        },
        "env_overrides_factory": lambda: {
            "CHECKPOINTS_DIR": os.path.join(
                os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints"), "momask"
            ),
        },
    },
    "priormdm": {
        "container_dir": _COMFYUI_ANIMOFLOW / "containers" / "priormdm",
        "module_name": "inference",
        "class_name": "PriorMDMInference",
        # PriorMDMInference.generate(prompt, num_frames, seed, guidance_param,
        # trajectory=, curve_2d=, accel_frac=, decel_frac=) — the extra
        # kwargs are forwarded by pipeline_hf.run() when model=="priormdm".
        # Upstream default cfg=2.5 (≠ MDM's 7.5). (The container also accepts
        # `sample_id` for legacy HumanML3D-dataset references; retired from
        # the public API on 2026-06-24 and no longer forwarded by the wrapper.)
        "generate_kwargs": {
            "guidance_param_default": 2.5,
            "guidance_param_name": "guidance_param",
        },
        # inference.py reads PRIORMDM_PATH, WEIGHTS_DIR, CHECKPOINT_DIR,
        # HML_DATASET_DIR at module-import time. We plant the first three;
        # HML_DATASET_DIR stays unset — sample_id path raises a clean 503
        # at request time, not at load time. PRIORMDM_PATH defaults to the
        # bootstrap clone target under EXTERNAL_DIR.
        "env_overrides_factory": lambda: {
            "PRIORMDM_PATH": os.environ.get(
                "PRIORMDM_PATH",
                str(_COMFYUI_ANIMOFLOW.parent / "priormdm-codes"),
            ),
            "WEIGHTS_DIR": os.path.join(
                os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints"),
                "priormdm",
            ),
            "CHECKPOINT_DIR": os.path.join(
                os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints"),
                "priormdm",
            ),
        },
    },
    # Kimodo lives in escape_hatch/ because of its NVIDIA stack.
}

# Singletons keyed by model name; populated lazily on first .generate().
# Once loaded, the model stays warm in process memory.
_INSTANCES: dict[str, Any] = {}
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED: dict[str, str] = {}  # model → error message


@contextmanager
def _isolated_sys_path(extra: Path):
    """Temporarily prepend `extra` to sys.path. Restores on exit so we don't
    leak a sibling-module on the path that could shadow other models'
    imports."""
    extra_str = str(extra)
    sys.path.insert(0, extra_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(extra_str)
        except ValueError:
            pass


# Top-level package names that BOTH MDM and MoMask vendor at the root of their
# respective repos (e.g. `utils/`, `model/`). If MoMask is imported first, its
# `utils` package gets cached in sys.modules — and a subsequent MDM import
# then sees MoMask's `utils` (no `config` submodule) and silently fails.
# Pre-load cleanup is the simplest fix that doesn't require subprocess
# isolation (and keeps failures loud — no silent fallback).
_SHADOWING_NAMESPACES = (
    "utils",
    "model",
    "diffusion",
    "data_loaders",
    "data_loader",
    "options",
    "common",
    "networks",
)


def _evict_shadowing_modules() -> None:
    """Pop modules whose names collide across our wrapped upstream repos.

    Called BEFORE each model load so the next model's top-level imports
    resolve against its own MDM_PATH / MOMASK_PATH directory, not against
    whatever the previous model left in sys.modules.
    """
    evicted: list[str] = []
    for ns in _SHADOWING_NAMESPACES:
        # Pop the namespace itself and any cached submodules under it.
        for key in [k for k in list(sys.modules) if k == ns or k.startswith(ns + ".")]:
            sys.modules.pop(key, None)
            evicted.append(key)
    if evicted:
        log.info("evicted %d shadowing modules from sys.modules: %s",
                 len(evicted), ", ".join(sorted(evicted)[:8]) + ("…" if len(evicted) > 8 else ""))


def _load(name: str) -> Any:
    """Load an upstream wrapper module and instantiate its inference class.

    Lazy + thread-safe + memoized. Returns the singleton or raises
    RuntimeError with a hopefully-actionable message.
    """
    with _LOAD_LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        if name in _LOAD_FAILED:
            raise RuntimeError(
                f"Model '{name}' previously failed to load: {_LOAD_FAILED[name]}"
            )
        if name not in _DEFINITIONS:
            raise RuntimeError(
                f"Unknown model '{name}'. Known: {sorted(_DEFINITIONS.keys())}. "
                "Outlier models (Kimodo, etc.) are handled by escape_hatch/, not this registry."
            )

        spec = _DEFINITIONS[name]
        container_dir: Path = spec["container_dir"]
        if not container_dir.is_dir():
            msg = (
                f"Container dir missing for '{name}': {container_dir}. "
                "Was the comfyui-animoflow clone created at Docker build time?"
            )
            _LOAD_FAILED[name] = msg
            raise RuntimeError(msg)

        # Determine device. Module-level loading goes on CPU (or ZeroGPU
        # emulation); the @spaces.GPU fork will materialize tensors to real
        # GPU at call time.
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass  # torch not installed in test env — fine, model load will fail later

        log.info("Loading model %r from %s (device=%s)", name, container_dir, device)
        # Install numpy compat shim BEFORE any model load. MDM (af061ca) uses
        # `np.float` which was removed in numpy 1.24; momask-codes uses
        # `np.core.umath_tests.inner1d` removed in numpy 2.0. The shim lives
        # in pipeline_hf because it's also needed by the retargeter, but it
        # must fire BEFORE inference too — not just at retarget time, when
        # MDM's `_load_model` has already crashed. Idempotent.
        from pipeline_hf import _install_numpy_compat_shim
        _install_numpy_compat_shim()
        # Evict any shadowing top-level packages cached from a previously-loaded
        # model's import (e.g. MoMask's `utils` shadowing MDM's `utils`). This
        # is what fixes the "[MDM] model load failed (No module named
        # 'utils.config')" placeholder regression diagnosed 2026-06-08.
        _evict_shadowing_modules()
        # Apply per-model env overrides (e.g. MoMask needs CHECKPOINTS_DIR
        # to point at its own subdir). Reads happen at module import time,
        # so set BEFORE importlib.import_module. Save originals so other
        # models (loaded later) see their own values.
        env_factory = spec.get("env_overrides_factory")
        env_saved: dict[str, str | None] = {}
        if env_factory:
            for k, v in env_factory().items():
                env_saved[k] = os.environ.get(k)
                os.environ[k] = v
                log.info("  env override: %s=%s", k, v)
        try:
            try:
                # Keep container_dir on sys.path PERMANENTLY (in addition to
                # the temporary insertion below). Some wrappers reference
                # container-local helper modules from inside their methods
                # — e.g. priormdm/inference.py:generate() does
                # `from trajectory_utils import bake_trajectory` lazily.
                # If container_dir gets popped post-load, that import 500s
                # at generate-time. This mirrors how MDM/MoMask wrappers
                # do `sys.path.insert(0, MDM_PATH)` at module-import time
                # and never remove it.
                container_dir_str = str(container_dir)
                if container_dir_str not in sys.path:
                    sys.path.append(container_dir_str)
                with _isolated_sys_path(container_dir):
                    # Re-import the wrapper's module fresh in case another
                    # model already imported a same-named module.
                    module_name = spec["module_name"]
                    saved_module = sys.modules.pop(module_name, None)
                    try:
                        mod = importlib.import_module(module_name)
                        klass = getattr(mod, spec["class_name"])
                        inst = klass(device=device)
                    finally:
                        sys.modules.pop(module_name, None)
                        if saved_module is not None:
                            sys.modules[module_name] = saved_module
            finally:
                # Restore env even if loading raised.
                for k, original in env_saved.items():
                    if original is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = original
            _INSTANCES[name] = inst
            log.info("Loaded %r: %s", name, type(inst).__name__)
            return inst
        except Exception as exc:  # noqa: BLE001  (we want to log the full error)
            msg = f"{type(exc).__name__}: {exc}"
            _LOAD_FAILED[name] = msg
            log.exception("Failed to load model %r", name)
            raise RuntimeError(f"Model '{name}' failed to load: {msg}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """Names of registered models (whether loaded or not)."""
    return sorted(_DEFINITIONS.keys())


def model_status() -> dict[str, str]:
    """Per-model status: 'loaded' | 'failed' | 'pending'."""
    out: dict[str, str] = {}
    for name in _DEFINITIONS:
        if name in _INSTANCES:
            out[name] = "loaded"
        elif name in _LOAD_FAILED:
            out[name] = "failed"
        else:
            out[name] = "pending"
    return out


def generate(
    name: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    guidance_param: float | None = None,
    **extra: Any,
) -> tuple[bytes, dict]:
    """Run the model's .generate() and return (npz_bytes, metadata).

    Loads the model on first call (lazy). Subsequent calls reuse the
    in-memory instance.

    Inside @spaces.GPU on HF, this runs on real GPU. Outside HF, runs on
    whatever device torch chose at load time.
    """
    inst = _load(name)
    spec = _DEFINITIONS[name]
    gen_cfg = spec.get("generate_kwargs", {})
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "seed": seed,
    }
    if guidance_param is None:
        guidance_param = gen_cfg.get("guidance_param_default", 7.5)
    # Each model uses its own kwarg name (MDM: guidance_param,
    # MoMask: cond_scale). MoMask raises on unknown kwargs.
    param_name = gen_cfg.get("guidance_param_name", "guidance_param")
    kwargs[param_name] = guidance_param
    kwargs.update(extra)

    npz_bytes, metadata = inst.generate(**kwargs)
    return npz_bytes, metadata
