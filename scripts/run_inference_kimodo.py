#!/usr/bin/env python
"""
Kimodo escape-hatch runner.

Contract (matches escape_hatch/invoke.py:invoke):
  - Stdin:  JSON {prompt, num_frames, seed, guidance_param,
                  num_denoising_steps?, cfg_text?, cfg_constraint?,
                  root2d?: {"frame_indices": [int], "smooth_root_2d": [[x,z]]}}
            num_frames is already in Kimodo's native 30 fps — animoflow-api
            converts duration→frames upstream (api/main.py:623-633).
            root2d is the unified XZ floor-plane constraint that powers both
            "trajectory" (dense frames) and "waypoint" (sparse frames) tasks.
            When omitted, Kimodo runs unconstrained text→motion.
  - Stdout: BVH bytes (22-joint MoMask hierarchy) from the SMPL-free
            rotation-carrying calibrated converter (container fmt="bvh22").
            The rotations carry the pose, so the downstream HF pipeline skips
            IK and consumes the BVH directly (retarget → GLB).
  - Stderr: free-form logs (escape_hatch surfaces last 400 chars on failure).
  - Exit:   0 success, non-zero on any failure.

Per the no-silent-fallback policy: every failure raises. No placeholder.

Runs from inside the Kimodo venv at $KIMODO_VENV_PYTHON. The orchestrator
(animoflow-app) spawns this subprocess from inside @spaces.GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Stdout is reserved for NPZ bytes — everything else goes to stderr.
# CRITICAL: kimodo / transformers / py-soma-x all print() to stdout
# (model load banners, LLM2Vec patch messages, progress bars). Without
# diversion those bytes would prepend to the NPZ payload and `np.load`
# would fail with "Failed to interpret file as a pickle". We dup the
# original stdout fd, redirect Python-level sys.stdout AND fd 1 to
# stderr for the duration of the library calls, then restore the
# saved fd only when we're ready to write the NPZ bytes.
# ---------------------------------------------------------------------------

_STDOUT_BACKUP_FD: int | None = None


def _divert_stdout_to_stderr() -> None:
    """Redirect sys.stdout AND raw fd 1 to stderr so library `print()`s
    don't corrupt our binary stdout payload. Saves the original fd so
    we can restore it just before writing the NPZ."""
    global _STDOUT_BACKUP_FD
    if _STDOUT_BACKUP_FD is not None:
        return  # idempotent
    sys.stdout.flush()
    _STDOUT_BACKUP_FD = os.dup(1)
    os.dup2(2, 1)  # point fd 1 at fd 2 (stderr)
    sys.stdout = sys.stderr


def _restore_stdout_for_npz() -> None:
    """Restore the original stdout fd for the final NPZ write."""
    global _STDOUT_BACKUP_FD
    if _STDOUT_BACKUP_FD is None:
        return
    sys.stderr.flush()
    os.dup2(_STDOUT_BACKUP_FD, 1)
    os.close(_STDOUT_BACKUP_FD)
    _STDOUT_BACKUP_FD = None
    sys.stdout = os.fdopen(1, "wb", closefd=False)


def _log(msg: str) -> None:
    print(f"[kimodo-runner] {msg}", file=sys.stderr, flush=True)


def _fail(msg: str, code: int = 1) -> "None":
    _log(f"FAIL: {msg}")
    sys.exit(code)


# ---------------------------------------------------------------------------
# Sentinel checks — defence-in-depth (escape_hatch.invoke gates first, but the
# runner must also refuse to start if the venv build hasn't finished or failed).
# ---------------------------------------------------------------------------

_EXTERNAL_DIR = Path(
    os.environ.get(
        "ANIMOFLOW_EXTERNAL_DIR",
        "/home/user/app/external"
        if os.path.isdir("/home/user/app")
        else str(Path(__file__).resolve().parent.parent / "external"),
    )
)
_READY_SENTINEL = _EXTERNAL_DIR / ".kimodo_ready"
_FAILED_SENTINEL = _EXTERNAL_DIR / ".kimodo_failed"


def _check_sentinels() -> None:
    if _FAILED_SENTINEL.is_file():
        body = ""
        try:
            body = _FAILED_SENTINEL.read_text()[:1200]
        except OSError:
            pass
        _fail(
            f"Kimodo venv build previously failed:\n{body}\n"
            f"Delete {_FAILED_SENTINEL} and restart the Space to retry."
        )
    if not _READY_SENTINEL.is_file():
        _fail(
            f"Kimodo venv not ready (sentinel missing: {_READY_SENTINEL}). "
            "The bootstrap thread may still be running."
        )


# ---------------------------------------------------------------------------
# Import the AnimoFlow-owned helpers from comfyui-animoflow/containers/kimodo/app.py.
#
# The container's app.py constructs a FastAPI app at module top, but nothing
# serves it — importing it has no runtime cost beyond the import statements.
# We reuse its _patch_llm2vec_for_ungated_llama, _load_model, and
# _motion_to_output helpers verbatim per [[Wrap, don't fork upstream
# model repos]] (this file is AnimoFlow-authored, not upstream NVIDIA).
# ---------------------------------------------------------------------------


def _load_kimodo_app_module():
    comfy_root = os.environ.get("COMFYUI_ANIMOFLOW_DIR")
    if not comfy_root:
        _fail(
            "COMFYUI_ANIMOFLOW_DIR not set — the runner can't locate the "
            "containers/kimodo/app.py helper module."
        )
    container_dir = Path(comfy_root) / "containers" / "kimodo"
    app_py = container_dir / "app.py"
    if not app_py.is_file():
        _fail(f"containers/kimodo/app.py not found at {app_py}")

    # Make `soma_smpl22_bvh` (a sibling module of app.py) importable.
    if str(container_dir) not in sys.path:
        sys.path.insert(0, str(container_dir))
    # Make the Kimodo source tree importable.
    kimodo_src = os.environ.get("KIMODO_SRC_DIR")
    if kimodo_src and kimodo_src not in sys.path:
        sys.path.insert(0, kimodo_src)

    spec = importlib.util.spec_from_file_location("kimodo_helpers", app_py)
    if spec is None or spec.loader is None:
        _fail(f"importlib failed to build spec for {app_py}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Memoize the loaded model across calls in the same subprocess. In practice
# this is one call per @spaces.GPU subprocess, but cheap insurance.
_KIMODO_MODEL = None
_KIMODO_HELPERS = None


def _ensure_model_loaded():
    global _KIMODO_MODEL, _KIMODO_HELPERS
    if _KIMODO_MODEL is not None:
        return _KIMODO_MODEL, _KIMODO_HELPERS

    _KIMODO_HELPERS = _load_kimodo_app_module()
    _log(f"loaded helpers from containers/kimodo/app.py (device={_KIMODO_HELPERS.DEVICE})")

    # _load_model populates the module-global _model AND patches LLM2Vec configs
    # to point at the ungated LLaMA mirror. Idempotent.
    _KIMODO_HELPERS._load_model()
    _KIMODO_MODEL = _KIMODO_HELPERS._model
    if _KIMODO_MODEL is None:
        _fail("_load_model() returned but kimodo_helpers._model is still None")
    _log(f"model ready: {type(_KIMODO_MODEL).__name__}")
    return _KIMODO_MODEL, _KIMODO_HELPERS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _check_sentinels()

    # Read stdin payload BEFORE diverting stdout (we don't write to stdout
    # while reading).
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not parse stdin JSON: {exc}")
        return 2  # unreachable — _fail exits, but keeps type-checkers happy

    # From here on, every library print() must go to stderr until we're
    # ready to dump the NPZ bytes. Kimodo / transformers / py-soma-x all
    # print to stdout (model load banners, "[Kimodo] patching adapter_config"
    # etc.) — without this diversion the NPZ payload is prepended with text
    # and `np.load(BytesIO(stdout_bytes))` fails with "Failed to interpret
    # file as a pickle".
    _divert_stdout_to_stderr()

    prompt = payload.get("prompt")
    num_frames = payload.get("num_frames")
    seed = payload.get("seed")
    if not isinstance(prompt, str) or not prompt.strip():
        _fail("payload.prompt missing or empty")
    if not isinstance(num_frames, int) or num_frames <= 0:
        _fail(f"payload.num_frames missing or non-positive: {num_frames!r}")
    if not isinstance(seed, int):
        _fail(f"payload.seed missing or not an int: {seed!r}")

    num_denoising_steps = int(payload.get("num_denoising_steps", 100))
    # Kimodo's cfg_weight is [cfg_text, cfg_constraint]. The animoflow-api UI
    # exposes a single `cfg` knob which pipeline_hf maps to both. The runner
    # accepts either explicit split values or the single guidance_param.
    if "cfg_text" in payload or "cfg_constraint" in payload:
        cfg_text = float(payload.get("cfg_text", 2.0))
        cfg_constraint = float(payload.get("cfg_constraint", 2.0))
    else:
        gp = payload.get("guidance_param")
        cfg = float(gp) if gp is not None else 2.0
        cfg_text = cfg
        cfg_constraint = cfg

    # Unified root2d constraint covers both trajectory (dense) and waypoint
    # (sparse) tasks. pipeline_hf builds the dict; we forward it to Kimodo's
    # constraint loader. Fail-fast on malformed shapes — no silent fallback.
    root2d = payload.get("root2d")
    if root2d is not None:
        if not isinstance(root2d, dict):
            _fail(f"payload.root2d must be a dict, got {type(root2d).__name__}")
        frame_indices = root2d.get("frame_indices")
        smooth_root_2d = root2d.get("smooth_root_2d")
        if not isinstance(frame_indices, list) or not frame_indices:
            _fail("payload.root2d.frame_indices must be a non-empty list of ints")
        if not isinstance(smooth_root_2d, list) or not smooth_root_2d:
            _fail("payload.root2d.smooth_root_2d must be a non-empty list of [x,z] pairs")
        if len(frame_indices) != len(smooth_root_2d):
            _fail(
                f"payload.root2d length mismatch: frame_indices={len(frame_indices)} "
                f"vs smooth_root_2d={len(smooth_root_2d)}"
            )
        for i, p in enumerate(smooth_root_2d):
            if not (isinstance(p, list) and len(p) == 2):
                _fail(f"payload.root2d.smooth_root_2d[{i}] must be [x, z]; got {p!r}")
        max_frame = max(frame_indices)
        if max_frame >= num_frames or min(frame_indices) < 0:
            _fail(
                f"payload.root2d.frame_indices out of range [0, {num_frames}): "
                f"min={min(frame_indices)} max={max_frame}"
            )

    model, helpers = _ensure_model_loaded()

    # Match the container's _generate_worker call path (containers/kimodo/app.py:472-481).
    import torch
    from kimodo.tools import seed_everything

    seed_everything(int(seed))

    # Build the constraint_lst kwarg. load_constraints_lst takes either a JSON
    # path or a list[dict] directly; we pass a list[dict] to avoid file I/O.
    # See kimodo/constraints.py:566-593 (TYPE_TO_CLASS["root2d"] → Root2DConstraintSet).
    constraint_lst: list = []
    if root2d is not None:
        from kimodo.constraints import load_constraints_lst
        dev = next(model.parameters()).device if hasattr(model, "parameters") else None
        constraint_lst = load_constraints_lst(
            [{"type": "root2d",
              "frame_indices": [int(f) for f in root2d["frame_indices"]],
              "smooth_root_2d": [[float(x), float(z)] for (x, z) in root2d["smooth_root_2d"]]}],
            model.skeleton,
            device=dev,
            dtype=torch.float32,
        )
        _log(
            f"root2d constraint: {len(root2d['frame_indices'])} frames "
            f"(min={min(root2d['frame_indices'])} max={max(root2d['frame_indices'])}) "
            f"XZ range x=[{min(p[0] for p in root2d['smooth_root_2d']):.2f},"
            f"{max(p[0] for p in root2d['smooth_root_2d']):.2f}] "
            f"z=[{min(p[1] for p in root2d['smooth_root_2d']):.2f},"
            f"{max(p[1] for p in root2d['smooth_root_2d']):.2f}]"
        )

    _log(
        f"generating: prompt={prompt[:60]!r} num_frames={num_frames} "
        f"seed={seed} steps={num_denoising_steps} cfg=[{cfg_text},{cfg_constraint}] "
        f"constraints={len(constraint_lst)}"
    )
    with torch.no_grad():
        output = model(
            prompts=prompt,
            num_frames=num_frames,
            num_denoising_steps=num_denoising_steps,
            cfg_weight=[cfg_text, cfg_constraint],
            constraint_lst=constraint_lst,
            return_numpy=False,
            post_processing=False,
        )

    # SOMA → 22-joint rig-ready BVH (SMPL-free rotation-carrying calibrated path).
    # The rotations carry the pose, so this feeds the rig stage directly — the
    # orchestrator's plan skips IK for Kimodo.
    data, is_npz = helpers._motion_to_output(output, model.output_skeleton, fmt="bvh22")
    if is_npz:
        _fail("_motion_to_output returned is_npz=True for fmt='bvh22' — expected a BVH string")
    if not isinstance(data, str) or not data:
        _fail(f"_motion_to_output returned empty/non-str BVH payload: {type(data)}")

    # Defensive: sanity-check the BVH header before handing it to the rig stage.
    bvh_bytes = data.encode()
    nframes = next((int(l.split(":", 1)[1]) for l in data.splitlines()
                    if l.startswith("Frames:")), None)
    if "HIERARCHY" not in data or nframes is None:
        _fail("bvh22 output missing HIERARCHY / Frames header")
    _log(f"output: BVH {nframes} frames ({len(bvh_bytes)//1024} KB)")

    # Restore the saved stdout fd and write the BVH payload as raw bytes.
    _restore_stdout_for_npz()
    os.write(1, bvh_bytes)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # Top-level guard so any uncaught error lands as a clear stderr trace
        # rather than a silent non-zero exit. No silent fallback.
        print(f"[kimodo-runner] FATAL UNCAUGHT: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
