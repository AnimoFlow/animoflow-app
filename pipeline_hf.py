"""
HF pipeline — replaces animoflow-api's ComfyUI-orchestrated `pipeline.run`.

Same call signature as animoflow-api/api/pipeline.py:run, so the FastAPI
in animoflow-api/api/main.py can call this without modification.

Compute layout:

  Receive request          (CPU, orchestrator)
  ──► run_inference(...)   (GPU, @spaces.GPU(duration=30))
       └── npz bytes
  ──► resample             (CPU, shared stage library)
  ──► IK npz → bvh         (CPU — momask Joint2BVHConvertor)
  ──► Blender bvh → fbx    (CPU subprocess)
  ──► Blender fbx → glb    (CPU subprocess, shared stage library script)
  ──► return filename

Only run_inference is GPU-billed. Everything else stays in the persistent
orchestrator process. ZeroGPU daily quota burns at ~10-15s per request,
not 60-120s.

PIPELINE SHAPE AND STAGE SEMANTICS ARE SHARED, NOT DUPLICATED: this
executor walks the plan from comfyui-animoflow/animoflow_stages/plan.py
— the exact plan the API server compiles into the local ComfyUI node
DAG. Per-model parameter mapping (genparams), the resample stage, and
the FBX→GLB Blender script (snap-to-ground, trajectory restore, Y_bot
brand tint, matte-character flatten, keyframe RDP) all come from the stage
library, so the hosted pipeline and the local graph cannot drift.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Make sure the orchestrator's own modules are importable when this module
# is loaded by uvicorn from a different cwd.
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from spaces_compat import GPU  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dir — match animoflow-api's OUTPUT_DIR convention
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/animoflow-output"))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# comfyui-animoflow checkout (cloned at Docker build time on the Space).
_COMFY_ROOT = Path(
    os.environ.get("COMFYUI_ANIMOFLOW_DIR", "/opt/comfyui-animoflow")
)

# ---------------------------------------------------------------------------
# Lazy imports — defer heavy modules so app.py boots cheaply for tests
# ---------------------------------------------------------------------------

_retargeter = None
_post_funcs = None
_stage_lib = None


def _stages():
    """Load the shared stage library from the comfyui-animoflow checkout
    (cached). Path-based import — the repo isn't pip-installed."""
    global _stage_lib
    if _stage_lib is None:
        pkg_init = _COMFY_ROOT / "animoflow_stages" / "__init__.py"
        if not pkg_init.is_file():
            raise RuntimeError(
                f"animoflow_stages not found at {pkg_init}. Was "
                "comfyui-animoflow cloned at Docker build time (and is the "
                "clone new enough to contain the stage library)?"
            )
        # Purge any stale entries first — `from animoflow_stages import x`
        # binds cached submodules WITHOUT attaching them to a freshly
        # inserted package object, leaving `stages.plan` unset.
        for k in [k for k in sys.modules
                  if k == "animoflow_stages" or k.startswith("animoflow_stages.")]:
            del sys.modules[k]
        spec = importlib.util.spec_from_file_location(
            "animoflow_stages", str(pkg_init),
            submodule_search_locations=[str(pkg_init.parent)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["animoflow_stages"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        from animoflow_stages import (  # noqa: F401 — register submodules
            fps, genparams, glb_export, glb_post, plan, resample,
        )
        _stage_lib = sys.modules["animoflow_stages"]
    return _stage_lib


def _install_numpy_compat_shim() -> None:
    """Make momask-codes' legacy numpy 1.x calls work under numpy 2.x.

    momask-codes was written against numpy < 1.24 and uses two APIs that
    have since been removed:

      * `np.float` / `np.int` / `np.bool` / `np.object` — removed in numpy 1.24.
        Used in `momask-codes/common/quaternion.py` and similar.
      * `numpy.core.umath_tests` — internal module removed in numpy 2.0.
        Used by `momask-codes/visualization/Animation.py` for its `inner1d`
        helper. We replace it with a tiny shim that uses `np.einsum`.

    Per the wrap-don't-modify rule we do not patch momask-codes itself.
    Instead we install these shims in our orchestrator process BEFORE the
    first momask import. Idempotent + no-op on numpy < 2.0.
    """
    import warnings as _warnings  # noqa: I001

    import numpy as _np

    # 1. np.float / np.int / np.bool / np.object aliases.
    # `hasattr` triggers a numpy FutureWarning on some legacy attrs even
    # though the attribute itself doesn't exist as a real symbol — silence
    # those during the probe so they don't pollute orchestrator startup.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", FutureWarning)
        _warnings.simplefilter("ignore", DeprecationWarning)
        for _legacy, _replacement in (
            ("float", _np.float64),
            ("int", _np.int64),
            ("bool", _np.bool_),
            ("object", _np.object_),
            ("complex", _np.complex128),
            ("str", _np.str_),
            ("long", _np.int64),
        ):
            if not hasattr(_np, _legacy):
                setattr(_np, _legacy, _replacement)

    # 2. numpy.core.umath_tests stub. The original module exposed several
    # ufunc helpers that momask uses; we provide drop-ins for all known
    # callers and surface a clear error for anything else so we discover
    # gaps via a useful exception, not a cryptic AttributeError.
    if "numpy.core.umath_tests" not in sys.modules:
        import types as _types

        _stub = _types.ModuleType("numpy.core.umath_tests")

        def _inner1d(a, b):
            """np.core.umath_tests.inner1d → einsum element-wise dot."""
            return _np.einsum("...i,...i->...", a, b)

        def _matrix_multiply(a, b):
            """np.core.umath_tests.matrix_multiply → np.matmul (handles
            stacked matrices the same way the old C ufunc did)."""
            return _np.matmul(a, b)

        def _innerwt(a, b, w):
            """np.core.umath_tests.innerwt → weighted inner product."""
            return _np.einsum("...i,...i,...i->...", a, b, w)

        _stub.inner1d = _inner1d
        _stub.matrix_multiply = _matrix_multiply
        _stub.innerwt = _innerwt
        sys.modules["numpy.core.umath_tests"] = _stub
        # Also attach to numpy.core so `from numpy.core import umath_tests`
        # style imports resolve.
        try:
            import numpy.core as _np_core

            if not hasattr(_np_core, "umath_tests"):
                _np_core.umath_tests = _stub
        except ImportError:
            pass


def _get_retargeter():
    """Lazy-load the comfyui-animoflow MotionRetargeter singleton.

    Loads the upstream momask Joint2BVHConvertor under the hood. Pure-Python
    (numpy / matplotlib / scipy / torch) — no GPU required for IK or BVH.
    """
    global _retargeter
    if _retargeter is None:
        # Install numpy-compat shim BEFORE importing anything from momask.
        _install_numpy_compat_shim()

        nodes_dir = Path(
            os.environ.get(
                "COMFYUI_ANIMOFLOW_NODES_DIR",
                str(_COMFY_ROOT / "nodes"),
            )
        )
        if not nodes_dir.is_dir():
            raise RuntimeError(
                f"comfyui-animoflow/nodes/ not found at {nodes_dir}. "
                "Was comfyui-animoflow cloned at Docker build time?"
            )
        # The package name has a hyphen so a plain `import` won't work —
        # same dynamic loading pattern as the AnimoFlow IK/Rig nodes.
        pkg_init = nodes_dir / "retargeter" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "animoflow_retargeter", str(pkg_init)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["animoflow_retargeter"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _retargeter = mod.MotionRetargeter()
        log.info("Loaded MotionRetargeter from %s", pkg_init)
    return _retargeter


def _get_post_funcs():
    """Lazy-load CPU post-processing helpers from comfyui-animoflow nodes.

    No post-process steps ship today — outlier_fix + foot_skating_fix were
    retired 2026-06-24 (see tests/outlier-post-proc/). The registry is kept
    as the integration point for any future BVH→BVH transformer added to
    comfyui-animoflow/motion_utils/; new entries get wired here and called
    from the ik stage of _run_tail.
    """
    global _post_funcs
    if _post_funcs is None:
        _post_funcs = {}
    return _post_funcs


# ---------------------------------------------------------------------------
# GPU-decorated inference — the only call that allocates ZeroGPU
# ---------------------------------------------------------------------------

# ZeroGPU forks a worker for each @spaces.GPU call and serializes the return
# value back across a pipe. Empirically, exception args are stripped to the
# class name (sometimes via repr — producing a literal `'RuntimeError'`
# symptom on the Space). To avoid relying on
# that serialization, every GPU function catches its own exceptions, JSON-
# encodes them with a sentinel prefix, and returns bytes. The CPU-side caller
# detects the sentinel and re-raises a real RuntimeError with the original
# message + traceback preserved.
_ERR_SENTINEL = b"\x00\x00ANIMOFLOW_ERR\x00\x00"


def _do_inference(
    model: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    guidance_param: float | None = None,
    **extra: Any,
) -> bytes:
    """Body of inference. Identical CPU/GPU behaviour for both @GPU wrappers.

    For models in the orchestrator venv (MDM family), the fork inherits the
    warm singleton via copy-on-write. For outliers (Kimodo, future), the
    function delegates to escape_hatch.invoke() which subprocess-calls into
    the model's own venv — that subprocess inherits the GPU from the fork.
    """
    from escape_hatch import is_outlier, invoke as escape_invoke

    if is_outlier(model):
        # Subprocess timeout tracks the active admission window minus a
        # margin, so OUR loud, classified timeout fires before ZeroGPU
        # kills the slot from outside (which would strip the error).
        _window = _kimodo_window_s() if model == "kimodo" else None
        return escape_invoke(
            model,
            prompt=prompt,
            num_frames=num_frames,
            seed=seed,
            guidance_param=guidance_param,
            **({"timeout": max(30, _window - 5)} if _window else {}),
            **extra,
        )

    from animoflow_models import generate as model_generate

    npz_bytes, _meta = model_generate(
        model,
        prompt=prompt,
        num_frames=num_frames,
        seed=seed,
        guidance_param=guidance_param,
        **extra,
    )
    return npz_bytes


def _safe_do_inference(
    model: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    guidance_param: float | None = None,
    **extra: Any,
) -> bytes:
    """_do_inference wrapped to survive ZeroGPU's cross-worker exception strip.

    Returns either the raw NPZ bytes, OR `_ERR_SENTINEL + json(...)` carrying
    the error class + message + truncated traceback. Callers MUST check the
    sentinel and re-raise.
    """
    try:
        return _do_inference(
            model, prompt, num_frames, seed,
            guidance_param=guidance_param, **extra,
        )
    except BaseException as e:
        payload = json.dumps({
            "error_class": type(e).__name__,
            "error_module": type(e).__module__,
            "message": str(e) or repr(e),
            "traceback": traceback.format_exc()[-4000:],
        }).encode()
        return _ERR_SENTINEL + payload


def _unwrap_gpu_result(npz_bytes: bytes) -> bytes:
    """Inverse of _safe_do_inference: raise if the sentinel is present,
    otherwise pass the NPZ bytes through.

    Reraises as a RuntimeError carrying the original message — preserved
    across the ZeroGPU process boundary via JSON instead of pickle.
    """
    if not npz_bytes.startswith(_ERR_SENTINEL):
        return npz_bytes
    try:
        err = json.loads(npz_bytes[len(_ERR_SENTINEL):].decode())
    except Exception:  # noqa: BLE001
        raise RuntimeError("GPU worker failed (error payload unparseable)")
    msg = err.get("message") or f"{err.get('error_class', 'Unknown')}: <no message>"
    tb = err.get("traceback") or ""
    if tb:
        log.error("GPU worker raised %s:\n%s", err.get("error_class"), tb)
    raise RuntimeError(msg)


# GPU-decorated wrappers around the same body. MDM/MoMask take the cheap
# 30-second slot. Kimodo's admission window is DYNAMIC (spaces supports
# duration=callable, resolved per call in this process): 300 s cold —
# the first call after boot pays LLaMA-encoder-sidecar startup (~150 s
# measured 2026-07-07) — then ~70 s warm (holds 25-42 s measured; wire =
# decorator × duration_factor 1.5 → 105 s, fitting even a fresh anonymous
# ZeroGPU bucket of 120 s). Behind KIMODO_WINDOW=dynamic; the default
# (static) keeps the flat 300 s. Dispatch happens at the orchestrator
# (CPU) layer in run() — MDM/MoMask quota burn is unaffected.

_KIMODO_COLD_WINDOW_S = 300
_kimodo_warmed = False  # set after the first successful inference; reset on
                        # a warm-window timeout so the next call self-heals
                        # with the cold window (loud, never silent)


def _kimodo_window_s() -> int:
    if os.environ.get("KIMODO_WINDOW", "static").strip().lower() != "dynamic":
        return _KIMODO_COLD_WINDOW_S
    if not _kimodo_warmed:
        return _KIMODO_COLD_WINDOW_S
    try:
        return int(os.environ.get("KIMODO_WARM_WINDOW_S", "70") or 70)
    except ValueError:
        return _KIMODO_COLD_WINDOW_S


def _kimodo_prewarm_loop() -> None:
    """Boot pre-warm (KIMODO_PREWARM=1): one token-less Kimodo inference as
    soon as the venv sentinel appears, so the first USER call gets the warm
    window instead of paying the ~150 s sidecar startup. Runs outside any
    gradio request → schedules token-less on the Space's shared pool by
    construction. Failure is LOUD and leaves the cold window active — the
    system stays correct, just slower (no silent fallback)."""
    import threading  # noqa: F401 — imported here to keep module deps flat

    try:
        from bootstrap import _kimodo_ready_sentinel
    except Exception:
        log.exception("[kimodo-prewarm] cannot resolve bootstrap sentinel — "
                      "pre-warm disabled, cold window stays active")
        return
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        if _kimodo_ready_sentinel().is_file():
            break
        time.sleep(10)
    else:
        log.error("[kimodo-prewarm] venv sentinel never appeared in 30 min — "
                  "pre-warm skipped, cold window stays active")
        return
    log.info("[kimodo-prewarm] venv ready — running the cold call now "
             "(shared pool, ~150 s)")
    try:
        raw = _run_inference_gpu_kimodo(
            "kimodo", "a person walks forward", 100, 0)
        _unwrap_gpu_result(raw)
        global _kimodo_warmed
        _kimodo_warmed = True
        log.info("[kimodo-prewarm] done — warm window (%ss) active",
                 _kimodo_window_s())
    except Exception:
        log.exception("[kimodo-prewarm] FAILED — cold window stays active")


def start_kimodo_prewarm_if_enabled() -> None:
    """Called at module import (Space boot). No-op unless KIMODO_PREWARM=1."""
    if os.environ.get("KIMODO_PREWARM", "").strip() != "1":
        return
    import threading
    threading.Thread(target=_kimodo_prewarm_loop, daemon=True,
                     name="kimodo-prewarm").start()
    log.info("[kimodo-prewarm] armed (waiting for venv sentinel)")


@GPU(duration=30)
def _run_inference_gpu_fast(
    model: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    guidance_param: float | None = None,
    **extra: Any,
) -> bytes:
    return _safe_do_inference(
        model, prompt, num_frames, seed,
        guidance_param=guidance_param, **extra,
    )


@GPU(duration=lambda *a, **k: _kimodo_window_s())
def _run_inference_gpu_kimodo(
    model: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    guidance_param: float | None = None,
    **extra: Any,
) -> bytes:
    return _safe_do_inference(
        model, prompt, num_frames, seed,
        guidance_param=guidance_param, **extra,
    )


# Timeline GPU wrapper — priorMDM double-take. Pass 1 (N segments × ~50 steps)
# + Pass 2 (handshake re-diffuse, ~5 steps) is ~30-50 s warm on H200 for a
# 4-segment ~6 s clip. 60 s budget is comfortable; runs on a separate
# decorator from the 30 s _run_inference_gpu_fast slot because ZeroGPU
# duration is a static decorator arg.
@GPU(duration=60)
def _run_timeline_gpu(
    segments: list[dict],
    seed: int,
    *,
    guidance_param: float = 2.5,
    handshake_size: int = 10,
    blend_len: int = 10,
) -> bytes:
    """ZeroGPU-wrapped priorMDM.generate_timeline.

    Same sentinel-prefix pattern as `_safe_do_inference` — ZeroGPU strips
    exception args across the worker boundary, so we round-trip errors as
    JSON and let `_unwrap_gpu_result` re-raise them on the orchestrator side.
    """
    try:
        from animoflow_models.registry import _load

        inst = _load("priormdm")
        npz_bytes, _meta = inst.generate_timeline(
            segments=segments,
            seed=seed,
            guidance_param=guidance_param,
            handshake_size=handshake_size,
            blend_len=blend_len,
        )
        return npz_bytes
    except BaseException as e:  # noqa: BLE001 — round-trip across the worker
        payload = json.dumps({
            "error_class": type(e).__name__,
            "error_module": type(e).__module__,
            "message": str(e) or repr(e),
            "traceback": traceback.format_exc()[-4000:],
        }).encode()
        return _ERR_SENTINEL + payload


# ---------------------------------------------------------------------------
# Plan executor — walks the SAME plan the API server compiles for ComfyUI
# ---------------------------------------------------------------------------


def _make_stage_cb(on_progress, on_stage):
    def _stage(label: str, progress: float) -> None:
        if on_stage:
            try:
                on_stage(label)
            except Exception:  # noqa: BLE001
                log.exception("on_stage callback raised")
        if on_progress:
            try:
                on_progress(progress)
            except Exception:  # noqa: BLE001
                log.exception("on_progress callback raised")
    return _stage


def _run_tail(
    *,
    job_id: str,
    npz_bytes: bytes,
    plan_specs: list,
    output_fps: int,
    compress_output: bool,
    _stage: Callable[[str, float], None],
    _timings: dict[str, float],
    t0: float,
) -> str:
    """Execute the post-generate stages of a plan (resample → IK → rig →
    glb_export) against the HF-side implementations. Returns the FBX
    filename written to OUTPUT_DIR."""
    stages = _stages()
    fbx_path: Path | None = None
    bvh_bytes: bytes | None = None

    # Kimodo emits a rig-ready BVH (SMPL-free rotation path), so its plan has no
    # `ik` stage — the generated bytes ARE the BVH. Seed bvh_bytes directly so the
    # rig stage consumes it; resample/ik simply don't appear in the plan.
    if not any(spec.kind == "ik" for spec in plan_specs):
        bvh_bytes = npz_bytes

    for spec in plan_specs:
        if spec.kind in ("generate", "generate_timeline"):
            continue  # handled by the caller (GPU dispatch differs)

        if spec.kind == "resample":
            # Always run — even at equal rates it stamps the
            # authoritative fps key downstream stages rely on.
            _t = time.perf_counter()
            in_fps = spec.params["input_fps"]
            npz_bytes = stages.resample.resample_npz(
                npz_bytes, in_fps, spec.params["output_fps"])
            _timings["resample"] = time.perf_counter() - _t
            log.info(
                "[%s] resampled %d→%dfps (%d KB NPZ)",
                job_id, in_fps, spec.params["output_fps"],
                len(npz_bytes) // 1024,
            )

        elif spec.kind == "ik":
            _stage("Inverse Kinematics…", 0.55)
            _t = time.perf_counter()
            retargeter = _get_retargeter()
            # Pass the authoritative post-resample rate — the BVH Frame
            # Time drives retarget_keemap's scene FPS and thus the final
            # GLB timing.
            bvh_bytes, ik_meta = retargeter.npz_to_bvh(npz_bytes, fps=output_fps)
            _timings["ik"] = time.perf_counter() - _t
            log.info(
                "[%s] IK done: %d frames, %d joints",
                job_id, ik_meta["num_frames"], ik_meta["num_joints"],
            )

            # Post-process slot intentionally empty — _get_post_funcs().

            # Optional debug dump (off by default): persist per-stage
            # intermediates. When on, /v1/files/{job_id}.{npz,bvh} become
            # fetchable alongside .fbx (each artifact needs a single-dot
            # extension — /v1/files splits on the last dot).
            if os.environ.get("ANIMOFLOW_DEBUG_DUMP", "0").lower() in ("1", "true", "yes"):
                try:
                    (_OUTPUT_DIR / f"{job_id}.npz").write_bytes(npz_bytes)
                    (_OUTPUT_DIR / f"{job_id}.bvh").write_bytes(bvh_bytes)
                    log.info(
                        "[%s] DEBUG_DUMP wrote .npz (%dKB) .bvh (%dKB, raw post-IK)",
                        job_id, len(npz_bytes)//1024, len(bvh_bytes)//1024,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("DEBUG_DUMP failed (non-fatal)")

        elif spec.kind == "rig":
            if bvh_bytes is None:
                raise RuntimeError("rig stage reached with no BVH (missing ik stage?)")
            _stage("Rigging…", 0.80)
            _t = time.perf_counter()
            character = spec.params["character"]
            chars_dir = Path(
                os.environ.get("CHARACTERS_DIR", str(_COMFY_ROOT / "characters"))
            )

            if str(_COMFY_ROOT) not in sys.path:
                sys.path.insert(0, str(_COMFY_ROOT))
            from robot_retarget.characters import (
                is_robot_character,
                retarget_to_robot_fbx,
            )

            if is_robot_character(character):
                fbx_bytes = retarget_to_robot_fbx(bvh_bytes, character, chars_dir)
            else:
                retargeter = _get_retargeter()
                fbx_template = chars_dir / f"{character}.fbx"
                if not fbx_template.exists():
                    raise RuntimeError(
                        f"Character template not found: {fbx_template}. "
                        f"Available: {[p.stem for p in chars_dir.glob('*.fbx')]}"
                    )
                fbx_bytes, _rig_meta = retargeter.bvh_to_fbx(
                    bvh_bytes, fbx_template=str(fbx_template)
                )
            _timings["retarget"] = time.perf_counter() - _t
            log.info("[%s] retarget done: %d KB FBX", job_id, len(fbx_bytes) // 1024)

            fbx_path = _OUTPUT_DIR / f"{job_id}.fbx"
            fbx_path.write_bytes(fbx_bytes)

        elif spec.kind == "glb_export":
            if fbx_path is None:
                raise RuntimeError("glb_export reached with no FBX written")
            _stage("Post processing…", 0.95)
            p = spec.params
            character = p["character"]
            traj = p.get("traj_restore") or {}
            _t = time.perf_counter()
            glb_path: Path | None = fbx_path.with_suffix(".glb")
            snap_info: dict | None = None
            snap_elapsed = 0.0
            try:
                options = stages.glb_export.GLBExportOptions(
                    fbx_path=str(fbx_path),
                    glb_path=str(glb_path),
                    character=character,
                    snap_to_ground=p["snap_to_ground"],
                    characters_dir=os.environ.get(
                        "CHARACTERS_DIR", str(_COMFY_ROOT / "characters")),
                    cache_dir=str(_OUTPUT_DIR),
                    comfy_root=str(_COMFY_ROOT),
                    traj_theta=float(traj.get("theta_rad", 0.0)),
                    traj_tx=float(traj.get("tx", 0.0)),
                    traj_tz=float(traj.get("tz", 0.0)),
                    keyframe_builder=p["keyframe_builder"],
                    keyframe_builder_error_degrees=p["keyframe_builder_error_degrees"],
                )
                result = stages.glb_export.run_glb_export(options, timeout=120)
                snap_info = result["snap_info"]
                snap_elapsed = result["snap_elapsed_s"]
            except Exception:  # noqa: BLE001
                # GLB export is best-effort on the Space — the FBX is
                # already written and downloadable. (The local ComfyUI
                # node raises instead: there the GLB IS the terminal
                # output.)
                log.exception("GLB export failed (FBX still written)")
                glb_path = None
            _timings["glb"] = time.perf_counter() - _t

            # Sidecar JSON the API layer reads and surfaces as
            # JobResponse.snap_info.
            if snap_info is None and not p["snap_to_ground"]:
                snap_info = {"applied": False,
                             "reason": "request snap_to_ground=false"}
            if snap_info is not None:
                try:
                    (_OUTPUT_DIR / f"{job_id}.snap.json").write_text(
                        json.dumps(snap_info))
                except OSError:
                    log.exception("Failed to write snap_info sidecar")
            # Snap wall time runs inside the Blender subprocess; surface
            # it separately and keep the timing sum honest.
            if snap_elapsed > 0.0:
                _timings["snap"] = snap_elapsed
                _timings["glb"] = max(_timings["glb"] - snap_elapsed, 0.0)

            # Embedded-texture downsize BEFORE gltfpack (alignment rules
            # — see animoflow_stages/glb_post.py). No-op on textureless
            # characters. Env-gated, default ON here — the Space's file
            # serving is throttled ~100 KB/s.
            if glb_path is not None and os.environ.get(
                "ENABLE_TEXTURE_DOWNSIZE", "true"
            ).lower() == "true":
                _t = time.perf_counter()
                stages.glb_post.downsize_glb_textures(glb_path)
                _timings["glb_texresize"] = time.perf_counter() - _t

            # gltfpack -cc. Two opt-outs: ENABLE_GLB_COMPRESSION env kill
            # switch, and the per-request compress_output flag (the
            # Blender addon opts out — Blender 5 rejects
            # EXT_meshopt_compression). Failures raise — no silent
            # fallback.
            if (
                glb_path is not None
                and compress_output
                and os.environ.get("ENABLE_GLB_COMPRESSION", "true").lower() == "true"
            ):
                _t = time.perf_counter()
                stages.glb_post.compress_glb(glb_path)
                _timings["glb_compress"] = time.perf_counter() - _t

        else:
            raise ValueError(f"Unknown stage kind {spec.kind!r}")

    if fbx_path is None:
        raise RuntimeError("plan finished without producing an FBX")

    _timings["total"] = time.perf_counter() - t0
    _timing_parts = " ".join(f"{k}={v:.2f}s" for k, v in _timings.items())
    log.info("[STAGE_TIMINGS] job=%s %s", job_id, _timing_parts)
    log.info(
        "[%s] HF pipeline complete in %.1fs total → %s",
        job_id, _timings["total"], fbx_path.name,
    )
    return fbx_path.name


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for animoflow-api/api/pipeline.py
# ---------------------------------------------------------------------------


def run(
    job_id: str,
    prompt: str,
    num_frames: int,
    seed: int,
    character: str = "Y_bot",
    model: str = "mdm",
    on_progress: Callable[[float], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    preprocess_stages: list[dict] | None = None,
    keyframe_builder: bool = False,
    keyframe_builder_error_degrees: float = 3.0,
    snap_to_ground: bool = True,
    output_fps: int = 30,
    cfg: float = 2.5,
    curve_2d: list[list[float]] | None = None,
    traj_restore: dict | None = None,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
    waypoints: list[dict] | None = None,
    compress_output: bool = True,
) -> str:
    """HF replacement for animoflow-api/api/pipeline.py:run.

    Returns the output filename (e.g. 'abc.fbx'). Same semantics: writes
    to OUTPUT_DIR, raises on failure.

    Conditioning args:
      - accel_frac, decel_frac: priorMDM-only velocity-profile shaping.
      - curve_2d: priorMDM (trajectory) or Kimodo (trajectory task — dense XZ
        path; frame_indices are evenly distributed across num_frames).
      - waypoints: Kimodo-only (waypoint task — sparse list of {x, z, t}
        where t is a frame index in [0, num_frames). Wins over curve_2d
        when both are set.
    """
    t0 = time.perf_counter()
    log.info(
        "[%s] HF pipeline run: model=%s prompt=%r num_frames=%d seed=%d character=%s",
        job_id, model, prompt[:60], num_frames, seed, character,
    )

    _timings: dict[str, float] = {}
    _stage = _make_stage_cb(on_progress, on_stage)
    # Fire the first stage label before touching the stage library so
    # callers see progress even when the checkout is missing/broken.
    _stage("Generating…", 0.05)

    stages = _stages()
    plan_specs = stages.plan.build_plan(
        model, character,
        prompt=prompt,
        num_frames=num_frames,
        seed=seed,
        output_fps=output_fps,
        snap_to_ground=snap_to_ground,
        keyframe_builder=keyframe_builder,
        keyframe_builder_error_degrees=keyframe_builder_error_degrees,
        traj_restore=traj_restore,
        compress_output=compress_output,
        curve_2d=curve_2d,
        waypoints=waypoints,
        accel_frac=accel_frac,
        decel_frac=decel_frac,
    )

    # ── Stage 1: GPU inference ───────────────────────────────────────────
    _t = time.perf_counter()
    # Per-model conditioning kwargs come from the SHARED mapping — the
    # same one the ComfyUI compiler uses for node inputs.
    extra = stages.genparams.build_hf_gen_extra(
        model,
        num_frames=num_frames,
        cfg=cfg,
        curve_2d=curve_2d,
        waypoints=waypoints,
        accel_frac=accel_frac,
        decel_frac=decel_frac,
    )

    # Dispatch to the appropriately-budgeted @spaces.GPU wrapper. Kimodo
    # needs ~300 s for cold-start (LLaMA encoder warmup + model load);
    # MDM/MoMask stay on the cheap 30 s slot so their daily ZeroGPU quota
    # is unaffected.
    gpu_fn = (
        _run_inference_gpu_kimodo if model == "kimodo" else _run_inference_gpu_fast
    )
    # Guidance scale from the request (cfg), already clamped to the
    # model's range by the API server. Was hardcoded None (each model's
    # internal default) — passing it explicitly is what makes the /v1
    # `cfg` field actually take effect for simple-mode tasks.
    global _kimodo_warmed
    try:
        npz_bytes = gpu_fn(
            model, prompt, num_frames, seed, guidance_param=cfg, **extra,
        )
        # Decode the sentinel-prefixed error payload if the GPU worker raised.
        npz_bytes = _unwrap_gpu_result(npz_bytes)
    except Exception as exc:
        if model == "kimodo" and _kimodo_warmed and (
                "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                or "aborted" in str(exc).lower()):
            # A warm-windowed run overran or was killed: drop back to the
            # cold window so the NEXT call self-heals. Loud by design.
            _kimodo_warmed = False
            log.error("[kimodo-window] warm-window run failed (%s) — warm "
                      "flag RESET, next call uses the %ss cold window",
                      type(exc).__name__, _KIMODO_COLD_WINDOW_S)
        raise
    if model == "kimodo" and not _kimodo_warmed:
        _kimodo_warmed = True
        log.info("[kimodo-window] first successful inference — warm window "
                 "(%ss) active for subsequent calls", _kimodo_window_s())
    _timings["gpu"] = time.perf_counter() - _t
    log.info(
        "[%s] inference done in %.1fs (%d KB NPZ)",
        job_id, _timings["gpu"], len(npz_bytes) // 1024,
    )

    return _run_tail(
        job_id=job_id,
        npz_bytes=npz_bytes,
        plan_specs=plan_specs,
        output_fps=output_fps,
        compress_output=compress_output,
        _stage=_stage,
        _timings=_timings,
        t0=t0,
    )


def run_timeline(
    job_id: str,
    segments: list[dict],
    seed: int,
    character: str = "Y_bot",
    cfg: float = 2.5,
    handshake_size: int = 10,
    blend_len: int = 10,
    on_progress: Callable[[float], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    keyframe_builder: bool = False,
    keyframe_builder_error_degrees: float = 3.0,
    snap_to_ground: bool = True,
    output_fps: int = 30,
    compress_output: bool = True,
) -> str:
    """HF replacement for animoflow-api/api/pipeline.py:run_timeline.

    Drives priorMDM's double-take long-motion path. Each segment is
    `{"prompt": str, "num_frames": int}` — frame counts are caller-resolved
    against priorMDM's 20 fps native rate (api/main.py converts segment
    duration → num_frames at _PRIORMDM_FPS=20).
    """
    t0 = time.perf_counter()
    log.info(
        "[%s] HF pipeline run_timeline: %d segments seed=%d character=%s",
        job_id, len(segments), seed, character,
    )

    _timings: dict[str, float] = {}
    _stage = _make_stage_cb(on_progress, on_stage)
    _stage("Generating long motion…", 0.05)

    stages = _stages()
    plan_specs = stages.plan.build_timeline_plan(
        segments, character,
        seed=seed,
        cfg=cfg,
        handshake_size=handshake_size,
        blend_len=blend_len,
        output_fps=output_fps,
        snap_to_ground=snap_to_ground,
        keyframe_builder=keyframe_builder,
        keyframe_builder_error_degrees=keyframe_builder_error_degrees,
        compress_output=compress_output,
    )

    # ── Stage 1: GPU inference (priorMDM double-take) ────────────────────
    _t = time.perf_counter()
    npz_bytes = _run_timeline_gpu(
        segments=segments,
        seed=seed,
        guidance_param=cfg,
        handshake_size=handshake_size,
        blend_len=blend_len,
    )
    npz_bytes = _unwrap_gpu_result(npz_bytes)
    _timings["gpu"] = time.perf_counter() - _t
    log.info(
        "[%s] timeline inference done in %.1fs (%d KB NPZ)",
        job_id, _timings["gpu"], len(npz_bytes) // 1024,
    )

    return _run_tail(
        job_id=job_id,
        npz_bytes=npz_bytes,
        plan_specs=plan_specs,
        output_fps=output_fps,
        compress_output=compress_output,
        _stage=_stage,
        _timings=_timings,
        t0=t0,
    )


# Boot pre-warm hook — module import happens once at Space startup (ui.py).
# No-op unless KIMODO_PREWARM=1.
start_kimodo_prewarm_if_enabled()
