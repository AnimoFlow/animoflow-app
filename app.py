"""
animoflow-app — orchestrator entry point.

Mounts a Gradio UI on top of the imported animoflow-api FastAPI app, then
monkeypatches animoflow-api's `pipeline.run` / `pipeline.run_timeline` with
our HF-flavored pipeline that calls the GPU-decorated inference function
in-process and runs all post-processing CPU-side in the orchestrator.

NOTHING in animoflow-api/api/main.py or animoflow-api/api/pipeline.py is
modified on disk. The wrap-don't-modify rule extends transitively to our
own repos here.

Order of operations matters:
  1. sys.path inject /opt/animoflow-api/api so its sibling-module imports
     (`import config`, `import job_store`, `from auth import …`) resolve.
  2. Import the `pipeline` module FROM that path.
  3. Replace pipeline.run / pipeline.run_timeline with our pipeline_hf
     versions. Functions are looked up at call time inside
     animoflow-api/api/main.py:_run_pipeline, so monkeypatching after import
     is enough.
  4. Import animoflow-api's `main` (which pulls in pipeline by name).
  5. Build Gradio Blocks and mount them on the FastAPI app at "/".
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("animoflow-app")

# ---------------------------------------------------------------------------
# Step -2: ensure OUTPUT_DIR is consistent across all modules.
#
# pipeline_hf defaults to /tmp/animoflow-output; animoflow-api/api/config.py
# defaults to a home-directory path. On HF Spaces (Gradio SDK, no Dockerfile ENV),
# the env var may not be set, so the two defaults diverge and /v1/files/
# returns 404 (config.OUTPUT_DIR points at a dir the pipeline never wrote to).
# Fix: plant the env var before any animoflow-api import.
# ---------------------------------------------------------------------------

os.environ.setdefault("OUTPUT_DIR", "/tmp/animoflow-output")

# HF Space's xet protocol fails with "Permission denied (os error 13)" trying
# to write its cache logs to /home/user/.cache/huggingface/xet/. Bootstrapping
# huggingface_hub via the legacy LFS-style download avoids the issue. Surfaced
# when sentence-transformers landed in requirements.txt and pip resolved
# transformers up to 5.12 which uses xet by default.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---------------------------------------------------------------------------
# Step -1: bootstrap external repos + checkpoints when running on HF Gradio
# Spaces (which have no Docker build phase). No-op in Docker mode.
# ---------------------------------------------------------------------------

# Make sure our own dir is on sys.path so `import bootstrap` works regardless
# of how the process was launched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import bootstrap as _bootstrap  # noqa: E402

_bootstrap.bootstrap_external_repos()

# ---------------------------------------------------------------------------
# Step 0: auto-detect Blender for the rig + GLB stages.
# `MotionRetargeter.bvh_to_fbx` reads $BLENDER_BIN at module load. If it's
# not set (e.g. a fresh OSS-local Mac dev), we plant the right value here
# before any retargeter import. The pipeline_hf._find_blender() helper
# uses the same priority order.
# ---------------------------------------------------------------------------

import shutil as _shutil
from pathlib import Path as _Path

if not os.environ.get("BLENDER_BIN", "").strip():
    for _candidate in (
        _shutil.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/blender/blender",
    ):
        if _candidate and _Path(_candidate).is_file():
            os.environ["BLENDER_BIN"] = _candidate
            log.info("Auto-detected Blender at %s", _candidate)
            break
    else:
        log.warning(
            "Blender not found on PATH or at common install locations. "
            "Rig + GLB stages will fail. Set BLENDER_BIN to override."
        )

# ---------------------------------------------------------------------------
# Step 1: locate animoflow-api and put its api/ on sys.path
# ---------------------------------------------------------------------------

_ANIMOFLOW_API_API = os.environ.get(
    "ANIMOFLOW_API_API_DIR", "/opt/animoflow-api/api"
)
if not Path(_ANIMOFLOW_API_API).is_dir():
    raise RuntimeError(
        f"animoflow-api/api not found at {_ANIMOFLOW_API_API!r}. "
        "Set ANIMOFLOW_API_API_DIR or rebuild the Docker image. "
        "The wrapper repo must be cloned at Docker build time."
    )
if _ANIMOFLOW_API_API not in sys.path:
    sys.path.insert(0, _ANIMOFLOW_API_API)

# ---------------------------------------------------------------------------
# Step 2: locate animoflow-app's own code on sys.path so pipeline_hf etc.
# import cleanly when uvicorn is invoked from another cwd.
# ---------------------------------------------------------------------------

_ANIMOFLOW_APP = Path(__file__).resolve().parent
if str(_ANIMOFLOW_APP) not in sys.path:
    sys.path.insert(0, str(_ANIMOFLOW_APP))

# ---------------------------------------------------------------------------
# Step 3: import animoflow-api's `pipeline` and replace its public functions
# ---------------------------------------------------------------------------

import pipeline as _animoflow_api_pipeline  # noqa: E402  (animoflow-api's module)
import pipeline_hf  # noqa: E402  (ours, replaces the ComfyUI-orchestrated path)

_animoflow_api_pipeline.run = pipeline_hf.run
_animoflow_api_pipeline.run_timeline = pipeline_hf.run_timeline
log.info("Monkeypatched animoflow-api pipeline.run / run_timeline → pipeline_hf")

# ---------------------------------------------------------------------------
# Step 4: import the FastAPI app from animoflow-api/api/main.py
# ---------------------------------------------------------------------------

from main import app as fastapi_app  # noqa: E402

log.info("Imported animoflow-api FastAPI app (title=%r)", fastapi_app.title)

# ---------------------------------------------------------------------------
# Step 4b: warm up the multilingual prompt rewriter in the parent process so
# the @spaces.GPU fork inherits the loaded Qwen + MiniLM + corpus via COW on
# every subsequent call. Without this, the very first GPU rewrite has to
# pay the model-load + CUDA-move cost INSIDE the @GPU budget, and gets killed
# by ZeroGPU with 'GPU task aborted' once it exceeds the duration.
# Same eager-load pattern as the MDM registry (animoflow_models/registry.py).
# ---------------------------------------------------------------------------
try:
    import rewriter as _rewriter  # noqa: E402  (animoflow-api's module)
    _rewriter.warmup()
    log.info("Rewriter warmed up in orchestrator process")
except ImportError:
    log.warning("Rewriter module not on sys.path — skipping warmup")
except Exception as e:
    log.warning("Rewriter warmup raised — first /v1/jobs request will retry: %s", e)

# ---------------------------------------------------------------------------
# Step 5: build the Gradio Blocks UI and mount on "/"
# ---------------------------------------------------------------------------

import gradio as gr  # noqa: E402
from ui import build_blocks  # noqa: E402

_blocks = build_blocks()

# Kimodo health surface. animoflow-api's _MODEL_HEALTH_URLS polls
# ${KIMODO_ENDPOINT}/health every HEALTH_POLL_INTERVAL seconds. Default
# KIMODO_ENDPOINT in animoflow-api/api/config.py points at localhost:8005
# (the local-stack Docker container), which is unreachable inside the HF
# Space. bootstrap._set_env_for_kimodo plants KIMODO_ENDPOINT pointing at
# this in-process route — Kimodo surfaces as available in /v1/models (and
# the webUI dropdown) once the venv build finishes.
@fastapi_app.get("/__internal/kimodo/health")
def _kimodo_internal_health():  # noqa: D401
    from fastapi.responses import JSONResponse
    # escape_hatch/__init__.py does `from .invoke import invoke` which
    # overwrites the submodule reference with the function. Even
    # `import escape_hatch.invoke as X` then binds X to the function. Pull
    # the module straight from sys.modules to bypass the shadowing.
    import escape_hatch  # noqa: F401 — ensure escape_hatch.invoke is in sys.modules
    import sys as _sys
    _ehi = _sys.modules["escape_hatch.invoke"]

    if os.environ.get("ENABLE_KIMODO", "true").strip().lower() == "false":
        return JSONResponse(status_code=503, content={"status": "disabled"})
    failed = _ehi._kimodo_failed_message()
    if failed:
        return JSONResponse(
            status_code=503,
            content={"status": "failed", "error": failed[:400]},
        )
    event = _ehi._KIMODO_READY
    ready_sentinel = _ehi._external_dir() / ".kimodo_ready"
    ready = ready_sentinel.is_file() or (event is not None and event.is_set())
    if ready:
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "warming"})


# `gr.mount_gradio_app` registers Gradio's routes on the FastAPI app.
# WEB_DIR=/nonexistent in our Dockerfile makes animoflow-api skip its
# StaticFiles("/") catch-all, so "/" is free for Gradio to claim.
_output_dir = os.environ.get("OUTPUT_DIR", "/tmp/animoflow-output")
fastapi_app = gr.mount_gradio_app(
    fastapi_app, _blocks, path="/",
    allowed_paths=[_output_dir],
)
log.info("Mounted Gradio UI at / (allowed_paths=%s)", [_output_dir])

# HF Gradio SDK auto-detection: expose `demo` at module level
demo = _blocks


def main() -> None:
    """Local-dev convenience entry: `python app.py` boots uvicorn."""
    import uvicorn

    uvicorn.run(
        "app:fastapi_app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        # On HF Gradio SDK — ZeroGPU requires demo.launch() for GPU
        # registration. We monkey-patch Gradio's app factory to inject
        # our /v1/* FastAPI routes into the Gradio-managed app.
        import gradio.routes as _gr_routes

        _orig_create = _gr_routes.App.create_app

        def _patched_create(blocks, **kw):
            gapp = _orig_create(blocks, **kw)
            # Inject animoflow-api's API routes. PREPEND, don't append —
            # Starlette matches routes in registration order and Gradio
            # ships its own `/openapi.json`, `/docs`, `/redoc` that would
            # otherwise win and serve Gradio's internal spec instead of
            # animoflow-api's branded one (title: "AnimoFlow API",
            # 10 documented /v1/* routes). Inserting at index 0 in
            # reverse iteration order preserves animoflow-api's own
            # ordering. The /v1/* + /oauth/* routes were already unique
            # to animoflow-api so prepending doesn't change their
            # behavior — only the spec-surface routes flip to ours.
            for route in reversed(list(fastapi_app.routes)):
                gapp.routes.insert(0, route)
            # Copy middleware
            for mw in fastapi_app.user_middleware:
                gapp.add_middleware(mw.cls, **mw.kwargs)
            # Copy exception handlers
            for exc_cls, handler in fastapi_app.exception_handlers.items():
                gapp.add_exception_handler(exc_cls, handler)
            # slowapi's RateLimitExceeded handler reads
            # request.app.state.limiter — and request.app here is THIS
            # Gradio app, not animoflow-api's. Without this line every
            # rate-limit hit 500s with AttributeError instead of a 429
            # (found 2026-07-06 by the quota probe's parallel round).
            gapp.state.limiter = fastapi_app.state.limiter
            log.info("Injected animoflow-api routes into Gradio app (prepended for /openapi.json + /redoc + /docs precedence)")
            return gapp

        _gr_routes.App.create_app = _patched_create
        demo.launch(allowed_paths=[_output_dir])
    else:
        main()
