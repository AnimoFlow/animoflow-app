"""
Subprocess-based invocation for outlier-model venvs.

For each outlier model, we maintain:
  /opt/venvs/comfy-<model>/  — its isolated Python venv (built at Docker time)
  scripts/run_inference_<model>.py — a tiny CLI we own that calls the
                                     model's wrapper code with JSON args

The decorated @spaces.GPU function in pipeline_hf.py calls invoke(model, …)
which spawns the subprocess. CUDA visibility is inherited via the fork
parent → subprocess env chain.

Adding an outlier model: add an entry to _OUTLIERS and ship the
matching CLI script.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Map model name → {venv_python, runner_script}.
# Paths are env-overridable so bootstrap.py (HF Gradio mode) and the Dockerfile
# (OSS local Docker mode) can plant them at the locations they actually build.
_OUTLIERS: dict[str, dict[str, str]] = {
    "kimodo": {
        "venv_python": os.environ.get(
            "KIMODO_VENV_PYTHON", "/home/user/app/venvs/kimodo/bin/python",
        ),
        "runner_script": os.environ.get(
            "KIMODO_RUNNER_SCRIPT",
            "/home/user/app/scripts/run_inference_kimodo.py",
        ),
    },
}

# Per-model subprocess timeout (seconds). Kimodo's first call after a Space
# restart pays the full model load (~120 s including LLaMA encode warmup);
# warm calls are ~10-15 s. The matching @spaces.GPU(duration=…) lives in
# pipeline_hf._run_inference_gpu_kimodo.
_TIMEOUTS: dict[str, int] = {
    "kimodo": 300,
}

# Set by bootstrap._install_kimodo_async once the venv build finishes. The
# runner script also reads the .kimodo_ready sentinel as a backup, so this is
# defence-in-depth (the event lets us fail fast without spawning a subprocess
# that would immediately exit with "venv not ready").
_KIMODO_READY: threading.Event | None = None


def _external_dir() -> Path:
    """Mirror bootstrap.EXTERNAL_DIR without importing bootstrap (which would
    re-run import-time work)."""
    return Path(
        os.environ.get(
            "ANIMOFLOW_EXTERNAL_DIR",
            "/home/user/app/external"
            if os.path.isdir("/home/user/app")
            else str(Path(__file__).resolve().parent.parent / "external"),
        )
    )


def _kimodo_failed_message() -> str | None:
    """Return the captured Kimodo bootstrap error, or None if no .kimodo_failed."""
    sentinel = _external_dir() / ".kimodo_failed"
    if not sentinel.is_file():
        return None
    try:
        return sentinel.read_text()[:1200]
    except OSError:
        return "(unreadable .kimodo_failed sentinel)"


def is_outlier(model: str) -> bool:
    """True if the model is registered as an outlier and should be invoked
    via subprocess instead of the in-orchestrator registry."""
    return model in _OUTLIERS


def invoke(
    model: str,
    *,
    prompt: str,
    num_frames: int,
    seed: int,
    guidance_param: float | None = None,
    timeout: int | None = None,
    **extra: Any,
) -> bytes:
    """Invoke an outlier model via subprocess. Returns NPZ bytes.

    The subprocess inherits CUDA_VISIBLE_DEVICES from the parent process —
    on HF this is the GPU fork created by @spaces.GPU. The subprocess
    re-initializes CUDA cleanly because it's a fresh Python process (spawn-
    style isolation), avoiding the "Cannot re-initialize CUDA in forked
    subprocess" error that bites multiprocessing.fork.

    ``timeout`` defaults to the per-model value in ``_TIMEOUTS`` (Kimodo: 300 s),
    falling back to 60 s.
    """
    if model not in _OUTLIERS:
        raise RuntimeError(
            f"Model {model!r} not registered as an outlier. Known outliers: "
            f"{sorted(_OUTLIERS.keys())}. Add to escape_hatch.invoke._OUTLIERS first."
        )

    spec = _OUTLIERS[model]
    venv_python = spec["venv_python"]
    runner_script = spec["runner_script"]

    # Kimodo-specific readiness gate. Bootstrap builds the venv in a background
    # thread; refuse to spawn the subprocess until .kimodo_ready exists (or the
    # ready Event is set). Per the no-silent-fallback policy: if the build
    # FAILED, surface the captured error verbatim.
    if model == "kimodo":
        failed_msg = _kimodo_failed_message()
        if failed_msg is not None:
            raise RuntimeError(
                f"Kimodo venv build FAILED at bootstrap. "
                f"Captured error:\n{failed_msg}"
            )
        ready_sentinel = _external_dir() / ".kimodo_ready"
        if not ready_sentinel.is_file():
            # Block briefly on the in-process Event so a request that lands
            # right at the tail of the build doesn't 503 needlessly.
            event = _KIMODO_READY
            wait_s = int(os.environ.get("KIMODO_READY_WAIT_S", "30"))
            if event is not None and not event.is_set():
                event.wait(timeout=wait_s)
            if not ready_sentinel.is_file() and (event is None or not event.is_set()):
                raise RuntimeError(
                    "Kimodo venv still warming up — bootstrap installs deps "
                    "and downloads the 16 GB LLaMA-3-8B encoder on first boot. "
                    "Typical cold-start is 5-10 minutes; please retry."
                )

    if not Path(venv_python).exists():
        raise RuntimeError(
            f"Outlier venv not found: {venv_python}. "
            f"Was the bootstrap venv builder run? (set ENABLE_KIMODO=true)"
        )
    if not Path(runner_script).exists():
        raise RuntimeError(f"Runner script missing: {runner_script}")

    payload = {
        "prompt": prompt,
        "num_frames": num_frames,
        "seed": seed,
        "guidance_param": guidance_param,
        **extra,
    }

    effective_timeout = timeout if timeout is not None else _TIMEOUTS.get(model, 60)

    log.info(
        "Invoking outlier %r via %s (timeout=%ds, cwd-isolated subprocess)",
        model,
        venv_python,
        effective_timeout,
    )

    # Pass payload via stdin to avoid shell-quoting nightmares
    result = subprocess.run(
        [venv_python, runner_script],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=effective_timeout,
        env={**os.environ},  # inherit CUDA_VISIBLE_DEVICES from GPU fork
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Outlier {model!r} subprocess exited {result.returncode}. "
            f"stderr (last 400 chars): {result.stderr.decode(errors='replace')[-400:]}"
        )

    # Stdout is the NPZ bytes; stderr carries logs only
    npz_bytes = result.stdout
    if not npz_bytes:
        raise RuntimeError(
            f"Outlier {model!r} produced empty NPZ output. "
            f"stderr: {result.stderr.decode(errors='replace')[-400:]}"
        )
    return npz_bytes
