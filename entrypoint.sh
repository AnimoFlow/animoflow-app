#!/usr/bin/env bash
# animoflow-app entrypoint — boots the orchestrator inside the HF Space (or
# any host running this image). Idempotent.
#
# DELIBERATE: not `set -e`. Weight download and Blender health-check are
# best-effort; the orchestrator must start even if either fails. /v1/health
# and /v1/tasks work without weights or Blender; inference / rig / GLB
# routes will surface clear errors at request time if their deps are
# missing. Better than dying silently in a Space we can't easily debug.
set -uo pipefail

cd /opt/animoflow-app

# 1. Download MDM checkpoints if missing. download_weights.sh is idempotent
#    AND best-effort: a non-zero exit logs a warning but does not abort
#    boot. Set SKIP_CHECKPOINTS=1 to skip the download step entirely.
if [ "${SKIP_CHECKPOINTS:-0}" != "1" ] && [ -x /opt/animoflow-app/scripts/download_weights.sh ]; then
    /opt/animoflow-app/scripts/download_weights.sh \
        || echo "[entrypoint] WARN: weight download script returned non-zero. Orchestrator will boot anyway; inference will fail with a clear 'checkpoint missing' error until weights are present."
fi

# 2. Health-check Blender (warn only — do not abort).
if ! "${BLENDER_BIN:-/opt/blender/blender}" --version >/dev/null 2>&1; then
    echo "[entrypoint] WARN: Blender not found at ${BLENDER_BIN:-/opt/blender/blender}"
    echo "[entrypoint] Rig + GLB stages will fail. Inference-only requests still work."
fi

# 3. Launch FastAPI + Gradio. uvicorn loads fastapi_app from app.py.
exec /opt/venvs/api/bin/uvicorn \
    --app-dir /opt/animoflow-app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    app:fastapi_app
