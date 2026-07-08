# syntax=docker/dockerfile:1.6
#
# animoflow-app — HF Space + OSS local Docker image.
#
# Single-venv orchestrator with warm MDM-family models loaded at module level.
# Per-(comfy+model) escape-hatch venvs isolate outlier models
# (Kimodo NVIDIA stack, future additions).
#
# ─────────────────────────────────────────────────────────────────────
# AUTH: this Dockerfile uses two BuildKit secrets, both never landing
# in any image layer:
#
#   gh_token  — read access to AnimoFlow/{comfyui-animoflow,animoflow-api}
#               (currently private; the secret-mount keeps working
#               after either repo flips public).
#   hf_token  — read access to AnimoFlow/animoflow-checkpoints (the
#               private HF Hub model repo where MDM weights live; baked
#               into the image at build time).
#
# HF Spaces autopopulate Space-level secrets with these names. For local
# builds you pass them via `--secret id=<name>,src=<path-to-token-file>`.
#
# Mac local build:
#     gh auth token > /tmp/gh_token
#     DOCKER_BUILDKIT=1 docker build \
#         --secret id=gh_token,src=/tmp/gh_token \
#         -t animoflow/app .
#     rm /tmp/gh_token
#
# HF Space build: configure a Space secret named `gh_token` in
# Settings → Variables and secrets. HF passes it to the build automatically.
#
# Run (OSS local, after build):
#     docker run --rm -p 7860:7860 animoflow/app
#
# Build args let you pin upstream SHAs reproducibly. CI bumps these
# deliberately, never via `git pull` behind our back (wrap upstream
# repos, don't fork them).

ARG PYTHON_VERSION=3.11
ARG BLENDER_VERSION=4.2.5
ARG COMFYUI_ANIMOFLOW_REPO=https://github.com/AnimoFlow/comfyui-animoflow.git
ARG ANIMOFLOW_API_REPO=https://github.com/AnimoFlow/animoflow-api.git
ARG MDM_REPO=https://github.com/GuyTevet/motion-diffusion-model.git
ARG MOMASK_REPO=https://github.com/EricGuo5513/momask-codes.git
# SHA pins — update deliberately, not via `git pull`
ARG COMFYUI_ANIMOFLOW_SHA=main
ARG ANIMOFLOW_API_SHA=main
ARG MDM_SHA=main
ARG MOMASK_SHA=main

# Baked checkpoints — pulled from a private HF Hub model repo at build
# time using the `hf_token` BuildKit secret. SHA-pinned for reproducibility;
# bump deliberately. The repo holds full training artifacts (final + earlier
# checkpoints + optimizer state + eval logs) but the Dockerfile filters via
# `allow_patterns` to bake only the inference-essential files.
ARG ANIMOFLOW_CHECKPOINTS_REPO=AnimoFlow/animoflow-checkpoints
ARG ANIMOFLOW_CHECKPOINTS_REVISION=9c8b352f849c5e81869a50f71667d43c064571eb

FROM python:${PYTHON_VERSION}-slim AS base

# ─────────────────────────────────────────────────────────────────────
# System deps
# ─────────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates xz-utils \
        xvfb xauth \
        libxi6 libxrender1 libxxf86vm1 libxfixes3 libxkbcommon0 \
        libsm6 libxext6 libgl1 libglu1-mesa libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────
# Blender 4.2 LTS (pinned tarball — apt's blender lags 3.x).
# Arch-aware: Docker BuildKit auto-populates TARGETARCH (amd64 / arm64).
# Default `docker build` (no buildx) leaves it empty and falls through to
# amd64, matching HF Spaces (x86_64 H200 hosts).
# ─────────────────────────────────────────────────────────────────────
ARG BLENDER_VERSION
ARG TARGETARCH
RUN BLENDER_MAJ=$(echo "${BLENDER_VERSION}" | cut -d. -f1,2) \
 && case "${TARGETARCH:-amd64}" in \
        amd64|x86_64|"") BLENDER_ARCH=linux-x64 ;; \
        arm64|aarch64)   BLENDER_ARCH=linux-arm64 ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2 ; exit 1 ;; \
    esac \
 && echo "Building for ${TARGETARCH:-amd64} → blender-${BLENDER_VERSION}-${BLENDER_ARCH}.tar.xz" \
 && curl -fsSL "https://download.blender.org/release/Blender${BLENDER_MAJ}/blender-${BLENDER_VERSION}-${BLENDER_ARCH}.tar.xz" -o /tmp/blender.tar.xz \
 && mkdir -p /opt/blender \
 && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
 && rm /tmp/blender.tar.xz \
 && /opt/blender/blender --version
ENV BLENDER_BIN=/opt/blender/blender

# ─────────────────────────────────────────────────────────────────────
# Clone wrapper repos — wrap-don't-modify, pinned by SHA build args.
#
# Both `comfyui-animoflow` and `animoflow-api` are currently private. We
# clone both via a BuildKit secret named `gh_token` (the token is
# interpolated into the URL temporarily and never lands in any image
# layer — we strip the remote URL after the clone). The same pattern
# works once either repo is flipped to public, so this keeps working
# across the public flip.
# ─────────────────────────────────────────────────────────────────────
ARG COMFYUI_ANIMOFLOW_REPO
ARG COMFYUI_ANIMOFLOW_SHA
ARG ANIMOFLOW_API_REPO
ARG ANIMOFLOW_API_SHA

RUN --mount=type=secret,id=gh_token,required=true \
    GH_TOKEN="$(cat /run/secrets/gh_token)" \
 && _AUTH_CAF=$(echo "${COMFYUI_ANIMOFLOW_REPO}" | sed -e "s|https://|https://x-access-token:${GH_TOKEN}@|") \
 && _AUTH_AFW=$(echo "${ANIMOFLOW_API_REPO}"     | sed -e "s|https://|https://x-access-token:${GH_TOKEN}@|") \
 && git clone "${_AUTH_CAF}" /opt/comfyui-animoflow \
 && git -C /opt/comfyui-animoflow checkout "${COMFYUI_ANIMOFLOW_SHA}" \
 && git -C /opt/comfyui-animoflow remote set-url origin "${COMFYUI_ANIMOFLOW_REPO}" \
 && git clone "${_AUTH_AFW}" /opt/animoflow-api \
 && git -C /opt/animoflow-api checkout "${ANIMOFLOW_API_SHA}" \
 && git -C /opt/animoflow-api remote set-url origin "${ANIMOFLOW_API_REPO}" \
 && unset GH_TOKEN _AUTH_CAF _AUTH_AFW

# ─────────────────────────────────────────────────────────────────────
# Clone upstream models.
# priorMDM and Kimodo come in later iterations.
# ─────────────────────────────────────────────────────────────────────
ARG MDM_REPO
ARG MDM_SHA
ARG MOMASK_REPO
ARG MOMASK_SHA
RUN git clone "${MDM_REPO}" /opt/mdm-codes \
 && git -C /opt/mdm-codes checkout "${MDM_SHA}"
RUN git clone "${MOMASK_REPO}" /opt/momask-codes \
 && git -C /opt/momask-codes checkout "${MOMASK_SHA}"

ENV MDM_PATH=/opt/mdm-codes
ENV MOMASK_PATH=/opt/momask-codes

# ─────────────────────────────────────────────────────────────────────
# Single Python venv (orchestrator + MDM family).
# `uv venv` creates a bare virtualenv (no pip bundled by default — uv
# expects `uv pip install --python <venv>` instead of bare pip). Use the
# `uv pip` form throughout.
# ─────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir uv==0.5.11

# Create venv + install orchestrator/MDM-family deps.
#
# `--seed` populates the venv with pip + setuptools + wheel, and
# `--no-build-isolation` makes uv build sdists against the target venv
# (which has those tools) instead of uv's pip-less isolated build env.
# That combo handles legacy packages whose setup.py imports pip internals,
# without disabling build isolation everywhere.
COPY requirements.txt /opt/animoflow-app/requirements.txt
RUN uv venv /opt/venvs/api --python ${PYTHON_VERSION} --seed \
 && uv pip install --python /opt/venvs/api --no-cache --no-build-isolation \
        -r /opt/animoflow-app/requirements.txt

# NOTE: we deliberately do NOT install
# `comfyui-animoflow/containers/mdm/requirements.txt` here. That file pins
# torch==2.3.0 which downgrades our 2.4.1 pin and removes
# `torch.library.register_fake` — an API the upstream MDM source uses, so
# the wrapper falls back to placeholder mode.
#
# Instead we lift the MDM-container deps that aren't already in our pin
# (einops) directly into requirements.txt above. The local Docker
# stack uses a different topology (each model in its own container with
# its own venv) so the version skew there doesn't bite.

# ─────────────────────────────────────────────────────────────────────
# Bake model checkpoints from the AnimoFlow/animoflow-checkpoints HF Hub
# model repo. Token only mounted for the duration of this RUN — never
# lands in any image layer. Failure here aborts the build (we don't want
# to ship an HF Space without weights).
#
# To override the source repo or pin a specific revision, pass build args:
#   --build-arg ANIMOFLOW_CHECKPOINTS_REPO=AnimoFlow/different-repo
#   --build-arg ANIMOFLOW_CHECKPOINTS_REVISION=<commit-sha>
# ─────────────────────────────────────────────────────────────────────
ARG ANIMOFLOW_CHECKPOINTS_REPO
ARG ANIMOFLOW_CHECKPOINTS_REVISION
RUN --mount=type=secret,id=hf_token,required=true \
    export HF_TOKEN="$(cat /run/secrets/hf_token)" \
 && echo "Downloading checkpoints from ${ANIMOFLOW_CHECKPOINTS_REPO}@${ANIMOFLOW_CHECKPOINTS_REVISION}…" \
 && /opt/venvs/api/bin/python -c "\
import os; \
from huggingface_hub import snapshot_download; \
path = snapshot_download( \
    repo_id='${ANIMOFLOW_CHECKPOINTS_REPO}', \
    revision='${ANIMOFLOW_CHECKPOINTS_REVISION}', \
    repo_type='model', \
    local_dir='/opt/checkpoints', \
    allow_patterns=[ \
        'humanml_enc_512_50steps/model000750000.pt', \
        'humanml_enc_512_50steps/args.json', \
    ], \
    token=os.environ['HF_TOKEN']); \
print('checkpoints baked at', path)" \
 && unset HF_TOKEN \
 && cp /opt/comfyui-animoflow/containers/mdm/t2m_mean.npy /opt/checkpoints/t2m_mean.npy \
 && cp /opt/comfyui-animoflow/containers/mdm/t2m_std.npy  /opt/checkpoints/t2m_std.npy \
 && ls -laR /opt/checkpoints/ \
 && du -sh /opt/checkpoints/

# WEIGHTS_DIR is what comfyui-animoflow/containers/mdm/inference.py reads for
# both the model checkpoint AND the t2m_mean/std normalization arrays.
# Setting it as ENV makes the MDM wrapper find them without per-run config.
ENV WEIGHTS_DIR=/opt/checkpoints

# ─────────────────────────────────────────────────────────────────────
# Multilingual prompt rewriter — Qwen2.5-1.5B-Instruct + RAFSL few-shot
# from the local HumanML3D caption corpus. Phase-0 verified at 90% pass
# rate / 0.32s server-only on ZeroGPU. See:
#   - animoflow-api/api/rewriter.py (the consumer)
#
# All three assets SHA-pinned. Bump deliberately, not via "latest".
# ─────────────────────────────────────────────────────────────────────
ARG REWRITER_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct
ARG REWRITER_MODEL_REVISION=989aa7980e4c
ARG REWRITER_RETRIEVER_REPO=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG REWRITER_RETRIEVER_REVISION=e8f8c211226b
ARG REWRITER_CORPUS_REPO=AnimoFlow/rewriter-corpus
ARG REWRITER_CORPUS_REVISION=5b98da4d2db075d40afd59badf51f2eefe1db5e2

RUN /opt/venvs/api/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='${REWRITER_MODEL_REPO}', \
    revision='${REWRITER_MODEL_REVISION}', \
    local_dir='/opt/rewriter/qwen', \
    allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','vocab*','merges*'], \
); \
snapshot_download( \
    repo_id='${REWRITER_RETRIEVER_REPO}', \
    revision='${REWRITER_RETRIEVER_REVISION}', \
    local_dir='/opt/rewriter/minilm', \
    allow_patterns=['*.safetensors','*.json','tokenizer*','*.txt','vocab*','*.bin'], \
); \
snapshot_download( \
    repo_id='${REWRITER_CORPUS_REPO}', \
    revision='${REWRITER_CORPUS_REVISION}', \
    repo_type='dataset', \
    local_dir='/opt/rewriter/corpus', \
    allow_patterns=['captions.json','embeddings.npy'], \
); \
print('rewriter baked at /opt/rewriter')" \
 && ls -laR /opt/rewriter/ \
 && du -sh /opt/rewriter/*

ENV REWRITER_DATA_DIR=/opt/rewriter

ENV PATH=/opt/venvs/api/bin:${PATH}
ENV PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────
# Copy the orchestrator code last (cache-friendly)
# ─────────────────────────────────────────────────────────────────────
COPY . /opt/animoflow-app

# ─────────────────────────────────────────────────────────────────────
# Runtime config
# ─────────────────────────────────────────────────────────────────────
ENV OUTPUT_DIR=/tmp/animoflow-output
ENV CHECKPOINTS_DIR=/opt/checkpoints
ENV CHARACTERS_DIR=/opt/comfyui-animoflow/characters
ENV WEB_DIR=/nonexistent
# Bogus model-server URLs — animoflow-api's health poller will log "unreachable"
# but no separate model-server processes run on HF. Harmless.
ENV MDM_ENDPOINT=http://127.0.0.1:65500
ENV PRIORMDM_ENDPOINT=http://127.0.0.1:65501
ENV MOMASK_ENDPOINT=http://127.0.0.1:65502
ENV KIMODO_ENDPOINT=http://127.0.0.1:65504
ENV COMFYUI_URL=http://127.0.0.1:65505
ENV HEALTH_POLL_INTERVAL=600
ENV PORT=7860

RUN mkdir -p ${OUTPUT_DIR} ${CHECKPOINTS_DIR} \
 && chmod +x /opt/animoflow-app/entrypoint.sh /opt/animoflow-app/scripts/*.sh

EXPOSE 7860
WORKDIR /opt/animoflow-app

# Healthcheck via the FastAPI health endpoint (exposed even before Gradio mounts)
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7860/v1/health || exit 1

CMD ["/opt/animoflow-app/entrypoint.sh"]
