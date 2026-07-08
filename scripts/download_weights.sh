#!/usr/bin/env bash
# Best-effort MDM weight download.
#
# DELIBERATE: this script never aborts the entrypoint. Failures log a
# warning + exit 0 so the orchestrator can boot in a "checkpoints missing"
# state. /v1/health, /v1/tasks, and the Gradio UI all work without
# weights; only actual inference requests fail (with a clear "checkpoint
# missing at <path>" error).
#
# Manual fallback if the HF mirror URLs below ever break: grab the MDM
# checkpoint from the upstream Google Drive link in
# https://github.com/GuyTevet/motion-diffusion-model README and drop it at
#   ${CHECKPOINTS_DIR:-/opt/checkpoints}/humanml_trans_enc_512/model000475000.pt
# alongside its args.json. Same for priorMDM and MoMask.

set -uo pipefail   # NOT -e — best-effort

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/opt/checkpoints}"
mkdir -p "${CHECKPOINTS_DIR}"

_warn() { echo "[weights] WARN: $*" >&2 ; }
_ok()   { echo "[weights] $*" ; }

# Fast-path: if the Docker build already baked the canonical 50-step MDM
# checkpoint via `huggingface_hub.snapshot_download`, there's nothing to
# fetch at runtime. Skip the legacy mirror dance entirely.
if [ -f "${CHECKPOINTS_DIR}/humanml_enc_512_50steps/model000750000.pt" ]; then
    _ok "MDM 50-step checkpoint already baked at ${CHECKPOINTS_DIR}/humanml_enc_512_50steps/ (skip)"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# MDM (humanml_trans_enc_512) — legacy fallback for OSS-local users who
# build without the hf_token secret. Best-effort; failures don't abort.
# ─────────────────────────────────────────────────────────────────────
MDM_DIR="${CHECKPOINTS_DIR}/humanml_trans_enc_512"
MDM_PT="${MDM_DIR}/model000475000.pt"
MDM_ARGS="${MDM_DIR}/args.json"

if [ -f "${MDM_PT}" ]; then
    _ok "MDM already present at ${MDM_DIR} (skip)"
else
    _ok "downloading MDM humanml_trans_enc_512 → ${MDM_DIR}"
    mkdir -p "${MDM_DIR}"

    # Candidate mirror URLs. Add more as we learn what's reachable; first
    # one that works wins. None of these are guaranteed — if all fail,
    # the warning at the bottom tells the user how to drop weights manually.
    _MIRRORS_PT=(
        "https://huggingface.co/guytevet/motion-diffusion-model/resolve/main/save/humanml_trans_enc_512/model000475000.pt"
        "https://huggingface.co/guytevet/MDM/resolve/main/humanml_trans_enc_512/model000475000.pt"
    )
    _MIRRORS_ARGS=(
        "https://huggingface.co/guytevet/motion-diffusion-model/resolve/main/save/humanml_trans_enc_512/args.json"
        "https://huggingface.co/guytevet/MDM/resolve/main/humanml_trans_enc_512/args.json"
    )

    _got_pt=0
    for url in "${_MIRRORS_PT[@]}"; do
        if curl -fsSL --max-time 300 "${url}" -o "${MDM_PT}.partial" 2>/dev/null; then
            mv "${MDM_PT}.partial" "${MDM_PT}"
            _got_pt=1
            _ok "MDM .pt fetched from ${url}"
            break
        fi
        rm -f "${MDM_PT}.partial" 2>/dev/null
    done

    if [ "${_got_pt}" = "1" ]; then
        for url in "${_MIRRORS_ARGS[@]}"; do
            if curl -fsSL --max-time 60 "${url}" -o "${MDM_ARGS}.partial" 2>/dev/null; then
                mv "${MDM_ARGS}.partial" "${MDM_ARGS}"
                _ok "MDM args.json fetched from ${url}"
                break
            fi
            rm -f "${MDM_ARGS}.partial" 2>/dev/null
        done
    fi

    if [ ! -f "${MDM_PT}" ]; then
        _warn "all MDM mirror URLs failed."
        _warn "To enable MDM inference, drop the checkpoint manually:"
        _warn "  ${MDM_PT}"
        _warn "  ${MDM_ARGS}"
        _warn "Source: https://github.com/GuyTevet/motion-diffusion-model README → 'Get data and checkpoints' (Google Drive)."
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# t2m_mean / t2m_std symlinks
# These are required for the HumanML3D normalization. They're checked
# into comfyui-animoflow/containers/mdm/ and the upstream MDM code
# expects them at $MDM_PATH/dataset/HumanML3D/.
# ─────────────────────────────────────────────────────────────────────
MDM_SRC="${MDM_PATH:-/opt/mdm-codes}"
if [ -d "${MDM_SRC}" ]; then
    mkdir -p "${MDM_SRC}/dataset/HumanML3D" 2>/dev/null
    for npy in t2m_mean t2m_std; do
        SRC="/opt/comfyui-animoflow/containers/mdm/${npy}.npy"
        DST="${MDM_SRC}/dataset/HumanML3D/${npy}.npy"
        if [ -f "${SRC}" ] && [ ! -e "${DST}" ]; then
            ln -sfn "${SRC}" "${DST}" 2>/dev/null && _ok "linked ${npy}.npy"
        fi
    done
fi

_ok "done (best-effort)"
exit 0
