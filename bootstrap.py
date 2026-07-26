"""
Bootstrap external repos + checkpoints when running on HF Gradio SDK
Spaces (no Docker build phase to do it for us).

In Docker mode the Dockerfile already cloned the wrapper + upstream repos
and downloaded the baked checkpoints, so this module is a no-op there.
On HF Gradio Spaces (or any clean OSS-local checkout where the user
hasn't already provisioned external repos), we git-clone everything to
a known base dir and call `huggingface_hub.snapshot_download` for the
checkpoints.

Auth:
  * Private GitHub repos via the `gh_token` env var (Space secret on HF;
    plain env var locally).
  * Private HF Hub repo via `hf_token` env var.

Both never persisted to disk: the gh_token is used in a clone URL only
during the clone, then stripped from the remote URL post-clone.

Idempotent — safe to call multiple times. Skips any repo / file that
already exists at the target path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — overridable via env vars so OSS users can point this anywhere.
# ---------------------------------------------------------------------------

# Base dir for cloned external repos. On HF Gradio Spaces the working
# directory is /home/user/app/, so external/ sits next to the app code.
# In Docker mode the Dockerfile uses /opt/* and this whole module is a no-op
# (the Docker-mode marker is COMFYUI_ANIMOFLOW_DIR pointing at /opt/...).
_DEFAULT_EXTERNAL = (
    "/home/user/app/external"
    if os.path.isdir("/home/user/app")
    else str(Path(__file__).resolve().parent / "external")
)

EXTERNAL_DIR = Path(os.environ.get("ANIMOFLOW_EXTERNAL_DIR", _DEFAULT_EXTERNAL))

# Pinned upstream SHAs / revisions. Bump deliberately, not via git pull.
# MDM repo HEAD diverged (BERT encoder, refactored encode_text → shape
# changes, new imports) and breaks text-conditioned generation.  Pin to the
# last known-good commit that matches the checkpoint's architecture.
MDM_PINNED_COMMIT = os.environ.get("MDM_PINNED_COMMIT", "af061ca")

# Optional pins for AnimoFlow's own repos (empty = track main, today's
# behavior). Set these as Space variables at release time so a rebuild can
# never silently pick up newer main — e.g. the v0.1.0-beta tag's SHAs.
COMFYUI_PINNED_COMMIT = os.environ.get("COMFYUI_PINNED_COMMIT", "")
API_PINNED_COMMIT = os.environ.get("API_PINNED_COMMIT", "")

# GMR (github.com/YanjieZe/GMR, MIT) powers robot-character retargeting.
# Cloned pinned; installed WITHOUT its setup.py (which would pull the
# research-only smplx package the BVH path never imports) — the solver
# dependencies ship in requirements.txt instead. GMR_HOME points the
# robot_retarget package at the checkout.
GMR_PINNED_COMMIT = os.environ.get("GMR_PINNED_COMMIT", "bb1bbe40774794fceb2a7c579a3464a28e68c844")

ANIMOFLOW_CHECKPOINTS_REPO = os.environ.get(
    "ANIMOFLOW_CHECKPOINTS_REPO", "AnimoFlow/animoflow-checkpoints"
)
ANIMOFLOW_CHECKPOINTS_REVISION = os.environ.get(
    "ANIMOFLOW_CHECKPOINTS_REVISION",
    # 841707a = Pete removed; f0af610 = The Boss removed (2026-07-03,
    # Guy's call). Predecessor
    # cfbfe3c = first commit with characters/** (2026-07-03 webui character
    # bake — Mixamo FBXs live in this private repo, NOT in public
    # comfyui-animoflow; current roster: Vanguard/Knight/Suzie/Doozy). Predecessor
    # ff21939 was the first commit with priormdm/** (2026-06-21); its
    # predecessor 6df4521 had momask/t2m/** but no priorMDM, so
    # snapshot_download with allow_patterns=["priormdm/**"] globbed nothing
    # and the registry's _load("priormdm") failed cleanly at first call —
    # same bug class would hit characters/** on any pre-cfbfe3c revision.
    # 2026-07-06: characters/** moved OUT to ANIMOFLOW_CHARACTER_ASSETS_REPO
    # (Adobe-content isolation); this repo/revision now feeds weights only.
    "841707a1584c2ffc6085fff44f5b00c555fab634",
)

ANIMOFLOW_CHARACTER_ASSETS_REPO = os.environ.get(
    "ANIMOFLOW_CHARACTER_ASSETS_REPO", "AnimoFlow/character-assets"
)
ANIMOFLOW_CHARACTER_ASSETS_REVISION = os.environ.get(
    "ANIMOFLOW_CHARACTER_ASSETS_REVISION",
    # 07d4ab0 = adds Y_bot + Kaya (2026-07-07), byte-identical to the FBXs
    # comfyui-animoflow@7a91fa3 untracked for public launch — full 6-rig
    # roster now lives here. Predecessor
    # eec5ed0 = initial commit (2026-07-06): byte-identical characters/**
    # split out of animoflow-checkpoints@841707a — Adobe Mixamo FBXs get
    # their own private repo so the checkpoints repo can be shared without
    # dragging Adobe-licensed content along, and so tokens can be scoped
    # per-repo. eec5ed0 only mirrored the 4 checkpoints-repo rigs
    # (Doozy/Knight/Suzie/Vanguard) — Y_bot/Kaya lived in comfyui HEAD, so
    # the roster silently dropped to 4 on the first deploy of the split.
    # Same silent-glob bug class as documented on
    # ANIMOFLOW_CHECKPOINTS_REVISION applies: allow_patterns=
    # ["characters/**"] globs NOTHING (no error) on a revision lacking the
    # folder — this repo has it from commit one, but keep the pin anyway.
    # NEVER super_squash_history either private repo: pins die with it.
    "07d4ab0955b80af50db35847bf9b01d33087e710",
)

# Marker that says "Docker mode already provisioned everything" — if any of
# these exist we skip the bootstrap entirely.
_DOCKER_MODE_PATHS = (
    "/opt/comfyui-animoflow",
    "/opt/animoflow-api",
)

# What we need to clone. (repo_name, target_subdir, is_private).
# The AnimoFlow repo URLs are env-overridable so a staging Space can run
# against private staging mirrors without touching this file.
_GIT_REPOS: tuple[tuple[str, str, str, bool], ...] = (
    # (full_url_template_or_url, target_subdir, friendly_name, private)
    (
        os.environ.get(
            "COMFYUI_REPO_URL",
            "https://github.com/AnimoFlow/comfyui-animoflow.git",
        ),
        "comfyui-animoflow",
        "comfyui-animoflow",
        True,
    ),
    (
        os.environ.get(
            "API_REPO_URL",
            "https://github.com/AnimoFlow/animoflow-api.git",
        ),
        "animoflow-api",
        "animoflow-api",
        True,
    ),
    (
        "https://github.com/YanjieZe/GMR.git",
        "GMR",
        "GMR (robot retargeting)",
        False,
    ),
    (
        "https://github.com/GuyTevet/motion-diffusion-model.git",
        "mdm-codes",
        "mdm-codes",
        False,
    ),
    (
        "https://github.com/EricGuo5513/momask-codes.git",
        "momask-codes",
        "momask-codes",
        False,
    ),
    (
        "https://github.com/priorMDM/priorMDM.git",
        "priormdm-codes",
        "priormdm-codes",
        False,
    ),
)


# Kimodo source repos cloned separately (not in _GIT_REPOS) so that the
# ENABLE_KIMODO kill-switch can skip them entirely without affecting MDM/MoMask.
# kimodo-viser is cloned NESTED inside kimodo-src/ to match the editable-install
# layout the upstream Dockerfile expects (containers/kimodo/Dockerfile:22).
_KIMODO_REPOS: tuple[tuple[str, str, str], ...] = (
    ("https://github.com/nv-tlabs/kimodo.git", "kimodo-src", "kimodo-src"),
    ("https://github.com/nv-tlabs/kimodo-viser.git", "kimodo-src/kimodo-viser", "kimodo-viser"),
)


def _docker_mode() -> bool:
    """Return True if the Dockerfile already provisioned the external repos."""
    return any(Path(p).is_dir() for p in _DOCKER_MODE_PATHS)


# Blender 5.0.1 portable Linux tarball. Replaces Debian Trixie's apt Blender
# (4.3.2) — its retarget runtime was ~75x slower per stage timings.
_BLENDER_VERSION = "5.0.1"
_BLENDER_TARBALL_URL = (
    f"https://download.blender.org/release/Blender5.0/"
    f"blender-{_BLENDER_VERSION}-linux-x64.tar.xz"
)
_BLENDER_INSTALL_DIR = Path("/home/user/app") / f"blender-{_BLENDER_VERSION}-linux-x64"
_BLENDER_BIN_PATH = _BLENDER_INSTALL_DIR / "blender"


# gltfpack (meshoptimizer) — Draco + Meshopt GLB compression for Plan A.
# Used by pipeline_hf._compress_glb. Pinning to a release tag (not SHA) because
# upstream releases are stable artifacts — see [[preview-perf-2026-06-08
# -handoff]] gotcha 10.
_GLTFPACK_VERSION = "1.1"
_GLTFPACK_ZIP_URL = (
    f"https://github.com/zeux/meshoptimizer/releases/download/"
    f"v{_GLTFPACK_VERSION}/gltfpack-ubuntu.zip"
)
_GLTFPACK_INSTALL_DIR = Path("/home/user/app/bin")
_GLTFPACK_BIN_PATH = _GLTFPACK_INSTALL_DIR / "gltfpack"


def _install_gltfpack() -> None:
    """Install gltfpack (from meshoptimizer) for GLB compression.

    Used by pipeline_hf._compress_glb to apply Draco (geometry) + Meshopt
    (animation tracks + remaining buffers) compression to the GLB output.
    Validated 9.74x compression on a production Y_bot GLB (2.65 MB → 272 KB)
    pre-deployment.

    Per the no-silent-fallback rule:
    if the download or extract fails, raise. The Space must boot loudly
    on this kind of failure, not boot cleanly and only blow up on the
    first /generate call with a confusing "gltfpack not found" error.

    Idempotent — if the binary is already present and executable, just
    plant GLTFPACK_BIN and return. No-op in Docker mode (the container
    image is responsible for its own tooling).
    """
    import time
    import urllib.request
    import zipfile

    if _docker_mode():
        log.info("_install_gltfpack: docker mode, skip (container handles it)")
        return

    if _GLTFPACK_BIN_PATH.is_file() and os.access(_GLTFPACK_BIN_PATH, os.X_OK):
        os.environ["GLTFPACK_BIN"] = str(_GLTFPACK_BIN_PATH)
        log.info("_install_gltfpack: already installed at %s", _GLTFPACK_BIN_PATH)
        return

    log.info("_install_gltfpack: downloading gltfpack %s from %s",
             _GLTFPACK_VERSION, _GLTFPACK_ZIP_URL)
    t0 = time.perf_counter()
    zip_path = Path("/tmp") / f"gltfpack-{_GLTFPACK_VERSION}.zip"

    # GitHub release downloads work with default Python urllib UA, but
    # mirror the Blender installer's defensive Mozilla UA — cheap insurance
    # if Cloudflare ever changes its mind about default Python clients.
    req = urllib.request.Request(
        _GLTFPACK_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=120) as src, open(zip_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    log.info("_install_gltfpack: downloaded in %.1fs (%d KB)",
             time.perf_counter() - t0, zip_path.stat().st_size // 1024)

    _GLTFPACK_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_GLTFPACK_INSTALL_DIR)
    try:
        zip_path.unlink()
    except OSError:
        pass

    if not _GLTFPACK_BIN_PATH.is_file():
        raise RuntimeError(
            f"_install_gltfpack: expected binary at {_GLTFPACK_BIN_PATH} after "
            f"extracting {_GLTFPACK_ZIP_URL} — upstream asset layout may have "
            f"changed; check `gh release view v{_GLTFPACK_VERSION} -R "
            f"zeux/meshoptimizer --json assets`"
        )
    os.chmod(_GLTFPACK_BIN_PATH, 0o755)
    os.environ["GLTFPACK_BIN"] = str(_GLTFPACK_BIN_PATH)
    log.info("_install_gltfpack: installed at %s (%d KB)",
             _GLTFPACK_BIN_PATH, _GLTFPACK_BIN_PATH.stat().st_size // 1024)


def _install_blender_portable() -> None:
    """Install Blender 5.0.1 portable Linux tarball and point BLENDER_BIN at it.

    Replaces Debian Trixie's apt Blender 4.3.2 which is ~75x slower at the
    BVH→FBX retarget step per the Space's [STAGE_TIMINGS] (~90s/job vs ~1.2s
    on Mac native Blender 5.0.1).

    Idempotent: if the binary already exists at the install path, just sets
    BLENDER_BIN and returns. No-op in Docker mode (the container's
    Dockerfile.cpu already pins its own Blender).
    """
    import subprocess as _subproc
    import tarfile
    import time
    import urllib.request

    if _docker_mode():
        log.info("_install_blender_portable: docker mode, skip (container handles Blender)")
        return

    # Idempotency check — survives Space restarts when /home/user/app persists
    if _BLENDER_BIN_PATH.is_file():
        os.environ["BLENDER_BIN"] = str(_BLENDER_BIN_PATH)
        log.info("_install_blender_portable: already installed at %s", _BLENDER_BIN_PATH)
        return

    log.info("_install_blender_portable: downloading Blender %s (~360 MB) from %s",
             _BLENDER_VERSION, _BLENDER_TARBALL_URL)
    t0 = time.perf_counter()
    tarball_path = Path("/tmp") / f"blender-{_BLENDER_VERSION}-linux-x64.tar.xz"
    # download.blender.org is Cloudflare-fronted and 403s on default Python UA.
    req = urllib.request.Request(
        _BLENDER_TARBALL_URL,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as src, open(tarball_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    except Exception as e:
        log.warning("_install_blender_portable: download FAILED (%s) — falling back to apt Blender", e)
        return
    log.info("_install_blender_portable: downloaded in %.1fs (%d MB)",
             time.perf_counter() - t0, tarball_path.stat().st_size // (1024 * 1024))

    t1 = time.perf_counter()
    install_parent = _BLENDER_INSTALL_DIR.parent
    install_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tarball_path, "r:xz") as tar:
            tar.extractall(install_parent)
    except Exception as e:
        log.warning("_install_blender_portable: extract FAILED (%s) — falling back to apt Blender", e)
        return
    try:
        tarball_path.unlink()
    except Exception:
        pass
    log.info("_install_blender_portable: extracted in %.1fs to %s",
             time.perf_counter() - t1, _BLENDER_INSTALL_DIR)

    if not _BLENDER_BIN_PATH.is_file():
        log.warning("_install_blender_portable: binary missing at %s after extract — falling back to apt Blender",
                    _BLENDER_BIN_PATH)
        return

    os.environ["BLENDER_BIN"] = str(_BLENDER_BIN_PATH)

    # Smoke test
    try:
        ver = _subproc.run(
            [str(_BLENDER_BIN_PATH), "--version"],
            capture_output=True, text=True, timeout=30,
        )
        first_line = (ver.stdout or "").splitlines()[0] if ver.stdout else ""
        log.info("_install_blender_portable: BLENDER_BIN=%s | %s",
                 _BLENDER_BIN_PATH, first_line or "<no output>")
    except Exception as e:
        log.warning("_install_blender_portable: version smoke failed (%s) but BLENDER_BIN set anyway", e)


# ---------------------------------------------------------------------------
# Kimodo escape-hatch venv builder
# ---------------------------------------------------------------------------
# Kimodo's NVIDIA stack pins transformers==5.1.0 + py-soma-x + warp-lang etc.
# which fight the MDM/MoMask pins in the orchestrator venv. We isolate it in
# /home/user/app/venvs/kimodo/ and invoke via subprocess from escape_hatch.
# Build runs in a background thread so the Gradio UI comes up promptly and
# MDM/MoMask stay usable while Kimodo warms (~5-10 min cold install: pip deps
# + 16 GB LLaMA-3-8B encoder download).
#
# Sentinels in EXTERNAL_DIR:
#   .kimodo_ready  — venv built, weights cached. Set on success.
#   .kimodo_failed — captured error text. Set on any failure during install.
#
# Per the no-silent-fallback rule: any failure raises loudly and writes
# .kimodo_failed; the runner script + escape_hatch.invoke surface it to the UI.

_KIMODO_VENV_DIR = Path("/home/user/app/venvs/kimodo")
_KIMODO_BUILD_THREAD: "threading.Thread | None" = None
_KIMODO_BUILD_EVENT: "threading.Event | None" = None  # set when ready


def _kimodo_ready_sentinel() -> Path:
    return EXTERNAL_DIR / ".kimodo_ready"


def _kimodo_failed_sentinel() -> Path:
    return EXTERNAL_DIR / ".kimodo_failed"


def _kimodo_text_encoders_dir() -> Path:
    return EXTERNAL_DIR / "text-encoders"


def _kimodo_assets_dir() -> Path:
    return EXTERNAL_DIR / "kimodo-assets"


def _kimodo_src_dir() -> Path:
    return EXTERNAL_DIR / "kimodo-src"


def _clone_kimodo_repos() -> None:
    """Clone Kimodo + kimodo-viser. Idempotent. Skipped via ENABLE_KIMODO=false."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for url, subdir, name in _KIMODO_REPOS:
        target = EXTERNAL_DIR / subdir
        if target.is_dir():
            log.info("kimodo-clone: %s already at %s — skip", name, target)
            continue
        log.info("kimodo-clone (public): %s → %s", name, target)
        _git_clone(url, target)


def _patch_kimodo_requirements() -> None:
    """Strip MotionCorrection from Kimodo's lockfile (C++ post-processor that
    needs cmake + CUDA; we don't use post-processing). Mirrors the Docker
    Dockerfile RUN sed -i '/MotionCorrection/d' line. Idempotent."""
    req_path = _kimodo_src_dir() / "docker_requirements.txt"
    if not req_path.is_file():
        log.warning("_patch_kimodo_requirements: %s not found, skipping", req_path)
        return
    text = req_path.read_text()
    if "MotionCorrection" not in text:
        log.info("_patch_kimodo_requirements: already stripped — skip")
        return
    new_lines = [ln for ln in text.splitlines(keepends=True) if "MotionCorrection" not in ln]
    req_path.write_text("".join(new_lines))
    log.info("_patch_kimodo_requirements: stripped MotionCorrection from %s", req_path)


def _kimodo_venv_python() -> Path:
    return _KIMODO_VENV_DIR / "bin" / "python"


def _ensure_kimodo_venv() -> None:
    """Create the venv with stdlib `venv`. Idempotent."""
    if _kimodo_venv_python().is_file():
        log.info("_ensure_kimodo_venv: venv already at %s — skip", _KIMODO_VENV_DIR)
        return
    log.info("_ensure_kimodo_venv: creating venv at %s", _KIMODO_VENV_DIR)
    _KIMODO_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(_KIMODO_VENV_DIR)],
        check=True, capture_output=True, text=True,
    )


def _kimodo_pip(*args: str, env_extra: dict | None = None, cwd: str | None = None) -> None:
    """Run pip inside the Kimodo venv. Raises on non-zero.

    ``cwd`` is required when the lockfile contains relative editable paths
    like ``-e ./kimodo-viser`` (Kimodo's ``docker_requirements.txt`` does
    this). The upstream Dockerfile changes directory before running pip;
    mirror that.
    """
    cmd = [str(_kimodo_venv_python()), "-m", "pip", *args]
    log.info(
        "kimodo-pip%s: %s",
        f" (cwd={cwd})" if cwd else "",
        " ".join(args[:3]) + (" …" if len(args) > 3 else ""),
    )
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    if proc.returncode != 0:
        # Surface tail of stderr AND stdout (pip writes most of its useful
        # diagnostics to stdout, not stderr) so the .kimodo_failed sentinel
        # captures the actionable detail. 4000 chars each is generous but
        # the sentinel is one-shot per build attempt.
        raise RuntimeError(
            f"kimodo-pip failed (rc={proc.returncode}): {' '.join(args[:8])}\n"
            f"stdout (last 4000 chars): {proc.stdout[-4000:]}\n"
            f"stderr (last 4000 chars): {proc.stderr[-4000:]}"
        )


def _install_kimodo_deps() -> None:
    """Install Kimodo's pinned lockfile into the Kimodo venv."""
    req = _kimodo_src_dir() / "docker_requirements.txt"
    if not req.is_file():
        raise RuntimeError(f"Kimodo docker_requirements.txt missing at {req}")
    # Upgrade pip + install wheel/setuptools first. The Kimodo Docker image
    # is based on nvcr.io/nvidia/pytorch which ships these; our stdlib venv
    # does not, and some pinned deps build sdists that need `wheel` present.
    _kimodo_pip("install", "--upgrade", "pip", "setuptools", "wheel")
    # docker_requirements.txt contains `-e ./kimodo-viser` (relative editable
    # install). pip resolves it against the current working directory, so we
    # must `cd` into kimodo-src — matching containers/kimodo/Dockerfile L30
    # (`RUN cd /workspace/kimodo-src && pip install -r docker_requirements.txt`).
    # The lockfile also triggers py-soma-x's setup.py which conditionally
    # imports MotionCorrection if SKIP_MOTION_CORRECTION_IN_SETUP is unset.
    _kimodo_pip(
        "install", "-r", "docker_requirements.txt",
        env_extra={"SKIP_MOTION_CORRECTION_IN_SETUP": "1"},
        cwd=str(_kimodo_src_dir()),
    )


def _prefetch_llama_encoder(hf_token: str | None) -> None:
    """Pre-fetch LLaMA-3-8B (NousResearch ungated mirror) into the
    text-encoders dir so first inference doesn't pay the ~16 GB download."""
    # Use the ORCHESTRATOR's huggingface_hub (cheaper than spawning the Kimodo
    # venv just to download). Both venvs see the same EXTERNAL_DIR mount.
    from huggingface_hub import snapshot_download

    target = _kimodo_text_encoders_dir() / "NousResearch" / "Meta-Llama-3-8B-Instruct"
    if (target / "config.json").is_file():
        log.info("_prefetch_llama_encoder: already at %s — skip", target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("_prefetch_llama_encoder: downloading ~16 GB to %s", target)
    snapshot_download(
        repo_id="NousResearch/Meta-Llama-3-8B-Instruct",
        local_dir=str(target),
        token=hf_token,  # works with or without token (this repo is ungated)
    )



def _patch_llm2vec_configs() -> None:
    """Rewrite LLM2Vec adapter configs to point at the ungated LLaMA mirror.

    Reuses comfyui-animoflow/containers/kimodo/app.py:_patch_llm2vec_for_ungated_llama
    by importing it as a plain function (no FastAPI app served)."""
    container_kimodo = (
        Path(os.environ.get("COMFYUI_ANIMOFLOW_DIR", str(EXTERNAL_DIR / "comfyui-animoflow")))
        / "containers" / "kimodo"
    )
    app_py = container_kimodo / "app.py"
    if not app_py.is_file():
        raise RuntimeError(
            f"_patch_llm2vec_configs: helper not found at {app_py}. "
            "comfyui-animoflow must be cloned before this step."
        )
    # Plant TEXT_ENCODERS_DIR + HF_HOME so the helper writes to our paths.
    os.environ["TEXT_ENCODERS_DIR"] = str(_kimodo_text_encoders_dir())
    # Import-via-spec so the module's top-level FastAPI construction doesn't
    # require Kimodo deps in the orchestrator venv (the helper itself only
    # needs huggingface_hub which we have).
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_kimodo_helpers", app_py)
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    # The helper imports torch + fastapi at module top. To dodge those when
    # we only want the config rewrite, monkey-load the source and exec only
    # the function we need.
    source = app_py.read_text()
    # Extract just the _patch_llm2vec_for_ungated_llama function by string
    # slicing — small, stable surface. Falls back to full module exec if the
    # markers move (with a clearer error than NameError later).
    start = source.find("def _patch_llm2vec_for_ungated_llama")
    if start < 0:
        raise RuntimeError(
            "_patch_llm2vec_configs: marker 'def _patch_llm2vec_for_ungated_llama' "
            f"not found in {app_py}. Source layout changed; update bootstrap."
        )
    # Find the end of the function — next top-level def or end of file.
    end_markers = ("\ndef ", "\nclass ", "\n# -")
    end = len(source)
    for m in end_markers:
        idx = source.find(m, start + 1)
        if idx > 0:
            end = min(end, idx)
    fn_source = source[start:end]
    # The function also references module-level constants we need to plant.
    preamble = (
        "import json, os\n"
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        f"_ENCODERS_DIR = {str(_kimodo_text_encoders_dir())!r}\n"
        "_UNGATED_LLAMA = 'NousResearch/Meta-Llama-3-8B-Instruct'\n"
        "_LLM2VEC_ADAPTERS = [\n"
        "    'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',\n"
        "    'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised',\n"
        "]\n"
    )
    ns: dict = {}
    exec(preamble + fn_source, ns)  # noqa: S102 — controlled source from our own repo
    ns["_patch_llm2vec_for_ungated_llama"]()
    log.info("_patch_llm2vec_configs: LLM2Vec adapter configs rewired to ungated LLaMA")


def _audit_disk() -> None:
    """Loud warning if /home/user/app is approaching the HF Space quota."""
    try:
        proc = subprocess.run(
            ["du", "-sh", "/home/user/app"],
            capture_output=True, text=True, timeout=30,
        )
        log.info("kimodo-bootstrap: disk usage %s", proc.stdout.strip())
    except Exception as e:  # noqa: BLE001
        log.warning("kimodo-bootstrap: disk audit failed: %s", e)


def _install_kimodo_sync(hf_token: str | None) -> None:
    """Synchronous Kimodo venv build. Called from a background thread by
    _install_kimodo_async. Sets .kimodo_ready on success or .kimodo_failed on
    any error, both consumed by run_inference_kimodo.py + escape_hatch.invoke.
    """
    import time
    t0 = time.perf_counter()
    failed_sentinel = _kimodo_failed_sentinel()
    ready_sentinel = _kimodo_ready_sentinel()
    # Clear stale sentinels from previous attempts.
    for s in (failed_sentinel, ready_sentinel):
        try:
            s.unlink()
        except FileNotFoundError:
            pass
    try:
        log.info("[kimodo-bootstrap] starting")
        _clone_kimodo_repos()
        _patch_kimodo_requirements()
        _ensure_kimodo_venv()
        _install_kimodo_deps()
        _prefetch_llama_encoder(hf_token)
        _patch_llm2vec_configs()
        _audit_disk()
        ready_sentinel.write_text("ok\n")
        if _KIMODO_BUILD_EVENT is not None:
            _KIMODO_BUILD_EVENT.set()
        log.info(
            "[kimodo-bootstrap] ready in %.0fs (venv=%s, ready=%s)",
            time.perf_counter() - t0, _KIMODO_VENV_DIR, ready_sentinel,
        )
    except Exception as exc:  # noqa: BLE001 — capture everything so UI can show it
        import traceback
        msg = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        try:
            failed_sentinel.write_text(msg)
        except Exception:
            pass
        log.exception("[kimodo-bootstrap] FAILED after %.0fs", time.perf_counter() - t0)
        # Don't re-raise: background thread exits, but the sentinel + UI surface
        # the error. escape_hatch.invoke reads .kimodo_failed and refuses to
        # call into a broken venv.


def _install_kimodo_async(hf_token: str | None) -> None:
    """Kick off Kimodo build in a daemon thread. Returns immediately. Calling
    again is a no-op if the build is already running, ready, or failed."""
    import threading as _threading
    global _KIMODO_BUILD_THREAD, _KIMODO_BUILD_EVENT

    if _KIMODO_BUILD_EVENT is None:
        _KIMODO_BUILD_EVENT = _threading.Event()
    # Expose the event on escape_hatch.invoke for the readiness gate.
    # NOTE: escape_hatch/__init__.py does `from .invoke import invoke` which
    # overwrites the submodule reference with the function — even
    # `import escape_hatch.invoke as X` then binds X to the function. Pull
    # the module from sys.modules to bypass the shadowing.
    try:
        import escape_hatch  # noqa: F401 — ensures escape_hatch.invoke is loaded
        import sys as _sys
        _ehi = _sys.modules["escape_hatch.invoke"]
        _ehi._KIMODO_READY = _KIMODO_BUILD_EVENT  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        # escape_hatch import may fail during early bootstrap; the runner
        # also reads the sentinel file as a backup.
        pass

    # Fast paths: nothing to do.
    if _kimodo_ready_sentinel().is_file():
        if not _KIMODO_BUILD_EVENT.is_set():
            _KIMODO_BUILD_EVENT.set()
        log.info("_install_kimodo_async: .kimodo_ready already present — skip")
        return
    if _KIMODO_BUILD_THREAD is not None and _KIMODO_BUILD_THREAD.is_alive():
        log.info("_install_kimodo_async: build already in progress — skip")
        return
    if _kimodo_failed_sentinel().is_file():
        log.warning(
            "_install_kimodo_async: previous build FAILED (see %s). "
            "Delete the sentinel + restart to retry.",
            _kimodo_failed_sentinel(),
        )
        return

    log.info("_install_kimodo_async: spawning build thread")
    _KIMODO_BUILD_THREAD = _threading.Thread(
        target=_install_kimodo_sync,
        args=(hf_token,),
        daemon=True,
        name="kimodo-bootstrap",
    )
    _KIMODO_BUILD_THREAD.start()


def _set_env_for_kimodo() -> None:
    """Plant env vars the runner script + escape_hatch + animoflow-api read
    to find the Kimodo venv, runner, assets, and health endpoint."""
    os.environ.setdefault("KIMODO_VENV_PYTHON", str(_kimodo_venv_python()))
    os.environ.setdefault(
        "KIMODO_RUNNER_SCRIPT",
        str(Path(__file__).resolve().parent / "scripts" / "run_inference_kimodo.py"),
    )
    os.environ.setdefault("KIMODO_SRC_DIR", str(_kimodo_src_dir()))
    os.environ.setdefault("KIMODO_ASSETS_DIR", str(_kimodo_assets_dir()))
    os.environ.setdefault("TEXT_ENCODERS_DIR", str(_kimodo_text_encoders_dir()))
    # Health endpoint: animoflow-api's _MODEL_HEALTH_URLS polls
    # ${KIMODO_ENDPOINT}/health. We expose /__internal/kimodo_health on the
    # same FastAPI app — see app.py.
    port = os.environ.get("PORT", "7860")
    os.environ.setdefault(
        "KIMODO_ENDPOINT", f"http://127.0.0.1:{port}/__internal/kimodo",
    )


def _git_clone(url: str, target: Path, *, depth: int = 1) -> None:
    """Shell out to `git clone --depth N`. Raises CalledProcessError on failure."""
    subprocess.run(
        ["git", "clone", f"--depth={depth}", url, str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


def _strip_remote_auth(target: Path, clean_url: str) -> None:
    """Replace the origin URL with the unauthenticated URL post-clone, so the
    token isn't persisted on disk in `.git/config`."""
    subprocess.run(
        ["git", "-C", str(target), "remote", "set-url", "origin", clean_url],
        check=True,
        capture_output=True,
        text=True,
    )


def _clone_all(gh_token: str | None) -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    for url, subdir, name, private in _GIT_REPOS:
        target = EXTERNAL_DIR / subdir
        if target.is_dir():
            log.info("clone: %s already at %s — skip", name, target)
            continue
        if private:
            if not gh_token:
                raise RuntimeError(
                    f"gh_token env var missing — cannot clone private repo {name}. "
                    "On HF Spaces, configure a Space secret named `gh_token` with "
                    "Read access to AnimoFlow private repos."
                )
            auth_url = url.replace("https://", f"https://x-access-token:{gh_token}@")
            log.info("clone (private): %s → %s", name, target)
            _git_clone(auth_url, target)
            _strip_remote_auth(target, url)
        else:
            log.info("clone (public): %s → %s", name, target)
            _git_clone(url, target)

        # Pin MDM to a known-good commit (HEAD diverged, see MDM_PINNED_COMMIT).
        if subdir == "mdm-codes" and MDM_PINNED_COMMIT:
            _pin_to_commit(target, MDM_PINNED_COMMIT, name)
        # Release pins for AnimoFlow's own repos (no-op while unset).
        if subdir == "comfyui-animoflow" and COMFYUI_PINNED_COMMIT:
            _pin_to_commit(target, COMFYUI_PINNED_COMMIT, name)
        if subdir == "animoflow-api" and API_PINNED_COMMIT:
            _pin_to_commit(target, API_PINNED_COMMIT, name)
        if subdir == "GMR" and GMR_PINNED_COMMIT:
            _pin_to_commit(target, GMR_PINNED_COMMIT, name)

    # Robot retargeting finds its pinned GMR checkout through GMR_HOME.
    os.environ.setdefault("GMR_HOME", str(EXTERNAL_DIR / "GMR"))


def _pin_to_commit(target: Path, commit: str, name: str) -> None:
    """Fetch a specific commit and checkout, so the clone stays at a
    known-good revision regardless of what HEAD points to.

    GitHub doesn't allow fetching arbitrary SHAs by default on shallow
    clones.  Strategy: try shallow fetch first (works on some hosts);
    if that fails, unshallow the repo and then checkout.
    """
    try:
        # Fast path: try fetching the SHA directly (works on GH if
        # uploadpack.allowReachableSHA1InWant is enabled).
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth=1", "origin", commit],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError:
        # Slow path: unshallow so the commit is reachable, then checkout.
        log.info("pin: shallow fetch of %s failed for %s — unshallowing", commit, name)
        try:
            subprocess.run(
                ["git", "-C", str(target), "fetch", "--unshallow"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            # Already unshallow or fetch failed — try a plain fetch
            subprocess.run(
                ["git", "-C", str(target), "fetch", "origin"],
                check=True, capture_output=True, text=True,
            )
    try:
        subprocess.run(
            ["git", "-C", str(target), "checkout", commit],
            check=True, capture_output=True, text=True,
        )
        log.info("pin: %s checked out at %s", name, commit)
    except subprocess.CalledProcessError as exc:
        log.warning("pin: failed to checkout %s at %s: %s", name, commit, exc.stderr)


def _download_checkpoints(hf_token: str | None) -> Path | None:
    """Download MDM checkpoints from HF Hub and return the local dir, or
    None if no token (caller decides whether that's fatal)."""
    ckpt_dir = EXTERNAL_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    target_pt = ckpt_dir / "humanml_enc_512_50steps" / "model000750000.pt"
    if target_pt.exists():
        log.info("checkpoints: already present at %s", ckpt_dir)
        return ckpt_dir

    if not hf_token:
        log.warning(
            "hf_token env var missing — MDM checkpoints not downloaded. "
            "MDM will fall back to placeholder mode. Configure `hf_token` "
            "Space secret with Read access to %s.",
            ANIMOFLOW_CHECKPOINTS_REPO,
        )
        return None

    from huggingface_hub import snapshot_download

    log.info(
        "checkpoints: snapshot_download %s @ %s → %s",
        ANIMOFLOW_CHECKPOINTS_REPO,
        ANIMOFLOW_CHECKPOINTS_REVISION[:8],
        ckpt_dir,
    )
    snapshot_download(
        repo_id=ANIMOFLOW_CHECKPOINTS_REPO,
        revision=ANIMOFLOW_CHECKPOINTS_REVISION,
        repo_type="model",
        local_dir=str(ckpt_dir),
        allow_patterns=[
            "humanml_enc_512_50steps/model000750000.pt",
            "humanml_enc_512_50steps/args.json",
            # MoMask T2M checkpoints — required by inference_momask.py which
            # reads `${CHECKPOINTS_DIR}/t2m/{rvq,t2m,tres,length_estimator}/...`.
            # We override CHECKPOINTS_DIR to `${ckpt_dir}/momask` per-model in
            # registry._load() so this path becomes `${ckpt_dir}/momask/t2m/...`.
            "momask/t2m/**",
            # priorMDM trajectory + timeline heads. The registry's
            # env_overrides_factory plants CHECKPOINT_DIR=${ckpt_dir}/priormdm
            # so inference.py's _discover_checkpoint() looks inside
            # priormdm/{root_horizontal_control_50steps,humanml-encoder-512-50steps}/.
            "priormdm/**",
        ],
        token=hf_token,
    )

    # Copy t2m_mean / t2m_std from the comfyui-animoflow clone — the
    # MDM wrapper expects them in WEIGHTS_DIR alongside the .pt.
    src_dir = EXTERNAL_DIR / "comfyui-animoflow" / "containers" / "mdm"
    for npy in ("t2m_mean.npy", "t2m_std.npy"):
        src = src_dir / npy
        dst = ckpt_dir / npy
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
            log.info("copied %s → %s", src, dst)

    # Mirror for priorMDM: its inference.py reads mean/std from WEIGHTS_DIR
    # which the registry sets to ${ckpt_dir}/priormdm. Same files shipped in
    # the priormdm container — copy them so the path resolves.
    pmdm_src_dir = EXTERNAL_DIR / "comfyui-animoflow" / "containers" / "priormdm"
    pmdm_dst_dir = ckpt_dir / "priormdm"
    if pmdm_src_dir.exists():
        pmdm_dst_dir.mkdir(parents=True, exist_ok=True)
        for npy in ("t2m_mean.npy", "t2m_std.npy"):
            src = pmdm_src_dir / npy
            dst = pmdm_dst_dir / npy
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)
                log.info("copied %s → %s", src, dst)

    return ckpt_dir


def _download_private_characters(hf_token: str | None) -> None:
    """Fetch the Mixamo character FBXs from the private character-assets
    repo into the cloned comfyui-animoflow characters dir.

    The public comfyui-animoflow repo only carries download instructions
    (Mixamo terms forbid redistribution) plus the small bone_map.json
    sidecars; the hosted demo gets the actual FBXs from
    AnimoFlow/character-assets `characters/**` (split out of the
    checkpoints repo 2026-07-06 to isolate Adobe-licensed content).
    Idempotent: skips files already present. No token → loud warning,
    characters absent → /v1/characters only lists whatever the public
    clone carries (nothing, since comfyui-animoflow@7a91fa3 untracked its
    dev-convenience FBXs). NOTE: a token that lacks read scope on
    the character-assets repo raises RepositoryNotFoundError here (HF
    404s private repos it can't see) and crashes startup — loud on
    purpose; extend the Space `hf_token` scope, don't wrap this in
    try/except.
    """
    chars_dir = EXTERNAL_DIR / "comfyui-animoflow" / "characters"
    if not chars_dir.is_dir():
        log.warning("characters: %s missing — clone step failed?", chars_dir)
        return
    if not hf_token:
        log.warning(
            "hf_token env var missing — private characters not downloaded; "
            "the character dropdown will only show the public-clone rigs."
        )
        return

    from huggingface_hub import snapshot_download

    log.info(
        "characters: snapshot_download %s @ %s characters/** → %s",
        ANIMOFLOW_CHARACTER_ASSETS_REPO,
        ANIMOFLOW_CHARACTER_ASSETS_REVISION[:8],
        chars_dir,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=ANIMOFLOW_CHARACTER_ASSETS_REPO,
            revision=ANIMOFLOW_CHARACTER_ASSETS_REVISION,
            repo_type="model",
            token=hf_token,
            local_dir=tmp,
            allow_patterns=["characters/**"],
        )
        src_dir = Path(tmp) / "characters"
        if not src_dir.is_dir():
            # The silent-glob failure mode: download "succeeded" but the
            # pattern matched nothing at this revision. Loud, per the
            # no-silent-fallback rule.
            log.warning(
                "characters: %s @ %s has NO characters/ folder — pattern "
                "globbed nothing; the character dropdown will be missing "
                "the private rigs. Check the repo/revision pin.",
                ANIMOFLOW_CHARACTER_ASSETS_REPO,
                ANIMOFLOW_CHARACTER_ASSETS_REVISION[:8],
            )
            return
        n = 0
        for f in sorted(src_dir.iterdir()):
            if not f.is_file():
                continue
            dst = chars_dir / f.name
            if dst.exists() and dst.stat().st_size == f.stat().st_size:
                continue
            shutil.copy2(f, dst)
            n += 1
        log.info("characters: installed %d file(s) into %s", n, chars_dir)


def _set_env_for_external() -> None:
    """Plant env vars that the rest of the orchestrator code reads, so
    pipeline_hf, models.registry, etc. find the cloned paths without
    knowing whether we're in Docker or Gradio mode."""
    os.environ.setdefault(
        "COMFYUI_ANIMOFLOW_DIR", str(EXTERNAL_DIR / "comfyui-animoflow")
    )
    os.environ.setdefault(
        "COMFYUI_ANIMOFLOW_NODES_DIR",
        str(EXTERNAL_DIR / "comfyui-animoflow" / "nodes"),
    )
    os.environ.setdefault(
        "ANIMOFLOW_API_API_DIR", str(EXTERNAL_DIR / "animoflow-api" / "api")
    )
    os.environ.setdefault("MDM_PATH", str(EXTERNAL_DIR / "mdm-codes"))
    os.environ.setdefault("MOMASK_PATH", str(EXTERNAL_DIR / "momask-codes"))
    os.environ.setdefault("PRIORMDM_PATH", str(EXTERNAL_DIR / "priormdm-codes"))
    os.environ.setdefault(
        "CHARACTERS_DIR",
        str(EXTERNAL_DIR / "comfyui-animoflow" / "characters"),
    )
    os.environ.setdefault("WEB_DIR", "/nonexistent")
    os.environ.setdefault("WEIGHTS_DIR", str(EXTERNAL_DIR / "checkpoints"))
    os.environ.setdefault("CHECKPOINTS_DIR", str(EXTERNAL_DIR / "checkpoints"))


def _patch_mdm_model_util():
    """Ensure MDM's utils/model_util.py has a ``load_saved_model`` function.

    The MDM inference wrapper (comfyui-animoflow/containers/mdm/inference.py)
    imports ``load_saved_model`` from ``utils.model_util``, but the upstream
    MDM repo only exposes ``load_model_wo_clip``.  Without this shim the
    import fails silently (caught by a broad except), and the model falls
    back to a hardcoded walk-cycle placeholder — ignoring the text prompt
    entirely.

    This patch adds a thin ``load_saved_model(model, path, **kw)`` wrapper
    that loads the state dict from *path* and delegates to
    ``load_model_wo_clip``.  Idempotent — skips if the function already
    exists in the file.
    """
    mdm_path = EXTERNAL_DIR / "mdm-codes"
    model_util = mdm_path / "utils" / "model_util.py"
    if not model_util.is_file():
        log.info("_patch_mdm_model_util: %s not found, skipping", model_util)
        return

    text = model_util.read_text()
    if "def load_saved_model" in text:
        log.info("_patch_mdm_model_util: load_saved_model already present — skip")
        return

    shim = '''

def load_saved_model(model, model_path, use_avg=False, **kwargs):
    """Compatibility shim: load checkpoint and delegate to load_model_wo_clip.

    The checkpoint stores only the non-CLIP weights (CLIP weights are
    stripped at training-time save). ``load_model_wo_clip`` loads them
    with ``strict=False`` so the freshly-downloaded CLIP weights in the
    model are preserved.
    """
    import torch as _torch
    state_dict = _torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
'''
    model_util.write_text(text + shim)
    log.info("_patch_mdm_model_util: injected load_saved_model shim into %s", model_util)


def _patch_priormdm_source() -> None:
    """Mirror comfyui-animoflow/containers/priormdm/Dockerfile lines 46-53.

    priorMDM's upstream source still uses numpy 1.x aliases (np.float, np.int,
    np.bool) removed in numpy 1.24+, and asserts dataset size > 1 which our
    single-sample inference path violates. The container's Dockerfile sed-fixes
    these at image build time; we mirror them here because we run priorMDM
    inside the orchestrator venv (numpy 2.x), not the container's 1.23 pin.

    Idempotent via a sentinel file. Safe to re-run after each bootstrap.
    """
    import re

    src = EXTERNAL_DIR / "priormdm-codes"
    if not src.is_dir():
        log.info("_patch_priormdm_source: %s not found, skipping", src)
        return
    sentinel = src / ".animoflow_patched"
    if sentinel.is_file():
        log.info("_patch_priormdm_source: already patched (%s) — skip", sentinel)
        return

    np_alias_subs = [
        (re.compile(r"\bnp\.float\b"), "float"),
        (re.compile(r"\bnp\.int\b"),   "int"),
        (re.compile(r"\bnp\.bool\b"),  "bool"),
    ]
    touched = 0
    for py in src.rglob("*.py"):
        text = py.read_text()
        new = text
        for pat, repl in np_alias_subs:
            new = pat.sub(repl, new)
        if new != text:
            py.write_text(new)
            touched += 1

    dset = src / "data_loaders" / "humanml" / "data" / "dataset.py"
    if dset.is_file():
        t = dset.read_text()
        t2 = t.replace(
            "assert len(self.t2m_dataset) > 1",
            "assert len(self.t2m_dataset) >= 1",
        )
        if t2 != t:
            dset.write_text(t2)

    sentinel.write_text("ok\n")
    log.info(
        "_patch_priormdm_source: patched %d .py files + dataset assert in %s",
        touched, src,
    )


def _patch_blender_bvh_addon():
    """Fix Blender's io_anim_bvh addon: 'rU' mode removed in Python 3.12.

    Strategy: create a full copy of Blender's scripts directory at
    /home/user/app/blender_scripts_patched, fix the BVH addon, and set
    BLENDER_SYSTEM_SCRIPTS to redirect Blender there.
    """
    import shutil

    sys_scripts = Path("/usr/share/blender/scripts")
    if not sys_scripts.is_dir():
        log.info("_patch_blender_bvh_addon: /usr/share/blender/scripts not found, skipping")
        return

    target_file = sys_scripts / "addons/io_anim_bvh/import_bvh.py"
    if not target_file.is_file():
        log.info("_patch_blender_bvh_addon: %s not found, skipping", target_file)
        return

    # Check if system file has the bug
    sys_text = target_file.read_text()
    if "'rU'" not in sys_text and '"rU"' not in sys_text:
        log.info("_patch_blender_bvh_addon: no 'rU' found in system file, already clean")
        return

    # Try direct fix first (might work if container allows it)
    try:
        fixed = sys_text.replace("'rU'", "'r'").replace('"rU"', '"r"')
        target_file.write_text(fixed)
        # Verify
        if "'rU'" not in target_file.read_text():
            log.info("_patch_blender_bvh_addon: patched system file directly at %s", target_file)
            return
    except PermissionError:
        log.info("_patch_blender_bvh_addon: no write access to system file, using BLENDER_SYSTEM_SCRIPTS override")

    # Fallback: copy entire scripts dir and set BLENDER_SYSTEM_SCRIPTS
    patched_dir = Path("/home/user/app/blender_scripts_patched")
    if patched_dir.is_dir():
        log.info("_patch_blender_bvh_addon: patched dir already exists at %s", patched_dir)
        os.environ["BLENDER_SYSTEM_SCRIPTS"] = str(patched_dir)
        return

    shutil.copytree(sys_scripts, patched_dir)
    patch_target = patched_dir / "addons/io_anim_bvh/import_bvh.py"
    text = patch_target.read_text()
    text = text.replace("'rU'", "'r'").replace('"rU"', '"r"')
    patch_target.write_text(text)
    os.environ["BLENDER_SYSTEM_SCRIPTS"] = str(patched_dir)
    log.info("_patch_blender_bvh_addon: created patched scripts at %s, set BLENDER_SYSTEM_SCRIPTS", patched_dir)


def _ensure_blender_numpy() -> None:
    """Install numpy so Blender's glTF2 addon can export GLB.

    On HF Gradio Spaces (Debian Trixie), apt Blender links against
    python3.11 but the Space's default ``pip3`` targets python3.12+.
    Running ``pip3 install numpy`` installs for the wrong interpreter.

    Strategy:
      1. Probe Blender for ``sys.version_info`` to know the Python version.
      2. Try ``python<ver> -m pip install numpy`` (works on tarball Blender).
      3. If that fails (apt Blender has no pip for 3.11), install numpy
         to a ``--target`` directory via the system pip, then set
         ``PYTHONPATH`` so Blender's Python can find it.
      4. Verify by actually importing numpy inside Blender.

    Idempotent: probes numpy inside Blender first (using sys.exit to
    get a reliable return code — Blender returns 0 on --python-expr
    exceptions otherwise).
    """
    import shutil as _shutil
    import subprocess as _subproc
    from pathlib import Path as _Path

    blender = os.environ.get("BLENDER_BIN", "").strip() or _shutil.which("blender")
    if not blender or not _Path(blender).exists():
        log.warning("Blender not found — skipping numpy install for GLB export")
        return

    # --- Step 1: probe Blender for numpy + Python version ---
    # Use sys.exit(code) to get a reliable return code from Blender.
    probe_script = (
        "import sys\n"
        "try:\n"
        "    import numpy; print('NUMPY_OK', numpy.__version__); sys.exit(0)\n"
        "except ImportError:\n"
        "    print('NUMPY_MISSING')\n"
        "    print('PYVER', f'{sys.version_info.major}.{sys.version_info.minor}')\n"
        "    sys.exit(42)\n"
    )
    probe = _subproc.run(
        [blender, "--background", "--python-expr", probe_script],
        capture_output=True, text=True, timeout=30,
    )
    # Parse stdout for our markers
    probe_data: dict[str, str] = {}
    for line in probe.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) >= 1 and parts[0] in ("NUMPY_OK", "NUMPY_MISSING", "PYVER"):
            probe_data[parts[0]] = parts[1].strip() if len(parts) == 2 else ""

    if "NUMPY_OK" in probe_data:
        log.info("Blender numpy: already present (numpy %s, skip)",
                 probe_data["NUMPY_OK"])
        return

    py_ver = probe_data.get("PYVER", "")
    log.info("Blender Python version: %s, numpy missing — installing", py_ver)

    # --- Step 2: get pip working for Blender's Python, then install numpy ---
    blender_python = _shutil.which(f"python{py_ver}") if py_ver else None
    if not blender_python and py_ver:
        candidate = f"/usr/bin/python{py_ver}"
        if _Path(candidate).is_file():
            blender_python = candidate
    if not blender_python:
        log.warning("Cannot find python%s — skipping numpy install", py_ver)
        return

    log.info("Blender's Python: %s", blender_python)
    installed = False

    # Strategy A: try ensurepip + pip install (works on tarball Blender)
    _subproc.run(
        [blender_python, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True, timeout=120,
    )
    direct = _subproc.run(
        [blender_python, "-m", "pip", "install", "--no-cache-dir",
         "--break-system-packages", "numpy"],
        capture_output=True, text=True, timeout=300,
    )
    if direct.returncode == 0:
        log.info("Installed numpy via %s -m pip", blender_python)
        installed = True

    # Strategy B: bootstrap pip for this Python via get-pip.py, then install
    if not installed:
        log.info("%s has no pip — bootstrapping via get-pip.py", blender_python)
        import urllib.request
        get_pip = "/tmp/get-pip.py"
        try:
            urllib.request.urlretrieve(
                "https://bootstrap.pypa.io/get-pip.py", get_pip,
            )
        except Exception as e:
            log.warning("Failed to download get-pip.py: %s", e)
            # Continue to Strategy C
        else:
            bootstrap_pip = _subproc.run(
                [blender_python, get_pip, "--break-system-packages"],
                capture_output=True, text=True, timeout=120,
            )
            if bootstrap_pip.returncode == 0:
                install_np = _subproc.run(
                    [blender_python, "-m", "pip", "install", "--no-cache-dir",
                     "--break-system-packages", "numpy"],
                    capture_output=True, text=True, timeout=300,
                )
                if install_np.returncode == 0:
                    log.info("Installed numpy via get-pip.py + %s -m pip", blender_python)
                    installed = True
                else:
                    log.info("pip install after get-pip failed (rc=%s): %s",
                             install_np.returncode, (install_np.stderr or "")[-300:])
            else:
                log.info("get-pip.py failed (rc=%s): %s",
                         bootstrap_pip.returncode, (bootstrap_pip.stderr or "")[-300:])

    # Strategy C: pip download wheels for the correct python version, install with --target
    if not installed:
        log.info("Trying pip download with --python-version %s ...", py_ver)
        pip3 = _shutil.which("pip3") or _shutil.which("pip")
        if pip3:
            dl_dir = _Path("/tmp/numpy_wheels")
            dl_dir.mkdir(parents=True, exist_ok=True)
            target_dir = _Path("/home/user/app/blender_numpy_packages")
            target_dir.mkdir(parents=True, exist_ok=True)

            dl = _subproc.run(
                [pip3, "download", "--no-cache-dir",
                 "--python-version", py_ver,
                 "--abi", f"cp{py_ver.replace('.', '')}",
                 "--platform", "manylinux2014_x86_64",
                 "--only-binary=:all:",
                 "-d", str(dl_dir), "numpy"],
                capture_output=True, text=True, timeout=120,
            )
            if dl.returncode == 0:
                wheels = list(dl_dir.glob("*.whl"))
                if wheels:
                    inst = _subproc.run(
                        [pip3, "install", "--no-deps", "--no-cache-dir",
                         "--break-system-packages",
                         "--target", str(target_dir)] + [str(w) for w in wheels],
                        capture_output=True, text=True, timeout=120,
                    )
                    if inst.returncode == 0:
                        existing = os.environ.get("PYTHONPATH", "")
                        os.environ["PYTHONPATH"] = (
                            f"{target_dir}:{existing}" if existing else str(target_dir)
                        )
                        log.info("Installed numpy wheels to %s, PYTHONPATH=%s",
                                 target_dir, os.environ["PYTHONPATH"])
                        installed = True

    if not installed:
        log.warning("All numpy install strategies failed for python%s", py_ver)
        return

    # --- Step 4: verify inside Blender ---
    verify_script = (
        "import sys\n"
        "try:\n"
        "    import numpy; print('VERIFY_OK', numpy.__version__); sys.exit(0)\n"
        "except ImportError as e:\n"
        "    print('VERIFY_FAIL', e); sys.exit(1)\n"
    )
    confirm = _subproc.run(
        [blender, "--background", "--python-expr", verify_script],
        capture_output=True, text=True, timeout=30,
    )
    for line in confirm.stdout.splitlines():
        if line.startswith("VERIFY_OK"):
            log.info("Blender numpy install verified: %s", line)
            return
        if line.startswith("VERIFY_FAIL"):
            log.warning("Blender numpy verification failed: %s", line)
            return
    log.warning("Blender numpy verification inconclusive (rc=%s, stderr=%s)",
                confirm.returncode, (confirm.stderr or "")[-300:])


_CONTAINER_SPECS_LOGGED = False


def _log_container_specs() -> None:
    """Log CPU/RAM/Blender specs once per process for ZeroGPU diagnostics."""
    global _CONTAINER_SPECS_LOGGED
    if _CONTAINER_SPECS_LOGGED:
        return
    _CONTAINER_SPECS_LOGGED = True

    import multiprocessing
    import time

    specs: dict[str, str] = {}

    # CPU counts
    specs["mp_cpu_count"] = str(multiprocessing.cpu_count())
    specs["os_cpu_count"] = str(os.cpu_count())

    # RAM
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        specs["total_ram_gb"] = f"{page * pages / 1024**3:.2f}"
    except (ValueError, OSError):
        specs["total_ram_gb"] = "unknown"

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    specs["avail_ram_gb"] = f"{kb / 1024**2:.2f}"
                    break
            else:
                specs["avail_ram_gb"] = "unknown"
    except OSError:
        specs["avail_ram_gb"] = "unknown"

    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    specs["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            else:
                specs["cpu_model"] = "unknown"
    except OSError:
        specs["cpu_model"] = "unknown"

    # Blender cold-start time
    import shutil as _shutil
    blender = os.environ.get("BLENDER_BIN", "").strip() or _shutil.which("blender")
    if blender and Path(blender).exists():
        try:
            t0 = time.monotonic()
            subprocess.run(
                [blender, "--version"],
                capture_output=True, text=True, timeout=60,
            )
            specs["blender_startup_s"] = f"{time.monotonic() - t0:.3f}"
        except Exception as e:
            specs["blender_startup_s"] = f"error:{e}"
    else:
        specs["blender_startup_s"] = "not_found"

    # cgroup limits (ZeroGPU containers use cgroup v2)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            specs["cgroup_cpu_max"] = f.read().strip()
    except Exception:
        specs["cgroup_cpu_max"] = "unavailable"
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
            if raw == "max":
                specs["cgroup_mem_max"] = "max (unlimited)"
            else:
                specs["cgroup_mem_max"] = f"{int(raw) / 1024**3:.2f}GB"
    except Exception:
        specs["cgroup_mem_max"] = "unavailable"

    parts = " | ".join(f"{k}={v}" for k, v in specs.items())
    log.info("[CONTAINER_SPECS] %s", parts)


def bootstrap_external_repos() -> None:
    """Idempotent. Clone external repos + download checkpoints if not
    already present. No-op in Docker mode."""
    if _docker_mode():
        log.info("Docker mode detected (/opt/* paths present) — skip bootstrap")
        return

    log.info("Bootstrap mode (HF Gradio or fresh OSS-local) — base=%s", EXTERNAL_DIR)

    gh_token = os.environ.get("gh_token") or os.environ.get("GH_TOKEN")
    hf_token = os.environ.get("hf_token") or os.environ.get("HF_TOKEN")

    _install_blender_portable()
    _install_gltfpack()
    _clone_all(gh_token)
    _download_checkpoints(hf_token)
    _download_private_characters(hf_token)
    _set_env_for_external()
    _patch_mdm_model_util()
    _patch_priormdm_source()
    _patch_blender_bvh_addon()
    _ensure_blender_numpy()
    # Kimodo escape-hatch venv build runs in a background thread so the Gradio
    # UI comes up promptly with MDM/MoMask. ENABLE_KIMODO=false disables (kill
    # switch). Per the no-silent-fallback rule: any failure writes the
    # .kimodo_failed sentinel and is surfaced to the UI; we never silently
    # degrade.
    if os.environ.get("ENABLE_KIMODO", "true").strip().lower() != "false":
        _set_env_for_kimodo()
        _install_kimodo_async(hf_token)
    else:
        log.info("ENABLE_KIMODO=false — Kimodo bootstrap skipped")
    _log_container_specs()
    log.info("Bootstrap complete: external repos at %s", EXTERNAL_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bootstrap_external_repos()
