---
title: AnimoFlow Demo
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.31.0
hardware: zero-a10g
python_version: "3.11"
pinned: false
license: other
license_name: animoflow-community-license-1.0.0
license_link: LICENSE
short_description: Text → Motion → FBX/GLB. MDM diffusion + Blender retarget.
hf_oauth: true
hf_oauth_expiration_minutes: 480
preload_from_hub:
  - "Qwen/Qwen2.5-1.5B-Instruct"
  - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
---

# animoflow-app

Deployment wrapper for [AnimoFlow](https://github.com/AnimoFlow). Same image runs as a
[Hugging Face Space ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu)
demo and as a one-command OSS local server.

The app is a thin layer on top of three sibling repos. None of them are forked
or modified — each is `git clone`d at Docker build time, pinned by SHA:

- [`comfyui-animoflow`](https://github.com/AnimoFlow/comfyui-animoflow) — model wrappers, IK + retarget code, Blender scripts
- [`animoflow-api`](https://github.com/AnimoFlow/animoflow-api) — `/v1/jobs` async FastAPI, validation models, task catalog
- upstream model repos — [MDM](https://github.com/GuyTevet/motion-diffusion-model), [momask-codes](https://github.com/EricGuo5513/momask-codes), more added per iteration

More documentation: the [AnimoFlow user guide](https://animoflow-alpha.pages.dev/guide/) covers all the ways to use the platform; all repositories are under [github.com/AnimoFlow](https://github.com/AnimoFlow).

Net new code in this repo: the `Dockerfile`, `entrypoint.sh`, `app.py` (mounts
Gradio on the imported animoflow-api FastAPI), `ui.py` (Gradio Blocks), and
`pipeline_hf.py` (replaces ComfyUI orchestration with in-process pipeline so
ZeroGPU's `@spaces.GPU` works).

## Architecture

Short version:

- **Single venv on HF.** Orchestrator + warm MDM-family models loaded at module
  level. ZeroGPU forks the orchestrator process at each `@spaces.GPU` call; the
  fork inherits warm models via copy-on-write, runs inference on real H200
  for ~10–15 s, dies, releases GPU.
- **GPU only on inference.** Resample, IK (momask `Joint2BVHConvertor`), foot/
  outlier fix, Blender retarget → FBX, Blender FBX → GLB all run CPU-side in the
  persistent orchestrator process — no GPU billing.
- **Escape hatch for outlier models.** Models whose deps don't merge with the
  orchestrator venv (Kimodo's NVIDIA stack, future Tier-2 models) live in
  `/opt/venvs/comfy-X/` and are subprocess-called from inside `@spaces.GPU`
  only when their model is selected. Cold-start tax (~15 s) accepted in
  exchange for full dep isolation.

## API

Same `/v1/*` surface as the rest of AnimoFlow:

```
POST /v1/jobs            { input: {type, prompt}, model, character, duration, seed }
GET  /v1/jobs/{id}       { status, progress, stage, download_url, error, ... }
GET  /v1/files/{name}    binary FBX or GLB
GET  /v1/tasks           { tasks: [...] }
GET  /v1/models          { stages: [...] }
GET  /v1/characters      { characters: [...] }
GET  /v1/health          { status: "ok" }
```

Anything Gradio can do, you can also do via `curl`, `gradio_client`,
`@gradio/client` JS, or the Blender add-on.

## Run locally (no HF account, no Docker)

Requires Python 3.11, Blender 4.2 LTS on PATH (or `BLENDER_BIN` set), and
the sibling repos cloned alongside.

```bash
git clone https://github.com/AnimoFlow/comfyui-animoflow ../comfyui-animoflow
git clone https://github.com/AnimoFlow/animoflow-api      ../animoflow-api
git clone https://github.com/GuyTevet/motion-diffusion-model ../mdm-codes
git clone https://github.com/EricGuo5513/momask-codes        ../momask-codes

cd animoflow-app
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Point at your local clones
cp .env.example .env
$EDITOR .env  # set CHECKPOINTS_DIR, CHARACTERS_DIR, and ANIMOFLOW_API_API_DIR (path to ../animoflow-api/api)

# Download MDM checkpoints
./scripts/download_weights.sh

.venv/bin/python app.py
# → http://localhost:7860
```

## Run locally (Docker)

The Dockerfile uses two [BuildKit secrets](https://docs.docker.com/build/building/secrets/), neither
of which lands in any image layer:

- `gh_token` — read access to `AnimoFlow/{comfyui-animoflow,animoflow-api}` (only needed while the repos are private)
- `hf_token` — read access to BOTH `AnimoFlow/animoflow-checkpoints` (private HF Hub model repo where MDM weights live) AND `AnimoFlow/character-assets` (private repo with the Mixamo character FBXs, split out of the checkpoints repo 2026-07-06)

```bash
# 1. Drop tokens into files (gh + hf CLIs extract them if you're logged in)
gh auth token                          > /tmp/gh_token
huggingface-cli auth token             > /tmp/hf_token   # or: cat ~/.cache/huggingface/token > /tmp/hf_token

# 2. Build with both secrets mounted
DOCKER_BUILDKIT=1 docker build \
    --secret id=gh_token,src=/tmp/gh_token \
    --secret id=hf_token,src=/tmp/hf_token \
    -t animoflow/app .

# 3. Clean up
rm /tmp/gh_token /tmp/hf_token

# 4. Run — checkpoints are already baked in, no extra mounts needed
docker run --rm -p 7860:7860 animoflow/app
# → http://localhost:7860
```

The Dockerfile is arch-aware (Blender 4.2 LTS for both `linux/amd64` and
`linux/arm64`). On Apple Silicon Macs, native arm64 builds work — but if
you want byte-parity with HF's amd64 ZeroGPU hosts, force the platform:

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
    --secret id=gh_token,src=/tmp/gh_token \
    --secret id=hf_token,src=/tmp/hf_token \
    -t animoflow/app .
```

This is the same image deployed to HF Spaces. `@spaces.GPU` no-ops when
`spaces` isn't on the PYTHONPATH (or when `SPACE_ID` isn't set), so locally
you get whatever GPU the host has — or CPU fallback.

## Run on HF Spaces

This repo's `README.md` has the Space card YAML at the top. To deploy:

1. Create a Space at <https://huggingface.co/new-space> — pick the `Gradio`
   SDK, `ZeroGPU` hardware, and the visibility you want.
2. **Configure two Space secrets** under Settings → Variables and secrets:
   - `gh_token` — fine-grained GitHub PAT with read access to `AnimoFlow/animoflow-api` and `AnimoFlow/comfyui-animoflow`
   - `hf_token` — HF token with read access to BOTH `AnimoFlow/animoflow-checkpoints` AND `AnimoFlow/character-assets` (fine-grained tokens must list both repos explicitly — a token missing the character-assets scope crashes bootstrap with RepositoryNotFoundError)

   HF makes Space secrets available to BuildKit's `--mount=type=secret`
   automatically, so the Dockerfile picks them up without further config.
3. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/AnimoFlow/animoflow-demo
   git push space main
   ```

HF builds the Dockerfile (cloning the private repos via `gh_token`,
baking the MDM checkpoints via `hf_token`), allocates ZeroGPU on each
request, and serves the Gradio UI at the Space URL.

**Note on file downloads:** On HF Spaces, generated FBX/GLB files are
served through Gradio's UI (the `gr.Model3D` widget renders them directly).
External clients should use `gradio_client` to fetch results. The `/v1/files`
REST endpoint works locally and in OSS Docker mode but is mangled by
Gradio's reverse proxy on HF Spaces — use the Gradio client API instead.

## Repo layout

```
animoflow-app/
├── Dockerfile                    # multi-arg build, Blender pinned tarball
├── entrypoint.sh                 # downloads weights → starts uvicorn
├── requirements.txt              # orchestrator + MDM family
├── app.py                        # mounts Gradio on imported animoflow-api FastAPI
├── ui.py                         # Gradio Blocks
├── pipeline_hf.py                # GPU-decorated inference + CPU pipeline
├── bootstrap.py                  # build/startup: repos, weights, characters, Blender
├── spaces_compat.py              # @spaces.GPU graceful fallback
├── animoflow_models/
│   ├── __init__.py
│   └── registry.py               # warm-loaded MDM family at module level
├── escape_hatch/
│   ├── __init__.py
│   └── invoke.py                 # subprocess into /opt/venvs/comfy-X
├── scripts/
│   ├── download_weights.sh       # idempotent weight download
│   └── run_inference_kimodo.py   # escape-hatch subprocess runner (kimodo)
├── tests/
│   ├── conftest.py
│   ├── test_app_imports.py       # FastAPI mounts, no model deps required
│   ├── test_pipeline_hf.py       # mocked GPU + real CPU pipeline
│   └── test_spaces_compat.py     # decorator behavior
├── .env.example
├── .gitignore
├── .dockerignore
└── LICENSE                       # AnimoFlow Community License 1.0.0
```

## License

**AnimoFlow Community License 1.0.0** (see [`LICENSE`](LICENSE)) — the same
source-available, perpetual threshold license as `animoflow-api`: free
forever for individuals, non-profits, academia, and organizations under
$5M USD (2026) total finances & 100 people; commercial license above
(guy@animoflow.ai). At runtime this wrapper fetches components that carry
their own licenses — `comfyui-animoflow` is AGPL-3.0, `animoflow-api` is
AFCL; see [AnimoFlow/legal](https://github.com/AnimoFlow/legal) for the
full per-repo map.

### Third-party model licenses

The Space loads inference code and weights from upstream projects with their
own terms. Bundling them does not change this wrapper's AnimoFlow
Community License declaration, but downstream users should observe each
model's license:

- **MDM** (Motion Diffusion Model) — code: [MIT](https://github.com/GuyTevet/motion-diffusion-model);
  HumanML3D-trained checkpoint published by the authors.
- **MoMask** — code: [MIT](https://github.com/EricGuo5513/momask-codes);
  HumanML3D-trained checkpoint with the dataset's research-use terms.
- **Kimodo (NVIDIA)** — code: Apache-2.0
  ([nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo)). Weights:
  `Kimodo-SOMA-RP-v1` ships under the NVIDIA Open Model License
  (commercial use permitted). The Space deliberately uses ONLY the
  commercial-OK variants; the research-only `Kimodo-SMPLX-RP-v1` is not loaded.
- **LLaMA-3-8B-Instruct** (text encoder used by Kimodo) — pulled from the
  ungated `NousResearch/Meta-Llama-3-8B-Instruct` mirror. Use is governed by
  Meta's [LLaMA-3 Community License](https://llama.meta.com/llama3/license/).
- **SOMA-X** ([NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X)) — used by
  Kimodo for SOMA→SMPL mesh skinning. See the upstream repo for terms.
