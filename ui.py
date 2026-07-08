"""
Gradio Blocks UI for the AnimoFlow HF Space.

The UI is a thin client over the FastAPI /v1/jobs endpoint that the
animoflow-api app already implements. We POST a job, poll for status,
then download the resulting FBX/GLB.

Same surface that the animoflow.ai webpage and Blender add-on consume.
The UI itself uses no privileged path — anything you can do here, you
can do via curl.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

log = logging.getLogger(__name__)

# Usage analytics (animoflow-api module; on sys.path via app.py step 1).
# No-op unless USAGE_LOG_* env vars are set — see usage_log.py docstring.
try:
    import usage_log
except ImportError:  # bare ui.py import outside the Space wiring (tests)
    usage_log = None
    log.warning("usage_log not importable — usage analytics disabled")

# Where animoflow-api's /v1 lives. In-process the FastAPI app is mounted at
# the same root, so http://localhost:<port> works.
_PORT = int(os.environ.get("PORT", "7860"))
_API_BASE = os.environ.get("ANIMOFLOW_API_BASE", f"http://127.0.0.1:{_PORT}")
_API_KEY = os.environ.get("ANIMOFLOW_API_KEY", "dev")

_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/animoflow-output"))


def _post_job_body(body: dict[str, Any], headers: dict[str, str] | None = None) -> str:
    """POST /v1/jobs with a pre-built body → job_id. Raises on non-2xx.

    ``headers`` lets callers add attribution (x-usage-actor) on the internal
    loopback hop without touching the auth header.
    """
    r = httpx.post(
        f"{_API_BASE}/v1/jobs",
        json=body,
        headers={"Authorization": f"Bearer {_API_KEY}", **(headers or {})},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()["job_id"]


def _submit_error(exc: httpx.HTTPStatusError) -> tuple[str, str]:
    """(user_message, error_code) for a non-2xx POST /v1/jobs.

    Prefer the API's own classified message (error_classify populates it on
    clean failures); otherwise fall back to a neutral per-status message.
    Never interpolate the raw exception — it carries internal URLs
    (e.g. http://127.0.0.1:7860/v1/jobs) that must not reach users.
    """
    try:
        err = exc.response.json().get("error", {})
    except Exception:  # noqa: BLE001 — non-JSON body (e.g. bare 500 page)
        err = {}
    if not isinstance(err, dict):
        # e.g. slowapi's stock 429 body was {"error": "<string>"} — main.py
        # now owns that envelope, but stay robust to any legacy shape.
        err = {}
    if err.get("message"):
        return err["message"], str(err.get("code") or exc.response.status_code)
    if exc.response.status_code == 429:
        return (
            "Too many requests — please wait a minute and try again.",
            "429",
        )
    return (
        "The server had a problem starting this job — please try again.",
        str(exc.response.status_code),
    )


def _post_job(prompt: str, model: str, character: str, duration: float, seed: int,
              headers: dict[str, str] | None = None) -> str:
    """POST /v1/jobs → job_id. Text-only convenience wrapper for the
    legacy in-Space `_generate` handler. Browser callers use
    `_generate_full` instead, which forwards the full GenerateRequest
    body verbatim and so supports trajectory / waypoints / timeline."""
    payload: dict[str, Any] = {
        "input": {"type": "text", "prompt": prompt},
        "model": model,
        "character": character,
        "duration": float(duration),
    }
    if seed >= 0:
        payload["seed"] = int(seed)
    return _post_job_body(payload, headers=headers)


def _poll_job(job_id: str, max_wait: float = 180.0) -> dict:
    """Poll /v1/jobs/{id} until done | failed | cancelled, or timeout."""
    deadline = time.time() + max_wait
    last_stage = None
    while time.time() < deadline:
        r = httpx.get(
            f"{_API_BASE}/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            timeout=15.0,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("stage") and body["stage"] != last_stage:
            last_stage = body["stage"]
            log.info("[%s] %s", job_id, last_stage)
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(1.5)
    raise TimeoutError(f"Job {job_id} did not finish in {max_wait}s")


def _generate(
    prompt: str,
    model: str,
    character: str,
    duration: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request | None = None,
) -> tuple[str, str, str]:
    """Submit → poll → return (path-to-glb-or-fbx, status text, rewritten_prompt)."""
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required")

    # In-Space (hf.space iframe) visitors carry an HF-injected X-IP-Token;
    # forward it across the loopback so ZeroGPU attributes their quota
    # (see _zerogpu_attribution in animoflow-api main.py).
    _ip_token = request.headers.get("x-ip-token") if request is not None else None
    _fwd_headers = {"x-ip-token": _ip_token} if _ip_token else None

    progress(0.05, desc="Submitting…")
    try:
        job_id = _post_job(prompt, model, character, duration, seed,
                           headers=_fwd_headers)
    except httpx.HTTPStatusError as exc:
        # NB: the previous try/except here caught its own gr.Error, so the
        # structured-message path never ran and everything collapsed to a
        # raw "API error: {exc}" leak. _submit_error keeps parsing and
        # raising separate.
        message, _ = _submit_error(exc)
        raise gr.Error(message) from exc

    progress(0.10, desc="Generating…")
    try:
        job = _poll_job(job_id)
    except TimeoutError as exc:
        raise gr.Error(str(exc)) from exc

    if job["status"] != "done":
        # The backend's ``error`` field is the single user-facing string —
        # populated by api.error_classify with a clean message. We render
        # it verbatim; no client-side formatting or branching on
        # error_info.code.
        msg = job.get("error") or "Job did not complete."
        raise gr.Error(msg)

    download_url = job.get("download_url", "")  # /v1/files/<job_id>.fbx
    if not download_url:
        raise gr.Error("Job done but no download_url returned")

    # The file is on disk in OUTPUT_DIR — return a local path so the
    # Model3D widget loads it without a second HTTP roundtrip.
    filename = download_url.rsplit("/", 1)[-1]
    fbx_path = _OUTPUT_DIR / filename
    glb_path = fbx_path.with_suffix(".glb")
    out_path = glb_path if glb_path.exists() else fbx_path
    if not out_path.exists():
        raise gr.Error(f"Output file missing: {out_path}")

    progress(1.0, desc="Done")
    summary = (
        f"Job {job_id} · model={model} · character={character} · "
        f"duration={duration}s · seed={seed}"
    )
    # Surface the rewritten prompt so the user can see what was actually fed
    # to the motion model. Empty string when the rewriter was skipped or the
    # heuristic short-circuited (input was already HumanML3D-style English).
    rewritten = job.get("rewritten_prompt") or ""
    original = job.get("original_prompt") or ""
    if rewritten and rewritten.strip() != (original or "").strip():
        rewritten_display = rewritten
    else:
        rewritten_display = ""
    return str(out_path), summary, rewritten_display


def _generate_full(
    request_json: str,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request | None = None,
):
    """Submit → poll → yield (path, summary, rewritten) for any task type.

    The browser-side `@gradio/client` calls this via
    ``Client.predict("/generate_full", request_json=...)`` with the
    full ``GenerateRequest`` body JSON-serialized as a string. Same
    body shape as ``POST /v1/jobs`` accepts — see animoflow-api's
    GenerateRequest schema (text / trajectory / waypoints / timeline
    discriminated union). Routing all task types through this single
    Gradio endpoint is what lets ZeroGPU's ``x-ip-token`` middleware
    attribute quota to the signed-in user's HF account.

    **Generator pattern intentionally.** Each ``yield`` emits a
    ``process_generating`` SSE event over the simplified
    ``/gradio_api/call/generate_full/<event_id>`` endpoint that the
    browser-direct custom client consumes (the simple SSE endpoint
    DROPS gr.Progress events, only forwards generator yields). The
    intermediate yields use ``gr.update()`` for the Model3D + rewritten
    textbox so the Gradio UI doesn't flicker the viewer between stages.
    Browser-direct callers read the status string from the second tuple
    slot and ignore the first/third on intermediate yields. Final yield
    has the actual file path + summary + rewritten_display.

    Output components: (Model3D, status Textbox, rewritten Textbox) —
    matches Blocks wiring at the bottom of this file.
    """
    if not request_json or not request_json.strip():
        raise gr.Error("request_json is required")

    try:
        body = json.loads(request_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"request_json is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise gr.Error("request_json must be a JSON object")

    # Progress encoding: the second-slot status string carries a trailing
    # "(NN%)" that the browser-direct client (web/app/gradio-client-custom.js
    # → app.js:_generateViaGradio) parses to drive the spinner row's
    # --progress CSS variable. The 4th slot is RESERVED for snap_info JSON
    # on the final yield (per commit d14ae86 — per-gen snap receipt) — we
    # keep it as gr.update() on every intermediate yield so the hidden
    # snap_info textbox isn't clobbered mid-run.
    # The simple SSE endpoint we call from the browser DROPS gr.Progress
    # events, so the only reliable carrier is the yielded tuple itself
    # (encode progress into a tuple slot).
    # Attribution for usage analytics: hash the browser-direct caller's
    # bearer token so the internal loopback /v1/jobs hop keeps the real
    # identity instead of "dev". Never carries raw token material.
    _actor = None
    if usage_log is not None and request is not None:
        try:
            _actor = usage_log.actor_from_authorization(
                request.headers.get("authorization"))
            if _actor is None:
                # Anonymous browser-direct caller: forward a hashed-IP
                # identity so the fair-use overflow cap can key on them.
                # Without this, anon loopback jobs collapse into the
                # internal API key's identity (gap found 2026-07-07).
                _ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                       or (request.client.host if request.client else ""))
                _actor = usage_log.actor_from_ip(_ip)
        except Exception:  # analytics must never break generation
            _actor = None
    _actor_headers = {usage_log.ACTOR_HEADER: _actor} if _actor else None
    # ZeroGPU attribution: forward the caller's X-IP-Token JWT across the
    # loopback hop. The spaces scheduler reads it from gradio's request
    # context, which the internal /v1/jobs request severs — main.py
    # replants it around the pipeline run (_zerogpu_attribution). Without
    # this, every GPU call runs token-less on the shared pool.
    _ip_token = request.headers.get("x-ip-token") if request is not None else None
    if _ip_token:
        _actor_headers = {**(_actor_headers or {}), "x-ip-token": _ip_token}

    def _gradio_call_event(phase: str, **extra) -> None:
        if usage_log is None:
            return
        kind, _, user = (_actor or "").partition(":")
        usage_log.emit({
            "event": "gradio_call", "phase": phase,
            "user": user or "internal", "user_kind": kind or "internal",
            "client": "browser-direct", **extra,
        })

    progress(0.05, desc="Submitting…")
    yield gr.update(), "Submitting… (5%)", gr.update(), gr.update()
    try:
        job_id = _post_job_body(body, headers=_actor_headers)
    except httpx.HTTPStatusError as exc:
        message, code = _submit_error(exc)
        _gradio_call_event(
            "submit_rejected", job_id=None,
            error_code=code,
            error_message=message[:300],
        )
        raise gr.Error(message) from exc

    _gradio_call_event("linked", job_id=job_id)

    progress(0.10, desc="Generating…")
    # Slot 3 carries the job_id early: gradio's simple /call SSE endpoint
    # serializes error events as `data: null` (the gr.Error message is
    # lost on the wire), so browser-direct callers need the job_id to
    # fetch the real classified error via GET /v1/jobs/{id} when the
    # stream errors out. The final yield reuses slot 3 for snap_info.
    yield gr.update(), "Generating… (10%)", gr.update(), json.dumps({"job_id": job_id})

    # Inline poll so we can yield each stage AND progress change.
    # 0.4 s poll matches the legacy /v1/jobs poller in app.js — at 1.5 s
    # short pipeline stages (e.g. IK, post-fix) flew past between polls
    # and the second-slot yield stayed "Generating…" the whole run.
    max_wait = 180.0
    poll_interval = 0.4
    deadline = time.time() + max_wait
    last_stage = None
    last_pct = -1
    job = None
    while time.time() < deadline:
        try:
            r = httpx.get(
                f"{_API_BASE}/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {_API_KEY}"},
                timeout=15.0,
            )
            r.raise_for_status()
            body_resp = r.json()
        except Exception as exc:  # transient HTTP / parse error — keep polling
            log.warning("[%s] poll error: %s", job_id, exc)
            time.sleep(poll_interval)
            continue
        cur_stage = body_resp.get("stage") or last_stage or "Running"
        cur_progress = max(0.0, min(1.0, float(body_resp.get("progress") or 0.0)))
        cur_pct = int(round(cur_progress * 100))
        if cur_stage != last_stage or cur_pct != last_pct:
            if cur_stage != last_stage:
                log.info("[%s] %s (%d%%)", job_id, cur_stage, cur_pct)
            last_stage = cur_stage
            last_pct = cur_pct
            yield gr.update(), f"{cur_stage} ({cur_pct}%)", gr.update(), gr.update()
        if body_resp["status"] in ("done", "failed", "cancelled"):
            job = body_resp
            break
        time.sleep(poll_interval)
    if job is None:
        _gradio_call_event("poll_timeout", job_id=job_id,
                           error_code="gradio_poll_timeout",
                           error_message=f"no terminal status within {max_wait}s")
        raise gr.Error(f"Job {job_id} did not finish in {max_wait}s")

    if job["status"] != "done":
        msg = job.get("error") or "Job did not complete."
        raise gr.Error(msg)

    download_url = job.get("download_url", "")
    if not download_url:
        raise gr.Error("Job done but no download_url returned")

    filename = download_url.rsplit("/", 1)[-1]
    fbx_path = _OUTPUT_DIR / filename
    glb_path = fbx_path.with_suffix(".glb")
    out_path = glb_path if glb_path.exists() else fbx_path
    if not out_path.exists():
        raise gr.Error(f"Output file missing: {out_path}")

    progress(1.0, desc="Done")

    input_type = (body.get("input") or {}).get("type", "?")
    model_name = body.get("model", "?")
    duration_val = body.get("duration", "?")
    summary = (
        f"Job {job_id} · task={input_type} · model={model_name} · duration={duration_val}s"
    )

    rewritten = job.get("rewritten_prompt") or ""
    original = job.get("original_prompt") or ""
    if rewritten and rewritten.strip() != (original or "").strip():
        rewritten_display = rewritten
    else:
        rewritten_display = ""
    # JSON-serialise snap_info into a hidden textbox slot. Browser-direct
    # callers (the AnimoFlow web app) read data[3]
    # and surface it in the debug-response panel — same per-gen "did the
    # snap run?" receipt that REST clients get via JobResponse.snap_info.
    snap_info_json = json.dumps(job.get("snap_info"))
    yield str(out_path), summary, rewritten_display, snap_info_json


def _list_characters() -> list[str]:
    """Best-effort character list from /v1/characters; fall back to filesystem."""
    try:
        r = httpx.get(
            f"{_API_BASE}/v1/characters",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            timeout=5.0,
        )
        r.raise_for_status()
        names = r.json().get("characters", [])
        if names:
            return names
    except Exception:  # noqa: BLE001 — best effort during boot
        pass
    chars_dir = Path(
        os.environ.get("CHARACTERS_DIR", "/opt/comfyui-animoflow/characters")
    )
    if chars_dir.is_dir():
        return sorted(p.stem for p in chars_dir.glob("*.fbx"))
    return ["Y_bot"]


def build_blocks() -> gr.Blocks:
    """Construct the Gradio Blocks app. Called once at orchestrator startup."""
    # Pull the character list once at boot. Refreshed on demand via the
    # "↻" button next to the dropdown.
    initial_chars = _list_characters() or ["Y_bot"]

    with gr.Blocks(title="AnimoFlow Demo", theme=gr.themes.Soft()) as blocks:
        gr.Markdown(
            """
            # AnimoFlow — Text → Motion → FBX

            Type a description, pick a character, hit **Generate**.
            Powered by [MDM](https://guytevet.github.io/mdm-page/) +
            momask Joint2BVH IK + Blender retarget.

            Same `/v1/jobs` API as the local OSS stack — you can also call
            it from `gradio_client`, `@gradio/client` JS, or curl.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a person walks forward and waves",
                    lines=3,
                )
                model = gr.Dropdown(
                    choices=["mdm", "momask", "kimodo"],
                    value="mdm",
                    label="Model",
                )
                character = gr.Dropdown(
                    choices=initial_chars,
                    value=initial_chars[0],
                    label="Character",
                )
                duration = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="Duration (seconds)",
                )
                seed = gr.Number(
                    value=-1, label="Seed (-1 = random)", precision=0
                )
                submit = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                viewer = gr.Model3D(
                    label="Preview", display_mode="solid", clear_color=[0, 0, 0, 0]
                )
                status = gr.Textbox(label="Job", interactive=False, lines=2)
                # Surfaces the rewritten prompt when the multilingual rewriter
                # actually fires (non-empty + different from the original). The
                # textbox stays blank for English-HumanML3D-style inputs that
                # the cheap skip heuristic short-circuits. Original/rewritten
                # are also in /v1/jobs/{id} for API consumers.
                rewritten_box = gr.Textbox(
                    label="Rewritten as (what the model actually saw)",
                    interactive=False,
                    lines=2,
                    placeholder="(your prompt was kept as-is)",
                )

        submit.click(
            fn=_generate,
            inputs=[prompt, model, character, duration, seed],
            outputs=[viewer, status, rewritten_box],
            # Stable named endpoint so the browser-side @gradio/client
            # in animoflow-api can call Client.predict("/generate", ...).
            api_name="generate",
        )

        # Hidden API-only endpoint that accepts the full GenerateRequest
        # body as a JSON string. Used by animoflow-api's browser-side
        # @gradio/client for all four task types (text / trajectory /
        # waypoints / timeline). Not visible in the Space's own UI —
        # external callers reach it via Client.predict("/generate_full", ...).
        request_json_in = gr.Textbox(visible=False)
        snap_info_box = gr.Textbox(visible=False)
        submit_full = gr.Button(visible=False)
        submit_full.click(
            fn=_generate_full,
            inputs=[request_json_in],
            outputs=[viewer, status, rewritten_box, snap_info_box],
            api_name="generate_full",
        )

        gr.Markdown(
            """
            ### API access

            ```bash
            curl -X POST $API/v1/jobs \\
              -H "Authorization: Bearer dev" \\
              -H "Content-Type: application/json" \\
              -d '{"input":{"type":"text","prompt":"a person walks"},"model":"mdm","duration":4}'
            ```

            Then poll `/v1/jobs/{id}` until `status=done` and download
            `/v1/files/{id}.fbx` (or `.glb`).
            """
        )
    return blocks
