"""
Pipeline tests — exercise the HF pipeline shape with mocked inference.

We do NOT load real model weights here. The goal is to verify:
  * The pipeline accepts the same call signature as animoflow-api/api/pipeline.py:run
  * Errors propagate cleanly when deps are missing
  * Stage callbacks fire in the expected order
  * The escape-hatch routing logic recognizes outliers correctly
"""

from __future__ import annotations

import pytest


def test_signature_compat_with_animoflow_api_pipeline():
    """The HF pipeline must accept every kwarg the original pipeline.run does.

    If this drifts, animoflow-api/api/main.py:_run_pipeline will start
    passing kwargs that pipeline_hf.run rejects.
    """
    import inspect

    import pipeline_hf

    sig = inspect.signature(pipeline_hf.run)
    params = sig.parameters

    # These are all the kwargs animoflow-api's main.py:_run_pipeline passes.
    required = {
        "job_id",
        "prompt",
        "num_frames",
        "seed",
        "character",
        "model",
        "on_progress",
        "on_stage",
        "preprocess_stages",
        "keyframe_builder",
        "keyframe_builder_error_degrees",
        "output_fps",
        "curve_2d",
        "accel_frac",
        "decel_frac",
    }
    missing = required - set(params.keys())
    assert not missing, f"pipeline_hf.run is missing kwargs: {missing}"


def test_run_timeline_signature():
    import inspect

    import pipeline_hf

    sig = inspect.signature(pipeline_hf.run_timeline)
    params = sig.parameters
    required = {
        "job_id",
        "segments",
        "seed",
        "character",
        "cfg",
        "handshake_size",
        "blend_len",
        "on_progress",
        "on_stage",
        "keyframe_builder",
        "keyframe_builder_error_degrees",
        "output_fps",
    }
    missing = required - set(params.keys())
    assert not missing, f"pipeline_hf.run_timeline is missing kwargs: {missing}"


def _point_at_sibling_checkout(monkeypatch):
    """Point pipeline_hf at the sibling comfyui-animoflow checkout so the
    shared stage library loads on dev machines (the Space uses
    /opt/comfyui-animoflow). Skip when neither exists."""
    from pathlib import Path

    import pipeline_hf

    sibling = Path(__file__).resolve().parents[2] / "comfyui-animoflow"
    if (sibling / "animoflow_stages" / "__init__.py").is_file():
        monkeypatch.setattr(pipeline_hf, "_COMFY_ROOT", sibling)
        monkeypatch.setattr(pipeline_hf, "_stage_lib", None)
    elif not (pipeline_hf._COMFY_ROOT / "animoflow_stages" / "__init__.py").is_file():
        pytest.skip("no comfyui-animoflow checkout with animoflow_stages available")


def test_run_timeline_wires_to_priormdm(monkeypatch):
    """run_timeline must call _run_timeline_gpu, unwrap the result, and
    pass through to the shared post-processing tail. We mock both ends so
    the test doesn't load real model weights or invoke Blender."""
    import pipeline_hf

    captured: dict = {}

    def _fake_gpu(segments, seed, *, guidance_param=2.5,
                  handshake_size=10, blend_len=10):
        captured["segments"] = segments
        captured["seed"] = seed
        captured["guidance_param"] = guidance_param
        captured["handshake_size"] = handshake_size
        captured["blend_len"] = blend_len
        return b"FAKE_NPZ_BYTES_NO_SENTINEL"

    def _fake_postprocess(**kw):
        captured["postprocess_kwargs"] = kw
        return "fake.fbx"

    monkeypatch.setattr(pipeline_hf, "_run_timeline_gpu", _fake_gpu)
    monkeypatch.setattr(pipeline_hf, "_run_tail", _fake_postprocess)
    _point_at_sibling_checkout(monkeypatch)
    # The whole tail (incl. resample) is mocked out via _run_tail, so the
    # fake NPZ bytes are never parsed.
    out = pipeline_hf.run_timeline(
        job_id="t1",
        segments=[{"prompt": "wave", "num_frames": 80},
                  {"prompt": "jump", "num_frames": 60}],
        seed=42,
        cfg=2.5,
        output_fps=20,
    )
    assert out == "fake.fbx"
    assert captured["segments"][0]["prompt"] == "wave"
    assert captured["seed"] == 42
    assert captured["guidance_param"] == 2.5
    assert captured["postprocess_kwargs"]["npz_bytes"] == b"FAKE_NPZ_BYTES_NO_SENTINEL"


def test_run_priormdm_forwards_curve_2d_to_gpu(monkeypatch):
    """run(model='priormdm') must forward curve_2d + accel/decel_frac into
    the GPU call's extra-kwargs so the priorMDM wrapper sees them."""
    import pipeline_hf

    captured: dict = {}

    def _fake_gpu(model, prompt, num_frames, seed, *,
                  guidance_param=None, **extra):
        captured["model"] = model
        captured["prompt"] = prompt
        captured.update(extra)
        return b"FAKE_NPZ_BYTES_NO_SENTINEL"

    def _fake_postprocess(**kw):
        return "fake.fbx"

    monkeypatch.setattr(pipeline_hf, "_run_inference_gpu_fast", _fake_gpu)
    monkeypatch.setattr(pipeline_hf, "_run_tail", _fake_postprocess)
    _point_at_sibling_checkout(monkeypatch)
    out = pipeline_hf.run(
        job_id="p1",
        prompt="walk forward",
        num_frames=80,
        seed=0,
        character="Y_bot",
        model="priormdm",
        curve_2d=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        accel_frac=0.2,
        decel_frac=0.3,
        output_fps=20,  # priormdm native, skip resample on the fake bytes
    )
    assert out == "fake.fbx"
    assert captured["model"] == "priormdm"
    assert captured["curve_2d"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    assert captured["accel_frac"] == 0.2
    assert captured["decel_frac"] == 0.3


def test_escape_hatch_recognizes_outliers():
    """Outlier registry should report unknown models as non-outliers."""
    import escape_hatch

    # The MDM family runs in-process (kimodo is the escape-hatch outlier)
    assert escape_hatch.is_outlier("mdm") is False
    assert escape_hatch.is_outlier("priormdm") is False
    assert escape_hatch.is_outlier("not_a_real_model") is False


def test_escape_hatch_raises_on_unregistered_invoke():
    """invoke() on a non-outlier model must raise a clear RuntimeError."""
    import escape_hatch

    with pytest.raises(RuntimeError) as excinfo:
        escape_hatch.invoke("mdm", prompt="test", num_frames=60, seed=0)
    assert "outlier" in str(excinfo.value).lower()


def test_numpy_compat_shim_installs_legacy_aliases():
    """Shim must add np.float / np.int / np.bool / numpy.core.umath_tests
    even on numpy 2.x. Idempotent."""
    import sys

    import numpy as np

    import pipeline_hf

    # Run twice to confirm idempotency
    pipeline_hf._install_numpy_compat_shim()
    pipeline_hf._install_numpy_compat_shim()

    # Legacy aliases present
    assert hasattr(np, "float")
    assert hasattr(np, "int")
    assert hasattr(np, "bool")
    assert hasattr(np, "object")

    # umath_tests stub is registered
    import numpy.core.umath_tests as ut

    # inner1d → element-wise dot product
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    expected_inner1d = np.array([6.0, 30.0])
    assert np.allclose(ut.inner1d(a, b), expected_inner1d), (
        f"inner1d shim returned {ut.inner1d(a, b)!r}, expected {expected_inner1d!r}"
    )

    # matrix_multiply → np.matmul (stacked matrix mult)
    M1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [0.0, 2.0]]])
    M2 = np.array([[[5.0, 6.0], [7.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]]])
    expected_mm = np.matmul(M1, M2)
    assert np.allclose(ut.matrix_multiply(M1, M2), expected_mm), (
        "matrix_multiply shim diverged from np.matmul"
    )

    # innerwt → weighted inner product, sum_i a_i * b_i * w_i over last axis
    w = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    # Row 0: 1·1·1 + 2·1·2 + 3·1·3 = 1+4+9 = 14
    # Row 1: 4·2·3 + 5·2·2 + 6·2·1 = 24+20+12 = 56
    expected_innerwt = np.array([14.0, 56.0])
    assert np.allclose(ut.innerwt(a, b, w), expected_innerwt), (
        f"innerwt shim returned {ut.innerwt(a, b, w)!r}, expected {expected_innerwt!r}"
    )

    # Module is in sys.modules
    assert "numpy.core.umath_tests" in sys.modules


def test_pipeline_run_propagates_callbacks(monkeypatch):
    """Stage / progress callbacks should fire even when inference fails.

    We monkeypatch _run_inference_gpu_fast (the MDM/MoMask GPU wrapper) so no
    real model loads. The pipeline should call on_stage("Generating…") at
    least once before failing.
    """
    import pipeline_hf

    stages: list[str] = []

    def _on_stage(s: str):
        stages.append(s)

    def _fake_inference(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("synthetic — model not loaded")

    monkeypatch.setattr(pipeline_hf, "_run_inference_gpu_fast", _fake_inference)
    _point_at_sibling_checkout(monkeypatch)

    with pytest.raises(RuntimeError):
        pipeline_hf.run(
            job_id="test-job-1",
            prompt="test prompt",
            num_frames=60,
            seed=42,
            character="Y_bot",
            model="mdm",
            on_stage=_on_stage,
        )

    assert "Generating…" in stages, f"expected 'Generating…' to fire, got {stages!r}"


def test_safe_do_inference_passes_through_npz_bytes(monkeypatch):
    """The sentinel wrapper must NOT modify successful return values."""
    import pipeline_hf

    monkeypatch.setattr(
        pipeline_hf, "_do_inference",
        lambda *a, **kw: b"FAKE_NPZ_BYTES_NO_SENTINEL",
    )
    out = pipeline_hf._safe_do_inference(
        model="mdm", prompt="x", num_frames=60, seed=0,
    )
    assert out == b"FAKE_NPZ_BYTES_NO_SENTINEL"


def test_safe_do_inference_serialises_exception_into_sentinel(monkeypatch):
    """When _do_inference raises, the wrapper must emit a sentinel-prefixed
    JSON payload — not propagate the exception. This is the fix for the
    'AppError: \\'RuntimeError\\'' Space symptom where ZeroGPU strips
    exception args across the worker boundary."""
    import json

    import pipeline_hf

    def _boom(*a, **kw):  # noqa: ARG001
        raise RuntimeError("Kimodo venv still warming up — please retry")

    monkeypatch.setattr(pipeline_hf, "_do_inference", _boom)

    out = pipeline_hf._safe_do_inference(
        model="kimodo", prompt="x", num_frames=60, seed=0,
    )

    assert out.startswith(pipeline_hf._ERR_SENTINEL), (
        "wrapper must prefix the error payload with _ERR_SENTINEL"
    )
    payload = json.loads(out[len(pipeline_hf._ERR_SENTINEL):].decode())
    assert payload["error_class"] == "RuntimeError"
    assert "warming up" in payload["message"]  # original message preserved
    assert "traceback" in payload


def test_unwrap_gpu_result_reraises_with_original_message():
    """Inverse of _safe_do_inference: detect the sentinel and re-raise the
    original message as a RuntimeError on the orchestrator side."""
    import json

    import pipeline_hf

    payload = json.dumps({
        "error_class": "RuntimeError",
        "message": "Kimodo venv still warming up — please retry",
        "traceback": "fake-traceback",
    }).encode()
    sentinel_blob = pipeline_hf._ERR_SENTINEL + payload

    with pytest.raises(RuntimeError) as excinfo:
        pipeline_hf._unwrap_gpu_result(sentinel_blob)
    assert "warming up" in str(excinfo.value)


def test_unwrap_gpu_result_passes_through_real_bytes():
    """No sentinel → pass through unchanged."""
    import pipeline_hf

    real = b"PK\x03\x04...npz bytes..."  # any non-sentinel byte string
    assert pipeline_hf._unwrap_gpu_result(real) is real
