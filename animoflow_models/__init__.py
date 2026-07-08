"""
Model registry for animoflow-app.

Imports the upstream model `inference.py` modules into the orchestrator
process at module level (per ZeroGPU best practice). Each model becomes
warm and is inherited by the @spaces.GPU fork via copy-on-write — no
per-request load cost.

Outlier models (Kimodo etc. — separate dep tree, won't merge here) live
in escape_hatch/ and are subprocess-called only when their model is
selected.

Carries MDM, priorMDM, and MoMask.
"""

from .registry import generate, list_models, model_status

__all__ = ["generate", "list_models", "model_status"]
