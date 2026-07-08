"""
ZeroGPU compatibility shim.

In an HF Space with ZeroGPU, the `spaces` package is installed and
`@spaces.GPU(duration=N)` allocates an H200 to the decorated function for
its duration. Outside HF (any OSS local install, CI, tests), the package
isn't present — we fall back to a pass-through decorator so the same code
runs unchanged on any host.

Usage:

    from spaces_compat import GPU

    @GPU(duration=30)
    def run_inference(...):
        ...
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable

log = logging.getLogger(__name__)


def _is_huggingface_space() -> bool:
    """Heuristic: HF Spaces always set this env var."""
    return bool(os.environ.get("SPACE_ID")) or bool(os.environ.get("SPACE_AUTHOR_NAME"))


try:
    import spaces  # type: ignore[import-not-found]

    GPU = spaces.GPU  # re-export
    HAVE_SPACES = True
    if _is_huggingface_space():
        log.info("ZeroGPU active — @spaces.GPU will allocate H200 per call")
    else:
        log.info("`spaces` importable but not in HF Space — decorator is best-effort")
except ImportError:  # pragma: no cover — exercised in OSS local
    HAVE_SPACES = False

    def GPU(*decorator_args: Any, **decorator_kwargs: Any) -> Callable:  # type: ignore[no-redef]
        """Pass-through decorator for non-HF environments.

        Supports both forms:
            @GPU
            def fn(...): ...

            @GPU(duration=30)
            def fn(...): ...
        """
        # Bare-decorator form: @GPU on a function with no parens
        if (
            len(decorator_args) == 1
            and callable(decorator_args[0])
            and not decorator_kwargs
        ):
            return decorator_args[0]

        # Parameterized form: @GPU(...) returns a decorator
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    log.info("`spaces` package not installed — GPU decorator is a no-op pass-through")
