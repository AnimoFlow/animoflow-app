"""
Test the @spaces.GPU compat shim.

Critical correctness: the decorator must work in BOTH forms (`@GPU` and
`@GPU(duration=N)`) on a non-HF environment, transparently passing the
function through. If this is wrong, every OSS local install breaks.
"""

from __future__ import annotations

import sys


def test_GPU_parameterized_form_passes_through():
    """`@GPU(duration=30)` should call the underlying function with all args."""
    from spaces_compat import GPU

    @GPU(duration=30)
    def fn(a, b):
        return a + b

    assert fn(2, 3) == 5
    assert fn(a=4, b=10) == 14


def test_GPU_bare_form_passes_through():
    """`@GPU` (no parens) should still work as a bare decorator."""
    from spaces_compat import GPU

    @GPU
    def fn(x):
        return x * 2

    assert fn(7) == 14


def test_GPU_kwargs_only_form():
    """`@GPU(duration=30, ...)` accepts arbitrary kwargs without breaking."""
    from spaces_compat import GPU

    @GPU(duration=60, queue=False)
    def fn():
        return "ok"

    assert fn() == "ok"


def test_GPU_preserves_metadata():
    """The decorated function should keep its name/docstring (functools.wraps)."""
    from spaces_compat import GPU

    @GPU(duration=10)
    def my_function():
        """An important docstring."""
        return None

    # In the spaces-installed branch we re-export spaces.GPU directly which
    # may or may not preserve metadata depending on spaces' implementation.
    # We only assert metadata preservation when the OSS fallback is active.
    import spaces_compat

    if not spaces_compat.HAVE_SPACES:
        assert my_function.__name__ == "my_function"
        assert "important" in (my_function.__doc__ or "")


def test_GPU_does_not_swallow_exceptions():
    """If the wrapped function raises, the wrapper must re-raise."""
    from spaces_compat import GPU

    class _Boom(Exception):
        pass

    @GPU(duration=5)
    def bad():
        raise _Boom("kaboom")

    try:
        bad()
    except _Boom:
        return
    raise AssertionError("decorator swallowed the exception")
