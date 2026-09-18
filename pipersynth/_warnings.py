from __future__ import annotations

import inspect
import warnings


def warn_external(
    message: str,
    category: type[Warning],
    *,
    package_prefix: str = "pipersynth",
) -> None:
    """Emit a warning whose source location points outside *package_prefix*.

    Walks the call stack outward until a frame whose ``__name__`` does not
    start with *package_prefix* is found, then emits the warning at that
    stack level so diagnostics point at the caller, not at an internal
    wrapper.
    """
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        stacklevel = 2
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if not module.startswith(package_prefix):
                break
            stacklevel += 1
            frame = frame.f_back
        warnings.warn(message, category, stacklevel=stacklevel)
    finally:
        del frame
