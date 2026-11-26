import pydoc
from typing import Any


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


def locate_type(spec: Any, default_module: str | None = None) -> Any:
    """Resolve a config ``type`` entry to a callable.

    Config factories accept ``type`` as either a bare/dotted import string or an
    already-imported callable. A bare string (no ``.``) is prefixed with
    ``default_module`` before being located.

    Args:
        spec: the ``type`` value from config — an import string or a callable.
        default_module: module prefix applied to bare string specs (no dot).

    Returns:
        The resolved callable.
    """
    if isinstance(spec, str):
        if default_module and "." not in spec:
            spec = f"{default_module}.{spec}"
        obj = locate(spec)
    else:
        obj = spec
    assert callable(obj), f"config type must resolve to a callable, but got {obj!r}"
    return obj
