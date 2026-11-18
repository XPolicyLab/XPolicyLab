"""Policy namespace, including literal versioned directory names."""
from pathlib import Path as _Path
from importlib.util import spec_from_file_location as _spec_from_file_location
from importlib.util import module_from_spec as _module_from_spec
import sys as _sys


def _register_versioned_policies():
    for directory in _Path(__file__).resolve().parent.iterdir():
        if "." not in directory.name or not (directory / "__init__.py").is_file():
            continue
        name = f"{__name__}.{directory.name}"
        if name in _sys.modules:
            continue
        spec = _spec_from_file_location(name, directory / "__init__.py",
                                       submodule_search_locations=[str(directory)])
        module = _module_from_spec(spec)
        _sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            _sys.modules.pop(name, None)
            raise
        setattr(_sys.modules[__name__], directory.name, module)


_register_versioned_policies()
