"""Historical inference modules, bundled without a training-repository dependency."""
import importlib
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"


def activate_vendor():
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"psi_policy", "hyp_bslii", "wan"}:
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(VENDOR):
                raise ImportError(f"External historical runtime already imported: {name}: {origin}")
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def historical_module(name):
    activate_vendor()
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(VENDOR):
        raise ImportError(f"Expected bundled runtime: {name}")
    return module
