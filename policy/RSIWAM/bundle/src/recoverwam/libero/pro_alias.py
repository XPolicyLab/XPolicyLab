import importlib
import sys


def route_liberopro_as_libero() -> None:
    """Install the aliases expected by the repository's LIBERO evaluator."""
    real_package = importlib.import_module("liberopro")
    real_core = importlib.import_module("liberopro.liberopro")
    aliases = {
        "libero": real_package,
        "libero.libero": real_core,
        "libero.libero.benchmark": importlib.import_module("liberopro.liberopro.benchmark"),
        "libero.libero.envs": importlib.import_module("liberopro.liberopro.envs"),
    }
    sys.modules.update(aliases)
