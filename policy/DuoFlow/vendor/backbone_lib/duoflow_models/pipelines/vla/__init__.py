# Modified 2026-10-19: runtime namespacing.
from importlib import import_module

from .action_core import ActionPipeline

__all__ = ['ActionPipeline', 'Pi0Pipeline']


def __getattr__(name):
    if name != 'Pi0Pipeline':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = import_module('.pi0', __name__).Pi0Pipeline
    globals()[name] = value
    return value
