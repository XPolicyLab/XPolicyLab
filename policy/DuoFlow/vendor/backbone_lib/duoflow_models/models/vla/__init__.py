# Modified 2026-10-19: runtime namespacing.
from importlib import import_module

from .action_core import ActionBackbone

__all__ = ['ActionBackbone', 'PI0Policy']


def __getattr__(name):
    if name != 'PI0Policy':
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = import_module('.pi0', __name__).PI0Policy
    globals()[name] = value
    return value
