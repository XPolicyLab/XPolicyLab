from typing import Callable
from omegaconf import OmegaConf


def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)


def register_default_resolvers() -> None:
    """
    Register resolvers used by public YAML configs.
    Safe to call multiple times.
    """
    _register("len", len)
