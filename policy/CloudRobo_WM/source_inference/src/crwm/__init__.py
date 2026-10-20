"""CR-WM inference package."""

__version__ = "1.0.0"


def __getattr__(name):
    if name == "CRWMPolicy":
        from .policy import CRWMPolicy
        return CRWMPolicy
    raise AttributeError(name)


__all__ = ["CRWMPolicy"]
