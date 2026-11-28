"""Public model API for the XPolicyLab RLinf adapter."""

__all__ = ["get_model", "XPolicyLabForRLActionPrediction"]


def __getattr__(name):
    # The simulator Python can import its bridge without loading the ML runtime.
    if name == "get_model":
        from .core.factory import get_model
        return get_model
    if name == "XPolicyLabForRLActionPrediction":
        from .core.action_model import XPolicyLabForRLActionPrediction
        return XPolicyLabForRLActionPrediction
    raise AttributeError(name)
