"""Local Focus-VLWA inference API."""

from typing import Any

__all__ = ["FocusVLWAPolicy"]


def __getattr__(name: str) -> Any:
    """Load the large model dependency only when the policy is requested."""
    if name == "FocusVLWAPolicy":
        from focus_vlwa.inference.policy import FocusVLWAPolicy

        return FocusVLWAPolicy
    raise AttributeError(name)
