"""Process-local deterministic-mode flag shared by trainer and ops."""

import os

_ENABLED = os.getenv("FALCON_DETERMINISTIC", "0") == "1"


def is_deterministic_enabled() -> bool:
    """Return whether Falcon deterministic mode is enabled in this process."""
    return _ENABLED


def set_deterministic_enabled(enabled: bool = True) -> None:
    """Set Falcon deterministic mode for this process."""
    global _ENABLED
    _ENABLED = enabled
