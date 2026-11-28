"""Model adapters retained for legacy VAP checkpoint construction.

The eval-only package does not include unrelated model or training adapters.
"""

from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.adapters.vga_model_adapter import VGAAdapter
from wall_x.trainer.adapters.vap_model_adapter import VAPAdapter

ADAPTER_REGISTRY = {"wan": VGAAdapter, "vap": VAPAdapter}
ADAPTER_IMPORT_ERRORS: dict[str, str] = {}


def format_adapter_error(model_type: str) -> str:
    return (
        f"Unsupported model type: {model_type}. "
        f"Registered: {sorted(ADAPTER_REGISTRY)}"
    )


__all__ = [
    "ModelAdapter",
    "VGAAdapter",
    "VAPAdapter",
    "ADAPTER_REGISTRY",
    "ADAPTER_IMPORT_ERRORS",
    "format_adapter_error",
]
