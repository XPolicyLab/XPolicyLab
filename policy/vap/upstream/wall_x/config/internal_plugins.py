"""Register the typed configuration schemas shipped with this eval-only build.

Dataset construction and RL/trainer implementations are intentionally not
installed. Complete-checkpoint serving only needs these dataclass registries.
"""

from __future__ import annotations

import os


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def register_internal_config_plugins() -> None:
    """Register VAP/VGA models and their checkpoint data configurations."""
    import wall_x.data.backends.x2robot_v1.config  # noqa: F401
    import wall_x.data.backends.x2robot_v2.config  # noqa: F401
    import wall_x.model.vap.config  # noqa: F401
    import wall_x.model.vapv2.config  # noqa: F401
    import wall_x.model.vga.config  # noqa: F401

    # The legacy Muon implementation initializes TorchDynamo on import. Config
    # loading for serving can skip it without changing any optimizer selection.
    if not _env_flag("WALLX_SKIP_LEGACY_MUON_PLUGIN"):
        import wall_x.trainer.optimizer.muon.config  # noqa: F401


__all__ = ["register_internal_config_plugins"]
