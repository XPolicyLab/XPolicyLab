"""Load XPolicyLab's external StarVLA data registry overlays.

This module is imported automatically by Python when ``starvla_adapter`` is on
``PYTHONPATH``. It lets the XPolicyLab policy keep dataset-specific overrides
outside the StarVLA source submodule.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _registry_dirs() -> list[Path]:
    configured = os.environ.get("STARVLA_EXTRA_DATA_REGISTRY")
    if configured:
        return [Path(path).expanduser().resolve() for path in configured.split(os.pathsep) if path]
    return [Path(__file__).resolve().parent / "data_registry"]


def _merge_external_registries() -> None:
    try:
        from starVLA.dataloader.gr00t_lerobot import registry
    except Exception as exc:  # pragma: no cover - best-effort startup hook
        LOGGER.debug("Skip StarVLA external registry overlay: %s", exc)
        return

    for registry_dir in _registry_dirs():
        data_config = registry_dir / "data_config.py"
        if not data_config.is_file():
            continue

        module = _load_module_from_path(
            f"_xpolicy_starvla_data_registry_{registry_dir.name}",
            data_config,
        )
        if module is None:
            continue

        if hasattr(module, "ROBOT_TYPE_CONFIG_MAP"):
            registry.ROBOT_TYPE_CONFIG_MAP.update(module.ROBOT_TYPE_CONFIG_MAP)
        if hasattr(module, "DATASET_NAMED_MIXTURES"):
            registry.DATASET_NAMED_MIXTURES.update(module.DATASET_NAMED_MIXTURES)
        if hasattr(module, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
            registry.ROBOT_TYPE_TO_EMBODIMENT_TAG.update(module.ROBOT_TYPE_TO_EMBODIMENT_TAG)

    if hasattr(registry, "_derive_tag_map"):
        registry.ROBOT_TYPE_TO_EMBODIMENT_TAG = registry._derive_tag_map()


_merge_external_registries()
