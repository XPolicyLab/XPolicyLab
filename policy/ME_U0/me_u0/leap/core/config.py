"""Configuration system.

Provides layered YAML configuration loading with OmegaConf, and recursive
instantiation of objects from _target_ specs (similar to Hydra's instantiate).
"""

from __future__ import annotations

import importlib
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


# Register custom resolvers
if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver(
        "now", lambda fmt: datetime.now().strftime(fmt)
    )


# Sentinel for _target_ key
_TARGET_KEY = "_target_"
_DEFAULTS_KEY = "defaults"

# Default config search paths
_CONFIG_SEARCH_PATHS: List[Path] = []


def add_config_search_path(path: Union[str, Path]) -> None:
    """Add a directory to the config search path."""
    _CONFIG_SEARCH_PATHS.append(Path(path))


def _resolve_config_path(reference: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a config reference to a file path.

    Supports:
        - Absolute paths: /path/to/config.yaml
        - Relative paths: ./config.yaml or config.yaml
        - Layer references: /model: qwen2_moe_vla -> configs/model/qwen2_moe_vla.yaml
    """
    # Layer reference: "/model: qwen2_moe_vla" or root file: "/defaults"
    if reference.startswith("/") and not os.path.exists(reference):
        stripped = reference.lstrip("/")
        if ":" in stripped:
            # "/model: qwen2_moe_vla" -> layer="model", name="qwen2_moe_vla"
            layer, name = stripped.split(":", 1)
            layer, name = layer.strip(), name.strip()
            for search_path in _CONFIG_SEARCH_PATHS:
                candidate = search_path / layer / f"{name}.yaml"
                if candidate.exists():
                    return candidate
            if base_dir:
                candidate = base_dir / layer / f"{name}.yaml"
                if candidate.exists():
                    return candidate
        else:
            # "/defaults" -> configs/defaults.yaml
            name = stripped.strip()
            for search_path in _CONFIG_SEARCH_PATHS:
                candidate = search_path / f"{name}.yaml"
                if candidate.exists():
                    return candidate
            if base_dir:
                candidate = base_dir / f"{name}.yaml"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Config not found for reference '{reference}'. "
            f"Searched: {_CONFIG_SEARCH_PATHS}"
        )

    # Absolute path
    if os.path.isabs(reference):
        return Path(reference)

    # Relative path
    if base_dir:
        candidate = base_dir / reference
        if candidate.exists():
            return candidate

    return Path(reference)


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load an experiment config, resolving defaults and applying overrides.

    Args:
        config_path: Path to the experiment YAML config file.
        overrides: Optional list of dotlist overrides (e.g., ["training.lr=1e-5"]).

    Returns:
        Merged DictConfig with all layers resolved.
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)

    # Determine base directory for resolving relative references
    base_dir = config_path.parent
    configs_dir = _find_configs_dir(base_dir)
    if configs_dir:
        add_config_search_path(configs_dir)

    # Resolve defaults (layered composition)
    defaults = cfg.pop(_DEFAULTS_KEY, None)
    if defaults:
        merged = OmegaConf.create({})
        for default_ref in defaults:
            if isinstance(default_ref, str):
                layer_path = _resolve_config_path(default_ref, configs_dir or base_dir)
                layer_cfg = OmegaConf.load(layer_path)
                merged = OmegaConf.merge(merged, layer_cfg)
            elif isinstance(default_ref, (dict, DictConfig)):
                # Check if this is a layer reference like {"/trainer": "sft"}
                items = (
                    default_ref.items()
                    if isinstance(default_ref, dict)
                    else OmegaConf.to_container(default_ref).items()
                )
                layer_loaded = False
                for key, value in items:
                    if isinstance(key, str) and key.startswith("/"):
                        ref_str = f"{key}: {value}"
                        layer_path = _resolve_config_path(
                            ref_str, configs_dir or base_dir
                        )
                        layer_cfg = OmegaConf.load(layer_path)
                        merged = OmegaConf.merge(merged, layer_cfg)
                        layer_loaded = True
                if not layer_loaded:
                    # Inline dict as part of defaults
                    merged = OmegaConf.merge(merged, OmegaConf.create(default_ref))
        # The experiment config overrides the defaults
        cfg = OmegaConf.merge(merged, cfg)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def _find_configs_dir(start_dir: Path) -> Optional[Path]:
    """Walk up to find a 'configs' directory."""
    current = start_dir.resolve()
    for _ in range(10):
        candidate = current / "configs"
        if candidate.is_dir():
            return candidate
        # Also check for leap/configs
        candidate2 = current / "leap" / "configs"
        if candidate2.is_dir():
            return candidate2
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _import_target(target: str) -> Any:
    """Import a class or function from a dotted path.

    Supports both module.Class and module.Class.method patterns.

    Examples:
        "leap.models.base_vla.BaseVLAModel" -> BaseVLAModel class
        "transformers.AutoModelForCausalLM.from_pretrained" -> from_pretrained method
    """
    parts = target.rsplit(".", 1)
    if len(parts) == 1:
        return importlib.import_module(parts[0])

    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        # Try deeper split: module.Class.method
        parts2 = module_path.rsplit(".", 1)
        if len(parts2) == 2:
            module = importlib.import_module(parts2[0])
            cls = getattr(module, parts2[1])
            return getattr(cls, attr_name)
        raise


def instantiate(cfg: Any, **kwargs: Any) -> Any:
    """Recursively instantiate an object from a config with _target_.

    Args:
        cfg: A DictConfig/dict with a _target_ key, or a plain value.
        **kwargs: Additional keyword arguments to pass to the constructor.

    Returns:
        The instantiated object, or the original value if no _target_.
    """
    if cfg is None:
        return None

    # Plain values pass through
    if not isinstance(cfg, (dict, DictConfig)):
        if isinstance(cfg, (list, ListConfig)):
            return [instantiate(item) for item in cfg]
        return cfg

    cfg = deepcopy(cfg) if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)

    if _TARGET_KEY not in cfg:
        # Recursively instantiate nested specs
        return {k: instantiate(v) for k, v in cfg.items()}

    target = cfg.pop(_TARGET_KEY)
    cls_or_fn = _import_target(target)

    # Recursively instantiate nested config values
    resolved_kwargs = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue  # Skip meta keys
        resolved_kwargs[k] = instantiate(v)

    # Merge with explicit kwargs
    resolved_kwargs.update(kwargs)

    return cls_or_fn(**resolved_kwargs)


def to_yaml(cfg: DictConfig) -> str:
    """Convert a DictConfig to a YAML string."""
    return OmegaConf.to_yaml(cfg, resolve=True)


def to_container(cfg: DictConfig) -> Dict:
    """Convert a DictConfig to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)
