from pathlib import Path
from typing import Dict, Any, Union

from omegaconf import OmegaConf as om  # type: ignore[import-not-found]


def _resolve_config_path(yaml_name: str, repo_root: Path) -> Path:
    """Resolve config path by trying multiple locations in order.

    Search order:
    1. configs/<yaml_name> (if yaml_name already has path like "experiments/xxx.yaml")
    2. configs/experiments/<yaml_name> (auto-try experiments subdirectory)
    3. launch_scripts/<yaml_name> (backward compatibility)
    """
    configs_dir = repo_root / "configs"

    

    # 1. Auto-try configs/experiments/ (for simple names like "libero_simulation.yaml")
    experiments_path = configs_dir / "experiments" / yaml_name
    if experiments_path.exists():
        return experiments_path

    # 2. Try direct path under configs/
    path = configs_dir / yaml_name
    if path.exists():
        return path

    # Not found anywhere - raise error with helpful message
    raise FileNotFoundError(
        f"Config file '{yaml_name}' not found. Tried:\n"
        f"  - {path}\n"
        f"  - {experiments_path}\n"
        f"Please ensure the config file exists in one of these locations."
    )


def read_vla_yaml_config(yaml_name: str) -> Dict[str, Any]:
    """Read VLA config with auto-discovery in configs/ directory.

    Automatically searches in multiple locations:
    - configs/experiments/<yaml_name>
    - configs/<yaml_name>
    

    Supports both single config files and combined configs that reference
    separate model and dataset configs.

    Args:
        yaml_name: Config file name (e.g., "libero_simulation.yaml")
                   or relative path (e.g., "experiments/libero_simulation.yaml")

    Returns:
        Merged configuration dict with both 'model' and 'datasets' sections
    """
    assert yaml_name is not None, "yaml_name is not set"

    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "configs"

    # Resolve the actual config file path
    yaml_path = _resolve_config_path(yaml_name, repo_root)
    cfg = om.load(str(yaml_path))

    # Check if this is a combined config with model_config and dataset_config references
    if isinstance(cfg, dict) or hasattr(cfg, 'keys'):
        cfg_dict = om.to_object(cfg)

        # Handle combined config format
        if 'model_config' in cfg_dict or 'dataset_config' in cfg_dict:
            merged_cfg = {}

            # Load model config if specified
            if 'model_config' in cfg_dict:
                model_path = configs_dir / cfg_dict['model_config']
                if not model_path.exists():
                    raise FileNotFoundError(f"Model config not found: {model_path}")
                model_cfg = om.load(str(model_path))
                merged_cfg['model'] = om.to_object(model_cfg)

            # Load dataset config if specified
            if 'dataset_config' in cfg_dict:
                dataset_path = configs_dir / cfg_dict['dataset_config']
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset config not found: {dataset_path}")
                dataset_cfg = om.load(str(dataset_path))
                merged_cfg['datasets'] = om.to_object(dataset_cfg)

            return merged_cfg

    # Return as-is for simple configs (backward compatible)
    return om.to_object(cfg)


def resolve_config_path(yaml_name: str) -> Path:
    """Resolve config path by trying multiple locations."""
    repo_root = Path(__file__).resolve().parents[2]
    return _resolve_config_path(yaml_name, repo_root)
