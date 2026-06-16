import math
from pathlib import Path
from typing import Any, List, Callable, Optional
from omegaconf import OmegaConf


def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)

def _oc_load(path: str, key: Optional[str] = None) -> Any:
    """
    Load a YAML/JSON config and optionally select a key.
    Uses Hydra's to_absolute_path to honor original working dir.
    """
    try:
        from hydra.utils import to_absolute_path
    except ImportError:
        to_absolute_path = None  # should not happen in normal Hydra runs
    load_path = Path(path)
    if not load_path.is_absolute() and to_absolute_path is not None:
        load_path = Path(to_absolute_path(path))
    cfg = OmegaConf.load(load_path)
    if key is None or key == "":
        return cfg
    return OmegaConf.select(cfg, key)

def sum_shapes(shape_meta_list):
    if not shape_meta_list:
        return 0
    total = sum(int(item['shape']) for item in shape_meta_list if item["key"] is not None)
    return total

def max_action_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            action_cfg = dataset_cfg.shape_meta.action
            current_dim = sum_shapes(action_cfg)
            
            if current_dim > max_dim:
                max_dim = current_dim
                
    return max_dim

def max_state_dim(embodiment_datasets_cfg):
    max_dim = 0
    for dataset_name, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            state_cfg = dataset_cfg.shape_meta.state
            current_dim = sum_shapes(state_cfg)
            if current_dim > max_dim:
                max_dim = current_dim
                
    return max_dim

def register_default_resolvers() -> None:
    """
    Register all resolvers commonly used across entrypoints.
    Safe to call multiple times.
    """
    _register("oc.load", _oc_load)
    _register("eval", eval) # allows arbitrary python code execution in configs using the ${eval:''} resolver
    _register("split", lambda s, idx: s.split('/')[int(idx)]) # split string
    _register("max", lambda x: max(x))
    _register("round_up", math.ceil)
    _register("round_down", math.floor)
    _register("sum_shapes", sum_shapes)
    _register("max_action_dim", max_action_dim)
    _register("max_state_dim", max_state_dim)
