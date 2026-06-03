import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import OmegaConf

# /source/training/utils/ -> parents[3] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Load config from source/config/execution_module_train.yaml
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "source" / "config" / "execution_module_train.yaml"


def load_config_with_train_variant(task_override: Optional[str] = None) -> Tuple:
    """
    Load config from source/config/execution_module_train.yaml.
    Note: task_override parameter is currently unused but kept for API compatibility.
    Returns (cfg, config_path, config_name).
    """
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file {DEFAULT_CONFIG_PATH} not found.")
    cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    config_name = "execution_module_train"
    return cfg, DEFAULT_CONFIG_PATH, config_name


def parse_args():
    parser = argparse.ArgumentParser(description="MemoryMatters execution module training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (defaults to source/config/execution_module_train.yaml)",
    )
    return parser.parse_args()
