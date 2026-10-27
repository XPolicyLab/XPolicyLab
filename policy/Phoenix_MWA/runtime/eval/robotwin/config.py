"""Portable evaluation paths; training datasets/caches are not prerequisites."""
import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/inference/phoenix_robotwin.yaml'
DEFAULT_STATS = ROOT / 'assets/stats/robotwin.json'


def resource_path(value):
    """Interpret relative model/config paths from the checkout, not the caller CWD."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=True)


def inference_paths(config=None, stats=None):
    # The old environment name remains an explicit compatibility fallback.
    config = resource_path(config or os.environ.get('ROBOTWIN_INFERENCE_CONFIG')
                           or os.environ.get('ROBOTWIN_TRAIN_CONFIG') or DEFAULT_CONFIG)
    cfg = OmegaConf.load(config)
    stats = resource_path(stats or os.environ.get('ROBOTWIN_EVAL_STATS')
                          or OmegaConf.select(cfg, 'inference.normalization_stats') or DEFAULT_STATS)
    if not config.is_file() or not stats.is_file():
        raise ValueError('RoboTwin inference config and normalization must be files')
    return config, stats
