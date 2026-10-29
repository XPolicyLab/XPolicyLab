"""External model/data paths and bundled Robotwin normalization statistics."""
import json
import os
from pathlib import Path


def robotwin_stats_path() -> str:
    return str(Path(__file__).resolve().parents[1] / 'assets' / 'stats' / 'robotwin.json')


def path_from_env(name: str, *, required: bool = True) -> str:
    value = os.environ.get(name, '').strip()
    if required and not value:
        raise ValueError(f'Set {name} in configs/runtime.conf')
    return value


def paths_from_env(name: str) -> list[str]:
    values = json.loads(os.environ.get(name, '[]'))
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f'{name} must be a JSON array of path strings')
    return values


def inference_weight_path(name: str, checkpoint: str, relative: str) -> str:
    """Derive initialization from the resolved checkpoint, never CKPT_PATH."""
    explicit = path_from_env(name, required=False)
    if explicit:
        return explicit
    if not checkpoint:
        raise ValueError(f'Select an inference checkpoint before resolving {name}')
    return str(Path(checkpoint).expanduser().resolve() / relative)
