"""Read earlier released metadata without modifying model files."""
from pathlib import Path
import json

_ADAPTER_NAMES = ('era0_adapter.json', 'hy_xvla_adapter.json')

def adapter_path(model):
    model = Path(model)
    return next((model / name for name in _ADAPTER_NAMES if (model / name).is_file()), model / _ADAPTER_NAMES[0])


def normalize_metadata(value):
    if isinstance(value, dict):
        return {normalize_metadata(k): normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_metadata(v) for v in value]
    if isinstance(value, str):
        for old, new in (('hy_xvla', 'era0'), ('HyXVLA', 'Era0Policy'),
                         ('XVLAProcessor', 'Era0Processor'), ('XVLAConfig', 'Era0Config'),
                         ('XVLA', 'Era0'), ('xvla', 'era0'), ('hy_vla', 'era0'), ('hy_', 'era0_')):
            value = value.replace(old, new)
    return value


def read_metadata(path):
    return normalize_metadata(json.loads(Path(path).read_text(encoding='utf-8')))
