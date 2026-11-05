"""Local source, model assets and writable caches for MR0 inference."""
import os
import sys
from pathlib import Path


def install_paths(assets, runs=None):
    policy = Path(__file__).resolve().parent
    assets = Path(assets).expanduser().resolve()
    runs = Path(runs or policy / 'runs').expanduser().resolve()
    paths = dict(
        assets=assets, runs=runs, source=policy / 'source_starvla',
        checkpoint=assets / 'Meituan_Robotics_0/model.ckpt',
        base_vlm=assets / 'base_vlm/Qwen3.5-4B-Action',
        teacher=assets / 'cdlam/model.pt', teacher_code=policy / 'source_cdlam',
    )
    cache = runs / 'cache'
    for key, folder in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                        'XDG_CACHE_HOME': 'xdg', 'TRITON_CACHE_DIR': 'triton',
                        'TORCH_EXTENSIONS_DIR': 'torch_extensions'}.items():
        directory = cache / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    (cache / 'xdg/torch/kernels').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(paths['source']))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    return paths
