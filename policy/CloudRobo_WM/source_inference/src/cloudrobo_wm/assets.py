from pathlib import Path

ASSET_NAMES={
    'text_encoder':'text_encoder.safetensors',
    'vae':'vae.safetensors',
    'dino':'dino.safetensors',
    'tokenizer':'tokenizer',
}
_ROOT=None

def configure_assets(root):
    global _ROOT
    _ROOT=Path(root).expanduser().resolve()
    return validate_assets()

def asset_path(name):
    if _ROOT is None:
        raise RuntimeError('Set the local asset directory before constructing the policy')
    path=_ROOT/ASSET_NAMES[name]
    if name=='tokenizer':
        if not path.is_dir() or not (path/'tokenizer_config.json').is_file():
            raise FileNotFoundError('The local tokenizer directory is incomplete')
    elif not path.is_file():
        raise FileNotFoundError('A required local encoder asset is missing: '+name)
    return path

def validate_assets():
    return {name:asset_path(name) for name in ASSET_NAMES}
