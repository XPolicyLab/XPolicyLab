from pathlib import Path
import os

def asset_path(name):
    root=Path(os.environ["CRWM_ASSETS_ROOT"]).expanduser().resolve()
    names={"vae":"vae.safetensors","dino":"dino.safetensors"}
    path=root/names[name]
    if not path.is_file():raise FileNotFoundError(f"Missing local {name} asset: {path}")
    return path
