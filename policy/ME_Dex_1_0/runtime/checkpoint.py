import json
from pathlib import Path

import torch


def load_metadata(export_dir):
    root = Path(export_dir).expanduser()
    config = json.loads((root / "model_config.json").read_text())
    tactile_path = Path(config["model"]["tactile"]["checkpoint_path"])
    return {
        "config": config,
        "model_path": root / "model.pt",
        "tactile_checkpoint": root / tactile_path,
    }


def load_model_state(model, metadata):
    checkpoint = torch.load(
        metadata["model_path"], map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint["module"])
