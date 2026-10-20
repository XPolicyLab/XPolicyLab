"""Load the frozen CR-WM architecture and one native RoboTwin checkpoint."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from .processing import JointEefNormalizer
from .processing.timing import inference_time_grids
from .assets import validate_assets


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_runtime(checkpoint_path: str | Path, *, model_config_path: str | Path | None = None,
                 normalization_path: str | Path | None = None, device: str = "cuda"):
    """Validate statistics before allocating GPU modules, then load strict weights."""
    from .models.crwm import CRWM

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CR-WM checkpoint not found: {checkpoint}")
    validate_assets()
    config = json.loads(Path(model_config_path or CONFIG_ROOT / "model.json").read_text(encoding="utf-8"))
    if config.pop('observation_ablation') != 'pointmap_zero_latent':
        raise ValueError('CR-WM uses a fixed current pointmap zero latent')
    normalizer = JointEefNormalizer(normalization_path or CONFIG_ROOT / "normalization.json")
    # Only the small contract is retained. Model construction never keeps a
    # second full CPU copy of the checkpoint while GPU modules are allocated.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    contract = payload["multisource_contract"]
    condition_id = normalizer.validate_checkpoint(contract)
    if payload['pointmap_encoding'] != config['pointmap_encoding']:
        raise ValueError('Pointmap prediction contract differs from the release')
    time_protocol = next((entry["source"].get("time_protocol") for entry in contract.get("sources", [])
                          if entry["source"]["stats_key"] == normalizer.stats_key), None)
    del payload
    gc.collect()
    condition_count = config.pop("condition_count")
    model = CRWM.from_wan22_pretrained(**config, device=device, torch_dtype=torch.bfloat16)
    model.init_multisource(condition_count=condition_count)
    model.bind_multisource_contract(contract)
    model.load_checkpoint(checkpoint)
    model.eval()
    model.prepare_for_inference()
    grids, valid = inference_time_grids(time_protocol)
    time_kwargs = dict(time_grid={k: torch.from_numpy(v)[None].to(device) for k, v in grids.items()},
                       time_valid={k: torch.from_numpy(v)[None].to(device) for k, v in valid.items()})
    return model, normalizer, condition_id, time_kwargs
