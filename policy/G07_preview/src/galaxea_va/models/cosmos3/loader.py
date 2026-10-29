"""Load the published Cosmos3-Nano diffusers-layout transformer (814 keys) into the
vendored :class:`Cosmos3Backbone`.

The generation tower (``*_moe_gen`` / ``add_*`` / ``norm_added_*`` / ``norm_moe_gen``)
is already trained in the release (und->gen init_moe is baked in), so no re-copy is
needed. Everything action/state related is FRESH-INIT and retrained from scratch:
``action_in`` / ``action_out`` (a single shared Linear over the padded action width),
``proprio_encoder``, and ``action_modality_embed`` (re-zeroed after load — it is coupled
to the action projection we retrain). Dropped (in ckpt, unused here): ``lm_head``,
``audio_*``, and the released ``action_proj_in/out.*`` DomainAware heads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from .transformer import Cosmos3Backbone, Cosmos3BackboneConfig

# Keys present in the checkpoint that we intentionally do NOT load.
_EXPECTED_UNEXPECTED = {
    "lm_head.weight",
    "audio_proj_in.weight",
    "audio_proj_in.bias",
    "audio_proj_out.weight",
    "audio_proj_out.bias",
    "audio_modality_embed",
    "action_proj_in.fc.weight",
    "action_proj_in.bias.weight",
    "action_proj_out.fc.weight",
    "action_proj_out.bias.weight",
}
# Prefixes of the fresh (not-in-ckpt) fastwam heads. Concrete param names are derived
# from the model at load time.
_FRESH_PREFIXES = ("action_in", "action_out", "proprio_encoder")

COSMOS3_NANO_ROOT = os.environ.get(
    "COSMOS3_NANO_ROOT",
    str(Path(__file__).resolve().parents[4] / "checkpoints" / "Cosmos3-Nano"),
)


def _read_transformer_config(root: str) -> dict:
    with open(os.path.join(root, "transformer", "config.json")) as f:
        return json.load(f)


def load_cosmos3_transformer(
    root: str,
    *,
    action_dim: int,
    proprio_dim: int,
    action_path: str = "cogen",
    action_hidden: int = 1024,
    action_ffn: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Cosmos3Backbone:
    """Build a :class:`Cosmos3Backbone` and load the released transformer weights.

    Builds on the ``meta`` device to avoid the (very slow) default fp32 init of the
    14.5B params, materializes empty tensors directly on ``device`` in ``dtype``,
    then copies the checkpoint values in. Fresh-init (retrained from scratch):
    ``proprio_encoder`` plus, for ``action_path='cogen'``, ``action_in`` / ``action_out``
    / ``action_modality_embed``; for ``action_path='split'``, the whole ``action_expert``.
    """
    cfg_json = _read_transformer_config(root)
    cfg = Cosmos3BackboneConfig.from_transformer_config_json(
        cfg_json, action_dim=action_dim, proprio_dim=proprio_dim,
        action_path=action_path, action_hidden=action_hidden, action_ffn=action_ffn,
    )
    # meta build (no memory, no init) -> empty on device in target dtype
    with torch.device("meta"):
        model = Cosmos3Backbone(cfg)
    model = model.to(dtype)
    model = model.to_empty(device=device)

    # Load checkpoint (bf16) shards.
    tdir = os.path.join(root, "transformer")
    index_path = os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(os.path.join(tdir, shard), device=str(device)))

    missing, unexpected = model.load_state_dict(state, strict=False, assign=False)
    missing = {m for m in missing if not m.endswith("rotary_emb.inv_freq")}

    # Fresh prefixes + expected-unexpected depend on the action pathway.
    if action_path == "cogen":
        fresh_prefixes = ("action_in", "action_out", "proprio_encoder")
        expected_unexpected = set(_EXPECTED_UNEXPECTED)
    elif action_path == "split":
        # `action_expert` replaces the shared action heads; ckpt's top-level
        # `action_modality_embed` has no home in the split model (it lives inside the expert).
        fresh_prefixes = ("action_expert", "proprio_encoder")
        expected_unexpected = set(_EXPECTED_UNEXPECTED) | {"action_modality_embed"}
    else:
        raise ValueError(f"action_path must be 'cogen' or 'split', got {action_path!r}")

    expected_missing = {n for n, _ in model.named_parameters() if n.startswith(fresh_prefixes)}
    unexpected_set = set(unexpected)
    missing_set = set(missing)
    if strict:
        if unexpected_set != expected_unexpected:
            raise RuntimeError(
                "Cosmos3 checkpoint unexpected-key mismatch.\n"
                f"  extra unexpected: {sorted(unexpected_set - expected_unexpected)}\n"
                f"  missing from unexpected: {sorted(expected_unexpected - unexpected_set)}"
            )
        if missing_set != expected_missing:
            raise RuntimeError(
                "Cosmos3 checkpoint missing-key mismatch.\n"
                f"  extra missing: {sorted(missing_set - expected_missing)}\n"
                f"  unexpectedly present: {sorted(expected_missing - missing_set)}"
            )

    # Fresh-init (materialized empty by ``to_empty``).
    torch.nn.init.xavier_uniform_(model.proprio_encoder.weight)
    torch.nn.init.zeros_(model.proprio_encoder.bias)
    if action_path == "cogen":
        torch.nn.init.xavier_uniform_(model.action_in.weight)
        torch.nn.init.zeros_(model.action_in.bias)
        torch.nn.init.zeros_(model.action_out.weight)  # start from ~identity velocity
        torch.nn.init.zeros_(model.action_out.bias)
        # Discard the pretrained action modality marker (coupled to the retrained projection).
        torch.nn.init.zeros_(model.action_modality_embed)
    else:
        model.action_expert.reset_parameters()

    return model
