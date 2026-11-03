"""Initialize Action2B and export portable inference bundles from joint weights."""

import json
import os
from pathlib import Path
import shutil

import torch
from omegaconf import OmegaConf

from .weight_resize import _resize_tensor_to_shape, _materialize_compact_tensor
from .checkpoint_compat import ACTION_FORMAT, canonicalize_config


def _atomic_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_action(video, config, output, *, expected_step=10000):
    from .models.wan21_14b.action_dit import ActionDiT

    cfg = OmegaConf.load(config)
    model = cfg.model
    action_cfg = OmegaConf.to_container(model.action_dit_config, resolve=True)
    source = torch.load(video, map_location="cpu", mmap=True, weights_only=True)
    step = source.get("step")
    state = source.get("dit")
    if state is None and "mot" in source:
        state = {
            k.removeprefix("mixtures.video."): v
            for k, v in source["mot"].items()
            if k.startswith("mixtures.video.")
        }
    if not state or step != expected_step:
        raise ValueError(f"RoboDojo initialization requires a history-conditioned Video step-{expected_step} checkpoint")
    with torch.device("meta"):
        expert = ActionDiT(**action_cfg)
    target = expert.state_dict()
    for key in ("num_heads", "attn_head_dim", "num_layers"):
        if int(action_cfg[key]) != int(model.video_dit_config[key]):
            raise ValueError(f"Video/Action {key} mismatch")
    backbone = {}
    for key in sorted(ActionDiT.backbone_key_set(target.keys())):
        if key not in state:
            raise ValueError(f"Missing source backbone tensor: {key}")
        src = state[key].to(torch.bfloat16)
        value = _resize_tensor_to_shape(src, tuple(target[key].shape))
        if src.ndim >= 2 and src.shape[-1] != target[key].shape[-1]:
            value = value.float() * (src.shape[-1] / target[key].shape[-1]) ** 0.5
        backbone[key] = _materialize_compact_tensor(value, dtype=torch.bfloat16)
    payload = dict(
        format="wan21_action_v1",
        backbone_state_dict=backbone,
        meta=action_cfg,
        policy=dict(
            source_step=step,
            model_family="wan21_14b",
            alpha_scaling=True,
            interpolation="sequential_1d_linear_align_corners_true",
            skip_prefixes=list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            projection_initialization="random",
        ),
    )
    _atomic_save(payload, output)
    return dict(tensors=len(backbone), parameters=sum(t.numel() for t in backbone.values()))


def deployment_config(cfg):
    """Serialize only inference architecture/preprocessing, without training paths."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model, data = cfg["model"], cfg["data"]["train"]
    model["_target_"] = "recoverwam.runtime.create_recoverwam_wan21_14b"
    model.update(
        model_id="WAN_MODEL_DIR",
        tokenizer_model_id=None,
        dit_checkpoint_path="video.pt",
        action_dit_pretrained_path=None,
        load_text_encoder=True,
    )
    keep = (
        "processor",
        "video_size",
        "video_resize_mode",
        "num_frames",
        "action_video_freq_ratio",
        "action_horizon",
        "condition_history_frames",
        "condition_history_stride",
        "condition_include_episode_first",
        "condition_history_target_prefix",
        "condition_observation_separate",
    )
    data = {k: data[k] for k in keep if k in data}
    # Accept old training configs for export, but never import the old namespace.
    value = dict(model=model, data=dict(train=data))
    return canonicalize_config(value)


def export(checkpoint, config, stats, output, step):
    cfg = OmegaConf.load(config)
    if cfg.train_mode != "joint" or cfg.model.action_dit_config.hidden_dim != 1024:
        raise ValueError("Expected the joint + Action2B training configuration")
    source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if int(source.get("step", -1)) != step or "mot" not in source:
        raise ValueError("Full checkpoint step/format mismatch")
    mot = source["mot"]
    if any(not k.startswith(("mixtures.video.", "mixtures.action.")) for k in mot):
        raise ValueError("Unexpected expert keys in full checkpoint")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh bundle directory: {output}")
    output.mkdir(parents=True)

    def extract(prefix):
        tensors = {
            k[len(prefix) :]: _materialize_compact_tensor(v, dtype=v.dtype)
            for k, v in mot.items()
            if k.startswith(prefix)
        }
        if not tensors:
            raise ValueError(f"Empty expert: {prefix}")
        return tensors

    video = extract("mixtures.video.")
    _atomic_save(dict(format="wan21_video_v1", step=step, dit=video), output / "video.pt")
    del video
    action = extract("mixtures.action.")
    proprio = {
        k: _materialize_compact_tensor(v, dtype=v.dtype)
        for k, v in source.get("proprio_encoder", {}).items()
    }
    if not proprio:
        raise ValueError("Missing proprio encoder")
    _atomic_save(
        dict(
            format=ACTION_FORMAT,
            step=step,
            action_expert=action,
            proprio_encoder=proprio,
            video_checkpoint={"path": "video.pt"},
            resolved_config=deployment_config(cfg),
        ),
        output / "action.pt",
    )
    shutil.copyfile(stats, output / "dataset_stats.json")
    manifest = dict(
        format="recoverwam-robodojo-bundle-v1",
        step=step,
        files={p.name: p.stat().st_size for p in output.iterdir()},
        inference=dict(steps=10, shift=1, horizon=32, replan=24, seed=1),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
