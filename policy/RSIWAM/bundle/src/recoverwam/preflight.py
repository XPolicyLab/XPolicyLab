import json
import os
from pathlib import Path
import torch


def expected_contract(
    stage: str,
    video_size=None,
    *,
    video_frames: int = 17,
    action_horizon: int = 32,
    action_dim: int = 7,
    proprio_steps: int = 1,
    proprio_dim: int = 8,
):
    stage = str(stage).strip().lower()
    if stage not in {"video", "action", "alternating", "joint"}:
        raise ValueError(f"Unsupported preflight stage: {stage}")
    if video_size is None:
        video_size = [224, 448]
    video_size = [int(value) for value in video_size]
    if len(video_size) != 2 or any(value <= 0 for value in video_size):
        raise ValueError(f"`video_size` must contain two positive integers, got {video_size}")
    dimensions = {
        "video_frames": int(video_frames),
        "action_horizon": int(action_horizon),
        "action_dim": int(action_dim),
        "proprio_steps": int(proprio_steps),
        "proprio_dim": int(proprio_dim),
    }
    invalid = {key: value for key, value in dimensions.items() if value <= 0}
    if invalid:
        raise ValueError(f"Preflight dimensions must be positive, got {invalid}")
    return {
        "video": [3, dimensions["video_frames"], *video_size],
        "action": [dimensions["action_horizon"], dimensions["action_dim"]],
        "proprio": [dimensions["proprio_steps"], dimensions["proprio_dim"]],
    }


def validate_batch_contract(
    sample,
    stage: str,
    video_size=None,
    *,
    video_frames: int = 17,
    action_horizon: int = 32,
    action_dim: int = 7,
    proprio_steps: int = 1,
    proprio_dim: int = 8,
):
    contract = expected_contract(
        stage,
        video_size=video_size,
        video_frames=video_frames,
        action_horizon=action_horizon,
        action_dim=action_dim,
        proprio_steps=proprio_steps,
        proprio_dim=proprio_dim,
    )
    observed = {}
    for key, expected_shape in contract.items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Preflight sample field `{key}` must be a tensor, got {type(value)}")
        if value.ndim != len(expected_shape) + 1:
            raise ValueError(
                f"Preflight `{key}` must include one batch dimension, "
                f"got shape {tuple(value.shape)}"
            )
        observed_shape = list(value.shape[1:])
        if observed_shape != expected_shape:
            raise ValueError(
                f"Preflight `{key}` shape mismatch: expected [B,{expected_shape}], "
                f"got {list(value.shape)}"
            )
        observed[key] = observed_shape
    return observed


def trainable_parameter_manifest(model):
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return {
        "names": [name for name, _ in parameters],
        "num_tensors": len(parameters),
        "num_parameters": sum(parameter.numel() for _, parameter in parameters),
    }


def assert_finite_gradients(model):
    non_finite = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            non_finite.append(name)
    if non_finite:
        raise FloatingPointError(f"Non-finite gradients in parameters: {non_finite[:20]}")


def write_json_atomic(payload, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)


def validate_preflight_config(cfg):
    probe = cfg.get("preflight")
    if not probe or not probe.get("enabled", False):
        return None
    if cfg.train_mode != "joint":
        raise ValueError("Only joint preflight is supported")
    return dict(
        stage="joint",
        expected_steps=int(probe.get("expected_steps", cfg.max_steps)),
        memory_headroom_fraction=float(probe.get("memory_headroom_fraction", 0.05)),
        output_file=str(probe.get("output_file", str(Path(cfg.output_dir) / "probe.json"))),
        video_size=list(cfg.data.train.video_size),
        contract_kwargs=dict(
            video_frames=17, action_horizon=32, action_dim=16, proprio_steps=1, proprio_dim=16
        ),
    )
