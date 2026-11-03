"""Standalone RoboDojo Action-LoRA, with frozen inference-equivalent video input.

Load the official paired checkpoint with RoboDojoRecoverWAMPolicy first. This module
does not load base weights, change normalizers, train video, or consume future
images. ``prepare_condition`` accepts the exact nine-frame anchor/history RGB
window (or its temporal VAE encoding), and always builds video KV from noise.

Typical calls::

    install_action_lora(policy, rank=8, alpha=8)
    condition = prepare_condition(policy, inputs, video_seed=1000004)
    loss, metrics = action_flow_loss(policy, condition, actions, valid_mask)
    loss.backward()
    prediction = infer_from_condition(policy, condition, action_seed=1)

Actions/proprio must already use the OFFICIAL checkpoint's normalizer. RGB must
use official T-shape composition and [-1,1] normalization. A true mask entry
means that the complete EE16 action at that frame is a verified training label.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint  # Official MoT accesses this attribute directly.


FORMAT = "recoverwam-robodojo-action-lora-v1"


class LoRALinear(nn.Module):
    """Frozen Linear plus FP32 low-rank weights, with an exactly zero start."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA requires Linear, rank > 0, and finite alpha > 0")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if not self.enabled:
            return result
        # Disabling autocast here keeps the small optimizer tensors and their
        # matrix products FP32 even if the surrounding model runs in BF16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return result + update.to(dtype=result.dtype)


def _model(policy):
    return getattr(policy, "model", policy)


def lora_modules(policy) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in _model(policy).action_expert.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameters(policy) -> Iterator[nn.Parameter]:
    for _, module in lora_modules(policy):
        yield module.lora_A
        yield module.lora_B


def _assert_robodojo(policy) -> None:
    model = _model(policy)
    if model.raw_action_dim != 16 or model.proprio_dim != 16:
        raise ValueError("Only official EE16 RoboDojo action/proprio are supported")
    if getattr(policy, "action_horizon", 32) != 32 or 32 % model.action_chunk_size:
        raise ValueError("Expected official 32-frame action horizon")
    for key in ("action_clean_first_frame_start_layer", "action_clean_first_frame_timestep_cutoff", "action_video_timestep_cutoff", "use_fixed_video_layers"):
        if getattr(model, key, None) is not None:
            raise ValueError(f"This official-checkpoint experiment excludes {key}")
    if getattr(model, "use_video_tokens", False) or getattr(model, "joint_denoising", False):
        raise ValueError("Only direct per-layer frozen video KV is supported")
    if model.train_video_scheduler.train_sampling_strategy != "pure_noise":
        raise ValueError("Expected the official joint100k pure-noise video scheduler")
    if model.mot.mixtures["action"] is not model.action_expert:
        raise ValueError("MoT and policy must share the same action expert")


def set_lora_train_mode(policy) -> None:
    """Enable action checkpointing without entering unsupported joint trainer.

    The outer official model must remain in eval mode because its frozen-video
    helper rejects joint training. eval does not disable autograd. Only MoT and
    ActionDiT need training=True for their non-reentrant checkpoints.
    """
    model = _model(policy)
    model.eval()
    model.mot.train()
    model.video_expert.eval()
    model.action_expert.train()


def install_action_lora(policy, rank: int = 8, alpha: float = 8.0, gradient_checkpointing: bool = True) -> dict[str, Any]:
    _assert_robodojo(policy)
    model = _model(policy)
    if list(lora_modules(policy)):
        raise ValueError("Action LoRA is already installed")
    model.requires_grad_(False)
    targets = []
    for i, block in enumerate(model.action_expert.blocks):
        for name in ("q", "v"):
            original = getattr(block.self_attn, name)
            setattr(block.self_attn, name, LoRALinear(original, rank=rank, alpha=alpha))
            targets.append(f"blocks.{i}.self_attn.{name}")
    model.action_expert.use_gradient_checkpointing = bool(gradient_checkpointing)
    model.mot.mot_checkpoint_mixed_attn = bool(gradient_checkpointing)
    config = {"format": FORMAT, "rank": int(rank), "alpha": float(alpha), "targets": targets,
              "gradient_checkpointing": bool(gradient_checkpointing), "raw_action_dim": 16,
              "action_horizon": 32, "history_frames": 8, "history_stride": 25,
              "trainable_parameters": sum(p.numel() for p in lora_parameters(policy))}
    model._action_lora_config = config
    set_lora_train_mode(policy)
    assert_only_lora_trainable(policy)
    return dict(config)


def assert_only_lora_trainable(policy) -> None:
    model = _model(policy)
    adapters = {id(p) for p in lora_parameters(policy)}
    if not adapters:
        raise RuntimeError("No action adapters installed")
    unexpected = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in adapters]
    missing = [n for n, p in model.named_parameters() if id(p) in adapters and not p.requires_grad]
    if unexpected or missing:
        raise RuntimeError(f"Invalid trainable tensors: unexpected={unexpected[:5]}, disabled_adapters={missing[:5]}")


def _validate_history(inputs: dict[str, Any]) -> int:
    if "history_indices" not in inputs or "current_indices" not in inputs:
        raise ValueError("history_indices and current_indices are required to verify no future observations")
    history = torch.as_tensor(inputs["history_indices"], dtype=torch.int64, device="cpu")
    current = torch.as_tensor(inputs["current_indices"], dtype=torch.int64, device="cpu").reshape(-1)
    if history.ndim == 1:
        history = history.unsqueeze(0)
    if history.shape != (len(current), 9) or (current < 0).any():
        raise ValueError("Expected history_indices[B,9] and nonnegative current_indices[B]")
    offsets = torch.arange(-175, 1, 25, dtype=torch.int64)
    expected = torch.cat([torch.zeros(len(current), 1, dtype=torch.int64), (current[:, None] + offsets).clamp_min(0)], dim=1)
    if not torch.equal(history, expected):
        raise ValueError("History must be [episode anchor, t-175, t-150, ..., t] with episode-start padding")
    return len(current)


def _to_model(model, value, *, dtype=None):
    # clone avoids inference-mode cache tensors being saved for action backward.
    return torch.as_tensor(value).detach().to(device=model.device, dtype=dtype or model.torch_dtype).clone()


@torch.no_grad()
def encode_condition_features(policy, prepared_inputs: dict[str, Any], *, tiled: bool = False) -> dict[str, Any]:
    """Encode small reusable features; never retain the ~1.5 GiB/window video KV.

    Required: history_indices[B,9], current_indices[B], normalized proprio[B,16],
    and prompt (already DEFAULT_PROMPT formatted) OR context/context_mask.
    Visuals: condition_video[B,3,9,240,416] OR condition_latents[B,16,3,30,52]
    plus clip_fea. A cached context is TEXT ONLY: proprio is appended here once.
    Returned context includes proprio, so feed this object to prepare_condition
    via its ``features`` keyword, not back into prepared_inputs.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Use torch.no_grad(), not inference_mode(), when preparing reusable training features")
    _assert_robodojo(policy)
    model = _model(policy)
    batch = _validate_history(prepared_inputs)
    forbidden = {"future_video", "target_video", "input_video", "video", "condition_observation_video"} & prepared_inputs.keys()
    if forbidden:
        raise ValueError(f"Future/alternate observation fields are forbidden: {sorted(forbidden)}")
    model.eval()
    if "condition_latents" in prepared_inputs:
        if "condition_video" in prepared_inputs:
            raise ValueError("Provide history RGB or cached temporal latents, not both")
        condition_latents = _to_model(model, prepared_inputs["condition_latents"])
        clip_fea = _to_model(model, prepared_inputs["clip_fea"])
    else:
        history = _to_model(model, prepared_inputs["condition_video"])
        if tuple(history.shape) != (batch, 3, 9, 240, 416):
            raise ValueError(f"Expected official T-shape history [B,3,9,240,416], got {tuple(history.shape)}")
        if not torch.isfinite(history).all() or history.min() < -1.001 or history.max() > 1.001:
            raise ValueError("History must be finite normalized RGB in [-1,1]")
        condition_latents = model._encode_separate_condition_latents(history, tiled=tiled)
        clip_fea = model._encode_clip_fea_from_video(history[:, :, :1])
    if tuple(condition_latents.shape) != (batch, 16, 3, 30, 52):
        raise ValueError(f"Expected nine-frame temporal VAE encoding [B,16,3,30,52], got {tuple(condition_latents.shape)}")
    if ("prompt" in prepared_inputs) == ("context" in prepared_inputs):
        raise ValueError("Provide prompt OR both text context/context_mask")
    if "prompt" in prepared_inputs:
        context, context_mask = model.encode_prompt(prepared_inputs["prompt"])
    else:
        context = _to_model(model, prepared_inputs["context"])
        context_mask = _to_model(model, prepared_inputs["context_mask"], dtype=torch.bool)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context.shape[0] != batch or context_mask.shape != context.shape[:2]:
        raise ValueError("Text context and mask batch/sequence shapes do not match")
    proprio = _to_model(model, prepared_inputs["proprio"])
    if tuple(proprio.shape) != (batch, 16):
        raise ValueError("Official normalized proprio must be [B,16]")
    context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
    return {"context": context.detach(), "context_mask": context_mask.detach(),
            "condition_latents": condition_latents.detach(), "clip_fea": clip_fea.detach(),
            "history_indices": torch.as_tensor(prepared_inputs["history_indices"]).cpu().clone(),
            "current_indices": torch.as_tensor(prepared_inputs["current_indices"]).cpu().clone(),
            "batch_size": batch, "feature_format": FORMAT}


@torch.no_grad()
def prepare_condition(policy, prepared_inputs: dict[str, Any] | None = None, *, video_seed: int,
                      rand_device: str = "cpu", tiled: bool = False,
                      features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prepare a fresh noise-conditioned frozen video cache for one microbatch.

    For cache reuse, encode_condition_features once, move that small dict's
    tensors to CPU if desired, then provide features=... here. Each video_seed
    produces a fresh pure-noise video; reuse the same seed for paired evaluation.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Use torch.no_grad(), not inference_mode(), when preparing video KV for training")
    model = _model(policy)
    _assert_robodojo(policy)
    if (prepared_inputs is None) == (features is None):
        raise ValueError("Provide prepared_inputs OR features")
    if features is None:
        features = encode_condition_features(policy, prepared_inputs, tiled=tiled)
    if features.get("feature_format") != FORMAT:
        raise ValueError("Unrecognized cached feature format")
    batch = _validate_history(features)
    model.eval()
    if any(p.requires_grad for p in model.video_expert.parameters()):
        raise RuntimeError("Video expert must be frozen before cache generation")
    context = _to_model(model, features["context"])
    context_mask = _to_model(model, features["context_mask"], dtype=torch.bool)
    latents = _to_model(model, features["condition_latents"])
    clip_fea = _to_model(model, features["clip_fea"])
    video_noise = model._sample_action_video_latents(
        first_frame_latents=latents, num_video_frames=17,
        generator=torch.Generator(device=rand_device).manual_seed(int(video_seed)), rand_device=rand_device)
    timestep_video = torch.full((batch,), float(model.train_video_scheduler.num_train_timesteps),
                                dtype=latents.dtype, device=model.device)
    cache, attention_mask, seq_len = model._build_frozen_video_cache(
        latents=video_noise, timestep_video=timestep_video, context=context,
        context_mask=context_mask, condition_latents=latents, clip_fea=clip_fea,
        action_seq_len=32 // model.action_chunk_size,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
    return {"context": context, "context_mask": context_mask, "video_kv_cache": cache,
            "attention_mask": attention_mask, "video_seq_len": seq_len,
            "batch_size": batch, "video_seed": int(video_seed), "condition_format": FORMAT}


def _predict_packed(policy, condition, packed_noisy_action, timesteps):
    model = _model(policy)
    if condition.get("condition_format") != FORMAT:
        raise ValueError("Unrecognized video condition")
    pre = model.action_expert.pre_dit(action_tokens=packed_noisy_action, timestep=timesteps,
                                    context=condition["context"], context_mask=condition["context_mask"])
    tokens = model.mot.forward_action_with_video_cache(
        action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
        action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_kv_cache=condition["video_kv_cache"], attention_mask=condition["attention_mask"],
        video_seq_len=condition["video_seq_len"],
        action_video_conditioning_mask=model._build_action_video_conditioning_mask(timesteps),
        action_clean_first_frame_conditioning_mask=model._build_action_clean_first_frame_conditioning_mask(timesteps))
    if hasattr(model, "recovery_bridge"):
        from .bridge import apply_bridge
        tokens = apply_bridge(model, tokens, condition["video_kv_cache"],
                              condition["context"], condition["context_mask"],
                              mode=condition.get("recovery_mode", "auto"),
                              state=condition.get("recovery_state"))
    return model.action_expert.post_dit(tokens, pre)


def predict_action_velocity(policy, condition: dict[str, Any], noisy_actions: torch.Tensor,
                            timesteps: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of the official @no_grad action predictor."""
    model = _model(policy)
    packed = model._pack_action_tokens(noisy_actions)
    return model._unpack_action_tokens(_predict_packed(policy, condition, packed, timesteps))


def masked_action_mse(prediction: torch.Tensor, target: torch.Tensor,
                      valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must share [B,T,D]")
    if tuple(valid_mask.shape) != tuple(prediction.shape[:2]) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T], True for verified labels")
    valid = valid_mask.to(device=prediction.device)
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every sample must contain at least one verified action frame")
    difference = prediction.float() - target.float()
    # Select before squaring: excluded labels cannot inject NaN/Inf via 0*NaN.
    difference = torch.where(valid[..., None], difference, torch.zeros_like(difference))
    per_dimension = difference.square().sum(dim=1) / counts[:, None]
    return per_dimension.mean(dim=1), per_dimension


def action_flow_loss(policy, condition: dict[str, Any], normalized_actions: torch.Tensor,
                     action_valid_mask: torch.Tensor, *, noise: torch.Tensor | None = None,
                     timesteps: torch.Tensor | None = None,
                     frame_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Official action flow target/weight, with verified-label masking by frame.

    Invalid labels are replaced by zero BEFORE action noising; otherwise even
    masked labels can leak into valid tokens through bidirectional attention.
    The caller averages recovery and replay microbatches in the desired ratio.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Action-LoRA loss cannot run under inference_mode()")
    model = _model(policy)
    assert_only_lora_trainable(policy)
    set_lora_train_mode(policy)
    action = torch.as_tensor(normalized_actions, device=model.device, dtype=model.torch_dtype)
    mask = torch.as_tensor(action_valid_mask, device=model.device)
    batch = condition["batch_size"]
    if tuple(action.shape) != (batch, 32, 16) or tuple(mask.shape) != (batch, 32) or mask.dtype != torch.bool:
        raise ValueError("Expected normalized action[B,32,16] and boolean action_valid_mask[B,32]")
    if not mask.any(dim=1).all() or not torch.isfinite(action[mask]).all():
        raise ValueError("Verified action labels must be finite and nonempty per sample")
    action = torch.where(mask[..., None], action, torch.zeros_like(action))
    if noise is None:
        noise = torch.randn_like(action)
    else:
        noise = torch.as_tensor(noise, device=model.device, dtype=action.dtype)
        if noise.shape != action.shape or not torch.isfinite(noise).all():
            raise ValueError("noise must be finite raw EE16 [B,32,16]")
    if timesteps is None:
        timesteps = model.train_action_scheduler.sample_training_t(batch, device=model.device, dtype=action.dtype)
    else:
        timesteps = torch.as_tensor(timesteps, device=model.device, dtype=action.dtype).reshape(-1)
        if tuple(timesteps.shape) != (batch,) or not torch.isfinite(timesteps).all() or (timesteps < 0).any() or (timesteps > model.train_action_scheduler.num_train_timesteps).any():
            raise ValueError("timesteps must be finite [B] within the scheduler interval")
    noisy_action = model.train_action_scheduler.add_noise(action, noise, timesteps)
    target = model.train_action_scheduler.training_target(action, noise, timesteps)
    prediction = predict_action_velocity(policy, condition, noisy_action, timesteps)
    per_sample, per_dimension = masked_action_mse(prediction, target, mask)
    if frame_weights is not None:
        weights = torch.as_tensor(frame_weights, device=model.device, dtype=torch.float32)
        if weights.shape != mask.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("frame_weights must be finite nonnegative [B,T]")
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        if (weights.sum(1) <= 0).any():
            raise ValueError("Every sample needs positive verified-frame weight")
        difference = torch.where(mask[..., None], prediction.float()-target.float(), 0)
        per_dimension = (difference.square()*weights[..., None]).sum(1)/weights.sum(1)[:, None]
        per_sample = per_dimension.mean(1)
    weight = model.train_action_scheduler.training_weight(timesteps).reshape(-1).to(per_sample)
    loss = (per_sample * weight).mean()
    return loss, {"loss": loss.detach(), "flow_mse": per_sample.mean().detach(),
                  "per_sample_mse": per_sample.detach(), "per_dimension_mse": per_dimension.detach(),
                  "valid_frames": mask.sum().detach(), "timesteps": timesteps.detach()}


@torch.no_grad()
def infer_from_condition(policy, condition: dict[str, Any], *, action_seed: int,
                         num_steps: int = 10, sigma_shift: float | None = None,
                         rand_device: str = "cpu") -> torch.Tensor:
    """Return normalized actions[B,32,16] using official Euler inference math.

    Unlike official infer_action's result['action'], this keeps the batch axis
    and device. For parity compare result[0].float().cpu() with that result.
    """
    model = _model(policy)
    model.eval()
    batch = condition["batch_size"]
    action = torch.randn((batch, 32 // model.action_chunk_size, model.action_expert.action_dim),
                         generator=torch.Generator(device=rand_device).manual_seed(int(action_seed)),
                         device=rand_device, dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
    if sigma_shift is None:
        sigma_shift = getattr(policy, "sigma_shift", 1.0)
    ts, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_steps, device=model.device, dtype=action.dtype, shift_override=sigma_shift)
    for t, delta in zip(ts, deltas):
        timestep = t.reshape(1).expand(batch).to(device=model.device, dtype=action.dtype)
        velocity = _predict_packed(policy, condition, action, timestep)
        action = model.infer_action_scheduler.step(velocity, delta, action)
    return model._unpack_action_tokens(action).detach()


@contextmanager
def adapters_disabled(policy):
    modules = list(lora_modules(policy))
    previous = [m.enabled for _, m in modules]
    try:
        for _, module in modules:
            module.enabled = False
        yield
    finally:
        for (_, module), enabled in zip(modules, previous):
            module.enabled = enabled


def adapter_state_dict(policy) -> dict[str, torch.Tensor]:
    return {f"{name}.{suffix}": getattr(module, suffix).detach().float().cpu().clone()
            for name, module in lora_modules(policy) for suffix in ("lora_A", "lora_B")}


def save_adapter(path, policy, *, metadata: dict[str, Any] | None = None) -> None:
    """Save only adapters and caller provenance; never overwrite base weights."""
    model = _model(policy)
    payload = {"format": FORMAT, "config": dict(model._action_lora_config),
               "state_dict": adapter_state_dict(policy), "metadata": dict(metadata or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def load_adapter(path, policy, *, expected_base_identity: str | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Unsupported adapter checkpoint format")
    if expected_base_identity is not None and payload.get("metadata", {}).get("base_identity") != expected_base_identity:
        raise ValueError("Adapter base checkpoint identity mismatch")
    config = payload["config"]
    if not list(lora_modules(policy)):
        install_action_lora(policy, rank=config["rank"], alpha=config["alpha"], gradient_checkpointing=config["gradient_checkpointing"])
    current = _model(policy)._action_lora_config
    for key in ("rank", "alpha", "targets", "raw_action_dim", "action_horizon", "history_frames", "history_stride"):
        if current[key] != config[key]:
            raise ValueError(f"Adapter config mismatch: {key}")
    destination = {f"{n}.{s}": getattr(m, s) for n, m in lora_modules(policy) for s in ("lora_A", "lora_B")}
    saved = payload["state_dict"]
    if destination.keys() != saved.keys():
        raise ValueError("Adapter tensor keys mismatch")
    # Validate all tensors before changing any installed adapter.
    for key, tensor in saved.items():
        if tensor.shape != destination[key].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid adapter tensor: {key}")
    for key, tensor in saved.items():
        destination[key].copy_(tensor)
    return dict(payload.get("metadata", {}))


@torch.no_grad()
def merged_action_state_dict(policy) -> OrderedDict[str, torch.Tensor]:
    """Export an ordinary ActionDiT state dict, without changing live weights.

    BF16 merging rounds weights once and is not guaranteed bit-identical to
    runtime FP32 LoRA addition. Re-evaluate the merged artifact before submitting.
    """
    model = _model(policy)
    modules = dict(lora_modules(policy))
    state = OrderedDict()
    for key, tensor in model.action_expert.state_dict().items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        converted = key.replace(".base.", ".")
        state[converted] = tensor.detach().cpu().clone()
    for name, module in modules.items():
        delta = module.lora_B.detach().float().cpu() @ module.lora_A.detach().float().cpu()
        state[name + ".weight"] = (module.base.weight.detach().float().cpu() + delta * module.scale).to(module.base.weight.dtype)
    return state
