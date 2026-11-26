"""Normal-state retention against the frozen pretrained Euler trajectory.

This loss contains no demonstration-action loss or built-in coefficient. Build
one frozen video condition, sample a discrete step uniformly in 0..9 outside
this module, and match the official velocity at that teacher trajectory state.
Teacher actions/velocities are detached; only the active adapter receives
student gradients. Returned anchors contain only small action tensors, never
video KV. The caller owns and releases its video condition.
"""
from __future__ import annotations

from contextlib import contextmanager
import operator

import torch

from recoverwam.recovery.lora import (_predict_packed, adapters_disabled, assert_only_lora_trainable,
    lora_modules, set_lora_train_mode)
from recoverwam.recovery.bridge import assert_only_bridge_trainable


FORMAT = "recoverwam-official-euler-retention-anchor-v1"


def _adapter_kind(policy):
    bridge = hasattr(policy.model, "recovery_bridge")
    lora = bool(list(lora_modules(policy)))
    if bridge == lora:
        raise RuntimeError("Retention requires exactly one recovery bridge or Action-LoRA")
    if bridge:
        assert_only_bridge_trainable(policy.model)
        return "bridge"
    assert_only_lora_trainable(policy)
    return "lora"


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@contextmanager
def _temporary_eval(model):
    # Restore each flag exactly, including the official mixed train/eval setup.
    previous = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in previous:
            module.training = training


@torch.no_grad()
def build_retention_anchor(policy, condition, *, action_seed, step_index,
                           num_steps=10, sigma_shift=1.0, rand_device="cpu",
                           return_teacher_final=False):
    """Capture teacher (packed x_t, timestep, velocity) at one Euler step.

    The initial packed noise, dtype conversion, schedule, per-step velocity and
    Euler update are exactly those in action_lora.infer_from_condition. The
    default stops after computing the selected step's teacher velocity: there
    are step_index+1 velocity forwards and step_index Euler updates. For a
    one-time full-model parity check, return_teacher_final=True additionally
    completes the official trajectory and returns normalized_actions[B,32,16].

    Call under ordinary grad mode or torch.no_grad(), never inference_mode():
    the detached selected state must be usable by student autograd afterward.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Retention anchors must not be created under inference_mode")
    action_seed = _integer(action_seed, "action_seed")
    step_index = _integer(step_index, "step_index")
    if _integer(num_steps, "num_steps") != 10 or float(sigma_shift) != 1.0:
        raise ValueError("This retention experiment requires the official 10-step, sigma=1 schedule")
    if not 0 <= step_index < 10:
        raise ValueError("step_index must be in 0..9")
    model = policy.model
    kind = _adapter_kind(policy)
    teacher_condition = (dict(condition, recovery_mode="off", recovery_state=None)
                         if kind == "bridge" else condition)
    batch = condition["batch_size"]
    if 32 % model.action_chunk_size:
        raise ValueError("Action chunk size must divide the official 32-step horizon")
    with _temporary_eval(model), adapters_disabled(policy):
        # Keep this construction in the same packed order as inference; drawing
        # an unpacked tensor first need not preserve the same random ordering.
        action = torch.randn((batch, 32 // model.action_chunk_size, model.action_expert.action_dim),
            generator=torch.Generator(device=rand_device).manual_seed(action_seed),
            device=rand_device, dtype=torch.float32).to(device=model.device, dtype=model.torch_dtype)
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=10, device=model.device, dtype=action.dtype, shift_override=1.0)
        if len(timesteps) != 10 or len(deltas) != 10:
            raise RuntimeError("Official scheduler returned an unexpected number of steps")
        anchor = None
        for index, (t, delta) in enumerate(zip(timesteps, deltas)):
            timestep = t.reshape(1).expand(batch).to(device=model.device, dtype=action.dtype)
            velocity = _predict_packed(policy, teacher_condition, action, timestep)
            if index == step_index:
                anchor = dict(format=FORMAT, packed_noisy_action=action.detach().clone(),
                    timesteps=timestep.detach().clone(), teacher_velocity=velocity.detach().clone(),
                    step_index=step_index, action_seed=action_seed, num_steps=10, sigma_shift=1.0,
                    batch_size=batch, video_seed=condition.get("video_seed"),
                    euler_delta=delta.detach().clone())
                if not return_teacher_final:
                    break
            action = model.infer_action_scheduler.step(velocity, delta, action)
        if anchor is None:
            raise RuntimeError("Selected teacher step was not reached")
        if return_teacher_final:
            anchor["teacher_final_action"] = model._unpack_action_tokens(action).detach().clone()
    return anchor


def retention_anchor_loss(policy, condition, anchor):
    """Return unweighted normalized-velocity MSE and detached raw diagnostics.

    Does not recreate video KV, draw fresh noise, change the teacher trajectory,
    train the base model, or apply any coefficient. The caller can use, e.g.,
    recovery_loss + coefficient * loss. An anchor is valid only with the same
    conditioning input/video noise that was used to construct it.
    """
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise RuntimeError("Student retention loss requires ordinary autograd mode")
    kind = _adapter_kind(policy)
    modules = list(lora_modules(policy))
    if any(not module.enabled for _, module in modules):
        raise RuntimeError("All student adapters must be enabled for retention loss")
    if anchor.get("format") != FORMAT:
        raise ValueError("Unrecognized retention anchor")
    if anchor.get("num_steps") != 10 or anchor.get("sigma_shift") != 1.0:
        raise ValueError("Anchor schedule differs from the official inference schedule")
    if anchor.get("batch_size") != condition["batch_size"]:
        raise ValueError("Anchor and condition batch sizes differ")
    if anchor.get("video_seed") != condition.get("video_seed"):
        raise ValueError("Anchor and condition video seeds differ")
    model = policy.model
    expected_shape = (condition["batch_size"], 32 // model.action_chunk_size, model.action_expert.action_dim)
    # model.device can be the unindexed alias "cuda" while tensors report
    # cuda:0; the resident action parameters give the canonical device.
    expected_device = next(model.action_expert.parameters()).device
    for key, shape in (("packed_noisy_action", expected_shape), ("teacher_velocity", expected_shape),
                       ("timesteps", (condition["batch_size"],))):
        tensor = anchor.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"Invalid anchor tensor {key}")
        if tensor.requires_grad or tensor.grad_fn is not None or torch.is_inference(tensor):
            raise ValueError(f"Teacher tensor must be ordinary and detached: {key}")
        if tensor.device != expected_device or tensor.dtype != model.torch_dtype:
            raise ValueError(f"Teacher tensor device/dtype differs from the model: {key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite teacher tensor {key}")
    # This mirrors the supported Action-LoRA training flags; the outer joint
    # model and video expert remain in eval mode. FP32 adapters stay FP32.
    if kind == "lora":
        set_lora_train_mode(policy)
        student_condition = condition
    else:
        model.eval()
        # Constrain the correction even when the learned gate would bypass it.
        student_condition = dict(condition, recovery_mode="on")
    student = _predict_packed(policy, student_condition, anchor["packed_noisy_action"], anchor["timesteps"])
    difference = student.float() - anchor["teacher_velocity"].float()
    per_sample = difference.square().flatten(1).mean(1)
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite retention loss")
    return loss, dict(retention_mse=loss.detach(), per_sample_mse=per_sample.detach(),
        step_index=anchor["step_index"], action_seed=anchor["action_seed"],
        timesteps=anchor["timesteps"].detach(),
        teacher_velocity_rms=anchor["teacher_velocity"].float().square().mean().sqrt().detach(),
        student_velocity_rms=student.float().square().mean().sqrt().detach(),
        max_abs_velocity_difference=difference.detach().abs().max())


def normal_teacher_anchor_loss(policy, condition, *, action_seed, step_index,
                              num_steps=10, sigma_shift=1.0, rand_device="cpu"):
    """Convenience wrapper; no ground-truth actions or built-in weighting."""
    anchor = build_retention_anchor(policy, condition, action_seed=action_seed,
        step_index=step_index, num_steps=num_steps, sigma_shift=sigma_shift, rand_device=rand_device)
    return retention_anchor_loss(policy, condition, anchor)
