"""Opt-in inference precision; does not change checkpoint or prefix semantics.

The trunk, action encoder, KV and memory/history tokens remain in the original
model dtype. Only the active [B,A,D] ODE state and, optionally, the action output
head use FP32. Initial action noise is separately opt-in: legacy model dtype
or a direct FP32 draw. Video noise, seed derivation and draw ordering stay intact.
"""

import torch

from wall_x.model.vap.model import ExpertOutputHead


MODES = ("legacy", "fp32_accum", "fp32_head_accum")


class FP32ActionOutputHead(ExpertOutputHead):
    def forward(self, hidden, time_embedding):
        if self.training:
            raise RuntimeError("FP32 action head override is inference-only")
        # Also defeat an enclosing autocast: FP32 weights alone do not guarantee
        # FP32 projection. This context is safe during CUDA Graph capture/replay.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return super().forward(hidden.float(), time_embedding.float())


def configure_action_precision(core, mode="legacy", *, initial_noise_dtype="legacy"):
    """Configure before constructing the accelerator; state_dict keys unchanged."""
    if mode not in MODES:
        raise ValueError(f"action precision must be one of {MODES}, got {mode!r}")
    if initial_noise_dtype not in ("legacy", "fp32"):
        raise ValueError("initial_noise_dtype must be legacy or fp32")
    if initial_noise_dtype == "fp32" and mode == "legacy":
        raise ValueError("Direct FP32 noise requires an FP32 action accumulator")
    if core.training:
        raise RuntimeError("action precision override requires eval mode")
    if getattr(core, "_v2_action_accelerator", None) is not None:
        raise RuntimeError("disable the CUDA Graph accelerator before changing precision")
    head = core.action_head
    if type(head) not in (ExpertOutputHead, FP32ActionOutputHead):
        raise TypeError(f"unsupported action output head: {type(head).__name__}")
    trunk_dtype = core.action_encoder.weight.dtype if isinstance(
        core.action_encoder, torch.nn.Linear
    ) else next(core.action_encoder.parameters()).dtype
    if mode == "fp32_head_accum":
        head.__class__ = FP32ActionOutputHead
        head.float()
        original = getattr(core, "_v2_action_head_checkpoint_fp32", None)
        if original is not None:
            # Recover the checkpoint's FP32 values, not a BF16 round trip.
            # This small CPU dict is not a parameter/buffer and is never used
            # on the denoising hot path or serialized into a checkpoint.
            head.load_state_dict(original, strict=True)
    else:
        head.__class__ = ExpertOutputHead
        head.to(dtype=trunk_dtype)
    core._v2_action_fp32_accum = mode != "legacy"
    core._v2_action_precision = mode
    core._v2_action_noise_dtype = initial_noise_dtype
    return {
        "mode": mode,
        "trunk_dtype": str(trunk_dtype),
        "action_head_dtype": str(next(head.parameters()).dtype),
        "accumulator_dtype": "torch.float32" if mode != "legacy" else str(trunk_dtype),
        "initial_noise": ("direct_fp32" if initial_noise_dtype == "fp32"
                          else "legacy_model_dtype_same_rng_stream"),
        "head_weight_source": ("checkpoint_fp32" if mode == "fp32_head_accum"
            and getattr(core, "_v2_action_head_checkpoint_fp32", None) is not None
            else "existing_model_values"),
        "prefix_memory_history": "unchanged",
    }
