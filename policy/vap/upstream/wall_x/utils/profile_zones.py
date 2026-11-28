"""Named regions for torch.profiler — central registry of `record_function` zones.

`torch.profiler.record_function(name)` writes a `user_annotation` slice into the
profiler trace. We use it to mark phase boundaries that would otherwise be
invisible when `with_stack=False` (the lightweight profiling mode).

Call sites use the `zone(...)` alias and the `Z` namespace:

    from wall_x.utils.profile_zones import zone, Z

    with zone(Z.FORWARD):
        outputs = self.adapter.forward(...)

Keeping the names here makes them grep-able and prevents string-typo drift
across the ~5 files that emit them.
"""

import torch

# Alias — same call site ergonomics as `torch.no_grad()`.
zone = torch.profiler.record_function


class Z:
    # All zone names use a dotted prefix so visualize_wallx can disambiguate
    # them with exact / prefix LIKE patterns (avoiding the substring fallout
    # we hit when "forward" matched "pi0.paligemma_forward").

    # ── Training phase (fsdp_trainer.train_loop) ──
    TRAIN_STEP = "train.step"
    DATA_TO_DEVICE = "train.data_to_device"
    FORWARD = "train.forward"
    BACKWARD = "train.backward"
    GRAD_NORMS = "train.grad_norms"
    GRAD_CLIP = "train.grad_clip"
    OPTIMIZER_STEP = "train.optimizer_step"
    ZERO_GRAD = "train.zero_grad"
    LOGGING = "train.logging"
    SAVE_CHECKPOINT = "train.save_checkpoint"

    # ── Pi0JointGemmaLayer.forward sub-phases ──
    # Sub-phase names are emitted as `joint_layer.{idx:02d}.{phase}` (dynamic)
    # inside Pi0JointGemmaLayer._forward_inner — see gemma_pytorch.py. The
    # idx-bearing form lets visualize_wallx slice both ways:
    #   joint_layer.%.attention       → pool across 18 layers, per-phase view
    #   joint_layer.05.%              → all sub-phases of layer 5
    # so we don't pre-declare them as constants here.

    # ── PI0Pytorch.forward sub-phases ──
    PI0_PREPROCESS_OBSERVATION = "pi0.preprocess_observation"
    PI0_EMBED_PREFIX = "pi0.embed_prefix"
    PI0_EMBED_SUFFIX = "pi0.embed_suffix"
    PI0_PALIGEMMA_FORWARD = "pi0.paligemma_forward"
    PI0_ACTION_OUT_PROJ = "pi0.action_out_proj"
    PI0_FLOW_LOSS = "pi0.flow_loss"

    # ── pi0 adapter ──
    ADAPTER_CONVERT_OBSERVATION = "adapter.convert_observation"
    ADAPTER_ACTION_PREPARE = "adapter.action_prepare"

    # ── Per-block outer wraps (replace the lost `nn.Module: <Cls>_<N>` annotations) ──
    # PaliGemma vision-side wrap (text-side is covered by the per-Pi0JointGemmaLayer
    # zones — pi05 training bypasses PaliGemmaForConditionalGeneration.forward).
    PALIGEMMA_VISION_TOWER = "paligemma.vision_tower"

    @staticmethod
    def joint_layer(layer_idx: int) -> str:
        """Outer wrap for one Pi0JointGemmaLayer — 18 such blocks per forward."""
        return f"Pi0JointGemmaLayer.{layer_idx:02d}"

    @staticmethod
    def siglip_layer(layer_idx: int) -> str:
        """Per-block wrap inside SiglipEncoder — 27 such blocks per vision pass."""
        return f"siglip_layer.{layer_idx:02d}"
