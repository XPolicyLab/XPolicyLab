"""Independent model family; old VAP configuration and defaults are untouched."""

from copy import copy
from dataclasses import dataclass

from wall_x.config.registry import register_model_config
from wall_x.model.vap.config import VAPModelConfig


@register_model_config("vapv2")
@dataclass
class VAPv2ModelConfig(VAPModelConfig):
    architecture_version: str = "vapv2.qwen_joint_ivava.v2"
    text_encoder_backend: str = "qwen35_online"
    # Persist this checkpoint field through the dataclass loader. Online
    # HEAD encoding must not silently fall back to the legacy 'none' default.
    qwen35_truncation: str = "none"
    text_projector_output_dim: int = 3072
    # VAPv2 keeps the Video Expert at Wan width but uses a narrower Action
    # Expert to reduce optimizer/FSDP state and activation cost.
    action_expert_dim: int = 1024
    text_injection_mode: str = "joint_ivava_prefix"
    condition_attention_mode: str = "bidirectional"
    condition_timestep_mode: str = "fixed_clean"
    text_position_mode: str = "negative_time_4d"
    first_frame_memory_injection_mode: str = "prefix_self_attention"
    # Keep the independent v2 schema complete on deployment mirrors whose
    # legacy VAP schema predates these optional training-only augmentations.
    action_prefix_noise_std_choices: tuple[float, ...] = ()
    memory_history_dropout_probability: float = 0.0

    def __post_init__(self):
        if self.text_encoder_backend not in {"qwen35_online", "qwen35_cache"}:
            raise ValueError("VAPv2 requires frozen Qwen3.5 online or raw cache, not T5")
        # Reuse horizon/flow/memory validation without changing the legacy
        # validator's 4096-dimensional Qwen-to-T5 bridge contract.
        legacy = copy(self)
        legacy.architecture_version = "vap.legacy.v1"
        legacy.text_projector_output_dim = 4096
        VAPModelConfig.__post_init__(legacy)
        for name, value in vars(legacy).items():
            if name not in {"text_projector_output_dim", "architecture_version"}:
                setattr(self, name, value)
        expected = {
            "architecture_version": "vapv2.qwen_joint_ivava.v2",
            "text_injection_mode": "joint_ivava_prefix",
            "condition_attention_mode": "bidirectional",
            "condition_timestep_mode": "fixed_clean",
            "text_position_mode": "negative_time_4d",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Unsupported VAPv2 {name}: {getattr(self, name)!r}")
        if self.text_projector_output_dim != 3072:
            raise ValueError("VAPv2 Wan2.2-5B text projector must output 3072, not T5's 4096")
        if self.action_expert_dim != 1024:
            raise ValueError(
                "VAPv2 action_expert_dim is fixed at 1024; "
                "use model_type=vap for the historical 1536-wide Action Expert"
            )
        if self.first_frame_memory_enabled and self.first_frame_memory_injection_mode != "prefix_self_attention":
            raise ValueError("VAPv2 supports prefix Memory only")
        if self.action_xm_best_of_k != 1 or self.action_soar_mode != "off":
            raise ValueError("VAPv2 joint v2 has not validated XM/SOAR auxiliary sampling")
        if (self.validation or {}).get("action_openloop", False):
            raise ValueError("VAPv2 trainer supports diffusion-loss validation; legacy openloop serving is incompatible")
