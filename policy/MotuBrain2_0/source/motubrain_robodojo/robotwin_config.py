"""Apply the RoboTwin DMD-1000 sampling and EEF configuration."""

from dataclasses import replace

from morph import EEF, EmbodiedPromptBuilder, NormalizationConfig
from soma.inference.config import SparseMemoryConfig, load_training_config


def configure(cfg, stats, model_config):
    snapshot = load_training_config(str(model_config))
    model = snapshot["model"]
    base = cfg.policy
    fields = []
    for field in base.action.profile.fields:
        if field.name in ("left_qpos", "right_qpos"):
            field = replace(field, input_used=False, model_used=False, output_used=False)
        elif field.name in ("left_eef", "right_eef"):
            field = EEF(name=field.name, model_repr="relative", input_rot="xyzw", output_rot="xyzw")
        fields.append(field)
    profile = replace(base.action.profile, name="robotwin-eef", fields=tuple(fields), use_ref_frame=True,
                      normalization=NormalizationConfig(method="quantile", stats=str(stats),
                                                        quantiles=(0.01, 0.99), eps=0.001, clip=(-5, 5)))
    order = ("left_eef", "left_gripper", "right_eef", "right_gripper")
    layers = model["action_conditions"]["geometry"]["layers"]
    cfg.policy = replace(base,
        vae=replace(base.vae, incremental_encode=True),
        action=replace(base.action, profile=profile, embodiment="aloha_agilex", state_key="eef",
                       camera_key="camera_extrinsic", state_field_order=order, output_field_order=order),
        # The shared model-config loader sets num_frames and video_action_freq.
        sparse_memory=SparseMemoryConfig(enabled=False),
        geometry=replace(base.geometry, layer_map={i: i for i in layers}),
        multiview=replace(base.multiview, rope_mode=model["multiview_rope_mode"]),
        prompt_builder=EmbodiedPromptBuilder(embodiment="aloha_agilex", lang="en"))
