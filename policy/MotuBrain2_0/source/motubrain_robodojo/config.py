"""Direct inference with the released RoboDojo sampling and action contract."""

import os

from morph import ActionProfile
from morph import Chassis
from morph import EEF
from morph import EmbodiedPromptBuilder
from morph import Gripper
from morph import Padding
from morph import Qpos
from morph.convs.image import ResizeConfig
from soma.data import conditions
from soma.inference.config import SparseMemoryConfig, InferenceConfig, CompileConfig, SimpleRTCConfig
from soma.inference.diffusion.solvers import CMSolver
from soma.inference.config import ActionConfig
from soma.inference.config import GeometryConfig
from soma.inference.config import MultiviewConfig
from soma.inference.config import ObservationConfig
from soma.inference.config import PolicyConfig
from soma.inference.config import TextEncoderConfig
from soma.inference.config import VaeConfig


_TEXT_ENCODER_PATH = os.environ["SOMA_TEXT_ENCODER_PATH"]
_VAE_PATH = os.environ["SOMA_VAE_PATH"]
_VIEW_NAMES = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
_ROBODOJO_QPOS14_ORDER = (
    "left_qpos",
    "left_gripper",
    "right_qpos",
    "right_gripper",
)
_MAIN_RESIZE = ResizeConfig(
    mode="area",
    target_area=120_000,
    multiple=16,
    interpolation="bicubic",
)
_WRIST_RESIZE = ResizeConfig(
    mode="area",
    target_area=30_000,
    multiple=16,
    interpolation="bicubic",
)

policy = PolicyConfig(
    type="soma.inference.policies.motubrain.MotubrainPolicy",
    dit_factory="soma.inference.runtime.motubrain.build_model",
    vae=VaeConfig(
        vae_pth=_VAE_PATH,
        precision="bf16",
        incremental_encode=False,
    ),
    text_encoder=TextEncoderConfig(
        pretrained_model_name_or_path=_TEXT_ENCODER_PATH,
    ),
    geometry=GeometryConfig(
        config_path=os.environ["DA3_CONFIG_PATH"],
        checkpoint_path=os.environ["DA3_CHECKPOINT_PATH"],
        layer_map={
            layer: layer
            for layer in (10, 12, 14, 16, 18, 20, 22, 24)
        },
        views={
            name: conditions.ConditionViewConfig(name)
            for name in _VIEW_NAMES
        },
        image_size=(280, 504),
        required_image_names=_VIEW_NAMES,
    ),
    observation=ObservationConfig(
        resize=_MAIN_RESIZE,
        num_frames=9,
        video_length=-1,
        image_format="raw",
        jpeg_quality=95,
    ),
    action=ActionConfig(
        video_action_freq=4,
        state_key="qpos",
        embodiment="arx_x5",
        state_field_order=_ROBODOJO_QPOS14_ORDER,
        output_field_order=_ROBODOJO_QPOS14_ORDER,
        profile=ActionProfile(
            name="robodojo-canonical55-absolute",
            fields=(
                Qpos(name="left_qpos", dim=6),
                Padding(name="left_qpos_pad"),
                EEF(
                    name="left_eef",
                    input_used=False,
                    output_used=False,
                ),
                Gripper(name="left_gripper"),
                Qpos(name="right_qpos", dim=6),
                Padding(name="right_qpos_pad"),
                EEF(
                    name="right_eef",
                    input_used=False,
                    output_used=False,
                ),
                Gripper(name="right_gripper"),
                Qpos(
                    name="torso_qpos",
                    dim=4,
                    input_used=False,
                    model_used=False,
                    output_used=False,
                ),
                Padding(name="torso_qpos_pad", dim=3),
                EEF(
                    name="torso_eef",
                    input_used=False,
                    model_used=False,
                    output_used=False,
                ),
                Qpos(
                    name="head_qpos",
                    dim=2,
                    input_used=False,
                    model_used=False,
                    output_used=False,
                ),
                Chassis(
                    name="chassis",
                    input_used=False,
                    model_used=False,
                    output_used=False,
                ),
            ),
        ),
    ),
    sparse_memory=SparseMemoryConfig(enabled=True, frame_stride=16),
    prompt_builder=EmbodiedPromptBuilder(
        embodiment="arx_x5",
        lang="zh",
    ),
    multiview=MultiviewConfig(
        num_views=len(_VIEW_NAMES),
        view_names=list(_VIEW_NAMES),
        rope_mode="per_view",
        view_resizes={
            "cam_high": _MAIN_RESIZE,
            "cam_left_wrist": _WRIST_RESIZE,
            "cam_right_wrist": _WRIST_RESIZE,
        },
    ),
)



inference = InferenceConfig(
    steps=4, solver=CMSolver, timeshift=6.0, action_timeshift=1.0,
    decode_video=False, action_smoothing_window=0,
    simple_rtc=SimpleRTCConfig(enabled=False),
    compile=CompileConfig(dit=False, vae=False),
    context_parallel_size=1, int8=False,
    cfg_scale=None, dit_cache_threshold=None, dit_cache_threshold_2=None,
)
