# Copyright 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from transformers import AutoTokenizer
 
from BeingH.model.llm.qwen2 import Qwen2Tokenizer
from BeingH.model.layers import InternVLConnector
from BeingH.model import (
    Qwen2ForCausalLM, Qwen2Config,
    Qwen3ForCausalLM, Qwen3Config,
    InternVisionModel, InternVisionConfig,
)

# ==============================================================================
# Model Architecture Registry
# ==============================================================================

LLM_MODEL_ARCH = {
    "Qwen2ForCausalLM": (Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer),
    "Qwen3ForCausalLM": (Qwen3Config, Qwen3ForCausalLM, AutoTokenizer),
}

VIT_MODEL_ARCH = {
    "InternVisionModel": (InternVisionConfig, InternVisionModel),
}

CONNECTOR_ARCH = {
    "internvl_connector": InternVLConnector
}


MAP_MLLM_LM = {
    "InternVL3-1B": "Qwen2.5-0.5B-Instruct",
    "InternVL3-2B": "Qwen2.5_1.5B",
    "InternVL3-8B": "Qwen2.5_7B",
    "InternVL3-14B": "Qwen2.5_14B",
    "InternVL3_5-1B": "Qwen3-0.6B",
    "InternVL3_5-2B": "Qwen3-1.7B",
    "InternVL3_5-4B": "Qwen3-4B"
}


# ==============================================================================
# Special Tokens
# ==============================================================================

# Basic tokens
BOS_TOKEN="<|im_start|>"
EOS_TOKEN="<|im_end|>"

# Vision tokens
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = "<|vision_start|>" #'<img>'
IMG_END_TOKEN = '<|vision_end|>' #'</img>'
IMAGE_TOKEN='<|image_pad|>'
VIDOE_TOKEN='<|video_pad|>'

# Spatial tokens
QUAD_START_TOKEN = "<|quad_start|>" #'<quad>'
QUAD_END_TOKEN = "<|quad_end|>" #'</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

# Action and state tokens
ACTION_TOKEN='<|action_pad|>'
STATE_TOKEN='<state_pad>'

# Prop tokens
PROP_START_TOKEN = '<prop>'
PROP_END_TOKEN = '</prop>'
PROP_CONTEXT_TOKEN = '<PROP_CONTEXT>'

# Absolute transform tokens - Used to mark absolute transform sequences
ABS_START_TOKEN = '<abs>'
ABS_END_TOKEN = '</abs>'
ABS_TRANS_TOKEN = '<ABS_TRANS>'  # Special token placeholder for absolute transform

# Latent vision tokens - Used to mark latent vision sequences
LAT_START_TOKEN = '<lat>'
LAT_END_TOKEN = '</lat>'
LAT_VIS_TOKEN = '<LAT_VIS>'  # Special token placeholder for latent vision


# ==============================================================================
# Constants
# ==============================================================================

BLOCK_SIZE = 130
IGNORE_INDEX = -100

# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


INSTRUCTION_TEMPLATE = (
    "According to the instruction '{task_description}', "
    "what's the micro-step actions in the next {k} steps?"
)

MULTI_DB_INSTRUCT_TEMPLATE = (
    "# Robot Configuration\n"
    "- Robot type: {arm_type} arm with {eef_type} end-effector\n"
    "- Available camera viewpoints: {view_list}\n\n"
    
    "# State Space\n"
    "- Maximum state dimension: {max_state_dim}\n"
    "- Active state dimension: {state_dim}\n"
    "- State vector ordering: {state_desc}\n\n"
    
    "# Action Space\n"
    "- Maximum action dimension: {max_action_dim}\n"
    "- Active action dimension: {action_dim}\n"
    "- Action vector ordering: {action_desc}\n\n"

    "# Task\n"
    "Given the instruction: \"{task_description}\"\n\n"
    
    "Generate a sequence of {k} micro-step actions to accomplish this task."
)

# ==============================================================================
# Embodiment Configuration
# ==============================================================================

class EmbodimentTag(Enum):
    """Enumeration of supported robot embodiments."""    
    LIBERO_FRANKA = "libero_franka_gripper"
    LIBERO = "libero"
    ROBOCASA = "robocasa"

    NEW_EMBODIMENT = "new_embodiment"


# Embodiment to projector index mapping for Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.LIBERO_FRANKA.value: 0,
    EmbodimentTag.LIBERO.value: 0,  # Same as LIBERO_FRANKA
    EmbodimentTag.ROBOCASA.value: 31,
    
    EmbodimentTag.NEW_EMBODIMENT.value: 31,

}

TARGET_STATE_ROTATION_TYPE = "axis_angle"
TARGET_ACTION_ROTATION_TYPE = "axis_angle"
AGIBOT_ABS_OR_RELA = "relative"

# Rotation dimension mapping
_ROTATION_DIM_MAP = {
    "rotation_6d": 6,
    "axis_angle": 3,
    "quaterion": 4,
}

def _get_rotation_dim(rotation_type: str) -> int:
    """Get rotation dimension based on rotation type."""
    if rotation_type in _ROTATION_DIM_MAP:
        return _ROTATION_DIM_MAP[rotation_type]
    elif "euler_angles" in rotation_type:
        return 3
    else:
        raise ValueError(f"Unknown rotation type: {rotation_type}")

TARGET_STATE_ROTATION_DIM = _get_rotation_dim(TARGET_STATE_ROTATION_TYPE)
TARGET_ACTION_ROTATION_DIM = _get_rotation_dim(TARGET_ACTION_ROTATION_TYPE)
