import os
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

IGNORE_INDEX = -100


# Lazy cache for VLA yaml contents.
_VLA_CFG: Optional[Dict[str, Any]] = None


def configure_vla_constants(yaml_name: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache VLA constants from a yaml file.

    - If `yaml_name` is not provided, it reads `VLA_CONFIG_YAML` from env.
    - This function is safe to call multiple times.
    """
    global _VLA_CFG

    if yaml_name is not None:
        os.environ["VLA_CONFIG_YAML"] = yaml_name
    yaml_name = yaml_name or os.getenv("VLA_CONFIG_YAML")

    if not yaml_name:
        raise RuntimeError(
            "VLA_CONFIG_YAML is not set. "
            "Set it before accessing VLA constants, or call `configure_vla_constants(yaml_name=...)`."
        )

    # Import here to avoid import-time side effects (and keep
    # `import a1.vla.constants` lightweight).
    from a1.vla.config_loader import read_vla_yaml_config

    _VLA_CFG = read_vla_yaml_config(yaml_name)
    return _VLA_CFG


def _get_action_head() -> Dict[str, Any]:
    global _VLA_CFG
    if _VLA_CFG is None:
        configure_vla_constants()
    # _VLA_CFG validated above
    return _VLA_CFG["model"]["action_head"]  # type: ignore[index]


def __getattr__(name: str) -> Any:
    """Lazy attributes for VLA constants.

    This avoids import-time side effects (e.g. parsing YAML at module import).
    """
    if name == "NUM_ACTIONS_CHUNK":
        return _get_action_head()["num_actions_chunk"]
    if name == "ACTION_DIM":
        return _get_action_head()["fixed_action_dim"]
    if name == "PROPRIO_DIM":
        return _get_action_head()["fixed_action_dim"]
    if name == "ACTION_DIMS_MAPPING":
        return _get_action_head()["action_tokens_mapping"]
    if name == "ACTION_DIMS":
        # Backward-compat for older code that imports ACTION_DIMS.
        # Now uses fixed_action_dim.
        return _get_action_head()["fixed_action_dim"]
    if name == "ACTION_PROPRIO_NORMALIZATION_TYPE":
        # Not consistently present in model.action_head; keep prior behavior.
        return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ACTION_DIMS = total_dim = sum(ACTION_DIMS_MAPPING.values())

# def build_action_vector(*, fill_value: float = 0.0, **kwargs) -> np.ndarray:
#     """Construct the action vector and set the unprovided part as the default value"""
#     vector = []
#     for part, dim in ACTION_DIMS.items():
#         vector.append(kwargs.get(part, [fill_value] * dim))
#     return np.concatenate(vector)

# # usage
# vector = build_action_vector(
#     fill_value=-1.0,
#     left_arm=[0.1, 0.2, 0.3, 0, 0, 0, 0.5],
#     base_movement=[0.5, 0, 0.1]
# )


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
# LIBERO_CONSTANTS = {
#     "NUM_ACTIONS_CHUNK": 8,
#     "ACTION_DIM": 7,
#     "PROPRIO_DIM": 7, # 8
#     "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
# }

# ALOHA_CONSTANTS = {
#     "NUM_ACTIONS_CHUNK": 8, #25,
#     "ACTION_DIM": 14,
#     "PROPRIO_DIM": 14,
#     "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
# }

# BRIDGE_CONSTANTS = {
#     "NUM_ACTIONS_CHUNK": 5,
#     "ACTION_DIM": 7,
#     "PROPRIO_DIM": 7,
#     "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
# }

# DEFAULT_CONSTANTS = {
#     "NUM_ACTIONS_CHUNK": 8,
#     "ACTION_DIM": 7,
#     "PROPRIO_DIM": 7,
#     "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
# }

# GLUE_ROBOT_CONSTANTS = {
#     "NUM_ACTIONS_CHUNK": 8,
#     "ACTION_DIM": 16,
#     "PROPRIO_DIM": 16,
#     "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
# }

# # 这种方式不好，不明确容易出错
# # Function to detect robot platform from command line arguments
# def detect_robot_platform():
#     cmd_args = " ".join(sys.argv).lower()

#     if "libero" in cmd_args:
#         return "LIBERO"
#     elif "aloha" in cmd_args:
#         return "ALOHA"
#     elif "bridge" in cmd_args:
#         return "BRIDGE"
#     elif "glue_lerobot" in cmd_args:
#         return "GLUE_ROBOT"
#     else:
#         # Check if we're using lerobot_dataset and infer from data
#         if "lerobot_dataset" in cmd_args:
#             return "ALOHA"  # Use ALOHA config for 14-dim data
#         # Default to LIBERO if unclear
#         return "DEFAULT"


# # Determine which robot platform to use
# ROBOT_PLATFORM = detect_robot_platform()

# # Set the appropriate constants based on the detected platform
# if ROBOT_PLATFORM == "LIBERO":
#     constants = LIBERO_CONSTANTS
# elif ROBOT_PLATFORM == "ALOHA":
#     constants = ALOHA_CONSTANTS
# elif ROBOT_PLATFORM == "BRIDGE":
#     constants = BRIDGE_CONSTANTS
# elif ROBOT_PLATFORM == "GLUE_ROBOT":
#     constants = GLUE_ROBOT_CONSTANTS
# elif ROBOT_PLATFORM == "DEFAULT":
#     constants = DEFAULT_CONSTANTS


# Assign constants to global variables
# NUM_ACTIONS_CHUNK = 8 # constants["NUM_ACTIONS_CHUNK"]
# ACTION_DIM = constants["ACTION_DIM"]
# PROPRIO_DIM = constants["PROPRIO_DIM"]
# ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]