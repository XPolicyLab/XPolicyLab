from collections.abc import Mapping

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES, OBS_IMAGE
from .utils import make_bool_mask


MASK_MAPPING = {
    # a1 old
    "piper": make_bool_mask(6, -1, 6, -1),  # split_aloha
    "arx_lift2": make_bool_mask(6, -1, 6, -1), 
    "split_aloha": make_bool_mask(6, -1, 6, -1), 
    "a2d": make_bool_mask(14, -2),  # agibotworld
    "genie1": make_bool_mask(14, -2), 
    "franka": make_bool_mask(7, -1), 
    "frankarobotiq": make_bool_mask(7, -1), 
    # a1 new
    "Franka": make_bool_mask(7, -1), 
    "ARX Lift-2": make_bool_mask(6, -1, 6, -1), 
    "AgileX Split Aloha": make_bool_mask(6, -1, 6, -1), 
    "Genie-1": make_bool_mask(14, -2), 
    "ARX AC One": make_bool_mask(6, -1, 6, -1), 
    # others
    "aloha": make_bool_mask(6, -1, 6, -1), 
    "panda": make_bool_mask(7, ), 
    "egodex_v": make_bool_mask(2),
    # RoboChallenge
    "ALOHA": make_bool_mask(6, -1, 6, -1),
    "ALOHA_STARVLA": make_bool_mask(6, -1, 6, -1),
    "UR5": make_bool_mask(6, -1), 
    "ARX5": make_bool_mask(6, -1),
    "FRANKA": make_bool_mask(7, -2),
    "DOS-W1": make_bool_mask(6, -1, 6, -1),
    "fourier_gr1_arms_waist": make_bool_mask(29),
    "GR1": make_bool_mask(29),
    "gr1": make_bool_mask(29),
    # custom real robot
    "real_lift2": make_bool_mask(6, -1, 6, -1),
    "real_piper": make_bool_mask(6, -1),
    "robotwin_arx5": make_bool_mask(6, -1, 6, -1),
    "ARX-X5": make_bool_mask(6, -1, 6, -1),
    "RoboTwin-ARX5": make_bool_mask(6, -1, 6, -1),
    # RoboCasa
    "PandaOmron": make_bool_mask(7),
    "cobot_magic_max":make_bool_mask(6,-1,6,-1),
    "wr_franka":make_bool_mask(7,-1)
}

LIBERO_FRANKA_MASK = make_bool_mask(7, -1)
MASK_MAPPING["libero_franka"] = LIBERO_FRANKA_MASK


FEATURE_MAPPING = dict(
    a2d={
        OBS_STATE: [
            "observation.states.joint.position", 
            "observation.states.effector.position", 
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.effector.position", 
        ], 
    }, 
    genie1={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.right_joint.position", 
            "states.left_gripper.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.right_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    arx_lift2={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    piper={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    r1lite={
        OBS_STATE: [
            'observation.state.left_arm', 
            'observation.state.right_arm', 
            'observation.state.left_gripper', 
            'observation.state.right_gripper',
        ], 
        ACTION: [
            "action.left_arm", 
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ], 
    },
    aloha={
        OBS_STATE: [
            'observation.state',
        ], 
        ACTION: [
            'action',
        ], 
    },
    franka={
        OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.gripper.position", 
        ], 
    }, 
    panda={
        OBS_STATE: [
            "observation.state", 
        ], 
        ACTION: [
            "action", 
        ], 
    },
    ALOHA={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    UR5={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    ARX5={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
    FRANKA={
        OBS_STATE: [
            "observation.state",
        ],
        ACTION: [
            "action",
        ],
    },
)
# a1 new
FEATURE_MAPPING["Franka"] = {
    OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
    ], 
    ACTION: [
        "actions.joint.position", 
        "actions.gripper.position", 
    ], 
}
FEATURE_MAPPING["ARX Lift-2"] = {
    OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["Genie-1"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.right_joint.position", 
        "states.left_gripper.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.right_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["AgileX Split Aloha"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["ARX AC One"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["egodex_v"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["libero_franka"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["real_lift2"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["real_piper"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["robotwin_arx5"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["ARX-X5"] = FEATURE_MAPPING["robotwin_arx5"]
FEATURE_MAPPING["RoboTwin-ARX5"] = FEATURE_MAPPING["robotwin_arx5"]
FEATURE_MAPPING["DOS-W1"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["ALOHA_STARVLA"] = FEATURE_MAPPING["ALOHA"]
FOURIER_GR1_ARMS_WAIST_FEATURE_MAPPING = {
    OBS_STATE: [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ],
    ACTION: [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ],
}
FEATURE_MAPPING["fourier_gr1_arms_waist"] = FOURIER_GR1_ARMS_WAIST_FEATURE_MAPPING
FEATURE_MAPPING["GR1"] = FOURIER_GR1_ARMS_WAIST_FEATURE_MAPPING
FEATURE_MAPPING["gr1"] = FOURIER_GR1_ARMS_WAIST_FEATURE_MAPPING
FEATURE_MAPPING["cobot_magic_max"]={
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
FEATURE_MAPPING["wr_franka"]={
    OBS_STATE: [
        "observation.state.endpose",
    ],
    ACTION: [
        "action.endpose",
    ],
}
IMAGE_MAPPING = dict(
    arx_lift2={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    piper={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    },
    genie1={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    a2d={
        "observation.images.head": f"{OBS_IMAGES}.image0", 
        "observation.images.hand_left": f"{OBS_IMAGES}.image1", 
        "observation.images.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    # todo, make sure what the key names are for franka
    franka={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand": f"{OBS_IMAGES}.image1", 
    }, 
    r1lite={
        "observation.images.head_rgb": f"{OBS_IMAGES}.image0", 
        "observation.images.left_wrist_rgb": f"{OBS_IMAGES}.image1", 
        "observation.images.right_wrist_rgb": f"{OBS_IMAGES}.image2", 
    },

    aloha={
        "observation.images.cam_high": f"{OBS_IMAGES}.image0", 
        "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1", 
        "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2", 
    },
    panda={
        "observation.images.image": f"{OBS_IMAGES}.image0", 
        "observation.images.image2": f"{OBS_IMAGES}.image1", 
    }
)
# a1 new
IMAGE_MAPPING["Franka"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand": f"{OBS_IMAGES}.image1", 
}
IMAGE_MAPPING["ARX Lift-2"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["Genie-1"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["AgileX Split Aloha"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["ARX AC One"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["ALOHA"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["ALOHA_STARVLA"] = {
    "observation.images.cam_high": f"{OBS_IMAGES}.image0",
    "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1",
    "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["UR5"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["ARX5"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
ARX5_GLOBAL_ARM_SIDE_IMAGE_MAPPING = {
    "observation.images.cam_global": f"{OBS_IMAGES}.image0",
    "observation.images.cam_arm": f"{OBS_IMAGES}.image1",
    "observation.images.cam_side": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["DOS-W1"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["FRANKA"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["egodex_v"] = {
    "observation.image": f"{OBS_IMAGES}.image0",
}
IMAGE_MAPPING["libero_franka"] = {
    "observation.images.image": f"{OBS_IMAGES}.image0",
    "observation.images.wrist_image": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["real_lift2"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["real_piper"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["robotwin_arx5"] = {
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["ARX-X5"] = IMAGE_MAPPING["robotwin_arx5"]
IMAGE_MAPPING["RoboTwin-ARX5"] = IMAGE_MAPPING["robotwin_arx5"]
FOURIER_GR1_ARMS_WAIST_IMAGE_MAPPING = {
    "video.ego_view": f"{OBS_IMAGES}.image0",
}
IMAGE_MAPPING["fourier_gr1_arms_waist"] = FOURIER_GR1_ARMS_WAIST_IMAGE_MAPPING
IMAGE_MAPPING["GR1"] = FOURIER_GR1_ARMS_WAIST_IMAGE_MAPPING
IMAGE_MAPPING["gr1"] = FOURIER_GR1_ARMS_WAIST_IMAGE_MAPPING
FEATURE_MAPPING["PandaOmron"] = {
    OBS_STATE: [
        "observation.state",
    ],
    ACTION: [
        "action",
    ],
}
IMAGE_MAPPING["PandaOmron"] = {
    "observation.images.cam_high": f"{OBS_IMAGES}.image0",
    "observation.images.cam_left": f"{OBS_IMAGES}.image1",
    "observation.images.cam_right": f"{OBS_IMAGES}.image2",
}

IMAGE_MAPPING["cobot_magic_max"]={
    "observation.images.head": f"{OBS_IMAGES}.image0",
    "observation.images.left": f"{OBS_IMAGES}.image1",
    "observation.images.right": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["wr_franka"]={
    "observation.images.front": f"{OBS_IMAGES}.image0",
    "observation.images.wrist": f"{OBS_IMAGES}.image1",
}


def _feature_key_set(feature_keys):
    if feature_keys is None:
        return None
    if isinstance(feature_keys, Mapping):
        return set(feature_keys.keys())
    return set(feature_keys)


def _feature_dim(feature_keys, key):
    if not isinstance(feature_keys, Mapping):
        return None
    feature = feature_keys.get(key)
    if not isinstance(feature, Mapping):
        return None
    shape = feature.get("shape")
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    return None


def infer_embodiment_variant(robot_type, feature_keys=None):
    resolved_robot_type = robot_type
    keys = _feature_key_set(feature_keys)

    if keys is not None:
        robocasa_gr1_keys = {
            "video.ego_view",
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
            "state.waist",
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
            "action.waist",
        }
        if robocasa_gr1_keys.issubset(keys):
            resolved_robot_type = "fourier_gr1_arms_waist"

    # RoboTwin and RoboChallenge both commonly use robot_type="aloha", but
    # their camera keys differ. Resolve the RoboChallenge LeRobot-v3 schema to
    # its own mapping so the RoboTwin cam_high/cam_*_wrist mapping is left
    # untouched.
    if robot_type == "aloha" and keys is not None:
        robochallenge_aloha_keys = {
            "observation.state",
            "action",
            "observation.images.head",
            "observation.images.left",
            "observation.images.right",
        }
        if robochallenge_aloha_keys.issubset(keys):
            resolved_robot_type = "ALOHA"

    if robot_type == "ALOHA" and keys is not None:
        starvla_aloha_keys = {
            "observation.state",
            "action",
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        }
        if starvla_aloha_keys.issubset(keys):
            resolved_robot_type = "ALOHA_STARVLA"

    if robot_type == "ur5" and keys is not None:
        robochallenge_ur5_keys = {
            "observation.state",
            "action",
            "observation.images.head",
            "observation.images.left",
        }
        if robochallenge_ur5_keys.issubset(keys):
            resolved_robot_type = "UR5"

    if robot_type in {"robotwin_arx5", "ARX-X5", "RoboTwin-ARX5"}:
        resolved_robot_type = "robotwin_arx5"

    arx5_head_left_right_keys = {
        "observation.state",
        "action",
        "observation.images.head",
        "observation.images.left",
        "observation.images.right",
    }

    if robot_type == "arx5" and keys is not None:
        if arx5_head_left_right_keys.issubset(keys):
            action_dim = _feature_dim(feature_keys, "action")
            resolved_robot_type = "robotwin_arx5" if action_dim == 14 else "ARX5"

    if robot_type == "ARX5" and keys is not None:
        if arx5_head_left_right_keys.issubset(keys):
            action_dim = _feature_dim(feature_keys, "action")
            if action_dim == 14:
                resolved_robot_type = "robotwin_arx5"

    if robot_type in {"dos_w1", "dos-w1", "DOS-W1"} and keys is not None:
        robochallenge_w1_keys = {
            "observation.state",
            "action",
            "observation.images.head",
            "observation.images.left",
            "observation.images.right",
        }
        if robochallenge_w1_keys.issubset(keys):
            resolved_robot_type = "DOS-W1"

    # LIBERO datasets are tagged as "franka" but use a different flattened
    # feature schema than the other Franka datasets in this codebase.
    if robot_type == "franka" and keys is not None:
        libero_keys = {
            "observation.state",
            "action",
            "observation.images.image",
            "observation.images.wrist_image",
        }
        if libero_keys.issubset(keys):
            resolved_robot_type = "libero_franka"

    return resolved_robot_type


def _get_required_mapping(mapping_name, mapping, robot_type, feature_keys=None):
    resolved_robot_type = infer_embodiment_variant(robot_type, feature_keys)
    if resolved_robot_type in mapping:
        return mapping[resolved_robot_type]

    available = ", ".join(sorted(str(key) for key in mapping.keys()))
    keys = _feature_key_set(feature_keys)
    feature_hint = ""
    if keys is not None:
        feature_hint = f" feature_keys={sorted(keys)}"
    raise KeyError(
        f"Missing {mapping_name} for robot_type={robot_type!r} "
        f"(resolved={resolved_robot_type!r}). Add an explicit mapping in "
        f"src/lerobot/transforms/constants.py instead of relying on a default."
        f"{feature_hint} Available robot_types: {available}"
    )


def get_mask_mapping(robot_type, feature_keys=None):
    return _get_required_mapping("MASK_MAPPING", MASK_MAPPING, robot_type, feature_keys)


def get_feature_mapping(robot_type, feature_keys=None):
    return _get_required_mapping("FEATURE_MAPPING", FEATURE_MAPPING, robot_type, feature_keys)


def get_image_mapping(robot_type, feature_keys=None):
    keys = _feature_key_set(feature_keys)
    if robot_type == "ARX5" and keys is not None:
        arx5_special_keys = set(ARX5_GLOBAL_ARM_SIDE_IMAGE_MAPPING)
        if arx5_special_keys.issubset(keys):
            return ARX5_GLOBAL_ARM_SIDE_IMAGE_MAPPING
        if arx5_special_keys & keys:
            missing = ", ".join(sorted(arx5_special_keys - keys))
            raise KeyError(
                "Partial ARX5 global/arm/side camera schema. "
                f"Missing keys: {missing}"
            )
    return _get_required_mapping("IMAGE_MAPPING", IMAGE_MAPPING, robot_type, feature_keys)
