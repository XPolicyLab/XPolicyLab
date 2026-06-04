# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

g2_kinematics_config = dict(
    urdf="./urdf/G2_omnipicker.urdf",
    arm_joint_id=[list(range(5, 12)), list(range(20, 27))],
    arm_link_keys=[
        [
            "arm_l_link1",
            "arm_l_link2",
            "arm_l_link3",
            "arm_l_link4",
            "arm_l_link5",
            "arm_l_link6",
            "arm_l_end_link",
        ],
        [
            "arm_r_link1",
            "arm_r_link2",
            "arm_r_link3",
            "arm_r_link4",
            "arm_r_link5",
            "arm_r_link6",
            "arm_r_end_link",
        ],
    ],
    finger_keys=[
        ["gripper_l_center_link"],
        ["gripper_r_center_link"],
    ],
    head_joint_id=list(range(35, 38)),
    head_link_keys=[
        "head_link1",
        "head_link2",
        "head_link3",
    ],
    body_joint_id=list(range(5)),
    body_link_keys=[
        "body_link1",
        "body_link2",
        "body_link3",
        "body_link4",
        "body_link5",
    ],
)


g2_deploy_camera_kinematics_config = {
    **g2_kinematics_config,
    "finger_keys": [
        ["gripper_l_base_link"],
        ["gripper_r_base_link"],
    ],
}

cam_names = ["hand_left", "hand_right", "top_head"]

dataset_name = "agibot_geniesim3_challenge"
data_paths = [
    "./data/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim",
]
# Prediction stride is measured in packed kept samples after frame filtering.
pred_interval = 1


def expand_ro_data_paths(patterns: list[str]) -> list[str]:
    from glob import glob
    from pathlib import Path

    paths: list[str] = []
    for pattern in patterns:
        for matched in glob(pattern):
            p = Path(matched)
            if (p / "state.json").exists():
                paths.append(str(p))
            else:
                for state in sorted(p.rglob("state.json")):
                    if state.is_file():
                        paths.append(str(state.parent))
    return sorted(set(paths))


def build_transforms(config, mode, kinematics_config=None):
    import numpy as np

    from robo_orchard_lab.dataset.agibot_geniesim.transforms import (
        GenieSim3CalibrationToExtrinsic,
        GenieSim3Kinematics,
        ZeroRobotState,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    t_base2world = np.eye(4).tolist()  # noqa: N806
    joint_mask = (
        ([True] * 7 + [False]) * 2
        + [True] * 3
        + [False, False, True, False, True]
    )

    base_joint_weights = [1] + [0] * 3 + [0] * 4
    gripper_weights = [1] + [1] * 3 + [0.1] * 4
    head_weights = [0] + [0] * 7
    body_weights = [1] + [0] * 7
    loss_weights = np.array(
        [
            [base_joint_weights] * 7
            + [gripper_weights]
            + [base_joint_weights] * 7
            + [gripper_weights]
            + [head_weights] * 3
            + [body_weights] * 5
        ]
    )
    state_loss_weights = (loss_weights * 0.2).tolist()
    fk_loss_weight = (loss_weights * 1.8).tolist()
    joint_scale_shift = [
        [3.057942390, 0.007310510],
        [2.093060077, -0.001349867],
        [3.091859937, 0.007050037],
        [1.762709975, -0.724399924],
        [3.156573414, 0.025336503],
        [1.303260088, -0.254849970],
        [1.570760011, 0.000049948],
        [-0.500000000, 0.500000000],
        [3.057942390, 0.007310510],
        [2.093060077, -0.001349867],
        [3.091859937, 0.007050037],
        [1.762709975, -0.724399924],
        [3.156573414, 0.025336503],
        [1.303260088, -0.254849970],
        [1.570760011, 0.000049948],
        [-0.500000000, 0.500000000],
        [0.138159990, -0.013749998],
        [0.045109998, -0.001300000],
        [0.115823776, 0.113486230],
        [0.128683850, -0.931335866],
        [0.249554917, 1.352255106],
        [0.380988687, -0.279345959],
        [0.023029560, 0.017360471],
        [1.702966928, 0.270821095],
    ]

    add_data_relative_items = dict(
        type=AddItems,
        T_base2world=t_base2world,
        joint_mask=joint_mask,
        joint_scale_shift=joint_scale_shift,
    )
    if mode == "training":
        add_data_relative_items.update(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=True,
        use_master_joint=False,
        limitation=1000,
        gripper_indices=[7, 15],
    )
    dst_wh = config.get("dst_wh", (308, 252))
    patch_size = config.get("patch_size", 1)
    dst_wh = tuple(x // patch_size * patch_size for x in dst_wh)
    resize = dict(type=Resize, dst_wh=dst_wh)
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
            joint_scale_shift="float32",
        ),
    )
    kinematics = dict(type=GenieSim3Kinematics, **kinematics_config)
    zero_robot_state = dict(
        type=ZeroRobotState,
        keys=config.get("zero_robot_state_keys", ["hist_robot_state"]),
        joint_indices=[7, 15],
        state_indices=[0],
    )

    if mode == "training":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "pred_mask",
                "joint_mask",
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=[
                [-0.02, 0.02] if mask else [0.0, 0.0] for mask in joint_mask
            ],
            add_to_pred=True,
        )
        return [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            joint_state_noise,
            convert_dtype,
            kinematics,
            zero_robot_state,
            item_selection,
        ]

    if mode == "validation":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        return [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            kinematics,
            zero_robot_state,
            item_selection,
        ]

    if mode == "deploy":
        calib_to_ext = dict(
            type=GenieSim3CalibrationToExtrinsic,
            calibration=None,
            cam_ee_joint_indices=dict(
                hand_left=7,
                hand_right=15,
                top_head=18,
            ),
            cam_names=cam_names,
            **g2_deploy_camera_kinematics_config,
        )
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "joint_scale_shift",
                "kinematics",
                "text",
                "remaining_actions",
                "delay_horizon",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        return [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            kinematics,
            zero_robot_state,
            item_selection,
            unsqueeze_batch,
        ]

    raise ValueError(f"Unsupported mode: {mode}")


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, **kwargs):
    from robo_orchard_lab.dataset.agibot_geniesim.agibot_geniesim3_ro_dataset import (  # noqa: E501
        AgibotGenieSim3RODataset,
    )
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    if dataset_name not in dataset_names:
        return []

    transforms = build_transforms(
        config,
        mode,
        kinematics_config=g2_kinematics_config,
    )
    dataset = AgibotGenieSim3RODataset(
        paths=expand_ro_data_paths(data_paths),
        cam_names=cam_names,
        target_columns=["joints", "actions"],
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        pred_interval=pred_interval,
        transforms=[build(x) for x in as_sequence(transforms)],
        gripper_indices=[7, 15],
        gripper_divisor=120.0,
    )

    return [dataset]


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    if dataset_name not in dataset_names:
        return {}

    transforms = build_transforms(
        config,
        mode="deploy",
        kinematics_config=g2_kinematics_config,
    )
    processor = HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            cam_names=cam_names,
            transforms=transforms,
        )
    )
    return {dataset_name: processor}
